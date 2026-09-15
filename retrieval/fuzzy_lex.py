from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from retrieval.query_lang import And, Exact, Near, QueryNode


DEFAULT_STOPWORDS = {
    # ultra-small set; we want auditability and to avoid fuzzing glue words
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "between",
    "explain",
    "relationship",
}


@dataclass(frozen=True)
class FuzzyLexConfig:
    enabled: bool = True
    min_similarity: float = 0.4
    top_k_per_token: int = 5
    max_total_variants: int = 50
    fuzz_oov_only: bool = False
    min_token_len: int = 3
    dictionary_build_id: Optional[int] = None  # if None, use latest for slice
    stopwords: Optional[Iterable[str]] = None
    # OCR-variant channel: skeleton-keyed lookup of plausible OCR corruptions
    # (see retrieval/ocr_variants.py). Rarity is the point — garbled forms are
    # rare — so that channel ranks chunk_freq ASC, unlike the trigram channel.
    ocr_enabled: bool = True
    ocr_max_cost: float = 1.5
    ocr_top_k: int = 4
    ocr_freq_ceiling: int = 500
    # Trigram expansion is skipped for tokens shorter than this. Below ~5
    # chars the similarity threshold admits mostly truncations and common
    # real-word neighbours (ranked chunk_freq DESC, so they win), while a
    # genuine one-letter OCR error can't reach the threshold at all — that
    # error class belongs to the ocr channel, which has its own len>=4 gate.
    trgm_min_token_len: int = 5

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        # stopwords not logged in full by default (too verbose); keep whether custom
        d["stopwords_custom"] = self.stopwords is not None
        d.pop("stopwords", None)
        return d


def tokenize_query(text: str, *, min_len: int = 3, stopwords: Optional[Iterable[str]] = None) -> List[str]:
    q = (text or "").lower()
    toks = re.findall(r"[a-z0-9_']+", q)
    toks = [t for t in toks if (len(t) >= min_len or t.isdigit())]
    toks = [t for t in toks if t]
    sw = set(stopwords) if stopwords is not None else DEFAULT_STOPWORDS
    toks = [t for t in toks if t not in sw]
    # preserve deterministic order while deduping
    seen = set()
    out = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def get_latest_dictionary_build_id(
    conn,
    *,
    chunk_pv: str,
    collection_slug: Optional[str],
    norm_version: str,
) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM corpus_dictionary_builds
            WHERE chunk_pv = %s
              AND collection_slug IS NOT DISTINCT FROM %s
              AND norm_version = %s
            ORDER BY built_at DESC, id DESC
            LIMIT 1;
            """,
            (chunk_pv, collection_slug, norm_version),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def fetch_fuzzy_variants_for_tokens(
    conn,
    *,
    tokens: List[str],
    chunk_pv: str,
    collection_slug: Optional[str],
    norm_version: str,
    config: FuzzyLexConfig,
) -> Tuple[Optional[int], Dict[str, List[Dict[str, Any]]]]:
    """
    Returns (build_id, expansions) where expansions maps token -> list of
    {lexeme, similarity, chunk_freq}.
    """
    if not config.enabled or not tokens:
        return (config.dictionary_build_id, {})

    build_id = config.dictionary_build_id
    if build_id is None:
        build_id = get_latest_dictionary_build_id(
            conn, chunk_pv=chunk_pv, collection_slug=collection_slug, norm_version=norm_version
        )
    if build_id is None:
        return (None, {})

    expansions: Dict[str, List[Dict[str, Any]]] = {}
    total_variants = 0

    with conn.cursor() as cur:
        for tok in tokens:
            if len(tok) < config.trgm_min_token_len:
                continue  # short tokens: ocr channel only (see config comment)
            # Fast pre-filter using similarity() against lexeme; rank primarily by corpus frequency, then similarity.
            cur.execute(
                """
                SELECT
                  l.lexeme,
                  word_similarity(%s, l.lexeme) AS ws,
                  l.chunk_freq
                FROM corpus_dictionary_lexemes l
                WHERE l.build_id = %s
                  AND similarity(%s, l.lexeme) >= %s
                ORDER BY l.chunk_freq DESC, ws DESC, l.lexeme ASC
                LIMIT %s;
                """,
                (tok, build_id, tok, float(config.min_similarity), int(config.top_k_per_token)),
            )
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for lexeme, ws, chunk_freq in rows:
                if ws is None:
                    continue
                ws_f = float(ws)
                if ws_f < float(config.min_similarity):
                    continue
                out.append({"lexeme": str(lexeme), "similarity": ws_f, "chunk_freq": int(chunk_freq or 0)})

            if out:
                expansions[tok] = out
                total_variants += len(out)
                if total_variants >= config.max_total_variants:
                    break

    # OCR-variant channel: expand query tokens to corpus tokens that are
    # plausible OCR corruptions (e.g. FUHR -> FUER), which trigram similarity
    # misses for short names. Lazy import: the module (and its config artifact)
    # may be absent; retrieval must never crash because of it.
    if config.ocr_enabled and build_id is not None and tokens:
        try:
            _ocr = importlib.import_module("retrieval.ocr_variants")
        except ImportError:
            _ocr = None
        if _ocr is not None:
            conf = _ocr.load_confusion()
            ocr_expansions = _ocr.fetch_ocr_variants(
                conn,
                tokens,
                build_id,
                conf,
                max_cost=config.ocr_max_cost,
                top_k=config.ocr_top_k,
                freq_ceiling=config.ocr_freq_ceiling,
            )
            _merge_ocr_expansions(
                tokens=tokens,
                expansions=expansions,
                ocr_expansions=ocr_expansions or {},
                max_total_variants=config.max_total_variants,
            )

    return (build_id, expansions)


def _merge_ocr_expansions(
    *,
    tokens: List[str],
    expansions: Dict[str, List[Dict[str, Any]]],
    ocr_expansions: Dict[str, List[Dict[str, Any]]],
    max_total_variants: int,
) -> None:
    """
    Merge OCR-variant entries into the trigram expansions, in place.

    Per token: trigram entries keep their existing order and gain
    "source":"trgm" — upgraded to "both" when the ocr channel found the same
    lexeme — then ocr-only entries ("source":"ocr", carrying "cost") follow,
    ordered by (cost asc, chunk_freq asc). max_total_variants is honored
    across the merged set: trigram entries first, then ocr entries cheapest
    (lowest cost) first across all tokens.
    """
    total_variants = 0
    seen_by_token: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tok, entries in expansions.items():
        by_lexeme: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            entry["source"] = "trgm"
            by_lexeme[str(entry.get("lexeme"))] = entry
        seen_by_token[tok] = by_lexeme
        total_variants += len(entries)

    # Collect ocr-only candidates globally so the cap keeps the cheapest first.
    candidates: List[Tuple[float, int, int, str, Dict[str, Any]]] = []
    for tok_idx, tok in enumerate(tokens):
        by_lexeme = seen_by_token.setdefault(tok, {})
        for ocr_entry in ocr_expansions.get(tok) or []:
            lex = ocr_entry.get("lexeme")
            if not lex:
                continue
            lex = str(lex)
            existing = by_lexeme.get(lex)
            if existing is not None:
                # Found by both channels: keep the trigram dict, relabel.
                if existing.get("source") == "trgm":
                    existing["source"] = "both"
                continue
            merged = dict(ocr_entry)
            merged["source"] = "ocr"
            by_lexeme[lex] = merged
            cost = float(merged.get("cost", 0.0))
            freq = int(merged.get("chunk_freq") or 0)
            candidates.append((cost, freq, tok_idx, tok, merged))

    # Global rank (cost asc, chunk_freq asc): rarity is the point for the ocr
    # channel — garbled forms are rare — so low chunk_freq wins, not high.
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    for cost, freq, tok_idx, tok, merged in candidates:
        if total_variants >= max_total_variants:
            break
        expansions.setdefault(tok, []).append(merged)
        total_variants += 1


def _tsquery_escape_token(t: str) -> str:
    """
    Conservative escaping for simple tsquery lexemes.
    We only allow [a-z0-9_'] tokens as produced by our tokenizer.
    """
    return t


def compile_slots_tsquery(tokens: List[str], expansions: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Compile tokens into a tsquery string where each token becomes:
      token | variant1 | variant2 ...
    (No AND/OR at top-level; caller chooses.)
    """
    slots: List[str] = []
    for tok in tokens:
        alts = [_tsquery_escape_token(tok)]
        for v in expansions.get(tok, []):
            lex = v.get("lexeme")
            if lex and lex != tok:
                alts.append(_tsquery_escape_token(str(lex)))
        # Dedup within slot, keep deterministic order
        seen = set()
        alts2 = []
        for a in alts:
            if a in seen:
                continue
            seen.add(a)
            alts2.append(a)
        slots.append(" | ".join(alts2))
    return slots and ("(" + ") & (".join(slots) + ")") or "___nomatch___"


def compile_querynode_tsquery(
    node: QueryNode,
    *,
    conn,
    chunk_pv: str,
    collection_slug: Optional[str],
    norm_version: str,
    config: FuzzyLexConfig,
) -> Tuple[str, Optional[int], Dict[str, List[Dict[str, Any]]]]:
    """
    Compile an explicit query language node (Exact/And/Near) into a tsquery string,
    applying per-token fuzzy expansion.
    Returns: (tsquery_text, build_id, expansions)
    """
    stopwords = config.stopwords if config.stopwords is not None else DEFAULT_STOPWORDS

    if isinstance(node, Exact):
        tokens = tokenize_query(node.term, min_len=config.min_token_len, stopwords=stopwords)
        build_id, expansions = fetch_fuzzy_variants_for_tokens(
            conn,
            tokens=tokens,
            chunk_pv=chunk_pv,
            collection_slug=collection_slug,
            norm_version=norm_version,
            config=config,
        )
        # Exact primitive: require the token slot(s). If multiple tokens, AND them.
        tsq = compile_slots_tsquery(tokens, expansions)
        return tsq, build_id, expansions

    if isinstance(node, And):
        # And primitive treats each provided term as its own slot group; tokenize each term.
        all_tokens: List[str] = []
        for term in node.terms:
            all_tokens.extend(tokenize_query(term, min_len=config.min_token_len, stopwords=stopwords))
        # Dedup while preserving order
        seen = set()
        tokens = []
        for t in all_tokens:
            if t in seen:
                continue
            seen.add(t)
            tokens.append(t)
        build_id, expansions = fetch_fuzzy_variants_for_tokens(
            conn,
            tokens=tokens,
            chunk_pv=chunk_pv,
            collection_slug=collection_slug,
            norm_version=norm_version,
            config=config,
        )
        tsq = compile_slots_tsquery(tokens, expansions)
        return tsq, build_id, expansions

    if isinstance(node, Near):
        # NEAR primitive: keep variant caps small by reusing config but caller should set top_k_per_token low.
        left_tokens = tokenize_query(node.a, min_len=config.min_token_len, stopwords=stopwords)
        right_tokens = tokenize_query(node.b, min_len=config.min_token_len, stopwords=stopwords)
        tokens = []
        for t in left_tokens + right_tokens:
            if t not in tokens:
                tokens.append(t)
        build_id, expansions = fetch_fuzzy_variants_for_tokens(
            conn,
            tokens=tokens,
            chunk_pv=chunk_pv,
            collection_slug=collection_slug,
            norm_version=norm_version,
            config=config,
        )
        # Build a near tsquery: (left_slot) <N> (right_slot)
        # If multiple tokens on either side, AND them within side before NEAR.
        def _side_ts(side_tokens: List[str]) -> str:
            if not side_tokens:
                return "___nomatch___"
            parts = []
            for tok in side_tokens:
                alts = [_tsquery_escape_token(tok)]
                for v in expansions.get(tok, []):
                    lex = v.get("lexeme")
                    if lex and lex != tok:
                        alts.append(_tsquery_escape_token(str(lex)))
                # dedup
                seen2 = set()
                alts2 = []
                for a in alts:
                    if a in seen2:
                        continue
                    seen2.add(a)
                    alts2.append(a)
                parts.append("(" + " | ".join(alts2) + ")")
            return " & ".join(parts)

        left_tsq = _side_ts(left_tokens)
        right_tsq = _side_ts(right_tokens)
        # distance operator: <-> is adjacent; <N> is within N? In tsquery: <-> and <N>
        # Use <N> where N is window_words.
        tsq = f"{left_tsq} <{int(node.window_words)}> {right_tsq}"
        return tsq, build_id, expansions

    return "___nomatch___", None, {}


def compile_hybrid_or_tsquery(
    query_text: str,
    *,
    conn,
    chunk_pv: str,
    collection_slug: Optional[str],
    norm_version: str,
    config: FuzzyLexConfig,
) -> Tuple[str, Optional[int], Dict[str, List[Dict[str, Any]]], List[str]]:
    """
    Hybrid lexical wants OR-ish behavior to avoid going empty for natural language.
    We tokenize query_text, expand per token, and return a tsquery that ORs token-slots.
    """
    stopwords = config.stopwords if config.stopwords is not None else DEFAULT_STOPWORDS
    tokens = tokenize_query(query_text, min_len=config.min_token_len, stopwords=stopwords)
    build_id, expansions = fetch_fuzzy_variants_for_tokens(
        conn,
        tokens=tokens,
        chunk_pv=chunk_pv,
        collection_slug=collection_slug,
        norm_version=norm_version,
        config=config,
    )

    slots: List[str] = []
    for tok in tokens:
        alts = [_tsquery_escape_token(tok)]
        for v in expansions.get(tok, []):
            lex = v.get("lexeme")
            if lex and lex != tok:
                alts.append(_tsquery_escape_token(str(lex)))
        # dedup
        seen = set()
        alts2 = []
        for a in alts:
            if a in seen:
                continue
            seen.add(a)
            alts2.append(a)
        slots.append("(" + " | ".join(alts2) + ")")

    tsq = " | ".join(slots) if slots else "___nomatch___"
    return tsq, build_id, expansions, tokens

