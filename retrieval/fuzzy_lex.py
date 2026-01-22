from __future__ import annotations

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

    return (build_id, expansions)


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

