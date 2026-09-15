#!/usr/bin/env python3
"""
Sweep a collection for plausible OCR-variant pairs: RARE corpus tokens that
share an OCR-confusion skeleton with a NAME-like token (garbled forms are
rare — e.g. typed FUHR OCR-read as FUER), ranked by confusion-weighted edit
distance. Read-only: issues SELECTs only, writes nothing to the database.

Modes:
  --build-id N    Read (lexeme, chunk_freq, skeleton) from
                  corpus_dictionary_lexemes for that dictionary build
                  (skeletons recomputed python-side where NULL).
  --from-chunks   No dictionary build required: aggregate word/ndoc via
                  ts_stat over to_tsvector('simple', COALESCE(clean_text,
                  text)) for the collection's chunks, UNIONed with the same
                  over raw c.text restricted to transcript-bearing documents
                  (documents.metadata ? 'transcript') so raw-OCR garble stays
                  a variant target; freqs summed. Skeletons computed
                  python-side via retrieval.ocr_variants.

Usage (PowerShell):
  $env:DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
  python scripts/sweep_ocr_variants.py --collection-slug silvermaster \
      --chunk-pv chunk_v1_silvermaster_structured_4k --from-chunks \
      --gazetteer names.txt --out sweep_silvermaster.csv

CSV columns: rare_token,candidate_name,cost,rare_freq,candidate_freq
sorted by (cost asc, candidate_freq desc). Pool sizes and row count are
printed to stderr; for --collection-slug silvermaster an explicit
acceptance line for fuer->fuhr is printed as well.
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

# Ensure repo root importable when running as script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2

CSV_HEADER = ["rare_token", "candidate_name", "cost", "rare_freq", "candidate_freq"]

# Small english stopword list: keeps common low-frequency-in-a-slice glue words
# out of the rare pool. Rare-pool candidates are len>=4 already, but common
# words can still be rare within a small collection slice.
STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "were", "which",
    "their", "there", "they", "them", "then", "than", "these", "those",
    "been", "being", "would", "could", "should", "about", "into", "upon",
    "over", "under", "only", "other", "some", "same", "such", "also",
    "when", "where", "while", "after", "before", "during", "between",
    "because", "through", "will", "shall", "said", "does", "each",
    "more", "most", "much", "many", "very", "what", "your",
}

# Mirrors _TRANSCRIPT_DOC_EXISTS in retrieval/ops.py (chunk_pages -> pages -> documents).
TRANSCRIPT_DOC_EXISTS = """EXISTS (
        SELECT 1
        FROM chunk_pages cp_tr
        JOIN pages p_tr ON p_tr.id = cp_tr.page_id
        JOIN documents d_tr ON d_tr.id = p_tr.document_id
        WHERE cp_tr.chunk_id = c.id
          AND d_tr.metadata ? 'transcript'
      )"""

# Inner queries handed to ts_stat as string literals. ts_stat() takes its
# query as text, so the two user-provided values (chunk_pv, collection_slug)
# are quoted with cursor.mogrify() — never raw f-string interpolation.
INNER_CLEAN_SQL = """SELECT to_tsvector('simple', COALESCE(c.clean_text, c.text)) AS tsv
FROM chunks c
JOIN chunk_metadata cm ON cm.chunk_id = c.id
WHERE c.pipeline_version = %s
  AND cm.pipeline_version = %s
  AND cm.collection_slug = %s"""

INNER_RAW_TRANSCRIPT_SQL = f"""SELECT to_tsvector('simple', c.text) AS tsv
FROM chunks c
JOIN chunk_metadata cm ON cm.chunk_id = c.id
WHERE c.pipeline_version = %s
  AND cm.pipeline_version = %s
  AND cm.collection_slug = %s
  AND {TRANSCRIPT_DOC_EXISTS}"""


class PairRow(NamedTuple):
    rare_token: str
    candidate_name: str
    cost: float
    rare_freq: int
    candidate_freq: int


@dataclass(frozen=True)
class PairingResult:
    rows: List[PairRow]
    name_pool: int
    rare_pool: int


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def load_gazetteer(path: str) -> Set[str]:
    """
    Load gazetteer name tokens: one name per line, '#' starts a comment.
    Lines are tokenized to lowercase word tokens so multi-word names
    ("Nathan Gregory Silvermaster") contribute each token.
    """
    entries: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for tok in re.findall(r"[a-z0-9']+", line.lower()):
                if len(tok) >= 3:
                    entries.add(tok)
    return entries


def _skeleton_column_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'corpus_dictionary_lexemes' AND column_name = 'skeleton';
            """
        )
        return cur.fetchone() is not None


def fetch_lexemes_from_build(
    conn, build_id: int
) -> Tuple[List[Tuple[str, int]], Dict[str, str]]:
    """
    Read (lexeme, chunk_freq[, skeleton]) for one dictionary build.
    Returns (lexemes, skeletons) where skeletons only holds non-NULL values
    (missing ones are recomputed python-side by the caller).
    """
    has_skeleton = _skeleton_column_exists(conn)
    cols = "lexeme, chunk_freq" + (", skeleton" if has_skeleton else "")
    lexemes: List[Tuple[str, int]] = []
    skeletons: Dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {cols} FROM corpus_dictionary_lexemes WHERE build_id = %s;",
            (build_id,),
        )
        for row in cur.fetchall():
            lexeme = str(row[0])
            lexemes.append((lexeme, int(row[1] or 0)))
            if has_skeleton and row[2]:
                skeletons[lexeme] = str(row[2])
    return lexemes, skeletons


def fetch_lexemes_from_chunks(
    conn, *, collection_slug: str, chunk_pv: str
) -> List[Tuple[str, int]]:
    """
    Aggregate (word, chunk_freq) straight from chunk tsvectors — works against
    a database with zero dictionary builds. The clean pass reads
    COALESCE(clean_text, text); the raw pass reads c.text for chunks of
    transcript-bearing documents (their garble lives only in c.text once a
    corrected transcript replaces clean_text). Frequencies are summed.
    """
    with conn.cursor() as cur:
        params = (chunk_pv, chunk_pv, collection_slug)
        inner_clean = cur.mogrify(INNER_CLEAN_SQL, params).decode("utf-8")
        inner_raw = cur.mogrify(INNER_RAW_TRANSCRIPT_SQL, params).decode("utf-8")
        cur.execute(
            """
            SELECT word, SUM(ndoc)::int AS chunk_freq
            FROM (
              SELECT word, ndoc FROM ts_stat(%s)
              UNION ALL
              SELECT word, ndoc FROM ts_stat(%s)
            ) u
            GROUP BY word;
            """,
            (inner_clean, inner_raw),
        )
        return [(str(word), int(freq or 0)) for word, freq in cur.fetchall()]


def pair_rare_to_names(
    lexemes: Iterable[Tuple[str, int]],
    *,
    skeleton_fn: Callable[[str], str],
    distance_fn: Callable[[str, str], float],
    gazetteer: Optional[Iterable[str]] = None,
    min_name_freq: int = 8,
    max_rare_freq: int = 5,
    max_cost: float = 1.5,
    stopwords: Optional[Iterable[str]] = None,
) -> PairingResult:
    """
    Pure pairing over (lexeme, freq) pairs — no DB, no config file.

    Name pool: lexemes with freq >= min_name_freq, UNION gazetteer entries
    present as lexemes (whatever their freq). Rare pool: freq <= max_rare_freq,
    len >= 4, not pure digits, not stopwords. A rare token matches a name when
    their skeletons are equal, the tokens differ, and
    distance_fn(rare, name) <= max_cost. Rows sorted (cost asc,
    candidate_freq desc), ties broken lexically for determinism.
    """
    sw = STOPWORDS if stopwords is None else set(stopwords)
    gaz = {str(g).lower() for g in (gazetteer or ())}

    freqs: Dict[str, int] = {}
    for lexeme, freq in lexemes:
        tok = str(lexeme)
        freqs[tok] = freqs.get(tok, 0) + int(freq or 0)

    names = [(tok, f) for tok, f in freqs.items() if f >= min_name_freq or tok in gaz]
    rares = [
        (tok, f)
        for tok, f in freqs.items()
        if f <= max_rare_freq and len(tok) >= 4 and not tok.isdigit() and tok not in sw
    ]

    names_by_skel: Dict[str, List[Tuple[str, int]]] = {}
    for tok, f in names:
        sk = skeleton_fn(tok)
        if not sk:
            continue
        names_by_skel.setdefault(sk, []).append((tok, f))

    rows: List[PairRow] = []
    for rare_tok, rare_freq in rares:
        sk = skeleton_fn(rare_tok)
        if not sk:
            continue
        for name_tok, name_freq in names_by_skel.get(sk, []):
            if name_tok == rare_tok:
                continue
            cost = float(distance_fn(rare_tok, name_tok))
            if cost <= max_cost:
                rows.append(PairRow(rare_tok, name_tok, cost, rare_freq, name_freq))

    rows.sort(key=lambda r: (r.cost, -r.candidate_freq, r.rare_token, r.candidate_name))
    return PairingResult(rows=rows, name_pool=len(names), rare_pool=len(rares))


def write_csv(rows: List[PairRow], out_path: Optional[str]) -> None:
    fh = open(out_path, "w", newline="", encoding="utf-8") if out_path else sys.stdout
    try:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([r.rare_token, r.candidate_name, f"{r.cost:.3f}", r.rare_freq, r.candidate_freq])
    finally:
        if out_path:
            fh.close()


def main():
    ap = argparse.ArgumentParser(
        description="Sweep a collection for plausible OCR-variant (rare garble -> name) pairs (read-only)"
    )
    ap.add_argument("--collection-slug", required=True, help="chunk_metadata.collection_slug scope")
    ap.add_argument("--chunk-pv", required=True, help="chunks/chunk_metadata pipeline_version")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-id", type=int, default=None, help="Read lexemes from this corpus_dictionary build")
    mode.add_argument(
        "--from-chunks",
        action="store_true",
        help="Aggregate lexemes straight from chunk tsvectors (no dictionary build needed)",
    )
    ap.add_argument("--gazetteer", default=None, help="Optional file of names (one per line) to force into the name pool")
    ap.add_argument("--min-name-freq", type=int, default=8, help="Min chunk_freq for name-pool membership")
    ap.add_argument("--max-rare-freq", type=int, default=5, help="Max chunk_freq for rare-pool membership")
    ap.add_argument("--max-cost", type=float, default=1.5, help="Max confusion-weighted edit distance")
    ap.add_argument("--out", default=None, help="CSV output path (default: stdout)")
    args = ap.parse_args()

    # Lazy import: the sweep needs the real confusion machinery, but keep the
    # module importable (for the pure-pairing unit tests) without it.
    try:
        from retrieval.ocr_variants import load_confusion, skeleton, weighted_distance
    except ImportError as exc:
        sys.exit(f"retrieval.ocr_variants is required to run the sweep ({exc})")

    conf = load_confusion()
    gazetteer = load_gazetteer(args.gazetteer) if args.gazetteer else set()

    conn = get_conn()
    try:
        if args.from_chunks:
            lexemes = fetch_lexemes_from_chunks(
                conn, collection_slug=args.collection_slug, chunk_pv=args.chunk_pv
            )
            db_skeletons: Dict[str, str] = {}
        else:
            lexemes, db_skeletons = fetch_lexemes_from_build(conn, args.build_id)
    finally:
        conn.close()

    # Early-exit cap must clear max_cost so the <= filter stays exact.
    cap = max(2.0, args.max_cost + 0.5)

    def skeleton_fn(tok: str) -> str:
        sk = db_skeletons.get(tok)
        return sk if sk else skeleton(tok, conf)

    def distance_fn(a: str, b: str) -> float:
        return weighted_distance(a, b, conf, cap=cap)

    result = pair_rare_to_names(
        lexemes,
        skeleton_fn=skeleton_fn,
        distance_fn=distance_fn,
        gazetteer=gazetteer,
        min_name_freq=args.min_name_freq,
        max_rare_freq=args.max_rare_freq,
        max_cost=args.max_cost,
    )

    write_csv(result.rows, args.out)

    # Diagnostics to stderr so CSV-on-stdout stays clean.
    print(
        f"lexemes={len(lexemes)} name_pool={result.name_pool} "
        f"rare_pool={result.rare_pool} pairs={len(result.rows)}",
        file=sys.stderr,
    )
    if args.out:
        print(f"Wrote {len(result.rows)} rows to {args.out}", file=sys.stderr)
    if args.collection_slug == "silvermaster":
        hit = any(r.rare_token == "fuer" and r.candidate_name == "fuhr" for r in result.rows)
        print(f"ACCEPTANCE fuer->fuhr: {'PASS' if hit else 'FAIL'}", file=sys.stderr)


if __name__ == "__main__":
    main()
