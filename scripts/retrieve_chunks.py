#!/usr/bin/env python3
import os
import argparse
import re
from typing import List, Tuple, Optional, Any, Dict

import psycopg2


# -----------------------
# DB + embedding helpers
# -----------------------

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)

def embed_query(text: str) -> List[float]:
    """Embed query using OpenAI embeddings. Assumes chunks.embedding is vector(1536)."""
    from openai import OpenAI  # pip install openai
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    vec = resp.data[0].embedding
    if len(vec) != 1536:
        raise RuntimeError(f"Query embedding dim {len(vec)} != 1536 (expected vector(1536))")
    return vec

def vector_literal(vec: List[float]) -> str:
    # pgvector accepts: '[1,2,3]'::vector
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def normalize_query_for_lex(q: str) -> str:
    """
    Keep lexical query sane for websearch_to_tsquery:
    - trim
    - collapse whitespace
    """
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q


# -----------------------
# Retrieval SQL
# -----------------------

SQL_VECTOR = """
SELECT
  cm.collection_slug,
  cm.document_id,
  cm.first_page_id,
  cm.last_page_id,
  c.id AS chunk_id,
  (c.embedding <=> %(qvec)s::vector) AS dist,
  NULL::real AS lex_rank,
  LEFT(c.text, %(preview_chars)s) AS preview
FROM chunks c
JOIN chunk_metadata cm
  ON cm.chunk_id = c.id
 AND cm.pipeline_version = c.pipeline_version
WHERE c.pipeline_version = %(chunk_pv)s
  AND (%(collection_slug)s IS NULL OR cm.collection_slug = %(collection_slug)s)
  AND c.embedding IS NOT NULL
  AND (%(doc_id)s IS NULL OR cm.document_id = %(doc_id)s)
ORDER BY c.embedding <=> %(qvec)s::vector
LIMIT %(k)s;
"""

SQL_LEX = """
WITH q AS (
  SELECT websearch_to_tsquery('simple', %(qtext)s) AS qts
)
SELECT
  cm.collection_slug,
  cm.document_id,
  cm.first_page_id,
  cm.last_page_id,
  c.id AS chunk_id,
  NULL::real AS dist,
  ts_rank_cd(c.text_tsv, q.qts) AS lex_rank,
  LEFT(c.text, %(preview_chars)s) AS preview
FROM q
JOIN chunks c ON TRUE
JOIN chunk_metadata cm
  ON cm.chunk_id = c.id
 AND cm.pipeline_version = c.pipeline_version
WHERE c.pipeline_version = %(chunk_pv)s
  AND (%(collection_slug)s IS NULL OR cm.collection_slug = %(collection_slug)s)
  AND (%(doc_id)s IS NULL OR cm.document_id = %(doc_id)s)
  AND c.text_tsv @@ q.qts
ORDER BY lex_rank DESC
LIMIT %(k)s;
"""

# Reciprocal Rank Fusion (RRF) over vector + lexical candidates
SQL_HYBRID = """
WITH
q AS (
  SELECT
    websearch_to_tsquery('simple', %(qtext)s) AS qts,
    %(qvec)s::vector AS qvec
),
vec AS (
  SELECT
    c.id AS chunk_id,
    row_number() OVER (ORDER BY c.embedding <=> q.qvec) AS r_vec,
    (c.embedding <=> q.qvec) AS dist
  FROM q
  JOIN chunks c ON c.embedding IS NOT NULL
  JOIN chunk_metadata cm
    ON cm.chunk_id = c.id
   AND cm.pipeline_version = c.pipeline_version
  WHERE c.pipeline_version = %(chunk_pv)s
    AND (%(collection_slug)s IS NULL OR cm.collection_slug = %(collection_slug)s)
    AND (%(doc_id)s IS NULL OR cm.document_id = %(doc_id)s)
  ORDER BY c.embedding <=> q.qvec
  LIMIT %(cand_vec)s
),
lex AS (
  SELECT
    c.id AS chunk_id,
    row_number() OVER (ORDER BY ts_rank_cd(c.text_tsv, q.qts) DESC) AS r_lex,
    ts_rank_cd(c.text_tsv, q.qts) AS lex_rank
  FROM q
  JOIN chunks c ON TRUE
  JOIN chunk_metadata cm
    ON cm.chunk_id = c.id
   AND cm.pipeline_version = c.pipeline_version
  WHERE c.pipeline_version = %(chunk_pv)s
    AND (%(collection_slug)s IS NULL OR cm.collection_slug = %(collection_slug)s)
    AND (%(doc_id)s IS NULL OR cm.document_id = %(doc_id)s)
    AND c.text_tsv @@ q.qts
  ORDER BY ts_rank_cd(c.text_tsv, q.qts) DESC
  LIMIT %(cand_lex)s
),
fused AS (
  SELECT
    COALESCE(vec.chunk_id, lex.chunk_id) AS chunk_id,
    vec.r_vec,
    lex.r_lex,
    vec.dist,
    lex.lex_rank,
    -- RRF score: sum(1/(k + rank))
    (CASE WHEN vec.r_vec IS NULL THEN 0 ELSE 1.0 / (%(rrf_k)s + vec.r_vec) END) +
    (CASE WHEN lex.r_lex IS NULL THEN 0 ELSE 1.0 / (%(rrf_k)s + lex.r_lex) END) AS score
  FROM vec
  FULL OUTER JOIN lex USING (chunk_id)
)
SELECT
  cm.collection_slug,
  cm.document_id,
  cm.first_page_id,
  cm.last_page_id,
  f.chunk_id,
  f.score,
  f.r_vec,
  f.r_lex,
  f.dist,
  f.lex_rank,
  LEFT(c.text, %(preview_chars)s) AS preview
FROM fused f
JOIN chunks c ON c.id = f.chunk_id
JOIN chunk_metadata cm
  ON cm.chunk_id = c.id
 AND cm.pipeline_version = c.pipeline_version
WHERE c.pipeline_version = %(chunk_pv)s
ORDER BY f.score DESC
LIMIT %(k)s;
"""


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User query text (English or Russian)")
    ap.add_argument("--mode", choices=["vector", "lex", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--chunk-pv", default="chunk_v1_full")
    ap.add_argument("--collection", default=None, help="Optional filter (venona, vassiliev, ...)")
    ap.add_argument("--doc-id", type=int, default=None, help="Optional filter by document_id")
    ap.add_argument("--preview-chars", type=int, default=1200)
    ap.add_argument("--probes", type=int, default=10, help="ivfflat probes (vector/hybrid)")
    ap.add_argument("--cand-vec", type=int, default=200, help="hybrid: vector candidate pool size")
    ap.add_argument("--cand-lex", type=int, default=200, help="hybrid: lexical candidate pool size")
    ap.add_argument("--rrf-k", type=int, default=60, help="hybrid: RRF constant (bigger = softer rank impact)")
    args = ap.parse_args()

    qtext = normalize_query_for_lex(args.query)

    params: Dict[str, Any] = {
        "chunk_pv": args.chunk_pv,
        "collection_slug": args.collection,
        "doc_id": args.doc_id,
        "k": args.k,
        "preview_chars": args.preview_chars,
        "qtext": qtext,
        "cand_vec": args.cand_vec,
        "cand_lex": args.cand_lex,
        "rrf_k": args.rrf_k,
        "qvec": None,
    }

    # Only embed if we need vectors
    if args.mode in ("vector", "hybrid"):
        qvec = embed_query(args.query)
        params["qvec"] = vector_literal(qvec)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # vector index recall tuning
            if args.mode in ("vector", "hybrid"):
                cur.execute("SET ivfflat.probes = %s;", (args.probes,))

            if args.mode == "vector":
                cur.execute(SQL_VECTOR, params)
                rows = cur.fetchall()
                print("\n=== Vector results ===\n")
                for i, r in enumerate(rows, 1):
                    (collection_slug, document_id, first_page_id, last_page_id, chunk_id, dist, _lex_rank, preview) = r
                    print(f"[{i}] chunk_id={chunk_id}  dist={dist:.4f}  collection={collection_slug} doc={document_id} pages={first_page_id}->{last_page_id}")
                    print(preview.replace("\n", " "))
                    print()

            elif args.mode == "lex":
                cur.execute(SQL_LEX, params)
                rows = cur.fetchall()
                print("\n=== Lexical results ===\n")
                for i, r in enumerate(rows, 1):
                    (collection_slug, document_id, first_page_id, last_page_id, chunk_id, _dist, lex_rank, preview) = r
                    print(f"[{i}] chunk_id={chunk_id}  lex_rank={lex_rank:.4f}  collection={collection_slug} doc={document_id} pages={first_page_id}->{last_page_id}")
                    print(preview.replace("\n", " "))
                    print()

            else:  # hybrid
                cur.execute(SQL_HYBRID, params)
                rows = cur.fetchall()
                print("\n=== Hybrid results (RRF) ===\n")
                for i, r in enumerate(rows, 1):
                    (collection_slug, document_id, first_page_id, last_page_id, chunk_id, score, r_vec, r_lex, dist, lex_rank, preview) = r
                    print(f"[{i}] chunk_id={chunk_id}  score={score:.6f}  r_vec={r_vec} r_lex={r_lex}  collection={collection_slug} doc={document_id} pages={first_page_id}->{last_page_id}")
                    if dist is not None:
                        print(f"    dist={dist:.4f}", end="")
                    if lex_rank is not None:
                        print(f"  lex_rank={lex_rank:.4f}", end="")
                    print()
                    print(preview.replace("\n", " "))
                    print()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
