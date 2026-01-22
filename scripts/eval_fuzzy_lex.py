#!/usr/bin/env python3
"""
Evaluate fuzzy lexical integration by comparing hybrid retrieval with and without
per-token fuzzy expansion. Records overlap@k into retrieval_evaluations.

Usage (PowerShell):
  $env:DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
  python scripts/eval_fuzzy_lex.py --k 10 --chunk-pv chunk_v1_full
"""

import argparse
import os
import sys
from typing import List, Dict, Any

# Ensure repo root importable when running as script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import psycopg2
from psycopg2.extras import Json

from retrieval.ops import hybrid_rrf, SearchFilters


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


GOLDEN = [
    ("silvermastre", "silvermaster", "chunk_v1_silvermaster_structured_4k"),
    ("silvehmaster", "silvermaster", "chunk_v1_silvermaster_structured_4k"),
    ("silverkaster", "silvermaster", "chunk_v1_silvermaster_structured_4k"),
]


def overlap_at_k(a: List[int], b: List[int], k: int) -> float:
    set_a = set(a[:k])
    set_b = set(b[:k])
    if k == 0:
        return 0.0
    return len(set_a & set_b) / k


def run_eval(
    conn,
    query: str,
    collection_slug: str,
    chunk_pv: str,
    k: int,
    fuzzy_top_k: int,
    fuzzy_max_variants: int,
    fuzzy_min_similarity: float,
):
    # First check if chunks exist for this filter
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) 
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.pipeline_version = %s
              AND cm.pipeline_version = %s
              AND cm.collection_slug = %s
            """,
            (chunk_pv, chunk_pv, collection_slug),
        )
        chunk_count = cur.fetchone()[0]
        print(f"  Chunks matching filter: {chunk_count}")
    
    if chunk_count == 0:
        print(f"  WARNING: No chunks found for chunk_pv={chunk_pv!r} collection_slug={collection_slug!r}")
        return {
            "overlap": 0.0,
            "base_ids": [],
            "fuzzy_ids": [],
            "num_base": 0,
            "num_fuzzy": 0,
        }
    
    filters = SearchFilters(chunk_pv=chunk_pv, collection_slugs=[collection_slug])

    print(f"  Running base query (fuzzy_lex_enabled=False)...")
    hits_base = hybrid_rrf(
        conn,
        query,
        filters=filters,
        k=k,
        expand_concordance=True,
        use_soft_lex=False,
        fuzzy_lex_enabled=False,
    )
    
    print(f"  Running fuzzy query (fuzzy_lex_enabled=True)...")
    hits_fuzzy = hybrid_rrf(
        conn,
        query,
        filters=filters,
        k=k,
        expand_concordance=True,
        use_soft_lex=False,
        fuzzy_lex_enabled=True,
        fuzzy_lex_top_k_per_token=fuzzy_top_k,
        fuzzy_lex_max_total_variants=fuzzy_max_variants,
        fuzzy_lex_min_similarity=fuzzy_min_similarity,
    )

    base_ids = [h.chunk_id for h in hits_base]
    fuzzy_ids = [h.chunk_id for h in hits_fuzzy]

    ov = overlap_at_k(base_ids, fuzzy_ids, k)
    return {
        "overlap": ov,
        "base_ids": base_ids,
        "fuzzy_ids": fuzzy_ids,
        "num_base": len(base_ids),
        "num_fuzzy": len(fuzzy_ids),
    }


def save_metric(
    conn,
    *,
    query_text: str,
    collection_slug: str,
    chunk_pv: str,
    k: int,
    overlap: float,
    eval_config: Dict[str, Any],
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO retrieval_evaluations (
                query_text,
                query_lang_version,
                metric_name,
                metric_value,
                search_type,
                chunk_pv,
                collection_slug,
                evaluation_config,
                retrieval_run_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL);
            """,
            (
                query_text,
                "qv_fuzzy_lex",
                f"overlap@{k}",
                float(overlap),
                "hybrid",
                chunk_pv,
                collection_slug,
                Json(eval_config),
            ),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Evaluate fuzzy lexical (overlap vs baseline)")
    ap.add_argument("--k", type=int, default=10, help="Top-k for overlap")
    ap.add_argument("--chunk-pv", default=None, help="Override chunk_pv (default taken from GOLDEN entries)")
    ap.add_argument("--fuzzy-top-k", type=int, default=5, help="Fuzzy: top-k variants per token (default: 5)")
    ap.add_argument("--fuzzy-max-variants", type=int, default=50, help="Fuzzy: max total variants (default: 50)")
    ap.add_argument("--fuzzy-min-similarity", type=float, default=0.4, help="Fuzzy: min similarity threshold (default: 0.4)")
    args = ap.parse_args()

    conn = get_conn()
    try:
        for query, collection_slug, default_pv in GOLDEN:
            chunk_pv = args.chunk_pv or default_pv
            print(f"\n=== Evaluating query={query!r} collection={collection_slug} chunk_pv={chunk_pv} ===")
            
            # Check if dictionary exists for this slice
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, built_at 
                    FROM corpus_dictionary_builds
                    WHERE chunk_pv = %s
                      AND collection_slug IS NOT DISTINCT FROM %s
                      AND norm_version = 'norm_v1'
                    ORDER BY built_at DESC
                    LIMIT 1
                    """,
                    (chunk_pv, collection_slug),
                )
                build_row = cur.fetchone()
                if build_row:
                    print(f"  Dictionary build_id={build_row[0]} found (built_at={build_row[1]})")
                else:
                    print(f"  WARNING: No dictionary build found for chunk_pv={chunk_pv!r} collection_slug={collection_slug!r}")
                    print(f"  Run: python scripts/build_corpus_dictionary.py --chunk-pv {chunk_pv} --collection-slug {collection_slug}")

            result = run_eval(
                conn,
                query,
                collection_slug,
                chunk_pv,
                args.k,
                args.fuzzy_top_k,
                args.fuzzy_max_variants,
                args.fuzzy_min_similarity,
            )
            print(f"overlap@{args.k}: {result['overlap']:.3f}  base={result['num_base']}  fuzzy={result['num_fuzzy']}")

            eval_config = {
                "k": args.k,
                "chunk_pv": chunk_pv,
                "collection_slug": collection_slug,
                "mode": "hybrid",
                "fuzzy_enabled": True,
            }
            save_metric(
                conn,
                query_text=query,
                collection_slug=collection_slug,
                chunk_pv=chunk_pv,
                k=args.k,
                overlap=result["overlap"],
                eval_config=eval_config,
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
