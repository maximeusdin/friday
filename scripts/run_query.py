#!/usr/bin/env python3
"""
run_query.py --session <id|label> <query> [options...]

Runs a query and assigns it to a session. Wraps retrieval functions with session support.

Usage:
    # Create session first
    python scripts/new_session.py --label "Oppenheimer investigation"
    # Then run queries in that session
    python scripts/run_query.py --session "Oppenheimer investigation" "Oppenheimer atomic bomb"
    python scripts/run_query.py --session 5 "Rosenberg spy"
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from retrieval.ops import (
    get_conn,
    hybrid_rrf,
    vector_search,
    SearchFilters,
)


def resolve_session_id(conn, session_arg: str) -> int:
    """
    Resolve session by ID (if numeric) or label (if string).
    Returns session_id.
    """
    with conn.cursor() as cur:
        # Try as ID first
        if session_arg.isdigit():
            cur.execute(
                "SELECT id FROM research_sessions WHERE id = %s",
                (int(session_arg),)
            )
            row = cur.fetchone()
            if row:
                return row[0]
        
        # Try as label
        cur.execute(
            "SELECT id FROM research_sessions WHERE label = %s",
            (session_arg,)
        )
        row = cur.fetchone()
        if row:
            return row[0]
        
        raise SystemExit(f"Session not found: '{session_arg}' (not an ID or label)")


def main():
    ap = argparse.ArgumentParser(description="Run a query in a session")
    ap.add_argument("--session", type=str, required=True, help="Session ID or label")
    ap.add_argument("query", help="Query text")
    
    # Retrieval options (similar to retrieve_chunks.py)
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid", help="Search mode")
    ap.add_argument("--k", type=int, default=15, help="Number of results")
    ap.add_argument("--chunk-pv", default="chunk_v1_full", help="Chunk pipeline version")
    ap.add_argument("--collection", action="append", default=[], help="Filter by collection slug (repeatable)")
    ap.add_argument("--preview-chars", type=int, default=1200, help="Preview length")
    ap.add_argument("--probes", type=int, default=10, help="ivfflat probes")
    ap.add_argument("--top-n-vec", type=int, default=200, help="Vector candidate pool")
    ap.add_argument("--top-n-lex", type=int, default=200, help="Lexical candidate pool")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant")
    
    # Fuzzy lexical options
    ap.add_argument("--fuzzy-top-k", type=int, default=5, help="Fuzzy lexical: top-k variants per token")
    ap.add_argument("--fuzzy-max-variants", type=int, default=50, help="Fuzzy lexical: max total variants")
    ap.add_argument("--fuzzy-min-similarity", type=float, default=0.4, help="Fuzzy lexical: min similarity")
    
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        # Resolve session
        session_id = resolve_session_id(conn, args.session)
        print(f"Session: {args.session} (ID: {session_id})", file=sys.stderr)
        
        # Build filters
        filters = SearchFilters(
            chunk_pv=args.chunk_pv,
            collection_slugs=args.collection if args.collection else None,
        )
        
        # Run query (session_id passed directly to retrieval functions)
        if args.mode == "vector":
            results = vector_search(
                conn,
                args.query,
                filters=filters,
                k=args.k,
                preview_chars=args.preview_chars,
                probes=args.probes,
                log_run=True,
                session_id=session_id,
            )
        else:  # hybrid
            results = hybrid_rrf(
                conn,
                args.query,
                filters=filters,
                k=args.k,
                preview_chars=args.preview_chars,
                probes=args.probes,
                top_n_vec=args.top_n_vec,
                top_n_lex=args.top_n_lex,
                rrf_k=args.rrf_k,
                fuzzy_lex_enabled=True,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
                log_run=True,
                session_id=session_id,
            )
        
        # Get run_id from the most recent retrieval_run for this query
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM retrieval_runs
                WHERE query_text = %s AND session_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (args.query, session_id)
            )
            run_row = cur.fetchone()
            if run_row:
                run_id = run_row[0]
                print(f"Run ID: {run_id}", file=sys.stderr)
        
        # Display results
        print(f"\n=== Results ({len(results)} chunks) ===\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] chunk_id={r.chunk_id}  collection={r.collection_slug}  doc={r.document_id}")
            if r.preview:
                print(f"    {r.preview[:200].replace(chr(10), ' ')}...")
            print()
            
    finally:
        conn.close()


if __name__ == "__main__":
    main()
