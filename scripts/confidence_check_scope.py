#!/usr/bin/env python3
"""
Confidence check: verify non-V/V retrieval works when excluding Venona/Vassiliev.

Run after Phase 2 to confirm:
1. hybrid_search with exclude_collections returns non-empty hits from other corpora
2. Those hits reach the model context (optional full flow)

Usage:
    python scripts/confidence_check_scope.py
    python scripts/confidence_check_scope.py --query "Silvermaster network" --exclude venona,vassiliev
"""
import argparse
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from retrieval.agent.tools import hybrid_search_tool


def main():
    ap = argparse.ArgumentParser(description="Confidence check: non-V/V retrieval")
    ap.add_argument("--query", default="Silvermaster network", help="Search query")
    ap.add_argument("--exclude", default="venona,vassiliev", help="Comma-separated collection slugs to exclude")
    ap.add_argument("--top-k", type=int, default=50, help="Top-k hits")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set. Run: source friday_env.sh", file=sys.stderr)
        sys.exit(1)

    import psycopg2
    conn = psycopg2.connect(dsn)
    exclude_list = [s.strip() for s in args.exclude.split(",") if s.strip()]

    print(f"Query: {args.query!r}", file=sys.stderr)
    print(f"Excluding collections: {exclude_list}", file=sys.stderr)

    result = hybrid_search_tool(
        conn=conn,
        query=args.query,
        top_k=args.top_k,
        collections=None,
        exclude_collections=exclude_list,
    )

    if not result.success:
        print(f"FAIL: hybrid_search failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    if not result.chunk_ids:
        print("FAIL: No hits returned. Non-V/V retrieval may be broken.", file=sys.stderr)
        sys.exit(1)

    # Load collection_slug for each hit to compute distribution
    from retrieval.agent.v11_tools import _load_catalog
    catalog = _load_catalog(conn, result.chunk_ids, result.scores)
    coll_counts = Counter(h.collection or "unknown" for h in catalog)

    print(f"PASS: {len(result.chunk_ids)} hits returned", file=sys.stderr)
    print(f"Per-collection distribution: {dict(coll_counts)}", file=sys.stderr)
    print(f"index_used: {result.metadata.get('index_used', '?')}", file=sys.stderr)

    # Sanity: we excluded V/V, so hits should not be from venona/vassiliev
    vv = sum(coll_counts.get(c, 0) for c in exclude_list)
    if vv > 0:
        print(f"WARN: {vv} hits from excluded collections (expected 0)", file=sys.stderr)

    print("Confidence check passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
