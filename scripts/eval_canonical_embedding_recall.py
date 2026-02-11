#!/usr/bin/env python3
"""
Evaluate canonical embedding recall — minimal harness for "did this help?".

Recall canaries:
  - "Office of Strategic Services" — expect chunks with cabin→OSS in block
  - "CABIN" — should not regress (at least as good as baseline)

Precision canary:
  - "FBI" — top K should not collapse to random Venona pages

Usage:
  python scripts/eval_canonical_embedding_recall.py --k 20 --chunk-pv chunk_v1_full
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import hybrid_rrf, SearchFilters


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    import psycopg2
    return psycopg2.connect(dsn)


def overlap_at_k(a: list, b: list, k: int) -> float:
    if k == 0:
        return 0.0
    set_a = set(a[:k])
    set_b = set(b[:k])
    return len(set_a & set_b) / k


def run_query(
    conn,
    query: str,
    scope: SearchFilters,
    k: int,
    use_canonical: bool,
) -> list:
    """Run hybrid_rrf and return chunk_ids in order."""
    hits = hybrid_rrf(
        conn,
        query,
        filters=scope,
        k=k,
        log_run=False,
        use_canonical_embeddings=use_canonical,
    )
    return [h.chunk_id for h in hits]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--chunk-pv", default="chunk_v1_full")
    ap.add_argument("--collection", default="venona", help="Collection for scope")
    args = ap.parse_args()

    conn = get_conn()
    scope = SearchFilters(
        chunk_pv=args.chunk_pv,
        collection_slugs=[args.collection],
    )

    print("Canonical embedding recall eval")
    print(f"  k={args.k} chunk_pv={args.chunk_pv} collection={args.collection}\n")

    # Recall canaries
    for query, label in [
        ("Office of Strategic Services", "OSS recall (canonical name)"),
        ("CABIN", "CABIN recall (codename)"),
    ]:
        baseline = run_query(conn, query, scope, args.k, use_canonical=False)
        canonical = run_query(conn, query, scope, args.k, use_canonical=True)
        ov = overlap_at_k(baseline, canonical, args.k)
        print(f"{label}: overlap@k={ov:.2f}")
        print(f"  baseline top5: {baseline[:5]}")
        print(f"  canonical top5: {canonical[:5]}")

    # Precision canary: FBI — distinct collections in top K
    print("\nPrecision canary (FBI):")
    scope_all = SearchFilters(chunk_pv=args.chunk_pv)
    baseline = run_query(conn, "FBI", scope_all, args.k, use_canonical=False)
    canonical = run_query(conn, "FBI", scope_all, args.k, use_canonical=True)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id, cm.collection_slug
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.id = ANY(%s)
        """, (baseline[:args.k],))
        base_colls = set(r[1] for r in cur.fetchall())
        cur.execute("""
            SELECT c.id, cm.collection_slug
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.id = ANY(%s)
        """, (canonical[:args.k],))
        can_colls = set(r[1] for r in cur.fetchall())
    print(f"  baseline distinct collections in top {args.k}: {len(base_colls)}")
    print(f"  canonical distinct collections in top {args.k}: {len(can_colls)}")
    if len(can_colls) < len(base_colls) * 0.5:
        print("  WARNING: canonical collapsed collections")
    else:
        print("  OK")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
