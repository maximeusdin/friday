#!/usr/bin/env python3
"""
Minimal script to debug why V11 search returns 0 hits.
Run: source friday_env.sh && python scripts/debug_search.py "Silvermaster network"
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Silvermaster network"
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set. Run: source friday_env.sh")
        sys.exit(1)

    import psycopg2
    conn = psycopg2.connect(dsn)

    # 1. Check chunks + metadata join
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.embedding IS NOT NULL
        """)
        n = cur.fetchone()[0]
        print(f"1. Chunks with embedding + metadata: {n}")

    # 2. Check pipeline_version values
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.pipeline_version, COUNT(*)
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.embedding IS NOT NULL
            GROUP BY c.pipeline_version
        """)
        rows = cur.fetchall()
        print(f"2. Pipeline versions: {dict(rows)}")

    # 3. Run hybrid_rrf directly
    from retrieval.ops import hybrid_rrf, SearchFilters
    filters = SearchFilters(chunk_pv=None, collection_slugs=None)
    hits = hybrid_rrf(conn, query, filters=filters, k=10, log_run=False)
    print(f"3. hybrid_rrf result: {len(hits)} hits")
    if hits:
        for h in hits[:3]:
            print(f"   - chunk_id={h.chunk_id} coll={h.collection_slug} score={h.score}")

    # 4. If 0, try with explicit chunk_pv
    if len(hits) == 0 and rows:
        pv = rows[0][0]
        print(f"4. Retrying with chunk_pv={pv!r}")
        filters2 = SearchFilters(chunk_pv=pv, collection_slugs=None)
        hits2 = hybrid_rrf(conn, query, filters=filters2, k=10, log_run=False)
        print(f"   Result: {len(hits2)} hits")

    conn.close()

if __name__ == "__main__":
    main()
