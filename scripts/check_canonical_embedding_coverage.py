"""
Health check: report missing canonical embedding rows.

Checks that count(chunks) == count(chunk_embeddings_canonical) for critical
pipeline versions and collections. Reports gaps for monitoring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL env var")
    import psycopg2
    return psycopg2.connect(dsn)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Check chunk_embeddings_canonical coverage")
    ap.add_argument("--chunk-pv", default="chunk_v1_full")
    ap.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    ap.add_argument("--collection-slug", default=None, help="Filter to one collection; default all")
    args = ap.parse_args()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Total chunks in scope
            if args.collection_slug:
                cur.execute("""
                    SELECT COUNT(*) FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE c.pipeline_version = %s
                    AND cm.pipeline_version = %s
                    AND cm.collection_slug = %s
                """, (args.chunk_pv, args.chunk_pv, args.collection_slug))
            else:
                cur.execute("""
                    SELECT COUNT(*) FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE c.pipeline_version = %s
                    AND cm.pipeline_version = %s
                """, (args.chunk_pv, args.chunk_pv))
            total_chunks = cur.fetchone()[0]

            # Chunks with canonical row (same scope)
            if args.collection_slug:
                cur.execute("""
                    SELECT COUNT(*) FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    JOIN chunk_embeddings_canonical cec
                      ON cec.chunk_id = c.id
                      AND cec.pipeline_version = %s
                      AND cec.embedding_model = %s
                    WHERE c.pipeline_version = %s
                    AND cm.pipeline_version = %s
                    AND cm.collection_slug = %s
                """, (args.chunk_pv, args.embedding_model, args.chunk_pv, args.chunk_pv, args.collection_slug))
            else:
                cur.execute("""
                    SELECT COUNT(*) FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    JOIN chunk_embeddings_canonical cec
                      ON cec.chunk_id = c.id
                      AND cec.pipeline_version = %s
                      AND cec.embedding_model = %s
                    WHERE c.pipeline_version = %s
                    AND cm.pipeline_version = %s
                """, (args.chunk_pv, args.embedding_model, args.chunk_pv, args.chunk_pv))
            total_canonical = cur.fetchone()[0]

            # Missing
            missing = total_chunks - total_canonical
            pct = (total_canonical / total_chunks * 100) if total_chunks else 0

            print(f"chunk_pv={args.chunk_pv} embedding_model={args.embedding_model}")
            if args.collection_slug:
                print(f"  collection={args.collection_slug}")
            print(f"  chunks: {total_chunks}")
            print(f"  canonical rows: {total_canonical}")
            print(f"  missing: {missing} ({100 - pct:.1f}%)")
            if missing > 0:
                print("  STATUS: INCOMPLETE - run scripts/embed_canonical_chunks.py")
                sys.exit(1)
            else:
                print("  STATUS: OK")
            sys.exit(0)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
