#!/usr/bin/env python3
"""
Check why a Venona chunk gets 0 PEM mappings.

Shows: chunk -> chunk_pages -> page_ids -> PEM rows for those pages.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def get_conn():
    import psycopg2
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "neh"),
        user=os.getenv("DB_USER", "neh"),
        password=os.getenv("DB_PASS", "neh"),
    )

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-id", type=int, default=41168, help="Venona chunk to inspect")
    ap.add_argument("--limit", type=int, default=5, help="Number of Venona chunks to sample")
    args = ap.parse_args()

    conn = get_conn()
    with conn.cursor() as cur:
        # Sample Venona chunks
        cur.execute("""
            SELECT c.id, d.source_name
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            JOIN chunk_pages cp ON cp.chunk_id = c.id
            JOIN pages p ON p.id = cp.page_id
            JOIN documents d ON d.id = p.document_id
            WHERE cm.collection_slug = 'venona'
              AND cm.pipeline_version = 'chunk_v1_full'
            ORDER BY c.id
            LIMIT %s
        """, (args.limit,))
        samples = cur.fetchall()
        print(f"Sample Venona chunks (first {args.limit} by id):")
        for chunk_id, doc_name in samples:
            cur.execute("SELECT page_id FROM chunk_pages WHERE chunk_id = %s ORDER BY span_order", (chunk_id,))
            page_ids = [r[0] for r in cur.fetchall()]
            cur.execute("""
                SELECT COUNT(*) FROM page_entity_mentions
                WHERE page_id = ANY(%s) AND collection_slug = 'venona'
            """, (page_ids,))
            pem_count = cur.fetchone()[0]
            print(f"  chunk {chunk_id} doc={doc_name!r} page_ids={page_ids} PEM_rows={pem_count}")

        # PEM coverage for Venona
        cur.execute("""
            SELECT COUNT(DISTINCT pem.page_id)
            FROM page_entity_mentions pem
            WHERE pem.collection_slug = 'venona'
        """)
        venona_pem_pages = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(DISTINCT cp.page_id)
            FROM chunk_pages cp
            JOIN chunks c ON c.id = cp.chunk_id
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE cm.collection_slug = 'venona'
        """)
        venona_chunk_pages = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(DISTINCT cp.page_id)
            FROM chunk_pages cp
            JOIN chunks c ON c.id = cp.chunk_id
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            JOIN page_entity_mentions pem ON pem.page_id = cp.page_id AND pem.collection_slug = 'venona'
            WHERE cm.collection_slug = 'venona'
        """)
        venona_covered = cur.fetchone()[0]
        print(f"\nVenona PEM coverage: {venona_covered} / {venona_chunk_pages} chunk pages have PEM rows")
        print(f"Venona PEM total: {venona_pem_pages} distinct pages with PEM")

    conn.close()

if __name__ == "__main__":
    main()
