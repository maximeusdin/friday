#!/usr/bin/env python3
"""
Check if a collection exists in the local database and show its stats.

Usage:
  python scripts/check_collection.py silvermaster
  python scripts/check_collection.py --list   # list all collections
  DATABASE_URL=postgresql://... python scripts/check_collection.py silvermaster
"""

import os
import sys

import psycopg2


def get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    return "postgresql://neh:neh@localhost:5432/neh"


def main():
    list_all = "--list" in sys.argv or "-l" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    search = args[0] if args else None

    conn = psycopg2.connect(get_dsn())
    try:
        with conn.cursor() as cur:
            if list_all:
                cur.execute("""
                    SELECT c.id, c.slug, c.title, COUNT(DISTINCT d.id) AS docs
                    FROM collections c
                    LEFT JOIN documents d ON d.collection_id = c.id
                    GROUP BY c.id, c.slug, c.title
                    ORDER BY c.slug
                """)
                rows = cur.fetchall()
                print("Collections in database:")
                for r in rows:
                    cid = r[0]
                    cur.execute("""
                        SELECT COUNT(DISTINCT ch.id) FROM chunks ch
                        JOIN chunk_pages cp ON cp.chunk_id = ch.id
                        JOIN pages p ON p.id = cp.page_id
                        JOIN documents d ON d.id = p.document_id
                        WHERE d.collection_id = %s
                    """, (cid,))
                    chunks = cur.fetchone()[0]
                    print(f"  {r[1]:30} id={cid}  docs={r[3]}  chunks={chunks}")
                return

            if not search:
                print("Usage: python scripts/check_collection.py <slug>")
                print("       python scripts/check_collection.py --list")
                return

            search_lower = search.lower()
            cur.execute("""
                SELECT id, slug, title FROM collections
                WHERE LOWER(slug) LIKE %s OR LOWER(title) LIKE %s
            """, (f"%{search_lower}%", f"%{search_lower}%"))
            collections = cur.fetchall()

            if not collections:
                print(f"No collection matching '{search}' found.")
                cur.execute("SELECT slug FROM collections ORDER BY slug")
                all_slugs = [r[0] for r in cur.fetchall()]
                print(f"Available collections: {', '.join(all_slugs[:20])}{'...' if len(all_slugs) > 20 else ''}")
                return

            for cid, slug, title in collections:
                cur.execute("""
                    SELECT COUNT(*) FROM documents WHERE collection_id = %s
                """, (cid,))
                doc_count = cur.fetchone()[0]
                cur.execute("""
                    SELECT COUNT(DISTINCT ch.id) FROM chunks ch
                    JOIN chunk_pages cp ON cp.chunk_id = ch.id
                    JOIN pages p ON p.id = cp.page_id
                    JOIN documents d ON d.id = p.document_id
                    WHERE d.collection_id = %s
                """, (cid,))
                chunk_count = cur.fetchone()[0]
                print(f"Collection: {slug} (id={cid})")
                print(f"  Title: {title}")
                print(f"  Documents: {doc_count}")
                print(f"  Chunks: {chunk_count}")
                if doc_count == 0:
                    print("  -> No documents. Run ingest for this collection.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
