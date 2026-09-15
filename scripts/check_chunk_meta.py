"""Inspect chunk_metadata + true text structure for doc 522 chunks."""
import os, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== chunk_metadata columns ===")
cur.execute("""
    SELECT column_name, data_type FROM information_schema.columns
    WHERE table_name = 'chunk_metadata' ORDER BY ordinal_position
""")
for r in cur.fetchall():
    print(r)

print("\n=== chunk_metadata sample for doc 522 chunks ===")
cur.execute("""
    SELECT DISTINCT cm.*
    FROM chunk_metadata cm
    WHERE cm.chunk_id IN (
        SELECT cp.chunk_id FROM chunk_pages cp
        JOIN pages p ON p.id = cp.page_id WHERE p.document_id = 522
    )
    LIMIT 3
""")
cols = [d[0] for d in cur.description]
for row in cur.fetchall():
    print({c: (str(v)[:120] if v is not None else None) for c, v in zip(cols, row)})

print("\n=== the Meekirk chunk ===")
cur.execute("""
    SELECT c.id, length(c.text), left(c.text, 200), right(c.text, 150)
    FROM chunks c
    WHERE c.pipeline_version = 'chunk_v1_silvermaster_structured_4k'
      AND c.text ILIKE '%meekirk%'
      AND c.id IN (SELECT cp.chunk_id FROM chunk_pages cp
                   JOIN pages p ON p.id = cp.page_id WHERE p.document_id = 522)
""")
for r in cur.fetchall():
    print(r)

print("\n=== its chunk_metadata row ===")
cur.execute("""
    SELECT cm.* FROM chunk_metadata cm
    WHERE cm.chunk_id IN (
        SELECT c.id FROM chunks c
        WHERE c.pipeline_version = 'chunk_v1_silvermaster_structured_4k'
          AND c.text ILIKE '%meekirk%'
          AND c.id IN (SELECT cp.chunk_id FROM chunk_pages cp
                       JOIN pages p ON p.id = cp.page_id WHERE p.document_id = 522))
""")
cols = [d[0] for d in cur.description]
for row in cur.fetchall():
    print({c: (str(v)[:200] if v is not None else None) for c, v in zip(cols, row)})

print("\n=== pages-per-chunk distribution (doc 522) ===")
cur.execute("""
    SELECT npages, count(*) FROM (
        SELECT cp.chunk_id, count(*) AS npages
        FROM chunk_pages cp JOIN pages p ON p.id = cp.page_id
        WHERE p.document_id = 522 GROUP BY cp.chunk_id
    ) t GROUP BY npages ORDER BY npages LIMIT 12
""")
print(cur.fetchall())
