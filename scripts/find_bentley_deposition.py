"""Locate the Bentley deposition: find 'meekirk' pages in prod, with document context."""
import os, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== DB identity check ===")
cur.execute("SELECT current_database(), inet_server_addr()")
print(cur.fetchone())

print("\n=== Collections matching silvermaster ===")
cur.execute("""
    SELECT c.id, c.slug, c.title, count(d.id) AS docs
    FROM collections c LEFT JOIN documents d ON d.collection_id = c.id
    WHERE c.slug ILIKE '%silver%' OR c.title ILIKE '%silver%'
    GROUP BY c.id ORDER BY c.id
""")
for r in cur.fetchall():
    print(r)

print("\n=== Pages containing 'meekirk' (any collection) ===")
cur.execute("""
    SELECT col.slug, d.id AS doc_id, d.source_name, d.volume, d.source_ref,
           p.id AS page_id, p.page_seq, p.pdf_page_number, p.logical_page_label
    FROM pages p
    JOIN documents d ON d.id = p.document_id
    JOIN collections col ON col.id = d.collection_id
    WHERE p.raw_text ILIKE '%meekirk%'
    ORDER BY d.id, p.page_seq
""")
rows = cur.fetchall()
for r in rows:
    print(r)

if rows:
    doc_id = rows[-1][1]
    print(f"\n=== Document {doc_id} summary ===")
    cur.execute("""
        SELECT d.source_name, d.volume, d.source_ref, d.metadata,
               count(p.id), min(p.page_seq), max(p.page_seq)
        FROM documents d JOIN pages p ON p.document_id = d.id
        WHERE d.id = %s GROUP BY d.id
    """, (doc_id,))
    print(cur.fetchone())
