"""Export Part 7's embedded OCR page texts from prod (read-only) for dry-run adjudication."""
import os
import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("""
    SELECT d.id, count(p.id) FROM documents d LEFT JOIN pages p ON p.document_id = d.id
    WHERE d.source_name LIKE 'FBI File Silvermaster Part 7 %' GROUP BY d.id
""")
row = cur.fetchone()
print("doc:", row)
doc_id = row[0]
os.makedirs("data/transcripts/dryrun_part7/embedded", exist_ok=True)
cur.execute("SELECT page_seq, raw_text FROM pages WHERE document_id = %s ORDER BY page_seq", (doc_id,))
n = 0
for seq, txt in cur.fetchall():
    with open(f"data/transcripts/dryrun_part7/embedded/{seq:04d}.txt", "w", encoding="utf-8") as f:
        f.write(txt or "")
    n += 1
print(f"wrote {n} embedded pages for doc {doc_id}")
