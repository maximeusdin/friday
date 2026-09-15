"""Which silvermaster chunk still matches whole-word 'fuer' in the clean layer, and embed freshness for doc 522."""
import os, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""
    SELECT c.id, cm.document_id,
           substring(COALESCE(c.clean_text, c.text) from '(?i).{0,60}\\mfuer\\M.{0,60}')
    FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE cm.collection_slug = 'silvermaster'
      AND COALESCE(c.clean_text, c.text) ~* '\\mfuer\\M'
""")
for r in cur.fetchall():
    print(r)

cur.execute("""
    SELECT count(*) FILTER (WHERE c.embedded_at::date = current_date) AS embedded_today,
           count(*) AS total
    FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE cm.document_id = 522
""")
print("embedded today / total:", cur.fetchone())
