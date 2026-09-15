"""Read-only: export the Silvermaster batch manifest (all docs except 522/532)."""
import json
import os
import urllib.parse

import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("""
    SELECT d.id, d.source_name, d.source_ref, count(p.id) AS pages
    FROM documents d JOIN pages p ON p.document_id = d.id
    WHERE d.collection_id = 77 AND d.id NOT IN (522, 532)
    GROUP BY d.id ORDER BY count(p.id) DESC
""")
docs = []
for doc_id, name, ref, pages in cur.fetchall():
    docs.append({
        "doc_id": doc_id,
        "source_name": name,
        "pages": pages,
        "pdf_url": "https://fridayarchive.org/" + urllib.parse.quote(ref),
    })
out = "data/transcripts/batch_silvermaster/manifest.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(docs, open(out, "w"), indent=1)
print(f"{len(docs)} docs, {sum(d['pages'] for d in docs)} pages -> {out}")
