"""Export adjudication inputs for the Bentley deposition pilot (doc 522).

Writes:
  data/transcripts/bentley_deposition/embedded/{page:04d}.txt  (existing PDF text layer)
  data/transcripts/bentley_deposition/gazetteer.txt            (known entity names)
"""
import os
import psycopg2

DOC_ID = 522
BASE = "data/transcripts/bentley_deposition"

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

os.makedirs(f"{BASE}/embedded", exist_ok=True)
cur.execute(
    "SELECT page_seq, raw_text FROM pages WHERE document_id = %s ORDER BY page_seq",
    (DOC_ID,),
)
n = 0
for seq, txt in cur.fetchall():
    with open(f"{BASE}/embedded/{seq:04d}.txt", "w", encoding="utf-8") as f:
        f.write(txt or "")
    n += 1
print(f"wrote {n} embedded page texts")

names = set()
cur.execute("SELECT canonical_name FROM entities")
for (nm,) in cur.fetchall():
    if nm:
        names.add(nm.strip())
try:
    cur.execute("SELECT surface FROM entity_aliases")
    for (nm,) in cur.fetchall():
        if nm:
            names.add(nm.strip())
except Exception as e:
    conn.rollback()
    print(f"entity_aliases skipped: {e}")

with open(f"{BASE}/gazetteer.txt", "w", encoding="utf-8") as f:
    for nm in sorted(names):
        f.write(nm + "\n")
print(f"wrote gazetteer with {len(names)} names")
