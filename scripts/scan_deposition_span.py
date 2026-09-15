"""Scan document 522 (Silvermaster Part 6) page-by-page to delimit the Bentley deposition."""
import os, re, psycopg2

DOC_ID = 522
conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute(
    "SELECT page_seq, raw_text FROM pages WHERE document_id = %s ORDER BY page_seq",
    (DOC_ID,),
)
pages = cur.fetchall()
print(f"doc {DOC_ID}: {len(pages)} pages\n")

MARKERS = [
    "meekirk", "fuhr", "fuer", "sworn", "deposition", "signature", "witnessed",
    "elizabeth", "bentley", "56402", "make the following statement",
    "read the foregoing", "voluntary",
]

for seq, txt in pages:
    t = (txt or "").strip()
    head = re.sub(r"\s+", " ", t[:110])
    low = t.lower()
    hits = [m for m in MARKERS if m in low]
    print(f"{seq:>3} | {len(t):>5}ch | {','.join(hits)[:60]:60} | {head}")

print("\n=== Full text of page 3 (the Meekirk page) ===")
for seq, txt in pages:
    if seq == 3:
        print(txt)
