"""Diagnose unmapped chunks: find first normalized divergence between chunk text and page run."""
import os, psycopg2

def norm(t):
    return "".join((t or "").split()).replace("-", "")

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

for chunk_id, start_page in [(53010, 9), (53020, 19), (53037, 36), (53040, 39), (53078, 77)]:
    cur.execute("SELECT text FROM chunks WHERE id = %s", (chunk_id,))
    ctext = norm(cur.fetchone()[0])
    cur.execute(
        "SELECT page_seq, raw_text FROM pages WHERE document_id = 522 AND page_seq >= %s ORDER BY page_seq LIMIT 4",
        (start_page,),
    )
    acc = ""
    print(f"\n=== chunk {chunk_id} (len {len(ctext)}) from page {start_page} ===")
    for seq, raw in cur.fetchall():
        p = norm(raw)
        joined = acc + p
        if ctext.startswith(joined):
            acc = joined
            print(f"  page {seq}: full prefix ok (acc {len(acc)})")
            continue
        i = 0
        while i < min(len(joined), len(ctext)) and joined[i] == ctext[i]:
            i += 1
        print(f"  page {seq}: diverges at char {i} (acc was {len(acc)})")
        print(f"    chunk : ...{ctext[max(0,i-40):i]}[{ctext[i:i+40]}]")
        print(f"    pages : ...{joined[max(0,i-40):i]}[{joined[i:i+40]}]")
        break
