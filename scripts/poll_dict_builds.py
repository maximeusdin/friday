"""Poll corpus_dictionary_builds until the batch finishes (count stable) — read-only."""
import os, time
import psycopg2

TARGET = 39  # silvermaster + 38 batch builds
last, stable = -1, 0
for i in range(60):  # up to ~60 min
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT count(*), coalesce(max(collection_slug), '') FROM corpus_dictionary_builds")
    n, latest = cur.fetchone()
    conn.close()
    print(f"poll {i}: {n} builds (latest slug: {latest})", flush=True)
    if n >= TARGET:
        print("all builds present")
        break
    stable = stable + 1 if n == last else 0
    if stable >= 5 and n > 3:
        print(f"count stable at {n} for 5 polls — batch appears finished or stalled")
        break
    last = n
    time.sleep(60)
