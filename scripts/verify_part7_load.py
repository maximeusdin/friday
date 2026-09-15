"""Read-only verification of the Part 7 transcript load + re-embed (doc 532)."""
import os, sys
import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== embeddings (doc 532) ===")
cur.execute("""
    SELECT count(*) FILTER (WHERE c.embedding IS NULL),
           count(*) FILTER (WHERE c.embedded_at::date = current_date),
           count(*)
    FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE cm.document_id = 532
""")
print("  null / embedded-today / total:", cur.fetchone())

print("\n=== the recovered blank page (chunk 53862, was 4 chars) ===")
cur.execute("SELECT length(clean_text), left(clean_text, 200) FROM chunks WHERE id = 53862")
n, t = cur.fetchone()
print(f"  {n} chars now: {t[:180]!r}")

print("\n=== union check: names in clean vs garble in raw (doc 532) ===")
for term in ("greenberg", "golos", "moscow"):
    cur.execute("""
        SELECT count(*) FILTER (WHERE COALESCE(c.clean_text, c.text) ~* ('\\m' || %s || '\\M')),
               count(*) FILTER (WHERE c.text ~* ('\\m' || %s || '\\M'))
        FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE cm.document_id = 532
    """, (term, term))
    clean_n, raw_n = cur.fetchone()
    print(f"  {term:10s}: clean-layer {clean_n:>3} chunks, raw-layer {raw_n:>3}")

print("\n=== provenance ===")
cur.execute("SELECT metadata -> 'transcript' FROM documents WHERE id = 532")
print(" ", cur.fetchone()[0])

print("\n=== search-tab index sanity: tsv_simple matches greenberg in doc 532? ===")
cur.execute("""
    SELECT count(*) FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE cm.document_id = 532 AND c.tsv_simple @@ to_tsquery('simple', 'greenberg')
""")
print("  chunks:", cur.fetchone()[0])
