"""Post-load verification for the Bentley deposition transcript (read-only)."""
import os, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== whole-word counts, silvermaster (baseline: fuhr=12, meekirk=3, fuer=1) ===")
for term in ("fuhr", "meekirk", "fuer"):
    cur.execute("""
        SELECT count(DISTINCT c.id)
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE cm.collection_slug = 'silvermaster'
          AND COALESCE(c.clean_text, c.text) ~* ('\\m' || %s || '\\M')
    """, (term,))
    clean_n = cur.fetchone()[0]
    cur.execute("""
        SELECT count(DISTINCT c.id)
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE cm.collection_slug = 'silvermaster'
          AND c.text ~* ('\\m' || %s || '\\M')
    """, (term,))
    raw_n = cur.fetchone()[0]
    print(f"  {term:8s}: clean-layer {clean_n:>3}   raw-layer {raw_n:>3}")

print("\n=== Meekirk chunk 53004 clean_text around the key line ===")
cur.execute("SELECT clean_text FROM chunks WHERE id = 53004")
t = cur.fetchone()[0] or ""
i = t.upper().find("MEEKIRK")
print("..." + t[max(0, i - 120):i + 160].replace("\n", " ") + "...")

print("\n=== document 522 provenance ===")
cur.execute("SELECT metadata -> 'transcript' FROM documents WHERE id = 522")
print(cur.fetchone()[0])

print("\n=== doc 522 chunk embeddings (NULL = awaiting refill) ===")
cur.execute("""
    SELECT count(*) FILTER (WHERE c.embedding IS NULL) AS null_emb, count(*) AS total
    FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE cm.document_id = 522
""")
print(cur.fetchone())

print("\n=== search-tab index: tsv_simple matches on chunk 53004 (fuhr, meekirk) ===")
cur.execute("""
    SELECT tsv_simple @@ to_tsquery('simple', 'fuhr'),
           tsv_simple @@ to_tsquery('simple', 'meekirk')
    FROM chunks WHERE id = 53004
""")
print(cur.fetchone())

print("\n=== union safety: old garble still reachable (fuer via raw / canonical)? ===")
cur.execute("""
    SELECT c.id,
           c.text ~* '\\mfuer\\M' AS raw_has_fuer,
           cec.text_canonical_tsv @@ to_tsquery('simple', 'fuer') AS canonical_has_fuer
    FROM chunks c
    LEFT JOIN chunk_embeddings_canonical cec ON cec.chunk_id = c.id
    WHERE c.id = 53004
""")
print(cur.fetchone())
