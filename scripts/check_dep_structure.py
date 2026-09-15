"""Pre-load checks for the Bentley pilot: chunk<->page mapping for doc 522, alias column, baseline counts."""
import os, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== entity_aliases columns ===")
cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'entity_aliases' ORDER BY ordinal_position
""")
print([r[0] for r in cur.fetchall()])

print("\n=== doc 522 chunk structure ===")
cur.execute("""
    SELECT count(DISTINCT c.id) AS chunks,
           count(DISTINCT cp.page_id) AS pages_with_chunks,
           max(pp.pages_per_chunk) AS max_pages_per_chunk
    FROM chunks c
    JOIN chunk_pages cp ON cp.chunk_id = c.id
    JOIN pages p ON p.id = cp.page_id
    JOIN (SELECT chunk_id, count(*) AS pages_per_chunk FROM chunk_pages GROUP BY chunk_id) pp
      ON pp.chunk_id = c.id
    WHERE p.document_id = 522
""")
print(cur.fetchone())

cur.execute("""
    SELECT c.pipeline_version, count(*), count(c.embedding), count(c.clean_text)
    FROM chunks c JOIN chunk_pages cp ON cp.chunk_id = c.id
    JOIN pages p ON p.id = cp.page_id
    WHERE p.document_id = 522 GROUP BY 1
""")
print("pipeline_version, n_chunks, n_embedded, n_clean:", cur.fetchall())

print("\n=== baseline lexical counts (silvermaster collection) ===")
for term in ("fuhr", "meekirk", "fuer"):
    cur.execute("""
        SELECT count(DISTINCT c.id)
        FROM chunks c
        JOIN chunk_pages cp ON cp.chunk_id = c.id
        JOIN pages p ON p.id = cp.page_id
        JOIN documents d ON d.id = p.document_id
        WHERE d.collection_id = 77
          AND COALESCE(c.clean_text, c.text) ~* ('\\m' || %s || '\\M')
    """, (term,))
    print(f"whole-word '{term}': {cur.fetchone()[0]} chunks")

print("\n=== doc 522 pages 3-120: chunks whose pages all lie in span ===")
cur.execute("""
    WITH doc_chunks AS (
        SELECT cp.chunk_id, bool_and(p.page_seq BETWEEN 3 AND 120) AS all_in_span
        FROM chunk_pages cp JOIN pages p ON p.id = cp.page_id
        WHERE p.document_id = 522
        GROUP BY cp.chunk_id
    )
    SELECT all_in_span, count(*) FROM doc_chunks GROUP BY 1
""")
print(cur.fetchall())
