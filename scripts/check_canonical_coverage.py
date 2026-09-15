"""Check chunk_embeddings_canonical coverage for doc 522 chunks (searchbox union question)."""
import os, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'chunk_embeddings_canonical' ORDER BY ordinal_position
""")
print("cec columns:", [r[0] for r in cur.fetchall()])

cur.execute("""
    SELECT count(*), array_agg(DISTINCT cec.embedding_model), array_agg(DISTINCT cec.pipeline_version)
    FROM chunk_embeddings_canonical cec
    WHERE cec.chunk_id IN (
        SELECT cm.chunk_id FROM chunk_metadata cm WHERE cm.document_id = 522
    )
""")
print("cec rows for doc 522 chunks:", cur.fetchone())

cur.execute("SELECT count(DISTINCT chunk_id) FROM chunk_metadata WHERE document_id = 522")
print("doc 522 chunks in chunk_metadata:", cur.fetchone()[0])

cur.execute("""
    SELECT cec.chunk_id, left(cec.text_canonical, 120)
    FROM chunk_embeddings_canonical cec
    WHERE cec.chunk_id = 53004
""")
print("canonical text of Meekirk chunk 53004:", cur.fetchall())
