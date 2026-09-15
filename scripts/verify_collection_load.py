"""Read-only: verify the full Silvermaster collection load + re-embed + badge state."""
import os
import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== documents with transcript provenance (collection 77) ===")
cur.execute("""
    SELECT metadata->'transcript'->>'status', count(*)
    FROM documents WHERE collection_id = 77 AND metadata ? 'transcript'
    GROUP BY 1
""")
print(" ", cur.fetchall())

print("\n=== chunk embeddings freshness (silvermaster) ===")
cur.execute("""
    SELECT count(*) FILTER (WHERE c.embedding IS NULL) AS null_emb,
           count(*) FILTER (WHERE c.embedded_at >= '2026-08-09') AS fresh,
           count(*) AS total
    FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE cm.collection_slug = 'silvermaster'
""")
print("  null / re-embedded-since-load / total:", cur.fetchone())

print("\n=== name recovery: clean layer vs raw layer (whole-word chunks) ===")
for term in ("ullmann", "perlo", "halperin", "golos", "silvermastor", "bontley", "gelos"):
    cur.execute("""
        SELECT count(DISTINCT c.id) FILTER (WHERE COALESCE(c.clean_text, c.text) ~* ('\\m' || %s || '\\M')),
               count(DISTINCT c.id) FILTER (WHERE c.text ~* ('\\m' || %s || '\\M'))
        FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE cm.collection_slug = 'silvermaster'
    """, (term, term))
    clean_n, raw_n = cur.fetchone()
    print(f"  {term:14s}: clean {clean_n:>4}  raw {raw_n:>4}")

print("\n=== the ROBERT KILLER->MILLER page (doc 507) ===")
cur.execute("""
    SELECT count(*) FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE cm.document_id = 507 AND COALESCE(c.clean_text, c.text) ~* '\\mkiller\\M'
""")
print("  clean-layer 'killer' chunks in 507:", cur.fetchone()[0], "(raw keeps them via union)")
