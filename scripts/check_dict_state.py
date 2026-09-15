"""Read-only: corpus dictionary build state in prod, and whether FUER survives as a lexeme."""
import os, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== corpus_dictionary_builds ===")
cur.execute("""
    SELECT b.id, b.chunk_pv, b.collection_slug, b.norm_version, b.built_at::date,
           (SELECT count(*) FROM corpus_dictionary_lexemes l WHERE l.build_id = b.id)
    FROM corpus_dictionary_builds b ORDER BY b.id
""")
for r in cur.fetchall():
    print(r)

print("\n=== lexeme columns ===")
cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'corpus_dictionary_lexemes' ORDER BY ordinal_position
""")
print([r[0] for r in cur.fetchall()])

print("\n=== fuer/fuhr lexemes in latest silvermaster build (if any) ===")
cur.execute("""
    SELECT l.build_id, l.lexeme, l.chunk_freq
    FROM corpus_dictionary_lexemes l
    JOIN corpus_dictionary_builds b ON b.id = l.build_id
    WHERE b.collection_slug = 'silvermaster' AND l.lexeme IN ('fuer','fuhr','meekirk','moekirk')
    ORDER BY l.build_id, l.lexeme
""")
for r in cur.fetchall():
    print(r)
