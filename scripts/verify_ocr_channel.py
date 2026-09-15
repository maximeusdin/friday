"""Read-only verification of the OCR-variant channel against prod after migration + dictionary build."""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== dictionary builds ===")
cur.execute("""
    SELECT b.id, b.chunk_pv, b.collection_slug, b.norm_version, b.built_at::date,
           (SELECT count(*) FROM corpus_dictionary_lexemes l WHERE l.build_id = b.id),
           (SELECT count(*) FROM corpus_dictionary_lexemes l WHERE l.build_id = b.id AND l.skeleton IS NOT NULL)
    FROM corpus_dictionary_builds b ORDER BY b.id
""")
for r in cur.fetchall():
    print(f"  build {r[0]}: {r[1]} / {r[2]} / {r[3]} built {r[4]} — {r[5]} lexemes, {r[6]} with skeletons")

print("\n=== fuer/fuhr lexemes ===")
cur.execute("""
    SELECT l.lexeme, l.chunk_freq, l.skeleton
    FROM corpus_dictionary_lexemes l
    JOIN corpus_dictionary_builds b ON b.id = l.build_id
    WHERE b.collection_slug = 'silvermaster' AND l.lexeme IN ('fuer','fuhr','silvermastor','bentloy','gelos')
    ORDER BY l.lexeme
""")
for r in cur.fetchall():
    print(f"  {r[0]:14s} freq {r[1]:>4}  skeleton {r[2]}")

print("\n=== live channel: fetch_fuzzy_variants_for_tokens('fuhr', 'golos', 'silvermaster') ===")
from retrieval.fuzzy_lex import FuzzyLexConfig, fetch_fuzzy_variants_for_tokens

build_id, expansions = fetch_fuzzy_variants_for_tokens(
    conn,
    tokens=["fuhr", "golos", "silvermaster"],
    chunk_pv="chunk_v1_silvermaster_structured_4k",
    collection_slug="silvermaster",
    norm_version="norm_v1",
    config=FuzzyLexConfig(),
)
print(f"  build_id={build_id}")
for tok, entries in expansions.items():
    for e in entries:
        cost = f" cost {e['cost']:.2f}" if "cost" in e else ""
        print(f"  {tok} -> {e['lexeme']} [{e.get('source')}]{cost} freq {e.get('chunk_freq')}")
