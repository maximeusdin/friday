"""Read-only: verify dictionary builds exist for all collections and the live variant channel works."""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

print("=== builds ===")
cur.execute("""
    SELECT b.collection_slug, b.chunk_pv, count(l.lexeme), count(l.skeleton)
    FROM corpus_dictionary_builds b
    LEFT JOIN corpus_dictionary_lexemes l ON l.build_id = b.id
    GROUP BY b.id ORDER BY count(l.lexeme) DESC
""")
rows = cur.fetchall()
tot_lex = 0
for slug, pv, nlex, nskel in rows:
    tot_lex += nlex
    flag = "" if nlex == nskel else "  <-- MISSING SKELETONS"
    print(f"  {slug:32s} {pv:28s} {nlex:>7} lexemes{flag}")
print(f"  TOTAL: {len(rows)} builds, {tot_lex} lexemes")

print("\n=== live channel checks (fetch_fuzzy_variants_for_tokens) ===")
from retrieval.fuzzy_lex import FuzzyLexConfig, fetch_fuzzy_variants_for_tokens

CASES = [
    ("rosenberg", "rosenberg_v1", ["greenglass", "sobell", "bloch"], ["groenglass", "sebell", "blech"]),
    ("solo", "solo_v1_memo", ["chicago", "soviets"], ["chicage", "soviots"]),
    ("hiss_chambers", "hiss_chambers_v1", ["chambers"], ["chambors"]),
    ("judith_coplon", "judith_coplon_v1", ["coplon"], ["ceplon"]),
    ("elizabeth_bentley", "elizabeth_bentley_v1", ["golos"], ["gelos"]),
    ("morris_childs", "morris_childs_v1", ["believed"], ["belioved"]),
]

ok = fail = 0
for slug, pv, tokens, expected in CASES:
    build_id, expansions = fetch_fuzzy_variants_for_tokens(
        conn, tokens=tokens, chunk_pv=pv, collection_slug=slug,
        norm_version="norm_v1", config=FuzzyLexConfig(),
    )
    found = {e["lexeme"] for entries in expansions.values() for e in entries}
    for exp in expected:
        status = "PASS" if exp in found else "FAIL"
        if exp in found:
            ok += 1
        else:
            fail += 1
        print(f"  [{status}] {slug}: query {tokens} -> expects variant '{exp}' (build {build_id})")
    ocr_entries = [(t, e["lexeme"], e.get("cost")) for t, entries in expansions.items()
                   for e in entries if e.get("source") in ("ocr", "both")]
    print(f"         ocr-channel expansions: {ocr_entries}")

print(f"\n{ok} PASS / {fail} FAIL")
