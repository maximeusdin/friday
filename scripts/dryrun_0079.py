#!/usr/bin/env python3
"""Read-only dry run of migrations/0079_drop_duplicate_ingests.sql.

Runs every guard the migration asserts, as a plain SELECT, and reports exactly what
the migration would delete. Executes no INSERT/UPDATE/DELETE and opens no write
transaction, so it is safe against prod.

Resolves the migration relative to this file, so it can be run from any directory.

Usage:
  DATABASE_URL=... python scripts/dryrun_0079.py
"""
import os
import re
import sys
from pathlib import Path

import psycopg2

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "0079_drop_duplicate_ingests.sql"

PAIRS_RE = re.compile(r"^\s*\(\s*(\d+),\s*(\d+)\)[,;]?\s*--", re.M)


def main():
    sql = MIGRATION.read_text(encoding="utf-8")
    pairs = [(int(a), int(b)) for a, b in PAIRS_RE.findall(sql)]
    drops = [p[0] for p in pairs]
    keeps = [p[1] for p in pairs]
    print(f"parsed {len(pairs)} drop/keep pairs from the migration\n")

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    ok = True

    def check(label, sql_text, params, want):
        nonlocal ok
        cur.execute(sql_text, params)
        got = cur.fetchone()[0]
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: {got} (want {want})")

    print("guards:")
    check("pair count", "SELECT %s::int", (len(pairs),), 26)
    check("targets missing", "SELECT count(*) FROM unnest(%s::bigint[]) x(id) "
          "WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = x.id)", (drops,), 0)
    check("keepers missing", "SELECT count(*) FROM unnest(%s::bigint[]) x(id) "
          "WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = x.id)", (keeps,), 0)
    check("pairs straddling collections",
          "SELECT count(*) FROM unnest(%s::bigint[], %s::bigint[]) p(d, k) "
          "JOIN documents dd ON dd.id = p.d JOIN documents dk ON dk.id = p.k "
          "WHERE dk.collection_id <> dd.collection_id", (drops, keeps), 0)
    check("keepers smaller than their drop",
          "SELECT count(*) FROM unnest(%s::bigint[], %s::bigint[]) p(d, k) "
          "WHERE (SELECT count(*) FROM pages WHERE document_id = p.k) "
          "    < (SELECT count(*) FROM pages WHERE document_id = p.d)", (drops, keeps), 0)
    check("chunk_pages reaching outside the target set",
          "SELECT count(*) FROM chunk_pages cp WHERE cp.chunk_id IN ("
          "  SELECT DISTINCT cp2.chunk_id FROM chunk_pages cp2 JOIN pages p2 ON p2.id = cp2.page_id"
          "  WHERE p2.document_id = ANY(%s)) "
          "AND cp.page_id NOT IN (SELECT id FROM pages WHERE document_id = ANY(%s))",
          (drops, drops), 0)
    check("retrieval_run_chunk_evidence citations (RESTRICT)",
          "SELECT count(*) FROM retrieval_run_chunk_evidence WHERE chunk_id IN ("
          "  SELECT DISTINCT cp.chunk_id FROM chunk_pages cp JOIN pages p ON p.id = cp.page_id"
          "  WHERE p.document_id = ANY(%s))", (drops,), 0)

    print("\nwould delete:")
    for label, q in (
        ("documents", "SELECT count(*) FROM documents WHERE id = ANY(%s)"),
        ("pages", "SELECT count(*) FROM pages WHERE document_id = ANY(%s)"),
        ("chunks", "SELECT count(DISTINCT cp.chunk_id) FROM chunk_pages cp "
                   "JOIN pages p ON p.id = cp.page_id WHERE p.document_id = ANY(%s)"),
        ("search_result_page_hits", "SELECT count(*) FROM search_result_page_hits WHERE document_id = ANY(%s)"),
        ("evidence_items", "SELECT count(*) FROM evidence_items WHERE chunk_id IN ("
                           "  SELECT DISTINCT cp.chunk_id FROM chunk_pages cp JOIN pages p ON p.id = cp.page_id"
                           "  WHERE p.document_id = ANY(%s))"),
    ):
        cur.execute(q, (drops,))
        print(f"  {label:26} {cur.fetchone()[0]}")

    print("\nper collection, before -> after:")
    cur.execute("""
        SELECT c.slug, count(*) AS docs,
               count(*) FILTER (WHERE d.id = ANY(%s)) AS dropped
        FROM documents d JOIN collections c ON c.id = d.collection_id
        GROUP BY c.slug HAVING count(*) FILTER (WHERE d.id = ANY(%s)) > 0
        ORDER BY 3 DESC
    """, (drops, drops))
    for slug, docs, dropped in cur.fetchall():
        print(f"  {slug:14} {docs:4} -> {docs - dropped:4} documents  (-{dropped})")

    print(f"\n{'ALL GUARDS PASS' if ok else 'GUARDS FAILED - DO NOT RUN THE MIGRATION'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
