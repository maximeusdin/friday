#!/usr/bin/env python3
"""Pre-flight for dropping documents: count every dependent row and flag blockers.

Discovers dependants by scanning information_schema for document_id / page_id /
chunk_id columns rather than trusting a hand-written list, then classifies each by
its FK delete rule:

  CASCADE   cleaned up automatically when the parent goes
  RESTRICT  blocks the delete unless emptied first
  NO FK     will be ORPHANED silently -- must be deleted explicitly

search_result_page_hits is the important NO-FK case: it stores document_id, page_id
and chunk_id but only has a foreign key to search_result_sets, so a document delete
leaves its rows pointing at ids that no longer exist.

Read-only -- runs counts only, never a delete. Usage:
  DATABASE_URL=... python scripts/preflight_drop_documents.py 188 1173 57 80 ...
  DATABASE_URL=... python scripts/preflight_drop_documents.py --file docids.txt
"""
import argparse
import os
import sys
from collections import defaultdict

import psycopg2

KEYS = ("document_id", "page_id", "chunk_id", "doc_id", "first_page_id",
        "last_page_id", "context_chunk_id", "related_document_id", "scope_document_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="*", type=int)
    ap.add_argument("--file")
    a = ap.parse_args()
    docs = list(a.docs)
    if a.file:
        docs += [int(x) for x in open(a.file).read().split()]
    if not docs:
        sys.exit("give document ids")

    conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"))
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '1800s'")

    cur.execute("""
        SELECT c.table_name, c.column_name
        FROM information_schema.columns c
        JOIN information_schema.tables t
          ON t.table_name = c.table_name AND t.table_schema = c.table_schema
        WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'
          AND c.column_name = ANY(%s)
        ORDER BY 1, 2
    """, (list(KEYS),))
    cols = cur.fetchall()

    cur.execute("""
        SELECT src.relname, a.attname, tgt.relname, c.confdeltype
        FROM pg_constraint c
        JOIN pg_class src ON src.oid = c.conrelid
        JOIN pg_class tgt ON tgt.oid = c.confrelid
        JOIN unnest(c.conkey) WITH ORDINALITY k(attnum, ord) ON true
        JOIN pg_attribute a ON a.attrelid = src.oid AND a.attnum = k.attnum
        WHERE c.contype = 'f'
    """)
    rule = {(t, col): (parent, {'c': 'CASCADE', 'n': 'SET NULL', 'r': 'RESTRICT',
                               'a': 'NO ACTION', 'd': 'SET DEFAULT'}[d])
            for t, col, parent, d in cur.fetchall()}

    # Scope sets.
    cur.execute("SELECT id, collection_id, source_name FROM documents WHERE id = ANY(%s)", (docs,))
    found = cur.fetchall()
    print(f"targets: {len(found)}/{len(docs)} document ids exist")
    missing = set(docs) - {r[0] for r in found}
    if missing:
        print(f"  !! not in the database: {sorted(missing)}")

    cur.execute("SELECT count(*) FROM pages WHERE document_id = ANY(%s)", (docs,))
    n_pages = cur.fetchone()[0]
    cur.execute("""SELECT count(DISTINCT cp.chunk_id) FROM chunk_pages cp
                   JOIN pages p ON p.id = cp.page_id WHERE p.document_id = ANY(%s)""", (docs,))
    n_chunks = cur.fetchone()[0]
    print(f"         {n_pages} pages, {n_chunks} chunks\n")

    # Chunks that also cover a page outside the target set would be damaged by the delete.
    cur.execute("""
        SELECT count(*) FROM chunk_pages cp
        WHERE cp.chunk_id IN (SELECT DISTINCT cp2.chunk_id FROM chunk_pages cp2
                              JOIN pages p2 ON p2.id = cp2.page_id
                              WHERE p2.document_id = ANY(%s))
          AND cp.page_id NOT IN (SELECT id FROM pages WHERE document_id = ANY(%s))
    """, (docs, docs))
    leak = cur.fetchone()[0]
    print(f"chunk_pages rows reaching outside the target documents: {leak}"
          f"{'   <-- MUST BE 0' if leak else '  (clean)'}\n")

    print(f"{'table':34} {'column':20} {'rows':>8}  delete rule")
    print("-" * 82)
    blockers, orphans = [], []
    for table, col in cols:
        if table in ("documents", "pages", "chunks"):
            continue
        if col in ("document_id", "doc_id", "related_document_id", "scope_document_id"):
            pred, params = f"{col} = ANY(%s)", (docs,)
        elif col in ("page_id", "first_page_id", "last_page_id"):
            pred, params = (f"{col} IN (SELECT id FROM pages WHERE document_id = ANY(%s))", (docs,))
        else:
            pred, params = (f"{col} IN (SELECT DISTINCT cp.chunk_id FROM chunk_pages cp "
                            f"JOIN pages p ON p.id = cp.page_id WHERE p.document_id = ANY(%s))", (docs,))
        try:
            cur.execute(f"SELECT count(*) FROM {table} WHERE {pred}", params)
            n = cur.fetchone()[0]
        except psycopg2.Error as e:
            conn.rollback()
            print(f"{table:34} {col:20} {'ERR':>8}  {str(e).splitlines()[0][:40]}")
            continue
        parent, r = rule.get((table, col), (None, "NO FK"))
        mark = ""
        if n and r in ("RESTRICT", "NO ACTION"):
            blockers.append((table, col, n)); mark = "  <-- BLOCKS"
        elif n and r == "NO FK":
            orphans.append((table, col, n)); mark = "  <-- ORPHANS, delete explicitly"
        if n:
            print(f"{table:34} {col:20} {n:8}  {r}{mark}")
    print("-" * 82)
    print("\nblocking (RESTRICT / NO ACTION) dependants:")
    for t, c, n in blockers:
        print(f"  {t}.{c}: {n} rows")
    if not blockers:
        print("  none")
    print("\nno-FK dependants that would be orphaned:")
    for t, c, n in orphans:
        print(f"  {t}.{c}: {n} rows")
    if not orphans:
        print("  none")


if __name__ == "__main__":
    main()
