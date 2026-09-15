#!/usr/bin/env python3
"""Delete every DB row belonging to one or more collections, in FK order.

Chunks carry no document_id of their own, so a collection's chunk set is
resolved through chunk_metadata.document_id *and* chunk_pages -> pages
(catching any chunk whose metadata row is missing).

Dry run by default; --execute commits. One transaction per invocation, so a
failure part-way leaves prod untouched.

Usage:  DATABASE_URL=<dsn> python scripts/delete_collection_rows.py [--execute] slug [slug ...]
"""
import argparse
import os
import sys

import psycopg2

# (table, column, which id set) in delete order — children before parents.
CHUNK_CHILDREN = [
    "chunk_embeddings_canonical", "chunk_ner_runs", "chunk_turns", "date_mentions",
    "entity_mentions", "focus_spans", "mention_candidates", "mention_review_queue",
    "ner_date_signals", "ner_surface_mentions", "result_set_chunks",
    "result_set_match_traces", "retrieval_run_chunk_evidence", "evidence_items",
    "v3_evidence_spans", "retrieval_chunks_v1",
]
DOC_CHILDREN = [
    "alias_referent_rules", "alias_stats", "document_anchors", "document_witnesses",
    "page_entity_mentions", "transcript_turns", "ocr_extraction_runs",
]


def is_base_table(cur, table):
    cur.execute("SELECT table_type FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name=%s", (table,))
    row = cur.fetchone()
    return bool(row) and row[0] == "BASE TABLE"


def has_column(cur, table, column):
    cur.execute("SELECT 1 FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=%s AND column_name=%s",
                (table, column))
    return cur.fetchone() is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--execute", action="store_true", help="commit (default: rollback)")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        sys.exit("Missing DATABASE_URL")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute("SELECT id, slug, title FROM collections WHERE slug = ANY(%s) ORDER BY id",
                (args.slugs,))
    colls = cur.fetchall()
    found = {c[1] for c in colls}
    for s in args.slugs:
        if s not in found:
            print(f"  {s}: not in this database — nothing to do")
    if not colls:
        return 0
    cids = [c[0] for c in colls]
    for cid, slug, title in colls:
        print(f"  {slug} (id={cid}): {title}")

    cur.execute("SELECT id FROM documents WHERE collection_id = ANY(%s)", (cids,))
    dids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT id FROM pages WHERE document_id = ANY(%s)", (dids,))
    pids = [r[0] for r in cur.fetchall()]
    cur.execute("""
        SELECT chunk_id FROM chunk_metadata WHERE document_id = ANY(%s)
        UNION
        SELECT cp.chunk_id FROM chunk_pages cp
          JOIN pages p ON p.id = cp.page_id WHERE p.document_id = ANY(%s)
    """, (dids, dids))
    chids = [r[0] for r in cur.fetchall()]
    print(f"\n  scope: {len(dids)} documents, {len(pids)} pages, {len(chids)} chunks")

    plan = []
    # Rows that merely *reference* a chunk/doc/page (search history, NER, eval).
    for t in CHUNK_CHILDREN:
        plan.append((t, "chunk_id", chids))
    plan += [
        ("entity_resolution_reviews", "context_chunk_id", chids),
        ("search_result_page_hits", "chunk_id", chids),
        ("search_result_page_hits", "document_id", dids),
        ("search_result_page_hits", "page_id", pids),
        ("search_result_page_hits", "collection_id", cids),
        ("entity_alias_overrides", "scope_document_id", dids),
        ("entity_alias_overrides", "scope_collection_id", cids),
    ]
    # The collection's own rows, children first.
    plan += [("chunk_pages", "chunk_id", chids),
             ("chunk_metadata", "chunk_id", chids),
             ("chunks", "id", chids)]
    for t in DOC_CHILDREN:
        plan.append((t, "document_id", dids))
    plan += [("page_metadata", "page_id", pids),
             ("pages", "id", pids),
             ("documents", "id", dids),
             ("collections", "id", cids)]

    print()
    total = 0
    for table, column, ids in plan:
        if not ids:
            continue
        if not is_base_table(cur, table) or not has_column(cur, table, column):
            continue
        cur.execute(f"DELETE FROM {table} WHERE {column} = ANY(%s)", (ids,))
        if cur.rowcount:
            print(f"  deleted {cur.rowcount:>6} from {table} ({column})")
            total += cur.rowcount

    # Slug-keyed leftovers with no FK to the rows above.
    for table in ["corpus_dictionary_builds", "entity_citations", "retrieval_evaluations",
                  "alias_referent_rules", "ocr_extraction_runs", "page_entity_mentions",
                  "chunk_metadata", "retrieval_chunks_v1"]:
        if not is_base_table(cur, table) or not has_column(cur, table, "collection_slug"):
            continue
        cur.execute(f"DELETE FROM {table} WHERE collection_slug = ANY(%s)", (args.slugs,))
        if cur.rowcount:
            print(f"  deleted {cur.rowcount:>6} from {table} (collection_slug)")
            total += cur.rowcount

    print(f"\n  {total} rows total")
    if args.execute:
        conn.commit()
        print("  COMMITTED")
    else:
        conn.rollback()
        print("  rolled back (dry run) — pass --execute to commit")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
