#!/usr/bin/env python3
"""Load a witness index JSON (from scripts/build_witness_index.py) into the
document_witnesses table for a given document.

Resolve the document by --document-id, or by --collection-slug (uses the single
document in that collection).

Usage:
    python -m scripts.load_witness_index \
        --collection-slug brothman_moskowitz_grand_jury \
        --index-json ocr_cache/brothman_witness_index.json
"""
import os
import json
import argparse

import psycopg2


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def resolve_document_id(cur, document_id, collection_slug):
    if document_id:
        return document_id
    cur.execute(
        """SELECT d.id FROM documents d JOIN collections c ON c.id = d.collection_id
           WHERE c.slug = %s ORDER BY d.id""",
        (collection_slug,),
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"No document found for collection slug {collection_slug!r}")
    if len(rows) > 1:
        raise SystemExit(f"Collection {collection_slug!r} has {len(rows)} documents; pass --document-id")
    return rows[0][0]


def main():
    ap = argparse.ArgumentParser(description="Load witness index into document_witnesses")
    ap.add_argument("--index-json", required=True)
    ap.add_argument("--document-id", type=int, default=None)
    ap.add_argument("--collection-slug", default=None)
    args = ap.parse_args()

    if not (args.document_id or args.collection_slug):
        raise SystemExit("Pass --document-id or --collection-slug")

    index = json.load(open(args.index_json, encoding="utf-8"))
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            doc_id = resolve_document_id(cur, args.document_id, args.collection_slug)
            cur.execute("DELETE FROM document_witnesses WHERE document_id = %s", (doc_id,))
            for seq, ev in enumerate(index, start=1):
                cur.execute(
                    """INSERT INTO document_witnesses
                       (document_id, appearance_seq, witness_name, start_page, end_page,
                        page_count, testimony_date, examiner)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (doc_id, seq, ev["witness"], ev["start_page"], ev["end_page"],
                     ev.get("page_count"), ev.get("date"), ev.get("examiner")),
                )
        conn.commit()
        print(f"Loaded {len(index)} witness appearances for document_id={doc_id}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
