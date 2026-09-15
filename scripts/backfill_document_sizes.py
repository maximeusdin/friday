#!/usr/bin/env python3
"""Backfill documents.size_bytes from the S3 PDF store.

Adds the size_bytes column if absent, lists s3://<bucket>/data/raw/ once, maps
every documents row to its S3 object (via scripts.s3_doc_mapping), and writes
the object size. Safe to re-run; only touches size_bytes.

Run (Mac):
  AWS_SHARED_CREDENTIALS_FILE=/Users/maxime/friday/.aws/credentials \
  DATABASE_URL=<dsn> python scripts/backfill_document_sizes.py [--dry-run]
"""
import argparse
import os
import sys

import boto3
import psycopg2

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s3_doc_mapping import (
    build_key_indexes,
    demote_basename_conflicts,
    map_collection_documents,
)


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def list_s3_objects(bucket: str, prefix: str, region: str) -> dict:
    s3 = boto3.client("s3", region_name=region)
    objects = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = obj["Size"]
    return objects


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="fridayarchive.org")
    ap.add_argument("--region", default="us-west-1")
    ap.add_argument("--prefix", default="data/raw/")
    ap.add_argument("--only-missing", action="store_true",
                    help="Skip documents that already have size_bytes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Listing s3://{args.bucket}/{args.prefix} ...")
    s3_objects = list_s3_objects(args.bucket, args.prefix, args.region)
    print(f"  {len(s3_objects)} objects, {sum(s3_objects.values()) / 1e9:.2f} GB")
    keys, by_basename = build_key_indexes(s3_objects)

    conn = get_conn()
    cur = conn.cursor()
    if not args.dry_run:
        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS size_bytes BIGINT")
        conn.commit()

    where = "WHERE d.size_bytes IS NULL" if args.only_missing else ""
    try:
        cur.execute(f"""
            SELECT c.slug, d.id, d.source_name, d.source_ref
            FROM documents d JOIN collections c ON c.id = d.collection_id
            {where} ORDER BY c.slug, d.source_name
        """)
    except psycopg2.errors.UndefinedColumn:
        if not args.dry_run:
            raise
        conn.rollback()
        cur.execute("""
            SELECT c.slug, d.id, d.source_name, d.source_ref
            FROM documents d JOIN collections c ON c.id = d.collection_id
            ORDER BY c.slug, d.source_name
        """)
    rows = cur.fetchall()

    by_collection = {}
    for slug, doc_id, source_name, source_ref in rows:
        by_collection.setdefault(slug, []).append((doc_id, source_name, source_ref))

    matched_by_slug = {}
    all_matched = []
    all_unmatched = []
    for slug in sorted(by_collection):
        matched, unmatched = map_collection_documents(
            by_collection[slug], slug, s3_objects, keys, by_basename)
        matched_by_slug[slug] = matched
        all_matched.extend(matched)
        all_unmatched.extend((slug, d, n) for d, n in unmatched)

    for m in demote_basename_conflicts(all_matched):
        print(f"WARNING: {m['slug']} doc={m['doc_id']} {m['source_name']!r} basename-matched "
              f"{m['key']} but another document claims that object — skipping")
        matched_by_slug[m["slug"]].remove(m)
        all_unmatched.append((m["slug"], m["doc_id"], m["source_name"]))

    total_matched = total_unmatched = 0
    unmatched_by_slug = {}
    for slug, d, n in all_unmatched:
        unmatched_by_slug.setdefault(slug, []).append((d, n))
    for slug in sorted(by_collection):
        matched = matched_by_slug[slug]
        unmatched = unmatched_by_slug.get(slug, [])
        total_matched += len(matched)
        total_unmatched += len(unmatched)
        if not args.dry_run:
            for m in matched:
                cur.execute("UPDATE documents SET size_bytes=%s WHERE id=%s",
                            (m["size"], m["doc_id"]))
            conn.commit()
        print(f"  {slug:<40} {len(matched):>4} matched"
              + (f", {len(unmatched)} UNMATCHED" if unmatched else ""))

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}matched {total_matched}, unmatched {total_unmatched}")
    for slug, doc_id, name in all_unmatched:
        print(f"  UNMATCHED {slug} doc={doc_id} {name!r}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
