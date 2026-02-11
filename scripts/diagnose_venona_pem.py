#!/usr/bin/env python3
"""
Diagnose why Venona PEM mappings fail.

Compares:
- Venona documents in DB (source_name, volume) → normalized keys
- Venona document titles in CSV → normalized keys
- Page resolution for a sample Venona row

Usage:
    python scripts/diagnose_venona_pem.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.validate_entity_mentions_from_citations import (
    normalize_document_name,
    build_citation_to_document_map,
)

try:
    from retrieval.ops import get_conn
except ImportError:
    get_conn = None


def _get_conn():
    import psycopg2
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "neh"),
        user=os.getenv("DB_USER", "neh"),
        password=os.getenv("DB_PASS", "neh"),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-db", action="store_true", help="Skip DB; only show CSV + normalization")
    args = parser.parse_args()

    csv_path = REPO_ROOT / "concordance_export" / "entry_document_pages.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    print("=" * 70)
    print("Venona PEM Diagnostic")
    print("=" * 70)

    # CSV-only: show what titles we expect and their norms
    csv_venona_titles = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row.get("document_title", "")
            if dt.lower().startswith("venona"):
                csv_venona_titles.add(dt)

    print(f"\nCSV Venona document titles ({len(csv_venona_titles)} unique):")
    for title in sorted(csv_venona_titles):
        norm = normalize_document_name(title)
        print(f"   {title!r} -> {norm!r}")

    if args.no_db:
        print("\n(Use without --no-db to compare with DB)")
        return

    conn = _get_conn()
    with conn.cursor() as cur:
        # 1. DB: Venona documents (source_name, volume)
        cur.execute("""
            SELECT d.id, d.source_name, d.volume
            FROM documents d
            JOIN collections c ON c.id = d.collection_id
            WHERE c.slug = 'venona'
            ORDER BY d.source_name
        """)
        db_docs = cur.fetchall()
        print(f"\n1. Venona documents in DB: {len(db_docs)}")
        for doc_id, source_name, volume in db_docs[:15]:
            norm = normalize_document_name(source_name)
            norm_vol = normalize_document_name(f"{source_name} {volume}") if volume else "(n/a)"
            print(f"   id={doc_id} source_name={source_name!r} volume={volume!r}")
            print(f"      -> norm: {norm!r}  (with vol: {norm_vol!r})")
        if len(db_docs) > 15:
            print(f"   ... and {len(db_docs) - 15} more")

        # 2. Doc map keys
        doc_map = build_citation_to_document_map(cur, "venona")
        print(f"\n2. Doc map keys (normalized): {len(doc_map)}")
        for i, key in enumerate(sorted(doc_map.keys())[:20]):
            print(f"   {key!r}")
        if len(doc_map) > 20:
            print(f"   ... and {len(doc_map) - 20} more")

        # 3. CSV vs doc_map
        print(f"\n3. CSV Venona titles vs doc_map ({len(csv_venona_titles)} unique in CSV):")
        mismatches = []
        for title in sorted(csv_venona_titles)[:20]:
            norm = normalize_document_name(title)
            in_map = norm in doc_map
            status = "OK" if in_map else "MISSING"
            print(f"   {title!r} -> {norm!r} [{status}]")
            if not in_map:
                mismatches.append((title, norm))
        if len(csv_venona_titles) > 20:
            for title in sorted(csv_venona_titles)[20:]:
                norm = normalize_document_name(title)
                if norm not in doc_map:
                    mismatches.append((title, norm))

        # 4. Sample: resolve one CSV row
        print("\n4. Sample resolution: first Venona row")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("document_title", "").lower().startswith("venona"):
                    break
        doc_title = row["document_title"]
        pages_str = row["pages"]
        page_nums = [int(p.strip()) for p in pages_str.split(",") if p.strip()]
        norm_doc = normalize_document_name(doc_title)
        matches = doc_map.get(norm_doc, [])
        print(f"   document_title: {doc_title!r}")
        print(f"   pages: {pages_str}")
        print(f"   norm: {norm_doc!r}")
        print(f"   doc_map match: {bool(matches)}")
        if matches:
            doc_id, doc_name = matches[0]
            print(f"   doc_id={doc_id} doc_name={doc_name!r}")
            # Check page resolution
            cur.execute("""
                SELECT p.id, p.pdf_page_number, p.logical_page_label
                FROM pages p
                WHERE p.document_id = %s
                ORDER BY p.pdf_page_number
            """, (doc_id,))
            pages = cur.fetchall()
            print(f"   Pages in doc: {len(pages)} (first 5: {[(p[1], p[2]) for p in pages[:5]]})")
            # Span map for first page_num
            if page_nums:
                pg = page_nums[0]
                cur.execute(
                    """
                    SELECT p.id, p.pdf_page_number
                    FROM pages p
                    WHERE p.document_id = %s
                    ORDER BY p.pdf_page_number
                    """,
                    (doc_id,),
                )
                entries = cur.fetchall()
                found = None
                for i, (pid, start) in enumerate(entries):
                    end = entries[i + 1][1] - 1 if i + 1 < len(entries) else start + 10000
                    if start <= pg <= end:
                        found = (pid, start, end)
                        break
                print(f"   Page {pg} in span? {found}")
        else:
            print("   CANNOT RESOLVE DOCUMENT - no doc_map match")

        # 5. Summary
        print("\n5. Summary")
        print(f"   CSV Venona titles missing from doc_map: {len(mismatches)}")
        for title, norm in mismatches[:10]:
            print(f"      {title!r} -> {norm!r}")

    conn.close()


if __name__ == "__main__":
    main()
