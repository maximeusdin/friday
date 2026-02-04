#!/usr/bin/env python3
"""
compare_db_data.py

Compare data between two PostgreSQL databases to verify that documents and chunks
have been successfully migrated/ingested.

Compares:
- Row counts in key tables (collections, documents, pages, chunks, chunk_pages)
- Collection-level document and page counts
- Optional: sample IDs to verify data presence

Usage examples:
  # Compare local DB vs AWS RDS
  python scripts/compare_db_data.py \
    --local "postgresql://neh:neh@localhost:5432/neh" \
    --remote "postgresql://friday:PASS@HOST:5432/friday?sslmode=verify-full&sslrootcert=..."

  # Using environment variables (set DATABASE_URL_LOCAL and DATABASE_URL_REMOTE)
  python scripts/compare_db_data.py

  # Verbose output with collection breakdown
  python scripts/compare_db_data.py --verbose

  # JSON output
  python scripts/compare_db_data.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import psycopg2


@dataclass
class TableCounts:
    collections: int = 0
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    chunk_pages: int = 0


@dataclass
class CollectionStats:
    slug: str
    title: str
    doc_count: int
    page_count: int
    chunk_count: int


@dataclass
class DatabaseSnapshot:
    name: str
    counts: TableCounts
    collections: List[CollectionStats] = field(default_factory=list)
    sample_doc_ids: List[int] = field(default_factory=list)
    sample_chunk_ids: List[int] = field(default_factory=list)


def get_table_counts(conn) -> TableCounts:
    """Get row counts for key tables."""
    counts = TableCounts()
    with conn.cursor() as cur:
        # Check if tables exist before counting
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('collections', 'documents', 'pages', 'chunks', 'chunk_pages')
        """)
        existing_tables = {row[0] for row in cur.fetchall()}
        
        if 'collections' in existing_tables:
            cur.execute("SELECT COUNT(*) FROM collections")
            counts.collections = cur.fetchone()[0]
        
        if 'documents' in existing_tables:
            cur.execute("SELECT COUNT(*) FROM documents")
            counts.documents = cur.fetchone()[0]
        
        if 'pages' in existing_tables:
            cur.execute("SELECT COUNT(*) FROM pages")
            counts.pages = cur.fetchone()[0]
        
        if 'chunks' in existing_tables:
            cur.execute("SELECT COUNT(*) FROM chunks")
            counts.chunks = cur.fetchone()[0]
        
        if 'chunk_pages' in existing_tables:
            cur.execute("SELECT COUNT(*) FROM chunk_pages")
            counts.chunk_pages = cur.fetchone()[0]
    
    return counts


def get_collection_stats(conn) -> List[CollectionStats]:
    """Get per-collection statistics."""
    stats = []
    with conn.cursor() as cur:
        # Check if required tables exist
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('collections', 'documents', 'pages', 'chunks', 'chunk_pages')
        """)
        existing_tables = {row[0] for row in cur.fetchall()}
        
        required = {'collections', 'documents', 'pages', 'chunks', 'chunk_pages'}
        if not required.issubset(existing_tables):
            return stats
        
        cur.execute("""
            SELECT 
                c.slug,
                c.title,
                COUNT(DISTINCT d.id) AS doc_count,
                COUNT(DISTINCT p.id) AS page_count,
                COUNT(DISTINCT ch.id) AS chunk_count
            FROM collections c
            LEFT JOIN documents d ON d.collection_id = c.id
            LEFT JOIN pages p ON p.document_id = d.id
            LEFT JOIN chunk_pages cp ON cp.page_id = p.id
            LEFT JOIN chunks ch ON ch.id = cp.chunk_id
            GROUP BY c.id, c.slug, c.title
            ORDER BY c.slug
        """)
        for row in cur.fetchall():
            stats.append(CollectionStats(
                slug=row[0],
                title=row[1],
                doc_count=row[2],
                page_count=row[3],
                chunk_count=row[4],
            ))
    return stats


def get_sample_ids(conn, table: str, limit: int = 100) -> List[int]:
    """Get sample IDs from a table for verification."""
    ids = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = %s
        """, (table,))
        if not cur.fetchone():
            return ids
        
        # Get a mix: first N/3, last N/3, random N/3
        cur.execute(f"SELECT id FROM {table} ORDER BY id ASC LIMIT %s", (limit // 3,))
        ids.extend([row[0] for row in cur.fetchall()])
        
        cur.execute(f"SELECT id FROM {table} ORDER BY id DESC LIMIT %s", (limit // 3,))
        ids.extend([row[0] for row in cur.fetchall()])
        
        cur.execute(f"""
            SELECT id FROM {table} 
            WHERE id NOT IN (
                SELECT id FROM {table} ORDER BY id ASC LIMIT %s
                UNION
                SELECT id FROM {table} ORDER BY id DESC LIMIT %s
            )
            ORDER BY RANDOM() LIMIT %s
        """, (limit // 3, limit // 3, limit // 3))
        ids.extend([row[0] for row in cur.fetchall()])
    
    return sorted(set(ids))


def snapshot_database(conn, name: str, include_samples: bool = False) -> DatabaseSnapshot:
    """Create a snapshot of the database state."""
    counts = get_table_counts(conn)
    collections = get_collection_stats(conn)
    
    sample_doc_ids = []
    sample_chunk_ids = []
    if include_samples:
        sample_doc_ids = get_sample_ids(conn, 'documents')
        sample_chunk_ids = get_sample_ids(conn, 'chunks')
    
    return DatabaseSnapshot(
        name=name,
        counts=counts,
        collections=collections,
        sample_doc_ids=sample_doc_ids,
        sample_chunk_ids=sample_chunk_ids,
    )


def check_ids_present(conn, table: str, ids: List[int]) -> Tuple[List[int], List[int]]:
    """Check which IDs are present/missing in the target database."""
    if not ids:
        return [], []
    
    present = []
    missing = []
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = %s
        """, (table,))
        if not cur.fetchone():
            return [], ids
        
        # Check in batches
        for i in range(0, len(ids), 500):
            batch = ids[i:i+500]
            cur.execute(f"SELECT id FROM {table} WHERE id = ANY(%s)", (batch,))
            found = {row[0] for row in cur.fetchall()}
            for id_ in batch:
                if id_ in found:
                    present.append(id_)
                else:
                    missing.append(id_)
    
    return present, missing


def compare_snapshots(local: DatabaseSnapshot, remote: DatabaseSnapshot, 
                      remote_conn=None, verbose: bool = False) -> Dict[str, Any]:
    """Compare two database snapshots and return diff report."""
    report: Dict[str, Any] = {
        "passed": True,
        "summary": {},
        "table_counts": {},
        "collection_comparison": [],
        "missing_ids": {},
    }
    
    # Compare table counts
    for table in ['collections', 'documents', 'pages', 'chunks', 'chunk_pages']:
        local_count = getattr(local.counts, table)
        remote_count = getattr(remote.counts, table)
        diff = remote_count - local_count
        match = local_count == remote_count
        
        report["table_counts"][table] = {
            "local": local_count,
            "remote": remote_count,
            "diff": diff,
            "match": match,
        }
        
        if not match:
            report["passed"] = False
    
    # Compare collections
    local_collections = {c.slug: c for c in local.collections}
    remote_collections = {c.slug: c for c in remote.collections}
    
    all_slugs = sorted(set(local_collections.keys()) | set(remote_collections.keys()))
    for slug in all_slugs:
        local_c = local_collections.get(slug)
        remote_c = remote_collections.get(slug)
        
        if local_c and not remote_c:
            report["collection_comparison"].append({
                "slug": slug,
                "status": "missing_in_remote",
                "local_docs": local_c.doc_count,
                "local_pages": local_c.page_count,
                "local_chunks": local_c.chunk_count,
            })
            report["passed"] = False
        elif remote_c and not local_c:
            report["collection_comparison"].append({
                "slug": slug,
                "status": "only_in_remote",
                "remote_docs": remote_c.doc_count,
                "remote_pages": remote_c.page_count,
                "remote_chunks": remote_c.chunk_count,
            })
        elif local_c and remote_c:
            doc_match = local_c.doc_count == remote_c.doc_count
            page_match = local_c.page_count == remote_c.page_count
            chunk_match = local_c.chunk_count == remote_c.chunk_count
            
            if not (doc_match and page_match and chunk_match):
                report["collection_comparison"].append({
                    "slug": slug,
                    "status": "mismatch",
                    "local_docs": local_c.doc_count,
                    "remote_docs": remote_c.doc_count,
                    "local_pages": local_c.page_count,
                    "remote_pages": remote_c.page_count,
                    "local_chunks": local_c.chunk_count,
                    "remote_chunks": remote_c.chunk_count,
                })
                report["passed"] = False
            elif verbose:
                report["collection_comparison"].append({
                    "slug": slug,
                    "status": "match",
                    "docs": local_c.doc_count,
                    "pages": local_c.page_count,
                    "chunks": local_c.chunk_count,
                })
    
    # Check sample IDs if remote connection provided
    if remote_conn and local.sample_doc_ids:
        present, missing = check_ids_present(remote_conn, 'documents', local.sample_doc_ids)
        if missing:
            report["missing_ids"]["documents"] = {
                "checked": len(local.sample_doc_ids),
                "missing_count": len(missing),
                "missing_samples": missing[:20],  # Show first 20
            }
            report["passed"] = False
        else:
            report["missing_ids"]["documents"] = {
                "checked": len(local.sample_doc_ids),
                "all_present": True,
            }
    
    if remote_conn and local.sample_chunk_ids:
        present, missing = check_ids_present(remote_conn, 'chunks', local.sample_chunk_ids)
        if missing:
            report["missing_ids"]["chunks"] = {
                "checked": len(local.sample_chunk_ids),
                "missing_count": len(missing),
                "missing_samples": missing[:20],
            }
            report["passed"] = False
        else:
            report["missing_ids"]["chunks"] = {
                "checked": len(local.sample_chunk_ids),
                "all_present": True,
            }
    
    # Summary
    report["summary"] = {
        "local_db": local.name,
        "remote_db": remote.name,
        "total_local_docs": local.counts.documents,
        "total_remote_docs": remote.counts.documents,
        "total_local_chunks": local.counts.chunks,
        "total_remote_chunks": remote.counts.chunks,
        "collections_checked": len(all_slugs),
        "passed": report["passed"],
    }
    
    return report


def print_report(report: Dict[str, Any], verbose: bool = False) -> None:
    """Print a human-readable comparison report."""
    summary = report["summary"]
    status = "[PASS]" if report["passed"] else "[FAIL]"
    
    print(f"\n{'='*60}")
    print(f"DATABASE COMPARISON REPORT: {status}")
    print(f"{'='*60}")
    print(f"Local:  {summary['local_db']}")
    print(f"Remote: {summary['remote_db']}")
    print()
    
    # Table counts
    print("TABLE COUNTS:")
    print("-" * 50)
    print(f"{'Table':<15} {'Local':>12} {'Remote':>12} {'Diff':>10} {'Status':>8}")
    print("-" * 50)
    for table, data in report["table_counts"].items():
        status_icon = "OK" if data["match"] else "DIFF"
        diff_str = f"{data['diff']:+d}" if data['diff'] != 0 else "0"
        print(f"{table:<15} {data['local']:>12,} {data['remote']:>12,} {diff_str:>10} {status_icon:>8}")
    print()
    
    # Collection comparison
    if report["collection_comparison"]:
        issues = [c for c in report["collection_comparison"] if c["status"] != "match"]
        matches = [c for c in report["collection_comparison"] if c["status"] == "match"]
        
        if issues:
            print("COLLECTION ISSUES:")
            print("-" * 50)
            for c in issues:
                if c["status"] == "missing_in_remote":
                    print(f"  [MISSING] {c['slug']}: MISSING in remote")
                    print(f"     Local has: {c['local_docs']} docs, {c['local_pages']} pages, {c['local_chunks']} chunks")
                elif c["status"] == "only_in_remote":
                    print(f"  [EXTRA] {c['slug']}: Only in remote (not in local)")
                    print(f"     Remote has: {c['remote_docs']} docs, {c['remote_pages']} pages, {c['remote_chunks']} chunks")
                elif c["status"] == "mismatch":
                    print(f"  [MISMATCH] {c['slug']}: COUNT MISMATCH")
                    print(f"     Docs:   local={c['local_docs']:,}, remote={c['remote_docs']:,}")
                    print(f"     Pages:  local={c['local_pages']:,}, remote={c['remote_pages']:,}")
                    print(f"     Chunks: local={c['local_chunks']:,}, remote={c['remote_chunks']:,}")
            print()
        
        if verbose and matches:
            print("MATCHING COLLECTIONS:")
            print("-" * 50)
            for c in matches:
                print(f"  [OK] {c['slug']}: {c['docs']} docs, {c['pages']} pages, {c['chunks']} chunks")
            print()
    
    # Missing IDs check
    if report.get("missing_ids"):
        print("ID VERIFICATION:")
        print("-" * 50)
        for table, data in report["missing_ids"].items():
            if data.get("all_present"):
                print(f"  [OK] {table}: {data['checked']} sample IDs all present in remote")
            else:
                print(f"  [MISSING] {table}: {data['missing_count']}/{data['checked']} IDs missing in remote")
                if data.get("missing_samples"):
                    print(f"     Sample missing IDs: {data['missing_samples'][:10]}")
        print()
    
    # Final verdict
    print("=" * 60)
    if report["passed"]:
        print("[PASS] All documents and chunks appear to be successfully synced!")
    else:
        print("[FAIL] Discrepancies found - review issues above")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare documents and chunks between two PostgreSQL databases"
    )
    parser.add_argument(
        "--local",
        default=os.getenv("DATABASE_URL_LOCAL", "postgresql://neh:neh@localhost:5432/neh"),
        help="Local database URL (default: postgresql://neh:neh@localhost:5432/neh)",
    )
    parser.add_argument(
        "--remote",
        default=os.getenv("DATABASE_URL_REMOTE") or os.getenv("DATABASE_URL"),
        help="Remote database URL (default: DATABASE_URL_REMOTE or DATABASE_URL env var)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including matching collections",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--check-ids",
        action="store_true",
        help="Verify sample document and chunk IDs exist in remote",
    )
    
    args = parser.parse_args()
    
    if not args.remote:
        print("ERROR: Remote database URL required.")
        print("Provide --remote or set DATABASE_URL_REMOTE/DATABASE_URL environment variable.")
        print("\nExample:")
        print('  source ./friday_env.sh')
        print('  python scripts/compare_db_data.py --local "postgresql://neh:neh@localhost:5432/neh"')
        sys.exit(2)
    
    # Mask password in display
    def mask_url(url: str) -> str:
        import re
        return re.sub(r':([^:@]+)@', ':***@', url)
    
    print(f"Connecting to local:  {mask_url(args.local)}")
    print(f"Connecting to remote: {mask_url(args.remote)}")
    
    try:
        conn_local = psycopg2.connect(args.local)
        conn_remote = psycopg2.connect(args.remote)
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}")
        sys.exit(1)
    
    try:
        print("\nSnapshotting local database...")
        local_snap = snapshot_database(conn_local, "local", include_samples=args.check_ids)
        
        print("Snapshotting remote database...")
        remote_snap = snapshot_database(conn_remote, "remote", include_samples=False)
        
        print("Comparing...")
        report = compare_snapshots(
            local_snap, 
            remote_snap, 
            remote_conn=conn_remote if args.check_ids else None,
            verbose=args.verbose
        )
        
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print_report(report, verbose=args.verbose)
        
        sys.exit(0 if report["passed"] else 1)
        
    finally:
        conn_local.close()
        conn_remote.close()


if __name__ == "__main__":
    main()
