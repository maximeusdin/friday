#!/usr/bin/env python3
"""
PEM Lane Health Check — verify page_entity_mentions coverage for PEM lane.

Reports:
  - Total rows in page_entity_mentions
  - Coverage: % of PEM page_ids that map to >= 1 chunk_id via chunk_pages
  - Top entities by alias surface count in alias-scoped corpora
  - PEM revision from app_kv
  - Warns if coverage < threshold (default 95%)

Usage:
    python scripts/pem_lane_health.py
    python scripts/pem_lane_health.py --threshold 90
    python scripts/pem_lane_health.py --top-n 20
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

ALIAS_SCOPED_COLLECTIONS = ("venona", "vassiliev")


def get_connection():
    """Get a database connection using the same env vars as the app."""
    import psycopg2
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "friday"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
    )


def check_pem_health(conn, threshold: float = 95.0, top_n: int = 15):
    """Run PEM lane health checks and print results."""
    print("=" * 70)
    print("PEM Lane Health Check")
    print("=" * 70)

    with conn.cursor() as cur:
        # --- PEM revision ---
        try:
            cur.execute(
                "SELECT value FROM app_kv WHERE key = 'page_entity_mentions_revision'"
            )
            row = cur.fetchone()
            revision = row[0] if row else "(not set)"
        except Exception:
            conn.rollback()
            revision = "(app_kv not available)"
        print(f"\nPEM Revision: {revision}")

        # --- Total rows ---
        try:
            cur.execute("SELECT COUNT(*) FROM page_entity_mentions")
            total_rows = cur.fetchone()[0]
        except Exception:
            conn.rollback()
            print("\nERROR: page_entity_mentions table not found or empty.")
            return False
        print(f"Total PEM rows: {total_rows:,}")

        if total_rows == 0:
            print("\nWARNING: page_entity_mentions is empty!")
            return False

        # --- Alias-scoped row counts ---
        cur.execute("""
            SELECT collection_slug, COUNT(*) AS cnt
            FROM page_entity_mentions
            WHERE collection_slug = ANY(%s)
            GROUP BY collection_slug
            ORDER BY cnt DESC
        """, (list(ALIAS_SCOPED_COLLECTIONS),))
        alias_counts = cur.fetchall()
        print(f"\nAlias-scoped rows:")
        for coll, cnt in alias_counts:
            print(f"  {coll}: {cnt:,}")

        # --- Coverage: % of PEM page_ids with >= 1 chunk via chunk_pages ---
        cur.execute("""
            SELECT COUNT(DISTINCT pem.page_id) AS total_pem_pages,
                   COUNT(DISTINCT CASE WHEN cp.chunk_id IS NOT NULL THEN pem.page_id END) AS covered_pages
            FROM page_entity_mentions pem
            LEFT JOIN chunk_pages cp ON cp.page_id = pem.page_id
        """)
        total_pages, covered_pages = cur.fetchone()
        coverage = (covered_pages / total_pages * 100) if total_pages > 0 else 0
        print(f"\nPage coverage: {covered_pages:,} / {total_pages:,} = {coverage:.1f}%")
        if coverage < threshold:
            print(f"  WARNING: Coverage {coverage:.1f}% is below threshold {threshold:.1f}%!")

        # --- Alias-scoped coverage ---
        cur.execute("""
            SELECT COUNT(DISTINCT pem.page_id) AS total_pages,
                   COUNT(DISTINCT CASE WHEN cp.chunk_id IS NOT NULL THEN pem.page_id END) AS covered
            FROM page_entity_mentions pem
            LEFT JOIN chunk_pages cp ON cp.page_id = pem.page_id
            WHERE pem.collection_slug = ANY(%s)
        """, (list(ALIAS_SCOPED_COLLECTIONS),))
        alias_total, alias_covered = cur.fetchone()
        alias_cov = (alias_covered / alias_total * 100) if alias_total > 0 else 0
        print(f"Alias-scoped coverage: {alias_covered:,} / {alias_total:,} = {alias_cov:.1f}%")

        # --- Top entities by alias surface diversity ---
        cur.execute("""
            SELECT pem.entity_id,
                   e.canonical_name,
                   COUNT(DISTINCT pem.surface_norm) AS n_surfaces,
                   COUNT(DISTINCT pem.page_id) AS n_pages,
                   COUNT(DISTINCT pem.collection_slug) AS n_colls
            FROM page_entity_mentions pem
            JOIN entities e ON e.id = pem.entity_id
            WHERE pem.collection_slug = ANY(%s)
            GROUP BY pem.entity_id, e.canonical_name
            ORDER BY n_surfaces DESC, n_pages DESC
            LIMIT %s
        """, (list(ALIAS_SCOPED_COLLECTIONS), top_n))
        top_entities = cur.fetchall()
        print(f"\nTop {top_n} entities by alias surface diversity (alias-scoped):")
        print(f"  {'Entity ID':>10}  {'Surfaces':>8}  {'Pages':>8}  {'Colls':>5}  Name")
        print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*40}")
        for eid, name, n_surf, n_pages, n_colls in top_entities:
            print(f"  {eid:>10}  {n_surf:>8}  {n_pages:>8}  {n_colls:>5}  {name[:50]}")

        # --- Source distribution ---
        cur.execute("""
            SELECT source, truth_level, COUNT(*) AS cnt
            FROM page_entity_mentions
            WHERE collection_slug = ANY(%s)
            GROUP BY source, truth_level
            ORDER BY cnt DESC
        """, (list(ALIAS_SCOPED_COLLECTIONS),))
        source_dist = cur.fetchall()
        print(f"\nSource distribution (alias-scoped):")
        for source, truth, cnt in source_dist:
            print(f"  {source or '(null)':30s} {truth:15s} {cnt:>10,}")

        # --- Summary ---
        print(f"\n{'=' * 70}")
        ok = coverage >= threshold
        status = "PASS" if ok else "FAIL"
        print(f"Status: {status} (coverage={coverage:.1f}%, threshold={threshold:.1f}%)")
        print(f"{'=' * 70}")
        return ok


def main():
    parser = argparse.ArgumentParser(description="PEM Lane Health Check")
    parser.add_argument(
        "--threshold", type=float, default=95.0,
        help="Minimum coverage %% (default: 95)",
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Number of top entities to show (default: 15)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        conn = get_connection()
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        sys.exit(1)

    try:
        ok = check_pem_health(conn, threshold=args.threshold, top_n=args.top_n)
        sys.exit(0 if ok else 1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
