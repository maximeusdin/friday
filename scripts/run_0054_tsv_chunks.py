#!/usr/bin/env python3
"""
Run the chunks.tsv migration (add tsv column + GIN index) with long timeout and optional progress.

Use after 0054_v9_sessions_evidence.sql. Reads DATABASE_URL from the environment.

Usage:
  export DATABASE_URL='postgresql://user:pass@host:5432/dbname?sslmode=require'
  python scripts/run_0054_tsv_chunks.py

  # No progress polling (just run with long timeout):
  python scripts/run_0054_tsv_chunks.py --no-progress

  # Custom timeout in seconds (default 0 = no limit):
  python scripts/run_0054_tsv_chunks.py --timeout 7200

  # If you hit "memory required is X MB, maintenance_work_mem is Y MB":
  python scripts/run_0054_tsv_chunks.py --maintenance-work-mem 256
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 required. pip install psycopg2-binary", file=sys.stderr)
    sys.exit(1)


def get_conn(database_url: str):
    return psycopg2.connect(database_url)


def add_tsv_column(conn, timeout_sec: int, maintenance_work_mem_mb: int) -> bool:
    """Add chunks.tsv if missing. Returns True if column was added, False if already existed."""
    cur = conn.cursor()
    cur.execute("SET statement_timeout = %s", ("0" if timeout_sec == 0 else f"{timeout_sec}s",))
    cur.execute("SET maintenance_work_mem = %s", (f"{maintenance_work_mem_mb}MB",))
    cur.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'tsv'
        """
    )
    if cur.fetchone():
        conn.commit()
        return False
    print("Adding chunks.tsv column (generated tsvector). This may take a long time on large tables...")
    cur.execute(
        """
        ALTER TABLE chunks ADD COLUMN tsv tsvector
          GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
        """
    )
    conn.commit()
    print("chunks.tsv column added.")
    return True


def create_index_concurrently(
    database_url: str, timeout_sec: int, progress: bool, maintenance_work_mem_mb: int
) -> None:
    """Create GIN index on chunks(tsv). CONCURRENTLY so we can poll progress from another connection."""
    index_done = threading.Event()
    index_error: list[Exception] = []

    def run_create_index():
        try:
            conn = get_conn(database_url)
            conn.autocommit = True  # CONCURRENTLY cannot run inside a transaction
            cur = conn.cursor()
            cur.execute("SET statement_timeout = %s", ("0" if timeout_sec == 0 else f"{timeout_sec}s",))
            cur.execute("SET maintenance_work_mem = %s", (f"{maintenance_work_mem_mb}MB",))
            cur.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_tsv_gin ON chunks USING GIN (tsv)"
            )
            conn.close()
        except Exception as e:
            index_error.append(e)
        finally:
            index_done.set()

    t = threading.Thread(target=run_create_index, daemon=True)
    t.start()

    if progress:
        # Poll progress from a separate connection (PostgreSQL 12+)
        time.sleep(1)
        try:
            prog_conn = get_conn(database_url)
            prog_conn.autocommit = True
            prog_cur = prog_conn.cursor()
            last_phase = None
            while not index_done.is_set():
                prog_cur.execute(
                    """
                    SELECT phase, blocks_total, blocks_done, tuples_total, tuples_done
                    FROM pg_stat_progress_create_index
                    WHERE relid = (SELECT oid FROM pg_class WHERE relname = 'chunks')
                    """
                )
                row = prog_cur.fetchone()
                if row:
                    phase, blocks_total, blocks_done, tuples_total, tuples_done = row
                    if phase != last_phase:
                        print(f"  Phase: {phase}")
                        last_phase = phase
                    if blocks_total and blocks_total > 0:
                        pct = 100.0 * (blocks_done or 0) / blocks_total
                        print(f"\r  Blocks: {blocks_done or 0}/{blocks_total} ({pct:.1f}%)  ", end="", flush=True)
                    elif tuples_total and tuples_total > 0:
                        pct = 100.0 * (tuples_done or 0) / tuples_total
                        print(f"\r  Tuples: {tuples_done or 0}/{tuples_total} ({pct:.1f}%)  ", end="", flush=True)
                else:
                    if index_done.is_set():
                        break
                    print("\r  Waiting for index build to appear in pg_stat_progress_create_index...  ", end="", flush=True)
                time.sleep(2)
            prog_conn.close()
        except Exception as e:
            # Progress view might not exist on older PG or different schema
            print(f"  (Progress polling skipped: {e})")
        print()

    t.join()
    if index_error:
        raise index_error[0]
    print("GIN index idx_chunks_tsv_gin created (or already existed).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run chunks.tsv migration with long timeout and optional progress.")
    ap.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Database URL (default: DATABASE_URL env).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Statement timeout in seconds; 0 = no limit (default: 0).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not poll pg_stat_progress_create_index during index build.",
    )
    ap.add_argument(
        "--maintenance-work-mem",
        type=int,
        default=256,
        metavar="MB",
        help="Session maintenance_work_mem in MB (default: 256). Increase if you hit memory limit.",
    )
    args = ap.parse_args()

    if not args.database_url:
        print("ERROR: Set DATABASE_URL or pass --database-url", file=sys.stderr)
        sys.exit(2)

    database_url = args.database_url
    timeout_sec = args.timeout
    progress = not args.no_progress
    maintenance_mb = args.maintenance_work_mem

    conn = get_conn(database_url)
    try:
        add_tsv_column(conn, timeout_sec, maintenance_mb)
    finally:
        conn.close()

    print("Creating GIN index on chunks(tsv) (CONCURRENTLY)...")
    create_index_concurrently(database_url, timeout_sec, progress, maintenance_mb)

    print("Done.")


if __name__ == "__main__":
    main()
