#!/usr/bin/env python3
"""Compare local DB and RDS, then optionally copy missing data.

Resume: Progress is saved to sync_progress.json (or SYNC_PROGRESS_FILE). If the
script is interrupted, run the same command again to continue from the last
batch. Use --no-resume to start from the beginning.

Usage:
    # Compare only (safe, read-only):
    python scripts/compare_and_sync_dbs.py

    # Copy specific table(s); resumes from last id if interrupted:
    python scripts/compare_and_sync_dbs.py --sync entity_mentions date_mentions

    # Copy all tables that are empty on RDS but populated locally:
    python scripts/compare_and_sync_dbs.py --sync-missing

    # Dry run (show what would be copied):
    python scripts/compare_and_sync_dbs.py --sync-missing --dry-run

    # Start from scratch (ignore progress file):
    python scripts/compare_and_sync_dbs.py --sync entity_mentions --no-resume

Reads from LOCAL_DATABASE_URL (default: postgresql://neh:neh@localhost:5432/neh)
Writes to DATABASE_URL (the RDS instance from friday_env.sh)
"""
import argparse
import json
import os
import sys
import time

import psycopg2
from psycopg2.extras import execute_values

# Progress file for resume (stores last_synced_id per table)
PROGRESS_FILE = os.getenv(
    "SYNC_PROGRESS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_progress.json"),
)

LOCAL_DSN = os.getenv("LOCAL_DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh")
REMOTE_DSN = os.getenv("DATABASE_URL")

# Tables to compare, in FK-safe copy order (parents before children).
# This covers all entity/mention/concordance/research tables.
COMPARE_TABLES = [
    # Core corpus
    "collections",
    "documents",
    "chunks",
    "chunk_metadata",
    # Concordance
    "concordance_sources",
    "concordance_entries",
    # Entities
    "entities",
    "entity_aliases",
    "entity_mentions",
    "entity_relationships",
    "entity_resolution_reviews",
    "entity_alias_preferred",
    # Date mentions
    "date_mentions",
    # Research infrastructure
    "research_sessions",
    "research_messages",
    "retrieval_runs",
    "result_sets",
    "result_set_items",
    # NER / corpus discovery
    "mention_candidates",
    "mention_review_queue",
    # Focus spans
    "focus_spans",
    # V9
    "evidence_sets",
    "v9_runs",
    "evidence_items",
    "v9_run_steps",
]

# Tables that don't have a simple `id` primary key — need special ON CONFLICT handling.
# Map: table -> conflict target for ON CONFLICT
CONFLICT_TARGETS = {
    "chunk_metadata": "(chunk_id)",
    "entity_mentions": "(id)",  # has id PK, plus unique(chunk_id, entity_id, start_char, end_char)
    "focus_spans": "(entity_id)",
}

# RDS only allows these entity_type values (from migration 0019)
ALLOWED_ENTITY_TYPES = {"person", "org", "place"}

# Map non-standard entity_type values to allowed ones
ENTITY_TYPE_MAP = {
    "cover_name": "person",
    "covername": "person",
    "code_name": "person",
    "codename": "person",
    "organization": "org",
}


def _row_transform_entities(columns, row):
    """Normalize entity_type to RDS-allowed values."""
    try:
        idx = columns.index("entity_type")
    except ValueError:
        return row
    val = row[idx]
    if val in ALLOWED_ENTITY_TYPES:
        return row
    mapped = ENTITY_TYPE_MAP.get(val, "person")  # default to person
    return tuple(row[i] if i != idx else mapped for i in range(len(row)))


# Per-table row transforms
ROW_TRANSFORMS = {
    "entities": _row_transform_entities,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_progress() -> dict:
    """Load progress from file. Returns dict table -> last_synced_id (int)."""
    if not os.path.isfile(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress(progress: dict) -> None:
    """Write progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=0)


def get_table_count(conn, table: str) -> int:
    """Get row count for a table. Returns -1 if table doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = %s)",
                (table,),
            )
            if not cur.fetchone()[0]:
                return -1
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            return cur.fetchone()[0]
    except Exception:
        conn.rollback()
        return -1


def get_all_tables(conn) -> list:
    """List all public tables in the database."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
        return [row[0] for row in cur.fetchall()]


def get_columns(conn, table: str) -> list:
    """Get column names for a table."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table} LIMIT 0")
        return [desc[0] for desc in cur.description]


def compare_databases(local_conn, remote_conn) -> list:
    """Compare all tables between local and remote. Returns list of diffs."""
    log("=" * 70)
    log(f"{'TABLE':<35} {'LOCAL':>12} {'RDS':>12} {'STATUS':<15}")
    log("=" * 70)

    diffs = []
    local_tables = set(get_all_tables(local_conn))
    remote_tables = set(get_all_tables(remote_conn))

    # Check all tables we care about, plus any others in either DB
    all_tables = sorted(set(COMPARE_TABLES) | local_tables | remote_tables)

    for table in all_tables:
        local_count = get_table_count(local_conn, table)
        remote_count = get_table_count(remote_conn, table)

        # Determine status
        if local_count == -1 and remote_count == -1:
            continue  # doesn't exist in either
        elif local_count == -1:
            status = "RDS only"
        elif remote_count == -1:
            status = "LOCAL only"
        elif remote_count == 0 and local_count > 0:
            status = "EMPTY on RDS"
        elif remote_count < local_count:
            status = "BEHIND"
        elif remote_count == local_count:
            status = "OK"
        else:
            status = "RDS ahead"

        local_str = str(local_count) if local_count >= 0 else "---"
        remote_str = str(remote_count) if remote_count >= 0 else "---"

        # Highlight problems
        marker = ""
        if status in ("EMPTY on RDS", "BEHIND"):
            marker = " <<<"
        elif status == "LOCAL only":
            marker = " (!)"

        log(f"  {table:<33} {local_str:>12} {remote_str:>12} {status:<15}{marker}")

        diffs.append({
            "table": table,
            "local": local_count,
            "remote": remote_count,
            "status": status,
        })

    log("=" * 70)

    # Summary
    empty_on_rds = [d for d in diffs if d["status"] == "EMPTY on RDS"]
    behind = [d for d in diffs if d["status"] == "BEHIND"]
    local_only = [d for d in diffs if d["status"] == "LOCAL only"]

    log("")
    if empty_on_rds:
        log(f"Tables EMPTY on RDS ({len(empty_on_rds)}):")
        for d in empty_on_rds:
            log(f"  - {d['table']} ({d['local']:,} rows locally)")
    if behind:
        log(f"\nTables BEHIND on RDS ({len(behind)}):")
        for d in behind:
            diff = d["local"] - d["remote"]
            log(f"  - {d['table']} (local={d['local']:,}, rds={d['remote']:,}, diff={diff:,})")
    if local_only:
        log(f"\nTables LOCAL only ({len(local_only)}):")
        for d in local_only:
            log(f"  - {d['table']} ({d['local']:,} rows)")

    if not empty_on_rds and not behind:
        log("All tables are in sync!")

    return diffs


def get_common_columns(local_conn, remote_conn, table: str) -> list:
    """Get columns that exist in BOTH local and remote versions of a table."""
    local_cols = set(get_columns(local_conn, table))
    remote_cols = set(get_columns(remote_conn, table))
    # Preserve local column order, but only include columns that exist on remote
    local_ordered = get_columns(local_conn, table)
    return [c for c in local_ordered if c in remote_cols]


def copy_table(
    local_conn,
    remote_conn,
    table: str,
    batch_size: int = 2000,
    dry_run: bool = False,
    resume: bool = True,
    progress: dict = None,
) -> int:
    """Copy rows from local to remote using ON CONFLICT DO NOTHING.

    Uses only columns that exist in both databases to handle schema differences.
    If resume=True and progress has last_id for this table, only copies rows with id > last_id.
    """
    if progress is None:
        progress = load_progress()

    # Get common columns
    common_cols = get_common_columns(local_conn, remote_conn, table)
    if not common_cols:
        log(f"  ERROR: No common columns between local and remote for {table}")
        return 0

    col_list = ", ".join(common_cols)
    placeholders = ", ".join(["%s"] * len(common_cols))

    # Determine conflict target
    conflict = CONFLICT_TARGETS.get(table, "(id)")

    has_id = "id" in common_cols
    id_idx = common_cols.index("id") if has_id else None
    last_id = None
    if resume and has_id and table in progress:
        try:
            last_id = progress[table]
            if last_id is not None and not isinstance(last_id, (int, float)):
                last_id = None
        except (TypeError, ValueError):
            last_id = None

    # Build query: optionally restrict to id > last_id for resume
    # For entity_mentions: only copy rows whose chunk_id and document_id exist on RDS (avoids FK violations)
    order_clause = "ORDER BY id" if has_id else ""
    where_parts = []
    where_params = []
    if table == "entity_mentions" and "chunk_id" in common_cols:
        with remote_conn.cursor() as rcur:
            rcur.execute("SELECT id FROM chunks")
            rds_chunk_ids = [row[0] for row in rcur.fetchall()]
            rcur.execute("SELECT id FROM documents")
            rds_document_ids = [row[0] for row in rcur.fetchall()]
        if not rds_chunk_ids:
            log(f"  ERROR: No chunks on RDS; cannot sync entity_mentions (FK chunk_id)")
            return 0
        if not rds_document_ids:
            log(f"  ERROR: No documents on RDS; cannot sync entity_mentions (FK document_id)")
            return 0
        where_parts.append("chunk_id = ANY(%s)")
        where_params.append(rds_chunk_ids)
        where_parts.append("document_id = ANY(%s)")
        where_params.append(rds_document_ids)
        log(f"  Filtering to mentions whose chunk_id and document_id exist on RDS "
            f"({len(rds_chunk_ids):,} chunks, {len(rds_document_ids):,} docs)")
    # For date_mentions: only copy rows whose chunk_id and document_id exist on RDS
    elif table == "date_mentions" and "chunk_id" in common_cols:
        with remote_conn.cursor() as rcur:
            rcur.execute("SELECT id FROM chunks")
            rds_chunk_ids = [row[0] for row in rcur.fetchall()]
            rcur.execute("SELECT id FROM documents")
            rds_document_ids = [row[0] for row in rcur.fetchall()]
        if not rds_chunk_ids or not rds_document_ids:
            log(f"  ERROR: No chunks or documents on RDS; cannot sync date_mentions")
            return 0
        where_parts.append("chunk_id = ANY(%s)")
        where_params.append(rds_chunk_ids)
        where_parts.append("document_id = ANY(%s)")
        where_params.append(rds_document_ids)
        log(f"  Filtering to date_mentions whose chunk_id and document_id exist on RDS "
            f"({len(rds_chunk_ids):,} chunks, {len(rds_document_ids):,} docs)")
    if last_id is not None:
        where_parts.append("id > %s")
        where_params.append(last_id)
    where_clause = "WHERE " + " AND ".join(where_parts) + " " if where_parts else ""
    where_params = tuple(where_params)

    with local_conn.cursor() as lcur:
        if where_clause:
            lcur.execute(f"SELECT COUNT(*) FROM {table} {where_clause}", where_params)
        else:
            lcur.execute(f"SELECT COUNT(*) FROM {table}")
        total = lcur.fetchone()[0]

    with remote_conn.cursor() as rcur:
        rcur.execute(f"SELECT COUNT(*) FROM {table}")
        remote_before = rcur.fetchone()[0]

    log(f"\n  {table}: {total:,} rows to copy locally, {remote_before:,} on RDS")
    if last_id is not None:
        log(f"  Resuming from id > {last_id:,}")
    log(f"  Columns: {len(common_cols)} common ({col_list[:80]}...)")
    log(f"  Conflict target: ON CONFLICT {conflict} DO NOTHING")

    if dry_run:
        log(f"  [DRY RUN] Would copy up to {total:,} rows")
        return 0

    if total == 0:
        log("  Nothing to copy.")
        return 0

    # Row transform (e.g. normalize entity_type for entities table)
    transform = ROW_TRANSFORMS.get(table)
    if transform:
        log(f"  Row transform: {transform.__name__}")

    start_time = time.time()
    copied = 0
    errors = 0
    select_sql = f"SELECT {col_list} FROM {table} {where_clause} {order_clause}".strip()

    with local_conn.cursor("server_cursor") as lcur:
        lcur.itersize = batch_size
        if where_params:
            lcur.execute(select_sql, where_params)
        else:
            lcur.execute(select_sql)

        batch = []
        for row in lcur:
            if transform:
                row = transform(common_cols, row)
            batch.append(row)
            if len(batch) >= batch_size:
                c, e = _insert_batch(remote_conn, table, col_list, placeholders, conflict, batch)
                copied += c
                errors += e
                # Persist progress (last_id) so we can resume
                if has_id and id_idx is not None and batch:
                    batch_last_id = max(row[id_idx] for row in batch)
                    progress[table] = batch_last_id
                    save_progress(progress)
                elapsed = time.time() - start_time
                rate = copied / elapsed if elapsed > 0 else 0
                log(f"    ... {copied:,}/{total:,} ({rate:.0f} rows/s, {errors} errors)")
                batch = []

        # Final batch
        if batch:
            c, e = _insert_batch(remote_conn, table, col_list, placeholders, conflict, batch)
            copied += c
            errors += e
            if has_id and id_idx is not None and batch:
                batch_last_id = max(row[id_idx] for row in batch)
                progress[table] = batch_last_id
                save_progress(progress)

    elapsed = time.time() - start_time

    with remote_conn.cursor() as rcur:
        rcur.execute(f"SELECT COUNT(*) FROM {table}")
        remote_after = rcur.fetchone()[0]

    new_rows = remote_after - remote_before
    log(f"  Done: {copied:,} attempted, {new_rows:,} new rows on RDS, "
        f"{errors} errors, {elapsed:.1f}s")
    return new_rows


def _insert_batch(remote_conn, table, col_list, placeholders, conflict, batch) -> tuple:
    """Insert a batch of rows in one round-trip. On failure, fall back to row-by-row.
    Returns (attempted, errors)."""
    if not batch:
        return 0, 0
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES %s "
        f"ON CONFLICT {conflict} DO NOTHING"
    )
    with remote_conn.cursor() as rcur:
        try:
            execute_values(rcur, sql, batch, page_size=len(batch))
            remote_conn.commit()
            return len(batch), 0
        except Exception:
            remote_conn.rollback()
        # Fallback: insert row-by-row so we can skip bad rows; show progress every 100 rows
        attempted = 0
        errors = 0
        batch_len = len(batch)
        for i, row in enumerate(batch):
            try:
                rcur.execute(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                    f"ON CONFLICT {conflict} DO NOTHING",
                    row,
                )
                attempted += 1
            except Exception as e:
                remote_conn.rollback()
                errors += 1
                if errors <= 5:
                    log(f"    SKIP: {e}")
                elif errors == 6:
                    log(f"    ... suppressing further error messages")
            if (i + 1) % 100 == 0:
                log(f"    fallback row-by-row: {i + 1:,}/{batch_len:,} (ok={attempted}, skip={errors})")
        remote_conn.commit()
        return attempted, errors


def main():
    parser = argparse.ArgumentParser(description="Compare local DB and RDS, optionally sync.")
    parser.add_argument("--sync", type=str, nargs="+", help="Copy one or more tables to RDS")
    parser.add_argument("--sync-missing", action="store_true",
                        help="Copy all tables that are empty on RDS but populated locally")
    parser.add_argument("--sync-behind", action="store_true",
                        help="Copy all tables where RDS has fewer rows than local")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without doing it")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for inserts (default: 1000)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore progress file and sync from the beginning")
    args = parser.parse_args()

    if not REMOTE_DSN:
        log("ERROR: DATABASE_URL not set. Run: source friday_env.sh")
        sys.exit(1)

    log("Connecting to local DB...")
    local_conn = psycopg2.connect(LOCAL_DSN)
    log(f"  Local: {LOCAL_DSN}")

    log("Connecting to RDS...")
    remote_conn = psycopg2.connect(REMOTE_DSN)
    with remote_conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        log(f"  Remote: {cur.fetchone()[0]}")
    log("")

    # Compare first (skip full table dump when syncing specific tables)
    if args.sync:
        diffs = []
    else:
        diffs = compare_databases(local_conn, remote_conn)

    # Sync specific table(s)
    if args.sync:
        progress = load_progress() if not args.no_resume else {}
        if args.no_resume:
            for table in args.sync:
                progress.pop(table, None)
            save_progress(progress)
        for table in args.sync:
            local_count = get_table_count(local_conn, table)
            remote_count = get_table_count(remote_conn, table)
            if local_count == -1:
                log(f"\nERROR: Table '{table}' does not exist locally — skipping")
                continue
            if remote_count == -1:
                log(f"\nERROR: Table '{table}' does not exist on RDS — skipping")
                continue
            log(f"\n>>> Syncing: {table}")
            copy_table(local_conn, remote_conn, table,
                       batch_size=args.batch_size, dry_run=args.dry_run,
                       resume=not args.no_resume, progress=progress)

    # Sync all empty tables
    elif args.sync_missing:
        targets = [d for d in diffs
                   if d["status"] == "EMPTY on RDS" and d["table"] in COMPARE_TABLES]
        if not targets:
            log("\nNo empty-on-RDS tables to sync.")
        else:
            log(f"\n>>> Syncing {len(targets)} empty-on-RDS tables:")
            for d in targets:
                log(f"  - {d['table']} ({d['local']:,} rows)")

            if not args.dry_run:
                confirm = input("\nProceed? (y/N): ").strip().lower()
                if confirm != "y":
                    log("Aborted.")
                    sys.exit(0)

            progress = load_progress() if not args.no_resume else {}
            if args.no_resume:
                for d in targets:
                    progress.pop(d["table"], None)
                save_progress(progress)
            for d in targets:
                copy_table(local_conn, remote_conn, d["table"],
                           batch_size=args.batch_size, dry_run=args.dry_run,
                           resume=not args.no_resume, progress=progress)

    # Sync all behind tables
    elif args.sync_behind:
        targets = [d for d in diffs
                   if d["status"] in ("EMPTY on RDS", "BEHIND") and d["table"] in COMPARE_TABLES]
        if not targets:
            log("\nNo behind tables to sync.")
        else:
            log(f"\n>>> Syncing {len(targets)} behind/empty tables:")
            for d in targets:
                diff = d["local"] - max(d["remote"], 0)
                log(f"  - {d['table']} (up to {diff:,} new rows)")

            if not args.dry_run:
                confirm = input("\nProceed? (y/N): ").strip().lower()
                if confirm != "y":
                    log("Aborted.")
                    sys.exit(0)

            progress = load_progress() if not args.no_resume else {}
            if args.no_resume:
                for d in targets:
                    progress.pop(d["table"], None)
                save_progress(progress)
            for d in targets:
                copy_table(local_conn, remote_conn, d["table"],
                           batch_size=args.batch_size, dry_run=args.dry_run,
                           resume=not args.no_resume, progress=progress)

    local_conn.close()
    remote_conn.close()
    log("\nDone.")


if __name__ == "__main__":
    main()
