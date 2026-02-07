#!/usr/bin/env python3
"""Sync concordance_sources, entities, entity_aliases from local DB to RDS.

Usage:
    python scripts/sync_entities_to_rds.py

By default syncs only the concordance index: vassiliev_venona_index_20260130
(that source plus its entities and entity_aliases). Override with
CONCORDANCE_SOURCE_SLUG=other_slug to sync a different source, or leave unset
and set SYNC_ALL_CONCORDANCE=1 to sync all sources (original behavior).

Reads from LOCAL_DATABASE_URL (default: postgresql://neh:neh@localhost:5432/neh)
Writes to DATABASE_URL (the RDS instance from friday_env.sh)
"""
import os
import sys
from typing import Callable, List, Optional, Tuple

import psycopg2

# RDS entities table only allows these entity_type values (0019 migration)
ALLOWED_ENTITY_TYPES = ("person", "org", "place")

LOCAL_DSN = os.getenv("LOCAL_DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh")
REMOTE_DSN = os.getenv("DATABASE_URL")

# Only sync this concordance index (one source + its entities/entity_aliases)
CONCORDANCE_SLUG = os.getenv("CONCORDANCE_SOURCE_SLUG", "vassiliev_venona_index_20260130")
SYNC_ALL = os.getenv("SYNC_ALL_CONCORDANCE", "").strip().lower() in ("1", "true", "yes")

def _log(msg: str) -> None:
    print(msg, flush=True)


# Check DATABASE_URL only when script is run (so we can print banner first)
def _check_env() -> bool:
    if not REMOTE_DSN:
        print("ERROR: DATABASE_URL not set. Run: source friday_env.sh (or set DATABASE_URL)", flush=True)
        return False
    return True

# Order matters: parents before children (FK dependencies).
# concordance_entries must exist on RDS before entities (entities.entry_id -> concordance_entries.id)
TABLES = [
    "concordance_sources",
    "concordance_entries",
    "entities",
    "entity_aliases",
]


def copy_table(
    table: str,
    local_conn,
    remote_conn,
    batch_size: int = 1000,
    where_clause: Optional[str] = None,
    where_params: Tuple[object, ...] = (),
    row_transform: Optional[Callable[[List[str], tuple], tuple]] = None,
):
    """Copy rows from local to remote. If where_clause/where_params set, only copy matching rows.
    row_transform(column_names, row) -> row to apply before insert (e.g. normalize entity_type).
    """
    with local_conn.cursor() as lcur:
        if where_clause:
            lcur.execute(f"SELECT COUNT(*) FROM {table} WHERE {where_clause}", where_params)
        else:
            lcur.execute(f"SELECT COUNT(*) FROM {table}")
        total = lcur.fetchone()[0]
        _log(f"  {table}: {total} rows to copy")

        if total == 0:
            return 0

        # Get column names
        lcur.execute(f"SELECT * FROM {table} LIMIT 0")
        columns = [desc[0] for desc in lcur.description]
        col_list = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))

        if where_clause:
            lcur.execute(
                f"SELECT {col_list} FROM {table} WHERE {where_clause} ORDER BY id",
                where_params,
            )
        else:
            lcur.execute(f"SELECT {col_list} FROM {table} ORDER BY id")

        copied = 0
        with remote_conn.cursor() as rcur:
            while True:
                rows = lcur.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    try:
                        out_row = row
                        if row_transform:
                            out_row = row_transform(columns, row)
                        rcur.execute(
                            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                            f"ON CONFLICT (id) DO NOTHING",
                            out_row,
                        )
                        copied += 1
                    except Exception as e:
                        # Log and skip problematic rows
                        remote_conn.rollback()
                        _log(f"    SKIP row id={row[0]}: {e}")
                        continue
                remote_conn.commit()
                _log(f"    ... {copied}/{total}")

        _log(f"    ... {copied}/{total} copied")
        return copied


def _entities_row_transform(columns: List[str], row: tuple) -> tuple:
    """Normalize entity_type to RDS-allowed values (person, org, place)."""
    try:
        idx = columns.index("entity_type")
    except ValueError:
        return row
    val = row[idx]
    if val in ALLOWED_ENTITY_TYPES:
        return row
    # RDS only allows person/org/place; map cover_name etc. to person
    return tuple(row[i] if i != idx else "person" for i in range(len(row)))


def main():
    _log("sync_entities_to_rds: starting...")
    _log(f"Source (local): {LOCAL_DSN}")
    _log(f"Target (RDS):   {REMOTE_DSN[:60]}...")
    if not SYNC_ALL:
        _log(f"Filter: concordance_sources.slug = {CONCORDANCE_SLUG!r} only")
    else:
        _log("Filter: none (sync all concordance data)")
    _log("")

    _log("Connecting to local DB...")
    try:
        local_conn = psycopg2.connect(LOCAL_DSN)
    except Exception as e:
        _log(f"ERROR: Failed to connect to local DB: {e}")
        sys.exit(1)
    _log("Connecting to RDS...")
    try:
        remote_conn = psycopg2.connect(REMOTE_DSN)
    except Exception as e:
        _log(f"ERROR: Failed to connect to RDS: {e}")
        local_conn.close()
        sys.exit(1)

    # Verify remote connection
    with remote_conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        db_name = cur.fetchone()[0]
        _log(f"Remote database: {db_name}")

    source_id = None
    if not SYNC_ALL:
        with local_conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM concordance_sources WHERE slug = %s",
                (CONCORDANCE_SLUG,),
            )
            row = cur.fetchone()
            if not row:
                _log(f"ERROR: No concordance_sources row with slug={CONCORDANCE_SLUG!r} in local DB")
                local_conn.close()
                remote_conn.close()
                sys.exit(1)
            source_id = row[0]
        _log(f"Local source_id for slug {CONCORDANCE_SLUG!r}: {source_id}\n")

    _log("=== Syncing tables ===")

    for table in TABLES:
        # Check if table exists on remote
        with remote_conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (table,),
            )
            if not cur.fetchone()[0]:
                _log(f"  {table}: TABLE DOES NOT EXIST on remote — skipping")
                continue

        if SYNC_ALL:
            copy_table(table, local_conn, remote_conn)
        else:
            if table == "concordance_sources":
                copy_table(
                    table,
                    local_conn,
                    remote_conn,
                    where_clause="slug = %s",
                    where_params=(CONCORDANCE_SLUG,),
                )
            elif table == "concordance_entries":
                copy_table(
                    table,
                    local_conn,
                    remote_conn,
                    where_clause="source_id = %s",
                    where_params=(source_id,),
                )
            elif table == "entities":
                copy_table(
                    table,
                    local_conn,
                    remote_conn,
                    where_clause="source_id = %s",
                    where_params=(source_id,),
                    row_transform=_entities_row_transform,
                )
            elif table == "entity_aliases":
                copy_table(
                    table,
                    local_conn,
                    remote_conn,
                    where_clause="source_id = %s",
                    where_params=(source_id,),
                )
            else:
                copy_table(table, local_conn, remote_conn)

    # Verify
    _log("")
    _log("=== Verification ===")
    for table in TABLES:
        with remote_conn.cursor() as cur:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                _log(f"  {table}: {count} rows on remote")
            except Exception:
                remote_conn.rollback()

    if not SYNC_ALL and source_id is not None:
        with remote_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM concordance_sources WHERE slug = %s",
                (CONCORDANCE_SLUG,),
            )
            cs_count = cur.fetchone()[0]
            _log(f"  concordance_sources slug={CONCORDANCE_SLUG!r}: {cs_count} row(s) on remote")

    # PAL test (only meaningful for vassiliev/venona data)
    with remote_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM entity_aliases WHERE LOWER(alias) = LOWER('PAL')"
        )
        pal_count = cur.fetchone()[0]
        _log(f"\n  PAL alias check: {pal_count} rows (expected: 2)")

    local_conn.close()
    remote_conn.close()
    _log("\nDone!")


if __name__ == "__main__":
    print("sync_entities_to_rds.py running...", flush=True)
    if not _check_env():
        sys.exit(1)
    main()
