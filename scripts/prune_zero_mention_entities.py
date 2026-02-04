#!/usr/bin/env python3
"""
Prune entities (and their aliases) that have zero corpus mentions.

Why this exists:
- Dedup / review CLIs get cluttered with entities that have aliases but no evidence.
- These "dead" entities often come from partial ingests, aborted workflows, or
  over-eager entity creation.

Safety model:
- We only delete entities that have:
  - 0 rows in entity_mentions (authoritative evidence table), AND
  - 0 rows in mention_candidates with resolved_entity_id (work-in-progress evidence)
- AND are NOT referenced by a small set of "non-cascading" tables that would
  block deletion or represent human overrides.

Default is DRY RUN. Use --apply to actually delete.

Usage:
  python scripts/prune_zero_mention_entities.py
  python scripts/prune_zero_mention_entities.py --limit 100
  python scripts/prune_zero_mention_entities.py --apply
  python scripts/prune_zero_mention_entities.py --apply --entity-type person
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


def get_conn():
    # Match existing ingest scripts' env conventions.
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", os.environ.get("POSTGRES_HOST", "localhost")),
        port=int(os.environ.get("DB_PORT", os.environ.get("POSTGRES_PORT", "5432"))),
        dbname=os.environ.get("DB_NAME", os.environ.get("POSTGRES_DB", "neh")),
        user=os.environ.get("DB_USER", os.environ.get("POSTGRES_USER", "neh")),
        password=os.environ.get("DB_PASS", os.environ.get("POSTGRES_PASSWORD", "neh")),
    )


def set_statement_timeout(cur, timeout_ms: int) -> None:
    """
    Configure per-session statement timeout.

    Postgres semantics:
    - 0 means "no timeout"
    - otherwise milliseconds
    """
    timeout_ms = int(timeout_ms)
    if timeout_ms < 0:
        timeout_ms = 0
    # SET does not accept a parameter placeholder reliably across drivers.
    cur.execute(f"SET statement_timeout = {timeout_ms}")


@dataclass(frozen=True)
class EntityRow:
    id: int
    entity_type: str
    canonical_name: str


NON_CASCADING_REFS: List[Tuple[str, str, str]] = [
    # (table, column, meaning)
    ("speaker_map", "entity_id", "speaker mapping"),
    ("entity_alias_overrides", "forced_entity_id", "human override (force)"),
    ("entity_alias_overrides", "banned_entity_id", "human override (ban entity)"),
    ("mention_candidates", "anchored_to_entity_id", "OCR anchoring hint"),
    ("ocr_variant_clusters", "canonical_entity_id", "OCR canonical entity"),
    ("ocr_cluster_variants", "current_entity_id", "OCR variant linked entity"),
]

INDEX_RECOMMENDATIONS: List[Tuple[str, str]] = [
    # (index_name, create_sql)
    (
        "idx_mention_candidates_resolved_entity_id",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mention_candidates_resolved_entity_id "
        "ON mention_candidates(resolved_entity_id) WHERE resolved_entity_id IS NOT NULL",
    ),
    (
        "idx_mention_candidates_anchored_to_entity_id",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mention_candidates_anchored_to_entity_id "
        "ON mention_candidates(anchored_to_entity_id) WHERE anchored_to_entity_id IS NOT NULL",
    ),
    (
        "idx_entity_mentions_entity_id_only",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_mentions_entity_id_only "
        "ON entity_mentions(entity_id)",
    ),
]


def _table_exists(cur, table: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return cur.fetchone() is not None


def _column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def ensure_recommended_indexes() -> None:
    """
    Optional optimization step for large corpora.

    Uses CREATE INDEX CONCURRENTLY, so it cannot run inside a transaction block.
    """
    conn = get_conn()
    try:
        # CREATE INDEX CONCURRENTLY requires autocommit and cannot run in a tx.
        conn.autocommit = True

        with conn.cursor() as cur:
            for table in ("mention_candidates", "entity_mentions"):
                if not _table_exists(cur, table):
                    return

        print("Ensuring helpful indexes (CONCURRENTLY). This may take a while on large tables...", flush=True)
        with conn.cursor() as cur:
            # Avoid statement_timeout during index builds.
            set_statement_timeout(cur, 0)
            for idx_name, sql in INDEX_RECOMMENDATIONS:
                # Only attempt indexes when referenced columns exist.
                if "anchored_to_entity_id" in sql:
                    if not _column_exists(cur, "mention_candidates", "anchored_to_entity_id"):
                        continue
                t0 = time.monotonic()
                print(f"- creating {idx_name} ...", flush=True)
                try:
                    cur.execute(sql)
                except Exception as e:
                    # Don't fail the whole run; index creation may fail due to perms.
                    print(f"  WARNING: could not create {idx_name}: {e}", flush=True)
                else:
                    dt = time.monotonic() - t0
                    print(f"  done in {dt/60:.1f}m", flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _blocking_ref_sources(cur) -> List[Tuple[str, str, str]]:
    """
    Filter NON_CASCADING_REFS to only those that exist in this DB.
    """
    refs: List[Tuple[str, str, str]] = []
    for table, column, meaning in NON_CASCADING_REFS:
        if _table_exists(cur, table) and _column_exists(cur, table, column):
            refs.append((table, column, meaning))
    return refs


def fetch_candidates_with_flags(
    cur,
    *,
    entity_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[int, int, int, List[Dict]]:
    """
    Single set-based query:
    - Finds "0 evidence" entities (no entity_mentions and no resolved mention_candidates)
    - Flags whether they are blocked by any non-cascading references

    Returns:
      (total_candidates, total_deletable, total_blocked, sample_rows)
    """
    refs = _blocking_ref_sources(cur)

    # Build EXISTS(...) blocked flags dynamically based on available tables/columns.
    blocked_terms: List[str] = []
    for table, column, _meaning in refs:
        blocked_terms.append(f"EXISTS (SELECT 1 FROM {table} r WHERE r.{column} = e.id)")
    blocked_expr = " OR ".join(blocked_terms) if blocked_terms else "FALSE"

    params: List[object] = []
    etype_sql = ""
    if entity_type:
        etype_sql = "AND e.entity_type = %s"
        params.append(entity_type)

    limit_sql = ""
    if limit:
        limit_sql = "LIMIT %s"
        params.append(limit)

    # Performance note:
    # - We keep NOT EXISTS here so LIMIT can short-circuit when indexes exist.
    # - On huge DBs with poor indexes, this can still be slow; use --ensure-indexes
    #   and/or --statement-timeout-ms 0.
    cur.execute(
        f"""
        WITH candidates AS (
            SELECT e.id, e.entity_type, e.canonical_name
            FROM entities e
            WHERE NOT EXISTS (SELECT 1 FROM entity_mentions em WHERE em.entity_id = e.id)
              AND NOT EXISTS (SELECT 1 FROM mention_candidates mc WHERE mc.resolved_entity_id = e.id)
              {etype_sql}
            ORDER BY e.id
            {limit_sql}
        )
        SELECT
            e.id,
            e.entity_type,
            e.canonical_name,
            ({blocked_expr}) AS is_blocked
        FROM candidates e
        ORDER BY e.id
        """,
        tuple(params),
    )
    rows = cur.fetchall()

    total_candidates = len(rows)
    total_blocked = sum(1 for r in rows if r["is_blocked"])
    total_deletable = total_candidates - total_blocked
    return total_candidates, total_deletable, total_blocked, rows


def fetch_alias_counts_for_entities(cur, entity_ids: Sequence[int]) -> Dict[int, int]:
    """
    Fetch alias counts for a small set of entities in one query.
    """
    if not entity_ids:
        return {}
    cur.execute(
        """
        SELECT entity_id, COUNT(*)::int AS alias_count
        FROM entity_aliases
        WHERE entity_id = ANY(%s)
        GROUP BY entity_id
        """,
        (list(entity_ids),),
    )
    return {int(r[0]): int(r[1]) for r in cur.fetchall()}


def count_entity_aliases(cur, entity_id: int) -> int:
    cur.execute("SELECT COUNT(*) FROM entity_aliases WHERE entity_id = %s", (entity_id,))
    return int(cur.fetchone()[0])


def find_blocking_references(cur, entity_id: int) -> List[str]:
    """
    Return human-readable reasons why we should not delete this entity automatically.
    """
    reasons: List[str] = []

    for table, column, meaning in NON_CASCADING_REFS:
        if not _table_exists(cur, table) or not _column_exists(cur, table, column):
            continue
        cur.execute(f"SELECT 1 FROM {table} WHERE {column} = %s LIMIT 1", (entity_id,))
        if cur.fetchone() is not None:
            reasons.append(f"{table}.{column} ({meaning})")

    return reasons


def delete_entities(cur, entity_ids: Sequence[int]) -> int:
    if not entity_ids:
        return 0
    cur.execute("DELETE FROM entities WHERE id = ANY(%s)", (list(entity_ids),))
    return cur.rowcount


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune entities with zero mentions")
    ap.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")
    ap.add_argument(
        "--ensure-indexes",
        action="store_true",
        help="Create helpful indexes (CONCURRENTLY) to speed up scans on huge tables",
    )
    ap.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=0,
        help="Override Postgres statement_timeout for this script (0 = no timeout). Default: 0",
    )
    ap.add_argument("--limit", type=int, default=None, help="Limit number of candidate entities scanned")
    ap.add_argument(
        "--entity-type",
        default=None,
        help="Filter: person|org|place (optional)",
    )
    ap.add_argument("--show", type=int, default=30, help="How many candidates to print (dry-run)")
    args = ap.parse_args()

    if args.entity_type and args.entity_type not in ("person", "org", "place"):
        print("ERROR: --entity-type must be one of: person, org, place", file=sys.stderr)
        return 2

    if args.ensure_indexes:
        ensure_recommended_indexes()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            set_statement_timeout(cur, args.statement_timeout_ms)
            # Ensure required tables exist.
            for t in ("entities", "entity_mentions", "mention_candidates", "entity_aliases"):
                if not _table_exists(cur, t):
                    print(f"ERROR: required table missing: {t}", file=sys.stderr)
                    return 2

        print("Scanning for 0-mention entities (this can take time on large corpora)...", flush=True)
        scan_t0 = time.monotonic()

        # For dry-run, we only need alias counts for the first N shown, not for all candidates.
        sample_n = max(0, int(args.show))
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            set_statement_timeout(cur, args.statement_timeout_ms)
            # First pass: get counts without alias counts to keep it cheap.
            total_candidates, total_deletable, total_blocked, rows = fetch_candidates_with_flags(
                cur,
                entity_type=args.entity_type,
                limit=args.limit,
            )

        if total_candidates == 0:
            print("No zero-mention entities found.")
            return 0

        dt = time.monotonic() - scan_t0
        print(f"Candidates (0 mentions): {total_candidates} (scan {dt/60:.1f}m)", flush=True)
        print(f"Deletable now:          {total_deletable}", flush=True)
        print(f"Blocked (refs):         {total_blocked}", flush=True)

        if not args.apply:
            print("\n[DRY RUN] Showing candidates (first N):")
            sample_ids = [int(r["id"]) for r in rows[:sample_n]]
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                set_statement_timeout(cur, args.statement_timeout_ms)
                alias_map = fetch_alias_counts_for_entities(cur, sample_ids)

                # For reasons, only compute per-entity for the small sample shown.
                refs = _blocking_ref_sources(cur)
                for r in rows[:sample_n]:
                    eid = int(r["id"])
                    is_blocked = bool(r["is_blocked"])
                    alias_ct = alias_map.get(eid, 0)
                    reasons: List[str] = []
                    if is_blocked and refs:
                        for table, column, meaning in refs:
                            cur.execute(f"SELECT 1 FROM {table} WHERE {column}=%s LIMIT 1", (eid,))
                            if cur.fetchone() is not None:
                                reasons.append(f"{table}.{column} ({meaning})")
                    status = "BLOCKED" if is_blocked else "OK"
                    reason_str = f" | refs: {', '.join(reasons)}" if reasons else ""
                    print(
                        f"- {status} entity_id={eid} type={r['entity_type']} aliases={alias_ct} "
                        f"name={r['canonical_name']!r}{reason_str}"
                    )
            print("\nRun with --apply to delete only the OK entries.")
            return 0

        if total_deletable == 0:
            print("Nothing deletable (all candidates are blocked by references).")
            return 0

        # Apply deletes in a single transaction.
        print("Deleting deletable entities in one transaction...", flush=True)
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                set_statement_timeout(cur, args.statement_timeout_ms)
                # Delete via a set-based DELETE to avoid pulling huge ID lists into Python.
                refs = _blocking_ref_sources(cur)
                blocked_terms: List[str] = []
                for table, column, _meaning in refs:
                    blocked_terms.append(f"EXISTS (SELECT 1 FROM {table} r WHERE r.{column} = e.id)")
                blocked_expr = " OR ".join(blocked_terms) if blocked_terms else "FALSE"

                params: List[object] = []
                etype_sql = ""
                if args.entity_type:
                    etype_sql = "AND e.entity_type = %s"
                    params.append(args.entity_type)

                limit_sql = ""
                if args.limit:
                    limit_sql = "LIMIT %s"
                    params.append(args.limit)

                cur.execute(
                    f"""
                    WITH candidates AS (
                        SELECT e.id
                        FROM entities e
                        WHERE NOT EXISTS (SELECT 1 FROM entity_mentions em WHERE em.entity_id = e.id)
                          AND NOT EXISTS (SELECT 1 FROM mention_candidates mc WHERE mc.resolved_entity_id = e.id)
                          {etype_sql}
                        ORDER BY e.id
                        {limit_sql}
                    ),
                    deletable AS (
                        SELECT e.id
                        FROM candidates c
                        JOIN entities e ON e.id = c.id
                        WHERE NOT ({blocked_expr})
                    )
                    DELETE FROM entities
                    WHERE id IN (SELECT id FROM deletable)
                    RETURNING id
                    """,
                    tuple(params),
                )
                deleted_ids = cur.fetchall()
                print(f"Deleted entities: {len(deleted_ids)}", flush=True)
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

