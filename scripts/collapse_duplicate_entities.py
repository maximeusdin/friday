#!/usr/bin/env python3
"""
Collapse duplicate entities with the same canonical_name.

For each group of entities sharing the same canonical_name:
- Pick the "winner" (most aliases, then most mentions, then lowest id)
- Transfer any entity_mentions from losers to winner
- Transfer any mention_candidates.resolved_entity_id from losers to winner
- Delete the loser entities (CASCADE handles entity_aliases)

Default is DRY RUN. Use --apply to actually merge/delete.

Usage:
    python scripts/collapse_duplicate_entities.py
    python scripts/collapse_duplicate_entities.py --apply
    python scripts/collapse_duplicate_entities.py --limit 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", os.environ.get("POSTGRES_HOST", "localhost")),
        port=int(os.environ.get("DB_PORT", os.environ.get("POSTGRES_PORT", "5432"))),
        dbname=os.environ.get("DB_NAME", os.environ.get("POSTGRES_DB", "neh")),
        user=os.environ.get("DB_USER", os.environ.get("POSTGRES_USER", "neh")),
        password=os.environ.get("DB_PASS", os.environ.get("POSTGRES_PASSWORD", "neh")),
    )


def set_statement_timeout(cur, timeout_ms: int) -> None:
    timeout_ms = int(timeout_ms)
    if timeout_ms < 0:
        timeout_ms = 0
    cur.execute(f"SET statement_timeout = {timeout_ms}")


@dataclass
class EntityInfo:
    id: int
    canonical_name: str
    entity_type: str
    alias_count: int
    mention_count: int


def find_duplicate_groups(cur, case_insensitive: bool = True) -> Dict[str, List[EntityInfo]]:
    """
    Find all canonical_names that have multiple entity rows.
    Returns dict: canonical_name -> list of EntityInfo (sorted by preference).
    
    If case_insensitive=True, groups "ANDREEV" and "Andreev" together.
    """
    print("Step 1/3: Querying entity stats (may take a moment on large DBs)...", flush=True)
    print(f"  Case-insensitive grouping: {case_insensitive}", flush=True)
    t0 = time.monotonic()
    
    # Use LOWER() for case-insensitive grouping
    name_expr = "LOWER(e.canonical_name)" if case_insensitive else "e.canonical_name"
    name_expr_stats = "LOWER(canonical_name)" if case_insensitive else "canonical_name"
    
    cur.execute(f"""
        WITH entity_stats AS (
            SELECT 
                e.id,
                e.canonical_name,
                {name_expr} AS canonical_name_grouped,
                e.entity_type,
                COALESCE((SELECT COUNT(*) FROM entity_aliases ea WHERE ea.entity_id = e.id), 0) AS alias_count,
                COALESCE((SELECT COUNT(*) FROM entity_mentions em WHERE em.entity_id = e.id), 0) AS mention_count
            FROM entities e
        ),
        duplicated_names AS (
            SELECT {name_expr_stats} AS name_grouped
            FROM entity_stats
            GROUP BY {name_expr_stats}
            HAVING COUNT(*) > 1
        )
        SELECT es.id, es.canonical_name, es.canonical_name_grouped, es.entity_type, es.alias_count, es.mention_count
        FROM entity_stats es
        JOIN duplicated_names dn ON dn.name_grouped = es.canonical_name_grouped
        ORDER BY es.canonical_name_grouped, es.alias_count DESC, es.mention_count DESC, es.id ASC
    """)
    
    dt = time.monotonic() - t0
    print(f"  Query completed in {dt:.1f}s", flush=True)
    
    print("Step 2/3: Fetching results...", flush=True)
    t0 = time.monotonic()
    rows = cur.fetchall()
    dt = time.monotonic() - t0
    print(f"  Fetched {len(rows)} rows in {dt:.1f}s", flush=True)
    
    print("Step 3/3: Building duplicate groups...", flush=True)
    t0 = time.monotonic()
    groups: Dict[str, List[EntityInfo]] = defaultdict(list)
    for i, row in enumerate(rows):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(rows)} rows...", flush=True)
        info = EntityInfo(
            id=row["id"],
            canonical_name=row["canonical_name"],
            entity_type=row["entity_type"],
            alias_count=row["alias_count"],
            mention_count=row["mention_count"],
        )
        # Use the grouped key (lowercase if case-insensitive)
        group_key = row.get("canonical_name_grouped", row["canonical_name"])
        groups[group_key].append(info)
    
    dt = time.monotonic() - t0
    print(f"  Built {len(groups)} groups in {dt:.1f}s", flush=True)
    
    return dict(groups)


def transfer_mentions(cur, from_entity_id: int, to_entity_id: int) -> int:
    """
    Transfer entity_mentions from one entity to another.
    Uses ON CONFLICT to handle cases where the same (entity_id, chunk_id, surface) might exist.
    Returns count of rows updated.
    """
    # First try simple update
    cur.execute("""
        UPDATE entity_mentions
        SET entity_id = %s
        WHERE entity_id = %s
        AND NOT EXISTS (
            SELECT 1 FROM entity_mentions em2
            WHERE em2.entity_id = %s
            AND em2.chunk_id = entity_mentions.chunk_id
            AND em2.surface = entity_mentions.surface
        )
    """, (to_entity_id, from_entity_id, to_entity_id))
    updated = cur.rowcount
    
    # Delete any remaining (would be duplicates)
    cur.execute("DELETE FROM entity_mentions WHERE entity_id = %s", (from_entity_id,))
    
    return updated


def transfer_resolved_candidates(cur, from_entity_id: int, to_entity_id: int) -> int:
    """
    Transfer mention_candidates.resolved_entity_id from one entity to another.
    """
    cur.execute("""
        UPDATE mention_candidates
        SET resolved_entity_id = %s
        WHERE resolved_entity_id = %s
    """, (to_entity_id, from_entity_id))
    return cur.rowcount


def transfer_other_references(cur, from_entity_id: int, to_entity_id: int) -> Dict[str, int]:
    """
    Transfer other entity references that might exist.
    Returns dict of table -> count updated.
    """
    results = {}
    
    # Tables with entity_id references that should be transferred
    transfer_tables = [
        ("entity_alias_preferred", "preferred_entity_id"),
        ("entity_alias_overrides", "forced_entity_id"),
        ("entity_alias_overrides", "banned_entity_id"),
        ("document_anchors", "entity_id"),
        ("ocr_variant_allowlist", "entity_id"),
        ("ocr_variant_clusters", "canonical_entity_id"),
        ("speaker_map", "entity_id"),
    ]
    
    for table, column in transfer_tables:
        try:
            cur.execute(f"""
                UPDATE {table}
                SET {column} = %s
                WHERE {column} = %s
            """, (to_entity_id, from_entity_id))
            if cur.rowcount > 0:
                results[f"{table}.{column}"] = cur.rowcount
        except Exception:
            # Table might not exist
            pass
    
    return results


def delete_entity(cur, entity_id: int) -> bool:
    """Delete an entity (CASCADE handles aliases)."""
    cur.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
    return cur.rowcount > 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Collapse duplicate entities by canonical_name")
    ap.add_argument("--apply", action="store_true", help="Actually merge/delete (default: dry-run)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of groups to process")
    ap.add_argument("--show", type=int, default=20, help="How many groups to show in dry-run")
    ap.add_argument("--statement-timeout-ms", type=int, default=0, help="Postgres statement timeout (0=none)")
    ap.add_argument("--case-sensitive", action="store_true", help="Use case-sensitive matching (default: case-insensitive)")
    args = ap.parse_args()
    
    case_insensitive = not args.case_sensitive
    
    print("=" * 60, flush=True)
    print("COLLAPSE DUPLICATE ENTITIES", flush=True)
    print("=" * 60, flush=True)
    print(f"  Mode: {'APPLY (will modify DB)' if args.apply else 'DRY RUN (read-only)'}")
    print(f"  Case matching: {'case-sensitive' if args.case_sensitive else 'case-insensitive'}")
    print(f"  Statement timeout: {args.statement_timeout_ms}ms (0=none)")
    if args.limit:
        print(f"  Limit: {args.limit} groups")
    print("", flush=True)
    
    print("Connecting to database...", flush=True)
    conn = get_conn()
    print("  Connected!", flush=True)
    print("", flush=True)
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            set_statement_timeout(cur, args.statement_timeout_ms)
            
            groups = find_duplicate_groups(cur, case_insensitive=case_insensitive)
        
        if not groups:
            print("No duplicate entities found.")
            return 0
        
        total_groups = len(groups)
        total_entities = sum(len(v) for v in groups.values())
        total_losers = total_entities - total_groups  # One winner per group
        
        print(f"\nFound {total_groups} duplicate groups ({total_entities} entities total)")
        print(f"Will merge {total_losers} entities into {total_groups} winners\n")
        
        if not args.apply:
            print("=" * 60, flush=True)
            print("[DRY RUN] Sample of duplicate groups:", flush=True)
            print("=" * 60, flush=True)
            print("", flush=True)
            
            shown = 0
            for group_key, entities in list(groups.items())[:args.show]:
                winner = entities[0]
                losers = entities[1:]
                # Show all the actual names in the group
                all_names = sorted(set(e.canonical_name for e in entities))
                names_str = ", ".join(f"'{n}'" for n in all_names)
                print(f"Group [{group_key}]: {names_str} ({len(entities)} entities)")
                print(f"  WINNER: id={winner.id} name='{winner.canonical_name}' type={winner.entity_type} aliases={winner.alias_count} mentions={winner.mention_count}")
                for loser in losers[:3]:
                    print(f"  LOSER:  id={loser.id} name='{loser.canonical_name}' type={loser.entity_type} aliases={loser.alias_count} mentions={loser.mention_count}")
                if len(losers) > 3:
                    print(f"  ... and {len(losers) - 3} more losers")
                print()
                shown += 1
            
            if total_groups > args.show:
                print(f"... and {total_groups - args.show} more groups\n")
            
            print("=" * 60, flush=True)
            print("SUMMARY", flush=True)
            print("=" * 60, flush=True)
            print(f"  Total duplicate groups: {total_groups}")
            print(f"  Total entities involved: {total_entities}")
            print(f"  Entities to DELETE: {total_losers}")
            print(f"  Entities to KEEP (winners): {total_groups}")
            print("")
            print("Run with --apply to merge duplicates.")
            return 0
        
        # Apply merges
        print("=" * 60, flush=True)
        print("APPLYING MERGES", flush=True)
        print("=" * 60, flush=True)
        
        stats = {
            "groups_processed": 0,
            "entities_deleted": 0,
            "mentions_transferred": 0,
            "candidates_transferred": 0,
            "other_refs_transferred": 0,
        }
        
        groups_to_process = list(groups.items())
        if args.limit:
            groups_to_process = groups_to_process[:args.limit]
        
        total_losers_to_delete = sum(len(entities) - 1 for _, entities in groups_to_process)
        print(f"Groups to process: {len(groups_to_process)}", flush=True)
        print(f"Entities to delete: {total_losers_to_delete}", flush=True)
        print("", flush=True)
        
        t0 = time.monotonic()
        last_progress_time = t0
        entities_deleted_since_last = 0
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            set_statement_timeout(cur, args.statement_timeout_ms)
            
            for i, (canonical_name, entities) in enumerate(groups_to_process, start=1):
                winner = entities[0]
                losers = entities[1:]
                
                # Progress: show every 50 groups, or every 30 seconds, or at start/end
                now = time.monotonic()
                should_log = (
                    i == 1 or 
                    i == len(groups_to_process) or 
                    i % 50 == 0 or 
                    (now - last_progress_time) >= 30
                )
                
                if should_log:
                    elapsed = now - t0
                    rate = stats["entities_deleted"] / elapsed if elapsed > 0 else 0
                    remaining = total_losers_to_delete - stats["entities_deleted"]
                    eta = remaining / rate if rate > 0 else 0
                    print(
                        f"[{i}/{len(groups_to_process)}] "
                        f"Deleted {stats['entities_deleted']}/{total_losers_to_delete} entities "
                        f"({elapsed:.0f}s elapsed, {rate:.1f}/s, ETA {eta:.0f}s)",
                        flush=True,
                    )
                    last_progress_time = now
                
                for loser in losers:
                    # Transfer mentions
                    mentions_moved = transfer_mentions(cur, loser.id, winner.id)
                    stats["mentions_transferred"] += mentions_moved
                    
                    # Transfer resolved candidates
                    candidates_moved = transfer_resolved_candidates(cur, loser.id, winner.id)
                    stats["candidates_transferred"] += candidates_moved
                    
                    # Transfer other references
                    other_refs = transfer_other_references(cur, loser.id, winner.id)
                    stats["other_refs_transferred"] += sum(other_refs.values())
                    
                    # Delete the loser
                    if delete_entity(cur, loser.id):
                        stats["entities_deleted"] += 1
                
                stats["groups_processed"] += 1
                
                # Commit periodically
                if i % 1000 == 0:
                    print(f"  [commit] Committing batch at group {i}...", flush=True)
                    conn.commit()
            
            # Final commit
            print("  [commit] Final commit...", flush=True)
            conn.commit()
        
        elapsed = time.monotonic() - t0
        print("", flush=True)
        print("=" * 60, flush=True)
        print("COMPLETE", flush=True)
        print("=" * 60, flush=True)
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Groups processed: {stats['groups_processed']}")
        print(f"  Entities deleted: {stats['entities_deleted']}")
        print(f"  Mentions transferred: {stats['mentions_transferred']}")
        print(f"  Candidates transferred: {stats['candidates_transferred']}")
        print(f"  Other refs transferred: {stats['other_refs_transferred']}")
        
        return 0
        
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
