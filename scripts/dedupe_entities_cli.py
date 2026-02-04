#!/usr/bin/env python3
"""
CLI for identifying and merging duplicate entities.

Finds duplicates by:
1. Exact canonical_name match
2. Similar canonical_name (fuzzy)
3. Shared alias_norm (multiple entities have same alias)

Merge operation:
- Moves all aliases from duplicate -> canonical
- Moves all entity_mentions from duplicate -> canonical
- Moves all entity_links from duplicate -> canonical
- Moves all entity_relationships from duplicate -> canonical
- Records the merge in entity_merges table
- Deletes or marks the duplicate entity

Usage:
    python scripts/dedupe_entities_cli.py --mode exact
    python scripts/dedupe_entities_cli.py --mode shared-alias
    python scripts/dedupe_entities_cli.py --mode fuzzy --threshold 0.8
    python scripts/dedupe_entities_cli.py --entity-type place  # Only places
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor


def get_conn():
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'localhost'),
        port=int(os.environ.get('POSTGRES_PORT', '5432')),
        dbname=os.environ.get('POSTGRES_DB', 'neh'),
        user=os.environ.get('POSTGRES_USER', 'neh'),
        password=os.environ.get('POSTGRES_PASSWORD', 'neh')
    )


@dataclass
class EntityInfo:
    id: int
    canonical_name: str
    entity_type: Optional[str]
    alias_count: int
    mention_count: int
    aliases: List[str]


def ensure_merge_table(conn) -> None:
    """Create entity_merges table if it doesn't exist."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_merges (
            id BIGSERIAL PRIMARY KEY,
            source_entity_id BIGINT NOT NULL,
            target_entity_id BIGINT NOT NULL,
            source_name TEXT,
            target_name TEXT,
            reason TEXT,
            merged_at TIMESTAMPTZ DEFAULT NOW(),
            merged_by TEXT DEFAULT 'cli_dedupe'
        )
    """)
    conn.commit()


def get_entity_info(conn, entity_id: int) -> Optional[EntityInfo]:
    """Get detailed info about an entity."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT 
            e.id,
            e.canonical_name,
            e.entity_type,
            (SELECT COUNT(*) FROM entity_aliases WHERE entity_id = e.id) as alias_count,
            (SELECT COUNT(*) FROM entity_mentions WHERE entity_id = e.id) as mention_count
        FROM entities e
        WHERE e.id = %s
    """, (entity_id,))
    
    row = cur.fetchone()
    if not row:
        return None
    
    # Get aliases
    cur.execute("""
        SELECT alias FROM entity_aliases WHERE entity_id = %s ORDER BY alias
    """, (entity_id,))
    aliases = [r['alias'] for r in cur.fetchall()]
    
    return EntityInfo(
        id=row['id'],
        canonical_name=row['canonical_name'],
        entity_type=row['entity_type'],
        alias_count=row['alias_count'],
        mention_count=row['mention_count'],
        aliases=aliases,
    )


def find_exact_name_duplicates(
    conn,
    entity_type: Optional[str] = None,
    limit: int = 100,
) -> List[Tuple[str, List[int]]]:
    """Find entities with exactly matching canonical_name."""
    cur = conn.cursor()
    
    type_filter = "AND e.entity_type = %s" if entity_type else ""
    params = [entity_type] if entity_type else []
    
    cur.execute(f"""
        SELECT canonical_name, array_agg(id ORDER BY id) as entity_ids
        FROM entities e
        WHERE canonical_name IS NOT NULL
        {type_filter}
        GROUP BY canonical_name
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC, canonical_name
        LIMIT %s
    """, params + [limit])
    
    return [(row[0], list(row[1])) for row in cur.fetchall()]


def find_shared_alias_duplicates(
    conn,
    entity_type: Optional[str] = None,
    limit: int = 100,
) -> List[Tuple[str, List[int]]]:
    """Find entities that share the same alias_norm."""
    cur = conn.cursor()
    
    type_filter = "AND e.entity_type = %s" if entity_type else ""
    params = [entity_type] if entity_type else []
    
    cur.execute(f"""
        SELECT ea.alias_norm, array_agg(DISTINCT ea.entity_id ORDER BY ea.entity_id) as entity_ids
        FROM entity_aliases ea
        JOIN entities e ON e.id = ea.entity_id
        WHERE ea.alias_norm IS NOT NULL
        {type_filter}
        GROUP BY ea.alias_norm
        HAVING COUNT(DISTINCT ea.entity_id) > 1
        ORDER BY COUNT(DISTINCT ea.entity_id) DESC, ea.alias_norm
        LIMIT %s
    """, params + [limit])
    
    return [(row[0], list(row[1])) for row in cur.fetchall()]


def find_fuzzy_name_duplicates(
    conn,
    threshold: float = 0.8,
    entity_type: Optional[str] = None,
    limit: int = 100,
) -> List[Tuple[str, str, float, int, int]]:
    """Find entities with similar canonical_name using trigram similarity."""
    cur = conn.cursor()
    
    type_filter = "AND e1.entity_type = %s AND e2.entity_type = %s" if entity_type else ""
    params = [entity_type, entity_type] if entity_type else []
    
    cur.execute(f"""
        SELECT 
            e1.canonical_name as name1,
            e2.canonical_name as name2,
            similarity(e1.canonical_name, e2.canonical_name) as sim,
            e1.id as id1,
            e2.id as id2
        FROM entities e1
        JOIN entities e2 ON e1.id < e2.id
        WHERE similarity(e1.canonical_name, e2.canonical_name) >= %s
        {type_filter}
        ORDER BY sim DESC
        LIMIT %s
    """, [threshold] + params + [limit])
    
    return [(row[0], row[1], float(row[2]), row[3], row[4]) for row in cur.fetchall()]


def delete_entity(conn, entity_id: int) -> Dict[str, int]:
    """
    Delete an entity entirely (not merge).
    
    Removes all associated data (aliases, mentions, etc.) via CASCADE or explicit delete.
    Returns dict with counts of deleted items.
    """
    cur = conn.cursor()
    stats = {
        'aliases_deleted': 0,
        'mentions_deleted': 0,
    }
    
    # Get name for logging
    cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
    row = cur.fetchone()
    if not row:
        return stats
    
    # Count what will be deleted
    cur.execute("SELECT COUNT(*) FROM entity_aliases WHERE entity_id = %s", (entity_id,))
    stats['aliases_deleted'] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM entity_mentions WHERE entity_id = %s", (entity_id,))
    stats['mentions_deleted'] = cur.fetchone()[0]
    
    # Delete aliases explicitly (in case no CASCADE)
    cur.execute("DELETE FROM entity_aliases WHERE entity_id = %s", (entity_id,))
    
    # Delete mentions explicitly
    cur.execute("DELETE FROM entity_mentions WHERE entity_id = %s", (entity_id,))
    
    # Clean up other references - use savepoints to handle missing tables gracefully
    def safe_execute(sql, params):
        try:
            cur.execute("SAVEPOINT safe_delete")
            cur.execute(sql, params)
            cur.execute("RELEASE SAVEPOINT safe_delete")
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT safe_delete")
    
    safe_execute("DELETE FROM entity_links WHERE entity_id = %s", (entity_id,))
    safe_execute("DELETE FROM entity_relationships WHERE source_entity_id = %s OR target_entity_id = %s", 
                 (entity_id, entity_id))
    safe_execute("UPDATE mention_candidates SET resolved_entity_id = NULL WHERE resolved_entity_id = %s", (entity_id,))
    safe_execute("DELETE FROM alias_lexicon_index WHERE entity_id = %s", (entity_id,))
    
    # Delete the entity
    cur.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
    
    conn.commit()
    return stats


def merge_entities(
    conn,
    source_id: int,
    target_id: int,
    reason: str = 'manual_merge',
) -> Dict[str, int]:
    """
    Merge source entity into target entity.
    
    Optimized version: uses INSERT..ON CONFLICT DO NOTHING + DELETE.
    This avoids memory-heavy NOT EXISTS / NOT IN anti-joins on large tables.
    Returns dict with counts of moved items.
    """
    cur = conn.cursor()
    stats = {
        'aliases_moved': 0,
        'mentions_moved': 0,
        'links_moved': 0,
        'relationships_moved': 0,
    }
    
    # Helper for optional table operations (uses savepoints to avoid transaction abort)
    def safe_execute(sql, params, count_result=False):
        try:
            cur.execute("SAVEPOINT merge_op")
            cur.execute(sql, params)
            result = cur.rowcount if count_result else 0
            cur.execute("RELEASE SAVEPOINT merge_op")
            return result
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT merge_op")
            return 0

    def get_table_columns(table_name: str) -> List[str]:
        """
        Return column names in order for a table.
        Works even if your schema has drifted (extra columns, etc.).
        """
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table_name,),
        )
        return [r[0] for r in cur.fetchall()]

    def copy_rows_insert_delete(table: str, id_col: str, source_fk_col: str, target_fk_col_value: int, source_fk_value: int) -> int:
        """
        Generic pattern:
          INSERT INTO table(target_fk_col, other_cols...)
          SELECT target_fk_value, other_cols...
          FROM table
          WHERE source_fk_col = source_fk_value
          ON CONFLICT DO NOTHING;
          DELETE FROM table WHERE source_fk_col = source_fk_value;

        Returns: estimated moved count (rows inserted OR already existed; see note below).
        """
        cols = get_table_columns(table)
        if id_col in cols:
            cols = [c for c in cols if c != id_col]

        # We must set FK to target; so exclude FK from the "copied" list and prepend it
        other_cols = [c for c in cols if c != source_fk_col]

        # If table doesn't have expected FK col, skip
        if source_fk_col not in get_table_columns(table):
            return 0

        insert_cols = [source_fk_col] + [c for c in other_cols if c != source_fk_col]
        select_exprs = ["%s"] + [f"s.{c}" for c in other_cols if c != source_fk_col]

        insert_sql = f"""
            INSERT INTO {table} ({", ".join(insert_cols)})
            SELECT {", ".join(select_exprs)}
            FROM {table} s
            WHERE s.{source_fk_col} = %s
            ON CONFLICT DO NOTHING
        """

        # Insert first (cheap), then delete source rows
        safe_execute(insert_sql, (target_fk_col_value, source_fk_value))
        deleted = safe_execute(f"DELETE FROM {table} WHERE {source_fk_col} = %s", (source_fk_value,), count_result=True)

        # We don't know exact inserted count without RETURNING; deleted count is a good proxy:
        # if we delete N source rows after inserting, we've effectively transferred/removed N.
        return deleted
    
    # Get names for logging
    cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (source_id,))
    row = cur.fetchone()
    source_name = row[0] if row else str(source_id)
    cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (target_id,))
    row = cur.fetchone()
    target_name = row[0] if row else str(target_id)
    
    # 1) Move aliases (insert+delete, conflict-safe)
    stats['aliases_moved'] = copy_rows_insert_delete(
        table="entity_aliases",
        id_col="id",
        source_fk_col="entity_id",
        target_fk_col_value=target_id,
        source_fk_value=source_id,
    )

    # 2) Move mentions (insert+delete, conflict-safe)
    stats['mentions_moved'] = copy_rows_insert_delete(
        table="entity_mentions",
        id_col="id",
        source_fk_col="entity_id",
        target_fk_col_value=target_id,
        source_fk_value=source_id,
    )
    
    # 3. Move entity_links (optional table)
    stats['links_moved'] = safe_execute(
        "UPDATE entity_links SET entity_id = %s WHERE entity_id = %s",
        (target_id, source_id), count_result=True
    )
    
    # 4. Move entity_relationships (optional table) - just move, ignore conflicts
    moved_src = safe_execute(
        "UPDATE entity_relationships SET source_entity_id = %s WHERE source_entity_id = %s",
        (target_id, source_id), count_result=True
    )
    moved_tgt = safe_execute(
        "UPDATE entity_relationships SET target_entity_id = %s WHERE target_entity_id = %s",
        (target_id, source_id), count_result=True
    )
    stats['relationships_moved'] = moved_src + moved_tgt
    safe_execute(
        "DELETE FROM entity_relationships WHERE source_entity_id = %s OR target_entity_id = %s",
        (source_id, source_id)
    )
    
    # 5. Update mention_candidates resolved_entity_id
    safe_execute(
        "UPDATE mention_candidates SET resolved_entity_id = %s WHERE resolved_entity_id = %s",
        (target_id, source_id)
    )
    
    # 6. alias_lexicon_index is derived. Remove stale rows for source; rebuild later if needed.
    safe_execute("DELETE FROM alias_lexicon_index WHERE entity_id = %s", (source_id,))
    
    # 7. Record the merge
    cur.execute("""
        INSERT INTO entity_merges (source_entity_id, target_entity_id, source_name, target_name, reason)
        VALUES (%s, %s, %s, %s, %s)
    """, (source_id, target_id, source_name, target_name, reason))
    
    # 8. Delete the source entity
    cur.execute("DELETE FROM entities WHERE id = %s", (source_id,))
    
    conn.commit()
    return stats


def display_duplicate_group(
    conn,
    key: str,
    entity_ids: List[int],
    idx: int,
    total: int,
    mode: str,
    max_display: int = 10,
) -> None:
    """Display a group of duplicate entities (limited preview)."""
    print("\n" + "=" * 70)
    print(f"[{idx + 1}/{total}]  {mode}: {key!r}")
    print(f"         ({len(entity_ids)} entities share this)")
    print("=" * 70)
    
    # Only show first max_display entities
    display_ids = entity_ids[:max_display]
    
    for i, eid in enumerate(display_ids, 1):
        info = get_entity_info(conn, eid)
        if not info:
            print(f"  {i}. (id={eid}) - NOT FOUND")
            continue
        
        print(f"\n  {i}. {info.canonical_name} (id={info.id})")
        print(f"     Type: {info.entity_type or 'unknown'}")
        print(f"     Aliases: {info.alias_count}  Mentions: {info.mention_count}")
        if info.aliases:
            alias_preview = ", ".join(info.aliases[:5])
            if len(info.aliases) > 5:
                alias_preview += f", ... (+{len(info.aliases) - 5} more)"
            print(f"     Aliases: {alias_preview}")
    
    if len(entity_ids) > max_display:
        print(f"\n  ... and {len(entity_ids) - max_display} more entities (use 'more' to see next batch)")
    
    print("\n" + "-" * 70)
    print("Commands:")
    print("  m N      - Merge all into entity N (moves aliases+mentions, then deletes others)")
    print("  k N      - Keep only entity N, DELETE others entirely (no merge)")
    print("  x N      - Delete just entity N (garbage entry)")
    print("  d        - Delete this alias from ALL entities (too ambiguous)")
    print("  s        - Skip this group")
    print("  i N      - Show more info about entity N")
    if len(entity_ids) > max_display:
        print("  more     - Show next batch of entities")
    print("  q        - Quit")


def display_fuzzy_pair(
    conn,
    name1: str,
    name2: str,
    sim: float,
    id1: int,
    id2: int,
    idx: int,
    total: int,
) -> None:
    """Display a fuzzy-matched pair of entities."""
    print("\n" + "=" * 70)
    print(f"[{idx + 1}/{total}]  Similarity: {sim:.2f}")
    print("=" * 70)
    
    for i, (name, eid) in enumerate([(name1, id1), (name2, id2)], 1):
        info = get_entity_info(conn, eid)
        if not info:
            print(f"  {i}. {name} (id={eid}) - NOT FOUND")
            continue
        
        print(f"\n  {i}. {info.canonical_name} (id={info.id})")
        print(f"     Type: {info.entity_type or 'unknown'}")
        print(f"     Aliases: {info.alias_count}  Mentions: {info.mention_count}")
        if info.aliases:
            alias_preview = ", ".join(info.aliases[:5])
            if len(info.aliases) > 5:
                alias_preview += f", ... (+{len(info.aliases) - 5} more)"
            print(f"     Aliases: {alias_preview}")
    
    print("\n" + "-" * 70)
    print("Commands:")
    print("  m 1      - Merge into entity 1 (delete 2)")
    print("  m 2      - Merge into entity 2 (delete 1)")
    print("  s        - Skip (not duplicates)")
    print("  q        - Quit")


def main():
    parser = argparse.ArgumentParser(description='CLI for deduplicating entities')
    parser.add_argument('--mode', choices=['exact', 'shared-alias', 'fuzzy'], default='shared-alias',
                        help='How to find duplicates (default: shared-alias)')
    parser.add_argument('--entity-type', help='Filter by entity type (person, org, place, etc.)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Similarity threshold for fuzzy mode (default: 0.8)')
    parser.add_argument('--limit', type=int, default=100,
                        help='Max duplicate groups to process')
    parser.add_argument('--max-group-size', type=int, default=20,
                        help='Skip groups with more than N entities (likely tags, not duplicates)')
    parser.add_argument('--min-group-size', type=int, default=2,
                        help='Only show groups with at least N entities')
    args = parser.parse_args()
    
    conn = get_conn()
    ensure_merge_table(conn)
    
    print(f"Finding duplicates (mode: {args.mode})...")
    
    stats = {
        'groups_processed': 0,
        'merged': 0,
        'skipped': 0,
        'entities_removed': 0,
    }
    
    if args.mode == 'exact':
        duplicates = find_exact_name_duplicates(conn, args.entity_type, args.limit * 5)
        # Filter by group size
        duplicates = [
            (name, ids) for name, ids in duplicates
            if args.min_group_size <= len(ids) <= args.max_group_size
        ][:args.limit]
        print(f"Found {len(duplicates)} groups with exact name matches ({args.min_group_size}-{args.max_group_size} entities each).")
        
        for idx, (name, entity_ids) in enumerate(duplicates):
            max_display = 10
            
            while True:
                display_duplicate_group(conn, name, entity_ids, idx, len(duplicates), "Exact name", max_display)
                
                try:
                    cmd = input("\n> ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    cmd = 'q'
                
                if cmd == 'q' or cmd == 'quit':
                    break
                
                if cmd == 's' or cmd == 'skip':
                    stats['skipped'] += 1
                    break
                
                if cmd == 'more':
                    max_display = min(max_display + 10, len(entity_ids))
                    continue
                
                if cmd.startswith('m '):
                    try:
                        keep_idx = int(cmd.split()[1]) - 1
                        if keep_idx < 0 or keep_idx >= len(entity_ids):
                            print(f"Invalid: choose 1-{len(entity_ids)}")
                            continue
                        
                        target_id = entity_ids[keep_idx]
                        source_ids = [eid for i, eid in enumerate(entity_ids) if i != keep_idx]
                        
                        for source_id in source_ids:
                            merge_stats = merge_entities(conn, source_id, target_id, reason='exact_name_match')
                            print(f"  Merged {source_id} -> {target_id}: {merge_stats}")
                            stats['entities_removed'] += 1
                        
                        stats['merged'] += 1
                        break
                    except (ValueError, IndexError) as e:
                        print(f"Error: {e}")
                        continue
                
                print(f"Unknown command: {cmd}")
            
            if cmd == 'q' or cmd == 'quit':
                break
            
            stats['groups_processed'] += 1
    
    elif args.mode == 'shared-alias':
        duplicates = find_shared_alias_duplicates(conn, args.entity_type, args.limit * 5)  # Fetch more, filter below
        # Filter by group size
        duplicates = [
            (alias, ids) for alias, ids in duplicates
            if args.min_group_size <= len(ids) <= args.max_group_size
        ][:args.limit]
        print(f"Found {len(duplicates)} aliases shared by {args.min_group_size}-{args.max_group_size} entities.")
        
        for idx, (alias, entity_ids) in enumerate(duplicates):
            display_offset = 0
            max_display = 10
            
            while True:
                display_duplicate_group(conn, alias, entity_ids, idx, len(duplicates), "Shared alias", max_display)
                
                try:
                    cmd = input("\n> ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    cmd = 'q'
                
                if cmd == 'q' or cmd == 'quit':
                    stats['groups_processed'] += 1
                    break
                
                if cmd == 's' or cmd == 'skip':
                    stats['skipped'] += 1
                    break
                
                if cmd == 'more':
                    # Show more entities by increasing max_display
                    max_display = min(max_display + 10, len(entity_ids))
                    continue
                
                if cmd == 'd' or cmd == 'delete':
                    # Delete this alias from all entities
                    cur = conn.cursor()
                    cur.execute(
                        "DELETE FROM entity_aliases WHERE alias_norm = %s RETURNING id",
                        (alias,)
                    )
                    deleted_count = len(cur.fetchall())
                    conn.commit()
                    print(f"Deleted alias '{alias}' from {deleted_count} entity_aliases rows.")
                    stats['aliases_deleted'] = stats.get('aliases_deleted', 0) + deleted_count
                    break
                
                if cmd.startswith('i '):
                    # Show more info
                    try:
                        info_idx = int(cmd.split()[1]) - 1
                        if info_idx < 0 or info_idx >= len(entity_ids):
                            print(f"Invalid: choose 1-{len(entity_ids)}")
                            continue
                        eid = entity_ids[info_idx]
                        info = get_entity_info(conn, eid)
                        if info:
                            print(f"\n--- Full info for {info.canonical_name} (id={info.id}) ---")
                            print(f"Type: {info.entity_type}")
                            print(f"Mentions: {info.mention_count}")
                            print(f"All aliases ({info.alias_count}):")
                            for a in info.aliases:
                                print(f"  - {a}")
                    except (ValueError, IndexError):
                        print("Invalid entity number")
                    continue
                
                if cmd.startswith('m '):
                    try:
                        keep_idx = int(cmd.split()[1]) - 1
                        if keep_idx < 0 or keep_idx >= len(entity_ids):
                            print(f"Invalid: choose 1-{len(entity_ids)}")
                            continue
                        
                        target_id = entity_ids[keep_idx]
                        source_ids = [eid for i, eid in enumerate(entity_ids) if i != keep_idx]
                        
                        for source_id in source_ids:
                            merge_stats = merge_entities(conn, source_id, target_id, reason='shared_alias')
                            print(f"  Merged {source_id} -> {target_id}: {merge_stats}")
                            stats['entities_removed'] += 1
                        
                        stats['merged'] += 1
                        break
                    except (ValueError, IndexError) as e:
                        print(f"Error: {e}")
                        continue
                
                if cmd.startswith('x '):
                    # Delete just entity N
                    try:
                        del_idx = int(cmd.split()[1]) - 1
                        if del_idx < 0 or del_idx >= len(entity_ids):
                            print(f"Invalid: choose 1-{len(entity_ids)}")
                            continue
                        
                        del_id = entity_ids[del_idx]
                        del_info = get_entity_info(conn, del_id)
                        
                        print(f"\nDeleting: {del_info.canonical_name} (id={del_id})")
                        print(f"  Aliases: {del_info.alias_count}, Mentions: {del_info.mention_count}")
                        confirm = input("Confirm? (y/n): ").strip().lower()
                        
                        if confirm == 'y':
                            del_stats = delete_entity(conn, del_id)
                            print(f"  Deleted: {del_stats}")
                            stats['entities_removed'] += 1
                            # Remove from list so display updates
                            entity_ids.pop(del_idx)
                            if len(entity_ids) <= 1:
                                print("Only 1 entity left in group, moving on.")
                                break
                            # Redisplay with updated list
                            max_display = 10
                            continue
                        else:
                            print("Cancelled.")
                            continue
                    except (ValueError, IndexError) as e:
                        print(f"Error: {e}")
                        continue
                
                if cmd.startswith('k '):
                    # Keep only N, delete all others entirely
                    try:
                        keep_idx = int(cmd.split()[1]) - 1
                        if keep_idx < 0 or keep_idx >= len(entity_ids):
                            print(f"Invalid: choose 1-{len(entity_ids)}")
                            continue
                        
                        keep_id = entity_ids[keep_idx]
                        delete_ids = [eid for i, eid in enumerate(entity_ids) if i != keep_idx]
                        
                        # Confirm for safety
                        keep_info = get_entity_info(conn, keep_id)
                        print(f"\nKeeping: {keep_info.canonical_name} (id={keep_id})")
                        print(f"Deleting {len(delete_ids)} entities entirely (not merging).")
                        confirm = input("Confirm? (y/n): ").strip().lower()
                        
                        if confirm == 'y':
                            for del_id in delete_ids:
                                del_info = get_entity_info(conn, del_id)
                                del_stats = delete_entity(conn, del_id)
                                print(f"  Deleted {del_id} ({del_info.canonical_name if del_info else '?'}): {del_stats}")
                                stats['entities_removed'] += 1
                            stats['kept'] = stats.get('kept', 0) + 1
                            break
                        else:
                            print("Cancelled.")
                            continue
                    except (ValueError, IndexError) as e:
                        print(f"Error: {e}")
                        continue
                
                print(f"Unknown command: {cmd}")
            
            if cmd == 'q' or cmd == 'quit':
                break
            
            stats['groups_processed'] += 1
    
    elif args.mode == 'fuzzy':
        pairs = find_fuzzy_name_duplicates(conn, args.threshold, args.entity_type, args.limit)
        print(f"Found {len(pairs)} fuzzy-matched pairs (threshold={args.threshold}).")
        
        for idx, (name1, name2, sim, id1, id2) in enumerate(pairs):
            # Check if entities still exist (might have been merged)
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM entities WHERE id = %s", (id1,))
            if not cur.fetchone():
                continue
            cur.execute("SELECT 1 FROM entities WHERE id = %s", (id2,))
            if not cur.fetchone():
                continue
            
            display_fuzzy_pair(conn, name1, name2, sim, id1, id2, idx, len(pairs))
            
            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                cmd = 'q'
            
            if cmd == 'q' or cmd == 'quit':
                break
            
            if cmd == 's' or cmd == 'skip':
                stats['skipped'] += 1
                continue
            
            if cmd == 'm 1':
                merge_stats = merge_entities(conn, id2, id1, reason='fuzzy_name_match')
                print(f"  Merged {id2} -> {id1}: {merge_stats}")
                stats['merged'] += 1
                stats['entities_removed'] += 1
            elif cmd == 'm 2':
                merge_stats = merge_entities(conn, id1, id2, reason='fuzzy_name_match')
                print(f"  Merged {id1} -> {id2}: {merge_stats}")
                stats['merged'] += 1
                stats['entities_removed'] += 1
            else:
                print(f"Unknown command: {cmd}")
                continue
            
            stats['groups_processed'] += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Groups processed: {stats['groups_processed']}")
    print(f"Merged:           {stats['merged']}")
    if stats.get('kept'):
        print(f"Kept (others deleted): {stats['kept']}")
    print(f"Skipped:          {stats['skipped']}")
    print(f"Entities removed: {stats['entities_removed']}")
    if stats.get('aliases_deleted'):
        print(f"Aliases deleted:  {stats['aliases_deleted']}")
    print("=" * 70)
    
    conn.close()


if __name__ == '__main__':
    main()
