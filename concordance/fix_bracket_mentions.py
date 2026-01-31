#!/usr/bin/env python3
"""
Fix entity mentions that are missing closing brackets.

This script:
1. Finds mentions with surface text ending with '[' but not ']'
2. Checks if the chunk text has the full version with closing bracket
3. Updates the surface text to include the closing bracket
4. Optionally deletes and re-extracts mentions for affected entities
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import DictCursor
from psycopg2 import sql

def get_db_connection():
    """Get database connection using environment variables or defaults."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "neh")
    db_user = os.getenv("DB_USER", "neh")
    db_pass = os.getenv("DB_PASS", "neh")
    return psycopg2.connect(host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_pass)


def fix_bracket_mentions(conn, entity_id: int = None, dry_run: bool = False):
    """
    Fix mentions that are missing closing brackets.
    
    If entity_id is provided, only fix mentions for that entity.
    """
    cur = conn.cursor()
    
    try:
        # Find mentions with surface text ending with '[' but not ']'
        base_query = """
            SELECT em.id, em.entity_id, em.chunk_id, em.surface, em.start_char, em.end_char,
                   e.canonical_name
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.id
            WHERE em.surface LIKE '%[%'
            AND em.surface NOT LIKE '%]%'
        """
        
        if entity_id is not None:
            query = sql.SQL(base_query + " AND em.entity_id = %s ORDER BY em.entity_id, em.id")
            cur.execute(query, [entity_id])
        else:
            query = sql.SQL(base_query + " ORDER BY em.entity_id, em.id")
            cur.execute(query)
        rows = cur.fetchall()
        
        # Convert to dict format
        mentions = []
        for row in rows:
            mentions.append({
                'id': row[0],
                'entity_id': row[1],
                'chunk_id': row[2],
                'surface': row[3],
                'start_char': row[4],
                'end_char': row[5],
                'canonical_name': row[6]
            })
        
        print(f"Found {len(mentions)} mentions with opening bracket but no closing bracket", file=sys.stderr)
        
        fixed_count = 0
        not_found_count = 0
        
        for mention in mentions:
            mention_id = mention['id']
            chunk_id = mention['chunk_id']
            old_surface = mention['surface']
            entity_name = mention['canonical_name']
            
            # Get chunk text
            cur.execute("SELECT text FROM chunks WHERE id = %s", (chunk_id,))
            chunk_row = cur.fetchone()
            if not chunk_row:
                print(f"  Mention {mention_id} ({entity_name}): Chunk {chunk_id} not found", file=sys.stderr)
                not_found_count += 1
                continue
            
            chunk_text = chunk_row[0]  # Regular cursor returns tuples
            
            # Look for the full version with closing bracket
            # Search for pattern like "AILERON [ELERON]" near the old surface
            import re
            
            # Escape the old surface for regex (but keep brackets as literal)
            escaped_old = re.escape(old_surface).replace(r'\[', r'\[').replace(r'\]', r'\]')
            # Look for old_surface followed by ']' within a few characters
            pattern = re.escape(old_surface) + r'\]'
            match = re.search(pattern, chunk_text)
            
            if match:
                new_surface = match.group(0)  # This includes the closing bracket
                print(f"  Mention {mention_id} ({entity_name}): '{old_surface}' -> '{new_surface}'", file=sys.stderr)
                
                if not dry_run:
                    # Update the surface text
                    cur.execute("""
                        UPDATE entity_mentions
                        SET surface = %s
                        WHERE id = %s
                    """, (new_surface, mention_id))
                    fixed_count += 1
            else:
                # Try case-insensitive search
                pattern_ci = re.escape(old_surface) + r'\]'
                match_ci = re.search(pattern_ci, chunk_text, re.IGNORECASE)
                if match_ci:
                    new_surface = match_ci.group(0)
                    print(f"  Mention {mention_id} ({entity_name}): '{old_surface}' -> '{new_surface}' (case-insensitive)", file=sys.stderr)
                    if not dry_run:
                        cur.execute("""
                            UPDATE entity_mentions
                            SET surface = %s
                            WHERE id = %s
                        """, (new_surface, mention_id))
                        fixed_count += 1
                else:
                    print(f"  Mention {mention_id} ({entity_name}): Could not find '{old_surface}]' in chunk", file=sys.stderr)
                    not_found_count += 1
        
        if not dry_run:
            conn.commit()
            print(f"\nFixed {fixed_count} mentions", file=sys.stderr)
        else:
            print(f"\nWould fix {fixed_count} mentions", file=sys.stderr)
        
        if not_found_count > 0:
            print(f"Could not fix {not_found_count} mentions (pattern not found in chunk)", file=sys.stderr)
    
    finally:
        cur.close()


def delete_and_reextract(conn, entity_id: int, dry_run: bool = False):
    """
    Delete existing mentions for an entity and re-extract them.
    This ensures the new extraction code (with bracket fix) is used.
    """
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # Get entity name
        cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
        entity_row = cur.fetchone()
        if not entity_row:
            print(f"Entity {entity_id} not found", file=sys.stderr)
            return
        
        entity_name = entity_row['canonical_name']
        
        # Count existing mentions
        cur.execute("SELECT COUNT(*) FROM entity_mentions WHERE entity_id = %s", (entity_id,))
        count = cur.fetchone()[0]
        
        print(f"Entity: {entity_name} (ID: {entity_id})", file=sys.stderr)
        print(f"Existing mentions: {count}", file=sys.stderr)
        
        if not dry_run:
            # Delete existing mentions
            cur.execute("DELETE FROM entity_mentions WHERE entity_id = %s", (entity_id,))
            conn.commit()
            print(f"Deleted {count} mentions", file=sys.stderr)
            print(f"\nNow re-run extraction:", file=sys.stderr)
            print(f"  python concordance/extract_entity_mentions_from_citations.py --entity-id {entity_id}", file=sys.stderr)
        else:
            print(f"Would delete {count} mentions", file=sys.stderr)
            print(f"Then re-run: python concordance/extract_entity_mentions_from_citations.py --entity-id {entity_id}", file=sys.stderr)
    
    finally:
        cur.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fix entity mentions missing closing brackets"
    )
    parser.add_argument(
        "--entity-id",
        type=int,
        help="Entity ID to fix (if not provided, fixes all affected entities)"
    )
    parser.add_argument(
        "--entity-name",
        help="Entity name to fix (looks up entity_id)"
    )
    parser.add_argument(
        "--delete-and-reextract",
        action="store_true",
        help="Delete existing mentions and re-extract (instead of just updating surface text)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    conn = get_db_connection()
    try:
        entity_id = args.entity_id
        
        if args.entity_name:
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute("SELECT id FROM entities WHERE canonical_name = %s", (args.entity_name,))
            row = cur.fetchone()
            if row:
                entity_id = int(row['id'])  # Ensure it's an integer
            else:
                print(f"Entity '{args.entity_name}' not found", file=sys.stderr)
                return 1
            cur.close()
        
        if args.delete_and_reextract:
            if not entity_id:
                print("Error: --entity-id or --entity-name required for --delete-and-reextract", file=sys.stderr)
                return 1
            delete_and_reextract(conn, entity_id, dry_run=args.dry_run)
        else:
            fix_bracket_mentions(conn, entity_id=entity_id, dry_run=args.dry_run)
    
    finally:
        conn.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
