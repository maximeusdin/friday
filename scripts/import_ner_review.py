#!/usr/bin/env python3
"""
Import reviewed NER surfaces from CSV.

Processes reviewer decisions:
- LINK_1/2/3: Links surface to existing entity via entity_aliases
- CREATE: Creates new entity with aliases
- REJECT: Marks surface as rejected in ner_surface_stats

Usage:
    python scripts/import_ner_review.py reviewed_file.csv
    python scripts/import_ner_review.py reviewed_file.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor


def get_conn():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'neh'),
        user=os.getenv('DB_USER', 'neh'),
        password=os.getenv('DB_PASS', 'neh')
    )


def parse_aliases(aliases_str: str) -> List[str]:
    """Parse semicolon-separated aliases."""
    if not aliases_str:
        return []
    return [a.strip() for a in aliases_str.split(';') if a.strip()]


def create_entity(
    conn,
    name: str,
    entity_type: str,
    description: str = None,
    dry_run: bool = False,
) -> Optional[int]:
    """Create a new entity and return its ID."""
    if dry_run:
        print(f"    [DRY RUN] Would create entity: {name} ({entity_type})")
        return None
    
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entities (canonical_name, entity_type, description)
        VALUES (%s, %s, %s)
        RETURNING id
    """, (name, entity_type, description))
    
    entity_id = cur.fetchone()[0]
    return entity_id


def add_alias(
    conn,
    entity_id: int,
    alias: str,
    kind: str = 'alt',
    dry_run: bool = False,
) -> bool:
    """Add an alias to an entity."""
    if dry_run:
        print(f"    [DRY RUN] Would add alias: {alias} -> entity {entity_id}")
        return True
    
    cur = conn.cursor()
    
    # Normalize alias
    alias_norm = alias.lower().strip()
    
    try:
        cur.execute("""
            INSERT INTO entity_aliases (entity_id, alias, alias_norm, kind, is_matchable)
            VALUES (%s, %s, %s, %s, TRUE)
            ON CONFLICT (entity_id, alias_norm) DO NOTHING
        """, (entity_id, alias, alias_norm, kind))
        return True
    except Exception as e:
        print(f"    Warning: Could not add alias {alias}: {e}")
        return False


def update_surface_status(
    conn,
    surface_id: int,
    status: str,
    entity_id: int = None,
    dry_run: bool = False,
) -> None:
    """Update the status of a surface in ner_surface_stats."""
    if dry_run:
        return
    
    cur = conn.cursor()
    cur.execute("""
        UPDATE ner_surface_stats
        SET status = %s,
            matched_entity_id = %s,
            reviewed_by = 'csv_import',
            updated_at = NOW()
        WHERE id = %s
    """, (status, entity_id, surface_id))


def process_review_file(
    conn,
    file_path: str,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Process a reviewed CSV file and apply decisions."""
    
    stats = {
        'total': 0,
        'linked': 0,
        'created': 0,
        'rejected': 0,
        'skipped': 0,
        'errors': 0,
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Processing {len(rows)} rows...")
    
    for row in rows:
        stats['total'] += 1
        
        decision = (row.get('decision') or '').strip().upper()
        surface_id = row.get('id')
        surface_norm = row.get('surface_norm', '')
        
        if not decision or decision == 'SKIP':
            stats['skipped'] += 1
            continue
        
        try:
            # LINK decisions
            if decision.startswith('LINK_'):
                candidate_num = decision.replace('LINK_', '')
                entity_id_col = f'candidate_{candidate_num}_id'
                entity_id = row.get(entity_id_col)
                
                if not entity_id:
                    print(f"  Warning: No entity ID for {surface_norm} (decision={decision})")
                    stats['errors'] += 1
                    continue
                
                entity_id = int(entity_id)
                
                # Add this surface as an alias to the entity
                print(f"  Linking {surface_norm!r} -> entity {entity_id}")
                add_alias(conn, entity_id, surface_norm, kind='alt', dry_run=dry_run)
                update_surface_status(conn, surface_id, 'linked', entity_id, dry_run)
                stats['linked'] += 1
            
            # CREATE decision
            elif decision == 'CREATE':
                is_new = (row.get('is_new_entity') or '').strip().upper()
                if is_new != 'TRUE':
                    print(f"  Warning: CREATE without is_new_entity=TRUE for {surface_norm}")
                    stats['errors'] += 1
                    continue
                
                new_name = (row.get('new_entity_name') or '').strip()
                new_type = (row.get('new_entity_type') or '').strip()
                new_desc = (row.get('new_entity_description') or '').strip()
                new_aliases = parse_aliases(row.get('new_entity_aliases', ''))
                
                if not new_name:
                    print(f"  Warning: CREATE without new_entity_name for {surface_norm}")
                    stats['errors'] += 1
                    continue
                
                if not new_type or new_type not in ('person', 'org', 'place'):
                    print(f"  Warning: Invalid entity_type '{new_type}' for {surface_norm}, defaulting to 'person'")
                    new_type = 'person'
                
                print(f"  Creating entity: {new_name} ({new_type})")
                entity_id = create_entity(conn, new_name, new_type, new_desc, dry_run)
                
                if entity_id or dry_run:
                    # Add primary alias (the canonical name)
                    if entity_id:
                        add_alias(conn, entity_id, new_name, kind='primary', dry_run=dry_run)
                    
                    # Add discovered surface as alias
                    if entity_id and surface_norm.lower() != new_name.lower():
                        add_alias(conn, entity_id, surface_norm, kind='alt', dry_run=dry_run)
                    
                    # Add additional aliases
                    for alias in new_aliases:
                        if alias.lower() != new_name.lower() and alias.lower() != surface_norm.lower():
                            if entity_id:
                                add_alias(conn, entity_id, alias, kind='alt', dry_run=dry_run)
                    
                    update_surface_status(conn, surface_id, 'created', entity_id, dry_run)
                    stats['created'] += 1
            
            # REJECT decision
            elif decision == 'REJECT':
                print(f"  Rejecting: {surface_norm!r}")
                update_surface_status(conn, surface_id, 'rejected', None, dry_run)
                stats['rejected'] += 1
            
            else:
                print(f"  Unknown decision '{decision}' for {surface_norm}")
                stats['errors'] += 1
        
        except Exception as e:
            print(f"  Error processing {surface_norm}: {e}")
            stats['errors'] += 1
    
    if not dry_run:
        conn.commit()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Import reviewed NER surfaces from CSV'
    )
    parser.add_argument('file', help='Reviewed CSV file to import')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)
    
    print("="*60)
    print("IMPORT NER REVIEW DECISIONS")
    print("="*60)
    print(f"File: {args.file}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    conn = get_conn()
    
    try:
        stats = process_review_file(conn, args.file, args.dry_run)
    finally:
        conn.close()
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total rows:     {stats['total']}")
    print(f"Linked:         {stats['linked']}")
    print(f"Created:        {stats['created']}")
    print(f"Rejected:       {stats['rejected']}")
    print(f"Skipped:        {stats['skipped']}")
    print(f"Errors:         {stats['errors']}")
    
    if args.dry_run:
        print("\n[DRY RUN - no changes made]")


if __name__ == '__main__':
    main()
