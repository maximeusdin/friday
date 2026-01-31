#!/usr/bin/env python3
"""
populate_review_queue.py [options]

Populates mention_review_queue from collision_queue items (from extract_entity_mentions.py).

This script reads collision_queue items (typically from match_summary.csv or a JSON file)
and inserts them into the mention_review_queue table for review.

Usage:
  python scripts/populate_review_queue.py --from-csv match_summary.csv
  python scripts/populate_review_queue.py --from-json collision_queue.json
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import execute_values, Json

from retrieval.ops import get_conn


def load_collision_queue_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load collision queue items from match_summary.csv."""
    items = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('category') == 'collision_queue' and row.get('subcategory') == 'high_value':
                # Parse the term to extract alias_norm and surface
                term = row.get('term', '')
                # Format: "alias_norm (surface: surface_text)"
                if '(surface: ' in term:
                    alias_norm, surface_part = term.split('(surface: ', 1)
                    alias_norm = alias_norm.strip()
                    surface = surface_part.rstrip(')').strip()
                else:
                    alias_norm = term
                    surface = term
                
                # Parse canonical_names to get entity IDs
                canonical_names_str = row.get('canonical_names', '')
                # Format: "entity_123; entity_456" or "Canonical Name 1; Canonical Name 2"
                # We need to look up entity IDs from canonical names
                items.append({
                    'alias_norm': alias_norm,
                    'surface': surface,
                    'canonical_names': canonical_names_str,
                    'description': row.get('description', ''),
                })
    
    return items


def load_collision_queue_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load collision queue items from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_entity_ids_from_canonical_names(conn, canonical_names: List[str]) -> Dict[str, int]:
    """Get entity IDs from canonical names."""
    if not canonical_names:
        return {}
    
    with conn.cursor() as cur:
        # Handle both "entity_123" format and actual canonical names
        entity_ids = {}
        for name in canonical_names:
            name = name.strip()
            if name.startswith('entity_'):
                try:
                    entity_id = int(name.replace('entity_', ''))
                    entity_ids[name] = entity_id
                except ValueError:
                    pass
            else:
                cur.execute("SELECT id FROM entities WHERE canonical_name = %s", (name,))
                row = cur.fetchone()
                if row:
                    entity_ids[name] = row[0]
        
        return entity_ids


def populate_review_queue_from_collision_items(
    conn,
    collision_items: List[Dict[str, Any]],
    *,
    dry_run: bool = False,
) -> int:
    """Populate mention_review_queue from collision_queue items."""
    review_items = []
    
    for item in collision_items:
        chunk_id = item.get('chunk_id')
        document_id = item.get('document_id')
        surface = item.get('surface', '')
        alias_norm = item.get('alias_norm', '')
        context_excerpt = item.get('context_excerpt', '')
        candidate_entity_ids = item.get('candidate_entity_ids', [])
        
        if not chunk_id or not document_id or not surface:
            continue
        
        # Build candidates JSONB
        # For entity mentions, we need canonical names
        candidates = []
        if candidate_entity_ids:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, canonical_name, entity_type
                    FROM entities
                    WHERE id = ANY(%s)
                """, (candidate_entity_ids,))
                for entity_id, canonical_name, entity_type in cur.fetchall():
                    candidates.append({
                        "entity_id": entity_id,
                        "canonical_name": canonical_name,
                        "entity_type": entity_type,
                        "score": 1.0,  # Default score for collision candidates
                    })
        
        if not candidates:
            continue
        
        review_items.append({
            'mention_type': 'entity',
            'chunk_id': chunk_id,
            'document_id': document_id,
            'surface': surface,
            'start_char': item.get('start_char'),
            'end_char': item.get('end_char'),
            'context_excerpt': context_excerpt or surface,
            'candidates': Json(candidates),
        })
    
    if dry_run:
        print(f"Would insert {len(review_items)} review items", file=sys.stderr)
        return len(review_items)
    
    if not review_items:
        return 0
    
    # Insert into review queue (check for duplicates)
    inserted = 0
    with conn.cursor() as cur:
        for item in review_items:
            # Check if duplicate exists
            cur.execute("""
                SELECT COUNT(*) FROM mention_review_queue
                WHERE chunk_id = %s
                  AND surface = %s
                  AND mention_type = %s
                  AND md5(candidates::text) = md5(%s::text)
                  AND status = 'pending'
            """, (item['chunk_id'], item['surface'], item['mention_type'], item['candidates']))
            
            if cur.fetchone()[0] > 0:
                continue  # Skip duplicate
            
            # Insert new item
            cur.execute("""
                INSERT INTO mention_review_queue
                    (mention_type, chunk_id, document_id, surface, start_char, end_char,
                     context_excerpt, candidates, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                item['mention_type'],
                item['chunk_id'],
                item['document_id'],
                item['surface'],
                item.get('start_char'),
                item.get('end_char'),
                item['context_excerpt'],
                item['candidates'],
                'pending',
            ))
            inserted += 1
    
    conn.commit()
    return inserted


def main():
    parser = argparse.ArgumentParser(
        description="Populate mention_review_queue from collision_queue items"
    )
    parser.add_argument('--from-csv', help='Load from match_summary.csv')
    parser.add_argument('--from-json', help='Load from JSON file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no inserts)')
    
    args = parser.parse_args()
    
    if not args.from_csv and not args.from_json:
        parser.error("Must specify --from-csv or --from-json")
    
    conn = get_conn()
    try:
        if args.from_csv:
            items = load_collision_queue_from_csv(args.from_csv)
        else:
            items = load_collision_queue_from_json(args.from_json)
        
        print(f"Loaded {len(items)} collision queue items", file=sys.stderr)
        
        # For CSV, we need to enrich with entity IDs
        # For now, this is a simplified version - you may need to adjust based on your CSV format
        enriched_items = []
        for item in items:
            # This is a placeholder - adjust based on your actual collision_queue structure
            enriched_items.append(item)
        
        count = populate_review_queue_from_collision_items(
            conn,
            enriched_items,
            dry_run=args.dry_run,
        )
        
        print(f"{'Would insert' if args.dry_run else 'Inserted'} {count} review items", file=sys.stderr)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
