#!/usr/bin/env python3
"""
Backfill grouping columns in mention_review_queue.

One-time migration to populate:
- surface_norm
- candidate_entity_ids
- candidate_set_hash
- group_key

This makes the queue batch-ready for grouped operations.
"""

import sys
import os
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from retrieval.surface_norm import normalize_surface
from retrieval.proposal_gating import compute_group_key, compute_candidate_set_hash


def backfill_grouping_columns():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "neh"),
        user=os.getenv("DB_USER", "neh"),
        password=os.getenv("DB_PASS", "neh"),
    )
    
    print("Backfilling grouping columns in mention_review_queue...")
    
    # Count items needing backfill
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM mention_review_queue
        WHERE group_key IS NULL OR surface_norm IS NULL
    """)
    total_to_backfill = cur.fetchone()[0]
    print(f"Items needing backfill: {total_to_backfill}")
    
    if total_to_backfill == 0:
        print("Nothing to backfill!")
        return
    
    # Process in batches
    batch_size = 1000
    processed = 0
    
    while True:
        # Fetch batch of items needing update
        cur.execute("""
            SELECT id, surface, candidates
            FROM mention_review_queue
            WHERE group_key IS NULL OR surface_norm IS NULL
            LIMIT %s
        """, (batch_size,))
        
        rows = cur.fetchall()
        if not rows:
            break
        
        updates = []
        for row_id, surface, candidates_json in rows:
            # Compute surface_norm
            surface_norm = normalize_surface(surface) if surface else None
            
            # Extract candidate entity IDs from JSONB
            candidate_entity_ids = []
            if candidates_json:
                try:
                    if isinstance(candidates_json, str):
                        candidates_data = json.loads(candidates_json)
                    else:
                        candidates_data = candidates_json
                    
                    # Handle both old format (list) and new format (dict with 'entities')
                    if isinstance(candidates_data, dict) and 'entities' in candidates_data:
                        entities = candidates_data['entities']
                    elif isinstance(candidates_data, list):
                        entities = candidates_data
                    else:
                        entities = []
                    
                    for ent in entities:
                        if isinstance(ent, dict) and 'entity_id' in ent:
                            candidate_entity_ids.append(ent['entity_id'])
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
            
            # Compute hashes
            candidate_entity_ids_sorted = sorted(set(candidate_entity_ids)) if candidate_entity_ids else []
            candidate_set_hash = compute_candidate_set_hash(candidate_entity_ids_sorted) if candidate_entity_ids_sorted else None
            group_key = compute_group_key(surface_norm, candidate_entity_ids_sorted) if surface_norm and candidate_entity_ids_sorted else None
            
            updates.append((
                surface_norm,
                candidate_entity_ids_sorted if candidate_entity_ids_sorted else None,
                candidate_set_hash,
                group_key,
                row_id,
            ))
        
        # Batch update
        cur.executemany("""
            UPDATE mention_review_queue
            SET surface_norm = %s,
                candidate_entity_ids = %s,
                candidate_set_hash = %s,
                group_key = %s
            WHERE id = %s
        """, updates)
        
        conn.commit()
        processed += len(rows)
        print(f"  Processed {processed}/{total_to_backfill} ({100*processed/total_to_backfill:.1f}%)", end="\r")
    
    print(f"\nBackfill complete! Processed {processed} items.")
    
    # Verify
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(group_key) as with_group_key,
            COUNT(surface_norm) as with_surface_norm,
            COUNT(DISTINCT group_key) as unique_groups
        FROM mention_review_queue
        WHERE status = 'pending'
    """)
    row = cur.fetchone()
    print(f"\nVerification:")
    print(f"  Total pending: {row[0]}")
    print(f"  With group_key: {row[1]}")
    print(f"  With surface_norm: {row[2]}")
    print(f"  Unique groups: {row[3]}")
    
    # Show top groups
    cur.execute("""
        SELECT group_key, surface_norm, COUNT(*) as cnt
        FROM mention_review_queue
        WHERE status = 'pending' AND group_key IS NOT NULL
        GROUP BY group_key, surface_norm
        ORDER BY cnt DESC
        LIMIT 10
    """)
    print("\nTop collision groups:")
    for row in cur.fetchall():
        print(f"  {row[1]:<25} ({row[0]}): {row[2]} items")
    
    conn.close()


if __name__ == "__main__":
    backfill_grouping_columns()
