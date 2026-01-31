#!/usr/bin/env python3
"""
review_mentions.py [command] [options]

CLI tool for reviewing and adjudicating ambiguous mentions in the mention_review_queue.

Commands:
  list          List pending reviews (default)
  show <id>     Show details and context for a specific review item
  accept <id>   Accept a review item (requires --entity-id or --date-range)
  reject <id>   Reject a review item (mark as not a valid mention)
  add-alias     Add alias to entity when accepting (requires --entity-id and --alias)

Usage:
  python scripts/review_mentions.py list
  python scripts/review_mentions.py show 123
  python scripts/review_mentions.py accept 123 --entity-id 456
  python scripts/review_mentions.py accept 123 --date-start 1945-06-23 --date-end 1945-06-23 --precision day
  python scripts/review_mentions.py reject 123 --note "Not a valid mention"
  python scripts/review_mentions.py accept 123 --entity-id 456 --add-alias "yakubovich"
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import date, datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import Json

from retrieval.ops import get_conn


# =============================================================================
# Database utilities
# =============================================================================

def get_review_item(conn, review_id: int) -> Optional[Dict[str, Any]]:
    """Get a review item by ID."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, mention_type, chunk_id, document_id, surface, start_char, end_char,
                   context_excerpt, candidates, status, decision, note, created_at, reviewed_at
            FROM mention_review_queue
            WHERE id = %s
        """, (review_id,))
        row = cur.fetchone()
        if not row:
            return None
        
        return {
            "id": row[0],
            "mention_type": row[1],
            "chunk_id": row[2],
            "document_id": row[3],
            "surface": row[4],
            "start_char": row[5],
            "end_char": row[6],
            "context_excerpt": row[7],
            "candidates": row[8],
            "status": row[9],
            "decision": row[10],
            "note": row[11],
            "created_at": row[12],
            "reviewed_at": row[13],
        }


def list_pending_reviews(
    conn,
    mention_type: Optional[str] = None,
    collision_category: Optional[str] = None,
    is_high_value: Optional[bool] = None,
    limit: Optional[int] = None,
    include_all_statuses: bool = False,
) -> List[Dict[str, Any]]:
    """List pending review items."""
    if include_all_statuses:
        conditions = []
    else:
        conditions = ["status = 'pending'"]
    params = []
    
    if mention_type:
        conditions.append("mention_type = %s")
        params.append(mention_type)
    
    # Filter by collision category or high-value flag using JSONB queries
    if collision_category:
        # Handle both new format (with collision_metadata) and legacy format
        conditions.append("""
            (candidates ? 'collision_metadata' AND candidates->'collision_metadata'->>'collision_category' = %s)
            OR (NOT (candidates ? 'collision_metadata') AND %s = 'unknown')
        """)
        params.append(collision_category)
        params.append(collision_category)
    
    if is_high_value is not None:
        conditions.append("candidates ? 'collision_metadata'")
        if is_high_value:
            conditions.append("(candidates->'collision_metadata'->>'is_high_value')::boolean = true")
        else:
            conditions.append("((candidates->'collision_metadata'->>'is_high_value')::boolean = false OR candidates->'collision_metadata'->>'is_high_value' IS NULL)")
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    with conn.cursor() as cur:
        # Debug: show the query being executed
        if not include_all_statuses:
            cur.execute("SELECT COUNT(*) FROM mention_review_queue WHERE status = 'pending'")
            total_pending = cur.fetchone()[0]
            print(f"  [DEBUG] Total pending items in DB: {total_pending}", file=sys.stderr)
        
        cur.execute(f"""
            SELECT id, mention_type, chunk_id, document_id, surface, candidates, status, created_at
            FROM mention_review_queue
            WHERE {where_clause}
            ORDER BY created_at DESC
            {limit_clause}
        """, params)
        
        rows = cur.fetchall()
        print(f"  [DEBUG] Query returned {len(rows)} items with where_clause: {where_clause}", file=sys.stderr)
        
        return [
            {
                "id": row[0],
                "mention_type": row[1],
                "chunk_id": row[2],
                "document_id": row[3],
                "surface": row[4],
                "candidates": row[5],
                "status": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]


def get_chunk_text(conn, chunk_id: int) -> Optional[str]:
    """Get chunk text for context."""
    with conn.cursor() as cur:
        cur.execute("SELECT text FROM chunks WHERE id = %s", (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None


def get_entity_info(conn, entity_id: int) -> Optional[Dict[str, Any]]:
    """Get entity information."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, canonical_name, entity_type
            FROM entities
            WHERE id = %s
        """, (entity_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "canonical_name": row[1],
            "entity_type": row[2],
        }


# =============================================================================
# Display utilities
# =============================================================================

def print_review_item(item: Dict[str, Any], show_context: bool = True):
    """Print a review item in a readable format."""
    print(f"\n{'='*70}")
    print(f"Review ID: {item['id']}")
    print(f"Type: {item['mention_type']}")
    print(f"Status: {item['status']}")
    print(f"Surface: '{item['surface']}'")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Document ID: {item['document_id']}")
    print(f"Created: {item['created_at']}")
    
    if item.get('reviewed_at'):
        print(f"Reviewed: {item['reviewed_at']}")
    
    if item.get('note'):
        print(f"Note: {item['note']}")
    
    print(f"\nCandidates:")
    candidates_raw = item.get('candidates', [])
    if isinstance(candidates_raw, str):
        candidates_raw = json.loads(candidates_raw)
    
    # Handle new structure with collision_metadata
    if isinstance(candidates_raw, dict) and 'entities' in candidates_raw:
        candidates = candidates_raw.get('entities', [])
        collision_metadata = candidates_raw.get('collision_metadata', {})
        
        # Display collision metadata
        if collision_metadata:
            print(f"\nCollision Metadata:")
            print(f"  Category: {collision_metadata.get('collision_category', 'unknown')}")
            print(f"  High-value: {collision_metadata.get('is_high_value', False)}")
            print(f"  Harmless: {collision_metadata.get('is_harmless', False)}")
            if collision_metadata.get('info'):
                print(f"  Info: {collision_metadata.get('info')}")
    else:
        # Legacy format (direct list)
        candidates = candidates_raw if isinstance(candidates_raw, list) else []
        collision_metadata = {}
    
    if item['mention_type'] == 'entity':
        for i, cand in enumerate(candidates, 1):
            entity_id = cand.get('entity_id')
            canonical_name = cand.get('canonical_name', '?')
            score = cand.get('score', 0.0)
            print(f"  {i}. Entity ID {entity_id}: {canonical_name} (score: {score:.2f})")
    else:  # date
        for i, cand in enumerate(candidates, 1):
            date_start = cand.get('date_start')
            date_end = cand.get('date_end')
            precision = cand.get('precision', 'unknown')
            confidence = cand.get('confidence', 0.0)
            print(f"  {i}. {date_start} to {date_end} (precision: {precision}, confidence: {confidence:.2f})")
    
    if show_context and item.get('context_excerpt'):
        print(f"\nContext excerpt:")
        print(f"  {item['context_excerpt']}")
    
    if item.get('decision'):
        decision = item['decision']
        if isinstance(decision, str):
            decision = json.loads(decision)
        print(f"\nDecision: {json.dumps(decision, indent=2)}")
    
    print(f"{'='*70}\n")


# =============================================================================
# Actions
# =============================================================================

def accept_review(
    conn,
    review_id: int,
    *,
    entity_id: Optional[int] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    precision: Optional[str] = None,
    alias_norm: Optional[str] = None,
    note: Optional[str] = None,
    add_alias: Optional[str] = None,
) -> bool:
    """Accept a review item."""
    item = get_review_item(conn, review_id)
    if not item:
        print(f"Error: Review item {review_id} not found", file=sys.stderr)
        return False
    
    if item['status'] != 'pending':
        print(f"Error: Review item {review_id} is not pending (status: {item['status']})", file=sys.stderr)
        return False
    
    decision = {}
    
    if item['mention_type'] == 'entity':
        if not entity_id:
            print("Error: --entity-id required for entity mentions", file=sys.stderr)
            return False
        
        # Verify entity exists
        entity_info = get_entity_info(conn, entity_id)
        if not entity_info:
            print(f"Error: Entity {entity_id} not found", file=sys.stderr)
            return False
        
        decision = {
            "entity_id": entity_id,
            "alias_norm": alias_norm or item.get('surface', '').lower(),
        }
        
        # Add alias if requested
        if add_alias:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO entity_aliases (entity_id, alias, alias_norm, alias_type, alias_class, is_auto_match, match_case)
                    VALUES (%s, %s, %s, 'human_review', 'unknown', true, 'any')
                    ON CONFLICT (entity_id, alias_norm) DO NOTHING
                """, (entity_id, add_alias, add_alias.lower()))
            print(f"Added alias '{add_alias}' to entity {entity_id}", file=sys.stderr)
    
    else:  # date
        if not date_start or not date_end or not precision:
            print("Error: --date-start, --date-end, and --precision required for date mentions", file=sys.stderr)
            return False
        
        decision = {
            "date_start": date_start,
            "date_end": date_end,
            "precision": precision,
        }
    
    # Update review item
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE mention_review_queue
            SET status = 'accepted',
                decision = %s,
                note = COALESCE(%s, note),
                reviewed_at = now()
            WHERE id = %s
        """, (Json(decision), note, review_id))
    
    conn.commit()
    print(f"Accepted review item {review_id}", file=sys.stderr)
    return True


def reject_review(
    conn,
    review_id: int,
    note: Optional[str] = None,
) -> bool:
    """Reject a review item."""
    item = get_review_item(conn, review_id)
    if not item:
        print(f"Error: Review item {review_id} not found", file=sys.stderr)
        return False
    
    if item['status'] != 'pending':
        print(f"Error: Review item {review_id} is not pending (status: {item['status']})", file=sys.stderr)
        return False
    
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE mention_review_queue
            SET status = 'rejected',
                note = COALESCE(%s, note),
                reviewed_at = now()
            WHERE id = %s
        """, (note, review_id))
    
    conn.commit()
    print(f"Rejected review item {review_id}", file=sys.stderr)
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Review and adjudicate ambiguous mentions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List pending reviews
  python scripts/review_mentions.py list
  
  # List pending entity reviews only
  python scripts/review_mentions.py list --type entity
  
  # Show details for a review item
  python scripts/review_mentions.py show 123
  
  # Accept an entity mention
  python scripts/review_mentions.py accept 123 --entity-id 456
  
  # Accept an entity mention and add alias
  python scripts/review_mentions.py accept 123 --entity-id 456 --add-alias "yakubovich"
  
  # Accept a date mention
  python scripts/review_mentions.py accept 123 --date-start 1945-06-23 --date-end 1945-06-23 --precision day
  
  # Reject a mention
  python scripts/review_mentions.py reject 123 --note "Not a valid mention"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List pending reviews')
    list_parser.add_argument('--type', choices=['entity', 'date'], help='Filter by mention type')
    list_parser.add_argument('--category', choices=['high_value_enqueued', 'high_value_too_many', 'dominance_none', 'harmless'], 
                            help='Filter by collision category')
    list_parser.add_argument(
            '--high-value',
            action='store_true',
            default=None,   # IMPORTANT: default None => no filter unless explicitly set
            dest='is_high_value',
            help='Filter to only high-value collisions'
        )
    list_parser.add_argument('--all-statuses', action='store_true', help='Include all statuses (not just pending)')
    list_parser.add_argument('--limit', type=int, help='Limit number of results')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show details for a review item')
    show_parser.add_argument('id', type=int, help='Review item ID')
    
    # Accept command
    accept_parser = subparsers.add_parser('accept', help='Accept a review item')
    accept_parser.add_argument('id', type=int, help='Review item ID')
    accept_parser.add_argument('--entity-id', type=int, help='Entity ID (for entity mentions)')
    accept_parser.add_argument('--date-start', help='Date start (YYYY-MM-DD, for date mentions)')
    accept_parser.add_argument('--date-end', help='Date end (YYYY-MM-DD, for date mentions)')
    accept_parser.add_argument('--precision', choices=['day', 'month', 'year', 'range'], help='Date precision (for date mentions)')
    accept_parser.add_argument('--alias-norm', help='Alias norm (optional, defaults to surface)')
    accept_parser.add_argument('--add-alias', help='Add this alias to the entity when accepting')
    accept_parser.add_argument('--note', help='Optional note')
    
    # Reject command
    reject_parser = subparsers.add_parser('reject', help='Reject a review item')
    reject_parser.add_argument('id', type=int, help='Review item ID')
    reject_parser.add_argument('--note', help='Reason for rejection')
    
    args = parser.parse_args()
    
    if not args.command:
        args.command = 'list'  # Default to list
    
    conn = get_conn()
    try:
        if args.command == 'list':
            reviews = list_pending_reviews(
                conn, 
                mention_type=args.type, 
                collision_category=args.category,
                is_high_value=args.is_high_value if hasattr(args, 'is_high_value') else None,
                limit=args.limit,
                include_all_statuses=args.all_statuses if hasattr(args, 'all_statuses') else False,
            )
            
            status_label = "reviews" if (hasattr(args, 'all_statuses') and args.all_statuses) else "pending review(s)"
            if not reviews:
                print(f"No {status_label} found", file=sys.stderr)
                return
            
            print(f"\nFound {len(reviews)} {status_label}:\n", file=sys.stderr)
            
            for review in reviews:
                candidates_raw = review.get('candidates', [])
                if isinstance(candidates_raw, str):
                    candidates_raw = json.loads(candidates_raw)
                
                # Handle new structure with collision_metadata
                if isinstance(candidates_raw, dict) and 'entities' in candidates_raw:
                    candidates = candidates_raw.get('entities', [])
                    collision_metadata = candidates_raw.get('collision_metadata', {})
                    category = collision_metadata.get('collision_category', '')
                    category_str = f" [{category}]" if category else ""
                else:
                    # Legacy format
                    candidates = candidates_raw if isinstance(candidates_raw, list) else []
                    category_str = ""
                
                if review['mention_type'] == 'entity':
                    cand_str = ", ".join([f"{c.get('canonical_name', '?')} (ID: {c.get('entity_id')})" for c in candidates[:3]])
                    if len(candidates) > 3:
                        cand_str += f" ... (+{len(candidates) - 3} more)"
                else:
                    cand_str = ", ".join([f"{c.get('date_start')} to {c.get('date_end')}" for c in candidates[:3]])
                    if len(candidates) > 3:
                        cand_str += f" ... (+{len(candidates) - 3} more)"
                
                status_str = f" [{review.get('status', 'pending')}]" if review.get('status') and review.get('status') != 'pending' else ""
                print(f"  [{review['id']}] {review['mention_type']:6}{category_str}{status_str} | '{review['surface']}' | {cand_str}")
            
            print()
        
        elif args.command == 'show':
            item = get_review_item(conn, args.id)
            if not item:
                print(f"Error: Review item {args.id} not found", file=sys.stderr)
                sys.exit(1)
            
            print_review_item(item, show_context=True)
        
        elif args.command == 'accept':
            success = accept_review(
                conn,
                args.id,
                entity_id=args.entity_id,
                date_start=args.date_start,
                date_end=args.date_end,
                precision=args.precision,
                alias_norm=args.alias_norm,
                note=args.note,
                add_alias=args.add_alias,
            )
            if not success:
                sys.exit(1)
        
        elif args.command == 'reject':
            success = reject_review(conn, args.id, note=args.note)
            if not success:
                sys.exit(1)
        
        elif args.command == 'stats':
            # Show statistics about review queue
            with conn.cursor() as cur:
                # Total counts by status
                cur.execute("""
                    SELECT status, COUNT(*) 
                    FROM mention_review_queue 
                    GROUP BY status
                    ORDER BY status
                """)
                print("\nReview Queue Statistics:\n")
                print("By Status:")
                for status, count in cur.fetchall():
                    print(f"  {status:12}: {count:>6}")
                
                # Counts by collision category (for pending items)
                cur.execute("""
                    SELECT 
                        candidates->'collision_metadata'->>'collision_category' as category,
                        COUNT(*) as count
                    FROM mention_review_queue
                    WHERE status = 'pending'
                      AND candidates ? 'collision_metadata'
                    GROUP BY category
                    ORDER BY count DESC
                """)
                print("\nBy Collision Category (pending only):")
                for category, count in cur.fetchall():
                    print(f"  {category or 'unknown':25}: {count:>6}")
                
                # High-value vs non-high-value
                cur.execute("""
                    SELECT 
                        CASE 
                            WHEN (candidates->'collision_metadata'->>'is_high_value')::boolean = true THEN 'high_value'
                            ELSE 'not_high_value'
                        END as value_type,
                        COUNT(*) as count
                    FROM mention_review_queue
                    WHERE status = 'pending'
                      AND candidates ? 'collision_metadata'
                    GROUP BY value_type
                    ORDER BY value_type
                """)
                print("\nBy Value Type (pending only):")
                for value_type, count in cur.fetchall():
                    print(f"  {value_type:15}: {count:>6}")
                
                print()
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
