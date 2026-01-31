#!/usr/bin/env python3
"""
Junk Pattern Learning

When a reviewer marks items as junk, this script adds them to ocr_junk_patterns.
Can be called from review workflow or batch from rejected queue items.

Usage:
    # Add single junk pattern
    python scripts/learn_junk_pattern.py --surface "DELETED" --type exact --note "Redaction marker"
    
    # Batch learn from rejected queue items
    python scripts/learn_junk_pattern.py --from-queue --min-rejections 3
"""

import argparse
import sys
from typing import Optional

import psycopg2

sys.path.insert(0, '.')
from retrieval.surface_norm import normalize_surface


def get_conn():
    return psycopg2.connect(
        host='localhost', port=5432, dbname='neh', user='neh', password='neh'
    )


def add_junk_pattern(
    conn,
    pattern_value: str,
    pattern_type: str = 'exact',
    note: Optional[str] = None,
    source_document_id: Optional[int] = None,
    source_surface_norm: Optional[str] = None,
    learned_from: str = 'manual'
) -> bool:
    """Add a junk pattern to the database."""
    cur = conn.cursor()
    
    # Normalize for exact patterns
    if pattern_type == 'exact':
        pattern_value = normalize_surface(pattern_value)
    
    try:
        cur.execute("""
            INSERT INTO ocr_junk_patterns 
                (pattern_type, pattern_value, note, source_document_id, 
                 source_surface_norm, learned_from, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT (pattern_type, pattern_value) DO UPDATE SET
                occurrence_count = ocr_junk_patterns.occurrence_count + 1,
                last_seen_at = NOW(),
                note = COALESCE(EXCLUDED.note, ocr_junk_patterns.note)
            RETURNING id
        """, (
            pattern_type, pattern_value, note,
            source_document_id, source_surface_norm, learned_from
        ))
        conn.commit()
        row = cur.fetchone()
        print(f"Added/updated junk pattern: {pattern_type}={pattern_value!r} (id={row[0]})")
        return True
    except Exception as e:
        print(f"Error adding pattern: {e}")
        conn.rollback()
        return False


def learn_from_rejected_queue_items(conn, min_rejections: int = 3):
    """
    Learn junk patterns from frequently rejected queue items.
    
    Looks for surfaces that have been marked 'rejected' multiple times
    and adds them as exact junk patterns.
    """
    cur = conn.cursor()
    
    # Find surfaces rejected multiple times
    cur.execute("""
        SELECT 
            surface_norm,
            COUNT(*) as rejection_count,
            array_agg(DISTINCT document_id) as doc_ids
        FROM mention_review_queue
        WHERE status = 'rejected'
        GROUP BY surface_norm
        HAVING COUNT(*) >= %s
        ORDER BY rejection_count DESC
    """, (min_rejections,))
    
    rows = cur.fetchall()
    print(f"Found {len(rows)} surfaces rejected >= {min_rejections} times")
    
    added = 0
    for surface_norm, count, doc_ids in rows:
        # Check if already a junk pattern
        cur.execute("""
            SELECT 1 FROM ocr_junk_patterns 
            WHERE pattern_type = 'exact' AND pattern_value = %s
        """, (surface_norm,))
        
        if cur.fetchone():
            print(f"  Already exists: {surface_norm!r}")
            continue
        
        if add_junk_pattern(
            conn,
            pattern_value=surface_norm,
            pattern_type='exact',
            note=f'Auto-learned from {count} rejections across {len(doc_ids)} docs',
            learned_from='review_queue'
        ):
            added += 1
    
    print(f"\nAdded {added} new junk patterns")
    return added


def mark_queue_item_junk(conn, queue_id: int, learn: bool = True) -> bool:
    """
    Mark a queue item as junk and optionally learn the pattern.
    
    Called from review workflow.
    """
    cur = conn.cursor()
    
    # Get the item
    cur.execute("""
        SELECT surface_norm, document_id 
        FROM mention_review_queue 
        WHERE id = %s
    """, (queue_id,))
    
    row = cur.fetchone()
    if not row:
        print(f"Queue item {queue_id} not found")
        return False
    
    surface_norm, document_id = row
    
    # Update status
    cur.execute("""
        UPDATE mention_review_queue 
        SET status = 'rejected', resolved_at = NOW()
        WHERE id = %s
    """, (queue_id,))
    
    conn.commit()
    print(f"Marked queue item {queue_id} as rejected")
    
    # Optionally learn the pattern
    if learn:
        add_junk_pattern(
            conn,
            pattern_value=surface_norm,
            pattern_type='exact',
            note='Learned from review rejection',
            source_document_id=document_id,
            source_surface_norm=surface_norm,
            learned_from='review_queue'
        )
    
    return True


def show_junk_stats(conn):
    """Show current junk pattern statistics."""
    cur = conn.cursor()
    
    print("\n=== Junk Pattern Statistics ===")
    
    cur.execute("""
        SELECT pattern_type, COUNT(*), SUM(occurrence_count)
        FROM ocr_junk_patterns
        WHERE is_active = TRUE
        GROUP BY pattern_type
        ORDER BY COUNT(*) DESC
    """)
    
    print("\nBy type:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} patterns ({row[2]} total occurrences)")
    
    cur.execute("""
        SELECT learned_from, COUNT(*)
        FROM ocr_junk_patterns
        WHERE is_active = TRUE
        GROUP BY learned_from
        ORDER BY COUNT(*) DESC
    """)
    
    print("\nBy source:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    # Top patterns
    cur.execute("""
        SELECT pattern_value, occurrence_count, learned_from
        FROM ocr_junk_patterns
        WHERE is_active = TRUE AND pattern_type = 'exact'
        ORDER BY occurrence_count DESC
        LIMIT 10
    """)
    
    print("\nTop 10 exact patterns:")
    for row in cur.fetchall():
        print(f"  {row[0]!r}: {row[1]} occurrences ({row[2]})")


def main():
    parser = argparse.ArgumentParser(description='Learn junk patterns')
    parser.add_argument('--surface', help='Surface to add as junk')
    parser.add_argument('--type', default='exact', choices=['exact', 'prefix', 'suffix', 'regex'],
                       help='Pattern type')
    parser.add_argument('--note', help='Note about the pattern')
    parser.add_argument('--from-queue', action='store_true', 
                       help='Learn from rejected queue items')
    parser.add_argument('--min-rejections', type=int, default=3,
                       help='Min rejections to auto-learn')
    parser.add_argument('--mark-junk', type=int, metavar='QUEUE_ID',
                       help='Mark specific queue item as junk')
    parser.add_argument('--stats', action='store_true', help='Show junk pattern stats')
    args = parser.parse_args()
    
    conn = get_conn()
    
    try:
        if args.stats:
            show_junk_stats(conn)
        elif args.surface:
            add_junk_pattern(conn, args.surface, args.type, args.note)
        elif args.from_queue:
            learn_from_rejected_queue_items(conn, args.min_rejections)
        elif args.mark_junk:
            mark_queue_item_junk(conn, args.mark_junk)
        else:
            parser.print_help()
    finally:
        conn.close()


if __name__ == '__main__':
    main()
