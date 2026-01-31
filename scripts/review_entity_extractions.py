#!/usr/bin/env python3
"""
Review queue manager for entity extractions.

Allows reviewing and adjudicating entity mentions that fall in the review threshold range.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from retrieval.ops import get_conn


def get_review_queue(
    conn,
    collection_slug: Optional[str] = None,
    min_confidence: float = 0.7,
    max_confidence: float = 0.9,
    limit: Optional[int] = None
) -> List[Tuple]:
    """Get entities in review queue."""
    cur = conn.cursor()
    
    conditions = []
    params = []
    
    if collection_slug:
        conditions.append("c.slug = %s")
        params.append(collection_slug)
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
        SELECT 
            em.id,
            em.surface,
            em.confidence,
            em.method,
            e.canonical_name,
            e.entity_type,
            c.text,
            cm.document_id,
            d.source_name,
            c.id as chunk_id
        FROM entity_mentions em
        JOIN entities e ON em.entity_id = e.id
        JOIN chunks c ON em.chunk_id = c.id
        JOIN chunk_metadata cm ON c.id = cm.chunk_id
        JOIN documents d ON cm.document_id = d.id
        JOIN collections col ON d.collection_id = col.id
        WHERE {where_clause}
        AND em.confidence >= %s AND em.confidence < %s
        ORDER BY em.confidence DESC
    """
    
    params.extend([min_confidence, max_confidence])
    
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query, params)
    return cur.fetchall()


def review_entity(
    conn,
    mention_id: int,
    decision: str,
    reviewer: Optional[str] = None,
    notes: Optional[str] = None
):
    """Record review decision."""
    cur = conn.cursor()
    
    # For now, just log the decision
    # In future, could update entity_mentions or create review records
    print(f"Review decision for mention {mention_id}: {decision}", file=sys.stderr)
    if reviewer:
        print(f"  Reviewer: {reviewer}", file=sys.stderr)
    if notes:
        print(f"  Notes: {notes}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Review entity extractions in confidence threshold range"
    )
    parser.add_argument(
        "--collection",
        help="Collection slug"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)"
    )
    parser.add_argument(
        "--max-confidence",
        type=float,
        default=0.9,
        help="Maximum confidence threshold (default: 0.9)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of results"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive review mode"
    )
    
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        results = get_review_queue(
            conn,
            collection_slug=args.collection,
            min_confidence=args.min_confidence,
            max_confidence=args.max_confidence,
            limit=args.limit
        )
        
        print(f"Found {len(results)} entities in review queue\n", file=sys.stderr)
        
        for row in results:
            mention_id, surface, confidence, method, canonical_name, entity_type, chunk_text, doc_id, doc_name, chunk_id = row
            
            # Show context around mention
            context_start = max(0, chunk_text.find(surface) - 50)
            context_end = min(len(chunk_text), chunk_text.find(surface) + len(surface) + 50)
            context = chunk_text[context_start:context_end]
            
            print(f"Mention ID: {mention_id}")
            print(f"  Surface: '{surface}'")
            print(f"  Entity: {canonical_name} ({entity_type})")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Method: {method}")
            print(f"  Document: {doc_name} (ID: {doc_id})")
            print(f"  Context: ...{context}...")
            print()
            
            if args.interactive:
                decision = input("Decision (accept/reject/skip): ").strip().lower()
                if decision in ['accept', 'reject']:
                    reviewer = input("Reviewer name (optional): ").strip() or None
                    notes = input("Notes (optional): ").strip() or None
                    review_entity(conn, mention_id, decision, reviewer, notes)
                elif decision == 'skip':
                    continue
                print()
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
