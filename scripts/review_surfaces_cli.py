#!/usr/bin/env python3
"""
Surface-level review CLI for OCR mention candidates.

This CLI shows unique surfaces (not individual mentions) and allows you to:
- Block a surface globally (adds to blocklist, marks all mentions as 'ignore')
- Skip to review later
- Accept a surface (keeps in queue for entity-level review later)

This is much faster than reviewing individual mentions since blocking
one bad surface can eliminate hundreds/thousands of mentions at once.

Usage:
    python scripts/review_surfaces_cli.py
    python scripts/review_surfaces_cli.py --min-count 10  # Only surfaces with 10+ mentions
    python scripts/review_surfaces_cli.py --collection silvermaster
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor


def get_conn():
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', os.environ.get('DB_HOST', 'localhost')),
        port=int(os.environ.get('POSTGRES_PORT', os.environ.get('DB_PORT', '5432'))),
        dbname=os.environ.get('POSTGRES_DB', os.environ.get('DB_NAME', 'neh')),
        user=os.environ.get('POSTGRES_USER', os.environ.get('DB_USER', 'neh')),
        password=os.environ.get('POSTGRES_PASSWORD', os.environ.get('DB_PASS', 'neh'))
    )


@dataclass
class SurfaceInfo:
    surface_norm: str
    mention_count: int
    avg_score: float
    sample_raw_spans: List[str]
    sample_context: str
    top_candidate_name: Optional[str]
    top_candidate_id: Optional[int]
    collections: List[str]


def get_surfaces_for_review(
    conn,
    *,
    collection: Optional[str] = None,
    min_count: int = 1,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: int = 500,
) -> List[SurfaceInfo]:
    """Get unique surfaces with their mention counts."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Increase timeout for this query
    cur.execute("SET statement_timeout = '180s'")
    
    conditions = ["mc.resolution_status = 'queue'"]
    params = []
    
    if collection:
        conditions.append("col.slug = %s")
        params.append(collection)
    
    if min_score is not None:
        conditions.append("mc.resolution_score >= %s")
        params.append(min_score)
    
    if max_score is not None:
        conditions.append("mc.resolution_score <= %s")
        params.append(max_score)
    
    where_clause = " AND ".join(conditions)
    
    # Get surfaces with counts, ordered by count descending (biggest impact first)
    cur.execute(f"""
        WITH surface_stats AS (
            SELECT 
                mc.surface_norm,
                COUNT(*) as mention_count,
                AVG(mc.resolution_score) as avg_score,
                array_agg(DISTINCT mc.raw_span) FILTER (WHERE mc.raw_span IS NOT NULL) as raw_spans,
                array_agg(DISTINCT col.slug) as collections,
                (array_agg(mc.top_candidates->0->>'entity_name'))[1] as top_candidate_name,
                (array_agg((mc.top_candidates->0->>'entity_id')::int))[1] as top_candidate_id
            FROM mention_candidates mc
            JOIN documents d ON d.id = mc.document_id
            JOIN collections col ON col.id = d.collection_id
            WHERE {where_clause}
            GROUP BY mc.surface_norm
            HAVING COUNT(*) >= %s
            ORDER BY COUNT(*) DESC
            LIMIT %s
        )
        SELECT 
            ss.*,
            (
                SELECT substring(c.text from greatest(1, mc2.char_start - 40) for 120)
                FROM mention_candidates mc2
                JOIN chunks c ON c.id = mc2.chunk_id
                WHERE mc2.surface_norm = ss.surface_norm
                  AND mc2.resolution_status = 'queue'
                LIMIT 1
            ) as sample_context
        FROM surface_stats ss
    """, params + [min_count, limit])
    
    surfaces = []
    for row in cur.fetchall():
        raw_spans = row['raw_spans'] or []
        # Deduplicate and limit raw spans
        unique_spans = list(dict.fromkeys(raw_spans))[:5]
        
        surfaces.append(SurfaceInfo(
            surface_norm=row['surface_norm'],
            mention_count=row['mention_count'],
            avg_score=float(row['avg_score']) if row['avg_score'] else 0,
            sample_raw_spans=unique_spans,
            sample_context=row['sample_context'] or '',
            top_candidate_name=row['top_candidate_name'],
            top_candidate_id=row['top_candidate_id'],
            collections=row['collections'] or [],
        ))
    
    return surfaces


def block_surface(conn, surface_norm: str) -> int:
    """
    Block a surface globally:
    1. Add to blocklist
    2. Mark all mentions as 'ignore'
    
    Returns count of mentions affected.
    """
    cur = conn.cursor()
    
    # Increase timeout for bulk update
    cur.execute("SET statement_timeout = '300s'")
    
    # Add to blocklist
    cur.execute("""
        INSERT INTO ocr_variant_blocklist (variant_key, block_type, reason, source)
        VALUES (%s, 'exact', 'surface_review_reject', 'review_surfaces_cli')
        ON CONFLICT (variant_key, pattern_signature) DO NOTHING
    """, (surface_norm,))
    
    # Mark all mentions as ignore (in batches if needed)
    cur.execute("""
        UPDATE mention_candidates
        SET resolution_status = 'ignore',
            resolution_method = 'surface_blocklist',
            resolved_at = NOW()
        WHERE surface_norm = %s AND resolution_status = 'queue'
    """, (surface_norm,))
    
    count = cur.rowcount
    conn.commit()
    
    return count


def display_surface(surface: SurfaceInfo, idx: int, total: int) -> None:
    """Display a surface for review."""
    print("\n" + "=" * 70)
    print(f"[{idx + 1}/{total}]  Surface: '{surface.surface_norm}'")
    print(f"         Mentions: {surface.mention_count}  |  Avg Score: {surface.avg_score:.2f}")
    print(f"         Collections: {', '.join(surface.collections[:3])}")
    print("-" * 70)
    
    # Show sample raw spans (how it appears in documents)
    if surface.sample_raw_spans:
        print(f"Raw forms: {surface.sample_raw_spans}")
    
    # Show sample context
    if surface.sample_context:
        # Clean up whitespace
        context = ' '.join(surface.sample_context.split())
        print(f"Context: ...{context}...")
    
    # Show top candidate if any
    if surface.top_candidate_name:
        print(f"\nTop auto-match: {surface.top_candidate_name} (id={surface.top_candidate_id})")
    
    print("-" * 70)
    print("Commands: [b]lock (reject globally), [s]kip, [q]uit")


def main():
    parser = argparse.ArgumentParser(description='Surface-level review CLI')
    parser.add_argument('--collection', help='Filter by collection slug')
    parser.add_argument('--min-count', type=int, default=1, help='Minimum mention count (default: 1)')
    parser.add_argument('--min-score', type=float, help='Minimum resolution score')
    parser.add_argument('--max-score', type=float, help='Maximum resolution score')
    parser.add_argument('--limit', type=int, default=500, help='Max surfaces to load')
    args = parser.parse_args()
    
    conn = get_conn()
    
    print("Loading surfaces for review...")
    print("(This may take a minute for large datasets)")
    
    surfaces = get_surfaces_for_review(
        conn,
        collection=args.collection,
        min_count=args.min_count,
        min_score=args.min_score,
        max_score=args.max_score,
        limit=args.limit,
    )
    
    if not surfaces:
        print("No surfaces to review.")
        return
    
    total_mentions = sum(s.mention_count for s in surfaces)
    print(f"Loaded {len(surfaces)} unique surfaces ({total_mentions:,} total mentions)")
    
    # Stats
    stats = {
        'blocked': 0,
        'blocked_mentions': 0,
        'skipped': 0,
    }
    
    idx = 0
    while idx < len(surfaces):
        surface = surfaces[idx]
        
        display_surface(surface, idx, len(surfaces))
        
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            cmd = 'q'
        
        if not cmd or cmd == 's' or cmd == 'skip':
            stats['skipped'] += 1
            idx += 1
            continue
        
        if cmd == 'q' or cmd == 'quit':
            break
        
        if cmd == 'b' or cmd == 'block':
            count = block_surface(conn, surface.surface_norm)
            stats['blocked'] += 1
            stats['blocked_mentions'] += count
            print(f"BLOCKED: '{surface.surface_norm}' - {count} mentions marked as ignore")
            idx += 1
            continue
        
        print(f"Unknown command: {cmd}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Surfaces blocked:  {stats['blocked']}")
    print(f"Mentions removed:  {stats['blocked_mentions']:,}")
    print(f"Surfaces skipped:  {stats['skipped']}")
    print("=" * 70)
    
    conn.close()


if __name__ == '__main__':
    main()
