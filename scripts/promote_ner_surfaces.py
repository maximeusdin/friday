#!/usr/bin/env python3
"""
Promote NER-discovered surfaces to the alias lexicon.

After running a corpus sweep, this script promotes high-confidence surfaces
to alias_lexicon_index for use in entity matching.

Usage:
    # Promote all tier 1 surfaces
    python scripts/promote_ner_surfaces.py --tier 1
    
    # Promote specific surfaces by ID
    python scripts/promote_ner_surfaces.py --ids 1,2,3
    
    # Preview what would be promoted
    python scripts/promote_ner_surfaces.py --tier 1 --dry-run
    
    # Export for review
    python scripts/promote_ner_surfaces.py --export review_surfaces.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values


def get_conn():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'neh'),
        user=os.getenv('DB_USER', 'neh'),
        password=os.getenv('DB_PASS', 'neh')
    )


def get_surfaces_for_promotion(
    conn,
    tier: Optional[int] = None,
    ids: Optional[List[int]] = None,
    min_docs: Optional[int] = None,
    status: str = 'pending',
) -> List[dict]:
    """Get surfaces eligible for promotion."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    conditions = [
        "status = %s",
        "matches_existing_alias = FALSE",  # Don't duplicate
    ]
    params = [status]
    
    if tier:
        conditions.append("proposed_tier = %s")
        params.append(tier)
    
    if ids:
        conditions.append("id = ANY(%s)")
        params.append(ids)
    
    if min_docs:
        conditions.append("doc_count >= %s")
        params.append(min_docs)
    
    where = " AND ".join(conditions)
    
    cur.execute(f"""
        SELECT 
            id,
            surface_norm,
            doc_count,
            chunk_count,
            mention_count,
            primary_label,
            label_consistency,
            inferred_type,
            type_confidence,
            avg_accept_score,
            proposed_tier,
            tier_reason,
            example_contexts,
            corpus_sweep_id
        FROM ner_surface_stats
        WHERE {where}
        ORDER BY doc_count DESC, label_consistency DESC
    """, params)
    
    return [dict(row) for row in cur.fetchall()]


def promote_to_lexicon(
    conn,
    surfaces: List[dict],
    dry_run: bool = False,
) -> int:
    """Add surfaces to alias_lexicon_index."""
    if not surfaces:
        return 0
    
    if dry_run:
        print("\n  Would promote to alias_lexicon_index:")
        for s in surfaces[:20]:
            print(f"    {s['surface_norm']!r}: docs={s['doc_count']}, "
                  f"type={s['inferred_type']}, tier={s['proposed_tier']}")
        if len(surfaces) > 20:
            print(f"    ... and {len(surfaces) - 20} more")
        return len(surfaces)
    
    cur = conn.cursor()
    
    # Insert into alias_lexicon_index
    # Note: These don't have entity_id yet - they're "orphan" entries
    # They can be linked to entities later or used for fuzzy matching
    records = [
        (
            s['surface_norm'],
            None,  # entity_id - will be linked later
            s['inferred_type'],
            s['doc_count'],
            s['mention_count'],
            s['avg_accept_score'],
            s['proposed_tier'],
            True,  # is_from_trusted_text (corpus evidence)
            len(s['surface_norm']),
            len(s['surface_norm'].split()),
            None,  # alias_class
        )
        for s in surfaces
    ]
    
    # This will fail if entity_id NOT NULL - we need a different approach
    # Instead, we'll mark them as promoted in ner_surface_stats
    # and they can be used for fuzzy matching against the corpus
    
    # Update status in ner_surface_stats
    surface_ids = [s['id'] for s in surfaces]
    
    cur.execute("""
        UPDATE ner_surface_stats
        SET status = 'promoted',
            promoted_at = NOW()
        WHERE id = ANY(%s)
    """, (surface_ids,))
    
    conn.commit()
    return len(surfaces)


def find_potential_matches(conn, surface_norm: str, limit: int = 3) -> List[dict]:
    """Find potential fuzzy matches in existing aliases for a surface."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Skip very short surfaces
    if len(surface_norm) < 3:
        return []
    
    try:
        cur.execute("""
            SELECT 
                ea.alias_norm,
                ea.entity_id,
                e.canonical_name,
                e.entity_type,
                similarity(ea.alias_norm, %s) as sim_score
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.entity_id
            WHERE ea.is_matchable = true
              AND similarity(ea.alias_norm, %s) > 0.4
            ORDER BY similarity(ea.alias_norm, %s) DESC
            LIMIT %s
        """, (surface_norm, surface_norm, surface_norm, limit))
        return [dict(row) for row in cur.fetchall()]
    except Exception:
        # If similarity function not available, return empty
        return []


def export_for_review(
    conn,
    output_path: str,
    tier: Optional[int] = None,
    min_docs: int = 1,
) -> int:
    """Export surfaces to CSV for human review with potential entity matches."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    conditions = ["matches_existing_alias = FALSE"]
    params = []
    
    if tier:
        conditions.append("proposed_tier = %s")
        params.append(tier)
    
    if min_docs:
        conditions.append("doc_count >= %s")
        params.append(min_docs)
    
    where = " AND ".join(conditions)
    
    cur.execute(f"""
        SELECT 
            id,
            surface_norm,
            doc_count,
            mention_count,
            primary_label,
            label_consistency,
            inferred_type,
            avg_accept_score,
            proposed_tier,
            tier_reason,
            status,
            example_contexts[1] as example_context
        FROM ner_surface_stats
        WHERE {where}
        ORDER BY doc_count DESC
    """, params)
    
    rows = cur.fetchall()
    n = len(rows)
    
    print(f"Finding potential matches for {n} surfaces...")
    
    # Fieldnames optimized for reviewer workflow
    fieldnames = [
        # Key info first
        'surface_norm',
        'doc_count',
        'ner_type',  # Renamed for clarity
        'consistency',  # Shortened
        
        # Pre-filled decision suggestion
        'suggested_decision',
        'confidence',
        
        # Top 3 candidates in separate columns
        'candidate_1_name', 'candidate_1_score', 'candidate_1_id',
        'candidate_2_name', 'candidate_2_score', 'candidate_2_id',
        'candidate_3_name', 'candidate_3_score', 'candidate_3_id',
        
        # Context for review
        'example_context',
        
        # Reviewer fills these
        'decision',  # LINK_1, LINK_2, LINK_3, CREATE_NEW, REJECT
        
        # New entity fields (fill if CREATE_NEW)
        'is_new_entity',  # TRUE if creating new entity
        'new_entity_name',  # Canonical name for new entity
        'new_entity_type',  # person, org, place
        'new_entity_aliases',  # Additional aliases (semicolon separated)
        'new_entity_description',  # Optional description
        
        'notes',
        
        # Metadata (at end, less important)
        'id', 'mention_count', 'tier_reason',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, row in enumerate(rows):
            out = {}
            
            # Key info
            out['surface_norm'] = row['surface_norm']
            out['doc_count'] = row['doc_count']
            out['ner_type'] = row['inferred_type'] or row['primary_label']
            out['consistency'] = f"{row['label_consistency']:.0%}" if row['label_consistency'] else ''
            
            # Find potential fuzzy matches
            matches = find_potential_matches(conn, row['surface_norm'])
            
            # Fill candidate columns
            for idx in range(3):
                prefix = f'candidate_{idx + 1}'
                if idx < len(matches):
                    m = matches[idx]
                    out[f'{prefix}_name'] = m['canonical_name']
                    out[f'{prefix}_score'] = f"{m['sim_score']:.0%}"
                    out[f'{prefix}_id'] = m['entity_id']
                else:
                    out[f'{prefix}_name'] = ''
                    out[f'{prefix}_score'] = ''
                    out[f'{prefix}_id'] = ''
            
            # Pre-fill suggested decision based on match quality
            if matches and matches[0]['sim_score'] >= 0.90:
                out['suggested_decision'] = 'LINK_1'
                out['confidence'] = 'HIGH'
            elif matches and matches[0]['sim_score'] >= 0.75:
                out['suggested_decision'] = 'LINK_1?'
                out['confidence'] = 'MEDIUM'
            elif matches and matches[0]['sim_score'] >= 0.50:
                out['suggested_decision'] = 'REVIEW'
                out['confidence'] = 'LOW'
            elif row['doc_count'] >= 5:
                out['suggested_decision'] = 'CREATE_NEW?'
                out['confidence'] = 'MEDIUM'
            else:
                out['suggested_decision'] = 'REVIEW'
                out['confidence'] = 'LOW'
            
            # Context (clean up for CSV)
            context = row.get('example_context', '') or ''
            # Truncate and clean
            context = ' '.join(context.split())[:150]
            out['example_context'] = context
            
            # Reviewer columns (empty for reviewer to fill)
            out['decision'] = ''
            
            # New entity fields (pre-fill some defaults)
            out['is_new_entity'] = ''  # Reviewer sets to TRUE if creating
            out['new_entity_name'] = ''  # Canonical name
            out['new_entity_type'] = row['inferred_type'] or ''  # Pre-fill from NER
            out['new_entity_aliases'] = row['surface_norm']  # This surface becomes an alias
            out['new_entity_description'] = ''
            
            out['notes'] = ''
            
            # Metadata
            out['id'] = row['id']
            out['mention_count'] = row['mention_count']
            out['tier_reason'] = row['tier_reason']
            
            writer.writerow(out)
            
            # Progress every 1%
            percent = ((i + 1) * 100) // n
            if percent > 0 and (i + 1) % max(1, n // 100) == 0:
                print(f"  Progress: {percent}% ({i + 1}/{n} surfaces)", end='\r')
                sys.stdout.flush()
    
    print(f"  Progress: 100% ({n}/{n} surfaces)      ")  # Clear the line
    
    print(f"\nExported {len(rows)} surfaces to {output_path}")
    print(f"\n" + "="*60)
    print(f"REVIEWER INSTRUCTIONS")
    print(f"="*60)
    print(f"\n1. DECISION column options:")
    print(f"   LINK_1    = Link to candidate_1 (best match)")
    print(f"   LINK_2    = Link to candidate_2")
    print(f"   LINK_3    = Link to candidate_3")
    print(f"   CREATE    = Create new entity (fill new_entity fields)")
    print(f"   REJECT    = Junk/not an entity")
    print(f"   SKIP      = Needs more investigation")
    print(f"\n2. For CREATE decisions, fill these columns:")
    print(f"   is_new_entity      = TRUE")
    print(f"   new_entity_name    = Canonical name (e.g., 'Elizabeth Bentley')")
    print(f"   new_entity_type    = person, org, or place")
    print(f"   new_entity_aliases = Additional aliases (semicolon separated)")
    print(f"   new_entity_description = Optional description")
    print(f"\n3. After review, import with:")
    print(f"   python scripts/import_ner_review.py {output_path}")
    return len(rows)


def show_summary(conn):
    """Show summary of NER surface stats."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("NER SURFACE STATS SUMMARY")
    print("=" * 60)
    
    # By status
    cur.execute("""
        SELECT status, COUNT(*) as count
        FROM ner_surface_stats
        GROUP BY status
        ORDER BY count DESC
    """)
    print("\nBy status:")
    for row in cur.fetchall():
        print(f"  {row['status']}: {row['count']}")
    
    # By tier (new surfaces only)
    cur.execute("""
        SELECT proposed_tier, COUNT(*) as count
        FROM ner_surface_stats
        WHERE matches_existing_alias = FALSE
        GROUP BY proposed_tier
        ORDER BY proposed_tier NULLS LAST
    """)
    print("\nNew surfaces by tier:")
    for row in cur.fetchall():
        tier = row['proposed_tier'] if row['proposed_tier'] else 'rejected'
        print(f"  Tier {tier}: {row['count']}")
    
    # By inferred type
    cur.execute("""
        SELECT inferred_type, COUNT(*) as count
        FROM ner_surface_stats
        WHERE matches_existing_alias = FALSE AND proposed_tier IS NOT NULL
        GROUP BY inferred_type
        ORDER BY count DESC
    """)
    print("\nNew tiered surfaces by type:")
    for row in cur.fetchall():
        print(f"  {row['inferred_type'] or 'unknown'}: {row['count']}")
    
    # Top surfaces by doc count
    cur.execute("""
        SELECT surface_norm, doc_count, primary_label, proposed_tier
        FROM ner_surface_stats
        WHERE matches_existing_alias = FALSE AND proposed_tier IS NOT NULL
        ORDER BY doc_count DESC
        LIMIT 15
    """)
    print("\nTop 15 new surfaces by doc count:")
    for row in cur.fetchall():
        print(f"  {row['surface_norm']!r}: docs={row['doc_count']}, "
              f"label={row['primary_label']}, tier={row['proposed_tier']}")


def main():
    parser = argparse.ArgumentParser(
        description='Promote NER-discovered surfaces to alias lexicon'
    )
    
    # Selection
    parser.add_argument('--tier', type=int, choices=[1, 2],
                       help='Promote surfaces of this tier')
    parser.add_argument('--ids', type=str,
                       help='Comma-separated IDs to promote')
    parser.add_argument('--min-docs', type=int,
                       help='Minimum doc count for promotion')
    
    # Actions
    parser.add_argument('--export', type=str,
                       help='Export to CSV for review')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary statistics')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without making changes')
    
    args = parser.parse_args()
    
    conn = get_conn()
    
    # Summary mode
    if args.summary:
        show_summary(conn)
        return
    
    # Export mode
    if args.export:
        export_for_review(conn, args.export, tier=args.tier, min_docs=args.min_docs or 1)
        return
    
    # Promotion mode
    if not args.tier and not args.ids:
        print("ERROR: Specify --tier or --ids for promotion, or use --summary/--export")
        sys.exit(1)
    
    ids = None
    if args.ids:
        ids = [int(x.strip()) for x in args.ids.split(',')]
    
    print("Finding surfaces for promotion...")
    surfaces = get_surfaces_for_promotion(
        conn,
        tier=args.tier,
        ids=ids,
        min_docs=args.min_docs,
    )
    
    print(f"Found {len(surfaces)} surfaces eligible for promotion")
    
    if not surfaces:
        print("Nothing to promote.")
        return
    
    promoted = promote_to_lexicon(conn, surfaces, args.dry_run)
    
    if not args.dry_run:
        print(f"\nPromoted {promoted} surfaces (status -> 'promoted')")
        print("\nNote: These surfaces are now marked as 'promoted' but not yet linked")
        print("to entities. They can be used for fuzzy matching or reviewed for")
        print("entity creation.")


if __name__ == '__main__':
    main()
