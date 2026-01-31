#!/usr/bin/env python3
"""
Build/refresh the alias_lexicon_index table for OCR-aware entity resolution.

This script populates the lexicon from:
1. entity_aliases (all known aliases)
2. entity_surface_stats (proposal corpus with frequency/confidence stats)
3. entity_surface_tiers (tier assignments)

Usage:
    python scripts/build_alias_lexicon.py [--rebuild]
    python scripts/build_alias_lexicon.py --rebuild --source-slug vassiliev_venona_index_20260130
"""

from typing import Optional

import argparse
import sys
import time
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values


def get_conn():
    """Get database connection."""
    return psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='neh',
        user='neh',
        password='neh'
    )


def build_lexicon(conn, rebuild: bool = False, source_slug: Optional[str] = None):
    """Build or refresh the alias lexicon index.
    
    Args:
        conn: Database connection
        rebuild: If True, truncate and rebuild from scratch
        source_slug: If provided, only include aliases from this concordance source
    """
    cur = conn.cursor()
    
    start_time = time.time()
    print(f"Building alias lexicon index...")
    print(f"  Rebuild mode: {rebuild}")
    if source_slug:
        print(f"  Source filter: {source_slug}")
    
    # If rebuild, truncate the table
    if rebuild:
        print("  Truncating existing lexicon...")
        cur.execute("TRUNCATE alias_lexicon_index RESTART IDENTITY")
        conn.commit()
    
    # Step 1: Get all matchable aliases with their entity info
    # Optionally filter by concordance source
    print("  Loading aliases from entity_aliases...")
    
    if source_slug:
        # Filter to only aliases from the specified concordance source
        cur.execute("""
            SELECT 
                ea.alias_norm,
                ea.entity_id,
                e.entity_type,
                ea.alias_class,
                LENGTH(ea.alias_norm) as alias_length,
                array_length(string_to_array(ea.alias_norm, ' '), 1) as token_count
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.entity_id
            JOIN concordance_sources cs ON ea.source_id = cs.id
            WHERE ea.is_matchable = TRUE
              AND ea.alias_norm IS NOT NULL
              AND LENGTH(ea.alias_norm) >= 2
              AND cs.slug = %s
            GROUP BY ea.alias_norm, ea.entity_id, e.entity_type, ea.alias_class
        """, (source_slug,))
    else:
        # No filter - include all matchable aliases
        cur.execute("""
            SELECT 
                ea.alias_norm,
                ea.entity_id,
                e.entity_type,
                ea.alias_class,
                LENGTH(ea.alias_norm) as alias_length,
                array_length(string_to_array(ea.alias_norm, ' '), 1) as token_count
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.entity_id
            WHERE ea.is_matchable = TRUE
              AND ea.alias_norm IS NOT NULL
              AND LENGTH(ea.alias_norm) >= 2
            GROUP BY ea.alias_norm, ea.entity_id, e.entity_type, ea.alias_class
        """)
    aliases = cur.fetchall()
    print(f"    Found {len(aliases)} matchable alias-entity pairs")
    
    # Step 2: Get proposal corpus stats (for aliases that have trusted mentions)
    print("  Loading proposal corpus stats...")
    cur.execute("""
        SELECT 
            surface_norm,
            entity_id,
            doc_freq,
            mention_count,
            confidence_mean
        FROM entity_surface_stats
    """)
    corpus_stats = {(row[0], row[1]): {
        'doc_freq': row[2],
        'mention_count': row[3],
        'confidence_mean': row[4]
    } for row in cur.fetchall()}
    print(f"    Found {len(corpus_stats)} corpus stat entries")
    
    # Step 3: Get tier assignments
    print("  Loading tier assignments...")
    cur.execute("""
        SELECT surface_norm, entity_id, tier
        FROM entity_surface_tiers
    """)
    tier_map = {(row[0], row[1]): row[2] for row in cur.fetchall()}
    print(f"    Found {len(tier_map)} tier assignments")
    
    # Step 4: Build lexicon entries
    print("  Building lexicon entries...")
    entries = []
    for alias_norm, entity_id, entity_type, alias_class, alias_length, token_count in aliases:
        key = (alias_norm, entity_id)
        
        # Get corpus stats if available
        stats = corpus_stats.get(key, {})
        tier = tier_map.get(key)
        
        # Determine if from trusted text (has corpus stats = was seen in trusted text)
        is_trusted = key in corpus_stats
        
        entries.append((
            alias_norm,
            entity_id,
            entity_type,
            stats.get('doc_freq', 0),
            stats.get('mention_count', 0),
            stats.get('confidence_mean'),
            tier,
            is_trusted,
            alias_length,
            token_count or 1,
            alias_class
        ))
    
    print(f"    Prepared {len(entries)} lexicon entries")
    
    # Step 5: Insert/update entries
    print("  Inserting into alias_lexicon_index...")
    
    # Use upsert for incremental updates
    insert_sql = """
        INSERT INTO alias_lexicon_index (
            alias_norm, entity_id, entity_type,
            doc_freq, mention_count, corpus_confidence,
            proposal_tier, is_from_trusted_text,
            alias_length, token_count, alias_class,
            updated_at
        ) VALUES %s
        ON CONFLICT (alias_norm, entity_id) DO UPDATE SET
            entity_type = EXCLUDED.entity_type,
            doc_freq = EXCLUDED.doc_freq,
            mention_count = EXCLUDED.mention_count,
            corpus_confidence = EXCLUDED.corpus_confidence,
            proposal_tier = EXCLUDED.proposal_tier,
            is_from_trusted_text = EXCLUDED.is_from_trusted_text,
            alias_length = EXCLUDED.alias_length,
            token_count = EXCLUDED.token_count,
            alias_class = EXCLUDED.alias_class,
            updated_at = NOW()
    """
    
    # Add timestamp to entries
    entries_with_ts = [e + (datetime.now(),) for e in entries]
    
    # Batch insert
    batch_size = 5000
    for i in range(0, len(entries_with_ts), batch_size):
        batch = entries_with_ts[i:i+batch_size]
        execute_values(cur, insert_sql, batch, page_size=batch_size)
        conn.commit()
        print(f"    Inserted batch {i//batch_size + 1} ({min(i+batch_size, len(entries_with_ts))}/{len(entries_with_ts)})")
    
    # Step 6: Verify and report stats
    print()
    print("=== LEXICON BUILD COMPLETE ===")
    
    cur.execute("SELECT COUNT(*) FROM alias_lexicon_index")
    total = cur.fetchone()[0]
    print(f"  Total entries: {total}")
    
    cur.execute("SELECT COUNT(DISTINCT alias_norm) FROM alias_lexicon_index")
    unique_norms = cur.fetchone()[0]
    print(f"  Unique alias_norms: {unique_norms}")
    
    cur.execute("SELECT COUNT(DISTINCT entity_id) FROM alias_lexicon_index")
    unique_entities = cur.fetchone()[0]
    print(f"  Unique entities: {unique_entities}")
    
    cur.execute("""
        SELECT proposal_tier, COUNT(*) 
        FROM alias_lexicon_index 
        GROUP BY proposal_tier 
        ORDER BY proposal_tier NULLS LAST
    """)
    print("  By tier:")
    for row in cur.fetchall():
        tier = row[0] if row[0] is not None else 'NULL'
        print(f"    Tier {tier}: {row[1]}")
    
    cur.execute("""
        SELECT is_from_trusted_text, COUNT(*) 
        FROM alias_lexicon_index 
        GROUP BY is_from_trusted_text
    """)
    print("  By trusted source:")
    for row in cur.fetchall():
        print(f"    Trusted={row[0]}: {row[1]}")
    
    cur.execute("""
        SELECT entity_type, COUNT(*) 
        FROM alias_lexicon_index 
        GROUP BY entity_type 
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """)
    print("  By entity type (top 10):")
    for row in cur.fetchall():
        print(f"    {row[0]}: {row[1]}")
    
    elapsed = time.time() - start_time
    print(f"\n  Build completed in {elapsed:.1f}s")
    
    return total


def main():
    parser = argparse.ArgumentParser(description='Build alias lexicon index for OCR resolution')
    parser.add_argument('--rebuild', action='store_true', help='Truncate and rebuild from scratch')
    parser.add_argument('--source-slug', type=str, default=None,
                        help='Only include aliases from this concordance source (e.g., vassiliev_venona_index_20260130)')
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        total = build_lexicon(conn, rebuild=args.rebuild, source_slug=args.source_slug)
        print(f"\nDone. Lexicon has {total} entries.")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
