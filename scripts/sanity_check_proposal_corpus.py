#!/usr/bin/env python3
"""
Sanity check queries for the proposal corpus system.

Run after:
1. Initial migration
2. After refreshing entity_surface_stats
3. After adding/modifying overrides

Usage:
    python scripts/sanity_check_proposal_corpus.py
"""

import sys
from pathlib import Path


import psycopg2
import os


def get_conn():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "neh"),
        user=os.getenv("DB_USER", "neh"),
        password=os.getenv("DB_PASS", "neh"),
    )


def run_sanity_checks():
    conn = get_conn()
    cur = conn.cursor()
    
    print("=" * 70)
    print("PROPOSAL CORPUS SANITY CHECKS")
    print("=" * 70)
    
    # 1. Tier summary
    print("\n1. TIER SUMMARY")
    print("-" * 50)
    cur.execute("""
        SELECT tier, COUNT(*) as surfaces, COUNT(DISTINCT entity_id) as entities,
               SUM(mention_count) as mentions, AVG(doc_freq)::numeric(6,2) as avg_doc_freq
        FROM entity_surface_tiers
        GROUP BY tier
        ORDER BY tier
    """)
    print(f"{'Tier':<6} {'Surfaces':>10} {'Entities':>10} {'Mentions':>12} {'AvgDocFreq':>12}")
    print("-" * 50)
    for row in cur.fetchall():
        print(f"{row[0]:<6} {row[1]:>10} {row[2]:>10} {row[3]:>12} {row[4]:>12}")
    
    # 2. Top Tier 2 surfaces by frequency (suspicious short/common tokens)
    print("\n2. SUSPICIOUS TIER 2 SURFACES (short or common)")
    print("-" * 50)
    print("Looking for surfaces that might need banning...")
    cur.execute("""
        SELECT surface_norm, surface_display, entity_type, doc_freq, surface_length
        FROM entity_surface_tiers
        WHERE tier = 2
          AND (surface_length <= 3 OR doc_freq >= 10)
        ORDER BY doc_freq DESC
        LIMIT 20
    """)
    print(f"{'Surface':<20} {'Display':<20} {'Type':<15} {'DocFreq':>8} {'Len':>4}")
    print("-" * 70)
    for row in cur.fetchall():
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<15} {row[3]:>8} {row[4]:>4}")
    
    # 3. Surfaces with mixed types (type_stable=false)
    print("\n3. MIXED-TYPE SURFACES (type_stable=false)")
    print("-" * 50)
    print("These are excluded from Tier 1/2 by design:")
    cur.execute("""
        SELECT surface_norm, surface_display, doc_freq, 
               (SELECT COUNT(DISTINCT entity_type) 
                FROM entity_mentions em 
                JOIN entities e ON e.id = em.entity_id 
                WHERE em.surface_norm = ess.surface_norm) as type_count
        FROM entity_surface_stats ess
        WHERE type_stable = FALSE
        ORDER BY doc_freq DESC
        LIMIT 15
    """)
    print(f"{'Surface':<25} {'Display':<20} {'DocFreq':>8} {'TypeCount':>10}")
    print("-" * 65)
    for row in cur.fetchall():
        print(f"{row[0]:<25} {row[1]:<20} {row[2]:>8} {row[3]:>10}")
    
    # 4. Override statistics
    print("\n4. OVERRIDE STATISTICS")
    print("-" * 50)
    
    # Count overrides by type
    cur.execute("""
        SELECT 
            'Prefer (forced_entity_id)' as override_type,
            COUNT(*) as count
        FROM entity_alias_overrides
        WHERE forced_entity_id IS NOT NULL
        UNION ALL
        SELECT 
            'Ban entity (banned_entity_id)',
            COUNT(*)
        FROM entity_alias_overrides
        WHERE banned_entity_id IS NOT NULL
        UNION ALL
        SELECT 
            'Ban surface (banned=true)',
            COUNT(*)
        FROM entity_alias_overrides
        WHERE banned = TRUE
        UNION ALL
        SELECT 
            'Total overrides',
            COUNT(*)
        FROM entity_alias_overrides
    """)
    print(f"{'Override Type':<35} {'Count':>10}")
    print("-" * 45)
    for row in cur.fetchall():
        print(f"{row[0]:<35} {row[1]:>10}")
    
    # 5. Override impact estimate (if any overrides exist)
    cur.execute("SELECT COUNT(*) FROM entity_alias_overrides")
    override_count = cur.fetchone()[0]
    
    if override_count > 0:
        print("\n5. OVERRIDE IMPACT")
        print("-" * 50)
        
        # Show which surfaces have overrides
        cur.execute("""
            SELECT surface_norm, scope, 
                   CASE 
                       WHEN banned = TRUE THEN 'BANNED SURFACE'
                       WHEN banned_entity_id IS NOT NULL THEN 'BAN ENTITY ' || banned_entity_id::text
                       WHEN forced_entity_id IS NOT NULL THEN 'PREFER ENTITY ' || forced_entity_id::text
                   END as action,
                   note
            FROM entity_alias_overrides
            ORDER BY surface_norm
            LIMIT 20
        """)
        print(f"{'Surface':<20} {'Scope':<12} {'Action':<25} {'Note':<30}")
        print("-" * 90)
        for row in cur.fetchall():
            note = (row[3] or "")[:30]
            print(f"{row[0]:<20} {row[1] or 'global':<12} {row[2]:<25} {note:<30}")
    else:
        print("\n5. NO OVERRIDES CONFIGURED")
        print("-" * 50)
        print("Consider adding overrides for problematic surfaces:")
        print("  INSERT INTO entity_alias_overrides (surface_norm, scope, banned, note)")
        print("  VALUES ('the', 'global', TRUE, 'Too common');")
    
    # 6. Review queue grouping stats
    print("\n6. REVIEW QUEUE GROUPING STATS")
    print("-" * 50)
    cur.execute("""
        SELECT 
            COUNT(*) as total_items,
            COUNT(DISTINCT group_key) as unique_groups,
            COUNT(DISTINCT surface_norm) as unique_surfaces,
            COUNT(*) FILTER (WHERE group_key IS NOT NULL) as items_with_group_key
        FROM mention_review_queue
        WHERE status = 'pending'
    """)
    row = cur.fetchone()
    print(f"Total pending items:     {row[0]}")
    print(f"Unique group keys:       {row[1]}")
    print(f"Unique surface norms:    {row[2]}")
    print(f"Items with group_key:    {row[3]}")
    
    # Top collision groups
    cur.execute("""
        SELECT group_key, surface_norm, COUNT(*) as occurrences
        FROM mention_review_queue
        WHERE status = 'pending' AND group_key IS NOT NULL
        GROUP BY group_key, surface_norm
        ORDER BY occurrences DESC
        LIMIT 10
    """)
    results = cur.fetchall()
    if results:
        print("\nTop collision groups (same surface + candidates):")
        print(f"{'Group Key':<25} {'Surface':<20} {'Occurrences':>12}")
        print("-" * 60)
        for row in results:
            print(f"{row[0]:<25} {row[1]:<20} {row[2]:>12}")
    
    # 7. Entity type distribution in Tier 1
    print("\n7. ENTITY TYPE DISTRIBUTION (Tier 1)")
    print("-" * 50)
    cur.execute("""
        SELECT entity_type, COUNT(*) as surfaces, SUM(doc_freq) as total_doc_freq
        FROM entity_surface_tiers
        WHERE tier = 1
        GROUP BY entity_type
        ORDER BY surfaces DESC
    """)
    print(f"{'Entity Type':<20} {'Surfaces':>10} {'TotalDocFreq':>15}")
    print("-" * 50)
    for row in cur.fetchall():
        print(f"{row[0]:<20} {row[1]:>10} {row[2]:>15}")
    
    print("\n" + "=" * 70)
    print("SANITY CHECK COMPLETE")
    print("=" * 70)
    
    conn.close()


if __name__ == "__main__":
    run_sanity_checks()
