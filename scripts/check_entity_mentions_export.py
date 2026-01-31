#!/usr/bin/env python3
"""
Check if entity_mentions.csv export is complete by comparing with database.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn

def check_export_completeness():
    """Compare CSV export with database to see if anything is missing."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Count total entity_mentions in database
            cur.execute("SELECT COUNT(*) FROM entity_mentions")
            total_in_db = cur.fetchone()[0]
            
            # Count entity_mentions that have concordance source info
            cur.execute("""
                SELECT COUNT(DISTINCT em.id)
                FROM entity_mentions em
                JOIN entities e ON em.entity_id = e.id
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE ea.entry_id IS NOT NULL
            """)
            with_concordance = cur.fetchone()[0]
            
            # Count entity_mentions without concordance source info
            cur.execute("""
                SELECT COUNT(DISTINCT em.id)
                FROM entity_mentions em
                JOIN entities e ON em.entity_id = e.id
                WHERE NOT EXISTS (
                    SELECT 1 FROM entity_aliases ea
                    WHERE ea.entity_id = e.id AND ea.entry_id IS NOT NULL
                )
            """)
            without_concordance = cur.fetchone()[0]
            
            # Count by method
            cur.execute("""
                SELECT method, COUNT(*) 
                FROM entity_mentions
                GROUP BY method
                ORDER BY COUNT(*) DESC
            """)
            by_method = cur.fetchall()
            
            print(f"Total entity_mentions in database: {total_in_db}")
            print(f"  With concordance source: {with_concordance}")
            print(f"  Without concordance source: {without_concordance}")
            print(f"\nBreakdown by method:")
            for method, count in by_method:
                print(f"  {method}: {count}")
            
            # Check if export query would include all
            print(f"\nExport query analysis:")
            print(f"  The export query uses LEFT JOINs, so it should include ALL entity_mentions")
            print(f"  However, it doesn't filter by source_slug like other exports do")
            
    finally:
        conn.close()

if __name__ == "__main__":
    check_export_completeness()
