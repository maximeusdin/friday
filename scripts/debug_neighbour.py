#!/usr/bin/env python3
"""Debug why NEIGHBOUR is being rejected"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn

conn = get_conn()
cur = conn.cursor()

print("=== NEIGHBOUR aliases (normalized to 'neighbour') ===")
cur.execute("""
    SELECT 
        ea.entity_id,
        ea.alias,
        ea.alias_norm,
        ea.alias_class,
        ea.is_auto_match,
        e.entity_type,
        e.canonical_name
    FROM entity_aliases ea
    JOIN entities e ON e.id = ea.entity_id
    WHERE ea.alias_norm = 'neighbour'
      AND EXISTS (
          SELECT 1 FROM entity_aliases ea_filter
          JOIN concordance_entries ce_filter ON ce_filter.id = ea_filter.entry_id
          JOIN concordance_sources cs_filter ON cs_filter.id = ce_filter.source_id
          WHERE ea_filter.entity_id = ea.entity_id
            AND cs_filter.slug = 'vassiliev_venona_index_full_spans'
      )
    ORDER BY ea.entity_id, ea.id
    LIMIT 20
""")

for row in cur.fetchall():
    eid, alias, norm, alias_class, is_auto_match, entity_type, canonical = row
    print(f"  Entity {eid} ({canonical}, type={entity_type}):")
    print(f"    alias='{alias}', norm='{norm}', class={alias_class}, is_auto_match={is_auto_match}")

conn.close()
