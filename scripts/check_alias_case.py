#!/usr/bin/env python3
"""Check alias case matching configuration for yakubovich and silin"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn

conn = get_conn()
cur = conn.cursor()

print("=== Yakubovich aliases ===")
cur.execute("""
    SELECT ea.alias, ea.alias_norm, ea.match_case, e.canonical_name, e.id
    FROM entity_aliases ea
    JOIN entities e ON e.id = ea.entity_id
    WHERE ea.alias_norm = 'yakubovich'
    LIMIT 5
""")
for row in cur.fetchall():
    alias, norm, match_case, canonical, eid = row
    print(f"  Entity {eid} ({canonical}): alias='{alias}', norm='{norm}', match_case={match_case}")

print("\n=== Silin aliases ===")
cur.execute("""
    SELECT ea.alias, ea.alias_norm, ea.match_case, e.canonical_name, e.id
    FROM entity_aliases ea
    JOIN entities e ON e.id = ea.entity_id
    WHERE ea.alias_norm = 'silin'
    LIMIT 5
""")
for row in cur.fetchall():
    alias, norm, match_case, canonical, eid = row
    print(f"  Entity {eid} ({canonical}): alias='{alias}', norm='{norm}', match_case={match_case}")

conn.close()
