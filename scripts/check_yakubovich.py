#!/usr/bin/env python3
"""
Check why Yakubovich has auto_match disabled.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn

conn = get_conn()
try:
    with conn.cursor() as cur:
        # Find entity with "Yakubovich" in name or aliases
        cur.execute("""
            SELECT DISTINCT
                e.id AS entity_id,
                e.canonical_name,
                e.entity_type,
                ea.id AS alias_id,
                ea.alias,
                ea.alias_norm,
                ea.is_auto_match,
                ea.alias_class,
                ea.allow_ambiguous_person_token
            FROM entities e
            LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
            WHERE LOWER(e.canonical_name) LIKE '%yakubovich%'
               OR LOWER(ea.alias) LIKE '%yakubovich%'
            ORDER BY e.id, ea.id
        """)
        
        rows = cur.fetchall()
        if not rows:
            print("No entity found with 'Yakubovich' in name or aliases")
        else:
            current_entity = None
            for row in rows:
                entity_id, canonical_name, entity_type, alias_id, alias, alias_norm, is_auto_match, alias_class, allow_ambiguous = row
                
                if current_entity != entity_id:
                    if current_entity is not None:
                        print()
                    print(f"Entity ID: {entity_id}")
                    print(f"  Canonical Name: {canonical_name}")
                    print(f"  Entity Type: {entity_type}")
                    print(f"  Aliases:")
                    current_entity = entity_id
                
                if alias:
                    print(f"    - Alias ID {alias_id}: '{alias}'")
                    print(f"      Normalized: '{alias_norm}'")
                    print(f"      is_auto_match: {is_auto_match}")
                    print(f"      alias_class: {alias_class}")
                    print(f"      allow_ambiguous_person_token: {allow_ambiguous}")
                    
                    # Check if it would be disabled by placeholder check
                    alias_lower = alias.lower()
                    placeholder_patterns = [
                        'unidentified', 'unknown', 'unnamed', 'template', 'placeholder',
                        'source/agent', 'intelligence source'
                    ]
                    is_placeholder = any(pattern in alias_lower for pattern in placeholder_patterns)
                    if is_placeholder:
                        print(f"      ⚠️  Would be disabled by placeholder check")
                    
                    # Check if it would be disabled by single-token person_given check
                    if entity_type == 'person' and alias_class == 'person_given':
                        tokens = alias_norm.split()
                        if len(tokens) == 1 and not allow_ambiguous:
                            print(f"      ⚠️  Would be disabled by single-token person_given check")
            
            print()
finally:
    conn.close()
