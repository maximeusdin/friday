#!/usr/bin/env python3
"""
Export all entities and aliases to CSV.

Outputs a CSV file with columns:
- entity_id
- entity_type
- canonical_name
- alias
- alias_norm
- alias_class
- is_auto_match
- match_case
- min_chars
- kind
- description
- external_ids
"""

import os
import sys
import csv
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn


def export_entities_aliases(output_file: str, conn):
    """Export all entities and aliases to CSV."""
    with conn.cursor() as cur:
        # Check which columns exist in entities table
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'entities' 
            AND column_name IN ('description', 'external_ids')
        """)
        entity_cols = {row[0] for row in cur.fetchall()}
        
        # Check which policy columns exist in entity_aliases table
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'entity_aliases' 
            AND column_name IN ('alias_class', 'is_auto_match', 'match_case', 'min_chars', 'kind')
        """)
        policy_cols = {row[0] for row in cur.fetchall()}
        
        # Build query with available columns
        select_cols = [
            "e.id AS entity_id",
            "e.entity_type",
            "e.canonical_name",
        ]
        
        if 'description' in entity_cols:
            select_cols.append("e.description")
        else:
            select_cols.append("NULL AS description")
        
        if 'external_ids' in entity_cols:
            select_cols.append("e.external_ids")
        else:
            select_cols.append("NULL AS external_ids")
        
        select_cols.extend([
            "ea.id AS alias_id",
            "ea.alias",
            "ea.alias_norm",
        ])
        
        if 'kind' in policy_cols:
            select_cols.append("ea.kind")
        else:
            select_cols.append("NULL AS kind")
        
        if 'alias_class' in policy_cols:
            select_cols.append("ea.alias_class")
        else:
            select_cols.append("NULL AS alias_class")
        
        if 'is_auto_match' in policy_cols:
            select_cols.append("COALESCE(ea.is_auto_match, true) AS is_auto_match")
        else:
            select_cols.append("true AS is_auto_match")
        
        if 'match_case' in policy_cols:
            select_cols.append("COALESCE(ea.match_case, 'any') AS match_case")
        else:
            select_cols.append("'any' AS match_case")
        
        if 'min_chars' in policy_cols:
            select_cols.append("COALESCE(ea.min_chars, 1) AS min_chars")
        elif 'min_token_len' in policy_cols:
            select_cols.append("COALESCE(ea.min_token_len, 1) AS min_chars")
        else:
            select_cols.append("1 AS min_chars")
        
        query = f"""
            SELECT {', '.join(select_cols)}
            FROM entities e
            LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
            ORDER BY e.id, ea.id
        """
        
        cur.execute(query)
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'entity_id',
                'entity_type',
                'canonical_name',
                'description',
                'external_ids',
                'alias_id',
                'alias',
                'alias_norm',
                'kind',
                'alias_class',
                'is_auto_match',
                'match_case',
                'min_chars',
            ])
            
            # Write rows
            for row in cur.fetchall():
                # Convert external_ids JSONB to string
                external_ids = row[4]
                if external_ids:
                    import json
                    external_ids_str = json.dumps(external_ids)
                else:
                    external_ids_str = ''
                
                writer.writerow([
                    row[0],  # entity_id
                    row[1],  # entity_type
                    row[2],  # canonical_name
                    row[3] or '',  # description
                    external_ids_str,  # external_ids
                    row[5] if row[5] else '',  # alias_id
                    row[6] if row[6] else '',  # alias
                    row[7] if row[7] else '',  # alias_norm
                    row[8] if len(row) > 8 else '',  # kind
                    row[9] if len(row) > 9 else '',  # alias_class
                    row[10] if len(row) > 10 else True,  # is_auto_match
                    row[11] if len(row) > 11 else 'any',  # match_case
                    row[12] if len(row) > 12 else 1,  # min_chars
                ])
        
        # Get counts
        cur.execute("SELECT COUNT(*) FROM entities")
        entity_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM entity_aliases")
        alias_count = cur.fetchone()[0]
        
        print(f"Exported {entity_count} entities and {alias_count} aliases to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Export all entities and aliases to CSV")
    parser.add_argument(
        "--output",
        "-o",
        default="entities_aliases.csv",
        help="Output CSV file path (default: entities_aliases.csv)",
    )
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        export_entities_aliases(args.output, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
