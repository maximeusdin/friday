#!/usr/bin/env python3
"""
Verify that required database indexes exist for index retrieval operations.

This script checks for the presence of indexes that are critical for
performance of index retrieval queries. Run this after migrations or
as part of CI to catch missing indexes early.

Usage:
    python scripts/verify_indexes.py
    
    # With custom database URL
    DATABASE_URL=postgresql://... python scripts/verify_indexes.py
"""

import os
import sys
from typing import List, Tuple, Dict, Any

# Required indexes for index retrieval performance
REQUIRED_INDEXES = [
    # Entity mentions indexes
    {
        "table": "entity_mentions",
        "columns": ["entity_id", "chunk_id"],
        "purpose": "Fast lookup by entity_id for FIRST_MENTION/MENTIONS",
    },
    {
        "table": "entity_mentions", 
        "columns": ["chunk_id", "entity_id"],
        "purpose": "Fast co-mention grouping (FIRST_CO_MENTION)",
    },
    {
        "table": "entity_mentions",
        "columns": ["chunk_id"],
        "purpose": "Join from chunk to mentions",
    },
    
    # Date mentions indexes
    {
        "table": "date_mentions",
        "columns": ["date_start", "date_end"],
        "purpose": "Date range queries (DATE_RANGE_FILTER)",
    },
    {
        "table": "date_mentions",
        "columns": ["chunk_id"],
        "purpose": "Join from chunk to date mentions",
    },
    
    # Result set pagination
    {
        "table": "result_set_chunks",
        "columns": ["result_set_id", "chunk_id"],
        "purpose": "Primary key for result set chunks",
    },
]

# Optional but recommended indexes
RECOMMENDED_INDEXES = [
    {
        "table": "entity_mentions",
        "columns": ["document_id", "entity_id"],
        "purpose": "Document-window co-occurrence queries",
    },
    {
        "table": "chunk_metadata",
        "columns": ["date_min"],
        "purpose": "Chronological ordering",
    },
]


def get_existing_indexes(conn) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all indexes from the database, grouped by table.
    
    Returns:
        dict mapping table_name -> list of index info dicts
    """
    query = """
        SELECT
            t.relname as table_name,
            i.relname as index_name,
            array_agg(a.attname ORDER BY x.ordinality) as columns,
            ix.indisunique as is_unique,
            ix.indisprimary as is_primary
        FROM
            pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            CROSS JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS x(attnum, ordinality)
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = x.attnum
        WHERE
            n.nspname = 'public'
            AND t.relname IN ('entity_mentions', 'date_mentions', 'result_set_chunks', 'chunk_metadata')
        GROUP BY t.relname, i.relname, ix.indisunique, ix.indisprimary
        ORDER BY t.relname, i.relname
    """
    
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    
    indexes_by_table = {}
    for row in rows:
        table, index_name, columns, is_unique, is_primary = row
        if table not in indexes_by_table:
            indexes_by_table[table] = []
        indexes_by_table[table].append({
            "name": index_name,
            "columns": columns,
            "is_unique": is_unique,
            "is_primary": is_primary,
        })
    
    return indexes_by_table


def check_index_exists(
    existing_indexes: Dict[str, List[Dict[str, Any]]],
    table: str,
    columns: List[str],
) -> Tuple[bool, str]:
    """
    Check if an index exists that covers the specified columns.
    
    An index "covers" columns if it has them as a prefix (leading columns).
    """
    table_indexes = existing_indexes.get(table, [])
    
    for idx in table_indexes:
        idx_cols = idx["columns"]
        # Check if required columns are a prefix of the index
        if len(idx_cols) >= len(columns):
            if idx_cols[:len(columns)] == columns:
                return True, idx["name"]
    
    return False, ""


def verify_indexes(conn, verbose: bool = True) -> Dict[str, Any]:
    """
    Verify all required indexes exist.
    
    Returns:
        dict with:
            - all_present: bool
            - missing: list of missing index specs
            - present: list of present index specs with names
            - recommended_missing: list of missing recommended indexes
    """
    existing = get_existing_indexes(conn)
    
    missing = []
    present = []
    
    for spec in REQUIRED_INDEXES:
        found, name = check_index_exists(existing, spec["table"], spec["columns"])
        if found:
            present.append({**spec, "index_name": name})
            if verbose:
                print(f"  [OK] {spec['table']}({', '.join(spec['columns'])}) -> {name}")
        else:
            missing.append(spec)
            if verbose:
                print(f"  [MISSING] {spec['table']}({', '.join(spec['columns'])}) - {spec['purpose']}")
    
    # Check recommended indexes
    recommended_missing = []
    for spec in RECOMMENDED_INDEXES:
        found, name = check_index_exists(existing, spec["table"], spec["columns"])
        if not found:
            recommended_missing.append(spec)
            if verbose:
                print(f"  [RECOMMENDED] {spec['table']}({', '.join(spec['columns'])}) - {spec['purpose']}")
    
    return {
        "all_present": len(missing) == 0,
        "missing": missing,
        "present": present,
        "recommended_missing": recommended_missing,
    }


def main():
    """Main entry point."""
    import psycopg2
    
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    print("Connecting to database...")
    conn = psycopg2.connect(database_url)
    
    print("\nVerifying required indexes:")
    print("-" * 60)
    
    result = verify_indexes(conn, verbose=True)
    
    print("-" * 60)
    
    if result["all_present"]:
        print(f"\n[OK] All {len(result['present'])} required indexes present")
    else:
        print(f"\n[FAIL] {len(result['missing'])} required indexes missing!")
        print("\nTo create missing indexes, run:")
        for spec in result["missing"]:
            cols = ", ".join(spec["columns"])
            idx_name = f"idx_{spec['table']}_{'_'.join(spec['columns'])}"
            print(f"  CREATE INDEX {idx_name} ON {spec['table']} ({cols});")
    
    if result["recommended_missing"]:
        print(f"\n{len(result['recommended_missing'])} recommended indexes missing (optional)")
    
    conn.close()
    
    return 0 if result["all_present"] else 1


if __name__ == "__main__":
    sys.exit(main())
