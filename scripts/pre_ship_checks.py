#!/usr/bin/env python3
"""
Pre-ship operational checks for the retrieval system.
Run: python scripts/pre_ship_checks.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2


def get_conn():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    return psycopg2.connect(dsn)


def refresh_statistics(conn):
    """Refresh table statistics for query planner."""
    print("\n=== 1. Refreshing Table Statistics ===")
    cur = conn.cursor()
    for table in ['entity_mentions', 'chunk_metadata', 'chunks']:
        cur.execute(f'ANALYZE {table}')
        print(f'  ANALYZE {table} done')
    conn.commit()
    cur.close()


def check_query_plan(conn):
    """Check the exact SQL the compiler generates for CO_OCCURS_WITH document-window."""
    print("\n=== 2. CO_OCCURS_WITH Document-Window Query Plan ===")
    print("Testing with entity_a=67279 (Harry Dexter White), entity_b=72144 (Silvermaster)")
    print()
    
    cur = conn.cursor()
    
    # This is the exact SQL pattern from compile_primitives_to_scope for CO_OCCURS_WITH document window
    sql = """
    EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
    SELECT c.id
    FROM chunks c
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE EXISTS (
        SELECT 1 FROM entity_mentions em_a
        JOIN entity_mentions em_b ON em_b.document_id = em_a.document_id AND em_b.entity_id = 72144
        WHERE em_a.entity_id = 67279
        AND em_a.document_id = cm.document_id
    )
    LIMIT 20
    """
    
    cur.execute(sql)
    plan = cur.fetchall()
    plan_text = '\n'.join(r[0] for r in plan)
    
    print('Query Plan:')
    print('-' * 70)
    for line in plan:
        print(line[0])
    print('-' * 70)
    
    # Check what indexes are used
    uses_doc_entity_idx = 'idx_entity_mentions_document_entity' in plan_text
    uses_entity_doc_idx = 'idx_entity_mentions_entity_document' in plan_text
    uses_seq_scan = 'Seq Scan on entity_mentions' in plan_text
    uses_any_index = 'Index' in plan_text
    
    print()
    if uses_doc_entity_idx:
        print('OK: Uses idx_entity_mentions_document_entity')
    elif uses_entity_doc_idx:
        print('OK: Uses idx_entity_mentions_entity_document')
    elif uses_seq_scan:
        print('WARNING: Sequential scan on entity_mentions detected!')
        print('   Consider: CREATE INDEX CONCURRENTLY idx_entity_mentions_entity_document ON entity_mentions (entity_id, document_id);')
    elif uses_any_index:
        print('OK: Uses indexes (check plan for specifics)')
    else:
        print('WARNING: No clear index usage detected')
    
    cur.close()
    return plan_text


def check_timeouts(conn):
    """Check current timeout settings."""
    print("\n=== 3. Current Timeout Settings ===")
    cur = conn.cursor()
    cur.execute("""SELECT name, setting, unit FROM pg_settings WHERE name IN ('statement_timeout', 'lock_timeout')""")
    for name, setting, unit in cur.fetchall():
        unit_str = unit if unit else ""
        status = "OK" if setting != '0' else "WARNING: No limit set"
        print(f'  {name}: {setting} {unit_str} ({status})')
    cur.close()
    
    print()
    print("  Recommended settings:")
    print("    ALTER ROLE neh SET statement_timeout = '60s';")
    print("    ALTER ROLE neh SET lock_timeout = '5s';")


def verify_schema(conn):
    """Verify all required tables and columns exist."""
    print("\n=== 4. Schema Verification ===")
    cur = conn.cursor()
    
    # Check core tables
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public' AND table_name IN (
          'research_plans','retrieval_runs','result_sets','entity_mentions','chunk_metadata'
        )
    """)
    tables = [r[0] for r in cur.fetchall()]
    print(f'  Core tables present: {len(tables)}/5')
    
    expected_tables = ['research_plans', 'retrieval_runs', 'result_sets', 'entity_mentions', 'chunk_metadata']
    missing = set(expected_tables) - set(tables)
    if not missing:
        print('  OK: All required tables exist')
    else:
        print(f'  ERROR: Missing tables: {missing}')
    
    # Check key columns in entity_mentions
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'entity_mentions' AND column_name IN ('chunk_id', 'document_id', 'entity_id')
    """)
    em_cols = [r[0] for r in cur.fetchall()]
    print(f'  entity_mentions columns: {em_cols} (need chunk_id, document_id, entity_id)')
    
    # Check key columns in research_plans
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'research_plans' AND column_name IN ('plan_json', 'status', 'retrieval_run_id', 'result_set_id')
    """)
    rp_cols = [r[0] for r in cur.fetchall()]
    print(f'  research_plans columns: {rp_cols}')
    
    cur.close()


def check_indexes(conn):
    """List all indexes on entity_mentions."""
    print("\n=== 5. Index Inventory ===")
    cur = conn.cursor()
    
    cur.execute("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'entity_mentions'
        ORDER BY indexname
    """)
    print('  entity_mentions indexes:')
    for (name,) in cur.fetchall():
        marker = '*' if name.startswith('idx_entity_mentions_') else ' '
        print(f'    {marker} {name}')
    
    # Check chunk_metadata too
    cur.execute("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'chunk_metadata'
        ORDER BY indexname
    """)
    print('\n  chunk_metadata indexes:')
    for (name,) in cur.fetchall():
        marker = '*' if name.startswith('idx_chunk_metadata_') else ' '
        print(f'    {marker} {name}')
    
    cur.close()


def print_rollback_plan():
    """Print rollback commands for the indexes."""
    print("\n" + "=" * 70)
    print("ROLLBACK PLAN (if needed):")
    print("=" * 70)
    print("""
-- Drop indexes if they cause issues:
DROP INDEX CONCURRENTLY IF EXISTS idx_entity_mentions_chunk_entity;
DROP INDEX CONCURRENTLY IF EXISTS idx_entity_mentions_document_entity;
DROP INDEX CONCURRENTLY IF EXISTS idx_chunk_metadata_chunk_document;
""")


def run_integrity_check():
    """Run the integrity validation script."""
    print("\n=== 6. Data Integrity Check ===")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/admin_validate_integrity.py"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0


def main():
    print("=" * 70)
    print("PRE-SHIP OPERATIONAL CHECKS")
    print("=" * 70)
    
    conn = get_conn()
    
    try:
        refresh_statistics(conn)
        check_query_plan(conn)
        check_timeouts(conn)
        verify_schema(conn)
        check_indexes(conn)
        print_rollback_plan()
        
        # Run integrity check
        run_integrity_check()
        
    finally:
        conn.close()
    
    print("\n" + "=" * 70)
    print("PRE-SHIP CHECKS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
