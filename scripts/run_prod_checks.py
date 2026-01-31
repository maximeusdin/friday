#!/usr/bin/env python3
"""
Production-grade smoke test checks.
Run: python scripts/run_prod_checks.py
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


def check_indexes():
    """Test T: Verify performance indexes exist."""
    print("\n=== Test T: Index Verification ===")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Check entity_mentions indexes
            cur.execute("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'entity_mentions'
                ORDER BY indexname
            """)
            rows = cur.fetchall()
            
            print(f"Found {len(rows)} indexes on entity_mentions:")
            for name, defn in rows:
                print(f"  - {name}")
            
            # Check for our performance indexes
            index_names = [r[0] for r in rows]
            expected = [
                'idx_entity_mentions_chunk_entity',
                'idx_entity_mentions_document_entity',
            ]
            
            missing = [e for e in expected if e not in index_names]
            if missing:
                print(f"\n⚠️  Missing expected indexes: {missing}")
                print("   Run: python scripts/run_migration.py migrations/0040_cooccurrence_performance_indexes.sql")
            else:
                print("\n✅ All expected performance indexes exist")
            
            # Check chunk_metadata indexes
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'chunk_metadata'
                ORDER BY indexname
            """)
            cm_rows = cur.fetchall()
            print(f"\nFound {len(cm_rows)} indexes on chunk_metadata")
            
    finally:
        conn.close()


def check_cold_start_session():
    """Test R: Create a fresh session and run entity-only query."""
    print("\n=== Test R: Cold Start Session ===")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Check research_sessions schema
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'research_sessions'
                ORDER BY ordinal_position
            """)
            cols = [r[0] for r in cur.fetchall()]
            print(f"research_sessions columns: {cols}")
            
            # Create new session
            if 'label' in cols:
                cur.execute(
                    "INSERT INTO research_sessions (label) VALUES (%s) RETURNING id",
                    ('cold_start_test',)
                )
            else:
                cur.execute("INSERT INTO research_sessions DEFAULT VALUES RETURNING id")
            
            session_id = cur.fetchone()[0]
            conn.commit()
            print(f"Created new session ID: {session_id}")
            
            # Check session has no prior runs
            cur.execute(
                "SELECT COUNT(*) FROM retrieval_runs WHERE session_id = %s",
                (session_id,)
            )
            run_count = cur.fetchone()[0]
            print(f"Prior runs in session: {run_count}")
            
            if run_count == 0:
                print("✅ Session is truly cold (no prior runs)")
            else:
                print("⚠️  Session has prior runs (not a cold start)")
            
            return session_id
            
    finally:
        conn.close()


def check_entity_mention_counts():
    """Check that entity mentions exist for testing."""
    print("\n=== Entity Mention Counts ===")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.canonical_name, COUNT(*) as cnt
                FROM entity_mentions em
                JOIN entities e ON e.id = em.entity_id
                GROUP BY e.canonical_name
                ORDER BY cnt DESC
                LIMIT 10
            """)
            rows = cur.fetchall()
            print("Top 10 entities by mention count:")
            for name, cnt in rows:
                print(f"  {cnt:>6}  {name}")
            
            # Total mentions
            cur.execute("SELECT COUNT(*) FROM entity_mentions")
            total = cur.fetchone()[0]
            print(f"\nTotal entity mentions: {total:,}")
            
            if total > 0:
                print("✅ Entity mentions exist for testing")
            else:
                print("⚠️  No entity mentions found!")
                
    finally:
        conn.close()


def check_explain_analyze():
    """Test T: Run EXPLAIN ANALYZE on CO_OCCURS_WITH query."""
    print("\n=== Test T: Query Plan Analysis ===")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Get two entity IDs to test with
            cur.execute("""
                SELECT entity_id FROM entity_mentions 
                GROUP BY entity_id 
                HAVING COUNT(*) > 100
                LIMIT 2
            """)
            rows = cur.fetchall()
            if len(rows) < 2:
                print("⚠️  Not enough high-frequency entities to test")
                return
            
            entity_a, entity_b = rows[0][0], rows[1][0]
            print(f"Testing CO_OCCURS_WITH with entity_a={entity_a}, entity_b={entity_b}")
            
            # Run EXPLAIN ANALYZE
            cur.execute(f"""
                EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
                SELECT c.id
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE EXISTS (
                    SELECT 1 FROM entity_mentions em 
                    WHERE em.chunk_id = c.id AND em.entity_id = {entity_a}
                )
                AND EXISTS (
                    SELECT 1 FROM entity_mentions em 
                    WHERE em.chunk_id = c.id AND em.entity_id = {entity_b}
                )
                LIMIT 20
            """)
            plan = cur.fetchall()
            
            plan_text = "\n".join(r[0] for r in plan)
            print("\nQuery plan (chunk-level CO_OCCURS_WITH):")
            print("-" * 60)
            
            # Check for index usage
            uses_index = "Index" in plan_text
            uses_seq_scan = "Seq Scan on entity_mentions" in plan_text
            
            # Print abbreviated plan
            for line in plan[:15]:
                print(line[0])
            if len(plan) > 15:
                print(f"... ({len(plan) - 15} more lines)")
            
            print("-" * 60)
            if uses_index and not uses_seq_scan:
                print("✅ Query uses indexes (good performance)")
            elif uses_seq_scan:
                print("⚠️  Query does sequential scan on entity_mentions (slow!)")
            else:
                print("ℹ️  Check plan manually for optimization opportunities")
                
    finally:
        conn.close()


def check_result_set_size():
    """Test S: Check result set sizes aren't too large."""
    print("\n=== Test S: Result Set Size Check ===")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, array_length(chunk_ids, 1) as chunk_count
                FROM result_sets
                ORDER BY id DESC
                LIMIT 10
            """)
            rows = cur.fetchall()
            
            print("Recent result sets:")
            for rs_id, chunk_count in rows:
                size_indicator = "✅" if (chunk_count or 0) <= 100 else "⚠️ large"
                print(f"  ID={rs_id}: {chunk_count or 0} chunks {size_indicator}")
            
            # Check for any very large result sets
            cur.execute("""
                SELECT COUNT(*) FROM result_sets 
                WHERE array_length(chunk_ids, 1) > 1000
            """)
            large_count = cur.fetchone()[0]
            
            if large_count > 0:
                print(f"\n⚠️  Found {large_count} result sets with >1000 chunks")
            else:
                print("\n✅ No excessively large result sets")
                
    finally:
        conn.close()


def main():
    print("=" * 60)
    print("PRODUCTION-GRADE SMOKE TEST CHECKS")
    print("=" * 60)
    
    check_entity_mention_counts()
    check_indexes()
    session_id = check_cold_start_session()
    check_explain_analyze()
    check_result_set_size()
    
    print("\n" + "=" * 60)
    print(f"Cold start session created: {session_id}")
    print("To complete Test R, run:")
    print(f"  $env:ENTITY_BEST_GUESS='1'; python scripts/plan_query.py --session {session_id} --text 'Find mentions of Silvermaster'")
    print("=" * 60)


if __name__ == "__main__":
    main()
