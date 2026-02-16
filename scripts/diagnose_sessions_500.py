#!/usr/bin/env python3
"""
Diagnose GET /api/sessions 500 errors.

Run against production DB (set DATABASE_URL or DB_HOST+DB_PASS):
  python scripts/diagnose_sessions_500.py

Checks:
  1. research_sessions has user_sub column (migration 0057)
  2. research_messages table exists
  3. Sample list_sessions query runs
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.db import get_dsn, ConfigError

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


def main():
    print("=== Diagnose GET /api/sessions 500 ===\n")

    try:
        dsn = get_dsn()
        print(f"DB: {dsn.split('@')[-1] if '@' in dsn else '(masked)'}\n")
    except ConfigError as e:
        print(f"ERROR: {e}")
        print("Set DATABASE_URL or DB_HOST+DB_PASS")
        sys.exit(1)

    conn = psycopg2.connect(dsn)
    try:
        cur = conn.cursor()

        # 1. research_sessions columns
        print("1. research_sessions columns:")
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'research_sessions'
            ORDER BY ordinal_position
            """
        )
        cols = cur.fetchall()
        col_names = [r[0] for r in cols]
        for name, dtype, nullable in cols:
            print(f"   - {name}: {dtype} (nullable={nullable})")

        if "user_sub" not in col_names:
            print("\n   FAIL: user_sub column MISSING. Run migration 0057_user_sub_sessions.sql")
            sys.exit(1)
        print("   OK: user_sub exists")

        # 2. research_messages exists
        print("\n2. research_messages table:")
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'research_messages')"
        )
        if not cur.fetchone()[0]:
            print("   FAIL: research_messages table not found")
            sys.exit(1)
        print("   OK: research_messages exists")

        # 3. Simulate list_sessions query (use a test sub)
        print("\n3. Simulate list_sessions query (user_sub='test-diagnostic'):")
        test_sub = "test-diagnostic"
        cur.execute(
            """
            SELECT 
                s.id,
                s.label,
                s.created_at,
                COUNT(m.id) AS message_count,
                MAX(m.created_at) AS last_activity
            FROM research_sessions s
            LEFT JOIN research_messages m ON m.session_id = s.id
            WHERE s.user_sub = %s
            GROUP BY s.id
            ORDER BY COALESCE(MAX(m.created_at), s.created_at) DESC
            LIMIT 5
            """,
            (test_sub,),
        )
        rows = cur.fetchall()
        print(f"   OK: Query succeeded. Rows: {len(rows)} (empty is fine for test sub)")

        # 4. Sample real user_sub values (if any)
        print("\n4. Sample user_sub values in research_sessions:")
        cur.execute(
            "SELECT user_sub, COUNT(*) FROM research_sessions GROUP BY user_sub ORDER BY COUNT(*) DESC LIMIT 5"
        )
        for sub, cnt in cur.fetchall():
            print(f"   - {sub!r}: {cnt} sessions")

        print("\n=== All checks passed. Schema is OK. ===")
        print("If /api/sessions still returns 500, check:")
        print("  - ECS/CloudWatch logs for the actual exception")
        print("  - Auth: cookie friday_session present and valid")
        print("  - Set FRIDAY_DEBUG=1 in ECS to see error detail in response")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
