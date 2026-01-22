#!/usr/bin/env python3
"""
list_sessions.py [--limit N]

Lists recent research sessions with counts of retrieval_runs and result_sets.

Usage:
    python scripts/list_sessions.py
    python scripts/list_sessions.py --limit 20
"""

import os
import sys
import argparse

import psycopg2


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def main():
    ap = argparse.ArgumentParser(description="List recent research sessions")
    ap.add_argument("--limit", type=int, default=10, help="Maximum number of sessions to list (default: 10)")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  rs.id,
                  rs.label,
                  rs.created_at,
                  COUNT(DISTINCT rr.id) AS retrieval_runs_count,
                  COUNT(DISTINCT rset.id) AS result_sets_count
                FROM research_sessions rs
                LEFT JOIN retrieval_runs rr ON rr.session_id = rs.id
                LEFT JOIN result_sets rset ON rset.session_id = rs.id
                GROUP BY rs.id, rs.label, rs.created_at
                ORDER BY rs.created_at DESC
                LIMIT %s
                """,
                (args.limit,)
            )
            
            rows = cur.fetchall()
            
            if not rows:
                print("No sessions found.")
                return
            
            print(f"{'ID':<8} {'Label':<40} {'Runs':<8} {'Result Sets':<12} {'Created'}")
            print("=" * 100)
            
            for session_id, label, created_at, runs_count, sets_count in rows:
                # Truncate label if too long
                label_display = label[:37] + "..." if len(label) > 40 else label
                created_str = created_at.strftime("%Y-%m-%d %H:%M")
                print(f"{session_id:<8} {label_display:<40} {runs_count:<8} {sets_count:<12} {created_str}")
                
    finally:
        conn.close()


if __name__ == "__main__":
    main()
