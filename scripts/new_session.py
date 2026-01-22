#!/usr/bin/env python3
"""
new_session.py --label "..."

Creates a new research session and prints the session ID.

Usage:
    python scripts/new_session.py --label "Oppenheimer investigation"
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
    ap = argparse.ArgumentParser(description="Create a new research session")
    ap.add_argument("--label", type=str, required=True, help="Session label (must be unique)")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Insert session (will fail if label already exists due to unique constraint)
            try:
                cur.execute(
                    """
                    INSERT INTO research_sessions (label)
                    VALUES (%s)
                    RETURNING id, created_at
                    """,
                    (args.label.strip(),)
                )
                session_id, created_at = cur.fetchone()
                conn.commit()
                
                print(session_id)
                
            except psycopg2.IntegrityError as e:
                if "research_sessions_label_uniq" in str(e):
                    # Label already exists - fetch existing session
                    cur.execute(
                        """
                        SELECT id, created_at
                        FROM research_sessions
                        WHERE label = %s
                        """,
                        (args.label.strip(),)
                    )
                    session_id, created_at = cur.fetchone()
                    print(f"Session '{args.label}' already exists (ID: {session_id})", file=sys.stderr)
                    print(session_id)
                else:
                    raise
                    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
