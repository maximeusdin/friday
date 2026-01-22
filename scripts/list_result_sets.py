#!/usr/bin/env python3
"""
List all result_sets with their IDs, names, and associated retrieval run info.
"""

import os
import sys

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from retrieval.ops import get_conn


def main():
    conn = get_conn()
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    rs.id,
                    rs.name,
                    rs.retrieval_run_id,
                    array_length(rs.chunk_ids, 1) AS num_chunks,
                    rr.query_text,
                    rr.search_type,
                    rr.query_lang_version,
                    rr.created_at AS run_created_at,
                    rs.created_at AS result_set_created_at
                FROM result_sets rs
                JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id
                ORDER BY rs.created_at DESC
            """)
            
            rows = cur.fetchall()
            
            if not rows:
                print("No result_sets found.")
                return
            
            print("=" * 100)
            print("RESULT SETS")
            print("=" * 100)
            print(f"\n{'ID':<6} {'Name':<40} {'Run ID':<8} {'Chunks':<8} {'Query':<30} {'Type':<10} {'Created':<20}")
            print("-" * 100)
            
            for row in rows:
                rs_id, name, run_id, num_chunks, query_text, search_type, qlv, run_created, rs_created = row
                query_preview = (query_text[:27] + "...") if query_text and len(query_text) > 30 else (query_text or "")
                created_str = rs_created.strftime("%Y-%m-%d %H:%M") if rs_created else ""
                
                print(f"{rs_id:<6} {name[:38]:<40} {run_id:<8} {num_chunks or 0:<8} {query_preview:<30} {search_type or 'N/A':<10} {created_str:<20}")
            
            print("\n" + "=" * 100)
            print(f"Total: {len(rows)} result_set(s)")
            print("\nTo export a result_set:")
            print("  python scripts/export_result_set.py --id <ID> --out results.csv")
            print("  python scripts/export_result_set.py --id <ID> --out results.csv --include-match-type")
            
    finally:
        conn.close()


if __name__ == "__main__":
    main()
