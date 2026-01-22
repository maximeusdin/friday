#!/usr/bin/env python3
"""
List recent retrieval_runs that can be saved as result_sets.
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
                    id,
                    query_text,
                    search_type,
                    query_lang_version,
                    retrieval_impl_version,
                    array_length(returned_chunk_ids, 1) AS num_results,
                    created_at
                FROM retrieval_runs
                ORDER BY created_at DESC
                LIMIT 20
            """)
            
            rows = cur.fetchall()
            
            if not rows:
                print("No retrieval_runs found.")
                return
            
            print("=" * 120)
            print("RECENT RETRIEVAL RUNS (can be saved as result_sets)")
            print("=" * 120)
            print(f"\n{'ID':<6} {'Query':<40} {'Type':<10} {'QLV':<15} {'RIV':<25} {'Results':<8} {'Created':<20}")
            print("-" * 120)
            
            for row in rows:
                run_id, query_text, search_type, qlv, riv, num_results, created_at = row
                query_preview = (query_text[:38] + "...") if query_text and len(query_text) > 40 else (query_text or "")
                created_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else ""
                
                print(f"{run_id:<6} {query_preview:<40} {search_type or 'N/A':<10} {qlv or 'N/A':<15} {riv or 'N/A':<25} {num_results or 0:<8} {created_str:<20}")
            
            print("\n" + "=" * 120)
            print(f"Total: {len(rows)} retrieval_run(s) shown")
            print("\nTo save a retrieval_run as a result_set:")
            print("  python scripts/save_result_set.py --run-id <ID> --name \"My Result Set\"")
            print("\nTo export an existing result_set:")
            print("  python scripts/export_result_set.py --id <ID> --out results.csv --include-match-type")
            
    finally:
        conn.close()


if __name__ == "__main__":
    main()
