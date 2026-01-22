#!/usr/bin/env python3
"""
Debug script to check what's happening with result_set export.
"""

import os
import sys
import psycopg2

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)

def main():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Check result_set 2
            print("=== Checking result_set id=2 ===")
            cur.execute("""
                SELECT id, name, retrieval_run_id, chunk_ids, array_length(chunk_ids, 1) as num_chunks
                FROM result_sets
                WHERE id = 2
            """)
            row = cur.fetchone()
            if row:
                rs_id, name, run_id, chunk_ids, num_chunks = row
                print(f"Result set: id={rs_id}, name={name}, run_id={run_id}, num_chunks={num_chunks}")
                if chunk_ids:
                    print(f"First 5 chunk_ids: {chunk_ids[:5]}")
                else:
                    print("WARNING: chunk_ids is empty or NULL")
            else:
                print("ERROR: result_set id=2 not found")
                return
            
            # Check if retrieval_chunks_v1 view exists
            print("\n=== Checking if retrieval_chunks_v1 view exists ===")
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.views 
                    WHERE table_schema = 'public' 
                    AND table_name = 'retrieval_chunks_v1'
                )
            """)
            view_exists = cur.fetchone()[0]
            print(f"retrieval_chunks_v1 view exists: {view_exists}")
            
            if not view_exists:
                print("\nERROR: retrieval_chunks_v1 view does not exist!")
                print("Checking what views exist...")
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.views 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                views = cur.fetchall()
                print(f"Available views: {[v[0] for v in views]}")
                return
            
            # Check if chunks exist for the chunk_ids
            print("\n=== Checking if chunks exist ===")
            if chunk_ids:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM chunks 
                    WHERE id = ANY(%s::bigint[])
                """, (chunk_ids,))
                chunk_count = cur.fetchone()[0]
                print(f"Found {chunk_count} chunks out of {len(chunk_ids)} chunk_ids")
                
                # Check retrieval_chunks_v1
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM retrieval_chunks_v1 
                    WHERE chunk_id = ANY(%s::bigint[])
                """, (chunk_ids,))
                view_count = cur.fetchone()[0]
                print(f"Found {view_count} chunks in retrieval_chunks_v1 view")
                
                if view_count == 0:
                    print("\nERROR: No chunks found in retrieval_chunks_v1 view!")
                    print("Checking view definition...")
                    cur.execute("""
                        SELECT pg_get_viewdef('retrieval_chunks_v1', true)
                    """)
                    view_def = cur.fetchone()[0]
                    print(f"\nView definition:\n{view_def}")
                    
                    # Check what columns the view has
                    print("\nView columns:")
                    cur.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'retrieval_chunks_v1'
                        ORDER BY ordinal_position
                    """)
                    for col_name, col_type in cur.fetchall():
                        print(f"  {col_name}: {col_type}")
            
            # Check evidence
            print("\n=== Checking evidence ===")
            cur.execute("""
                SELECT COUNT(*) 
                FROM retrieval_run_chunk_evidence 
                WHERE retrieval_run_id = %s
            """, (run_id,))
            evidence_count = cur.fetchone()[0]
            print(f"Evidence rows for run_id {run_id}: {evidence_count}")
            
    finally:
        conn.close()

if __name__ == "__main__":
    main()
