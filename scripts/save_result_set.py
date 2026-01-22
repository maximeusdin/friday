#!/usr/bin/env python3
"""
save_result_set.py --run-id <retrieval_run_id> --name <name> [--chunk-ids <id1,id2,...>]

Saves a retrieval_run as a named result_set.

If --chunk-ids is not provided, uses retrieval_runs.returned_chunk_ids.
If --chunk-ids is provided, saves that subset (useful for curation).

Usage:
    # Save entire retrieval run
    python scripts/save_result_set.py --run-id 42 --name "Oppenheimer search v1"
    
    # Save curated subset
    python scripts/save_result_set.py --run-id 42 --name "Oppenheimer curated" --chunk-ids "6595,6699,6710"
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
    ap = argparse.ArgumentParser(description="Save a retrieval_run as a named result_set")
    ap.add_argument("--run-id", type=int, required=True, help="retrieval_runs.id to save")
    ap.add_argument("--name", type=str, required=True, help="Name for the result_set")
    ap.add_argument("--chunk-ids", type=str, default=None, 
                    help="Comma-separated chunk IDs (default: use retrieval_runs.returned_chunk_ids)")
    ap.add_argument("--notes", type=str, default=None, help="Optional notes")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Check if retrieval_run exists
            cur.execute(
                "SELECT id, query_text, array_length(returned_chunk_ids, 1) FROM retrieval_runs WHERE id = %s",
                (args.run_id,)
            )
            run_row = cur.fetchone()
            if not run_row:
                raise SystemExit(f"No retrieval_run found with id={args.run_id}")
            
            run_id, query_text, num_results = run_row
            print(f"Found retrieval_run {run_id}: {query_text!r} ({num_results} results)")
            
            # Get chunk_ids
            if args.chunk_ids:
                # Parse comma-separated list
                chunk_ids = [int(x.strip()) for x in args.chunk_ids.split(",")]
                print(f"Using provided chunk IDs: {len(chunk_ids)} chunks")
            else:
                # Use returned_chunk_ids from retrieval_run
                cur.execute(
                    "SELECT returned_chunk_ids FROM retrieval_runs WHERE id = %s",
                    (args.run_id,)
                )
                chunk_ids = cur.fetchone()[0]
                if not chunk_ids:
                    raise SystemExit(f"retrieval_run {run_id} has no returned_chunk_ids")
                print(f"Using retrieval_run chunk IDs: {len(chunk_ids)} chunks")
            
            # Get session_id from retrieval_run (if present) for auto-assignment
            cur.execute(
                "SELECT session_id FROM retrieval_runs WHERE id = %s",
                (args.run_id,)
            )
            session_id = cur.fetchone()[0]
            
            # Insert result_set (auto-assign session_id from retrieval_run)
            cur.execute(
                """
                INSERT INTO result_sets (name, retrieval_run_id, chunk_ids, notes, session_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (args.name, args.run_id, chunk_ids, args.notes, session_id)
            )
            result_set_id, created_at = cur.fetchone()
            
            conn.commit()
            
            print(f"\n✅ Created result_set:")
            print(f"   ID: {result_set_id}")
            print(f"   Name: {args.name}")
            print(f"   Retrieval run: {run_id}")
            print(f"   Chunks: {len(chunk_ids)}")
            print(f"   Created: {created_at}")
            print(f"\nExport with:")
            print(f"   python scripts/export_result_set.py --id {result_set_id} --out {args.name.replace(' ', '_')}.csv")
            
    except psycopg2.IntegrityError as e:
        if "result_sets_name_uniq" in str(e):
            raise SystemExit(f"Error: A result_set with name '{args.name}' already exists. Choose a different name.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
