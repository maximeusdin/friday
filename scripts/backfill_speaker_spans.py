#!/usr/bin/env python3
"""
Backfill speaker_turn_spans and chunk_turns for existing McCarthy chunks.

This script:
1. Applies the migration to add speaker_turn_spans column
2. Computes speaker_turn_spans for each chunk based on its turns
3. Populates chunk_turns junction table

Can be run safely multiple times (idempotent).
"""
import os
import sys
import json
import argparse
import psycopg2
from typing import List, Dict, Tuple

# Database connection (same as ingest scripts)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")


def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def ensure_schema(cur):
    """Ensure required schema changes are applied."""
    # Add speaker_turn_spans column if missing
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'chunks' AND column_name = 'speaker_turn_spans'
    """)
    if not cur.fetchone():
        print("Adding speaker_turn_spans column...")
        cur.execute("ALTER TABLE chunks ADD COLUMN speaker_turn_spans JSONB")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_speaker_turn_spans 
            ON chunks USING GIN(speaker_turn_spans jsonb_path_ops)
        """)
    
    # Add speaker_norm to chunk_turns if missing
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'chunk_turns' AND column_name = 'speaker_norm'
    """)
    if not cur.fetchone():
        print("Adding speaker_norm to chunk_turns...")
        cur.execute("ALTER TABLE chunk_turns ADD COLUMN speaker_norm TEXT")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_turns_speaker_norm 
            ON chunk_turns(speaker_norm)
        """)


def compute_speaker_turn_spans(turns: List[Tuple]) -> List[Dict]:
    """
    Compute contiguous speaker spans from turns.
    
    Input: list of (turn_id, speaker_norm) tuples, sorted by turn_id
    Output: [{"speaker": "X", "turn_id_start": N, "turn_id_end": M}, ...]
    """
    if not turns:
        return []
    
    spans = []
    current_speaker = None
    current_start = None
    current_end = None
    
    for turn_id, speaker_norm in turns:
        if speaker_norm == current_speaker:
            # Extend current span
            current_end = turn_id
        else:
            # Flush previous span
            if current_speaker is not None:
                spans.append({
                    "speaker": current_speaker,
                    "turn_id_start": current_start,
                    "turn_id_end": current_end,
                })
            # Start new span
            current_speaker = speaker_norm
            current_start = turn_id
            current_end = turn_id
    
    # Flush final span
    if current_speaker is not None:
        spans.append({
            "speaker": current_speaker,
            "turn_id_start": current_start,
            "turn_id_end": current_end,
        })
    
    return spans


def backfill_speaker_turn_spans(cur, collection_slug: str = "mccarthy", batch_size: int = 500):
    """Backfill speaker_turn_spans for chunks in the collection."""
    
    # Get chunks that have turn metadata but missing speaker_turn_spans
    cur.execute("""
        SELECT ch.id, ch.turn_id_start, ch.turn_id_end, cm.document_id
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = %s
          AND ch.turn_id_start IS NOT NULL
          AND (ch.speaker_turn_spans IS NULL OR ch.speaker_turn_spans = 'null'::jsonb)
    """, (collection_slug,))
    
    chunks = cur.fetchall()
    print(f"Found {len(chunks)} chunks needing speaker_turn_spans backfill")
    
    if not chunks:
        return 0
    
    updated = 0
    for i, (chunk_id, turn_start, turn_end, document_id) in enumerate(chunks):
        # Get turns for this chunk
        cur.execute("""
            SELECT turn_id, speaker_norm 
            FROM transcript_turns
            WHERE document_id = %s AND turn_id >= %s AND turn_id <= %s
            ORDER BY turn_id
        """, (document_id, turn_start, turn_end))
        
        turns = cur.fetchall()
        if not turns:
            continue
        
        # Compute spans
        spans = compute_speaker_turn_spans(turns)
        spans_json = json.dumps(spans)
        
        # Update chunk
        cur.execute("""
            UPDATE chunks SET speaker_turn_spans = %s::jsonb WHERE id = %s
        """, (spans_json, chunk_id))
        
        updated += 1
        
        if (i + 1) % batch_size == 0:
            print(f"  Updated {i + 1}/{len(chunks)} chunks...")
    
    return updated


def backfill_chunk_turns(cur, collection_slug: str = "mccarthy", batch_size: int = 500):
    """Populate chunk_turns junction table for chunks in the collection."""
    
    # Get chunks with turn metadata
    cur.execute("""
        SELECT ch.id, ch.turn_id_start, ch.turn_id_end, cm.document_id
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = %s
          AND ch.turn_id_start IS NOT NULL
    """, (collection_slug,))
    
    chunks = cur.fetchall()
    print(f"Found {len(chunks)} chunks for chunk_turns backfill")
    
    if not chunks:
        return 0
    
    inserted = 0
    for i, (chunk_id, turn_start, turn_end, document_id) in enumerate(chunks):
        # Get turns for this chunk
        cur.execute("""
            SELECT id, turn_id, speaker_norm 
            FROM transcript_turns
            WHERE document_id = %s AND turn_id >= %s AND turn_id <= %s
            ORDER BY turn_id
        """, (document_id, turn_start, turn_end))
        
        turns = cur.fetchall()
        
        for span_order, (turn_db_id, turn_id, speaker_norm) in enumerate(turns, start=1):
            cur.execute("""
                INSERT INTO chunk_turns (chunk_id, turn_id, span_order, speaker_norm)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id, turn_id) DO UPDATE SET
                    span_order = EXCLUDED.span_order,
                    speaker_norm = EXCLUDED.speaker_norm
            """, (chunk_id, turn_db_id, span_order, speaker_norm))
            inserted += 1
        
        if (i + 1) % batch_size == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks...")
    
    return inserted


def main():
    ap = argparse.ArgumentParser(description="Backfill speaker_turn_spans and chunk_turns")
    ap.add_argument("--collection", default="mccarthy", help="Collection slug to backfill")
    ap.add_argument("--spans-only", action="store_true", help="Only backfill speaker_turn_spans")
    ap.add_argument("--turns-only", action="store_true", help="Only backfill chunk_turns")
    ap.add_argument("--batch-size", type=int, default=500, help="Batch size for commits")
    ap.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    args = ap.parse_args()
    
    print(f"Backfilling speaker data for collection: {args.collection}")
    print()
    
    with connect() as conn, conn.cursor() as cur:
        # Ensure schema
        ensure_schema(cur)
        conn.commit()
        
        # Backfill speaker_turn_spans
        if not args.turns_only:
            print("=== Backfilling speaker_turn_spans ===")
            updated = backfill_speaker_turn_spans(cur, args.collection, args.batch_size)
            print(f"Updated {updated} chunks with speaker_turn_spans")
            if not args.dry_run:
                conn.commit()
            print()
        
        # Backfill chunk_turns
        if not args.spans_only:
            print("=== Backfilling chunk_turns ===")
            inserted = backfill_chunk_turns(cur, args.collection, args.batch_size)
            print(f"Inserted/updated {inserted} chunk_turns records")
            if not args.dry_run:
                conn.commit()
            print()
        
        if args.dry_run:
            print("[DRY RUN] Rolling back changes...")
            conn.rollback()
        else:
            # Verify
            cur.execute("""
                SELECT COUNT(*) FROM chunks ch
                JOIN chunk_metadata cm ON cm.chunk_id = ch.id
                WHERE cm.collection_slug = %s AND ch.speaker_turn_spans IS NOT NULL
            """, (args.collection,))
            spans_count = cur.fetchone()[0]
            
            cur.execute("""
                SELECT COUNT(*) FROM chunk_turns ct
                JOIN chunks ch ON ct.chunk_id = ch.id
                JOIN chunk_metadata cm ON cm.chunk_id = ch.id
                WHERE cm.collection_slug = %s
            """, (args.collection,))
            turns_count = cur.fetchone()[0]
            
            print("=== Verification ===")
            print(f"Chunks with speaker_turn_spans: {spans_count}")
            print(f"Total chunk_turns records: {turns_count}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
