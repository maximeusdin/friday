#!/usr/bin/env python3
"""Backfill chunk_metadata for existing McCarthy chunks."""

import os
import psycopg2

def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return
    
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    # Get mccarthy collection id
    cur.execute("SELECT id FROM collections WHERE slug = 'mccarthy'")
    row = cur.fetchone()
    if not row:
        print("No mccarthy collection found")
        return
    collection_id = row[0]
    
    # Find chunks that have turn metadata but no chunk_metadata
    cur.execute("""
        SELECT ch.id, ch.pipeline_version
        FROM chunks ch
        WHERE ch.turn_id_start IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM chunk_metadata cm WHERE cm.chunk_id = ch.id)
    """)
    orphan_chunks = cur.fetchall()
    print(f"Found {len(orphan_chunks)} chunks without chunk_metadata")
    
    if not orphan_chunks:
        print("Nothing to backfill!")
        return
    
    # For each chunk, find its pages and determine document_id
    backfilled = 0
    for chunk_id, pipeline_version in orphan_chunks:
        # Get page links
        cur.execute("""
            SELECT cp.page_id, p.document_id
            FROM chunk_pages cp
            JOIN pages p ON p.id = cp.page_id
            WHERE cp.chunk_id = %s
            ORDER BY cp.span_order
        """, (chunk_id,))
        page_rows = cur.fetchall()
        
        if not page_rows:
            continue
        
        document_id = page_rows[0][1]
        first_page_id = page_rows[0][0]
        last_page_id = page_rows[-1][0]
        
        # Insert chunk_metadata (check first to avoid duplicates)
        cur.execute("SELECT 1 FROM chunk_metadata WHERE chunk_id = %s", (chunk_id,))
        if not cur.fetchone():
            cur.execute("""
                INSERT INTO chunk_metadata (chunk_id, document_id, collection_slug, pipeline_version,
                                           first_page_id, last_page_id, content_type)
                VALUES (%s, %s, 'mccarthy', %s, %s, %s, 'hearing_transcript')
            """, (chunk_id, document_id, pipeline_version or 'mccarthy_v2_turns', first_page_id, last_page_id))
        
        backfilled += 1
        if backfilled % 500 == 0:
            print(f"  Backfilled {backfilled}/{len(orphan_chunks)}...")
            conn.commit()
    
    conn.commit()
    print(f"\nDone! Backfilled {backfilled} chunk_metadata rows")
    
    # Verify
    cur.execute("""
        SELECT COUNT(*) FROM chunk_metadata WHERE collection_slug = 'mccarthy'
    """)
    print(f"Total mccarthy chunk_metadata rows: {cur.fetchone()[0]}")
    
    conn.close()

if __name__ == "__main__":
    main()
