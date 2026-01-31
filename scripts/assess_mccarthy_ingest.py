#!/usr/bin/env python3
"""Assess McCarthy ingest - verify speaker turns and chunk metadata."""

import os
import sys
import psycopg2

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


def main():
    conn = connect()
    cur = conn.cursor()
    
    # Check if mccarthy collection exists and get stats
    print("=" * 60)
    print("McCarthy Collection Stats")
    print("=" * 60)
    cur.execute("""
        SELECT c.id, c.slug, COUNT(d.id) as doc_count
        FROM collections c
        LEFT JOIN documents d ON d.collection_id = c.id
        WHERE c.slug = 'mccarthy'
        GROUP BY c.id, c.slug
    """)
    row = cur.fetchone()
    if row:
        print(f"Collection: {row[1]} (id={row[0]}), Documents: {row[2]}")
        collection_id = row[0]
    else:
        print("No mccarthy collection found!")
        conn.close()
        return
    
    # Check documents
    print("\n--- Documents ---")
    cur.execute("""
        SELECT d.id, d.source_name, 
               (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_count
        FROM documents d
        WHERE d.collection_id = %s
        ORDER BY d.id
    """, (collection_id,))
    for row in cur.fetchall():
        print(f"  Doc {row[0]}: {row[1]} ({row[2]} pages)")
    
    # Check transcript_turns
    print("\n" + "=" * 60)
    print("Transcript Turns")
    print("=" * 60)
    cur.execute("""
        SELECT COUNT(*) FROM transcript_turns tt
        JOIN documents d ON tt.document_id = d.id
        WHERE d.collection_id = %s
    """, (collection_id,))
    turn_count = cur.fetchone()[0]
    print(f"Total turns: {turn_count}")
    
    if turn_count == 0:
        print("\nWARNING: No transcript turns found!")
        print("The ingest may not have run, or speaker detection failed.")
    else:
        # Sample turns
        print("\n--- Sample Turns (first 10) ---")
        cur.execute("""
            SELECT tt.turn_id, tt.speaker_raw, tt.speaker_norm, tt.speaker_role, 
                   LEFT(tt.turn_text, 80) as text_preview, tt.page_start, tt.page_end,
                   tt.is_stage_direction
            FROM transcript_turns tt
            JOIN documents d ON tt.document_id = d.id
            WHERE d.collection_id = %s
            ORDER BY tt.document_id, tt.turn_id
            LIMIT 10
        """, (collection_id,))
        for row in cur.fetchall():
            stage = " [STAGE]" if row[7] else ""
            print(f"  Turn {row[0]}: '{row[1]}' -> '{row[2]}' (role={row[3]}){stage}")
            print(f"    Pages: {row[5]}-{row[6]}")
            text_preview = row[4].replace('\n', ' ')[:80] if row[4] else ""
            print(f"    Text: {text_preview}...")
        
        # Unique speakers
        print("\n--- Unique Speakers (top 20) ---")
        cur.execute("""
            SELECT tt.speaker_norm, tt.speaker_role, COUNT(*) as turn_count,
                   SUM(LENGTH(tt.turn_text)) as total_chars
            FROM transcript_turns tt
            JOIN documents d ON tt.document_id = d.id
            WHERE d.collection_id = %s AND tt.is_stage_direction = false
            GROUP BY tt.speaker_norm, tt.speaker_role
            ORDER BY turn_count DESC
            LIMIT 20
        """, (collection_id,))
        for row in cur.fetchall():
            print(f"  {row[0]:30} (role={row[1] or 'unknown':12}): {row[2]:5} turns, {row[3]:,} chars")
        
        # Stage directions
        cur.execute("""
            SELECT COUNT(*) FROM transcript_turns tt
            JOIN documents d ON tt.document_id = d.id
            WHERE d.collection_id = %s AND tt.is_stage_direction = true
        """, (collection_id,))
        stage_count = cur.fetchone()[0]
        print(f"\nStage directions: {stage_count}")
    
    # Check chunks with speaker metadata
    print("\n" + "=" * 60)
    print("Chunks with Speaker Metadata")
    print("=" * 60)
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'mccarthy'
    """, )
    chunk_count = cur.fetchone()[0]
    print(f"Total chunks: {chunk_count}")
    
    # Check speaker metadata on chunks
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'mccarthy' AND ch.turn_id_start IS NOT NULL
    """)
    chunks_with_turns = cur.fetchone()[0]
    print(f"Chunks with turn_id_start: {chunks_with_turns}")
    
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'mccarthy' 
          AND ch.speaker_norms IS NOT NULL 
          AND array_length(ch.speaker_norms, 1) > 0
    """)
    chunks_with_speakers = cur.fetchone()[0]
    print(f"Chunks with speaker_norms: {chunks_with_speakers}")
    
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'mccarthy' 
          AND ch.embed_text IS NOT NULL 
          AND ch.embed_text != ''
    """)
    chunks_with_embed = cur.fetchone()[0]
    print(f"Chunks with embed_text: {chunks_with_embed}")
    
    if chunk_count > 0:
        # Sample chunks
        print("\n--- Sample Chunks (first 5) ---")
        cur.execute("""
            SELECT ch.id, cm.document_id,
                   ch.turn_id_start, ch.turn_id_end, ch.turn_count,
                   ch.speaker_norms, ch.primary_speaker_norm,
                   LENGTH(ch.text) as text_len,
                   LENGTH(ch.embed_text) as embed_len,
                   cm.first_page_id, cm.last_page_id
            FROM chunks ch
            JOIN chunk_metadata cm ON cm.chunk_id = ch.id
            WHERE cm.collection_slug = 'mccarthy'
            ORDER BY cm.document_id, ch.id
            LIMIT 5
        """)
        for row in cur.fetchall():
            print(f"\n  Chunk {row[0]} (doc={row[1]})")
            print(f"    Pages: {row[9]}-{row[10]}")
            print(f"    Turns: {row[2]}-{row[3]} (count={row[4]})")
            print(f"    Speakers: {row[5]}")
            print(f"    Primary: {row[6]}")
            print(f"    Text length: {row[7]}, Embed length: {row[8]}")
        
        # Sample embed_text
        print("\n--- Sample embed_text (first chunk, first 800 chars) ---")
        cur.execute("""
            SELECT LEFT(ch.embed_text, 800)
            FROM chunks ch
            JOIN chunk_metadata cm ON cm.chunk_id = ch.id
            WHERE cm.collection_slug = 'mccarthy' AND ch.embed_text IS NOT NULL
            ORDER BY cm.document_id, ch.id
            LIMIT 1
        """)
        row = cur.fetchone()
        if row and row[0]:
            print(row[0])
        else:
            print("No embed_text found")
    
    # Check chunk_turns junction table
    print("\n" + "=" * 60)
    print("Chunk-Turn Links (chunk_turns table)")
    print("=" * 60)
    cur.execute("""
        SELECT COUNT(*) FROM chunk_turns ct
        JOIN chunks ch ON ct.chunk_id = ch.id
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'mccarthy'
    """)
    link_count = cur.fetchone()[0]
    print(f"Total chunk-turn links: {link_count}")
    
    # Speaker map
    print("\n" + "=" * 60)
    print("Speaker Map")
    print("=" * 60)
    cur.execute("SELECT COUNT(*) FROM speaker_map")
    map_count = cur.fetchone()[0]
    print(f"Speaker mappings: {map_count}")
    
    if map_count > 0:
        cur.execute("SELECT speaker_norm, canonical_name, entity_id, role FROM speaker_map LIMIT 10")
        for row in cur.fetchall():
            print(f"  {row[0]} -> {row[1]} (entity={row[2]}, role={row[3]})")
    
    conn.close()
    print("\n" + "=" * 60)
    print("Assessment Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
