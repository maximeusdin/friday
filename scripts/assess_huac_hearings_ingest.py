#!/usr/bin/env python3
"""Assess HUAC hearings ingest - verify transcripts, speakers, and chunks."""

import os
import sys
import io
import psycopg2

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
    
    print("=" * 70)
    print("HUAC Hearings Collection Assessment")
    print("=" * 70)
    
    cur.execute("""
        SELECT c.id, c.slug, c.title, COUNT(d.id) as doc_count
        FROM collections c
        LEFT JOIN documents d ON d.collection_id = c.id
        WHERE c.slug = 'huac_hearings'
        GROUP BY c.id, c.slug, c.title
    """)
    row = cur.fetchone()
    if not row:
        print("No huac_hearings collection found!")
        conn.close()
        return
    
    collection_id = row[0]
    print(f"Collection: {row[1]} (id={collection_id})")
    print(f"Title: {row[2]}")
    print(f"Documents: {row[3]}")
    
    # Document stats
    print("\n" + "=" * 70)
    print("Document Statistics")
    print("=" * 70)
    
    cur.execute("""
        SELECT d.source_name, d.volume,
               (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_count,
               (SELECT COUNT(*) FROM chunk_metadata cm WHERE cm.document_id = d.id) as chunk_count,
               d.metadata->>'topic' as topic
        FROM documents d
        WHERE d.collection_id = %s
        ORDER BY d.volume, d.id
    """, (collection_id,))
    
    print(f"{'Document':<50} {'Year':<6} {'Pages':>6} {'Chunks':>7} {'Topic'}")
    print("-" * 90)
    for row in cur.fetchall():
        name = row[0][:47] + "..." if len(row[0]) > 50 else row[0]
        topic = (row[4] or "")[:20]
        print(f"{name:<50} {row[1] or 'N/A':<6} {row[2]:>6} {row[3]:>7} {topic}")
    
    # Chunk statistics
    print("\n" + "=" * 70)
    print("Chunk Statistics")
    print("=" * 70)
    
    cur.execute("""
        SELECT COUNT(*),
               AVG(LENGTH(ch.text))::int,
               MIN(LENGTH(ch.text)),
               MAX(LENGTH(ch.text)),
               AVG(ch.turn_count)::numeric(10,1),
               SUM(LENGTH(ch.text))
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'huac_hearings'
    """)
    row = cur.fetchone()
    total_chunks = row[0]
    print(f"Total chunks: {total_chunks:,}")
    print(f"Avg chunk length: {row[1]:,} chars")
    print(f"Min/Max chunk: {row[2]:,} / {row[3]:,} chars")
    print(f"Avg turns per chunk: {row[4]}")
    print(f"Total text: {row[5]:,} chars (~{row[5]//4:,} tokens)")
    
    # Speaker statistics
    print("\n" + "=" * 70)
    print("Speaker Statistics")
    print("=" * 70)
    
    cur.execute("""
        SELECT unnest(ch.speaker_norms) as speaker, COUNT(*) as appearances
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'huac_hearings'
        GROUP BY speaker
        ORDER BY appearances DESC
        LIMIT 20
    """)
    print(f"{'Speaker':<30} {'Chunk Appearances':>15}")
    print("-" * 50)
    for row in cur.fetchall():
        print(f"{row[0]:<30} {row[1]:>15,}")
    
    # Famous witnesses/speakers
    print("\n--- Notable Speakers Search ---")
    notable = ['HISS', 'CHAMBERS', 'NIXON', 'STRIPLING', 'BENTLEY', 'WARNER']
    for name in notable:
        cur.execute("""
            SELECT COUNT(*) FROM chunks ch
            JOIN chunk_metadata cm ON cm.chunk_id = ch.id
            WHERE cm.collection_slug = 'huac_hearings'
            AND %s = ANY(ch.speaker_norms)
        """, (f'MR {name}',))
        count = cur.fetchone()[0]
        if count == 0:
            # Try without MR prefix
            cur.execute("""
                SELECT COUNT(*) FROM chunks ch
                JOIN chunk_metadata cm ON cm.chunk_id = ch.id
                WHERE cm.collection_slug = 'huac_hearings'
                AND EXISTS (SELECT 1 FROM unnest(ch.speaker_norms) s WHERE s LIKE %s)
            """, (f'%{name}%',))
            count = cur.fetchone()[0]
        print(f"  {name}: {count:,} chunks")
    
    # Quality checks
    print("\n" + "=" * 70)
    print("Data Quality Checks")
    print("=" * 70)
    
    # Documents without chunks
    cur.execute("""
        SELECT COUNT(*) FROM documents d
        WHERE d.collection_id = %s
        AND NOT EXISTS (SELECT 1 FROM chunk_metadata cm WHERE cm.document_id = d.id)
    """, (collection_id,))
    docs_no_chunks = cur.fetchone()[0]
    status = "OK" if docs_no_chunks == 0 else "WARN"
    print(f"[{status}] Documents without chunks: {docs_no_chunks}")
    
    # Small chunks
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'huac_hearings' AND LENGTH(ch.text) < 200
    """)
    small_chunks = cur.fetchone()[0]
    pct = 100 * small_chunks / total_chunks if total_chunks > 0 else 0
    status = "OK" if pct < 5 else "WARN"
    print(f"[{status}] Chunks < 200 chars: {small_chunks} ({pct:.1f}%)")
    
    # Empty chunks
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'huac_hearings' AND (ch.text IS NULL OR ch.text = '')
    """)
    empty = cur.fetchone()[0]
    status = "OK" if empty == 0 else "FAIL"
    print(f"[{status}] Empty chunks: {empty}")
    
    # Search quality
    print("\n" + "=" * 70)
    print("Search Quality Test")
    print("=" * 70)
    
    search_terms = [
        ('Communist Party', 'Target organization'),
        ('espionage', 'Investigation focus'),
        ('Hollywood', '1947 hearings'),
        ('Alger Hiss', 'Famous case'),
        ('Whittaker Chambers', 'Key witness'),
        ('Soviet', 'Foreign power'),
        ('FBI', 'Agency'),
    ]
    
    for term, desc in search_terms:
        cur.execute("""
            SELECT COUNT(*) FROM chunks ch
            JOIN chunk_metadata cm ON cm.chunk_id = ch.id
            WHERE cm.collection_slug = 'huac_hearings'
            AND ch.text ILIKE %s
        """, (f'%{term}%',))
        count = cur.fetchone()[0]
        print(f"  '{term}': {count:,} chunks - {desc}")
    
    conn.close()
    print("\n" + "=" * 70)
    print("Assessment Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
