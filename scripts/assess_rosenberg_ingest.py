#!/usr/bin/env python3
"""Assess Rosenberg FBI files ingest - verify documents, pages, and chunks."""

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
    
    # Check if rosenberg collection exists and get stats
    print("=" * 60)
    print("Rosenberg FBI Files Collection Stats")
    print("=" * 60)
    cur.execute("""
        SELECT c.id, c.slug, c.title, COUNT(d.id) as doc_count
        FROM collections c
        LEFT JOIN documents d ON d.collection_id = c.id
        WHERE c.slug = 'rosenberg'
        GROUP BY c.id, c.slug, c.title
    """)
    row = cur.fetchone()
    if row:
        print(f"Collection: {row[1]} (id={row[0]})")
        print(f"Title: {row[2]}")
        print(f"Documents: {row[3]}")
        collection_id = row[0]
    else:
        print("No rosenberg collection found!")
        conn.close()
        return
    
    # Document stats
    print("\n" + "=" * 60)
    print("Document Statistics")
    print("=" * 60)
    cur.execute("""
        SELECT COUNT(*), SUM(page_count), AVG(page_count)::int
        FROM (
            SELECT d.id, 
                   (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_count
            FROM documents d
            WHERE d.collection_id = %s
        ) sub
    """, (collection_id,))
    row = cur.fetchone()
    print(f"Total documents: {row[0]}")
    print(f"Total pages: {row[1]}")
    print(f"Avg pages/doc: {row[2]}")
    
    # Sample documents
    print("\n--- Sample Documents (first 10) ---")
    cur.execute("""
        SELECT d.id, d.source_name, d.volume,
               (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_count,
               (SELECT COUNT(*) FROM chunk_metadata cm WHERE cm.document_id = d.id) as chunk_count
        FROM documents d
        WHERE d.collection_id = %s
        ORDER BY d.id
        LIMIT 10
    """, (collection_id,))
    for row in cur.fetchall():
        print(f"  Doc {row[0]}: {row[1][:50]} (vol={row[2]}, {row[3]} pages, {row[4]} chunks)")
    
    # Page content roles
    print("\n" + "=" * 60)
    print("Page Content Analysis")
    print("=" * 60)
    cur.execute("""
        SELECT p.content_role, COUNT(*) as cnt
        FROM pages p
        JOIN documents d ON p.document_id = d.id
        WHERE d.collection_id = %s
        GROUP BY p.content_role
        ORDER BY cnt DESC
    """, (collection_id,))
    print("Content roles:")
    for row in cur.fetchall():
        print(f"  {row[0] or 'unset':15}: {row[1]:,} pages")
    
    # Chunk statistics
    print("\n" + "=" * 60)
    print("Chunk Statistics")
    print("=" * 60)
    cur.execute("""
        SELECT COUNT(*), 
               AVG(LENGTH(ch.text))::int as avg_len,
               MIN(LENGTH(ch.text)) as min_len,
               MAX(LENGTH(ch.text)) as max_len,
               SUM(LENGTH(ch.text)) as total_chars
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'rosenberg'
    """)
    row = cur.fetchone()
    print(f"Total chunks: {row[0]:,}")
    print(f"Avg chunk length: {row[1]:,} chars")
    print(f"Min chunk length: {row[2]:,} chars")
    print(f"Max chunk length: {row[3]:,} chars")
    print(f"Total text: {row[4]:,} chars (~{row[4]//4:,} tokens)")
    
    # Content type breakdown
    cur.execute("""
        SELECT cm.content_type, COUNT(*) as cnt
        FROM chunk_metadata cm
        WHERE cm.collection_slug = 'rosenberg'
        GROUP BY cm.content_type
        ORDER BY cnt DESC
        LIMIT 10
    """)
    print("\n--- Content Types ---")
    for row in cur.fetchall():
        print(f"  {row[0] or 'unset':20}: {row[1]:,} chunks")
    
    # Sample chunks
    print("\n--- Sample Chunks (first 5) ---")
    cur.execute("""
        SELECT ch.id, cm.document_id, cm.content_type,
               LENGTH(ch.text) as text_len,
               LEFT(ch.text, 300) as preview
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'rosenberg'
        ORDER BY ch.id
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"\nChunk {row[0]} (doc={row[1]}, type={row[2]}, len={row[3]})")
        preview = row[4].replace('\n', ' ')[:200] if row[4] else ""
        print(f"  {preview}...")
    
    # Check for potential issues
    print("\n" + "=" * 60)
    print("Data Quality Checks")
    print("=" * 60)
    
    # Documents without chunks
    cur.execute("""
        SELECT COUNT(*) FROM documents d
        WHERE d.collection_id = %s
        AND NOT EXISTS (SELECT 1 FROM chunk_metadata cm WHERE cm.document_id = d.id)
    """, (collection_id,))
    docs_no_chunks = cur.fetchone()[0]
    print(f"Documents without chunks: {docs_no_chunks}")
    
    if docs_no_chunks > 0:
        cur.execute("""
            SELECT d.id, d.source_name,
                   (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_count
            FROM documents d
            WHERE d.collection_id = %s
            AND NOT EXISTS (SELECT 1 FROM chunk_metadata cm WHERE cm.document_id = d.id)
            LIMIT 10
        """, (collection_id,))
        print("  Examples:")
        for row in cur.fetchall():
            print(f"    Doc {row[0]}: {row[1][:40]} ({row[2]} pages)")
    
    # Very small chunks
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'rosenberg' AND LENGTH(ch.text) < 200
    """)
    small_chunks = cur.fetchone()[0]
    print(f"Chunks < 200 chars: {small_chunks}")
    
    # Empty chunks
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'rosenberg' AND (ch.text IS NULL OR ch.text = '')
    """)
    empty_chunks = cur.fetchone()[0]
    print(f"Empty chunks: {empty_chunks}")
    
    # Chunk-page coverage
    cur.execute("""
        SELECT COUNT(DISTINCT cp.page_id)
        FROM chunk_pages cp
        JOIN chunks ch ON cp.chunk_id = ch.id
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'rosenberg'
    """)
    pages_in_chunks = cur.fetchone()[0]
    
    cur.execute("""
        SELECT COUNT(*) FROM pages p
        JOIN documents d ON p.document_id = d.id
        WHERE d.collection_id = %s AND p.content_role = 'primary'
    """, (collection_id,))
    primary_pages = cur.fetchone()[0]
    
    if primary_pages > 0:
        coverage = 100 * pages_in_chunks / primary_pages
        print(f"Page coverage: {pages_in_chunks:,} / {primary_pages:,} primary pages ({coverage:.1f}%)")
    
    # Search test
    print("\n" + "=" * 60)
    print("Sample Text Search")
    print("=" * 60)
    
    # Search for key terms
    search_terms = ['Rosenberg', 'espionage', 'Communist', 'FBI', 'atomic']
    for term in search_terms:
        cur.execute("""
            SELECT COUNT(*) FROM chunks ch
            JOIN chunk_metadata cm ON cm.chunk_id = ch.id
            WHERE cm.collection_slug = 'rosenberg' 
            AND ch.text ILIKE %s
        """, (f'%{term}%',))
        count = cur.fetchone()[0]
        print(f"  Chunks containing '{term}': {count:,}")
    
    conn.close()
    print("\n" + "=" * 60)
    print("Assessment Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
