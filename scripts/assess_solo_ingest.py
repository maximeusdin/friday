#!/usr/bin/env python3
"""
Assess FBI SOLO Operation files ingest - verify documents, memos, and chunks.

This assessment is tailored for memo-aware chunking where each chunk
represents a complete FBI memo (or part of a large memo).
"""

import os
import sys
import io
import psycopg2

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Database connection
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
    
    # =========================================================================
    # Collection Overview
    # =========================================================================
    print("=" * 70)
    print("FBI SOLO Operation Collection Assessment")
    print("=" * 70)
    
    cur.execute("""
        SELECT c.id, c.slug, c.title, c.description, COUNT(d.id) as doc_count
        FROM collections c
        LEFT JOIN documents d ON d.collection_id = c.id
        WHERE c.slug = 'solo'
        GROUP BY c.id, c.slug, c.title, c.description
    """)
    row = cur.fetchone()
    if not row:
        print("No SOLO collection found!")
        conn.close()
        return
    
    collection_id = row[0]
    print(f"Collection: {row[1]} (id={collection_id})")
    print(f"Title: {row[2]}")
    print(f"Documents: {row[4]}")
    
    # =========================================================================
    # Document Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("Document Statistics")
    print("=" * 70)
    
    cur.execute("""
        SELECT COUNT(*), 
               SUM(page_count), 
               AVG(page_count)::int,
               MIN(page_count),
               MAX(page_count)
        FROM (
            SELECT d.id, 
                   (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_count
            FROM documents d
            WHERE d.collection_id = %s
        ) sub
    """, (collection_id,))
    row = cur.fetchone()
    print(f"Total documents (PDF files): {row[0]}")
    print(f"Total pages: {row[1]:,}")
    print(f"Avg pages/document: {row[2]}")
    print(f"Page range: {row[3]} - {row[4]}")
    
    # Document breakdown by serial range
    print("\n--- Documents by Type (based on filename) ---")
    cur.execute("""
        SELECT file_type, COUNT(*) as cnt, SUM(page_count) as total_pages
        FROM (
            SELECT d.id,
                CASE 
                    WHEN d.source_name LIKE '%%Serial%%' THEN 'Serial Files'
                    WHEN d.source_name LIKE '%%EBF%%' THEN 'EBF Files'
                    WHEN d.source_name LIKE 'SOLO%%' OR d.source_name LIKE 'Solo%%' THEN 'SOLO Files'
                    ELSE 'Other'
                END as file_type,
                (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_count
            FROM documents d
            WHERE d.collection_id = %s
        ) sub
        GROUP BY file_type
        ORDER BY cnt DESC
    """, (collection_id,))
    for row in cur.fetchall():
        print(f"  {row[0]:15}: {row[1]:3} files, {row[2]:,} pages")
    
    # =========================================================================
    # Chunk Statistics (Memo-Aware)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Chunk Statistics (Memo-Aware)")
    print("=" * 70)
    
    cur.execute("""
        SELECT COUNT(*), 
               AVG(LENGTH(ch.text))::int as avg_len,
               MIN(LENGTH(ch.text)) as min_len,
               MAX(LENGTH(ch.text)) as max_len,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(ch.text))::int as median_len,
               SUM(LENGTH(ch.text)) as total_chars
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'solo'
    """)
    row = cur.fetchone()
    total_chunks = row[0]
    print(f"Total chunks: {total_chunks:,}")
    print(f"Avg chunk length: {row[1]:,} chars")
    print(f"Median chunk length: {row[4]:,} chars")
    print(f"Min chunk length: {row[2]:,} chars")
    print(f"Max chunk length: {row[3]:,} chars")
    print(f"Total text: {row[5]:,} chars (~{row[5]//4:,} tokens)")
    
    # Chunk size distribution
    print("\n--- Chunk Size Distribution ---")
    cur.execute("""
        SELECT 
            CASE 
                WHEN LENGTH(ch.text) < 1000 THEN '<1k'
                WHEN LENGTH(ch.text) < 2000 THEN '1k-2k'
                WHEN LENGTH(ch.text) < 3000 THEN '2k-3k'
                WHEN LENGTH(ch.text) < 4000 THEN '3k-4k'
                WHEN LENGTH(ch.text) < 5000 THEN '4k-5k'
                WHEN LENGTH(ch.text) < 6000 THEN '5k-6k'
                WHEN LENGTH(ch.text) < 7000 THEN '6k-7k'
                ELSE '>7k'
            END as size_bucket,
            COUNT(*) as cnt
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'solo'
        GROUP BY size_bucket
        ORDER BY MIN(LENGTH(ch.text))
    """)
    for row in cur.fetchall():
        pct = 100 * row[1] / total_chunks
        bar = '#' * int(pct / 2)
        print(f"  {row[0]:8}: {row[1]:5,} ({pct:5.1f}%) {bar}")
    
    # =========================================================================
    # Content Type Analysis (Memo Types)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Content Types (Memo Classification)")
    print("=" * 70)
    
    cur.execute("""
        SELECT cm.content_type, COUNT(*) as cnt,
               AVG(LENGTH(ch.text))::int as avg_len
        FROM chunk_metadata cm
        JOIN chunks ch ON ch.id = cm.chunk_id
        WHERE cm.collection_slug = 'solo'
        GROUP BY cm.content_type
        ORDER BY cnt DESC
    """)
    print(f"{'Content Type':25} {'Count':>8} {'Avg Len':>10}")
    print("-" * 45)
    for row in cur.fetchall():
        print(f"{row[0] or 'unset':25} {row[1]:>8,} {row[2]:>10,}")
    
    # =========================================================================
    # Sample Chunks by Type
    # =========================================================================
    print("\n" + "=" * 70)
    print("Sample Chunks by Memo Type")
    print("=" * 70)
    
    memo_types = ['fbi_memo', 'fbi_airtel', 'fbi_teletype', 'fbi_urgent']
    for memo_type in memo_types:
        cur.execute("""
            SELECT ch.id, cm.document_id, LENGTH(ch.text),
                   LEFT(ch.text, 400) as preview
            FROM chunks ch
            JOIN chunk_metadata cm ON cm.chunk_id = ch.id
            WHERE cm.collection_slug = 'solo' AND cm.content_type = %s
            ORDER BY ch.id
            LIMIT 1
        """, (memo_type,))
        row = cur.fetchone()
        if row:
            print(f"\n--- {memo_type.upper()} (chunk {row[0]}, doc={row[1]}, {row[2]} chars) ---")
            preview = row[3].replace('\n', ' ')[:300] if row[3] else ""
            print(f"  {preview}...")
    
    # =========================================================================
    # Data Quality Checks
    # =========================================================================
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
    
    if docs_no_chunks > 0:
        cur.execute("""
            SELECT d.id, d.source_name,
                   (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_count
            FROM documents d
            WHERE d.collection_id = %s
            AND NOT EXISTS (SELECT 1 FROM chunk_metadata cm WHERE cm.document_id = d.id)
            LIMIT 5
        """, (collection_id,))
        for row in cur.fetchall():
            print(f"    Doc {row[0]}: {row[1][:50]} ({row[2]} pages)")
    
    # Very small chunks (might indicate detection issues)
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'solo' AND LENGTH(ch.text) < 200
    """)
    small_chunks = cur.fetchone()[0]
    pct_small = 100 * small_chunks / total_chunks if total_chunks > 0 else 0
    status = "OK" if pct_small < 5 else "WARN"
    print(f"[{status}] Chunks < 200 chars: {small_chunks} ({pct_small:.1f}%)")
    
    # Empty chunks
    cur.execute("""
        SELECT COUNT(*) FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'solo' AND (ch.text IS NULL OR ch.text = '')
    """)
    empty_chunks = cur.fetchone()[0]
    status = "OK" if empty_chunks == 0 else "FAIL"
    print(f"[{status}] Empty chunks: {empty_chunks}")
    
    # Page coverage
    cur.execute("""
        SELECT COUNT(DISTINCT cp.page_id)
        FROM chunk_pages cp
        JOIN chunks ch ON cp.chunk_id = ch.id
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'solo'
    """)
    pages_in_chunks = cur.fetchone()[0]
    
    cur.execute("""
        SELECT COUNT(*) FROM pages p
        JOIN documents d ON p.document_id = d.id
        WHERE d.collection_id = %s
    """, (collection_id,))
    total_pages = cur.fetchone()[0]
    
    if total_pages > 0:
        coverage = 100 * pages_in_chunks / total_pages
        status = "OK" if coverage > 90 else "WARN"
        print(f"[{status}] Page coverage: {pages_in_chunks:,} / {total_pages:,} ({coverage:.1f}%)")
    
    # Chunks per document distribution
    cur.execute("""
        SELECT AVG(chunk_cnt)::int, MIN(chunk_cnt), MAX(chunk_cnt)
        FROM (
            SELECT d.id, COUNT(cm.chunk_id) as chunk_cnt
            FROM documents d
            LEFT JOIN chunk_metadata cm ON cm.document_id = d.id
            WHERE d.collection_id = %s
            GROUP BY d.id
        ) sub
    """, (collection_id,))
    row = cur.fetchone()
    print(f"[INFO] Chunks per document: avg={row[0]}, min={row[1]}, max={row[2]}")
    
    # =========================================================================
    # Search Quality Test
    # =========================================================================
    print("\n" + "=" * 70)
    print("Search Quality Test")
    print("=" * 70)
    
    # Key terms for SOLO operation
    search_terms = [
        ('SOLO', 'Operation name'),
        ('Communist Party', 'Target organization'),
        ('Soviet', 'Foreign connection'),
        ('Morris Childs', 'Key informant'),
        ('Jack Childs', 'Key informant'),
        ('Director', 'FBI communication'),
        ('Chicago', 'Primary field office'),
        ('New York', 'Key field office'),
        ('funds', 'Money tracking'),
        ('secret', 'Classification'),
    ]
    
    print(f"{'Term':20} {'Chunks':>8} {'Description'}")
    print("-" * 55)
    for term, desc in search_terms:
        cur.execute("""
            SELECT COUNT(*) FROM chunks ch
            JOIN chunk_metadata cm ON cm.chunk_id = ch.id
            WHERE cm.collection_slug = 'solo' 
            AND ch.text ILIKE %s
        """, (f'%{term}%',))
        count = cur.fetchone()[0]
        print(f"{term:20} {count:>8,}  {desc}")
    
    # =========================================================================
    # Sample Documents Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Sample Documents (High Memo Count)")
    print("=" * 70)
    
    cur.execute("""
        SELECT d.source_name, 
               (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as page_cnt,
               COUNT(cm.chunk_id) as chunk_cnt
        FROM documents d
        JOIN chunk_metadata cm ON cm.document_id = d.id
        WHERE d.collection_id = %s
        GROUP BY d.id, d.source_name
        ORDER BY chunk_cnt DESC
        LIMIT 10
    """, (collection_id,))
    print(f"{'Document':55} {'Pages':>7} {'Chunks':>7}")
    print("-" * 70)
    for row in cur.fetchall():
        name = row[0][:52] + "..." if len(row[0]) > 55 else row[0]
        print(f"{name:55} {row[1]:>7} {row[2]:>7}")
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("Assessment Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
