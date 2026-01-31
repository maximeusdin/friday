#!/usr/bin/env python3
"""Assess HUAC reports ingest - verify annual reports and publications."""

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
    print("HUAC Reports Collection Assessment")
    print("=" * 70)
    
    cur.execute("""
        SELECT c.id, c.slug, c.title, COUNT(d.id) as doc_count
        FROM collections c
        LEFT JOIN documents d ON d.collection_id = c.id
        WHERE c.slug = 'huac_reports'
        GROUP BY c.id, c.slug, c.title
    """)
    row = cur.fetchone()
    if not row:
        print("No huac_reports collection found!")
        conn.close()
        return
    
    collection_id = row[0]
    print(f"Collection: {row[1]} (id={collection_id})")
    print(f"Title: {row[2]}")
    print(f"Documents: {row[3]}")
    
    # Document breakdown
    print("\n" + "=" * 70)
    print("Document Breakdown")
    print("=" * 70)
    
    cur.execute("""
        SELECT d.volume as year,
               d.source_name,
               (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) as pages,
               (SELECT COUNT(*) FROM chunk_metadata cm WHERE cm.document_id = d.id) as chunks
        FROM documents d
        WHERE d.collection_id = %s
        ORDER BY d.volume, d.source_name
    """, (collection_id,))
    
    print(f"{'Year':<8} {'Document':<55} {'Pages':>6} {'Chunks':>7}")
    print("-" * 80)
    
    total_pages = 0
    total_chunks = 0
    for row in cur.fetchall():
        name = row[1][:52] + "..." if len(row[1]) > 55 else row[1]
        print(f"{row[0] or 'N/A':<8} {name:<55} {row[2]:>6} {row[3]:>7}")
        total_pages += row[2]
        total_chunks += row[3]
    
    print("-" * 80)
    print(f"{'TOTAL':<8} {'':<55} {total_pages:>6} {total_chunks:>7}")
    
    # Chunk statistics
    print("\n" + "=" * 70)
    print("Chunk Statistics")
    print("=" * 70)
    
    cur.execute("""
        SELECT COUNT(*),
               AVG(LENGTH(ch.text))::int,
               MIN(LENGTH(ch.text)),
               MAX(LENGTH(ch.text)),
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(ch.text))::int,
               SUM(LENGTH(ch.text))
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'huac_reports'
    """)
    row = cur.fetchone()
    total_chunks = row[0]
    print(f"Total chunks: {total_chunks:,}")
    print(f"Avg chunk length: {row[1]:,} chars")
    print(f"Median chunk length: {row[4]:,} chars")
    print(f"Min/Max: {row[2]:,} / {row[3]:,} chars")
    print(f"Total text: {row[5]:,} chars (~{row[5]//4:,} tokens)")
    
    # Chunk size distribution
    print("\n--- Chunk Size Distribution ---")
    cur.execute("""
        SELECT 
            CASE 
                WHEN LENGTH(ch.text) < 2000 THEN '<2k'
                WHEN LENGTH(ch.text) < 4000 THEN '2k-4k'
                WHEN LENGTH(ch.text) < 6000 THEN '4k-6k'
                ELSE '>6k'
            END as bucket,
            COUNT(*) as cnt
        FROM chunks ch
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'huac_reports'
        GROUP BY bucket
        ORDER BY MIN(LENGTH(ch.text))
    """)
    for row in cur.fetchall():
        pct = 100 * row[1] / total_chunks
        bar = '#' * int(pct / 2)
        print(f"  {row[0]:8} {row[1]:>5} ({pct:5.1f}%) {bar}")
    
    # Content type breakdown
    print("\n--- Content Types ---")
    cur.execute("""
        SELECT cm.content_type, COUNT(*) as cnt
        FROM chunk_metadata cm
        WHERE cm.collection_slug = 'huac_reports'
        GROUP BY cm.content_type
        ORDER BY cnt DESC
        LIMIT 10
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]:,} chunks")
    
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
        WHERE cm.collection_slug = 'huac_reports' AND LENGTH(ch.text) < 200
    """)
    small_chunks = cur.fetchone()[0]
    pct = 100 * small_chunks / total_chunks if total_chunks > 0 else 0
    status = "OK" if pct < 5 else "WARN"
    print(f"[{status}] Chunks < 200 chars: {small_chunks} ({pct:.1f}%)")
    
    # Page coverage
    cur.execute("""
        SELECT COUNT(DISTINCT cp.page_id)
        FROM chunk_pages cp
        JOIN chunks ch ON cp.chunk_id = ch.id
        JOIN chunk_metadata cm ON cm.chunk_id = ch.id
        WHERE cm.collection_slug = 'huac_reports'
    """)
    pages_covered = cur.fetchone()[0]
    
    cur.execute("""
        SELECT COUNT(*) FROM pages p
        JOIN documents d ON p.document_id = d.id
        WHERE d.collection_id = %s AND p.content_role = 'primary'
    """, (collection_id,))
    primary_pages = cur.fetchone()[0]
    
    if primary_pages > 0:
        coverage = 100 * pages_covered / primary_pages
        status = "OK" if coverage > 90 else "WARN"
        print(f"[{status}] Page coverage: {pages_covered:,} / {primary_pages:,} ({coverage:.1f}%)")
    
    # Search quality
    print("\n" + "=" * 70)
    print("Search Quality Test")
    print("=" * 70)
    
    search_terms = [
        ('Communist Party', 'Main subject'),
        ('Soviet Union', 'Foreign power'),
        ('espionage', 'Investigation focus'),
        ('subversive', 'Key term'),
        ('infiltration', 'Key term'),
        ('Congress', 'Legislative body'),
        ('investigation', 'Activity type'),
    ]
    
    for term, desc in search_terms:
        cur.execute("""
            SELECT COUNT(*) FROM chunks ch
            JOIN chunk_metadata cm ON cm.chunk_id = ch.id
            WHERE cm.collection_slug = 'huac_reports'
            AND ch.text ILIKE %s
        """, (f'%{term}%',))
        count = cur.fetchone()[0]
        print(f"  '{term}': {count:,} chunks - {desc}")
    
    # Year distribution
    print("\n--- Coverage by Year ---")
    cur.execute("""
        SELECT d.volume as year, COUNT(cm.chunk_id) as chunks
        FROM documents d
        JOIN chunk_metadata cm ON cm.document_id = d.id
        WHERE d.collection_id = %s
        GROUP BY d.volume
        ORDER BY d.volume
    """, (collection_id,))
    for row in cur.fetchall():
        print(f"  {row[0] or 'unknown'}: {row[1]:,} chunks")
    
    conn.close()
    print("\n" + "=" * 70)
    print("Assessment Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
