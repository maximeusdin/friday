#!/usr/bin/env python3
"""
List all documents that have chunks in the database.

This script shows:
- Document ID, name, and collection
- Number of chunks per document
- Number of pages per document
- Optionally filter by collection

Usage:
    python concordance/list_documents_in_chunks.py
    python concordance/list_documents_in_chunks.py --collection venona
    python concordance/list_documents_in_chunks.py --collection venona --min-chunks 10
    python concordance/list_documents_in_chunks.py --summary
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import DictCursor


def get_db_connection():
    """Get database connection using environment variables or defaults."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "neh")
    db_user = os.getenv("DB_USER", "neh")
    db_pass = os.getenv("DB_PASS", "neh")
    
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass
    )


def get_document_ids_with_chunks(
    cur,
    collection_slug: Optional[str] = None
) -> List[int]:
    """
    Simple function to get just the document IDs that have chunks.
    
    Useful for programmatic use.
    
    Returns: List of document IDs
    """
    query = """
        SELECT DISTINCT d.id
        FROM documents d
        INNER JOIN collections c ON d.collection_id = c.id
        INNER JOIN chunk_metadata cm ON d.id = cm.document_id
        WHERE cm.chunk_id IS NOT NULL
    """
    
    params = []
    if collection_slug:
        query += " AND c.slug = %s"
        params.append(collection_slug)
    
    query += " ORDER BY d.id"
    
    cur.execute(query, params)
    return [row[0] for row in cur.fetchall()]


def get_documents_with_chunks(
    cur,
    collection_slug: Optional[str] = None,
    min_chunks: int = 0
) -> List[Dict]:
    """
    Get list of documents that have chunks.
    
    Returns list of dicts with:
    - document_id
    - document_name
    - collection_slug
    - collection_name
    - chunk_count
    - page_count
    """
    query = """
        SELECT 
            d.id AS document_id,
            d.source_name AS document_name,
            d.volume,
            c.slug AS collection_slug,
            c.title AS collection_name,
            COUNT(DISTINCT cm.chunk_id) AS chunk_count,
            COUNT(DISTINCT p.id) AS page_count
        FROM documents d
        INNER JOIN collections c ON d.collection_id = c.id
        INNER JOIN chunk_metadata cm ON d.id = cm.document_id
        LEFT JOIN pages p ON d.id = p.document_id
        WHERE cm.chunk_id IS NOT NULL
    """
    
    params = []
    
    if collection_slug:
        query += " AND c.slug = %s"
        params.append(collection_slug)
    
    query += """
        GROUP BY d.id, d.source_name, d.volume, c.slug, c.title
        HAVING COUNT(DISTINCT cm.chunk_id) >= %s
        ORDER BY c.slug, d.source_name, d.volume
    """
    params.append(min_chunks)
    
    cur.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def get_document_chunk_summary(cur, document_id: int) -> Dict:
    """Get detailed summary for a specific document."""
    cur.execute("""
        SELECT 
            d.id,
            d.source_name,
            d.volume,
            c.slug AS collection_slug,
            c.title AS collection_name,
            COUNT(DISTINCT cm.chunk_id) AS chunk_count,
            COUNT(DISTINCT p.id) AS page_count,
            COUNT(DISTINCT cp.page_id) AS pages_with_chunks,
            COUNT(DISTINCT em.chunk_id) AS chunks_with_mentions
        FROM documents d
        INNER JOIN collections c ON d.collection_id = c.id
        LEFT JOIN chunk_metadata cm ON d.id = cm.document_id
        LEFT JOIN pages p ON d.id = p.document_id
        LEFT JOIN chunk_pages cp ON cm.chunk_id = cp.chunk_id
        LEFT JOIN entity_mentions em ON cm.chunk_id = em.chunk_id
        WHERE d.id = %s
        GROUP BY d.id, d.source_name, d.volume, c.slug, c.title
    """, (document_id,))
    
    row = cur.fetchone()
    return dict(row) if row else None


def print_documents_list(documents: List[Dict], show_summary: bool = False):
    """Print formatted list of documents."""
    if not documents:
        print("No documents found with chunks.")
        return
    
    print(f"\n{'='*100}")
    print(f"DOCUMENTS WITH CHUNKS ({len(documents)} total)")
    print(f"{'='*100}\n")
    
    current_collection = None
    total_chunks = 0
    total_pages = 0
    
    for doc in documents:
        # Print collection header when it changes
        if doc['collection_slug'] != current_collection:
            if current_collection is not None:
                print()  # Blank line between collections
            print(f"Collection: {doc['collection_name']} ({doc['collection_slug']})")
            print(f"{'-'*100}")
            current_collection = doc['collection_slug']
        
        # Print document info
        volume_str = f" ({doc['volume']})" if doc['volume'] else ""
        print(f"  ID: {doc['document_id']:5d} | "
              f"Chunks: {doc['chunk_count']:6d} | "
              f"Pages: {doc['page_count']:5d} | "
              f"{doc['document_name']}{volume_str}")
        
        total_chunks += doc['chunk_count']
        total_pages += doc['page_count']
    
    print(f"\n{'-'*100}")
    print(f"Total: {len(documents)} documents, {total_chunks:,} chunks, {total_pages:,} pages")
    print(f"{'='*100}\n")
    
    if show_summary:
        # Group by collection
        by_collection = {}
        for doc in documents:
            coll = doc['collection_slug']
            if coll not in by_collection:
                by_collection[coll] = {
                    'name': doc['collection_name'],
                    'doc_count': 0,
                    'chunk_count': 0,
                    'page_count': 0
                }
            by_collection[coll]['doc_count'] += 1
            by_collection[coll]['chunk_count'] += doc['chunk_count']
            by_collection[coll]['page_count'] += doc['page_count']
        
        print("\nSUMMARY BY COLLECTION:")
        print(f"{'='*100}")
        for coll_slug, stats in sorted(by_collection.items()):
            print(f"{stats['name']} ({coll_slug}):")
            print(f"  Documents: {stats['doc_count']}")
            print(f"  Chunks: {stats['chunk_count']:,}")
            print(f"  Pages: {stats['page_count']:,}")
            print()


def print_document_detail(cur, document_id: int):
    """Print detailed information about a specific document."""
    summary = get_document_chunk_summary(cur, document_id)
    
    if not summary:
        print(f"Document {document_id} not found or has no chunks.")
        return
    
    print(f"\n{'='*100}")
    print(f"DETAILED SUMMARY FOR DOCUMENT {document_id}")
    print(f"{'='*100}\n")
    print(f"Document Name: {summary['source_name']}")
    if summary['volume']:
        print(f"Volume: {summary['volume']}")
    print(f"Collection: {summary['collection_name']} ({summary['collection_slug']})")
    print(f"\nChunks: {summary['chunk_count']:,}")
    print(f"Pages: {summary['page_count']:,}")
    print(f"Pages with chunks: {summary['pages_with_chunks']:,}")
    print(f"Chunks with entity mentions: {summary['chunks_with_mentions']:,}")
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description="List documents that have chunks in the database"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Filter by collection slug (e.g., 'venona', 'vassiliev')"
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=0,
        help="Minimum number of chunks required (default: 0)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics by collection"
    )
    parser.add_argument(
        "--document-id",
        type=int,
        help="Show detailed information for a specific document ID"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output as CSV"
    )
    
    args = parser.parse_args()
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        if args.document_id:
            # Show detailed info for one document
            print_document_detail(cur, args.document_id)
        else:
            # List all documents
            documents = get_documents_with_chunks(
                cur,
                collection_slug=args.collection,
                min_chunks=args.min_chunks
            )
            
            if args.csv:
                # CSV output
                import csv
                import sys
                writer = csv.DictWriter(
                    sys.stdout,
                    fieldnames=['document_id', 'document_name', 'volume', 
                               'collection_slug', 'collection_name', 
                               'chunk_count', 'page_count']
                )
                writer.writeheader()
                writer.writerows(documents)
            else:
                # Formatted output
                print_documents_list(documents, show_summary=args.summary)
    
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
