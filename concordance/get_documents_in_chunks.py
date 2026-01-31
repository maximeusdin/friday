#!/usr/bin/env python3
"""
Simple utility to get list of documents that have chunks.

This is a simpler version for programmatic use.
Returns just document IDs or full document info.

Usage:
    from concordance.get_documents_in_chunks import get_documents_with_chunks
    
    conn = psycopg2.connect(...)
    cur = conn.cursor()
    docs = get_documents_with_chunks(cur, collection_slug="venona")
"""

import os
import psycopg2
from typing import List, Dict, Optional, Tuple
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
    Get list of document IDs that have chunks.
    
    Args:
        cur: Database cursor
        collection_slug: Optional collection slug to filter by
        
    Returns:
        List of document IDs
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
    collection_slug: Optional[str] = None
) -> List[Dict]:
    """
    Get list of documents that have chunks with full information.
    
    Args:
        cur: Database cursor (should be DictCursor)
        collection_slug: Optional collection slug to filter by
        
    Returns:
        List of dicts with document information:
        - document_id
        - document_name
        - volume
        - collection_slug
        - collection_name
        - chunk_count
    """
    query = """
        SELECT 
            d.id AS document_id,
            d.source_name AS document_name,
            d.volume,
            c.slug AS collection_slug,
            c.title AS collection_name,
            COUNT(DISTINCT cm.chunk_id) AS chunk_count
        FROM documents d
        INNER JOIN collections c ON d.collection_id = c.id
        INNER JOIN chunk_metadata cm ON d.id = cm.document_id
        WHERE cm.chunk_id IS NOT NULL
    """
    
    params = []
    if collection_slug:
        query += " AND c.slug = %s"
        params.append(collection_slug)
    
    query += """
        GROUP BY d.id, d.source_name, d.volume, c.slug, c.title
        ORDER BY c.slug, d.source_name, d.volume
    """
    
    cur.execute(query, params)
    if isinstance(cur, DictCursor):
        return [dict(row) for row in cur.fetchall()]
    else:
        # Fallback if not DictCursor
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_document_to_chunk_count_map(
    cur,
    collection_slug: Optional[str] = None
) -> Dict[int, int]:
    """
    Get a mapping of document_id -> chunk_count.
    
    Args:
        cur: Database cursor
        collection_slug: Optional collection slug to filter by
        
    Returns:
        Dictionary mapping document_id to chunk_count
    """
    query = """
        SELECT 
            d.id AS document_id,
            COUNT(DISTINCT cm.chunk_id) AS chunk_count
        FROM documents d
        INNER JOIN collections c ON d.collection_id = c.id
        INNER JOIN chunk_metadata cm ON d.id = cm.document_id
        WHERE cm.chunk_id IS NOT NULL
    """
    
    params = []
    if collection_slug:
        query += " AND c.slug = %s"
        params.append(collection_slug)
    
    query += """
        GROUP BY d.id
    """
    
    cur.execute(query, params)
    return {row[0]: row[1] for row in cur.fetchall()}


if __name__ == "__main__":
    # Example usage
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        print("Documents with chunks (all collections):")
        docs = get_documents_with_chunks(cur)
        print(f"Found {len(docs)} documents")
        
        # Group by collection
        by_collection = {}
        for doc in docs:
            coll = doc['collection_slug']
            if coll not in by_collection:
                by_collection[coll] = []
            by_collection[coll].append(doc)
        
        for coll_slug in sorted(by_collection.keys()):
            coll_docs = by_collection[coll_slug]
            print(f"\n{'-'*60}")
            print(f"{coll_slug.upper()}: {len(coll_docs)} documents")
            print(f"{'-'*60}")
            for doc in coll_docs[:10]:  # Show first 10 per collection
                print(f"  {doc['document_id']}: {doc['document_name']} - {doc['chunk_count']} chunks")
            if len(coll_docs) > 10:
                print(f"  ... and {len(coll_docs) - 10} more")
        
        print("\n" + "="*60)
        print("\nDocuments with chunks (venona only):")
        venona_docs = get_documents_with_chunks(cur, collection_slug="venona")
        print(f"Found {len(venona_docs)} documents")
        for doc in venona_docs[:10]:
            print(f"  {doc['document_id']}: {doc['document_name']} - {doc['chunk_count']} chunks")
        
        print("\n" + "="*60)
        print("\nDocuments with chunks (vassiliev only):")
        vassiliev_docs = get_documents_with_chunks(cur, collection_slug="vassiliev")
        print(f"Found {len(vassiliev_docs)} documents")
        for doc in vassiliev_docs:
            print(f"  {doc['document_id']}: {doc['document_name']} - {doc['chunk_count']} chunks")
        
        print("\n" + "="*60)
        print("\nDocument ID to chunk count map (all collections):")
        doc_map = get_document_to_chunk_count_map(cur)
        print(f"Found {len(doc_map)} documents")
        for doc_id, count in list(doc_map.items())[:20]:
            print(f"  Document {doc_id}: {count} chunks")
        if len(doc_map) > 20:
            print(f"  ... and {len(doc_map) - 20} more")
    
    finally:
        cur.close()
        conn.close()
