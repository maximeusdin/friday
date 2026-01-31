#!/usr/bin/env python3
"""
Show the link between entity citations and documents in the database.

This script:
1. Gets citations for an entity from entity_citations table
2. Shows how those citations map to documents
3. Shows which documents have chunks and could contain entity mentions

Usage:
    python concordance/show_entity_document_links.py --entity-name "AKIM"
    python concordance/show_entity_document_links.py --entity-id 138
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import DictCursor

# Import citation parsing functions
from concordance.validate_entity_mentions_from_citations import (
    parse_citation_text,
    normalize_document_name,
    build_citation_to_document_map
)


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


def get_entity_citations(cur, entity_id: int) -> List[Dict]:
    """Get all citations for an entity from entity_citations table."""
    cur.execute("""
        SELECT 
            id,
            entity_id,
            citation_text,
            collection_slug,
            document_label,
            page_list
        FROM entity_citations
        WHERE entity_id = %s
        ORDER BY id
    """, (entity_id,))
    
    return [dict(row) for row in cur.fetchall()]


def get_entity_info(cur, entity_id: Optional[int] = None, entity_name: Optional[str] = None) -> Optional[Dict]:
    """Get entity information by ID or name."""
    if entity_id:
        cur.execute("""
            SELECT id, canonical_name, entity_type
            FROM entities
            WHERE id = %s
        """, (entity_id,))
    elif entity_name:
        cur.execute("""
            SELECT id, canonical_name, entity_type
            FROM entities
            WHERE canonical_name ILIKE %s
            LIMIT 1
        """, (entity_name,))
    else:
        return None
    
    row = cur.fetchone()
    return dict(row) if row else None


def show_citation_to_document_mapping(
    cur,
    citations: List[Dict],
    entity_name: str,
    entity_id: int
):
    """Show how citations map to documents."""
    print(f"\n{'='*100}")
    print(f"CITATION TO DOCUMENT MAPPING FOR: {entity_name}")
    print(f"{'='*100}\n")
    
    # Parse all citations
    all_citation_locations = []
    for citation_row in citations:
        citation_text = citation_row['citation_text']
        if citation_text:
            locations = parse_citation_text(citation_text)
            for loc in locations:
                all_citation_locations.append((loc, citation_row))
    
    print(f"Found {len(all_citation_locations)} citation location(s) from {len(citations)} citation record(s)\n")
    
    # Group by collection
    by_collection = defaultdict(list)
    for loc, citation_row in all_citation_locations:
        if loc.source.startswith("Venona"):
            coll = "venona"
        elif loc.source.startswith("Vassiliev"):
            coll = "vassiliev"
        else:
            coll = "unknown"
        by_collection[coll].append((loc, citation_row))
    
    # Build document maps for each collection
    document_maps = {}
    for coll in by_collection.keys():
        if coll != "unknown":
            document_maps[coll] = build_citation_to_document_map(cur, coll)
    
    # Show mapping for each citation
    for coll in sorted(by_collection.keys()):
        locations = by_collection[coll]
        print(f"\n{'-'*100}")
        print(f"COLLECTION: {coll.upper()}")
        print(f"{'-'*100}\n")
        
        for loc, citation_row in locations:
            print(f"Citation: {loc.citation_text[:120]}{'...' if len(loc.citation_text) > 120 else ''}")
            print(f"  Source: {loc.source}")
            if loc.year_range:
                print(f"  Year: {loc.year_range}")
            print(f"  Pages: {len(loc.pages)} page reference(s) - {loc.pages[:3]}{'...' if len(loc.pages) > 3 else ''}")
            
            # Try to find matching documents
            if coll in document_maps:
                citation_normalized = normalize_document_name(loc.source)
                if loc.year_range:
                    citation_normalized = normalize_document_name(f"{loc.source} {loc.year_range}")
                
                # Check if we can find documents
                matching_docs = []
                if citation_normalized in document_maps[coll]:
                    matching_docs = document_maps[coll][citation_normalized]
                else:
                    # Try without year
                    if loc.year_range:
                        citation_no_year = normalize_document_name(loc.source)
                        if citation_no_year in document_maps[coll]:
                            # Filter by year
                            year_normalized = loc.year_range.replace('–', '-')
                            if len(year_normalized.split('-')) == 2:
                                start, end = year_normalized.split('-')
                                if len(end) == 2:
                                    year_normalized = f"{start}-{start[:2]}{end}"
                            
                            for doc_id, doc_name in document_maps[coll][citation_no_year]:
                                doc_normalized = normalize_document_name(doc_name)
                                if year_normalized in doc_normalized or loc.year_range.replace('–', '-') in doc_normalized:
                                    matching_docs.append((doc_id, doc_name))
                
                if matching_docs:
                    print(f"  -> Matched {len(matching_docs)} document(s):")
                    for doc_id, doc_name in matching_docs[:5]:
                        # Check if document has chunks and entity mentions
                        cur.execute("""
                            SELECT 
                                COUNT(DISTINCT cm.chunk_id) as chunk_count,
                                COUNT(DISTINCT em.chunk_id) as chunks_with_entity
                            FROM chunk_metadata cm
                            LEFT JOIN entity_mentions em ON cm.chunk_id = em.chunk_id 
                                AND em.entity_id = %s
                            WHERE cm.document_id = %s
                        """, (entity_id, doc_id))
                        chunk_count, chunks_with_entity = cur.fetchone()
                        status = "✓ HAS MENTIONS" if chunks_with_entity > 0 else ""
                        print(f"     - {doc_name} (ID: {doc_id}) - {chunk_count} chunks, {chunks_with_entity} with entity {status}")
                    if len(matching_docs) > 5:
                        print(f"     ... and {len(matching_docs) - 5} more")
                else:
                    print(f"  -> No matching documents found")
                    print(f"     Normalized citation: {citation_normalized}")
                    # Show similar documents
                    cur.execute("""
                        SELECT d.id, d.source_name, COUNT(DISTINCT cm.chunk_id) as chunk_count
                        FROM documents d
                        INNER JOIN collections c ON d.collection_id = c.id
                        LEFT JOIN chunk_metadata cm ON d.id = cm.document_id
                        WHERE c.slug = %s
                        GROUP BY d.id, d.source_name
                        ORDER BY d.source_name
                        LIMIT 5
                    """, (coll,))
                    similar_docs = cur.fetchall()
                    if similar_docs:
                        print(f"     Available documents in {coll} collection:")
                        for doc_id, doc_name, chunk_count in similar_docs:
                            print(f"       - {doc_name} (ID: {doc_id}) - {chunk_count} chunks")
            else:
                print(f"  -> Unknown collection, cannot map")
            
            print()


def show_document_chunk_status(cur, entity_id: int):
    """Show which documents have chunks that might contain this entity."""
    print(f"\n{'='*100}")
    print(f"DOCUMENTS WITH CHUNKS THAT MIGHT CONTAIN ENTITY")
    print(f"{'='*100}\n")
    
    # Get all documents with chunks
    cur.execute("""
        SELECT DISTINCT
            d.id,
            d.source_name,
            c.slug as collection_slug,
            c.title as collection_name,
            COUNT(DISTINCT cm.chunk_id) as chunk_count,
            COUNT(DISTINCT em.chunk_id) as chunks_with_entity
        FROM documents d
        INNER JOIN collections c ON d.collection_id = c.id
        INNER JOIN chunk_metadata cm ON d.id = cm.document_id
        LEFT JOIN entity_mentions em ON cm.chunk_id = em.chunk_id AND em.entity_id = %s
        GROUP BY d.id, d.source_name, c.slug, c.title
        ORDER BY c.slug, d.source_name
    """, (entity_id,))
    
    docs = cur.fetchall()
    
    by_collection = defaultdict(list)
    for doc in docs:
        by_collection[doc[2]].append(doc)
    
    for coll_slug in sorted(by_collection.keys()):
        coll_docs = by_collection[coll_slug]
        print(f"\n{'-'*100}")
        print(f"{coll_docs[0][3]} ({coll_slug.upper()}): {len(coll_docs)} documents")
        print(f"{'-'*100}")
        
        for doc_id, doc_name, coll_slug, coll_name, chunk_count, chunks_with_entity in coll_docs[:20]:
            status = "✓ HAS MENTIONS" if chunks_with_entity > 0 else "  no mentions"
            print(f"  {doc_id:5d} | {chunk_count:6d} chunks | {chunks_with_entity:4d} with entity | {status} | {doc_name}")
        
        if len(coll_docs) > 20:
            print(f"  ... and {len(coll_docs) - 20} more documents")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show link between entity citations and documents"
    )
    parser.add_argument(
        "--entity-id",
        type=int,
        help="Entity ID"
    )
    parser.add_argument(
        "--entity-name",
        type=str,
        help="Entity name (canonical name)"
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Also show documents with chunks that might contain entity"
    )
    
    args = parser.parse_args()
    
    if not args.entity_id and not args.entity_name:
        parser.error("Must provide either --entity-id or --entity-name")
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # Get entity info
        entity = get_entity_info(cur, entity_id=args.entity_id, entity_name=args.entity_name)
        
        if not entity:
            print(f"Entity not found")
            return
        
        entity_id = entity['id']
        entity_name = entity['canonical_name']
        
        print(f"\n{'='*100}")
        print(f"ENTITY: {entity_name} (ID: {entity_id}, Type: {entity['entity_type']})")
        print(f"{'='*100}")
        
        # Get citations
        citations = get_entity_citations(cur, entity_id)
        
        if not citations:
            print(f"\nNo citations found for this entity.")
            return
        
        print(f"\nFound {len(citations)} citation record(s) in entity_citations table")
        
        # Show citation to document mapping
        show_citation_to_document_mapping(cur, citations, entity_name, entity_id)
        
        # Show documents with chunks
        if args.show_chunks:
            show_document_chunk_status(cur, entity_id)
    
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
