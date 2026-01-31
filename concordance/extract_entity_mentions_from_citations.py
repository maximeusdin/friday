#!/usr/bin/env python3
"""
Extract entity mentions from chunks using citation-based targeting.

This improved pipeline:
1. Reads entity citations from entity_citations table
2. Parses citations to extract document names and page numbers
3. Maps citations to documents using normalization
4. Finds chunks that correspond to those document+page combinations
5. Extracts entity mentions from those targeted chunks
6. Stores mentions with proper provenance (document_id, page_id, chunk_id)

This helps resolve ambiguities by targeting specific document spans where entities
are known to appear, rather than scanning all chunks.

Usage:
    # Extract mentions for a specific entity
    python concordance/extract_entity_mentions_from_citations.py --entity-name "Albert"
    
    # Extract mentions for all entities with citations
    python concordance/extract_entity_mentions_from_citations.py --all-entities
    
    # Limit number of entities for debugging
    python concordance/extract_entity_mentions_from_citations.py --all-entities --limit 10
    
    # Dry run to see what would be extracted
    python concordance/extract_entity_mentions_from_citations.py --entity-name "Albert" --dry-run
    
    # Extract for specific collection
    python concordance/extract_entity_mentions_from_citations.py --collection venona --all-entities --limit 5
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import DictCursor, execute_values

# Import citation parsing and document mapping functions
from concordance.validate_entity_mentions_from_citations import (
    parse_citation_text,
    normalize_document_name,
    build_citation_to_document_map,
    find_documents_for_citation,
    find_pages_for_citation,
    CitationLocation
)

# Import entity mention extraction functions
from scripts.extract_entity_mentions import (
    load_all_aliases,
    find_exact_matches,
    extract_mentions_batch,
    AliasInfo
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


def find_chunks_for_pages(
    cur,
    document_id: int,
    page_ids: List[int]
) -> List[Tuple[int, str, int]]:
    """
    Find chunks that span the specified pages.
    
    Returns list of (chunk_id, chunk_text, document_id) tuples.
    """
    if not page_ids:
        return []
    
    # Find chunks that have any of these pages via chunk_pages
    cur.execute("""
        SELECT DISTINCT
            c.id AS chunk_id,
            c.text,
            cm.document_id
        FROM chunks c
        INNER JOIN chunk_pages cp ON c.id = cp.chunk_id
        INNER JOIN chunk_metadata cm ON c.id = cm.chunk_id
        WHERE cp.page_id = ANY(%s)
            AND cm.document_id = %s
        ORDER BY c.id
    """, (page_ids, document_id))
    
    return cur.fetchall()


def extract_mentions_for_entity_citations(
    conn,
    entity_id: int,
    entity_name: str,
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    *,
    dry_run: bool = False,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Extract entity mentions from chunks targeted by entity citations.
    
    Returns statistics dict with counts.
    """
    cur = conn.cursor(cursor_factory=DictCursor)
    
    stats = {
        'citations_processed': 0,
        'citations_with_documents': 0,
        'citations_with_pages': 0,
        'citations_with_chunks': 0,
        'chunks_processed': 0,
        'mentions_found': 0,
        'citations_no_document': 0,
        'citations_no_pages': 0,
        'citations_no_chunks': 0,
    }
    
    try:
        # Get all citations for this entity
        citations = get_entity_citations(cur, entity_id)
        
        if not citations:
            if verbose:
                print(f"  No citations found for {entity_name}")
            return stats
        
        if verbose:
            print(f"  Found {len(citations)} citation record(s) for {entity_name}")
        
        # Process each citation
        all_target_chunks = []  # (chunk_id, chunk_text, document_id)
        citation_chunk_map = {}  # chunk_id -> citation info for provenance
        
        for citation_row in citations:
            stats['citations_processed'] += 1
            citation_text = citation_row['citation_text']
            
            if not citation_text:
                continue
            
            # Parse citation text to get locations
            citation_locations = parse_citation_text(citation_text)
            
            for loc in citation_locations:
                # Find documents matching this citation
                matching_docs = find_documents_for_citation(cur, loc)
                
                if not matching_docs:
                    stats['citations_no_document'] += 1
                    if verbose:
                        print(f"    No documents found for citation: {loc.citation_text[:80]}...")
                    continue
                
                stats['citations_with_documents'] += 1
                
                # For each matching document, find pages and chunks
                for doc_id, doc_name in matching_docs:
                    # Find pages matching citation page numbers
                    page_ids = find_pages_for_citation(cur, doc_id, loc)
                    
                    if not page_ids:
                        stats['citations_no_pages'] += 1
                        if verbose:
                            print(f"    No pages found for {doc_name} (pages: {loc.pages[:3]}...)")
                        continue
                    
                    stats['citations_with_pages'] += 1
                    
                    # Find chunks that span these pages
                    chunks = find_chunks_for_pages(cur, doc_id, page_ids)
                    
                    if not chunks:
                        stats['citations_no_chunks'] += 1
                        if verbose:
                            print(f"    No chunks found for {doc_name} (pages: {len(page_ids)} pages)")
                        continue
                    
                    stats['citations_with_chunks'] += 1
                    
                    # Add chunks to target list
                    for chunk_id, chunk_text, chunk_doc_id in chunks:
                        if chunk_id not in citation_chunk_map:
                            all_target_chunks.append((chunk_id, chunk_text, chunk_doc_id))
                            citation_chunk_map[chunk_id] = {
                                'citation': loc,
                                'document_id': doc_id,
                                'document_name': doc_name,
                                'page_ids': page_ids
                            }
        
        if not all_target_chunks:
            if verbose:
                print(f"  No chunks found for any citations")
            return stats
        
        if verbose:
            print(f"  Found {len(all_target_chunks)} unique chunk(s) to process")
        
        # Extract mentions from targeted chunks
        # Use the existing extraction logic from extract_entity_mentions.py
        rejection_stats = {}
        collision_queue = []
        matched_alias_norms = set()
        unmatched_ngrams = {}
        preferred_entity_id_map = {}
        
        chunks_processed, mentions_found, sample_mentions = extract_mentions_batch(
            conn,
            all_target_chunks,
            aliases_by_norm,
            alias_norm_set,
            dry_run=dry_run,
            show_samples=False,
            max_samples=0,
            rejection_stats=rejection_stats,
            collision_queue=collision_queue,
            matched_alias_norms=matched_alias_norms,
            unmatched_ngrams=unmatched_ngrams,
            preferred_entity_id_map=preferred_entity_id_map,
            scope=None,
        )
        
        stats['chunks_processed'] = chunks_processed
        stats['mentions_found'] = mentions_found
        
        if verbose and mentions_found > 0:
            print(f"  Extracted {mentions_found} mention(s) from {chunks_processed} chunk(s)")
        
    finally:
        cur.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract entity mentions from chunks using citation-based targeting"
    )
    parser.add_argument(
        "--entity-id",
        type=int,
        help="Entity ID to extract mentions for"
    )
    parser.add_argument(
        "--entity-name",
        type=str,
        help="Entity name (canonical name) to extract mentions for"
    )
    parser.add_argument(
        "--all-entities",
        action="store_true",
        help="Extract mentions for all entities that have citations"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Filter by collection slug (only process citations from this collection)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be extracted without inserting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Process entities in batches (default: 100)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of entities to process (useful for debugging)"
    )
    
    args = parser.parse_args()
    
    if not args.entity_id and not args.entity_name and not args.all_entities:
        parser.error("Must provide either --entity-id, --entity-name, or --all-entities")
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # Load all aliases once (shared across all entities)
        print("Loading entity aliases...", file=sys.stderr)
        aliases_by_norm, alias_norm_set = load_all_aliases(conn, collection_slug=args.collection)
        print(f"  Loaded {len(aliases_by_norm)} unique normalized aliases", file=sys.stderr)
        
        # Get list of entities to process
        entities_to_process = []
        
        if args.entity_id:
            cur.execute("""
                SELECT id, canonical_name, entity_type
                FROM entities
                WHERE id = %s
            """, (args.entity_id,))
            row = cur.fetchone()
            if row:
                entities_to_process.append(dict(row))
        
        elif args.entity_name:
            cur.execute("""
                SELECT id, canonical_name, entity_type
                FROM entities
                WHERE canonical_name ILIKE %s
                LIMIT 1
            """, (args.entity_name,))
            row = cur.fetchone()
            if row:
                entities_to_process.append(dict(row))
        
        elif args.all_entities:
            # Get all entities that have citations
            query = """
                SELECT DISTINCT e.id, e.canonical_name, e.entity_type
                FROM entities e
                INNER JOIN entity_citations ec ON e.id = ec.entity_id
            """
            params = []
            
            if args.collection:
                query += " WHERE ec.collection_slug = %s"
                params.append(args.collection)
            
            query += " ORDER BY e.canonical_name"
            
            if args.limit:
                query += " LIMIT %s"
                params.append(args.limit)
            
            cur.execute(query, params)
            entities_to_process = [dict(row) for row in cur.fetchall()]
        
        if not entities_to_process:
            print("No entities found to process", file=sys.stderr)
            return
        
        total_entities = len(entities_to_process)
        if args.limit and args.all_entities:
            print(f"\nProcessing {total_entities} entity/entities (limited to {args.limit})...", file=sys.stderr)
        else:
            print(f"\nProcessing {total_entities} entity/entities...", file=sys.stderr)
        if args.dry_run:
            print("  [DRY RUN] No changes will be made\n", file=sys.stderr)
        
        # Process entities
        total_stats = defaultdict(int)
        
        for i, entity in enumerate(entities_to_process, 1):
            entity_id = entity['id']
            entity_name = entity['canonical_name']
            entity_type = entity['entity_type']
            
            print(f"\n[{i}/{len(entities_to_process)}] Processing: {entity_name} (ID: {entity_id}, Type: {entity_type})", file=sys.stderr)
            
            stats = extract_mentions_for_entity_citations(
                conn,
                entity_id,
                entity_name,
                aliases_by_norm,
                alias_norm_set,
                dry_run=args.dry_run,
                verbose=args.verbose
            )
            
            # Accumulate totals
            for key, value in stats.items():
                total_stats[key] += value
            
            # Print summary for this entity
            if stats['mentions_found'] > 0 or args.verbose:
                print(f"  Citations: {stats['citations_processed']} processed, "
                      f"{stats['citations_with_documents']} with documents, "
                      f"{stats['citations_with_pages']} with pages, "
                      f"{stats['citations_with_chunks']} with chunks", file=sys.stderr)
                print(f"  Chunks: {stats['chunks_processed']} processed, "
                      f"Mentions: {stats['mentions_found']} found", file=sys.stderr)
        
        # Print overall summary
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"SUMMARY", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"Entities processed: {len(entities_to_process)}", file=sys.stderr)
        print(f"Total citations processed: {total_stats['citations_processed']}", file=sys.stderr)
        print(f"  - With documents: {total_stats['citations_with_documents']}", file=sys.stderr)
        print(f"  - With pages: {total_stats['citations_with_pages']}", file=sys.stderr)
        print(f"  - With chunks: {total_stats['citations_with_chunks']}", file=sys.stderr)
        print(f"  - No document match: {total_stats['citations_no_document']}", file=sys.stderr)
        print(f"  - No pages match: {total_stats['citations_no_pages']}", file=sys.stderr)
        print(f"  - No chunks match: {total_stats['citations_no_chunks']}", file=sys.stderr)
        print(f"Total chunks processed: {total_stats['chunks_processed']}", file=sys.stderr)
        print(f"Total mentions {'would be extracted' if args.dry_run else 'extracted'}: {total_stats['mentions_found']}", file=sys.stderr)
        
        if args.dry_run:
            print(f"\nRun without --dry-run to actually insert mentions", file=sys.stderr)
    
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
