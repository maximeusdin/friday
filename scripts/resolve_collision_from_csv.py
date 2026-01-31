#!/usr/bin/env python3
"""
Resolve entity collisions using chunk_id from the CSV export.

This script reads entity mentions from the CSV and uses chunk_id to:
1. Look up the chunk's document and page information
2. Match against entity citations to resolve collisions
3. Output resolved matches

Usage:
    python scripts/resolve_collision_from_csv.py concordance_export/entity_mentions.csv
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from retrieval.entity_resolver import normalize_alias

# Import citation parsing from extract_entity_mentions
from scripts.extract_entity_mentions import (
    parse_citation_text,
    CitationLocation,
    expand_page_ranges,
    batch_load_entity_citations,
    batch_load_chunk_metadata
)

def get_conn():
    """Get database connection using environment variables."""
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "neh")
    DB_USER = os.getenv("DB_USER", "neh")
    DB_PASS = os.getenv("DB_PASS", "neh")
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def find_documents_for_citation(cur, citation_location: CitationLocation) -> List[Tuple[int, str]]:
    """
    Find documents matching a citation location.
    Simplified version - matches document source_name.
    """
    source_normalized = citation_location.source.lower().strip()
    
    cur.execute("""
        SELECT id, source_name
        FROM documents
        WHERE LOWER(source_name) LIKE %s
        OR LOWER(source_name) LIKE %s
    """, (f"%{source_normalized}%", f"{source_normalized}%"))
    
    return [(row[0], row[1]) for row in cur.fetchall()]


def resolve_collision_with_chunk(
    alias_infos: List[Dict],
    chunk_id: int,
    chunk_metadata_cache: Dict[int, Dict],
    entity_citations_cache: Dict[int, List[Dict]],
    document_cache: Optional[Dict[str, int]] = None,
    cur=None
) -> Tuple[Optional[int], float, Optional[str]]:
    """
    Resolve collision by checking if any candidate entity has citations
    that match the chunk's document and pages.
    
    Args:
        alias_infos: List of candidate entity info dicts with 'entity_id'
        chunk_id: The chunk ID from the CSV
        chunk_metadata_cache: Dict mapping chunk_id -> {'page_ids': [...], 'pdf_pages': [...], 'document_id': ...}
        entity_citations_cache: Dict mapping entity_id -> [{'citation_text': ..., 'page_list': ...}, ...]
        document_cache: Optional dict mapping document_name -> document_id
        cur: Optional database cursor (for fallback queries)
    
    Returns:
        (resolved_entity_id, confidence_score, method)
        - resolved_entity_id: The entity to use, or None if no match
        - confidence_score: 0.0-1.0 based on citation match quality
        - method: 'citation_exact', 'citation_fuzzy', or None
    """
    # Get chunk metadata
    chunk_meta = chunk_metadata_cache.get(chunk_id)
    if not chunk_meta:
        return None, 0.0, None
    
    document_id = chunk_meta.get('document_id')
    pdf_page_numbers = chunk_meta.get('pdf_pages', [])
    
    if not pdf_page_numbers or not document_id:
        return None, 0.0, None
    
    # Get document name for matching
    if cur and not document_cache:
        cur.execute("SELECT source_name FROM documents WHERE id = %s", (document_id,))
        row = cur.fetchone()
        document_name = row[0] if row else None
    elif document_cache:
        # Reverse lookup - find document name from cache
        document_name = None
        for name, doc_id in document_cache.items():
            if doc_id == document_id:
                document_name = name
                break
    else:
        document_name = None
    
    if not document_name:
        return None, 0.0, None
    
    # Check citations for each candidate entity
    candidate_scores = []
    for ai in alias_infos:
        entity_id = ai.get('entity_id') if isinstance(ai, dict) else ai
        if isinstance(ai, dict):
            entity_id = ai['entity_id']
        else:
            entity_id = ai  # Assume it's just the entity_id
        
        # Get all citations for this entity
        citations = entity_citations_cache.get(entity_id, [])
        
        best_score = 0.0
        best_method = None
        
        for citation in citations:
            citation_text = citation.get('citation_text')
            if not citation_text:
                continue
            
            # Parse citation to get document and pages
            try:
                citation_locations = parse_citation_text(citation_text)
            except Exception:
                continue  # Skip malformed citations
            
            for loc in citation_locations:
                # Check if citation document matches chunk document
                citation_docs = find_documents_for_citation(cur, loc) if cur else []
                citation_doc_ids = [doc_id for doc_id, _ in citation_docs]
                
                if document_id not in citation_doc_ids:
                    # Try fuzzy match on document name
                    source_normalized = loc.source.lower().strip()
                    doc_name_normalized = document_name.lower().strip()
                    if source_normalized not in doc_name_normalized and doc_name_normalized not in source_normalized:
                        continue  # Different document
                
                # Check page overlap
                citation_pages = expand_page_ranges(loc.pages)
                if not citation_pages:
                    continue
                
                overlap = set(pdf_page_numbers) & set(citation_pages)
                
                if overlap:
                    # Calculate match score
                    overlap_ratio = len(overlap) / len(pdf_page_numbers)
                    if len(overlap) == len(pdf_page_numbers) and len(overlap) == len(citation_pages):
                        score = 1.0  # Exact match
                        method = 'citation_exact'
                    elif overlap_ratio >= 0.5:
                        score = 0.8  # Good overlap
                        method = 'citation_fuzzy'
                    else:
                        score = 0.5  # Partial overlap
                        method = 'citation_fuzzy'
                    
                    if score > best_score:
                        best_score = score
                        best_method = method
        
        if best_score > 0.0:
            candidate_scores.append((entity_id, best_score, best_method))
    
    # Return the best match
    if candidate_scores:
        # Sort by score descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_entity_id, best_score, best_method = candidate_scores[0]
        
        # If there's a clear winner (score > 0.7 and significantly better than second)
        if best_score > 0.7:
            if len(candidate_scores) == 1 or candidate_scores[0][1] > candidate_scores[1][1] + 0.2:
                return best_entity_id, best_score, best_method
    
    return None, 0.0, None


def analyze_csv_collisions(csv_path: Path):
    """Analyze collisions from CSV and show how chunk_id can resolve them."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Read CSV
            print(f"Reading CSV: {csv_path}", file=sys.stderr)
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            print(f"Found {len(rows)} entity mentions in CSV", file=sys.stderr)
            
            # Group by alias_norm to find collisions
            alias_to_entities = defaultdict(set)
            alias_to_chunks = defaultdict(list)
            
            for row in rows:
                entity_id = int(row['entity_id'])
                canonical_name = row['canonical_name']
                chunk_id = int(row['chunk_id']) if row['chunk_id'] else None
                
                # Normalize the canonical name as alias
                alias_norm = normalize_alias(canonical_name)
                alias_to_entities[alias_norm].add(entity_id)
                
                if chunk_id:
                    alias_to_chunks[alias_norm].append({
                        'chunk_id': chunk_id,
                        'entity_id': entity_id,
                        'document_id': int(row['document_id']) if row['document_id'] else None,
                        'document_name': row['document_name'],
                        'pdf_page_number': int(row['pdf_page_number']) if row['pdf_page_number'] else None,
                    })
            
            # Find collisions (aliases with multiple entities)
            collisions = {alias: entities for alias, entities in alias_to_entities.items() 
                         if len(entities) > 1}
            
            print(f"\nFound {len(collisions)} aliases with collisions", file=sys.stderr)
            
            # Load chunk metadata for all chunks
            all_chunk_ids = set()
            for chunks in alias_to_chunks.values():
                all_chunk_ids.update(c['chunk_id'] for c in chunks if c['chunk_id'])
            
            print(f"Loading metadata for {len(all_chunk_ids)} chunks...", file=sys.stderr)
            chunk_metadata_cache = batch_load_chunk_metadata(cur, list(all_chunk_ids))
            
            # Add document_id to chunk metadata
            for chunk_id, meta in chunk_metadata_cache.items():
                # Get document_id from chunk_metadata table
                cur.execute("""
                    SELECT document_id
                    FROM chunk_metadata
                    WHERE chunk_id = %s
                    LIMIT 1
                """, (chunk_id,))
                row = cur.fetchone()
                if row:
                    meta['document_id'] = row[0]
            
            # Load entity citations for all entities
            all_entity_ids = set()
            for entities in collisions.values():
                all_entity_ids.update(entities)
            
            print(f"Loading citations for {len(all_entity_ids)} entities...", file=sys.stderr)
            entity_citations_cache = batch_load_entity_citations(cur, list(all_entity_ids))
            
            # Analyze each collision
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"COLLISION ANALYSIS USING CHUNK_ID", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            
            for alias_norm, entity_ids in sorted(collisions.items())[:10]:  # Show first 10
                print(f"\nAlias: '{alias_norm}'", file=sys.stderr)
                print(f"  Candidate entities: {sorted(entity_ids)}", file=sys.stderr)
                
                chunks_for_alias = alias_to_chunks.get(alias_norm, [])
                print(f"  Chunks with this alias: {len(chunks_for_alias)}", file=sys.stderr)
                
                # Try to resolve using chunk_id
                resolved_count = 0
                for chunk_info in chunks_for_alias[:5]:  # Show first 5
                    chunk_id = chunk_info['chunk_id']
                    chunk_meta = chunk_metadata_cache.get(chunk_id, {})
                    
                    entity_id_list = [{'entity_id': eid} for eid in entity_ids]
                    resolved_entity_id, score, method = resolve_collision_with_chunk(
                        entity_id_list,
                        chunk_id,
                        chunk_metadata_cache,
                        entity_citations_cache,
                        cur=cur
                    )
                    
                    if resolved_entity_id:
                        resolved_count += 1
                        print(f"    Chunk {chunk_id}: Resolved to entity {resolved_entity_id} "
                              f"(score: {score:.2f}, method: {method})", file=sys.stderr)
                    else:
                        print(f"    Chunk {chunk_id}: Could not resolve (document: {chunk_info['document_name']}, "
                              f"page: {chunk_info.get('pdf_page_number')})", file=sys.stderr)
                
                if resolved_count > 0:
                    print(f"  → {resolved_count}/{len(chunks_for_alias)} chunks resolved", file=sys.stderr)
            
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Resolve entity collisions using chunk_id from CSV")
    ap.add_argument("csv_path", type=Path, help="Path to entity_mentions.csv")
    
    args = ap.parse_args()
    
    if not args.csv_path.exists():
        print(f"ERROR: CSV file not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)
    
    analyze_csv_collisions(args.csv_path)


if __name__ == "__main__":
    main()
