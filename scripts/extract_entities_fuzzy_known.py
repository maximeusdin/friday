#!/usr/bin/env python3
"""
Fuzzy matching against known entities from entity_aliases table.

Extracts candidate surface forms and matches them against existing entities
using trigram similarity (pg_trgm).
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from retrieval.entity_resolver import normalize_alias
from retrieval.ops import get_conn

# Token extraction pattern (same as extract_entity_mentions.py)
TOKEN_PATTERN = r'\b[\w\']+(?:[./-][\w\']+)*\b'


def extract_candidate_surfaces(text: str, min_length: int = 3, max_tokens: int = 5) -> List[Tuple[int, int, str]]:
    """
    Extract candidate surface forms from text.
    
    Returns list of (start_pos, end_pos, surface) tuples.
    """
    candidates = []
    
    # Extract tokens
    tokens = []
    for match in re.finditer(TOKEN_PATTERN, text):
        token = match.group(0)
        if re.search(r'[\w]', token):  # Must contain at least one word char
            tokens.append((match.start(), match.end(), token))
    
    # Generate n-grams (1-gram through max_tokens)
    for n in range(1, min(max_tokens + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            token_slice = tokens[i:i + n]
            start_pos = token_slice[0][0]
            end_pos = token_slice[-1][1]
            surface = ' '.join(t[2] for t in token_slice)
            
            # Filter by length
            if len(surface) >= min_length:
                candidates.append((start_pos, end_pos, surface))
    
    return candidates


def fuzzy_match_against_known(
    conn,
    surface: str,
    text_quality: str = "unknown",
    similarity_threshold: float = 0.7,
    max_results: int = 10
) -> List[Dict]:
    """
    Match surface against known entities using trigram similarity.
    
    Returns list of match dictionaries with entity_id, similarity, etc.
    """
    surface_norm = normalize_alias(surface)
    
    # Adjust threshold based on text quality
    if text_quality == 'ocr':
        similarity_threshold = max(0.6, similarity_threshold - 0.1)
    
    cur = conn.cursor()
    
    # Use pg_trgm similarity operator
    cur.execute("""
        SELECT 
            ea.entity_id,
            ea.alias,
            ea.alias_norm,
            e.entity_type,
            e.canonical_name,
            similarity(ea.alias_norm, %s) as sim_score
        FROM entity_aliases ea
        JOIN entities e ON ea.entity_id = e.id
        WHERE ea.alias_norm % %s  -- Trigram similarity operator
        AND similarity(ea.alias_norm, %s) >= %s
        ORDER BY sim_score DESC
        LIMIT %s
    """, (surface_norm, surface_norm, surface_norm, similarity_threshold, max_results))
    
    matches = []
    for row in cur.fetchall():
        matches.append({
            'entity_id': row[0],
            'alias': row[1],
            'alias_norm': row[2],
            'entity_type': row[3],
            'canonical_name': row[4],
            'similarity': float(row[5]),
            'confidence': float(row[5])  # Use similarity as confidence
        })
    
    return matches


def process_chunks(
    conn,
    collection_slug: Optional[str] = None,
    document_id: Optional[int] = None,
    similarity_threshold: float = 0.7,
    confidence_threshold: float = 0.7,
    limit: Optional[int] = None,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Process chunks and extract entities using fuzzy matching."""
    cur = conn.cursor()
    
    # Build query
    conditions = []
    params = []
    
    if collection_slug:
        conditions.append("""
            EXISTS (
                SELECT 1 FROM chunk_metadata cm
                JOIN documents d ON cm.document_id = d.id
                JOIN collections c ON d.collection_id = c.id
                WHERE cm.chunk_id = chunks.id AND c.slug = %s
            )
        """)
        params.append(collection_slug)
    
    if document_id:
        conditions.append("""
            EXISTS (
                SELECT 1 FROM chunk_metadata cm
                WHERE cm.chunk_id = chunks.id AND cm.document_id = %s
            )
        """)
        params.append(document_id)
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
        SELECT c.id, c.text, c.text_quality, cm.document_id
        FROM chunks c
        JOIN chunk_metadata cm ON c.id = cm.chunk_id
        WHERE {where_clause}
        ORDER BY c.id
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query, params)
    chunks = cur.fetchall()
    
    total_entities = 0
    entities_to_insert = []
    seen_positions = {}  # (chunk_id, start, end) -> best match
    
    for chunk_id, text, text_quality, doc_id in chunks:
        if not text:
            continue
        
        text_quality = text_quality or 'unknown'
        
        # Extract candidate surfaces
        candidates = extract_candidate_surfaces(text)
        
        for start_pos, end_pos, surface in candidates:
            # Skip if too short or too long
            if len(surface) < 3 or len(surface) > 50:
                continue
            
            # Match against known entities
            matches = fuzzy_match_against_known(
                conn,
                surface,
                text_quality,
                similarity_threshold
            )
            
            if matches:
                # Take best match
                best_match = matches[0]
                
                if best_match['confidence'] >= confidence_threshold:
                    # Check for collision at same position
                    position_key = (chunk_id, start_pos, end_pos)
                    if position_key in seen_positions:
                        # Keep higher confidence match
                        existing = seen_positions[position_key]
                        if best_match['confidence'] > existing['confidence']:
                            seen_positions[position_key] = best_match
                    else:
                        seen_positions[position_key] = best_match
                        total_entities += 1
                        
                        if not dry_run:
                            entities_to_insert.append((
                                best_match['entity_id'],
                                chunk_id,
                                doc_id,
                                surface,
                                start_pos,
                                end_pos,
                                best_match['confidence'],
                                'fuzzy_known'
                            ))
    
    if not dry_run and entities_to_insert:
        cur.executemany("""
            INSERT INTO entity_mentions
            (entity_id, chunk_id, document_id, surface, start_char, end_char, confidence, method)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, entities_to_insert)
        conn.commit()
        print(f"Inserted {len(entities_to_insert)} entity mentions", file=sys.stderr)
    
    return len(chunks), total_entities


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities using fuzzy matching against known entities"
    )
    parser.add_argument(
        "--collection",
        help="Collection slug"
    )
    parser.add_argument(
        "--document-id",
        type=int,
        help="Document ID"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Minimum trigram similarity threshold (default: 0.7)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of chunks to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't insert, just show what would be extracted"
    )
    parser.add_argument(
        "--test-text",
        help="Test matching on provided text string"
    )
    
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        if args.test_text:
            candidates = extract_candidate_surfaces(args.test_text)
            print(f"Found {len(candidates)} candidate surfaces:")
            for start, end, surface in candidates[:20]:  # Show first 20
                matches = fuzzy_match_against_known(
                    conn,
                    surface,
                    similarity_threshold=args.similarity_threshold
                )
                if matches:
                    best = matches[0]
                    print(f"  '{surface}' -> {best['canonical_name']} ({best['entity_type']}, similarity: {best['similarity']:.2f})")
            return
        
        chunks_processed, entities_found = process_chunks(
            conn,
            collection_slug=args.collection,
            document_id=args.document_id,
            similarity_threshold=args.similarity_threshold,
            confidence_threshold=args.confidence_threshold,
            limit=args.limit,
            dry_run=args.dry_run
        )
        
        print(f"\nProcessed {chunks_processed} chunks", file=sys.stderr)
        print(f"Found {entities_found} entities above threshold", file=sys.stderr)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
