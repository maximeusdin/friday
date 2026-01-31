#!/usr/bin/env python3
"""
Pattern-based entity extraction using regex patterns.

Extracts person names, organizations, and places using rule-based patterns.
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

# Person name patterns
PERSON_PATTERNS = [
    # Full name: "John Smith"
    (r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'person_full_name', 0.8),
    # With title: "Mr. John Smith", "Dr. Jane Doe"
    (r'\b(Mr\.|Mrs\.|Ms\.|Miss|Dr\.|General|Colonel|Captain|Major|Lieutenant|Professor|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'person_with_title', 0.9),
    # Last, First: "Smith, John"
    (r'\b([A-Z][a-z]+),\s+([A-Z][a-z]+)', 'person_last_first', 0.85),
]

# Organization patterns
ORG_PATTERNS = [
    # With marker: "Ministry of Foreign Affairs", "Department of State"
    (r'\b([A-Z][a-z]+ (?:Ministry|Department|Bureau|Agency|Office|Committee|Commission|Board|Council))\s+(?:of|for|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'org_with_marker', 0.9),
    # Simple org with marker: "FBI", "CIA", "NATO" (require at least 3 letters)
    (r'\b([A-Z]{3,6})\b', 'org_acronym', 0.7),  # 3-6 letter acronyms only
    # Company: "Acme Inc.", "Corp.", "Ltd."
    (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|Corp\.|Corporation|Ltd\.|LLC)', 'org_company', 0.85),
    # "The X": "The White House", "The Pentagon" (but skip common "The [Common Word]")
    (r'\bThe\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'org_the', 0.75),
    # Known organization patterns: "United States", "House of Representatives"
    (r'\b(United States|House of Representatives|Senate|Congress)\b', 'org_known', 0.9),
]

# Place patterns
PLACE_PATTERNS = [
    # Geographic context: "in New York", "from Moscow", "to London"
    (r'\b(in|from|to|near|at|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'place_geographic', 0.8),
    # City, State: "New York, NY", "Washington, DC"
    (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+([A-Z]{2})', 'place_city_state', 0.85),
    # Removed standalone pattern - too many false positives
    # Use geographic context or city/state format instead
]

# Titles that indicate person names
PERSON_TITLES = {
    'Mr.', 'Mrs.', 'Ms.', 'Miss', 'Dr.', 'Doctor', 'General', 'Colonel',
    'Captain', 'Major', 'Lieutenant', 'Professor', 'Prof.', 'Senator',
    'Representative', 'Congressman', 'Congresswoman'
}


def extract_person_names(text: str) -> List[Dict]:
    """Extract person names using patterns."""
    results = []
    seen_positions = set()
    
    for pattern, pattern_type, base_confidence in PERSON_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.span()
            
            # Skip if position already matched
            if (start, end) in seen_positions:
                continue
            
            # Extract surface form
            if pattern_type == 'person_with_title':
                title = match.group(1)
                name = match.group(2)
                surface = f"{title} {name}"
                confidence = base_confidence + 0.05  # Boost for title
            elif pattern_type == 'person_last_first':
                last = match.group(1)
                first = match.group(2)
                surface = f"{last}, {first}"
                confidence = base_confidence
            else:
                surface = match.group(1)
                confidence = base_confidence
            
            # Validate: should be reasonable length
            tokens = surface.split()
            if len(tokens) < 2 or len(tokens) > 5:
                continue
            
            # Check if contains title (boost confidence)
            has_title = any(title.lower() in surface.lower() for title in PERSON_TITLES)
            if has_title:
                confidence += 0.05
            
            results.append({
                'entity_type': 'person',
                'surface': surface,
                'start_char': start,
                'end_char': end,
                'confidence': min(confidence, 1.0),
                'pattern_type': pattern_type,
            })
            
            seen_positions.add((start, end))
    
    return results


def extract_organizations(text: str) -> List[Dict]:
    """Extract organizations using patterns."""
    results = []
    seen_positions = set()
    
    for pattern, pattern_type, base_confidence in ORG_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.span()
            
            # Skip if position already matched
            if (start, end) in seen_positions:
                continue
            
            if pattern_type == 'org_known':
                # Known organizations - use as-is
                surface = match.group(1)
            elif pattern_type == 'org_the':
                # Extract the part after "The"
                surface = match.group(1)
                # Skip common words that follow "The"
                common_the_words = {
                    'White', 'House', 'Pentagon', 'Capitol', 'Treasury', 'State', 'Department',
                    'Federal', 'Bureau', 'Agency', 'Office', 'Committee', 'Commission', 'Board'
                }
                # Only skip if it's a single common word (multi-word like "White House" is OK)
                if surface in common_the_words and len(surface.split()) == 1:
                    continue
                surface = f"The {surface}"  # Reconstruct full form
            else:
                surface = match.group(0)  # Full match
            
            # Validate acronyms: should be 3-6 uppercase letters (2-letter too ambiguous)
            if pattern_type == 'org_acronym':
                # Require at least 3 letters for acronyms (2-letter are too ambiguous)
                if not re.match(r'^[A-Z]{3,6}$', surface):
                    continue
                
                # Skip common non-org acronyms and stopwords
                common_words = {
                    'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 
                    'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 
                    'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'OF', 'ON', 'TO', 'UN', 'IN', 'IS',
                    'IT', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'IF', 'MY', 'NO', 'OR', 'SO', 'UP', 'WE',
                    'PRINT', 'REPORT', 'PAGE', 'PAGES', 'CHAPTER', 'SECTION', 'TABLE', 'FIGURE', 'TITLE', 'HEADER',
                    'FOOTER', 'CONTENT', 'INDEX', 'APPENDIX'
                }
                if surface in common_words:
                    continue
                
                # Skip PDF metadata strings (expanded list)
                pdf_metadata = {
                    'PDF', 'ID', 'EOF', 'OBJ', 'END', 'XREF', 'ROOT', 'INFO', 'SIZE', 'PREV', 'TYPE', 
                    'CATALOG', 'PAGES', 'KIDS', 'COUNT', 'PARENT', 'LINEARIZED', 'W', 'LENGTH', 'FILTER',
                    'FONT', 'SUBTYPE', 'BASE', 'ENCODING', 'WIDTH', 'HEIGHT', 'BOUNDS', 'RECT', 'MEDIABOX',
                    'METADATA', 'STREAM', 'FLATE', 'DECODE', 'PARAMS', 'COLORSPACE', 'IMAGE', 'BITMAP',
                    'GE', 'OC', 'CP', 'BJ', 'XR', 'GS', 'RG', 'K', 'G', 'F', 'S', 'M', 'L', 'C', 'Q', 'CM', 'TM'
                }
                if surface in pdf_metadata:
                    continue
                
                # Skip if appears to be part of PDF stream/object notation
                # Check context around the match
                context_start = max(0, start - 10)
                context_end = min(len(text), end + 10)
                context = text[context_start:context_end].lower()
                if any(pdf_term in context for pdf_term in ['/stream', '/filter', '/length', 'obj', 'endobj', '/flatedecode']):
                    continue
            
            # General filtering for all organization types
            # Skip common document header words
            common_doc_words = {
                'PRINT', 'REPORT', 'PAGE', 'PAGES', 'CHAPTER', 'SECTION', 'TABLE', 'FIGURE', 
                'TITLE', 'HEADER', 'FOOTER', 'CONTENT', 'INDEX', 'APPENDIX', 'UNITED', 'STATES',
                'HOUSE', 'REPRESENTATIVES', 'SENATE', 'CONGRESS'
            }
            if surface.upper() in common_doc_words:
                continue
            
            # Skip PDF metadata words (case-insensitive)
            pdf_metadata_lower = {
                'metadata', 'length', 'filter', 'stream', 'flate', 'decode', 'params', 
                'colorspace', 'image', 'bitmap', 'font', 'subtype', 'base', 'encoding',
                'width', 'height', 'bounds', 'rect', 'mediabox'
            }
            if surface.lower() in pdf_metadata_lower:
                continue
            
            results.append({
                'entity_type': 'org',
                'surface': surface,
                'start_char': start,
                'end_char': end,
                'confidence': base_confidence,
                'pattern_type': pattern_type,
            })
            
            seen_positions.add((start, end))
    
    return results


def extract_places(text: str) -> List[Dict]:
    """Extract places using patterns."""
    results = []
    seen_positions = set()
    
    for pattern, pattern_type, base_confidence in PLACE_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.span()
            
            # Skip if position already matched
            if (start, end) in seen_positions:
                continue
            
            if pattern_type == 'place_geographic':
                marker = match.group(1)
                place = match.group(2)
                surface = place  # Just the place name
                # Require place name to be at least 2 words or a known place pattern
                # Single words are too ambiguous (could be common nouns)
                if len(surface.split()) == 1:
                    # Skip single-word places unless they're known place names
                    known_single_places = {
                        'London', 'Paris', 'Moscow', 'Berlin', 'Rome', 'Madrid', 'Vienna', 'Warsaw',
                        'Tokyo', 'Beijing', 'Delhi', 'Cairo', 'Sydney', 'Toronto', 'Mexico', 'Lima',
                        'Washington', 'Boston', 'Chicago', 'Miami', 'Seattle', 'Denver', 'Phoenix'
                    }
                    if surface not in known_single_places:
                        continue
            elif pattern_type == 'place_city_state':
                city = match.group(1)
                state = match.group(2)
                surface = f"{city}, {state}"
            else:
                # Should not reach here with current patterns
                surface = match.group(1)
            
            # Skip if too short or too long
            if len(surface) < 3 or len(surface) > 50:
                continue
            
            # Skip common words that match pattern (expanded list)
            common_words = {
                'The', 'This', 'That', 'There', 'These', 'Those', 'Then', 'Than',
                'Print', 'Report', 'Page', 'Pages', 'Chapter', 'Section', 'Table', 'Figure', 'Title', 'Header',
                'Footer', 'Content', 'Index', 'Appendix', 'Introduction', 'Conclusion', 'Summary',
                'Linearized', 'Root', 'Info', 'Size', 'Prev', 'Type', 'Catalog', 'Metadata', 'Length', 'Filter',
                'United', 'States', 'House', 'Representatives', 'Senate', 'Congress', 'Government', 'Department',
                'Ministry', 'Bureau', 'Agency', 'Office', 'Committee', 'Commission', 'Board', 'Council'
            }
            if surface in common_words:
                continue
            
            # Skip if looks like PDF metadata (contains common PDF keywords)
            pdf_keywords = [
                'PDF', 'Linearized', 'Root', 'Info', 'Size', 'Prev', 'Type', 'Catalog', 'Pages', 'Kids',
                'Metadata', 'Length', 'Filter', 'Stream', 'Flate', 'Decode', 'Params', 'Colorspace',
                'Image', 'Bitmap', 'Font', 'Subtype', 'Base', 'Encoding', 'Width', 'Height', 'Bounds', 'Rect'
            ]
            if any(keyword.lower() in surface.lower() for keyword in pdf_keywords):
                continue
            
            # Skip single-word places that are common document words
            if len(surface.split()) == 1 and surface.upper() in ['UNITED', 'STATES', 'HOUSE', 'REPRESENTATIVES', 'SENATE', 'CONGRESS', 'PRINT', 'REPORT', 'PAGE']:
                continue
            
            results.append({
                'entity_type': 'place',
                'surface': surface,
                'start_char': start,
                'end_char': end,
                'confidence': base_confidence,
                'pattern_type': pattern_type,
            })
            
            seen_positions.add((start, end))
    
    return results


def extract_pattern_based(text: str, entity_types: Optional[List[str]] = None) -> List[Dict]:
    """
    Extract entities using pattern matching.
    
    Args:
        text: Text to extract from
        entity_types: List of entity types to extract ('person', 'org', 'place')
                     If None, extracts all types
    
    Returns:
        List of entity dictionaries
    """
    if entity_types is None:
        entity_types = ['person', 'org', 'place']
    
    results = []
    
    if 'person' in entity_types:
        results.extend(extract_person_names(text))
    
    if 'org' in entity_types:
        results.extend(extract_organizations(text))
    
    if 'place' in entity_types:
        results.extend(extract_places(text))
    
    # Sort by position
    results.sort(key=lambda x: x['start_char'])
    
    return results


def process_chunks(
    conn,
    collection_slug: Optional[str] = None,
    document_id: Optional[int] = None,
    entity_types: Optional[List[str]] = None,
    confidence_threshold: float = 0.7,
    limit: Optional[int] = None,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Process chunks and extract entities."""
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
        SELECT c.id, c.text, cm.document_id
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
    
    for chunk_id, text, doc_id in chunks:
        entities = extract_pattern_based(text, entity_types)
        
        for entity in entities:
            if entity['confidence'] >= confidence_threshold:
                total_entities += 1
                
                if not dry_run:
                    # Check if entity exists
                    surface_norm = normalize_alias(entity['surface'])
                    cur.execute("""
                        SELECT id FROM entities
                        WHERE entity_type = %s
                        AND LOWER(canonical_name) = LOWER(%s)
                        LIMIT 1
                    """, (entity['entity_type'], entity['surface']))
                    
                    entity_row = cur.fetchone()
                    if entity_row:
                        entity_id = entity_row[0]
                    else:
                        # Create new entity
                        cur.execute("""
                            INSERT INTO entities (entity_type, canonical_name)
                            VALUES (%s, %s)
                            RETURNING id
                        """, (entity['entity_type'], entity['surface']))
                        entity_id = cur.fetchone()[0]
                    
                    # Insert mention
                    entities_to_insert.append((
                        entity_id,
                        chunk_id,
                        doc_id,
                        entity['surface'],
                        entity.get('start_char'),
                        entity.get('end_char'),
                        entity['confidence'],
                        'pattern_based'
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
        description="Extract entities using pattern-based matching"
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
        "--entity-type",
        action="append",
        choices=['person', 'org', 'place'],
        help="Entity types to extract (can specify multiple)"
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
        help="Test extraction on provided text string"
    )
    
    args = parser.parse_args()
    
    if args.test_text:
        entities = extract_pattern_based(args.test_text, args.entity_type)
        print(f"Found {len(entities)} entities:")
        for e in entities:
            print(f"  {e['entity_type']}: '{e['surface']}' (confidence: {e['confidence']:.2f})")
        return
    
    conn = get_conn()
    try:
        chunks_processed, entities_found = process_chunks(
            conn,
            collection_slug=args.collection,
            document_id=args.document_id,
            entity_types=args.entity_type,
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
