#!/usr/bin/env python3
"""
Validate entity mentions in chunks using citation information from concordance entries.

This script:
1. Extracts citation locations from concordance entries (e.g., "Venona New York KGB 1941–42, 16, 74–75")
2. Maps citations to documents and pages in the database
3. Checks if chunks from those pages contain mentions of the entity
4. Reports validation results

Usage:
    python concordance/validate_entity_mentions_from_citations.py --entity-id 123
    python concordance/validate_entity_mentions_from_citations.py --entity-name "AKIM"
    python concordance/validate_entity_mentions_from_citations.py --entry-key "AKIM (cover name in Venona)"
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import Json


@dataclass
class CitationLocation:
    """Parsed citation location from concordance entry"""
    source: str  # e.g., "Venona New York KGB"
    year_range: Optional[str]  # e.g., "1941–42" or "1941-1942"
    pages: List[Tuple[int, Optional[int]]]  # List of (start_page, end_page) tuples, end_page=None for single pages
    citation_text: str  # Original citation text


@dataclass
class ValidationResult:
    """Result of validating entity mention in a citation location"""
    citation_location: CitationLocation
    document_id: Optional[int]
    document_name: Optional[str]
    matched_pages: List[int]  # Page IDs that match the citation
    chunks_with_entity: List[int]  # Chunk IDs that contain the entity
    chunks_checked: int
    validation_status: str  # "validated", "no_document", "no_pages", "no_mentions"


def parse_citation_text(citation_text: str) -> List[CitationLocation]:
    """
    Parse citation text like:
    "Venona New York KGB 1941–42, 16, 74–75; Venona New York KGB 1943, 112–13, 161–62"
    "Venona Special Studies, 3–4, 93"
    "Vassiliev Yellow Notebook #2, 18, 21, 72, 83"
    
    Handles multi-line citations (e.g., "Venona San\nFrancisco KGB, 144")
    
    Returns list of CitationLocation objects.
    """
    locations = []
    
    # Normalize whitespace first - replace newlines with spaces, collapse multiple spaces
    citation_text = re.sub(r'\s+', ' ', citation_text.strip())
    
    # Split on semicolons to get separate citation groups
    citation_groups = citation_text.split(';')
    
    for group in citation_groups:
        group = group.strip()
        if not group:
            continue
        
        # Pattern: "Venona New York KGB 1941–42, 16, 74–75"
        # Pattern: "Venona Special Studies, 3–4, 93" (no year)
        # Pattern: "Vassiliev Yellow Notebook #2, 18, 21, 72, 83"
        
        # Try to match source + year/volume + pages
        # Pattern 1: "Venona [source] [year-range], [pages]"
        # Match: "Venona New York KGB 1941–42, 16, 74–75"
        venona_with_year = re.match(
            r'^Venona\s+([^,]+?)\s+(\d{4}[–-]\d{2,4})\s*,\s*(.+)$',
            group,
            re.IGNORECASE
        )
        
        # Pattern 2: "Venona [source], [pages]" (no year)
        # Match: "Venona Special Studies, 3–4, 93"
        venona_no_year = re.match(
            r'^Venona\s+([^,]+?)\s*,\s*(.+)$',
            group,
            re.IGNORECASE
        )
        
        # Pattern 3: "Vassiliev [notebook] [volume], [pages]"
        # Match: "Vassiliev Yellow Notebook #2, 18, 21, 72, 83"
        # Handle both straight and curly apostrophes: 's or 's
        vassiliev_match = re.match(
            r"^Vassiliev(?:['']s)?\s+([^,]+?)(?:\s+#\d+)?\s*,\s*(.+)$",
            group,
            re.IGNORECASE
        )
        
        if venona_with_year:
            source = f"Venona {venona_with_year.group(1).strip()}"
            year_range = venona_with_year.group(2).strip()
            pages_str = venona_with_year.group(3).strip()
            
            pages = parse_page_numbers(pages_str)
            
            locations.append(CitationLocation(
                source=source,
                year_range=year_range,
                pages=pages,
                citation_text=group
            ))
        
        elif venona_no_year:
            source = f"Venona {venona_no_year.group(1).strip()}"
            pages_str = venona_no_year.group(2).strip()
            
            pages = parse_page_numbers(pages_str)
            
            locations.append(CitationLocation(
                source=source,
                year_range=None,
                pages=pages,
                citation_text=group
            ))
        
        elif vassiliev_match:
            source = f"Vassiliev {vassiliev_match.group(1).strip()}"
            pages_str = vassiliev_match.group(2).strip()
            
            pages = parse_page_numbers(pages_str)
            
            locations.append(CitationLocation(
                source=source,
                year_range=None,  # Vassiliev citations don't always have year ranges
                pages=pages,
                citation_text=group
            ))
        
        else:
            # Try to extract just pages if source is unclear
            # Look for comma-separated numbers at the end
            pages_match = re.search(r',\s*([\d\s,–-]+)$', group)
            if pages_match:
                pages_str = pages_match.group(1).strip()
                pages = parse_page_numbers(pages_str)
                
                # Try to extract source from beginning
                source_part = group[:pages_match.start()].strip()
                source = source_part if source_part else "Unknown"
                
                locations.append(CitationLocation(
                    source=source,
                    year_range=None,
                    pages=pages,
                    citation_text=group
                ))
    
    return locations


def parse_page_numbers(pages_str: str) -> List[Tuple[int, Optional[int]]]:
    """
    Parse page number string like "16, 74–75, 112–13, 161–62" into list of (start, end) tuples.
    
    Handles:
    - Single pages: "16"
    - Full ranges: "74–75"
    - Abbreviated ranges: "112–13" (means 112-113), "161–62" (means 161-162)
    
    Returns list of (start_page, end_page) where end_page is None for single pages.
    """
    pages = []
    
    # Split on commas
    page_parts = pages_str.split(',')
    
    for part in page_parts:
        part = part.strip()
        if not part:
            continue
        
        # Match page ranges like "74–75" or "112–13" or single pages like "16"
        # Handle both en-dash (–) and hyphen (-)
        range_match = re.match(r'^(\d+)[–-](\d+)$', part)
        single_match = re.match(r'^(\d+)$', part)
        
        if range_match:
            start = int(range_match.group(1))
            end_str = range_match.group(2)
            end = int(end_str)
            
            # Handle abbreviated ranges like "112–13" (means 112-113)
            # or "161–62" (means 161-162)
            if end < start:
                # It's an abbreviation - reconstruct full end number
                start_str = str(start)
                # Take the last N digits of start and replace with end
                # e.g., "112" -> last 2 digits "12" -> "13" -> "113"
                # e.g., "161" -> last 2 digits "61" -> "62" -> "162"
                if len(end_str) < len(start_str):
                    # Abbreviated: reconstruct
                    prefix_len = len(start_str) - len(end_str)
                    prefix = start_str[:prefix_len]
                    end = int(prefix + end_str)
            
            pages.append((start, end))
        
        elif single_match:
            page_num = int(single_match.group(1))
            pages.append((page_num, None))
    
    return pages


def normalize_document_name(name: str) -> str:
    """
    Normalize document name for matching.
    
    Converts:
    - "Venona New York KGB 1943" -> "newyorkkgb1943"
    - "Venona_New_York_KGB_1943.pdf" -> "newyorkkgb1943"
    - "Venona San Francisco KGB" -> "sanfranciscokgb"
    """
    # Remove file extension first (before removing prefix)
    name = re.sub(r'\.(pdf|txt)$', '', name, flags=re.IGNORECASE)
    
    # Remove "Venona" or "Vassiliev" prefix
    # Handle both spaces and underscores after prefix
    # Handle both straight and curly apostrophes: 's or 's
    name = re.sub(r"^(Venona|Vassiliev(?:['']s)?)[_\s]+", '', name, flags=re.IGNORECASE)
    
    # Normalize year ranges (e.g., "1941–42" -> "1941-1942")
    # Handle en-dash, hyphen, and abbreviated years
    year_range_match = re.search(r'(\d{4})[–-](\d{2,4})', name)
    if year_range_match:
        start_year = year_range_match.group(1)
        end_part = year_range_match.group(2)
        if len(end_part) == 2:
            # Abbreviated: "1941–42" -> "1941-1942"
            end_year = start_year[:2] + end_part
        else:
            end_year = end_part
        name = name.replace(year_range_match.group(0), f"{start_year}-{end_year}")
    
    # Extract year if present (e.g., "1943" or "1941-1942")
    year_match = re.search(r'(\d{4}(?:-\d{4})?)', name)
    year_part = year_match.group(1) if year_match else None
    
    # Remove year from name for matching
    if year_part:
        name = re.sub(r'\s*\d{4}(?:-\d{4})?\s*', '', name)
    
    # Normalize: lowercase, remove punctuation and underscores, collapse whitespace
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation (but keep alphanumeric)
    name = re.sub(r'[_\s]+', '', name)  # Remove underscores and whitespace
    
    # Add year back if present
    if year_part:
        name = name + year_part
    
    return name


def build_citation_to_document_map(cur, collection_slug: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Build a mapping from normalized citation sources to document IDs.
    
    Returns dict mapping normalized citation -> list of (document_id, document_name) tuples.
    """
    # Get all documents in collection
    cur.execute("""
        SELECT d.id, d.source_name, d.volume
        FROM documents d
        INNER JOIN collections c ON d.collection_id = c.id
        WHERE c.slug = %s
    """, (collection_slug,))
    
    documents = cur.fetchall()
    
    # Build mapping
    citation_map: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    
    for doc_id, source_name, volume in documents:
        # Normalize document name
        normalized = normalize_document_name(source_name)
        
        # Also try with volume if present
        if volume:
            normalized_with_volume = normalize_document_name(f"{source_name} {volume}")
            citation_map[normalized_with_volume].append((doc_id, source_name))
        
        citation_map[normalized].append((doc_id, source_name))
    
    return citation_map


def find_documents_for_citation(
    cur,
    citation: CitationLocation
) -> List[Tuple[int, str]]:
    """
    Find documents in database that match the citation source and year/volume.
    
    Uses normalized name matching to avoid false positives.
    
    Returns list of (document_id, document_name) tuples.
    """
    # Map citation source to collection
    collection_slug = None
    citation_source_normalized = None
    
    if citation.source.startswith("Venona"):
        collection_slug = "venona"
        # Build full citation source with year if available
        full_source = citation.source
        if citation.year_range:
            # Year might already be in source, or we need to add it
            if citation.year_range not in full_source:
                full_source = f"{citation.source} {citation.year_range}"
        # Normalize citation source (e.g., "Venona New York KGB 1943" -> "newyorkkgb1943")
        citation_source_normalized = normalize_document_name(full_source)
    
    elif citation.source.startswith("Vassiliev"):
        collection_slug = "vassiliev"
        citation_source_normalized = normalize_document_name(citation.source)
    
    if not collection_slug or not citation_source_normalized:
        return []
    
    # Build mapping for this collection (could be cached, but for now build on demand)
    citation_map = build_citation_to_document_map(cur, collection_slug)
    
    # Try exact match first
    if citation_source_normalized in citation_map:
        return citation_map[citation_source_normalized]
    
    # Try without year if year was included
    if citation.year_range:
        citation_source_no_year = normalize_document_name(citation.source)
        if citation_source_no_year in citation_map:
            # Filter by year if possible
            matching_docs = []
            # Normalize year range for comparison (1941–42 -> 1941-1942)
            year_normalized = citation.year_range.replace('–', '-')
            if len(year_normalized.split('-')) == 2:
                start, end = year_normalized.split('-')
                if len(end) == 2:
                    year_normalized = f"{start}-{start[:2]}{end}"
            
            for doc_id, doc_name in citation_map[citation_source_no_year]:
                # Check if document name contains the year
                doc_normalized = normalize_document_name(doc_name)
                # Try various year formats
                if (year_normalized in doc_normalized or 
                    citation.year_range.replace('–', '-') in doc_normalized or
                    citation.year_range in doc_normalized):
                    matching_docs.append((doc_id, doc_name))
            if matching_docs:
                return matching_docs
            # If no year match, return all (might be wrong, but better than nothing)
            return citation_map[citation_source_no_year]
    
    # Fallback: try fuzzy matching with word boundaries
    # Extract key words from citation (e.g., "New York KGB" from "Venona New York KGB 1943")
    # Remove "Venona" prefix first
    source_without_prefix = re.sub(r'^Venona\s+', '', citation.source, flags=re.IGNORECASE)
    citation_words = re.findall(r'\b[A-Z][a-z]+\b', source_without_prefix)
    if len(citation_words) >= 2:
        # Try to match documents containing these key words in order
        # Build pattern that requires words to appear in order
        key_phrase = ' '.join(citation_words).lower()
        
        # More precise: require all words to be present
        words_pattern = '.*'.join([re.escape(w.lower()) for w in citation_words])
        
        cur.execute("""
            SELECT d.id, d.source_name, d.volume
            FROM documents d
            INNER JOIN collections c ON d.collection_id = c.id
            WHERE c.slug = %s
            AND LOWER(d.source_name) ~ %s
        """, (collection_slug, words_pattern))
        
        results = cur.fetchall()
        if results:
            return [(row[0], f"{row[1]}" + (f" ({row[2]})" if row[2] else "")) for row in results]
    
    return []


def find_pages_for_citation(
    cur,
    document_id: int,
    citation: CitationLocation
) -> List[int]:
    """
    Find page IDs in the document that match the citation page numbers.
    
    Returns list of page IDs.
    """
    page_ids = []
    
    # Get all pages for this document
    cur.execute("""
        SELECT id, pdf_page_number, logical_page_label
        FROM pages
        WHERE document_id = %s
        ORDER BY pdf_page_number NULLS LAST, page_seq
    """, (document_id,))
    
    pages = cur.fetchall()
    
    # Match pages based on citation page numbers
    for page_start, page_end in citation.pages:
        if page_end is None:
            # Single page
            # Try to match by pdf_page_number or logical_page_label
            for page_id, pdf_page_num, logical_label in pages:
                if pdf_page_num == page_start:
                    page_ids.append(page_id)
                elif logical_label and str(page_start) in logical_label:
                    page_ids.append(page_id)
        else:
            # Page range
            for page_id, pdf_page_num, logical_label in pages:
                if pdf_page_num and page_start <= pdf_page_num <= page_end:
                    page_ids.append(page_id)
    
    return list(set(page_ids))  # Remove duplicates


def find_chunks_with_entity(
    cur,
    entity_id: int,
    page_ids: List[int]
) -> Tuple[List[int], int]:
    """
    Find chunks that:
    1. Are associated with the given pages (via chunk_pages)
    2. Have document_id in chunk_metadata matching the pages' document_id
    3. Contain mentions of the entity (via entity_mentions)
    
    Returns (list of chunk_ids, total_chunks_checked).
    """
    if not page_ids:
        return [], 0
    
    # Get document_ids for these pages
    cur.execute("""
        SELECT DISTINCT document_id
        FROM pages
        WHERE id = ANY(%s)
    """, (page_ids,))
    
    document_ids = [row[0] for row in cur.fetchall()]
    
    if not document_ids:
        return [], 0
    
    # Get chunks associated with these pages via chunk_pages
    # AND verify document_id matches via chunk_metadata
    cur.execute("""
        SELECT DISTINCT c.id
        FROM chunks c
        INNER JOIN chunk_pages cp ON c.id = cp.chunk_id
        INNER JOIN chunk_metadata cm ON c.id = cm.chunk_id
        WHERE cp.page_id = ANY(%s)
        AND cm.document_id = ANY(%s)
    """, (page_ids, document_ids))
    
    all_chunk_ids = [row[0] for row in cur.fetchall()]
    
    if not all_chunk_ids:
        return [], 0
    
    # Check which chunks have entity mentions
    cur.execute("""
        SELECT DISTINCT chunk_id
        FROM entity_mentions
        WHERE entity_id = %s
        AND chunk_id = ANY(%s)
    """, (entity_id, all_chunk_ids))
    
    chunks_with_entity = [row[0] for row in cur.fetchall()]
    
    return chunks_with_entity, len(all_chunk_ids)


def validate_entity_mentions_from_citations(
    conn,
    entity_id: Optional[int] = None,
    entity_name: Optional[str] = None,
    entry_key: Optional[str] = None
) -> List[ValidationResult]:
    """
    Validate entity mentions using citation information from concordance entries.
    
    Args:
        entity_id: Entity ID to validate
        entity_name: Entity canonical name to validate
        entry_key: Concordance entry key to validate
    
    Returns list of ValidationResult objects.
    """
    results = []
    
    with conn.cursor() as cur:
        # Get entity_id if not provided
        if not entity_id:
            if entity_name:
                cur.execute("SELECT id FROM entities WHERE canonical_name = %s LIMIT 1", (entity_name,))
                row = cur.fetchone()
                if row:
                    entity_id = row[0]
                else:
                    print(f"Entity not found: {entity_name}")
                    return []
            elif entry_key:
                # Find entity via entry key
                cur.execute("""
                    SELECT e.id
                    FROM entities e
                    INNER JOIN concordance_entries ce ON e.entry_id = ce.id
                    WHERE ce.entry_key = %s
                    LIMIT 1
                """, (entry_key,))
                row = cur.fetchone()
                if row:
                    entity_id = row[0]
                else:
                    print(f"Entity not found for entry key: {entry_key}")
                    return []
            else:
                print("Must provide entity_id, entity_name, or entry_key")
                return []
        
        # Get entity info
        cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
        row = cur.fetchone()
        if not row:
            print(f"Entity not found: {entity_id}")
            return []
        entity_name = row[0]
        
        print(f"Validating entity: {entity_name} (ID: {entity_id})")
        
        # Get all citations for this entity from concordance
        cur.execute("""
            SELECT ec.citation_text
            FROM entity_citations ec
            INNER JOIN entities e ON ec.entity_id = e.id
            WHERE e.id = %s
            ORDER BY ec.id
        """, (entity_id,))
        
        citations = [row[0] for row in cur.fetchall()]
        
        if not citations:
            print(f"No citations found for entity {entity_name}")
            return []
        
        print(f"Found {len(citations)} citation(s)")
        
        # Parse each citation and validate
        for citation_text in citations:
            locations = parse_citation_text(citation_text)
            
            for location in locations:
                # Find matching documents
                documents = find_documents_for_citation(cur, location)
                
                if not documents:
                    results.append(ValidationResult(
                        citation_location=location,
                        document_id=None,
                        document_name=None,
                        matched_pages=[],
                        chunks_with_entity=[],
                        chunks_checked=0,
                        validation_status="no_document"
                    ))
                    continue
                
                # For each document, find pages and validate
                for doc_id, doc_name in documents:
                    page_ids = find_pages_for_citation(cur, doc_id, location)
                    
                    if not page_ids:
                        results.append(ValidationResult(
                            citation_location=location,
                            document_id=doc_id,
                            document_name=doc_name,
                            matched_pages=[],
                            chunks_with_entity=[],
                            chunks_checked=0,
                            validation_status="no_pages"
                        ))
                        continue
                    
                    # Find chunks with entity mentions
                    chunks_with_entity, chunks_checked = find_chunks_with_entity(
                        cur, entity_id, page_ids
                    )
                    
                    if chunks_with_entity:
                        status = "validated"
                    else:
                        status = "no_mentions"
                    
                    results.append(ValidationResult(
                        citation_location=location,
                        document_id=doc_id,
                        document_name=doc_name,
                        matched_pages=page_ids,
                        chunks_with_entity=chunks_with_entity,
                        chunks_checked=chunks_checked,
                        validation_status=status
                    ))
    
    return results


def print_validation_results(results: List[ValidationResult]):
    """Print validation results in a readable format."""
    if not results:
        print("No validation results to display")
        return
    
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS")
    print("=" * 100)
    
    # Group by status
    by_status = defaultdict(list)
    for result in results:
        by_status[result.validation_status].append(result)
    
    print(f"\nSummary:")
    print(f"  Validated (mentions found): {len(by_status['validated'])}")
    print(f"  No mentions: {len(by_status['no_mentions'])}")
    print(f"  No pages found: {len(by_status['no_pages'])}")
    print(f"  No document found: {len(by_status['no_document'])}")
    
    print("\n" + "-" * 100)
    
    for result in results:
        print(f"\nCitation: {result.citation_location.citation_text}")
        print(f"  Source: {result.citation_location.source}")
        if result.citation_location.year_range:
            print(f"  Year/Volume: {result.citation_location.year_range}")
        print(f"  Pages: {result.citation_location.pages}")
        print(f"  Status: {result.validation_status}")
        
        if result.document_id:
            print(f"  Document: {result.document_name} (ID: {result.document_id})")
            print(f"  Matched Pages: {len(result.matched_pages)} page(s)")
            print(f"  Chunks Checked: {result.chunks_checked}")
            print(f"  Chunks with Entity: {len(result.chunks_with_entity)}")
            
            if result.chunks_with_entity:
                print(f"    Chunk IDs: {result.chunks_with_entity[:10]}{'...' if len(result.chunks_with_entity) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate entity mentions using citation information from concordance entries"
    )
    parser.add_argument("--entity-id", type=int, help="Entity ID to validate")
    parser.add_argument("--entity-name", type=str, help="Entity canonical name to validate")
    parser.add_argument("--entry-key", type=str, help="Concordance entry key to validate")
    parser.add_argument("--db-url", type=str, help="Database URL (default: from DATABASE_URL env var)")
    
    args = parser.parse_args()
    
    if not any([args.entity_id, args.entity_name, args.entry_key]):
        parser.error("Must provide --entity-id, --entity-name, or --entry-key")
    
    # Connect to database
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Must provide --db-url or set DATABASE_URL environment variable")
    
    conn = psycopg2.connect(db_url)
    
    try:
        results = validate_entity_mentions_from_citations(
            conn,
            entity_id=args.entity_id,
            entity_name=args.entity_name,
            entry_key=args.entry_key
        )
        
        print_validation_results(results)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
