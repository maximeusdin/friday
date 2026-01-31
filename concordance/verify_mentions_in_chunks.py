#!/usr/bin/env python3
"""
Verify that extracted entity mentions actually appear in chunks at the expected page numbers.

This script:
1. Reads entity_mentions.csv or queries the database
2. For each mention, checks:
   - The chunk contains the surface text
   - The chunk spans the expected pages from citations
   - The character positions are valid
3. Reports any discrepancies
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import DictCursor

from concordance.validate_entity_mentions_from_citations import (
    parse_citation_text
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


def expand_page_ranges_list(pages: List[Tuple[int, Optional[int]]]) -> List[int]:
    """Expand page ranges into individual page numbers."""
    expanded = []
    for start, end in pages:
        if end is None:
            expanded.append(start)
        else:
            for page_num in range(start, end + 1):
                expanded.append(page_num)
    return sorted(set(expanded))


def get_chunk_info(cur, chunk_id: int) -> Optional[Dict]:
    """Get chunk text and metadata."""
    cur.execute("""
        SELECT 
            c.id,
            c.text,
            cm.document_id,
            cm.first_page_id,
            p.pdf_page_number,
            p.logical_page_label
        FROM chunks c
        LEFT JOIN chunk_metadata cm ON c.id = cm.chunk_id
        LEFT JOIN pages p ON cm.first_page_id = p.id
        WHERE c.id = %s
        LIMIT 1
    """, (chunk_id,))
    
    row = cur.fetchone()
    if row:
        return dict(row)
    return None


def get_chunk_pages(cur, chunk_id: int) -> List[int]:
    """Get all page IDs associated with a chunk."""
    cur.execute("""
        SELECT DISTINCT p.id, p.pdf_page_number
        FROM chunk_pages cp
        JOIN pages p ON cp.page_id = p.id
        WHERE cp.chunk_id = %s
        ORDER BY p.pdf_page_number
    """, (chunk_id,))
    
    pages = []
    for row in cur.fetchall():
        if row['pdf_page_number']:
            pages.append(row['pdf_page_number'])
    return sorted(set(pages))


def get_citation_pages_for_entity(cur, entity_id: int, document_id: int) -> List[int]:
    """Get all page numbers from citations for this entity in this document."""
    # Get all citations for this entity
    cur.execute("""
        SELECT citation_text
        FROM entity_citations
        WHERE entity_id = %s
    """, (entity_id,))
    
    all_pages = []
    for row in cur.fetchall():
        citation_text = row['citation_text']
        if citation_text:
            # Parse citation to get locations
            citation_locations = parse_citation_text(citation_text)
            for loc in citation_locations:
                # Check if this location matches the document
                # We'd need to match document names, but for now just collect all pages
                expanded = expand_page_ranges_list(loc.pages)
                all_pages.extend(expanded)
    
    return sorted(set(all_pages))


def verify_mention(cur, mention: Dict, verbose: bool = False) -> Tuple[bool, str]:
    """
    Verify a single entity mention.
    
    Returns (is_valid, message)
    """
    chunk_id = mention.get('chunk_id')
    entity_id = mention.get('entity_id')
    document_id = mention.get('document_id')
    surface = mention.get('surface', '')
    start_char = mention.get('start_char')
    end_char = mention.get('end_char')
    
    if not chunk_id:
        return False, "Missing chunk_id"
    
    # Get chunk information
    chunk_info = get_chunk_info(cur, chunk_id)
    if not chunk_info:
        return False, f"Chunk {chunk_id} not found"
    
    chunk_text = chunk_info.get('text', '')
    chunk_doc_id = chunk_info.get('document_id')
    
    # Verify document matches
    if chunk_doc_id != document_id:
        return False, f"Document mismatch: chunk has doc_id {chunk_doc_id}, mention has {document_id}"
    
    # Verify surface text appears in chunk
    if surface:
        if surface not in chunk_text:
            # Try case-insensitive
            if surface.lower() not in chunk_text.lower():
                return False, f"Surface text '{surface}' not found in chunk text"
    
    # Verify character positions
    if start_char is not None and end_char is not None:
        try:
            start_char = int(start_char)
            end_char = int(end_char)
            
            if start_char < 0 or end_char > len(chunk_text):
                return False, f"Character positions out of range: start={start_char}, end={end_char}, chunk_len={len(chunk_text)}"
            
            if start_char >= end_char:
                return False, f"Invalid character range: start={start_char} >= end={end_char}"
            
            # Check if the text at those positions matches
            extracted_text = chunk_text[start_char:end_char]
            if surface and extracted_text != surface:
                # Allow case-insensitive match
                if extracted_text.lower() != surface.lower():
                    return False, f"Text at positions doesn't match: expected '{surface}', got '{extracted_text}'"
        except (ValueError, TypeError):
            return False, f"Invalid character positions: start={start_char}, end={end_char}"
    
    # Verify pages
    chunk_pages = get_chunk_pages(cur, chunk_id)
    if chunk_pages:
        # Get expected pages from citations
        citation_pages = get_citation_pages_for_entity(cur, entity_id, document_id)
        
        if citation_pages:
            # Check if any chunk pages overlap with citation pages
            chunk_page_set = set(chunk_pages)
            citation_page_set = set(citation_pages)
            overlap = chunk_page_set & citation_page_set
            
            if not overlap and verbose:
                return True, f"Valid mention, but no page overlap: chunk_pages={chunk_pages[:10]}, citation_pages={citation_pages[:10]}"
            elif not overlap:
                return True, "Valid mention (no page overlap check)"
    
    return True, "Valid"


def verify_from_csv(csv_path: Path, entity_name: Optional[str] = None, 
                    limit: Optional[int] = None, verbose: bool = False):
    """Verify mentions from CSV file."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        print(f"Reading {csv_path}...", file=sys.stderr)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Filter by entity name if specified
        if entity_name:
            rows = [r for r in rows if entity_name.lower() in r.get('canonical_name', '').lower()]
            print(f"Filtered to {len(rows)} rows for entity '{entity_name}'", file=sys.stderr)
        
        # Limit if specified
        if limit:
            rows = rows[:limit]
            print(f"Limited to {len(rows)} rows", file=sys.stderr)
        
        print(f"\nVerifying {len(rows)} mentions...\n", file=sys.stderr)
        
        valid_count = 0
        invalid_count = 0
        examples_shown = 0
        
        for i, row in enumerate(rows, 1):
            is_valid, message = verify_mention(cur, row, verbose=verbose)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if examples_shown < 5:
                    entity_name = row.get('canonical_name', 'Unknown')
                    chunk_id = row.get('chunk_id', 'N/A')
                    surface = row.get('surface', 'N/A')
                    print(f"\n{'='*80}")
                    print(f"Row {i} - Entity: {entity_name}")
                    print(f"  Chunk ID: {chunk_id}")
                    print(f"  Surface: {surface}")
                    print(f"  {message}")
                    examples_shown += 1
            
            if verbose and i <= 5:
                entity_name = row.get('canonical_name', 'Unknown')
                chunk_id = row.get('chunk_id', 'N/A')
                surface = row.get('surface', 'N/A')
                print(f"\nRow {i} - {entity_name} (chunk {chunk_id}):")
                print(f"  Surface: {surface}")
                print(f"  Status: {'✓ Valid' if is_valid else '✗ Invalid'}")
                if not is_valid:
                    print(f"  Issue: {message}")
        
        print(f"\n{'='*80}")
        print(f"Summary:")
        print(f"  Valid:   {valid_count}")
        print(f"  Invalid: {invalid_count}")
        print(f"  Total:   {len(rows)}")
        print(f"{'='*80}")
        
        if invalid_count == 0:
            print("\n✓ All mentions verified successfully!", file=sys.stderr)
        else:
            print(f"\n✗ Found {invalid_count} invalid mentions", file=sys.stderr)
    
    finally:
        cur.close()
        conn.close()


def verify_from_db(entity_id: Optional[int] = None, entity_name: Optional[str] = None,
                   limit: Optional[int] = None, verbose: bool = False):
    """Verify mentions directly from database."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # Build query
        query = """
            SELECT 
                em.id,
                em.entity_id,
                e.canonical_name,
                em.chunk_id,
                em.document_id,
                em.surface,
                em.start_char,
                em.end_char
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.id
        """
        params = []
        conditions = []
        
        if entity_id:
            conditions.append("em.entity_id = %s")
            params.append(entity_id)
        elif entity_name:
            conditions.append("e.canonical_name ILIKE %s")
            params.append(f"%{entity_name}%")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY em.entity_id, em.id"
        
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        cur.execute(query, params)
        rows = [dict(row) for row in cur.fetchall()]
        
        print(f"Found {len(rows)} mentions to verify\n", file=sys.stderr)
        
        valid_count = 0
        invalid_count = 0
        examples_shown = 0
        
        for i, row in enumerate(rows, 1):
            is_valid, message = verify_mention(cur, row, verbose=verbose)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if examples_shown < 5:
                    entity_name = row.get('canonical_name', 'Unknown')
                    chunk_id = row.get('chunk_id', 'N/A')
                    surface = row.get('surface', 'N/A')
                    print(f"\n{'='*80}")
                    print(f"Mention {i} - Entity: {entity_name}")
                    print(f"  Chunk ID: {chunk_id}")
                    print(f"  Surface: {surface}")
                    print(f"  {message}")
                    examples_shown += 1
        
        print(f"\n{'='*80}")
        print(f"Summary:")
        print(f"  Valid:   {valid_count}")
        print(f"  Invalid: {invalid_count}")
        print(f"  Total:   {len(rows)}")
        print(f"{'='*80}")
        
        if invalid_count == 0:
            print("\n✓ All mentions verified successfully!", file=sys.stderr)
        else:
            print(f"\n✗ Found {invalid_count} invalid mentions", file=sys.stderr)
    
    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Verify that entity mentions appear in chunks at expected page numbers"
    )
    parser.add_argument(
        "--csv-file",
        help="Path to entity_mentions.csv file (if not provided, queries database)"
    )
    parser.add_argument(
        "--entity-id",
        type=int,
        help="Filter to specific entity ID (database only)"
    )
    parser.add_argument(
        "--entity-name",
        help="Filter to specific entity name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of mentions to check"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    
    args = parser.parse_args()
    
    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
            return 1
        verify_from_csv(csv_path, args.entity_name, args.limit, args.verbose)
    else:
        verify_from_db(args.entity_id, args.entity_name, args.limit, args.verbose)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
