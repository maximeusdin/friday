#!/usr/bin/env python3
"""
Update existing entity_citations.page_list with complete page numbers.

The ingest script's best_effort_parse_citation_fields was too simple and didn't
expand page ranges. This script re-parses citation_text using the proper parser
and updates page_list to include all pages (expanding ranges).

Usage:
    python concordance/update_citation_page_lists.py --dry-run
    python concordance/update_citation_page_lists.py
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import DictCursor

# Import the proper citation parser
from concordance.validate_entity_mentions_from_citations import (
    parse_citation_text,
    parse_page_numbers
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


def expand_page_ranges(pages: List[Tuple[int, Optional[int]]]) -> List[int]:
    """
    Expand page ranges into individual page numbers.
    
    Example: [(230, 231), (236, None), (314, None)] -> [230, 231, 236, 314]
    """
    expanded = []
    for start, end in pages:
        if end is None:
            expanded.append(start)
        else:
            # Expand range: start to end (inclusive)
            for page_num in range(start, end + 1):
                expanded.append(page_num)
    return sorted(set(expanded))  # Remove duplicates and sort


def update_citation_page_lists(conn, dry_run: bool = False):
    """
    Update page_list for all entity_citations by re-parsing citation_text.
    """
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # Get all citations
        cur.execute("""
            SELECT id, entity_id, citation_text, page_list
            FROM entity_citations
            WHERE citation_text IS NOT NULL
            ORDER BY id
        """)
        
        citations = cur.fetchall()
        print(f"Found {len(citations)} citations to process", file=sys.stderr)
        
        updated_count = 0
        unchanged_count = 0
        
        for citation in citations:
            citation_id = citation['id']
            citation_text = citation['citation_text']
            old_page_list = citation['page_list']
            
            # Parse citation text to get all page locations
            citation_locations = parse_citation_text(citation_text)
            
            # Collect all pages from all locations
            all_pages = []
            for loc in citation_locations:
                # Expand ranges to individual pages
                expanded = expand_page_ranges(loc.pages)
                all_pages.extend(expanded)
            
            # Remove duplicates and sort
            new_page_list = sorted(set(all_pages)) if all_pages else None
            
            # Compare with old page_list
            old_set = set(old_page_list) if old_page_list else set()
            new_set = set(new_page_list) if new_page_list else set()
            
            if old_set != new_set:
                # Update needed
                if not dry_run:
                    cur.execute("""
                        UPDATE entity_citations
                        SET page_list = %s
                        WHERE id = %s
                    """, (new_page_list, citation_id))
                
                updated_count += 1
                if len(new_set) > len(old_set):
                    added = new_set - old_set
                    print(f"  Citation {citation_id}: Adding pages {sorted(added)} "
                          f"(old: {len(old_set)} pages, new: {len(new_set)} pages)", file=sys.stderr)
            else:
                unchanged_count += 1
        
        if not dry_run:
            conn.commit()
            print(f"\nUpdated {updated_count} citations", file=sys.stderr)
        else:
            print(f"\nWould update {updated_count} citations", file=sys.stderr)
        
        print(f"Unchanged: {unchanged_count} citations", file=sys.stderr)
        
    finally:
        cur.close()


def main():
    parser = argparse.ArgumentParser(
        description="Update entity_citations.page_list with complete page numbers"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    
    args = parser.parse_args()
    
    conn = get_db_connection()
    try:
        update_citation_page_lists(conn, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
