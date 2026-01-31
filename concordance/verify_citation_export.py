#!/usr/bin/env python3
"""
Verify that citation_page_lists in entity_mentions.csv correctly match citation_texts.

This script:
1. Reads entity_mentions.csv
2. For each row, parses citation_texts to extract pages
3. Compares with citation_page_lists to verify they match
4. Shows examples and any mismatches
"""

import sys
import csv
from pathlib import Path
from typing import List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.validate_entity_mentions_from_citations import parse_citation_text


def expand_page_ranges(pages: List[Tuple[int, Optional[int]]]) -> List[int]:
    """Expand page ranges into individual page numbers."""
    expanded = []
    for start, end in pages:
        if end is None:
            expanded.append(start)
        else:
            for page_num in range(start, end + 1):
                expanded.append(page_num)
    return sorted(set(expanded))


def parse_citation_pages_by_document(citation_text: str) -> List[Tuple[str, List[int]]]:
    """Parse citation text and return list of (document_name, page_list) tuples."""
    if not citation_text:
        return []
    
    citation_locations = parse_citation_text(citation_text)
    result = []
    for loc in citation_locations:
        expanded_pages = expand_page_ranges(loc.pages)
        if expanded_pages:
            result.append((loc.source, expanded_pages))
    return result


def parse_page_list_string(page_list_str: str) -> List[int]:
    """
    Parse a page list string like "{51,52,53,54,58,67,79,87,181}" into a list of integers.
    """
    if not page_list_str or page_list_str.strip() == '':
        return []
    
    # Remove curly braces and split by comma
    cleaned = page_list_str.strip().strip('{}')
    if not cleaned:
        return []
    
    pages = []
    for part in cleaned.split(','):
        part = part.strip()
        if part:
            try:
                pages.append(int(part))
            except ValueError:
                pass
    
    return sorted(set(pages))


def verify_row(row: dict, verbose: bool = False) -> Tuple[bool, str]:
    """
    Verify that citation_page_lists matches what we'd extract from citation_texts.
    
    Returns (is_valid, message)
    """
    citation_texts = row.get('citation_texts', '')
    citation_page_lists = row.get('citation_page_lists', '')
    
    if not citation_texts:
        return True, "No citation_texts to verify"
    
    # Split citation_texts by ' | ' to get individual citation records
    citation_text_parts = citation_texts.split(' | ')
    
    # Parse each citation text to get expected page lists
    expected_page_lists = []
    for citation_text_part in citation_text_parts:
        doc_pages = parse_citation_pages_by_document(citation_text_part)
        formatted_pages = []
        for doc_name, pages in doc_pages:
            if pages:
                pages_str = ','.join(str(p) for p in sorted(pages))
                formatted_pages.append(f"{{{pages_str}}}")
        if formatted_pages:
            expected_page_lists.append(' | '.join(formatted_pages))
    
    expected_combined = ' | '.join(expected_page_lists) if expected_page_lists else None
    
    # Compare
    if expected_combined is None and (not citation_page_lists or citation_page_lists.strip() == ''):
        return True, "Both empty"
    
    if expected_combined != citation_page_lists:
        return False, f"Mismatch:\n  Expected: {expected_combined}\n  Got:      {citation_page_lists}"
    
    return True, "Match"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify citation_page_lists in entity_mentions.csv"
    )
    parser.add_argument(
        "--csv-file",
        default="concordance_export/entity_mentions.csv",
        help="Path to entity_mentions.csv file"
    )
    parser.add_argument(
        "--entity-name",
        help="Filter to specific entity name (e.g., 'Vladimir Pravdin')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of rows to check"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information for each row"
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of example rows to show (default: 3)"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1
    
    print(f"Reading {csv_path}...", file=sys.stderr)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter by entity name if specified
    if args.entity_name:
        rows = [r for r in rows if args.entity_name.lower() in r.get('canonical_name', '').lower()]
        print(f"Filtered to {len(rows)} rows for entity '{args.entity_name}'", file=sys.stderr)
    
    # Limit if specified
    if args.limit:
        rows = rows[:args.limit]
        print(f"Limited to {len(rows)} rows", file=sys.stderr)
    
    print(f"\nVerifying {len(rows)} rows...\n", file=sys.stderr)
    
    valid_count = 0
    invalid_count = 0
    examples_shown = 0
    
    for i, row in enumerate(rows, 1):
        is_valid, message = verify_row(row, verbose=args.verbose)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            if examples_shown < args.show_examples:
                entity_name = row.get('canonical_name', 'Unknown')
                print(f"\n{'='*80}")
                print(f"Row {i} - Entity: {entity_name}")
                print(f"{'='*80}")
                print(f"Citation texts:\n{row.get('citation_texts', '')[:500]}")
                print(f"\n{message}")
                examples_shown += 1
        
        if args.verbose and i <= args.show_examples:
            entity_name = row.get('canonical_name', 'Unknown')
            citation_texts = row.get('citation_texts', '')[:200]
            citation_page_lists = row.get('citation_page_lists', '')[:200]
            print(f"\nRow {i} - {entity_name}:")
            print(f"  Citation texts: {citation_texts}...")
            print(f"  Citation page lists: {citation_page_lists}...")
            print(f"  Status: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Valid:   {valid_count}")
    print(f"  Invalid: {invalid_count}")
    print(f"  Total:   {len(rows)}")
    print(f"{'='*80}")
    
    if invalid_count == 0:
        print("\n✓ All rows verified successfully!", file=sys.stderr)
        return 0
    else:
        print(f"\n✗ Found {invalid_count} mismatches", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
