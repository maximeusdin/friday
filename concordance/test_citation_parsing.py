#!/usr/bin/env python3
"""Test citation parsing for export."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.validate_entity_mentions_from_citations import parse_citation_text

def expand_page_ranges(pages):
    """Expand page ranges into individual page numbers."""
    expanded = []
    for start, end in pages:
        if end is None:
            expanded.append(start)
        else:
            for page_num in range(start, end + 1):
                expanded.append(page_num)
    return sorted(set(expanded))

def parse_citation_pages_by_document(citation_text):
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

# Test with example from user
test_citation = "Vassiliev Black Notebook, 51–54, 58, 67, 79, 87, 181; Vassiliev White Notebook #1, 44, 56–57, 59–61, 64, 66, 69, 71–74, 77–79"

print("Testing citation parsing:")
print(f"Input: {test_citation}")
print()

doc_pages = parse_citation_pages_by_document(test_citation)

print("Parsed documents and pages:")
for doc_name, pages in doc_pages:
    print(f"  {doc_name}: {pages}")

print()
print("Formatted page lists:")
formatted = []
for doc_name, pages in doc_pages:
    if pages:
        pages_str = ','.join(str(p) for p in sorted(pages))
        formatted.append(f"{{{pages_str}}}")
print(' | '.join(formatted))
