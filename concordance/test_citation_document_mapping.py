#!/usr/bin/env python3
"""
Test citation to document mapping to verify normalization works correctly.

This script tests that citation formats from concordance entries correctly
map to document source_names in the database.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.validate_entity_mentions_from_citations import (
    normalize_document_name,
    parse_citation_text,
    build_citation_to_document_map
)
import psycopg2


def test_normalization():
    """Test document name normalization."""
    test_cases = [
        # (citation_format, document_name, should_match)
        ("Venona New York KGB 1943", "Venona_New_York_KGB_1943.pdf", True),
        ("Venona New York KGB 1941–42", "Venona_New_York_KGB_1941-42.pdf", True),
        ("Venona San Francisco KGB", "Venona_San_Francisco_KGB.pdf", True),
        ("Venona Special Studies", "Venona_Special_Studies.pdf", True),
        ("Vassiliev Yellow Notebook", "Vassiliev_Yellow_Notebook.pdf", True),
        ("Venona New York KGB 1943", "Venona_Bogota_KGB.pdf", False),
        ("Venona San Francisco KGB", "Venona_London_GRU.pdf", False),
    ]
    
    print("Testing document name normalization:")
    print("=" * 80)
    
    for citation, doc_name, should_match in test_cases:
        citation_norm = normalize_document_name(citation)
        doc_norm = normalize_document_name(doc_name)
        
        matches = citation_norm == doc_norm
        status = "PASS" if (matches == should_match) else "FAIL"
        
        print(f"{status} Citation: {citation}")
        print(f"    Normalized: {citation_norm}")
        print(f"  Document: {doc_name}")
        print(f"    Normalized: {doc_norm}")
        print(f"  Match: {matches} (expected: {should_match})")
        print()


def test_citation_parsing():
    """Test citation text parsing."""
    citation_text = """Venona New York KGB 1941–42, 16, 74–75; Venona New York KGB 1943, 112–13, 161–62, 221; Venona San Francisco KGB, 144"""
    
    print("Testing citation parsing:")
    print("=" * 80)
    print(f"Input: {citation_text}")
    print()
    
    locations = parse_citation_text(citation_text)
    
    for i, loc in enumerate(locations, 1):
        print(f"Location {i}:")
        print(f"  Source: {loc.source}")
        print(f"  Year: {loc.year_range}")
        print(f"  Pages: {loc.pages}")
        print()


def test_document_mapping():
    """Test mapping citations to actual documents in database."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable not set")
        return
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    try:
        print("Testing document mapping:")
        print("=" * 80)
        
        # Build mapping for venona collection
        citation_map = build_citation_to_document_map(cur, "venona")
        
        print(f"Built mapping with {len(citation_map)} normalized document names")
        print()
        
        # Test specific citations
        test_citations = [
            "Venona New York KGB 1943",
            "Venona New York KGB 1941–42",
            "Venona San Francisco KGB",
            "Venona Special Studies",
        ]
        
        for citation in test_citations:
            normalized = normalize_document_name(citation)
            print(f"Citation: {citation}")
            print(f"  Normalized: {normalized}")
            
            if normalized in citation_map:
                docs = citation_map[normalized]
                print(f"  Found {len(docs)} matching document(s):")
                for doc_id, doc_name in docs[:3]:  # Show first 3
                    print(f"    - {doc_name} (ID: {doc_id})")
                if len(docs) > 3:
                    print(f"    ... and {len(docs) - 3} more")
            else:
                print(f"  No exact match found")
                # Try partial match
                matching_keys = [k for k in citation_map.keys() if normalized in k or k in normalized]
                if matching_keys:
                    print(f"  Found {len(matching_keys)} partial matches")
                    for key in matching_keys[:3]:
                        print(f"    - {key}: {len(citation_map[key])} document(s)")
            print()
    
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test citation to document mapping")
    parser.add_argument("--test-normalization", action="store_true", help="Test normalization function")
    parser.add_argument("--test-parsing", action="store_true", help="Test citation parsing")
    parser.add_argument("--test-mapping", action="store_true", help="Test document mapping")
    
    args = parser.parse_args()
    
    if args.test_normalization:
        test_normalization()
    elif args.test_parsing:
        test_citation_parsing()
    elif args.test_mapping:
        test_document_mapping()
    else:
        # Run all tests
        test_normalization()
        print("\n" + "=" * 80 + "\n")
        test_citation_parsing()
        print("\n" + "=" * 80 + "\n")
        test_document_mapping()
