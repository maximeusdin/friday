#!/usr/bin/env python3
"""
Example: Validate entity mentions for AKIM using citation information.

This demonstrates how to use the citation validation system with the example:
AKIM (cover name in Venona): Sergej Grigor'evich Luk'yanov. 
Venona New York KGB 1941–42, 16, 74–75; Venona New York KGB 1943, 112–13, 161–62, 221; 
Venona New York KGB 1944, 25–26, 175–76, 228–29, 234–36, 263–64, 353, 361–62, 394–95, 
442–43, 597–98, 630, 657–58, 670–71, 676–77, 710, 761; Venona New York KGB 1945, 
81–82, 102, 137, 143, 162, 165; Venona San Francisco KGB, 144; Venona Special Studies, 3–4, 93
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.validate_entity_mentions_from_citations import (
    parse_citation_text,
    CitationLocation,
    validate_entity_mentions_from_citations
)
import psycopg2


def test_akim_citation_parsing():
    """Test parsing the AKIM citation example."""
    citation_text = """Venona New York KGB 1941–42, 16, 74–75; Venona New York KGB 1943, 112–13, 161–62, 221; Venona New York KGB 1944, 25–26, 175–76, 228–29, 234–36, 263–64, 353, 361–62, 394–95, 442–43, 597–98, 630, 657–58, 670–71, 676–77, 710, 761; Venona New York KGB 1945, 81–82, 102, 137, 143, 162, 165; Venona San Francisco KGB, 144; Venona Special Studies, 3–4, 93"""
    
    print("Testing citation parsing for AKIM example:")
    print(f"Citation text: {citation_text[:100]}...")
    print()
    
    locations = parse_citation_text(citation_text)
    
    print(f"Parsed {len(locations)} citation location(s):\n")
    
    for i, loc in enumerate(locations, 1):
        print(f"Location {i}:")
        print(f"  Source: {loc.source}")
        print(f"  Year/Volume: {loc.year_range or 'N/A'}")
        print(f"  Pages: {loc.pages}")
        print(f"  Full text: {loc.citation_text}")
        print()
    
    # Expected results:
    # 1. Venona New York KGB 1941–42, pages: 16, 74–75
    # 2. Venona New York KGB 1943, pages: 112–13, 161–62, 221
    # 3. Venona New York KGB 1944, pages: 25–26, 175–76, 228–29, 234–36, 263–64, 353, 361–62, 394–95, 442–43, 597–98, 630, 657–58, 670–71, 676–77, 710, 761
    # 4. Venona New York KGB 1945, pages: 81–82, 102, 137, 143, 162, 165
    # 5. Venona San Francisco KGB, pages: 144
    # 6. Venona Special Studies, pages: 3–4, 93


def validate_akim_entity():
    """Validate AKIM entity mentions using citations."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable not set")
        return
    
    conn = psycopg2.connect(db_url)
    
    try:
        print("Validating AKIM entity mentions...")
        print("=" * 100)
        
        # Try to find AKIM entity
        # It might be stored as "AKIM" or "Sergej Grigor'evich Luk'yanov" (the referent)
        results = validate_entity_mentions_from_citations(
            conn,
            entity_name="AKIM"
        )
        
        if not results:
            # Try the referent name
            print("\nTrying referent name: Sergej Grigor'evich Luk'yanov")
            results = validate_entity_mentions_from_citations(
                conn,
                entity_name="Sergej Grigor'evich Luk'yanov"
            )
        
        if not results:
            # Try entry key
            print("\nTrying entry key: AKIM (cover name in Venona)")
            results = validate_entity_mentions_from_citations(
                conn,
                entry_key="AKIM (cover name in Venona)"
            )
        
        if results:
            from concordance.validate_entity_mentions_from_citations import print_validation_results
            print_validation_results(results)
        else:
            print("No results found. Entity may not exist in database yet.")
            print("Make sure the concordance has been ingested and entities have been created.")
    
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AKIM citation validation")
    parser.add_argument("--test-parsing", action="store_true", help="Test citation parsing only")
    parser.add_argument("--validate", action="store_true", help="Validate entity mentions in database")
    
    args = parser.parse_args()
    
    if args.test_parsing:
        test_akim_citation_parsing()
    elif args.validate:
        validate_akim_entity()
    else:
        # Run both by default
        test_akim_citation_parsing()
        print("\n" + "=" * 100 + "\n")
        validate_akim_entity()
