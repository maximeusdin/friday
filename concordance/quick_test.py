#!/usr/bin/env python3
"""
Quick test of concordance parsing without PDF extraction.

Tests parsing logic directly on sample entry blocks.

Usage:
    # Run all sample test cases
    python concordance/quick_test.py
    
    # Parse a custom entry string
    python concordance/quick_test.py --entry "KALIBR [CALIBER and CALIBRE]: David Greenglass. Venona..."
    
    # Parse from stdin
    echo "Entry text here" | python concordance/quick_test.py --entry -
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.ingest_concordance_tab_aware import parse_entry_block, print_parsed


# Sample entry blocks from the test cases
SAMPLE_ENTRIES = [
    """KALIBR [CALIBER and CALIBRE] (cover name in Venona): David Greenglass. Venona New York KGB
1944, 602, 643, 714, 716, 729; Venona New York KGB 1945, 24; Venona Special Studies, 32, 79,
141, 154.""",
    
    """"Kalibr" (Russian original of a cover name in Vassiliev's notebooks): See "Caliber".""",
    
    """"Kaiser" [Kayzer] (cover name in Vassiliev's notebooks): Unidentified Soviet intelligence contact, friend of
Harold Glasser. Described as American Army captain in Italy in 1944, then working in the
Treasury Depatment in Washington, and appointed to the staff of the Allied Control Commission
in Austria. Formerly active in the Washington CPUSA network. Vassiliev White Notebook #3,
52.""",
    
    """Kalinin, Mikhail Ivanovich: Bolshevik leader and official Soviet head of state, 1919–46. Vassiliev Yellow
Notebook #4, 66.""",
    
    """Kalinin, ?: Soviet sailor and Soviet internal security source. Cover name in Venona: ELKIN. Venona San
Francisco KGB, 88, 262; Venona Special Studies, 100.""",
    
    """Kalinin, Tikhon Ivanovich: SGPC official. Venona San Francisco KGB, 77.""",
]


def parse_single_entry(entry_text: str, entry_num: int = 1):
    """Parse a single entry and display results."""
    print("=" * 100)
    print(f"PARSING ENTRY {entry_num}")
    print("=" * 100)
    print(f"\nInput text:\n{entry_text[:500]}{'...' if len(entry_text) > 500 else ''}")
    print("\n" + "=" * 100)
    
    try:
        pe = parse_entry_block(entry_text, entry_seq=entry_num)
        print_parsed(pe)
    except Exception as e:
        print(f"\nERROR parsing entry: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def quick_test():
    """Run quick parsing test on sample entries."""
    print("=" * 100)
    print("QUICK CONCORDANCE PARSING TEST")
    print("=" * 100)
    print()
    
    for i, entry_block in enumerate(SAMPLE_ENTRIES, 1):
        print(f"\n{'='*100}")
        print(f"ENTRY {i}")
        print('='*100)
        
        try:
            pe = parse_entry_block(entry_block, entry_seq=1)
            print_parsed(pe)
        except Exception as e:
            print(f"ERROR parsing entry {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("Quick test complete")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Quick test of concordance parsing without PDF extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all sample test cases
  python concordance/quick_test.py
  
  # Parse a custom entry string
  python concordance/quick_test.py --entry "KALIBR [CALIBER]: David Greenglass. Venona..."
  
  # Parse from stdin (useful for pasting)
  echo "Entry text here" | python concordance/quick_test.py --entry -
        """
    )
    parser.add_argument(
        "--entry",
        type=str,
        default=None,
        help="Parse a single entry string. Use '-' to read from stdin."
    )
    
    args = parser.parse_args()
    
    if args.entry:
        # Parse single entry
        if args.entry == "-":
            # Read from stdin
            entry_text = sys.stdin.read()
        else:
            entry_text = args.entry
        
        if not entry_text.strip():
            print("ERROR: Empty entry text provided", file=sys.stderr)
            sys.exit(1)
        
        success = parse_single_entry(entry_text.strip())
        sys.exit(0 if success else 1)
    else:
        # Run all sample entries
        quick_test()


if __name__ == "__main__":
    main()
