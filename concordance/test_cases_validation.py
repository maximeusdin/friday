#!/usr/bin/env python3
"""
Test case validation for concordance ingest.

Validates that specific test cases from the concordance are parsed correctly.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.ingest_concordance_tab_aware import parse_entry_block


# Test cases from concordance
TEST_CASES = [
    {
        "name": "Multiple covernames with brackets and 'and'",
        "input": """KALIBR [CALIBER and CALIBRE] (cover name in Venona): David Greenglass. Venona New York KGB
1944, 602, 643, 714, 716, 729; Venona New York KGB 1945, 24; Venona Special Studies, 32, 79,
141, 154.""",
        "checks": [
            lambda pe: pe.entity_type == "cover_name",
            lambda pe: "KALIBR" in pe.covername_variants or "KALIBR" == pe.entity_canonical,
            lambda pe: "CALIBER" in pe.covername_variants,
            lambda pe: "CALIBRE" in pe.covername_variants,
            lambda pe: any(l.to_name == "David Greenglass" for l in pe.links if l.link_type == "cover_name_of"),
        ]
    },
    {
        "name": "Cross-reference covername alias",
        "input": """"Kalibr" (Russian original of a cover name in Vassiliev's notebooks): See "Caliber".""",
        "checks": [
            lambda pe: pe.is_crossref_only,
            lambda pe: pe.crossref_target == "Caliber",
            lambda pe: any(l.to_name == "Caliber" for l in pe.links if l.link_type == "alias_of"),
        ]
    },
    {
        "name": "Unidentified covername (no person)",
        "input": """"Kaiser" [Kayzer] (cover name in Vassiliev's notebooks): Unidentified Soviet intelligence contact, friend of
Harold Glasser. Described as American Army captain in Italy in 1944, then working in the
Treasury Depatment in Washington, and appointed to the staff of the Allied Control Commission
in Austria. Formerly active in the Washington CPUSA network. Vassiliev White Notebook #3,
52.""",
        "checks": [
            lambda pe: pe.entity_type == "cover_name",
            lambda pe: "Kaiser" in pe.covername_variants or "Kaiser" == pe.entity_canonical,
            lambda pe: "Kayzer" in pe.covername_variants,
            lambda pe: not any(l.link_type == "cover_name_of" for l in pe.links),  # No person link
        ]
    },
    {
        "name": "Person name with comma inversion",
        "input": """Kalinin, Mikhail Ivanovich: Bolshevik leader and official Soviet head of state, 1919–46. Vassiliev Yellow
Notebook #4, 66.""",
        "checks": [
            lambda pe: pe.entity_type == "person",
            lambda pe: pe.entity_canonical == "Mikhail Ivanovich Kalinin",
            lambda pe: any(a.alias == "Kalinin, Mikhail Ivanovich" for a in pe.aliases),
        ]
    },
    {
        "name": "Person with question mark and covername",
        "input": """Kalinin, ?: Soviet sailor and Soviet internal security source. Cover name in Venona: ELKIN. Venona San
Francisco KGB, 88, 262; Venona Special Studies, 100.""",
        "checks": [
            lambda pe: pe.entity_type == "person",
            lambda pe: pe.entity_canonical == "Kalinin",
            lambda pe: "?" not in pe.entity_canonical,
            lambda pe: any(l.from_name == "ELKIN" and l.to_name == "Kalinin" for l in pe.links if l.link_type == "cover_name_of"),
        ]
    },
    {
        "name": "Person with full name, no covername",
        "input": """Kalinin, Tikhon Ivanovich: SGPC official. Venona San Francisco KGB, 77.""",
        "checks": [
            lambda pe: pe.entity_type == "person",
            lambda pe: pe.entity_canonical == "Tikhon Ivanovich Kalinin",
            lambda pe: not any(l.link_type == "cover_name_of" for l in pe.links),
        ]
    },
]


def validate_test_cases():
    """Run all test cases and report results."""
    print("=" * 100)
    print("CONCORDANCE INGEST TEST CASE VALIDATION")
    print("=" * 100)
    print()
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 100)
        
        try:
            pe = parse_entry_block(test_case["input"], entry_seq=1)
            
            # Print parsed entry
            print(f"Entity: {pe.entity_canonical!r} (type: {pe.entity_type})")
            if pe.covername_variants:
                print(f"Covername variants: {pe.covername_variants}")
            if pe.aliases:
                print(f"Aliases: {[a.alias for a in pe.aliases]}")
            if pe.links:
                print(f"Links: {[(l.link_type, l.from_name, l.to_name) for l in pe.links]}")
            
            # Run checks
            check_results = []
            for check in test_case["checks"]:
                try:
                    result = check(pe)
                    check_results.append(result)
                    status = "PASS" if result else "FAIL"
                    print(f"  [{status}] Check: {check.__name__ if hasattr(check, '__name__') else 'anonymous'}")
                except Exception as e:
                    check_results.append(False)
                    print(f"  [FAIL] Check failed with error: {e}")
            
            if all(check_results):
                print(f"\n[PASSED]")
                passed += 1
            else:
                print(f"\n[FAILED] ({sum(check_results)}/{len(check_results)} checks passed)")
                failed += 1
                
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 100)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(TEST_CASES)} test cases")
    print("=" * 100)
    
    return failed == 0


if __name__ == "__main__":
    success = validate_test_cases()
    sys.exit(0 if success else 1)
