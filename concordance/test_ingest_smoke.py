#!/usr/bin/env python3
"""
Smoke test for concordance ingestion based on real-world examples.

This test validates that the fixes for malformed entity links work correctly.
All test cases are based on actual issues found in entity_links.csv.

Usage:
    python concordance/test_ingest_smoke.py
"""

import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concordance.ingest_concordance_tab_aware import parse_entry_block


# Test cases based on real examples from entity_links.csv
SMOKE_TEST_CASES = [
    {
        "name": "Leopol or Leopolo Arenal - should extract first part",
        "entry_key": "Arenal, Leopol or Leopolo",
        "body": "Arenal, Leopol or Leopolo: Soviet intelligece agent and Mexican Communist. Brother of Luis Arenal. Cover name in Vassiliev's notebooks: \"Alexander\" [\"Aleksandr\"]. Cover name in Venona: ALEKSANDR.",
        "expected_canonical": "Leopol Arenal",  # Should use first part of "or" pattern
        "expected_type": "person",
    },
    {
        "name": "Andrey Raina - extract from first sentence",
        "entry_key": "ARSENIJ (cover name in Venona)",
        "body": "ARSENIJ (cover name in Venona): Soviet intelligence officer Andrey Raina, pseudonym Andrey Shevchenko in the United States. \"Arseny\" was identified as Raina in Alexander Vassiliev's unpublished summary narrative.",
        "expected_referent": "Andrey Raina",  # Should extract from first sentence, not "was identified as"
        "expected_type": "person",
    },
    {
        "name": "A. Slavyagin - strip KGB officer prefix",
        "entry_key": "\"Artem\" (cover name in Vassiliev's notebooks)",
        "body": "\"Artem\" (cover name in Vassiliev's notebooks): A. Slavyagin, KGB officer. \"Artem\" was identified in the Venona decryptions as likely the cover name of either G. N. Ogloblin or M.N. Khvostov.",
        "expected_referent": "A. Slavyagin",  # Should extract from first sentence, stripping "KGB officer" prefix
        "expected_type": "person",
    },
    {
        "name": "Anatoly Gorsky - extract from pseudonym pattern",
        "entry_key": "BADEMUS (cover name in Venona)",
        "body": "Anatoly Gorsky, pseudonym Anatoly Gromov. BADEMUS is a play on the Russian word for \"bathhouse\" (banya).",
        "expected_referent": "Anatoly Gorsky",  # Should extract last two words from "pseudonym X Y"
        "expected_type": "person",
    },
    {
        "name": "Semen Kremer - extract from entry key",
        "entry_key": "BARCH (cover name in Venona) Semen Kremer",
        "body": "BARCH (cover name in Venona) Semen Kremer. Venona London GRU, 2, 6–9, 11, 13–14, 17–21, 30, 34, 36–38, 40–41, 43–44, 46–48, 50–53, 55, 57, 59–61, 63, 68–69, 72–74, 77, 79–80, 82–83, 86–87, 89–91, 94, 96–97, 100–101, 103–105, 107, 111–12, 114–17, 119, 122, 125–26, 128, 134, 139, 146, 148–49, 154–55, 157, 166–67, 245–46 Venona London GRU",
        "expected_referent": "Semen Kremer",  # Should extract from entry key
        "expected_type": "person",
    },
    {
        "name": "Glan - strip incomplete quotes",
        "entry_key": "\"Glan\"",
        "body": "Barkovsky, Vladimir B.: KGB officer in London, 1941–46, later at Moscow Center. Cover name in Vassiliev's notebooks: \"Glan.\" As Barkovsky: Vassiliev Yellow Notebook #1, 57. As \"Glan\": Vassiliev Yellow Notebook #1, 5, 7.",
        "expected_aliases_contain": ["Glan"],  # Should not have """Glan"
        "expected_type": "cover_name",
    },
    {
        "name": "METER and METRE - split into separate aliases",
        "entry_key": "Barr, Joel",
        "body": "Barr, Joel: Soviet intelligence source and member of the Rosenberg apparatus. Secret Communist, electrical engineer with Army Signal Corps laboratories and Western Electric. Cover name in Vassiliev's notebooks: \"Scout\" [\"Skaut\"] prior to September 1944, then \"Meter\" [\"Metr\"]. Cover name in Venona: SCOUT [SKAUT] prior to September 1944, then METER and METRE [METR]. As Barr: Vassiliev Black Notebook, 120; Venona New York KGB 1944, 255, 463, 559, 643, 675, 702, 716; Venona Special Studies, 47, 68, 77.",
        "expected_aliases_contain": ["METER", "METRE"],  # Should be separate aliases
        "expected_type": "person",
    },
    {
        "name": "Frank Oppenheimer - strip temporal qualifier",
        "entry_key": "\"Beam\" [\"Luch\"] (cover name in Vassiliev's notebooks)",
        "body": "\"Beam\" [\"Luch\"] (cover name in Vassiliev's notebooks): Frank Oppenheimer circa 1943–1944. Vassiliev Black Notebook, 119, 121, 135.",
        "expected_referent": "Frank Oppenheimer",  # Should strip "circa 1943–1944"
        "expected_type": "person",
    },
    {
        "name": "Ann Sidorovich - BELKA and SQUIRREL as aliases",
        "entry_key": "BELKA [SQUIRREL] (cover name in Venona)",
        "body": "BELKA [SQUIRREL] (cover name in Venona): BELKA [SQUIRREL] was identified in a single 1945 Venona message that also discussed LENS (Michael Sidorovich) and Venona analysts suggested that BELKA was possibly Ann Sidorovich.",
        "expected_referent": "Ann Sidorovich",
        "expected_aliases_contain": ["BELKA", "SQUIRREL"],  # Should be separate aliases
        "expected_type": "person",
    },
    {
        "name": "Myrna - strip 'then' qualifier and quotes",
        "entry_key": "Bentley, Elizabeth",
        "body": "Bentley, Elizabeth: Soviet intelligence agent. Liaison between CPUSA and Soviet intelligence during the 1930s and 1940s. Cover names in Vassiliev's notebooks: \"Artist\" [\"Khudozhnik\"] in 1939, \"Clever Girl\" [\"Umnitsa\"] (1940 until August 1944), then \"Myrna\" [\"Mirna\"]. Cover name in Venona: GOOD GIRL [UMNITSA], CLEVER GIRL [UMNITSA], and MYRNA [MIRNA].",
        "expected_aliases_contain": ["Myrna"],  # Should not have "then" or extra quotes
        "expected_type": "person",
    },
    {
        "name": "Vasily Zarubin - extract from first sentence",
        "entry_key": "\"Betty\" [\"Betti\"] (cover name in Vassiliev's notebooks)",
        "body": "\"Betty\" [\"Betti\"] (cover name in Vassiliev's notebooks): Vasily Zarubin in mid- and late 1930s. Vassiliev White Notebook #1, 133–34; Vassiliev Yellow Notebook #2, 18, 21, 72, 83; Vassiliev Yellow Notebook #3, 7, 9.",
        "expected_referent": "Vasily Zarubin",  # Should extract from first sentence, not citation
        "expected_type": "person",
    },
    {
        "name": "Harold Smeltzer - strip 'starting in' qualifier",
        "entry_key": "\"Armor\" [\"Bronya\"] (cover name in Vassiliev's notebooks)",
        "body": "\"Armor\" [\"Bronya\"] (cover name in Vassiliev's notebooks): Harold Smeltzer starting in October 1944. Vassiliev Black Notebook, 119, 121, 135.",
        "expected_referent": "Harold Smeltzer",  # Should strip "starting in October 1944"
        "expected_type": "person",
    },
    {
        "name": "Raina - remove 'in Alexander Vassiliev' pattern",
        "entry_key": "\"Arseny\" (cover name in Vassiliev's notebooks)",
        "body": "\"Arseny\" (cover name in Vassiliev's notebooks): Soviet intelligence officer Andrey Raina, pseudonym Andrey Shevchenko in the United States. \"Arseny\" was identified as Raina in Alexander Vassiliev's unpublished summary narrative.",
        "expected_referent": "Andrey Raina",  # Should use first sentence, not "was identified as Raina"
        "expected_type": "person",
    },
    {
        "name": "Rose Arenal - extract from 'Also know as'",
        "entry_key": "Beigel, Rose. Also know as Rose Arenal, wife of Luis Arenal. Cover name in Venona",
        "body": "Beigel, Rose. Also know as Rose Arenal, wife of Luis Arenal. Cover name in Venona: ROSE [ROZA]. As Beigel, Arenal, and ROSE [ROZA]: Venona New York KGB 1943, 279.",
        "expected_referent": "Rose Arenal",  # Should extract from "Also know as" part of entry key
        "expected_aliases_contain": ["ROSE", "ROZA"],  # Should extract from "Cover name in Venona: ROSE [ROZA]"
        "expected_type": "person",
    },
    {
        "name": "BAL - strip ellipsis",
        "entry_key": "BAL... (cover name in Venona)",
        "body": "BAL... (cover name in Venona): Partial decoding of a cover name, likely BALLOON/atomic bomb. Venona New York KGB 1945, 160–61.",
        "expected_canonical": "BAL",  # Should strip "..."
        "expected_type": "cover_name",
    },
    {
        "name": "Sonya - strip incomplete quotes",
        "entry_key": "Beurton, Ursula",
        "body": "Beurton, Ursula: Married name of Ursula Kuczynski. Cover name in Vassiliev's notebooks: \"Sonya.\" Cover name in Venona: SONYA [SONIA].",
        "expected_aliases_contain": ["Sonya"],  # Should not have """Sonya"
        "expected_type": "person",
    },
    {
        "name": "Vasily Zarubin from citation - complex citation pattern",
        "entry_key": "\"Betty\" [\"Betti\"] (cover name in Vassiliev's notebooks)",
        "body": "\"Betty\" [\"Betti\"] (cover name in Vassiliev's notebooks): Vasily Zarubin in mid- and late 1930s. Vassiliev White Notebook #1, 133–34; Vassiliev Yellow Notebook #2, 18, 21, 72, 83; Vassiliev Yellow Notebook #3, 7, 9.",
        "expected_referent": "Vasily Zarubin",  # Should extract from first sentence before citations
        "expected_type": "person",
    },
]


def run_smoke_test():
    """Run smoke tests and report results."""
    print("=" * 100)
    print("CONCORDANCE INGESTION SMOKE TEST")
    print("=" * 100)
    print()
    print("Testing fixes for malformed entity links based on real examples.")
    print()
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(SMOKE_TEST_CASES, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 100)
        
        try:
            # Construct full entry text (headword on first line, body follows)
            # The format is: "HEADWORD: body text..."
            entry_text = f"{test_case['entry_key']}: {test_case['body']}"
            
            # Parse entry
            pe = parse_entry_block(entry_text, entry_seq=i)
            
            # Print parsed entry summary
            print(f"Entity: {pe.entity_canonical!r} (type: {pe.entity_type})")
            if pe.aliases:
                alias_names = [a.alias for a in pe.aliases]
                print(f"Aliases: {alias_names}")
            if pe.links:
                link_summary = []
                for l in pe.links:
                    if l.link_type == "cover_name_of":
                        link_summary.append(f"{l.from_name} -> {l.to_name}")
                if link_summary:
                    print(f"Links: {link_summary}")
            
            # Run checks
            checks_passed = []
            checks_failed = []
            
            # Check entity type
            if "expected_type" in test_case:
                expected = test_case["expected_type"]
                actual = pe.entity_type
                if actual == expected:
                    checks_passed.append(f"Entity type: {actual}")
                else:
                    checks_failed.append(f"Entity type: expected {expected}, got {actual}")
            
            # Check canonical name
            if "expected_canonical" in test_case:
                expected = test_case["expected_canonical"]
                actual = pe.entity_canonical
                if actual == expected:
                    checks_passed.append(f"Canonical: {actual}")
                else:
                    checks_failed.append(f"Canonical: expected {expected!r}, got {actual!r}")
            
            # Check referent (for cover_name entries that should become person entries)
            if "expected_referent" in test_case:
                expected = test_case["expected_referent"]
                # Find referent from links
                referent_links = [l for l in pe.links if l.link_type == "cover_name_of"]
                if referent_links:
                    # If entity was inverted, the referent is now the entity itself
                    if pe.entity_type == "person" and pe.entity_canonical == expected:
                        checks_passed.append(f"Referent (inverted): {pe.entity_canonical}")
                    elif referent_links[0].to_name == expected:
                        checks_passed.append(f"Referent: {referent_links[0].to_name}")
                    else:
                        actual = referent_links[0].to_name if referent_links else pe.entity_canonical
                        checks_failed.append(f"Referent: expected {expected!r}, got {actual!r}")
                elif pe.entity_type == "person":
                    # For person entries, check if canonical matches (entity inversion case)
                    if pe.entity_canonical == expected:
                        checks_passed.append(f"Referent (person entity): {pe.entity_canonical}")
                    else:
                        checks_failed.append(f"Referent: expected {expected!r}, got {pe.entity_canonical!r} (person entity)")
                else:
                    checks_failed.append(f"Referent: expected {expected!r}, but no referent found (entity type: {pe.entity_type})")
            
            # Check aliases contain specific values
            if "expected_aliases_contain" in test_case:
                expected_aliases = set(test_case["expected_aliases_contain"])
                actual_aliases = set(a.alias for a in pe.aliases)
                actual_links = set(l.from_name for l in pe.links if l.link_type == "cover_name_of")
                all_aliases = actual_aliases | actual_links
                
                # Also check if aliases are contained within bracket variants (e.g., "METRE [METR]" contains "METRE")
                all_aliases_normalized = set()
                for alias in all_aliases:
                    all_aliases_normalized.add(alias)
                    # Extract name before brackets
                    bracket_pos = alias.find('[')
                    if bracket_pos > 0:
                        all_aliases_normalized.add(alias[:bracket_pos].strip())
                    # Extract bracket content
                    bracket_match = re.search(r'\[([^\]]+)\]', alias)
                    if bracket_match:
                        all_aliases_normalized.add(bracket_match.group(1).strip())
                
                missing = expected_aliases - all_aliases_normalized
                if not missing:
                    checks_passed.append(f"Aliases contain: {sorted(expected_aliases)}")
                else:
                    checks_failed.append(f"Aliases missing: {sorted(missing)} (found: {sorted(all_aliases_normalized)})")
            
            # Check aliases exact match
            if "expected_aliases" in test_case:
                expected = set(test_case["expected_aliases"])
                actual = set(a.alias for a in pe.aliases)
                if actual == expected:
                    checks_passed.append(f"Aliases: {sorted(actual)}")
                else:
                    checks_failed.append(f"Aliases: expected {sorted(expected)}, got {sorted(actual)}")
            
            # Report results
            if checks_failed:
                print(f"  [FAIL] {len(checks_failed)} check(s) failed:")
                for check in checks_failed:
                    print(f"    - {check}")
                if checks_passed:
                    print(f"  [PASS] {len(checks_passed)} check(s) passed:")
                    for check in checks_passed:
                        print(f"    - {check}")
                print(f"\n  [FAILED]")
                failed += 1
            else:
                print(f"  [PASS] All {len(checks_passed)} check(s) passed:")
                for check in checks_passed:
                    print(f"    - {check}")
                print(f"\n  [PASSED]")
                passed += 1
                
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 100)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(SMOKE_TEST_CASES)} tests")
    print("=" * 100)
    
    return failed == 0


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
