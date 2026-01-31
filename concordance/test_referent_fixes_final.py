#!/usr/bin/env python3
"""Test script for referent extraction fixes"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ingest_concordance_tab_aware import infer_referent_from_body_start

test_cases = [
    # Case 1: BARCH (cover name in Venona) Semen Kremer
    {
        "entry_key": "BARCH (cover name in Venona) Semen Kremer",
        "body": "BARCH (cover name in Venona) Semen Kremer. Venona London GRU, 2, 6–9...",
        "expected": "Semen Kremer"
    },
    # Case 2: """Glan" (incomplete quote)
    {
        "entry_key": None,
        "body": '"""Glan." As Barkovsky: ...',
        "expected": "Glan"
    },
    # Case 3: "then ""Meter"""
    {
        "entry_key": None,
        "body": 'then "Meter" ["Metr"]. Cover name in Venona: ...',
        "expected": "Meter"
    },
    # Case 4: Beigel, Rose. Also know as Rose Arenal...
    {
        "entry_key": "Beigel, Rose. Also know as Rose Arenal, wife of Luis Arenal. Cover name in Venona",
        "body": "Beigel, Rose. Also know as Rose Arenal, wife of Luis Arenal. Cover name in Venona: ROSE [ROZA].",
        "expected": "Rose Beigel"
    },
    # Case 5: BELKA [SQUIRREL] was identified... possibly Ann Sidorovich
    {
        "entry_key": "BELKA [SQUIRREL] (cover name in Venona)",
        "body": "BELKA [SQUIRREL] was identified in a single 1945 Venona message that also discussed LENS (Michael Sidorovich) and Venona analysts suggested that BELKA was possibly Ann Sidorovich.",
        "expected": "Ann Sidorovich"
    },
    # Case 6: "then ""Myrna"""
    {
        "entry_key": None,
        "body": 'then "Myrna" ["Mirna"]. Cover name in Venona: ...',
        "expected": "Myrna"
    },
    # Case 7: Analysts with the Venona project judged that VEKSEL was "possibly" J. Robert Oppenheimer
    {
        "entry_key": '"BILL OF EXCHANGE" ["VEKSEL\'"] (cover name in Venona)',
        "body": 'Analysts with the Venona project judged that VEKSEL was "possibly" J. Robert Oppenheimer. However, in light of information...',
        "expected": "J. Robert Oppenheimer"
    },
]

print("Testing referent extraction fixes...")
print("=" * 80)

all_passed = True
for i, test in enumerate(test_cases, 1):
    result = infer_referent_from_body_start(test["body"], entry_key=test["entry_key"])
    passed = result == test["expected"]
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    
    print(f"\nTest {i}: {status}")
    print(f"  Entry key: {test['entry_key']}")
    print(f"  Body: {test['body'][:80]}...")
    print(f"  Expected: {test['expected']}")
    print(f"  Got:      {result}")
    if not passed:
        print(f"  [FAIL] MISMATCH")

print("\n" + "=" * 80)
if all_passed:
    print("[PASS] All tests passed!")
else:
    print("[FAIL] Some tests failed")
sys.exit(0 if all_passed else 1)
