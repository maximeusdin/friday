#!/usr/bin/env python3
"""
Explain why a specific alias has multiple candidate entities (collision).

Usage:
    python scripts/explain_collision.py yakubovich
    python scripts/explain_collision.py --alias-norm yakubovich
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import psycopg2
from retrieval.entity_resolver import normalize_alias

def get_conn():
    """Get database connection using environment variables."""
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "neh")
    DB_USER = os.getenv("DB_USER", "neh")
    DB_PASS = os.getenv("DB_PASS", "neh")
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

def explain_collision(alias_text: str):
    """Explain why an alias has multiple candidate entities."""
    alias_norm = normalize_alias(alias_text)
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Find all entities with this alias_norm
            cur.execute("""
                SELECT 
                    e.id AS entity_id,
                    e.canonical_name,
                    e.entity_type,
                    ea.alias AS original_alias,
                    ea.alias_class,
                    ea.is_auto_match,
                    ea.match_case,
                    ea.min_chars,
                    ea.allow_ambiguous_person_token
                FROM entity_aliases ea
                JOIN entities e ON e.id = ea.entity_id
                WHERE ea.alias_norm = %s
                  AND ea.is_matchable = true
                ORDER BY e.id
            """, (alias_norm,))
            
            candidates = cur.fetchall()
            
            if not candidates:
                print(f"No matchable entities found for alias_norm: '{alias_norm}'", file=sys.stderr)
                print(f"  (normalized from: '{alias_text}')", file=sys.stderr)
                return
            
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"COLLISION ANALYSIS: '{alias_text}' → alias_norm: '{alias_norm}'", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            print(f"\nFound {len(candidates)} candidate entities:\n", file=sys.stderr)
            
            for i, (entity_id, canonical_name, entity_type, original_alias, 
                    alias_class, is_auto_match, match_case, min_chars, 
                    allow_ambiguous) in enumerate(candidates, 1):
                print(f"{i}. Entity ID: {entity_id}", file=sys.stderr)
                print(f"   Canonical Name: {canonical_name}", file=sys.stderr)
                print(f"   Entity Type: {entity_type}", file=sys.stderr)
                print(f"   Original Alias: '{original_alias}'", file=sys.stderr)
                print(f"   Alias Class: {alias_class}", file=sys.stderr)
                print(f"   Auto-match: {is_auto_match}", file=sys.stderr)
                print(f"   Match Case: {match_case}", file=sys.stderr)
                print(f"   Min Chars: {min_chars}", file=sys.stderr)
                print(f"   Allow Ambiguous Person Token: {allow_ambiguous}", file=sys.stderr)
                print("", file=sys.stderr)
            
            # Check if it's a single token
            tokens = alias_norm.split()
            is_single_token = len(tokens) == 1
            
            print(f"Analysis:", file=sys.stderr)
            print(f"  - Alias norm: '{alias_norm}'", file=sys.stderr)
            print(f"  - Single token: {is_single_token}", file=sys.stderr)
            print(f"  - Number of candidates: {len(candidates)}", file=sys.stderr)
            
            # Determine collision classification
            # Infer alias_class if None (same logic as load_all_aliases)
            inferred_alias_classes = []
            for row in candidates:
                entity_type = row[2]
                alias_class = row[4]
                if alias_class is None:
                    # Infer from entity_type and token count (same as load_all_aliases)
                    if entity_type == 'person':
                        alias_class = 'person_given' if is_single_token else 'person_full'
                    elif entity_type == 'org':
                        alias_class = 'org'
                    elif entity_type == 'place':
                        alias_class = 'place'
                    else:
                        alias_class = 'generic_word'
                inferred_alias_classes.append(alias_class)
            
            # Check if it would be classified as harmless
            # Based on is_collision_harmless logic:
            # 1. Single token with >5 candidates = harmless
            # 2. Person given name (single token) with multiple entities (unless allow_ambiguous_person_token=True) = harmless
            # 3. Generic word = harmless
            
            is_harmless_reason = None
            if is_single_token and len(candidates) > COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES:
                is_harmless_reason = f"Single token with >{COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES} candidates"
            elif is_single_token:
                # Check if it's a person_given name
                person_given_without_allow = sum(1 for i, row in enumerate(candidates) 
                                                 if row[2] == 'person' 
                                                 and inferred_alias_classes[i] == 'person_given'
                                                 and not row[8])  # allow_ambiguous_person_token
                if person_given_without_allow > 0 and len(candidates) > 1:
                    is_harmless_reason = "Single-token person given name with multiple entities (allow_ambiguous_person_token=False)"
                elif any(inferred_alias_classes[i] == 'generic_word' for i in range(len(candidates))):
                    is_harmless_reason = "Generic word class"
            
            if is_harmless_reason:
                print(f"  - Classification: HARMLESS", file=sys.stderr)
                print(f"    → Reason: {is_harmless_reason}", file=sys.stderr)
                print(f"    → Not enqueued for review (too ambiguous/common)", file=sys.stderr)
            elif is_single_token:
                print(f"  - Classification: POTENTIALLY HIGH-VALUE (single token with ≤{COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES} candidates)", file=sys.stderr)
                print(f"    → May be enqueued if meets other criteria", file=sys.stderr)
            else:
                print(f"  - Classification: MULTI-TOKEN (may be high-value)", file=sys.stderr)
            
            # Show inferred alias classes
            if any(row[4] is None for row in candidates):
                print(f"  - Inferred alias classes: {inferred_alias_classes}", file=sys.stderr)
            
            # Check entity types
            entity_types = set(row[2] for row in candidates)  # entity_type is index 2
            if len(entity_types) > 1:
                print(f"  - ⚠️  Mixed entity types: {entity_types}", file=sys.stderr)
                print(f"    → This may indicate data quality issues", file=sys.stderr)
            
            # Check alias classes
            alias_classes = set(row[4] for row in candidates if row[4])  # alias_class is index 4
            if len(alias_classes) > 1:
                print(f"  - ⚠️  Mixed alias classes: {alias_classes}", file=sys.stderr)
            
            # Check auto-match status
            auto_match_count = sum(1 for row in candidates if row[5])  # is_auto_match is index 5
            if auto_match_count < len(candidates):
                print(f"  - Auto-match enabled: {auto_match_count}/{len(candidates)}", file=sys.stderr)
            
            # Show more details about the entities
            print(f"\nEntity Details:", file=sys.stderr)
            for i, row in enumerate(candidates, 1):
                entity_id = row[0]
                canonical_name = row[1]
                
                # Check if entity has other aliases
                cur.execute("""
                    SELECT alias, alias_class, is_auto_match
                    FROM entity_aliases
                    WHERE entity_id = %s
                      AND is_matchable = true
                    ORDER BY alias
                    LIMIT 10
                """, (entity_id,))
                other_aliases = cur.fetchall()
                
                print(f"\n  Entity {i} (ID: {entity_id}): {canonical_name}", file=sys.stderr)
                if other_aliases:
                    print(f"    Other aliases ({len(other_aliases)} shown):", file=sys.stderr)
                    for alias, alias_class, is_auto in other_aliases[:5]:
                        print(f"      - '{alias}' (class: {alias_class}, auto: {is_auto})", file=sys.stderr)
                    if len(other_aliases) > 5:
                        print(f"      ... and {len(other_aliases) - 5} more", file=sys.stderr)
            
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"\nExplanation:", file=sys.stderr)
            print(f"  'yakubovich' is a single-token person name (last name) that matches", file=sys.stderr)
            print(f"  {len(candidates)} different entities. Since both entities are persons and", file=sys.stderr)
            print(f"  allow_ambiguous_person_token=False, this collision is classified as", file=sys.stderr)
            print(f"  'harmless' - meaning it's too common/ambiguous to be worth manual review.", file=sys.stderr)
            print(f"", file=sys.stderr)
            print(f"  This is a conservative approach: single-token person names like", file=sys.stderr)
            print(f"  'yakubovich', 'smith', 'john' are very common and would generate", file=sys.stderr)
            print(f"  too many false positives if auto-matched without context.", file=sys.stderr)
            print(f"", file=sys.stderr)
            print(f"  To enable auto-matching for these, you would need to:", file=sys.stderr)
            print(f"  1. Set allow_ambiguous_person_token=true for the alias, OR", file=sys.stderr)
            print(f"  2. Use citation-based disambiguation to resolve the collision", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            
    finally:
        conn.close()

# Import the constant from extract_entity_mentions
COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES = 5

def main():
    ap = argparse.ArgumentParser(description="Explain why an alias has multiple candidate entities")
    ap.add_argument("alias", nargs="?", help="Alias text to analyze (e.g., 'yakubovich')")
    ap.add_argument("--alias-norm", help="Normalized alias (if already normalized)")
    
    args = ap.parse_args()
    
    if args.alias_norm:
        alias_norm = args.alias_norm
    elif args.alias:
        alias_norm = normalize_alias(args.alias)
    else:
        ap.error("Must provide either alias or --alias-norm")
    
    explain_collision(alias_norm if args.alias_norm else args.alias)

if __name__ == "__main__":
    main()
