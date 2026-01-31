#!/usr/bin/env python3
"""
Derive acronym aliases for org/agency entities.

For org entities, creates derived acronym aliases if the canonical name
matches known acronym expansion patterns.
"""

import sys
import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn
from retrieval.entity_resolver import normalize_alias

# Known acronyms and their expansion patterns
KNOWN_ACRONYMS = {
    "kgb": [
        "komitet gosudarstvennoy bezopasnosti",
        "committee for state security",
        "committee of state security",
    ],
    "gru": [
        "glavnoye razvedyvatelnoye upravleniye",
        "main intelligence directorate",
        "glavnoye razvedyvatelnoe upravlenie",
    ],
    "mgb": [
        "ministerstvo gosudarstvennoy bezopasnosti",
        "ministry of state security",
        "ministry for state security",
    ],
    "nkvd": [
        "narodnyy komissariat vnutrennikh del",
        "people's commissariat for internal affairs",
        "peoples commissariat",
    ],
    "fbi": [
        "federal bureau of investigation",
    ],
    "cia": [
        "central intelligence agency",
    ],
    "nsa": [
        "national security agency",
    ],
    "sis": [
        "secret intelligence service",
    ],
    "mi6": [
        "military intelligence section 6",
        "military intelligence 6",
    ],
    "ussr": [
        "union of soviet socialist republics",
        "soviet union",
    ],
}


def matches_acronym_expansion(canonical_lower: str, expansion_patterns: list) -> bool:
    """Check if canonical name matches any expansion pattern."""
    for pattern in expansion_patterns:
        # Check if pattern appears in canonical (allowing for variations)
        pattern_words = pattern.split()
        # Try to match first few words of pattern
        if len(pattern_words) >= 2:
            first_words = " ".join(pattern_words[:2])
            if first_words in canonical_lower:
                return True
        if pattern in canonical_lower:
            return True
    return False


def derive_acronym_aliases(conn, source_slug: str, dry_run: bool = False):
    """
    For org entities, create derived acronym aliases if the canonical name
    matches a known acronym expansion pattern.
    """
    with conn.cursor() as cur:
        # Get source_id
        cur.execute("SELECT id FROM concordance_sources WHERE slug = %s", (source_slug,))
        source_row = cur.fetchone()
        if not source_row:
            print(f"ERROR: Source slug '{source_slug}' not found", file=sys.stderr)
            return
        source_id = source_row[0]

        derived_count = 0
        skipped_count = 0
        
        for acronym, expansion_patterns in KNOWN_ACRONYMS.items():
            # Find org entities whose canonical name contains the expansion
            cur.execute("""
                SELECT DISTINCT e.id, e.canonical_name, ea.alias_norm
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                JOIN concordance_entries ce ON ce.id = ea.entry_id
                JOIN concordance_sources cs ON cs.id = ce.source_id
                WHERE e.entity_type = 'org'
                  AND cs.slug = %s
                  AND ea.alias_type = 'canonical'
            """, (source_slug,))
            
            for entity_id, canonical_name, alias_norm in cur.fetchall():
                canonical_lower = canonical_name.lower() if canonical_name else ""
                alias_norm_lower = alias_norm.lower() if alias_norm else ""
                
                # Check if canonical matches expansion pattern
                if not matches_acronym_expansion(canonical_lower + " " + alias_norm_lower, expansion_patterns):
                    continue
                
                # Check if acronym alias already exists
                cur.execute("""
                    SELECT id FROM entity_aliases
                    WHERE entity_id = %s AND alias_norm = %s AND alias_type = 'derived_acronym'
                """, (entity_id, acronym))
                
                if cur.fetchone():
                    skipped_count += 1
                    continue
                
                if dry_run:
                    print(f"Would derive: entity_id={entity_id}, acronym='{acronym.upper()}' for '{canonical_name}'")
                    derived_count += 1
                else:
                    # Insert derived acronym alias
                    cur.execute("""
                        INSERT INTO entity_aliases
                        (source_id, entry_id, entity_id, alias, alias_norm, alias_type, alias_class,
                         is_auto_match, is_matchable, min_chars, match_case)
                        VALUES (%s, NULL, %s, %s, %s, 'derived_acronym', 'org',
                                true, true, 2, 'upper_only')
                    """, (source_id, entity_id, acronym.upper(), acronym))
                    derived_count += 1
        
        if not dry_run:
            conn.commit()
        
        print(f"Derived {derived_count} acronym aliases (skipped {skipped_count})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Derive acronym aliases for org entities")
    parser.add_argument("--source-slug", required=True, help="Concordance source slug")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be derived without inserting")
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        derive_acronym_aliases(conn, args.source_slug, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
