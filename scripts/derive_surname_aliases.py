#!/usr/bin/env python3
"""
Derive surname aliases for person entities.

For each person entity with multi-token canonical/original name,
creates a derived alias for the last token (surname).
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn
from retrieval.entity_resolver import normalize_alias

# Generic words to exclude from surname derivation
GENERIC_WORDS_TO_EXCLUDE = {
    "i", "which", "their", "them", "what", "when", "there", "whom", "this", "that", "these", "those",
    "work", "city", "group", "pages", "serial", "known", "sent", "ref", "pm", "case", "time", "terms",
    "refer", "secret", "affairs", "minister", "given", "funds", "reply", "link", "telegraph", "working",
    "note", "general", "real", "financial", "doctor", "agent", "cases", "currency", "cutting", "ministry",
    "reference", "distant", "president", "chief", "cipher", "internal", "line", "also",
}


def derive_surname_aliases(conn, source_slug: str, dry_run: bool = False):
    """
    For each person entity with multi-token canonical/original name,
    create a derived alias for the last token (surname).
    """
    with conn.cursor() as cur:
        # Get source_id
        cur.execute("SELECT id FROM concordance_sources WHERE slug = %s", (source_slug,))
        source_row = cur.fetchone()
        if not source_row:
            print(f"ERROR: Source slug '{source_slug}' not found", file=sys.stderr)
            return
        source_id = source_row[0]

        # Find person entities with multi-token canonical/original aliases
        cur.execute("""
            SELECT DISTINCT ON (e.id, ea.alias_norm) e.id, e.canonical_name, ea.alias_norm, ea.alias
            FROM entities e
            JOIN entity_aliases ea ON ea.entity_id = e.id
            JOIN concordance_entries ce ON ce.id = ea.entry_id
            JOIN concordance_sources cs ON cs.id = ce.source_id
            WHERE e.entity_type = 'person'
              AND cs.slug = %s
              AND ea.alias_type IN ('canonical', 'original_form')
              AND array_length(string_to_array(ea.alias_norm, ' '), 1) > 1
            ORDER BY e.id, ea.alias_norm, ea.id
        """, (source_slug,))
        
        derived_count = 0
        skipped_count = 0
        
        for entity_id, canonical, alias_norm, original_alias in cur.fetchall():
            tokens = alias_norm.split()
            surname = tokens[-1]
            
            # Only derive if surname is >= 4 chars and not generic
            if len(surname) < 4:
                skipped_count += 1
                continue
            
            if surname.lower() in GENERIC_WORDS_TO_EXCLUDE:
                skipped_count += 1
                continue
            
            # Check if surname alias already exists (any alias_type)
            # The unique constraint is on (entity_id, alias_norm), so if it exists, skip
            cur.execute("""
                SELECT id FROM entity_aliases
                WHERE entity_id = %s AND alias_norm = %s
            """, (entity_id, surname))
            
            if cur.fetchone():
                skipped_count += 1
                continue
            
            if dry_run:
                print(f"Would derive: entity_id={entity_id}, surname='{surname}' from '{original_alias}'")
                derived_count += 1
            else:
                # Insert derived surname alias
                # Use ON CONFLICT DO NOTHING to handle race conditions gracefully
                cur.execute("""
                    INSERT INTO entity_aliases 
                    (source_id, entry_id, entity_id, alias, alias_norm, alias_type, alias_class, 
                     is_auto_match, is_matchable, min_chars, match_case)
                    VALUES (%s, NULL, %s, %s, %s, 'derived_last_name', 'person_last',
                            true, true, 4, 'any')
                    ON CONFLICT (entity_id, alias_norm) DO NOTHING
                """, (source_id, entity_id, surname, surname))
                if cur.rowcount > 0:
                    derived_count += 1
                else:
                    skipped_count += 1
        
        if not dry_run:
            conn.commit()
        
        print(f"Derived {derived_count} surname aliases (skipped {skipped_count})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Derive surname aliases for person entities")
    parser.add_argument("--source-slug", required=True, help="Concordance source slug")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be derived without inserting")
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        derive_surname_aliases(conn, args.source_slug, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
