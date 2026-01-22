#!/usr/bin/env python3
"""
seed_entities_from_concordance.py [--source-slug <slug>] [--dry-run]

Migrates entities from the concordance index (old entities/entity_aliases tables)
to the new unified entity system (new entities/entity_aliases tables).

Usage:
    # Dry run to see what would be migrated
    python scripts/seed_entities_from_concordance.py --source-slug venona_vassiliev_concordance_v3 --dry-run
    
    # Actually migrate
    python scripts/seed_entities_from_concordance.py --source-slug venona_vassiliev_concordance_v3
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import Json

from retrieval.entity_resolver import normalize_alias, entity_create, entity_add_alias


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def infer_entity_type(canonical_name: str, aliases: List[str]) -> str:
    """
    Infer entity_type (person/org/place) from name and aliases.
    Simple heuristic: check for organization indicators, place indicators, default to person.
    """
    all_text = (canonical_name + " " + " ".join(aliases)).lower()
    
    # Organization indicators (expanded list)
    org_indicators = [
        "department", "bureau", "agency", "service", "committee", "commission",
        "ministry", "nkvd", "kgb", "mgb", "gru", "cia", "fbi", "treasury",
        "state department", "war department", "navy", "army", "intelligence",
        "office", "administration", "administration", "board", "council",
        "institute", "foundation", "corporation", "corp", "company", "co.",
        "organization", "organisation", "society", "union", "party", "party",
        "republic", "government", "govt", "military", "forces", "command",
        "headquarters", "hq", "station", "base", "unit", "division", "corps",
        "regiment", "battalion", "squadron", "fleet", "embassy", "consulate"
    ]
    if any(indicator in all_text for indicator in org_indicators):
        return "org"
    
    # Place indicators (less common in this corpus, but check)
    place_indicators = [
        "moscow", "washington", "new york", "london", "berlin", "paris",
        "city", "country", "state"
    ]
    if any(indicator in all_text for indicator in place_indicators):
        return "place"
    
    # Default to person
    return "person"


def migrate_concordance_entities(
    conn,
    source_slug: str,
    *,
    dry_run: bool = False,
    default_entity_type: str = "person",
) -> Tuple[int, int]:
    """
    Migrate entities from concordance index to new entity system.
    
    The old concordance tables have:
    - entities(source_id, canonical_name, entity_type, ...)
    - entity_aliases(source_id, entity_id, alias, alias_type, ...)
    
    The new unified tables have:
    - entities(entity_type, canonical_name, ...)  [no source_id]
    - entity_aliases(entity_id, alias, alias_norm, kind, ...)  [no source_id]
    
    Returns (entities_created, aliases_added).
    """
    # 1. Get source_id from concordance_sources
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM concordance_sources WHERE slug = %s LIMIT 1",
            (source_slug,)
        )
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"Concordance source not found: {source_slug}")
        source_id = row[0]
    
    # 2. Fetch all entities and their aliases from concordance (old tables)
    with conn.cursor() as cur:
        # Get entities (old table has source_id, entity_type, canonical_name)
        cur.execute("""
            SELECT id, canonical_name, entity_type
            FROM entities
            WHERE source_id = %s
            ORDER BY id
        """, (source_id,))
        concordance_entities = cur.fetchall()
        
        # Get aliases grouped by entity_id (old table has source_id, entity_id, alias, alias_type)
        cur.execute("""
            SELECT entity_id, alias, alias_type
            FROM entity_aliases
            WHERE source_id = %s
            ORDER BY entity_id, id
        """, (source_id,))
        concordance_aliases = cur.fetchall()
    
    # Group aliases by entity_id (map old entity_id -> list of (alias, alias_type))
    aliases_by_entity: Dict[int, List[Tuple[str, Optional[str]]]] = {}
    for entity_id, alias, alias_type in concordance_aliases:
        if entity_id not in aliases_by_entity:
            aliases_by_entity[entity_id] = []
        aliases_by_entity[entity_id].append((alias, alias_type))
    
    # 3. Check which entities already exist in new system (by canonical_name)
    existing_entities: Dict[str, int] = {}  # canonical_name -> new entity_id
    if not dry_run:
        with conn.cursor() as cur:
            cur.execute("SELECT id, canonical_name FROM entities")
            for entity_id, canonical_name in cur.fetchall():
                existing_entities[canonical_name.lower()] = entity_id
    
    entities_created = 0
    aliases_added = 0
    
    # 4. Migrate each entity
    for old_entity_id, canonical_name, old_entity_type in concordance_entities:
        alias_tuples = aliases_by_entity.get(old_entity_id, [])
        aliases = [alias for alias, _ in alias_tuples]
        
        # Use entity_type from concordance if available, otherwise infer
        if old_entity_type and old_entity_type in ('person', 'org', 'place'):
            entity_type = old_entity_type
        else:
            entity_type = infer_entity_type(canonical_name, aliases)
            if entity_type == "person" and default_entity_type != "person":
                entity_type = default_entity_type  # Allow override
        
        # Check if entity already exists
        existing_id = existing_entities.get(canonical_name.lower())
        
        if dry_run:
            if existing_id:
                print(f"[SKIP] Entity already exists: '{canonical_name}' (id={existing_id})")
                print(f"       Would add {len(aliases)} aliases")
                # Count aliases that would be added (even if some might already exist)
                aliases_added += len(aliases)
            else:
                print(f"[CREATE] Entity: '{canonical_name}' (type={entity_type})")
                print(f"         Aliases ({len(aliases)}): {', '.join(aliases[:5])}{'...' if len(aliases) > 5 else ''}")
                entities_created += 1
                aliases_added += len(aliases)
            continue
        
        # Create or get existing entity
        if existing_id:
            new_entity_id = existing_id
            print(f"[SKIP] Entity already exists: '{canonical_name}' (id={new_entity_id})")
            # Still need to add aliases that might be missing
            for alias, alias_type in alias_tuples:
                # Map old alias_type to new kind
                kind = "alt"  # default
                if alias_type:
                    kind_map = {
                        "primary": "primary",
                        "alt": "alt",
                        "misspelling": "misspelling",
                        "initials": "initials",
                        "ru_translit": "ru_translit",
                        "code_name": "code_name",
                    }
                    kind = kind_map.get(alias_type.lower(), "alt")
                
                try:
                    entity_add_alias(conn, new_entity_id, alias, kind=kind)
                    aliases_added += 1
                except Exception:
                    # Alias might already exist, skip
                    pass
        else:
            # Create new entity with all aliases
            # Map alias types to kinds for entity_create
            # Note: entity_create doesn't support kind per alias, so we'll add them after
            new_entity_id = entity_create(
                conn,
                canonical_name=canonical_name,
                entity_type=entity_type,
                aliases=aliases,  # entity_create adds these as 'alt' by default
            )
            entities_created += 1
            aliases_added += len(aliases)
            
            # Update alias kinds for non-default types
            for alias, alias_type in alias_tuples:
                if alias_type and alias_type.lower() != "alt":
                    kind_map = {
                        "primary": "primary",
                        "misspelling": "misspelling",
                        "initials": "initials",
                        "ru_translit": "ru_translit",
                        "code_name": "code_name",
                    }
                    kind = kind_map.get(alias_type.lower())
                    if kind:
                        # Delete and re-add with correct kind
                        # (Simpler: just update if we can, or skip if constraint prevents)
                        try:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    UPDATE entity_aliases
                                    SET kind = %s
                                    WHERE entity_id = %s AND alias_norm = %s
                                """, (kind, new_entity_id, normalize_alias(alias)))
                                conn.commit()
                        except Exception:
                            # If update fails, alias might not exist or constraint prevents
                            pass
            
            print(f"[CREATE] Entity: '{canonical_name}' (id={new_entity_id}, type={entity_type}, {len(aliases)} aliases)")
    
    return entities_created, aliases_added


def main():
    ap = argparse.ArgumentParser(description="Migrate entities from concordance index to new entity system")
    ap.add_argument("--source-slug", default="venona_vassiliev_concordance_v3",
                    help="Concordance source slug to migrate from")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be migrated without actually doing it")
    ap.add_argument("--default-entity-type", choices=["person", "org", "place"], default="person",
                    help="Default entity type if inference fails (default: person)")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        print(f"Migrating entities from concordance source: {args.source_slug}")
        if args.dry_run:
            print("[DRY RUN] No changes will be made\n")
        
        entities_created, aliases_added = migrate_concordance_entities(
            conn,
            args.source_slug,
            dry_run=args.dry_run,
            default_entity_type=args.default_entity_type,
        )
        
        print(f"\n{'Would create' if args.dry_run else 'Created'}:")
        print(f"  Entities: {entities_created}")
        print(f"  Aliases: {aliases_added}")
        
        if args.dry_run:
            print("\nRun without --dry-run to actually migrate")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
