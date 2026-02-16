#!/usr/bin/env python3
"""
Diagnostic: check OSS → Office of Strategic Services → izba/hut/cabin chain.

Run after: source ./friday_env.sh
  python scripts/check_oss_aliases.py

Answers:
1. What entities from the query are matched (Jacob Golos, OSS, etc.)
2. What PEM seed surfaces come from which entities
3. Why izba/hut/cabin might not appear (entity_aliases vs page_entity_mentions)
"""
import os
import sys

import psycopg2

def main():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set. Run: source ./friday_env.sh")
        sys.exit(1)

    conn = psycopg2.connect(dsn)

    print("=" * 70)
    print("1. ENTITIES FROM QUERY (from your terminal output)")
    print("=" * 70)
    print("""
Query: "who were Jacob Golos's sources in the OSS?"

Entity hypotheses (4): 69114, 78775, 76707, 84902
Promoted to entities_in_play (8):
  69114: Jacob Golos
  3431:  Golos, Jacob
  78775: Jacob Golos
  72076: OSS veterans organization
  76707: Office of Strategic Services   <-- OSS
  81888: OSS veterans organization
  84596: X-2
  84902: Office Strategic Services      <-- OSS (alternate entity)
""")

    print("=" * 70)
    print("2. PEM SEED SURFACES (from terminal: surfaces=['sound', 'sound zvuk',")
    print("   'tasin raisin', 'zvuk', 'rasin'])")
    print("=" * 70)
    print("""
These surfaces come from page_entity_mentions for the 8 entities.
PEM lane uses _get_seed_surfaces() which queries page_entity_mentions ONLY.
It does NOT use entity_aliases.
""")

    print("=" * 70)
    print("3. entity_aliases: OSS-related entities (76707, 84902) — do they have")
    print("   izba, hut, cabin?")
    print("=" * 70)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.canonical_name, ea.alias, ea.kind
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE e.id IN (76707, 84902, 72076, 81888)
                  AND (LOWER(ea.alias) IN ('oss', 'izba', 'hut', 'cabin', 'office of strategic services')
                       OR LOWER(ea.alias) LIKE '%izba%' OR LOWER(ea.alias) LIKE '%hut%'
                       OR LOWER(ea.alias) LIKE '%cabin%')
                ORDER BY e.id, ea.alias
            """)
            rows = cur.fetchall()
        if rows:
            for eid, canon, alias, kind in rows:
                print(f"  entity_id={eid} canonical={canon!r} alias={alias!r} kind={kind}")
        else:
            print("  (no rows — let's broaden the search)")
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.id, e.canonical_name, ea.alias, ea.kind
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE e.id IN (76707, 84902)
                    ORDER BY e.id, ea.alias
                    LIMIT 30
                """)
                for eid, canon, alias, kind in cur.fetchall():
                    print(f"  entity_id={eid} canonical={canon!r} alias={alias!r} kind={kind}")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("=" * 70)
    print("4. entity_aliases: any entity with izba, hut, or cabin?")
    print("=" * 70)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.canonical_name, ea.alias, ea.kind
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE LOWER(ea.alias) IN ('izba', 'hut', 'cabin')
                ORDER BY e.id
            """)
            rows = cur.fetchall()
        if rows:
            for eid, canon, alias, kind in rows:
                print(f"  entity_id={eid} canonical={canon!r} alias={alias!r} kind={kind}")
        else:
            print("  (no entity has izba, hut, or cabin in entity_aliases)")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("=" * 70)
    print("5. page_entity_mentions: surfaces for OSS entities (76707, 84902)")
    print("   PEM seeds from THIS table — if izba/hut/cabin not here, PEM won't use them")
    print("=" * 70)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, surface_norm, collection_slug,
                       COUNT(DISTINCT page_id) AS page_count
                FROM page_entity_mentions
                WHERE entity_id IN (76707, 84902)
                  AND collection_slug IN ('venona', 'vassiliev')
                GROUP BY entity_id, surface_norm, collection_slug
                ORDER BY entity_id, page_count DESC, surface_norm
                LIMIT 40
            """)
            rows = cur.fetchall()
        if rows:
            for eid, surf, coll, cnt in rows:
                print(f"  entity_id={eid} surface_norm={surf!r} coll={coll} pages={cnt}")
        else:
            print("  (no rows — page_entity_mentions may be empty or use different schema)")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("=" * 70)
    print("6. page_entity_mentions: any izba, hut, cabin surfaces?")
    print("=" * 70)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, surface_norm, collection_slug,
                       COUNT(DISTINCT page_id) AS page_count
                FROM page_entity_mentions
                WHERE surface_norm IN ('izba', 'hut', 'cabin')
                  AND collection_slug IN ('venona', 'vassiliev')
                GROUP BY entity_id, surface_norm, collection_slug
                ORDER BY page_count DESC
            """)
            rows = cur.fetchall()
        if rows:
            for eid, surf, coll, cnt in rows:
                print(f"  entity_id={eid} surface_norm={surf!r} coll={coll} pages={cnt}")
        else:
            print("  (no izba/hut/cabin in page_entity_mentions — PEM cannot seed with them)")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("=" * 70)
    print("7. Jacob Golos entities (69114, 3431, 78775): surfaces in PEM")
    print("   (these ARE appearing in PEM seeds: sound, zvuk, tasin raisin, rasin)")
    print("=" * 70)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, surface_norm, collection_slug,
                       COUNT(DISTINCT page_id) AS page_count
                FROM page_entity_mentions
                WHERE entity_id IN (69114, 3431, 78775)
                  AND collection_slug IN ('venona', 'vassiliev')
                  AND surface_norm IN ('sound', 'sound zvuk', 'tasin raisin', 'zvuk', 'rasin')
                GROUP BY entity_id, surface_norm, collection_slug
                ORDER BY entity_id, page_count DESC
            """)
            rows = cur.fetchall()
        if rows:
            for eid, surf, coll, cnt in rows:
                print(f"  entity_id={eid} surface_norm={surf!r} coll={coll} pages={cnt}")
        else:
            print("  (no match — surfaces may be normalized differently)")
    except Exception as e:
        print(f"  Error: {e}")

    conn.close()
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
PEM lane seeds from page_entity_mentions (mention index), NOT entity_aliases.
If izba/hut/cabin are in entity_aliases but NOT in page_entity_mentions for
the OSS entity, PEM will never use them. The mention index must be populated
from document text where those surfaces were linked to the entity.

Fix options:
  1. Populate page_entity_mentions with izba/hut/cabin → Office of Strategic Services
     (if concordance/ingest has that mapping but didn't emit to PEM)
  2. Extend PEM to fall back to entity_aliases when page_entity_mentions has
     no alias surfaces for an entity (would require code change)
""")


if __name__ == "__main__":
    main()
