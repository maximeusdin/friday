#!/usr/bin/env python3
"""
Show which entities an alias maps to.

Usage:
  python scripts/show_alias_entities.py OSS
  python scripts/show_alias_entities.py oss
  python scripts/show_alias_entities.py "Office of Strategic Services"
"""
import os
import re
import sys

import psycopg2


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/show_alias_entities.py <alias>")
        print("Example: python scripts/show_alias_entities.py OSS")
        sys.exit(1)

    alias = sys.argv[1]
    alias_lower = alias.lower()

    dsn = os.environ.get("DATABASE_URL") or "postgresql://neh:neh@localhost:5432/neh"
    conn = psycopg2.connect(dsn)

    with conn.cursor() as cur:
        # Normalize alias for alias_norm match (lower, strip punctuation, collapse spaces)
        norm = re.sub(r"[^\w\s]", "", alias_lower)
        norm = " ".join(norm.split()).strip()

        cur.execute("""
            SELECT e.id, e.canonical_name, e.entity_type, cs.slug, ea.alias
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.entity_id
            LEFT JOIN concordance_sources cs ON cs.id = ea.source_id
            WHERE LOWER(ea.alias) = %s
               OR ea.alias_norm = %s
            ORDER BY e.id, ea.alias
        """, (alias_lower, norm))

        rows = cur.fetchall()

    conn.close()

    if not rows:
        print(f"No entities found for alias: {alias!r}")
        return

    print(f"Alias {alias!r} maps to {len(set(r[0] for r in rows))} entity(ies):\n")
    for eid, canon, etype, source, a in rows:
        print(f"  entity_id={eid}")
        print(f"    canonical_name: {canon!r}")
        print(f"    entity_type: {etype}")
        print(f"    source: {source}")
        print(f"    alias: {a!r}")
        print()


if __name__ == "__main__":
    main()
