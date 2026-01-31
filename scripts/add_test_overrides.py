#!/usr/bin/env python3
"""Add real overrides for testing end-to-end."""

import psycopg2
import os

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "neh"),
    user=os.getenv("DB_USER", "neh"),
    password=os.getenv("DB_PASS", "neh"),
)
cur = conn.cursor()

print("Adding real overrides...")
print()

# 1. BAN surface 'well' - too common, causes false positives
print('1. Banning surface "well" (global) - too common English word')
cur.execute("""
    INSERT INTO entity_alias_overrides (surface_norm, scope, banned, note)
    VALUES ('well', 'global', TRUE, 'Common English word, causes false positives')
    ON CONFLICT DO NOTHING
""")
print("   Done.")

# 2. BAN entity 65411 (Well) for any surface containing 'well'
# Actually this is redundant since we banned the surface entirely
# Let's add a PREFER instead for something useful

# 3. For a collision that can be resolved with PREFER:
# Let's check 'fitin' - 1090 items
print()
print('2. Checking fitin collision candidates...')
cur.execute("""
    SELECT candidate_entity_ids, COUNT(*) as cnt
    FROM mention_review_queue
    WHERE surface_norm = 'fitin' AND status = 'pending'
    GROUP BY candidate_entity_ids
    ORDER BY cnt DESC
    LIMIT 3
""")
fitin_rows = cur.fetchall()
for row in fitin_rows:
    print(f"   Candidates {row[0]}: {row[1]} occurrences")
    if row[0]:
        cur.execute("SELECT id, canonical_name, entity_type FROM entities WHERE id = ANY(%s)", (row[0],))
        for ent in cur.fetchall():
            print(f"      - {ent[0]}: {ent[1]} ({ent[2]})")

# Pick the most common candidate set and prefer the main entity
if fitin_rows and fitin_rows[0][0]:
    # Check which entity is the "real" Fitin
    cur.execute("""
        SELECT id, canonical_name, entity_type 
        FROM entities 
        WHERE canonical_name ILIKE '%fitin%' AND entity_type = 'person'
        ORDER BY canonical_name
        LIMIT 5
    """)
    print()
    print("   Fitin-related entities:")
    for ent in cur.fetchall():
        print(f"      {ent[0]}: {ent[1]} ({ent[2]})")

conn.commit()

# Verify overrides
print()
print("=== Current overrides ===")
cur.execute("""
    SELECT surface_norm, scope, forced_entity_id, banned_entity_id, banned, note
    FROM entity_alias_overrides
    ORDER BY surface_norm
""")
for row in cur.fetchall():
    if row[4]:
        action = "BAN SURFACE"
    elif row[3]:
        action = f"BAN ENTITY {row[3]}"
    elif row[2]:
        action = f"PREFER ENTITY {row[2]}"
    else:
        action = "UNKNOWN"
    print(f"  {row[0]} ({row[1] or 'global'}): {action}")
    if row[5]:
        print(f"      Note: {row[5]}")

conn.close()
print()
print("Done!")
