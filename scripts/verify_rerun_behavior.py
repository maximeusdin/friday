#!/usr/bin/env python3
"""
Verify that the resolver would correctly handle allowlisted/blocklisted variants.

This simulates what the resolver would do without actually modifying data.
"""
import sys
sys.path.insert(0, '.')
import psycopg2
from retrieval.ocr_utils import compute_variant_key

conn = psycopg2.connect(host='localhost', port=5432, dbname='neh', user='neh', password='neh')
cur = conn.cursor()

# Load blocklist
cur.execute("SELECT variant_key FROM ocr_variant_blocklist WHERE source = 'adjudication'")
blocklist = {row[0] for row in cur.fetchall()}

# Load allowlist
cur.execute("SELECT variant_key, entity_id FROM ocr_variant_allowlist WHERE source = 'adjudication'")
allowlist = {row[0]: row[1] for row in cur.fetchall()}

print("=== Smoke Test Verification ===")
print(f"Blocklist: {len(blocklist)} variant keys")
print(f"Allowlist: {len(allowlist)} variant keys")

# Get sample queue items
print("\n=== Simulating Resolver Behavior ===")

cur.execute("""
    SELECT id, surface_norm, status, candidate_entity_ids
    FROM mention_review_queue
    WHERE status = 'pending'
    LIMIT 100
""")

blocked_items = []
autolinked_items = []
unchanged_items = []

for id, surface_norm, status, candidate_ids in cur.fetchall():
    vk = compute_variant_key(surface_norm)
    
    if vk in blocklist:
        blocked_items.append((id, surface_norm, vk))
    elif vk in allowlist:
        autolinked_items.append((id, surface_norm, vk, allowlist[vk]))
    else:
        unchanged_items.append((id, surface_norm, vk))

print(f"\nWould BLOCK (never appear again): {len(blocked_items)}")
for id, surface, vk in blocked_items[:5]:
    print(f"  queue {id}: '{surface}' -> key '{vk}' BLOCKED")

print(f"\nWould AUTO-LINK (no queue, direct entity link): {len(autolinked_items)}")
for id, surface, vk, entity_id in autolinked_items[:5]:
    print(f"  queue {id}: '{surface}' -> key '{vk}' -> entity {entity_id}")

print(f"\nUnchanged (still needs review): {len(unchanged_items)}")

print("\n=== Verdict ===")
total_affected = len(blocked_items) + len(autolinked_items)
print(f"Total queue items affected by smoke test decisions: {total_affected}")
print(f"  - Would be blocked: {len(blocked_items)}")
print(f"  - Would auto-link: {len(autolinked_items)}")

if len(blocked_items) > 0 and len(autolinked_items) > 0:
    print("\n[PASS] Both BLOCK and ALLOWLIST paths have affected items")
else:
    print("\n[WARN] Not all paths have affected items")

# Verify the entities exist
print("\n=== Verifying Entity Links ===")
for vk, entity_id in list(allowlist.items())[:5]:
    cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
    row = cur.fetchone()
    if row:
        print(f"  '{vk}' -> entity {entity_id} '{row[0]}' [EXISTS]")
    else:
        print(f"  '{vk}' -> entity {entity_id} [MISSING!]")

conn.close()
