#!/usr/bin/env python3
"""Check candidates that should be affected by smoke test decisions."""
import sys
sys.path.insert(0, '.')
import psycopg2
from retrieval.ocr_utils import compute_variant_key

conn = psycopg2.connect(host='localhost', port=5432, dbname='neh', user='neh', password='neh')
cur = conn.cursor()

# Get blocklisted variant keys
cur.execute("SELECT variant_key FROM ocr_variant_blocklist WHERE source = 'adjudication'")
blocked_keys = {row[0] for row in cur.fetchall()}
print(f"Blocked keys: {blocked_keys}")

# Get allowlisted variant keys
cur.execute("SELECT variant_key, entity_id FROM ocr_variant_allowlist WHERE source = 'adjudication'")
allowed = {row[0]: row[1] for row in cur.fetchall()}
print(f"Allowlisted keys: {list(allowed.keys())}")

# Check if any pending candidates match these
print("\n=== Pending candidates that should be BLOCKED ===")
cur.execute("""
    SELECT id, surface_norm, resolution_status
    FROM mention_candidates
    WHERE resolution_status = 'pending'
    LIMIT 1000
""")
blocked_count = 0
for id, surface_norm, status in cur.fetchall():
    vk = compute_variant_key(surface_norm)
    if vk in blocked_keys:
        print(f"  {id}: '{surface_norm}' -> key '{vk}' is BLOCKED")
        blocked_count += 1
        if blocked_count >= 5:
            print("  ...")
            break

print(f"\nTotal blocked candidates: {blocked_count}+")

print("\n=== Pending candidates that should AUTO-LINK ===")
cur.execute("""
    SELECT id, surface_norm, resolution_status
    FROM mention_candidates
    WHERE resolution_status = 'pending'
    LIMIT 1000
""")
allowed_count = 0
for id, surface_norm, status in cur.fetchall():
    vk = compute_variant_key(surface_norm)
    if vk in allowed:
        print(f"  {id}: '{surface_norm}' -> key '{vk}' -> entity {allowed[vk]}")
        allowed_count += 1
        if allowed_count >= 5:
            print("  ...")
            break

print(f"\nTotal allowlisted candidates: {allowed_count}+")

# Check queue items
print("\n=== Queue items that should be affected ===")
cur.execute("""
    SELECT id, surface_norm, status
    FROM mention_review_queue
    WHERE status = 'pending'
    LIMIT 500
""")
queue_blocked = 0
queue_allowed = 0
for id, surface_norm, status in cur.fetchall():
    vk = compute_variant_key(surface_norm)
    if vk in blocked_keys:
        queue_blocked += 1
    if vk in allowed:
        queue_allowed += 1

print(f"  Queue items that should be blocked: {queue_blocked}")
print(f"  Queue items that should auto-link: {queue_allowed}")

conn.close()
