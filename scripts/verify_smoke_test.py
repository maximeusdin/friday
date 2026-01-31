#!/usr/bin/env python3
"""Verify smoke test results."""
import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='neh', user='neh', password='neh')
cur = conn.cursor()

print("=== A) Review Events for smoke_test ===")
cur.execute("""
    SELECT decision, COUNT(*)
    FROM ocr_review_events
    WHERE reviewer = 'smoke_test'
    GROUP BY decision
""")
for dec, count in cur.fetchall():
    print(f"  {dec}: {count}")

print("\n=== B) Blocklist entries from adjudication ===")
cur.execute("""
    SELECT COUNT(*), source
    FROM ocr_variant_blocklist
    WHERE source = 'adjudication'
    GROUP BY source
""")
rows = cur.fetchall()
if rows:
    for count, source in rows:
        print(f"  {source}: {count} entries")
else:
    print("  No entries found")

# Show actual blocked variants
cur.execute("""
    SELECT variant_key, reason 
    FROM ocr_variant_blocklist 
    WHERE source = 'adjudication'
    LIMIT 10
""")
print("\n  Blocked variants:")
for vk, reason in cur.fetchall():
    print(f"    {vk}: {reason}")

print("\n=== C) New entity created (APPROVE_NEW_ENTITY) ===")
cur.execute("""
    SELECT entity_id, decision, payload
    FROM ocr_review_events
    WHERE decision = 'APPROVE_NEW_ENTITY' AND reviewer = 'smoke_test'
    ORDER BY id DESC
    LIMIT 5
""")
for entity_id, decision, payload in cur.fetchall():
    print(f"  entity_id={entity_id}, decision={decision}")
    if payload:
        print(f"    payload: {payload}")

# Verify entity exists
cur.execute("""
    SELECT id, canonical_name, entity_type 
    FROM entities 
    WHERE id = 65787
""")
row = cur.fetchone()
if row:
    print(f"\n  Verified: Entity {row[0]} exists: '{row[1]}' (type={row[2]})")

print("\n=== D) Allowlist entries from adjudication ===")
cur.execute("""
    SELECT variant_key, entity_id, reason
    FROM ocr_variant_allowlist
    WHERE source = 'adjudication'
    ORDER BY id DESC
    LIMIT 10
""")
print("  Allowlist entries:")
for vk, eid, reason in cur.fetchall():
    print(f"    {vk} -> entity {eid}: {reason}")

print("\n=== E) Cluster status changes ===")
cur.execute("""
    SELECT cluster_id, status, review_decision, reviewed_by
    FROM ocr_variant_clusters
    WHERE reviewed_by = 'smoke_test'
""")
for cid, status, decision, reviewer in cur.fetchall():
    print(f"  {cid[:16]}: status={status}, decision={decision}, by={reviewer}")

conn.close()
