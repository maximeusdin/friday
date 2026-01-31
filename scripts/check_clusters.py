#!/usr/bin/env python3
"""Quick check of pending clusters for smoke test."""
import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='neh', user='neh', password='neh')
cur = conn.cursor()

print("=== Pending Clusters by Recommendation ===")
cur.execute("""
    SELECT recommendation, COUNT(*) 
    FROM ocr_variant_clusters 
    WHERE status = 'pending' 
    GROUP BY recommendation
""")
for rec, count in cur.fetchall():
    print(f"  {rec}: {count}")

print("\n=== Top 15 Clusters (by priority) ===")
cur.execute("""
    SELECT cluster_id, proposed_canonical, canonical_entity_id, 
           variant_count, total_mentions, priority_score, recommendation
    FROM ocr_variant_clusters 
    WHERE status = 'pending' 
    ORDER BY priority_score DESC 
    LIMIT 15
""")

for row in cur.fetchall():
    cluster_id, proposed, entity_id, vars, mentions, priority, rec = row
    proposed_str = (proposed[:35] + '...') if proposed and len(proposed) > 35 else (proposed or '(none)')
    print(f"  {cluster_id[:10]} | {proposed_str:<40} | entity={entity_id or 'None':<6} | vars={vars} | mentions={mentions} | rec={rec}")

# Show a few with entity_id (good for APPROVE_MERGE)
print("\n=== Clusters WITH existing entity_id (good for APPROVE_MERGE) ===")
cur.execute("""
    SELECT cluster_id, proposed_canonical, canonical_entity_id, variant_count
    FROM ocr_variant_clusters 
    WHERE status = 'pending' AND canonical_entity_id IS NOT NULL
    ORDER BY priority_score DESC 
    LIMIT 5
""")
for row in cur.fetchall():
    print(f"  {row[0][:10]} | {row[1][:40] if row[1] else '(none)':<40} | entity_id={row[2]}")

# Show a few without entity_id (good for APPROVE_NEW_ENTITY)
print("\n=== Clusters WITHOUT entity_id (good for APPROVE_NEW_ENTITY) ===")
cur.execute("""
    SELECT cluster_id, proposed_canonical, variant_count
    FROM ocr_variant_clusters 
    WHERE status = 'pending' AND canonical_entity_id IS NULL
    ORDER BY priority_score DESC 
    LIMIT 5
""")
for row in cur.fetchall():
    print(f"  {row[0][:10]} | {row[1][:40] if row[1] else '(none)':<40}")

conn.close()
