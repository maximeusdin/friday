"""Read-only inventory: per collection — docs, pages, extractor mix, chunk pipeline versions."""
import os, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""
    SELECT col.slug,
           count(DISTINCT d.id) AS docs,
           count(p.id) AS pages,
           array_agg(DISTINCT COALESCE(d.metadata->>'extractor', '?')) AS extractors
    FROM collections col
    JOIN documents d ON d.collection_id = col.id
    LEFT JOIN pages p ON p.document_id = d.id
    GROUP BY col.slug
    ORDER BY count(p.id) DESC
""")
rows = cur.fetchall()

cur.execute("""
    SELECT collection_slug, array_agg(DISTINCT pipeline_version)
    FROM chunk_metadata GROUP BY collection_slug
""")
pvs = dict(cur.fetchall())

print(f"{'slug':28s} {'docs':>5} {'pages':>7}  extractors | chunk_pvs")
for slug, docs, pages, ext in rows:
    pv = ",".join(v for v in (pvs.get(slug) or []) if v)
    print(f"{slug:28s} {docs:>5} {pages:>7}  {','.join(e or '?' for e in ext):24s} | {pv[:55]}")
