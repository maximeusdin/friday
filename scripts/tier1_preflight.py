"""Read-only preflight for Tier-1 re-transcription: manifests + chunk-shape checks.

For each collection: build a manifest (doc_id, source_name, pages, pdf_url,
pipeline_version, collection_slug); test the text-walk chunk mapper on the two
largest docs; report max chunk byte-size (oversized-index risk).
"""
import json
import os
import sys
import urllib.parse

import psycopg2

sys.path.insert(0, ".")
from scripts.load_transcript import map_chunks_best  # noqa: E402

COLLECTIONS = {
    "rosenberg": "rosenberg_v1",
    "solo": "solo_v1_memo",
    "hiss_chambers": "hiss_chambers_v1",
    "david_greenglass": "david_greenglass_v1",
    "david_ruth_greenglass": "david_ruth_greenglass_v1",
}

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
all_docs = []
for slug, pv in COLLECTIONS.items():
    cur.execute("""
        SELECT d.id, d.source_name, d.source_ref, count(p.id)
        FROM documents d
        JOIN collections col ON col.id = d.collection_id AND col.slug = %s
        JOIN pages p ON p.document_id = d.id
        GROUP BY d.id ORDER BY count(p.id) DESC
    """, (slug,))
    docs = [{"doc_id": r[0], "source_name": r[1], "pages": r[3],
             "pdf_url": "https://fridayarchive.org/" + urllib.parse.quote(r[2] or ""),
             "collection_slug": slug, "pipeline_version": pv} for r in cur.fetchall()]
    all_docs.extend(docs)

    # Chunk stats for the collection's pipeline.
    cur.execute("""
        SELECT count(*), max(length(c.text)), percentile_cont(0.5) WITHIN GROUP (ORDER BY length(c.text))
        FROM chunks c JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE cm.collection_slug = %s AND c.pipeline_version = %s
    """, (slug, pv))
    nch, maxlen, medlen = cur.fetchone()

    # Mapper test on the two largest docs.
    verdicts = []
    for d in docs[:2]:
        cur.execute("SELECT page_seq, raw_text FROM pages WHERE document_id = %s ORDER BY page_seq", (d["doc_id"],))
        pages = cur.fetchall()
        cur.execute("""
            SELECT c.id, c.text FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE cm.document_id = %s AND c.pipeline_version = %s ORDER BY c.id
        """, (d["doc_id"], pv))
        chunks = cur.fetchall()
        if not chunks:
            verdicts.append(f"doc {d['doc_id']}: NO CHUNKS for pv")
            continue
        res, strat = map_chunks_best(pages, chunks)
        verdicts.append(f"doc {d['doc_id']}[{strat}]: {len(res.mapped)}/{len(chunks)} mapped, "
                        f"{len(res.unmapped)} unmapped, {len(res.leftover_pages)} leftover")
    print(f"{slug:24s} {len(docs):>3} docs {sum(d['pages'] for d in docs):>6} pages | "
          f"{nch} chunks, max {maxlen}B med {int(medlen)}B | " + "; ".join(verdicts))

out = "data/transcripts/batch_tier1/manifest_tier1.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(all_docs, open(out, "w"), indent=1)
print(f"\nmanifest: {len(all_docs)} docs, {sum(d['pages'] for d in all_docs)} pages -> {out}")
