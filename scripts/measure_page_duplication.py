#!/usr/bin/env python3
"""Measure near-duplicate page overlap between documents, for every collection.

Near-duplicate key: md5 of the first 400 alphanumeric-only lowercase characters of
pages.raw_text. Only pages whose normalized text is >= 200 characters count as
"text-bearing" -- shorter pages are stamps/blanks and collide trivially. That is
the key and floor behind the elizabeth_bentley measurement of 2026-09-14
(3933 text-bearing pages, 703 shared with another document, 18%).

The raw key over-reports, because FBI releases repeat standard forms verbatim:
the FOIPA "DELETED PAGE INFORMATION SHEET", the "BEST COPIES OBTAINABLE" notice,
a hearing's committee masthead. Those are real pages of each file, not evidence
that one document duplicates another. A key is therefore treated as BOILERPLATE
and excluded from the distinctive counts when either holds:

  * it appears more than once inside a single document (a form, not a unique page)
  * it appears in 4 or more distinct documents of the collection

A genuinely duplicated archival page shows up in exactly the omnibus scan and its
per-volume twin -- two documents, once each -- so it survives both filters.

Read-only; runs no writes. Usage:
  DATABASE_URL=... python scripts/measure_page_duplication.py
  DATABASE_URL=... python scripts/measure_page_duplication.py --pairs
  DATABASE_URL=... python scripts/measure_page_duplication.py --collection elizabeth_bentley --pairs
"""
import os
import sys

import psycopg2

MIN_NORM_LEN = 200      # normalized chars a page needs to count as text-bearing
KEY_PREFIX = 400        # normalized chars hashed into the near-duplicate key
MAX_DOCS_PER_KEY = 3    # a key in >= 4 documents of one collection is a standard form
MIN_PAIR_PAGES = 5      # pair detail threshold


def build(cur, only):
    cur.execute("SET statement_timeout = '1800s'")
    where = "AND c.slug = %s" if only else ""
    params = (only,) if only else ()
    cur.execute(f"""
        CREATE TEMP TABLE pk AS
        SELECT p.id AS page_id, p.document_id, d.collection_id, c.slug, p.page_seq,
               md5(substring(regexp_replace(lower(p.raw_text), '[^a-z0-9]', '', 'g')
                             from 1 for {KEY_PREFIX})) AS k
        FROM pages p
        JOIN documents d ON d.id = p.document_id
        JOIN collections c ON c.id = d.collection_id
        WHERE p.raw_text IS NOT NULL
          AND length(regexp_replace(lower(p.raw_text), '[^a-z0-9]', '', 'g')) >= {MIN_NORM_LEN}
          {where}
    """, params)
    cur.execute("CREATE INDEX ON pk (collection_id, k)")
    cur.execute("CREATE INDEX ON pk (document_id)")
    cur.execute("ANALYZE pk")

    cur.execute(f"""
        CREATE TEMP TABLE boiler AS
        SELECT collection_id, k
        FROM (
            SELECT collection_id, k,
                   count(DISTINCT document_id) AS ndocs,
                   max(per_doc) AS max_per_doc
            FROM (SELECT collection_id, k, document_id, count(*) AS per_doc
                  FROM pk GROUP BY 1, 2, 3) s
            GROUP BY collection_id, k
        ) t
        WHERE ndocs > {MAX_DOCS_PER_KEY} OR max_per_doc > 1
    """)
    cur.execute("CREATE INDEX ON boiler (collection_id, k)")
    cur.execute("ANALYZE boiler")

    # Pages sharing a key with a page in a different document of the same collection.
    for name, extra in (("dup_raw", ""),
                        ("dup", "AND NOT EXISTS (SELECT 1 FROM boiler bo "
                                "WHERE bo.collection_id = a.collection_id AND bo.k = a.k)")):
        cur.execute(f"""
            CREATE TEMP TABLE {name} AS
            SELECT a.* FROM pk a
            WHERE EXISTS (SELECT 1 FROM pk b
                          WHERE b.collection_id = a.collection_id
                            AND b.k = a.k AND b.document_id <> a.document_id)
              {extra}
        """)
        cur.execute(f"CREATE INDEX ON {name} (collection_id, k)")
        cur.execute(f"ANALYZE {name}")


def summary(cur):
    cur.execute("""
        SELECT c.slug,
               (SELECT count(*) FROM documents WHERE collection_id = c.id),
               (SELECT count(*) FROM pk      WHERE pk.collection_id = c.id),
               (SELECT count(*) FROM dup_raw WHERE dup_raw.collection_id = c.id),
               (SELECT count(*) FROM dup     WHERE dup.collection_id = c.id),
               (SELECT count(DISTINCT document_id) FROM dup WHERE dup.collection_id = c.id)
        FROM collections c
        ORDER BY 5 DESC, 1
    """)
    print(f"{'collection':32} {'docs':>5} {'text-pp':>8} {'raw dup':>8} "
          f"{'distinct dup':>13} {'pct':>6} {'docs':>5}")
    print("-" * 84)
    tb = tr = td = 0
    for slug, docs, bearing, raw, dist, touched in cur.fetchall():
        tb += bearing; tr += raw; td += dist
        if raw == 0:
            continue
        pct = 100.0 * dist / bearing if bearing else 0.0
        flag = "  <<<" if dist >= MIN_PAIR_PAGES else ""
        print(f"{slug:32} {docs:5} {bearing:8} {raw:8} {dist:13} {pct:5.1f}% {touched:5}{flag}")
    print("-" * 84)
    print(f"{'TOTAL':32} {'':5} {tb:8} {tr:8} {td:13} {100.0*td/max(tb,1):5.1f}%")


def pairs(cur):
    print(f"\n\n=== document pairs sharing >= {MIN_PAIR_PAGES} distinctive pages ===")
    cur.execute(f"""
        SELECT slug, lo, hi, count(*) AS shared_pages, count(DISTINCT k) AS shared_keys
        FROM (
            SELECT a.slug, a.k,
                   LEAST(a.document_id, b.document_id) AS lo,
                   GREATEST(a.document_id, b.document_id) AS hi
            FROM dup a
            JOIN dup b ON b.collection_id = a.collection_id
                      AND b.k = a.k AND b.document_id < a.document_id
        ) s
        GROUP BY slug, lo, hi
        HAVING count(*) >= {MIN_PAIR_PAGES}
        ORDER BY slug, shared_pages DESC
    """)
    rows = cur.fetchall()
    cur.execute("""
        SELECT d.id, d.source_name, d.size_bytes,
               (SELECT count(*) FROM pages p WHERE p.document_id = d.id),
               (SELECT count(*) FROM pk WHERE pk.document_id = d.id)
        FROM documents d
    """)
    info = {r[0]: r[1:] for r in cur.fetchall()}
    cur_slug = None
    for slug, lo, hi, n, nk in rows:
        if slug != cur_slug:
            print(f"\n-- {slug}")
            cur_slug = slug
        for did in (lo, hi):
            name, size, pp, tp = info[did]
            cov = 100.0 * nk / tp if tp else 0.0
            tag = "lo" if did == lo else "hi"
            print(f"   [{tag}] doc {did:5} {pp:5}pp {tp:5} text-pp  "
                  f"{(size or 0)/1e6:7.1f}MB  {nk:4} keys = {cov:5.1f}% of its text pages  {name}")
        print(f"        -> {n} shared pages / {nk} distinct keys\n")


def main():
    args = sys.argv[1:]
    only = args[args.index("--collection") + 1] if "--collection" in args else None
    conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"))
    cur = conn.cursor()
    build(cur, only)
    summary(cur)
    if "--pairs" in args:
        pairs(cur)


if __name__ == "__main__":
    main()
