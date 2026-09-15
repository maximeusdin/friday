#!/usr/bin/env python3
"""Corpus-wide sweep for duplicated documents, tolerant of OCR variance.

measure_page_duplication.py keys pages by an exact md5 of their first 400 normalized
characters. That is a lower bound: two independent scans of the same FBI file OCR
differently, so the exact key silently misses most cross-scan duplication (joel_barr
972/978 shows 22 pages by exact key and 81 by 6-gram containment).

This sweep instead sketches every text-bearing page with a bottom-k minhash over word
6-grams, indexes the sketches per collection, and reports document pairs whose pages
match. Pairs it surfaces are candidates; verify_duplicate_pairs.py then measures true
6-gram containment on the ones that matter.

Boilerplate (FOIPA deleted-page sheets, "BEST COPIES OBTAINABLE" notices, committee
mastheads) is suppressed by dropping sketch hashes that occur in too many pages of a
collection, and by ignoring pages whose text repeats inside their own document.

Read-only. Usage:
  DATABASE_URL=... python scripts/sweep_duplicate_documents.py
  DATABASE_URL=... python scripts/sweep_duplicate_documents.py --collection solo
"""
import argparse
import os
import re
from collections import Counter, defaultdict

import psycopg2

NGRAM = 6
SKETCH_K = 16          # bottom-k minhash size per page
MIN_SHARED = 5         # sketch hashes two pages must share to be a candidate (~J >= .3)
MAX_DF = 8             # a sketch hash in more than this many pages of a collection is boilerplate
MIN_NORM_LEN = 200
MIN_PAIR_PAGES = 4     # report document pairs with at least this many matched pages
WORD = re.compile(r"[a-z0-9]+")


def sketch(text):
    w = WORD.findall(text.lower())
    if len(w) < NGRAM:
        return ()
    hs = {hash(" ".join(w[i:i + NGRAM])) & 0xFFFFFFFFFFFF for i in range(len(w) - NGRAM + 1)}
    return tuple(sorted(hs)[:SKETCH_K])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection")
    ap.add_argument("--min-pages", type=int, default=MIN_PAIR_PAGES)
    a = ap.parse_args()

    conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"))
    cur = conn.cursor(name="pagestream")
    cur.itersize = 2000
    where = "AND c.slug = %s" if a.collection else ""
    cur.execute(f"""
        SELECT c.slug, d.id, p.id, p.page_seq, p.raw_text
        FROM pages p
        JOIN documents d ON d.id = p.document_id
        JOIN collections c ON c.id = d.collection_id
        WHERE p.raw_text IS NOT NULL
          AND length(regexp_replace(lower(p.raw_text), '[^a-z0-9]', '', 'g')) >= {MIN_NORM_LEN}
          {where}
        ORDER BY c.slug
    """, (a.collection,) if a.collection else ())

    by_coll = defaultdict(list)      # slug -> [(doc_id, page_id, page_seq, sketch)]
    n = 0
    for slug, doc_id, page_id, seq, txt in cur:
        s = sketch(txt)
        if s:
            by_coll[slug].append((doc_id, page_id, seq, s))
        n += 1
    cur.close()
    print(f"sketched {n} text-bearing pages across {len(by_coll)} collections "
          f"({NGRAM}-gram bottom-{SKETCH_K} minhash)\n")

    meta_cur = conn.cursor()
    meta_cur.execute("""
        SELECT d.id, d.source_name, d.size_bytes,
               (SELECT count(*) FROM pages WHERE document_id = d.id)
        FROM documents d
    """)
    info = {r[0]: r[1:] for r in meta_cur.fetchall()}

    findings = []
    for slug, pages in sorted(by_coll.items()):
        df = Counter()
        for _, _, _, s in pages:
            df.update(s)

        idx = defaultdict(list)
        for i, (_, _, _, s) in enumerate(pages):
            for h in s:
                if df[h] <= MAX_DF:
                    idx[h].append(i)

        # Count, per document pair, how many pages of each side matched the other.
        pair_pages = defaultdict(lambda: (set(), set()))
        for i, (doc_i, pid_i, _, s) in enumerate(pages):
            hits = Counter()
            for h in s:
                if df[h] > MAX_DF:
                    continue
                for j in idx[h]:
                    if pages[j][0] != doc_i:
                        hits[j] += 1
            for j, c in hits.items():
                if c < MIN_SHARED:
                    continue
                doc_j = pages[j][0]
                lo, hi = (doc_i, doc_j) if doc_i < doc_j else (doc_j, doc_i)
                a_set, b_set = pair_pages[(lo, hi)]
                (a_set if doc_i == lo else b_set).add(pid_i)

        for (lo, hi), (a_set, b_set) in pair_pages.items():
            matched = max(len(a_set), len(b_set))
            if matched < a.min_pages:
                continue
            lo_total = sum(1 for p in pages if p[0] == lo)
            hi_total = sum(1 for p in pages if p[0] == hi)
            findings.append((
                max(len(a_set) / max(lo_total, 1), len(b_set) / max(hi_total, 1)),
                slug, lo, hi, len(a_set), lo_total, len(b_set), hi_total))

    findings.sort(reverse=True)
    print(f"{'collection':24} {'doc A':>6} {'A matched':>16} {'doc B':>6} {'B matched':>16}  worst-case")
    print("-" * 104)
    for frac, slug, lo, hi, na, ta, nb, tb in findings:
        ln, lsz, lpp = info[lo]
        hn, hsz, hpp = info[hi]
        print(f"{slug:24} {lo:6} {na:5}/{ta:<5} ({100.0*na/max(ta,1):5.1f}%) "
              f"{hi:6} {nb:5}/{tb:<5} ({100.0*nb/max(tb,1):5.1f}%)  {frac*100:5.1f}%")
        print(f"{'':24}   A: {lpp:5}pp {(lsz or 0)/1e6:7.1f}MB  {ln}")
        print(f"{'':24}   B: {hpp:5}pp {(hsz or 0)/1e6:7.1f}MB  {hn}")
    if not findings:
        print("(no document pairs above threshold)")


if __name__ == "__main__":
    main()
