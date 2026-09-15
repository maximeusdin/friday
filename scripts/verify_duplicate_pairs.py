#!/usr/bin/env python3
"""Fuzzy page-level containment between two documents (or a hub doc vs its collection).

The md5-prefix key used by measure_page_duplication.py is exact: one OCR character
difference in the first 400 normalized characters breaks it. Between two independent
scans of the same FBI file that happens constantly, so the exact key is a lower bound
only. This script uses the measure migration 0077 relied on -- word 6-gram containment
-- to say how much of document B actually survives inside document A.

For every text-bearing page of B it finds the best-matching page of A and reports
containment = |shingles(B_page) & shingles(best A_page)| / |shingles(B_page)|.
A page is "covered" at >= 0.60, matching 0077's threshold.

Read-only. Usage:
  DATABASE_URL=... python scripts/verify_duplicate_pairs.py --a 186 --b 188
  DATABASE_URL=... python scripts/verify_duplicate_pairs.py --hub 1061 --collection elizabeth_bentley
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import psycopg2

NGRAM = 6
COVERED = 0.60
MIN_NORM_LEN = 200
WORD = re.compile(r"[a-z0-9]+")


def shingles(text):
    w = WORD.findall(text.lower())
    if len(w) < NGRAM:
        return set()
    return {hash(" ".join(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def load(cur, doc_id):
    cur.execute("""
        SELECT id, page_seq, pdf_page_number, raw_text FROM pages
        WHERE document_id = %s AND raw_text IS NOT NULL
          AND length(regexp_replace(lower(raw_text), '[^a-z0-9]', '', 'g')) >= %s
        ORDER BY page_seq
    """, (doc_id, MIN_NORM_LEN))
    out = []
    for pid, seq, pdfno, txt in cur.fetchall():
        s = shingles(txt)
        if s:
            out.append((pid, seq, pdfno, s, len(txt)))
    return out


def index_of(pages):
    idx = defaultdict(list)
    for i, (_, _, _, s, _) in enumerate(pages):
        for h in s:
            idx[h].append(i)
    return idx


def compare(a_pages, a_idx, b_pages):
    """Yield (b_page, best_a_page, containment) for each page of B."""
    for pid, seq, pdfno, s, ln in b_pages:
        hits = Counter()
        for h in s:
            for i in a_idx.get(h, ()):
                hits[i] += 1
        if not hits:
            yield (pid, seq, pdfno, ln), None, 0.0
            continue
        best_i, n = hits.most_common(1)[0]
        yield (pid, seq, pdfno, ln), a_pages[best_i], n / len(s)


def quality(cur, doc_id):
    cur.execute("""
        SELECT count(*), avg(ch.alpha_ratio), avg(ch.garbage_score),
               sum(CASE WHEN ch.is_trusted_text THEN 1 ELSE 0 END)
        FROM chunks ch
        JOIN chunk_pages cp ON cp.chunk_id = ch.id
        JOIN pages p ON p.id = cp.page_id
        WHERE p.document_id = %s
    """, (doc_id,))
    n, alpha, garbage, trusted = cur.fetchone()
    cur.execute("""
        SELECT count(*), avg(length(raw_text)), sum(length(raw_text))
        FROM pages WHERE document_id = %s AND raw_text IS NOT NULL
    """, (doc_id,))
    pp, avglen, total = cur.fetchone()
    return dict(chunks=n or 0, alpha=alpha, garbage=garbage, trusted=trusted or 0,
                pages=pp or 0, avglen=avglen, total_chars=total or 0)


def describe(cur, doc_id):
    cur.execute("""SELECT d.source_name, d.size_bytes,
                          (SELECT count(*) FROM pages WHERE document_id=d.id)
                   FROM documents d WHERE d.id=%s""", (doc_id,))
    return cur.fetchone()


def report_pair(cur, a_id, b_id, verbose=True):
    a_name, a_size, a_pp = describe(cur, a_id)
    b_name, b_size, b_pp = describe(cur, b_id)
    a_pages = load(cur, a_id)
    b_pages = load(cur, b_id)
    a_idx = index_of(a_pages)

    res = list(compare(a_pages, a_idx, b_pages))
    covered = [r for r in res if r[2] >= COVERED]
    partial = [r for r in res if 0.20 <= r[2] < COVERED]
    pct = 100.0 * len(covered) / len(res) if res else 0.0

    print(f"\n  A = doc {a_id:5} {a_pp:5}pp {(a_size or 0)/1e6:7.1f}MB  {a_name}")
    print(f"  B = doc {b_id:5} {b_pp:5}pp {(b_size or 0)/1e6:7.1f}MB  {b_name}")
    print(f"  {len(covered)}/{len(res)} of B's text pages are >= {COVERED:.0%} contained in A "
          f"({pct:.1f}%); {len(partial)} more at 0.20-0.60")

    # Is the match a constant page offset (a clean excerpt) or scattered?
    offs = Counter(r[1][1] - r[0][1] for r in covered if r[1])
    if offs:
        top, n = offs.most_common(1)[0]
        print(f"  dominant page offset A-B = {top:+d} on {n}/{len(covered)} covered pages "
              f"({100.0*n/len(covered):.0f}%)")
    if verbose and res:
        qa, qb = quality(cur, a_id), quality(cur, b_id)
        for tag, did, q in (("A", a_id, qa), ("B", b_id, qb)):
            print(f"  [{tag}] doc {did}: {q['chunks']:5} chunks  alpha={q['alpha'] or 0:.3f}  "
                  f"garbage={q['garbage'] or 0:.3f}  trusted={q['trusted']}/{q['chunks']}  "
                  f"avg {q['avglen'] or 0:6.0f} chars/page  total {q['total_chars']/1e6:.2f}M chars")
    return pct, len(covered), len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, help="document kept (the superset)")
    ap.add_argument("--b", type=int, help="document tested for containment in A")
    ap.add_argument("--hub", type=int, help="test every other doc of a collection against this one")
    ap.add_argument("--collection")
    ap.add_argument("--min", type=float, default=0.0, help="only print docs with >= this %% covered")
    a = ap.parse_args()

    conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"))
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '1800s'")

    if a.hub:
        cur.execute("""SELECT d.id FROM documents d JOIN collections c ON c.id=d.collection_id
                       WHERE c.slug=%s AND d.id<>%s ORDER BY d.id""", (a.collection, a.hub))
        others = [r[0] for r in cur.fetchall()]
        print(f"=== containment of each {a.collection} document inside hub doc {a.hub} "
              f"({len(others)} documents tested, {NGRAM}-gram, covered >= {COVERED}) ===")
        rows = []
        hub_pages = load(cur, a.hub)
        hub_idx = index_of(hub_pages)
        print(f"  hub: {len(hub_pages)} text-bearing pages indexed\n")
        for did in others:
            b_pages = load(cur, did)
            if not b_pages:
                continue
            res = list(compare(hub_pages, hub_idx, b_pages))
            cov = sum(1 for r in res if r[2] >= COVERED)
            pct = 100.0 * cov / len(res)
            name, size, pp = describe(cur, did)
            rows.append((pct, cov, len(res), did, pp, size, name))
        rows.sort(reverse=True)
        print(f"  {'doc':>6} {'pp':>5} {'text-pp':>8} {'covered':>8} {'pct':>7}   name")
        for pct, cov, tot, did, pp, size, name in rows:
            if pct < a.min:
                continue
            flag = " <<<" if pct >= 50 else ""
            print(f"  {did:6} {pp:5} {tot:8} {cov:8} {pct:6.1f}%   {name}{flag}")
        tot_cov = sum(r[1] for r in rows)
        tot_all = sum(r[2] for r in rows)
        print(f"\n  overall: {tot_cov}/{tot_all} ({100.0*tot_cov/max(tot_all,1):.1f}%) "
              f"of all other documents' text pages are contained in doc {a.hub}")
        return

    if not (a.a and a.b):
        sys.exit("need --a and --b, or --hub with --collection")
    print(f"=== {NGRAM}-gram containment (covered >= {COVERED}) ===")
    report_pair(cur, a.a, a.b)
    print("\n  --- reverse direction ---")
    report_pair(cur, a.b, a.a)


if __name__ == "__main__":
    main()
