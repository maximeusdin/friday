#!/usr/bin/env python3
"""Two-way coverage between one 'hub' document and the rest of its collection.

Answers the question a duplicate-removal decision actually turns on: if an omnibus
scan and a set of per-volume scans overlap, which side can be dropped without losing
pages? Reports

  * forward: how much of each other document is contained in the hub
  * reverse: how much of the hub is covered by the UNION of all other documents

Uses word 6-gram containment at the 0.60 threshold migration 0077 used.

Read-only. Usage:
  DATABASE_URL=... python scripts/check_omnibus_coverage.py --hub 1061 --collection elizabeth_bentley
"""
import argparse
import os
import re
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
        SELECT id, page_seq, raw_text FROM pages
        WHERE document_id = %s AND raw_text IS NOT NULL
          AND length(regexp_replace(lower(raw_text), '[^a-z0-9]', '', 'g')) >= %s
        ORDER BY page_seq
    """, (doc_id, MIN_NORM_LEN))
    out = []
    for pid, seq, txt in cur.fetchall():
        s = shingles(txt)
        if s:
            out.append((pid, seq, s))
    return out


def index_of(pages):
    idx = defaultdict(list)
    for i, (_, _, s) in enumerate(pages):
        for h in s:
            idx[h].append(i)
    return idx


def best(idx, s):
    hits = Counter()
    for h in s:
        for i in idx.get(h, ()):
            hits[i] += 1
    if not hits:
        return None, 0.0
    i, n = hits.most_common(1)[0]
    return i, n / len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", type=int, required=True)
    ap.add_argument("--collection", required=True)
    a = ap.parse_args()

    conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"))
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '1800s'")
    cur.execute("""SELECT d.id, d.source_name, (SELECT count(*) FROM pages WHERE document_id=d.id)
                   FROM documents d JOIN collections c ON c.id=d.collection_id
                   WHERE c.slug=%s ORDER BY d.id""", (a.collection,))
    docs = cur.fetchall()
    names = {d[0]: (d[1], d[2]) for d in docs}
    others = [d[0] for d in docs if d[0] != a.hub]

    hub = load(cur, a.hub)
    hub_idx = index_of(hub)
    print(f"hub doc {a.hub} ({names[a.hub][0]}): {names[a.hub][1]}pp, {len(hub)} text-bearing\n")

    print("=== FORWARD: how much of each other document lives inside the hub ===")
    print(f"  {'doc':>6} {'pp':>5} {'text':>6} {'covered':>8} {'pct':>7}  name")
    hub_touched = set()
    f_cov = f_tot = 0
    rows = []
    for did in others:
        pages = load(cur, did)
        if not pages:
            continue
        cov = 0
        for _, _, s in pages:
            i, frac = best(hub_idx, s)
            if frac >= COVERED:
                cov += 1
                hub_touched.add(i)
        rows.append((100.0 * cov / len(pages), did, len(pages), cov))
        f_cov += cov
        f_tot += len(pages)
    for pct, did, tot, cov in sorted(rows, reverse=True):
        flag = " <<<" if pct >= 75 else ""
        print(f"  {did:6} {names[did][1]:5} {tot:6} {cov:8} {pct:6.1f}%  {names[did][0]}{flag}")
    print(f"\n  TOTAL {f_cov}/{f_tot} ({100.0*f_cov/max(f_tot,1):.1f}%) of the other documents' "
          f"text pages are inside the hub")
    print(f"  they account for {len(hub_touched)}/{len(hub)} "
          f"({100.0*len(hub_touched)/max(len(hub),1):.1f}%) of the hub's own text pages")

    print("\n=== REVERSE: how much of the hub is covered by the UNION of the others ===")
    union = []
    for did in others:
        for pid, seq, s in load(cur, did):
            union.append((did, seq, s))
    u_idx = index_of([(d, q, s) for d, q, s in union])
    cov = 0
    uncovered = []
    for pid, seq, s in hub:
        i, frac = best(u_idx, s)
        if frac >= COVERED:
            cov += 1
        else:
            uncovered.append((seq, frac))
    print(f"  {cov}/{len(hub)} ({100.0*cov/max(len(hub),1):.1f}%) of the hub's text pages are "
          f"reproduced somewhere in the other {len(others)} documents")
    print(f"  {len(uncovered)} hub pages are NOT reproduced anywhere else "
          f"-> dropping the hub would lose them")
    if uncovered:
        seqs = [u[0] for u in uncovered]
        print(f"  uncovered hub page_seq sample: {seqs[:25]}{' ...' if len(seqs) > 25 else ''}")


if __name__ == "__main__":
    main()
