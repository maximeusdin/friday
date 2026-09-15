#!/usr/bin/env python3
"""Confirm two documents are the same scan by testing page-for-page alignment.

Needed because the 0.60 6-gram containment test used elsewhere is a same-OCR test.
The two solo ingests of 2026-02-01/02 are the same page images run through OCR twice,
and the text diverges enough that most pages fall under 0.60 even though the scans are
identical. What survives that divergence is the ALIGNMENT: page i of A matches page i
of B far better than it matches any other page of B.

For each pair this reports mean 6-gram Jaccard on the diagonal (page i vs page i)
against a random off-diagonal baseline. A diagonal mean an order of magnitude above
baseline, on documents with equal page counts, means the same scan ingested twice.

Read-only. Usage:
  DATABASE_URL=... python scripts/check_aligned_duplicates.py 75:90 93:195
  DATABASE_URL=... python scripts/check_aligned_duplicates.py --solo-pairs
"""
import argparse
import os
import random
import re

import psycopg2

NGRAM = 6
WORD = re.compile(r"[a-z0-9]+")


def shingles(text):
    w = WORD.findall((text or "").lower())
    if len(w) < NGRAM:
        return set()
    return {hash(" ".join(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pages_of(cur, doc_id):
    cur.execute("""SELECT page_seq, raw_text FROM pages WHERE document_id=%s ORDER BY page_seq""",
                (doc_id,))
    return [(seq, shingles(t)) for seq, t in cur.fetchall()]


SOLO_PAIRS = [(57, 70), (75, 90), (93, 195), (96, 219), (98, 220), (108, 225), (109, 230),
              (112, 267), (80, 275), (114, 326), (115, 336), (120, 370), (122, 373), (83, 388),
              (125, 390), (126, 391), (86, 392), (132, 393), (136, 394), (142, 395), (144, 396),
              (397, 399)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*")
    ap.add_argument("--solo-pairs", action="store_true")
    a = ap.parse_args()
    pairs = SOLO_PAIRS if a.solo_pairs else [tuple(int(x) for x in p.split(":")) for p in a.pairs]

    conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"))
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '1800s'")
    cur.execute("SELECT id, source_name FROM documents")
    names = dict(cur.fetchall())
    rng = random.Random(0)

    print(f"{'A':>6} {'B':>6} {'pp A':>5} {'pp B':>5} {'diag':>7} {'offdiag':>8} {'ratio':>7} "
          f"{'>=.30':>7}  verdict")
    print("-" * 92)
    same = []
    for x, y in pairs:
        A, B = pages_of(cur, x), pages_of(cur, y)
        n = min(len(A), len(B))
        diag = [jaccard(A[i][1], B[i][1]) for i in range(n) if A[i][1] and B[i][1]]
        off = []
        for _ in range(min(400, n * 4)):
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j and A[i][1] and B[j][1]:
                off.append(jaccard(A[i][1], B[j][1]))
        dm = sum(diag) / len(diag) if diag else 0.0
        om = sum(off) / len(off) if off else 0.0
        strong = sum(1 for d in diag if d >= 0.30)
        ratio = dm / om if om else float("inf")
        ok = len(A) == len(B) and dm >= 0.15 and ratio >= 5
        if ok:
            same.append((x, y))
        print(f"{x:6} {y:6} {len(A):5} {len(B):5} {dm:7.3f} {om:8.4f} {ratio:7.1f} "
              f"{100.0*strong/max(len(diag),1):6.0f}%  {'SAME SCAN' if ok else 'inconclusive'}")
    print(f"\n{len(same)}/{len(pairs)} pairs confirmed same-scan duplicates")
    if same:
        print("pairs:", ", ".join(f"{x}:{y}" for x, y in same))


if __name__ == "__main__":
    main()
