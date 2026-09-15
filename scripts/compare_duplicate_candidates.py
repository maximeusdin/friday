#!/usr/bin/env python3
"""Side-by-side keep/drop comparison for candidate duplicate document pairs.

For each pair it reports 6-gram containment in both directions (the measure
migration 0077 used) plus the signals that decide which copy to keep:

  text pages recovered, total characters, and an OCR legibility score.

The legibility score is computed here rather than read from chunks.alpha_ratio /
chunks.garbage_score -- those columns exist but are 0 for every row in prod, so
they carry no signal. It is the share of alphabetic tokens that look like real
words (>= 2 chars, contains a vowel, no 3-in-a-row letter repeat); clean FBI
typescript scores ~0.90, Textract mush scores ~0.60.

Read-only. Usage:
  DATABASE_URL=... python scripts/compare_duplicate_candidates.py 186:188 1172:1173
  DATABASE_URL=... python scripts/compare_duplicate_candidates.py --solo-pairs
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
ALPHA = re.compile(r"^[a-z]{2,}$")
VOWEL = re.compile(r"[aeiouy]")
REPEAT = re.compile(r"(.)\1\1")


def shingles(text):
    w = WORD.findall(text.lower())
    if len(w) < NGRAM:
        return set()
    return {hash(" ".join(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def legibility(texts):
    good = total = 0
    for t in texts:
        for tok in WORD.findall(t.lower()):
            if not ALPHA.match(tok):
                continue
            total += 1
            if VOWEL.search(tok) and not REPEAT.search(tok):
                good += 1
    return good / total if total else 0.0


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
            out.append((pid, seq, s, txt))
    return out


def stats(cur, doc_id):
    cur.execute("""SELECT d.source_name, d.size_bytes, d.created_at,
                          (SELECT count(*) FROM pages WHERE document_id=d.id),
                          (SELECT coalesce(sum(length(raw_text)),0) FROM pages WHERE document_id=d.id),
                          (SELECT count(DISTINCT ch.id) FROM chunks ch
                             JOIN chunk_pages cp ON cp.chunk_id=ch.id
                             JOIN pages p ON p.id=cp.page_id WHERE p.document_id=d.id)
                   FROM documents d WHERE d.id=%s""", (doc_id,))
    return cur.fetchone()


def containment(a_pages, b_pages):
    idx = defaultdict(list)
    for i, (_, _, s, _) in enumerate(a_pages):
        for h in s:
            idx[h].append(i)
    cov = 0
    offs = Counter()
    for _, seq, s, _ in b_pages:
        hits = Counter()
        for h in s:
            for i in idx.get(h, ()):
                hits[i] += 1
        if hits:
            i, n = hits.most_common(1)[0]
            if n / len(s) >= COVERED:
                cov += 1
                offs[a_pages[i][1] - seq] += 1
    return cov, offs


def report(cur, a_id, b_id):
    a_name, a_sz, a_cr, a_pp, a_ch, a_ck = stats(cur, a_id)
    b_name, b_sz, b_cr, b_pp, b_ch, b_ck = stats(cur, b_id)
    A, B = load(cur, a_id), load(cur, b_id)
    ab, ab_off = containment(A, B)      # of B inside A
    ba, _ = containment(B, A)           # of A inside B
    la, lb = legibility(p[3] for p in A), legibility(p[3] for p in B)

    print(f"\n{'':2}{'':-<100}")
    print(f"  doc {a_id:5}  {a_pp:4}pp {len(A):4} text  {a_ch/1000:7.0f}k chars  "
          f"{(a_sz or 0)/1e6:6.1f}MB  {a_ck:4} chunks  legib {la:.3f}  {a_cr:%Y-%m-%d}  {a_name}")
    print(f"  doc {b_id:5}  {b_pp:4}pp {len(B):4} text  {b_ch/1000:7.0f}k chars  "
          f"{(b_sz or 0)/1e6:6.1f}MB  {b_ck:4} chunks  legib {lb:.3f}  {b_cr:%Y-%m-%d}  {b_name}")
    pab = 100.0 * ab / max(len(B), 1)
    pba = 100.0 * ba / max(len(A), 1)
    print(f"  containment: {ab}/{len(B)} of B is in A ({pab:.1f}%)   "
          f"{ba}/{len(A)} of A is in B ({pba:.1f}%)")
    if ab_off:
        off, n = ab_off.most_common(1)[0]
        print(f"  dominant page offset A-B = {off:+d} on {n}/{max(ab,1)} matched pages")

    # Recommendation: keep the side that loses nothing and recovers more text.
    if pab >= 90 and pba >= 90:
        keep, drop = (a_id, b_id) if (a_ch, la) >= (b_ch, lb) else (b_id, a_id)
        why = "mutual duplicates; keep the copy with more recovered text"
    elif pab >= 90:
        keep, drop, why = a_id, b_id, "B is contained in A"
    elif pba >= 90:
        keep, drop, why = b_id, a_id, "A is contained in B"
    else:
        print("  => PARTIAL overlap only - not a safe delete")
        return None
    print(f"  => KEEP {keep}, DROP {drop}  ({why})")
    return keep, drop


SOLO_PAIRS = [(57, 70), (75, 90), (93, 195), (96, 219), (98, 220), (108, 225), (109, 230),
              (112, 267), (80, 275), (114, 326), (115, 336), (120, 370), (122, 373), (83, 388),
              (125, 390), (126, 391), (86, 392), (132, 393), (136, 394), (142, 395), (144, 396),
              (397, 399)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", help="A:B document id pairs")
    ap.add_argument("--solo-pairs", action="store_true",
                    help="the 22 zero-padded/unpadded solo ingests")
    a = ap.parse_args()
    pairs = SOLO_PAIRS if a.solo_pairs else [tuple(int(x) for x in p.split(":")) for p in a.pairs]

    conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"))
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '1800s'")
    verdicts = []
    for x, y in pairs:
        v = report(cur, x, y)
        if v:
            verdicts.append(v)
    print(f"\n\n=== verdict: {len(verdicts)}/{len(pairs)} pairs are safe deletes ===")
    print("keep:", sorted(v[0] for v in verdicts))
    print("drop:", sorted(v[1] for v in verdicts))


if __name__ == "__main__":
    main()
