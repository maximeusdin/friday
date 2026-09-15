#!/usr/bin/env python3
"""Rank each confirmed duplicate pair on which copy to keep.

Both copies are the same scan, so the choice is purely which OCR pass recovered more
usable text. The score is legible characters = total raw_text characters x legibility,
where legibility is the share of alphabetic tokens that look like real words. Raw
character count alone would reward an OCR pass that emitted more garbage.

chunks.alpha_ratio and chunks.garbage_score are 0 for every row in prod, so they are
not used.

Read-only. Usage:
  DATABASE_URL=... python scripts/rank_duplicate_keep_drop.py --solo-pairs
  DATABASE_URL=... python scripts/rank_duplicate_keep_drop.py 186:188
"""
import argparse
import os
import re

import psycopg2

WORD = re.compile(r"[a-z0-9]+")
ALPHA = re.compile(r"^[a-z]{2,}$")
VOWEL = re.compile(r"[aeiouy]")
REPEAT = re.compile(r"(.)\1\1")

SOLO_PAIRS = [(57, 70), (75, 90), (93, 195), (96, 219), (98, 220), (108, 225), (109, 230),
              (112, 267), (80, 275), (114, 326), (115, 336), (120, 370), (122, 373), (83, 388),
              (125, 390), (126, 391), (86, 392), (132, 393), (136, 394), (142, 395), (144, 396),
              (397, 399)]


def profile(cur, doc_id):
    cur.execute("""SELECT d.source_name, d.source_ref, d.size_bytes,
                          (SELECT count(*) FROM pages WHERE document_id=d.id)
                   FROM documents d WHERE d.id=%s""", (doc_id,))
    name, ref, size, pp = cur.fetchone()
    cur.execute("SELECT raw_text FROM pages WHERE document_id=%s AND raw_text IS NOT NULL", (doc_id,))
    good = total = chars = textpp = 0
    for (t,) in cur.fetchall():
        chars += len(t)
        if len(re.sub(r"[^a-z0-9]", "", t.lower())) >= 200:
            textpp += 1
        for tok in WORD.findall(t.lower()):
            if ALPHA.match(tok):
                total += 1
                if VOWEL.search(tok) and not REPEAT.search(tok):
                    good += 1
    leg = good / total if total else 0.0
    return dict(name=name, ref=ref, size=size or 0, pp=pp, chars=chars,
                textpp=textpp, leg=leg, score=chars * leg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*")
    ap.add_argument("--solo-pairs", action="store_true")
    a = ap.parse_args()
    pairs = SOLO_PAIRS if a.solo_pairs else [tuple(int(x) for x in p.split(":")) for p in a.pairs]

    conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"))
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '1800s'")

    keeps, drops, drop_pages = [], [], 0
    print(f"{'keep':>6} {'drop':>6} {'legible chars':>26} {'text pages':>13} {'margin':>8}  file kept")
    print("-" * 110)
    for x, y in pairs:
        px, py = profile(cur, x), profile(cur, y)
        (kid, kp), (did, dp) = ((x, px), (y, py)) if px["score"] >= py["score"] else ((y, py), (x, px))
        margin = 100.0 * (kp["score"] - dp["score"]) / max(dp["score"], 1)
        keeps.append(kid)
        drops.append(did)
        drop_pages += dp["pp"]
        print(f"{kid:6} {did:6} {kp['score']/1000:9.0f}k vs {dp['score']/1000:8.0f}k "
              f"{kp['textpp']:6} vs {dp['textpp']:4} {margin:7.1f}%  {kp['name']}")
        print(f"{'':13} {'':26} {'':13} {'':8}  drop: {dp['name']}")
    print("-" * 110)
    print(f"\nkeep ({len(keeps)}): {sorted(keeps)}")
    print(f"drop ({len(drops)}): {sorted(drops)}")
    print(f"pages removed: {drop_pages}")


if __name__ == "__main__":
    main()
