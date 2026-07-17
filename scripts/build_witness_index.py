#!/usr/bin/env python3
"""Construct a witness index from grand-jury / hearing transcript OCR text.

Grand jury transcripts of witness testimony have no table of contents mapping
witness -> pages. This detects each formal swearing-in event:

    <NAME>, called as a witness, having (first|previously) been [duly] sworn,
    testified as follows: BY MR. <EXAMINER>:

and emits witness name + start page + page span + date + examiner. The span of
an appearance runs to the page before the next appearance.

Input: a JSON list of per-page text (index 0 == PDF page 1), e.g. the Textract
cache produced by scripts/ocr_textract.ocr_pdf_parallel.

Usage:
    python -m scripts.build_witness_index --text-cache ocr_cache/brothman_textract.json \
        --out ocr_cache/brothman_witness_index.json
"""
import re
import json
import argparse
from typing import List, Dict, Optional

SWEAR = re.compile(r"called as a witness", re.I)
# real swearing-in is immediately followed (within ~80 chars) by the examiner
# header or the foreman administering the oath; a *reading* of prior testimony
# is followed by "[Testimony, pp..." or a bare quote.
FOLLOWS_REAL = re.compile(r"testif\w*\s+as\s+follows.{0,80}?(BY\s+M[RN]|THE\s+FOREMAN)", re.I | re.S)
READING_CUE = re.compile(r"(proceed to read|going to proceed and read|been read to the jury|\[Testimony)", re.I)

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"((?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}})", re.I)
EXAMINER_RE = re.compile(r"BY\s+(M[RN]\.?\s+[A-Z][A-Za-z]+)", re.I)

# A name token: ALLCAPS (>=2 letters) or De/Mc/Mac/O' prefixed (e.g. DeBUFF).
NAME_TOK = re.compile(r"^(?:[A-Z]{2,}|(?:De|Mc|Mac|O')[A-Z][A-Za-z]+)[.\-']?$")


def extract_name(before: str) -> Optional[str]:
    """Take the trailing run of name-like tokens immediately before the marker."""
    # window of last ~60 chars, tokenized on whitespace/newlines/commas
    window = before[-60:]
    toks = re.split(r"[\s,]+", window.strip())
    name_toks: List[str] = []
    for tok in reversed(toks):
        t = tok.strip(".,")
        if NAME_TOK.match(t):
            name_toks.insert(0, t)
        elif name_toks:
            break
    # drop a leading repeated surname label, e.g. "DeBuff BENEDICT DeBUFF"
    if len(name_toks) >= 2 and name_toks[0].lower() == name_toks[-1].lower():
        name_toks = name_toks[1:]
    return " ".join(name_toks) if name_toks else None


def build_index(pages: List[str]) -> List[Dict]:
    events = []
    for i, text in enumerate(pages):
        for m in SWEAR.finditer(text):
            tail = text[m.end():m.end() + 160]
            head = text[:m.start()]
            is_reading = bool(READING_CUE.search(text[max(0, m.start() - 120):m.start()]))
            is_real = bool(FOLLOWS_REAL.search(text[m.start():m.start() + 200])) and not is_reading
            if not is_real:
                continue
            name = extract_name(head) or "(unparsed)"
            date_m = DATE_RE.search(text[:m.start()]) or DATE_RE.search(tail)
            exam_m = EXAMINER_RE.search(tail)
            events.append({
                "witness": name,
                "start_page": i + 1,
                "date": date_m.group(1) if date_m else None,
                "examiner": exam_m.group(1).upper().replace("MN", "MR") if exam_m else None,
            })
    # compute end_page spans
    for j, ev in enumerate(events):
        ev["end_page"] = (events[j + 1]["start_page"] - 1) if j + 1 < len(events) else len(pages)
        ev["page_count"] = ev["end_page"] - ev["start_page"] + 1
    return events


def main():
    ap = argparse.ArgumentParser(description="Build witness index from transcript OCR text")
    ap.add_argument("--text-cache", required=True, help="JSON list of per-page text")
    ap.add_argument("--out", help="Write index JSON here")
    args = ap.parse_args()

    pages = json.load(open(args.text_cache, encoding="utf-8"))
    idx = build_index(pages)
    print(f"{'WITNESS':28} {'PAGES':>12}  {'DATE':16} EXAMINER")
    print("-" * 78)
    for ev in idx:
        span = f"{ev['start_page']}-{ev['end_page']}"
        print(f"{ev['witness'][:27]:28} {span:>12}  {(ev['date'] or ''):16} {ev['examiner'] or ''}")
    distinct = sorted({ev["witness"] for ev in idx})
    print(f"\n{len(idx)} appearances, {len(distinct)} distinct witnesses: {', '.join(distinct)}")
    if args.out:
        json.dump(idx, open(args.out, "w", encoding="utf-8"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
