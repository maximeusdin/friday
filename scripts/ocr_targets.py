#!/usr/bin/env python3
"""OCR a small set of specific PDFs into explicit cache paths, page-sharded.

Unlike preocr_solo (shards by FILE), this shards each file by PAGE RANGE so a
single huge file (e.g. the 2,552-page Albertson file) is split across processes.
Writes each (file, range) partial to disk (resumable), then merges per file into
the target cache JSON that ingest_dir_collection --ocr will reuse.
"""
import os, sys, json, tempfile
from concurrent.futures import ProcessPoolExecutor
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import fitz
import ocr_textract

REGION = "us-east-1"; DPI = 300; WORKERS = 8; PAGES_PER_SHARD = 450
R = r"C:\Users\maxim\friday"

TARGETS = [
    (os.path.join(R, r"data\raw\albertson\ALBERTSON, William - HQ 65-38100.pdf"),
     os.path.join(R, r"ocr_cache\albertson\ALBERTSON, William - HQ 65-38100.pdf.json")),
    (os.path.join(R, r"data\raw\eva_childs\Childs, Eva-HQ-1.pdf"),
     os.path.join(R, r"ocr_cache\eva_childs\Childs, Eva-HQ-1.pdf.json")),
    (os.path.join(R, r"data\raw\solo_addenda\Solo doc file list.PDF"),
     os.path.join(R, r"ocr_cache\solo\Solo doc file list.PDF.json")),
]

def pagecount(p):
    d = fitz.open(p); n = d.page_count; d.close(); return n

def run_shard(a):
    src, lo, hi, partial = a
    if os.path.exists(partial):
        try:
            if len(json.load(open(partial, encoding="utf-8"))) == hi - lo:
                return (os.path.basename(src), lo, hi, "cached")
        except Exception:
            pass
    full = ocr_textract.ocr_pdf_parallel(src, region=REGION, dpi=DPI, max_workers=WORKERS,
                                         page_indices=list(range(lo, hi)), progress=None)
    sub = full[lo:hi]
    os.makedirs(os.path.dirname(partial), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(partial), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(sub, fh)
    os.replace(tmp, partial)
    return (os.path.basename(src), lo, hi, "ocr")

if __name__ == "__main__":
    meta = []; shards = []
    for src, out in TARGETS:
        n = pagecount(src); meta.append((src, out, n))
        for lo in range(0, n, PAGES_PER_SHARD):
            hi = min(lo + PAGES_PER_SHARD, n)
            shards.append((src, lo, hi, out + f".part{lo:05d}"))
    print(f"{len(shards)} shards over {len(TARGETS)} files: "
          + ", ".join(f"{os.path.basename(s)}={n}p" for s, o, n in meta), flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run_shard, shards):
            print("  shard done:", r, flush=True)
    for src, out, n in meta:
        parts = sorted([s for s in shards if s[0] == src], key=lambda s: s[1])
        merged = []
        for (_, lo, hi, partial) in parts:
            merged.extend(json.load(open(partial, encoding="utf-8")))
        assert len(merged) == n, (out, len(merged), n)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(merged, fh)
        for (_, lo, hi, partial) in parts:
            try: os.remove(partial)
            except Exception: pass
        print(f"MERGED {os.path.basename(out)}: {len(merged)} pages", flush=True)
    print("OCR TARGETS COMPLETE", flush=True)
