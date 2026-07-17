#!/usr/bin/env python3
"""Pre-OCR only the scanned PDFs in the oscar_seborer dir (skip born-digital),
writing ocr_cache/oscar_seborer/<name>.json for ingest_dir_collection --ocr-auto."""
import os, sys, glob, json, tempfile
from concurrent.futures import ProcessPoolExecutor
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import fitz, ocr_textract

D = r"C:\Users\maxim\friday\data\raw\oscar_seborer"
CACHE = r"C:\Users\maxim\friday\ocr_cache\oscar_seborer"
os.makedirs(CACHE, exist_ok=True)

def has_text(f):
    d = fitz.open(f); n = d.page_count
    idxs = list(range(0, n, max(1, n//8)))[:8] if n else []
    avg = sum(len(d[i].get_text().strip()) for i in idxs)/max(1, len(idxs)); d.close()
    return avg >= 50

def pagecount(f):
    d = fitz.open(f); n = d.page_count; d.close(); return n

def ocr_one(f):
    name = os.path.basename(f)
    cache = os.path.join(CACHE, name + ".json")
    n = pagecount(f)
    if os.path.exists(cache):
        try:
            if len(json.load(open(cache, encoding="utf-8"))) == n:
                return (name, n, "cached")
        except Exception:
            pass
    pages = ocr_textract.ocr_pdf_parallel(f, region="us-east-1", dpi=300, max_workers=6, progress=None)
    fd, tmp = tempfile.mkstemp(dir=CACHE, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(pages, fh)
    os.replace(tmp, cache)
    return (name, n, "ocr")

if __name__ == "__main__":
    files = [f for f in glob.glob(os.path.join(D, "*.pdf")) if not has_text(f)]
    files.sort(key=lambda f: -pagecount(f))  # big first
    print(f"OCR-needed: {len(files)} files, {sum(pagecount(f) for f in files)} pages", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        for name, n, how in ex.map(ocr_one, files):
            done += 1
            print(f"  [{done}/{len(files)}] {how:6} {n:>4}p  {name[:60]}", flush=True)
    print("PREOCR SEBORER COMPLETE", flush=True)
