#!/usr/bin/env python3
"""Make scanned PDFs searchable: burn an invisible OCR text layer at word positions.

Why: the viewer's evidence-quote highlighting (EvidenceViewer.tsx) paints over the
PDF's embedded text layer via the CSS Custom Highlight API. Image-only scans
(fbi_solo, Morris/Jack Childs, Winton Burdett, pravdin, ...) have no layer, so no
highlight is possible and Ctrl+F in the viewer finds nothing. The corpus text in
Postgres (pages.raw_text) came from Textract LINE text with geometry discarded, so
this script re-runs Textract DetectDocumentText (~$1.50 / 1000 pages, us-east-1)
keeping WORD bounding boxes, and writes a copy of each PDF with the words drawn
invisibly (render mode 3) at their scan positions.

The full Textract block response is cached under ocr_cache_geom/ per PDF, so
re-running the script never re-bills OCR'd pages.

Output mirrors the input path under --out-root (default data/searchable/); originals
are never modified. Upload the outputs over the same S3 keys (data/raw/...) to make
the viewer pick them up without any DB or code change.

Usage:
  python scripts/make_searchable_pdfs.py --dir "data/raw/Morris Childs" --estimate
  python scripts/make_searchable_pdfs.py --dir "data/raw/Morris Childs" --glob "*HQ-6*"
  python scripts/make_searchable_pdfs.py --dir data/raw/fbi_solo --workers 8
"""
import argparse
import glob as globmod
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ocr_textract import make_client, render_page_image, RETRYABLE

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = Exception

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM_CACHE_ROOT = os.path.join(REPO, "ocr_cache_geom")

# A page whose embedded layer already has this much text is left untouched.
EXISTING_TEXT_THRESHOLD = 100

COST_PER_PAGE = 1.50 / 1000  # Textract DetectDocumentText


def detect_page_words(client, img_bytes: bytes, max_retries: int = 6):
    """Synchronous Textract on one page image; return list of WORD blocks as
    {"t": text, "bb": [left, top, width, height]} (relative 0-1 coords)."""
    delay = 1.0
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.detect_document_text(Document={"Bytes": img_bytes})
            return [
                {"t": b["Text"], "bb": [b["Geometry"]["BoundingBox"][k]
                                        for k in ("Left", "Top", "Width", "Height")]}
                for b in resp.get("Blocks", [])
                if b.get("BlockType") == "WORD" and b.get("Text")
            ]
        except ClientError as e:  # type: ignore[misc]
            code = e.response.get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
            last = e
            if code in RETRYABLE and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    if last:
        raise last
    return []


def page_needs_ocr(page: "fitz.Page") -> bool:
    return len(page.get_text().strip()) < EXISTING_TEXT_THRESHOLD


def _sanitize(word: str) -> str:
    """Base-14 helv is WinAnsi; replace unencodable chars so insert_text can't fail."""
    return word.encode("cp1252", "replace").decode("cp1252")


def overlay_words(page: "fitz.Page", words) -> int:
    """Draw invisible words at their scan positions. Returns words drawn.

    Textract bboxes are relative to the rendered (display-rotated) page image;
    map through derotation_matrix and pass rotate= so text runs the right way
    on rotated pages.
    """
    rect = page.rect  # display space (rotation applied)
    W, H = rect.width, rect.height
    drawn = 0
    for w in words:
        text = _sanitize(w["t"]).strip()
        if not text:
            continue
        left, top, bw, bh = w["bb"]
        x0, y0 = left * W, top * H
        wpt, hpt = max(bw * W, 0.5), max(bh * H, 0.5)
        natural = fitz.get_text_length(text, fontname="helv", fontsize=1.0)
        if natural <= 0:
            continue
        # Fit the word's rendered width to the scanned word's width so the
        # highlight box hugs the visual word; cap by height so short wide
        # boxes don't explode the glyph size.
        fs = min(wpt / natural, hpt * 1.6)
        if fs < 1.0:
            fs = 1.0
        baseline = fitz.Point(x0, y0 + hpt * 0.82)
        try:
            # Trailing space: makes extracted/copied text word-separated (pdf.js joins
            # items with no separator otherwise), so viewer Ctrl+F and quote matching
            # see normal spaced text. The space glyph is invisible and out-of-box harmless.
            page.insert_text(baseline * page.derotation_matrix, text + " ",
                             fontsize=fs, fontname="helv",
                             render_mode=3, rotate=page.rotation)
            drawn += 1
        except Exception:
            continue
    return drawn


def geom_cache_path(pdf_path: str) -> str:
    rel = os.path.relpath(os.path.abspath(pdf_path), REPO)
    return os.path.join(GEOM_CACHE_ROOT, rel.replace("\\", "/").replace("/", "__") + ".words.json.gz")


def load_geom_cache(pdf_path: str):
    p = geom_cache_path(pdf_path)
    if os.path.exists(p):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_geom_cache(pdf_path: str, cache) -> None:
    p = geom_cache_path(pdf_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def process_pdf(pdf_path: str, out_path: str, region: str, workers: int,
                dpi: int = 300, verbose: bool = True):
    """OCR (cached) + overlay one PDF. Returns (pages_total, pages_ocr_called, words)."""
    doc = fitz.open(pdf_path)
    need = [i for i in range(doc.page_count) if page_needs_ocr(doc[i])]
    cache = load_geom_cache(pdf_path)
    to_call = [i for i in need if str(i) not in cache]

    if to_call:
        # Thread-local doc + client (a fitz.Document is not safe to share across
        # threads; a boto3 client is reusable and expensive to construct per page).
        tl = threading.local()

        def worker(idx):
            if getattr(tl, "path", None) != pdf_path:
                tl.path = pdf_path
                tl.doc = fitz.open(pdf_path)
                tl.client = make_client(region=region)
            img = render_page_image(tl.doc[idx], dpi=dpi)
            return idx, detect_page_words(tl.client, img)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for idx, words in ex.map(worker, to_call):
                cache[str(idx)] = words
                done += 1
                if verbose and done % 50 == 0:
                    print(f"    OCR {done}/{len(to_call)} pages", flush=True)
                    save_geom_cache(pdf_path, cache)  # checkpoint: crash loses <=50 pages
        save_geom_cache(pdf_path, cache)

    total_words = 0
    for i in need:
        words = cache.get(str(i)) or []
        total_words += overlay_words(doc[i], words)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_pages = doc.page_count
    doc.save(out_path, deflate=True, garbage=3)
    doc.close()
    return n_pages, len(to_call), total_words


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="input dir (e.g. data/raw/fbi_solo); searched recursively")
    ap.add_argument("--glob", default="*.pdf", help="filename filter within --dir")
    ap.add_argument("--out-root", default=os.path.join("data", "searchable"),
                    help="output root; input path under data/raw/ is mirrored beneath it")
    ap.add_argument("--region", default="us-east-1", help="Textract region (NOT us-west-1)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--estimate", action="store_true", help="count OCR-needed pages + cost, no API calls")
    ap.add_argument("--limit", type=int, default=0, help="process at most N PDFs (0 = all)")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip PDFs whose output already exists and is newer than the geometry cache")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.dir)
    pdfs = sorted(globmod.glob(os.path.join(in_dir, "**", args.glob), recursive=True))
    pdfs = [p for p in pdfs if p.lower().endswith(".pdf")]
    if args.limit:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        print("No PDFs matched.")
        return

    if args.estimate:
        total_pages = 0
        total_need = 0
        cached = 0
        for p in pdfs:
            try:
                doc = fitz.open(p)
            except Exception as e:
                print(f"  SKIP (unreadable): {os.path.basename(p)}: {e}")
                continue
            need = [i for i in range(doc.page_count) if page_needs_ocr(doc[i])]
            c = load_geom_cache(p)
            cached += sum(1 for i in need if str(i) in c)
            total_pages += doc.page_count
            total_need += len(need)
            doc.close()
        billable = total_need - cached
        print(f"{len(pdfs)} PDFs, {total_pages} pages; {total_need} image-only pages "
              f"({cached} already geometry-cached)")
        print(f"Textract cost for uncached pages: ~${billable * COST_PER_PAGE:.2f}")
        return

    grand_ocr = 0
    t0 = time.time()
    for k, p in enumerate(pdfs, 1):
        rel = os.path.relpath(p, REPO) if p.startswith(REPO) else os.path.basename(p)
        out = os.path.join(REPO, args.out_root,
                           os.path.relpath(p, os.path.join(REPO, "data", "raw")))
        if args.skip_done and os.path.exists(out):
            gc = geom_cache_path(p)
            if os.path.exists(gc) and os.path.getmtime(out) >= os.path.getmtime(gc):
                print(f"[{k}/{len(pdfs)}] SKIP (done): {rel}", flush=True)
                continue
        print(f"[{k}/{len(pdfs)}] {rel}", flush=True)
        try:
            _, ocr_called, words = process_pdf(p, out, args.region, args.workers, dpi=args.dpi)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        grand_ocr += ocr_called
        print(f"  -> {out}  (OCR calls: {ocr_called}, words overlaid: {words})", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s. Textract pages billed this run: {grand_ocr} "
          f"(~${grand_ocr * COST_PER_PAGE:.2f})")


if __name__ == "__main__":
    main()
