#!/usr/bin/env python3
"""Textract OCR pass that retains geometry, for the Bentley deposition workflow.

For each page in --pages:
  1. Ensure BASE/images/{page:04d}.png exists. If missing, render it from
     --pdf with PyMuPDF at width 1700px (zoom = 1700 / page.rect.width),
     matching the shared data contract. Page numbers are 1-based.
  2. If BASE/textract/{page:04d}.json is missing, call the synchronous
     Textract DetectDocumentText API on the PNG bytes and write the full
     response (including Blocks with Geometry) as JSON. Existing JSON files
     are never re-fetched, so the script is resumable.
  3. Always (re)derive BASE/textract/{page:04d}.txt by joining the Text of
     LINE blocks in the order Textract returns them (reading order).

Credentials come from the standard AWS resolution chain (env vars, shared
credentials file, SSO, instance role) -- nothing is hardcoded here. For this
project the caller sets:

    AWS_SHARED_CREDENTIALS_FILE=/Users/maxime/friday/.aws/credentials

Cost: DetectDocumentText is ~$1.50 / 1000 pages. Calls are sequential (the
deposition is only 118 pages); throttling and other transient errors are
retried up to 3 times with exponential backoff.

Usage:
    AWS_SHARED_CREDENTIALS_FILE=/Users/maxime/friday/.aws/credentials \
    python scripts/textract_geometry.py \
        --pages 3-120 \
        --base data/transcripts/bentley_deposition \
        --pdf "data/raw/silvermaster/pdf/FBI File Silvermaster Part 6 late November 1945_text.pdf"
"""
import argparse
import json
import os
import sys
import time

# Rendered page width in pixels, per the shared data contract.
IMAGE_WIDTH = 1700

# Transient Textract errors worth retrying with backoff
# (same set as scripts/ocr_textract.py).
RETRYABLE = {
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "InternalServerError",
    "ServiceUnavailableException",
    "LimitExceededException",
}


def parse_pages(spec):
    """Parse 'A-B' (inclusive) or a single page 'N' into a list of ints."""
    if "-" in spec:
        a, b = spec.split("-", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(spec)
    if start < 1 or end < start:
        raise ValueError(f"bad --pages range: {spec!r}")
    return list(range(start, end + 1))


def make_client(region):
    import boto3
    from botocore.config import Config

    # Disable botocore's built-in retries; this script does its own
    # (simple x3 with backoff), keeping behavior explicit and logged.
    cfg = Config(retries={"max_attempts": 1})
    return boto3.Session().client("textract", region_name=region, config=cfg)


def png_ok(path):
    """True when `path` looks like a complete PNG: exists, is > 100 bytes,
    starts with the PNG magic bytes, and ends with an IEND chunk. Any failure
    (missing, tiny, truncated, unreadable) means "treat as missing"."""
    try:
        if os.path.getsize(path) <= 100:
            return False
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
            f.seek(-12, os.SEEK_END)
            tail = f.read(12)
        return b"IEND" in tail
    except OSError:
        return False


def ensure_image(image_path, page, pdf_path, doc_cache):
    """Render BASE/images/{page:04d}.png from the PDF if missing or truncated
    (fails png_ok). The write is atomic: render to image_path+".tmp", then
    os.replace, so an interrupt can never leave a partial PNG in place.

    Returns the open fitz document via doc_cache (dict with key 'doc') so the
    PDF is opened at most once across pages.
    """
    if png_ok(image_path):
        return
    if not pdf_path:
        sys.exit(
            f"ERROR: {image_path} is missing and no --pdf was given to render it from."
        )
    import fitz  # PyMuPDF

    if doc_cache.get("doc") is None:
        if not os.path.exists(pdf_path):
            sys.exit(f"ERROR: PDF not found: {pdf_path}")
        doc_cache["doc"] = fitz.open(pdf_path)
    doc = doc_cache["doc"]
    if page > doc.page_count:
        sys.exit(f"ERROR: page {page} out of range (PDF has {doc.page_count} pages)")
    fpage = doc.load_page(page - 1)  # contract pages are 1-based
    zoom = IMAGE_WIDTH / fpage.rect.width
    pix = fpage.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    tmp = image_path + ".tmp"
    pix.save(tmp, output="png")
    os.replace(tmp, image_path)  # atomic: no truncated PNG on interrupt
    print(f"  page {page:4d}: rendered {os.path.basename(image_path)}")


def detect_with_retry(client, png_bytes, page, max_retries=3):
    """Call DetectDocumentText, retrying transient errors up to max_retries."""
    from botocore.exceptions import ClientError

    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            return client.detect_document_text(Document={"Bytes": png_bytes})
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in RETRYABLE and attempt < max_retries:
                print(
                    f"  page {page:4d}: {code}, retry {attempt}/{max_retries - 1} "
                    f"in {delay:.0f}s"
                )
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def lines_from_response(resp):
    """LINE-joined text, in the order Textract returns the LINE blocks."""
    return "\n".join(
        b["Text"] for b in resp.get("Blocks", []) if b.get("BlockType") == "LINE"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Textract OCR pass retaining geometry (resumable)."
    )
    ap.add_argument("--pages", required=True, help="page range A-B (1-based, inclusive)")
    ap.add_argument("--base", required=True, help="base dir, e.g. data/transcripts/bentley_deposition")
    ap.add_argument("--region", default="us-east-1", help="AWS region (default us-east-1)")
    ap.add_argument("--pdf", default=None, help="source PDF, used only to render missing page images")
    args = ap.parse_args()

    pages = parse_pages(args.pages)
    images_dir = os.path.join(args.base, "images")
    textract_dir = os.path.join(args.base, "textract")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(textract_dir, exist_ok=True)

    client = None  # created lazily, only if an actual Textract call is needed
    doc_cache = {"doc": None}
    n_ocr_calls = 0
    n_skipped = 0
    n_txt = 0

    for page in pages:
        image_path = os.path.join(images_dir, f"{page:04d}.png")
        json_path = os.path.join(textract_dir, f"{page:04d}.json")
        txt_path = os.path.join(textract_dir, f"{page:04d}.txt")

        ensure_image(image_path, page, args.pdf, doc_cache)

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                resp = json.load(f)
            n_skipped += 1
            print(f"  page {page:4d}: textract json exists, skipping OCR")
        else:
            with open(image_path, "rb") as f:
                png_bytes = f.read()
            if client is None:
                client = make_client(args.region)
            resp = detect_with_retry(client, png_bytes, page)
            tmp = json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(resp, f, ensure_ascii=False)
            os.replace(tmp, json_path)  # atomic: no truncated JSON on interrupt
            n_ocr_calls += 1
            n_lines = sum(
                1 for b in resp.get("Blocks", []) if b.get("BlockType") == "LINE"
            )
            print(f"  page {page:4d}: OCR done ({n_lines} lines)")

        # Always (re)derive the .txt from the JSON.
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(lines_from_response(resp))
        n_txt += 1

    if doc_cache.get("doc") is not None:
        doc_cache["doc"].close()

    print(
        f"Done: {len(pages)} pages | {n_ocr_calls} Textract calls | "
        f"{n_skipped} already had JSON | {n_txt} .txt files written"
    )


if __name__ == "__main__":
    main()
