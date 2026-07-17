#!/usr/bin/env python3
"""AWS Textract OCR backend for scanned PDFs.

Renders each PDF page to a PNG (PyMuPDF) and calls Textract's *synchronous*
DetectDocumentText, one call per page. This keeps page boundaries exact and
avoids the async + S3 round-trip -- appropriate for documents up to a few
hundred pages. Cost: DetectDocumentText is ~$1.50 / 1000 pages.

Credentials use standard boto3 resolution (env vars, ~/.aws/credentials, SSO,
instance role). Choose account/region via make_client(region, profile).
Textract must be available in the chosen region (e.g. us-east-1, us-east-2,
us-west-2; it is NOT available in us-west-1).

For very large scanned PDFs (many hundreds of pages) the asynchronous API
(StartDocumentTextDetection on an S3 object) is cheaper per call, but is not
needed for the documents this project ingests.
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import fitz  # PyMuPDF

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    boto3 = None
    ClientError = Exception
    Config = None


# Transient Textract errors worth retrying with backoff.
RETRYABLE = {
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "InternalServerError",
    "ServiceUnavailableException",
    "LimitExceededException",
}


def make_client(region: Optional[str] = None, profile: Optional[str] = None):
    if boto3 is None:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}) if Config else None
    return session.client("textract", region_name=region, config=cfg)


def render_page_png(page: "fitz.Page", dpi: int = 300) -> bytes:
    """Render a single PDF page to PNG bytes."""
    return page.get_pixmap(dpi=dpi).tobytes("png")


# Textract synchronous DetectDocumentText caps inline image bytes at 5 MB.
# High-res scans blow past that as PNG, so render grayscale JPEG and back off
# the DPI until the encoded image fits.
TEXTRACT_MAX_BYTES = 4_500_000


def render_page_image(page: "fitz.Page", dpi: int = 300, max_bytes: int = TEXTRACT_MAX_BYTES) -> bytes:
    """Render a page to grayscale JPEG bytes (via PyMuPDF, no extra deps),
    reducing DPI if needed to stay under Textract's inline-image size limit."""
    data = b""
    for d in (dpi, 230, 180, 140, 110):
        pix = page.get_pixmap(dpi=d, colorspace=fitz.csGRAY)
        try:
            data = pix.tobytes("jpeg", jpg_quality=80)
        except (TypeError, ValueError):
            data = pix.tobytes("jpg")  # older PyMuPDF without quality arg
        if len(data) <= max_bytes:
            return data
    return data  # smallest attempt, even if still over (let Textract decide)


def detect_page_text(client, png_bytes: bytes, max_retries: int = 6) -> str:
    """Run synchronous Textract on one page image; return LINE-joined text."""
    delay = 1.0
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = client.detect_document_text(Document={"Bytes": png_bytes})
            lines = [
                b["Text"]
                for b in resp.get("Blocks", [])
                if b.get("BlockType") == "LINE"
            ]
            return "\n".join(lines)
        except ClientError as e:  # type: ignore[misc]
            code = e.response.get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
            last_exc = e
            if code in RETRYABLE and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    if last_exc:
        raise last_exc
    return ""


# Thread-local fitz document + Textract client so each worker renders/calls
# independently (a fitz.Document is not safe to share across threads).
_tl = threading.local()


def _worker_ocr(args):
    pdf_path, idx, dpi, region, profile = args
    if getattr(_tl, "path", None) != pdf_path:
        _tl.path = pdf_path
        _tl.doc = fitz.open(pdf_path)
        _tl.client = make_client(region=region, profile=profile)
    img = render_page_image(_tl.doc.load_page(idx), dpi=dpi)
    return idx, detect_page_text(_tl.client, img)


def ocr_pdf_parallel(
    pdf_path: str,
    region: Optional[str] = None,
    profile: Optional[str] = None,
    dpi: int = 300,
    max_workers: int = 8,
    page_indices: Optional[List[int]] = None,
    progress_every: int = 25,
    progress=print,
) -> List[str]:
    """OCR a whole PDF with parallel synchronous Textract calls.

    Returns a list of page texts in page order (index 0 == PDF page 1). If
    page_indices is given, only those pages are OCR'd; the rest are "".
    """
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    targets = page_indices if page_indices is not None else list(range(n))
    out = [""] * n
    done = 0
    args = [(pdf_path, i, dpi, region, profile) for i in targets]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, text in ex.map(_worker_ocr, args):
            out[idx] = text
            done += 1
            if progress and progress_every and done % progress_every == 0:
                progress(f"    OCR {done}/{len(targets)} pages")
    if progress:
        progress(f"    OCR complete: {len(targets)} pages")
    return out
