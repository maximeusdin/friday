#!/usr/bin/env python3
"""Ingest a directory of PDFs as one collection (each file = one document).

Generic version of the per-collection batch ingest. Uses the page-based
chunker from ingest_fbicomrap. Text source:
  * default: embedded PDF text (for pre-OCR'd "_text.pdf" FBI releases)
  * --ocr:   re-OCR every page with AWS Textract (grayscale JPEG, parallel)

Does NOT upload to S3 (do that separately). source_ref is set to the file's
data/raw/... relative path so the viewer resolves the PDF.

Run:  export DATABASE_URL=<prod>
      python -m scripts.ingest_dir_collection --dir data/raw/david_greenglass \
        --slug david_greenglass --title "David Greenglass FBI Files" \
        --pipeline-version david_greenglass_v1
"""
import os
import sys
import glob
import json
import argparse
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fitz
import ingest_runs
from ingest_report_pdf import get_or_create_collection, _connect
from ingest_fbicomrap import (
    ChunkingConfig, PageData, normalize_ocr_text, detect_boilerplate,
    remove_boilerplate, create_chunks_from_pages, sha256_file, upsert_document,
    delete_pages_for_document, delete_chunks_for_document,
    insert_page, insert_chunk, insert_chunk_pages, extract_page_text,
)


def s3_relative_ref(path: Path) -> str:
    """Derive the data/raw/... relative path the backend uses for S3 URLs."""
    p = str(path).replace("\\", "/")
    i = p.lower().find("/data/")
    if i >= 0:
        return p[i + 1:]
    return p


def get_pages_embedded(pdf_path):
    doc = fitz.open(str(pdf_path))
    pages = [(i + 1, extract_page_text(doc, i)) for i in range(doc.page_count)]
    doc.close()
    return pages


def _has_embedded_text(pdf_path, min_chars_per_page=50, sample=12):
    """True if the PDF carries a usable text layer (vs an image-only scan)."""
    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    idxs = list(range(0, n, max(1, n // sample)))[:sample]
    chars = sum(len(doc[i].get_text().strip()) for i in idxs)
    doc.close()
    return (chars // max(1, len(idxs))) >= min_chars_per_page


def main():
    ap = argparse.ArgumentParser(description="Ingest a directory of PDFs as one collection")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--glob", default="*.pdf")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--pipeline-version", required=True)
    ap.add_argument("--max-chars", type=int, default=7000,
                    help="Max chunk chars; keep under ~7700 so chunk text fits Postgres index row limit (8191 bytes)")
    ap.add_argument("--ocr", action="store_true", help="Re-OCR ALL files with Textract")
    ap.add_argument("--ocr-auto", action="store_true",
                    help="Per-file: use embedded text where present, OCR only image-only scans")
    ap.add_argument("--ocr-region", default="us-east-1")
    ap.add_argument("--cache-dir", default="ocr_cache")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    mode = "ocr" if args.ocr else ("ocr-auto" if args.ocr_auto else "embedded")
    print(f"{len(files)} files -> collection '{args.slug}' (mode={mode})")

    ocr_client = None
    if args.ocr or args.ocr_auto:
        import ocr_textract
        ocr_client = ocr_textract  # module; client made per-call inside parallel
        os.makedirs(os.path.join(args.cache_dir, args.slug), exist_ok=True)

    config = ChunkingConfig(max_chars=args.max_chars)
    conn = _connect()
    cur = conn.cursor()
    ingest_runs.ensure_ingest_runs_table(cur)
    collection_id = get_or_create_collection(cur, args.slug, args.title, args.description)
    conn.commit()
    print(f"collection {args.slug} id={collection_id}")

    tot_pages = tot_chunks = 0
    for n, f in enumerate(files, 1):
        name = os.path.basename(f)
        try:
            do_ocr = args.ocr or (args.ocr_auto and not _has_embedded_text(f))
            if do_ocr:
                cache = os.path.join(args.cache_dir, args.slug, name + ".json")
                if os.path.exists(cache):
                    pages_text = json.load(open(cache, encoding="utf-8"))
                else:
                    pages_text = ocr_client.ocr_pdf_parallel(f, region=args.ocr_region,
                                                              max_workers=8, progress=None)
                    json.dump(pages_text, open(cache, "w", encoding="utf-8"))
                raw_pages = [(i + 1, t) for i, t in enumerate(pages_text)]
                src_fmt, extractor = "pdf_scanned_ocr", "textract"
            else:
                raw_pages = get_pages_embedded(f)
                src_fmt, extractor = "pdf_embedded_text", "pymupdf"

            boiler = detect_boilerplate([t for _, t in raw_pages], threshold=config.boilerplate_threshold)
            meta = {"source_format": src_fmt, "extractor": extractor,
                    "page_count": len(raw_pages), "sha256": sha256_file(Path(f))}
            doc_id = upsert_document(cur, collection_id=collection_id, source_name=name,
                                     source_ref=s3_relative_ref(Path(f)), volume="", metadata=meta)
            delete_chunks_for_document(cur, doc_id)
            delete_pages_for_document(cur, doc_id)

            pdl = []
            for pno, txt in raw_pages:
                pid = insert_page(cur, doc_id, pno, pno, f"p{pno:04d}", txt)
                ct = normalize_ocr_text(txt)
                ct, _ = remove_boilerplate(ct, boiler)
                pdl.append(PageData(page_id=pid, pdf_page_number=pno, raw_text=txt, clean_text=ct))
            chunks = create_chunks_from_pages(pdl, config)
            for ch in chunks:
                cid = insert_chunk(cur, ch.text, ch.clean_text, args.pipeline_version)
                insert_chunk_pages(cur, cid, ch.page_ids)
            conn.commit()
            tot_pages += len(raw_pages)
            tot_chunks += len(chunks)
            print(f"[{n}/{len(files)}] {'OCR' if do_ocr else 'txt'} {name[:42]:42} {len(raw_pages):>4}pg {len(chunks):>4}ch doc={doc_id}", flush=True)
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[{n}/{len(files)}] SKIP {name[:42]:42} -> {type(e).__name__}: {str(e)[:90]}", flush=True)
            continue

    conn.close()
    print(f"\nDONE: {len(files)} docs, {tot_pages} pages, {tot_chunks} chunks into '{args.slug}'")


if __name__ == "__main__":
    main()
