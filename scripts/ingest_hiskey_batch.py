#!/usr/bin/env python3
"""Batch OCR + ingest of the FBI Clarence Hiskey files into one collection.

Many small 1940s FBI scans, each a separate document in collection `fbi_hiskey`.
Their embedded text layer is uniformly garbled, so every file is re-OCR'd with
AWS Textract (parallel sync). Per-file OCR is cached so the step is resumable.
source_ref is set to the S3-relative path so the viewer resolves the PDF.

Run:  export DATABASE_URL=<prod>;  python -m scripts.ingest_hiskey_batch
Then: build_chunk_metadata + embed_silvermaster_chunks for pipeline fbi_hiskey_v1.
"""
import os
import sys
import glob
import json
from pathlib import Path

import boto3

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ocr_textract
import ingest_runs
from ingest_report_pdf import get_or_create_collection, _connect
from ingest_fbicomrap import (
    ChunkingConfig, PageData, normalize_ocr_text, detect_boilerplate,
    remove_boilerplate, create_chunks_from_pages, sha256_file, upsert_document,
    delete_pages_for_document, delete_chunks_for_document,
    insert_page, insert_chunk, insert_chunk_pages,
)

DOWNLOADS = os.getenv("HISKEY_DIR", r"C:\Users\maxim\Downloads")
PATTERNS = ["20220318_*.pdf", "20220408_*.pdf",
            "1943*.pdf", "1944*.pdf", "1945*.pdf", "1948*.pdf"]
SLUG = "fbi_hiskey"
TITLE = "FBI Files on Clarence Hiskey"
DESC = ("FBI, Army CIC, and Military Intelligence files on Clarence F. Hiskey, "
        "Manhattan Project chemist investigated for Soviet espionage; incl. press clippings.")
PIPELINE = "fbi_hiskey_v1"
REGION = "us-east-1"
S3_BUCKET = "fridayarchive.org"
CACHE_DIR = os.path.join("ocr_cache", "hiskey")


def list_files():
    files = sorted(set(f for p in PATTERNS for f in glob.glob(os.path.join(DOWNLOADS, p))))
    return files


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    files = list_files()
    print(f"{len(files)} Hiskey files to OCR+ingest into '{SLUG}'")

    client = ocr_textract.make_client(region=REGION)
    s3 = boto3.client("s3", region_name="us-west-1")
    config = ChunkingConfig()

    conn = _connect()
    cur = conn.cursor()
    ingest_runs.ensure_ingest_runs_table(cur)
    collection_id = get_or_create_collection(cur, SLUG, TITLE, DESC)
    conn.commit()
    print(f"collection {SLUG} id={collection_id}")

    tot_pages = tot_chunks = 0
    for n, f in enumerate(files, 1):
        name = os.path.basename(f)
        cache = os.path.join(CACHE_DIR, name + ".json")
        if os.path.exists(cache):
            pages = json.load(open(cache, encoding="utf-8"))
        else:
            pages = ocr_textract.ocr_pdf_parallel(f, region=REGION, max_workers=8,
                                                  progress_every=0, progress=None)
            json.dump(pages, open(cache, "w", encoding="utf-8"))

        raw_pages = [(i + 1, t) for i, t in enumerate(pages)]
        boiler = detect_boilerplate([t for _, t in raw_pages],
                                    threshold=config.boilerplate_threshold)
        s3key = f"data/raw/{SLUG}/{name}"
        meta = {"source_format": "pdf_scanned_ocr", "extractor": "textract",
                "page_count": len(pages), "sha256": sha256_file(Path(f))}
        doc_id = upsert_document(cur, collection_id=collection_id, source_name=name,
                                 source_ref=s3key, volume="", metadata=meta)
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
            cid = insert_chunk(cur, ch.text, ch.clean_text, PIPELINE)
            insert_chunk_pages(cur, cid, ch.page_ids)
        conn.commit()

        with open(f, "rb") as fh:
            s3.put_object(Bucket=S3_BUCKET, Key=s3key, Body=fh,
                          ContentType="application/pdf", ContentDisposition="inline")

        tot_pages += len(pages)
        tot_chunks += len(chunks)
        safe = name.encode("ascii", "replace").decode()
        print(f"[{n}/{len(files)}] {safe[:55]:55} {len(pages):>3}pg {len(chunks):>3}ch doc={doc_id}")

    conn.close()
    print(f"\nDONE: {len(files)} docs, {tot_pages} pages, {tot_chunks} chunks into '{SLUG}'")
    print(f"Next:\n  python -m scripts.build_chunk_metadata --chunk-pipeline {PIPELINE} --collection-slug {SLUG}")
    print(f"  python -m scripts.embed_silvermaster_chunks --chunk-pv {PIPELINE} --collection-slug {SLUG} --prefer-clean-text --fill-missing-only")


if __name__ == "__main__":
    main()
