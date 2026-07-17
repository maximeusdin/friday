#!/usr/bin/env python3
"""Ingest the Rosenberg Trial Transcripts collection.

Mixed sources, normalized to viewable PDFs:
  * .docx  -> extract text from the OOXML, render to a paginated PDF (PyMuPDF
             Story), ingest the PDF's per-page text. No OCR needed.
  * .png   -> OCR with Textract (image bytes direct), wrap the image in a
             1-page PDF for the viewer. OCR needed (these are page scans).

Generated PDFs go to data/raw/rosenberg_trial_transcripts/pdf/ and are the
document source_ref (uploaded to S3 separately).

Run:  export DATABASE_URL=<prod>;  python -m scripts.ingest_rosenberg_trial
"""
import os
import sys
import re
import glob
import html
import zipfile
from pathlib import Path

import fitz

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

SRC_DIR = "data/raw/rosenberg_trial_transcripts"
PDF_DIR = os.path.join(SRC_DIR, "pdf")
SLUG = "rosenberg_trial_transcripts"
TITLE = "Rosenberg Trial Transcripts (1951)"
DESC = ("Trial testimony and exhibits from United States v. Rosenberg (SDNY, 1951): "
        "witness testimony transcripts and scanned trial pages.")
PIPELINE = "rosenberg_trial_v1"
REGION = "us-east-1"


def docx_text(path):
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8", "ignore")
    out = []
    for p in re.split(r"</w:p>", xml):
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
        t = html.unescape(re.sub(r"<[^>]+>", "", t))
        out.append(t)
    return "\n".join(out)


def docx_to_pdf(text, pdf_path):
    body = "".join(f"<p>{html.escape(l)}</p>" for l in text.split("\n") if l.strip())
    story = fitz.Story(html=f"<html><body style='font-family:sans-serif;font-size:11px'>{body}</body></html>")
    writer = fitz.DocumentWriter(pdf_path)
    mb = fitz.paper_rect("letter")
    where = mb + (54, 54, -54, -54)
    more = 1
    while more:
        dev = writer.begin_page(mb)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()


def png_to_pdf(png_path, pdf_path):
    img = fitz.open(png_path)
    Path(pdf_path).write_bytes(img.convert_to_pdf())


def pdf_page_texts(pdf_path):
    d = fitz.open(pdf_path)
    return [d[i].get_text() for i in range(d.page_count)]


def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    docx = sorted(glob.glob(os.path.join(SRC_DIR, "*.docx")))
    png = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
    print(f"{len(docx)} docx + {len(png)} png -> '{SLUG}'")

    client = ocr_textract.make_client(region=REGION)
    config = ChunkingConfig()
    conn = _connect()
    cur = conn.cursor()
    ingest_runs.ensure_ingest_runs_table(cur)
    collection_id = get_or_create_collection(cur, SLUG, TITLE, DESC)
    conn.commit()
    print(f"collection {SLUG} id={collection_id}")

    tot_chunks = 0

    def ingest(source_name, pdf_rel, page_texts, fmt, extractor):
        nonlocal tot_chunks
        raw_pages = [(i + 1, t) for i, t in enumerate(page_texts)]
        boiler = detect_boilerplate([t for _, t in raw_pages], threshold=config.boilerplate_threshold)
        meta = {"source_format": fmt, "extractor": extractor, "page_count": len(raw_pages)}
        doc_id = upsert_document(cur, collection_id=collection_id, source_name=source_name,
                                 source_ref=pdf_rel, volume="", metadata=meta)
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
        tot_chunks += len(chunks)
        print(f"  {source_name[:45]:45} {len(raw_pages):>3}pg {len(chunks):>3}ch doc={doc_id}")

    # DOCX (text, no OCR)
    for f in docx:
        stem = Path(f).stem.strip()
        pdf_path = os.path.join(PDF_DIR, stem + ".pdf")
        docx_to_pdf(docx_text(f), pdf_path)
        ingest(stem + ".pdf", f"{PDF_DIR}/{stem}.pdf".replace("\\", "/"),
               pdf_page_texts(pdf_path), "docx_text", "ooxml+story")

    # PNG (OCR)
    for f in png:
        stem = Path(f).stem
        pdf_path = os.path.join(PDF_DIR, stem + ".pdf")
        png_to_pdf(f, pdf_path)
        ocr = ocr_textract.detect_page_text(client, open(f, "rb").read())
        ingest(stem + ".pdf", f"{PDF_DIR}/{stem}.pdf".replace("\\", "/"),
               [ocr], "png_scan", "textract")

    conn.close()
    print(f"\nDONE: {len(docx)+len(png)} docs, {tot_chunks} chunks into '{SLUG}'")
    print(f"Next: build_chunk_metadata + embed (pipeline {PIPELINE}); upload {PDF_DIR}/*.pdf to S3")


if __name__ == "__main__":
    main()
