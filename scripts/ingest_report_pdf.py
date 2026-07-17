#!/usr/bin/env python3
"""
Generic single-PDF ingest for narrative government *reports* (FBI summary
reports, Congressional committee reports, etc.) -- each PDF becomes its OWN
collection.

This is a thin wrapper around the page-based, paragraph-block, overlapping
chunker already implemented and tuned in scripts/ingest_fbicomrap.py. It adds:
  * a parametrized collection (slug / title / description)
  * single-file ingest (instead of a directory glob)
  * an OPTIONAL inline OCR fallback for pages that have no embedded text layer
    (needed for fully-scanned PDFs such as the FBI CINRAD summary)

It reuses, verbatim, fbicomrap's:
  normalize_ocr_text, detect_boilerplate, remove_boilerplate,
  compute_quality_metrics, create_chunks_from_pages, ChunkingConfig, PageData,
  sha256_file, upsert_document, insert_page, insert_chunk, insert_chunk_pages,
  insert_chunk_metadata, connect, extract_page_text

After running this, populate the retrieval-facing metadata and embed:
  python -m scripts.build_chunk_metadata --chunk-pipeline <PIPELINE_VERSION>
  python -m scripts.embed_silvermaster_chunks \
      --chunk-pv <PIPELINE_VERSION> --collection-slug <SLUG> --fill-missing-only
"""
import os
import sys
import io
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import psycopg2
import fitz  # PyMuPDF

# Make sibling scripts importable whether run as a module or a file.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ingest_runs  # noqa: E402
from ingest_fbicomrap import (  # noqa: E402
    ChunkingConfig,
    PageData,
    normalize_ocr_text,
    detect_boilerplate,
    remove_boilerplate,
    create_chunks_from_pages,
    sha256_file,
    upsert_document,
    delete_pages_for_document,
    delete_chunks_for_document,
    insert_page,
    insert_chunk,
    insert_chunk_pages,
    insert_chunk_metadata,
    extract_page_text,
    connect,
)


def _connect():
    """Prefer DATABASE_URL (prod RDS / Secrets Manager) over fbicomrap's
    DB_HOST/DB_USER env defaults, which point at localhost."""
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return connect()


def get_or_create_collection(cur, slug: str, title: str, description: str) -> int:
    cur.execute("SELECT id FROM collections WHERE slug = %s", (slug,))
    r = cur.fetchone()
    if r:
        return int(r[0])
    try:
        cur.execute(
            "INSERT INTO collections (slug, title, description) VALUES (%s, %s, %s) RETURNING id",
            (slug, title, description),
        )
    except psycopg2.errors.UndefinedColumn:
        cur.connection.rollback()
        cur.execute(
            "INSERT INTO collections (slug, title) VALUES (%s, %s) RETURNING id",
            (slug, title),
        )
    return int(cur.fetchone()[0])


def _build_ocr_fn(engine: str, ocr_dpi: int, ocr_lang: str, aws_region: str, aws_profile: str):
    """Return a callable page -> text for the chosen OCR engine."""
    if engine == "textract":
        import ocr_textract
        client = ocr_textract.make_client(region=aws_region or None, profile=aws_profile or None)

        def _fn(page: "fitz.Page") -> str:
            img = ocr_textract.render_page_image(page, dpi=ocr_dpi)
            return ocr_textract.detect_page_text(client, img)

        return _fn

    if engine == "tesseract":
        import pytesseract
        from PIL import Image

        cmd = os.getenv("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

        def _fn(page: "fitz.Page") -> str:
            pix = page.get_pixmap(dpi=ocr_dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img, lang=ocr_lang) or ""

        return _fn

    raise ValueError(f"Unknown OCR engine: {engine}")


def extract_pages(
    pdf_path: Path,
    use_ocr: bool,
    ocr_dpi: int,
    ocr_lang: str,
    ocr_engine: str,
    aws_region: str,
    aws_profile: str,
) -> List[Tuple[int, str]]:
    doc = fitz.open(str(pdf_path))
    ocr_fn = _build_ocr_fn(ocr_engine, ocr_dpi, ocr_lang, aws_region, aws_profile) if use_ocr else None
    raw_pages: List[Tuple[int, str]] = []
    ocr_count = 0
    for i in range(doc.page_count):
        txt = extract_page_text(doc, i)
        if (not txt or not txt.strip()) and ocr_fn is not None:
            txt = ocr_fn(doc.load_page(i))
            if txt.strip():
                ocr_count += 1
        raw_pages.append((i + 1, txt))
        if use_ocr and (i + 1) % 5 == 0:
            print(f"    ...page {i + 1}/{doc.page_count} (ocr'd so far: {ocr_count})")
    doc.close()
    if use_ocr:
        print(f"    OCR ({ocr_engine}) applied to {ocr_count}/{len(raw_pages)} pages")
    return raw_pages


def main():
    ap = argparse.ArgumentParser(description="Ingest a single narrative report PDF as its own collection")
    ap.add_argument("--pdf", required=True, help="Path to the PDF file")
    ap.add_argument("--collection-slug", required=True, help="Unique collection slug, e.g. fbi_cinrad")
    ap.add_argument("--collection-title", required=True)
    ap.add_argument("--collection-description", default="")
    ap.add_argument("--pipeline-version", required=True, help="chunks.pipeline_version, e.g. fbi_cinrad_v1")
    ap.add_argument("--volume", default="", help="Optional volume label for the documents row")
    # Chunking knobs (same defaults as ingest_fbicomrap)
    ap.add_argument("--target-chars", type=int, default=5000)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--overlap-chars", type=int, default=1000)
    ap.add_argument("--boilerplate-threshold", type=float, default=0.35)
    # Reuse a precomputed OCR cache (JSON list of per-page text) instead of
    # extracting/OCRing during ingest (avoids re-running Textract).
    ap.add_argument("--text-cache", default="", help="JSON list of per-page text (index 0 == page 1)")
    # OCR (only needed for fully-scanned PDFs with no text layer)
    ap.add_argument("--ocr", action="store_true", help="OCR pages that have no embedded text")
    ap.add_argument("--ocr-engine", choices=["textract", "tesseract"], default="textract",
                    help="OCR backend (default: AWS Textract)")
    ap.add_argument("--ocr-dpi", type=int, default=300)
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language(s); ignored by Textract")
    ap.add_argument("--aws-region", default=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
                    help="AWS region for Textract (must support Textract; not us-west-1)")
    ap.add_argument("--aws-profile", default=os.getenv("AWS_PROFILE", ""),
                    help="Optional AWS named profile for credentials")
    ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    config = ChunkingConfig(
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        boilerplate_threshold=args.boilerplate_threshold,
    )

    print(f"Ingesting {pdf_path.name}")
    print(f"  collection={args.collection_slug}  pipeline={args.pipeline_version}  ocr={args.ocr}")

    if args.text_cache:
        import json as _json
        cached = _json.load(open(args.text_cache, encoding="utf-8"))
        raw_pages = [(i + 1, t) for i, t in enumerate(cached)]
        npages = fitz.open(str(pdf_path)).page_count
        if len(raw_pages) != npages:
            print(f"  WARNING: cache has {len(raw_pages)} pages but PDF has {npages}")
        print(f"  using OCR cache: {args.text_cache} ({len(raw_pages)} pages)")
    else:
        raw_pages = extract_pages(
            pdf_path, args.ocr, args.ocr_dpi, args.ocr_lang,
            args.ocr_engine, args.aws_region, args.aws_profile,
        )
    page_count = len(raw_pages)
    nonempty = sum(1 for _, t in raw_pages if t and t.strip())
    print(f"  pages={page_count}  pages_with_text={nonempty}")
    if nonempty == 0:
        raise SystemExit(
            "No extractable text on any page. This PDF is image-only -- rerun with --ocr.\n"
            "  Textract (default): needs AWS creds + a Textract region (--aws-region).\n"
            "  Tesseract (--ocr-engine tesseract): needs `pip install pytesseract pillow` + Tesseract binary."
        )

    boilerplate = detect_boilerplate([t for _, t in raw_pages], threshold=config.boilerplate_threshold)
    print(f"  detected {len(boilerplate)} boilerplate patterns")

    if args.dry_run:
        print("  [DRY RUN] not writing to database")
        return

    if args.text_cache:
        source_format, extractor = "pdf_scanned_ocr", "textract(cache)"
    elif args.ocr:
        source_format, extractor = "pdf_scanned_ocr", f"pymupdf+{args.ocr_engine}"
    else:
        source_format, extractor = "pdf_embedded_text", "pymupdf"
    meta = {
        "source_format": source_format,
        "extractor": extractor,
        "page_count": page_count,
        "boilerplate_patterns": len(boilerplate),
    }
    if not args.no_sha:
        meta["sha256"] = sha256_file(pdf_path)

    with _connect() as conn, conn.cursor() as cur:
        ingest_runs.ensure_ingest_runs_table(cur)
        collection_id = get_or_create_collection(
            cur, args.collection_slug, args.collection_title, args.collection_description
        )
        print(f"  collection id={collection_id}")

        doc_id = upsert_document(
            cur,
            collection_id=collection_id,
            source_name=pdf_path.name,
            source_ref=str(pdf_path),
            volume=args.volume,
            metadata=meta,
        )
        delete_chunks_for_document(cur, doc_id)
        delete_pages_for_document(cur, doc_id)

        page_data_list: List[PageData] = []
        for pdf_page_num, raw_text in raw_pages:
            logical_label = f"p{pdf_page_num:04d}"
            page_id = insert_page(cur, doc_id, pdf_page_num, pdf_page_num, logical_label, raw_text)
            clean_text = normalize_ocr_text(raw_text)
            clean_text, bp_removed = remove_boilerplate(clean_text, boilerplate)
            page_data_list.append(
                PageData(
                    page_id=page_id,
                    pdf_page_number=pdf_page_num,
                    raw_text=raw_text,
                    clean_text=clean_text,
                    boilerplate_removed=bp_removed,
                )
            )

        chunks = create_chunks_from_pages(page_data_list, config)
        print(f"  created {len(chunks)} chunks")

        for chunk in chunks:
            chunk_id = insert_chunk(cur, chunk.text, chunk.clean_text, args.pipeline_version)
            insert_chunk_pages(cur, chunk_id, chunk.page_ids)
            insert_chunk_metadata(cur, chunk_id, {
                "document_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "boilerplate_removed": chunk.boilerplate_removed,
                "redaction_ratio": chunk.metrics.redaction_ratio,
                "alpha_ratio": chunk.metrics.alpha_ratio,
                "low_signal": chunk.metrics.low_signal,
                "contains_table_like": chunk.metrics.contains_table_like,
                "contains_list_like": chunk.metrics.contains_list_like,
            })

        conn.commit()
        print(f"  done -> document_id={doc_id}, pages={page_count}, chunks={len(chunks)}")
        print(f"\nNext: build metadata, then embed:")
        print(f"  python -m scripts.build_chunk_metadata --chunk-pipeline {args.pipeline_version} --collection-slug {args.collection_slug}")
        print(f"  python -m scripts.embed_silvermaster_chunks --chunk-pv {args.pipeline_version} --collection-slug {args.collection_slug} --fill-missing-only")


if __name__ == "__main__":
    main()
