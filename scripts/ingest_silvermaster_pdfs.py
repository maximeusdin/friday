#!/usr/bin/env python3
import os
import re
import glob
import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import psycopg2

import fitz  # PyMuPDF

import ingest_runs


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")

COLLECTION_SLUG = "silvermaster"
PIPELINE_VERSION_DEFAULT = "silvermaster_pages_v1"


def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

def sha256_file(path: Path) -> str:
    # Kept for backwards compatibility; now delegated to shared helper.
    return ingest_runs.file_sha256(path)


def normalize_whitespace(text: str) -> str:
    # Keep it light: you want OCR artifacts preserved for provenance/search.
    # This mainly prevents crazy spacing from bloating chunks.
    text = text.replace("\u00a0", " ")  # NBSP -> space
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_or_create_collection(cur) -> int:
    cur.execute(
        "SELECT id FROM collections WHERE slug = %s",
        (COLLECTION_SLUG,),
    )
    r = cur.fetchone()
    if r:
        return int(r[0])

    title = "FBI Silvermaster files"
    description = "Silvermaster FBI file PDFs with embedded/OCR text layer (ingested page-by-page for citation)."

    # If your table doesn't have description, we'll handle that below.
    try:
        cur.execute(
            """
            INSERT INTO collections (slug, title, description)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (COLLECTION_SLUG, title, description),
        )
    except psycopg2.errors.UndefinedColumn:
        # Fallback if 'description' column doesn't exist
        cur.connection.rollback()
        cur.execute(
            """
            INSERT INTO collections (slug, title)
            VALUES (%s, %s)
            RETURNING id
            """,
            (COLLECTION_SLUG, title),
        )

    return int(cur.fetchone()[0])



def parse_part_volume(filename: str) -> str:
    # Supports: Part 25, Part 25.1, Part 25.2
    m = re.search(r"Part\s+(\d+(?:\.\d+)?)", filename, flags=re.IGNORECASE)
    return f"Part {m.group(1)}" if m else ""



def upsert_document(cur, collection_id: int, source_name: str, source_ref: str, volume: str, metadata: dict) -> int:
    cur.execute(
        """
        INSERT INTO documents (collection_id, source_name, source_ref, volume, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (collection_id, source_name, volume_key)
        DO UPDATE SET
          source_ref = EXCLUDED.source_ref,
          metadata = EXCLUDED.metadata
        RETURNING id
        """,
        (collection_id, source_name, source_ref, volume, json.dumps(metadata)),
    )
    return int(cur.fetchone()[0])


def delete_pages_for_document(cur, document_id: int):
    cur.execute("DELETE FROM pages WHERE document_id=%s", (document_id,))

def delete_chunks_for_document(cur, document_id: int):
    # If retrieval evidence was recorded for chunks in this document,
    # delete that dependent data first (FK blocks chunk deletion).
    cur.execute(
        """
        DELETE FROM retrieval_run_chunk_evidence
        WHERE chunk_id IN (
          SELECT cp.chunk_id
          FROM chunk_pages cp
          JOIN pages p ON p.id = cp.page_id
          WHERE p.document_id = %s
        )
        """,
        (document_id,),
    )
    cur.execute(
        """
        DELETE FROM chunks
        WHERE id IN (
          SELECT cp.chunk_id
          FROM chunk_pages cp
          JOIN pages p ON p.id = cp.page_id
          WHERE p.document_id = %s
        )
        """,
        (document_id,),
    )


def insert_page(cur, document_id: int, page_seq: int, pdf_page_number: int, logical_label: str, raw_text: str):
    cur.execute(
        """
        INSERT INTO pages (document_id, logical_page_label, pdf_page_number, page_seq, language, content_role, raw_text)
        VALUES (%s, %s, %s, %s, 'en', 'primary', %s)
        """,
        (document_id, logical_label, pdf_page_number, page_seq, raw_text),
    )


def extract_page_text(doc: fitz.Document, page_index: int) -> str:
    page = doc.load_page(page_index)

    # 1) Default extractor
    txt = page.get_text("text") or ""
    txt = normalize_whitespace(txt)
    if txt.strip():
        return txt

    # 2) Blocks extractor (sometimes works when "text" is empty)
    blocks = page.get_text("blocks") or []
    parts = []
    for b in blocks:
        if len(b) >= 5 and isinstance(b[4], str):
            parts.append(b[4])
    txt2 = normalize_whitespace("\n".join(parts))
    if txt2.strip():
        return txt2

    # 3) Dict/rawdict extractor (pull spans explicitly)
    # This often works on weird PDFs where text exists but isn't returned by "text"
    d = page.get_text("rawdict")
    span_texts = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "")
                if t:
                    span_texts.append(t)
    txt3 = normalize_whitespace("\n".join(span_texts))
    if txt3.strip():
        return txt3

    # 4) Nothing extractable
    return ""



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/raw/silvermaster/pdf")
    ap.add_argument("--glob", default="*.pdf", help="file glob within input-dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-sha", action="store_true", help="skip computing sha256 (faster)")
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION_DEFAULT, help="ingest pipeline version (for rerun tracking)")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.input_dir) / args.glob)))
    if args.limit is not None:
        paths = paths[: int(args.limit)]
    if not paths:
        raise SystemExit(f"No PDFs found at {args.input_dir}/{args.glob}")

    with connect() as conn, conn.cursor() as cur:
        ingest_runs.ensure_ingest_runs_table(cur)
        collection_id = get_or_create_collection(cur)

        for p in paths:
            pdf_path = Path(p)
            source_name = pdf_path.name
            volume = parse_part_volume(source_name)
            pipeline_version = str(args.pipeline_version)
            source_key = f"{COLLECTION_SLUG}:{source_name}"

            doc = None
            fp = None
            try:
                # Open PDF
                doc = fitz.open(str(pdf_path))
                page_count = doc.page_count

                meta = {
                    "source_format": "pdf_embedded_text",
                    "extractor": "pymupdf",
                    "page_count": page_count,
                    "pipeline_version": pipeline_version,
                    "source_key": source_key,
                }
                if args.no_sha:
                    fp = ingest_runs.file_fingerprint_fast(pdf_path)
                    meta["fingerprint"] = fp
                else:
                    fp = ingest_runs.file_sha256(pdf_path)
                    meta["sha256"] = fp

                if not ingest_runs.should_run(
                    cur, source_key=source_key, source_fingerprint=fp, pipeline_version=pipeline_version
                ):
                    print(f"[skip] {source_name} (already ingested: pipeline={pipeline_version})")
                    continue

                ingest_runs.mark_running(
                    cur, source_key=source_key, source_fingerprint=fp, pipeline_version=pipeline_version
                )

                doc_id = upsert_document(
                    cur,
                    collection_id=collection_id,
                    source_name=source_name,
                    source_ref=str(pdf_path),
                    volume=volume,
                    metadata=meta,
                )

                # Rerun safety: if chunks exist, pages can't be deleted (page_id is RESTRICT)
                delete_chunks_for_document(cur, doc_id)
                delete_pages_for_document(cur, doc_id)

                for i in range(page_count):
                    pdf_page_number = i + 1
                    page_seq = pdf_page_number
                    logical = f"p{pdf_page_number:04d}"
                    txt = extract_page_text(doc, i)
                    insert_page(cur, doc_id, page_seq, pdf_page_number, logical, txt)

                ingest_runs.mark_success(cur, source_key=source_key)
                conn.commit()
                print(f"[ok] {source_name} -> document_id={doc_id} pages={page_count} volume='{volume}'")

            except Exception as e:
                conn.rollback()
                if fp is None:
                    fp = ingest_runs.file_fingerprint_fast(pdf_path)
                ingest_runs.mark_failed_best_effort(
                    connect,
                    source_key=source_key,
                    source_fingerprint=fp,
                    pipeline_version=pipeline_version,
                    error=str(e),
                )
                print(f"[error] {source_name}: {e}")
                continue
            finally:
                if doc is not None:
                    doc.close()

    print("✅ Silvermaster PDF ingest complete.")


if __name__ == "__main__":
    main()
