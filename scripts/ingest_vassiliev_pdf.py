#!/usr/bin/env python3
"""
Ingest Vassiliev PDF into Postgres using *original PDF pages* as the canonical pages.

- 1 pages row per PDF page (page_seq = pdf_page_number).
- logical_page_label is stable "pdf.N".
- Does NOT use p.xx markers for splitting.
- Extracts any p.xx markers present on that PDF page and stores them in page_metadata.meta_raw
  for downstream chunking/citation heuristics.

Change requested:
- "with_markers" should mean: meta_raw contains the key "p_markers" ONLY WHEN markers exist.
  => We now OMIT "p_markers" and "p_marker_first" entirely when no markers are found.
"""

import os
import re
import sys
import hashlib
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import psycopg2
from psycopg2.extras import Json
import fitz  # PyMuPDF


PIPELINE_VERSION = "v4e_pdf_pages_only"
META_PIPELINE_VERSION = "meta_v1_pxx_markers"


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")


def connect():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)


def ensure_ingest_runs_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_runs (
          id BIGSERIAL PRIMARY KEY,
          source_key TEXT NOT NULL UNIQUE,
          source_sha256 TEXT NOT NULL,
          pipeline_version TEXT NOT NULL,
          status TEXT NOT NULL CHECK (status IN ('success','failed','running')),
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          finished_at TIMESTAMPTZ,
          error TEXT
        )
        """
    )


def page_metadata_table_exists(cur) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
          SELECT 1
          FROM information_schema.tables
          WHERE table_schema = 'public'
            AND table_name = 'page_metadata'
        )
        """
    )
    return bool(cur.fetchone()[0])


def should_run(cur, source_key: str, source_sha256: str, pipeline_version: str) -> bool:
    cur.execute(
        "SELECT source_sha256, pipeline_version, status FROM ingest_runs WHERE source_key = %s",
        (source_key,),
    )
    row = cur.fetchone()
    return (
        row is None
        or row[0] != source_sha256
        or row[1] != pipeline_version
        or row[2] != "success"
    )


def mark_running(cur, source_key: str, source_sha256: str, pipeline_version: str):
    cur.execute(
        """
        INSERT INTO ingest_runs (source_key, source_sha256, pipeline_version, status, started_at)
        VALUES (%s, %s, %s, 'running', now())
        ON CONFLICT (source_key) DO UPDATE
        SET source_sha256 = EXCLUDED.source_sha256,
            pipeline_version = EXCLUDED.pipeline_version,
            status = 'running',
            started_at = now(),
            finished_at = NULL,
            error = NULL
        """,
        (source_key, source_sha256, pipeline_version),
    )


def mark_success(cur, source_key: str):
    cur.execute(
        "UPDATE ingest_runs SET status='success', finished_at=now() WHERE source_key=%s",
        (source_key,),
    )


def mark_failed_best_effort(source_key: str, source_sha256: str, err: str):
    try:
        with connect() as conn, conn.cursor() as cur:
            ensure_ingest_runs_table(cur)
            cur.execute(
                """
                INSERT INTO ingest_runs (source_key, source_sha256, pipeline_version, status, started_at, finished_at, error)
                VALUES (%s, %s, %s, 'failed', now(), now(), %s)
                ON CONFLICT (source_key) DO UPDATE
                SET status='failed',
                    finished_at=now(),
                    error=EXCLUDED.error,
                    source_sha256=EXCLUDED.source_sha256,
                    pipeline_version=EXCLUDED.pipeline_version
                """,
                (source_key, source_sha256, PIPELINE_VERSION, err),
            )
            conn.commit()
    except Exception:
        pass


def upsert_collection(cur, slug: str, title: str, description: Optional[str] = None) -> int:
    cur.execute(
        """
        INSERT INTO collections (slug, title, description)
        VALUES (%s, %s, %s)
        ON CONFLICT (slug) DO UPDATE
        SET title = EXCLUDED.title,
            description = COALESCE(EXCLUDED.description, collections.description)
        RETURNING id
        """,
        (slug, title, description),
    )
    return int(cur.fetchone()[0])


def upsert_document(
    cur,
    collection_id: int,
    source_name: str,
    volume: Optional[str],
    source_ref: Optional[str],
    metadata: dict,
) -> int:
    cur.execute(
        """
        INSERT INTO documents (collection_id, source_name, volume, source_ref, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (collection_id, source_name, volume_key) DO UPDATE
        SET source_ref = COALESCE(EXCLUDED.source_ref, documents.source_ref),
            metadata = documents.metadata || EXCLUDED.metadata
        RETURNING id
        """,
        (collection_id, source_name, volume, source_ref, Json(metadata)),
    )
    return int(cur.fetchone()[0])


def delete_pages_for_document(cur, document_id: int):
    # Deterministic replay (OK before chunks exist; later you'll need to delete chunks first or upsert pages).
    cur.execute("DELETE FROM pages WHERE document_id = %s", (document_id,))


# Extract p.xx markers in the page text.
PXX_ANYWHERE_RE = re.compile(r"(?m)^\s*(p\.\s*\d+)\s*$")


def normalize_pxx(label: str) -> str:
    # "p. 71" -> "p.71"
    return re.sub(r"\s+", "", label.strip())


def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    """
    Returns list of dicts:
      pdf_page_number (1-indexed), raw_text (full page)
    """
    out: List[Dict] = []
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            page = doc[i]
            raw = page.get_text("text") or ""
            out.append({"pdf_page_number": i + 1, "raw_text": raw})
    finally:
        doc.close()
    return out


def upsert_page_metadata(cur, page_id: int, pipeline_version: str, meta_raw: dict):
    cur.execute(
        """
        INSERT INTO page_metadata (page_id, pipeline_version, meta_raw)
        VALUES (%s, %s, %s)
        ON CONFLICT (page_id, pipeline_version) DO UPDATE
        SET meta_raw = page_metadata.meta_raw || EXCLUDED.meta_raw,
            extracted_at = now()
        """,
        (page_id, pipeline_version, Json(meta_raw)),
    )


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Vassiliev PDF as original PDF pages; store p.xx markers in metadata only when present."
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Path to a single PDF (WSL path recommended, e.g. /mnt/c/.../file.pdf)")
    g.add_argument("--pdf-dir", help="Directory containing PDFs to ingest (non-recursive).")
    parser.add_argument(
        "--source-key",
        default=None,
        help=(
            "Stable ingest source key. For --pdf, defaults to 'vassiliev:<stem>'. "
            "For --pdf-dir, if you include '{stem}', it will be formatted per PDF; "
            "otherwise it is treated as a prefix and ':<stem>' is appended."
        ),
    )
    parser.add_argument(
        "--volume",
        default="Yellow Notebook #1",
        help=(
            "Document volume label stored on documents.volume. "
            "For --pdf-dir, if you include '{stem}', it will be formatted per PDF; "
            "otherwise it is treated as a prefix and ' – <stem>' is appended."
        ),
    )
    parser.add_argument("--language", default="en", help="Language code for pages (default: en)")
    args = parser.parse_args()

    # Preflight DB connection early so AWS shows a connection immediately (and SSL/auth errors surface fast).
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            params = conn.get_dsn_parameters()
        host = params.get("host", "?")
        dbname = params.get("dbname", "?")
        user = params.get("user", "?")
        port = params.get("port", "?")
        print(f"DB preflight OK: host={host} port={port} dbname={dbname} user={user}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: DB preflight failed: {e}", file=sys.stderr)
        raise

    def iter_pdfs() -> List[Path]:
        if args.pdf:
            return [Path(args.pdf)]
        d = Path(args.pdf_dir)
        if not d.exists() or not d.is_dir():
            print(f"ERROR: --pdf-dir is not a directory: {d}", file=sys.stderr)
            sys.exit(1)
        pdfs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        pdfs.sort(key=lambda p: p.name.lower())
        if not pdfs:
            print(f"No PDFs found in: {d}", file=sys.stderr)
        return pdfs

    def per_pdf_source_key(pdf_path: Path, *, bulk: bool) -> str:
        stem = pdf_path.stem
        if args.source_key:
            if "{stem}" in args.source_key:
                return args.source_key.format(stem=stem)
            return f"{args.source_key}:{stem}" if bulk else args.source_key
        return f"vassiliev:{stem}"

    def per_pdf_volume(pdf_path: Path, *, bulk: bool) -> str:
        stem = pdf_path.stem
        if "{stem}" in args.volume:
            return args.volume.format(stem=stem)
        return f"{args.volume} – {stem}" if bulk else args.volume

    def ingest_one_pdf(pdf_path: Path, *, bulk: bool) -> None:
        try:
            if not pdf_path.exists() or not pdf_path.is_file():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            pdf_bytes = pdf_path.read_bytes()
            pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()

            source_key = per_pdf_source_key(pdf_path, bulk=bulk)
            source_name = pdf_path.name
            volume = per_pdf_volume(pdf_path, bulk=bulk)

            extracted = extract_pdf_pages(pdf_path)
            if len(extracted) == 0:
                raise ValueError(f"PDF had 0 pages? {pdf_path}")

            with connect() as conn, conn.cursor() as cur:
                ensure_ingest_runs_table(cur)
                can_write_page_metadata = page_metadata_table_exists(cur)
                if not can_write_page_metadata:
                    print(
                        "WARNING: page_metadata table not found. "
                        "Pages will be ingested, but p.xx marker metadata will NOT be written.",
                        file=sys.stderr,
                    )

                if not should_run(cur, source_key, pdf_sha256, PIPELINE_VERSION):
                    print(f"No-op: already ingested ({pdf_path.name}) for this pipeline_version and PDF hash.")
                    return

                mark_running(cur, source_key, pdf_sha256, PIPELINE_VERSION)

                collection_id = upsert_collection(
                    cur,
                    slug="vassiliev",
                    title="Vassiliev Notebooks",
                    description="Notebook ingests (canonical unit = PDF page; p.xx markers stored as derived metadata).",
                )

                doc_id = upsert_document(
                    cur,
                    collection_id=collection_id,
                    source_name=source_name,
                    volume=volume,
                    source_ref=str(pdf_path),
                    metadata={
                        "ingested_by": "scripts/ingest_vassiliev_pdf.py",
                        "pipeline_version": PIPELINE_VERSION,
                        "pdf_sha256": pdf_sha256,
                        "split_rule": "canonical pages are original PDF pages (1 row per PDF page)",
                        "meta_pipeline_version": META_PIPELINE_VERSION,
                        "marker_policy": "meta_raw.p_markers key exists only when markers are present",
                    },
                )

                # Deterministic replay
                delete_pages_for_document(cur, doc_id)

                inserted_pages = 0
                inserted_meta = 0

                for item in extracted:
                    pdf_page_number = int(item["pdf_page_number"])
                    raw_text = item["raw_text"]

                    logical_label = f"pdf.{pdf_page_number}"
                    page_seq = pdf_page_number  # stable monotonic

                    cur.execute(
                        """
                        INSERT INTO pages (
                          document_id, logical_page_label, pdf_page_number, page_seq,
                          language, content_role, raw_text
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            doc_id,
                            logical_label,
                            pdf_page_number,
                            page_seq,
                            args.language,
                            "primary",
                            raw_text,
                        ),
                    )
                    page_id = int(cur.fetchone()[0])
                    inserted_pages += 1

                    if can_write_page_metadata:
                        markers = [normalize_pxx(m) for m in PXX_ANYWHERE_RE.findall(raw_text)]
                        markers_unique = dedupe_preserve_order(markers)

                        # Always include provenance keys; only include p_markers/p_marker_first when present.
                        meta_raw = {
                            "source": "scripts/ingest_vassiliev_pdf.py",
                            "note": "p.xx markers stored for downstream chunk labeling; not used for splitting",
                        }
                        if markers_unique:
                            meta_raw["p_markers"] = markers_unique
                            meta_raw["p_marker_first"] = markers_unique[0]

                        upsert_page_metadata(
                            cur,
                            page_id=page_id,
                            pipeline_version=META_PIPELINE_VERSION,
                            meta_raw=meta_raw,
                        )
                        inserted_meta += 1

                mark_success(cur, source_key)
                conn.commit()

            print(f"Done. Ingested {inserted_pages} PDF pages from {pdf_path.name}")
            if inserted_meta:
                print(f"Wrote page_metadata for {inserted_meta} pages (pipeline={META_PIPELINE_VERSION})")
            print(f"source_key={source_key}")
            print(f"pdf_sha256={pdf_sha256}")

        except Exception as e:
            try:
                source_key = locals().get("source_key")  # type: ignore[assignment]
                pdf_sha256 = locals().get("pdf_sha256")  # type: ignore[assignment]
            except Exception:
                source_key = None
                pdf_sha256 = None
            if source_key and pdf_sha256:
                mark_failed_best_effort(source_key, pdf_sha256, str(e))
            raise

    pdf_paths = iter_pdfs()
    if not pdf_paths:
        return

    bulk = args.pdf_dir is not None
    successes = 0
    failures = 0
    for i, pdf_path in enumerate(pdf_paths, start=1):
        if bulk:
            print(f"\n[{i}/{len(pdf_paths)}] Ingesting {pdf_path.name}")
        try:
            ingest_one_pdf(pdf_path, bulk=bulk)
            successes += 1
        except Exception as e:
            failures += 1
            print(f"ERROR: ingest failed for {pdf_path}: {e}", file=sys.stderr)
            if not bulk:
                raise

    if bulk:
        print(f"\nVassiliev ingest complete. successes={successes} failures={failures}", file=sys.stderr)
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
