import os
import re
import sys
import hashlib
import argparse
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

import fitz  # PyMuPDF


PIPELINE_VERSION = "v2_line_level_dict"


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")


def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


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


def upsert_collection(cur, slug: str, title: str, description: str | None = None) -> int:
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
    return cur.fetchone()[0]


def upsert_document(cur, collection_id: int, source_name: str, volume: str | None, source_ref: str | None, metadata: dict) -> int:
    # Uses UNIQUE(collection_id, source_name, volume_key) from your schema
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
    return cur.fetchone()[0]


def delete_pages_for_document(cur, document_id: int):
    # Safe right now because you aren't creating chunks yet; ensures deterministic page_seq and avoids uniqueness collisions.
    cur.execute("DELETE FROM pages WHERE document_id = %s", (document_id,))


PXX_RE = re.compile(r"(?m)^\s*(p\.\s*\d+)\s*$")


def normalize_label(label: str) -> str:
    # "p. 71" -> "p.71"
    return re.sub(r"\s+", "", label.strip())


PXX_LINE_RE = re.compile(r"^\s*(p\.\s*\d+)\s*$")

def extract_pxx_pages(pdf_path: Path) -> list[dict]:
    """
    Layout-aware, line-level extraction:
      - iterates lines (not blocks)
      - detects marker lines like 'p.71'
      - assigns subsequent lines until next marker line
    Returns list of dicts:
      logical_page_label, pdf_page_number, raw_text
    """
    out: list[dict] = []
    doc = fitz.open(str(pdf_path))
    try:
        for pdf_idx in range(len(doc)):
            page = doc[pdf_idx]
            d = page.get_text("dict")
            lines = []

            # Collect (y0, x0, text) for each line
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue  # non-text block
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    # Combine spans into the visual line text (preserve as-is)
                    text = "".join(s.get("text", "") for s in spans)
                    if not text or not text.strip():
                        continue

                    x0, y0, x1, y1 = line.get("bbox", [0, 0, 0, 0])
                    lines.append((float(y0), float(x0), text))

            if not lines:
                continue

            # Sort top-to-bottom, then left-to-right
            lines.sort(key=lambda t: (t[0], t[1]))

            # Find indices of marker lines
            marker_idxs = []
            marker_labels = []
            for i, (_, __, txt) in enumerate(lines):
                m = PXX_LINE_RE.match(txt.strip())
                if m:
                    label = normalize_label(m.group(1))  # your existing helper
                    marker_idxs.append(i)
                    marker_labels.append(label)

            if not marker_idxs:
                continue

            # Segment lines between markers
            for k, start_i in enumerate(marker_idxs):
                end_i = marker_idxs[k + 1] if k + 1 < len(marker_idxs) else len(lines)
                label = marker_labels[k]

                # Include the marker line and all following lines up to (not including) next marker
                segment_lines = [lines[j][2] for j in range(start_i, end_i)]

                # Join with newlines; do not strip internal whitespace
                raw_text = "\n".join(segment_lines)

                out.append(
                    {
                        "logical_page_label": label,
                        "pdf_page_number": pdf_idx + 1,
                        "raw_text": raw_text,
                    }
                )
    finally:
        doc.close()

    return out




def main():
    parser = argparse.ArgumentParser(description="Ingest Vassiliev PDF pages using p.xx markers as logical pages.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF (WSL path recommended, e.g. /mnt/c/.../file.pdf)")
    parser.add_argument("--source-key", default=None, help="Stable ingest source key (default derived from filename)")
    parser.add_argument("--volume", default="Yellow Notebook #1", help="Document volume label (stored on documents.volume)")
    parser.add_argument("--language", default="en", help="Language code for pages (default: en)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    pdf_bytes = pdf_path.read_bytes()
    pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    source_key = args.source_key or f"vassiliev:{pdf_path.stem}"
    source_name = pdf_path.name

    # Extract p.xx-defined pages
    extracted = extract_pxx_pages(pdf_path)
    if len(extracted) < 10:
        print(f"WARNING: extracted only {len(extracted)} p.xx pages. Check marker regex or PDF text.", file=sys.stderr)

    # Ensure unique logical_page_label within this document (your schema enforces it).
    # If duplicates occur, we suffix them deterministically while keeping original p.xx line inside raw_text.
    seen: dict[str, int] = {}
    pages: list[dict] = []
    seq = 0
    for item in extracted:
        seq += 1
        base_label = item["logical_page_label"]
        n = seen.get(base_label, 0) + 1
        seen[base_label] = n
        label = base_label if n == 1 else f"{base_label}__dup{n}"

        pages.append(
            {
                "logical_page_label": label,
                "pdf_page_number": item["pdf_page_number"],
                "page_seq": seq,
                "language": args.language,
                "content_role": "primary",
                "raw_text": item["raw_text"],
            }
        )

    try:
        with connect() as conn, conn.cursor() as cur:
            ensure_ingest_runs_table(cur)

            if not should_run(cur, source_key, pdf_sha256, PIPELINE_VERSION):
                print("No-op: already ingested for this pipeline_version and PDF hash.")
                return

            mark_running(cur, source_key, pdf_sha256, PIPELINE_VERSION)

            collection_id = upsert_collection(
                cur,
                slug="vassiliev",
                title="Vassiliev Notebooks",
                description="Notebook ingests (logical pages defined by p.xx markers; PDF page retained).",
            )

            doc_id = upsert_document(
                cur,
                collection_id=collection_id,
                source_name=source_name,
                volume=args.volume,
                source_ref=str(pdf_path),
                metadata={
                    "ingested_by": "scripts/ingest_vassiliev_pdf.py",
                    "pipeline_version": PIPELINE_VERSION,
                    "pdf_sha256": pdf_sha256,
                    "split_rule": "logical pages split by ^p\\.\\s*\\d+ markers",
                },
            )

            # Deterministic replay: replace pages for this doc on rerun
            delete_pages_for_document(cur, doc_id)

            # Insert pages
            for p in pages:
                cur.execute(
                    """
                    INSERT INTO pages (
                      document_id, logical_page_label, pdf_page_number, page_seq,
                      language, content_role, raw_text
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        doc_id,
                        p["logical_page_label"],
                        p["pdf_page_number"],
                        p["page_seq"],
                        p["language"],
                        p["content_role"],
                        p["raw_text"],
                    ),
                )

            mark_success(cur, source_key)
            conn.commit()

        print(f"Done. Ingested {len(pages)} logical pages from {pdf_path.name}")
        print(f"source_key={source_key}")
        print(f"pdf_sha256={pdf_sha256}")

    except Exception as e:
        mark_failed_best_effort(source_key, pdf_sha256, str(e))
        raise


if __name__ == "__main__":
    main()
