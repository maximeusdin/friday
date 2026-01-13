import os
import re
import sys
import hashlib
import argparse
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

import fitz  # PyMuPDF

PIPELINE_VERSION = "v0_venona_cables_state_machine"

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


def should_run(cur, source_key: str, sha: str, pipeline_version: str) -> bool:
    cur.execute(
        "SELECT source_sha256, pipeline_version, status FROM ingest_runs WHERE source_key=%s",
        (source_key,),
    )
    row = cur.fetchone()
    return (
        row is None
        or row[0] != sha
        or row[1] != pipeline_version
        or row[2] != "success"
    )


def mark_running(cur, source_key: str, sha: str, pipeline_version: str):
    cur.execute(
        """
        INSERT INTO ingest_runs (source_key, source_sha256, pipeline_version, status, started_at)
        VALUES (%s, %s, %s, 'running', now())
        ON CONFLICT (source_key) DO UPDATE
        SET source_sha256 = EXCLUDED.source_sha256,
            pipeline_version = EXCLUDED.pipeline_version,
            status='running',
            started_at=now(),
            finished_at=NULL,
            error=NULL
        """,
        (source_key, sha, pipeline_version),
    )


def mark_success(cur, source_key: str):
    cur.execute(
        "UPDATE ingest_runs SET status='success', finished_at=now() WHERE source_key=%s",
        (source_key,),
    )


def mark_failed_best_effort(source_key: str, sha: str, err: str):
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
                (source_key, sha, PIPELINE_VERSION, err),
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
    cur.execute("DELETE FROM pages WHERE document_id=%s", (document_id,))


# --- Parsing helpers ---

# USSR Ref. No formatting varies ("USSR Ref. No:" vs "USSR Ref No:")
USSR_REF_RE = re.compile(r"(?i)\bUSSR\s+Ref\.?\s*No\.?\s*:\s*([A-Za-z0-9/\-]+)")

# Venona header lines vary: colon or dash/en-dash
FROM_RE = re.compile(r"(?i)^\s*From\s*[:\-–]\s*(.+?)\s*$")
TO_RE   = re.compile(r"(?i)^\s*To\s*[:\-–]\s*(.+?)\s*$")

# "No:" / "No.:" / "No." variants; content may include commas
NO_RE   = re.compile(r"(?i)^\s*No\.?\s*:?\s*(.+?)\s*$")

# Cover pages / non-message pages often contain these; used only to reduce false positives.
FRONT_MATTER_HINT_RE = re.compile(r"(?i)\b(venona|arranged by|john earl haynes|national archives|introduction|contents|table of contents)\b")

# Page header artifacts that appear mid-message; do NOT trigger splits
PAGE_NUMBER_ARTIFACT_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")


def page_lines_in_reading_order(page: fitz.Page) -> list[str]:
    """
    Extract lines in a stable reading order using get_text('dict').
    We join spans for each line and sort by (y, x).
    """
    d = page.get_text("dict")
    lines = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans)
            if not text or not text.strip():
                continue
            x0, y0, x1, y1 = line.get("bbox", [0, 0, 0, 0])
            lines.append((float(y0), float(x0), text))
    lines.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in lines]


def normalize_label(s: str, max_len: int = 110) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # Keep safe-ish characters for readability
    s = re.sub(r"[^A-Za-z0-9 .:_\-/,#()]", "", s)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s or "message"


def detect_message_start(lines: list[str]) -> bool:
    """
    Conservative start detection:
      - If page contains USSR Ref. No, that's a strong signal.
      - Else: require From + To + No within a small window.
    """
    joined = "\n".join(lines)
    if USSR_REF_RE.search(joined):
        return True

    # Sliding window to find From/To/No cluster close together
    idx_from = idx_to = idx_no = None
    for i, ln in enumerate(lines[:80]):  # don't scan entire huge pages
        if idx_from is None and FROM_RE.match(ln.strip()):
            idx_from = i
        if idx_to is None and TO_RE.match(ln.strip()):
            idx_to = i
        if idx_no is None and NO_RE.match(ln.strip()):
            # guard against false "No." in body by requiring it's near top
            idx_no = i
        if idx_from is not None and idx_to is not None and idx_no is not None:
            # cluster must be reasonably close
            span = max(idx_from, idx_to, idx_no) - min(idx_from, idx_to, idx_no)
            if span <= 15 and min(idx_from, idx_to, idx_no) <= 40:
                return True

    return False


def extract_label_from_header(lines: list[str]) -> tuple[str | None, dict]:
    """
    Try to build a stable logical label + header metadata from the first ~80 lines.
    Returns (label, header_meta).
    """
    header_meta: dict = {}

    joined = "\n".join(lines[:120])
    mref = USSR_REF_RE.search(joined)
    if mref:
        header_meta["ussr_ref_no"] = mref.group(1)
        # Prefer USSR Ref No as label
        return f"USSRRef:{mref.group(1)}", header_meta

    # Otherwise capture From/To/No (best-effort)
    frm = to = no = None
    for ln in lines[:120]:
        s = ln.strip()
        mf = FROM_RE.match(s)
        if mf and frm is None:
            frm = mf.group(1).strip()
        mt = TO_RE.match(s)
        if mt and to is None:
            to = mt.group(1).strip()
        mn = NO_RE.match(s)
        if mn and no is None:
            # This may include date too; keep as-is
            no = mn.group(1).strip()
        if frm and to and no:
            break

    if frm:
        header_meta["from"] = frm
    if to:
        header_meta["to"] = to
    if no:
        header_meta["no"] = no

    if frm or to or no:
        pieces = []
        if frm:
            pieces.append(f"From:{normalize_label(frm, 40)}")
        if to:
            pieces.append(f"To:{normalize_label(to, 40)}")
        if no:
            pieces.append(f"No:{normalize_label(no, 50)}")
        return " | ".join(pieces), header_meta

    return None, header_meta


def main():
    ap = argparse.ArgumentParser(description="Ingest Venona cable PDFs into pages (one page per message/cable).")
    ap.add_argument("--pdf", required=True, help="Path to PDF (WSL path recommended).")
    ap.add_argument("--source-key", default=None, help="Stable ingest key (default: venona:<filename-stem>).")
    ap.add_argument("--volume", default=None, help="Optional volume label stored on documents.volume.")
    ap.add_argument("--language", default="en", help="pages.language (default: en).")
    ap.add_argument("--collection-slug", default="venona", help="collections.slug (default: venona).")
    ap.add_argument("--collection-title", default="Venona Decrypts", help="collections.title.")
    ap.add_argument("--collection-description", default="Venona cable ingests. Logical pages are message/cable units.", help="collections.description.")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    pdf_bytes = pdf_path.read_bytes()
    sha = hashlib.sha256(pdf_bytes).hexdigest()

    source_key = args.source_key or f"venona:{pdf_path.stem}"
    source_name = pdf_path.name

    # Extract messages by scanning pages in order
    messages = []
    current = None  # dict with: start_pdf_page, lines(list[str]), header_meta(dict), label(str|None)
    page_seq = 0

    doc = fitz.open(str(pdf_path))
    try:
        for pdf_idx in range(len(doc)):
            page = doc[pdf_idx]
            lines = page_lines_in_reading_order(page)

            # Light front-matter skip heuristic if no current message and no start markers
            if current is None:
                if not detect_message_start(lines):
                    # If it looks like cover/intro, skip
                    if FRONT_MATTER_HINT_RE.search("\n".join(lines[:60])) and pdf_idx < 5:
                        continue
                    # Otherwise just continue (blank/odd pages)
                    continue

                # Start new message
                label, header_meta = extract_label_from_header(lines)
                current = {
                    "start_pdf_page": pdf_idx + 1,  # 1-indexed
                    "lines": list(lines),
                    "label": label,
                    "header_meta": header_meta,
                }
                continue

            # If we are inside a message:
            # Check if THIS page begins a NEW message. If so, flush current and start new.
            if detect_message_start(lines):
                # Flush current
                page_seq += 1
                raw_text = "\n".join(current["lines"])
                messages.append(
                    {
                        "page_seq": page_seq,
                        "pdf_page_number": current["start_pdf_page"],
                        "logical_page_label": current["label"] or f"msg.{page_seq:04d}",
                        "header_meta": current["header_meta"],
                        "raw_text": raw_text,
                    }
                )

                # Start next
                label, header_meta = extract_label_from_header(lines)
                current = {
                    "start_pdf_page": pdf_idx + 1,
                    "lines": list(lines),
                    "label": label,
                    "header_meta": header_meta,
                }
            else:
                # Continuation page: append lines verbatim-ish
                current["lines"].extend(lines)

        # Flush last
        if current is not None:
            page_seq += 1
            raw_text = "\n".join(current["lines"])
            messages.append(
                {
                    "page_seq": page_seq,
                    "pdf_page_number": current["start_pdf_page"],
                    "logical_page_label": current["label"] or f"msg.{page_seq:04d}",
                    "header_meta": current["header_meta"],
                    "raw_text": raw_text,
                }
            )
    finally:
        doc.close()

    # Ensure labels are unique per document
    seen = {}
    pages = []
    for msg in messages:
        base = normalize_label(msg["logical_page_label"])
        n = seen.get(base, 0) + 1
        seen[base] = n
        label = base if n == 1 else f"{base}__dup{n}"

        pages.append(
            {
                "logical_page_label": label,
                "pdf_page_number": msg["pdf_page_number"],
                "page_seq": msg["page_seq"],
                "language": args.language,
                "content_role": "primary",
                "raw_text": msg["raw_text"],
                "header_meta": msg["header_meta"],
            }
        )

    if len(pages) == 0:
        print("ERROR: extracted 0 messages. This PDF may not match the Venona-cables format.", file=sys.stderr)
        sys.exit(2)

    try:
        with connect() as conn, conn.cursor() as cur:
            ensure_ingest_runs_table(cur)
            if not should_run(cur, source_key, sha, PIPELINE_VERSION):
                print("No-op: already ingested for this pipeline_version and PDF hash.")
                return

            mark_running(cur, source_key, sha, PIPELINE_VERSION)

            collection_id = upsert_collection(
                cur,
                slug=args.collection_slug,
                title=args.collection_title,
                description=args.collection_description,
            )

            doc_id = upsert_document(
                cur,
                collection_id=collection_id,
                source_name=source_name,
                volume=args.volume,
                source_ref=str(pdf_path),
                metadata={
                    "ingested_by": "scripts/ingest_venona_cables_pdf.py",
                    "pipeline_version": PIPELINE_VERSION,
                    "pdf_sha256": sha,
                    "split_rule": "messages split by USSR Ref No or From/To/No header clusters; continuation pages appended",
                },
            )

            # Deterministic replay: replace all pages for this doc
            delete_pages_for_document(cur, doc_id)

            # Insert pages. Store per-page header_meta inside raw_text? No. Keep raw_text sacred.
            # For now, we tuck header_meta into documents.metadata only by omission; if you later want it per-page,
            # add a JSONB column or a separate table. Today: keep it simple.
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

        print(f"Done. Ingested {len(pages)} messages from {source_name}")
        print(f"source_key={source_key}")
        print(f"sha256={sha}")

    except Exception as e:
        mark_failed_best_effort(source_key, sha, str(e))
        raise


if __name__ == "__main__":
    main()
