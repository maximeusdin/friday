#!/usr/bin/env python3
"""
Ingest Venona cable PDFs into Postgres using MESSAGE-LEVEL pages (one row per message/cable).

- Messages are detected by:
    * Presence of "USSR Ref. No" anywhere on the page, OR
    * A tight cluster of From/To/No lines near the top.

- Messages may span multiple PDF pages. We concatenate lines until the next message start.

- pages table:
    * 1 row per message
    * page_seq = message index within document (1..N)
    * pdf_page_number = start PDF page (1-indexed)
    * logical_page_label = derived from USSR Ref No or From/To/No (deduped with __dupN)

- page_metadata table:
    * Stores derived, rerunnable metadata (raw_text is NEVER mutated)
    * pipeline_versioned via META_PIPELINE_VERSION

Metadata extracted (best-effort, missing allowed):
    - USSR Ref. No
    - From (routing)
    - To (routing)  [can be multiple]
    - Cable number(s) (header No token; stored as cable_nos array + cable_no primary)
    - Date (message date from header No remainder or standalone date line; supports "10th Oct. 40")
    - Reissue / Extract flags
    - Issued date, Copy No (common in London volumes)
    - MB code (GRU volumes sometimes)
    - Signature-like "No. #### NAME" lines (captured, but not confused with cable No)

Requires:
    - tables from 002_schema.sql
    - page_metadata table with UNIQUE(page_id, pipeline_version)
"""

import os
import re
import sys
import json
import hashlib
import argparse
import datetime
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import psycopg2
from psycopg2.extras import Json

import fitz  # PyMuPDF


PIPELINE_VERSION = "v3_message_pages_with_metadata"
META_PIPELINE_VERSION = "meta_v3_venona_headers"


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


# ----------------------------
# Ingest runs bookkeeping
# ----------------------------
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


# ----------------------------
# DB helpers
# ----------------------------
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
    # If your page_metadata has FK ON DELETE CASCADE to pages, this is enough.
    # If it's RESTRICT, delete page_metadata first (see comment in main()).
    cur.execute("DELETE FROM pages WHERE document_id=%s", (document_id,))


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


def upsert_page_metadata(cur, page_id: int, pipeline_version: str, meta_raw: dict):
    cur.execute(
        """
        INSERT INTO page_metadata (page_id, pipeline_version, meta_raw)
        VALUES (%s, %s, %s)
        ON CONFLICT (page_id, pipeline_version) DO UPDATE
        SET meta_raw = EXCLUDED.meta_raw,
            extracted_at = now()
        """,
        (page_id, pipeline_version, Json(meta_raw)),
    )


# ----------------------------
# Parsing helpers
# ----------------------------

# Strong message start signal
USSR_REF_ANY_RE = re.compile(r"(?i)\bUSSR\s+Ref\.?\s*No\.?\s*:\s*([A-Za-z0-9/()\-]+)")

# Routing header
FROM_RE = re.compile(r"(?i)^\s*From\s*[:\-–]\s*(.+?)\s*$")
TO_RE   = re.compile(r"(?i)^\s*To\s*[:\-–]\s*(.+?)\s*$")

# Generic No line (used for start detection window)
NO_LINE_RE = re.compile(r"(?i)^\s*No\.?\s*:?\s*(.+?)\s*$")

# Header "No" line (cable number + optional remainder/date)
# Allow alnum token (some variants include slashes, etc.)
NO_HDR_RE = re.compile(r"(?i)^\s*No\.?\s*:?\s*([A-Za-z0-9/\-]{1,20})(?:\s+(.*?))?\s*$")

# Issued / Copy No patterns common in some volumes
ISSUED_RE = re.compile(r"(?i)^\s*Issued\s*:?\s*(?:/)?\s*([0-9]{1,2})\s*/\s*([0-9]{1,2})\s*/\s*([0-9]{2,4})\s*$")
COPYNO_RE = re.compile(r"(?i)^\s*Copy\s*No\.?\s*:?\s*([0-9]{1,6})\s*$")

# Reissue / extract
REISSUE_RE = re.compile(r"(?i)\b(?:(\d+)\s*(?:st|nd|rd|th)\s*)?RE[\- ]?ISSUE\b")
EXTRACT_RE = re.compile(r"(?i)\bEXTRACT\b")

# Front matter hint
FRONT_MATTER_HINT_RE = re.compile(
    r"(?i)\b(venona|arranged by|john earl haynes|national archives|introduction|contents|table of contents)\b"
)

# Signature-ish lines later in messages (avoid confusing with header No)
NO_SIG_RE = re.compile(r"(?i)^\s*No\.?\s*([0-9]{1,6})\s+([A-ZĀĒĪŌŪA-Z’'\-]+)\b(.*)?$")

# Numeric date fallback (dd/mm/yy or dd/mm/yyyy)
NUMERIC_DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*$")

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def _clean_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return s.strip()


def strip_bracket_footnotes(s: str) -> str:
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"\s+", " ", s)
    return _clean_ascii(s).strip(" .;:")


def split_addressees(s: str) -> List[str]:
    s = strip_bracket_footnotes(s).rstrip(".")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [p for p in parts if p]


def parse_numeric_date(s: str) -> Tuple[str, Optional[str]]:
    raw = _clean_ascii(s)
    m = NUMERIC_DATE_RE.match(raw)
    if not m:
        return raw, None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y = 1900 + y
    try:
        return raw, datetime.date(y, mo, d).isoformat()
    except ValueError:
        return raw, None


def parse_textual_date(s: str) -> Tuple[str, Optional[str]]:
    """
    Supports:
      - '16 Sept.45'
      - '17 July 40'
      - '24 March 1942'
      - '10th Oct. 40'
      - '21st September 1945'
    Returns (raw, iso_or_none)
    """
    raw = _clean_ascii(s)
    t = raw.lower()

    # Normalize punctuation/spaces
    t = t.replace(".", " ")
    t = re.sub(r"[,\s]+", " ", t).strip()

    # Remove ordinals: 10th -> 10
    t = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t)

    parts = t.split()
    if len(parts) < 3:
        return raw, None

    # Expect: day month year
    try:
        day = int(parts[0])
    except ValueError:
        return raw, None

    mon = MONTHS.get(parts[1], MONTHS.get(parts[1][:3], None))
    if not mon:
        return raw, None

    try:
        year = int(parts[2])
    except ValueError:
        return raw, None

    if year < 100:
        year = 1900 + year

    try:
        iso = datetime.date(year, mon, day).isoformat()
    except ValueError:
        iso = None
    return raw, iso


def parse_any_date(s: str) -> Tuple[str, Optional[str]]:
    raw = _clean_ascii(s)
    _, iso = parse_numeric_date(raw)
    if iso:
        return raw, iso
    return parse_textual_date(raw)


def page_lines_in_reading_order(page: fitz.Page) -> List[str]:
    """
    Extract lines in a stable reading order using get_text('dict').
    Join spans for each line and sort by (y, x).
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
    s = re.sub(r"[^A-Za-z0-9 .:_\-/,#()|]", "", s)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s or "message"


def detect_message_start(lines: List[str]) -> bool:
    """
    Conservative start detection:
      - If page contains USSR Ref. No anywhere, that's a strong signal.
      - Else: require From + To + No within a small window near top.
    """
    joined = "\n".join(lines)
    if USSR_REF_ANY_RE.search(joined):
        return True

    idx_from = idx_to = idx_no = None
    for i, ln in enumerate(lines[:90]):
        s = ln.strip()
        if idx_from is None and FROM_RE.match(s):
            idx_from = i
        if idx_to is None and TO_RE.match(s):
            idx_to = i
        if idx_no is None and NO_LINE_RE.match(s):
            idx_no = i
        if idx_from is not None and idx_to is not None and idx_no is not None:
            span = max(idx_from, idx_to, idx_no) - min(idx_from, idx_to, idx_no)
            if span <= 15 and min(idx_from, idx_to, idx_no) <= 45:
                return True
    return False


def extract_label_from_header(lines: List[str]) -> Tuple[Optional[str], dict]:
    """
    Build a stable logical label + minimal header_meta for labeling.
    """
    header_meta: dict = {}
    joined = "\n".join(lines[:140])

    mref = USSR_REF_ANY_RE.search(joined)
    if mref:
        header_meta["ussr_ref_no"] = mref.group(1)
        return f"USSRRef:{mref.group(1)}", header_meta

    frm = to = no = None
    for ln in lines[:140]:
        s = ln.strip()
        mf = FROM_RE.match(s)
        if mf and frm is None:
            frm = mf.group(1).strip()
        mt = TO_RE.match(s)
        if mt and to is None:
            to = mt.group(1).strip()
        mn = NO_LINE_RE.match(s)
        if mn and no is None:
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


def extract_venona_metadata_from_message(raw_text: str) -> dict:
    """
    Extract best-effort metadata from the message text (no mutation).
    """
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    header_zone = lines[:160]

    meta: Dict[str, Any] = {
        "ussr_ref_no": None,
        "issued_date_raw": None,
        "issued_date_iso": None,
        "copy_no": None,
        "reissue_raw": None,
        "reissue_ordinal": None,
        "extract_flag": False,
        "mb_code": None,

        "from": None,
        "to": [],              # routing To(s)
        "to_addressees": [],   # addressee To line(s), not routing

        "cable_no": None,
        "cable_nos": [],
        "cable_no_raw": None,

        "message_date_raw": None,
        "message_date_iso": None,

        "signature_no_lines": []
    }

    joined = "\n".join(header_zone)

    # USSR Ref No
    m = USSR_REF_ANY_RE.search(joined)
    if m:
        meta["ussr_ref_no"] = m.group(1)

    # MB code (GRU-style)
    mbm = re.search(r"(?i)\bMB\s*([0-9]{1,6}\s*/\s*[0-9]{1,6})\b", joined)
    if mbm:
        meta["mb_code"] = "MB " + mbm.group(1).replace(" ", "")

    # Issued / Copy / Reissue / Extract flags
    for ln in header_zone:
        s = ln.strip()

        mi = ISSUED_RE.match(s)
        if mi and meta["issued_date_raw"] is None:
            d, mth, y = int(mi.group(1)), int(mi.group(2)), int(mi.group(3))
            meta["issued_date_raw"] = s.split(":", 1)[-1].strip()
            if y < 100:
                y = 1900 + y
            try:
                meta["issued_date_iso"] = datetime.date(y, mth, d).isoformat()
            except ValueError:
                meta["issued_date_iso"] = None

        mc = COPYNO_RE.match(s)
        if mc and meta["copy_no"] is None:
            meta["copy_no"] = mc.group(1)

        mr = REISSUE_RE.search(s)
        if mr and meta["reissue_raw"] is None:
            meta["reissue_raw"] = _clean_ascii(s)
            if mr.group(1):
                try:
                    meta["reissue_ordinal"] = int(mr.group(1))
                except ValueError:
                    meta["reissue_ordinal"] = None

        if EXTRACT_RE.search(s):
            meta["extract_flag"] = True

    # Routing From/To and header No/date parsing
    for ln in header_zone:
        s = ln.strip()

        mf = FROM_RE.match(s)
        if mf and meta["from"] is None:
            meta["from"] = strip_bracket_footnotes(mf.group(1))

        mt = TO_RE.match(s)
        if mt:
            dest = strip_bracket_footnotes(mt.group(1))
            if dest and dest not in meta["to"]:
                meta["to"].append(dest)
            # IMPORTANT: don't treat routing "To:" line as addressee line too
            continue

        # Header No line: cable token + remainder (often date)
        mn = NO_HDR_RE.match(s)
        if mn and meta["cable_no"] is None:
            token = strip_bracket_footnotes(mn.group(1))
            rest = (mn.group(2) or "").strip()

            meta["cable_no"] = token
            meta["cable_nos"] = [token]
            meta["cable_no_raw"] = (token + (" " + rest if rest else "")).strip()

            # Try parse date from remainder
            if rest:
                raw, iso = parse_any_date(rest)
                if iso:
                    meta["message_date_raw"] = raw
                    meta["message_date_iso"] = iso

    # Standalone message date line (e.g., after No: 582 then "24 March 1942")
    if meta["message_date_raw"] is None:
        for ln in header_zone[:80]:
            raw, iso = parse_any_date(ln.strip())
            if iso:
                meta["message_date_raw"] = raw
                meta["message_date_iso"] = iso
                break

    # Addressee "To ..." lines inside header/body (not routing)
    # Heuristic: scan header_zone for lines that begin with "To" but are NOT routing "To: DEST"
    for ln in header_zone:
        s = ln.strip()
        if not s.lower().startswith("to"):
            continue
        if TO_RE.match(s):  # routing already handled
            continue
        # E.g., "To IGOR'.", "To SERGEJ, IGOR’ ..."
        # Avoid things like "TO:" empty weirdness
        after = re.sub(r"(?i)^\s*To\s*:?\s*", "", s).strip()
        if not after:
            continue
        addrs = split_addressees(after)
        for a in addrs:
            if a and (a not in meta["to_addressees"]) and (a not in meta["to"]):
                meta["to_addressees"].append(a)

    # Signature-ish No lines later (store but don't confuse with header cable No)
    scan_zone = lines[:280]
    for idx, ln in enumerate(scan_zone):
        ms = NO_SIG_RE.match(ln.strip())
        if not ms:
            continue
        no = ms.group(1)
        who = strip_bracket_footnotes(ms.group(2))
        tail = (ms.group(3) or "").strip()

        date_raw = None
        nxt = scan_zone[idx + 1].strip() if idx + 1 < len(scan_zone) else ""
        if nxt:
            _, iso = parse_any_date(nxt)
            if iso:
                date_raw = _clean_ascii(nxt)

        meta["signature_no_lines"].append(
            {
                "no": no,
                "who": who,
                "date_raw": date_raw or (_clean_ascii(tail) if tail else None),
            }
        )

    return meta


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Ingest Venona cable PDFs into pages (message/cable units) + write page_metadata."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Path to a single PDF (WSL path recommended).")
    g.add_argument("--pdf-dir", help="Directory containing PDFs to ingest (non-recursive).")
    ap.add_argument(
        "--source-key",
        default=None,
        help=(
            "Stable ingest key. For --pdf, defaults to '<collection-slug>:<stem>'. "
            "For --pdf-dir, if you include '{stem}', it will be formatted per PDF; "
            "otherwise it is treated as a prefix and ':<stem>' is appended."
        ),
    )
    ap.add_argument(
        "--volume",
        default=None,
        help=(
            "Document volume label stored on documents.volume. "
            "For --pdf-dir, if you include '{stem}', it will be formatted per PDF; "
            "otherwise it is treated as a prefix and ' – <stem>' is appended. "
            "If omitted in --pdf-dir mode, defaults to '<CollectionSlugCapitalized> – <stem>'."
        ),
    )
    ap.add_argument("--language", default="en", help="pages.language (default: en).")
    ap.add_argument("--collection-slug", default="venona", help="collections.slug (default: venona).")
    ap.add_argument("--collection-title", default="Venona Decrypts", help="collections.title.")
    ap.add_argument(
        "--collection-description",
        default="Venona cable ingests. Logical pages are message/cable units; pdf_page_number is start page of message.",
        help="collections.description.",
    )
    args = ap.parse_args()

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
        return f"{args.collection_slug}:{stem}"

    def per_pdf_volume(pdf_path: Path, *, bulk: bool) -> Optional[str]:
        stem = pdf_path.stem
        if args.volume is None:
            if bulk:
                return f"{args.collection_slug.capitalize()} – {stem}"
            return None
        if "{stem}" in args.volume:
            return args.volume.format(stem=stem)
        return f"{args.volume} – {stem}" if bulk else args.volume

    def ingest_one_pdf(pdf_path: Path, *, bulk: bool) -> None:
        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        sha: Optional[str] = None
        source_key = per_pdf_source_key(pdf_path, bulk=bulk)
        volume = per_pdf_volume(pdf_path, bulk=bulk)
        source_name = pdf_path.name

        try:
            pdf_bytes = pdf_path.read_bytes()
            sha = hashlib.sha256(pdf_bytes).hexdigest()

            # ----------------------------
            # 1) Parse PDF into messages
            # ----------------------------
            messages: List[dict] = []
            current: Optional[dict] = None
            page_seq = 0

            doc = fitz.open(str(pdf_path))
            try:
                for pdf_idx in range(len(doc)):
                    page = doc[pdf_idx]
                    lines = page_lines_in_reading_order(page)

                    # If we aren't in a message yet, try to start one
                    if current is None:
                        if not detect_message_start(lines):
                            if FRONT_MATTER_HINT_RE.search("\n".join(lines[:70])) and pdf_idx < 6:
                                continue
                            continue

                        label, header_meta = extract_label_from_header(lines)
                        current = {
                            "start_pdf_page": pdf_idx + 1,
                            "lines": list(lines),
                            "label": label,
                            "header_meta": header_meta,
                        }
                        continue

                    # If we are inside a message, check if a new message starts here
                    if detect_message_start(lines):
                        page_seq += 1
                        raw_text = "\n".join(current["lines"])
                        messages.append(
                            {
                                "page_seq": page_seq,
                                "pdf_page_number": current["start_pdf_page"],
                                "logical_page_label": current["label"] or f"msg.{page_seq:04d}",
                                "raw_text": raw_text,
                            }
                        )

                        label, header_meta = extract_label_from_header(lines)
                        current = {
                            "start_pdf_page": pdf_idx + 1,
                            "lines": list(lines),
                            "label": label,
                            "header_meta": header_meta,
                        }
                    else:
                        current["lines"].extend(lines)

                # Flush last message
                if current is not None:
                    page_seq += 1
                    raw_text = "\n".join(current["lines"])
                    messages.append(
                        {
                            "page_seq": page_seq,
                            "pdf_page_number": current["start_pdf_page"],
                            "logical_page_label": current["label"] or f"msg.{page_seq:04d}",
                            "raw_text": raw_text,
                        }
                    )
            finally:
                doc.close()

            if not messages:
                raise ValueError("extracted 0 messages (PDF may not match Venona-cables format)")

            # ----------------------------
            # 2) Normalize/dedupe labels
            # ----------------------------
            seen: Dict[str, int] = {}
            pages: List[dict] = []
            for msg in messages:
                base = normalize_label(msg["logical_page_label"])
                n = seen.get(base, 0) + 1
                seen[base] = n
                label = base if n == 1 else f"{base}__dup{n}"
                pages.append(
                    {
                        "logical_page_label": label,
                        "pdf_page_number": int(msg["pdf_page_number"]),
                        "page_seq": int(msg["page_seq"]),
                        "language": args.language,
                        "content_role": "primary",
                        "raw_text": msg["raw_text"],
                    }
                )

            # ----------------------------
            # 3) Write to DB (idempotent per source_key+sha+pipeline_version)
            # ----------------------------
            with connect() as conn, conn.cursor() as cur:
                ensure_ingest_runs_table(cur)

                can_write_page_metadata = page_metadata_table_exists(cur)
                if not can_write_page_metadata:
                    print(
                        "WARNING: page_metadata table not found; header metadata will not be stored.",
                        file=sys.stderr,
                    )

                if not should_run(cur, source_key, sha, PIPELINE_VERSION):
                    print(f"No-op: already ingested ({pdf_path.name}) for this pipeline_version and PDF hash.")
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
                    volume=volume,
                    source_ref=str(pdf_path),
                    metadata={
                        "ingested_by": "scripts/ingest_venona_pdf.py",
                        "pipeline_version": PIPELINE_VERSION,
                        "meta_pipeline_version": META_PIPELINE_VERSION,
                        "pdf_sha256": sha,
                        "split_rule": "messages split by USSR Ref No or From/To/No header clusters; continuation pages appended",
                    },
                )

                delete_pages_for_document(cur, doc_id)

                for p in pages:
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
                            p["logical_page_label"],
                            p["pdf_page_number"],
                            p["page_seq"],
                            p["language"],
                            p["content_role"],
                            p["raw_text"],
                        ),
                    )
                    page_id = int(cur.fetchone()[0])

                    if can_write_page_metadata:
                        ven_meta = extract_venona_metadata_from_message(p["raw_text"])
                        upsert_page_metadata(
                            cur,
                            page_id=page_id,
                            pipeline_version=META_PIPELINE_VERSION,
                            meta_raw={"venona": ven_meta, "source": "scripts/ingest_venona_pdf.py"},
                        )

                mark_success(cur, source_key)
                conn.commit()

            print(f"Done. Ingested {len(pages)} messages from {source_name}")
            print(f"source_key={source_key}")
            print(f"sha256={sha}")

        except Exception as e:
            # Record failure so reruns will retry this PDF.
            if sha is not None:
                mark_failed_best_effort(source_key, sha, str(e))
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
        print(f"\nVenona ingest complete. successes={successes} failures={failures}", file=sys.stderr)
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
