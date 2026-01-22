#!/usr/bin/env python3
"""
Chunking runner for archive.

Goals:
- Derived, rerunnable chunking (does NOT mutate pages.raw_text).
- Venona: 1 message/page => 1 chunk by default; optional oversize fallback split.
- Vassiliev: marker-aware excerpt chunking across PDF pages:
    * markers are p.xx lines (e.g., "p.71", "p. 71")
    * markers act as boundaries, NOT content (excluded from chunks.text)
    * marker streams can continue across pages
    * consecutive markers with no content in between are ignored (no empty chunks)

Writes:
- chunks(text, pipeline_version) with embedding NULL
- chunk_pages(chunk_id, page_id, span_order)

Rerun semantics:
- deletes existing chunks for (document_id, pipeline_version) before rebuilding.

Usage examples:
  # Venona, 1 message/page = 1 chunk (fallback splitting only if very long)
  python3 scripts/chunk_corpus.py --collection venona --pipeline-version chunk_v2_venona_msg_fallback24k

  # Vassiliev marker-aware (split within a marker stream if chunk > 4000 chars)
  python3 scripts/chunk_corpus.py --collection vassiliev --pipeline-version chunk_v2_vass_marker_stream_4k --max-chars 4000

  # One document
  python3 scripts/chunk_corpus.py --document-id 15 --pipeline-version chunk_test --max-chars 4000
"""

import os
import re
import sys
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import psycopg2

# ---------- DB config ----------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")


def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


# ---------- Models ----------
@dataclass
class PageRow:
    id: int
    page_seq: int
    pdf_page_number: Optional[int]
    logical_page_label: str
    raw_text: str


@dataclass
class ChunkSpec:
    text: str          # raw-ish
    clean_text: str    # cleaned for retrieval/embedding
    page_ids_in_order: List[int]



# ---------- Text helpers ----------
PXX_LINE_RE = re.compile(r"(?m)^\s*(p\.\s*\d+)\s*$")


def is_marker_line(line: str) -> Optional[str]:
    """
    If line is a p.xx marker line, returns normalized marker like 'p.71'. Else None.
    """
    m = PXX_LINE_RE.match(line.strip())
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(1).strip())


def split_paragraphs(text: str) -> List[str]:
    """
    Split on 2+ newlines into "paragraphs". Returns non-empty, stripped paragraphs.
    """
    parts = re.split(r"\n{2,}", text)
    out: List[str] = []
    for p in parts:
        p2 = p.strip()
        if p2:
            out.append(p2)
    return out


def dedupe_preserve_order_int(xs: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def has_real_content(text: str, min_chars: int = 15) -> bool:
    """
    Heuristic: text has "real content" if it contains at least min_chars
    after stripping whitespace and removing empty lines.
    """
    t = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
    return len(t) >= min_chars


# ---------- DB operations ----------
def iter_documents(cur, collection_slug: Optional[str], document_id: Optional[int]) -> List[Tuple[int, str]]:
    """
    Returns list of (document_id, collection_slug) for processing.
    """
    if document_id is not None:
        cur.execute(
            """
            SELECT d.id, c.slug
            FROM documents d
            JOIN collections c ON c.id = d.collection_id
            WHERE d.id = %s
            """,
            (document_id,),
        )
        rows = cur.fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    if not collection_slug:
        raise ValueError("Must provide --collection or --document-id")

    cur.execute(
        """
        SELECT d.id, c.slug
        FROM documents d
        JOIN collections c ON c.id = d.collection_id
        WHERE c.slug = %s
        ORDER BY d.id
        """,
        (collection_slug,),
    )
    rows = cur.fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def load_pages_for_document(cur, document_id: int) -> List[PageRow]:
    cur.execute(
        """
        SELECT id, page_seq, pdf_page_number, logical_page_label, raw_text
        FROM pages
        WHERE document_id = %s
        ORDER BY page_seq
        """,
        (document_id,),
    )
    out: List[PageRow] = []
    for r in cur.fetchall():
        out.append(
            PageRow(
                id=int(r[0]),
                page_seq=int(r[1]),
                pdf_page_number=(int(r[2]) if r[2] is not None else None),
                logical_page_label=str(r[3]),
                raw_text=str(r[4]),
            )
        )
    return out


def delete_chunks_for_document(cur, document_id: int, pipeline_version: str):
    """
    Delete derived chunks for this document and this pipeline version.
    chunk_pages rows delete via ON DELETE CASCADE (through chunks).
    """
    cur.execute(
        """
        DELETE FROM chunks
        WHERE pipeline_version = %s
          AND id IN (
            SELECT cp.chunk_id
            FROM chunk_pages cp
            JOIN pages p ON p.id = cp.page_id
            WHERE p.document_id = %s
          )
        """,
        (pipeline_version, document_id),
    )


def insert_chunk(cur, text: str, clean_text: str, pipeline_version: str) -> int:
    cur.execute(
        """
        INSERT INTO chunks (text, clean_text, pipeline_version)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (text, clean_text, pipeline_version),
    )
    return int(cur.fetchone()[0])



def insert_chunk_pages(cur, chunk_id: int, page_ids_in_order: List[int]):
    for i, page_id in enumerate(page_ids_in_order, start=1):
        cur.execute(
            """
            INSERT INTO chunk_pages (chunk_id, page_id, span_order)
            VALUES (%s, %s, %s)
            """,
            (chunk_id, page_id, i),
        )


# ---------- Strategy routing ----------
def pick_strategy(collection_slug: str) -> str:
    if collection_slug == "venona":
        return "venona_message"
    if collection_slug == "vassiliev":
        return "vassiliev_marker_stream"
    if collection_slug == "silvermaster":
        return "silvermaster_structured"
    return "fallback_paragraph"



# ---------- Chunking strategies ----------
def chunk_venona_one_message_per_page(
    pages: List[PageRow],
    oversize_max_chars: int = 24000,
) -> List[ChunkSpec]:
    """
    Venona: pages are already message-level units.
    Default: 1 page/message => 1 chunk.
    Fallback: if message text is very large, split deterministically by paragraphs
              into multiple chunks, all mapped to the same page_id.
    """
    out: List[ChunkSpec] = []

    for p in pages:
        txt = (p.raw_text or "").strip()
        if not txt:
            continue

        if len(txt) <= oversize_max_chars:
            out.append(ChunkSpec(text=txt, clean_text=txt, page_ids_in_order=[p.id]))
            continue

        # Oversize fallback: split by paragraphs with a hard char cap
        paras = split_paragraphs(txt)
        cur_parts: List[str] = []
        for para in paras:
            if not cur_parts:
                cur_parts = [para]
                continue
            cand = "\n\n".join(cur_parts + [para])
            if len(cand) <= oversize_max_chars:
                cur_parts.append(para)
            else:
                chunk_text = "\n\n".join(cur_parts).strip()
                if chunk_text:
                    out.append(ChunkSpec(text=chunk_text, page_ids_in_order=[p.id]))
                cur_parts = [para]

        # flush last
        chunk_text = "\n\n".join(cur_parts).strip()
        if chunk_text:
            out.append(ChunkSpec(text=chunk_text, page_ids_in_order=[p.id]))

    return out


def iter_vassiliev_marker_segments(pages: List[PageRow]) -> List[Tuple[Optional[str], str, List[int]]]:
    """
    Convert ordered PDF pages into a sequence of marker-defined segments (streams),
    spanning across pages until the marker changes.

    Returns list of tuples:
      (marker_or_none, segment_text_without_marker_lines, page_ids_in_order)

    Rules:
    - marker lines (p.xx) are boundaries and are EXCLUDED from segment text.
    - if markers occur back-to-back without content in between, ignore the empty segment.
    - if text exists before the first marker on a page, it belongs to marker=None stream
      (unmarked). We keep it, but it will NOT be merged with marked streams.
    """
    segments: List[Tuple[Optional[str], str, List[int]]] = []

    current_marker: Optional[str] = None
    buf_lines: List[str] = []
    buf_pages: List[int] = []

    def flush_if_content():
        nonlocal buf_lines, buf_pages, current_marker, segments
        text = "\n".join(buf_lines).strip()
        if has_real_content(text):
            segments.append((current_marker, text, dedupe_preserve_order_int(buf_pages)))
        buf_lines = []
        buf_pages = []
        # keep current_marker as-is; caller decides marker changes

    for p in pages:
        page_lines = (p.raw_text or "").splitlines()

        # We track whether we attached any non-marker content on this page
        attached_any = False

        for ln in page_lines:
            mk = is_marker_line(ln)
            if mk is not None:
                # boundary encountered
                flush_if_content()
                # switch marker
                current_marker = mk
                attached_any = True  # marker encountered; page is part of marker context
                buf_pages.append(p.id)  # marker appears on this page, include for citation span
                continue

            # non-marker content line
            if ln.strip() == "":
                # preserve blank lines lightly (they help paragraph breaks), but avoid infinite empties
                buf_lines.append("")
                buf_pages.append(p.id)
                continue

            buf_lines.append(ln)
            buf_pages.append(p.id)
            attached_any = True

        # If a page had absolutely nothing (rare), do nothing.
        # If a page had only marker lines, we already appended page id in marker handling
        # but no content will flush (flush_if_content ignores empties) — desired.

    # final flush
    # (do not create empty segment if last thing was marker-only)
    text = "\n".join(buf_lines).strip()
    if has_real_content(text):
        segments.append((current_marker, text, dedupe_preserve_order_int(buf_pages)))

    return segments


def chunk_vassiliev_marker_stream(
    pages: List[PageRow],
    max_chars: int = 4000,
) -> List[ChunkSpec]:
    """
    Produce chunks from marker-defined segments:
    - segments are marker-stream text without marker lines
    - within a segment, split by paragraphs into <= max_chars chunks
    - never merge across segments (prevents disparate excerpt mixing)
    - chunks can span PDF pages via page_ids_in_order captured in the segment
    """
    out: List[ChunkSpec] = []
    segments = iter_vassiliev_marker_segments(pages)

    for _marker, seg_text, seg_page_ids in segments:
        # Split segment text into paragraphs, then pack paragraphs into <= max_chars
        paras = split_paragraphs(seg_text)

        if not paras:
            continue

        cur_parts: List[str] = []
        for para in paras:
            if not cur_parts:
                cur_parts = [para]
                continue
            cand = "\n\n".join(cur_parts + [para])
            if len(cand) <= max_chars:
                cur_parts.append(para)
            else:
                chunk_text = "\n\n".join(cur_parts).strip()
                if chunk_text:
                        out.append(ChunkSpec(text=chunk_text, clean_text=chunk_text, page_ids_in_order=seg_page_ids))
                cur_parts = [para]

        # flush last
        chunk_text = "\n\n".join(cur_parts).strip()
        if chunk_text:
            out.append(ChunkSpec(text=chunk_text, clean_text=chunk_text, page_ids_in_order=seg_page_ids))

    return out

# ---------- Silvermaster helpers ----------
FOIPA_SHEET_RE = re.compile(
    r"(?is)FOIPA\s+DELETED\s+PAGE\s+INFORMATION\s+SHEET|Page\(s\)\s+withheld|Deleted\s+Page",
)

# Strong “document unit” anchors common in FBI files
UNIT_ANCHOR_RE = re.compile(
    r"(?im)^\s*(FEDERAL\s+BUREAU\s+OF\s+INVESTIGATION|Office\s+Memorandum|UNITED\s+STATES\s+GOVERNMENT|AIR(TEL|MAIL)|MEMORANDUM)\b"
)

# Date-ish header lines often near top of memos (very loose on purpose)
DATE_LINE_RE = re.compile(
    r"(?im)^\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\s*$"
)

# Some very common boilerplate lines worth stripping when they occur as standalone lines
BOILERPLATE_LINE_RE = re.compile(
    r"(?im)^\s*(FEDERAL\s+BUREAU\s+OF\s+INVESTIGATION|UNITED\s+STATES\s+GOVERNMENT|Office\s+Memorandum|FOIPA.*SHEET)\s*$"
)

# Subject line often near top of memos
SUBJECT_RE = re.compile(r"(?im)^\s*(Re:|SUBJECT:|SUBJ:)\s+.+$")


def page_quality_score(text: str) -> float:
    """
    Returns ~0..1 where 1 is clean English-ish text.
    Cheap heuristic: alphabetic ratio + wordlike ratio.
    """
    s = (text or "").strip()
    if not s:
        return 0.0

    letters = sum(ch.isalpha() for ch in s)
    alnum = sum(ch.isalnum() for ch in s)
    wordlike = len(re.findall(r"[A-Za-z]{3,}", s))

    letter_ratio = letters / max(len(s), 1)
    alnum_ratio = alnum / max(len(s), 1)
    word_density = wordlike / max(len(s) / 100.0, 1.0)  # words per ~100 chars

    # cap density influence
    density_term = min(1.0, word_density / 3.0)
    return max(0.0, min(1.0, 0.6 * letter_ratio + 0.3 * alnum_ratio + 0.1 * density_term))


def is_ocr_garbage_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    # Long runs of the same char / symbol salad
    if re.search(r"(.)\1\1\1\1\1", s):  # 6 repeats
        return True

    # Too many non-alphanumerics
    alnum = sum(ch.isalnum() for ch in s)
    if len(s) >= 30 and alnum / max(len(s), 1) < 0.35:
        return True

    # Looks like OCR noise: lots of stray single chars separated by spaces
    if len(s) >= 25 and re.fullmatch(r"(?:[A-Za-z0-9]\s+){10,}[A-Za-z0-9]?", s):
        return True

    return False

def page_starts_with_boundary(lines: List[str], head_n: int = 12) -> bool:
    """
    Boundary cues should be detected near the top of the page to avoid
    false boundaries from repeated headers/stamps mid-page.
    """
    head = [ln for ln in lines[:head_n] if ln.strip()]
    if not head:
        return False

    # Check first few non-empty lines for strong unit anchors
    for ln in head[:6]:
        if UNIT_ANCHOR_RE.match(ln):
            return True
        if SUBJECT_RE.match(ln):
            return True
        if DATE_LINE_RE.match(ln):
            return True

    return False


def clean_silvermaster_text(raw: str) -> str:
    """
    Light cleaning:
    - drop obvious boilerplate-only lines
    - drop OCR garbage lines
    - normalize hyphenation at line breaks: 'inter-\nnational' -> 'international'
    - keep everything else (we want provenance + searchability)
    """
    raw = raw.replace("\u00a0", " ")
    # dehyphenate linebreaks
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)

    lines = raw.splitlines()
    kept = []
    for ln in lines:
        if BOILERPLATE_LINE_RE.match(ln):
            continue
        if is_ocr_garbage_line(ln):
            continue
        kept.append(ln.rstrip())

    # collapse huge whitespace but keep paragraph structure
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_units(pages: List[PageRow]) -> List[Tuple[str, List[int]]]:
    """
    Produce a list of (unit_text, page_ids_in_order) where each unit is a coherent block.

    Silvermaster rules:
      - FOIPA deleted-page sheets => standalone unit
      - OCR-garbage / very low-quality pages => standalone unit (do not contaminate neighbors)
      - New unit starts only on *top-of-page* cues (anchors / subject / date) when buffer already has content
      - Do NOT split on mid-stream headers (too many false positives in FBI files)
    """
    units: List[Tuple[str, List[int]]] = []
    buf_lines: List[str] = []
    buf_pages: List[int] = []

    def flush(min_chars: int = 50):
        nonlocal buf_lines, buf_pages
        text = "\n".join(buf_lines).strip()
        if has_real_content(text, min_chars=min_chars):
            units.append((text, dedupe_preserve_order_int(buf_pages)))
        buf_lines = []
        buf_pages = []

    for p in pages:
        raw = p.raw_text or ""
        if not raw.strip():
            continue

        # FOIPA deleted page sheets: keep as standalone unit
        if FOIPA_SHEET_RE.search(raw):
            flush(min_chars=80)
            cleaned = clean_silvermaster_text(raw)
            if cleaned:
                units.append((cleaned, [p.id]))
            continue

        cleaned = clean_silvermaster_text(raw)
        if not cleaned:
            continue

        # Page-level OCR quality gating
        q = page_quality_score(cleaned)

        # Very low quality: isolate as standalone (keeps citations, prevents contamination)
        if q < 0.25:
            flush(min_chars=80)
            units.append((cleaned, [p.id]))
            continue

        lines = cleaned.splitlines()

        # Start a new unit only if boundary cues appear near the top of the page,
        # and we already have meaningful buffered content.
        if page_starts_with_boundary(lines, head_n=12) and has_real_content("\n".join(buf_lines), min_chars=200):
            flush(min_chars=80)

        # Append page content (no mid-stream boundary splitting)
        for ln in lines:
            if is_ocr_garbage_line(ln):
                continue
            buf_lines.append(ln)
            buf_pages.append(p.id)

        # Preserve page boundary
        buf_lines.append("")
        buf_pages.append(p.id)

    flush(min_chars=80)
    return units


def pack_units_into_chunks(units: List[Tuple[str, List[int]]], max_chars: int) -> List[ChunkSpec]:
    """
    Pack paragraphs inside each unit into <= max_chars chunks.
    Never merge across units (keeps memo boundaries meaningful).
    """
    out: List[ChunkSpec] = []
    for unit_text, page_ids in units:
        paras = split_paragraphs(unit_text)
        if not paras:
            continue

        cur_parts: List[str] = []
        for para in paras:
            if not cur_parts:
                cur_parts = [para]
                continue
            cand = "\n\n".join(cur_parts + [para])
            if len(cand) <= max_chars:
                cur_parts.append(para)
            else:
                chunk_text = "\n\n".join(cur_parts).strip()
                if chunk_text:
                    out.append(ChunkSpec(text=chunk_text, clean_text=chunk_text, page_ids_in_order=page_ids))
                cur_parts = [para]

        chunk_text = "\n\n".join(cur_parts).strip()
        if chunk_text:
            out.append(ChunkSpec(text=chunk_text, clean_text=chunk_text, page_ids_in_order=page_ids))
    return out


def chunk_silvermaster_structured(pages: List[PageRow], max_chars: int = 4000) -> List[ChunkSpec]:
    units = split_into_units(pages)
    return pack_units_into_chunks(units, max_chars=max_chars)


def chunk_fallback_paragraph(pages: List[PageRow], max_chars: int) -> List[ChunkSpec]:
    """
    Generic chunker for future sources:
    - split by paragraphs
    - allow spanning pages
    """
    out: List[ChunkSpec] = []
    cur_parts: List[str] = []
    cur_pages: List[int] = []

    def flush():
        nonlocal cur_parts, cur_pages
        text = "\n\n".join(cur_parts).strip()
        if text:
            out.append(ChunkSpec(text=text, page_ids_in_order=dedupe_preserve_order_int(cur_pages)))
        cur_parts = []
        cur_pages = []

    for p in pages:
        paras = split_paragraphs(p.raw_text or "")
        for para in paras:
            if not cur_parts:
                cur_parts = [para]
                cur_pages = [p.id]
                continue
            cand = "\n\n".join(cur_parts + [para])
            if len(cand) <= max_chars:
                cur_parts.append(para)
                cur_pages.append(p.id)
            else:
                flush()
                cur_parts = [para]
                cur_pages = [p.id]

    flush()
    return out


# ---------- Runner ----------
def main():
    ap = argparse.ArgumentParser(description="Chunk corpus documents into chunks + chunk_pages (rerunnable).")
    ap.add_argument("--collection", default=None, help="collection slug (e.g., vassiliev, venona, silvermaster...)")
    ap.add_argument("--document-id", type=int, default=None, help="chunk only one document_id")
    ap.add_argument("--pipeline-version", required=True, help="chunks.pipeline_version to write")
    ap.add_argument("--max-chars", type=int, default=4000, help="soft max chars per chunk (used for vassiliev and fallback)")
    ap.add_argument("--venona-max-chars", type=int, default=24000, help="oversize fallback cap for venona message chunks")
    ap.add_argument("--limit-docs", type=int, default=None, help="limit number of documents processed (debug)")
    args = ap.parse_args()

    if args.collection is None and args.document_id is None:
        print("ERROR: provide --collection or --document-id", file=sys.stderr)
        sys.exit(2)

    pipeline_version = args.pipeline_version
    max_chars = int(args.max_chars)
    venona_max_chars = int(args.venona_max_chars)

    with connect() as conn, conn.cursor() as cur:
        docs = iter_documents(cur, args.collection, args.document_id)
        if args.limit_docs is not None:
            docs = docs[: int(args.limit_docs)]

        if not docs:
            print("No matching documents found.")
            return

        total_docs = 0
        total_chunks = 0

        for (document_id, collection_slug) in docs:
            pages = load_pages_for_document(cur, document_id)
            if not pages:
                print(f"[skip] document_id={document_id} has 0 pages")
                continue

            strategy = pick_strategy(collection_slug)

            # Rerun safety: delete existing derived chunks for this doc + pipeline
            delete_chunks_for_document(cur, document_id, pipeline_version)

            if strategy == "venona_message":
                chunk_specs = chunk_venona_one_message_per_page(pages, oversize_max_chars=venona_max_chars)
            elif strategy == "vassiliev_marker_stream":
                chunk_specs = chunk_vassiliev_marker_stream(pages, max_chars=max_chars)
            elif strategy == "silvermaster_structured":
                chunk_specs = chunk_silvermaster_structured(pages, max_chars=max_chars)
            else:
                chunk_specs = chunk_fallback_paragraph(pages, max_chars=max_chars)

            inserted = 0
            for spec in chunk_specs:
                if not spec.text.strip():
                    continue
                chunk_id = insert_chunk(cur, spec.text, spec.clean_text, pipeline_version)
                insert_chunk_pages(cur, chunk_id, spec.page_ids_in_order)
                inserted += 1

            conn.commit()

            total_docs += 1
            total_chunks += inserted
            print(
                f"[ok] document_id={document_id} collection={collection_slug} "
                f"pages={len(pages)} chunks={inserted} pipeline={pipeline_version} strategy={strategy}"
            )

        print(f"Done. documents={total_docs} total_chunks={total_chunks} pipeline_version={pipeline_version}")


if __name__ == "__main__":
    main()
