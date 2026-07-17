#!/usr/bin/env python3
"""
SISS "Scope of Soviet Activity in the United States" Ingest - Turn-Aware + Gap Chunking

Ingests the Senate Internal Security Subcommittee (SISS) hearings
"Scope of Soviet Activity in the United States" (84th Congress, 1956- ).

These are large, multi-PART combined volumes (one PDF holds many parts, each
with its own title page + committee roster + one or more hearing days). The
transcripts use the standard speaker-turn format:

    Senator WELKER. ...
    Mr. MORRIS. ...
    Mr. RASTVOROV. ...

Differences from ingest_huac_hearings.py that this script handles:

1. PER-PAGE FRONT-MATTER SKIP. Each PART repeats Google-scan boilerplate, a
   title page ("Printed for the use of ..."), and a committee roster. These are
   detected per-page by markers (NOT by opener-region segmentation, because OCR
   drops some "subcommittee met" openers and region segmentation would then
   discard a whole part). Front-matter pages are still inserted as `pages`
   (so the PDF viewer resolves every page) but are excluded from chunking.

2. HYBRID CHUNKING. Testimony is chunked turn-aware (never splitting a turn).
   Substantial stretches of NON-turn text on content pages (exhibits, documents
   read into the record, appendix/report material, and the whole 7A appendix
   volume which has almost no live testimony) are captured with fixed-size
   "gap" chunks so they remain searchable. Without this, the purely turn-based
   HUAC pipeline would silently drop all exhibit/appendix text.

3. SISS-specific role map (Senate subcommittee, not House HUAC).

Known limitation: testimony quoted/read INTO the record (e.g. counsel reading a
prior Lattimore transcript) produces speaker turns for people not actually
present. Those turns are still attributed verbatim; disambiguating "live vs
read-in" speakers is out of scope here.

Collection slug: siss_scope_soviet

NOTE: like the HUAC script, this does NOT compute vector embeddings; it stores
`embed_text` for a separate embedding pass.
"""
import os
import sys
import io
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import Counter

import psycopg2
import fitz  # PyMuPDF

import ingest_runs

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# =============================================================================
# Configuration
# =============================================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")

COLLECTION_SLUG = "siss_scope_soviet"
COLLECTION_TITLE = "Scope of Soviet Activity in the United States (SISS, 1956- )"
COLLECTION_DESCRIPTION = """Senate Internal Security Subcommittee (Subcommittee to
Investigate the Administration of the Internal Security Act and Other Internal
Security Laws, Committee on the Judiciary) hearings, "Scope of Soviet Activity in
the United States," 84th Congress and following. Multi-part transcript volumes
with speaker-aware, turn-level attribution plus exhibit/appendix capture."""


# =============================================================================
# Speaker Detection
# =============================================================================

# Main speaker pattern: Mr. MORRIS., The CHAIRMAN., Senator WELKER.
# The surname must start with 2+ capitals so exhibit prose ("The Communist
# Party...", "The Basic...") is not mistaken for a speaker label, while OCR
# mixed-case surnames ("RASTVorov") are still accepted.
SPEAKER_PATTERN = re.compile(
    r"^((?:Mr|Mrs|Ms|Miss|Dr|Senator|Representative|The)\s*[.\s]*\s*[A-Z]{2}[A-Za-z]*)\s*[.\s]+\s*(?=[A-Z])",
    re.MULTILINE
)

# Stage directions: [Laughter], [Recess], [Discussion off the record]
STAGE_DIRECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")

# SISS-specific role mapping (Senate subcommittee).
ROLE_MAP = {
    # Subcommittee chairman / presiding senators
    "SENATOR EASTLAND": "chair",
    "THE CHAIRMAN": "chair",
    "SENATOR WELKER": "senator",
    "SENATOR JENNER": "senator",
    "SENATOR MCCLELLAN": "senator",
    "SENATOR BUTLER": "senator",
    "SENATOR JOHNSTON": "senator",
    "SENATOR HENNINGS": "senator",
    "SENATOR DANIEL": "senator",
    "SENATOR WATKINS": "senator",
    "SENATOR DIRKSEN": "senator",
    "SENATOR HRUSKA": "senator",
    "SENATOR KEATING": "senator",
    "SENATOR DODD": "senator",
    "SENATOR SMITH": "senator",

    # Counsel / staff
    "MR MORRIS": "counsel",       # Robert Morris, chief counsel
    "MR ARENS": "counsel",        # Richard Arens
    "MR CARPENTER": "counsel",    # Alva C. Carpenter
    "MR SOURWINE": "counsel",     # J.G. Sourwine
    "MR MANDEL": "research",      # Benjamin Mandel, research director
    "MR SCHROEDER": "counsel",
}


def normalize_speaker(speaker_raw: str) -> str:
    """Normalize speaker to canonical form: uppercase, no punctuation, single spaces."""
    norm = speaker_raw.strip().rstrip(".")
    norm = norm.upper()
    norm = re.sub(r"\s+", " ", norm)
    norm = re.sub(r"[^\w\s]", "", norm)
    return norm.strip()


def detect_role(speaker_norm: str) -> Optional[str]:
    """Detect role from normalized speaker name."""
    if speaker_norm in ROLE_MAP:
        return ROLE_MAP[speaker_norm]
    if speaker_norm.startswith("SENATOR"):
        return "senator"
    if speaker_norm.startswith("THE CHAIRMAN"):
        return "chair"
    if speaker_norm.startswith("MR") or speaker_norm.startswith("MRS") or speaker_norm.startswith("MISS"):
        return "witness"
    return None


# =============================================================================
# Front-matter detection (per page)
# =============================================================================

# Markers that identify Google-scan boilerplate, title pages, rosters, contents.
FRONT_MATTER_MARKERS = re.compile(
    r"Printed for the use of"
    r"|GOVERNMENT PRINTING OFFICE"
    r"|digitized\s+by\s+Google"
    r"|reproduction of a library book"
    r"|COMMITTEE ON THE JUDICIARY"
    r"|^\s*CONTENTS\s*$"
    r"|SUBCOMMITTEE TO INVESTIGATE",
    re.IGNORECASE | re.MULTILINE,
)


def is_front_matter_page(text: str) -> bool:
    """
    A page is front matter if it carries a title/roster/boilerplate marker AND
    has essentially no live testimony (fewer than 3 speaker turns). Pages with
    real dialogue are always treated as content, even if they mention a marker.
    """
    turn_count = len(SPEAKER_PATTERN.findall(text))
    if turn_count >= 3:
        return False
    return bool(FRONT_MATTER_MARKERS.search(text))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Turn:
    """A single speaker turn from the transcript."""
    turn_id: int
    speaker_raw: str
    speaker_norm: str
    speaker_role: Optional[str]
    turn_text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    is_stage_direction: bool = False


@dataclass
class ChunkingConfig:
    """Configuration for turn-aware + gap chunking."""
    target_chars: int = 4000
    max_chars: int = 6000
    overlap_turns: int = 2
    min_chunk_chars: int = 200
    max_turns_per_chunk: int = 25
    min_gap_chars: int = 400   # below this, non-turn text is ignored as noise


@dataclass
class Chunk:
    """A chunk (turn-aware or gap). Turn fields are None for gap chunks."""
    text: str
    embed_text: str
    page_start: int
    page_end: int
    chunk_index: int
    content_type: str
    turn_id_start: Optional[int] = None
    turn_id_end: Optional[int] = None
    turn_count: int = 0
    speaker_norms: List[str] = field(default_factory=list)
    primary_speaker_norm: Optional[str] = None
    turns: List[Turn] = field(default_factory=list)
    speaker_turn_spans: List[Dict] = field(default_factory=list)


# =============================================================================
# Content concatenation + turn parsing
# =============================================================================

def build_content_text(content_pages: List[Tuple[int, str]]) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Concatenate content-page text. Returns (all_text, page_char_ranges) where
    page_char_ranges is a list of (pdf_page, char_start, char_end).
    """
    all_text = ""
    ranges: List[Tuple[int, int, int]] = []
    for pdf_page, text in content_pages:
        start = len(all_text)
        all_text += text + "\n\n"
        end = len(all_text)
        ranges.append((pdf_page, start, end))
    return all_text, ranges


def page_for_char(char_pos: int, ranges: List[Tuple[int, int, int]]) -> int:
    for pdf_page, start, end in ranges:
        if start <= char_pos < end:
            return pdf_page
    return ranges[-1][0] if ranges else 1


def page_range_for_chars(char_start: int, char_end: int, ranges: List[Tuple[int, int, int]]) -> Tuple[int, int]:
    return page_for_char(char_start, ranges), page_for_char(max(char_start, char_end - 1), ranges)


def parse_turns(all_text: str, ranges: List[Tuple[int, int, int]]) -> List[Turn]:
    """Parse concatenated content text into speaker turns."""
    turns: List[Turn] = []
    turn_id = 0

    boundaries = []
    for m in SPEAKER_PATTERN.finditer(all_text):
        boundaries.append((m.start(), "speaker", m))
    for m in STAGE_DIRECTION.finditer(all_text):
        boundaries.append((m.start(), "stage", m))
    boundaries.sort(key=lambda x: x[0])

    for i, (pos, btype, match) in enumerate(boundaries):
        turn_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(all_text)

        if btype == "speaker":
            speaker_raw = match.group(1)
            speaker_norm = normalize_speaker(speaker_raw)
            speaker_role = detect_role(speaker_norm)
            turn_text = all_text[match.end():turn_end].strip()
            if turn_text:
                turn_id += 1
                ps, pe = page_range_for_chars(match.start(), turn_end, ranges)
                turns.append(Turn(
                    turn_id=turn_id, speaker_raw=speaker_raw, speaker_norm=speaker_norm,
                    speaker_role=speaker_role, turn_text=turn_text,
                    page_start=ps, page_end=pe,
                    char_start=match.start(), char_end=turn_end, is_stage_direction=False,
                ))
        else:  # stage
            stage_text = match.group(1)
            turn_id += 1
            ps, pe = page_range_for_chars(match.start(), match.end(), ranges)
            turns.append(Turn(
                turn_id=turn_id, speaker_raw=f"[{stage_text}]", speaker_norm="__STAGE__",
                speaker_role="stage", turn_text=stage_text,
                page_start=ps, page_end=pe,
                char_start=match.start(), char_end=match.end(), is_stage_direction=True,
            ))

    return turns


# =============================================================================
# Chunking
# =============================================================================

def split_long_turns(turns: List[Turn], all_text: str, ranges: List[Tuple[int, int, int]],
                     config: ChunkingConfig) -> List[Turn]:
    """
    Split any turn whose text exceeds target_chars into sequential same-speaker
    pieces at paragraph boundaries. Prevents a single huge turn (common when a
    speaker reads a multi-page exhibit into the record) from becoming one
    oversized chunk that insert_chunk would truncate and lose. Page ranges for
    each piece are mapped precisely via absolute char offsets in all_text.
    """
    out: List[Turn] = []
    next_id = 1
    for t in turns:
        if t.is_stage_direction or len(t.turn_text) <= config.target_chars:
            t.turn_id = next_id
            next_id += 1
            out.append(t)
            continue

        # Locate the turn_text within all_text for accurate page mapping.
        base = all_text.find(t.turn_text[:60], t.char_start, t.char_end + 10)
        if base < 0:
            base = t.char_start

        pieces = re.split(r"(\n\s*\n)", t.turn_text)  # keep separators to preserve offsets
        buf = ""
        buf_local_start = 0
        local_pos = 0

        def flush_piece(local_start: int, text: str):
            nonlocal next_id
            if not text.strip():
                return
            abs_s = base + local_start
            abs_e = abs_s + len(text)
            ps, pe = page_range_for_chars(abs_s, abs_e, ranges)
            out.append(Turn(
                turn_id=next_id, speaker_raw=t.speaker_raw, speaker_norm=t.speaker_norm,
                speaker_role=t.speaker_role, turn_text=text.strip(),
                page_start=ps, page_end=pe, char_start=abs_s, char_end=abs_e,
                is_stage_direction=False,
            ))
            next_id += 1

        for piece in pieces:
            if buf and len(buf) + len(piece) > config.target_chars:
                flush_piece(buf_local_start, buf)
                buf = ""
                buf_local_start = local_pos
            buf += piece
            local_pos += len(piece)
        if buf:
            flush_piece(buf_local_start, buf)

    return out


def compute_speaker_turn_spans(turns: List[Turn]) -> List[Dict]:
    """Compute contiguous speaker spans within a chunk's turns."""
    if not turns:
        return []
    spans = []
    cur_sp = cur_start = cur_end = None
    for turn in turns:
        sp = turn.speaker_norm
        if sp == cur_sp:
            cur_end = turn.turn_id
        else:
            if cur_sp is not None:
                spans.append({"speaker": cur_sp, "turn_id_start": cur_start, "turn_id_end": cur_end})
            cur_sp, cur_start, cur_end = sp, turn.turn_id, turn.turn_id
    if cur_sp is not None:
        spans.append({"speaker": cur_sp, "turn_id_start": cur_start, "turn_id_end": cur_end})
    return spans


def generate_embed_text(turns: List[Turn], doc_ref: str = "") -> str:
    """Generate speaker-tagged text for embeddings."""
    lines = []
    if turns:
        header = f"[Turns {turns[0].turn_id}-{turns[-1].turn_id} | Pages {turns[0].page_start}-{turns[-1].page_end}]"
        if doc_ref:
            header = f"[{doc_ref} | " + header[1:]
        lines.append(header)
    for turn in turns:
        if turn.is_stage_direction:
            lines.append(f"[{turn.turn_text}]")
        else:
            lines.append(f"SPEAKER: {turn.speaker_norm}")
            lines.append(f"TEXT: {turn.turn_text}")
        lines.append("")
    return "\n".join(lines)


def create_turn_chunks(turns: List[Turn], config: ChunkingConfig, doc_ref: str,
                       index_offset: int) -> List[Chunk]:
    """Create turn-aware chunks, never splitting mid-turn."""
    if not turns:
        return []
    chunks: List[Chunk] = []
    current: List[Turn] = []
    current_chars = 0

    def flush():
        nonlocal current, current_chars
        if not current:
            return
        parts = []
        for t in current:
            if t.is_stage_direction:
                parts.append(f"[{t.turn_text}]")
            else:
                parts.append(f"{t.speaker_raw}. {t.turn_text}")
        text = "\n\n".join(parts)
        if len(text) < config.min_chunk_chars:
            current = []
            current_chars = 0
            return
        counts: Counter = Counter()
        for t in current:
            if not t.is_stage_direction:
                counts[t.speaker_norm] += 1
        chunks.append(Chunk(
            text=text,
            embed_text=generate_embed_text(current, doc_ref),
            page_start=min(t.page_start for t in current),
            page_end=max(t.page_end for t in current),
            chunk_index=index_offset + len(chunks),
            content_type="siss_testimony",
            turn_id_start=current[0].turn_id,
            turn_id_end=current[-1].turn_id,
            turn_count=len(current),
            speaker_norms=list(counts.keys()),
            primary_speaker_norm=counts.most_common(1)[0][0] if counts else None,
            turns=current.copy(),
            speaker_turn_spans=compute_speaker_turn_spans(current),
        ))
        current = []
        current_chars = 0

    for turn in turns:
        turn_len = len(turn.turn_text)
        should_flush = (current_chars + turn_len > config.target_chars and current) or \
                       (len(current) >= config.max_turns_per_chunk)
        if should_flush:
            flush()
            if chunks and config.overlap_turns > 0:
                overlap = chunks[-1].turns[-config.overlap_turns:]
                current = list(overlap)
                current_chars = sum(len(t.turn_text) for t in current)
        current.append(turn)
        current_chars += turn_len

    flush()
    return chunks


def create_gap_chunks(all_text: str, turns: List[Turn], ranges: List[Tuple[int, int, int]],
                      config: ChunkingConfig, index_offset: int) -> List[Chunk]:
    """
    Capture substantial stretches of content text NOT covered by any turn
    (exhibits, read-in documents, appendix/report material). Fixed-size split
    on paragraph boundaries. No speaker metadata.
    """
    # Build covered intervals from turns, merge them.
    covered = sorted((t.char_start, t.char_end) for t in turns)
    merged: List[Tuple[int, int]] = []
    for s, e in covered:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Gaps are the complement within [0, len(all_text)).
    gaps: List[Tuple[int, int]] = []
    prev_end = 0
    for s, e in merged:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < len(all_text):
        gaps.append((prev_end, len(all_text)))

    chunks: List[Chunk] = []
    for gap_start, gap_end in gaps:
        segment = all_text[gap_start:gap_end]
        if len(segment.strip()) < config.min_gap_chars:
            continue
        # Split into ~target_chars pieces on paragraph boundaries, tracking
        # absolute offsets so pages map correctly.
        paras = re.split(r"(\n\s*\n)", segment)  # keep separators to preserve offsets
        buf = ""
        buf_local_start = 0
        local_pos = 0

        def emit(local_start: int, local_end: int, text: str):
            if len(text.strip()) < config.min_gap_chars:
                return
            abs_s = gap_start + local_start
            abs_e = gap_start + local_end
            ps, pe = page_range_for_chars(abs_s, abs_e, ranges)
            clean = text.strip()
            chunks.append(Chunk(
                text=clean,
                embed_text=clean,
                page_start=ps,
                page_end=pe,
                chunk_index=index_offset + len(chunks),
                content_type="siss_exhibit",
            ))

        for piece in paras:
            if buf and len(buf) + len(piece) > config.target_chars:
                emit(buf_local_start, local_pos, buf)
                buf = ""
                buf_local_start = local_pos
            buf += piece
            local_pos += len(piece)
        if buf:
            emit(buf_local_start, local_pos, buf)

    return chunks


# =============================================================================
# Text Normalization
# =============================================================================

# Running header tokens (the repeated page header on transcript pages).
_HEADER_TOKENS = {"SCOPE", "OF", "SOVIET", "ACTIVITY", "IN", "THE", "UNITED", "STATES"}


def normalize_text(raw_text: str) -> str:
    """Clean OCR artifacts and strip running headers / standalone page numbers."""
    text = raw_text.replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        # standalone page numbers
        if re.match(r"^\d+$", stripped):
            continue
        # standalone roman numerals
        if re.match(r"^\(?[IVXLCDM]+\)?$", stripped, re.IGNORECASE):
            continue
        # running-header fragments: a line composed only of header tokens
        alpha = re.sub(r"[^A-Za-z ]", "", stripped).strip().upper()
        if alpha:
            toks = alpha.split()
            if toks and all(t in _HEADER_TOKENS for t in toks):
                continue
        out.append(line)
    return "\n".join(out).strip()


# =============================================================================
# Database Operations
# =============================================================================

def connect():
    # Prefer DATABASE_URL (e.g. prod via `source ./friday_env.sh`); fall back to DB_* vars.
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)


def get_or_create_collection(cur) -> int:
    cur.execute("SELECT id FROM collections WHERE slug = %s", (COLLECTION_SLUG,))
    r = cur.fetchone()
    if r:
        return int(r[0])
    cur.execute(
        "INSERT INTO collections (slug, title, description) VALUES (%s, %s, %s) RETURNING id",
        (COLLECTION_SLUG, COLLECTION_TITLE, COLLECTION_DESCRIPTION),
    )
    return int(cur.fetchone()[0])


def upsert_document(cur, collection_id: int, source_name: str, source_ref: str,
                    year: str, metadata: dict) -> int:
    cur.execute(
        "SELECT id FROM documents WHERE collection_id = %s AND source_name = %s",
        (collection_id, source_name),
    )
    r = cur.fetchone()
    if r:
        doc_id = int(r[0])
        cur.execute(
            "UPDATE documents SET source_ref = %s, metadata = %s WHERE id = %s",
            (source_ref, json.dumps(metadata), doc_id),
        )
        return doc_id
    cur.execute(
        """
        INSERT INTO documents (collection_id, source_name, source_ref, volume, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        RETURNING id
        """,
        (collection_id, source_name, source_ref, year, json.dumps(metadata)),
    )
    return int(cur.fetchone()[0])


def delete_document_data(cur, document_id: int):
    cur.execute("""
        DELETE FROM chunks WHERE id IN (
            SELECT cp.chunk_id FROM chunk_pages cp
            JOIN pages p ON p.id = cp.page_id
            WHERE p.document_id = %s
        )
    """, (document_id,))
    cur.execute("DELETE FROM pages WHERE document_id = %s", (document_id,))


def insert_page(cur, document_id: int, page_seq: int, pdf_page_number: int,
                logical_label: str, content_role: str, raw_text: str) -> int:
    cur.execute(
        """
        INSERT INTO pages (document_id, logical_page_label, pdf_page_number, page_seq,
                          language, content_role, raw_text)
        VALUES (%s, %s, %s, %s, 'en', %s, %s)
        RETURNING id
        """,
        (document_id, logical_label, pdf_page_number, page_seq, content_role, raw_text),
    )
    return int(cur.fetchone()[0])


def safe_truncate_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore')


def insert_chunk(cur, chunk: Chunk, pipeline_version: str) -> int:
    """Insert a chunk with retry logic for B-tree index size limits."""
    speaker_norms = chunk.speaker_norms[:20] if chunk.speaker_norms else []
    spans_json = json.dumps(chunk.speaker_turn_spans) if chunk.speaker_turn_spans else None

    max_sizes = [None, 6000, 4000, 2500]
    for i, max_size in enumerate(max_sizes):
        text_to_insert = chunk.text if max_size is None else safe_truncate_bytes(chunk.text, max_size)
        embed_text = chunk.embed_text if max_size is None else (
            safe_truncate_bytes(chunk.embed_text, max_size) if chunk.embed_text else None)
        savepoint = f"insert_retry_{i}"
        try:
            cur.execute(f"SAVEPOINT {savepoint}")
            cur.execute(
                """
                INSERT INTO chunks (text, clean_text, pipeline_version,
                                   turn_id_start, turn_id_end, turn_count,
                                   speaker_norms, primary_speaker_norm, embed_text,
                                   speaker_turn_spans)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (text_to_insert, text_to_insert, pipeline_version,
                 chunk.turn_id_start, chunk.turn_id_end, chunk.turn_count,
                 speaker_norms, chunk.primary_speaker_norm, embed_text, spans_json),
            )
            result = int(cur.fetchone()[0])
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            return result
        except Exception as e:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            msg = str(e)
            if "index row requires" in msg and "maximum size" in msg:
                if max_size == max_sizes[-1]:
                    raise
                continue
            raise
    raise Exception("Failed to insert chunk after all truncation attempts")


def insert_chunk_pages(cur, chunk_id: int, page_ids: List[int]):
    for i, page_id in enumerate(page_ids, start=1):
        cur.execute(
            "INSERT INTO chunk_pages (chunk_id, page_id, span_order) VALUES (%s, %s, %s)",
            (chunk_id, page_id, i),
        )


def insert_chunk_metadata(cur, chunk_id: int, document_id: int, pipeline_version: str,
                          first_page_id: Optional[int], last_page_id: Optional[int],
                          content_type: str):
    cur.execute("SELECT 1 FROM chunk_metadata WHERE chunk_id = %s", (chunk_id,))
    if cur.fetchone():
        return
    cur.execute(
        """
        INSERT INTO chunk_metadata (chunk_id, document_id, collection_slug, pipeline_version,
                                   first_page_id, last_page_id, content_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (chunk_id, document_id, COLLECTION_SLUG, pipeline_version,
         first_page_id, last_page_id, content_type),
    )


# =============================================================================
# Metadata helpers
# =============================================================================

def extract_year(filename: str, content: str) -> Optional[str]:
    for src in (filename, content[:8000]):
        m = re.search(r'(19[45]\d|196\d)', src)
        if m:
            return m.group(1)
    return None


# =============================================================================
# PDF Processing
# =============================================================================

def process_pdf(pdf_path: Path, cur, collection_id: int, config: ChunkingConfig,
                pipeline_version: str, dry_run: bool) -> Tuple[int, int, int, int, int]:
    """
    Returns (doc_id, page_count, turn_count, turn_chunks, gap_chunks).
    """
    source_name = pdf_path.name
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    print(f"  Processing {source_name}: {page_count} pages")

    # Extract + normalize every page; classify front-matter vs content.
    pages_data: List[Tuple[int, str, bool]] = []  # (pdf_page, text, is_front_matter)
    content_pages: List[Tuple[int, str]] = []
    front_matter_count = 0
    for i in range(page_count):
        raw = doc.load_page(i).get_text("text") or ""
        txt = normalize_text(raw)
        fm = is_front_matter_page(raw)  # classify on raw (markers intact)
        pages_data.append((i + 1, txt, fm))
        if fm:
            front_matter_count += 1
        else:
            content_pages.append((i + 1, txt))
    doc.close()

    # Parse turns + gaps over content pages only.
    all_text, ranges = build_content_text(content_pages)
    turns = parse_turns(all_text, ranges)
    turns = split_long_turns(turns, all_text, ranges, config)

    year = extract_year(source_name, all_text[:8000]) or "1956"
    doc_ref = f"siss_scope_{year}"

    turn_chunks = create_turn_chunks(turns, config, doc_ref, index_offset=0)
    gap_chunks = create_gap_chunks(all_text, turns, ranges, config, index_offset=len(turn_chunks))
    all_chunks = turn_chunks + gap_chunks

    speaker_counts: Counter = Counter()
    for t in turns:
        if not t.is_stage_direction:
            speaker_counts[t.speaker_norm] += 1

    print(f"    Front-matter pages skipped: {front_matter_count} | content pages: {len(content_pages)}")
    print(f"    Parsed {len(turns)} turns | {len(speaker_counts)} unique speakers")
    if speaker_counts:
        print(f"    Top speakers: {speaker_counts.most_common(5)}")
    print(f"    Chunks: {len(turn_chunks)} turn-aware + {len(gap_chunks)} gap/exhibit")

    if dry_run:
        print(f"    [DRY RUN] no DB writes")
        return 0, page_count, len(turns), len(turn_chunks), len(gap_chunks)

    meta = {
        "source_format": "pdf_siss",
        "extractor": "pymupdf",
        "page_count": page_count,
        "front_matter_pages": front_matter_count,
        "turn_count": len(turns),
        "speaker_count": len(speaker_counts),
        "year": year,
    }
    doc_id = upsert_document(cur, collection_id, source_name, str(pdf_path), year, meta)
    delete_document_data(cur, doc_id)

    # Insert ALL pages (front matter included) so the viewer resolves every page.
    page_id_map: Dict[int, int] = {}
    for pdf_page, txt, fm in pages_data:
        role = "front_matter" if fm else "primary"
        page_id_map[pdf_page] = insert_page(cur, doc_id, pdf_page, pdf_page, f"p{pdf_page:04d}", role, txt)

    cur.execute("SAVEPOINT doc_start")
    inserted = 0
    failed = 0
    for chunk in all_chunks:
        try:
            chunk_id = insert_chunk(cur, chunk, pipeline_version)
            page_ids = list(dict.fromkeys(
                page_id_map[p] for p in range(chunk.page_start, chunk.page_end + 1) if p in page_id_map
            ))
            insert_chunk_pages(cur, chunk_id, page_ids)
            first_pid = page_ids[0] if page_ids else None
            last_pid = page_ids[-1] if page_ids else None
            insert_chunk_metadata(cur, chunk_id, doc_id, pipeline_version, first_pid, last_pid, chunk.content_type)
            inserted += 1
        except Exception as e:
            failed += 1
            print(f"      WARNING: chunk {chunk.chunk_index} skipped: {str(e).splitlines()[0][:100]}")
            cur.execute("ROLLBACK TO SAVEPOINT doc_start")
            cur.execute("SAVEPOINT doc_start")
            continue
    if failed:
        print(f"    Chunks: {inserted} inserted, {failed} skipped")

    turn_inserted = sum(1 for c in all_chunks[:len(turn_chunks)])  # informational
    return doc_id, page_count, len(turns), len(turn_chunks), len(gap_chunks)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="SISS 'Scope of Soviet Activity' - Turn-Aware + Gap Ingest")
    ap.add_argument("--input-dir", default=str(Path.home() / "Downloads"),
                    help="Directory containing the PDFs")
    ap.add_argument("--glob", default="Scope_of_Soviet_Activity_in_the_U_S part*.pdf",
                    help="File glob pattern")
    ap.add_argument("--pipeline-version", default="siss_scope_v1_turns")
    ap.add_argument("--target-chars", type=int, default=4000)
    ap.add_argument("--overlap-turns", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = ChunkingConfig(target_chars=args.target_chars, overlap_turns=args.overlap_turns)

    import glob as glob_mod
    paths = sorted(glob_mod.glob(str(Path(args.input_dir) / args.glob)))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"No PDFs found at {args.input_dir}/{args.glob}")
        return

    print("SISS 'Scope of Soviet Activity' - Turn-Aware + Gap Ingest")
    print(f"Found {len(paths)} PDF files")
    print(f"Config: target={config.target_chars}, overlap_turns={config.overlap_turns}, min_gap={config.min_gap_chars}")
    print()

    totals = {"pages": 0, "turns": 0, "turn_chunks": 0, "gap_chunks": 0, "docs": 0}

    with connect() as conn, conn.cursor() as cur:
        ingest_runs.ensure_ingest_runs_table(cur)
        collection_id = get_or_create_collection(cur)
        print(f"Collection: {COLLECTION_SLUG} (id={collection_id})")
        print()

        for pdf_path in paths:
            p = Path(pdf_path)
            pipeline_version = str(args.pipeline_version)
            source_key = f"{COLLECTION_SLUG}:{p.name}"
            fp = None
            try:
                fp = ingest_runs.file_fingerprint_fast(p) if args.dry_run else ingest_runs.file_sha256(p)

                if not args.dry_run and not ingest_runs.should_run(
                    cur, source_key=source_key, source_fingerprint=fp, pipeline_version=pipeline_version
                ):
                    print(f"[skip] {p.name} (already ingested: pipeline={pipeline_version})")
                    continue

                if not args.dry_run:
                    ingest_runs.mark_running(
                        cur, source_key=source_key, source_fingerprint=fp, pipeline_version=pipeline_version
                    )

                doc_id, pages, turns, tchunks, gchunks = process_pdf(
                    p, cur, collection_id, config, pipeline_version, args.dry_run
                )

                totals["pages"] += pages
                totals["turns"] += turns
                totals["turn_chunks"] += tchunks
                totals["gap_chunks"] += gchunks
                totals["docs"] += 1

                if not args.dry_run:
                    ingest_runs.mark_success(cur, source_key=source_key)
                    conn.commit()
                    print(f"    -> document_id={doc_id}")
                print()

            except Exception as e:
                print(f"    ERROR: {e}")
                conn.rollback()
                if not args.dry_run and fp is not None:
                    ingest_runs.mark_failed_best_effort(
                        connect, source_key=source_key, source_fingerprint=fp,
                        pipeline_version=pipeline_version, error=str(e),
                    )
                continue

        print(f"{'[DRY RUN] ' if args.dry_run else ''}Done!")
        print(f"  Documents: {totals['docs']}")
        print(f"  Total pages: {totals['pages']:,}")
        print(f"  Total turns: {totals['turns']:,}")
        print(f"  Turn-aware chunks: {totals['turn_chunks']:,}")
        print(f"  Gap/exhibit chunks: {totals['gap_chunks']:,}")
        print(f"  Pipeline version: {args.pipeline_version}")


if __name__ == "__main__":
    main()
