#!/usr/bin/env python3
"""
McCarthy Hearings Ingest - V2 (Full Speaker-Aware Strategy)

This implements the complete turn-aware ingest:
1. Parse into transcript_turns table first
2. Build chunks from turns (never splitting mid-turn)
3. Track speaker_norms[], turn_id_start/end per chunk
4. Generate embed_text with SPEAKER: tags
5. Enable speaker-level retrieval

Collection slug: mccarthy
"""
import os
import re
import json
import argparse
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict
from collections import Counter

import psycopg2
import fitz  # PyMuPDF


# =============================================================================
# Configuration
# =============================================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")

COLLECTION_SLUG = "mccarthy"
COLLECTION_TITLE = "McCarthy Hearings (1953-1954)"
COLLECTION_DESCRIPTION = "Executive Sessions of the Senate Permanent Subcommittee on Investigations. Speaker-aware transcript ingest with turn-level attribution."


# =============================================================================
# Speaker Detection & Normalization
# =============================================================================

# Main speaker pattern: Mr. COHN., Senator MCCARTHY., The CHAIRMAN.
SPEAKER_PATTERN = re.compile(
    r"^((?:Mr\.|Mrs\.|Ms\.|Dr\.|Senator|Representative|The)\s+[A-Z][A-Z\s\-']+)\.\s*",
    re.MULTILINE
)

# Stage directions: [Laughter], [Recess], [Discussion off the record]
STAGE_DIRECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")

# Known roles mapping (can be extended)
ROLE_MAP = {
    # Counsel
    "MR COHN": "counsel",
    "MR FLANAGAN": "counsel", 
    "MR KENNEDY": "counsel",
    "MR CARR": "counsel",
    "MR SCHINE": "counsel",
    "MR JULIANA": "counsel",
    "MR ANASTOS": "counsel",
    "MR SURINE": "counsel",
    
    # Chairman
    "THE CHAIRMAN": "chair",
    "SENATOR MCCARTHY": "chair",
    
    # Senators
    "SENATOR MCCLELLAN": "senator",
    "SENATOR JACKSON": "senator",
    "SENATOR SYMINGTON": "senator",
    "SENATOR MUNDT": "senator",
    "SENATOR DIRKSEN": "senator",
    "SENATOR POTTER": "senator",
}


def normalize_speaker(speaker_raw: str) -> str:
    """Normalize speaker to canonical form: uppercase, no punctuation, single spaces."""
    # Remove trailing period and extra whitespace
    norm = speaker_raw.strip().rstrip(".")
    # Uppercase
    norm = norm.upper()
    # Collapse whitespace
    norm = re.sub(r"\s+", " ", norm)
    # Remove remaining punctuation except spaces
    norm = re.sub(r"[^\w\s]", "", norm)
    return norm.strip()


def detect_role(speaker_norm: str) -> Optional[str]:
    """Detect role from normalized speaker name."""
    if speaker_norm in ROLE_MAP:
        return ROLE_MAP[speaker_norm]
    
    # Heuristics
    if speaker_norm.startswith("SENATOR"):
        return "senator"
    if speaker_norm.startswith("THE CHAIRMAN"):
        return "chair"
    if speaker_norm.startswith("MR") or speaker_norm.startswith("MRS"):
        # Could be counsel or witness - default to witness
        return "witness"
    
    return None


# =============================================================================
# Turn Parsing
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


def parse_turns_from_pages(pages: List[Tuple[int, int, str]]) -> List[Turn]:
    """
    Parse pages into speaker turns.
    
    Args:
        pages: List of (page_id, pdf_page_number, text)
    
    Returns:
        List of Turn objects with monotonic turn_id
    """
    turns: List[Turn] = []
    turn_id = 0
    char_offset = 0
    
    # Concatenate all page text to find turns across page boundaries
    all_text = ""
    page_char_ranges: List[Tuple[int, int, int, int]] = []  # (page_id, pdf_page, char_start, char_end)
    
    for page_id, pdf_page, text in pages:
        start = len(all_text)
        all_text += text + "\n\n"
        end = len(all_text)
        page_char_ranges.append((page_id, pdf_page, start, end))
    
    # Find all speaker boundaries
    matches = list(SPEAKER_PATTERN.finditer(all_text))
    
    # Also find stage directions
    stage_matches = list(STAGE_DIRECTION.finditer(all_text))
    
    # Combine and sort by position
    all_boundaries = []
    for m in matches:
        all_boundaries.append((m.start(), "speaker", m))
    for m in stage_matches:
        all_boundaries.append((m.start(), "stage", m))
    
    all_boundaries.sort(key=lambda x: x[0])
    
    def get_page_for_char(char_pos: int) -> Tuple[int, int]:
        """Get (page_id, pdf_page) for a character position."""
        for page_id, pdf_page, start, end in page_char_ranges:
            if start <= char_pos < end:
                return page_id, pdf_page
        # Default to last page
        return page_char_ranges[-1][0], page_char_ranges[-1][1]
    
    def get_page_range(char_start: int, char_end: int) -> Tuple[int, int]:
        """Get (page_start, page_end) for a character range."""
        _, start_page = get_page_for_char(char_start)
        _, end_page = get_page_for_char(char_end - 1)
        return start_page, end_page
    
    # Process each boundary
    for i, (pos, btype, match) in enumerate(all_boundaries):
        # Determine end of this turn (start of next boundary or end of text)
        if i + 1 < len(all_boundaries):
            turn_end = all_boundaries[i + 1][0]
        else:
            turn_end = len(all_text)
        
        if btype == "speaker":
            speaker_raw = match.group(1)
            speaker_norm = normalize_speaker(speaker_raw)
            speaker_role = detect_role(speaker_norm)
            
            # Text is everything after the speaker label until next boundary
            turn_text = all_text[match.end():turn_end].strip()
            
            if turn_text:  # Only create turn if there's content
                turn_id += 1
                page_start, page_end = get_page_range(match.start(), turn_end)
                
                turns.append(Turn(
                    turn_id=turn_id,
                    speaker_raw=speaker_raw,
                    speaker_norm=speaker_norm,
                    speaker_role=speaker_role,
                    turn_text=turn_text,
                    page_start=page_start,
                    page_end=page_end,
                    char_start=match.start(),
                    char_end=turn_end,
                    is_stage_direction=False,
                ))
        
        elif btype == "stage":
            stage_text = match.group(1)
            turn_id += 1
            page_start, page_end = get_page_range(match.start(), match.end())
            
            turns.append(Turn(
                turn_id=turn_id,
                speaker_raw=f"[{stage_text}]",
                speaker_norm="__STAGE__",
                speaker_role="stage",
                turn_text=stage_text,
                page_start=page_start,
                page_end=page_end,
                char_start=match.start(),
                char_end=match.end(),
                is_stage_direction=True,
            ))
    
    return turns


# =============================================================================
# Chunking from Turns
# =============================================================================

@dataclass
class ChunkingConfig:
    """Configuration for turn-aware chunking."""
    target_chars: int = 4000
    max_chars: int = 6000
    overlap_turns: int = 2          # Overlap by N turns (not chars)
    min_chunk_chars: int = 200
    max_turns_per_chunk: int = 20


@dataclass 
class TurnChunk:
    """A chunk composed of whole turns."""
    text: str                       # Plain transcript text
    embed_text: str                 # Speaker-tagged text for embeddings
    turn_id_start: int
    turn_id_end: int
    turn_count: int
    speaker_norms: List[str]        # Unique speakers in chunk
    primary_speaker_norm: Optional[str]  # Most frequent speaker
    page_start: int
    page_end: int
    chunk_index: int
    turns: List[Turn]               # The actual turn objects
    speaker_turn_spans: List[Dict]  # Contiguous speaker spans for attribution


def compute_speaker_turn_spans(turns: List[Turn]) -> List[Dict]:
    """
    Compute contiguous speaker spans within turns.
    
    Merges consecutive turns by same speaker into spans:
    [{"speaker":"WELCH","turn_id_start":450,"turn_id_end":454},
     {"speaker":"MCCARTHY","turn_id_start":455,"turn_id_end":455}]
    """
    if not turns:
        return []
    
    spans = []
    current_speaker = None
    current_start = None
    current_end = None
    
    for turn in turns:
        speaker = turn.speaker_norm
        
        if speaker == current_speaker:
            # Extend current span
            current_end = turn.turn_id
        else:
            # Flush previous span
            if current_speaker is not None:
                spans.append({
                    "speaker": current_speaker,
                    "turn_id_start": current_start,
                    "turn_id_end": current_end,
                })
            # Start new span
            current_speaker = speaker
            current_start = turn.turn_id
            current_end = turn.turn_id
    
    # Flush final span
    if current_speaker is not None:
        spans.append({
            "speaker": current_speaker,
            "turn_id_start": current_start,
            "turn_id_end": current_end,
        })
    
    return spans


def generate_embed_text(turns: List[Turn], doc_ref: str = "") -> str:
    """
    Generate speaker-tagged text for embeddings.
    
    Format:
    [Hearing: mccarthy | Turns 1-5 | Pages 10-12]
    SPEAKER: MR COHN
    TEXT: What is your name?
    SPEAKER: MR WITNESS
    TEXT: John Smith.
    """
    lines = []
    
    # Header
    if turns:
        header = f"[Turns {turns[0].turn_id}-{turns[-1].turn_id} | Pages {turns[0].page_start}-{turns[-1].page_end}]"
        if doc_ref:
            header = f"[{doc_ref} | " + header[1:]
        lines.append(header)
    
    # Turns with speaker tags
    for turn in turns:
        if turn.is_stage_direction:
            lines.append(f"[{turn.turn_text}]")
        else:
            lines.append(f"SPEAKER: {turn.speaker_norm}")
            lines.append(f"TEXT: {turn.turn_text}")
        lines.append("")
    
    return "\n".join(lines)


def create_chunks_from_turns(
    turns: List[Turn],
    config: ChunkingConfig,
    doc_ref: str = "",
) -> List[TurnChunk]:
    """
    Create chunks from turns, never splitting mid-turn.
    
    Rules:
    - Accumulate whole turns until target size
    - Overlap by N turns for context
    - Track all speakers in each chunk
    """
    if not turns:
        return []
    
    chunks: List[TurnChunk] = []
    current_turns: List[Turn] = []
    current_chars = 0
    
    def flush_chunk():
        nonlocal current_turns, current_chars
        
        if not current_turns:
            return
        
        # Build plain text
        text_parts = []
        for t in current_turns:
            if t.is_stage_direction:
                text_parts.append(f"[{t.turn_text}]")
            else:
                text_parts.append(f"{t.speaker_raw}. {t.turn_text}")
        text = "\n\n".join(text_parts)
        
        if len(text) < config.min_chunk_chars:
            return
        
        # Generate embed text
        embed_text = generate_embed_text(current_turns, doc_ref)
        
        # Compute speaker stats
        speaker_counts: Counter = Counter()
        for t in current_turns:
            if not t.is_stage_direction:
                speaker_counts[t.speaker_norm] += 1
        
        speaker_norms = list(speaker_counts.keys())
        primary_speaker = speaker_counts.most_common(1)[0][0] if speaker_counts else None
        
        # Compute speaker turn spans for precise attribution
        speaker_turn_spans = compute_speaker_turn_spans(current_turns)
        
        chunk = TurnChunk(
            text=text,
            embed_text=embed_text,
            turn_id_start=current_turns[0].turn_id,
            turn_id_end=current_turns[-1].turn_id,
            turn_count=len(current_turns),
            speaker_norms=speaker_norms,
            primary_speaker_norm=primary_speaker,
            page_start=min(t.page_start for t in current_turns),
            page_end=max(t.page_end for t in current_turns),
            chunk_index=len(chunks),
            turns=current_turns.copy(),
            speaker_turn_spans=speaker_turn_spans,
        )
        chunks.append(chunk)
        
        current_turns = []
        current_chars = 0
    
    # Process turns
    for turn in turns:
        turn_len = len(turn.turn_text)
        
        # Check if we should flush
        should_flush = False
        if current_chars + turn_len > config.target_chars and current_turns:
            should_flush = True
        if len(current_turns) >= config.max_turns_per_chunk:
            should_flush = True
        
        if should_flush:
            flush_chunk()
            
            # Apply overlap - keep last N turns
            if chunks and config.overlap_turns > 0:
                overlap = chunks[-1].turns[-config.overlap_turns:]
                current_turns = [Turn(
                    turn_id=t.turn_id,
                    speaker_raw=t.speaker_raw,
                    speaker_norm=t.speaker_norm,
                    speaker_role=t.speaker_role,
                    turn_text=t.turn_text,
                    page_start=t.page_start,
                    page_end=t.page_end,
                    char_start=t.char_start,
                    char_end=t.char_end,
                    is_stage_direction=t.is_stage_direction,
                ) for t in overlap]
                current_chars = sum(len(t.turn_text) for t in current_turns)
        
        current_turns.append(turn)
        current_chars += turn_len
    
    # Flush final chunk
    flush_chunk()
    
    return chunks


# =============================================================================
# Text Normalization
# =============================================================================

VERDATE_RE = re.compile(r"VerDate\s+\w+\s+\d+\s+\d+:\d+.*$", re.MULTILINE)
JKT_RE = re.compile(r"Jkt\s+\d+\s+PO\s+\d+\s+Frm.*$", re.MULTILINE)


def normalize_text(raw_text: str) -> str:
    """Clean GPO formatting artifacts."""
    text = raw_text
    text = VERDATE_RE.sub("", text)
    text = JKT_RE.sub("", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Remove standalone page numbers
    lines = []
    for line in text.split("\n"):
        if re.match(r"^\s*\d+\s*$", line):
            continue
        if re.match(r"^\s*\(?[IVXLCDM]+\)?\s*$", line, re.IGNORECASE):
            continue
        lines.append(line)
    
    return "\n".join(lines).strip()


# =============================================================================
# Database Operations
# =============================================================================

def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def get_or_create_collection(cur) -> int:
    cur.execute("SELECT id FROM collections WHERE slug = %s", (COLLECTION_SLUG,))
    r = cur.fetchone()
    if r:
        return int(r[0])
    
    cur.execute(
        """
        INSERT INTO collections (slug, title, description)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (COLLECTION_SLUG, COLLECTION_TITLE, COLLECTION_DESCRIPTION),
    )
    return int(cur.fetchone()[0])


def upsert_document(cur, collection_id: int, source_name: str, source_ref: str,
                    volume: str, metadata: dict) -> int:
    cur.execute(
        """
        INSERT INTO documents (collection_id, source_name, source_ref, volume, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (collection_id, source_name, volume_key)
        DO UPDATE SET source_ref = EXCLUDED.source_ref, metadata = EXCLUDED.metadata
        RETURNING id
        """,
        (collection_id, source_name, source_ref, volume, json.dumps(metadata)),
    )
    return int(cur.fetchone()[0])


def delete_document_data(cur, document_id: int):
    """Delete all derived data for a document."""
    # Delete chunk_turns first (if table exists)
    try:
        cur.execute("DELETE FROM chunk_turns WHERE turn_id IN (SELECT id FROM transcript_turns WHERE document_id = %s)", (document_id,))
    except:
        pass
    
    # Delete chunks
    cur.execute("""
        DELETE FROM chunks WHERE id IN (
            SELECT cp.chunk_id FROM chunk_pages cp
            JOIN pages p ON p.id = cp.page_id
            WHERE p.document_id = %s
        )
    """, (document_id,))
    
    # Delete turns
    try:
        cur.execute("DELETE FROM transcript_turns WHERE document_id = %s", (document_id,))
    except:
        pass
    
    # Delete pages
    cur.execute("DELETE FROM pages WHERE document_id = %s", (document_id,))


def insert_page(cur, document_id: int, page_seq: int, pdf_page_number: int,
                logical_label: str, raw_text: str) -> int:
    cur.execute(
        """
        INSERT INTO pages (document_id, logical_page_label, pdf_page_number, page_seq,
                          language, content_role, raw_text)
        VALUES (%s, %s, %s, %s, 'en', 'primary', %s)
        RETURNING id
        """,
        (document_id, logical_label, pdf_page_number, page_seq, raw_text),
    )
    return int(cur.fetchone()[0])


def insert_turn(cur, document_id: int, turn: Turn) -> int:
    """Insert a turn into transcript_turns."""
    cur.execute(
        """
        INSERT INTO transcript_turns (document_id, turn_id, speaker_raw, speaker_norm,
                                     speaker_role, turn_text, page_start, page_end,
                                     char_start, char_end, is_stage_direction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (document_id, turn.turn_id, turn.speaker_raw, turn.speaker_norm,
         turn.speaker_role, turn.turn_text, turn.page_start, turn.page_end,
         turn.char_start, turn.char_end, turn.is_stage_direction),
    )
    return int(cur.fetchone()[0])


def insert_chunk_v2(cur, chunk: TurnChunk, pipeline_version: str) -> int:
    """Insert a chunk with full speaker metadata."""
    # Limit speaker_norms to top 20 to avoid index size issues
    speaker_norms = chunk.speaker_norms[:20] if chunk.speaker_norms else []
    
    # Truncate text fields to avoid trigram index size limits (8KB max)
    # The indexes are on text/clean_text, so we need to keep those reasonable
    max_text_len = 7500  # Leave some margin below 8KB
    text = chunk.text[:max_text_len] if len(chunk.text) > max_text_len else chunk.text
    
    # embed_text can be NULL if too long - we'll regenerate at query time
    embed_text = chunk.embed_text[:max_text_len] if chunk.embed_text and len(chunk.embed_text) > max_text_len else chunk.embed_text
    
    # speaker_turn_spans as JSONB
    speaker_turn_spans_json = json.dumps(chunk.speaker_turn_spans) if chunk.speaker_turn_spans else None
    
    cur.execute(
        """
        INSERT INTO chunks (text, clean_text, pipeline_version,
                           turn_id_start, turn_id_end, turn_count,
                           speaker_norms, primary_speaker_norm, embed_text,
                           speaker_turn_spans)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        RETURNING id
        """,
        (text, text, pipeline_version,
         chunk.turn_id_start, chunk.turn_id_end, chunk.turn_count,
         speaker_norms, chunk.primary_speaker_norm, embed_text,
         speaker_turn_spans_json),
    )
    return int(cur.fetchone()[0])


def insert_chunk_metadata(cur, chunk_id: int, document_id: int, pipeline_version: str,
                          first_page_id: int, last_page_id: int):
    """Insert chunk_metadata to link chunk to document/collection."""
    # Check if already exists
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
         first_page_id, last_page_id, "hearing_transcript"),
    )


def insert_chunk_pages(cur, chunk_id: int, page_ids: List[int]):
    for i, page_id in enumerate(page_ids, start=1):
        cur.execute(
            "INSERT INTO chunk_pages (chunk_id, page_id, span_order) VALUES (%s, %s, %s)",
            (chunk_id, page_id, i),
        )


def insert_chunk_turns(cur, chunk_id: int, chunk: TurnChunk, turn_db_ids: Dict[int, int]):
    """
    Populate chunk_turns junction table for precise turn-level evidence.
    
    Enables:
    - "return all turns in this chunk"
    - "show me speaker X turns only"
    - "export exact transcript snippets with speaker labels"
    """
    for i, turn in enumerate(chunk.turns, start=1):
        turn_db_id = turn_db_ids.get(turn.turn_id)
        if turn_db_id is None:
            continue
        
        cur.execute(
            """
            INSERT INTO chunk_turns (chunk_id, turn_id, span_order, speaker_norm)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id, turn_id) DO NOTHING
            """,
            (chunk_id, turn_db_id, i, turn.speaker_norm),
        )


# =============================================================================
# PDF Extraction
# =============================================================================

def extract_page_text(doc, page_index: int) -> str:
    page = doc.load_page(page_index)
    txt = page.get_text("text") or ""
    return txt if txt.strip() else ""


# =============================================================================
# Main Processing
# =============================================================================

def process_pdf(
    pdf_path: Path,
    cur,
    collection_id: int,
    config: ChunkingConfig,
    pipeline_version: str,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Process PDF with full turn-aware ingest.
    
    Returns (doc_id, page_count, turn_count, chunk_count)
    """
    source_name = pdf_path.name
    volume_match = re.search(r"Vol(\d+)", source_name)
    volume = f"Volume {volume_match.group(1)}" if volume_match else ""
    doc_ref = f"mccarthy_{volume.replace(' ', '')}" if volume else "mccarthy"
    
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    
    print(f"  Processing {source_name}: {page_count} pages")
    
    # Extract and normalize pages
    pages_data: List[Tuple[int, int, str]] = []  # (temp_page_id, pdf_page, text)
    for i in range(page_count):
        txt = normalize_text(extract_page_text(doc, i))
        pages_data.append((i, i + 1, txt))
    
    doc.close()
    
    # Parse turns
    turns = parse_turns_from_pages(pages_data)
    print(f"    Parsed {len(turns)} speaker turns")
    
    # Compute speaker stats
    speaker_counts: Counter = Counter()
    for t in turns:
        if not t.is_stage_direction:
            speaker_counts[t.speaker_norm] += 1
    print(f"    Unique speakers: {len(speaker_counts)}")
    print(f"    Top speakers: {speaker_counts.most_common(5)}")
    
    if dry_run:
        # Create chunks but don't insert
        chunks = create_chunks_from_turns(turns, config, doc_ref)
        print(f"    [DRY RUN] Would create {len(chunks)} chunks")
        return 0, page_count, len(turns), len(chunks)
    
    # Create document
    meta = {
        "source_format": "pdf_gpo",
        "extractor": "pymupdf",
        "page_count": page_count,
        "turn_count": len(turns),
        "speaker_count": len(speaker_counts),
        "document_type": "hearing_transcript",
        "ingest_version": "v2_turn_aware",
    }
    
    doc_id = upsert_document(cur, collection_id, source_name, str(pdf_path), volume, meta)
    
    # Clear existing data
    delete_document_data(cur, doc_id)
    
    # Insert pages
    page_id_map: Dict[int, int] = {}  # pdf_page -> page_id
    for _, pdf_page, txt in pages_data:
        page_id = insert_page(cur, doc_id, pdf_page, pdf_page, f"p{pdf_page:04d}", txt)
        page_id_map[pdf_page] = page_id
    
    # Insert turns
    turn_db_ids: Dict[int, int] = {}  # turn.turn_id -> db id
    for turn in turns:
        db_id = insert_turn(cur, doc_id, turn)
        turn_db_ids[turn.turn_id] = db_id
    
    # Create and insert chunks
    chunks = create_chunks_from_turns(turns, config, doc_ref)
    print(f"    Created {len(chunks)} chunks")
    
    for chunk in chunks:
        chunk_id = insert_chunk_v2(cur, chunk, pipeline_version)
        
        # Link to pages
        page_ids = list(dict.fromkeys(
            page_id_map[p] for p in range(chunk.page_start, chunk.page_end + 1)
            if p in page_id_map
        ))
        insert_chunk_pages(cur, chunk_id, page_ids)
        
        # Insert chunk_metadata to link to document/collection
        first_page_id = page_ids[0] if page_ids else None
        last_page_id = page_ids[-1] if page_ids else None
        insert_chunk_metadata(cur, chunk_id, doc_id, pipeline_version, first_page_id, last_page_id)
        
        # Populate chunk_turns for turn-level evidence retrieval
        insert_chunk_turns(cur, chunk_id, chunk, turn_db_ids)
    
    return doc_id, page_count, len(turns), len(chunks)


def main():
    ap = argparse.ArgumentParser(description="McCarthy Hearings V2 - Turn-Aware Ingest")
    ap.add_argument("--input-dir", default="data/raw/mccarthy")
    ap.add_argument("--glob", default="*.pdf")
    ap.add_argument("--pipeline-version", default="mccarthy_v2_turns")
    ap.add_argument("--target-chars", type=int, default=4000)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--overlap-turns", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    
    config = ChunkingConfig(
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        overlap_turns=args.overlap_turns,
    )
    
    import glob as glob_mod
    paths = sorted(glob_mod.glob(str(Path(args.input_dir) / args.glob)))
    if args.limit:
        paths = paths[:args.limit]
    
    if not paths:
        print(f"No PDFs found")
        return
    
    print(f"McCarthy Hearings V2 - Turn-Aware Ingest")
    print(f"Config: target={config.target_chars}, overlap_turns={config.overlap_turns}")
    print()
    
    total_pages = 0
    total_turns = 0
    total_chunks = 0
    
    with connect() as conn, conn.cursor() as cur:
        collection_id = get_or_create_collection(cur)
        print(f"Collection: {COLLECTION_SLUG} (id={collection_id})")
        print()
        
        for pdf_path in paths:
            doc_id, pages, turns, chunks = process_pdf(
                Path(pdf_path), cur, collection_id, config,
                args.pipeline_version, args.dry_run,
            )
            total_pages += pages
            total_turns += turns
            total_chunks += chunks
            
            if not args.dry_run:
                conn.commit()
                print(f"    -> document_id={doc_id}")
        
        print()
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Done!")
        print(f"  Total pages: {total_pages}")
        print(f"  Total turns: {total_turns}")
        print(f"  Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
