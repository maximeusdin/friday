#!/usr/bin/env python3
"""
HUAC Hearings Ingest - Turn-Aware Chunking

Ingests House Un-American Activities Committee hearing transcripts with
speaker-aware chunking. Similar to McCarthy hearings but with HUAC-specific
speaker patterns.

Key hearings included:
- 1948 Communist Espionage hearings (Alger Hiss, Whittaker Chambers)
- 1947 Hollywood hearings (motion picture industry investigation)
- Soviet activity hearings

Collection slug: huac_hearings
"""
import os
import sys
import io
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
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

COLLECTION_SLUG = "huac_hearings"
COLLECTION_TITLE = "HUAC Hearings (1947-1957)"
COLLECTION_DESCRIPTION = """House Un-American Activities Committee hearing transcripts 
including the 1948 Communist espionage hearings (Alger Hiss case), 1947 Hollywood 
hearings, and Soviet activity investigations. Speaker-aware transcript ingest with 
turn-level attribution."""


# =============================================================================
# Speaker Detection
# =============================================================================

# Main speaker pattern: Mr. NIXON., The CHAIRMAN., Senator MUNDT.
# More forgiving for OCR errors (mixed case, extra periods/spaces)
SPEAKER_PATTERN = re.compile(
    r"^((?:Mr|Mrs|Ms|Miss|Dr|Senator|Representative|The)\s*[.\s]*\s*[A-Za-z][A-Za-z]+)\s*[.\s]+\s*(?=[A-Z])",
    re.MULTILINE
)

# Stage directions: [Laughter], [Recess], [Discussion off the record]
STAGE_DIRECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")

# HUAC-specific role mapping
ROLE_MAP = {
    # Committee leadership
    "THE CHAIRMAN": "chair",
    "MR THOMAS": "chair",  # J. Parnell Thomas
    "MR WOOD": "chair",    # John S. Wood
    
    # Notable committee members
    "MR NIXON": "committee",
    "MR MUNDT": "committee",
    "MR RANKIN": "committee",
    "MR HEBERT": "committee",
    "MR MCDOWELL": "committee",
    "MR PETERSON": "committee",
    "MR VAIL": "committee",
    
    # Senate subcommittee (for scope of soviet activity file)
    "SENATOR JENNER": "senator",
    "SENATOR WATKINS": "senator",
    
    # Staff/Counsel
    "MR STRIPLING": "counsel",
    "MR RUSSELL": "counsel",
    "MR WHEELER": "counsel",
    "MR TAVENNER": "counsel",
    "MR MANDEL": "counsel",
    "MR MORRIS": "counsel",
    
    # Famous witnesses
    "MR HISS": "witness",
    "MR CHAMBERS": "witness",
    "MR BENTLEY": "witness",
    "MISS BENTLEY": "witness",
}


def normalize_speaker(speaker_raw: str) -> str:
    """Normalize speaker to canonical form."""
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
    """Configuration for turn-aware chunking."""
    target_chars: int = 4000
    max_chars: int = 6000
    overlap_turns: int = 2
    min_chunk_chars: int = 200
    max_turns_per_chunk: int = 25


@dataclass
class TurnChunk:
    """A chunk composed of whole turns."""
    text: str
    embed_text: str
    turn_id_start: int
    turn_id_end: int
    turn_count: int
    speaker_norms: List[str]
    primary_speaker_norm: Optional[str]
    page_start: int
    page_end: int
    chunk_index: int
    turns: List[Turn]
    speaker_turn_spans: List[Dict]


# =============================================================================
# Turn Parsing
# =============================================================================

def parse_turns_from_pages(pages: List[Tuple[int, int, str]]) -> List[Turn]:
    """Parse pages into speaker turns."""
    turns: List[Turn] = []
    turn_id = 0
    
    # Concatenate all page text
    all_text = ""
    page_char_ranges: List[Tuple[int, int, int, int]] = []
    
    for page_id, pdf_page, text in pages:
        start = len(all_text)
        all_text += text + "\n\n"
        end = len(all_text)
        page_char_ranges.append((page_id, pdf_page, start, end))
    
    # Find all speaker boundaries
    matches = list(SPEAKER_PATTERN.finditer(all_text))
    stage_matches = list(STAGE_DIRECTION.finditer(all_text))
    
    # Combine and sort by position
    all_boundaries = []
    for m in matches:
        all_boundaries.append((m.start(), "speaker", m))
    for m in stage_matches:
        all_boundaries.append((m.start(), "stage", m))
    
    all_boundaries.sort(key=lambda x: x[0])
    
    def get_page_for_char(char_pos: int) -> Tuple[int, int]:
        for page_id, pdf_page, start, end in page_char_ranges:
            if start <= char_pos < end:
                return page_id, pdf_page
        return page_char_ranges[-1][0], page_char_ranges[-1][1]
    
    def get_page_range(char_start: int, char_end: int) -> Tuple[int, int]:
        _, start_page = get_page_for_char(char_start)
        _, end_page = get_page_for_char(char_end - 1)
        return start_page, end_page
    
    # Process each boundary
    for i, (pos, btype, match) in enumerate(all_boundaries):
        if i + 1 < len(all_boundaries):
            turn_end = all_boundaries[i + 1][0]
        else:
            turn_end = len(all_text)
        
        if btype == "speaker":
            speaker_raw = match.group(1)
            speaker_norm = normalize_speaker(speaker_raw)
            speaker_role = detect_role(speaker_norm)
            
            turn_text = all_text[match.end():turn_end].strip()
            
            if turn_text:
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
# Chunking
# =============================================================================

def compute_speaker_turn_spans(turns: List[Turn]) -> List[Dict]:
    """Compute contiguous speaker spans within turns."""
    if not turns:
        return []
    
    spans = []
    current_speaker = None
    current_start = None
    current_end = None
    
    for turn in turns:
        speaker = turn.speaker_norm
        
        if speaker == current_speaker:
            current_end = turn.turn_id
        else:
            if current_speaker is not None:
                spans.append({
                    "speaker": current_speaker,
                    "turn_id_start": current_start,
                    "turn_id_end": current_end,
                })
            current_speaker = speaker
            current_start = turn.turn_id
            current_end = turn.turn_id
    
    if current_speaker is not None:
        spans.append({
            "speaker": current_speaker,
            "turn_id_start": current_start,
            "turn_id_end": current_end,
        })
    
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


def create_chunks_from_turns(
    turns: List[Turn],
    config: ChunkingConfig,
    doc_ref: str = "",
) -> List[TurnChunk]:
    """Create chunks from turns, never splitting mid-turn."""
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
        
        embed_text = generate_embed_text(current_turns, doc_ref)
        
        speaker_counts: Counter = Counter()
        for t in current_turns:
            if not t.is_stage_direction:
                speaker_counts[t.speaker_norm] += 1
        
        speaker_norms = list(speaker_counts.keys())
        primary_speaker = speaker_counts.most_common(1)[0][0] if speaker_counts else None
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
    
    for turn in turns:
        turn_len = len(turn.turn_text)
        
        should_flush = False
        if current_chars + turn_len > config.target_chars and current_turns:
            should_flush = True
        if len(current_turns) >= config.max_turns_per_chunk:
            should_flush = True
        
        if should_flush:
            flush_chunk()
            
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
    
    flush_chunk()
    
    return chunks


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_text(raw_text: str) -> str:
    """Clean text artifacts."""
    text = raw_text
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
                    year: str, metadata: dict) -> int:
    cur.execute(
        "SELECT id FROM documents WHERE collection_id = %s AND source_name = %s",
        (collection_id, source_name)
    )
    r = cur.fetchone()
    if r:
        doc_id = int(r[0])
        cur.execute(
            "UPDATE documents SET source_ref = %s, metadata = %s WHERE id = %s",
            (source_ref, json.dumps(metadata), doc_id)
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
    """Delete all derived data for a document."""
    cur.execute("""
        DELETE FROM chunks WHERE id IN (
            SELECT cp.chunk_id FROM chunk_pages cp
            JOIN pages p ON p.id = cp.page_id
            WHERE p.document_id = %s
        )
    """, (document_id,))
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


def safe_truncate_bytes(text: str, max_bytes: int) -> str:
    """Truncate text to fit within max_bytes, respecting UTF-8 boundaries."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore')


def insert_chunk(cur, chunk: TurnChunk, pipeline_version: str) -> int:
    """Insert a chunk with speaker metadata and retry logic for index limits."""
    speaker_norms = chunk.speaker_norms[:20] if chunk.speaker_norms else []
    speaker_turn_spans_json = json.dumps(chunk.speaker_turn_spans) if chunk.speaker_turn_spans else None
    
    max_sizes = [None, 6000, 4000, 2500]
    
    for i, max_size in enumerate(max_sizes):
        text_to_insert = chunk.text if max_size is None else safe_truncate_bytes(chunk.text, max_size)
        embed_text = chunk.embed_text if max_size is None else safe_truncate_bytes(chunk.embed_text, max_size) if chunk.embed_text else None
        
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
                 speaker_norms, chunk.primary_speaker_norm, embed_text,
                 speaker_turn_spans_json),
            )
            result = int(cur.fetchone()[0])
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            return result
        except Exception as e:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            error_msg = str(e)
            if "index row requires" in error_msg and "maximum size" in error_msg:
                if max_size == max_sizes[-1]:
                    raise
                continue
            else:
                raise
    
    raise Exception("Failed to insert chunk after all truncation attempts")


def insert_chunk_pages(cur, chunk_id: int, page_ids: List[int]):
    for i, page_id in enumerate(page_ids, start=1):
        cur.execute(
            "INSERT INTO chunk_pages (chunk_id, page_id, span_order) VALUES (%s, %s, %s)",
            (chunk_id, page_id, i),
        )


def insert_chunk_metadata(cur, chunk_id: int, document_id: int, pipeline_version: str,
                          first_page_id: int, last_page_id: int, content_type: str):
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
# Document Classification
# =============================================================================

def classify_document(pages_text: List[str]) -> Tuple[str, str]:
    """
    Classify document type based on content.
    Returns (doc_type, content_type).
    
    Checks pages 5-55 for speaker density to distinguish
    transcripts from reports (skipping front matter).
    """
    # Count speaker patterns in pages 5-55 (skip front matter)
    speaker_count = 0
    for page_text in pages_text[5:55]:
        matches = SPEAKER_PATTERN.findall(page_text)
        speaker_count += len(matches)
    
    # Transcripts typically have 10+ speaker turns per page
    # Reports have very few
    if speaker_count > 50:  # ~1+ per page average
        return "transcript", "huac_testimony"
    else:
        return "report", "huac_report"


def extract_year(filename: str, content: str) -> Optional[str]:
    """Extract year from filename or content."""
    # Try filename first
    match = re.search(r'(194\d|195\d|196\d)', filename)
    if match:
        return match.group(1)
    
    # Try content
    match = re.search(r'(194\d|195\d|196\d)', content[:5000])
    if match:
        return match.group(1)
    
    return None


def extract_topic(filename: str, content: str) -> str:
    """Extract topic keywords from filename and content."""
    topics = []
    
    text = (filename + " " + content[:3000]).upper()
    
    if "HISS" in text:
        topics.append("Hiss")
    if "CHAMBERS" in text:
        topics.append("Chambers")
    if "HOLLYWOOD" in text or "MOTION PICTURE" in text or "MOTION-PICTURE" in text:
        topics.append("Hollywood")
    if "ESPIONAGE" in text:
        topics.append("Espionage")
    if "SOVIET" in text:
        topics.append("Soviet")
    if "COMMUNIST" in text:
        topics.append("Communist")
    
    return ", ".join(topics) if topics else "General"


# =============================================================================
# PDF Processing
# =============================================================================

def extract_page_text(doc, page_index: int) -> str:
    page = doc.load_page(page_index)
    txt = page.get_text("text") or ""
    return txt if txt.strip() else ""


def process_pdf(
    pdf_path: Path,
    cur,
    collection_id: int,
    config: ChunkingConfig,
    pipeline_version: str,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Process PDF with turn-aware ingest for transcripts.
    
    Returns (doc_id, page_count, turn_count, chunk_count)
    """
    source_name = pdf_path.name
    
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    
    print(f"  Processing {source_name}: {page_count} pages")
    
    # Extract and normalize pages
    pages_data: List[Tuple[int, int, str]] = []
    pages_text: List[str] = []
    
    for i in range(page_count):
        txt = normalize_text(extract_page_text(doc, i))
        pages_data.append((i, i + 1, txt))
        pages_text.append(txt)
    
    doc.close()
    
    # Classify document
    doc_type, content_type = classify_document(pages_text)
    
    if doc_type == "report":
        print(f"    Skipping - detected as REPORT (not transcript)")
        print(f"    Use ingest_huac_reports.py for this file")
        return 0, page_count, 0, 0
    
    # Extract metadata
    full_text = "\n".join(pages_text[:20])
    year = extract_year(source_name, full_text) or "unknown"
    topic = extract_topic(source_name, full_text)
    doc_ref = f"huac_{year}"
    
    print(f"    Type: {doc_type}, Year: {year}, Topic: {topic}")
    
    # Parse turns
    turns = parse_turns_from_pages(pages_data)
    print(f"    Parsed {len(turns)} speaker turns")
    
    # Compute speaker stats
    speaker_counts: Counter = Counter()
    for t in turns:
        if not t.is_stage_direction:
            speaker_counts[t.speaker_norm] += 1
    print(f"    Unique speakers: {len(speaker_counts)}")
    if speaker_counts:
        print(f"    Top speakers: {speaker_counts.most_common(5)}")
    
    if dry_run:
        chunks = create_chunks_from_turns(turns, config, doc_ref)
        print(f"    [DRY RUN] Would create {len(chunks)} chunks")
        return 0, page_count, len(turns), len(chunks)
    
    # Create document
    meta = {
        "source_format": "pdf_huac",
        "extractor": "pymupdf",
        "page_count": page_count,
        "turn_count": len(turns),
        "speaker_count": len(speaker_counts),
        "document_type": doc_type,
        "year": year,
        "topic": topic,
    }
    
    doc_id = upsert_document(cur, collection_id, source_name, str(pdf_path), year, meta)
    
    # Clear existing data
    delete_document_data(cur, doc_id)
    
    # Insert pages
    page_id_map: Dict[int, int] = {}
    for _, pdf_page, txt in pages_data:
        page_id = insert_page(cur, doc_id, pdf_page, pdf_page, f"p{pdf_page:04d}", txt)
        page_id_map[pdf_page] = page_id
    
    # Create and insert chunks
    chunks = create_chunks_from_turns(turns, config, doc_ref)
    print(f"    Created {len(chunks)} chunks")
    
    # Insert with error handling
    cur.execute("SAVEPOINT doc_start")
    chunks_inserted = 0
    chunks_failed = 0
    
    for chunk in chunks:
        try:
            chunk_id = insert_chunk(cur, chunk, pipeline_version)
            
            page_ids = list(dict.fromkeys(
                page_id_map[p] for p in range(chunk.page_start, chunk.page_end + 1)
                if p in page_id_map
            ))
            insert_chunk_pages(cur, chunk_id, page_ids)
            
            first_page_id = page_ids[0] if page_ids else None
            last_page_id = page_ids[-1] if page_ids else None
            insert_chunk_metadata(cur, chunk_id, doc_id, pipeline_version, 
                                 first_page_id, last_page_id, content_type)
            chunks_inserted += 1
        except Exception as e:
            chunks_failed += 1
            error_msg = str(e).split('\n')[0][:100]
            print(f"      WARNING: Chunk {chunk.chunk_index} skipped: {error_msg}")
            cur.execute("ROLLBACK TO SAVEPOINT doc_start")
            cur.execute("SAVEPOINT doc_start")
            continue
    
    if chunks_failed > 0:
        print(f"    Chunks: {chunks_inserted} inserted, {chunks_failed} skipped")
    
    return doc_id, page_count, len(turns), chunks_inserted


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="HUAC Hearings - Turn-Aware Ingest")
    ap.add_argument("--input-dir", default="data/raw/umamerican_hearings",
                    help="Directory containing PDFs")
    ap.add_argument("--glob", default="*.pdf", help="File glob pattern")
    ap.add_argument("--pipeline-version", default="huac_hearings_v1_turns",
                    help="Pipeline version")
    ap.add_argument("--target-chars", type=int, default=4000,
                    help="Target chunk size in chars")
    ap.add_argument("--overlap-turns", type=int, default=2,
                    help="Number of turns to overlap between chunks")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of PDFs to process")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write to database")
    args = ap.parse_args()
    
    config = ChunkingConfig(
        target_chars=args.target_chars,
        overlap_turns=args.overlap_turns,
    )
    
    import glob as glob_mod
    
    # Files that should be processed as reports (by huac_reports), not transcripts
    # These are explicitly excluded to avoid duplication
    REPORT_FILES = {
        "americannegroinc00unit.pdf",      # American Negro in Communist Party
        "communistconspir195601aunit.pdf", # Communist Conspiracy (1956)
        "communistinfiltr04unit.pdf",      # Communist Infiltration (Vol. 4)
        "communistoutlets02unit.pdf",      # Communist Outlets (Vol. 2)
        "scopeofsovietact2123unit.pdf",    # Scope of Soviet Activities (1952-53)
        "shamefulyearsthi1952unit.pdf",    # Shameful Years (1952)
        "sovietespionagea1949unit.pdf",    # Soviet Espionage Activities (1949)
        "spotlightonspies1949unit.pdf",    # Spotlight on Spies (1949)
    }
    
    all_paths = sorted(glob_mod.glob(str(Path(args.input_dir) / args.glob)))
    paths = [p for p in all_paths if Path(p).name not in REPORT_FILES]
    
    if args.limit:
        paths = paths[:args.limit]
    
    skipped_reports = len(all_paths) - len(paths)
    if skipped_reports > 0:
        print(f"Skipping {skipped_reports} report files (processed by huac_reports)")
    
    if not paths:
        print(f"No transcript PDFs found at {args.input_dir}/{args.glob}")
        return
    
    print(f"HUAC Hearings - Turn-Aware Ingest")
    print(f"Found {len(paths)} PDF files")
    print(f"Config: target={config.target_chars}, overlap_turns={config.overlap_turns}")
    print()
    
    total_pages = 0
    total_turns = 0
    total_chunks = 0
    transcripts_processed = 0
    reports_skipped = 0
    
    with connect() as conn, conn.cursor() as cur:
        ingest_runs.ensure_ingest_runs_table(cur)
        collection_id = get_or_create_collection(cur)
        print(f"Collection: {COLLECTION_SLUG} (id={collection_id})")
        print()
        
        for pdf_path in paths:
            try:
                p = Path(pdf_path)
                pipeline_version = str(args.pipeline_version)
                source_key = f"{COLLECTION_SLUG}:{p.name}"
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

                doc_id, pages, turns, chunks = process_pdf(
                    p, cur, collection_id, config,
                    pipeline_version, args.dry_run,
                )
                
                if turns > 0:  # Was processed as transcript
                    total_pages += pages
                    total_turns += turns
                    total_chunks += chunks
                    transcripts_processed += 1
                    
                    if not args.dry_run:
                        ingest_runs.mark_success(cur, source_key=source_key)
                        conn.commit()
                        print(f"    -> document_id={doc_id}")
                else:
                    reports_skipped += 1
            
            except Exception as e:
                print(f"    ERROR: {e}")
                conn.rollback()
                if not args.dry_run:
                    ingest_runs.mark_failed_best_effort(
                        connect,
                        source_key=source_key,
                        source_fingerprint=fp,
                        pipeline_version=pipeline_version,
                        error=str(e),
                    )
                continue
        
        print()
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Done!")
        print(f"  Transcripts processed: {transcripts_processed}")
        print(f"  Reports skipped: {reports_skipped}")
        print(f"  Total pages: {total_pages:,}")
        print(f"  Total turns: {total_turns:,}")
        print(f"  Total chunks: {total_chunks:,}")
        print(f"  Pipeline version: {args.pipeline_version}")


if __name__ == "__main__":
    main()
