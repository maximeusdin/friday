#!/usr/bin/env python3
"""
Ingest script for McCarthy Hearings (Senate Permanent Subcommittee on Investigations).

This script handles hearing transcripts with:
- Speaker turn detection (Mr. X:, Senator Y:, The CHAIRMAN:)
- Clean GPO-printed text (not OCR'd scans)
- Front matter detection and handling
- Volume/part tracking

Collection slug: mccarthy
"""
import os
import re
import json
import argparse
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from collections import Counter

import psycopg2
import fitz  # PyMuPDF

import ingest_runs


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
COLLECTION_DESCRIPTION = "Executive Sessions of the Senate Permanent Subcommittee on Investigations, Committee on Government Operations. Transcripts from Senator Joseph McCarthy's investigations, made public in 2003."


# =============================================================================
# Chunking Configuration for Hearings
# =============================================================================

@dataclass
class ChunkingConfig:
    """Configuration for hearing transcript chunking."""
    target_chars: int = 4000        # ~1,000 tokens
    max_chars: int = 6000           # ~1,500 tokens  
    overlap_chars: int = 800        # ~200 tokens
    boilerplate_threshold: float = 0.30  # More aggressive for clean GPO docs
    min_chunk_chars: int = 150
    # Hearing-specific
    respect_speaker_turns: bool = True
    max_turns_per_chunk: int = 15   # Don't let chunks get too fragmented


DEFAULT_CONFIG = ChunkingConfig()


# =============================================================================
# Database Connection
# =============================================================================

def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


# =============================================================================
# Speaker Detection for Hearings
# =============================================================================

# Pattern for speaker labels in transcripts
# Matches: Mr. COHN., Senator MCCARTHY., The CHAIRMAN., Mr. FLANAGAN.
SPEAKER_LABEL_RE = re.compile(
    r"^(Mr\.|Mrs\.|Ms\.|Dr\.|Senator|Representative|The)\s+[A-Z][A-Z\s\-']+\.",
    re.MULTILINE
)

# More permissive pattern for inline speaker references
SPEAKER_START_RE = re.compile(
    r"^(Mr\.|Mrs\.|Ms\.|Dr\.|Senator|Representative|The)\s+[A-Z][A-Z\-']+\.\s",
)


def is_speaker_line(line: str) -> bool:
    """Check if a line starts with a speaker label."""
    return bool(SPEAKER_START_RE.match(line.strip()))


def extract_speaker(line: str) -> Optional[str]:
    """Extract speaker name from a line if it starts with a speaker label."""
    m = SPEAKER_START_RE.match(line.strip())
    if m:
        return m.group(0).strip().rstrip('.')
    return None


# =============================================================================
# Text Normalization for Hearings
# =============================================================================

# GPO boilerplate patterns
VERDATE_RE = re.compile(r"VerDate\s+\w+\s+\d+\s+\d+\s+\d+:\d+\s+\w+\s+\d+,\s+\d+.*$", re.MULTILINE)
JKT_RE = re.compile(r"Jkt\s+\d+\s+PO\s+\d+\s+Frm\s+\d+.*$", re.MULTILINE)
PAGE_HEADER_RE = re.compile(r"^\d+\s*$", re.MULTILINE)  # Standalone page numbers


def normalize_hearing_text(raw_text: str) -> str:
    """
    Normalize hearing transcript text.
    GPO documents are cleaner than OCR'd FBI files.
    """
    text = raw_text
    
    # Remove VerDate stamps (GPO formatting artifacts)
    text = VERDATE_RE.sub("", text)
    text = JKT_RE.sub("", text)
    
    # NBSP -> space
    text = text.replace("\u00a0", " ")
    
    # Fix hyphenated line breaks (less common in GPO but still present)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Remove standalone page numbers at start of lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Skip lines that are just numbers (page numbers)
        if re.match(r"^\s*\d+\s*$", line):
            continue
        # Skip lines that are just roman numerals (front matter pagination)
        if re.match(r"^\s*\(?[IVXLCDM]+\)?\s*$", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines)
    
    return text.strip()


# =============================================================================
# Boilerplate Detection
# =============================================================================

def detect_boilerplate(pages_text: List[str], threshold: float = 0.30,
                       top_n: int = 4, bottom_n: int = 4) -> Set[str]:
    """Detect boilerplate lines in GPO documents."""
    line_counts: Counter = Counter()
    total_pages = len(pages_text)
    
    if total_pages < 5:
        return set()
    
    for page_text in pages_text:
        lines = page_text.strip().split("\n")
        top_lines = [l.strip() for l in lines[:top_n] if l.strip()]
        bottom_lines = [l.strip() for l in lines[-bottom_n:] if l.strip()]
        
        page_lines = set(top_lines + bottom_lines)
        for line in page_lines:
            norm = re.sub(r"\s+", " ", line.lower().strip())
            if len(norm) > 5:
                line_counts[norm] += 1
    
    boilerplate = set()
    min_count = int(total_pages * threshold)
    
    for line_norm, count in line_counts.items():
        if count >= min_count:
            boilerplate.add(line_norm)
    
    # GPO-specific boilerplate
    gpo_boilerplate = {
        "verdate",
        "jkt",
        "frm",
        "fmt",
        "sfmt",
    }
    boilerplate.update(gpo_boilerplate)
    
    return boilerplate


def remove_boilerplate(text: str, boilerplate: Set[str],
                       top_n: int = 4, bottom_n: int = 4) -> Tuple[str, bool]:
    """Remove boilerplate lines from text."""
    lines = text.split("\n")
    removed = False
    
    new_lines = []
    for i, line in enumerate(lines):
        norm = re.sub(r"\s+", " ", line.lower().strip())
        
        # Check if line matches boilerplate
        is_bp = norm in boilerplate
        
        # Also check if line contains GPO formatting
        if "verdate" in norm.lower() or "jkt" in norm.lower():
            is_bp = True
        
        if is_bp and (i < top_n or i >= len(lines) - bottom_n):
            removed = True
            continue
        
        new_lines.append(line)
    
    return "\n".join(new_lines), removed


# =============================================================================
# Quality Metrics
# =============================================================================

@dataclass
class TextQualityMetrics:
    """Quality metrics for a text chunk."""
    alpha_ratio: float = 0.0
    digit_ratio: float = 0.0
    speaker_count: int = 0          # Number of speaker turns
    avg_turn_length: float = 0.0    # Average chars per speaker turn
    is_front_matter: bool = False   # Title pages, TOC, etc.


def compute_quality_metrics(text: str) -> TextQualityMetrics:
    """Compute quality metrics for a hearing transcript chunk."""
    if not text:
        return TextQualityMetrics()
    
    total_chars = len(text)
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    
    # Count speaker turns
    speaker_matches = SPEAKER_LABEL_RE.findall(text)
    speaker_count = len(speaker_matches)
    
    # Detect front matter (committee lists, preface, etc.)
    is_front_matter = False
    front_matter_indicators = [
        "committee on", "subcommittee on", "staff director",
        "chief counsel", "preface", "contents", "table of contents",
        "printed for the use of"
    ]
    text_lower = text.lower()
    if any(ind in text_lower for ind in front_matter_indicators):
        # Check if it's predominantly front matter
        indicator_count = sum(1 for ind in front_matter_indicators if ind in text_lower)
        if indicator_count >= 2:
            is_front_matter = True
    
    # Average turn length
    avg_turn_length = total_chars / max(speaker_count, 1)
    
    return TextQualityMetrics(
        alpha_ratio=letters / total_chars if total_chars > 0 else 0.0,
        digit_ratio=digits / total_chars if total_chars > 0 else 0.0,
        speaker_count=speaker_count,
        avg_turn_length=avg_turn_length,
        is_front_matter=is_front_matter,
    )


# =============================================================================
# Chunking for Hearings
# =============================================================================

@dataclass
class PageData:
    """Data for a single page."""
    page_id: int
    pdf_page_number: int
    raw_text: str
    clean_text: str
    boilerplate_removed: bool = False


@dataclass
class BlockInfo:
    """A block with its source page info."""
    text: str
    page_id: int
    pdf_page_number: int
    is_speaker_turn: bool = False
    speaker: Optional[str] = None


@dataclass
class ChunkData:
    """Data for a single chunk."""
    text: str
    clean_text: str
    page_ids: List[int]
    page_start: int
    page_end: int
    chunk_index: int
    char_start: int
    char_end: int
    boilerplate_removed: bool = False
    metrics: TextQualityMetrics = field(default_factory=TextQualityMetrics)


def split_into_speaker_blocks(text: str, page_id: int, pdf_page_number: int) -> List[BlockInfo]:
    """
    Split text into blocks, respecting speaker turns.
    Each speaker turn becomes its own block.
    """
    blocks: List[BlockInfo] = []
    
    lines = text.split("\n")
    current_block_lines: List[str] = []
    current_speaker: Optional[str] = None
    is_speaker_block = False
    
    for line in lines:
        speaker = extract_speaker(line)
        
        if speaker:
            # New speaker - flush current block
            if current_block_lines:
                block_text = "\n".join(current_block_lines).strip()
                if block_text:
                    blocks.append(BlockInfo(
                        text=block_text,
                        page_id=page_id,
                        pdf_page_number=pdf_page_number,
                        is_speaker_turn=is_speaker_block,
                        speaker=current_speaker,
                    ))
            
            # Start new speaker block
            current_block_lines = [line]
            current_speaker = speaker
            is_speaker_block = True
        else:
            # Continue current block
            current_block_lines.append(line)
    
    # Flush final block
    if current_block_lines:
        block_text = "\n".join(current_block_lines).strip()
        if block_text:
            blocks.append(BlockInfo(
                text=block_text,
                page_id=page_id,
                pdf_page_number=pdf_page_number,
                is_speaker_turn=is_speaker_block,
                speaker=current_speaker,
            ))
    
    return blocks


def create_chunks_from_pages(
    pages: List[PageData],
    config: ChunkingConfig,
) -> List[ChunkData]:
    """
    Chunking algorithm for hearing transcripts.
    
    Key differences from FBI files:
    - Speaker turns are respected as boundaries
    - Cleaner text means fewer artifacts
    - Front matter is detected and handled
    """
    chunks: List[ChunkData] = []
    
    # Accumulator
    current_blocks: List[BlockInfo] = []
    current_char_count = 0
    current_turn_count = 0
    char_offset = 0
    
    # Overlap tracking
    prev_overlap_blocks: List[BlockInfo] = []
    
    def flush_chunk(is_final: bool = False):
        nonlocal current_blocks, current_char_count, current_turn_count
        nonlocal char_offset, prev_overlap_blocks
        
        if not current_blocks:
            return
        
        chunk_text = "\n\n".join(b.text for b in current_blocks)
        
        if len(chunk_text) < config.min_chunk_chars and not is_final:
            return
        
        metrics = compute_quality_metrics(chunk_text)
        metrics.speaker_count = sum(1 for b in current_blocks if b.is_speaker_turn)
        
        page_ids = list(dict.fromkeys(b.page_id for b in current_blocks))
        page_numbers = [b.pdf_page_number for b in current_blocks]
        
        page_id_set = set(page_ids)
        boilerplate_removed = any(
            p.boilerplate_removed for p in pages
            if p.page_id in page_id_set
        )
        
        chunk = ChunkData(
            text=chunk_text,
            clean_text=chunk_text,
            page_ids=page_ids,
            page_start=min(page_numbers) if page_numbers else 0,
            page_end=max(page_numbers) if page_numbers else 0,
            chunk_index=len(chunks),
            char_start=char_offset,
            char_end=char_offset + len(chunk_text),
            boilerplate_removed=boilerplate_removed,
            metrics=metrics,
        )
        chunks.append(chunk)
        
        char_offset += len(chunk_text) + 2
        
        # Compute overlap
        overlap_blocks: List[BlockInfo] = []
        overlap_chars = 0
        for i in range(len(current_blocks) - 1, -1, -1):
            block = current_blocks[i]
            if overlap_chars + len(block.text) > config.overlap_chars and overlap_blocks:
                break
            overlap_blocks.insert(0, block)
            overlap_chars += len(block.text)
        
        prev_overlap_blocks = overlap_blocks
        
        current_blocks = []
        current_char_count = 0
        current_turn_count = 0
    
    # Process pages
    for page in pages:
        if config.respect_speaker_turns:
            blocks = split_into_speaker_blocks(page.clean_text, page.page_id, page.pdf_page_number)
        else:
            # Fallback to paragraph splitting
            paras = re.split(r"\n\n+", page.clean_text)
            blocks = [
                BlockInfo(text=p.strip(), page_id=page.page_id, pdf_page_number=page.pdf_page_number)
                for p in paras if p.strip()
            ]
        
        for block in blocks:
            # Check if this is a new speaker turn that should trigger a boundary
            if (config.respect_speaker_turns and 
                block.is_speaker_turn and 
                current_blocks and
                current_turn_count >= config.max_turns_per_chunk):
                flush_chunk()
                current_blocks = [BlockInfo(b.text, b.page_id, b.pdf_page_number, b.is_speaker_turn, b.speaker) 
                                  for b in prev_overlap_blocks]
                current_char_count = sum(len(b.text) for b in current_blocks)
                current_turn_count = sum(1 for b in current_blocks if b.is_speaker_turn)
            
            # Check size limit
            potential_size = current_char_count + len(block.text) + 2
            
            if potential_size > config.target_chars and current_blocks:
                flush_chunk()
                current_blocks = [BlockInfo(b.text, b.page_id, b.pdf_page_number, b.is_speaker_turn, b.speaker) 
                                  for b in prev_overlap_blocks]
                current_char_count = sum(len(b.text) for b in current_blocks)
                current_turn_count = sum(1 for b in current_blocks if b.is_speaker_turn)
            
            # Add block
            current_blocks.append(block)
            current_char_count += len(block.text) + 2
            if block.is_speaker_turn:
                current_turn_count += 1
    
    flush_chunk(is_final=True)
    
    return chunks


# =============================================================================
# PDF Extraction
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_page_text(doc: fitz.Document, page_index: int) -> str:
    """Extract text from a PDF page."""
    page = doc.load_page(page_index)
    txt = page.get_text("text") or ""
    if txt.strip():
        return txt
    
    # Fallback methods
    blocks = page.get_text("blocks") or []
    parts = [b[4] for b in blocks if len(b) >= 5 and isinstance(b[4], str)]
    txt2 = "\n".join(parts)
    if txt2.strip():
        return txt2
    
    return ""


# =============================================================================
# Database Operations
# =============================================================================

def get_or_create_collection(cur) -> int:
    cur.execute("SELECT id FROM collections WHERE slug = %s", (COLLECTION_SLUG,))
    r = cur.fetchone()
    if r:
        return int(r[0])
    
    try:
        cur.execute(
            """
            INSERT INTO collections (slug, title, description)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (COLLECTION_SLUG, COLLECTION_TITLE, COLLECTION_DESCRIPTION),
        )
    except psycopg2.errors.UndefinedColumn:
        cur.connection.rollback()
        cur.execute(
            """
            INSERT INTO collections (slug, title)
            VALUES (%s, %s)
            RETURNING id
            """,
            (COLLECTION_SLUG, COLLECTION_TITLE),
        )
    
    return int(cur.fetchone()[0])


def upsert_document(cur, collection_id: int, source_name: str, source_ref: str,
                    volume: str, metadata: dict) -> int:
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


def insert_chunk_pages(cur, chunk_id: int, page_ids: List[int]):
    for i, page_id in enumerate(page_ids, start=1):
        cur.execute(
            """
            INSERT INTO chunk_pages (chunk_id, page_id, span_order)
            VALUES (%s, %s, %s)
            """,
            (chunk_id, page_id, i),
        )


# =============================================================================
# Main Processing
# =============================================================================

def process_pdf(
    pdf_path: Path,
    cur,
    collection_id: int,
    config: ChunkingConfig,
    pipeline_version: str,
    compute_sha: bool = True,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """Process a single McCarthy hearing volume."""
    source_name = pdf_path.name
    
    # Extract volume number
    volume_match = re.search(r"Vol(\d+)", source_name)
    volume = f"Volume {volume_match.group(1)}" if volume_match else ""
    
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    
    print(f"  Processing {source_name}: {page_count} pages")
    
    # Extract raw text from all pages
    raw_pages: List[Tuple[int, str]] = []
    for i in range(page_count):
        txt = extract_page_text(doc, i)
        raw_pages.append((i + 1, txt))
    
    doc.close()
    
    # Detect boilerplate
    all_raw_texts = [t for _, t in raw_pages]
    boilerplate = detect_boilerplate(all_raw_texts, threshold=config.boilerplate_threshold)
    print(f"    Detected {len(boilerplate)} boilerplate patterns")
    
    if dry_run:
        print(f"    [DRY RUN] Would process {page_count} pages")
        return 0, page_count, 0
    
    # Build metadata
    meta = {
        "source_format": "pdf_gpo",
        "extractor": "pymupdf",
        "page_count": page_count,
        "boilerplate_patterns": len(boilerplate),
        "document_type": "hearing_transcript",
    }
    if compute_sha:
        meta["sha256"] = sha256_file(pdf_path)
    
    doc_id = upsert_document(
        cur,
        collection_id=collection_id,
        source_name=source_name,
        source_ref=str(pdf_path),
        volume=volume,
        metadata=meta,
    )
    
    # Clear existing data
    delete_chunks_for_document(cur, doc_id)
    delete_pages_for_document(cur, doc_id)
    
    # Insert pages and build PageData list
    page_data_list: List[PageData] = []
    
    for pdf_page_num, raw_text in raw_pages:
        page_seq = pdf_page_num
        logical_label = f"p{pdf_page_num:04d}"
        
        page_id = insert_page(cur, doc_id, page_seq, pdf_page_num, logical_label, raw_text)
        
        # Normalize for chunking
        clean_text = normalize_hearing_text(raw_text)
        clean_text, bp_removed = remove_boilerplate(clean_text, boilerplate)
        
        page_data_list.append(PageData(
            page_id=page_id,
            pdf_page_number=pdf_page_num,
            raw_text=raw_text,
            clean_text=clean_text,
            boilerplate_removed=bp_removed,
        ))
    
    # Create chunks
    chunks = create_chunks_from_pages(page_data_list, config)
    print(f"    Created {len(chunks)} chunks")
    
    # Count speaker turns
    total_turns = sum(c.metrics.speaker_count for c in chunks)
    print(f"    Total speaker turns detected: {total_turns}")
    
    # Insert chunks
    for chunk in chunks:
        chunk_id = insert_chunk(cur, chunk.text, chunk.clean_text, pipeline_version)
        insert_chunk_pages(cur, chunk_id, chunk.page_ids)
    
    return doc_id, page_count, len(chunks)


def main():
    ap = argparse.ArgumentParser(description="Ingest McCarthy Hearings PDFs")
    ap.add_argument("--input-dir", default="data/raw/mccarthy", help="Directory containing PDFs")
    ap.add_argument("--glob", default="*.pdf", help="File glob pattern")
    ap.add_argument("--pipeline-version", default="mccarthy_v1", help="Chunking pipeline version")
    ap.add_argument("--target-chars", type=int, default=4000, help="Target chunk size")
    ap.add_argument("--max-chars", type=int, default=6000, help="Max chunk size")
    ap.add_argument("--overlap-chars", type=int, default=800, help="Overlap between chunks")
    ap.add_argument("--max-turns", type=int, default=15, help="Max speaker turns per chunk")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of PDFs")
    ap.add_argument("--no-sha", action="store_true", help="Skip SHA256 computation")
    ap.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = ap.parse_args()
    
    config = ChunkingConfig(
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        max_turns_per_chunk=args.max_turns,
    )
    
    import glob as glob_mod
    paths = sorted(glob_mod.glob(str(Path(args.input_dir) / args.glob)))
    if args.limit:
        paths = paths[:args.limit]
    
    if not paths:
        print(f"No PDFs found at {args.input_dir}/{args.glob}")
        return
    
    print(f"Found {len(paths)} PDF files to process")
    print(f"Config: target={config.target_chars}, max={config.max_chars}, overlap={config.overlap_chars}")
    print(f"        max_turns_per_chunk={config.max_turns_per_chunk}")
    print()
    
    total_pages = 0
    total_chunks = 0
    
    with connect() as conn, conn.cursor() as cur:
        ingest_runs.ensure_ingest_runs_table(cur)
        collection_id = get_or_create_collection(cur)
        print(f"Collection: {COLLECTION_SLUG} (id={collection_id})")
        print()
        
        for pdf_path in paths:
            p = Path(pdf_path)
            pipeline_version = str(args.pipeline_version)
            source_key = f"{COLLECTION_SLUG}:{p.name}"
            fp = ingest_runs.file_fingerprint_fast(p) if args.no_sha else ingest_runs.file_sha256(p)

            if not args.dry_run and not ingest_runs.should_run(
                cur, source_key=source_key, source_fingerprint=fp, pipeline_version=pipeline_version
            ):
                print(f"[skip] {p.name} (already ingested: pipeline={pipeline_version})")
                continue

            try:
                if not args.dry_run:
                    ingest_runs.mark_running(
                        cur, source_key=source_key, source_fingerprint=fp, pipeline_version=pipeline_version
                    )

                doc_id, pages, chunks = process_pdf(
                    p,
                    cur,
                    collection_id,
                    config,
                    pipeline_version,
                    compute_sha=not args.no_sha,
                    dry_run=args.dry_run,
                )
                total_pages += pages
                total_chunks += chunks
                
                if not args.dry_run:
                    ingest_runs.mark_success(cur, source_key=source_key)
                    conn.commit()
                    print(f"    -> document_id={doc_id}")
            except Exception as e:
                print(f"    ERROR: {p.name}: {e}")
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
        print(f"  Total pages: {total_pages}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Pipeline version: {args.pipeline_version}")


if __name__ == "__main__":
    main()
