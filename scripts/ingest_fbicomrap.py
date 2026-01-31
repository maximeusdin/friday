#!/usr/bin/env python3
"""
Ingest script for FBI COMRAP (Comintern Apparatus) files.

This script:
1. Extracts text from PDFs using PyMuPDF
2. Creates collection, documents, and pages in the database
3. Applies the chunking algorithm with OCR normalization

Collection slug: fbicomrap
"""
import os
import re
import json
import argparse
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
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

COLLECTION_SLUG = "fbicomrap"
COLLECTION_TITLE = "FBI COMRAP (Comintern Apparatus) Files"
COLLECTION_DESCRIPTION = "FBI investigation files on the Comintern Apparatus, covering Communist International activities in the United States."


# =============================================================================
# Chunking Configuration (per user strategy)
# =============================================================================

@dataclass
class ChunkingConfig:
    """Per-collection chunking configuration knobs."""
    target_chars: int = 5000        # ~1,000-1,500 tokens
    max_chars: int = 8000           # ~2,000 tokens
    overlap_chars: int = 1000       # ~150-250 tokens
    boilerplate_threshold: float = 0.35  # Line must appear on 35%+ pages to be boilerplate
    split_on_double_newline: bool = True
    keep_headers: bool = False      # FBI files: headers usually boilerplate
    min_chunk_chars: int = 100      # Don't create tiny chunks


# Default config for FBI files
DEFAULT_CONFIG = ChunkingConfig()


# =============================================================================
# Database Connection
# =============================================================================

def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


# =============================================================================
# Text Normalization (Step 1 from user strategy)
# =============================================================================

def normalize_ocr_text(raw_text: str) -> str:
    """
    Light-touch, lossless-ish OCR normalization.
    
    Steps (deterministic):
    1. Fix hyphenated line breaks: word-\nword -> wordword
    2. Join wrapped lines within paragraph (non-terminal + lowercase next)
    3. Collapse >2 consecutive newlines to 2
    4. NBSP -> space
    """
    text = raw_text
    
    # NBSP -> space
    text = text.replace("\u00a0", " ")
    
    # Fix hyphenated line breaks: only when both sides are letters
    # e.g., "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    # Join wrapped lines within a paragraph:
    # If line doesn't end in .?!:;") and next line starts lowercase, join with space
    lines = text.split("\n")
    joined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if we should join with next line
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            line_stripped = line.rstrip()
            # Join if: current line doesn't end with terminal punctuation
            # AND next line starts with lowercase
            if (line_stripped and 
                not re.search(r'[.?!:;"\'\)]$', line_stripped) and
                next_line and next_line[0].islower()):
                # Join with space
                joined_lines.append(line.rstrip() + " " + next_line)
                i += 2
                continue
        joined_lines.append(line)
        i += 1
    
    text = "\n".join(joined_lines)
    
    # Collapse >2 consecutive newlines to 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


# =============================================================================
# Boilerplate Detection (Step 2 from user strategy)
# =============================================================================

def detect_boilerplate(pages_text: List[str], threshold: float = 0.35, 
                       top_n: int = 3, bottom_n: int = 3) -> Set[str]:
    """
    Detect boilerplate lines that appear on many pages.
    
    Looks at top_n and bottom_n lines of each page.
    If a line appears on >= threshold of pages, mark as boilerplate.
    
    Returns set of normalized boilerplate lines.
    """
    line_counts: Counter = Counter()
    total_pages = len(pages_text)
    
    if total_pages < 3:
        return set()
    
    for page_text in pages_text:
        lines = page_text.strip().split("\n")
        
        # Get top and bottom lines
        top_lines = [l.strip() for l in lines[:top_n] if l.strip()]
        bottom_lines = [l.strip() for l in lines[-bottom_n:] if l.strip()]
        
        # Count unique lines per page (don't double count same line on same page)
        page_lines = set(top_lines + bottom_lines)
        for line in page_lines:
            # Normalize for comparison: lowercase, collapse whitespace
            norm = re.sub(r"\s+", " ", line.lower().strip())
            if len(norm) > 3:  # Ignore very short lines
                line_counts[norm] += 1
    
    # Find lines that appear on >= threshold of pages
    boilerplate = set()
    min_count = int(total_pages * threshold)
    
    for line_norm, count in line_counts.items():
        if count >= min_count:
            boilerplate.add(line_norm)
    
    # Also add common FBI boilerplate patterns
    fbi_boilerplate = {
        "federal bureau of investigation",
        "freedom of information/privacy acts section",
        "foipa",
        "confidential",
        "secret",
        "top secret",
        "declassified",
        "unclassified",
    }
    boilerplate.update(fbi_boilerplate)
    
    return boilerplate


def remove_boilerplate(text: str, boilerplate: Set[str], 
                       top_n: int = 3, bottom_n: int = 3) -> Tuple[str, bool]:
    """
    Remove boilerplate lines from text.
    Only removes from top and bottom of page.
    
    Returns (cleaned_text, boilerplate_removed_flag)
    """
    lines = text.split("\n")
    removed = False
    
    # Process top lines
    new_top = []
    for i, line in enumerate(lines[:top_n]):
        norm = re.sub(r"\s+", " ", line.lower().strip())
        if norm not in boilerplate:
            new_top.append(line)
        else:
            removed = True
    
    # Keep middle lines unchanged
    middle = lines[top_n:-bottom_n] if len(lines) > top_n + bottom_n else []
    
    # Process bottom lines
    new_bottom = []
    for line in lines[-bottom_n:] if len(lines) > bottom_n else []:
        norm = re.sub(r"\s+", " ", line.lower().strip())
        if norm not in boilerplate:
            new_bottom.append(line)
        else:
            removed = True
    
    result_lines = new_top + middle + new_bottom
    return "\n".join(result_lines), removed


# =============================================================================
# Quality Metrics (Step 5 from user strategy)
# =============================================================================

@dataclass
class TextQualityMetrics:
    """Quality metrics for a text chunk."""
    alpha_ratio: float = 0.0        # letters / total chars
    digit_ratio: float = 0.0        # digits / total chars  
    redaction_ratio: float = 0.0    # redaction chars / total chars
    garbage_ratio: float = 0.0      # non-ascii or repeated punctuation
    contains_table_like: bool = False
    contains_list_like: bool = False
    low_signal: bool = False        # True if redaction_ratio > 0.35


# Redaction patterns
REDACTION_PATTERNS = [
    r"█+",                    # Unicode block chars
    r"▓+",                    # Other block chars
    r"\[REDACTED\]",          # Explicit redaction
    r"\[DELETED\]",
    r"X{4,}",                 # XXXXX patterns
    r"_{10,}",                # Long underscores (often redaction)
    r"\*{4,}",                # Asterisk runs
]
REDACTION_RE = re.compile("|".join(REDACTION_PATTERNS), re.IGNORECASE)


def compute_quality_metrics(text: str) -> TextQualityMetrics:
    """Compute quality metrics for a text chunk."""
    if not text:
        return TextQualityMetrics()
    
    total_chars = len(text)
    
    # Count character types
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    
    # Find redactions
    redaction_matches = REDACTION_RE.findall(text)
    redaction_chars = sum(len(m) for m in redaction_matches)
    
    # Garbage: non-ASCII (excluding common chars) or repeated punctuation
    garbage_chars = 0
    for c in text:
        # Non-ASCII that's not common
        if ord(c) > 127 and c not in "àáâãäåæçèéêëìíîïñòóôõöùúûüý":
            garbage_chars += 1
    # Repeated punctuation (OCR artifacts)
    repeated_punct = re.findall(r"[^\w\s]{3,}", text)
    garbage_chars += sum(len(m) for m in repeated_punct)
    
    # Table-like: lines with many spaces or | or tabs
    lines = text.split("\n")
    table_like_lines = sum(1 for l in lines if l.count("  ") > 3 or "|" in l or "\t" in l)
    contains_table_like = table_like_lines > len(lines) * 0.3
    
    # List-like: lines starting with -, •, numbers
    list_like_lines = sum(1 for l in lines if re.match(r"^\s*[-•*]\s+", l) or re.match(r"^\s*\d+[.)]\s+", l))
    contains_list_like = list_like_lines > 3
    
    alpha_ratio = letters / total_chars if total_chars > 0 else 0.0
    digit_ratio = digits / total_chars if total_chars > 0 else 0.0
    redaction_ratio = redaction_chars / total_chars if total_chars > 0 else 0.0
    garbage_ratio = garbage_chars / total_chars if total_chars > 0 else 0.0
    
    low_signal = redaction_ratio > 0.35
    
    return TextQualityMetrics(
        alpha_ratio=alpha_ratio,
        digit_ratio=digit_ratio,
        redaction_ratio=redaction_ratio,
        garbage_ratio=garbage_ratio,
        contains_table_like=contains_table_like,
        contains_list_like=contains_list_like,
        low_signal=low_signal,
    )


# =============================================================================
# Hearing/Transcript Detection (Step 8 from user strategy)
# =============================================================================

SPEAKER_LABEL_RE = re.compile(r"^[A-Z][A-Z .'-]{2,30}:")


def is_speaker_label(line: str) -> bool:
    """Check if line starts with a speaker label like 'THE CHAIRMAN:' or 'Mr. X:'"""
    return bool(SPEAKER_LABEL_RE.match(line.strip()))


# =============================================================================
# Chunking Algorithm (Steps 3-4 from user strategy)
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
class ChunkData:
    """Data for a single chunk."""
    text: str                       # raw-ish text
    clean_text: str                 # cleaned for retrieval
    page_ids: List[int]             # page IDs in order
    page_start: int                 # first PDF page number
    page_end: int                   # last PDF page number
    chunk_index: int                # 0-based index
    char_start: int                 # start offset in concatenated clean text
    char_end: int                   # end offset
    boilerplate_removed: bool = False
    metrics: TextQualityMetrics = field(default_factory=TextQualityMetrics)


def split_into_blocks(text: str, split_on_double_newline: bool = True) -> List[str]:
    """
    Split text into blocks (paragraph-ish units).
    """
    if split_on_double_newline:
        # Split on double newline
        blocks = re.split(r"\n\n+", text)
    else:
        # Split on single newline (for very broken OCR)
        blocks = text.split("\n")
    
    # Filter empty blocks
    return [b.strip() for b in blocks if b.strip()]


def split_large_block(block: str, max_chars: int) -> List[str]:
    """
    Split a large block into smaller pieces using sentence-ish splitting.
    Fallback for huge OCR spew.
    """
    if len(block) <= max_chars:
        return [block]
    
    # Try to split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", block)
    
    result = []
    current = ""
    
    for sent in sentences:
        if not current:
            current = sent
        elif len(current) + len(sent) + 1 <= max_chars:
            current += " " + sent
        else:
            if current:
                result.append(current)
            current = sent
    
    if current:
        result.append(current)
    
    # If still too large, force split
    final = []
    for chunk in result:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            # Force split at max_chars boundaries (last resort)
            for i in range(0, len(chunk), max_chars):
                final.append(chunk[i:i+max_chars])
    
    return final


@dataclass
class BlockInfo:
    """A block with its source page info."""
    text: str
    page_id: int
    pdf_page_number: int


def create_chunks_from_pages(
    pages: List[PageData],
    config: ChunkingConfig,
) -> List[ChunkData]:
    """
    Main chunking algorithm following user strategy:
    
    1. Page-first, then paragraph blocks
    2. Accumulate blocks until size limit
    3. Handle speaker turns as boundaries (for transcripts)
    4. Track page provenance
    5. Apply overlap
    """
    chunks: List[ChunkData] = []
    
    # Current chunk accumulator - store (block_text, page_id, pdf_page_number) together
    current_blocks: List[BlockInfo] = []
    current_char_count = 0
    char_offset = 0  # Running offset in concatenated text
    
    # Previous chunk's last blocks for overlap
    prev_overlap_blocks: List[BlockInfo] = []
    
    def flush_chunk(is_final: bool = False):
        nonlocal current_blocks, current_char_count, char_offset, prev_overlap_blocks
        
        if not current_blocks:
            return
        
        # Build chunk text
        chunk_text = "\n\n".join(b.text for b in current_blocks)
        
        # Skip if too small (unless final)
        if len(chunk_text) < config.min_chunk_chars and not is_final:
            return
        
        # Compute metrics
        metrics = compute_quality_metrics(chunk_text)
        
        # Get unique page_ids preserving order
        page_ids = list(dict.fromkeys(b.page_id for b in current_blocks))
        page_numbers = [b.pdf_page_number for b in current_blocks]
        
        # Determine if any boilerplate was removed
        page_id_set = set(page_ids)
        boilerplate_removed = any(
            p.boilerplate_removed for p in pages 
            if p.page_id in page_id_set
        )
        
        chunk = ChunkData(
            text=chunk_text,
            clean_text=chunk_text,  # Same for now
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
        
        char_offset += len(chunk_text) + 2  # +2 for paragraph separator
        
        # Compute overlap for next chunk
        # Keep last ~overlap_chars worth of blocks
        overlap_target = config.overlap_chars
        overlap_blocks: List[BlockInfo] = []
        overlap_chars = 0
        
        for i in range(len(current_blocks) - 1, -1, -1):
            block = current_blocks[i]
            if overlap_chars + len(block.text) > overlap_target and overlap_blocks:
                break
            overlap_blocks.insert(0, block)
            overlap_chars += len(block.text)
        
        prev_overlap_blocks = overlap_blocks
        
        # Reset accumulators
        current_blocks = []
        current_char_count = 0
    
    # Process pages
    for page in pages:
        blocks = split_into_blocks(page.clean_text, config.split_on_double_newline)
        
        for block_text in blocks:
            block = BlockInfo(text=block_text, page_id=page.page_id, pdf_page_number=page.pdf_page_number)
            
            # Check for speaker label (hard boundary for transcripts)
            if is_speaker_label(block_text) and current_blocks:
                # Flush current chunk before starting new speaker
                flush_chunk()
                # Start new chunk with overlap
                current_blocks = [BlockInfo(b.text, b.page_id, b.pdf_page_number) for b in prev_overlap_blocks]
                current_char_count = sum(len(b.text) for b in current_blocks)
            
            # Handle large blocks
            if len(block_text) > config.max_chars:
                # Flush current, then split large block
                if current_blocks:
                    flush_chunk()
                    current_blocks = [BlockInfo(b.text, b.page_id, b.pdf_page_number) for b in prev_overlap_blocks]
                    current_char_count = sum(len(b.text) for b in current_blocks)
                
                sub_blocks = split_large_block(block_text, config.max_chars)
                for sub in sub_blocks:
                    current_blocks.append(BlockInfo(text=sub, page_id=page.page_id, pdf_page_number=page.pdf_page_number))
                    current_char_count += len(sub)
                    
                    if current_char_count >= config.target_chars:
                        flush_chunk()
                        current_blocks = [BlockInfo(b.text, b.page_id, b.pdf_page_number) for b in prev_overlap_blocks]
                        current_char_count = sum(len(b.text) for b in current_blocks)
                continue
            
            # Check if adding this block would exceed target
            potential_size = current_char_count + len(block_text) + 2  # +2 for \n\n
            
            if potential_size > config.target_chars and current_blocks:
                # Flush current chunk
                flush_chunk()
                # Start new chunk with overlap
                current_blocks = [BlockInfo(b.text, b.page_id, b.pdf_page_number) for b in prev_overlap_blocks]
                current_char_count = sum(len(b.text) for b in current_blocks)
            
            # Add block to current chunk
            current_blocks.append(block)
            current_char_count += len(block_text) + 2
    
    # Flush final chunk
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
    """Extract text from a PDF page using multiple methods."""
    page = doc.load_page(page_index)
    
    # Method 1: Default text extractor
    txt = page.get_text("text") or ""
    if txt.strip():
        return txt
    
    # Method 2: Blocks extractor
    blocks = page.get_text("blocks") or []
    parts = []
    for b in blocks:
        if len(b) >= 5 and isinstance(b[4], str):
            parts.append(b[4])
    txt2 = "\n".join(parts)
    if txt2.strip():
        return txt2
    
    # Method 3: Dict/rawdict extractor
    d = page.get_text("rawdict")
    span_texts = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "")
                if t:
                    span_texts.append(t)
    txt3 = "\n".join(span_texts)
    if txt3.strip():
        return txt3
    
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


def insert_chunk_metadata(cur, chunk_id: int, metadata: dict):
    """Insert chunk metadata if the table exists.
    
    NOTE: This is optional and may fail silently if the table doesn't exist.
    We use SAVEPOINT to avoid rolling back the main transaction.
    """
    try:
        # Use savepoint to avoid rolling back entire transaction on failure
        cur.execute("SAVEPOINT chunk_meta_insert")
        cur.execute(
            """
            INSERT INTO chunk_metadata (chunk_id, document_id, chunk_index, 
                                       page_start, page_end, char_start, char_end,
                                       boilerplate_removed, redaction_ratio, 
                                       alpha_ratio, low_signal,
                                       contains_table_like, contains_list_like)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                chunk_index = EXCLUDED.chunk_index,
                page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end
            """,
            (
                chunk_id,
                metadata.get("document_id"),
                metadata.get("chunk_index"),
                metadata.get("page_start"),
                metadata.get("page_end"),
                metadata.get("char_start"),
                metadata.get("char_end"),
                metadata.get("boilerplate_removed", False),
                metadata.get("redaction_ratio", 0.0),
                metadata.get("alpha_ratio", 0.0),
                metadata.get("low_signal", False),
                metadata.get("contains_table_like", False),
                metadata.get("contains_list_like", False),
            ),
        )
        cur.execute("RELEASE SAVEPOINT chunk_meta_insert")
    except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn):
        # Table or column doesn't exist, rollback just this operation
        cur.execute("ROLLBACK TO SAVEPOINT chunk_meta_insert")
    except Exception:
        # Any other error, rollback just this operation
        try:
            cur.execute("ROLLBACK TO SAVEPOINT chunk_meta_insert")
        except Exception:
            pass


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
    """
    Process a single PDF file.
    
    Returns (document_id, page_count, chunk_count)
    """
    source_name = pdf_path.name
    
    # Determine volume from filename
    volume = ""
    if "Report_1" in source_name:
        volume = "Report 1"
    elif "Report_2" in source_name:
        volume = "Report 2"
    elif "Summary" in source_name:
        volume = "Summary"
    
    # Open PDF
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    
    print(f"  Processing {source_name}: {page_count} pages")
    
    # Extract raw text from all pages
    raw_pages: List[Tuple[int, str]] = []  # (pdf_page_number, raw_text)
    for i in range(page_count):
        txt = extract_page_text(doc, i)
        raw_pages.append((i + 1, txt))
    
    doc.close()
    
    # Detect boilerplate across all pages
    all_raw_texts = [t for _, t in raw_pages]
    boilerplate = detect_boilerplate(all_raw_texts, threshold=config.boilerplate_threshold)
    print(f"    Detected {len(boilerplate)} boilerplate patterns")
    
    if dry_run:
        # Just show stats
        print(f"    [DRY RUN] Would process {page_count} pages")
        return 0, page_count, 0
    
    # Build metadata
    meta = {
        "source_format": "pdf_embedded_text",
        "extractor": "pymupdf",
        "page_count": page_count,
        "boilerplate_patterns": len(boilerplate),
    }
    if compute_sha:
        meta["sha256"] = sha256_file(pdf_path)
    
    # Upsert document
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
        
        # Insert raw page
        page_id = insert_page(cur, doc_id, page_seq, pdf_page_num, logical_label, raw_text)
        
        # Normalize and remove boilerplate for chunking
        clean_text = normalize_ocr_text(raw_text)
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
    
    # Insert chunks
    for chunk in chunks:
        chunk_id = insert_chunk(cur, chunk.text, chunk.clean_text, pipeline_version)
        insert_chunk_pages(cur, chunk_id, chunk.page_ids)
        
        # Try to insert metadata
        insert_chunk_metadata(cur, chunk_id, {
            "document_id": doc_id,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "boilerplate_removed": chunk.boilerplate_removed,
            "redaction_ratio": chunk.metrics.redaction_ratio,
            "alpha_ratio": chunk.metrics.alpha_ratio,
            "low_signal": chunk.metrics.low_signal,
            "contains_table_like": chunk.metrics.contains_table_like,
            "contains_list_like": chunk.metrics.contains_list_like,
        })
    
    return doc_id, page_count, len(chunks)


def main():
    ap = argparse.ArgumentParser(description="Ingest FBI COMRAP PDFs with chunking")
    ap.add_argument("--input-dir", default="data/raw/FBICOMRAP", help="Directory containing PDFs")
    ap.add_argument("--glob", default="*.pdf", help="File glob pattern")
    ap.add_argument("--pipeline-version", default="fbicomrap_v1", help="Chunking pipeline version")
    ap.add_argument("--target-chars", type=int, default=5000, help="Target chunk size in chars")
    ap.add_argument("--max-chars", type=int, default=8000, help="Max chunk size in chars")
    ap.add_argument("--overlap-chars", type=int, default=1000, help="Overlap between chunks")
    ap.add_argument("--boilerplate-threshold", type=float, default=0.35, help="Boilerplate detection threshold")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process")
    ap.add_argument("--no-sha", action="store_true", help="Skip SHA256 computation")
    ap.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = ap.parse_args()
    
    # Build config
    config = ChunkingConfig(
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        boilerplate_threshold=args.boilerplate_threshold,
    )
    
    # Find PDFs
    import glob as glob_mod
    paths = sorted(glob_mod.glob(str(Path(args.input_dir) / args.glob)))
    if args.limit:
        paths = paths[:args.limit]
    
    if not paths:
        print(f"No PDFs found at {args.input_dir}/{args.glob}")
        return
    
    print(f"Found {len(paths)} PDF files to process")
    print(f"Config: target={config.target_chars}, max={config.max_chars}, overlap={config.overlap_chars}")
    print()
    
    total_pages = 0
    total_chunks = 0
    
    with connect() as conn, conn.cursor() as cur:
        collection_id = get_or_create_collection(cur)
        print(f"Collection: {COLLECTION_SLUG} (id={collection_id})")
        print()
        
        for pdf_path in paths:
            doc_id, pages, chunks = process_pdf(
                Path(pdf_path),
                cur,
                collection_id,
                config,
                args.pipeline_version,
                compute_sha=not args.no_sha,
                dry_run=args.dry_run,
            )
            total_pages += pages
            total_chunks += chunks
            
            if not args.dry_run:
                conn.commit()
                print(f"    -> document_id={doc_id}")
        
        print()
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Done!")
        print(f"  Total pages: {total_pages}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Pipeline version: {args.pipeline_version}")


if __name__ == "__main__":
    main()
