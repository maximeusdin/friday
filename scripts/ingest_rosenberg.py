#!/usr/bin/env python3
"""
Ingest script for Julius Rosenberg FBI case files.

These are scanned FBI documents with:
- Variable OCR quality
- Multiple document types within each PDF (memos, reports, teletypes, letters)
- Standard FBI header structure (TO/FROM/RE/DATE)
- File/serial number cross-references

Key strategy differences from McCarthy (transcript) ingest:
- Document-boundary aware chunking (not speaker-turn based)
- Metadata extraction from FBI headers
- OCR cleaning for common FBI document artifacts
- Skip low-value pages (file indexes, cover pages, blanks)

Collection slug: rosenberg
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

import ingest_runs


# =============================================================================
# Configuration
# =============================================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neh")
DB_USER = os.getenv("DB_USER", "neh")
DB_PASS = os.getenv("DB_PASS", "neh")

COLLECTION_SLUG = "rosenberg"
COLLECTION_TITLE = "Julius Rosenberg FBI Case Files"
COLLECTION_DESCRIPTION = "FBI investigation files on Julius Rosenberg, covering espionage investigation 1950-1953. Includes memos, reports, teletypes, and correspondence."


# =============================================================================
# Chunking Configuration
# =============================================================================

@dataclass
class ChunkingConfig:
    """Per-collection chunking configuration."""
    target_chars: int = 4000        # ~1,000 tokens
    max_chars: int = 6000           # ~1,500 tokens
    overlap_chars: int = 800        # ~200 tokens
    boilerplate_threshold: float = 0.30
    min_chunk_chars: int = 150
    min_page_chars: int = 50        # Pages below this are considered blank
    skip_index_pages: bool = True   # Skip file index/inventory pages
    skip_cover_pages: bool = True   # Skip FOIA notice pages


DEFAULT_CONFIG = ChunkingConfig()


# =============================================================================
# Database Connection
# =============================================================================

def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


# =============================================================================
# Page Classification
# =============================================================================

# Patterns for pages to skip
SKIP_PATTERNS = {
    'cover_page': [
        r'FREEDOM\s+OF\s+INFORMATION',
        r'PRIVACY\s+ACTS?',
        r'FEDERAL\s+BUREAU\s+OF\s+INVESTIGATION\s*$',  # Standalone title
        r'NOTICE.*BEST\s+COPIES\s+OBTAINABLE',
        r'PAGES\s+INCLUDED\s+THAT\s+ARE\s+BLURRED',
    ],
    'file_index': [
        r'FILE\s+DESCRIPTION\s+NEW\s+YORK\s+FILE',
        r'SERIAL\s+DATE\s+DESCRIPTION',
        r'History\s+Worksheet',
        r'Designated\s+to\s+or\s+from\s+Bureau',
        r'No\.\s+of\s+Pages\s+Actual\s+Released',
        r'DOCUMENT\s+JUSTIFICATION',
        r'Inventory\s+Worksheet',
        r'Serial\s+Number.*Date\s+of\s+Serial',
        r'FILE\s+NO\.\s*VOLUME\s*NO\.\s*SERIALS',
        r'DELETION\s*\(S\)',
    ],
}

# Patterns for document type detection
DOC_TYPE_PATTERNS = {
    'memo': [
        r'^MEMO(?:RANDUM)?$',
        r'OFFICE\s+MEMORANDUM',
        r'MEMORANDUM\s+FOR',
    ],
    'teletype': [
        r'^TELETYPE$',
        r'^AIRTEL$',
        r'^URGENT$',
        r'DECODED\s+MESSAGE',
    ],
    'report': [
        r'INVESTIGATIVE\s+REPORT',
        r'REPORT\s+OF\s+(?:INVESTIGATION|INTERVIEW)',
        r'SPECIAL\s+AGENT\s+IN\s+CHARGE',
    ],
    'letter': [
        r'Dear\s+(?:Sir|Mr\.|Mrs\.|Miss)',
        r'Sincerely\s+yours',
        r'Respectfully\s+submitted',
    ],
    'form': [
        r'COMPLAINT\s+FORM',
        r'FD-\d+',
        r'FBI\s+FORM',
    ],
}

# Patterns for metadata extraction
METADATA_PATTERNS = {
    'date': [
        r'(?:DATE[:\s]+)?(\w+\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
        r'(\d{4}-\d{2}-\d{2})',
    ],
    'from': [
        r'FROM[:\s]+([A-Z][A-Za-z\s,\.]+?)(?:\n|TO:|RE:|SUBJECT:)',
        r'SAC[,\s]+([A-Z][A-Za-z\s]+)',
    ],
    'to': [
        r'TO[:\s]+([A-Z][A-Za-z\s,\.]+?)(?:\n|FROM:|RE:|SUBJECT:)',
        r'DIRECTOR[,\s]+FBI',
    ],
    'subject': [
        r'(?:RE|SUBJECT)[:\s]+(.+?)(?:\n\n|\n[A-Z]+:)',
    ],
    'file_number': [
        r'(\d{2,3}-\d{4,6})',
        r'FILE\s+NO\.?\s*[:\s]*(\d+-\d+)',
    ],
}


def classify_page(text: str) -> Tuple[str, bool]:
    """
    Classify a page and determine if it should be skipped.
    
    Returns (page_type, should_skip)
    """
    text_upper = text.upper()
    
    # Check skip patterns first
    for skip_type, patterns in SKIP_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_upper, re.IGNORECASE | re.MULTILINE):
                return (skip_type, True)
    
    # Detect document type
    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_upper, re.IGNORECASE | re.MULTILINE):
                return (doc_type, False)
    
    # Default: narrative content
    return ('narrative', False)


def extract_document_metadata(text: str) -> Dict[str, Optional[str]]:
    """Extract structured metadata from FBI document headers."""
    metadata = {
        'date': None,
        'from_field': None,
        'to_field': None,
        'subject': None,
        'file_number': None,
        'doc_type': None,
    }
    
    # Look only in first 1000 chars (header area)
    header_text = text[:1000]
    
    # Extract date
    for pattern in METADATA_PATTERNS['date']:
        m = re.search(pattern, header_text, re.IGNORECASE)
        if m:
            metadata['date'] = m.group(1).strip()
            break
    
    # Extract file number
    for pattern in METADATA_PATTERNS['file_number']:
        m = re.search(pattern, header_text)
        if m:
            metadata['file_number'] = m.group(1).strip()
            break
    
    # Detect document type
    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, header_text, re.IGNORECASE | re.MULTILINE):
                metadata['doc_type'] = doc_type
                break
        if metadata['doc_type']:
            break
    
    return metadata


# =============================================================================
# OCR Text Normalization
# =============================================================================

def normalize_fbi_ocr(raw_text: str) -> str:
    """
    Normalize OCR text from FBI documents.
    
    Handles:
    - Hyphenated line breaks
    - Common FBI OCR artifacts
    - Classification stamps
    - Excessive whitespace
    """
    text = raw_text
    
    # NBSP -> space
    text = text.replace("\u00a0", " ")
    
    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    # Remove classification stamps (often repeated)
    text = re.sub(r'\b(CONFIDENTIAL|SECRET|TOP SECRET|UNCLASSIFIED)\b', '', text, flags=re.IGNORECASE)
    
    # Clean up common OCR artifacts
    # Multiple spaces -> single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Fix common FBI OCR errors
    text = re.sub(r'\bKOSENBERG\b', 'ROSENBERG', text, flags=re.IGNORECASE)
    text = re.sub(r'\bROSEKBERG\b', 'ROSENBERG', text, flags=re.IGNORECASE)
    text = re.sub(r'\bROSENSERG\b', 'ROSENBERG', text, flags=re.IGNORECASE)
    
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Join wrapped lines within paragraphs
    lines = text.split("\n")
    joined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            line_stripped = line.rstrip()
            # Join if line doesn't end with terminal punctuation and next starts lowercase
            if (line_stripped and 
                not re.search(r'[.?!:;"\'\)]$', line_stripped) and
                next_line and len(next_line) > 0 and next_line[0].islower()):
                joined_lines.append(line.rstrip() + " " + next_line)
                i += 2
                continue
        joined_lines.append(line)
        i += 1
    
    text = "\n".join(joined_lines)
    
    return text.strip()


# =============================================================================
# Boilerplate Detection
# =============================================================================

def detect_boilerplate(pages_text: List[str], threshold: float = 0.30,
                       top_n: int = 4, bottom_n: int = 4) -> Set[str]:
    """Detect boilerplate lines that appear on many pages."""
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
    
    return boilerplate


def remove_boilerplate(text: str, boilerplate: Set[str],
                       top_n: int = 4, bottom_n: int = 4) -> str:
    """Remove boilerplate lines from text."""
    lines = text.split("\n")
    new_lines = []
    
    for i, line in enumerate(lines):
        norm = re.sub(r"\s+", " ", line.lower().strip())
        
        # Check if line matches boilerplate (only in header/footer region)
        is_bp = norm in boilerplate and (i < top_n or i >= len(lines) - bottom_n)
        
        if not is_bp:
            new_lines.append(line)
    
    return "\n".join(new_lines)


# =============================================================================
# Document Boundary Detection
# =============================================================================

def detect_document_start(text: str) -> bool:
    """
    Detect if a page likely starts a new FBI document.
    
    FBI documents typically start with:
    - MEMORANDUM headers
    - TELETYPE headers
    - TO:/FROM: blocks
    - Date headers
    """
    # Check first 500 chars
    header = text[:500].upper()
    
    start_patterns = [
        r'^MEMO(?:RANDUM)?',
        r'^OFFICE\s+MEMORANDUM',
        r'^TELETYPE',
        r'^AIRTEL',
        r'^TO\s*:\s*DIRECTOR',
        r'^TO\s*:\s*SAC',
        r'^FROM\s*:\s*SAC',
        r'^\s*UNITED\s+STATES\s+DEPARTMENT\s+OF\s+JUSTICE',
        r'^\s*FEDERAL\s+BUREAU\s+OF\s+INVESTIGATION\s*\n.*(?:REPORT|MEMO)',
    ]
    
    for pattern in start_patterns:
        if re.search(pattern, header, re.MULTILINE):
            return True
    
    return False


# =============================================================================
# Chunking
# =============================================================================

@dataclass
class PageData:
    """Data for a single page."""
    page_id: int
    pdf_page_number: int
    raw_text: str
    clean_text: str
    page_type: str
    should_skip: bool
    is_doc_start: bool
    metadata: Dict


@dataclass
class ChunkData:
    """Data for a single chunk."""
    text: str
    page_ids: List[int]
    page_start: int
    page_end: int
    chunk_index: int
    doc_type: Optional[str]
    file_number: Optional[str]
    doc_date: Optional[str]


def create_chunks_from_pages(
    pages: List[PageData],
    config: ChunkingConfig,
) -> List[ChunkData]:
    """
    Create chunks from pages, respecting document boundaries.
    
    Key rules:
    - Skip pages marked for skipping (indexes, covers)
    - Try to keep documents together when possible
    - Don't create chunks smaller than min_chunk_chars
    """
    chunks: List[ChunkData] = []
    
    # Filter to usable pages
    usable_pages = [p for p in pages if not p.should_skip and len(p.clean_text) >= config.min_page_chars]
    
    if not usable_pages:
        return chunks
    
    # Accumulator
    current_text_parts: List[str] = []
    current_page_ids: List[int] = []
    current_page_numbers: List[int] = []
    current_char_count = 0
    current_doc_type = None
    current_file_number = None
    current_doc_date = None
    
    # For overlap
    prev_overlap_text = ""
    prev_overlap_pages: List[int] = []
    
    def flush_chunk(is_final: bool = False):
        nonlocal current_text_parts, current_page_ids, current_page_numbers
        nonlocal current_char_count, current_doc_type, current_file_number, current_doc_date
        nonlocal prev_overlap_text, prev_overlap_pages
        
        if not current_text_parts:
            return
        
        chunk_text = "\n\n".join(current_text_parts)
        
        if len(chunk_text) < config.min_chunk_chars and not is_final:
            return
        
        chunk = ChunkData(
            text=chunk_text,
            page_ids=current_page_ids.copy(),
            page_start=min(current_page_numbers) if current_page_numbers else 0,
            page_end=max(current_page_numbers) if current_page_numbers else 0,
            chunk_index=len(chunks),
            doc_type=current_doc_type,
            file_number=current_file_number,
            doc_date=current_doc_date,
        )
        chunks.append(chunk)
        
        # Compute overlap for next chunk
        overlap_chars = 0
        overlap_parts = []
        overlap_pages = []
        for i in range(len(current_text_parts) - 1, -1, -1):
            part = current_text_parts[i]
            if overlap_chars + len(part) > config.overlap_chars and overlap_parts:
                break
            overlap_parts.insert(0, part)
            overlap_chars += len(part)
            if current_page_ids and i < len(current_page_ids):
                overlap_pages.insert(0, current_page_ids[i])
        
        prev_overlap_text = "\n\n".join(overlap_parts)
        prev_overlap_pages = overlap_pages
        
        # Reset accumulator
        current_text_parts = []
        current_page_ids = []
        current_page_numbers = []
        current_char_count = 0
    
    # Process pages
    for page in usable_pages:
        # Check if this is a document boundary
        if page.is_doc_start and current_text_parts:
            # Flush current chunk before starting new document
            flush_chunk()
            # Update document metadata from new document
            current_doc_type = page.metadata.get('doc_type')
            current_file_number = page.metadata.get('file_number')
            current_doc_date = page.metadata.get('date')
            # Apply overlap from previous chunk
            if prev_overlap_text:
                current_text_parts = [prev_overlap_text]
                current_page_ids = prev_overlap_pages.copy()
                current_char_count = len(prev_overlap_text)
        
        # Check if adding this page exceeds target
        page_len = len(page.clean_text)
        if current_char_count + page_len > config.target_chars and current_text_parts:
            flush_chunk()
            # Apply overlap
            if prev_overlap_text:
                current_text_parts = [prev_overlap_text]
                current_page_ids = prev_overlap_pages.copy()
                current_char_count = len(prev_overlap_text)
        
        # Update document metadata if this page has it
        if page.metadata.get('doc_type') and not current_doc_type:
            current_doc_type = page.metadata['doc_type']
        if page.metadata.get('file_number') and not current_file_number:
            current_file_number = page.metadata['file_number']
        if page.metadata.get('date') and not current_doc_date:
            current_doc_date = page.metadata['date']
        
        # Add page to current chunk
        current_text_parts.append(page.clean_text)
        current_page_ids.append(page.page_id)
        current_page_numbers.append(page.pdf_page_number)
        current_char_count += page_len
    
    # Flush final chunk
    flush_chunk(is_final=True)
    
    return chunks


# =============================================================================
# Database Operations
# =============================================================================

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
    # Check if document exists
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
                logical_label: str, raw_text: str, content_role: str = 'primary') -> int:
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
    """Truncate text to fit within max_bytes, respecting UTF-8 boundaries."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    return truncated


def insert_chunk(cur, text: str, pipeline_version: str, metadata: Dict = None) -> int:
    """
    Insert a chunk, with retry logic for trigram index size limits.
    
    GIST trigram indexes have an 8KB row limit. If insert fails due to
    index size, progressively truncate until it fits.
    Uses savepoints to allow retry after failure.
    """
    # Try progressively smaller sizes if we hit the index limit
    max_sizes = [None, 6000, 4000, 2500]  # None = no truncation
    
    for i, max_size in enumerate(max_sizes):
        text_to_insert = text if max_size is None else safe_truncate_bytes(text, max_size)
        savepoint = f"insert_retry_{i}"
        
        try:
            cur.execute(f"SAVEPOINT {savepoint}")
            cur.execute(
                """
                INSERT INTO chunks (text, clean_text, pipeline_version)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (text_to_insert, text_to_insert, pipeline_version),
            )
            result = int(cur.fetchone()[0])
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            return result
        except Exception as e:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            error_msg = str(e)
            if "index row requires" in error_msg and "maximum size is 8191" in error_msg:
                # Trigram index limit - try smaller size
                if max_size == max_sizes[-1]:
                    # Last resort - skip this chunk
                    print(f"      WARNING: Chunk too large for index even at {max_size} bytes, skipping")
                    raise
                # Continue to next size
                continue
            else:
                # Different error - re-raise
                raise
    
    raise Exception("Failed to insert chunk after all truncation attempts")


def insert_chunk_pages(cur, chunk_id: int, page_ids: List[int]):
    for i, page_id in enumerate(page_ids, start=1):
        cur.execute(
            """
            INSERT INTO chunk_pages (chunk_id, page_id, span_order)
            VALUES (%s, %s, %s)
            """,
            (chunk_id, page_id, i),
        )


def insert_chunk_metadata(cur, chunk_id: int, document_id: int, pipeline_version: str,
                          first_page_id: int, last_page_id: int, content_type: str,
                          doc_type: str = None, file_number: str = None):
    """Insert chunk_metadata to link chunk to document/collection."""
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
    
    # Fallback to blocks extraction
    blocks = page.get_text("blocks") or []
    parts = [b[4] for b in blocks if len(b) >= 5 and isinstance(b[4], str)]
    txt2 = "\n".join(parts)
    return txt2 if txt2.strip() else ""


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
) -> Tuple[int, int, int, int]:
    """
    Process a single Rosenberg FBI file.
    
    Returns (doc_id, page_count, pages_used, chunk_count)
    """
    source_name = pdf_path.name
    
    # Extract volume/batch identifier from filename
    volume_match = re.search(r'(\d+|Batch\s+\d+|Retort\s+\d+)', source_name)
    volume = volume_match.group(1) if volume_match else ""
    
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
    
    if dry_run:
        # Count pages that would be used
        usable = 0
        skipped_types = Counter()
        for pdf_page_num, raw_text in raw_pages:
            clean_text = normalize_fbi_ocr(raw_text)
            clean_text = remove_boilerplate(clean_text, boilerplate)
            page_type, should_skip = classify_page(clean_text)
            if should_skip or len(clean_text) < config.min_page_chars:
                skipped_types[page_type] += 1
            else:
                usable += 1
        
        print(f"    [DRY RUN] Pages: {usable} usable, skipped: {dict(skipped_types)}")
        return 0, page_count, usable, 0
    
    # Build metadata
    meta = {
        "source_format": "pdf_fbi_scan",
        "extractor": "pymupdf",
        "page_count": page_count,
        "boilerplate_patterns": len(boilerplate),
        "document_type": "fbi_case_file",
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
    
    # Process pages and build PageData list
    page_data_list: List[PageData] = []
    pages_skipped = 0
    
    for pdf_page_num, raw_text in raw_pages:
        page_seq = pdf_page_num
        logical_label = f"p{pdf_page_num:04d}"
        
        # Normalize and clean
        clean_text = normalize_fbi_ocr(raw_text)
        clean_text = remove_boilerplate(clean_text, boilerplate)
        
        # Classify page
        page_type, should_skip = classify_page(clean_text)
        
        # Check minimum content
        if len(clean_text) < config.min_page_chars:
            should_skip = True
            page_type = 'blank'
        
        # Detect document start
        is_doc_start = detect_document_start(clean_text) if not should_skip else False
        
        # Extract metadata
        doc_metadata = extract_document_metadata(clean_text) if not should_skip else {}
        
        # Determine content_role for DB
        if should_skip:
            content_role = 'auxiliary'
            pages_skipped += 1
        else:
            content_role = 'primary'
        
        # Insert page (we insert all pages, even skipped ones, for completeness)
        page_id = insert_page(cur, doc_id, page_seq, pdf_page_num, logical_label, 
                             raw_text, content_role)
        
        page_data_list.append(PageData(
            page_id=page_id,
            pdf_page_number=pdf_page_num,
            raw_text=raw_text,
            clean_text=clean_text,
            page_type=page_type,
            should_skip=should_skip,
            is_doc_start=is_doc_start,
            metadata=doc_metadata,
        ))
    
    pages_used = page_count - pages_skipped
    print(f"    Pages: {pages_used} usable, {pages_skipped} skipped")
    
    # Create chunks
    chunks = create_chunks_from_pages(page_data_list, config)
    print(f"    Created {len(chunks)} chunks")
    
    # Create a savepoint to recover from individual chunk failures
    cur.execute("SAVEPOINT doc_start")
    
    # Insert chunks
    # insert_chunk handles trigram index size errors internally with retries
    chunks_inserted = 0
    chunks_failed = 0
    for chunk in chunks:
        try:
            chunk_id = insert_chunk(cur, chunk.text, pipeline_version)
            insert_chunk_pages(cur, chunk_id, chunk.page_ids)
            
            # Insert chunk_metadata
            first_page_id = chunk.page_ids[0] if chunk.page_ids else None
            last_page_id = chunk.page_ids[-1] if chunk.page_ids else None
            
            content_type = chunk.doc_type or "fbi_document"
            insert_chunk_metadata(cur, chunk_id, doc_id, pipeline_version,
                                 first_page_id, last_page_id, content_type,
                                 chunk.doc_type, chunk.file_number)
            chunks_inserted += 1
        except Exception as e:
            chunks_failed += 1
            error_msg = str(e).split('\n')[0][:100]
            print(f"      WARNING: Chunk {chunk.chunk_index} skipped: {error_msg}")
            # Use a savepoint to recover from the failed state
            cur.execute("ROLLBACK TO SAVEPOINT doc_start")
            cur.execute("SAVEPOINT doc_start")
            continue
    
    if chunks_failed > 0:
        print(f"    Chunks: {chunks_inserted} inserted, {chunks_failed} skipped")
    
    return doc_id, page_count, pages_used, chunks_inserted


def main():
    ap = argparse.ArgumentParser(description="Ingest Julius Rosenberg FBI Files")
    ap.add_argument("--input-dir", default="data/raw/rosenberg", help="Directory containing PDFs")
    ap.add_argument("--glob", default="*.pdf", help="File glob pattern")
    ap.add_argument("--pipeline-version", default="rosenberg_v1", help="Chunking pipeline version")
    ap.add_argument("--target-chars", type=int, default=4000, help="Target chunk size")
    ap.add_argument("--max-chars", type=int, default=6000, help="Max chunk size")
    ap.add_argument("--overlap-chars", type=int, default=800, help="Overlap between chunks")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of PDFs")
    ap.add_argument("--no-sha", action="store_true", help="Skip SHA256 computation")
    ap.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = ap.parse_args()
    
    config = ChunkingConfig(
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    
    import glob as glob_mod
    paths = sorted(glob_mod.glob(str(Path(args.input_dir) / args.glob)))
    if args.limit:
        paths = paths[:args.limit]
    
    if not paths:
        print(f"No PDFs found at {args.input_dir}/{args.glob}")
        return
    
    print(f"Rosenberg FBI Files Ingest")
    print(f"Found {len(paths)} PDF files to process")
    print(f"Config: target={config.target_chars}, max={config.max_chars}, overlap={config.overlap_chars}")
    print()
    
    total_pages = 0
    total_pages_used = 0
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
                doc_id, pages, pages_used, chunks = process_pdf(
                    p,
                    cur,
                    collection_id,
                    config,
                    pipeline_version,
                    compute_sha=not args.no_sha,
                    dry_run=args.dry_run,
                )
                total_pages += pages
                total_pages_used += pages_used
                total_chunks += chunks
                
                if not args.dry_run:
                    ingest_runs.mark_success(cur, source_key=source_key)
                    conn.commit()
                    print(f"    -> document_id={doc_id}")
            
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
        print(f"  Total pages: {total_pages}")
        print(f"  Pages used: {total_pages_used} ({100*total_pages_used/total_pages:.1f}%)")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Pipeline version: {args.pipeline_version}")


if __name__ == "__main__":
    main()
