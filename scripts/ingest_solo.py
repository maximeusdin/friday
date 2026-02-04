#!/usr/bin/env python3
"""
Ingest script for FBI SOLO Operation files.

Key strategy: MEMO-AWARE CHUNKING
- Detect individual FBI memos within each PDF
- Keep each memo as a single chunk when possible
- Split large memos by page with overlap
- Preserve memo metadata (type, date, TO/FROM)

This is different from character-based chunking - it preserves
document integrity for better retrieval and attribution.

Collection slug: solo
FBI Case: 100-HQ-428091
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

COLLECTION_SLUG = "solo"
COLLECTION_TITLE = "FBI SOLO Operation Files"
COLLECTION_DESCRIPTION = """FBI surveillance operation monitoring Communist Party USA 
through informants Morris and Jack Childs (1958-1977). Case file 100-HQ-428091. 
Includes intelligence on Soviet funding, international Communist communications, 
and CP-USA leadership activities."""


# =============================================================================
# Chunking Configuration
# =============================================================================

@dataclass
class ChunkingConfig:
    """Memo-aware chunking configuration."""
    max_memo_chars: int = 6000      # Memos smaller than this stay as single chunk
    max_chunk_chars: int = 7000     # Hard limit for any chunk
    page_overlap_chars: int = 500   # Overlap when splitting large memos
    min_chunk_chars: int = 200      # Don't create tiny chunks
    min_page_chars: int = 100       # Pages below this are considered blank
    memo_detection_threshold: float = 2.0  # Minimum score to detect memo start


DEFAULT_CONFIG = ChunkingConfig()


# =============================================================================
# Database Connection
# =============================================================================

def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


# =============================================================================
# Memo Detection
# =============================================================================

def detect_memo_start(text: str) -> Tuple[float, str, Dict]:
    """
    Detect if a page starts a new FBI memo.
    
    Returns (confidence_score, memo_type, metadata)
    """
    first_800 = text[:800]
    upper_800 = first_800.upper()
    
    score = 0.0
    memo_type = "unknown"
    metadata = {}
    
    # Strong indicators (score += 2)
    # TO: DIRECTOR/SAC or To DIRECTOR/SAC (with or without colon)
    if re.search(r'TO\s*:?\s*(DIRECTOR|SAC)', upper_800):
        score += 2
        memo_type = "memo"
        m = re.search(r'TO\s*:?\s*([^\n]+)', first_800, re.IGNORECASE)
        if m:
            metadata['to'] = m.group(1).strip()[:100]
    
    # FROM: SAC/DIRECTOR or From SAC (with or without colon)
    if re.search(r'FROM\s*:?\s*(SAC|DIRECTOR)', upper_800):
        score += 2
        m = re.search(r'FROM\s*:?\s*([^\n]+)', first_800, re.IGNORECASE)
        if m:
            metadata['from'] = m.group(1).strip()[:100]
    
    # Memo headers - various formats
    if re.search(r'UNITED STATES GOVERNMENT', upper_800):
        score += 1
    
    # "Memorandum" or "Memorandum for" at start of line
    if re.search(r'^MEMORANDUM', first_800, re.MULTILINE | re.IGNORECASE):
        score += 1.5
        memo_type = "memo"
    
    # OFFICE MEMORANDUM header
    if re.search(r'OFFICE\s+MEMORANDUM', upper_800):
        score += 2
        memo_type = "memo"
    
    # Communication types
    if re.search(r'AIRTEL', upper_800):
        score += 2
        memo_type = "airtel"
    
    if re.search(r'TELETYPE', upper_800):
        score += 2
        memo_type = "teletype"
    
    if re.search(r'^URGENT\s*$', first_800, re.MULTILINE | re.IGNORECASE):
        score += 1.5
        memo_type = "urgent"
    
    # RE: or SUBJECT: header (strong indicator of memo start)
    if re.search(r'^(RE|SUBJECT)\s*:', first_800, re.MULTILINE | re.IGNORECASE):
        score += 1
    
    # File number with context
    if re.search(r'\(100-428091\)', first_800):
        score += 1
    elif re.search(r'100-428091', first_800):
        score += 0.5
    
    # "Priority or Method of Mailing" is a common FBI form header
    if re.search(r'Priority or Method of Mailing', first_800, re.IGNORECASE):
        score += 1.5
        memo_type = "memo"
    
    # Date extraction
    date_match = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}',
        first_800, re.IGNORECASE
    )
    if date_match:
        score += 0.5
        metadata['date'] = date_match.group(0)
    
    # Numeric date format
    if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', first_800):
        score += 0.3
    
    # Subject extraction
    subj_match = re.search(r'(?:SUBJECT|RE)\s*:\s*([^\n]+)', first_800, re.IGNORECASE)
    if subj_match:
        metadata['subject'] = subj_match.group(1).strip()[:200]
    
    return score, memo_type, metadata


def extract_serial_range(filename: str) -> Optional[Tuple[int, int]]:
    """Extract serial number range from filename."""
    # Pattern: Serial0001-0044 or Serial01-44
    match = re.search(r'Serial(\d+)-(\d+)', filename, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# =============================================================================
# OCR Text Normalization
# =============================================================================

def normalize_solo_text(raw_text: str) -> str:
    """Normalize OCR text from SOLO files."""
    text = raw_text
    
    # NBSP -> space
    text = text.replace("\u00a0", " ")
    
    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    # Multiple spaces -> single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PageData:
    """Data for a single page."""
    page_id: int
    pdf_page_number: int
    raw_text: str
    clean_text: str
    is_blank: bool


@dataclass
class MemoData:
    """Data for a detected memo."""
    memo_index: int
    memo_type: str
    start_page: int
    end_page: int
    pages: List[PageData]
    metadata: Dict
    total_chars: int


@dataclass
class ChunkData:
    """Data for a single chunk."""
    text: str
    page_ids: List[int]
    page_start: int
    page_end: int
    chunk_index: int
    memo_index: int
    memo_type: str
    memo_part: Optional[int]  # If memo was split, which part (1, 2, 3...)
    memo_total_parts: Optional[int]
    metadata: Dict


# =============================================================================
# Memo-Aware Chunking
# =============================================================================

def detect_memos(pages: List[PageData], config: ChunkingConfig) -> List[MemoData]:
    """
    Detect memo boundaries and group pages into memos.
    """
    memos: List[MemoData] = []
    
    # Find memo start pages
    memo_starts = []
    for i, page in enumerate(pages):
        if page.is_blank:
            continue
        
        score, memo_type, metadata = detect_memo_start(page.clean_text)
        if score >= config.memo_detection_threshold:
            memo_starts.append((i, memo_type, metadata))
    
    # If no memos detected, treat whole document as one
    if not memo_starts:
        total_chars = sum(len(p.clean_text) for p in pages if not p.is_blank)
        memos.append(MemoData(
            memo_index=0,
            memo_type="document",
            start_page=pages[0].pdf_page_number if pages else 0,
            end_page=pages[-1].pdf_page_number if pages else 0,
            pages=[p for p in pages if not p.is_blank],
            metadata={},
            total_chars=total_chars,
        ))
        return memos
    
    # Group pages into memos
    for j, (start_idx, memo_type, metadata) in enumerate(memo_starts):
        # Find end of this memo
        if j + 1 < len(memo_starts):
            end_idx = memo_starts[j + 1][0] - 1
        else:
            end_idx = len(pages) - 1
        
        # Collect pages for this memo (excluding blanks)
        memo_pages = [p for p in pages[start_idx:end_idx + 1] if not p.is_blank]
        total_chars = sum(len(p.clean_text) for p in memo_pages)
        
        if memo_pages:
            memos.append(MemoData(
                memo_index=j,
                memo_type=memo_type,
                start_page=memo_pages[0].pdf_page_number,
                end_page=memo_pages[-1].pdf_page_number,
                pages=memo_pages,
                metadata=metadata,
                total_chars=total_chars,
            ))
    
    return memos


def chunk_memo(memo: MemoData, config: ChunkingConfig, chunk_start_index: int) -> List[ChunkData]:
    """
    Convert a memo into one or more chunks.
    
    - Small memos become single chunks
    - Large memos are split by page with overlap
    """
    chunks: List[ChunkData] = []
    
    # Combine all page text
    full_text = "\n\n".join(p.clean_text for p in memo.pages)
    
    # Small memo -> single chunk
    if len(full_text) <= config.max_memo_chars:
        chunks.append(ChunkData(
            text=full_text,
            page_ids=[p.page_id for p in memo.pages],
            page_start=memo.start_page,
            page_end=memo.end_page,
            chunk_index=chunk_start_index,
            memo_index=memo.memo_index,
            memo_type=memo.memo_type,
            memo_part=None,
            memo_total_parts=None,
            metadata=memo.metadata,
        ))
        return chunks
    
    # Large memo -> split by pages
    current_text_parts: List[str] = []
    current_pages: List[PageData] = []
    current_chars = 0
    part_num = 0
    
    # First pass: count how many parts we'll need
    test_chars = 0
    test_parts = 1
    for page in memo.pages:
        if test_chars + len(page.clean_text) > config.max_chunk_chars and test_chars > 0:
            test_parts += 1
            test_chars = config.page_overlap_chars
        test_chars += len(page.clean_text)
    
    total_parts = test_parts
    
    # Second pass: create chunks
    for i, page in enumerate(memo.pages):
        page_text = page.clean_text
        
        # Check if adding this page exceeds limit
        if current_chars + len(page_text) > config.max_chunk_chars and current_text_parts:
            # Create chunk from accumulated pages
            part_num += 1
            chunk_text = "\n\n".join(current_text_parts)
            
            chunks.append(ChunkData(
                text=chunk_text,
                page_ids=[p.page_id for p in current_pages],
                page_start=current_pages[0].pdf_page_number,
                page_end=current_pages[-1].pdf_page_number,
                chunk_index=chunk_start_index + len(chunks),
                memo_index=memo.memo_index,
                memo_type=memo.memo_type,
                memo_part=part_num,
                memo_total_parts=total_parts,
                metadata=memo.metadata,
            ))
            
            # Start new chunk with overlap from last page
            if current_text_parts:
                overlap_text = current_text_parts[-1][-config.page_overlap_chars:]
                current_text_parts = [overlap_text]
                current_chars = len(overlap_text)
                current_pages = [current_pages[-1]]
            else:
                current_text_parts = []
                current_chars = 0
                current_pages = []
        
        current_text_parts.append(page_text)
        current_pages.append(page)
        current_chars += len(page_text)
    
    # Final chunk
    if current_text_parts and current_chars >= config.min_chunk_chars:
        part_num += 1
        chunk_text = "\n\n".join(current_text_parts)
        
        chunks.append(ChunkData(
            text=chunk_text,
            page_ids=[p.page_id for p in current_pages],
            page_start=current_pages[0].pdf_page_number,
            page_end=current_pages[-1].pdf_page_number,
            chunk_index=chunk_start_index + len(chunks),
            memo_index=memo.memo_index,
            memo_type=memo.memo_type,
            memo_part=part_num,
            memo_total_parts=total_parts,
            metadata=memo.metadata,
        ))
    
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
                    serial_range: str, metadata: dict) -> int:
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
        (collection_id, source_name, source_ref, serial_range, json.dumps(metadata)),
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


def safe_truncate_bytes(text: str, max_bytes: int) -> str:
    """Truncate text to fit within max_bytes, respecting UTF-8 boundaries."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    return truncated


def insert_chunk(cur, text: str, pipeline_version: str, memo_type: str = None,
                 memo_part: int = None, memo_total_parts: int = None) -> int:
    """Insert a chunk with retry logic for trigram index limits."""
    max_sizes = [None, 6000, 4000, 2500]
    
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
# PDF Processing
# =============================================================================

def extract_page_text(doc: fitz.Document, page_index: int) -> str:
    page = doc.load_page(page_index)
    txt = page.get_text("text") or ""
    return txt


def process_pdf(
    pdf_path: Path,
    cur,
    collection_id: int,
    config: ChunkingConfig,
    pipeline_version: str,
    dry_run: bool = False,
) -> Tuple[int, int, int, int, int]:
    """
    Process a single SOLO PDF with memo-aware chunking.
    
    Returns (doc_id, page_count, memo_count, chunk_count, pages_used)
    """
    source_name = pdf_path.name
    
    # Extract serial range from filename
    serial_range = extract_serial_range(source_name)
    serial_str = f"{serial_range[0]}-{serial_range[1]}" if serial_range else ""
    
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    
    print(f"  Processing {source_name}: {page_count} pages")
    
    # Extract and normalize all page texts
    page_data_list: List[PageData] = []
    
    for i in range(page_count):
        raw_text = extract_page_text(doc, i)
        clean_text = normalize_solo_text(raw_text)
        is_blank = len(clean_text) < config.min_page_chars
        
        page_data_list.append(PageData(
            page_id=0,  # Will be set after DB insert
            pdf_page_number=i + 1,
            raw_text=raw_text,
            clean_text=clean_text,
            is_blank=is_blank,
        ))
    
    doc.close()
    
    # Detect memos
    memos = detect_memos(page_data_list, config)
    memo_count = len(memos)
    
    # Count pages used
    pages_used = sum(1 for p in page_data_list if not p.is_blank)
    
    if dry_run:
        # Estimate chunks
        chunk_count = 0
        for memo in memos:
            if memo.total_chars <= config.max_memo_chars:
                chunk_count += 1
            else:
                chunk_count += (memo.total_chars // config.max_chunk_chars) + 1
        
        print(f"    [DRY RUN] Memos: {memo_count}, Est chunks: {chunk_count}, Pages used: {pages_used}")
        return 0, page_count, memo_count, chunk_count, pages_used
    
    # Build document metadata
    meta = {
        "source_format": "pdf_fbi_solo",
        "extractor": "pymupdf",
        "page_count": page_count,
        "memo_count": memo_count,
        "fbi_case": "100-HQ-428091",
    }
    if serial_range:
        meta["serial_start"] = serial_range[0]
        meta["serial_end"] = serial_range[1]
    
    doc_id = upsert_document(
        cur,
        collection_id=collection_id,
        source_name=source_name,
        source_ref=str(pdf_path),
        serial_range=serial_str,
        metadata=meta,
    )
    
    # Clear existing data
    delete_chunks_for_document(cur, doc_id)
    delete_pages_for_document(cur, doc_id)
    
    # Insert pages and update page_data with IDs
    for i, page_data in enumerate(page_data_list):
        logical_label = f"p{page_data.pdf_page_number:04d}"
        page_id = insert_page(cur, doc_id, i + 1, page_data.pdf_page_number,
                             logical_label, page_data.raw_text)
        page_data.page_id = page_id
    
    print(f"    Memos detected: {memo_count}")
    
    # Create chunks from memos
    cur.execute("SAVEPOINT doc_start")
    
    all_chunks: List[ChunkData] = []
    chunk_index = 0
    for memo in memos:
        memo_chunks = chunk_memo(memo, config, chunk_index)
        all_chunks.extend(memo_chunks)
        chunk_index += len(memo_chunks)
    
    print(f"    Created {len(all_chunks)} chunks")
    
    # Insert chunks
    chunks_inserted = 0
    chunks_failed = 0
    
    for chunk in all_chunks:
        try:
            chunk_id = insert_chunk(cur, chunk.text, pipeline_version,
                                   chunk.memo_type, chunk.memo_part, chunk.memo_total_parts)
            insert_chunk_pages(cur, chunk_id, chunk.page_ids)
            
            first_page_id = chunk.page_ids[0] if chunk.page_ids else None
            last_page_id = chunk.page_ids[-1] if chunk.page_ids else None
            
            content_type = f"fbi_{chunk.memo_type}" if chunk.memo_type != "unknown" else "fbi_document"
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
    
    return doc_id, page_count, memo_count, chunks_inserted, pages_used


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Ingest FBI SOLO Operation Files (memo-aware)")
    ap.add_argument("--input-dir", default="data/raw/solo", help="Directory containing PDFs")
    ap.add_argument("--glob", default="*.pdf", help="File glob pattern")
    ap.add_argument("--pipeline-version", default="solo_v1_memo", help="Pipeline version")
    ap.add_argument("--max-memo-chars", type=int, default=6000, help="Max chars for single-chunk memo")
    ap.add_argument("--max-chunk-chars", type=int, default=7000, help="Max chunk size")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of PDFs")
    ap.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = ap.parse_args()
    
    config = ChunkingConfig(
        max_memo_chars=args.max_memo_chars,
        max_chunk_chars=args.max_chunk_chars,
    )
    
    import glob as glob_mod
    paths = sorted(glob_mod.glob(str(Path(args.input_dir) / args.glob)))
    if args.limit:
        paths = paths[:args.limit]
    
    if not paths:
        print(f"No PDFs found at {args.input_dir}/{args.glob}")
        return
    
    print(f"FBI SOLO Operation Files Ingest (Memo-Aware)")
    print(f"Found {len(paths)} PDF files to process")
    print(f"Config: max_memo={config.max_memo_chars}, max_chunk={config.max_chunk_chars}")
    print()
    
    total_pages = 0
    total_pages_used = 0
    total_memos = 0
    total_chunks = 0
    
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
                fp = ingest_runs.file_sha256(p)

                if not args.dry_run and not ingest_runs.should_run(
                    cur, source_key=source_key, source_fingerprint=fp, pipeline_version=pipeline_version
                ):
                    print(f"[skip] {p.name} (already ingested: pipeline={pipeline_version})")
                    continue

                if not args.dry_run:
                    ingest_runs.mark_running(
                        cur, source_key=source_key, source_fingerprint=fp, pipeline_version=pipeline_version
                    )

                doc_id, pages, memos, chunks, pages_used = process_pdf(
                    p,
                    cur,
                    collection_id,
                    config,
                    pipeline_version,
                    dry_run=args.dry_run,
                )
                total_pages += pages
                total_pages_used += pages_used
                total_memos += memos
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
        print(f"  Total pages: {total_pages:,}")
        print(f"  Pages used: {total_pages_used:,} ({100*total_pages_used/total_pages:.1f}%)")
        print(f"  Total memos: {total_memos:,}")
        print(f"  Total chunks: {total_chunks:,}")
        print(f"  Avg pages/memo: {total_pages_used/total_memos:.1f}" if total_memos > 0 else "")
        print(f"  Pipeline version: {args.pipeline_version}")


if __name__ == "__main__":
    main()
