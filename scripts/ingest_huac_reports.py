#!/usr/bin/env python3
"""
HUAC Reports Ingest - Page-Based Chunking

Ingests House Un-American Activities Committee annual reports and
special publications. Uses page-based chunking since these are 
narrative reports without speaker turns.

Includes:
- Annual HUAC reports (1948-1964)
- Special reports (Communist Conspiracy, Spotlight on Spies, etc.)

Collection slug: huac_reports
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

import psycopg2
import fitz  # PyMuPDF

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

COLLECTION_SLUG = "huac_reports"
COLLECTION_TITLE = "HUAC Reports (1948-1964)"
COLLECTION_DESCRIPTION = """House Un-American Activities Committee annual reports 
and special publications documenting committee findings, investigations, and 
analyses of Communist activities in the United States."""


# =============================================================================
# Chunking Configuration
# =============================================================================

@dataclass
class ChunkingConfig:
    """Page-based chunking configuration."""
    target_chars: int = 5000
    max_chars: int = 7000
    overlap_chars: int = 500
    min_chunk_chars: int = 200
    min_page_chars: int = 100  # Skip nearly blank pages


DEFAULT_CONFIG = ChunkingConfig()


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
    char_count: int
    is_blank: bool


@dataclass
class ChunkData:
    """Data for a single chunk."""
    text: str
    page_ids: List[int]
    page_start: int
    page_end: int
    chunk_index: int


# =============================================================================
# Text Processing
# =============================================================================

def normalize_text(raw_text: str) -> str:
    """Clean text artifacts from scanned reports."""
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


def classify_page(text: str, page_num: int) -> Tuple[str, bool]:
    """
    Classify page type.
    Returns (page_type, should_skip).
    """
    clean = text.strip()
    
    # Very short pages are likely blank or decorative
    if len(clean) < 100:
        return "blank", True
    
    # Table of contents
    if page_num < 20 and re.search(r'(TABLE OF CONTENTS|CONTENTS)', clean, re.IGNORECASE):
        if clean.count('...') > 5 or clean.count('. .') > 5:
            return "toc", False  # Keep TOC but mark it
    
    # Index pages (usually at end)
    if re.search(r'^INDEX\s*$', clean, re.MULTILINE | re.IGNORECASE):
        return "index", False
    
    # Primary content
    return "primary", False


def extract_year_from_filename(filename: str) -> Optional[str]:
    """Extract year from filename like 'Report...1948.pdf'."""
    match = re.search(r'(194\d|195\d|196\d)', filename)
    if match:
        return match.group(1)
    return None


def extract_report_title(text: str, filename: str) -> str:
    """Extract report title from content or filename."""
    # Try to find title in first 2000 chars
    first_part = text[:2000]
    
    # Look for "REPORT OF THE COMMITTEE" pattern
    match = re.search(r'(REPORT\s+OF\s+THE\s+COMMITTEE[^\n]+)', first_part, re.IGNORECASE)
    if match:
        return match.group(1).strip()[:200]
    
    # Fall back to filename
    # Convert "Report of the Committee of Un-American Activities 1948.pdf" -> title
    base = Path(filename).stem
    return base[:200]


# =============================================================================
# Chunking
# =============================================================================

def create_chunks_from_pages(
    pages: List[PageData],
    config: ChunkingConfig,
) -> List[ChunkData]:
    """
    Create chunks from pages using character-based boundaries.
    """
    chunks: List[ChunkData] = []
    
    # Filter out blank pages
    content_pages = [p for p in pages if not p.is_blank]
    
    if not content_pages:
        return chunks
    
    current_text_parts: List[str] = []
    current_pages: List[PageData] = []
    current_chars = 0
    
    for page in content_pages:
        page_text = page.clean_text
        page_chars = len(page_text)
        
        # Check if adding this page exceeds target
        if current_chars + page_chars > config.target_chars and current_text_parts:
            # Create chunk
            chunk_text = "\n\n".join(current_text_parts)
            
            if len(chunk_text) >= config.min_chunk_chars:
                chunks.append(ChunkData(
                    text=chunk_text,
                    page_ids=[p.page_id for p in current_pages],
                    page_start=current_pages[0].pdf_page_number,
                    page_end=current_pages[-1].pdf_page_number,
                    chunk_index=len(chunks),
                ))
            
            # Start new chunk with overlap
            if current_text_parts:
                overlap_text = current_text_parts[-1][-config.overlap_chars:]
                current_text_parts = [overlap_text]
                current_chars = len(overlap_text)
                current_pages = [current_pages[-1]]
            else:
                current_text_parts = []
                current_chars = 0
                current_pages = []
        
        current_text_parts.append(page_text)
        current_pages.append(page)
        current_chars += page_chars
    
    # Final chunk
    if current_text_parts and current_chars >= config.min_chunk_chars:
        chunk_text = "\n\n".join(current_text_parts)
        chunks.append(ChunkData(
            text=chunk_text,
            page_ids=[p.page_id for p in current_pages],
            page_start=current_pages[0].pdf_page_number,
            page_end=current_pages[-1].pdf_page_number,
            chunk_index=len(chunks),
        ))
    
    return chunks


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
    cur.execute("""
        DELETE FROM chunks WHERE id IN (
            SELECT cp.chunk_id FROM chunk_pages cp
            JOIN pages p ON p.id = cp.page_id
            WHERE p.document_id = %s
        )
    """, (document_id,))
    cur.execute("DELETE FROM pages WHERE document_id = %s", (document_id,))


def insert_page(cur, document_id: int, page_seq: int, pdf_page_number: int,
                logical_label: str, raw_text: str, content_role: str) -> int:
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
    """Truncate text to fit within max_bytes."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore')


def insert_chunk(cur, text: str, pipeline_version: str) -> int:
    """Insert a chunk with retry logic for index limits."""
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
# PDF Processing
# =============================================================================

def extract_page_text(doc, page_index: int) -> str:
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
) -> Tuple[int, int, int, int]:
    """
    Process PDF with page-based chunking.
    
    Returns (doc_id, page_count, pages_used, chunk_count)
    """
    source_name = pdf_path.name
    
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    
    print(f"  Processing {source_name}: {page_count} pages")
    
    # Extract year from filename
    year = extract_year_from_filename(source_name) or "unknown"
    
    # Extract and process all pages
    page_data_list: List[PageData] = []
    all_text = ""
    
    for i in range(page_count):
        raw_text = extract_page_text(doc, i)
        clean_text = normalize_text(raw_text)
        page_type, should_skip = classify_page(clean_text, i + 1)
        
        is_blank = should_skip or len(clean_text) < config.min_page_chars
        
        page_data_list.append(PageData(
            page_id=0,  # Will be set after DB insert
            pdf_page_number=i + 1,
            raw_text=raw_text,
            clean_text=clean_text,
            char_count=len(clean_text),
            is_blank=is_blank,
        ))
        
        if not is_blank:
            all_text += clean_text + "\n"
    
    doc.close()
    
    # Extract title
    title = extract_report_title(all_text, source_name)
    
    # Count usable pages
    pages_used = sum(1 for p in page_data_list if not p.is_blank)
    
    print(f"    Year: {year}, Pages used: {pages_used}/{page_count}")
    
    if dry_run:
        chunks = create_chunks_from_pages(page_data_list, config)
        print(f"    [DRY RUN] Would create {len(chunks)} chunks")
        return 0, page_count, pages_used, len(chunks)
    
    # Build metadata
    meta = {
        "source_format": "pdf_huac_report",
        "extractor": "pymupdf",
        "page_count": page_count,
        "pages_used": pages_used,
        "year": year,
        "title": title,
        "document_type": "report",
    }
    
    doc_id = upsert_document(cur, collection_id, source_name, str(pdf_path), year, meta)
    
    # Clear existing data
    delete_document_data(cur, doc_id)
    
    # Insert pages
    for i, page_data in enumerate(page_data_list):
        content_role = "blank" if page_data.is_blank else "primary"
        logical_label = f"p{page_data.pdf_page_number:04d}"
        page_id = insert_page(cur, doc_id, i + 1, page_data.pdf_page_number,
                             logical_label, page_data.raw_text, content_role)
        page_data.page_id = page_id
    
    # Create chunks
    chunks = create_chunks_from_pages(page_data_list, config)
    print(f"    Created {len(chunks)} chunks")
    
    # Insert chunks with error handling
    cur.execute("SAVEPOINT doc_start")
    chunks_inserted = 0
    chunks_failed = 0
    
    for chunk in chunks:
        try:
            chunk_id = insert_chunk(cur, chunk.text, pipeline_version)
            insert_chunk_pages(cur, chunk_id, chunk.page_ids)
            
            first_page_id = chunk.page_ids[0] if chunk.page_ids else None
            last_page_id = chunk.page_ids[-1] if chunk.page_ids else None
            
            # Determine content type based on year
            if year != "unknown":
                content_type = f"huac_annual_report_{year}"
            else:
                content_type = "huac_report"
            
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
    
    return doc_id, page_count, pages_used, chunks_inserted


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="HUAC Reports - Page-Based Ingest")
    ap.add_argument("--input-dir", default=None,
                    help="Directory containing PDFs (processes both report directories by default)")
    ap.add_argument("--glob", default="*.pdf", help="File glob pattern")
    ap.add_argument("--pipeline-version", default="huac_reports_v1_pages",
                    help="Pipeline version")
    ap.add_argument("--target-chars", type=int, default=5000,
                    help="Target chunk size in chars")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of PDFs to process")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write to database")
    args = ap.parse_args()
    
    config = ChunkingConfig(
        target_chars=args.target_chars,
    )
    
    # Collect paths from both directories
    import glob as glob_mod
    
    if args.input_dir:
        paths = sorted(glob_mod.glob(str(Path(args.input_dir) / args.glob)))
    else:
        # Default: process both report directories
        paths = []
        
        # Annual reports
        annual_reports_dir = Path("data/raw/unamerican_reports")
        if annual_reports_dir.exists():
            paths.extend(sorted(glob_mod.glob(str(annual_reports_dir / args.glob))))
        
        # Special reports from hearings dir (those classified as reports)
        hearings_dir = Path("data/raw/umamerican_hearings")
        if hearings_dir.exists():
            # Include all report/compilation files from hearings directory
            # (excluding hearing transcripts which go to huac_hearings collection)
            report_files = [
                # Special reports
                "americannegroinc00unit.pdf",      # American Negro in Communist Party
                "communistconspir195601aunit.pdf", # Communist Conspiracy (1956)
                "shamefulyearsthi1952unit.pdf",    # Shameful Years (1952)
                "spotlightonspies1949unit.pdf",    # Spotlight on Spies (1949)
                # Additional compilations
                "communistinfiltr04unit.pdf",      # Communist Infiltration (Vol. 4)
                "communistoutlets02unit.pdf",      # Communist Outlets (Vol. 2)
                "scopeofsovietact2123unit.pdf",    # Scope of Soviet Activities (1952-53)
                "sovietespionagea1949unit.pdf",    # Soviet Espionage Activities (1949)
            ]
            for f in report_files:
                p = hearings_dir / f
                if p.exists():
                    paths.append(str(p))
    
    if args.limit:
        paths = paths[:args.limit]
    
    if not paths:
        print("No PDFs found")
        return
    
    print(f"HUAC Reports - Page-Based Ingest")
    print(f"Found {len(paths)} PDF files")
    print(f"Config: target={config.target_chars} chars")
    print()
    
    total_pages = 0
    total_pages_used = 0
    total_chunks = 0
    
    with connect() as conn, conn.cursor() as cur:
        collection_id = get_or_create_collection(cur)
        print(f"Collection: {COLLECTION_SLUG} (id={collection_id})")
        print()
        
        for pdf_path in paths:
            try:
                doc_id, pages, pages_used, chunks = process_pdf(
                    Path(pdf_path), cur, collection_id, config,
                    args.pipeline_version, args.dry_run,
                )
                total_pages += pages
                total_pages_used += pages_used
                total_chunks += chunks
                
                if not args.dry_run:
                    conn.commit()
                    print(f"    -> document_id={doc_id}")
            
            except Exception as e:
                print(f"    ERROR: {e}")
                conn.rollback()
                continue
        
        print()
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Done!")
        print(f"  Total pages: {total_pages:,}")
        print(f"  Pages used: {total_pages_used:,} ({100*total_pages_used/total_pages:.1f}%)" if total_pages > 0 else "")
        print(f"  Total chunks: {total_chunks:,}")
        print(f"  Pipeline version: {args.pipeline_version}")


if __name__ == "__main__":
    main()
