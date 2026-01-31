#!/usr/bin/env python3
"""
OCR Candidate Span Generator

Extracts candidate entity mention spans from OCR text with quality scoring.
Conservative rules: 1-6 tokens, must pass basic quality, avoid boilerplate zones.

Usage:
    python scripts/extract_ocr_candidates.py --collection silvermaster --limit 100
    python scripts/extract_ocr_candidates.py --chunk-id 12345
"""

import argparse
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set, Dict, Tuple

import psycopg2
from psycopg2.extras import execute_values, Json

# Add parent directory for imports
sys.path.insert(0, '.')
from retrieval.surface_norm import normalize_surface


# =============================================================================
# CONFIGURATION
# =============================================================================

# Quality thresholds
MIN_SPAN_QUALITY = 0.3          # Minimum quality score to keep candidate
MIN_TOKEN_LENGTH = 2            # Minimum length per token
MIN_SPAN_LETTERS = 3            # Minimum total letters in span
MAX_TOKENS = 6                  # Maximum tokens per span

# Hint tokens that suggest entity-like spans
HINT_TOKENS_PERSON = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'gen', 'general', 'col', 'colonel',
    'maj', 'major', 'capt', 'captain', 'lt', 'lieutenant', 'sgt', 'sergeant',
    'sen', 'senator', 'rep', 'representative', 'hon', 'honorable',
    'rev', 'reverend', 'fr', 'father', 'sr', 'sister', 'br', 'brother'
}

HINT_TOKENS_ORG = {
    'bureau', 'department', 'dept', 'office', 'committee', 'commission',
    'agency', 'administration', 'division', 'section', 'branch', 'unit',
    'company', 'corp', 'corporation', 'inc', 'incorporated', 'ltd', 'limited',
    'co', 'llc', 'assoc', 'association', 'society', 'institute', 'university',
    'college', 'school', 'foundation', 'organization', 'party', 'union',
    'federal', 'national', 'state', 'central', 'soviet', 'communist'
}

HINT_TOKENS_ALL = HINT_TOKENS_PERSON | HINT_TOKENS_ORG

# Boilerplate zone patterns
BOILERPLATE_PATTERNS = [
    r'^\s*page\s+\d+',
    r'^\s*-\s*\d+\s*-',
    r'^\s*\[\s*deleted\s*\]',
    r'^\s*\[\s*redacted\s*\]',
    r'^\s*unclassified',
    r'^\s*secret\s*$',
    r'^\s*confidential\s*$',
    r'^\s*top\s+secret',
    r'^\s*foipa',
    r'^\s*copy\s+\d+',
    r'^\s*continued',
    r'^\s*\(continued\)',
]

BOILERPLATE_RE = re.compile('|'.join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# OCR garbage patterns
GARBAGE_PATTERNS = [
    r'^[•\.\-_\*\s]+$',           # Punctuation-only
    r'^[0-9\.\-/\s]+$',           # Numeric-only
    r'^.{1,2}$',                   # Too short
    r'(.)\1{3,}',                  # Repeated characters (4+)
    r'[^\w\s]{3,}',               # 3+ consecutive non-word chars
]

GARBAGE_RE = re.compile('|'.join(GARBAGE_PATTERNS))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CandidateSpan:
    """A candidate entity mention span."""
    document_id: int
    chunk_id: int
    page_id: Optional[int]
    char_start: int
    char_end: int
    raw_span: str
    surface_norm: str
    quality_score: float
    doc_quality: str = 'ocr'
    token_count: int = 1
    has_hint_token: bool = False
    in_boilerplate_zone: bool = False


@dataclass
class TokenInfo:
    """Information about a single token."""
    text: str
    start: int
    end: int
    quality: float
    is_hint: bool = False


# =============================================================================
# QUALITY SCORING
# =============================================================================

def compute_token_quality(token: str) -> float:
    """
    Compute quality score for a single token.
    Returns 0-1, higher is better.
    """
    if not token or len(token) < MIN_TOKEN_LENGTH:
        return 0.0
    
    # Check for obvious garbage
    if GARBAGE_RE.search(token):
        return 0.0
    
    total_chars = len(token)
    letters = sum(1 for c in token if c.isalpha())
    digits = sum(1 for c in token if c.isdigit())
    
    # Letter ratio (most important for names)
    letter_ratio = letters / total_chars if total_chars > 0 else 0
    
    # Vowel ratio (English names typically have 20-40% vowels)
    vowels = sum(1 for c in token.lower() if c in 'aeiou')
    vowel_ratio = vowels / letters if letters > 0 else 0
    vowel_score = 1.0 if 0.15 <= vowel_ratio <= 0.50 else 0.5
    
    # Capital structure (proper nouns often start with capital)
    cap_score = 1.0 if token[0].isupper() else 0.7
    
    # Length score (sweet spot is 4-12 chars for names)
    if 4 <= len(token) <= 12:
        length_score = 1.0
    elif len(token) < 4:
        length_score = 0.6
    else:
        length_score = 0.8
    
    # Repeated character penalty
    max_repeat = max(len(list(g)) for _, g in __import__('itertools').groupby(token.lower()))
    repeat_penalty = 0.0 if max_repeat >= 3 else 1.0
    
    # Combine scores
    quality = (
        0.4 * letter_ratio +
        0.2 * vowel_score +
        0.15 * cap_score +
        0.15 * length_score +
        0.1 * repeat_penalty
    )
    
    return min(1.0, max(0.0, quality))


def compute_span_quality(tokens: List[TokenInfo]) -> float:
    """Compute overall quality score for a span of tokens."""
    if not tokens:
        return 0.0
    
    # Average token quality
    avg_quality = sum(t.quality for t in tokens) / len(tokens)
    
    # Bonus for hint tokens
    hint_bonus = 0.1 if any(t.is_hint for t in tokens) else 0.0
    
    # Penalty for very long spans (likely noise)
    length_penalty = 0.0 if len(tokens) <= 4 else 0.1 * (len(tokens) - 4)
    
    # Combined span text
    span_text = ' '.join(t.text for t in tokens)
    
    # Total letter count check
    total_letters = sum(1 for c in span_text if c.isalpha())
    if total_letters < MIN_SPAN_LETTERS:
        return 0.0
    
    return min(1.0, max(0.0, avg_quality + hint_bonus - length_penalty))


def is_boilerplate_line(line: str) -> bool:
    """Check if a line is likely boilerplate (header/footer/etc.)."""
    return bool(BOILERPLATE_RE.match(line.strip()))


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize_with_positions(text: str) -> List[TokenInfo]:
    """
    Tokenize text preserving character positions.
    Also computes quality score for each token.
    """
    tokens = []
    
    # Simple word tokenization (handles most cases)
    pattern = re.compile(r'\b[\w\'-]+\b')
    
    for match in pattern.finditer(text):
        token_text = match.group()
        start = match.start()
        end = match.end()
        
        # Skip very short tokens
        if len(token_text) < MIN_TOKEN_LENGTH:
            continue
        
        quality = compute_token_quality(token_text)
        is_hint = token_text.lower().rstrip('.') in HINT_TOKENS_ALL
        
        tokens.append(TokenInfo(
            text=token_text,
            start=start,
            end=end,
            quality=quality,
            is_hint=is_hint
        ))
    
    return tokens


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def generate_candidate_spans(
    text: str,
    document_id: int,
    chunk_id: int,
    page_id: Optional[int] = None,
    doc_quality: str = 'ocr',
    junk_patterns: Optional[Set[str]] = None
) -> List[CandidateSpan]:
    """
    Generate candidate entity mention spans from text.
    
    Rules:
    - 1-6 tokens per span
    - At least one token must pass quality threshold
    - Span must have minimum total letters
    - Avoid boilerplate zones
    - Prioritize spans with hint tokens
    """
    candidates = []
    junk_patterns = junk_patterns or set()
    
    # Split into lines to detect boilerplate zones
    lines = text.split('\n')
    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append((pos, is_boilerplate_line(line)))
        pos += len(line) + 1  # +1 for newline
    
    def is_in_boilerplate(char_pos: int) -> bool:
        """Check if position is in a boilerplate zone."""
        for i, (start, is_bp) in enumerate(line_starts):
            next_start = line_starts[i+1][0] if i+1 < len(line_starts) else len(text)
            if start <= char_pos < next_start:
                return is_bp
        return False
    
    # Tokenize
    tokens = tokenize_with_positions(text)
    
    if not tokens:
        return candidates
    
    # Generate spans of 1-6 tokens
    for span_len in range(1, MAX_TOKENS + 1):
        for i in range(len(tokens) - span_len + 1):
            span_tokens = tokens[i:i + span_len]
            
            # Check if at least one token passes quality threshold
            if not any(t.quality >= MIN_SPAN_QUALITY for t in span_tokens):
                continue
            
            # Get span boundaries
            char_start = span_tokens[0].start
            char_end = span_tokens[-1].end
            raw_span = text[char_start:char_end]
            
            # Compute span quality
            quality = compute_span_quality(span_tokens)
            if quality < MIN_SPAN_QUALITY:
                continue
            
            # Normalize
            surface_norm = normalize_surface(raw_span)
            if not surface_norm or len(surface_norm) < MIN_SPAN_LETTERS:
                continue
            
            # Check junk patterns
            if surface_norm in junk_patterns:
                continue
            
            # Check boilerplate zone
            in_boilerplate = is_in_boilerplate(char_start)
            
            # Has hint token?
            has_hint = any(t.is_hint for t in span_tokens)
            
            candidates.append(CandidateSpan(
                document_id=document_id,
                chunk_id=chunk_id,
                page_id=page_id,
                char_start=char_start,
                char_end=char_end,
                raw_span=raw_span,
                surface_norm=surface_norm,
                quality_score=quality,
                doc_quality=doc_quality,
                token_count=span_len,
                has_hint_token=has_hint,
                in_boilerplate_zone=in_boilerplate
            ))
    
    return candidates


def dedupe_overlapping_candidates(candidates: List[CandidateSpan]) -> List[CandidateSpan]:
    """
    Remove overlapping candidates, keeping highest quality.
    Uses greedy selection.
    """
    if not candidates:
        return []
    
    # Sort by quality descending, then by span length descending (prefer longer)
    sorted_cands = sorted(
        candidates,
        key=lambda c: (-c.quality_score, -c.token_count)
    )
    
    selected = []
    used_ranges = []  # List of (start, end) tuples
    
    for cand in sorted_cands:
        # Check overlap with already selected
        overlaps = False
        for start, end in used_ranges:
            if not (cand.char_end <= start or cand.char_start >= end):
                overlaps = True
                break
        
        if not overlaps:
            selected.append(cand)
            used_ranges.append((cand.char_start, cand.char_end))
    
    return selected


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def get_conn():
    """Get database connection."""
    return psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='neh',
        user='neh',
        password='neh'
    )


def load_junk_patterns(conn) -> Set[str]:
    """Load active junk patterns from database."""
    cur = conn.cursor()
    cur.execute("""
        SELECT pattern_value 
        FROM ocr_junk_patterns 
        WHERE is_active = TRUE AND pattern_type = 'exact'
    """)
    return {row[0].lower() for row in cur.fetchall()}


def get_chunks_to_process(
    conn,
    collection: Optional[str] = None,
    document_id: Optional[int] = None,
    chunk_id: Optional[int] = None,
    limit: Optional[int] = None,
    ocr_only: bool = True
) -> List[Tuple]:
    """Get chunks to process."""
    cur = conn.cursor()
    
    conditions = []
    params = []
    
    if chunk_id:
        conditions.append("c.id = %s")
        params.append(chunk_id)
    else:
        if collection:
            conditions.append("col.slug = %s")
            params.append(collection)
        if document_id:
            conditions.append("d.id = %s")
            params.append(document_id)
        if ocr_only:
            # For now, consider silvermaster, rosenberg as OCR-heavy
            conditions.append("col.slug IN ('silvermaster', 'rosenberg', 'solo', 'fbicomrap')")
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
        SELECT c.id, c.text, d.id as document_id, col.slug
        FROM chunks c
        JOIN chunk_pages cp ON cp.chunk_id = c.id
        JOIN pages p ON p.id = cp.page_id
        JOIN documents d ON d.id = p.document_id
        JOIN collections col ON col.id = d.collection_id
        {where_clause}
        GROUP BY c.id, c.text, d.id, col.slug
        ORDER BY c.id
        {limit_clause}
    """
    
    cur.execute(query, params)
    return cur.fetchall()


def insert_candidates(conn, candidates: List[CandidateSpan], batch_id: str):
    """Insert candidates into mention_candidates table."""
    if not candidates:
        return 0
    
    cur = conn.cursor()
    
    insert_sql = """
        INSERT INTO mention_candidates (
            document_id, chunk_id, page_id,
            char_start, char_end,
            raw_span, surface_norm,
            quality_score, doc_quality,
            token_count, has_hint_token, in_boilerplate_zone,
            batch_id
        ) VALUES %s
        ON CONFLICT DO NOTHING
    """
    
    values = [
        (
            c.document_id, c.chunk_id, c.page_id,
            c.char_start, c.char_end,
            c.raw_span, c.surface_norm,
            c.quality_score, c.doc_quality,
            c.token_count, c.has_hint_token, c.in_boilerplate_zone,
            batch_id
        )
        for c in candidates
    ]
    
    execute_values(cur, insert_sql, values, page_size=1000)
    conn.commit()
    
    return len(values)


def create_extraction_run(conn, batch_id: str, config: dict) -> int:
    """Create an extraction run record."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_extraction_runs (batch_id, config, status)
        VALUES (%s, %s, 'running')
        RETURNING id
    """, (batch_id, Json(config)))
    conn.commit()
    return cur.fetchone()[0]


def update_extraction_run(conn, batch_id: str, stats: dict):
    """Update extraction run with final stats."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE ocr_extraction_runs
        SET 
            completed_at = NOW(),
            status = 'completed',
            chunks_processed = %s,
            candidates_generated = %s
        WHERE batch_id = %s
    """, (
        stats.get('chunks_processed', 0),
        stats.get('candidates_generated', 0),
        batch_id
    ))
    conn.commit()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract OCR candidate spans')
    parser.add_argument('--collection', help='Collection slug to process')
    parser.add_argument('--document-id', type=int, help='Specific document ID')
    parser.add_argument('--chunk-id', type=int, help='Specific chunk ID')
    parser.add_argument('--limit', type=int, help='Limit number of chunks')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t insert, just report')
    parser.add_argument('--dedupe', action='store_true', default=True, help='Remove overlapping spans')
    args = parser.parse_args()
    
    conn = get_conn()
    
    # Generate batch ID
    batch_id = f"ocr_cand_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    config = {
        'collection': args.collection,
        'document_id': args.document_id,
        'chunk_id': args.chunk_id,
        'limit': args.limit,
        'dry_run': args.dry_run,
        'dedupe': args.dedupe
    }
    
    print(f"=== OCR Candidate Extraction ===")
    print(f"Batch ID: {batch_id}")
    print(f"Config: {config}")
    print()
    
    # Load junk patterns
    junk_patterns = load_junk_patterns(conn)
    print(f"Loaded {len(junk_patterns)} junk patterns")
    
    # Get chunks
    chunks = get_chunks_to_process(
        conn,
        collection=args.collection,
        document_id=args.document_id,
        chunk_id=args.chunk_id,
        limit=args.limit
    )
    print(f"Found {len(chunks)} chunks to process")
    
    if not chunks:
        print("No chunks to process.")
        return
    
    # Create extraction run record
    if not args.dry_run:
        create_extraction_run(conn, batch_id, config)
    
    # Process chunks
    start_time = time.time()
    total_candidates = 0
    chunks_processed = 0
    
    # Stats by quality bucket
    quality_buckets = {
        'high (>0.7)': 0,
        'medium (0.5-0.7)': 0,
        'low (0.3-0.5)': 0
    }
    
    for chunk_id, text, document_id, collection_slug in chunks:
        if not text:
            continue
        
        # Determine doc quality based on collection
        doc_quality = 'clean' if collection_slug in ('venona', 'mccarthy', 'huac_hearings') else 'ocr'
        
        # Generate candidates
        candidates = generate_candidate_spans(
            text=text,
            document_id=document_id,
            chunk_id=chunk_id,
            page_id=None,
            doc_quality=doc_quality,
            junk_patterns=junk_patterns
        )
        
        # Dedupe if requested
        if args.dedupe:
            candidates = dedupe_overlapping_candidates(candidates)
        
        # Track quality distribution
        for c in candidates:
            if c.quality_score > 0.7:
                quality_buckets['high (>0.7)'] += 1
            elif c.quality_score > 0.5:
                quality_buckets['medium (0.5-0.7)'] += 1
            else:
                quality_buckets['low (0.3-0.5)'] += 1
        
        # Insert
        if not args.dry_run and candidates:
            insert_candidates(conn, candidates, batch_id)
        
        total_candidates += len(candidates)
        chunks_processed += 1
        
        if chunks_processed % 100 == 0:
            print(f"  Processed {chunks_processed}/{len(chunks)} chunks, {total_candidates} candidates...")
    
    elapsed = time.time() - start_time
    
    # Update extraction run
    if not args.dry_run:
        update_extraction_run(conn, batch_id, {
            'chunks_processed': chunks_processed,
            'candidates_generated': total_candidates
        })
    
    # Report
    print()
    print("=== EXTRACTION COMPLETE ===")
    print(f"  Chunks processed: {chunks_processed}")
    print(f"  Candidates generated: {total_candidates}")
    print(f"  Avg candidates/chunk: {total_candidates/chunks_processed:.1f}" if chunks_processed > 0 else "  No chunks processed")
    print(f"  Time: {elapsed:.1f}s ({chunks_processed/elapsed:.1f} chunks/sec)" if elapsed > 0 else "")
    print()
    print("  Quality distribution:")
    for bucket, count in quality_buckets.items():
        pct = 100 * count / total_candidates if total_candidates > 0 else 0
        print(f"    {bucket}: {count} ({pct:.1f}%)")
    
    if args.dry_run:
        print()
        print("[DRY RUN] No candidates were inserted.")
    else:
        print()
        print(f"Candidates written to mention_candidates table (batch_id={batch_id})")
    
    conn.close()


if __name__ == '__main__':
    main()
