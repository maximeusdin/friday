#!/usr/bin/env python3
"""
extract_date_mentions.py [options]

Extracts explicit date expressions from chunks using deterministic regex patterns.

MVP patterns:
- "23 June 1945", "June 23, 1945" (day precision)
- "Jun 1945", "June 1945" (month precision)
- "1945" (year precision)
- "June–July 1945", "1943–1945" (ranges)

Persists to date_mentions table with:
- method='regex_day', 'regex_month', 'regex_year', 'regex_range'
- Fixed confidence per rule family (day=1.0, month=0.8, year=0.6, range=0.9)
- Idempotent inserts (ON CONFLICT DO NOTHING)

Usage:
  python scripts/extract_date_mentions.py --collection venona --dry-run --limit 10
"""

import os
import sys
import argparse
import re
import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import execute_values

from retrieval.ops import get_conn


# =============================================================================
# Date parsing utilities
# =============================================================================

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Date patterns (ordered by specificity - most specific first)
DATE_PATTERNS = [
    # Day precision: "23 June 1945", "June 23, 1945", "23rd June 1945"
    (
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)[a-z]*\s+(\d{4})\b',
        'regex_day',
        'day',
        1.0
    ),
    (
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
        'regex_day',
        'day',
        1.0
    ),
    # Month precision: "June 1945", "Jun 1945"
    (
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)[a-z]*\s+(\d{4})\b',
        'regex_month',
        'month',
        0.8
    ),
    # Year precision: "1945" (only in date-like contexts)
    # Match years that appear:
    # - After date prepositions: "in 1945", "during 1945", "by 1945" (but NOT "from page 1945")
    # - With date words: "year 1945", "dated 1945"
    # - Standalone with punctuation: "1945," "1945."
    # - NOT: "1945-1234" (reference), "page 1945" (page number), "No. 1945" (document number)
    (
        r'(?i)\b(?:(?:in|during|by|on|until|since)\s+(19\d{2}|20\d{2})|(?:year|dated|date)\s+(19\d{2}|20\d{2}))\b',
        'regex_year',
        'year',
        0.6
    ),
    # Standalone year with punctuation (but not part of reference numbers or page numbers)
    (
        r'(?<![-/\d])\b(19\d{2}|20\d{2})(?=[,.\s;:]|$)(?![-/\d])',
        'regex_year',
        'year',
        0.5  # Lower confidence for standalone years
    ),
    # Numeric dates: "23/06/1945", "06/23/1945" (assume DD/MM/YYYY or MM/DD/YYYY)
    (
        r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
        'regex_day',
        'day',
        1.0
    ),
]

# Range patterns
RANGE_PATTERNS = [
    # Month range: "June–July 1945", "June-July 1945"
    (
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)[a-z]*\s*[–\-]\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)[a-z]*\s+(\d{4})\b',
        'regex_range',
        'range',
        0.9
    ),
    # Year range: "1943–1945", "1943-1945"
    (
        r'\b(19\d{2}|20\d{2})\s*[–\-]\s*(19\d{2}|20\d{2})\b',
        'regex_range',
        'range',
        0.9
    ),
]


@dataclass
class DateMatch:
    """Represents a matched date expression."""
    surface: str
    start_char: int
    end_char: int
    date_start: Optional[datetime.date]
    date_end: Optional[datetime.date]
    precision: str
    confidence: float
    method: str


def parse_day_date(match, pattern_type: str) -> Optional[DateMatch]:
    """Parse a day-precision date match."""
    groups = match.groups()
    surface = match.group(0)
    
    if pattern_type == 'regex_day' and len(groups) == 3:
        # Check if it's numeric format (DD/MM/YYYY or MM/DD/YYYY)
        if '/' in surface:
            try:
                d1, d2, year = map(int, groups)
                
                if 1900 <= year <= 2100:
                    # Try DD/MM/YYYY first
                    if 1 <= d1 <= 31 and 1 <= d2 <= 12:
                        try:
                            date_start = datetime.date(year, d2, d1)
                            return DateMatch(
                                surface=surface,
                                start_char=match.start(),
                                end_char=match.end(),
                                date_start=date_start,
                                date_end=date_start,
                                precision='day',
                                confidence=1.0,
                                method='regex_day'
                            )
                        except ValueError:
                            pass
                    
                    # Try MM/DD/YYYY
                    if 1 <= d1 <= 12 and 1 <= d2 <= 31:
                        try:
                            date_start = datetime.date(year, d1, d2)
                            return DateMatch(
                                surface=surface,
                                start_char=match.start(),
                                end_char=match.end(),
                                date_start=date_start,
                                date_end=date_start,
                                precision='day',
                                confidence=1.0,
                                method='regex_day'
                            )
                        except ValueError:
                            pass
            except ValueError:
                pass
        else:
            # Textual format: "23 June 1945" or "June 23, 1945"
            # Determine format by checking which group is the month name
            try:
                g0, g1, g2 = groups
                
                # Check if first group is a month name
                if g0.lower()[:3] in MONTHS:
                    # Format: "June 23, 1945" (month, day, year)
                    month_str, day_str, year_str = g0, g1, g2
                    month = MONTHS.get(month_str.lower()[:3])
                    day = int(re.sub(r'\D', '', day_str))
                    year = int(year_str)
                elif g1.lower()[:3] in MONTHS:
                    # Format: "23 June 1945" (day, month, year)
                    day_str, month_str, year_str = g0, g1, g2
                    day = int(re.sub(r'\D', '', day_str))
                    month = MONTHS.get(month_str.lower()[:3])
                    year = int(year_str)
                else:
                    return None
                
                if month and 1 <= day <= 31 and 1900 <= year <= 2100:
                    try:
                        date_start = datetime.date(year, month, day)
                        return DateMatch(
                            surface=surface,
                            start_char=match.start(),
                            end_char=match.end(),
                            date_start=date_start,
                            date_end=date_start,
                            precision='day',
                            confidence=1.0,
                            method='regex_day'
                        )
                    except ValueError:
                        pass
            except (ValueError, IndexError, AttributeError):
                pass
    
    return None


def parse_month_date(match) -> Optional[DateMatch]:
    """Parse a month-precision date match."""
    groups = match.groups()
    
    if len(groups) == 2:
        month_str, year_str = groups
        month = MONTHS.get(month_str.lower()[:3])
        year = int(year_str)
        
        if month and 1900 <= year <= 2100:
            # Use first day of month as date_start, last day as date_end
            try:
                date_start = datetime.date(year, month, 1)
                # Get last day of month
                if month == 12:
                    date_end = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
                else:
                    date_end = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
                
                return DateMatch(
                    surface=match.group(0),
                    start_char=match.start(),
                    end_char=match.end(),
                    date_start=date_start,
                    date_end=date_end,
                    precision='month',
                    confidence=0.8,
                    method='regex_month'
                )
            except ValueError:
                pass
    
    return None


def parse_year_date(match) -> Optional[DateMatch]:
    """Parse a year-precision date match."""
    groups = match.groups()
    
    # Handle patterns with multiple groups (e.g., "in 1945" pattern has 2 groups, one may be None)
    year_str = None
    for g in groups:
        if g is not None and g.isdigit():
            year_str = g
            break
    
    if year_str is None and len(groups) == 1:
        year_str = groups[0]
    
    if year_str:
        try:
            year = int(year_str)
            
            if 1900 <= year <= 2100:
                # Use January 1 as date_start, December 31 as date_end
                date_start = datetime.date(year, 1, 1)
                date_end = datetime.date(year, 12, 31)
                
                return DateMatch(
                    surface=match.group(0),
                    start_char=match.start(),
                    end_char=match.end(),
                    date_start=date_start,
                    date_end=date_end,
                    precision='year',
                    confidence=0.6,
                    method='regex_year'
                )
        except (ValueError, TypeError):
            pass
    
    return None


def parse_range_date(match, pattern_type: str) -> Optional[DateMatch]:
    """Parse a range-precision date match."""
    groups = match.groups()
    
    if pattern_type == 'regex_range':
        if len(groups) == 3:
            # Month range: "June–July 1945"
            month1_str, month2_str, year_str = groups
            month1 = MONTHS.get(month1_str.lower()[:3])
            month2 = MONTHS.get(month2_str.lower()[:3])
            year = int(year_str)
            
            if month1 and month2 and 1900 <= year <= 2100:
                try:
                    date_start = datetime.date(year, month1, 1)
                    if month2 == 12:
                        date_end = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
                    else:
                        date_end = datetime.date(year, month2 + 1, 1) - datetime.timedelta(days=1)
                    
                    return DateMatch(
                        surface=match.group(0),
                        start_char=match.start(),
                        end_char=match.end(),
                        date_start=date_start,
                        date_end=date_end,
                        precision='range',
                        confidence=0.9,
                        method='regex_range'
                    )
                except ValueError:
                    pass
        elif len(groups) == 2:
            # Year range: "1943–1945"
            year1_str, year2_str = groups
            year1 = int(year1_str)
            year2 = int(year2_str)
            
            if 1900 <= year1 <= 2100 and 1900 <= year2 <= 2100 and year1 <= year2:
                date_start = datetime.date(year1, 1, 1)
                date_end = datetime.date(year2, 12, 31)
                
                return DateMatch(
                    surface=match.group(0),
                    start_char=match.start(),
                    end_char=match.end(),
                    date_start=date_start,
                    date_end=date_end,
                    precision='range',
                    confidence=0.9,
                    method='regex_range'
                )
    
    return None


def is_false_positive_year(text: str, start: int, end: int) -> bool:
    """
    Check if a year match is likely a false positive (page number, reference number, etc.).
    
    Returns True if this looks like a false positive and should be excluded.
    """
    # Check context around the match
    context_start = max(0, start - 20)
    context_end = min(len(text), end + 20)
    context = text[context_start:context_end].lower()
    match_text = text[start:end]
    
    # Exclude if preceded by common non-date words
    false_positive_prefixes = [
        'page', 'p.', 'pp.', 'no.', 'number', 'ref', 'reference', 
        'doc', 'document', 'file', 'item', 'entry', 'line'
    ]
    
    for prefix in false_positive_prefixes:
        # Check if prefix appears before the year (within reasonable distance)
        prefix_pos = context.rfind(prefix, 0, start - context_start + len(match_text))
        if prefix_pos >= 0:
            # Check if there's a reasonable gap (not too far)
            if (start - context_start) - prefix_pos < 15:
                return True
    
    # Exclude if followed by dash and digits (reference number pattern)
    if end < len(text) and text[end] == '-' and end + 1 < len(text):
        next_chars = text[end+1:end+5]
        if next_chars.isdigit():
            return True
    
    # Exclude if preceded by dash and digits
    if start > 0 and text[start-1] == '-' and start - 5 >= 0:
        prev_chars = text[start-5:start-1]
        if prev_chars.isdigit():
            return True
    
    return False


def extract_dates_from_text(text: str) -> List[DateMatch]:
    """
    Extract date expressions from text using deterministic regex patterns.
    
    Returns list of DateMatch objects, ordered by position.
    Prevents overlapping matches by preferring longer, more specific matches.
    Filters out false positives (page numbers, reference numbers, etc.).
    """
    matches: List[DateMatch] = []
    
    # First, extract all potential matches (ranges first, then single dates)
    all_matches: List[DateMatch] = []
    
    # Extract ranges (more specific, should be checked before single dates)
    for pattern, method, precision, confidence in RANGE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date_match = parse_range_date(match, method)
            if date_match:
                all_matches.append(date_match)
    
    # Extract single dates (check most specific first)
    for pattern, method, precision, confidence in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date_match = None
            if method == 'regex_day':
                date_match = parse_day_date(match, method)
            elif method == 'regex_month':
                date_match = parse_month_date(match)
            elif method == 'regex_year':
                date_match = parse_year_date(match)
                # Filter false positives for year-only matches
                if date_match and is_false_positive_year(text, date_match.start_char, date_match.end_char):
                    date_match = None
            
            if date_match:
                all_matches.append(date_match)
    
    # Filter out overlapping matches: prefer longer, more specific matches
    # Sort by: 1) start position, 2) length (descending), 3) confidence (descending)
    all_matches.sort(key=lambda m: (m.start_char, -(m.end_char - m.start_char), -m.confidence))
    
    for candidate in all_matches:
        # Check if this candidate overlaps with any already-selected match
        overlaps = False
        for selected in matches:
            # Check if candidate overlaps with selected
            # Overlap: candidate start < selected end AND candidate end > selected start
            if (candidate.start_char < selected.end_char and 
                candidate.end_char > selected.start_char):
                overlaps = True
                break
        
        if not overlaps:
            matches.append(candidate)
    
    # Sort by position for final output
    matches.sort(key=lambda m: m.start_char)
    
    return matches


# =============================================================================
# Database utilities
# =============================================================================

def parse_chunk_id_range(range_str: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse chunk ID range string (format: 'start:end')."""
    if ":" not in range_str:
        raise ValueError(f"Invalid chunk-id-range format: {range_str}. Use 'start:end'")
    a, b = range_str.split(":", 1)
    a = a.strip() or None
    b = b.strip() or None
    start = int(a) if a else None
    end = int(b) if b else None
    if start is not None and end is not None and start > end:
        raise ValueError(f"Invalid chunk-id-range: start ({start}) > end ({end})")
    return start, end


def get_chunks_query(
    conn,
    *,
    collection_slug: Optional[str] = None,
    document_id: Optional[int] = None,
    chunk_id_start: Optional[int] = None,
    chunk_id_end: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Tuple[int, str, Optional[int]]]:
    """Get chunks matching criteria."""
    conditions, params = [], []
    if collection_slug:
        conditions.append("cm.collection_slug = %s")
        params.append(collection_slug)
    if document_id:
        conditions.append("cm.document_id = %s")
        params.append(document_id)
    if chunk_id_start is not None:
        conditions.append("c.id >= %s")
        params.append(chunk_id_start)
    if chunk_id_end is not None:
        conditions.append("c.id <= %s")
        params.append(chunk_id_end)
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    limit_clause = f"LIMIT {limit}" if limit else ""
    q = f"""
        SELECT c.id AS chunk_id, c.text, cm.document_id
        FROM chunks c
        LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE {where_clause}
        ORDER BY c.id
        {limit_clause}
    """
    with conn.cursor() as cur:
        cur.execute(q, params)
        return cur.fetchall()


def extract_dates_batch(
    conn,
    chunks: List[Tuple[int, str, Optional[int]]],
    *,
    dry_run: bool = False,
    show_samples: bool = False,
    max_samples: int = 10,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Extract dates from a batch of chunks.
    
    Returns: (chunks_processed, dates_inserted, sample_mentions)
    """
    dates_to_insert: List[Dict[str, Any]] = []
    sample_mentions: List[Dict[str, Any]] = []
    chunks_processed = 0
    
    for chunk_id, text, document_id in chunks:
        chunks_processed += 1
        
        if not text or document_id is None:
            continue
        
        # Extract dates
        date_matches = extract_dates_from_text(text)
        
        for dm in date_matches:
            mention = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "surface": dm.surface,
                "start_char": dm.start_char,
                "end_char": dm.end_char,
                "date_start": dm.date_start,
                "date_end": dm.date_end,
                "precision": dm.precision,
                "confidence": dm.confidence,
                "method": dm.method,
            }
            
            dates_to_insert.append(mention)
            
            # Collect samples
            if show_samples and len(sample_mentions) < max_samples:
                sample_mentions.append(mention)
    
    if dry_run or not dates_to_insert:
        return chunks_processed, 0, sample_mentions
    
    # Insert idempotently using ON CONFLICT
    with conn.cursor() as cur:
        # Check which dates already exist
        check_tuples = [
            (m["chunk_id"], m["surface"], m["date_start"], m["date_end"], m["method"])
            for m in dates_to_insert
        ]
        
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS date_check (
                chunk_id BIGINT,
                surface TEXT,
                date_start DATE,
                date_end DATE,
                method TEXT
            ) ON COMMIT DROP
        """)
        execute_values(cur, "INSERT INTO date_check (chunk_id, surface, date_start, date_end, method) VALUES %s", check_tuples, page_size=1000)
        
        cur.execute("""
            SELECT dc.chunk_id, dc.surface, dc.date_start, dc.date_end, dc.method
            FROM date_check dc
            INNER JOIN date_mentions dm
                ON dm.chunk_id = dc.chunk_id
               AND dm.surface = dc.surface
               AND (dm.date_start = dc.date_start OR (dm.date_start IS NULL AND dc.date_start IS NULL))
               AND (dm.date_end = dc.date_end OR (dm.date_end IS NULL AND dc.date_end IS NULL))
               AND dm.method = dc.method
        """)
        existing = set(cur.fetchall())
        
        new_dates = [
            m for m in dates_to_insert
            if (m["chunk_id"], m["surface"], m["date_start"], m["date_end"], m["method"]) not in existing
        ]
        
        if not new_dates:
            conn.commit()
            return chunks_processed, 0, sample_mentions
        
        # Insert new dates
        execute_values(
            cur,
            """
            INSERT INTO date_mentions
                (chunk_id, document_id, surface, start_char, end_char, date_start, date_end, precision, confidence, method)
            VALUES %s
            ON CONFLICT (chunk_id, surface, date_start, date_end, method) DO NOTHING
            """,
            [
                (
                    m["chunk_id"], m["document_id"], m["surface"],
                    m["start_char"], m["end_char"],
                    m["date_start"], m["date_end"],
                    m["precision"], m["confidence"], m["method"]
                )
                for m in new_dates
            ],
            page_size=1000
        )
    
    conn.commit()
    return chunks_processed, len(new_dates), sample_mentions


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract date mentions from chunks using deterministic regex patterns"
    )
    parser.add_argument("--collection", type=str, help="Filter by collection slug")
    parser.add_argument("--document-id", type=int, help="Filter by document ID")
    parser.add_argument("--chunk-id-range", type=str, help="Filter by chunk ID range (format: start:end)")
    parser.add_argument("--dry-run", action="store_true", help="Print counts only, no inserts")
    parser.add_argument("--show-samples", action="store_true", help="Show sample date mentions (works with --dry-run)")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples to show (default: 10)")
    parser.add_argument("--limit", type=int, help="Limit number of chunks processed")
    parser.add_argument("--batch-size", type=int, default=100, help="Process chunks in batches (default: 100)")
    parser.add_argument("--test-text", type=str, help="Test extraction on provided text string")
    
    args = parser.parse_args()
    
    if not any([args.collection, args.document_id, args.chunk_id_range, args.test_text]):
        parser.error("Must specify at least one of: --collection, --document-id, --chunk-id-range, --test-text")
    
    # Test mode
    if args.test_text:
        matches = extract_dates_from_text(args.test_text)
        print(f"Found {len(matches)} date mentions:")
        for m in matches:
            print(f"  {m.method} ({m.precision}): '{m.surface}' -> {m.date_start} to {m.date_end} (confidence: {m.confidence:.2f})")
        return
    
    chunk_id_start = chunk_id_end = None
    if args.chunk_id_range:
        chunk_id_start, chunk_id_end = parse_chunk_id_range(args.chunk_id_range)
    
    conn = get_conn()
    try:
        # Get chunks
        chunks = get_chunks_query(
            conn,
            collection_slug=args.collection,
            document_id=args.document_id,
            chunk_id_start=chunk_id_start,
            chunk_id_end=chunk_id_end,
            limit=args.limit,
        )
        
        total_chunks = len(chunks)
        if total_chunks == 0:
            print("No chunks found matching criteria", file=sys.stderr)
            return
        
        print(f"Found {total_chunks} chunks to process", file=sys.stderr)
        if args.dry_run:
            print("  [DRY RUN] No changes will be made", file=sys.stderr)
        
        batch_size = args.batch_size
        total_processed = 0
        total_dates = 0
        all_samples: List[Dict[str, Any]] = []
        
        print(f"\nProcessing chunks in batches of {batch_size}...", file=sys.stderr)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            processed, dates, samples = extract_dates_batch(
                conn,
                batch,
                dry_run=args.dry_run,
                show_samples=args.show_samples,
                max_samples=args.max_samples,
            )
            
            total_processed += processed
            total_dates += dates
            
            if args.show_samples:
                all_samples.extend(samples)
                if len(all_samples) >= args.max_samples:
                    all_samples = all_samples[:args.max_samples]
            
            print(
                f"  Batch {batch_num}/{total_batches}: processed {min(i + len(batch), total_chunks)}/{total_chunks} chunks, "
                f"inserted {total_dates} date mentions (+{dates} this batch)",
                file=sys.stderr,
                end="\r",
            )
            sys.stderr.flush()
        
        if total_chunks > 0:
            print(" " * 100, file=sys.stderr, end="\r")
            sys.stderr.flush()
        
        print(f"\n{'='*70}", file=sys.stderr)
        print("SUMMARY:", file=sys.stderr)
        print(f"  ✅ Processed:      {total_processed:>10,} chunks", file=sys.stderr)
        print(f"  📅 Date mentions:  {total_dates:>10,}", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        
        # Show samples
        if args.show_samples and all_samples:
            print(f"\nSample date mentions (showing {len(all_samples)}):", file=sys.stderr)
            for i, m in enumerate(all_samples, 1):
                print(
                    f"  {i}. chunk_id={m['chunk_id']}, surface='{m['surface']}', "
                    f"date={m['date_start']} to {m['date_end']}, "
                    f"precision={m['precision']}, method={m['method']}, confidence={m['confidence']:.2f}",
                    file=sys.stderr
                )
        
        if args.dry_run:
            print("\nRun without --dry-run to actually insert date mentions", file=sys.stderr)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
