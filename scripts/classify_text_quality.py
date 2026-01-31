#!/usr/bin/env python3
"""
Classify text quality (OCR vs clean) for chunks or documents.

Heuristics:
- OCR indicators: common OCR errors, inconsistent spacing, character confusion
- Clean text indicators: proper capitalization, consistent punctuation, well-formed sentences
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from retrieval.ops import get_conn

# Common OCR error patterns
OCR_ERROR_PATTERNS = [
    (r'\brn\b', 'm'),  # "rn" often OCR'd as "m"
    (r'\bcl\b', 'd'),  # "cl" often OCR'd as "d"
    (r'\bvv\b', 'w'),  # "vv" often OCR'd as "w"
    (r'\bli\b', 'h'),  # "li" often OCR'd as "h"
    (r'\btlie\b', 'the'),  # Common word split
    (r'\bTlie\b', 'The'),
]

# Character confusion patterns
CHAR_CONFUSION_PATTERNS = [
    (r'[0O]', '0/O'),  # Zero vs O
    (r'[1lI]', '1/l/I'),  # One vs lowercase L vs uppercase I
]

# Common OCR errors dictionary for quick lookup
COMMON_OCR_ERRORS = {
    'rn': 'm',
    'cl': 'd',
    'vv': 'w',
    'li': 'h',
    'tlie': 'the',
    'Tlie': 'The',
}


def count_ocr_errors(text: str) -> int:
    """Count potential OCR errors in text."""
    count = 0
    text_lower = text.lower()
    
    # Check for common OCR error patterns
    for error, correction in COMMON_OCR_ERRORS.items():
        if error in text_lower:
            count += text_lower.count(error)
    
    # Check for character confusion
    for pattern, desc in CHAR_CONFUSION_PATTERNS:
        matches = len(re.findall(pattern, text))
        if matches > len(text) * 0.05:  # More than 5% confusion
            count += matches * 0.1
    
    return count


def detect_spacing_issues(text: str) -> int:
    """Detect inconsistent spacing patterns."""
    issues = 0
    
    # Multiple spaces
    if re.search(r' {2,}', text):
        issues += len(re.findall(r' {2,}', text))
    
    # Inconsistent spacing around punctuation
    if re.search(r'[.,;:]\S', text):  # Punctuation without space after
        issues += len(re.findall(r'[.,;:]\S', text)) * 0.5
    
    # Missing spaces between words
    if re.search(r'[a-z][A-Z]', text):  # Lowercase followed by uppercase (likely missing space)
        issues += len(re.findall(r'[a-z][A-Z]', text)) * 0.3
    
    return issues


def check_capitalization(text: str) -> float:
    """Check proper capitalization (returns score 0-1)."""
    if not text:
        return 0.0
    
    sentences = re.split(r'[.!?]\s+', text)
    if not sentences:
        return 0.5
    
    proper_caps = 0
    total_sentences = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 3:
            continue
        
        total_sentences += 1
        # Check if sentence starts with capital
        if sentence and sentence[0].isupper():
            proper_caps += 1
    
    if total_sentences == 0:
        return 0.5
    
    return proper_caps / total_sentences


def check_punctuation(text: str) -> float:
    """Check punctuation consistency (returns score 0-1)."""
    if not text:
        return 0.0
    
    # Count punctuation marks
    punct_count = len(re.findall(r'[.,!?;:]', text))
    char_count = len(re.findall(r'[a-zA-Z]', text))
    
    if char_count == 0:
        return 0.5
    
    # Normal punctuation rate is roughly 1 per 10-20 words
    # Too low or too high suggests issues
    punct_ratio = punct_count / char_count
    
    # Ideal range: 0.05 to 0.15
    if 0.05 <= punct_ratio <= 0.15:
        return 1.0
    elif 0.02 <= punct_ratio <= 0.25:
        return 0.7
    else:
        return 0.3


def classify_text_quality(text: str) -> str:
    """
    Classify text quality as 'ocr', 'clean', or 'unknown'.
    
    Returns:
        'ocr': Likely OCR-scanned text with errors
        'clean': High-quality text without OCR artifacts
        'unknown': Cannot determine with confidence
    """
    if not text or len(text.strip()) < 10:
        return 'unknown'
    
    # Check for PDF metadata (indicates binary/format data, not readable text)
    pdf_indicators = ['%PDF', '/Linearized', '/Root', '/Info', '/Type', '/Catalog', 'obj', 'endobj', 'xref', 'trailer']
    if any(indicator in text[:1000] for indicator in pdf_indicators):
        # This is PDF format data, not readable text
        # Try to detect if it's OCR'd PDF text vs raw PDF
        # If it has readable sentences mixed with PDF metadata, might be OCR output
        readable_portion = len(re.findall(r'\b[a-z]{3,}\b', text.lower())) / max(len(text.split()), 1)
        if readable_portion < 0.3:  # Less than 30% readable words = likely raw PDF
            return 'unknown'  # Can't classify raw PDF format data
    
    # Calculate scores
    ocr_score = 0.0
    clean_score = 0.0
    
    # OCR indicators
    ocr_errors = count_ocr_errors(text)
    ocr_score += min(ocr_errors / max(len(text) / 100, 1), 1.0) * 0.4
    
    spacing_issues = detect_spacing_issues(text)
    ocr_score += min(spacing_issues / max(len(text) / 100, 1), 1.0) * 0.2
    
    # Clean text indicators
    cap_score = check_capitalization(text)
    clean_score += cap_score * 0.4
    
    punct_score = check_punctuation(text)
    clean_score += punct_score * 0.3
    
    # Well-formed sentences (simple heuristic)
    sentence_count = len(re.split(r'[.!?]\s+', text))
    word_count = len(re.findall(r'\b\w+\b', text))
    if sentence_count > 0 and word_count > 0:
        avg_words_per_sentence = word_count / sentence_count
        if 5 <= avg_words_per_sentence <= 30:  # Reasonable sentence length
            clean_score += 0.3
    
    # Determine classification
    if ocr_score > 0.3:
        return 'ocr'
    elif clean_score > 0.6:
        return 'clean'
    else:
        return 'unknown'


def classify_chunks(
    conn,
    collection_slug: Optional[str] = None,
    document_id: Optional[int] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
    update_chunks: bool = True
) -> Tuple[int, dict]:
    """Classify chunks and optionally update database."""
    cur = conn.cursor()
    
    # Build query
    conditions = []
    params = []
    
    if collection_slug:
        conditions.append("""
            EXISTS (
                SELECT 1 FROM chunk_metadata cm
                JOIN documents d ON cm.document_id = d.id
                JOIN collections c ON d.collection_id = c.id
                WHERE cm.chunk_id = chunks.id AND c.slug = %s
            )
        """)
        params.append(collection_slug)
    
    if document_id:
        conditions.append("""
            EXISTS (
                SELECT 1 FROM chunk_metadata cm
                WHERE cm.chunk_id = chunks.id AND cm.document_id = %s
            )
        """)
        params.append(document_id)
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
        SELECT c.id, c.text, c.text_quality
        FROM chunks c
        WHERE {where_clause}
        ORDER BY c.id
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query, params)
    chunks = cur.fetchall()
    
    classifications = {'ocr': 0, 'clean': 0, 'unknown': 0, 'unchanged': 0}
    updates = []
    
    for chunk_id, text, current_quality in chunks:
        quality = classify_text_quality(text)
        classifications[quality] += 1
        
        if current_quality != quality:
            updates.append((chunk_id, quality))
        else:
            classifications['unchanged'] += 1
    
    # Update database if not dry run
    if not dry_run and update_chunks and updates:
        cur.executemany(
            "UPDATE chunks SET text_quality = %s WHERE id = %s",
            [(quality, chunk_id) for chunk_id, quality in updates]
        )
        conn.commit()
        print(f"Updated {len(updates)} chunks", file=sys.stderr)
    else:
        print(f"Would update {len(updates)} chunks", file=sys.stderr)
    
    return len(chunks), classifications


def main():
    parser = argparse.ArgumentParser(
        description="Classify text quality (OCR vs clean) for chunks"
    )
    parser.add_argument(
        "--collection",
        help="Collection slug to classify chunks for"
    )
    parser.add_argument(
        "--document-id",
        type=int,
        help="Document ID to classify chunks for"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of chunks to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't update database, just show what would be done"
    )
    parser.add_argument(
        "--update-chunks",
        action="store_true",
        default=True,
        help="Update chunks.text_quality column (default: True)"
    )
    parser.add_argument(
        "--test-text",
        help="Test classification on provided text string"
    )
    
    args = parser.parse_args()
    
    if args.test_text:
        quality = classify_text_quality(args.test_text)
        print(f"Classification: {quality}")
        print(f"Text: {args.test_text[:100]}...")
        return
    
    conn = get_conn()
    try:
        count, classifications = classify_chunks(
            conn,
            collection_slug=args.collection,
            document_id=args.document_id,
            limit=args.limit,
            dry_run=args.dry_run,
            update_chunks=args.update_chunks
        )
        
        print(f"\nProcessed {count} chunks:", file=sys.stderr)
        print(f"  OCR: {classifications['ocr']}", file=sys.stderr)
        print(f"  Clean: {classifications['clean']}", file=sys.stderr)
        print(f"  Unknown: {classifications['unknown']}", file=sys.stderr)
        print(f"  Unchanged: {classifications['unchanged']}", file=sys.stderr)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
