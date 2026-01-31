#!/usr/bin/env python3
"""
Text hygiene for OCR documents.

Cleans and normalizes text before entity extraction:
- Drops non-letter lines
- Collapses hyphenation
- Normalizes characters (quotes, ligatures, dashes)
- Detects and marks boilerplate
"""

import re
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# Boilerplate keywords (case-insensitive)
BOILERPLATE_KEYWORDS = [
    "deleted", "foipa", "page", "confidential", "classified",
    "top secret", "secret", "restricted", "for official use only",
    "fbi", "bureau", "department of justice"
]

# Common first names for person detection
COMMON_FIRST_NAMES = {
    "john", "james", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "charles", "mary", "patricia", "jennifer", "linda",
    "elizabeth", "barbara", "susan", "jessica", "sarah", "karen", "nancy",
    "lisa", "betty", "margaret", "sandra", "ashley", "kimberly", "emily"
}


def drop_non_letter_lines(text: str, min_letter_ratio: float = 0.3) -> Tuple[str, List[int]]:
    """
    Drop lines that are overwhelmingly non-letter.
    
    Args:
        text: Input text
        min_letter_ratio: Minimum ratio of letters to total chars (default 0.3)
    
    Returns:
        Tuple of (cleaned_text, dropped_line_numbers)
    """
    lines = text.split('\n')
    cleaned_lines = []
    dropped_lines = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            cleaned_lines.append(line)  # Keep empty lines
            continue
        
        total_chars = len([c for c in line if not c.isspace()])
        if total_chars == 0:
            cleaned_lines.append(line)
            continue
        
        letter_chars = len([c for c in line if c.isalpha()])
        letter_ratio = letter_chars / total_chars
        
        if letter_ratio >= min_letter_ratio:
            cleaned_lines.append(line)
        else:
            dropped_lines.append(i + 1)  # 1-indexed
    
    return '\n'.join(cleaned_lines), dropped_lines


def collapse_hyphenation(text: str) -> str:
    """
    Collapse hyphenated words split across lines.
    
    Pattern: word-\nword → wordword or word word
    """
    # Pattern: word followed by hyphen, newline, then word
    # Match: word-\nword
    pattern = r'(\w+)-\s*\n\s*(\w+)'
    
    def replace_hyphen(match):
        word1 = match.group(1)
        word2 = match.group(2)
        # Try to reconstruct: if word1+word2 is a valid word, use that
        # Otherwise, use word1 + " " + word2
        combined = word1 + word2
        # Simple heuristic: if both parts are capitalized or both lowercase, likely one word
        if (word1[0].isupper() and word2[0].isupper()) or (word1[0].islower() and word2[0].islower()):
            return combined
        else:
            return word1 + " " + word2
    
    return re.sub(pattern, replace_hyphen, text)


def normalize_characters(text: str) -> str:
    """
    Normalize quotation marks, ligatures, and dashes.
    """
    # Quotation marks
    text = text.replace('"', '"').replace('"', '"')  # Smart quotes to straight
    text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes to straight
    
    # Ligatures
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('ﬀ', 'ff').replace('ﬃ', 'ffi').replace('ﬄ', 'ffl')
    text = text.replace('æ', 'ae').replace('œ', 'oe')
    
    # Dashes
    text = text.replace('—', '-').replace('–', '-')  # Em/en dashes to hyphen
    text = text.replace('…', '...')  # Ellipsis
    
    # Other common OCR issues
    text = text.replace('°', 'o')  # Degree symbol sometimes misread
    
    return text


def detect_boilerplate_zones(text: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Detect boilerplate zones (headers, footers, etc.).
    
    Returns:
        Dict with 'headers', 'footers', 'keywords' zones
        Each zone is list of (start_line, end_line) tuples
    """
    lines = text.split('\n')
    zones = {
        'headers': [],
        'footers': [],
        'keywords': []
    }
    
    # Check first/last few lines for headers/footers
    header_lines = 3
    footer_lines = 3
    
    # Headers (first few lines)
    if len(lines) > header_lines:
        header_text = '\n'.join(lines[:header_lines]).lower()
        # Check for page numbers, document titles, etc.
        if re.search(r'page\s+\d+', header_text) or any(kw in header_text for kw in BOILERPLATE_KEYWORDS):
            zones['headers'].append((0, header_lines))
    
    # Footers (last few lines)
    if len(lines) > footer_lines:
        footer_text = '\n'.join(lines[-footer_lines:]).lower()
        if re.search(r'page\s+\d+', footer_text) or any(kw in footer_text for kw in BOILERPLATE_KEYWORDS):
            zones['footers'].append((len(lines) - footer_lines, len(lines)))
    
    # Keyword-based detection (anywhere in document)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for keyword in BOILERPLATE_KEYWORDS:
            if keyword in line_lower:
                # Mark line and surrounding context
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                zones['keywords'].append((start, end))
                break
    
    return zones


def is_in_boilerplate_zone(line_num: int, zones: Dict[str, List[Tuple[int, int]]]) -> bool:
    """Check if a line number is in a boilerplate zone."""
    for zone_type, zone_list in zones.items():
        for start, end in zone_list:
            if start <= line_num < end:
                return True
    return False


def clean_text(
    text: str,
    min_letter_ratio: float = 0.3,
    collapse_hyphens: bool = True,
    normalize_chars: bool = True,
    detect_boilerplate: bool = True
) -> Dict[str, any]:
    """
    Apply all text hygiene steps.
    
    Returns:
        Dict with:
        - 'cleaned_text': cleaned text
        - 'dropped_lines': list of dropped line numbers
        - 'boilerplate_zones': detected boilerplate zones
        - 'stats': statistics about cleaning
    """
    original_lines = len(text.split('\n'))
    original_chars = len(text)
    
    # Step 1: Normalize characters
    if normalize_chars:
        text = normalize_characters(text)
    
    # Step 2: Collapse hyphenation
    if collapse_hyphens:
        text = collapse_hyphenation(text)
    
    # Step 3: Drop non-letter lines
    text, dropped_lines = drop_non_letter_lines(text, min_letter_ratio)
    
    # Step 4: Detect boilerplate
    boilerplate_zones = {}
    if detect_boilerplate:
        boilerplate_zones = detect_boilerplate_zones(text)
    
    cleaned_lines = len(text.split('\n'))
    cleaned_chars = len(text)
    
    stats = {
        'original_lines': original_lines,
        'cleaned_lines': cleaned_lines,
        'dropped_lines_count': len(dropped_lines),
        'original_chars': original_chars,
        'cleaned_chars': cleaned_chars,
        'chars_removed': original_chars - cleaned_chars
    }
    
    return {
        'cleaned_text': text,
        'dropped_lines': dropped_lines,
        'boilerplate_zones': boilerplate_zones,
        'stats': stats
    }


if __name__ == "__main__":
    # Test
    test_text = """
    This is a test line with normal text.
    ••••••••••••••••••••••••••••••••••••••••••
    Another normal line here.
    Page 1 of 10
    FOIPA Document
    """
    
    result = clean_text(test_text)
    print("Cleaned text:")
    print(result['cleaned_text'])
    print("\nDropped lines:", result['dropped_lines'])
    print("\nBoilerplate zones:", result['boilerplate_zones'])
    print("\nStats:", result['stats'])
