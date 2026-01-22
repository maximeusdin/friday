"""
OCR-aware text normalization for improved retrieval on noisy corpora.

Normalization versions:
- norm_v1: No normalization (identity function)
- norm_v2: OCR-aware normalization (de-hyphenation, ligatures, OCR confusion mappings)
"""

import re
from typing import Dict


# Ligature mappings (common in OCR)
LIGATURES: Dict[str, str] = {
    'ﬁ': 'fi',
    'ﬂ': 'fl',
    'ﬀ': 'ff',
    'ﬃ': 'ffi',
    'ﬄ': 'ffl',
    'ﬅ': 'ft',
    'ﬆ': 'st',
    'æ': 'ae',
    'œ': 'oe',
    'Æ': 'AE',
    'Œ': 'OE',
}


def normalize_for_fts(text: str, version: str = "norm_v1") -> str:
    """
    Normalize text for full-text search indexing and matching.
    
    Args:
        text: Input text to normalize
        version: Normalization version
            - "norm_v1": No normalization (identity)
            - "norm_v2": OCR-aware normalization
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    if version == "norm_v1":
        return text
    
    if version == "norm_v2":
        return _normalize_v2(text)
    
    raise ValueError(f"Unknown normalization version: {version}")


def _normalize_v2(text: str) -> str:
    """
    OCR-aware normalization (norm_v2).
    
    Applies:
    1. Normalize ligatures (ﬁ → fi) - do this first so de-hyphenation regex works
    2. De-hyphenate line breaks (word-\nword → wordword)
    3. OCR confusion mappings (rn ↔ m, l ↔ I, context-aware)
    4. Collapse whitespace
    5. Preserve word boundaries
    """
    # Step 1: Normalize ligatures first (so de-hyphenation regex can match ASCII letters)
    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)
    
    # Step 2: De-hyphenate line breaks
    # Pattern: word-\nword or word-\r\nword → wordword
    # But preserve intentional hyphens (e.g., "well-known")
    # Use \w to match word characters (letters, digits, underscore) after ligature normalization
    text = re.sub(r'(\w)-\s*\r?\n\s*(\w)', r'\1\2', text, flags=re.IGNORECASE)
    
    # Step 3: OCR confusion mappings (context-aware)
    text = _fix_ocr_confusions(text)
    
    # Step 4: Collapse whitespace (but preserve word boundaries)
    text = re.sub(r'\s+', ' ', text)
    
    # Step 5: Trim
    text = text.strip()
    
    return text


def _fix_ocr_confusions(text: str) -> str:
    """
    Fix common OCR confusion patterns.
    
    Patterns:
    - rn ↔ m (at word boundaries or when context suggests OCR error)
    - l ↔ I (lowercase l vs uppercase I, context-aware)
    
    Strategy: Be conservative - only fix when context strongly suggests OCR error.
    """
    # Pattern 1: rn → m (when rn appears in positions where m is more likely)
    # Common OCR error: "rn" scanned as "m" (e.g., "silverrnaster" → "silvermaster")
    # But also: "rn" at word boundaries might be legitimate (e.g., "burn")
    # We'll be conservative and only fix when:
    # - rn appears in middle of word (not at start/end)
    # - And surrounded by lowercase letters (suggests OCR error, not intentional)
    
    # Pattern: lowercase letter + rn + lowercase letter (middle of word)
    # But avoid common words like "burn", "turn", "learn", etc.
    # Actually, let's be even more conservative: only fix if it's clearly wrong
    # For now, we'll skip automatic rn→m fixing and rely on soft lex matching
    
    # Pattern 2: l ↔ I confusion
    # This is tricky - need context about case
    # For now, we'll normalize: if we see "l" in all-caps context, it might be "I"
    # But this is risky, so we'll skip it for now
    
    # For Phase 3, we'll focus on de-hyphenation and ligatures
    # OCR confusion mappings can be added later with more testing
    
    return text


def normalize_query_term(term: str, version: str = "norm_v1") -> str:
    """
    Normalize a single query term for matching.
    
    This is a convenience function that applies the same normalization
    as normalize_for_fts but is optimized for single terms.
    """
    return normalize_for_fts(term, version)


def test_normalization():
    """Simple test function to verify normalization works."""
    # Test norm_v1 (identity)
    assert normalize_for_fts("test", "norm_v1") == "test"
    
    # Test norm_v2: de-hyphenation
    assert normalize_for_fts("word-\nword", "norm_v2") == "wordword"
    assert normalize_for_fts("word-\r\nword", "norm_v2") == "wordword"
    
    # Test norm_v2: ligatures
    assert normalize_for_fts("ﬁle", "norm_v2") == "file"
    assert normalize_for_fts("ﬂow", "norm_v2") == "flow"
    
    # Test norm_v2: whitespace collapse
    assert normalize_for_fts("word   word", "norm_v2") == "word word"
    assert normalize_for_fts("  word  word  ", "norm_v2") == "word word"
    
    print("✅ All normalization tests passed!")


if __name__ == "__main__":
    test_normalization()
