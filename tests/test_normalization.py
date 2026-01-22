#!/usr/bin/env python3
"""
Unit tests for OCR-aware text normalization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.normalization import normalize_for_fts, normalize_query_term


def test_norm_v1_identity():
    """Test that norm_v1 is identity function."""
    assert normalize_for_fts("test", "norm_v1") == "test"
    assert normalize_for_fts("word   word", "norm_v1") == "word   word"
    assert normalize_for_fts("ﬁle", "norm_v1") == "ﬁle"
    assert normalize_for_fts("word-\nword", "norm_v1") == "word-\nword"
    print("✅ norm_v1 (identity) tests passed")


def test_norm_v2_dehyphenation():
    """Test de-hyphenation of line breaks."""
    # Unix line break
    assert normalize_for_fts("word-\nword", "norm_v2") == "wordword"
    # Windows line break
    assert normalize_for_fts("word-\r\nword", "norm_v2") == "wordword"
    # With spaces
    assert normalize_for_fts("word-  \n  word", "norm_v2") == "wordword"
    # Multiple occurrences
    assert normalize_for_fts("word-\nword-\nword", "norm_v2") == "wordwordword"
    # Preserve intentional hyphens (not at line breaks)
    assert normalize_for_fts("well-known", "norm_v2") == "well-known"
    print("✅ De-hyphenation tests passed")


def test_norm_v2_ligatures():
    """Test ligature normalization."""
    assert normalize_for_fts("ﬁle", "norm_v2") == "file"
    assert normalize_for_fts("ﬂow", "norm_v2") == "flow"
    assert normalize_for_fts("oﬃce", "norm_v2") == "office"
    assert normalize_for_fts("æther", "norm_v2") == "aether"
    assert normalize_for_fts("cœur", "norm_v2") == "coeur"
    # Mixed case
    assert normalize_for_fts("Æther", "norm_v2") == "AEther"
    print("✅ Ligature normalization tests passed")


def test_norm_v2_whitespace():
    """Test whitespace collapsing."""
    assert normalize_for_fts("word   word", "norm_v2") == "word word"
    assert normalize_for_fts("  word  word  ", "norm_v2") == "word word"
    assert normalize_for_fts("word\tword", "norm_v2") == "word word"
    assert normalize_for_fts("word\nword", "norm_v2") == "word word"
    print("✅ Whitespace collapsing tests passed")


def test_norm_v2_combined():
    """Test combined normalization."""
    # Real-world OCR example
    text = "silver-\nmaster"
    normalized = normalize_for_fts(text, "norm_v2")
    assert normalized == "silvermaster"
    
    # With ligatures
    text = "oﬃ-\nce"
    normalized = normalize_for_fts(text, "norm_v2")
    assert normalized == "office"
    
    # Complex example
    text = "  ﬁle-\n  ﬂow  "
    normalized = normalize_for_fts(text, "norm_v2")
    assert normalized == "file flow"
    print("✅ Combined normalization tests passed")


def test_normalize_query_term():
    """Test query term normalization."""
    assert normalize_query_term("test", "norm_v1") == "test"
    assert normalize_query_term("silver-\nmaster", "norm_v2") == "silvermaster"
    assert normalize_query_term("ﬁle", "norm_v2") == "file"
    print("✅ Query term normalization tests passed")


def test_edge_cases():
    """Test edge cases."""
    assert normalize_for_fts("", "norm_v2") == ""
    assert normalize_for_fts("   ", "norm_v2") == ""
    assert normalize_for_fts("a", "norm_v2") == "a"
    assert normalize_for_fts("a-\nb", "norm_v2") == "ab"
    print("✅ Edge case tests passed")


def test_ocr_examples():
    """Test real OCR error examples."""
    # Common OCR errors that soft lex will handle
    # These are examples of what we might see in the corpus
    examples = [
        ("silvermaster", "silvermaster"),  # Clean
        ("silver-\nmaster", "silvermaster"),  # Line break hyphen
        ("silvermastre", "silvermastre"),  # OCR typo (soft lex will match)
    ]
    
    for input_text, expected in examples:
        result = normalize_for_fts(input_text, "norm_v2")
        # Note: We're not fixing "silvermastre" → "silvermaster" automatically
        # That's what soft lex matching is for
        if "-\n" in input_text:
            assert result == expected, f"Failed: {input_text} → {result} (expected {expected})"
    
    print("✅ OCR example tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running normalization tests...\n")
    
    test_norm_v1_identity()
    test_norm_v2_dehyphenation()
    test_norm_v2_ligatures()
    test_norm_v2_whitespace()
    test_norm_v2_combined()
    test_normalize_query_term()
    test_edge_cases()
    test_ocr_examples()
    
    print("\n✅ All normalization tests passed!")


if __name__ == "__main__":
    run_all_tests()
