#!/usr/bin/env python3
"""
Quick test script for normalization module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.normalization import normalize_for_fts, normalize_query_term


def main():
    print("Testing normalization module...\n")
    
    # Test norm_v1 (identity)
    print("1. Testing norm_v1 (identity):")
    assert normalize_for_fts("test", "norm_v1") == "test"
    print("   ✅ norm_v1 works")
    
    # Test norm_v2 de-hyphenation
    print("\n2. Testing norm_v2 de-hyphenation:")
    result = normalize_for_fts("silver-\nmaster", "norm_v2")
    assert result == "silvermaster"
    print(f"   ✅ 'silver-\\nmaster' → '{result}'")
    
    # Test norm_v2 ligatures
    print("\n3. Testing norm_v2 ligatures:")
    result = normalize_for_fts("ﬁle", "norm_v2")
    assert result == "file"
    print(f"   ✅ 'ﬁle' → '{result}'")
    
    result = normalize_for_fts("ﬂow", "norm_v2")
    assert result == "flow"
    print(f"   ✅ 'ﬂow' → '{result}'")
    
    # Test norm_v2 whitespace
    print("\n4. Testing norm_v2 whitespace collapsing:")
    result = normalize_for_fts("word   word", "norm_v2")
    assert result == "word word"
    print(f"   ✅ 'word   word' → '{result}'")
    
    # Test combined
    print("\n5. Testing combined normalization:")
    result = normalize_for_fts("  ﬁle-\n  ﬂow  ", "norm_v2")
    # After de-hyphenation: ﬁle-\nﬂow → fileflow (one word)
    # After whitespace collapse: "  fileflow  " → "fileflow"
    assert result == "fileflow"
    print(f"   ✅ '  ﬁle-\\n  ﬂow  ' → '{result}'")
    
    # Test that words separated by space remain separate
    result = normalize_for_fts("  ﬁle  ﬂow  ", "norm_v2")
    assert result == "file flow"
    print(f"   ✅ '  ﬁle  ﬂow  ' → '{result}'")
    
    # Test query term normalization
    print("\n6. Testing query term normalization:")
    result = normalize_query_term("silver-\nmaster", "norm_v2")
    assert result == "silvermaster"
    print(f"   ✅ Query term normalization works")
    
    print("\n✅ All normalization tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
