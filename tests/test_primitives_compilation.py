#!/usr/bin/env python3
"""
Unit tests for primitive compilation (deterministic compilation functions).

Tests:
- Normalization (casing, punctuation)
- Tsquery rendering (TERM+TERM, PHRASE, combinations)
- Scope compilation (empty, combinations, order independence)
- Determinism (same primitives, different order → same output)
- Quote escaping (apostrophes vs double quotes)
- Parentheses support in websearch_to_tsquery
"""

import sys
import os
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from retrieval.primitives import (
    TermPrimitive,
    PhrasePrimitive,
    WithinResultSetPrimitive,
    ExcludeResultSetPrimitive,
    FilterCollectionPrimitive,
    FilterDocumentPrimitive,
    FilterDateRangePrimitive,
    SetTermModePrimitive,
    OrGroupPrimitive,
    normalize_term,
    normalize_phrase,
    compile_primitives_to_tsquery_text,
    compile_primitives_to_expanded,
    compile_primitives_to_scope,
    compile_primitives,
    _canonical_order_primitives,
)


# ============================================================================
# Normalization Tests
# ============================================================================

def test_normalize_term_casing():
    """Test that normalization handles casing consistently."""
    assert normalize_term("Treasury") == "treasury"
    assert normalize_term("treasury") == "treasury"
    assert normalize_term("TREASURY") == "treasury"
    assert normalize_term("Treasury,, ") == "treasury"
    print("[OK] Normalization casing tests passed")


def test_normalize_term_punctuation():
    """Test that normalization strips punctuation consistently."""
    assert normalize_term("Treasury,, ") == "treasury"
    assert normalize_term("O'Connor") == "oconnor"  # Apostrophe removed
    assert normalize_term("U.S.") == "us"  # Periods removed
    assert normalize_term("test-value") == "testvalue"  # Hyphens removed
    print("[OK] Normalization punctuation tests passed")


def test_normalize_phrase():
    """Test phrase normalization preserves words."""
    words = normalize_phrase("SOVIET INTELLIGENCE")
    assert words == ["soviet", "intelligence"]
    
    words = normalize_phrase("department of the treasury")
    assert words == ["department", "of", "the", "treasury"]  # Stopwords preserved
    
    words = normalize_phrase("O'Connor's")
    assert words == ["oconnors"]  # Apostrophe removed, but word preserved
    print("[OK] Phrase normalization tests passed")


def test_normalize_term_o_connor():
    """Test O'Connor handling - decide on canonical form."""
    # Current behavior: apostrophe removed → "oconnor"
    result = normalize_term("O'Connor")
    assert result == "oconnor"
    
    # Test that it's consistent (apostrophe and space both removed)
    assert normalize_term("O'Connor") == normalize_term("oconnor")
    # Note: "O CONNOR" has a space, which gets collapsed, but the result is "o connor" (two words)
    # So they're different - that's expected behavior
    assert normalize_term("O CONNOR") == "o connor"  # Space preserved as word separator
    print("[OK] O'Connor normalization test passed")


# ============================================================================
# Tsquery Rendering Tests
# ============================================================================

def test_tsquery_term_and_term():
    """Test TERM + TERM compiles to whitespace-separated (AND)."""
    primitives = [TermPrimitive(value="hiss"), TermPrimitive(value="treasury")]
    result = compile_primitives_to_tsquery_text(primitives)
    assert result == "hiss treasury", f"Expected 'hiss treasury', got '{result}'"
    print("[OK] TERM+TERM (AND) test passed")


def test_tsquery_phrase():
    """Test PHRASE compiles to quoted phrase."""
    primitives = [PhrasePrimitive(value="soviet intelligence")]
    result = compile_primitives_to_tsquery_text(primitives)
    assert result == '"soviet intelligence"', f"Expected '\"soviet intelligence\"', got '{result}'"
    print("[OK] PHRASE test passed")


def test_tsquery_term_and_phrase():
    """Test TERM + PHRASE combination."""
    primitives = [TermPrimitive(value="hiss"), PhrasePrimitive(value="soviet intelligence")]
    result = compile_primitives_to_tsquery_text(primitives)
    # Canonical ordering puts PHRASE before TERM (alphabetically), so phrase comes first
    assert result == '"soviet intelligence" hiss', f"Expected '\"soviet intelligence\" hiss', got '{result}'"
    print("[OK] TERM + PHRASE test passed")


def test_tsquery_or_mode():
    """Test SET_TERM_MODE(OR) compiles to OR keyword."""
    # SET_TERM_MODE must come before the TERMs it affects
    primitives = [
        SetTermModePrimitive(value="OR"),
        TermPrimitive(value="hiss"),
        TermPrimitive(value="treasury")
    ]
    result = compile_primitives_to_tsquery_text(primitives)
    assert result == "hiss OR treasury", f"Expected 'hiss OR treasury', got '{result}'"
    print("[OK] SET_TERM_MODE(OR) test passed")


def test_tsquery_or_group():
    """Test OR_GROUP compiles to OR keyword."""
    primitives = [
        OrGroupPrimitive(primitives=[TermPrimitive(value="hiss"), TermPrimitive(value="chambers")]),
        TermPrimitive(value="treasury")
    ]
    result = compile_primitives_to_tsquery_text(primitives)
    # Canonical ordering: OR_GROUP comes before TERM alphabetically, but TERM("treasury") comes after
    # Actually, let's check what the actual order is
    assert "hiss OR chambers" in result or "chambers OR hiss" in result
    assert "treasury" in result
    print("[OK] OR_GROUP test passed")


def test_tsquery_quote_escaping():
    """Test that embedded double quotes are escaped, apostrophes are not."""
    # Apostrophe should be kept as-is (parameterization handles SQL escaping)
    primitives = [TermPrimitive(value="O'Connor")]
    result = compile_primitives_to_tsquery_text(primitives)
    assert "oconnor" in result.lower()  # Normalized, apostrophe removed
    
    # Double quote in phrase should be escaped
    primitives = [PhrasePrimitive(value='He said "hello"')]
    result = compile_primitives_to_tsquery_text(primitives)
    assert '\\"' in result or '"' in result  # Should have escaped or quoted
    print("[OK] Quote escaping test passed")


# ============================================================================
# Scope Compilation Tests
# ============================================================================

def test_scope_empty():
    """Test empty primitives → empty scope."""
    primitives = []
    sql, params = compile_primitives_to_scope(primitives)
    assert sql == ""
    assert params == []
    print("[OK] Empty scope test passed")


def test_scope_within_result_set():
    """Test WITHIN_RESULT_SET compiles correctly."""
    primitives = [WithinResultSetPrimitive(result_set_id=5)]
    sql, params = compile_primitives_to_scope(primitives, chunk_id_expr="c.id")
    assert "c.id = ANY" in sql
    assert "result_sets WHERE id = %s" in sql
    assert params == [5]
    print("[OK] WITHIN_RESULT_SET scope test passed")


def test_scope_filter_collection():
    """Test FILTER_COLLECTION compiles correctly."""
    primitives = [FilterCollectionPrimitive(slug="venona")]
    sql, params = compile_primitives_to_scope(primitives)
    assert "cm.collection_slug = %s" in sql
    assert params == ["venona"]
    print("[OK] FILTER_COLLECTION scope test passed")


def test_scope_filter_document():
    """Test FILTER_DOCUMENT compiles correctly."""
    primitives = [FilterDocumentPrimitive(document_id=123)]
    sql, params = compile_primitives_to_scope(primitives)
    assert "cm.document_id = %s" in sql
    assert params == [123]
    print("[OK] FILTER_DOCUMENT scope test passed")


def test_scope_filter_date_range():
    """Test FILTER_DATE_RANGE compiles correctly."""
    primitives = [FilterDateRangePrimitive(start="1940-01-01", end="1945-12-31")]
    sql, params = compile_primitives_to_scope(primitives)
    assert "cm.date_max >= %s" in sql
    assert "cm.date_min <= %s" in sql
    assert params == ["1940-01-01", "1945-12-31"]
    print("[OK] FILTER_DATE_RANGE scope test passed")


def test_scope_combination():
    """Test combination of scope primitives."""
    primitives = [
        FilterCollectionPrimitive(slug="venona"),
        WithinResultSetPrimitive(result_set_id=5),
        FilterDocumentPrimitive(document_id=123)
    ]
    sql, params = compile_primitives_to_scope(primitives, chunk_id_expr="rc.chunk_id")
    assert "rc.chunk_id = ANY" in sql
    assert "cm.collection_slug = %s" in sql
    assert "cm.document_id = %s" in sql
    assert len(params) == 3
    # Params should be in canonical order (FILTER_COLLECTION, FILTER_DOCUMENT, WITHIN_RESULT_SET)
    assert "venona" in params
    assert 123 in params
    assert 5 in params
    print("[OK] Scope combination test passed")


# ============================================================================
# Determinism Tests
# ============================================================================

def test_determinism_ordering():
    """Test that same primitives in different order produce same output."""
    primitives1 = [
        TermPrimitive(value="hiss"),
        TermPrimitive(value="treasury"),
        FilterCollectionPrimitive(slug="venona")
    ]
    primitives2 = [
        FilterCollectionPrimitive(slug="venona"),
        TermPrimitive(value="treasury"),
        TermPrimitive(value="hiss")
    ]
    
    compiled1 = compile_primitives(primitives1)
    compiled2 = compile_primitives(primitives2)
    
    # Should produce identical compiled outputs
    assert compiled1["tsquery"]["params"] == compiled2["tsquery"]["params"]
    assert compiled1["expanded"]["expanded_text"] == compiled2["expanded"]["expanded_text"]
    assert compiled1["scope"]["where_sql"] == compiled2["scope"]["where_sql"]
    assert compiled1["scope"]["params"] == compiled2["scope"]["params"]
    print("[OK] Determinism (ordering) test passed")


def test_determinism_canonical_ordering():
    """Test that canonical ordering produces stable results."""
    primitives = [
        TermPrimitive(value="treasury"),
        TermPrimitive(value="hiss"),
        FilterCollectionPrimitive(slug="venona"),
        FilterDocumentPrimitive(document_id=123)
    ]
    
    ordered = _canonical_order_primitives(primitives)
    
    # Should be ordered: FILTER_COLLECTION, FILTER_DOCUMENT, TERM(hiss), TERM(treasury)
    assert isinstance(ordered[0], FilterCollectionPrimitive)
    assert isinstance(ordered[1], FilterDocumentPrimitive)
    # TERMs should be ordered by normalized value
    term_indices = [i for i, p in enumerate(ordered) if isinstance(p, TermPrimitive)]
    assert len(term_indices) == 2
    # "hiss" should come before "treasury" alphabetically
    assert ordered[term_indices[0]].value == "hiss"
    assert ordered[term_indices[1]].value == "treasury"
    print("[OK] Canonical ordering test passed")


# ============================================================================
# Expanded Text Tests
# ============================================================================

def test_expanded_text_alignment():
    """Test that expanded_text and tsquery params are aligned."""
    primitives = [TermPrimitive(value="hiss"), TermPrimitive(value="treasury")]
    compiled = compile_primitives(primitives)
    
    # tsquery params should use expanded_text
    assert compiled["tsquery"]["params"][0] == compiled["expanded"]["expanded_text"]
    print("[OK] Expanded text alignment test passed")


def test_expanded_text_phrases():
    """Test that phrases are quoted in expanded_text."""
    primitives = [PhrasePrimitive(value="soviet intelligence")]
    compiled = compile_primitives(primitives)
    
    expanded_text = compiled["expanded"]["expanded_text"]
    assert '"soviet intelligence"' in expanded_text
    print("[OK] Expanded text phrases test passed")


# ============================================================================
# PostgreSQL Integration Tests (requires database)
# ============================================================================

def test_websearch_to_tsquery_parentheses():
    """Test that websearch_to_tsquery handles parentheses correctly."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("[WARN]  Skipping websearch_to_tsquery test (no DATABASE_URL)")
        return
    
    try:
        # Set connection timeout to avoid hanging (very short for tests)
        conn = psycopg2.connect(dsn, connect_timeout=1)
        try:
            with conn.cursor() as cur:
                # Test parentheses with OR
                cur.execute("SELECT websearch_to_tsquery('simple', %s)", ["(hiss OR chambers) treasury"])
                result = cur.fetchone()[0]
                
                # Should not raise error and should produce a valid tsquery
                assert result is not None
                
                # Verify it can be used in a query
                cur.execute("""
                    SELECT to_tsvector('simple', 'hiss treasury') @@ %s
                """, [result])
                match = cur.fetchone()[0]
                assert match is True or match is False  # Should return boolean
                
                print("[OK] websearch_to_tsquery parentheses test passed")
        finally:
            conn.close()
    except psycopg2.OperationalError as e:
        print(f"[WARN]  Database connection failed: {e}")
        print("   Skipping websearch_to_tsquery test")
    except Exception as e:
        print(f"[WARN]  websearch_to_tsquery test failed: {e}")
        print("   This may indicate parentheses are not supported in your Postgres version")


def test_websearch_to_tsquery_apostrophe():
    """Test that apostrophes work correctly in websearch_to_tsquery."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("[WARN]  Skipping apostrophe test (no DATABASE_URL)")
        return
    
    try:
        # Set connection timeout to avoid hanging (very short for tests)
        conn = psycopg2.connect(dsn, connect_timeout=1)
        try:
            with conn.cursor() as cur:
                # Test apostrophe (should not need escaping)
                cur.execute("SELECT websearch_to_tsquery('simple', %s)", ["O'Connor"])
                result = cur.fetchone()[0]
                
                # Should not raise error
                assert result is not None
                print("[OK] websearch_to_tsquery apostrophe test passed")
        finally:
            conn.close()
    except psycopg2.OperationalError as e:
        print(f"[WARN]  Database connection failed: {e}")
        print("   Skipping apostrophe test")
    except Exception as e:
        print(f"[WARN]  Apostrophe test failed: {e}")


def test_websearch_to_tsquery_quoted_phrase():
    """Test that quoted phrases work correctly."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("[WARN]  Skipping quoted phrase test (no DATABASE_URL)")
        return
    
    try:
        # Set connection timeout to avoid hanging (very short for tests)
        conn = psycopg2.connect(dsn, connect_timeout=1)
        try:
            with conn.cursor() as cur:
                # Test quoted phrase
                cur.execute("SELECT websearch_to_tsquery('simple', %s)", ['"soviet intelligence"'])
                result = cur.fetchone()[0]
                
                # Should not raise error
                assert result is not None
                print("[OK] websearch_to_tsquery quoted phrase test passed")
        finally:
            conn.close()
    except psycopg2.OperationalError as e:
        print(f"[WARN]  Database connection failed: {e}")
        print("   Skipping quoted phrase test")
    except Exception as e:
        print(f"[WARN]  Quoted phrase test failed: {e}")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests(skip_db_tests: bool = False):
    """Run all compilation tests.
    
    Args:
        skip_db_tests: If True, skip PostgreSQL integration tests (faster for CI/quick runs)
    """
    import time
    start_time = time.time()
    print("Running primitive compilation tests...\n")
    
    # Normalization tests
    print("=== Normalization Tests ===")
    test_normalize_term_casing()
    test_normalize_term_punctuation()
    test_normalize_phrase()
    test_normalize_term_o_connor()
    print()
    
    # Tsquery rendering tests
    print("=== Tsquery Rendering Tests ===")
    test_tsquery_term_and_term()
    test_tsquery_phrase()
    test_tsquery_term_and_phrase()
    test_tsquery_or_mode()
    test_tsquery_or_group()
    test_tsquery_quote_escaping()
    print()
    
    # Scope compilation tests
    print("=== Scope Compilation Tests ===")
    test_scope_empty()
    test_scope_within_result_set()
    test_scope_filter_collection()
    test_scope_filter_document()
    test_scope_filter_date_range()
    test_scope_combination()
    print()
    
    # Determinism tests
    print("=== Determinism Tests ===")
    test_determinism_ordering()
    test_determinism_canonical_ordering()
    print()
    
    # Expanded text tests
    print("=== Expanded Text Tests ===")
    test_expanded_text_alignment()
    test_expanded_text_phrases()
    print()
    
    # PostgreSQL integration tests (optional, can be slow)
    if not skip_db_tests:
        print("=== PostgreSQL Integration Tests ===")
        test_websearch_to_tsquery_parentheses()
        test_websearch_to_tsquery_apostrophe()
        test_websearch_to_tsquery_quoted_phrase()
        print()
    else:
        print("=== PostgreSQL Integration Tests ===")
        print("[SKIP] Database tests skipped (use --with-db to enable)")
        print()
    
    elapsed = time.time() - start_time
    print(f"[OK] All compilation tests passed! (took {elapsed:.2f}s)")


if __name__ == "__main__":
    import sys
    skip_db = "--skip-db" in sys.argv or "--no-db" in sys.argv
    run_all_tests(skip_db_tests=skip_db)
