#!/usr/bin/env python3
"""
Offline tests for raw-OCR union matching in retrieval/ops.py.

Transcript-bearing documents (documents.metadata ? 'transcript') keep raw OCR in
chunks.text while clean_text holds the corrected transcript. Lexical MATCHING must
search the union (clean OR raw-for-transcript-docs); DISPLAY (ts_headline, previews)
must stay on COALESCE(clean_text, text) only.

No database required: a fake connection captures the generated SQL.
"""

import re
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import retrieval.ops as ops
from retrieval.ops import (
    SearchFilters,
    _TRANSCRIPT_DOC_EXISTS,
    _fetch_highlights,
    hybrid_rrf,
    hybrid_rrf_sql,
    lex_and,
    lex_exact,
    lex_near,
)


# ============================================================================
# Fakes (no DB)
# ============================================================================

class FakeCursor:
    def __init__(self, executed, rows=None):
        self._executed = executed
        self._rows = rows or []

    def execute(self, sql, params=None):
        self._executed.append((sql, params))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeConn:
    def __init__(self, rows=None):
        self.executed = []
        self._rows = rows

    def cursor(self):
        return FakeCursor(self.executed, self._rows)

    def commit(self):
        pass

    def rollback(self):
        pass


def _named_placeholders(sql):
    return set(re.findall(r"%\((\w+)\)s", sql))


def _assert_params_cover_sql(sql, params):
    missing = _named_placeholders(sql) - set(params.keys() if isinstance(params, dict) else [])
    assert not missing, f"SQL references params not bound: {missing}"


TRANSCRIPT_MARKER = "d_tr.metadata ? 'transcript'"


def _assert_ordered_limit(sql, limit_placeholder, order_marker="ts_rank_cd"):
    """The lex CTE's LIMIT must be preceded by a deterministic CTE-level
    ORDER BY <rank expr> DESC, c.id ASC (otherwise truncation is plan-dependent
    and raw-only rows can displace high-ranked clean hits arbitrarily)."""
    idx_limit = sql.index(limit_placeholder)
    pre = sql[:idx_limit]
    idx_order = pre.rfind("ORDER BY")
    assert idx_order != -1, f"no ORDER BY before {limit_placeholder}"
    tail = pre[idx_order:]
    assert order_marker in tail, f"ORDER BY before {limit_placeholder} lacks {order_marker}"
    assert "DESC" in tail and "c.id ASC" in tail, (
        f"ORDER BY before {limit_placeholder} lacks 'DESC, c.id ASC' tie-break: {tail!r}"
    )
    # Must be a CTE-level ORDER BY immediately before LIMIT, not a window
    # ORDER BY buried inside ROW_NUMBER() OVER (...) earlier in the SELECT.
    assert "FROM" not in tail, (
        f"ORDER BY before {limit_placeholder} is not clause-level (window only?): {tail!r}"
    )


# ============================================================================
# lex_exact
# ============================================================================

def test_lex_exact_ilike_union():
    conn = FakeConn()
    lex_exact(conn, "silvermaster", filters=SearchFilters(), k=5, log_run=False)
    sql, params = conn.executed[0]
    assert "COALESCE(c.clean_text, c.text) ILIKE %(pat)s" in sql
    assert TRANSCRIPT_MARKER in sql
    assert "c.text ILIKE %(pat)s" in sql
    # Display stays clean
    assert "LEFT(COALESCE(c.clean_text, c.text), %(preview_chars)s)" in sql
    _assert_params_cover_sql(sql, params)
    print("[OK] lex_exact ILIKE union")


def test_lex_exact_position_union():
    conn = FakeConn()
    lex_exact(conn, "Silvermaster", filters=SearchFilters(), k=5,
              case_sensitive=True, log_run=False)
    sql, params = conn.executed[0]
    assert "POSITION(%(term)s IN COALESCE(c.clean_text, c.text)) > 0" in sql
    assert TRANSCRIPT_MARKER in sql
    assert "POSITION(%(term)s IN c.text) > 0" in sql
    _assert_params_cover_sql(sql, params)
    print("[OK] lex_exact POSITION union")


def test_lex_exact_short_word_regex_union():
    conn = FakeConn()
    lex_exact(conn, "OSS", filters=SearchFilters(), k=5, log_run=False)
    sql, params = conn.executed[0]
    assert "COALESCE(c.clean_text, c.text) ~* %(pat)s" in sql
    assert TRANSCRIPT_MARKER in sql
    assert "c.text ~* %(pat)s" in sql
    _assert_params_cover_sql(sql, params)
    print("[OK] lex_exact short-word regex union")


# ============================================================================
# lex_and
# ============================================================================

def test_lex_and_union_per_term():
    conn = FakeConn()
    lex_and(conn, ["bentley", "deposition"], filters=SearchFilters(), k=5, log_run=False)
    sql, params = conn.executed[0]
    for i in range(2):
        assert f"COALESCE(c.clean_text, c.text) ILIKE %(p{i})s" in sql
        assert f"c.text ILIKE %(p{i})s" in sql
    assert sql.count(TRANSCRIPT_MARKER) == 2  # one OR branch per term
    _assert_params_cover_sql(sql, params)
    print("[OK] lex_and per-term union")


# ============================================================================
# lex_near
# ============================================================================

def test_lex_near_sql_union_and_clean_display():
    conn = FakeConn(rows=[])
    lex_near(conn, "bentley", "courier", window_words=10,
             filters=SearchFilters(), log_run=False)
    sql, params = conn.executed[0]
    for p in ("pa", "pb"):
        assert f"COALESCE(c.clean_text, c.text) ILIKE %({p})s" in sql
        assert f"c.text ILIKE %({p})s" in sql
    # raw_text is fetched only for transcript docs, for Python-side verification
    assert "AS raw_text" in sql
    # Display stays clean
    assert "LEFT(COALESCE(c.clean_text, c.text), %(preview_chars)s) AS preview" in sql
    _assert_params_cover_sql(sql, params)
    print("[OK] lex_near SQL union + clean display")


def test_lex_near_verifies_proximity_in_raw_text():
    # clean text lacks "courier" entirely (transcription changed it);
    # raw OCR still has both terms within the window -> hit must survive.
    clean = "bentley delivered the papers to the contact in washington"
    raw = "bentley the courier delivered the papers"
    row = (1, "silvermaster", 522, 10, 10, None, None, clean, clean[:100], raw)
    conn = FakeConn(rows=[row])
    hits = lex_near(conn, "bentley", "courier", window_words=5,
                    filters=SearchFilters(), log_run=False)
    assert len(hits) == 1 and hits[0].chunk_id == 1
    # preview (display) still comes from the clean text
    assert "courier" not in hits[0].preview
    print("[OK] lex_near raw-OCR proximity verification")


def test_lex_near_no_raw_text_unchanged():
    clean = "bentley the courier delivered the papers"
    row = (2, "silvermaster", 522, 11, 11, None, None, clean, clean[:100], None)
    conn = FakeConn(rows=[row])
    hits = lex_near(conn, "bentley", "courier", window_words=5,
                    filters=SearchFilters(), log_run=False)
    assert len(hits) == 1 and hits[0].chunk_id == 2
    print("[OK] lex_near clean-text path unchanged")


# ============================================================================
# _fetch_highlights
# ============================================================================

def test_fetch_highlights_union_match_clean_display():
    conn = FakeConn(rows=[])
    _fetch_highlights(conn, [1, 2], "bentley", "silvermaster_pages_v1")
    sql, params = conn.executed[0]
    # Match filter is a union
    assert TRANSCRIPT_MARKER in sql
    assert "to_tsvector('simple', c.text) @@ websearch_to_tsquery('simple', %s)" in sql
    assert ("to_tsvector('simple', COALESCE(c.clean_text, c.text)) "
            "@@ websearch_to_tsquery('simple', %s)") in sql
    # ts_headline (display) stays on COALESCE only
    headline = sql.split("ts_headline(")[1].split("websearch_to_tsquery")[0]
    assert "COALESCE(c.clean_text, c.text)" in headline
    # Positional params: placeholder count must equal bound params
    assert sql.count("%s") == len(params) == 5
    print("[OK] _fetch_highlights union match, clean headline, 5 positional params")


# ============================================================================
# hybrid_rrf / hybrid_rrf_sql: deterministic lex-CTE truncation
# ============================================================================

def test_hybrid_rrf_lex_cte_order_by_before_limit():
    conn = FakeConn(rows=[])
    orig_embed = ops.embed_query
    ops.embed_query = lambda text: [0.0, 0.0, 0.0]
    try:
        hybrid_rrf(conn, "bentley courier", filters=SearchFilters(),
                   expand_concordance=False, fuzzy_lex_enabled=False,
                   use_soft_lex=True, log_run=False)
    finally:
        ops.embed_query = orig_embed
    hybrid_sqls = [s for s, _ in conn.executed if "LIMIT %(top_n_lex)s" in s]
    assert hybrid_sqls, "hybrid_rrf SQL not captured"
    sql = hybrid_sqls[0]
    _assert_ordered_limit(sql, "LIMIT %(top_n_lex)s")
    # soft_lex CTE truncation is deterministic too (similarity-ranked)
    _assert_ordered_limit(sql, "LIMIT %(soft_lex_max_results)s",
                          order_marker="word_similarity")
    print("[OK] hybrid_rrf lex + soft_lex CTE ORDER BY precedes LIMIT")


def test_hybrid_rrf_sql_lex_ranked_order_by_before_limit():
    class CountCursor(FakeCursor):
        def fetchone(self):
            return (0,)

    class CountConn(FakeConn):
        def cursor(self):
            return CountCursor(self.executed, self._rows)

    # max_hits=0 forces the cap path so count_sql is generated as well
    conn = CountConn(rows=[])
    hybrid_rrf_sql(conn, "bentley courier", [0.0, 0.0, 0.0], max_hits=0)
    sqls = [s for s, _ in conn.executed if "LIMIT %(lex_limit)s" in s]
    assert len(sqls) == 2, f"expected main + count SQL, got {len(sqls)}"
    for sql in sqls:
        _assert_ordered_limit(sql, "LIMIT %(lex_limit)s")
    print("[OK] hybrid_rrf_sql lex_ranked ORDER BY precedes LIMIT (main + count)")


# ============================================================================
# Fragment sanity
# ============================================================================

def test_transcript_exists_fragment_shape():
    assert "chunk_pages cp_tr" in _TRANSCRIPT_DOC_EXISTS
    assert "pages p_tr" in _TRANSCRIPT_DOC_EXISTS
    assert "documents d_tr" in _TRANSCRIPT_DOC_EXISTS
    assert "cp_tr.chunk_id = c.id" in _TRANSCRIPT_DOC_EXISTS
    assert TRANSCRIPT_MARKER in _TRANSCRIPT_DOC_EXISTS
    # no stray psycopg2 placeholders inside the fragment
    assert "%s" not in _TRANSCRIPT_DOC_EXISTS and "%(" not in _TRANSCRIPT_DOC_EXISTS
    print("[OK] transcript EXISTS fragment shape")


if __name__ == "__main__":
    test_lex_exact_ilike_union()
    test_lex_exact_position_union()
    test_lex_exact_short_word_regex_union()
    test_lex_and_union_per_term()
    test_lex_near_sql_union_and_clean_display()
    test_lex_near_verifies_proximity_in_raw_text()
    test_lex_near_no_raw_text_unchanged()
    test_fetch_highlights_union_match_clean_display()
    test_hybrid_rrf_lex_cte_order_by_before_limit()
    test_hybrid_rrf_sql_lex_ranked_order_by_before_limit()
    test_transcript_exists_fragment_shape()
    print("All raw-OCR union tests passed.")
