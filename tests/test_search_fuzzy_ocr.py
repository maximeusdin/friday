#!/usr/bin/env python3
"""
Offline tests for the Search-tab OCR-variant channel in
retrieval/search_executor.run_search_expand_fuzzy.

The fuzzy expand phase now asks retrieval.ocr_variants for corpus lexemes that
are plausible OCR corruptions of the query tokens (skeleton lookup; e.g. typed
FUHR OCR-read as FUER) and runs them through the exact page-hits machinery,
inserting before the word_similarity hits so ON CONFLICT dedupes in favor of
the variant-exact hits.

No database required: a scripted fake connection dispatches canned rows per SQL
marker and captures every statement. retrieval.ocr_variants is replaced by a
fake module in sys.modules so the module contract (not its implementation) is
exercised.
"""

import sys
import types
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import retrieval.search_executor as se


# ============================================================================
# Fakes (no DB)
# ============================================================================

class ScriptedCursor:
    def __init__(self, conn):
        self._conn = conn
        self._last = []

    def execute(self, sql, params=None):
        sql_text = sql.decode() if isinstance(sql, (bytes, bytearray)) else sql
        self._conn.executed.append((sql_text, params))
        self._last = self._conn.route(sql_text, params)

    def fetchone(self):
        return self._last[0] if self._last else None

    def fetchall(self):
        return list(self._last)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ScriptedConn:
    """Routes each executed statement to canned rows by first matching SQL marker.
    A route value may be a list of rows or a callable (sql, params) -> rows."""

    def __init__(self, routes):
        self.executed = []
        self.routes = routes
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return ScriptedCursor(self)

    def route(self, sql, params):
        for marker, result in self.routes:
            if marker in sql:
                return result(sql, params) if callable(result) else result
        return []

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


# ============================================================================
# Fixtures
# ============================================================================

SCOPE = {"mode": "custom", "included_collection_ids": [7, 8]}
COLLECTIONS = [(7, "silvermaster", "Silvermaster File"), (8, "venona", "Venona")]
# (collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text)
VARIANT_ROW = (7, 100, 1000, 1, 1, 555, "report on FUER activities")
FUZZY_ROW = (7, 100, 1001, 2, 2, 556, "trigram match text")


def make_conn(build_ids):
    """build_ids: collection_slug -> dictionary build id (or absent for none)."""

    def dict_build(sql, params):
        chunk_pv, slug, norm_version = params
        assert norm_version == "norm_v1"
        b = build_ids.get(slug)
        return [(b,)] if b else []

    return ScriptedConn([
        ("SELECT query_raw", [("fuhr", SCOPE, True)]),
        ("FROM collections WHERE id = ANY", [("silvermaster",), ("venona",)]),
        ("FROM collections WHERE slug = ANY", COLLECTIONS),
        ("GROUP BY cm.collection_slug, c.pipeline_version",
         [("silvermaster", "structured_4k"), ("venona", "venona_pages_v1")]),
        ("FROM corpus_dictionary_builds", dict_build),
        ("information_schema.columns", [(1,)]),
        ("MAX(hit_rank)", [(17,)]),
        ("ts_rank_cd", [VARIANT_ROW]),
        ("word_similarity", [FUZZY_ROW]),
        ("INSERT INTO search_result_page_hits", []),
        ("SELECT collection_id, COUNT(*)", [(7, 3), (8, 1)]),
        ("SELECT COUNT(*) FROM search_result_page_hits", [(4,)]),
        ("SELECT id, slug, title FROM collections", COLLECTIONS),
        ("UPDATE search_result_sets", []),
    ])


class FakeOcrModule:
    """Stand-in for retrieval.ocr_variants honoring its call contract."""

    def __init__(self, expansions=None, raise_on_fetch=False):
        self.expansions = expansions if expansions is not None else {
            "fuhr": [{"lexeme": "fuer", "cost": 0.25, "chunk_freq": 3, "source": "ocr"}],
        }
        self.raise_on_fetch = raise_on_fetch
        self.load_calls = 0
        self.fetch_calls = []

    def install(self):
        mod = types.ModuleType("retrieval.ocr_variants")
        mod.load_confusion = self.load_confusion
        mod.fetch_ocr_variants = self.fetch_ocr_variants
        self._prev = sys.modules.get("retrieval.ocr_variants")
        sys.modules["retrieval.ocr_variants"] = mod

    def uninstall(self):
        if self._prev is not None:
            sys.modules["retrieval.ocr_variants"] = self._prev
        else:
            sys.modules.pop("retrieval.ocr_variants", None)

    def load_confusion(self, path=None):
        self.load_calls += 1
        return {"version": "ocr_confusion_v1"}

    def fetch_ocr_variants(self, conn, tokens, build_id, conf=None, **kwargs):
        self.fetch_calls.append((list(tokens), build_id, conf, kwargs))
        if self.raise_on_fetch:
            raise RuntimeError("boom")
        return self.expansions


def _recorder_execute_values(cur, sql, argslist, template=None, page_size=100):
    """Replace psycopg2 execute_values: capture the INSERT and its rows."""
    cur.execute(sql, list(argslist))


def _run(build_ids, fake_ocr=None, stub_helper=False):
    """Run run_search_expand_fuzzy under fakes; returns (conn, result)."""
    conn = make_conn(build_ids)
    orig_ev = se.execute_values
    orig_helper = se._ocr_variant_terms_for_collections
    se.execute_values = _recorder_execute_values
    if stub_helper:
        # Byte-identical stand-in for the pre-edit code: every statement the
        # edit added is gated on this helper's result, so {} reproduces the
        # exact pre-edit SQL stream.
        se._ocr_variant_terms_for_collections = lambda conn, q, slugs: {}
    if fake_ocr:
        fake_ocr.install()
    try:
        out = se.run_search_expand_fuzzy(conn, "rs-1")
    finally:
        se.execute_values = orig_ev
        se._ocr_variant_terms_for_collections = orig_helper
        if fake_ocr:
            fake_ocr.uninstall()
    return conn, out


DICT_PROBE_MARKERS = (
    "GROUP BY cm.collection_slug, c.pipeline_version",  # pipeline-version resolve
    "FROM corpus_dictionary_builds",                     # latest-build lookup
)


def _without_dict_probes(executed):
    return [(s, p) for s, p in executed if not any(m in s for m in DICT_PROBE_MARKERS)]


# ============================================================================
# No dictionary builds (prod today): byte-identical behavior
# ============================================================================

def test_no_build_sql_identical_to_pre_edit():
    fake = FakeOcrModule()
    conn, out = _run({}, fake_ocr=fake)
    baseline_conn, baseline_out = _run({}, stub_helper=True)

    # Aside from the two read-only dictionary probes, the statement stream is
    # exactly the pre-edit one (same SQL, same params, same order).
    assert _without_dict_probes(conn.executed) == baseline_conn.executed
    # And nothing OCR-shaped ran: no variant page-hits query, no skeleton SQL,
    # no tsv_simple detection (gated on variants existing).
    for sql, _ in conn.executed:
        assert "ts_rank_cd" not in sql
        assert "corpus_dictionary_lexemes" not in sql
        assert "skeleton" not in sql
        assert "information_schema.columns" not in sql
    assert fake.fetch_calls == [] and fake.load_calls == 0
    assert out == baseline_out and out["status"] == "complete"
    print("[OK] no-build path emits identical SQL to pre-edit")


# ============================================================================
# With a dictionary build: variant-exact query added, per-collection
# ============================================================================

def test_with_build_adds_variant_query():
    fake = FakeOcrModule()
    conn, out = _run({"silvermaster": 42}, fake_ocr=fake)

    # The module was consulted once, for the collection that has a build,
    # with the tokenized query and the loaded confusion table.
    assert fake.load_calls == 1
    assert fake.fetch_calls == [(["fuhr"], 42, {"version": "ocr_confusion_v1"}, {})]

    # Exactly one variant-exact page-hits query (venona has no build: skipped),
    # scoped to silvermaster and searching for the variant lexeme.
    variant_idxs = [i for i, (s, _) in enumerate(conn.executed) if "ts_rank_cd" in s]
    assert len(variant_idxs) == 1
    v_sql, v_params = conn.executed[variant_idxs[0]]
    assert v_params["collection_slugs"] == ["silvermaster"]
    assert "fuer" in v_params.values()
    assert "to_tsquery('simple'" in v_sql

    # Both collections still ran the word_similarity scan, and the variant
    # query came before the first one (variant-exact hits first).
    fuzzy_idxs = [i for i, (s, _) in enumerate(conn.executed) if "word_similarity" in s]
    assert len(fuzzy_idxs) == 2
    assert variant_idxs[0] < fuzzy_idxs[0]

    # Variant hits are inserted exactly like fuzzy hits: same INSERT with the
    # PK ON CONFLICT DO NOTHING (dedupe against exact and trigram hits), and
    # hit_rank continues from the exact phase (MAX was 17).
    inserts = [(i, s, p) for i, (s, p) in enumerate(conn.executed)
               if "INSERT INTO search_result_page_hits" in s]
    assert len(inserts) == 3  # variant + 2 fuzzy collections
    variant_insert = inserts[0]
    assert variant_insert[0] > variant_idxs[0] and variant_insert[0] < fuzzy_idxs[0]
    for _, sql, _p in inserts:
        assert "ON CONFLICT (result_set_id, collection_id, document_id, page_id) DO NOTHING" in sql
    assert inserts[0][1] == inserts[1][1]  # byte-identical INSERT: same UI labeling applies
    variant_hit = variant_insert[2][0]
    assert variant_hit[6] == 555  # chunk_id from the variant page-hits row
    assert variant_hit[8] == 18   # hit_rank = 17 (exact phase max) + 1
    fuzzy_hit = inserts[1][2][0]
    assert fuzzy_hit[8] == 19     # trigram hits ranked after variant hits

    # Variant terms flow into coverage phrases so fetch_more_snippets can
    # center late snippets on the garbled form.
    assert out["status"] == "complete"
    assert out["coverage_json"]["phrases"] == ["fuhr", "fuer"]
    print("[OK] with-build path adds the variant-exact query, variant hits first")


def test_with_build_tsv_column_detected_once():
    fake = FakeOcrModule()
    conn, _ = _run({"silvermaster": 42}, fake_ocr=fake)
    tsv_checks = [s for s, _ in conn.executed if "information_schema.columns" in s]
    assert len(tsv_checks) == 1
    v_sql = next(s for s, _ in conn.executed if "ts_rank_cd" in s)
    assert "tsv_simple" in v_sql
    print("[OK] tsv column detection gated on variants and done once")


# ============================================================================
# Failure isolation: the channel must never break fuzzy expansion
# ============================================================================

def test_ocr_module_failure_rolls_back_and_falls_through():
    fake = FakeOcrModule(raise_on_fetch=True)
    conn, out = _run({"silvermaster": 42}, fake_ocr=fake)
    assert conn.rollbacks >= 1  # aborted-transaction state cleared before any write
    assert not any("ts_rank_cd" in s for s, _ in conn.executed)
    assert out["status"] == "complete"  # word_similarity phase still completed
    assert len([s for s, _ in conn.executed if "word_similarity" in s]) == 2
    print("[OK] OCR channel failure rolls back and leaves fuzzy expansion intact")


def test_ocr_module_absent_is_noop():
    # No fake installed and no real module importable => helper returns {}.
    prev = sys.modules.pop("retrieval.ocr_variants", None)
    real = REPO_ROOT / "retrieval" / "ocr_variants.py"
    try:
        if real.exists():
            # Real module present: absence can't be simulated; helper contract
            # is covered by the no-build test instead.
            print("[SKIP] retrieval.ocr_variants exists; absence path not simulated")
            return
        conn = make_conn({"silvermaster": 42})
        terms = se._ocr_variant_terms_for_collections(conn, "fuhr", ["silvermaster"])
        assert terms == {}
        assert conn.executed == []  # bails before any SQL
    finally:
        if prev is not None:
            sys.modules["retrieval.ocr_variants"] = prev
    print("[OK] missing retrieval.ocr_variants module is a silent no-op")


if __name__ == "__main__":
    test_no_build_sql_identical_to_pre_edit()
    test_with_build_adds_variant_query()
    test_with_build_tsv_column_detected_once()
    test_ocr_module_failure_rolls_back_and_falls_through()
    test_ocr_module_absent_is_noop()
    print("All search fuzzy OCR wiring tests passed.")
