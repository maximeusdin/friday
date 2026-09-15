#!/usr/bin/env python3
"""
Offline tests for retrieval/ocr_variants.py (OCR-variant channel, Phase 2).

Covers: builtin-prior class derivation, skeleton equivalences (the motivating
FUHR/FUER case), confusion-weighted edit distance (sub costs, rn<->m pair
transitions, cap early-exit), artifact loading/fallback/caching, and
fetch_ocr_variants against a fake DB connection (SQL shape incl. the
frequency ceiling, ranking, top_k, missing-column and build_id-None paths,
short-token skip).

No database required.
"""

import inspect
import json
import math
import sys
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import retrieval.ocr_variants as ov
from retrieval.ocr_variants import (
    BUILTIN_PRIORS,
    Confusion,
    derive_confusion_classes,
    fetch_ocr_variants,
    load_confusion,
    skeleton,
    weighted_distance,
)


@pytest.fixture()
def conf():
    return Confusion(BUILTIN_PRIORS)


# ============================================================================
# Fakes (no DB)
# ============================================================================

class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        self._rows = self.conn.answer(sql, params)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    """
    Answers the information_schema probe and the skeleton-keyed variant query.
    rows_by_skeleton maps skeleton -> [(lexeme, chunk_freq), ...]; the fake
    applies the chunk_freq ceiling like the real SQL would.
    """

    def __init__(self, *, column_exists=True, rows_by_skeleton=None, probe_raises=False):
        self.executed = []
        self.column_exists = column_exists
        self.rows_by_skeleton = rows_by_skeleton or {}
        self.probe_raises = probe_raises

    def cursor(self):
        return FakeCursor(self)

    def answer(self, sql, params):
        if "information_schema" in sql:
            if self.probe_raises:
                raise RuntimeError("probe failed")
            return [(1,)] if self.column_exists else []
        # variant query: params = (build_id, skeleton, token, freq_ceiling)
        _build_id, sk, _tok, ceiling = params
        rows = self.rows_by_skeleton.get(sk, [])
        return [(lx, freq) for lx, freq in rows if freq <= ceiling]


def variant_executes(conn):
    return [(sql, params) for sql, params in conn.executed if "information_schema" not in sql]


# ============================================================================
# Builtin priors / class derivation
# ============================================================================

def test_builtin_classes_derivation(conf):
    # Union-find over sub pairs with cost <= 0.35, cheapest merges first:
    # 1/i/l (0.3s), e/o (0.3) + h (via e|h 0.35), n/u (0.35).
    assert conf.classes == [["1", "i", "l"], ["e", "h", "o"], ["n", "u"]]
    assert BUILTIN_PRIORS["classes"] == conf.classes


def test_derive_classes_respects_member_cap():
    sub_costs = {"a|b": 0.1, "b|c": 0.2, "c|d": 0.25, "d|e": 0.3, "e|f": 0.35}
    classes = derive_confusion_classes(sub_costs, max_members=4)
    # a-b-c-d merge (4 members); d|e would make 5 so it's skipped; e|f still merges.
    assert classes == [["a", "b", "c", "d"], ["e", "f"]]


# ============================================================================
# skeleton()
# ============================================================================

def test_skeleton_fuhr_equals_fuer_under_builtin_priors(conf):
    # The motivating case: typed FUHR OCR-read as FUER. e|h at 0.35 is
    # deliberately within the class threshold so this holds on builtin priors.
    assert skeleton("fuhr", conf) == skeleton("fuer", conf)
    assert skeleton("fuhr", conf) == "fner"


def test_skeleton_class_collapses(conf):
    assert skeleton("hello", conf) == skeleton("hollo", conf)  # e~o
    assert skeleton("line", conf) == skeleton("lino", conf)  # i~l~1, e~o
    assert skeleton("unit", conf) == skeleton("nnit", conf)  # u~n


def test_skeleton_normalization(conf):
    # lowercase + strip non-alnum
    assert skeleton("FUHR.", conf) == skeleton("fuhr", conf)
    assert skeleton("o'brien", conf) == skeleton("OBRIEN", conf)
    assert skeleton("", conf) == ""
    # unmapped chars pass through
    assert skeleton("xyz", conf) == "xyz"


# ============================================================================
# weighted_distance()
# ============================================================================

def test_weighted_distance_identity_and_sub_cost(conf):
    assert weighted_distance("fuhr", "fuhr", conf) == 0.0
    # single h->e substitution priced from sub_costs (e|h = 0.35)
    assert weighted_distance("fuhr", "fuer", conf) == pytest.approx(0.35)
    # symmetric
    assert weighted_distance("fuer", "fuhr", conf) == pytest.approx(0.35)


def test_weighted_distance_ordering_fuer_nearer_than_ford(conf):
    d_fuer = weighted_distance("fuhr", "fuer", conf)
    d_ford = weighted_distance("fuhr", "ford", conf)
    assert d_fuer < d_ford


def test_weighted_distance_pair_transition_rn_m(conf):
    # rn|m prices "modern" vs "rnodern" at the pair cost, far below the
    # sub+indel route (1.0 + 0.8).
    assert weighted_distance("modern", "rnodern", conf) == pytest.approx(0.35)
    # and in the other direction (2 chars of `a` read as 1 char of `b`)
    assert weighted_distance("rnodern", "modern", conf) == pytest.approx(0.35)


def test_weighted_distance_cap_early_exit(conf):
    d = weighted_distance("modern", "zzzzzz", conf, cap=1.0)
    assert math.isinf(d)
    # final clamp: a true distance above cap comes back INF too
    assert math.isinf(weighted_distance("fuhr", "fuer", conf, cap=0.3))
    # and the same pair is finite under a roomier cap
    assert weighted_distance("fuhr", "fuer", conf, cap=2.0) == pytest.approx(0.35)


def test_weighted_distance_indel_cost(conf):
    # pure insertion priced at indel_cost
    assert weighted_distance("fuhr", "fuhre", conf) == pytest.approx(0.8)
    assert weighted_distance("fuhr", "fuher", conf) == pytest.approx(0.8)


# ============================================================================
# load_confusion()
# ============================================================================

def test_load_confusion_missing_file_falls_back_to_builtin(tmp_path):
    conf = load_confusion(path=str(tmp_path / "missing.json"))
    assert conf.indel_cost == pytest.approx(0.8)
    assert conf.default_sub_cost == pytest.approx(1.0)
    # fallback still carries the FUHR/FUER equivalence
    assert skeleton("fuhr", conf) == skeleton("fuer", conf)


def test_load_confusion_reads_artifact_and_caches(tmp_path):
    artifact = {
        "version": "ocr_confusion_v1",
        "generated": "2026-08-04T00:00:00Z",
        "sources": ["test"],
        "classes": [["a", "b"]],
        "sub_costs": {"a|b": 0.2},
        "pair_costs": {"xy|z": 0.3},
        "default_sub_cost": 0.9,
        "indel_cost": 0.7,
    }
    p = tmp_path / "ocr_confusion.json"
    p.write_text(json.dumps(artifact))

    conf = load_confusion(path=str(p))
    assert conf.version == "ocr_confusion_v1"
    assert conf.indel_cost == pytest.approx(0.7)
    assert conf.classes == [["a", "b"]]
    assert skeleton("bat", conf) == "aat"  # b collapses to class rep "a"
    assert weighted_distance("az", "bz", conf) == pytest.approx(0.2)
    assert weighted_distance("xyq", "zq", conf) == pytest.approx(0.3)

    # module-level cache: same object on repeat load
    assert load_confusion(path=str(p)) is conf


def test_load_confusion_malformed_file_falls_back(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    conf = load_confusion(path=str(p))
    assert skeleton("fuhr", conf) == skeleton("fuer", conf)


# ============================================================================
# fetch_ocr_variants()
# ============================================================================

def test_fetch_ocr_variants_sql_shape_ranking_and_top_k(conf):
    # Rows keyed by skeleton("fuhr") == "fner". Costs: fuer 0.35 (h->e),
    # fuhre/fuher 0.8 each (insert e) -> the chunk_freq ASC tiebreak decides;
    # frodo is beyond max_cost and drops out.
    fner_rows = [("fuer", 5), ("fuher", 50), ("fuhre", 2), ("frodo", 1)]
    conn = FakeConn(rows_by_skeleton={"fner": fner_rows})

    # exact call shape used by retrieval/fuzzy_lex.py
    out = fetch_ocr_variants(
        conn,
        ["fuhr"],
        7,
        conf,
        max_cost=1.5,
        top_k=2,
        freq_ceiling=500,
    )

    assert list(out.keys()) == ["fuhr"]
    assert [e["lexeme"] for e in out["fuhr"]] == ["fuer", "fuhre"]  # cost asc, then freq asc; top_k=2
    entry = out["fuhr"][0]
    assert entry == {
        "lexeme": "fuer",
        "cost": pytest.approx(0.35),
        "chunk_freq": 5,
        "source": "ocr",
    }

    executes = variant_executes(conn)
    assert len(executes) == 1
    sql, params = executes[0]
    assert "FROM corpus_dictionary_lexemes" in sql
    assert "build_id = %s" in sql
    assert "skeleton = %s" in sql
    assert "lexeme <> %s" in sql
    assert "chunk_freq <= %s" in sql
    assert params == (7, "fner", "fuhr", 500)


def test_fetch_ocr_variants_freq_ceiling(conf):
    conn = FakeConn(rows_by_skeleton={"fner": [("fuer", 5), ("fuhre", 400)]})
    out = fetch_ocr_variants(conn, ["fuhr"], 7, conf, max_cost=1.5, top_k=4, freq_ceiling=100)
    # the fake applies the ceiling like the SQL predicate would
    assert [e["lexeme"] for e in out["fuhr"]] == ["fuer"]
    _sql, params = variant_executes(conn)[0]
    assert params[3] == 100


def test_fetch_ocr_variants_ranks_rarity_on_cost_ties(conf):
    # equal cost (0.8 each): rarer lexeme first (chunk_freq ASC)
    conn = FakeConn(rows_by_skeleton={"fner": [("fuher", 50), ("fuhre", 2)]})
    out = fetch_ocr_variants(conn, ["fuhr"], 7, conf, max_cost=1.5, top_k=4, freq_ceiling=500)
    assert [e["lexeme"] for e in out["fuhr"]] == ["fuhre", "fuher"]


def test_fetch_ocr_variants_missing_column_returns_empty(conf):
    conn = FakeConn(column_exists=False, rows_by_skeleton={"fner": [("fuer", 5)]})
    assert fetch_ocr_variants(conn, ["fuhr"], 7, conf) == {}
    assert variant_executes(conn) == []  # only the probe ran


def test_fetch_ocr_variants_probe_error_returns_empty(conf):
    conn = FakeConn(probe_raises=True)
    assert fetch_ocr_variants(conn, ["fuhr"], 7, conf) == {}


def test_fetch_ocr_variants_build_id_none_returns_empty(conf):
    conn = FakeConn(rows_by_skeleton={"fner": [("fuer", 5)]})
    assert fetch_ocr_variants(conn, ["fuhr"], None, conf) == {}
    assert conn.executed == []  # not even the probe


def test_fetch_ocr_variants_skips_short_tokens(conf):
    conn = FakeConn(rows_by_skeleton={"fner": [("fuer", 5)]})
    out = fetch_ocr_variants(conn, ["abc", "of", "fuhr"], 7, conf)
    assert list(out.keys()) == ["fuhr"]
    # one variant query only (short tokens never hit the DB)
    assert len(variant_executes(conn)) == 1


def test_fetch_ocr_variants_no_tokens(conf):
    conn = FakeConn()
    assert fetch_ocr_variants(conn, [], 7, conf) == {}
    assert conn.executed == []


def test_fetch_ocr_variants_omits_empty_and_identical(conf):
    # a row echoing the token itself is dropped (SQL excludes it; guarded here too)
    conn = FakeConn(rows_by_skeleton={"fner": [("fuhr", 9)]})
    assert fetch_ocr_variants(conn, ["fuhr"], 7, conf) == {}


# ============================================================================
# API contract with the inherited call sites
# ============================================================================

def test_call_shapes_match_inherited_call_sites(conf):
    # scripts/build_corpus_dictionary.py: load_confusion(); skeleton(lx, conf)
    assert callable(load_confusion)
    assert skeleton("fuhr", conf) == "fner"
    # retrieval/fuzzy_lex.py: fetch_ocr_variants(conn, tokens, build_id, conf,
    #   max_cost=..., top_k=..., freq_ceiling=...)
    sig = inspect.signature(fetch_ocr_variants)
    sig.bind(FakeConn(), ["fuhr"], 7, conf, max_cost=1.5, top_k=4, freq_ceiling=500)
    # module is importable the way fuzzy_lex lazily imports it
    import importlib

    mod = importlib.import_module("retrieval.ocr_variants")
    assert mod.load_confusion is ov.load_confusion
    assert mod.fetch_ocr_variants is ov.fetch_ocr_variants
