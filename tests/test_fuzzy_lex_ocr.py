"""Offline tests for the OCR-variant channel merge in retrieval/fuzzy_lex.py."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.fuzzy_lex import (  # noqa: E402
    FuzzyLexConfig,
    fetch_fuzzy_variants_for_tokens,
    _merge_ocr_expansions,
)


class FakeCursor:
    """Returns no trigram rows; captures executed SQL."""

    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class FakeConn:
    def __init__(self):
        self.cursors = []

    def cursor(self):
        c = FakeCursor()
        self.cursors.append(c)
        return c


def _patch_ocr(monkeypatch, expansions):
    mod = types.ModuleType("retrieval.ocr_variants")
    mod.load_confusion = lambda path=None: object()
    calls = {}

    def fetch_ocr_variants(conn, tokens, build_id, conf, *, max_cost, top_k, freq_ceiling):
        calls["args"] = dict(tokens=list(tokens), build_id=build_id,
                             max_cost=max_cost, top_k=top_k, freq_ceiling=freq_ceiling)
        return expansions

    mod.fetch_ocr_variants = fetch_ocr_variants
    monkeypatch.setitem(sys.modules, "retrieval.ocr_variants", mod)
    return calls


def test_ocr_channel_fills_when_trigram_empty(monkeypatch):
    calls = _patch_ocr(monkeypatch, {
        "fuhr": [{"lexeme": "fuer", "cost": 0.35, "chunk_freq": 2, "source": "ocr"}],
    })
    build_id, expansions = fetch_fuzzy_variants_for_tokens(
        FakeConn(),
        tokens=["fuhr"],
        chunk_pv="pv",
        collection_slug="silvermaster",
        norm_version="norm_v1",
        config=FuzzyLexConfig(dictionary_build_id=7),
    )
    assert build_id == 7
    assert [e["lexeme"] for e in expansions["fuhr"]] == ["fuer"]
    assert expansions["fuhr"][0]["source"] == "ocr"
    assert calls["args"]["build_id"] == 7
    assert calls["args"]["max_cost"] == 1.5 and calls["args"]["top_k"] == 4


def test_ocr_disabled_and_no_build_paths(monkeypatch):
    calls = _patch_ocr(monkeypatch, {"fuhr": [{"lexeme": "fuer", "cost": 0.3, "chunk_freq": 1}]})
    _, expansions = fetch_fuzzy_variants_for_tokens(
        FakeConn(), tokens=["fuhr"], chunk_pv="pv", collection_slug=None,
        norm_version="norm_v1",
        config=FuzzyLexConfig(dictionary_build_id=7, ocr_enabled=False),
    )
    assert expansions == {} and "args" not in calls

    # build_id None (no dictionary anywhere): channel must not fire.
    _, expansions = fetch_fuzzy_variants_for_tokens(
        FakeConn(), tokens=["fuhr"], chunk_pv="pv", collection_slug=None,
        norm_version="norm_v1", config=FuzzyLexConfig(dictionary_build_id=None),
    )
    assert expansions == {} and "args" not in calls


def test_merge_dedupes_and_relabels_both():
    expansions = {"fuhr": [{"lexeme": "fuer", "similarity": 0.5, "chunk_freq": 2}]}
    _merge_ocr_expansions(
        tokens=["fuhr"],
        expansions=expansions,
        ocr_expansions={"fuhr": [
            {"lexeme": "fuer", "cost": 0.35, "chunk_freq": 2, "source": "ocr"},
            {"lexeme": "fubr", "cost": 0.4, "chunk_freq": 1, "source": "ocr"},
        ]},
        max_total_variants=50,
    )
    got = expansions["fuhr"]
    assert [e["lexeme"] for e in got] == ["fuer", "fubr"]
    assert got[0]["source"] == "both"      # trigram dict kept, relabeled
    assert "similarity" in got[0]          # trigram entry preserved
    assert got[1]["source"] == "ocr"


def test_merge_cap_prefers_cheapest_across_tokens():
    expansions = {}
    _merge_ocr_expansions(
        tokens=["fuhr", "golos"],
        expansions=expansions,
        ocr_expansions={
            "fuhr": [{"lexeme": "fuer", "cost": 0.35, "chunk_freq": 2}],
            "golos": [{"lexeme": "gol0s", "cost": 0.9, "chunk_freq": 1},
                      {"lexeme": "solos", "cost": 0.5, "chunk_freq": 4}],
        },
        max_total_variants=2,
    )
    all_lex = [e["lexeme"] for tok in expansions for e in expansions[tok]]
    assert sorted(all_lex) == ["fuer", "solos"]  # two cheapest survive the cap


def test_config_json_carries_ocr_knobs():
    d = FuzzyLexConfig().to_json()
    assert d["ocr_enabled"] is True
    assert d["ocr_max_cost"] == 1.5
    assert d["ocr_top_k"] == 4
    assert d["ocr_freq_ceiling"] == 500
    assert d["trgm_min_token_len"] == 5


def test_trgm_min_token_len_gates_trigram_but_not_ocr(monkeypatch):
    calls = _patch_ocr(monkeypatch, {
        "fuhr": [{"lexeme": "fuer", "cost": 0.35, "chunk_freq": 2, "source": "ocr"}],
    })
    conn = FakeConn()
    _, expansions = fetch_fuzzy_variants_for_tokens(
        conn, tokens=["fuhr", "greenglass"], chunk_pv="pv",
        collection_slug="silvermaster", norm_version="norm_v1",
        config=FuzzyLexConfig(dictionary_build_id=7),
    )
    # 'fuhr' (len 4 < 5): no trigram SQL; 'greenglass' (len 10): one query.
    trgm_queries = [p for c in conn.cursors for (_, p) in c.executed if p]
    assert [p[0] for p in trgm_queries] == ["greenglass"]
    # The ocr channel still received BOTH tokens and expanded fuhr.
    assert calls["args"]["tokens"] == ["fuhr", "greenglass"]
    assert [e["lexeme"] for e in expansions["fuhr"]] == ["fuer"]

    # Lowering the gate re-enables trigram for short tokens.
    conn2 = FakeConn()
    fetch_fuzzy_variants_for_tokens(
        conn2, tokens=["fuhr"], chunk_pv="pv", collection_slug="silvermaster",
        norm_version="norm_v1",
        config=FuzzyLexConfig(dictionary_build_id=7, trgm_min_token_len=4),
    )
    trgm_queries2 = [p for c in conn2.cursors for (_, p) in c.executed if p]
    assert [p[0] for p in trgm_queries2] == ["fuhr"]
