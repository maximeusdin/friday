"""Unit tests for the pure rare->name pairing in scripts/sweep_ocr_variants.py.

No database and no retrieval.ocr_variants import: skeleton/distance are
injected synthetic callables, mirroring how the CLI binds the real confusion
config. The synthetic confusion collapses {e, h} and {c, o} (representative =
alphabetically first member), so FUER — the OCR garble of typed FUHR — shares
a skeleton with fuhr and sits one cheap class-internal substitution away.
"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_ocr_variants.py"
_spec = importlib.util.spec_from_file_location("sweep_ocr_variants", SCRIPT)
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)

pair_rare_to_names = sw.pair_rare_to_names
PairRow = sw.PairRow


# ============================================================================
# Synthetic confusion (no config file): classes {e,h} -> e, {c,o} -> c
# ============================================================================

_REP = {"e": "e", "h": "e", "c": "c", "o": "c"}

_SUB_SAME_CLASS = 0.25
_SUB_OTHER = 1.0
_INDEL = 0.8


def _skel(tok):
    t = "".join(ch for ch in tok.lower() if ch.isalnum())
    return "".join(_REP.get(ch, ch) for ch in t)


def _dist(a, b):
    """Positional compare: same char 0, same class 0.25, else 1.0; unaligned
    tail costs 0.8/char. Enough structure for pairing tests — the real DP
    lives in retrieval.ocr_variants."""
    n = min(len(a), len(b))
    cost = 0.0
    for i in range(n):
        ca, cb = a[i], b[i]
        if ca == cb:
            continue
        if _REP.get(ca, ca) == _REP.get(cb, cb):
            cost += _SUB_SAME_CLASS
        else:
            cost += _SUB_OTHER
    return cost + _INDEL * (max(len(a), len(b)) - n)


def _pair(lexemes, **kw):
    kw.setdefault("skeleton_fn", _skel)
    kw.setdefault("distance_fn", _dist)
    return pair_rare_to_names(lexemes, **kw)


# ============================================================================
# Tests
# ============================================================================

def test_fuer_pairs_with_fuhr():
    """The motivating case: rare garble 'fuer' pairs with name 'fuhr'."""
    res = _pair([("fuer", 2), ("fuhr", 12), ("silvermaster", 40), ("the", 500)])
    assert res.rows == [PairRow("fuer", "fuhr", 0.25, 2, 12)]
    assert res.name_pool == 3  # fuhr, silvermaster, the
    assert res.rare_pool == 1  # fuer


def test_pool_thresholds_are_inclusive():
    # freq 6 > max_rare_freq=5 and freq 7 < min_name_freq=8: neither pool.
    res = _pair([("fuer", 6), ("fuhr", 7)])
    assert res.rows == []
    assert res.name_pool == 0
    assert res.rare_pool == 0
    # Boundary values qualify: freq 5 is rare, freq 8 is a name.
    res = _pair([("fuer", 5), ("fuhr", 8)])
    assert res.rows == [PairRow("fuer", "fuhr", 0.25, 5, 8)]


def test_gazetteer_promotes_low_freq_name():
    """A low-freq lexeme joins the name pool only via the gazetteer."""
    lexemes = [("fuer", 1), ("fuhr", 3)]
    res = _pair(lexemes)
    assert res.rows == []
    res = _pair(lexemes, gazetteer={"fuhr"})
    assert res.rows == [PairRow("fuer", "fuhr", 0.25, 1, 3)]
    assert res.name_pool == 1


def test_rare_pool_exclusions():
    """len<4, pure digits, stopwords, and empty-skeleton tokens never pair."""
    res = _pair(
        [
            ("fue", 1),      # too short
            ("1943", 1),     # pure digits
            ("that", 1),     # stopword
            ("''''", 1),     # skeleton collapses to empty
            ("garb", 1),     # the only legitimate rare candidate
            ("fuhr", 20),    # a name, so the pools are non-trivial
        ]
    )
    assert res.rare_pool == 2  # garb + '''' (dropped later on empty skeleton)
    assert res.rows == []      # garb's skeleton matches nothing


def test_max_cost_filters_pairs():
    lexemes = [("fuer", 2), ("fuhr", 12)]
    assert _pair(lexemes, max_cost=0.1).rows == []
    assert len(_pair(lexemes, max_cost=0.25).rows) == 1  # inclusive bound


def test_token_in_both_pools_never_pairs_with_itself():
    # freq 4 <= max_rare_freq AND gazetteer member: both pools, no self-pair.
    res = _pair([("fuhr", 4)], gazetteer={"fuhr"})
    assert res.name_pool == 1
    assert res.rare_pool == 1
    assert res.rows == []


def test_rows_sorted_cost_asc_then_candidate_freq_desc():
    # All four tokens share skeleton 'eeee' under {e,h} -> e.
    res = _pair(
        [
            ("hehe", 1),   # rare
            ("eehe", 20),  # cost 0.25
            ("heee", 9),   # cost 0.25, lower freq
            ("eeee", 15),  # cost 0.50
        ]
    )
    assert [(r.candidate_name, r.cost, r.candidate_freq) for r in res.rows] == [
        ("eehe", 0.25, 20),
        ("heee", 0.25, 9),
        ("eeee", 0.50, 15),
    ]
    assert all(r.rare_token == "hehe" for r in res.rows)


def test_csv_header_matches_row_shape():
    assert sw.CSV_HEADER == ["rare_token", "candidate_name", "cost", "rare_freq", "candidate_freq"]
    assert list(PairRow._fields) == sw.CSV_HEADER
