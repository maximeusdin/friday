"""Tests for scripts/adjudicate_transcript.py (ensemble adjudicator).

Synthetic multi-reading fixtures only: no network, no DB, pure stdlib + pytest.
"""

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adjudicate_transcript.py"
_spec = importlib.util.spec_from_file_location("adjudicate_transcript", SCRIPT)
adj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adj)


# ---------------------------------------------------------------- helpers

def make_base(tmp_path, gazetteer=""):
    base = tmp_path / "base"
    for sub in ("pages", "textract", "embedded"):
        (base / sub).mkdir(parents=True)
    (base / "gazetteer.txt").write_text(gazetteer, encoding="utf-8")
    return base


def write_vision(base, page, model, text):
    payload = {
        "page": page,
        "model": model,
        "text": text,
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }
    path = base / "pages" / f"{page:04d}.{model}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_textract(base, page, text):
    (base / "textract" / f"{page:04d}.txt").write_text(text, encoding="utf-8")


def write_embedded(base, page, text):
    (base / "embedded" / f"{page:04d}.txt").write_text(text, encoding="utf-8")


def run(base, pages="1"):
    assert adj.main(["--base", str(base), "--pages", pages]) == 0


def read_final(base, page):
    return (base / "final" / f"{page:04d}.txt").read_text(encoding="utf-8")


def read_queue(base):
    return json.loads((base / "review_queue.json").read_text(encoding="utf-8"))


def read_report(base):
    return json.loads(
        (base / "adjudication_report.json").read_text(encoding="utf-8")
    )


def read_confusions(base):
    with open(base / "confusion_pairs.csv", encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


# ---------------------------------------------------------------- tests

def test_unanimous(tmp_path):
    base = make_base(tmp_path)
    text = "The quick brown fox jumps.\nOver the lazy dog today."
    for model in ("gpt-5.5", "gpt-5.2", "gpt-4.1"):
        write_vision(base, 1, model, text)
    write_textract(base, 1, text)
    write_embedded(base, 1, text)
    run(base)

    assert read_final(base, 1) == text + "\n"
    assert read_queue(base) == []
    page = read_report(base)["pages"]["1"]
    assert page["positions"] == 10
    assert page["unanimous"] == 10
    assert page["review_items"] == 0


def test_majority_over_garble(tmp_path):
    # FUHR vs FUER, 3v2 across 5 readings: majority accepted silently.
    base = make_base(tmp_path)
    good = "Miss Bentley met FUHR in Washington."
    bad = "Miss Bentley met FUER in Washington."
    bad_emb = "Miss Bentley met FUER# in Washington."
    write_vision(base, 1, "gpt-5.5", good)
    write_vision(base, 1, "gpt-5.2", good)
    write_vision(base, 1, "gpt-4.1", bad)
    write_textract(base, 1, good)
    write_embedded(base, 1, bad_emb)
    run(base)

    final = read_final(base, 1)
    assert "FUHR" in final
    assert "FUER" not in final
    assert read_queue(base) == []
    page = read_report(base)["pages"]["1"]
    assert page["majority"] == 1
    assert page["unanimous"] == page["positions"] - 1

    # Embedded-vs-final confusion pair recorded with surfaces.
    rows = read_confusions(base)
    assert rows[0] == ["page", "embedded_token", "accepted_token"]
    assert ["1", "FUER#", "FUHR"] in rows[1:]


def test_gazetteer_default(tmp_path):
    # 2v2 among 4 readings (threshold 3): review item defaults to the single
    # gazetteer-matching candidate.
    base = make_base(tmp_path, gazetteer="Nathan Gregory Silvermaster\n")
    write_vision(base, 1, "gpt-5.5", "Talked with Silvermister about money")
    write_vision(base, 1, "gpt-5.2", "Talked with Silvermister about money")
    write_vision(base, 1, "gpt-4.1", "Talked with Silvermaster about money")
    write_textract(base, 1, "Talked with Silvermaster about money")
    run(base)

    assert "Silvermaster" in read_final(base, 1)
    items = read_queue(base)
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "p0001-t0002"
    assert item["page"] == 1
    assert item["pos"] == 2
    assert item["default"] == "Silvermaster"
    assert item["default_source"] == "gazetteer"
    assert item["reason"] == "name-like disagreement"
    assert item["name_like"] is True
    assert item["readings"]["gpt-5.5"] == "Silvermister"
    assert item["readings"]["gpt-4.1"] == "Silvermaster"


def test_unique_insertion_ignored(tmp_path):
    base = make_base(tmp_path)
    write_vision(base, 1, "gpt-5.5", "alpha beta gamma")
    write_vision(base, 1, "gpt-5.2", "alpha beta gamma")
    write_vision(base, 1, "gpt-4.1", "alpha beta EXTRA gamma")
    write_textract(base, 1, "alpha beta gamma")
    run(base)

    assert "EXTRA" not in read_final(base, 1)
    assert read_queue(base) == []


def test_two_reading_insertion_flagged(tmp_path):
    base = make_base(tmp_path)
    write_vision(base, 1, "gpt-5.5", "alpha beta gamma")
    write_vision(base, 1, "gpt-5.2", "alpha beta gamma")
    write_vision(base, 1, "gpt-4.1", "alpha beta COPY gamma")
    write_textract(base, 1, "alpha beta COPY gamma")
    run(base)

    # Never silently added...
    assert "COPY" not in read_final(base, 1)
    # ...but flagged because two readings agree on the insertion.
    items = read_queue(base)
    assert len(items) == 1
    item = items[0]
    assert item["reason"] == "agreed insertion"
    assert item["pos"] == 2
    assert item["readings"] == {"gpt-4.1": "COPY", "textract": "COPY"}
    assert item["default"] == ""


def test_stamp_always_review(tmp_path):
    # Serial-bearing stamps are review items even when every reading that has
    # them agrees.
    base = make_base(tmp_path)
    text = "First line of testimony.\n[stamp: 65-56402-220]\nLast line here."
    for model in ("gpt-5.5", "gpt-5.2", "gpt-4.1"):
        write_vision(base, 1, model, text)
    write_textract(base, 1, "First line of testimony.\nLast line here.")
    run(base)

    assert "[stamp: 65-56402-220]" in read_final(base, 1)
    items = read_queue(base)
    assert len(items) == 1
    item = items[0]
    assert item["reason"] == "stamp serial"
    assert "65-56402-220" in item["default"]
    assert item["default_source"] == "majority"


def test_missing_readings_two_sources(tmp_path):
    # Only gpt-5.5 + textract: threshold drops to 2; a 1v1 name-like split
    # becomes a review item defaulting to the reference reading.
    base = make_base(tmp_path)
    write_vision(base, 1, "gpt-5.5", "saw HELLER today")
    write_textract(base, 1, "saw MELLER today")
    run(base)

    assert "HELLER" in read_final(base, 1)
    items = read_queue(base)
    assert len(items) == 1
    item = items[0]
    assert item["default"] == "HELLER"
    assert item["default_source"] == "reference"
    assert item["reason"] == "name-like disagreement"
    assert item["readings"] == {"gpt-5.5": "HELLER", "textract": "MELLER"}
    page = read_report(base)["pages"]["1"]
    assert page["unanimous"] == 2
    assert page["review_items"] == 1
    assert page["readings_available"] == ["gpt-5.5", "textract"]


def test_empty_embedded_file(tmp_path):
    base = make_base(tmp_path)
    text = "plain agreed words here"
    write_vision(base, 1, "gpt-5.5", text)
    write_vision(base, 1, "gpt-4.1", text)
    write_textract(base, 1, text)
    write_embedded(base, 1, "")  # exists but empty: treated as absent
    run(base)

    assert read_final(base, 1) == text + "\n"
    page = read_report(base)["pages"]["1"]
    assert "embedded" not in page["readings_available"]
    assert read_confusions(base) == [["page", "embedded_token",
                                      "accepted_token"]]


def test_confusion_pairs_similarity_filter():
    # A replace pair is only emitted when the embedded/accepted keys are at
    # least 50% similar: "gamna"->"gamma" (0.8) kept, "qqqq"->"kappa" (0.0)
    # dropped as alignment noise.
    final_tokens = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    embedded = "alpha beta gamna delta epsilon zeta eta theta iota qqqq"
    rows = adj.confusion_pairs_for_page(7, embedded, final_tokens)
    assert (7, "gamna", "gamma") in rows
    assert all(emb != "qqqq" for _, emb, _ in rows)


def test_confusion_whole_page_misalignment_skipped():
    # An embedded layer that is entirely different text produces one giant
    # replace block (> 30% of the page tokens): skipped whole, no pairs.
    final_tokens = "one two three four five six seven eight nine ten".split()
    embedded = "uno dos tres cuatro cinco seis siete ocho nueve diez"
    assert adj.confusion_pairs_for_page(3, embedded, final_tokens) == []


def test_confusion_small_block_still_reported():
    # A short replace block within an otherwise aligned page stays reported
    # (2 tokens on a 10-token page is under the 30% misalignment cutoff).
    final_tokens = "one two three four five six seven eight nine ten".split()
    embedded = "one two three four fivo sax seven eight nine ten"
    rows = adj.confusion_pairs_for_page(4, embedded, final_tokens)
    assert (4, "fivo", "five") in rows
    assert (4, "sax", "six") in rows


def test_page_range_and_report_totals(tmp_path):
    base = make_base(tmp_path)
    write_vision(base, 1, "gpt-5.5", "one two three")
    write_vision(base, 1, "gpt-4.1", "one two three")
    write_vision(base, 2, "gpt-5.5", "four five")
    write_vision(base, 2, "gpt-4.1", "four five")
    run(base, "1-3")  # page 3 has no readings at all

    report = read_report(base)
    assert report["totals"]["positions"] == 5
    assert report["pages"]["3"] == {"skipped": True,
                                    "reason": "no vision reading"}
    assert (base / "final" / "0001.txt").exists()
    assert (base / "final" / "0002.txt").exists()
    assert not (base / "final" / "0003.txt").exists()
    for name in ("gpt-5.5", "gpt-4.1"):
        stats = report["per_reading"][name]
        assert stats["positions"] == 5
        assert stats["disagreements"] == 0
        assert stats["rate"] == 0.0
