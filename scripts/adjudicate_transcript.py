#!/usr/bin/env python3
"""
Ensemble adjudicator for the Bentley deposition transcription pipeline.

Merges up to five independent readings of each page (three vision models,
AWS Textract, and the PDF's embedded text layer) into:

  BASE/final/{page:04d}.txt      adjudicated transcript (reference line structure)
  BASE/review_queue.json         positions a human should look at
  BASE/confusion_pairs.csv       embedded-OCR token -> accepted token substitutions
  BASE/adjudication_report.json  per-page / total counts and per-reading stats

Pure stdlib, deterministic, offline: no network, no database.

Usage:
    python scripts/adjudicate_transcript.py \
        --base data/transcripts/bentley_deposition --pages 3-120
"""

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path

VISION_MODELS = ("gpt-5.5", "gpt-5.2", "gpt-4.1", "olmocr2")
READING_ORDER = ("gpt-5.5", "gpt-5.2", "gpt-4.1", "textract", "embedded")
READING_RANK = {name: i for i, name in enumerate(READING_ORDER)}
# Unknown (calibration) vision models rank after the known ones but before
# the OCR layers, so their surface forms still beat textract/embedded casing.
_UNKNOWN_VISION_RANK = READING_RANK["textract"] - 0.5


def reading_rank(name):
    return READING_RANK.get(name, _UNKNOWN_VISION_RANK)

# Strip leading/trailing punctuation, keep internal apostrophes/hyphens.
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_NUMBER_RE = re.compile(r"\d+([.,:/-]\d+)*")
_SERIAL_RE = re.compile(r"\d+(-\d+)+")
_ILLEGIBLE_RE = re.compile(r"\[(illegible|\?[^\]]*)\]", re.IGNORECASE)
_STAMP_LINE_RE = re.compile(r"^\s*\[stamp\b", re.IGNORECASE)
_SENTENCE_END = (".", "!", "?")


# ---------------------------------------------------------------- tokens

def comp_core(token):
    """Surface with leading/trailing punctuation stripped (case preserved)."""
    return _EDGE_PUNCT_RE.sub("", token)


def comp_key(token):
    """Comparison key: lowercased core."""
    return comp_core(token).lower()


def tokenize_lines(text):
    """Whitespace-tokenize preserving line structure.

    Returns (tokens, line_idxs) where tokens is the flat surface list and
    line_idxs[i] is the list of token indices on line i.
    """
    tokens = []
    line_idxs = []
    for line in text.split("\n"):
        idxs = []
        for tok in line.split():
            idxs.append(len(tokens))
            tokens.append(tok)
        line_idxs.append(idxs)
    return tokens, line_idxs


def rebuild_line(orig_line, chosen_tokens):
    """Substitute chosen tokens into the original line, keeping whitespace."""
    parts = re.split(r"(\s+)", orig_line)
    out = []
    ti = 0
    for part in parts:
        if part and not part.isspace():
            out.append(chosen_tokens[ti])
            ti += 1
        else:
            out.append(part)
    return "".join(out)


# ---------------------------------------------------------------- classifiers

def is_all_caps(surface):
    core = comp_core(surface)
    return len(core) >= 3 and core.isalpha() and core.isupper()


def is_capitalized(surface):
    core = comp_core(surface)
    return len(core) >= 2 and core[0].isupper() and core[1:].islower()


def is_number_like(surface):
    key = comp_key(surface)
    return bool(key) and bool(_NUMBER_RE.fullmatch(key))


def is_serial_like(surface):
    return bool(_SERIAL_RE.fullmatch(comp_key(surface)))


def has_illegible(surface):
    return bool(_ILLEGIBLE_RE.search(surface))


def review_reason(surfaces, ref_surface, sentence_initial, gazetteer):
    """(reason or None, name_like flag) for a disagreement position."""
    name_like = (
        any(is_all_caps(s) for s in surfaces)
        or (is_capitalized(ref_surface) and not sentence_initial)
        or any(comp_key(s) in gazetteer for s in surfaces)
    )
    if any(has_illegible(s) for s in surfaces):
        return "illegible marker", name_like
    if any(is_serial_like(s) for s in surfaces):
        return "serial-like disagreement", name_like
    if name_like:
        return "name-like disagreement", True
    if any(is_number_like(s) for s in surfaces):
        return "number disagreement", name_like
    return None, name_like


# ---------------------------------------------------------------- inputs

def load_readings(base, page):
    """Available readings for a page, in READING_ORDER. Empty texts excluded.

    Vision readings are discovered from pages/{page:04d}.*.json so calibration
    runs with other models (e.g. gpt-5.4-mini) adjudicate too; known models
    keep their READING_ORDER priority, unknown ones follow alphabetically.
    """
    readings = {}
    page_files = sorted((base / "pages").glob(f"{page:04d}.*.json"))
    discovered = [p.name.split(".", 1)[1].rsplit(".json", 1)[0] for p in page_files]
    ordered = [m for m in VISION_MODELS if m in discovered] + \
              [m for m in discovered if m not in VISION_MODELS]
    for model in ordered:
        path = base / "pages" / f"{page:04d}.{model}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            text = data.get("text")
            if isinstance(text, str) and text.strip():
                readings[model] = text
    for name, sub in (("textract", "textract"), ("embedded", "embedded")):
        path = base / sub / f"{page:04d}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                readings[name] = text
    return readings


def load_gazetteer(base):
    """Set of comparison keys: full entries plus their individual words."""
    keys = set()
    path = base / "gazetteer.txt"
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        full = comp_key(line)
        if full:
            keys.add(full)
        for word in line.split():
            wkey = comp_key(word)
            if len(wkey) >= 3:
                keys.add(wkey)
    return keys


# ---------------------------------------------------------------- alignment

def align_to_reference(ref_keys, cand_tokens):
    """Align a candidate reading against the reference key stream.

    Returns (aligned, insertions): aligned[i] is the candidate surface for
    reference position i or None on a gap; insertions maps a reference anchor
    position (token inserted *before* that position) to the list of inserted
    candidate surfaces.
    """
    cand_keys = [comp_key(t) for t in cand_tokens]
    sm = difflib.SequenceMatcher(None, ref_keys, cand_keys, autojunk=False)
    aligned = [None] * len(ref_keys)
    insertions = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                aligned[i1 + k] = cand_tokens[j1 + k]
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                aligned[i1 + k] = cand_tokens[j1 + k]
            if j2 - j1 > n:
                insertions.setdefault(i2, []).extend(cand_tokens[j1 + n:j2])
        elif tag == "insert":
            insertions.setdefault(i1, []).extend(cand_tokens[j1:j2])
        # "delete": reference positions keep None for this reading
    return aligned, insertions


# ---------------------------------------------------------------- adjudication

def adjudicate_page(page, readings, gazetteer):
    """Adjudicate one page. Returns a result dict (see keys at the bottom)."""
    # Reference = first vision reading (readings preserves discovery order,
    # known models first); textract/embedded are never the reference.
    ref_name = next(m for m in readings if m not in ("textract", "embedded"))
    ref_text = readings[ref_name]
    ref_tokens, line_idxs = tokenize_lines(ref_text)
    ref_keys = [comp_key(t) for t in ref_tokens]
    npos = len(ref_tokens)

    aligned = {}
    insertions_by_reading = {}
    for name, text in readings.items():
        if name == ref_name:
            aligned[name] = list(ref_tokens)
            continue
        cand_tokens, _ = tokenize_lines(text)
        aligned[name], insertions_by_reading[name] = align_to_reference(
            ref_keys, cand_tokens
        )

    n_readings = len(readings)
    threshold = 3 if n_readings >= 4 else 2

    raw_lines = ref_text.split("\n")
    stamp_pos = set()
    line_first = set()
    for idxs, raw_line in zip(line_idxs, raw_lines):
        if idxs:
            line_first.add(idxs[0])
        if _STAMP_LINE_RE.match(raw_line):
            stamp_pos.update(idxs)

    counts = {
        "positions": npos,
        "unanimous": 0,
        "majority": 0,
        "silent_auto": 0,
        "review_items": 0,
        "illegible": 0,
    }
    reading_stats = {
        name: {"positions": 0, "disagreements": 0} for name in readings
    }
    chosen = [None] * npos
    # (pos, tiebreak, readings_map, default, default_source, reason, name_like)
    raw_items = []

    for pos in range(npos):
        cand = {}
        for name in readings:  # READING_ORDER preserved by construction
            tok = aligned[name][pos]
            if tok is not None:
                cand[name] = tok
        votes = {}
        for name, tok in cand.items():
            votes.setdefault(comp_key(tok), []).append(name)
        ranked = sorted(
            votes.items(),
            key=lambda kv: (-len(kv[1]), reading_rank(kv[1][0])),
        )
        winner_key, supporters = ranked[0]
        winner_surface = cand[supporters[0]]
        nvotes = len(supporters)

        if nvotes == n_readings:
            category = "unanimous"
        elif nvotes >= threshold:
            category = "majority"
        else:
            category = "disagreement"

        ref_surface = ref_tokens[pos]
        sentence_initial = (
            pos == 0
            or pos in line_first
            or ref_tokens[pos - 1].endswith(_SENTENCE_END)
        )

        if pos in stamp_pos:
            # Stamps exist only in vision readings: keep the reference surface.
            final = ref_surface
            counts["unanimous" if category == "unanimous" else
                   "majority" if category == "majority" else
                   "silent_auto"] += 1
            if any(is_serial_like(s) for s in cand.values()):
                # Archival serials researchers cite: always reviewed.
                source = "majority" if category != "disagreement" else "reference"
                raw_items.append(
                    (pos, "", dict(cand), final, source, "stamp serial", False)
                )
                counts["review_items"] += 1
        elif category != "disagreement":
            final = winner_surface
            counts[category] += 1
        else:
            reason, name_like = review_reason(
                list(cand.values()), ref_surface, sentence_initial, gazetteer
            )
            if reason is None:
                final = winner_surface  # plurality, silently auto-accepted
                counts["silent_auto"] += 1
            else:
                gaz_keys = sorted(k for k in votes if k and k in gazetteer)
                if len(gaz_keys) == 1:
                    default = cand[votes[gaz_keys[0]][0]]
                    source = "gazetteer"
                else:
                    default = ref_surface
                    source = "reference"
                final = default
                raw_items.append(
                    (pos, "", dict(cand), default, source, reason, name_like)
                )
                counts["review_items"] += 1

        chosen[pos] = final
        if has_illegible(final):
            counts["illegible"] += 1
        for name, tok in cand.items():
            reading_stats[name]["positions"] += 1
            if comp_key(tok) != comp_key(final):
                reading_stats[name]["disagreements"] += 1

    # Insertions: ignored unless >=2 readings agree on the same tokens at the
    # same anchor -- then flagged for review, never silently added.
    ins_groups = {}
    for name, ins in insertions_by_reading.items():
        for anchor, toks in ins.items():
            keyjoin = " ".join(comp_key(t) for t in toks).strip()
            if not keyjoin:
                continue
            ins_groups.setdefault((anchor, keyjoin), {})[name] = " ".join(toks)
    ins_counter = {}
    for (anchor, keyjoin), group in sorted(ins_groups.items()):
        if len(group) < 2:
            continue
        n = ins_counter.get(anchor, 0) + 1
        ins_counter[anchor] = n
        suffix = "-ins" if n == 1 else f"-ins{n}"
        surfaces = list(group.values())
        name_like = any(
            is_all_caps(s) or comp_key(s) in gazetteer
            for surf in surfaces
            for s in surf.split()
        )
        raw_items.append(
            (anchor, suffix, dict(group), "", "reference",
             "agreed insertion", name_like)
        )
        counts["review_items"] += 1

    # Final text: reference line structure with chosen tokens substituted.
    out_lines = [
        rebuild_line(raw_line, [chosen[i] for i in idxs])
        for idxs, raw_line in zip(line_idxs, raw_lines)
    ]
    final_text = "\n".join(out_lines)
    if not final_text.endswith("\n"):
        final_text += "\n"

    # Contexts from the flattened final token stream.
    offsets = []
    cursor = 0
    for tok in chosen:
        if offsets:
            cursor += 1
        offsets.append((cursor, cursor + len(tok)))
        cursor += len(tok)
    joined = " ".join(chosen)

    review_items = []
    for pos, suffix, cand_map, default, source, reason, name_like in sorted(
        raw_items, key=lambda t: (t[0], t[1])
    ):
        if pos < npos:
            start, end = offsets[pos]
        else:
            start = end = len(joined)
        review_items.append({
            "id": f"p{page:04d}-t{pos:04d}{suffix}",
            "page": page,
            "pos": pos,
            "context_before": joined[max(0, start - 30):start],
            "context_after": joined[end:end + 30],
            "readings": cand_map,
            "default": default,
            "default_source": source,
            "reason": reason,
            "name_like": name_like,
        })

    return {
        "final_text": final_text,
        "final_tokens": chosen,
        "review_items": review_items,
        "counts": counts,
        "reading_stats": reading_stats,
        "readings_available": list(readings),
    }


# ---------------------------------------------------------------- confusions

# A replace block spanning more than this fraction of the page's tokens is
# whole-page misalignment (e.g. a rotated or unrelated embedded layer), not a
# run of real substitutions.
MISALIGN_BLOCK_FRACTION = 0.3
# Minimum character-level similarity between the embedded and accepted keys
# for a pair to count as a plausible OCR confusion rather than alignment noise.
CONFUSION_MIN_RATIO = 0.5


def confusion_pairs_for_page(page, embedded_text, final_tokens):
    """(page, embedded_token, accepted_token) substitution rows.

    Pairs come from difflib "replace" opcodes between the embedded and final
    key streams, with two noise filters: replace blocks whose token count
    exceeds MISALIGN_BLOCK_FRACTION of the page token count are skipped whole
    (whole-page misalignment), and individual pairs are kept only when the
    embedded/accepted keys are at least CONFUSION_MIN_RATIO similar.
    """
    emb_tokens, _ = tokenize_lines(embedded_text)
    fin_keys = [comp_key(t) for t in final_tokens]
    emb_keys = [comp_key(t) for t in emb_tokens]
    npos = len(final_tokens)
    sm = difflib.SequenceMatcher(None, fin_keys, emb_keys, autojunk=False)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        if npos and max(i2 - i1, j2 - j1) > MISALIGN_BLOCK_FRACTION * npos:
            continue  # whole-page misalignment, not real substitutions
        for k in range(min(i2 - i1, j2 - j1)):
            accepted = final_tokens[i1 + k]
            embedded = emb_tokens[j1 + k]
            akey, ekey = comp_key(accepted), comp_key(embedded)
            if not (akey and ekey) or akey == ekey:
                continue
            ratio = difflib.SequenceMatcher(None, ekey, akey).ratio()
            if ratio < CONFUSION_MIN_RATIO:
                continue
            rows.append((page, embedded, accepted))
    return rows


# ---------------------------------------------------------------- driver

def parse_pages(spec):
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", spec)
    if not m:
        raise ValueError(f"bad --pages value {spec!r}, expected A-B")
    first = int(m.group(1))
    last = int(m.group(2)) if m.group(2) else first
    if first < 1 or last < first:
        raise ValueError(f"bad --pages range {spec!r}")
    return first, last


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Adjudicate multi-reading page transcripts into finals."
    )
    ap.add_argument("--base", required=True,
                    help="transcript base dir (e.g. data/transcripts/bentley_deposition)")
    ap.add_argument("--pages", required=True,
                    help="1-based inclusive page range A-B (or a single page A)")
    args = ap.parse_args(argv)

    base = Path(args.base)
    try:
        first, last = parse_pages(args.pages)
    except ValueError as exc:
        ap.error(str(exc))

    gazetteer = load_gazetteer(base)
    (base / "final").mkdir(parents=True, exist_ok=True)

    all_items = []
    all_confusions = []
    report_pages = {}
    totals = {
        "positions": 0, "unanimous": 0, "majority": 0,
        "silent_auto": 0, "review_items": 0, "illegible": 0,
    }
    reading_totals = {}

    for page in range(first, last + 1):
        readings = load_readings(base, page)
        if not any(m not in ("textract", "embedded") for m in readings):
            print(f"page {page}: no vision reading, skipped", file=sys.stderr)
            report_pages[str(page)] = {
                "skipped": True, "reason": "no vision reading"
            }
            continue
        result = adjudicate_page(page, readings, gazetteer)

        (base / "final" / f"{page:04d}.txt").write_text(
            result["final_text"], encoding="utf-8"
        )
        all_items.extend(result["review_items"])
        if "embedded" in readings:
            all_confusions.extend(confusion_pairs_for_page(
                page, readings["embedded"], result["final_tokens"]
            ))

        page_report = dict(result["counts"])
        page_report["readings_available"] = result["readings_available"]
        report_pages[str(page)] = page_report
        for key in totals:
            totals[key] += result["counts"][key]
        for name, stats in result["reading_stats"].items():
            agg = reading_totals.setdefault(
                name, {"positions": 0, "disagreements": 0}
            )
            agg["positions"] += stats["positions"]
            agg["disagreements"] += stats["disagreements"]

    all_items.sort(key=lambda it: (it["page"], it["pos"], it["id"]))
    with open(base / "review_queue.json", "w", encoding="utf-8") as fh:
        json.dump(all_items, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    with open(base / "confusion_pairs.csv", "w", encoding="utf-8",
              newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["page", "embedded_token", "accepted_token"])
        writer.writerows(all_confusions)

    per_reading = {}
    for name in READING_ORDER:
        if name not in reading_totals:
            continue
        stats = reading_totals[name]
        rate = (round(stats["disagreements"] / stats["positions"], 4)
                if stats["positions"] else 0.0)
        per_reading[name] = {
            "positions": stats["positions"],
            "disagreements": stats["disagreements"],
            "rate": rate,
        }

    report = {"pages": report_pages, "totals": totals,
              "per_reading": per_reading}
    with open(base / "adjudication_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"pages {first}-{last}: {totals['positions']} positions, "
          f"{totals['review_items']} review items, "
          f"{len(all_confusions)} confusion pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
