#!/usr/bin/env python3
"""Apply reviewer decisions (the CSV exported by the review HTML) to the
adjudicated Bentley deposition transcripts BASE/final/{page:04d}.txt.

The CSV has header item_id,page,decision,note where decision is the chosen
reading text, or [illegible]. Each decision is matched to its
BASE/review_queue.json item and applied with the same in-place token
substitution the adjudicator uses: scripts/adjudicate_transcript.py's
tokenize_lines + rebuild_line (whitespace-preserving substitution into the
final line structure), coordinated via the item's `pos` -- the 0-based
whitespace-token index in the page's final text. The token currently at `pos`
is verified against the item's default reading, with a small +/- positional
search to absorb drift, and the decision is skipped when no safe match is
found. Insertion items (id suffix "-ins", empty default) insert the decision
before the anchor token instead of replacing it.

Applied/skipped counts are reported and review_queue.json is rewritten marking
decided items. Re-applying the same CSV is a no-op: items whose queue entry is
already decided with the same decision text are skipped with an "already
applied" notice, and "-ins" insertions are skipped when the decision token(s)
are already present at/immediately before the anchor. This script touches NO
database.

Usage:
  python scripts/apply_review_decisions.py \
      --base data/transcripts/bentley_deposition \
      --decisions review_decisions.csv
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import sys

SEARCH_WINDOW = 5  # +/- token positions to search when pos has drifted


def norm_token(t):
    """Lowercase and strip everything but letters/digits (matching the
    review-HTML fuzzy matcher), so 'FUHR,' == 'fuhr' == '[?FUHR]'."""
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def find_target(tokens, pos, expected_token):
    """Index of the token to substitute: `pos` when it matches
    `expected_token` (case/punctuation-insensitively), else the nearest
    matching position within +/- SEARCH_WINDOW. None when no safe match."""
    if not tokens:
        return None

    def matches(i):
        if i < 0 or i >= len(tokens):
            return False
        if expected_token is None:
            return True
        tok = tokens[i]
        return tok == expected_token or (
            norm_token(tok) != "" and norm_token(tok) == norm_token(expected_token)
        )

    if expected_token is None:
        return pos if 0 <= pos < len(tokens) else None
    for delta in range(SEARCH_WINDOW + 1):
        for i in ((pos,) if delta == 0 else (pos + delta, pos - delta)):
            if matches(i):
                return i
    return None


def load_adjudicator():
    """Import scripts/adjudicate_transcript.py by path (works regardless of
    how this script is invoked). Returns the module or None."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "adjudicate_transcript.py")
    if not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("adjudicate_transcript", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"warning: could not import scripts/adjudicate_transcript.py ({e})")
        return None


def make_substitution_fn():
    """(fn, description). fn(text, pos, new_text, expected_token) ->
    (new_full_text, matched_pos_or_None).

    Preferred path re-runs the adjudicator's own machinery: tokenize_lines to
    flatten the final text to the token stream `pos` indexes, then
    rebuild_line to substitute in place preserving all original whitespace --
    byte-identical to how the adjudicator wrote the final text. A local
    regex-based equivalent is used only if the adjudicator module is absent.
    """
    mod = load_adjudicator()
    if mod is not None and hasattr(mod, "tokenize_lines") and hasattr(mod, "rebuild_line"):

        def subst(text, pos, new_text, expected_token=None):
            tokens, line_idxs = mod.tokenize_lines(text)
            target = find_target(tokens, pos, expected_token)
            if target is None:
                return text, None
            tokens = list(tokens)
            tokens[target] = new_text
            lines = text.split("\n")
            out = [mod.rebuild_line(line, [tokens[i] for i in idxs])
                   for line, idxs in zip(lines, line_idxs)]
            return "\n".join(out), target

        return subst, "adjudicate_transcript.tokenize_lines + rebuild_line"

    def subst_local(text, pos, new_text, expected_token=None):
        toks = list(re.finditer(r"\S+", text))
        target = find_target([m.group() for m in toks], pos, expected_token)
        if target is None:
            return text, None
        m = toks[target]
        return text[:m.start()] + new_text + text[m.end():], target

    print("note: scripts/adjudicate_transcript.py unavailable; using the local "
          "equivalent substitution")
    return subst_local, "local whitespace-preserving substitution"


def insert_before_token(text, pos, new_text):
    """Insert `new_text` before the whitespace-token at index `pos` (used for
    the adjudicator's '-ins' agreed-insertion items, whose default is empty).
    pos past the end appends after the last token. Returns (text, pos)."""
    toks = list(re.finditer(r"\S+", text))
    if not toks:
        return new_text + text, 0
    if pos >= len(toks):
        last = toks[-1]
        return text[:last.end()] + " " + new_text + text[last.end():], len(toks)
    m = toks[max(0, pos)]
    return text[:m.start()] + new_text + " " + text[m.start():], max(0, pos)


def insertion_already_present(text, pos, decision):
    """True when the decision token(s) already sit at, or immediately before,
    the anchor token index `pos` -- i.e. a previous run already inserted them.
    (After an insertion the inserted tokens occupy [pos, pos+n) and the old
    anchor token has shifted right, so a re-run with the unchanged queue pos
    finds them at `pos`.)"""
    dec_toks = decision.split()
    if not dec_toks:
        return False
    toks = [m.group() for m in re.finditer(r"\S+", text)]
    n = len(dec_toks)

    def match_at(start):
        if start < 0 or start + n > len(toks):
            return False
        return all(
            toks[start + k] == dec_toks[k]
            or (norm_token(dec_toks[k]) != ""
                and norm_token(toks[start + k]) == norm_token(dec_toks[k]))
            for k in range(n)
        )

    if match_at(pos) or match_at(pos - n):
        return True
    # pos past the end appends after the last token: check the tail too.
    if pos >= len(toks) and match_at(len(toks) - n):
        return True
    return False


def canon_item(raw, idx):
    """The queue fields this script needs (same tolerant mapping as
    make_review_html.py)."""
    page = int(raw["page"])
    pos = int(raw.get("pos") or 0)
    iid = str(raw.get("id") or raw.get("item_id") or f"p{page:04d}_pos{pos:04d}_{idx}")
    default = str(raw.get("default") or raw.get("default_reading") or "")
    return {"id": iid, "page": page, "pos": pos, "default": default,
            "insertion": default == "",
            "decided": bool(raw.get("decided")),
            "prior_decision": str(raw.get("decision") or "")}


def main():
    ap = argparse.ArgumentParser(
        description="Apply exported review decisions to BASE/final page transcripts")
    ap.add_argument("--base", required=True,
                    help="base dir, e.g. data/transcripts/bentley_deposition")
    ap.add_argument("--decisions", required=True,
                    help="decisions CSV exported from the review HTML")
    args = ap.parse_args()

    queue_path = os.path.join(args.base, "review_queue.json")
    if not os.path.exists(queue_path):
        raise SystemExit(f"{queue_path} not found")
    with open(queue_path, encoding="utf-8") as f:
        raw_queue = json.load(f)
    raw_items = raw_queue.get("items", []) if isinstance(raw_queue, dict) else raw_queue
    by_id = {}
    for idx, raw in enumerate(raw_items):
        it = canon_item(raw, idx)
        by_id[it["id"]] = it

    with open(args.decisions, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"item_id", "page", "decision"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{args.decisions}: missing CSV columns: {sorted(missing)}")
        decisions = {}  # item_id -> row (last occurrence wins)
        for row in reader:
            iid = (row.get("item_id") or "").strip()
            if iid:
                decisions[iid] = row

    subst, subst_desc = make_substitution_fn()
    print(f"substitution: {subst_desc}")

    applied, skipped, already = 0, [], []
    by_page = {}
    for iid, row in decisions.items():
        it = by_id.get(iid)
        if it is None:
            skipped.append((iid, "?", "item_id not in review_queue.json"))
            continue
        decision = (row.get("decision") or "").strip()
        if not decision:
            skipped.append((iid, it["page"], "empty decision"))
            continue
        if it["decided"] and it["prior_decision"] == decision:
            already.append((iid, it["page"],
                            "already applied (queue records this same decision)"))
            continue
        try:
            csv_page = int(row.get("page") or it["page"])
        except ValueError:
            csv_page = it["page"]
        if csv_page != it["page"]:
            print(f"warning: {iid}: CSV page {csv_page} != queue page {it['page']}; using queue")
        by_page.setdefault(it["page"], []).append((it, decision, (row.get("note") or "").strip()))

    decided_ids = {}
    for page in sorted(by_page):
        path = os.path.join(args.base, "final", f"{page:04d}.txt")
        if not os.path.exists(path):
            for it, decision, _ in by_page[page]:
                skipped.append((it["id"], page, f"{path} missing"))
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        original = text
        # Apply highest token positions first so a multi-token decision cannot
        # shift the positions of substitutions still to come on this page.
        for it, decision, note in sorted(by_page[page], key=lambda t: -t[0]["pos"]):
            if it["insertion"]:
                if insertion_already_present(text, it["pos"], decision):
                    already.append((it["id"], page,
                                    "insertion already present at anchor"))
                    continue
                text, matched = insert_before_token(text, it["pos"], decision)
            else:
                text, matched = subst(text, it["pos"], decision,
                                      expected_token=it["default"])
            if matched is None:
                skipped.append((it["id"], page,
                                f"token '{it['default']}' not found near pos {it['pos']}"))
                continue
            applied += 1
            decided_ids[it["id"]] = (decision, note)
        if text != original:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
            print(f"patched {path}")

    if decided_ids:
        for idx, raw in enumerate(raw_items):
            iid = canon_item(raw, idx)["id"]
            if iid in decided_ids:
                decision, note = decided_ids[iid]
                raw["decided"] = True
                raw["decision"] = decision
                if note:
                    raw["decision_note"] = note
        tmp = queue_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw_queue, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, queue_path)
        print(f"rewrote {queue_path} ({len(decided_ids)} items marked decided)")

    print(f"applied {applied}, already applied {len(already)}, skipped {len(skipped)}")
    for iid, page, reason in already:
        print(f"  already applied {iid} (page {page}): {reason}")
    for iid, page, reason in skipped:
        print(f"  skipped {iid} (page {page}): {reason}")
    print("No database was touched. Push the reviewed transcript with: "
          f"python scripts/load_transcript.py --doc-id 522 --base {args.base} "
          "--pages 2-120 --status reviewed")
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
