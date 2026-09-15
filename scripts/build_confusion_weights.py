#!/usr/bin/env python3
"""
Learn OCR-confusion costs from adjudicated transcript pairs and write the
config/ocr_confusion.json artifact (schema ocr_confusion_v1) consumed by
retrieval/ocr_variants.py.

Input CSVs (default: the Bentley deposition pilot) carry rows
  page,embedded_token,accepted_token
where embedded_token is what the OCR read and accepted_token is what the
adjudicated transcript accepted. Tokens are lowercased and edge punctuation is
stripped; pairs with fewer than 2 alphanumeric chars on a side, or equal after
normalization, are skipped. Each surviving pair is char-aligned with a
unit-cost Levenshtein backtrace and we count:
  - single-char substitutions ("e|o": typed o read as e), alnum chars only;
  - adjacent 2-gram merges/splits ("rn|m": typed m read as rn) — a
    substitution plus an adjacent insertion/deletion collapses into one pair
    event instead of polluting the single-char counts;
  - bare insertions/deletions (reported only; indel_cost stays fixed).

Cost merge with the builtin priors (retrieval.ocr_variants.BUILTIN_PRIORS):
  - pairs observed >= MIN_OBSERVED times interpolate from their base cost
    (the prior, or NEW_PAIR_BASE = 0.5 for pairs without one, so new learned
    pairs enter at <= 0.5) toward TARGET_COST = 0.15 by frequency share
    (count / max observed count in the pool) — the dominant confusion lands
    on the floor, rarer ones move proportionally;
  - unobserved priors keep their base cost;
  - new pairs below MIN_OBSERVED are dropped as noise.
Confusion classes for skeletons are re-derived from the final single-char
costs (cost <= 0.35, class size cap 4, cheapest merges first — see
retrieval.ocr_variants.derive_confusion_classes).

Deterministic: identical input bytes produce a byte-identical artifact (the
"generated" field is a content digest, not a timestamp).

Hard acceptance (exits nonzero on failure): the written artifact, reloaded via
retrieval.ocr_variants.load_confusion, must satisfy
skeleton("fuhr") == skeleton("fuer") — the motivating FUHR->FUER case.

Usage:
  python scripts/build_confusion_weights.py
  python scripts/build_confusion_weights.py --csv a.csv --csv b.csv --out config/ocr_confusion.json

No database access; pure file in / file out.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Ensure repo root importable when running as script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval import ocr_variants
from retrieval.ocr_variants import BUILTIN_PRIORS, derive_confusion_classes, load_confusion, skeleton

DEFAULT_CSV = REPO_ROOT / "data" / "transcripts" / "bentley_deposition" / "confusion_pairs.csv"
DEFAULT_OUT = REPO_ROOT / "config" / "ocr_confusion.json"

ARTIFACT_VERSION = "ocr_confusion_v1"

MIN_OBSERVED = 5    # pairs seen fewer times keep their prior cost / are dropped
TARGET_COST = 0.15  # observed costs interpolate toward this floor
NEW_PAIR_BASE = 0.5  # learned pairs without a prior enter at this base (so <= 0.5)

TOP_LEARNED_PRINTED = 15


# ---------------------------------------------------------------- normalization


def normalize_token(tok: str) -> str:
    """Lowercase and strip edge punctuation (leading/trailing non-alnum runs)."""
    t = (tok or "").strip().lower()
    start, end = 0, len(t)
    while start < end and not t[start].isalnum():
        start += 1
    while end > start and not t[end - 1].isalnum():
        end -= 1
    return t[start:end]


def sub_key(x: str, y: str) -> str:
    """Canonical sorted single-char pair key, e.g. sub_key('o', 'e') == 'e|o'."""
    a, b = sorted((x, y))
    return f"{a}|{b}"


# ---------------------------------------------------------------- alignment


def levenshtein_ops(a: str, b: str) -> List[Tuple[str, str, str]]:
    """
    Unit-cost Levenshtein alignment of a (embedded/OCR) to b (accepted) with a
    deterministic backtrace (diagonal match/sub preferred, then deletion, then
    insertion). Returns ops in forward order:
      ("match", ca, cb) | ("sub", ca, cb) | ("del", ca, "") | ("ins", "", cb)
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            best = prev[j - 1] + (0 if ai == b[j - 1] else 1)
            d = prev[j] + 1
            if d < best:
                best = d
            d = row[j - 1] + 1
            if d < best:
                best = d
            row[j] = best

    ops: List[Tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            ops.append(("match" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", b[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def count_pair_confusions(a: str, b: str) -> Tuple[Counter, Counter, int, int]:
    """
    Confusion events for one aligned (embedded, accepted) pair:
      (single-char substitutions Counter keyed 'x|y' sorted,
       2-gram merge/split Counter keyed '<2ch>|<1ch>' e.g. 'rn|m',
       n_insertions, n_deletions).

    A substitution adjacent to an insertion/deletion is collapsed into one
    2-gram event (typed m read as rn, or typed rn read as m) and its
    components are NOT double-counted. Only alphanumeric chars are counted;
    punctuation-vs-punctuation noise (e.g. curly vs straight apostrophe) is
    not a learnable OCR confusion for skeleton/distance purposes.
    """
    subs: Counter = Counter()
    merges: Counter = Counter()
    n_ins = n_del = 0
    ops = levenshtein_ops(a, b)
    k = 0
    while k < len(ops):
        op = ops[k]
        nxt = ops[k + 1] if k + 1 < len(ops) else None
        merged = None
        if nxt is not None:
            if op[0] == "sub" and nxt[0] == "del":
                merged = (op[1] + nxt[1], op[2])  # embedded 2-gram read for accepted char
            elif op[0] == "del" and nxt[0] == "sub":
                merged = (op[1] + nxt[1], nxt[2])
            elif op[0] == "sub" and nxt[0] == "ins":
                merged = (op[2] + nxt[2], op[1])  # accepted 2-gram read as embedded char
            elif op[0] == "ins" and nxt[0] == "sub":
                merged = (op[2] + nxt[2], nxt[1])
        if merged is not None and merged[0].isalnum() and merged[1].isalnum():
            merges[f"{merged[0]}|{merged[1]}"] += 1
            k += 2
            continue
        if op[0] == "sub":
            if op[1].isalnum() and op[2].isalnum():
                subs[sub_key(op[1], op[2])] += 1
        elif op[0] == "del":
            n_del += 1
        elif op[0] == "ins":
            n_ins += 1
        k += 1
    return subs, merges, n_ins, n_del


# ---------------------------------------------------------------- input


def load_pairs(csv_paths: Iterable[Path]) -> List[Tuple[str, str]]:
    """
    Read (embedded_token, accepted_token) rows, normalized. Skips rows with
    fewer than 2 alnum chars on a side or equal after normalization.
    """
    pairs: List[Tuple[str, str]] = []
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                emb = normalize_token(row.get("embedded_token") or "")
                acc = normalize_token(row.get("accepted_token") or "")
                if sum(1 for ch in emb if ch.isalnum()) < 2:
                    continue
                if sum(1 for ch in acc if ch.isalnum()) < 2:
                    continue
                if emb == acc:
                    continue
                pairs.append((emb, acc))
    return pairs


# ---------------------------------------------------------------- cost merge


def interpolate_costs(
    counts: Counter,
    priors: Dict[str, float],
    *,
    min_observed: int = MIN_OBSERVED,
    target: float = TARGET_COST,
    new_pair_base: float = NEW_PAIR_BASE,
) -> Dict[str, float]:
    """
    Merge observed confusion counts with prior base costs.

    Pairs observed >= min_observed times interpolate from their base cost
    (prior, or new_pair_base when the pair has no prior) toward `target` by
    frequency share = count / max observed count in this pool. Unobserved
    priors keep their base cost; new pairs below min_observed are dropped.
    """
    out = {k: float(v) for k, v in priors.items()}
    observed = {k: n for k, n in counts.items() if n >= min_observed}
    if not observed:
        return out
    max_n = max(observed.values())
    for key in sorted(observed):
        n = observed[key]
        base = float(priors.get(key, new_pair_base))
        share = n / max_n
        out[key] = round(base - (base - target) * share, 4)
    return out


# ---------------------------------------------------------------- artifact


def _source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _input_digest(csv_paths: Iterable[Path]) -> str:
    """Content digest of the inputs — deterministic stand-in for a timestamp."""
    h = hashlib.sha256()
    for path in sorted(csv_paths, key=_source_label):
        h.update(_source_label(path).encode("utf-8"))
        h.update(b"\x00")
        h.update(Path(path).read_bytes())
        h.update(b"\x00")
    return f"inputs-sha256:{h.hexdigest()[:16]}"


def build_artifact(csv_paths: List[Path]) -> Dict[str, object]:
    """Count confusions across the CSVs and assemble the ocr_confusion_v1 dict."""
    pairs = load_pairs(csv_paths)
    sub_counts: Counter = Counter()
    merge_counts: Counter = Counter()
    n_ins = n_del = 0
    for emb, acc in pairs:
        subs, merges, ins_i, del_i = count_pair_confusions(emb, acc)
        sub_counts += subs
        merge_counts += merges
        n_ins += ins_i
        n_del += del_i

    sub_costs = interpolate_costs(sub_counts, BUILTIN_PRIORS["sub_costs"])
    pair_costs = interpolate_costs(merge_counts, BUILTIN_PRIORS["pair_costs"])

    artifact = {
        "version": ARTIFACT_VERSION,
        "generated": _input_digest(csv_paths),
        "sources": sorted(_source_label(p) for p in csv_paths),
        "classes": derive_confusion_classes(sub_costs),
        "sub_costs": sub_costs,
        "pair_costs": pair_costs,
        "default_sub_cost": float(BUILTIN_PRIORS["default_sub_cost"]),
        "indel_cost": float(BUILTIN_PRIORS["indel_cost"]),
    }

    print(f"Pairs used: {len(pairs)}")
    print(
        f"Events: {sum(sub_counts.values())} substitutions "
        f"({len(sub_counts)} distinct), {sum(merge_counts.values())} 2-gram merges "
        f"({len(merge_counts)} distinct), {n_ins} insertions, {n_del} deletions"
    )
    learned = sorted(
        ((n, k) for k, n in sub_counts.items() if n >= MIN_OBSERVED),
        key=lambda t: (-t[0], t[1]),
    )
    print(f"Top learned substitutions (count >= {MIN_OBSERVED}):")
    for n, key in learned[:TOP_LEARNED_PRINTED]:
        base = BUILTIN_PRIORS["sub_costs"].get(key)
        base_txt = f"prior {base}" if base is not None else f"new {NEW_PAIR_BASE}"
        print(f"  {key:>7}  x{n:<5} base {base_txt:>10} -> cost {sub_costs[key]}")
    print(f"Classes: {artifact['classes']}")

    return artifact


def write_artifact(artifact: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_acceptance(out_path: Path) -> bool:
    """
    Reload the written artifact through retrieval.ocr_variants.load_confusion
    and require skeleton("fuhr") == skeleton("fuer") (the motivating case).
    """
    # load_confusion caches per path; drop any stale entry so we read the
    # artifact we just wrote (matters when called twice in one process).
    ocr_variants._confusion_cache.pop(str(Path(out_path)), None)
    conf = load_confusion(str(out_path))
    sk_fuhr = skeleton("fuhr", conf)
    sk_fuer = skeleton("fuer", conf)
    if not sk_fuhr or sk_fuhr != sk_fuer:
        print(
            f"ACCEPTANCE FAILED: skeleton('fuhr')={sk_fuhr!r} != skeleton('fuer')={sk_fuer!r} "
            f"under {out_path}",
            file=sys.stderr,
        )
        return False
    print(f"Acceptance OK: skeleton('fuhr') == skeleton('fuer') == {sk_fuhr!r}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Learn OCR confusion costs from adjudicated transcript pairs (ocr_confusion_v1)"
    )
    ap.add_argument(
        "--csv",
        action="append",
        type=Path,
        default=None,
        help=f"Input confusion-pairs CSV (repeatable; default {DEFAULT_CSV})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output artifact path (default {DEFAULT_OUT})",
    )
    args = ap.parse_args()

    csv_paths = args.csv if args.csv else [DEFAULT_CSV]
    for path in csv_paths:
        if not Path(path).is_file():
            ap.error(f"CSV not found: {path}")

    artifact = build_artifact(list(csv_paths))
    write_artifact(artifact, args.out)
    print(f"Wrote {args.out}")

    if not check_acceptance(args.out):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
