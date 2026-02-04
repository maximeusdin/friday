#!/usr/bin/env python3
"""
Analyze a reviewed OCR calibration CSV and recommend threshold adjustments.

Input: the CSV produced by scripts/export_ocr_calibration_sample.py, after a human fills:
- review_judgement: CORRECT / INCORRECT / JUNK / MISS / UNSURE
- (optional) review_correct_entity_id
- review_notes

This script prints:
- Precision for resolved_near_threshold band
- "Queue yield" for queued_mid band (how many are not junk/unsure)
- False-negative signal from ignored_borderline band (MISS rate)
- Simple, conservative recommendations for which thresholds to adjust

It does NOT modify code automatically.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ALLOWED = {"CORRECT", "INCORRECT", "JUNK", "MISS", "UNSURE", ""}


def _iter_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        # Skip comment header lines
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return []
            if not line.startswith("#"):
                f.seek(pos)
                break
        r = csv.DictReader(f)
        return list(r)


def _norm_judgement(v: str) -> str:
    v = (v or "").strip().upper()
    if v in ALLOWED:
        return v
    return v  # keep as-is; will be counted as "invalid"


@dataclass
class BandStats:
    n_total: int = 0
    counts: Counter = None

    def __post_init__(self):
        if self.counts is None:
            self.counts = Counter()

    def add(self, judgement: str):
        self.n_total += 1
        self.counts[judgement] += 1

    def n_labeled(self) -> int:
        return self.n_total - self.counts.get("", 0)


def _pct(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{100.0 * num / den:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze reviewed OCR calibration sample")
    ap.add_argument("--in", dest="in_path", required=True, help="Reviewed CSV path")
    ap.add_argument("--target-auto-precision", type=float, default=0.95,
                    help="Target precision for auto-linked band (default 0.95)")
    ap.add_argument("--target-queue-yield", type=float, default=0.50,
                    help="Target useful yield for queued band (default 0.50)")
    args = ap.parse_args()

    path = Path(args.in_path)
    rows = _iter_rows(path)
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    bands: Dict[str, BandStats] = defaultdict(BandStats)
    invalid = Counter()

    for row in rows:
        band = (row.get("sample_group") or "").strip()
        j = _norm_judgement(row.get("review_judgement") or "")
        if j not in ALLOWED:
            invalid[j] += 1
        bands[band].add(j)

    # Key bands
    b_res = bands.get("resolved_near_threshold", BandStats())
    b_q = bands.get("queued_mid", BandStats())
    b_ign = bands.get("ignored_borderline", BandStats())

    # Metrics
    res_labeled = b_res.n_labeled()
    res_correct = b_res.counts.get("CORRECT", 0)
    res_incorrect = b_res.counts.get("INCORRECT", 0)
    res_precision = (res_correct / res_labeled) if res_labeled else None

    q_labeled = b_q.n_labeled()
    q_useful = q_labeled - b_q.counts.get("JUNK", 0) - b_q.counts.get("UNSURE", 0)
    q_yield = (q_useful / q_labeled) if q_labeled else None

    ign_labeled = b_ign.n_labeled()
    ign_miss = b_ign.counts.get("MISS", 0)
    ign_miss_rate = (ign_miss / ign_labeled) if ign_labeled else None

    print("============================================================")
    print("OCR CALIBRATION ANALYSIS")
    print("============================================================")
    print(f"Input: {path}")
    print()

    def dump_band(name: str, b: BandStats):
        print(f"{name}:")
        print(f"  total rows:   {b.n_total}")
        print(f"  labeled:      {b.n_labeled()} (blank={b.counts.get('', 0)})")
        for k in ["CORRECT", "INCORRECT", "JUNK", "MISS", "UNSURE"]:
            if b.counts.get(k, 0):
                print(f"  {k:<9}: {b.counts[k]}")
        if invalid:
            pass
        print()

    dump_band("resolved_near_threshold", b_res)
    dump_band("queued_mid", b_q)
    dump_band("ignored_borderline", b_ign)

    if invalid:
        print("Invalid judgements (typos):")
        for k, v in invalid.most_common():
            print(f"  {k}: {v}")
        print()

    if res_precision is not None:
        print(f"Auto-link precision (resolved band): {res_correct}/{res_labeled} = {_pct(res_correct, res_labeled)}")
    else:
        print("Auto-link precision (resolved band): n/a (no labeled rows)")

    if q_yield is not None:
        print(f"Queue yield (queued band): {q_useful}/{q_labeled} = {_pct(q_useful, q_labeled)}")
    else:
        print("Queue yield (queued band): n/a (no labeled rows)")

    if ign_miss_rate is not None:
        print(f"Ignore false-negative signal (ignored band, MISS rate): {ign_miss}/{ign_labeled} = {_pct(ign_miss, ign_labeled)}")
    else:
        print("Ignore false-negative signal (ignored band): n/a (no labeled rows)")

    print()
    print("============================================================")
    print("RECOMMENDATIONS (conservative)")
    print("============================================================")

    # Simple heuristic recommendations (no automatic changes)
    recs: List[str] = []

    if res_precision is not None and res_labeled >= 20:
        if res_precision < args.target_auto_precision:
            recs.append(
                "- Auto-link precision below target: increase STRONG_THRESHOLD by +0.02 and/or increase MARGIN_THRESHOLD by +0.05."
            )
        elif res_incorrect == 0 and res_labeled >= 50:
            recs.append(
                "- Auto-link slice looks clean: you can consider decreasing STRONG_THRESHOLD by -0.01 to recover recall, but only if queue volume is acceptable."
            )

    if q_yield is not None and q_labeled >= 20:
        if q_yield < args.target_queue_yield:
            recs.append(
                "- Queue yield is low (too much junk/unsure): increase QUEUE_THRESHOLD by +0.03 to queue fewer borderline items."
            )
        else:
            recs.append(
                "- Queue yield is decent: keep QUEUE_THRESHOLD as-is; focus next on collision handling / grouping to reduce per-item burden."
            )

    if ign_miss_rate is not None and ign_labeled >= 20:
        if ign_miss_rate > 0.10:
            recs.append(
                "- Ignored borderline band has non-trivial MISS rate: consider decreasing QUEUE_THRESHOLD by -0.02 (or increasing TOP_K) on a small slice, and re-measure queue yield."
            )

    if not recs:
        recs.append("- Not enough labeled rows yet. Label at least ~20 per band, then re-run this analysis.")

    for r in recs:
        print(r)

    print()
    print("Next sample command (exclude already reviewed):")
    print(f"  python scripts/export_ocr_calibration_sample.py --exclude-reviewed-csv \"{path}\" --out ocr_calibration_sample_next.csv")


if __name__ == "__main__":
    main()

