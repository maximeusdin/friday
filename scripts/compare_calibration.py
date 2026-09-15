"""Adjudicate each calibration config (model + textract + embedded) and compare
against the 5-reading baseline: token divergence, name divergence, measured
tokens/page, and projected cost for the remaining Silvermaster corpus."""
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

BASE = Path("data/transcripts/bentley_deposition")
CONFIGS = {
    "55none": ("data/transcripts/calib_55none", "gpt-5.5", "reasoning none"),
    "55low": ("data/transcripts/calib_55low", "gpt-5.5", "reasoning low"),
    "54mini": ("data/transcripts/calib_54mini", "gpt-5.4-mini", "default"),
}
PAGES = range(2, 121)
SILVERMASTER_REMAINING = 24_176

# ($/M input, $/M output) assumption bands: (low, high)
PRICES = {
    "gpt-5.5": ((1.25, 10.0), (2.5, 15.0)),
    "gpt-5.4-mini": ((0.25, 2.0), (0.6, 4.8)),
}
TEXTRACT_PER_PAGE = 0.0015


def key(t):
    return re.sub(r"^\W+|\W+$", "", t.lower())


print(f"{'config':10s} {'div%':>6} {'name div%':>9} {'review':>7} {'tok in/out per page':>20} {'silvermaster est':>22}")
for name, (path, model, note) in CONFIGS.items():
    cfg = Path(path)
    r = subprocess.run(
        [sys.executable, "scripts/adjudicate_transcript.py", "--base", str(cfg), "--pages", "2-120"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"{name}: adjudication failed: {r.stderr[-200:]}")
        continue

    tot = diff = caps_tot = caps_diff = 0
    for page in PAGES:
        a = (BASE / "final" / f"{page:04d}.txt").read_text(encoding="utf-8").split()
        bp = cfg / "final" / f"{page:04d}.txt"
        b = bp.read_text(encoding="utf-8").split() if bp.exists() else []
        ka, kb = [key(t) for t in a], [key(t) for t in b]
        tot += len(ka)
        caps_idx = {i for i, t in enumerate(a) if t.isupper() and len(key(t)) >= 3}
        caps_tot += len(caps_idx)
        matched = set()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes():
            if op == "equal":
                matched.update(range(i1, i2))
        diff += len(ka) - len([i for i in range(len(ka)) if i in matched])
        caps_diff += sum(1 for i in caps_idx if i not in matched)

    rep = json.load(open(cfg / "adjudication_report.json"))
    review = rep.get("totals", rep).get("review_items")

    tin = tout = n = 0
    for page in PAGES:
        pj = cfg / "pages" / f"{page:04d}.{model}.json"
        if pj.exists():
            u = json.load(open(pj)).get("usage", {})
            tin += u.get("prompt_tokens", 0)
            tout += u.get("completion_tokens", 0)
            n += 1
    mean_in, mean_out = tin / max(n, 1), tout / max(n, 1)
    (lo_in, lo_out), (hi_in, hi_out) = PRICES[model]
    lo = (mean_in * lo_in + mean_out * lo_out) / 1e6 + TEXTRACT_PER_PAGE
    hi = (mean_in * hi_in + mean_out * hi_out) / 1e6 + TEXTRACT_PER_PAGE
    est = f"${lo * SILVERMASTER_REMAINING:,.0f}-{hi * SILVERMASTER_REMAINING:,.0f}"
    print(f"{name:10s} {100 * diff / tot:6.2f} {100 * caps_diff / caps_tot:9.2f} {review:>7} "
          f"{mean_in:8.0f}/{mean_out:<8.0f} {est:>22}  ({note})")
