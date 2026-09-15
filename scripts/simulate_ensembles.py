"""Simulate cheaper transcription ensembles offline.

For each config (subset of the five readings), re-run the adjudicator on the
stored Bentley readings and measure divergence of its final text from the
full-ensemble baseline: total token divergence, ALL-CAPS (name-ish) token
divergence, and review-queue size.
"""
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path("data/transcripts/bentley_deposition")
WORK = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ensemble_sim")
PAGES = range(2, 121)

CONFIGS = {
    "B_drop41": ["gpt-5.5", "gpt-5.2"],          # + textract + embedded
    "C_drop55": ["gpt-5.2", "gpt-4.1"],          # + textract + embedded
    "D_single52": ["gpt-5.2"],                   # + textract + embedded
    "E_single55": ["gpt-5.5"],                   # + textract + embedded
}


def key(t):
    return re.sub(r"^\W+|\W+$", "", t.lower())


def load_final(base, page):
    p = base / "final" / f"{page:04d}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


results = {}
for name, models in CONFIGS.items():
    cfg = WORK / name
    if cfg.exists():
        shutil.rmtree(cfg)
    (cfg / "pages").mkdir(parents=True)
    for sub in ("textract", "embedded"):
        os.symlink((BASE / sub).resolve(), cfg / sub)
    os.symlink((BASE / "gazetteer.txt").resolve(), cfg / "gazetteer.txt")
    for page in PAGES:
        for m in models:
            src = BASE / "pages" / f"{page:04d}.{m}.json"
            if src.exists():
                os.symlink(src.resolve(), cfg / "pages" / f"{page:04d}.{m}.json")

    r = subprocess.run(
        [sys.executable, "scripts/adjudicate_transcript.py", "--base", str(cfg), "--pages", "2-120"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        results[name] = {"error": r.stderr[-300:]}
        continue

    tot = diff = caps_tot = caps_diff = 0
    for page in PAGES:
        a = load_final(BASE, page).split()
        b = load_final(cfg, page).split()
        ka = [key(t) for t in a]
        kb = [key(t) for t in b]
        tot += len(ka)
        caps_idx = {i for i, t in enumerate(a) if t.isupper() and len(key(t)) >= 3}
        caps_tot += len(caps_idx)
        sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
        matched = set()
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                matched.update(range(i1, i2))
        diff += tot and (len(ka) - sum(1 for i in range(len(ka)) if i in matched))
        caps_diff += sum(1 for i in caps_idx if i not in matched)

    report = json.load(open(cfg / "adjudication_report.json"))
    t = report.get("totals", report)
    results[name] = {
        "models": models + ["textract", "embedded"],
        "tokens": tot,
        "diverging_tokens": diff,
        "divergence_pct": round(100 * diff / tot, 2) if tot else None,
        "caps_tokens": caps_tot,
        "caps_diverging": caps_diff,
        "caps_divergence_pct": round(100 * caps_diff / caps_tot, 2) if caps_tot else None,
        "review_items": t.get("review_items"),
    }

print(json.dumps(results, indent=1))
