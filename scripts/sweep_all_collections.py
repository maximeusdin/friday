"""Run read-only OCR-variant sweeps across the major scanned collections, sequentially.

Produces data/sweeps/<slug>_ocr_variants.csv per collection plus a summary table,
with a name-shaped slice count (candidates that are gazetteer tokens but not
english dictionary words).
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

PY = sys.executable
TARGETS = [
    ("solo", "solo_v1_memo"),
    ("rosenberg", "rosenberg_v1"),
    ("siss_scope_soviet", "siss_scope_v1_turns"),
    ("judith_coplon", "judith_coplon_v1"),
    ("hiss_chambers", "hiss_chambers_v1"),
    ("thomas_black", "thomas_black_v1"),
    ("elizabeth_bentley", "elizabeth_bentley_v1"),
    ("mccarthy", "mccarthy_v2_turns"),
    ("morris_childs", "morris_childs_v1"),
    ("jack_childs", "jack_childs_v1"),
    ("david_greenglass", "david_greenglass_v1"),
    ("huac_reports", "huac_reports_v1_pages"),
]
GAZETTEER = "data/transcripts/bentley_deposition/gazetteer.txt"
OUT_DIR = Path("data/sweeps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

gaz = set()
for line in open(GAZETTEER, encoding="utf-8"):
    for tok in re.findall(r"[a-z0-9']+", line.lower()):
        if len(tok) >= 4:
            gaz.add(tok)
english = set(w.strip().lower() for w in open("/usr/share/dict/words", encoding="utf-8"))

summary = []
for slug, pv in TARGETS:
    out = OUT_DIR / f"{slug}_ocr_variants.csv"
    print(f"=== {slug} ({pv}) ===", flush=True)
    r = subprocess.run(
        [PY, "scripts/sweep_ocr_variants.py", "--collection-slug", slug,
         "--chunk-pv", pv, "--from-chunks", "--gazetteer", GAZETTEER,
         "--out", str(out)],
        capture_output=True, text=True,
    )
    sys.stderr.write(r.stderr[-500:] + "\n")
    if r.returncode != 0:
        summary.append((slug, "FAILED", 0, 0, 0, 0))
        continue
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    occ = sum(int(x["rare_freq"]) for x in rows)
    name_rows = [x for x in rows if x["candidate_name"] in gaz and x["candidate_name"] not in english]
    name_occ = sum(int(x["rare_freq"]) for x in name_rows)
    with open(OUT_DIR / f"{slug}_name_variants.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rare_token", "candidate_name", "cost", "rare_freq", "candidate_freq"])
        w.writeheader()
        w.writerows(sorted(name_rows, key=lambda x: (float(x["cost"]), -int(x["candidate_freq"]))))
    summary.append((slug, "ok", len(rows), occ, len(name_rows), name_occ))
    print(f"  {len(rows)} pairs / {occ} occurrences; name-shaped {len(name_rows)} / {name_occ}", flush=True)

print("\n=== SUMMARY (slug, status, pairs, occurrences, name_pairs, name_occurrences) ===")
for row in summary:
    print("  " + ", ".join(str(x) for x in row))
with open(OUT_DIR / "sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["slug", "status", "pairs", "occurrences", "name_pairs", "name_occurrences"])
    w.writerows(summary)
