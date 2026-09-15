"""Referee diverging name tokens + simulate tiered escalation for calibration configs.

Referee: at name positions where a calibration final diverges from baseline,
check which side's token appears in the page's Textract text (independent).

Tiered: rank pages by the calib config's own internal disagreement rate
(review_items + silent_auto per position, from its adjudication report);
escalating the top X% of pages to the full model (proxied by baseline final),
what name divergence remains?
"""
import difflib
import json
import re
from pathlib import Path

BASE = Path("data/transcripts/bentley_deposition")
CONFIGS = {"55low": "data/transcripts/calib_55low", "54mini": "data/transcripts/calib_54mini"}
PAGES = list(range(2, 121))


def key(t):
    return re.sub(r"^\W+|\W+$", "", t.lower())


for name, path in CONFIGS.items():
    cfg = Path(path)
    ref_wins = cal_wins = both = neither = 0
    page_stats = []  # (page, disagreement_rate, caps_tot, caps_div)
    for page in PAGES:
        a = (BASE / "final" / f"{page:04d}.txt").read_text(encoding="utf-8").split()
        fp = cfg / "final" / f"{page:04d}.txt"
        if not fp.exists():
            continue
        b = fp.read_text(encoding="utf-8").split()
        tx = set(key(t) for t in (cfg / "textract" / f"{page:04d}.txt").read_text(encoding="utf-8").split())
        ka, kb = [key(t) for t in a], [key(t) for t in b]
        caps_idx = {i for i, t in enumerate(a) if t.isupper() and len(key(t)) >= 3}
        caps_div = 0
        sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            for i in range(i1, i2):
                if i not in caps_idx:
                    continue
                caps_div += 1
                bt = ka[i]
                ct = kb[j1] if j1 < len(kb) else ""
                bin_, cin = bt in tx, ct in tx
                if bin_ and not cin:
                    ref_wins += 1
                elif cin and not bin_:
                    cal_wins += 1
                elif bin_ and cin:
                    both += 1
                else:
                    neither += 1
        rep = json.load(open(cfg / "adjudication_report.json"))
        pstats = rep.get("pages", {}).get(str(page), {})
        pos = pstats.get("positions", 1) or 1
        dis = (pstats.get("review_items", 0) + pstats.get("silent_auto", 0)) / pos
        page_stats.append((page, dis, len(caps_idx), caps_div))

    tot_caps = sum(c for _, _, c, _ in page_stats)
    tot_div = sum(d for _, _, _, d in page_stats)
    print(f"\n=== {name} ===")
    print(f"referee on diverging name tokens: baseline-in-textract-only {ref_wins}, "
          f"calib-in-textract-only {cal_wins}, both {both}, neither {neither}")
    print(f"overall name divergence: {tot_div}/{tot_caps} = {100*tot_div/tot_caps:.2f}%")
    ranked = sorted(page_stats, key=lambda x: -x[1])
    for frac in (0.2, 0.3, 0.4):
        n_esc = int(len(ranked) * frac)
        esc_pages = {p for p, _, _, _ in ranked[:n_esc]}
        kept_caps = sum(c for p, _, c, _ in page_stats if p not in esc_pages)
        kept_div = sum(d for p, _, _, d in page_stats if p not in esc_pages)
        resid = 100 * kept_div / max(kept_caps, 1) * kept_caps / max(tot_caps, 1)
        print(f"  escalate top {int(frac*100)}% pages -> residual name divergence "
              f"{resid:.2f}% of all name tokens (kept-page rate {100*kept_div/max(kept_caps,1):.2f}%)")
