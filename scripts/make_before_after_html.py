"""Generate a before/after comparison HTML for the Bentley deposition transcript.

Three columns per page: scan (downscaled), original embedded OCR, adjudicated
transcript. Token-level diff highlighting: changed tokens red in the before
column, green in the after column.
"""
import argparse
import base64
import difflib
import html
import io
import json
import os
import re
import sys

from PIL import Image

BASE = "data/transcripts/bentley_deposition"

ap = argparse.ArgumentParser()
ap.add_argument("--pages", default="3,19,112", help="comma-separated 1-based pages")
ap.add_argument("--final-dir", default=f"{BASE}/final", help="dir of after-transcripts")
ap.add_argument("--out", default=f"{BASE}/before_after_bentley.html")
ap.add_argument("--title-note", default="five independent readings per page")
args = ap.parse_args()
PAGES = [int(p) for p in args.pages.split(",")]
OUT = args.out

esc_path = os.path.join(os.path.dirname(args.final_dir), "escalated_pages.json")
ESCALATED = set(json.load(open(esc_path))) if os.path.exists(esc_path) else set()


def tokens(text):
    return text.split()


def key(t):
    return re.sub(r"^\W+|\W+$", "", t.lower())


def diff_mark(before_text, after_text):
    """Return (before_html, after_html) with changed tokens wrapped in spans."""
    a, b = tokens(before_text), tokens(after_text)
    sm = difflib.SequenceMatcher(None, [key(t) for t in a], [key(t) for t in b], autojunk=False)
    outa, outb = [], []
    changed = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            outa.extend(html.escape(t) for t in a[i1:i2])
            outb.extend(html.escape(t) for t in b[j1:j2])
        else:
            changed += max(i2 - i1, j2 - j1)
            outa.extend(f'<span class="del">{html.escape(t)}</span>' for t in a[i1:i2])
            outb.extend(f'<span class="ins">{html.escape(t)}</span>' for t in b[j1:j2])
    return " ".join(outa), " ".join(outb), changed, len(a)


def img_b64(page):
    im = Image.open(f"{BASE}/images/{page:04d}.png").convert("L")
    w = 720
    im = im.resize((w, int(im.height * w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=68)
    return base64.b64encode(buf.getvalue()).decode()


CAPTIONS = {
    3: "Page 3 — the deposition's opening and the Meekirk / Fuhr passage from the original bug report.",
    19: "Page 19 — a badly damaged photostat: the old text layer is mostly noise.",
    108: "Page 108 — the signature page (“this and 111 other pages”).",
    112: "Page 112 — the FBI's own name index to the statement: a roster of garbled names.",
}

sections = []
for p in PAGES:
    cap = CAPTIONS.get(p, f"Page {p}.")
    if ESCALATED:
        cap += " [escalated to full-reasoning read]" if p in ESCALATED else " [economy read only]"
    CAPTIONS[p] = cap
    before = open(f"{BASE}/embedded/{p:04d}.txt", encoding="utf-8").read()
    after = open(os.path.join(args.final_dir, f"{p:04d}.txt"), encoding="utf-8").read()
    bh, ah, changed, total = diff_mark(before, after)
    sections.append(f"""
<section>
  <h2>{html.escape(CAPTIONS[p])}</h2>
  <p class="stat">{changed} of {total} tokens differ between the layers.</p>
  <div class="row">
    <figure><figcaption>The scan (authoritative)</figcaption>
      <img src="data:image/jpeg;base64,{img_b64(p)}" alt="page {p} scan"></figure>
    <div class="col"><h3>Before — original OCR text layer</h3><pre>{bh}</pre></div>
    <div class="col"><h3>After — adjudicated transcript</h3><pre>{ah}</pre></div>
  </div>
</section>""")

page_html = f"""<meta charset="utf-8">
<title>Bentley Deposition — OCR Before &amp; After</title>
<style>
 body {{ font: 15px/1.45 -apple-system, Segoe UI, sans-serif; margin: 24px auto; max-width: 1500px; color: #222; background: #fafaf7; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin: 40px 0 4px; }} h3 {{ font-size: 13px; margin: 0 0 6px; color: #555; text-transform: uppercase; letter-spacing: .04em; }}
 .lede {{ color: #444; max-width: 900px; }}
 .stat {{ color: #777; font-size: 13px; margin: 2px 0 10px; }}
 .row {{ display: flex; gap: 14px; align-items: flex-start; }}
 figure {{ margin: 0; flex: 0 0 340px; }} figure img {{ width: 100%; border: 1px solid #ccc; border-radius: 4px; }}
 figcaption {{ font-size: 12px; color: #666; margin-bottom: 4px; }}
 .col {{ flex: 1 1 0; min-width: 0; }}
 pre {{ white-space: pre-wrap; word-break: break-word; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; font: 12.5px/1.5 ui-monospace, Menlo, monospace; max-height: 560px; overflow-y: auto; }}
 .del {{ background: #fde8e8; color: #8a1f1f; border-radius: 2px; }}
 .ins {{ background: #e2f4e5; color: #1d5c2f; border-radius: 2px; }}
</style>
<h1>Bentley Deposition (FBI 65-56402, Serial 220) — the searchable text, before and after</h1>
<p class="lede">Before: the text layer embedded in the source PDF — what Friday searched until this week.
After: the transcript produced by {html.escape(args.title_note)}, merged by majority vote with AWS Textract and the
original OCR. Roughly one word in ten differs from the old text layer. Highlighted tokens are where the layers differ;
the scan remains the authoritative document.</p>
{''.join(sections)}
"""

open(OUT, "w", encoding="utf-8").write(page_html)
print(f"wrote {OUT} ({len(page_html) // 1024} KB)")
