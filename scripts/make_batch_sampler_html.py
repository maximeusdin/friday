"""Before/after sampler HTML for batch-adjudicated Silvermaster documents.

Three columns per sampled page: scan (rendered from the source PDF), the old
embedded OCR layer, and the adjudicated transcript. Token-level diff
highlighting: changed tokens red in the before column, green in the after
column.

Unlike make_before_after_html.py (single pre-rendered doc), this works on any
doc in a batch directory: it looks up the PDF URL in manifest.json, downloads
the PDF once into a cache directory, and renders the requested page on the fly.

Usage:
  python scripts/make_batch_sampler_html.py \
      --docs "459:98,425:57,507:40" \
      --out data/transcripts/batch_silvermaster/batch_sampler.html \
      --cache /path/to/pdf_cache
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
import tempfile
import urllib.request

import fitz  # PyMuPDF
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True,
                help='comma-separated doc_id:page pairs, e.g. "459:98,425:57" (pages 1-based)')
ap.add_argument("--out", required=True, help="output HTML path")
ap.add_argument("--base", default="data/transcripts/batch_silvermaster",
                help="batch directory containing manifest.json and per-doc subdirs")
ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "friday_batch_pdf_cache"),
                help="directory to cache downloaded PDFs")
ap.add_argument("--width", type=int, default=640, help="scan image width in px")
ap.add_argument("--quality", type=int, default=60, help="scan JPEG quality")
args = ap.parse_args()

PICKS = []
for part in args.docs.split(","):
    doc_s, page_s = part.strip().split(":")
    PICKS.append((int(doc_s), int(page_s)))

MANIFEST = {d["doc_id"]: d for d in json.load(open(os.path.join(args.base, "manifest.json")))}
os.makedirs(args.cache, exist_ok=True)


def fetch_pdf(doc_id):
    """Download the doc's PDF into the cache (once) and return the local path."""
    path = os.path.join(args.cache, f"{doc_id}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = MANIFEST[doc_id]["pdf_url"]
    print(f"downloading doc {doc_id}: {url}")
    tmp = path + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)
    print(f"  cached {os.path.getsize(path) // (1 << 20)} MB -> {path}")
    return path


def render_page_b64(doc_id, page):
    """Render 1-based `page` of the doc's PDF to a base64 JPEG at --width."""
    pdf = fitz.open(fetch_pdf(doc_id))
    try:
        pg = pdf[page - 1]
        # Render at 2x target width, then Lanczos-downscale for crisper text.
        zoom = (args.width * 2) / pg.rect.width
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    finally:
        pdf.close()
    im = im.resize((args.width, int(im.height * args.width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=args.quality)
    return base64.b64encode(buf.getvalue()).decode()


def tokens(text):
    return text.split()


def key(t):
    return re.sub(r"^\W+|\W+$", "", t.lower())


def diff_mark(before_text, after_text):
    """Return (before_html, after_html, changed, total) with changed tokens wrapped."""
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


def read_layer(doc_id, sub, page):
    p = os.path.join(args.base, str(doc_id), sub, f"{page:04d}.txt")
    return open(p, encoding="utf-8", errors="replace").read()


sections = []
for doc_id, page in PICKS:
    if doc_id not in MANIFEST:
        sys.exit(f"doc {doc_id} not in manifest.json")
    src = MANIFEST[doc_id]["source_name"]
    src = re.sub(r"_text\.pdf$", "", src)
    caption = f"{src} — page {page} (doc {doc_id})"
    before = read_layer(doc_id, "embedded", page)
    after = read_layer(doc_id, "final", page)
    bh, ah, changed, total = diff_mark(before, after)
    scan = render_page_b64(doc_id, page)
    sections.append(f"""
<section>
  <h2>{html.escape(caption)}</h2>
  <p class="stat">{changed} of {total} tokens differ between the layers.</p>
  <div class="row">
    <figure><figcaption>The scan (authoritative)</figcaption>
      <img src="data:image/jpeg;base64,{scan}" alt="doc {doc_id} page {page} scan"></figure>
    <div class="col"><h3>Before — original OCR text layer</h3><pre>{bh}</pre></div>
    <div class="col"><h3>After — adjudicated transcript</h3><pre>{ah}</pre></div>
  </div>
</section>""")

page_html = f"""<meta charset="utf-8">
<title>Silvermaster batch — OCR Before &amp; After</title>
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
<h1>Silvermaster FBI file — the searchable text, before and after</h1>
<p class="lede">Sampled pages from the 151-document machine-transcription batch. Before: the text layer embedded
in the source PDF — what Friday searched until now. After: the transcript produced by a three-reading ensemble
(olmOCR-2 + Surya + the original OCR), merged by majority vote. Highlighted tokens are where the layers differ;
the scan remains the authoritative document.</p>
{''.join(sections)}
"""

open(args.out, "w", encoding="utf-8").write(page_html)
print(f"wrote {args.out} ({len(page_html) // 1024} KB)")
