#!/usr/bin/env python3
"""Build the single-file HTML review UI for the Bentley deposition name review.

Reads BASE/review_queue.json (list of review items produced by
scripts/adjudicate_transcript.py) and writes ONE self-contained HTML file --
zero external resources, works from file:// .

For every queue item an image crop is produced:
  1. The item context (context_before + default + context_after) is
     fuzzy-located in the page's Textract WORD blocks
     (BASE/textract/{page:04d}.json) using normalized-token SequenceMatcher.
  2. The LINE containing the best-matching WORD is found and that line plus
     one line above and below is cropped from BASE/images/{page:04d}.png
     (pad ~14px, upscale 1.6x, JPEG q80, base64-embedded). The estimated
     word region is outlined.
  3. Fallback when no geometry match: a full-width horizontal strip at
     pos/total_positions of the page height.

UI: sticky header (title, "n reviewed / total" progress, collection context,
always-visible Export Decisions CSV button); one row per item with the crop,
page number, candidate readings as large radio-style buttons labeled with
their sources, an "illegible [?]" button and a free-text override input.
Keyboard: j/k next/prev, 1-9 pick reading, e focus free text. Decisions
persist to localStorage on every change and restore on load. Keys are
namespaced per generated artifact -- NS is the first 10 hex chars of sha256
over the concatenated sorted item ids + defaults, and entries are stored
under NS + ":" + item id -- so two review HTMLs never share entries; stored
entries whose recorded default no longer matches the item default are
ignored. Export downloads a CSV with header item_id,page,decision,note where
decision is the chosen reading text, or [illegible] (item_id is the bare id,
unchanged). Items whose queue entry carried decided==true at build time are
already applied and are excluded from the export.

Usage:
  python scripts/make_review_html.py \
      --base data/transcripts/bentley_deposition \
      [--out data/transcripts/bentley_deposition/review_bentley_deposition.html] \
      [--title "Bentley Deposition — Name Review"]
"""
import argparse
import base64
import difflib
import hashlib
import html as html_mod
import io
import json
import os
import re
import sys

from PIL import Image, ImageDraw

PAD = 14            # px padding around the cropped lines
UPSCALE = 1.0       # crop upscale factor (1700px-wide sources are already legible)
JPEG_QUALITY = 55
STRIP_HALF = 80     # fallback strip: half-height in px
LOCATE_THRESHOLD = 0.6  # minimum SequenceMatcher ratio to accept a geometry match

DEFAULT_CONTEXT = (
    "Collection silvermaster — FBI File Silvermaster Part 6 (late November 1945), "
    "document 522; Bentley deposition pages 3–120."
)


def norm_token(t):
    """Lowercase and strip everything but letters/digits, for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def tokens_of(val):
    """Accept a token list or a plain string (split on whitespace)."""
    if val is None:
        return []
    if isinstance(val, str):
        return val.split()
    return [str(t) for t in val]


def canon_candidates(item, default):
    """Normalize the candidate readings to [(text, [sources]), ...].

    The adjudicator writes "readings" as a {source_name: surface} dict (e.g.
    {"gpt-5.5": "FUHR", "textract": "FUHR"}); distinct surfaces are grouped so
    each becomes one button labeled with its supporting sources. Also
    tolerated: a list of strings, a list of {text, sources} dicts, or a
    {surface: [sources]} dict. A non-empty default reading is guaranteed to be
    present (insertion items have default "" and get no default button).
    """
    raw = item.get("readings") or item.get("candidates") or item.get("options") or []
    pairs = []
    if isinstance(raw, dict):
        if all(isinstance(v, str) for v in raw.values()):
            for src, text in raw.items():  # {source: surface}
                pairs.append((str(text), [str(src)]))
        else:
            for text, srcs in raw.items():  # {surface: [sources]}
                if isinstance(srcs, str):
                    srcs = [srcs]
                pairs.append((str(text), [str(s) for s in (srcs or [])]))
    else:
        for c in raw:
            if isinstance(c, dict):
                text = c.get("text") or c.get("reading") or c.get("token") or c.get("value") or ""
                srcs = c.get("sources") or c.get("models") or c.get("source") or []
                if isinstance(srcs, str):
                    srcs = [srcs]
                pairs.append((str(text), [str(s) for s in srcs]))
            else:
                pairs.append((str(c), []))
    # dedupe by exact text, merging sources, preserving order
    out, index = [], {}
    for text, srcs in pairs:
        if not text:
            continue
        if text in index:
            merged = out[index[text]][1]
            for s in srcs:
                if s not in merged:
                    merged.append(s)
        else:
            index[text] = len(out)
            out.append((text, list(srcs)))
    if default and default not in index:
        out.insert(0, (default, ["default"]))
    return out


def canon_item(raw, idx):
    """Map one review_queue.json item to the canonical fields this UI needs.

    context_before/context_after may be token lists or plain strings (the
    adjudicator writes ~30-char strings cut from the final token stream);
    `pos` is the 0-based whitespace-token index in the page's final text.
    total_positions is optional -- when absent the caller derives it from
    BASE/final/{page:04d}.txt."""
    if "page" not in raw:
        raise SystemExit(f"review_queue.json item {idx} has no 'page' field: {raw!r}")
    page = int(raw["page"])
    pos = int(raw.get("pos") or 0)
    total = int(raw.get("total_positions") or raw.get("total") or 0)
    iid = str(raw.get("id") or raw.get("item_id") or f"p{page:04d}_pos{pos:04d}_{idx}")
    default = str(raw.get("default") or raw.get("default_reading") or "")
    before = tokens_of(raw.get("context_before"))
    after = tokens_of(raw.get("context_after"))
    return {
        "id": iid,
        "page": page,
        "pos": pos,
        "total": total,
        "default": default,
        "before": before,
        "after": after,
        "candidates": canon_candidates(raw, default),
        "decided": bool(raw.get("decided")),
        "prior": str(raw.get("decision") or ""),
    }


def bbox_of(block):
    """Normalized (left, top, width, height) from a Textract block, or None."""
    try:
        bb = block["Geometry"]["BoundingBox"]
        return (float(bb["Left"]), float(bb["Top"]), float(bb["Width"]), float(bb["Height"]))
    except (KeyError, TypeError, ValueError):
        return None


def load_textract(base, page):
    """Parse BASE/textract/{page:04d}.json into words, lines and the
    word->line mapping. Returns None if the file is missing/unreadable."""
    path = os.path.join(base, "textract", f"{page:04d}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"warning: cannot read {path}: {e}")
        return None
    words, lines, word_to_line = [], [], {}
    for b in data.get("Blocks", []):
        btype = b.get("BlockType")
        if btype == "WORD" and b.get("Text"):
            words.append({"id": b.get("Id"), "text": b["Text"], "bbox": bbox_of(b)})
        elif btype == "LINE":
            lb = {"id": b.get("Id"), "bbox": bbox_of(b)}
            if lb["bbox"]:
                lines.append(lb)
            for rel in b.get("Relationships") or []:
                if rel.get("Type") == "CHILD":
                    for cid in rel.get("Ids") or []:
                        word_to_line[cid] = b.get("Id")
    lines_sorted = sorted(lines, key=lambda l: l["bbox"][1])
    return {
        "words": words,
        "norms": [norm_token(w["text"]) for w in words],
        "word_to_line": word_to_line,
        "lines_sorted": lines_sorted,
        "line_index": {l["id"]: i for i, l in enumerate(lines_sorted)},
    }


def locate_word(tx, before, default, after):
    """Fuzzy-locate the item context in the page WORD sequence.

    Slides a window of len(context) over the normalized WORD tokens and
    scores each window with SequenceMatcher against the normalized context;
    the WORD at the default's offset within the best window wins. Returns the
    word dict, or None when nothing scores above LOCATE_THRESHOLD."""
    words = tx["words"]
    norms = tx["norms"]
    if not words:
        return None
    ctx = [norm_token(t) for t in before] + [norm_token(default)] + [norm_token(t) for t in after]
    tgt = len(before)
    ctx_join = " ".join(ctx)
    L = len(ctx)
    best_score, best_start = 0.0, None
    for s in range(len(words)):
        window = norms[s:s + L]
        score = difflib.SequenceMatcher(None, ctx_join, " ".join(window)).ratio()
        if tgt < len(window) and window[tgt] and window[tgt] == ctx[tgt]:
            score += 0.05  # tie-break: window aligns the default exactly
        if score > best_score:
            best_score, best_start = score, s
    if best_start is None or best_score < LOCATE_THRESHOLD:
        return None
    wi = best_start + tgt
    if wi >= len(words) or not words[wi].get("bbox"):
        return None
    return words[wi]


def to_data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def make_crop(img, tx, word):
    """Crop the word's LINE plus one line above/below, pad, upscale, outline
    the word region. Returns a JPEG data URI, or None on degenerate geometry."""
    W, H = img.size
    boxes = []
    line_id = tx["word_to_line"].get(word["id"])
    if line_id in tx["line_index"]:
        i = tx["line_index"][line_id]
        boxes = [l["bbox"] for l in tx["lines_sorted"][max(0, i - 1): i + 2]]
    if not boxes:
        boxes = [word["bbox"]]
    x0 = max(0, int(min(b[0] for b in boxes) * W) - PAD)
    y0 = max(0, int(min(b[1] for b in boxes) * H) - PAD)
    x1 = min(W, int(max(b[0] + b[2] for b in boxes) * W) + PAD)
    y1 = min(H, int(max(b[1] + b[3] for b in boxes) * H) + PAD)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img.crop((x0, y0, x1, y1)).convert("RGB")
    nw, nh = max(1, round(crop.width * UPSCALE)), max(1, round(crop.height * UPSCALE))
    crop = crop.resize((nw, nh), Image.LANCZOS)
    b = word["bbox"]
    rx0 = max(0, (b[0] * W - x0) * UPSCALE - 3)
    ry0 = max(0, (b[1] * H - y0) * UPSCALE - 3)
    rx1 = min(nw - 1, ((b[0] + b[2]) * W - x0) * UPSCALE + 3)
    ry1 = min(nh - 1, ((b[1] + b[3]) * H - y0) * UPSCALE + 3)
    if rx1 > rx0 and ry1 > ry0:
        ImageDraw.Draw(crop).rectangle([rx0, ry0, rx1, ry1], outline=(214, 40, 40), width=3)
    return to_data_uri(crop)


def fallback_strip(img, pos, total):
    """Full-width horizontal strip centered at pos/total of the page height."""
    W, H = img.size
    frac = pos / total if total > 0 else 0.5
    frac = min(max(frac, 0.0), 1.0)
    cy = int(frac * H)
    y0 = max(0, cy - STRIP_HALF)
    y1 = min(H, cy + STRIP_HALF)
    if y1 <= y0:
        y0, y1 = 0, min(H, 2 * STRIP_HALF)
    return to_data_uri(img.crop((0, y0, W, y1)).convert("RGB"))


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
* { box-sizing: border-box; }
body { margin:0; background:#f6f6f2; color:#1c1c1c;
       font:16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header { position:sticky; top:0; z-index:10; background:#fffdf7; border-bottom:1px solid #ddd;
         padding:10px 18px; display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
header h1 { font-size:18px; margin:0; }
#progress { font-variant-numeric:tabular-nums; color:#444; font-size:15px; }
#export { margin-left:auto; background:#1a66d0; border:0; color:#fff; padding:8px 16px;
          font-size:15px; border-radius:6px; cursor:pointer; }
#export:hover { background:#1554ae; }
#ctx { flex-basis:100%; color:#666; font-size:13px; }
main { max-width:1280px; margin:0 auto; padding:16px; }
.item { background:#fff; border:1px solid #ddd; border-left:5px solid #ddd; border-radius:8px;
        padding:14px 16px; margin-bottom:14px; }
.item.active { border-color:#1a66d0; box-shadow:0 0 0 2px rgba(26,102,208,.18); }
.item.done { border-left-color:#2e9e44; }
.meta { color:#666; font-size:13px; margin-bottom:8px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.tag { background:#eee; border-radius:4px; padding:1px 7px; }
.tag.approx { background:#fff3cd; }
.tag.applied { background:#d9f2df; }
.crop { max-width:100%; height:auto; border:1px solid #e3e3e3; border-radius:4px; display:block; background:#fff; }
.noimg { color:#999; font-style:italic; padding:24px; border:1px dashed #ccc; border-radius:4px; }
.ctxline { margin:10px 0 6px; font-size:17px; color:#333; }
.ctxline mark { background:#ffe89a; padding:0 4px; border-radius:3px; font-weight:600; }
.cands { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:6px; }
button.cand { font:inherit; background:#fafafa; border:2px solid #c9c9c9; border-radius:8px;
              padding:9px 16px; cursor:pointer; text-align:left; }
button.cand .kbd { font-size:11px; color:#999; border:1px solid #ccc; border-radius:3px;
                   padding:0 4px; margin-right:7px; vertical-align:2px; }
button.cand .rd { font-size:21px; font-weight:600; letter-spacing:.3px; }
button.cand .src { display:block; font-size:12px; color:#777; margin-top:2px; }
button.cand.default { border-style:dashed; border-color:#8fb2e0; background:#f4f8ff; }
button.cand.selected { background:#e3efff; border-color:#1a66d0; border-style:solid; }
button.cand.illeg .rd { color:#a33; }
input.override { font:inherit; font-size:16px; padding:10px 12px; border:2px solid #c9c9c9;
                 border-radius:8px; min-width:260px; background:#fff; }
input.override.selected { border-color:#1a66d0; background:#e3efff; }
footer { color:#888; text-align:center; padding:30px 0 40px; font-size:13px; }
kbd { background:#eee; border-radius:3px; padding:0 4px; border:1px solid #ccc; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span id="progress"></span>
  <button id="export" title="Download the decisions CSV">Export Decisions CSV</button>
  <div id="ctx">__CONTEXT__</div>
</header>
<main id="list"></main>
<footer><kbd>j</kbd>/<kbd>k</kbd> next/prev &middot; <kbd>1</kbd>&ndash;<kbd>9</kbd> pick reading &middot;
<kbd>e</kbd> free-text override &middot; decisions autosave to this browser (localStorage)</footer>
<script id="review-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("review-data").textContent);
const ITEMS = DATA.items;
// Per-artifact namespace: first 10 hex chars of sha256 over the concatenated
// sorted item ids + defaults, computed at build time. Keys are NS + ":" + id
// so two generated review HTMLs never share localStorage entries.
const NS = DATA.ns || "";

// localStorage-backed decision store (key: NS + ":" + item id); in-memory
// fallback if localStorage is unavailable (some browsers restrict file://).
// Every entry records the item default it was decided against; entries whose
// recorded default no longer matches the current item default are ignored.
const mem = {};
const store = {
  get(it) {
    let v = null;
    try {
      const raw = localStorage.getItem(NS + ":" + it.id);
      if (raw !== null) v = JSON.parse(raw);
    } catch (e) {}
    if (v === null) v = mem[it.id] || null;
    if (v && v.def !== it.default) return null;  // stale: default changed
    return v;
  },
  set(it, val) {
    val = Object.assign({}, val, { def: it.default });
    mem[it.id] = val;
    try { localStorage.setItem(NS + ":" + it.id, JSON.stringify(val)); } catch (e) {}
  },
  del(it) {
    delete mem[it.id];
    try { localStorage.removeItem(NS + ":" + it.id); } catch (e) {}
  },
};

function escHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function csvEsc(v) {
  v = String(v == null ? "" : v);
  return /[",\\n\\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}

const list = document.getElementById("list");
const rows = [];
let cur = -1;

ITEMS.forEach((it, idx) => {
  const row = document.createElement("div");
  row.className = "item";
  row.dataset.id = it.id;

  let meta = '<span class="tag">page ' + it.page + "</span><span>" + escHtml(it.id) + "</span>";
  if (!it.located) meta += '<span class="tag approx">approximate location</span>';
  if (it.decided) meta += '<span class="tag applied">already applied: ' + escHtml(it.prior) + "</span>";
  const imgHtml = it.img
    ? '<img class="crop" src="' + it.img + '" alt="page ' + it.page + ' crop">'
    : '<div class="noimg">page image unavailable</div>';
  row.innerHTML =
    '<div class="meta">' + meta + "</div>" + imgHtml +
    '<div class="ctxline">&hellip; ' + escHtml(it.before) + " <mark>" +
    (escHtml(it.default) || "&nbsp;") + "</mark> " + escHtml(it.after) + " &hellip;</div>";

  const cwrap = document.createElement("div");
  cwrap.className = "cands";
  const opts = it.candidates.concat([{ text: "[illegible]", sources: [], illeg: true }]);
  opts.forEach((c, ci) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "cand" + (c.illeg ? " illeg" : "") +
                  (!c.illeg && c.text === it.default ? " default" : "");
    b.dataset.value = c.illeg ? "[illegible]" : c.text;
    let inner = "";
    if (ci < 9) inner += '<span class="kbd">' + (ci + 1) + "</span>";
    inner += '<span class="rd"></span><span class="src"></span>';
    b.innerHTML = inner;
    b.querySelector(".rd").textContent = c.illeg ? "illegible [?]" : c.text;
    b.querySelector(".src").textContent =
      c.illeg ? "unreadable" : (c.sources.join(", ") || (c.text === it.default ? "default" : ""));
    b.addEventListener("click", () => {
      choose(idx, b.dataset.value,
             c.illeg ? "illegible" : (c.sources.join("|") || "candidate"));
    });
    cwrap.appendChild(b);
  });
  const inp = document.createElement("input");
  inp.type = "text";
  inp.className = "override";
  inp.placeholder = "free-text override…";
  inp.addEventListener("input", () => {
    const v = inp.value.trim();
    if (v) {
      choose(idx, v, "override");
    } else {
      const s = store.get(it);
      if (s && s.note === "override") { store.del(it); refreshRow(idx); updateProgress(); }
    }
  });
  inp.addEventListener("focus", () => setActive(idx, false));
  cwrap.appendChild(inp);
  row.appendChild(cwrap);
  row.addEventListener("click", () => setActive(idx, false));
  list.appendChild(row);
  rows.push(row);
});

function choose(idx, decision, note) {
  store.set(ITEMS[idx], { decision: decision, note: note });
  refreshRow(idx);
  updateProgress();
  setActive(idx, false);
}

function refreshRow(idx) {
  const it = ITEMS[idx];
  const row = rows[idx];
  const s = store.get(it);
  row.classList.toggle("done", !!(s && s.decision));
  row.querySelectorAll("button.cand").forEach(b => {
    b.classList.toggle("selected",
      !!(s && s.note !== "override" && s.decision === b.dataset.value));
  });
  const inp = row.querySelector("input.override");
  const isOv = !!(s && s.note === "override");
  inp.classList.toggle("selected", isOv);
  if (isOv && document.activeElement !== inp) inp.value = s.decision;
  if (!isOv && document.activeElement !== inp) inp.value = "";
}

function updateProgress() {
  let n = 0;
  ITEMS.forEach(it => { const s = store.get(it); if (s && s.decision) n++; });
  document.getElementById("progress").textContent = n + " reviewed / " + ITEMS.length;
}

function setActive(idx, scroll) {
  if (idx < 0 || idx >= rows.length) return;
  if (cur >= 0) rows[cur].classList.remove("active");
  cur = idx;
  rows[cur].classList.add("active");
  if (scroll) rows[cur].scrollIntoView({ block: "center" });
}

document.addEventListener("keydown", e => {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) {
    if (e.key === "Escape") t.blur();
    return;
  }
  if (e.key === "j") { setActive(cur + 1, true); e.preventDefault(); }
  else if (e.key === "k") { setActive(Math.max(cur - 1, 0), true); e.preventDefault(); }
  else if (e.key === "e") {
    if (cur >= 0) { rows[cur].querySelector("input.override").focus(); e.preventDefault(); }
  } else if (/^[1-9]$/.test(e.key)) {
    if (cur >= 0) {
      const bs = rows[cur].querySelectorAll("button.cand");
      const b = bs[Number(e.key) - 1];
      if (b) b.click();
      e.preventDefault();
    }
  }
});

document.getElementById("export").addEventListener("click", () => {
  const lines = ["item_id,page,decision,note"];
  ITEMS.forEach(it => {
    if (it.decided) return;  // already applied at build time: never re-export
    const s = store.get(it);
    if (s && s.decision) {
      lines.push([it.id, it.page, s.decision, s.note || ""].map(csvEsc).join(","));
    }
  });
  const blob = new Blob([lines.join("\\n") + "\\n"], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "review_decisions.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

ITEMS.forEach((_, i) => refreshRow(i));
updateProgress();
if (rows.length) setActive(0, false);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Build the single-file HTML review UI for name adjudication")
    ap.add_argument("--base", required=True, help="base dir, e.g. data/transcripts/bentley_deposition")
    ap.add_argument("--out", default=None,
                    help="output HTML path (default: BASE/review_bentley_deposition.html)")
    ap.add_argument("--title", default="Bentley Deposition — Name Review")
    ap.add_argument("--context", default=DEFAULT_CONTEXT,
                    help="collection context line shown in the header")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.base, "review_bentley_deposition.html")
    queue_path = os.path.join(args.base, "review_queue.json")
    if not os.path.exists(queue_path):
        raise SystemExit(f"{queue_path} not found -- run the adjudicator first")
    with open(queue_path, encoding="utf-8") as f:
        raw_items = json.load(f)
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("items", [])
    if not isinstance(raw_items, list):
        raise SystemExit(f"{queue_path}: expected a list of review items")

    img_cache, tx_cache = {}, {}

    def page_image(page):
        if page not in img_cache:
            path = os.path.join(args.base, "images", f"{page:04d}.png")
            if os.path.exists(path):
                img_cache[page] = Image.open(path)
            else:
                print(f"warning: {path} missing -- items on page {page} get no crop")
                img_cache[page] = None
        return img_cache[page]

    def page_textract(page):
        if page not in tx_cache:
            tx_cache[page] = load_textract(args.base, page)
        return tx_cache[page]

    ntok_cache = {}

    def page_ntokens(page):
        """Token count of BASE/final/{page:04d}.txt -- the denominator for the
        pos/total fallback strip when items carry no total_positions."""
        if page not in ntok_cache:
            path = os.path.join(args.base, "final", f"{page:04d}.txt")
            n = 0
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    n = len(re.findall(r"\S+", f.read()))
            ntok_cache[page] = n
        return ntok_cache[page]

    payload, n_located, n_fallback, n_noimg = [], 0, 0, 0
    for idx, raw in enumerate(raw_items):
        it = canon_item(raw, idx)
        if it["total"] <= 0:
            it["total"] = page_ntokens(it["page"])
        img = page_image(it["page"])
        tx = page_textract(it["page"])
        data_uri, located = None, False
        if img is not None and tx is not None:
            word = locate_word(tx, it["before"], it["default"], it["after"])
            if word is not None:
                data_uri = make_crop(img, tx, word)
                located = data_uri is not None
        if img is not None and not located:
            data_uri = fallback_strip(img, it["pos"], it["total"])
        if data_uri is None:
            n_noimg += 1
        elif located:
            n_located += 1
        else:
            n_fallback += 1
        payload.append({
            "id": it["id"],
            "page": it["page"],
            "pos": it["pos"],
            "default": it["default"],
            "before": " ".join(it["before"]),
            "after": " ".join(it["after"]),
            "candidates": [{"text": t, "sources": s} for t, s in it["candidates"]],
            "img": data_uri,
            "located": located,
            "decided": it["decided"],
            "prior": it["prior"],
        })

    # Per-artifact localStorage namespace: first 10 hex chars of sha256 over
    # the concatenated sorted item ids + defaults (separators keep id/default
    # pairs unambiguous). Any change to the item set or its defaults yields a
    # fresh namespace, so stale entries from another build are never read.
    ns_src = "".join(
        iid + "\x1f" + default + "\x1e"
        for iid, default in sorted((p["id"], p["default"]) for p in payload)
    )
    ns = hashlib.sha256(ns_src.encode("utf-8")).hexdigest()[:10]
    data_json = json.dumps({"ns": ns, "items": payload},
                           ensure_ascii=False).replace("</", "<\\/")
    doc = (TEMPLATE
           .replace("__TITLE__", html_mod.escape(args.title))
           .replace("__CONTEXT__", html_mod.escape(args.context))
           .replace("__DATA__", data_json))
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(doc)
    os.replace(tmp, out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"wrote {out_path}: {len(payload)} items "
          f"({n_located} located, {n_fallback} fallback strips, {n_noimg} without image), "
          f"{size_kb:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
