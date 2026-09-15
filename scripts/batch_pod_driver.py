"""Pod-side batch driver, PHASED: the GPU is owned by exactly one engine at a time.

Stage "olmocr": download/render each doc, run olmOCR via local vLLM, mark OLMOCR_DONE,
                delete images/pdf (surya stage re-renders).
Stage "surya":  re-download/render, run Surya (llamacpp backend, GPU to itself),
                mark DONE, delete images/pdf.
Stage "both":   legacy single-pass behavior (not used in phased operation).

The supervisor flips /root/STAGE from olmocr -> surya when the olmocr stage
completes, killing vLLM at the transition.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", default="/root/batch_out")
ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
ap.add_argument("--concurrency", type=int, default=10)
ap.add_argument("--stage", choices=["both", "olmocr", "surya"], default="both")
ap.add_argument("--limit-docs", type=int, default=None)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
docs = json.load(open(args.manifest))
if args.limit_docs:
    docs = docs[: args.limit_docs]
os.makedirs(args.out, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url, dest, attempts=4):
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dest + ".tmp")
            os.replace(dest + ".tmp", dest)
            return
        except Exception as e:
            if i == attempts - 1:
                raise
            log(f"  download retry {i+1}: {e}")
            time.sleep(5 * (i + 1))


def render(pdf, images_dir, pages):
    import fitz
    os.makedirs(images_dir, exist_ok=True)
    doc = fitz.open(pdf)
    n = min(pages, doc.page_count)
    for p in range(1, n + 1):
        out_png = os.path.join(images_dir, f"{p:04d}.png")
        if os.path.exists(out_png) and os.path.getsize(out_png) > 100:
            continue
        page = doc[p - 1]
        zoom = 1700 / page.rect.width
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(out_png + ".tmp", output="png")
        os.replace(out_png + ".tmp", out_png)
    doc.close()
    return n


def skip_marker(dd):
    """True when this stage has nothing to do for the doc."""
    if os.path.exists(os.path.join(dd, "DONE")):
        return True
    if args.stage == "olmocr" and os.path.exists(os.path.join(dd, "OLMOCR_DONE")):
        return True
    return False


def cleanup(dd, pdf):
    subprocess.run(["rm", "-rf", os.path.join(dd, "images"), pdf])


def prepare(d):
    """Download + render one doc (prefetch thread). Returns page count or None to skip."""
    dd = os.path.join(args.out, str(d["doc_id"]))
    if skip_marker(dd):
        return None
    if args.stage == "surya" and os.path.exists(os.path.join(dd, "FAILED_OLMOCR")):
        return None  # olmocr failed on this doc; surya alone is useless
    # (A missing pages/ dir is NOT a skip: on a fresh surya-only pod the olmocr
    # output lives elsewhere and surya must still run from re-rendered images.)
    if args.stage == "olmocr":
        pages_dir = os.path.join(dd, "pages")
        if os.path.isdir(pages_dir) and len(os.listdir(pages_dir)) >= d["pages"]:
            return d["pages"]  # fully cached: transcribe will no-op, no images needed
    os.makedirs(dd, exist_ok=True)
    pdf = os.path.join(dd, "doc.pdf")
    if not os.path.exists(pdf):
        download(d["pdf_url"], pdf)
    return render(pdf, os.path.join(dd, "images"), d["pages"])


def run_olmocr(dd, pdf, n, doc_id):
    r = None
    for attempt in (1, 2):
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "transcribe_document.py"),
             "--pdf", pdf, "--pages", f"1-{n}", "--models", "olmocr2",
             "--base", dd, "--api-base", args.api_base,
             "--max-tokens", "1600", "--temperature", "0.1",
             "--concurrency", str(args.concurrency)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True
        log(f"  olmocr2 attempt {attempt} failed for doc {doc_id}")
    open(os.path.join(dd, "FAILED_OLMOCR"), "w").write(
        (r.stdout + "\n--STDERR--\n" + r.stderr)[-3000:])
    return False


def run_surya(dd, n, doc_id):
    env = dict(os.environ,
               SURYA_INFERENCE_BACKEND="llamacpp",
               LLAMA_CPP_BINARY="/root/llama.cpp/build/bin/llama-server")
    res = os.path.join(dd, "surya", "images", "results.json")
    surya_timeout = max(900, n * 10)
    for attempt in (1, 2):
        try:
            subprocess.run(
                ["surya_ocr", os.path.join(dd, "images"),
                 "--output_dir", os.path.join(dd, "surya")],
                capture_output=True, text=True, env=env, timeout=surya_timeout,
            )
        except subprocess.TimeoutExpired:
            log(f"  surya TIMED OUT after {surya_timeout}s (attempt {attempt})")
        subprocess.run(["pkill", "-f", "llama-server"])
        if os.path.exists(res):
            return True
        time.sleep(10)
    open(os.path.join(dd, "FAILED_SURYA"), "w").write("no results after 2 attempts")
    return False


prep_pool = ThreadPoolExecutor(max_workers=2)
futures = {0: prep_pool.submit(prepare, docs[0])} if docs else {}
total_pages_done = 0
t_start = time.time()
for i, d in enumerate(docs):
    doc_id = d["doc_id"]
    dd = os.path.join(args.out, str(doc_id))
    for j in (i + 1, i + 2):
        if j < len(docs) and j not in futures:
            futures[j] = prep_pool.submit(prepare, docs[j])
    if shutil.disk_usage("/").free < 3 * 2**30:
        log("DISK LOW (<3GB free) — aborting so cached work survives")
        break
    try:
        n = futures.pop(i).result()
    except Exception as e:
        log(f"doc {doc_id}: prepare FAILED: {e}; continuing")
        continue
    if n is None:
        log(f"doc {doc_id}: nothing to do this stage, skipping")
        continue
    pdf = os.path.join(dd, "doc.pdf")
    log(f"doc {doc_id} ({i+1}/{len(docs)}) [{args.stage}]: {d['source_name'][:55]} [{n}pp]")

    ok = True
    if args.stage in ("both", "olmocr"):
        ok = run_olmocr(dd, pdf, n, doc_id)
    if ok and args.stage in ("both", "surya"):
        ok = run_surya(dd, n, doc_id)

    cleanup(dd, pdf)
    if ok:
        marker = "OLMOCR_DONE" if args.stage == "olmocr" else "DONE"
        open(os.path.join(dd, marker), "w").write("ok")
        total_pages_done += n
        rate = total_pages_done / max(time.time() - t_start, 1)
        log(f"  {marker} doc {doc_id}; stage total {total_pages_done} pages at {rate:.2f} p/s")
    else:
        log(f"  doc {doc_id} FAILED this stage; continuing")

log(f"STAGE {args.stage} COMPLETE")
if args.stage in ("both", "surya"):
    log("BATCH COMPLETE")
