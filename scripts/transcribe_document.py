"""Multi-model vision transcription of scanned PDF pages.

Renders each requested page to BASE/images/{page:04d}.png (width 1700px) and
transcribes each (page, model) pair to BASE/pages/{page:04d}.{model}.json.
Resumable: existing JSONs are skipped. Usage:

  python scripts/transcribe_document.py \
      --pdf "data/raw/silvermaster/pdf/FBI File Silvermaster Part 6 late November 1945_text.pdf" \
      --pages 3-120 --models gpt-5.5,gpt-5.2,gpt-4.1 \
      --base data/transcripts/bentley_deposition [--concurrency 4] [--limit N]
"""
import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz

MAX_ATTEMPTS = 5  # retries on rate-limit / transient API errors

PROMPT = """You are transcribing a 1945 typewritten FBI document from a poor-quality scanned photostat.

Rules:
- Transcribe EXACTLY what is typed on the page: preserve original spelling, capitalization, punctuation, and any typist errors. This is a historical source; fidelity to the page outranks readability.
- Do NOT normalize or "correct" personal names. Render names exactly as typed.
- If a character or word is genuinely unreadable, write [illegible].
- If you can read something but are uncertain, wrap your best reading like [?word].
- Preserve paragraph structure. Join words hyphenated across line breaks.
- Transcribe page headers and file numbers (e.g. "NY 65-14603") as they appear.
- Handwritten serial stamps or numbers: render on their own line as [stamp: ...]. Ignore other stray pen marks.
- Output ONLY the transcription. No commentary, no markdown."""

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def load_env():
    """Load .env without exporting to shell. Never print secret values."""
    env = {}
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v.strip().strip('"').strip("'")
    return env


def parse_pages(spec):
    """'A-B' -> [A..B] inclusive; 'A' -> [A]. 1-based pdf page numbers."""
    if "-" in spec:
        a, b = spec.split("-", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(spec)
    if start < 1 or end < start:
        raise ValueError(f"bad --pages range: {spec}")
    return list(range(start, end + 1))


def png_ok(path):
    """True when `path` looks like a complete PNG: exists, is > 100 bytes,
    starts with the PNG magic bytes, and ends with an IEND chunk. Any failure
    (missing, tiny, truncated, unreadable) means "treat as missing"."""
    try:
        if os.path.getsize(path) <= 100:
            return False
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
            f.seek(-12, os.SEEK_END)
            tail = f.read(12)
        return b"IEND" in tail
    except OSError:
        return False


def render_missing_pages(pdf_path, pages, images_dir):
    """Render any missing BASE/images/{page:04d}.png at width 1700px (serial: fitz).

    A PNG that exists but fails png_ok (truncated by an earlier interrupt) is
    re-rendered. Writes are atomic: render to out+".tmp", then os.replace."""
    missing = [p for p in pages if not png_ok(os.path.join(images_dir, f"{p:04d}.png"))]
    if not missing:
        return
    doc = fitz.open(pdf_path)
    if max(missing) > doc.page_count:
        doc.close()
        raise SystemExit(f"page {max(missing)} out of range: PDF has {doc.page_count} pages")
    for p in missing:
        out_png = os.path.join(images_dir, f"{p:04d}.png")
        page = doc[p - 1]
        zoom = 1700 / page.rect.width
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        tmp_png = out_png + ".tmp"
        pix.save(tmp_png, output="png")
        os.replace(tmp_png, out_png)  # atomic: no truncated PNG on interrupt
        log(f"rendered page {p}: {pix.width}x{pix.height} -> {out_png}")
    doc.close()


def is_transient(exc):
    import openai

    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
        return True
    return False


def call_model(client, model, b64, reasoning_effort=None, max_tokens=None, temperature=None):
    kwargs = {}
    if reasoning_effort:
        # gpt-5.x only; lets a calibration run price out reduced reasoning.
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens:
        # Local VLMs can repetition-loop on degraded pages; a hard cap makes
        # the loop terminate as one bad (outvotable) reading instead of hanging.
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ],
        }],
        **kwargs,
    )
    return resp


def transcribe_pair(client, page_num, model, images_dir, pages_dir, reasoning_effort=None,
                    max_tokens=None, temperature=None):
    """Transcribe one (page, model) pair. Returns (page, model, usage_dict) or raises."""
    png_path = os.path.join(images_dir, f"{page_num:04d}.png")
    b64 = base64.b64encode(open(png_path, "rb").read()).decode()

    resp = None
    empty_retried = False
    attempt = 0
    while True:
        attempt += 1
        try:
            t0 = time.time()
            resp = call_model(client, model, b64, reasoning_effort=reasoning_effort,
                              max_tokens=max_tokens, temperature=temperature)
            latency = time.time() - t0
        except Exception as e:
            if is_transient(e) and attempt < MAX_ATTEMPTS:
                delay = min(60, 2 ** attempt) + random.uniform(0, 1)
                log(f"page {page_num} {model}: transient error ({type(e).__name__}), "
                    f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
        text = resp.choices[0].message.content or ""
        if not text.strip():
            if not empty_retried:
                empty_retried = True
                log(f"page {page_num} {model}: empty response, retrying once")
                continue
            raise RuntimeError("empty response after retry")
        break

    usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
    out = {"page": page_num, "model": model, "text": text, "usage": usage}
    out_path = os.path.join(pages_dir, f"{page_num:04d}.{model}.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    log(f"page {page_num} {model}: {latency:.1f}s, {len(text)} chars")
    return page_num, model, usage


def main():
    ap = argparse.ArgumentParser(description="Multi-model vision transcription of scanned PDF pages")
    ap.add_argument("--pdf", required=True, help="path to source PDF")
    ap.add_argument("--pages", required=True, help="1-based inclusive page range, e.g. 3-120")
    ap.add_argument("--models", required=True, help="comma-separated model names, e.g. gpt-5.5,gpt-5.2,gpt-4.1")
    ap.add_argument("--base", required=True, help="output base dir, e.g. data/transcripts/bentley_deposition")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel (page,model) API calls (default 4)")
    ap.add_argument("--limit", type=int, default=None, help="max number of pending (page,model) pairs to run")
    ap.add_argument("--reasoning-effort", default=None,
                    help="reasoning_effort for gpt-5.x models (e.g. minimal, low); omit for model default")
    ap.add_argument("--api-base", default=None,
                    help="OpenAI-compatible endpoint (e.g. http://localhost:11434/v1 for Ollama); default = OpenAI")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap completion tokens (recommended ~1600 for local VLMs: ends repetition loops)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature (0.1 recommended for local OCR; omit for gpt-5.x)")
    ap.add_argument("--api-key", default=None,
                    help="API key for --api-base (local servers accept any string); default = OPENAI_API_KEY")
    args = ap.parse_args()

    pages = parse_pages(args.pages)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("no models given")

    images_dir = os.path.join(args.base, "images")
    pages_dir = os.path.join(args.base, "pages")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)

    pending = [
        (p, m)
        for p in pages
        for m in models
        if not os.path.exists(os.path.join(pages_dir, f"{p:04d}.{m}.json"))
    ]
    total = len(pages) * len(models)
    log(f"{total} (page,model) pairs in range; {total - len(pending)} already done; {len(pending)} pending")
    if pending:
        # Only touch images/PDF when there is actual work: a fully-cached doc
        # must succeed even if its images and source PDF were cleaned up.
        render_missing_pages(args.pdf, sorted({p for p, _ in pending}), images_dir)
    if args.limit is not None:
        pending = pending[: args.limit]
        log(f"limited to {len(pending)} pairs")
    if not pending:
        log("nothing to do")
        return 0

    env = load_env()
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.api_base:
        # A local OpenAI-compatible server (--api-base) needs no real key.
        raise SystemExit("OPENAI_API_KEY not found in .env or environment")

    from openai import OpenAI

    if args.api_base:
        client = OpenAI(base_url=args.api_base, api_key=args.api_key or "local")
    else:
        client = OpenAI(api_key=api_key)

    totals = {m: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0} for m in models}
    failed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(transcribe_pair, client, p, m, images_dir, pages_dir,
                        args.reasoning_effort, args.max_tokens, args.temperature): (p, m)
            for p, m in pending
        }
        for fut in as_completed(futures):
            p, m = futures[fut]
            try:
                _, _, usage = fut.result()
            except Exception as e:
                log(f"page {p} {m}: FAILED ({type(e).__name__}: {e})")
                failed.append((p, m, f"{type(e).__name__}: {e}"))
                continue
            totals[m]["calls"] += 1
            totals[m]["prompt_tokens"] += usage["prompt_tokens"]
            totals[m]["completion_tokens"] += usage["completion_tokens"]

    log("--- usage summary (this run) ---")
    for m in models:
        t = totals[m]
        log(f"{m}: {t['calls']} calls, prompt_tokens={t['prompt_tokens']}, completion_tokens={t['completion_tokens']}")

    if failed:
        log(f"--- {len(failed)} pair(s) permanently failed ---")
        for p, m, err in sorted(failed):
            log(f"page {p} {m}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
