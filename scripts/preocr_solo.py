#!/usr/bin/env python3
"""Pre-OCR a staged directory into the ingest cache, balanced across shards.

Renders+Textracts every PDF page and writes ocr_cache/<slug>/<name>.json
(a JSON list of page strings) -- exactly the cache shape that
ingest_dir_collection.py --ocr expects, so the later ingest just chunks+inserts
from cache (no re-OCR).

Two roles:
  plan : compute a page-balanced shard assignment (greedy LPT), write _shards.json
  run  : OCR the files assigned to one shard (resumable: skips valid caches)

Usage:
  python -m scripts.preocr_solo plan --dir data/raw/fbi_solo --slug fbi_solo --shards 6
  python -m scripts.preocr_solo run  --slug fbi_solo --shard 0 --workers 8
"""
import os, sys, glob, json, time, argparse, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import fitz
import ocr_textract

def cache_path(cache_dir, slug, name):
    return os.path.join(cache_dir, slug, name + ".json")

def shards_file(cache_dir, slug):
    return os.path.join(cache_dir, slug, "_shards.json")

def page_count(path):
    d = fitz.open(path); n = d.page_count; d.close(); return n

def valid_cache(path, expected_pages):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return isinstance(data, list) and len(data) == expected_pages
    except Exception:
        return False

def do_plan(args):
    files = sorted(glob.glob(os.path.join(args.dir, "*.pdf")))
    sizes = []
    for f in files:
        try:
            sizes.append((os.path.basename(f), page_count(f)))
        except Exception as e:
            print(f"  WARN cannot open {os.path.basename(f)}: {e}")
    # greedy LPT: largest first into least-loaded shard
    sizes.sort(key=lambda x: -x[1])
    buckets = [[] for _ in range(args.shards)]
    loads = [0] * args.shards
    for name, pg in sizes:
        i = loads.index(min(loads))
        buckets[i].append(name)
        loads[i] += pg
    os.makedirs(os.path.join(args.cache_dir, args.slug), exist_ok=True)
    plan = {str(i): buckets[i] for i in range(args.shards)}
    with open(shards_file(args.cache_dir, args.slug), "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=0)
    print(f"planned {len(files)} files, {sum(loads)} pages across {args.shards} shards:")
    for i in range(args.shards):
        print(f"  shard {i}: {len(buckets[i]):>3} files, {loads[i]:>6} pages")

def do_run(args):
    with open(shards_file(args.cache_dir, args.slug), encoding="utf-8") as fh:
        plan = json.load(fh)
    names = plan[str(args.shard)]
    cdir = os.path.join(args.cache_dir, args.slug)
    os.makedirs(cdir, exist_ok=True)
    tag = f"shard{args.shard}"
    done_pages = 0
    t_start = time.time()
    for k, name in enumerate(names, 1):
        src = os.path.join(args.dir, name)
        cpath = cache_path(args.cache_dir, args.slug, name)
        try:
            n = page_count(src)
        except Exception as e:
            print(f"[{tag}] OPENERR {name}: {e}", flush=True)
            continue
        if valid_cache(cpath, n):
            done_pages += n
            print(f"[{tag}] ({k}/{len(names)}) cached  {name[:46]:46} {n:>4}p", flush=True)
            continue
        t0 = time.time()
        try:
            pages = ocr_textract.ocr_pdf_parallel(src, region=args.region, dpi=args.dpi,
                                                  max_workers=args.workers, progress=None)
        except Exception as e:
            print(f"[{tag}] OCRFAIL {name[:46]}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        # atomic write
        fd, tmp = tempfile.mkstemp(dir=cdir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(pages, fh)
        os.replace(tmp, cpath)
        done_pages += n
        dt = time.time() - t0
        rate = done_pages / max(1, time.time() - t_start)
        print(f"[{tag}] ({k}/{len(names)}) OCR    {name[:46]:46} {n:>4}p {dt:5.1f}s  ~{rate:.2f}pg/s cum", flush=True)
    print(f"[{tag}] DONE {len(names)} files, {done_pages} pages in {(time.time()-t_start)/60:.1f} min", flush=True)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan");  p.add_argument("--dir", required=True); p.add_argument("--slug", required=True)
    p.add_argument("--shards", type=int, default=6); p.add_argument("--cache-dir", default="ocr_cache")
    r = sub.add_parser("run");   r.add_argument("--dir", default=None); r.add_argument("--slug", required=True)
    r.add_argument("--shard", type=int, required=True); r.add_argument("--workers", type=int, default=8)
    r.add_argument("--cache-dir", default="ocr_cache"); r.add_argument("--region", default="us-east-1")
    r.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    if args.cmd == "plan":
        do_plan(args)
    else:
        if not args.dir:
            args.dir = os.path.join("data", "raw", args.slug)
        do_run(args)

if __name__ == "__main__":
    main()
