"""Local (no-pod) batch post-processor for salvaged tier-1 docs.

Processes every manifest doc whose olmocr pages are complete on disk:
converts local surya results into the OCR-vote slot (when present), exports
the embedded layer from prod (read-only), adjudicates (skipped when final/
is already complete), dry-runs the load, and appends passing load commands
to WORK/loads_to_run.sh. Docs already logged OK in batch_log.tsv are skipped;
docs previously logged CHECK are re-attempted and re-logged.

    DATABASE_URL=... python scripts/batch_post_local.py \
        --manifests data/transcripts/batch_tier1/manifest_t1_shard1.json \
                    data/transcripts/batch_tier1/manifest_t1_shard2.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psycopg2

ap = argparse.ArgumentParser()
ap.add_argument("--manifests", nargs="+", required=True)
ap.add_argument("--exclude-manifest", help="skip docs listed here (e.g. docs a pod is still working on)")
ap.add_argument("--work", default="data/transcripts/batch_tier1")
args = ap.parse_args()

excluded = set()
if args.exclude_manifest:
    excluded = {d["doc_id"] for d in json.load(open(args.exclude_manifest))}

PY = sys.executable
WORK = Path(args.work)
GAZ = Path("data/transcripts/bentley_deposition/gazetteer.txt").resolve()
strip_html = re.compile(r"<[^>]+>")

docs = {}
for mf in args.manifests:
    for d in json.load(open(mf)):
        docs[d["doc_id"]] = d


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process_doc(doc_id):
    d = docs[doc_id]
    dd = WORK / str(doc_id)
    n = d["pages"]

    surya_res = dd / "surya" / "images" / "results.json"
    tex = dd / "textract"
    if surya_res.exists() and len(list(tex.glob("*.txt")) if tex.is_dir() else []) < 1:
        surya = json.load(open(surya_res))
        tex.mkdir(exist_ok=True)
        for key, pages_ in surya.items():
            text = "\n".join(strip_html.sub("", b.get("html", "")) for b in pages_[0]["blocks"])
            (tex / f"{int(key):04d}.txt").write_text(text, encoding="utf-8")

    emb = dd / "embedded"
    if len(list(emb.glob("*.txt")) if emb.is_dir() else []) < n:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()
        emb.mkdir(exist_ok=True)
        cur.execute("SELECT page_seq, raw_text FROM pages WHERE document_id = %s ORDER BY page_seq", (doc_id,))
        for seq, txt in cur.fetchall():
            (emb / f"{seq:04d}.txt").write_text(txt or "", encoding="utf-8")
        conn.close()

    gz = dd / "gazetteer.txt"
    if not gz.exists():
        gz.symlink_to(GAZ)

    final = dd / "final"
    if len(list(final.glob("*.txt")) if final.is_dir() else []) >= n:
        adj = "adjudication cached (final/ complete)"
    else:
        r = subprocess.run([PY, "scripts/adjudicate_transcript.py", "--base", str(dd), "--pages", f"1-{n}"],
                           capture_output=True, text=True, timeout=1800)
        adj = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
        if r.returncode != 0:
            log(f"  doc {doc_id}: adjudication FAILED: {adj}")
            return "FAIL", adj, ""

    guard = d["source_name"][:30]
    pv = d.get("pipeline_version", "chunk_v1_silvermaster_structured_4k")
    try:
        r = subprocess.run(
            [PY, "scripts/load_transcript.py", "--doc-id", str(doc_id), "--base", str(dd),
             "--pages", f"1-{n}", "--expect-source", guard, "--pipeline-version", pv, "--dry-run"],
            capture_output=True, text=True, env=dict(os.environ), timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return "CHECK", adj, "dry-run TIMEOUT >1800s (mapper pathology?)"
    m = re.search(r"Chunk mapping.*", r.stdout)
    mapline = m.group(0)[:180] if m else "no mapping line"
    stats = re.search(
        r"(\d+)/(\d+) chunks mapped over .*? (\d+) skipped with\s+warnings.*?(\d+) unmapped",
        r.stdout, re.DOTALL)
    oversized = len(re.findall(r"SKIPPED — new clean_text", r.stdout))
    ok = False
    if r.returncode == 0 and stats:
        mapped, total, file_skipped, unmapped = (int(g) for g in stats.groups())
        ok = file_skipped == 0 and total > 0 and unmapped / total <= 0.10
        mapline += f" [oversized={oversized}]"
    if ok:
        cmd = (f'"$PY" scripts/load_transcript.py --doc-id {doc_id} --base {dd} '
               f'--pages 1-{n} --expect-source "{guard}" --pipeline-version {pv} '
               f'--status machine_unverified --reset-embeddings '
               f'--method "vision-ensemble-free-v1 (olmOCR-2 bf16 + '
               f'Surya-2 + source OCR, majority-vote adjudication)"\n')
        with open(WORK / "loads_to_run.sh", "a") as f:
            f.write(cmd)
    return ("OK" if ok else "CHECK"), adj, mapline


logged = {}   # doc_id -> status from prior runs
logpath = WORK / "batch_log.tsv"
if logpath.exists():
    for line in open(logpath):
        parts = line.rstrip("\n").split("\t")
        if parts and parts[0].isdigit():
            logged[int(parts[0])] = parts[2] if len(parts) > 2 else "?"

todo = []
for doc_id, d in sorted(docs.items()):
    if doc_id in excluded or logged.get(doc_id) == "OK":
        continue
    dd = WORK / str(doc_id)
    if len(list((dd / "pages").glob("*.json"))) < d["pages"]:
        continue  # olmocr layer incomplete; not processable
    todo.append(doc_id)

log(f"{len(todo)} docs to process locally ({len(logged)} in prior log)")
counts = {"OK": 0, "CHECK": 0, "FAIL": 0}
for i, doc_id in enumerate(todo, 1):
    try:
        status, adj, mapline = process_doc(doc_id)
    except Exception as e:
        status, adj, mapline = "FAIL", f"EXCEPTION {e}", ""
    counts[status] += 1
    with open(logpath, "a") as f:
        f.write(f"{doc_id}\t{docs[doc_id]['pages']}\t{status}\t{adj}\t{mapline}\n")
    log(f"  [{i}/{len(todo)}] doc {doc_id}: {status} — {mapline or adj}")

log(f"LOCAL BATCH DONE: {counts}")
