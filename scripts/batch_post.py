"""Mac-side batch post-processor for the Silvermaster batch.

Polls the pod for per-doc DONE markers; for each completed doc: pulls the
olmOCR page JSONs + surya results (small text), converts surya into the
OCR-vote slot, exports the doc's embedded layer from prod (read-only),
adjudicates, runs a load --dry-run, and appends the live-load command to
WORK/loads_to_run.sh. Resumable; writes WORK/batch_log.tsv.

    DATABASE_URL=... python scripts/batch_post.py \
        --ssh "root@99.69.17.69" --port 10477 --key ~/.ssh/friday_runpod \
        --manifest data/transcripts/batch_silvermaster/manifest.json
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
ap.add_argument("--ssh", required=True)
ap.add_argument("--port", required=True)
ap.add_argument("--key", default=os.path.expanduser("~/.ssh/friday_runpod"))
ap.add_argument("--manifest", required=True)
ap.add_argument("--remote-out", default="/root/batch_out")
ap.add_argument("--work", default="data/transcripts/batch_silvermaster")
ap.add_argument("--poll-seconds", type=int, default=120)
ap.add_argument("--max-hours", type=float, default=40)
args = ap.parse_args()

PY = sys.executable
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-i", args.key, "-p", str(args.port), args.ssh]
WORK = Path(args.work)
GAZ = Path("data/transcripts/bentley_deposition/gazetteer.txt").resolve()
docs = {d["doc_id"]: d for d in json.load(open(args.manifest))}
strip_html = re.compile(r"<[^>]+>")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ssh_out(cmd):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout


def process_doc(doc_id):
    d = docs[doc_id]
    dd = WORK / str(doc_id)
    (dd / "pages").mkdir(parents=True, exist_ok=True)
    # Pull page JSONs + surya results in one tar stream. If the olmocr pages are
    # already local (e.g. restored from a dead pod's backup), pull only surya;
    # if surya results are local too (salvaged doc), skip the pull entirely.
    have_pages = len(list((dd / "pages").glob("*.json"))) >= d["pages"]
    have_surya = (dd / "surya" / "images" / "results.json").exists()
    if not (have_pages and have_surya):
        targets = "surya/images/results.json" if have_pages else "pages surya/images/results.json"
        r = subprocess.run(
            SSH + [f"cd {args.remote_out}/{doc_id} && tar czf - {targets}"],
            capture_output=True, timeout=600,
        )
        if r.returncode != 0:
            log(f"  doc {doc_id}: pull failed"); return False
        subprocess.run(["tar", "xzf", "-", "-C", str(dd)], input=r.stdout, check=True)

    surya = json.load(open(dd / "surya" / "images" / "results.json"))
    tex = dd / "textract"
    tex.mkdir(exist_ok=True)
    for key, pages_ in surya.items():
        text = "\n".join(strip_html.sub("", b.get("html", "")) for b in pages_[0]["blocks"])
        (tex / f"{int(key):04d}.txt").write_text(text, encoding="utf-8")

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    emb = dd / "embedded"
    emb.mkdir(exist_ok=True)
    cur.execute("SELECT page_seq, raw_text FROM pages WHERE document_id = %s ORDER BY page_seq", (doc_id,))
    for seq, txt in cur.fetchall():
        (emb / f"{seq:04d}.txt").write_text(txt or "", encoding="utf-8")
    conn.close()

    gz = dd / "gazetteer.txt"
    if not gz.exists():
        gz.symlink_to(GAZ)

    n = d["pages"]
    r = subprocess.run([PY, "scripts/adjudicate_transcript.py", "--base", str(dd), "--pages", f"1-{n}"],
                       capture_output=True, text=True)
    adj = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    if r.returncode != 0:
        log(f"  doc {doc_id}: adjudication FAILED: {adj}"); return False

    guard = d["source_name"][:30]
    pv = d.get("pipeline_version", "chunk_v1_silvermaster_structured_4k")
    r = subprocess.run(
        [PY, "scripts/load_transcript.py", "--doc-id", str(doc_id), "--base", str(dd),
         "--pages", f"1-{n}", "--expect-source", guard, "--pipeline-version", pv, "--dry-run"],
        capture_output=True, text=True, env=dict(os.environ),
    )
    m = re.search(r"Chunk mapping.*", r.stdout)
    mapline = m.group(0)[:180] if m else "no mapping line"
    # Gate: dry run succeeded, no chunks lacking transcript files, unmapped <=10%.
    # Oversized chunks (8000-byte index guard) keep their old text: count, don't fail.
    stats = re.search(
        r"(\d+)/(\d+) chunks mapped over .*? (\d+) skipped with\s+warnings.*?(\d+) unmapped",
        r.stdout, re.DOTALL)
    oversized = len(re.findall(r"SKIPPED — new clean_text", r.stdout))
    ok = False
    if r.returncode == 0 and stats:
        mapped, total, file_skipped, unmapped = (int(g) for g in stats.groups())
        ok = file_skipped == 0 and total > 0 and unmapped / total <= 0.10
        mapline += f" [oversized={oversized}]"
    with open(WORK / "batch_log.tsv", "a") as f:
        f.write(f"{doc_id}\t{d['pages']}\t{'OK' if ok else 'CHECK'}\t{adj}\t{mapline}\n")
    if ok:
        cmd = (f'"$PY" scripts/load_transcript.py --doc-id {doc_id} --base {dd} '
               f'--pages 1-{n} --expect-source "{guard}" --pipeline-version {pv} '
               f'--status machine_unverified --reset-embeddings '
               f'--method "vision-ensemble-free-v1 (olmOCR-2 bf16 + '
               f'Surya-2 + source OCR, majority-vote adjudication)"\n')
        with open(WORK / "loads_to_run.sh", "a") as f:
            f.write(cmd)
        log(f"  doc {doc_id}: OK — {adj}")
    else:
        log(f"  doc {doc_id}: NEEDS REVIEW — {mapline}")
    return True


processed = set()
if (WORK / "batch_log.tsv").exists():
    for line in open(WORK / "batch_log.tsv"):
        processed.add(int(line.split("\t")[0]))
log(f"{len(processed)} docs already post-processed")

deadline = time.time() + args.max_hours * 3600
while time.time() < deadline:
    rc, out = ssh_out(f"ls {args.remote_out}/*/DONE 2>/dev/null")
    if rc != 0 and not out.strip():
        rc2, complete = ssh_out(f"grep -c 'BATCH COMPLETE' /root/driver.log 2>/dev/null")
        log("no DONE markers yet")
    done_ids = {int(p.split("/")[-2]) for p in out.split() if p.strip()}
    todo = sorted(done_ids - processed)
    for doc_id in todo:
        if doc_id not in docs:
            continue
        log(f"processing doc {doc_id}")
        try:
            if process_doc(doc_id):
                processed.add(doc_id)
        except Exception as e:
            log(f"  doc {doc_id}: EXCEPTION {e}")
    if len(processed) >= len(docs):
        log("ALL DOCS POST-PROCESSED")
        break
    rc, out = ssh_out("grep -c 'BATCH COMPLETE' /root/driver.log 2>/dev/null || true")
    if out.strip() and out.strip() != "0" and len(processed) < len(docs):
        remaining = sorted(set(docs) - processed)
        log(f"driver finished but {len(remaining)} docs not processed: {remaining[:10]}...")
        break
    time.sleep(args.poll_seconds)

log(f"post-processing finished: {len(processed)}/{len(docs)} docs")
