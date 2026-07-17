#!/usr/bin/env python3
"""Quantify content overlap between the new SOLO section-scan release (data/raw/
fbi_solo) and the existing serial-organized `solo` collection.

Both are FBI file 100-HQ-428091. The existing collection encodes each document's
serial range in its filename; the new release is organized by FBI *section*. We
sample-OCR a handful of pages from every new section file, read the stamped
serial numbers (100-428091-NNNN), and compare serial coverage. That tells us how
much of the new release duplicates the existing collection vs. adds new serials
(notably the existing gap ~1989-3075 and anything past 7114)."""
import os, sys, re, json, glob
from concurrent.futures import ProcessPoolExecutor
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import fitz, ocr_textract
import psycopg2, boto3

R = r"C:\Users\maxim\friday"
DIR = os.path.join(R, "data", "raw", "fbi_solo")
EXCLUDE = {
    "ALBERTSON, William - HQ 65-38100.pdf", "Childs, Eva-HQ-1.pdf",
    "Solo Abstracts from FBI FOIA File 66.pdf", "Solo doc file list.PDF",
    "baumgardner-memo.pdf", "Pages from CPUSA-TOPLEV-HQ-68.pdf",
    "FBI Childs Morris Pages from Childs_Morris-Chicago-4.pdf",
    "Solo 1.pdf", "solo 2.pdf", "1174205-0 - Volume 6 (1).pdf",
}
SERIAL_RE = re.compile(r"428091[\s\-_]{0,4}(\d{1,5})")
SAMPLE = 14

def sample_one(path):
    d = fitz.open(path); n = d.page_count; d.close()
    idxs = sorted(set(int(i*(n-1)/(SAMPLE-1)) for i in range(SAMPLE))) if n > SAMPLE else list(range(n))
    pages = ocr_textract.ocr_pdf_parallel(path, region="us-east-1", dpi=300, max_workers=4,
                                          page_indices=idxs, progress=None)
    text = "\n".join(pages)
    serials = [int(m) for m in SERIAL_RE.findall(text) if 1 <= int(m) <= 9999]
    return (os.path.basename(path), n, serials)

def cluster_span(serials):
    """Drop lone cross-references; return (lo, hi) of the dense serial cluster."""
    s = sorted(set(serials))
    if not s:
        return None
    kept = [x for i, x in enumerate(s)
            if (i > 0 and x - s[i-1] <= 300) or (i < len(s)-1 and s[i+1] - x <= 300)]
    kept = kept or s
    return (min(kept), max(kept))

def existing_coverage():
    sm = boto3.client("secretsmanager", region_name="us-west-1")
    arn = "arn:aws:secretsmanager:us-west-1:682405977227:secret:friday/DATABASE_URL-8KlDQC"
    v = sm.get_secret_value(SecretId=arn)["SecretString"]
    url = json.loads(v).get("DATABASE_URL", v) if v.strip().startswith("{") else v
    conn = psycopg2.connect(url); cur = conn.cursor()
    cur.execute("""SELECT d.source_name FROM documents d JOIN collections c ON c.id=d.collection_id
                   WHERE c.slug='solo'""")
    names = [r[0] for r in cur.fetchall()]; conn.close()
    cov = set(); rng = re.compile(r"Serial0*(\d+)-0*(\d+)")
    for nm in names:
        m = rng.search(nm)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            cov.update(range(a, b+1))
    return cov, names

if __name__ == "__main__":
    files = [f for f in sorted(glob.glob(os.path.join(DIR, "*.pdf")))
             if os.path.basename(f) not in EXCLUDE]
    print(f"sampling {len(files)} section files ({SAMPLE} pages each)...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(sample_one, files):
            results.append(r)
            sp = cluster_span(r[2])
            print(f"  {r[0][:50]:50} {r[1]:>4}p  serial_span={sp}", flush=True)
    json.dump(results, open(os.path.join(R, "ocr_cache", "solo_overlap_raw.json"), "w", encoding="utf-8"))

    existing, ex_names = existing_coverage()
    new_cov = set(); per_file = []
    for name, n, serials in results:
        sp = cluster_span(serials)
        if sp:
            band = set(range(sp[0], sp[1]+1)); new_cov |= band
            in_ex = len(band & existing) / max(1, len(band))
            per_file.append((name, sp, round(in_ex, 2)))
        else:
            per_file.append((name, None, None))

    overlap = new_cov & existing
    new_only = new_cov - existing
    gap = set(range(1989, 3076))          # the hole in the existing collection
    beyond = set(range(7115, 10000))      # past existing's last serial
    print("\n" + "="*70)
    print(f"existing 'solo' serial coverage : {len(existing)} serials (max {max(existing)})")
    print(f"new release serial coverage     : {len(new_cov)} serials (max {max(new_cov) if new_cov else 0})")
    print(f"  overlap (in both)             : {len(overlap)}")
    print(f"  NEW serials (not in existing) : {len(new_only)}")
    print(f"  -> fills existing gap 1989-3075: {len(new_cov & gap)} of 1087")
    print(f"  -> extends beyond serial 7114  : {len(new_cov & beyond)}")
    dup_files = [p for p in per_file if p[2] is not None and p[2] >= 0.9]
    new_files = [p for p in per_file if p[2] is not None and p[2] < 0.5]
    print(f"\nfiles ~fully duplicate (>=90% serials already in existing): {len(dup_files)}")
    print(f"files mostly NEW (<50% serials in existing): {len(new_files)}")
    for p in new_files:
        print(f"   NEW {p[0][:54]:54} span={p[1]} in_existing={p[2]}")
    print("OVERLAP ANALYSIS COMPLETE", flush=True)
