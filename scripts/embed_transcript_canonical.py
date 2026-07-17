#!/usr/bin/env python3
"""Speaker-aware canonical embeddings (Index B) for transcript collections.

Builds chunk_embeddings_canonical.text_canonical = an attribution header
([TESTIMONY]/[SPEAKERS] block, derived from document_witnesses for grand juries
or the document/role for the trial) + the chunk text, then embeds it. Retrieval
ranks by COALESCE(cec.embedding, c.embedding), so once these rows exist the
speaker-enriched embedding replaces the plain one for vector search.

Display/citations are unchanged (they use chunks.text / the PDF).

Run:  export DATABASE_URL=<prod>
      python -m scripts.embed_transcript_canonical --collection-slug rosenberg_grand_jury --rebuild
"""
from __future__ import annotations
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.embed_venona_chunks import embed_texts, get_conn, vector_literal, truncate_text
from scripts.embed_canonical_chunks import upsert_canonical

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

PROCEEDING = {
    "rosenberg_grand_jury": "Rosenberg grand jury (SDNY, 1950-1951)",
    "brothman_moskowitz_grand_jury": "Brothman-Moskowitz grand jury (SDNY, 1947, 1950)",
    "rosenberg_trial_transcripts": "United States v. Rosenberg trial (SDNY, 1951)",
}
GRAND_JURY = {"rosenberg_grand_jury", "brothman_moskowitz_grand_jury"}

# Trial: filename stem -> (full name, role). Reorder "Last First" -> "First Last".
TRIAL_PEOPLE = {
    "Bentley Elizabeth": ("Elizabeth Bentley", "witness"),
    "Bloch Emanuel": ("Emanuel H. Bloch", "defense counsel"),
    "Cox Evelyn": ("Evelyn Cox", "witness"),
    "Elitcher Max": ("Max Elitcher", "witness"),
    "Gold Harry": ("Harry Gold", "witness"),
    "Greenglass David": ("David Greenglass", "witness"),
    "Greenglass Ruth": ("Ruth Greenglass", "witness"),
    "Greenglass drawings": ("Greenglass sketches", "exhibit"),
    "Kaufman Irving": ("Judge Irving R. Kaufman", "court"),
    "Koski Walter": ("Walter Koski", "witness"),
    "Rosenberg Ethel": ("Ethel Rosenberg", "defendant"),
    "Rosenberg Julius": ("Julius Rosenberg", "defendant"),
    "Saypol Irving": ("Irving Saypol", "prosecution"),
    "Schneider Ben": ("Ben Schneider", "witness"),
}
TRIAL_SPEAKERS_LEGEND = (
    "[SPEAKERS] THE COURT => Judge Irving R. Kaufman; MR. SAYPOL => Irving Saypol (prosecution); "
    "MR. COHN => Roy Cohn (prosecution); MR. E. H. BLOCH / MR. BLOCH => Emanuel Bloch (defense) [/SPEAKERS]"
)


def nice(name: str) -> str:
    """Title-case an ALL-CAPS record name while preserving initials and DeX caps."""
    out = []
    for w in name.split():
        if len(w) <= 2 and w.endswith("."):
            out.append(w.upper())          # initial, e.g. "Z." / "MR." handled below
        elif w.isupper():
            out.append(w[0] + w[1:].lower())
        else:
            out.append(w)                   # already mixed, e.g. DeBUFF
    return " ".join(out)


def fetch_pipeline_version(cur, slug: str) -> str:
    cur.execute("SELECT DISTINCT pipeline_version FROM chunk_metadata WHERE collection_slug=%s", (slug,))
    rows = [r[0] for r in cur.fetchall()]
    if len(rows) != 1:
        raise SystemExit(f"{slug}: expected 1 pipeline_version, got {rows}")
    return rows[0]


def fetch_chunks(cur, slug: str, pv: str):
    """(chunk_id, text, document_id, min_page, max_page) with chunk page span."""
    cur.execute("""
        SELECT c.id, c.text, p.document_id, MIN(p.pdf_page_number), MAX(p.pdf_page_number)
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id=c.id
        JOIN chunk_pages cp ON cp.chunk_id=c.id
        JOIN pages p ON p.id=cp.page_id
        WHERE cm.collection_slug=%s AND cm.pipeline_version=%s
        GROUP BY c.id, c.text, p.document_id
        ORDER BY c.id
    """, (slug, pv))
    return cur.fetchall()


def fetch_witnesses(cur, doc_ids: List[int]) -> Dict[int, List[tuple]]:
    out: Dict[int, List[tuple]] = {}
    if not doc_ids:
        return out
    cur.execute("""
        SELECT document_id, witness_name, start_page, end_page, testimony_date, examiner
        FROM document_witnesses WHERE document_id = ANY(%s) ORDER BY appearance_seq
    """, (doc_ids,))
    for did, name, sp, ep, date, exam in cur.fetchall():
        out.setdefault(did, []).append((name, sp, ep, date, exam))
    return out


def fetch_doc_names(cur, doc_ids: List[int]) -> Dict[int, str]:
    cur.execute("SELECT id, source_name FROM documents WHERE id = ANY(%s)", (doc_ids,))
    return {r[0]: r[1] for r in cur.fetchall()}


def gj_header(slug, witnesses, minpg, maxpg) -> Tuple[str, dict]:
    # find appearances overlapping the chunk's page span, ranked by overlap
    hits = []
    for name, sp, ep, date, exam in witnesses:
        ov = min(ep, maxpg) - max(sp, minpg) + 1
        if ov > 0:
            hits.append((ov, name, date, exam, sp, ep))
    hits.sort(reverse=True)
    proc = PROCEEDING[slug]
    if not hits:
        return f'[TESTIMONY proceeding="{proc}" pages={minpg}-{maxpg}]', {"witness": None}
    _, name, date, exam, sp, ep = hits[0]
    parts = [f'proceeding="{proc}"', f'witness="{nice(name)}"']
    if exam:
        parts.append(f'examiner="{nice(exam)}"')
    if date:
        parts.append(f'date="{date}"')
    parts.append(f'pages={minpg}-{maxpg}')
    also = [nice(h[1]) for h in hits[1:]]
    if also:
        parts.append(f'also={";".join(also)}')
    hdr = "[TESTIMONY " + " ".join(parts) + "]"
    return hdr, {"witness": nice(name), "date": date, "examiner": nice(exam) if exam else None,
                 "also": also, "source": "witness_index"}


def trial_header(slug, source_name) -> Tuple[str, dict]:
    proc = PROCEEDING[slug]
    stem = re.sub(r"\.pdf$", "", source_name or "", flags=re.I).strip()
    if stem in TRIAL_PEOPLE:
        person, role = TRIAL_PEOPLE[stem]
        hdr = f'[TESTIMONY proceeding="{proc}" speaker="{person}" role={role}]\n{TRIAL_SPEAKERS_LEGEND}'
        return hdr, {"speaker": person, "role": role, "source": "trial_doc"}
    # page scans (p6.pdf etc.) — no single speaker
    hdr = f'[TRIAL_PAGE proceeding="{proc}" page={stem}]\n{TRIAL_SPEAKERS_LEGEND}'
    return hdr, {"speaker": None, "page": stem, "source": "trial_page"}


def main():
    ap = argparse.ArgumentParser(description="Speaker-aware canonical embeddings for transcripts")
    ap.add_argument("--collection-slug", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rebuild", action="store_true", help="Delete existing canonical rows for this collection first")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    slug = args.collection_slug
    if slug not in PROCEEDING:
        raise SystemExit(f"Unknown transcript collection {slug}; known: {list(PROCEEDING)}")

    conn = get_conn(); conn.autocommit = False
    cur = conn.cursor()
    pv = fetch_pipeline_version(cur, slug)
    rows = fetch_chunks(cur, slug, pv)
    print(f"{slug}: pipeline={pv}, {len(rows)} chunks")

    doc_ids = sorted({r[2] for r in rows})
    witnesses = fetch_witnesses(cur, doc_ids) if slug in GRAND_JURY else {}
    doc_names = fetch_doc_names(cur, doc_ids) if slug not in GRAND_JURY else {}

    if args.rebuild and not args.dry_run:
        cur.execute("""
            DELETE FROM chunk_embeddings_canonical cec USING chunk_metadata cm
            WHERE cec.chunk_id=cm.chunk_id AND cm.collection_slug=%s
              AND cec.pipeline_version=%s AND cec.embedding_model=%s
        """, (slug, pv, EMBED_MODEL))
        conn.commit()

    # Build canonical texts
    items = []  # (chunk_id, text_canonical, manifest)
    for cid, text, did, minpg, maxpg in rows:
        if slug in GRAND_JURY:
            hdr, man = gj_header(slug, witnesses.get(did, []), minpg or 0, maxpg or 0)
        else:
            hdr, man = trial_header(slug, doc_names.get(did, ""))
        text_canonical = f"{hdr}\n\n{(text or '').rstrip()}"
        items.append((cid, text_canonical, man))

    if args.dry_run:
        for cid, tc, man in items[:5]:
            print(f"\n--- chunk {cid} ---\n{tc[:380]}")
        print(f"\n[dry-run] {len(items)} chunks would be embedded")
        return

    total = len(items)
    for i in range(0, total, args.batch_size):
        batch = items[i:i + args.batch_size]
        vecs = embed_texts([truncate_text(tc)[0] for _, tc, _ in batch], verbose=False)
        for (cid, tc, man), v in zip(batch, vecs):
            upsert_canonical(cur, cid, pv, EMBED_MODEL, tc, vector_literal(v), [man])
        conn.commit()
        print(f"  embedded {min(i+args.batch_size, total)}/{total}")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
