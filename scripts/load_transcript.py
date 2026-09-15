#!/usr/bin/env python3
"""
Load an adjudicated vision-ensemble transcript into prod chunks.clean_text.

Normal mode (load):
    DATABASE_URL=... python scripts/load_transcript.py \
        --doc-id 522 --base data/transcripts/bentley_deposition --pages 3-120 \
        [--pipeline-version chunk_v1_silvermaster_structured_4k] \
        [--dry-run] [--status machine_unverified|reviewed] [--reset-embeddings]

  1. Verifies the document is the expected one (source_name must contain
     "Silvermaster Part 6") -- hard-fails otherwise.
  2. Maps chunks to pages by TEXT, not via chunk_pages. (For this document the
     chunk_pages linkage is coarse block-level garbage: most chunks link to the
     same 128-page block, so span-based mapping is unusable.) Ground truth:
     chunks of the given --pipeline-version pack consecutive WHOLE pages of the
     document in chunk-id order -- each chunk.text equals the concatenation of
     pages.raw_text for a contiguous run of pages. The mapper walks pages in
     page_seq order and, per chunk (id order), greedily consumes pages while
     the normalized concatenation of consumed pages is a prefix of the
     normalized chunk text (normalization: "".join(text.split())), stopping at
     equality. A chunk that never reaches equality is reported UNMAPPED with
     diagnostics, and the walk resyncs on the next chunk's head so one bad
     chunk cannot cascade. Pages consumed by no mapped chunk are reported.
  3. A chunk is LOADABLE iff every mapped page has page_seq within --pages and
     BASE/final/{page:04d}.txt exists for it; its new clean_text is
     "\\n\\n".join(final page texts in page order). A HYBRID straddler (some
     mapped pages covered by finals within --pages, the rest outside --pages)
     gets final text for the covered pages and pages.raw_text for the rest;
     hybrids are counted separately. A chunk with an in-range page whose final
     file is missing is skipped with a warning. Chunks with zero covered pages
     and UNMAPPED chunks are untouched.
  4. Backs up current clean_text of every chunk to be updated to
     BASE/backup_chunks_{docid}.json BEFORE updating (atomic tmp+rename write).
     An existing backup's entries are never modified (they hold the pristine
     pre-load state); before extending one with new chunk ids, it is first
     copied to a timestamped .bak sibling. At first creation the backup also
     captures the pristine documents.metadata['transcript'] value (or an
     explicit absence marker, plus whether metadata itself was NULL).
  5. UPDATE chunks SET clean_text = the new text. With --reset-embeddings also
     NULLs embedding/embedding_status so the standard embed script
     (--fill-missing-only) refreshes them; WITHOUT it, prints a warning with
     the count of affected chunks whose non-NULL embedding may no longer match
     the new clean_text.
  6. Stamps provenance under documents.metadata['transcript'].
  7. Prints verification counts (fuhr/meekirk probes) before AND after, all in
     one transaction. --dry-run rolls back instead of committing and, per
     updated chunk, prints a silent-loss audit: old clean_text tokens absent
     from BOTH the new transcript and the chunk raw text.

Restore mode:
    DATABASE_URL=... python scripts/load_transcript.py \
        --doc-id 522 --base data/transcripts/bentley_deposition \
        --restore data/transcripts/bentley_deposition/backup_chunks_522.json

  Restores clean_text from the backup JSON and reinstates EXACTLY the pristine
  documents.metadata['transcript'] captured in the backup (absence marker ->
  key removed; metadata NULL-ness preserved when nothing else remains). Legacy
  flat backups (no capture) fall back to removing the key. Prints the same
  verification counts.

This script performs writes inside a single transaction and is meant to be run
by the operator against prod. It never creates chunks/pages, only updates
clean_text and document metadata.
"""

import argparse
import json
import os
import shutil
import string
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import psycopg2

METHOD = (
    "vision-ensemble-v1 (gpt-5.5, gpt-5.2, gpt-4.1 + textract + source OCR, "
    "majority-vote adjudication)"
)
DEFAULT_EXPECTED_SOURCE = "Silvermaster Part 6"
COLLECTION_SLUG = "silvermaster"
DEFAULT_PIPELINE_VERSION = "chunk_v1_silvermaster_structured_4k"
RESYNC_HEAD_CHARS = 60
BACKUP_FORMAT = 2


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def die(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_pages(spec: str):
    """Parse 'A-B' into an inclusive (lo, hi) tuple of ints."""
    try:
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
    except ValueError:
        die(f"--pages must look like A-B (e.g. 3-120), got {spec!r}")
    if lo < 1 or hi < lo:
        die(f"--pages range invalid: {spec!r}")
    return lo, hi


# --------------------------------------------------------------------------
# Pure chunk<->page mapping (unit-testable, no DB)
# --------------------------------------------------------------------------

def normalize_text(text):
    """Whitespace- and hyphen-insensitive comparison form.

    Hyphens must be dropped because the structured_4k chunker joined
    hyphenated line breaks ("sm- wg" in pages.raw_text -> "smwg" in
    chunks.text), so page text is otherwise never a prefix of chunk text.
    """
    return "".join((text or "").split()).replace("-", "")


@dataclass
class MapResult:
    mapped: list          # [(chunk_id, [page_seq, ...]), ...] in chunk-id order
    unmapped: list        # [diagnostic dict, ...]
    leftover_pages: list  # [page_seq, ...] consumed by no mapped chunk


def _page_starts_head(norm_page: str, head: str) -> bool:
    """True iff the page's normalized text starts the given chunk head."""
    if not norm_page or not head:
        return False
    if len(norm_page) >= len(head):
        return norm_page.startswith(head)
    return head.startswith(norm_page)


def map_chunks_to_pages(pages, chunks):
    """
    Map chunks to the contiguous run of pages whose concatenated raw_text
    equals the chunk text (whitespace-insensitive).

    pages:  ordered list of (page_seq, raw_text), in page_seq order
    chunks: ordered list of (chunk_id, chunk_text), in chunk-id order

    Walks pages with a single forward pointer. For each chunk, greedily
    consumes pages while the normalized concatenation of consumed pages is a
    prefix of the normalized chunk text; the chunk is mapped when the two are
    equal. If equality is never reached (page fails the prefix test, or pages
    run out), the chunk is recorded as UNMAPPED with a diagnostic, its
    tentatively consumed pages are released, and the pointer resyncs by
    scanning forward for the page that starts the NEXT chunk's normalized head
    (first RESYNC_HEAD_CHARS chars) so one bad chunk cannot cascade.

    Returns MapResult(mapped, unmapped, leftover_pages); leftover_pages are
    the page_seqs consumed by no mapped chunk.
    """
    norm_pages = [(seq, normalize_text(raw)) for seq, raw in pages]
    n = len(norm_pages)
    mapped = []
    unmapped = []
    consumed = set()  # indices into norm_pages
    ptr = 0

    for i, (chunk_id, chunk_text) in enumerate(chunks):
        target = normalize_text(chunk_text)
        start_ptr = ptr
        acc_len = 0
        seqs = []
        failure = None  # (fail_ptr, fail_seq, reason)
        # The structured_4k chunker dropped junk-only runs (dot leaders,
        # pure-garble lines) that ARE present in pages.raw_text, so a page is
        # consumed with a two-pointer walk that may skip page-side characters
        # as junk, within a per-chunk deletion budget.
        budget = max(64, len(target) // 5)
        deletions = 0

        while acc_len < len(target):
            if ptr >= n:
                failure = (ptr, None,
                           "ran out of pages before chunk text was complete")
                break
            seq, ptext = norm_pages[ptr]
            j = acc_len
            dele = deletions
            for ch in ptext:
                if j < len(target) and ch == target[j]:
                    j += 1
                else:
                    dele += 1
                    if dele > budget:
                        break
            if dele > budget:
                failure = (ptr, seq,
                           f"page_seq {seq} is not the next piece of the chunk "
                           f"text (junk-deletion budget {budget} exceeded)")
                break
            acc_len, deletions = j, dele
            seqs.append(seq)
            ptr += 1

        if failure is None:
            # acc_len == len(target): chunk fully accounted for.
            for k in range(start_ptr, ptr):
                consumed.add(k)
            mapped.append((chunk_id, seqs))
            continue

        # UNMAPPED: release the tentative consumption, record diagnostics.
        fail_ptr, fail_seq, reason = failure
        unmapped.append({
            "chunk_id": chunk_id,
            "reason": reason,
            "expected_head": target[:RESYNC_HEAD_CHARS],
            "expected_tail": target[-RESYNC_HEAD_CHARS:],
            "page_pointer": fail_ptr,
            "page_seq_at_pointer": fail_seq,
            "tentative_page_seqs": seqs,
        })
        ptr = start_ptr
        # RESYNC: find where the next chunk begins so this failure can't
        # cascade. Pages skipped over become leftovers.
        if i + 1 < len(chunks):
            next_head = normalize_text(chunks[i + 1][1])[:RESYNC_HEAD_CHARS]
            for j in range(ptr, n):
                if _page_starts_head(norm_pages[j][1], next_head):
                    ptr = j
                    break
            # If no page matches the next chunk's head, ptr stays put and the
            # next chunk gets its own attempt (and its own diagnostic).

    leftover = [seq for k, (seq, _) in enumerate(norm_pages) if k not in consumed]
    return MapResult(mapped=mapped, unmapped=unmapped, leftover_pages=leftover)




def map_chunks_to_pages_overlapping(pages, chunks, head_len=64, tail_len=64):
    """
    Mapping strategy for OVERLAPPING chunk pipelines (e.g. rosenberg_v1):
    chunks are sliding windows over the page stream, so each chunk is located
    INDEPENDENTLY: find its normalized head in the concatenated normalized
    page stream, then its tail at/after that point; the chunk covers every
    page overlapping [head_pos, tail_end). Chunks whose head or tail cannot
    be found are UNMAPPED. leftover_pages are pages covered by no chunk.
    """
    offsets = []  # (start_offset, page_seq, norm_len)
    stream_parts = []
    off = 0
    for seq, raw in pages:
        n = normalize_text(raw)
        offsets.append((off, seq, len(n)))
        stream_parts.append(n)
        off += len(n)
    stream = "".join(stream_parts)

    def pages_in_span(a, b):
        hit = [seq for start, seq, ln in offsets
               if ln > 0 and start + ln > a and start < b]
        if not hit:
            return []
        # Pages whose OLD ocr was blank contribute no characters to the stream,
        # so they can never overlap a span -- yet those are exactly the pages
        # re-transcription rescues (a blank page often yields a full page of new
        # text). Return the contiguous range so interior blank pages come along.
        return [seq for _, seq, _ in offsets if hit[0] <= seq <= hit[-1]]

    mapped, unmapped = [], []
    covered = set()
    search_from = 0
    for chunk_id, chunk_text in chunks:
        target = normalize_text(chunk_text)
        head = target[:head_len]
        tail = target[-tail_len:]
        if not head:
            unmapped.append({"chunk_id": chunk_id, "reason": "empty after normalize"})
            continue
        # Overlapping windows advance monotonically; search near the previous
        # hit first, fall back to a global search. A single corrupt character
        # at the chunk's start would defeat a head-only anchor (the chunk text
        # and the page text are independently-OCR'd copies), so fall back to
        # anchors taken from further into the chunk and back-project the start.
        near = max(0, search_from - 2 * len(target))
        pos, used_off = -1, 0
        for frac in (0.0, 0.1, 0.25, 0.5, 0.75):
            off = int(len(target) * frac)
            anchor = target[off:off + head_len]
            if len(anchor) < 24:
                continue
            a = stream.find(anchor, near)
            if a < 0:
                a = stream.find(anchor)
            if a >= 0:
                pos, used_off = max(0, a - off), off
                break
        if pos < 0:
            unmapped.append({"chunk_id": chunk_id, "reason": "head not found",
                             "expected_head": head})
            continue
        if used_off:
            # Back-projected anchor: the chunk's start was inferred, not found,
            # so confirm the chunk really lives here before claiming the span.
            # Writing a wrong page's transcript into a chunk is worse than
            # leaving the chunk on its old OCR.
            probes = [target[int(len(target) * f):int(len(target) * f) + 48]
                      for f in (0.05, 0.3, 0.55, 0.8)]
            probes = [q for q in probes if len(q) >= 24]
            window = stream[max(0, pos - 200):pos + len(target) + 200]
            hits = sum(1 for q in probes if q in window)
            if probes and hits / len(probes) < 0.5:
                unmapped.append({"chunk_id": chunk_id,
                                 "reason": f"anchor back-projection unconfirmed "
                                           f"({hits}/{len(probes)} probes)",
                                 "expected_head": head})
                continue
        tpos = stream.find(tail, pos)
        end = (tpos + len(tail)) if tpos >= 0 else min(len(stream), pos + len(target))
        seqs = pages_in_span(pos, end)
        if not seqs:
            unmapped.append({"chunk_id": chunk_id, "reason": "no pages in span"})
            continue
        mapped.append((chunk_id, seqs))
        covered.update(seqs)
        search_from = pos

    # Blank-OCR pages at a span edge can still fall between two chunks and end
    # up in none, which would drop their rescued transcript from search. Attach
    # each to the nearest span (preferring the one that ends just before it).
    all_seqs = [seq for _, seq, _ in offsets]
    blanks = {seq for _, seq, ln in offsets if ln == 0}
    for seq in sorted(blanks - covered):
        best_i, best_dist = None, None
        for i, (cid, seqs) in enumerate(mapped):
            dist = min(abs(seq - s) for s in seqs)
            # prefer the preceding chunk on ties: the page reads as its tail
            if best_dist is None or dist < best_dist or (dist == best_dist and max(seqs) < seq):
                best_i, best_dist = i, dist
        if best_i is not None:
            cid, seqs = mapped[best_i]
            mapped[best_i] = (cid, sorted(set(seqs) | {seq}))
            covered.add(seq)

    leftover = [seq for _, seq, ln in offsets if seq not in covered and ln > 0]
    return MapResult(mapped=mapped, unmapped=unmapped, leftover_pages=leftover)


def map_chunks_best(pages, chunks):
    """Run both strategies and keep whichever maps more chunks.

    Both are fast (well under a second even for 300-page docs), and an
    unmapped chunk silently keeps its old OCR text, so there is no reason to
    gate the second attempt behind a failure threshold: sequential routinely
    loses 10-15% on overlapping-window pipelines that overlapping maps whole.
    Ties keep sequential, whose page spans are contiguous by construction.
    Returns (result, strategy).
    """
    if not chunks:
        return map_chunks_to_pages(pages, chunks), "sequential"
    seq_res = map_chunks_to_pages(pages, chunks)
    if not seq_res.unmapped:
        return seq_res, "sequential"
    ov_res = map_chunks_to_pages_overlapping(pages, chunks)
    if len(ov_res.mapped) > len(seq_res.mapped):
        return ov_res, "overlapping"
    return seq_res, "sequential"


# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------

def fetch_document(cur, doc_id: int, expected_source: str = DEFAULT_EXPECTED_SOURCE):
    cur.execute(
        "SELECT id, collection_id, source_name, metadata FROM documents WHERE id = %s",
        (doc_id,),
    )
    row = cur.fetchone()
    if row is None:
        die(f"document id={doc_id} not found")
    _, collection_id, source_name, metadata = row
    if expected_source not in (source_name or ""):
        die(
            f"document id={doc_id} source_name={source_name!r} does not contain "
            f"{expected_source!r} -- refusing to touch it"
        )
    print(f"Document {doc_id}: {source_name} (collection_id={collection_id})")
    return collection_id, source_name, metadata


def fetch_pages(cur, doc_id: int):
    """Ordered [(page_seq, raw_text), ...] for the document."""
    cur.execute(
        "SELECT page_seq, raw_text FROM pages WHERE document_id = %s ORDER BY page_seq",
        (doc_id,),
    )
    return cur.fetchall()


def fetch_chunks(cur, doc_id: int, pipeline_version: str):
    """Ordered [(chunk_id, chunk_text), ...] for the document via chunk_metadata."""
    cur.execute(
        """
        SELECT DISTINCT c.id, c.text
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.document_id = %s
        WHERE c.pipeline_version = %s
        ORDER BY c.id
        """,
        (doc_id, pipeline_version),
    )
    return cur.fetchall()


def fetch_doc_chunk_ids(cur, doc_id: int):
    """Set of chunk ids linked to the document via chunk_metadata."""
    cur.execute(
        "SELECT DISTINCT chunk_id FROM chunk_metadata WHERE document_id = %s",
        (doc_id,),
    )
    return {r[0] for r in cur.fetchall()}


def run_verification(cur, doc_id: int, label: str):
    print(f"\n--- Verification ({label}) ---")
    for probe in ("%fuhr%", "%meekirk%"):
        cur.execute(
            """
            SELECT COUNT(DISTINCT p.id)
            FROM pages p
            JOIN chunk_pages cp ON cp.page_id = p.id
            JOIN chunks c ON c.id = cp.chunk_id
            WHERE p.document_id = %s
              AND COALESCE(c.clean_text, c.text) ILIKE %s
            """,
            (doc_id, probe),
        )
        n = cur.fetchone()[0]
        print(f"  doc {doc_id} pages whose chunk matches ILIKE {probe!r}: {n}")
    cur.execute(
        """
        SELECT COUNT(DISTINCT c.id)
        FROM chunks c
        JOIN chunk_pages cp ON cp.chunk_id = c.id
        JOIN pages p ON p.id = cp.page_id
        JOIN documents d ON d.id = p.document_id
        JOIN collections col ON col.id = d.collection_id
        WHERE col.slug = %s
          AND COALESCE(c.clean_text, c.text) ILIKE %s
        """,
        (COLLECTION_SLUG, "%fuhr%"),
    )
    n = cur.fetchone()[0]
    print(f"  collection {COLLECTION_SLUG!r} chunks matching ILIKE '%fuhr%': {n}")


def fetch_clean_text(cur, chunk_ids):
    cur.execute(
        "SELECT id, clean_text FROM chunks WHERE id = ANY(%s)",
        (list(chunk_ids),),
    )
    return {cid: clean for cid, clean in cur.fetchall()}


def warn_unreset_embeddings(cur, chunk_ids, reset_embeddings: bool):
    """Review fix 3: without --reset-embeddings, flag stale vectors."""
    if reset_embeddings or not chunk_ids:
        return
    cur.execute(
        "SELECT COUNT(*) FROM chunks WHERE id = ANY(%s) AND embedding IS NOT NULL",
        (list(chunk_ids),),
    )
    n = cur.fetchone()[0]
    if n:
        print(f"\nWARNING: --reset-embeddings not set; {n} of {len(chunk_ids)} "
              f"affected chunk(s) have a non-NULL embedding whose vector may no "
              f"longer match the updated clean_text. Rerun with --reset-embeddings "
              f"(or re-embed) if semantic search must stay consistent.")


# --------------------------------------------------------------------------
# Backup handling (review fixes 1 & 2)
# --------------------------------------------------------------------------

def _atomic_write_json(path: Path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_backup_file(path: Path):
    """
    Return (chunks_map, meta_capture_or_None). Handles both the v2 format
    ({"format": 2, "chunks": {...}, "document_metadata_transcript": {...}})
    and the legacy flat {chunk_id: clean_text} format (meta capture -> None).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        die(f"backup {path} is unreadable ({e})")
    if isinstance(data, dict) and isinstance(data.get("chunks"), dict):
        return data["chunks"], data.get("document_metadata_transcript")
    if isinstance(data, dict):
        return data, None  # legacy flat format
    die(f"backup {path} has an unrecognized structure")


def write_backup(backup_path: Path, update_ids, current_clean, doc_id: int,
                 doc_metadata):
    """
    Persist {chunk_id: pristine clean_text} plus, at FIRST creation only, the
    pristine documents.metadata['transcript'] value (or an explicit absence
    marker + whether metadata itself was NULL). Entries already present in an
    existing backup are never modified; only missing chunk ids are appended,
    and the existing file is first copied to a timestamped .bak sibling.
    All writes are atomic (tmp file + os.replace).
    """
    if backup_path.exists():
        chunks_map, meta_capture = _read_backup_file(backup_path)
        missing = [cid for cid in update_ids if str(cid) not in chunks_map]
        if not missing:
            print(f"Backup {backup_path} already exists and covers all "
                  f"{len(update_ids)} chunks to update -- reusing it, not overwriting.")
            return
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        bak = backup_path.with_name(f"{backup_path.name}.{ts}.bak")
        shutil.copy2(backup_path, bak)
        print(f"Copied existing backup to {bak} before extending it.")
        n_existing = len(chunks_map)
        for cid in missing:
            chunks_map[str(cid)] = current_clean.get(cid)
        doc = {"format": BACKUP_FORMAT, "document_id": doc_id, "chunks": chunks_map}
        if meta_capture is not None:
            doc["document_metadata_transcript"] = meta_capture
        # A backup without a capture (legacy) never recorded the pristine
        # transcript value; the current DB value is no longer pristine, so it
        # deliberately stays uncaptured.
        _atomic_write_json(backup_path, doc)
        print(f"Backup {backup_path} extended: kept its {n_existing} existing "
              f"entries untouched, appended {len(missing)} new chunk(s).")
    else:
        meta_capture = {
            "present": isinstance(doc_metadata, dict) and "transcript" in doc_metadata,
            "metadata_was_null": doc_metadata is None,
        }
        if meta_capture["present"]:
            meta_capture["value"] = doc_metadata["transcript"]
        doc = {
            "format": BACKUP_FORMAT,
            "document_id": doc_id,
            "document_metadata_transcript": meta_capture,
            "chunks": {str(cid): current_clean.get(cid) for cid in update_ids},
        }
        _atomic_write_json(backup_path, doc)
        print(f"Backup written: {backup_path} ({len(update_ids)} chunks; pristine "
              f"metadata transcript captured: present={meta_capture['present']}, "
              f"metadata_was_null={meta_capture['metadata_was_null']})")


def restore_document_metadata(cur, doc_id: int, meta_capture):
    """Reinstate exactly the pristine metadata['transcript'] state (fix 2)."""
    if meta_capture is None:
        # Legacy backup: pristine value unknown; old behavior (remove key).
        cur.execute(
            "UPDATE documents SET metadata = metadata - 'transcript' WHERE id = %s",
            (doc_id,),
        )
        print(f"Removed documents.metadata['transcript'] for document {doc_id} "
              f"(legacy backup: pristine value was not captured)")
        return
    if meta_capture.get("present"):
        cur.execute(
            """
            UPDATE documents
            SET metadata = COALESCE(metadata, '{}'::jsonb)
                           || jsonb_build_object('transcript', %s::jsonb)
            WHERE id = %s
            """,
            (json.dumps(meta_capture.get("value")), doc_id),
        )
        print(f"Reinstated pristine documents.metadata['transcript'] for "
              f"document {doc_id}: {json.dumps(meta_capture.get('value'))}")
    else:
        was_null = bool(meta_capture.get("metadata_was_null"))
        cur.execute(
            """
            UPDATE documents
            SET metadata = CASE
                WHEN %s AND (metadata - 'transcript') = '{}'::jsonb THEN NULL
                ELSE metadata - 'transcript'
            END
            WHERE id = %s
            """,
            (was_null, doc_id),
        )
        print(f"Removed documents.metadata['transcript'] for document {doc_id} "
              f"(absent at backup time"
              + ("; metadata reset to NULL as it originally was, since nothing "
                 "else remains)" if was_null else ")"))


# --------------------------------------------------------------------------
# Provenance / audit
# --------------------------------------------------------------------------

def stamp_provenance(cur, doc_id: int, doc_metadata, status: str, pages_spec: str,
                     method: str = METHOD):
    today = date.today().isoformat()
    transcript = {
        "status": status,
        "method": method,
        "pages": pages_spec,
        "date": today,
        "reviewed_by": None,
    }
    if status == "reviewed":
        existing = (doc_metadata or {}).get("transcript")
        if isinstance(existing, dict):
            # Keep existing fields (method, pages, reviewed_by, ...); only
            # bump status and date.
            transcript = dict(existing)
            transcript["status"] = "reviewed"
            transcript["date"] = today
    cur.execute(
        """
        UPDATE documents
        SET metadata = COALESCE(metadata, '{}'::jsonb)
                       || jsonb_build_object('transcript', %s::jsonb)
        WHERE id = %s
        """,
        (json.dumps(transcript), doc_id),
    )
    print(f"Provenance stamped on document {doc_id}: {json.dumps(transcript)}")


def tokenize_for_audit(text):
    """Normalized token set for the silent-loss audit (fix 4)."""
    toks = set()
    for t in (text or "").split():
        t = t.strip(string.punctuation).lower()
        if t:
            toks.add(t)
    return toks


def print_silent_loss_audit(old_clean, new_text, chunk_raw_text):
    """Old clean_text tokens findable in NEITHER new transcript NOR raw text."""
    lost = tokenize_for_audit(old_clean) - (
        tokenize_for_audit(new_text) | tokenize_for_audit(chunk_raw_text)
    )
    if lost:
        samples = sorted(lost)[:5]
        print(f"    silent-loss audit: {len(lost)} token(s) from old clean_text "
              f"would lose findability (absent from new transcript AND raw text); "
              f"samples: {samples}")
    else:
        print("    silent-loss audit: no tokens lose findability")


# --------------------------------------------------------------------------
# Load / restore
# --------------------------------------------------------------------------

def build_new_text(seqs, lo, hi, final_dir: Path, raw_by_seq):
    """Final text for in-range pages, pages.raw_text for out-of-range ones."""
    parts = []
    for seq in seqs:
        if lo <= seq <= hi:
            text = (final_dir / f"{seq:04d}.txt").read_text(encoding="utf-8")
        else:
            text = raw_by_seq.get(seq) or ""
        parts.append(text.rstrip("\n"))
    return "\n\n".join(parts)


def do_load(args, conn):
    base = Path(args.base)
    final_dir = base / "final"
    lo, hi = parse_pages(args.pages)

    with conn.cursor() as cur:
        _, _, doc_metadata = fetch_document(cur, args.doc_id, args.expect_source)

        run_verification(cur, args.doc_id, "before")

        pages_rows = fetch_pages(cur, args.doc_id)
        if not pages_rows:
            die(f"document {args.doc_id} has no pages")
        chunk_rows = fetch_chunks(cur, args.doc_id, args.pipeline_version)
        if not chunk_rows:
            die(f"no chunks with pipeline_version={args.pipeline_version!r} are "
                f"linked to document {args.doc_id} via chunk_metadata")
        chunk_raw = {cid: text for cid, text in chunk_rows}
        raw_by_seq = {seq: raw for seq, raw in pages_rows}

        result, strategy = map_chunks_best(pages_rows, chunk_rows)
        print(f"  (mapping strategy: {strategy})")

        for diag in result.unmapped:
            # Diag dicts differ by mapper: sequential carries pointer context,
            # overlapping only reason + expected head/tail. Print what exists.
            extras = ", ".join(f"{k}={diag[k]!r}" for k in
                               ("page_pointer", "page_seq_at_pointer",
                                "tentative_page_seqs", "expected_head",
                                "expected_tail") if k in diag)
            print(f"  WARNING: UNMAPPED chunk {diag['chunk_id']}: "
                  f"{diag.get('reason', '?')} ({extras})")
        if result.leftover_pages:
            print(f"  WARNING: {len(result.leftover_pages)} page(s) consumed by no "
                  f"mapped chunk (page_seq): {result.leftover_pages}")

        # Classify mapped chunks against --pages coverage + final availability.
        loadable = {}   # chunk_id -> [page_seq, ...] (all pages covered)
        hybrid = {}     # chunk_id -> [page_seq, ...] (straddles --pages)
        skipped = []    # (chunk_id, reason)
        untouched = 0   # zero covered pages
        for chunk_id, seqs in result.mapped:
            in_range = [s for s in seqs if lo <= s <= hi]
            out_range = [s for s in seqs if not (lo <= s <= hi)]
            if not in_range:
                untouched += 1
                continue
            missing = [s for s in in_range
                       if not (final_dir / f"{s:04d}.txt").exists()]
            if missing:
                skipped.append(
                    (chunk_id, "missing final transcript(s): "
                               + ", ".join(f"{final_dir}/{s:04d}.txt" for s in missing))
                )
                continue
            if out_range:
                hybrid[chunk_id] = seqs
            else:
                loadable[chunk_id] = seqs

        print(f"\nChunk mapping (text prefix-walk, "
              f"pipeline_version={args.pipeline_version}): "
              f"{len(result.mapped)}/{len(chunk_rows)} chunks mapped over "
              f"{len(pages_rows)} pages -- {len(loadable)} loadable, "
              f"{len(hybrid)} hybrid straddlers, {len(skipped)} skipped with "
              f"warnings, {untouched} untouched (zero pages covered by --pages "
              f"{lo}-{hi}), {len(result.unmapped)} unmapped, "
              f"{len(result.leftover_pages)} leftover pages.")
        for chunk_id, reason in skipped:
            print(f"  WARNING: skipping chunk {chunk_id}: {reason}")

        update_plan = {**loadable, **hybrid}
        if not update_plan:
            die("no loadable or hybrid chunks -- nothing to do")

        # Backup FIRST (pristine pre-load state), before any UPDATE.
        current_clean = fetch_clean_text(cur, update_plan.keys())
        backup_path = base / f"backup_chunks_{args.doc_id}.json"
        write_backup(backup_path, sorted(update_plan), current_clean,
                     args.doc_id, doc_metadata)

        warn_unreset_embeddings(cur, sorted(update_plan), args.reset_embeddings)

        # Apply updates.
        print(f"\nPlanned UPDATEs ({len(update_plan)} chunks: "
              f"{len(loadable)} loadable, {len(hybrid)} hybrid):")
        if args.reset_embeddings:
            sql = ("UPDATE chunks SET clean_text = %s, embedding = NULL, "
                   "embedding_status = NULL WHERE id = %s")
        else:
            sql = "UPDATE chunks SET clean_text = %s WHERE id = %s"
        for chunk_id in sorted(update_plan):
            seqs = update_plan[chunk_id]
            new_text = build_new_text(seqs, lo, hi, final_dir, raw_by_seq)
            old = current_clean.get(chunk_id)
            old_len = len(old) if old is not None else 0
            tag = ""
            if chunk_id in hybrid:
                raw_seqs = [s for s in seqs if not (lo <= s <= hi)]
                tag = f" [hybrid: raw_text for pages {raw_seqs}]"
            print(f"  chunk {chunk_id}{tag}: pages {seqs} old_len={old_len} "
                  f"new_len={len(new_text)}")
            if args.dry_run:
                print_silent_loss_audit(old, new_text, chunk_raw.get(chunk_id))
            if len(new_text.encode("utf-8")) > 8000:
                # gist_trgm index rows cap at 8191 bytes; an oversized chunk
                # keeps its old clean_text rather than aborting the whole load.
                print(f"  chunk {chunk_id}: SKIPPED — new clean_text "
                      f"{len(new_text.encode('utf-8'))} bytes exceeds the 8000-byte "
                      f"index-safe limit (keeps existing text)")
                continue
            cur.execute(sql, (new_text, chunk_id))
        if args.reset_embeddings:
            print("  (embedding + embedding_status set to NULL for the above; run the "
                  "standard embed script with --fill-missing-only to refresh)")

        stamp_provenance(cur, args.doc_id, doc_metadata, args.status, args.pages, args.method)

        run_verification(cur, args.doc_id, "after")

    if args.dry_run:
        conn.rollback()
        print("\n[DRY RUN] rolled back -- no changes committed.")
    else:
        conn.commit()
        print(f"\nCommitted: {len(update_plan)} chunks updated for document "
              f"{args.doc_id}.")


def do_restore(args, conn):
    backup_path = Path(args.restore)
    if not backup_path.exists():
        die(f"backup file not found: {backup_path}")
    chunks_map, meta_capture = _read_backup_file(backup_path)
    if not chunks_map:
        die(f"backup {backup_path} contains no chunks")

    with conn.cursor() as cur:
        fetch_document(cur, args.doc_id, args.expect_source)

        run_verification(cur, args.doc_id, "before")

        # Safety: only restore chunks that actually belong to this document.
        doc_chunk_ids = fetch_doc_chunk_ids(cur, args.doc_id)
        restorable = []
        for cid_str in sorted(chunks_map, key=lambda s: int(s)):
            cid = int(cid_str)
            if cid not in doc_chunk_ids:
                print(f"  WARNING: backup chunk {cid} is not linked to document "
                      f"{args.doc_id} via chunk_metadata -- skipping")
                continue
            restorable.append(cid)
        if not restorable:
            die("no restorable chunks found in backup")

        current_clean = fetch_clean_text(cur, restorable)

        warn_unreset_embeddings(cur, restorable, args.reset_embeddings)

        print(f"\nRestoring clean_text for {len(restorable)} chunks from {backup_path}:")
        if args.reset_embeddings:
            sql = ("UPDATE chunks SET clean_text = %s, embedding = NULL, "
                   "embedding_status = NULL WHERE id = %s")
        else:
            sql = "UPDATE chunks SET clean_text = %s WHERE id = %s"
        for cid in restorable:
            old = current_clean.get(cid)
            new = chunks_map[str(cid)]
            old_len = len(old) if old is not None else 0
            new_len = len(new) if new is not None else 0
            print(f"  chunk {cid}: old_len={old_len} restored_len={new_len}"
                  + ("" if new is not None else " (NULL)"))
            cur.execute(sql, (new, cid))

        restore_document_metadata(cur, args.doc_id, meta_capture)

        run_verification(cur, args.doc_id, "after")

    if args.dry_run:
        conn.rollback()
        print("\n[DRY RUN] rolled back -- no changes committed.")
    else:
        conn.commit()
        print(f"\nCommitted: {len(restorable)} chunks restored for document {args.doc_id}.")


def main():
    ap = argparse.ArgumentParser(
        description="Load adjudicated transcript into chunks.clean_text (or restore a backup)."
    )
    ap.add_argument("--doc-id", type=int, required=True, help="documents.id (expected: 522)")
    ap.add_argument("--base", required=True,
                    help="Transcript base dir, e.g. data/transcripts/bentley_deposition")
    ap.add_argument("--pages", default=None,
                    help="Inclusive page_seq range A-B, e.g. 3-120 (required unless --restore)")
    ap.add_argument("--pipeline-version", default=DEFAULT_PIPELINE_VERSION,
                    help="chunks.pipeline_version of the chunk set to map/update "
                         f"(default: {DEFAULT_PIPELINE_VERSION})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do everything inside the transaction (incl. the silent-loss "
                         "audit), then roll back")
    ap.add_argument("--method", default=METHOD,
                    help="provenance method string stamped into documents.metadata.transcript")
    ap.add_argument("--expect-source", default=DEFAULT_EXPECTED_SOURCE,
                    help="safety guard: document source_name must contain this substring")
    ap.add_argument("--status", choices=["machine_unverified", "reviewed"],
                    default="machine_unverified",
                    help="Provenance status to stamp (default: machine_unverified)")
    ap.add_argument("--restore", default=None, metavar="BACKUP.json",
                    help="Restore mode: restore clean_text from this backup JSON and "
                         "reinstate the pristine transcript metadata captured in it")
    ap.add_argument("--reset-embeddings", action="store_true",
                    help="Also NULL embedding/embedding_status on updated chunks so the "
                         "standard embed script --fill-missing-only refreshes them")
    args = ap.parse_args()

    if not args.restore and not args.pages:
        die("--pages A-B is required in load mode")

    conn = get_conn()
    try:
        if args.restore:
            do_restore(args, conn)
        else:
            do_load(args, conn)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
