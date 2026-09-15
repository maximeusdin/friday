# Duplicate documents across the corpus — 2026-09-15

Follow-up to the `elizabeth_bentley` finding of 2026-09-14. The same near-duplicate
measurement was run across all 44 collections, then extended, because the original key
turned out to under-report badly. Everything below is read-only measurement against
prod; nothing has been written.

## TL;DR

* **`solo` is the real duplication problem, not `elizabeth_bentley`.** 22 FBI SOLO
  serial ranges were ingested twice — once with zero-padded serial numbers, once
  without — on 2026-02-01/02. **3,669 redundant pages**, versus Bentley's ~700.
* **Bentley cannot be fixed by deleting anything.** The omnibus and the per-volume
  PDFs overlap ~79% *in both directions*. Dropping the omnibus loses 415 pages that
  exist nowhere else; dropping the volumes loses 416. It needs search-time dedup.
* **26 documents are safe to drop** (3,972 pages, 1,320 chunks) — migration `0078`.
* **Two source-data bugs found that a delete does not fix** — see "Not duplication".

## 1. Why the original key under-reports

The 2026-09-14 key — md5 of the first 400 alphanumeric-lowercase characters of
`pages.raw_text` — is an *exact* key. One OCR character difference in that prefix
breaks it. Reproducing the Bentley baseline pinned the floor used: pages need ≥200
normalized characters to count as text-bearing (3933 text-bearing, 703 shared, 17.9%
— exactly the reported numbers).

Two things then had to be corrected:

**It over-reports on boilerplate.** FBI releases repeat standard forms verbatim. Of
`winton_burdett`'s 153 "duplicate" pages, 137 are the FOIPA *DELETED PAGE INFORMATION
SHEET*. `thomas_black` and `david_greenglass` are the "BEST COPIES OBTAINABLE" notice;
`siss_scope_soviet` is a Senate committee masthead. None of that is document
duplication. `scripts/measure_page_duplication.py` now suppresses a key that repeats
inside one document or appears in ≥4 documents of a collection.

**It under-reports far more seriously on OCR variance.** Two independent scans of the
same file never produce the same 400-character prefix. `joel_barr` 972/978 shows 22
shared pages by exact key and 81 by 6-gram containment. Bentley doc 1082 shows 55 by
exact key (28.8% of its pages) and 187 (97.9%) once OCR variance is tolerated.

So the corpus sweep uses a bottom-16 minhash over word 6-grams
(`scripts/sweep_duplicate_documents.py`), and candidates are then verified with the
6-gram containment measure migration 0077 used.

## 2. Corpus-wide result

Exact-key figures for context (`scripts/measure_page_duplication.py`, 130,280
text-bearing pages). "raw dup" is the 2026-09-14 key; "distinct" excludes boilerplate:

| collection | docs | text pages | raw dup | distinct dup | % |
|---|---|---|---|---|---|
| elizabeth_bentley | 24 | 3,933 | 703 | 579 | 14.7% |
| harry_gold | 105 | 13,482 | 630 | 275 | 2.0% |
| rosenberg | 188 | 10,284 | 297 | 234 | 2.3% |
| winton_burdett | 6 | 1,063 | 153 | **6** | 0.6% |
| solo | 124 | 22,533 | 115 | 115 | 0.5% |
| siss_scope_soviet | 7 | 6,492 | 82 | **6** | 0.1% |
| joel_barr | 11 | 1,293 | 61 | 48 | 3.7% |
| *(20 more ≤48 pages)* | | | | | |
| **total** | | **130,280** | **2,256** | **1,346** | **1.0%** |

Note `solo` at 0.5% by the exact key. The minhash sweep puts it at **3,669 redundant
pages** — the exact key missed almost all of it, because the two solo ingests were
OCR'd separately.

## 3. Confirmed duplicates — drop these (migration 0078)

### solo: the zero-padded / unpadded double ingest — 23 documents, 3,714 pages

22 pairs, every one with **identical page counts**, matching **page-for-page at offset
0**. Mean 6-gram Jaccard on the diagonal runs 90–18,000× the off-diagonal baseline
(`scripts/check_aligned_duplicates.py`); the four weakest pairs were confirmed by
reading the page text side by side. These are the same scans OCR'd twice.

Which copy to keep was decided on **legible characters recovered** = raw characters ×
share of alphabetic tokens that are real words. (`chunks.alpha_ratio` and
`chunks.garbage_score` exist but are `0` for every row in prod, so they carry no
signal — `scripts/rank_duplicate_keep_drop.py` computes its own.) The unpadded ingest
wins 19 pairs by 1.9–8.9%; the padded one wins 3 (docs 75, 108, 114) by 1.4–5.9%.

The 23rd is doc 66, `EBF1405`, which sits complete inside `Serial1320-1395` at a
constant +196 page offset (100% of its 44 text pages).

### rosenberg: doc 188 `Rosenberg, Julius 48_text.pdf` — 138 pages

Page text is **byte-identical** to `47_text.pdf` (diagonal Jaccard 1.000 on all 138
pages), and both S3 objects are exactly 10,114,490 bytes. The "48" file is a copy of
"47". ⚠️ **The real volume 48 is therefore missing from the archive** — the 40–49
series is otherwise complete. Dropping doc 188 removes a document that was never
volume 48; re-sourcing the real one is separate work.

### harry_gold: doc 1173 `Gold- Harry-HQ-65-57449-13_Part2.pdf` — 116 pages

Same story: diagonal Jaccard 1.000 against `_Part1.pdf` on all 116 pages. The local
files are both 8,147,056 bytes with different md5s — two PDF renderings of the same
pages. ⚠️ **Ingested 2026-09-15, i.e. today**, along with 104 other harry_gold
documents, so the real Part 2 may still be to hand. Note `65-57449-11` is also absent
between docs 1170 and 1171.

### fbi_hiskey: doc 587 — 4 pages

A 4-page Army CIC report on Clarence Hiskey (`SPJIW 201`), re-downloaded on a later
pass (filename prefixes `20220318_` vs `20220408_`) and already present inside doc 632
at offset +12.

**Total: 26 documents, 3,972 pages, 1,320 chunks.**

## 4. Real overlap, but NOT safe to delete

### elizabeth_bentley — the omnibus is not redundant

`scripts/check_omnibus_coverage.py --hub 1061 --collection elizabeth_bentley`:

* **Forward** — 1,565/1,981 (79.0%) of the other 23 documents' text pages are inside
  doc 1061. Individually they run 100% (docs 1065, 1073, 1078, 1088) down to 21%.
* **Reverse** — 1,537/1,952 (78.7%) of doc 1061's text pages are reproduced somewhere
  in the other 23.
* **415 pages of doc 1061 appear nowhere else.** Of those, ~118 score below 0.30
  against every other page, so they are genuinely unique content, not OCR drift.

The overlap is symmetric: dropping the omnibus loses 415 pages, dropping the volumes
loses 416. **Neither side can go.** Bentley is two different FOIA releases of an
overlapping file series, and the fix is search-time deduplication — collapsing hits
whose page text is near-identical across documents — not a `DELETE`. That is the same
change that fixes the `"pleurisy"` example.

(Docs 1065, 1073, 1078 and 1088 *are* individually 100% contained in the omnibus and
could technically be dropped. I would not: they carry proper FBI file/serial names
— `NY-134-182-SUB-A-1-SER.1&2` — and are the better-OCR'd copies. Losing that
provenance to save 39 pages is a bad trade.)

### joel_barr, oscar_seborer — 80–83%, needs a human

`972` (VOL.01) / `978` (VOL.03): 81/101 of VOL.03 is inside VOL.01 at offset +68, but
20 pages are not. `972`/`975` (VOL.02): 26/32 at offset +28. The volume split looks
wrong, but a delete would lose pages. `oscar_seborer` 866/868 is 5/6 both ways.

### harry_gold HQ / PH / NY cross-filing — leave alone

Below ~65% the sweep is dominated by pairs like `HQ-65-57449-30` vs `PH-65-4307-1B-18`
(117 shared pages). That is the FBI filing the same memo in the Headquarters,
Philadelphia and New York case files. Both copies are legitimate archival objects with
distinct provenance; deduplicating them would destroy information. Same for the
`solo` serial-range boundary overlaps (376/377 at +65).

## 5. Not duplication — two source-data bugs

**`rosenberg_trial_transcripts` 672/673.** The sweep flagged `Rosenberg Ethel.pdf`
(30pp) as containing all of `Rosenberg Julius.pdf` (18pp). It does — because the Ethel
PDF was **built wrong**. Its pages 1–13 are Ethel's testimony; from page 14 it becomes
Julius's, and page 1 of *both* files carries the same source URL
(`famous-trials.com/rosenberg/2220-juliustest`). Deleting doc 673 — which the
automatic containment rule recommends — would leave Julius's testimony findable only
inside a document labelled "Rosenberg Ethel". **The fix is to rebuild doc 672 as
Ethel's 13 pages**, not to drop 673. Not in the migration.

**Missing volumes.** Both `Rosenberg, Julius 48` and `Gold- Harry-HQ-65-57449-13
Part 2` are copies of their predecessors, so those two volumes are absent from the
archive rather than duplicated.

## 6. What the delete has to clean up

`scripts/preflight_drop_documents.py` discovers dependants by scanning
`information_schema` rather than a hand-written list, and classifies each by its FK
delete rule. For the 26 targets:

| table | rows | rule |
|---|---|---|
| `chunk_metadata` (chunk_id) | 1,320 | CASCADE from `chunks` |
| `chunk_pages` | 4,375 | CASCADE from `chunks` |
| `chunk_embeddings_canonical` | 1,258 | CASCADE from `chunks` |
| `entity_mentions` | 912 | CASCADE from `documents` |
| `date_mentions` | 68 | CASCADE from `documents` |
| `page_entity_mentions` | 0 | CASCADE (only vassiliev/venona are populated) |
| **`search_result_page_hits`** | **28,521** | **NO FK — orphans** |
| **`evidence_items`** | **95** | **NO FK — orphans** |

Three things worth flagging:

1. **`chunk_metadata.document_id`, `first_page_id` and `last_page_id` are `NO ACTION`,
   not `CASCADE`.** They block a document delete. Deleting `chunks` first clears them
   via the `chunk_id` cascade — which is why 0077's order (chunks → pages → documents)
   matters and is kept.
2. **`search_result_page_hits` has no FK to `documents`/`pages`/`chunks`** — only to
   `search_result_sets`. 28,521 rows across 72 saved searches would be left pointing at
   ids that no longer exist. Deleting them is also what stops the duplicate hits from
   persisting in already-saved result sets.
3. **`evidence_items` — 95 rows — is the one 0077 did not have to handle** (its target
   had none). These are saved research-session evidence quotes. They have no FK either.
   Deleting them silently changes saved sessions; the alternative is leaving 95
   dangling `chunk_id`s. The migration deletes them — flagging it because it is a
   judgement call about user data, not derived data.

`retrieval_run_chunk_evidence` (the one `ON DELETE RESTRICT` chunk reference) has
**0 rows** for these chunks, as do `result_set_chunks`, `result_set_match_traces`,
`focus_spans` and `document_witnesses`. No chunk spans a page outside the target set.

## 7. Running it

```bash
bash scripts/drop_duplicate_ingests.sh
```

Order: read-only dry run → migration `0078` → collection zip rebuild. The S3 objects
are **not** deleted unless you pass `--s3`, because `data/raw/solo` is empty locally
and `ocr_cache/solo` holds nothing — S3 is the only rollback path for the 23 solo
documents, and restoring one would otherwise mean paying for OCR again.

Verify first, on its own — it writes nothing:

```bash
DATABASE_URL="$(aws secretsmanager get-secret-value --region us-west-1 --secret-id friday/DATABASE_URL --query SecretString --output text)" python scripts/dryrun_0078.py
```

Expected after: solo 124 → 101, rosenberg 188 → 187, harry_gold 105 → 104,
fbi_hiskey 63 → 62.

## 8. Scripts added

| script | what it does |
|---|---|
| `measure_page_duplication.py` | the 2026-09-14 exact key, corpus-wide, with boilerplate suppression |
| `sweep_duplicate_documents.py` | minhash sweep for duplicate documents, tolerant of OCR variance |
| `verify_duplicate_pairs.py` | 6-gram containment for a pair, or a hub vs its collection |
| `check_omnibus_coverage.py` | two-way coverage: hub vs the union of the rest |
| `check_aligned_duplicates.py` | same-scan test by page alignment (diagonal vs off-diagonal) |
| `compare_duplicate_candidates.py` | per-pair containment both ways + a keep/drop verdict |
| `rank_duplicate_keep_drop.py` | which copy to keep, on legible characters recovered |
| `preflight_drop_documents.py` | dependants of a delete, by FK rule; flags NO-FK orphans |
| `dryrun_0078.py` | read-only rehearsal of every guard in migration 0078 |

All read-only.
