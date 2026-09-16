-- 0079_drop_duplicate_ingests.sql
--
-- APPLIED TO PROD 2026-09-15. Verified after: solo 124->101, rosenberg 188->187,
-- harry_gold 105->104, fbi_hiskey 63->62; pages 151,468->147,496; 0 leftover pages
-- and 0 orphaned search_result_page_hits rows for the dropped documents.
-- Written and run as 0078; renumbered to 0079 because 0078_pages_tsv_simple.sql
-- (the page-attribution fix retrieval/search_executor.py refers to by number)
-- had already claimed 0078. Both ran against prod on the same day.
-- Remove 26 documents that are redundant re-ingests of a document already in the
-- same collection. Measured 2026-09-15 (scripts/sweep_duplicate_documents.py,
-- check_aligned_duplicates.py, rank_duplicate_keep_drop.py). Follows 0077.
--
--   solo        23 docs  22 of them are the same FBI SOLO serial ranges ingested
--                        twice on 2026-02-01/02, once with zero-padded serial
--                        numbers and once without. Each pair has identical page
--                        counts and matches page-for-page at offset 0; mean 6-gram
--                        Jaccard on the diagonal is 90-18000x the off-diagonal
--                        baseline, so they are the same scans OCR'd twice. The copy
--                        kept is the one with more legible characters recovered
--                        (raw chars x share of alphabetic tokens that are real
--                        words) -- the unpadded ingest in 19 pairs, the padded one
--                        in 3. The 23rd (doc 66) is EBF1405, which sits complete
--                        inside Serial1320-1395 at a constant +196 page offset.
--   rosenberg    1 doc   'Rosenberg, Julius 48_text.pdf' has byte-identical page
--                        text to '47_text.pdf' (diagonal Jaccard 1.000 on all 138
--                        pages) and both S3 objects are 10,114,490 bytes: the 48
--                        file is a copy of 47. See the note below -- the real
--                        volume 48 is missing from the archive.
--   harry_gold   1 doc   'Gold- Harry-HQ-65-57449-13_Part2.pdf' is likewise a copy
--                        of _Part1.pdf (diagonal Jaccard 1.000 on all 116 pages).
--                        Ingested 2026-09-15. The real Part 2 is missing.
--   fbi_hiskey   1 doc   a 4-page Army CIC report on Clarence Hiskey re-downloaded
--                        on a later pass; it sits inside doc 632 at offset +12.
--
-- Removes: 26 documents, 3972 pages, 1320 chunks. chunk_metadata, chunk_pages and
-- chunk_embeddings_canonical (1258 rows) cascade from chunks; date_mentions (68),
-- entity_mentions (912) and page_entity_mentions (0 rows -- only vassiliev and
-- venona are populated) cascade from documents. Checked before writing: 0 rows in
-- retrieval_run_chunk_evidence (the one RESTRICT FK), result_set_chunks,
-- result_set_match_traces, focus_spans and document_witnesses; 0 chunks spanning a
-- page outside the target set.
--
-- Two tables carry document_id / page_id / chunk_id with NO foreign key, so nothing
-- cascades and their rows would be left dangling. They are deleted explicitly:
--   search_result_page_hits  28521 rows across 72 saved searches
--   evidence_items              95 rows of saved research-session evidence
-- These are saved user artefacts, not derived data. 0077 did not have to deal with
-- either (its target had none). Removing them is what stops the duplicate hits this
-- change is about from coming back in already-saved result sets.
--
-- Rollback: the dropped PDFs all still exist in S3 under s3://fridayarchive.org/<source_ref>,
-- so re-ingest is possible -- but data/raw/solo is empty locally and ocr_cache/solo
-- holds nothing, so restoring a solo document means paying for OCR again. Delete the
-- S3 objects only after you are satisfied (scripts/drop_duplicate_ingests.sh does the
-- DB first and prompts before touching S3).
--
-- NOT included, deliberately -- see docs/DUPLICATE_DOCUMENTS_2026-09-15.md:
--   elizabeth_bentley 1061 vs its per-volume PDFs. The overlap is real but partial in
--     BOTH directions: dropping the omnibus loses 415 pages that appear nowhere else,
--     dropping the volumes loses 416. Neither side is safe to delete; that collection
--     needs search-time deduplication, not a DELETE.
--   rosenberg_trial_transcripts 672/673. 'Rosenberg Ethel.pdf' contains all of
--     'Rosenberg Julius.pdf', but because the Ethel PDF was built wrong (13 pages of
--     Ethel's testimony followed by 17 of Julius's, both scraped from the same
--     famous-trials.com URL). The fix is to rebuild 672, not to delete 673.
--   joel_barr 972/975/978 and oscar_seborer 866/868: 80-83% overlap, so a delete
--     would lose pages. Manual review.

BEGIN;

CREATE TEMP TABLE _dup_pair (drop_id bigint PRIMARY KEY, keep_id bigint NOT NULL) ON COMMIT DROP;
INSERT INTO _dup_pair (drop_id, keep_id) VALUES
  (  57,   70),  -- solo       100-HQ-428091-EBF0099_text.pdf        -> EBF99
  (  66,  177),  -- solo       100-HQ-428091-EBF1405_text.pdf        -> Serial1320-1395 pp.197-241
  (  80,  275),  -- solo       Serial0045-0069                       -> Serial45-69
  (  83,  388),  -- solo       Serial0070-0076                       -> Serial70-76
  (  86,  392),  -- solo       Serial0077-0162                       -> Serial77-162
  (  90,   75),  -- solo       Serial01-44                           -> Serial0001-0044 (padded wins here)
  (  93,  195),  -- solo       Serial0163-0206                       -> Serial163-206
  (  96,  219),  -- solo       Serial0206-0228                       -> Serial206-228
  (  98,  220),  -- solo       Serial0229-0316                       -> Serial229-316
  ( 109,  230),  -- solo       Serial0321-0431                       -> Serial321-431
  ( 112,  267),  -- solo       Serial0432-0509                       -> Serial432-509
  ( 115,  336),  -- solo       Serial0518-0585                       -> Serial518-585
  ( 120,  370),  -- solo       Serial0586-0599                       -> Serial586-599
  ( 122,  373),  -- solo       Serial0601-0711                       -> Serial601-711
  ( 125,  390),  -- solo       Serial0712-0725                       -> Serial712-725
  ( 126,  391),  -- solo       Serial0726-0828                       -> Serial726-828
  ( 132,  393),  -- solo       Serial0829-0907                       -> Serial829-907
  ( 136,  394),  -- solo       Serial0909-0958                       -> Serial909-958
  ( 142,  395),  -- solo       Serial0958-0997                       -> Serial958-997
  ( 144,  396),  -- solo       Serial0998-1065                       -> Serial998-1065
  ( 225,  108),  -- solo       Serial317-320                         -> Serial0317-0320 (padded wins here)
  ( 326,  114),  -- solo       Serial509-514                         -> Serial0509-0514 (padded wins here)
  ( 397,  399),  -- solo       SOLO-045_text.pdf                     -> SOLO-45_text.pdf
  ( 188,  186),  -- rosenberg  Rosenberg, Julius 48_text.pdf         -> 47_text.pdf
  (1173, 1172),  -- harry_gold Gold- Harry-HQ-65-57449-13_Part2.pdf  -> _Part1.pdf
  ( 587,  632);  -- fbi_hiskey 20220318_2, Previous investigation... -> 20220408_August 26, 1940.pdf

-- Refuse to run unless every target and every keeper is present exactly once.
DO $$
DECLARE n int;
BEGIN
  SELECT count(*) INTO n FROM _dup_pair;
  IF n <> 26 THEN
    RAISE EXCEPTION 'expected 26 pairs, found %', n;
  END IF;
  SELECT count(*) INTO n FROM _dup_pair p WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = p.drop_id);
  IF n <> 0 THEN
    RAISE EXCEPTION 'aborting: % target documents are already gone', n;
  END IF;
  SELECT count(*) INTO n FROM _dup_pair p WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = p.keep_id);
  IF n <> 0 THEN
    RAISE EXCEPTION 'aborting: % keeper documents are missing -- do not drop their duplicates', n;
  END IF;
END $$;

-- Refuse to run if a keeper is in a different collection, or is smaller than the
-- document being dropped for it: either means the pairing above is wrong.
DO $$
DECLARE n int;
BEGIN
  SELECT count(*) INTO n
  FROM _dup_pair p
  JOIN documents dd ON dd.id = p.drop_id
  JOIN documents dk ON dk.id = p.keep_id
  WHERE dk.collection_id <> dd.collection_id;
  IF n <> 0 THEN
    RAISE EXCEPTION 'aborting: % pairs straddle two collections', n;
  END IF;

  SELECT count(*) INTO n
  FROM _dup_pair p
  WHERE (SELECT count(*) FROM pages WHERE document_id = p.keep_id)
      < (SELECT count(*) FROM pages WHERE document_id = p.drop_id);
  IF n <> 0 THEN
    RAISE EXCEPTION 'aborting: % keepers have fewer pages than the document dropped for them', n;
  END IF;
END $$;

CREATE TEMP TABLE _dup_pages ON COMMIT DROP AS
SELECT id FROM pages WHERE document_id IN (SELECT drop_id FROM _dup_pair);

CREATE TEMP TABLE _dup_chunks ON COMMIT DROP AS
SELECT DISTINCT cp.chunk_id AS id
FROM chunk_pages cp
WHERE cp.page_id IN (SELECT id FROM _dup_pages);

-- Refuse to run if any target chunk also covers a page of a document we are keeping.
DO $$
DECLARE n int;
BEGIN
  SELECT count(*) INTO n
  FROM chunk_pages cp
  WHERE cp.chunk_id IN (SELECT id FROM _dup_chunks)
    AND cp.page_id NOT IN (SELECT id FROM _dup_pages);
  IF n <> 0 THEN
    RAISE EXCEPTION 'aborting: % chunk_pages rows reach outside the target documents', n;
  END IF;
END $$;

-- retrieval_run_chunk_evidence is ON DELETE RESTRICT: it would block the delete.
DO $$
DECLARE n int;
BEGIN
  SELECT count(*) INTO n FROM retrieval_run_chunk_evidence
  WHERE chunk_id IN (SELECT id FROM _dup_chunks);
  IF n <> 0 THEN
    RAISE EXCEPTION 'aborting: % retrieval_run_chunk_evidence rows cite these chunks', n;
  END IF;
END $$;

-- No FK, so nothing cascades: delete before the rows they point at disappear.
DELETE FROM search_result_page_hits WHERE document_id IN (SELECT drop_id FROM _dup_pair);
DELETE FROM evidence_items          WHERE chunk_id    IN (SELECT id FROM _dup_chunks);

-- chunk_metadata, chunk_pages, chunk_embeddings_canonical and the other chunk
-- children cascade from chunks; page_entity_mentions, page_metadata, entity_mentions
-- and date_mentions cascade from pages/documents.
DELETE FROM chunks    WHERE id          IN (SELECT id FROM _dup_chunks);
DELETE FROM pages     WHERE document_id IN (SELECT drop_id FROM _dup_pair);
DELETE FROM documents WHERE id          IN (SELECT drop_id FROM _dup_pair);

COMMIT;
