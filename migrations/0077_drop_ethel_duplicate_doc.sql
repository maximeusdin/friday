-- 0077_drop_ethel_duplicate_doc.sql
-- Remove the duplicate document 'Rosenberg, Ethel HQ see references.pdf' (doc 1164)
-- from the rosenberg_ethel collection. Verified 2026-09-14: its 77 pages are a
-- page-for-page copy of 'ROSENBERG, ETHEL HQ SEE REF A.pdf' pages 103-179
-- (constant +102 offset; 69/75 substantive pages >=0.60 6-gram overlap, the rest
-- confirmed identical by hand -- the gap is Textract variance on the same scans).
-- Nothing is lost: every page survives in REF A.
--
-- Removes: 1 document, 77 pages, 37 chunks (chunk_pages / chunk_metadata and the
-- other chunk children cascade from chunks). Checked before writing: 0 rows in
-- retrieval_run_chunk_evidence (the one RESTRICT FK), result_set_chunks,
-- chunk_embeddings_canonical, entity_mentions; 0 chunks spanning another document.
--
-- Rollback: re-ingest the file -- it is kept locally at
--   data/raw/rosenberg_ethel/Rosenberg, Ethel HQ see references.pdf
--   ocr_cache/rosenberg_ethel/Rosenberg, Ethel HQ see references.pdf.json  (free, no re-OCR)
--
-- After this runs the S3 object must go and the zip must be rebuilt; both are in
-- scripts/drop_ethel_duplicate.sh, which applies this file first.

BEGIN;

CREATE TEMP TABLE _dup_doc ON COMMIT DROP AS
SELECT d.id
FROM documents d
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'rosenberg_ethel'
  AND d.source_name = 'Rosenberg, Ethel HQ see references.pdf';

-- Refuse to run unless the target is exactly one document.
DO $$
DECLARE n int;
BEGIN
  SELECT count(*) INTO n FROM _dup_doc;
  IF n <> 1 THEN
    RAISE EXCEPTION 'expected exactly 1 target document, found %', n;
  END IF;
END $$;

CREATE TEMP TABLE _dup_chunks ON COMMIT DROP AS
SELECT DISTINCT cp.chunk_id AS id
FROM chunk_pages cp
JOIN pages p ON p.id = cp.page_id
WHERE p.document_id IN (SELECT id FROM _dup_doc);

-- Refuse to run if any target chunk also covers a page of another document.
DO $$
DECLARE n int;
BEGIN
  SELECT count(*) INTO n
  FROM chunk_pages cp
  WHERE cp.chunk_id IN (SELECT id FROM _dup_chunks)
    AND cp.page_id NOT IN (SELECT id FROM pages WHERE document_id IN (SELECT id FROM _dup_doc));
  IF n <> 0 THEN
    RAISE EXCEPTION 'aborting: % chunk_pages rows reach outside the target document', n;
  END IF;
END $$;

DELETE FROM chunks WHERE id IN (SELECT id FROM _dup_chunks);
DELETE FROM pages WHERE document_id IN (SELECT id FROM _dup_doc);
DELETE FROM documents WHERE id IN (SELECT id FROM _dup_doc);

COMMIT;
