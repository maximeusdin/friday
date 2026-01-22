\echo '--- A) Message counts per source (should be non-zero) ---'
SELECT d.id AS document_id, d.source_name, d.volume, COUNT(p.id) AS n_messages
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'venona'
GROUP BY d.id, d.source_name, d.volume
ORDER BY d.source_name;

\echo '--- B) page_seq monotonic coverage (min=1, max=n, no gaps, no dups) ---'
WITH per_doc AS (
  SELECT d.id AS document_id, d.source_name,
         COUNT(*) AS n,
         MIN(p.page_seq) AS min_seq,
         MAX(p.page_seq) AS max_seq,
         COUNT(DISTINCT p.page_seq) AS distinct_seq
  FROM pages p
  JOIN documents d ON d.id = p.document_id
  JOIN collections c ON c.id = d.collection_id
  WHERE c.slug = 'venona'
  GROUP BY d.id, d.source_name
)
SELECT source_name, n, min_seq, max_seq, distinct_seq,
       (min_seq = 1) AS min_is_1,
       (max_seq = n) AS max_eq_n,
       (distinct_seq = n) AS no_dup_seq
FROM per_doc
ORDER BY source_name;

\echo '--- C) Duplicate page_seq within a source (should be 0 rows) ---'
SELECT d.source_name, p.page_seq, COUNT(*) AS ct
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'venona'
GROUP BY d.source_name, p.page_seq
HAVING COUNT(*) > 1
ORDER BY d.source_name, p.page_seq;

\echo '--- D) Duplicate logical_page_label within a source (should be 0 rows) ---'
SELECT d.source_name, p.logical_page_label, COUNT(*) AS ct
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'venona'
GROUP BY d.source_name, p.logical_page_label
HAVING COUNT(*) > 1
ORDER BY d.source_name, ct DESC;

\echo '--- E) pdf_page_number present (should be 0 nulls; min>=1) ---'
SELECT d.source_name,
       COUNT(*) FILTER (WHERE p.pdf_page_number IS NULL) AS null_pdf_page_number,
       MIN(p.pdf_page_number) AS min_pdf,
       MAX(p.pdf_page_number) AS max_pdf
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'venona'
GROUP BY d.source_name
ORDER BY d.source_name;

\echo '--- F) Metadata row coverage (should be ~= message rows) ---'
-- set this to your actual META_PIPELINE_VERSION:
\set meta_pipeline 'meta_v3_venona_headers'

SELECT d.source_name,
       COUNT(p.id) AS message_rows,
       COUNT(pm.page_id) FILTER (WHERE pm.pipeline_version = :'meta_pipeline') AS meta_rows
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
LEFT JOIN page_metadata pm ON pm.page_id = p.id
WHERE c.slug = 'venona'
GROUP BY d.source_name
ORDER BY d.source_name;

\echo '--- G) Key field fill rates (best-effort; expect missing but not all-missing) ---'
SELECT d.source_name,
       COUNT(*) AS n,
       COUNT(*) FILTER (WHERE (pm.meta_raw->'venona'->>'ussr_ref_no') IS NOT NULL) AS with_ussr_ref,
       COUNT(*) FILTER (WHERE (pm.meta_raw->'venona'->>'from') IS NOT NULL) AS with_from,
       COUNT(*) FILTER (WHERE jsonb_array_length(COALESCE(pm.meta_raw->'venona'->'to','[]'::jsonb)) > 0) AS with_to,
       COUNT(*) FILTER (WHERE (pm.meta_raw->'venona'->>'cable_no') IS NOT NULL) AS with_cable_no,
       COUNT(*) FILTER (WHERE (pm.meta_raw->'venona'->>'message_date_iso') IS NOT NULL) AS with_msg_date,
       COUNT(*) FILTER (WHERE (pm.meta_raw->'venona'->>'reissue_raw') IS NOT NULL) AS with_reissue,
       COUNT(*) FILTER (WHERE (pm.meta_raw->'venona'->>'issued_date_iso') IS NOT NULL) AS with_issued
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
LEFT JOIN page_metadata pm ON pm.page_id = p.id
WHERE c.slug = 'venona'
  AND pm.pipeline_version = :'meta_pipeline'
GROUP BY d.source_name
ORDER BY d.source_name;

\echo '--- H) Spot-check 10 messages (label + pdf start + a few metadata fields) ---'
SELECT d.source_name, p.page_seq, p.pdf_page_number, p.logical_page_label,
       pm.meta_raw->'venona'->>'ussr_ref_no' AS ussr_ref_no,
       pm.meta_raw->'venona'->>'from' AS from,
       pm.meta_raw->'venona'->'to' AS to,
       pm.meta_raw->'venona'->>'cable_no' AS cable_no,
       pm.meta_raw->'venona'->>'message_date_raw' AS msg_date_raw,
       pm.meta_raw->'venona'->>'message_date_iso' AS msg_date_iso,
       pm.meta_raw->'venona'->>'reissue_raw' AS reissue
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
LEFT JOIN page_metadata pm ON pm.page_id = p.id AND pm.pipeline_version = :'meta_pipeline'
WHERE c.slug = 'venona'
ORDER BY d.source_name, p.page_seq
LIMIT 10;

\echo '--- I) Find messages with multiple routing To destinations ---'
SELECT d.source_name, p.page_seq, pm.meta_raw->'venona'->'to' AS to_list
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
JOIN page_metadata pm ON pm.page_id = p.id
WHERE c.slug = 'venona'
  AND pm.pipeline_version = :'meta_pipeline'
  AND jsonb_array_length(COALESCE(pm.meta_raw->'venona'->'to','[]'::jsonb)) > 1
ORDER BY d.source_name, p.page_seq
LIMIT 25;
