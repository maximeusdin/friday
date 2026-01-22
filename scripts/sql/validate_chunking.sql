\echo '--- Chunking validation ---'
\echo 'Set these variables before running:'
\echo '  \\set pipeline_version ''chunk_v1_day3'''
\echo '  \\set collection_slug ''vassiliev''  (or venona)'
\echo ''

-- Example usage:
-- \set pipeline_version 'chunk_v1_full'
-- \set collection_slug 'vassiliev'

\echo 'Pipeline:' :pipeline_version ' Collection:' :collection_slug
\echo ''

-- 1) Documents + page counts (canonical units)
\echo '1) Page/message counts per document'
SELECT
  d.id AS document_id,
  d.source_name,
  d.volume,
  COUNT(p.id) AS n_pages
FROM documents d
JOIN collections c ON c.id = d.collection_id
JOIN pages p ON p.document_id = d.id
WHERE c.slug = :'collection_slug'
GROUP BY d.id, d.source_name, d.volume
ORDER BY d.id;

\echo ''
-- 2) Chunks per document (for this pipeline)
\echo '2) Chunk counts per document for pipeline'
SELECT
  d.id AS document_id,
  d.source_name,
  COUNT(DISTINCT ch.id) AS n_chunks,
  COUNT(cp.*) AS n_chunk_page_links
FROM documents d
JOIN collections c ON c.id = d.collection_id
JOIN pages p ON p.document_id = d.id
LEFT JOIN chunk_pages cp ON cp.page_id = p.id
LEFT JOIN chunks ch ON ch.id = cp.chunk_id AND ch.pipeline_version = :'pipeline_version'
WHERE c.slug = :'collection_slug'
GROUP BY d.id, d.source_name
ORDER BY d.id;

\echo ''
-- 3) Docs missing chunks entirely (should be 0 once chunking ran)
\echo '3) Documents with pages but zero chunks for pipeline'
SELECT
  d.id AS document_id,
  d.source_name,
  COUNT(p.id) AS n_pages
FROM documents d
JOIN collections c ON c.id = d.collection_id
JOIN pages p ON p.document_id = d.id
LEFT JOIN chunk_pages cp ON cp.page_id = p.id
LEFT JOIN chunks ch ON ch.id = cp.chunk_id AND ch.pipeline_version = :'pipeline_version'
WHERE c.slug = :'collection_slug'
GROUP BY d.id, d.source_name
HAVING COUNT(DISTINCT ch.id) = 0
ORDER BY d.id;

\echo ''
-- 4) Orphaned chunks (chunks in pipeline not linked to any pages)
\echo '4) Orphan chunks for pipeline (should be 0)'
SELECT COUNT(*) AS orphan_chunks
FROM chunks ch
LEFT JOIN chunk_pages cp ON cp.chunk_id = ch.id
WHERE ch.pipeline_version = :'pipeline_version'
  AND cp.chunk_id IS NULL;

\echo ''
-- 5) Chunk size distribution quick look
\echo '5) Chunk size distribution (chars) for pipeline + collection'
WITH chunk_set AS (
  SELECT DISTINCT ch.id, ch.text
  FROM chunks ch
  JOIN chunk_pages cp ON cp.chunk_id = ch.id
  JOIN pages p ON p.id = cp.page_id
  JOIN documents d ON d.id = p.document_id
  JOIN collections c ON c.id = d.collection_id
  WHERE ch.pipeline_version = :'pipeline_version'
    AND c.slug = :'collection_slug'
)
SELECT
  COUNT(*) AS n_chunks,
  MIN(LENGTH(text)) AS min_chars,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LENGTH(text)) AS p50_chars,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY LENGTH(text)) AS p90_chars,
  MAX(LENGTH(text)) AS max_chars
FROM chunk_set;

\echo ''
-- 6) Venona-specific invariant: typically 1 page/message => 1 chunk (unless oversize fallback)
\echo '6) Venona check: message count vs chunk count per doc (expect equal or chunks >= pages)'
SELECT
  d.id AS document_id,
  d.source_name,
  COUNT(DISTINCT p.id) AS n_pages_messages,
  COUNT(DISTINCT ch.id) AS n_chunks
FROM documents d
JOIN collections c ON c.id = d.collection_id
JOIN pages p ON p.document_id = d.id
LEFT JOIN chunk_pages cp ON cp.page_id = p.id
LEFT JOIN chunks ch ON ch.id = cp.chunk_id AND ch.pipeline_version = :'pipeline_version'
WHERE c.slug = 'venona'
GROUP BY d.id, d.source_name
ORDER BY d.id;

\echo ''
-- 7) Vassiliev marker leakage check: markers should NOT appear in chunks.text
\echo '7) Vassiliev check: chunks containing marker-like lines (should be 0)'
SELECT COUNT(*) AS chunks_with_pxx_lines
FROM chunks ch
JOIN chunk_pages cp ON cp.chunk_id = ch.id
JOIN pages p ON p.id = cp.page_id
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE ch.pipeline_version = :'pipeline_version'
  AND c.slug = 'vassiliev'
  AND ch.text ~ E'(?m)^\\s*p\\.\\s*\\d+\\s*$';

\echo ''
\echo '--- Done ---'
