-- =========================================
-- Validation: Vassiliev Ingest
-- Canonical unit: PDF pages
-- p.xx markers stored only in page_metadata when present
-- =========================================

-- 1) Document + page counts (non-zero, sensible)
SELECT d.id AS document_id,
       d.source_name,
       d.volume,
       COUNT(p.id) AS n_pages
FROM documents d
JOIN collections c ON c.id = d.collection_id
LEFT JOIN pages p ON p.document_id = d.id
WHERE c.slug = 'vassiliev'
GROUP BY d.id, d.source_name, d.volume
ORDER BY d.source_name, d.volume;


-- 2) Ordering correctness (page_seq monotonic, matches pdf_page_number)
SELECT d.source_name,
       COUNT(*) AS n_pages,
       MIN(p.page_seq) AS min_seq,
       MAX(p.page_seq) AS max_seq,
       COUNT(*) FILTER (WHERE p.page_seq = p.pdf_page_number) AS seq_eq_pdf
FROM documents d
JOIN collections c ON c.id = d.collection_id
JOIN pages p ON p.document_id = d.id
WHERE c.slug = 'vassiliev'
GROUP BY d.source_name
ORDER BY d.source_name;


-- 3) Gaps or duplicates in page_seq (should return ZERO rows)
-- Duplicates
SELECT d.source_name, p.page_seq, COUNT(*)
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'vassiliev'
GROUP BY d.source_name, p.page_seq
HAVING COUNT(*) > 1;

-- Gaps (since page_seq == pdf_page_number)
SELECT d.source_name,
       COUNT(*) AS n_pages,
       MAX(p.page_seq) AS max_seq
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'vassiliev'
GROUP BY d.source_name
HAVING COUNT(*) <> MAX(p.page_seq);


-- 4) logical_page_label format check (should all be pdf.N)
SELECT d.source_name,
       COUNT(*) AS total,
       COUNT(*) FILTER (WHERE p.logical_page_label ~ '^pdf\.[0-9]+$') AS ok
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'vassiliev'
GROUP BY d.source_name
ORDER BY d.source_name;


-- 5) Raw text sanity (detect extraction failures)
SELECT d.source_name,
       COUNT(*) FILTER (WHERE length(p.raw_text) < 50) AS tiny_pages,
       ROUND(AVG(length(p.raw_text))) AS avg_chars,
       MAX(length(p.raw_text)) AS max_chars
FROM pages p
JOIN documents d ON d.id = p.document_id
JOIN collections c ON c.id = d.collection_id
WHERE c.slug = 'vassiliev'
GROUP BY d.source_name
ORDER BY tiny_pages DESC, d.source_name;


-- 6) page_metadata coverage + marker semantics
-- A) metadata rows should match page rows
SELECT d.source_name,
       COUNT(p.id) AS page_rows,
       COUNT(pm.page_id) FILTER (WHERE pm.pipeline_version = 'meta_v1_pxx_markers') AS meta_rows
FROM documents d
JOIN collections c ON c.id = d.collection_id
JOIN pages p ON p.document_id = d.id
LEFT JOIN page_metadata pm ON pm.page_id = p.id
WHERE c.slug = 'vassiliev'
GROUP BY d.source_name
ORDER BY d.source_name;

-- B) "with_markers" = key exists ONLY when present
SELECT d.source_name,
       COUNT(*) FILTER (WHERE pm.meta_raw ? 'p_markers') AS pages_with_markers,
       COUNT(*) AS total_pages
FROM documents d
JOIN collections c ON c.id = d.collection_id
JOIN pages p ON p.document_id = d.id
JOIN page_metadata pm
  ON pm.page_id = p.id
 AND pm.pipeline_version = 'meta_v1_pxx_markers'
WHERE c.slug = 'vassiliev'
GROUP BY d.source_name
ORDER BY pages_with_markers DESC, d.source_name;
