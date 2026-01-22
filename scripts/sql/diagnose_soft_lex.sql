-- Diagnose why soft lex is not matching
-- Checks multiple hypotheses without running new queries

-- 1. Check if chunks have clean_text column and if it's populated
SELECT 
    pipeline_version,
    COUNT(*) AS total_chunks,
    COUNT(clean_text) AS chunks_with_clean_text,
    COUNT(CASE WHEN clean_text IS NOT NULL AND clean_text != text THEN 1 END) AS chunks_with_different_clean_text,
    COUNT(CASE WHEN clean_text IS NULL THEN 1 END) AS chunks_without_clean_text
FROM chunks
WHERE pipeline_version IN ('chunk_v1_full', 'chunk_v1_silvermaster_structured_4k')
GROUP BY pipeline_version;

-- 2. Check soft lex threshold and configuration from recent runs
SELECT DISTINCT
    query_text,
    query_lang_version,
    (retrieval_config_json->>'soft_lex_threshold')::float AS soft_lex_threshold,
    (retrieval_config_json->>'soft_lex_max_results')::int AS soft_lex_max_results,
    (retrieval_config_json->>'soft_lex_weight')::float AS soft_lex_weight,
    normalization_version
FROM retrieval_runs
WHERE query_text IN ('silvermaster', 'silvermastre', 'hiss', 'fuchs', 'Oppenheimer')
  AND query_lang_version = 'qv2_softlex'
ORDER BY query_text, created_at DESC
LIMIT 10;

-- 3. Test similarity scores for a known query (silvermastre -> silvermaster)
-- This checks if soft lex WOULD match if threshold was lower
WITH test_query AS (
    SELECT 'silvermastre' AS query_term,
           'silvermaster' AS collection
),
sample_chunks AS (
    SELECT 
        c.id AS chunk_id,
        c.text,
        c.clean_text,
        COALESCE(c.clean_text, c.text) AS text_for_similarity,
        cm.collection_slug
    FROM chunks c
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE cm.collection_slug = 'silvermaster'
      AND c.pipeline_version = 'chunk_v1_silvermaster_structured_4k'
    LIMIT 20
)
SELECT 
    sc.chunk_id,
    sc.collection_slug,
    LEFT(sc.text_for_similarity, 100) AS text_preview,
    word_similarity(tq.query_term, sc.text_for_similarity) AS similarity_score,
    CASE 
        WHEN word_similarity(tq.query_term, sc.text_for_similarity) >= 0.3 THEN 'WOULD MATCH (threshold 0.3)'
        WHEN word_similarity(tq.query_term, sc.text_for_similarity) >= 0.2 THEN 'WOULD MATCH (threshold 0.2)'
        WHEN word_similarity(tq.query_term, sc.text_for_similarity) >= 0.15 THEN 'WOULD MATCH (threshold 0.15)'
        ELSE 'NO MATCH'
    END AS match_status,
    sc.text IS DISTINCT FROM sc.clean_text AS has_clean_text_difference
FROM test_query tq
CROSS JOIN sample_chunks sc
ORDER BY similarity_score DESC
LIMIT 20;

-- 4. Check if soft lex CTE would find matches for recent queries
-- This simulates what the soft lex CTE would do
WITH recent_query AS (
    SELECT 
        query_text,
        normalization_version,
        (retrieval_config_json->>'soft_lex_threshold')::float AS soft_lex_threshold,
        (retrieval_config_json->>'soft_lex_max_results')::int AS soft_lex_max_results
    FROM retrieval_runs
    WHERE query_text = 'silvermastre'
      AND query_lang_version = 'qv2_softlex'
    ORDER BY created_at DESC
    LIMIT 1
),
normalized_query AS (
    -- Simulate normalization (simplified - actual normalization might be more complex)
    SELECT 
        LOWER(TRIM(query_text)) AS normalized_term,
        query_text AS original_term,
        COALESCE(soft_lex_threshold, 0.3) AS threshold,
        COALESCE(soft_lex_max_results, 50) AS max_results
    FROM recent_query
),
chunk_similarities AS (
    SELECT 
        c.id AS chunk_id,
        cm.collection_slug,
        COALESCE(c.clean_text, c.text) AS text_for_similarity,
        word_similarity(nq.normalized_term, COALESCE(c.clean_text, c.text)) AS similarity_score
    FROM chunks c
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    CROSS JOIN normalized_query nq
    WHERE cm.collection_slug = 'silvermaster'
      AND c.pipeline_version = 'chunk_v1_silvermaster_structured_4k'
      AND COALESCE(c.clean_text, c.text) %% nq.normalized_term  -- Trigram filter (uses index)
)
SELECT 
    COUNT(*) AS total_candidates,
    COUNT(CASE WHEN similarity_score >= threshold THEN 1 END) AS would_match_at_threshold,
    MAX(similarity_score) AS max_similarity,
    AVG(similarity_score) AS avg_similarity,
    MIN(similarity_score) AS min_similarity,
    COUNT(CASE WHEN similarity_score >= 0.2 THEN 1 END) AS would_match_at_0_2,
    COUNT(CASE WHEN similarity_score >= 0.15 THEN 1 END) AS would_match_at_0_15
FROM chunk_similarities
CROSS JOIN normalized_query;

-- 5. Check if soft lex matches exist in evidence but are ranked below top-k
-- This checks if soft lex found matches but they didn't make it into top-k
WITH recent_run AS (
    SELECT id AS run_id, query_text, top_k
    FROM retrieval_runs
    WHERE query_text = 'silvermastre'
      AND query_lang_version = 'qv2_softlex'
    ORDER BY created_at DESC
    LIMIT 1
),
soft_lex_matches AS (
    SELECT 
        e.chunk_id,
        e.rank,
        (e.explain_json->'approx_lex'->>'r_soft_lex')::int AS r_soft_lex,
        (e.explain_json->'approx_lex'->>'score')::float AS soft_lex_score,
        e.score_hybrid
    FROM retrieval_run_chunk_evidence e
    JOIN recent_run r ON e.retrieval_run_id = r.run_id
    WHERE e.explain_json->'approx_lex'->>'r_soft_lex' IS NOT NULL
)
SELECT 
    COUNT(*) AS soft_lex_matches_found,
    MIN(rank) AS best_rank,
    MAX(rank) AS worst_rank,
    AVG(rank) AS avg_rank,
    COUNT(CASE WHEN rank <= (SELECT top_k FROM recent_run) THEN 1 END) AS in_top_k
FROM soft_lex_matches;

-- 6. Check normalization version usage
SELECT 
    normalization_version,
    COUNT(*) AS num_runs,
    COUNT(DISTINCT query_text) AS num_queries
FROM retrieval_runs
WHERE query_lang_version = 'qv2_softlex'
  AND normalization_version IS NOT NULL
GROUP BY normalization_version;

-- 7. Check if pg_trgm extension is enabled and working
SELECT 
    EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') AS pg_trgm_enabled,
    word_similarity('silvermastre', 'silvermaster') AS test_similarity,
    'silvermastre' %% 'silvermaster' AS trigram_match_test;
