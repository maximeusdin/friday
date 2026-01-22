-- Component Contribution Analysis
-- Analyzes existing retrieval_runs and retrieval_run_chunk_evidence
-- to understand which components (vector, lexical, soft lex) are driving results

-- Set up: Find runs for golden queries
WITH golden_queries AS (
    SELECT unnest(ARRAY['silvermaster', 'silvermastre', 'hiss', 'fuchs', 'Oppenheimer']) AS query_text
),
recent_runs AS (
    SELECT DISTINCT ON (rr.query_text, rr.query_lang_version)
        rr.id AS run_id,
        rr.query_text,
        rr.query_lang_version,
        rr.search_type,
        rr.expand_concordance,
        rr.retrieval_config_json,
        rr.top_k,
        (rr.retrieval_config_json->>'rrf_k')::int AS rrf_k,
        array_length(rr.returned_chunk_ids, 1) AS num_results
    FROM retrieval_runs rr
    JOIN golden_queries gq ON rr.query_text = gq.query_text
    ORDER BY rr.query_text, rr.query_lang_version, rr.created_at DESC
)
SELECT 
    r.run_id,
    r.query_text,
    r.query_lang_version,
    r.search_type,
    r.expand_concordance,
    r.top_k,
    COALESCE(r.rrf_k, 50) AS rrf_k,
    r.num_results,
    COUNT(e.chunk_id) AS evidence_count,
    COUNT(CASE WHEN e.explain_json->'semantic'->>'r_vec' IS NOT NULL THEN 1 END) AS chunks_with_vector,
    COUNT(CASE WHEN e.explain_json->'lex'->>'r_lex' IS NOT NULL THEN 1 END) AS chunks_with_lexical,
    COUNT(CASE WHEN e.explain_json->'approx_lex'->>'r_soft_lex' IS NOT NULL THEN 1 END) AS chunks_with_soft_lex,
    COUNT(CASE WHEN e.explain_json->'semantic'->>'r_vec' IS NOT NULL 
                AND e.explain_json->'lex'->>'r_lex' IS NOT NULL THEN 1 END) AS chunks_with_both,
    AVG((e.explain_json->'semantic'->>'r_vec')::int) AS avg_vector_rank,
    AVG((e.explain_json->'lex'->>'r_lex')::int) AS avg_lexical_rank,
    AVG((e.explain_json->'approx_lex'->>'r_soft_lex')::int) AS avg_soft_lex_rank
FROM recent_runs r
LEFT JOIN retrieval_run_chunk_evidence e ON e.retrieval_run_id = r.run_id
GROUP BY r.run_id, r.query_text, r.query_lang_version, r.search_type, 
         r.expand_concordance, r.top_k, r.rrf_k, r.num_results
ORDER BY r.query_text, r.query_lang_version;

-- Detailed score contribution analysis
WITH recent_runs AS (
    SELECT DISTINCT ON (rr.query_text, rr.query_lang_version)
        rr.id AS run_id,
        rr.query_text,
        rr.query_lang_version,
        (rr.retrieval_config_json->>'rrf_k')::int AS rrf_k,
        (rr.retrieval_config_json->>'soft_lex_weight')::float AS soft_lex_weight
    FROM retrieval_runs rr
    WHERE rr.query_text IN ('silvermaster', 'silvermastre', 'hiss', 'fuchs', 'Oppenheimer')
    ORDER BY rr.query_text, rr.query_lang_version, rr.created_at DESC
),
component_scores AS (
    SELECT 
        r.run_id,
        r.query_text,
        r.query_lang_version,
        COALESCE(r.rrf_k, 50) AS rrf_k,
        COALESCE(r.soft_lex_weight, 0.5) AS soft_lex_weight,
        e.chunk_id,
        e.rank,
        (e.explain_json->'semantic'->>'r_vec')::int AS r_vec,
        (e.explain_json->'lex'->>'r_lex')::int AS r_lex,
        (e.explain_json->'approx_lex'->>'r_soft_lex')::int AS r_soft_lex,
        e.score_hybrid AS hybrid_score
    FROM recent_runs r
    JOIN retrieval_run_chunk_evidence e ON e.retrieval_run_id = r.run_id
    WHERE e.explain_json IS NOT NULL
)
SELECT 
    query_text,
    query_lang_version,
    COUNT(*) AS num_chunks,
    -- Vector contribution
    COUNT(CASE WHEN r_vec IS NOT NULL THEN 1 END) AS vector_chunks,
    SUM(CASE WHEN r_vec IS NOT NULL THEN 1.0 / (rrf_k + r_vec) ELSE 0 END) AS total_vector_score,
    AVG(CASE WHEN r_vec IS NOT NULL THEN 1.0 / (rrf_k + r_vec) END) AS avg_vector_score,
    -- Lexical contribution
    COUNT(CASE WHEN r_lex IS NOT NULL THEN 1 END) AS lexical_chunks,
    SUM(CASE WHEN r_lex IS NOT NULL THEN 1.0 / (rrf_k + r_lex) ELSE 0 END) AS total_lexical_score,
    AVG(CASE WHEN r_lex IS NOT NULL THEN 1.0 / (rrf_k + r_lex) END) AS avg_lexical_score,
    -- Soft lex contribution
    COUNT(CASE WHEN r_soft_lex IS NOT NULL THEN 1 END) AS soft_lex_chunks,
    SUM(CASE WHEN r_soft_lex IS NOT NULL THEN soft_lex_weight * (1.0 / (rrf_k + r_soft_lex)) ELSE 0 END) AS total_soft_lex_score,
    AVG(CASE WHEN r_soft_lex IS NOT NULL THEN soft_lex_weight * (1.0 / (rrf_k + r_soft_lex)) END) AS avg_soft_lex_score,
    -- Total hybrid score
    SUM(COALESCE(hybrid_score, 0)) AS total_hybrid_score
FROM component_scores
GROUP BY query_text, query_lang_version
ORDER BY query_text, query_lang_version;

-- Overlap analysis: Compare qv1 vs qv2 for same queries
WITH qv1_runs AS (
    SELECT DISTINCT ON (query_text)
        id AS run_id,
        query_text,
        returned_chunk_ids
    FROM retrieval_runs
    WHERE query_text IN ('silvermaster', 'silvermastre', 'hiss', 'fuchs', 'Oppenheimer')
      AND query_lang_version = 'qv1'
    ORDER BY query_text, created_at DESC
),
qv2_runs AS (
    SELECT DISTINCT ON (query_text)
        id AS run_id,
        query_text,
        returned_chunk_ids
    FROM retrieval_runs
    WHERE query_text IN ('silvermaster', 'silvermastre', 'hiss', 'fuchs', 'Oppenheimer')
      AND query_lang_version = 'qv2_softlex'
    ORDER BY query_text, created_at DESC
),
overlap_analysis AS (
    SELECT 
        q1.query_text,
        q1.returned_chunk_ids AS qv1_chunks,
        q2.returned_chunk_ids AS qv2_chunks,
        array_length(q1.returned_chunk_ids, 1) AS qv1_count,
        array_length(q2.returned_chunk_ids, 1) AS qv2_count,
        (
            SELECT COUNT(*)
            FROM unnest(q1.returned_chunk_ids) AS chunk_id
            WHERE chunk_id = ANY(q2.returned_chunk_ids)
        ) AS overlap_count
    FROM qv1_runs q1
    JOIN qv2_runs q2 ON q1.query_text = q2.query_text
)
SELECT 
    query_text,
    qv1_count,
    qv2_count,
    overlap_count,
    ROUND(overlap_count::numeric / GREATEST(qv1_count, qv2_count, 1), 3) AS overlap_ratio,
    qv2_count - overlap_count AS new_in_qv2,
    qv1_count - overlap_count AS lost_in_qv2
FROM overlap_analysis
ORDER BY query_text;
