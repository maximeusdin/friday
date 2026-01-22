-- Query recent evaluation results for soft lex trigram threshold sweep
-- Run with: Get-Content scripts/sql/query_evaluation_results.sql | docker exec -i neh_postgres psql -U neh -d neh

SELECT 
    query_text,
    query_lang_version,
    metric_name,
    metric_value,
    evaluation_config->>'soft_lex_trigram_threshold' AS trigram_threshold,
    evaluation_config->>'soft_lex_threshold' AS soft_lex_threshold,
    evaluation_config->>'k' AS k_value,
    evaluated_at
FROM retrieval_evaluations
WHERE evaluation_config->>'soft_lex_trigram_threshold' IS NOT NULL
ORDER BY query_text, evaluated_at DESC, metric_name
LIMIT 100;

-- Summary by threshold
SELECT 
    evaluation_config->>'soft_lex_trigram_threshold' AS trigram_threshold,
    query_text,
    query_lang_version,
    COUNT(*) AS metric_count,
    MAX(CASE WHEN metric_name LIKE 'recall@%' THEN metric_value END) AS recall,
    MAX(CASE WHEN metric_name LIKE 'precision@%' THEN metric_value END) AS precision,
    MAX(CASE WHEN metric_name LIKE 'overlap@%' THEN metric_value END) AS overlap
FROM retrieval_evaluations
WHERE evaluation_config->>'soft_lex_trigram_threshold' IS NOT NULL
GROUP BY 
    evaluation_config->>'soft_lex_trigram_threshold',
    query_text,
    query_lang_version
ORDER BY 
    (evaluation_config->>'soft_lex_trigram_threshold')::text,
    query_text,
    query_lang_version;
