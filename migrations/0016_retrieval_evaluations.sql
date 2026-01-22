-- 0016_retrieval_evaluations.sql
-- Phase 7: Evaluation & Safety Harness
-- Stores evaluation results for measuring recall/precision improvements

BEGIN;

CREATE TABLE IF NOT EXISTS retrieval_evaluations (
    id BIGSERIAL PRIMARY KEY,
    
    -- Query identification
    query_text TEXT NOT NULL,
    query_lang_version TEXT NOT NULL,  -- 'qv1' or 'qv2_softlex'
    
    -- Evaluation metrics
    metric_name TEXT NOT NULL,  -- 'recall@10', 'precision@10', 'recall@20', etc.
    metric_value NUMERIC NOT NULL,
    
    -- Context
    search_type TEXT NOT NULL,  -- 'lex', 'vector', 'hybrid'
    chunk_pv TEXT NOT NULL,
    collection_slug TEXT,
    
    -- Evaluation metadata
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    evaluation_config JSONB DEFAULT '{}'::jsonb,  -- Store thresholds, k values, etc.
    
    -- Optional: link to specific retrieval run
    retrieval_run_id BIGINT REFERENCES retrieval_runs(id) ON DELETE SET NULL,
    
    CONSTRAINT retrieval_evaluations_metric_name_check
        CHECK (metric_name ~ '^(recall|precision|f1|overlap)@\d+$')
);

CREATE INDEX IF NOT EXISTS idx_retrieval_evaluations_query_text 
    ON retrieval_evaluations(query_text, query_lang_version, evaluated_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_evaluations_metric 
    ON retrieval_evaluations(metric_name, query_lang_version, evaluated_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_evaluations_retrieval_run 
    ON retrieval_evaluations(retrieval_run_id);

COMMENT ON TABLE retrieval_evaluations IS
'Evaluation results for measuring recall/precision improvements across query language versions. Used for Phase 7 safety harness.';

COMMENT ON COLUMN retrieval_evaluations.query_lang_version IS
'Query language version: qv1 (exact FTS) or qv2_softlex (soft lexical matching).';

COMMENT ON COLUMN retrieval_evaluations.metric_name IS
'Metric name format: {metric}@{k} where metric is recall/precision/f1/overlap and k is the cutoff (e.g., recall@10).';

COMMENT ON COLUMN retrieval_evaluations.evaluation_config IS
'JSONB storing evaluation parameters: k value, thresholds, normalization version, soft lex settings, etc.';

COMMIT;
