-- Migration: Focus Spans and FocusBundle Params for Agentic V2
-- These tables support the FocusBundle architecture where only FocusSpans can be cited.

-- Focus spans per retrieval run
-- Only these spans can be cited in answers (Invariant #1)
CREATE TABLE IF NOT EXISTS focus_spans (
    retrieval_run_id BIGINT NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    start_char INT NOT NULL,
    end_char INT NOT NULL,
    score NUMERIC NOT NULL,
    rank INT NOT NULL,
    source_lanes TEXT[] NOT NULL DEFAULT '{}',
    span_text TEXT NOT NULL,
    span_hash TEXT NOT NULL,  -- sha256 for reproducibility (Contract C1)
    doc_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    page_ref TEXT NOT NULL DEFAULT 'p0',  -- Contract C2: f"p{page_num}"
    
    PRIMARY KEY (retrieval_run_id, chunk_id, start_char, end_char)
);

-- Indexes for focus_spans
CREATE INDEX IF NOT EXISTS idx_focus_spans_run ON focus_spans(retrieval_run_id);
CREATE INDEX IF NOT EXISTS idx_focus_spans_rank ON focus_spans(retrieval_run_id, rank);
CREATE INDEX IF NOT EXISTS idx_focus_spans_chunk ON focus_spans(chunk_id);
CREATE INDEX IF NOT EXISTS idx_focus_spans_doc ON focus_spans(doc_id);
CREATE INDEX IF NOT EXISTS idx_focus_spans_hash ON focus_spans(span_hash);

-- FocusBundle params table (separate from result_sets to avoid bloat)
-- Stores builder params for reproducibility and audit
CREATE TABLE IF NOT EXISTS focus_bundle_params (
    retrieval_run_id BIGINT PRIMARY KEY REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    params_json JSONB NOT NULL,  -- sentence_splitter, embedding_model, etc.
    query_contract_json JSONB,   -- QueryContract for TARGET_ANCHORED mode
    total_spans_mined INT NOT NULL,
    total_chunks INT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'keyword_intent',  -- 'keyword_intent' | 'target_anchored'
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

-- Entity document frequency table for hubness penalty
CREATE TABLE IF NOT EXISTS entity_df (
    entity_id BIGINT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    doc_df INT NOT NULL,      -- number of unique documents mentioning entity
    chunk_df INT NOT NULL,    -- number of unique chunks mentioning entity
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entity_df_doc ON entity_df(doc_df DESC);
CREATE INDEX IF NOT EXISTS idx_entity_df_chunk ON entity_df(chunk_df DESC);

-- Refresh function for entity_df
CREATE OR REPLACE FUNCTION refresh_entity_df() RETURNS void AS $$
BEGIN
    -- Truncate and rebuild
    TRUNCATE entity_df;
    
    INSERT INTO entity_df (entity_id, doc_df, chunk_df, updated_at)
    SELECT 
        em.entity_id,
        COUNT(DISTINCT cm.document_id) as doc_df,
        COUNT(DISTINCT em.chunk_id) as chunk_df,
        now()
    FROM entity_mentions em
    JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
    WHERE em.entity_id IS NOT NULL
    GROUP BY em.entity_id;
END;
$$ LANGUAGE plpgsql;

-- Constraint support table (optional, for audit)
CREATE TABLE IF NOT EXISTS constraint_support (
    id BIGSERIAL PRIMARY KEY,
    retrieval_run_id BIGINT NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    candidate_key TEXT NOT NULL,
    constraint_key TEXT NOT NULL,  -- Contract C4: f"{name}:{object}"
    score NUMERIC NOT NULL,
    supporting_span_ids TEXT[] NOT NULL,
    feature_trace JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_constraint_support_run ON constraint_support(retrieval_run_id);
CREATE INDEX IF NOT EXISTS idx_constraint_support_candidate ON constraint_support(candidate_key);
