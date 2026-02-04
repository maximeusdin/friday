-- Migration 0047: Session summaries for LLM-generated structured summaries
-- Part of the LLM Summarizer implementation
--
-- Stores structured summary artifacts with:
-- - Signature-based caching/deduplication
-- - Selection metadata for reproducibility
-- - Full output JSON for UI rendering

BEGIN;

-- =============================================================================
-- Table: session_summaries
-- =============================================================================
-- Stores LLM-generated summaries with citation-backed claims.
-- Signature enables cache lookup and prevents duplicate generation.

CREATE TABLE IF NOT EXISTS session_summaries (
    id              BIGSERIAL PRIMARY KEY,
    summary_id      UUID NOT NULL DEFAULT gen_random_uuid(),
    result_set_id   BIGINT NOT NULL REFERENCES result_sets(id) ON DELETE CASCADE,
    session_id      BIGINT REFERENCES research_sessions(id) ON DELETE SET NULL,
    
    -- Caching/dedupe: hash of inputs for signature matching
    -- Hash of (result_set_id, retrieval_run_id, profile, summary_type, question, filters, chunk_ids, prompt_version, model_name)
    summary_signature TEXT NOT NULL,
    
    -- Selection metadata (reproducibility) - what was selected
    -- {chunk_ids: [...], policy: "greedy_soft_diversity", bundle_id_map: {...}}
    selection_spec  JSONB NOT NULL,
    
    -- Selection inputs (debugging) - why it was selected
    -- {candidate_pool_size, total_available, lambdas, doc_focus_mode, overrides, facet_snapshot, ...}
    selection_inputs JSONB NOT NULL,
    
    -- Denormalized for queries without JSONB parsing
    selected_chunk_ids BIGINT[] NOT NULL,
    
    -- Input context
    user_question   TEXT,
    profile         VARCHAR(50) NOT NULL DEFAULT 'conversational_answer',
    summary_type    VARCHAR(20) NOT NULL DEFAULT 'sample',  -- "sample" | "page_window"
    
    -- Structured output - Full SummaryArtifact JSON
    output_json     JSONB NOT NULL,
    
    -- Model info for reproducibility
    model_name      VARCHAR(100),
    prompt_version  VARCHAR(50),
    
    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Constraints
    UNIQUE(summary_id),
    UNIQUE(summary_signature)  -- Enables cache lookup, prevents dupes
);

-- =============================================================================
-- Indexes for common access patterns
-- =============================================================================

-- Lookup summaries by result set (most common query)
CREATE INDEX IF NOT EXISTS idx_session_summaries_result_set 
    ON session_summaries(result_set_id, created_at DESC);

-- Lookup summaries by session (for session history)
CREATE INDEX IF NOT EXISTS idx_session_summaries_session 
    ON session_summaries(session_id, created_at DESC)
    WHERE session_id IS NOT NULL;

-- Fast signature lookup for cache hits
CREATE INDEX IF NOT EXISTS idx_session_summaries_signature
    ON session_summaries(summary_signature);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE session_summaries IS 
    'LLM-generated structured summaries with citation-backed claims. '
    'Signature-based caching prevents redundant LLM calls for identical inputs.';

COMMENT ON COLUMN session_summaries.summary_signature IS 
    'SHA256 hash of (result_set_id, retrieval_run_id, profile, summary_type, '
    'question, filters, chunk_ids [ordered], prompt_version, model_name). '
    'Chunk IDs preserve selection order for reproducibility.';

COMMENT ON COLUMN session_summaries.selection_spec IS
    'What was selected: chunk_ids (ordered), policy, bundle_id_map. '
    'Enables exact reproduction of the summary inputs.';

COMMENT ON COLUMN session_summaries.selection_inputs IS
    'Debugging context: candidate_pool_size, total_available, lambdas, '
    'doc_focus_mode, effective_max_per_doc, overrides, facet_snapshot, pool_seed. '
    'Answers "why did it choose these chunks?"';

COMMENT ON COLUMN session_summaries.summary_type IS
    '"sample" = increase K on same policy (better synthesis), '
    '"page_window" = chronological window browsing (offset-based thorough mode).';

COMMENT ON COLUMN session_summaries.output_json IS
    'Full SummaryArtifact: coverage, claims with citations, themes, entities, '
    'dates, next_actions. UI-renderable structured output.';

COMMIT;
