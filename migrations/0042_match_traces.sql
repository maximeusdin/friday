-- Migration 0042: Match traces for "why surfaced" storage
-- Part of Two-Mode Retrieval System implementation
-- 
-- This creates the result_set_match_traces table which stores
-- first-class evidence of why each chunk was retrieved.

BEGIN;

-- =============================================================================
-- Table: result_set_match_traces
-- =============================================================================
-- Stores match trace information explaining why each chunk surfaced.
-- Keyed by (result_set_id, chunk_id) for easy API joins.
-- Split into "hot columns" (fast queries) and "cold" JSONB (audit detail).

CREATE TABLE IF NOT EXISTS result_set_match_traces (
    -- Primary key: keyed by result_set for API joins
    result_set_id   BIGINT NOT NULL REFERENCES result_sets(id) ON DELETE CASCADE,
    chunk_id        BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    
    -- Optional redundant column for direct run queries (simplifies debugging)
    -- Can be derived via result_sets.retrieval_run_id
    retrieval_run_id BIGINT REFERENCES retrieval_runs(id) ON DELETE SET NULL,
    
    -- Hot columns (fast queries) ------------------------------------------
    
    -- Entity IDs that matched query primitives (for aggregation)
    matched_entity_ids  INT[] DEFAULT '{}',
    
    -- Phrases that matched (Tier 2 enrichment fills this)
    matched_phrases     TEXT[] DEFAULT '{}',
    
    -- Scope filter result (always TRUE for included results)
    -- Renamed from filters_passed to clarify semantics
    scope_passed        BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Candidate set membership
    in_lexical          BOOLEAN NOT NULL DEFAULT FALSE,
    in_vector           BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Score columns -------------------------------------------------------
    
    -- Raw scores from each search method
    score_lexical       FLOAT,
    score_vector        FLOAT,
    score_hybrid        FLOAT,
    
    -- Vector distance semantics (for debugging/audit)
    vector_distance     FLOAT,
    vector_similarity   FLOAT,
    
    -- Rank info (primarily for conversational mode) -----------------------
    
    rank                INTEGER,
    rank_trace          JSONB,  -- Explanation of ranking position
    
    -- Audit log (cold storage) --------------------------------------------
    
    -- Full primitive match details for audit
    primitive_matches   JSONB DEFAULT '[]'::jsonb,
    
    -- Note: applied_filters stored at RUN level (retrieval_runs table)
    -- not per-chunk to avoid repetition
    
    -- Cap tracking --------------------------------------------------------
    
    was_capped          BOOLEAN NOT NULL DEFAULT FALSE,
    cap_reason          VARCHAR(100),
    
    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Primary key
    PRIMARY KEY (result_set_id, chunk_id)
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index for cursor pagination (WHERE rank > after_rank ORDER BY rank)
CREATE INDEX IF NOT EXISTS idx_result_set_match_traces_rank 
    ON result_set_match_traces(result_set_id, rank)
    WHERE rank IS NOT NULL;

-- Index for direct retrieval_run queries (debugging)
CREATE INDEX IF NOT EXISTS idx_result_set_match_traces_run 
    ON result_set_match_traces(retrieval_run_id)
    WHERE retrieval_run_id IS NOT NULL;

-- Note: GIN indexes on matched_entity_ids and primitive_matches are DEFERRED
-- per plan - only create when "find traces where entity X surfaced" queries
-- become common. Entity aggregation uses entity_mentions table, not traces.

-- CREATE INDEX IF NOT EXISTS idx_result_set_match_traces_entities 
--     ON result_set_match_traces USING gin(matched_entity_ids);
-- CREATE INDEX IF NOT EXISTS idx_result_set_match_traces_primitive_matches 
--     ON result_set_match_traces USING gin(primitive_matches);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE result_set_match_traces IS 
    'First-class match trace storage explaining why each chunk surfaced in a result set. '
    'Hot columns enable fast aggregation; JSONB audit log preserves full detail.';

COMMENT ON COLUMN result_set_match_traces.scope_passed IS 
    'Always TRUE for included results (chunks that failed filters are not stored). '
    'Kept for schema clarity; may be removed if adds no value.';

COMMENT ON COLUMN result_set_match_traces.matched_entity_ids IS 
    'Entity IDs that matched query primitives. Populated by Tier 1 enrichment.';

COMMENT ON COLUMN result_set_match_traces.matched_phrases IS 
    'Phrases that matched. Populated by Tier 2 enrichment (top N chunks only).';

COMMENT ON COLUMN result_set_match_traces.primitive_matches IS 
    'Full primitive match details for audit. Cold storage - not indexed by default.';

COMMIT;
