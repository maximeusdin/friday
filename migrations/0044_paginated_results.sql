-- Migration 0044: Paginated results storage (result_set_chunks)
-- Part of Two-Mode Retrieval System implementation
--
-- Creates a normalized table for result set chunks to enable efficient
-- cursor pagination for thorough mode (large result sets).

BEGIN;

-- =============================================================================
-- Table: result_set_chunks
-- =============================================================================
-- Normalized storage for result set chunks, enabling efficient pagination.
-- Each row represents a chunk's position in a result set.
--
-- Deterministic ordering rules:
--   Conversational mode: rank = score-based with tie-break (score DESC, chunk_id ASC)
--   Thorough mode: rank = deterministic by (document_id, chunk_id), NOT score-based
--
-- NULL document_id handling:
--   Chunks without document_id sort LAST using sentinel (2147483647)
--   SQL: ORDER BY COALESCE(document_id, 2147483647), chunk_id
--   Log warning if NULL document_ids found (data quality issue)

CREATE TABLE IF NOT EXISTS result_set_chunks (
    -- References
    result_set_id   BIGINT NOT NULL REFERENCES result_sets(id) ON DELETE CASCADE,
    chunk_id        BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    
    -- Rank for pagination (1-indexed)
    -- In conversational mode: based on hybrid score
    -- In thorough mode: based on (document_id, chunk_id) ordering
    rank            INTEGER NOT NULL,
    
    -- Optional: store document_id for debugging thorough mode ordering
    -- Can be derived from chunk, but storing here simplifies debugging
    document_id     BIGINT,
    
    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Primary key
    PRIMARY KEY (result_set_id, chunk_id)
);

-- =============================================================================
-- Unique constraint on rank (no ties allowed)
-- =============================================================================
-- Ensures deterministic ordering - no two chunks can have the same rank
-- within a result set.

CREATE UNIQUE INDEX IF NOT EXISTS idx_result_set_chunks_unique_rank
    ON result_set_chunks(result_set_id, rank);

-- =============================================================================
-- Index for cursor pagination
-- =============================================================================
-- Supports efficient: WHERE result_set_id = X AND rank > after_rank ORDER BY rank
-- This is the scalable path for thorough mode with large result sets.

CREATE INDEX IF NOT EXISTS idx_result_set_chunks_pagination
    ON result_set_chunks(result_set_id, rank);

-- =============================================================================
-- Index for chunk lookups
-- =============================================================================
-- Supports: "Is this chunk in this result set?"

CREATE INDEX IF NOT EXISTS idx_result_set_chunks_chunk
    ON result_set_chunks(chunk_id);

-- =============================================================================
-- View: result_set_chunks_with_details
-- =============================================================================
-- Convenience view joining chunk details for API responses.

CREATE OR REPLACE VIEW result_set_chunks_with_details AS
SELECT 
    rsc.result_set_id,
    rsc.chunk_id,
    rsc.rank,
    rsc.document_id,
    COALESCE(c.clean_text, c.text) AS chunk_content,
    cm.document_id AS chunk_document_id,
    rs.name AS result_set_name,
    rs.retrieval_run_id
FROM result_set_chunks rsc
JOIN chunks c ON rsc.chunk_id = c.id
LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
JOIN result_sets rs ON rsc.result_set_id = rs.id;

-- =============================================================================
-- Function: populate_result_set_chunks
-- =============================================================================
-- Helper function to populate result_set_chunks from a result_set.
-- This is called during execution to normalize the chunk_ids array.

CREATE OR REPLACE FUNCTION populate_result_set_chunks(
    p_result_set_id BIGINT,
    p_mode VARCHAR(20) DEFAULT 'conversational'
) RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    -- Delete any existing entries (for idempotency)
    DELETE FROM result_set_chunks WHERE result_set_id = p_result_set_id;
    
    IF p_mode = 'thorough' THEN
        -- Thorough mode: deterministic ordering by (document_id, chunk_id)
        -- NULL document_id sorts LAST
        INSERT INTO result_set_chunks (result_set_id, chunk_id, rank, document_id)
        SELECT 
            p_result_set_id,
            c.id,
            ROW_NUMBER() OVER (
                ORDER BY COALESCE(c.document_id, 2147483647), c.id
            ) AS rank,
            c.document_id
        FROM result_sets rs
        CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
        JOIN chunks c ON c.id = chunk_id
        WHERE rs.id = p_result_set_id;
    ELSE
        -- Conversational mode: preserve existing order from chunk_ids array
        -- (assumes chunk_ids array is already sorted by score)
        INSERT INTO result_set_chunks (result_set_id, chunk_id, rank, document_id)
        SELECT 
            p_result_set_id,
            c.id,
            ordinality AS rank,
            c.document_id
        FROM result_sets rs
        CROSS JOIN LATERAL unnest(rs.chunk_ids) WITH ORDINALITY AS t(chunk_id, ordinality)
        JOIN chunks c ON c.id = t.chunk_id
        WHERE rs.id = p_result_set_id;
    END IF;
    
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE result_set_chunks IS 
    'Normalized storage for result set chunks, enabling efficient cursor pagination. '
    'Rank is deterministic: score-based (conversational) or document/chunk order (thorough).';

COMMENT ON COLUMN result_set_chunks.rank IS 
    'Position in result set (1-indexed). Used for cursor pagination (WHERE rank > X). '
    'Unique within result_set_id.';

COMMENT ON COLUMN result_set_chunks.document_id IS 
    'Optional: document_id for debugging thorough mode ordering. Can be derived from chunk.';

COMMENT ON FUNCTION populate_result_set_chunks(BIGINT, VARCHAR) IS 
    'Populates result_set_chunks from result_set.chunk_ids array. '
    'Mode determines ordering: conversational preserves array order, thorough uses (doc_id, chunk_id).';

COMMIT;
