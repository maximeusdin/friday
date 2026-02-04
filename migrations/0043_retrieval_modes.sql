-- Migration 0043: Retrieval modes and vector semantics on retrieval_runs
-- Part of Two-Mode Retrieval System implementation
--
-- Adds mode tracking and explicit vector semantics columns to retrieval_runs.
-- Also ensures result_sets.retrieval_run_id is NOT NULL (audit trail integrity).

BEGIN;

-- =============================================================================
-- Add mode tracking columns to retrieval_runs
-- =============================================================================

-- Retrieval mode: "conversational" (fast + explainable) or "thorough" (exhaustive)
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS retrieval_mode VARCHAR(20) DEFAULT 'conversational';

-- How the mode was determined (for debugging/audit)
-- Values: 'ui_toggle', 'primitive', 'trigger_phrase', 'default'
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS mode_source VARCHAR(20) DEFAULT 'default';

-- =============================================================================
-- Add vector semantics columns (explicit, debuggable)
-- =============================================================================
-- Storing these explicitly avoids formula breakage when operators change.

-- Metric type: "cosine" | "l2" | "ip"
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS vector_metric_type VARCHAR(10) DEFAULT 'cosine';

-- pgvector operator used: "<=>" (cosine) | "<->" (L2) | "<#>" (inner product)
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS vector_operator VARCHAR(5) DEFAULT '<=>';

-- Formula used to transform distance to similarity
-- e.g., "1 - cosine_distance" for cosine
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS similarity_transform VARCHAR(50) DEFAULT '1 - cosine_distance';

-- Threshold used for filtering (in similarity space, e.g., [-1, 1] for cosine)
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS similarity_threshold FLOAT;

-- =============================================================================
-- Add cap tracking columns
-- =============================================================================

-- Total results before any cap was applied
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS total_hits_before_cap INTEGER;

-- The cap value that was applied (if any)
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS max_hits_cap INTEGER;

-- Whether a cap was applied
ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS cap_applied BOOLEAN DEFAULT FALSE;

-- =============================================================================
-- Add applied filters (run-level, not per-chunk)
-- =============================================================================
-- Stores the list of filters applied to this query, avoiding repetition per-chunk.

ALTER TABLE retrieval_runs 
    ADD COLUMN IF NOT EXISTS applied_filters JSONB DEFAULT '[]'::jsonb;

-- =============================================================================
-- Ensure result_sets.retrieval_run_id is NOT NULL (audit trail integrity)
-- =============================================================================
-- The existing schema already has this as NOT NULL, but let's verify with a check.

-- This will fail if there are any NULL values (should not be the case)
DO $$
BEGIN
    -- Verify constraint exists or add it
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'result_sets' 
        AND column_name = 'retrieval_run_id'
        AND is_nullable = 'NO'
    ) THEN
        -- This would fail if there are NULL values
        ALTER TABLE result_sets 
            ALTER COLUMN retrieval_run_id SET NOT NULL;
    END IF;
END $$;

-- =============================================================================
-- Add constraint for valid retrieval modes
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'retrieval_runs_mode_check'
    ) THEN
        ALTER TABLE retrieval_runs 
            ADD CONSTRAINT retrieval_runs_mode_check 
            CHECK (retrieval_mode IN ('conversational', 'thorough'));
    END IF;
END $$;

-- =============================================================================
-- Add constraint for valid mode sources
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'retrieval_runs_mode_source_check'
    ) THEN
        ALTER TABLE retrieval_runs 
            ADD CONSTRAINT retrieval_runs_mode_source_check 
            CHECK (mode_source IN ('ui_toggle', 'primitive', 'trigger_phrase', 'default'));
    END IF;
END $$;

-- =============================================================================
-- Indexes for mode-based queries
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_mode 
    ON retrieval_runs(retrieval_mode);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_mode_source 
    ON retrieval_runs(mode_source);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON COLUMN retrieval_runs.retrieval_mode IS 
    'Retrieval mode: conversational (fast, capped) or thorough (exhaustive, paginated)';

COMMENT ON COLUMN retrieval_runs.mode_source IS 
    'How mode was determined: ui_toggle > primitive > trigger_phrase > default';

COMMENT ON COLUMN retrieval_runs.vector_metric_type IS 
    'Vector metric: cosine, l2, or ip (inner product)';

COMMENT ON COLUMN retrieval_runs.vector_operator IS 
    'pgvector operator used: <=> (cosine), <-> (L2), <#> (IP)';

COMMENT ON COLUMN retrieval_runs.similarity_transform IS 
    'Formula to transform distance to similarity (e.g., 1 - cosine_distance)';

COMMENT ON COLUMN retrieval_runs.similarity_threshold IS 
    'Minimum similarity threshold for filtering. Range depends on metric (e.g., [-1, 1] for cosine).';

COMMENT ON COLUMN retrieval_runs.applied_filters IS 
    'List of scope filters applied to this query (stored at run level, not per-chunk)';

COMMIT;
