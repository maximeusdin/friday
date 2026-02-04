-- =============================================================================
-- Migration: 0048_evidence_bundles.sql
-- Creates evidence_bundles table for agentic workflow audit trail
-- =============================================================================

-- Evidence bundles store the full audit trail of agentic workflow execution
-- This enables:
-- - Debugging why a particular answer was generated
-- - Reproducing results from plan + bundle
-- - Comparing outputs across different plan configurations
-- - Versioning and history of evidence collection

CREATE TABLE IF NOT EXISTS evidence_bundles (
    id SERIAL PRIMARY KEY,
    
    -- Link to result set (if one was created)
    result_set_id INTEGER REFERENCES result_sets(id),
    
    -- Type and version for schema evolution
    bundle_type TEXT NOT NULL DEFAULT 'agentic_v1',
    
    -- Full audit trail stored as JSONB
    -- plan_json: The AgenticPlan that produced this bundle
    plan_json JSONB NOT NULL,
    
    -- lane_runs_json: All RetrievalLaneRun objects with hit stats
    lane_runs_json JSONB NOT NULL,
    
    -- bundle_json: Full EvidenceBundle with claims, entities, etc.
    bundle_json JSONB NOT NULL,
    
    -- Verification status and details
    verification_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (verification_status IN ('pending', 'passed', 'failed')),
    verification_errors JSONB,
    verification_warnings JSONB,
    
    -- Human-readable trace for debugging
    answer_trace_json JSONB,
    
    -- Stats for quick filtering/analysis
    rounds_executed INT,
    claims_count INT,
    entities_count INT,
    unresolved_count INT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_evidence_bundles_result_set 
    ON evidence_bundles(result_set_id);

CREATE INDEX IF NOT EXISTS idx_evidence_bundles_status 
    ON evidence_bundles(verification_status);

CREATE INDEX IF NOT EXISTS idx_evidence_bundles_created 
    ON evidence_bundles(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evidence_bundles_type 
    ON evidence_bundles(bundle_type);

-- GIN index for JSONB queries on plan_json (e.g., finding bundles by intent)
CREATE INDEX IF NOT EXISTS idx_evidence_bundles_plan_gin 
    ON evidence_bundles USING GIN (plan_json);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_evidence_bundles_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS evidence_bundles_updated_at ON evidence_bundles;
CREATE TRIGGER evidence_bundles_updated_at
    BEFORE UPDATE ON evidence_bundles
    FOR EACH ROW
    EXECUTE FUNCTION update_evidence_bundles_updated_at();

-- Comments for documentation
COMMENT ON TABLE evidence_bundles IS 
    'Stores evidence bundles from agentic workflow execution for audit and debugging';

COMMENT ON COLUMN evidence_bundles.bundle_type IS 
    'Schema version for bundle format (e.g., agentic_v1)';

COMMENT ON COLUMN evidence_bundles.plan_json IS 
    'The AgenticPlan that produced this bundle - enables reproduction';

COMMENT ON COLUMN evidence_bundles.lane_runs_json IS 
    'All RetrievalLaneRun objects with coverage stats';

COMMENT ON COLUMN evidence_bundles.bundle_json IS 
    'Full EvidenceBundle: claims, entities, constraints, etc.';

COMMENT ON COLUMN evidence_bundles.verification_status IS 
    'Result of deterministic verification: pending/passed/failed';

COMMENT ON COLUMN evidence_bundles.answer_trace_json IS 
    'Human-readable trace explaining why this answer was generated';
