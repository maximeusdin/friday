-- V4.2 Discovery Loop Schema
-- Adds persistence for discovery artifacts

-- Add discovery trace to v4_run_summary
ALTER TABLE v4_run_summary 
ADD COLUMN IF NOT EXISTS discovery_enabled BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS discovery_rounds INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS discovery_trace JSONB,
ADD COLUMN IF NOT EXISTS thorough_mode BOOLEAN DEFAULT FALSE;

-- Discovery plans table (optional, for detailed audit)
CREATE TABLE IF NOT EXISTS v4_discovery_plans (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    plan_json JSONB NOT NULL,
    plan_hash TEXT NOT NULL,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Index for lookups
    CONSTRAINT unique_run_round UNIQUE (run_id, round_num)
);

-- Discovery tool calls table (for debugging and analysis)
CREATE TABLE IF NOT EXISTS v4_discovery_tool_calls (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    step_num INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    params JSONB NOT NULL,
    action_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    new_chunk_count INTEGER DEFAULT 0,
    elapsed_ms REAL,
    success BOOLEAN DEFAULT TRUE,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_discovery_tool_calls_run 
ON v4_discovery_tool_calls(run_id);

CREATE INDEX IF NOT EXISTS idx_discovery_tool_calls_action_hash
ON v4_discovery_tool_calls(action_hash);

-- Discovery state snapshots (optional, for debugging)
CREATE TABLE IF NOT EXISTS v4_discovery_states (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    candidate_chunk_count INTEGER NOT NULL,
    discovered_entity_count INTEGER NOT NULL,
    coverage_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_discovery_states_run
ON v4_discovery_states(run_id);

-- Add comment explaining the discovery schema
COMMENT ON TABLE v4_discovery_plans IS 
'V4.2 Discovery Loop: Stores discovery plans proposed by 4o for iterative retrieval';

COMMENT ON TABLE v4_discovery_tool_calls IS 
'V4.2 Discovery Loop: Audit trail of tool calls executed during discovery';

COMMENT ON COLUMN v4_run_summary.discovery_trace IS 
'V4.2 Discovery Loop: Full trace JSON including rounds, coverage metrics, stop decision';
