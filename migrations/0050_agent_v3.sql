-- V3 Agent Framework Tables
-- Migration: 0050_agent_v3.sql
-- 
-- Tables for persisting V3 workflow artifacts:
-- - agent_plans_v3: Plan JSON with hash for reproducibility
-- - v3_tool_calls: Tool execution log
-- - v3_evidence_sets: Evidence with cite/harvest banding
-- - v3_claim_bundles: Claims with verification status

-- V3 Plans
CREATE TABLE IF NOT EXISTS agent_plans_v3 (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    session_id BIGINT REFERENCES research_sessions(id) ON DELETE SET NULL,
    retrieval_run_id BIGINT REFERENCES retrieval_runs(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    plan_json JSONB NOT NULL,
    plan_hash TEXT NOT NULL,
    model_version TEXT NOT NULL,
    constraints_json JSONB,
    budgets_json JSONB,
    reasoning TEXT,
    round_number INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_plans_v3_run_id ON agent_plans_v3(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_plans_v3_session_id ON agent_plans_v3(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_plans_v3_plan_hash ON agent_plans_v3(plan_hash);
CREATE INDEX IF NOT EXISTS idx_agent_plans_v3_created_at ON agent_plans_v3(created_at);

-- V3 Tool Calls
CREATE TABLE IF NOT EXISTS v3_tool_calls (
    id BIGSERIAL PRIMARY KEY,
    plan_id BIGINT REFERENCES agent_plans_v3(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    step_index INT NOT NULL,
    tool_name TEXT NOT NULL,
    params_json JSONB NOT NULL,
    result_chunk_ids BIGINT[],
    result_scores JSONB,
    metadata_json JSONB,
    elapsed_ms REAL,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v3_tool_calls_plan_id ON v3_tool_calls(plan_id);
CREATE INDEX IF NOT EXISTS idx_v3_tool_calls_run_id ON v3_tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_v3_tool_calls_tool_name ON v3_tool_calls(tool_name);

-- V3 Evidence Sets
CREATE TABLE IF NOT EXISTS v3_evidence_sets (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    plan_id BIGINT REFERENCES agent_plans_v3(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    evidence_set_id TEXT NOT NULL,
    cite_span_count INT NOT NULL DEFAULT 0,
    harvest_span_count INT NOT NULL DEFAULT 0,
    cite_span_ids TEXT[],
    harvest_span_ids TEXT[],
    stats_json JSONB,
    total_chunks INT,
    unique_docs INT,
    round_number INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v3_evidence_sets_run_id ON v3_evidence_sets(run_id);
CREATE INDEX IF NOT EXISTS idx_v3_evidence_sets_evidence_set_id ON v3_evidence_sets(evidence_set_id);

-- V3 Evidence Spans (denormalized for fast lookup)
CREATE TABLE IF NOT EXISTS v3_evidence_spans (
    id BIGSERIAL PRIMARY KEY,
    evidence_set_id BIGINT REFERENCES v3_evidence_sets(id) ON DELETE CASCADE,
    span_id TEXT NOT NULL,
    chunk_id BIGINT NOT NULL,
    doc_id BIGINT,
    page_ref TEXT,
    start_char INT NOT NULL,
    end_char INT NOT NULL,
    quote TEXT,
    score REAL,
    band TEXT NOT NULL, -- 'cite' or 'harvest'
    span_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v3_evidence_spans_evidence_set_id ON v3_evidence_spans(evidence_set_id);
CREATE INDEX IF NOT EXISTS idx_v3_evidence_spans_span_id ON v3_evidence_spans(span_id);
CREATE INDEX IF NOT EXISTS idx_v3_evidence_spans_band ON v3_evidence_spans(band);

-- V3 Claim Bundles
CREATE TABLE IF NOT EXISTS v3_claim_bundles (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    evidence_set_id BIGINT REFERENCES v3_evidence_sets(id) ON DELETE SET NULL,
    bundle_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    model_version TEXT,
    bundle_json JSONB NOT NULL,
    claim_count INT NOT NULL DEFAULT 0,
    verification_status TEXT, -- 'pending', 'passed', 'failed'
    verification_errors JSONB,
    verification_warnings JSONB,
    verification_stats JSONB,
    round_number INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v3_claim_bundles_run_id ON v3_claim_bundles(run_id);
CREATE INDEX IF NOT EXISTS idx_v3_claim_bundles_bundle_id ON v3_claim_bundles(bundle_id);
CREATE INDEX IF NOT EXISTS idx_v3_claim_bundles_verification_status ON v3_claim_bundles(verification_status);

-- V3 Run Summary (high-level audit)
CREATE TABLE IF NOT EXISTS v3_run_summary (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    session_id BIGINT REFERENCES research_sessions(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    success BOOLEAN DEFAULT FALSE,
    rounds_used INT DEFAULT 1,
    final_claim_count INT DEFAULT 0,
    final_cite_span_count INT DEFAULT 0,
    verification_passed BOOLEAN DEFAULT FALSE,
    total_elapsed_ms REAL,
    trace_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v3_run_summary_run_id ON v3_run_summary(run_id);
CREATE INDEX IF NOT EXISTS idx_v3_run_summary_session_id ON v3_run_summary(session_id);
CREATE INDEX IF NOT EXISTS idx_v3_run_summary_created_at ON v3_run_summary(created_at);
CREATE INDEX IF NOT EXISTS idx_v3_run_summary_success ON v3_run_summary(success);
