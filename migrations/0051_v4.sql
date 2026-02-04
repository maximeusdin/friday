-- V4 Agentic Framework Database Schema
-- Stores interpretations, verification reports, answer units, and run summaries

-- V4 Interpretations: Stores the structured interpretation JSON from 4o
CREATE TABLE IF NOT EXISTS v4_interpretations (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    response_shape TEXT NOT NULL,
    interpretation_json JSONB NOT NULL,
    interpretation_hash TEXT NOT NULL,
    model_version TEXT NOT NULL,
    input_span_ids TEXT[],
    input_span_hashes TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for lookups by run_id and hash
CREATE INDEX IF NOT EXISTS idx_v4_interpretations_run_id ON v4_interpretations(run_id);
CREATE INDEX IF NOT EXISTS idx_v4_interpretations_hash ON v4_interpretations(interpretation_hash);
CREATE INDEX IF NOT EXISTS idx_v4_interpretations_response_shape ON v4_interpretations(response_shape);

-- V4 Verification Reports: Stores verification results for each interpretation
CREATE TABLE IF NOT EXISTS v4_verification_reports (
    id BIGSERIAL PRIMARY KEY,
    interpretation_id BIGINT REFERENCES v4_interpretations(id) ON DELETE CASCADE,
    passed BOOLEAN NOT NULL,
    hard_errors JSONB,
    soft_warnings JSONB,
    per_unit_status JSONB,
    anchor_present BOOLEAN DEFAULT FALSE,
    stats JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for finding verification by interpretation
CREATE INDEX IF NOT EXISTS idx_v4_verification_reports_interpretation ON v4_verification_reports(interpretation_id);
CREATE INDEX IF NOT EXISTS idx_v4_verification_reports_passed ON v4_verification_reports(passed);

-- V4 Answer Units: Denormalized storage for individual answer units (for search/analysis)
CREATE TABLE IF NOT EXISTS v4_answer_units (
    id BIGSERIAL PRIMARY KEY,
    interpretation_id BIGINT REFERENCES v4_interpretations(id) ON DELETE CASCADE,
    unit_id TEXT NOT NULL,
    unit_type TEXT NOT NULL,          -- anchor|item|finding|note|uncertainty
    text TEXT NOT NULL,
    confidence TEXT NOT NULL,          -- supported|suggestive
    about_entities INTEGER[],          -- Entity IDs
    about_literals TEXT[],             -- Dates, numbers, etc.
    tags TEXT[],                       -- membership|role|codename|etc.
    citation_span_ids TEXT[],          -- Referenced span IDs
    verification_status TEXT,          -- passed|failed|downgraded
    final_confidence TEXT,             -- After any downgrades
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for answer unit queries
CREATE INDEX IF NOT EXISTS idx_v4_answer_units_interpretation ON v4_answer_units(interpretation_id);
CREATE INDEX IF NOT EXISTS idx_v4_answer_units_type ON v4_answer_units(unit_type);
CREATE INDEX IF NOT EXISTS idx_v4_answer_units_status ON v4_answer_units(verification_status);
CREATE INDEX IF NOT EXISTS idx_v4_answer_units_entities ON v4_answer_units USING GIN(about_entities);
CREATE INDEX IF NOT EXISTS idx_v4_answer_units_tags ON v4_answer_units USING GIN(tags);

-- V4 Run Summary: High-level statistics for each V4 run
CREATE TABLE IF NOT EXISTS v4_run_summary (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    response_shape TEXT,
    interpret_rounds INT DEFAULT 0,
    retrieval_rounds INT DEFAULT 0,
    units_generated INT DEFAULT 0,
    units_rendered INT DEFAULT 0,
    units_dropped INT DEFAULT 0,
    units_downgraded INT DEFAULT 0,
    verification_passed BOOLEAN,
    total_elapsed_ms REAL,
    model_version TEXT,
    cite_span_count INT DEFAULT 0,
    session_id BIGINT,
    result_set_id BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for run summary lookups
CREATE INDEX IF NOT EXISTS idx_v4_run_summary_run_id ON v4_run_summary(run_id);
CREATE INDEX IF NOT EXISTS idx_v4_run_summary_session ON v4_run_summary(session_id);
CREATE INDEX IF NOT EXISTS idx_v4_run_summary_created ON v4_run_summary(created_at DESC);

-- V4 Run Trace: Detailed trace of retrieval and interpretation rounds
CREATE TABLE IF NOT EXISTS v4_run_trace (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    round_type TEXT NOT NULL,          -- retrieval|interpret
    round_number INT NOT NULL,
    parent_round INT,                  -- For interpret rounds, which retrieval round
    chunk_count INT,
    span_count INT,
    unit_count INT,
    error_count INT,
    elapsed_ms REAL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for trace lookups
CREATE INDEX IF NOT EXISTS idx_v4_run_trace_run_id ON v4_run_trace(run_id);
CREATE INDEX IF NOT EXISTS idx_v4_run_trace_type ON v4_run_trace(round_type);

-- Comments for documentation
COMMENT ON TABLE v4_interpretations IS 'Stores V4 interpretation JSON objects from 4o reasoning model';
COMMENT ON TABLE v4_verification_reports IS 'Verification results with hard errors and soft warnings';
COMMENT ON TABLE v4_answer_units IS 'Denormalized answer units for search and analysis';
COMMENT ON TABLE v4_run_summary IS 'High-level run statistics and outcomes';
COMMENT ON TABLE v4_run_trace IS 'Detailed per-round trace for debugging and audit';

COMMENT ON COLUMN v4_answer_units.unit_type IS 'anchor: defines scope, item: list member, finding: factual statement, note: context, uncertainty: explicit gap';
COMMENT ON COLUMN v4_answer_units.confidence IS 'supported: explicit evidence, suggestive: implied evidence';
COMMENT ON COLUMN v4_answer_units.verification_status IS 'passed: all checks OK, failed: hard error, downgraded: soft warning';
