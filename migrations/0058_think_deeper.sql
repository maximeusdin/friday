-- 0058: Think Deeper tables (Actor/Judge architecture)
-- Tracks autonomous Think Deeper runs and per-step traces for debugging.

BEGIN;

-- ── think_deeper_runs ───────────────────────────────────────────────────────
-- One row per Think Deeper invocation.  Links to the parent v9_run.

CREATE TABLE IF NOT EXISTS think_deeper_runs (
    id              BIGSERIAL PRIMARY KEY,
    v9_run_id       BIGINT NOT NULL REFERENCES v9_runs(id),
    directive_json  JSONB NOT NULL,
    final_scores_json JSONB,
    stop_reason     TEXT,
    steps_executed  INT,
    tool_calls_used INT,
    elapsed_ms      FLOAT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_td_runs_v9_run ON think_deeper_runs (v9_run_id);


-- ── think_deeper_steps ──────────────────────────────────────────────────────
-- One row per controller loop step.  Full trace for debugging.

CREATE TABLE IF NOT EXISTS think_deeper_steps (
    id                  BIGSERIAL PRIMARY KEY,
    td_run_id           BIGINT NOT NULL REFERENCES think_deeper_runs(id) ON DELETE CASCADE,
    step_idx            INT NOT NULL,
    -- Actor output (all proposals)
    actor_proposals_json JSONB,
    -- Selected action (after Judge selection)
    action_json         JSONB NOT NULL,
    -- Execution results
    tool_calls_json     JSONB,
    candidates_count    INT,
    -- Rails filtering
    rails_report_json   JSONB,
    admitted_count      INT,
    -- Judge verdict (scores, gaps, findings, stop rec)
    judge_verdict_json  JSONB NOT NULL,
    -- Working set after this step
    selected_chunk_ids  INT[],
    -- FindingStore delta
    new_findings_count  INT DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_td_steps_run ON think_deeper_steps (td_run_id, step_idx);

COMMIT;
