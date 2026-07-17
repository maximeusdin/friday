-- 0054_v9_sessions_evidence.sql
-- V9 Sessions, Runs, Evidence Sets, Evidence Items, Run Steps
--
-- Supports: NEW_RETRIEVAL, FOLLOW_UP (evidence-only), THINK_DEEPER (resume)
-- See docs/v9_sessions.md for full design.

BEGIN;

-- ============================================================================
-- 1. Extend research_sessions with active pointers
-- ============================================================================

ALTER TABLE research_sessions
  ADD COLUMN IF NOT EXISTS active_run_id       BIGINT,         -- FK added after runs table
  ADD COLUMN IF NOT EXISTS active_evidence_set_id BIGINT,      -- FK added after evidence_sets table
  ADD COLUMN IF NOT EXISTS active_run_status   TEXT NOT NULL DEFAULT 'idle'
    CHECK (active_run_status IN ('idle', 'running', 'paused', 'completed', 'failed')),
  ADD COLUMN IF NOT EXISTS updated_at          TIMESTAMPTZ NOT NULL DEFAULT now();


-- ============================================================================
-- 2. evidence_sets (bounded local corpus per run)
-- ============================================================================

CREATE TABLE IF NOT EXISTS evidence_sets (
  id          BIGSERIAL PRIMARY KEY,
  session_id  BIGINT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
  run_id      BIGINT,          -- FK added after runs table
  is_active   BOOLEAN NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evidence_sets_session
  ON evidence_sets(session_id);
CREATE INDEX IF NOT EXISTS idx_evidence_sets_run
  ON evidence_sets(run_id) WHERE run_id IS NOT NULL;


-- ============================================================================
-- 3. runs (retrieval queries)
-- ============================================================================

CREATE TABLE IF NOT EXISTS v9_runs (
  id                  BIGSERIAL PRIMARY KEY,
  session_id          BIGINT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
  query_text          TEXT NOT NULL,
  query_index         INT NOT NULL DEFAULT 0,    -- monotonic per session
  label               TEXT,                       -- short title (auto or user)
  mode                TEXT NOT NULL DEFAULT 'new_retrieval'
    CHECK (mode IN ('new_retrieval', 'think_deeper')),
  status              TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running', 'paused', 'completed', 'failed')),
  last_step_idx       INT NOT NULL DEFAULT 0,
  budgets_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
  resume_state_json   JSONB,                      -- controller state for think_deeper
  evidence_set_id     BIGINT REFERENCES evidence_sets(id) ON DELETE SET NULL,
  evidence_summary    TEXT,                        -- auto-generated after completion
  top_entities_json   JSONB,                       -- small list for query reference
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT v9_runs_query_nonempty CHECK (btrim(query_text) <> '')
);

CREATE INDEX IF NOT EXISTS idx_v9_runs_session
  ON v9_runs(session_id, query_index DESC);
CREATE INDEX IF NOT EXISTS idx_v9_runs_evidence_set
  ON v9_runs(evidence_set_id) WHERE evidence_set_id IS NOT NULL;

-- Add FK from evidence_sets.run_id -> v9_runs
ALTER TABLE evidence_sets
  ADD CONSTRAINT fk_evidence_sets_run
  FOREIGN KEY (run_id) REFERENCES v9_runs(id) ON DELETE SET NULL;

-- Add FKs from research_sessions
ALTER TABLE research_sessions
  ADD CONSTRAINT fk_sessions_active_run
  FOREIGN KEY (active_run_id) REFERENCES v9_runs(id) ON DELETE SET NULL;
ALTER TABLE research_sessions
  ADD CONSTRAINT fk_sessions_active_evidence_set
  FOREIGN KEY (active_evidence_set_id) REFERENCES evidence_sets(id) ON DELETE SET NULL;


-- ============================================================================
-- 4. evidence_items (chunk references in an evidence set)
-- ============================================================================

CREATE TABLE IF NOT EXISTS evidence_items (
  id                BIGSERIAL PRIMARY KEY,
  evidence_set_id   BIGINT NOT NULL REFERENCES evidence_sets(id) ON DELETE CASCADE,
  chunk_id          BIGINT NOT NULL,               -- FK to chunks(id)
  quote_text        TEXT,                           -- optional snippet
  locators_json     JSONB NOT NULL DEFAULT '{}'::jsonb,  -- {doc_id, page, offset, source_label}
  retrieval_score   FLOAT,
  rank              INT,
  source_step_idx   INT,                           -- which run_step produced this
  is_adjacency      BOOLEAN NOT NULL DEFAULT FALSE, -- adjacency-expanded chunk
  dedup_hash        TEXT NOT NULL,                  -- sha1(evidence_set_id || chunk_id)
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Dedup: one chunk per evidence set
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_items_dedup
  ON evidence_items(evidence_set_id, dedup_hash);
-- Fast lookup by chunk
CREATE INDEX IF NOT EXISTS idx_evidence_items_set_chunk
  ON evidence_items(evidence_set_id, chunk_id);
-- Ranking
CREATE INDEX IF NOT EXISTS idx_evidence_items_set_rank
  ON evidence_items(evidence_set_id, rank)
  WHERE rank IS NOT NULL;


-- ============================================================================
-- 5. run_steps (tool trace per run)
-- ============================================================================

CREATE TABLE IF NOT EXISTS v9_run_steps (
  id                    BIGSERIAL PRIMARY KEY,
  run_id                BIGINT NOT NULL REFERENCES v9_runs(id) ON DELETE CASCADE,
  step_idx              INT NOT NULL,
  lane                  TEXT,                       -- e.g. 'search', 'fetch', 'expand'
  tool_name             TEXT NOT NULL,
  tool_args_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
  tool_result_refs_json JSONB,                      -- summary/refs, not full output
  elapsed_ms            FLOAT,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_run_steps_unique
  ON v9_run_steps(run_id, step_idx);
CREATE INDEX IF NOT EXISTS idx_run_steps_run
  ON v9_run_steps(run_id, step_idx ASC);

-- NOTE: chunks.tsv column and GIN index are in 0054_tsv_chunks.sql.
-- Run that file after this one, with a long statement_timeout (e.g. 24h or 0).
-- Use: scripts/run_0054_tsv_with_progress.sh (or .py) for progress during index build.

COMMIT;
