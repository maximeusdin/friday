-- 0056_scope_support.sql
-- Add session-level scope persistence and per-run scope tracking.
--
-- scope_json: user-selected scope stored on each session (Full archive default).
-- run_scope_json: scope actually used per run + expansion metadata (audit trail).

BEGIN;

-- Session-level user-selected scope (persists across queries)
ALTER TABLE research_sessions
  ADD COLUMN IF NOT EXISTS scope_json JSONB NOT NULL DEFAULT '{"mode": "full_archive"}'::jsonb;

-- Per-run scope actually used + expansion metadata (audit trail)
ALTER TABLE v9_runs
  ADD COLUMN IF NOT EXISTS run_scope_json JSONB;

COMMIT;
