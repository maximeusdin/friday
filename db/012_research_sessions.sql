-- Day 7: Research Sessions (named container for retrieval_runs + result_sets)
-- Implements:
--   - new table: research_sessions
--   - optional session_id on retrieval_runs + result_sets
-- Notes:
--   - session is a loose container (no users/permissions/branching)
--   - session_id is nullable; ON DELETE SET NULL keeps trails even if a session is removed

BEGIN;

-- 1) Sessions table
CREATE TABLE IF NOT EXISTS research_sessions (
  id         BIGSERIAL PRIMARY KEY,
  label      TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT research_sessions_label_nonempty
    CHECK (btrim(label) <> '')
);

-- Optional: prevent duplicate labels (nice for UI “find session by name”)
CREATE UNIQUE INDEX IF NOT EXISTS research_sessions_label_uniq
ON research_sessions (label);

CREATE INDEX IF NOT EXISTS research_sessions_created_at_idx
ON research_sessions (created_at DESC);


-- 2) Add session_id to retrieval_runs (nullable)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='public'
      AND table_name='retrieval_runs'
      AND column_name='session_id'
  ) THEN
    ALTER TABLE retrieval_runs
      ADD COLUMN session_id BIGINT
      REFERENCES research_sessions(id)
      ON DELETE SET NULL;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS retrieval_runs_session_id_idx
ON retrieval_runs (session_id, created_at DESC);


-- 3) Add session_id to result_sets (nullable)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='public'
      AND table_name='result_sets'
      AND column_name='session_id'
  ) THEN
    ALTER TABLE result_sets
      ADD COLUMN session_id BIGINT
      REFERENCES research_sessions(id)
      ON DELETE SET NULL;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS result_sets_session_id_idx
ON result_sets (session_id, created_at DESC);


-- 4) Optional: ensure immutability trigger still applies cleanly
-- (No changes required; but this comment clarifies intent.)
COMMENT ON TABLE research_sessions IS
'Loose, named container for grouping retrieval_runs and result_sets. No users/permissions/branching yet.';
COMMENT ON COLUMN retrieval_runs.session_id IS
'Optional FK to research_sessions. Used to group runs into a research trail; nullable.';
COMMENT ON COLUMN result_sets.session_id IS
'Optional FK to research_sessions. Used to group saved/citable outputs into a research trail; nullable.';

COMMIT;
