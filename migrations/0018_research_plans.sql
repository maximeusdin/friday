-- Day 9: Research Plans (deterministic planning layer for conversational query refinement)
-- Implements:
--   - new table: research_plans
--   - immutable plan artifacts with explicit status tracking
--   - parent-child relationships for plan revisions
-- Notes:
--   - Plans are immutable once created
--   - Revisions create new rows via parent_plan_id
--   - Only approved plans may be executed
--   - plan_json is the authoritative record of intent

BEGIN;

-- Research plans table
CREATE TABLE IF NOT EXISTS research_plans (
  id                  BIGSERIAL PRIMARY KEY,
  session_id          BIGINT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  
  status              TEXT NOT NULL CHECK (status IN ('proposed', 'approved', 'executed', 'superseded', 'rejected')),
  
  user_utterance     TEXT NOT NULL,
  
  plan_json           JSONB NOT NULL,
  plan_hash           TEXT NOT NULL,
  
  query_lang_version  TEXT NOT NULL,
  retrieval_impl_version TEXT NOT NULL,
  
  parent_plan_id      BIGINT NULL REFERENCES research_plans(id) ON DELETE SET NULL,
  
  approved_at         TIMESTAMPTZ NULL,
  executed_at         TIMESTAMPTZ NULL,
  
  retrieval_run_id    BIGINT NULL REFERENCES retrieval_runs(id) ON DELETE SET NULL,
  result_set_id       BIGINT NULL REFERENCES result_sets(id) ON DELETE SET NULL,
  
  CONSTRAINT research_plans_user_utterance_nonempty
    CHECK (btrim(user_utterance) <> ''),
  CONSTRAINT research_plans_plan_hash_nonempty
    CHECK (btrim(plan_hash) <> ''),
  CONSTRAINT research_plans_approved_only_when_approved
    CHECK (approved_at IS NULL OR status IN ('approved', 'executed')),
  CONSTRAINT research_plans_executed_only_when_executed
    CHECK (executed_at IS NULL OR status = 'executed'),
  CONSTRAINT research_plans_executed_requires_approved
    CHECK (executed_at IS NULL OR approved_at IS NOT NULL),
  CONSTRAINT research_plans_retrieval_run_only_when_executed
    CHECK (retrieval_run_id IS NULL OR status = 'executed'),
  CONSTRAINT research_plans_result_set_only_when_executed
    CHECK (result_set_id IS NULL OR status = 'executed')
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS research_plans_session_id_idx
  ON research_plans(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS research_plans_status_idx
  ON research_plans(status, created_at DESC);

CREATE INDEX IF NOT EXISTS research_plans_parent_plan_id_idx
  ON research_plans(parent_plan_id);

CREATE INDEX IF NOT EXISTS research_plans_plan_hash_idx
  ON research_plans(plan_hash);

CREATE INDEX IF NOT EXISTS research_plans_retrieval_run_id_idx
  ON research_plans(retrieval_run_id) WHERE retrieval_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS research_plans_result_set_id_idx
  ON research_plans(result_set_id) WHERE result_set_id IS NOT NULL;

-- GIN index for plan_json queries
CREATE INDEX IF NOT EXISTS research_plans_plan_json_gin_idx
  ON research_plans USING GIN (plan_json);

COMMENT ON TABLE research_plans IS
'Immutable research plans representing user intent as explicit primitive queries. Plans are compiled deterministically and executed only after approval.';

COMMENT ON COLUMN research_plans.status IS
'Plan lifecycle: proposed → approved → executed. Revisions create new rows with parent_plan_id.';

COMMENT ON COLUMN research_plans.plan_json IS
'Authoritative record of intent: explicit primitive query representation (deterministic compilation target).';

COMMENT ON COLUMN research_plans.plan_hash IS
'Deterministic hash of plan_json for deduplication and change detection.';

COMMENT ON COLUMN research_plans.parent_plan_id IS
'Links to previous version when plan is revised. Enables plan history/revision tracking.';

COMMIT;
