-- 016_research_plans.sql
-- Store research plans (LLM-generated query plans) with status tracking

BEGIN;

CREATE TABLE IF NOT EXISTS research_plans (
  id BIGSERIAL PRIMARY KEY,
  session_id BIGINT REFERENCES research_sessions(id) ON DELETE SET NULL,
  
  -- Plan content
  plan_json JSONB NOT NULL,
  query_lang_version TEXT NOT NULL DEFAULT 'qir_v1',
  
  -- Status tracking
  status TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'approved', 'executed', 'rejected')),
  
  -- Metadata
  raw_utterance TEXT NOT NULL,
  resolved_deictics JSONB,  -- Store detected/resolved pronouns/references
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  
  -- Optional notes/review
  notes TEXT
);

CREATE INDEX IF NOT EXISTS research_plans_session_id_idx 
  ON research_plans(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS research_plans_status_idx 
  ON research_plans(status);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_research_plans_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_update_research_plans_updated_at ON research_plans;
CREATE TRIGGER trg_update_research_plans_updated_at
  BEFORE UPDATE ON research_plans
  FOR EACH ROW
  EXECUTE FUNCTION update_research_plans_updated_at();

COMMENT ON TABLE research_plans IS
'Stores LLM-generated query plans with status tracking. Plans are proposed, reviewed, then approved/executed.';
COMMENT ON COLUMN research_plans.plan_json IS
'Complete plan structure (query.primitives, compiled, etc.) as JSONB.';
COMMENT ON COLUMN research_plans.resolved_deictics IS
'JSON object storing detected pronouns/deictics and their resolved values (e.g., {"those_results": {"detected": true, "resolved_to": 123}}).';
COMMENT ON COLUMN research_plans.status IS
'Plan lifecycle: proposed (awaiting review), approved (ready to execute), executed (retrieval completed), rejected (not used).';

COMMIT;
