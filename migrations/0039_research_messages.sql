-- 0039_research_messages.sql
-- Conversation messages for research sessions
-- Supports user, assistant, and system roles with extensible metadata

BEGIN;

CREATE TABLE IF NOT EXISTS research_messages (
  id            BIGSERIAL PRIMARY KEY,
  session_id    BIGINT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
  
  role          TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content       TEXT NOT NULL,
  
  -- Optional links to plans/results for this message
  plan_id       BIGINT REFERENCES research_plans(id) ON DELETE SET NULL,
  result_set_id BIGINT REFERENCES result_sets(id) ON DELETE SET NULL,
  
  -- Extensible metadata (debug info, plan pointers, etc.)
  metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
  
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Content must be non-empty
  CONSTRAINT research_messages_content_nonempty
    CHECK (btrim(content) <> '')
);

-- Index for fetching conversation by session (chronological order)
CREATE INDEX IF NOT EXISTS idx_research_messages_session_created
  ON research_messages(session_id, created_at ASC);

-- Index for finding messages linked to a specific plan
CREATE INDEX IF NOT EXISTS idx_research_messages_plan_id
  ON research_messages(plan_id)
  WHERE plan_id IS NOT NULL;

-- Index for finding messages linked to a specific result set
CREATE INDEX IF NOT EXISTS idx_research_messages_result_set_id
  ON research_messages(result_set_id)
  WHERE result_set_id IS NOT NULL;

-- GIN index for metadata queries (if needed later)
CREATE INDEX IF NOT EXISTS idx_research_messages_metadata_gin
  ON research_messages USING GIN (metadata);

COMMIT;
