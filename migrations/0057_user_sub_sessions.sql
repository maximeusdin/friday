-- 0057_user_sub_sessions.sql
-- Add user_sub to research_sessions for auth scoping (Cognito sub).
-- Phase 1.5: enforce session ownership by user.

BEGIN;

-- Add column (nullable first for backfill)
ALTER TABLE research_sessions
  ADD COLUMN IF NOT EXISTS user_sub TEXT;

-- Backfill existing rows so we can set NOT NULL
UPDATE research_sessions
SET user_sub = 'pre-auth'
WHERE user_sub IS NULL;

ALTER TABLE research_sessions
  ALTER COLUMN user_sub SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_research_sessions_user_sub
  ON research_sessions(user_sub);

COMMIT;
