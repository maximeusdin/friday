-- 0069_output_mode_preference.sql
-- Add session-level output mode preference for Chat.
-- evidence_only: default for historian persona, minimal widgets
-- evidence_summary: evidence + short summary
-- narrative: full narrative report with timeline/findings

ALTER TABLE research_sessions
  ADD COLUMN IF NOT EXISTS output_mode TEXT NOT NULL DEFAULT 'evidence_only'
    CHECK (output_mode IN ('evidence_only', 'evidence_summary', 'narrative'));
