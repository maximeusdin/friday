-- 0012_retrieval_runs_tsquery_text.sql
-- Add tsquery_text column to store exact lexical query representation used at runtime

BEGIN;

ALTER TABLE retrieval_runs
  ADD COLUMN IF NOT EXISTS tsquery_text TEXT;

COMMENT ON COLUMN retrieval_runs.tsquery_text IS
'Exact tsquery string used for lexical ranking/filtering. Stored for explainability and reproducibility.';

COMMIT;
