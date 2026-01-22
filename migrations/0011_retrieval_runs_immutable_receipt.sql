-- 0011_retrieval_runs_immutable_receipt.sql
-- Add forward-compatible versioning + immutable config receipt fields to retrieval_runs.

BEGIN;

ALTER TABLE retrieval_runs
  ADD COLUMN IF NOT EXISTS query_lang_version TEXT NOT NULL DEFAULT 'qv1',
  ADD COLUMN IF NOT EXISTS retrieval_impl_version TEXT NOT NULL DEFAULT 'retrieval_v1',
  ADD COLUMN IF NOT EXISTS normalization_version TEXT,

  -- immutable receipt of runtime config (even if you later add a normalized config table)
  ADD COLUMN IF NOT EXISTS retrieval_config_json JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- vector-specific stability fields
  ADD COLUMN IF NOT EXISTS vector_metric TEXT,
  ADD COLUMN IF NOT EXISTS embedding_dim INT,
  ADD COLUMN IF NOT EXISTS embed_text_version TEXT;

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_query_lang_version
  ON retrieval_runs(query_lang_version);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_retrieval_impl_version
  ON retrieval_runs(retrieval_impl_version);

COMMIT;

