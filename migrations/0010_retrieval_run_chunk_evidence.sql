-- 0010_retrieval_run_chunk_evidence.sql
-- Persist per-chunk evidence at retrieval time (explainable + export-ready)

BEGIN;

CREATE TABLE IF NOT EXISTS retrieval_run_chunk_evidence (
  retrieval_run_id BIGINT NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
  chunk_id         BIGINT NOT NULL REFERENCES chunks(id) ON DELETE RESTRICT,

  rank             INT NOT NULL,

  score_lex        REAL NULL,
  score_vec        REAL NULL,
  score_hybrid     REAL NULL,

  matched_lexemes  TEXT[] NULL,
  highlight        TEXT NULL,
  explain_json     JSONB NULL,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT retrieval_run_chunk_evidence_pk
    PRIMARY KEY (retrieval_run_id, chunk_id),

  CONSTRAINT retrieval_run_chunk_evidence_rank_positive
    CHECK (rank > 0)
);

-- Optional but nice: one rank per run
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conrelid = 'retrieval_run_chunk_evidence'::regclass
      AND conname = 'retrieval_run_chunk_evidence_run_rank_uniq'
  ) THEN
    ALTER TABLE retrieval_run_chunk_evidence
      ADD CONSTRAINT retrieval_run_chunk_evidence_run_rank_uniq
      UNIQUE (retrieval_run_id, rank);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_rrce_run_rank
  ON retrieval_run_chunk_evidence(retrieval_run_id, rank);

CREATE INDEX IF NOT EXISTS idx_rrce_chunk_id
  ON retrieval_run_chunk_evidence(chunk_id);

COMMIT;

