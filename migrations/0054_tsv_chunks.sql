-- 0054_tsv_chunks.sql
-- Add chunks.tsv column and GIN index for evidence-set follow-up search.
-- Run AFTER 0054_v9_sessions_evidence.sql. Uses no transaction so that
-- CREATE INDEX CONCURRENTLY can run and progress can be monitored.
--
-- Run with a long or no statement timeout, e.g.:
--   psql "$DATABASE_URL" -v statement_timeout=0 -f migrations/0054_tsv_chunks.sql
-- Or use scripts/run_0054_tsv_with_progress.sh for progress during index build.

SET statement_timeout = 0;

-- Add tsv column if not exists (generated from text; can be slow on large chunks table)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'chunks' AND column_name = 'tsv'
  ) THEN
    ALTER TABLE chunks ADD COLUMN tsv tsvector
      GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED;
  END IF;
END $$;

-- GIN index for full-text search. CONCURRENTLY avoids long table locks.
-- Progress visible in pg_stat_progress_create_index (PostgreSQL 12+).
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_tsv_gin
  ON chunks USING GIN (tsv);
