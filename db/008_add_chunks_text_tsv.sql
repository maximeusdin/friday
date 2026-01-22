-- Adds a lexical search column + index for fast "all occurrences" + co-occurrence + hybrid.

BEGIN;

-- 1) Add column
ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS text_tsv tsvector;

-- 2) Backfill for your active chunk pipeline(s)
-- If you only want chunk_v1_full now:
UPDATE chunks
SET text_tsv = to_tsvector('simple', COALESCE(text, ''))
WHERE pipeline_version = 'chunk_v1_full'
  AND (text_tsv IS NULL OR text_tsv = ''::tsvector);

-- 3) Index (GIN)
CREATE INDEX IF NOT EXISTS chunks_text_tsv_gin_idx
ON chunks USING gin (text_tsv);

-- 4) Helpful partial index for common filtering by pipeline_version (optional)
CREATE INDEX IF NOT EXISTS chunks_pv_idx
ON chunks (pipeline_version);

COMMIT;

-- Analyze for planner stats
ANALYZE chunks;
