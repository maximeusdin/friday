-- Day 10: Date Mentions (deterministic date extraction evidence)
-- Implements:
--   - date_mentions table (evidence linking dates to chunks)
-- Notes:
--   - Dates are extracted via deterministic regex patterns
--   - Mentions are evidence, not intent
--   - Supports ranges and various precision levels (day, month, year, range)

BEGIN;

-- ============================================================================
-- Date Mentions (evidence)
-- ============================================================================

CREATE TABLE IF NOT EXISTS date_mentions (
  id                  BIGSERIAL PRIMARY KEY,
  chunk_id            BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  document_id         BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,  -- Denormalized for speed
  surface             TEXT NOT NULL,  -- Exact substring as seen in source
  start_char          INT NULL,  -- Character offset in chunk (nullable for v1)
  end_char            INT NULL,
  date_start          DATE NULL,  -- Start of date range (or single date)
  date_end            DATE NULL,  -- End of date range (equals date_start for single-day)
  precision           TEXT NOT NULL CHECK (precision IN ('day', 'month', 'year', 'range', 'unknown')),
  confidence          REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
  method              TEXT NOT NULL CHECK (method IN ('regex_day', 'regex_month', 'regex_year', 'regex_range', 'human')),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  
  CONSTRAINT date_mentions_surface_nonempty
    CHECK (btrim(surface) <> ''),
  CONSTRAINT date_mentions_char_range_valid
    CHECK (start_char IS NULL OR end_char IS NULL OR (start_char >= 0 AND end_char > start_char)),
  CONSTRAINT date_mentions_date_range_valid
    CHECK (date_start IS NULL OR date_end IS NULL OR date_start <= date_end)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS date_mentions_chunk_id_idx
  ON date_mentions(chunk_id);

CREATE INDEX IF NOT EXISTS date_mentions_document_id_idx
  ON date_mentions(document_id);

CREATE INDEX IF NOT EXISTS date_mentions_date_range_idx
  ON date_mentions(date_start, date_end);

-- Optional index for review workflows (finding mentions by surface text)
CREATE INDEX IF NOT EXISTS date_mentions_surface_idx
  ON date_mentions(surface);

CREATE INDEX IF NOT EXISTS date_mentions_method_idx
  ON date_mentions(method, confidence);

-- Uniqueness constraint for idempotency
-- Prevents duplicate mentions from same extraction run
-- Note: Same date appearing multiple times in same chunk with same method = one mention
ALTER TABLE date_mentions
  ADD CONSTRAINT date_mentions_unique_mention
  UNIQUE (chunk_id, surface, date_start, date_end, method);

-- ============================================================================
-- Backfill document_id from chunk_metadata (safety net for any existing rows)
-- ============================================================================

-- Update any existing rows with missing or incorrect document_id from chunk_metadata
-- This is a safety net; extraction scripts should populate document_id directly
-- Uses the most recent pipeline_version for each chunk
UPDATE date_mentions dm
SET document_id = cm.document_id
FROM (
  SELECT DISTINCT ON (chunk_id)
    chunk_id,
    document_id
  FROM chunk_metadata
  ORDER BY chunk_id, derived_at DESC
) cm
WHERE dm.chunk_id = cm.chunk_id
  AND (dm.document_id IS NULL OR dm.document_id != cm.document_id);

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE date_mentions IS
'Evidence linking explicit date expressions to chunks. Mentions are evidence, not intent. Extracted via deterministic regex patterns.';

COMMENT ON COLUMN date_mentions.surface IS
'Exact substring as seen in source text (e.g., "23 June 1945", "1943-1945")';

COMMENT ON COLUMN date_mentions.date_start IS
'Start of date range. For single-day dates, equals date_end. For ranges, start of range.';

COMMENT ON COLUMN date_mentions.date_end IS
'End of date range. For single-day dates, equals date_start. For ranges, end of range.';

COMMENT ON COLUMN date_mentions.precision IS
'Precision level: day (exact date), month (month-year), year (year only), range (date range), unknown (could not determine)';

COMMENT ON COLUMN date_mentions.confidence IS
'Confidence score (0.0-1.0). Fixed per method: day=1.0, month=0.8, year=0.6, range=0.9.';

COMMENT ON COLUMN date_mentions.method IS
'Extraction method: regex_day, regex_month, regex_year, regex_range (deterministic patterns), or human (adjudicated).';

COMMENT ON COLUMN date_mentions.document_id IS
'Denormalized document_id from chunk_metadata for fast queries without joins.';

COMMIT;
