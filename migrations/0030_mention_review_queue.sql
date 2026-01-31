-- Day 10: Mention Review Queue (adjudication workflow)
-- Implements:
--   - mention_review_queue table (stores ambiguous/non-deterministic mentions for human review)
-- Notes:
--   - Supports both entity and date mentions
--   - Stores candidates and context for review
--   - Tracks review status and decisions

BEGIN;

-- ============================================================================
-- Mention Review Queue
-- ============================================================================

CREATE TABLE IF NOT EXISTS mention_review_queue (
  id                  BIGSERIAL PRIMARY KEY,
  mention_type        TEXT NOT NULL CHECK (mention_type IN ('entity', 'date')),
  chunk_id            BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  document_id          BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,  -- Denormalized for convenience
  surface              TEXT NOT NULL,  -- Exact substring as seen
  start_char           INT NULL,  -- Character offset in chunk
  end_char             INT NULL,
  context_excerpt      TEXT NOT NULL,  -- Short slice of surrounding text (e.g., ±100 chars)
  candidates           JSONB NOT NULL,  -- For entity: [{"entity_id": 123, "canonical_name": "...", "score": 0.8}, ...]
                                         -- For date: [{"date_start": "1945-06-23", "date_end": "1945-06-23", "precision": "day", "confidence": 1.0}, ...]
  status               TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'rejected')),
  decision             JSONB NULL,  -- For entity: {"entity_id": 123, "alias_norm": "..."}
                                    -- For date: {"date_start": "1945-06-23", "date_end": "1945-06-23", "precision": "day"}
  note                 TEXT NULL,  -- Optional note from reviewer
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  reviewed_at          TIMESTAMPTZ NULL,
  
  CONSTRAINT mention_review_queue_surface_nonempty
    CHECK (btrim(surface) <> ''),
  CONSTRAINT mention_review_queue_context_nonempty
    CHECK (btrim(context_excerpt) <> ''),
  CONSTRAINT mention_review_queue_char_range_valid
    CHECK (start_char IS NULL OR end_char IS NULL OR (start_char >= 0 AND end_char > start_char))
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS mention_review_queue_status_idx
  ON mention_review_queue(status, mention_type);

CREATE INDEX IF NOT EXISTS mention_review_queue_chunk_id_idx
  ON mention_review_queue(chunk_id);

CREATE INDEX IF NOT EXISTS mention_review_queue_document_id_idx
  ON mention_review_queue(document_id);

CREATE INDEX IF NOT EXISTS mention_review_queue_created_at_idx
  ON mention_review_queue(created_at DESC);

-- Index for filtering by mention_type and status (most common query)
CREATE INDEX IF NOT EXISTS mention_review_queue_type_status_idx
  ON mention_review_queue(mention_type, status, created_at DESC);

-- GIN index for JSONB candidates queries (useful for finding reviews by entity_id)
CREATE INDEX IF NOT EXISTS mention_review_queue_candidates_gin_idx
  ON mention_review_queue USING GIN (candidates);

-- Unique constraint to prevent duplicate pending review items
-- Same mention in same chunk with same candidates = duplicate
-- Note: Using md5 hash of candidates JSONB for uniqueness (PostgreSQL limitation)
CREATE UNIQUE INDEX IF NOT EXISTS mention_review_queue_unique_pending
  ON mention_review_queue (chunk_id, surface, mention_type, md5(candidates::text))
  WHERE status = 'pending';

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE mention_review_queue IS
'Queue of ambiguous or non-deterministic mentions requiring human review/adjudication. Supports both entity and date mentions.';

COMMENT ON COLUMN mention_review_queue.mention_type IS
'Type of mention: "entity" (person/org/place/covername) or "date" (date expression).';

COMMENT ON COLUMN mention_review_queue.surface IS
'Exact substring as seen in source text (e.g., "Yakubovich", "23 June 1945").';

COMMENT ON COLUMN mention_review_queue.context_excerpt IS
'Short slice of surrounding text (±100 chars) to provide context for review.';

COMMENT ON COLUMN mention_review_queue.candidates IS
'JSONB array of candidate matches. For entities: [{"entity_id": 123, "canonical_name": "...", "score": 0.8}, ...]. For dates: [{"date_start": "1945-06-23", "date_end": "1945-06-23", "precision": "day", "confidence": 1.0}, ...].';

COMMENT ON COLUMN mention_review_queue.status IS
'Review status: "pending" (awaiting review), "accepted" (approved with decision), "rejected" (not a valid mention).';

COMMENT ON COLUMN mention_review_queue.decision IS
'JSONB decision data. For entities: {"entity_id": 123, "alias_norm": "..."}. For dates: {"date_start": "1945-06-23", "date_end": "1945-06-23", "precision": "day"}.';

COMMENT ON COLUMN mention_review_queue.note IS
'Optional note from reviewer explaining the decision.';

COMMIT;
