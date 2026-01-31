-- OCR Pipeline V2: Idempotency + Evidence Improvements
-- Adds unique constraints and evidence payload columns

-- =============================================================================
-- 1. UNIQUE CONSTRAINTS FOR IDEMPOTENCY
-- =============================================================================

-- mention_candidates: unique on (batch_id, chunk_id, char_start, char_end, surface_norm)
-- Prevents duplicate candidates within a run
CREATE UNIQUE INDEX IF NOT EXISTS idx_mention_candidates_unique_span
    ON mention_candidates (batch_id, chunk_id, char_start, char_end, surface_norm)
    WHERE batch_id IS NOT NULL;

-- entity_mentions: unique on (chunk_id, entity_id, start_char, end_char)
-- Prevents duplicate mentions for same entity at same position
-- (May already exist, so use IF NOT EXISTS)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'entity_mentions_unique_position'
    ) THEN
        ALTER TABLE entity_mentions 
        ADD CONSTRAINT entity_mentions_unique_position 
        UNIQUE (chunk_id, entity_id, start_char, end_char);
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- mention_review_queue: unique on (chunk_id, surface_norm, start_char, end_char)
-- Prevents duplicate queue items for same surface at same position
CREATE UNIQUE INDEX IF NOT EXISTS idx_mention_review_queue_unique_span
    ON mention_review_queue (chunk_id, surface_norm, start_char, end_char)
    WHERE chunk_id IS NOT NULL AND surface_norm IS NOT NULL;

-- =============================================================================
-- 2. EVIDENCE PAYLOAD COLUMNS
-- =============================================================================

-- Add candidate_scores array to review queue
ALTER TABLE mention_review_queue 
ADD COLUMN IF NOT EXISTS candidate_scores NUMERIC[];

-- Add resolution_signals JSONB for detailed evidence
ALTER TABLE mention_review_queue 
ADD COLUMN IF NOT EXISTS resolution_signals JSONB;

-- Add similar columns to mention_candidates for audit trail
ALTER TABLE mention_candidates 
ADD COLUMN IF NOT EXISTS resolution_signals JSONB;

-- =============================================================================
-- 3. JUNK LEARNING SUPPORT
-- =============================================================================

-- Add columns to track where junk patterns came from
ALTER TABLE ocr_junk_patterns 
ADD COLUMN IF NOT EXISTS source_document_id BIGINT;

ALTER TABLE ocr_junk_patterns 
ADD COLUMN IF NOT EXISTS source_surface_norm TEXT;

ALTER TABLE ocr_junk_patterns 
ADD COLUMN IF NOT EXISTS rejection_count INTEGER DEFAULT 0;

-- Index for quick lookup during candidate generation
CREATE INDEX IF NOT EXISTS idx_ocr_junk_patterns_surface 
    ON ocr_junk_patterns (pattern_value) 
    WHERE pattern_type = 'exact' AND is_active = TRUE;

-- =============================================================================
-- 4. OCR EXTRACTION RUNS ENHANCEMENTS
-- =============================================================================

-- Add more tracking columns
ALTER TABLE ocr_extraction_runs 
ADD COLUMN IF NOT EXISTS candidates_existing INTEGER DEFAULT 0;

ALTER TABLE ocr_extraction_runs 
ADD COLUMN IF NOT EXISTS resolution_started_at TIMESTAMPTZ;

ALTER TABLE ocr_extraction_runs 
ADD COLUMN IF NOT EXISTS resolution_completed_at TIMESTAMPTZ;

-- =============================================================================
-- 5. PERFORMANCE INDEXES
-- =============================================================================

-- For batch retrieval: faster LATERAL join
CREATE INDEX IF NOT EXISTS idx_alias_lexicon_trgm_btree
    ON alias_lexicon_index (alias_norm text_pattern_ops);

-- For resolution status queries
CREATE INDEX IF NOT EXISTS idx_mention_candidates_batch_status
    ON mention_candidates (batch_id, resolution_status);

-- =============================================================================
-- Done
-- =============================================================================

SELECT 'OCR idempotency and evidence improvements applied' AS status;
