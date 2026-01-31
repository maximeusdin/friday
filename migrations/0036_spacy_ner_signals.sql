-- Migration: 0036_spacy_ner_signals.sql
-- Adds NER signal columns to mention_candidates and tracking table for NER runs
-- 
-- NER signals are stored as supplementary information, NOT overwriting existing scores.
-- quality_score = OCR span quality
-- ner_accept_score = NER confidence + context gating (separate concept)

-- =============================================================================
-- NER signal columns on mention_candidates
-- =============================================================================

ALTER TABLE mention_candidates 
ADD COLUMN IF NOT EXISTS ner_source VARCHAR(20),
ADD COLUMN IF NOT EXISTS ner_label VARCHAR(20),
ADD COLUMN IF NOT EXISTS ner_type_hint VARCHAR(20),
ADD COLUMN IF NOT EXISTS ner_accept_score FLOAT,
ADD COLUMN IF NOT EXISTS ner_context_features JSONB;

COMMENT ON COLUMN mention_candidates.ner_source IS 'Source of NER signal: spacy, heuristic, both';
COMMENT ON COLUMN mention_candidates.ner_label IS 'SpaCy NER label: PERSON, ORG, GPE, LOC, FAC';
COMMENT ON COLUMN mention_candidates.ner_type_hint IS 'Mapped type: person, org, place';
COMMENT ON COLUMN mention_candidates.ner_accept_score IS 'NER acceptance score (0-1), separate from quality_score';
COMMENT ON COLUMN mention_candidates.ner_context_features IS 'Context hints JSON: {person: N, org: N, loc: N}';

-- =============================================================================
-- Chunk NER runs tracking table
-- Tracks which chunks have been processed by which NER model/config
-- Enables proper skip-processed logic and reproducibility
-- =============================================================================

CREATE TABLE IF NOT EXISTS chunk_ner_runs (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model VARCHAR(50) NOT NULL,
    threshold FLOAT NOT NULL,
    pipeline_version VARCHAR(20) DEFAULT '1.0',
    -- Detailed stats for debugging (all with defaults for easy inserts)
    spans_extracted INTEGER NOT NULL DEFAULT 0,      -- Raw spans from spaCy
    spans_upserted INTEGER NOT NULL DEFAULT 0,       -- Candidates upserted (insert or update)
    spans_enhanced_existing INTEGER NOT NULL DEFAULT 0, -- Existing candidates that got NER signals
    spans_dropped_overlap INTEGER NOT NULL DEFAULT 0,   -- Dropped due to overlap with existing
    spans_dropped_filters INTEGER NOT NULL DEFAULT 0,   -- Dropped due to threshold + lowercase filters
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'completed',
    
    UNIQUE (chunk_id, model, threshold, pipeline_version)
);

CREATE INDEX IF NOT EXISTS idx_chunk_ner_runs_chunk ON chunk_ner_runs(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_ner_runs_model ON chunk_ner_runs(model);

COMMENT ON TABLE chunk_ner_runs IS 'Tracks NER processing runs per chunk for skip-processed logic';

-- =============================================================================
-- Index for NER queries
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_mention_candidates_ner_source 
ON mention_candidates(ner_source) WHERE ner_source IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mention_candidates_ner_label 
ON mention_candidates(ner_label) WHERE ner_label IS NOT NULL;

-- =============================================================================
-- Unique constraint for idempotent NER inserts
-- This is a REAL constraint (not partial) so ON CONFLICT works correctly.
-- Covers all candidates regardless of batch_id or ner_source.
-- =============================================================================

-- First, drop the partial index if it exists (it won't work with ON CONFLICT)
DROP INDEX IF EXISTS idx_mention_candidates_ner_unique_span;

-- Create a real unique constraint for ON CONFLICT to use
-- Note: This may fail if there are existing duplicates. Clean up first if needed.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'mention_candidates_span_unique'
    ) THEN
        ALTER TABLE mention_candidates 
        ADD CONSTRAINT mention_candidates_span_unique 
        UNIQUE (chunk_id, char_start, char_end, surface_norm);
    END IF;
EXCEPTION
    WHEN unique_violation THEN
        RAISE NOTICE 'Duplicate spans exist. Run deduplication before adding constraint.';
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint already exists.';
END $$;
