-- Migration: 0037_ner_date_extraction.sql
-- Creates NER date signals table (separate from date_mentions)
-- NER is a signal source, NOT a primary producer of date_mentions

BEGIN;

-- =============================================================================
-- NER Date Signals Table
-- Stores spaCy DATE/TIME extractions as signals for review/verification
-- NOT written directly to date_mentions (which requires deterministic rules)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ner_date_signals (
    id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Span location
    surface TEXT NOT NULL,
    surface_norm TEXT NOT NULL,  -- Normalized for deduplication
    start_char INT NOT NULL,
    end_char INT NOT NULL,
    
    -- NER extraction info
    ner_label VARCHAR(20) NOT NULL,  -- DATE or TIME
    ner_model VARCHAR(50) NOT NULL,
    
    -- Parser details (for debugging normalization)
    raw_parser VARCHAR(50) NOT NULL DEFAULT 'spacy_ent',  -- spacy_ent, dateparser, regex_v2, etc.
    parser_payload JSONB NULL,  -- Original match groups, normalization steps
    
    -- Parsed date info (best effort, may be NULL or ambiguous)
    parsed_date_start DATE NULL,
    parsed_date_end DATE NULL,
    parsed_precision VARCHAR(20) NOT NULL DEFAULT 'unknown',  -- day, month, year, range, ambiguous_numeric, unknown
    parse_confidence REAL NOT NULL DEFAULT 0.5,
    
    -- Review status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, verified, rejected, promoted
    promoted_to_date_mention_id BIGINT NULL REFERENCES date_mentions(id),
    
    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ NULL,
    reviewed_by TEXT NULL,
    
    -- Constraints
    CONSTRAINT ner_date_signals_surface_nonempty CHECK (btrim(surface) <> ''),
    CONSTRAINT ner_date_signals_char_range_valid CHECK (start_char >= 0 AND end_char > start_char),
    CONSTRAINT ner_date_signals_status_valid CHECK (status IN ('pending', 'verified', 'rejected', 'promoted'))
);

-- Named unique constraint for ON CONFLICT (required for idempotent upserts)
ALTER TABLE ner_date_signals 
    ADD CONSTRAINT ner_date_signals_span_unique 
    UNIQUE (chunk_id, start_char, end_char, surface_norm);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ner_date_signals_chunk ON ner_date_signals(chunk_id);
CREATE INDEX IF NOT EXISTS idx_ner_date_signals_document ON ner_date_signals(document_id);
CREATE INDEX IF NOT EXISTS idx_ner_date_signals_status ON ner_date_signals(status);
CREATE INDEX IF NOT EXISTS idx_ner_date_signals_parsed ON ner_date_signals(parsed_date_start, parsed_date_end) 
    WHERE parsed_date_start IS NOT NULL;

-- =============================================================================
-- Extend chunk_ner_runs with date extraction stats
-- =============================================================================

-- Add date extraction tracking columns to chunk_ner_runs
ALTER TABLE chunk_ner_runs 
    ADD COLUMN IF NOT EXISTS date_spans_extracted INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS date_spans_saved INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS dates_enabled BOOLEAN NOT NULL DEFAULT FALSE;

-- Comments
COMMENT ON TABLE ner_date_signals IS 
'NER-extracted date signals. These are NOT authoritative - they require verification before promotion to date_mentions.';

COMMENT ON COLUMN ner_date_signals.status IS
'pending: needs review, verified: correct but not promoted, rejected: false positive, promoted: inserted into date_mentions';

COMMENT ON COLUMN ner_date_signals.promoted_to_date_mention_id IS
'If promoted, references the created date_mentions row for audit trail';

COMMENT ON COLUMN ner_date_signals.raw_parser IS
'Parser that produced the normalization: spacy_ent, dateparser, regex_v2, etc.';

COMMENT ON COLUMN ner_date_signals.parser_payload IS
'JSONB with original match groups, normalization steps for debugging';

COMMENT ON COLUMN chunk_ner_runs.date_spans_extracted IS
'Number of DATE/TIME spans from spaCy before filtering';

COMMENT ON COLUMN chunk_ner_runs.date_spans_saved IS
'Number of date signals actually inserted to ner_date_signals';

COMMIT;
