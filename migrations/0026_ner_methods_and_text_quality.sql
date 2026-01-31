-- Migration 0026: Add NER methods and text quality support
-- Adds support for pattern-based, NER-based, and hybrid entity extraction
-- Adds text_quality column to chunks for OCR vs clean text classification

BEGIN;

-- ============================================================================
-- 1. Add text_quality to chunks table
-- ============================================================================

ALTER TABLE chunks 
  ADD COLUMN IF NOT EXISTS text_quality TEXT 
  CHECK (text_quality IN ('ocr', 'clean', 'unknown')) 
  DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS idx_chunks_text_quality 
  ON chunks(text_quality);

-- ============================================================================
-- 2. Update entity_mentions method constraint to include new NER methods
-- ============================================================================

-- Drop existing constraint
ALTER TABLE entity_mentions 
  DROP CONSTRAINT IF EXISTS entity_mentions_method_check;

-- Add new constraint with all methods
ALTER TABLE entity_mentions
  ADD CONSTRAINT entity_mentions_method_check
  CHECK (method IN (
    'alias_exact',      -- From concordance matching (exact)
    'alias_partial',    -- From concordance matching (partial)
    'alias_fuzzy',      -- From concordance matching (fuzzy)
    'pattern_based',    -- Rule-based pattern matching
    'ner_spacy',        -- SpaCy NER model
    'ner_transformer',  -- Transformer-based NER
    'fuzzy_known',      -- Fuzzy match against known entities
    'hybrid',           -- Combined approach
    'human',            -- Manual review
    'model'             -- Legacy/other model (keep for backward compatibility)
  ));

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON COLUMN chunks.text_quality IS
'Text quality classification: ocr (OCR-scanned text with errors), clean (high-quality text), unknown (not yet classified)';

COMMENT ON COLUMN entity_mentions.method IS
'Extraction method: alias_exact/partial/fuzzy (concordance-based), pattern_based (regex), ner_spacy/transformer (statistical NER), fuzzy_known (fuzzy match against known entities), hybrid (combined), human (manual review)';

COMMIT;
