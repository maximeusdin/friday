-- Day 9: Entity System (canonical entities, aliases, mentions, resolution)
-- Implements:
--   - entities table (canonical entities)
--   - entity_aliases table (for matching)
--   - entity_mentions table (evidence linking entities to chunks)
--   - entity_resolution_reviews table (human adjudication)
-- Notes:
--   - Entities are the only thing referenced by primitives (no ambiguous strings)
--   - Mentions are evidence, not intent
--   - Matching is deterministic 3-stage: exact alias → fuzzy alias → human review

BEGIN;

-- Ensure pg_trgm extension is available for fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- 1. Entities (canonical)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entities (
  id                  BIGSERIAL PRIMARY KEY,
  entity_type         TEXT NOT NULL CHECK (entity_type IN ('person', 'org', 'place')),
  canonical_name      TEXT NOT NULL,
  description         TEXT NULL,
  external_ids        JSONB NULL DEFAULT '{}'::jsonb,  -- Wikidata QID, VIAF, LOC, etc.
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  
  CONSTRAINT entities_canonical_name_nonempty
    CHECK (btrim(canonical_name) <> '')
);

CREATE INDEX IF NOT EXISTS entities_entity_type_idx
  ON entities(entity_type, canonical_name);

CREATE INDEX IF NOT EXISTS entities_canonical_name_idx
  ON entities(canonical_name);

-- ============================================================================
-- 2. Entity Aliases (for matching)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_aliases (
  id                  BIGSERIAL PRIMARY KEY,
  entity_id           BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  alias               TEXT NOT NULL,
  alias_norm          TEXT NOT NULL,  -- lowercased + stripped punctuation
  kind                TEXT NOT NULL DEFAULT 'alt' CHECK (kind IN ('primary', 'alt', 'misspelling', 'initials', 'ru_translit', 'code_name', 'other')),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  
  CONSTRAINT entity_aliases_alias_nonempty
    CHECK (btrim(alias) <> ''),
  CONSTRAINT entity_aliases_alias_norm_nonempty
    CHECK (btrim(alias_norm) <> ''),
  
  UNIQUE (entity_id, alias_norm)  -- One normalized alias per entity
);

-- Add missing columns if table exists but is incomplete
DO $$
BEGIN
  -- Add alias_norm column if missing
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'entity_aliases')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'entity_aliases' AND column_name = 'alias_norm') THEN
    ALTER TABLE entity_aliases ADD COLUMN alias_norm TEXT;
    -- Populate alias_norm from alias (basic normalization)
    UPDATE entity_aliases SET alias_norm = LOWER(REGEXP_REPLACE(alias, '[^\w\s]', '', 'g'));
    ALTER TABLE entity_aliases ALTER COLUMN alias_norm SET NOT NULL;
    ALTER TABLE entity_aliases ADD CONSTRAINT entity_aliases_alias_norm_nonempty CHECK (btrim(alias_norm) <> '');
  END IF;
  
  -- Add kind column if missing
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'entity_aliases')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'entity_aliases' AND column_name = 'kind') THEN
    ALTER TABLE entity_aliases ADD COLUMN kind TEXT NOT NULL DEFAULT 'alt';
    ALTER TABLE entity_aliases ADD CONSTRAINT entity_aliases_kind_check 
      CHECK (kind IN ('primary', 'alt', 'misspelling', 'initials', 'ru_translit', 'code_name', 'other'));
  END IF;
  
  -- Remove duplicates before adding unique constraint (keep row with lowest id)
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'entity_aliases')
     AND EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'entity_aliases' AND column_name = 'alias_norm') THEN
    DELETE FROM entity_aliases ea1
    WHERE EXISTS (
      SELECT 1 FROM entity_aliases ea2
      WHERE ea2.entity_id = ea1.entity_id
        AND ea2.alias_norm = ea1.alias_norm
        AND ea2.id < ea1.id
    );
  END IF;
  
  -- Add unique constraint if missing
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'entity_aliases')
     AND NOT EXISTS (
       SELECT 1 FROM pg_constraint 
       WHERE conrelid = 'entity_aliases'::regclass 
       AND conname = 'entity_aliases_entity_id_alias_norm_key'
     ) THEN
    ALTER TABLE entity_aliases ADD CONSTRAINT entity_aliases_entity_id_alias_norm_key 
      UNIQUE (entity_id, alias_norm);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS entity_aliases_alias_norm_idx
  ON entity_aliases(alias_norm);

-- Trigram index for fuzzy matching (requires pg_trgm)
CREATE INDEX IF NOT EXISTS entity_aliases_alias_norm_trgm_idx
  ON entity_aliases USING GIN (alias_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS entity_aliases_entity_id_idx
  ON entity_aliases(entity_id);

-- ============================================================================
-- 3. Entity Mentions (evidence)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_mentions (
  id                  BIGSERIAL PRIMARY KEY,
  entity_id           BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  chunk_id            BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  document_id         BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,  -- Denormalized for speed
  surface             TEXT NOT NULL,  -- Exact string in source
  start_char          INT NULL,  -- Character offset in chunk (nullable if hard to compute)
  end_char            INT NULL,
  confidence          REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
  method              TEXT NOT NULL DEFAULT 'alias_exact' CHECK (method IN ('alias_exact', 'alias_fuzzy', 'human', 'model')),
  matched_rule_id     BIGINT NULL,  -- Reference to rule/pattern that matched (optional)
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  
  CONSTRAINT entity_mentions_surface_nonempty
    CHECK (btrim(surface) <> ''),
  CONSTRAINT entity_mentions_char_range_valid
    CHECK (start_char IS NULL OR end_char IS NULL OR (start_char >= 0 AND end_char > start_char))
);

CREATE INDEX IF NOT EXISTS entity_mentions_entity_id_idx
  ON entity_mentions(entity_id, chunk_id);

CREATE INDEX IF NOT EXISTS entity_mentions_chunk_id_idx
  ON entity_mentions(chunk_id);

CREATE INDEX IF NOT EXISTS entity_mentions_document_id_idx
  ON entity_mentions(document_id);

CREATE INDEX IF NOT EXISTS entity_mentions_method_idx
  ON entity_mentions(method, confidence);

-- ============================================================================
-- 4. Entity Resolution Reviews (human adjudication)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_resolution_reviews (
  id                  BIGSERIAL PRIMARY KEY,
  surface             TEXT NOT NULL,
  context_chunk_id    BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  candidate_entity_ids BIGINT[] NOT NULL,  -- Array of candidate entity IDs
  chosen_entity_id    BIGINT NULL REFERENCES entities(id) ON DELETE SET NULL,
  decision            TEXT NOT NULL CHECK (decision IN ('accept', 'reject', 'split_new_entity')),
  reviewer            TEXT NULL,  -- Optional reviewer identifier
  reviewed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  notes               TEXT NULL,
  
  CONSTRAINT entity_resolution_reviews_surface_nonempty
    CHECK (btrim(surface) <> ''),
  CONSTRAINT entity_resolution_reviews_candidates_nonempty
    CHECK (array_length(candidate_entity_ids, 1) > 0),
  CONSTRAINT entity_resolution_reviews_accept_requires_chosen
    CHECK (decision != 'accept' OR chosen_entity_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS entity_resolution_reviews_surface_idx
  ON entity_resolution_reviews(surface);

CREATE INDEX IF NOT EXISTS entity_resolution_reviews_chunk_id_idx
  ON entity_resolution_reviews(context_chunk_id);

CREATE INDEX IF NOT EXISTS entity_resolution_reviews_reviewed_at_idx
  ON entity_resolution_reviews(reviewed_at DESC);

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE entities IS
'Canonical entities (persons, organizations, places). Only entities.id is referenced by primitives.';

COMMENT ON TABLE entity_aliases IS
'Aliases for entity matching. alias_norm is used for exact matching; supports fuzzy matching via trigram.';

COMMENT ON TABLE entity_mentions IS
'Evidence linking entities to chunks. Mentions are evidence, not intent.';

COMMENT ON TABLE entity_resolution_reviews IS
'Human adjudication for ambiguous entity matches. Enables disambiguation workflow.';

COMMIT;
