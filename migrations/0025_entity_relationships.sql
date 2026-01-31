-- Day 11: Entity Relationships + non-destructive alias disabling + review queue
BEGIN;

-- ============================================================
-- 0) Support columns on entity_aliases
-- ============================================================

ALTER TABLE entity_aliases
  ADD COLUMN IF NOT EXISTS is_matchable BOOLEAN NOT NULL DEFAULT true;

COMMENT ON COLUMN entity_aliases.is_matchable IS
'If false, alias is retained for provenance/audit but excluded from surface-form mention matching.';

-- Helpful partial index for mention extraction (fast alias_norm lookup)
CREATE INDEX IF NOT EXISTS entity_aliases_matchable_norm_idx
  ON entity_aliases(alias_norm)
  WHERE is_matchable = true;

-- ============================================================
-- 1) entity_relationships table
-- ============================================================

CREATE TABLE IF NOT EXISTS entity_relationships (
  id BIGSERIAL PRIMARY KEY,
  source_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  target_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  relationship_type TEXT NOT NULL CHECK (relationship_type IN (
    'covername_of',      -- covername → person (DOUGLAS → Joseph Katz)
    'alias_of',          -- future: alias entity → canonical entity
    'member_of',         -- person → org
    'located_in',        -- place → place
    'same_as',           -- entity → entity (dedupe)
    'derived_from'       -- provenance edge
  )),
  confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
  source TEXT,           -- provenance label (concordance_backfill, manual, NER, etc.)
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (source_entity_id, target_entity_id, relationship_type),
  CHECK (source_entity_id <> target_entity_id)
);

CREATE INDEX IF NOT EXISTS entity_relationships_source_idx
  ON entity_relationships(source_entity_id);

CREATE INDEX IF NOT EXISTS entity_relationships_target_idx
  ON entity_relationships(target_entity_id);

CREATE INDEX IF NOT EXISTS entity_relationships_type_idx
  ON entity_relationships(relationship_type);

CREATE INDEX IF NOT EXISTS entity_relationships_type_target_idx
  ON entity_relationships(relationship_type, target_entity_id);

COMMENT ON TABLE entity_relationships IS
'Graph edges between entities. Separates identity claims (relationships) from surface-form matching (aliases).';

-- ============================================================
-- 2) updated_at trigger
-- ============================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_entity_relationships_updated_at ON entity_relationships;

CREATE TRIGGER trg_entity_relationships_updated_at
BEFORE UPDATE ON entity_relationships
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- ============================================================
-- 3) Backfill review queue (ambiguity-safe)
-- ============================================================

CREATE TABLE IF NOT EXISTS entity_relationship_backfill_queue (
  id BIGSERIAL PRIMARY KEY,
  covername_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  alias_id BIGINT NOT NULL REFERENCES entity_aliases(id) ON DELETE CASCADE,
  person_alias TEXT NOT NULL,
  person_alias_norm TEXT NOT NULL,
  candidate_person_entity_ids BIGINT[] NULL,
  status TEXT NOT NULL DEFAULT 'needs_review' CHECK (status IN (
    'needs_review',
    'resolved',
    'ignored'
  )),
  proposed_relationship_type TEXT NOT NULL DEFAULT 'covername_of',
  source TEXT NOT NULL DEFAULT 'concordance_backfill',
  notes TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (covername_entity_id, alias_id, proposed_relationship_type)
);

CREATE INDEX IF NOT EXISTS erbq_status_idx
  ON entity_relationship_backfill_queue(status);

CREATE INDEX IF NOT EXISTS erbq_covername_idx
  ON entity_relationship_backfill_queue(covername_entity_id);

CREATE INDEX IF NOT EXISTS erbq_alias_norm_idx
  ON entity_relationship_backfill_queue(person_alias_norm);

DROP TRIGGER IF EXISTS trg_erbq_updated_at ON entity_relationship_backfill_queue;

CREATE TRIGGER trg_erbq_updated_at
BEFORE UPDATE ON entity_relationship_backfill_queue
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

COMMIT;
