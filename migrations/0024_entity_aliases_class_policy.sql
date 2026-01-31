-- Day 10: Entity Aliases Class Policy and Enhanced Case Rules
-- Adds alias_class and enhanced case matching for covernames

BEGIN;

-- ============================================================================
-- Add alias_class and enhanced case matching columns
-- ============================================================================

ALTER TABLE entity_aliases
  ADD COLUMN IF NOT EXISTS alias_class TEXT NULL 
    CHECK (alias_class IN ('covername', 'person_given', 'person_full', 'org', 'place', 'role_title', 'generic_word')),
  ADD COLUMN IF NOT EXISTS allow_ambiguous_person_token BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS requires_context TEXT NULL;  -- Optional context gate name

-- Update match_case to support new values
ALTER TABLE entity_aliases
  DROP CONSTRAINT IF EXISTS entity_aliases_match_case_check;

ALTER TABLE entity_aliases
  ADD CONSTRAINT entity_aliases_match_case_check
  CHECK (match_case IN ('any', 'case_sensitive', 'upper_only', 'titlecase_only'));

-- Update existing match_case='case_sensitive' to be more specific if needed
-- (Keep as-is for now, can be refined later)

-- ============================================================================
-- Set defaults based on alias characteristics
-- ============================================================================

-- Infer alias_class from entity_type and alias characteristics (heuristic)
-- This can be refined manually later
UPDATE entity_aliases ea
SET alias_class = CASE
    WHEN e.entity_type = 'person' AND LENGTH(TRIM(ea.alias)) - LENGTH(REPLACE(TRIM(ea.alias), ' ', '')) = 0 THEN 'person_given'
    WHEN e.entity_type = 'person' AND LENGTH(TRIM(ea.alias)) - LENGTH(REPLACE(TRIM(ea.alias), ' ', '')) > 0 THEN 'person_full'
    WHEN e.entity_type = 'org' THEN 'org'
    WHEN e.entity_type = 'place' THEN 'place'
    ELSE NULL
END
FROM entities e
WHERE ea.entity_id = e.id
  AND ea.alias_class IS NULL;

-- Set conservative defaults for short aliases based on class
-- Covernames: allow short, require upper_only
UPDATE entity_aliases
SET match_case = 'upper_only',
    is_auto_match = true,
    notes = COALESCE(notes || '; ', '') || 'Covername: upper_only case matching'
WHERE alias_class = 'covername'
  AND match_case = 'any'
  AND LENGTH(TRIM(alias)) <= 4;

-- Generic words: disable auto-match
UPDATE entity_aliases
SET is_auto_match = false,
    notes = COALESCE(notes || '; ', '') || 'Generic word: disabled auto-match'
WHERE alias_class = 'generic_word'
  AND is_auto_match = true;

-- Person given names: disable auto-match if single token (unless explicitly allowed)
UPDATE entity_aliases ea
SET is_auto_match = false,
    notes = COALESCE(ea.notes || '; ', '') || 'Person given name: disabled auto-match (ambiguous)'
FROM entities e
WHERE ea.entity_id = e.id
  AND ea.alias_class = 'person_given'
  AND e.entity_type = 'person'
  AND LENGTH(TRIM(ea.alias)) - LENGTH(REPLACE(TRIM(ea.alias), ' ', '')) = 0
  AND ea.is_auto_match = true
  AND NOT ea.allow_ambiguous_person_token;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON COLUMN entity_aliases.alias_class IS
'Alias classification: covername, person_given, person_full, org, place, role_title, generic_word. Used for policy defaults.';

COMMENT ON COLUMN entity_aliases.allow_ambiguous_person_token IS
'If true, allows single-token person given names to auto-match even when ambiguous. Default false.';

COMMENT ON COLUMN entity_aliases.requires_context IS
'Optional context gate name (deterministic context requirement for matching). NULL = no context requirement.';

COMMENT ON COLUMN entity_aliases.match_case IS
'Case matching policy: "any" (case-insensitive), "case_sensitive" (exact case), "upper_only" (all caps required), "titlecase_only" (initial cap required).';

COMMIT;
