-- Migration 0028: Add derived alias classes for surname and acronym aliases
-- Adds 'person_last' to allowed alias_class values
-- Note: Acronym aliases use 'org' as alias_class (they're org entities)
-- This is used by the derive_surname_aliases.py script

BEGIN;

-- Drop the existing constraint
ALTER TABLE entity_aliases
  DROP CONSTRAINT IF EXISTS entity_aliases_alias_class_check;

-- Add new constraint with derived alias classes
ALTER TABLE entity_aliases
  ADD CONSTRAINT entity_aliases_alias_class_check
  CHECK (alias_class IN (
    'covername', 
    'person_given', 
    'person_full', 
    'person_last',        -- Derived surname aliases (last token of person_full names)
    'org', 
    'place', 
    'role_title', 
    'generic_word'
  ));

-- Update comment to reflect new values
COMMENT ON COLUMN entity_aliases.alias_class IS
'Alias classification: covername, person_given, person_full, person_last (derived surnames), org, place, role_title, generic_word. Used for policy defaults.';

COMMIT;
