-- 0061_entities_entity_type_concordance.sql
-- Allow concordance ingest entity_type values: cover_name, topic, other
-- (original 0019 only allowed person, org, place)

BEGIN;

-- Drop the original check (name from 0019_entities.sql)
ALTER TABLE entities DROP CONSTRAINT IF EXISTS entities_entity_type_check;

-- Re-add with concordance types included
ALTER TABLE entities ADD CONSTRAINT entities_entity_type_check
  CHECK (entity_type IN ('person', 'org', 'place', 'cover_name', 'topic', 'other'));

COMMIT;
