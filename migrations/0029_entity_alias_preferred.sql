-- Migration 0029: Create entity_alias_preferred table for human overrides
-- Allows quick fixes for problematic tokens without changing algorithms
-- Used by extract_entity_mentions.py for preferred mappings (Rule 0)

BEGIN;

CREATE TABLE IF NOT EXISTS entity_alias_preferred (
    id BIGSERIAL PRIMARY KEY,
    scope TEXT,  -- Optional: collection slug or other scope identifier (NULL = global)
    alias_norm TEXT NOT NULL,  -- Normalized alias (lowercase, normalized)
    preferred_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    note TEXT,  -- Optional note explaining why this override exists
    
    -- Ensure one preferred entity per (scope, alias_norm) combination
    -- NULL scope means global override
    UNIQUE (scope, alias_norm)
);

-- Index for fast lookups during extraction
CREATE INDEX IF NOT EXISTS idx_entity_alias_preferred_lookup 
    ON entity_alias_preferred(scope, alias_norm);

-- Index for entity lookups
CREATE INDEX IF NOT EXISTS idx_entity_alias_preferred_entity 
    ON entity_alias_preferred(preferred_entity_id);

COMMENT ON TABLE entity_alias_preferred IS
'Human overrides for entity alias collisions. When alias_norm has multiple candidate entities, '
'this table specifies which entity_id should be preferred. Scope can be NULL (global) or a '
'collection slug for collection-specific overrides.';

COMMENT ON COLUMN entity_alias_preferred.scope IS
'Optional scope identifier (e.g., collection slug). NULL means global override applies to all collections.';

COMMENT ON COLUMN entity_alias_preferred.alias_norm IS
'Normalized alias string (lowercase, normalized). Must match the alias_norm used in entity_aliases.';

COMMENT ON COLUMN entity_alias_preferred.preferred_entity_id IS
'The entity_id that should be preferred when this alias_norm has collisions.';

COMMENT ON COLUMN entity_alias_preferred.note IS
'Optional note explaining why this override exists (e.g., "doctor" refers to person not role).';

-- Example usage:
-- 
-- Fix "doctor" to prefer person entity over role:
-- INSERT INTO entity_alias_preferred (scope, alias_norm, preferred_entity_id, note)
-- VALUES (NULL, 'doctor', 12345, 'Person name, not role title');
--
-- Fix "ussr" to prefer org entity over covername (collection-specific):
-- INSERT INTO entity_alias_preferred (scope, alias_norm, preferred_entity_id, note)
-- VALUES ('venona', 'ussr', 67890, 'Organization, not codename in Venona context');
--
-- Fix "viktor" globally:
-- INSERT INTO entity_alias_preferred (scope, alias_norm, preferred_entity_id, note)
-- VALUES (NULL, 'viktor', 11111, 'Person name, not codename');
--
-- The extract_entity_mentions.py script automatically loads these overrides.
-- They are applied in Rule 0 of collision resolution (highest priority).

COMMIT;
