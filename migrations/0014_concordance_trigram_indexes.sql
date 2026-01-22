BEGIN;

-- Add GIST indexes for trigram similarity searches on concordance tables
-- These indexes enable fast fuzzy expansion queries using word_similarity()

-- Index on entities.canonical_name for similarity searches
CREATE INDEX IF NOT EXISTS idx_entities_canonical_name_trgm 
  ON entities USING GIST (canonical_name gist_trgm_ops);

-- Index on entity_aliases.alias for similarity searches  
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_trgm
  ON entity_aliases USING GIST (alias gist_trgm_ops);

-- Also add indexes on LOWER() versions for case-insensitive matching
-- (PostgreSQL can use these when queries use LOWER())
CREATE INDEX IF NOT EXISTS idx_entities_canonical_name_lower_trgm
  ON entities USING GIST (LOWER(canonical_name) gist_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_lower_trgm
  ON entity_aliases USING GIST (LOWER(alias) gist_trgm_ops);

-- Note: These indexes will significantly speed up fuzzy expansion queries
-- but may take some time to build if there are many entities/aliases.
-- Consider running ANALYZE after creation to update statistics.

COMMIT;
