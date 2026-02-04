BEGIN;

-- Add GIST indexes for trigram similarity searches on concordance tables
-- These indexes enable fast fuzzy expansion queries using word_similarity()

-- NOTE: In a brand-new database, the entity system tables may not exist yet
-- at the time this migration is applied (depending on migration ordering).
-- Guard against that so "run from scratch" doesn't fail.
DO $$
BEGIN
  -- Indexes on entities.canonical_name for similarity searches
  IF to_regclass('public.entities') IS NOT NULL THEN
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_entities_canonical_name_trgm ON public.entities USING GIST (canonical_name gist_trgm_ops);';
    -- Also add index on LOWER() version for case-insensitive matching
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_entities_canonical_name_lower_trgm ON public.entities USING GIST (LOWER(canonical_name) gist_trgm_ops);';
  END IF;

  -- Indexes on entity_aliases.alias for similarity searches
  IF to_regclass('public.entity_aliases') IS NOT NULL THEN
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_trgm ON public.entity_aliases USING GIST (alias gist_trgm_ops);';
    -- Also add index on LOWER() version for case-insensitive matching
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_lower_trgm ON public.entity_aliases USING GIST (LOWER(alias) gist_trgm_ops);';
  END IF;
END $$;

-- Note: These indexes will significantly speed up fuzzy expansion queries
-- but may take some time to build if there are many entities/aliases.
-- Consider running ANALYZE after creation to update statistics.

COMMIT;
