-- Performance optimization for entity_aliases loading
-- Addresses slow performance in extract_entity_mentions.py load_all_aliases()
--
-- Issues addressed:
-- 1. Missing index on is_matchable (causes full table scan)
-- 2. Missing composite index for main query (ORDER BY ea.entity_id, ea.id)
-- 3. Inefficient "unidentified" query pattern matching

BEGIN;

-- ============================================================================
-- 1. Partial index on is_matchable (only indexes rows where is_matchable = true)
-- ============================================================================
-- This dramatically speeds up the WHERE ea.is_matchable = true filter
-- Partial indexes are smaller and faster than full indexes

CREATE INDEX IF NOT EXISTS idx_entity_aliases_is_matchable 
ON entity_aliases(is_matchable) 
WHERE is_matchable = true;

COMMENT ON INDEX idx_entity_aliases_is_matchable IS
'Partial index on is_matchable=true for fast filtering in load_all_aliases(). Only indexes matchable aliases, making it smaller and faster.';

-- ============================================================================
-- 2. Composite index for main query pattern
-- ============================================================================
-- Optimizes: WHERE is_matchable = true ORDER BY entity_id, id
-- This covers the filter, join key, and sort order

CREATE INDEX IF NOT EXISTS idx_entity_aliases_matchable_entity_id 
ON entity_aliases(is_matchable, entity_id, id) 
WHERE is_matchable = true;

COMMENT ON INDEX idx_entity_aliases_matchable_entity_id IS
'Composite index optimizing the main load_all_aliases() query: filters by is_matchable and orders by entity_id, id. Partial index for efficiency.';

-- ============================================================================
-- 3. Functional index on LOWER(alias) for "unidentified" pattern matching
-- ============================================================================
-- Enables efficient pattern matching for queries like:
-- WHERE LOWER(ea.alias) LIKE '%unidentified%'
--
-- Note: This uses trigram index which supports LIKE patterns efficiently
-- Requires pg_trgm extension (already enabled in 0013_enable_pg_trgm.sql)

CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_lower_trgm 
ON entity_aliases USING GIN (LOWER(alias) gin_trgm_ops);

COMMENT ON INDEX idx_entity_aliases_alias_lower_trgm IS
'Trigram index on LOWER(alias) for efficient pattern matching in "unidentified" entity detection. Supports LIKE queries with wildcards.';

-- ============================================================================
-- 4. Index on alias_norm for array-based "unidentified" queries
-- ============================================================================
-- Alternative approach: if we switch to alias_norm-based matching,
-- this index already exists (from 0019_entities.sql), but we can verify it's optimal

-- Verify existing alias_norm index exists and is optimal
-- (Already created in 0019_entities.sql as entity_aliases_alias_norm_idx)

-- ============================================================================
-- Performance Notes
-- ============================================================================
-- Expected improvements:
-- - is_matchable filtering: 10-50x faster (partial index)
-- - Main query ORDER BY: 2-5x faster (composite index)
-- - "Unidentified" pattern matching: 5-20x faster (trigram index)
--
-- Total expected improvement: 70-80% reduction in load_all_aliases() time
-- For 17,021 aliases: from ~30-60s to ~5-15s

COMMIT;
