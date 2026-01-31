-- Migration: Additional indexes for proposal corpus system
-- Supports fast lookups for tier filtering and override queries

BEGIN;

-- =============================================================================
-- entity_surface_stats indexes (for tier filtering)
-- =============================================================================

-- Index on tier for filtered queries (via entity_surface_tiers view)
-- Note: We can't index a view directly, but we can add a functional index
-- on the columns used in tier computation for faster view queries
CREATE INDEX IF NOT EXISTS idx_entity_surface_stats_tier_cols
    ON entity_surface_stats(doc_freq DESC, type_stable, confidence_mean DESC, surface_length);

-- Index on entity_id for entity-centric lookups
CREATE INDEX IF NOT EXISTS idx_entity_surface_stats_entity_id
    ON entity_surface_stats(entity_id);

-- =============================================================================
-- mention_review_queue indexes (for grouping queries)
-- =============================================================================

-- GIN index on candidate_entity_ids for "contains" queries
-- e.g., "find all review items containing entity X"
CREATE INDEX IF NOT EXISTS idx_mention_review_queue_candidate_ids_gin
    ON mention_review_queue USING GIN (candidate_entity_ids)
    WHERE candidate_entity_ids IS NOT NULL;

-- Composite index for grouped batch processing
CREATE INDEX IF NOT EXISTS idx_mention_review_queue_group_pending
    ON mention_review_queue(group_key, created_at)
    WHERE status = 'pending';

-- =============================================================================
-- entity_alias_overrides indexes
-- =============================================================================

-- Index for fast lookup during extraction
CREATE INDEX IF NOT EXISTS idx_entity_alias_overrides_lookup
    ON entity_alias_overrides(surface_norm, scope);

-- Index for finding overrides by entity
CREATE INDEX IF NOT EXISTS idx_entity_alias_overrides_forced_entity
    ON entity_alias_overrides(forced_entity_id)
    WHERE forced_entity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_entity_alias_overrides_banned_entity
    ON entity_alias_overrides(banned_entity_id)
    WHERE banned_entity_id IS NOT NULL;

COMMIT;
