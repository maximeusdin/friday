-- Day 10: Entity Mentions Surface Quality Tracking
-- Adds columns to track surface text quality (exact vs approximate)

BEGIN;

-- ============================================================================
-- Add surface quality columns to entity_mentions
-- ============================================================================

ALTER TABLE entity_mentions
  ADD COLUMN IF NOT EXISTS surface_norm TEXT NULL,  -- Normalized surface (for approximate matches)
  ADD COLUMN IF NOT EXISTS surface_quality TEXT NOT NULL DEFAULT 'exact' 
    CHECK (surface_quality IN ('exact', 'approx'));

-- Backfill: Set surface_quality based on whether surface matches normalized form
-- If surface is already normalized (matches alias_norm), mark as approx
-- This is a heuristic - future extractions will set it correctly
UPDATE entity_mentions em
SET surface_quality = CASE
    WHEN LOWER(REGEXP_REPLACE(em.surface, '[^\w\s]', '', 'g')) = 
         (SELECT ea.alias_norm FROM entity_aliases ea 
          JOIN entities e ON e.id = em.entity_id 
          WHERE ea.entity_id = e.id 
          AND LOWER(REGEXP_REPLACE(ea.alias, '[^\w\s]', '', 'g')) = 
              LOWER(REGEXP_REPLACE(em.surface, '[^\w\s]', '', 'g'))
          LIMIT 1)
    THEN 'exact'
    ELSE 'approx'  -- Conservative: mark existing as approx if we can't verify
END
WHERE surface_quality = 'exact';

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON COLUMN entity_mentions.surface_norm IS
'Normalized surface text (for approximate matches where original alignment failed). NULL if surface is exact.';

COMMENT ON COLUMN entity_mentions.surface_quality IS
'Quality of surface text: "exact" (aligned from original chunk text) or "approx" (normalized/approximate). '
'Historians should filter/triage approximate matches - they may not be literal page text due to OCR/punctuation differences.';

COMMIT;
