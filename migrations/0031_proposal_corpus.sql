-- Migration: Proposal Corpus System
-- Implements entity surface statistics and tiering for controlled extraction gating
--
-- Components:
--   1. normalize_surface() SQL function (mirrors Python)
--   2. chunk_quality_stats view (trusted span classification)
--   3. entity_surface_stats materialized view (proposal corpus)
--   4. entity_surface_tiers view (deterministic tiering)
--   5. entity_alias_overrides table (human override mechanism)
--   6. Review queue grouping columns

BEGIN;

-- =============================================================================
-- 1. Surface Normalization Function (Postgres mirror of Python)
-- =============================================================================

CREATE OR REPLACE FUNCTION normalize_surface(input_text TEXT)
RETURNS TEXT
LANGUAGE SQL
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT TRIM(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                LOWER(COALESCE(input_text, '')),
                '[^\w\s]', '', 'g'  -- Remove punctuation
            ),
            '\s+', ' ', 'g'  -- Collapse whitespace
        )
    )
$$;

COMMENT ON FUNCTION normalize_surface(TEXT) IS 
'Canonical surface normalization for entity matching. Mirrors Python retrieval.surface_norm.normalize_surface().';


-- =============================================================================
-- 2. Chunk Quality Stats (trusted span classification)
-- =============================================================================

-- Add quality metric columns to chunks if they don't exist
DO $$
BEGIN
    -- Alpha ratio: proportion of alphabetic characters
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'alpha_ratio') THEN
        ALTER TABLE chunks ADD COLUMN alpha_ratio REAL;
    END IF;
    
    -- Garbage indicators: suspected OCR garbage
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'garbage_score') THEN
        ALTER TABLE chunks ADD COLUMN garbage_score REAL;
    END IF;
    
    -- Is this a trusted (high-quality) chunk?
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'is_trusted_text') THEN
        ALTER TABLE chunks ADD COLUMN is_trusted_text BOOLEAN;
    END IF;
END $$;

-- View for chunk quality assessment
-- Trusted chunks: clean text OR high alpha ratio with low garbage
CREATE OR REPLACE VIEW v_chunk_quality AS
SELECT 
    c.id as chunk_id,
    c.text_quality,
    c.alpha_ratio,
    c.garbage_score,
    c.is_trusted_text,
    -- Computed trusted flag (fallback if is_trusted_text not set)
    CASE 
        WHEN c.is_trusted_text IS NOT NULL THEN c.is_trusted_text
        WHEN c.text_quality = 'clean' THEN TRUE
        -- Collection-based heuristics when text_quality unknown
        WHEN col.slug IN ('mccarthy', 'huac_hearings') THEN TRUE  -- Typed transcripts
        WHEN col.slug = 'venona' THEN TRUE  -- Decrypted cables
        -- Default: require explicit quality classification
        ELSE FALSE
    END as is_trusted,
    col.slug as collection_slug
FROM chunks c
JOIN chunk_pages cp ON cp.chunk_id = c.id
JOIN pages p ON p.id = cp.page_id
JOIN documents d ON d.id = p.document_id
JOIN collections col ON col.id = d.collection_id
GROUP BY c.id, c.text_quality, c.alpha_ratio, c.garbage_score, c.is_trusted_text, col.slug;


-- =============================================================================
-- 3. Entity Surface Stats (Proposal Corpus Core)
-- =============================================================================

-- Materialized view: aggregated stats per (entity_id, surface_norm)
-- Filtered to trusted spans + high-confidence exact matches + stop word filter
CREATE MATERIALIZED VIEW IF NOT EXISTS entity_surface_stats AS
WITH trusted_mentions AS (
    SELECT 
        em.entity_id,
        em.surface_norm,
        em.surface,
        em.document_id,
        em.confidence,
        e.entity_type
    FROM entity_mentions em
    JOIN entities e ON e.id = em.entity_id
    -- Join to chunk quality
    LEFT JOIN v_chunk_quality cq ON cq.chunk_id = em.chunk_id
    WHERE 
        -- High-confidence exact matches only
        em.method IN ('alias_exact', 'alias_exact_clean')
        AND em.confidence >= 0.95
        -- Trusted text OR null chunk (fallback to document-level trust)
        AND (cq.is_trusted = TRUE OR em.chunk_id IS NULL)
        -- Minimum surface length
        AND LENGTH(em.surface_norm) >= 2
        -- Exclude common stop words (even if they're valid covernames)
        -- These create too much noise in proposal-based matching
        AND em.surface_norm NOT IN (
            'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','as',
            'is','was','are','were','been','be','have','has','had','do','does','did',
            'will','would','could','should','may','might','must','shall','can',
            'this','that','these','those','it','its','they','them','he','she','him','her','his','hers',
            'we','us','our','i','me','my','said','says','told','asked','yes','no','not','also','just',
            'two','one','three','four','five','six','seven','eight','nine','ten',
            'group','office','state','general','chief','last','personal','cipher','first','second',
            'new','old','soviet','russian','american','german','british','french','chinese','japanese'
        )
)
SELECT 
    tm.entity_id,
    tm.surface_norm,
    -- Most common surface variant for display
    MODE() WITHIN GROUP (ORDER BY tm.surface) as surface_display,
    MODE() WITHIN GROUP (ORDER BY tm.entity_type) as entity_type,
    -- Stats
    COUNT(DISTINCT tm.document_id) as doc_freq,
    COUNT(*) as mention_count,
    AVG(tm.confidence)::NUMERIC(4,3) as confidence_mean,
    MIN(tm.confidence)::NUMERIC(4,3) as confidence_min,
    -- Type stability: single entity_type across all mentions
    COUNT(DISTINCT tm.entity_type) = 1 as type_stable,
    -- Surface length (for filtering short surfaces)
    LENGTH(tm.surface_norm) as surface_length,
    -- Metadata
    NOW() as computed_at
FROM trusted_mentions tm
GROUP BY tm.entity_id, tm.surface_norm;

-- Index for fast lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_surface_stats_pk 
    ON entity_surface_stats(entity_id, surface_norm);
CREATE INDEX IF NOT EXISTS idx_entity_surface_stats_surface_norm 
    ON entity_surface_stats(surface_norm);
CREATE INDEX IF NOT EXISTS idx_entity_surface_stats_doc_freq 
    ON entity_surface_stats(doc_freq DESC);

COMMENT ON MATERIALIZED VIEW entity_surface_stats IS 
'Aggregated entity surface statistics from trusted spans. Core of proposal corpus for extraction gating.';


-- =============================================================================
-- 4. Entity Surface Tiers (Deterministic Tiering)
-- =============================================================================

-- View that assigns tiers based on thresholds
-- Using view (not stored column) allows easy threshold tuning
CREATE OR REPLACE VIEW entity_surface_tiers AS
SELECT 
    ess.*,
    CASE 
        -- Tier 1: Auto-accept eligible (high confidence, well-attested)
        WHEN ess.doc_freq >= 3 
             AND ess.type_stable = TRUE 
             AND ess.confidence_mean >= 0.95
             AND ess.surface_length >= 4
        THEN 1
        
        -- Tier 1 exception: Known acronyms (short but valid)
        WHEN ess.doc_freq >= 3
             AND ess.type_stable = TRUE
             AND ess.confidence_mean >= 0.95
             AND ess.surface_length >= 2
             AND ess.entity_type IN ('organization', 'covername')
             AND UPPER(ess.surface_norm) = ess.surface_norm  -- All uppercase = acronym
        THEN 1
        
        -- Tier 2: Queue-eligible (attested, needs review for new contexts)
        WHEN ess.doc_freq >= 1
             AND ess.type_stable = TRUE
             AND ess.confidence_mean >= 0.85
        THEN 2
        
        -- Tier 3: Not proposal-eligible (insufficient evidence)
        ELSE 3
    END as tier,
    
    -- Tier explanation for debugging
    CASE 
        WHEN ess.doc_freq >= 3 AND ess.type_stable = TRUE AND ess.confidence_mean >= 0.95 AND ess.surface_length >= 4
        THEN 'tier1: doc_freq>=3, type_stable, conf>=0.95, len>=4'
        WHEN ess.doc_freq >= 3 AND ess.type_stable = TRUE AND ess.confidence_mean >= 0.95 AND ess.surface_length >= 2 
             AND ess.entity_type IN ('organization', 'covername')
        THEN 'tier1: acronym exception'
        WHEN ess.doc_freq >= 1 AND ess.type_stable = TRUE AND ess.confidence_mean >= 0.85
        THEN 'tier2: doc_freq>=1, type_stable, conf>=0.85'
        ELSE 'tier3: insufficient evidence'
    END as tier_reason
FROM entity_surface_stats ess;

COMMENT ON VIEW entity_surface_tiers IS 
'Deterministic tier assignment for proposal corpus. Tier 1=auto-accept, Tier 2=queue-eligible, Tier 3=not proposable.';


-- =============================================================================
-- 5. Entity Alias Overrides (Human Override Mechanism)
-- =============================================================================

CREATE TABLE IF NOT EXISTS entity_alias_overrides (
    id BIGSERIAL PRIMARY KEY,
    
    -- Surface to override (normalized)
    surface_norm TEXT NOT NULL,
    
    -- Scope of override
    scope TEXT NOT NULL DEFAULT 'global' CHECK (scope IN ('global', 'collection', 'document')),
    scope_collection_id BIGINT REFERENCES collections(id),
    scope_document_id BIGINT REFERENCES documents(id),
    
    -- Override action
    forced_entity_id BIGINT REFERENCES entities(id),  -- Force match to this entity
    banned_entity_id BIGINT REFERENCES entities(id),  -- Never match to this entity
    banned BOOLEAN DEFAULT FALSE,                      -- Ban surface entirely
    
    -- Metadata
    note TEXT,
    created_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT override_has_action CHECK (
        forced_entity_id IS NOT NULL OR banned_entity_id IS NOT NULL OR banned = TRUE
    )
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entity_alias_overrides_surface_norm 
    ON entity_alias_overrides(surface_norm);
CREATE INDEX IF NOT EXISTS idx_entity_alias_overrides_scope 
    ON entity_alias_overrides(scope, scope_collection_id, scope_document_id);

-- Unique constraint: one override per (surface_norm, scope, scope_id) combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_alias_overrides_unique
    ON entity_alias_overrides(
        surface_norm, 
        scope, 
        COALESCE(scope_collection_id, -1), 
        COALESCE(scope_document_id, -1),
        COALESCE(forced_entity_id, -1),
        COALESCE(banned_entity_id, -1)
    );

COMMENT ON TABLE entity_alias_overrides IS 
'Human overrides for entity-surface matching. Highest priority in resolution precedence.';

COMMENT ON COLUMN entity_alias_overrides.scope IS 
'Override scope: global (all contexts), collection (specific collection), document (specific document)';


-- =============================================================================
-- 6. Review Queue Grouping Columns
-- =============================================================================

-- Add grouping columns to mention_review_queue
DO $$
BEGIN
    -- Group key for batch processing (surface_norm + candidate_set_hash)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'mention_review_queue' AND column_name = 'group_key') THEN
        ALTER TABLE mention_review_queue ADD COLUMN group_key TEXT;
    END IF;
    
    -- Hash of candidate entity IDs for grouping similar ambiguities
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'mention_review_queue' AND column_name = 'candidate_set_hash') THEN
        ALTER TABLE mention_review_queue ADD COLUMN candidate_set_hash TEXT;
    END IF;
    
    -- Explicit array of candidate entity IDs (denormalized for convenience)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'mention_review_queue' AND column_name = 'candidate_entity_ids') THEN
        ALTER TABLE mention_review_queue ADD COLUMN candidate_entity_ids BIGINT[];
    END IF;
    
    -- Surface norm for consistent grouping
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'mention_review_queue' AND column_name = 'surface_norm') THEN
        ALTER TABLE mention_review_queue ADD COLUMN surface_norm TEXT;
    END IF;
END $$;

-- Index for group-based queries
CREATE INDEX IF NOT EXISTS idx_mention_review_queue_group_key 
    ON mention_review_queue(group_key) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_mention_review_queue_surface_norm 
    ON mention_review_queue(surface_norm);
CREATE INDEX IF NOT EXISTS idx_mention_review_queue_candidate_set_hash 
    ON mention_review_queue(candidate_set_hash);

COMMENT ON COLUMN mention_review_queue.group_key IS 
'Grouping key for batch operations: combines surface_norm + candidate_set_hash';
COMMENT ON COLUMN mention_review_queue.candidate_set_hash IS 
'MD5 hash of sorted candidate entity IDs for grouping similar ambiguities';


-- =============================================================================
-- 7. Helper Function: Check if surface is proposable
-- =============================================================================

CREATE OR REPLACE FUNCTION is_surface_proposable(
    p_surface_norm TEXT,
    p_min_tier INT DEFAULT 2,
    p_scope_collection_id BIGINT DEFAULT NULL,
    p_scope_document_id BIGINT DEFAULT NULL
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_override_exists BOOLEAN;
    v_tier INT;
BEGIN
    -- Check for ban override (highest priority)
    SELECT EXISTS (
        SELECT 1 FROM entity_alias_overrides
        WHERE surface_norm = p_surface_norm
          AND banned = TRUE
          AND (
              scope = 'global'
              OR (scope = 'collection' AND scope_collection_id = p_scope_collection_id)
              OR (scope = 'document' AND scope_document_id = p_scope_document_id)
          )
    ) INTO v_override_exists;
    
    IF v_override_exists THEN
        RETURN FALSE;
    END IF;
    
    -- Check tier from proposal corpus
    SELECT MIN(tier) INTO v_tier
    FROM entity_surface_tiers
    WHERE surface_norm = p_surface_norm;
    
    IF v_tier IS NULL THEN
        -- Not in proposal corpus = not proposable (conservative)
        RETURN FALSE;
    END IF;
    
    RETURN v_tier <= p_min_tier;
END;
$$;

COMMENT ON FUNCTION is_surface_proposable IS 
'Check if a surface is eligible for proposal-based extraction. Consults overrides then proposal corpus tiers.';


-- =============================================================================
-- 8. Helper Function: Get proposal surfaces by tier
-- =============================================================================

CREATE OR REPLACE FUNCTION get_proposal_surfaces(
    p_tier INT DEFAULT 2,
    p_entity_type TEXT DEFAULT NULL,
    p_limit INT DEFAULT 1000
)
RETURNS TABLE (
    entity_id BIGINT,
    surface_norm TEXT,
    surface_display TEXT,
    entity_type TEXT,
    doc_freq INT,
    tier INT
)
LANGUAGE SQL
STABLE
AS $$
    SELECT 
        est.entity_id,
        est.surface_norm,
        est.surface_display,
        est.entity_type,
        est.doc_freq::INT,
        est.tier
    FROM entity_surface_tiers est
    WHERE est.tier <= p_tier
      AND (p_entity_type IS NULL OR est.entity_type = p_entity_type)
      -- Exclude banned surfaces
      AND NOT EXISTS (
          SELECT 1 FROM entity_alias_overrides eao
          WHERE eao.surface_norm = est.surface_norm
            AND eao.banned = TRUE
            AND eao.scope = 'global'
      )
    ORDER BY est.doc_freq DESC, est.surface_norm
    LIMIT p_limit;
$$;

COMMENT ON FUNCTION get_proposal_surfaces IS 
'Get surfaces eligible for proposal-based extraction, filtered by tier and optionally entity type.';


-- =============================================================================
-- Summary/Audit Views
-- =============================================================================

-- Tier summary for monitoring
CREATE OR REPLACE VIEW v_proposal_corpus_summary AS
SELECT 
    tier,
    tier_reason,
    COUNT(*) as surface_count,
    COUNT(DISTINCT entity_id) as entity_count,
    SUM(mention_count) as total_mentions,
    AVG(doc_freq)::NUMERIC(6,2) as avg_doc_freq,
    AVG(confidence_mean)::NUMERIC(4,3) as avg_confidence
FROM entity_surface_tiers
GROUP BY tier, tier_reason
ORDER BY tier, tier_reason;

COMMENT ON VIEW v_proposal_corpus_summary IS 
'Summary statistics for proposal corpus tiers. Use for monitoring threshold effectiveness.';


COMMIT;

-- =============================================================================
-- Post-migration: Refresh materialized view
-- =============================================================================
-- Run after migration:
-- REFRESH MATERIALIZED VIEW entity_surface_stats;
