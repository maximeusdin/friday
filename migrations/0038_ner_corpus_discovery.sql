-- Migration: NER Corpus Discovery
-- Stores aggregated NER-discovered surfaces with corpus frequency evidence
-- Used to expand the alias lexicon beyond concordance-derived aliases

-- =============================================================================
-- 1. NER Surface Stats (aggregated from corpus-wide NER sweep)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ner_surface_stats (
    id BIGSERIAL PRIMARY KEY,
    
    -- Surface identification
    surface_norm TEXT NOT NULL,           -- Normalized surface (canonical form)
    
    -- Corpus frequency evidence
    doc_count INTEGER NOT NULL DEFAULT 0,       -- Documents containing this surface
    chunk_count INTEGER NOT NULL DEFAULT 0,     -- Chunks containing this surface
    mention_count INTEGER NOT NULL DEFAULT 0,   -- Total mentions across corpus
    
    -- NER label consistency
    primary_label TEXT,                   -- Most common NER label (PERSON, ORG, etc.)
    label_distribution JSONB,             -- {"PERSON": 45, "ORG": 2, ...}
    label_consistency NUMERIC(4,3),       -- 0-1, how consistent the labeling is
    
    -- Entity type inference
    inferred_type TEXT,                   -- person/org/place (mapped from NER)
    type_confidence NUMERIC(4,3),         -- Confidence in type inference
    
    -- Context evidence (aggregated)
    context_hints_agg JSONB,              -- Aggregated context hints across mentions
    example_contexts TEXT[],              -- Sample context snippets (for review)
    
    -- Quality signals
    avg_accept_score NUMERIC(4,3),        -- Average NER acceptance score
    min_accept_score NUMERIC(4,3),
    max_accept_score NUMERIC(4,3),
    
    -- Matching status
    matches_existing_alias BOOLEAN DEFAULT FALSE,  -- Already in entity_aliases?
    matched_entity_id BIGINT REFERENCES entities(id) ON DELETE SET NULL,
    match_type TEXT,                      -- 'exact', 'fuzzy', NULL
    
    -- Tiering (for lexicon promotion)
    proposed_tier INTEGER,                -- 1=high confidence, 2=needs review, NULL=reject
    tier_reason TEXT,                     -- Why this tier was assigned
    
    -- Status tracking
    status TEXT NOT NULL DEFAULT 'pending' 
        CHECK (status IN ('pending', 'promoted', 'rejected', 'needs_review')),
    promoted_at TIMESTAMPTZ,
    reviewed_by TEXT,
    review_notes TEXT,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    corpus_sweep_id TEXT,                 -- Links to the sweep run that discovered this
    
    -- Constraints
    UNIQUE (surface_norm)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ner_surface_stats_doc_count 
    ON ner_surface_stats (doc_count DESC);

CREATE INDEX IF NOT EXISTS idx_ner_surface_stats_status 
    ON ner_surface_stats (status);

CREATE INDEX IF NOT EXISTS idx_ner_surface_stats_tier 
    ON ner_surface_stats (proposed_tier) 
    WHERE proposed_tier IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ner_surface_stats_type 
    ON ner_surface_stats (inferred_type);

-- Trigram index for fuzzy matching against existing aliases
CREATE INDEX IF NOT EXISTS idx_ner_surface_stats_trgm 
    ON ner_surface_stats USING GIN (surface_norm gin_trgm_ops);

COMMENT ON TABLE ner_surface_stats IS 
    'Aggregated NER-discovered surfaces with corpus frequency evidence. Used to expand alias_lexicon_index beyond concordance.';

-- =============================================================================
-- 2. NER Corpus Sweep Runs (tracking table)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ner_corpus_sweeps (
    id BIGSERIAL PRIMARY KEY,
    sweep_id TEXT NOT NULL UNIQUE,        -- UUID for this sweep run
    
    -- Scope
    collection_filter TEXT,               -- NULL = all collections
    doc_filter TEXT,                      -- Specific doc IDs if limited
    
    -- Processing stats
    chunks_processed INTEGER DEFAULT 0,
    chunks_total INTEGER DEFAULT 0,
    
    -- Discovery stats
    raw_spans_extracted INTEGER DEFAULT 0,
    unique_surfaces_found INTEGER DEFAULT 0,
    surfaces_new INTEGER DEFAULT 0,       -- Not in existing aliases
    surfaces_matching INTEGER DEFAULT 0,  -- Match existing aliases
    
    -- Tier distribution
    tier1_count INTEGER DEFAULT 0,
    tier2_count INTEGER DEFAULT 0,
    rejected_count INTEGER DEFAULT 0,
    
    -- Configuration
    model TEXT NOT NULL,
    threshold NUMERIC(4,3),
    config JSONB,                         -- Full config for reproducibility
    
    -- Status
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ner_corpus_sweeps_status 
    ON ner_corpus_sweeps (status);

COMMENT ON TABLE ner_corpus_sweeps IS 
    'Tracks corpus-wide NER sweep runs for discovery of new entity surfaces.';

-- =============================================================================
-- 3. NER Surface Mentions (raw mentions before aggregation)
-- Optional: can be used for detailed provenance, or skipped for performance
-- =============================================================================

CREATE TABLE IF NOT EXISTS ner_surface_mentions (
    id BIGSERIAL PRIMARY KEY,
    
    -- Location
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Span details
    surface TEXT NOT NULL,
    surface_norm TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    
    -- NER info
    ner_label TEXT NOT NULL,
    ner_accept_score NUMERIC(4,3),
    context_hints JSONB,
    
    -- Context snippet
    context_window TEXT,                  -- ±50 chars around mention
    
    -- Sweep tracking
    sweep_id TEXT,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraint for idempotency
    UNIQUE (chunk_id, char_start, char_end, surface_norm)
);

CREATE INDEX IF NOT EXISTS idx_ner_surface_mentions_surface_norm 
    ON ner_surface_mentions (surface_norm);

CREATE INDEX IF NOT EXISTS idx_ner_surface_mentions_sweep 
    ON ner_surface_mentions (sweep_id);

CREATE INDEX IF NOT EXISTS idx_ner_surface_mentions_doc 
    ON ner_surface_mentions (document_id);

COMMENT ON TABLE ner_surface_mentions IS 
    'Raw NER mentions before aggregation. Provides provenance for ner_surface_stats.';

-- =============================================================================
-- Done
-- =============================================================================

SELECT 'NER corpus discovery schema created' AS status;
