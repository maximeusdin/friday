-- OCR Extraction Pipeline V1
-- Creates infrastructure for OCR-aware entity extraction

-- =============================================================================
-- 1. SEED LEXICON INDEX
-- Optimized table for fast fuzzy retrieval during OCR resolution
-- =============================================================================

CREATE TABLE IF NOT EXISTS alias_lexicon_index (
    id BIGSERIAL PRIMARY KEY,
    
    -- Core alias data
    alias_norm TEXT NOT NULL,
    entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_type TEXT,
    
    -- Frequency/confidence stats (from corpus evidence)
    doc_freq INTEGER DEFAULT 0,           -- Documents containing this alias
    mention_count INTEGER DEFAULT 0,      -- Total mentions in corpus
    corpus_confidence NUMERIC(4,3),       -- Avg confidence from entity_surface_stats
    
    -- Tiering info (from proposal corpus)
    proposal_tier INTEGER,                -- 1=auto-accept, 2=queue-eligible, NULL=not proposable
    is_from_trusted_text BOOLEAN DEFAULT FALSE,  -- Derived from clean/trusted sources
    
    -- Matching hints
    alias_length INTEGER,                 -- Length of alias_norm (for quick filtering)
    token_count INTEGER,                  -- Number of tokens
    alias_class TEXT,                     -- person_given, person_last, covername, etc.
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint: one entry per (alias_norm, entity_id)
    UNIQUE (alias_norm, entity_id)
);

-- Indexes for fast retrieval
-- Primary: trigram index for fuzzy matching
CREATE INDEX IF NOT EXISTS idx_alias_lexicon_trgm 
    ON alias_lexicon_index USING GIN (alias_norm gin_trgm_ops);

-- Secondary: exact lookups
CREATE INDEX IF NOT EXISTS idx_alias_lexicon_norm 
    ON alias_lexicon_index (alias_norm);

-- For filtering by tier/quality
CREATE INDEX IF NOT EXISTS idx_alias_lexicon_tier 
    ON alias_lexicon_index (proposal_tier) 
    WHERE proposal_tier IS NOT NULL;

-- For entity-centric queries
CREATE INDEX IF NOT EXISTS idx_alias_lexicon_entity 
    ON alias_lexicon_index (entity_id);

-- For length-based filtering (useful for OCR)
CREATE INDEX IF NOT EXISTS idx_alias_lexicon_length 
    ON alias_lexicon_index (alias_length);

COMMENT ON TABLE alias_lexicon_index IS 
    'Optimized index for OCR-aware entity resolution. Populated from entity_aliases + proposal corpus.';

-- =============================================================================
-- 2. MENTION CANDIDATES TABLE
-- Stores candidate spans extracted from OCR text before resolution
-- =============================================================================

CREATE TABLE IF NOT EXISTS mention_candidates (
    id BIGSERIAL PRIMARY KEY,
    
    -- Location
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id BIGINT REFERENCES chunks(id) ON DELETE CASCADE,
    page_id BIGINT,  -- Optional, if page-level granularity available
    
    -- Span coordinates
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    
    -- Raw and normalized text
    raw_span TEXT NOT NULL,               -- Exact text from OCR
    surface_norm TEXT NOT NULL,           -- Normalized for matching
    
    -- Quality signals
    quality_score NUMERIC(4,3),           -- 0-1 span quality score
    doc_quality TEXT DEFAULT 'ocr',       -- 'ocr', 'clean', 'mixed'
    
    -- Span characteristics
    token_count INTEGER,
    has_hint_token BOOLEAN DEFAULT FALSE, -- Contains Mr/Mrs/Dr/Bureau/etc.
    in_boilerplate_zone BOOLEAN DEFAULT FALSE,
    
    -- Resolution status
    resolution_status TEXT DEFAULT 'pending',  -- pending, resolved, queued, ignored, junk
    resolved_entity_id BIGINT REFERENCES entities(id) ON DELETE SET NULL,
    resolved_at TIMESTAMPTZ,
    resolution_method TEXT,               -- ocr_lexicon_strong, ocr_lexicon_propose, etc.
    resolution_score NUMERIC(5,4),
    resolution_margin NUMERIC(5,4),       -- Gap to 2nd best candidate
    
    -- Top candidates (stored for audit/review)
    top_candidates JSONB,                 -- [{entity_id, alias_norm, score, signals}, ...]
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    batch_id TEXT,                        -- For tracking extraction runs
    
    -- Constraints
    CONSTRAINT valid_char_range CHECK (char_end > char_start),
    CONSTRAINT valid_quality_score CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1))
);

-- Indexes for mention_candidates
CREATE INDEX IF NOT EXISTS idx_mention_candidates_chunk 
    ON mention_candidates (chunk_id);

CREATE INDEX IF NOT EXISTS idx_mention_candidates_document 
    ON mention_candidates (document_id);

CREATE INDEX IF NOT EXISTS idx_mention_candidates_status 
    ON mention_candidates (resolution_status);

CREATE INDEX IF NOT EXISTS idx_mention_candidates_surface 
    ON mention_candidates (surface_norm);

-- For finding unresolved candidates efficiently
CREATE INDEX IF NOT EXISTS idx_mention_candidates_pending 
    ON mention_candidates (chunk_id, quality_score DESC) 
    WHERE resolution_status = 'pending';

-- Trigram index for clustering similar candidates
CREATE INDEX IF NOT EXISTS idx_mention_candidates_trgm 
    ON mention_candidates USING GIN (surface_norm gin_trgm_ops);

COMMENT ON TABLE mention_candidates IS 
    'Candidate entity mention spans from OCR text, before resolution.';

-- =============================================================================
-- 3. OCR JUNK PATTERNS TABLE
-- Learned patterns to filter out recurring garbage
-- =============================================================================

CREATE TABLE IF NOT EXISTS ocr_junk_patterns (
    id BIGSERIAL PRIMARY KEY,
    
    pattern_type TEXT NOT NULL,           -- 'exact', 'prefix', 'suffix', 'regex', 'zone'
    pattern_value TEXT NOT NULL,          -- The pattern itself
    
    -- Stats
    occurrence_count INTEGER DEFAULT 1,
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Origin
    learned_from TEXT,                    -- 'manual', 'review_queue', 'auto_detect'
    note TEXT,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (pattern_type, pattern_value)
);

CREATE INDEX IF NOT EXISTS idx_ocr_junk_patterns_active 
    ON ocr_junk_patterns (pattern_type) 
    WHERE is_active = TRUE;

COMMENT ON TABLE ocr_junk_patterns IS 
    'Learned patterns for filtering OCR garbage. Updated via review queue feedback.';

-- =============================================================================
-- 4. OCR EXTRACTION RUNS TABLE
-- Track extraction batches for metrics and debugging
-- =============================================================================

CREATE TABLE IF NOT EXISTS ocr_extraction_runs (
    id BIGSERIAL PRIMARY KEY,
    
    batch_id TEXT UNIQUE NOT NULL,
    
    -- Scope
    collection_slug TEXT,
    document_id BIGINT,
    chunk_id_start BIGINT,
    chunk_id_end BIGINT,
    
    -- Timing
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Counts
    chunks_processed INTEGER DEFAULT 0,
    candidates_generated INTEGER DEFAULT 0,
    candidates_resolved INTEGER DEFAULT 0,
    candidates_queued INTEGER DEFAULT 0,
    candidates_ignored INTEGER DEFAULT 0,
    candidates_junk INTEGER DEFAULT 0,
    
    -- Rates (computed)
    link_rate NUMERIC(5,4),               -- resolved / (resolved + queued)
    queue_rate NUMERIC(5,4),              -- queued / total_non_junk
    junk_rate NUMERIC(5,4),               -- junk / candidates_generated
    
    -- Config snapshot
    config JSONB,
    
    -- Status
    status TEXT DEFAULT 'running',        -- running, completed, failed
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_ocr_extraction_runs_batch 
    ON ocr_extraction_runs (batch_id);

COMMENT ON TABLE ocr_extraction_runs IS 
    'Tracks OCR extraction runs for metrics and debugging.';

-- =============================================================================
-- 5. HELPER FUNCTIONS
-- =============================================================================

-- Function to compute token-level similarity (for multi-word aliases)
CREATE OR REPLACE FUNCTION token_overlap_score(text1 TEXT, text2 TEXT)
RETURNS NUMERIC
LANGUAGE SQL
IMMUTABLE
PARALLEL SAFE
AS $$
    WITH tokens1 AS (
        SELECT DISTINCT unnest(string_to_array(LOWER(text1), ' ')) AS token
    ),
    tokens2 AS (
        SELECT DISTINCT unnest(string_to_array(LOWER(text2), ' ')) AS token
    ),
    intersection AS (
        SELECT COUNT(*) AS cnt FROM tokens1 t1 JOIN tokens2 t2 ON t1.token = t2.token
    ),
    union_count AS (
        SELECT COUNT(DISTINCT token) AS cnt FROM (
            SELECT token FROM tokens1 UNION ALL SELECT token FROM tokens2
        ) all_tokens
    )
    SELECT CASE 
        WHEN (SELECT cnt FROM union_count) = 0 THEN 0
        ELSE (SELECT cnt FROM intersection)::NUMERIC / (SELECT cnt FROM union_count)
    END
$$;

COMMENT ON FUNCTION token_overlap_score IS 
    'Jaccard similarity of tokens between two strings.';

-- Function to get top-K alias matches for a surface
CREATE OR REPLACE FUNCTION get_top_alias_matches(
    p_surface TEXT,
    p_limit INTEGER DEFAULT 10,
    p_min_similarity NUMERIC DEFAULT 0.3
)
RETURNS TABLE (
    alias_norm TEXT,
    entity_id BIGINT,
    entity_type TEXT,
    proposal_tier INTEGER,
    trigram_sim NUMERIC,
    token_overlap NUMERIC,
    combined_score NUMERIC
)
LANGUAGE SQL
STABLE
AS $$
    SELECT 
        ali.alias_norm,
        ali.entity_id,
        ali.entity_type,
        ali.proposal_tier,
        similarity(ali.alias_norm, normalize_surface(p_surface)) AS trigram_sim,
        token_overlap_score(ali.alias_norm, p_surface) AS token_overlap,
        (0.6 * similarity(ali.alias_norm, normalize_surface(p_surface)) + 
         0.4 * token_overlap_score(ali.alias_norm, p_surface)) AS combined_score
    FROM alias_lexicon_index ali
    WHERE 
        similarity(ali.alias_norm, normalize_surface(p_surface)) >= p_min_similarity
        OR token_overlap_score(ali.alias_norm, p_surface) >= 0.5
    ORDER BY combined_score DESC
    LIMIT p_limit
$$;

COMMENT ON FUNCTION get_top_alias_matches IS 
    'Retrieve top-K alias candidates for a surface using trigram + token overlap.';

-- =============================================================================
-- 6. SEED DATA: Common junk patterns
-- =============================================================================

INSERT INTO ocr_junk_patterns (pattern_type, pattern_value, learned_from, note)
VALUES 
    -- Common OCR garbage patterns
    ('exact', '..', 'manual', 'Double dots'),
    ('exact', '...', 'manual', 'Triple dots'),
    ('exact', '....', 'manual', 'Quad dots'),
    ('prefix', '••', 'manual', 'Bullet artifacts'),
    ('regex', '^[•\.\-_\*]+$', 'manual', 'Punctuation-only spans'),
    ('regex', '^[0-9\.\-/]+$', 'manual', 'Numeric-only spans'),
    
    -- Boilerplate patterns
    ('exact', 'deleted', 'manual', 'Redaction marker'),
    ('exact', 'foipa', 'manual', 'FOIA marker'),
    ('exact', 'page', 'manual', 'Page marker'),
    ('prefix', 'page ', 'manual', 'Page number prefix'),
    ('exact', 'continued', 'manual', 'Continuation marker'),
    ('exact', 'copy', 'manual', 'Copy marker'),
    ('exact', 'unclassified', 'manual', 'Classification marker'),
    ('exact', 'secret', 'manual', 'Classification marker'),
    ('exact', 'confidential', 'manual', 'Classification marker'),
    ('exact', 'top secret', 'manual', 'Classification marker'),
    
    -- Common false positives
    ('exact', 'the', 'manual', 'Stop word'),
    ('exact', 'and', 'manual', 'Stop word'),
    ('exact', 'for', 'manual', 'Stop word'),
    ('exact', 'that', 'manual', 'Stop word'),
    ('exact', 'this', 'manual', 'Stop word'),
    ('exact', 'with', 'manual', 'Stop word')
ON CONFLICT (pattern_type, pattern_value) DO NOTHING;

-- =============================================================================
-- 7. UPDATE chunks TABLE: Add OCR quality classification column
-- =============================================================================

-- Add column if not exists (for explicit OCR flagging)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'chunks' AND column_name = 'is_ocr_source'
    ) THEN
        ALTER TABLE chunks ADD COLUMN is_ocr_source BOOLEAN;
    END IF;
END $$;

COMMENT ON COLUMN chunks.is_ocr_source IS 
    'TRUE if chunk comes from OCR text (vs clean/typed text).';

-- =============================================================================
-- Done
-- =============================================================================

SELECT 'OCR extraction pipeline tables created successfully' AS status;
