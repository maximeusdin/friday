-- OCR Advanced Features: Phases 2B-4 + Adjudication Infrastructure
-- Confusion table, context features, anchoring, clustering, allowlist/blocklist

-- =============================================================================
-- 1. OCR CONFUSION TABLE (Phase 2B)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ocr_confusions (
    id SERIAL PRIMARY KEY,
    pattern_from TEXT NOT NULL,      -- Source pattern (e.g., 'rn', 'm')
    pattern_to TEXT NOT NULL,        -- Target pattern
    weight NUMERIC(3,2) DEFAULT 0.3, -- Substitution cost (0-1, lower = more likely)
    bidirectional BOOLEAN DEFAULT TRUE,
    category TEXT,                   -- 'letter_shape', 'digit_letter', 'punctuation', etc.
    note TEXT,
    UNIQUE (pattern_from, pattern_to)
);

-- Seed with common OCR confusions
INSERT INTO ocr_confusions (pattern_from, pattern_to, weight, category, note) VALUES
    -- Letter shape confusions (very common)
    ('rn', 'm', 0.2, 'letter_shape', 'rn looks like m'),
    ('m', 'rn', 0.2, 'letter_shape', 'm looks like rn'),
    ('cl', 'd', 0.3, 'letter_shape', 'cl looks like d'),
    ('d', 'cl', 0.3, 'letter_shape', 'd looks like cl'),
    ('vv', 'w', 0.2, 'letter_shape', 'vv looks like w'),
    ('w', 'vv', 0.2, 'letter_shape', 'w looks like vv'),
    ('li', 'h', 0.4, 'letter_shape', 'li can look like h'),
    ('h', 'li', 0.4, 'letter_shape', 'h can look like li'),
    ('ri', 'n', 0.3, 'letter_shape', 'ri looks like n'),
    ('n', 'ri', 0.3, 'letter_shape', 'n looks like ri'),
    ('nn', 'm', 0.3, 'letter_shape', 'nn can look like m'),
    
    -- Single letter confusions
    ('l', '1', 0.1, 'digit_letter', 'l/1 very similar'),
    ('1', 'l', 0.1, 'digit_letter', '1/l very similar'),
    ('I', 'l', 0.1, 'digit_letter', 'I/l similar'),
    ('l', 'I', 0.1, 'digit_letter', 'l/I similar'),
    ('O', '0', 0.1, 'digit_letter', 'O/0 very similar'),
    ('0', 'O', 0.1, 'digit_letter', '0/O very similar'),
    ('S', '5', 0.2, 'digit_letter', 'S/5 similar'),
    ('5', 'S', 0.2, 'digit_letter', '5/S similar'),
    ('Z', '2', 0.3, 'digit_letter', 'Z/2 sometimes confused'),
    ('B', '8', 0.3, 'digit_letter', 'B/8 sometimes confused'),
    ('G', '6', 0.4, 'digit_letter', 'G/6 occasional'),
    
    -- Case confusions (OCR often gets case wrong)
    ('c', 'C', 0.1, 'case', 'Case confusion'),
    ('o', 'O', 0.1, 'case', 'Case confusion'),
    ('s', 'S', 0.1, 'case', 'Case confusion'),
    ('v', 'V', 0.1, 'case', 'Case confusion'),
    ('w', 'W', 0.1, 'case', 'Case confusion'),
    ('x', 'X', 0.1, 'case', 'Case confusion'),
    ('z', 'Z', 0.1, 'case', 'Case confusion'),
    
    -- Punctuation drops/adds
    ('.', '', 0.1, 'punctuation', 'Period drop'),
    (',', '', 0.1, 'punctuation', 'Comma drop'),
    ('''', '', 0.1, 'punctuation', 'Apostrophe drop'),
    ('-', '', 0.1, 'punctuation', 'Hyphen drop'),
    (' ', '', 0.2, 'punctuation', 'Space drop'),
    
    -- Similar letter pairs
    ('e', 'c', 0.4, 'letter_shape', 'e/c similar'),
    ('a', 'o', 0.4, 'letter_shape', 'a/o similar'),
    ('u', 'n', 0.3, 'letter_shape', 'u/n similar'),
    ('h', 'b', 0.4, 'letter_shape', 'h/b similar'),
    ('f', 't', 0.4, 'letter_shape', 'f/t similar'),
    
    -- Multi-character
    ('ii', 'u', 0.3, 'letter_shape', 'ii looks like u'),
    ('fi', 'h', 0.4, 'letter_shape', 'fi ligature issue')
ON CONFLICT (pattern_from, pattern_to) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_ocr_confusions_from ON ocr_confusions (pattern_from);

-- =============================================================================
-- 2. CONTEXT FEATURES (Phase 3A)
-- =============================================================================

-- Add context columns to mention_candidates
ALTER TABLE mention_candidates
ADD COLUMN IF NOT EXISTS context_window TEXT,           -- ±N tokens around span
ADD COLUMN IF NOT EXISTS context_type_hints JSONB,      -- {person: 2, org: 1, loc: 0}
ADD COLUMN IF NOT EXISTS context_score NUMERIC(4,3),    -- Aggregated context signal
ADD COLUMN IF NOT EXISTS has_person_hint BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS has_org_hint BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS has_loc_hint BOOLEAN DEFAULT FALSE;

-- =============================================================================
-- 3. DOCUMENT ANCHORING (Phase 3B)
-- =============================================================================

-- Track anchored mentions within documents
CREATE TABLE IF NOT EXISTS document_anchors (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    anchor_surface_norm TEXT NOT NULL,
    anchor_score NUMERIC(5,4),
    anchor_method TEXT,               -- 'exact', 'strong_match', 'tier1'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (document_id, entity_id, anchor_surface_norm)
);

CREATE INDEX IF NOT EXISTS idx_document_anchors_doc ON document_anchors (document_id);
CREATE INDEX IF NOT EXISTS idx_document_anchors_entity ON document_anchors (entity_id);

-- Add anchoring columns to mention_candidates
ALTER TABLE mention_candidates
ADD COLUMN IF NOT EXISTS anchored_to_entity_id BIGINT REFERENCES entities(id),
ADD COLUMN IF NOT EXISTS anchor_reason TEXT;

-- =============================================================================
-- 4. VARIANT CLUSTERING (Phase 4)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ocr_variant_clusters (
    id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL UNIQUE,  -- Hash-based ID for stability
    
    -- Proposed canonical
    proposed_canonical TEXT,
    canonical_entity_id BIGINT REFERENCES entities(id),
    canonical_source TEXT,            -- 'existing_entity', 'high_quality_doc', 'frequency'
    
    -- Stats
    variant_count INTEGER DEFAULT 0,
    total_mentions INTEGER DEFAULT 0,
    doc_count INTEGER DEFAULT 0,
    
    -- Danger flags
    maps_to_multiple_entities BOOLEAN DEFAULT FALSE,
    has_type_conflict BOOLEAN DEFAULT FALSE,
    
    -- Recommendation
    recommendation TEXT,              -- 'SAFE_ADD', 'NEEDS_REVIEW', 'BLOCK'
    priority_score NUMERIC(6,3),
    
    -- Status
    status TEXT DEFAULT 'pending',    -- 'pending', 'approved', 'rejected', 'blocked'
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,
    review_decision TEXT,             -- 'APPROVE_MERGE', 'APPROVE_NEW', 'BLOCK', 'DEFER'
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ocr_variant_clusters_status ON ocr_variant_clusters (status);
CREATE INDEX IF NOT EXISTS idx_ocr_variant_clusters_recommendation ON ocr_variant_clusters (recommendation);
CREATE INDEX IF NOT EXISTS idx_ocr_variant_clusters_priority ON ocr_variant_clusters (priority_score DESC);

-- Cluster members (variants)
CREATE TABLE IF NOT EXISTS ocr_cluster_variants (
    id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES ocr_variant_clusters(cluster_id) ON DELETE CASCADE,
    variant_key TEXT NOT NULL,        -- Normalized variant string
    raw_examples TEXT[],              -- Sample raw strings
    mention_count INTEGER DEFAULT 0,
    doc_count INTEGER DEFAULT 0,
    avg_quality_score NUMERIC(4,3),
    current_entity_id BIGINT REFERENCES entities(id),  -- If already linked to an entity
    example_citations JSONB,          -- [{doc_id, chunk_id, surface}, ...]
    UNIQUE (cluster_id, variant_key)
);

CREATE INDEX IF NOT EXISTS idx_ocr_cluster_variants_cluster ON ocr_cluster_variants (cluster_id);
CREATE INDEX IF NOT EXISTS idx_ocr_cluster_variants_key ON ocr_cluster_variants (variant_key);

-- =============================================================================
-- 5. ALLOWLIST / BLOCKLIST (Never-Review-Twice)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ocr_variant_allowlist (
    id BIGSERIAL PRIMARY KEY,
    variant_key TEXT NOT NULL,
    entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    reason TEXT,
    source TEXT,                      -- 'adjudication', 'manual', 'auto_anchor'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (variant_key, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_ocr_allowlist_variant ON ocr_variant_allowlist (variant_key);

CREATE TABLE IF NOT EXISTS ocr_variant_blocklist (
    id BIGSERIAL PRIMARY KEY,
    variant_key TEXT,                 -- Exact variant to block (NULL if using signature)
    pattern_signature TEXT,           -- Regex or pattern signature
    block_type TEXT NOT NULL,         -- 'exact', 'pattern', 'cluster'
    cluster_id TEXT,                  -- If blocking entire cluster
    reason TEXT,
    source TEXT,                      -- 'adjudication', 'manual', 'junk_learning'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE NULLS NOT DISTINCT (variant_key, pattern_signature)
);

CREATE INDEX IF NOT EXISTS idx_ocr_blocklist_variant ON ocr_variant_blocklist (variant_key);

-- =============================================================================
-- 6. REVIEW EVENTS (Audit Trail)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ocr_review_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,         -- 'cluster_review', 'variant_approve', 'variant_block'
    cluster_id TEXT,
    variant_key TEXT,
    entity_id BIGINT,
    decision TEXT,
    reviewer TEXT,
    source_file TEXT,                 -- Original review file name
    source_file_hash TEXT,            -- MD5 of review file
    payload JSONB,                    -- Full decision details
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ocr_review_events_cluster ON ocr_review_events (cluster_id);
CREATE INDEX IF NOT EXISTS idx_ocr_review_events_type ON ocr_review_events (event_type);

-- =============================================================================
-- Done
-- =============================================================================

SELECT 'OCR advanced features schema created' AS status;
