-- 0041_schema_parity_with_neh.sql
-- Brings a fresh DB schema to parity with the production "neh" database.
-- This covers tables, columns, constraints, indexes, and views that were
-- either created ad-hoc or exist in neh but are not in earlier migrations.

BEGIN;

-- ============================================================================
-- 1. EXTENSION: citext (used by entities.canonical_name and entity_aliases.alias)
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS citext;

-- ============================================================================
-- 2. CONCORDANCE SYSTEM (source of truth for entity/alias data)
-- ============================================================================

-- concordance_sources: registers external concordance sources
CREATE TABLE IF NOT EXISTS concordance_sources (
    id          BIGSERIAL PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- concordance_entries: raw parsed entries from concordance sources
CREATE TABLE IF NOT EXISTS concordance_entries (
    id          BIGSERIAL PRIMARY KEY,
    source_id   BIGINT NOT NULL REFERENCES concordance_sources(id) ON DELETE CASCADE,
    entry_key   TEXT NOT NULL,
    raw_text    TEXT NOT NULL,
    entry_seq   INTEGER NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, entry_key, entry_seq)
);

CREATE INDEX IF NOT EXISTS concordance_entries_source_id_idx
    ON concordance_entries(source_id);
CREATE INDEX IF NOT EXISTS concordance_entries_source_key_idx
    ON concordance_entries(source_id, entry_key);

-- ============================================================================
-- 3. RETRIEVAL CONFIG & APP KV STORES
-- ============================================================================

-- retrieval_config: stores current pipeline version settings
CREATE TABLE IF NOT EXISTS retrieval_config (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- app_kv: general application key-value store (e.g., concordance_source_slug)
CREATE TABLE IF NOT EXISTS app_kv (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- ============================================================================
-- 4. ALIAS STATS (document frequency for alias matching)
-- ============================================================================

CREATE TABLE IF NOT EXISTS alias_stats (
    alias_norm      TEXT NOT NULL,
    document_id     INTEGER NOT NULL,
    df_chunks       INTEGER NOT NULL,
    total_chunks    INTEGER NOT NULL,
    df_percent      NUMERIC NOT NULL,
    updated_at      TIMESTAMP DEFAULT now(),
    PRIMARY KEY (alias_norm, document_id)
);

CREATE INDEX IF NOT EXISTS idx_alias_stats_alias_norm ON alias_stats(alias_norm);

-- ============================================================================
-- 5. ENTITY MERGES (audit trail for entity deduplication)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_merges (
    id                  BIGSERIAL PRIMARY KEY,
    source_entity_id    BIGINT NOT NULL,
    target_entity_id    BIGINT NOT NULL,
    source_name         TEXT,
    target_name         TEXT,
    reason              TEXT,
    merged_at           TIMESTAMPTZ DEFAULT now(),
    merged_by           TEXT DEFAULT 'cli_dedupe'
);

-- ============================================================================
-- 6. MODIFY ENTITIES TABLE (add concordance linkage)
-- ============================================================================

-- Add concordance-related columns if missing
DO $$
BEGIN
    -- source_id (required FK to concordance_sources)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entities' AND column_name = 'source_id') THEN
        ALTER TABLE entities ADD COLUMN source_id BIGINT;
    END IF;

    -- entry_id (optional FK to concordance_entries)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entities' AND column_name = 'entry_id') THEN
        ALTER TABLE entities ADD COLUMN entry_id BIGINT;
    END IF;

    -- confidence
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entities' AND column_name = 'confidence') THEN
        ALTER TABLE entities ADD COLUMN confidence TEXT;
    END IF;

    -- notes
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entities' AND column_name = 'notes') THEN
        ALTER TABLE entities ADD COLUMN notes TEXT;
    END IF;
END $$;

-- Add FKs if not exist (use DO block to handle gracefully)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'entities_source_id_fkey') THEN
        -- Only add FK if source_id column exists and concordance_sources exists
        IF EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entities' AND column_name = 'source_id')
           AND EXISTS (SELECT 1 FROM information_schema.tables 
                       WHERE table_name = 'concordance_sources') THEN
            ALTER TABLE entities ADD CONSTRAINT entities_source_id_fkey 
                FOREIGN KEY (source_id) REFERENCES concordance_sources(id) ON DELETE CASCADE;
        END IF;
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not add entities_source_id_fkey: %', SQLERRM;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'entities_entry_id_fkey') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entities' AND column_name = 'entry_id')
           AND EXISTS (SELECT 1 FROM information_schema.tables 
                       WHERE table_name = 'concordance_entries') THEN
            ALTER TABLE entities ADD CONSTRAINT entities_entry_id_fkey 
                FOREIGN KEY (entry_id) REFERENCES concordance_entries(id) ON DELETE SET NULL;
        END IF;
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not add entities_entry_id_fkey: %', SQLERRM;
END $$;

-- Indexes on entities for concordance lookups
CREATE INDEX IF NOT EXISTS entities_source_id_idx ON entities(source_id);
CREATE INDEX IF NOT EXISTS entities_entry_id_idx ON entities(entry_id);

-- ============================================================================
-- 7. MODIFY ENTITY_ALIASES TABLE (add concordance linkage)
-- ============================================================================

DO $$
BEGIN
    -- source_id
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entity_aliases' AND column_name = 'source_id') THEN
        ALTER TABLE entity_aliases ADD COLUMN source_id BIGINT;
    END IF;

    -- entry_id
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entity_aliases' AND column_name = 'entry_id') THEN
        ALTER TABLE entity_aliases ADD COLUMN entry_id BIGINT;
    END IF;

    -- alias_type (distinct from 'kind' - stores concordance-derived type)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entity_aliases' AND column_name = 'alias_type') THEN
        ALTER TABLE entity_aliases ADD COLUMN alias_type TEXT;
    END IF;

    -- confidence
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entity_aliases' AND column_name = 'confidence') THEN
        ALTER TABLE entity_aliases ADD COLUMN confidence TEXT;
    END IF;
END $$;

-- FKs for entity_aliases
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'entity_aliases_source_id_fkey') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entity_aliases' AND column_name = 'source_id') THEN
            ALTER TABLE entity_aliases ADD CONSTRAINT entity_aliases_source_id_fkey 
                FOREIGN KEY (source_id) REFERENCES concordance_sources(id) ON DELETE CASCADE;
        END IF;
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not add entity_aliases_source_id_fkey: %', SQLERRM;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'entity_aliases_entry_id_fkey') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'entity_aliases' AND column_name = 'entry_id') THEN
            ALTER TABLE entity_aliases ADD CONSTRAINT entity_aliases_entry_id_fkey 
                FOREIGN KEY (entry_id) REFERENCES concordance_entries(id) ON DELETE SET NULL;
        END IF;
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not add entity_aliases_entry_id_fkey: %', SQLERRM;
END $$;

-- Indexes
CREATE INDEX IF NOT EXISTS entity_aliases_source_id_idx ON entity_aliases(source_id);
CREATE INDEX IF NOT EXISTS entity_aliases_entry_id_idx ON entity_aliases(entry_id);
CREATE INDEX IF NOT EXISTS entity_aliases_alias_idx ON entity_aliases(alias);

-- ============================================================================
-- 8. ENTITY LINKS (relationships between entities from concordance)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_links (
    id              BIGSERIAL PRIMARY KEY,
    source_id       BIGINT NOT NULL REFERENCES concordance_sources(id) ON DELETE CASCADE,
    entry_id        BIGINT REFERENCES concordance_entries(id) ON DELETE SET NULL,
    from_entity_id  BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_entity_id    BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    link_type       TEXT NOT NULL,
    confidence      TEXT,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (from_entity_id <> to_entity_id)
);

CREATE INDEX IF NOT EXISTS entity_links_from_idx ON entity_links(from_entity_id);
CREATE INDEX IF NOT EXISTS entity_links_to_idx ON entity_links(to_entity_id);
CREATE INDEX IF NOT EXISTS entity_links_type_idx ON entity_links(link_type);
CREATE INDEX IF NOT EXISTS entity_links_source_id_idx ON entity_links(source_id);
CREATE INDEX IF NOT EXISTS entity_links_entry_id_idx ON entity_links(entry_id);

-- ============================================================================
-- 9. ENTITY CITATIONS (documentary evidence for entities/aliases/links)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_citations (
    id              BIGSERIAL PRIMARY KEY,
    source_id       BIGINT NOT NULL REFERENCES concordance_sources(id) ON DELETE CASCADE,
    entry_id        BIGINT REFERENCES concordance_entries(id) ON DELETE SET NULL,
    entity_id       BIGINT REFERENCES entities(id) ON DELETE CASCADE,
    alias_id        BIGINT REFERENCES entity_aliases(id) ON DELETE CASCADE,
    link_id         BIGINT REFERENCES entity_links(id) ON DELETE CASCADE,
    citation_text   TEXT NOT NULL,
    collection_slug TEXT,
    document_label  TEXT,
    page_list       INTEGER[],
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- At least one of entity_id, alias_id, link_id must be set
    CHECK (((entity_id IS NOT NULL)::int + (alias_id IS NOT NULL)::int + (link_id IS NOT NULL)::int) >= 1)
);

CREATE INDEX IF NOT EXISTS entity_citations_entity_idx ON entity_citations(entity_id);
CREATE INDEX IF NOT EXISTS entity_citations_alias_idx ON entity_citations(alias_id);
CREATE INDEX IF NOT EXISTS entity_citations_link_idx ON entity_citations(link_id);
CREATE INDEX IF NOT EXISTS entity_citations_source_id_idx ON entity_citations(source_id);
CREATE INDEX IF NOT EXISTS entity_citations_entry_id_idx ON entity_citations(entry_id);

-- ============================================================================
-- 10. CHUNKS: add embedding metadata and clean_text columns
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'clean_text') THEN
        ALTER TABLE chunks ADD COLUMN clean_text TEXT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'clean_text_tsv') THEN
        ALTER TABLE chunks ADD COLUMN clean_text_tsv TSVECTOR;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'embedding_model') THEN
        ALTER TABLE chunks ADD COLUMN embedding_model TEXT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'embedding_dim') THEN
        ALTER TABLE chunks ADD COLUMN embedding_dim INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'embedding_status') THEN
        ALTER TABLE chunks ADD COLUMN embedding_status TEXT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'embedded_at') THEN
        ALTER TABLE chunks ADD COLUMN embedded_at TIMESTAMPTZ;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunks' AND column_name = 'token_count') THEN
        ALTER TABLE chunks ADD COLUMN token_count INTEGER;
    END IF;
END $$;

-- Index on clean_text_tsv for full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_clean_text_tsv ON chunks USING GIN(clean_text_tsv);

-- Trigram index on clean_text for fuzzy matching (if pg_trgm is available)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunks_clean_text_trgm ON chunks USING GIST(clean_text gist_trgm_ops)';
    END IF;
END $$;

-- ============================================================================
-- 11. CHUNK_METADATA: add content_type column
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chunk_metadata' AND column_name = 'content_type') THEN
        ALTER TABLE chunk_metadata ADD COLUMN content_type TEXT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chunk_metadata_content_type ON chunk_metadata(content_type);
CREATE INDEX IF NOT EXISTS idx_chunk_metadata_collection_slug ON chunk_metadata(collection_slug);
CREATE INDEX IF NOT EXISTS idx_chunk_metadata_document_id ON chunk_metadata(document_id);

-- ============================================================================
-- 12. ENTITY_MENTIONS: expand method CHECK constraint
-- ============================================================================

-- Drop old constraint and add expanded one
DO $$
BEGIN
    -- First try to drop the old constraint
    ALTER TABLE entity_mentions DROP CONSTRAINT IF EXISTS entity_mentions_method_check;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not drop entity_mentions_method_check: %', SQLERRM;
END $$;

DO $$
BEGIN
    -- Add the expanded constraint with OCR methods
    ALTER TABLE entity_mentions ADD CONSTRAINT entity_mentions_method_check 
        CHECK (method = ANY (ARRAY[
            'alias_exact', 'alias_partial', 'alias_fuzzy', 'pattern_based',
            'ner_spacy', 'ner_transformer', 'fuzzy_known', 'hybrid',
            'human', 'model', 'ocr_lexicon', 'ocr_fuzzy', 'alias_exact_clean'
        ]::text[]));
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not add entity_mentions_method_check: %', SQLERRM;
END $$;

-- ============================================================================
-- 13. RETRIEVAL_RUNS: add missing indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS retrieval_runs_created_at_idx ON retrieval_runs(created_at DESC);

-- ============================================================================
-- 14. MENTION_REVIEW_QUEUE: ensure unique constraint exists
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint 
                   WHERE conname = 'mention_review_queue_span_unique') THEN
        ALTER TABLE mention_review_queue ADD CONSTRAINT mention_review_queue_span_unique 
            UNIQUE (chunk_id, surface_norm, start_char, end_char);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not add mention_review_queue_span_unique: %', SQLERRM;
END $$;

-- ============================================================================
-- 15. VIEWS: retrieval_config_current (needed by other views)
-- ============================================================================

CREATE OR REPLACE VIEW retrieval_config_current AS
SELECT
    MAX(value) FILTER (WHERE key = 'current_chunk_pv') AS current_chunk_pv,
    MAX(value) FILTER (WHERE key = 'current_embedding_model') AS current_embedding_model
FROM retrieval_config;

-- ============================================================================
-- 16. VIEWS: concordance_current
-- ============================================================================

CREATE OR REPLACE VIEW concordance_current AS
SELECT
    cs.slug,
    cs.id AS source_id,
    e.id AS entity_id,
    e.canonical_name,
    e.entity_type,
    ea.alias,
    ea.alias_type,
    ea.confidence AS alias_confidence
FROM concordance_sources cs
JOIN entities e ON e.source_id = cs.id
LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
WHERE cs.slug = (SELECT value FROM app_kv WHERE key = 'concordance_source_slug');

-- ============================================================================
-- 17. VIEWS: retrieval_chunks_v1
-- ============================================================================

CREATE OR REPLACE VIEW retrieval_chunks_v1 AS
SELECT
    c.id AS chunk_id,
    c.pipeline_version AS chunk_pv,
    c.text,
    c.text_tsv,
    c.embedding,
    cm.document_id,
    cm.collection_slug,
    cm.first_page_id,
    cm.last_page_id,
    cm.date_min,
    cm.date_max,
    cm.sender_set,
    cm.recipient_set,
    cm.ussr_ref_no_set,
    cm.meta_raw
FROM chunks c
JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = c.pipeline_version
WHERE c.pipeline_version = 'chunk_v1_full';

-- ============================================================================
-- 18. VIEWS: retrieval_chunks_current
-- ============================================================================

CREATE OR REPLACE VIEW retrieval_chunks_current AS
SELECT
    c.id AS chunk_id,
    c.pipeline_version AS chunk_pv,
    c.text,
    c.text_tsv,
    c.embedding,
    cm.pipeline_version AS meta_pv,
    cm.document_id,
    cm.collection_slug,
    cm.first_page_id,
    cm.last_page_id,
    cm.date_min,
    cm.date_max,
    cm.sender_set,
    cm.recipient_set,
    cm.ussr_ref_no_set,
    cm.meta_raw,
    cm.derived_at
FROM chunks c
JOIN chunk_metadata cm ON cm.chunk_id = c.id
JOIN retrieval_config_current cfg ON c.pipeline_version = cfg.current_chunk_pv
WHERE cm.pipeline_version = (
    SELECT MAX(cm2.pipeline_version) FROM chunk_metadata cm2 WHERE cm2.chunk_id = c.id
);

-- ============================================================================
-- 19. UPDATE entity_surface_tiers view to match neh
-- ============================================================================

-- Drop dependent view first, then recreate both
DROP VIEW IF EXISTS v_proposal_corpus_summary;
DROP VIEW IF EXISTS entity_surface_tiers;

CREATE VIEW entity_surface_tiers AS
SELECT
    entity_id,
    surface_norm,
    surface_display,
    entity_type,
    doc_freq,
    mention_count,
    confidence_mean,
    confidence_min,
    type_stable,
    surface_length,
    computed_at,
    CASE
        WHEN doc_freq >= 3 AND type_stable = true AND confidence_mean >= 0.95 AND surface_length >= 4 THEN 1
        WHEN doc_freq >= 3 AND type_stable = true AND confidence_mean >= 0.95 AND surface_length >= 2
             AND entity_type = ANY(ARRAY['organization', 'covername', 'cover_name']) THEN 1
        WHEN doc_freq >= 1 AND type_stable = true AND confidence_mean >= 0.85 THEN 2
        ELSE 3
    END AS tier,
    'computed'::text AS tier_reason
FROM entity_surface_stats ess;

-- ============================================================================
-- 20. UPDATE v_proposal_corpus_summary view to match neh
-- ============================================================================

CREATE VIEW v_proposal_corpus_summary AS
SELECT
    tier,
    COUNT(*) AS surface_count,
    COUNT(DISTINCT entity_id) AS entity_count,
    SUM(mention_count) AS total_mentions,
    AVG(doc_freq)::numeric(6,2) AS avg_doc_freq,
    AVG(confidence_mean)::numeric(4,3) AS avg_confidence
FROM entity_surface_tiers
GROUP BY tier
ORDER BY tier;

-- ============================================================================
-- 21. UPDATE retrieval_runs_with_session view column order
-- ============================================================================

DROP VIEW IF EXISTS retrieval_runs_with_session;
CREATE VIEW retrieval_runs_with_session AS
SELECT
    rr.id,
    rr.created_at,
    rr.query_text,
    rr.search_type,
    rr.chunk_pv,
    rr.embedding_model,
    rr.top_k,
    rr.returned_chunk_ids,
    rr.expanded_query_text,
    rr.expansion_terms,
    rr.expand_concordance,
    rr.concordance_source_slug,
    rr.session_id,
    rs.label AS session_label
FROM retrieval_runs rr
LEFT JOIN research_sessions rs ON rs.id = rr.session_id;

COMMIT;

-- Summary
SELECT 'Schema parity migration 0041 complete' AS status;
