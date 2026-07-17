-- =============================================================================
-- V10.2 Page-Level Mention Facts Table
--
-- Each row is a fact: "On page P, the normalized surface S refers to entity E
-- (in this collection/document), with some truth level."
--
-- This is the single truth substrate for alias usage.  Tools read from
-- page_entity_mentions only (alias_referent_rules and entity_mentions are
-- authoritative/derived *input sources* that populate this table).
--
-- Normalization contract:
--   surface_norm = casefold + apostrophe normalization + trim + collapse spaces
--   (via v10_normalize.normalize_surface_for_lookup)
--   Possessive stripping is NOT baked in — kept as query-only normalization.
-- =============================================================================

BEGIN;

-- =============================================================================
-- 1. Main table
-- =============================================================================

CREATE TABLE IF NOT EXISTS page_entity_mentions (
    id                  BIGSERIAL PRIMARY KEY,

    -- Identity / location
    collection_slug     TEXT NOT NULL,
    document_id         BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id             BIGINT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,

    -- Surface
    surface_norm        TEXT NOT NULL,       -- canonical normalized key (v10_normalize)
    surface_raw         TEXT,                -- optional: original surface for debug/display

    -- Mapping
    entity_id           BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    surface_kind        TEXT,                -- codename_alias | acronym | name | general_alias | unknown | NULL

    -- Provenance
    truth_level         TEXT NOT NULL DEFAULT 'derived'
                        CHECK (truth_level IN ('authoritative', 'derived')),
    pipeline_version    TEXT NOT NULL DEFAULT '1',
    source              TEXT,                -- e.g. 'concordance', 'entity_mentions', 'alias_referent_rules'

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One fact per (page, surface, entity) — no duplicates
    CONSTRAINT page_entity_mentions_unique
        UNIQUE (page_id, surface_norm, entity_id)
);

COMMENT ON TABLE page_entity_mentions IS
    'V10.2 page-level mention facts: "on page P, surface S refers to entity E." '
    'Single truth substrate for alias usage. Tools read from this table only.';

COMMENT ON COLUMN page_entity_mentions.surface_norm IS
    'Canonical normalized surface key via v10_normalize.normalize_surface_for_lookup. '
    'No possessive stripping (query-only normalization).';

COMMENT ON COLUMN page_entity_mentions.truth_level IS
    'authoritative = from curated concordance / alias_referent_rules; '
    'derived = from entity_mentions extraction pipeline.';


-- =============================================================================
-- 2. Indexes for the 4 query shapes
-- =============================================================================

-- Shape 1: surface → entity distribution / where (optionally in collection)
-- "OSS maps to what, where?"
CREATE INDEX IF NOT EXISTS idx_pem_collection_surface
    ON page_entity_mentions (collection_slug, surface_norm);

-- Shape 1+2: surface + entity → locations
-- "CABIN where it maps to OSS"
CREATE INDEX IF NOT EXISTS idx_pem_collection_surface_entity
    ON page_entity_mentions (collection_slug, surface_norm, entity_id);

-- Shape 3: entity → surfaces (alias discovery)
-- "What codenames refer to OSS?"
CREATE INDEX IF NOT EXISTS idx_pem_collection_entity
    ON page_entity_mentions (collection_slug, entity_id);

-- Shape 4: entity → locations (evidence for entity)
-- Covered by shape 3 index + filter on page_id, but add covering if heavy
CREATE INDEX IF NOT EXISTS idx_pem_collection_entity_page
    ON page_entity_mentions (collection_slug, entity_id, page_id);

-- Optional: skip collection filter for cross-collection queries
CREATE INDEX IF NOT EXISTS idx_pem_surface_entity
    ON page_entity_mentions (surface_norm, entity_id);

-- For pipeline_version filtering during rebuild
CREATE INDEX IF NOT EXISTS idx_pem_pipeline_version
    ON page_entity_mentions (pipeline_version);


-- =============================================================================
-- 3. Index revision tracking (stored in app_kv)
-- =============================================================================

-- Ensure app_kv exists (it may already)
CREATE TABLE IF NOT EXISTS app_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed the revision key (initial value '0' — will be bumped by population job)
INSERT INTO app_kv (key, value)
VALUES ('page_entity_mentions_revision', '0')
ON CONFLICT (key) DO NOTHING;


COMMIT;
