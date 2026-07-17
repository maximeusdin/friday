-- =============================================================================
-- V10 Identity Layer Migration
--
-- Adds infrastructure for the V10 scope-aware alias identity system:
-- 1. chunk_mentions_json on evidence_items (per-chunk extraction artifacts)
-- 2. mapping_hypotheses_json on v9_runs (alias mapping state)
-- 3. alias_referent_rules table (doc/page-scoped referent mappings)
-- =============================================================================

BEGIN;

-- =============================================================================
-- 1. Evidence layer extensions
-- =============================================================================

-- chunk_mentions_json: stores ChunkMentionsV10 with embedded document_id/page_no
-- so that ThinkDeeper rehydration is independent from DB joins
ALTER TABLE evidence_items
    ADD COLUMN IF NOT EXISTS chunk_mentions_json JSONB;

COMMENT ON COLUMN evidence_items.chunk_mentions_json IS
    'V10 ChunkMentionsV10: entity/alias mentions + signals per chunk. '
    'Embeds document_id and page_no for ThinkDeeper rehydration independence.';

-- mapping_hypotheses_json: stores both contextual and general alias hypotheses
ALTER TABLE v9_runs
    ADD COLUMN IF NOT EXISTS mapping_hypotheses_json JSONB;

COMMENT ON COLUMN v9_runs.mapping_hypotheses_json IS
    'V10 alias mapping hypotheses (contextual doc/page-scoped + general collection-wide). '
    'Persisted for ThinkDeeper rehydration.';

-- Optional GIN index on chunk_mentions_json for analytical queries
CREATE INDEX IF NOT EXISTS idx_evidence_items_chunk_mentions_gin
    ON evidence_items USING gin (chunk_mentions_json)
    WHERE chunk_mentions_json IS NOT NULL;


-- =============================================================================
-- 2. Alias referent rules table
--
-- Known alias->entity mappings scoped to specific documents and optionally
-- page ranges.  These are the PRIMARY anti-ambiguity mechanism — consulted
-- BEFORE any general collection-level hypothesis.
--
-- Sources:
-- - Curated data (human-entered)
-- - Prior confirmed hypotheses (promoted by the agent)
-- - Deterministic extraction (e.g. "KING = Julius Rosenberg" in a decrypt)
-- =============================================================================

CREATE TABLE IF NOT EXISTS alias_referent_rules (
    id              BIGSERIAL PRIMARY KEY,
    collection_slug TEXT NOT NULL,
    alias_text      TEXT NOT NULL,
    document_id     BIGINT NOT NULL REFERENCES documents(id),
    page_from       INT,               -- NULL = entire document
    page_to         INT,
    entity_id       BIGINT NOT NULL REFERENCES entities(id),
    status          TEXT NOT NULL DEFAULT 'confirmed',  -- confirmed|possible|rejected
    note            TEXT DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Primary lookup index: fast resolution by (collection, alias, document)
-- with page range for interval queries
CREATE INDEX IF NOT EXISTS idx_alias_referent_rules_lookup
    ON alias_referent_rules (collection_slug, alias_text, document_id, page_from, page_to);

-- Reverse lookup: find all rules for a given entity
CREATE INDEX IF NOT EXISTS idx_alias_referent_rules_entity
    ON alias_referent_rules (entity_id);

-- Referent rule precedence semantics (enforced in application code):
-- When multiple rules match (alias, document_id, page_no):
--   1. Page-scoped (page_from IS NOT NULL) beats doc-wide (page_from IS NULL)
--   2. Narrower interval wins (smallest page_to - page_from)
--   3. Status priority: confirmed > possible > rejected
--   4. If still tied: return all candidates (agent handles)

COMMENT ON TABLE alias_referent_rules IS
    'V10 alias referent rules: known alias->entity mappings scoped to '
    'specific documents and optionally page ranges. Primary anti-ambiguity '
    'mechanism consulted before any collection-level hypothesis.';


-- =============================================================================
-- 3. Span lattice storage on v9_runs (for ThinkDeeper rehydration)
-- =============================================================================

ALTER TABLE v9_runs
    ADD COLUMN IF NOT EXISTS span_lattice_json JSONB;

COMMENT ON COLUMN v9_runs.span_lattice_json IS
    'V10 SpanLatticeV10: query candidate parse with overlapping spans. '
    'Persisted for ThinkDeeper rehydration.';


-- =============================================================================
-- 4. Resolution plan storage on v9_runs
-- =============================================================================

ALTER TABLE v9_runs
    ADD COLUMN IF NOT EXISTS resolution_plan_json JSONB;

COMMENT ON COLUMN v9_runs.resolution_plan_json IS
    'V10 ResolutionPlanV10: agent decisions per round. '
    'Persisted for ThinkDeeper to see what the prior run decided.';


COMMIT;
