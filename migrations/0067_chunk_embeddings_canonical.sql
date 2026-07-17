-- =============================================================================
-- Dual-Index Canonical Embeddings
--
-- Index B: canonicalized text + embeddings for semantic retrieval.
-- - Alias-scoped (venona, vassiliev): chunk text + PEM-derived MENTION_INDEX block
-- - Others: copy of chunks.text and chunks.embedding
--
-- Display/citations always use chunks (Index A). Vector search uses this table.
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS chunk_embeddings_canonical (
    id                  BIGSERIAL PRIMARY KEY,
    chunk_id            BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    pipeline_version    TEXT NOT NULL,
    embedding_model     TEXT NOT NULL DEFAULT 'text-embedding-3-small',

    text_canonical      TEXT NOT NULL,
    embedding           vector(1536),  -- same dimension as chunks.embedding; configurable in future
    rewrite_manifest    JSONB NOT NULL DEFAULT '[]'::jsonb,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (chunk_id, pipeline_version, embedding_model)
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_canonical_chunk
    ON chunk_embeddings_canonical(chunk_id);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_canonical_pv_model
    ON chunk_embeddings_canonical(pipeline_version, embedding_model);

-- IVFFlat for vector search (same as chunks)
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_canonical_embedding_ivfflat
    ON chunk_embeddings_canonical USING ivfflat (embedding vector_cosine_ops);

COMMENT ON TABLE chunk_embeddings_canonical IS
    'Index B: canonical embeddings for dual-index retrieval. Alias-scoped: text + PEM block; others: copy.';

COMMENT ON COLUMN chunk_embeddings_canonical.rewrite_manifest IS
    'Per-mapping: {surface_norm, entity_id, canonical_name, source, rule, pages, evidence}.';

COMMIT;
