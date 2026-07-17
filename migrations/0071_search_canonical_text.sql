-- =============================================================================
-- Make the canonical (enriched) text searchable in the lexical Search path.
--
-- chunk_embeddings_canonical.text_canonical holds, for enriched collections:
--   * venona / vassiliev: chunk text + [MENTION_INDEX] codename => real name
--   * grand jury / trial transcripts: [TESTIMONY witness/examiner] + chunk text
-- Indexing a 'simple' tsvector over it lets Search match a real name / witness
-- even when the chunk body only contains the codename or an unnamed Q/A turn.
-- (For non-enriched collections text_canonical == chunk text, so no harm.)
-- =============================================================================

BEGIN;

ALTER TABLE chunk_embeddings_canonical
    ADD COLUMN IF NOT EXISTS text_canonical_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text_canonical, ''))) STORED;

CREATE INDEX IF NOT EXISTS idx_cec_text_canonical_tsv
    ON chunk_embeddings_canonical USING GIN (text_canonical_tsv);

COMMIT;
