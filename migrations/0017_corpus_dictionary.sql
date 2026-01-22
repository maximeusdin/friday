-- Materialized corpus dictionary for per-token fuzzy lexical expansion
-- This supports traceable, reproducible fuzzy variants derived from the corpus.

-- A "build" is a snapshot of lexemes for a specific corpus slice:
-- (chunk_pv, collection_slug, norm_version). We keep builds append-only.

CREATE TABLE IF NOT EXISTS corpus_dictionary_builds (
  id BIGSERIAL PRIMARY KEY,
  built_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  chunk_pv TEXT NOT NULL,
  collection_slug TEXT NULL,
  norm_version TEXT NOT NULL DEFAULT 'norm_v1',
  notes TEXT
);

-- Unique lexemes for a given build.
CREATE TABLE IF NOT EXISTS corpus_dictionary_lexemes (
  build_id BIGINT NOT NULL REFERENCES corpus_dictionary_builds(id) ON DELETE CASCADE,
  lexeme TEXT NOT NULL,
  chunk_freq INT NOT NULL DEFAULT 0,
  PRIMARY KEY (build_id, lexeme)
);

-- Fast lookup by build.
CREATE INDEX IF NOT EXISTS idx_corpus_dictionary_lexemes_build
  ON corpus_dictionary_lexemes (build_id);

-- Fast trigram similarity lookup on lexeme text (requires pg_trgm).
-- Note: we index all lexemes; queries always filter by build_id.
CREATE INDEX IF NOT EXISTS idx_corpus_dictionary_lexemes_lexeme_trgm
  ON corpus_dictionary_lexemes USING GIN (lexeme gin_trgm_ops);

-- Helpful index to find latest build for a slice quickly.
CREATE INDEX IF NOT EXISTS idx_corpus_dictionary_builds_slice_latest
  ON corpus_dictionary_builds (chunk_pv, collection_slug, norm_version, built_at DESC, id DESC);

