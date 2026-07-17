-- 0068_search_concordance.sql
-- Search tab: tsv_simple for non-stemming FTS, search_result_sets, search_result_page_hits.
--
-- Run with statement_timeout=0 for large chunks table:
--   psql "$DATABASE_URL" -v statement_timeout=0 -f migrations/0068_search_concordance.sql

SET statement_timeout = 0;
-- tsv_simple ALTER needs ~69MB; default maintenance_work_mem is 64MB
SET maintenance_work_mem = '128MB';

-- UUID extension for search_result_sets.id
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- tsv_simple: non-stemming FTS for Search tab (proper nouns, codenames)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'chunks' AND column_name = 'tsv_simple'
  ) THEN
    ALTER TABLE chunks ADD COLUMN tsv_simple tsvector
      GENERATED ALWAYS AS (to_tsvector('simple', coalesce(clean_text, text, ''))) STORED;
  END IF;
END $$;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_tsv_simple_gin
  ON chunks USING GIN(tsv_simple);

-- search_result_sets: deterministic concordance-style search results
CREATE TABLE IF NOT EXISTS search_result_sets (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_sub TEXT NOT NULL,
  session_id BIGINT REFERENCES research_sessions(id) ON DELETE SET NULL,
  scope_json JSONB NOT NULL,
  query_raw TEXT NOT NULL,
  query_ast_json JSONB,
  mode TEXT NOT NULL DEFAULT 'exact',
  unit TEXT NOT NULL DEFAULT 'page',
  sort_order TEXT NOT NULL DEFAULT 'canonical',
  alias_expand BOOLEAN NOT NULL DEFAULT true,
  is_exhaustive BOOLEAN NOT NULL DEFAULT true,
  total_hits INT,
  coverage_json JSONB,
  expanded_terms_json JSONB,
  query_display TEXT,
  status TEXT NOT NULL DEFAULT 'running',
  error_message TEXT
);

-- search_result_page_hits: materialized page-level hits
CREATE TABLE IF NOT EXISTS search_result_page_hits (
  result_set_id UUID NOT NULL REFERENCES search_result_sets(id) ON DELETE CASCADE,
  collection_id BIGINT NOT NULL,
  document_id BIGINT NOT NULL,
  page_id BIGINT NOT NULL,
  page_seq INT NOT NULL,
  pdf_page_number INT NOT NULL,
  chunk_id BIGINT NOT NULL,
  snippet TEXT,
  hit_rank INT,
  PRIMARY KEY (result_set_id, collection_id, document_id, page_id)
);

CREATE INDEX IF NOT EXISTS idx_search_result_page_hits_set
  ON search_result_page_hits(result_set_id);

CREATE INDEX IF NOT EXISTS idx_search_result_page_hits_order
  ON search_result_page_hits(result_set_id, collection_id, document_id, page_seq);
