-- 002_schema.sql

-- Make sure pgvector is available (harmless if already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- collections
CREATE TABLE collections (
  id BIGSERIAL PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- documents
CREATE TABLE documents (
  id BIGSERIAL PRIMARY KEY,
  collection_id BIGINT NOT NULL REFERENCES collections(id) ON DELETE RESTRICT,
  source_name TEXT NOT NULL,
  source_ref TEXT,
  volume TEXT,
  -- normalize NULL volume so uniqueness behaves how you intended
  volume_key TEXT GENERATED ALWAYS AS (COALESCE(volume, '')) STORED,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (collection_id, source_name, volume_key)
);

CREATE INDEX idx_documents_collection ON documents(collection_id);

-- pages (canonical citation unit)
CREATE TABLE pages (
  id BIGSERIAL PRIMARY KEY,
  document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,

  logical_page_label TEXT NOT NULL,   -- archival-facing label
  pdf_page_number INT,                -- nullable
  page_seq INT NOT NULL,              -- internal monotonic order within document

  language TEXT NOT NULL,
  content_role TEXT NOT NULL DEFAULT 'primary',

  raw_text TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (document_id, logical_page_label),
  UNIQUE (document_id, page_seq),
  CHECK (pdf_page_number IS NULL OR pdf_page_number > 0)
);

CREATE INDEX idx_pages_doc_seq   ON pages(document_id, page_seq);
CREATE INDEX idx_pages_doc_label ON pages(document_id, logical_page_label);

-- chunks (derived retrieval unit; independent of single page)
CREATE TABLE chunks (
  id BIGSERIAL PRIMARY KEY,
  text TEXT NOT NULL,
  embedding vector(1536),
  pipeline_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- bridge table to allow chunks spanning multiple pages
CREATE TABLE chunk_pages (
  chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  page_id  BIGINT NOT NULL REFERENCES pages(id) ON DELETE RESTRICT,
  span_order INT NOT NULL,            -- 1,2,3... order of pages within the chunk
  -- optional offsets within page raw_text (nullable for now)
  start_char INT,
  end_char   INT,
  PRIMARY KEY (chunk_id, page_id)
);

CREATE INDEX idx_chunk_pages_page ON chunk_pages(page_id);
CREATE INDEX idx_chunk_pages_chunk ON chunk_pages(chunk_id);

-- optional: vector search index (will matter once embeddings are populated)
CREATE INDEX idx_chunks_embedding_ivfflat
ON chunks USING ivfflat (embedding vector_cosine_ops);
