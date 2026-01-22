-- chunk_metadata: derived, rerunnable, retrieval-facing metadata keyed by (chunk_id, pipeline_version)
CREATE TABLE chunk_metadata (
  id BIGSERIAL PRIMARY KEY,

  chunk_id BIGINT NOT NULL
    REFERENCES chunks(id) ON DELETE CASCADE,

  pipeline_version TEXT NOT NULL,
  derived_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- derived fields across spanned pages (source-dependent; may be NULL)
  date_min DATE,
  date_max DATE,

  sender_set TEXT[],
  recipient_set TEXT[],
  ussr_ref_no_set TEXT[],

  -- denormalized pointers for retrieval (avoid needing joins for basic queries)
  document_id BIGINT NOT NULL
    REFERENCES documents(id),

  collection_slug TEXT,

  -- provenance / citation helpers
  first_page_id BIGINT
    REFERENCES pages(id),

  last_page_id BIGINT
    REFERENCES pages(id),

  -- flexible extension point for source quirks and future fields
  meta_raw JSONB NOT NULL DEFAULT '{}'::jsonb,

  UNIQUE (chunk_id, pipeline_version)
);

CREATE INDEX idx_chunk_metadata_chunk
  ON chunk_metadata(chunk_id);

CREATE INDEX idx_chunk_metadata_date_rng
  ON chunk_metadata(date_min, date_max);
