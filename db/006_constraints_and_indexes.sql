-- enforce unique ordering of pages within a chunk
ALTER TABLE chunk_pages
  ADD CONSTRAINT uq_chunk_pages_span UNIQUE (chunk_id, span_order);

-- speed up ordered reads per chunk
CREATE INDEX idx_chunk_pages_chunk_order ON chunk_pages(chunk_id, span_order);