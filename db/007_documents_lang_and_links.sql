-- Future-proofing for multilingual / parallel editions

ALTER TABLE documents
  ADD COLUMN source_lang TEXT NOT NULL DEFAULT 'en',
  ADD COLUMN source_edition TEXT,
  ADD COLUMN related_document_id BIGINT REFERENCES documents(id) ON DELETE SET NULL;

CREATE INDEX idx_documents_related ON documents(related_document_id);