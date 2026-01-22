CREATE TABLE page_metadata (
  id BIGSERIAL PRIMARY KEY,
  page_id BIGINT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,

  pipeline_version TEXT NOT NULL,
  extracted_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Venona fields (nullable)
  ussr_ref_no TEXT,
  sender TEXT,
  recipient TEXT,
  cable_numbers TEXT[],        -- store multiple if present
  cable_date DATE,
  is_reissue BOOLEAN,
  is_extract BOOLEAN,

  -- Vassiliev-ish (nullable)
  base_page_label TEXT,        -- e.g. "p.123"
  dup_index INT,               -- e.g. 0..N
  notebook_id TEXT,
  volume_id TEXT,

  -- catch-all for quirks / raw captures
  meta_raw JSONB NOT NULL DEFAULT '{}'::jsonb,

  UNIQUE (page_id, pipeline_version)
);

CREATE INDEX idx_page_metadata_page      ON page_metadata(page_id);
CREATE INDEX idx_page_metadata_ref       ON page_metadata(ussr_ref_no);
CREATE INDEX idx_page_metadata_date      ON page_metadata(cable_date);
CREATE INDEX idx_page_metadata_sender    ON page_metadata(sender);
CREATE INDEX idx_page_metadata_recipient ON page_metadata(recipient);
