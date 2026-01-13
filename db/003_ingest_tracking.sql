CREATE TABLE IF NOT EXISTS ingest_runs (
  id BIGSERIAL PRIMARY KEY,

  -- stable identity for the source
  source_key TEXT NOT NULL UNIQUE,         -- e.g. "venona:decrypt_1943_01_15" or file path
  source_ref TEXT,                          -- optional (path/url/catalog id)
  source_sha256 TEXT NOT NULL,              -- fingerprint of raw input bytes/text

  pipeline_version TEXT NOT NULL,           -- e.g. "v0_pages", "v1_pages+chunks"
  status TEXT NOT NULL CHECK (status IN ('success','failed','running')),

  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  error TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingest_runs_status ON ingest_runs(status);
