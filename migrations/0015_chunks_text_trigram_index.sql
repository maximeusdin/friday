-- 0015_chunks_text_trigram_index.sql
-- Add GIST index on chunks.text for trigram similarity matching (soft lex)
-- This significantly speeds up word_similarity() queries used in soft lexical retrieval

BEGIN;

-- Ensure pg_trgm extension is enabled (should already be enabled by 0013)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create GIST index on chunks.text for trigram similarity matching
-- This allows PostgreSQL to use the index to filter candidates before computing similarity
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm
  ON chunks USING GIST (text gist_trgm_ops);

-- Also create index on clean_text if it exists (for future use)
-- Note: This will fail silently if clean_text column doesn't exist yet
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'chunks' AND column_name = 'clean_text'
  ) THEN
    CREATE INDEX IF NOT EXISTS idx_chunks_clean_text_trgm
      ON chunks USING GIST (clean_text gist_trgm_ops);
  END IF;
END $$;

COMMIT;
