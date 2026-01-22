-- 009_add_retrieval_runs_expansion.sql
-- Add columns to track query expansion in retrieval_runs

BEGIN;

-- Ensure retrieval_runs table exists (create if it doesn't)
CREATE TABLE IF NOT EXISTS retrieval_runs (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    search_type TEXT NOT NULL CHECK (search_type IN ('lex', 'vector', 'hybrid')),
    chunk_pv TEXT NOT NULL,
    embedding_model TEXT,
    top_k INT NOT NULL,
    returned_chunk_ids BIGINT[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Update existing constraint if it exists (drop old, add new)
DO $$
BEGIN
    -- Drop old constraint if it exists
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conrelid = 'retrieval_runs'::regclass 
          AND conname = 'retrieval_runs_search_type_check'
    ) THEN
        ALTER TABLE retrieval_runs 
        DROP CONSTRAINT retrieval_runs_search_type_check;
    END IF;
    
    -- Add new constraint that includes 'hybrid' and uses 'vector' (not 'vec')
    ALTER TABLE retrieval_runs
    ADD CONSTRAINT retrieval_runs_search_type_check 
    CHECK (search_type IN ('lex', 'vector', 'hybrid'));
END $$;

-- Add expansion tracking columns (nullable for backward compatibility)
ALTER TABLE retrieval_runs
    ADD COLUMN IF NOT EXISTS expanded_query_text TEXT,
    ADD COLUMN IF NOT EXISTS expansion_terms TEXT[],
    ADD COLUMN IF NOT EXISTS expand_concordance BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS concordance_source_slug TEXT;

-- Add index for querying by expansion status
CREATE INDEX IF NOT EXISTS idx_retrieval_runs_expand_concordance 
    ON retrieval_runs(expand_concordance);

-- Add index for querying by concordance source
CREATE INDEX IF NOT EXISTS idx_retrieval_runs_concordance_source 
    ON retrieval_runs(concordance_source_slug) 
    WHERE concordance_source_slug IS NOT NULL;

COMMIT;
