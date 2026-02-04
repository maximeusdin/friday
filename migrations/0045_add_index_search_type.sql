-- Migration: Add 'index' to allowed search_type values for index retrieval
-- This supports the new Index Retrieval system where results come from
-- entity_mentions, date_mentions, etc. rather than vector/lexical search.

-- First, drop the existing constraint
ALTER TABLE retrieval_runs DROP CONSTRAINT IF EXISTS retrieval_runs_search_type_check;

-- Add the new constraint with 'index' included
ALTER TABLE retrieval_runs 
ADD CONSTRAINT retrieval_runs_search_type_check 
CHECK (search_type IN ('lex', 'vector', 'hybrid', 'index'));

-- Add a comment explaining the search types
COMMENT ON COLUMN retrieval_runs.search_type IS 'Search type: lex (lexical/tsvector), vector (embedding), hybrid (both), index (mention indexes)';
