-- Distinguish researcher-run searches from Chat-generated ones in a session.
-- Chat's searches persist as normal result sets (so researchers can open, prune,
-- and continue them) but carry origin='chat' + the question that spawned them.
ALTER TABLE search_result_sets
  ADD COLUMN IF NOT EXISTS origin TEXT NOT NULL DEFAULT 'user';
ALTER TABLE search_result_sets
  ADD COLUMN IF NOT EXISTS origin_query TEXT;
