-- Reversible removal of search result hits: a hidden flag instead of hard delete,
-- so researchers can prune result lists and restore rows later.
ALTER TABLE search_result_page_hits
  ADD COLUMN IF NOT EXISTS hidden BOOLEAN NOT NULL DEFAULT false;
