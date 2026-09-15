-- 0076_collection_titles_smedley_siss.sql
-- Two display-title corrections (collections.title only; slugs/ids/data untouched).
--
-- Prior titles (for rollback):
--   smedley           = 'Agnes Smedley'
--   siss_scope_soviet = 'Scope of Soviet Activity in the United States hearings and reports, Senate Internal Security Subcommittee (1956-1959)'

UPDATE collections SET title = 'Smedley, Agnes FBI files' WHERE slug = 'smedley';
UPDATE collections SET title = 'Senate Internal Security Subcommittee (1956-1959), hearings and reports on the Scope of Soviet Activity in the United States' WHERE slug = 'siss_scope_soviet';
