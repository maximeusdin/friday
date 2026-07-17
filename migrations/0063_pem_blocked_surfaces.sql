-- Remove surface norms that must not appear in PEM (collection slugs / doc-type tokens).
-- These are not entity surfaces; they were excluded from future population in the populate script.
DELETE FROM page_entity_mentions
WHERE surface_norm IN ('huac', 'fbicomrap');
