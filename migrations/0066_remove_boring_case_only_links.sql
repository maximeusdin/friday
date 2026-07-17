-- Remove boring case-only links (gnome->GNOME) from PEM, entity_aliases, entity_links.
-- These add no value beyond the canonical form.

-- 1. entity_links: links where from and to entities have same canonical (case-insensitive)
DELETE FROM entity_links el
USING entities e1, entities e2
WHERE el.from_entity_id = e1.id
  AND el.to_entity_id = e2.id
  AND LOWER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(e1.canonical_name, '[^\w\s]', ' ', 'g'), '\s+', ' ', 'g')))
    = LOWER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(e2.canonical_name, '[^\w\s]', ' ', 'g'), '\s+', ' ', 'g')));

-- 2. entity_aliases: aliases that are just case variants of canonical (keep canonical, drop others)
DELETE FROM entity_aliases ea
USING entities e
WHERE ea.entity_id = e.id
  AND ea.alias_norm = LOWER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(e.canonical_name, '[^\w\s]', ' ', 'g'), '\s+', ' ', 'g')))
  AND ea.alias IS DISTINCT FROM e.canonical_name;

-- 3. PEM: surfaces that are just the canonical name (case-normalized)
DELETE FROM page_entity_mentions pem
USING entities e
WHERE pem.entity_id = e.id
  AND pem.surface_norm = LOWER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(e.canonical_name, '[^\w\s]', ' ', 'g'), '\s+', ' ', 'g')));
