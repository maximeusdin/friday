-- 0055_cleanup_concordance_garbage.sql
-- Remove garbage entity aliases and entities from the concordance source.
-- "Garbage" = canonical_name or alias with more than 3 words, or containing
-- semicolons, em-dashes, or digits (page references / sentence fragments).

BEGIN;

-- 1) Delete garbage aliases (> 3 words, or containing garbage characters)
DELETE FROM entity_aliases
WHERE id IN (
    SELECT ea.id
    FROM entity_aliases ea
    JOIN entities e ON e.id = ea.entity_id
    JOIN concordance_sources cs ON cs.id = ea.source_id
    WHERE cs.slug = (SELECT value FROM app_kv WHERE key = 'concordance_source_slug')
      AND (
          array_length(string_to_array(trim(ea.alias), ' '), 1) > 3
          OR ea.alias ~ '[0-9;–—]'
      )
);

-- 2) Delete garbage canonical names (entities whose canonical_name is garbage)
--    First delete dependent aliases, then the entity itself.
DELETE FROM entity_aliases
WHERE entity_id IN (
    SELECT e.id
    FROM entities e
    JOIN concordance_sources cs ON cs.id = e.source_id
    WHERE cs.slug = (SELECT value FROM app_kv WHERE key = 'concordance_source_slug')
      AND (
          array_length(string_to_array(trim(e.canonical_name), ' '), 1) > 3
          OR e.canonical_name ~ '[0-9;–—]'
      )
);

DELETE FROM entity_citations
WHERE entity_id IN (
    SELECT e.id
    FROM entities e
    JOIN concordance_sources cs ON cs.id = e.source_id
    WHERE cs.slug = (SELECT value FROM app_kv WHERE key = 'concordance_source_slug')
      AND (
          array_length(string_to_array(trim(e.canonical_name), ' '), 1) > 3
          OR e.canonical_name ~ '[0-9;–—]'
      )
);

DELETE FROM entity_links
WHERE from_entity_id IN (
    SELECT e.id
    FROM entities e
    JOIN concordance_sources cs ON cs.id = e.source_id
    WHERE cs.slug = (SELECT value FROM app_kv WHERE key = 'concordance_source_slug')
      AND (
          array_length(string_to_array(trim(e.canonical_name), ' '), 1) > 3
          OR e.canonical_name ~ '[0-9;–—]'
      )
)
OR to_entity_id IN (
    SELECT e.id
    FROM entities e
    JOIN concordance_sources cs ON cs.id = e.source_id
    WHERE cs.slug = (SELECT value FROM app_kv WHERE key = 'concordance_source_slug')
      AND (
          array_length(string_to_array(trim(e.canonical_name), ' '), 1) > 3
          OR e.canonical_name ~ '[0-9;–—]'
      )
);

DELETE FROM entities
WHERE id IN (
    SELECT e.id
    FROM entities e
    JOIN concordance_sources cs ON cs.id = e.source_id
    WHERE cs.slug = (SELECT value FROM app_kv WHERE key = 'concordance_source_slug')
      AND (
          array_length(string_to_array(trim(e.canonical_name), ' '), 1) > 3
          OR e.canonical_name ~ '[0-9;–—]'
      )
);

COMMIT;
