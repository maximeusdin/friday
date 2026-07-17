-- Remove all PEM rows derived from entity_mentions. PEM is now concordance-only
-- (alias_referent_rules + concordance). entity_mentions-derived rows polluted
-- PEM with generic words (american, bureau, soviet) via OCR fuzzy matching.
--
-- Run populate_page_entity_mentions after this to refresh from concordance.

DELETE FROM page_entity_mentions
WHERE source = 'entity_mentions';
