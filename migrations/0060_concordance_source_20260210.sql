-- 0060_concordance_source_20260210.sql
-- Add concordance source for export run 2026-02-10.

INSERT INTO concordance_sources (slug, title)
VALUES ('concordance_index_export_20260210', 'concordance_index_export_20260210')
ON CONFLICT (slug) DO UPDATE SET title = EXCLUDED.title;
