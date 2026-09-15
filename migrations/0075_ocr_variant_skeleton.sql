-- 0075_ocr_variant_skeleton.sql
-- OCR-variant channel (Phase 2): skeleton column on corpus_dictionary_lexemes.
--
-- A "skeleton" collapses each character of a lexeme to its OCR-confusion class
-- representative (see config/ocr_confusion.json + retrieval/ocr_variants.py),
-- so plausible OCR corruptions of a query token can be found with an exact
-- index lookup instead of a trigram scan.
--
-- Populated python-side by scripts/build_corpus_dictionary.py
-- (automatically after a build, or via --backfill-skeletons --build-id N).
--
-- Run:
--   psql "$DATABASE_URL" -f migrations/0075_ocr_variant_skeleton.sql

SET statement_timeout = 0;

-- Table is currently empty in prod, so plain (non-CONCURRENT) DDL is fine.
ALTER TABLE corpus_dictionary_lexemes
  ADD COLUMN IF NOT EXISTS skeleton TEXT;

CREATE INDEX IF NOT EXISTS idx_cdl_build_skeleton
  ON corpus_dictionary_lexemes(build_id, skeleton);
