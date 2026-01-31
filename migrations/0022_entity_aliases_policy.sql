-- Day 10: Entity Aliases Policy Columns
-- Adds policy controls to prevent false positive matches for short/common words
-- like "TO", "Ref", etc.

BEGIN;

-- ============================================================================
-- Add policy columns to entity_aliases
-- ============================================================================

ALTER TABLE entity_aliases
  ADD COLUMN IF NOT EXISTS is_auto_match BOOLEAN NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS min_chars INT NOT NULL DEFAULT 1,  -- Minimum characters (renamed from min_token_len)
  ADD COLUMN IF NOT EXISTS match_case TEXT NOT NULL DEFAULT 'any' CHECK (match_case IN ('any', 'case_sensitive')),
  ADD COLUMN IF NOT EXISTS match_mode TEXT NOT NULL DEFAULT 'token' CHECK (match_mode IN ('token', 'substring', 'phrase')),
  ADD COLUMN IF NOT EXISTS is_numeric_entity BOOLEAN NOT NULL DEFAULT false,  -- Opt-in for numeric entities
  ADD COLUMN IF NOT EXISTS notes TEXT NULL;

-- Migrate min_token_len to min_chars if column exists
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns 
             WHERE table_name = 'entity_aliases' AND column_name = 'min_token_len') THEN
    UPDATE entity_aliases
    SET min_chars = min_token_len
    WHERE min_chars = 1;  -- Only update if still at default
    ALTER TABLE entity_aliases DROP COLUMN IF EXISTS min_token_len;
  END IF;
END $$;

-- ============================================================================
-- Set conservative defaults for short/common aliases
-- ============================================================================

-- Disable auto-match for very short aliases (≤2 chars) unless explicitly enabled
-- These are often false positives (TO, Ref, etc.)
UPDATE entity_aliases
SET is_auto_match = false,
    notes = COALESCE(notes || '; ', '') || 'Auto-disabled: alias length ≤2 chars (likely false positive)'
WHERE LENGTH(TRIM(alias)) <= 2
  AND is_auto_match = true
  AND notes IS NULL OR notes NOT LIKE '%explicitly enabled%';

-- Set case-sensitive matching for short aliases (≤3 chars)
-- This helps prevent "TO" matching "to", "Ref" matching "ref", etc.
UPDATE entity_aliases
SET match_case = 'case_sensitive',
    notes = COALESCE(notes || '; ', '') || 'Case-sensitive: alias length ≤3 chars'
WHERE LENGTH(TRIM(alias)) <= 3
  AND match_case = 'any';

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON COLUMN entity_aliases.is_auto_match IS
'If false, this alias will not be automatically matched during extraction. Requires manual review/adjudication.';

COMMENT ON COLUMN entity_aliases.min_chars IS
'Minimum character length required for matching (applied to first token for multi-word aliases).';

COMMENT ON COLUMN entity_aliases.is_numeric_entity IS
'If true, allows purely numeric aliases to be matched as entities. Default false (numeric strings belong in date_mentions).';

COMMENT ON COLUMN entity_aliases.match_case IS
'Case matching policy: "any" (case-insensitive) or "case_sensitive" (must match case exactly).';

COMMENT ON COLUMN entity_aliases.match_mode IS
'Matching mode: "token" (word boundaries), "substring" (anywhere in text), "phrase" (exact phrase).';

COMMENT ON COLUMN entity_aliases.notes IS
'Notes about this alias policy, why it was configured, etc.';

COMMIT;
