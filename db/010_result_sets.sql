-- Persisted Result Sets (named, immutable snapshots of retrieval output)
-- Notes:
--   - result_sets references exactly one retrieval_run
--   - immutable: no UPDATE / DELETE
--   - deterministic regeneration is implied because retrieval_runs stores returned_chunk_ids + run params

BEGIN;

CREATE TABLE IF NOT EXISTS result_sets (
  id               BIGSERIAL PRIMARY KEY,
  name             TEXT NOT NULL,
  retrieval_run_id BIGINT NOT NULL REFERENCES retrieval_runs(id) ON DELETE RESTRICT,
  -- ordered final selection to cite (often equals retrieval_runs.returned_chunk_ids, but can be a subset)
  chunk_ids        BIGINT[] NOT NULL,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  notes            TEXT,

  CONSTRAINT result_sets_name_nonempty
    CHECK (btrim(name) <> ''),

  CONSTRAINT result_sets_chunk_ids_nonempty
    CHECK (array_length(chunk_ids, 1) IS NOT NULL AND array_length(chunk_ids, 1) > 0)
);

-- Pick ONE uniqueness policy:

-- A) Global unique names (stable handle like "venona_berlin_1950_v1")
CREATE UNIQUE INDEX IF NOT EXISTS result_sets_name_uniq
ON result_sets (name);

-- -- B) Alternatively: unique per run (uncomment and comment out A if you prefer)
-- DROP INDEX IF EXISTS result_sets_name_uniq;
-- CREATE UNIQUE INDEX IF NOT EXISTS result_sets_run_name_uniq
-- ON result_sets (retrieval_run_id, name);

CREATE INDEX IF NOT EXISTS result_sets_retrieval_run_id_idx
ON result_sets (retrieval_run_id, created_at DESC);

-- Immutability: disallow UPDATE/DELETE
CREATE OR REPLACE FUNCTION prevent_result_sets_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF TG_OP = 'UPDATE' THEN
    RAISE EXCEPTION 'result_sets are immutable: updates are not allowed (id=%)', OLD.id
      USING ERRCODE = '55000';
  ELSIF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'result_sets are immutable: deletes are not allowed (id=%)', OLD.id
      USING ERRCODE = '55000';
  END IF;
  RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_result_sets_update ON result_sets;
CREATE TRIGGER trg_prevent_result_sets_update
BEFORE UPDATE ON result_sets
FOR EACH ROW
EXECUTE FUNCTION prevent_result_sets_mutation();

DROP TRIGGER IF EXISTS trg_prevent_result_sets_delete ON result_sets;
CREATE TRIGGER trg_prevent_result_sets_delete
BEFORE DELETE ON result_sets
FOR EACH ROW
EXECUTE FUNCTION prevent_result_sets_mutation();

COMMENT ON TABLE result_sets IS
'Named, immutable snapshot of a retrieval run output. Stores ordered chunk_ids for citation and later comparison/export.';
COMMENT ON COLUMN result_sets.chunk_ids IS
'Ordered list of chunk IDs (ranked/cited order). Often equals retrieval_runs.returned_chunk_ids or a curated subset; immutable after insert.';
COMMENT ON COLUMN result_sets.retrieval_run_id IS
'References exactly one retrieval run; regeneration is deterministic given retrieval_runs parameters and returned_chunk_ids.';

COMMIT;