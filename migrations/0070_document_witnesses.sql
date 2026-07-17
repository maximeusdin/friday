-- =============================================================================
-- Witness index for transcript documents (grand jury / hearing testimony).
--
-- Grand jury transcripts of witness testimony have no table of contents mapping
-- witness -> pages. We construct one (scripts/build_witness_index.py) by
-- detecting swearing-in markers, and store it here keyed by document. One row
-- per witness *appearance* (a witness recalled on another day = another row).
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS document_witnesses (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    appearance_seq  INT NOT NULL,        -- order of appearance within the document
    witness_name    TEXT NOT NULL,
    start_page      INT NOT NULL,        -- PDF page where this appearance begins
    end_page        INT NOT NULL,        -- last PDF page of this appearance
    page_count      INT,
    testimony_date  TEXT,                -- as printed, e.g. "July 22, 1947"
    examiner        TEXT,                -- e.g. "MR. DONEGAN"
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (document_id, appearance_seq)
);

CREATE INDEX IF NOT EXISTS idx_document_witnesses_doc
    ON document_witnesses(document_id);

COMMIT;
