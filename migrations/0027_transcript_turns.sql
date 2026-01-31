-- Migration: Add transcript_turns table and speaker tracking for hearing transcripts
-- This enables speaker-aware retrieval and attribution

-- =============================================================================
-- 1. Transcript Turns Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS transcript_turns (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    turn_id INT NOT NULL,                    -- Monotonic within document (1, 2, 3...)
    
    -- Speaker info
    speaker_raw TEXT NOT NULL,               -- Exact label as printed: "Mr. COHN."
    speaker_norm TEXT NOT NULL,              -- Normalized: "MR COHN"
    speaker_role TEXT,                       -- 'counsel', 'witness', 'senator', 'chair', 'stage'
    
    -- Turn content
    turn_text TEXT NOT NULL,
    
    -- Location
    page_start INT NOT NULL,
    page_end INT NOT NULL,
    char_start INT,                          -- Optional: offset in concatenated doc text
    char_end INT,
    
    -- Metadata
    is_stage_direction BOOLEAN DEFAULT FALSE, -- [Laughter], [Recess], etc.
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(document_id, turn_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_transcript_turns_document ON transcript_turns(document_id);
CREATE INDEX IF NOT EXISTS idx_transcript_turns_speaker_norm ON transcript_turns(speaker_norm);
CREATE INDEX IF NOT EXISTS idx_transcript_turns_speaker_role ON transcript_turns(speaker_role);


-- =============================================================================
-- 2. Speaker Normalization/Override Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS speaker_map (
    id BIGSERIAL PRIMARY KEY,
    speaker_norm TEXT NOT NULL UNIQUE,       -- Normalized speaker key
    canonical_name TEXT,                     -- Human-readable name
    entity_id BIGINT REFERENCES entities(id),-- Link to entity if matched
    role TEXT,                               -- Default role for this speaker
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_speaker_map_entity ON speaker_map(entity_id);


-- =============================================================================
-- 3. Add speaker tracking columns to chunks
-- =============================================================================

-- Speaker arrays and turn tracking
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS turn_id_start INT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS turn_id_end INT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS turn_count INT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS speaker_norms TEXT[];
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS primary_speaker_norm TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embed_text TEXT;  -- Speaker-tagged version for embeddings

-- Index for speaker filtering
CREATE INDEX IF NOT EXISTS idx_chunks_speaker_norms ON chunks USING GIN(speaker_norms);
CREATE INDEX IF NOT EXISTS idx_chunks_primary_speaker ON chunks(primary_speaker_norm);


-- =============================================================================
-- 4. Chunk-Turn junction (optional, for precise turn mapping)
-- =============================================================================

CREATE TABLE IF NOT EXISTS chunk_turns (
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    turn_id BIGINT NOT NULL REFERENCES transcript_turns(id) ON DELETE CASCADE,
    span_order INT NOT NULL DEFAULT 1,
    PRIMARY KEY (chunk_id, turn_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_turns_turn ON chunk_turns(turn_id);


-- =============================================================================
-- 5. Helper view for speaker retrieval
-- =============================================================================

CREATE OR REPLACE VIEW v_speaker_chunks AS
SELECT 
    c.id as chunk_id,
    c.text,
    c.embed_text,
    c.speaker_norms,
    c.primary_speaker_norm,
    c.turn_id_start,
    c.turn_id_end,
    c.turn_count,
    d.id as document_id,
    d.source_name,
    col.slug as collection_slug
FROM chunks c
JOIN chunk_pages cp ON cp.chunk_id = c.id
JOIN pages p ON p.id = cp.page_id
JOIN documents d ON d.id = p.document_id
JOIN collections col ON col.id = d.collection_id
WHERE c.speaker_norms IS NOT NULL
GROUP BY c.id, d.id, col.slug;


COMMENT ON TABLE transcript_turns IS 'Individual speaker turns from hearing transcripts, enabling speaker-aware retrieval';
COMMENT ON TABLE speaker_map IS 'Maps normalized speaker names to canonical names and entities';
COMMENT ON TABLE chunk_turns IS 'Links chunks to their constituent turns for precise attribution';
