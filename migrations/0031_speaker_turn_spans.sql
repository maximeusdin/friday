-- Migration: Add speaker_turn_spans JSONB for precise per-speaker attribution
-- This enables deterministic quoting: "show me exactly where Welch speaks in this chunk"

-- =============================================================================
-- 1. Add speaker_turn_spans to chunks
-- =============================================================================

-- Format: [{"speaker":"WELCH","turn_id_start":450,"turn_id_end":454}, ...]
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS speaker_turn_spans JSONB;

-- Index for JSONB queries (e.g., find chunks where specific speaker has spans)
CREATE INDEX IF NOT EXISTS idx_chunks_speaker_turn_spans ON chunks USING GIN(speaker_turn_spans jsonb_path_ops);

-- =============================================================================
-- 2. Enhance chunk_turns with additional metadata for direct queries
-- =============================================================================

-- Add speaker_norm to chunk_turns for direct filtering without join
ALTER TABLE chunk_turns ADD COLUMN IF NOT EXISTS speaker_norm TEXT;

-- Index for speaker filtering on chunk_turns
CREATE INDEX IF NOT EXISTS idx_chunk_turns_speaker_norm ON chunk_turns(speaker_norm);

-- =============================================================================
-- 3. Comments
-- =============================================================================

COMMENT ON COLUMN chunks.speaker_turn_spans IS 'JSONB array of contiguous speaker spans within the chunk, for precise attribution';
COMMENT ON COLUMN chunk_turns.speaker_norm IS 'Denormalized speaker_norm from transcript_turns for direct filtering';
