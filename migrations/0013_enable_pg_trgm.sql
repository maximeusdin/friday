BEGIN;

-- Enable pg_trgm extension for trigram-based similarity matching
-- Required for soft lexical retrieval (qv2_softlex)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Verify extension is available
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_extension
        WHERE extname = 'pg_trgm'
    ) THEN
        RAISE EXCEPTION 'pg_trgm extension failed to install';
    END IF;
END $$;

COMMIT;
