-- Migration: RDS Schema Parity
-- Generated: 2026-02-04T21:53:29.834816
-- Brings target DB schema in line with reference
--
-- Target: postgresql://friday:.[r1b:iiAwRiHtvG85IjU0c(WY:***@friday-postgres.cheoc40uk4v7.us-west-1.rds.amazonaws.com:5432/friday?sslmode=verify-full&sslrootcert=/home/maxim/rds-ca/global-bundle.pem
-- Reference: postgresql://neh:***@localhost:5432/neh?sslmode=disable
--
-- Run with: psql "$DATABASE_URL" -f <this_file>

-- ============================================================================
-- MISSING TABLES (manual review required - copying table structure)
-- ============================================================================
-- TODO: CREATE TABLE query_metrics - copy structure from reference database

-- ============================================================================
-- MISSING INDEXES
-- ============================================================================
-- chunks.idx_chunks_embedding_ivfflat
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat ON public.chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists='200');

-- entities.idx_entities_canonical_name_lower_trgm
CREATE INDEX IF NOT EXISTS idx_entities_canonical_name_lower_trgm ON public.entities USING gist (lower((canonical_name)::text) gist_trgm_ops);

-- entity_aliases.entity_aliases_alias_trgm_idx
CREATE INDEX IF NOT EXISTS entity_aliases_alias_trgm_idx ON public.entity_aliases USING gin (((alias)::text) gin_trgm_ops);

-- entity_aliases.idx_entity_aliases_alias_lower_trgm
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_lower_trgm ON public.entity_aliases USING gist (lower((alias)::text) gist_trgm_ops);
