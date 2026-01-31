-- Performance indexes for CO_OCCURS_WITH and CO_LOCATED primitives
-- These primitives can become slow hotspots as corpus grows
--
-- Key query patterns:
-- 1. CO_OCCURS_WITH(entity_id, window="chunk"): 
--    EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = X AND em.entity_id = Y)
--
-- 2. CO_OCCURS_WITH(entity_id, window="document"):
--    EXISTS (SELECT 1 FROM entity_mentions em WHERE em.document_id = X AND em.entity_id = Y)
--
-- 3. CO_LOCATED(entity_a, entity_b, scope="chunk"):
--    EXISTS (entity_a in chunk) AND EXISTS (entity_b in chunk)
--
-- 4. CO_LOCATED(entity_a, entity_b, scope="document"):
--    EXISTS (SELECT 1 FROM entity_mentions em_a 
--            JOIN entity_mentions em_b ON em_b.document_id = em_a.document_id
--            WHERE em_a.entity_id = A AND em_b.entity_id = B AND em_a.document_id = X)
--
-- DEPLOYMENT NOTE:
-- These indexes use CONCURRENTLY to avoid locking tables during creation.
-- CONCURRENTLY cannot run inside a transaction, so each CREATE INDEX is separate.
-- Run migrations BEFORE deploying new code.

-- ============================================================================
-- 1. Composite index for chunk-level entity lookups
-- ============================================================================
-- Optimizes: WHERE chunk_id = X AND entity_id = Y
-- This is the most common CO_OCCURS_WITH pattern
-- Note: entity_mentions_entity_id_idx is (entity_id, chunk_id), this is reversed

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_mentions_chunk_entity 
ON entity_mentions(chunk_id, entity_id);

COMMENT ON INDEX idx_entity_mentions_chunk_entity IS
'Composite index for CO_OCCURS_WITH chunk-window queries: quickly find if entity X is mentioned in chunk Y.';

-- ============================================================================
-- 2. Composite index for document-level entity lookups  
-- ============================================================================
-- Optimizes: WHERE document_id = X AND entity_id = Y
-- Critical for CO_OCCURS_WITH document-window and CO_LOCATED document-scope

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_mentions_document_entity
ON entity_mentions(document_id, entity_id);

COMMENT ON INDEX idx_entity_mentions_document_entity IS
'Composite index for CO_OCCURS_WITH document-window and CO_LOCATED document-scope queries.';

-- ============================================================================
-- 3. Index for chunk_metadata document lookups
-- ============================================================================
-- Optimizes: SELECT document_id FROM chunk_metadata WHERE chunk_id = X
-- Used in document-window scope resolution

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunk_metadata_chunk_document
ON chunk_metadata(chunk_id, document_id);

COMMENT ON INDEX idx_chunk_metadata_chunk_document IS
'Covering index for quick chunk_id -> document_id lookups in co-occurrence primitives.';

-- ============================================================================
-- Performance Notes
-- ============================================================================
-- Expected improvements:
-- - CO_OCCURS_WITH chunk-window: index-only scan on idx_entity_mentions_chunk_entity
-- - CO_OCCURS_WITH document-window: index-only scan on idx_entity_mentions_document_entity  
-- - CO_LOCATED document-scope: efficient join on idx_entity_mentions_document_entity
--
-- Monitor these queries with EXPLAIN ANALYZE if performance degrades at scale.
-- Consider partial indexes if certain entity_ids dominate queries.
--
-- ROLLBACK if needed:
-- DROP INDEX CONCURRENTLY idx_entity_mentions_chunk_entity;
-- DROP INDEX CONCURRENTLY idx_entity_mentions_document_entity;
-- DROP INDEX CONCURRENTLY idx_chunk_metadata_chunk_document;
