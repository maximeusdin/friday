"""
Proposal Gating - Controls what surfaces can be proposed by OCR/NER extraction.

This module provides the read-only interface for extraction code to check
whether a surface is eligible for proposal-based matching.

Resolution Precedence:
1. Human override (entity_alias_overrides)
2. Exact alias match (entity_aliases)
3. Proposal corpus tier (entity_surface_tiers)
4. OCR/NER suggestions (lowest priority)

Usage:
    from retrieval.proposal_gating import ProposalGate
    
    gate = ProposalGate(conn)
    
    # Check single surface
    if gate.is_proposable("rosenberg", min_tier=2):
        # OK to propose this surface
        ...
    
    # Get all tier-1 surfaces for a type
    surfaces = gate.get_proposal_surfaces(tier=1, entity_type="person")
"""

import hashlib
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass

from retrieval.surface_norm import normalize_surface, compute_surface_hash


@dataclass
class ProposalSurface:
    """A surface eligible for proposal-based matching."""
    entity_id: int
    surface_norm: str
    surface_display: str
    entity_type: str
    doc_freq: int
    tier: int


@dataclass  
class ProposalStats:
    """Summary statistics for proposal corpus."""
    tier: int
    surface_count: int
    entity_count: int
    total_mentions: int
    avg_doc_freq: float
    avg_confidence: float


class ProposalGate:
    """
    Gating mechanism for proposal-based entity extraction.
    
    Extraction code should use this to check if surfaces can be proposed
    from noisy sources (OCR, NER).
    """
    
    def __init__(self, conn, cache_enabled: bool = True):
        """
        Initialize with database connection.
        
        Args:
            conn: psycopg2 connection
            cache_enabled: Whether to cache tier lookups (default True)
        """
        self.conn = conn
        self.cache_enabled = cache_enabled
        self._tier_cache: Dict[str, Optional[int]] = {}
        self._banned_cache: Set[str] = set()
        self._override_cache_loaded = False
    
    def _load_override_cache(self):
        """Load global overrides into cache."""
        if self._override_cache_loaded:
            return
        
        with self.conn.cursor() as cur:
            # Load globally banned surfaces
            cur.execute("""
                SELECT surface_norm FROM entity_alias_overrides
                WHERE banned = TRUE AND scope = 'global'
            """)
            self._banned_cache = {row[0] for row in cur.fetchall()}
        
        self._override_cache_loaded = True
    
    def is_proposable(
        self, 
        surface: str, 
        min_tier: int = 2,
        collection_id: Optional[int] = None,
        document_id: Optional[int] = None,
        allow_prefer_override: bool = True,
    ) -> bool:
        """
        Check if a surface is eligible for proposal-based matching.
        
        Args:
            surface: Raw surface text (will be normalized)
            min_tier: Maximum tier to accept (1=strict, 2=default, 3=permissive)
            collection_id: Optional collection scope for overrides
            document_id: Optional document scope for overrides
            allow_prefer_override: If True, a prefer override can make a surface
                                   proposable even if its tier is too low
        
        Returns:
            True if surface can be proposed
        """
        surface_norm = normalize_surface(surface)
        
        if not surface_norm:
            return False
        
        # Load override cache
        self._load_override_cache()
        
        # Check global ban first (bans always block)
        if surface_norm in self._banned_cache:
            return False
        
        # Query database for scope-specific overrides
        with self.conn.cursor() as cur:
            # Check for scope-specific ban
            if collection_id or document_id:
                cur.execute("""
                    SELECT 1 FROM entity_alias_overrides
                    WHERE surface_norm = %s AND banned = TRUE
                      AND (
                          (scope = 'collection' AND scope_collection_id = %s)
                          OR (scope = 'document' AND scope_document_id = %s)
                      )
                    LIMIT 1
                """, (surface_norm, collection_id, document_id))
                if cur.fetchone():
                    return False
            
            # Check for prefer override (can boost proposability)
            if allow_prefer_override:
                cur.execute("""
                    SELECT 1 FROM entity_alias_overrides
                    WHERE surface_norm = %s AND forced_entity_id IS NOT NULL
                      AND (
                          scope = 'global'
                          OR (scope = 'collection' AND scope_collection_id = %s)
                          OR (scope = 'document' AND scope_document_id = %s)
                      )
                    LIMIT 1
                """, (surface_norm, collection_id, document_id))
                if cur.fetchone():
                    # Has a prefer override - allow proposing regardless of tier
                    return True
        
        # Check cache for tier
        if self.cache_enabled and surface_norm in self._tier_cache:
            tier = self._tier_cache[surface_norm]
            return tier is not None and tier <= min_tier
        
        # Query tier from proposal corpus
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT MIN(tier) FROM entity_surface_tiers
                WHERE surface_norm = %s
            """, (surface_norm,))
            row = cur.fetchone()
            tier = row[0] if row else None
        
        # Cache result
        if self.cache_enabled:
            self._tier_cache[surface_norm] = tier
        
        return tier is not None and tier <= min_tier
    
    def get_tier(self, surface: str) -> Optional[int]:
        """
        Get the tier for a surface (1, 2, 3, or None if not in corpus).
        
        Args:
            surface: Raw surface text
        
        Returns:
            Tier number or None if not in proposal corpus
        """
        surface_norm = normalize_surface(surface)
        
        if self.cache_enabled and surface_norm in self._tier_cache:
            return self._tier_cache[surface_norm]
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT MIN(tier) FROM entity_surface_tiers
                WHERE surface_norm = %s
            """, (surface_norm,))
            row = cur.fetchone()
            tier = row[0] if row else None
        
        if self.cache_enabled:
            self._tier_cache[surface_norm] = tier
        
        return tier
    
    def get_proposal_surfaces(
        self,
        tier: int = 2,
        entity_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[ProposalSurface]:
        """
        Get surfaces eligible for proposal at the given tier.
        
        Args:
            tier: Maximum tier to include (1, 2, or 3)
            entity_type: Optional filter by entity type
            limit: Maximum number of results
        
        Returns:
            List of ProposalSurface objects
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, surface_norm, surface_display, entity_type, doc_freq, tier
                FROM get_proposal_surfaces(%s, %s, %s)
            """, (tier, entity_type, limit))
            
            return [
                ProposalSurface(
                    entity_id=row[0],
                    surface_norm=row[1],
                    surface_display=row[2],
                    entity_type=row[3],
                    doc_freq=row[4],
                    tier=row[5]
                )
                for row in cur.fetchall()
            ]
    
    def get_corpus_summary(self) -> List[ProposalStats]:
        """
        Get summary statistics for the proposal corpus.
        
        Returns:
            List of ProposalStats per tier
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT tier, surface_count, entity_count, total_mentions, 
                       avg_doc_freq, avg_confidence
                FROM v_proposal_corpus_summary
                ORDER BY tier
            """)
            
            return [
                ProposalStats(
                    tier=row[0],
                    surface_count=row[1],
                    entity_count=row[2],
                    total_mentions=row[3],
                    avg_doc_freq=float(row[4]) if row[4] else 0,
                    avg_confidence=float(row[5]) if row[5] else 0
                )
                for row in cur.fetchall()
            ]
    
    def is_banned(self, surface: str, scope: str = 'global',
                  collection_id: Optional[int] = None,
                  document_id: Optional[int] = None) -> bool:
        """
        Check if a surface is explicitly banned.
        
        Args:
            surface: Raw surface text
            scope: 'global', 'collection', or 'document'
            collection_id: Required if scope='collection'
            document_id: Required if scope='document'
        
        Returns:
            True if surface is banned at the given scope
        """
        surface_norm = normalize_surface(surface)
        
        self._load_override_cache()
        
        if scope == 'global':
            return surface_norm in self._banned_cache
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM entity_alias_overrides
                WHERE surface_norm = %s AND banned = TRUE
                  AND scope = %s
                  AND (scope_collection_id = %s OR %s IS NULL)
                  AND (scope_document_id = %s OR %s IS NULL)
                LIMIT 1
            """, (surface_norm, scope, collection_id, collection_id, 
                  document_id, document_id))
            return cur.fetchone() is not None
    
    def get_forced_entity(self, surface: str, scope: str = 'global',
                          collection_id: Optional[int] = None,
                          document_id: Optional[int] = None) -> Optional[int]:
        """
        Check if a surface has a forced entity override.
        
        Args:
            surface: Raw surface text
            scope: 'global', 'collection', or 'document'
        
        Returns:
            Entity ID if forced, None otherwise
        """
        surface_norm = normalize_surface(surface)
        
        with self.conn.cursor() as cur:
            # Check most specific scope first
            cur.execute("""
                SELECT forced_entity_id FROM entity_alias_overrides
                WHERE surface_norm = %s 
                  AND forced_entity_id IS NOT NULL
                  AND (
                      (scope = 'document' AND scope_document_id = %s)
                      OR (scope = 'collection' AND scope_collection_id = %s)
                      OR scope = 'global'
                  )
                ORDER BY 
                    CASE scope 
                        WHEN 'document' THEN 1 
                        WHEN 'collection' THEN 2 
                        ELSE 3 
                    END
                LIMIT 1
            """, (surface_norm, document_id, collection_id))
            row = cur.fetchone()
            return row[0] if row else None
    
    def clear_cache(self):
        """Clear all caches."""
        self._tier_cache.clear()
        self._banned_cache.clear()
        self._override_cache_loaded = False
    
    def refresh_materialized_view(self):
        """Refresh the entity_surface_stats materialized view."""
        with self.conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW entity_surface_stats")
        self.conn.commit()
        self.clear_cache()


def compute_group_key(surface_norm: str, candidate_entity_ids: List[int]) -> str:
    """
    Compute a grouping key for review queue items.
    
    Items with the same group_key represent the same ambiguity pattern
    and can be batch-processed together.
    
    Args:
        surface_norm: Normalized surface
        candidate_entity_ids: List of candidate entity IDs
    
    Returns:
        Group key string
    """
    sorted_ids = sorted(set(candidate_entity_ids))
    ids_str = ','.join(str(i) for i in sorted_ids)
    return f"{surface_norm}:{hashlib.md5(ids_str.encode()).hexdigest()[:8]}"


def compute_candidate_set_hash(candidate_entity_ids: List[int]) -> str:
    """
    Compute hash of candidate entity set.
    
    Args:
        candidate_entity_ids: List of candidate entity IDs
    
    Returns:
        MD5 hash (first 16 chars)
    """
    sorted_ids = sorted(set(candidate_entity_ids))
    ids_str = ','.join(str(i) for i in sorted_ids)
    return hashlib.md5(ids_str.encode()).hexdigest()[:16]


# Convenience function for use without class
def is_surface_proposable(
    conn,
    surface: str,
    min_tier: int = 2,
    collection_id: Optional[int] = None,
    document_id: Optional[int] = None
) -> bool:
    """
    Convenience function to check if a surface is proposable.
    
    For repeated checks, use ProposalGate class for caching.
    """
    gate = ProposalGate(conn, cache_enabled=False)
    return gate.is_proposable(surface, min_tier, collection_id, document_id)


if __name__ == "__main__":
    # Test
    import psycopg2
    import os
    
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "neh"),
        user=os.getenv("DB_USER", "neh"),
        password=os.getenv("DB_PASS", "neh")
    )
    
    gate = ProposalGate(conn)
    
    print("Proposal Corpus Summary:")
    print("-" * 60)
    for stats in gate.get_corpus_summary():
        print(f"Tier {stats.tier}: {stats.surface_count} surfaces, "
              f"{stats.entity_count} entities, "
              f"avg doc_freq={stats.avg_doc_freq:.1f}")
    
    print("\nSample Tier 1 surfaces:")
    for s in gate.get_proposal_surfaces(tier=1, limit=10):
        print(f"  {s.surface_display} ({s.entity_type}): doc_freq={s.doc_freq}")
    
    conn.close()
