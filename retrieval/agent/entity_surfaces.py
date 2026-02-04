"""
Entity Surface Index - Fast lookup of entity surface forms for attestation.

The EntitySurfaceIndex provides:
- Cached lookup of canonical names and aliases for entities
- Normalized surface forms for reliable matching
- Attestation check: does a quote contain a surface form of an entity?

Usage:
    surface_index = EntitySurfaceIndex(conn)
    
    # Get all surface forms for an entity
    surfaces = surface_index.get_surfaces(entity_id=123)
    # Returns: {'julius rosenberg', 'rosenberg', 'liberal'}
    
    # Check if a quote attests to an entity
    attested = surface_index.attests(entity_id=123, quote="LIBERAL reports that...")
    # Returns: True (because 'liberal' is in the quote)
"""

import re
import unicodedata
from typing import Dict, Set, List, Optional, Tuple


def normalize_surface(text: str) -> str:
    """
    Normalize a surface form for matching.
    
    - Lowercase
    - Collapse whitespace
    - Remove punctuation (except apostrophes in names)
    - Unicode normalize (NFKD)
    """
    if not text:
        return ""
    
    # Unicode normalize
    text = unicodedata.normalize('NFKD', text)
    
    # Lowercase
    text = text.lower()
    
    # Keep apostrophes for names like O'Brien, but remove other punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    return text.strip()


class EntitySurfaceIndex:
    """
    Cached index of entity surface forms for fast attestation.
    
    Loads canonical names and aliases from the database once per entity,
    then caches for reuse across multiple verification checks.
    """
    
    def __init__(self, conn):
        self.conn = conn
        self._cache: Dict[int, Set[str]] = {}
        self._canonical_cache: Dict[int, str] = {}
        self._equivalences: Dict[int, Set[int]] = {}  # entity_id -> equivalent entity_ids
        self._equivalences_loaded = False
    
    def get_surfaces(self, entity_id: int) -> Set[str]:
        """
        Get all normalized surface forms for an entity.
        
        Returns:
            Set of normalized surface strings (lowercase, collapsed whitespace)
        """
        if entity_id in self._cache:
            return self._cache[entity_id]
        
        surfaces = self._load_surfaces(entity_id)
        self._cache[entity_id] = surfaces
        return surfaces
    
    def get_canonical_name(self, entity_id: int) -> str:
        """Get the canonical name for an entity."""
        if entity_id in self._canonical_cache:
            return self._canonical_cache[entity_id]
        
        # Will populate cache as side effect
        self._load_surfaces(entity_id)
        return self._canonical_cache.get(entity_id, f"Entity {entity_id}")
    
    def _load_surfaces(self, entity_id: int) -> Set[str]:
        """Load surfaces from database for a single entity."""
        surfaces: Set[str] = set()
        
        if not self.conn:
            return surfaces
        
        try:
            with self.conn.cursor() as cur:
                # Load canonical name
                cur.execute(
                    "SELECT canonical_name FROM entities WHERE id = %s",
                    (entity_id,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    canonical = row[0]
                    self._canonical_cache[entity_id] = canonical
                    normalized = normalize_surface(canonical)
                    if normalized:
                        surfaces.add(normalized)
                
                # Load aliases
                cur.execute(
                    "SELECT alias FROM entity_aliases WHERE entity_id = %s",
                    (entity_id,)
                )
                for row in cur.fetchall():
                    if row[0]:
                        normalized = normalize_surface(row[0])
                        if normalized:
                            surfaces.add(normalized)
                
                # Load equivalences and union their surfaces
                equiv_ids = self._get_equivalences(entity_id)
                for equiv_id in equiv_ids:
                    if equiv_id != entity_id and equiv_id not in self._cache:
                        # Recursively load (but avoid infinite loops)
                        equiv_surfaces = self._load_surfaces_no_equiv(equiv_id)
                        surfaces.update(equiv_surfaces)
                
        except Exception as e:
            # Log but don't fail - return what we have
            pass
        
        return surfaces
    
    def _load_surfaces_no_equiv(self, entity_id: int) -> Set[str]:
        """Load surfaces without following equivalences (to avoid cycles)."""
        surfaces: Set[str] = set()
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT canonical_name FROM entities WHERE id = %s",
                    (entity_id,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    surfaces.add(normalize_surface(row[0]))
                
                cur.execute(
                    "SELECT alias FROM entity_aliases WHERE entity_id = %s",
                    (entity_id,)
                )
                for row in cur.fetchall():
                    if row[0]:
                        surfaces.add(normalize_surface(row[0]))
        except Exception:
            pass
        
        return surfaces
    
    def _get_equivalences(self, entity_id: int) -> Set[int]:
        """Get equivalent entity IDs (e.g., Soviet Union = USSR)."""
        if not self._equivalences_loaded:
            self._load_all_equivalences()
        
        return self._equivalences.get(entity_id, set())
    
    def _load_all_equivalences(self):
        """Load entity equivalences table if it exists."""
        self._equivalences_loaded = True
        
        if not self.conn:
            return
        
        try:
            with self.conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'entity_equivalences'
                    )
                """)
                if not cur.fetchone()[0]:
                    return
                
                # Load all equivalences
                cur.execute("""
                    SELECT entity_id, equivalent_entity_id 
                    FROM entity_equivalences
                    WHERE relation = 'same_as'
                """)
                for row in cur.fetchall():
                    eid, equiv_id = row
                    if eid not in self._equivalences:
                        self._equivalences[eid] = set()
                    self._equivalences[eid].add(equiv_id)
                    # Make it symmetric
                    if equiv_id not in self._equivalences:
                        self._equivalences[equiv_id] = set()
                    self._equivalences[equiv_id].add(eid)
                    
        except Exception:
            # Table might not exist yet, that's OK
            pass
    
    def attests(self, entity_id: int, quote: str) -> bool:
        """
        Check if a quote attests to an entity.
        
        Args:
            entity_id: The entity to check for
            quote: The quote text to search
        
        Returns:
            True if any surface form of the entity appears in the quote
        """
        surfaces = self.get_surfaces(entity_id)
        if not surfaces:
            return False
        
        quote_normalized = normalize_surface(quote)
        if not quote_normalized:
            return False
        
        for surface in surfaces:
            if self._surface_matches(surface, quote_normalized):
                return True
        
        return False
    
    def attests_any_quote(self, entity_id: int, quotes: List[str]) -> bool:
        """
        Check if any of the provided quotes attests to an entity.
        
        This is the primary check: entity must appear in at least ONE quote.
        """
        surfaces = self.get_surfaces(entity_id)
        if not surfaces:
            return False
        
        # Concatenate all quotes for efficient search
        combined = " ".join(normalize_surface(q) for q in quotes if q)
        
        for surface in surfaces:
            if self._surface_matches(surface, combined):
                return True
        
        return False
    
    def _surface_matches(self, surface: str, text: str) -> bool:
        """
        Check if a surface form matches in text.
        
        For short aliases (<= 3 chars), require word boundary match.
        For longer surfaces, substring match is sufficient.
        """
        if not surface or not text:
            return False
        
        if len(surface) <= 3:
            # Short alias - require word boundary match to avoid false positives
            # e.g., "kgb" shouldn't match "background"
            pattern = r'\b' + re.escape(surface) + r'\b'
            return bool(re.search(pattern, text))
        else:
            # Longer surface - substring match is fine
            return surface in text
    
    def get_matching_surface(self, entity_id: int, quote: str) -> Optional[str]:
        """
        Get the surface form that matched in the quote, if any.
        
        Useful for debugging which alias was found.
        """
        surfaces = self.get_surfaces(entity_id)
        quote_normalized = normalize_surface(quote)
        
        for surface in surfaces:
            if self._surface_matches(surface, quote_normalized):
                return surface
        
        return None
    
    def get_top_surfaces(self, entity_id: int, limit: int = 3) -> List[str]:
        """
        Get top N surface forms for an entity (for error messages).
        
        Prioritizes canonical name, then shortest aliases.
        """
        surfaces = self.get_surfaces(entity_id)
        if not surfaces:
            return []
        
        # Get canonical first
        canonical = self._canonical_cache.get(entity_id)
        canonical_norm = normalize_surface(canonical) if canonical else None
        
        result = []
        if canonical_norm and canonical_norm in surfaces:
            result.append(canonical_norm)
        
        # Add remaining surfaces sorted by length
        remaining = sorted(
            [s for s in surfaces if s != canonical_norm],
            key=len
        )
        result.extend(remaining)
        
        return result[:limit]
    
    def preload_entities(self, entity_ids: List[int]):
        """
        Preload surfaces for multiple entities in one batch.
        
        More efficient than loading one at a time.
        """
        if not entity_ids or not self.conn:
            return
        
        # Filter out already cached
        to_load = [eid for eid in entity_ids if eid not in self._cache]
        if not to_load:
            return
        
        try:
            with self.conn.cursor() as cur:
                # Load canonical names
                cur.execute(
                    "SELECT id, canonical_name FROM entities WHERE id = ANY(%s)",
                    (to_load,)
                )
                for row in cur.fetchall():
                    eid, canonical = row
                    if canonical:
                        self._canonical_cache[eid] = canonical
                        if eid not in self._cache:
                            self._cache[eid] = set()
                        self._cache[eid].add(normalize_surface(canonical))
                
                # Load aliases
                cur.execute(
                    "SELECT entity_id, alias FROM entity_aliases WHERE entity_id = ANY(%s)",
                    (to_load,)
                )
                for row in cur.fetchall():
                    eid, alias = row
                    if alias:
                        if eid not in self._cache:
                            self._cache[eid] = set()
                        self._cache[eid].add(normalize_surface(alias))
                
                # Initialize empty sets for entities with no data
                for eid in to_load:
                    if eid not in self._cache:
                        self._cache[eid] = set()
                        
        except Exception:
            pass
