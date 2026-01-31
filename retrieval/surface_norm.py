"""
Surface normalization - single source of truth.

All entity/alias matching keys off surface_norm. This module provides the
canonical normalization function used everywhere:
- entity_aliases.alias_norm
- entity_mentions.surface_norm
- review queue surfaces
- OCR/NER surfaces
- proposal corpus

IMPORTANT: Do not change this function without migration planning.
Existing surface_norm values in the database depend on this exact algorithm.
"""

import re
from typing import Optional

# Version tracking for normalization algorithm
# Increment when algorithm changes (requires recomputation)
SURFACE_NORM_VERSION = "v1"


def normalize_surface(text: str) -> str:
    """
    Normalize surface text for matching.
    
    Algorithm (v1):
    1. Lowercase
    2. Remove punctuation (keep alphanumeric and spaces)
    3. Collapse whitespace
    4. Strip leading/trailing whitespace
    
    This is the SINGLE SOURCE OF TRUTH for surface normalization.
    All code paths must use this function.
    
    Args:
        text: Raw surface text (e.g., "Mr. COHN", "FBI", "Silvermaster")
    
    Returns:
        Normalized surface (e.g., "mr cohn", "fbi", "silvermaster")
    
    Examples:
        >>> normalize_surface("Julius ROSENBERG")
        'julius rosenberg'
        >>> normalize_surface("Mr. Cohn")
        'mr cohn'
        >>> normalize_surface("F.B.I.")
        'fbi'
        >>> normalize_surface("  multiple   spaces  ")
        'multiple spaces'
    """
    if not text:
        return ""
    
    # Step 1: Lowercase
    normalized = text.lower()
    
    # Step 2: Remove punctuation (keep alphanumeric and spaces)
    # This handles: periods, commas, apostrophes, hyphens, etc.
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Step 3: Collapse whitespace (including newlines, tabs)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Step 4: Strip
    return normalized.strip()


# Alias for backwards compatibility with entity_resolver imports
normalize_alias = normalize_surface


def normalize_surface_for_hash(text: str) -> str:
    """
    Normalize surface for use as hash/grouping key.
    
    Same as normalize_surface but also removes ALL spaces.
    Used for candidate_set grouping where "John Smith" and "JohnSmith" 
    should be considered equivalent variants.
    """
    return normalize_surface(text).replace(' ', '')


def compute_surface_hash(surfaces: list[str]) -> str:
    """
    Compute a deterministic hash for a set of surface variants.
    
    Used for grouping review queue items by candidate set.
    """
    import hashlib
    normalized = sorted(set(normalize_surface(s) for s in surfaces if s))
    return hashlib.md5('|'.join(normalized).encode()).hexdigest()[:16]


def compute_candidate_set_hash(entity_ids: list[int]) -> str:
    """
    Compute a deterministic hash for a set of candidate entity IDs.
    
    Used for grouping review queue items by candidate set.
    
    Args:
        entity_ids: List of entity IDs (will be sorted for determinism)
    
    Returns:
        16-character hex hash
    """
    import hashlib
    sorted_ids = sorted(set(entity_ids))
    id_str = '|'.join(str(i) for i in sorted_ids)
    return hashlib.md5(id_str.encode()).hexdigest()[:16]


def compute_group_key(surface_norm: str, entity_ids: list[int]) -> str:
    """
    Compute a group key for review queue batch processing.
    
    Combines surface_norm with candidate_set_hash for grouping
    collisions that share the same surface and candidate entities.
    
    Args:
        surface_norm: Normalized surface text
        entity_ids: List of candidate entity IDs
    
    Returns:
        Group key string: "{surface_norm}::{candidate_set_hash}"
    """
    candidate_hash = compute_candidate_set_hash(entity_ids)
    return f"{surface_norm}::{candidate_hash}"


def is_valid_surface(text: str, min_length: int = 2) -> bool:
    """
    Check if a surface is valid for matching.
    
    Filters:
    - Must have at least min_length alphanumeric characters
    - Must not be purely numeric
    - Must not be common stop words
    """
    if not text:
        return False
    
    norm = normalize_surface(text)
    
    # Length check
    if len(norm) < min_length:
        return False
    
    # Must have at least some letters
    if not re.search(r'[a-z]', norm):
        return False
    
    # Filter common stop words that shouldn't be entity surfaces
    STOP_SURFACES = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
        'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'i', 'me', 'my',
        'said', 'says', 'told', 'asked', 'yes', 'no', 'not', 'also', 'just',
    }
    
    if norm in STOP_SURFACES:
        return False
    
    return True


# SQL function definition for Postgres mirror
POSTGRES_NORMALIZE_SURFACE_SQL = """
-- Mirror of Python normalize_surface() function
-- IMPORTANT: Keep in sync with retrieval/surface_norm.py
CREATE OR REPLACE FUNCTION normalize_surface(text TEXT)
RETURNS TEXT
LANGUAGE SQL
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT TRIM(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                LOWER(COALESCE(text, '')),
                '[^\\w\\s]', '', 'g'  -- Remove punctuation
            ),
            '\\s+', ' ', 'g'  -- Collapse whitespace
        )
    )
$$;

COMMENT ON FUNCTION normalize_surface(TEXT) IS 
'Canonical surface normalization for entity matching. Mirrors Python normalize_surface().';
"""


def test_normalize_surface():
    """Test cases for normalize_surface."""
    # Basic cases
    assert normalize_surface("Julius ROSENBERG") == "julius rosenberg"
    assert normalize_surface("Mr. Cohn") == "mr cohn"
    assert normalize_surface("F.B.I.") == "fbi"
    assert normalize_surface("  multiple   spaces  ") == "multiple spaces"
    
    # Punctuation
    assert normalize_surface("O'Brien") == "obrien"
    assert normalize_surface("Smith-Jones") == "smithjones"
    assert normalize_surface("[REDACTED]") == "redacted"
    
    # Edge cases
    assert normalize_surface("") == ""
    assert normalize_surface("   ") == ""
    assert normalize_surface("123") == "123"
    
    # Unicode
    assert normalize_surface("Müller") == "müller"
    
    print("✅ All normalize_surface tests passed!")


if __name__ == "__main__":
    test_normalize_surface()
