"""
Entity Resolution (lookup, creation, alias management)

Deterministic entity resolution for plan compilation. Converts ambiguous name
strings into resolved entity IDs before plans are approved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json


@dataclass
class EntityCandidate:
    """Candidate entity from lookup"""
    entity_id: int
    canonical_name: str
    entity_type: str
    confidence: float
    match_method: str  # 'exact', 'fuzzy', 'alias'
    matched_alias: Optional[str] = None
    similarity_score: Optional[float] = None


@dataclass
class EntityLookupResult:
    """Result of entity lookup"""
    exact_matches: List[EntityCandidate]
    near_matches: List[EntityCandidate]
    ambiguous: bool  # True if multiple high-confidence matches


def normalize_alias(alias: str) -> str:
    """
    Normalize alias for matching: lowercase, strip punctuation, collapse whitespace.
    """
    # Lowercase
    normalized = alias.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


def entity_lookup(
    conn,
    name: str,
    *,
    entity_type: Optional[str] = None,  # 'person', 'org', 'place', or None for any
    fuzzy_threshold: float = 0.87,
    max_candidates: int = 10,
) -> EntityLookupResult:
    """
    Lookup entity by name. Returns exact and near matches.
    
    Stage A: Exact alias match (alias_norm)
    Stage B: Fuzzy alias match (trigram similarity, if enabled)
    
    Returns EntityLookupResult with exact_matches and near_matches.
    """
    normalized = normalize_alias(name)
    exact_matches: List[EntityCandidate] = []
    near_matches: List[EntityCandidate] = []
    
    with conn.cursor() as cur:
        # Stage A: Exact alias match
        if entity_type:
            cur.execute("""
                SELECT 
                    e.id,
                    e.canonical_name,
                    e.entity_type,
                    ea.alias,
                    1.0 AS confidence
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE ea.alias_norm = %s
                  AND e.entity_type = %s
                ORDER BY e.id
            """, (normalized, entity_type))
        else:
            cur.execute("""
                SELECT 
                    e.id,
                    e.canonical_name,
                    e.entity_type,
                    ea.alias,
                    1.0 AS confidence
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE ea.alias_norm = %s
                ORDER BY e.id
            """, (normalized,))
        
        for row in cur.fetchall():
            entity_id, canonical_name, etype, matched_alias, confidence = row
            exact_matches.append(EntityCandidate(
                entity_id=entity_id,
                canonical_name=canonical_name,
                entity_type=etype,
                confidence=confidence,
                match_method="exact",
                matched_alias=matched_alias,
            ))
        
        # Stage B: Fuzzy alias match (if no exact matches or requested)
        if not exact_matches or fuzzy_threshold < 1.0:
            # Use trigram similarity on alias_norm
            if entity_type:
                cur.execute("""
                    SELECT 
                        e.id,
                        e.canonical_name,
                        e.entity_type,
                        ea.alias,
                        similarity(ea.alias_norm, %s) AS sim_score
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE e.entity_type = %s
                      AND ea.alias_norm % %s  -- Trigram filter (uses index)
                      AND similarity(ea.alias_norm, %s) >= %s
                    ORDER BY sim_score DESC, e.id
                    LIMIT %s
                """, (normalized, entity_type, normalized, normalized, fuzzy_threshold, max_candidates))
            else:
                cur.execute("""
                    SELECT 
                        e.id,
                        e.canonical_name,
                        e.entity_type,
                        ea.alias,
                        similarity(ea.alias_norm, %s) AS sim_score
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE ea.alias_norm % %s  -- Trigram filter (uses index)
                      AND similarity(ea.alias_norm, %s) >= %s
                    ORDER BY sim_score DESC, e.id
                    LIMIT %s
                """, (normalized, normalized, normalized, fuzzy_threshold, max_candidates))
            
            for row in cur.fetchall():
                entity_id, canonical_name, etype, matched_alias, sim_score = row
                # Skip if already in exact matches
                if any(c.entity_id == entity_id for c in exact_matches):
                    continue
                
                near_matches.append(EntityCandidate(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    entity_type=etype,
                    confidence=float(sim_score),
                    match_method="fuzzy",
                    matched_alias=matched_alias,
                    similarity_score=float(sim_score),
                ))
    
    # Determine if ambiguous (multiple high-confidence matches)
    high_confidence = [c for c in exact_matches + near_matches if c.confidence >= 0.9]
    ambiguous = len(high_confidence) > 1
    
    return EntityLookupResult(
        exact_matches=exact_matches,
        near_matches=near_matches,
        ambiguous=ambiguous,
    )


def entity_create(
    conn,
    canonical_name: str,
    entity_type: str,
    *,
    description: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    external_ids: Optional[Dict[str, str]] = None,
) -> int:
    """
    Create a new entity with canonical name and aliases.
    Returns the new entity_id.
    """
    with conn.cursor() as cur:
        # Insert entity
        cur.execute("""
            INSERT INTO entities (entity_type, canonical_name, description, external_ids)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (entity_type, canonical_name, description, Json(external_ids or {})))
        
        entity_id = cur.fetchone()[0]
        
        # Add canonical name as primary alias
        alias_norm = normalize_alias(canonical_name)
        cur.execute("""
            INSERT INTO entity_aliases (entity_id, alias, alias_norm, kind)
            VALUES (%s, %s, %s, 'primary')
            ON CONFLICT (entity_id, alias_norm) DO NOTHING
        """, (entity_id, canonical_name, alias_norm))
        
        # Add additional aliases
        if aliases:
            for alias in aliases:
                alias_norm = normalize_alias(alias)
                cur.execute("""
                    INSERT INTO entity_aliases (entity_id, alias, alias_norm, kind)
                    VALUES (%s, %s, %s, 'alt')
                    ON CONFLICT (entity_id, alias_norm) DO NOTHING
                """, (entity_id, alias, alias_norm))
        
        conn.commit()
        return entity_id


def entity_add_alias(
    conn,
    entity_id: int,
    alias: str,
    *,
    kind: str = "alt",
) -> None:
    """
    Add an alias to an existing entity.
    """
    alias_norm = normalize_alias(alias)
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO entity_aliases (entity_id, alias, alias_norm, kind)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (entity_id, alias_norm) DO NOTHING
        """, (entity_id, alias, alias_norm, kind))
        conn.commit()


def resolve_entity_name(
    conn,
    name: str,
    *,
    entity_type: Optional[str] = None,
    require_unique: bool = True,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Resolve entity name to entity_id. Returns (entity_id, error_message).
    
    If require_unique=True and multiple matches found, returns (None, error_message).
    If require_unique=False, returns first match.
    """
    result = entity_lookup(conn, name, entity_type=entity_type)
    
    # Prefer exact matches
    candidates = result.exact_matches
    if not candidates:
        candidates = result.near_matches[:1]  # Top fuzzy match
    
    if not candidates:
        return None, f"No entity found matching '{name}'"
    
    if require_unique and len(candidates) > 1:
        candidate_list = ", ".join([f"{c.canonical_name} (id={c.entity_id})" for c in candidates])
        return None, f"Ambiguous: '{name}' matches multiple entities: {candidate_list}"
    
    return candidates[0].entity_id, None
