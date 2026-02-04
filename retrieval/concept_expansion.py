"""
Concept Expansion Module

Expands conceptual queries into concrete searchable primitives by leveraging:
1. Entity resolution - Known entities in the database
2. Entity relationships - member_of, alias_of, covername_of, etc.
3. Term expansion - Fallback search terms for robustness
4. Domain knowledge - Known concepts that map to multiple entities/terms

This module bridges the gap between user intent ("silvermaster network") and
searchable primitives (entity IDs, terms, phrases).
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import re


@dataclass
class ExpandedConcept:
    """Result of concept expansion."""
    original_text: str
    
    # Primary entity (if resolved)
    entity_id: Optional[int] = None
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    
    # Related entities (from relationships)
    related_entity_ids: List[int] = field(default_factory=list)
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Search terms (for robust fallback)
    search_terms: List[str] = field(default_factory=list)
    
    # Expansion metadata
    expansion_method: str = "none"  # "entity", "relationship", "term", "domain_knowledge"
    confidence: float = 0.0
    
    def to_primitives(self) -> List[Dict[str, Any]]:
        """Convert expansion to retrieval primitives."""
        primitives = []
        
        # If we have a resolved entity, search for it
        if self.entity_id:
            primitives.append({
                "type": "ENTITY",
                "entity_id": self.entity_id,
            })
            
            # Also add RELATED_ENTITIES to find co-occurring people
            primitives.append({
                "type": "RELATED_ENTITIES",
                "entity_id": self.entity_id,
                "window": "document",
                "top_n": 30,
            })
        
        # Add related entities as OR group if we have them
        if self.related_entity_ids:
            # Don't create huge OR groups - just include top related
            for eid in self.related_entity_ids[:10]:
                primitives.append({
                    "type": "ENTITY",
                    "entity_id": eid,
                })
        
        # Always add term fallback for robustness
        for term in self.search_terms[:3]:  # Limit to 3 terms
            primitives.append({
                "type": "TERM",
                "value": term,
            })
        
        return primitives


def expand_concept(
    conn,
    text: str,
    *,
    include_relationships: bool = True,
    max_related: int = 20,
) -> ExpandedConcept:
    """
    Expand a conceptual term/phrase into searchable primitives.
    
    Args:
        conn: Database connection
        text: The concept text (e.g., "silvermaster network", "Rosenberg case")
        include_relationships: Whether to fetch related entities
        max_related: Maximum related entities to include
    
    Returns:
        ExpandedConcept with entity IDs, related entities, and search terms
    """
    result = ExpandedConcept(original_text=text)
    
    # Extract the likely entity name from conceptual phrases
    # "silvermaster network" -> "silvermaster"
    # "Rosenberg case" -> "Rosenberg"
    entity_name = extract_entity_name(text)
    
    # Try to resolve as entity
    entity_data = resolve_entity(conn, entity_name)
    
    if entity_data:
        result.entity_id = entity_data["id"]
        result.entity_name = entity_data["canonical_name"]
        result.entity_type = entity_data["entity_type"]
        result.expansion_method = "entity"
        result.confidence = entity_data.get("confidence", 0.8)
        
        # Get related entities
        if include_relationships:
            related = get_related_entities(conn, entity_data["id"], max_related)
            result.related_entity_ids = [r["entity_id"] for r in related]
            result.related_entities = related
            if related:
                result.expansion_method = "relationship"
    
    # Build search terms for fallback
    result.search_terms = build_search_terms(text, entity_name)
    
    if not result.entity_id and result.search_terms:
        result.expansion_method = "term"
        result.confidence = 0.5
    
    return result


def extract_entity_name(text: str) -> str:
    """
    Extract the likely entity name from a conceptual phrase.
    
    Examples:
        "silvermaster network" -> "silvermaster"
        "Rosenberg case" -> "Rosenberg"
        "CPUSA members" -> "CPUSA"
        "Treasury Department group" -> "Treasury Department"
    """
    text = text.strip()
    
    # Common suffixes to strip
    suffixes = [
        r"\s+network$",
        r"\s+case$",
        r"\s+affair$",
        r"\s+ring$",
        r"\s+group$",
        r"\s+members?$",
        r"\s+operation$",
        r"\s+project$",
        r"\s+investigation$",
        r"\s+trial$",
        r"\s+conspiracy$",
        r"\s+apparatus$",
        r"\s+cell$",
        r"\s+spy\s*ring$",
    ]
    
    result = text
    for suffix in suffixes:
        result = re.sub(suffix, "", result, flags=re.IGNORECASE)
    
    return result.strip()


def resolve_entity(conn, name: str) -> Optional[Dict[str, Any]]:
    """
    Resolve an entity name to its database record.
    Uses exact match first, then fuzzy matching.
    """
    if not name:
        return None
    
    # Normalize for matching
    name_norm = name.lower().strip()
    
    with conn.cursor() as cur:
        # Try exact match on alias_norm
        cur.execute(
            """
            SELECT e.id, e.canonical_name, e.entity_type, ea.alias, 1.0 as confidence
            FROM entities e
            JOIN entity_aliases ea ON ea.entity_id = e.id
            WHERE ea.alias_norm = %s AND ea.is_matchable = true
            LIMIT 1
            """,
            (name_norm,)
        )
        row = cur.fetchone()
        
        if row:
            return {
                "id": row[0],
                "canonical_name": row[1],
                "entity_type": row[2],
                "matched_alias": row[3],
                "confidence": row[4],
            }
        
        # Try fuzzy match using trigram similarity
        cur.execute(
            """
            SELECT e.id, e.canonical_name, e.entity_type, ea.alias,
                   similarity(ea.alias_norm, %s) as confidence
            FROM entities e
            JOIN entity_aliases ea ON ea.entity_id = e.id
            WHERE ea.is_matchable = true
              AND similarity(ea.alias_norm, %s) > 0.4
            ORDER BY confidence DESC
            LIMIT 1
            """,
            (name_norm, name_norm)
        )
        row = cur.fetchone()
        
        if row:
            return {
                "id": row[0],
                "canonical_name": row[1],
                "entity_type": row[2],
                "matched_alias": row[3],
                "confidence": row[4],
            }
    
    return None


def get_related_entities(
    conn,
    entity_id: int,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Get entities related to the given entity via:
    1. Direct relationships (entity_relationships table)
    2. Co-mentions (entities frequently mentioned together)
    """
    related = []
    
    with conn.cursor() as cur:
        # Get direct relationships
        cur.execute(
            """
            SELECT 
                CASE 
                    WHEN er.source_entity_id = %s THEN er.target_entity_id
                    ELSE er.source_entity_id
                END as related_id,
                e.canonical_name,
                e.entity_type,
                er.relationship_type,
                er.confidence
            FROM entity_relationships er
            JOIN entities e ON e.id = CASE 
                WHEN er.source_entity_id = %s THEN er.target_entity_id
                ELSE er.source_entity_id
            END
            WHERE er.source_entity_id = %s OR er.target_entity_id = %s
            ORDER BY er.confidence DESC
            LIMIT %s
            """,
            (entity_id, entity_id, entity_id, entity_id, max_results // 2)
        )
        
        for row in cur.fetchall():
            related.append({
                "entity_id": row[0],
                "canonical_name": row[1],
                "entity_type": row[2],
                "relationship_type": row[3],
                "confidence": row[4],
                "source": "relationship",
            })
        
        # Get co-mentioned entities (entities that appear in same documents)
        seen_ids = {entity_id} | {r["entity_id"] for r in related}
        remaining = max_results - len(related)
        
        if remaining > 0:
            cur.execute(
                """
                WITH source_docs AS (
                    SELECT DISTINCT document_id
                    FROM entity_mentions
                    WHERE entity_id = %s
                    LIMIT 100
                ),
                co_mentions AS (
                    SELECT em.entity_id, COUNT(DISTINCT em.document_id) as doc_count
                    FROM entity_mentions em
                    WHERE em.document_id IN (SELECT document_id FROM source_docs)
                      AND em.entity_id != %s
                    GROUP BY em.entity_id
                    HAVING COUNT(DISTINCT em.document_id) >= 2
                    ORDER BY doc_count DESC
                    LIMIT %s
                )
                SELECT cm.entity_id, e.canonical_name, e.entity_type, cm.doc_count
                FROM co_mentions cm
                JOIN entities e ON e.id = cm.entity_id
                """,
                (entity_id, entity_id, remaining + len(seen_ids))
            )
            
            for row in cur.fetchall():
                if row[0] not in seen_ids:
                    related.append({
                        "entity_id": row[0],
                        "canonical_name": row[1],
                        "entity_type": row[2],
                        "relationship_type": "co_mentioned",
                        "confidence": min(1.0, row[3] / 10.0),  # Normalize doc_count
                        "source": "co_mention",
                    })
                    seen_ids.add(row[0])
                    if len(related) >= max_results:
                        break
    
    return related


def build_search_terms(original_text: str, entity_name: str) -> List[str]:
    """
    Build robust search terms for fallback search.
    """
    terms = []
    
    # Add the entity name (most important)
    if entity_name:
        terms.append(entity_name.lower())
    
    # Add original text if different
    original_lower = original_text.lower().strip()
    if original_lower != entity_name.lower():
        # Don't add very long phrases as single terms
        if len(original_lower.split()) <= 4:
            terms.append(original_lower)
    
    # Add individual significant words
    words = original_text.split()
    for word in words:
        word_clean = word.strip().lower()
        # Keep capitalized words (likely proper nouns)
        if word[0].isupper() and len(word_clean) > 2:
            if word_clean not in terms:
                terms.append(word_clean)
    
    return terms


def expand_query_concepts(
    conn,
    utterance: str,
    resolved_entities: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Expand all concepts in a query utterance.
    
    Args:
        conn: Database connection
        utterance: User's natural language query
        resolved_entities: Already-resolved entities from the planner
    
    Returns:
        (additional_primitives, expansion_notes)
    """
    additional_primitives = []
    notes = []
    
    # Check if we have any resolved entities - if so, add RELATED_ENTITIES for each
    for ent in resolved_entities:
        if ent.get("entity_id"):
            # Add RELATED_ENTITIES to find co-occurring entities
            additional_primitives.append({
                "type": "RELATED_ENTITIES",
                "entity_id": ent["entity_id"],
                "window": "document",
                "top_n": 20,
            })
            notes.append(f"Expanded {ent.get('surface', 'entity')} to include related entities")
    
    # Look for conceptual phrases that might need expansion
    concept_patterns = [
        r"(\w+)\s+network",
        r"(\w+)\s+case",
        r"(\w+)\s+ring",
        r"(\w+)\s+group",
        r"(\w+)\s+affair",
        r"(\w+)\s+conspiracy",
        r"(\w+)\s+operation",
    ]
    
    already_expanded = {e.get("entity_id") for e in resolved_entities if e.get("entity_id")}
    
    for pattern in concept_patterns:
        match = re.search(pattern, utterance, re.IGNORECASE)
        if match:
            concept_name = match.group(1)
            # Try to expand this concept
            expansion = expand_concept(conn, concept_name)
            
            if expansion.entity_id and expansion.entity_id not in already_expanded:
                # Add the resolved entity
                additional_primitives.append({
                    "type": "ENTITY",
                    "entity_id": expansion.entity_id,
                })
                
                # Add related entities search
                additional_primitives.append({
                    "type": "RELATED_ENTITIES",
                    "entity_id": expansion.entity_id,
                    "window": "document",
                    "top_n": 20,
                })
                
                already_expanded.add(expansion.entity_id)
                notes.append(f"Expanded '{concept_name}' → {expansion.entity_name} (ID: {expansion.entity_id})")
            
            elif expansion.search_terms:
                # Fall back to term search
                for term in expansion.search_terms[:2]:
                    additional_primitives.append({
                        "type": "TERM",
                        "value": term,
                    })
                notes.append(f"Added search term '{expansion.search_terms[0]}' for concept '{concept_name}'")
    
    return additional_primitives, "; ".join(notes) if notes else ""


# Domain knowledge for well-known concepts
# This can be expanded over time or loaded from a database
KNOWN_CONCEPTS = {
    "silvermaster": {
        "description": "Soviet espionage network in US Treasury/government",
        "search_terms": ["silvermaster", "treasury group"],
        "related_concepts": ["white", "currie", "treasury"],
    },
    "rosenberg": {
        "description": "Julius and Ethel Rosenberg espionage case",
        "search_terms": ["rosenberg", "ethel", "julius"],
        "related_concepts": ["greenglass", "sobell", "atomic"],
    },
    "venona": {
        "description": "US signals intelligence project decrypting Soviet communications",
        "search_terms": ["venona", "decryption", "intercept"],
        "related_concepts": ["nsa", "soviet", "cables"],
    },
    "cpusa": {
        "description": "Communist Party of the United States of America",
        "search_terms": ["cpusa", "communist party", "party membership"],
        "related_concepts": ["browder", "foster", "communist"],
    },
}


def get_domain_expansion(concept: str) -> Optional[Dict[str, Any]]:
    """
    Look up known domain concepts for expansion hints.
    """
    concept_lower = concept.lower().strip()
    return KNOWN_CONCEPTS.get(concept_lower)
