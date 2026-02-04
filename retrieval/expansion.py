"""
Entity-Driven Expansion for Agentic V2.

Handles expansion loops for relationship/constraint queries:
1. Extract co-occurring entities from FocusBundle
2. Get their aliases/codenames
3. Run expanded retrieval
4. Rebuild FocusBundle
5. Stop on stability (Jaccard >= 0.85)

Also includes term extraction for "other names" queries (proximity fuse → VT fuse).
"""

import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.focus_bundle import FocusBundle
    from retrieval.query_intent import QueryContract


# Stopwords for term expansion
EXPANSION_STOP_LIST = {
    # Common words
    'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'BEEN', 'WERE',
    'WILL', 'CAN', 'NOT', 'BUT', 'ALL', 'ANY', 'WHO', 'WHAT', 'WHEN', 'WHERE',
    
    # Organizations (too common)
    'USA', 'USSR', 'FBI', 'CIA', 'KGB', 'NKVD', 'GRU', 'OSS',
    
    # Time-related
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
    
    # Document markers
    'TOP', 'SECRET', 'MEMO', 'REPORT', 'FILE', 'PAGE',
}


def extract_expansion_entities(
    focus_bundle: "FocusBundle",
    target_entity_ids: List[int],
    conn,
    max_entities: int = 10,
) -> List[int]:
    """
    Extract candidate entities that co-occur with targets in FocusBundle.
    
    For relationship queries: finds persons mentioned near target mentions.
    Returns entity_ids for second-pass retrieval.
    
    Args:
        focus_bundle: The FocusBundle to extract from
        target_entity_ids: Target entity IDs to find co-occurrences with
        conn: Database connection
        max_entities: Maximum entities to return
    
    Returns:
        List of entity_ids sorted by co-occurrence count
    """
    # Get target span IDs
    target_span_ids = set()
    for target_id in target_entity_ids:
        target_spans = focus_bundle.get_spans_for_entity(target_id, conn)
        target_span_ids.update(s.span_id for s in target_spans)
    
    if not target_span_ids:
        return []
    
    # Find other entities in target spans
    co_occurring = defaultdict(int)
    
    cur = conn.cursor()
    for span in focus_bundle.spans:
        if span.span_id in target_span_ids:
            # Get all entity mentions in this span's offset range
            cur.execute("""
                SELECT DISTINCT em.entity_id, e.entity_type
                FROM entity_mentions em
                JOIN entities e ON e.id = em.entity_id
                WHERE em.chunk_id = %s
                  AND em.start_char IS NOT NULL
                  AND em.start_char <= %s 
                  AND em.end_char >= %s
                  AND em.entity_id != ALL(%s)
                  AND e.entity_type = 'person'
            """, (span.chunk_id, span.end_char, span.start_char, target_entity_ids))
            
            for entity_id, _ in cur.fetchall():
                co_occurring[entity_id] += 1
    
    # Sort by co-occurrence count, return top N
    sorted_entities = sorted(co_occurring.items(), key=lambda x: -x[1])
    return [eid for eid, _ in sorted_entities[:max_entities]]


def get_entity_aliases(entity_ids: List[int], conn) -> List[str]:
    """
    Get aliases/codenames for entities.
    
    Args:
        entity_ids: List of entity IDs
        conn: Database connection
    
    Returns:
        List of alias strings (including canonical names)
    """
    if not entity_ids:
        return []
    
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT alias
        FROM entity_aliases
        WHERE entity_id = ANY(%s)
        UNION
        SELECT canonical_name
        FROM entities
        WHERE id = ANY(%s)
    """, (entity_ids, entity_ids))
    
    return [row[0] for row in cur.fetchall() if row[0]]


def extract_expansion_terms(
    focus_bundle: "FocusBundle",
    min_focus_count: int = 2,
    max_terms: int = 10,
) -> List[Tuple[str, int]]:
    """
    Extract high-specificity terms from FocusBundle for expansion.
    
    Targets:
    - ALL CAPS terms (codenames)
    - Hyphenated technical terms
    - Quoted phrases
    
    Guardrails:
    - Term must appear in >=2 FocusSpans OR appear in quotes/"code name" pattern
    - Stop list blocks common terms
    
    Args:
        focus_bundle: The FocusBundle to extract from
        min_focus_count: Minimum occurrence count (default 2)
        max_terms: Maximum terms to return
    
    Returns:
        List of (term, count) tuples sorted by count descending
    """
    term_counts = defaultdict(int)
    term_in_quotes = set()
    
    for span in focus_bundle.spans:
        # ALL CAPS (codenames) - 3-15 chars
        caps = re.findall(r'\b[A-Z]{3,15}\b', span.text)
        for cap in caps:
            if cap not in EXPANSION_STOP_LIST:
                term_counts[cap] += 1
        
        # Hyphenated terms (technical)
        hyphenated = re.findall(r'\b\w+-\w+(?:-\w+)*\b', span.text)
        for term in hyphenated:
            if len(term) >= 5 and term.upper() not in EXPANSION_STOP_LIST:
                term_counts[term.upper()] += 1
        
        # Check for quoted or "code name" pattern
        if 'code name' in span.text.lower() or 'codename' in span.text.lower():
            nearby_caps = re.findall(r'\b[A-Z]{3,15}\b', span.text)
            term_in_quotes.update(nearby_caps)
        
        # Quoted phrases (3-30 chars)
        quoted = re.findall(r'"([A-Z][^"]{2,30})"', span.text)
        term_in_quotes.update(quoted)
    
    # Filter: >=min_focus_count occurrences OR in quotes/codename pattern
    valid_terms = []
    for term, count in term_counts.items():
        if count >= min_focus_count or term in term_in_quotes:
            valid_terms.append((term, count))
    
    # Sort by count descending
    valid_terms.sort(key=lambda x: -x[1])
    
    return valid_terms[:max_terms]


def entity_expansion_loop(
    query_contract: "QueryContract",
    initial_focus_bundle: "FocusBundle",
    conn,
    retrieval_fn,  # Function to run retrieval: (query, terms, collections, conn) -> chunks
    max_rounds: int = 2,
    max_expanded_chunks: int = 500,
    stability_threshold: float = 0.85,
) -> "FocusBundle":
    """
    Entity-driven expansion loop for relationship/constraint queries.
    
    Process:
    1. Extract co-occurring entities from initial FocusBundle
    2. Get their aliases/codenames
    3. Run expanded retrieval
    4. Rebuild FocusBundle
    5. Stop when FocusBundle spans stabilize (Jaccard >= threshold)
    
    Args:
        query_contract: The QueryContract
        initial_focus_bundle: Starting FocusBundle
        conn: Database connection
        retrieval_fn: Function to run retrieval
        max_rounds: Maximum expansion rounds (default 2)
        max_expanded_chunks: Budget for total chunks (default 500)
        stability_threshold: Jaccard threshold for stopping (default 0.85)
    
    Returns:
        Final FocusBundle after expansion
    """
    from retrieval.focus_bundle import FocusBundleBuilder
    
    current_bundle = initial_focus_bundle
    target_ids = query_contract.get_target_entity_ids()
    
    total_chunks = current_bundle.params.get("total_chunks", 0)
    
    for round_num in range(max_rounds):
        # Check budget
        if total_chunks >= max_expanded_chunks:
            break
        
        # Extract expansion entities
        expansion_ids = extract_expansion_entities(
            current_bundle, 
            target_ids, 
            conn, 
            max_entities=10
        )
        
        if not expansion_ids:
            break
        
        # Get aliases for expanded entities
        expansion_terms = get_entity_aliases(expansion_ids, conn)
        
        if not expansion_terms:
            break
        
        # Run expanded retrieval
        expanded_chunks = retrieval_fn(
            query_contract.query_text,
            expansion_terms,
            query_contract.scope_collections,
            conn,
        )
        
        if not expanded_chunks:
            break
        
        total_chunks += len(expanded_chunks)
        
        # Rebuild FocusBundle
        builder = FocusBundleBuilder()
        new_bundle = builder.build(query_contract, expanded_chunks, conn)
        
        # Check stability (Jaccard on span IDs)
        old_span_ids = {s.span_id for s in current_bundle.spans}
        new_span_ids = {s.span_id for s in new_bundle.spans}
        
        if old_span_ids and new_span_ids:
            jaccard = len(old_span_ids & new_span_ids) / len(old_span_ids | new_span_ids)
        else:
            jaccard = 0.0
        
        if jaccard >= stability_threshold:
            break  # Stable - stop expanding
        
        current_bundle = new_bundle
    
    return current_bundle


def term_expansion_loop(
    query_contract: "QueryContract",
    initial_focus_bundle: "FocusBundle",
    conn,
    retrieval_fn,
    max_rounds: int = 2,
    max_expanded_chunks: int = 500,
    stability_threshold: float = 0.85,
) -> "FocusBundle":
    """
    Term-driven expansion loop for "other names" queries.
    
    Process:
    1. Extract high-specificity terms from FocusBundle
    2. Run multi-query retrieval (query + term_i)
    3. Rebuild FocusBundle
    4. Stop on stability
    
    Args:
        query_contract: The QueryContract
        initial_focus_bundle: Starting FocusBundle
        conn: Database connection
        retrieval_fn: Function to run retrieval
        max_rounds: Maximum expansion rounds
        max_expanded_chunks: Budget for total chunks
        stability_threshold: Jaccard threshold for stopping
    
    Returns:
        Final FocusBundle after expansion
    """
    from retrieval.focus_bundle import FocusBundleBuilder
    
    current_bundle = initial_focus_bundle
    total_chunks = current_bundle.params.get("total_chunks", 0)
    
    for round_num in range(max_rounds):
        if total_chunks >= max_expanded_chunks:
            break
        
        # Extract expansion terms
        terms_with_counts = extract_expansion_terms(current_bundle)
        expansion_terms = [t for t, _ in terms_with_counts[:5]]
        
        if not expansion_terms:
            break
        
        # Run expanded retrieval
        expanded_chunks = retrieval_fn(
            query_contract.query_text,
            expansion_terms,
            query_contract.scope_collections,
            conn,
        )
        
        if not expanded_chunks:
            break
        
        total_chunks += len(expanded_chunks)
        
        # Rebuild FocusBundle
        builder = FocusBundleBuilder()
        new_bundle = builder.build(
            query_contract, 
            expanded_chunks, 
            conn,
            anchor_terms=expansion_terms,
        )
        
        # Check stability
        old_span_ids = {s.span_id for s in current_bundle.spans}
        new_span_ids = {s.span_id for s in new_bundle.spans}
        
        if old_span_ids and new_span_ids:
            jaccard = len(old_span_ids & new_span_ids) / len(old_span_ids | new_span_ids)
        else:
            jaccard = 0.0
        
        if jaccard >= stability_threshold:
            break
        
        current_bundle = new_bundle
    
    return current_bundle


def should_expand(
    focus_bundle: "FocusBundle",
    query_contract: "QueryContract",
    min_spans_for_expansion: int = 10,
) -> bool:
    """
    Determine if expansion is likely to help.
    
    Expansion is useful when:
    - We have some initial results but not enough
    - Query has targets that could have related entities
    - FocusBundle has potential expansion terms
    
    Args:
        focus_bundle: Current FocusBundle
        query_contract: The QueryContract
        min_spans_for_expansion: Minimum spans before expansion makes sense
    
    Returns:
        True if expansion is recommended
    """
    # Need some initial results
    if len(focus_bundle.spans) < min_spans_for_expansion:
        return False
    
    # Need targets for entity expansion
    if query_contract.get_target_entity_ids():
        return True
    
    # Check if we have potential expansion terms
    terms = extract_expansion_terms(focus_bundle, min_focus_count=2, max_terms=5)
    if terms:
        return True
    
    return False
