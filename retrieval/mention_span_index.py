"""
Mention-to-Span Index for Agentic V2.

Provides O(1) lookups from entity_id to span_ids.

CONTRACT C9: Build an in-memory index once per FocusBundle for fast entity lookups.
Single query over entity_mentions for the bundle's chunks.
"""

from collections import defaultdict
from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.focus_bundle import FocusBundle


def build_mention_span_index(
    focus_bundle: "FocusBundle",
    conn,
) -> Dict[int, List[str]]:
    """
    Build entity_id -> [span_ids] mapping for all entities in FocusBundle chunks.
    
    Single query, O(1) lookup per candidate (Contract C9).
    
    Args:
        focus_bundle: The FocusBundle to index
        conn: Database connection
    
    Returns:
        Dict mapping entity_id to list of span_ids where that entity appears
    """
    if not focus_bundle.spans:
        return {}
    
    # Get all chunk_ids from focus bundle
    chunk_ids = list({s.chunk_id for s in focus_bundle.spans})
    
    if not chunk_ids:
        return {}
    
    # Single query for all mentions in these chunks
    cur = conn.cursor()
    cur.execute("""
        SELECT entity_id, chunk_id, start_char, end_char
        FROM entity_mentions
        WHERE chunk_id = ANY(%s) AND start_char IS NOT NULL
        ORDER BY entity_id, chunk_id, start_char
    """, (chunk_ids,))
    
    mentions = cur.fetchall()
    
    # Build span lookup by chunk_id for efficiency
    spans_by_chunk: Dict[int, List] = defaultdict(list)
    for span in focus_bundle.spans:
        spans_by_chunk[span.chunk_id].append(span)
    
    # Map mentions to spans
    entity_spans: Dict[int, Set[str]] = defaultdict(set)
    
    for entity_id, m_chunk, m_start, m_end in mentions:
        # Check all spans in this chunk for overlap
        for span in spans_by_chunk.get(m_chunk, []):
            # Check offset overlap (Contract C1: raw offsets)
            if span.start_char <= m_end and span.end_char >= m_start:
                entity_spans[entity_id].add(span.span_id)
    
    # Convert sets to lists for JSON serialization
    return {eid: list(sids) for eid, sids in entity_spans.items()}


def build_reverse_span_index(
    focus_bundle: "FocusBundle",
    conn,
) -> Dict[str, List[int]]:
    """
    Build span_id -> [entity_ids] reverse mapping.
    
    Useful for finding all entities mentioned in a span.
    
    Args:
        focus_bundle: The FocusBundle to index
        conn: Database connection
    
    Returns:
        Dict mapping span_id to list of entity_ids mentioned in that span
    """
    if not focus_bundle.spans:
        return {}
    
    chunk_ids = list({s.chunk_id for s in focus_bundle.spans})
    
    if not chunk_ids:
        return {}
    
    cur = conn.cursor()
    cur.execute("""
        SELECT entity_id, chunk_id, start_char, end_char
        FROM entity_mentions
        WHERE chunk_id = ANY(%s) AND start_char IS NOT NULL
        ORDER BY chunk_id, start_char
    """, (chunk_ids,))
    
    mentions = cur.fetchall()
    
    # Build span lookup by chunk_id
    spans_by_chunk: Dict[int, List] = defaultdict(list)
    for span in focus_bundle.spans:
        spans_by_chunk[span.chunk_id].append(span)
    
    # Map spans to entities
    span_entities: Dict[str, Set[int]] = defaultdict(set)
    
    for entity_id, m_chunk, m_start, m_end in mentions:
        for span in spans_by_chunk.get(m_chunk, []):
            if span.start_char <= m_end and span.end_char >= m_start:
                span_entities[span.span_id].add(entity_id)
    
    return {sid: list(eids) for sid, eids in span_entities.items()}


def get_entities_in_span(
    focus_bundle: "FocusBundle",
    span_id: str,
    conn,
) -> List[int]:
    """
    Get all entity_ids mentioned in a specific span.
    
    Args:
        focus_bundle: The FocusBundle
        span_id: The span_id to look up
        conn: Database connection
    
    Returns:
        List of entity_ids mentioned in the span
    """
    # Find the span
    span = focus_bundle.get_span(span_id)
    if not span:
        return []
    
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT entity_id
        FROM entity_mentions
        WHERE chunk_id = %s 
          AND start_char IS NOT NULL
          AND start_char <= %s 
          AND end_char >= %s
    """, (span.chunk_id, span.end_char, span.start_char))
    
    return [row[0] for row in cur.fetchall()]


def get_person_entities_in_bundle(
    focus_bundle: "FocusBundle",
    conn,
) -> Dict[int, dict]:
    """
    Get all person entities mentioned in the FocusBundle.
    
    Returns dict: entity_id -> {canonical_name, entity_type, span_ids}
    """
    if not focus_bundle.mention_span_index:
        return {}
    
    entity_ids = list(focus_bundle.mention_span_index.keys())
    if not entity_ids:
        return {}
    
    cur = conn.cursor()
    cur.execute("""
        SELECT id, canonical_name, entity_type
        FROM entities
        WHERE id = ANY(%s) AND entity_type = 'person'
    """, (entity_ids,))
    
    result = {}
    for eid, name, etype in cur.fetchall():
        result[eid] = {
            "canonical_name": name,
            "entity_type": etype,
            "span_ids": focus_bundle.mention_span_index.get(eid, []),
        }
    
    return result
