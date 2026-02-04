"""
Match Trace Construction for Two-Mode Retrieval System.

This module provides tiered enrichment of match traces:

Tier 0: Hot columns from search (scores, ranks) - always populated, no DB queries
Tier 1: Entity IDs from entity_mentions (batch query) - always populated
Tier 2: Full entity details for LLM synthesis - only top N chunks
Tier 3: Phrase positions, full text highlights - on-demand only

HARD RULE: Never load full chunk text for more than summarization_limit chunks
during execution, even in thorough mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class PositiveHit:
    """Detail for actual matches (entities, phrases)."""
    primitive: str  # e.g., "ENTITY", "PHRASE", "TERM"
    hit_type: str  # e.g., "entity_mention", "phrase_match", "term_match"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchTrace:
    """
    Match trace explaining why a chunk surfaced.
    
    Split into tiers for performance:
    - Tier 0: Always populated from search results
    - Tier 1: Entity IDs (batch query)
    - Tier 2: Full details for summarization (top N only)
    - Tier 3: On-demand (phrase positions)
    """
    chunk_id: int
    
    # Tier 0: Hot columns (always populated from search results)
    scope_passed: bool = True  # Always TRUE for included results
    in_lexical: bool = False
    in_vector: bool = False
    score_lexical: Optional[float] = None
    score_vector: Optional[float] = None
    score_hybrid: Optional[float] = None
    vector_distance: Optional[float] = None
    vector_similarity: Optional[float] = None
    rank: Optional[int] = None
    rank_trace: Optional[Dict[str, Any]] = None
    
    # Tier 1: Entity IDs (cheap batch query, always populated)
    matched_entity_ids: List[int] = field(default_factory=list)
    
    # Tier 2: Summarization details (only for top N chunks)
    matched_phrases: Optional[List[str]] = None
    entity_details: Optional[List[Dict[str, Any]]] = None
    
    # Tier 3: On-demand details (NOT stored, computed on request)
    # phrase_positions, full_highlight, context_window
    
    # Cap info
    was_capped: bool = False
    cap_reason: Optional[str] = None
    
    # Document info (from search result)
    document_id: Optional[int] = None
    
    def to_hot_columns(self) -> Dict[str, Any]:
        """Extract typed columns for fast queries (Tier 0 + Tier 1)."""
        return {
            "scope_passed": self.scope_passed,
            "in_lexical": self.in_lexical,
            "in_vector": self.in_vector,
            "score_lexical": self.score_lexical,
            "score_vector": self.score_vector,
            "score_hybrid": self.score_hybrid,
            "vector_distance": self.vector_distance,
            "vector_similarity": self.vector_similarity,
            "rank": self.rank,
            "matched_entity_ids": self.matched_entity_ids,
            "was_capped": self.was_capped,
            "cap_reason": self.cap_reason,
        }
    
    def to_db_row(self, result_set_id: int, retrieval_run_id: Optional[int] = None) -> Dict[str, Any]:
        """Convert to database row for result_set_match_traces table."""
        return {
            "result_set_id": result_set_id,
            "chunk_id": self.chunk_id,
            "retrieval_run_id": retrieval_run_id,
            "matched_entity_ids": self.matched_entity_ids or [],
            "matched_phrases": self.matched_phrases or [],
            "scope_passed": self.scope_passed,
            "in_lexical": self.in_lexical,
            "in_vector": self.in_vector,
            "score_lexical": self.score_lexical,
            "score_vector": self.score_vector,
            "score_hybrid": self.score_hybrid,
            "vector_distance": self.vector_distance,
            "vector_similarity": self.vector_similarity,
            "rank": self.rank,
            "rank_trace": self.rank_trace,
            "was_capped": self.was_capped,
            "cap_reason": self.cap_reason,
        }
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "chunk_id": self.chunk_id,
            "matched_entity_ids": self.matched_entity_ids,
            "matched_phrases": self.matched_phrases or [],
            "scope_passed": self.scope_passed,
            "in_lexical": self.in_lexical,
            "in_vector": self.in_vector,
            "scores": {
                "lexical": self.score_lexical,
                "vector": self.score_vector,
                "hybrid": self.score_hybrid,
            },
            "rank": self.rank,
            "rank_explanation": self._build_rank_explanation(),
            "entity_details": self.entity_details,
        }
    
    def _build_rank_explanation(self) -> Optional[str]:
        """Build human-readable rank explanation."""
        if self.rank_trace:
            return str(self.rank_trace)
        
        parts = []
        if self.in_lexical and self.in_vector:
            parts.append("matched both lexical and vector")
        elif self.in_lexical:
            parts.append("lexical match only")
        elif self.in_vector:
            parts.append("vector match only")
        
        if self.score_hybrid:
            parts.append(f"hybrid score: {self.score_hybrid:.4f}")
        
        return "; ".join(parts) if parts else None


# =============================================================================
# Tier 0: Build Base Traces (no DB queries)
# =============================================================================

def build_base_traces(
    results: List[Any],  # ThresholdSearchResult or similar
    search_metadata: Any,  # ThresholdSearchMetadata
    mode: str,
) -> List[MatchTrace]:
    """
    Tier 0: Build match traces from search results only. No DB queries.
    
    This is called immediately after search returns, populating hot columns
    directly from the search results.
    
    Args:
        results: List of search results (ThresholdSearchResult or similar)
        search_metadata: Metadata from search (ThresholdSearchMetadata)
        mode: Retrieval mode ("conversational" or "thorough")
        
    Returns:
        List of MatchTrace with Tier 0 fields populated
    """
    traces = []
    
    for result in results:
        trace = MatchTrace(
            chunk_id=result.chunk_id,
            scope_passed=True,  # All results in set passed scope
            in_lexical=getattr(result, 'in_lexical', False),
            in_vector=getattr(result, 'in_vector', False),
            score_lexical=getattr(result, 'score_lexical', None),
            score_vector=getattr(result, 'score_vector', None),
            score_hybrid=getattr(result, 'score_hybrid', None),
            vector_distance=getattr(result, 'vector_distance', None),
            vector_similarity=getattr(result, 'vector_similarity', None),
            rank=getattr(result, 'rank', None),
            document_id=getattr(result, 'document_id', None),
            was_capped=search_metadata.cap_applied if search_metadata else False,
            cap_reason=f"max_hits={search_metadata.cap_value}" if search_metadata and search_metadata.cap_applied else None,
        )
        
        # Build rank trace based on mode
        if mode == "conversational" and trace.in_lexical and trace.in_vector:
            trace.rank_trace = {
                "method": "rrf",
                "rank_lex": getattr(result, 'rank_lexical', None),
                "rank_vec": getattr(result, 'rank_vector', None),
            }
        elif mode == "thorough":
            trace.rank_trace = {
                "method": "deterministic",
                "order_by": "(document_id, chunk_id)",
            }
        
        traces.append(trace)
    
    return traces


# =============================================================================
# Tier 1: Entity ID Enrichment (single batch query)
# =============================================================================

def enrich_tier1_entity_ids(
    traces: List[MatchTrace],
    entity_primitive_ids: List[int],
    conn,
) -> List[MatchTrace]:
    """
    Tier 1: Add matched entity IDs from entity_mentions. Single batch query.
    
    This is cheap (single query) and always run, populating the matched_entity_ids
    field for all traces.
    
    Args:
        traces: List of MatchTrace from Tier 0
        entity_primitive_ids: Entity IDs from query primitives (ENTITY, EXCEPT_ENTITIES, etc.)
        conn: Database connection
        
    Returns:
        Same traces with matched_entity_ids populated
    """
    if not traces:
        return traces
    
    chunk_ids = [t.chunk_id for t in traces]
    
    # If no specific entities to match, get all entities in chunks
    if not entity_primitive_ids:
        query = """
            SELECT chunk_id, array_agg(DISTINCT entity_id ORDER BY entity_id) as entity_ids
            FROM entity_mentions
            WHERE chunk_id = ANY(%(chunk_ids)s)
            GROUP BY chunk_id
        """
        params = {"chunk_ids": chunk_ids}
    else:
        # Filter to entities from primitives
        query = """
            SELECT chunk_id, array_agg(DISTINCT entity_id ORDER BY entity_id) as entity_ids
            FROM entity_mentions
            WHERE chunk_id = ANY(%(chunk_ids)s)
              AND entity_id = ANY(%(entity_ids)s)
            GROUP BY chunk_id
        """
        params = {"chunk_ids": chunk_ids, "entity_ids": entity_primitive_ids}
    
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    
    # Build lookup
    chunk_to_entities = {row[0]: list(row[1]) if row[1] else [] for row in rows}
    
    # Populate traces
    for trace in traces:
        trace.matched_entity_ids = chunk_to_entities.get(trace.chunk_id, [])
    
    return traces


# =============================================================================
# Tier 2: Summarization Enrichment (top N only)
# =============================================================================

def enrich_tier2_for_summarization(
    traces: List[MatchTrace],
    primitives: List[Any],
    conn,
    limit: int = 20,
) -> List[MatchTrace]:
    """
    Tier 2: Full enrichment for top N chunks only (for LLM summarization).
    
    HARD RULE: Never load full chunk text for more than `limit` chunks.
    This prevents thorough mode from accidentally doing expensive LLM prep.
    
    Args:
        traces: List of MatchTrace from Tier 1
        primitives: Query primitives (for phrase extraction)
        conn: Database connection
        limit: Maximum chunks to enrich (default 20, the summarization limit)
        
    Returns:
        Traces with top N having matched_phrases and entity_details populated
    """
    if not traces:
        return traces
    
    # Sort by rank and take top N
    sorted_traces = sorted(traces, key=lambda t: t.rank if t.rank else float('inf'))
    top_traces = sorted_traces[:limit]
    top_chunk_ids = [t.chunk_id for t in top_traces]
    
    if not top_chunk_ids:
        return traces
    
    # Extract phrases from primitives
    phrases = _extract_phrases_from_primitives(primitives)
    
    # Get entity details for matched entity IDs
    all_entity_ids = set()
    for trace in top_traces:
        all_entity_ids.update(trace.matched_entity_ids)
    
    entity_details_map = {}
    if all_entity_ids:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, canonical_name, entity_type
                FROM entities
                WHERE id = ANY(%(entity_ids)s)
            """, {"entity_ids": list(all_entity_ids)})
            for row in cur.fetchall():
                entity_details_map[row[0]] = {
                    "entity_id": row[0],
                    "name": row[1],
                    "type": row[2],
                }
    
    # Populate entity details and matched phrases for top traces
    for trace in top_traces:
        # Entity details
        trace.entity_details = [
            entity_details_map.get(eid, {"entity_id": eid})
            for eid in trace.matched_entity_ids
        ]
        
        # Matched phrases (just the list, not positions - Tier 3)
        trace.matched_phrases = phrases if phrases else []
    
    return traces


def _extract_phrases_from_primitives(primitives: List[Any]) -> List[str]:
    """Extract PHRASE values from primitives."""
    phrases = []
    for p in primitives:
        ptype = getattr(p, 'type', None) or (p.get('type') if isinstance(p, dict) else None)
        if ptype == "PHRASE":
            value = getattr(p, 'value', None) or (p.get('value') if isinstance(p, dict) else None)
            if value:
                phrases.append(value)
        elif ptype == "TERM":
            value = getattr(p, 'value', None) or (p.get('value') if isinstance(p, dict) else None)
            if value:
                phrases.append(value)
    return phrases


# =============================================================================
# Tier 3: On-Demand Details (computed per request)
# =============================================================================

def get_phrase_positions_on_demand(
    chunk_id: int,
    phrases: List[str],
    conn,
    case_sensitive: bool = False,
    context_chars: int = 50,
) -> List[Dict[str, Any]]:
    """
    Tier 3: Compute phrase positions on-demand when UI requests.
    
    Called via API: GET /chunks/{chunk_id}/highlights?phrases=...
    NOT called during execution.
    
    Args:
        chunk_id: Chunk to search
        phrases: Phrases to find
        conn: Database connection
        case_sensitive: Whether matching is case-sensitive (default False)
        context_chars: Characters of context around match (default 50, max 200)
        
    Returns:
        List of match details with positions and context
    """
    if not phrases:
        return []
    
    # Cap context_chars for safety
    context_chars = min(context_chars, 200)
    
    # Load chunk text (single chunk only - Tier 3 rule)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COALESCE(clean_text, text) as content
            FROM chunks
            WHERE id = %(chunk_id)s
        """, {"chunk_id": chunk_id})
        result = cur.fetchone()
        if not result:
            return []
        
        text = result[0] or ""
    
    matches = []
    
    for phrase in phrases:
        # Find all occurrences
        if case_sensitive:
            pattern = re.escape(phrase)
        else:
            pattern = re.escape(phrase)
            flags = re.IGNORECASE
        
        for match in re.finditer(pattern, text, flags=0 if case_sensitive else re.IGNORECASE):
            start = match.start()
            end = match.end()
            
            # Extract context
            ctx_start = max(0, start - context_chars)
            ctx_end = min(len(text), end + context_chars)
            context = text[ctx_start:ctx_end]
            
            # Add ellipsis if truncated
            if ctx_start > 0:
                context = "..." + context
            if ctx_end < len(text):
                context = context + "..."
            
            matches.append({
                "phrase": phrase,
                "start": start,
                "end": end,
                "context": context,
            })
    
    # Sort by position
    matches.sort(key=lambda m: m["start"])
    
    return matches


# =============================================================================
# Persistence Functions
# =============================================================================

def persist_match_traces(
    traces: List[MatchTrace],
    result_set_id: int,
    retrieval_run_id: int,
    conn,
) -> int:
    """
    Persist match traces to result_set_match_traces table.
    
    Args:
        traces: List of MatchTrace to persist
        result_set_id: Result set ID
        retrieval_run_id: Retrieval run ID
        conn: Database connection
        
    Returns:
        Number of rows inserted
    """
    if not traces:
        return 0
    
    # Build values for batch insert
    values = []
    for trace in traces:
        values.append((
            result_set_id,
            trace.chunk_id,
            retrieval_run_id,
            trace.matched_entity_ids or [],
            trace.matched_phrases or [],
            trace.scope_passed,
            trace.in_lexical,
            trace.in_vector,
            trace.score_lexical,
            trace.score_vector,
            trace.score_hybrid,
            trace.vector_distance,
            trace.vector_similarity,
            trace.rank,
            trace.rank_trace,
            trace.was_capped,
            trace.cap_reason,
        ))
    
    # Batch insert
    from psycopg2.extras import execute_values, Json
    
    insert_sql = """
        INSERT INTO result_set_match_traces (
            result_set_id, chunk_id, retrieval_run_id,
            matched_entity_ids, matched_phrases,
            scope_passed, in_lexical, in_vector,
            score_lexical, score_vector, score_hybrid,
            vector_distance, vector_similarity,
            rank, rank_trace,
            was_capped, cap_reason
        ) VALUES %s
        ON CONFLICT (result_set_id, chunk_id) DO UPDATE SET
            retrieval_run_id = EXCLUDED.retrieval_run_id,
            matched_entity_ids = EXCLUDED.matched_entity_ids,
            matched_phrases = EXCLUDED.matched_phrases,
            scope_passed = EXCLUDED.scope_passed,
            in_lexical = EXCLUDED.in_lexical,
            in_vector = EXCLUDED.in_vector,
            score_lexical = EXCLUDED.score_lexical,
            score_vector = EXCLUDED.score_vector,
            score_hybrid = EXCLUDED.score_hybrid,
            vector_distance = EXCLUDED.vector_distance,
            vector_similarity = EXCLUDED.vector_similarity,
            rank = EXCLUDED.rank,
            rank_trace = EXCLUDED.rank_trace,
            was_capped = EXCLUDED.was_capped,
            cap_reason = EXCLUDED.cap_reason
    """
    
    with conn.cursor() as cur:
        # Convert rank_trace dicts to Json for psycopg2
        json_values = []
        for v in values:
            v_list = list(v)
            v_list[14] = Json(v_list[14]) if v_list[14] else None  # rank_trace
            json_values.append(tuple(v_list))
        
        execute_values(cur, insert_sql, json_values)
        row_count = cur.rowcount
    
    return row_count


def load_match_traces(
    result_set_id: int,
    conn,
    chunk_ids: Optional[List[int]] = None,
    include_audit: bool = False,
) -> List[MatchTrace]:
    """
    Load match traces from database.
    
    Args:
        result_set_id: Result set ID
        conn: Database connection
        chunk_ids: Optional filter to specific chunks
        include_audit: Whether to include full audit details
        
    Returns:
        List of MatchTrace
    """
    query = """
        SELECT 
            chunk_id, retrieval_run_id,
            matched_entity_ids, matched_phrases,
            scope_passed, in_lexical, in_vector,
            score_lexical, score_vector, score_hybrid,
            vector_distance, vector_similarity,
            rank, rank_trace,
            was_capped, cap_reason
        FROM result_set_match_traces
        WHERE result_set_id = %(result_set_id)s
    """
    params = {"result_set_id": result_set_id}
    
    if chunk_ids:
        query += " AND chunk_id = ANY(%(chunk_ids)s)"
        params["chunk_ids"] = chunk_ids
    
    query += " ORDER BY rank NULLS LAST, chunk_id"
    
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    
    traces = []
    for row in rows:
        (chunk_id, retrieval_run_id, matched_entity_ids, matched_phrases,
         scope_passed, in_lexical, in_vector,
         score_lexical, score_vector, score_hybrid,
         vector_distance, vector_similarity,
         rank, rank_trace, was_capped, cap_reason) = row
        
        trace = MatchTrace(
            chunk_id=chunk_id,
            matched_entity_ids=list(matched_entity_ids) if matched_entity_ids else [],
            matched_phrases=list(matched_phrases) if matched_phrases else [],
            scope_passed=scope_passed,
            in_lexical=in_lexical,
            in_vector=in_vector,
            score_lexical=score_lexical,
            score_vector=score_vector,
            score_hybrid=score_hybrid,
            vector_distance=vector_distance,
            vector_similarity=vector_similarity,
            rank=rank,
            rank_trace=dict(rank_trace) if rank_trace else None,
            was_capped=was_capped,
            cap_reason=cap_reason,
        )
        traces.append(trace)
    
    return traces


# =============================================================================
# Page Range Helper
# =============================================================================

def get_page_range(chunk_id: int, conn) -> Optional[str]:
    """
    Get page range for a chunk with deterministic ordering.
    
    Uses span_order for stable page ranges.
    """
    query = """
        SELECT string_agg(p.logical_page_label, '-' ORDER BY cp.span_order) as page_range
        FROM chunk_pages cp
        JOIN pages p ON cp.page_id = p.id
        WHERE cp.chunk_id = %(chunk_id)s
        GROUP BY cp.chunk_id
    """
    
    with conn.cursor() as cur:
        cur.execute(query, {"chunk_id": chunk_id})
        result = cur.fetchone()
        return result[0] if result else None
