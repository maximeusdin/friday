"""
V3/V4 Tool Registry - Typed wrappers around search and index primitives.

Each tool:
- Wraps an existing retrieval function
- Returns a typed ToolResult with chunk_ids, scores, and metadata
- Logs execution for audit trail

TOOL CATEGORIES:

1. SEARCH TOOLS (return chunk_ids from vector/lexical indexes):
   - hybrid_search: Combined vector + lexical via RRF (best for most queries)
   - vector_search: Pure semantic similarity (good for concepts, paraphrases)
   - lexical_search: Terms must appear in chunk (good for specific names/phrases)
   - lexical_exact: Exact substring match (good for codenames, exact phrases)

2. ENTITY TOOLS (use entity_mentions index):
   - entity_lookup: Find entity ID by name
   - entity_surfaces: Get all surface forms (names, aliases) for an entity
   - entity_mentions: Find all chunks mentioning an entity
   - co_mention_entities: Find entities that co-occur with another entity

3. CONCORDANCE TOOLS (use concordance/alias database):
   - expand_aliases: Get aliases/variants for a term from concordance

4. INDEX TOOLS (query structured indexes directly):
   - first_mention: Find chronologically earliest mention of an entity
   - first_co_mention: Find earliest co-occurrence of two entities
   - date_range_search: Search within specific date range
   
Each tool returns:
- chunk_ids: List of matching chunk IDs (empty for metadata-only tools)
- scores: Dict[chunk_id, score] for ranking
- metadata: Tool-specific results (aliases, entity info, etc.)
"""

import time
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Sequence, Union

from retrieval.ops import (
    hybrid_rrf,
    vector_search as ops_vector_search,
    lex_and,
    lex_exact,
    SearchFilters,
    ChunkHit,
    concordance_expand_terms,
)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    params: Dict[str, Any]
    chunk_ids: List[int]
    scores: Dict[int, float]  # chunk_id -> score
    metadata: Dict[str, Any]
    elapsed_ms: float
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "params": self.params,
            "chunk_ids": self.chunk_ids,
            "scores": self.scores,
            "metadata": self.metadata,
            "elapsed_ms": self.elapsed_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ToolSpec:
    """Specification for a tool."""
    name: str
    description: str
    params_schema: Dict[str, Any]
    fn: Callable


def _build_filters(
    collections: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    document_id: Optional[int] = None,
) -> SearchFilters:
    """Build SearchFilters from common parameters."""
    return SearchFilters(
        collection_slugs=collections if collections else None,
        date_from=date_from,
        date_to=date_to,
        document_id=document_id,
    )


def _extract_chunk_hits(hits: List[ChunkHit]) -> tuple:
    """Extract chunk_ids and scores from ChunkHit list."""
    chunk_ids = [h.chunk_id for h in hits]
    scores = {}
    for h in hits:
        if h.score is not None:
            scores[h.chunk_id] = h.score
        elif h.distance is not None:
            # Convert distance to score (lower distance = higher score)
            scores[h.chunk_id] = 1.0 / (1.0 + h.distance)
    return chunk_ids, scores


# =============================================================================
# Tool Implementations
# =============================================================================

def hybrid_search_tool(
    conn,
    query: str = None,
    top_k: int = 200,
    collections: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    expand_concordance: bool = True,
    fuzzy_enabled: bool = True,
) -> ToolResult:
    """
    Hybrid search combining vector + lexical via Reciprocal Rank Fusion.
    
    This is the primary search tool for most queries.
    ALWAYS expands concordance aliases automatically.
    GRACEFUL: Validates inputs and never crashes.
    """
    start = time.time()
    
    # Validate and normalize inputs
    if query is None:
        query = ""
    if not isinstance(query, str):
        query = str(query)
    query = query.strip()
    
    if not query:
        return ToolResult(
            tool_name="hybrid_search",
            params={"query": query, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"message": "Empty query provided"},
            elapsed_ms=0,
            success=True,  # Graceful empty result
        )
    
    # Validate top_k
    if not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except:
            top_k = 200
    top_k = min(max(top_k, 1), 500)
    
    try:
        filters = _build_filters(collections, date_from, date_to)
        
        # First, get the aliases that will be expanded (for reporting)
        expanded_aliases = []
        if expand_concordance:
            try:
                expanded_aliases = concordance_expand_terms(
                    conn=conn,
                    text=query,
                    max_aliases_out=25,
                )
            except Exception:
                pass
        
        hits = hybrid_rrf(
            conn=conn,
            query=query,
            filters=filters,
            k=top_k,
            expand_concordance=expand_concordance,
            fuzzy_lex_enabled=fuzzy_enabled,
            log_run=False,  # We log at tool level
        )
        
        chunk_ids, scores = _extract_chunk_hits(hits)
        
        elapsed = (time.time() - start) * 1000
        
        return ToolResult(
            tool_name="hybrid_search",
            params={
                "query": query,
                "top_k": top_k,
                "collections": collections,
            },
            chunk_ids=chunk_ids,
            scores=scores,
            metadata={
                "total_hits": len(hits),
                "has_scores": len(scores) > 0,
                "aliases_expanded": len(expanded_aliases) > 0,
                "expanded_aliases": expanded_aliases[:10] if expanded_aliases else [],
                "total_alias_count": len(expanded_aliases),
            },
            elapsed_ms=elapsed,
        )
        
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()
        except:
            pass
        return ToolResult(
            tool_name="hybrid_search",
            params={"query": query, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"error_type": type(e).__name__},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def vector_search_tool(
    conn,
    query: str = None,
    top_k: int = 200,
    collections: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    expand_concordance: bool = True,
) -> ToolResult:
    """
    Pure vector similarity search.
    
    Best for semantic queries where exact term matching isn't critical.
    ALWAYS expands concordance aliases for the embedding query.
    GRACEFUL: Validates inputs and never crashes.
    """
    start = time.time()
    
    # Validate and normalize inputs
    if query is None:
        query = ""
    if not isinstance(query, str):
        query = str(query)
    query = query.strip()
    
    if not query:
        return ToolResult(
            tool_name="vector_search",
            params={"query": query, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"message": "Empty query provided"},
            elapsed_ms=0,
            success=True,  # Graceful empty result
        )
    
    # Validate top_k
    if not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except:
            top_k = 200
    top_k = min(max(top_k, 1), 500)
    
    try:
        filters = _build_filters(collections, date_from, date_to)
        
        # Get aliases for reporting
        expanded_aliases = []
        if expand_concordance:
            try:
                expanded_aliases = concordance_expand_terms(
                    conn=conn,
                    text=query,
                    max_aliases_out=25,
                )
            except Exception:
                pass
        
        hits = ops_vector_search(
            conn=conn,
            query=query,
            filters=filters,
            k=top_k,
            expand_concordance=expand_concordance,
            log_run=False,
        )
        
        chunk_ids, scores = _extract_chunk_hits(hits)
        
        elapsed = (time.time() - start) * 1000
        
        return ToolResult(
            tool_name="vector_search",
            params={
                "query": query,
                "top_k": top_k,
                "collections": collections,
            },
            chunk_ids=chunk_ids,
            scores=scores,
            metadata={
                "total_hits": len(hits),
                "aliases_expanded": len(expanded_aliases) > 0,
                "expanded_aliases": expanded_aliases[:10] if expanded_aliases else [],
                "total_alias_count": len(expanded_aliases),
            },
            elapsed_ms=elapsed,
        )
        
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()
        except:
            pass
        return ToolResult(
            tool_name="vector_search",
            params={"query": query, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"error_type": type(e).__name__},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def lexical_search_tool(
    conn,
    terms: List[str],
    top_k: int = 200,
    collections: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    expand_aliases: bool = True,
) -> ToolResult:
    """
    Lexical search WITH automatic alias expansion.
    
    If single term: uses lex_exact (substring match)
    If multiple terms: uses lex_and (all terms must appear)
    
    Automatically expands concordance aliases for each term.
    Best for precise queries where specific terms must be present.
    """
    start = time.time()
    
    # Validate terms
    if not terms or not isinstance(terms, list):
        return ToolResult(
            tool_name="lexical_search",
            params={"terms": terms, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"message": "terms must be a non-empty list"},
            elapsed_ms=0,
            success=True,  # Graceful
        )
    
    # Filter out empty terms
    terms = [t.strip() for t in terms if t and isinstance(t, str) and t.strip()]
    if not terms:
        return ToolResult(
            tool_name="lexical_search",
            params={"terms": terms, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"message": "no valid terms after filtering"},
            elapsed_ms=0,
            success=True,  # Graceful
        )
    
    try:
        filters = _build_filters(collections, date_from, date_to)
        
        # Expand aliases for each term
        all_search_variants = []  # List of (original_term, [aliases])
        total_aliases = 0
        
        for term in terms:
            term_variants = [term]
            if expand_aliases:
                try:
                    expanded = concordance_expand_terms(
                        conn=conn,
                        text=term,
                        max_aliases_out=10,
                    )
                    for alias in expanded:
                        if alias.lower() != term.lower() and alias not in term_variants:
                            term_variants.append(alias)
                            total_aliases += 1
                except Exception:
                    pass
            all_search_variants.append((term, term_variants))
        
        # For single term with expansion: search all variants
        # For multiple terms: we need at least one variant of EACH term to match
        all_chunk_ids = set()
        all_scores = {}
        search_mode = "unknown"
        term_hit_details = {}
        
        if len(terms) == 1:
            # Single term: search all variants
            search_mode = "exact_with_aliases" if total_aliases > 0 else "exact"
            term, variants = all_search_variants[0]
            term_hit_details[term] = {"variants": variants, "hits_by_variant": {}}
            
            for variant in variants:
                hits = lex_exact(
                    conn=conn,
                    term=variant,
                    filters=filters,
                    k=top_k,
                    case_sensitive=False,
                    log_run=False,
                )
                term_hit_details[term]["hits_by_variant"][variant] = len(hits)
                
                for hit in hits:
                    cid = hit.chunk_id
                    if cid not in all_chunk_ids:
                        all_chunk_ids.add(cid)
                        all_scores[cid] = hit.score if hasattr(hit, 'score') and hit.score else 0.5
                    else:
                        all_scores[cid] = min(all_scores.get(cid, 0.5) + 0.1, 1.0)
        else:
            # Multiple terms: expand each term and search
            # For now, use the primary term for each with lex_and
            search_mode = "and_with_aliases" if total_aliases > 0 else "and"
            
            # First, run lex_and with original terms
            hits = lex_and(
                conn=conn,
                terms=terms,
                filters=filters,
                k=top_k,
                log_run=False,
            )
            for hit in hits:
                all_chunk_ids.add(hit.chunk_id)
                all_scores[hit.chunk_id] = hit.score if hasattr(hit, 'score') and hit.score else 0.5
            
            # Then search for alias combinations (union results)
            if total_aliases > 0:
                for term, variants in all_search_variants:
                    for variant in variants[1:]:  # Skip original (already searched)
                        # Replace this term with its variant
                        variant_terms = [v if t != term else variant for t, (_, variants_list) in zip(terms, all_search_variants) for v in [t]]
                        if len(set(variant_terms)) >= 2:  # Only if we have different terms
                            try:
                                hits = lex_and(
                                    conn=conn,
                                    terms=variant_terms,
                                    filters=filters,
                                    k=top_k // 2,
                                    log_run=False,
                                )
                                for hit in hits:
                                    if hit.chunk_id not in all_chunk_ids:
                                        all_chunk_ids.add(hit.chunk_id)
                                        all_scores[hit.chunk_id] = 0.4
                            except Exception:
                                pass
        
        chunk_ids = list(all_chunk_ids)[:top_k]
        scores = {cid: all_scores[cid] for cid in chunk_ids}
        
        elapsed = (time.time() - start) * 1000
        
        # Build alias summary for metadata
        alias_summary = {}
        for term, variants in all_search_variants:
            if len(variants) > 1:
                alias_summary[term] = variants[1:]  # Skip original
        
        return ToolResult(
            tool_name="lexical_search",
            params={
                "terms": terms,
                "top_k": top_k,
                "expand_aliases": expand_aliases,
            },
            chunk_ids=chunk_ids,
            scores=scores,
            metadata={
                "total_hits": len(chunk_ids),
                "search_mode": search_mode,
                "aliases_expanded": total_aliases > 0,
                "total_alias_count": total_aliases,
                "alias_summary": alias_summary,
            },
            elapsed_ms=elapsed,
        )
        
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()
        except:
            pass
        return ToolResult(
            tool_name="lexical_search",
            params={"terms": terms, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"error_type": type(e).__name__},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def lexical_exact_tool(
    conn,
    term: str = None,
    top_k: int = 200,
    collections: Optional[List[str]] = None,
    case_sensitive: bool = False,
    expand_aliases: bool = True,
) -> ToolResult:
    """
    Exact substring match search WITH automatic alias expansion.
    
    Best for finding specific phrases or names.
    Automatically expands concordance aliases and searches for all variants.
    GRACEFUL: Validates inputs and never crashes.
    """
    start = time.time()
    
    # Validate and normalize inputs
    if term is None:
        term = ""
    if not isinstance(term, str):
        term = str(term)
    term = term.strip()
    
    if not term:
        return ToolResult(
            tool_name="lexical_exact",
            params={"term": term, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"message": "Empty term provided"},
            elapsed_ms=0,
            success=True,  # Graceful empty result
        )
    
    # Validate top_k
    if not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except:
            top_k = 200
    top_k = min(max(top_k, 1), 500)
    
    try:
        filters = _build_filters(collections)
        
        # Expand aliases from concordance
        search_terms = [term]
        expanded_aliases = []
        if expand_aliases:
            try:
                expanded_aliases = concordance_expand_terms(
                    conn=conn,
                    text=term,
                    max_aliases_out=15,
                )
                if expanded_aliases:
                    # Add aliases that aren't the original term
                    for alias in expanded_aliases:
                        if alias.lower() != term.lower() and alias not in search_terms:
                            search_terms.append(alias)
            except Exception:
                pass  # Graceful - continue with original term
        
        # Search for all terms and collect results
        all_chunk_ids = set()
        all_scores = {}
        term_hit_counts = {}
        
        for search_term in search_terms:
            hits = lex_exact(
                conn=conn,
                term=search_term,
                filters=filters,
                k=top_k,
                case_sensitive=case_sensitive,
                log_run=False,
            )
            term_hit_counts[search_term] = len(hits)
            
            for hit in hits:
                cid = hit.chunk_id
                if cid not in all_chunk_ids:
                    all_chunk_ids.add(cid)
                    all_scores[cid] = hit.score if hasattr(hit, 'score') and hit.score else 0.5
                else:
                    # Boost score for chunks hit by multiple aliases
                    all_scores[cid] = min(all_scores.get(cid, 0.5) + 0.1, 1.0)
        
        chunk_ids = list(all_chunk_ids)[:top_k]
        scores = {cid: all_scores[cid] for cid in chunk_ids}
        
        elapsed = (time.time() - start) * 1000
        
        return ToolResult(
            tool_name="lexical_exact",
            params={
                "term": term,
                "top_k": top_k,
                "expand_aliases": expand_aliases,
            },
            chunk_ids=chunk_ids,
            scores=scores,
            metadata={
                "total_hits": len(chunk_ids),
                "search_terms": search_terms,
                "term_hit_counts": term_hit_counts,
                "aliases_expanded": len(search_terms) > 1,
                "alias_count": len(search_terms) - 1,
            },
            elapsed_ms=elapsed,
        )
        
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()
        except:
            pass
        return ToolResult(
            tool_name="lexical_exact",
            params={"term": term, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={"error_type": type(e).__name__},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def expand_aliases_tool(
    conn,
    term: str,
    max_aliases: int = 25,
) -> ToolResult:
    """
    Get aliases/variants for a term from the concordance.
    
    Useful for expanding searches to include alternate names.
    Returns aliases in metadata['aliases'], not as chunk_ids.
    """
    start = time.time()
    
    if not term or not term.strip():
        return ToolResult(
            tool_name="expand_aliases",
            params={"term": term, "max_aliases": max_aliases},
            chunk_ids=[],
            scores={},
            metadata={"aliases": [], "alias_count": 0, "error": "empty term"},
            elapsed_ms=0,
            success=False,
            error="term must be non-empty",
        )
    
    try:
        aliases = concordance_expand_terms(
            conn=conn,
            text=term,
            max_aliases_out=max_aliases,
        )
        
        elapsed = (time.time() - start) * 1000
        
        return ToolResult(
            tool_name="expand_aliases",
            params={"term": term, "max_aliases": max_aliases},
            chunk_ids=[],  # This tool doesn't return chunks - aliases are in metadata
            scores={},
            metadata={
                "aliases": aliases,
                "alias_count": len(aliases),
                "note": "Aliases returned in metadata, not as chunks. Use these for subsequent searches.",
            },
            elapsed_ms=elapsed,
        )
        
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        import traceback
        return ToolResult(
            tool_name="expand_aliases",
            params={"term": term, "max_aliases": max_aliases},
            chunk_ids=[],
            scores={},
            metadata={
                "aliases": [], 
                "alias_count": 0,
                "error_details": traceback.format_exc()[:500],
            },
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


# =============================================================================
# Entity Tools - Use entity_mentions index for precise entity-based search
# =============================================================================

def _lookup_entity_by_name(conn, name: str):
    """
    Internal helper to lookup entity by name with concordance expansion.
    
    Returns: (entity_id, canonical_name, entity_type, matched_via) or (None, None, None, None)
    """
    if not name or not name.strip():
        return None, None, None, None
    
    name = name.strip()
    
    with conn.cursor() as cur:
        # Try exact canonical name match first
        cur.execute(
            "SELECT id, canonical_name, entity_type FROM entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 1",
            (name,)
        )
        row = cur.fetchone()
        if row:
            return row[0], row[1], row[2], "canonical_name"
        
        # Try alias match
        cur.execute("""
            SELECT e.id, e.canonical_name, e.entity_type
            FROM entities e
            JOIN entity_aliases ea ON ea.entity_id = e.id
            WHERE LOWER(ea.alias) = LOWER(%s)
            LIMIT 1
        """, (name,))
        row = cur.fetchone()
        if row:
            return row[0], row[1], row[2], "alias"
        
        # Try partial match on canonical name
        cur.execute("""
            SELECT id, canonical_name, entity_type 
            FROM entities 
            WHERE LOWER(canonical_name) LIKE LOWER(%s)
            ORDER BY LENGTH(canonical_name)
            LIMIT 1
        """, (f"%{name}%",))
        row = cur.fetchone()
        if row:
            return row[0], row[1], row[2], "partial_match"
        
        # Try concordance expansion - maybe the name is a codename or variant
        try:
            expanded = concordance_expand_terms(
                conn=conn,
                text=name,
                max_aliases_out=10,
            )
            for alias in expanded:
                if alias.lower() != name.lower():
                    # Try each expanded alias
                    cur.execute(
                        "SELECT id, canonical_name, entity_type FROM entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 1",
                        (alias,)
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0], row[1], row[2], f"concordance:{alias}"
                    
                    # Also try alias table
                    cur.execute("""
                        SELECT e.id, e.canonical_name, e.entity_type
                        FROM entities e
                        JOIN entity_aliases ea ON ea.entity_id = e.id
                        WHERE LOWER(ea.alias) = LOWER(%s)
                        LIMIT 1
                    """, (alias,))
                    row = cur.fetchone()
                    if row:
                        return row[0], row[1], row[2], f"concordance:{alias}"
        except Exception:
            pass  # Concordance expansion failed - continue without it
        
        return None, None, None, None


def entity_lookup_tool(
    conn,
    name: str = None,
) -> ToolResult:
    """
    Look up an entity by name to get its ID.
    
    Searches canonical names, aliases, AND concordance expansions.
    This means searching for "PAL" will find "Nathan Gregory Silvermaster".
    
    GRACEFUL: Returns success=True with found=False if entity doesn't exist.
    Only returns success=False on actual errors.
    
    Example: entity_lookup("Silvermaster") -> {entity_id: 123, canonical_name: "Nathan Gregory Silvermaster"}
    Example: entity_lookup("PAL") -> {entity_id: 123, canonical_name: "Nathan Gregory Silvermaster", matched_via: "concordance:PAL"}
    """
    start = time.time()
    
    # Validate input
    if name is None:
        name = ""
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    
    if not name:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="entity_lookup",
            params={"name": name},
            chunk_ids=[],
            scores={},
            metadata={"found": False, "message": "Empty name provided"},
            elapsed_ms=elapsed,
            success=True,  # Not a crash - graceful empty result
        )
    
    try:
        entity_id, canonical_name, entity_type, matched_via = _lookup_entity_by_name(conn, name)
        
        elapsed = (time.time() - start) * 1000
        
        if entity_id:
            return ToolResult(
                tool_name="entity_lookup",
                params={"name": name},
                chunk_ids=[],  # No chunks - entity info in metadata
                scores={},
                metadata={
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                    "found": True,
                    "matched_via": matched_via,
                },
                elapsed_ms=elapsed,
                success=True,
            )
        else:
            # Entity not found - graceful, not an error
            return ToolResult(
                tool_name="entity_lookup",
                params={"name": name},
                chunk_ids=[],
                scores={},
                metadata={
                    "found": False, 
                    "message": f"No entity found matching '{name}'. Try lexical_exact or hybrid_search instead.",
                },
                elapsed_ms=elapsed,
                success=True,  # Graceful - not a crash
            )
                
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()
        except:
            pass
        return ToolResult(
            tool_name="entity_lookup",
            params={"name": name},
            chunk_ids=[],
            scores={},
            metadata={"error_type": type(e).__name__},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def entity_surfaces_tool(
    conn,
    entity_id: Optional[int] = None,
    name: Optional[str] = None,
) -> ToolResult:
    """
    Get all surface forms (names, aliases) for an entity.
    
    Returns canonical name and all known aliases in metadata.
    Use these surface forms for subsequent lexical searches.
    
    Can use EITHER entity_id OR name (name will be looked up with concordance expansion).
    
    Example: entity_surfaces(entity_id=123) -> {canonical: "Silvermaster", aliases: ["PAL", "Robert"]}
    Example: entity_surfaces(name="PAL") -> finds Silvermaster's surfaces via concordance
    """
    start = time.time()
    
    # Validate and normalize inputs
    if name is not None and not isinstance(name, str):
        name = str(name)
    if name:
        name = name.strip()
    
    if entity_id is not None:
        try:
            entity_id = int(entity_id)
        except (ValueError, TypeError):
            entity_id = None
    
    # Resolve entity_id from name if needed (with concordance expansion)
    resolved_id = entity_id
    matched_via = None
    
    if resolved_id is None and name:
        eid, canonical, etype, matched = _lookup_entity_by_name(conn, name)
        if eid:
            resolved_id = eid
            matched_via = matched
        else:
            elapsed = (time.time() - start) * 1000
            return ToolResult(
                tool_name="entity_surfaces",
                params={"name": name},
                chunk_ids=[],
                scores={},
                metadata={
                    "found": False,
                    "message": f"Entity '{name}' not found (checked concordance too).",
                },
                elapsed_ms=elapsed,
                success=True,  # Graceful
            )
    
    if resolved_id is None:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="entity_surfaces",
            params={"entity_id": entity_id, "name": name},
            chunk_ids=[],
            scores={},
            metadata={"message": "No entity_id or name provided."},
            elapsed_ms=elapsed,
            success=True,  # Graceful
        )
    
    try:
        from retrieval.agent.entity_surfaces import EntitySurfaceIndex
        
        index = EntitySurfaceIndex(conn)
        surfaces = index.get_surfaces(resolved_id)
        canonical = index.get_canonical_name(resolved_id)
        
        elapsed = (time.time() - start) * 1000
        
        return ToolResult(
            tool_name="entity_surfaces",
            params={"entity_id": resolved_id, "name": name},
            chunk_ids=[],  # No chunks - surface forms in metadata
            scores={},
            metadata={
                "entity_id": resolved_id,
                "canonical_name": canonical,
                "surfaces": list(surfaces),
                "surface_count": len(surfaces),
                "matched_via": matched_via,
            },
            elapsed_ms=elapsed,
        )
        
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()
        except:
            pass
        return ToolResult(
            tool_name="entity_surfaces",
            params={"entity_id": resolved_id, "name": name},
            chunk_ids=[],
            scores={},
            metadata={"error_type": type(e).__name__},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def entity_mentions_tool(
    conn,
    entity_id: Optional[int] = None,
    name: Optional[str] = None,
    top_k: int = 200,
) -> ToolResult:
    """
    Find all chunks that mention a specific entity.
    
    Uses the entity_mentions index which has pre-computed entity references.
    Much more precise than lexical search for finding specific people/orgs.
    
    Can use EITHER entity_id OR name (name will be looked up automatically).
    Name lookup includes CONCORDANCE EXPANSION - so "PAL" finds Silvermaster.
    
    GRACEFUL: Returns empty results (not error) if entity not found.
    
    Example: entity_mentions(name="Silvermaster", top_k=100) -> chunks mentioning Silvermaster
    Example: entity_mentions(name="PAL") -> finds Silvermaster via concordance
    """
    start = time.time()
    
    # Validate and normalize inputs
    if name is not None and not isinstance(name, str):
        name = str(name)
    if name:
        name = name.strip()
    
    if entity_id is not None:
        try:
            entity_id = int(entity_id)
        except (ValueError, TypeError):
            entity_id = None
    
    if top_k is None or not isinstance(top_k, int):
        top_k = 200
    top_k = min(max(top_k, 1), 500)  # Clamp to reasonable range
    
    # Resolve entity_id from name if needed (with concordance expansion)
    resolved_id = entity_id
    resolved_name = None
    matched_via = None
    
    if resolved_id is None and name:
        # Look up entity by name (includes concordance expansion)
        eid, canonical, etype, matched = _lookup_entity_by_name(conn, name)
        if eid:
            resolved_id = eid
            resolved_name = canonical
            matched_via = matched
        else:
            # Entity not found - return empty results gracefully
            elapsed = (time.time() - start) * 1000
            return ToolResult(
                tool_name="entity_mentions",
                params={"name": name, "top_k": top_k},
                chunk_ids=[],
                scores={},
                metadata={
                    "entity_found": False,
                    "message": f"Entity '{name}' not in database (checked concordance too). Try hybrid_search or lexical_exact instead.",
                    "suggestion": f"lexical_exact(term='{name}')",
                },
                elapsed_ms=elapsed,
                success=True,  # Graceful - not a crash
            )
    
    if resolved_id is None:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="entity_mentions",
            params={"entity_id": entity_id, "name": name, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={
                "message": "No entity_id or name provided. Provide one of these parameters.",
            },
            elapsed_ms=elapsed,
            success=True,  # Graceful - not a crash
        )
    
    try:
        with conn.cursor() as cur:
            # Get chunks mentioning this entity, ordered by document/page for coherence
            cur.execute("""
                SELECT DISTINCT em.chunk_id, cm.document_id, cm.first_page_id
                FROM entity_mentions em
                LEFT JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
                WHERE em.entity_id = %s
                ORDER BY cm.document_id, cm.first_page_id
                LIMIT %s
            """, (resolved_id, top_k))
            
            rows = cur.fetchall()
            chunk_ids = [row[0] for row in rows]
            
            # Get entity name for logging if not already resolved
            if not resolved_name:
                cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (resolved_id,))
                name_row = cur.fetchone()
                resolved_name = name_row[0] if name_row else f"Entity {resolved_id}"
            
            elapsed = (time.time() - start) * 1000
            
            # Assign scores based on position (earlier = higher score)
            scores = {cid: 1.0 - (i * 0.001) for i, cid in enumerate(chunk_ids)}
            
            # Get unique documents
            doc_ids = list({row[1] for row in rows if row[1]})
            
            return ToolResult(
                tool_name="entity_mentions",
                params={"entity_id": resolved_id, "name": name, "top_k": top_k},
                chunk_ids=chunk_ids,
                scores=scores,
                metadata={
                    "entity_id": resolved_id,
                    "entity_name": resolved_name,
                    "total_mentions": len(chunk_ids),
                    "unique_documents": len(doc_ids),
                    "document_ids": doc_ids[:10],
                    "matched_via": matched_via,
                },
                elapsed_ms=elapsed,
            )
            
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()  # Rollback to recover from error
        except:
            pass
        return ToolResult(
            tool_name="entity_mentions",
            params={"entity_id": entity_id, "name": name, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def co_mention_entities_tool(
    conn,
    entity_id: Optional[int] = None,
    name: Optional[str] = None,
    top_k: int = 30,
) -> ToolResult:
    """
    Find entities that frequently co-occur with a given entity.
    
    Returns other entities mentioned in the same chunks as the target entity.
    Useful for discovering network members, associates, or related concepts.
    
    Can use EITHER entity_id OR name (name will be looked up automatically).
    Name lookup includes CONCORDANCE EXPANSION - so "PAL" finds Silvermaster's co-mentions.
    
    GRACEFUL: Returns empty results (not error) if entity not found.
    
    Example: co_mention_entities(name="Silvermaster") -> [Harry White, Ullmann, Perlo, ...]
    Example: co_mention_entities(name="PAL") -> finds Silvermaster's co-mentions via concordance
    """
    start = time.time()
    
    # Validate and normalize inputs
    if name is not None and not isinstance(name, str):
        name = str(name)
    if name:
        name = name.strip()
    
    if entity_id is not None:
        try:
            entity_id = int(entity_id)
        except (ValueError, TypeError):
            entity_id = None
    
    if top_k is None or not isinstance(top_k, int):
        top_k = 30
    top_k = min(max(top_k, 1), 100)  # Clamp to reasonable range
    
    # Resolve entity_id from name if needed (with concordance expansion)
    resolved_id = entity_id
    resolved_name = None
    matched_via = None
    
    if resolved_id is None and name:
        # Look up entity by name (includes concordance expansion)
        eid, canonical, etype, matched = _lookup_entity_by_name(conn, name)
        if eid:
            resolved_id = eid
            resolved_name = canonical
            matched_via = matched
        else:
            # Entity not found - return empty results gracefully
            elapsed = (time.time() - start) * 1000
            return ToolResult(
                tool_name="co_mention_entities",
                params={"name": name, "top_k": top_k},
                chunk_ids=[],
                scores={},
                metadata={
                    "entity_found": False,
                    "message": f"Entity '{name}' not in database (checked concordance too). Try hybrid_search instead.",
                    "co_entities": [],  # Empty list, not missing key
                },
                elapsed_ms=elapsed,
                success=True,  # Graceful - not a crash
            )
    
    if resolved_id is None:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="co_mention_entities",
            params={"entity_id": entity_id, "name": name, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={
                "message": "No entity_id or name provided.",
                "co_entities": [],
            },
            elapsed_ms=elapsed,
            success=True,  # Graceful - not a crash
        )
    
    try:
        with conn.cursor() as cur:
            # Find entities mentioned in same chunks, ranked by co-occurrence count
            cur.execute("""
                SELECT 
                    em2.entity_id, 
                    e.canonical_name, 
                    e.entity_type,
                    COUNT(DISTINCT em1.chunk_id) as co_count
                FROM entity_mentions em1
                JOIN entity_mentions em2 ON em2.chunk_id = em1.chunk_id AND em2.entity_id != em1.entity_id
                JOIN entities e ON e.id = em2.entity_id
                WHERE em1.entity_id = %s
                GROUP BY em2.entity_id, e.canonical_name, e.entity_type
                ORDER BY co_count DESC
                LIMIT %s
            """, (resolved_id, top_k))
            
            co_entities = [
                {
                    "entity_id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "co_occurrence_count": row[3],
                }
                for row in cur.fetchall()
            ]
            
            # Also get chunks where top co-occurring entities appear together
            chunk_ids = []
            if co_entities:
                top_entity_ids = [e["entity_id"] for e in co_entities[:5]]
                cur.execute("""
                    SELECT DISTINCT em1.chunk_id
                    FROM entity_mentions em1
                    WHERE em1.entity_id = %s
                    AND EXISTS (
                        SELECT 1 FROM entity_mentions em2 
                        WHERE em2.chunk_id = em1.chunk_id AND em2.entity_id = ANY(%s)
                    )
                    LIMIT 100
                """, (resolved_id, top_entity_ids))
                chunk_ids = [row[0] for row in cur.fetchall()]
            
            elapsed = (time.time() - start) * 1000
            
            return ToolResult(
                tool_name="co_mention_entities",
                params={"entity_id": resolved_id, "name": name, "top_k": top_k},
                chunk_ids=chunk_ids,
                scores={cid: 0.5 for cid in chunk_ids},
                metadata={
                    "source_entity_id": resolved_id,
                    "source_entity_name": resolved_name,
                    "co_entities": co_entities,
                    "co_entity_count": len(co_entities),
                    "matched_via": matched_via,
                },
                elapsed_ms=elapsed,
            )
            
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()  # Rollback to recover from error
        except:
            pass
        return ToolResult(
            tool_name="co_mention_entities",
            params={"entity_id": entity_id, "name": name, "top_k": top_k},
            chunk_ids=[],
            scores={},
            metadata={},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def first_mention_tool(
    conn,
    entity_id: Optional[int] = None,
    name: Optional[str] = None,
) -> ToolResult:
    """
    Find the chronologically earliest mention of an entity.
    
    Uses date metadata to find the first document/chunk where entity appears.
    Useful for establishing when a person first appears in the record.
    
    Can use EITHER entity_id OR name (name will be looked up automatically).
    Name lookup includes CONCORDANCE EXPANSION - so "PAL" finds Silvermaster's first mention.
    
    GRACEFUL: Returns empty results (not error) if entity not found.
    
    Example: first_mention(name="Alger Hiss") -> earliest chunk mentioning Hiss
    Example: first_mention(name="ALES") -> finds Hiss's first mention via concordance
    """
    start = time.time()
    
    # Validate and normalize inputs
    if name is not None and not isinstance(name, str):
        name = str(name)
    if name:
        name = name.strip()
    
    if entity_id is not None:
        try:
            entity_id = int(entity_id)
        except (ValueError, TypeError):
            entity_id = None
    
    # Resolve entity_id from name if needed (with concordance expansion)
    resolved_id = entity_id
    resolved_name = None
    matched_via = None
    
    if resolved_id is None and name:
        # Look up entity by name (includes concordance expansion)
        eid, canonical, etype, matched = _lookup_entity_by_name(conn, name)
        if eid:
            resolved_id = eid
            resolved_name = canonical
            matched_via = matched
        else:
            # Entity not found - return empty results gracefully
            elapsed = (time.time() - start) * 1000
            return ToolResult(
                tool_name="first_mention",
                params={"name": name},
                chunk_ids=[],
                scores={},
                metadata={
                    "entity_found": False,
                    "message": f"Entity '{name}' not in database (checked concordance too).",
                },
                elapsed_ms=elapsed,
                success=True,  # Graceful - not a crash
            )
    
    if resolved_id is None:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="first_mention",
            params={"entity_id": entity_id, "name": name},
            chunk_ids=[],
            scores={},
            metadata={"message": "No entity_id or name provided."},
            elapsed_ms=elapsed,
            success=True,  # Graceful - not a crash
        )
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT em.chunk_id, cm.document_id, cm.date_min, d.source_name
                FROM entity_mentions em
                JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
                LEFT JOIN documents d ON d.id = cm.document_id
                WHERE em.entity_id = %s AND cm.date_min IS NOT NULL
                ORDER BY cm.date_min ASC
                LIMIT 1
            """, (resolved_id,))
            
            row = cur.fetchone()
            elapsed = (time.time() - start) * 1000
            
            if row:
                return ToolResult(
                    tool_name="first_mention",
                    params={"entity_id": resolved_id, "name": name},
                    chunk_ids=[row[0]],
                    scores={row[0]: 1.0},
                    metadata={
                        "entity_id": resolved_id,
                        "entity_name": resolved_name,
                        "document_id": row[1],
                        "date": str(row[2]) if row[2] else None,
                        "source": row[3],
                        "matched_via": matched_via,
                    },
                    elapsed_ms=elapsed,
                )
            else:
                return ToolResult(
                    tool_name="first_mention",
                    params={"entity_id": resolved_id, "name": name},
                    chunk_ids=[],
                    scores={},
                    metadata={
                        "message": "No dated mentions found", 
                        "entity_name": resolved_name,
                        "matched_via": matched_via,
                    },
                    elapsed_ms=elapsed,
                )
                
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        try:
            conn.rollback()
        except:
            pass
        return ToolResult(
            tool_name="first_mention",
            params={"entity_id": entity_id, "name": name},
            chunk_ids=[],
            scores={},
            metadata={},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "hybrid_search": ToolSpec(
        name="hybrid_search",
        description="Combines vector + lexical search via RRF. Best general-purpose search.",
        params_schema={
            "query": {"type": "string", "required": True},
            "top_k": {"type": "integer", "default": 200},
            "collections": {"type": "array", "items": "string"},
            "date_from": {"type": "string", "format": "date"},
            "date_to": {"type": "string", "format": "date"},
            "expand_concordance": {"type": "boolean", "default": True},
            "fuzzy_enabled": {"type": "boolean", "default": True},
        },
        fn=hybrid_search_tool,
    ),
    "vector_search": ToolSpec(
        name="vector_search",
        description="Pure vector similarity search. Best for semantic queries.",
        params_schema={
            "query": {"type": "string", "required": True},
            "top_k": {"type": "integer", "default": 200},
            "collections": {"type": "array", "items": "string"},
            "expand_concordance": {"type": "boolean", "default": True},
        },
        fn=vector_search_tool,
    ),
    "lexical_search": ToolSpec(
        name="lexical_search",
        description="All terms must appear in chunk. Best for precise term matching.",
        params_schema={
            "terms": {"type": "array", "items": "string", "required": True},
            "top_k": {"type": "integer", "default": 200},
            "collections": {"type": "array", "items": "string"},
        },
        fn=lexical_search_tool,
    ),
    "lexical_exact": ToolSpec(
        name="lexical_exact",
        description="Exact substring match. Best for specific phrases or names.",
        params_schema={
            "term": {"type": "string", "required": True},
            "top_k": {"type": "integer", "default": 200},
            "collections": {"type": "array", "items": "string"},
            "case_sensitive": {"type": "boolean", "default": False},
        },
        fn=lexical_exact_tool,
    ),
    "expand_aliases": ToolSpec(
        name="expand_aliases",
        description="Get aliases/variants for a term from concordance. Returns aliases in metadata['aliases'], not as chunks. Use these aliases for subsequent lexical_exact searches.",
        params_schema={
            "term": {"type": "string", "required": True},
            "max_aliases": {"type": "integer", "default": 25},
        },
        fn=expand_aliases_tool,
    ),
    # Entity tools
    "entity_lookup": ToolSpec(
        name="entity_lookup",
        description="Find an entity by name to get its ID. Searches canonical names and aliases. Returns entity_id, canonical_name, entity_type in metadata. Use before entity_mentions or entity_surfaces.",
        params_schema={
            "name": {"type": "string", "required": True},
        },
        fn=entity_lookup_tool,
    ),
    "entity_surfaces": ToolSpec(
        name="entity_surfaces",
        description="Get all surface forms (canonical name + aliases) for an entity. Returns surfaces in metadata. Use these for lexical searches.",
        params_schema={
            "entity_id": {"type": "integer", "required": True},
        },
        fn=entity_surfaces_tool,
    ),
    "entity_mentions": ToolSpec(
        name="entity_mentions",
        description="Find all chunks mentioning a specific entity. More precise than lexical search. Use 'name' parameter directly (e.g., name='Silvermaster') - no need to call entity_lookup first.",
        params_schema={
            "entity_id": {"type": "integer", "required": False},
            "name": {"type": "string", "required": False},  # Alternative to entity_id
            "top_k": {"type": "integer", "default": 200},
        },
        fn=entity_mentions_tool,
    ),
    "co_mention_entities": ToolSpec(
        name="co_mention_entities",
        description="Find entities that co-occur with a given entity. Great for discovering network members! Use 'name' parameter directly (e.g., name='Silvermaster') - no need to call entity_lookup first.",
        params_schema={
            "entity_id": {"type": "integer", "required": False},
            "name": {"type": "string", "required": False},  # Alternative to entity_id
            "top_k": {"type": "integer", "default": 30},
        },
        fn=co_mention_entities_tool,
    ),
    "first_mention": ToolSpec(
        name="first_mention",
        description="Find chronologically earliest mention of an entity. Use 'name' parameter directly (e.g., name='Alger Hiss').",
        params_schema={
            "entity_id": {"type": "integer", "required": False},
            "name": {"type": "string", "required": False},  # Alternative to entity_id
        },
        fn=first_mention_tool,
    ),
}


def get_tool(name: str) -> Optional[ToolSpec]:
    """Get a tool by name."""
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[str]:
    """List available tool names."""
    return list(TOOL_REGISTRY.keys())


def get_tools_for_prompt() -> str:
    """Get comprehensive tool descriptions formatted for LLM prompt."""
    lines = [
        "=" * 60,
        "AVAILABLE TOOLS AND PRIMITIVES",
        "=" * 60,
        "",
        "## SEARCH TOOLS (query vector/lexical indexes, return chunk_ids)",
        "",
        "1. hybrid_search(query, top_k=200)",
        "   - Combined vector + lexical search via Reciprocal Rank Fusion",
        "   - BEST for most queries - balances semantic and exact matching",
        "   - Example: hybrid_search(query='members of the Silvermaster network')",
        "",
        "2. vector_search(query, top_k=200)",
        "   - Pure semantic similarity search",
        "   - Good for concepts, paraphrases, questions without exact terms",
        "   - Example: vector_search(query='Soviet intelligence operations in US')",
        "",
        "3. lexical_search(terms, top_k=200)",
        "   - Terms must appear in chunk (AND semantics for multiple terms)",
        "   - Good for specific names, phrases that must be present",
        "   - Single term uses exact substring match",
        "   - Example: lexical_search(terms=['Silvermaster']) or lexical_search(terms=['Soviet', 'agent'])",
        "",
        "4. lexical_exact(term, top_k=200, case_sensitive=False)",
        "   - Exact substring match for a single term",
        "   - Best for codenames, exact phrases, proper nouns",
        "   - Example: lexical_exact(term='PAL') or lexical_exact(term='LIBERAL')",
        "",
        "## ENTITY TOOLS (query entity_mentions index)",
        "",
        "5. entity_lookup(name)",
        "   - Find entity ID by name (searches canonical names and aliases)",
        "   - Returns: metadata with entity_id, canonical_name, entity_type",
        "   - Optional: Other entity tools can use 'name' directly without lookup",
        "   - Example: entity_lookup(name='Silvermaster') -> {entity_id: 123, ...}",
        "",
        "6. entity_surfaces(entity_id)",
        "   - Get all surface forms (names, aliases) for an entity",
        "   - Returns: metadata with canonical_name and all aliases",
        "   - Use these surfaces for subsequent lexical searches",
        "   - Example: entity_surfaces(entity_id=123) -> {surfaces: ['Silvermaster', 'PAL', ...]}",
        "",
        "7. entity_mentions(name='...' OR entity_id=N, top_k=200)",
        "   - Find all chunks mentioning a specific entity",
        "   - MORE PRECISE than lexical search - uses pre-computed entity index",
        "   - Returns: chunk_ids directly",
        "   - RECOMMENDED: Use name directly, no need for entity_lookup first!",
        "   - Example: entity_mentions(name='Silvermaster') -> chunks where Silvermaster appears",
        "",
        "8. co_mention_entities(name='...' OR entity_id=N, top_k=30)",
        "   - Find entities that co-occur with a given entity",
        "   - EXCELLENT for discovering network members, associates",
        "   - Returns: related entities with co-occurrence counts + chunk_ids",
        "   - RECOMMENDED: Use name directly!",
        "   - Example: co_mention_entities(name='Silvermaster') -> people mentioned with Silvermaster",
        "",
        "9. first_mention(name='...' OR entity_id=N)",
        "   - Find chronologically earliest mention of an entity",
        "   - Uses date metadata",
        "   - Returns: first dated chunk",
        "   - Example: first_mention(name='Alger Hiss') -> earliest chunk mentioning Hiss",
        "",
        "## CONCORDANCE TOOLS",
        "",
        "10. expand_aliases(term, max_aliases=25)",
        "    - Get aliases/variants from concordance database",
        "    - Returns: metadata['aliases'] list (NOT chunk_ids)",
        "    - Use returned aliases for subsequent searches",
        "    - Example: expand_aliases(term='Silvermaster') -> {aliases: ['PAL', ...]}",
        "",
        "=" * 60,
        "TOOL STRATEGIES (use name parameter directly!)",
        "=" * 60,
        "",
        "For ROSTER/NETWORK queries (e.g., 'members of Silvermaster network'):",
        "  1. hybrid_search(query='members of the Silvermaster network')",
        "  2. co_mention_entities(name='Silvermaster')  <- discovers network members!",
        "  3. entity_mentions(name='Silvermaster') for all mentions",
        "",
        "For PERSON queries (e.g., 'what did X do'):",
        "  1. entity_mentions(name='Person Name') for precise references",
        "  2. entity_surfaces(entity_id=N) to get aliases, then...",
        "  3. lexical_exact(term='CODENAME') for codename references",
        "",
        "For EVIDENCE queries (e.g., 'evidence of X'):",
        "  1. hybrid_search(query='evidence of X')",
        "  2. lexical_search(terms=['key', 'terms'])",
        "  3. vector_search(query='semantic variant of X')",
        "",
        "For TIMELINE queries (e.g., 'when did X happen'):",
        "  1. first_mention(name='Entity') for chronological anchor",
        "  2. entity_mentions(name='Entity') for full context",
        "",
        "IMPORTANT: Use name='...' directly in entity tools. Do NOT use placeholder",
        "variable names like 'silvermaster_id'. Use the actual entity name string.",
        "",
    ]
    
    return "\n".join(lines)
