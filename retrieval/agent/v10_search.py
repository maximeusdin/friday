"""
V10 Structured Boost Search — search_chunks_v10().

Replaces query-string rewriting with structured EntityBoost / AliasScopedBoost
parameters.  Returns match_provenance per chunk so extraction can associate
hits with their originating boosts.

Builds on top of the existing hybrid_rrf() infrastructure in retrieval/ops.py.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    AliasScopedBoost,
    CatalogHitV10,
    EntityBoost,
    MatchProvenance,
)
from retrieval.ops import (
    ChunkHit,
    SearchFilters,
    hybrid_rrf,
    _build_where,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Boost query builder
# =============================================================================

def _build_boost_terms(
    entity_boosts: List[EntityBoost],
    alias_boosts_scoped: List[AliasScopedBoost],
    scope_collections: Optional[List[str]],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Build additional lexical terms from boosts.

    Returns:
        - extra_terms: list of strings to OR into the lexical query
        - term_provenance: map from normalised term -> {boost_type, entity_id, collection_scope, locked}
    """
    extra_terms: List[str] = []
    term_provenance: Dict[str, Dict[str, Any]] = {}

    # Entity boosts: always active (global)
    for eb in entity_boosts:
        for form in eb.forms:
            form_clean = form.strip()
            if form_clean:
                extra_terms.append(form_clean)
                term_provenance[form_clean.lower()] = {
                    "boost_type": "entity",
                    "entity_id": eb.entity_id,
                    "collection_scope": None,
                    "locked": False,
                    "weight": eb.weight,
                }

    # Alias boosts: only fire when scope includes the boost's collection_slug
    for ab in alias_boosts_scoped:
        # Check scope filtering — alias boost is ignored if scope excludes
        # its collection (invariant I1)
        if scope_collections is not None:
            if ab.collection_slug not in scope_collections:
                continue

        form_clean = ab.alias_text.strip()
        if form_clean:
            extra_terms.append(form_clean)
            term_provenance[form_clean.lower()] = {
                "boost_type": "alias",
                "entity_id": ab.locked_entity_id,
                "collection_scope": ab.collection_slug,
                "locked": ab.locked_entity_id is not None,
                "weight": ab.weight,
            }

    return extra_terms, term_provenance


def _build_boosted_query(query: str, extra_terms: List[str]) -> str:
    """Combine original query with boost terms for hybrid search.

    Rather than rewriting the query, we append boost terms so they
    contribute to both lexical and semantic matching.
    """
    if not extra_terms:
        return query
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_terms: List[str] = []
    query_lower = query.lower()
    for t in extra_terms:
        tl = t.lower()
        if tl not in seen and tl not in query_lower:
            seen.add(tl)
            unique_terms.append(t)
    if not unique_terms:
        return query
    return query + " " + " ".join(unique_terms)


# =============================================================================
# Provenance matching
# =============================================================================

def _compute_provenance(
    hits: List[ChunkHit],
    term_provenance: Dict[str, Dict[str, Any]],
) -> Dict[int, MatchProvenance]:
    """For each hit, determine which boost (if any) likely fired.

    Uses simple substring matching on the chunk preview text.
    This is a best-effort heuristic — perfect provenance would require
    modifying the SQL internals.
    """
    provenance: Dict[int, MatchProvenance] = {}
    for hit in hits:
        best_match: Optional[Dict[str, Any]] = None
        best_form = ""
        preview_lower = (hit.preview or "").lower()

        for term, prov_info in term_provenance.items():
            if term in preview_lower:
                if best_match is None or prov_info.get("weight", 1.0) > best_match.get("weight", 1.0):
                    best_match = prov_info
                    best_form = term

        if best_match:
            provenance[hit.chunk_id] = MatchProvenance(
                chunk_id=hit.chunk_id,
                boost_type=best_match["boost_type"],
                matched_form=best_form,
                collection_scope=best_match.get("collection_scope"),
                entity_id=best_match.get("entity_id"),
                locked=best_match.get("locked", False),
            )
        else:
            provenance[hit.chunk_id] = MatchProvenance(
                chunk_id=hit.chunk_id,
                boost_type="none",
            )

    return provenance


# =============================================================================
# Convert ChunkHit -> CatalogHitV10
# =============================================================================

def _resolve_page_no(conn, page_id: Optional[int]) -> Optional[int]:
    """Resolve raw page_id to PDF page number."""
    if page_id is None or conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT page_num FROM pages WHERE id = %s", (page_id,))
            row = cur.fetchone()
            if row:
                return row[0]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    return None


def _hits_to_catalog_v10(
    conn,
    hits: List[ChunkHit],
) -> List[CatalogHitV10]:
    """Convert ChunkHit list to CatalogHitV10 with full doc/page provenance."""
    results: List[CatalogHitV10] = []
    # Batch resolve page_ids for efficiency
    page_ids = [h.first_page_id for h in hits if h.first_page_id]
    page_no_map: Dict[int, int] = {}
    if page_ids and conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, page_num FROM pages WHERE id = ANY(%s)",
                    (page_ids,),
                )
                for row in cur.fetchall():
                    if row[1] is not None:
                        page_no_map[row[0]] = row[1]
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

    for hit in hits:
        page_id = getattr(hit, 'first_page_id', None)
        page_no = page_no_map.get(page_id) if page_id else None
        page_str = f"p{page_no}" if page_no is not None else None

        results.append(CatalogHitV10(
            chunk_id=hit.chunk_id,
            score=hit.score or 0.0,
            doc_id=hit.document_id,
            page=page_str,
            collection=hit.collection_slug,
            snippet=(hit.preview or "")[:400],
            document_id=hit.document_id,
            page_id=page_id,
            page_no=page_no,
            collection_slug=hit.collection_slug,
        ))

    return results


# =============================================================================
# Main entry point
# =============================================================================

def search_chunks_v10(
    conn,
    query: str,
    *,
    scope: Optional[SearchFilters] = None,
    entity_boosts: Optional[List[EntityBoost]] = None,
    alias_boosts_scoped: Optional[List[AliasScopedBoost]] = None,
    k: int = 50,
    session_id: Optional[int] = None,
) -> Tuple[List[CatalogHitV10], Dict[int, MatchProvenance]]:
    """V10 structured boost search.

    Supports:
    - entity_boosts: {entity_id, forms[], weight} — always active
    - alias_boosts_scoped: {collection_slug, alias_text, locked_entity_id?, weight}
      — only fires when scope includes the boost's collection_slug

    Returns:
    - List of CatalogHitV10 with full doc/page provenance
    - Dict of chunk_id -> MatchProvenance (which boost fired per chunk)

    Invariant I1: alias boosts are ignored when scope excludes venona/vassiliev.
    """
    entity_boosts = entity_boosts or []
    alias_boosts_scoped = alias_boosts_scoped or []

    # Build search filters
    if scope is None:
        scope = SearchFilters()

    scope_collections = scope.collection_slugs

    # Build boost terms + provenance map
    extra_terms, term_provenance = _build_boost_terms(
        entity_boosts, alias_boosts_scoped, scope_collections
    )

    # Build boosted query
    boosted_query = _build_boosted_query(query, extra_terms)

    # Execute hybrid search
    try:
        hits = hybrid_rrf(
            conn,
            boosted_query,
            filters=scope,
            k=k,
            expand_concordance=True,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("search_chunks_v10 failed: %s", e)
        return [], {}

    # Convert to CatalogHitV10
    catalog_hits = _hits_to_catalog_v10(conn, hits)

    # Compute provenance
    provenance = _compute_provenance(hits, term_provenance)

    return catalog_hits, provenance
