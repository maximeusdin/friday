"""
PEM-grounded search expansion and entity enumeration.

Thin layer over v10_pem_lane, v10_spans, v10_page_bridge. Provides:
- pem_expand_surfaces_for_query: scope-attested surfaces from PEM (no single-token → multi-token expansion)
- pem_enumerate_pages_for_entity: Pattern 3A — enumerate pages where entity is mentioned per PEM
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from retrieval.agent.v10_normalize import normalize_surface_for_lookup
from retrieval.agent.v10_page_bridge import chunks_for_pages, has_page_entity_mentions
from retrieval.agent.v10_types import ALIAS_SCOPED_COLLECTIONS
from retrieval.agent.v10_spans import resolve_surface_to_entity_ids
from retrieval.ops import SearchFilters, concordance_expand_terms

logger = logging.getLogger(__name__)

MAX_ENTITIES = 5
MAX_SURFACES = 25


def pem_expand_surfaces_for_query(
    conn,
    query_surface: str,
    collection_slugs: Optional[List[str]] = None,
    document_ids: Optional[List[int]] = None,
    *,
    max_surfaces: int = MAX_SURFACES,
) -> List[str]:
    """Expand query surface to PEM-attested surfaces within scope.

    Rule: single-token (no space) returns [query_surface] only — never expand
    to multi-token surfaces (e.g. "white" → "harry dexter white" is forbidden).

    When multi-token: resolve entity_ids, pull surfaces from PEM in scope,
    ranked by page frequency. Fallback to concordance_expand_terms when PEM missing.

    Args:
        conn: Database connection.
        query_surface: User query term or phrase.
        collection_slugs: Scope collections. When None, uses ALIAS_SCOPED_COLLECTIONS.
        document_ids: Optional document filter.
        max_surfaces: Cap on returned surfaces.

    Returns:
        List of surface strings (original first if matched, then expansions).
    """
    q = (query_surface or "").strip()
    if not q:
        return []

    norm_key = normalize_surface_for_lookup(q)
    if not norm_key:
        return [q]

    # Single-token: no multi-token expansion
    if " " not in norm_key:
        return [q]

    # PEM missing: fallback to concordance
    if not has_page_entity_mentions(conn):
        logger.warning("PEM missing; search expansion falling back to concordance_expand_terms")
        return concordance_expand_terms(conn, q, max_aliases_out=max_surfaces)

    colls = collection_slugs if collection_slugs else list(ALIAS_SCOPED_COLLECTIONS)
    if not colls:
        return [q]

    entity_ids = resolve_surface_to_entity_ids(conn, norm_key, scope_collections=colls, max_entities=MAX_ENTITIES)
    if not entity_ids:
        return [q]

    # Query PEM for surfaces attested in scope
    surfaces: List[Tuple[str, int]] = []  # (surface_norm, page_count)
    seen: set = set()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pem.surface_norm, COUNT(DISTINCT pem.page_id) AS cnt
                FROM page_entity_mentions pem
                WHERE pem.entity_id = ANY(%(entity_ids)s)
                  AND pem.collection_slug = ANY(%(colls)s)
                  AND (%(doc_ids)s IS NULL OR pem.document_id = ANY(%(doc_ids)s))
                GROUP BY pem.surface_norm
                ORDER BY cnt DESC, pem.surface_norm ASC
                LIMIT %(max)s
                """,
                {
                    "entity_ids": entity_ids,
                    "colls": colls,
                    "doc_ids": document_ids,
                    "max": max_surfaces,
                },
            )
            for row in cur.fetchall():
                s, cnt = row[0], row[1]
                if s and s not in seen:
                    seen.add(s)
                    surfaces.append((s, cnt))
    except Exception as e:
        logger.warning("PEM surfaces query failed: %s; falling back to concordance", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return concordance_expand_terms(conn, q, max_aliases_out=max_surfaces)

    if not surfaces:
        return [q]

    # Prefer original if it appears; otherwise prepend
    result = [s for s, _ in surfaces]
    if norm_key not in seen:
        result = [q] + result
    return result[:max_surfaces]


def pem_enumerate_pages_for_entity(
    conn,
    entity_id: int,
    filters: SearchFilters,
    *,
    max_pages: int = 500,
) -> List[Tuple[int, int, int, int, int]]:
    """Enumerate pages where entity is mentioned per PEM (Pattern 3A).

    Returns (collection_id, document_id, page_id, page_seq, pdf_page_number)
    for search_result_page_hits. Joins pages and documents for collection_id.

    Args:
        conn: Database connection.
        entity_id: Entity to enumerate.
        filters: SearchFilters with collection_slugs, document_ids.
        max_pages: Cap on returned pages.

    Returns:
        List of (collection_id, document_id, page_id, page_seq, pdf_page_number).
    """
    colls = filters.collection_slugs
    doc_ids = filters.document_ids
    if not colls:
        colls = list(ALIAS_SCOPED_COLLECTIONS)
    if not colls:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.collection_id, pem.document_id, pem.page_id,
                       p.page_seq, COALESCE(p.pdf_page_number, p.page_seq)::INT AS pdf_page_number
                FROM page_entity_mentions pem
                JOIN pages p ON p.id = pem.page_id
                JOIN documents d ON d.id = p.document_id
                WHERE pem.entity_id = %(eid)s
                  AND pem.collection_slug = ANY(%(colls)s)
                  AND (%(doc_ids)s IS NULL OR pem.document_id = ANY(%(doc_ids)s))
                GROUP BY d.collection_id, pem.document_id, pem.page_id, p.page_seq, p.pdf_page_number, pem.collection_slug
                ORDER BY pem.collection_slug, pem.document_id, p.page_seq
                LIMIT %(max)s
                """,
                {"eid": entity_id, "colls": colls, "doc_ids": doc_ids, "max": max_pages},
            )
            return cur.fetchall()
    except Exception as e:
        logger.warning("PEM enumerate pages failed for entity %d: %s", entity_id, e)
        try:
            conn.rollback()
        except Exception:
            pass
        return []
