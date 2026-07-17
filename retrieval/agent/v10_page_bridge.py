"""
V10.2 Page→Chunk Bridge — deterministic, deduped, capped, entity-preferring.

Single helper used by all tools that need to convert page_ids → chunk_ids.
Ensures consistent ordering and deterministic sampling across callers.

Contract:
  chunks_for_pages(conn, page_ids, max_chunks_per_page=10,
                   prefer_entity_id=None, prefer_surface_norm=None) -> List[int]
    - Joins chunk_pages ordered by span_order
    - Deterministic cap per page
    - When prefer_entity_id is set: chunks with entity_mentions for that
      entity are promoted before the span_order cap is applied
    - Deduplicates chunk_ids across pages
    - Stable ordering: (page_id ASC, span_order ASC)

  Optional materialized view (speed knob for heavy usage):
    CREATE MATERIALIZED VIEW surface_entity_stats AS
    SELECT collection_slug, surface_norm, entity_id,
           COUNT(DISTINCT page_id) AS count_pages,
           (SELECT value FROM app_kv WHERE key = 'page_entity_mentions_revision') AS revision
    FROM page_entity_mentions
    GROUP BY collection_slug, surface_norm, entity_id;
    -- Rebuild alongside PEM population. Not required for correctness.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


def chunks_for_pages(
    conn,
    page_ids: Sequence[int],
    *,
    max_chunks_per_page: int = 10,
    prefer_entity_id: Optional[int] = None,
    prefer_surface_norm: Optional[str] = None,
) -> List[int]:
    """Convert page_ids → chunk_ids via chunk_pages.

    Args:
        conn: Database connection (psycopg2).
        page_ids: Sequence of page IDs to resolve.
        max_chunks_per_page: Deterministic cap per page (keeps the first N
            by span_order for each page).
        prefer_entity_id: When set, chunks that have entity_mentions for
            this entity_id are promoted to the front of each page's list
            before the per-page cap is applied. Cheap JOIN — O(page_ids).
        prefer_surface_norm: When set alongside prefer_entity_id, further
            prefers chunks where the surface appears (best-effort).

    Returns:
        Deduplicated list of chunk_ids, in deterministic order
        (ordered by first appearance across pages sorted by page_id).
    """
    if not page_ids:
        return []

    # When prefer_entity_id is set, find "supporting" chunk_ids first
    supporting_chunks: Set[int] = set()
    if prefer_entity_id is not None:
        try:
            with conn.cursor() as cur:
                # Find chunk_ids on these pages that mention the entity
                cur.execute("""
                    SELECT DISTINCT cp.chunk_id
                    FROM chunk_pages cp
                    JOIN entity_mentions em ON em.chunk_id = cp.chunk_id
                    WHERE cp.page_id = ANY(%s)
                      AND em.entity_id = %s
                """, (list(page_ids), prefer_entity_id))
                supporting_chunks = {r[0] for r in cur.fetchall()}
        except Exception as e:
            logger.debug("chunks_for_pages supporting-chunk query failed: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    # Fetch all (page_id, chunk_id) pairs ordered deterministically
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cp.page_id, cp.chunk_id
                FROM chunk_pages cp
                WHERE cp.page_id = ANY(%s)
                ORDER BY cp.page_id ASC, cp.span_order ASC, cp.chunk_id ASC
            """, (list(page_ids),))
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("chunks_for_pages query failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return []

    if supporting_chunks:
        # Reorder: for each page, put supporting chunks first, then others
        from collections import defaultdict
        page_chunks: Dict[int, List[int]] = defaultdict(list)
        for page_id, chunk_id in rows:
            page_chunks[page_id].append(chunk_id)

        page_counts: Dict[int, int] = {}
        seen: set = set()
        result: List[int] = []

        for page_id in sorted(page_chunks.keys()):
            chunks = page_chunks[page_id]
            # Stable partition: supporting first, then rest (preserving span_order within each)
            preferred = [c for c in chunks if c in supporting_chunks]
            rest = [c for c in chunks if c not in supporting_chunks]
            ordered = preferred + rest

            count = 0
            for chunk_id in ordered:
                if count >= max_chunks_per_page:
                    break
                count += 1
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    result.append(chunk_id)
            page_counts[page_id] = count

        return result

    # Standard path (no entity preference)
    page_counts: Dict[int, int] = {}
    seen: set = set()
    result: List[int] = []

    for page_id, chunk_id in rows:
        count = page_counts.get(page_id, 0)
        if count >= max_chunks_per_page:
            continue
        page_counts[page_id] = count + 1
        if chunk_id not in seen:
            seen.add(chunk_id)
            result.append(chunk_id)

    return result


def pages_to_chunks_map(
    conn,
    page_ids: Sequence[int],
    *,
    max_chunks_per_page: int = 10,
    prefer_entity_id: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Convert page_ids → {page_id: [chunk_ids]} via chunk_pages.

    Same semantics as chunks_for_pages but preserves per-page grouping
    for callers that need to know which chunks came from which page.

    When prefer_entity_id is set, chunks with entity_mentions for that
    entity are promoted to the front of each page's list.
    """
    if not page_ids:
        return {}

    # When prefer_entity_id is set, find "supporting" chunk_ids first
    supporting_chunks: Set[int] = set()
    if prefer_entity_id is not None:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT cp.chunk_id
                    FROM chunk_pages cp
                    JOIN entity_mentions em ON em.chunk_id = cp.chunk_id
                    WHERE cp.page_id = ANY(%s)
                      AND em.entity_id = %s
                """, (list(page_ids), prefer_entity_id))
                supporting_chunks = {r[0] for r in cur.fetchall()}
        except Exception as e:
            logger.debug("pages_to_chunks_map supporting-chunk query failed: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cp.page_id, cp.chunk_id
                FROM chunk_pages cp
                WHERE cp.page_id = ANY(%s)
                ORDER BY cp.page_id ASC, cp.span_order ASC, cp.chunk_id ASC
            """, (list(page_ids),))
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("pages_to_chunks_map query failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return {}

    # Group by page, then reorder if entity preference is active
    from collections import defaultdict
    raw_groups: Dict[int, List[int]] = defaultdict(list)
    for page_id, chunk_id in rows:
        raw_groups[page_id].append(chunk_id)

    result: Dict[int, List[int]] = {}
    for page_id in sorted(raw_groups.keys()):
        chunks = raw_groups[page_id]
        if supporting_chunks:
            preferred = [c for c in chunks if c in supporting_chunks]
            rest = [c for c in chunks if c not in supporting_chunks]
            chunks = preferred + rest
        # Cap and deduplicate
        seen: set = set()
        capped: List[int] = []
        for chunk_id in chunks:
            if len(capped) >= max_chunks_per_page:
                break
            if chunk_id not in seen:
                seen.add(chunk_id)
                capped.append(chunk_id)
        result[page_id] = capped

    return result


def get_index_revision(conn) -> str:
    """Read current page_entity_mentions revision from app_kv.

    Returns '0' if not yet set or table doesn't exist.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM app_kv WHERE key = 'page_entity_mentions_revision'"
            )
            row = cur.fetchone()
            return row[0] if row else "0"
    except Exception as e:
        logger.debug("get_index_revision failed (app_kv may not exist): %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return "0"


def set_index_revision(conn, revision: str) -> None:
    """Update the page_entity_mentions revision in app_kv."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO app_kv (key, value)
                VALUES ('page_entity_mentions_revision', %s)
                ON CONFLICT (key) DO UPDATE SET value = %s
            """, (revision, revision))
            conn.commit()
    except Exception as e:
        logger.warning("set_index_revision failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass


def has_page_entity_mentions(conn) -> bool:
    """Check if page_entity_mentions table exists and has any rows.

    Used by tools to decide whether to use the new index or fall back.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'page_entity_mentions'
                )
            """)
            exists = cur.fetchone()[0]
            if not exists:
                return False
            cur.execute("SELECT 1 FROM page_entity_mentions LIMIT 1")
            return cur.fetchone() is not None
    except Exception as e:
        logger.debug("has_page_entity_mentions check failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return False
