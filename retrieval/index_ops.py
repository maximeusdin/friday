"""
Index Operations for Index Retrieval.

This module provides SQL builders and execution functions for querying
entity_mentions, date_mentions, and place indexes directly.

Unlike search operations (lexical/vector/hybrid), index operations:
- Query pre-built indexes (entity_mentions, date_mentions)
- Return deterministic, reproducible results
- Use canonical chronological ordering
- Support pagination for large result sets

All operations share the same infrastructure:
- Result sets (result_set_chunks)
- Match traces (result_set_match_traces)
- Cursor pagination
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Dict
from datetime import date

from retrieval.chronology import (
    CHRONOLOGY_V1_KEY,
    CHRONOLOGY_V1_ORDER_COMPACT,
    get_order_config,
    build_chronology_cte,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IndexHit:
    """
    A single hit from an index query.
    
    Represents one chunk returned from entity_mentions, date_mentions, etc.
    """
    chunk_id: int
    document_id: Optional[int] = None
    rank: int = 0
    date_key: Optional[date] = None
    page_seq: Optional[int] = None
    entity_ids: List[int] = field(default_factory=list)
    
    # Optional: mention-specific data
    mention_surface: Optional[str] = None
    mention_start_char: Optional[int] = None
    mention_end_char: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rank": self.rank,
            "date_key": self.date_key.isoformat() if self.date_key else None,
            "page_seq": self.page_seq,
            "entity_ids": self.entity_ids,
            "mention_surface": self.mention_surface,
            "mention_start_char": self.mention_start_char,
            "mention_end_char": self.mention_end_char,
        }


@dataclass
class IndexMetadata:
    """
    Metadata about an index query execution.
    
    Used for logging, tracing, and audit.
    """
    source: str = "entity_index"  # "entity_index" | "date_index" | "place_index"
    primitive_type: str = ""
    order_by: str = CHRONOLOGY_V1_KEY
    order_sql: str = CHRONOLOGY_V1_ORDER_COMPACT
    total_hits: int = 0
    scope_applied: bool = False
    
    # Deduplication policy - documents how multiple mentions in same chunk are handled
    dedupe_policy: str = "chunk_unique"  # "chunk_unique" = one row per chunk
    span_policy: str = "first_in_chunk"  # "first_in_chunk" = return first mention span
    
    # Date-specific
    time_basis: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    
    # Place-specific
    geo_basis: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for audit logging."""
        d = {
            "source": self.source,
            "primitive_type": self.primitive_type,
            "order_by": self.order_by,
            "order_sql": self.order_sql,
            "total_hits": self.total_hits,
            "scope_applied": self.scope_applied,
            "dedupe_policy": self.dedupe_policy,
            "span_policy": self.span_policy,
        }
        if self.time_basis:
            d["time_basis"] = self.time_basis
        if self.date_range:
            d["date_range"] = self.date_range
        if self.geo_basis:
            d["geo_basis"] = self.geo_basis
        return d


# =============================================================================
# Entity Index Operations
# =============================================================================

def first_mention(
    conn,
    entity_id: int,
    *,
    scope_sql: Optional[str] = None,
    scope_params: Optional[Dict[str, Any]] = None,
    order_by: str = "chronological",
) -> Tuple[Optional[IndexHit], IndexMetadata]:
    """
    Find the chronologically first chunk mentioning an entity.
    
    Queries entity_mentions and returns the earliest chunk by canonical ordering.
    
    Args:
        conn: Database connection
        entity_id: Entity ID to find first mention of
        scope_sql: Optional WHERE clause for scope filtering
        scope_params: Parameters for scope SQL
        order_by: "chronological" or "document_order"
        
    Returns:
        (IndexHit or None, IndexMetadata)
    """
    order_config = get_order_config(order_by)
    
    # Build parameters
    params: Dict[str, Any] = {
        "entity_id": entity_id,
    }
    if scope_params:
        params.update(scope_params)
    
    # Build scope clause
    scope_clause = ""
    if scope_sql:
        scope_clause = f"AND ({scope_sql})"
    
    sql = f"""
    WITH ranked_mentions AS (
        SELECT DISTINCT ON (em.chunk_id)
            em.chunk_id,
            em.document_id,
            em.surface,
            em.start_char,
            em.end_char,
            cm.date_min,
            p.page_seq,
            ROW_NUMBER() OVER (
                ORDER BY 
                    COALESCE(cm.date_min, '9999-12-31'::date) ASC,
                    COALESCE(p.page_seq, 2147483647) ASC,
                    em.chunk_id ASC
            ) as rank
        FROM entity_mentions em
        JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
        LEFT JOIN chunk_pages cp ON cp.chunk_id = em.chunk_id AND cp.span_order = 1
        LEFT JOIN pages p ON p.id = cp.page_id
        WHERE em.entity_id = %(entity_id)s
        {scope_clause}
    )
    SELECT chunk_id, document_id, surface, start_char, end_char, date_min, page_seq, rank
    FROM ranked_mentions
    WHERE rank = 1
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    
    metadata = IndexMetadata(
        source="entity_index",
        primitive_type="FIRST_MENTION",
        order_by=order_config.key,
        order_sql=CHRONOLOGY_V1_ORDER_COMPACT,
        total_hits=1 if row else 0,
        scope_applied=bool(scope_sql),
    )
    
    if not row:
        return None, metadata
    
    chunk_id, document_id, surface, start_char, end_char, date_min, page_seq, rank = row
    
    hit = IndexHit(
        chunk_id=chunk_id,
        document_id=document_id,
        rank=rank,
        date_key=date_min,
        page_seq=page_seq,
        entity_ids=[entity_id],
        mention_surface=surface,
        mention_start_char=start_char,
        mention_end_char=end_char,
    )
    
    return hit, metadata


def first_co_mention(
    conn,
    entity_ids: List[int],
    *,
    window: str = "chunk",
    scope_sql: Optional[str] = None,
    scope_params: Optional[Dict[str, Any]] = None,
    order_by: str = "chronological",
) -> Tuple[Optional[IndexHit], IndexMetadata]:
    """
    Find the first chunk where all specified entities co-occur.
    
    Args:
        conn: Database connection
        entity_ids: List of entity IDs that must all appear in same chunk
        window: "chunk" (same chunk) - "document" not yet supported
        scope_sql: Optional WHERE clause for scope filtering
        scope_params: Parameters for scope SQL
        order_by: "chronological" or "document_order"
        
    Returns:
        (IndexHit or None, IndexMetadata)
    """
    if window != "chunk":
        raise ValueError(f"window='{window}' not supported in v1, only 'chunk'")
    
    order_config = get_order_config(order_by)
    entity_count = len(entity_ids)
    
    params: Dict[str, Any] = {
        "entity_ids": entity_ids,
        "entity_count": entity_count,
    }
    if scope_params:
        params.update(scope_params)
    
    scope_clause = ""
    if scope_sql:
        scope_clause = f"WHERE ({scope_sql})"
    
    sql = f"""
    WITH entity_chunks AS (
        -- Find chunks containing ALL required entities
        SELECT em.chunk_id, em.document_id
        FROM entity_mentions em
        WHERE em.entity_id = ANY(%(entity_ids)s)
        GROUP BY em.chunk_id, em.document_id
        HAVING COUNT(DISTINCT em.entity_id) = %(entity_count)s
    ),
    ranked AS (
        SELECT 
            ec.chunk_id,
            ec.document_id,
            cm.date_min,
            p.page_seq,
            ROW_NUMBER() OVER (
                ORDER BY 
                    COALESCE(cm.date_min, '9999-12-31'::date) ASC,
                    COALESCE(p.page_seq, 2147483647) ASC,
                    ec.chunk_id ASC
            ) as rank
        FROM entity_chunks ec
        JOIN chunk_metadata cm ON cm.chunk_id = ec.chunk_id
        LEFT JOIN chunk_pages cp ON cp.chunk_id = ec.chunk_id AND cp.span_order = 1
        LEFT JOIN pages p ON p.id = cp.page_id
        {scope_clause}
    )
    SELECT chunk_id, document_id, date_min, page_seq, rank
    FROM ranked
    WHERE rank = 1
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    
    metadata = IndexMetadata(
        source="entity_index",
        primitive_type="FIRST_CO_MENTION",
        order_by=order_config.key,
        order_sql=CHRONOLOGY_V1_ORDER_COMPACT,
        total_hits=1 if row else 0,
        scope_applied=bool(scope_sql),
    )
    
    if not row:
        return None, metadata
    
    chunk_id, document_id, date_min, page_seq, rank = row
    
    hit = IndexHit(
        chunk_id=chunk_id,
        document_id=document_id,
        rank=rank,
        date_key=date_min,
        page_seq=page_seq,
        entity_ids=entity_ids,
    )
    
    return hit, metadata


def mentions_paginated(
    conn,
    entity_id: int,
    *,
    scope_sql: Optional[str] = None,
    scope_params: Optional[Dict[str, Any]] = None,
    order_by: str = "chronological",
    limit: int = 100,
    after_rank: Optional[int] = None,
) -> Tuple[List[IndexHit], IndexMetadata]:
    """
    Find all chunks mentioning an entity, paginated.
    
    Args:
        conn: Database connection
        entity_id: Entity ID to find mentions of
        scope_sql: Optional WHERE clause for scope filtering
        scope_params: Parameters for scope SQL
        order_by: "chronological" or "document_order"
        limit: Maximum results to return (for pagination)
        after_rank: Cursor - return results with rank > after_rank
        
    Returns:
        (List[IndexHit], IndexMetadata)
    """
    order_config = get_order_config(order_by)
    
    params: Dict[str, Any] = {
        "entity_id": entity_id,
        "limit": limit,
    }
    if scope_params:
        params.update(scope_params)
    
    scope_clause = ""
    if scope_sql:
        scope_clause = f"AND ({scope_sql})"
    
    # Cursor pagination
    cursor_clause = ""
    if after_rank is not None:
        cursor_clause = "WHERE rank > %(after_rank)s"
        params["after_rank"] = after_rank
    
    # First get total count (for metadata)
    count_sql = f"""
    SELECT COUNT(DISTINCT em.chunk_id)
    FROM entity_mentions em
    WHERE em.entity_id = %(entity_id)s
    {scope_clause}
    """
    
    with conn.cursor() as cur:
        cur.execute(count_sql, {"entity_id": entity_id, **(scope_params or {})})
        total_count = cur.fetchone()[0]
    
    # Then get paginated results
    sql = f"""
    WITH ranked_mentions AS (
        SELECT DISTINCT ON (em.chunk_id)
            em.chunk_id,
            em.document_id,
            em.surface,
            em.start_char,
            em.end_char,
            cm.date_min,
            p.page_seq,
            ROW_NUMBER() OVER (
                ORDER BY 
                    COALESCE(cm.date_min, '9999-12-31'::date) ASC,
                    COALESCE(p.page_seq, 2147483647) ASC,
                    em.chunk_id ASC
            ) as rank
        FROM entity_mentions em
        JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
        LEFT JOIN chunk_pages cp ON cp.chunk_id = em.chunk_id AND cp.span_order = 1
        LEFT JOIN pages p ON p.id = cp.page_id
        WHERE em.entity_id = %(entity_id)s
        {scope_clause}
    )
    SELECT chunk_id, document_id, surface, start_char, end_char, date_min, page_seq, rank
    FROM ranked_mentions
    {cursor_clause}
    ORDER BY rank
    LIMIT %(limit)s
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    metadata = IndexMetadata(
        source="entity_index",
        primitive_type="MENTIONS",
        order_by=order_config.key,
        order_sql=CHRONOLOGY_V1_ORDER_COMPACT,
        total_hits=total_count,
        scope_applied=bool(scope_sql),
    )
    
    hits = []
    for row in rows:
        chunk_id, document_id, surface, start_char, end_char, date_min, page_seq, rank = row
        hits.append(IndexHit(
            chunk_id=chunk_id,
            document_id=document_id,
            rank=rank,
            date_key=date_min,
            page_seq=page_seq,
            entity_ids=[entity_id],
            mention_surface=surface,
            mention_start_char=start_char,
            mention_end_char=end_char,
        ))
    
    return hits, metadata


# =============================================================================
# Date Index Operations
# =============================================================================

def date_range_filter(
    conn,
    date_start: Optional[date],
    date_end: Optional[date],
    *,
    time_basis: str = "mentioned_date",
    scope_sql: Optional[str] = None,
    scope_params: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    after_rank: Optional[int] = None,
) -> Tuple[List[IndexHit], IndexMetadata]:
    """
    Find chunks with date mentions in the specified range.
    
    Args:
        conn: Database connection
        date_start: Start of date range (inclusive), or None for open start
        date_end: End of date range (inclusive), or None for open end
        time_basis: "mentioned_date" (uses date_mentions) or "document_date" (uses chunk_metadata)
        scope_sql: Optional WHERE clause for scope filtering
        scope_params: Parameters for scope SQL
        limit: Maximum results to return
        after_rank: Cursor for pagination
        
    Returns:
        (List[IndexHit], IndexMetadata)
    """
    params: Dict[str, Any] = {"limit": limit}
    if scope_params:
        params.update(scope_params)
    
    date_conditions = []
    if date_start:
        date_conditions.append("dm.date_start >= %(date_start)s")
        params["date_start"] = date_start
    if date_end:
        date_conditions.append("dm.date_end <= %(date_end)s")
        params["date_end"] = date_end
    
    date_clause = " AND ".join(date_conditions) if date_conditions else "TRUE"
    
    scope_clause = ""
    if scope_sql:
        scope_clause = f"AND ({scope_sql})"
    
    cursor_clause = ""
    if after_rank is not None:
        cursor_clause = "WHERE rank > %(after_rank)s"
        params["after_rank"] = after_rank
    
    if time_basis == "mentioned_date":
        # Query date_mentions table
        sql = f"""
        WITH dated_chunks AS (
            SELECT DISTINCT ON (dm.chunk_id)
                dm.chunk_id,
                dm.document_id,
                dm.date_start as date_key,
                dm.surface as date_surface,
                dm.start_char,
                dm.end_char,
                ROW_NUMBER() OVER (
                    ORDER BY dm.date_start ASC, dm.chunk_id ASC
                ) as rank
            FROM date_mentions dm
            WHERE {date_clause}
            {scope_clause}
        )
        SELECT chunk_id, document_id, date_key, date_surface, start_char, end_char, rank
        FROM dated_chunks
        {cursor_clause}
        ORDER BY rank
        LIMIT %(limit)s
        """
    else:
        # Use chunk_metadata.date_min
        sql = f"""
        WITH dated_chunks AS (
            SELECT DISTINCT ON (cm.chunk_id)
                cm.chunk_id,
                cm.document_id,
                cm.date_min as date_key,
                NULL as date_surface,
                NULL as start_char,
                NULL as end_char,
                ROW_NUMBER() OVER (
                    ORDER BY cm.date_min ASC, cm.chunk_id ASC
                ) as rank
            FROM chunk_metadata cm
            WHERE cm.date_min IS NOT NULL
                AND cm.date_min >= %(date_start)s
                AND cm.date_min <= %(date_end)s
            {scope_clause}
        )
        SELECT chunk_id, document_id, date_key, date_surface, start_char, end_char, rank
        FROM dated_chunks
        {cursor_clause}
        ORDER BY rank
        LIMIT %(limit)s
        """
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    # Get total count
    if time_basis == "mentioned_date":
        count_sql = f"""
        SELECT COUNT(DISTINCT dm.chunk_id)
        FROM date_mentions dm
        WHERE {date_clause}
        {scope_clause}
        """
    else:
        count_sql = f"""
        SELECT COUNT(DISTINCT cm.chunk_id)
        FROM chunk_metadata cm
        WHERE cm.date_min IS NOT NULL
            AND cm.date_min >= %(date_start)s
            AND cm.date_min <= %(date_end)s
        {scope_clause}
        """
    
    with conn.cursor() as cur:
        cur.execute(count_sql, params)
        total_count = cur.fetchone()[0]
    
    date_range_str = {}
    if date_start:
        date_range_str["start"] = date_start.isoformat() if isinstance(date_start, date) else str(date_start)
    if date_end:
        date_range_str["end"] = date_end.isoformat() if isinstance(date_end, date) else str(date_end)
    
    metadata = IndexMetadata(
        source="date_index",
        primitive_type="DATE_RANGE_FILTER",
        order_by="date_asc",
        order_sql="date_start ASC, chunk_id ASC",
        total_hits=total_count,
        scope_applied=bool(scope_sql),
        time_basis=time_basis,
        date_range=date_range_str,
    )
    
    hits = []
    for row in rows:
        chunk_id, document_id, date_key, date_surface, start_char, end_char, rank = row
        hits.append(IndexHit(
            chunk_id=chunk_id,
            document_id=document_id,
            rank=rank,
            date_key=date_key,
            entity_ids=[],
            mention_surface=date_surface,
            mention_start_char=start_char,
            mention_end_char=end_char,
        ))
    
    return hits, metadata


def first_date_mention(
    conn,
    entity_id: int,
    *,
    time_basis: str = "mentioned_date",
    scope_sql: Optional[str] = None,
    scope_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[IndexHit], IndexMetadata]:
    """
    Find the earliest dated chunk where an entity is mentioned.
    
    Args:
        conn: Database connection
        entity_id: Entity to find earliest dated mention for
        time_basis: "mentioned_date" or "document_date"
        scope_sql: Optional scope filters
        scope_params: Parameters for scope SQL
        
    Returns:
        (IndexHit or None, IndexMetadata)
    """
    params: Dict[str, Any] = {"entity_id": entity_id}
    if scope_params:
        params.update(scope_params)
    
    scope_clause = ""
    if scope_sql:
        scope_clause = f"AND ({scope_sql})"
    
    if time_basis == "mentioned_date":
        # Find earliest mentioned date in chunks where entity appears
        sql = f"""
        WITH entity_dated_chunks AS (
            SELECT DISTINCT
                em.chunk_id,
                em.document_id,
                dm.date_start,
                dm.surface as date_surface,
                dm.start_char,
                dm.end_char
            FROM entity_mentions em
            JOIN date_mentions dm ON dm.chunk_id = em.chunk_id
            WHERE em.entity_id = %(entity_id)s
                AND dm.date_start IS NOT NULL
            {scope_clause}
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (ORDER BY date_start ASC, chunk_id ASC) as rank
            FROM entity_dated_chunks
        )
        SELECT chunk_id, document_id, date_start, date_surface, start_char, end_char, rank
        FROM ranked WHERE rank = 1
        """
    else:
        # Find earliest by document date (chunk_metadata.date_min)
        sql = f"""
        WITH entity_chunks AS (
            SELECT DISTINCT em.chunk_id, em.document_id
            FROM entity_mentions em
            WHERE em.entity_id = %(entity_id)s
            {scope_clause}
        ),
        ranked AS (
            SELECT 
                ec.chunk_id,
                ec.document_id,
                cm.date_min as date_start,
                NULL as date_surface,
                NULL as start_char,
                NULL as end_char,
                ROW_NUMBER() OVER (ORDER BY COALESCE(cm.date_min, '9999-12-31'::date) ASC, ec.chunk_id ASC) as rank
            FROM entity_chunks ec
            JOIN chunk_metadata cm ON cm.chunk_id = ec.chunk_id
        )
        SELECT chunk_id, document_id, date_start, date_surface, start_char, end_char, rank
        FROM ranked WHERE rank = 1
        """
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    
    metadata = IndexMetadata(
        source="date_index",
        primitive_type="FIRST_DATE_MENTION",
        order_by="date_asc",
        order_sql="COALESCE(date_start, '9999-12-31') ASC, chunk_id ASC",
        total_hits=1 if row else 0,
        scope_applied=bool(scope_sql),
        time_basis=time_basis,
    )
    
    if not row:
        return None, metadata
    
    chunk_id, document_id, date_start, date_surface, start_char, end_char, rank = row
    
    hit = IndexHit(
        chunk_id=chunk_id,
        document_id=document_id,
        rank=rank,
        date_key=date_start,
        entity_ids=[entity_id],
        mention_surface=date_surface,
        mention_start_char=start_char,
        mention_end_char=end_char,
    )
    
    return hit, metadata


# =============================================================================
# Place Index Operations
# =============================================================================

def place_mentions(
    conn,
    place_entity_id: int,
    *,
    scope_sql: Optional[str] = None,
    scope_params: Optional[Dict[str, Any]] = None,
    order_by: str = "chronological",
    limit: int = 100,
    after_rank: Optional[int] = None,
) -> Tuple[List[IndexHit], IndexMetadata]:
    """
    Find all chunks mentioning a place entity.
    
    This is essentially the same as mentions_paginated but typed for places.
    Place entities have entity_type='place' in the entities table.
    
    Args:
        conn: Database connection
        place_entity_id: Place entity ID
        scope_sql: Optional scope filters
        scope_params: Parameters for scope SQL
        order_by: "chronological" or "document_order"
        limit: Maximum results
        after_rank: Cursor for pagination
        
    Returns:
        (List[IndexHit], IndexMetadata)
    """
    # Delegate to mentions_paginated - same logic
    hits, metadata = mentions_paginated(
        conn,
        place_entity_id,
        scope_sql=scope_sql,
        scope_params=scope_params,
        order_by=order_by,
        limit=limit,
        after_rank=after_rank,
    )
    
    # Update metadata to reflect place source
    metadata.source = "place_index"
    metadata.primitive_type = "PLACE_MENTIONS"
    metadata.geo_basis = "resolved_place_entities"
    
    return hits, metadata


def related_places(
    conn,
    entity_id: int,
    *,
    window: str = "chunk",
    top_n: int = 20,
    scope_chunk_ids: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], IndexMetadata]:
    """
    Find places co-mentioned with an entity.
    
    Returns aggregated data (not individual chunks) - list of places
    with their mention counts.
    
    Args:
        conn: Database connection
        entity_id: Target entity to find related places for
        window: "chunk" or "document"
        top_n: Maximum places to return
        scope_chunk_ids: Optional list of chunk_ids to scope to (e.g., from result_set)
        
    Returns:
        (List of place dicts, IndexMetadata)
    """
    params: Dict[str, Any] = {
        "entity_id": entity_id,
        "top_n": top_n,
    }
    
    scope_clause = ""
    if scope_chunk_ids:
        scope_clause = "AND em.chunk_id = ANY(%(scope_chunk_ids)s)"
        params["scope_chunk_ids"] = scope_chunk_ids
    
    # Note: place_country column may not exist yet (added in migration 0045)
    # Use a graceful fallback when the column doesn't exist
    if window == "chunk":
        sql = f"""
        WITH target_chunks AS (
            SELECT DISTINCT chunk_id
            FROM entity_mentions
            WHERE entity_id = %(entity_id)s
            {scope_clause}
        ),
        place_co_mentions AS (
            SELECT 
                em.entity_id as place_entity_id,
                e.canonical_name,
                NULL as place_country,
                COUNT(DISTINCT em.chunk_id) as mention_count
            FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            WHERE em.chunk_id IN (SELECT chunk_id FROM target_chunks)
                AND e.entity_type = 'place'
                AND em.entity_id != %(entity_id)s
            GROUP BY em.entity_id, e.canonical_name
            ORDER BY mention_count DESC
            LIMIT %(top_n)s
        )
        SELECT place_entity_id, canonical_name, place_country, mention_count
        FROM place_co_mentions
        """
    else:
        # Document window
        sql = f"""
        WITH target_docs AS (
            SELECT DISTINCT document_id
            FROM entity_mentions
            WHERE entity_id = %(entity_id)s
            {scope_clause}
        ),
        place_co_mentions AS (
            SELECT 
                em.entity_id as place_entity_id,
                e.canonical_name,
                NULL as place_country,
                COUNT(DISTINCT em.document_id) as mention_count
            FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            WHERE em.document_id IN (SELECT document_id FROM target_docs)
                AND e.entity_type = 'place'
                AND em.entity_id != %(entity_id)s
            GROUP BY em.entity_id, e.canonical_name
            ORDER BY mention_count DESC
            LIMIT %(top_n)s
        )
        SELECT place_entity_id, canonical_name, place_country, mention_count
        FROM place_co_mentions
        """
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    metadata = IndexMetadata(
        source="place_index",
        primitive_type="RELATED_PLACES",
        total_hits=len(rows),
        geo_basis="resolved_place_entities",
    )
    
    results = []
    for row in rows:
        place_entity_id, canonical_name, place_country, mention_count = row
        results.append({
            "place_entity_id": place_entity_id,
            "canonical_name": canonical_name,
            "place_country": place_country,
            "mention_count": mention_count,
        })
    
    return results, metadata
