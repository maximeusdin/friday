"""
Canonical chronology definitions for Index Retrieval.

This module defines the standard ordering for "first" and "earliest" queries.
All index operations use these definitions to ensure deterministic, reproducible results.

Chronology Key (v1):
1. chunk_metadata.date_min (extracted dates) - NULLs sort last
2. pages.page_seq (document-internal order) - NULLs sort last  
3. chunks.id (deterministic tie-break)

This ordering ensures:
- Chunks with known dates come first, ordered by date
- Within same date, chunks are ordered by their position in the document
- Final tie-break by chunk_id ensures determinism
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import date


# =============================================================================
# Chronology Version Definitions
# =============================================================================

CHRONOLOGY_V1_KEY = "chronological_v1"

# SQL ORDER BY clause for chronological ordering
# Uses COALESCE to put NULLs last (sentinel values)
CHRONOLOGY_V1_ORDER = """
    COALESCE(cm.date_min, '9999-12-31'::date) ASC,
    COALESCE(p.page_seq, 2147483647) ASC,
    c.id ASC
"""

# Compact version for logging
CHRONOLOGY_V1_ORDER_COMPACT = "date_min ASC NULLS LAST, page_seq ASC NULLS LAST, chunk_id ASC"

# SQL for joining chronology tables
CHRONOLOGY_V1_JOIN = """
    LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
    LEFT JOIN chunk_pages cp ON cp.chunk_id = c.id AND cp.span_order = 1
    LEFT JOIN pages p ON p.id = cp.page_id
"""

# Alternative: using entity_mentions as base (em.chunk_id)
CHRONOLOGY_V1_JOIN_FROM_MENTIONS = """
    JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
    LEFT JOIN chunk_pages cp ON cp.chunk_id = em.chunk_id AND cp.span_order = 1
    LEFT JOIN pages p ON p.id = cp.page_id
"""


# =============================================================================
# Chronology Configuration
# =============================================================================

@dataclass(frozen=True)
class ChronologyConfig:
    """
    Configuration for chronological ordering.
    
    Attributes:
        key: Identifier for this chronology version
        order_sql: SQL ORDER BY clause
        null_date_sentinel: Value used for NULL dates (sorts last)
        null_page_sentinel: Value used for NULL page_seq (sorts last)
    """
    key: str
    order_sql: str
    null_date_sentinel: str = "9999-12-31"
    null_page_sentinel: int = 2147483647  # Max INT
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "key": self.key,
            "order_sql": self.order_sql.strip().replace("\n", " ").replace("  ", " "),
            "null_date_sentinel": self.null_date_sentinel,
            "null_page_sentinel": self.null_page_sentinel,
        }


# Default chronology configuration
DEFAULT_CHRONOLOGY = ChronologyConfig(
    key=CHRONOLOGY_V1_KEY,
    order_sql=CHRONOLOGY_V1_ORDER,
)


# =============================================================================
# Ordering Key Functions
# =============================================================================

def compute_chronology_key(
    date_min: Optional[date],
    page_seq: Optional[int],
    chunk_id: int,
) -> Tuple:
    """
    Compute a sortable chronology key tuple.
    
    Returns a tuple that can be used with Python's sort/sorted functions
    to achieve the same ordering as the SQL ORDER BY clause.
    
    Args:
        date_min: Earliest date in chunk (from chunk_metadata)
        page_seq: Page sequence number (from pages table)
        chunk_id: Chunk ID for tie-breaking
        
    Returns:
        Tuple for sorting: (date_key, page_key, chunk_id)
        - date_key: date or date.max for NULLs
        - page_key: page_seq or maxint for NULLs
        - chunk_id: for deterministic tie-break
    """
    date_key = date_min if date_min is not None else date.max
    page_key = page_seq if page_seq is not None else 2147483647
    return (date_key, page_key, chunk_id)


def chronology_sort_key(item: dict) -> Tuple:
    """
    Sort key function for dictionaries with chronology fields.
    
    Expects dict with keys: 'date_min', 'page_seq', 'chunk_id'
    
    Usage:
        sorted(items, key=chronology_sort_key)
    """
    return compute_chronology_key(
        item.get('date_min'),
        item.get('page_seq'),
        item['chunk_id'],
    )


# =============================================================================
# SQL Builders
# =============================================================================

def build_chronology_cte(
    base_table: str = "entity_mentions",
    base_alias: str = "em",
    chunk_id_column: str = "chunk_id",
    additional_columns: str = "",
    where_clause: str = "",
) -> str:
    """
    Build a CTE that adds chronology ordering to a base query.
    
    Args:
        base_table: Table to start from (e.g., "entity_mentions")
        base_alias: Alias for base table
        chunk_id_column: Column name for chunk_id in base table
        additional_columns: Extra columns to include (comma-prefixed)
        where_clause: WHERE conditions (without WHERE keyword)
        
    Returns:
        SQL CTE string
        
    Example:
        >>> build_chronology_cte(
        ...     base_table="entity_mentions",
        ...     where_clause="em.entity_id = %(entity_id)s"
        ... )
    """
    where = f"WHERE {where_clause}" if where_clause else ""
    
    return f"""
WITH chronological AS (
    SELECT DISTINCT ON ({base_alias}.{chunk_id_column})
        {base_alias}.{chunk_id_column} as chunk_id,
        {base_alias}.document_id,
        cm.date_min,
        p.page_seq{additional_columns},
        ROW_NUMBER() OVER (
            ORDER BY 
                COALESCE(cm.date_min, '9999-12-31'::date) ASC,
                COALESCE(p.page_seq, 2147483647) ASC,
                {base_alias}.{chunk_id_column} ASC
        ) as rank
    FROM {base_table} {base_alias}
    JOIN chunk_metadata cm ON cm.chunk_id = {base_alias}.{chunk_id_column}
    LEFT JOIN chunk_pages cp ON cp.chunk_id = {base_alias}.{chunk_id_column} AND cp.span_order = 1
    LEFT JOIN pages p ON p.id = cp.page_id
    {where}
)
"""


def get_chronology_order_by(
    config: Optional[ChronologyConfig] = None,
    table_alias: str = "",
) -> str:
    """
    Get the ORDER BY clause for chronological ordering.
    
    Args:
        config: Chronology configuration (uses default if None)
        table_alias: Optional table alias prefix (e.g., "t.")
        
    Returns:
        ORDER BY clause string
    """
    cfg = config or DEFAULT_CHRONOLOGY
    
    if table_alias:
        # Replace generic column references with aliased versions
        order = cfg.order_sql.replace("cm.date_min", f"{table_alias}date_min")
        order = order.replace("p.page_seq", f"{table_alias}page_seq")
        order = order.replace("c.id", f"{table_alias}chunk_id")
        return order
    
    return cfg.order_sql


# =============================================================================
# Validation
# =============================================================================

def validate_order_by(order_by: str) -> str:
    """
    Validate and normalize order_by parameter.
    
    Args:
        order_by: Order specification ("chronological" or "document_order")
        
    Returns:
        Validated order_by string
        
    Raises:
        ValueError: If order_by is not recognized
    """
    valid_orders = ("chronological", "document_order")
    
    if order_by not in valid_orders:
        raise ValueError(
            f"Invalid order_by: {order_by}. Must be one of: {valid_orders}"
        )
    
    return order_by


def get_order_config(order_by: str) -> ChronologyConfig:
    """
    Get the chronology config for an order_by value.
    
    Args:
        order_by: "chronological" or "document_order"
        
    Returns:
        ChronologyConfig for the specified ordering
    """
    if order_by == "chronological":
        return DEFAULT_CHRONOLOGY
    elif order_by == "document_order":
        # Document order: by document_id then page_seq then chunk_id
        return ChronologyConfig(
            key="document_order_v1",
            order_sql="""
                COALESCE(cm.document_id, 2147483647) ASC,
                COALESCE(p.page_seq, 2147483647) ASC,
                c.id ASC
            """,
        )
    else:
        raise ValueError(f"Unknown order_by: {order_by}")
