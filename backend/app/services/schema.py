"""
DB schema helpers.

Goal: keep the API stable even if the underlying schema evolves slightly
between environments (e.g., research_plans.user_utterance vs raw_utterance).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Set


@lru_cache(maxsize=128)
def get_table_columns(conn_dsn: str, table_name: str) -> Set[str]:
    """
    Fetch column names for a table from information_schema.

    We key the cache by (dsn, table) so local/dev/prod DBs don't mix.
    """
    import psycopg2

    with psycopg2.connect(conn_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                """,
                (table_name,),
            )
            return {r[0] for r in cur.fetchall()}


def plan_user_utterance_expr(columns: Set[str]) -> str:
    """
    Return SQL expression that yields the plan's user utterance across schema variants.
    """
    if "user_utterance" in columns and "raw_utterance" in columns:
        return "COALESCE(user_utterance, raw_utterance)"
    if "user_utterance" in columns:
        return "user_utterance"
    if "raw_utterance" in columns:
        return "raw_utterance"
    # Shouldn't happen, but avoid crashing on SELECT build
    return "''::text"


def has_column(columns: Set[str], name: str) -> bool:
    return name in columns
