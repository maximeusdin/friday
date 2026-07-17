"""
Concordance Index endpoints — browse and export the entity/alias concordance
(the master index of people, organizations, and cover names behind alias
expansion and codename resolution).
"""
import csv
import io
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.routes.auth_cognito import require_user
from app.services.db import get_conn

router = APIRouter()


class ConcordanceEntry(BaseModel):
    id: int
    canonical_name: str
    entity_type: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = []


class ConcordanceEntriesResponse(BaseModel):
    entries: List[ConcordanceEntry]
    total: int
    offset: int
    limit: int


@router.get("/concordance/summary")
def concordance_summary(user=Depends(require_user)):
    """Counts for the Concordance Index card (entities, aliases, by type)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM entities")
            entities = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM entity_aliases")
            aliases = cur.fetchone()[0]
            cur.execute(
                "SELECT COALESCE(entity_type, 'unknown'), count(*) FROM entities "
                "GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
            )
            by_type = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]
        return {"entities": entities, "aliases": aliases, "by_type": by_type}
    finally:
        conn.close()


@router.get("/concordance/entries", response_model=ConcordanceEntriesResponse)
def concordance_entries(
    query: Optional[str] = Query(None, description="Filter by name or alias (substring)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(require_user),
):
    """Paginated concordance entries (canonical name + aliases), A→Z, searchable."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if query and query.strip():
                like = f"%{query.strip()}%"
                cur.execute(
                    """
                    SELECT count(DISTINCT e.id)
                    FROM entities e
                    LEFT JOIN entity_aliases a ON a.entity_id = e.id
                    WHERE e.canonical_name ILIKE %s OR a.alias ILIKE %s
                    """,
                    (like, like),
                )
                total = cur.fetchone()[0]
                cur.execute(
                    """
                    SELECT e.id, e.canonical_name, e.entity_type, e.description,
                           COALESCE(array_agg(DISTINCT a.alias)
                                    FILTER (WHERE a.alias IS NOT NULL), '{}')
                    FROM entities e
                    LEFT JOIN entity_aliases a ON a.entity_id = e.id
                    WHERE e.id IN (
                        SELECT DISTINCT e2.id
                        FROM entities e2
                        LEFT JOIN entity_aliases a2 ON a2.entity_id = e2.id
                        WHERE e2.canonical_name ILIKE %s OR a2.alias ILIKE %s
                    )
                    GROUP BY e.id
                    ORDER BY e.canonical_name ASC
                    LIMIT %s OFFSET %s
                    """,
                    (like, like, limit, offset),
                )
            else:
                cur.execute("SELECT count(*) FROM entities")
                total = cur.fetchone()[0]
                cur.execute(
                    """
                    SELECT e.id, e.canonical_name, e.entity_type, e.description,
                           COALESCE(array_agg(DISTINCT a.alias)
                                    FILTER (WHERE a.alias IS NOT NULL), '{}')
                    FROM entities e
                    LEFT JOIN entity_aliases a ON a.entity_id = e.id
                    GROUP BY e.id
                    ORDER BY e.canonical_name ASC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
            rows = cur.fetchall()
        entries = [
            ConcordanceEntry(
                id=r[0],
                canonical_name=r[1] or "",
                entity_type=r[2],
                description=r[3],
                # Drop the alias that just repeats the canonical name
                aliases=sorted({x for x in (r[4] or []) if x and x != r[1]}),
            )
            for r in rows
        ]
        return ConcordanceEntriesResponse(entries=entries, total=total, offset=offset, limit=limit)
    finally:
        conn.close()


@router.get("/concordance/export")
def concordance_export(
    format: str = Query("csv", pattern="^csv$"),
    user=Depends(require_user),
):
    """Download the full Concordance Index as CSV (one row per entity,
    aliases joined with '; ')."""
    conn = get_conn()

    def generate():
        try:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["entity_id", "canonical_name", "entity_type", "description", "aliases"])
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)
            # Server-side cursor so the 25k-row export streams without
            # materializing everything in memory.
            with conn.cursor(name="concordance_export_cur") as cur:
                cur.itersize = 1000
                cur.execute(
                    """
                    SELECT e.id, e.canonical_name, e.entity_type, e.description,
                           COALESCE(string_agg(DISTINCT a.alias, '; ')
                                    FILTER (WHERE a.alias IS NOT NULL AND a.alias <> e.canonical_name), '')
                    FROM entities e
                    LEFT JOIN entity_aliases a ON a.entity_id = e.id
                    GROUP BY e.id
                    ORDER BY e.canonical_name ASC
                    """
                )
                for r in cur:
                    writer.writerow([
                        r[0],
                        r[1] or "",
                        r[2] or "",
                        (r[3] or "").replace("\n", " ").strip(),
                        r[4] or "",
                    ])
                    yield buf.getvalue()
                    buf.seek(0)
                    buf.truncate(0)
        finally:
            conn.close()

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="concordance_index.csv"'},
    )
