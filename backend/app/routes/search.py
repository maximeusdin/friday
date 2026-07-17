"""
Search tab endpoints: deterministic concordance-style search.

POST /api/search/result-sets - Create search
GET /api/search/result-sets/{id} - Metadata, coverage, status
GET /api/search/result-sets/{id}/items - Paginated page hits (denormalized)
GET /api/search/result-sets/{id}/export - CSV/JSON export
"""
import csv
import io
import json
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.routes.auth_cognito import require_user
from app.services.db import get_conn
from app.services.session_ownership import assert_session_owned

router = APIRouter()


def _assert_search_result_set_owned(conn, result_set_id: str, user_sub: str) -> None:
    """Raise 404 if search result set does not exist or is not owned by user_sub."""
    try:
        rid = uuid.UUID(result_set_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid result set ID")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM search_result_sets WHERE id = %s AND user_sub = %s",
            (str(rid), user_sub),
        )
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Search result set not found")


# =============================================================================
# Request/Response Models
# =============================================================================


class SearchCreateRequest(BaseModel):
    session_id: int  # Required: search must be part of a session (like chat)
    scope: Dict[str, Any] = {"mode": "full_archive"}
    query: str
    mode: str = "exact"
    unit: str = "page"
    sort: str = "canonical"
    alias_expand: bool = True
    fuzzy_progressive: bool = False  # When mode=fuzzy: run exact first, return; client calls expand-fuzzy


class SearchCreateResponse(BaseModel):
    result_set_id: str
    status: str
    fuzzy_pending: bool = False  # True when exact_complete; client should call expand-fuzzy
    notice: Optional[str] = None       # e.g. "relaxed your sentence to keywords"
    relaxed_query: Optional[str] = None  # the keyword query actually run, if relaxed


# Framing words dropped when relaxing a natural-language Search query to keywords.
_SEARCH_FRAMING = {
    "find", "show", "list", "get", "tell", "give", "identify", "evidence", "description",
    "describe", "who", "whom", "what", "when", "where", "how", "many", "much", "did", "does",
    "do", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "with", "for", "to",
    "by", "and", "or", "that", "this", "me", "us", "all", "any", "some", "please", "search",
    "spied", "spy", "spying", "spies", "recruited", "recruit", "about", "was", "there", "which",
    "into", "from", "between", "his", "her", "their", "our", "them", "he", "she", "they",
}


def _looks_like_nl_query(q: str) -> bool:
    """A multi-word question/sentence (has framing words) rather than a keyword query."""
    words = re.findall(r"[A-Za-z0-9']+", q or "")
    return len(words) >= 4 and any(w.lower() in _SEARCH_FRAMING for w in words)


def _keyword_relax(q: str) -> str:
    """Drop framing words, keep content nouns/names/numbers and quoted phrases."""
    parts = re.findall(r'"[^"]+"|\S+', q or "")
    kept: List[str] = []
    for p in parts:
        if p.startswith('"') and p.endswith('"'):
            kept.append(p)
            continue
        w = re.sub(r"[^A-Za-z0-9\-']", "", p)
        if w and len(w) >= 2 and w.lower() not in _SEARCH_FRAMING:
            kept.append(w)
    return " ".join(kept).strip()


class SearchResultSetResponse(BaseModel):
    id: str
    status: str
    total_hits: Optional[int] = None
    coverage_json: Optional[Dict[str, Any]] = None
    is_exhaustive: bool = True
    expanded_terms_json: Optional[Any] = None
    query_display: Optional[str] = None
    error_message: Optional[str] = None


class SearchResultSetSummary(BaseModel):
    """Lightweight row for listing a session's saved searches (no page hits)."""
    id: str
    created_at: str
    query_display: Optional[str] = None
    query_raw: Optional[str] = None
    mode: str = "exact"
    status: str = "complete"
    total_hits: Optional[int] = None
    is_exhaustive: bool = True


class SearchPageHitItem(BaseModel):
    collection: Dict[str, Any]
    document: Dict[str, Any]
    page: Dict[str, Any]
    snippet: Optional[str] = None
    chunk_id: int
    evidence_ref: Dict[str, Any]
    viewer_url: Optional[str] = None
    asset_url: Optional[str] = None


class SearchItemsResponse(BaseModel):
    items: List[SearchPageHitItem]
    next_cursor: Optional[str] = None
    total_hits: int


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/result-sets", response_model=SearchCreateResponse)
def create_search(
    req: SearchCreateRequest,
    user=Depends(require_user),
):
    """Create a search, run it synchronously, return result set ID. Requires session."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    assert_session_owned(req.session_id, user["sub"])
    conn = get_conn()
    try:
        result_set_id = str(uuid.uuid4())
        user_sub = user["sub"]
        scope = req.scope or {"mode": "full_archive"}

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_result_sets
                (id, user_sub, session_id, scope_json, query_raw, mode, unit, sort_order, alias_expand, is_exhaustive, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'running')
                """,
                (
                    result_set_id,
                    user_sub,
                    req.session_id,
                    json.dumps(scope),
                    req.query.strip(),
                    req.mode,
                    req.unit,
                    req.sort,
                    req.alias_expand,
                    req.mode == "exact",
                ),
            )
        conn.commit()

        # Run search synchronously
        from retrieval.search_executor import run_search

        exact_only = req.mode == "fuzzy" and req.fuzzy_progressive
        out = run_search(
            conn,
            result_set_id,
            req.query.strip(),
            scope,
            alias_expand=req.alias_expand,
            mode=req.mode,
            exact_only=exact_only,
        )

        if out.get("status") == "error":
            raise HTTPException(status_code=400, detail=out.get("error", "Search failed"))

        # VI: Search is an exact keyword engine. If the user typed a natural-language
        # sentence (e.g. "find evidence that Bentley spied on Waldo") the AND of every word
        # — including the verb "spied", which never co-occurs with the answer — yields zero.
        # Rather than silently return nothing, relax to the content keywords and re-run.
        notice = None
        relaxed_query = None
        if out.get("total_hits", 0) == 0 and _looks_like_nl_query(req.query):
            relaxed = _keyword_relax(req.query)
            if relaxed and relaxed.lower() != req.query.strip().lower() and len(relaxed.split()) >= 1:
                out2 = run_search(
                    conn, result_set_id, relaxed, scope,
                    alias_expand=req.alias_expand, mode=req.mode, exact_only=exact_only,
                )
                if out2.get("status") != "error" and out2.get("total_hits", 0) > 0:
                    out = out2
                    relaxed_query = relaxed
                    notice = (
                        f'No exact matches for your full sentence. Search matches exact terms, '
                        f'so I searched the keywords instead: "{relaxed}".'
                    )
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE search_result_sets SET query_display = %s WHERE id = %s",
                                (f'{relaxed}  (keywords from: "{req.query.strip()}")', result_set_id),
                            )
                        conn.commit()
                    except Exception:
                        conn.rollback()

        fuzzy_pending = out.get("status") == "exact_complete"
        return SearchCreateResponse(
            result_set_id=result_set_id,
            status=out.get("status", "complete"),
            fuzzy_pending=fuzzy_pending,
            notice=notice,
            relaxed_query=relaxed_query,
        )
    finally:
        conn.close()


@router.post("/result-sets/{result_set_id}/expand-fuzzy")
def expand_fuzzy(
    result_set_id: str,
    user=Depends(require_user),
):
    """Expand a result set (exact_complete) with fuzzy/trigram matches. For progressive fuzzy UX."""
    conn = get_conn()
    try:
        _assert_search_result_set_owned(conn, result_set_id, user["sub"])
        from retrieval.search_executor import run_search_expand_fuzzy

        out = run_search_expand_fuzzy(conn, result_set_id)
        if out.get("status") == "error":
            raise HTTPException(status_code=400, detail=out.get("error", "Expand failed"))
        return {"status": "complete", "total_hits": out.get("total_hits", 0)}
    finally:
        conn.close()


@router.post("/result-sets/{result_set_id}/fetch-more")
def fetch_more_snippets(
    result_set_id: str,
    batch_size: int = Query(100, ge=1, le=500),
    user=Depends(require_user),
):
    """Compute snippets for the next batch of hits that have empty snippets."""
    conn = get_conn()
    try:
        _assert_search_result_set_owned(conn, result_set_id, user["sub"])
        from retrieval.search_executor import fetch_more_snippets as do_fetch

        n = do_fetch(conn, result_set_id, batch_size=batch_size)
        return {"fetched": n}
    finally:
        conn.close()


@router.get("/suggest")
def search_suggest(
    q: str = Query("", min_length=1, max_length=50),
    limit: int = Query(10, ge=1, le=20),
    user=Depends(require_user),
):
    """Autocomplete suggestions for search query. Returns top terms matching prefix from corpus."""
    conn = get_conn()
    try:
        prefix = (q or "").strip().lower()
        if not prefix or len(prefix) < 2:
            return {"suggestions": []}

        # Use tsv_simple if available, else tsv
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'tsv_simple'
            """)
            tsv_col = "tsv_simple" if cur.fetchone() else "tsv"

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT word, ndoc FROM ts_stat(
                    'SELECT {tsv_col} FROM chunks WHERE {tsv_col} IS NOT NULL'
                )
                WHERE word LIKE %s
                ORDER BY ndoc DESC
                LIMIT %s
                """.format(tsv_col=tsv_col),
                (prefix + "%", limit),
            )
            rows = cur.fetchall()

        return {
            "suggestions": [{"term": r[0], "ndoc": r[1]} for r in rows],
        }
    finally:
        conn.close()


@router.get("/result-sets", response_model=List[SearchResultSetSummary])
def list_search_result_sets(
    session_id: int = Query(..., description="List saved searches for this session"),
    user=Depends(require_user),
):
    """List a session's saved searches (oldest first), so the Search tab can reload
    history the same way chat reloads its messages. Ownership is enforced per-session."""
    assert_session_owned(session_id, user["sub"])
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, query_display, query_raw, mode, status,
                       total_hits, is_exhaustive
                FROM search_result_sets
                WHERE session_id = %s AND user_sub = %s
                ORDER BY created_at ASC
                """,
                (session_id, user["sub"]),
            )
            rows = cur.fetchall()
        return [
            SearchResultSetSummary(
                id=str(r[0]),
                created_at=r[1].isoformat() if r[1] else "",
                query_display=r[2],
                query_raw=r[3],
                mode=r[4] or "exact",
                status=r[5] or "complete",
                total_hits=r[6],
                is_exhaustive=r[7] if r[7] is not None else True,
            )
            for r in rows
        ]
    finally:
        conn.close()


@router.delete("/result-sets/{result_set_id}")
def delete_search_result_set(
    result_set_id: str,
    user=Depends(require_user),
):
    """Delete a saved search (and its page hits via ON DELETE CASCADE)."""
    conn = get_conn()
    try:
        _assert_search_result_set_owned(conn, result_set_id, user["sub"])
        with conn.cursor() as cur:
            cur.execute("DELETE FROM search_result_sets WHERE id = %s", (result_set_id,))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@router.delete("/result-sets/{result_set_id}/items")
def delete_search_result_item(
    result_set_id: str,
    document_id: int = Query(...),
    page_id: int = Query(...),
    user=Depends(require_user),
):
    """Remove a single page hit from a saved search (persists across reloads),
    keeping total_hits in sync so numbering and "Showing X of Y" stay correct."""
    conn = get_conn()
    try:
        _assert_search_result_set_owned(conn, result_set_id, user["sub"])
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM search_result_page_hits
                WHERE result_set_id = %s AND document_id = %s AND page_id = %s
                """,
                (result_set_id, document_id, page_id),
            )
            deleted = cur.rowcount
            if deleted:
                cur.execute(
                    """
                    UPDATE search_result_sets
                    SET total_hits = GREATEST(COALESCE(total_hits, 0) - %s, 0)
                    WHERE id = %s
                    """,
                    (deleted, result_set_id),
                )
        conn.commit()
        return {"deleted": deleted}
    finally:
        conn.close()


@router.get("/result-sets/{result_set_id}", response_model=SearchResultSetResponse)
def get_search_result_set(
    result_set_id: str,
    user=Depends(require_user),
):
    """Get search result set metadata, coverage, status."""
    conn = get_conn()
    try:
        _assert_search_result_set_owned(conn, result_set_id, user["sub"])

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, total_hits, coverage_json, is_exhaustive,
                       expanded_terms_json, query_display, error_message
                FROM search_result_sets
                WHERE id = %s
                """,
                (result_set_id,),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Search result set not found")

        return SearchResultSetResponse(
            id=str(row[0]),
            status=row[1],
            total_hits=row[2],
            coverage_json=row[3],
            is_exhaustive=row[4] or True,
            expanded_terms_json=row[5],
            query_display=row[6],
            error_message=row[7],
        )
    finally:
        conn.close()


def _build_items_query(
    result_set_id: str,
    cursor: Optional[str],
    limit: int,
) -> tuple[str, Dict[str, Any]]:
    """Build query for paginated items with keyset cursor."""
    params: Dict[str, Any] = {"result_set_id": result_set_id, "limit": limit + 1}
    if cursor:
        # Cursor format: collection_id,document_id,page_seq,page_id
        parts = cursor.split(",")
        if len(parts) >= 4:
            params["c_col"] = int(parts[0])
            params["c_doc"] = int(parts[1])
            params["c_seq"] = int(parts[2])
            params["c_page"] = int(parts[3])
            cursor_where = """
                AND (h.collection_id, h.document_id, h.page_seq, h.page_id) > (%(c_col)s, %(c_doc)s, %(c_seq)s, %(c_page)s)
            """
        else:
            cursor_where = ""
    else:
        cursor_where = ""

    sql = f"""
    SELECT
        h.collection_id, h.document_id, h.page_id, h.page_seq, h.pdf_page_number,
        h.chunk_id, h.snippet,
        col.slug AS collection_slug, col.title AS collection_title,
        d.source_name AS document_title
    FROM search_result_page_hits h
    JOIN collections col ON col.id = h.collection_id
    JOIN documents d ON d.id = h.document_id
    WHERE h.result_set_id = %(result_set_id)s
    {cursor_where}
    ORDER BY h.collection_id, h.document_id, h.page_seq, h.page_id
    LIMIT %(limit)s
    """
    return sql, params


@router.get("/result-sets/{result_set_id}/items", response_model=SearchItemsResponse)
def get_search_result_set_items(
    result_set_id: str,
    user=Depends(require_user),
    cursor: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Get paginated page hits with denormalized link fields."""
    conn = get_conn()
    try:
        _assert_search_result_set_owned(conn, result_set_id, user["sub"])

        sql, params = _build_items_query(result_set_id, cursor, limit)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = []
        for r in rows:
            (col_id, doc_id, page_id, page_seq, pdf_page, chunk_id, snippet,
             col_slug, col_title, doc_title) = r

            evidence_ref = {
                "document_id": doc_id,
                "pdf_page": pdf_page or page_seq,
                "chunk_id": chunk_id,
                "quote": snippet,
            }
            items.append(
                SearchPageHitItem(
                    collection={"id": col_id, "slug": col_slug, "title": col_title},
                    document={"id": doc_id, "title": doc_title},
                    page={"id": page_id, "seq": page_seq, "pdf_page": pdf_page or page_seq},
                    snippet=snippet,
                    chunk_id=chunk_id,
                    evidence_ref=evidence_ref,
                    viewer_url=f"/?document_id={doc_id}&pdf_page={pdf_page or page_seq}",
                    asset_url=None,
                )
            )

        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = f"{last[0]},{last[1]},{last[3]},{last[2]}"

        with conn.cursor() as cur:
            cur.execute(
                "SELECT total_hits FROM search_result_sets WHERE id = %s",
                (result_set_id,),
            )
            row = cur.fetchone()
        total_hits = row[0] if row else 0

        return SearchItemsResponse(
            items=items,
            next_cursor=next_cursor,
            total_hits=total_hits or len(items),
        )
    finally:
        conn.close()


@router.get("/result-sets/{result_set_id}/export")
def export_search_result_set(
    result_set_id: str,
    user=Depends(require_user),
    format: str = Query("csv", pattern="^(csv|json)$"),
):
    """Export search results as CSV or JSON with friday_url column."""
    conn = get_conn()
    try:
        _assert_search_result_set_owned(conn, result_set_id, user["sub"])

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    h.collection_id, h.document_id, h.page_id, h.page_seq, h.pdf_page_number,
                    h.chunk_id, h.snippet,
                    col.slug, col.title,
                    d.source_name
                FROM search_result_page_hits h
                JOIN collections col ON col.id = h.collection_id
                JOIN documents d ON d.id = h.document_id
                WHERE h.result_set_id = %s
                ORDER BY h.collection_id, h.document_id, h.page_seq
                """,
                (result_set_id,),
            )
            rows = cur.fetchall()

        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["collection", "document", "page", "snippet", "page_id", "document_id", "friday_url"])
            for r in rows:
                col_id, doc_id, page_id, page_seq, pdf_page, chunk_id, snippet, col_slug, col_title, doc_title = r
                friday_url = f"/?document_id={doc_id}&pdf_page={pdf_page or page_seq}"
                writer.writerow([col_title or col_slug, doc_title, pdf_page or page_seq, snippet or "", page_id, doc_id, friday_url])
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=search_{result_set_id[:8]}.csv"},
            )
        else:
            export_data = []
            for r in rows:
                col_id, doc_id, page_id, page_seq, pdf_page, chunk_id, snippet, col_slug, col_title, doc_title = r
                friday_url = f"/?document_id={doc_id}&pdf_page={pdf_page or page_seq}"
                export_data.append({
                    "collection": col_title or col_slug,
                    "document": doc_title,
                    "page": pdf_page or page_seq,
                    "snippet": snippet,
                    "page_id": page_id,
                    "document_id": doc_id,
                    "friday_url": friday_url,
                })
            return StreamingResponse(
                iter([json.dumps(export_data, indent=2)]),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=search_{result_set_id[:8]}.json"},
            )
    finally:
        conn.close()
