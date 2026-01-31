"""
Result set endpoints
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.db import get_conn
from app.services.evidence import build_evidence_refs_from_chunk

router = APIRouter()


# =============================================================================
# Helpers
# =============================================================================

def get_plan_execution_mode(conn, result_set_id: int) -> Optional[str]:
    """
    Get the execution_mode for the plan that created this result set.
    
    Returns "retrieve", "count", or None if not found/not executed.
    Used to prevent operations on COUNT-mode results that expect evidence.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rp.plan_json->'_metadata'->>'execution_mode' as exec_mode
            FROM result_sets rs
            JOIN research_plans rp ON rp.result_set_id = rs.id
            WHERE rs.id = %s
            """,
            (result_set_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def check_result_set_has_evidence(conn, result_set_id: int) -> bool:
    """
    Check if a result set has evidence (chunk_ids).
    COUNT-mode executions don't create result sets, so this should always be true
    for result sets that exist, but we check anyway for safety.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_ids FROM result_sets WHERE id = %s",
            (result_set_id,),
        )
        row = cur.fetchone()
        return row is not None and row[0] is not None and len(row[0]) > 0


# =============================================================================
# Response Models
# =============================================================================

class EvidenceRef(BaseModel):
    document_id: int
    pdf_page: int
    chunk_id: Optional[int] = None
    span: Optional[dict] = None
    quote: Optional[str] = None
    why: Optional[str] = None


class ResultSummary(BaseModel):
    item_count: int
    document_count: int
    entity_count: Optional[int] = None


class ResultItem(BaseModel):
    id: str
    kind: Optional[str] = None
    rank: int
    text: str
    chunk_id: Optional[int] = None
    document_id: Optional[int] = None
    scores: Optional[dict] = None
    highlight: Optional[str] = None
    matched_terms: Optional[list] = None
    evidence_refs: list[EvidenceRef]


class ResultSetResponse(BaseModel):
    id: int
    name: str
    retrieval_run_id: int
    summary: ResultSummary
    items: list[ResultItem]
    created_at: datetime


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/{result_set_id}", response_model=ResultSetResponse)
def get_result_set(result_set_id: int):
    """Get a result set with all items and evidence refs."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Get result set metadata
            cur.execute(
                """
                SELECT id, name, retrieval_run_id, chunk_ids, created_at
                FROM result_sets
                WHERE id = %s
                """,
                (result_set_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Result set not found")
            
            rs_id, name, run_id, chunk_ids, created_at = row
            
            if not chunk_ids:
                return ResultSetResponse(
                    id=rs_id,
                    name=name,
                    retrieval_run_id=run_id,
                    summary=ResultSummary(item_count=0, document_count=0),
                    items=[],
                    created_at=created_at,
                )
            
            # Get chunk evidence data
            cur.execute(
                """
                SELECT 
                    e.chunk_id,
                    e.rank,
                    e.score_lex,
                    e.score_vec,
                    e.score_hybrid,
                    e.matched_lexemes,
                    e.highlight,
                    c.text
                FROM retrieval_run_chunk_evidence e
                JOIN chunks c ON c.id = e.chunk_id
                WHERE e.retrieval_run_id = %s
                    AND e.chunk_id = ANY(%s)
                ORDER BY e.rank
                """,
                (run_id, chunk_ids),
            )
            evidence_rows = cur.fetchall()
            
            # Build items with evidence refs
            items = []
            document_ids = set()
            
            for row in evidence_rows:
                chunk_id, rank, score_lex, score_vec, score_hybrid, lexemes, highlight, text = row
                
                # Build evidence refs for this chunk
                evidence_refs = build_evidence_refs_from_chunk(conn, chunk_id)
                
                # Track unique documents
                for ref in evidence_refs:
                    document_ids.add(ref["document_id"])
                
                items.append(ResultItem(
                    id=f"chunk-{chunk_id}",
                    kind="chunk",
                    rank=rank,
                    text=text[:500] if text else "",  # Truncate for UI
                    chunk_id=chunk_id,
                    document_id=evidence_refs[0]["document_id"] if evidence_refs else None,
                    scores={
                        "lex": score_lex,
                        "vec": score_vec,
                        "hybrid": score_hybrid,
                    } if any([score_lex, score_vec, score_hybrid]) else None,
                    highlight=highlight,
                    matched_terms=lexemes,
                    evidence_refs=[EvidenceRef(**ref) for ref in evidence_refs],
                ))
            
            return ResultSetResponse(
                id=rs_id,
                name=name,
                retrieval_run_id=run_id,
                summary=ResultSummary(
                    item_count=len(items),
                    document_count=len(document_ids),
                ),
                items=items,
                created_at=created_at,
            )
    finally:
        conn.close()


# =============================================================================
# Aggregation Models
# =============================================================================

class AggregateItem(BaseModel):
    """A single aggregation bucket."""
    key: str
    label: Optional[str] = None
    count: int


class AggregateResponse(BaseModel):
    """Aggregation response."""
    total_count: int
    group_by: Optional[str] = None
    buckets: List[AggregateItem]


# =============================================================================
# Aggregation Endpoints
# =============================================================================

@router.get("/{result_set_id}/aggregate", response_model=AggregateResponse)
def aggregate_result_set(
    result_set_id: int,
    group_by: Optional[str] = Query(
        None, 
        description="Group results by: entity, document, collection, or None for total count"
    ),
):
    """
    Aggregate a result set - get counts without full result data.
    
    Useful for "how many X in this result set?" queries.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Get result set metadata
            cur.execute(
                """
                SELECT id, chunk_ids, retrieval_run_id
                FROM result_sets
                WHERE id = %s
                """,
                (result_set_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Result set not found")
            
            rs_id, chunk_ids, run_id = row
            
            if not chunk_ids:
                return AggregateResponse(
                    total_count=0,
                    group_by=group_by,
                    buckets=[],
                )
            
            total_count = len(chunk_ids)
            
            if group_by is None:
                # Just return total count
                return AggregateResponse(
                    total_count=total_count,
                    group_by=None,
                    buckets=[],
                )
            
            elif group_by == "document":
                # Count chunks per document
                cur.execute(
                    """
                    SELECT 
                        c.document_id,
                        d.title,
                        COUNT(*) as cnt
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.id = ANY(%s)
                    GROUP BY c.document_id, d.title
                    ORDER BY cnt DESC
                    """,
                    (chunk_ids,),
                )
                buckets = [
                    AggregateItem(
                        key=str(row[0]),
                        label=row[1] or f"Document {row[0]}",
                        count=row[2],
                    )
                    for row in cur.fetchall()
                ]
                return AggregateResponse(
                    total_count=total_count,
                    group_by="document",
                    buckets=buckets,
                )
            
            elif group_by == "collection":
                # Count chunks per collection
                cur.execute(
                    """
                    SELECT 
                        col.slug,
                        col.title,
                        COUNT(*) as cnt
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    JOIN collections col ON col.id = d.collection_id
                    WHERE c.id = ANY(%s)
                    GROUP BY col.slug, col.title
                    ORDER BY cnt DESC
                    """,
                    (chunk_ids,),
                )
                buckets = [
                    AggregateItem(
                        key=row[0],
                        label=row[1] or row[0],
                        count=row[2],
                    )
                    for row in cur.fetchall()
                ]
                return AggregateResponse(
                    total_count=total_count,
                    group_by="collection",
                    buckets=buckets,
                )
            
            elif group_by == "entity":
                # Count entity mentions in these chunks
                cur.execute(
                    """
                    SELECT 
                        em.entity_id,
                        e.canonical_name,
                        COUNT(*) as cnt
                    FROM entity_mentions em
                    JOIN entities e ON e.id = em.entity_id
                    WHERE em.chunk_id = ANY(%s)
                    GROUP BY em.entity_id, e.canonical_name
                    ORDER BY cnt DESC
                    LIMIT 50
                    """,
                    (chunk_ids,),
                )
                buckets = [
                    AggregateItem(
                        key=str(row[0]),
                        label=row[1],
                        count=row[2],
                    )
                    for row in cur.fetchall()
                ]
                return AggregateResponse(
                    total_count=total_count,
                    group_by="entity",
                    buckets=buckets,
                )
            
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid group_by value: {group_by}. Must be one of: document, collection, entity, or None."
                )
    
    finally:
        conn.close()


@router.get("/stats/global", response_model=AggregateResponse)
def get_global_stats(
    group_by: str = Query(
        "collection",
        description="Group by: collection, document, entity"
    ),
):
    """
    Get global statistics across all data.
    
    Useful for "how many documents in the corpus?" type queries.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if group_by == "collection":
                cur.execute(
                    """
                    SELECT 
                        col.slug,
                        col.title,
                        COUNT(DISTINCT d.id) as doc_count
                    FROM collections col
                    LEFT JOIN documents d ON d.collection_id = col.id
                    GROUP BY col.slug, col.title
                    ORDER BY doc_count DESC
                    """
                )
                buckets = [
                    AggregateItem(
                        key=row[0],
                        label=row[1] or row[0],
                        count=row[2],
                    )
                    for row in cur.fetchall()
                ]
                
                # Get total document count
                cur.execute("SELECT COUNT(*) FROM documents")
                total = cur.fetchone()[0]
                
                return AggregateResponse(
                    total_count=total,
                    group_by="collection",
                    buckets=buckets,
                )
            
            elif group_by == "entity":
                # Top entities by mention count
                cur.execute(
                    """
                    SELECT 
                        e.id,
                        e.canonical_name,
                        COUNT(em.id) as mention_count
                    FROM entities e
                    LEFT JOIN entity_mentions em ON em.entity_id = e.id
                    GROUP BY e.id, e.canonical_name
                    ORDER BY mention_count DESC
                    LIMIT 50
                    """
                )
                buckets = [
                    AggregateItem(
                        key=str(row[0]),
                        label=row[1],
                        count=row[2],
                    )
                    for row in cur.fetchall()
                ]
                
                # Get total entity count
                cur.execute("SELECT COUNT(*) FROM entities")
                total = cur.fetchone()[0]
                
                return AggregateResponse(
                    total_count=total,
                    group_by="entity",
                    buckets=buckets,
                )
            
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid group_by value: {group_by}. Must be one of: collection, entity."
                )
    
    finally:
        conn.close()


# =============================================================================
# Summarization Models & Endpoints
# =============================================================================

class SummarizeRequest(BaseModel):
    """Request body for summarization."""
    summary_type: str = "brief"  # brief, detailed, thematic


class SummaryResponse(BaseModel):
    """Summarization response."""
    result_set_id: int
    result_set_name: str
    summary_type: str
    summary: str
    chunk_count: int
    summarized_count: int


@router.post("/{result_set_id}/summarize", response_model=SummaryResponse)
def summarize_result_set_endpoint(
    result_set_id: int,
    req: SummarizeRequest,
):
    """
    Generate an LLM summary of a result set.
    
    Summary types:
    - brief: 2-3 sentence overview
    - detailed: Comprehensive summary with key findings
    - thematic: Organized by themes/topics
    """
    import sys
    import os
    from pathlib import Path
    
    # Add repo root to path for imports
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    
    conn = get_conn()
    try:
        # Check if result set has evidence (chunks to summarize)
        if not check_result_set_has_evidence(conn, result_set_id):
            # Check if this is from a COUNT-mode execution
            exec_mode = get_plan_execution_mode(conn, result_set_id)
            if exec_mode == "count":
                raise HTTPException(
                    status_code=400,
                    detail="Cannot summarize COUNT-mode results. Use --materialize to convert to full retrieval first."
                )
            raise HTTPException(
                status_code=404,
                detail="Result set not found or has no evidence to summarize"
            )
        
        # Import the summarization function
        try:
            from scripts.summarize_results import summarize_result_set
        except ImportError as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Summarization module not available: {e}"
            )
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY not configured for summarization"
            )
        
        # Generate summary
        try:
            result = summarize_result_set(
                conn,
                result_set_id,
                summary_type=req.summary_type,
                save=False,  # Don't auto-save via API
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
        
        return SummaryResponse(
            result_set_id=result["result_set_id"],
            result_set_name=result.get("result_set_name", ""),
            summary_type=result["summary_type"],
            summary=result["summary"],
            chunk_count=result["chunk_count"],
            summarized_count=result["summarized_count"],
        )
    
    finally:
        conn.close()
