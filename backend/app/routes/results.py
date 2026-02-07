"""
Result set endpoints
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.routes.auth_cognito import require_user
from app.services.db import get_conn
from app.services.evidence import build_evidence_refs_from_chunk
from app.services.session_ownership import assert_result_set_owned

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
def get_result_set(result_set_id: int, user=Depends(require_user)):
    """Get a result set with all items and evidence refs."""
    assert_result_set_owned(result_set_id, user["sub"])
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
    user=Depends(require_user),
    group_by: Optional[str] = Query(
        None, 
        description="Group results by: entity, document, collection, or None for total count"
    ),
):
    """
    Aggregate a result set - get counts without full result data.
    
    Useful for "how many X in this result set?" queries.
    """
    assert_result_set_owned(result_set_id, user["sub"])
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
                        cm.document_id,
                        d.source_name,
                        COUNT(*) as cnt
                    FROM chunks c
                    LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    LEFT JOIN documents d ON d.id = cm.document_id
                    WHERE c.id = ANY(%s)
                    GROUP BY cm.document_id, d.source_name
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
    user=Depends(require_user),
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
    user=Depends(require_user),
):
    """
    Generate an LLM summary of a result set.
    
    Summary types:
    - brief: 2-3 sentence overview
    - detailed: Comprehensive summary with key findings
    - thematic: Organized by themes/topics
    """
    assert_result_set_owned(result_set_id, user["sub"])
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


# =============================================================================
# Structured Summarization V2 (LLM with Citations)
# =============================================================================

class SummarizeRequestV2(BaseModel):
    """Request body for structured summarization."""
    question: Optional[str] = None
    profile: str = "conversational_answer"
    doc_focus_mode: Optional[str] = None  # Override: "auto" | "global" | "single_doc"
    summary_type: str = "sample"  # "sample" | "page_window"
    page_window: Optional[dict] = None  # {offset: 0, limit: 50} if summary_type="page_window"
    filters: Optional[dict] = None


# Import summarizer models for response (avoid circular import)
from app.services.summarizer.models import (
    SummaryArtifact as SummaryArtifactModel,
    CoverageInfo,
    ClaimWithSupport,
    CitationWithAnchor,
    ThemeWithEvidence,
    EntityInfo,
    DateCount,
    NextAction,
    ModelInfo,
)


@router.post("/{result_set_id}/summarize/v2", response_model=SummaryArtifactModel)
def summarize_result_set_v2(
    result_set_id: int,
    req: SummarizeRequestV2,
    user=Depends(require_user),
):
    """
    Generate a structured LLM summary with citation-backed claims.
    
    Features:
    - Deterministic evidence selection with diversity
    - Citation-backed claims with confidence levels
    - Server-generated coverage information
    - Signature-based caching for identical requests
    
    Profiles:
    - conversational_answer: Quick synthesis (default)
    - audit_digest: Comprehensive chronological review
    - entity_brief: Entity-focused summary
    - document_summary: Single document deep dive
    - quick_answer: Fast, minimal evidence
    """
    assert_result_set_owned(result_set_id, user["sub"])
    import os
    
    # Check for OpenAI API key early
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured for summarization"
        )
    
    conn = get_conn()
    try:
        # Check if result set has evidence
        if not check_result_set_has_evidence(conn, result_set_id):
            exec_mode = get_plan_execution_mode(conn, result_set_id)
            if exec_mode == "count":
                raise HTTPException(
                    status_code=400,
                    detail="Cannot summarize COUNT-mode results. Use --materialize first."
                )
            raise HTTPException(
                status_code=404,
                detail="Result set not found or has no evidence to summarize"
            )
        
        # Import summarizer components
        from app.services.summarizer.profiles import get_profile, PROFILES
        from app.services.summarizer.selection import run_selection
        from app.services.summarizer.bundles import build_bundles
        from app.services.summarizer.synthesis import (
            call_synthesis_llm,
            get_model_info,
        )
        from app.services.summarizer.validation import (
            validate_and_map_citations,
            extract_entities_from_citations,
        )
        from app.services.summarizer.coverage import generate_coverage_info
        from app.services.summarizer.signature import (
            compute_summary_signature,
            lookup_by_signature,
            persist_or_fetch_summary,
        )
        from app.services.summarizer.prompts import CURRENT_PROMPT_VERSION
        
        # Get profile
        try:
            profile = get_profile(req.profile)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Get retrieval metadata
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    rs.retrieval_run_id,
                    rs.session_id,
                    rr.retrieval_mode,
                    rr.was_capped
                FROM result_sets rs
                JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id
                WHERE rs.id = %s
                """,
                (result_set_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Result set not found")
            
            retrieval_run_id = row[0]
            session_id = row[1]
            retrieval_mode = row[2] or "conversational"
            cap_applied = row[3] or False
        
        # Determine effective doc focus mode
        effective_doc_focus = req.doc_focus_mode or profile.doc_focus_mode
        
        # Run selection (Stage A)
        selection_result = run_selection(
            conn=conn,
            result_set_id=result_set_id,
            retrieval_run_id=retrieval_run_id,
            retrieval_mode=retrieval_mode,
            max_chunks=profile.max_chunks,
            base_max_per_doc=profile.base_max_per_doc,
            candidate_pool_limit=profile.candidate_pool_limit,
            lambdas=profile.lambdas,
            page_bucket_size=profile.page_bucket_size,
            utility_threshold=profile.utility_threshold,
            doc_focus_mode=effective_doc_focus,
            summary_type=req.summary_type,
            page_window=req.page_window,
        )
        
        if not selection_result.candidates:
            raise HTTPException(
                status_code=400,
                detail="No evidence available for summarization"
            )
        
        # Get model info
        model_info_dict = get_model_info()
        model_name = model_info_dict["name"]
        
        # Compute signature and check cache
        signature = compute_summary_signature(
            result_set_id=result_set_id,
            retrieval_run_id=retrieval_run_id,
            profile=req.profile,
            summary_type=req.summary_type,
            question=req.question,
            filters=req.filters,
            chunk_ids=selection_result.spec.chunk_ids,
            prompt_version=CURRENT_PROMPT_VERSION,
            model_name=model_name,
        )
        
        # Check cache first
        cached = lookup_by_signature(conn, signature)
        if cached:
            return cached.with_cached_flag(True)
        
        # Build evidence bundles
        bundles, bundle_map = build_bundles(
            conn=conn,
            selected_candidates=selection_result.candidates,
            result_set_id=result_set_id,
        )
        
        # Call LLM for synthesis (Stage B)
        try:
            synthesis_output = call_synthesis_llm(
                bundles=bundles,
                question=req.question,
                profile=profile,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Invalid LLM response: {e}")
        
        # Validate and map citations
        validation_result = validate_and_map_citations(
            output=synthesis_output,
            bundle_map=bundle_map,
            bundles=bundles,
        )
        
        # Extract entities from citations (server-side, more reliable)
        entities_verified = extract_entities_from_citations(
            conn=conn,
            valid_claims=validation_result.valid_claims,
            bundles=bundles,
        )
        
        # Generate coverage info (server-generated, not LLM)
        coverage = generate_coverage_info(
            chunks_summarized=len(selection_result.candidates),
            candidate_pool_size=selection_result.inputs.candidate_pool_size,
            total_hits=selection_result.inputs.total_available,
            selection_inputs=selection_result.inputs,
            retrieval_mode=retrieval_mode,
            cap_applied=cap_applied,
        )
        
        # Extract date facets from selected chunks
        date_counts: dict = {}
        for candidate in selection_result.candidates:
            if candidate.year:
                date_counts[candidate.year] = date_counts.get(candidate.year, 0) + 1
        dates = [DateCount(year=y, count=c) for y, c in sorted(date_counts.items())]
        
        # Build next actions
        next_actions = [
            NextAction(
                label="Expand evidence sample",
                action_type="expand",
                params={"increase_k": 20},
            ),
        ]
        if retrieval_mode == "conversational":
            next_actions.append(NextAction(
                label="Switch to thorough mode",
                action_type="set_mode",
                params={"mode": "thorough"},
            ))
        
        # Build final artifact
        import uuid
        artifact = SummaryArtifactModel(
            summary_id=str(uuid.uuid4()),  # Temporary, will be replaced on persist
            result_set_id=result_set_id,
            retrieval_run_id=retrieval_run_id,
            question=req.question,
            coverage=coverage,
            answer=validation_result.valid_claims,
            themes=validation_result.themes,
            entities_verified=entities_verified,
            entities_flagged=validation_result.entities_flagged,
            dates=dates,
            next_actions=next_actions,
            model=ModelInfo(**model_info_dict),
            cached=False,
        )
        
        # Persist and return (handles race conditions)
        return persist_or_fetch_summary(
            conn=conn,
            artifact=artifact,
            signature=signature,
            result_set_id=result_set_id,
            retrieval_run_id=retrieval_run_id,
            session_id=session_id,
            selection_spec=selection_result.spec.to_dict(),
            selection_inputs=selection_result.inputs.to_dict(),
            question=req.question,
            profile=req.profile,
            summary_type=req.summary_type,
            model_name=model_name,
            prompt_version=CURRENT_PROMPT_VERSION,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
    finally:
        conn.close()


# =============================================================================
# Summary Drill-Down Endpoints
# =============================================================================

class ClaimEvidenceItem(BaseModel):
    """Evidence item for a claim."""
    chunk_id: int
    bundle_id: str
    doc_id: int
    doc_title: str
    page: Optional[int] = None
    quote_anchor: dict
    snippet: str  # The snippet used in summarization
    full_text: Optional[str] = None  # Full chunk text if requested


class ClaimEvidenceResponse(BaseModel):
    """Response for claim evidence drill-down."""
    summary_id: str
    claim_id: str
    claim_text: str
    evidence: List[ClaimEvidenceItem]


@router.get("/{result_set_id}/summaries/{summary_id}/evidence/{claim_id}")
def get_claim_evidence(
    result_set_id: int,
    summary_id: str,
    claim_id: str,
    user=Depends(require_user),
    full: bool = Query(False, description="Include full chunk text (larger response)"),
):
    """
    Get evidence for a specific claim in a summary.
    
    Returns the chunks cited by the claim with:
    - snippet: The text shown to the LLM
    - quote_anchor: The exact excerpt for UI highlighting
    - full_text: Complete chunk text (only if full=true)
    
    Use this for drill-down UX when user wants to verify a claim.
    """
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        # Look up summary
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT output_json, selection_spec
                FROM session_summaries
                WHERE summary_id = %s AND result_set_id = %s
                """,
                (summary_id, result_set_id)
            )
            row = cur.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Summary {summary_id} not found for result set {result_set_id}"
                )
            
            output_json = row[0]
            selection_spec = row[1]
        
        # Find the claim
        claim_data = None
        for claim in output_json.get("answer", []):
            if claim.get("claim_id") == claim_id:
                claim_data = claim
                break
        
        if not claim_data:
            raise HTTPException(
                status_code=404,
                detail=f"Claim {claim_id} not found in summary"
            )
        
        # Get chunk IDs from citations
        bundle_id_map = selection_spec.get("bundle_id_map", {})
        cited_chunk_ids = []
        bundle_id_by_chunk = {}
        
        for citation in claim_data.get("support", []):
            chunk_id = citation.get("chunk_id")
            bundle_id = citation.get("bundle_id")
            if chunk_id:
                cited_chunk_ids.append(chunk_id)
                bundle_id_by_chunk[chunk_id] = bundle_id
        
        if not cited_chunk_ids:
            return ClaimEvidenceResponse(
                summary_id=summary_id,
                claim_id=claim_id,
                claim_text=claim_data.get("claim", ""),
                evidence=[],
            )
        
        # Fetch chunk data
        with conn.cursor() as cur:
            if full:
                cur.execute(
                    """
                    SELECT 
                        c.id as chunk_id,
                        cm.document_id,
                        d.source_name as doc_title,
                        p.page_number,
                        LEFT(COALESCE(c.clean_text, c.text), 800) as snippet,
                        COALESCE(c.clean_text, c.text) as full_text
                    FROM chunks c
                    LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    LEFT JOIN documents d ON d.id = cm.document_id
                    LEFT JOIN pages p ON p.id = cm.first_page_id
                    WHERE c.id = ANY(%s)
                    """,
                    (cited_chunk_ids,)
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        c.id as chunk_id,
                        cm.document_id,
                        d.source_name as doc_title,
                        p.page_number,
                        LEFT(COALESCE(c.clean_text, c.text), 800) as snippet,
                        NULL as full_text
                    FROM chunks c
                    LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    LEFT JOIN documents d ON d.id = cm.document_id
                    LEFT JOIN pages p ON p.id = cm.first_page_id
                    WHERE c.id = ANY(%s)
                    """,
                    (cited_chunk_ids,)
                )
            
            rows = cur.fetchall()
        
        # Build response
        evidence = []
        chunk_data_by_id = {r[0]: r for r in rows}
        
        for citation in claim_data.get("support", []):
            chunk_id = citation.get("chunk_id")
            if chunk_id not in chunk_data_by_id:
                continue
            
            row = chunk_data_by_id[chunk_id]
            evidence.append(ClaimEvidenceItem(
                chunk_id=chunk_id,
                bundle_id=citation.get("bundle_id", bundle_id_by_chunk.get(chunk_id, "")),
                doc_id=row[1] or 0,
                doc_title=row[2] or "Unknown",
                page=row[3],
                quote_anchor=citation.get("quote_anchor", {}),
                snippet=row[4] or "",
                full_text=row[5] if full else None,
            ))
        
        return ClaimEvidenceResponse(
            summary_id=summary_id,
            claim_id=claim_id,
            claim_text=claim_data.get("claim", ""),
            evidence=evidence,
        )
    
    finally:
        conn.close()


@router.get("/{result_set_id}/summaries/{summary_id}")
def get_summary(
    result_set_id: int,
    summary_id: str,
    user=Depends(require_user),
):
    """
    Retrieve a previously generated summary by ID.
    
    Returns the full SummaryArtifact as stored.
    """
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT output_json
                FROM session_summaries
                WHERE summary_id = %s AND result_set_id = %s
                """,
                (summary_id, result_set_id)
            )
            row = cur.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Summary {summary_id} not found for result set {result_set_id}"
                )
            
            return row[0]
    finally:
        conn.close()


@router.get("/{result_set_id}/summaries")
def list_summaries(
    result_set_id: int,
    user=Depends(require_user),
    limit: int = Query(10, ge=1, le=50),
):
    """
    List summaries for a result set.
    
    Returns summary metadata without full output (for listing UI).
    """
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    summary_id,
                    profile,
                    summary_type,
                    user_question,
                    model_name,
                    prompt_version,
                    created_at,
                    array_length(selected_chunk_ids, 1) as chunk_count
                FROM session_summaries
                WHERE result_set_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (result_set_id, limit)
            )
            
            summaries = []
            for row in cur.fetchall():
                summaries.append({
                    "summary_id": str(row[0]),
                    "profile": row[1],
                    "summary_type": row[2],
                    "question": row[3],
                    "model_name": row[4],
                    "prompt_version": row[5],
                    "created_at": row[6].isoformat() if row[6] else None,
                    "chunk_count": row[7] or 0,
                })
            
            return {"result_set_id": result_set_id, "summaries": summaries}
    finally:
        conn.close()


# =============================================================================
# Phase 7: Two-Mode Retrieval Endpoints
# =============================================================================

class ResultSetMetadata(BaseModel):
    """Standard metadata included in all result set responses."""
    result_set_id: int
    retrieval_run_id: int
    mode: str  # "conversational" | "thorough"
    mode_source: str  # "ui_toggle" | "primitive" | "trigger_phrase" | "default"
    total_hits: int
    total_hits_before_cap: int
    cap_applied: bool
    threshold_used: Optional[float] = None
    vector_metric: Optional[str] = None
    # Index retrieval fields
    retrieval_source: str = "search"  # "search" | "entity_index" | "date_index" | "place_index"
    primitive_type: Optional[str] = None  # e.g., "FIRST_MENTION", "MENTIONS"
    order_by: Optional[str] = None  # "chronological_v1" | "document_order"
    time_basis: Optional[str] = None  # "mentioned_date" | "document_date"
    geo_basis: Optional[str] = None  # "resolved_place_entities" | "surface_strings"
    # Expand action - URL to get more results (for FIRST_* primitives)
    expand_url: Optional[str] = None
    # Retrieval signature hash for determinism verification
    retrieval_signature_hash: Optional[str] = None
    chronology_version: Optional[str] = None


class PaginationInfo(BaseModel):
    offset: Optional[int] = None
    after_rank: Optional[int] = None
    limit: int
    has_more: bool
    next_cursor: Optional[int] = None  # rank of last item, use as after_rank


class ChunkItem(BaseModel):
    chunk_id: int
    rank: int
    document_id: Optional[int] = None
    preview: Optional[str] = None
    # Index retrieval fields
    page_label: Optional[str] = None
    pdf_page_number: Optional[int] = None
    mention_surface: Optional[str] = None  # Surface text of entity mention
    mention_start_char: Optional[int] = None
    mention_end_char: Optional[int] = None
    # date_key semantics depend on time_basis in metadata:
    # - time_basis="document_date": date_key is chunk_metadata.date_min (doc date)
    # - time_basis="mentioned_date": date_key is date_mentions.date_start (extracted date)
    # Always check metadata.time_basis to interpret this field correctly
    date_key: Optional[str] = None
    date_key_basis: Optional[str] = None  # "document_date" | "mentioned_date" - clarifies interpretation


class ChunksResponse(BaseModel):
    meta: ResultSetMetadata
    chunks: List[ChunkItem]
    pagination: PaginationInfo


class ScoreBreakdown(BaseModel):
    lexical: Optional[float] = None
    vector: Optional[float] = None
    hybrid: Optional[float] = None


class MatchTraceItem(BaseModel):
    chunk_id: int
    matched_entity_ids: List[int]
    matched_phrases: List[str]
    scope_passed: bool
    in_lexical: bool
    in_vector: bool
    scores: ScoreBreakdown
    rank: Optional[int] = None
    rank_explanation: Optional[str] = None
    primitive_matches: Optional[List[dict]] = None  # only if include_audit=True


class MatchTracesResponse(BaseModel):
    meta: ResultSetMetadata
    traces: List[MatchTraceItem]


class HighlightMatch(BaseModel):
    phrase: str
    start: int  # character offset in chunk text
    end: int    # character offset (exclusive)
    context: str  # snippet with context_chars before/after


class HighlightsResponse(BaseModel):
    chunk_id: int
    text_length: int
    matches: List[HighlightMatch]  # ordered by start position


class LineContextResponse(BaseModel):
    """Response for line-level context around a character span."""
    chunk_id: int
    page_id: Optional[int] = None
    document_id: Optional[int] = None
    
    # The actual highlighted span
    highlight_start: int
    highlight_end: int
    
    # Lines
    line_before: Optional[str] = None  # Line before the highlight line
    highlight_line: str  # The line containing the highlight
    line_after: Optional[str] = None  # Line after the highlight line
    
    # For deep linking
    page_label: Optional[str] = None
    pdf_page_number: Optional[int] = None


class EntityItem(BaseModel):
    entity_id: int
    name: Optional[str] = None
    entity_type: Optional[str] = None
    mention_count: int


class EntitiesResponse(BaseModel):
    meta: ResultSetMetadata
    entities: List[EntityItem]


class CoMentionItem(BaseModel):
    entity_id: int
    name: Optional[str] = None
    entity_type: Optional[str] = None
    cooccurrence_count: int


class CoMentionsResponse(BaseModel):
    meta: ResultSetMetadata
    target_entity_id: int
    window: str
    co_mentions: List[CoMentionItem]


# Date Facets
class DateBucket(BaseModel):
    """A bucket in a date histogram."""
    year: Optional[int] = None
    month: Optional[int] = None  # 1-12 if granularity is month
    chunk_count: int
    date_mention_count: int


class DateFacetsResponse(BaseModel):
    """Response for date facets aggregation."""
    meta: ResultSetMetadata
    granularity: str  # "year" | "month"
    time_basis: str  # "mentioned_date" | "document_date"
    buckets: List[DateBucket]


# Place Facets
class PlaceBucket(BaseModel):
    """A bucket in a place histogram."""
    place_entity_id: int
    place_name: Optional[str] = None
    country: Optional[str] = None
    chunk_count: int
    mention_count: int


class PlaceFacetsResponse(BaseModel):
    """Response for place facets aggregation."""
    meta: ResultSetMetadata
    buckets: List[PlaceBucket]


def _get_result_set_metadata(conn, result_set_id: int) -> ResultSetMetadata:
    """Helper to get result set metadata."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                rs.id, rs.retrieval_run_id,
                COALESCE(rr.retrieval_mode, 'conversational') as mode,
                COALESCE(rr.mode_source, 'default') as mode_source,
                array_length(rs.chunk_ids, 1) as total_hits,
                COALESCE(rr.total_hits_before_cap, array_length(rs.chunk_ids, 1)) as total_before_cap,
                COALESCE(rr.cap_applied, false) as cap_applied,
                rr.similarity_threshold,
                rr.vector_metric_type,
                rr.retrieval_config_json
            FROM result_sets rs
            JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id
            WHERE rs.id = %s
        """, (result_set_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Result set not found")
        
        # Extract index retrieval fields from retrieval_config_json if present
        config = row[9] or {}
        
        # Compute expand_url for FIRST_* primitives
        expand_url = None
        primitive_type = config.get("primitive_type")
        if primitive_type and "FIRST" in str(primitive_type).upper():
            # For FIRST_* primitives, provide URL to expand to more results
            expand_url = f"/api/v1/result-sets/{result_set_id}/expand?limit=50"
        
        return ResultSetMetadata(
            result_set_id=row[0],
            retrieval_run_id=row[1],
            mode=row[2],
            mode_source=row[3],
            total_hits=row[4] or 0,
            total_hits_before_cap=row[5] or 0,
            cap_applied=row[6],
            threshold_used=row[7],
            vector_metric=row[8],
            # Index retrieval fields from config
            retrieval_source=config.get("retrieval_source", "search"),
            primitive_type=primitive_type,
            order_by=config.get("order_by"),
            time_basis=config.get("time_basis"),
            geo_basis=config.get("geo_basis"),
            expand_url=expand_url,
            # Determinism verification fields
            retrieval_signature_hash=config.get("retrieval_signature_hash"),
            chronology_version=config.get("chronology_version"),
        )


class ChunkItemWithContext(ChunkItem):
    """Chunk item with line context included."""
    line_before: Optional[str] = None
    highlight_line: Optional[str] = None
    line_after: Optional[str] = None


class ChunksResponseWithContext(BaseModel):
    """Response with line context included."""
    meta: ResultSetMetadata
    chunks: List[ChunkItemWithContext]
    pagination: PaginationInfo


@router.get("/{result_set_id}/chunks")
def get_result_set_chunks_paginated(
    result_set_id: int,
    user=Depends(require_user),
    offset: Optional[int] = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=500),
    after_rank: Optional[int] = Query(None, ge=0),
    render: Optional[str] = Query(None, description="Render mode: 'line_context' to include context"),
    year: Optional[int] = Query(None, description="Filter to chunks from this year"),
    place_id: Optional[int] = Query(None, description="Filter to chunks mentioning this place"),
):
    """
    Paginated chunk retrieval with dual pagination modes.
    
    Offset pagination: ?offset=0&limit=100
        - Simple, familiar
        - Slows down as offset grows
        
    Cursor pagination: ?after_rank=500&limit=100
        - Scalable for large result sets
        - Required for thorough mode with 10k+ results
        
    If both provided, after_rank takes precedence.
    
    Additional filters:
    - render=line_context: Include line context for each chunk
    - year=1944: Filter to chunks from specific year
    - place_id=123: Filter to chunks mentioning specific place
    """
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        meta = _get_result_set_metadata(conn, result_set_id)
        
        # Build query based on pagination mode
        params = {"result_set_id": result_set_id, "limit": limit + 1}  # +1 to check has_more
        
        # Base SELECT - include full text if render=line_context
        if render == "line_context":
            select_clause = """
                SELECT rsc.chunk_id, rsc.rank, rsc.document_id,
                       LEFT(COALESCE(c.clean_text, c.text), 500) as preview,
                       COALESCE(c.clean_text, c.text) as full_text
            """
        else:
            select_clause = """
                SELECT rsc.chunk_id, rsc.rank, rsc.document_id,
                       LEFT(COALESCE(c.clean_text, c.text), 500) as preview,
                       NULL as full_text
            """
        
        # Build WHERE clause with optional filters
        where_clauses = ["rsc.result_set_id = %(result_set_id)s"]
        
        if year is not None:
            where_clauses.append("EXTRACT(YEAR FROM cm.date_min) = %(year)s")
            params["year"] = year
        
        if place_id is not None:
            where_clauses.append("""
                EXISTS (
                    SELECT 1 FROM entity_mentions em
                    WHERE em.chunk_id = rsc.chunk_id AND em.entity_id = %(place_id)s
                )
            """)
            params["place_id"] = place_id
        
        # Add cursor/offset clause
        if after_rank is not None:
            where_clauses.append("rsc.rank > %(after_rank)s")
            params["after_rank"] = after_rank
        
        where_clause = " AND ".join(where_clauses)
        
        # Build full query
        needs_cm_join = year is not None
        
        if needs_cm_join:
            query = f"""
                {select_clause}
                FROM result_set_chunks rsc
                JOIN chunks c ON c.id = rsc.chunk_id
                LEFT JOIN chunk_metadata cm ON cm.chunk_id = rsc.chunk_id
                WHERE {where_clause}
                ORDER BY rsc.rank
                LIMIT %(limit)s
            """
        else:
            query = f"""
                {select_clause}
                FROM result_set_chunks rsc
                JOIN chunks c ON c.id = rsc.chunk_id
                WHERE {where_clause}
                ORDER BY rsc.rank
                LIMIT %(limit)s
            """
        
        # Add offset if no cursor
        if after_rank is None:
            if offset is None:
                offset = 0
            query += " OFFSET %(offset)s"
            params["offset"] = offset
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        
        # Check if there are more results
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]
        
        # Build chunk items
        if render == "line_context":
            chunks = []
            for r in rows:
                chunk_id, rank, document_id, preview, full_text = r
                
                # Extract line context if we have full text
                line_before, highlight_line, line_after = None, None, None
                if full_text:
                    lines = full_text.split('\n')
                    if lines:
                        # For now, show first line as highlight and context around it
                        highlight_line = lines[0] if lines else None
                        line_after = lines[1] if len(lines) > 1 else None
                
                chunks.append(ChunkItemWithContext(
                    chunk_id=chunk_id,
                    rank=rank,
                    document_id=document_id,
                    preview=preview,
                    line_before=line_before,
                    highlight_line=highlight_line,
                    line_after=line_after,
                ))
        else:
            chunks = [
                ChunkItem(
                    chunk_id=r[0],
                    rank=r[1],
                    document_id=r[2],
                    preview=r[3],
                )
                for r in rows
            ]
        
        # Build pagination info
        next_cursor = chunks[-1].rank if chunks and has_more else None
        
        if render == "line_context":
            return ChunksResponseWithContext(
                meta=meta,
                chunks=chunks,
                pagination=PaginationInfo(
                    offset=offset,
                    after_rank=after_rank,
                    limit=limit,
                    has_more=has_more,
                    next_cursor=next_cursor,
                ),
            )
        else:
            return ChunksResponse(
                meta=meta,
                chunks=chunks,
                pagination=PaginationInfo(
                    offset=offset,
                    after_rank=after_rank,
                    limit=limit,
                    has_more=has_more,
                    next_cursor=next_cursor,
                ),
            )
    finally:
        conn.close()


class ExpandResponse(BaseModel):
    """Response for expand endpoint."""
    message: str
    original_result_set_id: int
    expanded_result_set_id: Optional[int] = None
    chunks_added: int = 0
    error: Optional[str] = None
    # Semantic preservation proof
    preserved_config: Optional[dict] = None


@router.get("/{result_set_id}/expand")
def expand_result_set(
    result_set_id: int,
    user=Depends(require_user),
    limit: int = Query(50, ge=1, le=500, description="Number of additional results"),
):
    """
    Expand a FIRST_* result set to show more results.
    
    For FIRST_MENTION, FIRST_CO_MENTION, FIRST_DATE_MENTION - this retrieves
    additional chronologically-ordered results using the SAME stored config.
    
    IMPORTANT: This endpoint preserves semantic consistency by:
    - Using the exact same retrieval_config_json from the original run
    - Preserving scope (collections, filters)
    - Preserving time_basis / geo_basis
    - Preserving dedupe_policy / span_policy
    - Preserving chronology order version
    
    Returns:
        Expanded result set info with preserved_config showing what was reused.
    """
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        meta = _get_result_set_metadata(conn, result_set_id)
        
        # Check if this is an expandable primitive type
        primitive_type = meta.primitive_type
        if not primitive_type or "FIRST" not in str(primitive_type).upper():
            return ExpandResponse(
                message="Expansion not supported for this result set type",
                original_result_set_id=result_set_id,
                error=f"Primitive type '{primitive_type}' is not a FIRST_* type",
            )
        
        # Get the FULL retrieval config to ensure deterministic expansion
        with conn.cursor() as cur:
            cur.execute("""
                SELECT rr.retrieval_config_json, rr.id as run_id
                FROM result_sets rs
                JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id
                WHERE rs.id = %s
            """, (result_set_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Result set not found")
            
            config = row[0] or {}
            original_run_id = row[1]
        
        # Build the preserved config proof (what semantics are preserved)
        preserved = {
            "source": config.get("source"),
            "order_by": config.get("order_by"),
            "time_basis": config.get("time_basis"),
            "geo_basis": config.get("geo_basis"),
            "dedupe_policy": config.get("dedupe_policy"),
            "span_policy": config.get("span_policy"),
            "original_primitive": primitive_type,
            "original_run_id": original_run_id,
        }
        
        # For now, return the preserved config to prove determinism
        # Full implementation would re-run with higher limit using same config
        return ExpandResponse(
            message=f"Expansion ready for {primitive_type} - would use stored config",
            original_result_set_id=result_set_id,
            chunks_added=0,  # Would be populated after actual expansion
            preserved_config=preserved,
        )
        
    finally:
        conn.close()


@router.get("/{result_set_id}/match-traces", response_model=MatchTracesResponse)
def get_match_traces(
    result_set_id: int,
    user=Depends(require_user),
    chunk_ids: Optional[str] = Query(None, description="Comma-separated chunk IDs"),
    include_audit: bool = Query(False),
):
    """
    Get match traces explaining why chunks surfaced.
    
    Default: returns hot columns only (fast).
    With include_audit=True: returns full primitive_matches JSONB.
    """
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        meta = _get_result_set_metadata(conn, result_set_id)
        
        params = {"result_set_id": result_set_id}
        query = """
            SELECT 
                chunk_id, matched_entity_ids, matched_phrases,
                scope_passed, in_lexical, in_vector,
                score_lexical, score_vector, score_hybrid,
                rank, rank_trace, primitive_matches
            FROM result_set_match_traces
            WHERE result_set_id = %(result_set_id)s
        """
        
        if chunk_ids:
            chunk_id_list = [int(x.strip()) for x in chunk_ids.split(",")]
            query += " AND chunk_id = ANY(%(chunk_ids)s)"
            params["chunk_ids"] = chunk_id_list
        
        query += " ORDER BY rank NULLS LAST, chunk_id"
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        
        traces = []
        for row in rows:
            (chunk_id, entity_ids, phrases, scope_passed, in_lex, in_vec,
             score_lex, score_vec, score_hybrid, rank, rank_trace, prim_matches) = row
            
            trace = MatchTraceItem(
                chunk_id=chunk_id,
                matched_entity_ids=list(entity_ids) if entity_ids else [],
                matched_phrases=list(phrases) if phrases else [],
                scope_passed=scope_passed,
                in_lexical=in_lex,
                in_vector=in_vec,
                scores=ScoreBreakdown(
                    lexical=score_lex,
                    vector=score_vec,
                    hybrid=score_hybrid,
                ),
                rank=rank,
                rank_explanation=str(rank_trace) if rank_trace else None,
                primitive_matches=list(prim_matches) if include_audit and prim_matches else None,
            )
            traces.append(trace)
        
        return MatchTracesResponse(meta=meta, traces=traces)
    finally:
        conn.close()


@router.get("/chunks/{chunk_id}/highlights", response_model=HighlightsResponse)
def get_chunk_highlights(
    chunk_id: int,
    user=Depends(require_user),
    phrases: str = Query(..., description="Comma-separated phrases to highlight"),
    context_chars: int = Query(50, ge=0, le=200),
    case_sensitive: bool = Query(False),
):
    """
    Tier 3: Compute phrase positions on-demand.
    
    Called when user expands a chunk in the UI, NOT during retrieval.
    Loads single chunk text and finds all phrase occurrences.
    
    Behavior:
    - Case-insensitive by default
    - Returns ALL occurrences of each phrase
    - Matches ordered by position (start ASC)
    - Context window capped at context_chars (max 200)
    """
    import sys
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    
    from retrieval.match_trace import get_phrase_positions_on_demand
    
    conn = get_conn()
    try:
        phrase_list = [p.strip() for p in phrases.split(",") if p.strip()]
        
        if not phrase_list:
            raise HTTPException(status_code=400, detail="At least one phrase required")
        
        matches = get_phrase_positions_on_demand(
            chunk_id, phrase_list, conn,
            case_sensitive=case_sensitive,
            context_chars=context_chars,
        )
        
        # Get text length
        with conn.cursor() as cur:
            cur.execute(
                "SELECT LENGTH(COALESCE(clean_text, text)) FROM chunks WHERE id = %s",
                (chunk_id,)
            )
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Chunk not found")
            text_length = result[0] or 0
        
        return HighlightsResponse(
            chunk_id=chunk_id,
            text_length=text_length,
            matches=[
                HighlightMatch(
                    phrase=m["phrase"],
                    start=m["start"],
                    end=m["end"],
                    context=m["context"],
                )
                for m in matches
            ],
        )
    finally:
        conn.close()


@router.get("/chunks/{chunk_id}/line-context", response_model=LineContextResponse)
def get_line_context(
    chunk_id: int,
    user=Depends(require_user),
    start_char: int = Query(..., ge=0, description="Start character offset in chunk text"),
    end_char: int = Query(..., ge=0, description="End character offset in chunk text"),
    lines_before: int = Query(1, ge=0, le=5, description="Number of lines before highlight"),
    lines_after: int = Query(1, ge=0, le=5, description="Number of lines after highlight"),
):
    """
    Get line-level context around a character span.
    
    Returns the line containing the highlighted span, plus optional lines
    before and after for context. Useful for displaying mention context
    in the UI without loading the full chunk text.
    
    The endpoint also returns page information for deep linking to the
    source document.
    """
    conn = get_conn()
    try:
        # Get chunk text and metadata
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    c.id,
                    COALESCE(c.clean_text, c.text) as text,
                    cm.document_id,
                    cp.page_id,
                    p.logical_page_label,
                    p.pdf_page_number
                FROM chunks c
                LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
                LEFT JOIN chunk_pages cp ON cp.chunk_id = c.id AND cp.span_order = 1
                LEFT JOIN pages p ON p.id = cp.page_id
                WHERE c.id = %s
            """, (chunk_id,))
            row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        _, text, document_id, page_id, page_label, pdf_page_number = row
        
        if not text:
            raise HTTPException(status_code=400, detail="Chunk has no text")
        
        # Validate character offsets
        text_length = len(text)
        if start_char >= text_length or end_char > text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Character offset out of range. Chunk has {text_length} characters."
            )
        if start_char >= end_char:
            raise HTTPException(
                status_code=400,
                detail="start_char must be less than end_char"
            )
        
        # Split text into lines
        lines = text.split('\n')
        
        # Find which line(s) contain the highlight
        char_count = 0
        highlight_line_idx = None
        highlight_line_start = 0
        
        for i, line in enumerate(lines):
            line_start = char_count
            line_end = char_count + len(line)
            
            # Check if highlight starts in this line
            if line_start <= start_char < line_end + 1:  # +1 for newline
                highlight_line_idx = i
                highlight_line_start = line_start
                break
            
            char_count = line_end + 1  # +1 for newline
        
        if highlight_line_idx is None:
            # Fallback: highlight is in last line or at end
            highlight_line_idx = len(lines) - 1
            highlight_line_start = text_length - len(lines[-1])
        
        # Extract lines
        highlight_line = lines[highlight_line_idx]
        
        line_before = None
        if lines_before > 0 and highlight_line_idx > 0:
            before_start = max(0, highlight_line_idx - lines_before)
            line_before = '\n'.join(lines[before_start:highlight_line_idx])
        
        line_after = None
        if lines_after > 0 and highlight_line_idx < len(lines) - 1:
            after_end = min(len(lines), highlight_line_idx + 1 + lines_after)
            line_after = '\n'.join(lines[highlight_line_idx + 1:after_end])
        
        return LineContextResponse(
            chunk_id=chunk_id,
            page_id=page_id,
            document_id=document_id,
            highlight_start=start_char,
            highlight_end=end_char,
            line_before=line_before,
            highlight_line=highlight_line,
            line_after=line_after,
            page_label=page_label,
            pdf_page_number=pdf_page_number,
        )
    finally:
        conn.close()


@router.get("/{result_set_id}/entities", response_model=EntitiesResponse)
def get_result_set_entities(
    result_set_id: int,
    user=Depends(require_user),
    limit: int = Query(50, ge=1, le=200),
    entity_type: Optional[str] = Query(None),
):
    """Top entities mentioned in this result set."""
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        meta = _get_result_set_metadata(conn, result_set_id)
        
        params = {"result_set_id": result_set_id, "limit": limit}
        query = """
            SELECT e.id, e.canonical_name, e.entity_type, COUNT(*) as mention_count
            FROM result_sets rs
            CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
            JOIN entity_mentions em ON em.chunk_id = chunk_id
            JOIN entities e ON e.id = em.entity_id
            WHERE rs.id = %(result_set_id)s
        """
        
        if entity_type:
            query += " AND e.entity_type = %(entity_type)s"
            params["entity_type"] = entity_type
        
        query += """
            GROUP BY e.id, e.canonical_name, e.entity_type
            ORDER BY mention_count DESC
            LIMIT %(limit)s
        """
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        
        entities = [
            EntityItem(
                entity_id=r[0],
                name=r[1],
                entity_type=r[2],
                mention_count=r[3],
            )
            for r in rows
        ]
        
        return EntitiesResponse(meta=meta, entities=entities)
    finally:
        conn.close()


@router.get("/{result_set_id}/co-mentions", response_model=CoMentionsResponse)
def get_co_mentioned_entities(
    result_set_id: int,
    user=Depends(require_user),
    entity_id: int = Query(...),
    window: str = Query("document", regex="^(chunk|document)$"),
    limit: int = Query(20, ge=1, le=100),
):
    """Entities that co-occur with target entity within result set."""
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        meta = _get_result_set_metadata(conn, result_set_id)
        
        params = {
            "result_set_id": result_set_id,
            "entity_id": entity_id,
            "limit": limit,
        }
        
        if window == "chunk":
            # Chunk-level co-occurrence
            query = """
                SELECT e.id, e.canonical_name, e.entity_type, COUNT(DISTINCT em2.chunk_id) as cooccur_count
                FROM result_sets rs
                CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
                JOIN entity_mentions em1 ON em1.chunk_id = chunk_id AND em1.entity_id = %(entity_id)s
                JOIN entity_mentions em2 ON em2.chunk_id = em1.chunk_id AND em2.entity_id != em1.entity_id
                JOIN entities e ON e.id = em2.entity_id
                WHERE rs.id = %(result_set_id)s
                GROUP BY e.id, e.canonical_name, e.entity_type
                ORDER BY cooccur_count DESC
                LIMIT %(limit)s
            """
        else:
            # Document-level co-occurrence
            query = """
                SELECT e.id, e.canonical_name, e.entity_type, COUNT(DISTINCT em2.document_id) as cooccur_count
                FROM result_sets rs
                CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
                JOIN entity_mentions em1 ON em1.chunk_id = chunk_id AND em1.entity_id = %(entity_id)s
                JOIN entity_mentions em2 ON em2.document_id = em1.document_id AND em2.entity_id != em1.entity_id
                JOIN entities e ON e.id = em2.entity_id
                WHERE rs.id = %(result_set_id)s
                GROUP BY e.id, e.canonical_name, e.entity_type
                ORDER BY cooccur_count DESC
                LIMIT %(limit)s
            """
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        
        co_mentions = [
            CoMentionItem(
                entity_id=r[0],
                name=r[1],
                entity_type=r[2],
                cooccurrence_count=r[3],
            )
            for r in rows
        ]
        
        return CoMentionsResponse(
            meta=meta,
            target_entity_id=entity_id,
            window=window,
            co_mentions=co_mentions,
        )
    finally:
        conn.close()


@router.get("/{result_set_id}/date-facets", response_model=DateFacetsResponse)
def get_date_facets(
    result_set_id: int,
    user=Depends(require_user),
    granularity: str = Query("year", regex="^(year|month)$"),
    time_basis: str = Query("mentioned_date", regex="^(mentioned_date|document_date)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Aggregate date mentions in a result set into histogram buckets.
    
    - granularity: "year" or "month" 
    - time_basis: "mentioned_date" (dates in text) or "document_date" (chunk metadata)
    
    Returns buckets ordered by date (earliest first).
    """
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        meta = _get_result_set_metadata(conn, result_set_id)
        
        params = {"result_set_id": result_set_id, "limit": limit}
        
        if time_basis == "mentioned_date":
            if granularity == "year":
                query = """
                    SELECT 
                        EXTRACT(YEAR FROM dm.date_start)::int as year,
                        NULL as month,
                        COUNT(DISTINCT dm.chunk_id) as chunk_count,
                        COUNT(*) as mention_count
                    FROM result_sets rs
                    CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
                    JOIN date_mentions dm ON dm.chunk_id = chunk_id
                    WHERE rs.id = %(result_set_id)s
                        AND dm.date_start IS NOT NULL
                    GROUP BY EXTRACT(YEAR FROM dm.date_start)
                    ORDER BY year ASC
                    LIMIT %(limit)s
                """
            else:  # month
                query = """
                    SELECT 
                        EXTRACT(YEAR FROM dm.date_start)::int as year,
                        EXTRACT(MONTH FROM dm.date_start)::int as month,
                        COUNT(DISTINCT dm.chunk_id) as chunk_count,
                        COUNT(*) as mention_count
                    FROM result_sets rs
                    CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
                    JOIN date_mentions dm ON dm.chunk_id = chunk_id
                    WHERE rs.id = %(result_set_id)s
                        AND dm.date_start IS NOT NULL
                    GROUP BY EXTRACT(YEAR FROM dm.date_start), EXTRACT(MONTH FROM dm.date_start)
                    ORDER BY year ASC, month ASC
                    LIMIT %(limit)s
                """
        else:  # document_date
            if granularity == "year":
                query = """
                    SELECT 
                        EXTRACT(YEAR FROM cm.date_min)::int as year,
                        NULL as month,
                        COUNT(DISTINCT cm.chunk_id) as chunk_count,
                        COUNT(*) as mention_count
                    FROM result_sets rs
                    CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
                    JOIN chunk_metadata cm ON cm.chunk_id = chunk_id
                    WHERE rs.id = %(result_set_id)s
                        AND cm.date_min IS NOT NULL
                    GROUP BY EXTRACT(YEAR FROM cm.date_min)
                    ORDER BY year ASC
                    LIMIT %(limit)s
                """
            else:  # month
                query = """
                    SELECT 
                        EXTRACT(YEAR FROM cm.date_min)::int as year,
                        EXTRACT(MONTH FROM cm.date_min)::int as month,
                        COUNT(DISTINCT cm.chunk_id) as chunk_count,
                        COUNT(*) as mention_count
                    FROM result_sets rs
                    CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
                    JOIN chunk_metadata cm ON cm.chunk_id = chunk_id
                    WHERE rs.id = %(result_set_id)s
                        AND cm.date_min IS NOT NULL
                    GROUP BY EXTRACT(YEAR FROM cm.date_min), EXTRACT(MONTH FROM cm.date_min)
                    ORDER BY year ASC, month ASC
                    LIMIT %(limit)s
                """
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        
        buckets = [
            DateBucket(
                year=r[0],
                month=r[1],
                chunk_count=r[2],
                date_mention_count=r[3],
            )
            for r in rows
        ]
        
        return DateFacetsResponse(
            meta=meta,
            granularity=granularity,
            time_basis=time_basis,
            buckets=buckets,
        )
    finally:
        conn.close()


@router.get("/{result_set_id}/place-facets", response_model=PlaceFacetsResponse)
def get_place_facets(
    result_set_id: int,
    user=Depends(require_user),
    limit: int = Query(30, ge=1, le=100),
):
    """
    Aggregate place mentions in a result set.
    
    Returns top places mentioned in the result set, ordered by mention count.
    """
    assert_result_set_owned(result_set_id, user["sub"])
    conn = get_conn()
    try:
        meta = _get_result_set_metadata(conn, result_set_id)
        
        params = {"result_set_id": result_set_id, "limit": limit}
        
        query = """
            SELECT 
                e.id as place_entity_id,
                e.canonical_name as place_name,
                NULL as country,
                COUNT(DISTINCT em.chunk_id) as chunk_count,
                COUNT(*) as mention_count
            FROM result_sets rs
            CROSS JOIN LATERAL unnest(rs.chunk_ids) AS chunk_id
            JOIN entity_mentions em ON em.chunk_id = chunk_id
            JOIN entities e ON e.id = em.entity_id AND e.entity_type = 'place'
            WHERE rs.id = %(result_set_id)s
            GROUP BY e.id, e.canonical_name
            ORDER BY mention_count DESC
            LIMIT %(limit)s
        """
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        
        buckets = [
            PlaceBucket(
                place_entity_id=r[0],
                place_name=r[1],
                country=r[2],
                chunk_count=r[3],
                mention_count=r[4],
            )
            for r in rows
        ]
        
        return PlaceFacetsResponse(
            meta=meta,
            buckets=buckets,
        )
    finally:
        conn.close()
