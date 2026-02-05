"""
Chat endpoint - V7 Agentic Workflow Integration

This endpoint provides a chat-style interface using the V7 retrieval workflow.
V7 features (extends V6):
- All V6 features (CONTROL/CONTENT parsing, entity linking, bottleneck, etc.)
- Citation enforcement: every claim MUST have citations
- Claim enumeration: extracts atomic claims from answer
- Stop gate validation: verifies all claims are grounded
- Expanded summary: output includes claims with their citations
"""
import sys
import os
import time
import json
import queue
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Generator, Tuple
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from psycopg2.extras import Json

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.db import get_conn

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request to send a chat message."""
    message: str
    pre_bundling_mode: str = "off"  # "off", "passthrough", "micro", "semantic"


class Citation(BaseModel):
    """A citation reference for a claim."""
    span_id: str
    chunk_id: Optional[int] = None
    document_id: Optional[int] = None
    page_number: Optional[int] = None
    quote: str
    source_name: Optional[str] = None
    relevance_score: Optional[float] = None


class Claim(BaseModel):
    """A claim with evidence citations."""
    text: str
    confidence: str = "supported"
    citations: List[Citation] = []


class Member(BaseModel):
    """An identified member (for roster queries)."""
    name: str
    citations: List[Citation] = []


class WorkflowAction(BaseModel):
    """A single action/step in the V6 workflow."""
    step: str  # e.g., "query_parsing", "entity_linking", "retrieval", "bottleneck"
    status: str  # "running", "completed", "skipped"
    message: str
    details: Optional[Dict[str, Any]] = None
    elapsed_ms: Optional[float] = None


class V7Stats(BaseModel):
    """Statistics about the V7 workflow execution."""
    task_type: str
    rounds_executed: int
    total_spans: int
    unique_docs: int
    elapsed_ms: float
    entity_linking: Dict[str, Any] = {}
    responsiveness: str
    actions: List[WorkflowAction] = []  # Track all workflow actions
    # V7-specific fields
    citation_validation_passed: bool = False
    claims_extracted: int = 0
    claims_valid: int = 0
    claims_dropped: int = 0


class ChatMessage(BaseModel):
    """A message in the chat history."""
    id: int
    session_id: int
    role: str  # 'user' | 'assistant'
    content: str
    claims: Optional[List[Claim]] = None
    members: Optional[List[Member]] = None
    v7_stats: Optional[V7Stats] = None
    result_set_id: Optional[int] = None
    created_at: datetime


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    user_message: ChatMessage
    assistant_message: ChatMessage
    is_responsive: bool
    result_set_id: Optional[int] = None


# =============================================================================
# V6 Workflow Execution
# =============================================================================

def _parse_page_number(page: Any) -> Optional[int]:
    """Parse page number from various formats."""
    if page is None:
        return None
    if isinstance(page, int):
        return page
    if isinstance(page, str):
        # Handle formats like "p.5", "page 5", "5", etc.
        import re
        match = re.search(r'\d+', str(page))
        if match:
            return int(match.group())
    return None




def run_v7_chat_query(conn, session_id: int, question: str, pre_bundling_mode: str = "off") -> Dict[str, Any]:
    """
    Run the V7 workflow (V6 + citation enforcement) and return structured results.
    
    Args:
        conn: Database connection
        session_id: Session ID for result set creation
        question: User's question
        pre_bundling_mode: Pre-bundling mode for concordance-aware evidence grouping
    
    Returns:
        Dictionary with answer, claims, members, stats, and result_set_id
    """
    try:
        from retrieval.agent.v7_runner import run_v7_query
        from retrieval.ops import log_retrieval_run
        from scripts.execute_plan import create_result_set
    except ImportError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"V7 workflow dependencies not available: {e}"
        )
    
    start_time = time.time()
    
    # Check for required environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable is not set. The V7 workflow requires an OpenAI API key for LLM-powered retrieval."
        )
    
    # Quick sanity check on API key format
    if not api_key.startswith("sk-"):
        raise HTTPException(
            status_code=500,
            detail=f"OPENAI_API_KEY appears malformed (should start with 'sk-', got '{api_key[:10]}...')"
        )
    
    # Run V7 workflow with verbose=True so we can see what's happening in logs
    result = run_v7_query(
        conn=conn,
        question=question,
        max_bottleneck_spans=40,
        max_rounds=5,
        drop_uncited_claims=True,
        verbose=True,  # Enable logging to see what's happening
        pre_bundling_mode=pre_bundling_mode,
    )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Build workflow actions from V7 trace (using V6 trace inside)
    v6_trace = result.trace.v6_trace if result.trace else None
    actions = build_workflow_actions_v7(result, elapsed_ms) if result.trace else []
    
    # Get span list from V6 trace's bottleneck spans
    span_list = v6_trace.bottleneck_spans if v6_trace else []
    
    def _build_citation_from_span(span, span_idx: int) -> Citation:
        """Build a Citation from a BottleneckSpan object."""
        if hasattr(span, 'chunk_id'):
            # It's a BottleneckSpan object
            return Citation(
                span_id=str(span_idx),
                chunk_id=span.chunk_id,
                document_id=span.doc_id,
                page_number=_parse_page_number(span.page),
                quote=span.span_text[:200] if hasattr(span, 'span_text') else "",
                source_name=span.source_label if hasattr(span, 'source_label') else None,
                relevance_score=span.relevance_score if hasattr(span, 'relevance_score') else None,
            )
        elif isinstance(span, dict):
            # It's a dict
            return Citation(
                span_id=str(span_idx),
                chunk_id=span.get('chunk_id'),
                document_id=span.get('doc_id'),
                page_number=_parse_page_number(span.get('page')),
                quote=span.get('span_text', '')[:200],
                source_name=span.get('source_label'),
                relevance_score=span.get('relevance_score'),
            )
        else:
            return Citation(span_id=str(span_idx), quote="")
    
    # Build a span_id -> span mapping for efficient lookup
    # V7 uses span_ids like "sp_0", "sp_1" or "bs_123" (BottleneckSpan format)
    span_by_id: Dict[str, Tuple[Any, int]] = {}
    for idx, span in enumerate(span_list):
        # BottleneckSpan uses "bs_<chunk_id>" format
        if hasattr(span, 'span_id'):
            span_by_id[span.span_id] = (span, idx)
        # Also register with sp_<idx> format for V7 claim enumerator fallback
        span_by_id[f"sp_{idx}"] = (span, idx)
        # And with bs_<idx> for index-based lookup
        span_by_id[f"bs_{idx}"] = (span, idx)
    
    # Build claims list from V7 result's expanded summary
    claims = []
    if result.expanded_summary and result.expanded_summary.claims:
        for claim_obj in result.expanded_summary.claims[:20]:  # Limit to 20 claims
            citations = []
            # V7 claims have citation IDs (span_ids)
            for cit_id in claim_obj.citations[:3]:  # Max 3 citations per claim
                # Try to find the span by ID
                if cit_id in span_by_id:
                    span, idx = span_by_id[cit_id]
                    citations.append(_build_citation_from_span(span, idx))
                else:
                    # Try numeric extraction: "sp_5" -> index 5
                    found = False
                    try:
                        if cit_id.startswith("sp_"):
                            idx = int(cit_id[3:])
                            if 0 <= idx < len(span_list):
                                citations.append(_build_citation_from_span(span_list[idx], idx))
                                found = True
                        elif cit_id.startswith("bs_"):
                            idx = int(cit_id[3:])
                            if 0 <= idx < len(span_list):
                                citations.append(_build_citation_from_span(span_list[idx], idx))
                                found = True
                    except (ValueError, IndexError):
                        pass
                    if not found:
                        # Fallback: empty citation with ID
                        citations.append(Citation(span_id=cit_id, quote=""))
            
            claims.append(Claim(
                text=claim_obj.claim_text,
                confidence=claim_obj.support_level,
                citations=citations,
            ))
    
    # Build members list from V7 result
    member_citations: Dict[str, List[Citation]] = {}
    for idx, span in enumerate(span_list):
        member_name = None
        if hasattr(span, 'member_name') and span.member_name:
            member_name = span.member_name
        elif isinstance(span, dict) and span.get('member_name'):
            member_name = span['member_name']
        
        if member_name:
            if member_name not in member_citations:
                member_citations[member_name] = []
            if len(member_citations[member_name]) < 3:  # Max 3 citations per member
                member_citations[member_name].append(_build_citation_from_span(span, idx))
    
    members = []
    for member_name in result.members_identified[:30]:
        cits = member_citations.get(member_name, [])
        members.append(Member(name=member_name, citations=cits))
    
    # Calculate total spans from the last round's total_spans
    total_spans = 0
    if v6_trace and v6_trace.rounds:
        last_round = v6_trace.rounds[-1]
        if isinstance(last_round, dict):
            total_spans = last_round.get("total_spans", 0)
    
    # Build stats
    stats = V7Stats(
        task_type=v6_trace.parsed_query.task_type.value if v6_trace and v6_trace.parsed_query else "unknown",
        rounds_executed=len(v6_trace.rounds) if v6_trace else 0,
        total_spans=total_spans,
        unique_docs=0,  # Would need to calculate
        elapsed_ms=elapsed_ms,
        entity_linking={
            "total_linked": v6_trace.entity_linking.total_linked if v6_trace and v6_trace.entity_linking else 0,
            "used_for_retrieval": v6_trace.entity_linking.used_for_retrieval if v6_trace and v6_trace.entity_linking else 0,
        } if v6_trace else {},
        responsiveness=result.responsiveness_status,
        actions=actions,
        # V7-specific fields
        citation_validation_passed=result.citation_validation_passed,
        claims_extracted=result.trace.claims_extracted if result.trace else 0,
        claims_valid=result.trace.claims_valid if result.trace else 0,
        claims_dropped=result.trace.claims_unsupported if result.trace else 0,
    )
    
    result_set_id = None
    
    return {
        "answer": result.answer,
        "claims": claims,
        "members": members,
        "stats": stats,
        "is_responsive": result.citation_validation_passed,  # V7 uses citation validation
        "result_set_id": result_set_id,
    }


def build_workflow_actions_v7(result, total_elapsed_ms: float) -> List[WorkflowAction]:
    """
    Build a list of workflow actions from the V7 result trace.
    
    This creates a CLI-like log of what the V7 workflow did.
    """
    actions = []
    trace = result.trace
    v6_trace = trace.v6_trace if trace else None
    
    if not v6_trace:
        return actions
    
    # Step 1: Query Parsing
    if v6_trace.parsed_query:
        pq = v6_trace.parsed_query
        topic_terms = pq.topic_terms if hasattr(pq, 'topic_terms') else []
        control_count = len(pq.control_tokens) if hasattr(pq, 'control_tokens') else 0
        task_type = pq.task_type.value if hasattr(pq, 'task_type') else "unknown"
        
        actions.append(WorkflowAction(
            step="query_parsing",
            status="completed",
            message=f"Parsed query as {task_type} task",
            details={
                "task_type": task_type,
                "topic_terms": topic_terms[:5],
                "control_tokens_filtered": control_count,
            }
        ))
    
    # Step 2: Entity Linking
    if v6_trace.entity_linking:
        el = v6_trace.entity_linking
        total_linked = el.total_linked if hasattr(el, 'total_linked') else 0
        used = el.used_for_retrieval if hasattr(el, 'used_for_retrieval') else 0
        rejected = el.rejected_control_tokens if hasattr(el, 'rejected_control_tokens') else 0
        
        # Get linked entity details
        retrieval_entities = []
        if hasattr(el, 'retrieval_entities'):
            for e in el.retrieval_entities[:5]:
                retrieval_entities.append({
                    "name": e.canonical_name if hasattr(e, 'canonical_name') else str(e),
                    "id": e.entity_id if hasattr(e, 'entity_id') else None,
                })
        
        actions.append(WorkflowAction(
            step="entity_linking",
            status="completed",
            message=f"Linked {total_linked} entities, {used} used for retrieval",
            details={
                "total_linked": total_linked,
                "used_for_retrieval": used,
                "rejected_control_tokens": rejected,
                "retrieval_entities": retrieval_entities,
            }
        ))
    
    # Step 3: Retrieval Rounds
    for i, round_data in enumerate(v6_trace.rounds):
        round_num = i + 1
        if isinstance(round_data, dict):
            chunks = round_data.get("chunks_retrieved", 0)
            spans = round_data.get("spans_after_bottleneck", 0)
            total_spans = round_data.get("total_spans", 0)
            members = round_data.get("members_found", [])
            
            actions.append(WorkflowAction(
                step=f"retrieval_round_{round_num}",
                status="completed",
                message=f"Round {round_num}: {chunks} chunks → {spans} new spans ({total_spans} total)",
                details={
                    "round": round_num,
                    "chunks_retrieved": chunks,
                    "spans_after_bottleneck": spans,
                    "total_spans": total_spans,
                    "members_found": len(members) if isinstance(members, list) else 0,
                }
            ))
    
    # Step 4: Synthesis (V6)
    claims_count = len(v6_trace.final_claims) if v6_trace.final_claims else 0
    actions.append(WorkflowAction(
        step="synthesis",
        status="completed",
        message=f"Generated answer with {claims_count} claims",
        details={
            "claims_count": claims_count,
        }
    ))
    
    # Step 5: V7 Citation Enforcement
    if trace:
        actions.append(WorkflowAction(
            step="claim_enumeration",
            status="completed",
            message=f"Extracted {trace.claims_extracted} claims",
            details={
                "claims_extracted": trace.claims_extracted,
                "claims_valid": trace.claims_valid,
                "claims_unsupported": trace.claims_unsupported,
            }
        ))
        
        stop_gate_status = "passed" if trace.stop_gate_passed else "failed"
        actions.append(WorkflowAction(
            step="stop_gate",
            status="completed",
            message=f"Citation validation: {stop_gate_status}",
            details={
                "passed": trace.stop_gate_passed,
                "reason": trace.stop_gate_reason,
            }
        ))
    
    # Final summary action
    actions.append(WorkflowAction(
        step="complete",
        status="completed",
        message=f"V7 workflow complete in {total_elapsed_ms:.0f}ms",
        elapsed_ms=total_elapsed_ms,
        details={
            "citation_validation_passed": result.citation_validation_passed,
            "claims_valid": trace.claims_valid if trace else 0,
            "claims_dropped": trace.claims_unsupported if trace else 0,
        }
    ))
    
    return actions


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: int, req: ChatRequest):
    """
    Send a chat message and get a V7-powered response.
    
    This is the main interaction endpoint for the research console.
    It uses the V7 retrieval workflow which features:
    - All V6 features (CONTROL/CONTENT parsing, entity linking, bottleneck)
    - Citation enforcement: every claim must have citations
    - Claim enumeration and stop gate validation
    - Expanded summary with claims & citations
    
    Args:
        session_id: The session to chat in
        req: The chat request with the user's message
    
    Returns:
        ChatResponse with user message, assistant response, and V7 stats
    """
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Verify session exists
            cur.execute("SELECT 1 FROM research_sessions WHERE id = %s", (session_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Store user message
            cur.execute(
                """
                INSERT INTO research_messages (session_id, role, content)
                VALUES (%s, 'user', %s)
                RETURNING id, session_id, role, content, created_at
                """,
                (session_id, message),
            )
            user_row = cur.fetchone()
            user_message = ChatMessage(
                id=user_row[0],
                session_id=user_row[1],
                role=user_row[2],
                content=user_row[3],
                created_at=user_row[4],
            )
            conn.commit()
        
        # Run V7 workflow
        v7_result = run_v7_chat_query(conn, session_id, message, pre_bundling_mode=req.pre_bundling_mode)
        
        with conn.cursor() as cur:
            # Store assistant message with V7 results
            metadata = {
                "v7_stats": v7_result["stats"].model_dump(),
                "claims_count": len(v7_result["claims"]),
                "members_count": len(v7_result["members"]),
                "is_responsive": v7_result["is_responsive"],
                "citation_validation_passed": v7_result["stats"].citation_validation_passed,
                "pre_bundling_mode": req.pre_bundling_mode,
            }
            
            cur.execute(
                """
                INSERT INTO research_messages (session_id, role, content, result_set_id, metadata)
                VALUES (%s, 'assistant', %s, %s, %s)
                RETURNING id, session_id, role, content, result_set_id, created_at
                """,
                (session_id, v7_result["answer"], v7_result["result_set_id"], Json(metadata)),
            )
            asst_row = cur.fetchone()
            assistant_message = ChatMessage(
                id=asst_row[0],
                session_id=asst_row[1],
                role=asst_row[2],
                content=asst_row[3],
                claims=v7_result["claims"],
                members=v7_result["members"],
                v7_stats=v7_result["stats"],
                result_set_id=asst_row[4],
                created_at=asst_row[5],
            )
            conn.commit()
        
        return ChatResponse(
            user_message=user_message,
            assistant_message=assistant_message,
            is_responsive=v7_result["is_responsive"],
            result_set_id=v7_result["result_set_id"],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"V7 workflow error: {str(e)}"
        )
    finally:
        conn.close()


@router.get("/{session_id}/chat/history", response_model=List[ChatMessage])
def get_chat_history(session_id: int):
    """
    Get the chat history for a session.
    
    Returns messages with their V7 metadata (claims, members, stats) reconstructed
    from the stored metadata.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Verify session exists
            cur.execute("SELECT 1 FROM research_sessions WHERE id = %s", (session_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Session not found")
            
            cur.execute(
                """
                SELECT id, session_id, role, content, result_set_id, metadata, created_at
                FROM research_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            
            messages = []
            for row in rows:
                metadata = row[5] or {}
                
                # Reconstruct V7 stats from metadata if available
                v7_stats = None
                if "v7_stats" in metadata:
                    v7_stats = V7Stats(**metadata["v7_stats"])
                elif "v6_stats" in metadata:
                    # Legacy V6 stats - convert to V7Stats with defaults
                    v6_data = metadata["v6_stats"]
                    v7_stats = V7Stats(
                        **v6_data,
                        citation_validation_passed=v6_data.get("responsiveness") == "responsive",
                        claims_extracted=0,
                        claims_valid=0,
                        claims_dropped=0,
                    )
                elif "stats" in metadata:
                    # Streaming endpoint stores stats differently
                    stats_data = metadata["stats"]
                    v7_stats = V7Stats(**stats_data)
                
                # Extract claims and members from metadata
                claims = metadata.get("claims", [])
                members = metadata.get("members", [])
                
                messages.append(ChatMessage(
                    id=row[0],
                    session_id=row[1],
                    role=row[2],
                    content=row[3],
                    claims=claims if claims else None,
                    members=members if members else None,
                    result_set_id=row[4],
                    v7_stats=v7_stats,
                    created_at=row[6],
                ))
            
            return messages
    finally:
        conn.close()


@router.get("/debug/openai-test")
def test_openai_connection():
    """
    Diagnostic endpoint to test OpenAI API connectivity.
    
    Tests:
    1. API key is set
    2. API key format is valid
    3. Can actually connect to OpenAI
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    results = {
        "api_key_set": bool(api_key),
        "api_key_format": None,
        "connection_test": None,
        "error": None,
    }
    
    if not api_key:
        results["error"] = "OPENAI_API_KEY not set"
        return results
    
    # Check format
    if api_key.startswith("sk-proj-"):
        results["api_key_format"] = "project_key"
    elif api_key.startswith("sk-"):
        results["api_key_format"] = "standard_key"
    else:
        results["api_key_format"] = "unknown"
        results["error"] = f"Unexpected key format: {api_key[:15]}..."
        return results
    
    # Test connection
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5,
        )
        
        results["connection_test"] = "success"
        results["response"] = response.choices[0].message.content
        
    except Exception as e:
        results["connection_test"] = "failed"
        results["error"] = f"{type(e).__name__}: {str(e)}"
    
    return results


# =============================================================================
# Streaming Chat Endpoint
# =============================================================================

def _run_v7_with_progress(
    conn,
    session_id: int,
    question: str,
    progress_queue: queue.Queue,
    pre_bundling_mode: str = "off",
) -> Dict[str, Any]:
    """
    Run V7 query with progress callback that pushes events to queue.
    """
    try:
        from retrieval.agent.v7_runner import run_v7_query
        from retrieval.agent.v6_controller import V6Config
        from retrieval.agent.v7_controller import V7Config
    except ImportError as e:
        progress_queue.put({"type": "error", "error": f"Import error: {e}"})
        return {"error": str(e)}
    
    def progress_callback(step: str, status: str, message: str, details: Dict[str, Any]):
        """Push progress events to the queue."""
        event = {
            "type": "progress",
            "step": step,
            "status": status,
            "message": message,
            "details": details,
            "timestamp": time.time(),
        }
        progress_queue.put(event)
    
    start_time = time.time()
    
    try:
        # Run V7 workflow
        result = run_v7_query(
            conn=conn,
            question=question,
            max_bottleneck_spans=40,
            max_rounds=5,
            drop_uncited_claims=True,
            verbose=True,
            progress_callback=progress_callback,
            pre_bundling_mode=pre_bundling_mode,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Get V6 trace from V7 result
        v6_trace = result.trace.v6_trace if result.trace else None
        span_list = v6_trace.bottleneck_spans if v6_trace else []
        
        def _build_citation_from_span_stream(span, span_idx: int) -> Dict[str, Any]:
            """Build a citation dict from a BottleneckSpan object."""
            if hasattr(span, 'chunk_id'):
                return {
                    "span_id": str(span_idx),
                    "chunk_id": span.chunk_id,
                    "document_id": span.doc_id,
                    "page_number": span.page if isinstance(span.page, int) else None,
                    "quote": span.span_text[:200] if hasattr(span, 'span_text') else "",
                    "source_name": span.source_label if hasattr(span, 'source_label') else None,
                }
            elif isinstance(span, dict):
                return {
                    "span_id": str(span_idx),
                    "chunk_id": span.get('chunk_id'),
                    "document_id": span.get('doc_id'),
                    "page_number": span.get('page'),
                    "quote": span.get('span_text', '')[:200],
                    "source_name": span.get('source_label'),
                }
            return {"span_id": str(span_idx), "quote": ""}
        
        # Build a span_id -> span mapping for efficient lookup
        # V7 uses span_ids like "sp_0", "sp_1" or "bs_123" 
        span_by_id: Dict[str, Tuple[Any, int]] = {}
        for idx, span in enumerate(span_list):
            # BottleneckSpan uses "bs_<chunk_id>" format
            if hasattr(span, 'span_id'):
                span_by_id[span.span_id] = (span, idx)
            # Also register with sp_<idx> format for V7 claim enumerator fallback
            span_by_id[f"sp_{idx}"] = (span, idx)
            # And with bs_<idx> for index-based lookup
            span_by_id[f"bs_{idx}"] = (span, idx)
        
        # Build claims from V7 expanded summary
        claims = []
        if result.expanded_summary and result.expanded_summary.claims:
            for claim_obj in result.expanded_summary.claims[:20]:
                citations = []
                for cit_id in claim_obj.citations[:3]:
                    # Try to find the span by ID
                    if cit_id in span_by_id:
                        span, idx = span_by_id[cit_id]
                        citations.append(_build_citation_from_span_stream(span, idx))
                    else:
                        # Try numeric extraction: "sp_5" -> index 5
                        try:
                            if cit_id.startswith("sp_"):
                                idx = int(cit_id[3:])
                                if 0 <= idx < len(span_list):
                                    citations.append(_build_citation_from_span_stream(span_list[idx], idx))
                                    continue
                            elif cit_id.startswith("bs_"):
                                idx = int(cit_id[3:])
                                if 0 <= idx < len(span_list):
                                    citations.append(_build_citation_from_span_stream(span_list[idx], idx))
                                    continue
                        except (ValueError, IndexError):
                            pass
                        # Fallback: empty citation with ID
                        citations.append({"span_id": cit_id, "quote": ""})
                
                claims.append({
                    "text": claim_obj.claim_text,
                    "confidence": claim_obj.support_level,
                    "citations": citations,
                })
        
        # Build members with citations
        member_citations: Dict[str, List[Dict]] = {}
        for idx, span in enumerate(span_list):
            member_name = None
            if hasattr(span, 'member_name') and span.member_name:
                member_name = span.member_name
            elif isinstance(span, dict) and span.get('member_name'):
                member_name = span['member_name']
            if member_name:
                if member_name not in member_citations:
                    member_citations[member_name] = []
                if len(member_citations[member_name]) < 3:
                    member_citations[member_name].append(_build_citation_from_span_stream(span, idx))
        
        members = []
        for member_name in result.members_identified[:30]:
            cits = member_citations.get(member_name, [])
            members.append({"name": member_name, "citations": cits})
        
        # Build actions from trace
        actions = []
        if v6_trace:
            # Query parsing
            if v6_trace.parsed_query:
                pq = v6_trace.parsed_query
                actions.append({
                    "step": "query_parsing",
                    "status": "completed",
                    "message": f"Parsed as {pq.task_type.value} task",
                    "details": {"task_type": pq.task_type.value, "topic_terms": pq.topic_terms[:5]},
                })
            # Entity linking
            if v6_trace.entity_linking:
                el = v6_trace.entity_linking
                actions.append({
                    "step": "entity_linking",
                    "status": "completed",
                    "message": f"Linked {el.total_linked} entities",
                    "details": {"total_linked": el.total_linked, "used_for_retrieval": el.used_for_retrieval},
                })
            # Rounds
            for i, rd in enumerate(v6_trace.rounds):
                if isinstance(rd, dict):
                    actions.append({
                        "step": f"retrieval_round_{i+1}",
                        "status": "completed",
                        "message": f"Round {i+1}: {rd.get('chunks_retrieved', 0)} chunks → {rd.get('spans_after_bottleneck', 0)} spans",
                        "details": rd,
                    })
            # Synthesis
            actions.append({
                "step": "synthesis",
                "status": "completed",
                "message": f"Generated {len(v6_trace.final_claims) if v6_trace.final_claims else 0} claims",
                "details": {"claims_count": len(v6_trace.final_claims) if v6_trace.final_claims else 0},
            })
        
        # V7 citation enforcement actions
        if result.trace:
            actions.append({
                "step": "claim_enumeration",
                "status": "completed",
                "message": f"Extracted {result.trace.claims_extracted} claims",
                "details": {
                    "claims_extracted": result.trace.claims_extracted,
                    "claims_valid": result.trace.claims_valid,
                },
            })
            actions.append({
                "step": "stop_gate",
                "status": "completed",
                "message": f"Citation validation: {'passed' if result.citation_validation_passed else 'failed'}",
                "details": {"passed": result.citation_validation_passed},
            })
        
        # Build stats
        total_spans = 0
        if v6_trace and v6_trace.rounds:
            last_round = v6_trace.rounds[-1]
            if isinstance(last_round, dict):
                total_spans = last_round.get("total_spans", 0)
        
        stats = {
            "task_type": v6_trace.parsed_query.task_type.value if v6_trace and v6_trace.parsed_query else "unknown",
            "rounds_executed": len(v6_trace.rounds) if v6_trace else 0,
            "total_spans": total_spans,
            "unique_docs": 0,
            "elapsed_ms": elapsed_ms,
            "entity_linking": {
                "total_linked": v6_trace.entity_linking.total_linked if v6_trace and v6_trace.entity_linking else 0,
                "used_for_retrieval": v6_trace.entity_linking.used_for_retrieval if v6_trace and v6_trace.entity_linking else 0,
            } if v6_trace else {},
            "responsiveness": result.responsiveness_status,
            "actions": actions,
            # V7-specific
            "citation_validation_passed": result.citation_validation_passed,
            "claims_extracted": result.trace.claims_extracted if result.trace else 0,
            "claims_valid": result.trace.claims_valid if result.trace else 0,
            "claims_dropped": result.trace.claims_unsupported if result.trace else 0,
        }
        
        return {
            "answer": result.answer,
            "claims": claims,
            "members": members,
            "stats": stats,
            "is_responsive": result.citation_validation_passed,
        }
        
    except Exception as e:
        import traceback
        progress_queue.put({"type": "error", "error": str(e), "traceback": traceback.format_exc()})
        return {"error": str(e)}


def _sse_event_stream(
    session_id: int,
    message: str,
    pre_bundling_mode: str = "off",
) -> Generator[str, None, None]:
    """
    Generator that yields SSE events as V7 progresses.
    """
    conn = get_conn()
    progress_queue: queue.Queue = queue.Queue()
    result_holder: List[Dict] = []
    
    def run_v7_thread():
        """Run V7 in a thread, pushing progress to queue."""
        result = _run_v7_with_progress(conn, session_id, message, progress_queue, pre_bundling_mode=pre_bundling_mode)
        result_holder.append(result)
        progress_queue.put({"type": "done"})
    
    # Start V7 in background thread
    thread = threading.Thread(target=run_v7_thread, daemon=True)
    thread.start()
    
    # Yield SSE events as they come
    while True:
        try:
            event = progress_queue.get(timeout=300)  # 5 min timeout
            
            if event.get("type") == "done":
                # V6 finished, save assistant message and yield final result
                if result_holder:
                    final_result = result_holder[0]
                    
                    # Save assistant message to database
                    answer = final_result.get("answer", "")
                    if answer and not final_result.get("error"):
                        try:
                            with conn.cursor() as cur:
                                # Build metadata for the message
                                metadata = {
                                    "claims": final_result.get("claims", []),
                                    "members": final_result.get("members", []),
                                    "stats": final_result.get("stats", {}),
                                    "is_responsive": final_result.get("is_responsive", False),
                                }
                                cur.execute(
                                    """
                                    INSERT INTO research_messages (session_id, role, content, metadata)
                                    VALUES (%s, 'assistant', %s, %s)
                                    """,
                                    (session_id, answer, json.dumps(metadata)),
                                )
                                conn.commit()
                        except Exception as save_err:
                            print(f"[SSE] Error saving assistant message: {save_err}", file=sys.stderr)
                    
                    yield f"event: result\ndata: {json.dumps(final_result)}\n\n"
                break
            elif event.get("type") == "error":
                yield f"event: error\ndata: {json.dumps(event)}\n\n"
                break
            else:
                # Progress event
                yield f"event: progress\ndata: {json.dumps(event)}\n\n"
                
        except queue.Empty:
            # Timeout - send keepalive
            yield f": keepalive\n\n"
    
    conn.close()


@router.post("/{session_id}/chat/stream")
def chat_stream(session_id: int, req: ChatRequest):
    """
    Streaming chat endpoint that yields Server-Sent Events (SSE) as V7 progresses.
    
    Events:
    - progress: Workflow step updates (query_parsing, entity_linking, retrieval_round_N, 
                synthesis, claim_enumeration, stop_gate)
    - result: Final result with answer, claims, members, stats (including citation validation)
    - error: Error message if something went wrong
    
    Use EventSource in the browser to receive these events.
    """
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Verify API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    
    # Verify session exists
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM research_sessions WHERE id = %s", (session_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Store user message first
            cur.execute(
                """
                INSERT INTO research_messages (session_id, role, content)
                VALUES (%s, 'user', %s)
                RETURNING id
                """,
                (session_id, message),
            )
            conn.commit()
    finally:
        conn.close()
    
    return StreamingResponse(
        _sse_event_stream(session_id, message, pre_bundling_mode=req.pre_bundling_mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
