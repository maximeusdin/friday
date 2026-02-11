"""
Session and Message endpoints
"""
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from psycopg2.extras import Json

from app.routes.auth_cognito import require_user
from app.services.db import get_conn, get_dsn
from app.services.planner import propose_plan, render_plan_summary
from app.services.schema import get_table_columns, has_column
from app.services.session_ownership import assert_session_owned

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateSessionRequest(BaseModel):
    label: str


class Session(BaseModel):
    id: int
    label: str
    created_at: datetime
    message_count: Optional[int] = None
    last_activity: Optional[datetime] = None
    scope_json: Optional[dict] = None


class Message(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    plan_id: Optional[int] = None
    result_set_id: Optional[int] = None
    metadata: Optional[dict] = None
    created_at: datetime


class SendMessageRequest(BaseModel):
    content: str


class Plan(BaseModel):
    id: int
    session_id: int
    status: str
    user_utterance: str
    plan_json: dict
    plan_summary: str
    parent_plan_id: Optional[int] = None
    retrieval_run_id: Optional[int] = None
    result_set_id: Optional[int] = None
    created_at: datetime
    approved_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None


class SendMessageResponse(BaseModel):
    user_message: Message
    assistant_message: Message
    plan: Plan


class SessionState(BaseModel):
    """
    Lightweight session state for rehydrating the UI after refresh.
    """
    session_id: int
    latest_plan_id: Optional[int] = None
    latest_result_set_id: Optional[int] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=Session)
def create_session(req: CreateSessionRequest, user=Depends(require_user)):
    """Create a new research session."""
    label = req.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label cannot be empty")
    sub = user["sub"]
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_sessions (label, user_sub)
                VALUES (%s, %s)
                RETURNING id, label, created_at
                """,
                (label, sub),
            )
            row = cur.fetchone()
            conn.commit()
            return Session(
                id=row[0],
                label=row[1],
                created_at=row[2],
            )
    finally:
        conn.close()


@router.get("", response_model=List[Session])
def list_sessions(user=Depends(require_user)):
    """List all sessions with message counts (owned by current user)."""
    sub = user["sub"]
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    s.id,
                    s.label,
                    s.created_at,
                    COUNT(m.id) AS message_count,
                    MAX(m.created_at) AS last_activity
                FROM research_sessions s
                LEFT JOIN research_messages m ON m.session_id = s.id
                WHERE s.user_sub = %s
                GROUP BY s.id
                ORDER BY COALESCE(MAX(m.created_at), s.created_at) DESC
                """,
                (sub,),
            )
            rows = cur.fetchall()
            return [
                Session(
                    id=row[0],
                    label=row[1],
                    created_at=row[2],
                    message_count=row[3],
                    last_activity=row[4],
                )
                for row in rows
            ]
    finally:
        conn.close()


@router.get("/{session_id}", response_model=Session)
def get_session(session_id: int, user=Depends(require_user)):
    """Get a single session including scope_json."""
    assert_session_owned(session_id, user["sub"])
    conn = get_conn()
    try:
        cols = get_table_columns(get_dsn(), "research_sessions")
        has_scope = has_column(cols, "scope_json")
        with conn.cursor() as cur:
            if has_scope:
                cur.execute(
                    """
                    SELECT 
                        s.id,
                        s.label,
                        s.created_at,
                        COUNT(m.id) AS message_count,
                        MAX(m.created_at) AS last_activity,
                        s.scope_json
                    FROM research_sessions s
                    LEFT JOIN research_messages m ON m.session_id = s.id
                    WHERE s.id = %s AND s.user_sub = %s
                    GROUP BY s.id, s.scope_json
                    """,
                    (session_id, user["sub"]),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        s.id,
                        s.label,
                        s.created_at,
                        COUNT(m.id) AS message_count,
                        MAX(m.created_at) AS last_activity
                    FROM research_sessions s
                    LEFT JOIN research_messages m ON m.session_id = s.id
                    WHERE s.id = %s AND s.user_sub = %s
                    GROUP BY s.id
                    """,
                    (session_id, user["sub"]),
                )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found")
            scope = row[5] if has_scope else {"mode": "full_archive"}
            if isinstance(scope, str):
                import json as _json
                try:
                    scope = _json.loads(scope)
                except Exception:
                    scope = {"mode": "full_archive"}
            elif scope is None:
                scope = {"mode": "full_archive"}
            return Session(
                id=row[0],
                label=row[1],
                created_at=row[2],
                message_count=row[3],
                last_activity=row[4],
                scope_json=scope,
            )
    finally:
        conn.close()


@router.delete("/{session_id}")
def delete_session(session_id: int, user=Depends(require_user)):
    """
    Delete a session and all its related data (messages, plans, v9_runs, evidence_sets,
    think_deeper_runs, etc.). Deletes in FK-safe order.
    """
    assert_session_owned(session_id, user["sub"])
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 1. Null out session pointers to break circular refs
            cur.execute(
                "UPDATE research_sessions SET active_run_id = NULL, active_evidence_set_id = NULL "
                "WHERE id = %s AND user_sub = %s",
                (session_id, user["sub"]),
            )
            # 2. Delete think_deeper_steps (references think_deeper_runs)
            cur.execute(
                "DELETE FROM think_deeper_steps WHERE td_run_id IN "
                "(SELECT id FROM think_deeper_runs WHERE v9_run_id IN "
                "(SELECT id FROM v9_runs WHERE session_id = %s))",
                (session_id,),
            )
            # 3. Delete think_deeper_runs (references v9_runs, blocks v9_runs delete)
            cur.execute(
                "DELETE FROM think_deeper_runs WHERE v9_run_id IN "
                "(SELECT id FROM v9_runs WHERE session_id = %s)",
                (session_id,),
            )
            # 4. Delete v9_run_steps (references v9_runs)
            cur.execute(
                "DELETE FROM v9_run_steps WHERE run_id IN "
                "(SELECT id FROM v9_runs WHERE session_id = %s)",
                (session_id,),
            )
            # 5. Delete evidence_items (references evidence_sets)
            cur.execute(
                "DELETE FROM evidence_items WHERE evidence_set_id IN "
                "(SELECT id FROM evidence_sets WHERE session_id = %s)",
                (session_id,),
            )
            # 6. Delete v9_runs (references research_sessions)
            cur.execute("DELETE FROM v9_runs WHERE session_id = %s", (session_id,))
            # 7. Delete evidence_sets (references research_sessions)
            cur.execute("DELETE FROM evidence_sets WHERE session_id = %s", (session_id,))
            # 8. Delete messages and plans
            cur.execute("DELETE FROM research_messages WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM research_plans WHERE session_id = %s", (session_id,))
            # 9. Delete the session
            cur.execute(
                "DELETE FROM research_sessions WHERE id = %s AND user_sub = %s",
                (session_id, user["sub"]),
            )
            conn.commit()
        return {"ok": True}
    finally:
        conn.close()


# =============================================================================
# Scope Management
# =============================================================================

class UpdateScopeRequest(BaseModel):
    mode: str  # "full_archive" | "custom"
    included_collection_ids: Optional[List[int]] = None
    included_document_ids: Optional[List[int]] = None


@router.post("/{session_id}/scope")
def update_session_scope(session_id: int, req: UpdateScopeRequest, user=Depends(require_user)):
    """Update the user-selected scope for a session.

    Empty-selection guard: if mode=custom and both collection_ids and
    document_ids are empty, rejects with 400 to prevent "search nothing" states.
    """
    assert_session_owned(session_id, user["sub"])

    if req.mode not in ("full_archive", "custom"):
        raise HTTPException(status_code=400, detail="mode must be 'full_archive' or 'custom'")

    if req.mode == "custom":
        has_collections = req.included_collection_ids and len(req.included_collection_ids) > 0
        has_documents = req.included_document_ids and len(req.included_document_ids) > 0
        if not has_collections and not has_documents:
            raise HTTPException(
                status_code=400,
                detail="Custom scope requires at least one collection or document selected",
            )

    scope_json = {
        "mode": req.mode,
        "included_collection_ids": req.included_collection_ids or [],
        "included_document_ids": req.included_document_ids or [],
    }

    conn = get_conn()
    try:
        cols = get_table_columns(get_dsn(), "research_sessions")
        if has_column(cols, "scope_json"):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE research_sessions
                    SET scope_json = %s, updated_at = now()
                    WHERE id = %s AND user_sub = %s
                    """,
                    (Json(scope_json), session_id, user["sub"]),
                )
                conn.commit()
        return {"ok": True, "scope_json": scope_json}
    finally:
        conn.close()


@router.get("/{session_id}/messages", response_model=List[Message])
def get_messages(session_id: int, user=Depends(require_user)):
    """Get all messages in a session."""
    assert_session_owned(session_id, user["sub"])
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            
            cur.execute(
                """
                SELECT id, session_id, role, content, plan_id, result_set_id, metadata, created_at
                FROM research_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            return [
                Message(
                    id=row[0],
                    session_id=row[1],
                    role=row[2],
                    content=row[3],
                    plan_id=row[4],
                    result_set_id=row[5],
                    metadata=row[6],
                    created_at=row[7],
                )
                for row in rows
            ]
    finally:
        conn.close()


@router.get("/{session_id}/state", response_model=SessionState)
def get_session_state(session_id: int, user=Depends(require_user)):
    """
    Return the latest plan/result pointers for a session.

    This is used by the UI to reload a session and show the most recent plan/result
    without requiring the send-message response to be in memory.
    """
    assert_session_owned(session_id, user["sub"])
    conn = get_conn()
    try:
        with conn.cursor() as cur:

            # Latest plan id (by created_at)
            cur.execute(
                "SELECT id FROM research_plans WHERE session_id = %s ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            )
            plan_row = cur.fetchone()
            latest_plan_id = int(plan_row[0]) if plan_row else None

            # Latest result set id:
            # Prefer plan.result_set_id if column exists, otherwise fall back to latest result_sets by session_id if available.
            latest_result_set_id: Optional[int] = None
            plan_cols = get_table_columns(get_dsn(), "research_plans")
            if latest_plan_id and has_column(plan_cols, "result_set_id"):
                cur.execute(
                    "SELECT result_set_id FROM research_plans WHERE id = %s",
                    (latest_plan_id,),
                )
                rs_row = cur.fetchone()
                if rs_row and rs_row[0]:
                    latest_result_set_id = int(rs_row[0])

            if latest_result_set_id is None:
                rs_cols = get_table_columns(get_dsn(), "result_sets")
                if has_column(rs_cols, "session_id"):
                    cur.execute(
                        "SELECT id FROM result_sets WHERE session_id = %s ORDER BY created_at DESC LIMIT 1",
                        (session_id,),
                    )
                    rs2 = cur.fetchone()
                    if rs2:
                        latest_result_set_id = int(rs2[0])

            return SessionState(
                session_id=session_id,
                latest_plan_id=latest_plan_id,
                latest_result_set_id=latest_result_set_id,
            )
    finally:
        conn.close()

@router.post("/{session_id}/messages", response_model=SendMessageResponse)
def send_message(session_id: int, req: SendMessageRequest, user=Depends(require_user)):
    """
    Send a user message and get an assistant response with a proposed plan.
    
    This is the heart of the UI loop:
    1. Store the user message
    2. Call the planner to generate a plan
    3. Store the assistant message (with plan_id)
    4. Return all three objects
    """
    assert_session_owned(session_id, user["sub"])
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            
            # 1. Store user message
            cur.execute(
                """
                INSERT INTO research_messages (session_id, role, content)
                VALUES (%s, 'user', %s)
                RETURNING id, session_id, role, content, plan_id, result_set_id, metadata, created_at
                """,
                (session_id, content),
            )
            user_row = cur.fetchone()
            user_message = Message(
                id=user_row[0],
                session_id=user_row[1],
                role=user_row[2],
                content=user_row[3],
                plan_id=user_row[4],
                result_set_id=user_row[5],
                metadata=user_row[6],
                created_at=user_row[7],
            )
            conn.commit()
        
        # 2. Generate plan using existing planner
        plan_result = propose_plan(conn, session_id, content)

        with conn.cursor() as cur:
            # 3. Store assistant message with plan reference
            plan_summary = render_plan_summary(plan_result["plan_json"])

            if plan_result["plan_json"].get("needs_clarification"):
                assistant_content = (
                    "I need one clarification before I can execute a deterministic plan:\n\n"
                    f"{plan_summary}\n\n"
                    "Reply with the number (1/2/3) or paste the exact choice."
                )
                assistant_metadata = {
                    "needs_clarification": True,
                    "choices": plan_result["plan_json"].get("choices", []),
                }
            else:
                assistant_content = f"I've created a research plan for your query.\n\n{plan_summary}"
                assistant_metadata = {}
            
            cur.execute(
                """
                INSERT INTO research_messages (session_id, role, content, plan_id, metadata)
                VALUES (%s, 'assistant', %s, %s, %s)
                RETURNING id, session_id, role, content, plan_id, result_set_id, metadata, created_at
                """,
                (session_id, assistant_content, plan_result["id"], Json(assistant_metadata)),
            )
            asst_row = cur.fetchone()
            assistant_message = Message(
                id=asst_row[0],
                session_id=asst_row[1],
                role=asst_row[2],
                content=asst_row[3],
                plan_id=asst_row[4],
                result_set_id=asst_row[5],
                metadata=asst_row[6],
                created_at=asst_row[7],
            )
            conn.commit()
        
        # Build plan response
        plan = Plan(
            id=plan_result["id"],
            session_id=session_id,
            status=plan_result["status"],
            user_utterance=content,
            plan_json=plan_result["plan_json"],
            plan_summary=plan_summary,
            parent_plan_id=plan_result.get("parent_plan_id"),
            retrieval_run_id=plan_result.get("retrieval_run_id"),
            result_set_id=plan_result.get("result_set_id"),
            created_at=plan_result["created_at"],
            approved_at=plan_result.get("approved_at"),
            executed_at=plan_result.get("executed_at"),
        )
        
        return SendMessageResponse(
            user_message=user_message,
            assistant_message=assistant_message,
            plan=plan,
        )
    finally:
        conn.close()
