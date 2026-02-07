"""
Plan endpoints - view, approve, execute
"""
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from psycopg2.extras import Json

from fastapi import Depends

from app.routes.auth_cognito import require_user
from app.services.db import get_conn
from app.services.planner import render_plan_summary, propose_plan
from app.services.executor import execute_plan
from app.services.db import get_dsn
from app.services.schema import get_table_columns, plan_user_utterance_expr, has_column
from app.services.session_ownership import assert_plan_owned
import json

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================

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
    evidence_refs: list


class ResultSetResponse(BaseModel):
    id: int
    name: str
    retrieval_run_id: int
    summary: ResultSummary
    items: list[ResultItem]
    created_at: datetime


class ExecutePlanResponse(BaseModel):
    plan: Plan
    result_set: ResultSetResponse


class ClarifyPlanRequest(BaseModel):
    """
    Clarify a plan that has `needs_clarification=true`.
    Provide either `choice_id` (1-based) or `choice_text`.
    """
    choice_id: Optional[int] = None
    choice_text: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/{plan_id}", response_model=Plan)
def get_plan(plan_id: int, user=Depends(require_user)):
    """Get a plan by ID."""
    assert_plan_owned(plan_id, user["sub"])
    conn = get_conn()
    try:
        cols = get_table_columns(get_dsn(), "research_plans")
        utter_expr = plan_user_utterance_expr(cols)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT 
                    id, session_id, status, 
                    {utter_expr} as user_utterance,
                    plan_json,
                    {('parent_plan_id' if has_column(cols, 'parent_plan_id') else 'NULL::bigint')} as parent_plan_id,
                    {('retrieval_run_id' if has_column(cols, 'retrieval_run_id') else 'NULL::bigint')} as retrieval_run_id,
                    {('result_set_id' if has_column(cols, 'result_set_id') else 'NULL::bigint')} as result_set_id,
                    created_at,
                    {('approved_at' if has_column(cols, 'approved_at') else 'NULL::timestamptz')} as approved_at,
                    {('executed_at' if has_column(cols, 'executed_at') else 'NULL::timestamptz')} as executed_at
                FROM research_plans
                WHERE id = %s
                """,
                (plan_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Plan not found")
            
            plan_json = row[4]
            if isinstance(plan_json, str):
                try:
                    plan_json = json.loads(plan_json)
                except Exception:
                    plan_json = {}
            return Plan(
                id=row[0],
                session_id=row[1],
                status=row[2],
                user_utterance=row[3],
                plan_json=plan_json,
                plan_summary=render_plan_summary(plan_json),
                parent_plan_id=row[5],
                retrieval_run_id=row[6],
                result_set_id=row[7],
                created_at=row[8],
                approved_at=row[9],
                executed_at=row[10],
            )
    finally:
        conn.close()


@router.post("/{plan_id}/approve", response_model=Plan)
def approve_plan(plan_id: int, user=Depends(require_user)):
    """Approve a proposed plan."""
    assert_plan_owned(plan_id, user["sub"])
    conn = get_conn()
    try:
        cols = get_table_columns(get_dsn(), "research_plans")
        utter_expr = plan_user_utterance_expr(cols)
        with conn.cursor() as cur:
            # Check current status
            cur.execute(
                "SELECT status FROM research_plans WHERE id = %s",
                (plan_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Plan not found")
            
            if row[0] != "proposed":
                raise HTTPException(
                    status_code=409,
                    detail=f"Plan cannot be approved: current status is '{row[0]}'"
                )

            # Disallow approval if the plan is asking for clarification
            cur.execute("SELECT plan_json FROM research_plans WHERE id = %s", (plan_id,))
            plan_json_row = cur.fetchone()
            plan_json_val = plan_json_row[0] if plan_json_row else None
            if isinstance(plan_json_val, str):
                try:
                    plan_json_val = json.loads(plan_json_val)
                except Exception:
                    plan_json_val = {}
            if isinstance(plan_json_val, dict) and plan_json_val.get("needs_clarification"):
                raise HTTPException(
                    status_code=409,
                    detail="Plan needs clarification before it can be approved/executed",
                )

            if not has_column(cols, "approved_at"):
                raise HTTPException(
                    status_code=500,
                    detail="This database schema does not support plan approval timestamps (approved_at missing).",
                )
            
            # Update status
            cur.execute(
                """
                UPDATE research_plans
                SET status = 'approved', approved_at = now()
                WHERE id = %s
                """,
                (plan_id,),
            )
            conn.commit()
            
            # Re-fetch updated plan using schema-tolerant SELECT
            cur.execute(
                f"""
                SELECT 
                    id, session_id, status,
                    {utter_expr} as user_utterance,
                    plan_json,
                    {('parent_plan_id' if has_column(cols, 'parent_plan_id') else 'NULL::bigint')} as parent_plan_id,
                    {('retrieval_run_id' if has_column(cols, 'retrieval_run_id') else 'NULL::bigint')} as retrieval_run_id,
                    {('result_set_id' if has_column(cols, 'result_set_id') else 'NULL::bigint')} as result_set_id,
                    created_at,
                    {('approved_at' if has_column(cols, 'approved_at') else 'NULL::timestamptz')} as approved_at,
                    {('executed_at' if has_column(cols, 'executed_at') else 'NULL::timestamptz')} as executed_at
                FROM research_plans
                WHERE id = %s
                """,
                (plan_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Plan not found after approval")

            plan_json = row[4]
            if isinstance(plan_json, str):
                try:
                    plan_json = json.loads(plan_json)
                except Exception:
                    plan_json = {}

            return Plan(
                id=row[0],
                session_id=row[1],
                status=row[2],
                user_utterance=row[3],
                plan_json=plan_json,
                plan_summary=render_plan_summary(plan_json),
                parent_plan_id=row[5],
                retrieval_run_id=row[6],
                result_set_id=row[7],
                created_at=row[8],
                approved_at=row[9],
                executed_at=row[10],
            )
    finally:
        conn.close()


@router.post("/{plan_id}/execute", response_model=ExecutePlanResponse)
def execute_plan_endpoint(plan_id: int, user=Depends(require_user)):
    """Execute an approved plan."""
    assert_plan_owned(plan_id, user["sub"])
    conn = get_conn()
    try:
        cols = get_table_columns(get_dsn(), "research_plans")
        utter_expr = plan_user_utterance_expr(cols)
        with conn.cursor() as cur:
            # Check current status
            cur.execute(
                "SELECT status FROM research_plans WHERE id = %s",
                (plan_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Plan not found")
            
            if row[0] != "approved":
                raise HTTPException(
                    status_code=409,
                    detail=f"Plan cannot be executed: current status is '{row[0]}' (must be 'approved')"
                )
        
        # Execute using service
        result = execute_plan(conn, plan_id)
        
        # Fetch updated plan
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT 
                    id, session_id, status,
                    {utter_expr} as user_utterance,
                    plan_json,
                    {('parent_plan_id' if has_column(cols, 'parent_plan_id') else 'NULL::bigint')} as parent_plan_id,
                    {('retrieval_run_id' if has_column(cols, 'retrieval_run_id') else 'NULL::bigint')} as retrieval_run_id,
                    {('result_set_id' if has_column(cols, 'result_set_id') else 'NULL::bigint')} as result_set_id,
                    created_at,
                    {('approved_at' if has_column(cols, 'approved_at') else 'NULL::timestamptz')} as approved_at,
                    {('executed_at' if has_column(cols, 'executed_at') else 'NULL::timestamptz')} as executed_at
                FROM research_plans
                WHERE id = %s
                """,
                (plan_id,),
            )
            row = cur.fetchone()
            plan_json = row[4]
            if isinstance(plan_json, str):
                try:
                    plan_json = json.loads(plan_json)
                except Exception:
                    plan_json = {}
            plan = Plan(
                id=row[0],
                session_id=row[1],
                status=row[2],
                user_utterance=row[3],
                plan_json=plan_json,
                plan_summary=render_plan_summary(plan_json),
                parent_plan_id=row[5],
                retrieval_run_id=row[6],
                result_set_id=row[7],
                created_at=row[8],
                approved_at=row[9],
                executed_at=row[10],
            )
        
        return ExecutePlanResponse(
            plan=plan,
            result_set=result["result_set"],
        )
    finally:
        conn.close()


@router.post("/{plan_id}/clarify", response_model=Plan)
def clarify_plan(plan_id: int, req: ClarifyPlanRequest, user=Depends(require_user)):
    """
    Resolve a clarification request for a plan by choosing one option.

    This creates a new plan (via planner) in the same session, and returns it.
    """
    assert_plan_owned(plan_id, user["sub"])
    if req.choice_id is None and (req.choice_text is None or not req.choice_text.strip()):
        raise HTTPException(status_code=400, detail="Provide choice_id or choice_text")

    conn = get_conn()
    try:
        cols = get_table_columns(get_dsn(), "research_plans")
        utter_expr = plan_user_utterance_expr(cols)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT session_id, {utter_expr} as user_utterance, plan_json
                FROM research_plans
                WHERE id = %s
                """,
                (plan_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Plan not found")

            session_id, user_utterance, plan_json = row
            if isinstance(plan_json, str):
                try:
                    plan_json = json.loads(plan_json)
                except Exception:
                    plan_json = {}

        if not isinstance(plan_json, dict) or not plan_json.get("needs_clarification"):
            raise HTTPException(status_code=409, detail="Plan does not require clarification")

        choices = plan_json.get("choices") or []
        choice_text: Optional[str] = None

        if req.choice_id is not None:
            if not isinstance(choices, list) or not choices:
                raise HTTPException(status_code=409, detail="Plan has no choices to select")
            idx = req.choice_id - 1
            if idx < 0 or idx >= len(choices):
                raise HTTPException(status_code=400, detail=f"choice_id out of range (1..{len(choices)})")
            choice_text = str(choices[idx])
        else:
            choice_text = (req.choice_text or "").strip()

        combined_text = f"{user_utterance}\nClarification: {choice_text}"

        # Create a new plan using the existing planner (may still ask for clarification)
        new_plan = propose_plan(conn, int(session_id), combined_text)

        # Persist a system message recording the clarification selection
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_messages (session_id, role, content, plan_id, metadata)
                VALUES (%s, 'system', %s, %s, %s)
                """,
                (
                    int(session_id),
                    f"Clarification selected: {choice_text}",
                    new_plan["id"],
                    Json(
                        {
                            "clarified_from_plan_id": plan_id,
                            "choice_id": req.choice_id,
                            "choice_text": choice_text,
                            "new_plan_id": new_plan["id"],
                        }
                    ),
                ),
            )
            conn.commit()

        plan_json_obj = new_plan["plan_json"]
        return Plan(
            id=new_plan["id"],
            session_id=int(session_id),
            status=new_plan["status"],
            user_utterance=new_plan["user_utterance"],
            plan_json=plan_json_obj,
            plan_summary=render_plan_summary(plan_json_obj),
            parent_plan_id=new_plan.get("parent_plan_id"),
            retrieval_run_id=new_plan.get("retrieval_run_id"),
            result_set_id=new_plan.get("result_set_id"),
            created_at=new_plan["created_at"],
            approved_at=new_plan.get("approved_at"),
            executed_at=new_plan.get("executed_at"),
        )
    finally:
        conn.close()
