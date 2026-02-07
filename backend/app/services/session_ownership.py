"""
Session ownership guard: assert that a session (or plan/result_set) belongs to the given user (Cognito sub).
Use before any chat/plans/results operation.
"""
from fastapi import HTTPException

from app.services.db import get_conn


def assert_session_owned(session_id: int, user_sub: str) -> None:
    """
    Raise 404 if the session does not exist or does not belong to user_sub.
    Call this at the start of any endpoint that takes session_id.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM research_sessions WHERE id = %s AND user_sub = %s",
                (session_id, user_sub),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Session not found")
    finally:
        conn.close()


def assert_plan_owned(plan_id: int, user_sub: str) -> None:
    """Raise 404 if the plan does not exist or its session is not owned by user_sub."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM research_plans p
                JOIN research_sessions s ON s.id = p.session_id AND s.user_sub = %s
                WHERE p.id = %s
                """,
                (user_sub, plan_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Plan not found")
    finally:
        conn.close()


def assert_result_set_owned(result_set_id: int, user_sub: str) -> None:
    """Raise 404 if the result set does not exist or is not owned by user_sub (via session)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Resolve session via retrieval_runs (result_sets.retrieval_run_id -> retrieval_runs.session_id)
            cur.execute(
                """
                SELECT rr.session_id
                FROM result_sets rs
                JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id
                JOIN research_sessions s ON s.id = rr.session_id AND s.user_sub = %s
                WHERE rs.id = %s
                """,
                (user_sub, result_set_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Result set not found")
    finally:
        conn.close()
