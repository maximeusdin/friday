"""
Plan proposal service.
Wraps scripts/plan_query.py functionality.

Supports two execution modes:
1. Traditional: Compiles primitives → retrieval → result_set
2. Agentic: Plan → Execute → Verify → Render (with evidence bundle)
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import re

from psycopg2.extras import Json

REPO_ROOT = Path(__file__).parent.parent.parent.parent
PLAN_QUERY_SCRIPT = REPO_ROOT / "scripts" / "plan_query.py"

# Feature flag for agentic mode (enabled by default)
AGENTIC_MODE_ENABLED = os.getenv("AGENTIC_MODE_ENABLED", "1") == "1"


def propose_plan(conn, session_id: int, user_text: str) -> Dict[str, Any]:
    """
    Propose a plan for a user query.
    
    Calls the existing plan_query.py script via subprocess, then
    fetches the created plan from the database.
    
    Args:
        conn: Database connection
        session_id: The session to associate the plan with
        user_text: The user's query text
    
    Returns:
        Dict with plan data: id, status, plan_json, created_at, etc.
    """
    # Run the plan_query script
    result = subprocess.run(
        [
            sys.executable,
            str(PLAN_QUERY_SCRIPT),
            "--session", str(session_id),
            "--text", user_text,
        ],
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout for LLM call
    )
    
    if result.returncode != 0:
        # The planner sometimes deterministically returns "needs_clarification".
        # In that case, we should treat it as a successful response and persist
        # a plan row so the UI can render it.
        combined = "\n".join(
            [s for s in [result.stderr, result.stdout] if s and s.strip()]
        )
        clarification = _try_extract_clarification_plan_json(combined)
        if clarification is not None:
            plan_id, created_at = _save_clarification_plan(
                conn=conn,
                session_id=session_id,
                user_utterance=user_text,
                plan_json=clarification,
                query_lang_version="qir_v1",
                retrieval_impl_version="retrieval_v1",
            )
            return {
                "id": plan_id,
                "session_id": session_id,
                "status": "proposed",
                "user_utterance": user_text,
                "plan_json": clarification,
                "parent_plan_id": None,
                "retrieval_run_id": None,
                "result_set_id": None,
                "created_at": created_at,
                "approved_at": None,
                "executed_at": None,
            }

        error_msg = combined or "Unknown error"
        raise RuntimeError(f"Plan proposal failed: {error_msg}")

    # Prefer reading the exact plan_id from stderr (avoids race / multiple plans).
    plan_id = _try_extract_plan_id(result.stderr or "")
    if plan_id is None:
        # Fallback: latest plan in session
        plan_id = _fetch_latest_plan_id_for_session(conn, session_id)
        if plan_id is None:
            raise RuntimeError(
                f"Plan creation succeeded but no plan found for session {session_id}. "
                f"Script output: {(result.stderr or result.stdout or '')[:500]}"
            )

    return _fetch_plan_row(conn, plan_id)


def render_plan_summary(plan_json: Dict[str, Any]) -> str:
    """
    Render a human-readable summary of a plan.
    
    Args:
        plan_json: The plan's JSON structure with primitives
    
    Returns:
        Human-readable summary string
    """
    if plan_json.get("needs_clarification"):
        choices = plan_json.get("choices") or []
        if isinstance(choices, list) and choices:
            lines = ["Needs clarification. Please choose one:"]
            for i, c in enumerate(choices, 1):
                lines.append(f"{i}. {c}")
            return "\n".join(lines)
        return "Needs clarification."

    primitives = _extract_primitives(plan_json)
    if not primitives:
        return "Empty plan"
    
    lines = []
    for i, prim in enumerate(primitives, 1):
        ptype = prim.get("type", "unknown")
        
        # Planner uses uppercase primitive types (TERM/PHRASE/...)
        # Older/internal code sometimes uses lowercase. Support both.
        ptype_norm = str(ptype).lower()

        if ptype_norm == "term":
            lines.append(f"{i}. Search for term: \"{prim.get('value', '')}\"")
        elif ptype_norm == "phrase":
            lines.append(f"{i}. Search for phrase: \"{prim.get('value', '')}\"")
        elif ptype_norm == "entity":
            lines.append(f"{i}. Find entity: \"{prim.get('name', '')}\"")
        elif ptype_norm == "filter_collection":
            lines.append(f"{i}. Filter to collection: {prim.get('slug', '')}")
        elif ptype_norm == "filter_date_range":
            lines.append(
                f"{i}. Filter date range: {prim.get('start', '?')} to {prim.get('end', '?')}"
            )
        elif ptype_norm == "set_top_k":
            lines.append(f"{i}. Limit results to top {prim.get('k', '?')}")
        elif ptype_norm == "set_search_type":
            lines.append(f"{i}. Search type: {prim.get('search_type', 'hybrid')}")
        elif ptype_norm == "co_occurs_with":
            lines.append(f"{i}. Co-occurs with: \"{prim.get('value', '')}\"")
        elif ptype_norm == "or_group":
            terms = prim.get("terms", [])
            lines.append(f"{i}. Any of: {', '.join(repr(t) for t in terms)}")
        # Two-Mode Retrieval primitives
        elif ptype_norm == "set_retrieval_mode":
            lines.append(f"{i}. Retrieval mode: {prim.get('mode', 'conversational')}")
        elif ptype_norm == "set_similarity_threshold":
            lines.append(f"{i}. Similarity threshold: {prim.get('threshold', 0.35)}")
        elif ptype_norm == "related_entities":
            window = prim.get('window', 'document')
            top_n = prim.get('top_n', 20)
            lines.append(f"{i}. Find related entities (window={window}, top_n={top_n})")
        elif ptype_norm == "entity_role":
            role = prim.get('role', '')
            lines.append(f"{i}. Filter to entity role: {role}")
        elif ptype_norm == "except_entities":
            entity_ids = prim.get('entity_ids', [])
            lines.append(f"{i}. Exclude entities: {entity_ids}")
        # Index Retrieval primitives (entity)
        elif ptype_norm == "first_mention":
            entity_id = prim.get('entity_id')
            order_by = prim.get('order_by', 'chronological')
            lines.append(f"{i}. Find first mention of entity {entity_id} (order: {order_by})")
        elif ptype_norm == "first_co_mention":
            entity_ids = prim.get('entity_ids', [])
            window = prim.get('window', 'chunk')
            lines.append(f"{i}. Find first co-mention of entities {entity_ids} (window: {window})")
        elif ptype_norm == "mentions":
            entity_id = prim.get('entity_id')
            order_by = prim.get('order_by', 'chronological')
            lines.append(f"{i}. Find all mentions of entity {entity_id} (order: {order_by})")
        # Index Retrieval primitives (date)
        elif ptype_norm == "date_range_filter":
            date_start = prim.get('date_start', '?')
            date_end = prim.get('date_end', '?')
            time_basis = prim.get('time_basis', 'mentioned_date')
            lines.append(f"{i}. Date filter: {date_start} to {date_end} (basis: {time_basis})")
        elif ptype_norm == "date_mentions":
            date_start = prim.get('date_start', '?')
            date_end = prim.get('date_end', '?')
            lines.append(f"{i}. Find dated chunks: {date_start} to {date_end}")
        elif ptype_norm == "first_date_mention":
            entity_id = prim.get('entity_id')
            time_basis = prim.get('time_basis', 'mentioned_date')
            lines.append(f"{i}. Find first dated mention of entity {entity_id} (basis: {time_basis})")
        # Index Retrieval primitives (place)
        elif ptype_norm == "place_mentions":
            place_entity_id = prim.get('place_entity_id')
            lines.append(f"{i}. Find mentions of place {place_entity_id}")
        elif ptype_norm == "related_places":
            entity_id = prim.get('entity_id')
            window = prim.get('window', 'chunk')
            lines.append(f"{i}. Find places related to entity {entity_id} (window: {window})")
        elif ptype_norm == "within_country":
            country = prim.get('country', '?')
            lines.append(f"{i}. Filter to places in: {country}")
        else:
            # Generic fallback
            params = {k: v for k, v in prim.items() if k != "type"}
            if params:
                lines.append(f"{i}. {ptype}: {params}")
            else:
                lines.append(f"{i}. {ptype}")
    
    # Add execution envelope info if present
    envelope = plan_json.get("execution_envelope", {}) or plan_json.get("execution", {}) or {}
    if envelope:
        env_parts = []
        if envelope.get("top_k"):
            env_parts.append(f"top_k={envelope['top_k']}")
        if envelope.get("search_type"):
            env_parts.append(f"search={envelope['search_type']}")
        if env_parts:
            lines.append(f"\nExecution: {', '.join(env_parts)}")
    
    return "\n".join(lines)


def _extract_primitives(plan_json: Dict[str, Any]) -> list:
    """
    Planner outputs are typically shaped like:
      { query: { primitives: [...] }, execution_envelope: {...}, ... }
    But older/other shapes may put primitives at top-level.
    """
    if isinstance(plan_json.get("primitives"), list):
        return plan_json.get("primitives")  # type: ignore[return-value]
    query = plan_json.get("query")
    if isinstance(query, dict) and isinstance(query.get("primitives"), list):
        return query.get("primitives")  # type: ignore[return-value]
    return []


def _try_extract_clarification_plan_json(output_text: str) -> Optional[Dict[str, Any]]:
    """
    plan_query.py prints a JSON object on failure when it needs clarification.
    We parse the JSON block from the combined stderr/stdout.
    
    The output typically looks like:
        Session: 44 (ID: 44)
        Calling LLM to generate plan...
        ERROR: Plan validation failed:
          - LLM requested clarification - cannot proceed
        {
          "query": { ... },
          "needs_clarification": true,
          "choices": [...]
        }
    
    We need to find the *outermost* JSON object, not a nested one.
    """
    if not output_text:
        return None

    lines = output_text.splitlines()
    
    # Find all line indices where a line starts with '{' (potential JSON start)
    json_start_candidates = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("{"):
            json_start_candidates.append(i)
    
    if not json_start_candidates:
        return None
    
    # Try each candidate (from first to last) - the outermost JSON object
    # is likely the first '{' that starts at column 0 or near it
    for start_idx in json_start_candidates:
        candidate = "\n".join(lines[start_idx:]).strip()
        try:
            obj = json.loads(candidate)
            # Check if this is the clarification object we're looking for
            if isinstance(obj, dict) and obj.get("needs_clarification") is True and isinstance(obj.get("choices"), list):
                return obj
        except json.JSONDecodeError:
            # This candidate didn't parse - try next one
            continue
    
    return None


def _compute_plan_hash(plan_json: Dict[str, Any]) -> str:
    payload = json.dumps(plan_json, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_clarification_plan(
    *,
    conn,
    session_id: int,
    user_utterance: str,
    plan_json: Dict[str, Any],
    query_lang_version: str,
    retrieval_impl_version: str,
) -> tuple[int, Any]:
    plan_hash = _compute_plan_hash(plan_json)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_plans (
                session_id,
                plan_json,
                plan_hash,
                query_lang_version,
                retrieval_impl_version,
                status,
                user_utterance
            )
            VALUES (%s, %s, %s, %s, %s, 'proposed', %s)
            RETURNING id, created_at
            """,
            (session_id, Json(plan_json), plan_hash, query_lang_version, retrieval_impl_version, user_utterance),
        )
        row = cur.fetchone()
        conn.commit()
        return row[0], row[1]


def _try_extract_plan_id(stderr_text: str) -> Optional[int]:
    """
    plan_query.py prints:
      '✅ Plan saved with ID: {plan_id} (session=..., status: proposed)'
    """
    if not stderr_text:
        return None
    m = re.search(r"Plan saved with ID:\s*(\d+)", stderr_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _fetch_latest_plan_id_for_session(conn, session_id: int) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM research_plans WHERE session_id = %s ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def _jsonb_to_obj(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def _fetch_plan_row(conn, plan_id: int) -> Dict[str, Any]:
    """
    Fetch a plan row in a schema-tolerant way.
    """
    from app.services.db import get_dsn
    from app.services.schema import get_table_columns, plan_user_utterance_expr, has_column

    cols = get_table_columns(get_dsn(), "research_plans")
    utter_expr = plan_user_utterance_expr(cols)

    select_cols = [
        "id",
        "session_id",
        "status",
        f"{utter_expr} AS user_utterance",
        "plan_json",
        "created_at",
    ]
    # Optional columns across schema versions
    if has_column(cols, "parent_plan_id"):
        select_cols.append("parent_plan_id")
    else:
        select_cols.append("NULL::bigint AS parent_plan_id")
    if has_column(cols, "retrieval_run_id"):
        select_cols.append("retrieval_run_id")
    else:
        select_cols.append("NULL::bigint AS retrieval_run_id")
    if has_column(cols, "result_set_id"):
        select_cols.append("result_set_id")
    else:
        select_cols.append("NULL::bigint AS result_set_id")
    if has_column(cols, "approved_at"):
        select_cols.append("approved_at")
    else:
        select_cols.append("NULL::timestamptz AS approved_at")
    if has_column(cols, "executed_at"):
        select_cols.append("executed_at")
    else:
        select_cols.append("NULL::timestamptz AS executed_at")

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM research_plans
        WHERE id = %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, (plan_id,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"Plan {plan_id} not found after creation")

        # Row mapping matches select_cols order above
        (
            pid,
            session_id,
            status,
            user_utt,
            plan_json_val,
            created_at,
            parent_plan_id,
            retrieval_run_id,
            result_set_id,
            approved_at,
            executed_at,
        ) = row

        plan_json_obj = plan_json_val if isinstance(plan_json_val, dict) else _jsonb_to_obj(plan_json_val)

        return {
            "id": pid,
            "session_id": session_id,
            "status": status,
            "user_utterance": user_utt,
            "plan_json": plan_json_obj,
            "parent_plan_id": parent_plan_id,
            "retrieval_run_id": retrieval_run_id,
            "result_set_id": result_set_id,
            "created_at": created_at,
            "approved_at": approved_at,
            "executed_at": executed_at,
        }


# =============================================================================
# Agentic Workflow Support
# =============================================================================

def execute_agentic_workflow(
    conn,
    session_id: int,
    user_text: str,
    *,
    store_bundle: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full agentic workflow for a query.
    
    This is the main entry point for agentic mode execution.
    Uses: Plan → Execute → Verify → Render architecture.
    
    Args:
        conn: Database connection
        session_id: Session ID
        user_text: User's query text
        store_bundle: Whether to store evidence bundle in database
        
    Returns:
        Dict with rendered_answer, evidence_bundle, verification_result, etc.
    """
    if not AGENTIC_MODE_ENABLED:
        raise RuntimeError(
            "Agentic mode is not enabled. Set AGENTIC_MODE_ENABLED=1 to enable."
        )
    
    # Import agentic execution function
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.execute_plan import execute_plan_agentic, store_evidence_bundle
    
    # Build plan_data structure
    plan_data = {
        "session_id": session_id,
        "user_utterance": user_text,
        "plan_json": {},  # Agentic mode builds its own plan
    }
    
    # Execute agentic workflow
    result = execute_plan_agentic(conn, plan_data)
    
    # Optionally store evidence bundle
    if store_bundle and "evidence_bundle" in result:
        try:
            # Create minimal result_set for bundle storage
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO result_sets (name, chunk_ids, session_id)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (
                        f"Agentic: {user_text[:50]}...",
                        [],  # Empty - bundle has the data
                        session_id,
                    ),
                )
                result_set_id = cur.fetchone()[0]
            
            bundle_id = store_evidence_bundle(
                conn,
                result_set_id,
                result["evidence_bundle"],
                result.get("plan", {}),
                result.get("verification_result"),
                result.get("answer_trace"),
            )
            result["evidence_bundle_id"] = bundle_id
            result["result_set_id"] = result_set_id
        except Exception as e:
            # Don't fail if bundle storage fails
            result["bundle_storage_error"] = str(e)
    
    return result


def is_agentic_query(user_text: str) -> bool:
    """
    Determine if a query should use agentic mode.
    
    Uses heuristics to identify complex queries that benefit from
    the full agentic workflow.
    """
    if not AGENTIC_MODE_ENABLED:
        return False
    
    text_lower = user_text.lower()
    
    # Patterns that benefit from agentic mode
    agentic_patterns = [
        r"\bwho\s+were\s+(?:the\s+)?(?:members?|people|handlers?|officers?)\b",
        r"\blist\s+(?:all\s+)?(?:the\s+)?(?:members?|people)\b",
        r"\bidentify\s+(?:all\s+)?(?:the\s+)?(?:members?|people)\b",
        r"\brelationship\s+between\b",
        r"\bconnected?\s+to\b.*\bintelligence\b",
        r"\bhandler\s+(?:of|for)\b",
        r"\bcase\s+officer\b",
        r"\bcodename\b.*\bidentify\b",
        r"\b(?:Soviet|Russian)\s+(?:intelligence\s+)?officers?\b",
    ]
    
    for pattern in agentic_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def get_execution_mode(user_text: str, explicit_mode: Optional[str] = None) -> str:
    """
    Determine which execution mode to use.
    
    Args:
        user_text: The user's query
        explicit_mode: Explicitly requested mode (if any)
        
    Returns:
        "agentic" or "traditional"
    """
    if explicit_mode:
        return explicit_mode
    
    if is_agentic_query(user_text):
        return "agentic"
    
    return "traditional"
