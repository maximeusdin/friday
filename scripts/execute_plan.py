#!/usr/bin/env python3
"""
execute_plan.py --plan-id <id>

Plan Approval & Execution
- Requires plan status = approved
- Compiles primitives → executable retrieval (uses plan's compiled components)
- Executes retrieval using compiled tsquery, scope, and expanded text
- Creates retrieval_runs and retrieval_run_chunk_evidence records
- Optionally creates result_sets
- Links plan → run → result_set
- Updates plan status → executed

Acceptance:
- Execution is fully reproducible from primitives + compiled artifacts
- No LLM involvement during execution
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import Json
from datetime import datetime

from retrieval.primitives import (
    ResearchPlan, compile_primitives, CountPrimitive, SetQueryModePrimitive
)
from retrieval.plan_validation import validate_plan
from retrieval.ops import (
    get_conn,
    log_retrieval_run,
    _insert_run_chunk_evidence,
    _parse_lexemes_from_tsquery,
    embed_query,
    vector_literal,
    SearchFilters,
    ChunkHit,
)

# =============================================================================
# Database
# =============================================================================

def _check_session_lookup_allowed():
    """
    Check if session lookups are disallowed (for testing self-contained execution).
    
    This is a defensive check. Currently execute_plan.py does NOT query research_sessions
    or retrieval_runs by session_id - all execution parameters come from the plan's
    execution_envelope. This function serves as documentation and a safeguard if future
    code attempts to query sessions.
    
    If you add code that queries sessions, call this function before the query.
    """
    if os.getenv("NEH_DISALLOW_SESSION_LOOKUPS") == "1":
        raise RuntimeError(
            "Session lookups disallowed (NEH_DISALLOW_SESSION_LOOKUPS=1). "
            "Execution must be self-contained and not query research_sessions or retrieval_runs by session_id."
        )

def get_plan(conn, plan_id: int, *, for_update: bool = False) -> Optional[Dict[str, Any]]:
    """
    Retrieve plan by ID.
    
    Args:
        conn: Database connection
        plan_id: Plan ID to retrieve
        for_update: If True, acquires row-level lock (SELECT ... FOR UPDATE)
                   to prevent concurrent execution races. Use within a transaction.
    """
    # Note: This queries research_plans by plan_id, not by session_id, so it's allowed
    lock_clause = "FOR UPDATE NOWAIT" if for_update else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, session_id, user_utterance, plan_json, plan_hash,
                   query_lang_version, retrieval_impl_version, status, parent_plan_id,
                   retrieval_run_id, result_set_id, executed_at
            FROM research_plans
            WHERE id = %s
            {lock_clause}
            """,
            (plan_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "session_id": row[1],
            "user_utterance": row[2],
            "plan_json": row[3],
            "plan_hash": row[4],
            "query_lang_version": row[5],
            "retrieval_impl_version": row[6],
            "status": row[7],
            "parent_plan_id": row[8],
            "retrieval_run_id": row[9],
            "result_set_id": row[10],
            "executed_at": row[11],
        }


class PlanExecutionLockError(Exception):
    """Raised when plan is locked by another process."""
    pass

# =============================================================================
# Concurrency / Locking
# =============================================================================

def acquire_plan_advisory_lock(conn, plan_id: int) -> None:
    """
    Acquire a session-level advisory lock for a plan_id.

    This prevents concurrent executions even if the code commits multiple times.
    Lock is released when the connection closes (or via release_plan_advisory_lock()).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (plan_id,))
        locked = cur.fetchone()[0]
        if not locked:
            raise PlanExecutionLockError(
                f"Plan {plan_id} is currently being executed by another process (advisory lock held)."
            )


def release_plan_advisory_lock(conn, plan_id: int) -> None:
    """Best-effort unlock (connection close also releases)."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (plan_id,))
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

# =============================================================================
# Execution
# =============================================================================

def execute_plan_retrieval(
    conn,
    plan: ResearchPlan,
    *,
    k: int = 20,
    preview_chars: int = 2000,
    probes: int = 20,
    top_n_vec: int = 200,
    top_n_lex: int = 200,
    rrf_k: int = 50,
    chunk_pv: str = "chunk_v1_full",
) -> Tuple[List[ChunkHit], int]:
    """
    Execute retrieval using plan's compiled components.
    Returns (hits, run_id).
    
    Uses the plan's compiled tsquery, expanded text, and scope constraints.
    Delegates to existing retrieval functions but applies scope constraints post-retrieval
    or via SQL filtering.
    """
    if not plan.compiled:
        plan.compile()
    
    compiled = plan.compiled
    expanded = compiled.get("expanded", {})
    expanded_text = expanded.get("expanded_text", plan.query.raw)
    scope = compiled.get("scope", {})
    
    # Determine search type from primitives
    search_type = "hybrid"  # default
    for p in plan.query.primitives:
        if hasattr(p, "type") and p.type == "SET_SEARCH_TYPE":
            search_type = p.value
            break
    
    # Get top_k from primitives
    for p in plan.query.primitives:
        if hasattr(p, "type") and p.type == "SET_TOP_K":
            k = p.value
            break
    
    # Build filters from primitives
    collection_slugs = None
    for p in plan.query.primitives:
        if hasattr(p, "type") and p.type == "FILTER_COLLECTION":
            if collection_slugs is None:
                collection_slugs = []
            collection_slugs.append(p.slug)
    
    filters = SearchFilters(
        chunk_pv=chunk_pv,
        collection_slugs=collection_slugs,
    )
    
    # Apply scope constraints by filtering results after retrieval
    # (Simpler than trying to merge scope SQL into retrieval queries)
    scope_chunk_ids = None
    if scope and scope.get("where_sql"):
        scope_where = scope["where_sql"]
        scope_params = scope.get("params", [])
        required_joins = scope.get("required_joins", {})
        
        # Build FROM clause with necessary joins
        from_clause = "FROM chunks c"
        join_clauses = []
        
        # Add joins based on required_joins dict
        if required_joins.get("chunk_metadata"):
            join_clauses.append("JOIN chunk_metadata cm ON cm.chunk_id = c.id")
        if required_joins.get("entity_mentions"):
            # Entity mentions join is handled via EXISTS subquery in WHERE clause
            # No explicit JOIN needed, but we note it for documentation
            pass
        if required_joins.get("date_mentions"):
            # Date mentions join is handled via EXISTS subquery in WHERE clause
            # No explicit JOIN needed, but we note it for documentation
            pass
        
        # Execute scope query to get chunk_ids
        if join_clauses:
            scope_sql = f"""
                SELECT DISTINCT c.id 
                {from_clause}
                {' '.join(join_clauses)}
                WHERE {scope_where}
            """
        else:
            scope_sql = f"SELECT DISTINCT c.id {from_clause} WHERE {scope_where}"
        
        with conn.cursor() as cur:
            cur.execute(scope_sql, scope_params)
            scope_chunk_ids = {row[0] for row in cur.fetchall()}
    
    # Execute retrieval using existing functions with compiled query text
    from retrieval.ops import hybrid_rrf, vector_search

    scope_only_executed = False

    # If the compiled expanded_text is empty, this usually means the plan is scope-only
    # (e.g., ENTITY / CO_OCCURS_WITH primitives without TERM/PHRASE). In that case,
    # do NOT attempt vector/hybrid retrieval (which would embed an empty string).
    if not (expanded_text or "").strip():
        if not scope_chunk_ids:
            hits = []
        else:
            # Fetch chunk metadata + preview for these chunks (deterministic ordering)
            # IMPORTANT: Do NOT filter by pipeline_version here. The chunk_ids from scope
            # are already tied to specific pipeline outputs via entity_mentions. We just
            # need to fetch the chunk content and the "best available" metadata for each.
            chunk_ids_list = sorted(scope_chunk_ids)[:k]
            with conn.cursor() as cur:
                # Use LATERAL join to get "best available" metadata per chunk:
                # 1. Prefer metadata matching the chunk's pipeline_version (if stored)
                # 2. Fallback to most recent derived_at
                cur.execute(
                    """
                    SELECT
                      c.id AS chunk_id,
                      COALESCE(cm.collection_slug, 'unknown') AS collection_slug,
                      COALESCE(cm.document_id, 0) AS document_id,
                      COALESCE(cm.first_page_id, 0) AS first_page_id,
                      COALESCE(cm.last_page_id, 0) AS last_page_id,
                      cm.date_min,
                      cm.date_max,
                      LEFT(COALESCE(c.clean_text, c.text), %s) AS preview
                    FROM chunks c
                    LEFT JOIN LATERAL (
                        SELECT
                          cm2.collection_slug,
                          cm2.document_id,
                          cm2.first_page_id,
                          cm2.last_page_id,
                          cm2.date_min,
                          cm2.date_max,
                          cm2.pipeline_version,
                          cm2.derived_at
                        FROM chunk_metadata cm2
                        WHERE cm2.chunk_id = c.id
                        ORDER BY
                          -- Prefer matching pipeline_version
                          CASE WHEN cm2.pipeline_version = c.pipeline_version THEN 0 ELSE 1 END,
                          -- Then most recent
                          cm2.derived_at DESC NULLS LAST
                        LIMIT 1
                    ) cm ON true
                    WHERE c.id = ANY(%s)
                    ORDER BY COALESCE(cm.document_id, 0), COALESCE(cm.first_page_id, 0), c.id
                    """,
                    (preview_chars, chunk_ids_list),
                )
                rows = cur.fetchall()

            # GUARDRAIL: Detect referential drift (chunk_ids exist but no rows fetched)
            if chunk_ids_list and len(rows) == 0:
                print(f"⚠️  INVARIANT VIOLATION: scope returned {len(chunk_ids_list)} chunk_ids but preview fetch returned 0 rows!")
                print(f"    Sample chunk_ids from scope: {chunk_ids_list[:10]}")
                # Check if chunks actually exist
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM chunks WHERE id = ANY(%s)", (chunk_ids_list,))
                    existing_count = cur.fetchone()[0]
                    print(f"    Chunks actually in DB: {existing_count} / {len(chunk_ids_list)}")
                    if existing_count < len(chunk_ids_list):
                        print("    This indicates deleted chunks or bad migration. Check entity_mentions referential integrity.")
            elif chunk_ids_list and len(rows) < len(chunk_ids_list):
                missing_count = len(chunk_ids_list) - len(rows)
                print(f"⚠️  Partial fetch: {len(rows)} rows for {len(chunk_ids_list)} chunk_ids ({missing_count} missing)")

            hits = [
                ChunkHit(
                    chunk_id=int(r[0]),
                    collection_slug=r[1],
                    document_id=int(r[2]),
                    first_page_id=int(r[3]),
                    last_page_id=int(r[4]),
                    date_min=r[5],
                    date_max=r[6],
                    preview=r[7] or "",
                )
                for r in rows
            ]

        # Treat this as a lexical run for logging/evidence schema purposes.
        # (There is no embedding input and no tsquery.)
        search_type = "lex"
        expanded_text = ""  # keep empty; indicates scope-only execution
        scope_only_executed = True

    # If we already produced hits via scope-only execution, skip vector/lex/hybrid retrieval functions.
    if not scope_only_executed and search_type == "vector":
        hits = vector_search(
            conn,
            expanded_text,
            filters=filters,
            k=k * 3 if scope_chunk_ids else k,  # Get more if we need to filter
            preview_chars=preview_chars,
            probes=probes,
            expand_concordance=False,  # Already expanded in plan
            log_run=False,  # We'll log it ourselves
            session_id=plan.session_id if hasattr(plan, 'session_id') else None,
        )
    elif not scope_only_executed and search_type == "lex":
        # Use hybrid_rrf but with top_n_vec=0 to make it lexical-only
        hits = hybrid_rrf(
            conn,
            expanded_text,
            filters=filters,
            k=k * 3 if scope_chunk_ids else k,
            preview_chars=preview_chars,
            probes=probes,
            top_n_vec=0,  # No vector component
            top_n_lex=top_n_lex,
            rrf_k=rrf_k,
            expand_concordance=False,  # Already expanded in plan
            log_run=False,  # We'll log it ourselves
            session_id=plan.session_id if hasattr(plan, 'session_id') else None,
        )
    elif not scope_only_executed:  # hybrid
        hits = hybrid_rrf(
            conn,
            expanded_text,
            filters=filters,
            k=k * 3 if scope_chunk_ids else k,
            preview_chars=preview_chars,
            probes=probes,
            top_n_vec=top_n_vec,
            top_n_lex=top_n_lex,
            rrf_k=rrf_k,
            expand_concordance=False,  # Already expanded in plan
            log_run=False,  # We'll log it ourselves
            session_id=plan.session_id if hasattr(plan, 'session_id') else None,
        )
    
    # Apply scope filtering if needed
    if scope_chunk_ids:
        hits = [h for h in hits if h.chunk_id in scope_chunk_ids]
        hits = hits[:k]  # Take top k after filtering
    
    # Log retrieval run with plan metadata
    tsquery = compiled.get("tsquery", {})
    tsquery_text = tsquery.get("text", "___nomatch___")
    embedding_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small") if search_type in ("vector", "hybrid") else None
    
    chunk_ids = [h.chunk_id for h in hits]
    retrieval_config_json = {
        "probes": probes,
        "top_n_vec": top_n_vec,
        "top_n_lex": top_n_lex,
        "rrf_k": rrf_k,
    } if search_type == "hybrid" else {}
    if not (expanded_text or "").strip() and scope_chunk_ids:
        retrieval_config_json = {
            **(retrieval_config_json or {}),
            "scope_only": True,
            "scope_chunk_id_count": len(scope_chunk_ids),
            "scope_only_reason": "empty_expanded_text",
        }

    run_id = log_retrieval_run(
        conn,
        query_text=plan.query.raw,
        search_type=search_type,
        chunk_pv=filters.chunk_pv,
        embedding_model=embedding_model,
        top_k=k,
        returned_chunk_ids=chunk_ids,
        expanded_query_text=expanded_text,
        expansion_terms=None,
        expand_concordance=False,
        concordance_source_slug=None,
        query_lang_version=plan.query_lang_version if hasattr(plan, 'query_lang_version') else "qir_v1",
        retrieval_impl_version=plan.retrieval_impl_version if hasattr(plan, 'retrieval_impl_version') else "unknown",
        normalization_version=None,
        retrieval_config_json=retrieval_config_json,
        vector_metric="cosine" if search_type in ("vector", "hybrid") else None,
        embedding_dim=1536 if search_type in ("vector", "hybrid") else None,
        embed_text_version="embed_text_v1" if search_type in ("vector", "hybrid") else None,
        tsquery_text=tsquery_text,
        session_id=plan.session_id if hasattr(plan, 'session_id') else None,
        auto_commit=False,
    )
    
    # Insert evidence
    matched_lexemes = _parse_lexemes_from_tsquery(tsquery_text) if search_type in ("lex", "hybrid") else None
    _insert_run_chunk_evidence(
        conn,
        retrieval_run_id=run_id,
        hits=hits,
        search_type=search_type,
        tsquery_text=tsquery_text,
        matched_lexemes=matched_lexemes,
        embedding_model=embedding_model,
        chunk_pv=filters.chunk_pv,
        auto_commit=False,
    )
    
    conn.commit()
    return hits, run_id

# =============================================================================
# Result Set & Plan Status Updates
# =============================================================================

def create_result_set(
    conn,
    run_id: int,
    chunk_ids: List[int],
    name: Optional[str] = None,
    session_id: Optional[int] = None,
) -> int:
    """Create a result_set from retrieval run.
    
    Note: result_sets table is immutable (has trigger preventing updates),
    so all fields must be set during INSERT.
    """
    if not chunk_ids:
        raise ValueError("Cannot create result_set with empty chunk_ids (schema constraint)")
    if not name:
        name = f"Plan execution {run_id}"
    
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO result_sets (name, retrieval_run_id, chunk_ids, session_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (name, run_id, chunk_ids, session_id),
        )
        result_set_id = cur.fetchone()[0]
        conn.commit()
        return result_set_id

MAX_EXECUTION_HISTORY = 20  # Keep last N executions in _metadata.executions[]


def update_plan_status(
    conn,
    plan_id: int,
    run_id: int,
    result_set_id: Optional[int] = None,
    *,
    is_reexecution: bool = False,
    previous_run_id: Optional[int] = None,
    previous_result_set_id: Optional[int] = None,
    execution_mode: str = "retrieve",  # "retrieve" | "count"
):
    """
    Update plan status to executed and link to run/result_set.
    
    For re-executions (--force), appends previous execution to history in plan_json._metadata.
    Execution mode is stored in _metadata so downstream code knows if result_set exists.
    
    Args:
        execution_mode: "retrieve" (normal, creates result_set) or "count" (aggregation only, no result_set)
    """
    with conn.cursor() as cur:
        # Get current plan_json to update metadata
        cur.execute("SELECT plan_json FROM research_plans WHERE id = %s", (plan_id,))
        row = cur.fetchone()
        plan_json = row[0] if row else {}
        if isinstance(plan_json, str):
            import json as json_mod
            plan_json = json_mod.loads(plan_json)
        
        # Initialize metadata
        plan_json.setdefault("_metadata", {})
        
        # Store execution mode so downstream code knows if result_set exists
        plan_json["_metadata"]["execution_mode"] = execution_mode
        plan_json["_metadata"]["last_executed_at"] = __import__("datetime").datetime.now().isoformat()
        
        # If re-execution, append previous run to history (with limit)
        if is_reexecution and previous_run_id:
            plan_json["_metadata"].setdefault("executions", [])
            
            # Append previous execution to history
            plan_json["_metadata"]["executions"].append({
                "retrieval_run_id": previous_run_id,
                "result_set_id": previous_result_set_id,
                "execution_mode": plan_json["_metadata"].get("previous_execution_mode"),
                "superseded_at": __import__("datetime").datetime.now().isoformat(),
                "superseded_by_run_id": run_id,
            })
            
            # Limit history size to prevent unbounded growth
            if len(plan_json["_metadata"]["executions"]) > MAX_EXECUTION_HISTORY:
                # Keep only the most recent N entries
                plan_json["_metadata"]["executions"] = plan_json["_metadata"]["executions"][-MAX_EXECUTION_HISTORY:]
                plan_json["_metadata"]["executions_truncated"] = True
        
        # Store current execution_mode as previous for next re-execution
        plan_json["_metadata"]["previous_execution_mode"] = execution_mode
        
        # Update plan with all metadata
        cur.execute(
            """
            UPDATE research_plans
            SET status = 'executed',
                executed_at = now(),
                retrieval_run_id = %s,
                result_set_id = %s,
                plan_json = %s
            WHERE id = %s
            """,
            (run_id, result_set_id, Json(plan_json), plan_id),
        )
        conn.commit()

# =============================================================================
# Failure Handling
# =============================================================================

def record_execution_failure(
    conn,
    plan_id: int,
    error: Exception,
    *,
    partial_run_id: Optional[int] = None,
) -> None:
    """
    Record execution failure in plan metadata for debugging.
    
    Stores error details in plan_json._metadata.last_error
    without changing plan status (keeps as 'approved' for retry).
    """
    import traceback
    
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc()[-2000:],  # Limit traceback size
        "failed_at": __import__("datetime").datetime.now().isoformat(),
        "partial_run_id": partial_run_id,
    }
    
    try:
        with conn.cursor() as cur:
            # Get current plan_json
            cur.execute("SELECT plan_json FROM research_plans WHERE id = %s", (plan_id,))
            row = cur.fetchone()
            if not row:
                return
            
            plan_json = row[0]
            if isinstance(plan_json, str):
                import json as json_mod
                plan_json = json_mod.loads(plan_json)
            
            plan_json.setdefault("_metadata", {})
            plan_json["_metadata"]["last_error"] = error_info
            
            # Increment failure count
            plan_json["_metadata"]["failure_count"] = plan_json["_metadata"].get("failure_count", 0) + 1
            
            cur.execute(
                """
                UPDATE research_plans
                SET plan_json = %s
                WHERE id = %s
                """,
                (Json(plan_json), plan_id),
            )
            conn.commit()
    except Exception as record_err:
        # Don't let failure recording fail the main error path
        print(f"Warning: Could not record execution failure: {record_err}", file=sys.stderr)


def create_failed_retrieval_run(
    conn,
    session_id: int,
    query_text: str,
    error: Exception,
) -> int:
    """
    Create a retrieval_run record for a failed execution (for audit trail).
    
    Returns the run_id even for failures so we have a record.
    """
    # retrieval_runs schema does NOT have tsquery/run_metadata columns in this repo.
    # Use the canonical logger which matches the schema, and store details in retrieval_config_json.
    retrieval_config_json = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error)[:2000],
        "failed_at": __import__("datetime").datetime.now().isoformat(),
    }

    chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")

    run_id = log_retrieval_run(
        conn,
        query_text=f"[FAILED] {query_text}",
        search_type="lex",  # must satisfy retrieval_runs_search_type_check
        chunk_pv=chunk_pv,
        embedding_model=None,
        top_k=0,
        returned_chunk_ids=[],
        retrieval_config_json=retrieval_config_json,
        session_id=session_id,
        auto_commit=True,
    )
    return run_id


# =============================================================================
# COUNT-Mode Execution
# =============================================================================

def is_count_mode(plan: ResearchPlan) -> Tuple[bool, Optional[str]]:
    """
    Check if plan is in count/aggregation mode.
    
    Returns (is_count_mode, group_by) where group_by is the aggregation field or None.
    """
    for p in plan.query.primitives:
        if isinstance(p, CountPrimitive):
            return (True, p.group_by)
        if isinstance(p, SetQueryModePrimitive) and p.value in ("count", "aggregate"):
            return (True, None)
    return (False, None)


def execute_count_mode(
    conn,
    plan: ResearchPlan,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a COUNT-mode query without retrieving full results.
    
    Returns aggregation statistics instead of chunk hits.
    """
    if not plan.compiled:
        plan.compile()
    
    compiled = plan.compiled
    scope = compiled.get("scope", {})
    scope_where = scope.get("where_sql", "")
    scope_params = scope.get("params", [])
    required_joins = scope.get("required_joins", {})
    
    # Build base query
    from_clause = "FROM chunks c"
    join_clauses = []
    
    # Always join chunk_metadata (document_id is in cm, not c)
    join_clauses.append("JOIN chunk_metadata cm ON cm.chunk_id = c.id")
    
    # Add document join for document-level grouping (via chunk_metadata.document_id)
    join_clauses.append("JOIN documents d ON d.id = cm.document_id")
    join_clauses.append("JOIN collections col ON col.id = d.collection_id")
    
    joins_sql = " ".join(join_clauses)
    where_clause = f"WHERE {scope_where}" if scope_where else ""
    
    with conn.cursor() as cur:
        if group_by is None:
            # Simple count
            count_sql = f"""
                SELECT COUNT(DISTINCT c.id) as total_count
                {from_clause}
                {joins_sql}
                {where_clause}
            """
            cur.execute(count_sql, scope_params)
            total = cur.fetchone()[0]
            return {
                "mode": "count",
                "total_count": total,
                "group_by": None,
                "buckets": [],
            }
        
        elif group_by == "document":
            count_sql = f"""
                SELECT d.id, d.title, COUNT(DISTINCT c.id) as chunk_count
                {from_clause}
                {joins_sql}
                {where_clause}
                GROUP BY d.id, d.title
                ORDER BY chunk_count DESC
                LIMIT 50
            """
            cur.execute(count_sql, scope_params)
            buckets = [
                {"key": str(row[0]), "label": row[1] or f"Document {row[0]}", "count": row[2]}
                for row in cur.fetchall()
            ]
            total = sum(b["count"] for b in buckets)
            return {
                "mode": "count",
                "total_count": total,
                "group_by": "document",
                "buckets": buckets,
            }
        
        elif group_by == "collection":
            count_sql = f"""
                SELECT col.slug, col.title, COUNT(DISTINCT c.id) as chunk_count
                {from_clause}
                {joins_sql}
                {where_clause}
                GROUP BY col.slug, col.title
                ORDER BY chunk_count DESC
            """
            cur.execute(count_sql, scope_params)
            buckets = [
                {"key": row[0], "label": row[1] or row[0], "count": row[2]}
                for row in cur.fetchall()
            ]
            total = sum(b["count"] for b in buckets)
            return {
                "mode": "count",
                "total_count": total,
                "group_by": "collection",
                "buckets": buckets,
            }
        
        elif group_by == "entity":
            # Count by entity mentions in matching chunks
            count_sql = f"""
                SELECT em.entity_id, e.canonical_name, COUNT(DISTINCT c.id) as chunk_count
                {from_clause}
                {joins_sql}
                JOIN entity_mentions em ON em.chunk_id = c.id
                JOIN entities e ON e.id = em.entity_id
                {where_clause}
                GROUP BY em.entity_id, e.canonical_name
                ORDER BY chunk_count DESC
                LIMIT 50
            """
            cur.execute(count_sql, scope_params)
            buckets = [
                {"key": str(row[0]), "label": row[1], "count": row[2]}
                for row in cur.fetchall()
            ]
            # Get total distinct chunks
            total_sql = f"""
                SELECT COUNT(DISTINCT c.id)
                {from_clause}
                {joins_sql}
                {where_clause}
            """
            cur.execute(total_sql, scope_params)
            total = cur.fetchone()[0]
            return {
                "mode": "count",
                "total_count": total,
                "group_by": "entity",
                "buckets": buckets,
            }
        
        else:
            raise ValueError(f"Unsupported group_by: {group_by}")


# Compiler version for tracking changes to primitive compilation
COMPILER_VERSION = "primitives_v1.2"  # Increment when compilation logic changes


def _serialize_param_with_type(param) -> dict:
    """
    Serialize a parameter with type information for reproducibility.
    
    Returns dict with {value, pg_type} to preserve type fidelity
    when replaying queries.
    """
    if param is None:
        return {"value": None, "pg_type": "NULL"}
    elif isinstance(param, bool):
        return {"value": param, "pg_type": "boolean"}
    elif isinstance(param, int):
        return {"value": param, "pg_type": "integer"}
    elif isinstance(param, float):
        return {"value": param, "pg_type": "numeric"}
    elif isinstance(param, str):
        return {"value": param, "pg_type": "text"}
    elif hasattr(param, 'isoformat'):  # datetime/date
        return {"value": param.isoformat(), "pg_type": "timestamp" if hasattr(param, 'hour') else "date"}
    elif isinstance(param, (list, tuple)):
        return {"value": list(param), "pg_type": "array"}
    else:
        return {"value": str(param), "pg_type": "text"}


def log_count_execution(
    conn,
    plan: ResearchPlan,
    count_result: Dict[str, Any],
    *,
    plan_id: int,
    session_id: int,
) -> int:
    """
    Create a retrieval_run audit record for COUNT-mode execution.
    
    This ensures COUNT executions are auditable and reproducible.
    The run_metadata stores the aggregation parameters and results,
    INCLUDING scope parameters so the exact query can be reproduced.
    """
    compiled = plan.compiled or {}
    scope = compiled.get("scope", {})
    
    # Store COUNT execution details in retrieval_config_json (schema-supported)
    retrieval_config_json = {
        "mode": "count",
        "plan_id": plan_id,
        "compiler_version": COMPILER_VERSION,
        "compiled_at": __import__("datetime").datetime.now().isoformat(),
        "group_by": count_result.get("group_by"),
        "total_count": count_result.get("total_count"),
        "bucket_count": len(count_result.get("buckets", [])),
        "scope": {
            "where_sql": scope.get("where_sql", ""),
            "params_typed": [_serialize_param_with_type(p) for p in scope.get("params", [])],
            "required_joins": scope.get("required_joins", {}),
        },
        "primitive_types": [p.type for p in plan.query.primitives if hasattr(p, "type")],
    }

    chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
    expanded_text = (compiled.get("expanded", {}) or {}).get("expanded_text") if isinstance(compiled.get("expanded", {}), dict) else None
    tsquery_text = (compiled.get("tsquery", {}) or {}).get("text") if isinstance(compiled.get("tsquery", {}), dict) else None

    run_id = log_retrieval_run(
        conn,
        query_text=f"[COUNT] {plan.query.raw}",
        expanded_query_text=expanded_text,
        search_type="lex",  # schema constraint; this is a count-only run
        chunk_pv=chunk_pv,
        embedding_model=None,
        top_k=0,
        returned_chunk_ids=[],
        retrieval_config_json=retrieval_config_json,
        tsquery_text=tsquery_text,
        session_id=session_id,
        auto_commit=True,
    )
    return run_id


# =============================================================================
# Idempotent Execution Support
# =============================================================================

def get_existing_execution(conn, plan_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if a plan was already executed and return existing execution details.
    
    Returns dict with retrieval_run_id, result_set_id if executed, else None.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT retrieval_run_id, result_set_id, executed_at
            FROM research_plans
            WHERE id = %s AND status = 'executed'
            """,
            (plan_id,),
        )
        row = cur.fetchone()
        if row and row[0]:  # Has retrieval_run_id
            return {
                "retrieval_run_id": row[0],
                "result_set_id": row[1],
                "executed_at": row[2],
            }
    return None


def get_result_set_chunks(conn, result_set_id: int) -> List[int]:
    """Get chunk IDs from an existing result set."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_ids FROM result_sets WHERE id = %s",
            (result_set_id,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else []


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Execute an approved research plan")
    ap.add_argument("--plan-id", type=int, required=True, help="ID of plan to execute")
    ap.add_argument("--create-result-set", action="store_true", help="Create a result_set from execution results")
    ap.add_argument("--result-set-name", type=str, default=None, help="Name for result_set (default: auto-generated)")
    ap.add_argument("--force", action="store_true", help="Force re-execution even if already executed (creates new run/result_set)")
    ap.add_argument("--dry-run", action="store_true", help="Validate and compile plan without executing")
    ap.add_argument("--no-approval", action="store_true", help="[DEV ONLY] Execute proposed plans without approval (for testing)")
    ap.add_argument("--materialize", action="store_true", help="Convert COUNT-mode plan to full retrieval (drill-down from count results)")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        # =====================================================================
        # CRITICAL SECTION: Acquire plan execution lock
        # =====================================================================
        # Advisory lock is session-level and survives commits (unlike row locks).
        acquire_plan_advisory_lock(conn, args.plan_id)

        plan_data = get_plan(conn, args.plan_id, for_update=False)
        
        if not plan_data:
            print(f"ERROR: Plan {args.plan_id} not found", file=sys.stderr)
            sys.exit(1)
        
        # Check for idempotent execution (using locked row data)
        existing = None
        if plan_data["status"] == "executed" and plan_data.get("retrieval_run_id"):
            existing = {
                "retrieval_run_id": plan_data["retrieval_run_id"],
                "result_set_id": plan_data.get("result_set_id"),
                "executed_at": plan_data.get("executed_at"),
            }
        
        if existing and not args.force:
            print(f"Plan {args.plan_id} was already executed at {existing['executed_at']}", file=sys.stderr)
            print(f"   Retrieval Run ID: {existing['retrieval_run_id']}", file=sys.stderr)
            if existing.get('result_set_id'):
                print(f"   Result Set ID: {existing['result_set_id']}", file=sys.stderr)
                chunk_ids = get_result_set_chunks(conn, existing['result_set_id'])
                print(f"   Chunk count: {len(chunk_ids)}", file=sys.stderr)
            print(f"\nUse --force to re-execute (will create new run/result_set)", file=sys.stderr)
            sys.exit(0)
        
        # Verify status (allow 'executed' if --force, 'proposed' if --no-approval)
        allowed_statuses = ["approved"]
        if args.force:
            allowed_statuses.append("executed")
        if args.no_approval:
            allowed_statuses.append("proposed")
            print("⚠️  [DEV MODE] --no-approval: Executing without approval check", file=sys.stderr)
        
        if plan_data["status"] not in allowed_statuses:
            status_msg = "must be 'approved'"
            if args.force:
                status_msg += " (or 'executed' with --force)"
            print(f"ERROR: Plan {args.plan_id} status is '{plan_data['status']}', {status_msg}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Executing Plan ID: {args.plan_id} (session={plan_data['session_id']})", file=sys.stderr)
        print(f"Utterance: {plan_data['user_utterance']}", file=sys.stderr)
        if args.force and existing:
            print(f"⚠️  Re-executing (--force): Previous run={existing['retrieval_run_id']}", file=sys.stderr)
        
        # Validate plan before execution
        validation_errors = validate_plan(plan_data["plan_json"])
        if validation_errors:
            print(f"ERROR: Plan {args.plan_id} validation failed:", file=sys.stderr)
            for err in validation_errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)

        # Manual smoke-test hook: induce a controlled failure after we've loaded the plan
        # (so failure audit logging can run).
        if os.getenv("NEH_INDUCE_FAILURE") == "1":
            raise RuntimeError("Induced failure (NEH_INDUCE_FAILURE=1)")
        
        # Parse plan
        plan = ResearchPlan.from_dict(plan_data["plan_json"])
        plan.query.raw = plan_data["plan_json"]["query"]["raw"]
        plan.compile()
        
        # Add metadata
        plan.query_lang_version = plan_data["query_lang_version"]
        plan.retrieval_impl_version = plan_data["retrieval_impl_version"]
        plan.session_id = plan_data["session_id"]
        
        if args.dry_run:
            print("\n[DRY RUN] Plan validated and compiled successfully", file=sys.stderr)
            print(f"  Compiled components: {list(plan.compiled.keys()) if plan.compiled else 'None'}", file=sys.stderr)
            count_mode, group_by = is_count_mode(plan)
            if count_mode:
                print(f"  Mode: COUNT (group_by={group_by})", file=sys.stderr)
            sys.exit(0)
        
        # Check for COUNT mode
        count_mode, group_by = is_count_mode(plan)
        
        if count_mode and not args.materialize:
            # COUNT mode: return aggregation statistics without full retrieval
            print(f"Executing COUNT mode (group_by={group_by})...", file=sys.stderr)
            count_result = execute_count_mode(conn, plan, group_by)
            
            # Create audit trail (retrieval_run) even for COUNT mode
            # This ensures provenance and reproducibility
            count_run_id = log_count_execution(
                conn,
                plan,
                count_result,
                plan_id=args.plan_id,
                session_id=plan_data["session_id"],
            )
            count_result["retrieval_run_id"] = count_run_id
            
            # Update plan to mark as executed (with run_id, no result_set)
            # execution_mode="count" signals downstream that result_set_id will be NULL
            update_plan_status(
                conn, args.plan_id, count_run_id, None,
                is_reexecution=bool(existing and args.force),
                previous_run_id=existing["retrieval_run_id"] if existing else None,
                previous_result_set_id=existing.get("result_set_id") if existing else None,
                execution_mode="count",
            )
            
            print(f"\n✅ COUNT execution complete", file=sys.stderr)
            print(f"   Run ID: {count_run_id}", file=sys.stderr)
            print(f"   Total count: {count_result['total_count']}", file=sys.stderr)
            
            if count_result.get("buckets"):
                print(f"\n=== Aggregation by {group_by} ===", file=sys.stderr)
                for b in count_result["buckets"][:10]:
                    print(f"   {b['label']}: {b['count']}", file=sys.stderr)
                if len(count_result["buckets"]) > 10:
                    print(f"   ... and {len(count_result['buckets']) - 10} more", file=sys.stderr)
            
            # Output JSON for programmatic use
            import json
            print(json.dumps(count_result))
            sys.exit(0)
        
        # Handle --materialize for COUNT-mode plans (drill-down to full retrieval)
        if count_mode and args.materialize:
            print(f"📊 Materializing COUNT-mode plan to full retrieval (drill-down)...", file=sys.stderr)
        
        # Execute retrieval (no session queries - all params from envelope)
        print("Executing retrieval...", file=sys.stderr)
        hits, run_id = execute_plan_retrieval(
            conn,
            plan,
        )
        
        print(f"Retrieval complete: {len(hits)} chunks, run_id={run_id}", file=sys.stderr)
        
        # Create result_set if requested (or by default if first execution)
        result_set_id = None
        should_create_result_set = args.create_result_set or (not existing)  # Always create on first exec
        
        if should_create_result_set:
            chunk_ids = [h.chunk_id for h in hits]
            if not chunk_ids:
                print("No chunks returned; skipping result_set creation (result_sets.chunk_ids is non-empty).", file=sys.stderr)
                result_set_id = None
            else:
                # Use unique name including run_id (result_sets.name has unique constraint)
                result_set_name = args.result_set_name or f"Plan {args.plan_id} run {run_id}"
                result_set_id = create_result_set(
                    conn,
                    run_id,
                    chunk_ids,
                    name=result_set_name,
                    session_id=plan_data["session_id"],  # Included in INSERT (table is immutable)
                )
                print(f"Result set created: ID={result_set_id}", file=sys.stderr)
        
        # Update plan status (with execution history tracking for re-executions)
        update_plan_status(
            conn, args.plan_id, run_id, result_set_id,
            is_reexecution=bool(existing and args.force),
            previous_run_id=existing["retrieval_run_id"] if existing else None,
            previous_result_set_id=existing.get("result_set_id") if existing else None,
            execution_mode="retrieve",
        )
        
        print(f"\n✅ Plan {args.plan_id} executed successfully", file=sys.stderr)
        print(f"   Run ID: {run_id}", file=sys.stderr)
        if result_set_id:
            print(f"   Result Set ID: {result_set_id}", file=sys.stderr)
        
        # Display results summary
        print(f"\n=== Results ({len(hits)} chunks) ===", file=sys.stderr)
        for i, h in enumerate(hits[:10], 1):  # Show first 10
            print(f"[{i}] chunk_id={h.chunk_id}  collection={h.collection_slug}  doc={h.document_id}", file=sys.stderr)
            if h.preview:
                print(f"    {h.preview[:200].replace(chr(10), ' ')}...", file=sys.stderr)
        if len(hits) > 10:
            print(f"... and {len(hits) - 10} more", file=sys.stderr)
    
    except PlanExecutionLockError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        # Record failure for audit trail
        print(f"\n❌ Execution failed: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            conn.rollback()
        except Exception:
            pass
        
        try:
            # Create failed retrieval_run for audit
            if 'plan_data' in dir() and plan_data:
                failed_run_id = create_failed_retrieval_run(
                    conn,
                    plan_data["session_id"],
                    plan_data["user_utterance"],
                    e,
                )
                print(f"   Failed run logged: {failed_run_id}", file=sys.stderr)
                
                # Record error in plan metadata
                record_execution_failure(conn, args.plan_id, e, partial_run_id=failed_run_id)
                print(f"   Error recorded in plan metadata", file=sys.stderr)
        except Exception as record_err:
            print(f"   Warning: Could not record failure: {record_err}", file=sys.stderr)
        
        sys.exit(1)
    
    finally:
        try:
            release_plan_advisory_lock(conn, args.plan_id)
        except Exception:
            pass
        conn.close()

if __name__ == "__main__":
    main()
