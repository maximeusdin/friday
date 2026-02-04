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
                SELECT d.id, d.source_name, COUNT(DISTINCT c.id) as chunk_count
                {from_clause}
                {joins_sql}
                {where_clause}
                GROUP BY d.id, d.source_name
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

# =============================================================================
# Phase 6: Mode-Aware Execution
# =============================================================================

def execute_plan_retrieval_v2(
    conn,
    plan: ResearchPlan,
    *,
    ui_toggle_mode: Optional[str] = None,
    force_rerun: bool = False,
    existing_result_set_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute plan retrieval with mode-aware logic (Two-Mode Retrieval System).
    
    Mode Precedence (highest to lowest):
    1. ui_toggle_mode - explicit user selection
    2. SET_RETRIEVAL_MODE primitive - from plan
    3. Trigger phrase detection - from utterance
    4. Default - conversational
    
    Rerun Semantics:
    - If existing_result_set_id and not force_rerun: return existing (idempotent)
    - If force_rerun: create new result_set (audit trail preserved)
    
    Args:
        conn: Database connection
        plan: Compiled research plan
        ui_toggle_mode: Mode from UI toggle (if set)
        force_rerun: Force new execution even if result exists
        existing_result_set_id: Existing result set ID (for idempotent reuse)
        
    Returns:
        {
            "result_set_id": int,
            "retrieval_run_id": int,
            "mode": str,
            "mode_source": str,
            "total_results": int,
            "total_before_cap": int,
            "cap_applied": bool,
            "threshold_used": float,
            "vector_metric": str,
            "traces_count": int,
            "was_reused": bool,
        }
    """
    from retrieval.config import (
        resolve_retrieval_mode, get_mode_config, DEFAULT_VECTOR_CONFIG,
    )
    from retrieval.primitives import SetRetrievalModePrimitive, SetSimilarityThresholdPrimitive
    from retrieval.ops import hybrid_rrf_sql, compute_thorough_rank, ThresholdSearchResult
    from retrieval.match_trace import (
        build_base_traces, enrich_tier1_entity_ids, enrich_tier2_for_summarization,
        persist_match_traces,
    )
    
    # Idempotent rerun check
    if existing_result_set_id and not force_rerun:
        # Reuse existing result - load metadata
        with conn.cursor() as cur:
            cur.execute("""
                SELECT rs.id, rs.retrieval_run_id, rr.retrieval_mode, rr.mode_source,
                       array_length(rs.chunk_ids, 1) as total_results
                FROM result_sets rs
                JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id
                WHERE rs.id = %s
            """, (existing_result_set_id,))
            row = cur.fetchone()
            if row:
                return {
                    "result_set_id": row[0],
                    "retrieval_run_id": row[1],
                    "mode": row[2] or "conversational",
                    "mode_source": row[3] or "default",
                    "total_results": row[4] or 0,
                    "total_before_cap": row[4] or 0,
                    "cap_applied": False,
                    "threshold_used": 0.0,
                    "vector_metric": "cosine",
                    "traces_count": 0,
                    "was_reused": True,
                }
    
    # Ensure plan is compiled
    if not plan.compiled:
        plan.compile()
    
    # 1. Extract primitive mode if present
    primitive_mode = None
    custom_threshold = None
    entity_primitive_ids = []
    
    for p in plan.query.primitives:
        if isinstance(p, SetRetrievalModePrimitive):
            primitive_mode = p.mode
        elif isinstance(p, SetSimilarityThresholdPrimitive):
            custom_threshold = p.threshold
        # Collect entity IDs for trace enrichment
        ptype = getattr(p, 'type', None)
        if ptype == "ENTITY":
            entity_primitive_ids.append(p.entity_id)
    
    # 2. Get utterance for trigger detection
    utterance = plan.query.raw or ""
    
    # 3. Resolve mode with precedence
    mode, mode_source = resolve_retrieval_mode(ui_toggle_mode, primitive_mode, utterance)
    
    # 4. Load config
    config = get_mode_config(mode)
    vector_config = DEFAULT_VECTOR_CONFIG
    
    # Apply custom threshold if set
    threshold = custom_threshold if custom_threshold is not None else config.similarity_threshold
    
    # Determine cap based on mode
    max_hits = config.answer_k if mode == "conversational" else None
    
    # 5. Get compiled components
    compiled = plan.compiled
    expanded = compiled.get("expanded", {})
    expanded_text = expanded.get("expanded_text", plan.query.raw)
    scope = compiled.get("scope", {})
    scope_sql = scope.get("where_sql") if scope else None
    scope_params = scope.get("params", []) if scope else []
    
    # Build filters
    collection_slugs = None
    for p in plan.query.primitives:
        if hasattr(p, "type") and p.type == "FILTER_COLLECTION":
            if collection_slugs is None:
                collection_slugs = []
            collection_slugs.append(p.slug)
    
    filters = SearchFilters(
        chunk_pv="chunk_v1_full",
        collection_slugs=collection_slugs,
    )
    
    # 6. Execute search
    if expanded_text and expanded_text.strip():
        # Get embedding for query
        embedding = embed_query(expanded_text)
        
        # Use threshold-based hybrid search
        results, metadata = hybrid_rrf_sql(
            conn,
            expanded_text,
            embedding,
            similarity_threshold=threshold,
            combine_mode="union",
            max_hits=max_hits,
            scope_sql=scope_sql,
            scope_params=scope_params,
            vector_config=vector_config,
            filters=filters,
            rrf_k=config.rrf_k,
            retrieval_mode=mode,
        )
        
        # Apply thorough mode ranking if needed
        if mode == "thorough" and results:
            results = compute_thorough_rank(results)
    else:
        # Scope-only execution (no search text)
        results = []
        metadata = None
        
        if scope_sql:
            with conn.cursor() as cur:
                query = f"""
                    SELECT DISTINCT c.id, cm.document_id
                    FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE {scope_sql}
                    ORDER BY COALESCE(cm.document_id, 2147483647), c.id
                """
                cur.execute(query, scope_params)
                rows = cur.fetchall()
                
                for i, row in enumerate(rows, 1):
                    results.append(ThresholdSearchResult(
                        chunk_id=row[0],
                        document_id=row[1],
                        rank=i,
                        in_lexical=False,
                        in_vector=False,
                    ))
    
    # 7. Build and enrich traces
    from retrieval.match_trace import ThresholdSearchMetadata as TSM
    mock_metadata = metadata if metadata else TSM(mode=mode, cap_applied=False)
    
    traces = build_base_traces(results, mock_metadata, mode)
    traces = enrich_tier1_entity_ids(traces, entity_primitive_ids, conn)
    
    # Tier 2 enrichment for summarization (top N only)
    summarization_limit = config.answer_k if hasattr(config, 'answer_k') else 20
    traces = enrich_tier2_for_summarization(traces, plan.query.primitives, conn, limit=summarization_limit)
    
    # 8. Log retrieval run
    chunk_ids = [r.chunk_id for r in results]
    embedding_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    
    run_id = log_retrieval_run(
        conn,
        query_text=expanded_text or plan.query.raw,
        search_type="hybrid" if expanded_text else "lex",
        chunk_pv=filters.chunk_pv or "all",
        embedding_model=embedding_model,
        top_k=len(chunk_ids),
        returned_chunk_ids=chunk_ids,
        expanded_query_text=expanded_text,
        expansion_terms=None,
        expand_concordance=False,
        concordance_source_slug=None,
        query_lang_version="qv3_two_mode",
        retrieval_impl_version="retrieval_v3_two_mode",
        normalization_version=None,
        vector_metric=vector_config.metric,
        embedding_dim=1536,
        embed_text_version="embed_text_v1",
        retrieval_config_json={
            "mode": mode,
            "mode_source": mode_source,
            "threshold": threshold,
            "rrf_k": config.rrf_k,
        },
        tsquery_text=None,
        auto_commit=False,
    )
    
    # 9. Create result set
    result_set_name = f"Plan {plan.query.raw[:50] if plan.query.raw else 'unknown'}... (mode={mode})"
    result_set_id = create_result_set(
        conn,
        run_id,
        chunk_ids,
        name=result_set_name,
    )
    
    # 10. Persist traces
    traces_count = persist_match_traces(traces, result_set_id, run_id, conn)
    
    # 11. Populate result_set_chunks for pagination
    with conn.cursor() as cur:
        cur.execute("SELECT populate_result_set_chunks(%s, %s)", (result_set_id, mode))
    
    conn.commit()
    
    return {
        "result_set_id": result_set_id,
        "retrieval_run_id": run_id,
        "mode": mode,
        "mode_source": mode_source,
        "total_results": len(results),
        "total_before_cap": metadata.total_before_cap if metadata else len(results),
        "cap_applied": metadata.cap_applied if metadata else False,
        "threshold_used": threshold,
        "vector_metric": vector_config.metric,
        "traces_count": traces_count,
        "was_reused": False,
    }


# =============================================================================
# Index Retrieval Execution
# =============================================================================

def execute_index_retrieval(
    conn,
    primitive,  # Union[FirstMentionPrimitive, FirstCoMentionPrimitive, MentionsPrimitive, ...]
    *,
    mode: str = "conversational",
    scope_sql: Optional[str] = None,
    scope_params: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    after_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute an index retrieval primitive and create a result set.
    
    This is the main entry point for index-based retrieval (as opposed to
    search-based retrieval via execute_plan_retrieval_v2).
    
    Args:
        conn: Database connection
        primitive: An index primitive (FIRST_MENTION, FIRST_CO_MENTION, MENTIONS, etc.)
        mode: "conversational" or "thorough"
        scope_sql: Optional SQL WHERE clause for additional filtering
        scope_params: Parameters for scope SQL
        limit: Override for pagination limit
        after_rank: Cursor for pagination
        
    Returns:
        Dict with result_set_id, retrieval_run_id, metadata, etc.
    """
    from retrieval.primitives import (
        FirstMentionPrimitive, FirstCoMentionPrimitive, MentionsPrimitive,
        DateRangeFilterPrimitive, DateMentionsPrimitive, FirstDateMentionPrimitive,
        PlaceMentionsPrimitive, RelatedPlacesPrimitive,
    )
    from retrieval.index_ops import (
        first_mention, first_co_mention, mentions_paginated,
        date_range_filter, first_date_mention, place_mentions, related_places,
        IndexHit, IndexMetadata,
    )
    from retrieval.match_trace import MatchTrace, persist_match_traces
    from retrieval.config import ConversationalModeConfig, ThoroughModeConfig
    
    # Get mode config for limits
    if mode == "conversational":
        config = ConversationalModeConfig()
        effective_limit = limit or config.answer_k
    else:
        config = ThoroughModeConfig()
        effective_limit = limit or config.pagination_default_limit
    
    # Route to appropriate index operation
    hits: List[IndexHit] = []
    metadata: IndexMetadata = IndexMetadata()
    primitive_type = type(primitive).__name__
    
    if isinstance(primitive, FirstMentionPrimitive):
        hit, metadata = first_mention(
            conn,
            primitive.entity_id,
            scope_sql=scope_sql,
            scope_params=scope_params,
            order_by=primitive.order_by,
        )
        if hit:
            hits = [hit]
    
    elif isinstance(primitive, FirstCoMentionPrimitive):
        hit, metadata = first_co_mention(
            conn,
            primitive.entity_ids,
            window=primitive.window,
            scope_sql=scope_sql,
            scope_params=scope_params,
            order_by=primitive.order_by,
        )
        if hit:
            hits = [hit]
    
    elif isinstance(primitive, MentionsPrimitive):
        hits, metadata = mentions_paginated(
            conn,
            primitive.entity_id,
            scope_sql=scope_sql,
            scope_params=scope_params,
            order_by=primitive.order_by,
            limit=effective_limit,
            after_rank=after_rank,
        )
    
    elif isinstance(primitive, DateRangeFilterPrimitive):
        from datetime import date as date_type
        date_start = None
        date_end = None
        if primitive.date_start:
            date_start = date_type.fromisoformat(primitive.date_start) if isinstance(primitive.date_start, str) else primitive.date_start
        if primitive.date_end:
            date_end = date_type.fromisoformat(primitive.date_end) if isinstance(primitive.date_end, str) else primitive.date_end
        
        hits, metadata = date_range_filter(
            conn,
            date_start=date_start,
            date_end=date_end,
            time_basis=primitive.time_basis,
            scope_sql=scope_sql,
            scope_params=scope_params,
            limit=effective_limit,
            after_rank=after_rank,
        )
    
    elif isinstance(primitive, DateMentionsPrimitive):
        from datetime import date as date_type
        date_start = None
        date_end = None
        if primitive.date_start:
            date_start = date_type.fromisoformat(primitive.date_start) if isinstance(primitive.date_start, str) else primitive.date_start
        if primitive.date_end:
            date_end = date_type.fromisoformat(primitive.date_end) if isinstance(primitive.date_end, str) else primitive.date_end
        
        hits, metadata = date_range_filter(
            conn,
            date_start=date_start,
            date_end=date_end,
            time_basis=primitive.time_basis,
            scope_sql=scope_sql,
            scope_params=scope_params,
            limit=effective_limit,
            after_rank=after_rank,
        )
        metadata.primitive_type = "DATE_MENTIONS"
    
    elif isinstance(primitive, FirstDateMentionPrimitive):
        hit, metadata = first_date_mention(
            conn,
            primitive.entity_id,
            time_basis=primitive.time_basis,
            scope_sql=scope_sql,
            scope_params=scope_params,
        )
        if hit:
            hits = [hit]
    
    elif isinstance(primitive, PlaceMentionsPrimitive):
        hits, metadata = place_mentions(
            conn,
            primitive.place_entity_id,
            scope_sql=scope_sql,
            scope_params=scope_params,
            order_by=primitive.order_by,
            limit=effective_limit,
            after_rank=after_rank,
        )
    
    elif isinstance(primitive, RelatedPlacesPrimitive):
        # Related places returns aggregated data, not individual hits
        scope_chunk_ids = None
        if primitive.scope and "result_set_id" in primitive.scope:
            # Get chunk IDs from result set
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id FROM result_set_chunks WHERE result_set_id = %s",
                    (primitive.scope["result_set_id"],)
                )
                scope_chunk_ids = [row[0] for row in cur.fetchall()]
        
        results, metadata = related_places(
            conn,
            primitive.entity_id,
            window=primitive.window,
            top_n=primitive.top_n,
            scope_chunk_ids=scope_chunk_ids,
        )
        
        # For analysis primitives, we return the aggregated results directly
        return {
            "primitive_type": primitive_type,
            "mode": mode,
            "results": results,
            "metadata": metadata.to_dict(),
            "total_results": len(results),
        }
    
    else:
        raise ValueError(f"Unknown index primitive type: {primitive_type}")
    
    # Build match traces for index hits
    traces = _build_index_match_traces(hits, metadata, primitive_type)
    
    # Create retrieval run
    chunk_ids = [h.chunk_id for h in hits]
    
    run_id = log_retrieval_run(
        conn,
        query_text=f"[INDEX] {primitive_type}",
        search_type="index",
        chunk_pv="all",
        embedding_model=None,
        top_k=len(chunk_ids),
        returned_chunk_ids=chunk_ids,
        expanded_query_text=None,
        expansion_terms=None,
        expand_concordance=False,
        concordance_source_slug=None,
        query_lang_version="qv3_index_retrieval",
        retrieval_impl_version="retrieval_v3_index",
        normalization_version=None,
        vector_metric=None,
        embedding_dim=None,
        embed_text_version=None,
        retrieval_config_json=_build_retrieval_config(
            mode=mode,
            primitive_type=primitive_type,
            metadata=metadata,
        ),
        tsquery_text=None,
        auto_commit=False,
    )
    
    # Create result set with unique name
    import time
    result_set_name = f"Index: {primitive_type} (mode={mode}) @ {int(time.time() * 1000)}"
    result_set_id = create_result_set(
        conn,
        run_id,
        chunk_ids,
        name=result_set_name,
    )
    
    # Persist traces
    traces_count = persist_match_traces(traces, result_set_id, run_id, conn)
    
    # Populate result_set_chunks for pagination
    # Note: For index retrieval, ranks are already assigned by the index ops
    with conn.cursor() as cur:
        # Insert directly with pre-computed ranks
        for hit in hits:
            cur.execute("""
                INSERT INTO result_set_chunks (result_set_id, chunk_id, rank)
                VALUES (%s, %s, %s)
                ON CONFLICT (result_set_id, chunk_id) DO NOTHING
            """, (result_set_id, hit.chunk_id, hit.rank))
    
    conn.commit()
    
    return {
        "result_set_id": result_set_id,
        "retrieval_run_id": run_id,
        "mode": mode,
        "primitive_type": primitive_type,
        "total_results": len(hits),
        "total_before_cap": metadata.total_hits,
        "metadata": metadata.to_dict(),
        "traces_count": traces_count,
    }


def _build_retrieval_config(
    mode: str,
    primitive_type: str,
    metadata,  # IndexMetadata
) -> dict:
    """
    Build retrieval config with signature hash for determinism verification.
    
    The signature_hash allows comparing whether two queries have identical semantics,
    which is critical for:
    - Verifying expand preserves original semantics
    - Debugging why results differ between runs
    - Caching/deduplication
    """
    import hashlib
    import json
    from retrieval.chronology import CHRONOLOGY_V1_KEY
    
    config = {
        "mode": mode,
        "primitive_type": primitive_type,
        "source": metadata.source,
        "order_by": metadata.order_by,
        "dedupe_policy": metadata.dedupe_policy,
        "span_policy": metadata.span_policy,
        "retrieval_source": metadata.source,
        "time_basis": metadata.time_basis,
        "geo_basis": metadata.geo_basis,
        "chronology_version": CHRONOLOGY_V1_KEY,
    }
    
    # Compute signature hash for determinism verification
    # Includes all semantically-significant fields
    hash_input = json.dumps({
        "source": config["source"],
        "order_by": config["order_by"],
        "dedupe_policy": config["dedupe_policy"],
        "span_policy": config["span_policy"],
        "time_basis": config["time_basis"],
        "geo_basis": config["geo_basis"],
        "chronology_version": config["chronology_version"],
    }, sort_keys=True)
    
    config["retrieval_signature_hash"] = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    return config


def _build_index_match_traces(
    hits: List,  # List[IndexHit]
    metadata,  # IndexMetadata
    primitive_type: str,
) -> List:
    """
    Build match traces for index retrieval hits.
    
    These are simpler than search traces - no scores, just entity IDs and
    source information.
    """
    from retrieval.match_trace import MatchTrace
    
    traces = []
    for hit in hits:
        # Build audit dict for rank_trace
        audit = {
            "source": metadata.source,
            "primitive": primitive_type,
            "order_by": metadata.order_by,
            "order_sql": metadata.order_sql,
            "retrieval_type": "index",
            # Explicit deduplication policy documentation
            "dedupe_policy": metadata.dedupe_policy,
            "span_policy": metadata.span_policy,
        }
        
        if metadata.time_basis:
            audit["time_basis"] = metadata.time_basis
        if metadata.date_range:
            audit["date_range"] = metadata.date_range
        if metadata.geo_basis:
            audit["geo_basis"] = metadata.geo_basis
        if hit.mention_surface:
            audit["mention_surface"] = hit.mention_surface
        
        trace = MatchTrace(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            rank=hit.rank,
            
            # Hot columns
            matched_entity_ids=hit.entity_ids,
            matched_phrases=[hit.mention_surface] if hit.mention_surface else [],
            scope_passed=True,
            
            # For index retrieval, in_lexical and in_vector are False
            in_lexical=False,
            in_vector=False,
            
            # Audit/trace info stored in rank_trace
            rank_trace=audit,
            
            # No scores for index retrieval
            score_lexical=None,
            score_vector=None,
            score_hybrid=None,
        )
        traces.append(trace)
    
    return traces


def is_index_primitive(primitive) -> bool:
    """
    Check if a primitive is an index retrieval primitive.
    
    Index primitives query mention indexes directly, not vector/lexical search.
    """
    from retrieval.primitives import (
        FirstMentionPrimitive, FirstCoMentionPrimitive, MentionsPrimitive,
        DateRangeFilterPrimitive, DateMentionsPrimitive, FirstDateMentionPrimitive,
        PlaceMentionsPrimitive, RelatedPlacesPrimitive, WithinCountryPrimitive,
    )
    
    return isinstance(primitive, (
        FirstMentionPrimitive,
        FirstCoMentionPrimitive,
        MentionsPrimitive,
        DateRangeFilterPrimitive,
        DateMentionsPrimitive,
        FirstDateMentionPrimitive,
        PlaceMentionsPrimitive,
        RelatedPlacesPrimitive,
        WithinCountryPrimitive,
    ))


# =============================================================================
# Agentic Workflow Execution
# =============================================================================

# Feature flag for agentic mode (enabled by default)
AGENTIC_MODE_ENABLED = os.getenv("AGENTIC_MODE_ENABLED", "1") == "1"


def execute_plan_agentic(
    conn,
    plan_data: Dict[str, Any],
    *,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Execute a plan using the full agentic workflow.
    
    Architecture: Plan → Execute → Verify → Render
    
    Features:
    - Iterative multi-round execution with Jaccard+novelty stability
    - Multi-lane retrieval (entity, lexical, hybrid, ephemeral expansion, co-mention)
    - Deterministic claim extraction with optional LLM refinement
    - Intent-specific verification with claim-scoped role evidence
    - Per-bullet citations in rendered answer
    
    Args:
        conn: Database connection
        plan_data: Plan data with user_utterance, plan_json, session_id
        max_retries: Maximum verification retry attempts
        
    Returns:
        Dict with evidence_bundle, verification_result, rendered_answer, etc.
    """
    from retrieval.plan import (
        AgenticPlan, LaneSpec, ExtractionSpec, VerificationSpec,
        Budgets, StopConditions, BundleConstraints,
        build_default_lanes_for_intent, build_verification_spec_for_intent,
    )
    from retrieval.intent import (
        IntentFamily, classify_intent, extract_anchors,
    )
    from retrieval.lanes import execute_plan_iterative
    from retrieval.claim_extraction import extract_claims
    from retrieval.codename_resolution import resolve_codenames
    from retrieval.verifier import verify_evidence_bundle, filter_claims_by_verification
    from retrieval.answer_trace import generate_answer_trace
    from backend.app.services.summarizer.synthesis import render_from_bundle
    
    user_text = plan_data.get("user_utterance", "")
    session_id = plan_data.get("session_id")
    
    # Step 1: Classify intent
    print("  [Agentic] Classifying intent...", file=sys.stderr)
    intent_result = classify_intent(user_text, [])
    
    # Step 2: Extract anchors and build constraints
    anchors = intent_result.anchors
    constraints = BundleConstraints(
        collection_scope=anchors.constraints.get("collection_scope", []),
        required_role_evidence_patterns=anchors.constraints.get("role_evidence_patterns", []),
    )
    
    # Step 3: Build lanes based on intent
    lanes = build_default_lanes_for_intent(
        intent=intent_result.intent_family,
        entity_ids=anchors.target_entities,
        query_terms=anchors.key_concepts + anchors.target_tokens,
        collection_scope=constraints.collection_scope,
    )
    
    # Step 4: Build verification spec
    verification = build_verification_spec_for_intent(
        intent=intent_result.intent_family,
        role_patterns=constraints.required_role_evidence_patterns,
        collection_scope=constraints.collection_scope,
    )
    
    # Step 5: Build the typed plan
    agentic_plan = AgenticPlan(
        intent=intent_result.intent_family,
        constraints=constraints,
        lanes=lanes,
        extraction=ExtractionSpec(
            predicates=[],
            allow_llm_refinement=False,
        ),
        verification=verification,
        budgets=Budgets(),
        stop_conditions=StopConditions(),
        query_text=user_text,
        session_id=str(session_id) if session_id else None,
    )
    
    # Validate plan
    plan_errors = agentic_plan.validate()
    if plan_errors:
        raise ValueError(f"Invalid agentic plan: {plan_errors}")
    
    print(f"  [Agentic] Intent: {intent_result.intent_family.value}, "
          f"Lanes: {len(lanes)}, Confidence: {intent_result.confidence:.2f}", file=sys.stderr)
    
    # Step 6: Execute iterative retrieval
    print("  [Agentic] Executing iterative retrieval...", file=sys.stderr)
    bundle = execute_plan_iterative(agentic_plan, conn)
    
    print(f"  [Agentic] Retrieved {len(bundle.all_chunks)} chunks, "
          f"{len(bundle.entities)} entities in {bundle.rounds_executed} rounds", file=sys.stderr)
    
    # Step 7: Extract claims
    print("  [Agentic] Extracting claims...", file=sys.stderr)
    chunks_list = list(bundle.all_chunks.values())
    candidates_dict = {e.key: e for e in bundle.entities}
    
    claims = extract_claims(
        chunks=chunks_list,
        candidates=candidates_dict,
        intent=intent_result.intent_family,
        extraction_spec=agentic_plan.extraction,
        conn=conn,
    )
    bundle.claims = claims
    
    print(f"  [Agentic] Extracted {len(claims)} claims", file=sys.stderr)
    
    # Step 8: Resolve codenames if any unresolved tokens
    if bundle.unresolved_tokens:
        print(f"  [Agentic] Resolving {len(bundle.unresolved_tokens)} codenames...", file=sys.stderr)
        resolution = resolve_codenames(
            bundle.unresolved_tokens,
            constraints.collection_scope,
            conn,
        )
        # Add resolved claims
        bundle.claims.extend(resolution.claims)
        bundle.unresolved_tokens = resolution.unresolved
    
    # Step 9: Verify with retry loop
    print("  [Agentic] Verifying evidence bundle...", file=sys.stderr)
    verification_result = None
    
    for retry in range(max_retries + 1):
        verification_result = verify_evidence_bundle(bundle, conn)
        
        if verification_result.passed:
            print(f"  [Agentic] Verification passed!", file=sys.stderr)
            break
        
        if retry < max_retries:
            print(f"  [Agentic] Verification failed, retry {retry + 1}/{max_retries}...", file=sys.stderr)
            # Filter out invalid claims and continue
            verified_claims, dropped = filter_claims_by_verification(
                bundle.claims, constraints
            )
            bundle.claims = verified_claims
        else:
            print(f"  [Agentic] Verification failed after {max_retries} retries", file=sys.stderr)
    
    # Step 10: Generate answer trace
    trace = generate_answer_trace(bundle, verification_result)
    
    # Step 11: Render answer from verified bundle
    print("  [Agentic] Rendering answer...", file=sys.stderr)
    rendered = render_from_bundle(bundle)
    
    print(f"  [Agentic] Complete: {rendered.claims_rendered} claims, "
          f"{rendered.citations_included} citations", file=sys.stderr)
    
    return {
        "mode": "agentic",
        "evidence_bundle": bundle.to_dict(),
        "verification_passed": verification_result.passed if verification_result else False,
        "verification_result": verification_result.to_dict() if verification_result else None,
        "rendered_answer": rendered.to_dict(),
        "answer_trace": trace.to_dict(),
        "intent": intent_result.to_dict(),
        "plan": agentic_plan.to_dict(),
    }


def store_evidence_bundle(
    conn,
    result_set_id: int,
    bundle_data: Dict[str, Any],
    plan_data: Dict[str, Any],
    verification_data: Optional[Dict[str, Any]],
    trace_data: Optional[Dict[str, Any]],
) -> int:
    """
    Store evidence bundle in database for audit/debug.
    
    Returns evidence_bundle_id.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evidence_bundles (
                result_set_id,
                bundle_type,
                plan_json,
                lane_runs_json,
                bundle_json,
                verification_status,
                verification_errors,
                answer_trace_json,
                rounds_executed,
                claims_count,
                entities_count,
                unresolved_count
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                result_set_id,
                "agentic_v1",
                Json(plan_data),
                Json(bundle_data.get("retrieval_runs", [])),
                Json(bundle_data),
                "passed" if verification_data and verification_data.get("passed") else "failed",
                Json(verification_data.get("errors", []) if verification_data else []),
                Json(trace_data),
                bundle_data.get("rounds_executed", 0),
                len(bundle_data.get("claims", [])),
                len(bundle_data.get("entities", [])),
                len(bundle_data.get("unresolved_tokens", [])),
            ),
        )
        bundle_id = cur.fetchone()[0]
        conn.commit()
        return bundle_id


if __name__ == "__main__":
    main()
