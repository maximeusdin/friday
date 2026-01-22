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

from retrieval.primitives import ResearchPlan, compile_primitives
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

def get_plan(conn, plan_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve plan by ID."""
    # Note: This queries research_plans by plan_id, not by session_id, so it's allowed
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, session_id, user_utterance, plan_json, plan_hash,
                   query_lang_version, retrieval_impl_version, status, parent_plan_id
            FROM research_plans
            WHERE id = %s
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
        }

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
        # Execute scope query to get chunk_ids
        # Need to JOIN chunk_metadata if scope references cm.*
        needs_metadata_join = "cm." in scope_where
        if needs_metadata_join:
            scope_sql = f"""
                SELECT DISTINCT c.id 
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE {scope_where}
            """
        else:
            scope_sql = f"SELECT DISTINCT c.id FROM chunks c WHERE {scope_where}"
        
        with conn.cursor() as cur:
            cur.execute(scope_sql, scope_params)
            scope_chunk_ids = {row[0] for row in cur.fetchall()}
    
    # Execute retrieval using existing functions with compiled query text
    from retrieval.ops import hybrid_rrf, vector_search
    
    if search_type == "vector":
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
    elif search_type == "lex":
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
    else:  # hybrid
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
        retrieval_config_json={
            "probes": probes,
            "top_n_vec": top_n_vec,
            "top_n_lex": top_n_lex,
            "rrf_k": rrf_k,
        } if search_type == "hybrid" else {},
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
) -> int:
    """Create a result_set from retrieval run."""
    if not name:
        name = f"Plan execution {run_id}"
    
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO result_sets (name, retrieval_run_id, chunk_ids)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (name, run_id, chunk_ids),
        )
        result_set_id = cur.fetchone()[0]
        conn.commit()
        return result_set_id

def update_plan_status(
    conn,
    plan_id: int,
    run_id: int,
    result_set_id: Optional[int] = None,
):
    """Update plan status to executed and link to run/result_set."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE research_plans
            SET status = 'executed',
                executed_at = now(),
                retrieval_run_id = %s,
                result_set_id = %s
            WHERE id = %s
            """,
            (run_id, result_set_id, plan_id),
        )
        conn.commit()

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Execute an approved research plan")
    ap.add_argument("--plan-id", type=int, required=True, help="ID of plan to execute")
    ap.add_argument("--create-result-set", action="store_true", help="Create a result_set from execution results")
    ap.add_argument("--result-set-name", type=str, default=None, help="Name for result_set (default: auto-generated)")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        # Get plan
        plan_data = get_plan(conn, args.plan_id)
        if not plan_data:
            print(f"ERROR: Plan {args.plan_id} not found", file=sys.stderr)
            sys.exit(1)
        
        # Verify status
        if plan_data["status"] != "approved":
            print(f"ERROR: Plan {args.plan_id} status is '{plan_data['status']}', must be 'approved'", file=sys.stderr)
            sys.exit(1)
        
        print(f"Executing Plan ID: {args.plan_id} (session={plan_data['session_id']})", file=sys.stderr)
        print(f"Utterance: {plan_data['user_utterance']}", file=sys.stderr)
        
        # Validate plan before execution
        validation_errors = validate_plan(plan_data["plan_json"])
        if validation_errors:
            print(f"ERROR: Plan {args.plan_id} validation failed:", file=sys.stderr)
            for err in validation_errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)
        
        # Parse plan
        plan = ResearchPlan.from_dict(plan_data["plan_json"])
        plan.query.raw = plan_data["plan_json"]["query"]["raw"]
        plan.compile()
        
        # Add metadata
        plan.query_lang_version = plan_data["query_lang_version"]
        plan.retrieval_impl_version = plan_data["retrieval_impl_version"]
        plan.session_id = plan_data["session_id"]
        
        # Execute retrieval (no session queries - all params from envelope)
        print("Executing retrieval...", file=sys.stderr)
        hits, run_id = execute_plan_retrieval(
            conn,
            plan,
        )
        
        print(f"Retrieval complete: {len(hits)} chunks, run_id={run_id}", file=sys.stderr)
        
        # Create result_set if requested
        result_set_id = None
        if args.create_result_set:
            chunk_ids = [h.chunk_id for h in hits]
            result_set_id = create_result_set(
                conn,
                run_id,
                chunk_ids,
                name=args.result_set_name,
            )
            print(f"Result set created: ID={result_set_id}", file=sys.stderr)
        
        # Update plan status
        update_plan_status(conn, args.plan_id, run_id, result_set_id)
        
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
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()
