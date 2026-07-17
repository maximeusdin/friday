"""
V10 Dispatch — Message dispatch for V10 identity-aware pipeline.

Follows the same pattern as V9 dispatch but uses the V10 runner
with scope-aware alias identity.

Routes:
- new_retrieval: full V10 pipeline (span lattice + structured boosts)
- think_deeper: resume with ThinkDeeper + V10 artifacts
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from retrieval.agent.v9_types import ScopeFilter
from retrieval.agent.v10_runner import V10Result, run_v10_query
from retrieval.agent.v10_workspace import (
    persist_chunk_mentions,
    persist_v10_run_state,
    rehydrate_v10_state,
)

logger = logging.getLogger(__name__)


@dataclass
class V10DispatchResult:
    """Result from V10 message dispatch."""
    intent: str = "new_retrieval"  # new_retrieval | think_deeper
    answer: str = ""
    cited_chunk_ids: List[int] = field(default_factory=list)
    confidence: str = "medium"
    run_id: Optional[int] = None
    evidence_set_id: Optional[int] = None
    run_status: str = "completed"
    can_think_deeper: bool = False
    suggestion: str = ""
    v10_result: Optional[V10Result] = None
    citation_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unresolved_aliases: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0


def dispatch_v10_message(
    conn,
    session_id: int,
    user_message: str,
    *,
    explicit_action: Optional[str] = None,
    scope: Optional[ScopeFilter] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
) -> V10DispatchResult:
    """Dispatch a message through the V10 pipeline.

    For now, supports:
    - new_retrieval: full V10 pipeline
    - think_deeper: placeholder for ThinkDeeper resume with V10 artifacts

    V9 endpoints remain unchanged — V10 is opt-in.
    """
    start_time = time.time()
    result = V10DispatchResult()

    try:
        if explicit_action == "think_deeper":
            # ThinkDeeper with V10 artifacts
            result = _run_think_deeper_v10(
                conn, session_id, user_message, scope, verbose, progress_callback
            )
        else:
            # New retrieval
            result = _run_new_retrieval_v10(
                conn, session_id, user_message, scope, verbose, progress_callback
            )

    except Exception as e:
        logger.error("V10 dispatch failed: %s", e, exc_info=True)
        result.answer = f"V10 pipeline error: {e}"
        result.run_status = "failed"

    result.elapsed_ms = (time.time() - start_time) * 1000
    return result


# =============================================================================
# New retrieval
# =============================================================================

def _run_new_retrieval_v10(
    conn,
    session_id: int,
    question: str,
    scope: Optional[ScopeFilter],
    verbose: bool,
    progress_callback: Optional[Callable],
) -> V10DispatchResult:
    """Run a new V10 retrieval."""
    result = V10DispatchResult(intent="new_retrieval")

    # Run the V10 pipeline
    v10_result = run_v10_query(
        conn,
        question,
        scope=scope,
        verbose=verbose,
        progress_callback=progress_callback,
    )

    # Create evidence set and run record
    run_id, evidence_set_id = _create_run_record(conn, session_id, question, v10_result)

    # Persist V10 artifacts
    if run_id and v10_result.lattice:
        persist_v10_run_state(
            conn, run_id,
            lattice=v10_result.lattice,
            lexicon=v10_result.lexicon,
            plan=v10_result.plan,
        )

    if evidence_set_id and v10_result.chunk_mentions:
        persist_chunk_mentions(conn, evidence_set_id, v10_result.chunk_mentions)

    # Build result — Think Deeper only available after 5+ tool calls
    _THINK_DEEPER_MIN_TOOL_CALLS = 5
    result.answer = v10_result.narrative
    result.v10_result = v10_result
    result.run_id = run_id
    result.evidence_set_id = evidence_set_id
    result.run_status = "completed"
    result.can_think_deeper = v10_result.tool_call_count >= _THINK_DEEPER_MIN_TOOL_CALLS
    if result.can_think_deeper:
        result.suggestion = "Think Deeper is now available — extend the investigation with additional searches."
    result.unresolved_aliases = v10_result.unresolved_aliases

    # Cited chunk IDs from claims
    cited = set()
    for claim in v10_result.claims:
        cited.update(claim.evidence_chunk_ids or [])
    result.cited_chunk_ids = sorted(cited)

    # Build citation map
    for chunk_id, wc in v10_result.chunks_fetched.items():
        result.citation_map[str(chunk_id)] = {
            "chunk_id": chunk_id,
            "document_id": wc.doc_id,
            "page": wc.page,
            "collection": wc.collection_slug,
        }

    return result


# =============================================================================
# ThinkDeeper with V10
# =============================================================================

def _run_think_deeper_v10(
    conn,
    session_id: int,
    question: str,
    scope: Optional[ScopeFilter],
    verbose: bool,
    progress_callback: Optional[Callable],
) -> V10DispatchResult:
    """Resume with ThinkDeeper using V10 artifacts.

    Rehydrates the V10 identity state (lattice, lexicon, mentions)
    from the prior run, then continues with the V10 runner.
    """
    result = V10DispatchResult(intent="think_deeper")

    # Find the latest V10 run for this session
    run_id, evidence_set_id = _find_latest_run(conn, session_id)
    if not run_id:
        result.answer = "No prior V10 run found for this session."
        result.run_status = "failed"
        return result

    # Rehydrate V10 state
    lattice, lexicon, chunk_mentions, plan = rehydrate_v10_state(
        conn, run_id, evidence_set_id
    )

    if verbose:
        print(f"[V10 ThinkDeeper] Rehydrated: lattice={'yes' if lattice else 'no'}, "
              f"lexicon={'yes' if lexicon else 'no'}, "
              f"mentions={len(chunk_mentions)}", flush=True)

    # Run V10 with resumed state
    v10_result = run_v10_query(
        conn,
        question,
        scope=scope,
        verbose=verbose,
        _resume_lexicon=lexicon,
        _resume_lattice=lattice,
        _resume_mentions=chunk_mentions,
        progress_callback=progress_callback,
    )

    # Update run record
    if run_id:
        persist_v10_run_state(
            conn, run_id,
            lattice=v10_result.lattice,
            lexicon=v10_result.lexicon,
            plan=v10_result.plan,
        )

    if evidence_set_id and v10_result.chunk_mentions:
        persist_chunk_mentions(conn, evidence_set_id, v10_result.chunk_mentions)

    # Think Deeper only available after 5+ tool calls (total from this run)
    _THINK_DEEPER_MIN_TOOL_CALLS = 5
    result.answer = v10_result.narrative
    result.v10_result = v10_result
    result.run_id = run_id
    result.evidence_set_id = evidence_set_id
    result.run_status = "completed"
    result.can_think_deeper = v10_result.tool_call_count >= _THINK_DEEPER_MIN_TOOL_CALLS
    if result.can_think_deeper:
        result.suggestion = "Think Deeper is now available — extend the investigation with additional searches."
    result.unresolved_aliases = v10_result.unresolved_aliases

    cited = set()
    for claim in v10_result.claims:
        cited.update(claim.evidence_chunk_ids or [])
    result.cited_chunk_ids = sorted(cited)

    return result


# =============================================================================
# DB helpers
# =============================================================================

def _create_run_record(
    conn,
    session_id: int,
    question: str,
    v10_result: V10Result,
) -> tuple:
    """Create v9_runs + evidence_sets records. Returns (run_id, evidence_set_id)."""
    run_id = None
    evidence_set_id = None

    try:
        with conn.cursor() as cur:
            # Create evidence set
            cur.execute("""
                INSERT INTO evidence_sets (session_id)
                VALUES (%s)
                RETURNING id
            """, (session_id,))
            evidence_set_id = cur.fetchone()[0]

            # Create run record
            cur.execute("""
                INSERT INTO v9_runs (session_id, question, evidence_set_id, status, pipeline_version)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (session_id, question, evidence_set_id, "completed", "v10"))
            run_id = cur.fetchone()[0]

            # Insert evidence items for fetched chunks
            for chunk_id, wc in v10_result.chunks_fetched.items():
                locators = {
                    "doc_id": wc.doc_id,
                    "page": wc.page,
                    "source_label": wc.source_label,
                    "collection_slug": wc.collection_slug,
                }
                import hashlib
                dedup_hash = hashlib.md5(
                    f"{evidence_set_id}:{chunk_id}".encode()
                ).hexdigest()

                cur.execute("""
                    INSERT INTO evidence_items
                        (evidence_set_id, chunk_id, quote_text, locators_json,
                         retrieval_score, rank, dedup_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (evidence_set_id, dedup_hash) DO NOTHING
                """, (
                    evidence_set_id,
                    chunk_id,
                    wc.text[:2000] if wc.text else "",
                    json.dumps(locators),
                    wc.score,
                    None,
                    dedup_hash,
                ))

        conn.commit()
    except Exception as e:
        logger.warning("Failed to create run record: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    return run_id, evidence_set_id


def _find_latest_run(conn, session_id: int) -> tuple:
    """Find the latest V10 run for a session. Returns (run_id, evidence_set_id)."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, evidence_set_id
                FROM v9_runs
                WHERE session_id = %s AND pipeline_version = 'v10'
                ORDER BY id DESC
                LIMIT 1
            """, (session_id,))
            row = cur.fetchone()
            if row:
                return row[0], row[1]
    except Exception as e:
        logger.warning("Failed to find latest V10 run: %s", e)

    return None, None
