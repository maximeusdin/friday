"""
V9 Dispatch — Main entry point for session-aware retrieval.

Routes messages to the appropriate execution path:
  NEW_RETRIEVAL → full v9_runner with evidence set population
  FOLLOW_UP    → evidence-only search (no tools)
  THINK_DEEPER → resume paused run from saved controller state

Handles step persistence, evidence set management, and pause/resume.
"""
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from retrieval.agent.v9_types import (
    ResearchWorkspace,
    V9Result,
    V9Synthesis,
    ScopeFilter,
    SufficiencyCheck,
    WorkspaceChunk,
)
from retrieval.config import (
    CONCORDANCE_EXPANSION_TARGET_COLLECTIONS,
    CONCORDANCE_EXPANSION_MAX_ENTITIES,
    CONCORDANCE_EXPANSION_MIN_ALIAS_LEN,
    CONCORDANCE_EXPANSION_MAX_ALIASES_PER_ENTITY,
    CONCORDANCE_EXPANSION_MAX_EXTRA_CHUNKS,
    CONCORDANCE_EXPANSION_SCORE_PENALTY,
)
from retrieval.agent.v9_session import (
    SessionState,
    RunRecord,
    RecentQueryContext,
    load_session,
    create_session,
    update_session_active,
    update_session_scope,
    session_scope_to_filter,
    run_scope_to_filter,
    normalize_scope,
    create_run,
    load_run,
    update_run_status,
    update_run_scope_json,
    load_recent_runs,
    add_evidence_items,
    add_adjacency_chunks,
    prune_evidence_set,
    get_evidence_set_size,
    get_evidence_set_document_count,
    search_evidence_set,
    persist_step,
    save_resume_state,
    build_resume_state,
    generate_evidence_summary,
    extract_top_entities,
    rehydrate_workspace_from_evidence,
)
from retrieval.agent.v9_router import (
    RouterDecision,
    route_message,
)
from retrieval.agent.v9_followup import (
    execute_followup,
    verify_followup_result,
)
from retrieval.agent.v9_runner import (
    run_v9_query,
    detect_scope,
    detect_scope_override_and_filters,
    ScopeDetectionResult,
    strip_scope_syntax,
    _execute_tool,
    _prime_workspace_from_question,
    _snapshot_counts,
    _compute_delta,
    _trim_messages,
    _call_with_retry,
    _parse_content,
    _build_artifact_dict,
    _build_minimal_synthesis_context,
    _build_needs_more_evidence_synthesis,
    _update_investigation,
    _validate_finalization,
    TOOLS_DEF,
    V9_OUTPUT_SCHEMA,
    V9_RESPONSE_FORMAT,
    MAX_HISTORY_MESSAGES,
    MAX_MODEL_TURNS,
    TOOL_TURN_MAX_TOKENS,
    SYNTHESIS_MAX_TOKENS,
)
from retrieval.agent.v9_context import build_context_pack, _estimate_tokens
from retrieval.agent.v9_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    V9_MODEL,
    V9_MAX_WORKSPACE_CHUNKS,
    V9_MAX_TOOL_CALLS,
)
from retrieval.agent.v9_workspace import (
    merge_search_result,
    merge_fetched_chunks,
    merge_catalog_hits,
    merge_entities,
    merge_entity_candidates,
    apply_pin_suggestions,
    merge_evidence_summary_update,
    build_chunk_doc_map,
    expand_query_with_aliases,
    link_chunks_to_entities,
    build_alias_context_for_summarizer,
)
from retrieval.agent.v9_grounding import ground_claims
from retrieval.agent.v9_verify import build_verification_report


# =============================================================================
# Scope metadata (follow-up context for UI)
# =============================================================================

@dataclass
class ScopeMeta:
    """Evidence set scope context shown to the user during follow-ups."""
    origin_query: str               # original query that created the evidence set
    origin_run_id: Optional[int]    # run that created it
    evidence_set_id: int
    chunk_count: int                # total evidence items
    document_count: int             # distinct documents
    top_entities: List[Dict]        # [{canonical_name, aliases}]
    time_range: Optional[str] = None  # e.g. "1939–1946" or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_query": self.origin_query,
            "origin_run_id": self.origin_run_id,
            "evidence_set_id": self.evidence_set_id,
            "chunk_count": self.chunk_count,
            "document_count": self.document_count,
            "top_entities": self.top_entities,
            "time_range": self.time_range,
        }


# =============================================================================
# Escalation option (offered when follow-up evidence is insufficient)
# =============================================================================

@dataclass
class EscalationOption:
    """A structured next-action offered to the user when follow-up confidence is low."""
    action: str          # "think_deeper" | "new_retrieval" | "show_evidence"
    label: str           # button text
    description: str     # one-line explanation
    prefilled_query: Optional[str] = None   # for new_retrieval: suggested query text
    carry_entities: List[Dict] = field(default_factory=list)  # entities + aliases to forward
    recommended: bool = False               # highlight as primary action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "label": self.label,
            "description": self.description,
            "prefilled_query": self.prefilled_query,
            "carry_entities": self.carry_entities,
            "recommended": self.recommended,
        }


# =============================================================================
# Dispatch result (unified output for all execution paths)
# =============================================================================

@dataclass
class DispatchResult:
    """Unified result from all execution paths."""
    intent: str                             # "new_retrieval" | "follow_up" | "think_deeper"
    answer: str = ""
    cited_chunk_ids: List[int] = field(default_factory=list)
    confidence: str = "medium"

    # Run metadata
    run_id: Optional[int] = None
    evidence_set_id: Optional[int] = None
    run_status: str = "completed"
    can_think_deeper: bool = False

    # V9Result for new_retrieval / think_deeper
    v9_result: Optional[V9Result] = None

    # Citation map: label -> {chunk_id, document_id, page} for frontend PDF viewer links
    citation_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Follow-up specific
    suggestion: str = ""
    scope_meta: Optional[ScopeMeta] = None
    escalations: List[EscalationOption] = field(default_factory=list)

    # Router decision
    router_decision: Optional[RouterDecision] = None

    # V12 clarification (pre-investigation follow-up questions)
    needs_clarification: bool = False
    clarification: Optional[Dict[str, Any]] = None

    # Think Deeper enrichment (only for think_deeper intent)
    novelty_report: Optional[Dict[str, Any]] = None
    stop_reason_detail: Optional[str] = None
    deep_dive_trace: Optional[List[Dict[str, Any]]] = None

    # Timing
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "intent": self.intent,
            "answer": self.answer,
            "cited_chunk_ids": self.cited_chunk_ids,
            "confidence": self.confidence,
            "run_id": self.run_id,
            "evidence_set_id": self.evidence_set_id,
            "run_status": self.run_status,
            "can_think_deeper": self.can_think_deeper,
            "citation_map": self.citation_map,
            "suggestion": self.suggestion,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.scope_meta:
            d["scope_meta"] = self.scope_meta.to_dict()
        if self.escalations:
            d["escalations"] = [e.to_dict() for e in self.escalations]
        if self.router_decision:
            d["routing"] = self.router_decision.to_dict()
        return d


# =============================================================================
# Constants (aligned with friday_cli.py)
# =============================================================================

def _max_tool_calls_from_env() -> int:
    """Same as CLI: V11_QUERY_MAX_TURNS (default 10)."""
    return int(os.getenv("V11_QUERY_MAX_TURNS", "10"))


def _think_deeper_budget_from_env() -> int:
    """Same as CLI: THINK_DEEPER_MAX_TOOL_CALLS (default 10)."""
    return int(os.getenv("THINK_DEEPER_MAX_TOOL_CALLS", "10"))


DEFAULT_MAX_TOOL_CALLS = V9_MAX_TOOL_CALLS       # 5 (fallback when env not set)
THINK_DEEPER_EXTRA_BUDGET = 10                    # fallback; prefer _think_deeper_budget_from_env()
THINK_DEEPER_MIN_TOOL_CALLS = 5                   # minimum tool calls before Think Deeper button appears


# =============================================================================
# Scope resolution
# =============================================================================

@dataclass
class RetrievalContext:
    """Search parameters from Stage 1, forwarded to Stage 1.5 for consistency."""
    query_text: str              # stripped user query
    top_k: int                   # same K used in Stage 1
    vector_threshold: float      # same threshold
    mode: str                    # "hybrid" | "fts" | "vector"


def _resolve_scope_for_run(
    conn,
    session: SessionState,
    user_message: str,
    *,
    explicit_scope: Optional[ScopeFilter] = None,
    verbose: bool = True,
) -> tuple:
    """Resolve the effective scope for a new retrieval run.

    Returns (resolved_scope_filter: ScopeFilter, run_scope_json: dict).

    Precedence:
    1. explicit_scope from API (if provided, use directly)
    2. query scope override (collection directives in the query text)
    3. session user-selected scope (stored in research_sessions.scope_json)

    Date filters from the query are ALWAYS merged, regardless of scope source.
    """
    # If caller passed an explicit scope, use it directly (API override)
    if explicit_scope and not explicit_scope.is_empty():
        run_scope_json = {
            "mode": "custom" if explicit_scope.collections or explicit_scope.document_ids else "full_archive",
            "source": "api_override",
            "expansion": {
                "policy": "venona_vassiliev_only",
                "collections": list(CONCORDANCE_EXPANSION_TARGET_COLLECTIONS),
                "triggered": False,
                "reason": None,
            },
        }
        return explicit_scope, run_scope_json

    # Detect scope override + filters from query text
    detection = detect_scope_override_and_filters(user_message)

    # Determine effective scope
    if detection.has_override:
        # Query scope override wins
        scope_filter = detection.scope_override or ScopeFilter()
        source = "query_override"
        reason = detection.reason

        # Build run scope json from override
        run_scope_json = {
            "mode": "custom" if scope_filter.collections else "full_archive",
            "included_collection_ids": [],  # slugs in ScopeFilter, ids here for audit
            "source": source,
            "reason": reason,
        }
        if scope_filter.collections:
            # Resolve slugs to IDs for audit trail
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM collections WHERE slug = ANY(%s)",
                    (scope_filter.collections,),
                )
                run_scope_json["included_collection_ids"] = [r[0] for r in cur.fetchall()]
    else:
        # Use session scope
        session_scope = session.scope_json
        scope_filter = session_scope_to_filter(conn, session_scope)
        source = "user_selected"
        reason = "using session-selected scope"

        run_scope_json = {
            **session_scope,
            "source": source,
            "reason": reason,
        }

    # Always merge date filters from query
    if detection.filter_overrides.date_from:
        scope_filter.date_from = detection.filter_overrides.date_from
    if detection.filter_overrides.date_to:
        scope_filter.date_to = detection.filter_overrides.date_to

    # Record filters in run_scope_json
    if detection.filter_overrides.date_from or detection.filter_overrides.date_to:
        run_scope_json["filters"] = {
            "date_from": detection.filter_overrides.date_from,
            "date_to": detection.filter_overrides.date_to,
        }

    # Set expansion policy (capability, not decision)
    run_scope_json["expansion"] = {
        "policy": "venona_vassiliev_only",
        "collections": list(CONCORDANCE_EXPANSION_TARGET_COLLECTIONS),
        "triggered": False,
        "reason": None,
    }

    if verbose:
        print(
            f"  [V9 Dispatch] Scope resolved: source={source}, "
            f"collections={scope_filter.collections}, "
            f"doc_ids={scope_filter.document_ids}, "
            f"dates={scope_filter.date_from}-{scope_filter.date_to}",
            file=sys.stderr,
        )

    return scope_filter, run_scope_json


# =============================================================================
# Stage 1.5: Targeted concordance expansion
# =============================================================================

def maybe_expand_from_target_collections(
    conn,
    initial_chunks: List[WorkspaceChunk],
    scope_filter: Optional[ScopeFilter],
    run_scope_json: Dict[str, Any],
    retrieval_ctx: RetrievalContext,
    *,
    run_id: Optional[int] = None,
    verbose: bool = True,
) -> List[WorkspaceChunk]:
    """Stage 1.5: Targeted concordance expansion for Venona/Vassiliev chunks.

    Fires after initial retrieval if Stage 1 returned chunks from target collections.
    Extracts entities from those chunks, looks up aliases in concordance tables,
    and runs a second search restricted to target collections.

    Scope rule (v1):
    - If user scope has document_ids: intersect with docs belonging to target collections.
      If empty intersection, skip.
    - If user scope has only collections: intersect with target collections.
      If empty intersection, skip.
    - If full_archive: search target collections unrestricted.
    - Date filters always applied.

    Returns extra chunks (callers should merge/de-dup).
    """
    from retrieval.agent.v9_tools import search_chunks as _search_chunks

    target_collections = CONCORDANCE_EXPANSION_TARGET_COLLECTIONS

    # 1. Filter Stage 1 chunks by collection_slug
    target_chunk_ids = [
        c.chunk_id for c in initial_chunks
        if c.collection_slug in target_collections
    ]

    triggered = len(target_chunk_ids) > 0

    # Structured log line
    log_entry = {
        "stage": "1.5_expansion",
        "run_id": run_id,
        "triggered": triggered,
        "target_chunk_count": len(target_chunk_ids),
        "entity_count": 0,
        "extra_chunks_added": 0,
    }

    if not triggered:
        run_scope_json["expansion"]["triggered"] = False
        run_scope_json["expansion"]["reason"] = "no target-collection chunks in initial results"
        print(json.dumps(log_entry), file=sys.stderr)
        return []

    # 2. Determine expansion scope (respect user selection)
    expansion_scope = _compute_expansion_scope(conn, scope_filter, target_collections)
    if expansion_scope is None:
        run_scope_json["expansion"]["triggered"] = False
        run_scope_json["expansion"]["reason"] = "user scope does not intersect target collections"
        print(json.dumps(log_entry), file=sys.stderr)
        return []

    # 3. Extract entities from target chunks via concordance tables
    entities = _extract_expansion_entities(conn, target_chunk_ids)
    log_entry["entity_count"] = len(entities)

    if not entities:
        run_scope_json["expansion"]["triggered"] = False
        run_scope_json["expansion"]["reason"] = "no entities found in target chunks"
        print(json.dumps(log_entry), file=sys.stderr)
        return []

    # 4. Build canonical entity query (boost pattern, not raw OR)
    canonical_names = [e["canonical_name"] for e in entities]
    expansion_query = " ".join(canonical_names[:8])  # top names by frequency

    if verbose:
        print(
            f"  [V9 Dispatch] Stage 1.5: {len(target_chunk_ids)} target chunks, "
            f"{len(entities)} entities. Expansion query: {expansion_query[:80]}",
            file=sys.stderr,
        )

    # 5. Run second search with expansion scope
    try:
        result, catalog = _search_chunks(
            conn,
            expansion_query,
            top_k=CONCORDANCE_EXPANSION_MAX_EXTRA_CHUNKS * 2,  # over-fetch, then cap
            scope=expansion_scope,
            mode=retrieval_ctx.mode,
        )
    except Exception as ex:
        if verbose:
            print(f"  [V9 Dispatch] Stage 1.5 search error: {ex}", file=sys.stderr)
        try:
            conn.rollback()
        except Exception:
            pass
        run_scope_json["expansion"]["triggered"] = True
        run_scope_json["expansion"]["reason"] = f"search failed: {ex}"
        log_entry["triggered"] = True
        print(json.dumps(log_entry), file=sys.stderr)
        return []

    if not result.chunk_ids:
        run_scope_json["expansion"]["triggered"] = True
        run_scope_json["expansion"]["reason"] = "expansion search returned no results"
        log_entry["triggered"] = True
        print(json.dumps(log_entry), file=sys.stderr)
        return []

    # 6. Fetch expansion chunks with score penalty
    from retrieval.agent.v9_tools import fetch_chunks_with_neighbors
    existing_ids = {c.chunk_id for c in initial_chunks}
    new_chunk_ids = [cid for cid in result.chunk_ids if cid not in existing_ids]
    new_chunk_ids = new_chunk_ids[:CONCORDANCE_EXPANSION_MAX_EXTRA_CHUNKS]

    if not new_chunk_ids:
        run_scope_json["expansion"]["triggered"] = True
        run_scope_json["expansion"]["reason"] = "all expansion results already in initial set"
        log_entry["triggered"] = True
        print(json.dumps(log_entry), file=sys.stderr)
        return []

    extra_chunks = fetch_chunks_with_neighbors(
        conn, chunk_ids=new_chunk_ids, include_neighbors=False,
    )

    # Apply score penalty to expansion-only hits
    for ec in extra_chunks:
        if ec.score is not None:
            ec.score = ec.score * CONCORDANCE_EXPANSION_SCORE_PENALTY

    # 7. Update expansion metadata
    run_scope_json["expansion"]["triggered"] = True
    run_scope_json["expansion"]["reason"] = (
        f"Stage 1 returned {len(target_chunk_ids)} target-collection chunks; "
        f"expanded with {len(entities)} entities, added {len(extra_chunks)} chunks"
    )

    log_entry["triggered"] = True
    log_entry["extra_chunks_added"] = len(extra_chunks)
    print(json.dumps(log_entry), file=sys.stderr)

    return extra_chunks


def _compute_expansion_scope(
    conn,
    scope_filter: Optional[ScopeFilter],
    target_collections: tuple,
) -> Optional[ScopeFilter]:
    """Compute the scope for Stage 1.5 expansion search.

    Returns None if user scope doesn't intersect target collections (skip expansion).
    """
    if not scope_filter or scope_filter.is_empty():
        # Full archive: search target collections unrestricted
        return ScopeFilter(collections=list(target_collections))

    # If document_ids present: intersect with docs belonging to target collections
    if scope_filter.document_ids:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.id FROM documents d
                JOIN collections c ON c.id = d.collection_id
                WHERE d.id = ANY(%s) AND c.slug = ANY(%s)
            """, (scope_filter.document_ids, list(target_collections)))
            intersected_doc_ids = [r[0] for r in cur.fetchall()]
        if not intersected_doc_ids:
            return None
        return ScopeFilter(
            document_ids=intersected_doc_ids,
            date_from=scope_filter.date_from,
            date_to=scope_filter.date_to,
        )

    # If collections present: intersect with target collections
    if scope_filter.collections:
        intersected = [c for c in scope_filter.collections if c in target_collections]
        if not intersected:
            return None
        return ScopeFilter(
            collections=intersected,
            date_from=scope_filter.date_from,
            date_to=scope_filter.date_to,
        )

    # No restrictions that need intersection; use target collections + date filters
    return ScopeFilter(
        collections=list(target_collections),
        date_from=scope_filter.date_from,
        date_to=scope_filter.date_to,
    )


def _extract_expansion_entities(
    conn,
    target_chunk_ids: List[int],
) -> List[Dict[str, Any]]:
    """Extract entities from target chunks with quality guardrails.

    Returns list of dicts with: id, canonical_name, aliases.
    Ordered by mention frequency, aliases capped per entity.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.canonical_name,
                       COUNT(em.id) AS mention_count,
                       (SELECT array_agg(sub.alias ORDER BY
                            (sub.alias = e.canonical_name) DESC,
                            length(sub.alias) ASC
                        )
                        FROM (
                            SELECT DISTINCT ea2.alias
                            FROM entity_aliases ea2
                            WHERE ea2.entity_id = e.id
                              AND length(ea2.alias) >= %(min_alias_len)s
                              AND ea2.alias ~ '^[A-Za-z]'
                            LIMIT %(max_aliases)s
                        ) sub) AS aliases
                FROM entity_mentions em
                JOIN entities e ON e.id = em.entity_id
                WHERE em.chunk_id = ANY(%(chunk_ids)s)
                GROUP BY e.id, e.canonical_name
                ORDER BY mention_count DESC
                LIMIT %(max_entities)s
            """, {
                "chunk_ids": target_chunk_ids,
                "min_alias_len": CONCORDANCE_EXPANSION_MIN_ALIAS_LEN,
                "max_aliases": CONCORDANCE_EXPANSION_MAX_ALIASES_PER_ENTITY,
                "max_entities": CONCORDANCE_EXPANSION_MAX_ENTITIES,
            })
            rows = cur.fetchall()
    except Exception as e:
        logging.getLogger(__name__).warning("_extract_expansion_entities query failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return []

    entities = []
    for row in rows:
        aliases = row[3] or []
        entities.append({
            "id": row[0],
            "canonical_name": row[1],
            "mention_count": row[2],
            "aliases": aliases,
        })
    return entities


# =============================================================================
# New retrieval path (with evidence persistence)
# =============================================================================

def _run_new_retrieval(
    conn,
    session_id: int,
    question: str,
    *,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    scope: Optional[ScopeFilter] = None,
    run_scope_json: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
    carry_context: Optional[Dict[str, Any]] = None,
) -> DispatchResult:
    """Execute a new retrieval run with evidence set population.

    Wraps run_v9_query and persists:
    - Run record + steps (via run_steps table)
    - Evidence items (chunks → evidence_items)
    - Adjacency expansion
    - Resume state for think_deeper
    - Evidence summary + top entities for query reference
    - Stage 1.5 concordance expansion (if Venona/Vassiliev chunks found)
    """
    t0 = time.time()
    if progress_callback:
        progress_callback("retrieval_prepare", "running", "Preparing search...", {})

    # Create run + evidence set (with scope metadata)
    run = create_run(conn, session_id, question, budgets={
        "max_tool_calls": max_tool_calls,
    }, run_scope_json=run_scope_json)

    use_v9_fallback = os.getenv("USE_V9_AGENT", "0").strip().lower() in ("1", "true", "yes")
    _v9_env_raw = os.getenv("USE_V9_AGENT", "<unset>")
    if verbose:
        print(
            f"  [{'V9' if use_v9_fallback else 'V11'} Dispatch] USE_V9_AGENT={_v9_env_raw} -> use_v9_fallback={use_v9_fallback}",
            file=sys.stderr,
        )
        print(
            f"  [{'V9' if use_v9_fallback else 'V11'} Dispatch] NEW_RETRIEVAL: run_id={run.run_id}, "
            f"evidence_set_id={run.evidence_set_id}, "
            f"max_tool_calls={max_tool_calls}",
            file=sys.stderr,
        )

    # Run the v9 or v11 query
    _profile = "v11"  # engine profile; the default branch may promote to "v13"
    try:
        search_result_set_id = (carry_context or {}).get("search_result_set_id")

        # --- V12 clarification round-trip (optional, in front of V11) ---
        use_v12 = os.getenv("USE_V12_AGENT", "0").strip().lower() in ("1", "true", "yes")
        if use_v12 and not use_v9_fallback:
            from retrieval.agent.v12_runner import run_v12_query, V12ClarificationPending
            from retrieval.agent.v12_clarifier import ClarificationPlan, ClarificationAnswer
            from retrieval.agent.v11_runner import run_v11_query
            cc = carry_context or {}
            try:
                answers = ([ClarificationAnswer.from_dict(a) for a in cc["clarification_answers"]]
                           if cc.get("clarification_answers") is not None else None)
                plan = ClarificationPlan.from_dict(cc["clarification_plan"]) if cc.get("clarification_plan") else None
                v12 = run_v12_query(
                    conn, question,
                    clarification_answers=answers, clarification_plan=plan,
                    use_llm=os.getenv("V12_DISABLE_LLM", "0").strip().lower() not in ("1", "true", "yes"),
                    max_tool_calls=max_tool_calls, scope=scope, verbose=verbose,
                    progress_callback=progress_callback,
                    use_lightweight_pem=os.getenv("V11_USE_LIGHTWEIGHT_PEM", "0").strip().lower() in ("1", "true", "yes"),
                )
            except Exception as _v12e:
                # Fail open: a clarifier error must never break chat -> recover the
                # transaction and fall back to a normal V11 investigation.
                print(f"  [V12] error, falling back to V11: {_v12e}", file=sys.stderr)
                try:
                    conn.rollback()
                except Exception:
                    pass
                v12 = run_v11_query(
                    conn, question, max_tool_calls=max_tool_calls, scope=scope, verbose=verbose,
                    progress_callback=progress_callback,
                    use_lightweight_pem=os.getenv("V11_USE_LIGHTWEIGHT_PEM", "0").strip().lower() in ("1", "true", "yes"),
                )
            if isinstance(v12, V12ClarificationPending):
                update_run_status(conn, run.run_id, "paused")  # 'paused' is a valid v9_runs.status
                return DispatchResult(
                    intent="clarify", needs_clarification=True, clarification=v12.to_dict(),
                    run_id=run.run_id, elapsed_ms=(time.time() - t0) * 1000.0,
                )
            result = v12
        elif use_v9_fallback:
            result = run_v9_query(
                conn, question,
                max_tool_calls=max_tool_calls,
                scope=scope,
                verbose=verbose,
                progress_callback=progress_callback,
                session_id=session_id,
                search_result_set_id=search_result_set_id,
            )
        else:
            from retrieval.agent.v11_runner import run_v11_query
            # Engine selection: default v13 (query planning + priming + anti-false-negative).
            # Fall back to the unchanged v11 engine with FRIDAY_CHAT_ENGINE=v11 (or V13_DISABLE=1).
            _engine = os.getenv("FRIDAY_CHAT_ENGINE", "v13").strip().lower()
            if os.getenv("V13_DISABLE", "0").strip().lower() in ("1", "true", "yes"):
                _engine = "v11"
            _profile = "v13" if _engine == "v13" else "v11"
            if verbose:
                print(f"  [Dispatch] chat engine profile = {_profile}", file=sys.stderr)
            result = run_v11_query(
                conn, question,
                max_tool_calls=max_tool_calls,
                scope=scope,
                verbose=verbose,
                progress_callback=progress_callback,
                use_lightweight_pem=os.getenv("V11_USE_LIGHTWEIGHT_PEM", "0").strip().lower() in ("1", "true", "yes"),
                engine_profile=_profile,
            )
    except Exception as e:
        update_run_status(conn, run.run_id, "failed")
        update_session_active(conn, session_id, active_run_status="failed")
        raise

    # Think Deeper is only via the button — no auto-trigger (Tier 1 removed).

    # Stage 1.5 removed: auto concordance expansion was V/V-biased.
    # Use expand_query / expand_from_evidence agent tools instead.

    # Persist evidence items from workspace
    if run.evidence_set_id and result.workspace:
        ws = result.workspace
        step_idx = len(ws.investigation.trace)

        # Add fulltext chunks as evidence items
        added = add_evidence_items(
            conn, run.evidence_set_id, ws.fulltext_chunks,
            step_idx=step_idx,
            scores={c.chunk_id: c.score for c in ws.catalog_hits if c.score},
        )

        # Adjacency expansion
        primary_cids = [c.chunk_id for c in ws.fulltext_chunks if not c.is_neighbor]
        adj_added = add_adjacency_chunks(
            conn, run.evidence_set_id, primary_cids, step_idx=step_idx,
        )

        # Prune to cap
        pruned = prune_evidence_set(conn, run.evidence_set_id)
        ev_size = get_evidence_set_size(conn, run.evidence_set_id)

        if verbose:
            print(
                f"  [V9 Dispatch] Evidence: added={added}, adjacency={adj_added}, "
                f"pruned={pruned}, total={ev_size}",
                file=sys.stderr,
            )

        # Persist steps
        for i, step in enumerate(ws.investigation.trace):
            persist_step(
                conn, run.run_id, step.step_idx,
                tool_name=step.action,
                tool_args=step.inputs or {},
                lane=step.action.split("_")[0] if step.action else None,
                result_refs={"summary": step.outputs_summary},
                elapsed_ms=None,
            )

        # Build evidence summary + top entities
        ev_summary = generate_evidence_summary(ws)
        top_ents = extract_top_entities(ws)

        # Save resume state
        resume_state = build_resume_state(
            ws,
            tool_calls_executed=len(ws.investigation.trace),
            model_turns=0,  # not tracked in current runner
            step_idx=step_idx,
            max_tool_calls=max_tool_calls,
        )

        # Determine run status
        suf = result.sufficiency
        if suf and suf.sufficient:
            run_status = "completed"
        else:
            run_status = "paused"

        # Update run record
        update_run_status(
            conn, run.run_id, run_status,
            last_step_idx=step_idx,
            resume_state_json=resume_state,
            evidence_summary=ev_summary,
            top_entities_json=top_ents,
            label=_auto_label(question),
        )

        # Update session
        update_session_active(
            conn, session_id,
            active_run_id=run.run_id,
            active_evidence_set_id=run.evidence_set_id,
            active_run_status=run_status,
        )

    elapsed = (time.time() - t0) * 1000
    # Think Deeper button always available after any retrieval (user-initiated only)
    can_deeper = bool(run.evidence_set_id)
    suggestion = ""
    if can_deeper:
        suggestion = "Think Deeper is now available — extend the investigation with additional searches."

    # Build citation detail map for frontend PDF viewer linking
    cit_map = result.build_citation_detail_map() if result else {}

    answer_text = result.format_answer() if result else ""
    # V13: scrub cosmetic "(unresolved codename)"/"[AMBIGUOUS]" tags that format_answer's
    # ambiguity gate renders (often from duplicate entity rows), which the narrative-level
    # guard can't reach because format_answer regenerates the text from claims + alias map.
    if _profile == "v13" and answer_text:
        try:
            from retrieval.agent.v13_planner import _scrub_codename_noise
            answer_text = _scrub_codename_noise(answer_text)
        except Exception:
            pass

    return DispatchResult(
        intent="new_retrieval",
        answer=answer_text,
        cited_chunk_ids=_extract_cited_chunk_ids(result),
        confidence="high" if (result.sufficiency and result.sufficiency.sufficient) else "medium",
        run_id=run.run_id,
        evidence_set_id=run.evidence_set_id,
        run_status=run_status if run.evidence_set_id else "completed",
        can_think_deeper=can_deeper,
        suggestion=suggestion,
        v9_result=result,
        citation_map=cit_map,
        elapsed_ms=elapsed,
    )


def _build_scope_meta(
    conn,
    evidence_set_id: int,
    origin_run: Optional[RunRecord] = None,
) -> Optional[ScopeMeta]:
    """Build scope metadata for an evidence set (used in follow-up answers)."""
    try:
        chunk_count = get_evidence_set_size(conn, evidence_set_id)
        doc_count = get_evidence_set_document_count(conn, evidence_set_id)

        origin_query = ""
        origin_run_id = None
        top_entities: List[Dict] = []

        if origin_run:
            origin_query = origin_run.query_text or ""
            origin_run_id = origin_run.run_id
            top_entities = origin_run.top_entities_json or []

        return ScopeMeta(
            origin_query=origin_query,
            origin_run_id=origin_run_id,
            evidence_set_id=evidence_set_id,
            chunk_count=chunk_count,
            document_count=doc_count,
            top_entities=top_entities,
            time_range=None,  # deferred to future iteration
        )
    except Exception:
        return None  # scope_meta is best-effort


def _build_escalation_options(
    followup_result: Dict[str, Any],
    user_message: str,
    original_query: Optional[str],
    top_entities: List[Dict],
    evidence_set_id: int,
) -> List[EscalationOption]:
    """Build structured escalation options when follow-up confidence is low/insufficient."""
    confidence = followup_result.get("confidence", "medium")
    suggestion = followup_result.get("suggestion", "")

    if confidence in ("high", "medium"):
        return []

    options: List[EscalationOption] = []

    # Build a human-readable entity hint for the search-wider description
    entity_names = [e.get("canonical_name", "") for e in top_entities[:3] if e.get("canonical_name")]
    entity_hint = ", ".join(entity_names) if entity_names else "related entities"

    # Option 1: Search wider (think_deeper) — recommended by default
    options.append(EscalationOption(
        action="think_deeper",
        label="Search wider",
        description=f"Extend the previous search around {entity_hint} in the archive",
        prefilled_query=user_message,
        carry_entities=top_entities,
        recommended=(suggestion == "think_deeper" or confidence == "low"),
    ))

    # Option 2: Start a new search (new_retrieval)
    prefilled = user_message
    if original_query and user_message.lower() != original_query.lower():
        # Combine the intent: use the follow-up question as a fresh query
        prefilled = user_message
    options.append(EscalationOption(
        action="new_retrieval",
        label="Start a new search",
        description=f"Search the full archive for: \"{prefilled[:80]}\"",
        prefilled_query=prefilled,
        carry_entities=top_entities,
        recommended=(suggestion == "new_retrieval"),
    ))

    # Option 3: Show what we have (show_evidence)
    items_searched = followup_result.get("evidence_items_searched", 0)
    options.append(EscalationOption(
        action="show_evidence",
        label="Show what we have",
        description=f"Highlight the {items_searched} most relevant chunks found in this evidence set",
        carry_entities=[],
        recommended=False,
    ))

    return options


def _run_follow_up(
    conn,
    session_id: int,
    user_message: str,
    evidence_set_id: int,
    *,
    original_query: Optional[str] = None,
    origin_run: Optional[RunRecord] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
) -> DispatchResult:
    """Execute a follow-up query against an existing evidence set.

    No retrieval tools called. Uses FTS join to chunks.text.
    """
    t0 = time.time()

    if verbose:
        print(
            f"  [V9 Dispatch] FOLLOW_UP: evidence_set_id={evidence_set_id}",
            file=sys.stderr,
        )

    if progress_callback:
        progress_callback("follow_up", "running", "Searching existing evidence set...", {
            "evidence_set_id": evidence_set_id,
        })

    result = execute_followup(
        conn, user_message, evidence_set_id,
        original_query=original_query,
        verbose=verbose,
    )

    # Verify invariants
    violations = verify_followup_result(result, evidence_set_id, conn)
    if violations and verbose:
        print(f"  [V9 Dispatch] Follow-up violations: {violations}", file=sys.stderr)

    elapsed = (time.time() - t0) * 1000

    # Build citation map from chunk IDs (follow_up has no V9Result workspace)
    follow_cited = result.get("cited_chunk_ids", [])
    cit_map = _build_citation_map_from_chunks(conn, follow_cited, evidence_set_id=evidence_set_id)

    # Post-process answer: replace "Chunk X" with document labels [DocName pN]
    answer_text = result.get("answer", "")
    chunk_id_to_label = result.get("chunk_id_to_label") or {}
    for cid in follow_cited:
        label = chunk_id_to_label.get(cid)
        if label:
            # Replace "Chunk 39163" or "chunk 39163" with [Vassiliev p42]
            answer_text = re.sub(
                rf"\b[Cc]hunk\s+{cid}\b", f"[{label}]", answer_text,
            )

    # Build scope metadata for the UI
    scope_meta = _build_scope_meta(conn, evidence_set_id, origin_run=origin_run)

    # Build escalation options if confidence is low/insufficient
    top_entities = (origin_run.top_entities_json or []) if origin_run else []
    escalations = _build_escalation_options(
        result, user_message, original_query, top_entities, evidence_set_id,
    )

    return DispatchResult(
        intent="follow_up",
        answer=answer_text,
        cited_chunk_ids=follow_cited,
        confidence=result.get("confidence", "medium"),
        evidence_set_id=evidence_set_id,
        run_status="completed",
        can_think_deeper=False,
        citation_map=cit_map,
        suggestion=result.get("suggestion", ""),
        scope_meta=scope_meta,
        escalations=escalations,
        elapsed_ms=elapsed,
    )


def _run_think_deeper(
    conn,
    session_id: int,
    user_message: str,
    target_run_id: int,
    *,
    extra_budget: Optional[int] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
) -> DispatchResult:
    """Resume a paused run with the autonomous Actor/Judge Think Deeper controller.

    1. Loads the target run + its resume state
    2. Rehydrates workspace from saved evidence set + entity state
    3. Runs the Think Deeper controller loop (Actor/Judge architecture)
    4. Persists new evidence items back to the evidence set
    5. run_id stays the same (invariant: THINK_DEEPER continues the same run)
    """
    from retrieval.agent.v9_deep_runner import think_deeper

    t0 = time.time()

    # Load the target run
    run = load_run(conn, target_run_id)
    if not run:
        return DispatchResult(
            intent="think_deeper",
            answer="Could not find the run to resume.",
            confidence="insufficient",
            run_status="failed",
            elapsed_ms=0,
        )

    resume = run.resume_state_json or {}
    prev_tool_calls = resume.get("tool_calls_executed", 0)
    prev_step_idx = run.last_step_idx

    if verbose:
        print(
            f"  [V9 Dispatch] THINK_DEEPER (Actor/Judge): run_id={run.run_id}, "
            f"last_step={prev_step_idx}, "
            f"prev_tool_calls={prev_tool_calls}, "
            f"prev_status={run.status}",
            file=sys.stderr,
        )

    # Calculate new budget: extend from where we left off (same as CLI: THINK_DEEPER_MAX_TOOL_CALLS)
    new_max = _think_deeper_budget_from_env() if extra_budget is None else extra_budget

    # Update run status to running
    update_run_status(conn, run.run_id, "running")
    update_session_active(conn, session_id, active_run_status="running")

    # Resolve scope from run (Think Deeper must stay within original run's scope)
    scope = run_scope_to_filter(conn, run.run_scope_json) if run.run_scope_json else None
    if scope and scope.is_empty():
        scope = None
    if verbose and scope:
        coll_str = ",".join(scope.collections[:5]) if scope.collections else "none"
        doc_cnt = len(scope.document_ids) if scope.document_ids else 0
        print(
            f"  [V9 Dispatch] Think Deeper scope: collections=[{coll_str}], doc_ids_count={doc_cnt}",
            file=sys.stderr,
        )

    # Rehydrate workspace from evidence set + resume state.
    # Always rehydrate when evidence_set_id exists — evidence_items persist baseline chunks
    # regardless of resume_state_json. Empty resume is fine (rehydrate uses .get() defaults).
    workspace = None
    if run.evidence_set_id:
        try:
            workspace = rehydrate_workspace_from_evidence(
                conn, run.evidence_set_id, run.query_text,
                resume or {}, scope=scope,
            )
            if verbose:
                print(
                    f"  [V9 Dispatch] Rehydrated workspace: "
                    f"{len(workspace.fulltext_chunks)} chunks, "
                    f"{len(workspace.entities)} entities, "
                    f"goal='{workspace.investigation.goal[:50]}'",
                    file=sys.stderr,
                )
        except Exception as e:
            if verbose:
                print(f"  [V9 Dispatch] Rehydration failed: {e}; running fresh", file=sys.stderr)
            try:
                conn.rollback()  # Clear aborted transaction so think_deeper can use conn
            except Exception:
                pass
            workspace = None

    # Determine user follow-up directive (if user sent a message beyond just "think deeper")
    user_followup = None
    if user_message and user_message.strip().lower() not in ("think deeper", "think_deeper", ""):
        user_followup = user_message

    # Run the Think Deeper controller loop (scope enforced in execute_action)
    try:
        td_result = think_deeper(
            conn,
            seed_question=run.query_text,
            workspace=workspace,
            user_followup=user_followup,
            max_steps=8,
            max_tool_calls=new_max,
            verbose=verbose,
            progress_callback=progress_callback,
            v9_run_id=run.run_id,
            run_scope=scope,
        )
    except Exception as e:
        update_run_status(conn, run.run_id, "failed")
        update_session_active(conn, session_id, active_run_status="failed")
        raise

    # Persist new evidence items from Think Deeper results
    run_status = "completed"
    can_think_deeper_after = False
    added = 0
    if run.evidence_set_id and td_result.selected_chunks:
        new_step_base = prev_step_idx + 1

        # Convert CandidateChunks to WorkspaceChunks for evidence persistence
        new_ws_chunks = []
        for cc in td_result.selected_chunks:
            if cc.chunk_id not in (set()):
                new_ws_chunks.append(WorkspaceChunk(
                    chunk_id=cc.chunk_id,
                    doc_id=cc.doc_id,
                    page=cc.page,
                    text=cc.text,
                    source_label=cc.collection_slug,
                    score=cc.score,
                ))

        if new_ws_chunks:
            added = add_evidence_items(
                conn, run.evidence_set_id, new_ws_chunks,
                step_idx=new_step_base,
                scores={c.chunk_id: c.score for c in new_ws_chunks if c.score},
            )
            pruned = prune_evidence_set(conn, run.evidence_set_id)

        # Persist Think Deeper steps as run steps
        for i, verdict in enumerate(td_result.verdict_history):
            action = td_result.verdict_history[i] if i < len(td_result.verdict_history) else None
            persist_step(
                conn, run.run_id, new_step_base + i,
                tool_name="think_deeper_step",
                tool_args={"step": i, "verdict": verdict.to_dict()},
                lane="think_deeper",
                result_refs={"stop_reason": td_result.stop_reason},
            )

        # Update resume state
        new_step_idx = new_step_base + td_result.steps_executed - 1
        total_tool_calls = prev_tool_calls + td_result.tool_calls_used

        # Determine if we should mark as paused (can_think_deeper again)
        final_verdict = td_result.verdict_history[-1] if td_result.verdict_history else None
        is_saturated = (
            final_verdict and final_verdict.stop_recommendation
            and final_verdict.confidence > 0.8
        )
        run_status = "completed" if is_saturated else "paused"
        can_think_deeper_after = (
            run_status == "paused"
            and run.evidence_set_id
            and total_tool_calls >= THINK_DEEPER_MIN_TOOL_CALLS
        )

        update_run_status(
            conn, run.run_id, run_status,
            last_step_idx=new_step_idx,
            resume_state_json={
                "tool_calls_executed": total_tool_calls,
                "step_idx": new_step_idx,
                "think_deeper_stop_reason": td_result.stop_reason,
            },
        )
        update_session_active(
            conn, session_id,
            active_run_status=run_status,
        )

        # Verify think_deeper invariants
        violations = _verify_think_deeper(run, target_run_id, new_step_idx, prev_step_idx)
        if violations and verbose:
            print(f"  [V9 Dispatch] Think deeper violations: {violations}", file=sys.stderr)

        if verbose:
            print(
                f"  [V9 Dispatch] Think deeper (Actor/Judge): "
                f"new evidence={added}, steps={td_result.steps_executed}, "
                f"total_tool_calls={total_tool_calls}, status={run_status}, "
                f"stop_reason={td_result.stop_reason}",
                file=sys.stderr,
            )

    elapsed = (time.time() - t0) * 1000

    # Re-synthesize with merged workspace to produce grounded summary + evidence bullets
    answer = td_result.narrative if td_result.narrative else td_result.stop_reason
    cited_cids = sorted({c.chunk_id for c in td_result.selected_chunks})
    v9_result_for_dispatch = None

    if td_result.selected_chunks and run.evidence_set_id:
        try:
            # Build or merge workspace for re-synthesis
            resynth_workspace = workspace
            if resynth_workspace is None:
                from retrieval.agent.v11_types import V11ResearchWorkspace
                from retrieval.agent.v9_types import CatalogHit
                resynth_workspace = V11ResearchWorkspace(
                    question=run.query_text,
                    scope=scope or ScopeFilter(),
                )
                for cc in td_result.selected_chunks:
                    resynth_workspace.fulltext_chunks.append(WorkspaceChunk(
                        chunk_id=cc.chunk_id,
                        text=cc.text,
                        doc_id=cc.doc_id,
                        page=cc.page,
                        source_label=cc.collection_slug or "",
                        collection_slug=cc.collection_slug or "",
                        score=cc.score,
                    ))
                    resynth_workspace.catalog_hits.append(CatalogHit(
                        chunk_id=cc.chunk_id,
                        score=cc.score or 0.0,
                        doc_id=cc.doc_id,
                        page=cc.page,
                        collection=cc.collection_slug or "",
                        snippet=(cc.text or "")[:300],
                    ))
            else:
                existing_ids = {c.chunk_id for c in resynth_workspace.fulltext_chunks}
                for cc in td_result.selected_chunks:
                    if cc.chunk_id not in existing_ids:
                        resynth_workspace.fulltext_chunks.append(WorkspaceChunk(
                            chunk_id=cc.chunk_id,
                            text=cc.text,
                            doc_id=cc.doc_id,
                            page=cc.page,
                            source_label=cc.collection_slug or "",
                            collection_slug=cc.collection_slug or "",
                            score=cc.score,
                        ))
                        existing_ids.add(cc.chunk_id)

            # Emit evidence bullets from Think Deeper chunks (resynth may not run fetch)
            if progress_callback:
                try:
                    from retrieval.agent.v9_summarize import summarize_delta_chunks
                    chunks_to_summarize = [
                        WorkspaceChunk(
                            chunk_id=c.chunk_id,
                            text=c.text,
                            doc_id=c.doc_id,
                            page=c.page,
                            source_label=c.collection_slug or "",
                            collection_slug=c.collection_slug or "",
                            score=c.score,
                        )
                        for c in td_result.selected_chunks
                    ]
                    if chunks_to_summarize:
                        ev_update = summarize_delta_chunks(
                            chunks_to_summarize[:30],  # cap for summarizer
                            run.query_text,
                        )
                        if ev_update.bullets:
                            chunk_doc_map = {c.chunk_id: c.doc_id for c in td_result.selected_chunks if c.doc_id}
                            chunk_to_page = {c.chunk_id: c.page for c in td_result.selected_chunks if c.page}
                            all_doc_ids = []
                            for b in ev_update.bullets:
                                for cid in (b.supporting_chunk_ids or []):
                                    if cid in chunk_doc_map:
                                        all_doc_ids.append(chunk_doc_map[cid])
                            doc_names = {}
                            if all_doc_ids:
                                with conn.cursor() as cur:
                                    cur.execute(
                                        "SELECT id, source_name FROM documents WHERE id = ANY(%s)",
                                        (list(set(all_doc_ids)),),
                                    )
                                    doc_names = {r[0]: (r[1] or "").strip() for r in cur.fetchall()}

                            def _parse_page(p):
                                if not p:
                                    return None
                                s = str(p).strip().lstrip("pP")
                                try:
                                    return int(s) if s else None
                                except ValueError:
                                    return None

                            bullet_payloads = []
                            for b in ev_update.bullets:
                                cids = b.supporting_chunk_ids or []
                                b_doc_ids = sorted(set(chunk_doc_map[cid] for cid in cids if cid in chunk_doc_map))
                                bullet_payloads.append({
                                    "text": b.text,
                                    "tags": b.tags,
                                    "chunk_ids": cids,
                                    "doc_ids": b_doc_ids,
                                    "pages": [_parse_page(chunk_to_page.get(cid)) for cid in cids],
                                    "source_names": [doc_names.get(did, "") for did in b_doc_ids],
                                })
                            progress_callback("evidence_update", "completed",
                                f"Discovered {len(ev_update.bullets)} evidence bullets from Think Deeper",
                                {
                                    "bullets": bullet_payloads,
                                    "open_questions": ev_update.open_questions,
                                    "leads": ev_update.leads,
                                    "total_bullet_count": len(ev_update.bullets),
                                })
                except Exception as emit_err:
                    if verbose:
                        print(f"  [V9 Dispatch] Think Deeper evidence emit failed: {emit_err}", file=sys.stderr)

            # Build findings brief for synthesis context
            from retrieval.agent.v9_deep_findings import build_findings_brief
            chunk_id_to_label = {
                c.chunk_id: f"{c.collection_slug or ''} {c.page or ''}".strip()
                for c in td_result.selected_chunks
            }
            findings_brief = build_findings_brief(
                td_result.finding_store_entries, top_n=10,
                chunk_id_to_label=chunk_id_to_label,
            ) if td_result.finding_store_entries else None

            use_v9_fallback = os.getenv("USE_V9_AGENT", "0").strip().lower() in ("1", "true", "yes")
            if use_v9_fallback:
                resynth = run_v9_query(
                    conn, run.query_text,
                    verbose=False,
                    _resume_workspace=resynth_workspace,
                    max_tool_calls=3,
                    _findings_brief=findings_brief,
                    progress_callback=progress_callback,
                    session_id=session_id,
                )
            else:
                from retrieval.agent.v11_runner import run_v11_query
                resynth = run_v11_query(
                    conn, run.query_text,
                    verbose=False,
                    _resume_workspace=resynth_workspace,
                    max_tool_calls=3,
                    use_lightweight_pem=os.getenv("V11_USE_LIGHTWEIGHT_PEM", "0").strip().lower() in ("1", "true", "yes"),
                    _findings_brief=findings_brief,
                    progress_callback=progress_callback,
                )

            from retrieval.agent.v9_deep_tiering import _count_grounded_claims
            resynth_grounded = _count_grounded_claims(resynth)
            if resynth_grounded > 0:
                answer = resynth.format_answer()
                v9_result_for_dispatch = resynth
                if verbose:
                    print(
                        f"  [V9 Dispatch] Think Deeper re-synthesis: {resynth_grounded} grounded claims",
                        file=sys.stderr,
                    )
        except Exception as resynth_err:
            if verbose:
                print(f"  [V9 Dispatch] Think Deeper re-synthesis failed (non-fatal): {resynth_err}", file=sys.stderr)

    # Build citation map for document links in the narrative
    cit_map = _build_citation_map_from_chunks(
        conn, cited_cids, evidence_set_id=run.evidence_set_id
    ) if cited_cids else {}

    # Determine confidence from final verdict
    confidence = "medium"
    if td_result.verdict_history:
        final = td_result.verdict_history[-1]
        if final.confidence > 0.8:
            confidence = "high"
        elif final.confidence < 0.3:
            confidence = "low"

    # Build Think Deeper enrichment for API response
    novelty_report_dict = td_result.novelty_report.to_dict() if td_result.novelty_report else None
    deep_trace = None
    if td_result.verdict_history:
        deep_trace = []
        for i, v in enumerate(td_result.verdict_history):
            step_action = (td_result.verdict_history[i].selection_reasoning[:80]
                           if hasattr(v, "selection_reasoning") else "")
            deep_trace.append({
                "step": i,
                "answeredness": v.answeredness,
                "material_novelty": v.material_novelty,
                "confidence": v.confidence,
                "new_findings": len(v.new_findings),
                "stop_recommendation": v.stop_recommendation,
            })

    suggestion = ""
    if can_think_deeper_after:
        suggestion = "Think Deeper is now available — extend the investigation with additional searches."

    return DispatchResult(
        intent="think_deeper",
        answer=answer or "",
        cited_chunk_ids=cited_cids,
        confidence=confidence,
        run_id=run.run_id,
        evidence_set_id=run.evidence_set_id,
        run_status=run_status if run.evidence_set_id else "completed",
        can_think_deeper=can_think_deeper_after,
        suggestion=suggestion,
        citation_map=cit_map,
        novelty_report=novelty_report_dict,
        stop_reason_detail=td_result.stop_reason,
        deep_dive_trace=deep_trace,
        elapsed_ms=elapsed,
        v9_result=v9_result_for_dispatch,
    )


# =============================================================================
# Helpers
# =============================================================================

def _build_citation_map_from_chunks(
    conn, cited_chunk_ids: List[int], *, evidence_set_id: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """Build citation label -> {chunk_id, document_id, page} map from chunk IDs.

    Used for execution paths that don't have a V9Result workspace (e.g. follow_up).
    When evidence_set_id is provided, prefers locators from evidence_items (doc_id, page)
    for accurate clickable citations. Falls back to chunk_metadata for any missing chunks.
    """
    import re as _re

    if not cited_chunk_ids:
        return {}

    detail_map: Dict[str, Dict[str, Any]] = {}

    # Prefer evidence_items locators when available (source of truth for follow-up)
    if evidence_set_id:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ei.chunk_id, ei.locators_json
                FROM evidence_items ei
                WHERE ei.evidence_set_id = %s AND ei.chunk_id = ANY(%s)
            """, (evidence_set_id, cited_chunk_ids))
            for row in cur.fetchall():
                chunk_id, locators = row
                if isinstance(locators, str):
                    try:
                        locators = json.loads(locators)
                    except Exception:
                        locators = {}
                elif locators is None:
                    locators = {}
                doc_id = locators.get("doc_id")
                page_str = locators.get("page") or ""
                source_label = locators.get("source_label") or locators.get("collection_slug") or ""
                # Parse page: "p4" -> 4 (PDF page); "p12345" -> resolve page_id to pdf_page_number
                page_num = None
                if page_str:
                    m = _re.search(r"(\d+)", str(page_str))
                    if m:
                        raw = int(m.group(1))
                        if raw < 10000:
                            page_num = raw
                        else:
                            # Likely page_id; resolve to pdf_page_number
                            with conn.cursor() as pcur:
                                pcur.execute(
                                    "SELECT pdf_page_number, page_seq FROM pages WHERE id = %s",
                                    (raw,),
                                )
                                prow = pcur.fetchone()
                                if prow:
                                    page_num = prow[0] if prow[0] is not None else prow[1]
                if doc_id:
                    parts = []
                    if source_label:
                        parts.append(str(source_label).replace("_", " ").title())
                    if page_num is not None:
                        parts.append(f"p{page_num}")
                    human_label = " ".join(parts) if parts else f"chunk {chunk_id}"
                    detail = {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "page": page_num,
                        "label": human_label,
                    }
                    if human_label not in detail_map:
                        detail_map[human_label] = detail
                    detail_map[str(chunk_id)] = detail
                    detail_map[f"chunk {chunk_id}"] = detail

    # Fill in chunks missing document_id via chunk_metadata (with pipeline_version match)
    chunks_needing_fallback = [
        cid for cid in cited_chunk_ids
        if str(cid) not in detail_map or not detail_map.get(str(cid), {}).get("document_id")
    ]
    if chunks_needing_fallback:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, cm.document_id,
                       COALESCE(p.pdf_page_number, p.page_seq) AS page_num,
                       cm.collection_slug
                FROM chunks c
                LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = c.pipeline_version
                LEFT JOIN pages p ON p.id = cm.first_page_id
                WHERE c.id = ANY(%s)
            """, (chunks_needing_fallback,))
            for row in cur.fetchall():
                chunk_id, doc_id, page_num, source_label = row
                if doc_id is None:
                    continue
                parts = []
                if source_label:
                    parts.append(source_label.replace("_", " ").title())
                if page_num is not None:
                    parts.append(f"p{int(page_num)}")
                human_label = " ".join(parts) if parts else f"chunk {chunk_id}"
                detail = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "page": int(page_num) if page_num is not None else None,
                    "label": human_label,
                }
                if human_label not in detail_map:
                    detail_map[human_label] = detail
                detail_map[str(chunk_id)] = detail
                detail_map[f"chunk {chunk_id}"] = detail

    return detail_map


def _auto_label(question: str) -> str:
    """Generate a short label from the question."""
    # First 50 chars, cleaned up
    label = question[:50].strip()
    if len(question) > 50:
        label += "..."
    return label


def _extract_cited_chunk_ids(result: Optional[V9Result]) -> List[int]:
    """Extract all cited chunk IDs from a V9Result."""
    if not result:
        return []
    cids: set = set()
    for claim in (result.claims or []):
        if hasattr(claim, "citation_chunk_ids") and claim.citation_chunk_ids:
            cids.update(claim.citation_chunk_ids)
        if hasattr(claim, "support_chunk_ids") and claim.support_chunk_ids:
            cids.update(claim.support_chunk_ids)
    return sorted(cids)


# =============================================================================
# Verifiers
# =============================================================================

def _verify_think_deeper(
    run: RunRecord,
    target_run_id: int,
    new_step_idx: int,
    prev_last_step_idx: int,
) -> List[str]:
    """Verify think_deeper invariants per spec section 9.

    Checks:
    - run_id matches target (same run continued)
    - step_idx is monotonic (new steps come after previous)
    - new evidence has source_step_idx > previous_last_step_idx
    """
    violations = []
    if run.run_id != target_run_id:
        violations.append(f"Run ID mismatch: expected {target_run_id}, got {run.run_id}")
    if new_step_idx <= prev_last_step_idx:
        violations.append(
            f"Step index not monotonic: new={new_step_idx} <= prev={prev_last_step_idx}"
        )
    return violations


# =============================================================================
# Main dispatch entry point
# =============================================================================

def dispatch_message(
    conn,
    session_id: int,
    user_message: str,
    *,
    explicit_action: Optional[str] = None,
    carry_context: Optional[Dict[str, Any]] = None,
    max_tool_calls: Optional[int] = None,
    scope: Optional[ScopeFilter] = None,
    selected_scope: Optional[ScopeFilter] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
) -> DispatchResult:
    """Main entry point: route a user message and execute the appropriate path.

    Scope precedence for new_retrieval:
      1. natural-language scope in the query (confirmed with the user)
      2. selected_scope (the side-panel scope, sent with the query — no persistence race)
      3. explicit `scope` arg / session scope_json (legacy)


    Uses same V11 + Think Deeper stack as friday_cli.py:
    - new_retrieval: run_v11_query (unless USE_V9_AGENT=1)
    - think_deeper: v9_deep_runner.think_deeper (Actor/Judge)

    Env: V11_QUERY_MAX_TURNS, THINK_DEEPER_MAX_TOOL_CALLS (same as CLI)

    Args:
        conn: database connection
        session_id: session ID (must exist)
        user_message: the user's text
        explicit_action: override from API ("think_deeper")
        carry_context: forwarded entity/intent context from escalation buttons
        max_tool_calls: tool budget for new_retrieval (default from V11_QUERY_MAX_TURNS)
        scope: optional scope filter override
        verbose: log to stderr
        progress_callback: optional callback(step, status, message, details)
            for streaming progress to the frontend via SSE

    Returns:
        DispatchResult with answer, metadata, and routing info
    """
    t0 = time.time()
    if max_tool_calls is None:
        max_tool_calls = _max_tool_calls_from_env()

    if progress_callback:
        progress_callback("routing_start", "running", "Understanding your question...", {})

    # Ensure session exists
    session = load_session(conn, session_id)
    if not session:
        return DispatchResult(
            intent="new_retrieval",
            answer="Session not found.",
            confidence="insufficient",
            run_status="failed",
            elapsed_ms=0,
        )

    # Load recent runs for context
    context = load_recent_runs(conn, session_id)

    # Route the message
    intent_hint = (carry_context or {}).get("intent_hint") if carry_context else None
    decision = route_message(
        user_message, context,
        explicit_action=explicit_action,
        intent_hint=intent_hint,
        carry_context=carry_context,
        verbose=verbose,
    )

    # A v12 clarification answer is a fresh investigation with resolved intent —
    # never a follow-up to the (empty) paused clarification run that preceded it.
    if (carry_context or {}).get("clarification_answers") is not None and decision.intent != "new_retrieval":
        decision.intent = "new_retrieval"
        decision.target_run_id = None
        decision.target_evidence_set_id = None
        decision.reasoning = "clarification resume -> new_retrieval"

    if verbose:
        print(
            f"  [V9 Dispatch] Routing: intent={decision.intent}, "
            f"confidence={decision.confidence:.2f}, "
            f"reasoning={decision.reasoning[:80]}",
            file=sys.stderr,
        )

    if progress_callback:
        progress_callback("routing", "completed", f"Routed as: {decision.intent.replace('_', ' ')}", {
            "intent": decision.intent,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        })

    # Emit progress before execution (hooks for UI to show ongoing activity)
    if progress_callback and decision.intent == "new_retrieval":
        progress_callback("investigation_start", "running", "Starting investigation...", {"intent": "new_retrieval"})
    elif progress_callback and decision.intent == "follow_up":
        progress_callback("follow_up_start", "running", "Searching evidence set...", {"intent": "follow_up"})
    elif progress_callback and decision.intent == "think_deeper":
        progress_callback("think_deeper_start", "running", "Resuming Think Deeper...", {"intent": "think_deeper"})

    # Execute the appropriate path
    result: DispatchResult

    if decision.intent == "follow_up":
        # Get original query + run record for context and scope metadata
        original_query = None
        origin_run = None
        if decision.target_run_id:
            origin_run = load_run(conn, decision.target_run_id)
            if origin_run:
                original_query = origin_run.query_text

        result = _run_follow_up(
            conn, session_id, user_message,
            evidence_set_id=decision.target_evidence_set_id,
            original_query=original_query,
            origin_run=origin_run,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    elif decision.intent == "think_deeper":
        result = _run_think_deeper(
            conn, session_id, user_message,
            target_run_id=decision.target_run_id,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    else:
        # new_retrieval -- handle natural-language scope + confirmation first.
        scope_confirmed, confirmed_scope, retrieval_question = _resolve_scope_confirmation(
            conn, session, user_message, carry_context,
        )
        if not scope_confirmed:
            from retrieval.agent.scope_nl import detect_nl_scope
            nl = detect_nl_scope(conn, user_message, verbose=verbose)
            if nl.collections:
                # Ask the user to confirm the scope we parsed from their words.
                clar = _build_scope_clarification(conn, nl, session, user_message)
                if progress_callback:
                    progress_callback("scope_confirm", "completed",
                                      f"Confirm scope: {', '.join(nl.collections)}", {"collections": nl.collections})
                result = DispatchResult(
                    intent="clarify", needs_clarification=True, clarification=clar,
                    answer="", confidence="medium",
                )
                result.router_decision = decision
                result.elapsed_ms = (time.time() - t0) * 1000
                return result
            elif nl.full_archive:
                # Explicit "full archive" reset — apply directly, no confirmation needed.
                scope_confirmed, confirmed_scope = True, ScopeFilter()

        if scope_confirmed:
            resolved_scope = confirmed_scope
            run_scope_json = {
                "mode": "custom" if (confirmed_scope.collections or confirmed_scope.document_ids) else "full_archive",
                "source": "nl_scope_confirmed",
                "reason": "user confirmed natural-language scope",
                "expansion": {"policy": "venona_vassiliev_only",
                              "collections": list(CONCORDANCE_EXPANSION_TARGET_COLLECTIONS),
                              "triggered": False, "reason": None},
            }
        elif selected_scope is not None and not selected_scope.is_empty():
            # Panel scope sent with the query — authoritative for this run, no persist race.
            resolved_scope = selected_scope
            run_scope_json = {
                "mode": "custom" if (selected_scope.collections or selected_scope.document_ids) else "full_archive",
                "source": "panel_selected",
                "reason": "side-panel scope sent with query",
                "expansion": {"policy": "venona_vassiliev_only",
                              "collections": list(CONCORDANCE_EXPANSION_TARGET_COLLECTIONS),
                              "triggered": False, "reason": None},
            }
        else:
            resolved_scope, run_scope_json = _resolve_scope_for_run(
                conn, session, user_message, explicit_scope=scope, verbose=verbose,
            )

        result = _run_new_retrieval(
            conn, session_id, retrieval_question or user_message,
            max_tool_calls=max_tool_calls,
            scope=resolved_scope,
            run_scope_json=run_scope_json,
            verbose=verbose,
            progress_callback=progress_callback,
            carry_context=carry_context,
        )

    result.router_decision = decision
    result.elapsed_ms = (time.time() - t0) * 1000
    return result


# =============================================================================
# Natural-language scope confirmation
# =============================================================================

_SCOPE_PLAN_KIND = "scope_confirm"


def _build_scope_clarification(conn, nl, session, user_message: str) -> Dict[str, Any]:
    """Build a single-choice clarification confirming the NL-detected scope."""
    from retrieval.agent.scope_nl import strip_nl_scope
    # Human-readable names for the detected collections
    names = _collection_titles_for(conn, nl.collections)
    detected_label = ", ".join(names) if names else ", ".join(nl.collections)
    stripped = strip_nl_scope(user_message, nl.matched_phrases) or user_message

    options = [
        {"id": "opt_scoped", "label": f"Yes — search only {detected_label}",
         "value": "collections:" + ",".join(nl.collections),
         "hint": "Restrict this query to those sources"},
        {"id": "opt_full", "label": "No — search the full archive",
         "value": "full_archive", "hint": "Ignore the scope and search everything"},
    ]
    # If the panel scope is a meaningful custom scope different from the detection, offer it.
    panel = session.scope_json if isinstance(session.scope_json, dict) else {}
    if panel.get("mode") == "custom" and panel.get("included_collection_ids"):
        panel_filter = session_scope_to_filter(conn, panel)
        if panel_filter.collections and sorted(panel_filter.collections) != sorted(nl.collections):
            panel_names = ", ".join(panel_filter.collections)
            options.append({
                "id": "opt_panel", "label": f"Use my panel scope ({panel_names})",
                "value": "collections:" + ",".join(panel_filter.collections),
                "hint": "Keep the scope selected in the side panel",
            })

    return {
        "_kind": _SCOPE_PLAN_KIND,
        "_stripped_query": stripped,
        "rationale": "Confirming the search scope parsed from your question.",
        "questions": [{
            "id": "scope_confirm",
            "question": f"It looks like you want to search only **{detected_label}**. Is that right?",
            "kind": "single_choice",
            "category": "scope",
            "options": options,
            "allow_free_text": False,
            "why": "Natural-language scope is easy to misread, so I confirm before searching.",
        }],
    }


def _resolve_scope_confirmation(conn, session, user_message: str, carry_context):
    """If this call is the user's answer to a scope-confirmation clarification, resolve the
    chosen scope. Returns (confirmed: bool, scope: Optional[ScopeFilter], retrieval_question).
    """
    cc = carry_context or {}
    plan = cc.get("clarification_plan")
    answers = cc.get("clarification_answers")
    if not answers or not isinstance(plan, dict) or plan.get("_kind") != _SCOPE_PLAN_KIND:
        return False, None, None

    # Map the selected option id -> value
    selected_value = None
    opt_by_id = {}
    for q in plan.get("questions", []):
        for o in q.get("options", []):
            opt_by_id[o.get("id")] = o.get("value")
    for a in answers:
        for oid in (a.get("option_ids") or []):
            if oid in opt_by_id:
                selected_value = opt_by_id[oid]
                break
        if selected_value:
            break

    retrieval_question = plan.get("_stripped_query") or user_message
    if not selected_value or selected_value == "full_archive":
        return True, ScopeFilter(), retrieval_question
    if selected_value.startswith("collections:"):
        slugs = [s for s in selected_value.split(":", 1)[1].split(",") if s]
        return True, ScopeFilter(collections=slugs), retrieval_question
    return True, ScopeFilter(), retrieval_question


def _collection_titles_for(conn, slugs: List[str]) -> List[str]:
    if not slugs:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT slug, title FROM collections WHERE slug = ANY(%s)", (slugs,))
            m = {r[0]: (r[1] or r[0]) for r in cur.fetchall()}
        return [m.get(s, s) for s in slugs]
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return slugs
