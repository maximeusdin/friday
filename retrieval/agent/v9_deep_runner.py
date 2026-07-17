"""
V9 Think Deeper — Controller Loop.

Actor proposes 2-3 -> Judge selects -> Execute -> Rails -> FindingStore update
-> Judge scores -> stall guard -> stop/continue.

Deterministic budget, rails, and stall-guard logic.  LLM calls only in Actor
and Judge.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set

from retrieval.agent.v9_deep_types import (
    ACTION_EXPAND_SEEDS,
    ACTION_RETRIEVE,
    ACTION_STOP,
    ACTION_SYNTHESIZE,
    ACTION_VERIFY,
    GAP_TYPE_COVERAGE,
    GAP_TYPE_PRECISION,
    GAP_TYPE_ENTITY,
    GAP_TYPE_CONTRADICTION,
    LEAD_TYPE_CODENAME,
    LEAD_TYPE_DOC,
    LEAD_TYPE_ENTITY,
    QUERY_ORIGIN_LEAD_CHASE,
    TOOL_CALL_UNITS,
    VALID_ACTIONS,
    VALID_RETRIEVE_MODES,
    CandidateChunk,
    DeepState,
    JudgeVerdict,
    NextAction,
    NoveltyReport,
    ResearchDirective,
    ThinkDeeperResult,
    strip_for_judge,
)
from retrieval.agent.v9_deep_findings import FindingStore
from retrieval.agent.v9_deep_rails import RailsConfig, RailsReport, apply_rails, _infer_roster_intent
from retrieval.agent.v9_deep_judge import (
    build_coverage_stats,
    build_evidence_sample,
    judge_score_delta,
    judge_select_action,
    pivot_top_gap_to_lead,
    validate_verdict,
)
from retrieval.agent.v9_deep_prompts import (
    ACTOR_SYSTEM_PROMPT,
    build_actor_user_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_ACTOR_MODEL = os.getenv("V9_DEEP_ACTOR_MODEL", "gpt-4.1-mini-2025-04-14")


def _resolve_entity_ids(param_values: List[Any], lead_pool: Optional[Any]) -> List[int]:
    """
    Resolve entity_ids param: Actor may pass lead_ids (hex strings) instead of
    integer entity_ids. Resolve via LeadPool when needed.
    Returns list of integer entity_ids suitable for entity_mentions_tool.
    """
    resolved: List[int] = []
    lead_by_id: Dict[str, int] = {}
    if lead_pool and hasattr(lead_pool, "leads"):
        for lead in lead_pool.leads:
            lid = getattr(lead, "lead_id", None)
            eid = getattr(lead, "entity_id", None)
            if lid and eid is not None:
                lead_by_id[str(lid).strip().lower()] = int(eid)
    for val in param_values[:10]:
        if val is None:
            continue
        try:
            eid = int(val)
            if 0 < eid < 10**9:  # reasonable entity_id range
                resolved.append(eid)
                continue
        except (ValueError, TypeError):
            pass
        # Likely lead_id (hex string): resolve via LeadPool
        s = str(val).strip().lower()
        if s in lead_by_id:
            resolved.append(lead_by_id[s])
    return list(dict.fromkeys(resolved))[:5]  # dedupe, cap


# ── DB Persistence helpers ───────────────────────────────────────────────────

def _persist_td_run(
    conn,
    v9_run_id: int,
    directive: ResearchDirective,
) -> Optional[int]:
    """Create a think_deeper_runs row. Returns td_run_id or None on failure."""
    try:
        conn.rollback()  # Clear any aborted transaction from earlier operations
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO think_deeper_runs (v9_run_id, directive_json)
                   VALUES (%s, %s) RETURNING id""",
                (v9_run_id, json.dumps(directive.to_dict())),
            )
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
    except Exception as e:
        logger.warning("Failed to persist think_deeper_run: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def _persist_td_step(
    conn,
    td_run_id: int,
    step_idx: int,
    proposals: List[NextAction],
    selected_action: NextAction,
    candidates_count: int,
    rails_report: Optional[RailsReport],
    verdict: JudgeVerdict,
    selected_chunk_ids: List[int],
    new_findings_count: int,
) -> None:
    """Persist a single step to think_deeper_steps."""
    try:
        conn.rollback()  # Clear any aborted transaction from earlier operations
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO think_deeper_steps
                   (td_run_id, step_idx, actor_proposals_json, action_json,
                    candidates_count, rails_report_json, admitted_count,
                    judge_verdict_json, selected_chunk_ids, new_findings_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    td_run_id,
                    step_idx,
                    json.dumps([p.to_dict() for p in proposals]),
                    json.dumps(selected_action.to_dict()),
                    candidates_count,
                    json.dumps(rails_report.to_dict()) if rails_report else None,
                    rails_report.admitted_count if rails_report else 0,
                    json.dumps(verdict.to_dict()),
                    list(selected_chunk_ids),
                    new_findings_count,
                ),
            )
            conn.commit()
    except Exception as e:
        logger.exception("Failed to persist think_deeper_step %d: %s", step_idx, e, exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            status = conn.get_transaction_status() if hasattr(conn, "get_transaction_status") else "unknown"
            logger.warning("Transaction status after persist failure: %s", status)
        except Exception:
            pass


def _finalize_td_run(
    conn,
    td_run_id: int,
    result: ThinkDeeperResult,
) -> None:
    """Update the think_deeper_runs row with final scores and stop reason."""
    try:
        conn.rollback()  # Clear any aborted transaction from earlier operations
        final_scores = {}
        if result.verdict_history:
            v = result.verdict_history[-1]
            final_scores = {
                "answeredness": v.answeredness,
                "material_novelty": v.material_novelty,
                "confidence": v.confidence,
            }
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE think_deeper_runs
                   SET final_scores_json = %s, stop_reason = %s,
                       steps_executed = %s, tool_calls_used = %s, elapsed_ms = %s
                   WHERE id = %s""",
                (
                    json.dumps(final_scores),
                    result.stop_reason,
                    result.steps_executed,
                    result.tool_calls_used,
                    result.elapsed_ms,
                    td_run_id,
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to finalize think_deeper_run %d: %s", td_run_id, e)
        try:
            conn.rollback()
        except Exception:
            pass


# ── Directive builder ────────────────────────────────────────────────────────

def build_directive(
    seed_question: str,
    user_followup: Optional[str],
    workspace,  # ResearchWorkspace or None
) -> ResearchDirective:
    """Build a ResearchDirective from seed question + optional followup.

    If a user_followup is present, uses a small LLM call to extract:
      - entity names / date ranges / collections to focus on (must_include)
      - sub-questions the user wants answered (must_answer)
      - dissatisfaction classification -> directive weight adjustments
    Falls back to keyword heuristics if the LLM call fails.
    """
    from retrieval.agent.v9_deep_types import (
        AvoidSpec,
        DirectiveWeights,
        MustInclude,
    )

    directive = ResearchDirective(
        primary_question=seed_question,
        user_directive=user_followup,
    )

    # Extract entity IDs from workspace if available
    if workspace:
        entity_ids = []
        for ent in getattr(workspace, "entities", []):
            eid = getattr(ent, "entity_id", None)
            if eid:
                entity_ids.append(eid)
        if entity_ids:
            directive.must_include.entity_ids = entity_ids[:20]

    # If user_followup present, extract must_include and dissatisfaction via LLM
    if user_followup:
        extracted = _extract_followup_intent(seed_question, user_followup)
        if extracted:
            # Must-include entities (names — will be resolved to IDs later if needed)
            if extracted.get("focus_entities"):
                # Store names in collections as a hint for the Actor
                directive.must_include.collections.extend(
                    extracted.get("focus_collections", [])
                )
            if extracted.get("focus_date_from") or extracted.get("focus_date_to"):
                dr: Dict[str, str] = {}
                if extracted.get("focus_date_from"):
                    dr["from"] = extracted["focus_date_from"]
                if extracted.get("focus_date_to"):
                    dr["to"] = extracted["focus_date_to"]
                directive.must_include.date_ranges.append(dr)
            if extracted.get("sub_questions"):
                directive.must_answer = extracted["sub_questions"][:5]
            if extracted.get("avoid_claims"):
                directive.avoid.claims = extracted["avoid_claims"][:5]

            # Dissatisfaction-based weight adjustments
            dissatisfaction = extracted.get("dissatisfaction_type", "none")
            if dissatisfaction == "not_thorough":
                directive.weights.coverage = 1.5
            elif dissatisfaction == "not_novel":
                directive.weights.novelty = 1.5
            elif dissatisfaction == "weak_evidence":
                directive.weights.support = 1.5
                directive.weights.verification = 1.5
            elif dissatisfaction == "off_topic":
                directive.weights.coverage = 1.3
                directive.weights.novelty = 1.3
        else:
            # Fallback: keyword heuristic for dissatisfaction
            _apply_keyword_dissatisfaction(directive, user_followup)

    return directive


def _extract_followup_intent(
    seed_question: str,
    user_followup: str,
) -> Optional[Dict[str, Any]]:
    """Small LLM call to extract must_include + dissatisfaction from followup.

    Returns dict with:
      - focus_entities: List[str]    (entity names to focus on)
      - focus_collections: List[str] (collection slugs to prioritize)
      - focus_date_from: Optional[str]
      - focus_date_to: Optional[str]
      - sub_questions: List[str]     (up to 3 sub-questions)
      - avoid_claims: List[str]      (claims to avoid repeating)
      - dissatisfaction_type: str    (none|not_thorough|not_novel|weak_evidence|off_topic)
    """
    try:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": _FOLLOWUP_EXTRACTION_SYSTEM},
                {"role": "user", "content": (
                    f"Original question: {seed_question}\n\n"
                    f"User follow-up: {user_followup}"
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=500,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.warning("Followup intent extraction failed: %s; using keyword fallback", e)
        return None


_FOLLOWUP_EXTRACTION_SYSTEM = """\
You extract structured intent from a user's follow-up message in a research session.

The user has already received an initial answer and is now asking for more.
Analyze their follow-up to determine:

1. **Focus entities**: Names of people, organizations, or codenames they want \
to focus on. Return as a list of strings.
2. **Focus collections**: If they mention specific document collections \
(e.g. "Vassiliev notebooks", "Venona decrypts"), list them.
3. **Date range**: If they specify a time period, extract date_from/date_to.
4. **Sub-questions**: Up to 3 specific sub-questions implicit in their follow-up.
5. **Avoid claims**: Things they already know or don't want repeated.
6. **Dissatisfaction type**: Classify their sentiment:
   - "none": Neutral follow-up, just wants more info
   - "not_thorough": Thinks the answer was incomplete ("dig deeper", "more detail")
   - "not_novel": Thinks the answer repeated known info ("already knew this", \
     "nothing new")
   - "weak_evidence": Wants stronger proof ("need evidence", "verify this")
   - "off_topic": Thinks the answer missed the point ("that's not what I asked")

Output JSON:
```json
{
  "focus_entities": ["..."],
  "focus_collections": ["..."],
  "focus_date_from": null,
  "focus_date_to": null,
  "sub_questions": ["..."],
  "avoid_claims": ["..."],
  "dissatisfaction_type": "none"
}
```
"""


def _apply_keyword_dissatisfaction(
    directive: ResearchDirective,
    user_followup: str,
) -> None:
    """Fallback keyword-based dissatisfaction detection."""
    lower = user_followup.lower()
    if any(w in lower for w in ("not thorough", "more detail", "dig deeper", "incomplete")):
        directive.weights.coverage = 1.5
    if any(w in lower for w in ("same stuff", "already knew", "nothing new", "repetitive")):
        directive.weights.novelty = 1.5
    if any(w in lower for w in ("proof", "evidence", "verify", "support")):
        directive.weights.support = 1.5
        directive.weights.verification = 1.5


# ── State initialization ─────────────────────────────────────────────────────

def init_deep_state(
    conn,
    seed_question: str,
    directive: ResearchDirective,
    workspace,  # ResearchWorkspace
) -> DeepState:
    """Initialize DeepState from workspace + evidence."""
    from retrieval.ops import embed_query

    # Precompute seed embedding for drift detection
    try:
        seed_embedding = embed_query(seed_question)
    except Exception as e:
        logger.warning("Failed to embed seed question: %s; using empty vector", e)
        seed_embedding = []

    # Baseline tracking from workspace
    baseline_chunk_ids: Set[int] = set()
    baseline_doc_ids: Set[int] = set()
    baseline_entity_ids: Set[int] = set()
    selected_chunks: List[CandidateChunk] = []

    if workspace:
        for c in getattr(workspace, "fulltext_chunks", []):
            cid = c.chunk_id
            baseline_chunk_ids.add(cid)
            if c.doc_id:
                baseline_doc_ids.add(c.doc_id)
            selected_chunks.append(CandidateChunk(
                chunk_id=cid,
                doc_id=c.doc_id or 0,
                collection_slug=getattr(c, "source_label", "") or "",
                page=c.page,
                score=getattr(c, "score", 0.0) or 0.0,
                entity_ids=[],
                text=c.text[:500] if c.text else "",
                is_new_vs_baseline=False,
                source_step=0,
            ))
        for ent in getattr(workspace, "entities", []):
            eid = getattr(ent, "entity_id", None)
            if eid:
                baseline_entity_ids.add(eid)

    # Seed FindingStore from evidence memory
    evidence_memory = getattr(workspace, "evidence_memory", []) if workspace else []
    finding_store = FindingStore.seed_from_evidence_summary(evidence_memory)

    # Extract initial leads from baseline chunks (for step 0 actor)
    lead_pool = None
    if selected_chunks and conn:
        from retrieval.agent.v9_deep_leads import extract_leads
        lead_pool = extract_leads(
            conn,
            selected_chunks,
            baseline_entity_ids,
            baseline_doc_ids,
            current_step=0,
            baseline_collection_slugs=set(),
        )

    return DeepState(
        seed_question=seed_question,
        seed_embedding=seed_embedding,
        directive=directive,
        baseline_chunk_ids=baseline_chunk_ids,
        baseline_doc_ids=baseline_doc_ids,
        baseline_entity_ids=baseline_entity_ids,
        selected_chunks=selected_chunks,
        finding_store=finding_store,
        lead_pool=lead_pool,
    )


# ── Actor: propose actions ───────────────────────────────────────────────────

def actor_propose(
    state: DeepState,
    prev_verdict: Optional[JudgeVerdict],
    must_target_unseen: bool,
    max_tool_calls: int,
    *,
    force_recovery_mode: bool = False,
    force_lead_chase: bool = False,
    model: str = "",
    verbose: bool = True,
) -> List[NextAction]:
    """Ask the Actor LLM to propose 2-3 candidate actions."""
    import json as _json
    from openai import OpenAI

    model = model or _DEFAULT_ACTOR_MODEL
    client = OpenAI()

    # Build state summary
    state_summary = _build_state_summary(state)
    directive_summary = _build_directive_summary(state.directive)
    prev_summary = _build_prev_verdict_summary(prev_verdict) if prev_verdict else ""
    budget_remaining = max_tool_calls - state.tool_calls_used
    pressure_summary = compute_pressure_summary(state, prev_verdict, must_target_unseen)

    user_prompt = build_actor_user_prompt(
        seed_question=state.seed_question,
        directive_summary=directive_summary,
        state_summary=state_summary,
        prev_verdict_summary=prev_summary,
        must_target_unseen=must_target_unseen,
        budget_remaining=budget_remaining,
        pressure_summary=pressure_summary,
        force_recovery_mode=force_recovery_mode,
        baseline_entity_ids=state.baseline_entity_ids,
        lead_pool=state.lead_pool,
        pivot_gap_phrase=pivot_top_gap_to_lead(state.verdict_history, state.lead_pool),
        force_lead_chase=force_lead_chase,
    )

    if verbose:
        print(f"  [ThinkDeeper] Actor proposing actions (step {state.step})...",
              file=sys.stderr)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500,
        )
        content = response.choices[0].message.content or "[]"
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        # Parse — could be array or {"proposals": [...]}
        parsed = _json.loads(content)
        if isinstance(parsed, dict):
            proposals_raw = parsed.get("proposals") or parsed.get("actions") or []
        elif isinstance(parsed, list):
            proposals_raw = parsed
        else:
            proposals_raw = []
        if not isinstance(proposals_raw, list):
            proposals_raw = []

        proposals = []
        for p in proposals_raw[:3]:
            if not isinstance(p, dict):
                continue
            action = p.get("action", ACTION_STOP)
            if action not in VALID_ACTIONS:
                action = ACTION_STOP
            try:
                proposals.append(NextAction.from_dict(p))
            except Exception as parse_err:
                logger.warning("Actor proposal parse failed for %r: %s", p, parse_err)

        if not proposals:
            if verbose:
                raw_preview = (
                    list(parsed.keys()) if isinstance(parsed, dict) else f"list(len={len(parsed)})"
                )
                content_preview = content[:200] + "..." if len(content) > 200 else content
                print(
                    f"  [ThinkDeeper] Actor returned empty proposals; parsed={raw_preview}, "
                    f"proposals_raw={proposals_raw[:3] if proposals_raw else []}, "
                    f"content_preview={content_preview!r}",
                    file=sys.stderr,
                )
            # Fallback: propose a generic RETRIEVE
            proposals = [NextAction(
                action=ACTION_RETRIEVE,
                params={"queries": [state.seed_question], "mode": "hybrid", "top_k": 10},
                why="Fallback: no valid proposals from Actor",
            )]

        if verbose:
            for i, p in enumerate(proposals):
                print(f"  [ThinkDeeper]   Proposal {i}: {p.action} — {p.why[:80]}",
                      file=sys.stderr)
        return proposals

    except Exception as e:
        logger.warning("Actor propose failed: %s; returning fallback", e)
        return [NextAction(
            action=ACTION_RETRIEVE,
            params={"queries": [state.seed_question], "mode": "hybrid", "top_k": 10},
            why=f"Fallback after Actor error: {e}",
        )]


# ── Execute action ───────────────────────────────────────────────────────────

def _effective_scope(
    run_scope: Optional["ScopeFilter"],
    actor_scope_params: Optional[Dict[str, Any]],
) -> Optional["ScopeFilter"]:
    """Compute effective scope: run_scope constrains; actor can only narrow within it."""
    from retrieval.agent.v9_types import ScopeFilter

    if run_scope and not run_scope.is_empty():
        # Run scope is the constraint; actor params can narrow but not widen
        if not actor_scope_params:
            return run_scope
        actor_colls = actor_scope_params.get("collections") or []
        actor_doc_ids = actor_scope_params.get("doc_ids") or []
        if not actor_colls and not actor_doc_ids:
            return run_scope
        # Intersect: only allow actor's choices that are within run scope
        eff_colls = None
        if run_scope.collections and actor_colls:
            in_scope = set(run_scope.collections)
            eff_colls = [c for c in actor_colls if c in in_scope]
            if not eff_colls:
                eff_colls = run_scope.collections  # actor picked outside scope, use full run scope
        elif run_scope.collections:
            eff_colls = run_scope.collections
        eff_doc_ids = None
        if run_scope.document_ids and actor_doc_ids:
            in_scope = set(run_scope.document_ids)
            eff_doc_ids = [d for d in actor_doc_ids if d in in_scope]
            if not eff_doc_ids:
                eff_doc_ids = run_scope.document_ids
        elif run_scope.document_ids:
            eff_doc_ids = run_scope.document_ids
        return ScopeFilter(
            collections=eff_colls,
            document_ids=eff_doc_ids,
            date_from=run_scope.date_from,
            date_to=run_scope.date_to,
        )
    # No run scope: use actor's scope if any
    if actor_scope_params:
        return ScopeFilter(
            collections=actor_scope_params.get("collections"),
            document_ids=actor_scope_params.get("doc_ids"),
            date_from=actor_scope_params.get("date_from"),
            date_to=actor_scope_params.get("date_to"),
        )
    return None


def execute_action(
    conn,
    action: NextAction,
    state: DeepState,
    *,
    run_scope: Optional["ScopeFilter"] = None,
    verbose: bool = True,
) -> List[CandidateChunk]:
    """Execute a single action via v9_tools.  Returns new candidate chunks.

    When run_scope is provided (from Think Deeper), it constrains all searches
    to stay within the original run's scope. Actor-proposed scope can only narrow.
    """
    from retrieval.agent.v9_tools import search_chunks, fetch_chunks, expand_entities
    from retrieval.agent.v9_types import ScopeFilter

    candidates: List[CandidateChunk] = []
    scope_params = action.params.get("scope", {}) if action.params else {}
    scope = _effective_scope(run_scope, scope_params)
    collections = scope.collections if scope else (scope_params.get("collections") if scope_params else None)

    if action.action == ACTION_RETRIEVE:
        queries = action.params.get("queries", [state.seed_question])
        mode = action.params.get("mode", "hybrid")
        top_k = action.params.get("top_k", 10)
        entity_ids_param = action.params.get("entity_ids", [])

        # mode="evidence_leads": extract leads from chunks, generate queries, search
        if mode == "evidence_leads":
            from_chunk_ids = action.params.get("from_chunk_ids") or []
            lead_types = action.params.get("lead_types") or ["entity", "org"]
            k_leads = action.params.get("k_leads", 5)
            if from_chunk_ids:
                chunks_for_ids = [c for c in state.selected_chunks if c.chunk_id in from_chunk_ids]
                if not chunks_for_ids:
                    chunks_for_ids = [c for c in state.selected_chunks[:20]]
                from retrieval.agent.v9_deep_leads import extract_leads
                pool = extract_leads(
                    conn, chunks_for_ids, state.baseline_entity_ids, state.baseline_doc_ids,
                    state.step, baseline_collection_slugs=set(),
                )
                all_queries = []
                for lead in pool.leads[:k_leads]:
                    if lead.type not in lead_types:
                        continue
                    if lead.type == "entity" and lead.entity_id:
                        from retrieval.agent.tools import entity_surfaces_tool
                        try:
                            res = entity_surfaces_tool(conn, entity_id=lead.entity_id)
                            if res.success:
                                surfaces = res.metadata.get("surfaces", []) or [lead.value]
                                all_queries.extend(surfaces[:3])
                            else:
                                all_queries.append(lead.value)
                        except Exception:
                            all_queries.append(lead.value)
                    else:
                        all_queries.append(lead.value)
                for q in all_queries[:8]:
                    try:
                        tool_result, catalog_hits = search_chunks(
                            conn, str(q), top_k=min(top_k, 15), scope=scope, mode="lexical_exact",
                        )
                        for hit in catalog_hits:
                            if hit.chunk_id not in state.selected_chunk_ids:
                                candidates.append(CandidateChunk(
                                    chunk_id=hit.chunk_id, doc_id=hit.doc_id or 0,
                                    collection_slug=hit.collection or "", page=hit.page,
                                    score=hit.score, text=hit.snippet,
                                    is_new_vs_baseline=hit.chunk_id not in state.baseline_chunk_ids,
                                    source_step=state.step,
                                ))
                    except Exception:
                        pass
            if verbose and from_chunk_ids:
                print(f"  [ThinkDeeper]   RETRIEVE(mode=evidence_leads, from={len(from_chunk_ids)} chunks) -> {len(candidates)} chunks",
                      file=sys.stderr)

        # mode="adjacent": same-doc chunks within window (chunk count, not page)
        elif mode == "adjacent":
            around_ids = action.params.get("around_chunk_ids") or []
            window = action.params.get("window_pages", 6)  # chunks before/after
            if around_ids:
                from retrieval.ops import get_chunk_neighbors
                for cid in around_ids[:5]:
                    try:
                        nbrs = get_chunk_neighbors(conn, cid, before=window, after=window, include_seed=False)
                        for n in nbrs:
                            if n.chunk_id not in state.selected_chunk_ids:
                                candidates.append(CandidateChunk(
                                    chunk_id=n.chunk_id, doc_id=n.document_id or 0,
                                    collection_slug=n.collection_slug or "", page=None,
                                    score=0.5, text=n.text[:500] if n.text else "",
                                    is_new_vs_baseline=n.chunk_id not in state.baseline_chunk_ids,
                                    source_step=state.step,
                                ))
                    except Exception:
                        pass
            if verbose and around_ids:
                print(f"  [ThinkDeeper]   RETRIEVE(mode=adjacent, around={len(around_ids)}) -> {len(candidates)} chunks",
                      file=sys.stderr)

        # mode="mentions" + entity_ids: roster retrieval via entity_mentions_tool
        # Actor may pass lead_ids (hex) instead of entity_ids — resolve via LeadPool
        elif mode == "mentions" and entity_ids_param:
            from retrieval.agent.tools import entity_mentions_tool
            resolved_eids = _resolve_entity_ids(entity_ids_param, state.lead_pool)
            if not resolved_eids:
                logger.warning("entity_ids=%s resolved to 0 (lead_ids?); check LeadPool", entity_ids_param[:3])
            all_cids = []
            for eid in resolved_eids:
                try:
                    res = entity_mentions_tool(
                        conn, entity_id=eid, top_k=min(top_k, 50),
                        collections=collections,
                    )
                    if res.success and res.chunk_ids:
                        all_cids.extend(res.chunk_ids[:30])
                except Exception as e:
                    logger.warning("entity_mentions_tool failed for entity %s: %s", eid, e)
            all_cids = list(dict.fromkeys(all_cids))[:50]
            if all_cids:
                full_chunks = fetch_chunks(conn, chunk_ids=all_cids)
                for c in full_chunks:
                    if c.chunk_id not in state.selected_chunk_ids:
                        candidates.append(CandidateChunk(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id or 0,
                            collection_slug=getattr(c, "source_label", "") or "",
                            page=c.page,
                            score=0.6,
                            text=c.text[:500] if c.text else "",
                            is_new_vs_baseline=c.chunk_id not in state.baseline_chunk_ids,
                            source_step=state.step,
                        ))
            if verbose:
                print(f"  [ThinkDeeper]   RETRIEVE(mode=mentions, entity_ids={resolved_eids}) -> {len(candidates)} chunks",
                      file=sys.stderr)
        else:
            # Map mode for query-based search
            tool_mode = "hybrid"
            if mode == "lexical":
                tool_mode = "lexical_exact"
            elif mode == "mentions":
                tool_mode = "hybrid"  # mentions without entity_ids falls back to hybrid

            for q in queries[:3]:
                try:
                    tool_result, catalog_hits = search_chunks(
                        conn, q, top_k=top_k, scope=scope, mode=tool_mode,
                    )
                    if verbose:
                        print(f"  [ThinkDeeper]   search_chunks('{q[:60]}') -> {len(catalog_hits)} hits",
                              file=sys.stderr)

                    # Convert to CandidateChunk
                    for hit in catalog_hits:
                        if hit.chunk_id in state.selected_chunk_ids:
                            continue  # already selected
                        candidates.append(CandidateChunk(
                            chunk_id=hit.chunk_id,
                            doc_id=hit.doc_id or 0,
                            collection_slug=hit.collection or "",
                            page=hit.page,
                            score=hit.score,
                            text=hit.snippet,
                            is_new_vs_baseline=hit.chunk_id not in state.baseline_chunk_ids,
                            source_step=state.step,
                        ))
                except Exception as e:
                    logger.warning("search_chunks failed for query '%s': %s", q[:50], e)

        # Fetch full text for top candidates (to populate text field) — skip if already fetched
        top_cids = [c.chunk_id for c in sorted(candidates, key=lambda x: x.score, reverse=True)[:20]]
        if top_cids:
            try:
                full_chunks = fetch_chunks(conn, chunk_ids=top_cids)
                text_map = {c.chunk_id: c.text for c in full_chunks}
                entity_map: Dict[int, List[int]] = {}
                for c in full_chunks:
                    # entity_ids may come from workspace entity resolution
                    entity_map[c.chunk_id] = []
                for cand in candidates:
                    if cand.chunk_id in text_map:
                        cand.text = text_map[cand.chunk_id][:500]
                    if cand.chunk_id in entity_map:
                        cand.entity_ids = entity_map[cand.chunk_id]
            except Exception as e:
                logger.warning("fetch_chunks failed: %s", e)

    elif action.action == ACTION_EXPAND_SEEDS:
        seed_ids_raw = action.params.get("seed_entity_ids", [])
        seed_ids = _resolve_entity_ids(seed_ids_raw, state.lead_pool) if seed_ids_raw else []
        if not seed_ids:
            seed_ids = list(state.baseline_entity_ids)[:5]
        try:
            result = expand_entities(
                conn, entity_ids=seed_ids, include_comentions=True,
                scope=scope,
            )
            # expand_entities returns chunk_ids at top level (not mention_chunk_ids)
            mention_chunks = result.get("chunk_ids", [])
            if verbose:
                print(f"  [ThinkDeeper]   expand_entities -> {len(mention_chunks)} mention chunks",
                      file=sys.stderr)

            # Convert mention chunk_ids to candidates via fetch
            if mention_chunks:
                full_chunks = fetch_chunks(conn, chunk_ids=mention_chunks[:30])
                for c in full_chunks:
                    if c.chunk_id not in state.selected_chunk_ids:
                        candidates.append(CandidateChunk(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id or 0,
                            collection_slug=getattr(c, "source_label", "") or "",
                            page=c.page,
                            score=0.5,  # default score for entity expansion
                            text=c.text[:500] if c.text else "",
                            is_new_vs_baseline=c.chunk_id not in state.baseline_chunk_ids,
                            source_step=state.step,
                        ))
            # Fallback: when 0 mention chunks but entities resolved, RETRIEVE via suggested queries
            elif result.get("suggested_retrieval_queries"):
                suggested = result["suggested_retrieval_queries"][:3]
                for q in suggested:
                    try:
                        tool_result, catalog_hits = search_chunks(
                            conn, q, top_k=10, scope=scope, mode="lexical_exact",
                        )
                        if verbose:
                            print(f"  [ThinkDeeper]   expand fallback search_chunks('{q[:40]}') -> {len(catalog_hits)} hits",
                                  file=sys.stderr)
                        for hit in catalog_hits:
                            if hit.chunk_id not in state.selected_chunk_ids:
                                candidates.append(CandidateChunk(
                                    chunk_id=hit.chunk_id,
                                    doc_id=hit.doc_id or 0,
                                    collection_slug=hit.collection or "",
                                    page=hit.page,
                                    score=0.5,
                                    text=hit.snippet,
                                    is_new_vs_baseline=hit.chunk_id not in state.baseline_chunk_ids,
                                    source_step=state.step,
                                ))
                    except Exception as e:
                        logger.warning("expand fallback search_chunks failed: %s", e)
        except Exception as e:
            logger.warning("expand_entities failed: %s", e)

    elif action.action in (ACTION_SYNTHESIZE, ACTION_VERIFY, ACTION_STOP):
        # No new evidence — these are LLM-only actions
        pass

    # Dedup candidates
    seen: Set[int] = set()
    deduped = []
    for c in candidates:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            deduped.append(c)

    return deduped


# ── Stall guard ──────────────────────────────────────────────────────────────

def check_novelty_stall(state: DeepState) -> bool:
    """Returns True if stalled on EITHER dimension for 2+ steps AND novelty low.

    Stall = (no new docs OR no new findings) for 2 steps AND material_novelty < 0.4.
    """
    if len(state.new_doc_count_by_step) < 2:
        return False
    last_verdict = state.verdict_history[-1] if state.verdict_history else None
    novelty_low = last_verdict and last_verdict.material_novelty < 0.4
    if not novelty_low:
        return False

    no_new_docs = all(c == 0 for c in state.new_doc_count_by_step[-2:])
    no_new_findings = (
        state.finding_store.new_findings_count_last_n_steps(2, state.step) == 0
        if state.finding_store else True
    )
    return no_new_docs or no_new_findings


def compute_pressure_summary(
    state: DeepState,
    prev_verdict: Optional[JudgeVerdict],
    must_target_unseen: bool,
) -> str:
    """Compute explore/exploit pressure signals for Actor/Judge prompts.

    Exploit pressure: high when confidence low, need triangulation.
    Explore pressure: high when novelty stalled, material_novelty low, need new sources.
    """
    lines: List[str] = []
    if not prev_verdict:
        return ""

    # Exploit pressure
    exploit_signals = []
    if prev_verdict.confidence < 0.5:
        exploit_signals.append("confidence is low — prioritize triangulation")
    precision_gaps = [g for g in prev_verdict.top_gaps if g.type == GAP_TYPE_PRECISION]
    entity_gaps = [g for g in prev_verdict.top_gaps if g.type == GAP_TYPE_ENTITY]
    if precision_gaps or entity_gaps:
        typ = "precision" if precision_gaps else "entity"
        exploit_signals.append(f"need {typ} detail — prioritize exploit (tightening/triangulation)")

    if exploit_signals:
        lines.append("Exploit pressure: " + "; ".join(exploit_signals))

    # Explore pressure
    explore_signals = []
    if must_target_unseen:
        explore_signals.append("novelty stalled — prioritize finding new independent sources")
    if prev_verdict.material_novelty < 0.4 and prev_verdict.top_gaps:
        coverage_gaps = [g for g in prev_verdict.top_gaps if g.type == GAP_TYPE_COVERAGE]
        if coverage_gaps:
            explore_signals.append("coverage gaps — prioritize new docs/collections")

    if explore_signals:
        lines.append("Explore pressure: " + "; ".join(explore_signals))

    return "\n".join(lines) if lines else ""


def _build_lead_chase_retrieve(lead_pool) -> Optional[NextAction]:
    """Build a deterministic LEAD_CHASE RETRIEVE from top 1-2 leads."""
    leads = getattr(lead_pool, "leads", []) or []
    if not leads:
        return None
    top = leads[0]
    lead_type = getattr(top, "type", "")
    value = getattr(top, "value", "") or ""
    entity_id = getattr(top, "entity_id", None)
    doc_id = getattr(top, "doc_id", None)
    lead_id = getattr(top, "lead_id", "")

    if lead_type == LEAD_TYPE_ENTITY and entity_id:
        return NextAction(
            action=ACTION_RETRIEVE,
            params={"mode": "mentions", "entity_ids": [entity_id], "top_k": 15},
            why="Auto-injected: lead-chase from top entity lead",
            proposal_intent="explore",
            query_origin="LEAD_CHASE",
            leads_used=[lead_id] if lead_id else [],
        )
    if lead_type == LEAD_TYPE_CODENAME and value:
        return NextAction(
            action=ACTION_RETRIEVE,
            params={"queries": [value], "mode": "lexical", "top_k": 15},
            why="Auto-injected: lead-chase from top codename lead",
            proposal_intent="explore",
            query_origin="LEAD_CHASE",
            leads_used=[lead_id] if lead_id else [],
        )
    if lead_type == LEAD_TYPE_DOC and doc_id:
        return NextAction(
            action=ACTION_RETRIEVE,
            params={"queries": [value or f"doc {doc_id}"], "mode": "lexical", "top_k": 15,
                    "scope": {"doc_ids": [doc_id]}},
            why="Auto-injected: lead-chase from top doc lead",
            proposal_intent="explore",
            query_origin="LEAD_CHASE",
            leads_used=[lead_id] if lead_id else [],
        )
    # Fallback: use first lead's value as query
    if value:
        return NextAction(
            action=ACTION_RETRIEVE,
            params={"queries": [value], "mode": "lexical", "top_k": 15},
            why="Auto-injected: lead-chase from top lead",
            proposal_intent="explore",
            query_origin="LEAD_CHASE",
            leads_used=[lead_id] if lead_id else [],
        )
    return None


def compute_gap_types_summary(prev_verdict: Optional[JudgeVerdict]) -> str:
    """Format gap types for Judge selection prompt."""
    if not prev_verdict or not prev_verdict.top_gaps:
        return ""
    lines = []
    for i, g in enumerate(prev_verdict.top_gaps[:3], 1):
        lines.append(f"  {i}. [{g.type}] {g.target} (priority={g.priority:.1f})")
    return "\n".join(lines)


# ── Stop condition ───────────────────────────────────────────────────────────

def should_stop(
    verdict: JudgeVerdict,
    state: DeepState,
    config: RailsConfig,
) -> bool:
    """Evaluate whether the loop should stop."""
    # Budget exhausted (non-negotiable)
    if state.tool_calls_used >= config.max_tool_calls:
        return True

    # Min-evidence-growth stop
    if len(state.filtered_admitted_by_step) >= 2:
        last_2 = state.filtered_admitted_by_step[-2:]
        if all(c < 2 for c in last_2) and verdict.ev_next_step < 0.25:
            return True

    # Self-consistency disagreement -> force one more step
    if verdict.self_consistency_divergence > 0.3:
        return False

    # Judge-led stop
    judge_wants_stop = verdict.stop_recommendation and verdict.ev_next_step < 0.25
    judge_confident_stop = verdict.stop_recommendation and verdict.confidence > 0.8

    return judge_confident_stop or (judge_wants_stop and state.step >= 2)


# ── Main controller loop ────────────────────────────────────────────────────

def think_deeper(
    conn,
    seed_question: str,
    workspace,  # ResearchWorkspace
    user_followup: Optional[str] = None,
    *,
    max_steps: int = 8,
    max_tool_calls: int = 10,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
    v9_run_id: Optional[int] = None,
    run_scope: Optional["ScopeFilter"] = None,
) -> ThinkDeeperResult:
    """Run the Think Deeper controller loop.

    Actor proposes -> Judge selects -> Execute -> Rails -> Judge scores ->
    FindingStore update -> stall guard -> stop/continue.
    """
    t0 = time.time()

    # 1. Build directive
    directive = build_directive(seed_question, user_followup, workspace)
    roster_intent = _infer_roster_intent(directive.primary_question, directive.user_directive)
    rails_config = RailsConfig(max_tool_calls=max_tool_calls, roster_intent=roster_intent)

    # 2. Initialize state
    state = init_deep_state(conn, seed_question, directive, workspace)

    # Persist run start (if we have a v9_run_id to link to)
    td_run_id = None
    if v9_run_id is not None:
        td_run_id = _persist_td_run(conn, v9_run_id, directive)

    if verbose:
        print(
            f"  [ThinkDeeper] Starting: baseline={len(state.baseline_chunk_ids)} chunks, "
            f"{len(state.baseline_doc_ids)} docs, {len(state.baseline_entity_ids)} entities, "
            f"{state.finding_store.total_count()} baseline findings",
            file=sys.stderr,
        )

    # Rollback any aborted transaction before loop (fixes "current transaction is aborted")
    try:
        conn.rollback()
    except Exception:
        pass

    # 3. Loop
    prev_verdict: Optional[JudgeVerdict] = None
    stop_reason = "max_steps reached"

    for step in range(max_steps):
        state.step = step

        if progress_callback:
            progress_callback("think_deeper_step", "running",
                f"Think Deeper step {step + 1}/{max_steps}",
                {"step": step + 1, "max_steps": max_steps})

        # ── Stall check ──────────────────────────────────────────────
        must_target_unseen = check_novelty_stall(state)
        if must_target_unseen and verbose:
            print(f"  [ThinkDeeper] Novelty stall detected at step {step}; "
                  "forcing unseen targets", file=sys.stderr)

        # ── Zero-hits recovery: nudge Actor when RETRIEVE returned 0 twice ──
        force_recovery_mode = state.consecutive_zero_hits >= 2

        # ── Force lead-chase when overlap high + zero frontier expansion ──
        force_lead_chase = False
        if state.frontier_metrics_by_step:
            last_m = state.frontier_metrics_by_step[-1]
            overlap = last_m.get("overlap_with_prev_queries", 0)
            frontier = (last_m.get("new_entity_count", 0) + last_m.get("new_doc_count", 0)
                       + last_m.get("new_collection_count", 0))
            if overlap > 0.6 and frontier == 0:
                force_lead_chase = True

        # ── Explore/exploit cadence: every 2 steps suggest explore ──
        if step % 2 == 1 and not force_lead_chase and state.lead_pool and getattr(state.lead_pool, "leads", []):
            # Alternate: step 1, 3, 5... encourage explore
            force_lead_chase = True

        # ── Actor proposes 2-3 actions ───────────────────────────────
        proposals = actor_propose(
            state, prev_verdict, must_target_unseen, max_tool_calls,
            force_recovery_mode=force_recovery_mode,
            force_lead_chase=force_lead_chase,
            verbose=verbose,
        )

        # ── Build cost-aware proposals for Judge ─────────────────────
        selected_collections = {c.collection_slug for c in state.selected_chunks if c.collection_slug}
        proposals_for_judge = strip_for_judge(
            proposals, state.tool_calls_used, max_tool_calls,
            selected_collections=selected_collections,
        )

        # ── Deterministic patch: must LEAD_CHASE and none provided → auto-inject ──
        has_lead_chase = any(p.query_origin == QUERY_ORIGIN_LEAD_CHASE for p in proposals_for_judge)
        must_lead_chase = (
            prev_verdict and prev_verdict.material_novelty < 0.4
            and state.lead_pool and getattr(state.lead_pool, "leads", [])
        )
        if must_lead_chase and not has_lead_chase:
            injected = _build_lead_chase_retrieve(state.lead_pool)
            if injected:
                proposals.append(injected)
                proposals_for_judge = strip_for_judge(
                    proposals, state.tool_calls_used, max_tool_calls,
                    selected_collections=selected_collections,
                )

        # ── Enforce must_target_unseen: restrict to satisfying proposals or auto-inject ──
        unseen_satisfying = [i for i, p in enumerate(proposals_for_judge) if p.satisfies_unseen_constraint]
        if must_target_unseen and not unseen_satisfying:
            # Auto-inject RETRIEVE for a new collection (venona/vassiliev)
            for coll in ("venona", "vassiliev"):
                if coll not in selected_collections:
                    injected = NextAction(
                        action=ACTION_RETRIEVE,
                        params={"queries": [state.seed_question], "mode": "hybrid", "top_k": 10,
                                "scope": {"collections": [coll]}},
                        why="Auto-injected: must_target_unseen with no proposal targeting new collection",
                        proposal_intent="explore",
                    )
                    proposals.append(injected)
                    proposals_for_judge = strip_for_judge(
                        proposals, state.tool_calls_used, max_tool_calls,
                        selected_collections=selected_collections,
                    )
                    unseen_satisfying = [i for i, p in enumerate(proposals_for_judge) if p.satisfies_unseen_constraint]
                    break

        # Narrow to unseen-satisfying when must_target_unseen (Judge must pick from these)
        judge_proposals = proposals_for_judge
        judge_unseen_indices = None
        if must_target_unseen and unseen_satisfying:
            judge_proposals = [proposals_for_judge[i] for i in unseen_satisfying]
            judge_unseen_indices = list(range(len(judge_proposals)))  # indices in narrowed list

        # ── Judge selects best action ────────────────────────────────
        findings_summary = state.finding_store.summary_for_judge()
        coverage_stats = build_coverage_stats(state)
        pressure_summary = compute_pressure_summary(state, prev_verdict, must_target_unseen)
        gap_types_summary = compute_gap_types_summary(prev_verdict)
        recent_failures = [a for a, c in state.action_failure_counts.items() if c >= 2]

        selected_idx = judge_select_action(
            seed_question=seed_question,
            directive=directive,
            prev_verdict=prev_verdict,
            findings_summary=findings_summary,
            coverage_stats=coverage_stats,
            actor_proposals=judge_proposals,
            step_number=step,
            pressure_summary=pressure_summary,
            gap_types_summary=gap_types_summary,
            must_target_unseen=must_target_unseen,
            unseen_satisfying_indices=judge_unseen_indices,
            recent_failures=recent_failures if recent_failures else None,
            verbose=verbose,
        )

        # Map back to full proposals list if we narrowed
        if must_target_unseen and unseen_satisfying:
            selected_idx = unseen_satisfying[selected_idx]
        selected_action = proposals[selected_idx]
        state.action_history.append(selected_action)

        # ── Budget check (pre-execution) ─────────────────────────────
        action_cost = TOOL_CALL_UNITS.get(selected_action.action, 0)
        if selected_action.action == ACTION_RETRIEVE and action_cost == 2:
            mode = (selected_action.params or {}).get("mode", "")
            top_k = (selected_action.params or {}).get("top_k", 10)
            if mode == "lexical" and top_k <= 10:
                action_cost = 1  # Micro-retrieve: genuinely cheaper
        if state.tool_calls_used + action_cost > max_tool_calls:
            if verbose:
                print(f"  [ThinkDeeper] Budget exhausted (would use {state.tool_calls_used + action_cost}/{max_tool_calls}); stopping",
                      file=sys.stderr)
            stop_reason = "budget exhausted"
            break

        # ── Execute selected action ──────────────────────────────────
        try:
            new_candidates = execute_action(
                conn, selected_action, state,
                run_scope=run_scope,
                verbose=verbose,
            )
        except Exception as e:
            logger.exception("execute_action failed at step %d: %s", step, e, exc_info=True)
            try:
                conn.rollback()
            except Exception:
                pass
            raise

        # ── Zero-hits recovery: track RETRIEVE 0-raw-candidates ──────
        if selected_action.action == ACTION_RETRIEVE:
            if len(new_candidates) == 0:
                state.consecutive_zero_hits += 1
            else:
                state.consecutive_zero_hits = 0

        # ── Track tool costs ─────────────────────────────────────────
        state.tool_calls_used += action_cost

        # ── Embed gap phrase for rails ───────────────────────────────
        gap_embedding = None
        if prev_verdict and prev_verdict.top_gap_target_phrase:
            try:
                from retrieval.ops import embed_query
                gap_embedding = embed_query(prev_verdict.top_gap_target_phrase)
            except Exception:
                pass

        # ── Rails filter ─────────────────────────────────────────────
        doc_overflow_ids = set(prev_verdict.doc_overflow_request or []) if prev_verdict else set()
        must_include_eids = set(directive.must_include.entity_ids) if directive.must_include.entity_ids else set()

        try:
            filtered, rails_report = apply_rails(
                new_candidates,
                state.selected_chunks,
                state.seed_embedding,
                rails_config,
                conn=conn,
                doc_overflow_ids=doc_overflow_ids if doc_overflow_ids else None,
                must_include_entity_ids=must_include_eids if must_include_eids else None,
                gap_embedding=gap_embedding,
            )
        except Exception as e:
            logger.exception("apply_rails failed at step %d: %s", step, e, exc_info=True)
            try:
                conn.rollback()
            except Exception:
                pass
            raise

        if verbose:
            print(
                f"  [ThinkDeeper] Rails: {len(new_candidates)} candidates -> "
                f"{rails_report.admitted_count} admitted "
                f"(dup={len(rails_report.filtered_dup)}, "
                f"drift={len(rails_report.filtered_drift)}, "
                f"doc_cap={len(rails_report.filtered_doc_cap)})",
                file=sys.stderr,
            )

        # ── Action failure memory ─────────────────────────────────────
        if len(filtered) == 0 and selected_action.action in (ACTION_RETRIEVE, ACTION_EXPAND_SEEDS):
            state.action_failure_counts[selected_action.action] = \
                state.action_failure_counts.get(selected_action.action, 0) + 1
            state.zero_admissible_streak += 1
            if state.zero_admissible_streak >= 3:
                stop_reason = "no admissible candidates after rails filtering for 3 consecutive actions"
                if verbose:
                    print(f"  [ThinkDeeper] {stop_reason}; stopping", file=sys.stderr)
                break
        else:
            state.zero_admissible_streak = 0
            if selected_action.action in (ACTION_RETRIEVE, ACTION_EXPAND_SEEDS):
                state.action_failure_counts.pop(selected_action.action, None)

        # ── Merge filtered candidates ────────────────────────────────
        new_doc_ids = set()
        for c in filtered:
            state.selected_chunks.append(c)
            if c.doc_id not in state.baseline_doc_ids:
                new_doc_ids.add(c.doc_id)

        state.new_doc_count_by_step.append(len(new_doc_ids))
        state.filtered_admitted_by_step.append(len(filtered))

        # ── Frontier metrics ─────────────────────────────────────────
        new_entity_count = 0
        if filtered:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT entity_id FROM entity_mentions WHERE chunk_id = ANY(%s)",
                        ([c.chunk_id for c in filtered],),
                    )
                    step_entity_ids = {r[0] for r in cur.fetchall() if r[0]}
                    new_entity_count = len(step_entity_ids - state.baseline_entity_ids)
            except Exception as e:
                logger.warning("entity_mentions frontier metrics query failed: %s", e)
                try:
                    conn.rollback()
                except Exception:
                    pass
        new_collection_count = len(
            {c.collection_slug for c in filtered if c.collection_slug}
            - {c.collection_slug for c in state.selected_chunks[:-len(filtered)] if c.collection_slug}
        )
        overlap = 0.0
        if selected_action.action == ACTION_RETRIEVE and len(state.action_history) >= 2:
            curr_queries = selected_action.params.get("queries", []) or []
            prev_retrieves = [a for a in state.action_history[:-1] if a.action == ACTION_RETRIEVE]
            if prev_retrieves and curr_queries:
                prev_q = prev_retrieves[-1].params.get("queries", []) or []
                curr_tokens = set()
                for q in curr_queries:
                    curr_tokens.update(str(q).lower().split())
                prev_tokens = set()
                for q in prev_q:
                    prev_tokens.update(str(q).lower().split())
                if curr_tokens or prev_tokens:
                    overlap = len(curr_tokens & prev_tokens) / len(curr_tokens | prev_tokens) if (curr_tokens | prev_tokens) else 0.0
        state.frontier_metrics_by_step.append({
            "new_entity_count": new_entity_count,
            "new_doc_count": len(new_doc_ids),
            "new_collection_count": new_collection_count,
            "overlap_with_prev_queries": overlap,
        })

        # ── Extract leads from newly admitted chunks ──────────────────
        before_count = len(state.selected_chunks) - len(filtered)
        baseline_colls = {c.collection_slug for c in state.selected_chunks[:before_count] if c.collection_slug}
        from retrieval.agent.v9_deep_leads import extract_leads
        state.lead_pool = extract_leads(
            conn,
            filtered,
            state.baseline_entity_ids,
            state.baseline_doc_ids,
            step,
            prev_lead_pool=getattr(state, "lead_pool", None),
            baseline_collection_slugs=baseline_colls,
        )

        # ── Judge scores delta ───────────────────────────────────────
        evidence_sample = build_evidence_sample(filtered, state)
        coverage_stats = build_coverage_stats(state)

        verdict = judge_score_delta(
            seed_question=seed_question,
            directive=directive,
            prev_verdict=prev_verdict,
            findings_summary=state.finding_store.summary_for_judge(),
            coverage_stats=coverage_stats,
            new_evidence_sample=evidence_sample,
            step_number=step,
            verbose=verbose,
        )

        # ── Validate verdict + update FindingStore ───────────────────
        valid_chunk_ids = state.selected_chunk_ids
        verdict = validate_verdict(verdict, valid_chunk_ids)

        new_findings_added = 0
        if state.finding_store and verdict.new_findings:
            new_findings_added = state.finding_store.add_from_judge_findings(
                verdict.new_findings, step, valid_chunk_ids,
            )

        state.new_findings_count_by_step.append(new_findings_added)

        # ── Record ───────────────────────────────────────────────────
        state.verdict_history.append(verdict)

        # Persist step trace to DB
        if td_run_id is not None:
            _persist_td_step(
                conn, td_run_id, step,
                proposals=proposals,
                selected_action=selected_action,
                candidates_count=len(new_candidates),
                rails_report=rails_report,
                verdict=verdict,
                selected_chunk_ids=sorted(state.selected_chunk_ids),
                new_findings_count=new_findings_added,
            )

        if verbose:
            print(
                f"  [ThinkDeeper] Step {step} done: "
                f"+{len(filtered)} chunks, +{len(new_doc_ids)} docs, "
                f"+{new_findings_added} findings, "
                f"total_tool_calls={state.tool_calls_used}/{max_tool_calls}",
                file=sys.stderr,
            )

        # ── Stop check ───────────────────────────────────────────────
        if should_stop(verdict, state, rails_config):
            stop_reason = verdict.stop_reason or _infer_stop_reason(verdict, state, rails_config)
            if verbose:
                print(f"  [ThinkDeeper] Stopping: {stop_reason}", file=sys.stderr)
            break

        prev_verdict = verdict

    # 4. Build result
    elapsed_ms = (time.time() - t0) * 1000

    # Build novelty report
    new_doc_ids_all = state.selected_doc_ids - state.baseline_doc_ids
    remaining_gaps = state.verdict_history[-1].top_gaps_as_strings if state.verdict_history else []
    novelty_report = NoveltyReport(
        new_docs=[{"doc_id": did} for did in sorted(new_doc_ids_all)],
        remaining_gaps=remaining_gaps,
        what_changed=_summarize_changes(state),
    )

    # Generate LLM-suggested queries from gaps (user can click to explore)
    if remaining_gaps:
        novelty_report.suggested_queries = _generate_suggested_queries(
            state.seed_question, remaining_gaps, state,
        )

    # Always synthesize a narrative from findings — never surface defeatist "not possible"
    narrative = _build_think_deeper_narrative(state, novelty_report, stop_reason)

    # Sanitize stop_reason for storage (downstream may display it)
    defeatist = ("not possible", "nothing more", "nothing new", "cannot find", "impossible")
    safe_stop_reason = stop_reason
    if stop_reason and any(d in stop_reason.lower() for d in defeatist):
        safe_stop_reason = "Evidence search complete; findings summarized above"

    result = ThinkDeeperResult(
        narrative=narrative,
        novelty_report=novelty_report,
        stop_reason=safe_stop_reason,
        verdict_history=state.verdict_history,
        finding_store_entries=state.finding_store.entries if state.finding_store else [],
        steps_executed=state.step + 1,
        tool_calls_used=state.tool_calls_used,
        elapsed_ms=elapsed_ms,
        selected_chunks=state.selected_chunks,
    )

    # Finalize persistence
    if td_run_id is not None:
        _finalize_td_run(conn, td_run_id, result)

    if verbose:
        print(
            f"  [ThinkDeeper] Complete: {result.steps_executed} steps, "
            f"{result.tool_calls_used} tool calls, "
            f"{len(new_doc_ids_all)} new docs, "
            f"stop_reason={stop_reason}, "
            f"elapsed={elapsed_ms:.0f}ms",
            file=sys.stderr,
        )

    return result


# ── Helper formatters ────────────────────────────────────────────────────────

def _build_state_summary(state: DeepState) -> str:
    """Summarize current state for Actor prompt."""
    lines = [
        f"Step: {state.step}",
        f"Selected chunks: {len(state.selected_chunks)} "
        f"({len(state.selected_chunk_ids - state.baseline_chunk_ids)} new vs baseline)",
        f"Documents: {len(state.selected_doc_ids)} "
        f"({len(state.selected_doc_ids - state.baseline_doc_ids)} new)",
        f"Findings: {state.finding_store.total_count() if state.finding_store else 0}",
        f"Tool calls used: {state.tool_calls_used}",
    ]

    # Recent actions
    if state.action_history:
        lines.append("\nRecent actions:")
        for a in state.action_history[-3:]:
            lines.append(f"  - {a.action}: {json.dumps(a.params)[:100]}")

    # Collections covered
    colls = {c.collection_slug for c in state.selected_chunks if c.collection_slug}
    if colls:
        lines.append(f"\nCollections: {', '.join(sorted(colls))}")

    return "\n".join(lines)


def _build_directive_summary(directive: ResearchDirective) -> str:
    """Summarize directive for prompts."""
    parts = [f"Primary question: {directive.primary_question}"]
    if directive.user_directive:
        parts.append(f"User follow-up: {directive.user_directive}")
    if directive.must_answer:
        parts.append(f"Must answer: {', '.join(directive.must_answer)}")
    weights = directive.weights
    if weights.coverage != 1.0 or weights.novelty != 1.0:
        parts.append(
            f"Priority weights: coverage={weights.coverage}, novelty={weights.novelty}, "
            f"support={weights.support}, verification={weights.verification}"
        )
    return "\n".join(parts)


def _generate_suggested_queries(
    seed_question: str,
    remaining_gaps: List[str],
    state: DeepState,
    *,
    max_queries: int = 4,
) -> List[str]:
    """Generate natural-language query suggestions from remaining gaps via LLM."""
    if not remaining_gaps:
        return []
    findings_summary = ""
    if state.finding_store and state.finding_store.entries:
        texts = [e.text[:80] for e in state.finding_store.entries[:5] if e.text]
        if texts:
            findings_summary = "Key findings so far: " + "; ".join(texts)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("V9_DEEP_ACTOR_MODEL", "gpt-4.1-mini-2025-04-14")
        prompt = f"""You suggest follow-up search queries for a historical research system.

Original question: {seed_question}
{findings_summary}

Research gaps identified (areas not yet fully answered):
{chr(10).join(f"  - {g}" for g in remaining_gaps[:5])}

Generate 3-4 short, natural-language search queries a user could run to explore these gaps.
Each query should be a complete question or search phrase (e.g. "Who was Berg's role in Soviet intelligence?" or "Berg connections to Major Dahl").
Return JSON: {{"queries": ["query1", "query2", ...]}}"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content:
            out = json.loads(content)
            queries = out.get("queries", [])
            return [str(q).strip() for q in queries if q][:max_queries]
    except Exception as e:
        logger.warning("_generate_suggested_queries failed: %s", e)
    return []


def _chunk_to_label(c: CandidateChunk) -> str:
    """Build human-readable citation label from chunk (e.g. Vassiliev p42)."""
    source = (c.collection_slug or getattr(c, "source_label", "") or "").replace("_", " ").title()
    page = ""
    if c.page:
        m = re.search(r"(\d+)", str(c.page))
        if m:
            page = f" p{m.group(1)}"
    return f"{source}{page}".strip() or f"Chunk {c.chunk_id}"


def _build_think_deeper_narrative(
    state: DeepState,
    novelty_report: NoveltyReport,
    stop_reason: str,
) -> str:
    """Build a user-facing narrative from findings. Never surfaces technical jargon."""
    parts: List[str] = []
    chunk_by_id = {c.chunk_id: c for c in state.selected_chunks}

    def _cite(chunk_ids: List[int]) -> str:
        labels = []
        seen = set()
        for cid in chunk_ids[:3]:
            if cid in chunk_by_id and cid not in seen:
                seen.add(cid)
                labels.append(_chunk_to_label(chunk_by_id[cid]))
        if not labels:
            return ""
        return " [" + ", ".join(labels) + "]"

    # New findings: each with document citation
    entries = state.finding_store.entries if state.finding_store else []
    new_findings = [e for e in entries if e.finding_type != "baseline"]
    if new_findings:
        parts.append("**New findings from this search:**")
        for e in new_findings[:10]:
            cite = _cite(e.supporting_chunk_ids) if e.supporting_chunk_ids else ""
            parts.append(f"  • {e.text}{cite}")
        parts.append("")

    # What changed — user-friendly phrasing
    if novelty_report.what_changed and novelty_report.what_changed != "No significant changes":
        parts.append(f"**What changed:** {novelty_report.what_changed}")
        parts.append("")

    # LLM-generated suggested queries (clickable in UI)
    if novelty_report.suggested_queries:
        parts.append("**Suggested queries to explore:**")
        for q in novelty_report.suggested_queries[:5]:
            parts.append(f"  • {q}")
        parts.append("")

    # When no new findings / unsatisfactory: list useful documents with links
    technical_stop = (
        "no admissible", "rails filtering", "zero admissible", "consecutive actions",
        "budget exhausted", "max_steps", "not possible", "nothing more", "nothing new",
        "cannot find", "impossible",
    )
    is_technical = stop_reason and any(t in stop_reason.lower() for t in technical_stop)

    if not parts:
        new_docs = len(state.selected_doc_ids - state.baseline_doc_ids)
        new_chunks = len(state.selected_chunk_ids - state.baseline_chunk_ids)
        if new_docs or new_chunks:
            parts.append(
                f"We explored {new_docs} new document(s) and {new_chunks} new evidence chunk(s). "
                "Review the evidence above for relevant details."
            )
        else:
            parts.append(
                "This search round reinforced the evidence base. "
                "Try a different angle or more specific query for additional findings."
            )

    # When no new findings but we have evidence: list documents that may be useful (with links)
    if not new_findings and state.selected_chunks:
        parts.append("**Documents that may be useful:**")
        seen_labels: Set[str] = set()
        for c in state.selected_chunks[:8]:
            label = _chunk_to_label(c)
            if label not in seen_labels and "Chunk" not in label:
                seen_labels.add(label)
                parts.append(f"  • [{label}]")
        parts.append("")

    # Never append technical stop_reason
    if stop_reason and not is_technical:
        parts.append(f"\n*({stop_reason})*")
    elif is_technical or (not new_findings and state.selected_chunks):
        parts.append("\n*(Evidence search complete; see documents above for useful references.)*")

    return "\n".join(parts).strip()


def _build_prev_verdict_summary(verdict: JudgeVerdict) -> str:
    """Summarize previous verdict for Actor (includes typed gaps)."""
    lines = [
        f"Scores: answeredness={verdict.answeredness:.2f}, "
        f"material_novelty={verdict.material_novelty:.2f}, "
        f"confidence={verdict.confidence:.2f}",
    ]
    if verdict.top_gaps:
        gap_strs = [f"[{g.type}] {g.target} (priority={g.priority:.1f})" for g in verdict.top_gaps]
        lines.append(f"Top gaps: {'; '.join(gap_strs)}")
    if verdict.stop_recommendation:
        lines.append(f"Judge recommended stop: {verdict.stop_reason or 'no reason given'}")
    lines.append(
        f"EV next step: retrieve={verdict.ev_next_step_retrieve:.2f}, "
        f"expand={verdict.ev_next_step_expand:.2f}"
    )
    return "\n".join(lines)


def _infer_stop_reason(verdict: JudgeVerdict, state: DeepState, config: RailsConfig) -> str:
    """Infer a user-readable stop reason from state."""
    if state.tool_calls_used >= config.max_tool_calls:
        return "budget exhausted"
    if verdict.confidence > 0.8 and verdict.stop_recommendation:
        return f"high confidence ({verdict.confidence:.2f}) with Judge stop recommendation"
    if verdict.ev_next_step < 0.25:
        return f"low expected value for next step ({verdict.ev_next_step:.2f})"
    if len(state.filtered_admitted_by_step) >= 2 and all(c < 2 for c in state.filtered_admitted_by_step[-2:]):
        return "minimal evidence growth over last 2 steps"
    return "investigation complete"


def _summarize_changes(state: DeepState) -> str:
    """Generate a what_changed summary."""
    new_docs = len(state.selected_doc_ids - state.baseline_doc_ids)
    new_chunks = len(state.selected_chunk_ids - state.baseline_chunk_ids)
    total_findings = state.finding_store.total_count() if state.finding_store else 0
    baseline_findings = sum(
        1 for e in (state.finding_store.entries if state.finding_store else [])
        if e.finding_type == "baseline"
    )
    new_findings = total_findings - baseline_findings

    parts = []
    if new_docs > 0:
        parts.append(f"Added {new_docs} new documents")
    if new_chunks > 0:
        parts.append(f"added {new_chunks} new evidence chunks")
    if new_findings > 0:
        parts.append(f"discovered {new_findings} new findings")

    return "; ".join(parts) if parts else "No significant changes"
