"""
V10 Runner — Agentic-first identity-aware pipeline.

V10.2 architecture:
  A: Query interpretation (SpanLattice -> LLM span selection)
  D: LLM-driven investigation loop (agent calls search/fetch/extract tools)
     - Post-search enrichment (LLM + deterministic extraction)
     - Live lexicon updates (alias resolution, entity backfill)
     - Finalization with grounding + verification (bounded retries)

Stages B/C (deterministic bulk prefetch) are bypassed by default
(V10_AGENTIC_FIRST=1). The LLM drives all retrieval from round 1.

Uses OpenAI Structured Outputs for deterministic LLM output parsing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    AliasScopedBoost,
    AliasContext,
    CatalogHitV10,
    ChunkMentionsV10,
    EntityBoost,
    LexiconV10,
    MatchProvenance,
    ResolutionPlanV10,
    ResolvedAlias,
    SpanCandidate,
    SpanEntry,
    SpanLatticeV10,
    SpanSelection,
)
from retrieval.agent.v10_extract import (
    _collect_alias_surfaces_from_text,
    extract_chunk_mentions_v10,
    extract_chunk_mentions_v10_deterministic,
    extract_mentions_dispatched,
)
from retrieval.agent.v10_lexicon import (
    backfill_alias_namespace,
    build_entity_forms,
    build_lexicon_from_lattice,
    load_referent_rules_into_lexicon,
    serialize_lexicon,
    update_from_mentions,
)
from retrieval.agent.v10_prompts import (
    V10_ALIAS_RESOLUTION_PROMPT,
    V10_GLOBAL_RETRIEVAL_PROMPT,
    V10_MAX_INVESTIGATION_ROUNDS,
    V10_MAX_TOOL_CALLS,
    V10_MAX_WORKSPACE_CHUNKS,
    V10_MODEL,
    V10_OUTPUT_SCHEMA,
    V10_RESPONSE_FORMAT,
    V10_SPAN_SELECTION_PROMPT,
    V10_SPAN_SELECTION_SCHEMA,
    V10_SYNTHESIS_PROMPT,
    V10_SYSTEM_PROMPT,
    V10_TEMPERATURE,
    V10_TOOL_TURN_MAX_TOKENS,
    V10_TOOLS_DEF,
    V10_SYNTHESIS_MAX_TOKENS,
)
from retrieval.agent.v10_page_bridge import get_index_revision
from retrieval.agent.v10_pem_lane import (
    PemLaneResult,
    build_chunk_pem_annotation,
    pem_lane_seed_chunks,
)
from retrieval.agent.v10_search import search_chunks_v10
from retrieval.agent.v10_spans import spot_query_spans_v10
from retrieval.agent.v9_types import (
    ScopeFilter,
    V9Result,
    V9Synthesis,
    V9Claim,
    GroundedClaim,
    InvestigationStep,
    WorkspaceChunk,
    ResearchWorkspace,
    EvidenceBullet,
    EvidenceMemoryView,
    InvestigationState,
)
from retrieval.agent.v9_workspace import (
    merge_fetched_chunks,
    link_chunks_to_entities,
    merge_evidence_summary_update,
    apply_pin_suggestions,
    build_chunk_doc_map,
    enforce_pin_cap,
)
from retrieval.agent.v9_summarize import summarize_delta_chunks
from retrieval.agent.v9_context import select_evidence_memory_view
from retrieval.ops import SearchFilters

logger = logging.getLogger(__name__)

# Feature flags
V10_AGENTIC_FIRST = os.getenv("V10_AGENTIC_FIRST", "1") == "1"


def _check_and_sync_index_revision(conn, lexicon: LexiconV10) -> None:
    """Check page_entity_mentions revision and invalidate stale permissions.

    Called on lexicon init/rehydrate and before first tool use each session.
    If the DB revision differs from lexicon.alias_index_revision, all alias
    permissions are cleared (they were granted against a stale index).
    """
    try:
        db_revision = get_index_revision(conn)
        if lexicon.alias_index_revision is None:
            # First use: just record the current revision
            lexicon.alias_index_revision = db_revision
        elif lexicon.alias_index_revision != db_revision:
            logger.info(
                "Index revision mismatch: lexicon=%s, db=%s — clearing alias permissions",
                lexicon.alias_index_revision, db_revision,
            )
            lexicon.alias_permissions.clear()
            lexicon.alias_index_revision = db_revision
    except Exception as e:
        logger.debug("_check_and_sync_index_revision failed: %s", e)
V10_EXPLORE_EXPLOIT = os.getenv("V10_EXPLORE_EXPLOIT", "1") == "1"

# Controller constants
MAX_VERIFY_RETRIES = 2
MAX_SEARCHES_PER_ROUND = 3
MAX_CHUNKS_PER_DOC = 8  # cap extraction per doc unless exploit mode


# =============================================================================
# Progress tracking (multi-signal stall detection)
# =============================================================================

@dataclass
class ProgressDelta:
    """Multi-signal progress measurement for stall detection."""
    new_entities: int = 0
    new_mappings: int = 0
    ambiguity_reduced: int = 0
    new_support_chunks: int = 0
    new_collections_covered: int = 0

    @property
    def score(self) -> float:
        return (self.new_entities * 3.0
                + self.new_mappings * 2.0
                + self.ambiguity_reduced * 1.5
                + self.new_support_chunks * 0.5
                + self.new_collections_covered * 2.0)

    @property
    def is_stalled(self) -> bool:
        return self.score < 0.5


def _compute_progress_delta(
    lexicon: LexiconV10,
    prev_entity_count: int,
    prev_hyp_count: int,
    prev_support_total: int,
    prev_ambiguity_total: int,
) -> ProgressDelta:
    """Compute progress delta from lexicon state diffs."""
    entity_count = len(lexicon.entities_in_play)
    hyp_count = len(lexicon.alias_mapping_hypotheses)
    support_total = sum(
        len(info.get("evidence_chunk_ids", []))
        for info in lexicon.entities_in_play.values()
    )
    ambiguity_total = sum(
        len(h.candidates) for h in lexicon.alias_mapping_hypotheses.values()
        if h.status in ("unresolved", "ambiguous")
    )

    return ProgressDelta(
        new_entities=max(0, entity_count - prev_entity_count),
        new_mappings=max(0, hyp_count - prev_hyp_count),
        ambiguity_reduced=max(0, prev_ambiguity_total - ambiguity_total),
        new_support_chunks=max(0, support_total - prev_support_total),
    )


def _snapshot_lexicon_counts(lexicon: LexiconV10) -> Tuple[int, int, int, int]:
    """Snapshot lexicon counts for progress delta computation."""
    entity_count = len(lexicon.entities_in_play)
    hyp_count = len(lexicon.alias_mapping_hypotheses)
    support_total = sum(
        len(info.get("evidence_chunk_ids", []))
        for info in lexicon.entities_in_play.values()
    )
    ambiguity_total = sum(
        len(h.candidates) for h in lexicon.alias_mapping_hypotheses.values()
        if h.status in ("unresolved", "ambiguous")
    )
    return entity_count, hyp_count, support_total, ambiguity_total


# =============================================================================
# V10 Result type
# =============================================================================

@dataclass
class V10Result:
    """Complete V10 query result."""
    narrative: str = ""
    claims: List[GroundedClaim] = field(default_factory=list)
    lattice: Optional[SpanLatticeV10] = None
    lexicon: Optional[LexiconV10] = None
    plan: Optional[ResolutionPlanV10] = None
    chunk_mentions: Dict[int, ChunkMentionsV10] = field(default_factory=dict)
    investigation_trace: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    chunks_fetched: Dict[int, WorkspaceChunk] = field(default_factory=dict)
    unresolved_aliases: List[Dict[str, Any]] = field(default_factory=list)

    def to_v9_result(self) -> V9Result:
        """Convert to V9Result for API compatibility."""
        from retrieval.agent.v9_types import ResearchWorkspace
        ws = ResearchWorkspace(question=self.narrative or "")
        ws.fulltext_chunks = list(self.chunks_fetched.values())
        return V9Result(
            narrative=self.narrative,
            claims=self.claims,
            workspace=ws,
            investigation_trace=[
                InvestigationStep(
                    step_idx=i,
                    action=t.get("action", ""),
                    rationale=t.get("rationale", t.get("summary", "")),
                    inputs=t.get("inputs", {}),
                    outputs_summary=t.get("outputs_summary", t.get("summary", "")),
                )
                for i, t in enumerate(self.investigation_trace)
            ],
        )


# =============================================================================
# Scope conversion helper
# =============================================================================

def _scope_to_filters(scope: Optional[ScopeFilter]) -> SearchFilters:
    """Convert V9 ScopeFilter to ops.SearchFilters."""
    if scope is None:
        return SearchFilters()
    return SearchFilters(
        collection_slugs=scope.collections,
        document_ids=scope.document_ids,
        date_from=scope.date_from,
        date_to=scope.date_to,
    )


# =============================================================================
# OpenAI client helper
# =============================================================================

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def _call_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    response_format: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    temperature: float = V10_TEMPERATURE,
    max_tokens: int = 4096,
    verbose: bool = True,
) -> Any:
    """Call OpenAI chat completions with 429 retry and visible output."""
    import sys

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate" in err_str or "429" in err_str

            if is_rate_limit and attempt < max_retries:
                retry_after = getattr(e, "retry_after", None)
                wait_s = float(retry_after) if retry_after else 2.0 * (2 ** attempt)
                if verbose:
                    print(
                        f"[V10] Rate limit (429) — attempt {attempt + 1}/{max_retries + 1}, "
                        f"retrying in {wait_s:.1f}s...",
                        file=sys.stderr,
                        flush=True,
                    )
                logger.warning("LLM 429 (attempt %d/%d), retry in %.1fs: %s", attempt + 1, max_retries + 1, wait_s, e)
                time.sleep(wait_s)
                continue

            logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
            if verbose:
                print(f"[V10] LLM call failed: {e}", file=sys.stderr, flush=True)
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after retries")


# =============================================================================
# Stage A: Query interpretation
# =============================================================================

def _run_stage_a(
    conn,
    client: OpenAI,
    model: str,
    question: str,
    scope: Optional[ScopeFilter],
    verbose: bool,
) -> Tuple[SpanLatticeV10, SpanSelection, LexiconV10, ResolutionPlanV10]:
    """Stage A: Spot spans, present to LLM, get span selection."""
    if verbose:
        print("[V10] Stage A: Query interpretation...", flush=True)

    # 1. Enumerate spans (pass scope_hint for codename_alias scoping)
    scope_collections = None
    if scope and hasattr(scope, 'collection_slugs') and scope.collection_slugs:
        scope_collections = scope.collection_slugs
    lattice = spot_query_spans_v10(conn, question, scope_hint=scope_collections)

    if verbose:
        print(f"[V10]   SpanLattice: {len(lattice.spans)} spans", flush=True)

    # 2. If no spans with candidates, return empty selection
    if not lattice.spans:
        selection = SpanSelection()
        lexicon = LexiconV10()
        plan = ResolutionPlanV10()
        return lattice, selection, lexicon, plan

    # 3. Present lattice to LLM for span selection
    lattice_json = json.dumps(lattice.to_dict(), indent=2)
    prompt = V10_SPAN_SELECTION_PROMPT.format(lattice_json=lattice_json)

    messages = [
        {"role": "system", "content": V10_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = _call_llm(
        client, model, messages,
        response_format={
            "type": "json_schema",
            "json_schema": V10_SPAN_SELECTION_SCHEMA,
        },
        max_tokens=2048,
        verbose=verbose,
    )

    # 4. Parse span selection
    content = response.choices[0].message.content
    try:
        selection_data = json.loads(content)
        selection = SpanSelection.from_dict(selection_data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to parse span selection: %s", e)
        # Fall back: select all non-overlapping spans
        selection = SpanSelection(
            chosen_span_ids=[s.span_id for s in lattice.spans if not s.dominates],
        )

    if verbose:
        print(f"[V10]   Selected spans: {selection.chosen_span_ids}", flush=True)
        print(f"[V10]   Entity hypotheses: {len(selection.entity_hypotheses)}", flush=True)
        print(f"[V10]   Alias spans: {len(selection.alias_spans)}", flush=True)

    # 5. Build initial lexicon from lattice + selection
    lexicon = build_lexicon_from_lattice(conn, lattice, selection)

    # 5a. Promote entity_hypotheses into entities_in_play.
    # The LLM may choose long composite spans whose candidates are empty,
    # while the actual entity candidates live on shorter suppressed spans.
    # entity_hypotheses correctly identify the entities — promote them so
    # PEM lane, backfill, and boosts all work from round 1.
    if selection.entity_hypotheses:
        eid_to_name: Dict[int, str] = {}
        for span in lattice.spans:
            for cand in span.candidates:
                if cand.entity_id not in eid_to_name:
                    eid_to_name[cand.entity_id] = cand.canonical_name
        for hyp in selection.entity_hypotheses:
            eid = hyp.get("entity_id")
            if eid and eid not in lexicon.entities_in_play:
                canonical = eid_to_name.get(eid, "")
                if not canonical:
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT canonical_name FROM entities WHERE id = %s",
                                (eid,),
                            )
                            row = cur.fetchone()
                            if row:
                                canonical = row[0]
                    except Exception:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                if canonical:
                    lexicon.register_entity(
                        entity_id=eid,
                        canonical_name=canonical,
                    )
        if verbose and lexicon.entities_in_play:
            promoted = [
                f"{eid}:{info.get('canonical_name','?')}"
                for eid, info in lexicon.entities_in_play.items()
            ]
            print(
                f"[V10]   Promoted {len(promoted)} entity hypotheses → entities_in_play: "
                f"{', '.join(promoted[:6])}",
                flush=True,
            )

    # 5b. Sync index revision with DB (clears stale permissions if needed)
    _check_and_sync_index_revision(conn, lexicon)

    # Load referent rules for any alias-scoped entities
    alias_doc_keys = _collect_alias_doc_keys_from_selection(selection, lattice)
    if alias_doc_keys:
        load_referent_rules_into_lexicon(conn, lexicon, alias_doc_keys)

    # 6. Commit sticky referents for unambiguous spans (Phase 4)
    chosen_set = set(selection.chosen_span_ids)
    for span in lattice.spans:
        if span.span_id not in chosen_set:
            continue
        if span.resolution_status == "resolved" and len(span.candidates) == 1:
            c = span.candidates[0]
            lexicon.set_resolved_referent(
                norm_key=span.norm_key,
                start=span.start,
                end=span.end,
                entity_id=c.entity_id,
                status="confirmed",
            )

    # 7. Build resolution plan
    plan = ResolutionPlanV10(
        selected_spans=[
            s for s in lattice.spans if s.span_id in chosen_set
        ],
        span_selection=selection,
        entity_hypotheses=selection.entity_hypotheses,
    )

    return lattice, selection, lexicon, plan


def _collect_alias_doc_keys_from_selection(
    selection: SpanSelection,
    lattice: SpanLatticeV10,
) -> List[Tuple[str, str, int]]:
    """Collect (collection, alias, document_id) keys from selection for referent rules."""
    # At Stage A we don't have document_ids yet — rules will be loaded later
    return []


# =============================================================================
# Stage B: Alias resolution gate
# =============================================================================

def _needs_alias_resolution(selection: SpanSelection) -> bool:
    """Check if any selected spans require alias resolution."""
    return any(
        a.get("activate_alias_resolution", False)
        for a in selection.alias_spans
    )


# =============================================================================
# Stage C: Retrieval
# =============================================================================

def _run_alias_resolution(
    conn,
    client: OpenAI,
    model: str,
    question: str,
    selection: SpanSelection,
    lattice: SpanLatticeV10,
    lexicon: LexiconV10,
    plan: ResolutionPlanV10,
    verbose: bool,
) -> Tuple[List[CatalogHitV10], Dict[int, ChunkMentionsV10]]:
    """Stage C1: Alias-resolution retrieval (scoped to Venona/Vassiliev)."""
    if verbose:
        print("[V10] Stage C1: Alias resolution retrieval...", flush=True)

    # Build alias boosts from alias spans
    alias_boosts: List[AliasScopedBoost] = []
    alias_span_ids = {a["span_id"] for a in selection.alias_spans if a.get("activate_alias_resolution")}

    for span in lattice.spans:
        if span.span_id in alias_span_ids:
            for coll in ALIAS_SCOPED_COLLECTIONS:
                # Check if any candidate is alias-scoped to this collection
                has_alias_cand = any(
                    coll in c.valid_collections
                    for c in span.candidates
                    if c.valid_collections != ["*"]
                )
                if has_alias_cand or span.candidates:
                    alias_boosts.append(AliasScopedBoost(
                        collection_slug=coll,
                        alias_text=span.text,
                        weight=1.2,
                    ))

    # Search within Venona/Vassiliev
    alias_scope = SearchFilters(collection_slugs=list(ALIAS_SCOPED_COLLECTIONS))
    hits, provenance = search_chunks_v10(
        conn,
        question,
        scope=alias_scope,
        alias_boosts_scoped=alias_boosts,
        k=30,
    )

    if verbose:
        print(f"[V10]   Alias resolution: {len(hits)} hits", flush=True)

    # Extract mentions from hits
    chunk_mentions: Dict[int, ChunkMentionsV10] = {}
    for hit in hits:
        if hit.snippet:
            cm = extract_chunk_mentions_v10(
                conn,
                chunk_id=hit.chunk_id,
                collection_slug=hit.collection_slug or "",
                document_id=hit.document_id or 0,
                page_no=hit.page_no,
                text=hit.snippet,
                lexicon=lexicon,
            )
            chunk_mentions[hit.chunk_id] = cm
            # Update lexicon from mentions
            update_from_mentions(conn, lexicon, cm)

    return hits, chunk_mentions


def _run_global_retrieval(
    conn,
    question: str,
    lexicon: LexiconV10,
    scope: Optional[ScopeFilter],
    verbose: bool,
) -> Tuple[List[CatalogHitV10], Dict[int, ChunkMentionsV10]]:
    """Stage C2: Global retrieval with entity_id boosts."""
    if verbose:
        print("[V10] Stage C2: Global retrieval...", flush=True)

    # Build entity boosts from lexicon
    entity_boosts: List[EntityBoost] = []
    scope_collections = scope.collections if scope else None

    for entity_id, info in lexicon.entities_in_play.items():
        forms = build_entity_forms(conn, entity_id, scope=scope_collections)
        if forms:
            entity_boosts.append(EntityBoost(
                entity_id=entity_id,
                forms=forms,
                weight=1.0,
            ))

    filters = _scope_to_filters(scope)
    hits, provenance = search_chunks_v10(
        conn,
        question,
        scope=filters,
        entity_boosts=entity_boosts,
        k=50,
    )

    if verbose:
        print(f"[V10]   Global retrieval: {len(hits)} hits", flush=True)

    # Extract mentions from hits
    chunk_mentions: Dict[int, ChunkMentionsV10] = {}
    for hit in hits:
        if hit.snippet:
            cm = extract_chunk_mentions_v10(
                conn,
                chunk_id=hit.chunk_id,
                collection_slug=hit.collection_slug or "",
                document_id=hit.document_id or 0,
                page_no=hit.page_no,
                text=hit.snippet,
                lexicon=lexicon,
            )
            chunk_mentions[hit.chunk_id] = cm
            update_from_mentions(conn, lexicon, cm)

    return hits, chunk_mentions


# =============================================================================
# Stage D: Synthesis (LLM-driven investigation loop)
# =============================================================================

def _build_identity_summary(lexicon: LexiconV10) -> str:
    """Build a human-readable summary of the current identity state."""
    lines = []

    # Entities
    lines.append("### Entities in play")
    for eid, info in lexicon.entities_in_play.items():
        canonical = info.get("canonical_name", "?")
        n_evidence = len(info.get("evidence_chunk_ids", []))
        lines.append(f"- **{canonical}** (id={eid}, evidence_chunks={n_evidence})")

    # Alias hypotheses
    ctx_hyps = [h for h in lexicon.alias_mapping_hypotheses.values() if h.is_contextual]
    gen_hyps = [h for h in lexicon.alias_mapping_hypotheses.values() if not h.is_contextual]

    if ctx_hyps:
        lines.append("\n### Contextual alias mappings (document-specific)")
        for h in ctx_hyps:
            cands = ", ".join(c.canonical_name for c in h.candidates[:3])
            lines.append(
                f"- **{h.alias_text}** in doc {h.document_id}: "
                f"{h.status} -> [{cands}] (conf={h.confidence:.2f})"
            )

    if gen_hyps:
        lines.append("\n### General alias mappings (collection-wide)")
        for h in gen_hyps:
            cands = ", ".join(c.canonical_name for c in h.candidates[:3])
            lines.append(
                f"- **{h.alias_text}** in {h.collection_slug}: "
                f"{h.status} -> [{cands}] (conf={h.confidence:.2f})"
            )

    return "\n".join(lines)


def _run_synthesis(
    conn,
    client: OpenAI,
    model: str,
    question: str,
    lexicon: LexiconV10,
    all_hits: List[CatalogHitV10],
    all_mentions: Dict[int, ChunkMentionsV10],
    chunks_fetched: Dict[int, WorkspaceChunk],
    scope: Optional[ScopeFilter],
    verbose: bool,
    progress_callback: Optional[Callable] = None,
    lattice_summary: Optional[Dict[str, Any]] = None,
    initial_boosts: Optional[Dict[str, Any]] = None,
    pem_seed: Optional[PemLaneResult] = None,
    max_investigation_rounds: Optional[int] = None,
) -> Tuple[str, List[GroundedClaim], List[Dict[str, Any]], ResearchWorkspace]:
    """Stage D+E: LLM-driven investigation loop with tool calls, live lexicon,
    exploration/exploitation controller, and finalization with verification.

    Fixed-index messages:
      [0] system prompt
      [1] user (investigation framing + question + lattice + boosts)
      [2] assistant (Current Lexicon State — overwritten each round)
      [3] assistant (Strategy Hint — overwritten each round, V10.3)
      [4] assistant (Evidence Memory — overwritten after each fetch/search round)

    Returns (narrative, claims, trace, workspace).
    """
    if verbose:
        print("[V10.2] Stage D: LLM-driven investigation...", flush=True)

    # --- Build initial investigation prompt ---
    if lattice_summary:
        # V10.2 agentic-first: no evidence dump, lattice + boosts instead
        lattice_json_str = json.dumps(lattice_summary, indent=2, default=str)
        boosts_json_str = json.dumps(initial_boosts or {}, indent=2, default=str)
        user_content = (
            f"## Question\n{question}\n\n"
            f"## Span Lattice Summary\n"
            f"The following structured spans were detected in your query. "
            f"Use them as your starting anchor — avoid drifting from these identities.\n"
            f"```json\n{lattice_json_str}\n```\n\n"
            f"## Initial Recommended Boosts\n"
            f"Use these in your first search_v10 call:\n"
            f"```json\n{boosts_json_str}\n```\n\n"
            f"## Investigation Instructions\n"
            f"You have NO pre-fetched evidence. Drive all retrieval yourself using search_v10.\n"
            f"1. Start by searching with the initial boosts above.\n"
            f"2. After each search, review enrichment results and updated lexicon.\n"
            f"3. Use recommended_boosts from search results for follow-up searches.\n"
            f"4. When you have relevant chunk_ids (from search results or alias_index_lookup_v10), "
            f"call fetch_chunks to read full text before synthesizing.\n"
            f"5. When ready, set final=true to synthesize your answer with cited claims.\n"
            f"6. Your claims will be grounded and verified — if verification fails, "
            f"you'll get actionable errors (max {MAX_VERIFY_RETRIES} retries).\n"
        )
    else:
        # Legacy path: pre-fetched evidence
        collections_seen = set(h.collection_slug for h in all_hits if h.collection_slug)
        synthesis_prompt = V10_SYNTHESIS_PROMPT.format(
            n_chunks=len(all_hits),
            n_collections=len(collections_seen),
            identity_summary="(see Current Lexicon State below)",
        )
        evidence_lines = []
        for hit in all_hits[:30]:
            mention_info = ""
            cm = all_mentions.get(hit.chunk_id)
            if cm and cm.mentions:
                mention_info = f" [mentions: {', '.join(m.surface for m in cm.mentions[:3])}]"
            evidence_lines.append(
                f"chunk_{hit.chunk_id} ({hit.collection_slug or '?'}, doc={hit.document_id}, "
                f"p{hit.page_no or '?'}): {hit.snippet[:200]}...{mention_info}"
            )
        evidence_text = "\n".join(evidence_lines)
        user_content = (
            f"## Question\n{question}\n\n"
            f"## Evidence ({len(all_hits)} chunks retrieved)\n{evidence_text}\n\n"
            f"{synthesis_prompt}"
        )

    # Build initial lexicon briefing
    briefing_json, briefing_text = build_lexicon_briefing(lexicon)
    lexicon_content = (
        "## Current Lexicon State\n"
        f"```json\n{json.dumps(briefing_json, default=str)}\n```\n\n"
        f"{briefing_text}"
    )

    # Initial strategy hint (placeholder — updated by controller)
    strategy_content = "## Strategy Hint\nMode: explore. Begin your investigation."

    # Initial evidence memory content (empty until first fetch)
    evidence_memory_content = (
        "## Evidence Memory\nNo evidence gathered yet. "
        "Use search_v10 and fetch_chunks to build your evidence base."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": V10_SYSTEM_PROMPT},            # [0]
        {"role": "user", "content": user_content},                     # [1]
        {"role": "assistant", "content": lexicon_content},             # [2] LEXICON_MSG_IDX
        {"role": "assistant", "content": strategy_content},            # [3] STRATEGY_MSG_IDX
        {"role": "assistant", "content": evidence_memory_content},     # [4] EVIDENCE_MEMORY_MSG_IDX
    ]
    LEXICON_MSG_IDX = 2
    STRATEGY_MSG_IDX = 3
    EVIDENCE_MEMORY_MSG_IDX = 4

    # --- Create ResearchWorkspace (V9-style evidence memory container) ---
    workspace = ResearchWorkspace(
        question=question,
        scope=scope or ScopeFilter(),
    )

    # Investigation loop state
    trace: List[Dict[str, Any]] = []
    tool_calls_remaining = V10_MAX_TOOL_CALLS
    narrative = ""
    claims: List[GroundedClaim] = []
    stall_rounds = 0
    empty_content_nudges = 0

    # Shared enrichment state
    surfaces_cache: Dict = {}
    seen_doc_ids: Set[int] = {h.document_id for h in all_hits if h.document_id}

    # Multi-signal progress tracking
    prev_snapshot = _snapshot_lexicon_counts(lexicon)

    # V10.3 controller state (initialized if enabled)
    coverage_map = None
    scoreboard = None
    if V10_EXPLORE_EXPLOIT:
        try:
            from retrieval.agent.v10_policy import (
                CoverageMapV10,
                ScoreboardV10,
                update_coverage_from_hits,
                update_scoreboard_from_enrichment,
                compute_gaps,
                choose_mode,
                build_explore_exploit_hint,
            )
            coverage_map = CoverageMapV10()
            scoreboard = ScoreboardV10()
        except ImportError:
            logger.debug("v10_policy not available, disabling explore/exploit")

    max_rounds = max_investigation_rounds if max_investigation_rounds is not None else V10_MAX_INVESTIGATION_ROUNDS
    for round_num in range(V10_MAX_TOOL_CALLS):
        if tool_calls_remaining <= 0:
            break
        if round_num >= max_rounds:
            if verbose:
                print(
                    f"[V10.2]   Round limit ({max_rounds}) reached, stopping.",
                    flush=True,
                )
            break

        if verbose:
            print(f"[V10.2]   Round {round_num + 1}: calling model...", flush=True)
        response = _call_llm(
            client, model, messages,
            response_format=V10_RESPONSE_FORMAT,
            tools=V10_TOOLS_DEF,
            max_tokens=V10_TOOL_TURN_MAX_TOKENS,
            verbose=verbose,
        )

        msg = response.choices[0].message

        # Handle tool calls
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            had_search = False
            last_enrichment: Dict[str, Any] = {}
            search_count_this_round = 0
            round_hits: List[CatalogHitV10] = []

            for tc in msg.tool_calls:
                tool_calls_remaining -= 1
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                # Per-round search budget enforcement
                if tool_name == "search_v10":
                    search_count_this_round += 1
                    if search_count_this_round > MAX_SEARCHES_PER_ROUND:
                        tool_result = {
                            "error": f"Search budget exceeded: max {MAX_SEARCHES_PER_ROUND} searches per round. "
                                     "Analyze existing results before searching again."
                        }
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tool_result, default=str),
                        })
                        trace.append({
                            "action": f"tool:{tool_name}:budget_exceeded",
                            "args": tool_args,
                            "round": round_num,
                        })
                        continue

                # Determine controller mode for enrichment
                current_mode = "explore"
                if coverage_map and scoreboard:
                    try:
                        gaps = compute_gaps(coverage_map, scoreboard, lexicon)
                        current_mode = choose_mode(scoreboard, gaps)
                    except Exception:
                        pass

                tool_result = _execute_tool(
                    conn, tool_name, tool_args, lexicon,
                    all_hits, all_mentions, workspace, scope, verbose,
                    client=client,
                    surfaces_cache=surfaces_cache,
                    seen_doc_ids=seen_doc_ids,
                    controller_mode=current_mode,
                    pem_seed=pem_seed,
                    question=question,
                    progress_callback=progress_callback,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, default=str),
                })
                trace.append({
                    "action": f"tool:{tool_name}",
                    "args": tool_args,
                    "round": round_num,
                })
                if tool_name == "search_v10":
                    had_search = True
                    last_enrichment = tool_result.get("enrichment") or {}

            # After tool calls: update lexicon briefing + strategy hint
            if had_search:
                # Update lexicon briefing (compact + delta)
                new_briefing_json, new_briefing_text = build_lexicon_briefing(
                    lexicon, compact=True, enrichment_delta=last_enrichment
                )
                new_lexicon_content = (
                    "## Current Lexicon State\n"
                    f"```json\n{json.dumps(new_briefing_json, default=str)}\n```\n\n"
                    f"{new_briefing_text}"
                )
                messages[LEXICON_MSG_IDX]["content"] = new_lexicon_content

                # Multi-signal progress delta
                current_snapshot = _snapshot_lexicon_counts(lexicon)
                delta = _compute_progress_delta(
                    lexicon, *prev_snapshot,
                )
                prev_snapshot = current_snapshot

                if delta.is_stalled:
                    stall_rounds += 1
                else:
                    stall_rounds = 0

                # Update strategy hint (V10.3 controller)
                if coverage_map and scoreboard:
                    try:
                        # Update coverage and scoreboard from latest hits
                        update_coverage_from_hits(coverage_map, all_hits)
                        update_scoreboard_from_enrichment(scoreboard, lexicon)

                        gaps = compute_gaps(coverage_map, scoreboard, lexicon)
                        mode = choose_mode(scoreboard, gaps)
                        hint = build_explore_exploit_hint(
                            mode, gaps, scoreboard, coverage_map, delta, stall_rounds,
                        )
                        messages[STRATEGY_MSG_IDX]["content"] = hint
                    except Exception as e:
                        logger.debug("Strategy hint update failed: %s", e)
                elif stall_rounds >= 2:
                    # Basic stall nudge without full controller
                    messages[STRATEGY_MSG_IDX]["content"] = (
                        "## Strategy Hint\n"
                        f"WARNING: Investigation appears stalled ({stall_rounds} rounds with minimal progress). "
                        "Consider: (1) broadening search to new collections, "
                        "(2) trying different entity forms, or "
                        "(3) setting final=true to synthesize with current evidence."
                    )

                if progress_callback:
                    try:
                        progress_callback(
                            "investigation",
                            "searching",
                            f"Round {round_num + 1}: {len(all_hits)} total chunks, "
                            f"{len(lexicon.entities_in_play)} entities",
                            {"round": round_num, "delta_score": delta.score},
                        )
                    except Exception:
                        pass

            # --- Update evidence memory slot (after any tool round) ---
            if workspace.evidence_memory:
                try:
                    view = select_evidence_memory_view(
                        workspace, question, workspace.investigation.gaps,
                    )
                    messages[EVIDENCE_MEMORY_MSG_IDX]["content"] = _render_evidence_memory(view)
                    if verbose:
                        _print_evidence_bullets_at_stage(
                            round_num + 1,
                            view,
                            len(workspace._bullet_index),
                        )
                except Exception as e:
                    logger.debug("Evidence memory update failed: %s", e)

            continue

        # No tool calls — parse the response
        if msg.content:
            try:
                output = json.loads(msg.content)
            except json.JSONDecodeError:
                narrative = msg.content
                break

            if output.get("final"):
                synthesis = output.get("synthesis", {})
                if synthesis:
                    narrative = synthesis.get("answer", "")
                    for claim_data in synthesis.get("claims", []):
                        v9_claim = V9Claim(
                            text=claim_data.get("text", ""),
                            confidence=claim_data.get("confidence", "medium"),
                            citation_chunk_ids=claim_data.get("evidence_chunk_ids", []),
                            linked_entity_ids=claim_data.get("entity_ids", []),
                        )
                        claims.append(GroundedClaim(
                            claim=v9_claim,
                            status="grounded",
                            citation_chunk_ids=claim_data.get("evidence_chunk_ids", []),
                        ))

                # --- Finalization: grounding + verification ---
                if claims:
                    verification_ok = _verify_claims(
                        conn, claims, lexicon, all_mentions, verbose,
                    )
                    if not verification_ok and round_num < (V10_MAX_TOOL_CALLS - 1):
                        # Feed errors back for bounded retries
                        verify_retries = 0
                        while not verification_ok and verify_retries < MAX_VERIFY_RETRIES:
                            verify_retries += 1
                            error_msg = _build_verification_error_msg(claims, lexicon)
                            messages.append({"role": "assistant", "content": msg.content})
                            messages.append({
                                "role": "user",
                                "content": error_msg,
                            })
                            if verbose:
                                print(f"[V10.2]   Verification failed, retry {verify_retries}/{MAX_VERIFY_RETRIES}", flush=True)

                            retry_resp = _call_llm(
                                client, model, messages,
                                response_format=V10_RESPONSE_FORMAT,
                                tools=V10_TOOLS_DEF,
                                max_tokens=V10_SYNTHESIS_MAX_TOKENS,
                                verbose=verbose,
                            )
                            retry_msg = retry_resp.choices[0].message
                            if retry_msg.content:
                                try:
                                    retry_output = json.loads(retry_msg.content)
                                    if retry_output.get("final"):
                                        retry_synth = retry_output.get("synthesis", {})
                                        if retry_synth:
                                            narrative = retry_synth.get("answer", "")
                                            claims = []
                                            for cd in retry_synth.get("claims", []):
                                                v9c = V9Claim(
                                                    text=cd.get("text", ""),
                                                    confidence=cd.get("confidence", "medium"),
                                                    citation_chunk_ids=cd.get("evidence_chunk_ids", []),
                                                    linked_entity_ids=cd.get("entity_ids", []),
                                                )
                                                claims.append(GroundedClaim(
                                                    claim=v9c,
                                                    status="grounded",
                                                    citation_chunk_ids=cd.get("evidence_chunk_ids", []),
                                                ))
                                        verification_ok = _verify_claims(
                                            conn, claims, lexicon, all_mentions, verbose,
                                        )
                                except json.JSONDecodeError:
                                    break
                            else:
                                break
                break
            else:
                # Investigation mode — parse scratchpad_update and continue
                scratchpad = output.get("scratchpad_update") or {}

                # Apply V9-style investigation state updates
                _apply_investigation_update(workspace.investigation, scratchpad)

                # Apply pin suggestions
                pin_suggestions = scratchpad.get("pin_suggestions", [])
                if pin_suggestions and isinstance(pin_suggestions, list):
                    apply_pin_suggestions(workspace, pin_suggestions)
                    if verbose:
                        print(f"[V10.2]   Pin suggestions applied: {len(pin_suggestions)}", flush=True)

                # Apply V10-style alias mappings / promotion actions (existing logic)
                _apply_v10_alias_actions(lexicon, scratchpad, conn, verbose)

                # Update evidence memory slot after scratchpad changes
                if workspace.evidence_memory:
                    try:
                        view = select_evidence_memory_view(
                            workspace, question, workspace.investigation.gaps,
                        )
                        messages[EVIDENCE_MEMORY_MSG_IDX]["content"] = _render_evidence_memory(view)
                    except Exception as e:
                        logger.debug("Evidence memory update failed: %s", e)

                messages.append({"role": "assistant", "content": msg.content})
                messages.append({
                    "role": "user",
                    "content": "Continue your investigation. Use tools to gather more evidence, "
                               "or set final=true if ready to synthesize."
                })
        else:
            # Model returned no content and no tool calls — nudge once instead of exiting with empty answer
            empty_content_nudges += 1
            if empty_content_nudges <= 1 and tool_calls_remaining > 0:
                messages.append({"role": "assistant", "content": msg.content or "(no content)"})
                messages.append({
                    "role": "user",
                    "content": (
                        "You did not provide an answer or more tool calls. "
                        "You must either: (1) use fetch_chunks to read full text of relevant chunks, then set final=true with your synthesis and cited claims, "
                        "or (2) run more searches/lookups and then synthesize. "
                        "Set final=true and provide at least a short answer (or state what evidence is missing)."
                    ),
                })
                if verbose:
                    print("[V10.2]   Model returned empty content — nudging for synthesis", flush=True)
            else:
                break

    # Fallback when we exited with no narrative (e.g. model kept returning empty or we hit nudge limit)
    if not narrative and not claims:
        narrative = (
            "No synthesis was produced. The model may have stopped without setting final=true or returned an empty answer. "
            "Try rephrasing the question or using /deeper to continue the investigation."
        )

    return narrative, claims, trace, workspace


def _verify_claims(
    conn,
    claims: List[GroundedClaim],
    lexicon: LexiconV10,
    all_mentions: Dict[int, ChunkMentionsV10],
    verbose: bool,
) -> bool:
    """Verify claims against entity-id evidence. Returns True if all pass."""
    all_ok = True
    for gc in claims:
        claim = gc.claim
        if not claim:
            continue
        # Check citation chunk_ids exist in collected mentions
        for cid in (claim.citation_chunk_ids or []):
            if cid not in all_mentions:
                gc.status = "ungrounded"
                all_ok = False
                break
        # Check linked entity_ids are in lexicon
        for eid in (claim.linked_entity_ids or []):
            if eid not in lexicon.entities_in_play:
                gc.status = "ungrounded"
                all_ok = False
                break
    if verbose and not all_ok:
        ungrounded = [gc for gc in claims if gc.status == "ungrounded"]
        print(f"[V10.2]   Verification: {len(ungrounded)}/{len(claims)} claims ungrounded", flush=True)
    return all_ok


def _build_verification_error_msg(
    claims: List[GroundedClaim],
    lexicon: LexiconV10,
) -> str:
    """Build actionable error message for failed verification."""
    errors = []
    for i, gc in enumerate(claims):
        if gc.status == "ungrounded":
            claim = gc.claim
            missing_chunks = [
                cid for cid in (claim.citation_chunk_ids or [])
            ]
            missing_entities = [
                eid for eid in (claim.linked_entity_ids or [])
                if eid not in lexicon.entities_in_play
            ]
            error = f"Claim {i + 1}: \"{(claim.text or '')[:80]}...\" is ungrounded."
            if missing_entities:
                error += f" Unknown entity_ids: {missing_entities}."
            if missing_chunks:
                error += f" Cited chunk_ids: {missing_chunks} — verify they exist."
            errors.append(error)

    return (
        "## Verification Failed\n"
        "Some claims could not be grounded. Fix and resubmit with final=true:\n"
        + "\n".join(errors) + "\n\n"
        "Use search_v10/fetch_chunks to gather missing evidence, or remove ungrounded claims."
    )


# =============================================================================
# Workspace helpers
# =============================================================================

def _workspace_chunk_lookup(workspace: ResearchWorkspace, chunk_id: int) -> Optional[WorkspaceChunk]:
    """Look up a chunk by ID in the workspace's fulltext_chunks list."""
    for c in workspace.fulltext_chunks:
        if c.chunk_id == chunk_id:
            return c
    return None


def _workspace_chunks_dict(workspace: ResearchWorkspace) -> Dict[int, WorkspaceChunk]:
    """Build a {chunk_id: WorkspaceChunk} dict from workspace for backward compatibility."""
    return {c.chunk_id: c for c in workspace.fulltext_chunks}


def _build_alias_context_from_lexicon(lexicon: LexiconV10) -> str:
    """Build a compact alias context string for the summarizer prompt.

    Analogous to V9's build_alias_context_for_summarizer but uses
    V10's LexiconV10 instead of workspace entities.

    Example output:
      Known identities: CABIN / OSS = Office of Strategic Services;
      LIBERAL = Julius Rosenberg
    """
    parts: List[str] = []
    for eid, info in lexicon.entities_in_play.items():
        canonical = info.get("canonical_name", "")
        if not canonical:
            continue
        # Collect scoped alias surfaces for this entity
        alias_surfaces: List[str] = []
        for scope_key, aliases in lexicon.aliases_by_entity_scoped.items():
            # scope_key is (entity_id, collection_slug)
            if isinstance(scope_key, tuple) and len(scope_key) >= 1 and scope_key[0] == eid:
                for alias in aliases[:5]:
                    surface = alias.get("surface", "") if isinstance(alias, dict) else str(alias)
                    if surface and surface.lower() != canonical.lower():
                        alias_surfaces.append(surface)
        # Also add global forms
        forms = info.get("forms", [])
        for f in forms[:3]:
            if f and f.lower() != canonical.lower() and f not in alias_surfaces:
                alias_surfaces.append(f)

        if alias_surfaces:
            alias_str = " / ".join(alias_surfaces[:5])
            parts.append(f"{alias_str} = {canonical}")
        else:
            parts.append(canonical)

    if not parts:
        return ""
    return "Known identities: " + "; ".join(parts)


def _render_evidence_memory(view: EvidenceMemoryView) -> str:
    """Render the evidence memory view as a string for the message slot.

    Ported from V9's context pack builder (v9_context.py lines 277-327),
    adapted for use as a standalone message slot in V10.
    """
    sections: List[str] = []
    sections.append("## Evidence Memory")

    total = (
        len(view.pinned_bullets)
        + len(view.recent_bullets)
        + len(view.top_relevant_bullets)
    )

    if total == 0:
        sections.append("No evidence gathered yet.")
        return "\n".join(sections)

    if view.pinned_bullets:
        block = f"### Pinned Evidence ({len(view.pinned_bullets)})\n"
        for b in view.pinned_bullets:
            block += _render_bullet_line(b)
        sections.append(block)

    if view.recent_bullets:
        block = f"### Recent Evidence ({len(view.recent_bullets)})\n"
        for b in view.recent_bullets:
            block += _render_bullet_line(b)
        sections.append(block)

    if view.top_relevant_bullets:
        block = f"### Relevant Evidence ({len(view.top_relevant_bullets)})\n"
        for b in view.top_relevant_bullets:
            block += _render_bullet_line(b)
        sections.append(block)

    # Open questions / leads / warnings
    meta_lines = ""
    if view.open_questions:
        meta_lines += "**Open questions:** " + "; ".join(view.open_questions[:4]) + "\n"
    if view.leads:
        meta_lines += "**Leads:** " + "; ".join(view.leads[:6]) + "\n"
    if view.warnings:
        meta_lines += "**Warnings:** " + "; ".join(view.warnings[:3]) + "\n"
    if meta_lines:
        sections.append(meta_lines)

    return "\n".join(sections)


def _render_bullet_line(b: EvidenceBullet) -> str:
    """Render one evidence bullet as a compact context line."""
    chunks_str = ",".join(str(c) for c in b.supporting_chunk_ids[:6])
    line = f"  - [B:{b.bullet_id}] {b.text} (chunks: {chunks_str})"
    if b.tags:
        line += f" [tags: {','.join(b.tags)}]"
    return line + "\n"


def _print_evidence_bullets_at_stage(
    round_num: int,
    view: EvidenceMemoryView,
    total_bullets: int,
) -> None:
    """Print evidence bullets at current stage (verbose output)."""
    lines = [
        f"[V10.2]   Evidence (Round {round_num}): "
        f"pinned={len(view.pinned_bullets)}, recent={len(view.recent_bullets)}, "
        f"top={len(view.top_relevant_bullets)}, total={total_bullets}",
    ]
    for label, bullets in [
        ("Pinned", view.pinned_bullets),
        ("Recent", view.recent_bullets),
        ("Top relevant", view.top_relevant_bullets),
    ]:
        if bullets:
            for b in bullets[:6]:  # up to 6 per section
                txt = (b.text[:120] + "…") if len(b.text) > 120 else b.text
                chunks_str = ",".join(str(c) for c in b.supporting_chunk_ids[:4])
                lines.append(f"    • [{label}] {txt} (chunks: {chunks_str})")
    print("\n".join(lines), flush=True)


def _apply_investigation_update(investigation: InvestigationState, scratchpad: Dict[str, Any]) -> None:
    """Apply V9-style investigation state fields from scratchpad_update.

    Ported from V9 runner's _update_investigation. Updates goal, leads,
    hypotheses, gaps, next_actions, and ready_to_synthesize.
    """
    if not scratchpad:
        return
    if "goal" in scratchpad and scratchpad["goal"]:
        investigation.goal = str(scratchpad["goal"])[:500]
    if "leads" in scratchpad and isinstance(scratchpad["leads"], list):
        investigation.leads = [str(l)[:200] for l in scratchpad["leads"][:10]]
    if "hypotheses" in scratchpad and isinstance(scratchpad["hypotheses"], list):
        investigation.hypotheses = [str(h)[:200] for h in scratchpad["hypotheses"][:10]]
    if "gaps" in scratchpad and isinstance(scratchpad["gaps"], list):
        investigation.gaps = [str(g)[:200] for g in scratchpad["gaps"][:10]]
    if "next_actions" in scratchpad and isinstance(scratchpad["next_actions"], list):
        investigation.next_actions = [str(a)[:200] for a in scratchpad["next_actions"][:10]]
    if "ready_to_synthesize" in scratchpad:
        investigation.ready_to_synthesize = bool(scratchpad.get("ready_to_synthesize", False))
    if "notes" in scratchpad and scratchpad["notes"]:
        # Append notes to investigation trace for persistence
        investigation.trace.append(InvestigationStep(
            step_idx=len(investigation.trace),
            action="notes",
            rationale=str(scratchpad["notes"])[:500],
        ))


def _apply_v10_alias_actions(
    lexicon: LexiconV10,
    scratchpad: Dict[str, Any],
    conn,
    verbose: bool,
) -> None:
    """Apply V10-specific alias_mappings and promotion_actions from scratchpad.

    These are identity layer updates that V10 adds on top of V9's
    investigation state.
    """
    from retrieval.agent.v10_types import AliasMappingHypothesis, SpanCandidate

    # alias_mappings: model-observed alias -> entity mappings
    alias_mappings = scratchpad.get("alias_mappings", [])
    if alias_mappings and isinstance(alias_mappings, list):
        for mapping in alias_mappings[:10]:
            if not isinstance(mapping, dict):
                continue
            alias_text = mapping.get("alias", "")
            entity_name = mapping.get("entity_name", "")
            confidence = mapping.get("confidence", "low")
            doc_context = mapping.get("document_context", "")
            if alias_text and entity_name:
                # Try to find matching entity in lexicon
                matched_eid = None
                for eid, info in lexicon.entities_in_play.items():
                    if info.get("canonical_name", "").lower() == entity_name.lower():
                        matched_eid = eid
                        break
                if matched_eid is not None:
                    # Record as a contextual hypothesis on the lexicon
                    hyp = AliasMappingHypothesis(
                        collection_slug="*",  # cross-collection
                        alias_text=alias_text.lower(),
                        candidates=[SpanCandidate(
                            entity_id=matched_eid,
                            canonical_name=entity_name,
                            match_type="model_observed",
                        )],
                        status="provisional" if confidence in ("high", "medium") else "unresolved",
                        confidence=0.8 if confidence == "high" else 0.5 if confidence == "medium" else 0.3,
                        support=[{"source": "model_scratchpad", "context": doc_context[:200]}],
                    )
                    lexicon.set_hypothesis(hyp)
                    if verbose:
                        print(
                            f"[V10.2]   Alias mapping: {alias_text} -> {entity_name} "
                            f"(entity_id={matched_eid}, conf={confidence})",
                            flush=True,
                        )

    # promotion_actions: model-requested status changes for alias hypotheses
    promotion_actions = scratchpad.get("promotion_actions", [])
    if promotion_actions and isinstance(promotion_actions, list):
        for action in promotion_actions[:10]:
            if not isinstance(action, dict):
                continue
            alias_text = action.get("alias_text", "")
            collection_slug = action.get("collection_slug", "")
            new_status = action.get("new_status", "")
            entity_id = action.get("entity_id", 0)
            if alias_text and new_status:
                # Find matching hypothesis and update status
                for key, hyp in lexicon.alias_mapping_hypotheses.items():
                    if (hyp.alias_text.lower() == alias_text.lower()
                            and (not collection_slug or hyp.collection_slug == collection_slug)):
                        hyp.status = new_status
                        if verbose:
                            print(
                                f"[V10.2]   Promotion: {alias_text} -> {new_status} "
                                f"(entity_id={entity_id})",
                                flush=True,
                            )
                        break


# =============================================================================
# Tool execution
# =============================================================================

def _execute_tool(
    conn,
    tool_name: str,
    tool_args: Dict[str, Any],
    lexicon: LexiconV10,
    all_hits: List[CatalogHitV10],
    all_mentions: Dict[int, ChunkMentionsV10],
    workspace: ResearchWorkspace,
    scope: Optional[ScopeFilter],
    verbose: bool,
    client=None,
    surfaces_cache: Optional[Dict] = None,
    seen_doc_ids: Optional[Set[int]] = None,
    controller_mode: str = "explore",
    pem_seed: Optional[PemLaneResult] = None,
    question: str = "",
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Execute a V10 tool call and return the result."""
    try:
        if tool_name == "search_v10":
            return _tool_search_v10(
                conn, tool_args, lexicon, all_hits, all_mentions, scope, verbose,
                client=client,
                surfaces_cache=surfaces_cache or {},
                seen_doc_ids=seen_doc_ids or set(),
                controller_mode=controller_mode,
                pem_seed=pem_seed,
            )
        elif tool_name == "fetch_chunks":
            return _tool_fetch_chunks(
                conn, tool_args, workspace, lexicon, question, verbose,
                progress_callback=progress_callback,
            )
        elif tool_name == "extract_mentions":
            return _tool_extract_mentions(
                conn, tool_args, lexicon, all_mentions, workspace, verbose
            )
        elif tool_name == "resolve_referent_v10":
            from retrieval.agent.v10_tools import tool_resolve_referent
            result = tool_resolve_referent(conn, tool_args)
            if verbose:
                print(f"[V10.2]   resolve_referent_v10: {result.get('total_candidates', 0)} candidates", flush=True)
            return result
        elif tool_name == "alias_index_summary_v10":
            from retrieval.agent.v10_tools import tool_alias_index_summary
            result = tool_alias_index_summary(conn, tool_args)
            if verbose:
                print(f"[V10.2]   alias_index_summary_v10: {result.get('unique_entities', 0)} entities", flush=True)
            return result
        elif tool_name == "alias_index_lookup_v10":
            from retrieval.agent.v10_tools import tool_alias_index_lookup
            result = tool_alias_index_lookup(conn, tool_args)
            if verbose:
                print(f"[V10.2]   alias_index_lookup_v10: {result.get('total_returned', 0)} occurrences", flush=True)
            return result
        elif tool_name == "aliases_for_entity_v10":
            from retrieval.agent.v10_tools import tool_aliases_for_entity
            result = tool_aliases_for_entity(conn, tool_args)
            if verbose:
                cn = result.get('canonical_name', '?')
                n_scoped = len(result.get('scoped_aliases', []))
                print(f"[V10.2]   aliases_for_entity_v10: {cn} → {n_scoped} scoped aliases", flush=True)
            return result
        elif tool_name == "alias_index_sample_v10":
            from retrieval.agent.v10_tools import tool_alias_index_sample
            result = tool_alias_index_sample(conn, tool_args)
            if verbose:
                print(f"[V10.2]   alias_index_sample_v10: {len(result.get('sample', []))} samples", flush=True)
            return result
        elif tool_name == "grant_alias_power_v10":
            from retrieval.agent.v10_tools import tool_grant_alias_power
            result = tool_grant_alias_power(conn, tool_args, lexicon)
            if verbose:
                granted = result.get('granted', False)
                print(f"[V10.2]   grant_alias_power_v10: granted={granted}", flush=True)
            return result
        elif tool_name == "surface_top_referent_v10":
            from retrieval.agent.v10_tools import tool_surface_top_referent
            result = tool_surface_top_referent(conn, tool_args)
            if verbose:
                eid = result.get('entity_id', '?')
                share = result.get('share', 0)
                print(f"[V10.2]   surface_top_referent_v10: entity_id={eid}, share={share}", flush=True)
            return result
        elif tool_name == "get_lexicon_state_v10":
            detail = tool_args.get("detail", "full")
            briefing_json, briefing_text = build_lexicon_briefing(lexicon, compact=(detail != "full"))
            return {
                "detail": detail,
                "briefing_json": briefing_json,
                "briefing_text": briefing_text,
            }
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, e)
        return {"error": str(e)}


def _tool_search_v10(
    conn,
    args: Dict[str, Any],
    lexicon: LexiconV10,
    all_hits: List[CatalogHitV10],
    all_mentions: Dict[int, ChunkMentionsV10],
    scope: Optional[ScopeFilter],
    verbose: bool,
    client=None,
    surfaces_cache: Optional[Dict] = None,
    seen_doc_ids: Optional[Set[int]] = None,
    controller_mode: str = "explore",
    pem_seed: Optional[PemLaneResult] = None,
) -> Dict[str, Any]:
    """Execute search_v10 tool with search rails and post-search enrichment.

    Search rails:
    - k clamped to [5, 30]
    - entity_ids validated against lexicon (unknown IDs dropped with warning)
    - alias_boosts only for ALIAS_SCOPED_COLLECTIONS
    - locked_entity_id only when context-safe
    - per-doc chunk cap in enrichment
    - PEM lane seeds merged into first search response
    """
    query = args.get("query", "")
    raw_k = args.get("top_k", 15)
    top_k = min(max(raw_k, 5), 30)  # Clamp to [5, 30]

    warnings: List[str] = []
    if raw_k != top_k:
        warnings.append(f"top_k clamped from {raw_k} to {top_k}")

    # Parse and validate entity boosts
    entity_boosts = []
    for eb_data in args.get("entity_boosts", []):
        eid = eb_data.get("entity_id", 0)
        # Validate entity_id against lexicon
        if eid and eid not in lexicon.entities_in_play:
            warnings.append(f"Dropped unknown entity_id {eid} from entity_boosts")
            continue
        entity_boosts.append(EntityBoost(
            entity_id=eid,
            forms=eb_data.get("forms", []),
            weight=eb_data.get("weight", 1.0),
        ))

    # Parse and validate alias boosts (with permission gating)
    from retrieval.agent.v10_normalize import normalize_alias_surface as _normalize_alias
    alias_boosts = []
    for ab_data in args.get("alias_boosts", []):
        coll = ab_data.get("collection_slug", "")
        alias_text = ab_data.get("alias_text", "")
        alias_norm = _normalize_alias(alias_text)
        locked_eid = ab_data.get("locked_entity_id")

        # Gate 1: Only alias-scoped collections
        if coll not in ALIAS_SCOPED_COLLECTIONS:
            warnings.append(f"Dropped alias boost for non-alias collection '{coll}' [alias_boost_dropped_not_scoped]")
            continue

        # Gate 2: Permission check (codename aliases must be granted)
        perm = lexicon.has_alias_permission(coll, alias_norm, locked_eid)
        if not perm:
            # Check if ANY permission exists for this (coll, alias) regardless of entity_id
            any_perm = lexicon.has_alias_permission(coll, alias_norm, None)
            if not any_perm:
                warnings.append(
                    f"Dropped alias boost for '{alias_text}' in '{coll}' — no permission granted "
                    f"[alias_boost_dropped_no_permission]. Call grant_alias_power_v10 first."
                )
                continue
            perm = any_perm

        # Gate 3: Lock validation (provisional permission forbids locks)
        if locked_eid:
            perm_status = perm.get("status", "provisional") if perm else "provisional"
            if perm_status == "provisional":
                warnings.append(
                    f"Removed locked_entity_id={locked_eid} for alias '{alias_text}' "
                    f"(provisional permission — boosts only, no lock)"
                )
                locked_eid = None
            else:
                # Confirmed permission: still check _is_lock_safe
                is_safe_to_lock = _is_lock_safe(lexicon, alias_text, coll, locked_eid)
                if not is_safe_to_lock:
                    warnings.append(
                        f"Removed locked_entity_id={locked_eid} for alias '{alias_text}' "
                        f"(not confirmed/unambiguous)"
                    )
                    locked_eid = None

        alias_boosts.append(AliasScopedBoost(
            collection_slug=coll,
            alias_text=alias_text,
            locked_entity_id=locked_eid,
            weight=ab_data.get("weight", 1.0),
        ))

    # Scope override
    scope_collections = args.get("scope_collections")
    if scope_collections:
        filters = SearchFilters(collection_slugs=scope_collections)
    elif scope:
        filters = _scope_to_filters(scope)
    else:
        filters = SearchFilters()

    hits, provenance = search_chunks_v10(
        conn, query,
        scope=filters,
        entity_boosts=entity_boosts,
        alias_boosts_scoped=alias_boosts,
        k=top_k,
    )

    # Dedup by chunk_id when adding to accumulated hits
    existing_ids = {h.chunk_id for h in all_hits}
    for h in hits:
        if h.chunk_id not in existing_ids:
            all_hits.append(h)
            existing_ids.add(h.chunk_id)

    if verbose:
        print(f"[V10.2]   search_v10: {len(hits)} results for '{query[:50]}' (k={top_k})", flush=True)

    # --- PEM lane seed merge (first search only) — BEFORE enrichment so PEM seeds get LLM priority ---
    is_first_search = len(existing_ids) == len(hits)
    pem_seed_results: List[Dict[str, Any]] = []
    pem_seed_hits: List[CatalogHitV10] = []
    search_chunk_ids = {h.chunk_id for h in hits}
    if is_first_search and pem_seed and pem_seed.chunk_ids:
        pem_only_ids = [cid for cid in pem_seed.chunk_ids if cid not in search_chunk_ids]
        if pem_only_ids:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT c.id,
                               cm.collection_slug,
                               cm.document_id,
                               cm.first_page_id,
                               LEFT(COALESCE(c.clean_text, c.text), 300) AS snippet
                        FROM chunks c
                        JOIN chunk_metadata cm ON cm.chunk_id = c.id
                        WHERE c.id = ANY(%s)
                    """, (pem_only_ids,))
                    pem_meta = {r[0]: r for r in cur.fetchall()}
            except Exception as e:
                logger.debug("PEM seed metadata fetch failed: %s", e)
                pem_meta = {}

            for cid in pem_seed.chunk_ids:
                if cid in search_chunk_ids:
                    continue
                meta = pem_meta.get(cid)
                if not meta:
                    continue
                _, coll_slug, doc_id, page_id, snippet = meta
                page_no = None
                if page_id:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT page_num FROM pages WHERE id = %s", (page_id,))
                            prow = cur.fetchone()
                            if prow:
                                page_no = prow[0]
                    except Exception:
                        pass
                surface_reason = pem_seed.chunk_surface_map.get(cid, "")
                pem_seed_results.append({
                    "chunk_id": cid,
                    "score": 0.0,
                    "collection": coll_slug,
                    "document_id": doc_id,
                    "page_no": page_no,
                    "snippet": (snippet or "")[:300],
                    "origin": "pem_seed",
                    "reason_code": f"pem_surface:{surface_reason}" if surface_reason else "pem_seed",
                })
                pem_seed_hits.append(CatalogHitV10(
                    chunk_id=cid,
                    score=0.0,
                    collection_slug=coll_slug,
                    document_id=doc_id,
                    page_no=page_no,
                    snippet=snippet or "",
                    origin="pem_seed",
                ))
                all_hits.append(pem_seed_hits[-1])

        for cid in pem_seed.chunk_ids:
            if cid in search_chunk_ids:
                surface_reason = pem_seed.chunk_surface_map.get(cid, "")
                pem_seed_results.append({
                    "chunk_id": cid,
                    "origin": "pem_seed_overlap",
                    "reason_code": f"pem_surface:{surface_reason}" if surface_reason else "pem_seed",
                })
                # Mark overlap hit for LLM priority
                for h in hits:
                    if h.chunk_id == cid:
                        h.origin = "pem_seed"
                        break

        if verbose and pem_seed_results:
            n_new = sum(1 for r in pem_seed_results if r.get("origin") == "pem_seed")
            n_overlap = sum(1 for r in pem_seed_results if r.get("origin") == "pem_seed_overlap")
            print(f"[V10.2]   PEM seeds merged: {n_new} new + {n_overlap} overlap", flush=True)

        pem_seed.chunk_ids = []

    # Mark search hits for prioritization
    for h in hits:
        if getattr(h, "origin", None) is None:
            h.origin = "search"

    # --- Post-search enrichment (PEM seeds first, cap LLM at 4, batch fetch, concurrent) ---
    enrichment_dict: Dict[str, Any] = {}
    recommended_boosts: Dict[str, Any] = {}
    enrich_hits_input = pem_seed_hits + hits
    # Prefer DATABASE_URL for worker conns: conn.dsn often omits password (psycopg2 security)
    dsn = os.environ.get("DATABASE_URL") or getattr(conn, "dsn", None)
    try:
        if verbose:
            print(f"[V10.2]   enrichment: starting ({len(enrich_hits_input)} chunks)...", flush=True)
        enrichment_summary = enrich_hits_with_mentions(
            conn, client, enrich_hits_input, lexicon, all_mentions,
            seen_doc_ids if seen_doc_ids is not None else set(),
            surfaces_cache if surfaces_cache is not None else {},
            controller_mode=controller_mode,
            dsn=dsn,
            verbose=verbose,
        )
        enrichment_dict = {
            "chunks_extracted": enrichment_summary.chunks_extracted,
            "chunks_llm_extracted": enrichment_summary.chunks_llm_extracted,
            "chunks_deterministic": enrichment_summary.chunks_deterministic,
            "new_contextual_mappings": enrichment_summary.new_contextual_mappings,
            "new_general_hypotheses": enrichment_summary.new_general_hypotheses,
            "ambiguities_opened": enrichment_summary.ambiguities_opened,
            "ambiguities_closed": enrichment_summary.ambiguities_closed,
            "new_signals": enrichment_summary.new_signals,
            "aliases_backfilled": {
                str(k): v for k, v in enrichment_summary.aliases_backfilled.items()
            },
        }
        if enrichment_summary.batch_load_ms is not None:
            enrichment_dict["batch_load_ms"] = round(enrichment_summary.batch_load_ms, 1)
        if enrichment_summary.extract_total_ms is not None:
            enrichment_dict["extract_total_ms"] = round(enrichment_summary.extract_total_ms, 1)
        if enrichment_summary.llm_call_count:
            enrichment_dict["llm_call_count"] = enrichment_summary.llm_call_count
            enrichment_dict["llm_concurrent"] = enrichment_summary.llm_concurrent
        if enrichment_summary.llm_latency_p50_ms is not None:
            enrichment_dict["llm_latency_p50_ms"] = round(enrichment_summary.llm_latency_p50_ms, 1)
        if enrichment_summary.llm_latency_p95_ms is not None:
            enrichment_dict["llm_latency_p95_ms"] = round(enrichment_summary.llm_latency_p95_ms, 1)

        if verbose:
            print(f"[V10.2]   enrichment: computing recommended_boosts...", flush=True)
        boosts_start = time.perf_counter()
        recommended_boosts = _compute_recommended_boosts(
            lexicon, conn, seen_doc_ids if seen_doc_ids is not None else set(),
        )
        if verbose:
            n_ent = len(recommended_boosts.get("entity_boosts", []))
            n_alias = len(recommended_boosts.get("alias_boosts_scoped", []))
            print(
                f"[V10.2]   enrichment: recommended_boosts done in {(time.perf_counter() - boosts_start) * 1000:.0f}ms "
                f"({n_ent} entity, {n_alias} alias)",
                flush=True,
            )
        if verbose and enrichment_summary.chunks_extracted > 0:
            timing = ""
            if enrichment_summary.batch_load_ms is not None:
                timing += f" batch_load={enrichment_summary.batch_load_ms:.0f}ms"
            if enrichment_summary.extract_total_ms is not None:
                timing += f" extract_total={enrichment_summary.extract_total_ms:.0f}ms"
            if enrichment_summary.llm_call_count:
                timing += f" llm_calls={enrichment_summary.llm_call_count}"
                if enrichment_summary.llm_concurrent:
                    timing += "(concurrent)"
                if enrichment_summary.llm_latency_p50_ms is not None:
                    timing += f" p50={enrichment_summary.llm_latency_p50_ms:.0f}ms"
                if enrichment_summary.llm_latency_p95_ms is not None:
                    timing += f" p95={enrichment_summary.llm_latency_p95_ms:.0f}ms"
            print(
                f"[V10.2]   enrichment: {enrichment_summary.chunks_extracted} chunks "
                f"({enrichment_summary.chunks_llm_extracted} LLM, "
                f"{enrichment_summary.chunks_deterministic} deterministic), "
                f"{len(enrichment_summary.aliases_backfilled)} entities backfilled"
                f"{timing}",
                flush=True,
            )
    except Exception as e:
        logger.debug("Post-search enrichment failed: %s", e)

    # Build result — PEM seeds (full snippet) + top-K search snippets; rest metadata-only
    pem_new = [r for r in pem_seed_results if r.get("origin") == "pem_seed"]
    search_results_full = [
        {
            "chunk_id": h.chunk_id,
            "score": h.score,
            "collection": h.collection_slug,
            "document_id": h.document_id,
            "page_no": h.page_no,
            "snippet": h.snippet[:300],
            "provenance": provenance.get(h.chunk_id, {}).to_dict() if h.chunk_id in provenance else {},
        }
        for h in hits[:SEARCH_RESULTS_SNIPPET_TOP_K]
    ]
    search_results_meta = [
        {
            "chunk_id": h.chunk_id,
            "score": h.score,
            "collection": h.collection_slug,
            "document_id": h.document_id,
            "page_no": h.page_no,
        }
        for h in hits[SEARCH_RESULTS_SNIPPET_TOP_K:20]
    ]
    combined_results = pem_new + search_results_full + search_results_meta

    result = {
        "results": combined_results,
        "total": len(hits) + len(pem_new),
        "enrichment": enrichment_dict,
        "recommended_boosts": _truncate_recommended_boosts_for_tool(recommended_boosts),
    }
    if pem_seed_results:
        result["pem_seed_info"] = {
            "new_seeds": len(pem_new),
            "overlaps": [r for r in pem_seed_results if r.get("origin") == "pem_seed_overlap"],
        }
    if warnings:
        result["warnings"] = warnings
    return result


def _is_lock_safe(
    lexicon: LexiconV10,
    alias_text: str,
    collection_slug: str,
    entity_id: int,
) -> bool:
    """Check if locking alias to entity_id is safe.

    Safe when:
    1. Permission is 'confirmed' (not 'provisional') for this (coll, alias, entity_id), AND
    2. A referent rule matched for this alias in this collection, OR
       the alias is unambiguous (maps to exactly one entity_id), OR
       a confirmed hypothesis exists.

    Provisional permissions allow boosts only, not locks (D2).
    """
    # Gate: check permission status first
    from retrieval.agent.v10_normalize import normalize_alias_surface
    alias_norm = normalize_alias_surface(alias_text)
    perm = lexicon.has_alias_permission(collection_slug, alias_norm, entity_id)
    if perm and perm.get("status") == "provisional":
        return False  # provisional = boosts only, no lock

    # Check referent rules
    for rule_key, rules_list in lexicon.alias_referent_rules.items():
        for rule in rules_list:
            if (rule.alias_text.lower() == alias_text.lower()
                    and rule.collection_slug == collection_slug
                    and rule.entity_id == entity_id):
                return True

    # Check if alias is unambiguous
    # Structure: collection_slug -> {alias_text -> [entity_ids]}
    entity_ids_for_alias = lexicon.entities_by_alias_scoped.get(
        collection_slug, {}
    ).get(alias_text.lower(), [])
    if len(entity_ids_for_alias) == 1 and entity_ids_for_alias[0] == entity_id:
        return True

    # Check confirmed hypotheses
    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        if (hyp.alias_text.lower() == alias_text.lower()
                and hyp.collection_slug == collection_slug
                and hyp.status == "confirmed"
                and hyp.candidates
                and hyp.candidates[0].entity_id == entity_id):
            return True

    return False


def _tool_fetch_chunks(
    conn,
    args: Dict[str, Any],
    workspace: ResearchWorkspace,
    lexicon: LexiconV10,
    question: str,
    verbose: bool,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Fetch full text for chunks by ID, then run V9-style summarizer pipeline."""
    chunk_ids = args.get("chunk_ids", [])
    if not chunk_ids:
        return {"error": "No chunk_ids provided"}

    results = []
    new_chunks: List[WorkspaceChunk] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, COALESCE(c.clean_text, c.text) AS full_text,
                       cm.document_id, cm.collection_slug, cm.first_page_id
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE c.id = ANY(%s)
            """, (chunk_ids,))
            for row in cur.fetchall():
                chunk_id, text, doc_id, coll_slug, page_id = row

                # Resolve page number
                page_no = None
                if page_id:
                    cur.execute("SELECT page_num FROM pages WHERE id = %s", (page_id,))
                    prow = cur.fetchone()
                    if prow:
                        page_no = prow[0]

                # PEM annotation for alias-scoped chunks (model-only)
                pem_annotation = ""
                if coll_slug in ALIAS_SCOPED_COLLECTIONS:
                    try:
                        pem_annotation = build_chunk_pem_annotation(
                            conn, chunk_id,
                            alias_scoped_collections=list(ALIAS_SCOPED_COLLECTIONS),
                        )
                    except Exception as e_ann:
                        logger.debug("PEM annotation failed for chunk %d: %s", chunk_id, e_ann)

                model_text = (text or "")[:3000]
                if pem_annotation:
                    model_text = model_text + pem_annotation

                wc = WorkspaceChunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    doc_id=doc_id,
                    page=f"p{page_no}" if page_no else None,
                    collection_slug=coll_slug,
                    source_label=coll_slug,
                )
                new_chunks.append(wc)
                results.append({
                    "chunk_id": chunk_id,
                    "text": model_text,
                    "collection_slug": coll_slug,
                    "document_id": doc_id,
                    "page_no": page_no,
                })
    except Exception as e:
        return {"error": str(e)}

    if verbose:
        print(f"[V10]   fetch_chunks: {len(results)} chunks fetched", flush=True)

    # --- V9-style summarizer pipeline ---
    if new_chunks:
        # 1. Merge into workspace
        merge_fetched_chunks(workspace, new_chunks)
        # 2. Link chunks to entities (bidirectional name matching)
        link_chunks_to_entities(workspace, new_chunks)
        # 3. Delta detection: only summarize chunks not yet summarized
        delta_chunks = [
            c for c in new_chunks
            if c.chunk_id not in workspace._summarized_chunk_ids
        ]
        if delta_chunks:
            try:
                alias_ctx = _build_alias_context_from_lexicon(lexicon)
                ev_update = summarize_delta_chunks(
                    delta_chunks, question, alias_context=alias_ctx,
                )
                cdm = build_chunk_doc_map(workspace)
                merge_evidence_summary_update(workspace, ev_update, cdm)
                if verbose:
                    bullet_count = len(ev_update.bullets) if ev_update.bullets else 0
                    print(
                        f"[V10.2]   Summarizer: {bullet_count} bullets from "
                        f"{len(delta_chunks)} delta chunks",
                        flush=True,
                    )
                # Emit evidence_update so frontend Investigation panel shows bullets (V9 behavior)
                if progress_callback and ev_update.bullets:
                    try:
                        progress_callback(
                            "evidence_update",
                            "completed",
                            f"Discovered {len(ev_update.bullets)} new evidence bullets",
                            {
                                "bullets": [
                                    {
                                        "text": b.text,
                                        "tags": b.tags,
                                        "chunk_ids": b.supporting_chunk_ids,
                                        "doc_ids": b.doc_ids,
                                    }
                                    for b in ev_update.bullets
                                ],
                                "open_questions": getattr(ev_update, "open_questions", []) or [],
                                "leads": getattr(ev_update, "leads", []) or [],
                                "total_bullet_count": len(workspace._bullet_index),
                            },
                        )
                    except Exception:
                        pass
            except Exception as e_sum:
                logger.debug("Summarizer failed (non-fatal): %s", e_sum)

    return {"chunks": results}


def _tool_extract_mentions(
    conn,
    args: Dict[str, Any],
    lexicon: LexiconV10,
    all_mentions: Dict[int, ChunkMentionsV10],
    workspace: ResearchWorkspace,
    verbose: bool,
) -> Dict[str, Any]:
    """Extract mentions from a fetched chunk."""
    chunk_id = args.get("chunk_id")
    if chunk_id is None:
        return {"error": "No chunk_id provided"}

    # Get chunk from workspace (lookup by chunk_id in fulltext_chunks)
    wc = _workspace_chunk_lookup(workspace, chunk_id)
    if not wc:
        return {"error": f"Chunk {chunk_id} not fetched yet. Use fetch_chunks first."}

    # Need document_id and page_no — try to get from chunk metadata
    document_id = wc.doc_id or 0
    page_no = None
    if wc.page and wc.page.startswith("p"):
        try:
            page_no = int(wc.page[1:])
        except ValueError:
            pass

    cm = extract_chunk_mentions_v10(
        conn,
        chunk_id=chunk_id,
        collection_slug=wc.collection_slug or "",
        document_id=document_id,
        page_no=page_no,
        text=wc.text,
        lexicon=lexicon,
    )

    all_mentions[chunk_id] = cm
    update_from_mentions(conn, lexicon, cm)

    if verbose:
        print(f"[V10]   extract_mentions: {len(cm.mentions)} mentions, {len(cm.signals)} signals", flush=True)

    return {
        "chunk_id": chunk_id,
        "collection_slug": cm.collection_slug,
        "document_id": cm.document_id,
        "page_no": cm.page_no,
        "mentions": [m.to_dict() for m in cm.mentions[:20]],
        "signals": [s.to_dict() for s in cm.signals],
    }


# =============================================================================
# Lattice summary builder (structured JSON for LLM)
# =============================================================================

def _build_lattice_summary_json(
    lattice: SpanLatticeV10,
    selection: SpanSelection,
    lexicon: LexiconV10,
) -> Dict[str, Any]:
    """Build a structured JSON lattice summary for the LLM.

    Includes chosen spans, suppressed subspans, candidate entity_ids,
    surface_kind, resolution_status, norm_key, and alias-scoped vs global
    flags.  This is the deterministic anchor that prevents identity drift.
    """
    chosen_ids = set(selection.chosen_span_ids)
    chosen_spans = []
    suppressed_spans = []

    for span in lattice.spans:
        span_data: Dict[str, Any] = {
            "span_id": span.span_id,
            "text": span.text,
            "norm_key": span.norm_key,
            "start": span.start,
            "end": span.end,
            "span_kind": span.span_kind,
            "resolution_status": span.resolution_status,
            "candidates": [
                {
                    "entity_id": c.entity_id,
                    "canonical_name": c.canonical_name,
                    "surface_kind": c.surface_kind,
                    "score": c.score,
                    "alias_scoped": c.valid_collections != ["*"],
                    "valid_collections": c.valid_collections,
                    **({"prior_count": c.prior_count_global} if c.prior_count_global else {}),
                }
                for c in span.candidates[:5]
            ],
        }
        if span.span_id in chosen_ids:
            # Check if alias-typed
            is_alias_typed = any(
                a.get("span_id") == span.span_id
                for a in selection.alias_spans
            )
            span_data["is_alias_typed"] = is_alias_typed
            chosen_spans.append(span_data)
        else:
            span_data["suppressed_by"] = span.dominates
            suppressed_spans.append(span_data)

    # Entity hypotheses from span selection
    entity_hypotheses = []
    for eh in selection.entity_hypotheses:
        entity_hypotheses.append({
            "entity_id": eh.get("entity_id"),
            "canonical_name": eh.get("canonical_name", ""),
            "confidence": eh.get("confidence", 0.5),
            "from_span_ids": eh.get("from_span_ids", []),
        })

    return {
        "query": lattice.query,
        "chosen_spans": chosen_spans,
        "suppressed_spans": suppressed_spans[:5],  # limit size
        "entity_hypotheses": entity_hypotheses,
        "alias_spans_activated": len(selection.alias_spans),
    }


# =============================================================================
# Main entry point
# =============================================================================

def run_v10_query(
    conn,
    question: str,
    model: str = V10_MODEL,
    scope: Optional[ScopeFilter] = None,
    verbose: bool = True,
    max_investigation_rounds: Optional[int] = None,
    _resume_lexicon: Optional[LexiconV10] = None,
    _resume_lattice: Optional[SpanLatticeV10] = None,
    _resume_mentions: Optional[Dict[int, ChunkMentionsV10]] = None,
    progress_callback: Optional[Callable] = None,
) -> V10Result:
    """Run the V10.2 agentic-first pipeline.

    Stages:
    A. Query interpretation (SpanLattice -> LLM span selection)
    D. LLM-driven investigation loop (agent drives all retrieval)
       - Post-search enrichment, live lexicon, finalization

    When V10_AGENTIC_FIRST=1 (default), Stages B/C are bypassed:
    the LLM drives retrieval from round 1 with initial boosts and
    a structured lattice summary.

    Returns V10Result with full identity state for ThinkDeeper persistence.
    """
    client = _get_client()
    result = V10Result()

    all_hits: List[CatalogHitV10] = []
    all_mentions: Dict[int, ChunkMentionsV10] = _resume_mentions or {}
    chunks_fetched: Dict[int, WorkspaceChunk] = {}

    try:
        # --- Stage A: Query interpretation ---
        if _resume_lattice and _resume_lexicon:
            lattice = _resume_lattice
            selection = SpanSelection(
                chosen_span_ids=[s.span_id for s in lattice.spans]
            )
            lexicon = _resume_lexicon
            plan = ResolutionPlanV10(selected_spans=lattice.spans)
            # Sync index revision on resume
            _check_and_sync_index_revision(conn, lexicon)
        else:
            lattice, selection, lexicon, plan = _run_stage_a(
                conn, client, model, question, scope, verbose
            )

        result.lattice = lattice
        result.lexicon = lexicon
        result.plan = plan

        if V10_AGENTIC_FIRST:
            # --- V10.2: Agentic-first — skip Stages B/C ---
            if verbose:
                print("[V10.2] Agentic-first: bypassing deterministic Stages B/C", flush=True)

            # Backfill alias namespaces for entities already known from Stage A
            initial_entity_ids = set(lexicon.entities_in_play.keys())
            if initial_entity_ids:
                backfill_alias_namespace(conn, lexicon, initial_entity_ids)
                if verbose:
                    print(f"[V10.2]   Initial backfill for {len(initial_entity_ids)} entities", flush=True)

            # Compute initial recommended boosts
            initial_boosts = _compute_recommended_boosts(
                lexicon, conn, set(),
            )

            # Build structured lattice summary for LLM
            lattice_summary = _build_lattice_summary_json(lattice, selection, lexicon)

            # --- Debug: Stage A → PEM lane entity handoff ---
            if verbose:
                _eip = lexicon.entities_in_play
                _rr = lexicon.resolved_referents
                _hyps = getattr(selection, "entity_hypotheses", [])
                _amh = {
                    k: {"alias": h.alias_text, "status": h.status, "n_cands": len(h.candidates)}
                    for k, h in lexicon.alias_mapping_hypotheses.items()
                }
                _eip_str = ", ".join(
                    f"{eid}: {info.get('canonical_name', '?')}"
                    for eid, info in list(_eip.items())[:8]
                )
                print(f"[V10.2] PRE-PEM entities_in_play ({len(_eip)}): {{{_eip_str}}}", flush=True)
                print(f"[V10.2] PRE-PEM resolved_referents ({len(_rr)}): "
                      f"{json.dumps({k: v.get('entity_id') for k, v in list(_rr.items())[:8]}, default=str)}", flush=True)
                print(f"[V10.2] PRE-PEM selection.entity_hypotheses ({len(_hyps)}): "
                      f"{json.dumps(_hyps[:8], default=str)}", flush=True)
                print(f"[V10.2] PRE-PEM alias_mapping_hypotheses ({len(_amh)}): "
                      f"{json.dumps(dict(list(_amh.items())[:6]), default=str)}", flush=True)
                if not _eip and _hyps:
                    print("[V10.2] *** WARNING: entities_in_play is EMPTY but entity_hypotheses exist — "
                          "Stage A resolved hypotheses but didn't promote to entities_in_play. "
                          "PEM lane will have nothing to seed. ***", flush=True)

            # --- PEM Lane: deterministic alias-surface seeding ---
            pem_seed: Optional[PemLaneResult] = None
            try:
                pem_seed = pem_lane_seed_chunks(conn, lexicon, scope, question)
                if verbose and pem_seed.chunk_ids:
                    print(
                        f"[V10.2] PEM lane: {len(pem_seed.chunk_ids)} chunks seeded "
                        f"(surfaces={pem_seed.seeded_surfaces[:12]}, "
                        f"revision={pem_seed.pem_revision})",
                        flush=True,
                    )
                elif verbose:
                    print(
                        f"[V10.2] PEM lane: skipped ({pem_seed.reason_codes})",
                        flush=True,
                    )
            except Exception as e:
                logger.debug("PEM lane failed (non-fatal): %s", e)

            # --- Stage D: LLM investigation loop (from scratch) ---
            narrative, claims, trace, synth_workspace = _run_synthesis(
                conn, client, model, question, lexicon,
                all_hits, all_mentions, chunks_fetched,
                scope, verbose, progress_callback,
                lattice_summary=lattice_summary,
                initial_boosts=initial_boosts,
                pem_seed=pem_seed,
                max_investigation_rounds=max_investigation_rounds,
            )
        else:
            # --- Legacy V10.0: deterministic Stages B/C ---
            needs_alias = _needs_alias_resolution(selection)
            if verbose:
                print(f"[V10] Stage B: Alias resolution needed = {needs_alias}", flush=True)

            if needs_alias:
                alias_hits, alias_mentions = _run_alias_resolution(
                    conn, client, model, question, selection, lattice,
                    lexicon, plan, verbose,
                )
                all_hits.extend(alias_hits)
                all_mentions.update(alias_mentions)

            global_hits, global_mentions = _run_global_retrieval(
                conn, question, lexicon, scope, verbose,
            )
            existing_ids = {h.chunk_id for h in all_hits}
            for h in global_hits:
                if h.chunk_id not in existing_ids:
                    all_hits.append(h)
                    existing_ids.add(h.chunk_id)
            all_mentions.update(global_mentions)

            narrative, claims, trace, synth_workspace = _run_synthesis(
                conn, client, model, question, lexicon,
                all_hits, all_mentions, chunks_fetched,
                scope, verbose, progress_callback,
                max_investigation_rounds=max_investigation_rounds,
            )

        result.narrative = narrative
        result.claims = claims
        result.chunk_mentions = all_mentions
        result.investigation_trace = trace
        # Populate chunks_fetched from workspace for backward compatibility
        result.chunks_fetched = _workspace_chunks_dict(synth_workspace)
        result.lexicon = lexicon

        # Collect unresolved aliases
        for key, hyp in lexicon.alias_mapping_hypotheses.items():
            if hyp.status in ("unresolved", "ambiguous"):
                result.unresolved_aliases.append({
                    "alias": hyp.alias_text,
                    "collection": hyp.collection_slug,
                    "possible_entities": [c.canonical_name for c in hyp.candidates],
                    "document_id": hyp.document_id,
                })

        if verbose:
            print(f"[V10] Complete. Narrative length: {len(narrative)}, Claims: {len(claims)}", flush=True)

    except Exception as e:
        logger.error("V10 pipeline failed: %s", e, exc_info=True)
        result.narrative = f"V10 pipeline error: {e}"

    return result


# =============================================================================
# Live Lexicon Enrichment — helpers
# =============================================================================

MAX_ALIAS_BOOSTS = 50
MAX_LLM_MENTIONS_PER_ROUND = 2   # LLM extraction cap per search; PEM seed + top-1 only
EXTRACT_CONCURRENCY = 3          # in-flight LLM calls for mention extraction

# V10.2 compact response caps (reduce token bloat, V9-like speed)
BOOST_TOOL_ENTITY_TOP_K = 5
BOOST_TOOL_ALIAS_TOP_K = 10
BOOST_TOOL_BACKLINKS_TOP_K = 3
BOOST_TOOL_ALIASES_PER_BACKLINK = 5
SEARCH_RESULTS_SNIPPET_TOP_K = 8


def _batch_fetch_chunk_text(conn, chunk_ids: List[int]) -> Dict[int, str]:
    """Batch-fetch full text for chunks. Returns {chunk_id: text}."""
    if not chunk_ids:
        return {}
    result: Dict[int, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, COALESCE(clean_text, text)
                FROM chunks
                WHERE id = ANY(%s)
            """, (chunk_ids,))
            for row in cur.fetchall():
                result[row[0]] = row[1] or ""
    except Exception as e:
        logger.warning("Batch fetch chunk text failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return result


def enrich_hits_with_mentions(
    conn,
    client,
    hits: List[CatalogHitV10],
    lexicon: LexiconV10,
    all_mentions: Dict[int, ChunkMentionsV10],
    seen_doc_ids: Set[int],
    surfaces_cache: Dict,
    mode: str = "llm",
    max_alias_scoped: int = 20,
    max_other: int = 5,
    controller_mode: str = "explore",
    dsn: Optional[str] = None,
    verbose: bool = True,
) -> "EnrichmentSummary":
    """Post-search enrichment: extract mentions, update lexicon, backfill aliases.

    1. Batch-fetch text for ALL selected hits (single DB round-trip)
    2. Prioritize: origin=pem_seed first, then top-ranked search hits
    3. LLM extraction restricted to PEM seeds + top-1 search hit,
       capped at MAX_LLM_MENTIONS_PER_ROUND (2); everything else deterministic
    4. Run LLM extraction concurrently when dsn provided
    5. update_from_mentions() per chunk
    6. backfill_alias_namespace() for new entities
    7. Return EnrichmentSummary with timing fields

    Rationale: PEM annotation handles identity mapping (codename→entity);
    LLM extraction is reserved for relationship/roster discovery on the
    highest-value chunks.  This keeps round-1 latency close to V9.

    Per-doc cap: max MAX_CHUNKS_PER_DOC extracted from same document
    unless controller_mode == "exploit".
    """
    from collections import Counter

    import psycopg2
    from retrieval.agent.v10_lexicon import backfill_alias_namespace
    from retrieval.agent.v10_types import EnrichmentSummary

    # Clear any aborted transaction from prior ops (e.g. PEM merge, search)
    try:
        conn.rollback()
    except Exception:
        pass

    summary = EnrichmentSummary()
    extract_start = time.perf_counter()

    if verbose:
        print(f"[V10.2]   enrichment: {len(hits)} hits → selecting for extraction...", flush=True)

    # Separate alias-scoped vs other hits
    alias_hits = [h for h in hits if h.collection_slug in ALIAS_SCOPED_COLLECTIONS]
    other_hits = [h for h in hits if h.collection_slug not in ALIAS_SCOPED_COLLECTIONS]

    # Prioritize: origin=pem_seed first, then novelty (doc_unseen), then score
    def _alias_sort_key(h: CatalogHitV10):
        pem_first = 0 if getattr(h, "origin", None) == "pem_seed" else 1
        doc_unseen = 0 if (h.document_id and h.document_id not in seen_doc_ids) else 1
        return (pem_first, doc_unseen, -h.score)

    alias_hits.sort(key=_alias_sort_key)
    selected_alias = alias_hits[:max_alias_scoped]
    selected_other = sorted(other_hits, key=lambda h: -h.score)[:max_other]
    selected_hits = selected_alias + selected_other

    if not selected_hits:
        return summary

    # --- Batch-fetch text for ALL selected hits (one DB round-trip) ---
    if verbose:
        print(f"[V10.2]   enrichment: batch-fetching text for {len(selected_hits)} chunks...", flush=True)
    batch_start = time.perf_counter()
    all_chunk_ids = [h.chunk_id for h in selected_hits]
    full_texts = _batch_fetch_chunk_text(conn, all_chunk_ids)
    for hit in selected_hits:
        if hit.chunk_id in full_texts and full_texts[hit.chunk_id]:
            hit.snippet = full_texts[hit.chunk_id]
    summary.batch_load_ms = (time.perf_counter() - batch_start) * 1000
    if verbose:
        print(f"[V10.2]   enrichment: batch_load done in {summary.batch_load_ms:.0f}ms", flush=True)

    # --- Per-doc chunk cap ---
    doc_extract_count: Counter = Counter()
    enforce_doc_cap = controller_mode != "exploit"

    # --- Build extraction tasks (which hits, LLM vs deterministic) ---
    # LLM extraction is reserved for:
    #   - origin=pem_seed chunks (high-value alias pages from PEM lane)
    #   - top-1 search hit by score (highest-ranked result)
    # Everything else gets deterministic extraction + PEM annotation for identity.
    tasks: List[Tuple[CatalogHitV10, bool]] = []
    llm_count = 0
    # Identify the top-1 search hit (non-PEM, alias-scoped, highest score)
    top1_chunk_id: Optional[int] = None
    for h in selected_hits:
        if getattr(h, "origin", None) != "pem_seed" and h.collection_slug in ALIAS_SCOPED_COLLECTIONS:
            if h.chunk_id not in all_mentions:
                top1_chunk_id = h.chunk_id
                break  # selected_hits are already sorted by priority

    for hit in selected_hits:
        if hit.chunk_id in all_mentions:
            continue
        if enforce_doc_cap and hit.document_id:
            if doc_extract_count[hit.document_id] >= MAX_CHUNKS_PER_DOC:
                continue
            doc_extract_count[hit.document_id] += 1
        text = hit.snippet or ""
        if not text.strip():
            continue
        is_alias_scoped = hit.collection_slug in ALIAS_SCOPED_COLLECTIONS
        is_pem = getattr(hit, "origin", None) == "pem_seed"
        is_top1 = hit.chunk_id == top1_chunk_id
        # LLM only for PEM seeds and top-1, capped at MAX_LLM_MENTIONS_PER_ROUND
        use_llm = (
            is_alias_scoped
            and llm_count < MAX_LLM_MENTIONS_PER_ROUND
            and mode == "llm"
            and (is_pem or is_top1)
        )
        if use_llm:
            llm_count += 1
        tasks.append((hit, use_llm))

    llm_tasks = [(h, u) for h, u in tasks if u]
    det_tasks = [(h, u) for h, u in tasks if not u]
    llm_latencies_ms: List[float] = []

    if verbose:
        print(
            f"[V10.2]   enrichment: {len(tasks)} tasks (LLM: {len(llm_tasks)}, deterministic: {len(det_tasks)})",
            flush=True,
        )

    # --- Pre-scan all chunk texts for alias surfaces → batch lookup before extraction ---
    alias_pre_surfaces: Set[str] = set()
    for hit, _ in tasks:
        for s in _collect_alias_surfaces_from_text(
            hit.snippet or "", hit.collection_slug or "", lexicon
        ):
            alias_pre_surfaces.add(s)
    alias_table_cache: Dict[str, Any] = {}
    if alias_pre_surfaces and conn is not None:
        alias_start = time.perf_counter()
        from retrieval.agent.v10_resolve import _lookup_alias_table_batch
        alias_table_cache = _lookup_alias_table_batch(conn, list(alias_pre_surfaces))
        if verbose:
            print(
                f"[V10.2]   enrichment: alias pre-scan {len(alias_pre_surfaces)} surfaces, "
                f"batch lookup in {(time.perf_counter() - alias_start) * 1000:.0f}ms",
                flush=True,
            )

    def _extract_one(
        hit: CatalogHitV10,
        use_llm: bool,
        worker_conn,
    ) -> Tuple[CatalogHitV10, ChunkMentionsV10, bool, float]:
        t0 = time.perf_counter()
        text = hit.snippet or ""
        cm = extract_mentions_dispatched(
            worker_conn,
            client if use_llm else None,
            hit.chunk_id,
            hit.collection_slug or "",
            hit.document_id or 0,
            hit.page_no,
            text,
            lexicon,
            mode="llm" if use_llm else "deterministic",
            surfaces_cache=surfaces_cache,
            alias_table_cache=alias_table_cache or None,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return (hit, cm, use_llm, elapsed_ms)

    # --- Run LLM extraction (concurrent when dsn provided) ---
    use_concurrent = bool(dsn and llm_tasks and client)
    summary.llm_concurrent = use_concurrent
    summary.llm_call_count = len(llm_tasks)

    if verbose and (llm_tasks or det_tasks):
        if llm_tasks:
            print(
                f"[V10.2]   enrichment: LLM extraction {len(llm_tasks)} chunks "
                f"({'concurrent' if use_concurrent else 'sequential'})...",
                flush=True,
            )
        if det_tasks:
            print(f"[V10.2]   enrichment: deterministic extraction {len(det_tasks)} chunks...", flush=True)

    if use_concurrent:
        worker_conns: List[Any] = []
        try:
            for _ in range(min(EXTRACT_CONCURRENCY, len(llm_tasks))):
                worker_conns.append(psycopg2.connect(dsn))
            with ThreadPoolExecutor(max_workers=min(EXTRACT_CONCURRENCY, len(llm_tasks))) as ex:
                futures = {}
                for i, (hit, _) in enumerate(llm_tasks):
                    wconn = worker_conns[i % len(worker_conns)]
                    fut = ex.submit(_extract_one, hit, True, wconn)
                    futures[fut] = hit.chunk_id
                for fut in as_completed(futures):
                    try:
                        hit, cm, use_llm, elapsed_ms = fut.result()
                        llm_latencies_ms.append(elapsed_ms)
                        all_mentions[hit.chunk_id] = cm
                        summary.chunks_llm_extracted += 1
                        summary.chunks_extracted += 1
                        if hit.document_id:
                            seen_doc_ids.add(hit.document_id)
                    except Exception as e:
                        logger.debug("Concurrent extraction failed for chunk: %s", e)
        except Exception as e:
            logger.warning("Worker conn pool failed, falling back to sequential: %s", e)
            use_concurrent = False
            for hit, _ in llm_tasks:
                if hit.chunk_id not in all_mentions:
                    t0 = time.perf_counter()
                    cm = extract_mentions_dispatched(
                        conn, client, hit.chunk_id, hit.collection_slug or "",
                        hit.document_id or 0, hit.page_no, hit.snippet or "",
                        lexicon, mode="llm", surfaces_cache=surfaces_cache,
                        alias_table_cache=alias_table_cache or None,
                    )
                    llm_latencies_ms.append((time.perf_counter() - t0) * 1000)
                    all_mentions[hit.chunk_id] = cm
                    summary.chunks_llm_extracted += 1
                    summary.chunks_extracted += 1
                    if hit.document_id:
                        seen_doc_ids.add(hit.document_id)
        finally:
            for w in worker_conns:
                try:
                    w.close()
                except Exception:
                    pass

    # --- Run LLM+deterministic sequential when not concurrent ---
    if not use_concurrent:
        for hit, use_llm in llm_tasks + det_tasks:
            t0 = time.perf_counter()
            cm = extract_mentions_dispatched(
                conn,
                client if use_llm else None,
                hit.chunk_id,
                hit.collection_slug or "",
                hit.document_id or 0,
                hit.page_no,
                hit.snippet or "",
                lexicon,
                mode="llm" if use_llm else "deterministic",
                surfaces_cache=surfaces_cache,
                alias_table_cache=alias_table_cache or None,
            )
            if use_llm:
                llm_latencies_ms.append((time.perf_counter() - t0) * 1000)
                summary.chunks_llm_extracted += 1
            else:
                summary.chunks_deterministic += 1
            all_mentions[hit.chunk_id] = cm
            summary.chunks_extracted += 1
            if hit.document_id:
                seen_doc_ids.add(hit.document_id)

    # --- Run deterministic tasks when concurrent (LLM already ran in parallel) ---
    elif det_tasks:
        for hit, _ in det_tasks:
            cm = extract_mentions_dispatched(
                conn,
                None,
                hit.chunk_id,
                hit.collection_slug or "",
                hit.document_id or 0,
                hit.page_no,
                hit.snippet or "",
                lexicon,
                mode="deterministic",
                surfaces_cache=surfaces_cache,
                alias_table_cache=alias_table_cache or None,
            )
            summary.chunks_deterministic += 1
            all_mentions[hit.chunk_id] = cm
            summary.chunks_extracted += 1
            if hit.document_id:
                seen_doc_ids.add(hit.document_id)

    extracted_mentions = [all_mentions[h.chunk_id] for h, _ in tasks if h.chunk_id in all_mentions]

    if verbose:
        elapsed = (time.perf_counter() - extract_start) * 1000
        print(
            f"[V10.2]   enrichment: extraction done in {elapsed:.0f}ms "
            f"({summary.chunks_llm_extracted} LLM, {summary.chunks_deterministic} deterministic)",
            flush=True,
        )

    if llm_latencies_ms:
        s = sorted(llm_latencies_ms)
        n = len(s)
        summary.llm_latency_p50_ms = s[n // 2] if n else None
        summary.llm_latency_p95_ms = s[int(n * 0.95)] if n > 1 else (s[0] if s else None)
    summary.extract_total_ms = (time.perf_counter() - extract_start) * 1000

    # --- Merge any new alias surfaces from LLM extraction into cache for update_from_mentions ---
    if verbose and extracted_mentions:
        print(f"[V10.2]   enrichment: updating lexicon from {len(extracted_mentions)} chunks...", flush=True)
    if extracted_mentions and conn is not None:
        alias_set: Set[str] = set()
        for cm in extracted_mentions:
            if cm.collection_slug not in ALIAS_SCOPED_COLLECTIONS:
                continue
            for m in cm.mentions:
                if m.kind == "alias_surface" and m.surface:
                    alias_set.add(m.surface.lower().strip())
        missing = [s for s in alias_set if s not in alias_table_cache]
        if missing:
            from retrieval.agent.v10_resolve import _lookup_alias_table_batch
            extra = _lookup_alias_table_batch(conn, missing)
            alias_table_cache = {**alias_table_cache, **extra}

    # --- Post-extraction: update lexicon ---
    update_start = time.perf_counter()
    for cm in extracted_mentions:
        try:
            update_from_mentions(
                conn, lexicon, cm,
                alias_table_cache=alias_table_cache if alias_table_cache else None,
            )
        except Exception as e:
            logger.debug("update_from_mentions failed for chunk %d: %s", cm.chunk_id, e)
    if verbose and extracted_mentions:
        print(
            f"[V10.2]   enrichment: lexicon update done in {(time.perf_counter() - update_start) * 1000:.0f}ms",
            flush=True,
        )

    # --- Alias backfill for newly registered entities ---
    new_entity_ids = {
        eid for eid in lexicon.entities_in_play
        if eid not in lexicon._backfilled_entity_ids
    }
    if new_entity_ids:
        if verbose:
            print(f"[V10.2]   enrichment: backfilling {len(new_entity_ids)} new entities...", flush=True)
        backfill_start = time.perf_counter()
        backfill_result = backfill_alias_namespace(conn, lexicon, new_entity_ids)
        summary.aliases_backfilled = backfill_result
        if verbose:
            print(
                f"[V10.2]   enrichment: backfill done in {(time.perf_counter() - backfill_start) * 1000:.0f}ms",
                flush=True,
            )

    if verbose:
        total_ms = (time.perf_counter() - extract_start) * 1000
        print(f"[V10.2]   enrichment: complete in {total_ms:.0f}ms total", flush=True)

    return summary


def _compute_recommended_boosts(
    lexicon: LexiconV10,
    conn,
    seen_doc_ids: Set[int],
) -> Dict[str, Any]:
    """Compute recommended boosts from current lexicon state.

    - Entity boosts from entities_in_play
    - Alias backlink boosts (no locked_entity_id)
    - Bounded ambiguous alias boosts (MAX_ALIAS_BOOSTS cap, priority sort)
    """
    entity_boosts: List[Dict[str, Any]] = []
    alias_boosts: List[Dict[str, Any]] = []
    alias_backlinks: List[Dict[str, Any]] = []

    for eid, info in lexicon.entities_in_play.items():
        canonical = info.get("canonical_name", "")
        forms = build_entity_forms(conn, eid)
        if forms:
            entity_boosts.append({
                "entity_id": eid,
                "forms": forms[:8],
                "weight": 1.0 + min(0.3, len(info.get("evidence_chunk_ids", [])) * 0.05),
            })

        # Alias backlinks: codename aliases from backfill
        venona_aliases = lexicon.aliases_by_entity_scoped.get(eid, {}).get("venona", [])
        vassiliev_aliases = lexicon.aliases_by_entity_scoped.get(eid, {}).get("vassiliev", [])
        if venona_aliases or vassiliev_aliases:
            alias_backlinks.append({
                "entity_id": eid,
                "name": canonical,
                "venona_aliases": venona_aliases[:5],
                "vassiliev_aliases": vassiliev_aliases[:5],
            })
            # Auto-generate scoped alias boosts for backfilled aliases
            for alias in venona_aliases[:5]:
                alias_boosts.append({
                    "collection_slug": "venona",
                    "alias_text": alias,
                    "weight": 1.1,
                    # NO locked_entity_id -- alias may be reused
                })
            for alias in vassiliev_aliases[:5]:
                alias_boosts.append({
                    "collection_slug": "vassiliev",
                    "alias_text": alias,
                    "weight": 1.1,
                })

    # Bounded inclusion of unresolved/ambiguous aliases
    # HARDENING: only alias-scoped collections get alias boosts
    ambiguous_boosts = []
    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        if hyp.status in ("unresolved", "ambiguous"):
            # Only include alias boosts for alias-scoped collections
            if hyp.collection_slug not in ALIAS_SCOPED_COLLECTIONS:
                continue
            ambiguous_boosts.append({
                "collection_slug": hyp.collection_slug,
                "alias_text": hyp.alias_text,
                "weight": 1.1,
                "document_id": hyp.document_id,
            })

    ambiguous_boosts.sort(key=lambda b: (
        0 if b.get("document_id") in seen_doc_ids else 1,
        b.get("document_id") or 999999999,
    ))
    for b in ambiguous_boosts:
        if len(alias_boosts) >= MAX_ALIAS_BOOSTS:
            break
        alias_boosts.append({
            "collection_slug": b["collection_slug"],
            "alias_text": b["alias_text"],
            "weight": b["weight"],
        })

    return {
        "entity_boosts": entity_boosts,
        "alias_boosts_scoped": alias_boosts,
        "alias_backlinks": alias_backlinks,
    }


def _truncate_recommended_boosts_for_tool(boosts: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact boosts for tool result; full boosts kept server-side for telemetry."""
    return {
        "entity_boosts": boosts.get("entity_boosts", [])[:BOOST_TOOL_ENTITY_TOP_K],
        "alias_boosts_scoped": boosts.get("alias_boosts_scoped", [])[:BOOST_TOOL_ALIAS_TOP_K],
        "alias_backlinks": [
            {
                "entity_id": b["entity_id"],
                "name": b.get("name", ""),
                "venona_aliases": (b.get("venona_aliases") or [])[:BOOST_TOOL_ALIASES_PER_BACKLINK],
                "vassiliev_aliases": (b.get("vassiliev_aliases") or [])[:BOOST_TOOL_ALIASES_PER_BACKLINK],
            }
            for b in boosts.get("alias_backlinks", [])[:BOOST_TOOL_BACKLINKS_TOP_K]
        ],
    }


LEXICON_COMPACT_ENTITIES_MAX = 8
LEXICON_COMPACT_AMBIGUOUS_MAX = 5
LEXICON_COMPACT_GRANTS_MAX = 5


def build_lexicon_briefing(
    lexicon: LexiconV10,
    compact: bool = True,
    enrichment_delta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Build lexicon briefing. compact=True (default): delta + compact for speed."""
    if compact:
        return _build_lexicon_briefing_compact(lexicon, enrichment_delta)
    return _build_lexicon_briefing_full(lexicon)


def _build_lexicon_briefing_compact(
    lexicon: LexiconV10,
    enrichment_delta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Compact format: Entities (max 8), New this round, Ambiguous, Alias grants."""
    entities_in_play = []
    for eid, info in list(lexicon.entities_in_play.items())[:LEXICON_COMPACT_ENTITIES_MAX]:
        entities_in_play.append({
            "entity_id": eid,
            "name": info.get("canonical_name", ""),
            "support_chunks": len(info.get("evidence_chunk_ids", [])),
        })

    confirmed = []
    provisional = []
    ambiguous = []
    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        if hyp.is_contextual:
            entry = {"alias": hyp.alias_text, "collection": hyp.collection_slug, "doc": hyp.document_id}
            if hyp.status == "confirmed" and hyp.candidates:
                entry["entity_id"] = hyp.candidates[0].entity_id
                entry["name"] = hyp.candidates[0].canonical_name
                confirmed.append(entry)
            elif hyp.status == "provisional" and hyp.candidates:
                entry["entity_id"] = hyp.candidates[0].entity_id
                entry["name"] = hyp.candidates[0].canonical_name
                provisional.append(entry)
            else:
                ambiguous.append({"alias": hyp.alias_text, "collection": hyp.collection_slug, "candidates": len(hyp.candidates or [])})
        else:
            if hyp.status in ("unresolved", "ambiguous") and hyp.candidates:
                ambiguous.append({"alias": hyp.alias_text, "collection": hyp.collection_slug, "candidates": len(hyp.candidates)})

    # Alias grants = confirmed + provisional (use alias_boosts_scoped)
    grants = confirmed[:LEXICON_COMPACT_GRANTS_MAX] + provisional[:LEXICON_COMPACT_GRANTS_MAX]
    amb_short = ambiguous[:LEXICON_COMPACT_AMBIGUOUS_MAX]

    new_this_round = []
    if enrichment_delta:
        n = enrichment_delta.get("new_contextual_mappings", 0) or 0
        if n:
            new_this_round.append(f"{n} contextual mappings")
        n = enrichment_delta.get("new_general_hypotheses", 0) or 0
        if n:
            new_this_round.append(f"{n} general hypotheses")

    lines = []
    lines.append(f"**Entities in play ({len(entities_in_play)}):** " + "; ".join(
        f"{e['name']} (id={e['entity_id']}, {e['support_chunks']} chunks)" for e in entities_in_play
    ) or "none")
    if new_this_round:
        lines.append(f"**New this round:** {', '.join(new_this_round)}")
    if amb_short:
        lines.append(f"**Ambiguous:** " + "; ".join(
            f"{a.get('alias', '?')} in {a.get('collection', '?')} ({a.get('candidates', 0)} candidates)" for a in amb_short
        ))
    if grants:
        lines.append(f"**Alias grants:** " + "; ".join(
            f"{g.get('alias', '?')}→{g.get('name', '?')}" for g in grants[:LEXICON_COMPACT_GRANTS_MAX]
        ))

    briefing_json = {"entities": entities_in_play, "ambiguous": amb_short, "grants": grants}
    return briefing_json, "\n".join(lines) or "No identity state yet."


def _build_lexicon_briefing_full(lexicon: LexiconV10) -> Tuple[Dict[str, Any], str]:
    """Full JSON dump for get_lexicon_state_v10 tool."""
    confirmed_contextual: List[Dict[str, Any]] = []
    provisional: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []

    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        if hyp.is_contextual:
            entry = {
                "collection": hyp.collection_slug,
                "document_id": hyp.document_id,
                "page_from": hyp.page_from,
                "page_to": hyp.page_to,
                "alias": hyp.alias_text,
                "candidates": [
                    {"entity_id": c.entity_id, "name": c.canonical_name}
                    for c in hyp.candidates[:3]
                ],
                "support": hyp.support[:3],
            }
            if hyp.status == "confirmed":
                if hyp.candidates:
                    entry["entity_id"] = hyp.candidates[0].entity_id
                    entry["name"] = hyp.candidates[0].canonical_name
                confirmed_contextual.append(entry)
            elif hyp.status == "provisional":
                if hyp.candidates:
                    entry["entity_id"] = hyp.candidates[0].entity_id
                    entry["name"] = hyp.candidates[0].canonical_name
                provisional.append(entry)
            else:
                ambiguous.append(entry)
        else:
            if hyp.status in ("unresolved", "ambiguous") and hyp.candidates:
                ambiguous.append({
                    "collection": hyp.collection_slug,
                    "alias": hyp.alias_text,
                    "candidates": [
                        {"entity_id": c.entity_id, "name": c.canonical_name}
                        for c in hyp.candidates[:3]
                    ],
                })

    entities_in_play = []
    for eid, info in lexicon.entities_in_play.items():
        entities_in_play.append({
            "entity_id": eid,
            "name": info.get("canonical_name", ""),
            "support_chunks": len(info.get("evidence_chunk_ids", [])),
        })

    alias_backlinks = []
    for eid, info in lexicon.entities_in_play.items():
        venona_aliases = lexicon.aliases_by_entity_scoped.get(eid, {}).get("venona", [])
        vassiliev_aliases = lexicon.aliases_by_entity_scoped.get(eid, {}).get("vassiliev", [])
        if venona_aliases or vassiliev_aliases:
            alias_backlinks.append({
                "entity_id": eid,
                "name": info.get("canonical_name", ""),
                "venona_aliases": venona_aliases[:5],
                "vassiliev_aliases": vassiliev_aliases[:5],
                "scope_warning": "Use ONLY in Venona/Vassiliev searches as alias_boosts_scoped",
            })

    briefing_json = {
        "confirmed_contextual": confirmed_contextual,
        "provisional": provisional,
        "ambiguous": ambiguous,
        "entities_in_play": entities_in_play,
        "alias_backlinks": alias_backlinks,
    }

    # Build human-readable text
    text_parts = []
    if entities_in_play:
        ent_strs = []
        for e in entities_in_play[:10]:
            aliases_str = ""
            for bl in alias_backlinks:
                if bl["entity_id"] == e["entity_id"]:
                    all_aliases = bl.get("venona_aliases", []) + bl.get("vassiliev_aliases", [])
                    if all_aliases:
                        aliases_str = f", aliases[Venona/Vassiliev ONLY]: {', '.join(set(all_aliases)[:3])}"
                    break
            ent_strs.append(f"{e['name']} ({e['support_chunks']} chunks{aliases_str})")
        text_parts.append(f"Entities: {'; '.join(ent_strs)}.")

    if confirmed_contextual:
        conf_strs = []
        for c in confirmed_contextual[:5]:
            support_types = [s.get("signal_type", "mention") for s in c.get("support", [])]
            conf_strs.append(
                f"{c['alias']} = {c.get('name', '?')} ({c['collection']} doc {c.get('document_id', '?')}, "
                f"{', '.join(support_types) or 'mention'})"
            )
        text_parts.append(f"Confirmed: {'; '.join(conf_strs)}.")

    if ambiguous:
        amb_strs = []
        for a in ambiguous[:5]:
            n_cands = len(a.get("candidates", []))
            amb_strs.append(
                f"{a['alias']} in {a['collection']} ({n_cands} candidates)"
            )
        text_parts.append(f"Ambiguous: {'; '.join(amb_strs)}.")

    briefing_text = "\n".join(text_parts) or "No identity state yet."

    return briefing_json, briefing_text


# =============================================================================
# Formatting helper
# =============================================================================

def format_v10_result(result: V10Result, include_identity: bool = True) -> str:
    """Format V10 result for CLI display."""
    parts: List[str] = []

    # Main narrative / answer
    if result.narrative:
        parts.append("=" * 72)
        parts.append("ANSWER")
        parts.append("=" * 72)
        parts.append(result.narrative)

    # Grounded claims
    if result.claims:
        parts.append("")
        parts.append("-" * 72)
        parts.append(f"GROUNDED CLAIMS ({len(result.claims)})")
        parts.append("-" * 72)
        for i, c in enumerate(result.claims, 1):
            status = getattr(c, "status", "")
            claim_obj = getattr(c, "claim", None)
            text = getattr(claim_obj, "text", "") if claim_obj else str(c)
            line = f"  {i}. [{status}] {text}" if status else f"  {i}. {text}"
            parts.append(line)

    # Identity resolution summary
    if include_identity:
        parts.append("")
        parts.append("-" * 72)
        parts.append("IDENTITY RESOLUTION SUMMARY")
        parts.append("-" * 72)

        if result.lexicon:
            lex = result.lexicon
            parts.append(f"  Entities in play: {len(lex.entities_in_play)}")
            parts.append(f"  Scoped alias namespaces: {len(lex.aliases_by_entity_scoped)}")
            gen_hyps = [h for h in lex.alias_mapping_hypotheses.values() if not h.is_contextual]
            ctx_hyps = [h for h in lex.alias_mapping_hypotheses.values() if h.is_contextual]
            parts.append(f"  General hypotheses: {len(gen_hyps)}")
            parts.append(f"  Contextual hypotheses: {len(ctx_hyps)}")
            confirmed = [h for h in lex.alias_mapping_hypotheses.values() if h.status == "confirmed"]
            parts.append(f"  Confirmed mappings: {len(confirmed)}")
            if lex.alias_referent_rules:
                parts.append(f"  Referent rules loaded: {len(lex.alias_referent_rules)}")

        if result.unresolved_aliases:
            parts.append(f"\n  Unresolved aliases ({len(result.unresolved_aliases)}):")
            for ua in result.unresolved_aliases[:10]:
                parts.append(f"    - {ua.get('alias', '?')} in {ua.get('collection', '?')}: "
                             f"{', '.join(ua.get('possible_entities', []))}")

        if result.lattice:
            parts.append(f"\n  Span lattice: {len(result.lattice.spans)} spans")

        parts.append(f"  Chunks fetched: {len(result.chunks_fetched)}")
        parts.append(f"  Mentions extracted: {len(result.chunk_mentions)}")
        parts.append(f"  Tool calls: {result.tool_call_count}")

    if not parts:
        parts.append("(No V10 result)")

    return "\n".join(parts)
