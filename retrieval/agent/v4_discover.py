"""
V4 Discovery Loop - ChatGPT-like iterative retrieval for comprehensive evidence.

This module orchestrates the discovery loop:
1. Propose discovery actions (via 4o)
2. Execute actions deterministically (via tool registry)
3. Build observation from results
4. Update state and check stop conditions
5. Repeat until coverage is good or budgets hit

The discovery loop runs BETWEEN initial V3 retrieval and V4 publishing,
expanding evidence coverage while maintaining strict grounding.

Usage:
    from retrieval.agent.v4_discover import run_discovery
    
    expanded_chunks, discovery_trace = run_discovery(
        query="Who were members of the Silvermaster network?",
        initial_chunk_ids=[...],
        conn=conn,
    )
"""

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from retrieval.agent.tools import (
    ToolResult,
    hybrid_search_tool,
    vector_search_tool,
    lexical_search_tool,
    lexical_exact_tool,
    expand_aliases_tool,
)
from retrieval.agent.v3_evidence import EvidenceSet, EvidenceSpan, EvidenceBuilder
from retrieval.agent.v4_discovery_prompt import (
    DiscoveryAction,
    DiscoveryPlan,
    DiscoveryObservation,
    ToolSummary,
    SpanPreview,
    CoverageSignals,
    DISCOVERY_SYSTEM_PROMPT,
    DISCOVERY_ACTION_SCHEMA,
    build_discovery_plan_prompt,
    build_initial_plan_prompt,
)
from retrieval.agent.v4_discovery_metrics import (
    DiscoveryBudgets,
    CoverageMetrics,
    StopDecision,
    DEFAULT_BUDGETS,
    compute_coverage_metrics,
    evaluate_stop_conditions,
)


# =============================================================================
# Discovery State
# =============================================================================

@dataclass
class DiscoveryState:
    """
    State maintained across discovery rounds.
    
    Carries:
    - Query and constraints
    - Accumulated candidate chunk_ids (with scores)
    - Tool call history (for deduplication)
    - Discovered entities and aliases
    - Coverage metrics history
    """
    query: str
    constraints: Dict[str, Any]
    
    # Accumulated candidates
    candidate_chunk_ids: Set[int] = field(default_factory=set)
    chunk_scores: Dict[int, float] = field(default_factory=dict)
    
    # Entity discoveries
    discovered_entity_ids: Set[int] = field(default_factory=set)
    known_aliases: Dict[str, List[str]] = field(default_factory=dict)
    
    # Tool call tracking (for deduplication)
    seen_action_hashes: Set[str] = field(default_factory=set)
    action_history: List[DiscoveryAction] = field(default_factory=list)
    
    # Metrics history
    metrics_history: List[CoverageMetrics] = field(default_factory=list)
    
    def add_chunks(self, chunk_ids: List[int], scores: Dict[int, float]):
        """Add chunks to candidates, keeping best scores."""
        for cid in chunk_ids:
            self.candidate_chunk_ids.add(cid)
            if cid in scores:
                if cid not in self.chunk_scores or scores[cid] > self.chunk_scores[cid]:
                    self.chunk_scores[cid] = scores[cid]
    
    def add_action(self, action: DiscoveryAction):
        """Record an executed action."""
        self.seen_action_hashes.add(action.action_hash())
        self.action_history.append(action)
    
    def is_action_seen(self, action: DiscoveryAction) -> bool:
        """Check if action was already executed."""
        return action.action_hash() in self.seen_action_hashes
    
    def get_sorted_chunk_ids(self, limit: Optional[int] = None) -> List[int]:
        """Get chunk IDs sorted by score (best first)."""
        sorted_ids = sorted(
            self.candidate_chunk_ids,
            key=lambda cid: self.chunk_scores.get(cid, 0.0),
            reverse=True,
        )
        if limit:
            return sorted_ids[:limit]
        return sorted_ids
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "constraints": self.constraints,
            "candidate_chunk_count": len(self.candidate_chunk_ids),
            "discovered_entity_count": len(self.discovered_entity_ids),
            "action_count": len(self.action_history),
            "known_alias_count": sum(len(v) for v in self.known_aliases.values()),
        }


# =============================================================================
# Discovery Trace
# =============================================================================

@dataclass
class DiscoveryRound:
    """Record of a single discovery round."""
    round_num: int
    plan: DiscoveryPlan
    tool_results: List[ToolResult]
    observation: DiscoveryObservation
    metrics: CoverageMetrics
    elapsed_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "plan": self.plan.to_dict(),
            "tool_count": len(self.tool_results),
            "new_chunks": sum(r.metadata.get("new_count", 0) for r in self.tool_results),
            "coverage": self.metrics.to_dict(),
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class DiscoveryTrace:
    """Complete trace of discovery execution."""
    query: str
    rounds: List[DiscoveryRound]
    final_chunk_count: int
    final_coverage: CoverageMetrics
    stop_decision: StopDecision
    total_elapsed_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_chunk_count": self.final_chunk_count,
            "final_coverage": self.final_coverage.to_dict(),
            "stop_decision": self.stop_decision.to_dict(),
            "total_elapsed_ms": self.total_elapsed_ms,
        }


# =============================================================================
# Tool Execution (Discovery-specific wrappers)
# =============================================================================

def execute_discovery_action(
    action: DiscoveryAction,
    conn,
    state: DiscoveryState,
    verbose: bool = True,
) -> ToolResult:
    """
    Execute a single discovery action.
    
    Wraps existing tool implementations with discovery-specific metadata.
    """
    tool_name = action.tool
    params = action.params
    
    if verbose:
        print(f"      [{tool_name}] Executing...", file=sys.stderr)
        _log_action_params(tool_name, params)
    
    # Call appropriate tool
    if tool_name == "hybrid_search":
        result = hybrid_search_tool(
            conn,
            query=params.get("query", ""),
            top_k=params.get("top_k", 200),
            collections=params.get("collections"),
        )
    elif tool_name == "vector_search":
        result = vector_search_tool(
            conn,
            query=params.get("query", ""),
            top_k=params.get("top_k", 200),
            collections=params.get("collections"),
        )
    elif tool_name == "lexical_search":
        result = lexical_search_tool(
            conn,
            terms=params.get("terms", []),
            top_k=params.get("top_k", 200),
            collections=params.get("collections"),
        )
    elif tool_name == "lexical_exact":
        result = lexical_exact_tool(
            conn,
            term=params.get("term", ""),
            top_k=params.get("top_k", 200),
            collections=params.get("collections"),
        )
    elif tool_name == "concordance_expand":
        result = expand_aliases_tool(
            conn,
            term=params.get("term", ""),
            max_aliases=params.get("max_aliases", 25),
        )
    elif tool_name == "entity_lookup":
        result = _entity_lookup_tool(conn, params.get("name", ""))
    elif tool_name == "entity_surfaces":
        result = _entity_surfaces_tool(conn, params.get("entity_id", 0))
    elif tool_name == "entity_mentions":
        result = _entity_mentions_tool(
            conn, 
            params.get("entity_id", 0),
            params.get("top_k", 100),
        )
    elif tool_name == "co_mention_expand":
        result = _co_mention_expand_tool(
            conn,
            params.get("entity_id", 0),
            params.get("top_k", 20),
        )
    else:
        result = ToolResult(
            tool_name=tool_name,
            params=params,
            chunk_ids=[],
            scores={},
            metadata={"error": f"Unknown tool: {tool_name}"},
            elapsed_ms=0,
            success=False,
            error=f"Unknown tool: {tool_name}",
        )
    
    # Add metadata about new vs existing chunks
    if result.success:
        new_chunks = [cid for cid in result.chunk_ids if cid not in state.candidate_chunk_ids]
        result.metadata["new_count"] = len(new_chunks)
        result.metadata["total_count"] = len(result.chunk_ids)
        
        if verbose:
            _log_action_result(tool_name, result)
    elif verbose:
        print(f"        ERROR: {result.error}", file=sys.stderr)
    
    return result


def _log_action_params(tool_name: str, params: dict):
    """Log detailed parameters for discovery action."""
    if tool_name in ("hybrid_search", "vector_search"):
        print(f"        query=\"{params.get('query', '')[:60]}\"", file=sys.stderr)
    elif tool_name == "lexical_search":
        print(f"        terms={params.get('terms', [])}", file=sys.stderr)
    elif tool_name == "lexical_exact":
        print(f"        term=\"{params.get('term', '')}\"", file=sys.stderr)
    elif tool_name in ("entity_lookup", "concordance_expand"):
        print(f"        name/term=\"{params.get('name', params.get('term', ''))}\"", file=sys.stderr)
    elif tool_name in ("entity_surfaces", "entity_mentions", "co_mention_expand"):
        print(f"        entity_id={params.get('entity_id')}", file=sys.stderr)


def _log_action_result(tool_name: str, result: ToolResult):
    """Log detailed result from discovery action."""
    new_count = result.metadata.get("new_count", 0)
    total_count = result.metadata.get("total_count", len(result.chunk_ids))
    
    print(f"        -> {total_count} chunks ({new_count} new) in {result.elapsed_ms:.0f}ms", file=sys.stderr)
    
    # Tool-specific result logging
    if tool_name == "entity_lookup":
        if result.metadata.get("entity_id"):
            print(f"           Found: {result.metadata.get('canonical_name')} "
                  f"(id={result.metadata.get('entity_id')}, type={result.metadata.get('entity_type')})", 
                  file=sys.stderr)
        else:
            print(f"           Not found in entity database", file=sys.stderr)
    elif tool_name == "entity_surfaces":
        surfaces = result.metadata.get("surfaces", [])
        print(f"           Surfaces ({len(surfaces)}): {surfaces[:5]}{'...' if len(surfaces) > 5 else ''}", 
              file=sys.stderr)
    elif tool_name == "co_mention_expand":
        co_entities = result.metadata.get("co_entities", [])
        if co_entities:
            print(f"           Co-mentioned entities ({len(co_entities)}):", file=sys.stderr)
            for e in co_entities[:5]:
                print(f"             - {e.get('name')} (co-count={e.get('co_count')})", file=sys.stderr)
    elif tool_name == "concordance_expand":
        aliases = result.metadata.get("aliases", [])
        if aliases:
            print(f"           Aliases found ({len(aliases)}): {aliases[:5]}{'...' if len(aliases) > 5 else ''}", 
                  file=sys.stderr)
        else:
            print(f"           No aliases found in concordance", file=sys.stderr)


def _entity_lookup_tool(conn, name: str) -> ToolResult:
    """Look up entity by name."""
    start = time.time()
    
    try:
        with conn.cursor() as cur:
            # Try exact match first
            cur.execute(
                "SELECT id, canonical_name, entity_type FROM entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 1",
                (name,)
            )
            row = cur.fetchone()
            
            if not row:
                # Try alias match
                cur.execute("""
                    SELECT e.id, e.canonical_name, e.entity_type
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE LOWER(ea.alias) = LOWER(%s)
                    LIMIT 1
                """, (name,))
                row = cur.fetchone()
            
            elapsed = (time.time() - start) * 1000
            
            if row:
                return ToolResult(
                    tool_name="entity_lookup",
                    params={"name": name},
                    chunk_ids=[],  # No chunks
                    scores={},
                    metadata={
                        "entity_id": row[0],
                        "canonical_name": row[1],
                        "entity_type": row[2],
                    },
                    elapsed_ms=elapsed,
                    success=True,
                )
            else:
                return ToolResult(
                    tool_name="entity_lookup",
                    params={"name": name},
                    chunk_ids=[],
                    scores={},
                    metadata={"error": "Entity not found"},
                    elapsed_ms=elapsed,
                    success=False,
                    error=f"Entity not found: {name}",
                )
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="entity_lookup",
            params={"name": name},
            chunk_ids=[],
            scores={},
            metadata={},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def _entity_surfaces_tool(conn, entity_id: int) -> ToolResult:
    """Get all surface forms for an entity."""
    start = time.time()
    
    try:
        from retrieval.agent.entity_surfaces import EntitySurfaceIndex
        
        index = EntitySurfaceIndex(conn)
        surfaces = index.get_surfaces(entity_id)
        canonical = index.get_canonical_name(entity_id)
        
        elapsed = (time.time() - start) * 1000
        
        return ToolResult(
            tool_name="entity_surfaces",
            params={"entity_id": entity_id},
            chunk_ids=[],
            scores={},
            metadata={
                "canonical_name": canonical,
                "surfaces": list(surfaces),
                "surface_count": len(surfaces),
            },
            elapsed_ms=elapsed,
            success=True,
        )
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="entity_surfaces",
            params={"entity_id": entity_id},
            chunk_ids=[],
            scores={},
            metadata={},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def _entity_mentions_tool(conn, entity_id: int, top_k: int = 100) -> ToolResult:
    """Find all chunks mentioning an entity."""
    start = time.time()
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT em.chunk_id
                FROM entity_mentions em
                WHERE em.entity_id = %s
                ORDER BY em.chunk_id
                LIMIT %s
            """, (entity_id, top_k))
            
            chunk_ids = [row[0] for row in cur.fetchall()]
            
            elapsed = (time.time() - start) * 1000
            
            # Assign uniform scores
            scores = {cid: 0.5 for cid in chunk_ids}
            
            return ToolResult(
                tool_name="entity_mentions",
                params={"entity_id": entity_id, "top_k": top_k},
                chunk_ids=chunk_ids,
                scores=scores,
                metadata={"total_hits": len(chunk_ids)},
                elapsed_ms=elapsed,
                success=True,
            )
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="entity_mentions",
            params={"entity_id": entity_id},
            chunk_ids=[],
            scores={},
            metadata={},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


def _co_mention_expand_tool(conn, entity_id: int, top_k: int = 20) -> ToolResult:
    """Find entities that co-occur with a given entity."""
    start = time.time()
    
    try:
        with conn.cursor() as cur:
            # Find entities mentioned in same chunks
            cur.execute("""
                SELECT em2.entity_id, e.canonical_name, COUNT(*) as co_count
                FROM entity_mentions em1
                JOIN entity_mentions em2 ON em2.chunk_id = em1.chunk_id AND em2.entity_id != em1.entity_id
                JOIN entities e ON e.id = em2.entity_id
                WHERE em1.entity_id = %s
                GROUP BY em2.entity_id, e.canonical_name
                ORDER BY co_count DESC
                LIMIT %s
            """, (entity_id, top_k))
            
            co_entities = [
                {"entity_id": row[0], "name": row[1], "co_count": row[2]}
                for row in cur.fetchall()
            ]
            
            elapsed = (time.time() - start) * 1000
            
            # Also get chunk_ids where they co-occur
            entity_ids = [e["entity_id"] for e in co_entities[:5]]
            chunk_ids = []
            if entity_ids:
                cur.execute("""
                    SELECT DISTINCT em1.chunk_id
                    FROM entity_mentions em1
                    WHERE em1.entity_id = %s
                    AND EXISTS (
                        SELECT 1 FROM entity_mentions em2 
                        WHERE em2.chunk_id = em1.chunk_id AND em2.entity_id = ANY(%s)
                    )
                    LIMIT 50
                """, (entity_id, entity_ids))
                chunk_ids = [row[0] for row in cur.fetchall()]
            
            return ToolResult(
                tool_name="co_mention_expand",
                params={"entity_id": entity_id, "top_k": top_k},
                chunk_ids=chunk_ids,
                scores={cid: 0.4 for cid in chunk_ids},
                metadata={
                    "co_entities": co_entities,
                    "co_entity_count": len(co_entities),
                },
                elapsed_ms=elapsed,
                success=True,
            )
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return ToolResult(
            tool_name="co_mention_expand",
            params={"entity_id": entity_id},
            chunk_ids=[],
            scores={},
            metadata={},
            elapsed_ms=elapsed,
            success=False,
            error=str(e),
        )


# =============================================================================
# Discovery Plan Generation
# =============================================================================

def propose_discovery_plan(
    state: DiscoveryState,
    round_num: int,
    previous_observation: Optional[DiscoveryObservation] = None,
    verbose: bool = True,
) -> DiscoveryPlan:
    """
    Call 4o to propose discovery actions for the next round.
    
    Args:
        state: Current discovery state
        round_num: Current round number
        previous_observation: Observation from previous round
        verbose: Print progress
    
    Returns:
        DiscoveryPlan with proposed actions
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: generate default plan
        return _default_discovery_plan(state, round_num)
    
    # Build prompt
    if round_num == 1:
        prompt = build_initial_plan_prompt(
            query=state.query,
            constraints=state.constraints,
        )
    else:
        prompt = build_discovery_plan_prompt(
            query=state.query,
            round_num=round_num,
            previous_observation=previous_observation,
            constraints=state.constraints,
            seen_action_hashes=state.seen_action_hashes,
        )
    
    if verbose:
        print(f"  [Plan] Generating discovery plan...", file=sys.stderr)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        model = os.getenv("OPENAI_MODEL_V4_PLANNER", "gpt-4o-mini")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500,
        )
        
        content = response.choices[0].message.content
        if not content:
            return _default_discovery_plan(state, round_num)
        
        data = json.loads(content)
        
        # Parse actions
        actions = []
        for action_data in data.get("actions", []):
            action = DiscoveryAction.from_dict(action_data)
            # Validate
            errors = action.validate()
            if errors:
                if verbose:
                    print(f"    Skipping invalid action: {errors}", file=sys.stderr)
                continue
            # Skip duplicates
            if state.is_action_seen(action):
                if verbose:
                    print(f"    Skipping duplicate action: {action.tool}", file=sys.stderr)
                continue
            actions.append(action)
        
        if verbose:
            print(f"    Proposed {len(actions)} actions:", file=sys.stderr)
            for a in actions:
                params_summary = _summarize_params_for_log(a.params)
                print(f"      - {a.tool}({params_summary})", file=sys.stderr)
                if a.rationale:
                    print(f"        Rationale: {a.rationale[:60]}", file=sys.stderr)
        
        return DiscoveryPlan(
            round_num=round_num,
            actions=actions,
            stop=data.get("stop", False),
            stop_reason=data.get("stop_reason", ""),
        )
        
    except Exception as e:
        if verbose:
            print(f"    Plan generation error: {e}", file=sys.stderr)
        return _default_discovery_plan(state, round_num)


def _default_discovery_plan(state: DiscoveryState, round_num: int) -> DiscoveryPlan:
    """Generate a default discovery plan when LLM is unavailable."""
    actions = []
    
    if round_num == 1:
        # Initial broad search
        actions.append(DiscoveryAction(
            tool="hybrid_search",
            params={"query": state.query, "top_k": 200},
            rationale="Initial broad search",
        ))
    else:
        # Subsequent rounds: try lexical variants
        actions.append(DiscoveryAction(
            tool="lexical_search",
            params={"terms": state.query.split()[:3], "top_k": 100},
            rationale="Lexical search with key terms",
        ))
    
    return DiscoveryPlan(
        round_num=round_num,
        actions=actions,
        stop=False,
        stop_reason="",
    )


# =============================================================================
# Observation Building
# =============================================================================

def build_observation(
    state: DiscoveryState,
    round_num: int,
    tool_results: List[ToolResult],
    evidence_set: EvidenceSet,
    metrics: CoverageMetrics,
) -> DiscoveryObservation:
    """
    Build observation from round results for next planning.
    
    Args:
        state: Current discovery state
        round_num: Completed round number
        tool_results: Results from executed tools
        evidence_set: Current evidence set
        metrics: Coverage metrics
    
    Returns:
        DiscoveryObservation for next round's prompt
    """
    # Build tool summaries
    tool_summaries = []
    for result in tool_results:
        summary = ToolSummary(
            tool=result.tool_name,
            params_summary=_summarize_params(result.params),
            chunk_count=len(result.chunk_ids),
            new_chunk_count=result.metadata.get("new_count", 0),
            entity_ids=result.metadata.get("entity_ids", []),
            top_docs=result.metadata.get("top_docs", []),
            elapsed_ms=result.elapsed_ms,
            error=result.error if not result.success else None,
        )
        tool_summaries.append(summary)
    
    # Build span previews
    span_previews = []
    for i, span in enumerate(evidence_set.cite_spans[:10]):
        span_previews.append(SpanPreview(
            span_idx=i,
            doc_id=span.doc_id,
            page_ref=span.page_ref,
            quote_preview=span.quote[:150].strip(),
            score=span.score,
        ))
    
    # Build coverage signals
    coverage = CoverageSignals(
        entity_attest_counts=metrics.entity_attest_counts,
        list_like_span_count=metrics.list_like_span_count,
        definitional_span_count=metrics.definitional_span_count,
        doc_concentration=metrics.doc_concentration,
        unique_docs=metrics.unique_docs,
        unique_pages=metrics.unique_pages,
        total_spans=metrics.total_spans,
        marginal_gain_pct=metrics.marginal_gain * 100,
    )
    
    # Candidate entities (from tool results)
    candidate_entities = []
    for result in tool_results:
        if result.tool_name == "co_mention_expand":
            for e in result.metadata.get("co_entities", []):
                candidate_entities.append(e)
        elif result.tool_name == "entity_lookup" and result.success:
            candidate_entities.append({
                "id": result.metadata.get("entity_id"),
                "name": result.metadata.get("canonical_name"),
                "mention_count": 0,
            })
    
    return DiscoveryObservation(
        round_num=round_num,
        tool_summaries=tool_summaries,
        coverage=coverage,
        span_previews=span_previews,
        candidate_entities=candidate_entities,
        known_aliases=dict(state.known_aliases),
    )


def _summarize_params(params: Dict[str, Any]) -> str:
    """Create short summary of tool params for observation."""
    if "query" in params:
        q = params["query"]
        return f'"{q[:40]}..."' if len(q) > 40 else f'"{q}"'
    if "term" in params:
        return f'"{params["term"]}"'
    if "terms" in params:
        return f'{params["terms"][:3]}'
    if "entity_id" in params:
        return f'entity:{params["entity_id"]}'
    if "name" in params:
        return f'"{params["name"]}"'
    return str(params)[:50]


def _summarize_params_for_log(params: Dict[str, Any]) -> str:
    """Create detailed summary for logging."""
    parts = []
    if "query" in params:
        q = params["query"][:50]
        parts.append(f'query="{q}"')
    if "term" in params:
        parts.append(f'term="{params["term"]}"')
    if "terms" in params:
        parts.append(f'terms={params["terms"]}')
    if "entity_id" in params:
        parts.append(f'entity_id={params["entity_id"]}')
    if "name" in params:
        parts.append(f'name="{params["name"]}"')
    if "top_k" in params:
        parts.append(f'top_k={params["top_k"]}')
    return ", ".join(parts) if parts else str(params)[:60]


# =============================================================================
# Main Discovery Loop
# =============================================================================

def run_discovery(
    query: str,
    initial_chunk_ids: List[int],
    conn,
    constraints: Optional[Dict[str, Any]] = None,
    budgets: DiscoveryBudgets = DEFAULT_BUDGETS,
    verbose: bool = True,
) -> Tuple[List[int], DiscoveryTrace]:
    """
    Run the discovery loop to expand evidence coverage.
    
    Args:
        query: User's research query
        initial_chunk_ids: Chunks from initial V3 retrieval
        conn: Database connection
        constraints: Query constraints (collections, etc.)
        budgets: Budget configuration
        verbose: Print progress
    
    Returns:
        (expanded_chunk_ids, trace) tuple
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[Discovery Loop] Starting for: {query[:50]}...", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
    
    # Initialize state
    state = DiscoveryState(
        query=query,
        constraints=constraints or {},
    )
    
    # Add initial chunks
    initial_scores = {cid: 1.0 for cid in initial_chunk_ids}
    state.add_chunks(initial_chunk_ids, initial_scores)
    
    # Build initial evidence for metrics
    evidence_builder = EvidenceBuilder(
        cite_cap=budgets.max_cite_spans,
        harvest_cap=budgets.max_cite_spans,
        verbose=False,
    )
    
    rounds: List[DiscoveryRound] = []
    prev_metrics: Optional[CoverageMetrics] = None
    observation: Optional[DiscoveryObservation] = None
    
    # Discovery loop
    for round_num in range(1, budgets.max_rounds + 1):
        round_start = time.time()
        
        if verbose:
            print(f"\n[Discovery Round {round_num}/{budgets.max_rounds}]", file=sys.stderr)
        
        # 1. Generate plan
        plan = propose_discovery_plan(
            state=state,
            round_num=round_num,
            previous_observation=observation,
            verbose=verbose,
        )
        
        # Check if model wants to stop
        if plan.stop:
            if verbose:
                print(f"  [Stop] Model requested stop: {plan.stop_reason}", file=sys.stderr)
            break
        
        # 2. Execute actions
        if verbose:
            print(f"  [Execute] Running {len(plan.actions)} actions...", file=sys.stderr)
        
        tool_results = []
        for action in plan.actions:
            result = execute_discovery_action(action, conn, state, verbose=verbose)
            tool_results.append(result)
            
            # Update state
            state.add_action(action)
            if result.success:
                state.add_chunks(result.chunk_ids, result.scores)
                
                # Track entity discoveries
                if "entity_id" in result.metadata:
                    state.discovered_entity_ids.add(result.metadata["entity_id"])
                if "co_entities" in result.metadata:
                    for e in result.metadata["co_entities"]:
                        state.discovered_entity_ids.add(e["entity_id"])
                if "aliases" in result.metadata:
                    term = action.params.get("term", "")
                    if term:
                        state.known_aliases[term] = result.metadata["aliases"]
            
            if verbose:
                if result.success:
                    new_count = result.metadata.get("new_count", 0)
                    print(f"    [{action.tool}] -> {len(result.chunk_ids)} chunks ({new_count} new)", 
                          file=sys.stderr)
                else:
                    print(f"    [{action.tool}] -> ERROR: {result.error}", file=sys.stderr)
        
        # 3. Rebuild evidence
        current_chunks = state.get_sorted_chunk_ids(limit=budgets.max_candidate_chunks)
        evidence_set = evidence_builder.build(
            chunk_ids=current_chunks,
            query=query,
            conn=conn,
            scores=state.chunk_scores,
        )
        
        # 4. Compute metrics
        metrics = compute_coverage_metrics(
            evidence_set=evidence_set,
            prev_metrics=prev_metrics,
        )
        state.metrics_history.append(metrics)
        
        if verbose:
            print(f"  [Coverage] {metrics.total_spans} spans, {metrics.unique_docs} docs, "
                  f"{metrics.list_like_span_count} list-like, {metrics.marginal_gain:.1%} gain", 
                  file=sys.stderr)
        
        # 5. Build observation for next round
        observation = build_observation(
            state=state,
            round_num=round_num,
            tool_results=tool_results,
            evidence_set=evidence_set,
            metrics=metrics,
        )
        
        round_elapsed = (time.time() - round_start) * 1000
        
        rounds.append(DiscoveryRound(
            round_num=round_num,
            plan=plan,
            tool_results=tool_results,
            observation=observation,
            metrics=metrics,
            elapsed_ms=round_elapsed,
        ))
        
        # 6. Check stop conditions
        stop_decision = evaluate_stop_conditions(
            metrics=metrics,
            round_num=round_num,
            budgets=budgets,
        )
        
        if stop_decision.should_stop:
            if verbose:
                print(f"  [Stop] {stop_decision.reason}", file=sys.stderr)
            break
        
        prev_metrics = metrics
    
    # Final results
    final_chunks = state.get_sorted_chunk_ids(limit=budgets.max_candidate_chunks)
    total_elapsed = (time.time() - start_time) * 1000
    
    # Get final metrics
    final_metrics = state.metrics_history[-1] if state.metrics_history else CoverageMetrics(
        total_spans=0, total_chunks=0, unique_docs=0, unique_pages=0,
        list_like_span_count=0, definitional_span_count=0,
        entity_attest_counts={}, doc_concentration={},
    )
    
    final_stop = stop_decision if 'stop_decision' in dir() else StopDecision(
        should_stop=True, reason="Loop completed", confidence="high"
    )
    
    trace = DiscoveryTrace(
        query=query,
        rounds=rounds,
        final_chunk_count=len(final_chunks),
        final_coverage=final_metrics,
        stop_decision=final_stop,
        total_elapsed_ms=total_elapsed,
    )
    
    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[Discovery Complete] {len(rounds)} rounds, {len(final_chunks)} chunks", file=sys.stderr)
        print(f"  Coverage: {final_metrics.total_spans} spans, "
              f"{final_metrics.list_like_span_count} list-like, "
              f"{final_metrics.definitional_span_count} definitional", file=sys.stderr)
        print(f"  Time: {total_elapsed:.0f}ms", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
    
    # Persist discovery artifacts (optional)
    if conn:
        _persist_discovery_trace(
            conn=conn,
            trace=trace,
            state=state,
            verbose=verbose,
        )
    
    return final_chunks, trace


def _persist_discovery_trace(
    conn,
    trace: DiscoveryTrace,
    state: DiscoveryState,
    verbose: bool = True,
):
    """
    Persist discovery artifacts to database for audit trail.
    
    Saves:
    - Tool calls with timing and results
    - State snapshots per round
    """
    try:
        # Generate run_id from trace
        run_id = hashlib.sha256(f"{trace.query}:{trace.total_elapsed_ms}".encode()).hexdigest()[:16]
        
        with conn.cursor() as cur:
            # Save tool calls
            for round_record in trace.rounds:
                for i, result in enumerate(round_record.tool_results):
                    cur.execute("""
                        INSERT INTO v4_discovery_tool_calls 
                        (run_id, round_num, step_num, tool_name, params, action_hash, 
                         chunk_count, new_chunk_count, elapsed_ms, success, error)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        run_id,
                        round_record.round_num,
                        i,
                        result.tool_name,
                        json.dumps(result.params),
                        hashlib.sha256(f"{result.tool_name}:{json.dumps(result.params, sort_keys=True)}".encode()).hexdigest()[:16],
                        len(result.chunk_ids),
                        result.metadata.get("new_count", 0),
                        result.elapsed_ms,
                        result.success,
                        result.error,
                    ))
                
                # Save state snapshot
                cur.execute("""
                    INSERT INTO v4_discovery_states
                    (run_id, round_num, candidate_chunk_count, discovered_entity_count, coverage_metrics)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    run_id,
                    round_record.round_num,
                    len(state.candidate_chunk_ids),
                    len(state.discovered_entity_ids),
                    json.dumps(round_record.metrics.to_dict()),
                ))
            
            conn.commit()
            
    except Exception as e:
        # Don't fail discovery on persistence errors
        if verbose:
            print(f"  [Discovery] Warning: Could not persist trace: {e}", file=sys.stderr)
        try:
            conn.rollback()
        except:
            pass
