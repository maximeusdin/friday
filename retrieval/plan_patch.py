"""
Plan Patch - LLM-generated action patches for the agentic controller.

The LLM outputs a structured PlanPatch that tells the deterministic
executor what operations to perform.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


class ActionOp(str, Enum):
    """Valid patch operations."""
    EXPAND_TERMS = "expand_terms"
    RERUN_RETRIEVAL = "rerun_retrieval"
    BUILD_FOCUS_BUNDLE = "build_focus_bundle"
    RENDER = "render"
    RETURN_NEGATIVE = "return_negative"
    DROP_BULLETS = "drop_bullets"
    WEAKEN_LANGUAGE = "weaken_language"
    SWAP_CITATIONS = "swap_citations"


@dataclass
class Action:
    """
    A single operation in a PlanPatch.
    
    Operations are executed in order by the deterministic executor.
    Parameters are stored flat (not nested) for OpenAI strict mode compatibility.
    """
    op: str  # ActionOp value
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "params": self.params}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        op = data.get("op", "")
        # Extract params from flat structure (OpenAI strict mode)
        params = {}
        for key in ["terms", "lanes", "top_n", "require_anchor_hits", 
                    "preference", "reason", "indices", "prefix"]:
            if key in data and data[key] is not None:
                params[key] = data[key]
        # Also check for nested params (backward compatibility)
        if "params" in data and isinstance(data["params"], dict):
            params.update(data["params"])
        return cls(op=op, params=params)


@dataclass
class PlanPatch:
    """
    LLM-generated patch specifying what actions to take.
    
    The LLM reads observations and outputs this structured patch.
    Code then executes it deterministically.
    """
    actions: List[Action] = field(default_factory=list)
    reasoning: str = ""  # Brief explanation for audit
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "reasoning": self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanPatch":
        actions = [Action.from_dict(a) for a in data.get("actions", [])]
        return cls(actions=actions, reasoning=data.get("reasoning", ""))
    
    def has_retrieval(self) -> bool:
        """Check if patch includes retrieval rerun."""
        return any(a.op == ActionOp.RERUN_RETRIEVAL.value for a in self.actions)
    
    def has_expansion(self) -> bool:
        """Check if patch includes term expansion."""
        return any(a.op == ActionOp.EXPAND_TERMS.value for a in self.actions)
    
    def is_negative_report(self) -> bool:
        """Check if patch is a negative evidence report."""
        return any(a.op == ActionOp.RETURN_NEGATIVE.value for a in self.actions)


@dataclass
class ExecutionState:
    """
    Mutable state passed through patch execution.
    
    Each action can read and modify this state.
    """
    # Query terms
    anchor_terms: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)
    
    # Retrieval results
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # FocusBundle
    focus_bundle: Any = None  # FocusBundle
    
    # Candidates and assessments
    candidates: List[Any] = field(default_factory=list)
    assessments: List[Any] = field(default_factory=list)
    hubness_scores: List[Any] = field(default_factory=list)
    
    # Rendered answer
    answer: Any = None  # RenderedAnswer
    
    # Verification result
    verification: Any = None  # VerificationResult
    
    # Retrieval config (modified by actions)
    retrieval_lanes: Dict[str, int] = field(default_factory=dict)
    require_anchor_hits: bool = False
    focus_bundle_config: Dict[str, Any] = field(default_factory=dict)
    render_preference: List[str] = field(default_factory=list)
    
    # Metadata
    action_log: List[str] = field(default_factory=list)


# JSON Schema for OpenAI Structured Outputs
# Note: strict mode requires additionalProperties: false everywhere
# We use a flexible params structure with known optional fields
PLAN_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "expand_terms",
                            "rerun_retrieval",
                            "build_focus_bundle",
                            "render",
                            "return_negative",
                            "drop_bullets",
                            "weaken_language",
                            "swap_citations"
                        ],
                        "description": "The operation to perform"
                    },
                    "terms": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "For expand_terms: terms to add"
                    },
                    "lanes": {
                        "type": ["object", "null"],
                        "properties": {
                            "hybrid_rrf": {"type": ["integer", "null"]},
                            "lexical_must_hit": {"type": ["integer", "null"]}
                        },
                        "required": ["hybrid_rrf", "lexical_must_hit"],
                        "additionalProperties": False,
                        "description": "For rerun_retrieval: lane budgets"
                    },
                    "top_n": {
                        "type": ["integer", "null"],
                        "description": "For build_focus_bundle: max spans"
                    },
                    "require_anchor_hits": {
                        "type": ["boolean", "null"],
                        "description": "For build_focus_bundle: require anchor hits"
                    },
                    "preference": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "For render: preference order"
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "For return_negative: explanation"
                    },
                    "indices": {
                        "type": ["array", "null"],
                        "items": {"type": "integer"},
                        "description": "For drop_bullets/weaken_language: bullet indices"
                    },
                    "prefix": {
                        "type": ["string", "null"],
                        "description": "For weaken_language: hedging prefix"
                    }
                },
                "required": ["op", "terms", "lanes", "top_n", "require_anchor_hits", "preference", "reason", "indices", "prefix"],
                "additionalProperties": False
            },
            "description": "Ordered list of actions to execute"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation for the chosen actions"
        }
    },
    "required": ["actions", "reasoning"],
    "additionalProperties": False
}


# Documentation of action parameters for LLM prompt
# Note: Parameters are FLAT in the action object, not nested under "params"
ACTION_PARAM_DOCS = """
Available operations and their parameters (parameters are FLAT, not nested):

1. expand_terms
   Add synonym/variant terms to the search.
   {"op": "expand_terms", "terms": ["vt fuse", "proximity fuze", "radio proximity"], ...}
   
2. rerun_retrieval
   Re-run retrieval with different lane configuration.
   {"op": "rerun_retrieval", "lanes": {"hybrid_rrf": 200, "lexical_must_hit": 200}, ...}
   
3. build_focus_bundle
   Build FocusBundle with specific configuration.
   {"op": "build_focus_bundle", "top_n": 120, "require_anchor_hits": true, ...}
   
4. render
   Render the answer with specified preference.
   {"op": "render", "preference": ["evidence_spans", "entities"], ...}
   
5. return_negative
   Return a negative evidence report (no results found).
   {"op": "return_negative", "reason": "No evidence found for anchor terms", ...}
   
6. drop_bullets (fix patch only)
   Remove specific bullets that failed verification.
   {"op": "drop_bullets", "indices": [0, 2], ...}
   
7. weaken_language (fix patch only)
   Add hedging to specific bullets.
   {"op": "weaken_language", "indices": [1], "prefix": "Possibly", ...}

IMPORTANT: Set unused fields to null. Example:
{"op": "expand_terms", "terms": ["vt fuse"], "lanes": null, "top_n": null, "require_anchor_hits": null, "preference": null, "reason": null, "indices": null, "prefix": null}
"""


def create_default_patch() -> PlanPatch:
    """Create a default patch for straightforward queries."""
    return PlanPatch(
        actions=[
            Action(op=ActionOp.RERUN_RETRIEVAL.value, params={
                "lanes": {"hybrid_rrf": 200, "lexical_must_hit": None}
            }),
            Action(op=ActionOp.BUILD_FOCUS_BUNDLE.value, params={
                "top_n": 80,
                "require_anchor_hits": False
            }),
            Action(op=ActionOp.RENDER.value, params={
                "preference": ["evidence_spans", "entities"]
            }),
        ],
        reasoning="Default execution path for standard query"
    )


def create_expansion_patch(terms: List[str]) -> PlanPatch:
    """Create a patch that expands terms and re-runs retrieval."""
    return PlanPatch(
        actions=[
            Action(op=ActionOp.EXPAND_TERMS.value, params={"terms": terms}),
            Action(op=ActionOp.RERUN_RETRIEVAL.value, params={
                "lanes": {"hybrid_rrf": 200, "lexical_must_hit": 200}
            }),
            Action(op=ActionOp.BUILD_FOCUS_BUNDLE.value, params={
                "top_n": 80,
                "require_anchor_hits": True
            }),
            Action(op=ActionOp.RENDER.value, params={
                "preference": ["evidence_spans"]
            }),
        ],
        reasoning="Expanding terms due to 0 anchor hits in probe"
    )


def create_negative_patch(reason: str) -> PlanPatch:
    """Create a patch that returns a negative evidence report."""
    return PlanPatch(
        actions=[
            Action(op=ActionOp.RETURN_NEGATIVE.value, params={"reason": reason}),
        ],
        reasoning="No evidence found after expansion attempts"
    )


# =============================================================================
# LLM Call for Plan Patch
# =============================================================================

def build_plan_patch_prompt(
    query_analysis: "QueryAnalysis",
    observations: "ObservationBundle",
) -> str:
    """Build prompt for plan patch LLM call."""
    from retrieval.query_analysis import QueryAnalysis
    from retrieval.observations import ObservationBundle
    
    return f"""Based on the probe retrieval results, decide what actions to take.

=== ORIGINAL QUERY ===
{query_analysis.query_text}

=== QUERY ANALYSIS ===
Core concepts: {query_analysis.core_concepts}
Anchor terms: {query_analysis.anchor_terms}
Scope: {query_analysis.scope_filters or "all collections"}
Do not anchor: {query_analysis.do_not_anchor}
Suggested synonyms: {query_analysis.suggested_synonyms}

=== PROBE OBSERVATIONS ===
{observations.to_prompt_string()}

=== AVAILABLE ACTIONS ===
{ACTION_PARAM_DOCS}

=== DECISION GUIDANCE ===

1. If anchor_hit_count is 0 (no anchor hits):
   - This is a critical signal - your anchor terms may be wrong
   - Consider: expand_terms with synonyms/variants
   - Consider: the term may not exist in the corpus (return_negative)
   - DO NOT proceed with rendering if no anchor hits

2. If red flags include "retrieval drift":
   - Scores are low and flat - semantic search found unrelated content
   - Consider: expand_terms with more specific terminology
   - Consider: rerun_retrieval with lexical_must_hit lane

3. If red flags include "candidate quality" issues:
   - Top candidates are stopwords or generic terms
   - This means the FocusBundle contains off-topic content
   - Consider: rerun_retrieval with tighter filters
   - Consider: render preference = evidence_spans (not entities)

4. If probe looks good (anchor hits > 0, no major red flags):
   - Proceed with standard execution
   - rerun_retrieval with full budget (k=200)
   - build_focus_bundle with top_n=80
   - render with appropriate preference

5. Always prefer evidence_spans rendering for existence/keyword queries
   - Only use entities rendering for list/roster queries
   - When in doubt, use evidence_spans

OUTPUT: JSON with "actions" array and "reasoning" string."""


def get_plan_patch(
    query_analysis: "QueryAnalysis",
    observations: "ObservationBundle",
) -> PlanPatch:
    """
    Use LLM to decide what actions to take based on probe observations.
    
    The LLM reads the observation signals and outputs a structured patch
    that tells the executor what operations to perform.
    """
    import os
    import json
    from openai import OpenAI
    
    # Build prompt
    prompt = build_plan_patch_prompt(query_analysis, observations)
    
    # Call LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")
    
    model = os.getenv("OPENAI_MODEL_PLAN", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an agentic controller for a historical research system. "
                                          "Based on probe retrieval signals, decide what actions to take. "
                                          "Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "plan_patch", "schema": PLAN_PATCH_SCHEMA, "strict": True},
        },
    }
    
    resp = client.beta.chat.completions.parse(**request_params)
    
    if not resp.choices:
        raise RuntimeError("OpenAI returned no choices")
    
    msg = resp.choices[0].message
    
    # Parse response
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            result = parsed.model_dump()
        elif hasattr(parsed, "dict"):
            result = parsed.dict()
        elif isinstance(parsed, dict):
            result = parsed
        else:
            result = json.loads(json.dumps(parsed, default=str))
    else:
        content = msg.content
        if not content:
            raise RuntimeError("LLM returned no content")
        result = json.loads(content)
    
    return PlanPatch.from_dict(result)


def get_fix_patch(
    verification_errors: List[str],
    state: "ExecutionState",
    observations: "ObservationBundle",
) -> PlanPatch:
    """
    Use LLM to generate a fix patch when verification fails.
    
    The LLM sees the errors and current state, and decides how to fix
    (drop bullets, weaken language, swap citations).
    """
    import os
    import json
    from openai import OpenAI
    
    # Build fix prompt
    prompt = f"""Verification failed. Generate a fix patch.

=== VERIFICATION ERRORS ===
{chr(10).join(f"- {e}" for e in verification_errors)}

=== CURRENT STATE ===
Rendered bullets: {len(state.answer.bullets) if state.answer else 0}
FocusBundle spans: {len(state.focus_bundle.spans) if state.focus_bundle else 0}

=== AVAILABLE FIX ACTIONS ===
- drop_bullets: {{"indices": [0, 2]}} - Remove specific bullets that failed
- weaken_language: {{"indices": [1], "prefix": "Possibly"}} - Add hedging
- swap_citations: {{"bullet_index": 0, "new_span_ids": ["span_1"]}} - Replace citations
- return_negative: {{"reason": "..."}} - Give up and return negative report

=== GUIDANCE ===
- If atom grounding failed: drop the bullet or swap citations
- If claim strength failed: weaken language with hedging
- If multiple bullets failed: consider return_negative with evidence_spans fallback
- Prefer minimal fixes over drastic changes

OUTPUT: JSON with "actions" array and "reasoning" string."""

    # Call LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to dropping all failed bullets
        return PlanPatch(
            actions=[Action(op=ActionOp.DROP_BULLETS.value, params={"indices": list(range(10))})],
            reasoning="API key missing, dropping all bullets as fallback"
        )
    
    model = os.getenv("OPENAI_MODEL_PLAN", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are fixing a verification failure. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "plan_patch", "schema": PLAN_PATCH_SCHEMA, "strict": True},
        },
    }
    
    try:
        resp = client.beta.chat.completions.parse(**request_params)
        
        if not resp.choices:
            raise RuntimeError("OpenAI returned no choices")
        
        msg = resp.choices[0].message
        
        parsed = getattr(msg, "parsed", None)
        if parsed is not None:
            if hasattr(parsed, "model_dump"):
                result = parsed.model_dump()
            elif hasattr(parsed, "dict"):
                result = parsed.dict()
            elif isinstance(parsed, dict):
                result = parsed
            else:
                result = json.loads(json.dumps(parsed, default=str))
        else:
            content = msg.content
            if not content:
                raise RuntimeError("LLM returned no content")
            result = json.loads(content)
        
        return PlanPatch.from_dict(result)
        
    except Exception as e:
        # Fallback: drop all bullets
        return PlanPatch(
            actions=[Action(op=ActionOp.RETURN_NEGATIVE.value, params={
                "reason": f"Verification failed and fix generation failed: {e}"
            })],
            reasoning=f"Fix generation failed: {e}"
        )


# Import type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from retrieval.query_analysis import QueryAnalysis
    from retrieval.observations import ObservationBundle


# =============================================================================
# Deterministic Patch Executor
# =============================================================================

def execute_patch(
    patch: PlanPatch,
    query_analysis: "QueryAnalysis",
    conn,
) -> ExecutionState:
    """
    Execute patch actions deterministically.
    
    Each action modifies the ExecutionState. Actions are executed in order.
    This is pure code execution - no LLM involvement.
    """
    import sys
    import time
    from retrieval.ops import hybrid_rrf, SearchFilters
    from retrieval.focus_bundle import FocusBundleBuilder
    from retrieval.query_intent import QueryContract, FocusBundleMode
    from retrieval.candidate_proposer import propose_all_candidates
    from retrieval.constraints import assess_candidate
    from retrieval.hubness import score_candidates_with_hubness, load_entity_df
    from retrieval.rendering import render_from_focus_bundle_with_constraints, RenderedAnswer
    from retrieval.verifier_v2 import Bullet
    
    print(f"\n  [Executor] Executing patch with {len(patch.actions)} actions", file=sys.stderr)
    print(f"    Reasoning: {patch.reasoning}", file=sys.stderr)
    
    state = ExecutionState()
    state.anchor_terms = list(query_analysis.anchor_terms)
    state.expanded_terms = list(query_analysis.suggested_synonyms)
    
    for i, action in enumerate(patch.actions):
        print(f"\n    [Action {i+1}] {action.op}", file=sys.stderr)
        state.action_log.append(f"{action.op}: {action.params}")
        
        if action.op == ActionOp.EXPAND_TERMS.value:
            _execute_expand_terms(state, action.params)
            
        elif action.op == ActionOp.RERUN_RETRIEVAL.value:
            _execute_rerun_retrieval(state, action.params, query_analysis, conn)
            
        elif action.op == ActionOp.BUILD_FOCUS_BUNDLE.value:
            _execute_build_focus_bundle(state, action.params, query_analysis, conn)
            
        elif action.op == ActionOp.RENDER.value:
            _execute_render(state, action.params, query_analysis, conn)
            
        elif action.op == ActionOp.RETURN_NEGATIVE.value:
            _execute_return_negative(state, action.params)
            
        elif action.op == ActionOp.DROP_BULLETS.value:
            _execute_drop_bullets(state, action.params)
            
        elif action.op == ActionOp.WEAKEN_LANGUAGE.value:
            _execute_weaken_language(state, action.params)
            
        elif action.op == ActionOp.SWAP_CITATIONS.value:
            _execute_swap_citations(state, action.params)
            
        else:
            print(f"      Unknown action: {action.op}", file=sys.stderr)
    
    return state


def _execute_expand_terms(state: ExecutionState, params: Dict[str, Any]):
    """Add synonym/variant terms."""
    import sys
    
    terms = params.get("terms", [])
    for term in terms:
        if term not in state.anchor_terms and term not in state.expanded_terms:
            state.expanded_terms.append(term)
    
    print(f"      Expanded terms: {state.expanded_terms}", file=sys.stderr)


def _execute_rerun_retrieval(
    state: ExecutionState,
    params: Dict[str, Any],
    query_analysis: "QueryAnalysis",
    conn,
):
    """Re-run retrieval with specified configuration."""
    import sys
    from retrieval.ops import hybrid_rrf, SearchFilters, lex_and
    
    lanes = params.get("lanes", {"hybrid_rrf": 200})
    scope = params.get("scope", query_analysis.scope_filters)
    
    # Build filters
    filters = SearchFilters()
    if scope:
        collections = scope.get("collections", [])
        if collections:
            filters = SearchFilters(collection_slugs=collections)
    
    # Build query with expanded terms
    all_terms = state.anchor_terms + state.expanded_terms
    retrieval_query = query_analysis.get_retrieval_query()
    
    # Add anchor terms to query for better semantic matching
    if all_terms:
        retrieval_query = f"{retrieval_query} {' '.join(all_terms)}"
    
    print(f"      Query: {retrieval_query[:100]}...", file=sys.stderr)
    print(f"      Lanes: {lanes}", file=sys.stderr)
    
    all_chunks = []
    
    # Run configured lanes
    for lane_name, k in lanes.items():
        # Skip None values (lane disabled)
        if k is None:
            continue
            
        if lane_name == "hybrid_rrf":
            try:
                chunks = hybrid_rrf(
                    conn=conn,
                    query=retrieval_query,
                    filters=filters,
                    k=k,
                    expand_concordance=True,
                    fuzzy_lex_enabled=True,
                )
                print(f"      hybrid_rrf: {len(chunks)} chunks", file=sys.stderr)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"      hybrid_rrf error: {e}", file=sys.stderr)
                
        elif lane_name == "lexical_must_hit":
            try:
                # Use anchor terms for lexical must-hit via lex_and
                if all_terms:
                    chunks = lex_and(
                        conn=conn,
                        terms=all_terms,
                        filters=filters,
                        k=k,
                    )
                    print(f"      lexical_must_hit (lex_and): {len(chunks)} chunks", file=sys.stderr)
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"      lexical_must_hit error: {e}", file=sys.stderr)
    
    # Dedupe by chunk_id (handle both ChunkHit objects and dicts)
    from retrieval.ops import ChunkHit
    
    seen_ids = set()
    deduped = []
    for chunk in all_chunks:
        if isinstance(chunk, ChunkHit):
            chunk_id = chunk.chunk_id
        elif isinstance(chunk, dict):
            chunk_id = chunk.get('chunk_id') or chunk.get('id')
        else:
            chunk_id = getattr(chunk, 'chunk_id', None) or getattr(chunk, 'id', None)
        
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            deduped.append(chunk)
    
    # Load texts (handles ChunkHit conversion)
    from retrieval.observations import _load_chunk_texts
    state.chunks = _load_chunk_texts(conn, deduped)
    
    print(f"      Total chunks after dedup: {len(state.chunks)}", file=sys.stderr)
    state.retrieval_lanes = lanes


def _execute_build_focus_bundle(
    state: ExecutionState,
    params: Dict[str, Any],
    query_analysis: "QueryAnalysis",
    conn,
):
    """Build FocusBundle with specified configuration."""
    import sys
    from retrieval.focus_bundle import FocusBundleBuilder
    from retrieval.query_intent import QueryContract, FocusBundleMode
    from retrieval.candidate_proposer import propose_all_candidates
    from retrieval.constraints import assess_candidate
    from retrieval.hubness import score_candidates_with_hubness, load_entity_df
    
    top_n = params.get("top_n", 80)
    min_score = params.get("min_score", 0.0)
    require_anchor_hits = params.get("require_anchor_hits", False)
    
    if not state.chunks:
        print(f"      No chunks available, skipping FocusBundle", file=sys.stderr)
        return
    
    # Build QueryContract
    contract = QueryContract(
        query_text=query_analysis.query_text,
        mode=FocusBundleMode.KEYWORD_INTENT,
    )
    
    # All anchor terms (original + expanded)
    all_anchors = state.anchor_terms + state.expanded_terms
    
    # Build FocusBundle
    builder = FocusBundleBuilder(
        top_n_spans=top_n,
        min_span_score=min_score,
    )
    state.focus_bundle = builder.build(contract, state.chunks, conn, anchor_terms=all_anchors)
    
    print(f"      Built FocusBundle with {len(state.focus_bundle.spans)} spans", file=sys.stderr)
    
    # Check anchor hits if required
    if require_anchor_hits and all_anchors:
        anchor_hits = state.focus_bundle.params.get("anchor_hit_count", 0)
        if anchor_hits == 0:
            print(f"      WARNING: require_anchor_hits=True but 0 hits!", file=sys.stderr)
    
    # Propose candidates
    state.candidates = propose_all_candidates(state.focus_bundle, conn)
    print(f"      Proposed {len(state.candidates)} candidates", file=sys.stderr)
    
    # Assess candidates (no constraints for now)
    state.assessments = []
    for candidate in state.candidates:
        assessment = assess_candidate(candidate, [], state.focus_bundle, conn)
        state.assessments.append(assessment)
    
    # Hubness scoring
    try:
        entity_df = load_entity_df(conn)
        state.hubness_scores = score_candidates_with_hubness(
            state.candidates,
            state.focus_bundle,
            entity_df,
            conn,
        )
        print(f"      Hubness scored {len(state.hubness_scores)} candidates", file=sys.stderr)
    except Exception as e:
        print(f"      Hubness scoring failed: {e}", file=sys.stderr)
        state.hubness_scores = []
    
    state.focus_bundle_config = params


def _execute_render(
    state: ExecutionState,
    params: Dict[str, Any],
    query_analysis: "QueryAnalysis",
    conn,
):
    """Render answer with specified preference."""
    import sys
    from retrieval.rendering import render_from_focus_bundle_with_constraints, RenderedAnswer
    from retrieval.verifier_v2 import Bullet
    
    preference = params.get("preference", ["evidence_spans", "entities"])
    max_items = params.get("max_items", 25)
    
    if not state.focus_bundle:
        print(f"      No FocusBundle, creating empty answer", file=sys.stderr)
        state.answer = RenderedAnswer(
            short_answer="No evidence found.",
            bullets=[],
            focus_bundle_id=None,
            total_candidates=0,
            rendered_count=0,
        )
        return
    
    # Render based on preference
    if "evidence_spans" in preference:
        # Prefer rendering evidence spans directly
        state.answer = render_from_focus_bundle_with_constraints(
            focus_bundle=state.focus_bundle,
            assessments=state.assessments or [],
            constraints=[],
            max_items=max_items,
            conservative_language=True,
            hubness_scores=state.hubness_scores,
        )
    else:
        # Render entity list
        state.answer = render_from_focus_bundle_with_constraints(
            focus_bundle=state.focus_bundle,
            assessments=state.assessments or [],
            constraints=[],  # No constraints for now
            max_items=max_items,
            conservative_language=True,
            hubness_scores=state.hubness_scores,
        )
    
    print(f"      Rendered {len(state.answer.bullets)} bullets", file=sys.stderr)
    state.render_preference = preference


def _execute_return_negative(state: ExecutionState, params: Dict[str, Any]):
    """Return a negative evidence report."""
    import sys
    from retrieval.rendering import RenderedAnswer
    from retrieval.verifier_v2 import Bullet
    
    reason = params.get("reason", "No evidence found in searched collections.")
    
    # Create negative report with evidence spans if available
    bullets = []
    if state.focus_bundle and state.focus_bundle.spans:
        # Include top spans as context
        for span in state.focus_bundle.spans[:5]:
            text = span.text[:200].strip()
            text = ' '.join(text.split())
            bullets.append(Bullet(
                text=f"[{span.page_ref}] {text}...",
                cited_span_ids=[span.span_id],
                confidence="low",
                candidate_key=f"negative:{span.span_id}",
            ))
    
    state.answer = RenderedAnswer(
        short_answer=reason,
        bullets=bullets,
        focus_bundle_id=state.focus_bundle.retrieval_run_id if state.focus_bundle else None,
        total_candidates=len(state.candidates),
        rendered_count=len(bullets),
        negative_findings=reason,
    )
    
    print(f"      Negative report: {reason}", file=sys.stderr)


def _execute_drop_bullets(state: ExecutionState, params: Dict[str, Any]):
    """Drop specific bullets from the answer."""
    import sys
    
    if not state.answer:
        return
    
    indices = params.get("indices", [])
    new_bullets = [
        b for i, b in enumerate(state.answer.bullets)
        if i not in indices
    ]
    
    print(f"      Dropped {len(state.answer.bullets) - len(new_bullets)} bullets", file=sys.stderr)
    state.answer.bullets = new_bullets
    state.answer.rendered_count = len(new_bullets)


def _execute_weaken_language(state: ExecutionState, params: Dict[str, Any]):
    """Add hedging to specific bullets."""
    import sys
    from retrieval.verifier_v2 import Bullet
    
    if not state.answer:
        return
    
    indices = params.get("indices", [])
    prefix = params.get("prefix", "Possibly")
    
    for i in indices:
        if i < len(state.answer.bullets):
            old_text = state.answer.bullets[i].text
            state.answer.bullets[i] = Bullet(
                text=f"{prefix}: {old_text}",
                cited_span_ids=state.answer.bullets[i].cited_span_ids,
                confidence="low",
                candidate_key=state.answer.bullets[i].candidate_key,
            )
    
    print(f"      Weakened {len(indices)} bullets", file=sys.stderr)


def _execute_swap_citations(state: ExecutionState, params: Dict[str, Any]):
    """Swap citations on a bullet."""
    import sys
    from retrieval.verifier_v2 import Bullet
    
    if not state.answer:
        return
    
    bullet_index = params.get("bullet_index", 0)
    new_span_ids = params.get("new_span_ids", [])
    
    if bullet_index < len(state.answer.bullets):
        old = state.answer.bullets[bullet_index]
        state.answer.bullets[bullet_index] = Bullet(
            text=old.text,
            cited_span_ids=new_span_ids,
            confidence=old.confidence,
            candidate_key=old.candidate_key,
        )
        print(f"      Swapped citations on bullet {bullet_index}", file=sys.stderr)
