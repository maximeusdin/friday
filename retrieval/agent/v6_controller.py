"""
V6 Controller - No heuristics, principled architecture

The V6 pipeline:
1. PARSE: Split query into CONTROL vs CONTENT tokens
2. LINK: Entity-link ONLY content tokens, with use_for_retrieval flag
3. AGENTIC RETRIEVAL: Searcher decides tools, but with parsed context
4. BOTTLENECK: Hard filter to 30-50 spans (forces convergence)
5. SYNTHESIZE: Generate answer from bottlenecked evidence only
6. VERIFY RESPONSIVENESS: Does answer satisfy the question?
7. PROGRESS GATE: Only expand if gaining quality evidence

No heuristics - LLM decides tools, but with filtered entity context.
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field

from retrieval.agent.v6_query_parser import QueryParser, ParsedQuery, TaskType
from retrieval.agent.v6_entity_linker import EntityLinker, EntityLinkingResult, LinkedEntity
from retrieval.agent.v6_evidence_bottleneck import (
    EvidenceBottleneck, BottleneckResult, GradingMode, DEFAULT_GRADING_MODE,
    TOURNAMENT_BATCH_BY_DEFAULT,
)
from retrieval.agent.v6_responsiveness import ResponsivenessVerifier, ResponsivenessResult, ResponsivenessStatus
from retrieval.agent.v6_progress_gate import ProgressGate, ProgressResult, RoundDecision
from retrieval.agent.tools import TOOL_REGISTRY, ToolResult, get_tools_for_prompt
from retrieval.agent.v7_types import RoundSummary


# =============================================================================
# V6 Configuration
# =============================================================================

@dataclass
class V6Config:
    """Configuration for V6 pipeline."""
    
    # Bottleneck
    max_bottleneck_spans: int = 40
    # Grading mode: "tournament" (default, pairwise comparison) or "absolute" (0-10 scoring)
    bottleneck_grading_mode: GradingMode = DEFAULT_GRADING_MODE
    # Tournament batching: True = batch matchups for speed, False = 1 API call per matchup (slower but more accurate)
    tournament_batch: bool = TOURNAMENT_BATCH_BY_DEFAULT
    
    # Rounds
    max_rounds: int = 5
    min_progress_per_round: int = 3
    max_no_progress_rounds: int = 2
    
    # Entity linking
    entity_confidence_threshold: float = 0.6
    
    # Retrieval
    chunks_per_search: int = 150
    
    # Synthesis
    synthesis_model: str = "gpt-4o"
    
    verbose: bool = True
    
    # Progress callback for streaming updates
    # Called with (step: str, status: str, message: str, details: dict)
    progress_callback: Optional[Callable[[str, str, str, Dict[str, Any]], None]] = None
    
    # V7 Phase 2: Novelty controls
    # off = no exclusion, soft = exclude but allow override, hard = always exclude
    exclude_seen_mode: str = "soft"  # off | soft | hard
    # Multiplier for top_k when exclusion is active (to ensure enough results)
    top_k_budget_multiplier: float = 1.5
    
    # V7 Phase 2: Round summary
    # If True, generate RoundSummary after each round for smarter decision-making
    enable_round_summary: bool = True
    round_summary_model: str = "gpt-4o-mini"


# =============================================================================
# V6 Trace
# =============================================================================

@dataclass
class V6Trace:
    """Complete trace of V6 execution."""
    
    question: str = ""
    
    # Step 1: Parse
    parsed_query: Optional[ParsedQuery] = None
    
    # Step 2: Entity linking
    entity_linking: Optional[EntityLinkingResult] = None
    
    # Step 3-4: Retrieval + Bottleneck per round
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    bottleneck_spans: List[Any] = field(default_factory=list)  # All spans after bottleneck
    
    # Step 5: Final synthesis
    final_answer: str = ""
    final_claims: List[Dict[str, Any]] = field(default_factory=list)
    
    # Step 6: Responsiveness
    responsiveness: Optional[ResponsivenessResult] = None
    
    # Progress
    progress_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Outcome
    stopped_reason: str = ""
    total_elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "parsed_query": self.parsed_query.to_dict() if self.parsed_query else None,
            "entity_linking": self.entity_linking.to_dict() if self.entity_linking else None,
            "rounds": self.rounds,
            "final_answer": self.final_answer,
            "final_claims": self.final_claims,
            "responsiveness": self.responsiveness.to_dict() if self.responsiveness else None,
            "progress_summary": self.progress_summary,
            "stopped_reason": self.stopped_reason,
            "total_elapsed_ms": self.total_elapsed_ms,
        }


# =============================================================================
# V6 Controller
# =============================================================================

class V6Controller:
    """
    The V6 controller - no heuristics, principled architecture.
    
    Key principles:
    1. CONTROL vs CONTENT separation prevents bad entity linking
    2. use_for_retrieval flag prevents random entities as seeds
    3. Hard bottleneck forces convergence
    4. Responsiveness check ensures answer satisfies question
    5. Progress gate prevents thrashing
    """
    
    def __init__(
        self,
        config: Optional[V6Config] = None,
    ):
        self.config = config or V6Config()
        
        # Components
        self.parser = QueryParser(verbose=self.config.verbose)
        self.linker = EntityLinker(
            confidence_threshold=self.config.entity_confidence_threshold,
            verbose=self.config.verbose,
        )
        self.bottleneck = EvidenceBottleneck(
            max_spans=self.config.max_bottleneck_spans,
            grading_mode=self.config.bottleneck_grading_mode,
            batch_tournament=self.config.tournament_batch,
            verbose=self.config.verbose,
        )
        self.responsiveness_verifier = ResponsivenessVerifier(verbose=self.config.verbose)
        self.progress_gate = ProgressGate(
            min_progress=self.config.min_progress_per_round,
            max_rounds=self.config.max_rounds,
            max_no_progress=self.config.max_no_progress_rounds,
            verbose=self.config.verbose,
        )
        
        # State
        self.trace = V6Trace()
        self.all_bottleneck_spans: List[Any] = []
        self.all_members_found: Set[str] = set()
        
        # V7 Phase 2: Track seen chunk IDs for novelty control
        self.seen_chunk_ids: Set[int] = set()
        self.seen_page_ids: Set[int] = set()
        self.seen_document_ids: Set[int] = set()
        
        # V7 Phase 2: Round summary
        self.round_summary_generator = None
        self.previous_round_summary: Optional[RoundSummary] = None
        self.round_observations: List[Dict[str, Any]] = []  # Tool observations for current round
        if self.config.enable_round_summary:
            try:
                from retrieval.agent.v7_controller import RoundSummaryGenerator
                self.round_summary_generator = RoundSummaryGenerator(
                    verbose=self.config.verbose,
                    model=self.config.round_summary_model,
                )
            except ImportError:
                pass  # V7 controller not available
    
    def _emit_progress(self, step: str, status: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Emit a progress event if callback is configured."""
        if self.config.progress_callback:
            try:
                self.config.progress_callback(step, status, message, details or {})
            except Exception:
                pass  # Don't let callback errors break the workflow
    
    def run(
        self,
        question: str,
        conn,
    ) -> V6Trace:
        """Run the V6 pipeline."""
        
        start_time = time.time()
        self.trace.question = question
        
        if self.config.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V6 Controller] Question: {question}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        # Step 1: PARSE - Split into CONTROL vs CONTENT
        if self.config.verbose:
            print(f"\n[Step 1] QUERY PARSING - Separating CONTROL from CONTENT", file=sys.stderr)
        
        parsed = self.parser.parse(question)
        self.trace.parsed_query = parsed
        
        # Emit progress for query parsing
        self._emit_progress("query_parsing", "completed", f"Parsed as {parsed.task_type.value} task", {
            "task_type": parsed.task_type.value,
            "topic_terms": parsed.topic_terms[:5],
            "control_tokens_count": len(parsed.control_tokens),
        })
        
        # EXPLICIT: Show exactly what the model classified
        if self.config.verbose:
            print(f"\n  ╔═══════════════════════════════════════════════════════════════", file=sys.stderr)
            print(f"  ║ PARSE RESULTS", file=sys.stderr)
            print(f"  ╠═══════════════════════════════════════════════════════════════", file=sys.stderr)
            print(f"  ║ Task Type: {parsed.task_type.value}", file=sys.stderr)
            print(f"  ║", file=sys.stderr)
            print(f"  ║ CONTROL TOKENS (will NOT be entity-linked):", file=sys.stderr)
            for token in sorted(parsed.control_tokens):
                print(f"  ║   - \"{token}\"", file=sys.stderr)
            print(f"  ║", file=sys.stderr)
            print(f"  ║ CONTENT TOKENS (WILL be entity-linked):", file=sys.stderr)
            for token in sorted(parsed.content_tokens):
                print(f"  ║   - \"{token}\"", file=sys.stderr)
            print(f"  ║", file=sys.stderr)
            print(f"  ║ TOPIC TERMS (primary subjects to search):", file=sys.stderr)
            for term in parsed.topic_terms:
                print(f"  ║   - \"{term}\"", file=sys.stderr)
            print(f"  ║", file=sys.stderr)
            print(f"  ║ SCOPE CONSTRAINTS: {parsed.scope_constraints or 'None'}", file=sys.stderr)
            print(f"  ║ OUTPUT REQUIREMENTS: {parsed.output_requirements or 'None'}", file=sys.stderr)
            print(f"  ╚═══════════════════════════════════════════════════════════════", file=sys.stderr)
        
        # Step 2: LINK - Entity-link ONLY content tokens
        if self.config.verbose:
            print(f"\n[Step 2] ENTITY LINKING - Only CONTENT tokens from Step 1", file=sys.stderr)
        
        linking = self.linker.link(parsed, conn)
        self.trace.entity_linking = linking
        
        # Get retrieval entity IDs (only those marked for retrieval)
        retrieval_entity_ids = linking.get_retrieval_entity_ids()
        
        # Emit progress for entity linking
        retrieval_entities = [{"name": e.canonical_name, "id": e.entity_id} for e in linking.retrieval_entities[:5]]
        self._emit_progress("entity_linking", "completed", f"Linked {linking.total_linked} entities, {linking.used_for_retrieval} for retrieval", {
            "total_linked": linking.total_linked,
            "used_for_retrieval": linking.used_for_retrieval,
            "rejected_control_tokens": len(linking.rejected_control_tokens),
            "retrieval_entities": retrieval_entities,
        })
        
        # EXPLICIT: Show exactly what was linked and what will be used
        if self.config.verbose:
            print(f"\n  ╔═══════════════════════════════════════════════════════════════", file=sys.stderr)
            print(f"  ║ ENTITY LINKING RESULTS", file=sys.stderr)
            print(f"  ╠═══════════════════════════════════════════════════════════════", file=sys.stderr)
            print(f"  ║ Terms attempted to link (from topic_terms): {len(parsed.topic_terms)}", file=sys.stderr)
            print(f"  ║ Successfully linked: {linking.total_linked}", file=sys.stderr)
            print(f"  ║ Marked for retrieval: {linking.used_for_retrieval}", file=sys.stderr)
            print(f"  ║", file=sys.stderr)
            if linking.rejected_control_tokens:
                print(f"  ║ REJECTED (were actually control tokens):", file=sys.stderr)
                for token in linking.rejected_control_tokens:
                    print(f"  ║   ✗ \"{token}\" - NOT linked (control token)", file=sys.stderr)
            print(f"  ║", file=sys.stderr)
            print(f"  ║ ALL LINKED ENTITIES:", file=sys.stderr)
            for e in linking.linked_entities:
                status = "✓ FOR RETRIEVAL" if e.use_for_retrieval else "✗ NOT for retrieval"
                print(f"  ║   [{e.entity_id}] \"{e.surface_form}\" → {e.canonical_name}", file=sys.stderr)
                print(f"  ║       Type: {e.entity_type}, Confidence: {e.link_confidence:.2f}, Match: {e.match_type}", file=sys.stderr)
                print(f"  ║       {status}: {e.retrieval_reason}", file=sys.stderr)
            print(f"  ║", file=sys.stderr)
            print(f"  ║ FINAL RETRIEVAL SEEDS (entity IDs):", file=sys.stderr)
            if retrieval_entity_ids:
                for e in linking.retrieval_entities:
                    print(f"  ║   → {e.entity_id}: {e.canonical_name}", file=sys.stderr)
            else:
                print(f"  ║   (none - will use lexical search only)", file=sys.stderr)
            print(f"  ╚═══════════════════════════════════════════════════════════════", file=sys.stderr)
        
        # Step 3-4: RETRIEVE + BOTTLENECK (with progress gating)
        if self.config.verbose:
            print(f"\n[Step 3-4] RETRIEVAL - Using ONLY parsed entities", file=sys.stderr)
            print(f"  ╔═══════════════════════════════════════════════════════════════", file=sys.stderr)
            print(f"  ║ RETRIEVAL INPUTS (derived from parsing)", file=sys.stderr)
            print(f"  ╠═══════════════════════════════════════════════════════════════", file=sys.stderr)
            print(f"  ║ Topic terms for search: {parsed.topic_terms}", file=sys.stderr)
            print(f"  ║ Entity IDs for retrieval:", file=sys.stderr)
            if linking.retrieval_entities:
                for e in linking.retrieval_entities:
                    print(f"  ║   → {e.entity_id}: {e.canonical_name} (surface: \"{e.surface_form}\")", file=sys.stderr)
            else:
                print(f"  ║   (none - only lexical search will be used)", file=sys.stderr)
            print(f"  ║", file=sys.stderr)
            print(f"  ║ NOT using these (CONTROL tokens):", file=sys.stderr)
            print(f"  ║   {list(parsed.control_tokens)[:10]}", file=sys.stderr)
            print(f"  ╚═══════════════════════════════════════════════════════════════", file=sys.stderr)
        
        round_num = 0
        previous_evidence_count = 0
        previous_members: Set[str] = set()
        
        while True:
            round_num += 1
            round_start = time.time()
            
            # V7 Phase 2: Reset round observations
            self.round_observations = []
            
            if self.config.verbose:
                print(f"\n  [Round {round_num}]", file=sys.stderr)
            
            # Emit round start
            self._emit_progress(f"retrieval_round_{round_num}", "running", f"Starting retrieval round {round_num}", {
                "round": round_num,
            })
            
            # Retrieve
            chunks = self._retrieve(parsed, linking, conn, round_num)
            
            # Bottleneck
            bottleneck_result = self.bottleneck.filter(chunks, parsed, conn)
            
            # Merge with previous spans
            for span in bottleneck_result.spans:
                if span.span_id not in {s.span_id for s in self.all_bottleneck_spans}:
                    self.all_bottleneck_spans.append(span)
            
            self.all_members_found.update(bottleneck_result.members_identified)
            
            # V7 Phase 2: Generate round summary for smarter decision-making
            if self.round_summary_generator:
                try:
                    chunk_dicts = [{"text": c.get("text", ""), "source_label": c.get("source_label", ""), "page": c.get("page", "")} for c in chunks]
                    round_summary = self.round_summary_generator.generate(
                        round_number=round_num,
                        question=parsed.original_query,
                        evidence_chunks=chunk_dicts,
                        previous_summary=self.previous_round_summary,
                        tool_observations=self.round_observations,
                    )
                    self.previous_round_summary = round_summary
                    
                    if self.config.verbose:
                        print(f"    [RoundSummary] {round_summary.decision.value}: {round_summary.decision_rationale}", file=sys.stderr)
                        if round_summary.actionable_leads:
                            print(f"    [RoundSummary] {len(round_summary.actionable_leads)} leads identified", file=sys.stderr)
                except Exception as e:
                    if self.config.verbose:
                        print(f"    [RoundSummary] Error: {e}", file=sys.stderr)
            
            # Record round
            round_data = {
                "round": round_num,
                "chunks_retrieved": len(chunks),
                "spans_after_bottleneck": bottleneck_result.spans_passed,
                "total_spans": len(self.all_bottleneck_spans),
                "members_found": list(self.all_members_found),
                "elapsed_ms": (time.time() - round_start) * 1000,
            }
            # V7 Phase 2: Include round summary in trace
            if self.previous_round_summary:
                round_data["round_summary"] = self.previous_round_summary.to_dict()
            self.trace.rounds.append(round_data)
            
            # Emit round complete
            self._emit_progress(f"retrieval_round_{round_num}", "completed", f"Round {round_num}: {len(chunks)} chunks → {bottleneck_result.spans_passed} spans ({len(self.all_bottleneck_spans)} total)", {
                "round": round_num,
                "chunks_retrieved": len(chunks),
                "spans_after_bottleneck": bottleneck_result.spans_passed,
                "total_spans": len(self.all_bottleneck_spans),
                "members_found": len(self.all_members_found),
            })
            
            # Progress gate
            progress = self.progress_gate.evaluate(
                bottleneck_result=bottleneck_result,
                task_type=parsed.task_type,
                previous_evidence_count=previous_evidence_count,
                previous_members=previous_members,
            )
            
            previous_evidence_count = len(self.all_bottleneck_spans)
            previous_members = self.all_members_found.copy()
            
            # Decision
            if progress.decision == RoundDecision.STOP:
                if self.config.verbose:
                    print(f"    Stopping: {progress.reason}", file=sys.stderr)
                break
            elif progress.decision == RoundDecision.PIVOT:
                if self.config.verbose:
                    print(f"    Pivoting: {progress.reason}", file=sys.stderr)
                # TODO: implement pivot logic
                break
            # else: CONTINUE
        
        self.trace.progress_summary = self.progress_gate.get_summary()
        
        # Step 5: SYNTHESIZE from bottlenecked spans only
        if self.config.verbose:
            print(f"\n[Step 5] Synthesizing from {len(self.all_bottleneck_spans)} bottlenecked spans...", 
                  file=sys.stderr)
        
        self._emit_progress("synthesis", "running", f"Synthesizing answer from {len(self.all_bottleneck_spans)} spans...", {
            "total_spans": len(self.all_bottleneck_spans),
        })
        
        answer, claims = self._synthesize(parsed)
        self.trace.final_answer = answer
        self.trace.final_claims = claims
        
        self._emit_progress("synthesis", "completed", f"Generated answer with {len(claims)} claims", {
            "claims_count": len(claims),
            "answer_length": len(answer),
        })
        
        # Step 6: VERIFY RESPONSIVENESS
        if self.config.verbose:
            print(f"\n[Step 6] Verifying responsiveness...", file=sys.stderr)
        
        self._emit_progress("responsiveness", "running", "Verifying answer responsiveness...", {})
        
        responsiveness = self.responsiveness_verifier.verify(answer, claims, parsed)
        self.trace.responsiveness = responsiveness
        
        self._emit_progress("responsiveness", "completed", f"Responsiveness: {responsiveness.status.value}", {
            "status": responsiveness.status.value,
            "members_with_citations": len(responsiveness.members_with_citations) if responsiveness.members_with_citations else 0,
        })
        
        # Determine outcome
        if responsiveness.status == ResponsivenessStatus.RESPONSIVE:
            self.trace.stopped_reason = "responsive_answer"
        elif responsiveness.status == ResponsivenessStatus.PARTIALLY_RESPONSIVE:
            self.trace.stopped_reason = "partial_answer"
        else:
            self.trace.stopped_reason = "not_responsive"
        
        self.trace.total_elapsed_ms = (time.time() - start_time) * 1000
        
        if self.config.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V6] EXECUTION COMPLETE", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"  Outcome: {self.trace.stopped_reason}", file=sys.stderr)
            print(f"  Rounds: {round_num}", file=sys.stderr)
            print(f"  Final spans: {len(self.all_bottleneck_spans)}", file=sys.stderr)
            print(f"  Members found: {len(self.all_members_found)}", file=sys.stderr)
            print(f"  Responsiveness: {responsiveness.status.value}", file=sys.stderr)
            print(f"  Elapsed: {self.trace.total_elapsed_ms:.0f}ms", file=sys.stderr)
            print(f"\n  V6 KEY ACTIONS:", file=sys.stderr)
            print(f"    1. PARSED query into CONTROL vs CONTENT tokens", file=sys.stderr)
            print(f"       - Task type: {parsed.task_type.value}", file=sys.stderr)
            print(f"       - Topic terms (CONTENT): {parsed.topic_terms}", file=sys.stderr)
            print(f"       - Control tokens (SKIPPED): {len(parsed.control_tokens)} tokens", file=sys.stderr)
            print(f"    2. ENTITY LINKED only topic terms", file=sys.stderr)
            print(f"       - Linked: {linking.total_linked}", file=sys.stderr)
            print(f"       - Used for retrieval: {linking.used_for_retrieval}", file=sys.stderr)
            print(f"       - Rejected as control: {len(linking.rejected_control_tokens)}", file=sys.stderr)
            print(f"    3. RETRIEVAL used ONLY approved entity IDs", file=sys.stderr)
            print(f"       - Entity IDs used: {linking.get_retrieval_entity_ids()}", file=sys.stderr)
            print(f"    4. BOTTLENECK hard-filtered to {self.config.max_bottleneck_spans} max spans", file=sys.stderr)
            print(f"    5. RESPONSIVENESS verified: {responsiveness.status.value}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        # Populate bottleneck spans on trace before returning
        self.trace.bottleneck_spans = self.all_bottleneck_spans
        
        return self.trace
    
    def _retrieve(
        self,
        parsed: ParsedQuery,
        linking: EntityLinkingResult,
        conn,
        round_num: int,
    ) -> List[Dict[str, Any]]:
        """
        AGENTIC retrieval - LLM decides what tools to call.
        
        V6 provides the parsed query context (CONTROL/CONTENT, approved entities)
        but the searcher LLM decides the actual tool calls.
        
        SCOPE ENFORCEMENT: collections from scope_constraints are INJECTED into
        every tool call, regardless of what the LLM requests.
        """
        
        # Reset transaction state to handle any previous errors
        try:
            conn.rollback()
        except:
            pass
        
        all_chunks = []
        max_tool_calls = 6  # Budget per round
        
        # SCOPE ENFORCEMENT: Extract required collections from parsed query
        scope_collections = parsed.scope_constraints.get("collections", []) if parsed.scope_constraints else []
        
        if self.config.verbose:
            print(f"\n    ┌─────────────────────────────────────────────────────────────", file=sys.stderr)
            print(f"    │ AGENTIC RETRIEVAL (Round {round_num})", file=sys.stderr)
            print(f"    │ LLM will decide tools. Context provided:", file=sys.stderr)
            print(f"    │   - Topic terms: {parsed.topic_terms[:3]}", file=sys.stderr)
            print(f"    │   - Approved entity IDs: {[e.entity_id for e in linking.retrieval_entities[:3]]}", file=sys.stderr)
            print(f"    │   - Control tokens (avoid): {list(parsed.control_tokens)[:5]}", file=sys.stderr)
            if scope_collections:
                print(f"    │   - ENFORCED SCOPE: collections={scope_collections}", file=sys.stderr)
            print(f"    └─────────────────────────────────────────────────────────────", file=sys.stderr)
        
        # Build context for the searcher LLM
        v6_context = self._build_searcher_context(parsed, linking, round_num)
        
        # Let LLM decide tools (up to max_tool_calls)
        observations = []
        calls_made = set()  # Track tool+params to prevent duplicates
        actual_calls = 0  # Count actual (non-skipped) calls
        total_attempts = 0  # Safety limit to prevent infinite loops
        max_attempts = max_tool_calls * 3  # Allow some retries but not forever
        
        while actual_calls < max_tool_calls and total_attempts < max_attempts:
            total_attempts += 1
            action = self._ask_searcher_for_action(
                question=parsed.original_query,
                v6_context=v6_context,
                observations=observations,
                chunks_so_far=len(all_chunks),
            )
            
            if action is None:
                if self.config.verbose:
                    print(f"    [Searcher] No action returned, stopping", file=sys.stderr)
                break
            
            if action.get("action") == "STOP":
                if self.config.verbose:
                    print(f"    [Searcher] Decided to stop: {action.get('reason', 'enough evidence')}", file=sys.stderr)
                break
            
            if action.get("action") == "CALL_TOOL":
                tool_name = action.get("tool_name")
                params = action.get("params", {})
                
                # SCOPE ENFORCEMENT: Inject collections into search tools
                if scope_collections and tool_name in ("hybrid_search", "lexical_search", "lexical_exact", "vector_search"):
                    params["collections"] = scope_collections
                    if self.config.verbose:
                        print(f"    [Scope] INJECTED collections={scope_collections} into {tool_name}", file=sys.stderr)
                
                # DEDUPLICATION: Skip if we already made this exact call
                call_key = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
                if call_key in calls_made:
                    if self.config.verbose:
                        print(f"    [SKIP] Already called {tool_name} with same params - LLM must try different tool", file=sys.stderr)
                    # Add to observations so LLM sees it was skipped
                    observations.append({
                        "tool": tool_name,
                        "params": params,
                        "skipped": True,
                        "reason": "duplicate call",
                    })
                    continue  # Don't count this as a call, but still loop
                calls_made.add(call_key)
                actual_calls += 1  # Only count non-skipped calls
                
                if self.config.verbose:
                    print(f"    [Searcher] Tool: {tool_name}({params})", file=sys.stderr)
                
                result = self._call_tool(tool_name, params, conn)
                chunks = self._load_chunks(result.chunk_ids, conn)
                
                # SCOPE ENFORCEMENT: Filter chunks that don't match scope
                # This catches entity-based searches that can't take collections param
                if scope_collections:
                    before_filter = len(chunks)
                    chunks = [c for c in chunks if c.get("source_label", "") in scope_collections or not c.get("source_label")]
                    filtered_out = before_filter - len(chunks)
                    if filtered_out > 0 and self.config.verbose:
                        print(f"    [Scope] FILTERED {filtered_out} chunks outside {scope_collections}", file=sys.stderr)
                
                # V7 Phase 2: Record seen chunk IDs for novelty control
                for chunk in chunks:
                    cid = chunk.get("id")
                    if cid:
                        self.seen_chunk_ids.add(cid)
                    page_id = chunk.get("page_id")
                    if page_id:
                        self.seen_page_ids.add(page_id)
                    doc_id = chunk.get("doc_id")
                    if doc_id:
                        self.seen_document_ids.add(doc_id)
                
                if self.config.verbose:
                    print(f"             → {len(chunks)} chunks", file=sys.stderr)
                
                all_chunks.extend(chunks)
                
                # Record observation with error info
                obs = {
                    "tool": tool_name,
                    "params": params,
                    "chunks_found": len(chunks),
                }
                if result.error:
                    obs["error"] = result.error
                observations.append(obs)
                
                # Check for repeated failures - if same tool/params failed twice, stop trying
                failed_calls = [o for o in observations if o.get("chunks_found", 0) == 0]
                if len(failed_calls) >= 4:
                    if self.config.verbose:
                        print(f"    [Searcher] 4+ failed calls, stopping retrieval loop", file=sys.stderr)
                    break
        
        # V7 Phase 2: Save observations for round summary generation
        self.round_observations = observations.copy()
        
        # Deduplicate by chunk_id
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            cid = chunk.get("id")
            if cid and cid not in seen:
                seen.add(cid)
                unique_chunks.append(chunk)
        
        if self.config.verbose:
            print(f"    [Total] {len(unique_chunks)} unique chunks after deduplication", file=sys.stderr)
        
        return unique_chunks[:self.config.chunks_per_search]
    
    def _build_searcher_context(
        self,
        parsed: ParsedQuery,
        linking: EntityLinkingResult,
        round_num: int,
    ) -> str:
        """Build context for the searcher LLM about V6 filtering."""
        
        lines = [
            "V6 QUERY ANALYSIS:",
            f"  Task type: {parsed.task_type.value}",
            f"  Topic terms (search for these): {parsed.topic_terms}",
            f"  Scope constraints: {parsed.scope_constraints}",
            "",
            "APPROVED ENTITIES FOR RETRIEVAL:",
        ]
        
        if linking.retrieval_entities:
            for e in linking.retrieval_entities:
                lines.append(f"  - [{e.entity_id}] {e.canonical_name} (surface: \"{e.surface_form}\")")
        else:
            lines.append("  (none linked - use lexical search with topic terms)")
        
        lines.extend([
            "",
            "CONTROL TOKENS (do NOT search for these directly):",
            f"  {list(parsed.control_tokens)[:8]}",
            "",
            "GUIDANCE:",
            "- Use topic_terms for lexical/hybrid search",
            "- Use approved entity IDs for entity_mentions/co_mentions",
            "- Do NOT search for control tokens like 'provide', 'cite', collection names as content",
        ])
        
        # V7 Phase 2: Include previous round summary if available
        if self.previous_round_summary:
            lines.extend([
                "",
                "=== PREVIOUS ROUND INSIGHTS ===",
                self.previous_round_summary.format_for_context(),
            ])
            
            # Highlight high-priority leads
            high_priority = self.previous_round_summary.get_high_priority_leads()
            if high_priority:
                lines.append("\nRECOMMENDED NEXT ACTIONS:")
                for lead in high_priority[:3]:
                    tool_hint = f" (try: {lead.suggested_tool})" if lead.suggested_tool else ""
                    lines.append(f"  → {lead.lead_type}: {lead.target}{tool_hint}")
        
        return "\n".join(lines)
    
    def _ask_searcher_for_action(
        self,
        question: str,
        v6_context: str,
        observations: List[Dict],
        chunks_so_far: int,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM searcher what tool to call next."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        # Build observation summary - CRITICAL: Show what already tried so LLM doesn't repeat
        obs_text = ""
        if observations:
            obs_lines = ["PREVIOUS ACTIONS (do NOT repeat these):"]
            for obs in observations:
                if obs.get('skipped'):
                    status = f"SKIPPED (duplicate - you already tried this!)"
                elif obs.get('error'):
                    status = f"ERROR: {obs['error'][:50]}"
                elif obs.get('chunks_found', 0) == 0:
                    status = "FAILED (0 results)"
                else:
                    status = f"got {obs['chunks_found']} chunks"
                obs_lines.append(f"  - {obs['tool']}({obs.get('params', {})}) → {status}")
            obs_text = "\n".join(obs_lines) + "\n\nCRITICAL: You MUST try a DIFFERENT tool or DIFFERENT parameters. Repeating the same call is not allowed."
        
        # Get tool descriptions
        tools_desc = self._get_tools_description()
        
        prompt = f"""You are a research agent searching archival documents.

QUESTION: {question}

{v6_context}

{obs_text}

CHUNKS FOUND SO FAR: {chunks_so_far}

{tools_desc}

Decide your next action. Output JSON:
- {{"action": "CALL_TOOL", "tool_name": "...", "params": {{...}}, "rationale": "..."}}
- {{"action": "STOP", "reason": "have enough evidence or tried all options"}}

IMPORTANT:
- Do NOT repeat a tool call that already returned 0 results
- Try DIFFERENT tools or DIFFERENT parameters
- If previous calls failed, try hybrid_search or lexical_search with the topic terms
- If you have 50+ chunks, stop
- If you've tried 4+ different approaches with no results, stop"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast model for tool selection
                messages=[
                    {"role": "system", "content": "You select search tools to find evidence. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=500,
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
                
        except Exception as e:
            if self.config.verbose:
                print(f"    [Searcher] Error: {e}", file=sys.stderr)
        
        return None
    
    def _get_tools_description(self) -> str:
        """Get tool descriptions for the searcher."""
        
        lines = ["AVAILABLE TOOLS:"]
        
        key_tools = ["hybrid_search", "lexical_search", "entity_mentions", "co_mention_entities", "lexical_exact"]
        
        for name in key_tools:
            if name in TOOL_REGISTRY:
                spec = TOOL_REGISTRY[name]
                params = []
                for pname, pinfo in spec.params_schema.items():
                    if pinfo.get("required", False):
                        params.append(f"{pname} (required)")
                    elif pname in ["query", "term", "entity_id", "name", "top_k"]:
                        params.append(f"{pname}")
                
                lines.append(f"  {name}: {spec.description[:100]}")
                if params:
                    lines.append(f"    params: {', '.join(params[:4])}")
        
        return "\n".join(lines)
    
    def _call_tool(self, tool_name: str, params: Dict[str, Any], conn) -> ToolResult:
        """Call a retrieval tool."""
        
        # Always rollback before calling to clear any bad transaction state
        try:
            conn.rollback()
        except:
            pass
        
        if tool_name not in TOOL_REGISTRY:
            if self.config.verbose:
                print(f"    [!] Tool '{tool_name}' not in TOOL_REGISTRY", file=sys.stderr)
            return ToolResult(
                tool_name=tool_name, params=params, chunk_ids=[], scores={},
                metadata={}, elapsed_ms=0, success=False, error="unknown tool",
            )
        
        tool_spec = TOOL_REGISTRY[tool_name]
        
        # V7 Phase 2: Auto-inject exclusion params for search tools
        search_tools = {"hybrid_search", "vector_search", "lexical_search", "lexical_exact"}
        if tool_name in search_tools and self.config.exclude_seen_mode != "off":
            # Inject exclude_chunk_ids if we have seen chunks
            if self.seen_chunk_ids:
                # Only inject if not already provided (soft mode allows override)
                if self.config.exclude_seen_mode == "hard" or "exclude_chunk_ids" not in params:
                    params["exclude_chunk_ids"] = list(self.seen_chunk_ids)
                    if self.config.verbose:
                        print(f"    [Novelty] Injected exclude_chunk_ids ({len(self.seen_chunk_ids)} IDs)", file=sys.stderr)
            
            # Optionally inject page-level exclusion for broader deduplication
            if self.seen_page_ids and len(self.seen_page_ids) < 500:  # Reasonable limit
                if self.config.exclude_seen_mode == "hard" or "exclude_page_ids" not in params:
                    params["exclude_page_ids"] = list(self.seen_page_ids)
            
            # Apply top_k budget multiplier when exclusion is active
            if self.seen_chunk_ids and "top_k" in params:
                original_k = params.get("top_k", 200)
                params["top_k"] = min(int(original_k * self.config.top_k_budget_multiplier), 500)
        
        try:
            result = tool_spec.fn(conn, **params)
            # Log if the result indicates an error
            if not result.success:
                if self.config.verbose:
                    print(f"    [!] Tool returned error: {result.error}", file=sys.stderr)
                # Rollback after error
                try:
                    conn.rollback()
                except:
                    pass
            elif not result.chunk_ids and self.config.verbose:
                # Log if no results but success
                meta_msg = result.metadata.get("message", "")
                if meta_msg:
                    print(f"    [!] No results: {meta_msg}", file=sys.stderr)
            return result
        except Exception as e:
            if self.config.verbose:
                print(f"    [!] Tool exception: {type(e).__name__}: {e}", file=sys.stderr)
            try:
                conn.rollback()
            except:
                pass
            return ToolResult(
                tool_name=tool_name, params=params, chunk_ids=[], scores={},
                metadata={}, elapsed_ms=0, success=False, error=str(e),
            )
    
    def _load_chunks(self, chunk_ids: List[int], conn) -> List[Dict[str, Any]]:
        """Load chunk data."""
        
        if not chunk_ids:
            return []
        
        try:
            # Rollback first to clear any bad transaction state
            try:
                conn.rollback()
            except:
                pass
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.id, COALESCE(c.clean_text, c.text), cm.document_id, cm.first_page_id, cm.collection_slug
                    FROM chunks c
                    LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE c.id = ANY(%s)
                """, (chunk_ids[:200],))
                
                return [
                    {
                        "id": row[0],
                        "text": row[1] or "",
                        "doc_id": row[2],
                        "page_id": row[3],  # V7 Phase 2: Include raw page_id for seen tracking
                        "page": f"p{row[3]}" if row[3] else "",
                        "source_label": row[4] or "",
                    }
                    for row in cur.fetchall()
                ]
        except Exception as e:
            if self.config.verbose:
                print(f"    [!] _load_chunks error: {e}", file=sys.stderr)
            try:
                conn.rollback()
            except:
                pass
            return []
    
    def _synthesize(self, parsed: ParsedQuery) -> tuple:
        """Synthesize answer from bottlenecked spans."""
        
        import os
        
        # CRITICAL: If no evidence, return "insufficient evidence" - DO NOT hallucinate
        if not self.all_bottleneck_spans:
            if self.config.verbose:
                print(f"    [Synthesis] NO EVIDENCE - refusing to synthesize", file=sys.stderr)
            return "Insufficient evidence found. No documents matched the search criteria.", []
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_synthesis(parsed)
        
        # Build context from bottlenecked spans ONLY
        spans_context = []
        for i, span in enumerate(self.all_bottleneck_spans[:self.config.max_bottleneck_spans]):
            spans_context.append(
                f"[{i}] ({span.source_label}, p.{span.page})\n"
                f'"{span.span_text}"\n'
                f"Supports: {span.claim_supported}"
            )
        
        spans_text = "\n\n".join(spans_context)
        
        # Build prompt based on task type
        if parsed.task_type == TaskType.ROSTER_ENUMERATION:
            task_instruction = """
You must output a LIST OF PEOPLE who were members.
Each person must cite evidence by index [0], [1], etc.
If you cannot identify members with citations, say "Insufficient evidence."
Do NOT list organizations as members.
Do NOT say "associated with" - either they were members or they weren't."""
        else:
            task_instruction = "Answer the question using ONLY the provided evidence."
        
        prompt = f"""QUESTION: {parsed.original_query}

EVIDENCE (cite by index):
{spans_text}

{task_instruction}

Output JSON:
{{
  "answer": "Your answer text with [0], [1] citations",
  "claims": [
    {{"claim": "specific claim", "evidence_ids": [0, 1]}}
  ]
}}"""
        
        try:
            from openai import OpenAI
            import json
            
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.config.synthesis_model,
                messages=[
                    {"role": "system", "content": "You synthesize answers from evidence. Cite every claim."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                return data.get("answer", ""), data.get("claims", [])
            
        except Exception as e:
            if self.config.verbose:
                print(f"    [Synthesis] Error: {e}", file=sys.stderr)
        
        return self._fallback_synthesis(parsed)
    
    def _fallback_synthesis(self, parsed: ParsedQuery) -> tuple:
        """Fallback synthesis when LLM fails."""
        
        if not self.all_bottleneck_spans:
            return "Insufficient evidence found.", []
        
        # Simple bullet point list
        lines = [f"Based on the evidence ({len(self.all_bottleneck_spans)} spans):"]
        claims = []
        
        for i, span in enumerate(self.all_bottleneck_spans[:10]):
            if span.claim_supported:
                lines.append(f"• {span.claim_supported} [{i}]")
                claims.append({"claim": span.claim_supported, "evidence_ids": [i]})
        
        return "\n".join(lines), claims
