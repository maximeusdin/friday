"""
V5 Controller V2 - Enhanced with rerank, hypotheses, and tournament grading

Key improvements over V1:
1. Rerank: Extract 1-2 citeable spans from chunks, rate those
2. Tournament: Pairwise comparison for eviction (more stable than absolute scoring)
3. Hypotheses: Form working hypotheses that guide what to search next

This is "learn as you go" - what survives shapes future tool calls.
"""
import sys
import time
from typing import List, Dict, Any, Optional, Union

from retrieval.agent.v5_types import (
    CandidateSpan,
    EvidenceItem,
    EvidenceStore,
    EvidenceStatus,
    V5Budgets,
    V5Trace,
    StepLog,
    ToolCallAction,
    StopAnswerAction,
    StopInsufficientAction,
)
from retrieval.agent.v5_grader import Grader, TournamentGrader
from retrieval.agent.v5_rerank import SpanExtractor, SpanReranker, ExtractedSpan, RerankResult
from retrieval.agent.v5_hypothesis import HypothesisSet, HypothesisGenerator, WorkingHypothesis
from retrieval.agent.v5_searcher import Searcher
from retrieval.agent.tools import TOOL_REGISTRY, ToolResult


# =============================================================================
# Enhanced Budgets
# =============================================================================

class V5BudgetsV2(V5Budgets):
    """Extended budgets for V2 controller."""
    
    # Rerank settings
    chunks_to_rerank: int = 100  # Max chunks to send through rerank
    spans_per_chunk: int = 2  # Max spans to extract per chunk
    top_spans_to_keep: int = 40  # After rerank, keep this many
    
    # Tournament settings
    tournament_threshold: float = 0.6  # Confidence needed to replace
    
    # Hypothesis settings
    max_hypotheses: int = 15
    hypothesis_refresh_interval: int = 3  # Generate new hypotheses every N steps


# =============================================================================
# Enhanced Observation Builder
# =============================================================================

def build_observation_v2(
    tool_name: str,
    chunks_retrieved: int,
    spans_extracted: int,
    spans_after_rerank: int,
    items_added: List[str],
    items_evicted: List[str],
    hypothesis_updates: Dict[str, Any],
    evidence_store: EvidenceStore,
    concordance_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Build rich observation for searcher."""
    
    lines = []
    
    # Tool result with rerank info
    lines.append(f"Tool '{tool_name}': {chunks_retrieved} chunks -> {spans_extracted} spans extracted -> {spans_after_rerank} passed rerank")
    
    # Concordance
    if concordance_info:
        aliases = concordance_info.get("aliases", [])
        if aliases:
            lines.append(f"Concordance aliases used: {aliases[:5]}{'...' if len(aliases) > 5 else ''}")
    
    # Store mutations
    if items_added:
        lines.append(f"Added to store: {', '.join(items_added[:5])}" + 
                    (f" +{len(items_added) - 5} more" if len(items_added) > 5 else ""))
    
    if items_evicted:
        lines.append(f"Evicted (weaker evidence): {', '.join(items_evicted)}")
    
    # Hypothesis updates
    if hypothesis_updates:
        new_h = hypothesis_updates.get("new_hypotheses", [])
        supported = hypothesis_updates.get("newly_supported", [])
        if new_h:
            lines.append(f"New hypotheses formed: {', '.join(new_h[:3])}")
        if supported:
            lines.append(f"Hypotheses now supported: {', '.join(supported[:3])}")
    
    # Store status
    lines.append(f"Evidence store: {len(evidence_store)}/{evidence_store.capacity}")
    
    return "\n".join(lines)


# =============================================================================
# Enhanced Searcher Prompt Section
# =============================================================================

def build_hypothesis_section(hypothesis_set: HypothesisSet) -> str:
    """Build hypothesis section for searcher prompt."""
    
    if not hypothesis_set.hypotheses:
        return ""
    
    lines = ["\nWORKING HYPOTHESES:"]
    
    # Untested - these need evidence
    untested = hypothesis_set.get_untested_hypotheses()
    if untested:
        lines.append("  Need testing:")
        for h in untested[:5]:
            terms = ", ".join(h.search_terms[:3]) if h.search_terms else "no specific terms"
            lines.append(f"    - {h.claim[:50]}... (search: {terms})")
    
    # Supported - these are confirmed
    supported = hypothesis_set.get_supported_hypotheses()
    if supported:
        lines.append("  Supported:")
        for h in supported[:5]:
            lines.append(f"    ✓ {h.claim[:50]}...")
    
    # Search suggestions
    suggestions = hypothesis_set.get_search_suggestions()
    if suggestions:
        lines.append(f"  Suggested searches: {', '.join(suggestions[:5])}")
    
    return "\n".join(lines)


# =============================================================================
# Controller V2
# =============================================================================

class ControllerV2:
    """
    Enhanced V5 controller with rerank, hypotheses, and tournament grading.
    
    The loop:
    1. Searcher decides tool (informed by hypotheses)
    2. Execute tool, get raw chunks
    3. RERANK: Extract spans, score, keep top M
    4. TOURNAMENT: Compare each span to worst in store, swap if better
    5. UPDATE HYPOTHESES: Link evidence, generate new hypotheses
    6. Observation -> loop
    """
    
    def __init__(
        self,
        budgets: Optional[V5BudgetsV2] = None,
        verbose: bool = True,
    ):
        self.budgets = budgets or V5BudgetsV2()
        self.verbose = verbose
        
        # Components
        self.searcher = Searcher(verbose=verbose)
        self.span_extractor = SpanExtractor(verbose=verbose)
        self.span_reranker = SpanReranker(verbose=verbose)
        self.tournament = TournamentGrader(verbose=verbose)
        self.hypothesis_gen = HypothesisGenerator(verbose=verbose)
        
        # State
        self.evidence_store = EvidenceStore(capacity=self.budgets.evidence_budget)
        self.hypothesis_set = HypothesisSet(max_hypotheses=self.budgets.max_hypotheses)
        self.steps_used = 0
        self.trace = V5Trace(budgets=self.budgets)
        self.last_observation = "Starting search. No evidence collected yet."
        
        # Stats
        self.total_chunks_retrieved = 0
        self.total_spans_extracted = 0
        self.total_spans_after_rerank = 0
        self.total_tournament_comparisons = 0
    
    def run(
        self,
        question: str,
        conn,
        conversation_context: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> V5Trace:
        """Run the enhanced controller loop."""
        
        start_time = time.time()
        self.trace.question = question
        
        if self.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V5.2 Controller] Question: {question}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"  Budgets: {self.budgets.max_steps} steps, {self.budgets.evidence_budget} evidence", 
                  file=sys.stderr)
            print(f"  Rerank: {self.budgets.chunks_to_rerank} chunks -> {self.budgets.top_spans_to_keep} spans",
                  file=sys.stderr)
        
        # Main loop
        while self.steps_used < self.budgets.max_steps:
            step_start = time.time()
            step_log = StepLog(step_number=self.steps_used)
            
            if self.verbose:
                print(f"\n[Step {self.steps_used + 1}/{self.budgets.max_steps}]", file=sys.stderr)
            
            # Build enhanced prompt context
            hypothesis_section = build_hypothesis_section(self.hypothesis_set)
            enhanced_context = (conversation_context or "") + hypothesis_section
            
            # Step 1: Searcher decides action
            action = self.searcher.decide_action(
                question=question,
                budgets=self.budgets,
                steps_used=self.steps_used,
                grader_calls_used=self.tournament.total_comparisons,
                evidence_store=self.evidence_store,
                last_observation=self.last_observation,
                conversation_context=enhanced_context,
            )
            
            # Handle stop actions
            if isinstance(action, StopAnswerAction):
                step_log.stop_attempted = True
                validation = action.validate_citations(
                    self.evidence_store, self.budgets.min_citation_strength
                )
                
                if validation["valid"]:
                    step_log.stop_accepted = True
                    step_log.action_type = "STOP_ANSWER"
                    
                    if self.verbose:
                        print(f"  [STOP] Answer accepted", file=sys.stderr)
                    
                    self.trace.stopped_reason = "answer_accepted"
                    self.trace.final_answer = action.answer
                    self.trace.final_claims = action.major_claims
                    break
                else:
                    step_log.stop_rejection_reason = "; ".join(validation["errors"])
                    self.last_observation = f"Stop rejected: {step_log.stop_rejection_reason}"
                    
                    if self.verbose:
                        print(f"  [STOP REJECTED] {step_log.stop_rejection_reason}", file=sys.stderr)
                    continue
            
            elif isinstance(action, StopInsufficientAction):
                step_log.stop_attempted = True
                step_log.stop_accepted = True
                step_log.action_type = "STOP_INSUFFICIENT"
                
                if self.verbose:
                    print(f"  [STOP] Insufficient: {action.what_missing[:80]}", file=sys.stderr)
                
                self.trace.stopped_reason = "insufficient"
                self.trace.final_answer = action.partial_answer
                break
            
            elif isinstance(action, ToolCallAction):
                step_log.action_type = "CALL_TOOL"
                step_log.tool_name = action.tool_name
                step_log.tool_params = action.params
                
                # Step 2: Execute tool
                tool_result = self._execute_tool(action, conn)
                chunks_retrieved = len(tool_result.chunk_ids)
                self.total_chunks_retrieved += chunks_retrieved
                
                # Step 3: Load chunk texts for rerank
                chunks_data = self._load_chunks(
                    conn, 
                    tool_result.chunk_ids[:self.budgets.chunks_to_rerank]
                )
                
                # Step 4: RERANK - Extract spans and score
                rerank_results = self._rerank_chunks(question, chunks_data)
                spans_after_rerank = len(rerank_results)
                self.total_spans_after_rerank += spans_after_rerank
                
                step_log.candidates_produced = spans_after_rerank
                
                # Step 5: TOURNAMENT - Update evidence store
                items_added, items_evicted = self._tournament_update(
                    question, rerank_results
                )
                
                step_log.items_added = items_added
                step_log.items_evicted = items_evicted
                
                # Step 6: UPDATE HYPOTHESES
                hypothesis_updates = self._update_hypotheses(question)
                
                # Build observation
                concordance_info = self._extract_concordance_info(tool_result)
                
                self.last_observation = build_observation_v2(
                    tool_name=action.tool_name,
                    chunks_retrieved=chunks_retrieved,
                    spans_extracted=self.span_extractor.total_spans_extracted,
                    spans_after_rerank=spans_after_rerank,
                    items_added=items_added,
                    items_evicted=items_evicted,
                    hypothesis_updates=hypothesis_updates,
                    evidence_store=self.evidence_store,
                    concordance_info=concordance_info,
                )
                
                if self.verbose:
                    print(f"  Summary: {chunks_retrieved} chunks -> {spans_after_rerank} spans -> "
                          f"+{len(items_added)}/-{len(items_evicted)} store", file=sys.stderr)
            
            # Record step
            step_log.elapsed_ms = (time.time() - step_start) * 1000
            self.trace.steps.append(step_log)
            self.steps_used += 1
        
        # Finalize
        if not self.trace.stopped_reason:
            self.trace.stopped_reason = "budget_exhausted"
        
        self.trace.total_steps = self.steps_used
        self.trace.total_grader_calls = self.tournament.total_comparisons
        self.trace.final_evidence_count = len(self.evidence_store)
        self.trace.total_elapsed_ms = (time.time() - start_time) * 1000
        
        if self.verbose:
            print(f"\n[V5.2] Complete: {self.trace.stopped_reason}", file=sys.stderr)
            print(f"  Steps: {self.trace.total_steps}", file=sys.stderr)
            print(f"  Chunks retrieved: {self.total_chunks_retrieved}", file=sys.stderr)
            print(f"  Spans after rerank: {self.total_spans_after_rerank}", file=sys.stderr)
            print(f"  Tournament comparisons: {self.tournament.total_comparisons}", file=sys.stderr)
            print(f"  Final evidence: {self.trace.final_evidence_count}", file=sys.stderr)
            print(f"  Supported hypotheses: {len(self.hypothesis_set.get_supported_hypotheses())}", file=sys.stderr)
        
        return self.trace
    
    def _execute_tool(self, action: ToolCallAction, conn) -> ToolResult:
        """Execute a tool."""
        
        tool_name = action.tool_name
        params = action.params
        
        if self.verbose:
            print(f"  Executing: {tool_name}", file=sys.stderr)
        
        if tool_name not in TOOL_REGISTRY:
            return ToolResult(
                tool_name=tool_name, params=params, chunk_ids=[], scores={},
                metadata={"error": "unknown tool"}, elapsed_ms=0, success=False,
                error="unknown tool",
            )
        
        tool_spec = TOOL_REGISTRY[tool_name]
        valid_params = {k: v for k, v in params.items() if k in tool_spec.params_schema}
        
        try:
            result = tool_spec.fn(conn, **valid_params)
            
            if self.verbose:
                print(f"    -> {len(result.chunk_ids)} chunks", file=sys.stderr)
                self._log_concordance(result)
            
            return result
        except Exception as e:
            if self.verbose:
                print(f"    -> ERROR: {e}", file=sys.stderr)
            try:
                conn.rollback()
            except:
                pass
            return ToolResult(
                tool_name=tool_name, params=params, chunk_ids=[], scores={},
                metadata={"error": str(e)}, elapsed_ms=0, success=False, error=str(e),
            )
    
    def _load_chunks(self, conn, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Load chunk data for reranking."""
        if not chunk_ids:
            return []
        
        try:
            # Rollback to clear any bad transaction state
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
                """, (chunk_ids,))
                
                return [
                    {
                        "id": row[0],
                        "text": row[1] or "",
                        "doc_id": row[2],
                        "page": f"p{row[3]}" if row[3] else "",
                        "source_label": row[4] or "",
                    }
                    for row in cur.fetchall()
                ]
        except Exception as e:
            if self.verbose:
                print(f"    [!] Error loading chunks: {e}", file=sys.stderr)
            try:
                conn.rollback()
            except:
                pass
            return []
    
    def _rerank_chunks(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]],
    ) -> List[RerankResult]:
        """Extract spans and rerank them."""
        
        if not chunks:
            return []
        
        if self.verbose:
            print(f"  [Rerank] {len(chunks)} chunks...", file=sys.stderr)
        
        # Step 1: Extract spans
        spans = self.span_extractor.extract_spans(question, chunks)
        self.total_spans_extracted += len(spans)
        
        if not spans:
            return []
        
        # Step 2: Rerank
        results = self.span_reranker.rerank(
            question, spans, top_k=self.budgets.top_spans_to_keep
        )
        
        return results
    
    def _tournament_update(
        self,
        question: str,
        rerank_results: List[RerankResult],
    ) -> tuple:
        """Tournament-style update to evidence store."""
        
        items_added = []
        items_evicted = []
        
        for result in rerank_results:
            if not result.useful:
                continue
            
            # Create evidence item from reranked span
            span = result.span
            item = EvidenceItem(
                evidence_id=f"ev_{span.span_id}",
                candidate_id=span.span_id,
                span_text=span.span_text,
                chunk_id=span.chunk_id,
                doc_id=span.doc_id,
                page=span.page,
                source_label=span.source_label,
                support_strength=int(result.score / 3.33),  # Map 0-10 to 0-3
                quote_grade=result.score >= 7.0,
                claim_supported=result.supports_claim,
                grader_notes=span.relevance_reason,
                status=EvidenceStatus.ACTIVE,
                added_at_step=self.steps_used,
            )
            
            # Check if store has room
            if len(self.evidence_store) < self.evidence_store.capacity:
                self.evidence_store.items[item.evidence_id] = item
                items_added.append(item.evidence_id)
            else:
                # Tournament: compare to worst
                worst = self.evidence_store.get_worst_active()
                if worst:
                    should_replace, reason = self.tournament.should_replace(
                        question=question,
                        new_span_text=item.span_text,
                        worst_span_text=worst.span_text,
                        new_claim=item.claim_supported,
                        worst_claim=worst.claim_supported,
                        confidence_threshold=self.budgets.tournament_threshold,
                    )
                    
                    if should_replace:
                        # Evict worst
                        worst.status = EvidenceStatus.EVICTED
                        worst.evicted_at_step = self.steps_used
                        items_evicted.append(worst.evidence_id)
                        
                        # Add new
                        self.evidence_store.items[item.evidence_id] = item
                        items_added.append(item.evidence_id)
                        
                        if self.verbose:
                            print(f"    [Tournament] {item.evidence_id} beat {worst.evidence_id}: {reason[:50]}", 
                                  file=sys.stderr)
        
        return items_added, items_evicted
    
    def _update_hypotheses(self, question: str) -> Dict[str, Any]:
        """Update hypotheses based on new evidence."""
        
        updates = {"new_hypotheses": [], "newly_supported": []}
        
        # Get recent evidence
        recent_evidence = [
            {
                "evidence_id": item.evidence_id,
                "claim_supported": item.claim_supported,
                "span_text": item.span_text[:200],
            }
            for item in self.evidence_store.active_items[:20]
        ]
        
        # Link evidence to existing hypotheses
        self.hypothesis_gen.link_evidence_to_hypotheses(
            self.hypothesis_set, recent_evidence, self.steps_used
        )
        
        # Check for newly supported
        for h in self.hypothesis_set.get_supported_hypotheses():
            if h.last_updated_step == self.steps_used:
                updates["newly_supported"].append(h.claim[:30])
        
        # Generate new hypotheses periodically
        if self.steps_used % self.budgets.hypothesis_refresh_interval == 0:
            new_h = self.hypothesis_gen.generate_hypotheses(
                question=question,
                hypothesis_set=self.hypothesis_set,
                recent_evidence=recent_evidence,
                current_step=self.steps_used,
            )
            updates["new_hypotheses"] = [h.claim[:30] for h in new_h]
        
        return updates
    
    def _extract_concordance_info(self, result: ToolResult) -> Dict[str, Any]:
        """Extract concordance info from tool result."""
        info = {}
        metadata = result.metadata
        
        if metadata.get("expanded_aliases"):
            info["aliases"] = metadata["expanded_aliases"]
        elif metadata.get("search_terms") and len(metadata.get("search_terms", [])) > 1:
            info["aliases"] = metadata["search_terms"][1:]
        
        matched_via = metadata.get("matched_via", "")
        if matched_via and "concordance" in str(matched_via):
            info["entity_matched_via"] = matched_via
        
        return info
    
    def _log_concordance(self, result: ToolResult):
        """Log concordance expansion."""
        metadata = result.metadata
        
        if metadata.get("aliases_expanded"):
            aliases = metadata.get("expanded_aliases", [])[:5]
            if aliases:
                print(f"    [Concordance] Expanded: {aliases}{'...' if len(metadata.get('expanded_aliases', [])) > 5 else ''}", 
                      file=sys.stderr)
        
        matched_via = metadata.get("matched_via", "")
        if matched_via and "concordance" in str(matched_via):
            print(f"    [Concordance] Entity via: {matched_via}", file=sys.stderr)
    
    def get_final_evidence(self) -> List[EvidenceItem]:
        """Get final evidence items."""
        return self.evidence_store.active_items
    
    def get_supported_hypotheses(self) -> List[WorkingHypothesis]:
        """Get hypotheses with evidence support."""
        return self.hypothesis_set.get_supported_hypotheses()
