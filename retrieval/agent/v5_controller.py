"""
V5 Controller - The main orchestration loop

The Controller enforces:
- Budget limits
- Stop contract validation
- The A-E step loop

It does NOT make semantic judgments - that's the Grader's job.
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
from retrieval.agent.v5_grader import Grader
from retrieval.agent.v5_searcher import Searcher, build_observation
from retrieval.agent.tools import TOOL_REGISTRY, ToolResult


# =============================================================================
# Candidate Extractor - Convert tool results to CandidateSpans
# =============================================================================

def extract_candidates(
    tool_result: ToolResult,
    conn,
    max_candidates: int = 30,
) -> List[CandidateSpan]:
    """
    Convert tool results into CandidateSpans for grading.
    
    This is the only place where we do "truncation" - limiting volume.
    No semantic judgment here.
    """
    candidates = []
    
    chunk_ids = tool_result.chunk_ids[:max_candidates]
    
    if not chunk_ids:
        return candidates
    
    # Load chunk texts
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
            
            rows = {row[0]: row for row in cur.fetchall()}
            
            for cid in chunk_ids:
                if cid not in rows:
                    continue
                
                row = rows[cid]
                chunk_id, content, doc_id, page_id, source_label = row
                page_ref = f"p{page_id}" if page_id else ""
                
                # Create candidate
                candidate = CandidateSpan(
                    candidate_id="",  # Will be generated
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=page_ref,
                    span_text=content[:1500] if content else "",  # Truncate very long chunks
                    source_label=source_label or "",
                    source_tool=tool_result.tool_name,
                    score=tool_result.scores.get(cid, 0.0),
                    metadata=tool_result.metadata,
                )
                candidates.append(candidate)
                
    except Exception as e:
        print(f"    [Controller] Error extracting candidates: {e}", file=sys.stderr)
    
    return candidates


# =============================================================================
# Controller Class
# =============================================================================

class Controller:
    """
    The main orchestration loop for V5 agentic retrieval.
    
    Implements the A-E step loop:
    A. Decide next action (Searcher)
    B. Execute tool
    C. Grade candidates (Grader)
    D. Update evidence store
    E. Feed observation back to searcher
    """
    
    def __init__(
        self,
        budgets: Optional[V5Budgets] = None,
        verbose: bool = True,
    ):
        self.budgets = budgets or V5Budgets()
        self.verbose = verbose
        
        # Initialize components
        self.searcher = Searcher(verbose=verbose)
        self.grader = Grader(verbose=verbose)
        
        # State
        self.evidence_store = EvidenceStore(capacity=self.budgets.evidence_budget)
        self.steps_used = 0
        self.grader_calls_used = 0
        self.trace = V5Trace(budgets=self.budgets)
        self.last_observation = "Starting search. No evidence collected yet."
    
    def run(
        self,
        question: str,
        conn,
        conversation_context: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> V5Trace:
        """
        Run the controller loop until stop or budget exhausted.
        
        Returns:
            V5Trace with complete execution history
        """
        start_time = time.time()
        
        self.trace.question = question
        
        if self.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V5 Controller] Question: {question}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"  Budgets: {self.budgets.max_steps} steps, {self.budgets.evidence_budget} evidence slots", 
                  file=sys.stderr)
        
        # Main loop
        while self.steps_used < self.budgets.max_steps:
            step_start = time.time()
            step_log = StepLog(step_number=self.steps_used)
            
            if self.verbose:
                print(f"\n[Step {self.steps_used + 1}/{self.budgets.max_steps}]", file=sys.stderr)
            
            # Step A: Decide next action
            action = self.searcher.decide_action(
                question=question,
                budgets=self.budgets,
                steps_used=self.steps_used,
                grader_calls_used=self.grader_calls_used,
                evidence_store=self.evidence_store,
                last_observation=self.last_observation,
                conversation_context=conversation_context,
            )
            
            # Handle different action types
            if isinstance(action, StopAnswerAction):
                # Validate stop
                step_log.stop_attempted = True
                validation = action.validate_citations(
                    self.evidence_store, 
                    self.budgets.min_citation_strength
                )
                
                if validation["valid"]:
                    step_log.stop_accepted = True
                    step_log.action_type = "STOP_ANSWER"
                    
                    if self.verbose:
                        print(f"  [STOP] Answer accepted with {len(action.major_claims)} cited claims", 
                              file=sys.stderr)
                    
                    self.trace.stopped_reason = "answer_accepted"
                    self.trace.final_answer = action.answer
                    self.trace.final_claims = action.major_claims
                    break
                else:
                    # Reject stop - continue searching
                    step_log.stop_rejection_reason = "; ".join(validation["errors"])
                    self.last_observation = f"Stop rejected: {step_log.stop_rejection_reason}\nKeep searching."
                    
                    if self.verbose:
                        print(f"  [STOP REJECTED] {step_log.stop_rejection_reason}", file=sys.stderr)
                    
                    # Don't count this as a step - let searcher try again
                    continue
            
            elif isinstance(action, StopInsufficientAction):
                step_log.stop_attempted = True
                step_log.stop_accepted = True
                step_log.action_type = "STOP_INSUFFICIENT"
                
                if self.verbose:
                    print(f"  [STOP] Insufficient evidence: {action.what_missing[:100]}", file=sys.stderr)
                
                self.trace.stopped_reason = "insufficient"
                self.trace.final_answer = action.partial_answer
                break
            
            elif isinstance(action, ToolCallAction):
                step_log.action_type = "CALL_TOOL"
                step_log.tool_name = action.tool_name
                step_log.tool_params = action.params
                
                # Step B: Execute tool
                tool_result = self._execute_tool(action, conn)
                
                # Step C: Extract candidates and grade
                candidates = extract_candidates(
                    tool_result, 
                    conn, 
                    max_candidates=self.budgets.max_candidates_per_step
                )
                step_log.candidates_produced = len(candidates)
                
                if self.verbose:
                    print(f"  Tool returned {len(candidates)} candidates", file=sys.stderr)
                
                # Grade candidates
                grades = []
                if candidates:
                    grades = self.grader.grade_candidates(question, candidates)
                    step_log.grader_calls = len(candidates)
                    self.grader_calls_used += len(candidates)
                
                # Step D: Update evidence store
                items_added = []
                items_evicted = []
                supporting_count = 0
                strong_count = 0
                
                for candidate, grade in zip(candidates, grades):
                    step_log.graded_candidates.append({
                        "candidate_id": candidate.candidate_id,
                        "supports": grade.supports_question,
                        "strength": grade.support_strength,
                        "claim": grade.claim_supported,
                    })
                    
                    if grade.supports_question:
                        supporting_count += 1
                        if grade.support_strength >= 2:
                            strong_count += 1
                    
                    # Convert to evidence item
                    evidence_item = self.grader.grade_to_evidence_item(
                        candidate, grade, self.steps_used
                    )
                    
                    if evidence_item:
                        result = self.evidence_store.add_item(evidence_item, self.steps_used)
                        if result["added"]:
                            items_added.append(evidence_item.evidence_id)
                            if result["evicted"]:
                                items_evicted.append(result["evicted"])
                
                step_log.items_added = items_added
                step_log.items_evicted = items_evicted
                
                # Extract concordance info for observation
                concordance_info = self._extract_concordance_info(tool_result)
                
                # Step E: Build observation for next iteration
                self.last_observation = build_observation(
                    tool_name=action.tool_name,
                    candidates_count=len(candidates),
                    items_added=items_added,
                    items_evicted=items_evicted,
                    grader_summary={"supporting": supporting_count, "strong": strong_count},
                    evidence_store=self.evidence_store,
                    concordance_info=concordance_info,
                )
                
                if self.verbose:
                    print(f"  Added {len(items_added)} items, evicted {len(items_evicted)}", file=sys.stderr)
                    print(f"  Evidence store: {len(self.evidence_store)}/{self.budgets.evidence_budget}", 
                          file=sys.stderr)
            
            # Record step
            step_log.elapsed_ms = (time.time() - step_start) * 1000
            self.trace.steps.append(step_log)
            self.steps_used += 1
        
        # Budget exhausted without stop
        if self.trace.stopped_reason == "":
            self.trace.stopped_reason = "budget_exhausted"
            if self.verbose:
                print(f"\n[V5] Budget exhausted after {self.steps_used} steps", file=sys.stderr)
        
        # Finalize trace
        self.trace.total_steps = self.steps_used
        self.trace.total_grader_calls = self.grader_calls_used
        self.trace.final_evidence_count = len(self.evidence_store)
        self.trace.total_elapsed_ms = (time.time() - start_time) * 1000
        
        if self.verbose:
            print(f"\n[V5] Complete: {self.trace.stopped_reason}", file=sys.stderr)
            print(f"  Total steps: {self.trace.total_steps}", file=sys.stderr)
            print(f"  Total grader calls: {self.trace.total_grader_calls}", file=sys.stderr)
            print(f"  Final evidence: {self.trace.final_evidence_count} items", file=sys.stderr)
            print(f"  Elapsed: {self.trace.total_elapsed_ms:.0f}ms", file=sys.stderr)
        
        return self.trace
    
    def _execute_tool(self, action: ToolCallAction, conn) -> ToolResult:
        """Execute a tool and return its result."""
        
        tool_name = action.tool_name
        params = action.params
        
        if self.verbose:
            print(f"  Executing: {tool_name}({params})", file=sys.stderr)
        
        # Get tool spec
        if tool_name not in TOOL_REGISTRY:
            if self.verbose:
                print(f"  [!] Unknown tool: {tool_name}", file=sys.stderr)
            return ToolResult(
                tool_name=tool_name,
                params=params,
                chunk_ids=[],
                scores={},
                metadata={"error": f"Unknown tool: {tool_name}"},
                elapsed_ms=0,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
        
        tool_spec = TOOL_REGISTRY[tool_name]
        
        # Filter params to match schema
        valid_params = {}
        for key, value in params.items():
            if key in tool_spec.params_schema:
                valid_params[key] = value
        
        try:
            start = time.time()
            result = tool_spec.fn(conn, **valid_params)
            elapsed = (time.time() - start) * 1000
            
            if self.verbose:
                print(f"    -> {len(result.chunk_ids)} chunks in {elapsed:.0f}ms", file=sys.stderr)
                
                # Log concordance expansion if it happened
                self._log_concordance_expansion(result)
                
                # Log entity resolution if it happened
                self._log_entity_resolution(result)
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"    -> ERROR: {e}", file=sys.stderr)
            
            try:
                conn.rollback()
            except:
                pass
            
            return ToolResult(
                tool_name=tool_name,
                params=params,
                chunk_ids=[],
                scores={},
                metadata={"error": str(e)},
                elapsed_ms=0,
                success=False,
                error=str(e),
            )
    
    def _extract_concordance_info(self, result: ToolResult) -> Dict[str, Any]:
        """Extract concordance expansion info from tool result for observation."""
        info = {}
        metadata = result.metadata
        
        # Search tool aliases
        if metadata.get("expanded_aliases"):
            info["aliases"] = metadata["expanded_aliases"]
        elif metadata.get("search_terms") and len(metadata.get("search_terms", [])) > 1:
            info["aliases"] = metadata["search_terms"][1:]  # Skip original term
        
        # Entity resolution via concordance
        matched_via = metadata.get("matched_via", "")
        if matched_via and "concordance" in str(matched_via):
            info["entity_matched_via"] = matched_via
        
        return info
    
    def _log_concordance_expansion(self, result: ToolResult):
        """Log concordance expansion details from tool result."""
        metadata = result.metadata
        
        # Check for alias expansion in search tools
        if metadata.get("aliases_expanded"):
            aliases = metadata.get("expanded_aliases", [])
            total = metadata.get("total_alias_count", 0)
            if aliases:
                print(f"    [Concordance] Expanded to {total} aliases: {aliases[:5]}{'...' if len(aliases) > 5 else ''}", 
                      file=sys.stderr)
        
        # Check for alias expansion in lexical_exact
        if metadata.get("search_terms") and len(metadata.get("search_terms", [])) > 1:
            terms = metadata["search_terms"]
            hits = metadata.get("term_hit_counts", {})
            print(f"    [Concordance] Searched {len(terms)} terms (original + aliases):", file=sys.stderr)
            for term in terms[:5]:
                hit_count = hits.get(term, 0)
                if hit_count > 0:
                    print(f"      - '{term}': {hit_count} hits", file=sys.stderr)
        
        # Check for alias expansion in lexical_search
        if metadata.get("alias_summary"):
            summary = metadata["alias_summary"]
            for term, aliases in list(summary.items())[:3]:
                if aliases:
                    print(f"    [Concordance] '{term}' -> {aliases[:3]}{'...' if len(aliases) > 3 else ''}", 
                          file=sys.stderr)
    
    def _log_entity_resolution(self, result: ToolResult):
        """Log entity resolution details from tool result."""
        metadata = result.metadata
        
        # Check for entity resolution via concordance
        matched_via = metadata.get("matched_via", "")
        if matched_via and "concordance" in str(matched_via):
            entity_name = metadata.get("entity_name") or metadata.get("canonical_name", "")
            print(f"    [Concordance] Resolved entity via: {matched_via}", file=sys.stderr)
            if entity_name:
                print(f"      -> Found: {entity_name}", file=sys.stderr)
        
        # Check for entity lookup
        if metadata.get("found") and metadata.get("canonical_name"):
            if matched_via and matched_via != "canonical_name":
                pass  # Already logged above
            else:
                print(f"    [Entity] Found: {metadata['canonical_name']} (id={metadata.get('entity_id')})", 
                      file=sys.stderr)
    
    def get_final_evidence(self) -> List[EvidenceItem]:
        """Get the final evidence items for answer generation."""
        return self.evidence_store.active_items
    
    def get_cited_evidence(self, evidence_ids: List[str]) -> List[EvidenceItem]:
        """Get specific evidence items by ID."""
        return [
            self.evidence_store.items[eid] 
            for eid in evidence_ids 
            if eid in self.evidence_store.items
        ]
