"""
V4.2 Runner - Main orchestrator for V4 agentic workflow with Discovery Loop.

The V4.2 runner orchestrates:
1. Plan generation (reuses V3)
2. Tool execution (reuses V3)
3. Evidence building (reuses V3)
4. **Discovery Loop (NEW V4.2)** - ChatGPT-like iterative retrieval
5. V4 Interpretation (4o reasoning model)
6. V4 Verification (hard/soft checks)
7. V4 Rendering (grounded response builder)
8. Repair loop (interpretation retry + evidence expansion)

The Discovery Loop (Step 4) iterates:
- 4o proposes search/pivot actions
- executor runs tools deterministically  
- evidence is rebuilt/expanded
- 4o sees compact observations and iterates
- stops when coverage is good or budgets hit

Usage:
    from retrieval.agent.v4_runner import run_v4_query
    
    result = run_v4_query("Who were members of the Silvermaster network?", conn)
    
    print(result.response.format_text())
"""

import sys
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from retrieval.agent import DEFAULT_BUDGETS
from retrieval.agent.v3_plan import AgentPlanV3, generate_plan, revise_plan
from retrieval.agent.executor import ToolExecutor, ExecutionResult
from retrieval.agent.v3_evidence import EvidenceBuilder, EvidenceSet
from retrieval.agent.v4_interpret import (
    InterpretationV4,
    PreparedSpan,
    V4_BUDGETS,
    V4_VERSION,
    interpret_evidence,
    prepare_spans_for_interpretation,
    detect_response_shape,
)
from retrieval.agent.v4_verify import (
    V4VerificationReport,
    verify_interpretation,
)
from retrieval.agent.v4_render import (
    V4RenderedResponse,
    render_interpretation,
)
from retrieval.agent.v4_discover import (
    run_discovery,
    DiscoveryTrace,
)
from retrieval.agent.v4_discovery_metrics import (
    DiscoveryBudgets,
    DEFAULT_BUDGETS as DISCOVERY_DEFAULT_BUDGETS,
    THOROUGH_BUDGETS as DISCOVERY_THOROUGH_BUDGETS,
    should_run_discovery,
)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class V4RunTrace:
    """Trace of a V4 run."""
    retrieval_rounds: List[Dict[str, Any]] = field(default_factory=list)
    interpret_rounds: List[Dict[str, Any]] = field(default_factory=list)
    discovery_trace: Optional[Dict[str, Any]] = None  # V4.2 discovery loop trace
    final_retrieval_round: int = 0
    final_interpret_round: int = 0
    discovery_rounds: int = 0  # V4.2 discovery rounds executed
    total_elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_rounds": self.retrieval_rounds,
            "interpret_rounds": self.interpret_rounds,
            "discovery_trace": self.discovery_trace,
            "discovery_rounds": self.discovery_rounds,
            "final_retrieval_round": self.final_retrieval_round,
            "final_interpret_round": self.final_interpret_round,
            "total_elapsed_ms": self.total_elapsed_ms,
        }


@dataclass
class V4Result:
    """Result from V4 runner."""
    query: str
    response_shape: str
    interpretation: InterpretationV4
    verification: V4VerificationReport
    response: V4RenderedResponse
    evidence_set: EvidenceSet
    prepared_spans: List[PreparedSpan]
    plan: AgentPlanV3
    trace: V4RunTrace
    success: bool
    run_id: str = ""
    
    def __post_init__(self):
        if not self.run_id:
            content = f"{self.query}:{time.time()}"
            self.run_id = hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "response_shape": self.response_shape,
            "success": self.success,
            "interpretation": self.interpretation.to_dict(),
            "verification": self.verification.to_dict(),
            "response": self.response.to_dict(),
            "evidence_stats": self.evidence_set.stats.to_dict() if self.evidence_set.stats else {},
            "plan": self.plan.to_dict() if self.plan else {},
            "trace": self.trace.to_dict(),
        }


# =============================================================================
# V4 Runner
# =============================================================================

class V4Runner:
    """
    Main V4.2 orchestrator with Discovery Loop.
    
    Executes the full V4 pipeline with:
    - Optional discovery loop for ChatGPT-like iterative retrieval
    - Repair loop for interpretation + evidence expansion
    """
    
    def __init__(
        self,
        max_retrieval_rounds: int = None,
        max_interpret_rounds: int = None,
        discovery_enabled: bool = True,  # V4.2: Enable discovery loop
        discovery_budgets: DiscoveryBudgets = None,  # V4.2: Discovery budgets
        thorough_mode: bool = False,  # V4.2: Double budgets for thorough search
        verbose: bool = True,
    ):
        self.max_retrieval_rounds = max_retrieval_rounds or V4_BUDGETS["max_retrieval_rounds"]
        self.max_interpret_rounds = max_interpret_rounds or V4_BUDGETS["max_interpret_rounds"]
        self.discovery_enabled = discovery_enabled
        self.thorough_mode = thorough_mode
        self.verbose = verbose
        
        # Set discovery budgets
        if discovery_budgets:
            self.discovery_budgets = discovery_budgets
        elif thorough_mode:
            self.discovery_budgets = DISCOVERY_THOROUGH_BUDGETS
        else:
            self.discovery_budgets = DISCOVERY_DEFAULT_BUDGETS
        
        self.executor = ToolExecutor(verbose=verbose)
        self.evidence_builder = EvidenceBuilder(
            cite_cap=DEFAULT_BUDGETS["max_cite_spans"],
            harvest_cap=DEFAULT_BUDGETS["max_harvest_spans"],
            verbose=verbose,
        )
    
    def run(
        self,
        query: str,
        conn,
        response_shape: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> V4Result:
        """
        Execute the V4 pipeline.
        
        Args:
            query: User's query
            conn: Database connection
            response_shape: Optional shape override
            context: Optional context (available collections, etc.)
        
        Returns:
            V4Result with interpretation, verification, and rendered response
        """
        start_time = time.time()
        trace = V4RunTrace()
        
        if self.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V4 Runner] Query: {query}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        # Detect response shape
        if response_shape is None:
            response_shape = detect_response_shape(query)
            if self.verbose:
                print(f"  [V4] Auto-detected shape: {response_shape}", file=sys.stderr)
        
        # Initialize state
        plan = None
        evidence_set = None
        prepared_spans = None
        interpretation = None
        verification = None
        
        retrieval_round = 0
        
        # =================================================================
        # Retrieval Loop (outer loop)
        # =================================================================
        while retrieval_round < self.max_retrieval_rounds:
            retrieval_round += 1
            retrieval_start = time.time()
            
            if self.verbose:
                print(f"\n[Retrieval Round {retrieval_round}/{self.max_retrieval_rounds}]", file=sys.stderr)
            
            # 1. Generate or revise plan
            if plan is None:
                if self.verbose:
                    print(f"  [Plan] Generating initial plan...", file=sys.stderr)
                plan = generate_plan(query, conn, context)
            else:
                if self.verbose:
                    print(f"  [Plan] Revising plan based on errors...", file=sys.stderr)
                plan = revise_plan(
                    plan,
                    verification.get_error_messages() if verification else [],
                    evidence_set.stats.to_dict() if evidence_set else {},
                    conn,
                )
            
            if self.verbose:
                print(f"    Steps: {[s.tool_name for s in plan.steps]}", file=sys.stderr)
            
            # 2. Execute tools
            if self.verbose:
                print(f"\n  [Execute] Running {len(plan.steps)} tool steps...", file=sys.stderr)
                if plan.reasoning:
                    print(f"  Plan reasoning: {plan.reasoning[:120]}...", file=sys.stderr)
            
            execution_result = self.executor.execute_plan(plan, conn)
            
            if self.verbose:
                print(f"\n    Retrieved {len(execution_result.chunk_ids)} unique chunks", file=sys.stderr)
                if execution_result.trace.errors:
                    print(f"    Errors: {execution_result.trace.errors}", file=sys.stderr)
            
            # 3. Build evidence
            evidence_set = self.evidence_builder.build(
                execution_result.chunk_ids,
                query,
                conn,
                execution_result.scores,
            )
            
            # 3.5 V4.2 Discovery Loop (NEW)
            # Run discovery to expand evidence coverage before interpretation
            discovery_trace_data = None
            if self.discovery_enabled and retrieval_round == 1:
                # Only run discovery on first retrieval round
                should_discover, discover_reason = should_run_discovery(
                    evidence_set, self.discovery_budgets
                )
                
                if should_discover:
                    if self.verbose:
                        print(f"\n  [Discovery] {discover_reason}", file=sys.stderr)
                    
                    # Extract collection constraint from plan if available
                    constraints = {}
                    if context and context.get("collections"):
                        constraints["collections"] = context["collections"]
                    
                    # Run discovery loop
                    expanded_chunks, discovery_trace = run_discovery(
                        query=query,
                        initial_chunk_ids=execution_result.chunk_ids,
                        conn=conn,
                        constraints=constraints,
                        budgets=self.discovery_budgets,
                        verbose=self.verbose,
                    )
                    
                    # Store trace
                    discovery_trace_data = discovery_trace.to_dict()
                    trace.discovery_rounds = len(discovery_trace.rounds)
                    
                    # Rebuild evidence with expanded chunks
                    if len(expanded_chunks) > len(execution_result.chunk_ids):
                        if self.verbose:
                            print(f"\n  [Discovery] Expanded from {len(execution_result.chunk_ids)} "
                                  f"to {len(expanded_chunks)} chunks", file=sys.stderr)
                        
                        # Rebuild evidence set with expanded chunks
                        evidence_set = self.evidence_builder.build(
                            expanded_chunks,
                            query,
                            conn,
                            # Use scores from discovery trace
                            None,
                        )
                elif self.verbose:
                    print(f"  [Discovery] Skipped: {discover_reason}", file=sys.stderr)
            
            trace.discovery_trace = discovery_trace_data
            
            # 4. Prepare spans for interpretation
            if self.verbose:
                print(f"\n  [Prepare] Preparing spans for interpretation...", file=sys.stderr)
            
            prepared_spans = prepare_spans_for_interpretation(
                evidence_set=evidence_set,
                conn=conn,
            )
            
            if self.verbose:
                print(f"    Prepared {len(prepared_spans)} spans", file=sys.stderr)
            
            retrieval_elapsed = (time.time() - retrieval_start) * 1000
            trace.retrieval_rounds.append({
                "round": retrieval_round,
                "chunks_retrieved": len(execution_result.chunk_ids),
                "cite_spans": len(evidence_set.cite_spans),
                "prepared_spans": len(prepared_spans),
                "elapsed_ms": retrieval_elapsed,
            })
            
            # =============================================================
            # Interpretation Loop (inner loop)
            # =============================================================
            interpret_round = 0
            verifier_errors = None
            
            if self.verbose:
                print(f"\n  [Interpret] Starting interpretation with {len(prepared_spans)} evidence spans...", file=sys.stderr)
            
            while interpret_round < self.max_interpret_rounds:
                interpret_round += 1
                interpret_start = time.time()
                
                if self.verbose:
                    print(f"\n  [Interpret Round {interpret_round}/{self.max_interpret_rounds}]", file=sys.stderr)
                
                # 5. Generate interpretation
                interpretation = interpret_evidence(
                    evidence_set=evidence_set,
                    query=query,
                    conn=conn,
                    response_shape=response_shape,
                    verifier_errors=verifier_errors,
                )
                
                # 6. Verify interpretation
                verification = verify_interpretation(
                    interpretation=interpretation,
                    prepared_spans=prepared_spans,
                    conn=conn,
                )
                
                interpret_elapsed = (time.time() - interpret_start) * 1000
                trace.interpret_rounds.append({
                    "retrieval_round": retrieval_round,
                    "interpret_round": interpret_round,
                    "units_generated": len(interpretation.answer_units),
                    "verification_passed": verification.passed,
                    "error_count": len(verification.hard_errors),
                    "elapsed_ms": interpret_elapsed,
                })
                
                # Check if passed (or partial success)
                if verification.passed or verification.stats.get("passed_count", 0) > 0:
                    if self.verbose:
                        passed = verification.stats.get("passed_count", 0)
                        print(f"    [OK] {passed} units passed verification", file=sys.stderr)
                    break
                
                # Stage A: Check if errors are fixable by reinterpretation
                # These are "model didn't follow contract" errors, not "evidence missing"
                stage_a_error_types = {
                    "missing_citations",
                    "invalid_span_idx", 
                    "supporting_phrase_missing",
                    "entity_not_attested",
                }
                
                fixable_by_reinterpret = all(
                    e.error_type in stage_a_error_types
                    for e in verification.hard_errors
                )
                
                if not fixable_by_reinterpret:
                    if self.verbose:
                        print(f"    [!] Errors not fixable by reinterpretation", file=sys.stderr)
                    break  # Can't fix with current evidence
                
                # Prepare errors for retry
                verifier_errors = verification.get_error_messages()
                if self.verbose:
                    print(f"    [Retry Interpretation] {len(verifier_errors)} errors to fix", file=sys.stderr)
            
            trace.final_interpret_round = interpret_round
            
            # Check if we should exit retrieval loop
            if verification.passed or verification.stats.get("passed_count", 0) > 0:
                break
            
            # Stage B: Check if we should expand evidence
            # ONLY expand if:
            # 1. Model explicitly says it can't find support (check diagnostics)
            # 2. AND the needed terms are absent from attest_text
            needs_more_evidence = self._needs_evidence_expansion(
                interpretation, 
                evidence_set, 
                verification,
            )
            
            if not needs_more_evidence:
                if self.verbose:
                    print(f"  [!] Evidence expansion not needed or won't help", file=sys.stderr)
                break
            
            if self.verbose:
                print(f"  [Retry Retrieval] Expanding evidence in next round", file=sys.stderr)
        
        trace.final_retrieval_round = retrieval_round
        trace.total_elapsed_ms = (time.time() - start_time) * 1000
        
        # 7. Render final response
        if self.verbose:
            print(f"\n  [Render] Building response...", file=sys.stderr)
        
        response = render_interpretation(
            interpretation=interpretation,
            report=verification,
            prepared_spans=prepared_spans,
        )
        
        success = verification.passed
        
        if self.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            status = "SUCCESS" if success else "PARTIAL"
            print(f"[V4 Runner] {status} in {trace.total_elapsed_ms:.0f}ms", file=sys.stderr)
            print(f"  Retrieval rounds: {retrieval_round}", file=sys.stderr)
            print(f"  Interpret rounds: {trace.final_interpret_round}", file=sys.stderr)
            print(f"  Units rendered: {response.stats.units_rendered}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
        
        return V4Result(
            query=query,
            response_shape=response_shape,
            interpretation=interpretation,
            verification=verification,
            response=response,
            evidence_set=evidence_set,
            prepared_spans=prepared_spans,
            plan=plan,
            trace=trace,
            success=success,
        )
    
    def _needs_evidence_expansion(
        self,
        interpretation: InterpretationV4,
        evidence_set: EvidenceSet,
        verification: V4VerificationReport,
    ) -> bool:
        """
        Determine if Stage B (evidence expansion) should be triggered.
        
        Only expand evidence if:
        1. Model explicitly says it can't find support (via diagnostics.missing_info_questions)
        2. AND the needed terms are absent from all cite_spans' attest_text
        
        This prevents runaway retrieval when the issue is interpretation, not evidence.
        """
        # Check if model indicated missing information
        if not interpretation.diagnostics.missing_info_questions:
            return False
        
        # Gather all attest_text from evidence
        all_attest = " ".join([
            s.attest_text or s.quote 
            for s in evidence_set.cite_spans
        ]).lower()
        
        # Extract keywords from missing info questions
        for question in interpretation.diagnostics.missing_info_questions:
            keywords = self._extract_keywords(question)
            # If ANY keyword is present in evidence, model should be able to find it
            if any(kw.lower() in all_attest for kw in keywords):
                return False  # Evidence exists, model just missed it - retry interpretation
        
        # Keywords are truly absent - need more evidence
        return True
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract meaningful keywords from a question."""
        # Remove common question words
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'which',
            'is', 'are', 'was', 'were', 'did', 'does', 'do',
            'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
            'and', 'or', 'but', 'about', 'with'
        }
        
        # Simple word tokenization
        words = question.lower().split()
        keywords = [
            w.strip('?.!,;:"\'-') 
            for w in words 
            if len(w) > 2 and w.lower() not in stop_words
        ]
        
        return keywords[:5]  # Max 5 keywords


# =============================================================================
# Convenience Functions
# =============================================================================

def run_v4_query(query: str, conn, **kwargs) -> V4Result:
    """
    Run a V4 query.
    
    Convenience wrapper around V4Runner.
    """
    runner = V4Runner(**kwargs)
    return runner.run(query, conn)


def execute_v4_query(
    conn, 
    session_id: int, 
    query_text: str,
    discovery_enabled: bool = True,
    thorough_mode: bool = False,
) -> Dict[str, Any]:
    """
    Execute a V4.2 query and return result dict.
    
    For CLI integration - matches execute_agentic_v3_query signature.
    
    Args:
        conn: Database connection
        session_id: Research session ID
        query_text: User's query
        discovery_enabled: Enable V4.2 discovery loop (default True)
        thorough_mode: Enable thorough mode with doubled budgets (default False)
    """
    import os
    from retrieval.ops import log_retrieval_run
    
    print("  [V4] Initializing reasoning-first workflow...", file=sys.stderr)
    
    try:
        runner = V4Runner(
            verbose=True,
            discovery_enabled=discovery_enabled,
            thorough_mode=thorough_mode,
        )
        result = runner.run(query_text, conn)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"V4 execution failed: {e}"}
    
    # Create result_set for /summarize compatibility
    result_set_id = None
    if result.evidence_set and result.evidence_set.cite_spans:
        try:
            from scripts.execute_plan import create_result_set
            
            chunk_ids = list({s.chunk_id for s in result.evidence_set.cite_spans})
            chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
            
            run_id = log_retrieval_run(
                conn,
                query_text=f"[V4] {query_text}",
                search_type="hybrid",
                chunk_pv=chunk_pv,
                embedding_model=None,
                top_k=len(chunk_ids),
                returned_chunk_ids=chunk_ids,
                retrieval_config_json={
                    "mode": "v4",
                    "response_shape": result.response_shape,
                    "cite_spans": len(result.evidence_set.cite_spans),
                    "units_rendered": result.response.stats.units_rendered,
                    "verification_passed": result.verification.passed,
                    "retrieval_rounds": result.trace.final_retrieval_round,
                    "interpret_rounds": result.trace.final_interpret_round,
                },
                auto_commit=False,
                session_id=session_id,
            )
            conn.commit()
            
            result_set_id = create_result_set(
                conn,
                run_id=run_id,
                chunk_ids=chunk_ids,
                name=f"V4: {query_text[:40]}... (run {run_id})",
                session_id=session_id,
            )
            print(f"  [V4] Created result set #{result_set_id}", file=sys.stderr)
        except Exception as e:
            print(f"  [V4] Warning: Could not create result set: {e}", file=sys.stderr)
    
    return {
        "mode": "v4",
        "success": result.success,
        "run_id": result.run_id,
        "response_shape": result.response_shape,
        "interpretation": result.interpretation,
        "verification": result.verification,
        "response": result.response,
        "evidence_set": result.evidence_set,
        "prepared_spans": result.prepared_spans,
        "plan": result.plan,
        "trace": result.trace,
        "result_set_id": result_set_id,
    }


def format_v4_result(result: Dict[str, Any]) -> str:
    """Format V4 result for CLI display."""
    if result.get("error"):
        return f"Error: {result['error']}"
    
    response = result.get("response")
    if response:
        return response.format_text()
    
    return "No response generated."
