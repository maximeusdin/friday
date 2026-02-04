"""
Agentic Controller - LLM-driven control policy with deterministic execution.

The controller orchestrates the agentic pipeline:
1. Phase 0: Query Analysis (LLM) - decompose query into scope vs anchors
2. Phase 1: Probe (Code) - cheap retrieval to gather signals
3. Phase 2: Plan Patch (LLM) - decide actions based on signals
4. Phase 3: Execute (Code) - deterministic execution of patch
5. Phase 4: Verify (Code) - enforce truth boundaries
6. (Optional) Fix Patch (LLM) - retry if verification fails

Key invariants (code-enforced, non-negotiable):
- All citations must come from FocusBundle spans
- Candidates without supporting spans are not renderable
- Atoms in claims must appear in cited spans
- LLM variants are for retrieval only, never for claims
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from retrieval.query_analysis import QueryAnalysis, analyze_query
from retrieval.observations import ObservationBundle, run_probe
from retrieval.plan_patch import (
    PlanPatch, ExecutionState, ActionOp, Action,
    get_plan_patch, get_fix_patch, execute_patch,
    create_default_patch, create_negative_patch,
)
from retrieval.rendering import RenderedAnswer
from retrieval.verifier_v2 import FocusBundleVerifier, VerificationResult


@dataclass
class ControllerResult:
    """Result from the agentic controller."""
    
    # Final answer
    answer: RenderedAnswer
    
    # Audit trail
    query_analysis: QueryAnalysis = None
    observations: ObservationBundle = None
    patch: PlanPatch = None
    execution_state: ExecutionState = None
    verification: VerificationResult = None
    
    # Metadata
    total_time_ms: float = 0.0
    llm_calls: int = 0
    retries: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "short_answer": self.answer.short_answer if self.answer else None,
            "bullet_count": len(self.answer.bullets) if self.answer else 0,
            "query_analysis": self.query_analysis.to_dict() if self.query_analysis else None,
            "observations": self.observations.to_dict() if self.observations else None,
            "patch_reasoning": self.patch.reasoning if self.patch else None,
            "verification_passed": self.verification.passed if self.verification else None,
            "total_time_ms": self.total_time_ms,
            "llm_calls": self.llm_calls,
            "retries": self.retries,
        }


class AgenticController:
    """
    LLM-driven controller with deterministic execution.
    
    The LLM makes control decisions (expand, retry, render preference).
    Code executes them deterministically and enforces truth boundaries.
    """
    
    def __init__(
        self,
        max_retries: int = 1,
        probe_k: int = 50,
        probe_bundle_top_n: int = 20,
    ):
        """
        Initialize controller.
        
        Args:
            max_retries: Maximum verification retries (default 1)
            probe_k: Chunks to retrieve in probe (default 50)
            probe_bundle_top_n: Spans in probe FocusBundle (default 20)
        """
        self.max_retries = max_retries
        self.probe_k = probe_k
        self.probe_bundle_top_n = probe_bundle_top_n
        self.verifier = FocusBundleVerifier()
    
    def execute(self, query_text: str, conn) -> ControllerResult:
        """
        Execute the agentic pipeline.
        
        Args:
            query_text: User's query
            conn: Database connection
        
        Returns:
            ControllerResult with answer and audit trail
        """
        start_time = time.time()
        llm_calls = 0
        
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[AgenticController] Query: {query_text}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        
        # Phase 0: Query Analysis (LLM)
        print(f"\n[Phase 0: Query Analysis]", file=sys.stderr)
        try:
            query_analysis = analyze_query(query_text, conn)
            llm_calls += 1
            print(f"  Core concepts: {query_analysis.core_concepts}", file=sys.stderr)
            print(f"  Anchor terms: {query_analysis.anchor_terms}", file=sys.stderr)
            print(f"  Scope: {query_analysis.scope_filters}", file=sys.stderr)
            print(f"  Do not anchor: {query_analysis.do_not_anchor}", file=sys.stderr)
            print(f"  Suggested synonyms: {query_analysis.suggested_synonyms}", file=sys.stderr)
        except Exception as e:
            print(f"  Query analysis failed: {e}", file=sys.stderr)
            # Fallback: use query as-is
            query_analysis = QueryAnalysis(
                query_text=query_text,
                core_concepts=[query_text],
                anchor_terms=self._extract_simple_anchors(query_text),
            )
        
        # Phase 1: Probe (Code)
        print(f"\n[Phase 1: Probe Retrieval]", file=sys.stderr)
        observations = run_probe(
            query_analysis=query_analysis,
            conn=conn,
            probe_k=self.probe_k,
            probe_bundle_top_n=self.probe_bundle_top_n,
        )
        
        # Phase 2: Plan Patch (LLM)
        print(f"\n[Phase 2: LLM Control Decision]", file=sys.stderr)
        try:
            patch = get_plan_patch(query_analysis, observations)
            llm_calls += 1
            print(f"  Patch reasoning: {patch.reasoning}", file=sys.stderr)
            print(f"  Actions: {[a.op for a in patch.actions]}", file=sys.stderr)
        except Exception as e:
            print(f"  Plan patch failed: {e}", file=sys.stderr)
            # Fallback: use default patch
            patch = create_default_patch()
        
        # Phase 3: Execute (Code - deterministic)
        print(f"\n[Phase 3: Deterministic Execution]", file=sys.stderr)
        state = execute_patch(patch, query_analysis, conn)
        
        # Phase 4: Verify (Code)
        print(f"\n[Phase 4: Verification]", file=sys.stderr)
        retries = 0
        verification = None
        
        if state.answer and state.focus_bundle:
            verification = self.verifier.verify_answer(
                bullets=state.answer.bullets,
                focus_bundle=state.focus_bundle,
            )
            print(f"  Passed: {verification.passed}", file=sys.stderr)
            if verification.errors:
                print(f"  Errors: {verification.errors[:3]}", file=sys.stderr)
            if verification.warnings:
                print(f"  Warnings: {verification.warnings[:3]}", file=sys.stderr)
            
            # Retry if failed
            while not verification.passed and retries < self.max_retries:
                print(f"\n[Phase 4.{retries+1}: Fix Patch]", file=sys.stderr)
                retries += 1
                
                try:
                    fix_patch = get_fix_patch(
                        verification_errors=verification.errors,
                        state=state,
                        observations=observations,
                    )
                    llm_calls += 1
                    print(f"  Fix reasoning: {fix_patch.reasoning}", file=sys.stderr)
                    
                    # Execute fix
                    for action in fix_patch.actions:
                        if action.op == ActionOp.DROP_BULLETS.value:
                            from retrieval.plan_patch import _execute_drop_bullets
                            _execute_drop_bullets(state, action.params)
                        elif action.op == ActionOp.WEAKEN_LANGUAGE.value:
                            from retrieval.plan_patch import _execute_weaken_language
                            _execute_weaken_language(state, action.params)
                        elif action.op == ActionOp.RETURN_NEGATIVE.value:
                            from retrieval.plan_patch import _execute_return_negative
                            _execute_return_negative(state, action.params)
                    
                    # Re-verify
                    verification = self.verifier.verify_answer(
                        bullets=state.answer.bullets,
                        focus_bundle=state.focus_bundle,
                    )
                    print(f"  Re-verification passed: {verification.passed}", file=sys.stderr)
                    
                except Exception as e:
                    print(f"  Fix patch failed: {e}", file=sys.stderr)
                    break
        
        # Build final answer (fallback if no answer)
        if not state.answer:
            state.answer = RenderedAnswer(
                short_answer="Unable to find relevant evidence.",
                bullets=[],
                focus_bundle_id=None,
                total_candidates=0,
                rendered_count=0,
            )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[AgenticController] Complete in {total_time_ms:.0f}ms, {llm_calls} LLM calls", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        
        return ControllerResult(
            answer=state.answer,
            query_analysis=query_analysis,
            observations=observations,
            patch=patch,
            execution_state=state,
            verification=verification,
            total_time_ms=total_time_ms,
            llm_calls=llm_calls,
            retries=retries,
        )
    
    def _extract_simple_anchors(self, query_text: str) -> List[str]:
        """
        Fallback anchor extraction (simple heuristics).
        
        Used when LLM query analysis fails.
        """
        import re
        
        # Remove common scope words
        scope_words = {'vassiliev', 'notebooks', 'venona', 'documents', 'evidence', 
                      'information', 'soviets', 'soviet', 'russian', 'american'}
        
        # Extract potential anchors
        words = query_text.lower().split()
        anchors = []
        
        # Look for multi-word phrases (2-3 words)
        for i in range(len(words) - 1):
            if words[i] not in scope_words and words[i+1] not in scope_words:
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 5:
                    anchors.append(bigram)
        
        # Also include capitalized terms
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', query_text)
        anchors.extend(caps)
        
        return anchors[:5]


def execute_agentic_query(query_text: str, conn) -> ControllerResult:
    """
    Convenience function to execute an agentic query.
    
    This is the main entry point for the V3 agentic workflow.
    """
    controller = AgenticController()
    return controller.execute(query_text, conn)
