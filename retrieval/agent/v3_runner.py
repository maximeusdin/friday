"""
V3 Runner - Main orchestrator with retry loop.

The runner executes the full V3 pipeline:
1. Generate plan (LLM)
2. Execute tools (deterministic)
3. Build evidence (spans + rerank)
4. Synthesize claims (LLM)
5. Verify (universal rules)
6. Retry if needed (up to max_rounds)

All artifacts are persisted for audit.
"""

import sys
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from retrieval.agent import DEFAULT_BUDGETS, V3_VERSION
from retrieval.agent.v3_plan import AgentPlanV3, generate_plan, revise_plan
from retrieval.agent.executor import ToolExecutor, ExecutionResult, ExecutionTrace
from retrieval.agent.v3_evidence import EvidenceBuilder, EvidenceSet
from retrieval.agent.v3_claims import ClaimBundleV3, synthesize_claims
from retrieval.agent.v3_verifier import VerifierV3, VerificationReport


@dataclass
class V3RunTrace:
    """Complete trace of a V3 run."""
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    final_round: int = 0
    total_elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rounds": self.rounds,
            "final_round": self.final_round,
            "total_elapsed_ms": self.total_elapsed_ms,
        }


@dataclass
class V3Result:
    """Result from V3 runner."""
    claims: ClaimBundleV3
    evidence_set: EvidenceSet
    report: VerificationReport
    trace: V3RunTrace
    plan: AgentPlanV3
    success: bool
    run_id: str = ""
    
    def __post_init__(self):
        if not self.run_id:
            content = f"{self.plan.query_text}:{time.time()}"
            self.run_id = hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "success": self.success,
            "claim_count": len(self.claims.claims),
            "cite_span_count": len(self.evidence_set.cite_spans),
            "verification_passed": self.report.passed,
            "rounds_used": self.trace.final_round,
            "total_elapsed_ms": self.trace.total_elapsed_ms,
            "claims": self.claims.to_dict(),
            "evidence_stats": self.evidence_set.stats.to_dict(),
            "verification": self.report.to_dict(),
            "trace": self.trace.to_dict(),
            "plan": self.plan.to_dict(),
        }


class V3Runner:
    """
    Main V3 orchestrator.
    
    Usage:
        runner = V3Runner()
        result = runner.run("Who were Soviet agents in the OSS?", conn)
        
        if result.success:
            for claim in result.claims.claims:
                print(f"- {claim.text}")
        else:
            print(f"Verification failed: {result.report.errors}")
    """
    
    def __init__(
        self,
        max_rounds: int = None,
        cite_cap: int = None,
        harvest_cap: int = None,
        verbose: bool = True,
    ):
        self.max_rounds = max_rounds or DEFAULT_BUDGETS["max_rounds"]
        self.cite_cap = cite_cap or DEFAULT_BUDGETS["max_cite_spans"]
        self.harvest_cap = harvest_cap or DEFAULT_BUDGETS["max_harvest_spans"]
        self.verbose = verbose
        
        self.executor = ToolExecutor(verbose=verbose)
        self.evidence_builder = EvidenceBuilder(
            cite_cap=self.cite_cap,
            harvest_cap=self.harvest_cap,
            verbose=verbose,
        )
        self.verifier = VerifierV3(verbose=verbose)
    
    def run(
        self,
        query: str,
        conn,
        context: Optional[Dict] = None,
    ) -> V3Result:
        """
        Execute the V3 pipeline.
        
        Args:
            query: User's query
            conn: Database connection
            context: Optional context (available collections, etc.)
        
        Returns:
            V3Result with claims, evidence, verification, and trace
        """
        start_time = time.time()
        trace = V3RunTrace()
        
        if self.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[V3 Runner] Query: {query}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        # Initialize
        plan = None
        execution_result = None
        evidence_set = None
        claims = None
        report = None
        
        round_num = 0
        
        while round_num < self.max_rounds:
            round_num += 1
            round_start = time.time()
            
            if self.verbose:
                print(f"\n[Round {round_num}/{self.max_rounds}]", file=sys.stderr)
            
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
                    [e.details for e in report.errors],
                    evidence_set.stats.to_dict() if evidence_set else {},
                    conn,
                )
            
            if self.verbose:
                print(f"    Steps: {[s.tool_name for s in plan.steps]}", file=sys.stderr)
                print(f"    Reasoning: {plan.reasoning[:100]}...", file=sys.stderr)
            
            # 2. Execute tools
            if self.verbose:
                print(f"  [Execute] Running {len(plan.steps)} tool steps...", file=sys.stderr)
            
            execution_result = self.executor.execute_plan(plan, conn)
            
            if self.verbose:
                print(f"    Retrieved {len(execution_result.chunk_ids)} chunks", file=sys.stderr)
            
            # 3. Build evidence
            evidence_set = self.evidence_builder.build(
                execution_result.chunk_ids,
                query,
                conn,
                execution_result.scores,
            )
            
            # 4. Synthesize claims
            if self.verbose:
                print(f"  [Synthesize] Generating claims from evidence...", file=sys.stderr)
            
            claims = synthesize_claims(query, evidence_set, conn)
            
            if self.verbose:
                print(f"    Generated {len(claims.claims)} claims", file=sys.stderr)
            
            # 5. Verify
            report = self.verifier.verify(claims, evidence_set, conn)
            
            # Record round
            round_elapsed = (time.time() - round_start) * 1000
            trace.rounds.append({
                "round": round_num,
                "plan_hash": plan.plan_hash,
                "chunks_retrieved": len(execution_result.chunk_ids),
                "cite_spans": len(evidence_set.cite_spans),
                "claims_generated": len(claims.claims),
                "verification_passed": report.passed,
                "error_count": len(report.errors),
                "elapsed_ms": round_elapsed,
            })
            
            # Check if passed
            if report.passed:
                if self.verbose:
                    print(f"\n  [Result] Verification PASSED on round {round_num}", file=sys.stderr)
                break
            
            # Check if errors are actionable
            actionable_errors = [e for e in report.errors if e.actionable]
            if not actionable_errors:
                if self.verbose:
                    print(f"\n  [Result] No actionable errors, stopping", file=sys.stderr)
                break
            
            if self.verbose:
                print(f"  [Retry] {len(actionable_errors)} actionable errors, will retry", file=sys.stderr)
        
        trace.final_round = round_num
        trace.total_elapsed_ms = (time.time() - start_time) * 1000
        
        # Build final result
        success = report.passed if report else False
        
        if self.verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            status = "SUCCESS" if success else "PARTIAL"
            print(f"[V3 Runner] {status} in {trace.total_elapsed_ms:.0f}ms ({round_num} rounds)", 
                  file=sys.stderr)
            print(f"  Claims: {len(claims.claims) if claims else 0}", file=sys.stderr)
            print(f"  Cite spans: {len(evidence_set.cite_spans) if evidence_set else 0}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
        
        return V3Result(
            claims=claims or ClaimBundleV3(claims=[], query_text=query, evidence_set_id=""),
            evidence_set=evidence_set or self.evidence_builder._empty_evidence_set(query),
            report=report or VerificationReport(passed=False, errors=[], warnings=[], stats={}),
            trace=trace,
            plan=plan or generate_plan(query, conn, context),
            success=success,
        )
    
    def run_with_plan(
        self,
        plan: AgentPlanV3,
        conn,
    ) -> V3Result:
        """
        Execute a pre-defined plan (for testing/debugging).
        
        Args:
            plan: Pre-defined AgentPlanV3
            conn: Database connection
        
        Returns:
            V3Result from single-round execution
        """
        start_time = time.time()
        trace = V3RunTrace()
        
        # Execute
        execution_result = self.executor.execute_plan(plan, conn)
        evidence_set = self.evidence_builder.build(
            execution_result.chunk_ids,
            plan.query_text,
            conn,
            execution_result.scores,
        )
        claims = synthesize_claims(plan.query_text, evidence_set, conn)
        report = self.verifier.verify(claims, evidence_set, conn)
        
        trace.rounds.append({
            "round": 1,
            "plan_hash": plan.plan_hash,
            "chunks_retrieved": len(execution_result.chunk_ids),
            "cite_spans": len(evidence_set.cite_spans),
            "claims_generated": len(claims.claims),
            "verification_passed": report.passed,
            "error_count": len(report.errors),
            "elapsed_ms": (time.time() - start_time) * 1000,
        })
        trace.final_round = 1
        trace.total_elapsed_ms = (time.time() - start_time) * 1000
        
        return V3Result(
            claims=claims,
            evidence_set=evidence_set,
            report=report,
            trace=trace,
            plan=plan,
            success=report.passed,
        )


def run_v3_query(query: str, conn, **kwargs) -> V3Result:
    """
    Convenience function to run a V3 query.
    
    Args:
        query: User's query
        conn: Database connection
        **kwargs: Additional args for V3Runner
    
    Returns:
        V3Result
    """
    runner = V3Runner(**kwargs)
    return runner.run(query, conn)
