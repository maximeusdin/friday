"""
V5 Runner - Entry point for LLM-only grading agentic retrieval

Two versions:
- V5 (basic): Simple grading + evidence store
- V5.2 (enhanced): Rerank + Tournament + Hypotheses

Usage:
    from retrieval.agent.v5_runner import run_v5_query, run_v5_2_query
    
    # Basic V5
    result = run_v5_query(conn, "Who were members of the Silvermaster network?")
    
    # Enhanced V5.2 with rerank + hypotheses
    result = run_v5_2_query(conn, "Who were members of the Silvermaster network?")
"""
import sys
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from retrieval.agent.v5_types import V5Budgets, V5Trace, EvidenceItem
from retrieval.agent.v5_controller import Controller
from retrieval.agent.v5_controller_v2 import ControllerV2, V5BudgetsV2


# =============================================================================
# V5 Result
# =============================================================================

@dataclass
class V5Result:
    """Result of a V5 query execution."""
    
    question: str
    answer: str
    claims: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    stopped_reason: str
    trace: V5Trace
    
    @property
    def success(self) -> bool:
        """Whether the query completed with a valid answer."""
        return self.stopped_reason == "answer_accepted"
    
    def format_answer_with_citations(self) -> str:
        """Format the answer with inline citations."""
        # Build evidence lookup
        evidence_by_id = {e["evidence_id"]: e for e in self.evidence}
        
        lines = [self.answer, "", "CITATIONS:"]
        
        for claim_data in self.claims:
            claim = claim_data.get("claim", "")
            evidence_ids = claim_data.get("evidence_ids", [])
            
            lines.append(f"\n• {claim}")
            for eid in evidence_ids:
                ev = evidence_by_id.get(eid)
                if ev:
                    source = ev.get("source_label", "unknown source")
                    page = ev.get("page", "")
                    quote = ev.get("span_text", "")[:200]
                    lines.append(f"  [{eid}] ({source}, p.{page})")
                    lines.append(f"    \"{quote}...\"")
        
        return "\n".join(lines)


# =============================================================================
# V5 Runner Class
# =============================================================================

class V5Runner:
    """
    Runner for V5 agentic retrieval.
    
    This is the main entry point for the LLM-only grading architecture.
    """
    
    def __init__(
        self,
        budgets: Optional[V5Budgets] = None,
        verbose: bool = True,
    ):
        self.budgets = budgets or V5Budgets()
        self.verbose = verbose
    
    def run(
        self,
        question: str,
        conn,
        conversation_context: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> V5Result:
        """
        Run a query through the V5 system.
        
        Args:
            question: The user's question
            conn: Database connection
            conversation_context: Optional context from conversation
            filters: Optional filters (collection, date range)
        
        Returns:
            V5Result with answer, citations, and trace
        """
        if self.verbose:
            print(f"\n[V5 Runner] Starting query: {question[:80]}...", file=sys.stderr)
        
        # Create controller
        controller = Controller(
            budgets=self.budgets,
            verbose=self.verbose,
        )
        
        # Run the loop
        trace = controller.run(
            question=question,
            conn=conn,
            conversation_context=conversation_context,
            filters=filters,
        )
        
        # Get final evidence
        evidence_items = controller.get_final_evidence()
        evidence_dicts = [
            {
                "evidence_id": item.evidence_id,
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "page": item.page,
                "span_text": item.span_text,
                "source_label": item.source_label,
                "support_strength": item.support_strength,
                "quote_grade": item.quote_grade,
                "claim_supported": item.claim_supported,
            }
            for item in evidence_items
        ]
        
        return V5Result(
            question=question,
            answer=trace.final_answer,
            claims=trace.final_claims,
            evidence=evidence_dicts,
            stopped_reason=trace.stopped_reason,
            trace=trace,
        )


# =============================================================================
# Convenience Function
# =============================================================================

def run_v5_query(
    conn,
    question: str,
    max_steps: int = 12,
    evidence_budget: int = 25,
    verbose: bool = True,
    conversation_context: Optional[str] = None,
) -> V5Result:
    """
    Convenience function to run a V5 query.
    
    Args:
        conn: Database connection
        question: The user's question
        max_steps: Maximum tool calls (default: 12)
        evidence_budget: Max evidence items to keep (default: 25)
        verbose: Print progress to stderr (default: True)
        conversation_context: Optional conversation context
    
    Returns:
        V5Result with answer, citations, and trace
    
    Example:
        result = run_v5_query(conn, "Who were members of the Silvermaster network?")
        print(result.answer)
        print(result.format_answer_with_citations())
    """
    budgets = V5Budgets(
        max_steps=max_steps,
        evidence_budget=evidence_budget,
    )
    
    runner = V5Runner(budgets=budgets, verbose=verbose)
    return runner.run(question, conn, conversation_context)


# =============================================================================
# V5.2 Runner (Enhanced with Rerank + Hypotheses)
# =============================================================================

class V5RunnerV2:
    """
    Enhanced V5 runner with:
    - Span extraction (1-2 citeable spans per chunk)
    - Reranking (score and filter spans)
    - Tournament comparison (pairwise eviction)
    - Working hypotheses (guide search iteratively)
    """
    
    def __init__(
        self,
        budgets: Optional[V5BudgetsV2] = None,
        verbose: bool = True,
    ):
        self.budgets = budgets or V5BudgetsV2()
        self.verbose = verbose
    
    def run(
        self,
        question: str,
        conn,
        conversation_context: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> V5Result:
        """Run enhanced V5.2 query."""
        
        if self.verbose:
            print(f"\n[V5.2 Runner] Starting with rerank + hypotheses", file=sys.stderr)
        
        controller = ControllerV2(
            budgets=self.budgets,
            verbose=self.verbose,
        )
        
        trace = controller.run(
            question=question,
            conn=conn,
            conversation_context=conversation_context,
            filters=filters,
        )
        
        # Get final evidence
        evidence_items = controller.get_final_evidence()
        evidence_dicts = [
            {
                "evidence_id": item.evidence_id,
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "page": item.page,
                "span_text": item.span_text,
                "source_label": item.source_label,
                "support_strength": item.support_strength,
                "quote_grade": item.quote_grade,
                "claim_supported": item.claim_supported,
            }
            for item in evidence_items
        ]
        
        # Get supported hypotheses
        supported_h = controller.get_supported_hypotheses()
        
        result = V5Result(
            question=question,
            answer=trace.final_answer,
            claims=trace.final_claims,
            evidence=evidence_dicts,
            stopped_reason=trace.stopped_reason,
            trace=trace,
        )
        
        # Add hypothesis info to result
        result.supported_hypotheses = [h.to_dict() for h in supported_h]
        
        return result


def run_v5_2_query(
    conn,
    question: str,
    max_steps: int = 12,
    evidence_budget: int = 25,
    chunks_to_rerank: int = 100,
    top_spans_to_keep: int = 40,
    verbose: bool = True,
    conversation_context: Optional[str] = None,
) -> V5Result:
    """
    Run enhanced V5.2 query with rerank + hypotheses.
    
    Args:
        conn: Database connection
        question: The user's question
        max_steps: Maximum tool calls
        evidence_budget: Max evidence items to keep
        chunks_to_rerank: Max chunks to extract spans from
        top_spans_to_keep: Spans to keep after rerank
        verbose: Print progress
        conversation_context: Optional context
    
    Returns:
        V5Result with answer, citations, hypotheses, and trace
    """
    budgets = V5BudgetsV2(
        max_steps=max_steps,
        evidence_budget=evidence_budget,
        chunks_to_rerank=chunks_to_rerank,
        top_spans_to_keep=top_spans_to_keep,
    )
    
    runner = V5RunnerV2(budgets=budgets, verbose=verbose)
    return runner.run(question, conn, conversation_context)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for V5 runner."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="V5 Agentic Retrieval")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--max-steps", type=int, default=12, help="Max tool calls")
    parser.add_argument("--evidence-budget", type=int, default=25, help="Max evidence items")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--v2", action="store_true", help="Use V5.2 with rerank + hypotheses")
    
    args = parser.parse_args()
    
    # Get database connection
    from retrieval.db import get_connection
    
    conn = get_connection()
    
    try:
        if args.v2:
            result = run_v5_2_query(
                conn=conn,
                question=args.question,
                max_steps=args.max_steps,
                evidence_budget=args.evidence_budget,
                verbose=not args.quiet,
            )
        else:
            result = run_v5_query(
                conn=conn,
                question=args.question,
                max_steps=args.max_steps,
                evidence_budget=args.evidence_budget,
                verbose=not args.quiet,
            )
        
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result.format_answer_with_citations())
        
        print("\n" + "="*60)
        print("EXECUTION SUMMARY:")
        print("="*60)
        print(f"  Stopped: {result.stopped_reason}")
        print(f"  Steps: {result.trace.total_steps}")
        print(f"  Tournament comparisons: {result.trace.total_grader_calls}")
        print(f"  Evidence items: {len(result.evidence)}")
        print(f"  Elapsed: {result.trace.total_elapsed_ms:.0f}ms")
        
        # Show supported hypotheses if V5.2
        if hasattr(result, 'supported_hypotheses') and result.supported_hypotheses:
            print(f"\nSUPPORTED HYPOTHESES:")
            for h in result.supported_hypotheses[:5]:
                print(f"  ✓ {h['claim'][:60]}...")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
