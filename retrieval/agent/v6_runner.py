"""
V6 Runner - Entry point for principled, no-heuristics retrieval

V6 fixes fundamental issues:
1. CONTROL vs CONTENT separation - don't entity-link "Provide"
2. use_for_retrieval flag - don't use random entities as seeds
3. Hard bottleneck - force convergence to 30-50 spans
4. Responsiveness verification - check answer satisfies question
5. Progress-gated rounds - stop thrashing

Usage:
    from retrieval.agent.v6_runner import run_v6_query
    
    result = run_v6_query(conn, "Who were members of the Silvermaster network?")
    print(result.answer)
"""
import sys
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from retrieval.agent.v6_controller import V6Controller, V6Config, V6Trace
from retrieval.agent.v6_responsiveness import ResponsivenessStatus


# =============================================================================
# V6 Result
# =============================================================================

@dataclass
class V6Result:
    """Result of a V6 query."""
    
    question: str
    answer: str
    claims: List[Dict[str, Any]]
    
    # Status
    is_responsive: bool = False
    responsiveness_status: str = ""
    
    # Members (for roster queries)
    members_identified: List[str] = field(default_factory=list)
    
    # Trace
    trace: Optional[V6Trace] = None
    
    # Bottleneck spans - needed for citation document linking
    bottleneck_spans: List[Any] = field(default_factory=list)
    
    def format_answer(self) -> str:
        """Format the answer with citations and V6-specific information."""
        
        lines = [self.answer, ""]
        
        if self.members_identified:
            lines.append(f"MEMBERS IDENTIFIED ({len(self.members_identified)}):")
            for m in self.members_identified[:20]:
                lines.append(f"  • {m}")
        
        if self.claims:
            lines.append("\nCLAIMS:")
            for c in self.claims[:10]:
                claim = c.get("claim", "")
                evidence = c.get("evidence_ids", [])
                lines.append(f"  • {claim[:80]}... (cites: {evidence})")
        
        lines.append(f"\nResponsiveness: {self.responsiveness_status}")
        
        # V6-specific info showing CONTROL vs CONTENT separation
        if self.trace and self.trace.parsed_query:
            pq = self.trace.parsed_query
            lines.append("\n" + "="*50)
            lines.append("V6 QUERY PARSING:")
            lines.append("="*50)
            lines.append(f"  Task type: {pq.task_type.value}")
            lines.append(f"  Topic terms (CONTENT - used for search):")
            for term in pq.topic_terms:
                lines.append(f"    → \"{term}\"")
            lines.append(f"  Control tokens (NOT entity-linked):")
            for token in list(pq.control_tokens)[:8]:
                lines.append(f"    ✗ \"{token}\"")
            if len(pq.control_tokens) > 8:
                lines.append(f"    ... and {len(pq.control_tokens) - 8} more")
        
        if self.trace and self.trace.entity_linking:
            el = self.trace.entity_linking
            lines.append("\nENTITY LINKING:")
            lines.append(f"  Total linked: {el.total_linked}")
            lines.append(f"  Used for retrieval: {el.used_for_retrieval}")
            lines.append(f"  Retrieval entity IDs:")
            for e in el.retrieval_entities:
                lines.append(f"    → [{e.entity_id}] {e.canonical_name}")
        
        return "\n".join(lines)


# =============================================================================
# V6 Runner
# =============================================================================

class V6Runner:
    """Runner for V6 principled retrieval."""
    
    def __init__(
        self,
        config: Optional[V6Config] = None,
    ):
        self.config = config or V6Config()
    
    def run(
        self,
        question: str,
        conn,
    ) -> V6Result:
        """Run a V6 query."""
        
        controller = V6Controller(config=self.config)
        trace = controller.run(question, conn)
        
        # Build result
        result = V6Result(
            question=question,
            answer=trace.final_answer,
            claims=trace.final_claims,
            trace=trace,
            bottleneck_spans=trace.bottleneck_spans or [],
        )
        
        # Set responsiveness
        if trace.responsiveness:
            result.is_responsive = trace.responsiveness.status == ResponsivenessStatus.RESPONSIVE
            result.responsiveness_status = trace.responsiveness.status.value
            result.members_identified = trace.responsiveness.members_with_citations
        
        # Also get members from progress
        if trace.progress_summary:
            all_members = trace.progress_summary.get("members_found", [])
            if all_members and not result.members_identified:
                result.members_identified = all_members
        
        return result


# =============================================================================
# Convenience Function
# =============================================================================

def run_v6_query(
    conn,
    question: str,
    max_bottleneck_spans: int = 40,
    max_rounds: int = 5,
    verbose: bool = True,
) -> V6Result:
    """
    Run a V6 query with principled architecture.
    
    Args:
        conn: Database connection
        question: The user's question
        max_bottleneck_spans: Max spans after bottleneck (forces convergence)
        max_rounds: Max retrieval rounds
        verbose: Print progress
    
    Returns:
        V6Result with answer, claims, and responsiveness check
    """
    config = V6Config(
        max_bottleneck_spans=max_bottleneck_spans,
        max_rounds=max_rounds,
        verbose=verbose,
    )
    
    runner = V6Runner(config=config)
    return runner.run(question, conn)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for V6 runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="V6 Principled Retrieval")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--max-spans", type=int, default=40, help="Max bottleneck spans")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max retrieval rounds")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    from retrieval.db import get_connection
    
    conn = get_connection()
    
    try:
        result = run_v6_query(
            conn=conn,
            question=args.question,
            max_bottleneck_spans=args.max_spans,
            max_rounds=args.max_rounds,
            verbose=not args.quiet,
        )
        
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result.format_answer())
        
        print("\n" + "="*60)
        print("V6 EXECUTION SUMMARY:")
        print("="*60)
        if result.trace:
            pq = result.trace.parsed_query
            el = result.trace.entity_linking
            
            print("\n  [1] QUERY PARSING (CONTROL vs CONTENT):")
            if pq:
                print(f"      Task type: {pq.task_type.value}")
                print(f"      Topic terms (CONTENT - searched): {pq.topic_terms}")
                print(f"      Control tokens (NOT entity-linked): {len(pq.control_tokens)} tokens")
                print(f"        Examples: {list(pq.control_tokens)[:5]}")
            
            print("\n  [2] ENTITY LINKING (topic terms only):")
            if el:
                print(f"      Total linked: {el.total_linked}")
                print(f"      Used for retrieval: {el.used_for_retrieval}")
                print(f"      Rejected control tokens: {el.rejected_control_tokens}")
                print(f"      Retrieval entity IDs: {el.get_retrieval_entity_ids()}")
                for e in el.retrieval_entities:
                    print(f"        → [{e.entity_id}] {e.canonical_name} ({e.retrieval_reason})")
            
            print("\n  [3] RETRIEVAL + BOTTLENECK:")
            print(f"      Rounds: {len(result.trace.rounds)}")
            print(f"      Final evidence spans: {len(result.trace.progress_summary.get('members_found', []))}")
            
            print("\n  [4] RESPONSIVENESS:")
            print(f"      Status: {result.responsiveness_status}")
            print(f"      Members identified: {len(result.members_identified)}")
            
            print(f"\n  Elapsed: {result.trace.total_elapsed_ms:.0f}ms")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
