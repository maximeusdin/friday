"""
V7 Runner - Entry point for V7 with citation enforcement

V7 extends V6 with:
1. Claim enumeration - extract atomic claims from answer
2. Stop gate - validate all claims have citations
3. Expanded summary - output format with claims & citations
4. Citation validation - ensure answer trustworthiness

Usage:
    from retrieval.agent.v7_runner import run_v7_query
    
    result = run_v7_query(conn, "Who were members of the Silvermaster network?")
    print(result.format_expanded())  # Full output with claims & citations
"""
import sys
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from retrieval.agent.v6_controller import V6Config
from retrieval.agent.v7_controller import V7Controller, V7Config, V7Trace
from retrieval.agent.v7_types import V7Result, ExpandedSummary
from retrieval.agent.v7_expanded_summary import format_expanded_text


# =============================================================================
# V7 Runner
# =============================================================================

class V7Runner:
    """Runner for V7 retrieval with citation enforcement."""
    
    def __init__(
        self,
        config: Optional[V7Config] = None,
    ):
        self.config = config or V7Config()
    
    def run(
        self,
        question: str,
        conn,
    ) -> V7Result:
        """Run a V7 query."""
        
        controller = V7Controller(config=self.config)
        return controller.run(question, conn)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_v7_query(
    conn,
    question: str,
    max_bottleneck_spans: int = 40,
    max_rounds: int = 5,
    drop_uncited_claims: bool = True,
    verbose: bool = True,
    progress_callback: Optional[Callable[[str, str, str, Dict[str, Any]], None]] = None,
) -> V7Result:
    """
    Run a V7 query with citation enforcement.
    
    Args:
        conn: Database connection
        question: The user's question
        max_bottleneck_spans: Max spans after bottleneck
        max_rounds: Max retrieval rounds
        drop_uncited_claims: If True, drop claims without citations; if False, fail validation
        verbose: Print progress
        progress_callback: Optional callback for streaming progress updates
    
    Returns:
        V7Result with answer, expanded summary, and validation status
    """
    v6_config = V6Config(
        max_bottleneck_spans=max_bottleneck_spans,
        max_rounds=max_rounds,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    
    config = V7Config(
        v6_config=v6_config,
        drop_uncited_claims=drop_uncited_claims,
        verbose=verbose,
        progress_callback=progress_callback,  # Pass to V7 for V7-specific steps
    )
    
    runner = V7Runner(config=config)
    return runner.run(question, conn)


def format_v7_result(
    result: V7Result,
    include_evidence: bool = True,
    include_v6_trace: bool = False,
) -> str:
    """
    Format a V7 result for display.
    
    Args:
        result: V7Result to format
        include_evidence: Include full evidence quotes
        include_v6_trace: Include V6 parsing/linking details
    
    Returns:
        Formatted string
    """
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("V7 ANSWER (with citation enforcement)")
    lines.append("=" * 60)
    lines.append("")
    
    # Validation status
    if result.citation_validation_passed:
        lines.append("[✓] All claims have valid citations")
    else:
        lines.append("[!] Citation validation incomplete")
    
    lines.append("")
    
    # Main answer
    lines.append("ANSWER:")
    lines.append("-" * 40)
    lines.append(result.answer)
    lines.append("")
    
    # Expanded summary
    if result.expanded_summary:
        lines.append("CLAIMS & CITATIONS:")
        lines.append("-" * 40)
        
        for i, claim in enumerate(result.expanded_summary.claims, 1):
            support = f"[{claim.support_level}]" if claim.support_level != "strong" else ""
            cites = ", ".join(claim.citations[:3])
            if len(claim.citations) > 3:
                cites += f", +{len(claim.citations) - 3} more"
            
            lines.append(f"  {i}. {claim.claim_text}")
            lines.append(f"     → Citations: {cites} {support}")
        
        lines.append("")
        
        # Unsupported claims
        if result.expanded_summary.unsupported_claims:
            lines.append("UNSUPPORTED CLAIMS (dropped):")
            lines.append("-" * 40)
            for claim in result.expanded_summary.unsupported_claims:
                lines.append(f"  • {claim}")
            lines.append("")
        
        # Statistics
        lines.append("STATISTICS:")
        lines.append(f"  Total claims extracted: {result.expanded_summary.total_claims}")
        lines.append(f"  Valid claims: {result.expanded_summary.valid_claims}")
        lines.append(f"  Dropped claims: {result.expanded_summary.dropped_claims}")
        lines.append(f"  Evidence used: {len(result.expanded_summary.evidence_used)} spans")
    
    # Members (for roster queries)
    if result.members_identified:
        lines.append("")
        lines.append("MEMBERS IDENTIFIED:")
        lines.append("-" * 40)
        for m in result.members_identified[:20]:
            lines.append(f"  • {m}")
        if len(result.members_identified) > 20:
            lines.append(f"  ... and {len(result.members_identified) - 20} more")
    
    # V6 trace details
    if include_v6_trace and result.trace and hasattr(result.trace, 'v6_trace'):
        v6 = result.trace.v6_trace
        if v6:
            lines.append("")
            lines.append("=" * 60)
            lines.append("V6 PIPELINE DETAILS:")
            lines.append("=" * 60)
            
            if v6.parsed_query:
                pq = v6.parsed_query
                lines.append("")
                lines.append("[1] QUERY PARSING:")
                lines.append(f"    Task type: {pq.task_type.value}")
                lines.append(f"    Topic terms: {pq.topic_terms}")
                lines.append(f"    Control tokens: {len(pq.control_tokens)}")
            
            if v6.entity_linking:
                el = v6.entity_linking
                lines.append("")
                lines.append("[2] ENTITY LINKING:")
                lines.append(f"    Linked: {el.total_linked}")
                lines.append(f"    For retrieval: {el.used_for_retrieval}")
                for e in el.retrieval_entities[:5]:
                    lines.append(f"      → [{e.entity_id}] {e.canonical_name}")
            
            lines.append("")
            lines.append(f"[3] ROUNDS: {len(v6.rounds)}")
            lines.append(f"[4] FINAL SPANS: {len(v6.bottleneck_spans)}")
            lines.append(f"[5] RESPONSIVENESS: {v6.responsiveness.status.value if v6.responsiveness else 'N/A'}")
    
    # Timing
    if result.trace:
        lines.append("")
        lines.append("-" * 40)
        lines.append(f"V7 overhead: {result.trace.v7_elapsed_ms:.0f}ms")
        lines.append(f"Total time: {result.trace.total_elapsed_ms:.0f}ms")
    
    return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for V7 runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="V7 Retrieval with Citation Enforcement")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--max-spans", type=int, default=40, help="Max bottleneck spans")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max retrieval rounds")
    parser.add_argument("--keep-uncited", action="store_true", 
                       help="Keep uncited claims (fail validation instead of dropping)")
    parser.add_argument("--show-evidence", action="store_true", help="Show full evidence quotes")
    parser.add_argument("--show-v6", action="store_true", help="Show V6 pipeline details")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    from retrieval.db import get_connection
    import json
    
    conn = get_connection()
    
    try:
        result = run_v7_query(
            conn=conn,
            question=args.question,
            max_bottleneck_spans=args.max_spans,
            max_rounds=args.max_rounds,
            drop_uncited_claims=not args.keep_uncited,
            verbose=not args.quiet,
        )
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + format_v7_result(
                result,
                include_evidence=args.show_evidence,
                include_v6_trace=args.show_v6,
            ))
            
            # Summary line
            print("\n" + "=" * 60)
            status = "PASSED" if result.citation_validation_passed else "FAILED"
            print(f"Citation Validation: {status}")
            if result.expanded_summary:
                print(f"Claims: {result.expanded_summary.valid_claims} valid, "
                      f"{result.expanded_summary.dropped_claims} dropped")
            print("=" * 60)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
