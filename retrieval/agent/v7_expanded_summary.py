"""
V7 Expanded Summary Renderer - Format claims & citations output

This produces researcher-grade output where:
- Every claim is listed with its citations
- Evidence is enumerated with full quotes and sources
- Unsupported claims are explicitly noted (and excluded)

Output format:
## Answer
<short answer>

## Claims & Citations
1. <claim> [1][2]
2. <claim> [3]

## Evidence
[1] "<quote>" (Vassiliev, p.45)
[2] "<quote>" (Vassiliev, p.46)
"""
import sys
from typing import List, Dict, Any, Optional

from retrieval.agent.v7_types import ClaimWithCitation, ExpandedSummary


# =============================================================================
# Expanded Summary Renderer
# =============================================================================

class ExpandedSummaryRenderer:
    """
    Render claims & citations in researcher-grade format.
    
    Output includes:
    - Short answer (1-2 paragraphs)
    - Enumerated claims with citation references
    - Full evidence with quotes and sources
    - Notes about dropped/unsupported claims
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def render(
        self,
        short_answer: str,
        claims: List[ClaimWithCitation],
        evidence_spans: List[Any],
        unsupported_claims: List[str] = None,
    ) -> ExpandedSummary:
        """
        Render the expanded summary.
        
        Args:
            short_answer: Brief answer text
            claims: Valid claims with citations
            evidence_spans: All evidence spans (for quote lookup)
            unsupported_claims: Claims that couldn't be cited
        
        Returns:
            ExpandedSummary object
        """
        if unsupported_claims is None:
            unsupported_claims = []
        
        # Build span lookup
        span_lookup = self._build_span_lookup(evidence_spans)
        
        # Get all evidence IDs used
        evidence_used = set()
        for claim in claims:
            evidence_used.update(claim.citations)
        
        summary = ExpandedSummary(
            short_answer=short_answer,
            claims=claims,
            unsupported_claims=unsupported_claims,
            evidence_used=list(evidence_used),
        )
        
        if self.verbose:
            print(f"  [Renderer] Created expanded summary:", file=sys.stderr)
            print(f"    Valid claims: {summary.valid_claims}", file=sys.stderr)
            print(f"    Dropped claims: {summary.dropped_claims}", file=sys.stderr)
            print(f"    Evidence used: {len(summary.evidence_used)} spans", file=sys.stderr)
        
        return summary
    
    def _build_span_lookup(self, spans: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Build lookup from span_id to span data."""
        lookup = {}
        for i, span in enumerate(spans):
            if hasattr(span, 'span_id'):
                span_id = span.span_id
            elif isinstance(span, dict):
                span_id = span.get("span_id", f"sp_{i}")
            else:
                span_id = f"sp_{i}"
            
            if hasattr(span, 'to_dict'):
                lookup[span_id] = span.to_dict()
            elif isinstance(span, dict):
                lookup[span_id] = span
            else:
                lookup[span_id] = {"text": str(span)}
        
        return lookup
    
    def format_text(
        self,
        summary: ExpandedSummary,
        evidence_spans: List[Any],
        include_evidence: bool = True,
    ) -> str:
        """
        Format the expanded summary as readable text.
        
        Args:
            summary: The ExpandedSummary object
            evidence_spans: Evidence spans for quote lookup
            include_evidence: Whether to include full evidence section
        
        Returns:
            Formatted text output
        """
        lines = []
        
        # Build span lookup for evidence section
        span_lookup = self._build_span_lookup(evidence_spans)
        
        # --- Answer Section ---
        lines.append("=" * 60)
        lines.append("ANSWER")
        lines.append("=" * 60)
        lines.append("")
        lines.append(summary.short_answer)
        lines.append("")
        
        # --- Claims & Citations Section ---
        if summary.claims:
            lines.append("=" * 60)
            lines.append("CLAIMS & CITATIONS")
            lines.append("=" * 60)
            lines.append("")
            
            for i, claim in enumerate(summary.claims, 1):
                cite_refs = " ".join(f"[{c}]" for c in claim.citations)
                support_marker = ""
                if claim.support_level == "weak":
                    support_marker = " (weak)"
                elif claim.support_level == "inferred":
                    support_marker = " (inferred)"
                
                lines.append(f"{i}. {claim.claim_text}{support_marker} {cite_refs}")
            
            lines.append("")
        
        # --- Evidence Section ---
        if include_evidence and summary.evidence_used:
            lines.append("=" * 60)
            lines.append("EVIDENCE")
            lines.append("=" * 60)
            lines.append("")
            
            for span_id in summary.evidence_used:
                span = span_lookup.get(span_id, {})
                text = span.get("text", span.get("span_text", "[no text]"))[:300]
                source = span.get("source_label", "unknown")
                page = span.get("page", "?")
                
                lines.append(f"[{span_id}] ({source}, p.{page})")
                lines.append(f'  "{text}"')
                lines.append("")
        
        # --- Unsupported Claims Section ---
        if summary.unsupported_claims:
            lines.append("=" * 60)
            lines.append("UNSUPPORTED CLAIMS (excluded from answer)")
            lines.append("=" * 60)
            lines.append("")
            
            for claim in summary.unsupported_claims:
                lines.append(f"- {claim}")
            
            lines.append("")
        
        # --- Stats Section ---
        lines.append("-" * 60)
        lines.append(f"Total claims: {summary.total_claims}")
        lines.append(f"Valid (cited): {summary.valid_claims}")
        lines.append(f"Dropped (unsupported): {summary.dropped_claims}")
        lines.append("-" * 60)
        
        return "\n".join(lines)
    
    def format_json(self, summary: ExpandedSummary) -> Dict[str, Any]:
        """Format the expanded summary as JSON."""
        return summary.to_dict()


# =============================================================================
# Convenience function
# =============================================================================

def render_expanded_summary(
    short_answer: str,
    claims: List[ClaimWithCitation],
    evidence_spans: List[Any],
    unsupported_claims: List[str] = None,
    verbose: bool = True,
) -> ExpandedSummary:
    """
    Convenience function to render expanded summary.
    
    Returns:
        ExpandedSummary object
    """
    renderer = ExpandedSummaryRenderer(verbose=verbose)
    return renderer.render(short_answer, claims, evidence_spans, unsupported_claims)


def format_expanded_text(
    summary: ExpandedSummary,
    evidence_spans: List[Any],
    include_evidence: bool = True,
) -> str:
    """
    Format expanded summary as readable text.
    
    Returns:
        Formatted string
    """
    renderer = ExpandedSummaryRenderer(verbose=False)
    return renderer.format_text(summary, evidence_spans, include_evidence)
