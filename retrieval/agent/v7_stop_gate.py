"""
V7 Stop Gate - Citation-backed stop validation

The Stop Gate ensures that the agent cannot return an answer unless:
1. Every claim has at least one citation
2. All cited span_ids exist in the evidence store
3. For roster queries: every member has at least one citation

This is a HARD gate - if validation fails, the answer cannot be returned.
"""
import sys
from typing import List, Dict, Any, Optional, Set, Tuple

from retrieval.agent.v7_types import ClaimWithCitation, StopGateResult


# =============================================================================
# Stop Gate
# =============================================================================

class StopGate:
    """
    Validates that all claims have citations before allowing stop.
    
    Rules:
    1. Every claim MUST have non-empty citations list
    2. Every citation MUST reference an existing span in the evidence store
    3. If validation fails, suggest repair or return "insufficient evidence"
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def validate(
        self,
        claims: List[ClaimWithCitation],
        evidence_span_ids: Set[str],
        task_type: str = "general",
        members_identified: List[str] = None,
    ) -> StopGateResult:
        """
        Validate that all claims have valid citations.
        
        Args:
            claims: List of claims to validate
            evidence_span_ids: Set of valid span_ids from evidence store
            task_type: Type of task (affects validation rules)
            members_identified: For roster queries, list of members
        
        Returns:
            StopGateResult with validation outcome
        """
        if self.verbose:
            print(f"  [StopGate] Validating {len(claims)} claims against "
                  f"{len(evidence_span_ids)} evidence spans", file=sys.stderr)
        
        invalid_claims = []
        invalid_citations = []
        
        # Check each claim
        for claim in claims:
            # Check 1: Claim must have at least one citation
            if not claim.citations:
                invalid_claims.append(claim)
                if self.verbose:
                    print(f"    [!] Claim without citations: {claim.claim_text[:50]}...", 
                          file=sys.stderr)
                continue
            
            # Check 2: All citations must reference existing evidence
            for cite_id in claim.citations:
                if cite_id not in evidence_span_ids:
                    if cite_id not in invalid_citations:
                        invalid_citations.append(cite_id)
                    if self.verbose:
                        print(f"    [!] Invalid citation '{cite_id}' in claim: "
                              f"{claim.claim_text[:50]}...", file=sys.stderr)
        
        # Determine outcome
        if invalid_claims or invalid_citations:
            can_stop = False
            reasons = []
            if invalid_claims:
                reasons.append(f"{len(invalid_claims)} claims without citations")
            if invalid_citations:
                reasons.append(f"{len(invalid_citations)} invalid citation references")
            reason = "Validation failed: " + ", ".join(reasons)
        else:
            can_stop = True
            reason = f"All {len(claims)} claims have valid citations"
        
        # Additional check for roster queries
        if task_type == "roster_enumeration" and members_identified:
            uncited_members = self._check_roster_citations(claims, members_identified)
            if uncited_members:
                can_stop = False
                reason += f"; {len(uncited_members)} members without citations"
        
        result = StopGateResult(
            can_stop=can_stop,
            reason=reason,
            invalid_claims=invalid_claims,
            invalid_citations=invalid_citations,
        )
        
        if self.verbose:
            status = "PASSED" if can_stop else "FAILED"
            print(f"    Stop gate: {status} - {reason}", file=sys.stderr)
        
        return result
    
    def _check_roster_citations(
        self,
        claims: List[ClaimWithCitation],
        members: List[str],
    ) -> List[str]:
        """Check that roster members are cited in claims."""
        # Get all member names mentioned in claims
        cited_members = set()
        for claim in claims:
            claim_lower = claim.claim_text.lower()
            for member in members:
                if member.lower() in claim_lower:
                    cited_members.add(member)
        
        # Find uncited members
        uncited = [m for m in members if m not in cited_members]
        return uncited
    
    def suggest_repair(
        self,
        result: StopGateResult,
        evidence_span_ids: Set[str],
    ) -> str:
        """Suggest how to repair validation failures."""
        suggestions = []
        
        if result.invalid_claims:
            suggestions.append(
                f"Remove or rewrite {len(result.invalid_claims)} uncited claims"
            )
        
        if result.invalid_citations:
            valid_examples = list(evidence_span_ids)[:3]
            suggestions.append(
                f"Fix {len(result.invalid_citations)} invalid citations. "
                f"Valid span_ids include: {valid_examples}"
            )
        
        if not suggestions:
            return "No repairs needed"
        
        return "; ".join(suggestions)


# =============================================================================
# Convenience function
# =============================================================================

def validate_stop(
    claims: List[ClaimWithCitation],
    evidence_span_ids: Set[str],
    task_type: str = "general",
    verbose: bool = True,
) -> StopGateResult:
    """
    Convenience function to validate claims have citations.
    
    Returns:
        StopGateResult with can_stop, reason, and invalid items
    """
    gate = StopGate(verbose=verbose)
    return gate.validate(claims, evidence_span_ids, task_type)
