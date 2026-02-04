"""
Answer Trace Generator for Agentic Workflow.

Generates human-readable traces for historians to understand
"Why did I answer this?" - includes:
- What was searched and why
- What candidates were found
- What was dropped and why
- What verification checks passed/failed
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from retrieval.evidence_bundle import (
    EvidenceBundle,
    Claim,
    EntityCandidate,
    RetrievalLaneRun,
)
from retrieval.intent import IntentFamily
from retrieval.verifier import VerificationResult


# =============================================================================
# Lane Summary for Trace
# =============================================================================

@dataclass
class LaneSummary:
    """Summary of a retrieval lane execution."""
    lane_id: str
    query_terms: List[str]
    hit_count: int
    doc_count: int
    top_matches: List[str]
    execution_ms: float
    coverage_achieved: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "query_terms": self.query_terms,
            "hit_count": self.hit_count,
            "doc_count": self.doc_count,
            "top_matches": self.top_matches,
            "execution_ms": self.execution_ms,
            "coverage_achieved": self.coverage_achieved,
        }


# =============================================================================
# Candidate Flow for Trace
# =============================================================================

@dataclass
class CandidateTrace:
    """Trace of a candidate through the workflow."""
    key: str
    display_name: str
    first_seen_round: int
    final_status: str           # "kept" | "dropped" | "merged"
    mention_count: int
    doc_count: int
    evidence_count: int
    drop_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "first_seen_round": self.first_seen_round,
            "final_status": self.final_status,
            "mention_count": self.mention_count,
            "doc_count": self.doc_count,
            "evidence_count": self.evidence_count,
            "drop_reason": self.drop_reason,
        }


# =============================================================================
# Claim Trace
# =============================================================================

@dataclass
class ClaimTrace:
    """Trace of a claim through verification."""
    claim_id: str
    subject: str
    predicate: str
    object: str
    support_type: str
    support_strength: int
    verified: bool
    drop_reason: Optional[str] = None
    evidence_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "support_type": self.support_type,
            "support_strength": self.support_strength,
            "verified": self.verified,
            "drop_reason": self.drop_reason,
            "evidence_count": self.evidence_count,
        }


# =============================================================================
# Main Answer Trace
# =============================================================================

@dataclass
class AnswerTrace:
    """
    Human-readable trace explaining "Why did I answer this?"
    
    For historian users - provides transparency about:
    - What was searched and why
    - What candidates were found/dropped
    - What verification checks passed/failed
    """
    trace_id: str
    query: str
    intent_classified: IntentFamily
    confidence: float
    
    # Lane execution summary
    lanes_executed: List[LaneSummary]
    total_chunks_retrieved: int
    unique_docs: int
    
    # Candidate flow
    candidates_extracted: List[str]
    candidates_kept: List[str]
    candidates_dropped: List[Tuple[str, str]]  # (candidate, reason)
    candidate_traces: List[CandidateTrace]
    
    # Claim flow
    claims_extracted: int
    claims_verified: int
    claims_dropped: int
    claim_traces: List[ClaimTrace]
    
    # Verification
    verifier_checks_passed: List[str]
    verifier_checks_failed: List[str]
    coverage_achieved: Dict[str, float]
    
    # Expansion
    expansions_tried: List[str]
    expansions_validated: List[str]
    expansions_rejected: List[Tuple[str, str]]  # (term, reason)
    
    # Timing
    total_execution_ms: float
    rounds_executed: int
    
    # Rendering
    claims_rendered: int
    citations_included: int
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "intent_classified": self.intent_classified.value,
            "confidence": self.confidence,
            "lanes_executed": [l.to_dict() for l in self.lanes_executed],
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "unique_docs": self.unique_docs,
            "candidates_extracted": self.candidates_extracted,
            "candidates_kept": self.candidates_kept,
            "candidates_dropped": [
                {"candidate": c, "reason": r}
                for c, r in self.candidates_dropped
            ],
            "candidate_traces": [c.to_dict() for c in self.candidate_traces],
            "claims_extracted": self.claims_extracted,
            "claims_verified": self.claims_verified,
            "claims_dropped": self.claims_dropped,
            "claim_traces": [c.to_dict() for c in self.claim_traces],
            "verifier_checks_passed": self.verifier_checks_passed,
            "verifier_checks_failed": self.verifier_checks_failed,
            "coverage_achieved": self.coverage_achieved,
            "expansions_tried": self.expansions_tried,
            "expansions_validated": self.expansions_validated,
            "expansions_rejected": [
                {"term": t, "reason": r}
                for t, r in self.expansions_rejected
            ],
            "total_execution_ms": self.total_execution_ms,
            "rounds_executed": self.rounds_executed,
            "claims_rendered": self.claims_rendered,
            "citations_included": self.citations_included,
            "created_at": self.created_at,
        }
    
    def to_human_readable(self) -> str:
        """Generate human-readable trace summary."""
        lines = []
        
        lines.append(f"=== Answer Trace: {self.trace_id} ===")
        lines.append(f"Query: {self.query}")
        lines.append(f"Intent: {self.intent_classified.value} (confidence: {self.confidence:.2f})")
        lines.append("")
        
        # Lanes
        lines.append("--- Search Lanes ---")
        for lane in self.lanes_executed:
            lines.append(
                f"  {lane.lane_id}: {lane.hit_count} hits in {lane.doc_count} docs "
                f"({lane.execution_ms:.0f}ms)"
            )
        lines.append(f"Total: {self.total_chunks_retrieved} chunks from {self.unique_docs} docs")
        lines.append("")
        
        # Candidates
        lines.append("--- Candidates ---")
        lines.append(f"Extracted: {len(self.candidates_extracted)}")
        lines.append(f"Kept: {len(self.candidates_kept)}")
        if self.candidates_dropped:
            lines.append(f"Dropped: {len(self.candidates_dropped)}")
            for candidate, reason in self.candidates_dropped[:5]:
                lines.append(f"  - {candidate}: {reason}")
        lines.append("")
        
        # Claims
        lines.append("--- Claims ---")
        lines.append(f"Extracted: {self.claims_extracted}")
        lines.append(f"Verified: {self.claims_verified}")
        lines.append(f"Dropped: {self.claims_dropped}")
        lines.append("")
        
        # Verification
        lines.append("--- Verification ---")
        if self.verifier_checks_passed:
            lines.append(f"Passed: {', '.join(self.verifier_checks_passed)}")
        if self.verifier_checks_failed:
            lines.append(f"Failed: {', '.join(self.verifier_checks_failed)}")
        lines.append("")
        
        # Expansions
        if self.expansions_tried:
            lines.append("--- Expansions ---")
            lines.append(f"Tried: {', '.join(self.expansions_tried)}")
            lines.append(f"Validated: {', '.join(self.expansions_validated)}")
            if self.expansions_rejected:
                lines.append("Rejected:")
                for term, reason in self.expansions_rejected[:5]:
                    lines.append(f"  - {term}: {reason}")
            lines.append("")
        
        # Summary
        lines.append("--- Summary ---")
        lines.append(f"Rounds: {self.rounds_executed}")
        lines.append(f"Execution time: {self.total_execution_ms:.0f}ms")
        lines.append(f"Rendered: {self.claims_rendered} claims, {self.citations_included} citations")
        
        return "\n".join(lines)


# =============================================================================
# Trace Generation
# =============================================================================

def generate_answer_trace(
    bundle: EvidenceBundle,
    verification_result: Optional[VerificationResult] = None,
    expansion_results: Optional[Dict[str, Any]] = None,
) -> AnswerTrace:
    """
    Generate human-readable trace explaining:
    - What was searched and why
    - What candidates were found
    - What was dropped and why
    - What verification checks passed/failed
    
    Args:
        bundle: The evidence bundle
        verification_result: Optional verification result
        expansion_results: Optional expansion tracking info
        
    Returns:
        AnswerTrace with full audit trail
    """
    # Generate trace ID
    trace_key = f"{bundle.plan.query_text if bundle.plan else ''}|{datetime.utcnow().isoformat()}"
    trace_id = hashlib.sha1(trace_key.encode()).hexdigest()[:16]
    
    # Lane summaries
    lanes_executed = []
    total_chunks = 0
    all_docs = set()
    total_ms = 0.0
    
    for run in bundle.retrieval_runs:
        lanes_executed.append(LaneSummary(
            lane_id=run.lane_id,
            query_terms=run.query_terms,
            hit_count=run.hit_count,
            doc_count=run.doc_count,
            top_matches=run.top_terms_matched,
            execution_ms=run.execution_ms,
            coverage_achieved=run.coverage_achieved,
        ))
        total_chunks += run.hit_count
        total_ms += run.execution_ms
        # Track unique docs (approximation)
        all_docs.update(range(run.doc_count))  # Placeholder
    
    # Candidate tracking
    candidates_extracted = []
    candidates_kept = []
    candidates_dropped = []
    candidate_traces = []
    
    for entity in bundle.entities:
        candidates_extracted.append(entity.display_name)
        candidates_kept.append(entity.display_name)
        
        candidate_traces.append(CandidateTrace(
            key=entity.key,
            display_name=entity.display_name,
            first_seen_round=entity.first_seen_round,
            final_status="kept",
            mention_count=entity.mention_count,
            doc_count=entity.doc_count,
            evidence_count=len(entity.evidence_refs),
        ))
    
    # Track dropped candidates from verification
    if verification_result:
        for claim, reason in verification_result.dropped_claims:
            # Find corresponding candidate
            candidates_dropped.append((claim.subject, reason))
    
    # Claim tracking
    claim_traces = []
    claims_verified = 0
    claims_dropped_count = 0
    
    for claim in bundle.claims:
        verified = True
        drop_reason = None
        
        # Check if claim was dropped
        if verification_result:
            for dropped_claim, reason in verification_result.dropped_claims:
                if dropped_claim.claim_id == claim.claim_id:
                    verified = False
                    drop_reason = reason
                    claims_dropped_count += 1
                    break
        
        if verified:
            claims_verified += 1
        
        claim_traces.append(ClaimTrace(
            claim_id=claim.claim_id,
            subject=claim.subject,
            predicate=claim.predicate.value,
            object=claim.object,
            support_type=claim.support_type.value,
            support_strength=claim.support_strength,
            verified=verified,
            drop_reason=drop_reason,
            evidence_count=len(claim.evidence),
        ))
    
    # Verification checks
    verifier_passed = []
    verifier_failed = []
    coverage_achieved = {}
    
    if verification_result:
        if verification_result.passed:
            verifier_passed.append("overall_verification")
        else:
            verifier_failed.append("overall_verification")
        
        for error in verification_result.errors:
            verifier_failed.append(f"{error.error_type.value}: {error.message}")
        
        coverage_achieved = verification_result.coverage_achieved
    
    # Expansion tracking
    expansions_tried = []
    expansions_validated = []
    expansions_rejected = []
    
    if expansion_results:
        expansions_tried = expansion_results.get("tried", [])
        expansions_validated = expansion_results.get("validated", [])
        expansions_rejected = [
            (term, reason)
            for term, reason in expansion_results.get("rejected", {}).items()
        ]
    
    # Calculate citations
    citations_count = sum(len(c.evidence) for c in bundle.claims)
    
    return AnswerTrace(
        trace_id=trace_id,
        query=bundle.plan.query_text if bundle.plan else "",
        intent_classified=bundle.plan.intent if bundle.plan else IntentFamily.EXISTENCE_EVIDENCE,
        confidence=1.0,  # Would get from intent classification
        lanes_executed=lanes_executed,
        total_chunks_retrieved=total_chunks,
        unique_docs=len(all_docs),
        candidates_extracted=candidates_extracted,
        candidates_kept=candidates_kept,
        candidates_dropped=candidates_dropped,
        candidate_traces=candidate_traces,
        claims_extracted=len(bundle.claims),
        claims_verified=claims_verified,
        claims_dropped=claims_dropped_count,
        claim_traces=claim_traces,
        verifier_checks_passed=verifier_passed,
        verifier_checks_failed=verifier_failed,
        coverage_achieved=coverage_achieved,
        expansions_tried=expansions_tried,
        expansions_validated=expansions_validated,
        expansions_rejected=expansions_rejected,
        total_execution_ms=total_ms,
        rounds_executed=bundle.rounds_executed,
        claims_rendered=claims_verified,
        citations_included=citations_count,
    )


def generate_compact_trace(bundle: EvidenceBundle) -> Dict[str, Any]:
    """
    Generate a compact trace for logging/storage.
    
    Includes key metrics without full candidate/claim details.
    """
    return {
        "query": bundle.plan.query_text if bundle.plan else "",
        "intent": bundle.plan.intent.value if bundle.plan else None,
        "rounds": bundle.rounds_executed,
        "stable": bundle.stable,
        "lanes_count": len(bundle.retrieval_runs),
        "total_hits": sum(r.hit_count for r in bundle.retrieval_runs),
        "entities_count": len(bundle.entities),
        "claims_count": len(bundle.claims),
        "unresolved_count": len(bundle.unresolved_tokens),
    }
