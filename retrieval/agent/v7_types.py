"""
V7 Core Types - Citation Enforcement Data Structures

V7 extends V6 with citation enforcement: every claim must have at least one citation.
V7 Phase 2 adds RoundSummary for structured decision state between rounds.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


# =============================================================================
# Claim with Citation
# =============================================================================

@dataclass
class ClaimWithCitation:
    """
    An atomic claim with required citation(s).
    
    Every claim in the final answer MUST have at least one citation.
    This is enforced by the StopGate.
    """
    
    claim_text: str  # Single atomic assertion
    citations: List[str]  # span_ids or bundle_ids (MUST be non-empty)
    support_level: Literal["strong", "weak", "inferred"] = "strong"
    
    # Optional metadata
    claim_type: Optional[str] = None  # e.g., "member_identification", "relationship", "date"
    
    def is_valid(self) -> bool:
        """Check if claim has required citations."""
        return len(self.citations) > 0 and bool(self.claim_text.strip())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "citations": self.citations,
            "support_level": self.support_level,
            "claim_type": self.claim_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimWithCitation":
        return cls(
            claim_text=data.get("claim_text", ""),
            citations=data.get("citations", []),
            support_level=data.get("support_level", "strong"),
            claim_type=data.get("claim_type"),
        )


# =============================================================================
# Expanded Summary
# =============================================================================

@dataclass
class ExpandedSummary:
    """
    Final output with enumerated claims, each backed by citations.
    
    This is the researcher-grade output format where:
    - Every claim has at least one citation
    - Unsupported claims are explicitly listed (and excluded from main claims)
    - Evidence is enumerated with full quotes and sources
    """
    
    short_answer: str  # Brief answer (1-2 paragraphs)
    claims: List[ClaimWithCitation] = field(default_factory=list)  # All claims with citations
    unsupported_claims: List[str] = field(default_factory=list)  # Claims that couldn't be cited
    evidence_used: List[str] = field(default_factory=list)  # All span/bundle IDs referenced
    
    # Metadata
    total_claims: int = 0
    valid_claims: int = 0
    dropped_claims: int = 0
    
    def __post_init__(self):
        """Calculate stats after initialization."""
        self.total_claims = len(self.claims) + len(self.unsupported_claims)
        self.valid_claims = len(self.claims)
        self.dropped_claims = len(self.unsupported_claims)
    
    def is_valid(self) -> bool:
        """Check if all claims have citations."""
        return all(c.is_valid() for c in self.claims)
    
    def get_all_citations(self) -> List[str]:
        """Get all unique citation IDs used."""
        all_cites = set()
        for claim in self.claims:
            all_cites.update(claim.citations)
        return list(all_cites)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "short_answer": self.short_answer,
            "claims": [c.to_dict() for c in self.claims],
            "unsupported_claims": self.unsupported_claims,
            "evidence_used": self.evidence_used,
            "total_claims": self.total_claims,
            "valid_claims": self.valid_claims,
            "dropped_claims": self.dropped_claims,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpandedSummary":
        return cls(
            short_answer=data.get("short_answer", ""),
            claims=[ClaimWithCitation.from_dict(c) for c in data.get("claims", [])],
            unsupported_claims=data.get("unsupported_claims", []),
            evidence_used=data.get("evidence_used", []),
        )


# =============================================================================
# V7 Result
# =============================================================================

@dataclass
class V7Result:
    """
    Complete V7 query result.
    
    Extends V6 result with expanded summary and citation validation.
    """
    
    # Core output
    answer: str  # Short answer
    expanded_summary: Optional[ExpandedSummary] = None  # Full claims & citations
    
    # From V6
    claims: List[Dict[str, Any]] = field(default_factory=list)  # Raw claims (for compatibility)
    members_identified: List[str] = field(default_factory=list)  # For roster queries
    responsiveness_status: str = ""
    
    # Citation validation
    all_claims_cited: bool = False  # Did all claims have citations?
    citation_validation_passed: bool = False  # Did stop gate pass?
    
    # Trace (optional, for debugging)
    trace: Any = None  # V6Trace or V7Trace
    
    def format_expanded(self) -> str:
        """Format the expanded summary for display."""
        if not self.expanded_summary:
            return self.answer
        
        lines = []
        
        # Short answer
        lines.append("## Answer")
        lines.append("")
        lines.append(self.expanded_summary.short_answer)
        lines.append("")
        
        # Claims & Citations
        if self.expanded_summary.claims:
            lines.append("## Claims & Citations")
            lines.append("")
            for i, claim in enumerate(self.expanded_summary.claims, 1):
                cite_refs = "".join(f"[{c}]" for c in claim.citations)
                lines.append(f"{i}. {claim.claim_text} {cite_refs}")
            lines.append("")
        
        # Unsupported claims (if any)
        if self.expanded_summary.unsupported_claims:
            lines.append("## Unsupported Claims (excluded)")
            lines.append("")
            for claim in self.expanded_summary.unsupported_claims:
                lines.append(f"- {claim}")
            lines.append("")
        
        # Stats
        lines.append("---")
        lines.append(f"Valid claims: {self.expanded_summary.valid_claims}")
        lines.append(f"Dropped claims: {self.expanded_summary.dropped_claims}")
        lines.append(f"Citation validation: {'PASSED' if self.citation_validation_passed else 'FAILED'}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "expanded_summary": self.expanded_summary.to_dict() if self.expanded_summary else None,
            "claims": self.claims,
            "members_identified": self.members_identified,
            "responsiveness_status": self.responsiveness_status,
            "all_claims_cited": self.all_claims_cited,
            "citation_validation_passed": self.citation_validation_passed,
        }


# =============================================================================
# Stop Gate Result
# =============================================================================

@dataclass
class StopGateResult:
    """Result of stop gate validation."""
    
    can_stop: bool  # Is it safe to return this answer?
    reason: str  # Why (or why not)
    
    invalid_claims: List[ClaimWithCitation] = field(default_factory=list)  # Claims without citations
    invalid_citations: List[str] = field(default_factory=list)  # Citations that don't exist
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_stop": self.can_stop,
            "reason": self.reason,
            "invalid_claims": [c.to_dict() for c in self.invalid_claims],
            "invalid_citations": self.invalid_citations,
        }


# =============================================================================
# V7 Phase 2: RoundSummary - Structured Decision State
# =============================================================================

class LeadPriority(str, Enum):
    """Priority level for actionable leads."""
    HIGH = "high"        # Should pursue immediately
    MEDIUM = "medium"    # Worth exploring if time permits
    LOW = "low"          # Background/optional
    EXHAUSTED = "exhausted"  # Already fully explored


@dataclass
class ActionableLead:
    """
    A lead to follow up in subsequent rounds.
    
    Leads are generated by analyzing current evidence and identifying
    entities, documents, or search strategies worth pursuing.
    """
    
    lead_type: str  # "entity", "document", "term", "codename", "date_range"
    target: str     # Entity name, document ID, search term, etc.
    rationale: str  # Why this lead is worth pursuing
    priority: LeadPriority = LeadPriority.MEDIUM
    
    # Optional specifics
    entity_id: Optional[int] = None
    document_id: Optional[int] = None
    suggested_tool: Optional[str] = None  # e.g., "entity_mentions", "lexical_exact"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_type": self.lead_type,
            "target": self.target,
            "rationale": self.rationale,
            "priority": self.priority.value,
            "entity_id": self.entity_id,
            "document_id": self.document_id,
            "suggested_tool": self.suggested_tool,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionableLead":
        return cls(
            lead_type=data.get("lead_type", "term"),
            target=data.get("target", ""),
            rationale=data.get("rationale", ""),
            priority=LeadPriority(data.get("priority", "medium")),
            entity_id=data.get("entity_id"),
            document_id=data.get("document_id"),
            suggested_tool=data.get("suggested_tool"),
        )


@dataclass
class KeyFinding:
    """
    A key finding from the current round of evidence.
    
    Findings summarize what we've learned, independent of specific claims.
    """
    
    finding: str       # The insight or fact discovered
    confidence: float  # 0.0 - 1.0
    evidence_ids: List[int] = field(default_factory=list)  # Chunk IDs supporting this
    
    # Classification
    finding_type: Optional[str] = None  # "fact", "relationship", "date", "context"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding": self.finding,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "finding_type": self.finding_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyFinding":
        return cls(
            finding=data.get("finding", ""),
            confidence=data.get("confidence", 0.5),
            evidence_ids=data.get("evidence_ids", []),
            finding_type=data.get("finding_type"),
        )


class RoundDecisionType(str, Enum):
    """Decision type for next round."""
    CONTINUE = "continue"       # Keep searching with current strategy
    PIVOT = "pivot"             # Change search strategy
    NARROW = "narrow"           # Focus on specific lead
    EXPAND = "expand"           # Broaden search
    STOP_SUFFICIENT = "stop_sufficient"    # Have enough evidence
    STOP_EXHAUSTED = "stop_exhausted"      # No more productive leads


@dataclass
class RoundSummary:
    """
    V7 Phase 2: Structured summary after each retrieval round.
    
    This captures the decision state between rounds, enabling:
    - Smarter tool selection based on what worked/didn't work
    - Prioritized follow-up on promising leads
    - Clear articulation of what we know vs. what we need
    
    Generated by LLM analysis of round results.
    """
    
    round_number: int
    
    # What we learned
    key_findings: List[KeyFinding] = field(default_factory=list)
    
    # What to do next
    actionable_leads: List[ActionableLead] = field(default_factory=list)
    
    # Decision state
    decision: RoundDecisionType = RoundDecisionType.CONTINUE
    decision_rationale: str = ""
    
    # Progress metrics
    evidence_count: int = 0           # Total evidence spans so far
    new_evidence_count: int = 0       # New evidence this round
    unique_entities_found: int = 0    # Distinct entities discovered
    coverage_estimate: float = 0.0    # Estimated % of relevant docs found (0-1)
    
    # What worked / didn't work
    successful_strategies: List[str] = field(default_factory=list)  # Tools/queries that found evidence
    failed_strategies: List[str] = field(default_factory=list)      # Tools/queries that didn't help
    
    # Gaps identified
    information_gaps: List[str] = field(default_factory=list)  # Questions still unanswered
    
    def get_high_priority_leads(self) -> List[ActionableLead]:
        """Get leads marked as high priority."""
        return [l for l in self.actionable_leads if l.priority == LeadPriority.HIGH]
    
    def should_continue(self) -> bool:
        """Whether to continue with another round."""
        return self.decision in (
            RoundDecisionType.CONTINUE, 
            RoundDecisionType.PIVOT,
            RoundDecisionType.NARROW,
            RoundDecisionType.EXPAND,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "key_findings": [f.to_dict() for f in self.key_findings],
            "actionable_leads": [l.to_dict() for l in self.actionable_leads],
            "decision": self.decision.value,
            "decision_rationale": self.decision_rationale,
            "evidence_count": self.evidence_count,
            "new_evidence_count": self.new_evidence_count,
            "unique_entities_found": self.unique_entities_found,
            "coverage_estimate": self.coverage_estimate,
            "successful_strategies": self.successful_strategies,
            "failed_strategies": self.failed_strategies,
            "information_gaps": self.information_gaps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoundSummary":
        return cls(
            round_number=data.get("round_number", 0),
            key_findings=[KeyFinding.from_dict(f) for f in data.get("key_findings", [])],
            actionable_leads=[ActionableLead.from_dict(l) for l in data.get("actionable_leads", [])],
            decision=RoundDecisionType(data.get("decision", "continue")),
            decision_rationale=data.get("decision_rationale", ""),
            evidence_count=data.get("evidence_count", 0),
            new_evidence_count=data.get("new_evidence_count", 0),
            unique_entities_found=data.get("unique_entities_found", 0),
            coverage_estimate=data.get("coverage_estimate", 0.0),
            successful_strategies=data.get("successful_strategies", []),
            failed_strategies=data.get("failed_strategies", []),
            information_gaps=data.get("information_gaps", []),
        )
    
    def format_for_context(self) -> str:
        """Format summary for inclusion in LLM context."""
        lines = [
            f"=== Round {self.round_number} Summary ===",
            f"Evidence: {self.evidence_count} total ({self.new_evidence_count} new)",
            f"Decision: {self.decision.value} - {self.decision_rationale}",
            "",
        ]
        
        if self.key_findings:
            lines.append("Key Findings:")
            for f in self.key_findings[:5]:
                lines.append(f"  - {f.finding} (confidence: {f.confidence:.1f})")
        
        if self.actionable_leads:
            lines.append("\nActionable Leads:")
            for l in self.actionable_leads[:5]:
                lines.append(f"  [{l.priority.value}] {l.lead_type}: {l.target}")
                lines.append(f"      → {l.rationale}")
        
        if self.information_gaps:
            lines.append("\nInformation Gaps:")
            for gap in self.information_gaps[:3]:
                lines.append(f"  ? {gap}")
        
        if self.failed_strategies:
            lines.append("\nFailed Strategies (avoid repeating):")
            for s in self.failed_strategies[:3]:
                lines.append(f"  ✗ {s}")
        
        return "\n".join(lines)


# =============================================================================
# V7 Phase 2: Evidence Bundles
# =============================================================================

class BundleStatus(str, Enum):
    """Lifecycle status of an evidence bundle."""
    FORMING = "forming"        # Still collecting evidence
    COMPLETE = "complete"      # Has enough evidence
    CITED = "cited"            # Used in a claim
    SUPERSEDED = "superseded"  # Replaced by better bundle


@dataclass
class BundleSpan:
    """A span that belongs to a bundle."""
    span_id: str
    chunk_id: int
    text: str
    relevance_score: float = 0.0  # How relevant to the bundle topic
    source_label: Optional[str] = None
    page: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "relevance_score": self.relevance_score,
            "source_label": self.source_label,
            "page": self.page,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleSpan":
        return cls(
            span_id=data.get("span_id", ""),
            chunk_id=data.get("chunk_id", 0),
            text=data.get("text", ""),
            relevance_score=data.get("relevance_score", 0.0),
            source_label=data.get("source_label"),
            page=data.get("page"),
        )


@dataclass
class EvidenceBundle:
    """
    V7 Phase 2: A collection of related evidence spans grouped by topic.
    
    Evidence bundles enable:
    - Thematic grouping of related evidence
    - Single-citation references (cite bundle_id instead of multiple span_ids)
    - Coherent evidence presentation
    - Redundancy reduction (multiple sources for same fact → one bundle)
    
    Bundle lifecycle:
    1. FORMING: Created when related spans are identified
    2. COMPLETE: Has sufficient evidence (2+ spans, high confidence)
    3. CITED: Used in synthesis (claim references this bundle)
    4. SUPERSEDED: Better evidence found, bundle deprioritized
    """
    
    bundle_id: str  # Unique identifier for citation (e.g., "b_123")
    topic: str      # What this bundle is about (e.g., "Silvermaster's Treasury connections")
    
    # Evidence spans in this bundle
    spans: List[BundleSpan] = field(default_factory=list)
    
    # Status and metadata
    status: BundleStatus = BundleStatus.FORMING
    confidence: float = 0.0       # 0.0 - 1.0 based on evidence strength
    coverage: float = 0.0         # How much of the topic is covered
    
    # Summary (generated by LLM)
    summary: str = ""             # One-paragraph summary of the bundle's evidence
    key_claims: List[str] = field(default_factory=list)  # Claims this bundle supports
    
    # Source tracking
    source_collections: List[str] = field(default_factory=list)
    unique_documents: int = 0
    
    # Timestamps
    created_round: int = 0        # Which round this bundle was created
    last_updated_round: int = 0   # Last round that added evidence
    
    def add_span(self, span: BundleSpan) -> None:
        """Add a span to this bundle."""
        # Check for duplicate
        if span.span_id in {s.span_id for s in self.spans}:
            return
        self.spans.append(span)
        # Update unique documents
        unique_docs = {s.chunk_id // 1000 for s in self.spans}  # Rough heuristic
        self.unique_documents = len(unique_docs)
    
    def span_count(self) -> int:
        """Number of spans in this bundle."""
        return len(self.spans)
    
    def is_sufficient(self, min_spans: int = 2, min_confidence: float = 0.5) -> bool:
        """Check if bundle has sufficient evidence."""
        return self.span_count() >= min_spans and self.confidence >= min_confidence
    
    def get_representative_quote(self) -> str:
        """Get the most representative quote from the bundle."""
        if not self.spans:
            return ""
        # Return the span with highest relevance
        best = max(self.spans, key=lambda s: s.relevance_score)
        return best.text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "topic": self.topic,
            "spans": [s.to_dict() for s in self.spans],
            "status": self.status.value,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "summary": self.summary,
            "key_claims": self.key_claims,
            "source_collections": self.source_collections,
            "unique_documents": self.unique_documents,
            "created_round": self.created_round,
            "last_updated_round": self.last_updated_round,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceBundle":
        return cls(
            bundle_id=data.get("bundle_id", ""),
            topic=data.get("topic", ""),
            spans=[BundleSpan.from_dict(s) for s in data.get("spans", [])],
            status=BundleStatus(data.get("status", "forming")),
            confidence=data.get("confidence", 0.0),
            coverage=data.get("coverage", 0.0),
            summary=data.get("summary", ""),
            key_claims=data.get("key_claims", []),
            source_collections=data.get("source_collections", []),
            unique_documents=data.get("unique_documents", 0),
            created_round=data.get("created_round", 0),
            last_updated_round=data.get("last_updated_round", 0),
        )
    
    def format_for_citation(self) -> str:
        """Format bundle for inclusion in synthesized answer."""
        lines = [f"[{self.bundle_id}] {self.topic}"]
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        lines.append(f"  Evidence: {self.span_count()} spans from {self.unique_documents} documents")
        if self.spans:
            best = self.get_representative_quote()
            if len(best) > 150:
                best = best[:150] + "..."
            lines.append(f'  Quote: "{best}"')
        return "\n".join(lines)


@dataclass
class BundleCollection:
    """
    Collection of evidence bundles for a query.
    
    Manages the lifecycle of bundles: creation, updates, merging, citation.
    """
    
    bundles: List[EvidenceBundle] = field(default_factory=list)
    
    def add_bundle(self, bundle: EvidenceBundle) -> None:
        """Add a bundle to the collection."""
        # Check for existing bundle with same topic
        for existing in self.bundles:
            if existing.topic.lower() == bundle.topic.lower():
                # Merge spans into existing bundle
                for span in bundle.spans:
                    existing.add_span(span)
                existing.confidence = max(existing.confidence, bundle.confidence)
                return
        self.bundles.append(bundle)
    
    def get_bundle(self, bundle_id: str) -> Optional[EvidenceBundle]:
        """Get a bundle by ID."""
        for b in self.bundles:
            if b.bundle_id == bundle_id:
                return b
        return None
    
    def get_sufficient_bundles(self) -> List[EvidenceBundle]:
        """Get bundles with sufficient evidence."""
        return [b for b in self.bundles if b.is_sufficient()]
    
    def get_bundles_for_topic(self, topic: str) -> List[EvidenceBundle]:
        """Get bundles related to a topic."""
        topic_lower = topic.lower()
        return [b for b in self.bundles if topic_lower in b.topic.lower()]
    
    def total_spans(self) -> int:
        """Total spans across all bundles."""
        return sum(b.span_count() for b in self.bundles)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundles": [b.to_dict() for b in self.bundles],
            "total_bundles": len(self.bundles),
            "total_spans": self.total_spans(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleCollection":
        return cls(
            bundles=[EvidenceBundle.from_dict(b) for b in data.get("bundles", [])]
        )
