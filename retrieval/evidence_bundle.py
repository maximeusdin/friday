"""
Evidence Bundle Schema for Agentic Workflow.

The Evidence Bundle is the structured intermediate representation between
the executor and verifier/renderer. It contains all retrieval provenance,
extracted claims, and entity candidates.

Key design decisions:
- Composable predicates (8 stable predicates, interpret at render time)
- Support Type/Strength enables "explicit only" enforcement
- Deterministic IDs via sha1 for stable tracing
- Quote spans capped at 500 chars to keep JSONB reasonable
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.plan import AgenticPlan, BundleConstraints


# =============================================================================
# Predicates (Composable)
# =============================================================================

class Predicate(str, Enum):
    """
    Composable predicates - keep small and stable, interpret at render time.
    
    Network membership becomes composable:
    - associated_with(person, silvermaster_network) + strong evidence phrasing
    - Only render as "member" if evidence phrase matches membership patterns
    """
    MENTIONS = "mentions"                # subject mentions object (concept/entity)
    DESCRIBES = "describes"              # technical description/details
    EVALUATES = "evaluates"              # impact/significance assessment
    ASSOCIATED_WITH = "associated_with"  # weak link
    HANDLED_BY = "handled_by"            # strong handler relationship
    MET_WITH = "met_with"                # meeting/contact
    IDENTIFIED_AS = "identified_as"      # explicit mapping statement
    CODENAME_OF = "codename_of"          # covername mapping claim


class SupportType(str, Enum):
    """How strong is the textual support for a claim."""
    EXPLICIT_STATEMENT = "explicit_statement"  # direct assertion
    DEFINITION = "definition"                   # "X is a..."
    ASSESSMENT = "assessment"                   # evaluation/opinion
    CO_MENTION = "co_mention"                   # appears together


# Quote span storage: char_start/char_end are primary, quote_span capped
MAX_QUOTE_SPAN_CHARS = 500


# =============================================================================
# Evidence Reference
# =============================================================================

@dataclass
class EvidenceRef:
    """
    Reference to evidence in the corpus.
    
    char_start/char_end are primary for reconstruction.
    quote_span is capped at MAX_QUOTE_SPAN_CHARS for JSONB size.
    """
    chunk_id: int
    doc_id: int
    page_ref: str
    char_start: int
    char_end: int
    quote_span: str                       # exact quoted text (capped)
    
    def __post_init__(self):
        """Cap quote_span if needed."""
        if len(self.quote_span) > MAX_QUOTE_SPAN_CHARS:
            self.quote_span = self.quote_span[:MAX_QUOTE_SPAN_CHARS] + "..."
    
    @property
    def ref_id(self) -> str:
        """Deterministic ID for deduplication and tracing."""
        key = f"{self.chunk_id}|{self.char_start}|{self.char_end}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ref_id": self.ref_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_ref": self.page_ref,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "quote_span": self.quote_span,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRef":
        """Create from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            page_ref=data.get("page_ref", ""),
            char_start=data["char_start"],
            char_end=data["char_end"],
            quote_span=data["quote_span"],
        )


# =============================================================================
# Claim
# =============================================================================

@dataclass
class Claim:
    """
    An atomic, verifiable claim extracted from evidence.
    
    Claims use composable predicates and must always have evidence refs
    with quote spans. The verifier checks claims against constraints.
    """
    subject: str                          # entity key or token (e.g., "entity:66161")
    predicate: Predicate
    object: str                           # entity key or token
    evidence: List[EvidenceRef]
    support_type: SupportType             # how strong is the textual support
    support_strength: int                 # ordinal 1-3 (weak/moderate/strong)
    source_lane: str                      # provenance (which lane found this)
    confidence: float = 1.0               # confidence score (0-1)
    
    @property
    def claim_id(self) -> str:
        """
        Deterministic ID: sha1(subject|predicate|object|first_evidence_ref).
        
        Stable across runs for the same claim.
        """
        first_ref = self.evidence[0] if self.evidence else None
        key = f"{self.subject}|{self.predicate.value}|{self.object}"
        if first_ref:
            key += f"|{first_ref.chunk_id}|{first_ref.char_start}|{first_ref.char_end}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]
    
    def has_valid_evidence(self) -> bool:
        """Check if claim has valid evidence with quote spans."""
        return bool(self.evidence) and all(e.quote_span for e in self.evidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate.value,
            "object": self.object,
            "evidence": [e.to_dict() for e in self.evidence],
            "support_type": self.support_type.value,
            "support_strength": self.support_strength,
            "source_lane": self.source_lane,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Create from dictionary."""
        return cls(
            subject=data["subject"],
            predicate=Predicate(data["predicate"]),
            object=data["object"],
            evidence=[EvidenceRef.from_dict(e) for e in data.get("evidence", [])],
            support_type=SupportType(data["support_type"]),
            support_strength=data["support_strength"],
            source_lane=data.get("source_lane", ""),
            confidence=data.get("confidence", 1.0),
        )


# =============================================================================
# Entity Candidate
# =============================================================================

@dataclass
class EntityCandidate:
    """
    Unified candidate representation - can be resolved entity OR unresolved token.
    
    Built from entity_mentions (for resolved) or pattern extraction (for tokens).
    Supports merging evidence across rounds.
    """
    key: str                              # "entity:66161" or "token:TWAIN" (dedup key)
    entity_id: Optional[int]              # set if resolved
    token: Optional[str]                  # set if unresolved (e.g., codename)
    display_name: str                     # canonical name or token for display
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    mention_count: int = 0                # how many times mentioned across chunks
    doc_count: int = 0                    # unique documents
    score: float = 1.0                    # relevance score (aggregated)
    first_seen_round: int = 1             # which round discovered this
    source: str = "mention"               # "mention" | "claim" | "comention"
    
    @classmethod
    def from_entity_mention(
        cls,
        entity_id: int,
        canonical_name: str,
        evidence_ref: EvidenceRef,
        round_num: int,
    ) -> "EntityCandidate":
        """Create from a resolved entity mention."""
        return cls(
            key=f"entity:{entity_id}",
            entity_id=entity_id,
            token=None,
            display_name=canonical_name,
            evidence_refs=[evidence_ref],
            mention_count=1,
            doc_count=1,
            score=1.0,
            first_seen_round=round_num,
            source="mention",
        )
    
    @classmethod
    def from_unresolved_token(
        cls,
        token: str,
        evidence_ref: EvidenceRef,
        round_num: int,
    ) -> "EntityCandidate":
        """Create from an unresolved token (e.g., codename)."""
        return cls(
            key=f"token:{token.upper()}",
            entity_id=None,
            token=token,
            display_name=token,
            evidence_refs=[evidence_ref],
            mention_count=1,
            doc_count=1,
            score=0.8,  # lower confidence for unresolved
            first_seen_round=round_num,
            source="mention",
        )
    
    def merge(self, other: "EntityCandidate") -> None:
        """
        Merge evidence from another candidate with same key.
        
        Used during iterative execution to accumulate evidence.
        """
        assert self.key == other.key, f"Cannot merge different keys: {self.key} vs {other.key}"
        
        # Merge evidence refs (dedupe by ref_id)
        existing_ref_ids = {ref.ref_id for ref in self.evidence_refs}
        for ref in other.evidence_refs:
            if ref.ref_id not in existing_ref_ids:
                self.evidence_refs.append(ref)
                existing_ref_ids.add(ref.ref_id)
        
        # Update counts
        self.mention_count += other.mention_count
        
        # Dedupe docs and update doc_count
        doc_ids = {ref.doc_id for ref in self.evidence_refs}
        self.doc_count = len(doc_ids)
        
        # Take best score
        self.score = max(self.score, other.score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "entity_id": self.entity_id,
            "token": self.token,
            "display_name": self.display_name,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "mention_count": self.mention_count,
            "doc_count": self.doc_count,
            "score": self.score,
            "first_seen_round": self.first_seen_round,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityCandidate":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            entity_id=data.get("entity_id"),
            token=data.get("token"),
            display_name=data["display_name"],
            evidence_refs=[EvidenceRef.from_dict(e) for e in data.get("evidence_refs", [])],
            mention_count=data.get("mention_count", 0),
            doc_count=data.get("doc_count", 0),
            score=data.get("score", 1.0),
            first_seen_round=data.get("first_seen_round", 1),
            source=data.get("source", "mention"),
        )


# =============================================================================
# Retrieval Lane Run
# =============================================================================

@dataclass
class RetrievalLaneRun:
    """
    Record of a single lane execution with coverage stats.
    
    Every lane run must include hit stats for the verifier's
    "no global negative without coverage proof" rule.
    """
    lane_id: str
    query_terms: List[str]
    filters_applied: Dict[str, Any]
    hit_count: int                        # total chunks returned
    doc_count: int                        # unique documents
    unique_pages: int                     # page diversity
    top_terms_matched: List[str]          # which terms actually matched
    sample_refs: List[EvidenceRef]        # sample citations
    execution_ms: float                   # execution time
    coverage_achieved: float              # % of expected coverage (0-1)
    
    # Optional metadata
    round_num: int = 1
    chunk_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lane_id": self.lane_id,
            "query_terms": self.query_terms,
            "filters_applied": self.filters_applied,
            "hit_count": self.hit_count,
            "doc_count": self.doc_count,
            "unique_pages": self.unique_pages,
            "top_terms_matched": self.top_terms_matched,
            "sample_refs": [r.to_dict() for r in self.sample_refs],
            "execution_ms": self.execution_ms,
            "coverage_achieved": self.coverage_achieved,
            "round_num": self.round_num,
            "chunk_ids": self.chunk_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalLaneRun":
        """Create from dictionary."""
        return cls(
            lane_id=data["lane_id"],
            query_terms=data.get("query_terms", []),
            filters_applied=data.get("filters_applied", {}),
            hit_count=data.get("hit_count", 0),
            doc_count=data.get("doc_count", 0),
            unique_pages=data.get("unique_pages", 0),
            top_terms_matched=data.get("top_terms_matched", []),
            sample_refs=[EvidenceRef.from_dict(r) for r in data.get("sample_refs", [])],
            execution_ms=data.get("execution_ms", 0.0),
            coverage_achieved=data.get("coverage_achieved", 0.0),
            round_num=data.get("round_num", 1),
            chunk_ids=data.get("chunk_ids", []),
        )


# =============================================================================
# Chunk With Provenance
# =============================================================================

@dataclass
class ChunkWithProvenance:
    """
    A chunk with provenance tracking across lanes.
    
    Used during iterative execution to:
    - Dedupe chunks across lanes
    - Track which lanes found each chunk
    - Enable lazy loading of chunk text
    """
    chunk_id: int
    doc_id: int
    source_lanes: List[str] = field(default_factory=list)  # which lanes found this
    best_score: float = 0.0
    first_seen_round: int = 1
    text: Optional[str] = None            # loaded lazily for extraction
    
    def add_lane(self, lane_id: str, score: float = 0.0, round_num: int = 1) -> None:
        """Add a lane to provenance tracking."""
        if lane_id not in self.source_lanes:
            self.source_lanes.append(lane_id)
        self.best_score = max(self.best_score, score)
        self.first_seen_round = min(self.first_seen_round, round_num)


# =============================================================================
# Evidence Bundle
# =============================================================================

@dataclass
class EvidenceBundle:
    """
    The complete evidence bundle produced by the executor.
    
    This is the intermediate representation between executor and verifier/renderer.
    The final answer must be rendered only from the verified bundle.
    """
    plan: Optional["AgenticPlan"] = None  # the plan that produced this
    retrieval_runs: List[RetrievalLaneRun] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    entities: List[EntityCandidate] = field(default_factory=list)
    unresolved_tokens: List[str] = field(default_factory=list)
    constraints: Optional["BundleConstraints"] = None
    rounds_executed: int = 0
    stable: bool = False                  # did we reach stability?
    
    # Chunk tracking (for debugging/tracing)
    all_chunks: Dict[int, ChunkWithProvenance] = field(default_factory=dict)
    
    def add_claim(self, claim: Claim) -> None:
        """Add a claim to the bundle (dedupes by claim_id)."""
        existing_ids = {c.claim_id for c in self.claims}
        if claim.claim_id not in existing_ids:
            self.claims.append(claim)
    
    def add_entity(self, entity: EntityCandidate) -> None:
        """Add or merge an entity candidate."""
        for existing in self.entities:
            if existing.key == entity.key:
                existing.merge(entity)
                return
        self.entities.append(entity)
    
    def get_claims_for_entity(self, entity_key: str) -> List[Claim]:
        """Get all claims involving an entity."""
        return [
            c for c in self.claims
            if c.subject == entity_key or c.object == entity_key
        ]
    
    def get_total_evidence_refs(self) -> int:
        """Count total evidence references across all claims."""
        return sum(len(c.evidence) for c in self.claims)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan": self.plan.to_dict() if self.plan else None,
            "retrieval_runs": [r.to_dict() for r in self.retrieval_runs],
            "claims": [c.to_dict() for c in self.claims],
            "entities": [e.to_dict() for e in self.entities],
            "unresolved_tokens": self.unresolved_tokens,
            "rounds_executed": self.rounds_executed,
            "stable": self.stable,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceBundle":
        """Create from dictionary."""
        from retrieval.plan import AgenticPlan, BundleConstraints
        from retrieval.intent import IntentFamily
        
        bundle = cls(
            retrieval_runs=[RetrievalLaneRun.from_dict(r) for r in data.get("retrieval_runs", [])],
            claims=[Claim.from_dict(c) for c in data.get("claims", [])],
            entities=[EntityCandidate.from_dict(e) for e in data.get("entities", [])],
            unresolved_tokens=data.get("unresolved_tokens", []),
            rounds_executed=data.get("rounds_executed", 0),
            stable=data.get("stable", False),
        )
        
        # Reconstruct plan if present
        if data.get("plan"):
            plan_data = data["plan"]
            intent = IntentFamily(plan_data["intent"]) if plan_data.get("intent") else IntentFamily.EXISTENCE_EVIDENCE
            bundle.plan = AgenticPlan.from_dict(plan_data, intent)
            bundle.constraints = bundle.plan.constraints
        
        return bundle


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_claim(claim: Claim) -> List[str]:
    """
    Validate a claim for completeness.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    if not claim.subject:
        errors.append("Claim missing subject")
    
    if not claim.object:
        errors.append("Claim missing object")
    
    if not claim.evidence:
        errors.append("Claim has no evidence refs")
    
    for i, ref in enumerate(claim.evidence):
        if not ref.quote_span:
            errors.append(f"Evidence ref {i} has no quote_span")
    
    if claim.support_strength < 1 or claim.support_strength > 3:
        errors.append(f"Invalid support_strength: {claim.support_strength}")
    
    return errors


def validate_evidence_bundle(bundle: EvidenceBundle) -> List[str]:
    """
    Validate an evidence bundle for completeness.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    if not bundle.retrieval_runs:
        errors.append("Bundle has no retrieval runs")
    
    # Validate all claims
    for claim in bundle.claims:
        claim_errors = validate_claim(claim)
        for err in claim_errors:
            errors.append(f"Claim {claim.claim_id}: {err}")
    
    # Check entities have evidence
    for entity in bundle.entities:
        if not entity.evidence_refs:
            errors.append(f"Entity {entity.key} has no evidence refs")
    
    return errors
