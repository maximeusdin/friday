"""
Typed Plan Schema for Agentic Workflow.

The planner outputs a typed, executable AgenticPlan object - not ad-hoc JSON.
This keeps orchestration clean and testable.

Architecture: Plan → Execute → Verify → Render
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.intent import IntentFamily


class SupportType(str, Enum):
    """How strong is the textual support for a claim."""
    EXPLICIT_STATEMENT = "explicit_statement"  # direct assertion
    DEFINITION = "definition"                   # "X is a..."
    ASSESSMENT = "assessment"                   # evaluation/opinion
    CO_MENTION = "co_mention"                   # appears together


@dataclass
class LaneSpec:
    """
    Specification for a single retrieval lane.
    
    Lanes are executed in parallel during each round, with results merged
    and deduplicated. Each lane has its own search strategy and parameters.
    """
    lane_id: str                              # "entity_codename", "lexical_must_hit", etc.
    query_terms: List[str] = field(default_factory=list)  # search terms
    entity_ids: List[int] = field(default_factory=list)   # entity targets
    filters: Dict[str, Any] = field(default_factory=dict) # collection, date range, etc.
    must_hit_terms: List[str] = field(default_factory=list)  # required lexical matches
    k: int = 100                              # max results for this lane
    priority: int = 1                         # execution priority (1=highest)
    
    # Lane-specific options
    include_aliases: bool = True              # expand entity aliases
    normalize_terms: bool = True              # apply deterministic normalizations
    
    def __post_init__(self):
        """Validate lane spec."""
        if not self.lane_id:
            raise ValueError("lane_id is required")


@dataclass
class CommentionLaneSpec(LaneSpec):
    """
    Specialized lane spec for co-mention expansion (roster queries).
    
    SQL-driven, very fast - finds all entities co-mentioned with seed entities.
    """
    lane_id: str = "comention_expand"
    seed_entity_ids: List[int] = field(default_factory=list)  # from prior rounds
    target_entity_type: str = "person"        # entity type to find
    min_comention_docs: int = 1               # minimum shared documents


@dataclass
class ExtractionSpec:
    """
    Specification for claim extraction from chunks.
    
    Controls which predicates to extract, whether LLM refinement is allowed,
    and what support types are required.
    """
    predicates: List[str] = field(default_factory=list)  # which predicates to extract
    require_quote_spans: bool = True          # always True - spans required for verification
    support_type_required: Optional[SupportType] = None  # e.g., EXPLICIT_STATEMENT
    allow_llm_refinement: bool = False        # LLM for ambiguous spans/support_type only
    min_support_strength: int = 1             # minimum support strength (1-3)
    
    # Pattern extraction options
    extract_codename_mappings: bool = True
    extract_role_evidence: bool = True
    extract_membership: bool = True


@dataclass
class VerificationSpec:
    """
    Coverage thresholds and verification requirements (per intent).
    
    These thresholds determine when the verifier considers coverage "sufficient"
    for making claims or negative statements.
    """
    required_lanes: List[str] = field(default_factory=list)  # lanes for coverage proof
    role_evidence_patterns: List[str] = field(default_factory=list)  # lexical patterns
    collection_scope: List[str] = field(default_factory=list)  # allowed collections
    
    # Coverage thresholds (vary by intent)
    min_doc_diversity: int = 3                # minimum unique docs for coverage proof
    min_hits_per_lane: int = 5                # minimum hits for lane to count as "ran"
    target_pages: int = 20                    # for EXISTENCE_EVIDENCE coverage calc
    min_relation_claims: int = 1              # for RELATIONSHIP_CONSTRAINED coverage
    
    # Verification strictness
    require_explicit_role_evidence: bool = False  # require role patterns in quote_span
    allow_inferential_claims: bool = True         # allow CO_MENTION support type


@dataclass
class Budgets:
    """
    Resource caps for the agentic workflow.
    
    These are hard limits - no coverage thresholds here (those go in VerificationSpec).
    """
    max_rounds: int = 3                       # maximum iteration rounds
    max_chunks_per_lane: int = 100            # cap per lane
    max_total_chunks: int = 200               # total chunks considered
    max_candidates_forward: int = 50          # candidates carried to next round
    max_expansion_terms: int = 10             # LLM-proposed expansion candidates
    max_validated_expansion_terms: int = 8    # after corpus validation + specificity cap
    
    # Time budgets (optional, 0 = unlimited)
    max_execution_seconds: int = 0
    max_llm_calls: int = 5


@dataclass
class StopConditions:
    """
    Conditions for stopping the iterative execution loop.
    
    Uses Jaccard + novelty to avoid premature stopping on extraction regressions.
    """
    # Jaccard + novelty (fixes premature stopping)
    jaccard_threshold: float = 0.85           # |A∩B| / |A∪B| must be >= this
    novelty_cap: int = 2                      # |B - A| must be <= this
    
    # Fallback conditions
    min_coverage_docs: int = 5                # minimum docs before stopping
    max_iterations: int = 3                   # hard limit on rounds
    
    # Early termination
    stop_on_zero_new_chunks: bool = True      # stop if no new chunks in round
    stop_on_stable_claims: bool = True        # stop if claims are stable


@dataclass
class BundleConstraints:
    """
    Constraints that apply to the entire evidence bundle.
    
    These are query-derived, not hardcoded - e.g., role patterns come from
    parsing "Soviet intelligence officers" in the query.
    """
    collection_scope: List[str] = field(default_factory=list)  # allowed collections
    required_role_evidence_patterns: List[str] = field(default_factory=list)  # query-derived
    support_type_required: Optional[SupportType] = None
    min_support_strength: int = 1
    
    # Date constraints (optional)
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    
    # Entity constraints
    required_entity_types: List[str] = field(default_factory=list)
    excluded_entity_ids: List[int] = field(default_factory=list)


@dataclass
class AgenticPlan:
    """
    The complete typed plan for agentic workflow execution.
    
    This is the contract between planner and executor - the executor
    simply executes this spec without ad-hoc logic.
    """
    intent: "IntentFamily"                    # classified intent
    constraints: BundleConstraints            # query-derived constraints
    lanes: List[LaneSpec]                     # lanes to execute
    extraction: ExtractionSpec                # how to extract claims
    verification: VerificationSpec            # verification requirements
    budgets: Budgets                          # resource caps
    stop_conditions: StopConditions           # when to stop iterating
    
    # Metadata
    query_text: str = ""                      # original query
    session_id: Optional[str] = None          # for tracking
    plan_version: str = "agentic_v1"          # schema version
    
    def validate(self) -> List[str]:
        """
        Validate the plan and return any errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not self.lanes:
            errors.append("Plan must have at least one lane")
        
        for i, lane in enumerate(self.lanes):
            if not lane.lane_id:
                errors.append(f"Lane {i} missing lane_id")
            if lane.k <= 0:
                errors.append(f"Lane {lane.lane_id} has invalid k={lane.k}")
        
        if self.budgets.max_rounds < 1:
            errors.append("max_rounds must be >= 1")
        
        if self.stop_conditions.jaccard_threshold < 0 or self.stop_conditions.jaccard_threshold > 1:
            errors.append("jaccard_threshold must be between 0 and 1")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "intent": self.intent.value if self.intent else None,
            "constraints": {
                "collection_scope": self.constraints.collection_scope,
                "required_role_evidence_patterns": self.constraints.required_role_evidence_patterns,
                "support_type_required": self.constraints.support_type_required.value if self.constraints.support_type_required else None,
                "min_support_strength": self.constraints.min_support_strength,
                "date_range_start": self.constraints.date_range_start,
                "date_range_end": self.constraints.date_range_end,
            },
            "lanes": [
                {
                    "lane_id": lane.lane_id,
                    "query_terms": lane.query_terms,
                    "entity_ids": lane.entity_ids,
                    "filters": lane.filters,
                    "must_hit_terms": lane.must_hit_terms,
                    "k": lane.k,
                    "priority": lane.priority,
                }
                for lane in self.lanes
            ],
            "extraction": {
                "predicates": self.extraction.predicates,
                "require_quote_spans": self.extraction.require_quote_spans,
                "support_type_required": self.extraction.support_type_required.value if self.extraction.support_type_required else None,
                "allow_llm_refinement": self.extraction.allow_llm_refinement,
            },
            "verification": {
                "required_lanes": self.verification.required_lanes,
                "role_evidence_patterns": self.verification.role_evidence_patterns,
                "collection_scope": self.verification.collection_scope,
                "min_doc_diversity": self.verification.min_doc_diversity,
                "min_hits_per_lane": self.verification.min_hits_per_lane,
                "target_pages": self.verification.target_pages,
            },
            "budgets": {
                "max_rounds": self.budgets.max_rounds,
                "max_chunks_per_lane": self.budgets.max_chunks_per_lane,
                "max_total_chunks": self.budgets.max_total_chunks,
                "max_candidates_forward": self.budgets.max_candidates_forward,
            },
            "stop_conditions": {
                "jaccard_threshold": self.stop_conditions.jaccard_threshold,
                "novelty_cap": self.stop_conditions.novelty_cap,
                "max_iterations": self.stop_conditions.max_iterations,
            },
            "query_text": self.query_text,
            "session_id": self.session_id,
            "plan_version": self.plan_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], intent_family: "IntentFamily") -> "AgenticPlan":
        """Create AgenticPlan from dictionary."""
        constraints_data = data.get("constraints", {})
        constraints = BundleConstraints(
            collection_scope=constraints_data.get("collection_scope", []),
            required_role_evidence_patterns=constraints_data.get("required_role_evidence_patterns", []),
            support_type_required=SupportType(constraints_data["support_type_required"]) if constraints_data.get("support_type_required") else None,
            min_support_strength=constraints_data.get("min_support_strength", 1),
            date_range_start=constraints_data.get("date_range_start"),
            date_range_end=constraints_data.get("date_range_end"),
        )
        
        lanes = [
            LaneSpec(
                lane_id=lane_data["lane_id"],
                query_terms=lane_data.get("query_terms", []),
                entity_ids=lane_data.get("entity_ids", []),
                filters=lane_data.get("filters", {}),
                must_hit_terms=lane_data.get("must_hit_terms", []),
                k=lane_data.get("k", 100),
                priority=lane_data.get("priority", 1),
            )
            for lane_data in data.get("lanes", [])
        ]
        
        extraction_data = data.get("extraction", {})
        extraction = ExtractionSpec(
            predicates=extraction_data.get("predicates", []),
            require_quote_spans=extraction_data.get("require_quote_spans", True),
            support_type_required=SupportType(extraction_data["support_type_required"]) if extraction_data.get("support_type_required") else None,
            allow_llm_refinement=extraction_data.get("allow_llm_refinement", False),
        )
        
        verification_data = data.get("verification", {})
        verification = VerificationSpec(
            required_lanes=verification_data.get("required_lanes", []),
            role_evidence_patterns=verification_data.get("role_evidence_patterns", []),
            collection_scope=verification_data.get("collection_scope", []),
            min_doc_diversity=verification_data.get("min_doc_diversity", 3),
            min_hits_per_lane=verification_data.get("min_hits_per_lane", 5),
            target_pages=verification_data.get("target_pages", 20),
        )
        
        budgets_data = data.get("budgets", {})
        budgets = Budgets(
            max_rounds=budgets_data.get("max_rounds", 3),
            max_chunks_per_lane=budgets_data.get("max_chunks_per_lane", 100),
            max_total_chunks=budgets_data.get("max_total_chunks", 200),
            max_candidates_forward=budgets_data.get("max_candidates_forward", 50),
        )
        
        stop_data = data.get("stop_conditions", {})
        stop_conditions = StopConditions(
            jaccard_threshold=stop_data.get("jaccard_threshold", 0.85),
            novelty_cap=stop_data.get("novelty_cap", 2),
            max_iterations=stop_data.get("max_iterations", 3),
        )
        
        return cls(
            intent=intent_family,
            constraints=constraints,
            lanes=lanes,
            extraction=extraction,
            verification=verification,
            budgets=budgets,
            stop_conditions=stop_conditions,
            query_text=data.get("query_text", ""),
            session_id=data.get("session_id"),
            plan_version=data.get("plan_version", "agentic_v1"),
        )


# =============================================================================
# Plan Builder Helpers
# =============================================================================

def build_default_lanes_for_intent(
    intent: "IntentFamily",
    entity_ids: List[int],
    query_terms: List[str],
    collection_scope: List[str],
) -> List[LaneSpec]:
    """
    Build default lane specs based on intent family.
    
    Args:
        intent: The classified intent family
        entity_ids: Resolved entity IDs from query
        query_terms: Key concepts/terms from query
        collection_scope: Allowed collections
        
    Returns:
        List of LaneSpec for the intent
    """
    from retrieval.intent import IntentFamily
    
    filters = {"collection_scope": collection_scope} if collection_scope else {}
    lanes = []
    
    # Lane 1: Entity/Codename (when we have entities)
    if entity_ids:
        lanes.append(LaneSpec(
            lane_id="entity_codename",
            entity_ids=entity_ids,
            filters=filters,
            k=100,
            priority=1,
            include_aliases=True,
        ))
    
    # Lane 2: Lexical Must-Hit (for concept queries)
    if query_terms:
        lanes.append(LaneSpec(
            lane_id="lexical_must_hit",
            query_terms=query_terms,
            must_hit_terms=query_terms,
            filters=filters,
            k=100,
            priority=1,
            normalize_terms=True,
        ))
    
    # Lane 3: Hybrid (always)
    lanes.append(LaneSpec(
        lane_id="hybrid",
        query_terms=query_terms,
        entity_ids=entity_ids,
        filters=filters,
        k=100,
        priority=2,
    ))
    
    # Lane 4: Ephemeral Expansion (for existence queries)
    if intent == IntentFamily.EXISTENCE_EVIDENCE and query_terms:
        lanes.append(LaneSpec(
            lane_id="ephemeral_expansion",
            query_terms=query_terms,
            filters=filters,
            k=50,
            priority=3,
        ))
    
    return lanes


def build_verification_spec_for_intent(
    intent: "IntentFamily",
    role_patterns: List[str],
    collection_scope: List[str],
) -> VerificationSpec:
    """
    Build verification spec based on intent family.
    
    Args:
        intent: The classified intent family
        role_patterns: Query-derived role evidence patterns
        collection_scope: Allowed collections
        
    Returns:
        VerificationSpec configured for the intent
    """
    from retrieval.intent import IntentFamily
    
    if intent == IntentFamily.EXISTENCE_EVIDENCE:
        # For existence queries, require hybrid (always runs)
        # lexical and ephemeral are optional based on query terms
        return VerificationSpec(
            required_lanes=["hybrid"],
            role_evidence_patterns=role_patterns,
            collection_scope=collection_scope,
            min_doc_diversity=3,
            target_pages=20,
        )
    
    elif intent == IntentFamily.RELATIONSHIP_CONSTRAINED:
        # For relationship queries, require hybrid
        # entity_codename only runs when we have entity_ids
        return VerificationSpec(
            required_lanes=["hybrid"],
            role_evidence_patterns=role_patterns,
            collection_scope=collection_scope,
            min_doc_diversity=2,
            require_explicit_role_evidence=True,
        )
    
    elif intent == IntentFamily.ROSTER_ENUMERATION:
        # For roster queries, hybrid is always required
        # entity_codename is optional (only when we have resolved entities)
        # lexical_must_hit is required if we have query terms
        return VerificationSpec(
            required_lanes=["hybrid"],  # Only require hybrid - most flexible
            role_evidence_patterns=role_patterns,
            collection_scope=collection_scope,
            min_doc_diversity=3,  # Lower threshold since we may not have entity-based search
        )
    
    # Default
    return VerificationSpec(
        required_lanes=["hybrid"],
        role_evidence_patterns=role_patterns,
        collection_scope=collection_scope,
    )
