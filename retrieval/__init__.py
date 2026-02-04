from .ops import (
    get_conn,
    SearchFilters,
    ChunkHit,
    lex_exact,
    lex_and,
    lex_near,
    vector_search,
    hybrid_rrf,
)

# V2 Agentic Architecture exports
from .query_intent import (
    QueryContract,
    FocusBundleMode,
    TargetSpec,
    ConstraintSpec,
    build_keyword_intent_contract,
    build_relationship_contract,
    build_affiliation_contract,
)

from .focus_bundle import (
    FocusBundle,
    FocusSpan,
    FocusBundleBuilder,
    persist_focus_bundle,
    load_focus_bundle,
)

from .spans import (
    Span,
    SpanMiner,
    get_page_ref,
)

from .constraints import (
    ConstraintSupport,
    CandidateAssessment,
    AffiliationConstraint,
    RelationshipConstraint,
    RoleConstraint,
    build_constraint_scorers,
    assess_candidate,
)

from .candidate_proposer import (
    ProposedCandidate,
    propose_person_candidates,
    propose_codename_candidates,
    propose_all_candidates,
)

from .hubness import (
    CandidateScore,
    score_candidates_with_hubness,
    load_entity_df,
)

from .expansion import (
    extract_expansion_entities,
    entity_expansion_loop,
    term_expansion_loop,
)

from .rendering import (
    RenderedAnswer,
    render_from_focus_bundle_with_constraints,
)

from .verifier_v2 import (
    FocusBundleVerifier,
    VerificationResult,
    Bullet,
)

from .agent_plan_v2 import (
    AgentPlanV2,
    build_keyword_intent_plan,
    build_relationship_plan,
    build_affiliation_plan,
)

from .executor_v2 import (
    ExecutorV2,
    ExecutionResult,
    execute_v2_plan,
    execute_relationship_query,
    execute_affiliation_query,
    execute_keyword_query,
)

# V3 Agentic Controller exports
from .query_analysis import (
    QueryAnalysis,
    analyze_query,
    QUERY_ANALYSIS_SCHEMA,
)

from .observations import (
    ObservationBundle,
    run_probe,
    build_observation_bundle,
    detect_red_flags,
)

from .plan_patch import (
    PlanPatch,
    Action,
    ActionOp,
    ExecutionState,
    get_plan_patch,
    get_fix_patch,
    execute_patch,
    PLAN_PATCH_SCHEMA,
)

from .variant_generator import (
    generate_variants,
    expand_anchor_terms,
    validate_variant,
)

from .controller import (
    AgenticController,
    ControllerResult,
    execute_agentic_query,
)

__all__ = [
    # Original ops exports
    "get_conn",
    "SearchFilters",
    "ChunkHit",
    "lex_exact",
    "lex_and",
    "lex_near",
    "vector_search",
    "hybrid_rrf",
    
    # V2 Query Intent
    "QueryContract",
    "FocusBundleMode",
    "TargetSpec",
    "ConstraintSpec",
    "build_keyword_intent_contract",
    "build_relationship_contract",
    "build_affiliation_contract",
    
    # V2 FocusBundle
    "FocusBundle",
    "FocusSpan",
    "FocusBundleBuilder",
    "persist_focus_bundle",
    "load_focus_bundle",
    
    # V2 Spans
    "Span",
    "SpanMiner",
    "get_page_ref",
    
    # V2 Constraints
    "ConstraintSupport",
    "CandidateAssessment",
    "AffiliationConstraint",
    "RelationshipConstraint",
    "RoleConstraint",
    "build_constraint_scorers",
    "assess_candidate",
    
    # V2 Candidates
    "ProposedCandidate",
    "propose_person_candidates",
    "propose_codename_candidates",
    "propose_all_candidates",
    
    # V2 Hubness
    "CandidateScore",
    "score_candidates_with_hubness",
    "load_entity_df",
    
    # V2 Expansion
    "extract_expansion_entities",
    "entity_expansion_loop",
    "term_expansion_loop",
    
    # V2 Rendering
    "RenderedAnswer",
    "render_from_focus_bundle_with_constraints",
    
    # V2 Verification
    "FocusBundleVerifier",
    "VerificationResult",
    "Bullet",
    
    # V2 Plan
    "AgentPlanV2",
    "build_keyword_intent_plan",
    "build_relationship_plan",
    "build_affiliation_plan",
    
    # V2 Execution
    "ExecutorV2",
    "ExecutionResult",
    "execute_v2_plan",
    "execute_relationship_query",
    "execute_affiliation_query",
    "execute_keyword_query",
    
    # V3 Query Analysis
    "QueryAnalysis",
    "analyze_query",
    "QUERY_ANALYSIS_SCHEMA",
    
    # V3 Observations
    "ObservationBundle",
    "run_probe",
    "build_observation_bundle",
    "detect_red_flags",
    
    # V3 Plan Patch
    "PlanPatch",
    "Action",
    "ActionOp",
    "ExecutionState",
    "get_plan_patch",
    "get_fix_patch",
    "execute_patch",
    "PLAN_PATCH_SCHEMA",
    
    # V3 Variant Generator
    "generate_variants",
    "expand_anchor_terms",
    "validate_variant",
    
    # V3 Controller
    "AgenticController",
    "ControllerResult",
    "execute_agentic_query",
]
