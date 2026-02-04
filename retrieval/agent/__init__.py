"""
V3/V4/V5 Agentic Framework - Tool-based search with evidence banding and verification.

This module provides a generalizable agentic architecture:
- Tool Registry: Typed wrappers around search primitives
- Evidence Builder: Span mining + rerank + cite/harvest banding
- V3: Claim Synthesis with citations and universal verifier
- V4: Interpretation Synthesis with reasoning model (4o) and two-stage verification
- V5: LLM-only grading with free tool choice (NEW)

V3 Usage:
    from retrieval.agent import V3Runner
    
    runner = V3Runner()
    result = runner.run("Who were Soviet agents in the OSS?", conn)
    
    for claim in result.claims.claims:
        print(f"{claim.text}")

V4 Usage:
    from retrieval.agent import V4Runner
    
    runner = V4Runner()
    result = runner.run("Who were members of the Silvermaster network?", conn)
    
    print(result.response.format_text())

V5 Usage (LLM-only grading):
    from retrieval.agent import V5Runner, run_v5_query
    
    result = run_v5_query(conn, "Who were members of the Silvermaster network?")
    print(result.format_answer_with_citations())
"""

import os

# Feature flag for V3 (default off until stable)
V3_ENABLED = os.getenv("V3_ENABLED", "0") == "1"

# Version info
V3_VERSION = "3.0.0"
V3_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_V3", "gpt-4o-mini")

# Default budgets
DEFAULT_BUDGETS = {
    "max_rounds": 2,
    "max_steps": 12,  # Increased to allow more tool chaining
    "max_chunks": 200,
    "max_cite_spans": 120,
    "max_harvest_spans": 240,
    "max_citations_per_claim": 2,
    "span_window_chars": 500,
}

# Import submodules for exports
from retrieval.agent.tools import (
    ToolResult,
    ToolSpec,
    TOOL_REGISTRY,
    get_tool,
    list_tools,
    get_tools_for_prompt,
    # Search tools
    hybrid_search_tool,
    vector_search_tool,
    lexical_search_tool,
    lexical_exact_tool,
    # Entity tools
    entity_lookup_tool,
    entity_surfaces_tool,
    entity_mentions_tool,
    co_mention_entities_tool,
    first_mention_tool,
    # Concordance tools
    expand_aliases_tool,
)

from retrieval.agent.executor import (
    ToolStep,
    ExecutionTrace,
    ExecutionResult,
    ToolExecutor,
    merge_tool_results,
)

from retrieval.agent.v3_evidence import (
    EvidenceSpan,
    EvidenceStats,
    EvidenceSet,
    EvidenceBuilder,
)

from retrieval.agent.v3_plan import (
    PlanConstraints,
    PlanBudgets,
    AgentPlanV3,
    generate_plan,
    revise_plan,
)

from retrieval.agent.v3_claims import (
    EvidenceRef,
    ClaimV3,
    ClaimBundleV3,
    synthesize_claims,
)

from retrieval.agent.entity_surfaces import (
    EntitySurfaceIndex,
    normalize_surface,
)

from retrieval.agent.v3_verifier import (
    VerificationError,
    VerificationReport,
    VerifierV3,
)

from retrieval.agent.entity_surfaces import (
    EntitySurfaceIndex,
    normalize_surface,
)

from retrieval.agent.v3_runner import (
    V3RunTrace,
    V3Result,
    V3Runner,
    run_v3_query,
)

from retrieval.agent.v3_summarizer import (
    SummaryBullet,
    SummarySection,
    V3SummaryResult,
    RESPONSE_SHAPES,
    doc_balanced_sample,
    summarize_from_evidence_set,
    summarize_v3_result,
    detect_response_shape,
)

# V4 Imports
from retrieval.agent.v4_interpret import (
    V4_VERSION,
    V4_MODEL_DEFAULT,
    V4_BUDGETS,
    SpanCitation,
    AnswerUnit,
    DiagnosticsInfo,
    InterpretationV4,
    PreparedSpan,
    interpret_evidence,
    prepare_spans_for_interpretation,
    detect_response_shape as v4_detect_response_shape,
)

from retrieval.agent.v4_verify import (
    VerificationError as V4VerificationError,
    VerificationWarning,
    UnitVerificationStatus,
    V4VerificationReport,
    V4Verifier,
    verify_interpretation,
)

from retrieval.agent.v4_render import (
    RenderedCitation,
    RenderedUnit,
    RenderedSection,
    RenderStats,
    V4RenderedResponse,
    V4Renderer,
    render_interpretation,
)

from retrieval.agent.v4_runner import (
    V4RunTrace,
    V4Result,
    V4Runner,
    run_v4_query,
    execute_v4_query,
    format_v4_result,
)

# V4.2 Discovery Loop
from retrieval.agent.v4_discover import (
    DiscoveryState,
    DiscoveryTrace,
    run_discovery,
)

from retrieval.agent.v4_discovery_metrics import (
    DiscoveryBudgets,
    CoverageMetrics,
    StopDecision,
    DEFAULT_BUDGETS as DISCOVERY_DEFAULT_BUDGETS,
    THOROUGH_BUDGETS as DISCOVERY_THOROUGH_BUDGETS,
    compute_coverage_metrics,
    evaluate_stop_conditions,
    should_run_discovery,
)

from retrieval.agent.v4_discovery_prompt import (
    DiscoveryAction,
    DiscoveryPlan,
    DiscoveryObservation,
    DiscoveryTool,
)

# V5 LLM-only grading
from retrieval.agent.v5_types import (
    CandidateSpan,
    GraderResult,
    EvidenceItem as V5EvidenceItem,
    EvidenceStore,
    EvidenceStatus,
    V5Budgets,
    V5Trace,
    StepLog,
    ToolCallAction,
    StopAnswerAction,
    StopInsufficientAction,
)

from retrieval.agent.v5_grader import (
    Grader,
    PairwiseGrader,
)

from retrieval.agent.v5_searcher import (
    Searcher,
    build_observation,
)

from retrieval.agent.v5_controller import (
    Controller as V5Controller,
    extract_candidates,
)

from retrieval.agent.v5_runner import (
    V5Result,
    V5Runner,
    run_v5_query,
    V5RunnerV2,
    run_v5_2_query,
)

from retrieval.agent.v5_rerank import (
    ExtractedSpan,
    SpanExtractor,
    SpanReranker,
    RerankResult,
    rerank_chunks,
)

from retrieval.agent.v5_hypothesis import (
    WorkingHypothesis,
    HypothesisStatus,
    HypothesisSet,
    HypothesisGenerator,
)

from retrieval.agent.v5_controller_v2 import (
    ControllerV2 as V5ControllerV2,
    V5BudgetsV2,
)

# V6 Principled (no heuristics)
from retrieval.agent.v6_query_parser import (
    QueryParser,
    ParsedQuery,
    TaskType,
)

from retrieval.agent.v6_entity_linker import (
    EntityLinker as V6EntityLinker,
    LinkedEntity,
    EntityLinkingResult,
)

from retrieval.agent.v6_evidence_bottleneck import (
    EvidenceBottleneck,
    BottleneckSpan,
    BottleneckResult,
    apply_bottleneck,
)

from retrieval.agent.v6_responsiveness import (
    ResponsivenessVerifier,
    ResponsivenessResult,
    ResponsivenessStatus,
    verify_responsiveness,
)

from retrieval.agent.v6_progress_gate import (
    ProgressGate,
    ProgressResult,
    ProgressStatus,
    RoundDecision,
)

from retrieval.agent.v6_controller import (
    V6Controller,
    V6Config,
    V6Trace,
)

from retrieval.agent.v6_runner import (
    V6Result,
    V6Runner,
    run_v6_query,
)

__all__ = [
    # Config
    "V3_ENABLED",
    "V3_VERSION",
    "V3_MODEL_DEFAULT",
    "DEFAULT_BUDGETS",
    # Tools
    "ToolResult",
    "ToolSpec",
    "TOOL_REGISTRY",
    "get_tool",
    "list_tools",
    "get_tools_for_prompt",
    # Search tools
    "hybrid_search_tool",
    "vector_search_tool",
    "lexical_search_tool",
    "lexical_exact_tool",
    # Entity tools
    "entity_lookup_tool",
    "entity_surfaces_tool",
    "entity_mentions_tool",
    "co_mention_entities_tool",
    "first_mention_tool",
    # Concordance tools
    "expand_aliases_tool",
    # Executor
    "ToolStep",
    "ExecutionTrace",
    "ExecutionResult",
    "ToolExecutor",
    "merge_tool_results",
    # Evidence
    "EvidenceSpan",
    "EvidenceStats",
    "EvidenceSet",
    "EvidenceBuilder",
    # Plan
    "PlanConstraints",
    "PlanBudgets",
    "AgentPlanV3",
    "generate_plan",
    "revise_plan",
    # Claims
    "EvidenceRef",
    "ClaimV3",
    "ClaimBundleV3",
    "synthesize_claims",
    # Entity Surfaces
    "EntitySurfaceIndex",
    "normalize_surface",
    # Verifier
    "VerificationError",
    "VerificationReport",
    "VerifierV3",
    # Entity Surfaces
    "EntitySurfaceIndex",
    "normalize_surface",
    # Runner
    "V3RunTrace",
    "V3Result",
    "V3Runner",
    "run_v3_query",
    # Summarizer
    "SummaryBullet",
    "SummarySection",
    "V3SummaryResult",
    "RESPONSE_SHAPES",
    "doc_balanced_sample",
    "summarize_from_evidence_set",
    "summarize_v3_result",
    "detect_response_shape",
    # V4 Config
    "V4_VERSION",
    "V4_MODEL_DEFAULT",
    "V4_BUDGETS",
    # V4 Interpretation
    "SpanCitation",
    "AnswerUnit",
    "DiagnosticsInfo",
    "InterpretationV4",
    "PreparedSpan",
    "interpret_evidence",
    "prepare_spans_for_interpretation",
    "v4_detect_response_shape",
    # V4 Verification
    "V4VerificationError",
    "VerificationWarning",
    "UnitVerificationStatus",
    "V4VerificationReport",
    "V4Verifier",
    "verify_interpretation",
    # V4 Rendering
    "RenderedCitation",
    "RenderedUnit",
    "RenderedSection",
    "RenderStats",
    "V4RenderedResponse",
    "V4Renderer",
    "render_interpretation",
    # V4 Runner
    "V4RunTrace",
    "V4Result",
    "V4Runner",
    "run_v4_query",
    "execute_v4_query",
    "format_v4_result",
    # V4.2 Discovery Loop
    "DiscoveryState",
    "DiscoveryTrace",
    "run_discovery",
    "DiscoveryBudgets",
    "CoverageMetrics", 
    "StopDecision",
    "DISCOVERY_DEFAULT_BUDGETS",
    "DISCOVERY_THOROUGH_BUDGETS",
    "compute_coverage_metrics",
    "evaluate_stop_conditions",
    "should_run_discovery",
    "DiscoveryAction",
    "DiscoveryPlan",
    "DiscoveryObservation",
    "DiscoveryTool",
    # V5 LLM-only grading
    "CandidateSpan",
    "GraderResult",
    "V5EvidenceItem",
    "EvidenceStore",
    "EvidenceStatus",
    "V5Budgets",
    "V5Trace",
    "StepLog",
    "ToolCallAction",
    "StopAnswerAction",
    "StopInsufficientAction",
    "Grader",
    "PairwiseGrader",
    "Searcher",
    "build_observation",
    "V5Controller",
    "extract_candidates",
    "V5Result",
    "V5Runner",
    "run_v5_query",
    # V5.2 Enhanced (Rerank + Hypotheses)
    "V5RunnerV2",
    "run_v5_2_query",
    "ExtractedSpan",
    "SpanExtractor",
    "SpanReranker",
    "RerankResult",
    "rerank_chunks",
    "WorkingHypothesis",
    "HypothesisStatus",
    "HypothesisSet",
    "HypothesisGenerator",
    "V5ControllerV2",
    "V5BudgetsV2",
    # V6 Principled (no heuristics)
    "QueryParser",
    "ParsedQuery",
    "TaskType",
    "V6EntityLinker",
    "LinkedEntity",
    "EntityLinkingResult",
    "EvidenceBottleneck",
    "BottleneckSpan",
    "BottleneckResult",
    "apply_bottleneck",
    "ResponsivenessVerifier",
    "ResponsivenessResult",
    "ResponsivenessStatus",
    "verify_responsiveness",
    "ProgressGate",
    "ProgressResult",
    "ProgressStatus",
    "RoundDecision",
    "V6Controller",
    "V6Config",
    "V6Trace",
    "V6Result",
    "V6Runner",
    "run_v6_query",
]
