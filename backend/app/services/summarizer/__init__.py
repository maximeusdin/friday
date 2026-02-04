"""
LLM Summarizer Service

A structured summarization system with:
- Deterministic evidence selection (Stage A)
- LLM synthesis with citation-backed claims (Stage B)
- Signature-based caching
- Full coverage transparency

Usage:
    from backend.app.services.summarizer import summarize_result_set
    
    artifact = summarize_result_set(
        conn=conn,
        result_set_id=123,
        question="What happened in 1944?",
        profile="conversational_answer"
    )
"""

from .models import (
    SummaryArtifact,
    CoverageInfo,
    ClaimWithSupport,
    CitationWithAnchor,
    ThemeWithEvidence,
    EntityInfo,
    ModelInfo,
    NextAction,
    EvidenceBundle,
    QuoteAnchor,
    SelectionSpec,
    SelectionInputs,
    SelectionResult,
    SelectionLambdas,
    ChunkCandidate,
    SynthesisOutput,
    ClaimOutput,
    ThemeOutput,
    ValidationResult,
    MatchTraceInfo,
    LineContext,
    DateCount,
)

from .profiles import get_profile, list_profiles, PROFILES, SummarizationProfile

__all__ = [
    # Main artifact
    "SummaryArtifact",
    "CoverageInfo",
    "ClaimWithSupport",
    "CitationWithAnchor",
    "ThemeWithEvidence",
    "EntityInfo",
    "ModelInfo",
    "NextAction",
    "DateCount",
    # Bundles
    "EvidenceBundle",
    "QuoteAnchor",
    "MatchTraceInfo",
    "LineContext",
    # Selection
    "SelectionSpec",
    "SelectionInputs",
    "SelectionResult",
    "SelectionLambdas",
    "ChunkCandidate",
    # Synthesis
    "SynthesisOutput",
    "ClaimOutput",
    "ThemeOutput",
    # Validation
    "ValidationResult",
    # Profiles
    "get_profile",
    "list_profiles",
    "PROFILES",
    "SummarizationProfile",
]
