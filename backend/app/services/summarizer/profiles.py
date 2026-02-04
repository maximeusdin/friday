"""
Summarization Profiles

Defines pre-configured profiles for different summarization use cases:
- conversational_answer: Quick synthesis for research queries
- audit_digest: Comprehensive chronological review
- entity_brief: Entity-focused summary
- document_summary: Single document deep dive
"""

from dataclasses import dataclass, field
from typing import Literal, Dict, Any

from .models import SelectionLambdas


@dataclass
class SummarizationProfile:
    """Configuration for a summarization profile."""
    
    # Profile identifier
    name: str
    description: str
    
    # Core limits
    max_chunks: int = 40
    max_claims: int = 10
    candidate_pool_limit: int = 5000
    
    # Doc focus (adaptive)
    doc_focus_mode: Literal["auto", "global", "single_doc"] = "auto"
    base_max_per_doc: int = 3  # Before adaptive adjustment
    
    # Diversity settings
    ensure_time_coverage: bool = True
    ensure_doc_diversity: bool = True
    ensure_page_coverage: bool = True
    page_bucket_size: int = 5  # Pages per bucket for intra-doc diversity
    
    # Selection utility threshold
    utility_threshold: float = -float('inf')  # Accept all by default
    
    # Penalty weights
    lambdas: SelectionLambdas = field(default_factory=SelectionLambdas)
    
    # Output options
    output_themes: bool = True
    output_timeline: bool = False
    chronological_bias: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "max_chunks": self.max_chunks,
            "max_claims": self.max_claims,
            "candidate_pool_limit": self.candidate_pool_limit,
            "doc_focus_mode": self.doc_focus_mode,
            "base_max_per_doc": self.base_max_per_doc,
            "ensure_time_coverage": self.ensure_time_coverage,
            "ensure_doc_diversity": self.ensure_doc_diversity,
            "ensure_page_coverage": self.ensure_page_coverage,
            "page_bucket_size": self.page_bucket_size,
            "utility_threshold": self.utility_threshold,
            "lambdas": {
                "doc": self.lambdas.doc,
                "entity": self.lambdas.entity,
                "time": self.lambdas.time,
                "page": self.lambdas.page,
            },
            "output_themes": self.output_themes,
            "output_timeline": self.output_timeline,
            "chronological_bias": self.chronological_bias,
        }


# =============================================================================
# Pre-defined Profiles
# =============================================================================

PROFILES: Dict[str, SummarizationProfile] = {
    "conversational_answer": SummarizationProfile(
        name="conversational_answer",
        description="Quick synthesis for research queries with balanced coverage",
        max_chunks=40,
        max_claims=10,
        candidate_pool_limit=5000,
        doc_focus_mode="auto",
        base_max_per_doc=3,
        ensure_time_coverage=True,
        ensure_doc_diversity=True,
        ensure_page_coverage=True,
        page_bucket_size=5,
        lambdas=SelectionLambdas(doc=0.3, entity=0.2, time=0.2, page=0.15),
        output_themes=True,
        output_timeline=False,
        chronological_bias=False,
    ),
    
    "audit_digest": SummarizationProfile(
        name="audit_digest",
        description="Comprehensive chronological review for thorough analysis",
        max_chunks=100,
        max_claims=25,
        candidate_pool_limit=10000,
        doc_focus_mode="auto",
        base_max_per_doc=5,
        ensure_time_coverage=True,
        ensure_doc_diversity=True,
        ensure_page_coverage=True,
        page_bucket_size=5,
        lambdas=SelectionLambdas(doc=0.2, entity=0.15, time=0.3, page=0.1),
        output_themes=True,
        output_timeline=True,
        chronological_bias=True,
    ),
    
    "entity_brief": SummarizationProfile(
        name="entity_brief",
        description="Entity-focused summary with timeline and co-mentions",
        max_chunks=50,
        max_claims=15,
        candidate_pool_limit=5000,
        doc_focus_mode="auto",
        base_max_per_doc=4,
        ensure_time_coverage=True,
        ensure_doc_diversity=True,
        ensure_page_coverage=True,
        page_bucket_size=5,
        lambdas=SelectionLambdas(doc=0.3, entity=0.1, time=0.25, page=0.15),  # Lower entity penalty
        output_themes=True,
        output_timeline=True,
        chronological_bias=False,
    ),
    
    "document_summary": SummarizationProfile(
        name="document_summary",
        description="Deep dive into a single document with page coverage",
        max_chunks=60,
        max_claims=20,
        candidate_pool_limit=5000,
        doc_focus_mode="single_doc",
        base_max_per_doc=60,  # No practical doc limit
        ensure_time_coverage=False,
        ensure_doc_diversity=False,
        ensure_page_coverage=True,
        page_bucket_size=3,  # Finer page buckets for single doc
        lambdas=SelectionLambdas(doc=0.0, entity=0.2, time=0.1, page=0.3),  # Heavy page diversity
        output_themes=True,
        output_timeline=False,
        chronological_bias=False,
    ),
    
    "quick_answer": SummarizationProfile(
        name="quick_answer",
        description="Fast, focused answer with minimal evidence",
        max_chunks=20,
        max_claims=5,
        candidate_pool_limit=2000,
        doc_focus_mode="auto",
        base_max_per_doc=3,
        ensure_time_coverage=False,
        ensure_doc_diversity=True,
        ensure_page_coverage=False,
        page_bucket_size=10,
        lambdas=SelectionLambdas(doc=0.4, entity=0.2, time=0.1, page=0.1),
        output_themes=False,
        output_timeline=False,
        chronological_bias=False,
    ),
}


def get_profile(profile_name: str) -> SummarizationProfile:
    """
    Get a summarization profile by name.
    
    Args:
        profile_name: Name of the profile
        
    Returns:
        The profile configuration
        
    Raises:
        ValueError: If profile doesn't exist
    """
    if profile_name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")
    return PROFILES[profile_name]


def list_profiles() -> Dict[str, str]:
    """List all available profiles with descriptions."""
    return {name: p.description for name, p in PROFILES.items()}
