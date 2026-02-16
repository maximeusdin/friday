"""
Pydantic models for the LLM Summarizer system.

This module defines all data structures used throughout the summarization pipeline:
- Selection inputs and outputs (Stage A)
- Evidence bundles with quote anchors
- LLM synthesis outputs (Stage B)
- Final summary artifacts for API responses
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel


# =============================================================================
# Selection Stage Models (Stage A)
# =============================================================================

@dataclass
class SelectionLambdas:
    """Penalty weights for diversity-based selection."""
    doc: float = 0.3       # Penalty for same-doc chunks after soft cap
    entity: float = 0.2    # Penalty for entity overlap with already-selected
    time: float = 0.2      # Penalty for same time bucket (year)
    page: float = 0.15     # Penalty for same page bucket (intra-doc)


@dataclass
class ChunkCandidate:
    """A chunk being considered for selection."""
    chunk_id: int
    rank: int
    doc_id: int
    doc_title: str
    page: Optional[int]
    year: Optional[int]
    score_hybrid: float
    score_lexical: Optional[float]
    score_vector: Optional[float]
    matched_entity_ids: List[int] = field(default_factory=list)
    matched_phrases: List[str] = field(default_factory=list)
    in_lexical: bool = False
    in_vector: bool = False


@dataclass
class SelectionInputs:
    """
    Debugging context for selection - answers "why did it choose these chunks?"
    Persisted in selection_inputs JSONB column.
    """
    # Pool context
    candidate_pool_size: int     # How many we considered for selection
    total_available: int         # Total hits in result set after retrieval
    pool_seed: Optional[int]     # Seed used if any randomness in pool building
    
    # Summary type (future-proofs for page-window vs sample expansion)
    summary_type: str            # "sample" | "page_window"
    page_window: Optional[Dict[str, int]]  # {offset: 0, limit: 50} if page_window mode
    
    # Selection parameters
    lambdas: Dict[str, float]    # Serialized SelectionLambdas
    utility_threshold: float
    
    # Adaptive mode
    doc_scope_size: int
    doc_focus_mode: str          # "auto" resolved to actual ("global", "single_doc", etc.)
    effective_max_per_doc: int   # After adaptive adjustment
    
    # Overrides applied
    overrides: Dict[str, Any]    # {max_per_doc: 40, reason: "single_doc scope"}
    
    # Facet snapshot for debugging
    facet_snapshot: Dict[str, Any]  # {year_buckets: [...], top_docs: [...]}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSONB storage."""
        return {
            "candidate_pool_size": self.candidate_pool_size,
            "total_available": self.total_available,
            "pool_seed": self.pool_seed,
            "summary_type": self.summary_type,
            "page_window": self.page_window,
            "lambdas": self.lambdas,
            "utility_threshold": self.utility_threshold,
            "doc_scope_size": self.doc_scope_size,
            "doc_focus_mode": self.doc_focus_mode,
            "effective_max_per_doc": self.effective_max_per_doc,
            "overrides": self.overrides,
            "facet_snapshot": self.facet_snapshot,
        }


@dataclass
class SelectionSpec:
    """
    What was selected - enables exact reproduction.
    Persisted in selection_spec JSONB column.
    """
    chunk_ids: List[int]           # Ordered list of selected chunk IDs
    policy: str                    # "greedy_soft_diversity"
    bundle_id_map: Dict[str, int]  # {"B1": 123, "B2": 456, ...}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSONB storage."""
        return {
            "chunk_ids": self.chunk_ids,
            "policy": self.policy,
            "bundle_id_map": self.bundle_id_map,
        }


@dataclass
class SelectionResult:
    """Full result of Stage A selection."""
    spec: SelectionSpec
    inputs: SelectionInputs
    candidates: List[ChunkCandidate]  # The selected candidates in order


# =============================================================================
# Evidence Bundle Models
# =============================================================================

@dataclass
class MatchTraceInfo:
    """Match trace information for a chunk."""
    matched_entity_ids: List[int] = field(default_factory=list)
    matched_phrases: List[str] = field(default_factory=list)
    in_lexical: bool = False
    in_vector: bool = False
    score_lexical: Optional[float] = None
    score_vector: Optional[float] = None
    score_hybrid: Optional[float] = None
    highlight_spans: Optional[List[Dict[str, int]]] = None  # [{start, end}, ...]


@dataclass
class LineContext:
    """Line context for a chunk (3-line window)."""
    line_before: Optional[str] = None
    highlight_line: Optional[str] = None
    line_after: Optional[str] = None


@dataclass
class QuoteAnchor:
    """
    Tight anchor for drill-down UX.
    Server-generated, not from LLM.
    """
    start_char: int
    end_char: int
    quote_excerpt: str  # <=120 chars
    anchor_method: str  # "highlight" | "line_context" | "snippet_fallback"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote_excerpt": self.quote_excerpt,
            "anchor_method": self.anchor_method,
        }


@dataclass
class EvidenceBundle:
    """
    A bundle of evidence for LLM synthesis.
    LLM cites bundle_id (B1, B2...), server maps back to chunk_id.
    """
    bundle_id: str      # "B1", "B2", ... (LLM cites these)
    chunk_id: int       # Mapped after synthesis
    doc_id: int
    doc_title: str
    page: Optional[int]
    snippet: str        # 300-800 chars
    quote_anchor: QuoteAnchor
    line_context: Optional[LineContext]
    match_trace: MatchTraceInfo
    date_key: Optional[str]
    place_ids: List[int] = field(default_factory=list)
    
    def to_prompt_format(self) -> str:
        """Format bundle for inclusion in LLM prompt."""
        match_info = []
        if self.match_trace.matched_entity_ids:
            match_info.append(f"entities matched: {len(self.match_trace.matched_entity_ids)}")
        if self.match_trace.matched_phrases:
            match_info.append(f"phrases: {', '.join(self.match_trace.matched_phrases[:3])}")
        if self.match_trace.in_lexical and self.match_trace.in_vector:
            match_info.append("hybrid match")
        elif self.match_trace.in_lexical:
            match_info.append("lexical match")
        elif self.match_trace.in_vector:
            match_info.append("semantic match")
        
        why_surfaced = "; ".join(match_info) if match_info else "relevance score"
        
        return f"""[{self.bundle_id}] Document: {self.doc_title}, Page {self.page or 'N/A'}
Date: {self.date_key or 'Unknown'}
Why surfaced: {why_surfaced}
---
{self.snippet}
---"""


# =============================================================================
# LLM Synthesis Output Models (Stage B)
# =============================================================================

class ClaimOutput(BaseModel):
    """Raw claim output from LLM (uses bundle IDs)."""
    claim_id: str
    claim: str
    citations: List[str]  # Bundle IDs: ["B1", "B3"]
    confidence: Literal["high", "medium", "low"]
    limitations: Optional[str] = None


class ThemeOutput(BaseModel):
    """Theme identified by LLM."""
    theme: str
    description: Optional[str] = None
    evidence: List[str] = []  # Bundle IDs


class SynthesisOutput(BaseModel):
    """Raw output from LLM synthesis (before validation)."""
    claims: List[ClaimOutput]
    themes: List[ThemeOutput] = []
    entities_mentioned: List[str] = []  # For advisory validation
    coverage_notes: Optional[str] = None  # Model's optional notes (NOT the coverage line)
    followups: List[str] = []


# =============================================================================
# Validated/Final Output Models
# =============================================================================

class CitationWithAnchor(BaseModel):
    """A citation with its quote anchor for UI rendering."""
    chunk_id: int
    bundle_id: str
    quote_anchor: Dict[str, Any]  # QuoteAnchor as dict
    doc_title: str
    page: Optional[int] = None


class ClaimWithSupport(BaseModel):
    """A validated claim with mapped chunk_id citations."""
    claim_id: str
    claim: str
    support: List[CitationWithAnchor]
    confidence: str
    confidence_reason: Optional[str] = None  # If downgraded, why
    limitations: Optional[str] = None


class ThemeWithEvidence(BaseModel):
    """A theme with mapped chunk_id evidence."""
    theme: str
    description: Optional[str] = None
    evidence: List[CitationWithAnchor] = []


class EntityInfo(BaseModel):
    """Entity information extracted from citations."""
    entity_id: int
    name: str
    entity_type: Optional[str] = None
    mention_count: int = 1


class DateCount(BaseModel):
    """Date facet in summary."""
    year: int
    count: int


class NextAction(BaseModel):
    """Suggested next action for user."""
    label: str
    action_type: str  # "expand", "set_mode", "drill_down", "filter"
    params: Dict[str, Any] = {}


class ModelInfo(BaseModel):
    """Information about the model used for synthesis."""
    name: str
    snapshot: Optional[str] = None  # e.g., "gpt-4.1-mini-2025-04-14"
    provider: str = "openai"
    temperature: float = 0.3
    max_tokens: int = 2000
    prompt_version: str = "v1"


class CoverageInfo(BaseModel):
    """Coverage information for the summary (server-generated)."""
    # What was summarized
    chunks_summarized: int
    
    # Pool vs total distinction
    candidate_pool_size: int    # How many chunks we CONSIDERED (scored)
    total_hits: int             # Total in result set AFTER retrieval filters
    
    # Selection metadata
    selection_policy: str       # "greedy_soft_diversity"
    doc_focus_mode: str         # "global" | "single_doc" | "small_corpus"
    summary_type: str           # "sample" | "page_window"
    selection_notes: str        # Human-readable notes
    
    # What wasn't checked
    not_checked: str            # "Remaining 12,812 chunks not reviewed"
    
    # Retrieval context (defensibility)
    retrieval_mode: str         # "conversational" | "thorough"
    cap_applied: bool
    threshold_used: Optional[float] = None
    
    def to_coverage_line(self) -> str:
        """Generate human-readable coverage line."""
        return (
            f"Selected {self.chunks_summarized} from a candidate pool of "
            f"{self.candidate_pool_size:,} (of {self.total_hits:,} total hits). "
            f"{self.not_checked}"
        )


class SummaryArtifact(BaseModel):
    """
    The complete summary artifact returned by the API.
    UI-renderable structured output.
    """
    summary_id: str
    result_set_id: int
    retrieval_run_id: int
    question: Optional[str] = None
    
    # Coverage (server-generated)
    coverage: CoverageInfo
    
    # Claims with citations
    answer: List[ClaimWithSupport]
    
    # Themes
    themes: List[ThemeWithEvidence] = []
    
    # Entities (inferred from citations, more reliable than LLM list)
    entities_verified: List[EntityInfo] = []
    entities_flagged: List[str] = []  # "not verified in sample"
    
    # Date facets
    dates: List[DateCount] = []
    
    # Next actions
    next_actions: List[NextAction] = []
    
    # Model info
    model: ModelInfo
    
    # Cache indicator
    cached: bool = False
    
    def with_cached_flag(self, cached: bool) -> "SummaryArtifact":
        """Return a copy with the cached flag set."""
        return SummaryArtifact(
            **{**self.model_dump(), "cached": cached}
        )


# =============================================================================
# Validation Result Models
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating LLM synthesis output."""
    valid_claims: List[ClaimWithSupport]
    rejected_claims: List[ClaimOutput]  # Claims with invalid/missing citations
    themes: List[ThemeWithEvidence]
    entities_verified: List[EntityInfo]
    entities_flagged: List[str]
    validation_notes: List[str]  # Notes about validation actions taken


@dataclass
class EntityValidationResult:
    """Result of entity validation (advisory)."""
    verified_entities: List[str]
    flagged_entities: List[str]  # "not verified in reviewed sample"
    inferred_entity_ids: List[int]  # More reliable - from citations
    validation_notes: str
