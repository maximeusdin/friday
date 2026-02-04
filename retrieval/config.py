"""
Retrieval configuration for Two-Mode Retrieval System.

This module defines:
- Vector metric configuration (explicit semantics)
- Mode-specific configurations (conversational vs thorough)
- Mode detection and resolution logic

Mode Precedence (highest to lowest):
1. UI toggle - explicit user selection
2. Primitive override - SET_RETRIEVAL_MODE in plan
3. Trigger phrase detection - "exhaustive", "everything", etc.
4. Default - conversational
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
import re


# =============================================================================
# Vector Metric Configuration
# =============================================================================

@dataclass(frozen=True)
class VectorMetricConfig:
    """
    Explicit vector metric configuration to avoid formula breakage.
    
    For cosine distance (<=>):
    - similarity = 1 - distance
    - range: [-1, 1] (can be negative for opposite-direction embeddings)
    - threshold of 0.3 means distance <= 0.7
    
    For L2 distance (<->):
    - similarity = 1 / (1 + distance) or other transforms
    - range: [0, 1]
    
    For inner product (<#>):
    - similarity = -distance (pgvector returns negative IP)
    - range depends on embedding normalization
    """
    metric: str = "cosine"           # "cosine" | "l2" | "ip"
    operator: str = "<=>"            # "<=>" | "<->" | "<#>"
    similarity_transform: str = "1 - distance"  # formula for UI display
    similarity_range: Tuple[float, float] = (-1.0, 1.0)  # possible range
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "metric": self.metric,
            "operator": self.operator,
            "similarity_transform": self.similarity_transform,
            "similarity_range": list(self.similarity_range),
        }
    
    def validate_threshold(self, threshold: float) -> bool:
        """Check if threshold is within valid range for this metric."""
        return self.similarity_range[0] <= threshold <= self.similarity_range[1]
    
    def transform_to_distance(self, threshold: float) -> float:
        """
        Transform similarity threshold to distance threshold for SQL.
        
        For cosine: distance = 1 - similarity
        """
        if self.metric == "cosine":
            return 1.0 - threshold
        elif self.metric == "l2":
            # For L2, if similarity = 1/(1+d), then d = 1/s - 1
            if threshold > 0:
                return (1.0 / threshold) - 1.0
            return float('inf')
        elif self.metric == "ip":
            # For IP, distance = -similarity
            return -threshold
        return 1.0 - threshold  # default to cosine behavior


# Default configuration for cosine distance
DEFAULT_VECTOR_CONFIG = VectorMetricConfig()

# Alternative configs
L2_VECTOR_CONFIG = VectorMetricConfig(
    metric="l2",
    operator="<->",
    similarity_transform="1 / (1 + distance)",
    similarity_range=(0.0, 1.0),
)

IP_VECTOR_CONFIG = VectorMetricConfig(
    metric="ip",
    operator="<#>",
    similarity_transform="-distance",
    similarity_range=(-float('inf'), float('inf')),
)


# =============================================================================
# Mode-Specific Configurations
# =============================================================================

@dataclass(frozen=True)
class ConversationalModeConfig:
    """
    Configuration for conversational (fast + explainable) mode.
    
    Characteristics:
    - Returns top-k results quickly
    - Optimized for summarization and follow-up questions
    - Applies soft cap for UX protection
    - Score-based ranking with tie-breakers
    """
    # Result set size for "answer set" (renamed from top_k to avoid impl leak)
    answer_k: int = 20
    
    # Safety cap for UX (prevents overwhelming UI with results)
    max_hits_soft_cap: int = 2000
    
    # Default similarity threshold (cosine, range [-1, 1])
    similarity_threshold: float = 0.35
    
    # Enable detailed rank explanations
    enable_rank_trace: bool = True
    
    # Weights for hybrid ranking: (lexical, vector)
    rank_weights: Tuple[float, float] = (0.5, 0.5)
    
    # RRF parameter (60 is standard)
    rrf_k: int = 60
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "mode": "conversational",
            "answer_k": self.answer_k,
            "max_hits_soft_cap": self.max_hits_soft_cap,
            "similarity_threshold": self.similarity_threshold,
            "enable_rank_trace": self.enable_rank_trace,
            "rank_weights": list(self.rank_weights),
            "rrf_k": self.rrf_k,
        }


@dataclass(frozen=True)
class ThoroughModeConfig:
    """
    Configuration for thorough (exhaustive) mode.
    
    Characteristics:
    - Returns all results above threshold (no top-k cap)
    - Paginated delivery for large result sets
    - Deterministic ordering by (document_id, chunk_id)
    - Lower threshold for higher recall
    """
    # Lower threshold for recall (cosine, range [-1, 1])
    similarity_threshold: float = 0.25
    
    # No cap by default (None = unlimited)
    max_hits_hard_cap: Optional[int] = None
    
    # Pagination defaults
    pagination_default_limit: int = 100
    pagination_max_limit: int = 500
    
    # Rank trace is optional in thorough mode (ordering is deterministic)
    enable_rank_trace: bool = False
    
    # RRF parameter (only used if hybrid search still needed)
    rrf_k: int = 60
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "mode": "thorough",
            "similarity_threshold": self.similarity_threshold,
            "max_hits_hard_cap": self.max_hits_hard_cap,
            "pagination_default_limit": self.pagination_default_limit,
            "pagination_max_limit": self.pagination_max_limit,
            "enable_rank_trace": self.enable_rank_trace,
            "rrf_k": self.rrf_k,
        }


# =============================================================================
# Mode Detection
# =============================================================================

# Trigger phrases that indicate thorough/exhaustive intent
THOROUGH_MODE_TRIGGERS = frozenset([
    "thorough",
    "exhaustive", 
    "everything",
    "all",
    "complete",
    "don't miss",
    "dont miss",
    "comprehensive",
    "full search",
    "every mention",
    "all occurrences",
    "entire corpus",
    "nothing missed",
    "find all",
    "show all",
    "list all",
    "all references",
    "all mentions",
    "complete list",
    "exhaustive search",
])

# Compiled regex for faster matching
_THOROUGH_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in THOROUGH_MODE_TRIGGERS) + r')\b',
    re.IGNORECASE
)


def detect_retrieval_mode(utterance: str) -> Literal["conversational", "thorough"]:
    """
    Detect retrieval mode from utterance text.
    
    This is the LOWEST precedence in mode resolution.
    UI toggle and primitive override take precedence.
    
    Args:
        utterance: User's query text
        
    Returns:
        "thorough" if trigger phrase detected, else "conversational"
    """
    if not utterance:
        return "conversational"
    
    # Check for trigger phrases
    if _THOROUGH_PATTERN.search(utterance):
        return "thorough"
    
    return "conversational"


def resolve_retrieval_mode(
    ui_toggle: Optional[str],
    primitive_mode: Optional[str],
    utterance: str,
) -> Tuple[Literal["conversational", "thorough"], str]:
    """
    Resolve final retrieval mode with precedence logic.
    
    Precedence (highest to lowest):
    1. ui_toggle - explicit user selection in UI
    2. primitive_mode - SET_RETRIEVAL_MODE in plan
    3. trigger phrase - detected from utterance
    4. default - conversational
    
    Args:
        ui_toggle: Mode from UI toggle (if set)
        primitive_mode: Mode from SET_RETRIEVAL_MODE primitive (if present)
        utterance: User's query text for trigger phrase detection
        
    Returns:
        (mode, source) tuple where:
        - mode is "conversational" or "thorough"
        - source is one of: "ui_toggle", "primitive", "trigger_phrase", "default"
    """
    # Priority 1: UI toggle (highest precedence)
    if ui_toggle and ui_toggle in ("conversational", "thorough"):
        return ui_toggle, "ui_toggle"
    
    # Priority 2: Primitive override
    if primitive_mode and primitive_mode in ("conversational", "thorough"):
        return primitive_mode, "primitive"
    
    # Priority 3: Trigger phrase detection
    detected = detect_retrieval_mode(utterance)
    if detected == "thorough":
        return "thorough", "trigger_phrase"
    
    # Priority 4: Default
    return "conversational", "default"


def get_mode_config(mode: Literal["conversational", "thorough"]):
    """
    Get the configuration object for a given mode.
    
    Args:
        mode: "conversational" or "thorough"
        
    Returns:
        ConversationalModeConfig or ThoroughModeConfig
    """
    if mode == "thorough":
        return ThoroughModeConfig()
    return ConversationalModeConfig()


# =============================================================================
# Threshold UI Helpers
# =============================================================================

def ui_threshold_to_internal(
    ui_value: float,
    ui_range: Tuple[float, float] = (0.0, 1.0),
    internal_range: Tuple[float, float] = (-1.0, 1.0),
) -> float:
    """
    Transform UI threshold (e.g., 0-100% slider) to internal [-1, 1] range.
    
    This helps users who think in "0 to 1 similarity" while
    internally we use cosine similarity [-1, 1].
    
    Args:
        ui_value: Value from UI (e.g., 0.5 for 50%)
        ui_range: Range of UI values (default 0-1)
        internal_range: Internal similarity range (default -1 to 1 for cosine)
        
    Returns:
        Transformed threshold in internal range
    """
    # Normalize to 0-1
    ui_min, ui_max = ui_range
    normalized = (ui_value - ui_min) / (ui_max - ui_min)
    
    # Scale to internal range
    int_min, int_max = internal_range
    return int_min + normalized * (int_max - int_min)


def internal_threshold_to_ui(
    internal_value: float,
    internal_range: Tuple[float, float] = (-1.0, 1.0),
    ui_range: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Transform internal threshold to UI range.
    
    Inverse of ui_threshold_to_internal.
    """
    int_min, int_max = internal_range
    normalized = (internal_value - int_min) / (int_max - int_min)
    
    ui_min, ui_max = ui_range
    return ui_min + normalized * (ui_max - ui_min)
