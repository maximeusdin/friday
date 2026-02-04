"""
V4 Discovery Metrics - Coverage signals and stop conditions.

This module provides deterministic coverage metrics computed from evidence spans,
used to decide when discovery is sufficient.

Coverage Signals (shape-agnostic):
- entity_attest_counts: How many spans attest to each entity surface
- list_like_span_count: Spans with list formatting (numbered, bullets)
- definitional_span_count: Spans with "X is..." patterns
- doc_concentration: Distribution of spans across documents
- marginal_gain: New content added this round

Stop Conditions:
- Coverage satisfied: Sufficient list/definitional spans + entity attestation
- Marginal gain low: <10% new content per round
- Budget reached: Max rounds/actions/chunks
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter

from retrieval.agent.v3_evidence import EvidenceSpan, EvidenceSet


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DiscoveryBudgets:
    """Budget limits for discovery."""
    max_rounds: int = 2
    max_actions_per_round: int = 6
    max_candidate_chunks: int = 1500
    max_cite_spans: int = 120
    
    # Stop condition thresholds
    min_list_spans_for_good_coverage: int = 5
    min_definitional_spans_for_good_coverage: int = 3
    min_entity_attests_for_good_coverage: int = 4
    marginal_gain_threshold: float = 0.10  # 10%
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "max_actions_per_round": self.max_actions_per_round,
            "max_candidate_chunks": self.max_candidate_chunks,
            "max_cite_spans": self.max_cite_spans,
            "min_list_spans": self.min_list_spans_for_good_coverage,
            "min_definitional_spans": self.min_definitional_spans_for_good_coverage,
            "min_entity_attests": self.min_entity_attests_for_good_coverage,
            "marginal_gain_threshold": self.marginal_gain_threshold,
        }


DEFAULT_BUDGETS = DiscoveryBudgets()

THOROUGH_BUDGETS = DiscoveryBudgets(
    max_rounds=4,
    max_actions_per_round=8,
    max_candidate_chunks=2500,
    max_cite_spans=200,
)


# =============================================================================
# Pattern Detection
# =============================================================================

# Patterns that indicate list-like content (roster pages, etc.)
LIST_PATTERNS = [
    r'^\s*\d+[\.\)]\s',          # "1. " or "1) "
    r'^\s*[-\*\•]\s',            # "- " or "* " or "• "
    r'^\s*[a-zA-Z][\.\)]\s',     # "a. " or "A) "
    r'\bsources?\s+include\b',   # "sources include"
    r'\bmembers?\s+(?:are|were|include)\b',  # "members are/were/include"
    r'\bgroup\s+(?:consisted|comprised|included)\b',  # "group consisted of"
    r'\bnetwork\s+(?:consisted|comprised|included)\b',
    r'\b(?:included|comprising|consisting\s+of)\b',
]

# Patterns that indicate definitional content
DEFINITIONAL_PATTERNS = [
    r'\b(?:is|was|were)\s+(?:a|an|the)\s+\w+',  # "X is a Y"
    r'\bknown\s+as\b',           # "known as"
    r'\bcalled\b',               # "called"
    r'\bcodename[d]?\b',         # "codenamed"
    r'\balias(?:es)?\b',         # "alias/aliases"
    r'\bcover\s+name\b',         # "cover name"
    r'\bidentified\s+as\b',      # "identified as"
    r'\b(?:real|true)\s+name\b', # "real/true name"
]


def is_list_like(text: str) -> bool:
    """Check if text has list-like formatting."""
    text_lower = text.lower()
    for pattern in LIST_PATTERNS:
        if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
            return True
    return False


def is_definitional(text: str) -> bool:
    """Check if text has definitional patterns."""
    text_lower = text.lower()
    for pattern in DEFINITIONAL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def count_entity_attestations(
    spans: List[EvidenceSpan],
    target_surfaces: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Count how many spans attest to each entity surface.
    
    Args:
        spans: List of evidence spans
        target_surfaces: Optional list of surfaces to check (if None, extract from spans)
    
    Returns:
        Dict mapping surface -> count of spans containing it
    """
    counts: Dict[str, int] = Counter()
    
    for span in spans:
        text_lower = span.quote.lower()
        attest_lower = (span.attest_text or span.quote).lower()
        
        if target_surfaces:
            for surface in target_surfaces:
                surface_lower = surface.lower()
                if surface_lower in attest_lower:
                    counts[surface] += 1
        else:
            # Extract potential entity-like tokens (capitalized words)
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', span.quote)
            for word in words:
                if len(word) > 2:
                    counts[word.lower()] += 1
    
    return dict(counts)


def compute_doc_concentration(spans: List[EvidenceSpan]) -> Dict[int, int]:
    """Compute span count per document."""
    counts: Dict[int, int] = Counter()
    for span in spans:
        counts[span.doc_id] += 1
    return dict(counts)


# =============================================================================
# Coverage Metrics
# =============================================================================

@dataclass
class CoverageMetrics:
    """Complete coverage metrics for evidence set."""
    total_spans: int
    total_chunks: int
    unique_docs: int
    unique_pages: int
    
    list_like_span_count: int
    definitional_span_count: int
    
    entity_attest_counts: Dict[str, int]
    doc_concentration: Dict[int, int]
    
    # For marginal gain calculation
    prev_total_spans: int = 0
    prev_unique_docs: int = 0
    
    @property
    def marginal_gain_spans(self) -> float:
        """Percentage of new spans this round."""
        if self.prev_total_spans == 0:
            return 1.0  # First round is 100% new
        new_spans = self.total_spans - self.prev_total_spans
        return new_spans / max(self.prev_total_spans, 1)
    
    @property
    def marginal_gain_docs(self) -> float:
        """Percentage of new docs this round."""
        if self.prev_unique_docs == 0:
            return 1.0
        new_docs = self.unique_docs - self.prev_unique_docs
        return new_docs / max(self.prev_unique_docs, 1)
    
    @property
    def marginal_gain(self) -> float:
        """Combined marginal gain (average of spans and docs)."""
        return (self.marginal_gain_spans + self.marginal_gain_docs) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_spans": self.total_spans,
            "total_chunks": self.total_chunks,
            "unique_docs": self.unique_docs,
            "unique_pages": self.unique_pages,
            "list_like_span_count": self.list_like_span_count,
            "definitional_span_count": self.definitional_span_count,
            "entity_attest_counts": dict(list(self.entity_attest_counts.items())[:20]),
            "marginal_gain": round(self.marginal_gain * 100, 1),
        }


def compute_coverage_metrics(
    evidence_set: EvidenceSet,
    target_surfaces: Optional[List[str]] = None,
    prev_metrics: Optional[CoverageMetrics] = None,
) -> CoverageMetrics:
    """
    Compute coverage metrics from evidence set.
    
    Args:
        evidence_set: Current evidence set
        target_surfaces: Optional surfaces to track attestation for
        prev_metrics: Previous metrics for marginal gain calculation
    
    Returns:
        CoverageMetrics with all signals computed
    """
    all_spans = evidence_set.cite_spans + evidence_set.harvest_spans
    
    # Basic counts
    total_spans = len(all_spans)
    unique_docs = len({s.doc_id for s in all_spans})
    unique_pages = len({s.page_ref for s in all_spans})
    
    # Pattern counts
    list_like_count = sum(1 for s in all_spans if is_list_like(s.quote))
    definitional_count = sum(1 for s in all_spans if is_definitional(s.quote))
    
    # Entity attestation
    entity_attests = count_entity_attestations(all_spans, target_surfaces)
    
    # Doc concentration
    doc_concentration = compute_doc_concentration(all_spans)
    
    return CoverageMetrics(
        total_spans=total_spans,
        total_chunks=evidence_set.stats.total_chunks if evidence_set.stats else 0,
        unique_docs=unique_docs,
        unique_pages=unique_pages,
        list_like_span_count=list_like_count,
        definitional_span_count=definitional_count,
        entity_attest_counts=entity_attests,
        doc_concentration=doc_concentration,
        prev_total_spans=prev_metrics.total_spans if prev_metrics else 0,
        prev_unique_docs=prev_metrics.unique_docs if prev_metrics else 0,
    )


# =============================================================================
# Stop Condition Logic
# =============================================================================

@dataclass
class StopDecision:
    """Decision about whether to stop discovery."""
    should_stop: bool
    reason: str
    confidence: str  # high|medium|low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "reason": self.reason,
            "confidence": self.confidence,
        }


def evaluate_stop_conditions(
    metrics: CoverageMetrics,
    round_num: int,
    budgets: DiscoveryBudgets = DEFAULT_BUDGETS,
) -> StopDecision:
    """
    Evaluate whether discovery should stop.
    
    Stop conditions (any one triggers stop):
    1. Budget reached: max rounds
    2. Good coverage: sufficient list/definitional spans + entity attestation
    3. Low marginal gain: <threshold new content
    
    Args:
        metrics: Current coverage metrics
        round_num: Current round number
        budgets: Budget configuration
    
    Returns:
        StopDecision with recommendation
    """
    # Budget stop
    if round_num >= budgets.max_rounds:
        return StopDecision(
            should_stop=True,
            reason=f"Max rounds reached ({budgets.max_rounds})",
            confidence="high",
        )
    
    # Good coverage stop
    has_list_coverage = metrics.list_like_span_count >= budgets.min_list_spans_for_good_coverage
    has_definitional_coverage = metrics.definitional_span_count >= budgets.min_definitional_spans_for_good_coverage
    
    # Count well-attested entities (>= 2 mentions)
    well_attested = sum(1 for count in metrics.entity_attest_counts.values() if count >= 2)
    has_entity_coverage = well_attested >= budgets.min_entity_attests_for_good_coverage
    
    if has_list_coverage and has_entity_coverage:
        return StopDecision(
            should_stop=True,
            reason=f"Good coverage: {metrics.list_like_span_count} list spans, {well_attested} well-attested entities",
            confidence="high",
        )
    
    if has_definitional_coverage and has_entity_coverage:
        return StopDecision(
            should_stop=True,
            reason=f"Good coverage: {metrics.definitional_span_count} definitional spans, {well_attested} well-attested entities",
            confidence="high",
        )
    
    # Low marginal gain stop (only after round 1)
    if round_num > 1 and metrics.marginal_gain < budgets.marginal_gain_threshold:
        return StopDecision(
            should_stop=True,
            reason=f"Low marginal gain ({metrics.marginal_gain:.1%} < {budgets.marginal_gain_threshold:.1%})",
            confidence="medium",
        )
    
    # Continue discovery
    missing = []
    if not has_list_coverage:
        missing.append(f"list spans ({metrics.list_like_span_count}/{budgets.min_list_spans_for_good_coverage})")
    if not has_entity_coverage:
        missing.append(f"entity attestation ({well_attested}/{budgets.min_entity_attests_for_good_coverage})")
    
    return StopDecision(
        should_stop=False,
        reason=f"Coverage gaps: {', '.join(missing)}",
        confidence="medium",
    )


def should_run_discovery(
    initial_evidence: EvidenceSet,
    budgets: DiscoveryBudgets = DEFAULT_BUDGETS,
) -> Tuple[bool, str]:
    """
    Decide if discovery loop should run based on initial evidence.
    
    Always run at least 1 round by default, but skip if initial coverage is very good.
    
    Args:
        initial_evidence: Evidence from initial V3 plan execution
        budgets: Budget configuration
    
    Returns:
        (should_run, reason) tuple
    """
    metrics = compute_coverage_metrics(initial_evidence)
    
    # If very good initial coverage, might skip discovery
    has_strong_list = metrics.list_like_span_count >= budgets.min_list_spans_for_good_coverage * 2
    has_strong_def = metrics.definitional_span_count >= budgets.min_definitional_spans_for_good_coverage * 2
    
    if has_strong_list and has_strong_def:
        return False, "Strong initial coverage - skipping discovery"
    
    # Default: always run at least 1 round
    return True, "Running discovery to improve coverage"
