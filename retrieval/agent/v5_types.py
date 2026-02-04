"""
V5 Agentic Retrieval - Type Definitions

LLM-only grading architecture with:
- Searcher: free tool choice agent
- Grader: LLM-based evidence evaluation
- EvidenceStore: budget-constrained best-evidence keeper
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
import hashlib
import time


# =============================================================================
# Enums
# =============================================================================

class EvidenceStatus(Enum):
    ACTIVE = "active"
    EVICTED = "evicted"


class ActionType(Enum):
    CALL_TOOL = "call_tool"
    STOP_ANSWER = "stop_answer"
    STOP_INSUFFICIENT = "stop_insufficient"


# =============================================================================
# CandidateSpan - What tools return / what we grade
# =============================================================================

@dataclass
class CandidateSpan:
    """Raw candidate from a tool, before grading."""
    
    candidate_id: str  # Unique ID for this candidate
    chunk_id: int
    doc_id: Optional[int]
    page: Optional[str]
    span_text: str
    surrounding_context: str = ""  # Small window around the span
    source_label: str = ""  # Collection/doc type
    source_tool: str = ""  # Which tool produced this
    
    # Metadata from the tool
    score: float = 0.0  # Tool-assigned score (if any)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.candidate_id:
            # Generate stable ID from content
            content = f"{self.chunk_id}:{self.span_text[:100]}"
            self.candidate_id = hashlib.md5(content.encode()).hexdigest()[:12]


# =============================================================================
# GraderResult - What the grader LLM returns
# =============================================================================

@dataclass
class GraderResult:
    """Result of grading a single candidate span."""
    
    candidate_id: str
    supports_question: bool
    support_strength: int  # 0-3: 0=none, 1=weak, 2=moderate, 3=strong
    quote_grade: bool  # Is it directly citable?
    claim_supported: str  # Short text of the claim it supports
    notes: str  # One sentence why (or why not)
    
    def __post_init__(self):
        # Clamp support_strength to valid range
        self.support_strength = max(0, min(3, self.support_strength))


# =============================================================================
# EvidenceItem - What we keep in the store
# =============================================================================

@dataclass
class EvidenceItem:
    """Graded evidence item kept in the store."""
    
    evidence_id: str  # Stable ID
    candidate_id: str  # Provenance to original candidate
    
    # Content
    span_text: str
    chunk_id: int
    doc_id: Optional[int]
    page: Optional[str]
    source_label: str
    
    # Grading results
    support_strength: int
    quote_grade: bool
    claim_supported: str
    grader_notes: str
    
    # Status
    status: EvidenceStatus = EvidenceStatus.ACTIVE
    added_at_step: int = 0
    evicted_at_step: Optional[int] = None
    
    def __post_init__(self):
        if not self.evidence_id:
            self.evidence_id = f"ev_{self.candidate_id}"
    
    @property
    def sort_key(self) -> tuple:
        """Sort key for ranking: higher is better."""
        # Primary: support_strength (higher better)
        # Secondary: quote_grade (True better)
        return (self.support_strength, int(self.quote_grade))
    
    def to_compact_view(self) -> str:
        """Compact representation for searcher prompt."""
        grade = "★" * self.support_strength
        quote = "Q" if self.quote_grade else ""
        return f"[{self.evidence_id}] {grade}{quote} - {self.claim_supported[:60]}"


# =============================================================================
# EvidenceStore - Budget-constrained best-evidence keeper
# =============================================================================

@dataclass
class EvidenceStore:
    """
    Keeps top evidence items by grader score under a budget.
    
    Eviction rule:
    - Keep highest support_strength items first
    - Prefer quote_grade=True
    - If full, evict lowest-ranked items
    """
    
    capacity: int = 25
    items: Dict[str, EvidenceItem] = field(default_factory=dict)
    eviction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len([i for i in self.items.values() if i.status == EvidenceStatus.ACTIVE])
    
    @property
    def active_items(self) -> List[EvidenceItem]:
        """Get all active items sorted by quality (best first)."""
        active = [i for i in self.items.values() if i.status == EvidenceStatus.ACTIVE]
        return sorted(active, key=lambda x: x.sort_key, reverse=True)
    
    def get_worst_active(self) -> Optional[EvidenceItem]:
        """Get the lowest-ranked active item."""
        active = self.active_items
        return active[-1] if active else None
    
    def add_item(self, item: EvidenceItem, current_step: int) -> Dict[str, Any]:
        """
        Add an item to the store, evicting if necessary.
        
        Returns:
            Dict with 'added': bool, 'evicted': Optional[str]
        """
        result = {"added": False, "evicted": None, "reason": ""}
        
        # Check if this candidate is already in store
        if item.evidence_id in self.items:
            existing = self.items[item.evidence_id]
            if existing.status == EvidenceStatus.ACTIVE:
                result["reason"] = "duplicate"
                return result
        
        item.added_at_step = current_step
        
        # If under capacity, just add
        if len(self) < self.capacity:
            self.items[item.evidence_id] = item
            result["added"] = True
            result["reason"] = "under_capacity"
            return result
        
        # Store is full - check if new item is better than worst
        worst = self.get_worst_active()
        if worst is None:
            result["reason"] = "no_active_items"
            return result
        
        # Compare: new item must be strictly better
        if item.sort_key > worst.sort_key:
            # Evict the worst
            worst.status = EvidenceStatus.EVICTED
            worst.evicted_at_step = current_step
            self.eviction_history.append({
                "evidence_id": worst.evidence_id,
                "evicted_at_step": current_step,
                "evicted_by": item.evidence_id,
                "old_strength": worst.support_strength,
                "new_strength": item.support_strength,
            })
            
            # Add the new item
            self.items[item.evidence_id] = item
            result["added"] = True
            result["evicted"] = worst.evidence_id
            result["reason"] = "replaced_worse"
        else:
            result["reason"] = "not_better_than_worst"
        
        return result
    
    def get_compact_view(self) -> str:
        """Generate compact view for searcher prompt."""
        if not self.active_items:
            return "Evidence store is empty."
        
        lines = [f"Evidence store ({len(self)}/{self.capacity}):"]
        for item in self.active_items[:15]:  # Show top 15
            lines.append(f"  {item.to_compact_view()}")
        
        if len(self) > 15:
            lines.append(f"  ... and {len(self) - 15} more")
        
        return "\n".join(lines)
    
    def get_claims_coverage(self) -> Dict[str, List[str]]:
        """Get mapping of claims to supporting evidence IDs."""
        coverage = {}
        for item in self.active_items:
            claim = item.claim_supported
            if claim not in coverage:
                coverage[claim] = []
            coverage[claim].append(item.evidence_id)
        return coverage
    
    def get_citation_ready_items(self, min_strength: int = 2) -> List[EvidenceItem]:
        """Get items that meet citation threshold."""
        return [i for i in self.active_items if i.support_strength >= min_strength]


# =============================================================================
# Searcher Actions - What the searcher can do
# =============================================================================

@dataclass
class ToolCallAction:
    """Searcher wants to call a tool."""
    tool_name: str
    params: Dict[str, Any]
    rationale: str = ""


@dataclass
class StopAnswerAction:
    """Searcher wants to stop with a complete answer."""
    answer: str
    major_claims: List[Dict[str, Any]]  # Each has 'claim' and 'evidence_ids'
    
    def validate_citations(self, store: EvidenceStore, min_strength: int = 2) -> Dict[str, Any]:
        """Check if all claims have valid citations."""
        errors = []
        valid_items = {i.evidence_id for i in store.get_citation_ready_items(min_strength)}
        
        for claim_data in self.major_claims:
            claim = claim_data.get("claim", "")
            evidence_ids = claim_data.get("evidence_ids", [])
            
            if not evidence_ids:
                errors.append(f"Claim '{claim[:50]}...' has no citations")
                continue
            
            for eid in evidence_ids:
                if eid not in valid_items:
                    errors.append(f"Evidence '{eid}' not found or below strength threshold")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }


@dataclass
class StopInsufficientAction:
    """Searcher wants to stop but evidence is insufficient."""
    partial_answer: str
    what_missing: str
    suggested_next_tool: Optional[ToolCallAction] = None


# =============================================================================
# Budgets
# =============================================================================

@dataclass
class V5Budgets:
    """Budget constraints for V5 agentic retrieval."""
    
    max_steps: int = 12  # Total tool calls allowed
    max_candidates_per_step: int = 30  # Raw candidates per tool call
    evidence_budget: int = 25  # Max items in evidence store
    max_grader_calls: int = 500  # Total grader invocations
    min_citation_strength: int = 2  # Required strength for citations
    repair_budget: int = 2  # Extra steps for repair after stop
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "max_candidates_per_step": self.max_candidates_per_step,
            "evidence_budget": self.evidence_budget,
            "max_grader_calls": self.max_grader_calls,
            "min_citation_strength": self.min_citation_strength,
            "repair_budget": self.repair_budget,
        }


# =============================================================================
# Step Log - For auditing
# =============================================================================

@dataclass
class StepLog:
    """Log of a single step in the controller loop."""
    
    step_number: int
    timestamp: float = field(default_factory=time.time)
    
    # Action taken
    action_type: str = ""
    action_details: Dict[str, Any] = field(default_factory=dict)
    
    # Tool execution (if applicable)
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    candidates_produced: int = 0
    
    # Grading results
    grader_calls: int = 0
    graded_candidates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Store mutations
    items_added: List[str] = field(default_factory=list)
    items_evicted: List[str] = field(default_factory=list)
    
    # Stop attempt (if applicable)
    stop_attempted: bool = False
    stop_accepted: bool = False
    stop_rejection_reason: str = ""
    
    # Timing
    elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "candidates_produced": self.candidates_produced,
            "grader_calls": self.grader_calls,
            "items_added": self.items_added,
            "items_evicted": self.items_evicted,
            "stop_attempted": self.stop_attempted,
            "stop_accepted": self.stop_accepted,
            "stop_rejection_reason": self.stop_rejection_reason,
            "elapsed_ms": self.elapsed_ms,
        }


# =============================================================================
# V5 Trace - Complete execution trace
# =============================================================================

@dataclass
class V5Trace:
    """Complete trace of V5 execution for auditing."""
    
    question: str = ""
    budgets: V5Budgets = field(default_factory=V5Budgets)
    
    # Execution
    steps: List[StepLog] = field(default_factory=list)
    total_steps: int = 0
    total_grader_calls: int = 0
    
    # Final state
    final_evidence_count: int = 0
    final_answer: str = ""
    final_claims: List[Dict[str, Any]] = field(default_factory=list)
    
    # Outcome
    stopped_reason: str = ""  # "answer_accepted", "insufficient", "budget_exhausted"
    verification_passed: bool = False
    
    # Timing
    total_elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "budgets": self.budgets.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": self.total_steps,
            "total_grader_calls": self.total_grader_calls,
            "final_evidence_count": self.final_evidence_count,
            "final_answer": self.final_answer,
            "final_claims": self.final_claims,
            "stopped_reason": self.stopped_reason,
            "verification_passed": self.verification_passed,
            "total_elapsed_ms": self.total_elapsed_ms,
        }
