"""
V6 Progress Gate - Only expand if making actual progress

Problem: "50 entities in 3 rounds" while thrashing without quality gains.

Solution: Only do another round if:
- Evidence store gained new high-quality spans
- For roster: new members were identified
- Otherwise: stop or pivot strategy

This prevents thrashing and forces the system to converge.
"""
import sys
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from retrieval.agent.v6_query_parser import TaskType
from retrieval.agent.v6_evidence_bottleneck import BottleneckResult


# =============================================================================
# Progress Status
# =============================================================================

class ProgressStatus(Enum):
    MAKING_PROGRESS = "making_progress"  # Found new quality evidence
    DIMINISHING_RETURNS = "diminishing_returns"  # Some new evidence, but less
    NO_PROGRESS = "no_progress"  # Nothing new of value
    SATURATED = "saturated"  # Evidence store is full and stable


# =============================================================================
# Progress Decision
# =============================================================================

class RoundDecision(Enum):
    CONTINUE = "continue"  # Do another round
    STOP = "stop"  # Stop and synthesize
    PIVOT = "pivot"  # Try a different search strategy


# =============================================================================
# Progress State
# =============================================================================

@dataclass
class ProgressState:
    """Tracks progress across rounds."""
    
    # Per-round stats
    rounds_completed: int = 0
    evidence_per_round: List[int] = field(default_factory=list)
    new_members_per_round: List[int] = field(default_factory=list)
    
    # Cumulative
    total_evidence: int = 0
    all_members_found: Set[str] = field(default_factory=set)
    
    # Thresholds
    min_progress_threshold: int = 3  # Need at least this many new items
    max_rounds: int = 5
    max_no_progress_rounds: int = 2
    
    # State
    consecutive_no_progress: int = 0
    last_status: ProgressStatus = ProgressStatus.MAKING_PROGRESS


@dataclass
class ProgressResult:
    """Result of progress evaluation."""
    
    status: ProgressStatus = ProgressStatus.NO_PROGRESS
    decision: RoundDecision = RoundDecision.STOP
    
    # Stats from this round
    new_evidence_count: int = 0
    new_members_count: int = 0
    
    # Explanation
    reason: str = ""
    
    # Suggestions if pivoting
    pivot_suggestions: List[str] = field(default_factory=list)


# =============================================================================
# Progress Gate
# =============================================================================

class ProgressGate:
    """
    Evaluates whether to continue retrieval rounds.
    
    Prevents thrashing by requiring actual progress:
    - New high-quality evidence added
    - New members identified (for roster)
    - Otherwise: stop or pivot
    """
    
    def __init__(
        self,
        min_progress: int = 3,
        max_rounds: int = 5,
        max_no_progress: int = 2,
        verbose: bool = True,
    ):
        self.min_progress = min_progress
        self.max_rounds = max_rounds
        self.max_no_progress = max_no_progress
        self.verbose = verbose
        
        self.state = ProgressState(
            min_progress_threshold=min_progress,
            max_rounds=max_rounds,
            max_no_progress_rounds=max_no_progress,
        )
    
    def evaluate(
        self,
        bottleneck_result: BottleneckResult,
        task_type: TaskType,
        previous_evidence_count: int,
        previous_members: Set[str],
    ) -> ProgressResult:
        """
        Evaluate progress from this round.
        
        Args:
            bottleneck_result: Result from evidence bottleneck
            task_type: Type of task
            previous_evidence_count: Evidence count before this round
            previous_members: Members found before this round
        
        Returns:
            ProgressResult with decision
        """
        result = ProgressResult()
        
        # Calculate new items
        new_evidence = bottleneck_result.spans_passed - previous_evidence_count
        result.new_evidence_count = max(0, new_evidence)
        
        # For roster: count new members
        current_members = set(bottleneck_result.members_identified)
        new_members = current_members - previous_members
        result.new_members_count = len(new_members)
        
        # Update state
        self.state.rounds_completed += 1
        self.state.evidence_per_round.append(result.new_evidence_count)
        self.state.new_members_per_round.append(result.new_members_count)
        self.state.total_evidence = bottleneck_result.spans_passed
        self.state.all_members_found.update(current_members)
        
        # Determine status
        if task_type == TaskType.ROSTER_ENUMERATION:
            # For roster, focus on new members found
            if result.new_members_count >= self.min_progress:
                result.status = ProgressStatus.MAKING_PROGRESS
                self.state.consecutive_no_progress = 0
            elif result.new_members_count > 0:
                result.status = ProgressStatus.DIMINISHING_RETURNS
                self.state.consecutive_no_progress = 0
            else:
                result.status = ProgressStatus.NO_PROGRESS
                self.state.consecutive_no_progress += 1
        else:
            # For other types, focus on new evidence
            if result.new_evidence_count >= self.min_progress:
                result.status = ProgressStatus.MAKING_PROGRESS
                self.state.consecutive_no_progress = 0
            elif result.new_evidence_count > 0:
                result.status = ProgressStatus.DIMINISHING_RETURNS
                self.state.consecutive_no_progress = 0
            else:
                result.status = ProgressStatus.NO_PROGRESS
                self.state.consecutive_no_progress += 1
        
        self.state.last_status = result.status
        
        # Make decision
        result.decision, result.reason = self._make_decision(result, task_type)
        
        if self.verbose:
            print(f"  [ProgressGate] Round {self.state.rounds_completed}: {result.status.value}", 
                  file=sys.stderr)
            print(f"    New evidence: {result.new_evidence_count}, New members: {result.new_members_count}", 
                  file=sys.stderr)
            print(f"    Decision: {result.decision.value} - {result.reason}", file=sys.stderr)
        
        return result
    
    def _make_decision(
        self,
        progress: ProgressResult,
        task_type: TaskType,
    ) -> tuple:
        """Make continue/stop/pivot decision."""
        
        # Check hard limits
        if self.state.rounds_completed >= self.max_rounds:
            return RoundDecision.STOP, f"Max rounds ({self.max_rounds}) reached"
        
        if self.state.consecutive_no_progress >= self.max_no_progress:
            # Should we pivot or stop?
            if self.state.total_evidence < 10:
                return RoundDecision.PIVOT, f"No progress for {self.max_no_progress} rounds, try different strategy"
            else:
                return RoundDecision.STOP, f"No progress for {self.max_no_progress} rounds, have enough evidence"
        
        # Check progress status
        if progress.status == ProgressStatus.MAKING_PROGRESS:
            return RoundDecision.CONTINUE, "Making good progress"
        
        if progress.status == ProgressStatus.DIMINISHING_RETURNS:
            # Continue if we don't have much yet
            if self.state.total_evidence < 20:
                return RoundDecision.CONTINUE, "Diminishing returns but need more evidence"
            else:
                return RoundDecision.STOP, "Diminishing returns, have enough evidence"
        
        if progress.status == ProgressStatus.NO_PROGRESS:
            if self.state.total_evidence < 10:
                return RoundDecision.PIVOT, "No progress, try different strategy"
            else:
                return RoundDecision.STOP, "No progress, synthesize with current evidence"
        
        return RoundDecision.STOP, "Default: stop and synthesize"
    
    def get_pivot_suggestions(self, task_type: TaskType) -> List[str]:
        """Get suggestions for pivoting search strategy."""
        
        suggestions = []
        
        if task_type == TaskType.ROSTER_ENUMERATION:
            suggestions = [
                "Try searching for specific known members by name",
                "Search for 'recruited by' or 'worked with' patterns",
                "Expand to related networks or groups",
                "Try different collection sources",
            ]
        else:
            suggestions = [
                "Broaden search terms",
                "Try related concepts",
                "Search different collections",
            ]
        
        return suggestions
    
    def should_continue(self) -> bool:
        """Simple check: should we do another round?"""
        
        if self.state.rounds_completed >= self.max_rounds:
            return False
        if self.state.consecutive_no_progress >= self.max_no_progress:
            return False
        if self.state.last_status == ProgressStatus.NO_PROGRESS:
            return False
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of progress across all rounds."""
        
        return {
            "rounds_completed": self.state.rounds_completed,
            "total_evidence": self.state.total_evidence,
            "total_members_found": len(self.state.all_members_found),
            "members_found": list(self.state.all_members_found),
            "evidence_per_round": self.state.evidence_per_round,
            "new_members_per_round": self.state.new_members_per_round,
            "consecutive_no_progress": self.state.consecutive_no_progress,
            "last_status": self.state.last_status.value,
        }
