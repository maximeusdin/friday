"""
V5 Hypothesis - Working hypotheses for iterative refinement

The system forms intermediate hypotheses and seeks evidence to test them.
This is what makes the search "reasoning-like" - the closed loop where
intermediate structure (hypotheses) changes what the agent seeks next.

For "Silvermaster network members":
- Hypothesis: "X was a member" -> seek evidence
- Refine: "X was member, role was Y" -> seek more specific evidence
- Test: Does evidence support or contradict?
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Hypothesis Status
# =============================================================================

class HypothesisStatus(Enum):
    PROPOSED = "proposed"  # Just formed, needs evidence
    TESTING = "testing"    # Actively seeking evidence
    SUPPORTED = "supported"  # Has strong evidence
    WEAKLY_SUPPORTED = "weakly_supported"  # Has some evidence
    CONTRADICTED = "contradicted"  # Evidence against
    UNCERTAIN = "uncertain"  # Mixed or no evidence


# =============================================================================
# Working Hypothesis
# =============================================================================

@dataclass
class WorkingHypothesis:
    """
    A working hypothesis the system is testing.
    
    The hypothesis shapes what evidence the agent seeks next.
    """
    
    hypothesis_id: str
    claim: str  # The hypothesis statement (e.g., "Harry White was a member")
    category: str  # Type: "membership", "date", "relationship", etc.
    
    # Evidence tracking
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    
    # Status
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = 0.0  # 0-1
    
    # Metadata
    proposed_at_step: int = 0
    last_updated_step: int = 0
    search_terms: List[str] = field(default_factory=list)  # Terms to search for
    
    def update_status(self):
        """Update status based on evidence."""
        supporting = len(self.supporting_evidence_ids)
        contradicting = len(self.contradicting_evidence_ids)
        
        if contradicting > supporting:
            self.status = HypothesisStatus.CONTRADICTED
            self.confidence = 0.0
        elif supporting >= 2:
            self.status = HypothesisStatus.SUPPORTED
            self.confidence = min(1.0, 0.5 + 0.2 * supporting)
        elif supporting == 1:
            self.status = HypothesisStatus.WEAKLY_SUPPORTED
            self.confidence = 0.4
        elif supporting == 0 and contradicting == 0:
            self.status = HypothesisStatus.PROPOSED
            self.confidence = 0.0
        else:
            self.status = HypothesisStatus.UNCERTAIN
            self.confidence = 0.2
    
    def to_compact_view(self) -> str:
        """Compact view for prompts."""
        status_emoji = {
            HypothesisStatus.PROPOSED: "?",
            HypothesisStatus.TESTING: "🔍",
            HypothesisStatus.SUPPORTED: "✓",
            HypothesisStatus.WEAKLY_SUPPORTED: "~",
            HypothesisStatus.CONTRADICTED: "✗",
            HypothesisStatus.UNCERTAIN: "?",
        }.get(self.status, "?")
        
        evidence_count = f"+{len(self.supporting_evidence_ids)}/-{len(self.contradicting_evidence_ids)}"
        return f"[{self.hypothesis_id}] {status_emoji} {self.claim} ({evidence_count})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "category": self.category,
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence_ids,
            "contradicting_evidence": self.contradicting_evidence_ids,
            "search_terms": self.search_terms,
        }


# =============================================================================
# Hypothesis Set
# =============================================================================

@dataclass
class HypothesisSet:
    """
    The set of working hypotheses being tested.
    
    This is what changes what the agent seeks next.
    """
    
    hypotheses: Dict[str, WorkingHypothesis] = field(default_factory=dict)
    max_hypotheses: int = 15
    
    def add_hypothesis(
        self,
        claim: str,
        category: str,
        current_step: int,
        search_terms: Optional[List[str]] = None,
    ) -> Optional[WorkingHypothesis]:
        """Add a new hypothesis if not duplicate."""
        
        # Check for duplicates (similar claims)
        claim_lower = claim.lower()
        for h in self.hypotheses.values():
            if claim_lower in h.claim.lower() or h.claim.lower() in claim_lower:
                return None  # Duplicate
        
        # Generate ID
        import hashlib
        h_id = f"h_{hashlib.md5(claim.encode()).hexdigest()[:8]}"
        
        # Check capacity
        if len(self.hypotheses) >= self.max_hypotheses:
            # Evict least confident unsupported hypothesis
            to_evict = None
            min_conf = 1.0
            for h in self.hypotheses.values():
                if h.status not in (HypothesisStatus.SUPPORTED, HypothesisStatus.WEAKLY_SUPPORTED):
                    if h.confidence < min_conf:
                        min_conf = h.confidence
                        to_evict = h.hypothesis_id
            
            if to_evict:
                del self.hypotheses[to_evict]
            else:
                return None  # All hypotheses are important
        
        hypothesis = WorkingHypothesis(
            hypothesis_id=h_id,
            claim=claim,
            category=category,
            proposed_at_step=current_step,
            last_updated_step=current_step,
            search_terms=search_terms or [],
        )
        
        self.hypotheses[h_id] = hypothesis
        return hypothesis
    
    def add_evidence(
        self,
        hypothesis_id: str,
        evidence_id: str,
        supports: bool,
        current_step: int,
    ):
        """Add evidence to a hypothesis."""
        if hypothesis_id not in self.hypotheses:
            return
        
        h = self.hypotheses[hypothesis_id]
        
        if supports:
            if evidence_id not in h.supporting_evidence_ids:
                h.supporting_evidence_ids.append(evidence_id)
        else:
            if evidence_id not in h.contradicting_evidence_ids:
                h.contradicting_evidence_ids.append(evidence_id)
        
        h.last_updated_step = current_step
        h.update_status()
    
    def get_untested_hypotheses(self) -> List[WorkingHypothesis]:
        """Get hypotheses that need more evidence."""
        return [
            h for h in self.hypotheses.values()
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING, HypothesisStatus.UNCERTAIN)
        ]
    
    def get_supported_hypotheses(self) -> List[WorkingHypothesis]:
        """Get hypotheses with strong support."""
        return [
            h for h in self.hypotheses.values()
            if h.status in (HypothesisStatus.SUPPORTED, HypothesisStatus.WEAKLY_SUPPORTED)
        ]
    
    def get_search_suggestions(self) -> List[str]:
        """Get search terms from untested hypotheses."""
        terms = []
        for h in self.get_untested_hypotheses():
            terms.extend(h.search_terms)
        return list(set(terms))[:10]
    
    def get_compact_view(self) -> str:
        """Generate compact view for prompts."""
        if not self.hypotheses:
            return "No working hypotheses yet."
        
        lines = ["Working Hypotheses:"]
        
        # Group by status
        by_status = {}
        for h in self.hypotheses.values():
            status = h.status.value
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(h)
        
        for status in ["supported", "weakly_supported", "testing", "proposed", "uncertain", "contradicted"]:
            if status in by_status:
                lines.append(f"  {status.upper()}:")
                for h in by_status[status][:5]:
                    lines.append(f"    {h.to_compact_view()}")
        
        return "\n".join(lines)


# =============================================================================
# Hypothesis Generator Prompt
# =============================================================================

HYPOTHESIS_SYSTEM_PROMPT = """You are forming working hypotheses to guide evidence search.

Given a research question and current evidence, propose specific testable hypotheses.

For "Who were Silvermaster network members?":
- "Harry Dexter White was a member" (search: "White" + "Silvermaster")
- "William Ullmann was a member" (search: "Ullmann")
- "The network operated at Treasury Department" (search: "Treasury" + "Silvermaster")

Each hypothesis should be:
- SPECIFIC: A concrete claim that can be confirmed or denied
- TESTABLE: Suggests what evidence to seek
- USEFUL: Helps answer the main question"""


def build_hypothesis_prompt(
    question: str,
    current_hypotheses: HypothesisSet,
    recent_evidence: List[Dict[str, Any]],
    recent_claims: List[str],
) -> str:
    """Build prompt for generating new hypotheses."""
    
    hyp_view = current_hypotheses.get_compact_view()
    
    evidence_section = ""
    if recent_evidence:
        evidence_lines = []
        for ev in recent_evidence[:10]:
            evidence_lines.append(f"  - {ev.get('claim_supported', 'unknown')[:80]}")
        evidence_section = f"\nRecent evidence suggests:\n" + "\n".join(evidence_lines)
    
    claims_section = ""
    if recent_claims:
        claims_section = f"\nClaims found in evidence:\n" + "\n".join(f"  - {c}" for c in recent_claims[:10])
    
    return f"""RESEARCH QUESTION: {question}

{hyp_view}
{evidence_section}
{claims_section}

Based on the question and current state, propose 2-4 NEW hypotheses to test.
Focus on hypotheses that would help complete the answer.

Output JSON:
{{
  "hypotheses": [
    {{
      "claim": "specific testable claim",
      "category": "membership/date/relationship/role/event",
      "search_terms": ["term1", "term2"],
      "rationale": "why test this"
    }}
  ]
}}"""


# =============================================================================
# Hypothesis Generator
# =============================================================================

class HypothesisGenerator:
    """
    Generates working hypotheses to guide search.
    
    This is the "reasoning" part - forming intermediate structure
    that changes what the agent seeks next.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self.total_calls = 0
    
    def generate_hypotheses(
        self,
        question: str,
        hypothesis_set: HypothesisSet,
        recent_evidence: List[Dict[str, Any]],
        current_step: int,
    ) -> List[WorkingHypothesis]:
        """
        Generate new hypotheses based on current state.
        
        Returns:
            List of newly added hypotheses
        """
        # Extract claims from recent evidence
        recent_claims = [
            ev.get("claim_supported", "") 
            for ev in recent_evidence 
            if ev.get("claim_supported")
        ]
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []
        
        prompt = build_hypothesis_prompt(
            question=question,
            current_hypotheses=hypothesis_set,
            recent_evidence=recent_evidence,
            recent_claims=recent_claims,
        )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1000,
            )
            
            self.total_calls += 1
            
            content = response.choices[0].message.content
            if not content:
                return []
            
            data = json.loads(content)
            
            new_hypotheses = []
            for h_data in data.get("hypotheses", []):
                h = hypothesis_set.add_hypothesis(
                    claim=h_data.get("claim", ""),
                    category=h_data.get("category", "unknown"),
                    current_step=current_step,
                    search_terms=h_data.get("search_terms", []),
                )
                if h:
                    new_hypotheses.append(h)
            
            if self.verbose and new_hypotheses:
                print(f"    [Hypothesis] Generated {len(new_hypotheses)} new hypotheses", file=sys.stderr)
                for h in new_hypotheses:
                    print(f"      - {h.claim[:60]}...", file=sys.stderr)
            
            return new_hypotheses
            
        except Exception as e:
            if self.verbose:
                print(f"    [Hypothesis] Error: {e}", file=sys.stderr)
            return []
    
    def link_evidence_to_hypotheses(
        self,
        hypothesis_set: HypothesisSet,
        evidence_items: List[Dict[str, Any]],
        current_step: int,
    ):
        """
        Link new evidence to relevant hypotheses.
        
        This is how hypotheses get tested - evidence either supports or contradicts them.
        """
        for ev in evidence_items:
            claim = ev.get("claim_supported", "").lower()
            evidence_id = ev.get("evidence_id", "")
            
            if not claim or not evidence_id:
                continue
            
            # Check each hypothesis
            for h in hypothesis_set.hypotheses.values():
                h_claim = h.claim.lower()
                
                # Simple keyword matching for now
                # Could be made smarter with embeddings
                claim_words = set(h_claim.split())
                evidence_words = set(claim.split())
                overlap = claim_words & evidence_words
                
                if len(overlap) >= 2:  # At least 2 words overlap
                    # Check if supporting or contradicting
                    # For now, assume supporting unless explicitly contradicting
                    supports = "not" not in claim and "never" not in claim
                    
                    hypothesis_set.add_evidence(
                        hypothesis_id=h.hypothesis_id,
                        evidence_id=evidence_id,
                        supports=supports,
                        current_step=current_step,
                    )
