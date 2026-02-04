"""
Constraint Scoring for Agentic V2.

Pluggable constraint scorers that compute scores from FocusSpans
and return supporting spans per score.

Constraints are soft-scored, but enforced via evidence at render/verify time.
This gives robustness + flexibility without letting the model "make stuff up."

Contracts:
- C4: constraint_key = f"{name}:{object}"
- C5: RelationshipConstraint has direct (0.5) + neighborhood (0.25) modes
- C6: RoleConstraint scores direct + neighborhood role terms
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Protocol, TYPE_CHECKING
from abc import abstractmethod

if TYPE_CHECKING:
    from retrieval.focus_bundle import FocusBundle


@dataclass
class ConstraintSupport:
    """Result of evaluating a constraint for a candidate."""
    constraint_name: str           # Contract C4: f"{name}:{object}"
    score: float                   # 0.0 - 1.0
    supporting_span_ids: List[str] # spans that justify this score
    feature_trace: Dict            # audit: which signals fired


@dataclass
class CandidateAssessment:
    """Full assessment of a candidate across all constraints."""
    candidate_entity_id: Optional[int]
    candidate_key: str              # stable ID: f"entity:{id}" or f"token:{surface}"
    display_name: str
    supports: List[ConstraintSupport]
    final_score: float              # aggregated score


class ConstraintScorer(Protocol):
    """Protocol for pluggable constraint scorers."""
    
    @property
    def constraint_key(self) -> str:
        """Standard key: f"{name}:{object}"."""
        ...
    
    @abstractmethod
    def score(
        self,
        candidate_key: str,
        candidate_entity_id: Optional[int],
        focus_bundle: "FocusBundle",
        conn,
    ) -> ConstraintSupport:
        """Score a candidate against this constraint."""
        ...


class AffiliationConstraint:
    """
    Score affiliation to an organization (OSS, State Dept, etc.).
    
    Scoring signals:
    - Lexical frames: "worked at", "employed by", "member of" + org
    - Mention proximity: candidate mention in same span as org mention
    - Cross-doc corroboration: multiple docs mention both
    """
    
    # Known organization aliases
    ORG_ALIASES = {
        "OSS": ["OSS", "Office of Strategic Services", "O.S.S."],
        "State Department": ["State Department", "State Dept", "DOS", "Department of State"],
        "Treasury": ["Treasury", "Treasury Department", "Dept of Treasury"],
        "FBI": ["FBI", "Federal Bureau of Investigation"],
        "NKVD": ["NKVD", "People's Commissariat for Internal Affairs"],
        "KGB": ["KGB", "Committee for State Security"],
        "GRU": ["GRU", "Main Intelligence Directorate"],
    }
    
    # Affiliation frame patterns
    AFFILIATION_FRAMES = [
        r'\b(?:worked|employed|working)\b.*\b(?:at|for|with)\b',
        r'\b(?:member|officer|agent|employee|staff)\b.*\b(?:of|at|in)\b',
        r'\bjoin(?:ed)?\b',
        r'\b(?:recruited|assigned|transferred)\b.*\bto\b',
        r'\b(?:position|role|job)\b.*\b(?:at|in|with)\b',
    ]
    
    def __init__(self, org: str, aliases: List[str] = None):
        self.org = org
        self.aliases = aliases or self.ORG_ALIASES.get(org, [org])
        if org not in self.aliases:
            self.aliases = [org] + self.aliases
    
    @property
    def constraint_key(self) -> str:
        return f"affiliation:{self.org}"
    
    def score(
        self,
        candidate_key: str,
        candidate_entity_id: Optional[int],
        focus_bundle: "FocusBundle",
        conn,
    ) -> ConstraintSupport:
        """Score affiliation via lexical frames, proximity, cross-doc."""
        signals = {"lexical_frame": 0, "proximity": 0, "cross_doc": 0}
        supporting = []
        
        # Get candidate spans
        if candidate_entity_id:
            candidate_spans = focus_bundle.get_spans_for_entity(candidate_entity_id, conn)
        else:
            candidate_spans = focus_bundle.get_spans_for_term(candidate_key.split(":")[-1])
        
        for span in candidate_spans:
            # Check if org is present in span
            org_present = any(
                _normalize(alias) in _normalize(span.text)
                for alias in self.aliases
            )
            
            if not org_present:
                continue
            
            supporting.append(span.span_id)
            signals["proximity"] += 1
            
            # Check for affiliation frames
            text_lower = span.text.lower()
            for frame in self.AFFILIATION_FRAMES:
                if re.search(frame, text_lower):
                    signals["lexical_frame"] += 1
                    break
        
        # Cross-doc corroboration
        if supporting:
            doc_ids = {
                focus_bundle.get_span(sid).doc_id 
                for sid in supporting 
                if focus_bundle.get_span(sid)
            }
            if len(doc_ids) > 1:
                signals["cross_doc"] = len(doc_ids)
        
        # Compute score (weighted signals)
        score = min(1.0, (
            0.3 * min(signals["lexical_frame"], 2) +
            0.3 * min(signals["proximity"], 3) +
            0.2 * min(signals["cross_doc"], 2)
        ) / 0.8)
        
        return ConstraintSupport(
            constraint_name=self.constraint_key,
            score=score,
            supporting_span_ids=supporting[:3],  # cap citations
            feature_trace=signals,
        )


class RelationshipConstraint:
    """
    Score relationship to a target entity.
    
    Contract C5: Two evidence modes with different weights:
    - Direct co-mention (same span): strong (0.5)
    - Neighborhood linkage (same doc/page): medium (0.25)
    
    Also checks for relationship frames: "handler", "recruited by", "worked with"
    """
    
    RELATIONSHIP_FRAMES = [
        r'\b(?:handler|handled)\b',
        r'\b(?:recruited|recruiter|recruit)\b',
        r'\b(?:contact|contacted)\b',
        r'\b(?:source|asset)\b',
        r'\b(?:worked with|working with)\b',
        r'\b(?:associated with|connected to|linked to)\b',
        r'\b(?:met with|meeting with)\b',
        r'\b(?:reported to|reports to)\b',
        r'\b(?:received from|sent to)\b',
        r'\b(?:introduced|introduction)\b',
    ]
    
    NETWORK_PATTERNS = [
        r'\b(?:network|group|ring|apparatus|cell)\b',
        r'\b(?:organization|org)\b',
        r'\b(?:circle|clique)\b',
    ]
    
    def __init__(self, target_entity_id: int, target_name: str, relation_type: str = "associated"):
        self.target_entity_id = target_entity_id
        self.target_name = target_name
        self.relation_type = relation_type
    
    @property
    def constraint_key(self) -> str:
        return f"relationship:{self.target_name}"
    
    def score(
        self,
        candidate_key: str,
        candidate_entity_id: Optional[int],
        focus_bundle: "FocusBundle",
        conn,
    ) -> ConstraintSupport:
        """
        Score relationship with two modes (Contract C5).
        """
        signals = {
            "direct_co_mention": 0,      # same span: strong
            "neighborhood_linkage": 0,   # same doc/page: medium
            "relationship_frame": 0,
            "network": 0,
        }
        supporting = []
        
        # Get candidate spans
        if candidate_entity_id:
            candidate_spans = focus_bundle.get_spans_for_entity(candidate_entity_id, conn)
        else:
            candidate_spans = focus_bundle.get_spans_for_term(candidate_key.split(":")[-1])
        
        # Get target spans
        target_spans = focus_bundle.get_spans_for_entity(self.target_entity_id, conn)
        target_span_ids = {s.span_id for s in target_spans}
        target_neighborhoods = {(s.doc_id, s.page_ref) for s in target_spans}
        
        for span in candidate_spans:
            # Mode 1: Direct co-mention (same span) - strong signal
            if span.span_id in target_span_ids:
                supporting.append(span.span_id)
                signals["direct_co_mention"] += 1
                
                text_lower = span.text.lower()
                
                # Check relationship frames
                for frame in self.RELATIONSHIP_FRAMES:
                    if re.search(frame, text_lower):
                        signals["relationship_frame"] += 1
                        break
                
                # Check network patterns
                for pattern in self.NETWORK_PATTERNS:
                    if re.search(pattern, text_lower):
                        signals["network"] += 1
                        break
            
            # Mode 2: Neighborhood linkage (same doc/page) - medium signal
            elif (span.doc_id, span.page_ref) in target_neighborhoods:
                supporting.append(span.span_id)
                signals["neighborhood_linkage"] += 1
        
        # Weighted score (Contract C5)
        score = min(1.0, (
            0.5 * min(signals["direct_co_mention"], 2) +
            0.25 * min(signals["neighborhood_linkage"], 3) +
            0.15 * min(signals["relationship_frame"], 2) +
            0.05 * min(signals["network"], 2)
        ))
        
        return ConstraintSupport(
            constraint_name=self.constraint_key,
            score=score,
            supporting_span_ids=supporting[:3],
            feature_trace=signals,
        )


class RoleConstraint:
    """
    Score role (officer, agent, source, etc.).
    
    Contract C6: Scores based on role terms in candidate spans 
    OR doc/page neighborhood around candidate spans.
    """
    
    ROLE_PATTERNS = {
        "officer": [
            r'\bofficer\b', 
            r'\boperative\b', 
            r'\bcase officer\b', 
            r'\bintelligence officer\b',
            r'\bcontrolling officer\b',
        ],
        "agent": [
            r'\bagent\b', 
            r'\bspy\b', 
            r'\binformant\b', 
            r'\bsource\b', 
            r'\basset\b',
            r'\binformer\b',
        ],
        "Soviet intelligence": [
            r'\bSoviet\b.*\b(?:agent|spy|intelligence|source)\b',
            r'\bNKVD\b.*\b(?:agent|source|asset)\b',
            r'\bKGB\b.*\b(?:agent|source|asset)\b',
            r'\bGRU\b.*\b(?:agent|source|asset)\b',
            r'\bRussian\b.*\b(?:agent|spy|intelligence)\b',
            r'\bMoscow\b.*\b(?:agent|source)\b',
        ],
        "handler": [
            r'\bhandler\b',
            r'\bcontroller\b',
            r'\bcase officer\b',
            r'\bcontrolling officer\b',
        ],
        "courier": [
            r'\bcourier\b',
            r'\bmessenger\b',
            r'\bcut-out\b',
            r'\bcutout\b',
        ],
    }
    
    def __init__(self, role: str):
        self.role = role
        self.patterns = self.ROLE_PATTERNS.get(role, [rf'\b{re.escape(role)}\b'])
    
    @property
    def constraint_key(self) -> str:
        return f"role:{self.role}"
    
    def score(
        self,
        candidate_key: str,
        candidate_entity_id: Optional[int],
        focus_bundle: "FocusBundle",
        conn,
    ) -> ConstraintSupport:
        """
        Score role with direct and neighborhood modes (Contract C6).
        """
        signals = {"direct_role": 0, "neighborhood_role": 0}
        supporting = []
        
        # Get candidate spans
        if candidate_entity_id:
            candidate_spans = focus_bundle.get_spans_for_entity(candidate_entity_id, conn)
        else:
            candidate_spans = focus_bundle.get_spans_for_term(candidate_key.split(":")[-1])
        
        candidate_span_ids = {s.span_id for s in candidate_spans}
        candidate_neighborhoods = {(s.doc_id, s.page_ref) for s in candidate_spans}
        
        # Mode 1: Role term in candidate's own spans
        for span in candidate_spans:
            for pattern in self.patterns:
                if re.search(pattern, span.text, re.I):
                    supporting.append(span.span_id)
                    signals["direct_role"] += 1
                    break
        
        # Mode 2: Role term in neighborhood spans (same doc/page)
        for span in focus_bundle.spans:
            if span.span_id not in candidate_span_ids:
                if (span.doc_id, span.page_ref) in candidate_neighborhoods:
                    for pattern in self.patterns:
                        if re.search(pattern, span.text, re.I):
                            if span.span_id not in supporting:
                                supporting.append(span.span_id)
                            signals["neighborhood_role"] += 1
                            break
        
        # Weighted score
        score = min(1.0, (
            0.5 * min(signals["direct_role"], 2) +
            0.2 * min(signals["neighborhood_role"], 3)
        ))
        
        return ConstraintSupport(
            constraint_name=self.constraint_key,
            score=score,
            supporting_span_ids=supporting[:3],
            feature_trace=signals,
        )


def _normalize(text: str) -> str:
    """Normalize text for matching."""
    return ' '.join(text.lower().split())


# Constraint scorer registry
CONSTRAINT_REGISTRY = {
    "affiliation": AffiliationConstraint,
    "relationship": RelationshipConstraint,
    "role": RoleConstraint,
}


def build_constraint_scorers(
    constraints: List["ConstraintSpec"],
    target_entity_map: Dict[str, int] = None,
) -> List[ConstraintScorer]:
    """
    Build constraint scorers from ConstraintSpec list.
    
    Args:
        constraints: List of ConstraintSpec from QueryContract
        target_entity_map: Map from target name to entity_id (for relationships)
    
    Returns:
        List of ConstraintScorer instances
    """
    from retrieval.query_intent import ConstraintSpec
    
    target_entity_map = target_entity_map or {}
    scorers = []
    
    for spec in constraints:
        if spec.name == "affiliation":
            scorers.append(AffiliationConstraint(spec.object))
        elif spec.name == "relationship":
            # Need target entity_id for relationship
            target_id = target_entity_map.get(spec.object)
            if target_id:
                scorers.append(RelationshipConstraint(target_id, spec.object))
        elif spec.name == "role":
            scorers.append(RoleConstraint(spec.object))
    
    return scorers


def assess_candidate(
    candidate_key: str,
    candidate_entity_id: Optional[int],
    display_name: str,
    scorers: List[ConstraintScorer],
    focus_bundle: "FocusBundle",
    conn,
) -> CandidateAssessment:
    """
    Assess a candidate against all constraints.
    
    Returns CandidateAssessment with all constraint supports and final score.
    """
    supports = []
    
    for scorer in scorers:
        support = scorer.score(candidate_key, candidate_entity_id, focus_bundle, conn)
        supports.append(support)
    
    # Aggregate score: average of constraint scores
    if supports:
        final_score = sum(s.score for s in supports) / len(supports)
    else:
        final_score = 0.0
    
    return CandidateAssessment(
        candidate_entity_id=candidate_entity_id,
        candidate_key=candidate_key,
        display_name=display_name,
        supports=supports,
        final_score=final_score,
    )
