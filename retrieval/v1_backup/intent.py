"""
Intent Classification for Agentic Workflow.

Classifies user queries into intent families to guide plan generation.
Returns confidence and uncertainties so the planner can decide to:
- Ask for clarification (UI contexts)
- Default to stricter predicates and evidence requirements

Intent families:
- EXISTENCE_EVIDENCE: "is there info about X..."
- ROSTER_ENUMERATION: "who were members of..."
- RELATIONSHIP_CONSTRAINED: "officers associated with X..."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from retrieval.plan import VerificationSpec
from retrieval.evidence_bundle import RetrievalLaneRun


# =============================================================================
# Intent Families
# =============================================================================

class IntentFamily(str, Enum):
    """
    High-level intent families that determine execution strategy.
    
    This is not brittle "predict every use" - it's a handful of intent families
    that decide coverage strategy.
    """
    EXISTENCE_EVIDENCE = "existence_evidence"      # "is there info about X..."
    ROSTER_ENUMERATION = "roster_enumeration"      # "who were members of..."
    RELATIONSHIP_CONSTRAINED = "relationship_constrained"  # "officers associated with X..."


# =============================================================================
# Query Anchors
# =============================================================================

@dataclass
class QueryAnchors:
    """
    Extracted anchors from the query for plan generation.
    """
    target_entities: List[int] = field(default_factory=list)   # resolved entity_ids
    target_tokens: List[str] = field(default_factory=list)     # unresolved names/codenames
    key_concepts: List[str] = field(default_factory=list)      # "proximity fuse", etc.
    constraints: Dict[str, Any] = field(default_factory=dict)  # collection scope, date range
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_entities": self.target_entities,
            "target_tokens": self.target_tokens,
            "key_concepts": self.key_concepts,
            "constraints": self.constraints,
        }


# =============================================================================
# Intent Classification Result
# =============================================================================

@dataclass
class IntentClassification:
    """
    Result of intent classification.
    
    Returns uncertainties so planner can decide to ask for clarification
    or default to stricter requirements.
    """
    intent_family: IntentFamily
    confidence: float                                 # 0-1
    anchors: QueryAnchors
    uncertainties: List[str] = field(default_factory=list)  # ambiguities found
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent_family": self.intent_family.value,
            "confidence": self.confidence,
            "anchors": self.anchors.to_dict(),
            "uncertainties": self.uncertainties,
        }


# =============================================================================
# Pattern Definitions
# =============================================================================

# Patterns for ROSTER_ENUMERATION
ROSTER_PATTERNS = [
    r"\bwho\s+(?:were|are|was)\b.*\b(?:members?|people|persons?|individuals?)\b",
    r"\blist\s+(?:of\s+)?(?:the\s+)?(?:members?|people|persons?)\b",
    r"\bwhat\s+(?:were|are)\s+the\s+names\b",
    r"\bidentify\s+(?:all\s+)?(?:the\s+)?(?:members?|people)\b",
    r"\b(?:members?|roster|personnel)\s+of\b",
    r"\bbelonged?\s+to\b",
    r"\bpart\s+of\s+(?:the\s+)?(?:network|group|cell|ring)\b",
]

# Patterns for RELATIONSHIP_CONSTRAINED
RELATIONSHIP_PATTERNS = [
    r"\b(?:handlers?|officers?|contacts?|agents?)\s+(?:of|for|associated\s+with)\b",
    r"\b(?:who|which)\s+(?:were|was|are|is)\s+(?:the\s+)?(?:handlers?|officers?)\b",
    r"\bcase\s+officers?\b",
    r"\b(?:handler|officer)\s+(?:for|of)\b",
    r"\bwho\s+(?:handled|ran|controlled|recruited)\b",
    r"\bconnected?\s+(?:to|with)\s+(?:Soviet|Russian|NKVD|MGB|GRU)\b",
    r"\b(?:Soviet|Russian|NKVD|MGB|GRU)\s+(?:intelligence\s+)?officers?\b",
]

# Patterns for EXISTENCE_EVIDENCE
EXISTENCE_PATTERNS = [
    r"\bis\s+there\s+(?:any\s+)?(?:info|information|evidence|mention)\b",
    r"\bwhat\s+(?:do\s+we\s+know|is\s+known|does\s+.+\s+say)\s+about\b",
    r"\b(?:find|search\s+for|look\s+for)\s+(?:info|information|mentions?)\b",
    r"\bmentions?\s+of\b",
    r"\bevidence\s+(?:of|for|about)\b",
    r"\breferences?\s+to\b",
    r"\b(?:did|was|were)\s+.+\s+(?:mention|discuss|report|describe)\b",
]

# Patterns indicating role constraints
ROLE_CONSTRAINT_PATTERNS = [
    (r"\bSoviet\s+(?:intelligence\s+)?officers?\b", ["NKVD", "MGB", "GRU", "intelligence officer", "Soviet officer"]),
    (r"\bNKVD\s+(?:officers?|agents?|operatives?)\b", ["NKVD", "officer", "agent", "operative"]),
    (r"\bMGB\s+(?:officers?|agents?|operatives?)\b", ["MGB", "officer", "agent", "operative"]),
    (r"\bGRU\s+(?:officers?|agents?|operatives?)\b", ["GRU", "officer", "agent", "operative"]),
    (r"\bcase\s+officers?\b", ["case officer", "handler"]),
    (r"\bhandlers?\b", ["handler", "case officer"]),
    (r"\brezidents?\b", ["rezident", "station chief"]),
    (r"\boperatives?\b", ["operative", "agent"]),
]

# Collection scope patterns - only match explicit collection references
# NOT topic/subject mentions like "Silvermaster network"
COLLECTION_PATTERNS = [
    (r"\bin\s+(?:the\s+)?venona\b", ["venona"]),
    (r"\bfrom\s+(?:the\s+)?venona\b", ["venona"]),
    (r"\bvenona\s+(?:cables?|decrypts?|materials?|documents?)\b", ["venona"]),
    (r"\bin\s+(?:the\s+)?vassiliev\b", ["vassiliev"]),
    (r"\bfrom\s+(?:the\s+)?vassiliev\b", ["vassiliev"]),
    (r"\bvassiliev\s+(?:notebooks?|materials?|documents?)\b", ["vassiliev"]),
    (r"\bin\s+(?:the\s+)?huac\s+(?:files?|records?|hearings?)\b", ["huac"]),
    (r"\bfrom\s+(?:the\s+)?huac\b", ["huac"]),
    (r"\bin\s+(?:the\s+)?mccarthy\s+(?:files?|records?|hearings?)\b", ["mccarthy"]),
    (r"\bfrom\s+(?:the\s+)?mccarthy\b", ["mccarthy"]),
    # NOTE: Don't auto-filter by "rosenberg" or "silvermaster" as topic/subject names
    # Only filter when explicitly referencing the collection source
    (r"\brosenberg\s+(?:files?|trial\s+(?:files?|records?))\b", ["rosenberg"]),
    (r"\bsilvermaster\s+(?:files?|fbi\s+files?)\b", ["silvermaster"]),
]

# Uncertainty indicators
UNCERTAINTY_INDICATORS = [
    (r"\b(?:closely|loosely)\s+associated\b", "role relationship unclear"),
    (r"\b(?:possibly|potentially|allegedly)\s+", "confidence qualifier detected"),
    (r"\b(?:might|may|could)\s+have\b", "uncertainty about relationship"),
    (r"\bconnected?\s+(?:to|with)\b", "connection type unspecified"),
    (r"\brelat(?:ed|ion)\s+(?:to|with)\b", "relationship type unspecified"),
]


# =============================================================================
# Classification Functions
# =============================================================================

def classify_intent(
    utterance: str,
    resolved_entities: List[Dict[str, Any]] = None,
) -> IntentClassification:
    """
    Pattern-based intent classification.
    
    Args:
        utterance: The user's query text
        resolved_entities: Already-resolved entities from pre-processing
        
    Returns:
        IntentClassification with intent family, confidence, anchors, uncertainties
    """
    resolved_entities = resolved_entities or []
    utterance_lower = utterance.lower()
    
    # Extract anchors
    anchors = extract_anchors(utterance, resolved_entities)
    
    # Detect uncertainties
    uncertainties = detect_uncertainties(utterance)
    
    # Score each intent family
    roster_score = score_patterns(utterance_lower, ROSTER_PATTERNS)
    relationship_score = score_patterns(utterance_lower, RELATIONSHIP_PATTERNS)
    existence_score = score_patterns(utterance_lower, EXISTENCE_PATTERNS)
    
    # Determine intent family
    max_score = max(roster_score, relationship_score, existence_score)
    
    if max_score == 0:
        # Default to existence if no patterns match
        return IntentClassification(
            intent_family=IntentFamily.EXISTENCE_EVIDENCE,
            confidence=0.5,
            anchors=anchors,
            uncertainties=uncertainties + ["no clear intent pattern detected"],
        )
    
    if roster_score == max_score:
        intent = IntentFamily.ROSTER_ENUMERATION
        confidence = min(1.0, roster_score / 2.0)  # Normalize
    elif relationship_score == max_score:
        intent = IntentFamily.RELATIONSHIP_CONSTRAINED
        confidence = min(1.0, relationship_score / 2.0)
    else:
        intent = IntentFamily.EXISTENCE_EVIDENCE
        confidence = min(1.0, existence_score / 2.0)
    
    return IntentClassification(
        intent_family=intent,
        confidence=confidence,
        anchors=anchors,
        uncertainties=uncertainties,
    )


def score_patterns(text: str, patterns: List[str]) -> float:
    """Score text against a list of patterns."""
    score = 0.0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1.0
    return score


def extract_anchors(
    utterance: str,
    resolved_entities: List[Dict[str, Any]],
) -> QueryAnchors:
    """
    Extract anchors from the query.
    
    Args:
        utterance: The user's query text
        resolved_entities: Already-resolved entities
        
    Returns:
        QueryAnchors with extracted targets and constraints
    """
    anchors = QueryAnchors()
    
    # Extract resolved entity IDs
    for entity in resolved_entities:
        if entity.get("entity_id"):
            anchors.target_entities.append(entity["entity_id"])
        elif entity.get("surface"):
            anchors.target_tokens.append(entity["surface"])
    
    # Extract potential unresolved tokens (capitalized words, quoted strings)
    # Look for ALL CAPS tokens that might be codenames
    codename_pattern = r'\b([A-Z]{3,})\b'
    for match in re.finditer(codename_pattern, utterance):
        token = match.group(1)
        if token not in anchors.target_tokens:
            anchors.target_tokens.append(token)
    
    # Extract Title Case names/concepts (e.g., "Silvermaster", "Rosenberg", "Venona")
    # These are likely proper nouns that are key search terms
    titlecase_pattern = r'\b([A-Z][a-z]{2,})\b'
    stopwords = {'Who', 'What', 'Were', 'Was', 'Are', 'The', 'Members', 'People', 'Persons', 
                 'List', 'Find', 'Search', 'Evidence', 'Information', 'About', 'From', 'With'}
    for match in re.finditer(titlecase_pattern, utterance):
        token = match.group(1)
        if token not in stopwords and token not in anchors.target_tokens:
            anchors.target_tokens.append(token)
    
    # Extract quoted strings as potential concepts
    quoted_pattern = r'"([^"]+)"'
    for match in re.finditer(quoted_pattern, utterance):
        concept = match.group(1)
        if concept not in anchors.key_concepts:
            anchors.key_concepts.append(concept)
    
    # Extract key concepts from known patterns
    concept_patterns = [
        r"proximity\s+fuse",
        r"atomic\s+(?:secrets?|bomb|weapons?)",
        r"manhattan\s+project",
        r"nuclear\s+(?:secrets?|weapons?|program)",
    ]
    utterance_lower = utterance.lower()
    for pattern in concept_patterns:
        if re.search(pattern, utterance_lower):
            match = re.search(pattern, utterance_lower)
            if match and match.group(0) not in anchors.key_concepts:
                anchors.key_concepts.append(match.group(0))
    
    # Extract noun phrases after "of the X network/group/cell/ring"
    network_pattern = r'(?:of\s+the\s+)?(\w+)\s+(?:network|group|cell|ring|spy\s+ring)'
    for match in re.finditer(network_pattern, utterance_lower):
        concept = match.group(1)
        # Don't add stopwords
        if concept.lower() not in {'the', 'a', 'an', 'this', 'that'}:
            if concept not in anchors.key_concepts:
                anchors.key_concepts.append(concept)
    
    # Also extract the full network phrase (e.g., "silvermaster network")
    full_network_pattern = r'(\w+\s+(?:network|group|cell|ring|spy\s+ring))'
    for match in re.finditer(full_network_pattern, utterance_lower):
        concept = match.group(1)
        if concept not in anchors.key_concepts:
            anchors.key_concepts.append(concept)
    
    # Extract collection scope constraints
    anchors.constraints["collection_scope"] = extract_collection_scope(utterance)
    
    # Extract role evidence patterns
    anchors.constraints["role_evidence_patterns"] = extract_role_patterns(utterance)
    
    return anchors


def extract_collection_scope(utterance: str) -> List[str]:
    """Extract collection scope from query."""
    collections = []
    utterance_lower = utterance.lower()
    
    for pattern, scope in COLLECTION_PATTERNS:
        if re.search(pattern, utterance_lower):
            collections.extend(scope)
    
    return list(set(collections))


def extract_role_patterns(utterance: str) -> List[str]:
    """Extract role evidence patterns from query."""
    patterns = []
    utterance_lower = utterance.lower()
    
    for pattern, role_terms in ROLE_CONSTRAINT_PATTERNS:
        if re.search(pattern, utterance_lower):
            patterns.extend(role_terms)
    
    return list(set(patterns))


def detect_uncertainties(utterance: str) -> List[str]:
    """Detect uncertainty indicators in the query."""
    uncertainties = []
    utterance_lower = utterance.lower()
    
    for pattern, uncertainty in UNCERTAINTY_INDICATORS:
        if re.search(pattern, utterance_lower):
            uncertainties.append(uncertainty)
    
    return uncertainties


# =============================================================================
# Coverage Calculation
# =============================================================================

def compute_coverage(
    lane_run: RetrievalLaneRun,
    intent: IntentFamily,
    verification: VerificationSpec,
    claim_count: int = 0,
) -> float:
    """
    Compute coverage achieved by a lane run.
    
    Coverage must be computed deterministically so the verifier isn't "mushy".
    
    Args:
        lane_run: The retrieval lane run
        intent: The intent family
        verification: Verification spec with thresholds
        claim_count: Number of relation claims (for RELATIONSHIP_CONSTRAINED)
        
    Returns:
        Coverage as float 0-1
    """
    if intent == IntentFamily.EXISTENCE_EVIDENCE:
        # coverage = min(1.0, unique_pages / target_pages)
        target_pages = verification.target_pages or 20
        return min(1.0, lane_run.unique_pages / target_pages)
    
    elif intent == IntentFamily.ROSTER_ENUMERATION:
        # coverage = min(1.0, doc_count / min_doc_diversity)
        min_docs = verification.min_doc_diversity or 5
        return min(1.0, lane_run.doc_count / min_docs)
    
    elif intent == IntentFamily.RELATIONSHIP_CONSTRAINED:
        # Coverage tied to claim success, not raw hits
        min_docs = verification.min_doc_diversity or 2
        min_claims = verification.min_relation_claims or 1
        
        doc_coverage = lane_run.doc_count / max(min_docs, 1)
        claim_coverage = claim_count / max(min_claims, 1)
        
        return min(1.0, min(doc_coverage, claim_coverage))
    
    # Default
    return min(1.0, lane_run.doc_count / 3)


def check_coverage_requirements(
    intent: IntentFamily,
    retrieval_runs: List[RetrievalLaneRun],
    verification: VerificationSpec,
) -> Tuple[bool, List[str]]:
    """
    Check if coverage requirements are met for an intent.
    
    Args:
        intent: The intent family
        retrieval_runs: All retrieval lane runs
        verification: Verification spec with requirements
        
    Returns:
        (met, errors) tuple
    """
    errors = []
    lanes_run = {run.lane_id for run in retrieval_runs}
    
    # Check required lanes ran
    for required_lane in verification.required_lanes:
        if required_lane not in lanes_run:
            errors.append(f"Required lane '{required_lane}' did not run")
    
    # Check minimum hits per lane
    for run in retrieval_runs:
        if run.hit_count < verification.min_hits_per_lane:
            # This is a warning, not necessarily an error
            pass
    
    # Check overall doc diversity
    all_doc_ids = set()
    for run in retrieval_runs:
        # Approximate - would need to track actual doc_ids
        all_doc_ids.update(range(run.doc_count))  # Placeholder
    
    if len(all_doc_ids) < verification.min_doc_diversity:
        errors.append(
            f"Insufficient doc diversity: {len(all_doc_ids)} < {verification.min_doc_diversity}"
        )
    
    return len(errors) == 0, errors


# =============================================================================
# Intent-Specific Configuration
# =============================================================================

# Coverage requirements per intent family
COVERAGE_REQUIREMENTS = {
    IntentFamily.EXISTENCE_EVIDENCE: {
        "required_lanes": ["lexical_must_hit", "ephemeral_expansion"],
        "min_doc_diversity": 3,
    },
    IntentFamily.RELATIONSHIP_CONSTRAINED: {
        "required_lanes": ["entity_codename", "hybrid"],
        "codename_resolution_required": True,
        "min_doc_diversity": 2,
    },
    IntentFamily.ROSTER_ENUMERATION: {
        "required_lanes": ["entity_codename", "hybrid"],
        "expansion_loop_required": True,
        "min_doc_diversity": 5,
    },
}


def get_coverage_requirements(intent: IntentFamily) -> Dict[str, Any]:
    """Get coverage requirements for an intent family."""
    return COVERAGE_REQUIREMENTS.get(intent, {
        "required_lanes": ["hybrid"],
        "min_doc_diversity": 3,
    })


# =============================================================================
# Query Entity Linking (Pre-Retrieval)
# =============================================================================

@dataclass
class LinkedEntity:
    """A candidate entity linked from query text."""
    entity_id: int
    canonical_name: str
    matched_surface: str           # What in the query matched
    score: float                   # 0-1 confidence
    match_type: str                # "exact", "alias", "fuzzy"
    entity_type: Optional[str] = None  # "person", "organization", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "matched_surface": self.matched_surface,
            "score": self.score,
            "match_type": self.match_type,
            "entity_type": self.entity_type,
        }


@dataclass
class EntityLinkingResult:
    """Result of query entity linking."""
    linked_entities: List[LinkedEntity]
    unlinked_tokens: List[str]     # Tokens that didn't match any entity
    
    @property
    def entity_ids(self) -> List[int]:
        """Get entity IDs above threshold."""
        return [e.entity_id for e in self.linked_entities if e.score >= 0.5]
    
    @property
    def high_confidence_ids(self) -> List[int]:
        """Get only high-confidence entity IDs (score >= 0.7)."""
        return [e.entity_id for e in self.linked_entities if e.score >= 0.7]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "linked_entities": [e.to_dict() for e in self.linked_entities],
            "unlinked_tokens": self.unlinked_tokens,
        }


# Tokens to skip during entity linking (too generic)
LINKING_STOPWORDS = {
    'the', 'a', 'an', 'of', 'in', 'to', 'for', 'and', 'or', 'is', 'was', 'were',
    'who', 'what', 'which', 'where', 'when', 'how', 'why',
    'members', 'member', 'network', 'group', 'cell', 'ring', 'spy',
    'soviet', 'soviets', 'russian', 'american', 'intelligence',
    'handler', 'handlers', 'officer', 'officers', 'agent', 'agents',
    'evidence', 'information', 'about', 'find', 'search', 'list',
}


def extract_linkable_tokens(query: str) -> List[str]:
    """
    Extract tokens from query that are worth trying to link.
    
    Focuses on proper-noun-ish tokens:
    - Title case words (Silvermaster, Rosenberg)
    - ALL CAPS (NKVD, FBI)
    - Multi-word phrases in quotes
    
    Skips generic stopwords.
    """
    tokens = []
    
    # Extract quoted phrases first
    quoted = re.findall(r'"([^"]+)"', query)
    tokens.extend(quoted)
    
    # Remove quotes from query for further processing
    query_no_quotes = re.sub(r'"[^"]+"', '', query)
    
    # Extract Title Case words (likely proper nouns)
    title_case = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b', query_no_quotes)
    for tc in title_case:
        if tc.lower() not in LINKING_STOPWORDS:
            tokens.append(tc)
    
    # Extract ALL CAPS words (acronyms, codenames)
    all_caps = re.findall(r'\b([A-Z]{2,})\b', query_no_quotes)
    for ac in all_caps:
        if ac.lower() not in LINKING_STOPWORDS:
            tokens.append(ac)
    
    # Dedupe while preserving order
    seen = set()
    unique = []
    for t in tokens:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)
    
    return unique


def link_query_entities(
    query: str,
    conn,
    collection_scope: List[str] = None,
    score_threshold: float = 0.4,
    max_candidates: int = 10,
) -> EntityLinkingResult:
    """
    Pre-retrieval entity linking: find candidate entity_ids from query text.
    
    Uses fast lexical lookup (not LLM) against:
    - entities.canonical_name (exact and fuzzy)
    - entity_aliases.surface (exact match)
    
    Only links proper-noun-ish tokens to avoid over-linking generic terms.
    
    Args:
        query: User query text
        conn: Database connection
        collection_scope: Optional collection filter
        score_threshold: Minimum score to include (default 0.4)
        max_candidates: Maximum candidates to return
        
    Returns:
        EntityLinkingResult with linked entities and unlinked tokens
    """
    linked = []
    unlinked = []
    
    # Extract tokens worth linking
    tokens = extract_linkable_tokens(query)
    
    if not tokens:
        return EntityLinkingResult(linked_entities=[], unlinked_tokens=[])
    
    with conn.cursor() as cur:
        for token in tokens:
            candidates = []
            token_lower = token.lower()
            
            # 1. Exact match on canonical_name (highest confidence)
            cur.execute("""
                SELECT id, canonical_name, entity_type
                FROM entities
                WHERE LOWER(canonical_name) = %s
                LIMIT 5
            """, (token_lower,))
            
            for row in cur.fetchall():
                candidates.append(LinkedEntity(
                    entity_id=row[0],
                    canonical_name=row[1],
                    matched_surface=token,
                    score=1.0,  # Exact match
                    match_type="exact",
                    entity_type=row[2],
                ))
            
            # 2. Check entity_aliases for exact alias match
            cur.execute("""
                SELECT ea.entity_id, e.canonical_name, ea.alias, e.entity_type
                FROM entity_aliases ea
                JOIN entities e ON e.id = ea.entity_id
                WHERE ea.alias_norm = %s
                LIMIT 5
            """, (token_lower,))
            
            for row in cur.fetchall():
                # Avoid duplicates
                if not any(c.entity_id == row[0] for c in candidates):
                    candidates.append(LinkedEntity(
                        entity_id=row[0],
                        canonical_name=row[1],
                        matched_surface=token,
                        score=0.95,  # Alias match is very good
                        match_type="alias",
                        entity_type=row[3],
                    ))
            
            # 3. Prefix/contains match on canonical_name (lower confidence)
            if not candidates:
                # Try prefix match
                cur.execute("""
                    SELECT id, canonical_name, entity_type,
                           similarity(LOWER(canonical_name), %s) as sim
                    FROM entities
                    WHERE LOWER(canonical_name) LIKE %s
                       OR LOWER(canonical_name) LIKE %s
                    ORDER BY sim DESC
                    LIMIT 5
                """, (token_lower, f"{token_lower}%", f"% {token_lower}%"))
                
                for row in cur.fetchall():
                    sim = row[3] if row[3] else 0.5
                    candidates.append(LinkedEntity(
                        entity_id=row[0],
                        canonical_name=row[1],
                        matched_surface=token,
                        score=min(0.8, 0.5 + sim * 0.3),  # Cap at 0.8 for partial
                        match_type="fuzzy",
                        entity_type=row[2],
                    ))
            
            # 4. Trigram similarity search (if still no candidates and token is long enough)
            if not candidates and len(token) >= 4:
                try:
                    cur.execute("""
                        SELECT id, canonical_name, entity_type,
                               similarity(LOWER(canonical_name), %s) as sim
                        FROM entities
                        WHERE similarity(LOWER(canonical_name), %s) > 0.3
                        ORDER BY sim DESC
                        LIMIT 3
                    """, (token_lower, token_lower))
                    
                    for row in cur.fetchall():
                        sim = row[3]
                        if sim >= 0.4:  # Only include decent matches
                            candidates.append(LinkedEntity(
                                entity_id=row[0],
                                canonical_name=row[1],
                                matched_surface=token,
                                score=sim * 0.9,  # Scale similarity
                                match_type="fuzzy",
                                entity_type=row[2],
                            ))
                except Exception:
                    # pg_trgm might not be available
                    pass
            
            # Add best candidate(s) above threshold
            if candidates:
                # Sort by score descending
                candidates.sort(key=lambda c: -c.score)
                for cand in candidates[:2]:  # Take top 2 per token
                    if cand.score >= score_threshold:
                        # Avoid duplicates across tokens
                        if not any(l.entity_id == cand.entity_id for l in linked):
                            linked.append(cand)
            else:
                unlinked.append(token)
    
    # Sort by score and cap
    linked.sort(key=lambda e: -e.score)
    linked = linked[:max_candidates]
    
    return EntityLinkingResult(
        linked_entities=linked,
        unlinked_tokens=unlinked,
    )
