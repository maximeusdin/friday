"""
Claim Extraction for Agentic Workflow.

Extracts claims from chunks using deterministic patterns first,
with optional LLM refinement for ambiguous cases.

Strategy: Deterministic First, LLM Assist Second
- Deterministic extraction (always runs first):
  - Codename mappings → pattern-based
  - Must-hit mentions → string span extraction
  - Role evidence → pattern match in span
  - Entity relationships → from entity_mentions + entity_relationships tables
- LLM-assisted (only when needed):
  - Selecting best span boundaries when text is messy
  - Classifying ambiguous support_type
  - Summarizing from already-verified claims at render time

Key constraint: Claims must always include spans and pass verifier,
regardless of extraction method.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from retrieval.plan import ExtractionSpec
from retrieval.evidence_bundle import (
    Claim,
    EvidenceRef,
    EntityCandidate,
    ChunkWithProvenance,
    Predicate,
    SupportType,
)
from retrieval.intent import IntentFamily


# =============================================================================
# Pattern Definitions
# =============================================================================

# Patterns for support type classification
DEFINITION_PATTERNS = [
    r"(\w+)\s+is\s+a\s+",
    r"(\w+)\s+refers\s+to\s+",
    r"(\w+)\s+means\s+",
    r"(\w+)\s+,\s+known\s+as\s+",
    r"defined\s+as\s+",
]

EXPLICIT_STATEMENT_PATTERNS = [
    r"was\s+described\s+as\s+",
    r"according\s+to\s+",
    r"states?\s+that\s+",
    r"reported\s+that\s+",
    r"indicated\s+that\s+",
    r"confirmed\s+that\s+",
    r"revealed\s+that\s+",
    r"identified\s+as\s+",
]

ASSESSMENT_PATTERNS = [
    r"assessed\s+(?:as|to\s+be)\s+",
    r"evaluated\s+as\s+",
    r"considered\s+(?:to\s+be\s+)?",
    r"believed\s+to\s+be\s+",
    r"thought\s+to\s+be\s+",
    r"suspected\s+(?:of|to\s+be)\s+",
]

# Role evidence patterns (deterministic)
ROLE_PATTERNS = [
    r"case\s+officer",
    r"handler",
    r"rezident",
    r"station\s+chief",
    r"NKVD",
    r"MGB",
    r"GRU",
    r"intelligence\s+officer",
    r"agent",
    r"operative",
    r"contact",
    r"source",
    r"courier",
    r"recruiter",
]

# Handler relationship patterns
HANDLER_PATTERNS = [
    r"handled\s+by\s+",
    r"handler\s+(?:was|is)\s+",
    r"case\s+officer\s+(?:was|is)\s+",
    r"(?:was|is)\s+the\s+handler\s+(?:of|for)\s+",
    r"ran\s+(?:the\s+)?agent",
    r"controlled\s+by\s+",
    r"recruited\s+by\s+",
]

# Meeting patterns
MEETING_PATTERNS = [
    r"met\s+with\s+",
    r"meeting\s+with\s+",
    r"rendezvous\s+with\s+",
    r"contact(?:ed)?\s+",
    r"liaison\s+with\s+",
]

# Membership patterns (for roster queries - checked at render time)
MEMBERSHIP_PATTERNS = [
    r"member\s+of\s+",
    r"part\s+of\s+",
    r"belonged\s+to\s+",
    r"joined\s+",
    r"affiliated\s+with\s+",
    r"associated\s+with\s+",
]


# =============================================================================
# Support Type Classification
# =============================================================================

def classify_support_type(quote_span: str, subject: str = "") -> SupportType:
    """
    Deterministic classification of support type based on patterns.
    
    Args:
        quote_span: The quoted text
        subject: The subject of the claim (for pattern matching)
        
    Returns:
        SupportType classification
    """
    span_lower = quote_span.lower()
    
    # Check definition patterns
    for pattern in DEFINITION_PATTERNS:
        if subject:
            # Replace placeholder with subject
            full_pattern = pattern.replace(r"(\w+)", re.escape(subject))
            if re.search(full_pattern, span_lower, re.IGNORECASE):
                return SupportType.DEFINITION
        elif re.search(pattern, span_lower, re.IGNORECASE):
            return SupportType.DEFINITION
    
    # Check explicit statement patterns
    for pattern in EXPLICIT_STATEMENT_PATTERNS:
        if re.search(pattern, span_lower, re.IGNORECASE):
            return SupportType.EXPLICIT_STATEMENT
    
    # Check assessment patterns
    for pattern in ASSESSMENT_PATTERNS:
        if re.search(pattern, span_lower, re.IGNORECASE):
            return SupportType.ASSESSMENT
    
    # Default to co-mention
    return SupportType.CO_MENTION


def classify_support_strength(
    support_type: SupportType,
    has_role_evidence: bool = False,
) -> int:
    """
    Classify support strength (1-3).
    
    Args:
        support_type: The support type
        has_role_evidence: Whether role evidence was found
        
    Returns:
        Support strength (1=weak, 2=moderate, 3=strong)
    """
    if support_type == SupportType.EXPLICIT_STATEMENT:
        return 3 if has_role_evidence else 2
    elif support_type == SupportType.DEFINITION:
        return 3
    elif support_type == SupportType.ASSESSMENT:
        return 2
    else:  # CO_MENTION
        return 1


# =============================================================================
# Span Extraction
# =============================================================================

def find_entity_span(
    text: str,
    entity_name: str,
    context_chars: int = 100,
) -> Optional[Tuple[int, int, str]]:
    """
    Find the span containing an entity mention with context.
    
    Args:
        text: Full chunk text
        entity_name: Entity name to find
        context_chars: Characters of context on each side
        
    Returns:
        (start, end, quote_span) or None
    """
    # Try exact match first
    pattern = re.escape(entity_name)
    match = re.search(pattern, text, re.IGNORECASE)
    
    if not match:
        # Try partial match (first/last name)
        name_parts = entity_name.split()
        for part in name_parts:
            if len(part) > 2:
                match = re.search(re.escape(part), text, re.IGNORECASE)
                if match:
                    break
    
    if not match:
        return None
    
    # Expand to context
    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)
    
    # Adjust to sentence boundaries if possible
    # Look for sentence start
    for i in range(match.start() - 1, start - 1, -1):
        if i >= 0 and text[i] in '.!?':
            start = i + 1
            break
    
    # Look for sentence end
    for i in range(match.end(), end):
        if i < len(text) and text[i] in '.!?':
            end = i + 1
            break
    
    quote_span = text[start:end].strip()
    return (start, end, quote_span)


def find_cooccurrence_span(
    text: str,
    entity1_name: str,
    entity2_name: str,
    max_distance: int = 500,
) -> Optional[Tuple[int, int, str]]:
    """
    Find a span containing both entities.
    
    Args:
        text: Full chunk text
        entity1_name: First entity name
        entity2_name: Second entity name
        max_distance: Maximum characters between entities
        
    Returns:
        (start, end, quote_span) or None
    """
    # Find both entities
    match1 = re.search(re.escape(entity1_name), text, re.IGNORECASE)
    match2 = re.search(re.escape(entity2_name), text, re.IGNORECASE)
    
    if not match1 or not match2:
        # Try partial matches
        for name in entity1_name.split():
            if len(name) > 2:
                match1 = re.search(re.escape(name), text, re.IGNORECASE)
                if match1:
                    break
        for name in entity2_name.split():
            if len(name) > 2:
                match2 = re.search(re.escape(name), text, re.IGNORECASE)
                if match2:
                    break
    
    if not match1 or not match2:
        return None
    
    # Check distance
    distance = abs(match1.start() - match2.start())
    if distance > max_distance:
        return None
    
    # Build span
    start = min(match1.start(), match2.start())
    end = max(match1.end(), match2.end())
    
    # Add context
    context_start = max(0, start - 50)
    context_end = min(len(text), end + 50)
    
    quote_span = text[context_start:context_end].strip()
    return (context_start, context_end, quote_span)


# =============================================================================
# Deterministic Extraction Functions
# =============================================================================

def extract_mentions_deterministic(
    chunks: List[ChunkWithProvenance],
    candidates: Dict[str, EntityCandidate],
) -> List[Claim]:
    """
    Extract mention claims from chunks.
    
    Creates MENTIONS claims for each entity found in chunks.
    """
    claims = []
    
    for chunk in chunks:
        if not chunk.text:
            continue
        
        for candidate in candidates.values():
            if not candidate.display_name:
                continue
            
            # Find entity in chunk
            result = find_entity_span(chunk.text, candidate.display_name)
            if not result:
                continue
            
            start, end, quote_span = result
            
            # Classify support type
            support_type = classify_support_type(quote_span, candidate.display_name)
            support_strength = classify_support_strength(support_type)
            
            evidence_ref = EvidenceRef(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page_ref="",
                char_start=start,
                char_end=end,
                quote_span=quote_span,
            )
            
            claim = Claim(
                subject=candidate.key,  # The entity being mentioned
                predicate=Predicate.MENTIONS,
                object=f"chunk:{chunk.chunk_id}",  # The chunk where it's mentioned
                evidence=[evidence_ref],
                support_type=support_type,
                support_strength=support_strength,
                source_lane=chunk.source_lanes[0] if chunk.source_lanes else "unknown",
                confidence=0.9,
            )
            claims.append(claim)
    
    return claims


def extract_relations_deterministic(
    chunks: List[ChunkWithProvenance],
    candidates: Dict[str, EntityCandidate],
    intent: IntentFamily,
) -> List[Claim]:
    """
    Extract relationship claims from chunks.
    
    For RELATIONSHIP_CONSTRAINED: extracts handled_by, met_with
    For ROSTER_ENUMERATION: extracts associated_with
    """
    claims = []
    
    # Get pairs of candidates to check
    candidate_list = list(candidates.values())
    
    for chunk in chunks:
        if not chunk.text:
            continue
        
        text_lower = chunk.text.lower()
        
        for i, cand1 in enumerate(candidate_list):
            for cand2 in candidate_list[i+1:]:
                # Find co-occurrence span
                result = find_cooccurrence_span(
                    chunk.text,
                    cand1.display_name,
                    cand2.display_name,
                )
                
                if not result:
                    continue
                
                start, end, quote_span = result
                span_lower = quote_span.lower()
                
                # Check for role evidence in span
                has_role_evidence = any(
                    re.search(p, span_lower, re.IGNORECASE)
                    for p in ROLE_PATTERNS
                )
                
                # Check for handler patterns
                has_handler_pattern = any(
                    re.search(p, span_lower, re.IGNORECASE)
                    for p in HANDLER_PATTERNS
                )
                
                # Check for meeting patterns
                has_meeting_pattern = any(
                    re.search(p, span_lower, re.IGNORECASE)
                    for p in MEETING_PATTERNS
                )
                
                # Determine predicate
                if has_handler_pattern and has_role_evidence:
                    predicate = Predicate.HANDLED_BY
                    support_type = SupportType.EXPLICIT_STATEMENT
                elif has_meeting_pattern:
                    predicate = Predicate.MET_WITH
                    support_type = SupportType.EXPLICIT_STATEMENT
                else:
                    predicate = Predicate.ASSOCIATED_WITH
                    support_type = SupportType.CO_MENTION
                
                support_strength = classify_support_strength(support_type, has_role_evidence)
                
                evidence_ref = EvidenceRef(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page_ref="",
                    char_start=start,
                    char_end=end,
                    quote_span=quote_span,
                )
                
                claim = Claim(
                    subject=cand1.key,
                    predicate=predicate,
                    object=cand2.key,
                    evidence=[evidence_ref],
                    support_type=support_type,
                    support_strength=support_strength,
                    source_lane=chunk.source_lanes[0] if chunk.source_lanes else "unknown",
                    confidence=0.8 if has_role_evidence else 0.5,
                )
                claims.append(claim)
    
    return claims


def extract_codename_mappings_deterministic(
    chunks: List[ChunkWithProvenance],
) -> List[Claim]:
    """
    Extract codename mapping claims from chunks.
    
    Looks for patterns like:
    - "TWAIN" was [Name]
    - [Name] (TWAIN)
    - [Name], codenamed TWAIN
    - TWAIN = [Name]
    """
    claims = []
    
    # Codename patterns
    codename_patterns = [
        # TWAIN was identified as John Smith
        r'([A-Z]{3,})\s+(?:was\s+)?identified\s+as\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        # John Smith (TWAIN)
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(([A-Z]{3,})\)',
        # John Smith, codenamed TWAIN
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:code)?named\s+([A-Z]{3,})',
        # TWAIN = John Smith
        r'([A-Z]{3,})\s*=\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        # TWAIN (John Smith)
        r'([A-Z]{3,})\s*\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\)',
    ]
    
    for chunk in chunks:
        if not chunk.text:
            continue
        
        for pattern in codename_patterns:
            for match in re.finditer(pattern, chunk.text):
                groups = match.groups()
                if len(groups) < 2:
                    continue
                
                # Determine which is codename and which is real name
                g1, g2 = groups[0], groups[1]
                if g1.isupper():
                    codename, real_name = g1, g2
                else:
                    codename, real_name = g2, g1
                
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(chunk.text), match.end() + 50)
                quote_span = chunk.text[start:end].strip()
                
                evidence_ref = EvidenceRef(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page_ref="",
                    char_start=match.start(),
                    char_end=match.end(),
                    quote_span=quote_span,
                )
                
                # Create IDENTIFIED_AS claim
                claim = Claim(
                    subject=f"token:{codename}",
                    predicate=Predicate.IDENTIFIED_AS,
                    object=real_name,
                    evidence=[evidence_ref],
                    support_type=SupportType.EXPLICIT_STATEMENT,
                    support_strength=3,
                    source_lane=chunk.source_lanes[0] if chunk.source_lanes else "unknown",
                    confidence=0.9,
                )
                claims.append(claim)
                
                # Also create CODENAME_OF claim
                claim2 = Claim(
                    subject=codename,
                    predicate=Predicate.CODENAME_OF,
                    object=real_name,
                    evidence=[evidence_ref],
                    support_type=SupportType.EXPLICIT_STATEMENT,
                    support_strength=3,
                    source_lane=chunk.source_lanes[0] if chunk.source_lanes else "unknown",
                    confidence=0.9,
                )
                claims.append(claim2)
    
    return claims


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_claims(
    chunks: List[ChunkWithProvenance],
    candidates: Dict[str, EntityCandidate],
    intent: IntentFamily,
    extraction_spec: ExtractionSpec,
    conn=None,
) -> List[Claim]:
    """
    Extract claims from chunks.
    
    Two-phase extraction:
    1. Deterministic pattern extraction (always runs)
    2. LLM refinement (optional, controlled by extraction_spec.allow_llm_refinement)
    
    Args:
        chunks: Chunks to extract from (with provenance)
        candidates: Entity candidates to look for
        intent: Intent family for extraction focus
        extraction_spec: Extraction configuration
        conn: Database connection (for LLM refinement)
        
    Returns:
        List of claims with evidence refs and quote spans
    """
    claims = []
    
    # Phase 1: Deterministic extraction
    
    # Extract mentions
    mention_claims = extract_mentions_deterministic(chunks, candidates)
    claims.extend(mention_claims)
    
    # Extract relations (based on intent)
    if intent in (IntentFamily.RELATIONSHIP_CONSTRAINED, IntentFamily.ROSTER_ENUMERATION):
        relation_claims = extract_relations_deterministic(chunks, candidates, intent)
        claims.extend(relation_claims)
    
    # Extract codename mappings
    codename_claims = extract_codename_mappings_deterministic(chunks)
    claims.extend(codename_claims)
    
    # Phase 2: LLM refinement (optional)
    if extraction_spec.allow_llm_refinement:
        claims = refine_claims_with_llm(claims, chunks)
    
    # Filter claims based on extraction spec
    if extraction_spec.support_type_required:
        claims = [
            c for c in claims
            if c.support_type == extraction_spec.support_type_required
        ]
    
    if extraction_spec.min_support_strength > 1:
        claims = [
            c for c in claims
            if c.support_strength >= extraction_spec.min_support_strength
        ]
    
    # Filter claims that have valid evidence
    claims = [c for c in claims if c.has_valid_evidence()]
    
    # Deduplicate claims by claim_id
    seen_ids = set()
    unique_claims = []
    for claim in claims:
        if claim.claim_id not in seen_ids:
            unique_claims.append(claim)
            seen_ids.add(claim.claim_id)
    
    return unique_claims


def refine_claims_with_llm(
    claims: List[Claim],
    chunks: List[ChunkWithProvenance],
) -> List[Claim]:
    """
    Refine claims using LLM (placeholder).
    
    In production, this would:
    - Select best span boundaries when text is messy
    - Classify ambiguous support_type
    - Never invent new claims - only refine existing ones
    
    For now, just returns claims unchanged.
    """
    # TODO: Implement LLM refinement
    # This would call an LLM to:
    # 1. Improve span boundaries
    # 2. Reclassify ambiguous support types
    # 3. Adjust confidence scores
    return claims


# =============================================================================
# Utility Functions
# =============================================================================

def check_role_evidence_in_span(quote_span: str, patterns: List[str]) -> bool:
    """
    Check if any role evidence patterns appear in the quote span.
    
    This is the claim-scoped check that blocks false positives like Fuchs.
    """
    span_lower = quote_span.lower()
    for pattern in patterns:
        if re.search(pattern, span_lower, re.IGNORECASE):
            return True
    return False


def get_claims_by_predicate(
    claims: List[Claim],
    predicate: Predicate,
) -> List[Claim]:
    """Get claims with a specific predicate."""
    return [c for c in claims if c.predicate == predicate]


def get_claims_for_subject(
    claims: List[Claim],
    subject_key: str,
) -> List[Claim]:
    """Get claims for a specific subject."""
    return [c for c in claims if c.subject == subject_key]


def merge_claims(
    existing_claims: List[Claim],
    new_claims: List[Claim],
) -> List[Claim]:
    """
    Merge new claims into existing claims.
    
    Deduplicates by claim_id and merges evidence for duplicates.
    """
    claim_map: Dict[str, Claim] = {c.claim_id: c for c in existing_claims}
    
    for claim in new_claims:
        if claim.claim_id in claim_map:
            # Merge evidence
            existing = claim_map[claim.claim_id]
            existing_ref_ids = {r.ref_id for r in existing.evidence}
            for ref in claim.evidence:
                if ref.ref_id not in existing_ref_ids:
                    existing.evidence.append(ref)
            # Take higher confidence
            existing.confidence = max(existing.confidence, claim.confidence)
        else:
            claim_map[claim.claim_id] = claim
    
    return list(claim_map.values())
