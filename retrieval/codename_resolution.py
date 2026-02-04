"""
Codename Resolution for Agentic Workflow.

Separates "mapping extraction" from "mapping assertion":
- extract_explicit_mappings(): Pattern-based extraction from text
- resolve_codenames(): Uses alias table + extraction for resolution

Only creates codename_of() claims with evidence - no guesses.
Unresolved codenames go to unresolved_tokens[].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from retrieval.evidence_bundle import (
    Claim,
    EvidenceRef,
    ChunkWithProvenance,
    Predicate,
    SupportType,
)


# =============================================================================
# Mapping Patterns
# =============================================================================

# Patterns for explicit codename mappings
CODENAME_MAPPING_PATTERNS = [
    # Pattern: "TWAIN" was [identified as] Name
    (r'(?P<code>[A-Z]{3,})["\']?\s+(?:was\s+)?(?:identified\s+as|is)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "identified_as"),
    
    # Pattern: Name (TWAIN)
    (r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((?P<code>[A-Z]{3,})\)', "parenthetical"),
    
    # Pattern: Name, codenamed TWAIN
    (r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:code)?named\s+(?P<code>[A-Z]{3,})', "codenamed"),
    
    # Pattern: TWAIN = Name
    (r'(?P<code>[A-Z]{3,})\s*[=:]\s*(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "equals"),
    
    # Pattern: "TWAIN" (Name)
    (r'["\']?(?P<code>[A-Z]{3,})["\']?\s*\((?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\)', "code_parenthetical"),
    
    # Pattern: cover name "TWAIN" was Name
    (r'cover\s+name\s+["\']?(?P<code>[A-Z]{3,})["\']?\s+(?:was|is)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "covername"),
    
    # Pattern: Name's cover name was "TWAIN"
    (r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[\'\s]+s?\s*cover\s+name\s+(?:was|is)\s+["\']?(?P<code>[A-Z]{3,})', "possessive_covername"),
]

# Pattern confidence scores
PATTERN_CONFIDENCE = {
    "identified_as": 0.95,
    "parenthetical": 0.85,
    "codenamed": 0.90,
    "equals": 0.80,
    "code_parenthetical": 0.85,
    "covername": 0.95,
    "possessive_covername": 0.90,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MappingCandidate:
    """A candidate mapping from codename to real name."""
    codename: str
    real_name: str
    pattern_type: str
    confidence: float
    chunk_id: int
    doc_id: int
    char_start: int
    char_end: int
    quote_span: str


@dataclass
class ResolutionResult:
    """Result from codename resolution."""
    resolved: Dict[str, str]              # codename -> real_name
    claims: List[Claim]                   # mapping claims with evidence
    unresolved: List[str]                 # codenames that couldn't be resolved


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_explicit_mappings(
    chunks: List[ChunkWithProvenance],
    candidate_tokens: List[str] = None,
) -> List[MappingCandidate]:
    """
    Pattern-based extraction of explicit codename mappings from text.
    
    Looks for patterns like:
    - "TWAIN" was [Name]
    - [Name] (TWAIN)
    - [Name], codenamed TWAIN
    - TWAIN = [Name]
    
    This is deterministic and auditable - no LLM involved.
    
    Args:
        chunks: Chunks to search
        candidate_tokens: Optional list of specific codenames to look for
        
    Returns:
        List of MappingCandidate with evidence
    """
    candidates = []
    
    for chunk in chunks:
        if not chunk.text:
            continue
        
        for pattern, pattern_type in CODENAME_MAPPING_PATTERNS:
            for match in re.finditer(pattern, chunk.text, re.IGNORECASE):
                try:
                    codename = match.group("code")
                    real_name = match.group("name")
                except IndexError:
                    continue
                
                if not codename or not real_name:
                    continue
                
                # Normalize codename to uppercase
                codename = codename.upper()
                
                # If we have specific tokens to look for, filter
                if candidate_tokens and codename not in [t.upper() for t in candidate_tokens]:
                    continue
                
                # Get context for quote span
                context_start = max(0, match.start() - 50)
                context_end = min(len(chunk.text), match.end() + 50)
                quote_span = chunk.text[context_start:context_end].strip()
                
                confidence = PATTERN_CONFIDENCE.get(pattern_type, 0.7)
                
                candidates.append(MappingCandidate(
                    codename=codename,
                    real_name=real_name,
                    pattern_type=pattern_type,
                    confidence=confidence,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    char_start=match.start(),
                    char_end=match.end(),
                    quote_span=quote_span,
                ))
    
    return candidates


def extract_mappings_from_alias_table(
    tokens: List[str],
    conn,
) -> Dict[str, Tuple[int, str]]:
    """
    Look up codenames in the existing alias table.
    
    Returns mappings that have been curated from prior evidence.
    
    Args:
        tokens: Codename tokens to look up
        conn: Database connection
        
    Returns:
        Dict mapping codename -> (entity_id, canonical_name)
    """
    if not tokens:
        return {}
    
    mappings = {}
    
    with conn.cursor() as cur:
        # Look for aliases with kind='code_name' or alias_class='covername'
        cur.execute(
            """
            SELECT ea.alias_norm, ea.entity_id, e.canonical_name
            FROM entity_aliases ea
            JOIN entities e ON ea.entity_id = e.id
            WHERE ea.alias_norm = ANY(%(tokens)s)
              AND (ea.kind = 'code_name' OR ea.alias_class = 'covername')
              AND ea.is_matchable = true
            """,
            {"tokens": [t.lower() for t in tokens]}
        )
        
        for alias_norm, entity_id, canonical_name in cur.fetchall():
            # Map back to original case token
            for token in tokens:
                if token.lower() == alias_norm:
                    mappings[token.upper()] = (entity_id, canonical_name)
                    break
    
    return mappings


def search_for_mapping_evidence(
    token: str,
    allowed_collections: List[str],
    conn,
    limit: int = 10,
) -> List[ChunkWithProvenance]:
    """
    Search for chunks that might contain mapping evidence for a token.
    
    Args:
        token: The codename to search for
        allowed_collections: Allowed collections
        conn: Database connection
        limit: Maximum chunks to return
        
    Returns:
        List of chunks with text loaded
    """
    chunks = []
    
    with conn.cursor() as cur:
        # Search for chunks containing the token
        query = """
            SELECT c.id, cm.document_id, COALESCE(c.clean_text, c.text) as text
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.text_tsv @@ to_tsquery('simple', %(token)s)
        """
        params = {"token": token}
        
        if allowed_collections:
            query += """
                AND cm.collection_slug = ANY(%(collections)s)
            """
            params["collections"] = allowed_collections
        
        query += " LIMIT %(limit)s"
        params["limit"] = limit
        
        cur.execute(query, params)
        
        for chunk_id, doc_id, text in cur.fetchall():
            chunks.append(ChunkWithProvenance(
                chunk_id=chunk_id,
                doc_id=doc_id or 0,
                source_lanes=["codename_search"],
                best_score=1.0,
                first_seen_round=0,
                text=text,
            ))
    
    return chunks


# =============================================================================
# Resolution Functions
# =============================================================================

def resolve_codenames(
    tokens: List[str],
    allowed_collections: List[str],
    conn,
) -> ResolutionResult:
    """
    Resolve codename tokens to real names.
    
    Uses:
    1. Alias table (curated from prior evidence)
    2. Pattern extraction from relevant chunks
    
    Only creates codename_of() claims with evidence.
    
    Args:
        tokens: List of codename tokens to resolve
        allowed_collections: Allowed collections for evidence search
        conn: Database connection
        
    Returns:
        ResolutionResult with resolved mappings, claims, and unresolved tokens
    """
    if not tokens:
        return ResolutionResult(resolved={}, claims=[], unresolved=[])
    
    resolved: Dict[str, str] = {}
    claims: List[Claim] = []
    unresolved: List[str] = []
    
    # Normalize tokens
    normalized_tokens = [t.upper() for t in tokens]
    
    # Step 1: Check alias table
    alias_mappings = extract_mappings_from_alias_table(normalized_tokens, conn)
    
    for codename, (entity_id, canonical_name) in alias_mappings.items():
        resolved[codename] = canonical_name
        
        # Create claim from alias table (no quote span, but entity_id evidence)
        claim = Claim(
            subject=f"token:{codename}",
            predicate=Predicate.CODENAME_OF,
            object=f"entity:{entity_id}",
            evidence=[],  # From alias table, not chunk
            support_type=SupportType.DEFINITION,
            support_strength=3,
            source_lane="alias_table",
            confidence=0.95,
        )
        claims.append(claim)
    
    # Step 2: For remaining tokens, search for mapping evidence
    remaining_tokens = [t for t in normalized_tokens if t not in resolved]
    
    for token in remaining_tokens:
        # Search for chunks with this token
        search_chunks = search_for_mapping_evidence(
            token, allowed_collections, conn
        )
        
        if not search_chunks:
            unresolved.append(token)
            continue
        
        # Extract mappings from found chunks
        mapping_candidates = extract_explicit_mappings(
            search_chunks, candidate_tokens=[token]
        )
        
        if not mapping_candidates:
            unresolved.append(token)
            continue
        
        # Take the best mapping (highest confidence)
        best_candidate = max(mapping_candidates, key=lambda c: c.confidence)
        resolved[token] = best_candidate.real_name
        
        # Create claim with evidence
        evidence_ref = EvidenceRef(
            chunk_id=best_candidate.chunk_id,
            doc_id=best_candidate.doc_id,
            page_ref="",
            char_start=best_candidate.char_start,
            char_end=best_candidate.char_end,
            quote_span=best_candidate.quote_span,
        )
        
        claim = Claim(
            subject=f"token:{token}",
            predicate=Predicate.CODENAME_OF,
            object=best_candidate.real_name,
            evidence=[evidence_ref],
            support_type=SupportType.EXPLICIT_STATEMENT,
            support_strength=3,
            source_lane="pattern_extraction",
            confidence=best_candidate.confidence,
        )
        claims.append(claim)
    
    return ResolutionResult(
        resolved=resolved,
        claims=claims,
        unresolved=unresolved,
    )


def resolve_single_codename(
    token: str,
    allowed_collections: List[str],
    conn,
) -> Optional[Tuple[str, Claim]]:
    """
    Resolve a single codename token.
    
    Args:
        token: The codename to resolve
        allowed_collections: Allowed collections
        conn: Database connection
        
    Returns:
        (real_name, claim) or None if not resolved
    """
    result = resolve_codenames([token], allowed_collections, conn)
    
    if token.upper() in result.resolved:
        real_name = result.resolved[token.upper()]
        claim = next((c for c in result.claims if token.upper() in c.subject), None)
        return (real_name, claim)
    
    return None


# =============================================================================
# Utility Functions
# =============================================================================

def is_likely_codename(token: str) -> bool:
    """
    Check if a token looks like a codename.
    
    Codenames are typically:
    - All uppercase
    - 3+ characters
    - Not common English words
    """
    if len(token) < 3:
        return False
    
    if not token.isupper():
        return False
    
    # Common words that shouldn't be treated as codenames
    common_words = {
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
        "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS",
        "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "WAY",
        "WHO", "DID", "GET", "HIM", "HIS", "LET", "PUT", "SAY",
        "SHE", "TOO", "USE", "FBI", "CIA", "USA", "USSR", "KGB",
    }
    
    if token in common_words:
        return False
    
    return True


def extract_potential_codenames(text: str) -> List[str]:
    """
    Extract potential codenames from text.
    
    Looks for uppercase words that might be codenames.
    """
    pattern = r'\b([A-Z]{3,})\b'
    matches = re.findall(pattern, text)
    
    # Filter to likely codenames
    return [m for m in set(matches) if is_likely_codename(m)]


def get_codename_context(
    text: str,
    codename: str,
    context_chars: int = 100,
) -> Optional[str]:
    """
    Get context around a codename in text.
    """
    pattern = re.escape(codename)
    match = re.search(pattern, text, re.IGNORECASE)
    
    if not match:
        return None
    
    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)
    
    return text[start:end].strip()
