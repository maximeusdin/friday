"""
Validation Layer

Enforces the product contract:
- All citations must be valid bundle IDs
- Confidence auto-downgrade based on evidence strength
- Entity validation (advisory, not hard failure)
- Citation mapping from bundle IDs to chunk IDs
"""

from typing import List, Dict, Set, Tuple, Optional

from .models import (
    SynthesisOutput,
    ClaimOutput,
    ThemeOutput,
    EvidenceBundle,
    ValidationResult,
    EntityValidationResult,
    ClaimWithSupport,
    ThemeWithEvidence,
    CitationWithAnchor,
    EntityInfo,
)


# =============================================================================
# Citation Mapping and Validation
# =============================================================================

def validate_and_map_citations(
    output: SynthesisOutput,
    bundle_map: Dict[str, int],  # bundle_id -> chunk_id
    bundles: List[EvidenceBundle],
) -> ValidationResult:
    """
    Validate LLM output and map bundle IDs to chunk IDs.
    
    Steps:
    1. Map bundle_ids to chunk_ids
    2. Validate all citations are valid bundle_ids
    3. Reject claims with no valid citations
    4. Auto-downgrade confidence if criteria not met
    5. Build final claims with chunk_id citations
    
    Args:
        output: Raw synthesis output from LLM
        bundle_map: Mapping from bundle_id to chunk_id
        bundles: List of evidence bundles
    
    Returns:
        ValidationResult with valid claims, rejected claims, and validation notes
    """
    bundles_by_chunk = {b.chunk_id: b for b in bundles}
    bundles_by_id = {b.bundle_id: b for b in bundles}
    
    valid_claims: List[ClaimWithSupport] = []
    rejected_claims: List[ClaimOutput] = []
    validation_notes: List[str] = []
    
    for claim in output.claims:
        # Map and validate citations
        mapped_citations: List[CitationWithAnchor] = []
        invalid_citations: List[str] = []
        
        for bundle_id in claim.citations:
            if bundle_id not in bundle_map:
                invalid_citations.append(bundle_id)
                continue
            
            chunk_id = bundle_map[bundle_id]
            bundle = bundles_by_id.get(bundle_id)
            
            if bundle:
                mapped_citations.append(CitationWithAnchor(
                    chunk_id=chunk_id,
                    bundle_id=bundle_id,
                    quote_anchor=bundle.quote_anchor.to_dict(),
                    doc_title=bundle.doc_title,
                    page=bundle.page,
                ))
        
        # Reject claims with no valid citations
        if not mapped_citations:
            rejected_claims.append(claim)
            validation_notes.append(
                f"Claim {claim.claim_id} rejected: no valid citations "
                f"(attempted: {claim.citations}, invalid: {invalid_citations})"
            )
            continue
        
        if invalid_citations:
            validation_notes.append(
                f"Claim {claim.claim_id}: dropped invalid citations {invalid_citations}"
            )
        
        # Validate and potentially downgrade confidence
        confidence, confidence_reason = validate_confidence(
            claim=claim,
            citations=[c.chunk_id for c in mapped_citations],
            bundles_by_chunk=bundles_by_chunk,
        )
        
        if confidence != claim.confidence:
            validation_notes.append(
                f"Claim {claim.claim_id}: confidence downgraded from "
                f"{claim.confidence} to {confidence} ({confidence_reason})"
            )
        
        valid_claims.append(ClaimWithSupport(
            claim_id=claim.claim_id,
            claim=claim.claim,
            support=mapped_citations,
            confidence=confidence,
            confidence_reason=confidence_reason if confidence != claim.confidence else None,
            limitations=claim.limitations,
        ))
    
    # Validate themes
    valid_themes = validate_themes(output.themes, bundle_map, bundles_by_id)
    
    # Validate entities (advisory)
    entity_result = validate_entities(output, bundles, bundle_map)
    
    return ValidationResult(
        valid_claims=valid_claims,
        rejected_claims=rejected_claims,
        themes=valid_themes,
        entities_verified=entity_result.inferred_entity_ids,
        entities_flagged=entity_result.flagged_entities,
        validation_notes=validation_notes,
    )


# =============================================================================
# Confidence Auto-Downgrade
# =============================================================================

def validate_confidence(
    claim: ClaimOutput,
    citations: List[int],  # chunk_ids
    bundles_by_chunk: Dict[int, EvidenceBundle],
) -> Tuple[str, Optional[str]]:
    """
    Auto-downgrade confidence if criteria not met.
    
    confidence="high" requires:
    - >=2 citations from different documents, OR
    - >=2 citations + strong match_trace (ENTITY+PHRASE matched)
    
    Otherwise: downgrade to "medium"
    
    Args:
        claim: The claim being validated
        citations: List of chunk_ids
        bundles_by_chunk: Bundle lookup by chunk_id
    
    Returns:
        (effective_confidence, reason_if_downgraded)
    """
    if claim.confidence != "high":
        return claim.confidence, None
    
    if len(citations) < 1:
        return "low", "no citations"
    
    # Check document diversity
    doc_ids = set()
    has_strong_trace = False
    
    for chunk_id in citations:
        bundle = bundles_by_chunk.get(chunk_id)
        if bundle:
            doc_ids.add(bundle.doc_id)
            # Strong trace = both entity and phrase matches
            if bundle.match_trace.matched_entity_ids and bundle.match_trace.matched_phrases:
                has_strong_trace = True
    
    # High confidence criteria
    if len(doc_ids) >= 2:
        return "high", None  # Multiple docs - high is valid
    
    if len(citations) >= 2 and has_strong_trace:
        return "high", None  # Multiple citations with strong trace
    
    # Downgrade
    if len(doc_ids) == 1:
        return "medium", "single document source"
    
    return "medium", "insufficient corroboration"


# =============================================================================
# Theme Validation
# =============================================================================

def validate_themes(
    themes: List[ThemeOutput],
    bundle_map: Dict[str, int],
    bundles_by_id: Dict[str, EvidenceBundle],
) -> List[ThemeWithEvidence]:
    """Validate and map theme citations."""
    valid_themes = []
    
    for theme in themes:
        mapped_evidence = []
        for bundle_id in theme.evidence:
            if bundle_id in bundle_map:
                bundle = bundles_by_id.get(bundle_id)
                if bundle:
                    mapped_evidence.append(CitationWithAnchor(
                        chunk_id=bundle_map[bundle_id],
                        bundle_id=bundle_id,
                        quote_anchor=bundle.quote_anchor.to_dict(),
                        doc_title=bundle.doc_title,
                        page=bundle.page,
                    ))
        
        valid_themes.append(ThemeWithEvidence(
            theme=theme.theme,
            description=theme.description,
            evidence=mapped_evidence,
        ))
    
    return valid_themes


# =============================================================================
# Entity Validation (Advisory)
# =============================================================================

def validate_entities(
    synthesis_output: SynthesisOutput,
    bundles: List[EvidenceBundle],
    bundle_map: Dict[str, int],
) -> EntityValidationResult:
    """
    Entity validation is ADVISORY - flags issues but doesn't reject summaries.
    
    Validation is finicky due to synonyms, casing, abbreviations.
    Don't overfit it early.
    
    PREFERRED approach: infer entities from cited bundles, not model's list.
    
    Args:
        synthesis_output: Raw LLM output
        bundles: All evidence bundles
        bundle_map: Bundle ID to chunk ID mapping
    
    Returns:
        EntityValidationResult with verified, flagged, and inferred entities
    """
    # Collect known entity IDs from all cited bundles
    cited_entity_ids: Set[int] = set()
    bundles_by_id = {b.bundle_id: b for b in bundles}
    
    for claim in synthesis_output.claims:
        for bundle_id in claim.citations:
            if bundle_id in bundles_by_id:
                bundle = bundles_by_id[bundle_id]
                cited_entity_ids.update(bundle.match_trace.matched_entity_ids or [])
    
    # Also include entities from all bundles (not just cited)
    all_entity_ids: Set[int] = set()
    for bundle in bundles:
        all_entity_ids.update(bundle.match_trace.matched_entity_ids or [])
    
    # If model outputs entities_mentioned, validate as advisory
    # Note: We don't have entity name lookup here, so we just flag all as advisory
    verified = []
    flagged = []
    
    if synthesis_output.entities_mentioned:
        # Without DB lookup, we treat all model-mentioned entities as unverified
        # This is intentionally conservative
        flagged = list(synthesis_output.entities_mentioned)
    
    return EntityValidationResult(
        verified_entities=verified,
        flagged_entities=flagged,
        inferred_entity_ids=list(cited_entity_ids),
        validation_notes="Entity validation is advisory; flagged entities require manual verification"
    )


def extract_entities_from_citations(
    conn,
    valid_claims: List[ClaimWithSupport],
    bundles: List[EvidenceBundle],
) -> List[EntityInfo]:
    """
    Server extracts entities from cited bundles.
    More reliable than asking model to list entities.
    
    Args:
        conn: Database connection
        valid_claims: Validated claims with citations
        bundles: All evidence bundles
    
    Returns:
        List of EntityInfo from cited bundles
    """
    # Collect all entity IDs from cited bundles
    entity_ids: Set[int] = set()
    bundles_by_chunk = {b.chunk_id: b for b in bundles}
    
    for claim in valid_claims:
        for citation in claim.support:
            bundle = bundles_by_chunk.get(citation.chunk_id)
            if bundle:
                entity_ids.update(bundle.match_trace.matched_entity_ids or [])
    
    if not entity_ids:
        return []
    
    # Look up entity details
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, entity_type
            FROM entities
            WHERE id = ANY(%s)
            """,
            (list(entity_ids),)
        )
        
        entities = []
        for row in cur.fetchall():
            entities.append(EntityInfo(
                entity_id=row[0],
                name=row[1],
                entity_type=row[2],
                mention_count=1,  # Could count from bundles if needed
            ))
        
        return entities
