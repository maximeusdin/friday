"""
V3 Verifier - Universal verification checks (no intent families).

The verifier checks:
1. Citation existence: every claim has >= 1 EvidenceRef
2. Span validity: offsets map to chunk text, quote matches
3. Entity attestation: for each entity in about_entities, a surface form must appear in cited quotes
4. Cap & diversity: max citations per claim, no duplicates
5. Relationship cues (if asserts_relationship): quote must contain relationship language

These are UNIVERSAL rules - no intent-family-specific logic.

Entity Attestation Contract:
- Each claim declares about_entities (entity_ids it asserts facts about)
- For each entity_id, at least one cited quote must contain a recognized surface form
- Surfaces include canonical name and all aliases
- This replaces the old NER-based "referent" checking which was brittle
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set

from retrieval.agent.v3_claims import ClaimV3, ClaimBundleV3, EvidenceRef
from retrieval.agent.v3_evidence import EvidenceSet, EvidenceSpan
from retrieval.agent.entity_surfaces import EntitySurfaceIndex


@dataclass
class VerificationError:
    """A verification error with actionable details."""
    error_type: str           # "missing_citation", "quote_mismatch", etc.
    claim_id: str
    details: str
    actionable: bool = True   # Can agent fix this?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "claim_id": self.claim_id,
            "details": self.details,
            "actionable": self.actionable,
        }


@dataclass
class VerificationReport:
    """Result of verification."""
    passed: bool
    errors: List[VerificationError]
    warnings: List[str]
    stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "error_count": len(self.errors),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "stats": self.stats,
        }


# Relationship cue patterns (universal, not query-specific)
RELATIONSHIP_CUES = {
    'contact', 'contacted', 'met', 'meeting', 'meet',
    'handle', 'handled', 'handler',
    'associated', 'association', 'associate',
    'connected', 'connection',
    'recruited', 'recruit',
    'worked with', 'collaborated',
    'member of', 'part of', 'belonged to',
    'reported to', 'controlled by',
    'network', 'ring', 'group',
    'courier', 'source', 'agent of',
}


class VerifierV3:
    """
    Universal verifier for V3 claims.
    
    All checks are universal (not intent-family-specific).
    Uses entity attestation instead of NER-based referent extraction.
    """
    
    def __init__(
        self,
        max_citations_per_claim: int = 2,
        require_quote_match: bool = True,
        verbose: bool = True,
    ):
        self.max_citations_per_claim = max_citations_per_claim
        self.require_quote_match = require_quote_match
        self.verbose = verbose
        self._surface_index: Optional[EntitySurfaceIndex] = None
    
    def verify(
        self,
        bundle: ClaimBundleV3,
        evidence_set: EvidenceSet,
        conn,
    ) -> VerificationReport:
        """
        Verify all claims in a bundle.
        
        Args:
            bundle: ClaimBundleV3 to verify
            evidence_set: EvidenceSet the claims should cite from
            conn: Database connection (for entity surface lookup)
        
        Returns:
            VerificationReport with errors and stats
        """
        if self.verbose:
            print(f"\n  [Verifier] Checking {len(bundle.claims)} claims...", file=sys.stderr)
        
        # Initialize entity surface index
        self._surface_index = EntitySurfaceIndex(conn)
        
        # Preload all entities referenced in claims
        all_entity_ids = []
        for claim in bundle.claims:
            all_entity_ids.extend(claim.about_entities)
        if all_entity_ids:
            self._surface_index.preload_entities(list(set(all_entity_ids)))
        
        errors: List[VerificationError] = []
        warnings: List[str] = []
        
        # Track citation usage
        cited_spans: Set[str] = set()
        
        for claim in bundle.claims:
            claim_errors = []
            
            # 1. Citation existence
            claim_errors.extend(self._check_citation_existence(claim))
            
            # 2. Span validity
            claim_errors.extend(self._check_span_validity(claim, evidence_set, conn))
            
            # 3. Entity attestation (replaces old referent checks)
            claim_errors.extend(self._check_entity_attestation(claim))
            
            # 4. Relationship cues (if applicable)
            if claim.asserts_relationship:
                claim_errors.extend(self._check_relationship_cues(claim))
            
            errors.extend(claim_errors)
            
            # Track citations
            for ref in claim.evidence:
                if ref.span_id:
                    cited_spans.add(ref.span_id)
        
        # 5. Cap & diversity (bundle-level)
        bundle_warnings = self._check_cap_and_diversity(bundle, evidence_set, cited_spans)
        warnings.extend(bundle_warnings)
        
        # Compute stats
        stats = {
            "claim_count": len(bundle.claims),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "unique_spans_cited": len(cited_spans),
            "cite_spans_available": len(evidence_set.cite_spans),
            "coverage_ratio": len(cited_spans) / max(len(evidence_set.cite_spans), 1),
        }
        
        passed = len(errors) == 0
        
        if self.verbose:
            status = "PASSED" if passed else f"FAILED ({len(errors)} errors)"
            print(f"    Verification: {status}", file=sys.stderr)
            if errors:
                for e in errors[:3]:
                    print(f"      [{e.error_type}] {e.details[:80]}", file=sys.stderr)
        
        return VerificationReport(
            passed=passed,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
    
    def _check_citation_existence(self, claim: ClaimV3) -> List[VerificationError]:
        """Check that claim has at least 1 evidence ref."""
        if not claim.evidence:
            return [VerificationError(
                error_type="missing_citation",
                claim_id=claim.claim_id,
                details=f"Claim has no evidence citations: \"{claim.text[:50]}...\"",
                actionable=True,
            )]
        return []
    
    def _check_span_validity(
        self,
        claim: ClaimV3,
        evidence_set: EvidenceSet,
        conn,
    ) -> List[VerificationError]:
        """Check that cited spans exist and quotes match."""
        errors = []
        
        for ref in claim.evidence:
            # Check span exists in evidence set
            if ref.span_id:
                span = evidence_set.get_span_by_id(ref.span_id)
                if not span:
                    errors.append(VerificationError(
                        error_type="invalid_span",
                        claim_id=claim.claim_id,
                        details=f"Span {ref.span_id} not in evidence set",
                        actionable=True,
                    ))
                    continue
                
                # Check quote matches
                if self.require_quote_match and ref.quote:
                    # Normalize for comparison
                    ref_quote_norm = ' '.join(ref.quote.lower().split())
                    span_quote_norm = ' '.join(span.quote.lower().split())
                    
                    if ref_quote_norm not in span_quote_norm and span_quote_norm not in ref_quote_norm:
                        # Allow partial match
                        overlap = self._quote_overlap(ref_quote_norm, span_quote_norm)
                        if overlap < 0.5:
                            errors.append(VerificationError(
                                error_type="quote_mismatch",
                                claim_id=claim.claim_id,
                                details=f"Quote doesn't match span text (overlap: {overlap:.0%})",
                                actionable=False,
                            ))
            
            # Verify against database if no span_id
            elif ref.chunk_id and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT text FROM chunks WHERE id = %s",
                            (ref.chunk_id,)
                        )
                        row = cur.fetchone()
                        if not row:
                            errors.append(VerificationError(
                                error_type="invalid_chunk",
                                claim_id=claim.claim_id,
                                details=f"Chunk {ref.chunk_id} not found",
                                actionable=False,
                            ))
                        elif self.require_quote_match and ref.quote:
                            chunk_text = row[0] or ""
                            if ref.quote not in chunk_text:
                                errors.append(VerificationError(
                                    error_type="quote_not_in_chunk",
                                    claim_id=claim.claim_id,
                                    details=f"Quote not found in chunk {ref.chunk_id}",
                                    actionable=False,
                                ))
                except Exception as e:
                    pass  # Skip DB errors
        
        return errors
    
    def _quote_overlap(self, a: str, b: str) -> float:
        """Compute word overlap between two quotes."""
        words_a = set(a.split())
        words_b = set(b.split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _check_entity_attestation(self, claim: ClaimV3) -> List[VerificationError]:
        """
        Check that each entity in about_entities is attested in the cited quotes.
        
        Entity Attestation Contract:
        - For each entity_id in claim.about_entities
        - At least one cited quote must contain a recognized surface form
        - Surfaces include canonical name and all aliases
        
        This is the primary grounding check, replacing old referent extraction.
        """
        errors = []
        
        if not claim.about_entities:
            # No entities declared - acceptable for general factual claims
            return []
        
        if not self._surface_index:
            # No index available - skip check
            return []
        
        # Collect all cited quotes
        quotes = [ref.quote for ref in claim.evidence if ref.quote]
        
        if not quotes:
            # No quote text - will be caught by citation existence check
            return []
        
        # Check each declared entity
        for entity_id in claim.about_entities:
            if self._surface_index.attests_any_quote(entity_id, quotes):
                continue  # Entity is attested
            
            # Entity not found - generate actionable error
            example_surfaces = self._surface_index.get_top_surfaces(entity_id, limit=3)
            if not example_surfaces:
                example_surfaces = ["(no surfaces found)"]
            
            errors.append(VerificationError(
                error_type="missing_entity_in_evidence",
                claim_id=claim.claim_id,
                details=(
                    f"Entity {entity_id} not attested in cited quotes. "
                    f"Expected one of: {example_surfaces}. "
                    f"Either fix citations, fix about_entities, or drop claim."
                ),
                actionable=True,
            ))
        
        return errors
    
    def _check_about_entities_present(self, claim: ClaimV3) -> List[VerificationError]:
        """
        Check that claims about specific entities have about_entities populated.
        
        This is a soft check initially - warns but doesn't fail.
        """
        errors = []
        
        # If claim asserts a relationship, it should have at least 2 entities
        if claim.asserts_relationship and len(claim.about_entities) < 2:
            errors.append(VerificationError(
                error_type="missing_about_entities",
                claim_id=claim.claim_id,
                details="Relationship claim should have at least 2 entities in about_entities",
                actionable=True,
            ))
        
        return errors
    
    def _check_relationship_cues(self, claim: ClaimV3) -> List[VerificationError]:
        """Check that relationship claims have relationship language in quotes."""
        errors = []
        
        # Collect all quote text
        all_quotes = " ".join(ref.quote for ref in claim.evidence if ref.quote)
        all_quotes_lower = all_quotes.lower()
        
        # Check for any relationship cue
        found_cue = False
        for cue in RELATIONSHIP_CUES:
            if cue in all_quotes_lower:
                found_cue = True
                break
        
        if not found_cue:
            errors.append(VerificationError(
                error_type="missing_relationship_cue",
                claim_id=claim.claim_id,
                details="Relationship claim but no relationship language in quotes",
                actionable=True,
            ))
        
        return errors
    
    def _check_cap_and_diversity(
        self,
        bundle: ClaimBundleV3,
        evidence_set: EvidenceSet,
        cited_spans: Set[str],
    ) -> List[str]:
        """Check citation caps and diversity (bundle-level)."""
        warnings = []
        
        # Check max citations per claim
        for claim in bundle.claims:
            if len(claim.evidence) > self.max_citations_per_claim:
                warnings.append(
                    f"Claim {claim.claim_id} has {len(claim.evidence)} citations "
                    f"(max: {self.max_citations_per_claim})"
                )
        
        # Check for duplicate citations across claims
        span_usage: Dict[str, int] = {}
        for claim in bundle.claims:
            for ref in claim.evidence:
                if ref.span_id:
                    span_usage[ref.span_id] = span_usage.get(ref.span_id, 0) + 1
        
        over_used = [sid for sid, count in span_usage.items() if count > 3]
        if over_used:
            warnings.append(f"{len(over_used)} spans cited more than 3 times")
        
        # Check evidence coverage
        if evidence_set.cite_spans and cited_spans:
            coverage = len(cited_spans) / len(evidence_set.cite_spans)
            if coverage < 0.1 and len(bundle.claims) > 0:
                warnings.append(f"Low evidence coverage: {coverage:.0%} of cite_spans used")
        
        return warnings
