"""
Deterministic Verifier for Agentic Workflow.

The verifier is the gate between executor and renderer. It ensures:
- Universal rules: no claim without evidence refs + quote span
- Intent-specific coverage rules: which lanes must run
- Constraint-specific rules: collection scope, role evidence

Key design:
- Role evidence is checked on the CLAIM's quote_span, not the whole chunk
  This is what blocks Fuchs: his mention is co_mention, not handler evidence
- Coverage requirements vary by intent family
- Returns actionable errors for retry
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from retrieval.plan import BundleConstraints, VerificationSpec
from retrieval.evidence_bundle import (
    EvidenceBundle,
    Claim,
    RetrievalLaneRun,
    Predicate,
    SupportType,
)
from retrieval.intent import IntentFamily, get_coverage_requirements


# =============================================================================
# Verification Errors
# =============================================================================

class VerificationErrorType(str, Enum):
    """Types of verification errors."""
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_QUOTE_SPAN = "missing_quote_span"
    INVALID_COLLECTION = "invalid_collection"
    MISSING_ROLE_EVIDENCE = "missing_role_evidence"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    REQUIRED_LANE_MISSING = "required_lane_missing"
    SUPPORT_TYPE_MISMATCH = "support_type_mismatch"
    LOW_SUPPORT_STRENGTH = "low_support_strength"


@dataclass
class VerificationError:
    """A verification error with details."""
    error_type: VerificationErrorType
    message: str
    claim_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "claim_id": self.claim_id,
            "details": self.details,
        }


@dataclass
class VerificationResult:
    """Result of evidence bundle verification."""
    passed: bool
    errors: List[VerificationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dropped_claims: List[Tuple[Claim, str]] = field(default_factory=list)  # (claim, reason)
    coverage_achieved: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "dropped_claims": [
                {"claim_id": c.claim_id, "reason": r}
                for c, r in self.dropped_claims
            ],
            "coverage_achieved": self.coverage_achieved,
        }


# =============================================================================
# Universal Rules
# =============================================================================

def verify_claim_has_evidence(claim: Claim) -> Optional[VerificationError]:
    """
    Universal rule: No claim without evidence refs.
    """
    if not claim.evidence:
        return VerificationError(
            error_type=VerificationErrorType.MISSING_EVIDENCE,
            message=f"Claim {claim.claim_id} has no evidence refs",
            claim_id=claim.claim_id,
        )
    return None


def verify_claim_has_quote_spans(claim: Claim) -> Optional[VerificationError]:
    """
    Universal rule: All evidence refs must have quote spans.
    """
    for ref in claim.evidence:
        if not ref.quote_span:
            return VerificationError(
                error_type=VerificationErrorType.MISSING_QUOTE_SPAN,
                message=f"Claim {claim.claim_id} has evidence without quote_span",
                claim_id=claim.claim_id,
                details={"ref_id": ref.ref_id},
            )
    return None


def verify_universal_rules(bundle: EvidenceBundle) -> List[VerificationError]:
    """
    Check all universal rules on bundle claims.
    """
    errors = []
    
    for claim in bundle.claims:
        error = verify_claim_has_evidence(claim)
        if error:
            errors.append(error)
        
        error = verify_claim_has_quote_spans(claim)
        if error:
            errors.append(error)
    
    return errors


# =============================================================================
# Collection Scope Rules
# =============================================================================

def verify_collection_scope(
    claim: Claim,
    allowed_collections: List[str],
    conn=None,
) -> Optional[VerificationError]:
    """
    Verify claim evidence comes from allowed collections.
    
    If allowed_collections is empty, all collections are allowed.
    """
    if not allowed_collections:
        return None
    
    # Look up collection from chunk_metadata (collection_slug is there, not in documents)
    # For now, skip this check if no connection
    if not conn:
        return None
    
    for ref in claim.evidence:
        with conn.cursor() as cur:
            # collection_slug is in chunk_metadata, not documents
            cur.execute(
                "SELECT collection_slug FROM chunk_metadata WHERE chunk_id = %s",
                (ref.chunk_id,)
            )
            row = cur.fetchone()
            if row and row[0] and row[0] not in allowed_collections:
                return VerificationError(
                    error_type=VerificationErrorType.INVALID_COLLECTION,
                    message=f"Claim {claim.claim_id} has evidence from non-allowed collection: {row[0]}",
                    claim_id=claim.claim_id,
                    details={"collection": row[0], "allowed": allowed_collections},
                )
    
    return None


# =============================================================================
# Role Evidence Rules (CRITICAL)
# =============================================================================

def verify_role_evidence(
    claim: Claim,
    patterns: List[str],
) -> bool:
    """
    Check if quote_span contains a role indicator.
    
    This is CLAIM-SCOPED, not bundle-scoped.
    The check is on the specific evidence span, not anywhere in the chunk.
    
    This is what blocks the Rosenberg→Fuchs mistake:
    - Fuchs may appear in the chunk
    - But his mention is NOT in a handler span with officer language
    - So claim fails role verification
    
    Args:
        claim: The claim to check
        patterns: Role evidence patterns to look for
        
    Returns:
        True if role evidence found in quote_span, False otherwise
    """
    if not patterns:
        return True  # No patterns required
    
    for evidence_ref in claim.evidence:
        span_text = evidence_ref.quote_span.lower()
        
        for pattern in patterns:
            if re.search(pattern.lower(), span_text, re.IGNORECASE):
                return True
    
    return False


def verify_claim_role_evidence(
    claim: Claim,
    constraints: BundleConstraints,
) -> Optional[VerificationError]:
    """
    Verify claim has required role evidence in quote_span.
    
    Only applies to claims that need role evidence (handler, officer, etc.)
    """
    if not constraints.required_role_evidence_patterns:
        return None
    
    # Only check for relationship predicates
    relationship_predicates = {Predicate.HANDLED_BY, Predicate.MET_WITH}
    if claim.predicate not in relationship_predicates:
        return None
    
    if not verify_role_evidence(claim, constraints.required_role_evidence_patterns):
        return VerificationError(
            error_type=VerificationErrorType.MISSING_ROLE_EVIDENCE,
            message=f"Claim {claim.claim_id} missing role evidence in quote_span",
            claim_id=claim.claim_id,
            details={
                "predicate": claim.predicate.value,
                "required_patterns": constraints.required_role_evidence_patterns,
            },
        )
    
    return None


# =============================================================================
# Support Type Rules
# =============================================================================

def verify_support_type(
    claim: Claim,
    constraints: BundleConstraints,
) -> Optional[VerificationError]:
    """
    Verify claim meets support type requirements.
    """
    if constraints.support_type_required:
        if claim.support_type != constraints.support_type_required:
            return VerificationError(
                error_type=VerificationErrorType.SUPPORT_TYPE_MISMATCH,
                message=f"Claim {claim.claim_id} has {claim.support_type.value}, requires {constraints.support_type_required.value}",
                claim_id=claim.claim_id,
                details={
                    "actual": claim.support_type.value,
                    "required": constraints.support_type_required.value,
                },
            )
    
    if claim.support_strength < constraints.min_support_strength:
        return VerificationError(
            error_type=VerificationErrorType.LOW_SUPPORT_STRENGTH,
            message=f"Claim {claim.claim_id} has strength {claim.support_strength}, requires {constraints.min_support_strength}",
            claim_id=claim.claim_id,
            details={
                "actual": claim.support_strength,
                "required": constraints.min_support_strength,
            },
        )
    
    return None


# =============================================================================
# Coverage Rules (Intent-Specific)
# =============================================================================

def verify_required_lanes_ran(
    bundle: EvidenceBundle,
    verification: VerificationSpec,
) -> List[VerificationError]:
    """
    Verify that required lanes actually ran.
    """
    errors = []
    lanes_run = {run.lane_id for run in bundle.retrieval_runs}
    
    for required_lane in verification.required_lanes:
        if required_lane not in lanes_run:
            errors.append(VerificationError(
                error_type=VerificationErrorType.REQUIRED_LANE_MISSING,
                message=f"Required lane '{required_lane}' did not run",
                details={"required": required_lane, "ran": list(lanes_run)},
            ))
    
    return errors


def verify_lane_coverage(
    bundle: EvidenceBundle,
    verification: VerificationSpec,
) -> Tuple[bool, Dict[str, float], List[str]]:
    """
    Verify coverage thresholds for lanes.
    
    Returns:
        (met, coverage_dict, warnings)
    """
    coverage_dict = {}
    warnings = []
    
    for run in bundle.retrieval_runs:
        # Check minimum hits
        if run.hit_count < verification.min_hits_per_lane:
            warnings.append(
                f"Lane '{run.lane_id}' has low hit count: {run.hit_count}"
            )
        
        # Record coverage achieved
        coverage_dict[run.lane_id] = run.coverage_achieved
    
    # Check doc diversity
    total_docs = sum(run.doc_count for run in bundle.retrieval_runs)
    if total_docs < verification.min_doc_diversity:
        warnings.append(
            f"Low doc diversity: {total_docs} < {verification.min_doc_diversity}"
        )
    
    coverage_dict["total_docs"] = total_docs
    
    # Determine if coverage is sufficient
    met = total_docs >= verification.min_doc_diversity
    
    return met, coverage_dict, warnings


def can_make_negative_statement(
    bundle: EvidenceBundle,
    intent: IntentFamily,
    verification: VerificationSpec,
) -> Tuple[bool, str]:
    """
    Check if we can make a global negative statement.
    
    For "no evidence found" statements, we need:
    - Required lanes ran
    - Coverage thresholds met
    - No claims found
    
    Returns:
        (can_make, reason)
    """
    # Check required lanes ran
    lanes_run = {run.lane_id for run in bundle.retrieval_runs}
    for required_lane in verification.required_lanes:
        if required_lane not in lanes_run:
            return False, f"Cannot make negative statement: lane '{required_lane}' did not run"
    
    # Check coverage
    met, coverage, _ = verify_lane_coverage(bundle, verification)
    if not met:
        return False, f"Cannot make negative statement: insufficient coverage"
    
    # If we have claims, can't make negative statement
    if bundle.claims:
        return False, "Cannot make negative statement: claims exist"
    
    return True, "Coverage sufficient for negative statement"


# =============================================================================
# Main Verification Function
# =============================================================================

def verify_evidence_bundle(
    bundle: EvidenceBundle,
    conn=None,
) -> VerificationResult:
    """
    Verify an evidence bundle against all rules.
    
    1. Check universal rules (evidence refs, quote spans)
    2. Check intent-specific coverage requirements
    3. Check constraint-specific rules (collection scope, role evidence)
    4. Return actionable errors for retry
    
    Args:
        bundle: The evidence bundle to verify
        conn: Optional database connection for collection lookup
        
    Returns:
        VerificationResult with pass/fail, errors, warnings, dropped claims
    """
    errors = []
    warnings = []
    dropped_claims = []
    
    # Get plan and constraints
    if not bundle.plan:
        return VerificationResult(
            passed=False,
            errors=[VerificationError(
                error_type=VerificationErrorType.MISSING_EVIDENCE,
                message="Bundle has no plan",
            )],
        )
    
    constraints = bundle.constraints or bundle.plan.constraints
    verification = bundle.plan.verification
    intent = bundle.plan.intent
    
    # 1. Universal rules
    universal_errors = verify_universal_rules(bundle)
    errors.extend(universal_errors)
    
    # 2. Required lanes ran
    lane_errors = verify_required_lanes_ran(bundle, verification)
    errors.extend(lane_errors)
    
    # 3. Coverage verification
    coverage_met, coverage_dict, coverage_warnings = verify_lane_coverage(
        bundle, verification
    )
    warnings.extend(coverage_warnings)
    
    if not coverage_met:
        errors.append(VerificationError(
            error_type=VerificationErrorType.INSUFFICIENT_COVERAGE,
            message="Coverage thresholds not met",
            details=coverage_dict,
        ))
    
    # 4. Per-claim verification
    verified_claims = []
    
    for claim in bundle.claims:
        claim_errors = []
        
        # Collection scope
        if constraints:
            error = verify_collection_scope(
                claim, constraints.collection_scope, conn
            )
            if error:
                claim_errors.append(error)
        
        # Role evidence (claim-scoped!)
        if constraints:
            error = verify_claim_role_evidence(claim, constraints)
            if error:
                claim_errors.append(error)
                dropped_claims.append((claim, "missing_role_evidence"))
                continue
        
        # Support type
        if constraints:
            error = verify_support_type(claim, constraints)
            if error:
                claim_errors.append(error)
                dropped_claims.append((claim, "support_type_mismatch"))
                continue
        
        if claim_errors:
            errors.extend(claim_errors)
        else:
            verified_claims.append(claim)
    
    # Determine overall pass/fail
    # Critical errors that cause failure
    critical_error_types = {
        VerificationErrorType.REQUIRED_LANE_MISSING,
        VerificationErrorType.INSUFFICIENT_COVERAGE,
    }
    
    has_critical_error = any(
        e.error_type in critical_error_types for e in errors
    )
    
    # Pass if no critical errors (claim-level errors result in dropped claims, not failure)
    passed = not has_critical_error
    
    return VerificationResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        dropped_claims=dropped_claims,
        coverage_achieved=coverage_dict,
    )


# =============================================================================
# Verification Helpers
# =============================================================================

def filter_claims_by_verification(
    claims: List[Claim],
    constraints: BundleConstraints,
) -> Tuple[List[Claim], List[Tuple[Claim, str]]]:
    """
    Filter claims based on verification rules.
    
    Returns:
        (verified_claims, dropped_claims_with_reasons)
    """
    verified = []
    dropped = []
    
    for claim in claims:
        # Check role evidence
        if constraints.required_role_evidence_patterns:
            relationship_predicates = {Predicate.HANDLED_BY, Predicate.MET_WITH}
            if claim.predicate in relationship_predicates:
                if not verify_role_evidence(claim, constraints.required_role_evidence_patterns):
                    dropped.append((claim, "missing_role_evidence"))
                    continue
        
        # Check support type
        if constraints.support_type_required:
            if claim.support_type != constraints.support_type_required:
                dropped.append((claim, f"requires {constraints.support_type_required.value}"))
                continue
        
        # Check support strength
        if claim.support_strength < constraints.min_support_strength:
            dropped.append((claim, f"strength {claim.support_strength} < {constraints.min_support_strength}"))
            continue
        
        verified.append(claim)
    
    return verified, dropped


def get_verification_summary(result: VerificationResult) -> str:
    """
    Get a human-readable summary of verification result.
    """
    if result.passed:
        summary = "Verification PASSED"
    else:
        summary = "Verification FAILED"
    
    if result.errors:
        summary += f"\n  Errors: {len(result.errors)}"
        for error in result.errors[:3]:
            summary += f"\n    - {error.message}"
    
    if result.warnings:
        summary += f"\n  Warnings: {len(result.warnings)}"
    
    if result.dropped_claims:
        summary += f"\n  Dropped claims: {len(result.dropped_claims)}"
        for claim, reason in result.dropped_claims[:3]:
            summary += f"\n    - {claim.claim_id}: {reason}"
    
    if result.coverage_achieved:
        summary += f"\n  Coverage: {result.coverage_achieved}"
    
    return summary
