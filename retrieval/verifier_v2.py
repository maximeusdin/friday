"""
Verifier V2 for Agentic V2.

Enforces the two core invariants:
1. All citations must come from FocusBundle spans
2. All rendered statements must map to supported evidence

Also enforces:
- Per-constraint citation overlap (Contract C8)
- Claim strength verification (strong verbs require supporting language)
- Atom grounding (entities, dates, numbers must appear in cited spans)
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.focus_bundle import FocusBundle, FocusSpan
    from retrieval.query_intent import ConstraintSpec
    from retrieval.constraints import CandidateAssessment


@dataclass
class VerificationResult:
    """Result of verification."""
    passed: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, any]


@dataclass
class Bullet:
    """A rendered bullet point with citations."""
    text: str
    cited_span_ids: List[str]
    confidence: str = "high"  # "high", "medium", "low"
    candidate_key: str = ""


class FocusBundleVerifier:
    """
    Enforces FocusBundle invariants and constraint support.
    
    Invariants:
    1. All citations must be in FocusBundle
    2. All rendered statements must map to supported evidence
    
    Additional checks:
    - Contract C8: Each required constraint has at least one cited span
    - Claim strength: Strong verbs require supporting language in spans
    - Atom grounding: Key facts must appear in cited spans
    """
    
    # Strong claims that require supporting language
    STRONG_CLAIMS = [
        (r'\bhandled\b', "relationship verb 'handled'"),
        (r'\brecruited\b', "relationship verb 'recruited'"),
        (r'\bwas an? (?:Soviet )?agent\b', "role claim 'agent'"),
        (r'\bwas an? (?:Soviet )?spy\b', "role claim 'spy'"),
        (r'\bpassed (?:secrets|information|documents)\b', "action 'passed secrets'"),
        (r'\bstole\b', "action 'stole'"),
        (r'\bbetrayed\b', "action 'betrayed'"),
    ]
    
    def verify_citation(
        self,
        span_id: str,
        focus_bundle: "FocusBundle",
    ) -> bool:
        """Check if a citation is valid (Invariant #1)."""
        return focus_bundle.contains_span(span_id)
    
    def verify_bullet_grounding(
        self,
        bullet_text: str,
        cited_spans: List["FocusSpan"],
    ) -> Tuple[bool, List[str]]:
        """
        Verify bullet is grounded in cited spans (Invariant #2).
        
        Cheap deterministic "atom coverage" check:
        1. Extract key "atoms" from bullet (entities, dates, numbers, quoted phrases)
        2. Require each atom to appear in at least one cited span (normalized)
        3. Fail with structured error listing missing atoms
        
        Args:
            bullet_text: The bullet text to verify
            cited_spans: The FocusSpans cited by this bullet
        
        Returns:
            (passed, errors)
        """
        atoms = self._extract_atoms(bullet_text)
        
        if not atoms:
            return True, []  # No atoms to verify
        
        if not cited_spans:
            if atoms:
                return False, [f"No cited spans but bullet contains atoms: {atoms}"]
            return True, []
        
        cited_text = " ".join(s.text for s in cited_spans)
        cited_norm = _normalize_for_match(cited_text)
        
        missing = []
        for atom in atoms:
            atom_norm = _normalize_for_match(atom)
            if atom_norm not in cited_norm:
                missing.append(atom)
        
        if missing:
            return False, [f"Ungrounded atoms: {missing}"]
        
        return True, []
    
    def _extract_atoms(self, text: str) -> List[str]:
        """
        Extract verifiable atoms from bullet text.
        
        Atoms are factual elements that must be grounded:
        - Named entities (Title Case sequences)
        - Years (4-digit numbers)
        - Full dates
        - Numbers with context
        - Quoted phrases
        """
        atoms = []
        
        # Named entities (Title Case sequences, 2+ words)
        # e.g., "Julius Rosenberg", "State Department"
        atoms.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text))
        
        # Years (4-digit numbers, likely 1900-2100)
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        atoms.extend(years)
        
        # Full dates
        # e.g., "January 15, 1943"
        dates = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b',
            text
        )
        atoms.extend(dates)
        
        # Numbers with context
        # e.g., "50 percent", "$500", "12 people"
        numbers = re.findall(
            r'\b\d+(?:\.\d+)?\s*(?:percent|%|dollars|\$|people|members|documents|pages)\b',
            text, re.I
        )
        atoms.extend(numbers)
        
        # Quoted phrases
        quoted = re.findall(r'"([^"]{3,50})"', text)
        atoms.extend(quoted)
        
        return atoms
    
    def verify_claim_strength(
        self,
        bullet: Bullet,
        focus_bundle: "FocusBundle",
    ) -> List[str]:
        """
        Verify strong claims have supporting language in cited spans.
        
        Strong claims (e.g., "handled", "recruited", "was an agent") require
        the same or similar language to appear in the cited spans.
        
        Args:
            bullet: The bullet to verify
            focus_bundle: The FocusBundle for span lookups
        
        Returns:
            List of errors
        """
        errors = []
        
        # Get cited text
        cited_spans = [
            focus_bundle.get_span(sid) 
            for sid in bullet.cited_span_ids 
            if focus_bundle.get_span(sid)
        ]
        
        if not cited_spans:
            return errors  # No spans to check against
        
        cited_text = " ".join(s.text for s in cited_spans)
        cited_lower = cited_text.lower()
        
        for pattern, claim_type in self.STRONG_CLAIMS:
            if re.search(pattern, bullet.text, re.I):
                # Check if cited spans contain supporting language
                if not re.search(pattern, cited_lower):
                    errors.append(f"Strong claim '{claim_type}' not supported by cited spans")
        
        return errors
    
    def verify_constraint_citations(
        self,
        bullet: Bullet,
        assessment: "CandidateAssessment",
        constraints: List["ConstraintSpec"],
        focus_bundle: "FocusBundle",
    ) -> List[str]:
        """
        Verify each required constraint has at least one cited span (Contract C8).
        
        Args:
            bullet: The bullet to verify
            assessment: The CandidateAssessment for this bullet
            constraints: The constraints from QueryContract
            focus_bundle: The FocusBundle
        
        Returns:
            List of errors
        """
        errors = []
        cited_set = set(bullet.cited_span_ids)
        
        for constraint in constraints:
            # Check constraints that require citation
            if constraint.strength == "required" or constraint.min_score > 0:
                # Find support for this constraint
                support = next(
                    (s for s in assessment.supports 
                     if s.constraint_name == constraint.constraint_key),
                    None
                )
                
                if support:
                    # Check overlap between cited and supporting spans
                    supporting_set = set(support.supporting_span_ids)
                    overlap = cited_set & supporting_set
                    
                    if not overlap:
                        errors.append(
                            f"Constraint '{constraint.constraint_key}' has no cited span "
                            f"(cited: {cited_set}, supporting: {supporting_set})"
                        )
        
        return errors
    
    def verify_bullet(
        self,
        bullet: Bullet,
        focus_bundle: "FocusBundle",
        assessment: "CandidateAssessment" = None,
        constraints: List["ConstraintSpec"] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Verify a single bullet point.
        
        Checks:
        1. All cited spans are in FocusBundle (Invariant #1)
        2. Atoms are grounded in cited spans (Invariant #2)
        3. Strong claims have supporting language
        4. Constraint citations are valid (if assessment provided)
        
        Args:
            bullet: The bullet to verify
            focus_bundle: The FocusBundle
            assessment: Optional CandidateAssessment for constraint checks
            constraints: Optional constraints for constraint checks
        
        Returns:
            (passed, errors)
        """
        errors = []
        
        # Invariant #1: All citations must be in FocusBundle
        for span_id in bullet.cited_span_ids:
            if not self.verify_citation(span_id, focus_bundle):
                errors.append(f"Citation {span_id} not in FocusBundle")
        
        # Invariant #2: Atoms must be grounded
        cited_spans = [
            focus_bundle.get_span(sid) 
            for sid in bullet.cited_span_ids 
            if focus_bundle.get_span(sid)
        ]
        passed, atom_errors = self.verify_bullet_grounding(bullet.text, cited_spans)
        errors.extend(atom_errors)
        
        # Claim strength check
        strength_errors = self.verify_claim_strength(bullet, focus_bundle)
        errors.extend(strength_errors)
        
        # Constraint citation check (Contract C8)
        if assessment and constraints:
            constraint_errors = self.verify_constraint_citations(
                bullet, assessment, constraints, focus_bundle
            )
            errors.extend(constraint_errors)
        
        return len(errors) == 0, errors
    
    def verify_answer(
        self,
        bullets: List[Bullet],
        focus_bundle: "FocusBundle",
        assessments: List["CandidateAssessment"] = None,
        constraints: List["ConstraintSpec"] = None,
    ) -> VerificationResult:
        """
        Verify entire answer against FocusBundle.
        
        Args:
            bullets: List of Bullet objects
            focus_bundle: The FocusBundle
            assessments: Optional list of CandidateAssessments (parallel to bullets)
            constraints: Optional constraints from QueryContract
        
        Returns:
            VerificationResult with errors, warnings, and stats
        """
        all_errors = []
        all_warnings = []
        stats = {
            "total_bullets": len(bullets),
            "bullets_with_citations": 0,
            "total_citations": 0,
            "unique_span_ids": set(),
            "constraint_errors": 0,
            "anchor_consistency": True,
        }
        
        assessments = assessments or [None] * len(bullets)
        
        for i, bullet in enumerate(bullets):
            assessment = assessments[i] if i < len(assessments) else None
            
            # Count stats
            if bullet.cited_span_ids:
                stats["bullets_with_citations"] += 1
                stats["total_citations"] += len(bullet.cited_span_ids)
                stats["unique_span_ids"].update(bullet.cited_span_ids)
            
            # Verify bullet
            passed, errors = self.verify_bullet(
                bullet, focus_bundle, assessment, constraints
            )
            
            for error in errors:
                all_errors.append(f"Bullet {i+1}: {error}")
                if "constraint" in error.lower():
                    stats["constraint_errors"] += 1
        
        # Check coverage (all bullets should have citations)
        if stats["bullets_with_citations"] < len(bullets):
            missing = len(bullets) - stats["bullets_with_citations"]
            all_warnings.append(f"{missing} bullets have no citations")
        
        # Check evidence density
        if bullets and stats["total_citations"] > 0:
            avg_citations = stats["total_citations"] / len(bullets)
            if avg_citations < 1.0:
                all_warnings.append(f"Low evidence density: {avg_citations:.2f} citations/bullet")
            elif avg_citations > 3.0:
                all_warnings.append(f"High evidence density: {avg_citations:.2f} citations/bullet")
        
        # Check doc diversity
        if focus_bundle.spans:
            unique_docs = len({s.doc_id for s in focus_bundle.spans})
            if unique_docs < 3:
                all_warnings.append(f"Low doc diversity: only {unique_docs} unique documents")
        
        # ANCHOR CONSISTENCY CHECK:
        # If anchor hits exist in FocusBundle but we rendered nothing -> pipeline is broken
        anchor_hit_count = focus_bundle.params.get("anchor_hit_count", 0)
        anchor_terms = focus_bundle.params.get("anchor_terms", [])
        
        if anchor_terms and anchor_hit_count > 0 and len(bullets) == 0:
            # Anchor hits exist but nothing rendered - this is a bug
            all_errors.append(
                f"Anchor consistency violation: {anchor_hit_count} spans have anchor hits "
                f"for {anchor_terms} but 0 bullets rendered. Pipeline should have surfaced these."
            )
            stats["anchor_consistency"] = False
        
        # Check if any rendered bullets cite anchor-hit spans
        if bullets and anchor_terms:
            cited_span_ids = set()
            for b in bullets:
                cited_span_ids.update(b.cited_span_ids)
            
            anchor_hits_cited = 0
            for span in focus_bundle.spans:
                if span.span_id in cited_span_ids:
                    text_lower = span.text.lower()
                    if any(a in text_lower for a in anchor_terms):
                        anchor_hits_cited += 1
            
            if anchor_hit_count > 0 and anchor_hits_cited == 0:
                all_warnings.append(
                    f"No anchor-hit spans cited: {anchor_hit_count} spans have anchors but none cited"
                )
        
        stats["unique_span_ids"] = len(stats["unique_span_ids"])
        
        return VerificationResult(
            passed=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            stats=stats,
        )
    
    def verify_with_retry(
        self,
        bullets: List[Bullet],
        focus_bundle: "FocusBundle",
        fix_fn=None,
        max_retries: int = 1,
    ) -> Tuple[List[Bullet], VerificationResult]:
        """
        Verify with optional retry/fix.
        
        If verification fails and fix_fn is provided, attempt to fix
        the errors and retry verification.
        
        Args:
            bullets: List of Bullet objects
            focus_bundle: The FocusBundle
            fix_fn: Optional function to fix errors: (bullets, errors) -> fixed_bullets
            max_retries: Maximum retry attempts
        
        Returns:
            (final_bullets, verification_result)
        """
        current_bullets = bullets
        
        for attempt in range(max_retries + 1):
            result = self.verify_answer(current_bullets, focus_bundle)
            
            if result.passed or not fix_fn or attempt >= max_retries:
                return current_bullets, result
            
            # Try to fix errors
            current_bullets = fix_fn(current_bullets, result.errors)
        
        return current_bullets, result


def _normalize_for_match(text: str) -> str:
    """Normalize text for matching."""
    return ' '.join(text.lower().split())


def create_bullet(
    text: str,
    span_ids: List[str],
    confidence: str = "high",
    candidate_key: str = "",
) -> Bullet:
    """Helper to create a Bullet object."""
    return Bullet(
        text=text,
        cited_span_ids=span_ids,
        confidence=confidence,
        candidate_key=candidate_key,
    )
