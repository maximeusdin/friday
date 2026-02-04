"""
V4 Verification - Universal hard errors and soft warnings.

This is the TRUST BOUNDARY. The verifier enforces mechanical truth constraints
that apply to ANY answer unit, regardless of response shape.

Philosophy: The verifier checks contracts, not "understanding."
It avoids trying to interpret the world. Shape-specific logic (roster anchors,
relationship tags) belongs to the model, not the verifier.

Hard Errors (block the unit):
- Missing citations (for non-uncertainty units)
- Invalid span_idx (out of bounds)
- Supporting phrase missing (verbatim not in cited quotes)
- Entity attestation failure (if about_entities provided)

Soft Warnings (downgrade confidence):
- Alias-only attestation
- Low lexical overlap
- Span overuse (same span cited by many units)
- Unit too long (exceeds 2 sentences)

Usage:
    from retrieval.agent.v4_verify import verify_interpretation
    
    report = verify_interpretation(
        interpretation=interpretation,
        prepared_spans=prepared_spans,
        conn=conn,
    )
    
    if report.passed:
        render(interpretation)
    else:
        retry_or_render_partial(report.passed_units, report.warnings)
"""

import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set

from retrieval.agent.v4_interpret import (
    InterpretationV4,
    AnswerUnit,
    SpanCitation,
    PreparedSpan,
)
from retrieval.agent.entity_surfaces import EntitySurfaceIndex, normalize_surface


# =============================================================================
# Constants
# =============================================================================

# Maximum sentences allowed in a unit before warning
MAX_SENTENCES_PER_UNIT = 2

# Threshold for span overuse warning
SPAN_OVERUSE_THRESHOLD = 3

# Threshold for low lexical overlap
LOW_OVERLAP_THRESHOLD = 0.3


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class VerificationError:
    """A hard verification error that blocks a unit."""
    error_type: str
    unit_id: str
    details: str
    actionable: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "unit_id": self.unit_id,
            "details": self.details,
            "actionable": self.actionable,
        }


@dataclass
class VerificationWarning:
    """A soft warning that may downgrade confidence."""
    warning_type: str
    unit_id: str
    details: str
    downgrade_confidence: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "warning_type": self.warning_type,
            "unit_id": self.unit_id,
            "details": self.details,
            "downgrade_confidence": self.downgrade_confidence,
        }


@dataclass
class UnitVerificationStatus:
    """Verification status for a single answer unit."""
    unit_id: str
    status: str  # passed|failed|downgraded
    errors: List[VerificationError]
    warnings: List[VerificationWarning]
    final_confidence: str  # After any downgrades
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "status": self.status,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "final_confidence": self.final_confidence,
        }


@dataclass
class V4VerificationReport:
    """Complete verification report for an interpretation."""
    passed: bool
    total_units: int
    passed_units: List[str]  # unit_ids that passed
    failed_units: List[str]  # unit_ids that failed
    downgraded_units: List[str]  # unit_ids with downgraded confidence
    hard_errors: List[VerificationError]
    soft_warnings: List[VerificationWarning]
    per_unit_status: Dict[str, UnitVerificationStatus]
    stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "total_units": self.total_units,
            "passed_units": self.passed_units,
            "failed_units": self.failed_units,
            "downgraded_units": self.downgraded_units,
            "hard_errors": [e.to_dict() for e in self.hard_errors],
            "soft_warnings": [w.to_dict() for w in self.soft_warnings],
            "per_unit_status": {k: v.to_dict() for k, v in self.per_unit_status.items()},
            "stats": self.stats,
        }
    
    def get_error_messages(self) -> List[str]:
        """Get error messages for retry prompt."""
        return [f"[{e.error_type}] {e.details}" for e in self.hard_errors]


# =============================================================================
# Verification Implementation
# =============================================================================

class V4Verifier:
    """
    Universal verifier for V4 interpretations.
    
    Performs hard checks (block unit) and soft checks (downgrade confidence).
    All checks are universal - no shape-specific logic.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._surface_index: Optional[EntitySurfaceIndex] = None
    
    def verify(
        self,
        interpretation: InterpretationV4,
        prepared_spans: List[PreparedSpan],
        conn,
    ) -> V4VerificationReport:
        """
        Verify an interpretation against the evidence.
        
        Args:
            interpretation: The InterpretationV4 to verify
            prepared_spans: The prepared spans used for interpretation
            conn: Database connection (for entity surfaces)
        
        Returns:
            V4VerificationReport with pass/fail status and details
        """
        if self.verbose:
            print(f"\n  [V4 Verify] Checking {len(interpretation.answer_units)} units...", file=sys.stderr)
        
        # Initialize entity surface index
        self._surface_index = EntitySurfaceIndex(conn)
        
        # Preload all entities referenced in units
        all_entity_ids = []
        for unit in interpretation.answer_units:
            all_entity_ids.extend(unit.about_entities)
        if all_entity_ids:
            self._surface_index.preload_entities(list(set(all_entity_ids)))
        
        # Build span lookup
        span_lookup = {ps.span_idx: ps for ps in prepared_spans}
        
        # Count span usage across all units (for overuse detection)
        span_usage = Counter()
        for unit in interpretation.answer_units:
            for cit in unit.citations:
                span_usage[cit.span_idx] += 1
        
        all_errors: List[VerificationError] = []
        all_warnings: List[VerificationWarning] = []
        per_unit_status: Dict[str, UnitVerificationStatus] = {}
        passed_units: List[str] = []
        failed_units: List[str] = []
        downgraded_units: List[str] = []
        
        # Check each unit
        for unit in interpretation.answer_units:
            unit_errors = []
            unit_warnings = []
            
            # Hard checks (universal)
            unit_errors.extend(self._check_citations(unit, span_lookup))
            unit_errors.extend(self._check_supporting_phrases(unit, span_lookup))
            unit_errors.extend(self._check_entity_attestation(unit, span_lookup))
            
            # Soft checks (universal)
            unit_warnings.extend(self._check_alias_only(unit, span_lookup))
            unit_warnings.extend(self._check_lexical_overlap(unit, span_lookup))
            unit_warnings.extend(self._check_span_overuse(unit, span_usage))
            unit_warnings.extend(self._check_unit_length(unit))
            
            all_errors.extend(unit_errors)
            all_warnings.extend(unit_warnings)
            
            # Determine unit status
            if unit_errors:
                status = "failed"
                failed_units.append(unit.unit_id)
                final_confidence = unit.confidence
            elif any(w.downgrade_confidence for w in unit_warnings):
                status = "downgraded"
                downgraded_units.append(unit.unit_id)
                final_confidence = "suggestive"
            else:
                status = "passed"
                passed_units.append(unit.unit_id)
                final_confidence = unit.confidence
            
            per_unit_status[unit.unit_id] = UnitVerificationStatus(
                unit_id=unit.unit_id,
                status=status,
                errors=unit_errors,
                warnings=unit_warnings,
                final_confidence=final_confidence,
            )
        
        # Compute pass status (at least some units passed)
        passed = len(failed_units) == 0 or len(passed_units) > 0
        
        # Build stats
        stats = {
            "total_units": len(interpretation.answer_units),
            "passed_count": len(passed_units),
            "failed_count": len(failed_units),
            "downgraded_count": len(downgraded_units),
            "error_count": len(all_errors),
            "warning_count": len(all_warnings),
        }
        
        if self.verbose:
            status_str = "PASSED" if len(all_errors) == 0 else f"PARTIAL ({len(all_errors)} errors)"
            print(f"    Verification: {status_str}", file=sys.stderr)
            if all_errors:
                for e in all_errors[:3]:
                    print(f"      [{e.error_type}] {e.details[:60]}...", file=sys.stderr)
        
        return V4VerificationReport(
            passed=passed,
            total_units=len(interpretation.answer_units),
            passed_units=passed_units,
            failed_units=failed_units,
            downgraded_units=downgraded_units,
            hard_errors=all_errors,
            soft_warnings=all_warnings,
            per_unit_status=per_unit_status,
            stats=stats,
        )
    
    def _check_citations(
        self,
        unit: AnswerUnit,
        span_lookup: Dict[int, PreparedSpan],
    ) -> List[VerificationError]:
        """Check citation validity (hard errors)."""
        errors = []
        
        # Uncertainty units (suggestive + no citations) are valid
        if unit.is_uncertainty:
            return []
        
        # Units with "supported" confidence MUST have citations
        # Units with "suggestive" confidence and citations are also grounded
        if not unit.citations:
            if unit.confidence == "supported":
                errors.append(VerificationError(
                    error_type="missing_citations",
                    unit_id=unit.unit_id,
                    details=f"Supported unit has no citations: \"{unit.text[:50]}...\"",
                    actionable=True,
                ))
            # suggestive with no citations = valid uncertainty (caught by is_uncertainty above)
            return errors
        
        # Check each citation index is valid
        for cit in unit.citations:
            if cit.span_idx not in span_lookup:
                errors.append(VerificationError(
                    error_type="invalid_span_idx",
                    unit_id=unit.unit_id,
                    details=f"Citation references invalid span index {cit.span_idx}",
                    actionable=True,
                ))
        
        return errors
    
    def _check_supporting_phrases(
        self,
        unit: AnswerUnit,
        span_lookup: Dict[int, PreparedSpan],
    ) -> List[VerificationError]:
        """
        Check that supporting_phrases appear in cited evidence.
        
        V4.2 Enhancement: Use attest_text (extended context window) for more
        robust phrase matching, with OCR-tolerant normalization.
        """
        errors = []
        
        # Uncertainty units don't need supporting phrases
        if unit.is_uncertainty:
            return []
        
        # If no supporting_phrases provided, that's a soft issue (handled elsewhere)
        if not unit.supporting_phrases:
            return []
        
        # Gather attest_text from cited spans (includes quote + context window)
        # Fall back to quote if attest_text not available
        combined_attest = ""
        for cit in unit.citations:
            if cit.span_idx in span_lookup:
                ps = span_lookup[cit.span_idx]
                # Prefer attest_text (has context window), fall back to quote
                text = ps.attest_text if ps.attest_text else ps.span.quote
                combined_attest += " " + text
        
        if not combined_attest:
            return []  # Will be caught by missing_citations
        
        # Normalize for OCR-tolerant matching
        combined_normalized = self._normalize_for_matching(combined_attest)
        
        # Check each supporting phrase appears
        for phrase in unit.supporting_phrases:
            phrase_normalized = self._normalize_for_matching(phrase)
            
            if not self._phrase_matches(phrase_normalized, combined_normalized):
                errors.append(VerificationError(
                    error_type="supporting_phrase_missing",
                    unit_id=unit.unit_id,
                    details=f"Supporting phrase not found in cited quotes: \"{phrase[:50]}...\"",
                    actionable=True,
                ))
        
        return errors
    
    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize text for OCR-tolerant phrase matching.
        
        - Lowercase
        - Collapse whitespace
        - Remove punctuation except apostrophes
        - Handle common OCR confusions (l/I, 0/O)
        """
        import re
        import unicodedata
        
        if not text:
            return ""
        
        # Unicode normalize
        text = unicodedata.normalize('NFKD', text)
        
        # Lowercase
        text = text.lower()
        
        # Collapse whitespace
        text = ' '.join(text.split())
        
        # Remove most punctuation but keep apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _phrase_matches(self, phrase: str, text: str) -> bool:
        """
        Check if a phrase matches in text, with fuzzy tolerance.
        
        Handles minor variations like OCR errors and whitespace.
        """
        if not phrase or not text:
            return False
        
        # Direct substring match (most common case)
        if phrase in text:
            return True
        
        # Try word-by-word matching with some slack
        # This helps when OCR inserts/removes spaces
        phrase_words = phrase.split()
        if len(phrase_words) >= 3:
            # Check if most words appear in sequence
            text_words = text.split()
            for i in range(len(text_words) - len(phrase_words) + 1):
                window = text_words[i:i + len(phrase_words)]
                matches = sum(1 for pw, tw in zip(phrase_words, window) 
                             if pw == tw or self._fuzzy_word_match(pw, tw))
                if matches >= len(phrase_words) * 0.8:  # 80% word match
                    return True
        
        return False
    
    def _fuzzy_word_match(self, w1: str, w2: str) -> bool:
        """
        Check if two words are fuzzy-equal (allowing for OCR errors).
        """
        if w1 == w2:
            return True
        
        # Very short words must match exactly
        if len(w1) <= 2 or len(w2) <= 2:
            return False
        
        # Allow single character difference for longer words
        if abs(len(w1) - len(w2)) <= 1:
            # Common OCR confusions
            ocr_equivalents = {
                'l': 'i', 'i': 'l', '1': 'l', 'l': '1',
                '0': 'o', 'o': '0',
                'rn': 'm', 'm': 'rn',
            }
            
            # Check if one is OCR variant of other
            for old, new in ocr_equivalents.items():
                if w1.replace(old, new) == w2 or w2.replace(old, new) == w1:
                    return True
        
        return False
    
    def _check_entity_attestation(
        self,
        unit: AnswerUnit,
        span_lookup: Dict[int, PreparedSpan],
    ) -> List[VerificationError]:
        """Check that entities in about_entities appear in cited evidence."""
        errors = []
        
        # Only check if about_entities is provided
        if not unit.about_entities:
            return []
        
        if not self._surface_index:
            return []
        
        # Gather all attest_text from cited spans
        attest_texts = []
        for cit in unit.citations:
            if cit.span_idx in span_lookup:
                ps = span_lookup[cit.span_idx]
                attest_texts.append(ps.attest_text)
        
        if not attest_texts:
            return []  # Will be caught by missing_citations
        
        combined_text = " ".join(attest_texts)
        
        # Check each entity
        for entity_id in unit.about_entities:
            if not self._surface_index.attests(entity_id, combined_text):
                top_surfaces = self._surface_index.get_top_surfaces(entity_id, limit=3)
                if not top_surfaces:
                    top_surfaces = ["(no surfaces found)"]
                
                errors.append(VerificationError(
                    error_type="entity_not_attested",
                    unit_id=unit.unit_id,
                    details=(
                        f"Entity {entity_id} not attested in cited evidence. "
                        f"Expected one of: {top_surfaces}"
                    ),
                    actionable=True,
                ))
        
        return errors
    
    def _check_alias_only(
        self,
        unit: AnswerUnit,
        span_lookup: Dict[int, PreparedSpan],
    ) -> List[VerificationWarning]:
        """Check if entity attestation used alias only (soft warning)."""
        warnings = []
        
        if not unit.about_entities or not self._surface_index:
            return []
        
        # Gather attest_text
        attest_texts = []
        for cit in unit.citations:
            if cit.span_idx in span_lookup:
                ps = span_lookup[cit.span_idx]
                attest_texts.append(ps.attest_text)
        
        if not attest_texts:
            return []
        
        combined_text = " ".join(attest_texts)
        combined_normalized = normalize_surface(combined_text)
        
        for entity_id in unit.about_entities:
            canonical = self._surface_index.get_canonical_name(entity_id)
            if not canonical:
                continue
            canonical_normalized = normalize_surface(canonical)
            
            # Check if canonical name is present
            if canonical_normalized and canonical_normalized not in combined_normalized:
                # Entity attested by alias only
                warnings.append(VerificationWarning(
                    warning_type="alias_only",
                    unit_id=unit.unit_id,
                    details=f"Entity {entity_id} attested by alias only, not canonical name '{canonical}'",
                    downgrade_confidence=False,  # Don't downgrade, just note
                ))
        
        return warnings
    
    def _check_lexical_overlap(
        self,
        unit: AnswerUnit,
        span_lookup: Dict[int, PreparedSpan],
    ) -> List[VerificationWarning]:
        """Check lexical overlap between unit text and cited quotes."""
        warnings = []
        
        if not unit.citations or unit.is_uncertainty:
            return []
        
        # Gather quote text
        quote_texts = []
        for cit in unit.citations:
            if cit.span_idx in span_lookup:
                ps = span_lookup[cit.span_idx]
                quote_texts.append(ps.span.quote)
        
        if not quote_texts:
            return []
        
        # Compute word overlap
        unit_words = set(normalize_surface(unit.text).split())
        quote_words = set()
        for qt in quote_texts:
            quote_words.update(normalize_surface(qt).split())
        
        if not unit_words or not quote_words:
            return []
        
        # Filter out very common words
        common_words = {'the', 'a', 'an', 'is', 'was', 'were', 'are', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'and', 'or', 'that', 'this', 'with', 'as'}
        unit_words -= common_words
        quote_words -= common_words
        
        if not unit_words:
            return []
        
        overlap = len(unit_words & quote_words) / len(unit_words)
        
        if overlap < LOW_OVERLAP_THRESHOLD:
            warnings.append(VerificationWarning(
                warning_type="low_overlap",
                unit_id=unit.unit_id,
                details=f"Low lexical overlap ({overlap:.0%}) between unit text and cited quotes",
                downgrade_confidence=True,  # Downgrade to suggestive
            ))
        
        return warnings
    
    def _check_span_overuse(
        self,
        unit: AnswerUnit,
        span_usage: Counter,
    ) -> List[VerificationWarning]:
        """Check if unit cites spans that are overused across interpretation."""
        warnings = []
        
        for cit in unit.citations:
            if span_usage[cit.span_idx] >= SPAN_OVERUSE_THRESHOLD:
                warnings.append(VerificationWarning(
                    warning_type="span_overuse",
                    unit_id=unit.unit_id,
                    details=f"Span S{cit.span_idx} cited by {span_usage[cit.span_idx]} units",
                    downgrade_confidence=False,  # Note only
                ))
                break  # One warning per unit is enough
        
        return warnings
    
    def _check_unit_length(
        self,
        unit: AnswerUnit,
    ) -> List[VerificationWarning]:
        """Check if unit text exceeds recommended length."""
        warnings = []
        
        # Count sentences (rough heuristic)
        sentence_endings = unit.text.count('.') + unit.text.count('!') + unit.text.count('?')
        
        if sentence_endings > MAX_SENTENCES_PER_UNIT:
            warnings.append(VerificationWarning(
                warning_type="unit_too_long",
                unit_id=unit.unit_id,
                details=f"Unit has ~{sentence_endings} sentences (recommended max: {MAX_SENTENCES_PER_UNIT})",
                downgrade_confidence=True,  # Long units may contain ungrounded claims
            ))
        
        return warnings


# =============================================================================
# Convenience Function
# =============================================================================

def verify_interpretation(
    interpretation: InterpretationV4,
    prepared_spans: List[PreparedSpan],
    conn,
    **kwargs,
) -> V4VerificationReport:
    """
    Verify an interpretation against evidence.
    
    Convenience wrapper around V4Verifier.
    """
    verifier = V4Verifier(**kwargs)
    return verifier.verify(interpretation, prepared_spans, conn)
