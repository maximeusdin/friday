"""
Constraint-Aware Rendering for Agentic V2.

Renders answers from FocusBundle with:
- Constraint-aware citation allocation (Contract C8)
- Conservative language for weak support
- Proper grounding in FocusSpans only (Invariant #1)

Rules:
1. Only render if required constraints are satisfied (score >= threshold)
2. Each bullet cites constraint-supporting spans
3. Conservative language based on strength of support
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.focus_bundle import FocusBundle
    from retrieval.query_intent import ConstraintSpec
    from retrieval.constraints import CandidateAssessment, ConstraintSupport
    from retrieval.verifier_v2 import Bullet


@dataclass
class RenderedAnswer:
    """The final rendered answer."""
    short_answer: str
    bullets: List["Bullet"]
    focus_bundle_id: Optional[int]
    total_candidates: int
    rendered_count: int
    negative_findings: Optional[str] = None


def render_from_focus_bundle_with_constraints(
    focus_bundle: "FocusBundle",
    assessments: List["CandidateAssessment"],
    constraints: List["ConstraintSpec"],
    max_items: int = 25,
    max_citations_per_item: int = 2,
    conservative_language: bool = True,
    hubness_scores: List = None,  # CandidateScore list for evidence linkage check
) -> RenderedAnswer:
    """
    Render answer from FocusBundle using constraint scorer's supporting spans.
    
    INVARIANT: Only render items that have DIRECT evidence linkage to FocusSpans.
    If candidate has no supporting spans -> NOT renderable.
    
    RULES (Contract C8):
    1. Only render if required constraints are satisfied (score >= threshold)
    2. Only render if candidate has direct evidence linkage (supporting spans)
    3. Allocate citations per constraint (1 span per constraint)
    4. Conservative language for weak support (hedging)
    5. Each bullet cites only FocusSpans
    
    FALLBACK: If no candidates are renderable, show the best evidence spans directly.
    
    Args:
        focus_bundle: The FocusBundle (source of truth)
        assessments: List of CandidateAssessment from constraint scoring
        constraints: List of ConstraintSpec from QueryContract
        max_items: Maximum items to render
        max_citations_per_item: Maximum citations per item
        conservative_language: Use hedging for weak support
        hubness_scores: Optional list of CandidateScore with source_span_ids for linkage check
    
    Returns:
        RenderedAnswer with bullets and metadata
    """
    import sys
    from retrieval.verifier_v2 import Bullet
    
    # Build a map from candidate_key to source_span_ids for evidence linkage check
    evidence_linkage = {}
    if hubness_scores:
        for hs in hubness_scores:
            evidence_linkage[hs.candidate_key] = hs.source_span_ids
    
    bullets = []
    skipped_no_evidence = 0
    
    for assessment in assessments[:max_items * 2]:  # Check more than we need
        if len(bullets) >= max_items:
            break
        
        # CRITICAL: Check evidence linkage FIRST
        # If candidate has no supporting spans, it is NOT renderable
        source_spans = evidence_linkage.get(assessment.candidate_key, [])
        if not source_spans:
            skipped_no_evidence += 1
            continue
        
        # Check if required constraints are satisfied
        constraint_satisfied, weak_constraints = _check_constraints(
            assessment, constraints
        )
        
        if not constraint_satisfied:
            continue
        
        # Allocate citations from the candidate's supporting spans
        # NOT from unrelated spans
        cited_spans = []
        for sid in source_spans[:max_citations_per_item]:
            if focus_bundle.contains_span(sid):
                cited_spans.append(sid)
        
        if not cited_spans:
            # Candidate has source_spans but none are in FocusBundle (shouldn't happen)
            skipped_no_evidence += 1
            continue
        
        # Build bullet text
        if conservative_language and weak_constraints:
            bullet_text = _build_conservative_bullet(
                assessment.display_name,
                weak_constraints,
                focus_bundle,
                cited_spans,
            )
            confidence = "medium"
        else:
            bullet_text = _build_standard_bullet(
                assessment.display_name,
                focus_bundle,
                cited_spans,
            )
            confidence = "high"
        
        bullets.append(Bullet(
            text=bullet_text,
            cited_span_ids=cited_spans,
            confidence=confidence,
            candidate_key=assessment.candidate_key,
        ))
    
    if skipped_no_evidence > 0:
        print(f"    [Rendering] Skipped {skipped_no_evidence} candidates with no evidence linkage", file=sys.stderr)
    
    # Build short answer
    short_answer = _build_short_answer(bullets, constraints)
    
    # Build negative findings if no results
    negative_findings = None
    if not bullets:
        negative_findings = _build_negative_findings(focus_bundle, constraints)
    
    # FALLBACK: If no candidates are renderable, show evidence spans directly
    # This handles keyword/existence queries where we want to show the evidence itself
    if not bullets and focus_bundle.spans:
        print(f"    [Rendering] No candidates renderable, falling back to evidence spans", file=sys.stderr)
        bullets = _render_evidence_spans(focus_bundle, max_items)
        if bullets:
            short_answer = f"Found {len(bullets)} relevant passages."
            negative_findings = None
    
    return RenderedAnswer(
        short_answer=short_answer,
        bullets=bullets,
        focus_bundle_id=focus_bundle.retrieval_run_id,
        total_candidates=len(assessments),
        rendered_count=len(bullets),
        negative_findings=negative_findings,
    )


def _render_evidence_spans(
    focus_bundle: "FocusBundle",
    max_items: int,
) -> List["Bullet"]:
    """
    Render top spans as evidence bullets for keyword queries.
    
    Used when no constraints are specified and no entity candidates match.
    PRIORITIZES spans that contain anchor terms (query-relevant).
    """
    import sys
    from retrieval.verifier_v2 import Bullet
    
    # Get anchor terms from bundle params
    anchor_terms = focus_bundle.params.get("anchor_terms", [])
    
    # Separate spans with anchor hits from those without
    anchor_hit_spans = []
    other_spans = []
    
    for span in focus_bundle.spans:
        text_lower = span.text.lower()
        has_anchor = any(a in text_lower for a in anchor_terms)
        if has_anchor:
            anchor_hit_spans.append(span)
        else:
            other_spans.append(span)
    
    print(f"    [Rendering] Evidence spans: {len(anchor_hit_spans)} with anchor hits, {len(other_spans)} without", file=sys.stderr)
    
    # Prioritize anchor-hit spans, then fill with others
    prioritized_spans = anchor_hit_spans + other_spans
    
    bullets = []
    for span in prioritized_spans[:max_items]:
        # Create a bullet with the span text (truncated)
        text = span.text[:300].strip()
        if len(span.text) > 300:
            text += "..."
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Mark anchor-hit spans
        text_lower = span.text.lower()
        has_anchor = any(a in text_lower for a in anchor_terms)
        confidence = "high" if has_anchor else "medium"
        
        bullets.append(Bullet(
            text=f"[{span.page_ref}] {text}",
            cited_span_ids=[span.span_id],
            confidence=confidence,
            candidate_key=f"span:{span.span_id}",
        ))
    
    return bullets


def _check_constraints(
    assessment: "CandidateAssessment",
    constraints: List["ConstraintSpec"],
) -> tuple:
    """
    Check if all required constraints are satisfied.
    
    Returns (satisfied: bool, weak_constraints: List[str])
    """
    weak_constraints = []
    
    for constraint in constraints:
        support = next(
            (s for s in assessment.supports 
             if s.constraint_name == constraint.constraint_key),
            None
        )
        
        if support:
            # Check minimum score
            if support.score < constraint.min_score:
                return False, []
            
            # Track weak constraints (score < 0.5)
            if support.score < 0.5:
                weak_constraints.append(constraint.constraint_key)
        elif constraint.strength == "required":
            # Required constraint has no support
            return False, []
    
    return True, weak_constraints


def _allocate_citations(
    assessment: "CandidateAssessment",
    constraints: List["ConstraintSpec"],
    max_citations: int,
    focus_bundle: "FocusBundle" = None,
    conn = None,
) -> List[str]:
    """
    Allocate citations per constraint (Contract C8).
    
    Picks 1 span per constraint (up to max_citations), ensuring each
    required constraint has at least one cited span from its supporting spans.
    
    For keyword queries (no constraints), allocates from mention_span_index.
    """
    cited_spans = []
    cited_set = set()
    
    # First pass: allocate one span per constraint
    for constraint in constraints:
        if len(cited_spans) >= max_citations:
            break
        
        support = next(
            (s for s in assessment.supports 
             if s.constraint_name == constraint.constraint_key),
            None
        )
        
        if support and support.supporting_span_ids:
            # Pick best span not already cited
            for sid in support.supporting_span_ids:
                if sid not in cited_set:
                    cited_spans.append(sid)
                    cited_set.add(sid)
                    break
    
    # Second pass: fill remaining slots with best supporting spans
    if len(cited_spans) < max_citations:
        for support in assessment.supports:
            for sid in support.supporting_span_ids:
                if sid not in cited_set:
                    cited_spans.append(sid)
                    cited_set.add(sid)
                    if len(cited_spans) >= max_citations:
                        break
            if len(cited_spans) >= max_citations:
                break
    
    # Third pass: For keyword queries (no constraints), use mention_span_index
    if len(cited_spans) < max_citations and focus_bundle:
        # Get spans for this entity from mention_span_index
        if assessment.candidate_entity_id:
            span_ids = focus_bundle.mention_span_index.get(assessment.candidate_entity_id, [])
            for sid in span_ids:
                if sid not in cited_set:
                    cited_spans.append(sid)
                    cited_set.add(sid)
                    if len(cited_spans) >= max_citations:
                        break
        
        # For unresolved candidates, use term matching
        if len(cited_spans) < max_citations and not assessment.candidate_entity_id:
            candidate_name = assessment.display_name
            for span in focus_bundle.spans:
                if candidate_name.lower() in span.text.lower():
                    if span.span_id not in cited_set:
                        cited_spans.append(span.span_id)
                        cited_set.add(span.span_id)
                        if len(cited_spans) >= max_citations:
                            break
    
    return cited_spans


def _build_conservative_bullet(
    display_name: str,
    weak_constraints: List[str],
    focus_bundle: "FocusBundle",
    span_ids: List[str],
) -> str:
    """
    Build bullet with hedging for weak evidence.
    
    Examples:
    - "X (possibly connected to query subjects)"
    - "X (reported as a possible Soviet source)"
    - "X, mentioned in connection with OSS"
    """
    # Determine hedge based on weak constraints
    hedges = []
    
    for constraint_key in weak_constraints:
        if "role" in constraint_key and "agent" in constraint_key.lower():
            hedges.append("reported as a possible source")
        elif "role" in constraint_key:
            hedges.append("possibly serving in this role")
        elif "affiliation" in constraint_key:
            hedges.append("mentioned in connection with")
        elif "relationship" in constraint_key:
            hedges.append("possibly connected to")
    
    if hedges:
        hedge_text = hedges[0]  # Use first relevant hedge
        return f"{display_name} ({hedge_text})"
    else:
        return f"{display_name} (evidence limited)"


def _build_standard_bullet(
    display_name: str,
    focus_bundle: "FocusBundle",
    span_ids: List[str],
) -> str:
    """Build standard bullet text with strong evidence."""
    return display_name


def _build_short_answer(
    bullets: List["Bullet"],
    constraints: List["ConstraintSpec"],
) -> str:
    """Build short answer summarizing the results."""
    if not bullets:
        return "No candidates found meeting the criteria."
    
    count = len(bullets)
    
    # Build constraint description
    constraint_desc = ""
    if constraints:
        constraint_names = [c.name for c in constraints]
        if "relationship" in constraint_names:
            constraint_desc = " with the specified relationship"
        elif "affiliation" in constraint_names:
            constraint_desc = " with the specified affiliation"
    
    if count == 1:
        return f"Found 1 candidate{constraint_desc}."
    else:
        return f"Found {count} candidates{constraint_desc}."


def _build_negative_findings(
    focus_bundle: "FocusBundle",
    constraints: List["ConstraintSpec"],
) -> str:
    """Build explanation for no results."""
    parts = []
    
    # Report what was searched
    parts.append(f"Searched {len(focus_bundle.spans)} spans from {len(focus_bundle.get_unique_doc_ids())} documents.")
    
    # Report constraints that weren't satisfied
    if constraints:
        constraint_names = [f"{c.name}:{c.object}" if c.object else c.name for c in constraints]
        parts.append(f"No candidates satisfied constraints: {', '.join(constraint_names)}")
    
    parts.append("This is not proof of absence - evidence may exist in unsearched documents.")
    
    return " ".join(parts)


def format_bullet_with_citations(
    bullet: "Bullet",
    focus_bundle: "FocusBundle",
    include_span_text: bool = True,
    max_span_chars: int = 200,
) -> str:
    """
    Format a bullet with its citations for display.
    
    Args:
        bullet: The Bullet to format
        focus_bundle: The FocusBundle for span lookups
        include_span_text: Include span text in output
        max_span_chars: Maximum characters to show from each span
    
    Returns:
        Formatted string with bullet and citations
    """
    lines = [f"- {bullet.text}"]
    
    if bullet.confidence != "high":
        lines[0] += f" [{bullet.confidence} confidence]"
    
    if include_span_text:
        for sid in bullet.cited_span_ids:
            span = focus_bundle.get_span(sid)
            if span:
                text = span.text[:max_span_chars]
                if len(span.text) > max_span_chars:
                    text += "..."
                lines.append(f"  [{span.page_ref}] \"{text}\"")
    else:
        lines.append(f"  Citations: {', '.join(bullet.cited_span_ids)}")
    
    return "\n".join(lines)


def render_answer_text(
    answer: RenderedAnswer,
    focus_bundle: "FocusBundle",
    include_citations: bool = True,
    max_bullets: int = 20,
) -> str:
    """
    Render full answer as text.
    
    Args:
        answer: The RenderedAnswer
        focus_bundle: The FocusBundle for span lookups
        include_citations: Include citation text
        max_bullets: Maximum bullets to show
    
    Returns:
        Formatted answer text
    """
    parts = []
    
    # Short answer
    parts.append(answer.short_answer)
    parts.append("")
    
    # Bullets
    if answer.bullets:
        parts.append("Results:")
        for bullet in answer.bullets[:max_bullets]:
            parts.append(format_bullet_with_citations(
                bullet, focus_bundle, include_citations
            ))
        
        if len(answer.bullets) > max_bullets:
            remaining = len(answer.bullets) - max_bullets
            parts.append(f"  ... and {remaining} more")
    
    # Negative findings
    if answer.negative_findings:
        parts.append("")
        parts.append(answer.negative_findings)
    
    return "\n".join(parts)
