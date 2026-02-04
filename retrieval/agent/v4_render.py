"""
V4 Renderer - Grounded response builder from verified interpretations.

The renderer takes verified InterpretationV4 and outputs:
- Final answer prose with citations
- Warnings as badges (optional)
- Sections organized by response_shape

Rendering Rules:
- DROP units that fail hard verification
- DOWNGRADE confidence based on soft warnings
- GROUP by response_shape (rendering preference only)
- Always include "What's unclear" section for uncertainty units

The same AnswerUnits work for any shape - just different organization.

Usage:
    from retrieval.agent.v4_render import render_interpretation
    
    response = render_interpretation(
        interpretation=interpretation,
        report=verification_report,
        prepared_spans=prepared_spans,
    )
    
    print(response.format_text())
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from retrieval.agent.v4_interpret import (
    InterpretationV4,
    AnswerUnit,
    PreparedSpan,
    DiagnosticsInfo,
)
from retrieval.agent.v4_verify import V4VerificationReport, UnitVerificationStatus


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class RenderedCitation:
    """A citation rendered for display."""
    span_id: str
    doc_id: int
    page_ref: str
    quote_preview: str  # First 100 chars of quote
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "doc_id": self.doc_id,
            "page_ref": self.page_ref,
            "quote_preview": self.quote_preview,
        }


@dataclass
class RenderedUnit:
    """A rendered answer unit with resolved citations."""
    unit_id: str
    text: str
    confidence: str
    citations: List[RenderedCitation]
    supporting_phrases: List[str]
    badges: List[str]  # Warning badges (e.g., "alias-only", "low-overlap")
    is_uncertainty: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "confidence": self.confidence,
            "citations": [c.to_dict() for c in self.citations],
            "supporting_phrases": self.supporting_phrases,
            "badges": self.badges,
            "is_uncertainty": self.is_uncertainty,
        }


@dataclass
class RenderedSection:
    """A section of the rendered response."""
    heading: str
    units: List[RenderedUnit]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "heading": self.heading,
            "units": [u.to_dict() for u in self.units],
        }


@dataclass
class RenderStats:
    """Statistics about rendering."""
    total_units: int
    units_rendered: int
    units_dropped: int
    units_downgraded: int
    citations_rendered: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_units": self.total_units,
            "units_rendered": self.units_rendered,
            "units_dropped": self.units_dropped,
            "units_downgraded": self.units_downgraded,
            "citations_rendered": self.citations_rendered,
        }


@dataclass
class V4RenderedResponse:
    """Complete rendered response from V4 interpretation."""
    response_shape: str
    sections: List[RenderedSection]
    warnings: List[str]  # Visible warnings
    diagnostics: DiagnosticsInfo
    stats: RenderStats
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_shape": self.response_shape,
            "sections": [s.to_dict() for s in self.sections],
            "warnings": self.warnings,
            "diagnostics": self.diagnostics.to_dict(),
            "stats": self.stats.to_dict(),
        }
    
    def format_text(self) -> str:
        """Format as readable text output."""
        lines = []
        
        # Show warnings at top
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings[:3]:
                lines.append(f"    - {w}")
            lines.append("")
        
        # Render sections
        for section in self.sections:
            lines.append(f"\n{section.heading}")
            lines.append("-" * len(section.heading))
            
            for unit in section.units:
                # Choose marker based on confidence
                if unit.is_uncertainty:
                    marker = "?"
                elif unit.confidence == "supported":
                    marker = "•"
                else:
                    marker = "○"
                
                # Format citations
                cit_str = ""
                if unit.citations:
                    cit_refs = [f"{c.page_ref}" for c in unit.citations[:2]]
                    cit_str = f" [{', '.join(cit_refs)}]"
                
                # Format badges
                badge_str = ""
                if unit.badges:
                    badge_str = f" ({', '.join(unit.badges)})"
                
                lines.append(f"  {marker} {unit.text}{cit_str}{badge_str}")
                
                # Show supporting phrases if present
                if unit.supporting_phrases and not unit.is_uncertainty:
                    for phrase in unit.supporting_phrases[:2]:
                        lines.append(f"      \"{phrase[:60]}...\"")
        
        # Show diagnostics
        if self.diagnostics.missing_info_questions:
            lines.append("\n  Questions the evidence doesn't answer:")
            for q in self.diagnostics.missing_info_questions[:3]:
                lines.append(f"    - {q}")
        
        if self.diagnostics.followup_queries:
            lines.append("\n  Suggested followup searches:")
            for q in self.diagnostics.followup_queries[:3]:
                lines.append(f"    - {q}")
        
        # Show stats
        lines.append(f"\n  --- Rendering Stats ---")
        lines.append(f"  Rendered: {self.stats.units_rendered} units")
        if self.stats.units_dropped > 0:
            lines.append(f"  Dropped: {self.stats.units_dropped} units (failed verification)")
        if self.stats.units_downgraded > 0:
            lines.append(f"  Downgraded: {self.stats.units_downgraded} units")
        lines.append(f"  Citations: {self.stats.citations_rendered}")
        
        return "\n".join(lines)


# =============================================================================
# Rendering Implementation
# =============================================================================

class V4Renderer:
    """
    Renderer for V4 interpretations.
    
    Takes verified interpretation and produces grounded response.
    Response shape is purely a rendering preference - same units, different organization.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def render(
        self,
        interpretation: InterpretationV4,
        report: V4VerificationReport,
        prepared_spans: List[PreparedSpan],
    ) -> V4RenderedResponse:
        """
        Render an interpretation into final response.
        
        Args:
            interpretation: The InterpretationV4 to render
            report: Verification report
            prepared_spans: Prepared spans for citation lookup
        
        Returns:
            V4RenderedResponse
        """
        if self.verbose:
            print(f"  [V4 Render] Rendering {len(interpretation.answer_units)} units...", file=sys.stderr)
        
        # Build span lookup
        span_lookup = {ps.span_idx: ps for ps in prepared_spans}
        
        # Collect warnings for display
        visible_warnings = []
        if report.stats.get("error_count", 0) > 0:
            visible_warnings.append(f"{report.stats['error_count']} verification errors")
        if report.downgraded_units:
            visible_warnings.append(f"{len(report.downgraded_units)} units downgraded to suggestive")
        
        # Render units that passed or were downgraded
        rendered_units: List[RenderedUnit] = []
        dropped_count = 0
        downgraded_count = 0
        total_citations = 0
        
        for unit in interpretation.answer_units:
            unit_status = report.per_unit_status.get(unit.unit_id)
            
            if unit_status and unit_status.status == "failed":
                dropped_count += 1
                continue  # Skip failed units
            
            # Check if downgraded
            final_confidence = unit.confidence
            badges = []
            if unit_status:
                final_confidence = unit_status.final_confidence
                if unit_status.status == "downgraded":
                    downgraded_count += 1
                
                # Add badges for warnings
                for warning in unit_status.warnings:
                    if warning.warning_type == "alias_only":
                        badges.append("alias-only")
                    elif warning.warning_type == "low_overlap":
                        badges.append("low-overlap")
                    elif warning.warning_type == "span_overuse":
                        badges.append("overused-span")
                    elif warning.warning_type == "unit_too_long":
                        badges.append("long")
            
            # Render citations
            citations = []
            for cit in unit.citations:
                if cit.span_idx in span_lookup:
                    ps = span_lookup[cit.span_idx]
                    citations.append(RenderedCitation(
                        span_id=ps.span.span_id,
                        doc_id=ps.span.doc_id,
                        page_ref=ps.span.page_ref,
                        quote_preview=ps.span.quote[:100],
                    ))
                    total_citations += 1
            
            rendered_units.append(RenderedUnit(
                unit_id=unit.unit_id,
                text=unit.text,
                confidence=final_confidence,
                citations=citations,
                supporting_phrases=unit.supporting_phrases,
                badges=badges,
                is_uncertainty=unit.is_uncertainty,
            ))
        
        # Organize into sections by response_shape
        sections = self._organize_sections(
            rendered_units,
            interpretation.response_shape,
        )
        
        # Build stats
        stats = RenderStats(
            total_units=len(interpretation.answer_units),
            units_rendered=len(rendered_units),
            units_dropped=dropped_count,
            units_downgraded=downgraded_count,
            citations_rendered=total_citations,
        )
        
        if self.verbose:
            print(f"    Rendered {len(rendered_units)} units, dropped {dropped_count}", file=sys.stderr)
        
        # Check for empty result - will be handled by fallback if needed
        response = V4RenderedResponse(
            response_shape=interpretation.response_shape,
            sections=sections,
            warnings=visible_warnings,
            diagnostics=interpretation.diagnostics,
            stats=stats,
        )
        
        # Safety valve: if nothing rendered, return fallback
        if stats.units_rendered == 0 and prepared_spans:
            return self._render_fallback(prepared_spans, interpretation.diagnostics)
        
        return response
    
    def _organize_sections(
        self,
        units: List[RenderedUnit],
        response_shape: str,
    ) -> List[RenderedSection]:
        """Organize units into sections based on response shape."""
        
        # Separate uncertainty units - they always go to their own section
        grounded_units = [u for u in units if not u.is_uncertainty]
        uncertainty_units = [u for u in units if u.is_uncertainty]
        
        # Organize grounded units by shape
        if response_shape == "roster":
            sections = self._organize_roster(grounded_units)
        elif response_shape == "timeline":
            sections = self._organize_timeline(grounded_units)
        elif response_shape == "index":
            sections = self._organize_index(grounded_units)
        else:  # narrative, qa, or default
            sections = self._organize_narrative(grounded_units)
        
        # Always add uncertainty section if there are uncertainty units
        if uncertainty_units:
            sections.append(RenderedSection(
                heading="What's unclear",
                units=uncertainty_units,
            ))
        
        return sections
    
    def _organize_roster(self, units: List[RenderedUnit]) -> List[RenderedSection]:
        """Organize units for roster shape - group by entity if possible."""
        sections = []
        
        # Split by confidence
        supported = [u for u in units if u.confidence == "supported"]
        suggestive = [u for u in units if u.confidence == "suggestive"]
        
        if supported:
            sections.append(RenderedSection(
                heading="Confirmed (supported by evidence)",
                units=supported,
            ))
        
        if suggestive:
            sections.append(RenderedSection(
                heading="Possible (suggestive evidence)",
                units=suggestive,
            ))
        
        return sections
    
    def _organize_timeline(self, units: List[RenderedUnit]) -> List[RenderedSection]:
        """Organize units for timeline shape - try to sort by date."""
        sections = []
        
        # Simple date extraction (look for years)
        def extract_year(unit: RenderedUnit) -> int:
            match = re.search(r'\b(19\d{2}|20\d{2})\b', unit.text)
            return int(match.group(1)) if match else 9999
        
        # Sort by year
        sorted_units = sorted(units, key=extract_year)
        
        if sorted_units:
            sections.append(RenderedSection(
                heading="Timeline",
                units=sorted_units,
            ))
        
        return sections
    
    def _organize_index(self, units: List[RenderedUnit]) -> List[RenderedSection]:
        """Organize units for index shape."""
        sections = []
        
        if units:
            sections.append(RenderedSection(
                heading="Documents and sources",
                units=units,
            ))
        
        return sections
    
    def _organize_narrative(self, units: List[RenderedUnit]) -> List[RenderedSection]:
        """Organize units for narrative/qa shape - by confidence."""
        sections = []
        
        # Primary findings (supported)
        supported = [u for u in units if u.confidence == "supported"]
        if supported:
            sections.append(RenderedSection(
                heading="Key findings",
                units=supported,
            ))
        
        # Suggestive findings
        suggestive = [u for u in units if u.confidence == "suggestive"]
        if suggestive:
            sections.append(RenderedSection(
                heading="Additional context (suggestive)",
                units=suggestive,
            ))
        
        return sections
    
    def _render_fallback(
        self, 
        prepared_spans: List[PreparedSpan],
        diagnostics: DiagnosticsInfo,
    ) -> V4RenderedResponse:
        """
        Return top evidence when interpretation fails.
        
        Safety valve: never return 0 units to the user.
        """
        if self.verbose:
            print(f"    [Fallback] Returning raw evidence (interpretation failed)", file=sys.stderr)
        
        fallback_units = []
        for ps in prepared_spans[:10]:
            entities_str = ""
            if ps.entities_in_span:
                names = [e.get("canonical_name", "") for e in ps.entities_in_span[:3]]
                entities_str = f" (mentions: {', '.join(names)})"
            
            fallback_units.append(RenderedUnit(
                unit_id=f"fallback_{ps.span_idx}",
                text=f"[{ps.span.page_ref}] \"{ps.span.quote[:200]}...\"{entities_str}",
                confidence="raw_evidence",
                citations=[RenderedCitation(
                    span_id=ps.span.span_id,
                    doc_id=ps.span.doc_id,
                    page_ref=ps.span.page_ref,
                    quote_preview=ps.span.quote[:100],
                )],
                supporting_phrases=[],
                badges=["raw-evidence"],
                is_uncertainty=False,
            ))
        
        return V4RenderedResponse(
            response_shape="fallback",
            sections=[RenderedSection(
                heading="Top Evidence (interpretation failed - showing raw evidence)",
                units=fallback_units,
            )],
            warnings=["All answer units failed verification. Showing raw evidence instead."],
            diagnostics=diagnostics,
            stats=RenderStats(
                total_units=0, 
                units_rendered=len(fallback_units), 
                units_dropped=0, 
                units_downgraded=0, 
                citations_rendered=len(fallback_units),
            ),
        )


# =============================================================================
# Convenience Function
# =============================================================================

def render_interpretation(
    interpretation: InterpretationV4,
    report: V4VerificationReport,
    prepared_spans: List[PreparedSpan],
    **kwargs,
) -> V4RenderedResponse:
    """
    Render an interpretation into final response.
    
    Convenience wrapper around V4Renderer.
    """
    renderer = V4Renderer(**kwargs)
    return renderer.render(interpretation, report, prepared_spans)
