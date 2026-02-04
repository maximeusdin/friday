"""
V3 Summarizer - Structured, cited summaries from evidence spans.

Key improvements over V1/V2:
1. Summarizes cite_spans (reranked), not raw chunks
2. Doc-balanced sampling to avoid clustering
3. Structured output with citations (bullets + span indices)
4. Response shapes (roster, timeline, narrative, key_docs)
5. Every sentence must cite evidence - "if you can't cite it, don't say it"
6. Graceful fallback when evidence is insufficient

Usage:
    from retrieval.agent.v3_summarizer import summarize_from_evidence_set
    
    result = summarize_from_evidence_set(
        evidence_set=evidence_set,
        query="Who were members of the Silvermaster network?",
        response_shape="roster",  # or "narrative", "timeline", "key_docs"
        conn=conn,
    )
    
    for bullet in result.bullets:
        print(f"- {bullet.text} [{', '.join(bullet.citation_ids)}]")
"""

import os
import sys
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from retrieval.agent.v3_evidence import EvidenceSet, EvidenceSpan
from retrieval.agent.v3_claims import EvidenceRef


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SummaryBullet:
    """A single bullet point with citations."""
    text: str
    citation_indices: List[int]  # Indices into the spans list
    citation_ids: List[str]      # Resolved span_ids
    confidence: str              # "supported" | "suggestive"
    bullet_type: str             # "finding" | "entity" | "uncertainty" | "gap"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "citation_indices": self.citation_indices,
            "citation_ids": self.citation_ids,
            "confidence": self.confidence,
            "bullet_type": self.bullet_type,
        }


@dataclass
class SummarySection:
    """A section of the summary (e.g., 'Core Findings', 'Named Individuals')."""
    heading: str
    bullets: List[SummaryBullet]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "heading": self.heading,
            "bullets": [b.to_dict() for b in self.bullets],
        }


@dataclass
class V3SummaryResult:
    """Result from V3 summarization."""
    response_shape: str
    sections: List[SummarySection]
    fallback_message: Optional[str]  # Set if evidence was insufficient
    total_spans_used: int
    unique_docs: int
    verification_passed: bool
    verification_errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_shape": self.response_shape,
            "sections": [s.to_dict() for s in self.sections],
            "fallback_message": self.fallback_message,
            "total_spans_used": self.total_spans_used,
            "unique_docs": self.unique_docs,
            "verification_passed": self.verification_passed,
            "verification_errors": self.verification_errors,
        }
    
    def format_text(self) -> str:
        """Format as readable text."""
        lines = []
        
        if self.fallback_message:
            lines.append(self.fallback_message)
            lines.append("")
        
        for section in self.sections:
            lines.append(f"\n{section.heading}")
            lines.append("-" * len(section.heading))
            for bullet in section.bullets:
                citations = f"[{', '.join(bullet.citation_ids[:2])}]" if bullet.citation_ids else ""
                marker = "•" if bullet.confidence == "supported" else "○"
                lines.append(f"  {marker} {bullet.text} {citations}")
        
        return "\n".join(lines)


# =============================================================================
# Response Shapes
# =============================================================================

RESPONSE_SHAPES = {
    "roster": {
        "description": "List of people/entities with citations",
        "sections": ["People named in the evidence", "People mentioned but unclear role"],
        "prompt_hint": "Output a list of named individuals/entities, each with a brief role/context quote and citation.",
    },
    "timeline": {
        "description": "Events organized chronologically",
        "sections": ["Chronological findings", "Undated events"],
        "prompt_hint": "Output events in chronological order, each with date and citation.",
    },
    "narrative": {
        "description": "Flowing summary of key findings",
        "sections": ["Core findings", "Supporting evidence", "Gaps and uncertainties"],
        "prompt_hint": "Output a narrative summary with each sentence citing evidence.",
    },
    "key_docs": {
        "description": "Summary organized by key documents",
        "sections": ["Key documents and their contents", "Cross-document patterns"],
        "prompt_hint": "Output findings organized by source document, with citations.",
    },
}


# =============================================================================
# Doc-Balanced Sampling
# =============================================================================

def doc_balanced_sample(
    spans: List[EvidenceSpan],
    max_total: int = 60,
    max_per_doc: int = 5,
) -> List[EvidenceSpan]:
    """
    Select spans with doc-balanced sampling to avoid clustering.
    
    Algorithm:
    1. Group spans by doc_id
    2. Take top K per doc (by score)
    3. Fill remaining with highest-ranked across all docs
    
    This gives the model a more representative set of evidence.
    """
    if not spans:
        return []
    
    # Group by doc_id
    by_doc: Dict[int, List[EvidenceSpan]] = defaultdict(list)
    for span in spans:
        by_doc[span.doc_id].append(span)
    
    # Sort each doc's spans by score (highest first)
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda s: -s.score)
    
    selected: List[EvidenceSpan] = []
    selected_ids: Set[str] = set()
    
    # Round 1: Take top K from each doc
    for doc_id, doc_spans in by_doc.items():
        for span in doc_spans[:max_per_doc]:
            if span.span_id not in selected_ids:
                selected.append(span)
                selected_ids.add(span.span_id)
    
    # Round 2: Fill remaining with highest-ranked overall
    if len(selected) < max_total:
        remaining = [s for s in spans if s.span_id not in selected_ids]
        remaining.sort(key=lambda s: -s.score)
        
        for span in remaining:
            if len(selected) >= max_total:
                break
            if span.span_id not in selected_ids:
                selected.append(span)
                selected_ids.add(span.span_id)
    
    # Final sort by score for prompt ordering
    selected.sort(key=lambda s: -s.score)
    
    return selected[:max_total]


# =============================================================================
# Prompt Building
# =============================================================================

def build_summary_prompt(
    query: str,
    spans: List[EvidenceSpan],
    response_shape: str,
    max_bullets: int = 15,
) -> str:
    """Build prompt for structured summary generation."""
    
    shape_config = RESPONSE_SHAPES.get(response_shape, RESPONSE_SHAPES["narrative"])
    
    # Format spans with indices
    spans_text = []
    for i, span in enumerate(spans):
        spans_text.append(
            f"[S{i}] (doc:{span.doc_id}, {span.page_ref})\n{span.quote[:400]}..."
        )
    
    spans_formatted = "\n\n".join(spans_text)
    
    return f"""Analyze the evidence spans to answer this query.

QUERY: {query}

RESPONSE FORMAT: {response_shape} - {shape_config['description']}
{shape_config['prompt_hint']}

EVIDENCE SPANS (cite by index S0, S1, etc.):
{spans_formatted}

CRITICAL RULES:
1. Every bullet/sentence MUST end with 1-2 citations like [S3] or [S1, S7]
2. If you cannot cite it from the provided spans, DO NOT say it
3. Maximum {max_bullets} bullets total
4. Use confidence "supported" for explicit evidence, "suggestive" for implied

SECTIONS TO OUTPUT:
{chr(10).join(f"- {s}" for s in shape_config['sections'])}

OUTPUT FORMAT (JSON):
{{
  "sections": [
    {{
      "heading": "Section name",
      "bullets": [
        {{
          "text": "Finding text",
          "citations": ["S0", "S3"],
          "confidence": "supported",
          "bullet_type": "finding"
        }}
      ]
    }}
  ],
  "insufficient_evidence": false,
  "fallback_message": null
}}

If the spans don't clearly support answering the query, set:
- insufficient_evidence: true
- fallback_message: "I don't have enough cited evidence to answer confidently. Here are the most relevant spans found: ..."
- Still output any partial findings in sections

OUTPUT:"""


# =============================================================================
# Summary Generation
# =============================================================================

def summarize_from_evidence_set(
    evidence_set: EvidenceSet,
    query: str,
    conn,
    response_shape: str = "narrative",
    max_spans: int = 60,
    max_per_doc: int = 5,
    max_bullets: int = 15,
    verify: bool = True,
) -> V3SummaryResult:
    """
    Generate a structured, cited summary from an EvidenceSet.
    
    Args:
        evidence_set: EvidenceSet with cite_spans
        query: User's query
        conn: Database connection
        response_shape: "roster", "timeline", "narrative", or "key_docs"
        max_spans: Maximum spans to include in prompt
        max_per_doc: Maximum spans per document (for balancing)
        max_bullets: Maximum bullets to generate
        verify: Whether to verify citations
    
    Returns:
        V3SummaryResult with sections and bullets
    """
    if not evidence_set.cite_spans:
        return V3SummaryResult(
            response_shape=response_shape,
            sections=[],
            fallback_message="No evidence spans available to summarize.",
            total_spans_used=0,
            unique_docs=0,
            verification_passed=True,
            verification_errors=[],
        )
    
    # Doc-balanced sampling
    selected_spans = doc_balanced_sample(
        evidence_set.cite_spans,
        max_total=max_spans,
        max_per_doc=max_per_doc,
    )
    
    if not selected_spans:
        return V3SummaryResult(
            response_shape=response_shape,
            sections=[],
            fallback_message="No evidence spans selected after balancing.",
            total_spans_used=0,
            unique_docs=0,
            verification_passed=True,
            verification_errors=[],
        )
    
    print(f"  [Summarizer] Selected {len(selected_spans)} spans from {len(set(s.doc_id for s in selected_spans))} docs", file=sys.stderr)
    
    # Build prompt
    prompt = build_summary_prompt(
        query=query,
        spans=selected_spans,
        response_shape=response_shape,
        max_bullets=max_bullets,
    )
    
    # Call LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_summary(selected_spans, query, response_shape)
    
    try:
        from openai import OpenAI
        
        model = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research assistant. Output JSON only. Every statement must cite evidence."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2000,
        )
        
        content = response.choices[0].message.content
        if not content:
            return _fallback_summary(selected_spans, query, response_shape)
        
        # Parse response
        data = json.loads(content)
        
        # Build result
        sections = []
        for section_data in data.get("sections", []):
            bullets = []
            for bullet_data in section_data.get("bullets", []):
                # Parse citation indices
                citation_strs = bullet_data.get("citations", [])
                citation_indices = []
                citation_ids = []
                
                for cit in citation_strs:
                    # Parse "S3" -> 3
                    if isinstance(cit, str) and cit.startswith("S"):
                        try:
                            idx = int(cit[1:])
                            if 0 <= idx < len(selected_spans):
                                citation_indices.append(idx)
                                citation_ids.append(selected_spans[idx].span_id)
                        except ValueError:
                            pass
                    elif isinstance(cit, int) and 0 <= cit < len(selected_spans):
                        citation_indices.append(cit)
                        citation_ids.append(selected_spans[cit].span_id)
                
                bullets.append(SummaryBullet(
                    text=bullet_data.get("text", ""),
                    citation_indices=citation_indices,
                    citation_ids=citation_ids,
                    confidence=bullet_data.get("confidence", "suggestive"),
                    bullet_type=bullet_data.get("bullet_type", "finding"),
                ))
            
            sections.append(SummarySection(
                heading=section_data.get("heading", "Findings"),
                bullets=bullets,
            ))
        
        # Handle insufficient evidence
        fallback_message = None
        if data.get("insufficient_evidence"):
            fallback_message = data.get("fallback_message", "Evidence may be insufficient to fully answer the query.")
        
        # Verify if requested
        verification_errors = []
        verification_passed = True
        if verify:
            verification_errors = _verify_summary(sections, selected_spans)
            verification_passed = len(verification_errors) == 0
        
        return V3SummaryResult(
            response_shape=response_shape,
            sections=sections,
            fallback_message=fallback_message,
            total_spans_used=len(selected_spans),
            unique_docs=len(set(s.doc_id for s in selected_spans)),
            verification_passed=verification_passed,
            verification_errors=verification_errors,
        )
        
    except Exception as e:
        print(f"  [Summarizer] Error: {e}", file=sys.stderr)
        return _fallback_summary(selected_spans, query, response_shape)


def _fallback_summary(
    spans: List[EvidenceSpan],
    query: str,
    response_shape: str,
) -> V3SummaryResult:
    """Generate fallback summary when LLM is unavailable."""
    bullets = []
    
    for i, span in enumerate(spans[:10]):
        preview = span.quote[:150].strip()
        preview = ' '.join(preview.split())
        
        bullets.append(SummaryBullet(
            text=f"Evidence from {span.page_ref}: \"{preview}...\"",
            citation_indices=[i],
            citation_ids=[span.span_id],
            confidence="suggestive",
            bullet_type="finding",
        ))
    
    return V3SummaryResult(
        response_shape=response_shape,
        sections=[SummarySection(heading="Top Evidence Spans", bullets=bullets)],
        fallback_message="LLM unavailable. Showing top evidence spans.",
        total_spans_used=len(spans),
        unique_docs=len(set(s.doc_id for s in spans)),
        verification_passed=True,
        verification_errors=[],
    )


def _verify_summary(
    sections: List[SummarySection],
    spans: List[EvidenceSpan],
) -> List[str]:
    """Verify that all bullets have valid citations."""
    errors = []
    
    for section in sections:
        for i, bullet in enumerate(section.bullets):
            # Check citation existence
            if not bullet.citation_indices:
                errors.append(f"Bullet '{bullet.text[:30]}...' has no citations")
                continue
            
            # Check citation validity
            for idx in bullet.citation_indices:
                if idx < 0 or idx >= len(spans):
                    errors.append(f"Invalid citation index {idx} in bullet '{bullet.text[:30]}...'")
    
    return errors


# =============================================================================
# Response Shape Detection (Optional)
# =============================================================================

def detect_response_shape(query: str) -> str:
    """
    Auto-detect the best response shape based on query.
    
    This is a simple heuristic - the model can also choose.
    """
    query_lower = query.lower()
    
    # Roster indicators
    roster_words = ["who", "members", "people", "individuals", "agents", "list"]
    if any(word in query_lower for word in roster_words):
        return "roster"
    
    # Timeline indicators
    timeline_words = ["when", "timeline", "chronology", "sequence", "dates", "history"]
    if any(word in query_lower for word in timeline_words):
        return "timeline"
    
    # Key docs indicators
    doc_words = ["documents", "reports", "files", "sources", "which document"]
    if any(word in query_lower for word in doc_words):
        return "key_docs"
    
    # Default to narrative
    return "narrative"


# =============================================================================
# Integration with V3 Runner
# =============================================================================

def summarize_v3_result(
    result: "V3Result",
    conn,
    response_shape: Optional[str] = None,
) -> V3SummaryResult:
    """
    Generate summary from a V3Result.
    
    Args:
        result: V3Result from V3Runner
        conn: Database connection
        response_shape: Optional shape override (auto-detected if None)
    
    Returns:
        V3SummaryResult
    """
    query = result.plan.query_text if result.plan else ""
    
    # Auto-detect shape if not provided
    if response_shape is None:
        response_shape = detect_response_shape(query)
    
    return summarize_from_evidence_set(
        evidence_set=result.evidence_set,
        query=query,
        conn=conn,
        response_shape=response_shape,
    )


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from retrieval.agent.v3_runner import V3Result
