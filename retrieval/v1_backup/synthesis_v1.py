"""
Stage B: LLM Synthesis Service

Handles LLM interaction for generating structured summaries:
- Prompt building with bundle IDs
- LLM API calls with JSON mode
- Response parsing
"""

import os
import json
from typing import List, Optional

from openai import OpenAI

from .models import (
    EvidenceBundle,
    SynthesisOutput,
    ClaimOutput,
    ThemeOutput,
)
from .profiles import SummarizationProfile
from .prompts import (
    CURRENT_PROMPT_VERSION,
    CURRENT_SYSTEM_PROMPT,
    build_current_user_prompt,
)
from .bundles import format_bundles_for_prompt


# =============================================================================
# Model Configuration
# =============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2000


def get_model_config() -> dict:
    """Get model configuration from environment or defaults."""
    return {
        "name": os.getenv("OPENAI_MODEL_SUMMARY", DEFAULT_MODEL),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", DEFAULT_TEMPERATURE)),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
    }


# =============================================================================
# Prompt Building
# =============================================================================

def build_synthesis_prompt(
    bundles: List[EvidenceBundle],
    question: Optional[str],
    profile: SummarizationProfile,
) -> str:
    """
    Build the user prompt for LLM synthesis.
    
    Uses bundle IDs (B1, B2...) for citations - server maps back to chunk IDs.
    """
    bundles_text = format_bundles_for_prompt(bundles)
    
    return build_current_user_prompt(
        question=question,
        bundles_text=bundles_text,
        bundle_count=len(bundles),
        max_claims=profile.max_claims,
        include_themes=profile.output_themes,
    )


# =============================================================================
# LLM Calling
# =============================================================================

def call_synthesis_llm(
    bundles: List[EvidenceBundle],
    question: Optional[str],
    profile: SummarizationProfile,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> SynthesisOutput:
    """
    Call LLM to synthesize evidence into structured claims.
    
    Args:
        bundles: Evidence bundles to synthesize
        question: Optional user question
        profile: Summarization profile
        model_name: Override model name
        temperature: Override temperature
        max_tokens: Override max tokens
    
    Returns:
        Parsed SynthesisOutput
    
    Raises:
        RuntimeError: If OPENAI_API_KEY not set
        ValueError: If LLM response is invalid
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    config = get_model_config()
    model = model_name or config["name"]
    temp = temperature if temperature is not None else config["temperature"]
    tokens = max_tokens or config["max_tokens"]
    
    # Build prompt
    user_prompt = build_synthesis_prompt(bundles, question, profile)
    
    # Call OpenAI
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CURRENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            max_tokens=tokens,
            response_format={"type": "json_object"},  # Force JSON output
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")
    
    # Parse response
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from LLM")
    
    return parse_synthesis_response(content)


def parse_synthesis_response(content: str) -> SynthesisOutput:
    """
    Parse LLM JSON response into SynthesisOutput.
    
    Handles common LLM response variations gracefully.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}")
    
    # Parse claims
    claims = []
    for claim_data in data.get("claims", []):
        try:
            claims.append(ClaimOutput(
                claim_id=claim_data.get("claim_id", f"C{len(claims)+1}"),
                claim=claim_data.get("claim", ""),
                citations=claim_data.get("citations", []),
                confidence=claim_data.get("confidence", "medium"),
                limitations=claim_data.get("limitations"),
            ))
        except Exception:
            # Skip malformed claims
            continue
    
    # Parse themes
    themes = []
    for theme_data in data.get("themes", []):
        try:
            themes.append(ThemeOutput(
                theme=theme_data.get("theme", ""),
                description=theme_data.get("description"),
                evidence=theme_data.get("evidence", []),
            ))
        except Exception:
            continue
    
    return SynthesisOutput(
        claims=claims,
        themes=themes,
        entities_mentioned=data.get("entities_mentioned", []),
        coverage_notes=data.get("coverage_notes"),
        followups=data.get("followups", []),
    )


# =============================================================================
# Model Info
# =============================================================================

def get_model_info(model_name: Optional[str] = None) -> dict:
    """Get model information for artifact storage."""
    config = get_model_config()
    
    return {
        "name": model_name or config["name"],
        "snapshot": None,  # Would need API call to get snapshot
        "provider": "openai",
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "prompt_version": CURRENT_PROMPT_VERSION,
    }


# =============================================================================
# Agentic Workflow: Render from Evidence Bundle
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BulletWithCitation:
    """A rendered bullet point with atomic citations."""
    text: str
    entity_id: Optional[int] = None
    claims_used: List[str] = field(default_factory=list)  # claim IDs
    citations: List[Dict[str, Any]] = field(default_factory=list)  # evidence refs
    inference_level: str = "explicit"  # "explicit" | "inferential"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_id": self.entity_id,
            "claims_used": self.claims_used,
            "citations": self.citations,
            "inference_level": self.inference_level,
        }


@dataclass
class LaneSummary:
    """Summary of a retrieval lane for negative answers."""
    lane_id: str
    query_terms: List[str]
    hit_count: int
    doc_count: int
    top_matches: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "query_terms": self.query_terms,
            "hit_count": self.hit_count,
            "doc_count": self.doc_count,
            "top_matches": self.top_matches,
        }


@dataclass
class NegativeAnswerTemplate:
    """
    Template for "no claims found" cases.
    
    Avoids the "wrong no" output that triggers bad answers.
    Shows what was searched and why we can't make positive claims.
    """
    query_summary: str
    lanes_executed: List[LaneSummary]
    validated_expansions_tried: List[str]
    coverage_achieved: Dict[str, float]
    statement: str = "No explicit evidence found in retrieved material"
    caveat: str = "This is not proof of absence"
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_summary": self.query_summary,
            "lanes_executed": [l.to_dict() for l in self.lanes_executed],
            "validated_expansions_tried": self.validated_expansions_tried,
            "coverage_achieved": self.coverage_achieved,
            "statement": self.statement,
            "caveat": self.caveat,
            "suggestions": self.suggestions,
        }


@dataclass
class RenderedAnswer:
    """
    Complete rendered answer from verified evidence bundle.
    """
    short_answer: str
    bullets: List[BulletWithCitation]
    negative_answer: Optional[NegativeAnswerTemplate] = None
    answer_trace_id: Optional[str] = None
    claims_rendered: int = 0
    citations_included: int = 0
    summary: Optional[str] = None  # LLM-generated summary of findings
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "short_answer": self.short_answer,
            "bullets": [b.to_dict() for b in self.bullets],
            "negative_answer": self.negative_answer.to_dict() if self.negative_answer else None,
            "answer_trace_id": self.answer_trace_id,
            "claims_rendered": self.claims_rendered,
            "citations_included": self.citations_included,
            "summary": self.summary,
        }


def render_negative_answer(bundle) -> RenderedAnswer:
    """
    Render a negative answer when no claims were found.
    
    Shows:
    - Lanes executed + validated expansions used
    - Clear statement that no explicit evidence was found
    - Caveat that this is not proof of absence
    - Suggestions for alternative searches
    """
    from retrieval.evidence_bundle import EvidenceBundle as AgenticBundle
    
    # Build lane summaries
    lane_summaries = []
    for run in bundle.retrieval_runs:
        lane_summaries.append(LaneSummary(
            lane_id=run.lane_id,
            query_terms=run.query_terms,
            hit_count=run.hit_count,
            doc_count=run.doc_count,
            top_matches=run.top_terms_matched[:5],
        ))
    
    # Get validated expansions if available
    validated_expansions = []
    # Would extract from ephemeral expansion lane results
    
    # Build coverage dict
    coverage = {}
    for run in bundle.retrieval_runs:
        coverage[run.lane_id] = run.coverage_achieved
    
    # Build suggestions
    suggestions = []
    if bundle.plan and bundle.plan.constraints:
        if bundle.plan.constraints.collection_scope:
            suggestions.append(
                f"Try expanding search beyond {', '.join(bundle.plan.constraints.collection_scope)}"
            )
    suggestions.append("Try using different search terms or synonyms")
    suggestions.append("Consider searching for related concepts")
    
    negative = NegativeAnswerTemplate(
        query_summary=bundle.plan.query_text if bundle.plan else "",
        lanes_executed=lane_summaries,
        validated_expansions_tried=validated_expansions,
        coverage_achieved=coverage,
        suggestions=suggestions,
    )
    
    return RenderedAnswer(
        short_answer="No explicit evidence found in the retrieved material.",
        bullets=[],
        negative_answer=negative,
        claims_rendered=0,
        citations_included=0,
    )


def render_from_bundle(bundle) -> RenderedAnswer:
    """
    Generate answer from verified evidence bundle ONLY.
    
    STRICT RULES:
    - Every statement must cite evidence from bundle
    - Per-bullet citations, not per-section
    - Cannot generate facts not present in bundle claims
    - Group claims by entity when rendering roster answers
    
    If bundle.claims is empty, uses NegativeAnswerTemplate.
    
    Args:
        bundle: The verified EvidenceBundle from agentic workflow
        
    Returns:
        RenderedAnswer with bullets, citations, and optional negative template
    """
    from retrieval.evidence_bundle import (
        EvidenceBundle as AgenticBundle,
        Predicate,
        SupportType,
    )
    from retrieval.intent import IntentFamily
    
    # Handle empty claims case
    if not bundle.claims:
        return render_negative_answer(bundle)
    
    # Build bullets from claims
    bullets = []
    total_citations = 0
    
    # Group claims by entity for roster queries
    intent = bundle.plan.intent if bundle.plan else None
    
    if intent == IntentFamily.ROSTER_ENUMERATION:
        # Group by entity
        entity_claims = {}
        for claim in bundle.claims:
            entity_key = claim.subject
            if entity_key not in entity_claims:
                entity_claims[entity_key] = []
            entity_claims[entity_key].append(claim)
        
        # Render per-entity bullets
        for entity_key, claims in entity_claims.items():
            # Get entity display name
            entity_name = entity_key
            for entity in bundle.entities:
                if entity.key == entity_key:
                    entity_name = entity.display_name
                    break
            
            # Build bullet text from claims
            claim_texts = []
            citations = []
            claim_ids = []
            
            for claim in claims:
                predicate_text = _predicate_to_text(claim.predicate)
                claim_texts.append(predicate_text)
                claim_ids.append(claim.claim_id)
                
                for ref in claim.evidence:
                    citations.append(ref.to_dict())
                    total_citations += 1
            
            # Determine inference level
            inference_level = "explicit"
            if all(c.support_type == SupportType.CO_MENTION for c in claims):
                inference_level = "inferential"
            
            bullet_text = f"{entity_name}: {'; '.join(set(claim_texts))}"
            
            # Extract entity_id from subject if possible
            entity_id = None
            subject = claims[0].subject
            if isinstance(subject, str) and ":" in subject:
                try:
                    entity_id = int(subject.split(":")[-1])
                except (ValueError, IndexError):
                    pass
            elif isinstance(subject, int):
                entity_id = subject
            
            bullets.append(BulletWithCitation(
                text=bullet_text,
                entity_id=entity_id,
                claims_used=claim_ids,
                citations=citations,
                inference_level=inference_level,
            ))
    else:
        # Standard claim-per-bullet rendering
        for claim in bundle.claims:
            bullet_text = _claim_to_bullet_text(claim, bundle.entities)
            
            citations = []
            for ref in claim.evidence:
                citations.append(ref.to_dict())
                total_citations += 1
            
            inference_level = "explicit"
            if claim.support_type == SupportType.CO_MENTION:
                inference_level = "inferential"
            
            entity_id = None
            if claim.subject.startswith("entity:"):
                try:
                    entity_id = int(claim.subject.split(":")[1])
                except (ValueError, IndexError):
                    pass
            
            bullets.append(BulletWithCitation(
                text=bullet_text,
                entity_id=entity_id,
                claims_used=[claim.claim_id],
                citations=citations,
                inference_level=inference_level,
            ))
    
    # Build short answer
    short_answer = _build_short_answer(bundle, bullets)
    
    # Generate LLM summary if we have the query text
    summary = None
    if bundle.plan and bundle.plan.query_text and bundle.all_chunks:
        import sys
        print("  [Agentic] Generating summary...", file=sys.stderr)
        summary = generate_agentic_summary(bundle, bundle.plan.query_text)
    
    return RenderedAnswer(
        short_answer=short_answer,
        bullets=bullets,
        claims_rendered=len(bundle.claims),
        citations_included=total_citations,
        summary=summary,
    )


def _predicate_to_text(predicate) -> str:
    """Convert predicate to human-readable text."""
    from retrieval.evidence_bundle import Predicate
    
    mapping = {
        Predicate.MENTIONS: "mentioned",
        Predicate.DESCRIBES: "described",
        Predicate.EVALUATES: "evaluated",
        Predicate.ASSOCIATED_WITH: "associated with network",
        Predicate.HANDLED_BY: "handler relationship",
        Predicate.MET_WITH: "met with",
        Predicate.IDENTIFIED_AS: "identified as",
        Predicate.CODENAME_OF: "codename for",
    }
    return mapping.get(predicate, str(predicate))


def _claim_to_bullet_text(claim, entities) -> str:
    """Convert a claim to bullet point text."""
    from retrieval.evidence_bundle import Predicate
    
    # Get display names
    subject_name = claim.subject
    object_name = claim.object
    
    for entity in entities:
        if entity.key == claim.subject:
            subject_name = entity.display_name
        if entity.key == claim.object:
            object_name = entity.display_name
    
    # If object is just a chunk reference, use quote_span from evidence instead
    if object_name.startswith("chunk:") and claim.evidence:
        quote = claim.evidence[0].quote_span
        if quote:
            # Truncate long quotes
            if len(quote) > 150:
                quote = quote[:150] + "..."
            object_name = f'"{quote}"'
    
    # Build text based on predicate
    if claim.predicate == Predicate.MENTIONS:
        # For mentions, show the entity/subject and the evidence
        if object_name.startswith('"'):
            return f"{subject_name}: {object_name}"
        return f"Evidence mentions {object_name}"
    elif claim.predicate == Predicate.DESCRIBES:
        return f"Describes {object_name}"
    elif claim.predicate == Predicate.EVALUATES:
        return f"Evaluates {object_name}"
    elif claim.predicate == Predicate.ASSOCIATED_WITH:
        return f"{subject_name} associated with {object_name}"
    elif claim.predicate == Predicate.HANDLED_BY:
        return f"{subject_name} handled by {object_name}"
    elif claim.predicate == Predicate.MET_WITH:
        return f"{subject_name} met with {object_name}"
    elif claim.predicate == Predicate.IDENTIFIED_AS:
        return f"{subject_name} identified as {object_name}"
    elif claim.predicate == Predicate.CODENAME_OF:
        return f"{subject_name} is codename for {object_name}"
    else:
        return f"{subject_name} → {object_name}"


def _build_short_answer(bundle, bullets) -> str:
    """Build a short summary answer from bullets."""
    if not bullets:
        return "No explicit evidence found."
    
    if len(bullets) == 1:
        return bullets[0].text
    
    # Count explicit vs inferential
    explicit_count = sum(1 for b in bullets if b.inference_level == "explicit")
    
    if explicit_count == len(bullets):
        return f"Found {len(bullets)} explicit findings with supporting evidence."
    else:
        inferential_count = len(bullets) - explicit_count
        return f"Found {explicit_count} explicit and {inferential_count} inferential findings."


def generate_agentic_summary(bundle, query_text: str) -> Optional[str]:
    """
    Generate an LLM-based summary of the agentic search results.
    
    Uses the highest-scored evidence chunks to create a coherent answer to the query.
    Prioritizes chunks by:
    1. Score (best_score from retrieval)
    2. Whether they contain query terms
    """
    import os
    import re
    
    # Only generate if we have OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Extract key terms from query for prioritization
        query_lower = query_text.lower()
        query_terms = set(re.findall(r'\b\w{4,}\b', query_lower))  # Words 4+ chars
        # Add important terms that might be in the corpus
        if "proximity" in query_lower or "fuse" in query_lower:
            query_terms.update(["proximity", "fuse", "fuze", "vt"])  # VT fuse variants
        
        # Sort chunks by score (highest first), then by term relevance
        def chunk_priority(item):
            chunk_id, chunk = item
            score = chunk.best_score if hasattr(chunk, 'best_score') else 0.0
            # Boost chunks containing query terms
            term_bonus = 0
            if chunk.text:
                text_lower = chunk.text.lower()
                term_bonus = sum(10 for term in query_terms if term in text_lower)
            return -(score + term_bonus)  # Negative for descending sort
        
        sorted_chunks = sorted(bundle.all_chunks.items(), key=chunk_priority)
        
        # Collect evidence text from top chunks
        evidence_texts = []
        for chunk_id, chunk in sorted_chunks[:30]:
            if chunk.text:
                # Take first 600 chars from each chunk
                text = chunk.text[:600]
                if len(chunk.text) > 600:
                    text += "..."
                evidence_texts.append(text)
        
        if not evidence_texts:
            return None
        
        # Build prompt
        evidence_section = "\n\n---\n\n".join(evidence_texts)
        
        prompt = f"""You are a research assistant helping historians analyze archival materials.

The user asked: "{query_text}"

Below are excerpts from documents retrieved by the search system. Based ONLY on this evidence, provide a concise summary that directly answers the user's question. Be specific and cite what the documents actually say. If the evidence doesn't directly answer the question, say so clearly.

EVIDENCE EXCERPTS:
{evidence_section}

INSTRUCTIONS:
- Summarize what the evidence shows regarding the user's question
- Be specific about what IS and IS NOT found
- If the evidence mentions relevant topics (even with different terminology), point that out
- Keep the summary to 2-3 paragraphs
- Focus on answering the question, not describing the documents"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a precise research assistant. Summarize archival evidence to answer the user's question."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        import sys
        print(f"  [Agentic] Warning: Could not generate summary: {e}", file=sys.stderr)
        return None
