"""
V4 Interpretation Synthesis - Reasoning-first structured answer generation.

V4 replaces claim synthesis with a stronger reasoning model (4o) that produces
structured InterpretationV4 objects with explicit citations and uncertainty labels.

Key principles:
- Every answer_unit must have 1-2 citations (except uncertainty units)
- Model proposes, verifier decides - trust boundary is code, not LLM
- Response shapes guide output structure (roster, narrative, timeline, etc.)
- "If you cannot cite it, do not state it. Emit an uncertainty unit instead."

Usage:
    from retrieval.agent.v4_interpret import interpret_evidence
    
    interpretation = interpret_evidence(
        evidence_set=evidence_set,
        query="Who were members of the Silvermaster network?",
        conn=conn,
    )
    
    for unit in interpretation.answer_units:
        print(f"[{unit.confidence}] {unit.text}")
        print(f"  Phrases: {unit.supporting_phrases}")
        print(f"  Citations: {unit.citations}")
"""

import os
import sys
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from retrieval.agent.v3_evidence import EvidenceSet, EvidenceSpan
from retrieval.agent.v3_summarizer import doc_balanced_sample


# =============================================================================
# V4 Constants
# =============================================================================

V4_VERSION = "4.0.0"
V4_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_V4", "gpt-4o")
V4_MODEL_PLANNER = os.getenv("OPENAI_MODEL_V4_PLANNER", "gpt-4o-mini")

# Default budgets
V4_BUDGETS = {
    "max_interpret_rounds": 2,
    "max_retrieval_rounds": 2,
    "max_spans_to_interpret": 60,
    "max_spans_per_doc": 5,
    "context_window_chars": 200,
    "max_answer_units": 25,
}

# Response shapes (rendering preference only)
RESPONSE_SHAPES = {"roster", "narrative", "timeline", "qa", "index"}



# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SpanCitation:
    """Citation referencing a span by index."""
    span_idx: int
    span_id: Optional[str] = None  # Resolved after verification
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_idx": self.span_idx,
            "span_id": self.span_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanCitation":
        return cls(
            span_idx=data.get("span_idx", 0),
            span_id=data.get("span_id"),
        )


@dataclass
class AnswerUnit:
    """
    A grounded answer unit - the universal primitive for all V4 responses.
    
    Every user-visible response is a list of AnswerUnits. The same format
    works for roster, narrative, timeline, Q&A - just different rendering.
    
    Core contract:
    - text: 1-2 sentences max, constrained to what citations support
    - citations: 1-3 spans required (except uncertainty units with empty citations)
    - supporting_phrases: Verbatim substrings from cited quotes that justify the text
    """
    text: str                                      # 1-2 sentences max
    citations: List[SpanCitation]                  # 1-3 spans (required for grounded units)
    supporting_phrases: List[str] = field(default_factory=list)  # Verbatim from quotes
    confidence: str = "supported"                  # supported|suggestive
    about_entities: List[int] = field(default_factory=list)  # Optional entity attestation
    unit_id: str = ""                              # Auto-generated hash
    
    def __post_init__(self):
        if not self.unit_id:
            content = f"{self.text}:{len(self.citations)}:{len(self.supporting_phrases)}"
            self.unit_id = hashlib.sha256(content.encode()).hexdigest()[:12]
    
    @property
    def is_uncertainty(self) -> bool:
        """
        Uncertainty units have no citations and suggestive confidence.
        
        A unit with confidence="supported" but no citations is NOT an uncertainty unit -
        it's an error that should be caught by verification.
        """
        return len(self.citations) == 0 and self.confidence == "suggestive"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "supporting_phrases": self.supporting_phrases,
            "confidence": self.confidence,
            "about_entities": self.about_entities,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerUnit":
        return cls(
            text=data.get("text", ""),
            citations=[SpanCitation.from_dict(c) for c in data.get("citations", [])],
            supporting_phrases=data.get("supporting_phrases", []),
            confidence=data.get("confidence", "supported"),
            about_entities=data.get("about_entities", []),
            unit_id=data.get("unit_id", ""),
        )


@dataclass
class DiagnosticsInfo:
    """Diagnostic information from interpretation."""
    missing_info_questions: List[str]
    followup_queries: List[str]
    model_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "missing_info_questions": self.missing_info_questions,
            "followup_queries": self.followup_queries,
            "model_notes": self.model_notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosticsInfo":
        return cls(
            missing_info_questions=data.get("missing_info_questions", []),
            followup_queries=data.get("followup_queries", []),
            model_notes=data.get("model_notes"),
        )


@dataclass
class InterpretationV4:
    """
    The main V4 interpretation object.
    
    Contains structured answer_units with explicit citations,
    confidence levels, and semantic tags.
    """
    query: str
    response_shape: str
    answer_units: List[AnswerUnit]
    diagnostics: DiagnosticsInfo
    model_version: str
    interpretation_hash: str
    input_span_ids: List[str] = field(default_factory=list)
    input_span_hashes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.interpretation_hash:
            content = json.dumps({
                "query": self.query,
                "response_shape": self.response_shape,
                "units": [u.to_dict() for u in self.answer_units],
            }, sort_keys=True)
            self.interpretation_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response_shape": self.response_shape,
            "answer_units": [u.to_dict() for u in self.answer_units],
            "diagnostics": self.diagnostics.to_dict(),
            "model_version": self.model_version,
            "interpretation_hash": self.interpretation_hash,
            "input_span_ids": self.input_span_ids,
            "input_span_hashes": self.input_span_hashes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterpretationV4":
        return cls(
            query=data.get("query", ""),
            response_shape=data.get("response_shape", "narrative"),
            answer_units=[AnswerUnit.from_dict(u) for u in data.get("answer_units", [])],
            diagnostics=DiagnosticsInfo.from_dict(data.get("diagnostics", {})),
            model_version=data.get("model_version", V4_MODEL_DEFAULT),
            interpretation_hash=data.get("interpretation_hash", ""),
            input_span_ids=data.get("input_span_ids", []),
            input_span_hashes=data.get("input_span_hashes", []),
        )


# =============================================================================
# Span Preparation with Context Window
# =============================================================================

@dataclass
class PreparedSpan:
    """A span prepared for interpretation with context window."""
    span_idx: int
    span: EvidenceSpan
    attest_text: str  # Extended context for attestation (+/- 200 chars)
    entities_in_span: List[Dict[str, Any]]  # Entities with aliases
    span_type_hint: str = "paragraph"  # paragraph|list|header
    
    def to_prompt_dict(self) -> Dict[str, Any]:
        """Format for LLM prompt."""
        return {
            "idx": self.span_idx,
            "quote": self.span.quote[:500],  # Cap quote length
            "doc_id": self.span.doc_id,
            "page_ref": self.span.page_ref,
            "type": self.span_type_hint,
            "entities": [
                {
                    "id": e["entity_id"], 
                    "name": e["canonical_name"],
                    "aliases": e.get("aliases", [])[:2],  # Top 2 aliases
                }
                for e in self.entities_in_span[:5]  # Cap entities shown
            ] if self.entities_in_span else [],
        }


def _detect_span_type(quote: str) -> str:
    """
    Cheaply detect span type from formatting.
    
    Helps 4o understand context (list items, headers, paragraphs).
    """
    import re
    
    # List indicators: numbered list, bullet points
    if re.match(r'^\s*[\d]+[\.\)]\s', quote):
        return "list"
    if re.match(r'^\s*[-\*\•]\s', quote):
        return "list"
    
    # Header indicators: short, often uppercase or title case
    if len(quote) < 100:
        upper_ratio = sum(1 for c in quote if c.isupper()) / max(len(quote), 1)
        if upper_ratio > 0.5:
            return "header"
    
    return "paragraph"


def prepare_spans_for_interpretation(
    evidence_set: EvidenceSet,
    conn,
    max_spans: int = 60,
    max_per_doc: int = 5,
    context_window: int = 200,
) -> List[PreparedSpan]:
    """
    Prepare spans for interpretation using 3-bucket selection strategy.
    
    V4.2 Enhancement: Smarter span selection for better recall:
    1. Top bucket: Top spans by rerank score
    2. Feature bucket: Spans with entity surfaces, list cues, definitional patterns
    3. Doc-balance bucket: Fill from remaining docs for diversity
    
    This ensures we don't miss roster pages or key evidence that may not 
    have the highest similarity score but contains important structural content.
    """
    # 3-bucket selection for recall insurance
    selected = _three_bucket_span_selection(
        cite_spans=evidence_set.cite_spans,
        max_total=max_spans,
        max_per_doc=max_per_doc,
    )
    
    if not selected:
        return []
    
    # Load chunk texts for context windows
    chunk_texts = _load_chunk_texts(conn, selected)
    
    # Load entities in spans
    entity_mentions = _load_entity_mentions(conn, selected)
    
    # Build prepared spans
    prepared = []
    for i, span in enumerate(selected):
        chunk_text = chunk_texts.get(span.chunk_id, "")
        
        # Compute context window
        if chunk_text:
            start = max(0, span.start_char - context_window)
            end = min(len(chunk_text), span.end_char + context_window)
            attest_text = chunk_text[start:end]
        else:
            attest_text = span.quote
        
        # Get entities for this span (now includes aliases)
        entities = entity_mentions.get(span.chunk_id, [])
        
        # Detect span type from formatting
        span_type = _detect_span_type(span.quote)
        
        prepared.append(PreparedSpan(
            span_idx=i,
            span=span,
            attest_text=attest_text,
            entities_in_span=entities,
            span_type_hint=span_type,
        ))
    
    return prepared


def _three_bucket_span_selection(
    cite_spans: List[EvidenceSpan],
    max_total: int = 60,
    max_per_doc: int = 5,
) -> List[EvidenceSpan]:
    """
    3-bucket span selection for recall insurance.
    
    Buckets:
    1. Score bucket (~40%): Top spans by rerank score
    2. Feature bucket (~30%): Spans with list/definitional patterns
    3. Doc-balance bucket (~30%): Doc-balanced fill for diversity
    
    This ensures coverage of:
    - High-relevance content (bucket 1)
    - Structural content like rosters (bucket 2)  
    - Diverse document coverage (bucket 3)
    """
    import re
    
    if not cite_spans:
        return []
    
    if len(cite_spans) <= max_total:
        return cite_spans
    
    # Allocate buckets
    score_budget = int(max_total * 0.4)
    feature_budget = int(max_total * 0.3)
    doc_balance_budget = max_total - score_budget - feature_budget
    
    selected: List[EvidenceSpan] = []
    selected_ids: set = set()
    
    # Sort by score for bucket 1
    sorted_by_score = sorted(cite_spans, key=lambda s: -s.score)
    
    # Bucket 1: Top by score
    for span in sorted_by_score:
        if len(selected) >= score_budget:
            break
        if span.span_id not in selected_ids:
            selected.append(span)
            selected_ids.add(span.span_id)
    
    # Bucket 2: Feature-rich spans (list-like, definitional)
    # Patterns that indicate high-value structural content
    list_patterns = [
        r'^\s*\d+[\.\)]\s',          # "1. " or "1) "
        r'^\s*[-\*\•]\s',            # "- " or "• "
        r'\bmembers?\s+(?:include|were|are)\b',
        r'\bsources?\s+include\b',
        r'\bgroup\s+(?:consisted|included)\b',
    ]
    
    def_patterns = [
        r'\b(?:is|was)\s+(?:a|an|the)\s+\w+',  # "X is a Y"
        r'\bknown\s+as\b',
        r'\bcodename[d]?\b',
        r'\balias(?:es)?\b',
        r'\bidentified\s+as\b',
    ]
    
    def has_features(text: str) -> bool:
        text_lower = text.lower()
        for pattern in list_patterns + def_patterns:
            if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
                return True
        return False
    
    # Select feature-rich spans not already selected
    feature_spans = [s for s in cite_spans if has_features(s.quote) and s.span_id not in selected_ids]
    feature_spans.sort(key=lambda s: -s.score)  # Still prefer higher scores
    
    for span in feature_spans:
        if len(selected) >= score_budget + feature_budget:
            break
        if span.span_id not in selected_ids:
            selected.append(span)
            selected_ids.add(span.span_id)
    
    # Bucket 3: Doc-balanced fill
    # Group remaining spans by doc
    remaining = [s for s in cite_spans if s.span_id not in selected_ids]
    by_doc: dict = {}
    for span in remaining:
        if span.doc_id not in by_doc:
            by_doc[span.doc_id] = []
        by_doc[span.doc_id].append(span)
    
    # Sort docs by how many spans we've already selected from them (prefer underrepresented)
    doc_counts = {}
    for span in selected:
        doc_counts[span.doc_id] = doc_counts.get(span.doc_id, 0) + 1
    
    sorted_docs = sorted(by_doc.keys(), key=lambda d: doc_counts.get(d, 0))
    
    # Round-robin from underrepresented docs
    doc_idx = 0
    while len(selected) < max_total and by_doc:
        if doc_idx >= len(sorted_docs):
            doc_idx = 0
        
        doc_id = sorted_docs[doc_idx]
        if doc_id in by_doc and by_doc[doc_id]:
            # Check doc cap
            current_doc_count = sum(1 for s in selected if s.doc_id == doc_id)
            if current_doc_count < max_per_doc:
                span = by_doc[doc_id].pop(0)
                selected.append(span)
                selected_ids.add(span.span_id)
            
            if not by_doc[doc_id]:
                del by_doc[doc_id]
                sorted_docs = [d for d in sorted_docs if d in by_doc]
        
        doc_idx += 1
        
        # Safety: break if no progress
        if not sorted_docs:
            break
    
    return selected


def _load_chunk_texts(conn, spans: List[EvidenceSpan]) -> Dict[int, str]:
    """Load full chunk texts for context window computation."""
    if not spans or not conn:
        return {}
    
    chunk_ids = list({s.chunk_id for s in spans})
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text FROM chunks WHERE id = ANY(%s)",
                (chunk_ids,)
            )
            return {row[0]: row[1] or "" for row in cur.fetchall()}
    except Exception:
        return {}


def _load_entity_mentions(conn, spans: List[EvidenceSpan]) -> Dict[int, List[Dict]]:
    """Load entity mentions for spans (by chunk_id), including aliases."""
    if not spans or not conn:
        return {}
    
    chunk_ids = list({s.chunk_id for s in spans})
    
    try:
        with conn.cursor() as cur:
            # First get all entities mentioned in these chunks
            cur.execute("""
                SELECT em.chunk_id, e.id as entity_id, e.canonical_name, e.entity_type
                FROM entity_mentions em
                JOIN entities e ON e.id = em.entity_id
                WHERE em.chunk_id = ANY(%s)
                ORDER BY em.chunk_id, e.canonical_name
            """, (chunk_ids,))
            
            entity_rows = cur.fetchall()
            entity_ids = list({row[1] for row in entity_rows})
            
            # Load aliases for these entities
            aliases_by_entity: Dict[int, List[str]] = defaultdict(list)
            if entity_ids:
                try:
                    cur.execute("""
                        SELECT entity_id, alias_text
                        FROM entity_aliases
                        WHERE entity_id = ANY(%s)
                        ORDER BY entity_id, alias_text
                    """, (entity_ids,))
                    for eid, alias in cur.fetchall():
                        aliases_by_entity[eid].append(alias)
                except Exception:
                    pass  # Table might not exist, continue without aliases
            
            # Build result with aliases
            result: Dict[int, List[Dict]] = defaultdict(list)
            for chunk_id, entity_id, canonical_name, entity_type in entity_rows:
                result[chunk_id].append({
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                    "aliases": aliases_by_entity.get(entity_id, [])[:3],  # Top 3 aliases
                })
            
            return dict(result)
    except Exception:
        return {}


# =============================================================================
# Response Shape Detection
# =============================================================================

def detect_response_shape(query: str) -> str:
    """
    Auto-detect the best response shape based on query.
    """
    query_lower = query.lower()
    
    # Roster indicators
    roster_words = ["who", "members", "people", "individuals", "agents", "list", 
                    "group", "network", "ring", "sources", "contacts"]
    if any(word in query_lower for word in roster_words):
        return "roster"
    
    # Timeline indicators
    timeline_words = ["when", "timeline", "chronology", "sequence", "dates", "history", "events"]
    if any(word in query_lower for word in timeline_words):
        return "timeline"
    
    # Index/lookup indicators
    index_words = ["where is", "which document", "find", "locate", "what documents"]
    if any(word in query_lower for word in index_words):
        return "index"
    
    # QA indicators (direct questions)
    qa_words = ["what is", "how did", "why did", "explain"]
    if any(word in query_lower for word in qa_words):
        return "qa"
    
    # Default to narrative
    return "narrative"


# =============================================================================
# Interpretation Prompt Building
# =============================================================================

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "response_shape": {
            "type": "string",
            "enum": ["roster", "narrative", "timeline", "qa", "index"]
        },
        "answer_units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "span_idx": {"type": "integer"}
                            },
                            "required": ["span_idx"],
                            "additionalProperties": False
                        }
                    },
                    "supporting_phrases": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["supported", "suggestive"]
                    },
                    "about_entities": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                },
                "required": ["text", "citations", "supporting_phrases"],
                "additionalProperties": False
            }
        },
        "diagnostics": {
            "type": "object",
            "properties": {
                "missing_info_questions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "followup_queries": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "model_notes": {"type": ["string", "null"]}
            },
            "required": ["missing_info_questions", "followup_queries"],
            "additionalProperties": False
        }
    },
    "required": ["response_shape", "answer_units", "diagnostics"],
    "additionalProperties": False
}


def build_interpretation_prompt(
    query: str,
    prepared_spans: List[PreparedSpan],
    suggested_shape: str,
    max_units: int = 25,
) -> str:
    """
    Build cite-first prompt for 4o interpretation.
    
    Key behavioral change: model must identify evidence FIRST, 
    then constrain its statement to that evidence.
    """
    
    # Format spans with rich metadata
    spans_text = []
    for ps in prepared_spans:
        entity_str = ""
        if ps.entities_in_span:
            entities = [f"{e['entity_id']}:{e['canonical_name']}" for e in ps.entities_in_span[:3]]
            entity_str = f"\n  Entities: {', '.join(entities)}"
        
        type_hint = getattr(ps, 'span_type_hint', 'paragraph')
        
        spans_text.append(
            f"[S{ps.span_idx}] (doc:{ps.span.doc_id}, {ps.span.page_ref}, type:{type_hint}){entity_str}\n"
            f'"{ps.span.quote[:400]}"'
        )
    
    spans_formatted = "\n\n".join(spans_text)
    
    return f"""Answer this research query using ONLY the provided evidence spans.

QUERY: {query}

EVIDENCE SPANS:
{spans_formatted}

=== CITE-FIRST GENERATION RULES ===

For each fact you want to assert, follow this EXACT order:
1. FIRST: Select 1-3 span indices that support the fact
2. SECOND: Extract the EXACT supporting phrase(s) from those spans (verbatim substrings)
3. THIRD: Write a 1-2 sentence statement constrained to what those phrases say

CRITICAL: If you cannot find a supporting phrase in the spans, do NOT state it.
Instead, emit an uncertainty unit with empty citations.

=== OUTPUT FORMAT ===

Each answer_unit must have:
{{
  "text": "Your 1-2 sentence statement",
  "citations": [{{"span_idx": 5}}, {{"span_idx": 12}}],
  "supporting_phrases": ["exact phrase from S5", "exact phrase from S12"],
  "confidence": "supported",
  "about_entities": [123, 456]  // Optional: entity IDs being discussed
}}

For uncertainty (gaps in evidence):
{{
  "text": "The spans don't explicitly state X.",
  "citations": [],
  "supporting_phrases": [],
  "confidence": "suggestive"
}}

=== CONFIDENCE LEVELS ===
- "supported": The supporting_phrases explicitly state the fact
- "suggestive": The fact is implied but not directly stated

=== RESPONSE SHAPE ===
Suggested shape: {suggested_shape}
(The system will organize your units based on this. Just ensure every unit is grounded.)

=== DIAGNOSTICS ===
Include in diagnostics:
- missing_info_questions: Questions the evidence doesn't answer
- followup_queries: Suggested search terms to find missing info

Maximum {max_units} answer_units. Uncertainty is NOT failure - it's the correct response to limited evidence.

OUTPUT: JSON with response_shape, answer_units, and diagnostics.
"""


# =============================================================================
# Interpretation Generation
# =============================================================================

def interpret_evidence(
    evidence_set: EvidenceSet,
    query: str,
    conn,
    response_shape: Optional[str] = None,
    max_spans: int = None,
    max_units: int = None,
    verifier_errors: Optional[List[str]] = None,
) -> InterpretationV4:
    """
    Generate V4 interpretation from evidence using 4o.
    
    Args:
        evidence_set: EvidenceSet with cite_spans
        query: User's query
        conn: Database connection
        response_shape: Optional shape override (auto-detected if None)
        max_spans: Max spans to send to model
        max_units: Max answer units to request
        verifier_errors: Optional errors from previous verification (for retry)
    
    Returns:
        InterpretationV4 with structured answer_units
    """
    max_spans = max_spans or V4_BUDGETS["max_spans_to_interpret"]
    max_units = max_units or V4_BUDGETS["max_answer_units"]
    
    if not evidence_set.cite_spans:
        return _empty_interpretation(query)
    
    # Prepare spans
    print(f"  [V4 Interpret] Preparing {len(evidence_set.cite_spans)} cite_spans...", file=sys.stderr)
    prepared = prepare_spans_for_interpretation(
        evidence_set=evidence_set,
        conn=conn,
        max_spans=max_spans,
        max_per_doc=V4_BUDGETS["max_spans_per_doc"],
        context_window=V4_BUDGETS["context_window_chars"],
    )
    
    if not prepared:
        return _empty_interpretation(query)
    
    print(f"    Selected {len(prepared)} spans (doc-balanced)", file=sys.stderr)
    
    # Detect or use provided shape
    if response_shape is None:
        response_shape = detect_response_shape(query)
        print(f"    Auto-detected shape: {response_shape}", file=sys.stderr)
    
    # Build prompt
    prompt = build_interpretation_prompt(
        query=query,
        prepared_spans=prepared,
        suggested_shape=response_shape,
        max_units=max_units,
    )
    
    # Add retry context if provided
    if verifier_errors:
        error_section = "\n\nPREVIOUS VERIFICATION ERRORS (fix these):\n"
        for err in verifier_errors[:5]:
            error_section += f"- {err}\n"
        prompt += error_section
    
    # Call LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("    Warning: No API key, returning fallback interpretation", file=sys.stderr)
        return _fallback_interpretation(query, prepared, response_shape)
    
    model = V4_MODEL_DEFAULT
    prompt_len = len(prompt)
    print(f"    Calling {model} (prompt: {prompt_len:,} chars, {len(prepared)} spans)...", file=sys.stderr)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research analyst. Produce structured JSON interpretations of archival evidence. Every statement must cite evidence. If you cannot cite it, do not say it."
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=3000,
        )
        
        content = response.choices[0].message.content
        print(f"    LLM response received ({len(content) if content else 0} chars)", file=sys.stderr)
        
        if not content:
            return _fallback_interpretation(query, prepared, response_shape)
        
        # Parse response
        data = json.loads(content)
        
        # Build InterpretationV4
        answer_units = []
        for unit_data in data.get("answer_units", []):
            # Parse citations
            citations = []
            for cit in unit_data.get("citations", []):
                span_idx = cit.get("span_idx", 0)
                span_id = None
                if 0 <= span_idx < len(prepared):
                    span_id = prepared[span_idx].span.span_id
                citations.append(SpanCitation(span_idx=span_idx, span_id=span_id))
            
            answer_units.append(AnswerUnit(
                text=unit_data.get("text", ""),
                citations=citations,
                supporting_phrases=unit_data.get("supporting_phrases", []),
                confidence=unit_data.get("confidence", "supported"),
                about_entities=unit_data.get("about_entities", []),
            ))
        
        diagnostics = DiagnosticsInfo.from_dict(data.get("diagnostics", {}))
        
        print(f"    Generated {len(answer_units)} answer_units", file=sys.stderr)
        
        return InterpretationV4(
            query=query,
            response_shape=data.get("response_shape", response_shape),
            answer_units=answer_units,
            diagnostics=diagnostics,
            model_version=model,
            interpretation_hash="",  # Auto-generated
            input_span_ids=[ps.span.span_id for ps in prepared],
            input_span_hashes=[ps.span.span_hash for ps in prepared],
        )
        
    except Exception as e:
        print(f"    Interpretation error: {e}", file=sys.stderr)
        return _fallback_interpretation(query, prepared, response_shape)


def _empty_interpretation(query: str) -> InterpretationV4:
    """Return empty interpretation when no evidence available."""
    return InterpretationV4(
        query=query,
        response_shape="narrative",
        answer_units=[
            AnswerUnit(
                text="No evidence spans available to analyze.",
                citations=[],  # Uncertainty unit - no citations
                supporting_phrases=[],
                confidence="suggestive",
            )
        ],
        diagnostics=DiagnosticsInfo(
            missing_info_questions=["No evidence was retrieved for this query."],
            followup_queries=[],
        ),
        model_version=V4_MODEL_DEFAULT,
        interpretation_hash="",
    )


def _fallback_interpretation(
    query: str,
    prepared: List[PreparedSpan],
    response_shape: str,
) -> InterpretationV4:
    """Generate fallback interpretation when LLM unavailable."""
    units = []
    
    for ps in prepared[:10]:
        preview = ps.span.quote[:150].strip()
        preview = ' '.join(preview.split())
        
        units.append(AnswerUnit(
            text=f"Evidence from {ps.span.page_ref}: \"{preview}...\"",
            citations=[SpanCitation(span_idx=ps.span_idx, span_id=ps.span.span_id)],
            supporting_phrases=[preview[:50]],  # Extract a phrase
            confidence="suggestive",
            about_entities=[e["entity_id"] for e in ps.entities_in_span[:2]],
        ))
    
    return InterpretationV4(
        query=query,
        response_shape=response_shape,
        answer_units=units,
        diagnostics=DiagnosticsInfo(
            missing_info_questions=[],
            followup_queries=[],
            model_notes="Fallback interpretation - LLM unavailable",
        ),
        model_version="fallback",
        interpretation_hash="",
        input_span_ids=[ps.span.span_id for ps in prepared],
        input_span_hashes=[ps.span.span_hash for ps in prepared],
    )
