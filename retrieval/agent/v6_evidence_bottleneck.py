"""
V6 Evidence Bottleneck - Force convergence before synthesis

The SINGLE MOST IMPORTANT "make it smart" step:
- After retrieval, grade spans to keep 30-50 MAX
- Claim extraction / answer writing sees ONLY those
- This prevents "164 chunks → thousands of claims"

The bottleneck is a HARD gate - synthesis cannot proceed without it.
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from retrieval.agent.v6_query_parser import ParsedQuery, TaskType


# =============================================================================
# Configuration
# =============================================================================

BOTTLENECK_MODEL = "gpt-4o-mini"
BOTTLENECK_BATCH_SIZE = 15
DEFAULT_BOTTLENECK_SIZE = 40  # Max spans after bottleneck


# =============================================================================
# Bottleneck Span
# =============================================================================

@dataclass
class BottleneckSpan:
    """A span that passed the evidence bottleneck."""
    
    span_id: str
    chunk_id: int
    doc_id: Optional[int]
    page: Optional[str]
    source_label: str
    
    # The actual quotable text
    span_text: str
    
    # Grading from bottleneck
    relevance_score: float  # 0-10
    claim_supported: str  # What this span supports
    is_directly_responsive: bool  # Does it DIRECTLY answer the question?
    
    # For roster queries: does this identify a member?
    identifies_member: bool = False
    member_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page": self.page,
            "source_label": self.source_label,
            "span_text": self.span_text,
            "relevance_score": self.relevance_score,
            "claim_supported": self.claim_supported,
            "is_directly_responsive": self.is_directly_responsive,
            "identifies_member": self.identifies_member,
            "member_name": self.member_name,
        }


# =============================================================================
# Bottleneck Result
# =============================================================================

@dataclass
class BottleneckResult:
    """Result of the evidence bottleneck."""
    
    spans: List[BottleneckSpan] = field(default_factory=list)
    
    # Stats
    chunks_input: int = 0
    spans_extracted: int = 0
    spans_passed: int = 0
    
    # For roster queries
    members_identified: List[str] = field(default_factory=list)
    
    # Timing
    elapsed_ms: float = 0.0
    
    def get_synthesis_context(self) -> str:
        """Get the context for synthesis (ONLY these spans)."""
        lines = []
        for i, span in enumerate(self.spans):
            lines.append(f"[{i}] (chunk:{span.chunk_id}, {span.source_label}, p.{span.page})")
            lines.append(f'    "{span.span_text}"')
            lines.append(f"    Supports: {span.claim_supported}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spans": [s.to_dict() for s in self.spans],
            "chunks_input": self.chunks_input,
            "spans_extracted": self.spans_extracted,
            "spans_passed": self.spans_passed,
            "members_identified": self.members_identified,
            "elapsed_ms": self.elapsed_ms,
        }


# =============================================================================
# Bottleneck Prompt
# =============================================================================

BOTTLENECK_SYSTEM_PROMPT = """You are an evidence quality gate. Your job is to identify the BEST spans for answering a specific question.

Be STRICT. Only spans with direct, quotable evidence should pass.

For roster queries ("who were members"):
- PASS: "X was a member of Y network" → identifies_member=true
- PASS: "X reported to Y about Z" (implies membership)
- FAIL: "The network operated in Washington" (no member named)
- FAIL: "HUAC investigated espionage" (too general)

For each span, output:
- pass: true if this is quality evidence
- score: 0-10 relevance
- claim: what specific claim does this support?
- responsive: does it DIRECTLY answer the question?
- identifies_member: (for roster) does it name a member?
- member_name: (for roster) who is named as member?"""


def build_bottleneck_prompt(
    question: str,
    task_type: TaskType,
    chunks: List[Dict[str, Any]],
) -> str:
    """Build prompt for bottleneck grading."""
    
    task_guidance = ""
    if task_type == TaskType.ROSTER_ENUMERATION:
        task_guidance = """
TASK: Roster enumeration - identify MEMBERS
Only pass spans that:
- Explicitly name someone as a member
- Describe someone's role in the network
- Show evidence of someone's involvement
Do NOT pass spans that just mention the network generally."""
    
    chunks_section = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")[:1000]
        source = chunk.get("source_label", "")
        page = chunk.get("page", "")
        chunks_section.append(f"""
CHUNK {i} ({source}, p.{page}):
"{text}"
""")
    
    chunks_text = "\n---\n".join(chunks_section)
    
    return f"""QUESTION: {question}
{task_guidance}

CHUNKS TO EVALUATE:
{chunks_text}

For EACH chunk, output whether it should PASS the bottleneck.
Be strict - only high-quality, directly responsive evidence should pass.

Output JSON:
{{
  "evaluations": [
    {{
      "chunk_index": 0,
      "pass": true,
      "score": 8,
      "claim": "X was a member of Y",
      "responsive": true,
      "identifies_member": true,
      "member_name": "Harry White"
    }},
    {{
      "chunk_index": 1,
      "pass": false,
      "score": 2,
      "claim": "",
      "responsive": false,
      "identifies_member": false,
      "member_name": ""
    }}
  ]
}}"""


# =============================================================================
# Evidence Bottleneck
# =============================================================================

class EvidenceBottleneck:
    """
    The HARD gate that forces convergence.
    
    Synthesis CANNOT proceed without passing through this bottleneck.
    Only 30-50 spans maximum emerge from the other side.
    """
    
    def __init__(
        self,
        max_spans: int = DEFAULT_BOTTLENECK_SIZE,
        model: str = BOTTLENECK_MODEL,
        verbose: bool = True,
    ):
        self.max_spans = max_spans
        self.model = model
        self.verbose = verbose
    
    def filter(
        self,
        chunks: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        conn = None,
    ) -> BottleneckResult:
        """
        Filter chunks through the bottleneck.
        
        This is the HARD gate - synthesis sees ONLY what passes.
        
        Args:
            chunks: Raw chunks from retrieval (can be 100-300)
            parsed_query: The parsed query with task type
            conn: Database connection (for loading chunk text if needed)
        
        Returns:
            BottleneckResult with max 30-50 spans
        """
        start = time.time()
        result = BottleneckResult(chunks_input=len(chunks))
        
        if not chunks:
            return result
        
        if self.verbose:
            print(f"  [Bottleneck] Filtering {len(chunks)} chunks -> max {self.max_spans} spans", 
                  file=sys.stderr)
        
        # Grade in batches (preserve original chunk indices)
        all_graded = []
        for batch_start in range(0, len(chunks), BOTTLENECK_BATCH_SIZE):
            batch = chunks[batch_start:batch_start + BOTTLENECK_BATCH_SIZE]
            graded = self._grade_batch(batch, parsed_query)
            # Adjust chunk indices to global indices
            for g in graded:
                g["chunk_index"] = batch_start + g["chunk_index"]
            all_graded.extend(graded)
        
        result.spans_extracted = len(all_graded)
        
        # Filter to only passing spans
        passing = [g for g in all_graded if g["pass"]]
        
        # Sort by score descending
        passing.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top N
        top_spans = passing[:self.max_spans]
        
        # Convert to BottleneckSpan objects
        members_seen = set()
        for g in top_spans:
            chunk = chunks[g["chunk_index"]] if g["chunk_index"] < len(chunks) else {}
            
            span = BottleneckSpan(
                span_id=f"bs_{chunk.get('id', g['chunk_index'])}",
                chunk_id=chunk.get("id", 0),
                doc_id=chunk.get("doc_id"),
                page=chunk.get("page", ""),
                source_label=chunk.get("source_label", ""),
                span_text=chunk.get("text", "")[:800],
                relevance_score=g["score"],
                claim_supported=g["claim"],
                is_directly_responsive=g["responsive"],
                identifies_member=g.get("identifies_member", False),
                member_name=g.get("member_name", ""),
            )
            result.spans.append(span)
            
            if span.identifies_member and span.member_name:
                members_seen.add(span.member_name)
        
        result.spans_passed = len(result.spans)
        result.members_identified = list(members_seen)
        result.elapsed_ms = (time.time() - start) * 1000
        
        if self.verbose:
            print(f"    ┌─────────────────────────────────────────────────────────────", file=sys.stderr)
            print(f"    │ BOTTLENECK RESULT (HARD GATE)", file=sys.stderr)
            print(f"    │ Input: {result.chunks_input} chunks", file=sys.stderr)
            print(f"    │ Graded: {result.spans_extracted} spans", file=sys.stderr)
            print(f"    │ Passed: {result.spans_passed} spans (max allowed: {self.max_spans})", file=sys.stderr)
            print(f"    │ Rejected: {result.spans_extracted - result.spans_passed} spans", file=sys.stderr)
            if result.members_identified:
                print(f"    │ Members identified: {result.members_identified[:10]}", file=sys.stderr)
            print(f"    │", file=sys.stderr)
            print(f"    │ TOP PASSING SPANS:", file=sys.stderr)
            for i, span in enumerate(result.spans[:5]):
                print(f"    │   [{i}] score={span.relevance_score:.1f}, member={span.member_name or 'N/A'}", file=sys.stderr)
                print(f"    │       \"{span.span_text[:100]}...\"", file=sys.stderr)
            if len(result.spans) > 5:
                print(f"    │   ... and {len(result.spans) - 5} more spans", file=sys.stderr)
            print(f"    └─────────────────────────────────────────────────────────────", file=sys.stderr)
        
        return result
    
    def _grade_batch(
        self,
        batch: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
    ) -> List[Dict[str, Any]]:
        """Grade a batch of chunks."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_grade(batch)
        
        prompt = build_bottleneck_prompt(
            question=parsed_query.original_query,
            task_type=parsed_query.task_type,
            chunks=batch,
        )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": BOTTLENECK_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_grade(batch)
            
            data = json.loads(content)
            return data.get("evaluations", [])
            
        except Exception as e:
            if self.verbose:
                print(f"    [Bottleneck] Error: {e}", file=sys.stderr)
            return self._fallback_grade(batch)
    
    def _fallback_grade(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple fallback grading."""
        results = []
        for i, chunk in enumerate(batch):
            text = chunk.get("text", "").lower()
            # Simple heuristic: pass if contains key indicators
            has_member_indicator = any(w in text for w in ["member", "agent", "source", "recruited"])
            results.append({
                "chunk_index": i,
                "pass": has_member_indicator,
                "score": 5.0 if has_member_indicator else 2.0,
                "claim": "",
                "responsive": has_member_indicator,
                "identifies_member": False,
                "member_name": "",
            })
        return results


# =============================================================================
# Convenience function
# =============================================================================

def apply_bottleneck(
    chunks: List[Dict[str, Any]],
    parsed_query: ParsedQuery,
    max_spans: int = DEFAULT_BOTTLENECK_SIZE,
    verbose: bool = True,
) -> BottleneckResult:
    """
    Apply the evidence bottleneck to chunks.
    
    This MUST be called before any synthesis/claim extraction.
    """
    bottleneck = EvidenceBottleneck(max_spans=max_spans, verbose=verbose)
    return bottleneck.filter(chunks, parsed_query)
