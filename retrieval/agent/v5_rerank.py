"""
V5 Rerank - Span extraction and reranking

Instead of grading full chunks, we:
1. Extract 1-2 citeable spans from each chunk
2. Rate those spans for usefulness
3. Keep only top M spans

This prevents "HUAC dominates" type failures because the model can say
"this chunk is about HUAC broadly, not about network membership."
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from retrieval.agent.v5_types import CandidateSpan, GraderResult


# =============================================================================
# Configuration
# =============================================================================

RERANK_MODEL = "gpt-4o-mini"  # Fast model for extraction
RERANK_BATCH_SIZE = 10  # Chunks per batch
MAX_SPANS_PER_CHUNK = 2  # Extract up to 2 spans per chunk


# =============================================================================
# Extracted Span
# =============================================================================

@dataclass
class ExtractedSpan:
    """A citeable span extracted from a chunk."""
    
    span_id: str  # Unique ID
    chunk_id: int
    doc_id: Optional[int]
    page: Optional[str]
    source_label: str
    
    # The extracted span text (the quotable part)
    span_text: str
    
    # What claim this span supports (from extraction)
    supports_claim: str
    
    # Why this span is relevant (from extraction)
    relevance_reason: str
    
    # Original chunk context (for verification)
    original_chunk_text: str
    
    # Extraction metadata
    extraction_confidence: str = "medium"  # low/medium/high
    
    def to_candidate(self) -> CandidateSpan:
        """Convert to CandidateSpan for grading."""
        return CandidateSpan(
            candidate_id=self.span_id,
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            page=self.page,
            span_text=self.span_text,
            surrounding_context=self.original_chunk_text[:300],
            source_label=self.source_label,
            source_tool="span_extraction",
            metadata={
                "supports_claim": self.supports_claim,
                "relevance_reason": self.relevance_reason,
                "extraction_confidence": self.extraction_confidence,
            },
        )


# =============================================================================
# Span Extractor Prompt
# =============================================================================

EXTRACTOR_SYSTEM_PROMPT = """You are a research assistant extracting citeable evidence from archival documents.

Your job: Find the 1-2 most quotable, evidence-rich spans from each chunk that could help answer a specific research question.

Rules:
- Extract EXACT quotes from the text (copy verbatim, don't paraphrase)
- Each span should be 1-3 sentences that stand alone as evidence
- Prefer spans with names, dates, facts, or direct statements
- Skip chunks that have nothing relevant (return empty spans)
- Be conservative: if a chunk is only tangentially related, say so"""


def build_extractor_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Build prompt for span extraction."""
    
    chunks_section = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "")[:1500]  # Truncate long chunks
        source = chunk.get("source_label", "unknown")
        page = chunk.get("page", "")
        
        chunks_section.append(f"""
CHUNK {i} (source: {source}, page: {page}):
\"\"\"{chunk_text}\"\"\"
""")
    
    chunks_text = "\n---\n".join(chunks_section)
    
    return f"""RESEARCH QUESTION: {question}

CHUNKS TO ANALYZE:
{chunks_text}

For EACH chunk, extract 0-2 spans that could help answer the question.

Output JSON:
{{
  "extractions": [
    {{
      "chunk_index": 0,
      "spans": [
        {{
          "span_text": "exact quote from the chunk",
          "supports_claim": "what specific claim this supports (1 sentence)",
          "relevance": "why this is useful evidence (1 sentence)",
          "confidence": "high/medium/low"
        }}
      ]
    }},
    {{
      "chunk_index": 1,
      "spans": []  // Empty if nothing relevant
    }}
  ]
}}

IMPORTANT:
- span_text must be an EXACT quote from the chunk (copy-paste)
- If a chunk is about a related but different topic, return empty spans
- Prefer specific evidence over general statements"""


# =============================================================================
# Span Extractor Class
# =============================================================================

class SpanExtractor:
    """
    Extracts citeable spans from chunks.
    
    This is the "rerank" step that massively improves signal quality
    by having the model identify the best 1-2 quotable spans per chunk.
    """
    
    def __init__(
        self,
        model: str = RERANK_MODEL,
        batch_size: int = RERANK_BATCH_SIZE,
        max_spans_per_chunk: int = MAX_SPANS_PER_CHUNK,
        verbose: bool = True,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_spans_per_chunk = max_spans_per_chunk
        self.verbose = verbose
        self.total_calls = 0
        self.total_chunks_processed = 0
        self.total_spans_extracted = 0
    
    def extract_spans(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
    ) -> List[ExtractedSpan]:
        """
        Extract citeable spans from chunks.
        
        Args:
            question: The research question
            chunks: List of chunk dicts with 'id', 'text', 'doc_id', 'page', 'source_label'
        
        Returns:
            List of ExtractedSpan objects
        """
        if not chunks:
            return []
        
        all_spans = []
        
        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_spans = self._extract_batch(question, batch)
            all_spans.extend(batch_spans)
        
        if self.verbose:
            print(f"    [Extractor] {len(chunks)} chunks -> {len(all_spans)} spans extracted", 
                  file=sys.stderr)
        
        return all_spans
    
    def _extract_batch(
        self,
        question: str,
        batch: List[Dict[str, Any]],
    ) -> List[ExtractedSpan]:
        """Extract spans from a batch of chunks."""
        
        start = time.time()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if self.verbose:
                print("    [Extractor] No API key, using fallback", file=sys.stderr)
            return self._fallback_extraction(batch)
        
        prompt = build_extractor_prompt(question, batch)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=3000,
            )
            
            self.total_calls += 1
            elapsed = (time.time() - start) * 1000
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_extraction(batch)
            
            data = json.loads(content)
            extractions = data.get("extractions", [])
            
            # Convert to ExtractedSpan objects
            spans = []
            for extraction in extractions:
                chunk_idx = extraction.get("chunk_index", 0)
                if chunk_idx >= len(batch):
                    continue
                
                chunk = batch[chunk_idx]
                self.total_chunks_processed += 1
                
                for span_data in extraction.get("spans", [])[:self.max_spans_per_chunk]:
                    span_text = span_data.get("span_text", "").strip()
                    if not span_text:
                        continue
                    
                    # Generate unique span ID
                    import hashlib
                    span_id = hashlib.md5(
                        f"{chunk.get('id')}:{span_text[:50]}".encode()
                    ).hexdigest()[:12]
                    
                    span = ExtractedSpan(
                        span_id=span_id,
                        chunk_id=chunk.get("id"),
                        doc_id=chunk.get("doc_id"),
                        page=chunk.get("page"),
                        source_label=chunk.get("source_label", ""),
                        span_text=span_text,
                        supports_claim=span_data.get("supports_claim", ""),
                        relevance_reason=span_data.get("relevance", ""),
                        original_chunk_text=chunk.get("text", "")[:500],
                        extraction_confidence=span_data.get("confidence", "medium"),
                    )
                    spans.append(span)
                    self.total_spans_extracted += 1
            
            if self.verbose and spans:
                high_conf = sum(1 for s in spans if s.extraction_confidence == "high")
                print(f"    [Extractor] Batch: {len(batch)} chunks -> {len(spans)} spans "
                      f"({high_conf} high-conf) in {elapsed:.0f}ms", file=sys.stderr)
            
            return spans
            
        except Exception as e:
            if self.verbose:
                print(f"    [Extractor] Error: {e}", file=sys.stderr)
            return self._fallback_extraction(batch)
    
    def _fallback_extraction(self, batch: List[Dict[str, Any]]) -> List[ExtractedSpan]:
        """Fallback: use first 300 chars of each chunk as span."""
        spans = []
        for chunk in batch:
            text = chunk.get("text", "")[:300]
            if not text:
                continue
            
            import hashlib
            span_id = hashlib.md5(f"{chunk.get('id')}:fallback".encode()).hexdigest()[:12]
            
            span = ExtractedSpan(
                span_id=span_id,
                chunk_id=chunk.get("id"),
                doc_id=chunk.get("doc_id"),
                page=chunk.get("page"),
                source_label=chunk.get("source_label", ""),
                span_text=text,
                supports_claim="(fallback extraction)",
                relevance_reason="(fallback extraction)",
                original_chunk_text=text,
                extraction_confidence="low",
            )
            spans.append(span)
            self.total_chunks_processed += 1
            self.total_spans_extracted += 1
        
        return spans


# =============================================================================
# Reranker - Scores and filters extracted spans
# =============================================================================

RERANK_SCORE_PROMPT = """You are scoring evidence spans for a research question.

For each span, rate:
- useful_for_answer: Does this DIRECTLY help answer the question? (boolean)
- score: 0-10 (0=irrelevant, 5=somewhat relevant, 10=directly answers the question)
- what_it_supports: What specific claim does this support? (1 sentence)

Be STRICT. A span about "HUAC generally" is NOT useful for "who were network members."
Only spans with specific, quotable facts should score 7+."""


def build_rerank_prompt(question: str, spans: List[ExtractedSpan]) -> str:
    """Build prompt for reranking spans."""
    
    spans_section = []
    for i, span in enumerate(spans):
        spans_section.append(f"""
SPAN {i}:
Source: {span.source_label}, page {span.page}
Text: "{span.span_text}"
Initial claim: {span.supports_claim}
""")
    
    spans_text = "\n---\n".join(spans_section)
    
    return f"""QUESTION: {question}

SPANS TO SCORE:
{spans_text}

Output JSON:
{{
  "scores": [
    {{"span_index": 0, "useful": true, "score": 8, "supports": "X was a member of Y"}},
    {{"span_index": 1, "useful": false, "score": 2, "supports": ""}}
  ]
}}"""


@dataclass
class RerankResult:
    """Result of reranking a span."""
    span: ExtractedSpan
    useful: bool
    score: float  # 0-10
    supports_claim: str  # Refined claim from reranker


class SpanReranker:
    """
    Scores and filters extracted spans.
    
    This is the second pass that ensures we only keep
    the most useful evidence for synthesis.
    """
    
    def __init__(
        self,
        model: str = RERANK_MODEL,
        batch_size: int = 15,
        verbose: bool = True,
    ):
        self.model = model
        self.batch_size = batch_size
        self.verbose = verbose
        self.total_calls = 0
    
    def rerank(
        self,
        question: str,
        spans: List[ExtractedSpan],
        top_k: int = 30,
    ) -> List[RerankResult]:
        """
        Rerank spans and return top K.
        
        Args:
            question: The research question
            spans: Extracted spans to rerank
            top_k: Number of top spans to keep
        
        Returns:
            Top K RerankResults sorted by score
        """
        if not spans:
            return []
        
        all_results = []
        
        # Score in batches
        for i in range(0, len(spans), self.batch_size):
            batch = spans[i:i + self.batch_size]
            batch_results = self._rerank_batch(question, batch)
            all_results.extend(batch_results)
        
        # Sort by score descending
        all_results.sort(key=lambda r: r.score, reverse=True)
        
        # Keep only useful spans
        useful_results = [r for r in all_results if r.useful]
        
        if self.verbose:
            avg_score = sum(r.score for r in useful_results) / len(useful_results) if useful_results else 0
            print(f"    [Reranker] {len(spans)} spans -> {len(useful_results)} useful (avg score: {avg_score:.1f})", 
                  file=sys.stderr)
        
        return useful_results[:top_k]
    
    def _rerank_batch(
        self,
        question: str,
        batch: List[ExtractedSpan],
    ) -> List[RerankResult]:
        """Rerank a batch of spans."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_rerank(batch)
        
        prompt = build_rerank_prompt(question, batch)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RERANK_SCORE_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            
            self.total_calls += 1
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_rerank(batch)
            
            data = json.loads(content)
            scores_data = data.get("scores", [])
            
            # Map back to spans
            results = []
            scores_by_idx = {s.get("span_index"): s for s in scores_data}
            
            for i, span in enumerate(batch):
                score_data = scores_by_idx.get(i, {})
                results.append(RerankResult(
                    span=span,
                    useful=score_data.get("useful", False),
                    score=float(score_data.get("score", 0)),
                    supports_claim=score_data.get("supports", span.supports_claim),
                ))
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"    [Reranker] Error: {e}", file=sys.stderr)
            return self._fallback_rerank(batch)
    
    def _fallback_rerank(self, batch: List[ExtractedSpan]) -> List[RerankResult]:
        """Fallback: use extraction confidence as proxy for score."""
        results = []
        for span in batch:
            score = {"high": 7.0, "medium": 5.0, "low": 3.0}.get(
                span.extraction_confidence, 5.0
            )
            results.append(RerankResult(
                span=span,
                useful=score >= 5.0,
                score=score,
                supports_claim=span.supports_claim,
            ))
        return results


# =============================================================================
# Combined Rerank Pipeline
# =============================================================================

def rerank_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 30,
    verbose: bool = True,
) -> List[RerankResult]:
    """
    Full rerank pipeline: extract spans, then score and filter.
    
    Args:
        question: Research question
        chunks: Raw chunks from retrieval (list of dicts with 'id', 'text', etc.)
        top_k: Number of top spans to return
        verbose: Print progress
    
    Returns:
        Top K RerankResults
    """
    if verbose:
        print(f"  [Rerank] Processing {len(chunks)} chunks...", file=sys.stderr)
    
    # Step 1: Extract spans
    extractor = SpanExtractor(verbose=verbose)
    spans = extractor.extract_spans(question, chunks)
    
    if not spans:
        if verbose:
            print(f"  [Rerank] No spans extracted", file=sys.stderr)
        return []
    
    # Step 2: Rerank and filter
    reranker = SpanReranker(verbose=verbose)
    results = reranker.rerank(question, spans, top_k=top_k)
    
    if verbose:
        print(f"  [Rerank] Final: {len(results)} useful spans", file=sys.stderr)
    
    return results
