"""
V3 Evidence Builder - Span mining, reranking, and banding.

The Evidence Builder:
1. Mines spans from retrieved chunks using SpanMiner
2. Embeds spans and reranks by query similarity
3. Bands into cite_spans (can cite) and harvest_spans (search hints only)

V3 Rule: Final answers cite ONLY from cite_spans unless labeled "suggestive".
"""

import sys
import time
import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from retrieval.spans import SpanMiner, Span


@dataclass
class EvidenceSpan:
    """A cite-able span with score and band assignment."""
    span_id: str           # chunk_id:start:end
    chunk_id: int
    doc_id: int
    page_ref: str
    start_char: int
    end_char: int
    quote: str             # The actual text
    score: float           # Reranked similarity
    band: str              # "cite" or "harvest"
    span_hash: str = ""    # SHA256 for reproducibility
    attest_text: str = ""  # Extended context window (+/- 200 chars) for V4 attestation
    
    def __post_init__(self):
        if not self.span_hash:
            self.span_hash = hashlib.sha256(
                f"{self.chunk_id}:{self.start_char}:{self.end_char}:{self.quote}".encode()
            ).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "span_id": self.span_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_ref": self.page_ref,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote": self.quote[:500],  # Cap for storage
            "score": self.score,
            "band": self.band,
            "span_hash": self.span_hash,
        }
        # Only include attest_text if present (avoid bloating storage)
        if self.attest_text and self.attest_text != self.quote:
            result["attest_text"] = self.attest_text[:1000]  # Cap context
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceSpan":
        return cls(
            span_id=data.get("span_id", ""),
            chunk_id=data.get("chunk_id", 0),
            doc_id=data.get("doc_id", 0),
            page_ref=data.get("page_ref", "p0"),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            quote=data.get("quote", ""),
            score=data.get("score", 0.0),
            band=data.get("band", "harvest"),
            span_hash=data.get("span_hash", ""),
            attest_text=data.get("attest_text", ""),
        )


def attach_context_window(span: EvidenceSpan, chunk_text: str, window: int = 200) -> str:
    """
    Compute extended context window for attestation.
    
    The context window extends +/- window chars from the quote within the same chunk.
    Used by V4 verifier for OCR-robust entity attestation.
    
    Args:
        span: The EvidenceSpan to attach context to
        chunk_text: Full text of the source chunk
        window: Number of chars to extend on each side (default 200)
    
    Returns:
        Extended context text
    """
    if not chunk_text:
        return span.quote
    
    start = max(0, span.start_char - window)
    end = min(len(chunk_text), span.end_char + window)
    return chunk_text[start:end]


@dataclass
class EvidenceStats:
    """Statistics about the evidence set."""
    total_chunks: int
    total_spans_mined: int
    cite_span_count: int
    harvest_span_count: int
    unique_docs: int
    score_distribution: Dict[str, float]  # min, max, mean, median
    elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "total_spans_mined": self.total_spans_mined,
            "cite_span_count": self.cite_span_count,
            "harvest_span_count": self.harvest_span_count,
            "unique_docs": self.unique_docs,
            "score_distribution": self.score_distribution,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class EvidenceSet:
    """
    The evidence set with cite and harvest bands.
    
    cite_spans: Top N spans that can be cited in claims
    harvest_spans: Next M spans that can inform search but not cited
    """
    cite_spans: List[EvidenceSpan]
    harvest_spans: List[EvidenceSpan]
    stats: EvidenceStats
    query_text: str = ""
    evidence_set_id: str = ""
    
    def __post_init__(self):
        if not self.evidence_set_id:
            # Generate deterministic ID from content
            content = f"{self.query_text}:{len(self.cite_spans)}:{len(self.harvest_spans)}"
            self.evidence_set_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_span_by_id(self, span_id: str) -> Optional[EvidenceSpan]:
        """Look up a span by ID."""
        for span in self.cite_spans + self.harvest_spans:
            if span.span_id == span_id:
                return span
        return None
    
    def get_cite_span_ids(self) -> List[str]:
        """Get list of cite span IDs."""
        return [s.span_id for s in self.cite_spans]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_set_id": self.evidence_set_id,
            "query_text": self.query_text,
            "cite_spans": [s.to_dict() for s in self.cite_spans],
            "harvest_spans": [s.to_dict() for s in self.harvest_spans],
            "stats": self.stats.to_dict(),
        }


class EvidenceBuilder:
    """
    Builds EvidenceSet from retrieved chunks.
    
    Pipeline:
    1. Load chunk texts from database
    2. Mine spans using SpanMiner
    3. Embed spans + query
    4. Rerank by cosine similarity
    5. Band into cite_spans (top N) and harvest_spans (next M)
    """
    
    def __init__(
        self,
        cite_cap: int = 120,
        harvest_cap: int = 240,
        span_window_chars: int = 500,
        verbose: bool = True,
    ):
        self.cite_cap = cite_cap
        self.harvest_cap = harvest_cap
        self.span_window_chars = span_window_chars
        self.verbose = verbose
        self.span_miner = SpanMiner()
    
    def build(
        self,
        chunk_ids: List[int],
        query: str,
        conn,
        scores: Optional[Dict[int, float]] = None,
    ) -> EvidenceSet:
        """
        Build EvidenceSet from chunk IDs.
        
        Args:
            chunk_ids: List of chunk IDs from tool execution
            query: Query text for reranking
            conn: Database connection
            scores: Optional pre-computed scores from retrieval
        
        Returns:
            EvidenceSet with cite and harvest bands
        """
        if self.verbose:
            print(f"\n  [Evidence] Building from {len(chunk_ids)} chunks...", file=sys.stderr)
        
        start_time = time.time()
        
        if not chunk_ids:
            return self._empty_evidence_set(query)
        
        # 1. Load chunk texts
        chunks = self._load_chunks(conn, chunk_ids)
        if self.verbose:
            print(f"    Loaded {len(chunks)} chunk texts", file=sys.stderr)
        
        if not chunks:
            return self._empty_evidence_set(query)
        
        # 2. Mine spans
        all_spans = self._mine_spans(chunks, conn)
        if self.verbose:
            print(f"    Mined {len(all_spans)} spans", file=sys.stderr)
        
        if not all_spans:
            return self._empty_evidence_set(query)
        
        # 3. Embed and rerank
        scored_spans = self._rerank_spans(all_spans, query)
        if self.verbose:
            print(f"    Reranked {len(scored_spans)} spans", file=sys.stderr)
        
        # 4. Band into cite and harvest
        cite_spans, harvest_spans = self._band_spans(scored_spans)
        if self.verbose:
            print(f"    Cite: {len(cite_spans)}, Harvest: {len(harvest_spans)}", file=sys.stderr)
        
        # 5. Compute stats
        elapsed_ms = (time.time() - start_time) * 1000
        stats = self._compute_stats(chunks, all_spans, cite_spans, harvest_spans, elapsed_ms)
        
        return EvidenceSet(
            cite_spans=cite_spans,
            harvest_spans=harvest_spans,
            stats=stats,
            query_text=query,
        )
    
    def _load_chunks(self, conn, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Load chunk texts from database."""
        if not chunk_ids:
            return []
        
        # Ensure we're in a clean transaction state
        try:
            conn.rollback()
        except:
            pass
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.text, cm.document_id, cm.first_page_id
                FROM chunks c
                LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE c.id = ANY(%s)
            """, (chunk_ids,))
            
            chunks = []
            for row in cur.fetchall():
                chunk_id, text, doc_id, first_page_id = row
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': text or "",
                    'doc_id': doc_id or 0,
                    'page_ref': f"p{first_page_id}" if first_page_id else "p0",
                })
        
        return chunks
    
    def _mine_spans(self, chunks: List[Dict], conn) -> List[Tuple[Span, Dict]]:
        """Mine spans from chunks using SpanMiner."""
        all_spans = []
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            text = chunk.get('text', '')
            doc_id = chunk.get('doc_id', 0)
            page_ref = chunk.get('page_ref', 'p0')
            
            if not text:
                continue
            
            # Mine spans from this chunk
            spans = self.span_miner.mine_spans(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_ref=page_ref,
                text=text,
            )
            
            for span in spans:
                all_spans.append((span, chunk))
        
        return all_spans
    
    def _rerank_spans(
        self,
        spans: List[Tuple[Span, Dict]],
        query: str,
    ) -> List[Tuple[EvidenceSpan, float]]:
        """
        Embed spans and rerank by query similarity.
        
        Returns list of (EvidenceSpan, score) sorted by score descending.
        """
        if not spans:
            return []
        
        # Get embeddings
        try:
            from retrieval.ops import embed_query
            
            query_emb = embed_query(query)
            
            # Batch embed spans (in groups to avoid API limits)
            span_texts = [s[0].text for s in spans]
            span_embeddings = self._batch_embed(span_texts)
            
            # Score each span
            scored = []
            for i, (span, chunk) in enumerate(spans):
                if i < len(span_embeddings):
                    score = self._cosine_similarity(query_emb, span_embeddings[i])
                else:
                    score = 0.0
                
                evidence_span = EvidenceSpan(
                    span_id=span.span_id,
                    chunk_id=span.chunk_id,
                    doc_id=span.doc_id,
                    page_ref=span.page_ref,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    quote=span.text,
                    score=score,
                    band="cite",  # Will be updated in banding
                    span_hash=span.span_hash,
                )
                scored.append((evidence_span, score))
            
            # Sort by score descending, then span_id for determinism
            scored.sort(key=lambda x: (-x[1], x[0].span_id))
            
            return scored
            
        except Exception as e:
            print(f"    Rerank error: {e}", file=sys.stderr)
            # Fallback: return spans without scoring
            return [
                (EvidenceSpan(
                    span_id=s[0].span_id,
                    chunk_id=s[0].chunk_id,
                    doc_id=s[0].doc_id,
                    page_ref=s[0].page_ref,
                    start_char=s[0].start_char,
                    end_char=s[0].end_char,
                    quote=s[0].text,
                    score=0.5,
                    band="cite",
                    span_hash=s[0].span_hash,
                ), 0.5)
                for s in spans
            ]
    
    def _batch_embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Batch embed texts."""
        import os
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return [[0.0] * 1536] * len(texts)
        
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        client = OpenAI(api_key=api_key)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate long texts
            batch = [t[:8000] for t in batch]
            
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                )
                for item in response.data:
                    all_embeddings.append(item.embedding)
            except Exception as e:
                print(f"    Embed batch error: {e}", file=sys.stderr)
                # Fill with zeros
                all_embeddings.extend([[0.0] * 1536] * len(batch))
        
        return all_embeddings
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def _band_spans(
        self,
        scored_spans: List[Tuple[EvidenceSpan, float]],
    ) -> Tuple[List[EvidenceSpan], List[EvidenceSpan]]:
        """
        Band spans into cite (can cite) and harvest (search hints only).
        """
        cite_spans = []
        harvest_spans = []
        
        for i, (span, score) in enumerate(scored_spans):
            span.score = score
            
            if i < self.cite_cap:
                span.band = "cite"
                cite_spans.append(span)
            elif i < self.cite_cap + self.harvest_cap:
                span.band = "harvest"
                harvest_spans.append(span)
            else:
                break  # Cap reached
        
        return cite_spans, harvest_spans
    
    def _compute_stats(
        self,
        chunks: List[Dict],
        all_spans: List,
        cite_spans: List[EvidenceSpan],
        harvest_spans: List[EvidenceSpan],
        elapsed_ms: float,
    ) -> EvidenceStats:
        """Compute statistics for the evidence set."""
        all_scores = [s.score for s in cite_spans + harvest_spans if s.score > 0]
        unique_docs = len({s.doc_id for s in cite_spans + harvest_spans})
        
        score_dist = {}
        if all_scores:
            score_dist = {
                "min": min(all_scores),
                "max": max(all_scores),
                "mean": statistics.mean(all_scores),
                "median": statistics.median(all_scores),
            }
        
        return EvidenceStats(
            total_chunks=len(chunks),
            total_spans_mined=len(all_spans),
            cite_span_count=len(cite_spans),
            harvest_span_count=len(harvest_spans),
            unique_docs=unique_docs,
            score_distribution=score_dist,
            elapsed_ms=elapsed_ms,
        )
    
    def _empty_evidence_set(self, query: str) -> EvidenceSet:
        """Create an empty evidence set."""
        return EvidenceSet(
            cite_spans=[],
            harvest_spans=[],
            stats=EvidenceStats(
                total_chunks=0,
                total_spans_mined=0,
                cite_span_count=0,
                harvest_span_count=0,
                unique_docs=0,
                score_distribution={},
            ),
            query_text=query,
        )
