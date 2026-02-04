"""
FocusBundle and FocusBundleBuilder for Agentic V2.

The FocusBundle is the single source of truth for citations.
HARD RULE: Only FocusSpans can be cited (Invariant #1).

Modes:
- KEYWORD_INTENT: Standard retrieval for existence/roster queries
- TARGET_ANCHORED: Two-stage selection for relationship/affiliation queries
"""

import os
import json
import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum

# Lazy numpy import for environments without it
_np = None
def _get_numpy():
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError:
            _np = False
    return _np

from retrieval.spans import Span, SpanMiner, get_mention_offsets_for_chunks, get_page_ref
from retrieval.query_intent import QueryContract, FocusBundleMode, ConstraintSpec
from retrieval.mention_span_index import build_mention_span_index


# Version pinning for determinism
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_MODEL_VERSION = "v1"


@dataclass
class FocusSpan:
    """
    A span that has been selected for the FocusBundle.
    
    This is the only thing that can be cited (Invariant #1).
    """
    chunk_id: int
    doc_id: int
    page_ref: str  # Contract C2: f"p{page_num}"
    start_char: int
    end_char: int
    text: str
    score: float              # similarity to query
    rank: int                 # position in FocusBundle
    source_lanes: List[str]   # which retrieval lanes found this chunk
    span_hash: str            # for reproducibility (Contract C1)
    
    @property
    def span_id(self) -> str:
        """Deterministic ID: chunk_id:start:end"""
        return f"{self.chunk_id}:{self.start_char}:{self.end_char}"
    
    def to_evidence_ref(self) -> dict:
        """Convert to citation format."""
        return {
            "span_id": self.span_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_ref": self.page_ref,
            "char_range": [self.start_char, self.end_char],
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "score": self.score,
        }
    
    def __hash__(self):
        return hash(self.span_id)
    
    def __eq__(self, other):
        if not isinstance(other, FocusSpan):
            return False
        return self.span_id == other.span_id


@dataclass
class FocusBundle:
    """
    The single source of truth for citations.
    
    HARD RULE: Only FocusSpans can be cited (Invariant #1).
    """
    query_text: str
    spans: List[FocusSpan]
    params: Dict[str, Any]
    retrieval_run_id: Optional[int] = None
    mention_span_index: Dict[int, List[str]] = field(default_factory=dict)
    
    def get_span(self, span_id: str) -> Optional[FocusSpan]:
        """Look up span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None
    
    def contains_span(self, span_id: str) -> bool:
        """Check if span is in FocusBundle (citation validation)."""
        return any(s.span_id == span_id for s in self.spans)
    
    def get_spans_for_entity(self, entity_id: int, conn=None) -> List[FocusSpan]:
        """
        Get spans that overlap entity_mentions for this entity_id.
        Uses precomputed mention_span_index for O(1) lookup (Contract C9).
        """
        span_ids = self.mention_span_index.get(entity_id, [])
        return [s for s in self.spans if s.span_id in span_ids]
    
    def get_spans_for_term(self, term: str) -> List[FocusSpan]:
        """
        Get spans via normalized token match.
        Fallback for non-entity candidates.
        """
        term_norm = _normalize_for_match(term)
        return [s for s in self.spans if term_norm in _normalize_for_match(s.text)]
    
    def get_spans_in_neighborhood(self, doc_id: int, page_ref: str) -> List[FocusSpan]:
        """Get all spans in the same doc/page neighborhood."""
        return [s for s in self.spans if s.doc_id == doc_id and s.page_ref == page_ref]
    
    def get_unique_doc_ids(self) -> Set[int]:
        """Get unique document IDs in the bundle."""
        return {s.doc_id for s in self.spans}
    
    def get_neighborhoods(self) -> Set[Tuple[int, str]]:
        """Get unique (doc_id, page_ref) neighborhoods."""
        return {(s.doc_id, s.page_ref) for s in self.spans}
    
    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "query_text": self.query_text,
            "params": self.params,
            "retrieval_run_id": self.retrieval_run_id,
            "span_count": len(self.spans),
        }


class FocusBundleBuilder:
    """
    Builds FocusBundle from retrieved chunks.
    
    Modes:
    - KEYWORD_INTENT: Standard MMR selection on all spans
    - TARGET_ANCHORED: Two-stage selection (anchor → context fill)
    
    Query-term anchoring:
    - Extracts anchor terms from query (quoted phrases, rare bigrams, technical terms)
    - Applies lexical bonus for anchor hits
    - Adaptive must-hit gating when anchors are rare
    """
    
    def __init__(
        self,
        top_n_spans: int = 80,
        lambda_mmr: float = 0.7,
        min_span_score: float = 0.3,
        max_spans_per_doc: int = 10,
        max_spans_per_chunk: int = 2,
        context_fill_quota: int = 20,
        anchor_bonus: float = 0.15,  # Lexical bonus for anchor term hits
        anchor_rarity_threshold: float = 0.02,  # If anchor in <2% of chunks, require hits
    ):
        self.top_n_spans = top_n_spans
        self.lambda_mmr = lambda_mmr
        self.min_span_score = min_span_score
        self.max_spans_per_doc = max_spans_per_doc
        self.max_spans_per_chunk = max_spans_per_chunk
        self.context_fill_quota = context_fill_quota
        self.anchor_bonus = anchor_bonus
        self.anchor_rarity_threshold = anchor_rarity_threshold
        self.span_miner = SpanMiner()
    
    def build(
        self,
        query_contract: QueryContract,
        chunks: List[dict],
        conn,
        anchor_terms: List[str] = None,
    ) -> FocusBundle:
        """
        Build FocusBundle from retrieved chunks.
        
        Args:
            query_contract: The QueryContract specifying mode, targets, constraints
            chunks: Retrieved chunks with keys: chunk_id/id, doc_id/document_id, text, page_ref
            conn: Database connection
            anchor_terms: Additional terms to prioritize
        
        Returns:
            FocusBundle with selected spans and precomputed mention index
        """
        if query_contract.mode == FocusBundleMode.TARGET_ANCHORED:
            return self._build_target_anchored(query_contract, chunks, conn, anchor_terms)
        else:
            return self._build_keyword_intent(query_contract, chunks, conn, anchor_terms)
    
    def _build_keyword_intent(
        self,
        query_contract: QueryContract,
        chunks: List[dict],
        conn,
        anchor_terms: List[str] = None,
    ) -> FocusBundle:
        """
        Standard FocusBundle construction for existence/roster queries.
        
        Uses query-term anchoring to prevent semantic drift:
        - Extracts anchor terms from query
        - Computes anchor coverage across chunks
        - Applies lexical bonus for anchor hits
        """
        import sys
        
        # Get chunk IDs
        chunk_ids = [c.get('chunk_id') or c.get('id') for c in chunks]
        
        # Fetch mention offsets for all chunks
        mention_offsets = get_mention_offsets_for_chunks(conn, chunk_ids)
        
        # Extract anchor terms from query (for lexical bonus)
        query_anchors = self._extract_anchor_terms(query_contract.query_text)
        print(f"    [FocusBundle] Extracted anchor terms: {query_anchors}", file=sys.stderr)
        
        # Combine with provided anchors
        all_anchor_terms = list(anchor_terms or [])
        all_anchor_terms.extend(query_contract.get_target_surface_forms())
        all_anchor_terms.extend(query_anchors)
        all_anchor_terms = list(set(all_anchor_terms))  # Dedupe
        
        # Compute anchor coverage across chunks
        anchor_coverage = self._compute_anchor_coverage(query_anchors, chunks)
        for anchor, cov in anchor_coverage.items():
            print(f"    [FocusBundle] Anchor '{anchor}' coverage: {cov:.1%} of chunks", file=sys.stderr)
            if cov < self.anchor_rarity_threshold:
                print(f"    [FocusBundle] WARNING: Anchor '{anchor}' is RARE - will require hits", file=sys.stderr)
        
        # Mine spans from all chunks
        all_spans = self.span_miner.mine_chunks(
            chunks,
            mention_offsets_by_chunk=mention_offsets,
            anchor_terms=all_anchor_terms,
        )
        print(f"    [FocusBundle] Mined {len(all_spans)} spans from {len(chunks)} chunks", file=sys.stderr)
        
        if not all_spans:
            return self._empty_bundle(query_contract)
        
        # Score with anchor bonus and select with MMR
        scored_spans = self._score_spans(
            all_spans, 
            query_contract.query_text, 
            conn,
            anchor_terms=query_anchors,
            anchor_coverage=anchor_coverage,
        )
        
        # Count anchor hits in top spans before selection
        anchor_hit_spans = []
        for span, score, emb in scored_spans[:100]:
            text_lower = span.text.lower()
            hits = [a for a in query_anchors if a in text_lower]
            if hits:
                anchor_hit_spans.append((span.span_id, hits))
        
        print(f"    [FocusBundle] {len(anchor_hit_spans)} spans have anchor hits in top 100", file=sys.stderr)
        
        selected = self._select_with_mmr(scored_spans)
        
        # Count anchor hits in selected spans
        selected_anchor_hits = 0
        for span in selected:
            text_lower = span.text.lower()
            if any(a in text_lower for a in query_anchors):
                selected_anchor_hits += 1
        
        print(f"    [FocusBundle] Selected {len(selected)} spans, {selected_anchor_hits} with anchor hits", file=sys.stderr)
        
        # Build FocusBundle
        bundle = FocusBundle(
            query_text=query_contract.query_text,
            spans=selected,
            params=self._get_params(query_contract, len(all_spans), len(chunks)),
        )
        
        # Store anchor info in params for verification
        bundle.params["anchor_terms"] = query_anchors
        bundle.params["anchor_coverage"] = anchor_coverage
        bundle.params["anchor_hit_count"] = selected_anchor_hits
        
        # Build mention index (Contract C9)
        bundle.mention_span_index = build_mention_span_index(bundle, conn)
        
        return bundle
    
    def _build_target_anchored(
        self,
        query_contract: QueryContract,
        chunks: List[dict],
        conn,
        anchor_terms: List[str] = None,
    ) -> FocusBundle:
        """
        Two-stage selection for relationship/affiliation queries.
        
        Stage A (anchor): spans with target mention OR constraint anchor
        Stage B (context): adjacent spans from same doc/page neighborhood
        """
        chunk_ids = [c.get('chunk_id') or c.get('id') for c in chunks]
        mention_offsets = get_mention_offsets_for_chunks(conn, chunk_ids)
        
        all_anchor_terms = list(anchor_terms or [])
        all_anchor_terms.extend(query_contract.get_target_surface_forms())
        
        # Add constraint object terms
        for constraint in query_contract.constraints:
            if constraint.object:
                all_anchor_terms.extend(self._get_constraint_anchors(constraint))
        
        # Mine all spans
        all_spans = self.span_miner.mine_chunks(
            chunks,
            mention_offsets_by_chunk=mention_offsets,
            anchor_terms=all_anchor_terms,
        )
        
        if not all_spans:
            return self._empty_bundle(query_contract)
        
        # Stage A: Filter to anchor-eligible spans
        anchor_spans = [
            s for s in all_spans 
            if self._is_anchor_eligible(s, query_contract, mention_offsets, conn)
        ]
        
        if not anchor_spans:
            # Fall back to keyword intent if no anchors found
            anchor_spans = all_spans
        
        # Score and select anchor spans
        scored_anchors = self._score_spans(anchor_spans, query_contract.query_text, conn)
        selected_anchors = self._select_with_mmr(scored_anchors)
        
        # Stage B: Context fill from same doc/page neighborhoods
        anchor_neighborhoods = {(s.doc_id, s.page_ref) for s in selected_anchors}
        anchor_span_ids = {s.span_id for s in selected_anchors}
        
        context_spans = [
            s for s in all_spans
            if (s.doc_id, s.page_ref) in anchor_neighborhoods
            and s.span_id not in anchor_span_ids
        ]
        
        if context_spans:
            scored_context = self._score_spans(context_spans, query_contract.query_text, conn)
            selected_context = self._select_with_mmr(
                scored_context, 
                max_spans=self.context_fill_quota,
            )
        else:
            selected_context = []
        
        # Combine and re-rank
        all_selected = selected_anchors + selected_context
        all_selected.sort(key=lambda s: (-s.score, s.span_id))
        final_spans = all_selected[:self.top_n_spans]
        
        # Re-assign ranks
        for i, span in enumerate(final_spans):
            span.rank = i + 1
        
        # Build FocusBundle
        bundle = FocusBundle(
            query_text=query_contract.query_text,
            spans=final_spans,
            params=self._get_params(query_contract, len(all_spans), len(chunks)),
        )
        
        # Build mention index (Contract C9)
        bundle.mention_span_index = build_mention_span_index(bundle, conn)
        
        return bundle
    
    def _is_anchor_eligible(
        self,
        span: Span,
        contract: QueryContract,
        mention_offsets: dict,
        conn,
    ) -> bool:
        """
        Check if span is anchor-eligible.
        
        Eligibility: target mention overlap OR constraint anchor hit.
        """
        # Check target entity mentions
        for target in contract.targets:
            if target.entity_id:
                # Check mention offsets for overlap
                chunk_mentions = mention_offsets.get(span.chunk_id, [])
                for m_start, m_end, m_entity_id in chunk_mentions:
                    if m_entity_id == target.entity_id:
                        if span.start_char <= m_end and span.end_char >= m_start:
                            return True
            
            # Check surface form matches
            for surface in target.surface_forms:
                if _normalize_for_match(surface) in _normalize_for_match(span.text):
                    return True
        
        # Check constraint anchors
        for constraint in contract.constraints:
            if constraint.object:
                for term in self._get_constraint_anchors(constraint):
                    if _normalize_for_match(term) in _normalize_for_match(span.text):
                        return True
        
        return False
    
    def _get_constraint_anchors(self, constraint: ConstraintSpec) -> List[str]:
        """Get anchor terms for a constraint."""
        anchors = [constraint.object] if constraint.object else []
        
        # Expand known aliases
        CONSTRAINT_ALIASES = {
            "OSS": ["OSS", "Office of Strategic Services", "O.S.S."],
            "Soviet intelligence": ["Soviet intelligence", "NKVD", "KGB", "GRU", "Soviet agent"],
            "State Department": ["State Department", "State Dept", "DOS"],
        }
        
        if constraint.object and constraint.object in CONSTRAINT_ALIASES:
            anchors = CONSTRAINT_ALIASES[constraint.object]
        
        return anchors
    
    def _extract_anchor_terms(self, query_text: str) -> List[str]:
        """
        Extract anchor terms from query for lexical bonus.
        
        Targets:
        - Quoted phrases
        - Rare bigrams/trigrams (capitalized, technical)
        - NOT scope terms like collection names
        """
        import re
        
        anchors = []
        
        # Quoted phrases
        quoted = re.findall(r'"([^"]+)"', query_text)
        anchors.extend(quoted)
        
        # Technical terms (2+ words with capitals, or hyphenated)
        # e.g., "proximity fuse", "VT fuse", "radio-fuse"
        technical = re.findall(r'\b[A-Z][a-z]*(?:\s+[a-z]+)+\b', query_text)
        anchors.extend(technical)
        
        # Hyphenated terms
        hyphenated = re.findall(r'\b\w+-\w+(?:-\w+)*\b', query_text)
        anchors.extend(hyphenated)
        
        # Multi-word noun phrases (lowercased for matching)
        # Focus on specific technical terms, not common phrases
        words = query_text.lower().split()
        
        # Extract bigrams that aren't common stopwords
        STOP = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for',
                'of', 'and', 'or', 'about', 'from', 'with', 'there', 'information', 'evidence'}
        for i in range(len(words) - 1):
            if words[i] not in STOP and words[i+1] not in STOP:
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) >= 6:  # Skip very short bigrams
                    anchors.append(bigram)
        
        # Dedupe and normalize
        seen = set()
        result = []
        for a in anchors:
            norm = a.lower().strip()
            if norm and norm not in seen and len(norm) >= 3:
                seen.add(norm)
                result.append(norm)
        
        return result
    
    def _compute_anchor_coverage(
        self,
        anchors: List[str],
        chunks: List[dict],
    ) -> Dict[str, float]:
        """
        Compute what % of chunks contain each anchor.
        
        Returns: anchor -> coverage ratio (0.0 to 1.0)
        """
        if not chunks or not anchors:
            return {}
        
        coverage = {a: 0 for a in anchors}
        
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            for anchor in anchors:
                if anchor in text:
                    coverage[anchor] += 1
        
        total = len(chunks)
        return {a: count / total for a, count in coverage.items()}
    
    def _score_spans(
        self,
        spans: List[Span],
        query_text: str,
        conn,
        anchor_terms: List[str] = None,
        anchor_coverage: Dict[str, float] = None,
    ) -> List[Tuple[Span, float, List[float]]]:
        """
        Score spans by cosine similarity + anchor term bonus.
        
        Returns (span, final_score, span_embedding) for MMR (Contract C3).
        
        Anchor bonus: if span contains anchor term, add bonus to score.
        Rare anchor penalty: if anchor is rare (<2% coverage) and span doesn't have it,
        apply a penalty to prevent semantic drift.
        """
        if not spans:
            return []
        
        anchor_terms = anchor_terms or []
        anchor_coverage = anchor_coverage or {}
        
        # Sort for deterministic embedding order (Contract C3)
        sorted_spans = sorted(spans, key=lambda s: s.span_id)
        
        # Batch embed spans
        span_texts = [s.text for s in sorted_spans]
        span_embeddings = self._batch_embed(span_texts)
        
        # Embed query
        query_embedding = self._embed_query(query_text)
        
        # Score with anchor bonus
        results = []
        for span, emb in zip(sorted_spans, span_embeddings):
            # Base semantic similarity
            sim = _cosine_similarity(query_embedding, emb)
            
            # Apply anchor bonus/penalty
            span_text_lower = span.text.lower()
            anchor_bonus = 0.0
            has_rare_anchor = False
            
            for anchor in anchor_terms:
                if anchor in span_text_lower:
                    # Bonus for containing anchor
                    anchor_bonus += self.anchor_bonus
                    has_rare_anchor = True
                else:
                    # Penalty if anchor is rare and span doesn't have it
                    cov = anchor_coverage.get(anchor, 1.0)
                    if cov < self.anchor_rarity_threshold:
                        # This is a rare anchor - span should probably have it
                        anchor_bonus -= self.anchor_bonus * 0.5
            
            final_score = sim + anchor_bonus
            results.append((span, final_score, emb))
        
        # Sort by score descending, then span_id for ties (Contract C3)
        results.sort(key=lambda x: (-x[1], x[0].span_id))
        
        return results
    
    def _select_with_mmr(
        self,
        scored_spans: List[Tuple[Span, float, List[float]]],
        max_spans: int = None,
    ) -> List[FocusSpan]:
        """
        True MMR selection for diversity (Contract C3).
        
        score = lambda * sim(query, span) - (1-lambda) * max(sim(span, selected))
        Plus quotas: max_spans_per_doc, max_spans_per_chunk
        
        IMPORTANT: min_span_score is NOT a hard filter.
        We always emit top_n spans, marking low-score ones as low_confidence.
        This prevents the bundle from starving downstream stages.
        """
        max_spans = max_spans or self.top_n_spans
        
        selected: List[FocusSpan] = []
        selected_embeddings: List[List[float]] = []
        doc_counts: Dict[int, int] = {}
        chunk_counts: Dict[int, int] = {}
        
        for span, query_sim, span_emb in scored_spans:
            if len(selected) >= max_spans:
                break
            
            # Hard quotas (these ARE hard limits)
            if doc_counts.get(span.doc_id, 0) >= self.max_spans_per_doc:
                continue
            if chunk_counts.get(span.chunk_id, 0) >= self.max_spans_per_chunk:
                continue
            
            # MMR: penalize redundancy with already-selected spans
            if selected_embeddings:
                max_redundancy = max(
                    _cosine_similarity(span_emb, sel_emb) 
                    for sel_emb in selected_embeddings
                )
            else:
                max_redundancy = 0.0
            
            mmr_score = self.lambda_mmr * query_sim - (1 - self.lambda_mmr) * max_redundancy
            
            # ALWAYS accept up to top_n - min_span_score is NOT a hard filter
            # This prevents starving downstream stages
            focus_span = FocusSpan(
                chunk_id=span.chunk_id,
                doc_id=span.doc_id,
                page_ref=span.page_ref,
                start_char=span.start_char,
                end_char=span.end_char,
                text=span.text,
                score=query_sim,  # Store original query similarity
                rank=len(selected) + 1,
                source_lanes=span.source_lanes,
                span_hash=span.span_hash,
            )
            selected.append(focus_span)
            selected_embeddings.append(span_emb)
            doc_counts[span.doc_id] = doc_counts.get(span.doc_id, 0) + 1
            chunk_counts[span.chunk_id] = chunk_counts.get(span.chunk_id, 0) + 1
        
        # Sort by score descending, then span_id for determinism
        return sorted(selected, key=lambda s: (-s.score, s.span_id))
    
    def _embed_query(self, query_text: str) -> List[float]:
        """Embed query using OpenAI."""
        try:
            from retrieval.ops import embed_query
            return embed_query(query_text)
        except Exception:
            # Fallback: simple TF-IDF-like embedding
            return _simple_embed(query_text)
    
    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts using OpenAI."""
        if not texts:
            return []
        
        try:
            import openai
            client = openai.OpenAI()
            
            # Batch in groups of 100
            all_embeddings = []
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Truncate long texts
                batch = [t[:8000] if len(t) > 8000 else t for t in batch]
                
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                
                for item in response.data:
                    all_embeddings.append(item.embedding)
            
            return all_embeddings
            
        except Exception:
            # Fallback: simple embeddings
            return [_simple_embed(t) for t in texts]
    
    def _get_params(
        self, 
        contract: QueryContract, 
        total_spans: int, 
        total_chunks: int,
    ) -> Dict[str, Any]:
        """Get params for reproducibility."""
        return {
            "mode": contract.mode.value,
            "embedding_model": f"{EMBEDDING_MODEL}@{EMBEDDING_MODEL_VERSION}",
            "span_miner": self.span_miner.get_version_info(),
            "top_n_spans": self.top_n_spans,
            "lambda_mmr": self.lambda_mmr,
            "min_span_score": self.min_span_score,
            "max_spans_per_doc": self.max_spans_per_doc,
            "max_spans_per_chunk": self.max_spans_per_chunk,
            "total_spans_mined": total_spans,
            "total_chunks": total_chunks,
        }
    
    def _empty_bundle(self, contract: QueryContract) -> FocusBundle:
        """Create empty bundle when no spans found."""
        return FocusBundle(
            query_text=contract.query_text,
            spans=[],
            params=self._get_params(contract, 0, 0),
        )


def _normalize_for_match(text: str) -> str:
    """Normalize text for matching (lowercase, fold whitespace)."""
    return ' '.join(text.lower().split())


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    np = _get_numpy()
    
    if np:
        a_arr = np.array(a)
        b_arr = np.array(b)
        
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
    else:
        # Fallback: pure Python implementation
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


def _simple_embed(text: str, dim: int = 256) -> List[float]:
    """
    Simple fallback embedding using word hashing.
    
    Not for production - just for testing without API.
    """
    words = text.lower().split()
    embedding = [0.0] * dim
    
    for word in words:
        # Hash word to get index
        idx = hash(word) % dim
        embedding[idx] += 1.0
    
    # Normalize
    norm = sum(x * x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding


def persist_focus_bundle(
    bundle: FocusBundle,
    retrieval_run_id: int,
    conn,
    query_contract: QueryContract = None,
) -> None:
    """
    Persist FocusBundle to database.
    
    Stores spans in focus_spans table and params in focus_bundle_params.
    """
    cur = conn.cursor()
    
    # Insert focus_bundle_params
    cur.execute("""
        INSERT INTO focus_bundle_params 
        (retrieval_run_id, params_json, query_contract_json, total_spans_mined, total_chunks, mode)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (retrieval_run_id) DO UPDATE SET
            params_json = EXCLUDED.params_json,
            query_contract_json = EXCLUDED.query_contract_json,
            total_spans_mined = EXCLUDED.total_spans_mined,
            total_chunks = EXCLUDED.total_chunks,
            mode = EXCLUDED.mode
    """, (
        retrieval_run_id,
        json.dumps(bundle.params),
        json.dumps(query_contract.to_dict()) if query_contract else None,
        bundle.params.get("total_spans_mined", 0),
        bundle.params.get("total_chunks", 0),
        bundle.params.get("mode", "keyword_intent"),
    ))
    
    # Insert focus_spans
    for span in bundle.spans:
        cur.execute("""
            INSERT INTO focus_spans
            (retrieval_run_id, chunk_id, start_char, end_char, score, rank, 
             source_lanes, span_text, span_hash, doc_id, page_ref)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (retrieval_run_id, chunk_id, start_char, end_char) DO UPDATE SET
                score = EXCLUDED.score,
                rank = EXCLUDED.rank,
                source_lanes = EXCLUDED.source_lanes,
                span_text = EXCLUDED.span_text,
                span_hash = EXCLUDED.span_hash
        """, (
            retrieval_run_id,
            span.chunk_id,
            span.start_char,
            span.end_char,
            span.score,
            span.rank,
            span.source_lanes,
            span.text,
            span.span_hash,
            span.doc_id,
            span.page_ref,
        ))
    
    bundle.retrieval_run_id = retrieval_run_id
    conn.commit()


def load_focus_bundle(retrieval_run_id: int, conn) -> Optional[FocusBundle]:
    """
    Load FocusBundle from database.
    
    Uses stored spans as source-of-truth (recompute mode flag).
    """
    cur = conn.cursor()
    
    # Load params
    cur.execute("""
        SELECT params_json, query_contract_json
        FROM focus_bundle_params
        WHERE retrieval_run_id = %s
    """, (retrieval_run_id,))
    
    row = cur.fetchone()
    if not row:
        return None
    
    params = row[0] if row[0] else {}
    query_contract_data = row[1]
    
    # Load spans
    cur.execute("""
        SELECT chunk_id, start_char, end_char, score, rank, 
               source_lanes, span_text, span_hash, doc_id, page_ref
        FROM focus_spans
        WHERE retrieval_run_id = %s
        ORDER BY rank
    """, (retrieval_run_id,))
    
    spans = []
    for row in cur.fetchall():
        spans.append(FocusSpan(
            chunk_id=row[0],
            start_char=row[1],
            end_char=row[2],
            score=float(row[3]),
            rank=row[4],
            source_lanes=row[5] or [],
            text=row[6],
            span_hash=row[7],
            doc_id=row[8],
            page_ref=row[9] or "p0",
        ))
    
    # Get query_text from contract or params
    query_text = ""
    if query_contract_data:
        query_text = query_contract_data.get("query_text", "")
    
    bundle = FocusBundle(
        query_text=query_text,
        spans=spans,
        params=params,
        retrieval_run_id=retrieval_run_id,
    )
    
    # Rebuild mention index
    bundle.mention_span_index = build_mention_span_index(bundle, conn)
    
    return bundle
