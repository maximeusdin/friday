"""
Observation Bundle - Computed signals from probe retrieval.

Pure code computation of retrieval quality signals.
No LLM involvement - this is deterministic analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import statistics
import time
import sys

if TYPE_CHECKING:
    from retrieval.query_analysis import QueryAnalysis
    from retrieval.focus_bundle import FocusBundle


@dataclass
class ObservationBundle:
    """
    Signals computed from probe retrieval.
    
    Used by LLM to decide what actions to take (expand, retry, render, etc.)
    All fields are computed by code - no LLM involvement.
    """
    
    # === Query analysis (from Phase 0) ===
    scope_filters: Dict[str, Any] = field(default_factory=dict)
    core_concepts: List[str] = field(default_factory=list)
    anchor_terms: List[str] = field(default_factory=list)
    do_not_anchor: List[str] = field(default_factory=list)
    
    # === Probe retrieval signals ===
    total_chunks_retrieved: int = 0
    anchor_hit_count: int = 0  # Critical signal - spans containing anchor terms
    anchor_coverage: Dict[str, float] = field(default_factory=dict)  # term -> % of chunks
    
    # === Top span analysis ===
    top_span_snippets: List[str] = field(default_factory=list)  # First 5 spans, 100 chars
    span_score_distribution: Dict[str, float] = field(default_factory=dict)  # min, max, mean, median
    unique_docs: int = 0
    unique_chunks: int = 0
    
    # === Candidate analysis ===
    resolved_entity_count: int = 0
    unresolved_token_count: int = 0
    top_candidates: List[str] = field(default_factory=list)  # First 5 candidate names
    
    # === Red flags (computed heuristics) ===
    red_flags: List[str] = field(default_factory=list)
    
    # === Metadata ===
    probe_retrieval_ms: float = 0.0
    probe_bundle_span_count: int = 0
    
    def has_anchor_hits(self) -> bool:
        """Check if any anchor terms were found."""
        return self.anchor_hit_count > 0
    
    def is_low_diversity(self) -> bool:
        """Check if results are concentrated in few documents."""
        return self.unique_docs < 3
    
    def is_retrieval_drifted(self) -> bool:
        """Check if retrieval scores suggest semantic drift."""
        if not self.span_score_distribution:
            return True
        max_score = self.span_score_distribution.get("max", 0.0)
        mean_score = self.span_score_distribution.get("mean", 0.0)
        # Flat, low distribution suggests drift
        return max_score < 0.4 and mean_score < 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for LLM prompt."""
        return {
            "scope_filters": self.scope_filters,
            "core_concepts": self.core_concepts,
            "anchor_terms": self.anchor_terms,
            "do_not_anchor": self.do_not_anchor,
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "anchor_hit_count": self.anchor_hit_count,
            "anchor_coverage": self.anchor_coverage,
            "top_span_snippets": self.top_span_snippets,
            "span_score_distribution": self.span_score_distribution,
            "unique_docs": self.unique_docs,
            "unique_chunks": self.unique_chunks,
            "resolved_entity_count": self.resolved_entity_count,
            "unresolved_token_count": self.unresolved_token_count,
            "top_candidates": self.top_candidates,
            "red_flags": self.red_flags,
        }
    
    def to_prompt_string(self) -> str:
        """Format for LLM prompt (human-readable)."""
        lines = [
            "=== Probe Retrieval Observations ===",
            f"Query concepts: {', '.join(self.core_concepts)}",
            f"Anchor terms: {', '.join(self.anchor_terms)}",
            f"Scope: {self.scope_filters or 'all collections'}",
            "",
            "--- Retrieval Signals ---",
            f"Chunks retrieved: {self.total_chunks_retrieved}",
            f"Anchor hits: {self.anchor_hit_count} spans contain anchor terms",
        ]
        
        if self.anchor_coverage:
            lines.append("Anchor coverage:")
            for term, cov in self.anchor_coverage.items():
                lines.append(f"  '{term}': {cov:.1%} of chunks")
        
        if self.span_score_distribution:
            lines.append(f"Score distribution: min={self.span_score_distribution.get('min', 0):.3f}, "
                        f"max={self.span_score_distribution.get('max', 0):.3f}, "
                        f"mean={self.span_score_distribution.get('mean', 0):.3f}")
        
        lines.append(f"Unique docs: {self.unique_docs}, unique chunks: {self.unique_chunks}")
        
        if self.top_span_snippets:
            lines.append("")
            lines.append("--- Top Span Snippets ---")
            for i, snippet in enumerate(self.top_span_snippets[:5], 1):
                lines.append(f"{i}. {snippet[:100]}...")
        
        if self.top_candidates:
            lines.append("")
            lines.append(f"--- Candidates ({self.resolved_entity_count} entities, {self.unresolved_token_count} tokens) ---")
            for cand in self.top_candidates[:5]:
                lines.append(f"  - {cand}")
        
        if self.red_flags:
            lines.append("")
            lines.append("--- RED FLAGS ---")
            for flag in self.red_flags:
                lines.append(f"  [!] {flag}")
        
        return "\n".join(lines)


# Common stopwords for red flag detection
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
}


def detect_red_flags(
    focus_bundle: "FocusBundle",
    anchor_terms: List[str],
    anchor_hit_count: int,
    candidates: List[str],
    unique_docs: int,
    span_scores: List[float],
) -> List[str]:
    """
    Detect quality issues with the probe results.
    
    These are heuristics, not LLM decisions.
    """
    flags = []
    
    # No anchor hits is critical
    if anchor_terms and anchor_hit_count == 0:
        flags.append(f"No anchor hits: none of {anchor_terms} found in any span")
    
    # Low diversity
    if unique_docs < 3:
        flags.append(f"Low diversity: only {unique_docs} unique documents")
    
    # Retrieval drift (flat, low scores)
    if span_scores:
        max_score = max(span_scores)
        mean_score = statistics.mean(span_scores)
        if max_score < 0.4 and mean_score < 0.3:
            flags.append(f"Retrieval drift: scores flat and low (max={max_score:.3f}, mean={mean_score:.3f})")
    
    # Stopword-heavy candidates
    if candidates:
        stopword_candidates = []
        for cand in candidates[:10]:
            tokens = cand.lower().split()
            if len(tokens) <= 2:
                stopword_ratio = sum(1 for t in tokens if t in STOPWORDS) / max(len(tokens), 1)
                if stopword_ratio > 0.5:
                    stopword_candidates.append(cand)
        
        if len(stopword_candidates) >= 3:
            flags.append(f"Candidate quality: top candidates include stopwords: {stopword_candidates[:3]}")
    
    # Empty or tiny bundle
    if focus_bundle and len(focus_bundle.spans) < 5:
        flags.append(f"Sparse bundle: only {len(focus_bundle.spans)} spans selected")
    
    return flags


def compute_anchor_coverage(
    chunks: List[Any],
    anchor_terms: List[str],
) -> Dict[str, float]:
    """
    Compute what % of chunks contain each anchor term.
    
    Returns: anchor -> coverage ratio (0.0 to 1.0)
    """
    if not chunks or not anchor_terms:
        return {}
    
    coverage = {term: 0 for term in anchor_terms}
    
    for chunk in chunks:
        # Handle both dicts and objects
        if isinstance(chunk, dict):
            text = chunk.get('text', '').lower()
        else:
            text = getattr(chunk, 'text', '') or ''
            text = text.lower()
        
        for term in anchor_terms:
            if term.lower() in text:
                coverage[term] += 1
    
    total = len(chunks)
    return {term: count / total for term, count in coverage.items()}


def count_anchor_hits(
    focus_bundle: "FocusBundle",
    anchor_terms: List[str],
) -> int:
    """Count how many spans contain at least one anchor term."""
    if not focus_bundle or not anchor_terms:
        return 0
    
    count = 0
    for span in focus_bundle.spans:
        text_lower = span.text.lower()
        if any(term.lower() in text_lower for term in anchor_terms):
            count += 1
    
    return count


def compute_span_score_distribution(
    focus_bundle: "FocusBundle",
) -> Dict[str, float]:
    """Compute score distribution stats for spans."""
    if not focus_bundle or not focus_bundle.spans:
        return {}
    
    scores = [s.score for s in focus_bundle.spans]
    
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
    }


def build_observation_bundle(
    query_analysis: "QueryAnalysis",
    chunks: List[Dict[str, Any]],
    focus_bundle: "FocusBundle",
    candidates: List[Any],
    probe_time_ms: float = 0.0,
) -> ObservationBundle:
    """
    Build ObservationBundle from probe retrieval results.
    
    This is pure code - no LLM involvement.
    """
    # Count resolved vs unresolved candidates
    resolved_count = sum(1 for c in candidates if getattr(c, 'entity_id', None))
    unresolved_count = len(candidates) - resolved_count
    
    # Get candidate names
    top_candidates = [
        getattr(c, 'display_name', str(c)) 
        for c in candidates[:10]
    ]
    
    # Compute anchor coverage
    anchor_coverage = compute_anchor_coverage(chunks, query_analysis.anchor_terms)
    
    # Count anchor hits in spans
    anchor_hit_count = count_anchor_hits(focus_bundle, query_analysis.anchor_terms)
    
    # Get span scores
    span_scores = [s.score for s in focus_bundle.spans] if focus_bundle else []
    
    # Get unique docs/chunks
    unique_docs = len({s.doc_id for s in focus_bundle.spans}) if focus_bundle else 0
    unique_chunks = len({s.chunk_id for s in focus_bundle.spans}) if focus_bundle else 0
    
    # Get top span snippets
    top_snippets = []
    if focus_bundle:
        for span in focus_bundle.spans[:5]:
            snippet = span.text[:100].strip()
            snippet = ' '.join(snippet.split())  # Normalize whitespace
            top_snippets.append(snippet)
    
    # Detect red flags
    red_flags = detect_red_flags(
        focus_bundle=focus_bundle,
        anchor_terms=query_analysis.anchor_terms,
        anchor_hit_count=anchor_hit_count,
        candidates=top_candidates,
        unique_docs=unique_docs,
        span_scores=span_scores,
    )
    
    return ObservationBundle(
        # From query analysis
        scope_filters=query_analysis.scope_filters,
        core_concepts=query_analysis.core_concepts,
        anchor_terms=query_analysis.anchor_terms,
        do_not_anchor=query_analysis.do_not_anchor,
        
        # Retrieval signals
        total_chunks_retrieved=len(chunks),
        anchor_hit_count=anchor_hit_count,
        anchor_coverage=anchor_coverage,
        
        # Span analysis
        top_span_snippets=top_snippets,
        span_score_distribution=compute_span_score_distribution(focus_bundle),
        unique_docs=unique_docs,
        unique_chunks=unique_chunks,
        
        # Candidate analysis
        resolved_entity_count=resolved_count,
        unresolved_token_count=unresolved_count,
        top_candidates=top_candidates,
        
        # Quality signals
        red_flags=red_flags,
        
        # Metadata
        probe_retrieval_ms=probe_time_ms,
        probe_bundle_span_count=len(focus_bundle.spans) if focus_bundle else 0,
    )


# =============================================================================
# Probe Retrieval
# =============================================================================

def run_probe(
    query_analysis: "QueryAnalysis",
    conn,
    probe_k: int = 50,
    probe_bundle_top_n: int = 20,
) -> ObservationBundle:
    """
    Run a cheap probe retrieval to gather signals for LLM decision-making.
    
    This is a small, fast retrieval (k=50) followed by minimal FocusBundle
    construction (top_n=20) to detect quality issues before the main run.
    
    Args:
        query_analysis: Structured query decomposition from Phase 0
        conn: Database connection
        probe_k: Number of chunks to retrieve (default 50, cheap)
        probe_bundle_top_n: Number of spans in probe FocusBundle (default 20)
    
    Returns:
        ObservationBundle with computed signals and red flags
    """
    from retrieval.ops import hybrid_rrf, SearchFilters
    from retrieval.focus_bundle import FocusBundleBuilder
    from retrieval.query_intent import QueryContract, FocusBundleMode
    from retrieval.candidate_proposer import propose_all_candidates
    
    print(f"\n  [Probe] Starting probe retrieval...", file=sys.stderr)
    start_time = time.time()
    
    # Build retrieval query from core concepts
    retrieval_query = query_analysis.get_retrieval_query()
    print(f"    Query: {retrieval_query}", file=sys.stderr)
    print(f"    Anchors: {query_analysis.anchor_terms}", file=sys.stderr)
    
    # Build filters from scope (only if collections explicitly specified)
    filters = SearchFilters()
    if query_analysis.scope_filters:
        collections = query_analysis.scope_filters.get("collections")
        # Only filter if collections is a non-empty list (not None, not [])
        if collections and len(collections) > 0:
            filters = SearchFilters(collection_slugs=collections)
            print(f"    Scope: {collections}", file=sys.stderr)
        else:
            print(f"    Scope: all collections", file=sys.stderr)
    else:
        print(f"    Scope: all collections", file=sys.stderr)
    
    # Run cheap probe retrieval
    try:
        chunks = hybrid_rrf(
            conn=conn,
            query=retrieval_query,
            filters=filters,
            k=probe_k,
            expand_concordance=True,
            fuzzy_lex_enabled=True,
        )
        print(f"    Retrieved {len(chunks)} chunks", file=sys.stderr)
    except Exception as e:
        print(f"    Retrieval error: {e}", file=sys.stderr)
        chunks = []
    
    # Load chunk texts if needed
    if chunks:
        chunks = _load_chunk_texts(conn, chunks)
    
    # Build minimal FocusBundle for signal computation
    focus_bundle = None
    candidates = []
    
    if chunks:
        try:
            # Build QueryContract for FocusBundleBuilder
            contract = QueryContract(
                query_text=query_analysis.query_text,
                mode=FocusBundleMode.KEYWORD_INTENT,
            )
            
            # Build probe FocusBundle (small)
            builder = FocusBundleBuilder(
                top_n_spans=probe_bundle_top_n,
                min_span_score=0.0,  # Don't filter - we want to see everything
            )
            focus_bundle = builder.build(contract, chunks, conn)
            print(f"    Built probe FocusBundle with {len(focus_bundle.spans)} spans", file=sys.stderr)
            
            # Propose candidates (for signal computation)
            candidates = propose_all_candidates(focus_bundle, conn)
            print(f"    Proposed {len(candidates)} candidates", file=sys.stderr)
            
        except Exception as e:
            print(f"    FocusBundle/candidate error: {e}", file=sys.stderr)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Build observation bundle
    observations = build_observation_bundle(
        query_analysis=query_analysis,
        chunks=chunks,
        focus_bundle=focus_bundle,
        candidates=candidates,
        probe_time_ms=elapsed_ms,
    )
    
    # Log red flags
    if observations.red_flags:
        print(f"    RED FLAGS:", file=sys.stderr)
        for flag in observations.red_flags:
            print(f"      [!] {flag}", file=sys.stderr)
    else:
        print(f"    No red flags detected", file=sys.stderr)
    
    print(f"    Probe completed in {elapsed_ms:.0f}ms", file=sys.stderr)
    
    return observations


def _load_chunk_texts(conn, chunks: List[Any]) -> List[Dict[str, Any]]:
    """
    Load chunk texts if not already present.
    
    Retrieval may return ChunkHit objects or dicts; we need dicts with text for FocusBundle.
    """
    if not chunks:
        return []
    
    # Convert ChunkHit objects to dicts if needed
    from retrieval.ops import ChunkHit
    
    converted = []
    for c in chunks:
        if isinstance(c, ChunkHit):
            converted.append({
                'chunk_id': c.chunk_id,
                'id': c.chunk_id,
                'score': c.score,
                'doc_id': getattr(c, 'doc_id', None),
                'text': getattr(c, 'text', None),
            })
        elif isinstance(c, dict):
            converted.append(c)
        else:
            # Try to extract chunk_id from unknown type
            chunk_id = getattr(c, 'chunk_id', None) or getattr(c, 'id', None)
            if chunk_id:
                converted.append({'chunk_id': chunk_id, 'id': chunk_id})
    
    chunks = converted
    
    # Check if text is already present
    if chunks and chunks[0].get('text'):
        return chunks
    
    # Get chunk IDs
    chunk_ids = [c.get('chunk_id') or c.get('id') for c in chunks]
    if not chunk_ids:
        return chunks
    
    # Load texts from database
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id, c.text, cm.document_id, cm.first_page_id
            FROM chunks c
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.id = ANY(%s)
        """, (chunk_ids,))
        
        text_map = {}
        for row in cur.fetchall():
            chunk_id, text, document_id, first_page_id = row
            text_map[chunk_id] = {
                'text': text,
                'doc_id': document_id,
                'page_ref': f"p{first_page_id}" if first_page_id else "p0",
            }
    
    # Merge text into chunks
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id') or chunk.get('id')
        if chunk_id in text_map:
            chunk['text'] = text_map[chunk_id]['text']
            chunk['doc_id'] = text_map[chunk_id]['doc_id']
            chunk['page_ref'] = text_map[chunk_id]['page_ref']
    
    return chunks
