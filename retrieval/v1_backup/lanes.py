"""
Multi-Lane Retrieval with Iterative Execution for Agentic Workflow.

This module implements:
- Lane execution (Entity/Codename, Lexical Must-Hit, Hybrid, Ephemeral Expansion, Co-mention)
- Iterative orchestration with Jaccard+novelty stability check
- Chunk deduplication with provenance tracking
- Candidate extraction from merged chunks

Architecture:
    Round 1: seed lanes (entity, must-hit, hybrid)
    → Extract candidate tokens/entities from merged chunks
    Round 2+: targeted lanes per candidate (evidence bucketing, codename resolution)
    → Stop when stable (Jaccard + novelty)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from retrieval.plan import (
    AgenticPlan,
    LaneSpec,
    CommentionLaneSpec,
    Budgets,
    StopConditions,
)
from retrieval.evidence_bundle import (
    EvidenceBundle,
    EvidenceRef,
    EntityCandidate,
    RetrievalLaneRun,
    ChunkWithProvenance,
)
from retrieval.intent import IntentFamily, compute_coverage
import re


# =============================================================================
# TSQuery Helper
# =============================================================================

def term_to_tsquery(term: str, operator: str = "&") -> str:
    """
    Convert a term (possibly multi-word) to valid tsquery format.
    
    Examples:
        "Proximity Fuse" -> "'proximity':* & 'fuse':*"
        "VT fuse" -> "'vt':* & 'fuse':*"
        "silvermaster" -> "'silvermaster':*"
    
    Args:
        term: The search term (can be multi-word)
        operator: "&" for AND, "|" for OR between words
    
    Returns:
        Valid tsquery string
    """
    # Split on whitespace and filter empty
    words = [w.strip().lower() for w in term.split() if w.strip()]
    if not words:
        return ""
    
    # Each word becomes 'word':* and joined by operator
    parts = [f"'{w}':*" for w in words]
    return f" {operator} ".join(parts)


def terms_to_tsquery(terms: List[str], inter_term_op: str = "|", intra_term_op: str = "&") -> str:
    """
    Convert multiple terms to valid tsquery format.
    
    Each term's words are joined by intra_term_op (default AND).
    Terms are joined by inter_term_op (default OR).
    
    Example with defaults:
        ["Proximity Fuse", "VT fuse"] -> "('proximity':* & 'fuse':*) | ('vt':* & 'fuse':*)"
    """
    term_queries = []
    for term in terms:
        q = term_to_tsquery(term, intra_term_op)
        if q:
            # Wrap in parens if multiple words
            if " " in q:
                q = f"({q})"
            term_queries.append(q)
    
    if not term_queries:
        return ""
    
    return f" {inter_term_op} ".join(term_queries)


# =============================================================================
# Lane Execution Results
# =============================================================================

@dataclass
class LaneResult:
    """Result from executing a single lane."""
    lane_id: str
    chunk_ids: List[int]
    scores: Dict[int, float]              # chunk_id -> score
    doc_ids: Dict[int, int]               # chunk_id -> doc_id
    hit_count: int
    doc_count: int
    unique_pages: int
    execution_ms: float
    terms_matched: List[str]


# =============================================================================
# Lane Executors
# =============================================================================

def execute_entity_codename_lane(
    spec: LaneSpec,
    conn,
) -> LaneResult:
    """
    Execute entity/codename lane.
    
    Searches by entity_id mentions and expands codename aliases.
    Must-hit lexical for ALLCAPS codenames (embeddings fail on these).
    """
    start_time = time.time()
    
    chunk_ids = []
    scores = {}
    doc_ids = {}
    terms_matched = []
    
    if not spec.entity_ids:
        return LaneResult(
            lane_id=spec.lane_id,
            chunk_ids=[],
            scores={},
            doc_ids={},
            hit_count=0,
            doc_count=0,
            unique_pages=0,
            execution_ms=(time.time() - start_time) * 1000,
            terms_matched=[],
        )
    
    # Build filters
    filters = spec.filters.copy()
    collection_scope = filters.get("collection_scope", [])
    
    with conn.cursor() as cur:
        # Get chunks with entity mentions
        query = """
            SELECT DISTINCT em.chunk_id, em.entity_id, cm.document_id,
                   e.canonical_name
            FROM entity_mentions em
            JOIN chunks c ON em.chunk_id = c.id
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            JOIN entities e ON em.entity_id = e.id
            WHERE em.entity_id = ANY(%(entity_ids)s)
        """
        params = {"entity_ids": spec.entity_ids}
        
        # Add collection filter if specified
        if collection_scope:
            query += """
                AND cm.collection_slug = ANY(%(collections)s)
            """
            params["collections"] = collection_scope
        
        query += " LIMIT %(k)s"
        params["k"] = spec.k
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        for chunk_id, entity_id, document_id, canonical_name in rows:
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
                scores[chunk_id] = 1.0  # Entity matches get full score
                doc_ids[chunk_id] = document_id or 0
                if canonical_name not in terms_matched:
                    terms_matched.append(canonical_name)
    
    # Calculate stats
    unique_docs = set(d for d in doc_ids.values() if d)
    
    return LaneResult(
        lane_id=spec.lane_id,
        chunk_ids=chunk_ids[:spec.k],
        scores=scores,
        doc_ids=doc_ids,
        hit_count=len(chunk_ids),
        doc_count=len(unique_docs),
        unique_pages=len(unique_docs),  # Approximate
        execution_ms=(time.time() - start_time) * 1000,
        terms_matched=terms_matched,
    )


def execute_lexical_must_hit_lane(
    spec: LaneSpec,
    conn,
) -> LaneResult:
    """
    Execute lexical must-hit lane.
    
    All query terms must appear in the chunk. Uses deterministic normalization.
    """
    import sys
    start_time = time.time()
    
    if not spec.must_hit_terms and not spec.query_terms:
        return LaneResult(
            lane_id=spec.lane_id,
            chunk_ids=[],
            scores={},
            doc_ids={},
            hit_count=0,
            doc_count=0,
            unique_pages=0,
            execution_ms=(time.time() - start_time) * 1000,
            terms_matched=[],
        )
    
    terms = spec.must_hit_terms or spec.query_terms
    filters = spec.filters.copy()
    collection_scope = filters.get("collection_scope", [])
    
    print(f"  [Lexical Lane] Terms: {terms}, Collections: {collection_scope}", file=sys.stderr)
    
    chunk_ids = []
    scores = {}
    doc_ids = {}
    
    with conn.cursor() as cur:
        # Build tsquery - use OR matching for better recall
        # Handle multi-word terms by splitting them into individual words
        all_words = []
        for term in terms:
            # Split multi-word terms into individual words
            words = term.split()
            all_words.extend(words)
        
        # Dedupe while preserving order
        seen = set()
        unique_words = []
        for w in all_words:
            w_lower = w.lower()
            if w_lower not in seen:
                seen.add(w_lower)
                unique_words.append(w_lower)
        
        # Build tsquery with OR matching for better recall
        # (AND was too restrictive - "silvermaster" AND "network" missed silvermaster collection)
        tsquery_parts = [f"'{word}':*" for word in unique_words]
        tsquery = " | ".join(tsquery_parts)  # OR instead of AND
        
        print(f"  [Lexical Lane] tsquery: {tsquery}", file=sys.stderr)
        
        # Debug: Check which collections have matches for the first term
        first_term = unique_words[0] if unique_words else ""
        if first_term:
            cur.execute("""
                SELECT cm.collection_slug, COUNT(*) as cnt
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE to_tsvector('simple', COALESCE(c.clean_text, c.text)) @@ to_tsquery('simple', %s)
                GROUP BY cm.collection_slug
                ORDER BY cnt DESC
                LIMIT 10
            """, (f"'{first_term}':*",))
            debug_rows = cur.fetchall()
            print(f"  [Lexical Lane] Collections with '{first_term}': {debug_rows}", file=sys.stderr)
        
        # Use dynamic to_tsvector on text content (same as ops.py)
        query = """
            SELECT c.id, cm.document_id, cm.collection_slug,
                   ts_rank(to_tsvector('simple', COALESCE(c.clean_text, c.text)), 
                           to_tsquery('simple', %(tsquery)s)) as score
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE to_tsvector('simple', COALESCE(c.clean_text, c.text)) @@ to_tsquery('simple', %(tsquery)s)
        """
        params = {"tsquery": tsquery}
        
        if collection_scope:
            query += """
                AND cm.collection_slug = ANY(%(collections)s)
            """
            params["collections"] = collection_scope
        
        query += " ORDER BY score DESC LIMIT %(k)s"
        params["k"] = spec.k
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        print(f"  [Lexical Lane] Got {len(rows)} results", file=sys.stderr)
        
        # Show first few collections for debug
        collections_found = set()
        for chunk_id, document_id, collection_slug, score in rows:
            chunk_ids.append(chunk_id)
            scores[chunk_id] = float(score) if score else 0.0
            doc_ids[chunk_id] = document_id or 0
            if collection_slug:
                collections_found.add(collection_slug)
        
        if collections_found:
            print(f"  [Lexical Lane] Collections: {list(collections_found)[:5]}", file=sys.stderr)
    
    unique_docs = set(d for d in doc_ids.values() if d)
    
    return LaneResult(
        lane_id=spec.lane_id,
        chunk_ids=chunk_ids,
        scores=scores,
        doc_ids=doc_ids,
        hit_count=len(chunk_ids),
        doc_count=len(unique_docs),
        unique_pages=len(unique_docs),
        execution_ms=(time.time() - start_time) * 1000,
        terms_matched=terms,
    )


def execute_hybrid_lane(
    spec: LaneSpec,
    conn,
) -> LaneResult:
    """
    Execute hybrid (vector + lexical) lane.
    
    Uses RRF to combine vector and lexical results.
    """
    import sys
    start_time = time.time()
    
    # Import here to avoid circular imports
    from retrieval.ops import hybrid_rrf, SearchFilters
    
    # Build query text
    query_text = " ".join(spec.query_terms) if spec.query_terms else ""
    
    if not query_text:
        return LaneResult(
            lane_id=spec.lane_id,
            chunk_ids=[],
            scores={},
            doc_ids={},
            hit_count=0,
            doc_count=0,
            unique_pages=0,
            execution_ms=(time.time() - start_time) * 1000,
            terms_matched=[],
        )
    
    filters = spec.filters.copy()
    collection_scope = filters.get("collection_scope", [])
    
    # Build SearchFilters dataclass for hybrid_rrf
    search_filters = SearchFilters(
        collection_slugs=collection_scope if collection_scope else None,
    )
    
    print(f"  [Hybrid Lane] Query: '{query_text}', Collections: {collection_scope}", file=sys.stderr)
    
    try:
        results = hybrid_rrf(
            conn=conn,
            query=query_text,
            filters=search_filters,
            k=spec.k,
        )
        print(f"  [Hybrid Lane] Got {len(results)} results", file=sys.stderr)
    except Exception as e:
        # Log the exception for debugging
        print(f"  [Hybrid Lane] Error: {e}", file=sys.stderr)
        return LaneResult(
            lane_id=spec.lane_id,
            chunk_ids=[],
            scores={},
            doc_ids={},
            hit_count=0,
            doc_count=0,
            unique_pages=0,
            execution_ms=(time.time() - start_time) * 1000,
            terms_matched=[],
        )
    
    chunk_ids = []
    scores = {}
    doc_ids = {}
    
    for result in results:
        chunk_id = result.chunk_id if hasattr(result, 'chunk_id') else result.get('chunk_id')
        score = result.score if hasattr(result, 'score') else result.get('score', 0.0)
        doc_id = result.document_id if hasattr(result, 'document_id') else result.get('document_id', 0)
        
        if chunk_id:
            chunk_ids.append(chunk_id)
            scores[chunk_id] = float(score) if score else 0.0
            doc_ids[chunk_id] = doc_id
    
    unique_docs = set(doc_ids.values())
    
    return LaneResult(
        lane_id=spec.lane_id,
        chunk_ids=chunk_ids,
        scores=scores,
        doc_ids=doc_ids,
        hit_count=len(chunk_ids),
        doc_count=len(unique_docs),
        unique_pages=len(unique_docs),
        execution_ms=(time.time() - start_time) * 1000,
        terms_matched=spec.query_terms,
    )


def execute_comention_lane(
    spec: CommentionLaneSpec,
    conn,
) -> LaneResult:
    """
    Execute co-mention expansion lane.
    
    SQL-driven, very fast - finds all entities co-mentioned with seed entities.
    Massively improves roster completeness.
    """
    start_time = time.time()
    
    if not spec.seed_entity_ids:
        return LaneResult(
            lane_id=spec.lane_id,
            chunk_ids=[],
            scores={},
            doc_ids={},
            hit_count=0,
            doc_count=0,
            unique_pages=0,
            execution_ms=(time.time() - start_time) * 1000,
            terms_matched=[],
        )
    
    chunk_ids = []
    scores = {}
    doc_ids = {}
    
    with conn.cursor() as cur:
        # Find all chunks where seed entities co-occur with other entities
        query = """
            SELECT DISTINCT em2.chunk_id, cm.document_id, e.canonical_name
            FROM entity_mentions em1
            JOIN entity_mentions em2 ON em1.chunk_id = em2.chunk_id
            JOIN entities e ON em2.entity_id = e.id
            JOIN chunks c ON em2.chunk_id = c.id
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE em1.entity_id = ANY(%(seed_entity_ids)s)
              AND e.entity_type = %(target_type)s
              AND em2.entity_id != em1.entity_id
            LIMIT %(k)s
        """
        params = {
            "seed_entity_ids": spec.seed_entity_ids,
            "target_type": spec.target_entity_type,
            "k": spec.k,
        }
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        terms_matched = []
        for chunk_id, document_id, canonical_name in rows:
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
                scores[chunk_id] = 0.9  # Co-mentions get high score
                doc_ids[chunk_id] = document_id or 0
                if canonical_name not in terms_matched:
                    terms_matched.append(canonical_name)
    
    unique_docs = set(d for d in doc_ids.values() if d)
    
    return LaneResult(
        lane_id=spec.lane_id,
        chunk_ids=chunk_ids,
        scores=scores,
        doc_ids=doc_ids,
        hit_count=len(chunk_ids),
        doc_count=len(unique_docs),
        unique_pages=len(unique_docs),
        execution_ms=(time.time() - start_time) * 1000,
        terms_matched=terms_matched,
    )


# =============================================================================
# Ephemeral Expansion
# =============================================================================

@dataclass
class HitTestResult:
    """Result from lexical hit test."""
    term: str
    hit_count: int
    doc_count: int
    sample_refs: List[EvidenceRef]
    sample_chunk_ids: List[int]


@dataclass
class ValidatedTerm:
    """A validated expansion term with stats."""
    term: str
    hit_count: int
    doc_count: int
    specificity_score: float              # higher = more specific (fewer corpus hits)
    sample_refs: List[EvidenceRef]


@dataclass
class ExpansionResult:
    """Result from ephemeral query expansion."""
    validated_terms: List[ValidatedTerm]  # ranked by specificity
    rejected_terms: List[str]
    rejection_reasons: Dict[str, str]


def lexical_hit_test(
    term: str,
    collections: List[str],
    conn,
    limit: int = 5,
) -> HitTestResult:
    """
    Fast corpus validation for a term.
    
    Checks if term has hits in allowed collections and returns stats.
    Cacheable (term + filters hash).
    
    Args:
        term: Term to test
        collections: Allowed collections (empty = all)
        conn: Database connection
        limit: Max sample refs to return
        
    Returns:
        HitTestResult with hit count, doc count, and samples
    """
    sample_refs = []
    sample_chunk_ids = []
    
    with conn.cursor() as cur:
        # Count hits
        # Use plainto_tsquery for plain text input (handles spaces, punctuation safely)
        query = """
            SELECT COUNT(DISTINCT c.id) as chunk_count,
                   COUNT(DISTINCT cm.document_id) as doc_count
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE c.text_tsv @@ plainto_tsquery('simple', %(term)s)
        """
        params = {"term": term}
        
        if collections:
            query += """
                AND cm.collection_slug = ANY(%(collections)s)
            """
            params["collections"] = collections
        
        cur.execute(query, params)
        row = cur.fetchone()
        hit_count = row[0] if row else 0
        doc_count = row[1] if row else 0
        
        # Get sample chunks if hits exist
        if hit_count > 0:
            sample_query = """
                SELECT c.id, cm.document_id
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE c.text_tsv @@ plainto_tsquery('simple', %(term)s)
            """
            if collections:
                sample_query += """
                    AND cm.collection_slug = ANY(%(collections)s)
                """
            sample_query += " LIMIT %(limit)s"
            params["limit"] = limit
            
            cur.execute(sample_query, params)
            for chunk_id, doc_id in cur.fetchall():
                sample_chunk_ids.append(chunk_id)
                sample_refs.append(EvidenceRef(
                    chunk_id=chunk_id,
                    doc_id=doc_id or 0,
                    page_ref="",
                    char_start=0,
                    char_end=0,
                    quote_span="",
                ))
    
    return HitTestResult(
        term=term,
        hit_count=hit_count,
        doc_count=doc_count,
        sample_refs=sample_refs,
        sample_chunk_ids=sample_chunk_ids,
    )


def deterministic_normalizations(concept: str) -> List[str]:
    """
    Generate deterministic normalization variants for a concept.
    
    Handles:
    - fuze/fuse variants
    - Punctuation variants (VT / V.T. / V-T)
    - Hyphen variants (variable-time / variable time)
    """
    variants = [concept]
    concept_lower = concept.lower()
    
    # Fuse/fuze variants
    if "fuse" in concept_lower:
        variants.append(concept_lower.replace("fuse", "fuze"))
    if "fuze" in concept_lower:
        variants.append(concept_lower.replace("fuze", "fuse"))
    
    # Punctuation variants
    # VT -> V.T., V-T, V T
    if len(concept) <= 4 and concept.isupper():
        # Add dotted version
        dotted = ".".join(concept) + "."
        variants.append(dotted)
        # Add hyphenated version
        if len(concept) >= 2:
            hyphenated = "-".join(concept)
            variants.append(hyphenated)
    
    # Remove periods
    if "." in concept:
        variants.append(concept.replace(".", ""))
    
    # Hyphen variants
    if "-" in concept:
        variants.append(concept.replace("-", " "))
    if " " in concept:
        variants.append(concept.replace(" ", "-"))
    
    # Dedupe and return
    return list(set(variants))


def passes_short_acronym_filter(
    term: str,
    anchor_concept: str,
    hit_result: HitTestResult,
    conn,
) -> bool:
    """
    Ambiguity filter for short acronyms (≤3 chars).
    
    For terms like "VT", require EITHER:
    - Co-occurrence with another validated anchor in same chunk
      (e.g., "fuse/fuze" appears with "VT")
    - Definition pattern match in nearby text
      (e.g., "variable time (VT)" or "VT fuse")
    
    Prevents "VT" from pulling in unrelated uses.
    No curated synonym lists needed - generic guardrail.
    """
    import re
    
    if not hit_result.sample_chunk_ids:
        return False
    
    # Check for co-occurrence with anchor concept
    anchor_variants = deterministic_normalizations(anchor_concept)
    
    with conn.cursor() as cur:
        # Check if any sample chunks contain both the term and anchor variants
        for chunk_id in hit_result.sample_chunk_ids[:3]:
            cur.execute(
                "SELECT COALESCE(clean_text, text) FROM chunks WHERE id = %s",
                (chunk_id,)
            )
            row = cur.fetchone()
            if not row or not row[0]:
                continue
            
            text = row[0].lower()
            
            # Check for co-occurrence with anchor
            for variant in anchor_variants:
                if variant.lower() in text:
                    return True
            
            # Check for definition patterns
            # e.g., "variable time (VT)" or "VT fuse" or "VT (variable time)"
            definition_patterns = [
                rf"\({re.escape(term)}\)",  # (VT)
                rf"{re.escape(term)}\s+(?:fuse|fuze)",  # VT fuse
                rf"(?:fuse|fuze)\s+{re.escape(term)}",  # fuse VT
                rf"variable\s+time",  # variable time (context)
            ]
            
            for pattern in definition_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
    
    return False


def expand_query_ephemeral(
    concept: str,
    collections: List[str],
    conn,
    budgets: Budgets,
) -> ExpansionResult:
    """
    Ephemeral query expansion with validation.
    
    1. Deterministic normalizations (always)
    2. LLM-proposed rewrites (bounded by budgets.max_expansion_terms)
    3. Corpus validation via lexical_hit_test()
    4. Ambiguity filter for short acronyms
    5. Cap validated terms by specificity (avoid generic terms dominating)
    
    Args:
        concept: The concept to expand
        collections: Allowed collections
        conn: Database connection
        budgets: Budgets with expansion limits
        
    Returns:
        ExpansionResult with validated terms and rejection info
    """
    candidates = []
    rejected = []
    rejection_reasons: Dict[str, str] = {}
    
    # Step 1: Deterministic normalizations
    candidates.extend(deterministic_normalizations(concept))
    
    # Step 2: LLM-proposed rewrites (bounded)
    # For now, we'll use a simple heuristic approach
    # In production, this would call an LLM
    llm_rewrites = get_simple_rewrites(concept, max_terms=budgets.max_expansion_terms)
    candidates.extend(llm_rewrites)
    
    # Dedupe candidates
    candidates = list(set(candidates))
    
    # Step 3: Corpus validation + stats collection
    validated_with_stats = []
    
    for term in candidates:
        hit_result = lexical_hit_test(term, collections, conn)
        
        if hit_result.hit_count == 0:
            rejected.append(term)
            rejection_reasons[term] = "no_corpus_hits"
            continue
        
        # Step 4: Ambiguity filter for short acronyms
        if len(term) <= 3:
            if not passes_short_acronym_filter(term, concept, hit_result, conn):
                rejected.append(term)
                rejection_reasons[term] = "short_acronym_ambiguous"
                continue
        
        # Compute specificity (inverse of hit prevalence)
        # Higher score = more specific = fewer hits = better
        specificity = 1.0 / (1.0 + hit_result.hit_count)
        
        validated_with_stats.append(ValidatedTerm(
            term=term,
            hit_count=hit_result.hit_count,
            doc_count=hit_result.doc_count,
            specificity_score=specificity,
            sample_refs=hit_result.sample_refs,
        ))
    
    # Step 5: Cap by specificity to avoid generic terms dominating
    # Sort by specificity (most specific first), then cap
    validated_with_stats.sort(key=lambda t: (-t.specificity_score, t.term))
    capped = validated_with_stats[:budgets.max_validated_expansion_terms]
    
    # Reject terms that got capped
    for term_obj in validated_with_stats[budgets.max_validated_expansion_terms:]:
        rejected.append(term_obj.term)
        rejection_reasons[term_obj.term] = "capped_low_specificity"
    
    return ExpansionResult(
        validated_terms=capped,
        rejected_terms=rejected,
        rejection_reasons=rejection_reasons,
    )


def get_simple_rewrites(concept: str, max_terms: int = 10) -> List[str]:
    """
    Simple heuristic rewrites (placeholder for LLM).
    
    In production, this would call an LLM to get rewrites.
    For now, uses simple heuristics.
    """
    rewrites = []
    
    # Add lowercase/uppercase variants
    rewrites.append(concept.lower())
    rewrites.append(concept.upper())
    rewrites.append(concept.title())
    
    # Common abbreviation expansions
    abbreviation_expansions = {
        "vt": ["variable time", "variable-time"],
        "v.t.": ["variable time", "variable-time"],
        "cpusa": ["communist party", "communist party usa"],
        "nkvd": ["people's commissariat"],
        "mgb": ["ministry of state security"],
        "gru": ["military intelligence"],
    }
    
    concept_lower = concept.lower()
    if concept_lower in abbreviation_expansions:
        rewrites.extend(abbreviation_expansions[concept_lower])
    
    return rewrites[:max_terms]


def execute_ephemeral_expansion_lane(
    spec: LaneSpec,
    conn,
) -> LaneResult:
    """
    Execute ephemeral expansion lane.
    
    Expands query concepts with validation, then searches.
    """
    start_time = time.time()
    
    if not spec.query_terms:
        return LaneResult(
            lane_id=spec.lane_id,
            chunk_ids=[],
            scores={},
            doc_ids={},
            hit_count=0,
            doc_count=0,
            unique_pages=0,
            execution_ms=(time.time() - start_time) * 1000,
            terms_matched=[],
        )
    
    filters = spec.filters.copy()
    collection_scope = filters.get("collection_scope", [])
    
    # Expand each concept
    all_validated_terms = []
    budgets = Budgets()  # Use defaults
    
    for concept in spec.query_terms:
        expansion = expand_query_ephemeral(concept, collection_scope, conn, budgets)
        all_validated_terms.extend(expansion.validated_terms)
    
    if not all_validated_terms:
        return LaneResult(
            lane_id=spec.lane_id,
            chunk_ids=[],
            scores={},
            doc_ids={},
            hit_count=0,
            doc_count=0,
            unique_pages=0,
            execution_ms=(time.time() - start_time) * 1000,
            terms_matched=[],
        )
    
    # Search using validated terms
    chunk_ids = []
    scores = {}
    doc_ids = {}
    terms_matched = []
    
    # Collect unique chunks from all validated terms
    seen_chunks = set()
    for vt in all_validated_terms:
        for ref in vt.sample_refs:
            if ref.chunk_id not in seen_chunks:
                chunk_ids.append(ref.chunk_id)
                scores[ref.chunk_id] = vt.specificity_score
                doc_ids[ref.chunk_id] = ref.doc_id
                seen_chunks.add(ref.chunk_id)
        terms_matched.append(vt.term)
    
    # Also do a broader search with the top validated terms
    top_terms = [vt.term for vt in all_validated_terms[:3]]
    if top_terms:
        with conn.cursor() as cur:
            # OR search with top terms - use helper for multi-word safety
            tsquery = terms_to_tsquery(top_terms, inter_term_op="|", intra_term_op="&")
            
            if not tsquery:
                return LaneResult(
                    lane_id=spec.lane_id,
                    chunk_ids=[],
                    scores={},
                    doc_ids={},
                    hit_count=0,
                    doc_count=0,
                    unique_pages=0,
                    terms_matched=[],
                    execution_ms=0,
                )
            
            query = """
                SELECT c.id, cm.document_id, ts_rank(c.text_tsv, to_tsquery('simple', %(tsquery)s)) as score
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE c.text_tsv @@ to_tsquery('simple', %(tsquery)s)
            """
            params = {"tsquery": tsquery}
            
            if collection_scope:
                query += """
                    AND cm.collection_slug = ANY(%(collections)s)
                """
                params["collections"] = collection_scope
            
            query += " ORDER BY score DESC LIMIT %(k)s"
            params["k"] = spec.k
            
            cur.execute(query, params)
            for chunk_id, document_id, score in cur.fetchall():
                if chunk_id not in seen_chunks:
                    chunk_ids.append(chunk_id)
                    scores[chunk_id] = float(score) if score else 0.0
                    doc_ids[chunk_id] = document_id or 0
                    seen_chunks.add(chunk_id)
    
    unique_docs = set(doc_ids.values())
    
    return LaneResult(
        lane_id=spec.lane_id,
        chunk_ids=chunk_ids[:spec.k],
        scores=scores,
        doc_ids=doc_ids,
        hit_count=len(chunk_ids),
        doc_count=len(unique_docs),
        unique_pages=len(unique_docs),
        execution_ms=(time.time() - start_time) * 1000,
        terms_matched=terms_matched,
    )


# Lane executor mapping
LANE_EXECUTORS: Dict[str, Callable] = {
    "entity_codename": execute_entity_codename_lane,
    "lexical_must_hit": execute_lexical_must_hit_lane,
    "hybrid": execute_hybrid_lane,
    "comention_expand": execute_comention_lane,
    "ephemeral_expansion": execute_ephemeral_expansion_lane,
}


def execute_lane(spec: LaneSpec, conn) -> LaneResult:
    """Execute a lane by its lane_id."""
    executor = LANE_EXECUTORS.get(spec.lane_id)
    
    if executor:
        if spec.lane_id == "comention_expand" and isinstance(spec, CommentionLaneSpec):
            return executor(spec, conn)
        return executor(spec, conn)
    
    # Default to hybrid for unknown lanes
    return execute_hybrid_lane(spec, conn)


# =============================================================================
# Chunk Merging and Provenance
# =============================================================================

def merge_lane_chunks(
    all_chunks: Dict[int, ChunkWithProvenance],
    lane_result: LaneResult,
    round_num: int,
) -> List[int]:
    """
    Merge chunks from a lane result into the global chunk dict.
    
    Deduplicates by chunk_id and tracks which lanes found each chunk.
    
    Args:
        all_chunks: Global chunk dict (mutated)
        lane_result: Result from lane execution
        round_num: Current round number
        
    Returns:
        List of new chunk_ids added in this merge
    """
    new_chunk_ids = []
    
    for chunk_id in lane_result.chunk_ids:
        score = lane_result.scores.get(chunk_id, 0.0)
        doc_id = lane_result.doc_ids.get(chunk_id, 0)
        
        if chunk_id in all_chunks:
            # Existing chunk - add lane to provenance
            all_chunks[chunk_id].add_lane(lane_result.lane_id, score, round_num)
        else:
            # New chunk
            all_chunks[chunk_id] = ChunkWithProvenance(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_lanes=[lane_result.lane_id],
                best_score=score,
                first_seen_round=round_num,
            )
            new_chunk_ids.append(chunk_id)
    
    return new_chunk_ids


def load_chunk_texts(
    all_chunks: Dict[int, ChunkWithProvenance],
    chunk_ids: List[int],
    conn,
) -> None:
    """
    Load text for chunks (lazy loading).
    
    Only loads for chunks that don't have text yet.
    """
    chunks_to_load = [
        cid for cid in chunk_ids
        if cid in all_chunks and all_chunks[cid].text is None
    ]
    
    if not chunks_to_load:
        return
    
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, COALESCE(clean_text, text) as text
            FROM chunks
            WHERE id = ANY(%(chunk_ids)s)
            """,
            {"chunk_ids": chunks_to_load}
        )
        
        for chunk_id, text in cur.fetchall():
            if chunk_id in all_chunks:
                all_chunks[chunk_id].text = text


# =============================================================================
# Candidate Extraction
# =============================================================================

def extract_candidates(
    chunks: List[ChunkWithProvenance],
    conn,
    round_num: int = 1,
) -> Dict[str, EntityCandidate]:
    """
    Extract EntityCandidates from chunks.
    
    Uses:
    1. entity_mentions table (resolved entities)
    2. Pattern extraction (unresolved tokens like codenames)
    
    Args:
        chunks: Chunks to extract from
        conn: Database connection
        round_num: Current round number
        
    Returns:
        Dict keyed by candidate.key for deduplication
    """
    candidates: Dict[str, EntityCandidate] = {}
    
    if not chunks:
        return candidates
    
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_map = {c.chunk_id: c for c in chunks}
    
    # Get entity mentions for these chunks
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT em.chunk_id, em.entity_id, em.surface, em.start_char, em.end_char,
                   e.canonical_name, cm.document_id
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.id
            JOIN chunks c ON em.chunk_id = c.id
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE em.chunk_id = ANY(%(chunk_ids)s)
            """,
            {"chunk_ids": chunk_ids}
        )
        
        for row in cur.fetchall():
            chunk_id, entity_id, surface, start_char, end_char, canonical_name, doc_id = row
            
            # Create evidence ref
            chunk_data = chunk_map.get(chunk_id)
            text = chunk_data.text if chunk_data else ""
            quote_span = text[start_char:end_char] if text and start_char and end_char else surface or ""
            
            evidence_ref = EvidenceRef(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_ref="",  # Would need to look up
                char_start=start_char or 0,
                char_end=end_char or 0,
                quote_span=quote_span[:500],  # Cap at 500 chars
            )
            
            # Create or merge candidate
            key = f"entity:{entity_id}"
            if key in candidates:
                candidates[key].merge(EntityCandidate.from_entity_mention(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    evidence_ref=evidence_ref,
                    round_num=round_num,
                ))
            else:
                candidates[key] = EntityCandidate.from_entity_mention(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    evidence_ref=evidence_ref,
                    round_num=round_num,
                )
    
    # Extract unresolved tokens (codenames) via pattern matching
    import re
    codename_pattern = re.compile(r'\b([A-Z]{3,})\b')
    
    for chunk in chunks:
        if not chunk.text:
            continue
        
        for match in codename_pattern.finditer(chunk.text):
            token = match.group(1)
            start = match.start()
            end = match.end()
            
            # Get context around the token
            context_start = max(0, start - 50)
            context_end = min(len(chunk.text), end + 50)
            quote_span = chunk.text[context_start:context_end]
            
            evidence_ref = EvidenceRef(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page_ref="",
                char_start=start,
                char_end=end,
                quote_span=quote_span,
            )
            
            key = f"token:{token}"
            if key in candidates:
                candidates[key].merge(EntityCandidate.from_unresolved_token(
                    token=token,
                    evidence_ref=evidence_ref,
                    round_num=round_num,
                ))
            else:
                candidates[key] = EntityCandidate.from_unresolved_token(
                    token=token,
                    evidence_ref=evidence_ref,
                    round_num=round_num,
                )
    
    return candidates


def prioritize_candidates(
    candidates: Dict[str, EntityCandidate],
    max_candidates: int,
) -> Dict[str, EntityCandidate]:
    """
    Prioritize and cap candidates to budget.
    
    Prioritizes by:
    1. Resolved entities over unresolved tokens
    2. Higher mention count
    3. Higher score
    """
    if len(candidates) <= max_candidates:
        return candidates
    
    # Sort by priority
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda c: (
            1 if c.entity_id else 0,  # Resolved first
            c.mention_count,
            c.score,
        ),
        reverse=True,
    )
    
    # Take top N
    return {c.key: c for c in sorted_candidates[:max_candidates]}


# =============================================================================
# Stability Check
# =============================================================================

def check_stability(
    old_candidates: Dict[str, EntityCandidate],
    new_candidates: Dict[str, EntityCandidate],
    stop_conditions: StopConditions,
) -> Tuple[bool, float, int]:
    """
    Check if candidate set is stable using Jaccard + novelty.
    
    Args:
        old_candidates: Candidates from previous round
        new_candidates: Candidates from current round
        stop_conditions: Thresholds for stability
        
    Returns:
        (is_stable, jaccard_score, novelty_count)
    """
    old_keys = set(old_candidates.keys())
    new_keys = set(new_candidates.keys())
    
    if not old_keys and not new_keys:
        return True, 1.0, 0
    
    intersection = old_keys & new_keys
    union = old_keys | new_keys
    
    jaccard = len(intersection) / max(len(union), 1)
    novelty = len(new_keys - old_keys)
    
    is_stable = (
        jaccard >= stop_conditions.jaccard_threshold and
        novelty <= stop_conditions.novelty_cap
    )
    
    return is_stable, jaccard, novelty


# =============================================================================
# Iterative Execution
# =============================================================================

def execute_seed_lanes(
    lanes: List[LaneSpec],
    conn,
) -> List[LaneResult]:
    """Execute seed lanes (Round 1)."""
    results = []
    
    # Sort by priority
    sorted_lanes = sorted(lanes, key=lambda l: l.priority)
    
    for spec in sorted_lanes:
        result = execute_lane(spec, conn)
        results.append(result)
    
    return results


def build_targeted_lanes(
    candidates: Dict[str, EntityCandidate],
    plan: AgenticPlan,
) -> List[LaneSpec]:
    """
    Build targeted lane specs for Round 2+.
    
    Creates co-mention expansion lane for roster queries.
    """
    lanes = []
    
    # Get entity IDs from candidates
    entity_ids = [
        c.entity_id for c in candidates.values()
        if c.entity_id is not None
    ]
    
    # Add co-mention expansion lane for roster queries
    if plan.intent == IntentFamily.ROSTER_ENUMERATION and entity_ids:
        lanes.append(CommentionLaneSpec(
            lane_id="comention_expand",
            seed_entity_ids=entity_ids[:20],  # Cap seed entities
            target_entity_type="person",
            k=plan.budgets.max_chunks_per_lane,
        ))
    
    return lanes


def execute_targeted_lanes(
    lanes: List[LaneSpec],
    conn,
) -> List[LaneResult]:
    """Execute targeted lanes (Round 2+)."""
    results = []
    
    for spec in lanes:
        result = execute_lane(spec, conn)
        results.append(result)
    
    return results


def build_lane_run(
    result: LaneResult,
    round_num: int,
    intent: IntentFamily,
    verification,
) -> RetrievalLaneRun:
    """Convert LaneResult to RetrievalLaneRun with coverage stats."""
    # Build sample refs (first 3 chunks)
    sample_refs = []
    for chunk_id in result.chunk_ids[:3]:
        sample_refs.append(EvidenceRef(
            chunk_id=chunk_id,
            doc_id=result.doc_ids.get(chunk_id, 0),
            page_ref="",
            char_start=0,
            char_end=0,
            quote_span="",
        ))
    
    lane_run = RetrievalLaneRun(
        lane_id=result.lane_id,
        query_terms=result.terms_matched,
        filters_applied={},
        hit_count=result.hit_count,
        doc_count=result.doc_count,
        unique_pages=result.unique_pages,
        top_terms_matched=result.terms_matched[:5],
        sample_refs=sample_refs,
        execution_ms=result.execution_ms,
        coverage_achieved=0.0,  # Will be computed
        round_num=round_num,
        chunk_ids=result.chunk_ids,
    )
    
    # Compute coverage
    lane_run.coverage_achieved = compute_coverage(
        lane_run, intent, verification
    )
    
    return lane_run


def execute_plan_iterative(
    plan: AgenticPlan,
    conn,
) -> EvidenceBundle:
    """
    Execute plan with iterative multi-round strategy.
    
    Round 1: seed lanes (entity, must-hit, hybrid)
    → Extract candidate tokens/entities from merged chunks
    Round 2+: targeted lanes per candidate (evidence bucketing, codename resolution)
    → Stop when stable (Jaccard + novelty)
    
    Args:
        plan: The typed execution plan
        conn: Database connection
        
    Returns:
        EvidenceBundle with all retrieval runs, candidates, etc.
    """
    bundle = EvidenceBundle(plan=plan, constraints=plan.constraints)
    all_chunks: Dict[int, ChunkWithProvenance] = {}
    all_candidates: Dict[str, EntityCandidate] = {}
    
    for round_num in range(1, plan.budgets.max_rounds + 1):
        if round_num == 1:
            # Seed lanes: broad discovery
            lane_results = execute_seed_lanes(plan.lanes, conn)
        else:
            # Targeted lanes: per-candidate evidence + co-mention expansion
            targeted_specs = build_targeted_lanes(all_candidates, plan)
            if not targeted_specs:
                # No targeted lanes to run
                break
            lane_results = execute_targeted_lanes(targeted_specs, conn)
        
        # Convert to RetrievalLaneRun and add to bundle
        for result in lane_results:
            lane_run = build_lane_run(
                result, round_num, plan.intent, plan.verification
            )
            bundle.retrieval_runs.append(lane_run)
        
        # Merge chunks with provenance
        new_chunk_ids = []
        for result in lane_results:
            new_ids = merge_lane_chunks(all_chunks, result, round_num)
            new_chunk_ids.extend(new_ids)
        
        # Check for early termination: no new chunks
        if not new_chunk_ids and plan.stop_conditions.stop_on_zero_new_chunks:
            bundle.stable = True
            break
        
        # Load text for new chunks (lazy loading)
        load_chunk_texts(all_chunks, new_chunk_ids, conn)
        
        # Extract candidates from new chunks
        round_chunks = [all_chunks[cid] for cid in new_chunk_ids]
        new_candidates = extract_candidates(round_chunks, conn, round_num)
        
        # Check stability with Jaccard + novelty
        is_stable, jaccard, novelty = check_stability(
            all_candidates, new_candidates, plan.stop_conditions
        )
        
        # Merge new candidates into all_candidates
        for key, candidate in new_candidates.items():
            if key in all_candidates:
                all_candidates[key].merge(candidate)
            else:
                all_candidates[key] = candidate
        
        if is_stable:
            bundle.stable = True
            break
        
        # Budget check: prioritize and cap
        if len(all_candidates) > plan.budgets.max_candidates_forward:
            all_candidates = prioritize_candidates(
                all_candidates, plan.budgets.max_candidates_forward
            )
        
        # Total chunks budget check
        if len(all_chunks) > plan.budgets.max_total_chunks:
            break
    
    # Finalize bundle
    bundle.rounds_executed = round_num
    bundle.entities = list(all_candidates.values())
    bundle.all_chunks = all_chunks
    
    # Extract unresolved tokens
    bundle.unresolved_tokens = [
        c.token for c in all_candidates.values()
        if c.token is not None
    ]
    
    return bundle
