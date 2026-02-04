"""
Stage A: Evidence Selection Module

Deterministic, reproducible selection of evidence chunks for summarization.

Key features:
- Adaptive doc scope rules (single_doc, small_corpus, global)
- Soft diversity caps (penalties, not hard limits)
- Greedy selection with per-iteration utility recomputation
- Deterministic pool building for reproducibility
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from .models import (
    ChunkCandidate,
    SelectionLambdas,
    SelectionInputs,
    SelectionSpec,
    SelectionResult,
)


# =============================================================================
# Adaptive Doc Scope Rules
# =============================================================================

def determine_doc_focus_mode(
    doc_scope_size: int,
    max_chunks: int,
    base_max_per_doc: int,
    explicit_mode: Optional[str] = None,  # "auto" | "global" | "single_doc"
) -> Tuple[str, int, bool]:
    """
    Determine effective doc focus mode based on scope size.
    
    Returns:
        (effective_mode, effective_max_per_doc, ensure_doc_diversity)
    
    Rules:
    - doc_scope_size == 1: single_doc mode, max_per_doc=max_chunks
    - 2 <= doc_scope_size <= 5: small_corpus, max_per_doc=ceil(max_chunks/doc_scope_size)
    - doc_scope_size > 5: global mode, max_per_doc=base_max_per_doc, strict diversity
    """
    if explicit_mode and explicit_mode != "auto":
        if explicit_mode == "single_doc":
            return "single_doc", max_chunks, False
        elif explicit_mode == "global":
            return "global", base_max_per_doc, True
        # small_corpus explicit
        return "small_corpus", max(base_max_per_doc, math.ceil(max_chunks / max(doc_scope_size, 1))), True
    
    # Auto mode
    if doc_scope_size == 1:
        return "single_doc", max_chunks, False
    elif doc_scope_size <= 5:
        effective_max = math.ceil(max_chunks / doc_scope_size)
        return "small_corpus", effective_max, True
    else:
        return "global", base_max_per_doc, True


# =============================================================================
# Candidate Pool Building
# =============================================================================

def compute_pool_seed(
    result_set_id: int,
    profile: str,
    prompt_version: str,
    model_name: str
) -> int:
    """Deterministic seed for any randomized operations."""
    seed_str = f"{result_set_id}:{profile}:{prompt_version}:{model_name}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)


def build_candidate_pool(
    conn,
    result_set_id: int,
    pool_limit: int = 5000,
    retrieval_mode: str = "conversational",
) -> Tuple[List[ChunkCandidate], int, Optional[int]]:
    """
    Build bounded candidate pool from result set.
    
    Args:
        conn: Database connection
        result_set_id: ID of the result set
        pool_limit: Maximum candidates to consider
        retrieval_mode: "conversational" or "thorough"
    
    Returns:
        (candidates, total_available, seed_used)
    """
    with conn.cursor() as cur:
        # Get total count first
        cur.execute(
            "SELECT COUNT(*) FROM result_set_chunks WHERE result_set_id = %s",
            (result_set_id,)
        )
        total_available = cur.fetchone()[0]
        
        # Build candidate pool based on mode
        if retrieval_mode == "thorough":
            # Thorough mode: chronological ordering
            # Use representative bucket sampling for later items
            candidates = _build_thorough_pool(conn, result_set_id, pool_limit, total_available)
            seed_used = None  # No randomness in our implementation
        else:
            # Conversational mode: top N by rank
            candidates = _build_conversational_pool(conn, result_set_id, pool_limit)
            seed_used = None
        
        return candidates, total_available, seed_used


def _build_conversational_pool(
    conn,
    result_set_id: int,
    pool_limit: int
) -> List[ChunkCandidate]:
    """Build pool by rank (hybrid score) for conversational mode."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                rsc.chunk_id,
                rsc.rank,
                rsc.document_id,
                d.source_name as doc_title,
                cm.first_page_id,
                EXTRACT(YEAR FROM cm.date_min)::int as year,
                COALESCE(mt.score_hybrid, 0) as score_hybrid,
                mt.score_lexical,
                mt.score_vector,
                COALESCE(mt.matched_entity_ids, '{}') as matched_entity_ids,
                COALESCE(mt.matched_phrases, '{}') as matched_phrases,
                COALESCE(mt.in_lexical, false) as in_lexical,
                COALESCE(mt.in_vector, false) as in_vector
            FROM result_set_chunks rsc
            JOIN documents d ON d.id = rsc.document_id
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = rsc.chunk_id
            LEFT JOIN result_set_match_traces mt 
                ON mt.result_set_id = rsc.result_set_id AND mt.chunk_id = rsc.chunk_id
            WHERE rsc.result_set_id = %s
            ORDER BY rsc.rank ASC
            LIMIT %s
            """,
            (result_set_id, pool_limit)
        )
        
        candidates = []
        for row in cur.fetchall():
            # Get page number from first_page_id if available
            page = None
            if row[4]:  # first_page_id
                cur.execute("SELECT page_number FROM pages WHERE id = %s", (row[4],))
                page_row = cur.fetchone()
                if page_row:
                    page = page_row[0]
            
            candidates.append(ChunkCandidate(
                chunk_id=row[0],
                rank=row[1],
                doc_id=row[2],
                doc_title=row[3] or "Unknown",
                page=page,
                year=row[5],
                score_hybrid=row[6] or 0.0,
                score_lexical=row[7],
                score_vector=row[8],
                matched_entity_ids=list(row[9] or []),
                matched_phrases=list(row[10] or []),
                in_lexical=row[11],
                in_vector=row[12],
            ))
        
        return candidates


def _build_thorough_pool(
    conn,
    result_set_id: int,
    pool_limit: int,
    total_available: int
) -> List[ChunkCandidate]:
    """
    Build pool for thorough mode: chronological + representative sampling.
    
    Strategy (Option A from plan - no randomness):
    - First 70% of pool: earliest N chronologically
    - Remaining 30%: top 1 per year bucket from remaining hits
    """
    primary_window = int(pool_limit * 0.7)
    
    with conn.cursor() as cur:
        # Get primary window (earliest by date, then by rank)
        cur.execute(
            """
            SELECT 
                rsc.chunk_id,
                rsc.rank,
                rsc.document_id,
                d.source_name as doc_title,
                cm.first_page_id,
                EXTRACT(YEAR FROM cm.date_min)::int as year,
                COALESCE(mt.score_hybrid, 0) as score_hybrid,
                mt.score_lexical,
                mt.score_vector,
                COALESCE(mt.matched_entity_ids, '{}') as matched_entity_ids,
                COALESCE(mt.matched_phrases, '{}') as matched_phrases,
                COALESCE(mt.in_lexical, false) as in_lexical,
                COALESCE(mt.in_vector, false) as in_vector
            FROM result_set_chunks rsc
            JOIN documents d ON d.id = rsc.document_id
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = rsc.chunk_id
            LEFT JOIN result_set_match_traces mt 
                ON mt.result_set_id = rsc.result_set_id AND mt.chunk_id = rsc.chunk_id
            WHERE rsc.result_set_id = %s
            ORDER BY cm.date_min ASC NULLS LAST, rsc.rank ASC
            LIMIT %s
            """,
            (result_set_id, primary_window)
        )
        
        candidates = []
        seen_chunk_ids = set()
        
        for row in cur.fetchall():
            page = None
            if row[4]:
                cur.execute("SELECT page_number FROM pages WHERE id = %s", (row[4],))
                page_row = cur.fetchone()
                if page_row:
                    page = page_row[0]
            
            candidates.append(ChunkCandidate(
                chunk_id=row[0],
                rank=row[1],
                doc_id=row[2],
                doc_title=row[3] or "Unknown",
                page=page,
                year=row[5],
                score_hybrid=row[6] or 0.0,
                score_lexical=row[7],
                score_vector=row[8],
                matched_entity_ids=list(row[9] or []),
                matched_phrases=list(row[10] or []),
                in_lexical=row[11],
                in_vector=row[12],
            ))
            seen_chunk_ids.add(row[0])
        
        # Get year buckets from remaining (not already in primary)
        if len(candidates) < pool_limit and total_available > primary_window:
            remaining_slots = pool_limit - len(candidates)
            
            # Get distinct years not fully covered
            cur.execute(
                """
                SELECT DISTINCT EXTRACT(YEAR FROM cm.date_min)::int as year
                FROM result_set_chunks rsc
                LEFT JOIN chunk_metadata cm ON cm.chunk_id = rsc.chunk_id
                WHERE rsc.result_set_id = %s
                  AND rsc.chunk_id NOT IN (SELECT unnest(%s::bigint[]))
                  AND cm.date_min IS NOT NULL
                ORDER BY year ASC
                """,
                (result_set_id, list(seen_chunk_ids))
            )
            year_buckets = [row[0] for row in cur.fetchall() if row[0]]
            
            # Take top 1 per year bucket
            for year in year_buckets[:remaining_slots]:
                cur.execute(
                    """
                    SELECT 
                        rsc.chunk_id,
                        rsc.rank,
                        rsc.document_id,
                        d.source_name as doc_title,
                        cm.first_page_id,
                        %s as year,
                        COALESCE(mt.score_hybrid, 0) as score_hybrid,
                        mt.score_lexical,
                        mt.score_vector,
                        COALESCE(mt.matched_entity_ids, '{}') as matched_entity_ids,
                        COALESCE(mt.matched_phrases, '{}') as matched_phrases,
                        COALESCE(mt.in_lexical, false) as in_lexical,
                        COALESCE(mt.in_vector, false) as in_vector
                    FROM result_set_chunks rsc
                    JOIN documents d ON d.id = rsc.document_id
                    LEFT JOIN chunk_metadata cm ON cm.chunk_id = rsc.chunk_id
                    LEFT JOIN result_set_match_traces mt 
                        ON mt.result_set_id = rsc.result_set_id AND mt.chunk_id = rsc.chunk_id
                    WHERE rsc.result_set_id = %s
                      AND EXTRACT(YEAR FROM cm.date_min)::int = %s
                      AND rsc.chunk_id NOT IN (SELECT unnest(%s::bigint[]))
                    ORDER BY mt.score_hybrid DESC NULLS LAST
                    LIMIT 1
                    """,
                    (year, result_set_id, year, list(seen_chunk_ids))
                )
                row = cur.fetchone()
                if row:
                    page = None
                    if row[4]:
                        cur.execute("SELECT page_number FROM pages WHERE id = %s", (row[4],))
                        page_row = cur.fetchone()
                        if page_row:
                            page = page_row[0]
                    
                    candidates.append(ChunkCandidate(
                        chunk_id=row[0],
                        rank=row[1],
                        doc_id=row[2],
                        doc_title=row[3] or "Unknown",
                        page=page,
                        year=row[5],
                        score_hybrid=row[6] or 0.0,
                        score_lexical=row[7],
                        score_vector=row[8],
                        matched_entity_ids=list(row[9] or []),
                        matched_phrases=list(row[10] or []),
                        in_lexical=row[11],
                        in_vector=row[12],
                    ))
                    seen_chunk_ids.add(row[0])
        
        return candidates


# =============================================================================
# Selection Algorithm (Greedy with Soft Caps)
# =============================================================================

def select_evidence(
    candidates: List[ChunkCandidate],
    max_chunks: int,
    effective_max_per_doc: int,
    lambdas: SelectionLambdas,
    page_bucket_size: int = 5,
    utility_threshold: float = -float('inf'),
) -> List[ChunkCandidate]:
    """
    Greedy selection with soft diversity caps.
    
    IMPORTANT: This is a greedy loop that RECOMPUTES utility each iteration.
    Penalties are dynamic based on what's already selected.
    
    Complexity: O(K * N) where K=max_chunks (~40), N=pool_size (~5000)
    = ~200k utility computations. Fast enough for real-time use.
    
    Args:
        candidates: Pool of candidates to select from
        max_chunks: Maximum chunks to select (K)
        effective_max_per_doc: Soft cap per document
        lambdas: Penalty weights
        page_bucket_size: Pages per bucket for intra-doc diversity
        utility_threshold: Minimum utility to accept (default: accept all)
    
    Returns:
        List of selected candidates in selection order
    """
    if not candidates:
        return []
    
    selected: List[ChunkCandidate] = []
    selected_ids: set = set()
    doc_counts: Dict[int, int] = defaultdict(int)
    page_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    time_counts: Dict[int, int] = defaultdict(int)
    entity_coverage: set = set()
    
    while len(selected) < max_chunks:
        best_candidate = None
        best_utility = float('-inf')
        
        # Greedy: find best remaining candidate given current state
        for candidate in candidates:
            if candidate.chunk_id in selected_ids:
                continue
            
            # Compute penalties based on CURRENT selection state
            # Doc penalty: kicks in after soft cap
            doc_count = doc_counts[candidate.doc_id]
            doc_penalty = lambdas.doc * max(0, doc_count - effective_max_per_doc + 1)
            
            # Page penalty: within-document clustering
            page_bucket = (candidate.page or 0) // page_bucket_size
            page_penalty = lambdas.page * page_counts[candidate.doc_id][page_bucket]
            
            # Time penalty: prefer time diversity
            time_bucket = candidate.year
            time_penalty = 0.0
            if time_bucket:
                time_penalty = lambdas.time * time_counts.get(time_bucket, 0)
            
            # Entity penalty: prefer entity diversity
            candidate_entities = set(candidate.matched_entity_ids or [])
            entity_overlap = len(candidate_entities & entity_coverage)
            entity_penalty = lambdas.entity * entity_overlap
            
            # Compute effective utility
            utility = candidate.score_hybrid - doc_penalty - page_penalty - time_penalty - entity_penalty
            
            if utility > best_utility:
                best_utility = utility
                best_candidate = candidate
        
        # Check if we found a viable candidate
        if best_candidate is None or best_utility < utility_threshold:
            break
        
        # Add to selection and update state
        selected.append(best_candidate)
        selected_ids.add(best_candidate.chunk_id)
        doc_counts[best_candidate.doc_id] += 1
        
        page_bucket = (best_candidate.page or 0) // page_bucket_size
        page_counts[best_candidate.doc_id][page_bucket] += 1
        
        if best_candidate.year:
            time_counts[best_candidate.year] += 1
        
        entity_coverage.update(best_candidate.matched_entity_ids or [])
    
    return selected


# =============================================================================
# Main Selection Function
# =============================================================================

def run_selection(
    conn,
    result_set_id: int,
    retrieval_run_id: int,
    retrieval_mode: str,
    max_chunks: int,
    base_max_per_doc: int,
    candidate_pool_limit: int,
    lambdas: SelectionLambdas,
    page_bucket_size: int,
    utility_threshold: float,
    doc_focus_mode: str,
    summary_type: str = "sample",
    page_window: Optional[Dict[str, int]] = None,
) -> SelectionResult:
    """
    Run the full selection pipeline (Stage A).
    
    Args:
        conn: Database connection
        result_set_id: ID of the result set
        retrieval_run_id: ID of the retrieval run
        retrieval_mode: "conversational" or "thorough"
        max_chunks: Maximum chunks to select
        base_max_per_doc: Base soft cap per document
        candidate_pool_limit: Max candidates to consider
        lambdas: Penalty weights
        page_bucket_size: Pages per bucket
        utility_threshold: Minimum utility
        doc_focus_mode: "auto", "global", or "single_doc"
        summary_type: "sample" or "page_window"
        page_window: Optional {offset, limit} for page_window mode
    
    Returns:
        SelectionResult with spec, inputs, and selected candidates
    """
    # Build candidate pool
    candidates, total_available, pool_seed = build_candidate_pool(
        conn, result_set_id, candidate_pool_limit, retrieval_mode
    )
    
    if not candidates:
        # Empty result
        return SelectionResult(
            spec=SelectionSpec(chunk_ids=[], policy="greedy_soft_diversity", bundle_id_map={}),
            inputs=SelectionInputs(
                candidate_pool_size=0,
                total_available=total_available,
                pool_seed=pool_seed,
                summary_type=summary_type,
                page_window=page_window,
                lambdas={"doc": lambdas.doc, "entity": lambdas.entity, "time": lambdas.time, "page": lambdas.page},
                utility_threshold=utility_threshold,
                doc_scope_size=0,
                doc_focus_mode=doc_focus_mode,
                effective_max_per_doc=base_max_per_doc,
                overrides={},
                facet_snapshot={"year_buckets": [], "top_docs": []},
            ),
            candidates=[],
        )
    
    # Determine doc scope and adaptive mode
    doc_ids = {c.doc_id for c in candidates}
    doc_scope_size = len(doc_ids)
    
    effective_mode, effective_max_per_doc, _ = determine_doc_focus_mode(
        doc_scope_size, max_chunks, base_max_per_doc, doc_focus_mode
    )
    
    # Compute facet snapshot for debugging
    year_counts: Dict[int, int] = defaultdict(int)
    doc_counts_snapshot: Dict[int, int] = defaultdict(int)
    for c in candidates:
        if c.year:
            year_counts[c.year] += 1
        doc_counts_snapshot[c.doc_id] += 1
    
    year_buckets = [{"year": y, "count": c} for y, c in sorted(year_counts.items())]
    top_docs = sorted(
        [{"doc_id": d, "count": c} for d, c in doc_counts_snapshot.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:10]
    
    # Track overrides
    overrides = {}
    if effective_mode != doc_focus_mode and doc_focus_mode == "auto":
        overrides["doc_focus_mode"] = {
            "from": "auto",
            "to": effective_mode,
            "reason": f"doc_scope_size={doc_scope_size}"
        }
    if effective_max_per_doc != base_max_per_doc:
        overrides["max_per_doc"] = {
            "from": base_max_per_doc,
            "to": effective_max_per_doc,
            "reason": f"adaptive for {effective_mode} mode"
        }
    
    # Run selection
    selected = select_evidence(
        candidates=candidates,
        max_chunks=max_chunks,
        effective_max_per_doc=effective_max_per_doc,
        lambdas=lambdas,
        page_bucket_size=page_bucket_size,
        utility_threshold=utility_threshold,
    )
    
    # Build bundle ID map
    bundle_id_map = {f"B{i+1}": c.chunk_id for i, c in enumerate(selected)}
    
    # Build result
    spec = SelectionSpec(
        chunk_ids=[c.chunk_id for c in selected],
        policy="greedy_soft_diversity",
        bundle_id_map=bundle_id_map,
    )
    
    inputs = SelectionInputs(
        candidate_pool_size=len(candidates),
        total_available=total_available,
        pool_seed=pool_seed,
        summary_type=summary_type,
        page_window=page_window,
        lambdas={"doc": lambdas.doc, "entity": lambdas.entity, "time": lambdas.time, "page": lambdas.page},
        utility_threshold=utility_threshold,
        doc_scope_size=doc_scope_size,
        doc_focus_mode=effective_mode,
        effective_max_per_doc=effective_max_per_doc,
        overrides=overrides,
        facet_snapshot={"year_buckets": year_buckets, "top_docs": top_docs},
    )
    
    return SelectionResult(spec=spec, inputs=inputs, candidates=selected)
