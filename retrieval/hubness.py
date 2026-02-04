"""
Hubness Scoring for Agentic V2.

Prevents hub entities (Moscow, NYU, etc.) from dominating results
through specificity-based scoring.

Formula:
  support(x) = best FocusSpan score among spans containing x
  spec(x) = log((df_focus_chunks + 1) / (df_global + 1))
  final_score(x) = support(x) + lambda * clamp(spec, floor, cap)

This prevents hubs from winning even if co-mentioned everywhere.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.focus_bundle import FocusBundle
    from retrieval.candidate_proposer import ProposedCandidate


@dataclass
class CandidateScore:
    """Score breakdown for a candidate."""
    candidate_key: str
    display_name: str
    entity_id: Optional[int]
    support: float          # best FocusSpan score containing this candidate
    df_focus_chunks: int    # unique chunks in FocusBundle containing candidate
    df_global: int          # global chunk frequency from entity_df
    specificity: float      # log((df_focus+1)/(df_global+1)) clamped
    final_score: float      # support + lambda * specificity
    source_span_ids: List[str]  # spans supporting this score


def load_entity_df(conn) -> Dict[int, Tuple[int, int]]:
    """
    Load entity document/chunk frequencies from entity_df table.
    
    Returns dict: entity_id -> (doc_df, chunk_df)
    Raises error if table doesn't exist.
    """
    cur = conn.cursor()
    
    # Check if table exists first
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'entity_df'
        )
    """)
    exists = cur.fetchone()[0]
    
    if not exists:
        raise RuntimeError(
            "Table 'entity_df' does not exist. "
            "Run migration: psql $DATABASE_URL -f migrations/0049_focus_spans.sql"
        )
    
    cur.execute("SELECT entity_id, doc_df, chunk_df FROM entity_df")
    result = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    
    if not result:
        import sys
        print(
            "WARNING: entity_df table is empty. "
            "Run: SELECT refresh_entity_df(); to populate it.",
            file=sys.stderr
        )
    
    return result


def refresh_entity_df(conn) -> int:
    """
    Refresh entity_df table from entity_mentions.
    
    Returns number of entities updated.
    """
    cur = conn.cursor()
    cur.execute("SELECT refresh_entity_df()")
    conn.commit()
    
    cur.execute("SELECT COUNT(*) FROM entity_df")
    return cur.fetchone()[0]


def score_candidates_with_hubness(
    candidates: List["ProposedCandidate"],
    focus_bundle: "FocusBundle",
    entity_df: Dict[int, Tuple[int, int]],
    conn,
    lambda_weight: float = 0.3,
    spec_floor: float = -2.0,
    spec_cap: float = 2.0,
    support_min: float = 0.0,  # Changed: no minimum, let rendering/verification decide
) -> List[CandidateScore]:
    """
    Score candidates using support + hubness penalty.
    
    INVARIANT: Support is defined ONLY as max score over supporting spans.
    If supporting_spans is empty => support=0.0 and candidate is NOT renderable.
    NO FALLBACK to bundle-wide max score.
    
    Args:
        candidates: List of ProposedCandidate from candidate_proposer
        focus_bundle: The FocusBundle for span lookups
        entity_df: Global entity frequencies from load_entity_df()
        conn: Database connection
        lambda_weight: Weight for specificity term (default 0.3)
        spec_floor: Minimum specificity (default -2.0)
        spec_cap: Maximum specificity to avoid niche over-boost (default 2.0)
        support_min: Minimum support threshold (default 0.0 - let downstream filter)
    
    Returns:
        List of CandidateScore, sorted by final_score descending
        Candidates with 0 supporting spans are EXCLUDED (not renderable)
    """
    scores = []
    
    for candidate in candidates:
        # Get supporting spans via mention index (entity) or term matching (token)
        if candidate.entity_id:
            supporting_spans = focus_bundle.get_spans_for_entity(candidate.entity_id, conn)
        else:
            # For unresolved tokens, use STRICT term matching
            supporting_spans = focus_bundle.get_spans_for_term(candidate.display_name)
        
        # CRITICAL FIX: If no supporting spans, candidate is NOT renderable
        # Do NOT use any fallback - this prevents garbage candidates
        if not supporting_spans:
            continue  # Skip entirely - no evidence linkage
        
        # Support = max score over supporting spans ONLY
        # No default, no fallback - we already checked supporting_spans is non-empty
        support = max(s.score for s in supporting_spans)
        
        # Skip candidates below support threshold (if any)
        if support < support_min:
            continue
        
        # df_focus: unique chunks (not raw span count)
        df_focus_chunks = len({s.chunk_id for s in supporting_spans})
        
        # df_global from entity_df (chunk_df)
        df_global = 0
        if candidate.entity_id and candidate.entity_id in entity_df:
            df_global = entity_df[candidate.entity_id][1]  # chunk_df
        
        # Compute specificity with floor AND cap
        if df_global > 0:
            specificity = math.log((df_focus_chunks + 1) / (df_global + 1))
            specificity = max(min(specificity, spec_cap), spec_floor)
        else:
            specificity = 0.0  # Unknown entity, neutral
        
        # Final score
        final_score = support + lambda_weight * specificity
        
        scores.append(CandidateScore(
            candidate_key=candidate.key,
            display_name=candidate.display_name,
            entity_id=candidate.entity_id,
            support=support,
            df_focus_chunks=df_focus_chunks,
            df_global=df_global,
            specificity=specificity,
            final_score=final_score,
            source_span_ids=[s.span_id for s in supporting_spans[:5]],
        ))
    
    # Sort by final_score descending, then candidate_key for determinism
    scores.sort(key=lambda x: (-x.final_score, x.candidate_key))
    
    return scores


def filter_hub_entities(
    scores: List[CandidateScore],
    hub_threshold_global_df: int = 500,
    hub_threshold_specificity: float = -1.5,
) -> List[CandidateScore]:
    """
    Filter out likely hub entities based on global frequency and specificity.
    
    Hub entities are those with:
    - Very high global frequency (df_global > threshold)
    - Low specificity (appearing everywhere)
    
    Args:
        scores: List of CandidateScore
        hub_threshold_global_df: Global chunk frequency threshold
        hub_threshold_specificity: Specificity below which to filter
    
    Returns:
        Filtered list with hub entities removed
    """
    filtered = []
    
    for score in scores:
        # Skip if both conditions indicate hub
        is_hub = (
            score.df_global > hub_threshold_global_df and
            score.specificity < hub_threshold_specificity
        )
        
        if not is_hub:
            filtered.append(score)
    
    return filtered


def get_top_candidates(
    scores: List[CandidateScore],
    max_candidates: int = 25,
    require_min_support: float = 0.0,
) -> List[CandidateScore]:
    """
    Get top candidates with optional additional filtering.
    
    Args:
        scores: List of CandidateScore, already sorted
        max_candidates: Maximum to return
        require_min_support: Additional support threshold
    
    Returns:
        Top candidates
    """
    filtered = [s for s in scores if s.support >= require_min_support]
    return filtered[:max_candidates]


def explain_score(score: CandidateScore) -> str:
    """
    Generate human-readable explanation of a candidate's score.
    
    Useful for debugging and audit.
    """
    parts = [
        f"Candidate: {score.display_name} ({score.candidate_key})",
        f"  Support (best span score): {score.support:.3f}",
        f"  Focus chunks: {score.df_focus_chunks}",
        f"  Global chunks: {score.df_global}",
        f"  Specificity: {score.specificity:.3f}",
        f"  Final score: {score.final_score:.3f}",
    ]
    
    if score.df_global > 100:
        parts.append(f"  Note: High global frequency may indicate hub entity")
    
    if score.specificity < -1.0:
        parts.append(f"  Note: Low specificity - appears broadly in corpus")
    
    return "\n".join(parts)


def compute_hubness_stats(
    entity_df: Dict[int, Tuple[int, int]],
) -> Dict[str, float]:
    """
    Compute statistics about entity frequencies for tuning.
    
    Returns stats like median, 90th percentile, etc.
    """
    if not entity_df:
        return {}
    
    chunk_dfs = [v[1] for v in entity_df.values()]
    doc_dfs = [v[0] for v in entity_df.values()]
    
    chunk_dfs.sort()
    doc_dfs.sort()
    
    n = len(chunk_dfs)
    
    return {
        "total_entities": n,
        "chunk_df_median": chunk_dfs[n // 2] if n else 0,
        "chunk_df_90pct": chunk_dfs[int(n * 0.9)] if n else 0,
        "chunk_df_max": chunk_dfs[-1] if n else 0,
        "doc_df_median": doc_dfs[n // 2] if n else 0,
        "doc_df_90pct": doc_dfs[int(n * 0.9)] if n else 0,
        "doc_df_max": doc_dfs[-1] if n else 0,
    }
