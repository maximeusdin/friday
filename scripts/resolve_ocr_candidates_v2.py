#!/usr/bin/env python3
"""
OCR Resolver V2 - Batch/Set-based Resolution

Key improvements over V1:
- Batch retrieval: ONE query for N candidates (10x+ throughput)
- Idempotent: unique keys prevent duplicates
- Evidence payload: stores full scoring details for review
- Restartable: can resume mid-run

Usage:
    python scripts/resolve_ocr_candidates_v2.py --batch-id xxx --batch-size 500
    python scripts/resolve_ocr_candidates_v2.py --collection silvermaster --limit 5000
"""

import argparse
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict

import psycopg2
from psycopg2.extras import execute_values, Json

sys.path.insert(0, '.')
from retrieval.surface_norm import normalize_surface, compute_group_key, compute_candidate_set_hash


# =============================================================================
# CONFIGURATION
# =============================================================================

# Batch settings
DEFAULT_BATCH_SIZE = 500        # Candidates per batch
TOP_K = 10                      # Top-K matches to retrieve per surface

# Scoring weights
WEIGHT_TRIGRAM = 0.45
WEIGHT_TOKEN_OVERLAP = 0.25
WEIGHT_EDIT_DISTANCE = 0.20
WEIGHT_TIER_BONUS = 0.10

# Decision thresholds
STRONG_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.15
QUEUE_THRESHOLD = 0.50
IGNORE_THRESHOLD = 0.35

# Quality adjustments
QUALITY_BONUS_HIGH = 0.05
QUALITY_PENALTY_LOW = 0.10
TIER1_BONUS = 0.10
TIER2_BONUS = 0.05


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CandidateInfo:
    """Info about a candidate being resolved."""
    id: int
    surface_norm: str
    quality_score: float
    doc_quality: str
    chunk_id: int
    document_id: int
    char_start: int
    char_end: int
    raw_span: str


@dataclass
class AliasMatch:
    """A match from the alias lexicon."""
    alias_norm: str
    entity_id: int
    entity_type: Optional[str]
    proposal_tier: Optional[int]
    doc_freq: int
    trigram_sim: float
    token_overlap: float = 0.0
    edit_distance: float = 0.0
    combined_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'alias_norm': self.alias_norm,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'proposal_tier': self.proposal_tier,
            'doc_freq': self.doc_freq,
            'trgm': round(self.trigram_sim, 4),
            'tok': round(self.token_overlap, 4),
            'edit': round(self.edit_distance, 4),
            'score': round(self.combined_score, 4)
        }


@dataclass 
class ResolutionResult:
    """Result of resolving a candidate."""
    candidate: CandidateInfo
    decision: str  # 'resolved', 'queue', 'ignore'
    best_match: Optional[AliasMatch]
    top_matches: List[AliasMatch]
    score: float
    margin: float
    reason: str
    signals: Dict[str, any] = field(default_factory=dict)


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Normalized edit distance (0 = identical, 1 = different)."""
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    return levenshtein_distance(s1, s2) / max_len if max_len > 0 else 0.0


def token_overlap(s1: str, s2: str) -> float:
    """Jaccard token overlap."""
    t1, t2 = set(s1.lower().split()), set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def compute_combined_score(match: AliasMatch, surface_norm: str) -> AliasMatch:
    """Compute combined score for a match."""
    # Token overlap
    match.token_overlap = token_overlap(surface_norm, match.alias_norm)
    
    # Edit distance (convert to similarity)
    edit_dist = normalized_edit_distance(surface_norm, match.alias_norm)
    match.edit_distance = edit_dist
    edit_score = 1.0 - edit_dist
    
    # Tier bonus
    tier_bonus = 0.0
    if match.proposal_tier == 1:
        tier_bonus = TIER1_BONUS
    elif match.proposal_tier == 2:
        tier_bonus = TIER2_BONUS
    
    # Combined
    match.combined_score = (
        WEIGHT_TRIGRAM * match.trigram_sim +
        WEIGHT_TOKEN_OVERLAP * match.token_overlap +
        WEIGHT_EDIT_DISTANCE * edit_score +
        WEIGHT_TIER_BONUS * tier_bonus
    )
    
    return match


# =============================================================================
# BATCH RETRIEVAL (THE KEY OPTIMIZATION)
# =============================================================================

def batch_retrieve_matches(
    cur,
    surface_norms: List[str],
    top_k: int = TOP_K,
    min_similarity: float = 0.25
) -> Dict[str, List[AliasMatch]]:
    """
    Retrieve top-K matches for MULTIPLE surfaces in ONE query.
    
    Uses GIN-indexed % operator for fast retrieval, then computes similarity.
    """
    if not surface_norms:
        return {}
    
    # Dedupe
    unique_surfaces = list(dict.fromkeys(surface_norms))
    
    # Set similarity threshold for % operator
    cur.execute("SET pg_trgm.similarity_threshold = %s", (min_similarity,))
    
    # Fast batch query using % operator (GIN-indexed) 
    # Then rank by similarity in Python for top-K per surface
    query = """
        WITH surfaces AS (
            SELECT unnest(%s::text[]) AS surface_norm
        )
        SELECT 
            s.surface_norm AS query_surface,
            ali.alias_norm,
            ali.entity_id,
            ali.entity_type,
            ali.proposal_tier,
            ali.doc_freq,
            similarity(ali.alias_norm, s.surface_norm) AS trgm_sim
        FROM surfaces s
        JOIN alias_lexicon_index ali ON ali.alias_norm %% s.surface_norm
    """
    
    cur.execute(query, (unique_surfaces,))
    rows = cur.fetchall()
    
    # Group by query surface and take top-K
    results: Dict[str, List[AliasMatch]] = defaultdict(list)
    for row in rows:
        query_surface, alias_norm, entity_id, entity_type, tier, doc_freq, trgm_sim = row
        results[query_surface].append(AliasMatch(
            alias_norm=alias_norm,
            entity_id=entity_id,
            entity_type=entity_type,
            proposal_tier=tier,
            doc_freq=doc_freq or 0,
            trigram_sim=float(trgm_sim)
        ))
    
    # Sort each surface's matches by similarity and take top-K
    for surface in results:
        results[surface].sort(key=lambda m: -m.trigram_sim)
        results[surface] = results[surface][:top_k]
    
    return dict(results)


# =============================================================================
# RESOLUTION LOGIC
# =============================================================================

def resolve_candidate(
    candidate: CandidateInfo,
    matches: List[AliasMatch],
    ban_surfaces: Set[str],
    ban_entity_map: Dict[str, Set[int]],
    prefer_map: Dict[str, int]
) -> ResolutionResult:
    """Resolve a single candidate given pre-fetched matches."""
    
    surface_norm = candidate.surface_norm
    # Convert Decimal to float for JSON serialization
    quality = float(candidate.quality_score) if candidate.quality_score else 0.5
    signals = {'quality': quality, 'doc_quality': candidate.doc_quality}
    
    # Surface ban check
    if surface_norm in ban_surfaces:
        return ResolutionResult(
            candidate=candidate, decision='ignore', best_match=None,
            top_matches=[], score=0.0, margin=0.0,
            reason='surface_banned', signals=signals
        )
    
    if not matches:
        return ResolutionResult(
            candidate=candidate, decision='ignore', best_match=None,
            top_matches=[], score=0.0, margin=0.0,
            reason='no_matches', signals=signals
        )
    
    # Compute full scores for matches
    scored_matches = [compute_combined_score(m, surface_norm) for m in matches]
    
    # Apply entity bans
    if surface_norm in ban_entity_map:
        banned = ban_entity_map[surface_norm]
        scored_matches = [m for m in scored_matches if m.entity_id not in banned]
        if not scored_matches:
            return ResolutionResult(
                candidate=candidate, decision='ignore', best_match=None,
                top_matches=[], score=0.0, margin=0.0,
                reason='all_banned', signals=signals
            )
    
    # Sort by score
    scored_matches.sort(key=lambda m: -m.combined_score)
    top_matches = scored_matches[:TOP_K]
    
    # Apply prefer override
    if surface_norm in prefer_map:
        pref_id = prefer_map[surface_norm]
        for m in top_matches:
            if m.entity_id == pref_id:
                signals['prefer_override'] = pref_id
                return ResolutionResult(
                    candidate=candidate, decision='resolved', best_match=m,
                    top_matches=top_matches, score=m.combined_score, margin=1.0,
                    reason='prefer_override', signals=signals
                )
    
    best = top_matches[0]
    margin = best.combined_score - top_matches[1].combined_score if len(top_matches) > 1 else 1.0
    
    signals['best_score'] = best.combined_score
    signals['margin'] = margin
    signals['top_entity_ids'] = [m.entity_id for m in top_matches[:5]]
    
    # Adaptive thresholds
    strong_thresh = STRONG_THRESHOLD
    queue_thresh = QUEUE_THRESHOLD
    
    if candidate.quality_score > 0.7:
        strong_thresh -= QUALITY_BONUS_HIGH
        queue_thresh -= QUALITY_BONUS_HIGH
    elif candidate.quality_score < 0.5:
        strong_thresh += QUALITY_PENALTY_LOW
        queue_thresh += QUALITY_PENALTY_LOW
    
    if candidate.doc_quality == 'ocr':
        strong_thresh -= 0.05
        queue_thresh -= 0.05
    
    # Decision
    if best.combined_score >= strong_thresh and margin >= MARGIN_THRESHOLD:
        return ResolutionResult(
            candidate=candidate, decision='resolved', best_match=best,
            top_matches=top_matches, score=best.combined_score, margin=margin,
            reason='strong_match', signals=signals
        )
    
    if best.combined_score >= queue_thresh:
        # Check for collision (multiple distinct entities with close scores)
        close = [m for m in top_matches if best.combined_score - m.combined_score < 0.1]
        unique_entities = len(set(m.entity_id for m in close))
        
        if unique_entities > 1:
            return ResolutionResult(
                candidate=candidate, decision='queue', best_match=best,
                top_matches=top_matches, score=best.combined_score, margin=margin,
                reason=f'collision_{unique_entities}', signals=signals
            )
        else:
            return ResolutionResult(
                candidate=candidate, decision='resolved', best_match=best,
                top_matches=top_matches, score=best.combined_score, margin=margin,
                reason='single_entity', signals=signals
            )
    
    # IMPORTANT: Do NOT queue low-confidence matches.
    # In OCR corpora, queuing the entire long tail (scores just above IGNORE_THRESHOLD)
    # explodes the review volume. If it's not good enough to pass queue_thresh, we ignore.
    if best.combined_score >= IGNORE_THRESHOLD:
        return ResolutionResult(
            candidate=candidate, decision='ignore', best_match=best,
            top_matches=top_matches, score=best.combined_score, margin=margin,
            reason='below_queue_threshold', signals=signals
        )
    
    return ResolutionResult(
        candidate=candidate, decision='ignore', best_match=best,
        top_matches=top_matches, score=best.combined_score, margin=margin,
        reason='below_threshold', signals=signals
    )


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def get_conn():
    return psycopg2.connect(
        host='localhost', port=5432, dbname='neh', user='neh', password='neh'
    )


def load_overrides(conn) -> Tuple[Set[str], Dict[str, Set[int]], Dict[str, int]]:
    """Load overrides."""
    cur = conn.cursor()
    
    cur.execute("SELECT surface_norm FROM entity_alias_overrides WHERE banned = TRUE AND scope = 'global'")
    ban_surfaces = {row[0] for row in cur.fetchall()}
    
    cur.execute("SELECT surface_norm, banned_entity_id FROM entity_alias_overrides WHERE banned_entity_id IS NOT NULL AND scope = 'global'")
    ban_entity_map = defaultdict(set)
    for row in cur.fetchall():
        ban_entity_map[row[0]].add(row[1])
    
    cur.execute("SELECT surface_norm, forced_entity_id FROM entity_alias_overrides WHERE forced_entity_id IS NOT NULL AND scope = 'global'")
    prefer_map = {row[0]: row[1] for row in cur.fetchall()}
    
    return ban_surfaces, dict(ban_entity_map), prefer_map


def get_pending_candidates(
    conn,
    batch_id: Optional[str] = None,
    collection: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[CandidateInfo]:
    """Get pending candidates."""
    cur = conn.cursor()
    
    conditions = ["mc.resolution_status = 'pending'"]
    params = []
    
    if batch_id:
        conditions.append("mc.batch_id = %s")
        params.append(batch_id)
    
    if collection:
        conditions.append("col.slug = %s")
        params.append(collection)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    limit_clause = f"LIMIT {limit}" if limit else ""
    offset_clause = f"OFFSET {offset}" if offset else ""
    
    query = f"""
        SELECT 
            mc.id, mc.surface_norm, mc.quality_score, mc.doc_quality,
            mc.chunk_id, mc.document_id, mc.char_start, mc.char_end, mc.raw_span
        FROM mention_candidates mc
        JOIN documents d ON d.id = mc.document_id
        JOIN collections col ON col.id = d.collection_id
        {where_clause}
        ORDER BY mc.id
        {limit_clause} {offset_clause}
    """
    
    cur.execute(query, params)
    results = []
    for row in cur.fetchall():
        # Convert Decimal to float for quality_score
        row_list = list(row)
        if row_list[2] is not None:  # quality_score
            row_list[2] = float(row_list[2])
        results.append(CandidateInfo(*row_list))
    return results


def write_results_batch(conn, results: List[ResolutionResult]):
    """Write resolution results in batch."""
    if not results:
        return
    
    cur = conn.cursor()
    
    # Update mention_candidates
    update_values = []
    mention_values = []
    queue_values = []
    
    for r in results:
        # Candidate update
        status = 'resolved' if r.decision == 'resolved' else r.decision
        update_values.append((
            status,
            r.best_match.entity_id if r.best_match else None,
            f'ocr_{r.reason}',
            r.score,
            r.margin,
            Json([m.to_dict() for m in r.top_matches]),
            Json(r.signals),
            r.candidate.id
        ))
        
        # Entity mention insert
        if r.decision == 'resolved' and r.best_match:
            mention_values.append((
                r.best_match.entity_id,
                r.candidate.chunk_id,
                r.candidate.document_id,
                r.candidate.raw_span,
                r.candidate.surface_norm,
                r.candidate.char_start,
                r.candidate.char_end,
                r.score,
                'ocr_lexicon'
            ))
        
        # Queue insert
        if r.decision == 'queue' and r.top_matches:
            candidate_ids = list(dict.fromkeys(m.entity_id for m in r.top_matches))
            candidate_scores = [float(m.combined_score) for m in r.top_matches]
            group_key = compute_group_key(r.candidate.surface_norm, sorted(set(candidate_ids)))
            set_hash = compute_candidate_set_hash(sorted(set(candidate_ids)))
            
            queue_values.append((
                'entity',
                r.candidate.chunk_id,
                r.candidate.document_id,
                r.candidate.raw_span,
                r.candidate.surface_norm,
                r.candidate.char_start,
                r.candidate.char_end,
                r.candidate.raw_span,  # context_excerpt (could expand)
                Json([m.to_dict() for m in r.top_matches]),
                candidate_scores,  # Pass as list, not Json
                Json(r.signals),
                'pending',
                group_key,
                set_hash,
                candidate_ids  # Pass as list, psycopg2 handles arrays
            ))
    
    # Batch update candidates
    execute_values(cur, """
        UPDATE mention_candidates AS mc SET
            resolution_status = v.status,
            resolved_entity_id = v.entity_id::bigint,
            resolved_at = NOW(),
            resolution_method = v.method,
            resolution_score = v.score::numeric,
            resolution_margin = v.margin::numeric,
            top_candidates = v.top_cands::jsonb
        FROM (VALUES %s) AS v(status, entity_id, method, score, margin, top_cands, signals, id)
        WHERE mc.id = v.id::bigint
    """, update_values)
    
    # Batch insert mentions (with conflict handling)
    if mention_values:
        execute_values(cur, """
            INSERT INTO entity_mentions (
                entity_id, chunk_id, document_id,
                surface, surface_norm,
                start_char, end_char,
                confidence, method
            ) VALUES %s
            ON CONFLICT (chunk_id, entity_id, start_char, end_char) DO NOTHING
        """, mention_values)
    
    # Batch insert queue items (with conflict handling)
    if queue_values:
        execute_values(cur, """
            INSERT INTO mention_review_queue (
                mention_type, chunk_id, document_id,
                surface, surface_norm,
                start_char, end_char,
                context_excerpt, candidates,
                candidate_scores, resolution_signals,
                status, group_key, candidate_set_hash, candidate_entity_ids
            ) VALUES %s
            ON CONFLICT (chunk_id, surface_norm, start_char, end_char) DO NOTHING
        """, queue_values)
    
    conn.commit()


# =============================================================================
# MAIN RESOLUTION LOOP
# =============================================================================

def resolve_batch(
    candidates: List[CandidateInfo],
    cur,
    ban_surfaces: Set[str],
    ban_entity_map: Dict[str, Set[int]],
    prefer_map: Dict[str, int]
) -> List[ResolutionResult]:
    """Resolve a batch of candidates."""
    
    # Get unique surface norms
    surface_norms = [c.surface_norm for c in candidates]
    
    # BATCH RETRIEVAL - one query for all
    matches_by_surface = batch_retrieve_matches(cur, surface_norms)
    
    # Resolve each candidate
    results = []
    for candidate in candidates:
        matches = matches_by_surface.get(candidate.surface_norm, [])
        result = resolve_candidate(
            candidate, matches,
            ban_surfaces, ban_entity_map, prefer_map
        )
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='OCR Resolver V2 (batch)')
    parser.add_argument('--batch-id', help='Batch ID from candidate extraction')
    parser.add_argument('--collection', help='Collection slug')
    parser.add_argument('--limit', type=int, help='Total candidates to process')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Candidates per batch')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t write results')
    args = parser.parse_args()
    
    conn = get_conn()
    cur = conn.cursor()
    
    print("=== OCR Resolver V2 (Batch) ===")
    print(f"Batch ID: {args.batch_id or 'all'}")
    print(f"Collection: {args.collection or 'all'}")
    print(f"Limit: {args.limit or 'none'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Load overrides
    ban_surfaces, ban_entity_map, prefer_map = load_overrides(conn)
    print(f"Loaded {len(ban_surfaces)} surface bans, {len(ban_entity_map)} entity bans, {len(prefer_map)} prefer mappings")
    
    # Stats
    stats = {'resolved': 0, 'queue': 0, 'ignore': 0, 'total': 0}
    reason_counts = defaultdict(int)
    
    start_time = time.time()
    offset = 0
    batch_num = 0
    
    while True:
        # Get next batch
        batch_limit = min(args.batch_size, args.limit - offset) if args.limit else args.batch_size
        if batch_limit <= 0:
            break
        
        candidates = get_pending_candidates(
            conn,
            batch_id=args.batch_id,
            collection=args.collection,
            limit=batch_limit,
            offset=0  # Always 0 since we update status
        )
        
        if not candidates:
            break
        
        batch_num += 1
        batch_start = time.time()
        
        # Resolve batch
        results = resolve_batch(
            candidates, cur,
            ban_surfaces, ban_entity_map, prefer_map
        )
        
        # Write results
        if not args.dry_run:
            write_results_batch(conn, results)
        
        # Update stats
        for r in results:
            stats[r.decision] += 1
            stats['total'] += 1
            reason_counts[r.reason] += 1
        
        batch_elapsed = time.time() - batch_start
        rate = len(candidates) / batch_elapsed if batch_elapsed > 0 else 0
        print(f"  Batch {batch_num}: {len(candidates)} candidates in {batch_elapsed:.1f}s ({rate:.0f}/sec)")
        
        offset += len(candidates)
        
        if args.limit and offset >= args.limit:
            break
    
    elapsed = time.time() - start_time
    
    # Report
    print()
    print("=== RESOLUTION COMPLETE ===")
    print(f"  Total: {stats['total']}")
    print(f"  Resolved: {stats['resolved']} ({100*stats['resolved']/stats['total']:.1f}%)" if stats['total'] else "")
    print(f"  Queued: {stats['queue']} ({100*stats['queue']/stats['total']:.1f}%)" if stats['total'] else "")
    print(f"  Ignored: {stats['ignore']} ({100*stats['ignore']/stats['total']:.1f}%)" if stats['total'] else "")
    
    if stats['resolved'] + stats['queue'] > 0:
        link_rate = stats['resolved'] / (stats['resolved'] + stats['queue'])
        print(f"\n  Link rate: {100*link_rate:.1f}%")
    
    print(f"\n  Time: {elapsed:.1f}s ({stats['total']/elapsed:.0f} candidates/sec)" if elapsed > 0 else "")
    
    print("\n  Reason breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
    
    conn.close()


if __name__ == '__main__':
    main()
