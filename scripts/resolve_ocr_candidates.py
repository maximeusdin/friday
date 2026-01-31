#!/usr/bin/env python3
"""
OCR Resolver - IR-style top-K matching + scoring

Resolves candidate spans against the alias lexicon using:
- Trigram similarity (pg_trgm)
- Token overlap
- Edit distance
- Adaptive thresholds based on span quality and alias popularity

Decisions:
- auto-link: strong match with good margin → insert entity_mentions
- propose/queue: ambiguous → enqueue to mention_review_queue
- ignore: below threshold

Usage:
    python scripts/resolve_ocr_candidates.py --batch-id ocr_cand_xxx
    python scripts/resolve_ocr_candidates.py --collection silvermaster --limit 1000
"""

import argparse
import sys
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Set

import psycopg2
from psycopg2.extras import execute_values, Json

# Add parent directory for imports
sys.path.insert(0, '.')
from retrieval.surface_norm import normalize_surface


# =============================================================================
# CONFIGURATION
# =============================================================================

# Retrieval settings
TOP_K = 10                      # Number of candidates to retrieve
MIN_TRIGRAM_SIM = 0.25          # Minimum trigram similarity for retrieval

# Scoring weights
WEIGHT_TRIGRAM = 0.45
WEIGHT_TOKEN_OVERLAP = 0.25
WEIGHT_EDIT_DISTANCE = 0.20
WEIGHT_TIER_BONUS = 0.10

# Decision thresholds
STRONG_THRESHOLD = 0.75         # Auto-link if score >= this
MARGIN_THRESHOLD = 0.15         # Auto-link if margin to 2nd best >= this
QUEUE_THRESHOLD = 0.50          # Queue if score >= this (but not strong enough)
IGNORE_THRESHOLD = 0.35         # Ignore if below this

# Quality-based threshold adjustment
QUALITY_BONUS_HIGH = 0.05       # Lower threshold for high-quality spans
QUALITY_PENALTY_LOW = 0.10      # Raise threshold for low-quality spans

# Tier-based adjustments
TIER1_BONUS = 0.10              # Bonus for Tier 1 aliases
TIER2_BONUS = 0.05              # Bonus for Tier 2 aliases


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AliasCandidate:
    """A candidate alias match from the lexicon."""
    alias_norm: str
    entity_id: int
    entity_type: Optional[str]
    proposal_tier: Optional[int]
    doc_freq: int
    trigram_sim: float
    token_overlap: float
    edit_distance: float
    combined_score: float
    
    def to_dict(self) -> dict:
        return {
            'alias_norm': self.alias_norm,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'proposal_tier': self.proposal_tier,
            'doc_freq': self.doc_freq,
            'trigram_sim': round(self.trigram_sim, 4),
            'token_overlap': round(self.token_overlap, 4),
            'edit_distance': round(self.edit_distance, 4),
            'combined_score': round(self.combined_score, 4)
        }


@dataclass
class ResolutionResult:
    """Result of resolving a candidate span."""
    candidate_id: int
    surface_norm: str
    decision: str  # 'auto_link', 'queue', 'ignore'
    best_match: Optional[AliasCandidate]
    top_candidates: List[AliasCandidate]
    score: float
    margin: float
    reason: str


# =============================================================================
# EDIT DISTANCE (simple implementation)
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
    """Compute normalized edit distance (0 = identical, 1 = completely different)."""
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s1, s2) / max_len


def token_overlap(s1: str, s2: str) -> float:
    """Compute Jaccard token overlap."""
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


# =============================================================================
# MATCHING
# =============================================================================

def get_top_k_matches(
    cur,
    surface_norm: str,
    top_k: int = TOP_K,
    min_sim: float = MIN_TRIGRAM_SIM
) -> List[AliasCandidate]:
    """
    Retrieve top-K alias matches from lexicon using trigram similarity.
    """
    # Use pg_trgm for fast retrieval
    cur.execute("""
        SELECT 
            ali.alias_norm,
            ali.entity_id,
            ali.entity_type,
            ali.proposal_tier,
            ali.doc_freq,
            similarity(ali.alias_norm, %s) as trigram_sim
        FROM alias_lexicon_index ali
        WHERE similarity(ali.alias_norm, %s) >= %s
        ORDER BY trigram_sim DESC
        LIMIT %s
    """, (surface_norm, surface_norm, min_sim, top_k * 2))  # Get more, then score
    
    rows = cur.fetchall()
    
    candidates = []
    for row in rows:
        alias_norm, entity_id, entity_type, proposal_tier, doc_freq, trigram_sim = row
        
        # Compute additional scores
        tok_overlap = token_overlap(surface_norm, alias_norm)
        edit_dist = normalized_edit_distance(surface_norm, alias_norm)
        edit_score = 1.0 - edit_dist  # Convert distance to similarity
        
        # Tier bonus
        tier_bonus = 0.0
        if proposal_tier == 1:
            tier_bonus = TIER1_BONUS
        elif proposal_tier == 2:
            tier_bonus = TIER2_BONUS
        
        # Combined score
        combined = (
            WEIGHT_TRIGRAM * trigram_sim +
            WEIGHT_TOKEN_OVERLAP * tok_overlap +
            WEIGHT_EDIT_DISTANCE * edit_score +
            WEIGHT_TIER_BONUS * tier_bonus
        )
        
        candidates.append(AliasCandidate(
            alias_norm=alias_norm,
            entity_id=entity_id,
            entity_type=entity_type,
            proposal_tier=proposal_tier,
            doc_freq=doc_freq or 0,
            trigram_sim=trigram_sim,
            token_overlap=tok_overlap,
            edit_distance=edit_dist,
            combined_score=combined
        ))
    
    # Sort by combined score and take top-K
    candidates.sort(key=lambda c: -c.combined_score)
    return candidates[:top_k]


def resolve_candidate(
    cur,
    candidate_id: int,
    surface_norm: str,
    quality_score: float,
    doc_quality: str,
    ban_surfaces: Set[str],
    ban_entity_map: Dict[str, Set[int]],
    prefer_map: Dict[str, int]
) -> ResolutionResult:
    """
    Resolve a single candidate span against the lexicon.
    """
    # Check surface ban first
    if surface_norm in ban_surfaces:
        return ResolutionResult(
            candidate_id=candidate_id,
            surface_norm=surface_norm,
            decision='ignore',
            best_match=None,
            top_candidates=[],
            score=0.0,
            margin=0.0,
            reason='surface_banned'
        )
    
    # Get top-K matches
    matches = get_top_k_matches(cur, surface_norm)
    
    if not matches:
        return ResolutionResult(
            candidate_id=candidate_id,
            surface_norm=surface_norm,
            decision='ignore',
            best_match=None,
            top_candidates=[],
            score=0.0,
            margin=0.0,
            reason='no_matches'
        )
    
    # Apply entity bans
    if surface_norm in ban_entity_map:
        banned_entities = ban_entity_map[surface_norm]
        matches = [m for m in matches if m.entity_id not in banned_entities]
        if not matches:
            return ResolutionResult(
                candidate_id=candidate_id,
                surface_norm=surface_norm,
                decision='ignore',
                best_match=None,
                top_candidates=[],
                score=0.0,
                margin=0.0,
                reason='all_entities_banned'
            )
    
    # Apply prefer override
    if surface_norm in prefer_map:
        preferred_entity = prefer_map[surface_norm]
        preferred_matches = [m for m in matches if m.entity_id == preferred_entity]
        if preferred_matches:
            best = preferred_matches[0]
            return ResolutionResult(
                candidate_id=candidate_id,
                surface_norm=surface_norm,
                decision='auto_link',
                best_match=best,
                top_candidates=matches[:5],
                score=best.combined_score,
                margin=1.0,  # Override = max margin
                reason='prefer_override'
            )
    
    # Compute adaptive thresholds based on quality
    strong_thresh = STRONG_THRESHOLD
    queue_thresh = QUEUE_THRESHOLD
    ignore_thresh = IGNORE_THRESHOLD
    
    if quality_score > 0.7:
        strong_thresh -= QUALITY_BONUS_HIGH
        queue_thresh -= QUALITY_BONUS_HIGH
    elif quality_score < 0.5:
        strong_thresh += QUALITY_PENALTY_LOW
        queue_thresh += QUALITY_PENALTY_LOW
    
    # For OCR docs, be more lenient
    if doc_quality == 'ocr':
        strong_thresh -= 0.05
        queue_thresh -= 0.05
    
    # Get best match and margin
    best = matches[0]
    margin = best.combined_score - matches[1].combined_score if len(matches) > 1 else 1.0
    
    # Decision logic
    if best.combined_score >= strong_thresh and margin >= MARGIN_THRESHOLD:
        decision = 'auto_link'
        reason = f'strong_match(score={best.combined_score:.3f},margin={margin:.3f})'
    elif best.combined_score >= queue_thresh:
        # Check if multiple entities with similar scores
        close_matches = [m for m in matches if best.combined_score - m.combined_score < 0.1]
        unique_entities = len(set(m.entity_id for m in close_matches))
        if unique_entities > 1:
            decision = 'queue'
            reason = f'collision({unique_entities}_entities,score={best.combined_score:.3f})'
        else:
            decision = 'auto_link'
            reason = f'single_entity(score={best.combined_score:.3f})'
    elif best.combined_score >= ignore_thresh:
        decision = 'queue'
        reason = f'low_confidence(score={best.combined_score:.3f})'
    else:
        decision = 'ignore'
        reason = f'below_threshold(score={best.combined_score:.3f})'
    
    return ResolutionResult(
        candidate_id=candidate_id,
        surface_norm=surface_norm,
        decision=decision,
        best_match=best,
        top_candidates=matches[:5],
        score=best.combined_score,
        margin=margin,
        reason=reason
    )


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def get_conn():
    """Get database connection."""
    return psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='neh',
        user='neh',
        password='neh'
    )


def load_overrides(conn) -> Tuple[Set[str], Dict[str, Set[int]], Dict[str, int]]:
    """Load ban and prefer overrides."""
    cur = conn.cursor()
    
    # Surface bans
    cur.execute("""
        SELECT surface_norm FROM entity_alias_overrides
        WHERE banned = TRUE AND scope = 'global'
    """)
    ban_surfaces = {row[0] for row in cur.fetchall()}
    
    # Entity bans (surface -> set of banned entity_ids)
    cur.execute("""
        SELECT surface_norm, banned_entity_id FROM entity_alias_overrides
        WHERE banned_entity_id IS NOT NULL AND scope = 'global'
    """)
    ban_entity_map = {}
    for row in cur.fetchall():
        if row[0] not in ban_entity_map:
            ban_entity_map[row[0]] = set()
        ban_entity_map[row[0]].add(row[1])
    
    # Prefer mappings (surface -> preferred entity_id)
    cur.execute("""
        SELECT surface_norm, forced_entity_id FROM entity_alias_overrides
        WHERE forced_entity_id IS NOT NULL AND scope = 'global'
    """)
    prefer_map = {row[0]: row[1] for row in cur.fetchall()}
    
    return ban_surfaces, ban_entity_map, prefer_map


def get_pending_candidates(
    conn,
    batch_id: Optional[str] = None,
    collection: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Tuple]:
    """Get pending candidates to resolve."""
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
    
    # Join with documents/collections for filtering
    query = f"""
        SELECT 
            mc.id,
            mc.surface_norm,
            mc.quality_score,
            mc.doc_quality,
            mc.chunk_id,
            mc.document_id,
            mc.char_start,
            mc.char_end,
            mc.raw_span
        FROM mention_candidates mc
        JOIN documents d ON d.id = mc.document_id
        JOIN collections col ON col.id = d.collection_id
        {where_clause}
        ORDER BY mc.id
        {limit_clause}
    """
    
    cur.execute(query, params)
    return cur.fetchall()


def update_candidate_resolution(conn, result: ResolutionResult):
    """Update candidate with resolution result."""
    cur = conn.cursor()
    
    cur.execute("""
        UPDATE mention_candidates
        SET 
            resolution_status = %s,
            resolved_entity_id = %s,
            resolved_at = NOW(),
            resolution_method = %s,
            resolution_score = %s,
            resolution_margin = %s,
            top_candidates = %s
        WHERE id = %s
    """, (
        result.decision if result.decision != 'auto_link' else 'resolved',
        result.best_match.entity_id if result.best_match else None,
        f'ocr_{result.reason.split("(")[0]}',  # e.g. ocr_strong_match, ocr_collision
        result.score,
        result.margin,
        Json([c.to_dict() for c in result.top_candidates]),
        result.candidate_id
    ))
    
    conn.commit()


def insert_entity_mention(
    conn,
    result: ResolutionResult,
    chunk_id: int,
    document_id: int,
    char_start: int,
    char_end: int,
    raw_span: str
):
    """Insert resolved mention into entity_mentions."""
    cur = conn.cursor()
    
    if not result.best_match:
        return
    
    # Use 'ocr_lexicon' method (constraint-safe)
    method = 'ocr_lexicon'
    
    cur.execute("""
        INSERT INTO entity_mentions (
            entity_id, chunk_id, document_id,
            surface, surface_norm,
            start_char, end_char,
            confidence, method
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (
        result.best_match.entity_id,
        chunk_id,
        document_id,
        raw_span,
        result.surface_norm,
        char_start,
        char_end,
        result.score,
        method
    ))
    
    conn.commit()


def enqueue_for_review(
    conn,
    result: ResolutionResult,
    chunk_id: int,
    document_id: int,
    char_start: int,
    char_end: int,
    raw_span: str
):
    """Enqueue ambiguous candidate for human review."""
    cur = conn.cursor()
    
    # Build candidate info
    candidates_json = [c.to_dict() for c in result.top_candidates]
    candidate_entity_ids = [c.entity_id for c in result.top_candidates]
    
    # Compute group key
    from retrieval.surface_norm import compute_group_key, compute_candidate_set_hash
    group_key = compute_group_key(result.surface_norm, sorted(set(candidate_entity_ids)))
    candidate_set_hash = compute_candidate_set_hash(sorted(set(candidate_entity_ids)))
    
    cur.execute("""
        INSERT INTO mention_review_queue (
            mention_type, chunk_id, document_id,
            surface, surface_norm,
            start_char, end_char,
            context_excerpt, candidates,
            status,
            group_key, candidate_set_hash, candidate_entity_ids
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (
        'entity',  # OCR-derived entity mention
        chunk_id,
        document_id,
        raw_span,
        result.surface_norm,
        char_start,
        char_end,
        raw_span,  # Context could be expanded
        Json(candidates_json),
        'pending',
        group_key,
        candidate_set_hash,
        candidate_entity_ids
    ))
    
    conn.commit()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Resolve OCR candidate spans')
    parser.add_argument('--batch-id', help='Batch ID from candidate extraction')
    parser.add_argument('--collection', help='Collection slug to process')
    parser.add_argument('--limit', type=int, help='Limit number of candidates')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t update, just report')
    args = parser.parse_args()
    
    conn = get_conn()
    
    print("=== OCR Resolution ===")
    print(f"Batch ID: {args.batch_id or 'all'}")
    print(f"Collection: {args.collection or 'all'}")
    print(f"Limit: {args.limit or 'none'}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Load overrides
    ban_surfaces, ban_entity_map, prefer_map = load_overrides(conn)
    print(f"Loaded {len(ban_surfaces)} surface bans, {len(ban_entity_map)} entity bans, {len(prefer_map)} prefer mappings")
    
    # Get candidates
    candidates = get_pending_candidates(
        conn,
        batch_id=args.batch_id,
        collection=args.collection,
        limit=args.limit
    )
    print(f"Found {len(candidates)} pending candidates")
    
    if not candidates:
        print("No candidates to resolve.")
        return
    
    # Process
    start_time = time.time()
    cur = conn.cursor()
    
    stats = {
        'auto_link': 0,
        'queue': 0,
        'ignore': 0,
        'total': 0
    }
    
    reason_counts = {}
    
    for row in candidates:
        (candidate_id, surface_norm, quality_score, doc_quality,
         chunk_id, document_id, char_start, char_end, raw_span) = row
        
        result = resolve_candidate(
            cur,
            candidate_id=candidate_id,
            surface_norm=surface_norm,
            quality_score=quality_score or 0.5,
            doc_quality=doc_quality or 'ocr',
            ban_surfaces=ban_surfaces,
            ban_entity_map=ban_entity_map,
            prefer_map=prefer_map
        )
        
        stats[result.decision] += 1
        stats['total'] += 1
        
        reason_key = result.reason.split('(')[0]
        reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
        
        if not args.dry_run:
            # Update candidate status
            update_candidate_resolution(conn, result)
            
            # Take action based on decision
            if result.decision == 'auto_link':
                insert_entity_mention(
                    conn, result, chunk_id, document_id,
                    char_start, char_end, raw_span
                )
            elif result.decision == 'queue':
                enqueue_for_review(
                    conn, result, chunk_id, document_id,
                    char_start, char_end, raw_span
                )
        
        if stats['total'] % 500 == 0:
            print(f"  Processed {stats['total']}/{len(candidates)}...")
    
    elapsed = time.time() - start_time
    
    # Report
    print()
    print("=== RESOLUTION COMPLETE ===")
    print(f"  Total processed: {stats['total']}")
    print(f"  Auto-linked: {stats['auto_link']} ({100*stats['auto_link']/stats['total']:.1f}%)")
    print(f"  Queued: {stats['queue']} ({100*stats['queue']/stats['total']:.1f}%)")
    print(f"  Ignored: {stats['ignore']} ({100*stats['ignore']/stats['total']:.1f}%)")
    print(f"  Time: {elapsed:.1f}s ({stats['total']/elapsed:.1f} candidates/sec)" if elapsed > 0 else "")
    
    # Link rate = auto_link / (auto_link + queue)
    link_rate = stats['auto_link'] / (stats['auto_link'] + stats['queue']) if (stats['auto_link'] + stats['queue']) > 0 else 0
    print(f"\n  Link rate: {100*link_rate:.1f}%")
    
    print("\n  Reason breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes were made.")
    
    conn.close()


if __name__ == '__main__':
    main()
