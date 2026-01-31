#!/usr/bin/env python3
"""
OCR Resolver V3 - Full Implementation

Includes:
- Phase 2B: OCR-weighted edit distance
- Phase 3A: Local context features
- Phase 3B: Document-level anchoring (two-pass)
- Phase 3C: Cross-document priority scoring
- Allowlist/Blocklist integration

Usage:
    python scripts/resolve_ocr_candidates_v3.py --collection silvermaster --batch-size 500
"""

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import psycopg2
from psycopg2.extras import execute_values, Json

sys.path.insert(0, '.')
from retrieval.surface_norm import normalize_surface, compute_group_key, compute_candidate_set_hash
from retrieval.ocr_utils import (
    OCRConfusionTable, get_confusion_table,
    normalized_weighted_edit_distance,
    extract_context_features, ContextFeatures,
    compute_variant_key, compute_priority_score
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BATCH_SIZE = 500
TOP_K = 10
MIN_TRIGRAM_SIM = 0.25

# Scoring weights
WEIGHT_TRIGRAM = 0.35
WEIGHT_OCR_EDIT = 0.25       # OCR-weighted edit distance
WEIGHT_TOKEN_OVERLAP = 0.20
WEIGHT_CONTEXT = 0.10        # Context type hints
WEIGHT_TIER_BONUS = 0.10

# Decision thresholds
STRONG_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.15
QUEUE_THRESHOLD = 0.50
IGNORE_THRESHOLD = 0.35
ANCHOR_THRESHOLD = 0.80      # Threshold for creating anchors

# Bonuses
TIER1_BONUS = 0.10
TIER2_BONUS = 0.05
ANCHOR_BONUS = 0.15          # Bonus for matching an anchor
ALLOWLIST_BONUS = 0.20       # Bonus for being on allowlist


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CandidateInfo:
    id: int
    surface_norm: str
    quality_score: float
    doc_quality: str
    chunk_id: int
    document_id: int
    char_start: int
    char_end: int
    raw_span: str
    chunk_text: Optional[str] = None  # For context extraction


@dataclass
class AliasMatch:
    alias_norm: str
    entity_id: int
    entity_type: Optional[str]
    proposal_tier: Optional[int]
    doc_freq: int
    trigram_sim: float
    ocr_edit_sim: float = 0.0
    token_overlap: float = 0.0
    context_bonus: float = 0.0
    anchor_bonus: float = 0.0
    combined_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'alias_norm': self.alias_norm,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'tier': self.proposal_tier,
            'doc_freq': self.doc_freq,
            'trgm': round(self.trigram_sim, 4),
            'ocr_edit': round(self.ocr_edit_sim, 4),
            'tok': round(self.token_overlap, 4),
            'ctx': round(self.context_bonus, 4),
            'anchor': round(self.anchor_bonus, 4),
            'score': round(self.combined_score, 4)
        }


@dataclass
class ResolutionResult:
    candidate: CandidateInfo
    decision: str
    best_match: Optional[AliasMatch]
    top_matches: List[AliasMatch]
    score: float
    margin: float
    reason: str
    context_features: Optional[ContextFeatures] = None
    anchored: bool = False
    signals: Dict = field(default_factory=dict)


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def token_overlap(s1: str, s2: str) -> float:
    t1, t2 = set(s1.lower().split()), set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def compute_combined_score(
    match: AliasMatch,
    surface_norm: str,
    confusion_table: OCRConfusionTable,
    context: Optional[ContextFeatures] = None,
    is_anchored: bool = False
) -> AliasMatch:
    """Compute full combined score with all signals."""
    
    # Token overlap
    match.token_overlap = token_overlap(surface_norm, match.alias_norm)
    
    # OCR-weighted edit distance (convert to similarity)
    ocr_dist = normalized_weighted_edit_distance(surface_norm, match.alias_norm, confusion_table)
    match.ocr_edit_sim = 1.0 - ocr_dist
    
    # Context bonus
    if context and match.entity_type:
        type_hint = context.best_type_hint
        if type_hint:
            if type_hint == 'person' and match.entity_type == 'person':
                match.context_bonus = 0.1
            elif type_hint == 'organization' and match.entity_type in ('organization', 'other'):
                match.context_bonus = 0.1
            elif type_hint == 'location' and match.entity_type == 'location':
                match.context_bonus = 0.1
            # Penalty for mismatch
            elif type_hint == 'person' and match.entity_type not in ('person', 'cover_name'):
                match.context_bonus = -0.05
    
    # Anchor bonus
    if is_anchored:
        match.anchor_bonus = ANCHOR_BONUS
    
    # Tier bonus
    tier_bonus = 0.0
    if match.proposal_tier == 1:
        tier_bonus = TIER1_BONUS
    elif match.proposal_tier == 2:
        tier_bonus = TIER2_BONUS
    
    # Combined score
    match.combined_score = (
        WEIGHT_TRIGRAM * match.trigram_sim +
        WEIGHT_OCR_EDIT * match.ocr_edit_sim +
        WEIGHT_TOKEN_OVERLAP * match.token_overlap +
        WEIGHT_CONTEXT * (0.5 + match.context_bonus) +  # Normalize context to 0-1 range
        WEIGHT_TIER_BONUS * tier_bonus +
        match.anchor_bonus  # Direct add, not weighted
    )
    
    return match


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def get_conn():
    return psycopg2.connect(
        host='localhost', port=5432, dbname='neh', user='neh', password='neh'
    )


def load_confusion_table(conn) -> OCRConfusionTable:
    """Load OCR confusion table from database."""
    return OCRConfusionTable.from_database(conn)


def load_overrides(conn) -> Tuple[Set[str], Dict[str, Set[int]], Dict[str, int]]:
    """Load ban/prefer overrides."""
    cur = conn.cursor()
    
    cur.execute("SELECT surface_norm FROM entity_alias_overrides WHERE banned = TRUE AND scope = 'global'")
    ban_surfaces = {row[0] for row in cur.fetchall()}
    
    cur.execute("SELECT surface_norm, banned_entity_id FROM entity_alias_overrides WHERE banned_entity_id IS NOT NULL")
    ban_entity_map = defaultdict(set)
    for row in cur.fetchall():
        ban_entity_map[row[0]].add(row[1])
    
    cur.execute("SELECT surface_norm, forced_entity_id FROM entity_alias_overrides WHERE forced_entity_id IS NOT NULL")
    prefer_map = {row[0]: row[1] for row in cur.fetchall()}
    
    return ban_surfaces, dict(ban_entity_map), prefer_map


def load_allowlist(conn) -> Dict[str, int]:
    """Load allowlist (variant_key -> entity_id)."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT variant_key, entity_id FROM ocr_variant_allowlist")
        return {row[0]: row[1] for row in cur.fetchall()}
    except:
        return {}


def load_blocklist(conn) -> Set[str]:
    """Load blocklist (variant_keys to ignore)."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT variant_key FROM ocr_variant_blocklist WHERE variant_key IS NOT NULL")
        return {row[0] for row in cur.fetchall()}
    except:
        return set()


def load_document_anchors(conn, document_ids: List[int]) -> Dict[int, Dict[int, str]]:
    """
    Load existing anchors for documents.
    Returns: {document_id: {entity_id: anchor_surface_norm}}
    """
    if not document_ids:
        return {}
    
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT document_id, entity_id, anchor_surface_norm
            FROM document_anchors
            WHERE document_id = ANY(%s)
        """, (document_ids,))
        
        anchors: Dict[int, Dict[int, str]] = defaultdict(dict)
        for doc_id, entity_id, surface in cur.fetchall():
            anchors[doc_id][entity_id] = surface
        return dict(anchors)
    except:
        return {}


def save_document_anchor(conn, document_id: int, entity_id: int, surface_norm: str, score: float, method: str):
    """Save a new document anchor."""
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO document_anchors (document_id, entity_id, anchor_surface_norm, anchor_score, anchor_method)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id, entity_id, anchor_surface_norm) DO UPDATE SET
                anchor_score = GREATEST(document_anchors.anchor_score, EXCLUDED.anchor_score)
        """, (document_id, entity_id, surface_norm, score, method))
        conn.commit()
    except Exception as e:
        conn.rollback()


def batch_retrieve_matches(
    cur,
    surface_norms: List[str],
    top_k: int = TOP_K,
    min_similarity: float = MIN_TRIGRAM_SIM
) -> Dict[str, List[AliasMatch]]:
    """Batch retrieve matches using GIN-indexed similarity."""
    if not surface_norms:
        return {}
    
    unique_surfaces = list(dict.fromkeys(surface_norms))
    cur.execute("SET pg_trgm.similarity_threshold = %s", (min_similarity,))
    
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
    
    results: Dict[str, List[AliasMatch]] = defaultdict(list)
    for row in cur.fetchall():
        query_surface, alias_norm, entity_id, entity_type, tier, doc_freq, trgm_sim = row
        results[query_surface].append(AliasMatch(
            alias_norm=alias_norm,
            entity_id=entity_id,
            entity_type=entity_type,
            proposal_tier=tier,
            doc_freq=doc_freq or 0,
            trigram_sim=float(trgm_sim)
        ))
    
    for surface in results:
        results[surface].sort(key=lambda m: -m.trigram_sim)
        results[surface] = results[surface][:top_k]
    
    return dict(results)


def get_chunk_text(conn, chunk_id: int) -> Optional[str]:
    """Get chunk text for context extraction."""
    cur = conn.cursor()
    cur.execute("SELECT text FROM chunks WHERE id = %s", (chunk_id,))
    row = cur.fetchone()
    return row[0] if row else None


def get_pending_candidates_with_context(
    conn,
    batch_id: Optional[str] = None,
    collection: Optional[str] = None,
    limit: Optional[int] = None
) -> List[CandidateInfo]:
    """Get pending candidates with chunk text for context."""
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
    
    query = f"""
        SELECT 
            mc.id, mc.surface_norm, mc.quality_score, mc.doc_quality,
            mc.chunk_id, mc.document_id, mc.char_start, mc.char_end, mc.raw_span,
            c.text
        FROM mention_candidates mc
        JOIN documents d ON d.id = mc.document_id
        JOIN collections col ON col.id = d.collection_id
        LEFT JOIN chunks c ON c.id = mc.chunk_id
        {where_clause}
        ORDER BY mc.document_id, mc.id
        {limit_clause}
    """
    
    cur.execute(query, params)
    results = []
    for row in cur.fetchall():
        row_list = list(row)
        if row_list[2] is not None:
            row_list[2] = float(row_list[2])
        results.append(CandidateInfo(*row_list[:9], chunk_text=row_list[9]))
    return results


# =============================================================================
# RESOLUTION LOGIC
# =============================================================================

def resolve_candidate(
    candidate: CandidateInfo,
    matches: List[AliasMatch],
    confusion_table: OCRConfusionTable,
    ban_surfaces: Set[str],
    ban_entity_map: Dict[str, Set[int]],
    prefer_map: Dict[str, int],
    allowlist: Dict[str, int],
    blocklist: Set[str],
    doc_anchors: Dict[int, str],  # entity_id -> surface for this doc
    context: Optional[ContextFeatures] = None
) -> ResolutionResult:
    """Resolve a candidate with all features."""
    
    surface_norm = candidate.surface_norm
    variant_key = compute_variant_key(surface_norm)
    signals = {
        'quality': candidate.quality_score,
        'doc_quality': candidate.doc_quality,
        'variant_key': variant_key
    }
    
    # Blocklist check
    if variant_key in blocklist:
        return ResolutionResult(
            candidate=candidate, decision='ignore', best_match=None,
            top_matches=[], score=0.0, margin=0.0,
            reason='blocklisted', context_features=context, signals=signals
        )
    
    # Surface ban check
    if surface_norm in ban_surfaces:
        return ResolutionResult(
            candidate=candidate, decision='ignore', best_match=None,
            top_matches=[], score=0.0, margin=0.0,
            reason='surface_banned', context_features=context, signals=signals
        )
    
    # Allowlist check (auto-link)
    if variant_key in allowlist:
        entity_id = allowlist[variant_key]
        signals['allowlist_entity'] = entity_id
        # Find match for this entity
        for m in matches:
            if m.entity_id == entity_id:
                m.combined_score = 1.0  # Max score
                return ResolutionResult(
                    candidate=candidate, decision='resolved', best_match=m,
                    top_matches=[m], score=1.0, margin=1.0,
                    reason='allowlisted', context_features=context, signals=signals
                )
    
    if not matches:
        return ResolutionResult(
            candidate=candidate, decision='ignore', best_match=None,
            top_matches=[], score=0.0, margin=0.0,
            reason='no_matches', context_features=context, signals=signals
        )
    
    # Check which matches are anchored in this document
    anchored_entities = set(doc_anchors.keys())
    
    # Compute full scores
    scored_matches = []
    for m in matches:
        is_anchored = m.entity_id in anchored_entities
        scored = compute_combined_score(m, surface_norm, confusion_table, context, is_anchored)
        scored_matches.append(scored)
    
    # Apply entity bans
    if surface_norm in ban_entity_map:
        banned = ban_entity_map[surface_norm]
        scored_matches = [m for m in scored_matches if m.entity_id not in banned]
        if not scored_matches:
            return ResolutionResult(
                candidate=candidate, decision='ignore', best_match=None,
                top_matches=[], score=0.0, margin=0.0,
                reason='all_banned', context_features=context, signals=signals
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
                    reason='prefer_override', context_features=context, signals=signals
                )
    
    best = top_matches[0]
    margin = best.combined_score - top_matches[1].combined_score if len(top_matches) > 1 else 1.0
    
    signals['best_score'] = best.combined_score
    signals['margin'] = margin
    signals['anchored'] = best.entity_id in anchored_entities
    
    if context:
        signals['context'] = context.to_dict()
    
    # Adaptive thresholds
    strong_thresh = STRONG_THRESHOLD
    queue_thresh = QUEUE_THRESHOLD
    
    if candidate.quality_score > 0.7:
        strong_thresh -= 0.05
        queue_thresh -= 0.05
    elif candidate.quality_score < 0.5:
        strong_thresh += 0.10
        queue_thresh += 0.10
    
    if candidate.doc_quality == 'ocr':
        strong_thresh -= 0.05
        queue_thresh -= 0.05
    
    # Anchor bonus for thresholds
    if best.entity_id in anchored_entities:
        strong_thresh -= 0.10  # Easier to auto-link if anchored
    
    # Decision
    if best.combined_score >= strong_thresh and margin >= MARGIN_THRESHOLD:
        return ResolutionResult(
            candidate=candidate, decision='resolved', best_match=best,
            top_matches=top_matches, score=best.combined_score, margin=margin,
            reason='strong_match', context_features=context,
            anchored=best.entity_id in anchored_entities, signals=signals
        )
    
    if best.combined_score >= queue_thresh:
        close = [m for m in top_matches if best.combined_score - m.combined_score < 0.1]
        unique_entities = len(set(m.entity_id for m in close))
        
        if unique_entities > 1:
            return ResolutionResult(
                candidate=candidate, decision='queue', best_match=best,
                top_matches=top_matches, score=best.combined_score, margin=margin,
                reason=f'collision_{unique_entities}', context_features=context,
                signals=signals
            )
        else:
            return ResolutionResult(
                candidate=candidate, decision='resolved', best_match=best,
                top_matches=top_matches, score=best.combined_score, margin=margin,
                reason='single_entity', context_features=context,
                anchored=best.entity_id in anchored_entities, signals=signals
            )
    
    if best.combined_score >= IGNORE_THRESHOLD:
        return ResolutionResult(
            candidate=candidate, decision='queue', best_match=best,
            top_matches=top_matches, score=best.combined_score, margin=margin,
            reason='low_confidence', context_features=context, signals=signals
        )
    
    return ResolutionResult(
        candidate=candidate, decision='ignore', best_match=best,
        top_matches=top_matches, score=best.combined_score, margin=margin,
        reason='below_threshold', context_features=context, signals=signals
    )


def two_pass_resolve_document(
    candidates: List[CandidateInfo],
    conn,
    cur,
    confusion_table: OCRConfusionTable,
    ban_surfaces: Set[str],
    ban_entity_map: Dict[str, Set[int]],
    prefer_map: Dict[str, int],
    allowlist: Dict[str, int],
    blocklist: Set[str],
    existing_anchors: Dict[int, str]
) -> List[ResolutionResult]:
    """
    Two-pass resolution for a document's candidates.
    
    Pass 1: Find strong matches that become anchors
    Pass 2: Resolve remaining candidates with anchor bonuses
    """
    if not candidates:
        return []
    
    document_id = candidates[0].document_id
    
    # Get unique surfaces for batch retrieval
    surface_norms = [c.surface_norm for c in candidates]
    matches_by_surface = batch_retrieve_matches(cur, surface_norms)
    
    # Extract context for each candidate
    contexts = {}
    for c in candidates:
        if c.chunk_text:
            contexts[c.id] = extract_context_features(
                c.chunk_text, c.char_start, c.char_end
            )
        else:
            contexts[c.id] = None
    
    # Start with existing anchors
    doc_anchors = dict(existing_anchors)
    
    # PASS 1: Find new anchors (strong matches)
    pass1_results = []
    anchor_candidates = []  # Candidates that become anchors
    
    for candidate in candidates:
        matches = matches_by_surface.get(candidate.surface_norm, [])
        context = contexts.get(candidate.id)
        
        result = resolve_candidate(
            candidate, matches, confusion_table,
            ban_surfaces, ban_entity_map, prefer_map,
            allowlist, blocklist, doc_anchors, context
        )
        
        # Check if this should become an anchor
        if result.decision == 'resolved' and result.best_match:
            if result.score >= ANCHOR_THRESHOLD:
                anchor_candidates.append((result, candidate))
        
        pass1_results.append(result)
    
    # Save new anchors
    for result, candidate in anchor_candidates:
        entity_id = result.best_match.entity_id
        if entity_id not in doc_anchors:
            doc_anchors[entity_id] = candidate.surface_norm
            save_document_anchor(
                conn, document_id, entity_id,
                candidate.surface_norm, result.score, 'strong_match'
            )
    
    # PASS 2: Re-resolve candidates that weren't strong matches
    # Now with anchor bonuses applied
    final_results = []
    
    for i, result in enumerate(pass1_results):
        candidate = candidates[i]
        
        # Keep strong matches from pass 1
        if result.decision == 'resolved' and result.score >= ANCHOR_THRESHOLD:
            final_results.append(result)
            continue
        
        # Re-resolve with updated anchors
        matches = matches_by_surface.get(candidate.surface_norm, [])
        context = contexts.get(candidate.id)
        
        new_result = resolve_candidate(
            candidate, matches, confusion_table,
            ban_surfaces, ban_entity_map, prefer_map,
            allowlist, blocklist, doc_anchors, context
        )
        
        final_results.append(new_result)
    
    return final_results


# =============================================================================
# BATCH WRITE
# =============================================================================

def write_results_batch(conn, results: List[ResolutionResult]):
    """Write resolution results in batch."""
    if not results:
        return
    
    cur = conn.cursor()
    
    update_values = []
    mention_values = []
    queue_values = []
    
    for r in results:
        # Candidate update
        status = 'resolved' if r.decision == 'resolved' else r.decision
        
        context_dict = r.context_features.to_dict() if r.context_features else None
        signals_with_context = {**r.signals}
        if context_dict:
            signals_with_context['context'] = context_dict
        
        update_values.append((
            status,
            r.best_match.entity_id if r.best_match else None,
            f'ocr_v3_{r.reason}',
            r.score,
            r.margin,
            Json([m.to_dict() for m in r.top_matches]),
            Json(signals_with_context),
            r.best_match.entity_id if r.anchored and r.best_match else None,
            'anchored' if r.anchored else None,
            r.candidate.id
        ))
        
        # Entity mention
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
                r.candidate.raw_span,
                Json([m.to_dict() for m in r.top_matches]),
                candidate_scores,
                Json(r.signals),
                'pending',
                group_key,
                set_hash,
                candidate_ids
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
            top_candidates = v.top_cands::jsonb,
            resolution_signals = v.signals::jsonb,
            anchored_to_entity_id = v.anchor_entity::bigint,
            anchor_reason = v.anchor_reason
        FROM (VALUES %s) AS v(status, entity_id, method, score, margin, top_cands, signals, anchor_entity, anchor_reason, id)
        WHERE mc.id = v.id::bigint
    """, update_values)
    
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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='OCR Resolver V3 (Full Features)')
    parser.add_argument('--batch-id', help='Batch ID')
    parser.add_argument('--collection', help='Collection slug')
    parser.add_argument('--limit', type=int, help='Total candidates')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    conn = get_conn()
    cur = conn.cursor()
    
    print("=== OCR Resolver V3 (Full Features) ===")
    print(f"Collection: {args.collection or 'all'}")
    print(f"Limit: {args.limit or 'none'}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load resources
    print("Loading resources...")
    confusion_table = load_confusion_table(conn)
    print(f"  Confusion table: {len(confusion_table.confusions)} patterns")
    
    ban_surfaces, ban_entity_map, prefer_map = load_overrides(conn)
    print(f"  Overrides: {len(ban_surfaces)} bans, {len(prefer_map)} prefers")
    
    allowlist = load_allowlist(conn)
    blocklist = load_blocklist(conn)
    print(f"  Allowlist: {len(allowlist)}, Blocklist: {len(blocklist)}")
    
    # Get candidates
    print("\nLoading candidates...")
    all_candidates = get_pending_candidates_with_context(
        conn, args.batch_id, args.collection, args.limit
    )
    print(f"  Found {len(all_candidates)} pending candidates")
    
    if not all_candidates:
        print("No candidates to process.")
        return
    
    # Group by document for two-pass resolution
    by_document: Dict[int, List[CandidateInfo]] = defaultdict(list)
    for c in all_candidates:
        by_document[c.document_id].append(c)
    
    print(f"  Across {len(by_document)} documents")
    
    # Load existing anchors
    doc_ids = list(by_document.keys())
    all_anchors = load_document_anchors(conn, doc_ids)
    print(f"  Existing anchors: {sum(len(a) for a in all_anchors.values())}")
    
    # Process
    start_time = time.time()
    stats = {'resolved': 0, 'queue': 0, 'ignore': 0, 'total': 0, 'anchored': 0}
    reason_counts = defaultdict(int)
    
    doc_num = 0
    for document_id, doc_candidates in by_document.items():
        doc_num += 1
        
        existing_anchors = all_anchors.get(document_id, {})
        
        results = two_pass_resolve_document(
            doc_candidates, conn, cur, confusion_table,
            ban_surfaces, ban_entity_map, prefer_map,
            allowlist, blocklist, existing_anchors
        )
        
        if not args.dry_run:
            write_results_batch(conn, results)
        
        for r in results:
            stats[r.decision] += 1
            stats['total'] += 1
            if r.anchored:
                stats['anchored'] += 1
            reason_counts[r.reason] += 1
        
        if doc_num % 10 == 0:
            elapsed = time.time() - start_time
            rate = stats['total'] / elapsed if elapsed > 0 else 0
            print(f"  Processed {doc_num}/{len(by_document)} docs, {stats['total']} candidates ({rate:.0f}/sec)")
    
    elapsed = time.time() - start_time
    
    # Report
    print()
    print("=== RESOLUTION COMPLETE ===")
    print(f"  Total: {stats['total']}")
    print(f"  Resolved: {stats['resolved']} ({100*stats['resolved']/stats['total']:.1f}%)" if stats['total'] else "")
    print(f"    - Anchored: {stats['anchored']}")
    print(f"  Queued: {stats['queue']} ({100*stats['queue']/stats['total']:.1f}%)" if stats['total'] else "")
    print(f"  Ignored: {stats['ignore']} ({100*stats['ignore']/stats['total']:.1f}%)" if stats['total'] else "")
    
    if stats['resolved'] + stats['queue'] > 0:
        print(f"\n  Link rate: {100*stats['resolved']/(stats['resolved']+stats['queue']):.1f}%")
    
    print(f"\n  Time: {elapsed:.1f}s ({stats['total']/elapsed:.0f}/sec)" if elapsed > 0 else "")
    
    print("\n  Reason breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
    
    conn.close()


if __name__ == '__main__':
    main()
