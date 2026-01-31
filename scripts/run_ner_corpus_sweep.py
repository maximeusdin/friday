#!/usr/bin/env python3
"""
NER Corpus Sweep - Discover new entity surfaces from corpus-wide NER

This script runs NER across the corpus to discover entity surfaces that aren't
in the concordance-derived alias lexicon. It aggregates frequency evidence
and proposes tiers for lexicon expansion.

The key insight: corpus frequency validates NER proposals.
- High doc_freq + consistent NER label = likely real entity
- Low doc_freq + inconsistent label = likely noise

Usage:
    # Dry run on small subset
    python scripts/run_ner_corpus_sweep.py --limit 100 --dry-run
    
    # Test on specific collection
    python scripts/run_ner_corpus_sweep.py --collection venona --limit 500
    
    # Full corpus sweep (will take time)
    python scripts/run_ner_corpus_sweep.py --all
    
    # Resume a previous sweep
    python scripts/run_ner_corpus_sweep.py --resume <sweep_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values, Json

from retrieval.ner_integration import (
    NERExtractor, NERSpan, SPACY_AVAILABLE,
)
from retrieval.ner_guardrails import NERGuardrails
from retrieval.normalization import normalize_for_fts as normalize_surface

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum thresholds for tier assignment
TIER1_MIN_DOCS = 10          # Tier 1: appears in 10+ documents
TIER1_MIN_CONSISTENCY = 0.8  # Tier 1: 80%+ label consistency
TIER2_MIN_DOCS = 3           # Tier 2: appears in 3+ documents
TIER2_MIN_CONSISTENCY = 0.5  # Tier 2: 50%+ label consistency

# Context window for snippets
CONTEXT_WINDOW = 50  # chars before/after

# Batch sizes
CHUNK_BATCH_SIZE = 50
DB_BATCH_SIZE = 1000
COMMIT_INTERVAL = 100  # Commit every N chunks


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_conn():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'neh'),
        user=os.getenv('DB_USER', 'neh'),
        password=os.getenv('DB_PASS', 'neh')
    )


def check_schema(conn) -> bool:
    """Check that required tables exist."""
    cur = conn.cursor()
    required_tables = ['ner_surface_stats', 'ner_corpus_sweeps', 'ner_surface_mentions']
    
    for table in required_tables:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table,))
        if not cur.fetchone()[0]:
            print(f"ERROR: Table '{table}' does not exist.")
            print("Run migration: psql -f migrations/0038_ner_corpus_discovery.sql")
            return False
    
    return True


def get_chunks_for_sweep(
    conn,
    collection: Optional[str] = None,
    doc_ids: Optional[List[int]] = None,
    limit: Optional[int] = None,
    processed_chunk_ids: Optional[Set[int]] = None,
) -> List[Dict]:
    """Get chunks to process."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    conditions = ["c.text IS NOT NULL", "LENGTH(c.text) > 50"]
    params = []
    
    if collection:
        conditions.append("(col.slug = %s OR col.title = %s)")
        params.extend([collection, collection])
    
    if doc_ids:
        conditions.append("d.id = ANY(%s)")
        params.append(doc_ids)
    
    if processed_chunk_ids:
        conditions.append("c.id != ALL(%s)")
        params.append(list(processed_chunk_ids))
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
        SELECT 
            c.id as chunk_id,
            c.text,
            d.id as document_id,
            d.source_name as document_name,
            col.slug as collection_slug
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id
        JOIN documents d ON d.id = cm.document_id
        JOIN collections col ON col.id = d.collection_id
        WHERE {where_clause}
        ORDER BY c.id
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def get_existing_aliases(conn) -> Set[str]:
    """Get all normalized surfaces already in entity_aliases."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT alias_norm 
        FROM entity_aliases 
        WHERE alias_norm IS NOT NULL
    """)
    return {row[0] for row in cur.fetchall()}


def get_existing_lexicon_surfaces(conn) -> Set[str]:
    """Get surfaces already in alias_lexicon_index."""
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT alias_norm FROM alias_lexicon_index")
    return {row[0] for row in cur.fetchall()}


# =============================================================================
# SWEEP MANAGEMENT
# =============================================================================

def create_sweep_record(
    conn,
    sweep_id: str,
    collection: Optional[str],
    doc_ids: Optional[List[int]],
    model: str,
    threshold: float,
    config: Dict,
    chunks_total: int,
) -> None:
    """Create a new sweep record."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ner_corpus_sweeps 
            (sweep_id, collection_filter, doc_filter, model, threshold, config, chunks_total, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'running')
    """, (
        sweep_id,
        collection,
        json.dumps(doc_ids) if doc_ids else None,
        model,
        threshold,
        Json(config),
        chunks_total,
    ))
    conn.commit()


def update_sweep_progress(
    conn,
    sweep_id: str,
    chunks_processed: int,
    raw_spans: int,
    unique_surfaces: int,
) -> None:
    """Update sweep progress."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE ner_corpus_sweeps SET
            chunks_processed = %s,
            raw_spans_extracted = %s,
            unique_surfaces_found = %s
        WHERE sweep_id = %s
    """, (chunks_processed, raw_spans, unique_surfaces, sweep_id))


def complete_sweep(
    conn,
    sweep_id: str,
    stats: Dict,
) -> None:
    """Mark sweep as completed with final stats."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE ner_corpus_sweeps SET
            status = 'completed',
            completed_at = NOW(),
            chunks_processed = %s,
            raw_spans_extracted = %s,
            unique_surfaces_found = %s,
            surfaces_new = %s,
            surfaces_matching = %s,
            tier1_count = %s,
            tier2_count = %s,
            rejected_count = %s
        WHERE sweep_id = %s
    """, (
        stats['chunks_processed'],
        stats['raw_spans'],
        stats['unique_surfaces'],
        stats['surfaces_new'],
        stats['surfaces_matching'],
        stats['tier1_count'],
        stats['tier2_count'],
        stats['rejected_count'],
        sweep_id,
    ))
    conn.commit()


def get_sweep_processed_chunks(conn, sweep_id: str) -> Set[int]:
    """Get chunk IDs already processed in a sweep."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT chunk_id 
        FROM ner_surface_mentions 
        WHERE sweep_id = %s
    """, (sweep_id,))
    return {row[0] for row in cur.fetchall()}


# =============================================================================
# SURFACE AGGREGATION
# =============================================================================

class SurfaceAggregator:
    """Aggregates NER spans into surface statistics."""
    
    def __init__(self):
        self.surfaces: Dict[str, Dict] = defaultdict(lambda: {
            'doc_ids': set(),
            'chunk_ids': set(),
            'mention_count': 0,
            'labels': defaultdict(int),
            'accept_scores': [],
            'context_hints_agg': defaultdict(int),
            'example_contexts': [],
            'raw_surfaces': set(),  # Original non-normalized forms
        })
    
    def add_span(
        self,
        span: NERSpan,
        surface_norm: str,
        doc_id: int,
        chunk_id: int,
        context_window: str,
    ) -> None:
        """Add a span to the aggregator."""
        s = self.surfaces[surface_norm]
        s['doc_ids'].add(doc_id)
        s['chunk_ids'].add(chunk_id)
        s['mention_count'] += 1
        s['labels'][span.label] += 1
        s['accept_scores'].append(span.accept_score)
        s['raw_surfaces'].add(span.surface)
        
        # Aggregate context hints
        for hint, count in span.context_hints.items():
            if isinstance(count, (int, float)) and count > 0:
                s['context_hints_agg'][hint] += count
        
        # Keep sample contexts (up to 5)
        if len(s['example_contexts']) < 5 and context_window:
            s['example_contexts'].append(context_window)
    
    def get_stats(self) -> List[Dict]:
        """Convert aggregated data to stats records."""
        results = []
        
        for surface_norm, data in self.surfaces.items():
            # Calculate label distribution and consistency
            total_labels = sum(data['labels'].values())
            label_dist = dict(data['labels'])
            primary_label = max(data['labels'], key=data['labels'].get) if data['labels'] else None
            label_consistency = data['labels'][primary_label] / total_labels if primary_label else 0
            
            # Map NER label to entity type
            inferred_type = self._map_label_to_type(primary_label)
            
            # Calculate score stats
            scores = data['accept_scores']
            avg_score = sum(scores) / len(scores) if scores else 0
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 0
            
            results.append({
                'surface_norm': surface_norm,
                'doc_count': len(data['doc_ids']),
                'chunk_count': len(data['chunk_ids']),
                'mention_count': data['mention_count'],
                'primary_label': primary_label,
                'label_distribution': label_dist,
                'label_consistency': label_consistency,
                'inferred_type': inferred_type,
                'type_confidence': label_consistency,  # Use consistency as confidence
                'context_hints_agg': dict(data['context_hints_agg']),
                'example_contexts': data['example_contexts'],
                'avg_accept_score': avg_score,
                'min_accept_score': min_score,
                'max_accept_score': max_score,
                'raw_surfaces': list(data['raw_surfaces'])[:10],  # Keep up to 10 variants
            })
        
        return results
    
    def _map_label_to_type(self, label: str) -> Optional[str]:
        """Map NER label to entity type."""
        if not label:
            return None
        mapping = {
            'PERSON': 'person',
            'ORG': 'org',
            'GPE': 'place',
            'LOC': 'place',
            'FAC': 'place',
            'NORP': 'org',  # Nationalities/groups
        }
        return mapping.get(label)


def cluster_ocr_variants(
    surface_stats: List[Dict],
    similarity_threshold: float = 0.85,
) -> List[Dict]:
    """
    Cluster OCR variants together to combine their evidence.
    
    For example: WASHINGTON, WASHINGT0N, WASHINCTON → merged stats
    
    Uses simple pairwise similarity with progress tracking.
    Slower but catches all OCR variants regardless of error position.
    """
    from difflib import SequenceMatcher
    import sys
    
    if not surface_stats:
        return []
    
    n = len(surface_stats)
    print(f"  Clustering {n} surfaces (this may take a moment)...")
    
    # Sort by doc_count descending (canonical forms first)
    sorted_stats = sorted(surface_stats, key=lambda x: x['doc_count'], reverse=True)
    
    clustered = []
    used_indices = set()
    
    # Progress tracking
    last_percent = -1
    comparisons = 0
    merges = 0
    
    for i, canonical in enumerate(sorted_stats):
        if i in used_indices:
            continue
        
        # Progress update every 1%
        percent = (i * 100) // n
        if percent > last_percent:
            print(f"    Progress: {percent}% ({i}/{n} surfaces, {merges} merges so far)", end='\r')
            sys.stdout.flush()
            last_percent = percent
        
        canonical_norm = canonical['surface_norm']
        canonical_lower = canonical_norm.lower()
        canonical_len = len(canonical_norm)
        
        merged = canonical.copy()
        merged['ocr_variants'] = [canonical_norm]
        
        # Find similar surfaces to merge
        for j in range(i + 1, n):
            if j in used_indices:
                continue
            
            candidate = sorted_stats[j]
            candidate_norm = candidate['surface_norm']
            
            # Quick length filter (OCR rarely changes length by more than 2)
            if abs(len(candidate_norm) - canonical_len) > 2:
                continue
            
            comparisons += 1
            
            # Compute similarity
            ratio = SequenceMatcher(None, canonical_lower, candidate_norm.lower()).ratio()
            
            if ratio >= similarity_threshold:
                merges += 1
                
                # Merge counts
                merged['doc_count'] += candidate['doc_count']
                merged['chunk_count'] += candidate['chunk_count']
                merged['mention_count'] += candidate['mention_count']
                merged['ocr_variants'].append(candidate_norm)
                
                # Merge label distributions
                for label, count in candidate.get('label_distribution', {}).items():
                    if 'label_distribution' not in merged:
                        merged['label_distribution'] = {}
                    merged['label_distribution'][label] = merged['label_distribution'].get(label, 0) + count
                
                # Merge example contexts (limit to 5)
                if len(merged.get('example_contexts', [])) < 5:
                    merged['example_contexts'] = (merged.get('example_contexts', []) + 
                                                  candidate.get('example_contexts', []))[:5]
                
                used_indices.add(j)
        
        # Recalculate consistency if we merged variants
        if len(merged['ocr_variants']) > 1:
            label_dist = merged.get('label_distribution', {})
            total = sum(label_dist.values()) if label_dist else 0
            if total > 0:
                primary_count = max(label_dist.values())
                merged['label_consistency'] = primary_count / total
                merged['primary_label'] = max(label_dist, key=label_dist.get)
        
        used_indices.add(i)
        clustered.append(merged)
    
    print(f"    Done: {comparisons} comparisons, {merges} merges, {len(clustered)} clusters" + " " * 20)
    
    return clustered


# =============================================================================
# TIER ASSIGNMENT
# =============================================================================

def assign_tier(
    stats: Dict,
    existing_aliases: Set[str],
    tier1_min_docs: int = TIER1_MIN_DOCS,
    tier2_min_docs: int = TIER2_MIN_DOCS,
    guardrails: Optional[NERGuardrails] = None,
) -> Tuple[Optional[int], str]:
    """
    Assign tier based on corpus evidence with guardrail checks.
    
    Returns (tier, reason) where:
    - tier 1 = high confidence, auto-accept
    - tier 2 = medium confidence, needs review
    - None = reject (not enough evidence or guardrail block)
    
    Key rules:
    - Single-token PERSON = always tier 2 (surname trap)
    - Single-token ORG/GPE = tier 1 if known acronym OR very high evidence
    - Multi-token with high evidence = tier 1
    - NORP (nationalities) = reject
    """
    doc_count = stats['doc_count']
    consistency = stats['label_consistency']
    surface_norm = stats['surface_norm']
    ner_label = stats.get('primary_label')
    
    # Skip if already in aliases
    if surface_norm in existing_aliases:
        return (None, 'already_in_aliases')
    
    # Use guardrails if provided
    if guardrails:
        # Check if junk
        is_junk, junk_reason = guardrails.is_junk_surface(surface_norm)
        if is_junk:
            return (None, f'guardrail_junk:{junk_reason}')
    
    # Skip single-character or too short
    if len(surface_norm) < 3:
        return (None, 'too_short')
    
    # Skip if mostly digits
    alpha_ratio = sum(c.isalpha() for c in surface_norm) / len(surface_norm)
    if alpha_ratio < 0.5:
        return (None, 'mostly_non_alpha')
    
    # NORP labels (American, Russian, Soviet) are not entities
    if ner_label == 'NORP':
        return (None, f'norp_not_entity:{surface_norm}')
    
    # Tier 1: High frequency + high consistency
    if doc_count >= tier1_min_docs and consistency >= TIER1_MIN_CONSISTENCY:
        # Check with guardrails for safety
        if guardrails:
            can_create, reason = guardrails.can_auto_create_entity(
                surface_norm, ner_label, doc_count, consistency
            )
            if can_create:
                return (1, f'tier1:{reason}')
            else:
                # Downgrade to tier 2 if guardrails block (but don't reject outright)
                # Only reject if it's truly junk (blocklisted, garbage patterns)
                if 'junk' in reason or 'blocklist' in reason:
                    return (None, f'guardrail_reject:{reason}')
                # Single-token with good evidence = tier 2 for review, not rejection
                return (2, f'guardrail_downgrade:{reason}')
        else:
            # No guardrails, use basic rules
            return (1, f'docs={doc_count}, consistency={consistency:.2f}')
    
    # Tier 2: Medium frequency or medium consistency
    if doc_count >= tier2_min_docs and consistency >= TIER2_MIN_CONSISTENCY:
        return (2, f'docs={doc_count}, consistency={consistency:.2f}')
    
    # Reject: Not enough evidence
    return (None, f'insufficient: docs={doc_count}, consistency={consistency:.2f}')


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def save_mentions_batch(
    conn,
    mentions: List[Dict],
    sweep_id: str,
    dry_run: bool = False,
) -> int:
    """Save raw mentions to ner_surface_mentions."""
    if not mentions:
        return 0
    
    if dry_run:
        return len(mentions)
    
    cur = conn.cursor()
    
    records = [
        (
            m['chunk_id'],
            m['document_id'],
            m['surface'],
            m['surface_norm'],
            m['char_start'],
            m['char_end'],
            m['ner_label'],
            m['accept_score'],
            Json(m['context_hints']),
            m['context_window'],
            sweep_id,
        )
        for m in mentions
    ]
    
    execute_values(
        cur,
        """
        INSERT INTO ner_surface_mentions 
            (chunk_id, document_id, surface, surface_norm, char_start, char_end,
             ner_label, ner_accept_score, context_hints, context_window, sweep_id)
        VALUES %s
        ON CONFLICT (chunk_id, char_start, char_end, surface_norm) DO NOTHING
        """,
        records,
        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    
    return len(records)


def save_surface_stats(
    conn,
    stats_list: List[Dict],
    existing_aliases: Set[str],
    sweep_id: str,
    dry_run: bool = False,
    tier1_min_docs: int = TIER1_MIN_DOCS,
    tier2_min_docs: int = TIER2_MIN_DOCS,
    guardrails: Optional[NERGuardrails] = None,
) -> Dict[str, int]:
    """Save aggregated stats to ner_surface_stats with guardrail checks."""
    if not stats_list:
        return {'new': 0, 'matching': 0, 'tier1': 0, 'tier2': 0, 'rejected': 0}
    
    counts = {'new': 0, 'matching': 0, 'tier1': 0, 'tier2': 0, 'rejected': 0}
    records = []
    
    for stats in stats_list:
        surface_norm = stats['surface_norm']
        matches_existing = surface_norm in existing_aliases
        
        # Assign tier (with guardrail checks)
        tier, tier_reason = assign_tier(
            stats, existing_aliases,
            tier1_min_docs=tier1_min_docs,
            tier2_min_docs=tier2_min_docs,
            guardrails=guardrails,
        )
        
        # Track counts
        if matches_existing:
            counts['matching'] += 1
        else:
            counts['new'] += 1
        
        if tier == 1:
            counts['tier1'] += 1
        elif tier == 2:
            counts['tier2'] += 1
        else:
            counts['rejected'] += 1
        
        records.append({
            'surface_norm': surface_norm,
            'doc_count': stats['doc_count'],
            'chunk_count': stats['chunk_count'],
            'mention_count': stats['mention_count'],
            'primary_label': stats['primary_label'],
            'label_distribution': stats['label_distribution'],
            'label_consistency': stats['label_consistency'],
            'inferred_type': stats['inferred_type'],
            'type_confidence': stats['type_confidence'],
            'context_hints_agg': stats['context_hints_agg'],
            'example_contexts': stats['example_contexts'],
            'avg_accept_score': stats['avg_accept_score'],
            'min_accept_score': stats['min_accept_score'],
            'max_accept_score': stats['max_accept_score'],
            'matches_existing_alias': matches_existing,
            'proposed_tier': tier,
            'tier_reason': tier_reason,
            'sweep_id': sweep_id,
        })
    
    if dry_run:
        # Print sample in dry run
        print("\n  Sample surface stats (first 10):")
        for r in sorted(records, key=lambda x: x['doc_count'], reverse=True)[:10]:
            tier_str = f"tier={r['proposed_tier']}" if r['proposed_tier'] else "rejected"
            match_str = "[EXISTING]" if r['matches_existing_alias'] else "[NEW]"
            print(f"    {r['surface_norm']!r}: docs={r['doc_count']}, "
                  f"label={r['primary_label']}, consistency={r['label_consistency']:.2f}, "
                  f"{tier_str} {match_str}")
        return counts
    
    cur = conn.cursor()
    
    db_records = [
        (
            r['surface_norm'],
            r['doc_count'],
            r['chunk_count'],
            r['mention_count'],
            r['primary_label'],
            Json(r['label_distribution']),
            r['label_consistency'],
            r['inferred_type'],
            r['type_confidence'],
            Json(r['context_hints_agg']),
            r['example_contexts'],
            r['avg_accept_score'],
            r['min_accept_score'],
            r['max_accept_score'],
            r['matches_existing_alias'],
            r['proposed_tier'],
            r['tier_reason'],
            'pending' if r['proposed_tier'] else 'rejected',
            r['sweep_id'],
        )
        for r in records
    ]
    
    execute_values(
        cur,
        """
        INSERT INTO ner_surface_stats 
            (surface_norm, doc_count, chunk_count, mention_count,
             primary_label, label_distribution, label_consistency,
             inferred_type, type_confidence,
             context_hints_agg, example_contexts,
             avg_accept_score, min_accept_score, max_accept_score,
             matches_existing_alias, proposed_tier, tier_reason, status,
             corpus_sweep_id)
        VALUES %s
        ON CONFLICT (surface_norm) DO UPDATE SET
            doc_count = EXCLUDED.doc_count,
            chunk_count = EXCLUDED.chunk_count,
            mention_count = EXCLUDED.mention_count,
            primary_label = EXCLUDED.primary_label,
            label_distribution = EXCLUDED.label_distribution,
            label_consistency = EXCLUDED.label_consistency,
            inferred_type = EXCLUDED.inferred_type,
            type_confidence = EXCLUDED.type_confidence,
            context_hints_agg = EXCLUDED.context_hints_agg,
            example_contexts = EXCLUDED.example_contexts,
            avg_accept_score = EXCLUDED.avg_accept_score,
            min_accept_score = EXCLUDED.min_accept_score,
            max_accept_score = EXCLUDED.max_accept_score,
            matches_existing_alias = EXCLUDED.matches_existing_alias,
            proposed_tier = EXCLUDED.proposed_tier,
            tier_reason = EXCLUDED.tier_reason,
            corpus_sweep_id = EXCLUDED.corpus_sweep_id,
            updated_at = NOW()
        """,
        db_records,
        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    
    return counts


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NER Corpus Sweep - Discover new entity surfaces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on 100 chunks
  python scripts/run_ner_corpus_sweep.py --limit 100 --dry-run
  
  # Run on venona collection
  python scripts/run_ner_corpus_sweep.py --collection venona --limit 500
  
  # Full corpus (will take time)
  python scripts/run_ner_corpus_sweep.py --all
        """
    )
    
    # Scope
    parser.add_argument('--collection', type=str, help='Collection to process')
    parser.add_argument('--doc-ids', type=str, help='Comma-separated document IDs')
    parser.add_argument('--all', action='store_true', 
                       help='Process all collections (required if no collection/doc-ids)')
    parser.add_argument('--limit', type=int, help='Limit number of chunks')
    
    # NER settings
    parser.add_argument('--model', type=str, default='en_core_web_lg',
                       help='spaCy model to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='NER acceptance threshold')
    
    # Tier thresholds
    parser.add_argument('--tier1-docs', type=int, default=TIER1_MIN_DOCS,
                       help=f'Min docs for tier 1 (default: {TIER1_MIN_DOCS})')
    parser.add_argument('--tier2-docs', type=int, default=TIER2_MIN_DOCS,
                       help=f'Min docs for tier 2 (default: {TIER2_MIN_DOCS})')
    
    # Execution
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without saving')
    parser.add_argument('--resume', type=str, 
                       help='Resume a previous sweep by sweep_id')
    parser.add_argument('--batch-size', type=int, default=CHUNK_BATCH_SIZE,
                       help=f'Chunks per batch (default: {CHUNK_BATCH_SIZE})')
    parser.add_argument('--save-mentions', action='store_true',
                       help='Save individual mentions (more storage, full provenance)')
    parser.add_argument('--cluster-ocr-variants', action='store_true', default=False,
                       help='Cluster ALL OCR variants (slow for large corpus)')
    parser.add_argument('--cluster-tiered-only', action='store_true', default=True,
                       help='Cluster only tier 1+2 surfaces (much faster, default: True)')
    parser.add_argument('--no-cluster-ocr', action='store_true',
                       help='Disable all OCR variant clustering')
    parser.add_argument('--ocr-similarity', type=float, default=0.85,
                       help='Similarity threshold for OCR clustering (default: 0.85)')
    
    args = parser.parse_args()
    
    # Validate scope
    if not args.collection and not args.doc_ids and not args.all and not args.resume:
        print("ERROR: Specify --collection, --doc-ids, --all, or --resume")
        sys.exit(1)
    
    if not SPACY_AVAILABLE:
        print("ERROR: spaCy not available. Install with: pip install spacy")
        sys.exit(1)
    
    # Setup
    sweep_id = args.resume or str(uuid.uuid4())[:8]
    
    print("=" * 70)
    print("NER CORPUS SWEEP - Entity Surface Discovery")
    print("=" * 70)
    print(f"Sweep ID: {sweep_id}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Tier 1 min docs: {args.tier1_docs}")
    print(f"Tier 2 min docs: {args.tier2_docs}")
    print(f"Save mentions: {args.save_mentions}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    conn = get_conn()
    
    # Check schema
    if not args.dry_run:
        if not check_schema(conn):
            sys.exit(1)
    
    # Load existing aliases for comparison
    print("Loading existing aliases...")
    existing_aliases = get_existing_aliases(conn)
    print(f"  Found {len(existing_aliases)} existing alias_norms")
    
    # Initialize guardrails
    guardrails = NERGuardrails(conn)
    print("Guardrails initialized")
    
    # Get chunks to process
    processed_chunk_ids = None
    if args.resume:
        print(f"Resuming sweep {args.resume}...")
        processed_chunk_ids = get_sweep_processed_chunks(conn, args.resume)
        print(f"  Already processed: {len(processed_chunk_ids)} chunks")
    
    doc_ids = None
    if args.doc_ids:
        doc_ids = [int(x.strip()) for x in args.doc_ids.split(',')]
    
    print("Fetching chunks...")
    chunks = get_chunks_for_sweep(
        conn,
        collection=args.collection,
        doc_ids=doc_ids,
        limit=args.limit,
        processed_chunk_ids=processed_chunk_ids,
    )
    print(f"  Found {len(chunks)} chunks to process")
    
    if not chunks:
        print("No chunks to process.")
        return
    
    # Create sweep record
    if not args.dry_run and not args.resume:
        create_sweep_record(
            conn, sweep_id, args.collection, doc_ids,
            args.model, args.threshold,
            {
                'tier1_min_docs': TIER1_MIN_DOCS,
                'tier2_min_docs': TIER2_MIN_DOCS,
                'save_mentions': args.save_mentions,
            },
            len(chunks),
        )
    
    # Initialize NER
    print("\nLoading spaCy model...")
    extractor = NERExtractor(model_name=args.model)
    print(f"Model loaded: {args.model}")
    
    # Process chunks
    print("\nProcessing chunks...")
    aggregator = SurfaceAggregator()
    
    stats = {
        'chunks_processed': 0,
        'raw_spans': 0,
        'mentions_saved': 0,
    }
    
    batch_texts = []
    batch_chunks = []
    mentions_batch = []
    chunks_since_commit = 0
    
    for i, chunk in enumerate(chunks):
        batch_texts.append(chunk['text'] or '')
        batch_chunks.append(chunk)
        
        # Process batch
        if len(batch_texts) >= args.batch_size or i == len(chunks) - 1:
            try:
                # Extract NER spans
                batch_results = extractor.extract_batch(
                    batch_texts,
                    batch_size=args.batch_size,
                    acceptance_threshold=args.threshold,
                )
                
                # Process each chunk's results
                for chunk_info, spans in zip(batch_chunks, batch_results):
                    chunk_id = chunk_info['chunk_id']
                    doc_id = chunk_info['document_id']
                    text = chunk_info['text'] or ''
                    
                    for span in spans:
                        surface_norm = normalize_surface(span.surface)
                        
                        # Skip very short or empty
                        if len(surface_norm) < 2:
                            continue
                        
                        stats['raw_spans'] += 1
                        
                        # Get context window
                        start = max(0, span.start_char - CONTEXT_WINDOW)
                        end = min(len(text), span.end_char + CONTEXT_WINDOW)
                        context_window = text[start:end]
                        
                        # Add to aggregator
                        aggregator.add_span(
                            span, surface_norm, doc_id, chunk_id, context_window
                        )
                        
                        # Optionally save raw mention
                        if args.save_mentions:
                            mentions_batch.append({
                                'chunk_id': chunk_id,
                                'document_id': doc_id,
                                'surface': span.surface,
                                'surface_norm': surface_norm,
                                'char_start': span.start_char,
                                'char_end': span.end_char,
                                'ner_label': span.label,
                                'accept_score': span.accept_score,
                                'context_hints': span.context_hints,
                                'context_window': context_window,
                            })
                    
                    stats['chunks_processed'] += 1
                    chunks_since_commit += 1
                
                # Save mentions batch
                if args.save_mentions and len(mentions_batch) >= DB_BATCH_SIZE:
                    saved = save_mentions_batch(conn, mentions_batch, sweep_id, args.dry_run)
                    stats['mentions_saved'] += saved
                    mentions_batch = []
                
                # Commit periodically
                if not args.dry_run and chunks_since_commit >= COMMIT_INTERVAL:
                    conn.commit()
                    chunks_since_commit = 0
                
                # Progress
                if stats['chunks_processed'] % 100 == 0:
                    print(f"  Processed {stats['chunks_processed']}/{len(chunks)} chunks, "
                          f"{stats['raw_spans']} spans, "
                          f"{len(aggregator.surfaces)} unique surfaces...")
                    
            except Exception as e:
                if not args.dry_run:
                    conn.rollback()
                print(f"\nERROR at chunk {i}: {e}")
                raise
            
            # Clear batch
            batch_texts = []
            batch_chunks = []
    
    # Save remaining mentions
    if args.save_mentions and mentions_batch:
        saved = save_mentions_batch(conn, mentions_batch, sweep_id, args.dry_run)
        stats['mentions_saved'] += saved
    
    # Aggregate and save surface stats
    print("\nAggregating surface statistics...")
    surface_stats = aggregator.get_stats()
    stats['unique_surfaces_raw'] = len(surface_stats)
    print(f"  Found {len(surface_stats)} unique surfaces (raw)")
    
    # Determine clustering strategy
    cluster_all = args.cluster_ocr_variants and not args.no_cluster_ocr
    cluster_tiered = args.cluster_tiered_only and not args.no_cluster_ocr and not cluster_all
    
    # Option 1: Cluster ALL surfaces upfront (slow for large corpus)
    if cluster_all:
        print(f"\nClustering ALL OCR variants (similarity >= {args.ocr_similarity})...")
        print(f"  Warning: This may be slow for {len(surface_stats)} surfaces")
        surface_stats = cluster_ocr_variants(surface_stats, similarity_threshold=args.ocr_similarity)
        clustered_count = stats['unique_surfaces_raw'] - len(surface_stats)
        print(f"  Merged {clustered_count} OCR variants into {len(surface_stats)} canonical surfaces")
        
        if args.dry_run:
            merged_examples = [s for s in surface_stats if len(s.get('ocr_variants', [])) > 1][:5]
            if merged_examples:
                print("  Example OCR clusters:")
                for s in merged_examples:
                    variants = s.get('ocr_variants', [])
                    print(f"    {s['surface_norm']!r} <- {variants[:5]} (docs={s['doc_count']})")
    
    stats['unique_surfaces'] = len(surface_stats)
    
    # First pass: Assign tiers WITHOUT clustering (to identify tier 1+2 candidates)
    print("\nAssigning tiers (first pass)...")
    tier_counts = save_surface_stats(
        conn, surface_stats, existing_aliases, sweep_id, args.dry_run,
        tier1_min_docs=args.tier1_docs,
        tier2_min_docs=args.tier2_docs,
        guardrails=guardrails,
    )
    
    stats['surfaces_new'] = tier_counts['new']
    stats['surfaces_matching'] = tier_counts['matching']
    stats['tier1_count'] = tier_counts['tier1']
    stats['tier2_count'] = tier_counts['tier2']
    stats['rejected_count'] = tier_counts['rejected']
    
    # Option 2: Cluster only TIERED surfaces (fast - much smaller set)
    if cluster_tiered and (tier_counts['tier1'] + tier_counts['tier2']) > 0:
        tiered_surfaces = [s for s in surface_stats 
                         if s.get('doc_count', 0) >= args.tier2_docs]
        
        print(f"\nClustering tiered candidates only ({len(tiered_surfaces)} surfaces)...")
        
        if len(tiered_surfaces) > 1:
            clustered_tiered = cluster_ocr_variants(tiered_surfaces, similarity_threshold=args.ocr_similarity)
            merged_count = len(tiered_surfaces) - len(clustered_tiered)
            
            if merged_count > 0:
                print(f"  Merged {merged_count} OCR variants among tiered surfaces")
                
                # Update the surface_stats with merged versions
                # (Replace tiered surfaces with clustered versions)
                tiered_norms = {s['surface_norm'] for s in tiered_surfaces}
                non_tiered = [s for s in surface_stats if s['surface_norm'] not in tiered_norms]
                surface_stats = non_tiered + clustered_tiered
                
                # Re-run tier assignment for clustered surfaces
                print("  Re-assigning tiers after clustering...")
                tier_counts = save_surface_stats(
                    conn, clustered_tiered, existing_aliases, sweep_id, args.dry_run,
                    tier1_min_docs=args.tier1_docs,
                    tier2_min_docs=args.tier2_docs,
                    guardrails=guardrails,
                )
                
                # Update stats
                stats['tier1_count'] = tier_counts['tier1']
                stats['tier2_count'] = tier_counts['tier2']
                
                if args.dry_run:
                    merged_examples = [s for s in clustered_tiered if len(s.get('ocr_variants', [])) > 1][:5]
                    if merged_examples:
                        print("  Example OCR clusters:")
                        for s in merged_examples:
                            variants = s.get('ocr_variants', [])
                            print(f"    {s['surface_norm']!r} <- {variants[:5]} (docs={s['doc_count']})")
            else:
                print("  No OCR variants found among tiered surfaces")
    
    # Complete sweep
    if not args.dry_run:
        complete_sweep(conn, sweep_id, stats)
        conn.commit()
    
    # Summary
    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Chunks processed: {stats['chunks_processed']}")
    print(f"Raw NER spans: {stats['raw_spans']}")
    print(f"Unique surfaces discovered: {stats['unique_surfaces']}")
    print()
    print("Surface breakdown:")
    print(f"  Already in aliases: {stats['surfaces_matching']}")
    print(f"  NEW surfaces: {stats['surfaces_new']}")
    print()
    print("Tier distribution (new surfaces):")
    print(f"  Tier 1 (high confidence): {stats['tier1_count']}")
    print(f"  Tier 2 (needs review): {stats['tier2_count']}")
    print(f"  Rejected (low evidence): {stats['rejected_count']}")
    
    if args.save_mentions:
        print(f"\nMentions saved: {stats['mentions_saved']}")
    
    if not args.dry_run:
        print(f"\nSweep ID: {sweep_id}")
        print("\nNext steps:")
        print("  1. Review tier 1 surfaces: SELECT * FROM ner_surface_stats WHERE proposed_tier = 1 ORDER BY doc_count DESC")
        print("  2. Review tier 2 surfaces: SELECT * FROM ner_surface_stats WHERE proposed_tier = 2 ORDER BY doc_count DESC")
        print("  3. Promote to lexicon: python scripts/promote_ner_surfaces.py --tier 1")


if __name__ == '__main__':
    main()
