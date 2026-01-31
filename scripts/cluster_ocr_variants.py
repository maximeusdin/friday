#!/usr/bin/env python3
"""
OCR Variant Clustering (Phase 4)

Builds alias similarity graph and clusters variants to generate merge proposals.

Usage:
    python scripts/cluster_ocr_variants.py --source queued --min-mentions 2
    python scripts/cluster_ocr_variants.py --source all --threshold 0.35
"""

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import psycopg2
from psycopg2.extras import execute_values, Json

sys.path.insert(0, '.')
from retrieval.surface_norm import normalize_surface
from retrieval.ocr_utils import (
    OCRConfusionTable, get_confusion_table,
    normalized_weighted_edit_distance,
    compute_variant_key, compute_priority_score,
    VariantClusterer, are_variants_similar
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_THRESHOLD = 0.30       # OCR distance threshold for clustering
MIN_CLUSTER_SIZE = 2           # Minimum variants to form a cluster
MIN_MENTIONS_FOR_CLUSTERING = 2  # Minimum mentions to consider a variant


@dataclass
class VariantInfo:
    """Information about a variant."""
    variant_key: str
    surface_norm: str
    raw_examples: List[str]
    mention_count: int
    doc_count: int
    avg_quality: float
    current_entity_id: Optional[int]
    example_citations: List[dict]


@dataclass
class ClusterInfo:
    """Information about a cluster."""
    cluster_id: str
    variants: List[VariantInfo]
    proposed_canonical: Optional[str]
    canonical_entity_id: Optional[int]
    canonical_source: Optional[str]
    total_mentions: int
    doc_count: int
    maps_to_multiple_entities: bool
    has_type_conflict: bool
    recommendation: str
    priority_score: float


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def get_conn():
    return psycopg2.connect(
        host='localhost', port=5432, dbname='neh', user='neh', password='neh'
    )


def get_variants_from_queue(conn, min_mentions: int = 2) -> List[VariantInfo]:
    """Get variants from the review queue."""
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            surface_norm,
            array_agg(DISTINCT surface) as raw_examples,
            COUNT(*) as mention_count,
            COUNT(DISTINCT document_id) as doc_count,
            AVG(COALESCE((resolution_signals->>'quality')::numeric, 0.5)) as avg_quality,
            array_agg(DISTINCT jsonb_build_object(
                'doc_id', document_id,
                'chunk_id', chunk_id,
                'surface', surface
            )) as citations
        FROM mention_review_queue
        WHERE status = 'pending'
        GROUP BY surface_norm
        HAVING COUNT(*) >= %s
        ORDER BY COUNT(*) DESC
    """, (min_mentions,))
    
    variants = []
    for row in cur.fetchall():
        surface_norm, raw_examples, mention_count, doc_count, avg_quality, citations = row
        variant_key = compute_variant_key(surface_norm)
        
        variants.append(VariantInfo(
            variant_key=variant_key,
            surface_norm=surface_norm,
            raw_examples=list(raw_examples)[:5] if raw_examples else [],
            mention_count=mention_count,
            doc_count=doc_count,
            avg_quality=float(avg_quality) if avg_quality else 0.5,
            current_entity_id=None,
            example_citations=citations[:3] if citations else []
        ))
    
    return variants


def get_variants_from_candidates(conn, min_mentions: int = 2, include_ignored: bool = False) -> List[VariantInfo]:
    """Get variants from mention_candidates."""
    cur = conn.cursor()
    
    status_filter = "IN ('queue', 'ignore')" if include_ignored else "= 'queue'"
    
    cur.execute(f"""
        SELECT 
            surface_norm,
            array_agg(DISTINCT raw_span) as raw_examples,
            COUNT(*) as mention_count,
            COUNT(DISTINCT document_id) as doc_count,
            AVG(quality_score) as avg_quality,
            resolved_entity_id,
            array_agg(DISTINCT jsonb_build_object(
                'doc_id', document_id,
                'chunk_id', chunk_id,
                'surface', raw_span
            )) as citations
        FROM mention_candidates
        WHERE resolution_status {status_filter}
        GROUP BY surface_norm, resolved_entity_id
        HAVING COUNT(*) >= %s
        ORDER BY COUNT(*) DESC
    """, (min_mentions,))
    
    variants = []
    for row in cur.fetchall():
        surface_norm, raw_examples, mention_count, doc_count, avg_quality, entity_id, citations = row
        variant_key = compute_variant_key(surface_norm)
        
        variants.append(VariantInfo(
            variant_key=variant_key,
            surface_norm=surface_norm,
            raw_examples=list(raw_examples)[:5] if raw_examples else [],
            mention_count=mention_count,
            doc_count=doc_count,
            avg_quality=float(avg_quality) if avg_quality else 0.5,
            current_entity_id=entity_id,
            example_citations=citations[:3] if citations else []
        ))
    
    return variants


def find_matching_entity(conn, surface_norm: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Find the best matching entity for a surface.
    
    Returns: (entity_id, canonical_name, source)
    """
    cur = conn.cursor()
    
    # First check exact match in alias_lexicon_index
    cur.execute("""
        SELECT entity_id, alias_norm, proposal_tier
        FROM alias_lexicon_index
        WHERE alias_norm = %s
        ORDER BY proposal_tier NULLS LAST, doc_freq DESC
        LIMIT 1
    """, (surface_norm,))
    
    row = cur.fetchone()
    if row:
        entity_id = row[0]
        cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
        name_row = cur.fetchone()
        return entity_id, name_row[0] if name_row else None, 'exact_alias'
    
    # Check entity_surface_stats (high confidence matches)
    cur.execute("""
        SELECT entity_id, surface_display
        FROM entity_surface_stats
        WHERE surface_norm = %s
        ORDER BY doc_freq DESC
        LIMIT 1
    """, (surface_norm,))
    
    row = cur.fetchone()
    if row:
        entity_id = row[0]
        cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
        name_row = cur.fetchone()
        return entity_id, name_row[0] if name_row else None, 'corpus_stats'
    
    return None, None, None


def get_entity_type(conn, entity_id: int) -> Optional[str]:
    """Get entity type."""
    cur = conn.cursor()
    cur.execute("SELECT entity_type FROM entities WHERE id = %s", (entity_id,))
    row = cur.fetchone()
    return row[0] if row else None


def compute_cluster_id(variants: List[VariantInfo]) -> str:
    """Compute stable cluster ID from variant keys."""
    sorted_keys = sorted(v.variant_key for v in variants)
    hash_input = '|'.join(sorted_keys)
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def analyze_cluster(
    conn,
    variants: List[VariantInfo],
    confusion_table: OCRConfusionTable
) -> ClusterInfo:
    """Analyze a cluster and generate recommendations."""
    
    cluster_id = compute_cluster_id(variants)
    
    # Aggregate stats
    total_mentions = sum(v.mention_count for v in variants)
    doc_ids = set()
    for v in variants:
        for cit in v.example_citations:
            if isinstance(cit, dict) and 'doc_id' in cit:
                doc_ids.add(cit['doc_id'])
    doc_count = len(doc_ids)
    
    avg_quality = sum(v.avg_quality * v.mention_count for v in variants) / total_mentions if total_mentions > 0 else 0.5
    
    # Find current entity mappings
    entity_ids = set(v.current_entity_id for v in variants if v.current_entity_id)
    maps_to_multiple = len(entity_ids) > 1
    
    # Check type conflicts
    has_type_conflict = False
    if entity_ids:
        types = set()
        for eid in entity_ids:
            etype = get_entity_type(conn, eid)
            if etype:
                types.add(etype)
        has_type_conflict = len(types) > 1
    
    # Find proposed canonical
    proposed_canonical = None
    canonical_entity_id = None
    canonical_source = None
    
    # Try to find matching entity for most frequent variant
    most_frequent = max(variants, key=lambda v: v.mention_count)
    entity_id, canonical_name, source = find_matching_entity(conn, most_frequent.surface_norm)
    
    if entity_id:
        proposed_canonical = canonical_name
        canonical_entity_id = entity_id
        canonical_source = source
    else:
        # Use most frequent surface as proposed
        proposed_canonical = most_frequent.surface_norm
        canonical_source = 'frequency'
    
    # Determine recommendation
    if maps_to_multiple or has_type_conflict:
        recommendation = 'NEEDS_REVIEW'
    elif canonical_entity_id and total_mentions >= 5:
        recommendation = 'SAFE_ADD'
    elif canonical_entity_id:
        recommendation = 'NEEDS_REVIEW'
    else:
        recommendation = 'NEEDS_REVIEW'
    
    # Compute priority
    priority = compute_priority_score(
        doc_count=doc_count,
        mention_count=total_mentions,
        avg_quality=avg_quality,
        has_tier1_match=canonical_source == 'exact_alias',
        has_danger_flags=maps_to_multiple or has_type_conflict
    )
    
    return ClusterInfo(
        cluster_id=cluster_id,
        variants=variants,
        proposed_canonical=proposed_canonical,
        canonical_entity_id=canonical_entity_id,
        canonical_source=canonical_source,
        total_mentions=total_mentions,
        doc_count=doc_count,
        maps_to_multiple_entities=maps_to_multiple,
        has_type_conflict=has_type_conflict,
        recommendation=recommendation,
        priority_score=priority
    )


def save_cluster(conn, cluster: ClusterInfo):
    """Save cluster to database."""
    cur = conn.cursor()
    
    # Insert/update cluster
    cur.execute("""
        INSERT INTO ocr_variant_clusters (
            cluster_id, proposed_canonical, canonical_entity_id, canonical_source,
            variant_count, total_mentions, doc_count,
            maps_to_multiple_entities, has_type_conflict,
            recommendation, priority_score, status, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', NOW())
        ON CONFLICT (cluster_id) DO UPDATE SET
            proposed_canonical = EXCLUDED.proposed_canonical,
            canonical_entity_id = EXCLUDED.canonical_entity_id,
            canonical_source = EXCLUDED.canonical_source,
            variant_count = EXCLUDED.variant_count,
            total_mentions = EXCLUDED.total_mentions,
            doc_count = EXCLUDED.doc_count,
            maps_to_multiple_entities = EXCLUDED.maps_to_multiple_entities,
            has_type_conflict = EXCLUDED.has_type_conflict,
            recommendation = EXCLUDED.recommendation,
            priority_score = EXCLUDED.priority_score,
            updated_at = NOW()
    """, (
        cluster.cluster_id,
        cluster.proposed_canonical,
        cluster.canonical_entity_id,
        cluster.canonical_source,
        len(cluster.variants),
        cluster.total_mentions,
        cluster.doc_count,
        cluster.maps_to_multiple_entities,
        cluster.has_type_conflict,
        cluster.recommendation,
        cluster.priority_score
    ))
    
    # Insert/update variants
    for v in cluster.variants:
        cur.execute("""
            INSERT INTO ocr_cluster_variants (
                cluster_id, variant_key, raw_examples,
                mention_count, doc_count, avg_quality_score,
                current_entity_id, example_citations
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cluster_id, variant_key) DO UPDATE SET
                raw_examples = EXCLUDED.raw_examples,
                mention_count = EXCLUDED.mention_count,
                doc_count = EXCLUDED.doc_count,
                avg_quality_score = EXCLUDED.avg_quality_score,
                current_entity_id = EXCLUDED.current_entity_id,
                example_citations = EXCLUDED.example_citations
        """, (
            cluster.cluster_id,
            v.variant_key,
            v.raw_examples,
            v.mention_count,
            v.doc_count,
            v.avg_quality,
            v.current_entity_id,
            Json(v.example_citations)
        ))
    
    conn.commit()


# =============================================================================
# MAIN CLUSTERING LOGIC
# =============================================================================

def build_variant_clusters(
    variants: List[VariantInfo],
    confusion_table: OCRConfusionTable,
    threshold: float = DEFAULT_THRESHOLD
) -> List[List[VariantInfo]]:
    """Cluster variants by OCR similarity."""
    
    if not variants:
        return []
    
    # Build variant lookup by key
    by_key = {v.variant_key: v for v in variants}
    
    # Use clusterer
    clusterer = VariantClusterer(confusion_table, threshold)
    for v in variants:
        clusterer.add_variant(v.surface_norm)
    
    raw_clusters = clusterer.cluster()
    
    # Convert back to VariantInfo clusters
    result = []
    for root, surfaces in raw_clusters.items():
        cluster_variants = []
        for surface in surfaces:
            key = compute_variant_key(surface)
            if key in by_key:
                cluster_variants.append(by_key[key])
        
        if len(cluster_variants) >= MIN_CLUSTER_SIZE:
            result.append(cluster_variants)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Cluster OCR variants')
    parser.add_argument('--source', choices=['queued', 'candidates', 'all'], default='queued',
                       help='Source of variants')
    parser.add_argument('--min-mentions', type=int, default=MIN_MENTIONS_FOR_CLUSTERING,
                       help='Minimum mentions per variant')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                       help='OCR distance threshold for clustering')
    parser.add_argument('--include-ignored', action='store_true',
                       help='Include ignored candidates')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t save clusters')
    parser.add_argument('--limit', type=int, help='Limit number of clusters to save')
    args = parser.parse_args()
    
    conn = get_conn()
    
    print("=== OCR Variant Clustering ===")
    print(f"Source: {args.source}")
    print(f"Min mentions: {args.min_mentions}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Load confusion table
    confusion_table = get_confusion_table(conn)
    print(f"Loaded {len(confusion_table.confusions)} confusion patterns")
    
    # Get variants
    print("\nLoading variants...")
    variants = []
    
    if args.source in ('queued', 'all'):
        queue_variants = get_variants_from_queue(conn, args.min_mentions)
        print(f"  From queue: {len(queue_variants)} variants")
        variants.extend(queue_variants)
    
    if args.source in ('candidates', 'all'):
        cand_variants = get_variants_from_candidates(conn, args.min_mentions, args.include_ignored)
        print(f"  From candidates: {len(cand_variants)} variants")
        variants.extend(cand_variants)
    
    # Dedupe by variant_key
    by_key = {}
    for v in variants:
        if v.variant_key not in by_key or v.mention_count > by_key[v.variant_key].mention_count:
            by_key[v.variant_key] = v
    variants = list(by_key.values())
    
    print(f"\nTotal unique variants: {len(variants)}")
    
    if len(variants) < 2:
        print("Not enough variants to cluster.")
        return
    
    # Cluster
    print("\nClustering...")
    start_time = time.time()
    
    clusters_raw = build_variant_clusters(variants, confusion_table, args.threshold)
    
    elapsed = time.time() - start_time
    print(f"  Found {len(clusters_raw)} clusters in {elapsed:.1f}s")
    
    # Analyze clusters
    print("\nAnalyzing clusters...")
    clusters = []
    for cluster_variants in clusters_raw:
        cluster = analyze_cluster(conn, cluster_variants, confusion_table)
        clusters.append(cluster)
    
    # Sort by priority
    clusters.sort(key=lambda c: -c.priority_score)
    
    if args.limit:
        clusters = clusters[:args.limit]
    
    # Report
    print()
    print("=== CLUSTERING RESULTS ===")
    print(f"  Total clusters: {len(clusters)}")
    
    by_rec = defaultdict(int)
    for c in clusters:
        by_rec[c.recommendation] += 1
    
    print("  By recommendation:")
    for rec, count in sorted(by_rec.items()):
        print(f"    {rec}: {count}")
    
    danger_count = sum(1 for c in clusters if c.maps_to_multiple_entities or c.has_type_conflict)
    print(f"  With danger flags: {danger_count}")
    
    total_variants = sum(len(c.variants) for c in clusters)
    total_mentions = sum(c.total_mentions for c in clusters)
    print(f"  Total variants covered: {total_variants}")
    print(f"  Total mentions covered: {total_mentions}")
    
    # Show top clusters
    print("\nTop 10 clusters by priority:")
    for i, c in enumerate(clusters[:10]):
        flags = []
        if c.maps_to_multiple_entities:
            flags.append("MULTI_ENTITY")
        if c.has_type_conflict:
            flags.append("TYPE_CONFLICT")
        flags_str = f" [{', '.join(flags)}]" if flags else ""
        
        print(f"  {i+1}. {c.proposed_canonical or '(unknown)'} - {c.recommendation}")
        print(f"      {len(c.variants)} variants, {c.total_mentions} mentions, {c.doc_count} docs{flags_str}")
        print(f"      Priority: {c.priority_score:.1f}")
    
    # Save
    if not args.dry_run:
        print("\nSaving clusters...")
        for c in clusters:
            save_cluster(conn, c)
        print(f"  Saved {len(clusters)} clusters")
    else:
        print("\n[DRY RUN] Clusters not saved.")
    
    conn.close()


if __name__ == '__main__':
    main()
