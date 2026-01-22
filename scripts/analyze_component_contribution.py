#!/usr/bin/env python3
"""
Analyze component contribution in existing retrieval runs.

Analyzes existing retrieval_runs and retrieval_run_chunk_evidence to understand:
- Which components (vector, lexical, soft lex) are driving results
- Whether components are redundant
- Impact of expansion
- Score contribution breakdowns

Usage:
    export DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
    python scripts/analyze_component_contribution.py [--query "silvermaster"] [--limit 5]
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import statistics

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    try:
        import psycopg
        PSYCOPG2_AVAILABLE = False
    except ImportError:
        raise RuntimeError("Neither psycopg2 nor psycopg3 available. Install psycopg2-binary or psycopg")


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    if PSYCOPG2_AVAILABLE:
        return psycopg2.connect(dsn)
    else:
        return psycopg.connect(dsn)


# Golden queries from Phase 7 evaluation
GOLDEN_QUERIES = [
    "silvermaster",
    "silvermastre",
    "hiss",
    "fuchs",
    "Oppenheimer",
]


def analyze_run_component_contribution(conn, run_id: int, rrf_k: int = 50) -> Dict:
    """Analyze component contribution for a single retrieval run."""
    with conn.cursor() as cur:
        # Get run metadata
        cur.execute("""
            SELECT 
                id, query_text, search_type, query_lang_version,
                expand_concordance, expansion_terms,
                retrieval_config_json, top_k
            FROM retrieval_runs
            WHERE id = %s
        """, (run_id,))
        run_row = cur.fetchone()
        if not run_row:
            return None
        
        run_id, query_text, search_type, qlv, expand_concordance, expansion_terms, config_json, top_k = run_row
        
        # Get RRF k from config or default
        if config_json and isinstance(config_json, dict):
            rrf_k = config_json.get("rrf_k", rrf_k)
        
        # Get evidence rows
        cur.execute("""
            SELECT 
                chunk_id, rank,
                score_lex, score_vec, score_hybrid,
                matched_lexemes, explain_json
            FROM retrieval_run_chunk_evidence
            WHERE retrieval_run_id = %s
            ORDER BY rank ASC
        """, (run_id,))
        
        evidence_rows = cur.fetchall()
        
        if not evidence_rows:
            return {
                "run_id": run_id,
                "query_text": query_text,
                "search_type": search_type,
                "query_lang_version": qlv,
                "expand_concordance": expand_concordance,
                "expansion_terms": expansion_terms,
                "top_k": top_k,
                "num_results": 0,
                "error": "No evidence rows found",
            }
        
        # Extract component ranks from explain_json
        vector_ranks = []
        lexical_ranks = []
        soft_lex_ranks = []
        vector_scores = []
        lexical_scores = []
        soft_lex_scores = []
        hybrid_scores = []
        
        chunks_with_vector = []
        chunks_with_lexical = []
        chunks_with_soft_lex = []
        chunks_with_both = []
        
        for chunk_id, rank, score_lex, score_vec, score_hybrid, matched_lexemes, explain_json in evidence_rows:
            r_vec = None
            r_lex = None
            r_soft_lex = None
            soft_lex_score_val = None
            
            if explain_json:
                if isinstance(explain_json, str):
                    import json
                    try:
                        explain_json = json.loads(explain_json)
                    except:
                        pass
                
                if isinstance(explain_json, dict):
                    # Extract ranks
                    if "semantic" in explain_json:
                        r_vec = explain_json["semantic"].get("r_vec")
                    if "lex" in explain_json:
                        r_lex = explain_json["lex"].get("r_lex")
                    if "approx_lex" in explain_json:
                        r_soft_lex = explain_json["approx_lex"].get("r_soft_lex")
                        soft_lex_score_val = explain_json["approx_lex"].get("score")
            
            # Compute component contributions
            vec_contrib = 0.0
            lex_contrib = 0.0
            soft_lex_contrib = 0.0
            
            if r_vec is not None:
                vec_contrib = 1.0 / (rrf_k + r_vec)
                vector_ranks.append(r_vec)
                vector_scores.append(vec_contrib)
                chunks_with_vector.append(chunk_id)
            
            if r_lex is not None:
                lex_contrib = 1.0 / (rrf_k + r_lex)
                lexical_ranks.append(r_lex)
                lexical_scores.append(lex_contrib)
                chunks_with_lexical.append(chunk_id)
            
            if r_soft_lex is not None:
                # Get soft lex weight from config
                soft_lex_weight = 0.5
                if config_json and isinstance(config_json, dict):
                    soft_lex_weight = config_json.get("soft_lex_weight", 0.5)
                soft_lex_contrib = soft_lex_weight * (1.0 / (rrf_k + r_soft_lex))
                soft_lex_ranks.append(r_soft_lex)
                soft_lex_scores.append(soft_lex_contrib)
                chunks_with_soft_lex.append(chunk_id)
            
            # Check if chunk has both vector and lexical
            if r_vec is not None and r_lex is not None:
                chunks_with_both.append(chunk_id)
            
            hybrid_scores.append(score_hybrid or 0.0)
        
        # Compute statistics
        total_score = sum(hybrid_scores) if hybrid_scores else 1.0
        vector_total = sum(vector_scores) if vector_scores else 0.0
        lexical_total = sum(lexical_scores) if lexical_scores else 0.0
        soft_lex_total = sum(soft_lex_scores) if soft_lex_scores else 0.0
        
        vector_pct = (vector_total / total_score * 100) if total_score > 0 else 0.0
        lexical_pct = (lexical_total / total_score * 100) if total_score > 0 else 0.0
        soft_lex_pct = (soft_lex_total / total_score * 100) if total_score > 0 else 0.0
        
        # Component coverage
        vector_only = set(chunks_with_vector) - set(chunks_with_lexical) - set(chunks_with_soft_lex)
        lexical_only = set(chunks_with_lexical) - set(chunks_with_vector) - set(chunks_with_soft_lex)
        soft_lex_only = set(chunks_with_soft_lex) - set(chunks_with_vector) - set(chunks_with_lexical)
        
        return {
            "run_id": run_id,
            "query_text": query_text,
            "search_type": search_type,
            "query_lang_version": qlv,
            "expand_concordance": expand_concordance,
            "expansion_terms": expansion_terms,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "num_results": len(evidence_rows),
            
            # Component coverage
            "chunks_with_vector": len(chunks_with_vector),
            "chunks_with_lexical": len(chunks_with_lexical),
            "chunks_with_soft_lex": len(chunks_with_soft_lex),
            "chunks_with_both": len(chunks_with_both),
            "vector_only_count": len(vector_only),
            "lexical_only_count": len(lexical_only),
            "soft_lex_only_count": len(soft_lex_only),
            
            # Score contribution
            "vector_contribution_pct": vector_pct,
            "lexical_contribution_pct": lexical_pct,
            "soft_lex_contribution_pct": soft_lex_pct,
            
            # Rank statistics
            "avg_vector_rank": statistics.mean(vector_ranks) if vector_ranks else None,
            "avg_lexical_rank": statistics.mean(lexical_ranks) if lexical_ranks else None,
            "avg_soft_lex_rank": statistics.mean(soft_lex_ranks) if soft_lex_ranks else None,
            
            # Component sets for overlap analysis
            "vector_chunks": set(chunks_with_vector),
            "lexical_chunks": set(chunks_with_lexical),
            "soft_lex_chunks": set(chunks_with_soft_lex),
        }


def compare_runs(conn, query_text: str, limit: int = 5) -> List[Dict]:
    """Compare multiple runs for the same query."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id
            FROM retrieval_runs
            WHERE query_text = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (query_text, limit))
        
        run_ids = [row[0] for row in cur.fetchall()]
    
    results = []
    for run_id in run_ids:
        analysis = analyze_run_component_contribution(conn, run_id)
        if analysis:
            results.append(analysis)
    
    return results


def compute_overlap(set1: Set, set2: Set) -> float:
    """Compute overlap between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_rank_correlation(ranks1: List[int], ranks2: List[int], chunk_ids1: List[int], chunk_ids2: List[int]) -> Optional[float]:
    """Compute rank correlation for chunks that appear in both."""
    if not ranks1 or not ranks2:
        return None
    
    # Find common chunks
    common_chunks = set(chunk_ids1) & set(chunk_ids2)
    if not common_chunks:
        return None
    
    # Get ranks for common chunks
    rank_map1 = {chunk_id: rank for chunk_id, rank in zip(chunk_ids1, ranks1)}
    rank_map2 = {chunk_id: rank for chunk_id, rank in zip(chunk_ids2, ranks2)}
    
    common_ranks1 = [rank_map1[cid] for cid in common_chunks if cid in rank_map1]
    common_ranks2 = [rank_map2[cid] for cid in common_chunks if cid in rank_map2]
    
    if len(common_ranks1) < 2:
        return None
    
    # Simple correlation: 1 - (sum of squared differences / normalization)
    # Normalized to 0-1 range
    diff_sum = sum((r1 - r2) ** 2 for r1, r2 in zip(common_ranks1, common_ranks2))
    max_diff = len(common_ranks1) * (max(common_ranks1 + common_ranks2) ** 2)
    correlation = 1.0 - (diff_sum / max_diff) if max_diff > 0 else 1.0
    
    return correlation


def print_analysis_report(analyses: List[Dict], query_text: str):
    """Print formatted analysis report."""
    print("\n" + "=" * 100)
    print(f"COMPONENT CONTRIBUTION ANALYSIS: {query_text!r}")
    print("=" * 100)
    
    if not analyses:
        print("\n  No runs found for this query")
        return
    
    for i, analysis in enumerate(analyses, 1):
        print(f"\n{'─' * 100}")
        print(f"Run #{i} (ID: {analysis['run_id']})")
        print(f"{'─' * 100}")
        print(f"  Query Language Version: {analysis['query_lang_version']}")
        print(f"  Search Type: {analysis['search_type']}")
        print(f"  Expansion: {'Yes' if analysis['expand_concordance'] else 'No'}")
        if analysis['expansion_terms']:
            print(f"  Expansion Terms: {len(analysis['expansion_terms'])} terms")
        print(f"  Results: {analysis['num_results']} chunks")
        print(f"  RRF k: {analysis['rrf_k']}")
        
        if analysis.get('error'):
            print(f"  ERROR: {analysis['error']}")
            continue
        
        # Component coverage
        print(f"\n  Component Coverage:")
        print(f"    Vector:      {analysis['chunks_with_vector']} chunks")
        print(f"    Lexical:     {analysis['chunks_with_lexical']} chunks")
        print(f"    Soft Lex:    {analysis['chunks_with_soft_lex']} chunks")
        print(f"    Both (V+L):  {analysis['chunks_with_both']} chunks")
        print(f"    Vector-only: {analysis['vector_only_count']} chunks")
        print(f"    Lexical-only: {analysis['lexical_only_count']} chunks")
        print(f"    Soft Lex-only: {analysis['soft_lex_only_count']} chunks")
        
        # Score contribution
        print(f"\n  Score Contribution:")
        print(f"    Vector:      {analysis['vector_contribution_pct']:.1f}%")
        print(f"    Lexical:     {analysis['lexical_contribution_pct']:.1f}%")
        print(f"    Soft Lex:    {analysis['soft_lex_contribution_pct']:.1f}%")
        
        # Rank statistics
        if analysis['avg_vector_rank'] is not None:
            print(f"\n  Average Ranks:")
            print(f"    Vector:      {analysis['avg_vector_rank']:.1f}")
        if analysis['avg_lexical_rank'] is not None:
            print(f"    Lexical:     {analysis['avg_lexical_rank']:.1f}")
        if analysis['avg_soft_lex_rank'] is not None:
            print(f"    Soft Lex:    {analysis['avg_soft_lex_rank']:.1f}")
    
    # Cross-run comparison
    if len(analyses) > 1:
        print(f"\n{'─' * 100}")
        print("Cross-Run Comparison")
        print(f"{'─' * 100}")
        
        # Compare qv1 vs qv2 if both exist
        qv1_runs = [a for a in analyses if a['query_lang_version'] == 'qv1']
        qv2_runs = [a for a in analyses if a['query_lang_version'] == 'qv2_softlex']
        
        if qv1_runs and qv2_runs:
            qv1 = qv1_runs[0]
            qv2 = qv2_runs[0]
            
            print(f"\n  Comparing qv1 (Run {qv1['run_id']}) vs qv2_softlex (Run {qv2['run_id']}):")
            
            # Overlap
            vector_overlap = compute_overlap(qv1['vector_chunks'], qv2['vector_chunks'])
            lexical_overlap = compute_overlap(qv1['lexical_chunks'], qv2['lexical_chunks'])
            all_chunks_overlap = compute_overlap(
                qv1['vector_chunks'] | qv1['lexical_chunks'],
                qv2['vector_chunks'] | qv2['lexical_chunks'] | qv2['soft_lex_chunks']
            )
            
            print(f"\n  Overlap:")
            print(f"    Vector chunks:  {vector_overlap:.3f}")
            print(f"    Lexical chunks: {lexical_overlap:.3f}")
            print(f"    All chunks:     {all_chunks_overlap:.3f}")
            
            # Contribution comparison
            print(f"\n  Score Contribution Comparison:")
            print(f"    Vector:      qv1={qv1['vector_contribution_pct']:.1f}%  qv2={qv2['vector_contribution_pct']:.1f}%")
            print(f"    Lexical:     qv1={qv1['lexical_contribution_pct']:.1f}%  qv2={qv2['lexical_contribution_pct']:.1f}%")
            print(f"    Soft Lex:    qv1={qv1['soft_lex_contribution_pct']:.1f}%  qv2={qv2['soft_lex_contribution_pct']:.1f}%")
            
            # New chunks in qv2
            qv1_all = qv1['vector_chunks'] | qv1['lexical_chunks']
            qv2_all = qv2['vector_chunks'] | qv2['lexical_chunks'] | qv2['soft_lex_chunks']
            new_in_qv2 = qv2_all - qv1_all
            lost_in_qv2 = qv1_all - qv2_all
            
            print(f"\n  Result Differences:")
            print(f"    New in qv2:  {len(new_in_qv2)} chunks")
            print(f"    Lost in qv2: {len(lost_in_qv2)} chunks")
            
            if new_in_qv2:
                print(f"    New chunk IDs: {list(new_in_qv2)[:10]}")
            if lost_in_qv2:
                print(f"    Lost chunk IDs: {list(lost_in_qv2)[:10]}")


def main():
    ap = argparse.ArgumentParser(description="Analyze component contribution in retrieval runs")
    ap.add_argument("--query", type=str, default=None, help="Analyze specific query (default: all golden queries)")
    ap.add_argument("--limit", type=int, default=5, help="Max runs per query to analyze (default: 5)")
    args = ap.parse_args()
    
    conn = get_conn()
    
    try:
        print("=" * 100)
        print("COMPONENT CONTRIBUTION ANALYSIS")
        print("=" * 100)
        print("\nAnalyzing existing retrieval runs to understand component contributions...")
        
        queries_to_analyze = [args.query] if args.query else GOLDEN_QUERIES
        
        all_results = {}
        
        for query_text in queries_to_analyze:
            analyses = compare_runs(conn, query_text, limit=args.limit)
            all_results[query_text] = analyses
            print_analysis_report(analyses, query_text)
        
        # Summary across all queries
        print("\n" + "=" * 100)
        print("SUMMARY ACROSS ALL QUERIES")
        print("=" * 100)
        
        # Aggregate statistics
        total_runs = sum(len(analyses) for analyses in all_results.values())
        qv1_runs = []
        qv2_runs = []
        
        for analyses in all_results.values():
            for analysis in analyses:
                if analysis.get('error'):
                    continue
                if analysis['query_lang_version'] == 'qv1':
                    qv1_runs.append(analysis)
                elif analysis['query_lang_version'] == 'qv2_softlex':
                    qv2_runs.append(analysis)
        
        if qv1_runs:
            avg_vector_pct_qv1 = statistics.mean([r['vector_contribution_pct'] for r in qv1_runs])
            avg_lexical_pct_qv1 = statistics.mean([r['lexical_contribution_pct'] for r in qv1_runs])
            print(f"\n  qv1 (exact FTS) - Average across {len(qv1_runs)} runs:")
            print(f"    Vector contribution:  {avg_vector_pct_qv1:.1f}%")
            print(f"    Lexical contribution: {avg_lexical_pct_qv1:.1f}%")
        
        if qv2_runs:
            avg_vector_pct_qv2 = statistics.mean([r['vector_contribution_pct'] for r in qv2_runs])
            avg_lexical_pct_qv2 = statistics.mean([r['lexical_contribution_pct'] for r in qv2_runs])
            avg_soft_lex_pct_qv2 = statistics.mean([r['soft_lex_contribution_pct'] for r in qv2_runs])
            print(f"\n  qv2_softlex - Average across {len(qv2_runs)} runs:")
            print(f"    Vector contribution:  {avg_vector_pct_qv2:.1f}%")
            print(f"    Lexical contribution: {avg_lexical_pct_qv2:.1f}%")
            print(f"    Soft Lex contribution: {avg_soft_lex_pct_qv2:.1f}%")
        
        # Key findings
        print("\n" + "=" * 100)
        print("KEY FINDINGS")
        print("=" * 100)
        
        if qv1_runs:
            if avg_vector_pct_qv1 > 80:
                print(f"\n  ⚠️  Vector dominates qv1 results ({avg_vector_pct_qv1:.1f}% contribution)")
                print("     → Lexical component may be redundant")
            elif avg_lexical_pct_qv1 > 80:
                print(f"\n  ⚠️  Lexical dominates qv1 results ({avg_lexical_pct_qv1:.1f}% contribution)")
                print("     → Vector component may be redundant")
            else:
                print(f"\n  ✓ Balanced contribution in qv1")
                print(f"     → Vector: {avg_vector_pct_qv1:.1f}%, Lexical: {avg_lexical_pct_qv1:.1f}%")
        
        if qv2_runs:
            if avg_soft_lex_pct_qv2 < 1.0:
                print(f"\n  ⚠️  Soft Lex contribution is minimal ({avg_soft_lex_pct_qv2:.1f}%)")
                print("     → Soft lex may not be contributing meaningfully")
            else:
                print(f"\n  ✓ Soft Lex is contributing ({avg_soft_lex_pct_qv2:.1f}%)")
        
        if qv1_runs and qv2_runs:
            if abs(avg_vector_pct_qv1 - avg_vector_pct_qv2) < 5:
                print(f"\n  ⚠️  Vector contribution similar in qv1 and qv2")
                print("     → Soft lex may not be changing vector behavior")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
