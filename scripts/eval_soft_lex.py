#!/usr/bin/env python3
"""
Phase 7: Evaluation & Safety Harness for Soft Lexical Retrieval

Evaluates recall/precision improvements of qv2_softlex vs qv1_exact.

Golden queries (known to fail under OCR):
- silvermaster (OCR: silvermastre, silverrnaster)
- hiss (OCR: hiss variants)
- fuchs (OCR: fuchs variants)

Usage:
    export DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
    python scripts/eval_soft_lex.py [--k 10] [--collection venona] [--save]
"""

import os
import sys
import argparse
from typing import List, Dict, Set, Any, Tuple

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.ops import (
    get_conn,
    hybrid_rrf,
    SearchFilters,
)
import psycopg2
from psycopg2.extras import Json


# -----------------------
# Golden Queries
# -----------------------

GOLDEN_QUERIES = [
    {
        "query": "silvermaster",
        "description": "Silvermaster (OCR errors: silvermastre, silverrnaster)",
        "collection": "silvermaster",
        "chunk_pv": "chunk_v1_silvermaster_structured_4k",
        "expected_improvement": "high",  # Should see significant recall improvement
    },
    {
        "query": "silvermastre",  # User typo / OCR variant
        "description": "silvermastre (typo - missing 'r')",
        "collection": "silvermaster",
        "chunk_pv": "chunk_v1_silvermaster_structured_4k",
        "expected_improvement": "high",
    },
    {
        "query": "hiss",
        "description": "Hiss (name with OCR variants)",
        "collection": "venona",
        "chunk_pv": "chunk_v1_full",
        "expected_improvement": "medium",
    },
    {
        "query": "fuchs",
        "description": "Fuchs (name with OCR variants)",
        "collection": "venona",
        "chunk_pv": "chunk_v1_full",
        "expected_improvement": "medium",
    },
    {
        "query": "Oppenheimer",
        "description": "Oppenheimer (baseline - should work with both)",
        "collection": "venona",
        "chunk_pv": "chunk_v1_full",
        "expected_improvement": "low",  # Should work with exact match too
    },
]


# -----------------------
# Evaluation Metrics
# -----------------------

def compute_recall_at_k(relevant_chunks: Set[int], retrieved_chunks: List[int], k: int) -> float:
    """Compute recall@k: fraction of relevant chunks found in top k."""
    if not relevant_chunks:
        return 0.0
    retrieved_set = set(retrieved_chunks[:k])
    found = len(relevant_chunks & retrieved_set)
    return found / len(relevant_chunks)


def compute_precision_at_k(relevant_chunks: Set[int], retrieved_chunks: List[int], k: int) -> float:
    """Compute precision@k: fraction of top k that are relevant."""
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_chunks[:k])
    relevant_retrieved = len(relevant_chunks & retrieved_set)
    return relevant_retrieved / k


def compute_overlap(baseline_chunks: List[int], improved_chunks: List[int], k: int) -> float:
    """Compute overlap@k: fraction of top k that overlap between two result sets."""
    if k == 0:
        return 0.0
    baseline_set = set(baseline_chunks[:k])
    improved_set = set(improved_chunks[:k])
    overlap = len(baseline_set & improved_set)
    return overlap / k


def get_relevant_chunks_for_query(
    conn,
    query: str,
    filters: SearchFilters,
    k: int = 100,
    soft_lex_trigram_threshold: float = 0.1,
) -> Set[int]:
    """
    Get "relevant" chunks by taking union of qv1 and qv2_softlex results.
    This is a proxy for ground truth (in real evaluation, you'd have manual labels).
    """
    # Get results from both versions
    results_qv1 = hybrid_rrf(
        conn,
        query,
        filters=filters,
        k=k,
        expand_concordance=True,
        use_soft_lex=False,
        normalization_version=None,
    )
    
    results_qv2 = hybrid_rrf(
        conn,
        query,
        filters=filters,
        k=k,
        expand_concordance=True,
        use_soft_lex=True,
        soft_lex_threshold=0.3,
        soft_lex_trigram_threshold=soft_lex_trigram_threshold,
        normalization_version="norm_v2",
    )
    
    # Union of both = proxy for "relevant" chunks
    relevant = set(r.chunk_id for r in results_qv1) | set(r.chunk_id for r in results_qv2)
    return relevant


# -----------------------
# Evaluation Runner
# -----------------------

def evaluate_query(
    conn,
    query_config: Dict[str, Any],
    k: int = 10,
    save_results: bool = False,
    soft_lex_trigram_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Evaluate a single query comparing qv1 vs qv2_softlex."""
    query = query_config["query"]
    description = query_config["description"]
    collection = query_config["collection"]
    chunk_pv = query_config["chunk_pv"]
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {description}")
    print(f"Query: {query!r}")
    print(f"{'='*80}")
    
    filters = SearchFilters(
        chunk_pv=chunk_pv,
        collection_slugs=[collection] if collection else None,
    )
    
    # Get proxy "relevant" chunks (union of both versions)
    print("\n[Step 1] Computing proxy relevant chunks (union of qv1 + qv2)...")
    relevant_chunks = get_relevant_chunks_for_query(
        conn, query, filters, k=k * 2, soft_lex_trigram_threshold=soft_lex_trigram_threshold
    )
    print(f"  Relevant chunks (proxy): {len(relevant_chunks)}")
    
    # Run qv1 (baseline)
    print(f"\n[Step 2] Running qv1 (exact FTS only)...")
    results_qv1 = hybrid_rrf(
        conn,
        query,
        filters=filters,
        k=k,
        expand_concordance=True,
        use_soft_lex=False,
        normalization_version=None,
    )
    chunk_ids_qv1 = [r.chunk_id for r in results_qv1]
    print(f"  Results: {len(results_qv1)} chunks")
    print(f"  Chunk IDs: {chunk_ids_qv1[:5]}...")
    
    # Run qv2_softlex
    print(f"\n[Step 3] Running qv2_softlex (soft lexical matching)...")
    results_qv2 = hybrid_rrf(
        conn,
        query,
        filters=filters,
        k=k,
        expand_concordance=True,
        use_soft_lex=True,
        soft_lex_threshold=0.3,
        soft_lex_trigram_threshold=soft_lex_trigram_threshold,
        normalization_version="norm_v2",
    )
    chunk_ids_qv2 = [r.chunk_id for r in results_qv2]
    print(f"  Results: {len(results_qv2)} chunks")
    print(f"  Chunk IDs: {chunk_ids_qv2[:5]}...")
    
    # Count soft lex matches
    soft_lex_matches = sum(1 for r in results_qv2 if r.r_soft_lex is not None)
    print(f"  Soft lex matches: {soft_lex_matches}/{len(results_qv2)}")
    
    # Compute metrics
    print(f"\n[Step 4] Computing metrics...")
    recall_qv1 = compute_recall_at_k(relevant_chunks, chunk_ids_qv1, k)
    recall_qv2 = compute_recall_at_k(relevant_chunks, chunk_ids_qv2, k)
    precision_qv1 = compute_precision_at_k(relevant_chunks, chunk_ids_qv1, k)
    precision_qv2 = compute_precision_at_k(relevant_chunks, chunk_ids_qv2, k)
    overlap = compute_overlap(chunk_ids_qv1, chunk_ids_qv2, k)
    
    recall_improvement = recall_qv2 - recall_qv1
    precision_change = precision_qv2 - precision_qv1
    
    print(f"\n  Metrics @{k}:")
    print(f"    Recall:    qv1={recall_qv1:.3f}, qv2={recall_qv2:.3f} (Δ={recall_improvement:+.3f})")
    print(f"    Precision: qv1={precision_qv1:.3f}, qv2={precision_qv2:.3f} (Δ={precision_change:+.3f})")
    print(f"    Overlap:   {overlap:.3f}")
    
    # Find new chunks in qv2
    new_chunks = set(chunk_ids_qv2[:k]) - set(chunk_ids_qv1[:k])
    lost_chunks = set(chunk_ids_qv1[:k]) - set(chunk_ids_qv2[:k])
    print(f"\n  Result differences:")
    print(f"    New in qv2:  {len(new_chunks)} chunks {list(new_chunks)[:5]}")
    print(f"    Lost in qv2: {len(lost_chunks)} chunks {list(lost_chunks)[:5]}")
    
    # Get retrieval run IDs for linking
    run_id_qv1 = None
    run_id_qv2 = None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM retrieval_runs
            WHERE query_text = %s
              AND query_lang_version = 'qv1'
            ORDER BY created_at DESC
            LIMIT 1
        """, (query,))
        row = cur.fetchone()
        if row:
            run_id_qv1 = row[0]
        
        cur.execute("""
            SELECT id FROM retrieval_runs
            WHERE query_text = %s
              AND query_lang_version = 'qv2_softlex'
            ORDER BY created_at DESC
            LIMIT 1
        """, (query,))
        row = cur.fetchone()
        if row:
            run_id_qv2 = row[0]
    
    # Save to database if requested
    if save_results:
        print(f"\n[Step 5] Saving evaluation results to database...")
        with conn.cursor() as cur:
            eval_config = {
                "k": k,
                "soft_lex_threshold": 0.3,
                "soft_lex_trigram_threshold": soft_lex_trigram_threshold,
                "normalization_version": "norm_v2",
                "collection": collection,
                "chunk_pv": chunk_pv,
            }
            
            # Save qv1 metrics
            for metric_name, metric_value in [
                (f"recall@{k}", recall_qv1),
                (f"precision@{k}", precision_qv1),
            ]:
                cur.execute("""
                    INSERT INTO retrieval_evaluations (
                        query_text, query_lang_version, metric_name, metric_value,
                        search_type, chunk_pv, collection_slug, evaluation_config,
                        retrieval_run_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    query, "qv1", metric_name, float(metric_value),
                    "hybrid", chunk_pv, collection, Json(eval_config),
                    run_id_qv1,
                ))
            
            # Save qv2 metrics
            for metric_name, metric_value in [
                (f"recall@{k}", recall_qv2),
                (f"precision@{k}", precision_qv2),
            ]:
                cur.execute("""
                    INSERT INTO retrieval_evaluations (
                        query_text, query_lang_version, metric_name, metric_value,
                        search_type, chunk_pv, collection_slug, evaluation_config,
                        retrieval_run_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    query, "qv2_softlex", metric_name, float(metric_value),
                    "hybrid", chunk_pv, collection, Json(eval_config),
                    run_id_qv2,
                ))
            
            # Save overlap metric
            cur.execute("""
                INSERT INTO retrieval_evaluations (
                    query_text, query_lang_version, metric_name, metric_value,
                    search_type, chunk_pv, collection_slug, evaluation_config
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                query, "comparison", f"overlap@{k}", float(overlap),
                "hybrid", chunk_pv, collection, Json(eval_config),
            ))
        
        conn.commit()
        print(f"  Saved evaluation results")
    
    return {
        "query": query,
        "description": description,
        "k": k,
        "soft_lex_trigram_threshold": soft_lex_trigram_threshold,
        "recall_qv1": recall_qv1,
        "recall_qv2": recall_qv2,
        "recall_improvement": recall_improvement,
        "precision_qv1": precision_qv1,
        "precision_qv2": precision_qv2,
        "precision_change": precision_change,
        "overlap": overlap,
        "new_chunks": len(new_chunks),
        "lost_chunks": len(lost_chunks),
        "soft_lex_matches": soft_lex_matches,
        "run_id_qv1": run_id_qv1,
        "run_id_qv2": run_id_qv2,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate soft lexical retrieval (Phase 7)")
    ap.add_argument("--k", type=int, default=10, help="Top-k for evaluation (default: 10)")
    ap.add_argument("--collection", type=str, default=None, help="Filter by collection (default: per query)")
    ap.add_argument("--save", action="store_true", help="Save evaluation results to database")
    ap.add_argument("--query", type=str, default=None, help="Evaluate single query (default: all golden queries)")
    ap.add_argument("--soft-lex-trigram-threshold", type=float, default=0.1, help="Pre-filter threshold using similarity() (default: 0.1)")
    ap.add_argument("--sweep-trigram-thresholds", type=str, default=None, help="Comma-separated list of thresholds to sweep (e.g., 0.3,0.2,0.1)")
    args = ap.parse_args()
    
    conn = get_conn()
    
    try:
        print("=" * 80)
        print("PHASE 7: SOFT LEXICAL RETRIEVAL EVALUATION")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Top-k: {args.k}")
        print(f"  Save results: {'Yes' if args.save else 'No'}")
        print(f"  Soft lex trigram threshold: {args.soft_lex_trigram_threshold}")
        if args.sweep_trigram_thresholds:
            print(f"  Sweep thresholds: {args.sweep_trigram_thresholds}")
        
        # Select queries to evaluate
        if args.query:
            queries_to_eval = [q for q in GOLDEN_QUERIES if q["query"] == args.query]
            if not queries_to_eval:
                print(f"\nERROR: Query '{args.query}' not found in golden queries")
                return
        else:
            queries_to_eval = GOLDEN_QUERIES
        
        print(f"  Queries to evaluate: {len(queries_to_eval)}")
        
        # Determine thresholds to run
        if args.sweep_trigram_thresholds:
            thresholds = [float(t.strip()) for t in args.sweep_trigram_thresholds.split(",") if t.strip()]
        else:
            thresholds = [args.soft_lex_trigram_threshold]
        
        # Run evaluations
        results = []
        for trigram_threshold in thresholds:
            print("\n" + "-" * 80)
            print(f"Running evaluation with soft_lex_trigram_threshold={trigram_threshold}")
            print("-" * 80)
            for query_config in queries_to_eval:
                try:
                    result = evaluate_query(
                        conn,
                        query_config,
                        k=args.k,
                        save_results=args.save,
                        soft_lex_trigram_threshold=trigram_threshold,
                    )
                    result["soft_lex_trigram_threshold"] = trigram_threshold
                    results.append(result)
                except Exception as e:
                    print(f"\n[ERROR] Failed to evaluate {query_config['query']!r}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        if not results:
            print("\n  No results to summarize")
            return
        
        print(f"\n{'Threshold':<10} {'Query':<20} {'Recall Δ':<12} {'Precision Δ':<14} {'Overlap':<10} {'Soft Lex':<10}")
        print("-" * 100)
        
        for trigram_threshold in thresholds:
            subset = [r for r in results if r["soft_lex_trigram_threshold"] == trigram_threshold]
            if not subset:
                continue
            total_recall_improvement = sum(r["recall_improvement"] for r in subset)
            total_precision_change = sum(r["precision_change"] for r in subset)
            avg_recall_improvement = total_recall_improvement / len(subset)
            avg_precision_change = total_precision_change / len(subset)
            
            for r in subset:
                query_display = r["query"][:18]
                print(f"{trigram_threshold:<10.2f} {query_display:<20} {r['recall_improvement']:+.3f}        {r['precision_change']:+.3f}          {r['overlap']:.3f}      {r['soft_lex_matches']:<10}")
            print("-" * 100)
            print(f"{'Avg':<10} {'':<20} {avg_recall_improvement:+.3f}        {avg_precision_change:+.3f}")
        
        # Safety checks
        print("\n" + "=" * 80)
        print("SAFETY CHECKS")
        print("=" * 80)
        
        # Safety checks (first threshold)
        if thresholds:
            first_threshold = thresholds[0]
            subset = [r for r in results if r["soft_lex_trigram_threshold"] == first_threshold]
            if subset:
                avg_recall_improvement = sum(r["recall_improvement"] for r in subset) / len(subset)
                avg_precision_change = sum(r["precision_change"] for r in subset) / len(subset)
                
                recall_improved = avg_recall_improvement > 0.0
                precision_acceptable = avg_precision_change >= -0.1  # Allow small precision drop
                
                print(f"\n  Recall improvement (@{first_threshold}): {avg_recall_improvement:+.3f} {'✓' if recall_improved else '✗'}")
                print(f"  Precision change   (@{first_threshold}): {avg_precision_change:+.3f} {'✓' if precision_acceptable else '✗'}")
                
                if recall_improved and precision_acceptable:
                    print("\n  [PASS] qv2_softlex shows measurable recall gains without significant precision collapse")
                else:
                    print("\n  [WARN] Check results - may need threshold adjustment")
        
        # Thresholds for regression detection
        print("\n" + "=" * 80)
        print("REGRESSION THRESHOLDS")
        print("=" * 80)
        print("\n  Expected improvements:")
        print(f"    Recall:    > 0.0 (current: computed per threshold)")
        print(f"    Precision: >= -0.1 (current: computed per threshold)")
        print(f"    Overlap:   > 0.7 (maintains baseline results)")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
