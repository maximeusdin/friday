#!/usr/bin/env python3
"""
Evaluate concordance expansion: A/B comparison queries.

Usage:
    export DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
    possibly export OPENAI_API_KEY="..."
    python scripts/eval_concordance_expansion.py

This will:
1. Check concordance sources
2. Run test queries with/without expansion
3. Show side-by-side comparisons
"""

import os
import sys
from typing import List, Dict, Any

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from retrieval.ops import (
    get_conn,
    concordance_expand_terms,
    build_expanded_query_string,
    vector_search,
    hybrid_rrf,
    SearchFilters,
)
from scripts.query_chunks import fts_or_query


# -----------------------
# Test queries
# -----------------------

TEST_QUERIES = [
    # Acronyms (should expand to full names)
    "AAF",
    "FBI",
    "OSS",
    
    # Cover names (should expand to aliases)
    "ANTENNA",
    "LIBERAL",
    
    # Person names (should expand to variants)
    "SOBLE",
    "Soble",
    
    # Full names (should normalize/match)
    "Army Air Force, U.S.",
    "Army Air Force, U",
    
    # Undeciphered names
    "Undeciphered Name No. 19",
]


# -----------------------
# Evaluation helpers
# -----------------------

def print_expansions(conn, query: str, source_slug: str = "venona_vassiliev_concordance_v3"):
    """Print what a query expands to."""
    expansions = concordance_expand_terms(conn, query, source_slug=source_slug)
    expanded_str = build_expanded_query_string(query, expansions)
    
    print(f"\nQuery: {query!r}")
    print(f"  Expansions ({len(expansions)}): {expansions[:10]}")
    if len(expansions) > 10:
        print(f"    ... ({len(expansions) - 10} more)")
    print(f"  Expanded string: {expanded_str[:200]}")
    return expansions


def compare_results(
    query: str,
    results_with: List[Any],
    results_without: List[Any],
    mode: str,
) -> Dict[str, Any]:
    """Compare two result sets and return metrics."""
    ids_with = {r.chunk_id for r in results_with}
    ids_without = {r.chunk_id for r in results_without}
    
    overlap = ids_with & ids_without
    only_with = ids_with - ids_without
    only_without = ids_without - ids_with
    
    return {
        "query": query,
        "mode": mode,
        "with_count": len(results_with),
        "without_count": len(results_without),
        "overlap_count": len(overlap),
        "only_with_count": len(only_with),
        "only_without_count": len(only_without),
        "overlap_pct": len(overlap) / max(len(ids_without), 1) * 100,
        "only_with_ids": list(only_with)[:5],  # first 5
        "only_without_ids": list(only_without)[:5],
    }


def print_comparison(metrics: Dict[str, Any]):
    """Print a formatted comparison."""
    print(f"\n{'='*80}")
    print(f"Query: {metrics['query']!r}  Mode: {metrics['mode']}")
    print(f"  With expansion:    {metrics['with_count']} results")
    print(f"  Without expansion:  {metrics['without_count']} results")
    print(f"  Overlap:            {metrics['overlap_count']} ({metrics['overlap_pct']:.1f}% of baseline)")
    print(f"  New with expansion: {metrics['only_with_count']} chunks")
    print(f"  Lost with expansion: {metrics['only_without_count']} chunks")
    
    if metrics['only_with_ids']:
        print(f"    New chunk_ids: {metrics['only_with_ids']}")
    if metrics['only_without_ids']:
        print(f"    Lost chunk_ids: {metrics['only_without_ids']}")


# -----------------------
# Main evaluation
# -----------------------

def main():
    conn = get_conn()
    
    # Check concordance sources
    print("=" * 80)
    print("CONCORDANCE EXPANSION EVALUATION")
    print("=" * 80)
    
    with conn.cursor() as cur:
        cur.execute("SELECT id, slug, title FROM concordance_sources ORDER BY id DESC LIMIT 5")
        sources = cur.fetchall()
        if not sources:
            print("\n⚠️  No concordance sources found in database!")
            print("   Run concordance ingestion first.")
            return
        
        print("\nAvailable concordance sources:")
        for sid, slug, title in sources:
            print(f"  [{sid}] {slug} - {title}")
        
        default_slug = sources[0][1]  # use most recent
        print(f"\nUsing source: {default_slug}")
    
    # Test expansion coverage
    print("\n" + "=" * 80)
    print("EXPANSION COVERAGE TEST")
    print("=" * 80)
    
    for query in TEST_QUERIES[:5]:  # first 5 for coverage
        try:
            print_expansions(conn, query, source_slug=default_slug)
        except Exception as e:
            print(f"\nQuery: {query!r}")
            print(f"  ERROR: {e}")
    
    # A/B retrieval comparisons
    print("\n" + "=" * 80)
    print("A/B RETRIEVAL COMPARISON")
    print("=" * 80)
    
    filters = SearchFilters(
        chunk_pv="chunk_v1_full",
        collection_slugs=["venona"],  # focus on one collection for cleaner results
    )
    
    # Test a few queries with hybrid mode (most representative)
    test_queries = ["AAF", "SOBLE", "Army Air Force, U.S."]
    
    for query in test_queries:
        try:
            print(f"\n{'─'*80}")
            print(f"Testing: {query!r}")
            
            # Without expansion (explicitly disable)
            results_without = hybrid_rrf(
                conn,
                query,
                filters=filters,
                k=20,
                expand_concordance=False,  # explicitly disable
            )
            
            # With expansion
            results_with = hybrid_rrf(
                conn,
                query,
                filters=filters,
                k=20,
                expand_concordance=True,
                concordance_source_slug=default_slug,
            )
            
            metrics = compare_results(query, results_with, results_without, "hybrid")
            print_comparison(metrics)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nInterpretation:")
    print("  - High overlap (>80%): expansion preserves existing results")
    print("  - New chunks: expansion improves recall (good if relevant)")
    print("  - Lost chunks: expansion may have shifted ranking (check if bad)")
    
    conn.close()


if __name__ == "__main__":
    main()
