#!/usr/bin/env python3
"""Quick test to verify evidence insertion works."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.ops import get_conn, hybrid_rrf, SearchFilters

conn = get_conn()
try:
    filters = SearchFilters(
        chunk_pv="chunk_v1_full",
        collection_slugs=["venona"],
    )
    
    hits = hybrid_rrf(
        conn,
        query="Rosenberg AND ENORMOUS",
        filters=filters,
        k=5,
        expand_concordance=True,
        concordance_source_slug="venona_vassiliev_concordance_v3",
        log_run=True,
    )
    
    print(f"Got {len(hits)} hits")
    
    # Get the latest run_id
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, query_text, search_type, tsquery_text
            FROM retrieval_runs
            ORDER BY id DESC
            LIMIT 1
        """)
        run = cur.fetchone()
        if run:
            run_id, query_text, search_type, tsquery_text = run
            print(f"\nLatest run: {run_id}")
            print(f"  Query: {query_text}")
            print(f"  Search type: {search_type}")
            print(f"  tsquery_text: {tsquery_text}")
            
            # Check evidence
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN matched_lexemes IS NOT NULL THEN 1 END) as with_lexemes,
                    COUNT(CASE WHEN highlight IS NOT NULL THEN 1 END) as with_highlight,
                    COUNT(CASE WHEN matched_lexemes IS NOT NULL OR highlight IS NOT NULL THEN 1 END) as with_either
                FROM retrieval_run_chunk_evidence
                WHERE retrieval_run_id = %s
            """, (run_id,))
            
            total, with_lexemes, with_highlight, with_either = cur.fetchone()
            print(f"\nEvidence rows: {total}")
            print(f"  With matched_lexemes: {with_lexemes}")
            print(f"  With highlight: {with_highlight}")
            print(f"  With either: {with_either}")
            
            # Show sample
            cur.execute("""
                SELECT rank, chunk_id, matched_lexemes, LEFT(highlight, 80) as highlight_preview
                FROM retrieval_run_chunk_evidence
                WHERE retrieval_run_id = %s
                ORDER BY rank
                LIMIT 3
            """, (run_id,))
            
            print("\nSample evidence:")
            for rank, chunk_id, lexemes, highlight_preview in cur.fetchall():
                print(f"  Rank {rank} (chunk {chunk_id}):")
                print(f"    matched_lexemes: {lexemes}")
                print(f"    highlight: {highlight_preview}...")
finally:
    conn.close()
