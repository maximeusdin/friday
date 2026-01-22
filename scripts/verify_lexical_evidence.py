#!/usr/bin/env python3
"""
Verify lexical explainability: check that lex/hybrid runs have matched_lexemes or highlight.

Usage:
    export DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
    python scripts/verify_lexical_evidence.py [--run-id <id>] [--recent N]
"""

import os
import sys
import argparse

import psycopg2

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def verify_run(conn, run_id: int):
    """Verify a specific retrieval_run has proper lexical evidence."""
    cur = conn.cursor()
    
    # Get run info
    cur.execute("""
        SELECT 
            id,
            query_text,
            tsquery_text,
            search_type,
            array_length(returned_chunk_ids, 1) as num_chunks
        FROM retrieval_runs
        WHERE id = %s
    """, (run_id,))
    
    run = cur.fetchone()
    if not run:
        print(f"❌ No retrieval_run found with id={run_id}")
        return False
    
    run_id, query_text, tsquery_text, search_type, num_chunks = run
    
    print(f"\n{'=' * 80}")
    print(f"VERIFYING RETRIEVAL RUN {run_id}")
    print(f"{'=' * 80}")
    print(f"Query: {query_text!r}")
    print(f"Search type: {search_type}")
    print(f"tsquery_text: {tsquery_text!r if tsquery_text else 'NULL'}")
    print(f"Expected chunks: {num_chunks}")
    
    # Get evidence
    cur.execute("""
        SELECT 
            COUNT(*) as total_evidence,
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
    
    # Check acceptance criterion
    if search_type in ("lex", "hybrid"):
        if total == 0:
            print(f"\n⚠️  No evidence rows found (migration may not be applied)")
            return False
        elif total != num_chunks:
            print(f"\n❌ Evidence count mismatch: {total} rows but {num_chunks} chunks")
            return False
        elif with_either < total:
            print(f"\n❌ ACCEPTANCE CRITERION FAILED:")
            print(f"   {total - with_either} rows missing both matched_lexemes AND highlight")
            return False
        else:
            print(f"\n✅ ACCEPTANCE CRITERION PASSED:")
            print(f"   All {total} evidence rows have matched_lexemes or highlight")
            if with_lexemes == total and with_highlight == total:
                print(f"   Perfect: all rows have BOTH matched_lexemes AND highlight")
            elif with_lexemes == total:
                print(f"   All rows have matched_lexemes, {with_highlight} have highlight")
            elif with_highlight == total:
                print(f"   All rows have highlight, {with_lexemes} have matched_lexemes")
            return True
    else:
        print(f"\nℹ️  Search type '{search_type}' doesn't require lexical explainability")
        return True


def show_sample_evidence(conn, run_id: int, limit: int = 5):
    """Show sample evidence rows."""
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            rank,
            chunk_id,
            matched_lexemes,
            LEFT(highlight, 150) as highlight_preview,
            score_lex,
            score_hybrid
        FROM retrieval_run_chunk_evidence
        WHERE retrieval_run_id = %s
        ORDER BY rank
        LIMIT %s
    """, (run_id, limit))
    
    rows = cur.fetchall()
    
    if not rows:
        print("\nNo evidence rows to display")
        return
    
    print(f"\n{'=' * 80}")
    print(f"SAMPLE EVIDENCE (first {len(rows)} rows)")
    print(f"{'=' * 80}")
    
    for rank, chunk_id, lexemes, highlight_preview, score_lex, score_hybrid in rows:
        print(f"\nRank {rank} (chunk_id={chunk_id}):")
        if lexemes:
            print(f"  matched_lexemes: {lexemes}")
        else:
            print(f"  matched_lexemes: NULL")
        
        if highlight_preview:
            print(f"  highlight: {highlight_preview}...")
        else:
            print(f"  highlight: NULL")
        
        if score_lex is not None:
            print(f"  score_lex: {score_lex:.4f}")
        if score_hybrid is not None:
            print(f"  score_hybrid: {score_hybrid:.4f}")


def verify_recent_runs(conn, limit: int = 10):
    """Verify recent lex/hybrid runs."""
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            rr.id,
            rr.query_text,
            rr.search_type,
            rr.tsquery_text,
            array_length(rr.returned_chunk_ids, 1) as num_chunks,
            COUNT(e.chunk_id) as evidence_count,
            COUNT(CASE WHEN e.matched_lexemes IS NOT NULL OR e.highlight IS NOT NULL THEN 1 END) as explained_count
        FROM retrieval_runs rr
        LEFT JOIN retrieval_run_chunk_evidence e ON e.retrieval_run_id = rr.id
        WHERE rr.search_type IN ('lex', 'hybrid')
          AND rr.created_at > now() - interval '24 hours'
        GROUP BY rr.id, rr.query_text, rr.search_type, rr.tsquery_text, rr.returned_chunk_ids
        ORDER BY rr.created_at DESC
        LIMIT %s
    """, (limit,))
    
    runs = cur.fetchall()
    
    if not runs:
        print("No recent lex/hybrid runs found")
        return
    
    print(f"\n{'=' * 80}")
    print(f"VERIFYING RECENT LEX/HYBRID RUNS (last 24 hours)")
    print(f"{'=' * 80}")
    
    all_ok = True
    for run_id, query_text, search_type, tsquery_text, num_chunks, evidence_count, explained_count in runs:
        status = "✅"
        if evidence_count == 0:
            status = "⚠️  (no evidence)"
            all_ok = False
        elif evidence_count != num_chunks:
            status = "❌ (count mismatch)"
            all_ok = False
        elif explained_count < evidence_count:
            status = "❌ (missing explainability)"
            all_ok = False
        
        print(f"\n{status} Run {run_id}: {query_text[:50]!r}")
        print(f"    Type: {search_type} | Chunks: {num_chunks} | Evidence: {evidence_count} | Explained: {explained_count}")
        if tsquery_text:
            print(f"    tsquery: {tsquery_text[:60]!r}...")
    
    if all_ok:
        print(f"\n✅ All recent runs pass explainability checks")
    else:
        print(f"\n⚠️  Some runs have issues - see details above")


def main():
    ap = argparse.ArgumentParser(description="Verify lexical explainability in evidence")
    ap.add_argument("--run-id", type=int, help="Verify specific retrieval_run_id")
    ap.add_argument("--recent", type=int, default=10, help="Check N recent runs (default: 10)")
    ap.add_argument("--sample", type=int, default=5, help="Show N sample evidence rows (default: 5)")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        if args.run_id:
            # Verify specific run
            success = verify_run(conn, args.run_id)
            show_sample_evidence(conn, args.run_id, args.sample)
            sys.exit(0 if success else 1)
        else:
            # Verify recent runs
            verify_recent_runs(conn, args.recent)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
