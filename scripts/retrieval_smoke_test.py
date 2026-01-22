#!/usr/bin/env python3
"""
Retrieval smoke test covering key queries and verifying logging.

Tests:
- ENORMOUS
- Rosenberg AND ENORMOUS
- Oppenheimer
- One Silvermaster-specific query (Kaplan)
- User typo tests (silvermastre, Oppenheirner) - tests fuzzy lexical expansion
- OCR error tolerance tests - tests fuzzy lexical matching on OCR-damaged text

Usage:
    export DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
    python scripts/retrieval_smoke_test.py [--no-expand] [--top-k N] [--skip-typo-tests]
"""

import os
import sys
import argparse

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.ops import (
    get_conn,
    hybrid_rrf,
    SearchFilters,
)


def test_query(
    conn,
    query,
    search_func,
    filters: SearchFilters,
    k: int = 10,
    description: str = None,
    expand_concordance: bool = True,
    concordance_source_slug: str = "venona_vassiliev_concordance_v3",
    fuzzy_lex_enabled: bool = True,
    fuzzy_lex_min_similarity: float = 0.4,
    fuzzy_lex_top_k_per_token: int = 5,
    fuzzy_lex_max_total_variants: int = 50,
) -> dict:
    """Run a query and return results summary."""
    print(f"\n{'=' * 80}")
    print(f"TEST: {description or str(query)}")
    print(f"{'=' * 80}")
    
    # Format query for display
    query_display = query if isinstance(query, str) else " AND ".join(query)
    query_for_log = query_display
    
    print(f"Query: {query_display!r}")
    print(f"Search function: {search_func.__name__}")
    print(f"Filters: chunk_pv={filters.chunk_pv}, collections={filters.collection_slugs}")
    print(f"Expansion: {'enabled' if expand_concordance else 'disabled'}")
    if fuzzy_lex_enabled:
        print(f"Fuzzy Lexical: enabled (min_similarity={fuzzy_lex_min_similarity}, "
              f"top_k_per_token={fuzzy_lex_top_k_per_token}, "
              f"max_total_variants={fuzzy_lex_max_total_variants})")
    
    try:
        # Call with expansion parameters if using hybrid_rrf
        if search_func.__name__ == "hybrid_rrf":
            results = search_func(
                conn, 
                query, 
                filters=filters, 
                k=k,
                expand_concordance=expand_concordance,
                concordance_source_slug=concordance_source_slug,
                use_soft_lex=False,  # Disable soft-lex sidecar; use fuzzy lexical instead
                fuzzy_lex_enabled=fuzzy_lex_enabled,
                fuzzy_lex_min_similarity=fuzzy_lex_min_similarity,
                fuzzy_lex_top_k_per_token=fuzzy_lex_top_k_per_token,
                fuzzy_lex_max_total_variants=fuzzy_lex_max_total_variants,
            )
        else:
            results = search_func(conn, query, filters=filters, k=k)
    except Exception as e:
        # Rollback transaction on error to allow subsequent queries
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"\n[ERROR] Query failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "results": 0, "logged": 0, "error": str(e)}
    
    print(f"\nResults: {len(results)} chunks returned")
    if results:
        print(f"Top 3 chunk IDs: {[r.chunk_id for r in results[:3]]}")
        print(f"Top 3 collections: {[r.collection_slug for r in results[:3]]}")
        
        # Show expansion info if used
        if results[0].expand_concordance:
            print(f"Expansion was used: {results[0].expanded_query_text[:100] if results[0].expanded_query_text else 'N/A'}...")
            if results[0].expansion_terms:
                print(f"Expansion terms: {results[0].expansion_terms}")
        
        # Show fuzzy lexical info if used (check logged config)
        if fuzzy_lex_enabled and results:
            # Info will be shown from logged config below
            pass
        
        # Show preview of first result
        if results[0].preview:
            preview = results[0].preview[:200].replace('\n', ' ')
            print(f"First result preview: {preview}...")
    
    # Verify logging
    cur = conn.cursor()
    cur.execute("""
        SELECT
                id,
                search_type,
                array_length(returned_chunk_ids, 1) as num_results,
                expand_concordance,
                query_lang_version,
                retrieval_impl_version,
                normalization_version,
                retrieval_config_json,
                created_at
            FROM retrieval_runs
            WHERE query_text = %s
            ORDER BY created_at DESC
            LIMIT 1;
        """, (query_for_log,))
    
    log_entry = cur.fetchone()
    
    if log_entry:
        log_id, stype, num_results, expand, qlv, riv, nv, config_json, created_at = log_entry
        print(f"\nLogged entry (ID: {log_id}):")
        print(f"  Search type: {stype}")
        print(f"  Results logged: {num_results}")
        print(f"  Expansion: {expand}")
        print(f"  Query lang version: {qlv}")
        print(f"  Retrieval impl version: {riv}")
        if nv:
            print(f"  Normalization version: {nv}")
        if config_json and config_json.get("fuzzy_lex"):
            fuzzy_cfg = config_json.get("fuzzy_lex", {})
            if fuzzy_cfg.get("enabled"):
                print(f"  Fuzzy lexical config: enabled=True, "
                      f"min_similarity={fuzzy_cfg.get('min_similarity')}, "
                      f"top_k_per_token={fuzzy_cfg.get('top_k_per_token')}, "
                      f"dictionary_build_id={fuzzy_cfg.get('dictionary_build_id')}")
                expansions = fuzzy_cfg.get("expansions", {})
                tokens = fuzzy_cfg.get("tokens", [])
                if tokens:
                    print(f"  Fuzzy tokens ({len(tokens)}): {tokens}")
                if expansions:
                    total_variants = sum(len(v) for v in expansions.values())
                    print(f"  Fuzzy expansions: {len(expansions)} tokens expanded, {total_variants} total variants")
                    # Show up to 5 tokens with up to 5 variants each
                    shown = 0
                    for tok, variants in expansions.items():
                        if shown >= 5:
                            break
                        variant_lexemes = [v.get("lexeme") for v in variants[:5]]
                        print(f"    '{tok}' -> {variant_lexemes}")
                        shown += 1
        print(f"  Timestamp: {created_at}")
        
        # Verify consistency
        if num_results != len(results):
            print(f"  [WARNING] Mismatch: Log shows {num_results}, query returned {len(results)}")
            return {"status": "warning", "results": len(results), "logged": num_results, "run_id": log_id}
        else:
            print(f"  [OK] Logging consistent")
            
            # Verify evidence was written (acceptance criterion)
            cur.execute("""
                SELECT COUNT(*) 
                FROM retrieval_run_chunk_evidence 
                WHERE retrieval_run_id = %s
            """, (log_id,))
            evidence_count = cur.fetchone()[0]
            
            # For lex/hybrid: verify matched_lexemes or highlight exists (acceptance criterion)
            lex_explainability_ok = True
            if stype in ("lex", "hybrid") and evidence_count > 0:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM retrieval_run_chunk_evidence
                    WHERE retrieval_run_id = %s
                      AND (matched_lexemes IS NOT NULL OR highlight IS NOT NULL)
                """, (log_id,))
                explained_count = cur.fetchone()[0]
                if explained_count < evidence_count:
                    lex_explainability_ok = False
                    print(f"  [WARN] Lexical explainability: {explained_count}/{evidence_count} rows have matched_lexemes or highlight")
            
            if evidence_count == num_results:
                explain_status = ""
                if stype in ("lex", "hybrid") and lex_explainability_ok:
                    explain_status = " (lex explainability ✓)"
                elif stype in ("lex", "hybrid") and not lex_explainability_ok:
                    explain_status = " (lex explainability ✗)"
                print(f"  [OK] Evidence rows: {evidence_count} (matches returned chunks){explain_status}")
                
                # Verify rank matches returned_chunk_ids order (transaction acceptance criterion)
                cur.execute("""
                    SELECT returned_chunk_ids
                    FROM retrieval_runs
                    WHERE id = %s
                """, (log_id,))
                returned_chunk_ids_row = cur.fetchone()
                if returned_chunk_ids_row and returned_chunk_ids_row[0]:
                    returned_chunk_ids = returned_chunk_ids_row[0]
                    cur.execute("""
                        SELECT chunk_id, rank
                        FROM retrieval_run_chunk_evidence
                        WHERE retrieval_run_id = %s
                        ORDER BY rank
                    """, (log_id,))
                    evidence_rows = cur.fetchall()
                    
                    rank_matches = True
                    if len(evidence_rows) == len(returned_chunk_ids):
                        for i, (chunk_id, rank) in enumerate(evidence_rows):
                            if chunk_id != returned_chunk_ids[i] or rank != i + 1:
                                rank_matches = False
                                break
                    else:
                        rank_matches = False
                    
                    if rank_matches:
                        print(f"  [OK] Rank ordering: matches returned_chunk_ids order")
                    else:
                        print(f"  [ERROR] Rank ordering: mismatch between evidence rank and returned_chunk_ids order")
                        print(f"    Expected order: {returned_chunk_ids[:5]}...")
                        print(f"    Evidence order: {[r[0] for r in evidence_rows[:5]]}...")
                        return {"status": "error", "results": len(results), "logged": num_results, "run_id": log_id, "evidence_count": evidence_count, "rank_mismatch": True}
                
                return {"status": "ok", "results": len(results), "logged": num_results, "run_id": log_id, "evidence_count": evidence_count, "lex_explainability_ok": lex_explainability_ok}
            elif evidence_count == 0:
                print(f"  [WARN] No evidence rows found (table may not exist or migration not run)")
                return {"status": "warning", "results": len(results), "logged": num_results, "run_id": log_id, "evidence_count": 0}
            else:
                print(f"  [ERROR] Evidence mismatch: {evidence_count} rows but {num_results} chunks")
                return {"status": "error", "results": len(results), "logged": num_results, "run_id": log_id, "evidence_count": evidence_count}
    else:
        print(f"  [ERROR] No log entry found!")
        return {"status": "error", "results": len(results), "logged": None}


def main():
    ap = argparse.ArgumentParser(description="Retrieval smoke test")
    ap.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve (default: 10)")
    ap.add_argument("--no-expand", action="store_true", help="Disable concordance expansion")
    ap.add_argument("--collection", default="venona", help="Collection to filter by (default: venona)")
    ap.add_argument("--concordance-source", default="venona_vassiliev_concordance_v3", 
                    help="Concordance source slug (default: venona_vassiliev_concordance_v3)")
    ap.add_argument("--skip-typo-tests", action="store_true", help="Skip user typo and OCR error tests")
    ap.add_argument("--fuzzy-top-k", type=int, default=5, help="Fuzzy lexical: top-k variants per token (default: 5)")
    ap.add_argument("--fuzzy-max-variants", type=int, default=50, help="Fuzzy lexical: max total variants (default: 50)")
    ap.add_argument("--fuzzy-min-similarity", type=float, default=0.4, help="Fuzzy lexical: min similarity threshold (default: 0.4)")
    args = ap.parse_args()

    conn = get_conn()
    
    try:
        print("=" * 80)
        print("RETRIEVAL SMOKE TEST (Hybrid Search with Expansion)")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Search type: hybrid (RRF)")
        print(f"  Top K: {args.top_k}")
        print(f"  Collection filter: {args.collection}")
        print(f"  Expansion: {'disabled' if args.no_expand else 'enabled'}")
        if not args.no_expand:
            print(f"  Concordance source: {args.concordance_source}")
        print(f"  Fuzzy lexical: min_similarity={args.fuzzy_min_similarity}, "
              f"top_k_per_token={args.fuzzy_top_k}, max_total_variants={args.fuzzy_max_variants}")
        
        # Base filters for Venona/Vassiliev queries
        filters_venona = SearchFilters(
            chunk_pv="chunk_v1_full",
            collection_slugs=[args.collection] if args.collection else None,
        )
        
        # Filters for Silvermaster (different pipeline version)
        filters_silvermaster = SearchFilters(
            chunk_pv="chunk_v1_silvermaster_structured_4k",
            collection_slugs=["silvermaster"],
        )
        
        # Test queries - all using hybrid_rrf with expansion
        test_results = []
        expand_concordance = not args.no_expand
        
        # 1. ENORMOUS
        result = test_query(
            conn,
            "ENORMOUS",
            hybrid_rrf,
            filters_venona,
            k=args.top_k,
            description="ENORMOUS (single term)",
            expand_concordance=expand_concordance,
            concordance_source_slug=args.concordance_source,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
        )
        test_results.append(("ENORMOUS", result))
        
        # 2. Rosenberg AND ENORMOUS
        # hybrid_rrf takes a string query; lexical component will tokenize and OR terms
        result = test_query(
            conn,
            "Rosenberg AND ENORMOUS",
            hybrid_rrf,
            filters_venona,
            k=args.top_k,
            description="Rosenberg AND ENORMOUS",
            expand_concordance=expand_concordance,
            concordance_source_slug=args.concordance_source,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
        )
        test_results.append(("Rosenberg AND ENORMOUS", result))
        
        # 3. Oppenheimer
        result = test_query(
            conn,
            "Oppenheimer",
            hybrid_rrf,
            filters_venona,
            k=args.top_k,
            description="Oppenheimer (single term)",
            expand_concordance=expand_concordance,
            concordance_source_slug=args.concordance_source,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
        )
        test_results.append(("Oppenheimer", result))
        
        # 4. Silvermaster-specific query (Kaplan - a key figure in Silvermaster network)
        result = test_query(
            conn,
            "Kaplan",
            hybrid_rrf,
            filters_silvermaster,
            k=args.top_k,
            description="Kaplan (Silvermaster-specific)",
            expand_concordance=expand_concordance,
            concordance_source_slug=args.concordance_source,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
        )
        test_results.append(("Kaplan (Silvermaster)", result))
        
        # Typo and OCR error tests (can be skipped with --skip-typo-tests)
        if not args.skip_typo_tests:
            # 5. User typo test: "silvermastre" (missing 'r') - should match "silvermaster" via fuzzy lexical expansion
            result = test_query(
                conn,
                "silvermastre",  # Typo: missing 'r'
                hybrid_rrf,
                filters_silvermaster,
                k=args.top_k,
                description="silvermastre (USER TYPO - missing 'r', fuzzy lexical should expand)",
                expand_concordance=expand_concordance,
                concordance_source_slug=args.concordance_source,
                fuzzy_lex_enabled=True,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
            )
            test_results.append(("silvermastre (typo)", result))
            
            # 6. User typo test: "Oppenheirner" (typo) - should match "Oppenheimer" via fuzzy lexical
            result = test_query(
                conn,
                "Oppenheirner",  # Typo
                hybrid_rrf,
                filters_venona,
                k=args.top_k,
                description="Oppenheirner (USER TYPO - fuzzy lexical should expand)",
                expand_concordance=expand_concordance,
                concordance_source_slug=args.concordance_source,
                fuzzy_lex_enabled=True,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
            )
            test_results.append(("Oppenheirner (typo)", result))
            
            # 7. OCR error tolerance: Query correct term, expect to match OCR-damaged text
            # Note: This tests that fuzzy lexical can handle OCR errors in document text
            # We query for the correct spelling and expect fuzzy lexical to expand to OCR variants
            result = test_query(
                conn,
                "silvermaster",  # Correct spelling
                hybrid_rrf,
                filters_silvermaster,
                k=args.top_k,
                description="silvermaster (correct spelling, fuzzy lexical should match OCR variants)",
                expand_concordance=expand_concordance,
                concordance_source_slug=args.concordance_source,
                fuzzy_lex_enabled=True,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
            )
            test_results.append(("silvermaster (OCR tolerance)", result))
            
            # 8. OCR variant in text: Query for known OCR misspelling, verify it matches
            result = test_query(
                conn,
                "silvehmaster",  # Known OCR variant
                hybrid_rrf,
                filters_silvermaster,
                k=args.top_k,
                description="silvehmaster (OCR variant in query, should match correct spelling)",
                expand_concordance=expand_concordance,
                concordance_source_slug=args.concordance_source,
                fuzzy_lex_enabled=True,
                fuzzy_lex_min_similarity=args.fuzzy_min_similarity,
                fuzzy_lex_top_k_per_token=args.fuzzy_top_k,
                fuzzy_lex_max_total_variants=args.fuzzy_max_variants,
            )
            test_results.append(("silvehmaster (OCR variant)", result))
        else:
            print("\n[Skipping typo/OCR tests (--skip-typo-tests)]")
        
        # Summary
        print("\n" + "=" * 80)
        print("SMOKE TEST SUMMARY")
        print("=" * 80)
        
        all_passed = True
        evidence_checks_passed = True
        
        for query, result in test_results:
            status = result.get("status", "unknown")
            results_count = result.get("results", 0)
            logged_count = result.get("logged", 0)
            evidence_count = result.get("evidence_count", None)
            
            if status == "ok" and results_count > 0:
                evidence_status = ""
                if evidence_count is not None:
                    if evidence_count == results_count:
                        evidence_status = f", evidence: {evidence_count} rows ✓"
                    else:
                        evidence_status = f", evidence: {evidence_count} rows ✗ (expected {results_count})"
                        evidence_checks_passed = False
                print(f"\n[PASS] {query}: {results_count} results, logged correctly{evidence_status}")
            elif status == "warning":
                evidence_status = ""
                if evidence_count is not None:
                    evidence_status = f", evidence: {evidence_count} rows"
                print(f"\n[WARN] {query}: {results_count} results, but log shows {logged_count}{evidence_status}")
                all_passed = False
            elif status == "error":
                evidence_status = ""
                if evidence_count is not None:
                    evidence_status = f", evidence: {evidence_count} rows"
                print(f"\n[FAIL] {query}: Error - {result.get('error', 'Unknown error')}{evidence_status}")
                all_passed = False
            elif results_count == 0:
                print(f"\n[WARN] {query}: No results returned (may be expected)")
            else:
                print(f"\n[UNKNOWN] {query}: Status={status}")
                all_passed = False
        
        # Evidence validation summary
        print("\n" + "=" * 80)
        print("EVIDENCE VALIDATION")
        print("=" * 80)
        
        cur = conn.cursor()
        # Check if evidence table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'retrieval_run_chunk_evidence'
            )
        """)
        evidence_table_exists = cur.fetchone()[0]
        
        if not evidence_table_exists:
            print("\n  [WARN] Evidence table does not exist (migration may not be run)")
            evidence_checks_passed = False
        else:
            # 1. Validate evidence rows count == top_k for recent runs
            cur.execute("""
                SELECT 
                    rr.id,
                    rr.query_text,
                    rr.search_type,
                    rr.top_k,
                    array_length(rr.returned_chunk_ids, 1) as num_chunks,
                    COUNT(e.chunk_id) as evidence_rows
                FROM retrieval_runs rr
                LEFT JOIN retrieval_run_chunk_evidence e ON e.retrieval_run_id = rr.id
                WHERE rr.created_at > now() - interval '1 hour'
                GROUP BY rr.id, rr.query_text, rr.search_type, rr.top_k, rr.returned_chunk_ids
                ORDER BY rr.created_at DESC
                LIMIT 10
            """)
            
            evidence_rows = cur.fetchall()
            
            if not evidence_rows:
                print("\n  No recent retrieval runs found (last hour)")
            else:
                print(f"\n  Checking {len(evidence_rows)} recent runs:")
                for run_id, query_text, search_type, top_k, num_chunks, evidence_rows_count in evidence_rows:
                    if num_chunks == evidence_rows_count == top_k:
                        print(f"    ✓ Run {run_id}: {query_text[:40]!r} ({search_type}) - {num_chunks} chunks, {evidence_rows_count} evidence rows")
                    elif evidence_rows_count == 0:
                        print(f"    ⚠ Run {run_id}: {query_text[:40]!r} - {num_chunks} chunks, NO evidence (old run or migration not applied)")
                        evidence_checks_passed = False
                    else:
                        print(f"    ✗ Run {run_id}: {query_text[:40]!r} - {num_chunks} chunks, {evidence_rows_count} evidence rows (MISMATCH)")
                        evidence_checks_passed = False
            
            # 2. Spot check highlights and evidence quality
            cur.execute("""
                SELECT 
                  rr.id AS run_id,
                  rr.query_text,
                  rr.search_type,
                  rrc.rank,
                  LEFT(rrc.highlight, 80) AS highlight_preview,
                  rrc.matched_lexemes IS NOT NULL AS has_matched_lexemes,
                  (rrc.explain_json->'semantic'->'semantic_terms') IS NOT NULL AS has_semantic_terms
                FROM retrieval_run_chunk_evidence rrc
                JOIN retrieval_runs rr ON rr.id = rrc.retrieval_run_id
                WHERE rr.created_at > now() - interval '1 hour'
                  AND rr.search_type IN ('lex', 'hybrid', 'vector')
                ORDER BY rr.created_at DESC, rrc.rank
                LIMIT 10
            """)
            evidence_samples = cur.fetchall()
            
            if evidence_samples:
                print(f"\n  Evidence samples (last 10 rows):")
                print(f"    {'Run':<6} {'Type':<8} {'Rank':<5} {'Lex':<5} {'Sem':<5} {'Highlight'}")
                print(f"    {'-'*6} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*50}")
                for run_id, query_text, search_type, rank, highlight, has_lex, has_sem in evidence_samples:
                    highlight_str = (highlight[:47] + "...") if highlight and len(highlight) > 50 else (highlight or "")
                    print(f"    {run_id:<6} {search_type:<8} {rank:<5} {'✓' if has_lex else '✗':<5} {'✓' if has_sem else '✗':<5} {highlight_str}")
                
                # Check for missing evidence
                missing_lex = sum(1 for _, _, st, _, _, has_lex, _ in evidence_samples if st in ('lex', 'hybrid') and not has_lex)
                missing_sem = sum(1 for _, _, st, _, _, _, has_sem in evidence_samples if st in ('vector', 'hybrid') and not has_sem)
                
                if missing_lex > 0:
                    print(f"\n  [WARN] {missing_lex} lex/hybrid samples missing matched_lexemes")
                    evidence_checks_passed = False
                else:
                    print(f"\n  [OK] Lexical evidence: All lex/hybrid samples have matched_lexemes")
                
                if missing_sem > 0:
                    print(f"\n  [WARN] {missing_sem} vector/hybrid samples missing semantic_terms")
                    evidence_checks_passed = False
                else:
                    print(f"\n  [OK] Semantic evidence: All vector/hybrid samples have semantic_terms")
        
        if evidence_checks_passed:
            print("\n  [OK] All evidence checks passed")
        else:
            print("\n  [WARNING] Some evidence checks failed - see details above")
            all_passed = False
        
        print("\n" + "=" * 80)
        if all_passed and evidence_checks_passed:
            print("[SUCCESS] All tests passed! (including evidence validation)")
        elif all_passed:
            print("[PARTIAL] Query tests passed, but evidence validation had issues")
        else:
            print("[WARNING] Some tests had issues - see details above")
        print("=" * 80)
        
        # Show recent log entries
        print("\n" + "=" * 80)
        print("RECENT LOG ENTRIES (last 5)")
        print("=" * 80)
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                query_text,
                search_type,
                chunk_pv,
                array_length(returned_chunk_ids, 1) as num_results,
                expand_concordance,
                created_at
            FROM retrieval_runs
            ORDER BY created_at DESC
            LIMIT 5;
        """)
        logs = cur.fetchall()
        for query, stype, pv, num_results, expand, created_at in logs:
            expand_str = "Yes" if expand else "No"
            print(f"\n  [{created_at}] {query!r}")
            print(f"    {stype} | PV={pv} | {num_results} results | Expansion={expand_str}")
        
        # Evaluation check (fuzzy lexical and soft lex evaluations)
        print("\n" + "=" * 80)
        print("EVALUATION STATUS")
        print("=" * 80)
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'retrieval_evaluations'
            )
        """)
        eval_table_exists = cur.fetchone()[0]
        
        if eval_table_exists:
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT query_text) as num_queries,
                    COUNT(*) as num_metrics,
                    MAX(evaluated_at) as last_evaluation
                FROM retrieval_evaluations
            """)
            row = cur.fetchone()
            if row and row[0] > 0:
                num_queries, num_metrics, last_eval = row
                print(f"\n  Evaluation table exists: ✓")
                print(f"  Queries evaluated: {num_queries}")
                print(f"  Metrics recorded: {num_metrics}")
                print(f"  Last evaluation: {last_eval}")
                
                # Show recent evaluation summary
                cur.execute("""
                    SELECT 
                        query_text,
                        query_lang_version,
                        metric_name,
                        metric_value,
                        evaluated_at
                    FROM retrieval_evaluations
                    WHERE evaluated_at > now() - interval '7 days'
                    ORDER BY evaluated_at DESC
                    LIMIT 10
                """)
                recent_evals = cur.fetchall()
                if recent_evals:
                    print(f"\n  Recent evaluations (last 7 days):")
                    for qtext, qlv, metric, value, eval_at in recent_evals:
                        print(f"    [{eval_at}] {qtext[:30]!r} | {qlv} | {metric} = {value:.3f}")
            else:
                print(f"\n  Evaluation table exists but empty")
                print(f"  Run: python scripts/eval_fuzzy_lex.py --k 10")
        else:
            print(f"\n  Evaluation table not found")
            print(f"  Run migration: make eval-table")
            print(f"  Then run: python scripts/eval_fuzzy_lex.py --k 10")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
