#!/usr/bin/env python3
"""
admin_validate_integrity.py

Validate data integrity for denormalized fields and relationships.

Checks:
1. entity_mentions.document_id matches chunk_metadata.document_id (via chunk_id)
2. date_mentions.document_id matches chunk_metadata.document_id (via chunk_id)
3. result_sets reference valid retrieval_runs
4. Plans with execution_mode='count' have NULL result_set_id

Usage:
    python scripts/admin_validate_integrity.py                    # Run all checks
    python scripts/admin_validate_integrity.py --check entity     # Check entity_mentions only
    python scripts/admin_validate_integrity.py --check date       # Check date_mentions only
    python scripts/admin_validate_integrity.py --sample 1000      # Sample size for large tables
    python scripts/admin_validate_integrity.py --fix              # Attempt to fix drift (dry-run by default)
    python scripts/admin_validate_integrity.py --fix --commit     # Actually apply fixes
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn


def check_entity_mentions_document_id(conn, sample_size: int = 10000) -> dict:
    """
    Check that entity_mentions.document_id matches chunk_metadata.document_id.
    
    Returns dict with check results and any mismatched rows.
    """
    with conn.cursor() as cur:
        # Count total entity_mentions
        cur.execute("SELECT COUNT(*) FROM entity_mentions")
        total_rows = cur.fetchone()[0]
        
        # Find mismatches (sample for large tables)
        cur.execute(
            """
            SELECT em.id, em.chunk_id, em.document_id AS em_doc_id, cm.document_id AS cm_doc_id
            FROM entity_mentions em
            JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
            WHERE em.document_id != cm.document_id
            LIMIT %s
            """,
            (sample_size,),
        )
        mismatches = cur.fetchall()
        
        # Count total mismatches
        cur.execute(
            """
            SELECT COUNT(*)
            FROM entity_mentions em
            JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
            WHERE em.document_id != cm.document_id
            """
        )
        mismatch_count = cur.fetchone()[0]
        
        return {
            "check": "entity_mentions.document_id",
            "total_rows": total_rows,
            "mismatch_count": mismatch_count,
            "sample_mismatches": [
                {"id": r[0], "chunk_id": r[1], "em_doc_id": r[2], "cm_doc_id": r[3]}
                for r in mismatches[:10]
            ],
            "passed": mismatch_count == 0,
        }


def check_date_mentions_document_id(conn, sample_size: int = 10000) -> dict:
    """
    Check that date_mentions.document_id matches chunk_metadata.document_id.
    """
    with conn.cursor() as cur:
        # Count total date_mentions
        cur.execute("SELECT COUNT(*) FROM date_mentions")
        total_rows = cur.fetchone()[0]
        
        # Find mismatches
        cur.execute(
            """
            SELECT dm.id, dm.chunk_id, dm.document_id AS dm_doc_id, cm.document_id AS cm_doc_id
            FROM date_mentions dm
            JOIN chunk_metadata cm ON cm.chunk_id = dm.chunk_id
            WHERE dm.document_id != cm.document_id
            LIMIT %s
            """,
            (sample_size,),
        )
        mismatches = cur.fetchall()
        
        # Count total mismatches
        cur.execute(
            """
            SELECT COUNT(*)
            FROM date_mentions dm
            JOIN chunk_metadata cm ON cm.chunk_id = dm.chunk_id
            WHERE dm.document_id != cm.document_id
            """
        )
        mismatch_count = cur.fetchone()[0]
        
        return {
            "check": "date_mentions.document_id",
            "total_rows": total_rows,
            "mismatch_count": mismatch_count,
            "sample_mismatches": [
                {"id": r[0], "chunk_id": r[1], "dm_doc_id": r[2], "cm_doc_id": r[3]}
                for r in mismatches[:10]
            ],
            "passed": mismatch_count == 0,
        }


def check_result_sets_run_reference(conn) -> dict:
    """
    Check that result_sets reference valid retrieval_runs.
    """
    with conn.cursor() as cur:
        # Count total result_sets
        cur.execute("SELECT COUNT(*) FROM result_sets")
        total_rows = cur.fetchone()[0]
        
        # Find orphaned result_sets (reference non-existent run)
        cur.execute(
            """
            SELECT rs.id, rs.retrieval_run_id
            FROM result_sets rs
            LEFT JOIN retrieval_runs rr ON rr.id = rs.retrieval_run_id
            WHERE rs.retrieval_run_id IS NOT NULL AND rr.id IS NULL
            LIMIT 100
            """
        )
        orphans = cur.fetchall()
        
        return {
            "check": "result_sets.retrieval_run_id FK integrity",
            "total_rows": total_rows,
            "orphan_count": len(orphans),
            "sample_orphans": [{"id": r[0], "run_id": r[1]} for r in orphans[:10]],
            "passed": len(orphans) == 0,
        }


def check_count_mode_null_result_set(conn) -> dict:
    """
    Check that plans with execution_mode='count' have NULL result_set_id.
    """
    with conn.cursor() as cur:
        # Find count-mode plans with non-null result_set_id
        cur.execute(
            """
            SELECT id, status, result_set_id, plan_json->'_metadata'->>'execution_mode' AS exec_mode
            FROM research_plans
            WHERE plan_json->'_metadata'->>'execution_mode' = 'count'
              AND result_set_id IS NOT NULL
            LIMIT 100
            """
        )
        violations = cur.fetchall()
        
        # Count total count-mode plans
        cur.execute(
            """
            SELECT COUNT(*)
            FROM research_plans
            WHERE plan_json->'_metadata'->>'execution_mode' = 'count'
            """
        )
        total_count_mode = cur.fetchone()[0]
        
        return {
            "check": "count-mode plans have NULL result_set_id",
            "total_count_mode_plans": total_count_mode,
            "violation_count": len(violations),
            "sample_violations": [
                {"id": r[0], "status": r[1], "result_set_id": r[2]}
                for r in violations[:10]
            ],
            "passed": len(violations) == 0,
        }


def fix_entity_mentions_document_id(conn, commit: bool = False) -> dict:
    """
    Fix entity_mentions.document_id drift from chunk_metadata.
    """
    with conn.cursor() as cur:
        # Update mismatched rows
        cur.execute(
            """
            UPDATE entity_mentions em
            SET document_id = cm.document_id
            FROM chunk_metadata cm
            WHERE cm.chunk_id = em.chunk_id
              AND em.document_id != cm.document_id
            """
        )
        fixed_count = cur.rowcount
        
        if commit:
            conn.commit()
            return {"fixed": fixed_count, "committed": True}
        else:
            conn.rollback()
            return {"would_fix": fixed_count, "committed": False}


def fix_date_mentions_document_id(conn, commit: bool = False) -> dict:
    """
    Fix date_mentions.document_id drift from chunk_metadata.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE date_mentions dm
            SET document_id = cm.document_id
            FROM chunk_metadata cm
            WHERE cm.chunk_id = dm.chunk_id
              AND dm.document_id != cm.document_id
            """
        )
        fixed_count = cur.rowcount
        
        if commit:
            conn.commit()
            return {"fixed": fixed_count, "committed": True}
        else:
            conn.rollback()
            return {"would_fix": fixed_count, "committed": False}


def print_result(result: dict):
    """Print a check result."""
    status = "✅ PASS" if result["passed"] else "❌ FAIL"
    print(f"\n{status}: {result['check']}")
    
    for key, value in result.items():
        if key in ("check", "passed"):
            continue
        if key.endswith("_mismatches") or key.endswith("_orphans") or key.endswith("_violations"):
            if value:
                print(f"  {key}: (showing first {len(value)})")
                for item in value[:5]:
                    print(f"    - {item}")
        else:
            print(f"  {key}: {value}")


def main():
    ap = argparse.ArgumentParser(description="Validate data integrity")
    ap.add_argument("--check", choices=["entity", "date", "result_set", "count_mode", "all"], 
                   default="all", help="Which check to run")
    ap.add_argument("--sample", type=int, default=10000, help="Sample size for large tables")
    ap.add_argument("--fix", action="store_true", help="Attempt to fix drift issues")
    ap.add_argument("--commit", action="store_true", help="Actually commit fixes (default: dry-run)")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()
    
    conn = get_conn()
    results = []
    
    try:
        print(f"=== Data Integrity Validation ({datetime.now().isoformat()}) ===")
        
        if args.check in ("entity", "all"):
            result = check_entity_mentions_document_id(conn, args.sample)
            results.append(result)
            if not args.json:
                print_result(result)
            
            if args.fix and not result["passed"]:
                fix_result = fix_entity_mentions_document_id(conn, args.commit)
                print(f"  FIX: {fix_result}")
        
        if args.check in ("date", "all"):
            result = check_date_mentions_document_id(conn, args.sample)
            results.append(result)
            if not args.json:
                print_result(result)
            
            if args.fix and not result["passed"]:
                fix_result = fix_date_mentions_document_id(conn, args.commit)
                print(f"  FIX: {fix_result}")
        
        if args.check in ("result_set", "all"):
            result = check_result_sets_run_reference(conn)
            results.append(result)
            if not args.json:
                print_result(result)
        
        if args.check in ("count_mode", "all"):
            result = check_count_mode_null_result_set(conn)
            results.append(result)
            if not args.json:
                print_result(result)
        
        # Summary
        all_passed = all(r["passed"] for r in results)
        print(f"\n{'='*50}")
        print(f"Overall: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
        
        if args.json:
            import json
            print(json.dumps(results, indent=2, default=str))
        
        sys.exit(0 if all_passed else 1)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
