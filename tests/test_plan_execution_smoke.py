#!/usr/bin/env python3
"""
Smoke test for plan creation, revision, approval, and execution workflow.

Tests:
1. Create a session
2. Create a plan (plan_query.py)
3. Revise the plan (revise_plan.py)
4. Approve the plan (approve_plan.py)
5. Execute the plan (execute_plan.py)
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn

def run_command(cmd: list, description: str, env: Optional[Dict[str, str]] = None) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Running: {description}", file=sys.stderr)
    print(f"Command: {' '.join(cmd)}", file=sys.stderr)
    if env and "NEH_DISALLOW_SESSION_LOOKUPS" in env:
        print(f"Environment: NEH_DISALLOW_SESSION_LOOKUPS={env['NEH_DISALLOW_SESSION_LOOKUPS']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env if env else None,
    )
    
    if result.stdout:
        print("STDOUT:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
    if result.stderr:
        print("STDERR:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    
    return result.returncode, result.stdout, result.stderr

def get_plan_id_from_output(output: str) -> int:
    """Extract plan ID from plan_query.py or revise_plan.py output."""
    # plan_query.py: "✅ Plan saved with ID: {plan_id} (status: proposed)"
    # revise_plan.py: "✅ Revised plan saved with ID: {revised_plan_id} (status: proposed, parent: {args.plan_id})"
    import re
    match = re.search(r'saved with ID:\s*(\d+)', output, re.IGNORECASE)
    if not match:
        match = re.search(r'Plan ID:\s*(\d+)', output, re.IGNORECASE)
    if not match:
        match = re.search(r'plan_id[=:]\s*(\d+)', output, re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract plan ID from output: {output[:500]}")

def get_session_id_from_output(output: str) -> int:
    """Extract session ID from new_session.py output."""
    # new_session.py prints just the session ID as a number on stdout
    import re
    # First try to find a standalone number (session ID is printed alone)
    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.isdigit():
            return int(line)
    # Fallback: look for "Session ID: <number>" or "session_id[=:]<number>"
    match = re.search(r'Session ID:\s*(\d+)', output, re.IGNORECASE)
    if not match:
        match = re.search(r'session_id[=:]\s*(\d+)', output, re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract session ID from output: {output[:500]}")

def test_plan_execution_workflow():
    """Test the full plan execution workflow."""
    print("="*60, file=sys.stderr)
    print("PLAN EXECUTION SMOKE TEST", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    # Step 1: Create two sessions for poison pill test
    import time
    timestamp = int(time.time())
    session_a_label = f"Plan execution smoke test A {timestamp}"
    session_b_label = f"Plan execution smoke test B {timestamp}"
    
    print("\n[1/8] Creating Session A...", file=sys.stderr)
    returncode, stdout, stderr = run_command(
        ["python", "scripts/new_session.py", "--label", session_a_label],
        "Create Session A"
    )
    if returncode != 0:
        print(f"ERROR: Failed to create Session A", file=sys.stderr)
        return False
    
    session_a_id = get_session_id_from_output(stdout + stderr)
    print(f"✓ Session A created: ID={session_a_id}", file=sys.stderr)
    
    print("\n[2/8] Creating Session B...", file=sys.stderr)
    returncode, stdout, stderr = run_command(
        ["python", "scripts/new_session.py", "--label", session_b_label],
        "Create Session B"
    )
    if returncode != 0:
        print(f"ERROR: Failed to create Session B", file=sys.stderr)
        return False
    
    session_b_id = get_session_id_from_output(stdout + stderr)
    print(f"✓ Session B created: ID={session_b_id}", file=sys.stderr)
    
    # Step 3: Create a plan in Session A with explicit venona scope
    print("\n[3/8] Creating plan in Session A (with venona scope)...", file=sys.stderr)
    returncode, stdout, stderr = run_command(
        ["python", "scripts/plan_query.py", "--session", str(session_a_id), "--text", "find mentions of Treasury in venona"],
        "Create plan in Session A"
    )
    if returncode != 0:
        print(f"ERROR: Failed to create plan", file=sys.stderr)
        return False
    
    plan_id = get_plan_id_from_output(stdout + stderr)
    print(f"✓ Plan created: ID={plan_id} (session={session_a_id})", file=sys.stderr)
    
    # Verify plan has execution envelope with explicit parameters
    print("\n[Verification] Checking plan execution envelope...", file=sys.stderr)
    conn_check = get_conn()
    try:
        with conn_check.cursor() as cur:
            cur.execute(
                "SELECT plan_json FROM research_plans WHERE id = %s",
                (plan_id,),
            )
            row = cur.fetchone()
            if row:
                plan_json = row[0]
                envelope = plan_json.get("execution_envelope")
                if not envelope:
                    print("ERROR: Plan missing execution_envelope", file=sys.stderr)
                    return False
                
                required_keys = ["collection_scope", "chunk_pipeline_version", "retrieval_config", "k"]
                missing = [k for k in required_keys if k not in envelope]
                if missing:
                    print(f"ERROR: Plan execution_envelope missing keys: {missing}", file=sys.stderr)
                    return False
                
                collection_scope = envelope.get('collection_scope')
                print(f"  ✓ collection_scope: {collection_scope}", file=sys.stderr)
                print(f"  ✓ chunk_pipeline_version: {envelope.get('chunk_pipeline_version')}", file=sys.stderr)
                print(f"  ✓ k: {envelope.get('k')}", file=sys.stderr)
                print(f"  ✓ retrieval_config: {envelope.get('retrieval_config')}", file=sys.stderr)
                
                # Assert correct scope (poison pill test)
                expected_scope = ["venona"]  # Plan was created with "in venona"
                if collection_scope != expected_scope:
                    print(f"ERROR: Expected collection_scope={expected_scope}, got {collection_scope}", file=sys.stderr)
                    return False
                print(f"  ✓ collection_scope matches expected: {expected_scope}", file=sys.stderr)
            else:
                print(f"ERROR: Plan {plan_id} not found", file=sys.stderr)
                return False
    finally:
        conn_check.close()
    
    # Step 4: Revise the plan (still in Session A context)
    print("\n[4/8] Revising plan...", file=sys.stderr)
    returncode, stdout, stderr = run_command(
        ["python", "scripts/revise_plan.py", "--plan-id", str(plan_id), "--text", "broaden this to include state department"],
        "Revise plan"
    )
    if returncode != 0:
        print(f"ERROR: Failed to revise plan", file=sys.stderr)
        return False
    
    revised_plan_id = get_plan_id_from_output(stdout + stderr)
    print(f"✓ Plan revised: ID={revised_plan_id} (parent: {plan_id}, session={session_a_id})", file=sys.stderr)
    
    # Verify envelope preservation after revision (Upgrade 2)
    print("\n[Verification] Checking envelope preservation after revision...", file=sys.stderr)
    conn_check2 = get_conn()
    try:
        with conn_check2.cursor() as cur:
            cur.execute(
                "SELECT plan_json FROM research_plans WHERE id = %s",
                (revised_plan_id,),
            )
            row = cur.fetchone()
            if row:
                revised_plan_json = row[0]
                revised_envelope = revised_plan_json.get("execution_envelope", {})
                if not revised_envelope:
                    print("ERROR: Revised plan missing execution_envelope", file=sys.stderr)
                    return False
                
                # Check that envelope fields are preserved
                if revised_envelope.get("collection_scope") != ["venona"]:
                    print(f"ERROR: Revised plan collection_scope changed: {revised_envelope.get('collection_scope')}", file=sys.stderr)
                    return False
                if revised_envelope.get("chunk_pipeline_version") != "chunk_v1_full":
                    print(f"ERROR: Revised plan chunk_pipeline_version changed: {revised_envelope.get('chunk_pipeline_version')}", file=sys.stderr)
                    return False
                if revised_envelope.get("k") != 20:
                    print(f"ERROR: Revised plan k changed: {revised_envelope.get('k')}", file=sys.stderr)
                    return False
                
                print(f"  ✓ collection_scope preserved: {revised_envelope.get('collection_scope')}", file=sys.stderr)
                print(f"  ✓ chunk_pipeline_version preserved: {revised_envelope.get('chunk_pipeline_version')}", file=sys.stderr)
                print(f"  ✓ k preserved: {revised_envelope.get('k')}", file=sys.stderr)
                print(f"  ✓ retrieval_config preserved: {revised_envelope.get('retrieval_config')}", file=sys.stderr)
                
                # Drift detector: Verify primitives + envelope agree (Improvement 1)
                print("\n[Drift Detector] Verifying primitives and envelope agree...", file=sys.stderr)
                query = revised_plan_json.get("query", {})
                primitives = query.get("primitives", [])
                
                # Check that FILTER_COLLECTION primitive still exists
                has_filter = any(
                    isinstance(p, dict) and 
                    p.get("type") == "FILTER_COLLECTION" and 
                    p.get("slug") == "venona"
                    for p in primitives
                )
                if not has_filter:
                    print("ERROR: Revised plan lost FILTER_COLLECTION primitive (slug='venona')", file=sys.stderr)
                    return False
                print("  ✓ FILTER_COLLECTION primitive preserved: slug='venona'", file=sys.stderr)
                
                # Derive scope from primitives and compare to envelope
                from retrieval.plan_validation import derive_collection_scope_from_primitives
                derived_scope = derive_collection_scope_from_primitives(primitives)
                envelope_scope = revised_envelope.get("collection_scope")
                
                if derived_scope != envelope_scope:
                    print(f"ERROR: Envelope-primitive drift detected:", file=sys.stderr)
                    print(f"  - Primitives derive: {derived_scope}", file=sys.stderr)
                    print(f"  - Envelope says: {envelope_scope}", file=sys.stderr)
                    return False
                print(f"  ✓ Envelope matches derived scope: {envelope_scope}", file=sys.stderr)
            else:
                print(f"ERROR: Revised plan {revised_plan_id} not found", file=sys.stderr)
                return False
    finally:
        conn_check2.close()
    
    # Step 5: Create and execute a conflicting plan in Session B (Upgrade 1: Poison pill)
    print("\n[5a/8] Creating conflicting plan in Session B (vassiliev scope)...", file=sys.stderr)
    returncode, stdout, stderr = run_command(
        ["python", "scripts/plan_query.py", "--session", str(session_b_id), "--text", "find mentions of Treasury in vassiliev"],
        "Create conflicting plan in Session B"
    )
    if returncode != 0:
        print(f"ERROR: Failed to create conflicting plan", file=sys.stderr)
        return False
    
    conflicting_plan_id = get_plan_id_from_output(stdout + stderr)
    print(f"✓ Conflicting plan created: ID={conflicting_plan_id}", file=sys.stderr)
    
    # Approve and execute the conflicting plan
    print("\n[5b/8] Approving conflicting plan...", file=sys.stderr)
    returncode, stdout, stderr = run_command(
        ["python", "scripts/approve_plan.py", "--plan-id", str(conflicting_plan_id)],
        "Approve conflicting plan"
    )
    if returncode != 0:
        print(f"ERROR: Failed to approve conflicting plan", file=sys.stderr)
        return False
    
    print("\n[5c/8] Executing conflicting plan...", file=sys.stderr)
    returncode, stdout, stderr = run_command(
        ["python", "scripts/execute_plan.py", "--plan-id", str(conflicting_plan_id), "--create-result-set"],
        "Execute conflicting plan"
    )
    if returncode != 0:
        print(f"ERROR: Failed to execute conflicting plan", file=sys.stderr)
        return False
    
    print(f"✓ Conflicting plan executed (Session B now has vassiliev scope)", file=sys.stderr)
    
    # Step 6: Approve the original revised plan
    print("\n[6/8] Approving revised plan...", file=sys.stderr)
    returncode, stdout, stderr = run_command(
        ["python", "scripts/approve_plan.py", "--plan-id", str(revised_plan_id)],
        "Approve plan"
    )
    if returncode != 0:
        print(f"ERROR: Failed to approve plan", file=sys.stderr)
        return False
    
    print(f"✓ Plan approved: ID={revised_plan_id}", file=sys.stderr)
    
    # Step 7: Execute the plan (poison pill: should use Session A scope even if Session B was just used)
    # Improvement 2: Run with NEH_DISALLOW_SESSION_LOOKUPS=1 to prove no session queries
    print("\n[7/8] Executing plan (poison pill: plan should be self-contained)...", file=sys.stderr)
    env = os.environ.copy()
    env["NEH_DISALLOW_SESSION_LOOKUPS"] = "1"
    returncode, stdout, stderr = run_command(
        ["python", "scripts/execute_plan.py", "--plan-id", str(revised_plan_id), "--create-result-set"],
        "Execute plan",
        env=env
    )
    if returncode != 0:
        print(f"ERROR: Failed to execute plan", file=sys.stderr)
        return False
    
    print(f"✓ Plan executed: ID={revised_plan_id}", file=sys.stderr)
    
    # Step 8: Verify final state and poison pill
    print("\n[8/8] Verifying final plan state and poison pill...", file=sys.stderr)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, approved_at, executed_at, retrieval_run_id, result_set_id
                FROM research_plans
                WHERE id = %s
                """,
                (revised_plan_id,),
            )
            row = cur.fetchone()
            if row:
                status, approved_at, executed_at, run_id, result_set_id = row
                print(f"  Status: {status}", file=sys.stderr)
                print(f"  Approved at: {approved_at}", file=sys.stderr)
                print(f"  Executed at: {executed_at}", file=sys.stderr)
                print(f"  Retrieval run ID: {run_id}", file=sys.stderr)
                print(f"  Result set ID: {result_set_id}", file=sys.stderr)
                
                if status != "executed":
                    print(f"ERROR: Expected status 'executed', got '{status}'", file=sys.stderr)
                    return False
                if not approved_at:
                    print(f"ERROR: approved_at is NULL", file=sys.stderr)
                    return False
                if not executed_at:
                    print(f"ERROR: executed_at is NULL", file=sys.stderr)
                    return False
                if not run_id:
                    print(f"ERROR: retrieval_run_id is NULL", file=sys.stderr)
                    return False
                
                print("✓ All checks passed!", file=sys.stderr)
                
                # Poison pill test: verify plan scope is self-contained AND correct
                print("\n[Poison Pill Test] Verifying plan scope is self-contained and correct...", file=sys.stderr)
                with conn.cursor() as cur2:
                    cur2.execute(
                        "SELECT plan_json FROM research_plans WHERE id = %s",
                        (revised_plan_id,),
                    )
                    row2 = cur2.fetchone()
                    if row2:
                        plan_json = row2[0]
                        envelope = plan_json.get("execution_envelope", {})
                        collection_scope = envelope.get("collection_scope")
                        if collection_scope is None:
                            print("ERROR: Plan missing collection_scope in execution_envelope", file=sys.stderr)
                            return False
                        
                        expected_scope = ["venona"]
                        if collection_scope != expected_scope:
                            print(f"ERROR: Expected collection_scope={expected_scope}, got {collection_scope}", file=sys.stderr)
                            return False
                        
                        print(f"  ✓ Plan has explicit collection_scope: {collection_scope}", file=sys.stderr)
                        # Note: "self-contained" is proven by execution with NEH_DISALLOW_SESSION_LOOKUPS=1
                        
                        # Verify actual results are from venona only (Upgrade 3: Schema-resilient)
                        if result_set_id:
                            # Check if result_set_items table exists (normalized schema)
                            cur2.execute("""
                                SELECT EXISTS (
                                    SELECT 1 FROM information_schema.tables 
                                    WHERE table_name = 'result_set_items'
                                )
                            """)
                            has_items_table = cur2.fetchone()[0]
                            
                            if has_items_table:
                                # Use normalized table
                                cur2.execute(
                                    """
                                    SELECT COUNT(*) FILTER (WHERE cm.collection_slug <> 'venona') AS non_venona,
                                           COUNT(*) AS total
                                    FROM result_set_items rsi
                                    JOIN chunk_metadata cm ON cm.chunk_id = rsi.chunk_id
                                    WHERE rsi.result_set_id = %s
                                    """,
                                    (result_set_id,),
                                )
                            else:
                                # Fall back to array unnest
                                cur2.execute(
                                    """
                                    SELECT COUNT(*) FILTER (WHERE cm.collection_slug <> 'venona') AS non_venona,
                                           COUNT(*) AS total
                                    FROM result_sets rs
                                    CROSS JOIN LATERAL unnest(rs.chunk_ids) AS u(chunk_id)
                                    JOIN chunk_metadata cm ON cm.chunk_id = u.chunk_id
                                    WHERE rs.id = %s
                                    """,
                                    (result_set_id,),
                                )
                            
                            row3 = cur2.fetchone()
                            if row3:
                                non_venona, total = row3
                                if non_venona > 0:
                                    print(f"ERROR: Result set contains {non_venona} non-venona chunks out of {total} total", file=sys.stderr)
                                    return False
                                print(f"  ✓ All {total} chunks in result set are from venona collection", file=sys.stderr)
                
                # Improvement 2: Verify no session queries during execution
                # (This is proven by the fact that execution succeeded with NEH_DISALLOW_SESSION_LOOKUPS=1)
                print("  ✓ Plan execution is self-contained (no session queries)", file=sys.stderr)
                
                # Improvement 3: Print artifact summary for easy copy/paste
                print("\n" + "="*60, file=sys.stderr)
                print("ARTIFACT SUMMARY (for debugging)", file=sys.stderr)
                print("="*60, file=sys.stderr)
                print(f"Session A ID: {session_a_id}", file=sys.stderr)
                print(f"Session B ID: {session_b_id}", file=sys.stderr)
                print(f"Original Plan ID: {plan_id}", file=sys.stderr)
                print(f"Revised Plan ID: {revised_plan_id}", file=sys.stderr)
                print(f"Conflicting Plan ID: {conflicting_plan_id}", file=sys.stderr)
                print(f"Retrieval Run ID: {run_id}", file=sys.stderr)
                if result_set_id:
                    print(f"Result Set ID: {result_set_id}", file=sys.stderr)
                print("="*60, file=sys.stderr)
                
                return True
            else:
                print(f"ERROR: Plan {revised_plan_id} not found", file=sys.stderr)
                return False
    finally:
        conn.close()

def main():
    """Run the smoke test."""
    try:
        success = test_plan_execution_workflow()
        if success:
            print("\n" + "="*60, file=sys.stderr)
            print("✅ SMOKE TEST PASSED", file=sys.stderr)
            print("="*60, file=sys.stderr)
            sys.exit(0)
        else:
            print("\n" + "="*60, file=sys.stderr)
            print("❌ SMOKE TEST FAILED", file=sys.stderr)
            print("="*60, file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Smoke test failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
