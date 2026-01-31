#!/usr/bin/env python3
"""
approve_plan.py --plan-id <id> [--reject --reason "..."]

Approve or reject a research plan.

Approve: Updates plan status to 'approved' and sets approved_at timestamp.
Reject:  Updates plan status to 'rejected' and stores rejection reason.

Usage:
    python scripts/approve_plan.py --plan-id 123
    python scripts/approve_plan.py --plan-id 123 --reject --reason "Ambiguous query"
    python scripts/approve_plan.py --plan-id 123 --show  # Show plan details
"""

import os
import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn
from psycopg2.extras import Json


def get_plan_details(conn, plan_id: int) -> dict:
    """Get plan details for display."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, session_id, user_utterance, plan_json, status, 
                   created_at, approved_at, executed_at
            FROM research_plans 
            WHERE id = %s
            """,
            (plan_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        
        plan_json = row[3]
        if isinstance(plan_json, str):
            plan_json = json.loads(plan_json)
        
        return {
            "id": row[0],
            "session_id": row[1],
            "user_utterance": row[2],
            "plan_json": plan_json,
            "status": row[4],
            "created_at": row[5],
            "approved_at": row[6],
            "executed_at": row[7],
        }


def show_plan(conn, plan_id: int) -> bool:
    """Display plan details including resolution info and execution mode."""
    plan = get_plan_details(conn, plan_id)
    if not plan:
        print(f"ERROR: Plan {plan_id} not found", file=sys.stderr)
        return False
    
    print(f"Plan ID: {plan['id']}")
    print(f"Session ID: {plan['session_id']}")
    print(f"Status: {plan['status']}")
    print(f"User Query: {plan['user_utterance']}")
    print(f"Created: {plan['created_at']}")
    
    if plan.get('approved_at'):
        print(f"Approved: {plan['approved_at']}")
    if plan.get('executed_at'):
        print(f"Executed: {plan['executed_at']}")
    
    plan_json = plan['plan_json']
    metadata = plan_json.get('_metadata', {})
    
    # Show execution mode (COUNT vs RETRIEVE)
    execution_mode = metadata.get('execution_mode')
    if execution_mode:
        print(f"Execution Mode: {execution_mode.upper()}")
    else:
        # Infer from primitives
        primitives = plan_json.get('query', {}).get('primitives', [])
        has_count = any(p.get('type') == 'COUNT' for p in primitives)
        predicted_mode = "COUNT" if has_count else "RETRIEVE"
        print(f"Predicted Execution Mode: {predicted_mode}")
    
    # Show clarification needs
    if plan_json.get('needs_clarification'):
        print(f"\n⚠️  NEEDS CLARIFICATION")
        choices = plan_json.get('choices', [])
        if choices:
            print("Choices:")
            for i, choice in enumerate(choices, 1):
                print(f"  {i}. {choice}")
    
    # Show resolution details with BEST-GUESS WARNINGS
    resolution = metadata.get('resolution', {})
    resolved_entities = resolution.get('resolved_entities', [])
    resolved_dates = resolution.get('resolved_dates', [])
    
    if resolved_entities:
        print(f"\n📍 Resolved Entities ({len(resolved_entities)}):")
        for e in resolved_entities:
            best_guess_warning = ""
            if e.get('is_best_guess'):
                best_guess_warning = " ⚠️  BEST-GUESS"
            confidence = e.get('confidence', 0)
            conf_warning = ""
            if confidence < 0.8:
                conf_warning = " (LOW CONFIDENCE)"
            
            print(f"  - \"{e.get('surface')}\" → {e.get('canonical_name')} (ID: {e.get('entity_id')})")
            print(f"    Confidence: {confidence:.2f}{conf_warning}{best_guess_warning}")
            
            # Show alternatives for best-guess
            alternatives = e.get('alternatives', [])
            if alternatives:
                alt_names = ", ".join([f"{a.get('canonical_name')} ({a.get('confidence', 0):.2f})" for a in alternatives[:3]])
                print(f"    Alternatives: {alt_names}")
    
    if resolved_dates:
        print(f"\n📅 Resolved Dates ({len(resolved_dates)}):")
        for d in resolved_dates:
            print(f"  - \"{d.get('expression')}\" → {d.get('start', '?')} to {d.get('end', '?')}")
    
    # Show resolution settings
    settings = resolution.get('entity_resolution_settings', {})
    if settings:
        print(f"\nResolution Settings:")
        print(f"  Best-guess mode: {settings.get('best_guess_mode', False)}")
        print(f"  Confidence threshold: {settings.get('confidence_threshold', 0.85)}")
    
    # Show primitives
    primitives = plan_json.get('query', {}).get('primitives', [])
    if primitives:
        print(f"\nPrimitives ({len(primitives)}):")
        for p in primitives[:10]:  # Show first 10
            print(f"  - {p.get('type', 'UNKNOWN')}: {json.dumps({k: v for k, v in p.items() if k != 'type'})}")
        if len(primitives) > 10:
            print(f"  ... and {len(primitives) - 10} more")
    
    return True


def approve_plan(conn, plan_id: int) -> bool:
    """Approve a plan by updating its status."""
    with conn.cursor() as cur:
        # Get plan details including plan_json
        cur.execute(
            "SELECT id, status, plan_json FROM research_plans WHERE id = %s",
            (plan_id,),
        )
        row = cur.fetchone()
        if not row:
            print(f"ERROR: Plan {plan_id} not found", file=sys.stderr)
            return False
        
        current_status = row[1]
        plan_json = row[2]
        
        # Parse plan_json if needed
        if isinstance(plan_json, str):
            try:
                plan_json = json.loads(plan_json)
            except json.JSONDecodeError:
                plan_json = {}
        
        # Check if plan needs clarification
        if isinstance(plan_json, dict) and plan_json.get('needs_clarification'):
            choices = plan_json.get('choices', [])
            print(f"ERROR: Plan {plan_id} needs clarification before it can be approved.", file=sys.stderr)
            if choices:
                print("Please resolve clarification using: python scripts/clarify_plan.py --plan {plan_id}", file=sys.stderr)
                print(f"Choices: {', '.join(choices[:3])}{'...' if len(choices) > 3 else ''}", file=sys.stderr)
            return False
        
        if current_status == "approved":
            print(f"Plan {plan_id} is already approved", file=sys.stderr)
            return True
        
        if current_status not in ("proposed", "superseded", "rejected"):
            print(f"ERROR: Plan {plan_id} status is '{current_status}', cannot approve (must be 'proposed', 'superseded', or 'rejected')", file=sys.stderr)
            return False
        
        # Update status to approved
        cur.execute(
            """
            UPDATE research_plans
            SET status = 'approved',
                approved_at = now()
            WHERE id = %s
            """,
            (plan_id,),
        )
        conn.commit()
        print(f"✅ Plan {plan_id} approved successfully", file=sys.stderr)
        return True


def reject_plan(conn, plan_id: int, reason: str) -> bool:
    """Reject a plan with a reason."""
    with conn.cursor() as cur:
        # Check current status
        cur.execute(
            "SELECT id, status, plan_json FROM research_plans WHERE id = %s",
            (plan_id,),
        )
        row = cur.fetchone()
        if not row:
            print(f"ERROR: Plan {plan_id} not found", file=sys.stderr)
            return False
        
        current_status = row[1]
        plan_json = row[2]
        
        # Parse plan_json if needed
        if isinstance(plan_json, str):
            try:
                plan_json = json.loads(plan_json)
            except json.JSONDecodeError:
                plan_json = {}
        
        if current_status == "rejected":
            print(f"Plan {plan_id} is already rejected", file=sys.stderr)
            return True
        
        if current_status == "executed":
            print(f"ERROR: Plan {plan_id} is already executed, cannot reject", file=sys.stderr)
            return False
        
        # Store rejection reason in plan_json._metadata
        if not isinstance(plan_json, dict):
            plan_json = {}
        
        plan_json.setdefault("_metadata", {})
        plan_json["_metadata"]["rejection_reason"] = reason
        plan_json["_metadata"]["rejected_at"] = __import__("datetime").datetime.now().isoformat()
        
        # Update status to rejected and store reason in plan_json
        cur.execute(
            """
            UPDATE research_plans
            SET status = 'rejected',
                plan_json = %s
            WHERE id = %s
            """,
            (Json(plan_json), plan_id),
        )
        conn.commit()
        print(f"❌ Plan {plan_id} rejected. Reason: {reason}", file=sys.stderr)
        return True


def main():
    ap = argparse.ArgumentParser(description="Approve or reject a research plan")
    ap.add_argument("--plan-id", type=int, required=True, help="ID of plan")
    ap.add_argument("--reject", action="store_true", help="Reject the plan instead of approving")
    ap.add_argument("--reason", type=str, default="", help="Rejection reason (required with --reject)")
    ap.add_argument("--show", action="store_true", help="Show plan details without changing status")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        if args.show:
            success = show_plan(conn, args.plan_id)
        elif args.reject:
            if not args.reason.strip():
                print("ERROR: --reason is required when rejecting a plan", file=sys.stderr)
                sys.exit(1)
            success = reject_plan(conn, args.plan_id, args.reason.strip())
        else:
            success = approve_plan(conn, args.plan_id)
        
        sys.exit(0 if success else 1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
