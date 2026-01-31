#!/usr/bin/env python3
"""
admin_resolution_audit.py

Admin helper to query and audit entity/date resolution metadata in plans.

Usage:
    python scripts/admin_resolution_audit.py --best-guess     # Plans with best-guess entity resolutions
    python scripts/admin_resolution_audit.py --low-confidence # Plans with resolutions below threshold
    python scripts/admin_resolution_audit.py --stale-dates    # Plans with relative dates resolved long ago
    python scripts/admin_resolution_audit.py --plan-id 123    # Show resolution details for specific plan
    python scripts/admin_resolution_audit.py --recent 10      # Show 10 most recent plans with resolutions
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn


def get_plans_with_resolution_metadata(conn, limit: int = 100):
    """Get plans that have resolution metadata."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, session_id, user_utterance, plan_json, status, created_at
            FROM research_plans
            WHERE plan_json->'_metadata'->'resolution' IS NOT NULL
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "user_utterance": row[2],
                "plan_json": row[3],
                "status": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]


def get_plans_with_best_guess(conn, limit: int = 50):
    """Find plans that used best-guess entity resolution."""
    plans = get_plans_with_resolution_metadata(conn, limit=500)
    best_guess_plans = []
    
    for plan in plans:
        plan_json = plan["plan_json"]
        if isinstance(plan_json, str):
            plan_json = json.loads(plan_json)
        
        resolution = plan_json.get("_metadata", {}).get("resolution", {})
        entities = resolution.get("resolved_entities", [])
        
        best_guess_entities = [e for e in entities if e.get("is_best_guess")]
        if best_guess_entities:
            best_guess_plans.append({
                **plan,
                "best_guess_entities": best_guess_entities,
            })
            if len(best_guess_plans) >= limit:
                break
    
    return best_guess_plans


def get_plans_with_low_confidence(conn, threshold: float = 0.7, limit: int = 50):
    """Find plans with entity resolutions below confidence threshold."""
    plans = get_plans_with_resolution_metadata(conn, limit=500)
    low_conf_plans = []
    
    for plan in plans:
        plan_json = plan["plan_json"]
        if isinstance(plan_json, str):
            plan_json = json.loads(plan_json)
        
        resolution = plan_json.get("_metadata", {}).get("resolution", {})
        entities = resolution.get("resolved_entities", [])
        
        low_conf_entities = [
            e for e in entities 
            if e.get("confidence") is not None and e["confidence"] < threshold
        ]
        if low_conf_entities:
            low_conf_plans.append({
                **plan,
                "low_confidence_entities": low_conf_entities,
            })
            if len(low_conf_plans) >= limit:
                break
    
    return low_conf_plans


def get_plans_with_stale_dates(conn, days_old: int = 30, limit: int = 50):
    """
    Find plans where relative dates were resolved more than N days ago.
    
    This could indicate drift if the plan was meant to reflect "current" timeframe
    but hasn't been re-executed.
    """
    plans = get_plans_with_resolution_metadata(conn, limit=500)
    cutoff = datetime.now() - timedelta(days=days_old)
    stale_plans = []
    
    for plan in plans:
        plan_json = plan["plan_json"]
        if isinstance(plan_json, str):
            plan_json = json.loads(plan_json)
        
        resolution = plan_json.get("_metadata", {}).get("resolution", {})
        resolved_at_str = resolution.get("resolved_at")
        resolved_dates = resolution.get("resolved_dates", [])
        
        if not resolved_dates or not resolved_at_str:
            continue
        
        try:
            resolved_at = datetime.fromisoformat(resolved_at_str.replace("Z", "+00:00"))
            if resolved_at.tzinfo:
                resolved_at = resolved_at.replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        
        if resolved_at < cutoff:
            stale_plans.append({
                **plan,
                "resolved_at": resolved_at,
                "days_old": (datetime.now() - resolved_at).days,
                "resolved_dates": resolved_dates,
            })
            if len(stale_plans) >= limit:
                break
    
    return stale_plans


def get_plan_resolution_details(conn, plan_id: int):
    """Get detailed resolution metadata for a specific plan."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, session_id, user_utterance, plan_json, status, created_at,
                   executed_at, retrieval_run_id, result_set_id
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
            "executed_at": row[6],
            "retrieval_run_id": row[7],
            "result_set_id": row[8],
        }


def print_best_guess_report(plans):
    """Print report of plans with best-guess resolutions."""
    print(f"\n=== Plans with Best-Guess Entity Resolutions ({len(plans)}) ===\n")
    
    for plan in plans:
        print(f"Plan {plan['id']} [{plan['status']}]")
        print(f"  Query: {plan['user_utterance'][:80]}...")
        print(f"  Created: {plan['created_at']}")
        print(f"  Best-guess entities:")
        for e in plan["best_guess_entities"]:
            alts = e.get("alternatives", [])
            alt_str = ", ".join([f"{a['canonical_name']} ({a.get('confidence', '?'):.2f})" for a in alts[:2]])
            print(f"    - \"{e['surface']}\" → {e['canonical_name']} (conf: {e.get('confidence', '?'):.2f})")
            if alt_str:
                print(f"      Alternatives: {alt_str}")
        print()


def print_low_confidence_report(plans, threshold: float):
    """Print report of plans with low-confidence resolutions."""
    print(f"\n=== Plans with Low-Confidence Resolutions (< {threshold}) ({len(plans)}) ===\n")
    
    for plan in plans:
        print(f"Plan {plan['id']} [{plan['status']}]")
        print(f"  Query: {plan['user_utterance'][:80]}...")
        print(f"  Low-confidence entities:")
        for e in plan["low_confidence_entities"]:
            print(f"    - \"{e['surface']}\" → {e['canonical_name']} (conf: {e.get('confidence', '?'):.2f})")
        print()


def print_stale_dates_report(plans):
    """Print report of plans with stale date resolutions."""
    print(f"\n=== Plans with Stale Date Resolutions ({len(plans)}) ===\n")
    
    for plan in plans:
        print(f"Plan {plan['id']} [{plan['status']}]")
        print(f"  Query: {plan['user_utterance'][:80]}...")
        print(f"  Resolved: {plan['resolved_at']} ({plan['days_old']} days ago)")
        print(f"  Date expressions:")
        for d in plan["resolved_dates"]:
            print(f"    - \"{d['expression']}\" → {d.get('start', '?')} to {d.get('end', '?')}")
        print()


def print_plan_details(plan):
    """Print detailed resolution information for a single plan."""
    print(f"\n=== Plan {plan['id']} Details ===\n")
    print(f"Session: {plan['session_id']}")
    print(f"Status: {plan['status']}")
    print(f"Created: {plan['created_at']}")
    if plan.get("executed_at"):
        print(f"Executed: {plan['executed_at']}")
        print(f"Run ID: {plan.get('retrieval_run_id')}")
        print(f"Result Set ID: {plan.get('result_set_id')}")
    print(f"\nQuery: {plan['user_utterance']}")
    
    plan_json = plan["plan_json"]
    metadata = plan_json.get("_metadata", {})  # includes resolution + execution + errors
    resolution = plan_json.get("_metadata", {}).get("resolution", {})

    # Show last error if present (useful for smoke testing failure paths)
    last_error = metadata.get("last_error")
    if last_error:
        print("\nLast Error:")
        print(f"  Type: {last_error.get('error_type')}")
        print(f"  Message: {last_error.get('error_message')}")
        print(f"  Failed at: {last_error.get('failed_at')}")
        if last_error.get("partial_run_id"):
            print(f"  Partial run ID: {last_error.get('partial_run_id')}")
    
    if not resolution:
        print("\nNo resolution metadata found.")
        return
    
    print(f"\nResolved at: {resolution.get('resolved_at', 'unknown')}")
    
    settings = resolution.get("entity_resolution_settings", {})
    if settings:
        print(f"Best-guess mode: {settings.get('best_guess_mode', False)}")
        print(f"Confidence threshold: {settings.get('confidence_threshold', 0.85)}")
    
    entities = resolution.get("resolved_entities", [])
    if entities:
        print(f"\nResolved Entities ({len(entities)}):")
        for e in entities:
            flags = []
            if e.get("is_best_guess"):
                flags.append("BEST-GUESS")
            flags_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  - \"{e['surface']}\" → {e['canonical_name']} (ID: {e['entity_id']}){flags_str}")
            print(f"    Confidence: {e.get('confidence', '?')}, Method: {e.get('match_method', '?')}")
            alts = e.get("alternatives", [])
            if alts:
                print(f"    Alternatives: {', '.join([a['canonical_name'] for a in alts])}")
    
    dates = resolution.get("resolved_dates", [])
    if dates:
        print(f"\nResolved Dates ({len(dates)}):")
        for d in dates:
            print(f"  - \"{d['expression']}\" → {d.get('start', '?')} to {d.get('end', '?')}")
    
    # Check for execution history
    executions = plan_json.get("_metadata", {}).get("executions", [])
    if executions:
        print(f"\nExecution History ({len(executions)}):")
        for ex in executions:
            print(f"  - Run {ex.get('retrieval_run_id')} (superseded at {ex.get('superseded_at', '?')})")


def main():
    ap = argparse.ArgumentParser(description="Audit entity/date resolution metadata in plans")
    ap.add_argument("--best-guess", action="store_true", help="List plans with best-guess entity resolutions")
    ap.add_argument("--low-confidence", action="store_true", help="List plans with resolutions below threshold")
    ap.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for --low-confidence (default: 0.7)")
    ap.add_argument("--stale-dates", action="store_true", help="List plans with date resolutions older than N days")
    ap.add_argument("--days", type=int, default=30, help="Days threshold for --stale-dates (default: 30)")
    ap.add_argument("--plan-id", type=int, help="Show resolution details for a specific plan")
    ap.add_argument("--recent", type=int, help="List N most recent plans with resolution metadata")
    ap.add_argument("--json", action="store_true", help="Output as JSON instead of formatted report")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        if args.plan_id:
            plan = get_plan_resolution_details(conn, args.plan_id)
            if not plan:
                print(f"ERROR: Plan {args.plan_id} not found", file=sys.stderr)
                sys.exit(1)
            if args.json:
                print(json.dumps(plan, indent=2, default=str))
            else:
                print_plan_details(plan)
        
        elif args.best_guess:
            plans = get_plans_with_best_guess(conn)
            if args.json:
                print(json.dumps(plans, indent=2, default=str))
            else:
                print_best_guess_report(plans)
        
        elif args.low_confidence:
            plans = get_plans_with_low_confidence(conn, args.threshold)
            if args.json:
                print(json.dumps(plans, indent=2, default=str))
            else:
                print_low_confidence_report(plans, args.threshold)
        
        elif args.stale_dates:
            plans = get_plans_with_stale_dates(conn, args.days)
            if args.json:
                print(json.dumps(plans, indent=2, default=str))
            else:
                print_stale_dates_report(plans)
        
        elif args.recent:
            plans = get_plans_with_resolution_metadata(conn, limit=args.recent)
            if args.json:
                print(json.dumps(plans, indent=2, default=str))
            else:
                print(f"\n=== {len(plans)} Most Recent Plans with Resolution Metadata ===\n")
                for plan in plans:
                    plan_json = plan["plan_json"]
                    if isinstance(plan_json, str):
                        plan_json = json.loads(plan_json)
                    resolution = plan_json.get("_metadata", {}).get("resolution", {})
                    n_entities = len(resolution.get("resolved_entities", []))
                    n_dates = len(resolution.get("resolved_dates", []))
                    print(f"Plan {plan['id']} [{plan['status']}] - {n_entities} entities, {n_dates} dates")
                    print(f"  {plan['user_utterance'][:70]}...")
        
        else:
            ap.print_help()
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
