#!/usr/bin/env python3
"""
approve_plan.py --plan-id <id>

Approve a research plan for execution.
Updates plan status to 'approved' and sets approved_at timestamp.
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn

def approve_plan(conn, plan_id: int):
    """Approve a plan by updating its status."""
    with conn.cursor() as cur:
        # Check current status
        cur.execute(
            "SELECT id, status FROM research_plans WHERE id = %s",
            (plan_id,),
        )
        row = cur.fetchone()
        if not row:
            print(f"ERROR: Plan {plan_id} not found", file=sys.stderr)
            return False
        
        current_status = row[1]
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
        print(f"Plan {plan_id} approved successfully", file=sys.stderr)
        return True

def main():
    ap = argparse.ArgumentParser(description="Approve a research plan")
    ap.add_argument("--plan-id", type=int, required=True, help="ID of plan to approve")
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        success = approve_plan(conn, args.plan_id)
        sys.exit(0 if success else 1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
