"""
Plan execution service.
Wraps scripts/execute_plan.py functionality.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

from app.services.db import get_conn
from app.services.evidence import build_evidence_refs_from_chunk

REPO_ROOT = Path(__file__).parent.parent.parent.parent
EXECUTE_PLAN_SCRIPT = REPO_ROOT / "scripts" / "execute_plan.py"


def execute_plan(conn, plan_id: int) -> Dict[str, Any]:
    """
    Execute an approved plan.
    
    Calls the existing execute_plan.py script via subprocess.
    Returns execution results including the result_set.
    
    Args:
        conn: Database connection (passed to build result_set response)
        plan_id: The plan to execute
    
    Returns:
        Dict with:
        - plan: updated plan data
        - result_set: ResultSetResponse-shaped dict
    """
    # Run execution script
    result = subprocess.run(
        [
            sys.executable,
            str(EXECUTE_PLAN_SCRIPT),
            "--plan-id", str(plan_id),
            "--create-result-set",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout for execution
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Plan execution failed: {result.stderr}")
    
    # Get the result set from the database
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT result_set_id, retrieval_run_id
            FROM research_plans
            WHERE id = %s
            """,
            (plan_id,),
        )
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError("Execution completed but no result_set_id was set")
        
        result_set_id, run_id = row
        
        # Get result set details
        cur.execute(
            """
            SELECT id, name, retrieval_run_id, chunk_ids, created_at
            FROM result_sets
            WHERE id = %s
            """,
            (result_set_id,),
        )
        rs_row = cur.fetchone()
        if not rs_row:
            raise RuntimeError(f"Result set {result_set_id} not found")
        
        rs_id, name, run_id, chunk_ids, created_at = rs_row
        
        # Build items from chunk evidence
        items = []
        document_ids = set()
        
        if chunk_ids:
            cur.execute(
                """
                SELECT 
                    e.chunk_id,
                    e.rank,
                    e.score_lex,
                    e.score_vec,
                    e.score_hybrid,
                    e.matched_lexemes,
                    e.highlight,
                    c.text
                FROM retrieval_run_chunk_evidence e
                JOIN chunks c ON c.id = e.chunk_id
                WHERE e.retrieval_run_id = %s
                    AND e.chunk_id = ANY(%s)
                ORDER BY e.rank
                """,
                (run_id, chunk_ids),
            )
            evidence_rows = cur.fetchall()
            
            for row in evidence_rows:
                chunk_id, rank, score_lex, score_vec, score_hybrid, lexemes, highlight, text = row
                
                # Build evidence refs
                evidence_refs = build_evidence_refs_from_chunk(conn, chunk_id)
                
                for ref in evidence_refs:
                    document_ids.add(ref["document_id"])
                
                items.append({
                    "id": f"chunk-{chunk_id}",
                    "kind": "chunk",
                    "rank": rank,
                    "text": text[:500] if text else "",
                    "chunk_id": chunk_id,
                    "document_id": evidence_refs[0]["document_id"] if evidence_refs else None,
                    "scores": {
                        "lex": score_lex,
                        "vec": score_vec,
                        "hybrid": score_hybrid,
                    } if any([score_lex, score_vec, score_hybrid]) else None,
                    "highlight": highlight,
                    "matched_terms": lexemes,
                    "evidence_refs": evidence_refs,
                })
        
        result_set = {
            "id": rs_id,
            "name": name,
            "retrieval_run_id": run_id,
            "summary": {
                "item_count": len(items),
                "document_count": len(document_ids),
            },
            "items": items,
            "created_at": created_at.isoformat() if created_at else None,
        }
        
        return {
            "result_set": result_set,
        }
