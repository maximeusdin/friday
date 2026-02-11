#!/usr/bin/env python3
"""
friday_cli.py - Interactive CLI for Friday Research Console

A unified interface for submitting queries and receiving responses.

Usage:
    # Interactive mode (recommended)
    python scripts/friday_cli.py
    
    # One-shot query
    python scripts/friday_cli.py --query "Who was involved in the Rosenberg case?"
    
    # Use specific session
    python scripts/friday_cli.py --session "my-research"
    
    # Auto-execute (skip approval)
    python scripts/friday_cli.py --query "atomic espionage" --auto-execute
    
    # Force agentic mode
    python scripts/friday_cli.py --query "Who were handlers of Rosenberg?" --agentic

The CLI provides a conversational interface:
1. Creates or selects a session
2. Accepts natural language queries
3. Shows the research plan (or uses agentic workflow)
4. Executes on approval
5. Displays results with optional summarization

Agentic Mode (default for complex queries):
- Uses Plan → Execute → Verify → Render architecture
- Multi-lane retrieval with ephemeral expansion
- Deterministic claim extraction with evidence refs
- Per-bullet citations in rendered answer
"""

import os
import sys
import argparse
import json
import textwrap
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Feature flag for agentic mode (enabled by default)
AGENTIC_MODE_ENABLED = os.getenv("AGENTIC_MODE_ENABLED", "1") == "1"

import psycopg2
from psycopg2.extras import Json

# =============================================================================
# Database
# =============================================================================

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL environment variable not set.")
        print("Run: source friday_env.sh")
        sys.exit(1)
    return psycopg2.connect(dsn)

# =============================================================================
# Session Management
# =============================================================================

def list_sessions(conn, limit=10):
    """List recent sessions."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, label, created_at
            FROM research_sessions
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,)
        )
        return cur.fetchall()

def create_session(conn, label: str) -> int:
    """Create a new session or return existing by label."""
    with conn.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO research_sessions (label) VALUES (%s) RETURNING id",
                (label,)
            )
            session_id = cur.fetchone()[0]
            conn.commit()
            return session_id
        except psycopg2.IntegrityError:
            conn.rollback()
            cur.execute(
                "SELECT id FROM research_sessions WHERE label = %s",
                (label,)
            )
            return cur.fetchone()[0]

def resolve_session(conn, session_arg: str) -> int:
    """Resolve session by ID or label."""
    with conn.cursor() as cur:
        if session_arg.isdigit():
            cur.execute("SELECT id FROM research_sessions WHERE id = %s", (int(session_arg),))
        else:
            cur.execute("SELECT id FROM research_sessions WHERE label = %s", (session_arg,))
        row = cur.fetchone()
        if row:
            return row[0]
        raise ValueError(f"Session not found: {session_arg}")

# =============================================================================
# Query Planning
# =============================================================================

def plan_query(conn, session_id: int, query_text: str):
    """Generate a research plan for the query using LLM."""
    # Import the planner service
    try:
        from backend.app.services.planner import propose_plan
        return propose_plan(conn, session_id, query_text)
    except ImportError:
        pass
    
    # Fall back to subprocess call
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plan_query.py"),
            "--session", str(session_id),
            "--text", query_text,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ},
        timeout=120,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Plan generation failed: {result.stderr}")
    
    # Fetch the created plan
    return get_pending_plan(conn, session_id)

def get_pending_plan(conn, session_id: int):
    """Get the most recent pending/proposed plan for a session."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, plan_json, user_utterance, status, created_at
            FROM research_plans
            WHERE session_id = %s AND status IN ('proposed', 'approved')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,)
        )
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "plan_json": row[1],
                "user_utterance": row[2],
                "status": row[3],
                "created_at": row[4],
            }
        return None

def format_plan(plan_json: dict) -> str:
    """Format plan for display."""
    lines = []
    
    # Primitives are nested under query.primitives
    query = plan_json.get("query", {})
    primitives = query.get("primitives", [])
    
    # Group primitives by type for cleaner display
    search_primitives = []
    filter_primitives = []
    control_primitives = []
    
    for p in primitives:
        ptype = p.get("type", "unknown")
        if ptype in ("TERM", "PHRASE", "ENTITY", "RELATED_ENTITIES", "MENTIONS", "FIRST_MENTION"):
            search_primitives.append(p)
        elif ptype in ("FILTER_COLLECTION", "FILTER_DATE_RANGE", "WITHIN_RESULT_SET", "ENTITY_ROLE"):
            filter_primitives.append(p)
        else:
            control_primitives.append(p)
    
    # Display search primitives
    if search_primitives:
        lines.append("  Search:")
        for p in search_primitives:
            ptype = p.get("type", "unknown")
            if ptype == "TERM":
                lines.append(f"    • Term: \"{p.get('value', '')}\"")
            elif ptype == "PHRASE":
                lines.append(f"    • Phrase: \"{p.get('value', '')}\"")
            elif ptype == "ENTITY":
                lines.append(f"    • Entity ID: {p.get('entity_id', '')}")
            elif ptype == "RELATED_ENTITIES":
                lines.append(f"    • Related to entity ID: {p.get('entity_id', '')} (top {p.get('top_n', 20)})")
            elif ptype == "MENTIONS":
                lines.append(f"    • All mentions of entity: {p.get('entity_id', '')}")
            elif ptype == "FIRST_MENTION":
                lines.append(f"    • First mention of entity: {p.get('entity_id', '')}")
    
    # Display filters
    if filter_primitives:
        lines.append("  Filters:")
        for p in filter_primitives:
            ptype = p.get("type", "unknown")
            if ptype == "FILTER_COLLECTION":
                lines.append(f"    • Collection: {p.get('slug', '')}")
            elif ptype == "FILTER_DATE_RANGE":
                lines.append(f"    • Date: {p.get('start', '')} to {p.get('end', '')}")
            elif ptype == "WITHIN_RESULT_SET":
                lines.append(f"    • Within result set: {p.get('result_set_id', 'previous')}")
            elif ptype == "ENTITY_ROLE":
                lines.append(f"    • Entity type: {p.get('role', 'any')}")
    
    # Display controls (only if non-default)
    interesting_controls = []
    for p in control_primitives:
        ptype = p.get("type", "unknown")
        if ptype == "SET_SEARCH_TYPE" and p.get("value") != "hybrid":
            interesting_controls.append(f"    • Search: {p.get('value')}")
        elif ptype == "SET_RETRIEVAL_MODE" and p.get("mode") != "conversational":
            interesting_controls.append(f"    • Mode: {p.get('mode')}")
        elif ptype == "SET_TOP_K":
            interesting_controls.append(f"    • Limit: {p.get('value', 20)} results")
    
    if interesting_controls:
        lines.append("  Options:")
        lines.extend(interesting_controls)
    
    # Show hybrid indicator
    has_hybrid = any(
        p.get("type") == "SET_SEARCH_TYPE" and p.get("value") == "hybrid"
        for p in control_primitives
    )
    if has_hybrid:
        lines.append("\n  [OK] Using hybrid search (vector + lexical)")
    
    # Show compiled query if available
    compiled = plan_json.get("_compiled", {})
    if compiled.get("tsquery_text"):
        lines.append(f"\n  Compiled query: {compiled['tsquery_text'][:100]}")
    
    return "\n".join(lines) if lines else "  (empty plan)"

# =============================================================================
# Execution
# =============================================================================

def approve_plan(conn, plan_id: int):
    """Approve a plan for execution."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE research_plans
            SET status = 'approved', approved_at = now()
            WHERE id = %s AND status = 'proposed'
            RETURNING id
            """,
            (plan_id,)
        )
        row = cur.fetchone()
        conn.commit()
        return row is not None

def execute_plan_via_script(plan_id: int, already_approved: bool = True):
    """Execute a plan using the execute_plan.py script."""
    import subprocess
    
    cmd = [
        sys.executable, 
        str(REPO_ROOT / "scripts" / "execute_plan.py"),
        "--plan-id", str(plan_id),
        "--create-result-set",
    ]
    
    # If not already approved, use --no-approval to execute proposed plans
    if not already_approved:
        cmd.append("--no-approval")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ},
        timeout=120,  # 2 minute timeout
    )
    
    # Combine stdout and stderr for parsing
    output = (result.stdout or "") + (result.stderr or "")
    
    if result.returncode != 0:
        # Include actual error message for debugging
        error_msg = result.stderr or result.stdout or "Execution failed"
        # Truncate very long errors
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return {"error": error_msg, "output": output}
    
    # Parse output to get result set ID
    import re
    # Match various output formats:
    # "Result set created: ID=120" or "Result Set ID: 120" or "result_set_id: 120"
    rs_match = re.search(r'(?:result[_\s]?set[_\s]?(?:id|created)[:\s=]+(?:ID[=\s]*)?)(\d+)', output, re.IGNORECASE)
    # "run_id=388" or "Run ID: 388" or "retrieval_run_id: 388"
    run_match = re.search(r'(?:retrieval_)?run[_\s]?id[:\s=]+(\d+)', output, re.IGNORECASE)
    # "7 chunks" or "7 results" or "Results (7 chunks)"
    count_match = re.search(r'(?:Results?\s*\()?(\d+)\s*(?:chunks?|hits?|results?)', output, re.IGNORECASE)
    
    return {
        "plan_id": plan_id,
        "result_set_id": int(rs_match.group(1)) if rs_match else None,
        "retrieval_run_id": int(run_match.group(1)) if run_match else None,
        "hit_count": int(count_match.group(1)) if count_match else 0,
        "output": output,
    }


def execute_plan_direct(conn, plan_id: int, already_approved: bool = True):
    """Execute a plan and return results."""
    # First try via API if server is running
    try:
        import requests
        response = requests.post(
            f"http://localhost:8000/api/plans/{plan_id}/execute",
            timeout=120,
        )
        if response.ok:
            data = response.json()
            return {
                "plan_id": plan_id,
                "result_set_id": data.get("result_set_id"),
                "retrieval_run_id": data.get("retrieval_run_id"),
                "hit_count": data.get("hit_count", 0),
            }
        elif response.status_code != 404:
            # API responded but with error
            try:
                error_data = response.json()
                return {"error": error_data.get("detail", str(response.status_code))}
            except Exception:
                return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        pass  # Server not running, fall back to script
    except Exception as e:
        # Unexpected error, try script fallback
        pass
    
    # Fall back to script execution
    return execute_plan_via_script(plan_id, already_approved=already_approved)

# =============================================================================
# Results Display
# =============================================================================

def get_result_set(conn, result_set_id: int):
    """Get result set details."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rs.id, rs.name, rs.created_at, array_length(rs.chunk_ids, 1)
            FROM result_sets rs
            WHERE rs.id = %s
            """,
            (result_set_id,)
        )
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "created_at": row[2],
                "chunk_count": row[3] or 0,
            }
        return None

def get_result_preview(conn, result_set_id: int, limit=5):
    """Get a preview of results."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                rsc.rank,
                rsc.chunk_id,
                COALESCE(d.source_name, 'Unknown') as doc_name,
                LEFT(COALESCE(c.clean_text, c.text), 200) as preview
            FROM result_set_chunks rsc
            JOIN chunks c ON c.id = rsc.chunk_id
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = rsc.chunk_id
            LEFT JOIN documents d ON d.id = cm.document_id
            WHERE rsc.result_set_id = %s
            ORDER BY rsc.rank
            LIMIT %s
            """,
            (result_set_id, limit)
        )
        return cur.fetchall()

def format_results_preview(results):
    """Format results for display."""
    lines = []
    for rank, chunk_id, title, preview in results:
        lines.append(f"\n  [{rank}] {title or 'Unknown document'}")
        # Wrap preview text
        wrapped = textwrap.fill(preview or "", width=70, initial_indent="      ", subsequent_indent="      ")
        lines.append(wrapped + "...")
    return "\n".join(lines)


def get_detailed_evidence(conn, result_set_id: int, limit=10):
    """Get detailed evidence with full text excerpts."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                rsc.rank,
                rsc.chunk_id,
                COALESCE(d.source_name, 'Unknown') as doc_name,
                cm.collection_slug,
                cm.page_number,
                COALESCE(c.clean_text, c.text) as full_text
            FROM result_set_chunks rsc
            JOIN chunks c ON c.id = rsc.chunk_id
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = rsc.chunk_id
            LEFT JOIN documents d ON d.id = cm.document_id
            WHERE rsc.result_set_id = %s
            ORDER BY rsc.rank
            LIMIT %s
            """,
            (result_set_id, limit)
        )
        return cur.fetchall()


def format_detailed_evidence(results):
    """Format detailed evidence for display."""
    lines = []
    for rank, chunk_id, doc_name, collection, page, full_text in results:
        lines.append(f"\n{'='*60}")
        lines.append(f"  [{rank}] {doc_name or 'Unknown'}")
        if collection:
            lines.append(f"  Collection: {collection}")
        if page:
            lines.append(f"  Page: {page}")
        lines.append(f"  Chunk ID: {chunk_id}")
        lines.append("-"*60)
        # Show more text (up to 800 chars)
        text = (full_text or "")[:800]
        wrapped = textwrap.fill(text, width=76, initial_indent="  ", subsequent_indent="  ")
        lines.append(wrapped)
        if len(full_text or "") > 800:
            lines.append("  ...")
    return "\n".join(lines)

# =============================================================================
# Summarization
# =============================================================================

def summarize_results(conn, result_set_id: int, summary_type="brief"):
    """Generate a summary of results."""
    try:
        from scripts.summarize_results import summarize_result_set
        result = summarize_result_set(conn, result_set_id, summary_type=summary_type)
        return result.get("summary", "")
    except ImportError:
        return "(Summarization requires OpenAI API key)"
    except Exception as e:
        return f"(Summarization failed: {e})"


# =============================================================================
# Agentic Workflow
# =============================================================================

def is_agentic_query(query_text: str) -> bool:
    """
    Determine if a query should use agentic mode.
    
    Agentic mode is best for:
    - Roster queries ("who were members of...")
    - Relationship queries ("handlers of...", "connected to...")
    - Complex queries requiring evidence verification
    """
    import re
    text_lower = query_text.lower()
    
    # Patterns that benefit from agentic mode
    agentic_patterns = [
        r"\bwho\s+were\s+(?:the\s+)?(?:members?|people|handlers?|officers?)\b",
        r"\blist\s+(?:all\s+)?(?:the\s+)?(?:members?|people)\b",
        r"\bidentify\s+(?:all\s+)?(?:the\s+)?(?:members?|people)\b",
        r"\brelationship\s+between\b",
        r"\bconnected?\s+to\b.*\bintelligence\b",
        r"\bhandler\s+(?:of|for)\b",
        r"\bcase\s+officer\b",
        r"\bcodename\b.*\bidentify\b",
        r"\b(?:Soviet|Russian)\s+(?:intelligence\s+)?officers?\b",
        r"\bwho\s+(?:handled|ran|controlled|recruited)\b",
        r"\bnetwork\s+members?\b",
    ]
    
    for pattern in agentic_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def execute_agentic_query(conn, session_id: int, query_text: str):
    """
    Execute a query using the agentic workflow.
    
    Returns:
        Dict with rendered_answer, evidence_bundle, etc.
    """
    print("  [Agentic] Initializing workflow...", file=sys.stderr)
    
    try:
        from retrieval.plan import (
            AgenticPlan, LaneSpec, ExtractionSpec, VerificationSpec,
            Budgets, StopConditions, BundleConstraints,
            build_default_lanes_for_intent, build_verification_spec_for_intent,
        )
        from retrieval.intent import classify_intent, IntentFamily, link_query_entities
        from retrieval.lanes import execute_plan_iterative
        from retrieval.claim_extraction import extract_claims
        from retrieval.codename_resolution import resolve_codenames
        from retrieval.verifier import verify_evidence_bundle, filter_claims_by_verification
        from retrieval.answer_trace import generate_answer_trace
        from backend.app.services.summarizer.synthesis import render_from_bundle
    except ImportError as e:
        return {"error": f"Agentic workflow dependencies not available: {e}"}
    
    # Step 1: Classify intent
    print("  [Agentic] Classifying intent...", file=sys.stderr)
    intent_result = classify_intent(query_text, [])
    
    # Step 1.5: Pre-retrieval entity linking
    print("  [Agentic] Linking query entities...", file=sys.stderr)
    linking_result = link_query_entities(query_text, conn)
    
    if linking_result.linked_entities:
        for le in linking_result.linked_entities[:5]:
            print(f"    Linked: '{le.matched_surface}' -> {le.canonical_name} "
                  f"(id={le.entity_id}, score={le.score:.2f}, type={le.match_type})", file=sys.stderr)
    else:
        print("    No entities linked from query", file=sys.stderr)
    
    if linking_result.unlinked_tokens:
        print(f"    Unlinked tokens: {linking_result.unlinked_tokens}", file=sys.stderr)
    
    # Step 2: Build constraints from anchors
    anchors = intent_result.anchors
    constraints = BundleConstraints(
        collection_scope=anchors.constraints.get("collection_scope", []),
        required_role_evidence_patterns=anchors.constraints.get("role_evidence_patterns", []),
    )
    
    # Step 3: Build query terms - combine all extracted anchors
    query_terms = list(set(anchors.key_concepts + anchors.target_tokens))
    
    # If still no query terms, extract key words from the query itself
    if not query_terms:
        import re
        # Extract significant words (3+ chars, not common stopwords)
        stopwords = {'who', 'what', 'were', 'was', 'are', 'the', 'members', 'people', 
                     'list', 'find', 'search', 'evidence', 'information', 'about', 
                     'from', 'with', 'for', 'and', 'that', 'this', 'there', 'any'}
        words = re.findall(r'\b([a-zA-Z]{3,})\b', query_text.lower())
        query_terms = [w for w in words if w not in stopwords][:5]  # Take top 5
    
    print(f"  [Agentic] Query terms: {query_terms}", file=sys.stderr)
    
    # Step 3.5: Merge linked entity IDs with any from intent anchors
    entity_ids = list(set(anchors.target_entities + linking_result.entity_ids))
    
    if entity_ids:
        print(f"  [Agentic] Entity IDs for retrieval: {entity_ids}", file=sys.stderr)
    
    # Step 4: Build lanes based on intent
    lanes = build_default_lanes_for_intent(
        intent=intent_result.intent_family,
        entity_ids=entity_ids,  # Now includes linked entities!
        query_terms=query_terms,
        collection_scope=constraints.collection_scope,
    )
    
    # Step 5: Build verification spec
    verification = build_verification_spec_for_intent(
        intent=intent_result.intent_family,
        role_patterns=constraints.required_role_evidence_patterns,
        collection_scope=constraints.collection_scope,
    )
    
    # Step 6: Build the typed plan
    agentic_plan = AgenticPlan(
        intent=intent_result.intent_family,
        constraints=constraints,
        lanes=lanes,
        extraction=ExtractionSpec(predicates=[], allow_llm_refinement=False),
        verification=verification,
        budgets=Budgets(),
        stop_conditions=StopConditions(),
        query_text=query_text,
        session_id=str(session_id),
    )
    
    print(f"  [Agentic] Intent: {intent_result.intent_family.value}, "
          f"Lanes: {len(lanes)}, Confidence: {intent_result.confidence:.2f}", file=sys.stderr)
    
    # Step 7: Execute iterative retrieval
    print("  [Agentic] Executing multi-lane retrieval...", file=sys.stderr)
    bundle = execute_plan_iterative(agentic_plan, conn)
    
    print(f"  [Agentic] Retrieved {len(bundle.all_chunks)} chunks, "
          f"{len(bundle.entities)} entities in {bundle.rounds_executed} rounds", file=sys.stderr)
    
    # Step 8: Extract claims
    print("  [Agentic] Extracting claims...", file=sys.stderr)
    chunks_list = list(bundle.all_chunks.values())
    candidates_dict = {e.key: e for e in bundle.entities}
    
    claims = extract_claims(
        chunks=chunks_list,
        candidates=candidates_dict,
        intent=intent_result.intent_family,
        extraction_spec=agentic_plan.extraction,
        conn=conn,
    )
    bundle.claims = claims
    
    print(f"  [Agentic] Extracted {len(claims)} claims", file=sys.stderr)
    
    # Step 9: Resolve codenames if any
    if bundle.unresolved_tokens:
        print(f"  [Agentic] Resolving {len(bundle.unresolved_tokens)} codenames...", file=sys.stderr)
        resolution = resolve_codenames(
            bundle.unresolved_tokens,
            constraints.collection_scope,
            conn,
        )
        bundle.claims.extend(resolution.claims)
        bundle.unresolved_tokens = resolution.unresolved
    
    # Step 10: Verify
    print("  [Agentic] Verifying evidence bundle...", file=sys.stderr)
    verification_result = verify_evidence_bundle(bundle, conn)
    
    if verification_result.passed:
        print("  [Agentic] Verification passed!", file=sys.stderr)
    else:
        print(f"  [Agentic] Verification: {len(verification_result.errors)} errors, "
              f"{len(verification_result.dropped_claims)} claims dropped", file=sys.stderr)
        # Show specific errors
        for error in verification_result.errors[:5]:
            print(f"    Error: {error.error_type.value}: {error.message}", file=sys.stderr)
            if error.details:
                print(f"      Details: {error.details}", file=sys.stderr)
        # Filter out invalid claims
        verified_claims, dropped = filter_claims_by_verification(bundle.claims, constraints)
        bundle.claims = verified_claims
    
    # Step 11: Generate trace
    trace = generate_answer_trace(bundle, verification_result)
    
    # Step 12: Render answer
    print("  [Agentic] Rendering answer...", file=sys.stderr)
    rendered = render_from_bundle(bundle)
    
    print(f"  [Agentic] Complete: {rendered.claims_rendered} claims, "
          f"{rendered.citations_included} citations", file=sys.stderr)
    
    # Step 13: Create result_set for /summarize compatibility
    result_set_id = None
    if bundle.all_chunks:
        try:
            from retrieval.ops import log_retrieval_run
            from scripts.execute_plan import create_result_set
            
            chunk_ids = list(bundle.all_chunks.keys())
            chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
            
            # Create a retrieval run for audit trail
            # Note: search_type must be 'lex', 'vector', 'hybrid', or 'index' per DB constraint
            run_id = log_retrieval_run(
                conn,
                query_text=f"[AGENTIC] {query_text}",
                search_type="hybrid",  # closest match for agentic (multi-lane)
                chunk_pv=chunk_pv,
                embedding_model=None,
                top_k=len(chunk_ids),
                returned_chunk_ids=chunk_ids,
                retrieval_config_json={
                    "mode": "agentic",
                    "intent": intent_result.intent_family.value,
                    "lanes_executed": len(lanes),
                    "rounds": bundle.rounds_executed,
                },
                auto_commit=False,
                session_id=session_id,
            )
            conn.commit()
            
            # Create result_set
            result_set_id = create_result_set(
                conn,
                run_id=run_id,
                chunk_ids=chunk_ids,
                name=f"Agentic: {query_text[:40]}... (run {run_id})",
                session_id=session_id,
            )
            print(f"  [Agentic] Created result set #{result_set_id}", file=sys.stderr)
        except Exception as e:
            print(f"  [Agentic] Warning: Could not create result set: {e}", file=sys.stderr)
    
    return {
        "mode": "agentic",
        "rendered_answer": rendered.to_dict(),
        "verification_passed": verification_result.passed,
        "intent": intent_result.intent_family.value,
        "confidence": intent_result.confidence,
        "rounds_executed": bundle.rounds_executed,
        "chunks_retrieved": len(bundle.all_chunks),
        "entities_found": len(bundle.entities),
        "claims_count": len(bundle.claims),
        "trace_id": trace.trace_id,
        "result_set_id": result_set_id,
    }


# =============================================================================
# Agentic V2 Workflow
# =============================================================================

def execute_agentic_v2_query(conn, session_id: int, query_text: str):
    """
    Execute a query using the V2 agentic workflow.
    
    V2 architecture features:
    - FocusBundle as single source of truth for citations
    - Span-level evidence (not chunk-level)
    - Constraint scoring (affiliation, relationship, role)
    - Hubness penalty for specificity
    - True MMR diversity
    - Verification of invariants
    
    Returns:
        Dict with answer, focus_bundle stats, verification result
    """
    print("  [V2 Agentic] Initializing workflow...", file=sys.stderr)
    
    try:
        from retrieval.executor_v2 import execute_keyword_query, ExecutionResult
        from retrieval.query_intent import QueryContract, FocusBundleMode
        from retrieval.rendering import render_answer_text
    except ImportError as e:
        return {"error": f"V2 workflow dependencies not available: {e}"}
    
    # Execute V2 workflow
    print("  [V2 Agentic] Building FocusBundle...", file=sys.stderr)
    
    try:
        result = execute_keyword_query(query_text, conn)
    except Exception as e:
        return {"error": f"V2 execution failed: {e}"}
    
    print(f"  [V2 Agentic] FocusBundle: {len(result.focus_bundle.spans)} spans from "
          f"{len(result.focus_bundle.get_unique_doc_ids())} docs", file=sys.stderr)
    print(f"  [V2 Agentic] Candidates: {len(result.candidates)} scored", file=sys.stderr)
    print(f"  [V2 Agentic] Bullets: {len(result.answer.bullets)} rendered", file=sys.stderr)
    print(f"  [V2 Agentic] Verification: {'PASSED' if result.verification.passed else 'FAILED'}", 
          file=sys.stderr)
    
    # Create result_set for /summarize compatibility
    result_set_id = None
    if result.focus_bundle.spans:
        try:
            from retrieval.ops import log_retrieval_run
            from scripts.execute_plan import create_result_set
            
            chunk_ids = list({s.chunk_id for s in result.focus_bundle.spans})
            chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
            
            run_id = log_retrieval_run(
                conn,
                query_text=f"[AGENTIC_V2] {query_text}",
                search_type="hybrid",
                chunk_pv=chunk_pv,
                embedding_model=None,
                top_k=len(chunk_ids),
                returned_chunk_ids=chunk_ids,
                retrieval_config_json={
                    "mode": "agentic_v2",
                    "focus_spans": len(result.focus_bundle.spans),
                    "candidates_scored": len(result.candidates),
                    "verification_passed": result.verification.passed,
                },
                auto_commit=False,
                session_id=session_id,
            )
            conn.commit()
            
            result_set_id = create_result_set(
                conn,
                run_id=run_id,
                chunk_ids=chunk_ids,
                name=f"V2 Agentic: {query_text[:40]}... (run {run_id})",
                session_id=session_id,
            )
            print(f"  [V2 Agentic] Created result set #{result_set_id}", file=sys.stderr)
        except Exception as e:
            print(f"  [V2 Agentic] Warning: Could not create result set: {e}", file=sys.stderr)
    
    return {
        "mode": "agentic_v2",
        "answer": result.answer,
        "focus_bundle": result.focus_bundle,
        "candidates": result.candidates,
        "verification": result.verification,
        "stats": result.stats,
        "result_set_id": result_set_id,
    }


def format_agentic_v2_result(result: dict) -> str:
    """Format V2 agentic workflow result for display."""
    lines = []
    
    if result.get("error"):
        return f"Error: {result['error']}"
    
    answer = result.get("answer")
    focus_bundle = result.get("focus_bundle")
    verification = result.get("verification")
    stats = result.get("stats", {})
    
    # Short answer
    if answer:
        lines.append(f"\n  {answer.short_answer}")
    
    # Bullets with citations
    if answer and answer.bullets:
        lines.append(f"\n  Findings ({len(answer.bullets)} total):")
        for i, bullet in enumerate(answer.bullets[:20], 1):
            text = bullet.text
            confidence = bullet.confidence
            citation_count = len(bullet.cited_span_ids)
            
            # Truncate long text
            if len(text) > 150:
                text = text[:150] + "..."
            
            marker = "•" if confidence == "high" else "○"
            lines.append(f"    {marker} {text}")
            if citation_count > 0:
                lines.append(f"      [{citation_count} citation(s) from FocusBundle]")
        
        if len(answer.bullets) > 20:
            lines.append(f"\n    ... and {len(answer.bullets) - 20} more")
    
    # Negative findings
    if answer and answer.negative_findings:
        lines.append(f"\n  {answer.negative_findings}")
    
    # Verification status
    lines.append(f"\n  --- V2 Agentic Workflow Stats ---")
    lines.append(f"  FocusBundle: {len(focus_bundle.spans) if focus_bundle else 0} spans")
    if focus_bundle:
        lines.append(f"  Documents: {len(focus_bundle.get_unique_doc_ids())}")
    lines.append(f"  Candidates scored: {len(result.get('candidates', []))}")
    lines.append(f"  Pipeline stages: {stats.get('pipeline_stages', [])}")
    
    if verification:
        status = "[OK] Passed" if verification.passed else "[FAIL] Issues found"
        lines.append(f"  Verification: {status}")
        if verification.errors:
            lines.append(f"    Errors: {len(verification.errors)}")
            for err in verification.errors[:3]:
                lines.append(f"      - {err[:80]}")
        if verification.warnings:
            lines.append(f"    Warnings: {len(verification.warnings)}")
    
    if result.get('result_set_id'):
        lines.append(f"  Result Set: #{result['result_set_id']}")
    
    return "\n".join(lines)


# =============================================================================
# V3 Agentic Workflow (Tool-based with evidence banding and claim verification)
# =============================================================================

def execute_agentic_v3_query(conn, session_id: int, query_text: str, dump_plan: bool = False, dump_evidence: bool = False, dump_verifier: bool = False):
    """
    Execute a query using the V3 agentic workflow.
    
    V3 architecture features:
    - Tool Registry: Typed wrappers around search primitives
    - Evidence Builder: Span mining + rerank + cite/harvest banding
    - Claim Synthesis: LLM generates claims with strict citation requirements
    - Universal Verifier: No intent families, just universal grounding rules
    - Retry Loop: Agent receives error feedback and revises plan
    
    Args:
        conn: Database connection
        session_id: Research session ID
        query_text: User's query
        dump_plan: If True, return plan JSON for debugging
        dump_evidence: If True, return evidence set for debugging
        dump_verifier: If True, return verification report for debugging
    
    Returns:
        Dict with claims, evidence, verification, trace
    """
    print("  [V3 Agentic] Initializing tool-based workflow...", file=sys.stderr)
    
    try:
        from retrieval.agent.v3_runner import V3Runner, V3Result
    except ImportError as e:
        return {"error": f"V3 workflow dependencies not available: {e}"}
    
    # Execute V3 workflow
    try:
        runner = V3Runner(verbose=True)
        result = runner.run(query_text, conn)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"V3 execution failed: {e}"}
    
    # Create result_set for /summarize compatibility
    result_set_id = None
    if result.evidence_set and result.evidence_set.cite_spans:
        try:
            from retrieval.ops import log_retrieval_run
            from scripts.execute_plan import create_result_set
            
            chunk_ids = list({s.chunk_id for s in result.evidence_set.cite_spans})
            chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
            
            run_id = log_retrieval_run(
                conn,
                query_text=f"[AGENTIC_V3] {query_text}",
                search_type="hybrid",
                chunk_pv=chunk_pv,
                embedding_model=None,
                top_k=len(chunk_ids),
                returned_chunk_ids=chunk_ids,
                retrieval_config_json={
                    "mode": "agentic_v3",
                    "cite_spans": len(result.evidence_set.cite_spans),
                    "harvest_spans": len(result.evidence_set.harvest_spans),
                    "claims": len(result.claims.claims),
                    "rounds_used": result.trace.final_round,
                    "verification_passed": result.report.passed,
                },
                auto_commit=False,
                session_id=session_id,
            )
            conn.commit()
            
            result_set_id = create_result_set(
                conn,
                run_id=run_id,
                chunk_ids=chunk_ids,
                name=f"V3 Agentic: {query_text[:40]}... (run {run_id})",
                session_id=session_id,
            )
            print(f"  [V3 Agentic] Created result set #{result_set_id}", file=sys.stderr)
        except Exception as e:
            print(f"  [V3 Agentic] Warning: Could not create result set: {e}", file=sys.stderr)
    
    return {
        "mode": "agentic_v3",
        "success": result.success,
        "run_id": result.run_id,
        "claims": result.claims,
        "evidence_set": result.evidence_set,
        "report": result.report,
        "trace": result.trace,
        "plan": result.plan,
        "result_set_id": result_set_id,
        "dump_plan": dump_plan,
        "dump_evidence": dump_evidence,
        "dump_verifier": dump_verifier,
    }


def format_agentic_v3_result(result: dict) -> str:
    """Format V3 agentic workflow result for display."""
    import json
    lines = []
    
    if result.get("error"):
        return f"Error: {result['error']}"
    
    claims_bundle = result.get("claims")
    evidence_set = result.get("evidence_set")
    report = result.get("report")
    trace = result.get("trace")
    plan = result.get("plan")
    
    # Debug dumps
    if result.get("dump_plan") and plan:
        lines.append("\n=== PLAN JSON ===")
        lines.append(json.dumps(plan.to_dict(), indent=2))
        lines.append("=== END PLAN ===\n")
    
    if result.get("dump_evidence") and evidence_set:
        lines.append("\n=== EVIDENCE SET ===")
        lines.append(f"Cite spans ({len(evidence_set.cite_spans)}):")
        for span in evidence_set.cite_spans[:10]:
            lines.append(f"  [{span.span_id}] score={span.score:.3f}")
            lines.append(f"    {span.quote[:100]}...")
        if len(evidence_set.cite_spans) > 10:
            lines.append(f"  ... and {len(evidence_set.cite_spans) - 10} more cite spans")
        lines.append(f"Harvest spans: {len(evidence_set.harvest_spans)}")
        lines.append("=== END EVIDENCE ===\n")
    
    if result.get("dump_verifier") and report:
        lines.append("\n=== VERIFICATION REPORT ===")
        lines.append(json.dumps(report.to_dict(), indent=2))
        lines.append("=== END VERIFICATION ===\n")
    
    # Success status
    success = result.get("success", False)
    status_icon = "✓" if success else "✗"
    lines.append(f"\n  [{status_icon}] V3 Agentic Result ({result.get('run_id', 'unknown')})")
    
    # Claims
    if claims_bundle and claims_bundle.claims:
        lines.append(f"\n  Claims ({len(claims_bundle.claims)}):")
        for i, claim in enumerate(claims_bundle.claims[:15], 1):
            text = claim.text
            confidence = claim.confidence
            citation_count = len(claim.evidence)
            
            # Truncate long text
            if len(text) > 150:
                text = text[:150] + "..."
            
            marker = "•" if confidence == "supported" else "○"
            lines.append(f"    {marker} {text}")
            if citation_count > 0:
                lines.append(f"      [{citation_count} citation(s)]")
        
        if len(claims_bundle.claims) > 15:
            lines.append(f"\n    ... and {len(claims_bundle.claims) - 15} more claims")
    else:
        lines.append("\n  No claims generated.")
    
    # Evidence stats
    if evidence_set:
        lines.append(f"\n  --- Evidence Stats ---")
        stats = evidence_set.stats
        lines.append(f"  Total chunks: {stats.total_chunks}")
        lines.append(f"  Spans mined: {stats.total_spans_mined}")
        lines.append(f"  Cite spans: {stats.cite_span_count}")
        lines.append(f"  Harvest spans: {stats.harvest_span_count}")
        lines.append(f"  Unique docs: {stats.unique_docs}")
    
    # Verification
    if report:
        status = "[PASSED]" if report.passed else f"[FAILED: {len(report.errors)} errors]"
        lines.append(f"\n  Verification: {status}")
        if report.errors:
            for err in report.errors[:5]:
                lines.append(f"    - [{err.error_type}] {err.details[:60]}...")
    
    # Trace
    if trace:
        lines.append(f"\n  --- Execution Trace ---")
        lines.append(f"  Rounds: {trace.final_round}")
        lines.append(f"  Total time: {trace.total_elapsed_ms:.0f}ms")
        for round_info in trace.rounds:
            lines.append(f"    Round {round_info['round']}: {round_info['chunks_retrieved']} chunks, {round_info['claims_generated']} claims")
    
    # Plan info
    if plan:
        lines.append(f"\n  Plan: {len(plan.steps)} steps, hash={plan.plan_hash}")
        lines.append(f"  Steps: {[s.tool_name for s in plan.steps]}")
    
    if result.get('result_set_id'):
        lines.append(f"\n  Result Set: #{result['result_set_id']}")
    
    return "\n".join(lines)


def format_agentic_result(result: dict) -> str:
    """Format agentic workflow result for display."""
    lines = []
    
    if result.get("error"):
        return f"Error: {result['error']}"
    
    rendered = result.get("rendered_answer", {})
    
    # Show summary (LLM-generated) if available
    summary = rendered.get("summary")
    if summary:
        lines.append(f"\n  Summary:")
        lines.append(f"  {'-' * 50}")
        # Wrap summary text
        import textwrap
        wrapped = textwrap.fill(summary, width=70, initial_indent="  ", subsequent_indent="  ")
        lines.append(wrapped)
        lines.append(f"  {'-' * 50}")
    else:
        # Fallback to short answer if no summary
        short_answer = rendered.get("short_answer", "")
        if short_answer:
            lines.append(f"\n  {short_answer}")
    
    # Show bullets with citations (limit to top 20)
    bullets = rendered.get("bullets", [])
    MAX_BULLETS_SHOWN = 20
    if bullets:
        lines.append(f"\n  Findings ({len(bullets)} total, showing top {min(len(bullets), MAX_BULLETS_SHOWN)}):")
        for i, bullet in enumerate(bullets[:MAX_BULLETS_SHOWN], 1):
            text = bullet.get("text", "")
            inference = bullet.get("inference_level", "explicit")
            citation_count = len(bullet.get("citations", []))
            
            # Truncate very long bullet text
            if len(text) > 200:
                text = text[:200] + "..."
            
            marker = "•" if inference == "explicit" else "o"
            lines.append(f"    {marker} {text}")
            if citation_count > 0:
                lines.append(f"      [{citation_count} citation(s)]")
        
        if len(bullets) > MAX_BULLETS_SHOWN:
            lines.append(f"\n    ... and {len(bullets) - MAX_BULLETS_SHOWN} more findings")
    
    # Show negative answer if no findings
    negative = rendered.get("negative_answer")
    if negative and not bullets:
        lines.append(f"\n  {negative.get('statement', 'No evidence found')}")
        lines.append(f"  ({negative.get('caveat', '')})")
        
        suggestions = negative.get("suggestions", [])
        if suggestions:
            lines.append("\n  Suggestions:")
            for s in suggestions[:3]:
                lines.append(f"    • {s}")
    
    # Show metadata
    lines.append(f"\n  --- Agentic Workflow Stats ---")
    lines.append(f"  Intent: {result.get('intent', 'unknown')} (confidence: {result.get('confidence', 0):.2f})")
    lines.append(f"  Rounds: {result.get('rounds_executed', 0)}")
    lines.append(f"  Chunks: {result.get('chunks_retrieved', 0)}")
    lines.append(f"  Entities: {result.get('entities_found', 0)}")
    lines.append(f"  Claims: {result.get('claims_count', 0)}")
    lines.append(f"  Verification: {'[OK] Passed' if result.get('verification_passed') else '[FAIL] Issues found'}")
    if result.get('result_set_id'):
        lines.append(f"  Result Set: #{result['result_set_id']}")
    
    return "\n".join(lines)

# =============================================================================
# Interactive CLI
# =============================================================================

def print_header():
    """Print CLI header."""
    print("\n" + "="*60)
    print("  FRIDAY - AI Research Assistant")
    print("  Cold War Archival Materials")
    print("="*60)
    print()

def print_help():
    """Print help message."""
    print("""
Commands:
  <query>       Submit a research query (V11 default: canonical embeddings, stripped V9)
  
  === V11 is the DEFAULT (plain query) ===
  V11 features:
    - Model drives tools (search, fetch, expand_entities w/ co-mentions)
    - Auto-detects scope (collections, date range) from question
    - Structured output: roster[]/timeline[]/evidence[]/edges[] by task_type
    - Sufficiency contract with per-task-type thresholds
    - Advisory verification (never blocks answer)
  
  /v9 <query>   V9 Reasoning Workspace Loop (full v9: PEM, concordance, query resolution)
  /v10 <query>  V10 Scope-aware Alias Identity Layer
  /deeper [msg]  Think Deeper: autonomous Actor/Judge loop on last result
                 Optional follow-up message to guide the investigation
                 Runs additional retrieval+verification steps to find new evidence
                 Can be chained: each /deeper builds on the previous result
  /v7 <query>   Use V7 with citation enforcement
  /v6 <query>   Use V6 (no citation enforcement)
  /v4 <query>   Use V4.2 agentic workflow (reasoning-first + discovery loop)
                Flags: --dump-plan, --dump-evidence, --dump-verify, --dump-discovery
                       --no-discovery, --thorough
  /v3 <query>   Use V3 agentic workflow (tool-based + claim verification)
                Flags: --dump-plan, --dump-evidence, --dump-verifier
  /v2 <query>   Use V2 agentic workflow (FocusBundle + constraints)
  /discover <q> Run discovery loop only (debug mode)
  /traditional  Use traditional (non-agentic) workflow for next query
  /sessions     List recent sessions
  /new <label>  Create a new session
  /use <id>     Switch to a different session
  /results      Show current result set (brief preview)
  /evidence     Show detailed evidence with full text excerpts
  /summarize    Summarize current results (V4: interpretation with verification)
                Shapes: /summarize roster|timeline|narrative|qa|index [--v3]
  /mode         Show current mode (agentic/traditional)
  /help         Show this help
  /quit         Exit the CLI

Workflow Modes:
  Default (V1): Multi-lane retrieval with claim extraction
  V2 (/v2):     FocusBundle architecture with constraint scoring
  V3 (/v3):     Tool-based workflow with claim synthesis + universal verifier
  V4.2 (/v4):   Reasoning-first interpretation + Discovery Loop (recommended):
                - V4.2 Discovery Loop: ChatGPT-like iterative retrieval
                  - 4o proposes search/pivot actions
                  - Executor runs tools deterministically
                  - Evidence rebuilt/expanded iteratively
                  - Stops when coverage is good or budgets hit
                - 4o reasoning model generates InterpretationV4 (structured answer units)
                - Hard verification: citations, entity attestation, relationship cues
                - Two-stage repair loop: interpretation retry → evidence expansion
                - Response shapes: roster, narrative, timeline, qa, index
  V6 (/v6):     Principled architecture with NO heuristics:
                - Query parsing separates CONTROL from CONTENT tokens
                - Entity linking only on CONTENT (never links "Provide" or "Vassiliev" as filter)
                - use_for_retrieval flag prevents random entities as seeds
                - Hard bottleneck forces convergence to 30-50 spans
                - Responsiveness check: does answer actually satisfy question?
                - Progress-gated rounds: stop if no quality evidence gained
  V11 (default): Stripped V9 with canonical embeddings:
                - No PEM lane, no query entity resolution, no concordance expansion
                - Canonical embeddings (chunk_embeddings_canonical) by default
                - Optional lightweight PEM via V11_USE_LIGHTWEIGHT_PEM=1
  V10 (/v10):   Scope-aware Alias Identity Layer:
                - Span-lattice query interpretation with dominance relationships
                - Collection-scoped alias semantics (Venona/Vassiliev codenames stay scoped)
                - Multi-tier alias resolution: Referent Rules → Contextual → General → Fallback
                - Structured entity/alias boosts (replaces query rewriting)
                - Entity-aware grounding and alias-annotated rendering
                - Full ThinkDeeper compatibility with V10 artifact persistence
  V9 (/v9):     Reasoning Workspace Loop V9.2:
                - Model drives tools (search_chunks, fetch_chunks, expand_entities)
                - expand_entities supports names[] + include_comentions for roster power
                - Auto-detects scope (collections/dates) from question text
                - Model self-labels task_type → structured output (roster/timeline/evidence/edges)
                - Sufficiency contract with per-task-type thresholds
                - Advisory responsiveness check → auto-extends if not responsive
                - Verification is advisory only (never blocks or drops answer)
  V7 (/v7):     V6 + Citation Enforcement:
                - All V6 features plus strict citation requirements
                - Claim enumeration, stop gate, expanded summary
                - Uncited claims are dropped from output

V4.2 Query Examples:
  /v4 who were members of the Silvermaster network  (roster query with discovery)
  /v4 --thorough evidence of Soviet proximity fuse  (thorough mode - doubled budgets)
  /v4 --no-discovery timeline of Rosenberg contacts (skip discovery loop)
  /discover who were Pal's group members            (discovery loop debug mode)

V6 Query Examples:
  /v6 who were members of the Silvermaster network? Provide citations from Vassiliev.
  /v6 list Soviet agents in the Treasury Department

V7 Query Examples:
  /v7 who were members of the Silvermaster network?  (all claims will have citations)
  /v7 list Soviet agents in the Treasury Department  (unsupported claims are dropped)

Plain query (V11 default):
  who handled Julius Rosenberg?                      (plain queries use V11 by default)

Environment / Manual fallbacks:
  USE_V9_AGENT=1              Use v9 as default for plain queries and dispatch (instead of v11)
  USE_ORIGINAL_EMBEDDINGS=1   Use original embeddings from chunk_embeddings (not canonical)
  V11_USE_LIGHTWEIGHT_PEM=1   Enable lightweight PEM for venona/vassiliev chunks (v11)
  THINK_DEEPER_MAX_STEPS=N    Max turns for /deeper (default 8). Use 2-3 for quick tests.
  THINK_DEEPER_MAX_TOOL_CALLS Max tool calls for /deeper (default 10). Use 15 for full Tier 2.
  THINK_DEEPER_TIER1_AUTO=0   Disable auto Tier 1 deepen (0 = only /deeper runs deepen).
  V11_QUERY_MAX_TURNS=N       Max tool calls (V9/V11) or rounds (V10) for initial query (default 5). Use 2 for fast /deeper setup.
""")

def interactive_mode(session_id: int, auto_execute: bool = False, pre_bundling_mode: str = "micro", bottleneck_mode: str = "score", deeper_max_steps: int = 8, deeper_max_tool_calls: int = 10, query_max_turns: int = 5, tier1_auto: bool = True):
    """Run interactive CLI mode."""
    conn = get_conn()
    
    # Get session info
    with conn.cursor() as cur:
        cur.execute("SELECT label FROM research_sessions WHERE id = %s", (session_id,))
        row = cur.fetchone()
        session_label = row[0] if row else f"Session {session_id}"
    
    current_result_set_id = None
    current_v3_result = None  # Track V3 result for structured summarization
    last_v9_workspace = None  # Track last V9 workspace for /deeper
    last_v9_question = None   # Track last V9 question for /deeper
    last_v9_run_id = None     # Track last V9 run_id (v9_runs) for /deeper DB persistence
    last_use_v9_fallback = False  # True if last run used v9 agent (USE_V9_AGENT=1), else v11
    
    print_header()
    print(f"Session: {session_label} (ID: {session_id})")
    print("Type your research query, or /help for commands.\n")
    
    while True:
        try:
            user_input = input(f"friday [{session_label}]> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            
            elif cmd == "/help":
                print_help()
            
            elif cmd == "/sessions":
                sessions = list_sessions(conn)
                print("\nRecent sessions:")
                for sid, label, created in sessions:
                    marker = " *" if sid == session_id else ""
                    print(f"  [{sid}] {label} ({created.strftime('%Y-%m-%d')}){marker}")
                print()
            
            elif cmd == "/new":
                if not arg:
                    print("Usage: /new <session-label>")
                    continue
                new_id = create_session(conn, arg)
                session_id = new_id
                session_label = arg
                print(f"\nSwitched to session: {session_label} (ID: {session_id})\n")
            
            elif cmd == "/use":
                if not arg:
                    print("Usage: /use <session-id or label>")
                    continue
                try:
                    new_id = resolve_session(conn, arg)
                    session_id = new_id
                    with conn.cursor() as cur:
                        cur.execute("SELECT label FROM research_sessions WHERE id = %s", (session_id,))
                        session_label = cur.fetchone()[0]
                    print(f"\nSwitched to session: {session_label} (ID: {session_id})\n")
                except ValueError as e:
                    print(f"Error: {e}")
            
            elif cmd == "/results":
                if current_result_set_id:
                    rs = get_result_set(conn, current_result_set_id)
                    if rs:
                        print(f"\nResult Set #{rs['id']}: {rs['chunk_count']} chunks")
                        preview = get_result_preview(conn, current_result_set_id)
                        print(format_results_preview(preview))
                        print()
                    else:
                        print("Result set not found.")
                else:
                    print("No current result set. Submit a query first.")
            
            elif cmd == "/evidence":
                if current_result_set_id:
                    rs = get_result_set(conn, current_result_set_id)
                    if rs:
                        limit = int(arg) if arg and arg.isdigit() else 10
                        print(f"\nResult Set #{rs['id']}: Showing {min(limit, rs['chunk_count'])} of {rs['chunk_count']} chunks")
                        evidence = get_detailed_evidence(conn, current_result_set_id, limit=limit)
                        print(format_detailed_evidence(evidence))
                        print()
                    else:
                        print("Result set not found.")
                else:
                    print("No current result set. Submit a query first.")
            
            elif cmd == "/summarize":
                # Parse optional response shape and flags from arg
                use_v3 = "--v3" in (arg or "")
                arg_clean = (arg or "").replace("--v3", "").strip()
                response_shape = arg_clean if arg_clean else None
                valid_shapes = ["roster", "timeline", "narrative", "key_docs", "qa", "index"]
                if response_shape and response_shape not in valid_shapes:
                    print(f"Invalid response shape. Valid options: {', '.join(valid_shapes)}")
                    print("Usage: /summarize [roster|timeline|narrative|qa|index] [--v3]")
                    continue
                
                # Use V4 interpretation if we have evidence (default), or V3 if --v3 flag
                if current_v3_result and current_v3_result.get("evidence_set"):
                    evidence_set = current_v3_result["evidence_set"]
                    query = current_v3_result.get("plan").query_text if current_v3_result.get("plan") else ""
                    
                    if use_v3:
                        # Use V3 summarizer (legacy)
                        print("\nGenerating V3 structured summary...")
                        try:
                            from retrieval.agent.v3_summarizer import summarize_from_evidence_set, detect_response_shape
                            
                            if not response_shape:
                                response_shape = detect_response_shape(query)
                                print(f"  [V3 Summarizer] Auto-detected shape: {response_shape}", file=sys.stderr)
                            
                            summary_result = summarize_from_evidence_set(
                                evidence_set=evidence_set,
                                query=query,
                                conn=conn,
                                response_shape=response_shape,
                            )
                            
                            print(summary_result.format_text())
                            
                            if not summary_result.verification_passed:
                                print(f"\n  Warning: {len(summary_result.verification_errors)} verification issues")
                            print()
                        except Exception as e:
                            print(f"\nV3 summarization error: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        # Use V4 interpretation (new default)
                        print("\nGenerating V4 interpretation...")
                        try:
                            from retrieval.agent.v4_interpret import (
                                interpret_evidence, 
                                detect_response_shape as v4_detect_shape,
                                prepare_spans_for_interpretation,
                            )
                            from retrieval.agent.v4_verify import verify_interpretation
                            from retrieval.agent.v4_render import render_interpretation
                            
                            if not response_shape:
                                response_shape = v4_detect_shape(query)
                                print(f"  [V4 Interpret] Auto-detected shape: {response_shape}", file=sys.stderr)
                            
                            # Prepare spans
                            prepared_spans = prepare_spans_for_interpretation(
                                evidence_set=evidence_set,
                                conn=conn,
                            )
                            
                            # Interpret
                            interpretation = interpret_evidence(
                                evidence_set=evidence_set,
                                query=query,
                                conn=conn,
                                response_shape=response_shape,
                            )
                            
                            # Verify
                            report = verify_interpretation(
                                interpretation=interpretation,
                                prepared_spans=prepared_spans,
                                conn=conn,
                            )
                            
                            # Render
                            response = render_interpretation(
                                interpretation=interpretation,
                                report=report,
                                prepared_spans=prepared_spans,
                            )
                            
                            print(response.format_text())
                            print()
                        except Exception as e:
                            print(f"\nV4 interpretation error: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fall back to V3
                            print("\nFalling back to V3 summarizer...")
                            try:
                                from retrieval.agent.v3_summarizer import summarize_from_evidence_set
                                summary_result = summarize_from_evidence_set(
                                    evidence_set=evidence_set,
                                    query=query,
                                    conn=conn,
                                )
                                print(summary_result.format_text())
                            except Exception as e2:
                                print(f"V3 fallback also failed: {e2}")
                elif current_result_set_id:
                    print("\nGenerating summary...")
                    summary = summarize_results(conn, current_result_set_id)
                    print(f"\n{summary}\n")
                else:
                    print("No current result set. Submit a query first.")
            
            elif cmd == "/mode":
                print(f"\nDefault mode: V7 (V6 + Citation Enforcement)")
                print("  V7 is now the default for all plain queries.")
                print("  Use /v7 <query> for explicit V7 (citation enforcement)")
                print("  Use /v6 <query> for V6 (no citation enforcement)")
                print("  Use /v4 <query> for V4 workflow")
                print("  Use /v3 <query> for V3 workflow")
                print("  Use /traditional <query> for traditional (non-agentic) workflow")
                print()
            
            elif cmd == "/agentic":
                if not arg:
                    print("Usage: /agentic <query>")
                    continue
                # Force agentic mode for this query
                print(f"\n[Agentic Mode] Processing: \"{arg}\"...")
                try:
                    result = execute_agentic_query(conn, session_id, arg)
                    print(format_agentic_result(result))
                    # Set result_set_id for /summarize
                    if result.get("result_set_id"):
                        current_result_set_id = result["result_set_id"]
                        print(f"\nType /summarize for an AI summary, or enter a new query.\n")
                    else:
                        print()
                except Exception as e:
                    print(f"\nAgentic workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/v2":
                if not arg:
                    print("Usage: /v2 <query>")
                    print("Examples:")
                    print("  /v2 members of the Silvermaster network")
                    print("  /v2 officers closely associated with Julius Rosenberg")
                    print("  /v2 Soviet agents in the OSS")
                    continue
                # Use V2 agentic workflow
                print(f"\n[V2 Agentic Mode] Processing: \"{arg}\"...")
                try:
                    result = execute_agentic_v2_query(conn, session_id, arg)
                    print(format_agentic_v2_result(result))
                    # Set result_set_id for /summarize
                    if result.get("result_set_id"):
                        current_result_set_id = result["result_set_id"]
                        print(f"\nType /summarize for an AI summary, or enter a new query.\n")
                    else:
                        print()
                except Exception as e:
                    print(f"\nV2 Agentic workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/v3":
                if not arg:
                    print("Usage: /v3 <query> [--dump-plan] [--dump-evidence] [--dump-verifier]")
                    print("V3 uses tool-based workflow with evidence banding and claim verification")
                    print("Examples:")
                    print("  /v3 evidence of Soviet proximity fuse acquisition")
                    print("  /v3 officers closely associated with Julius Rosenberg")
                    print("  /v3 Soviet agents in the OSS")
                    print("Debug flags:")
                    print("  --dump-plan      Show plan JSON")
                    print("  --dump-evidence  Show evidence set details")
                    print("  --dump-verifier  Show verification report")
                    continue
                
                # Parse debug flags
                dump_plan = "--dump-plan" in arg
                dump_evidence = "--dump-evidence" in arg
                dump_verifier = "--dump-verifier" in arg
                query_text = arg.replace("--dump-plan", "").replace("--dump-evidence", "").replace("--dump-verifier", "").strip()
                
                if not query_text:
                    print("Error: Please provide a query text after /v3")
                    continue
                
                # Use V3 agentic workflow (tool-based with claims)
                print(f"\n[V3 Agentic Mode] Processing: \"{query_text}\"...")
                try:
                    result = execute_agentic_v3_query(
                        conn, session_id, query_text,
                        dump_plan=dump_plan,
                        dump_evidence=dump_evidence,
                        dump_verifier=dump_verifier,
                    )
                    print(format_agentic_v3_result(result))
                    # Set result_set_id for /summarize
                    if result.get("result_set_id"):
                        current_result_set_id = result["result_set_id"]
                    # Store V3 result for structured summarization
                    if result.get("evidence_set"):
                        current_v3_result = result
                        print(f"\nType /summarize for a structured summary, or enter a new query.\n")
                    else:
                        current_v3_result = None
                        print()
                except Exception as e:
                    print(f"\nV3 Agentic workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/v4":
                if not arg:
                    print("Usage: /v4 <query> [flags]")
                    print("V4.2 uses reasoning-first interpretation with discovery loop")
                    print("Examples:")
                    print("  /v4 who were members of the Silvermaster network")
                    print("  /v4 --thorough evidence of Soviet proximity fuse acquisition")
                    print("  /v4 --no-discovery timeline of Rosenberg contacts")
                    print("Flags:")
                    print("  --no-discovery   Skip discovery loop")
                    print("  --thorough       Enable thorough mode (doubled budgets)")
                    print("  --dump-plan      Show plan JSON")
                    print("  --dump-evidence  Show evidence set details")
                    print("  --dump-verify    Show verification report")
                    print("  --dump-discovery Show discovery loop trace")
                    continue
                
                # Parse flags
                dump_plan = "--dump-plan" in arg
                dump_evidence = "--dump-evidence" in arg
                dump_verify = "--dump-verify" in arg
                dump_discovery = "--dump-discovery" in arg
                no_discovery = "--no-discovery" in arg
                thorough = "--thorough" in arg
                
                # Remove flags from query
                query_text = arg
                for flag in ["--dump-plan", "--dump-evidence", "--dump-verify", 
                             "--dump-discovery", "--no-discovery", "--thorough"]:
                    query_text = query_text.replace(flag, "")
                query_text = query_text.strip()
                
                if not query_text:
                    print("Error: Please provide a query text after /v4")
                    continue
                
                # Use V4.2 agentic workflow (reasoning-first + discovery)
                mode_str = "V4.2 Reasoning Mode"
                if thorough:
                    mode_str += " (thorough)"
                if no_discovery:
                    mode_str += " (no discovery)"
                print(f"\n[{mode_str}] Processing: \"{query_text}\"...")
                try:
                    from retrieval.agent.v4_runner import execute_v4_query, format_v4_result
                    
                    result = execute_v4_query(
                        conn, 
                        session_id, 
                        query_text,
                        discovery_enabled=not no_discovery,
                        thorough_mode=thorough,
                    )
                    
                    # Debug dumps
                    if dump_plan and result.get("plan"):
                        print("\n--- Plan ---")
                        import json
                        print(json.dumps(result["plan"].to_dict(), indent=2, default=str))
                    
                    if dump_evidence and result.get("evidence_set"):
                        print("\n--- Evidence Set ---")
                        es = result["evidence_set"]
                        print(f"  Cite spans: {len(es.cite_spans)}")
                        print(f"  Harvest spans: {len(es.harvest_spans)}")
                        if es.cite_spans:
                            print("  Top cite_spans:")
                            for i, s in enumerate(es.cite_spans[:5]):
                                print(f"    [{i}] {s.page_ref}: {s.quote[:80]}...")
                    
                    if dump_verify and result.get("verification"):
                        print("\n--- Verification Report ---")
                        v = result["verification"]
                        print(f"  Passed: {v.passed}")
                        print(f"  Hard errors: {len(v.hard_errors)}")
                        if v.hard_errors:
                            for e in v.hard_errors[:5]:
                                print(f"    [{e.error_type}] {e.details[:60]}...")
                        print(f"  Soft warnings: {len(v.soft_warnings)}")
                    
                    if dump_discovery and result.get("trace"):
                        print("\n--- Discovery Trace ---")
                        trace = result["trace"]
                        if hasattr(trace, 'discovery_trace') and trace.discovery_trace:
                            dt = trace.discovery_trace
                            print(f"  Rounds: {dt.get('rounds', [])}")
                            print(f"  Final chunks: {dt.get('final_chunk_count', 0)}")
                            print(f"  Stop reason: {dt.get('stop_decision', {}).get('reason', 'N/A')}")
                        elif hasattr(trace, 'to_dict'):
                            import json
                            d = trace.to_dict()
                            if d.get('discovery_trace'):
                                print(json.dumps(d['discovery_trace'], indent=2, default=str)[:2000])
                            else:
                                print("  Discovery not run or no trace available")
                    
                    # Print formatted result
                    print(format_v4_result(result))
                    
                    # Set result_set_id for /summarize
                    if result.get("result_set_id"):
                        current_result_set_id = result["result_set_id"]
                    
                    # Store V4 result for structured summarization (as V3 compatible format)
                    if result.get("evidence_set"):
                        current_v3_result = result
                        print(f"\nType /summarize for a re-interpretation, or enter a new query.\n")
                    else:
                        current_v3_result = None
                        print()
                        
                except Exception as e:
                    print(f"\nV4 workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/v6":
                if not arg:
                    print("Usage: /v6 <query>")
                    print("V6 uses principled architecture with CONTROL/CONTENT separation")
                    print("Examples:")
                    print("  /v6 who were members of the Silvermaster network?")
                    print("  /v6 list Soviet agents in the Treasury Department")
                    print("  /v6 provide citations from Vassiliev for Silvermaster members")
                    print("\nV6 features:")
                    print("  - Parses query into CONTROL vs CONTENT tokens")
                    print("  - Entity-links ONLY content tokens")
                    print("  - Hard bottleneck forces convergence (30-50 spans max)")
                    print("  - Responsiveness verification")
                    print("  - Progress-gated retrieval rounds")
                    continue
                
                query_text = arg.strip()
                
                if not query_text:
                    print("Error: Please provide a query text after /v6")
                    continue
                
                # Use V6 workflow
                print(f"\n[V6 Principled Mode] Processing: \"{query_text}\"...")
                try:
                    from retrieval.agent.v6_runner import run_v6_query
                    
                    result = run_v6_query(
                        conn=conn,
                        question=query_text,
                        max_bottleneck_spans=40,
                        max_rounds=5,
                        verbose=True,
                        pre_bundling_mode=pre_bundling_mode,
                        bottleneck_grading_mode=bottleneck_mode,
                    )
                    
                    # Print formatted answer
                    print("\n" + "=" * 60)
                    print("ANSWER:")
                    print("=" * 60)
                    print(result.format_answer())
                    
                    # Create result_set for /summarize compatibility
                    if result.trace and result.trace.rounds:
                        try:
                            from retrieval.ops import log_retrieval_run
                            from scripts.execute_plan import create_result_set
                            
                            # Get chunk IDs from bottlenecked spans (try both attribute names)
                            chunk_ids = []
                            spans = getattr(result.trace, 'bottleneck_spans', None) or getattr(result.trace, 'all_bottleneck_spans', []) or []
                            for span in spans:
                                cid = getattr(span, 'chunk_id', None)
                                if cid and cid not in chunk_ids:
                                    chunk_ids.append(cid)
                            
                            if chunk_ids:
                                chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
                                
                                run_id = log_retrieval_run(
                                    conn,
                                    query_text=f"[AGENTIC_V6] {query_text}",
                                    search_type="hybrid",
                                    chunk_pv=chunk_pv,
                                    embedding_model=None,
                                    top_k=len(chunk_ids),
                                    returned_chunk_ids=chunk_ids,
                                    retrieval_config_json={
                                        "mode": "agentic_v6",
                                        "task_type": result.trace.parsed_query.task_type.value if result.trace.parsed_query else "unknown",
                                        "rounds": len(result.trace.rounds),
                                        "responsiveness": result.responsiveness_status,
                                        "members_found": len(result.members_identified),
                                    },
                                    auto_commit=False,
                                    session_id=session_id,
                                )
                                conn.commit()
                                
                                result_set_id_v6 = create_result_set(
                                    conn,
                                    run_id=run_id,
                                    chunk_ids=chunk_ids,
                                    name=f"V6: {query_text[:40]}... (run {run_id})",
                                    session_id=session_id,
                                )
                                current_result_set_id = result_set_id_v6
                                print(f"\n  Created result set #{result_set_id_v6}", file=sys.stderr)
                        except Exception as e:
                            print(f"  Warning: Could not create result set: {e}", file=sys.stderr)
                    
                    print()
                        
                except Exception as e:
                    print(f"\nV6 workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/v9":
                if not arg:
                    print("Usage: /v9 <query>")
                    print("V9.2 Reasoning Workspace Loop")
                    print("Examples:")
                    print("  /v9 who were members of the Silvermaster network?")
                    print("  /v9 what evidence is there for Soviet awareness of the proximity fuse?")
                    print("  /v9 only from Vassiliev, list handlers of Rosenberg")
                    print("\nV9.2 features:")
                    print("  - Model drives tools (search, fetch, expand_entities w/ co-mentions)")
                    print("  - Auto-detects scope (collections, date range) from question")
                    print("  - Model self-labels task_type (roster/timeline/evidence/factual/relationship)")
                    print("  - Structured output: roster[]/timeline[]/evidence[]/edges[] by task_type")
                    print("  - Sufficiency contract with per-task-type thresholds")
                    print("  - Advisory responsiveness check → auto-extends if budget remains")
                    print("  - Verification is advisory only (never blocks answer)")
                    continue
                
                query_text = arg.strip()
                if not query_text:
                    print("Error: Please provide a query text after /v9")
                    continue
                
                print(f"\n[V9 Mode] Processing: \"{query_text}\"...")
                try:
                    from retrieval.agent.v9_runner import run_v9_query, format_v9_result
                    result = run_v9_query(conn=conn, question=query_text, verbose=True, max_tool_calls=query_max_turns)
                    print(format_v9_result(result, include_verification=True))
                    # Save workspace for /deeper
                    if result.workspace:
                        last_v9_workspace = result.workspace
                        last_v9_question = query_text
                        last_use_v9_fallback = True  # v9 resynthesis uses v9
                    if result.workspace and result.workspace.chunks:
                        # Clear any aborted transaction state from the V9 pipeline
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        try:
                            chunk_ids = result.workspace.chunk_ids()
                            if chunk_ids:
                                from retrieval.ops import log_retrieval_run
                                from scripts.execute_plan import create_result_set
                                chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
                                with conn.cursor() as cur:
                                    cur.execute("SELECT current_chunk_pv FROM retrieval_config_current LIMIT 1")
                                    row = cur.fetchone()
                                    if row and row[0]:
                                        chunk_pv = row[0]
                                run_id = log_retrieval_run(
                                    conn, query_text=f"[AGENTIC_V9] {query_text}",
                                    search_type="hybrid", chunk_pv=chunk_pv, embedding_model=None,
                                    top_k=len(chunk_ids), returned_chunk_ids=chunk_ids,
                                    retrieval_config_json={"mode": "agentic_v9"},
                                    auto_commit=False, session_id=session_id,
                                )
                                conn.commit()
                                result_set_id_v9 = create_result_set(
                                    conn, run_id=run_id, chunk_ids=chunk_ids,
                                    name=f"V9: {query_text[:40]}... (run {run_id})", session_id=session_id,
                                )
                                current_result_set_id = result_set_id_v9
                                print(f"\n  Created result set #{result_set_id_v9}", file=sys.stderr)
                        except Exception as e:
                            print(f"  Warning: Could not create result set: {e}", file=sys.stderr)
                    # Create v9_runs entry for Think Deeper DB persistence
                    try:
                        conn.rollback()  # clear any aborted transaction
                    except Exception:
                        pass
                    try:
                        from retrieval.agent.v9_session import create_run as create_v9_run, update_run_status
                        v9_run = create_v9_run(
                            conn, session_id=session_id, query_text=query_text,
                            mode="new_retrieval",
                        )
                        last_v9_run_id = v9_run.run_id
                        update_run_status(conn, v9_run.run_id, "paused")
                        print(f"  V9 run #{v9_run.run_id} created (paused, ready for /deeper)", file=sys.stderr)
                    except Exception as e:
                        print(f"  Warning: Could not create v9_run for Think Deeper: {e}", file=sys.stderr)
                        last_v9_run_id = None
                    print()
                except Exception as e:
                    print(f"\nV9 workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/v10":
                if not arg:
                    print("Usage: /v10 <query>")
                    print("V10 Scope-aware Alias Identity Layer")
                    print("Examples:")
                    print("  /v10 who was the Soviet handler for codename LIBERAL?")
                    print("  /v10 what do the Venona decrypts say about ANTENNA?")
                    print("  /v10 only from Vassiliev, list handlers of Rosenberg")
                    print("\nV10 features:")
                    print("  - Span-lattice query interpretation with dominance")
                    print("  - Collection-scoped alias semantics (no alias leakage)")
                    print("  - Multi-tier alias resolution: Referent Rules → Contextual → General → Fallback")
                    print("  - Structured entity/alias boosts (no query rewriting)")
                    print("  - Entity-aware grounding and alias-annotated rendering")
                    print("  - Full ThinkDeeper compatibility")
                    continue
                
                query_text = arg.strip()
                if not query_text:
                    print("Error: Please provide a query text after /v10")
                    continue
                
                print(f"\n[V10 Mode] Processing: \"{query_text}\"...")
                try:
                    from retrieval.agent.v10_runner import run_v10_query, format_v10_result
                    v10_result = run_v10_query(conn=conn, question=query_text, verbose=True, max_investigation_rounds=query_max_turns)
                    print(format_v10_result(v10_result, include_identity=True))
                    # Convert to V9Result for /deeper compatibility
                    result = v10_result.to_v9_result()
                    if result.workspace:
                        last_v9_workspace = result.workspace
                        last_v9_question = query_text
                        last_use_v9_fallback = True  # v10 resynthesis uses v9
                    if result.workspace and result.workspace.chunks:
                        # Clear any aborted transaction state
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        try:
                            chunk_ids = result.workspace.chunk_ids()
                            if chunk_ids:
                                from retrieval.ops import log_retrieval_run
                                from scripts.execute_plan import create_result_set
                                chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
                                with conn.cursor() as cur:
                                    cur.execute("SELECT current_chunk_pv FROM retrieval_config_current LIMIT 1")
                                    row = cur.fetchone()
                                    if row and row[0]:
                                        chunk_pv = row[0]
                                run_id = log_retrieval_run(
                                    conn, query_text=f"[AGENTIC_V10] {query_text}",
                                    search_type="hybrid", chunk_pv=chunk_pv, embedding_model=None,
                                    top_k=len(chunk_ids), returned_chunk_ids=chunk_ids,
                                    retrieval_config_json={"mode": "agentic_v10"},
                                    auto_commit=False, session_id=session_id,
                                )
                                conn.commit()
                                result_set_id_v10 = create_result_set(
                                    conn, run_id=run_id, chunk_ids=chunk_ids,
                                    name=f"V10: {query_text[:40]}... (run {run_id})", session_id=session_id,
                                )
                                current_result_set_id = result_set_id_v10
                                print(f"\n  Created result set #{result_set_id_v10}", file=sys.stderr)
                        except Exception as e:
                            print(f"  Warning: Could not create result set: {e}", file=sys.stderr)
                    # Create v9_runs entry for Think Deeper DB persistence
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        from retrieval.agent.v9_session import create_run as create_v9_run, update_run_status
                        v9_run = create_v9_run(
                            conn, session_id=session_id, query_text=query_text,
                            mode="new_retrieval",
                        )
                        last_v9_run_id = v9_run.run_id
                        update_run_status(conn, v9_run.run_id, "paused")
                        print(f"  V10 run #{v9_run.run_id} created (paused, ready for /deeper)", file=sys.stderr)
                    except Exception as e:
                        print(f"  Warning: Could not create v9_run for Think Deeper: {e}", file=sys.stderr)
                        last_v9_run_id = None
                    print()
                except Exception as e:
                    print(f"\nV10 workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/deeper":
                # Think Deeper: autonomous Actor/Judge loop on last V9 result
                if last_v9_workspace is None:
                    print("No result to deepen. Run a query first (plain text or /v10 <query>).")
                    continue
                
                user_followup = arg.strip() if arg else None
                followup_msg = f" with follow-up: \"{user_followup}\"" if user_followup else ""
                baseline_chunks = len(last_v9_workspace.fulltext_chunks)
                print(f"\n[Think Deeper] Deepening last V9 result for: \"{last_v9_question}\"{followup_msg}")
                print(f"  Baseline: {baseline_chunks} chunks in workspace")
                print("  Actor/Judge autonomous loop running...\n")
                
                try:
                    from retrieval.agent.v9_deep_runner import think_deeper
                    from retrieval.agent.v9_types import WorkspaceChunk
                    
                    td_result = think_deeper(
                        conn=conn,
                        seed_question=last_v9_question,
                        workspace=last_v9_workspace,
                        user_followup=user_followup,
                        max_steps=deeper_max_steps,
                        max_tool_calls=deeper_max_tool_calls,
                        verbose=True,
                        v9_run_id=last_v9_run_id,  # enables DB persistence
                    )
                    
                    # ── Display results ────────────────────────────────────
                    print(f"\n{'='*60}")
                    print(f"  THINK DEEPER RESULTS")
                    print(f"{'='*60}")
                    print(f"  Steps executed:   {td_result.steps_executed}")
                    print(f"  Tool calls used:  {td_result.tool_calls_used}")
                    print(f"  Stop reason:      {td_result.stop_reason}")
                    print(f"  Elapsed:          {td_result.elapsed_ms:.0f}ms")
                    
                    # Novelty report (prominent display)
                    nr = td_result.novelty_report
                    if nr:
                        if nr.new_docs:
                            print(f"  New documents:    {len(nr.new_docs)}")
                        if nr.what_changed:
                            print(f"\n  What changed:")
                            print(f"    {nr.what_changed}")
                        if nr.remaining_gaps:
                            print(f"\n  Remaining gaps:")
                            for gap in nr.remaining_gaps[:5]:
                                print(f"    - {gap}")
                    
                    # Verdict history summary
                    if td_result.verdict_history:
                        print(f"\n  Step-by-step scores:")
                        for i, v in enumerate(td_result.verdict_history):
                            print(
                                f"    Step {i}: ans={v.answeredness:.2f} "
                                f"nov={v.material_novelty:.2f} "
                                f"conf={v.confidence:.2f} "
                                f"{'[STOP]' if v.stop_recommendation else ''}"
                            )
                    
                    # Findings (all new, non-baseline)
                    if td_result.finding_store_entries:
                        non_baseline = [e for e in td_result.finding_store_entries if e.finding_type != "baseline"]
                        if non_baseline:
                            print(f"\n  New findings ({len(non_baseline)}):")
                            for entry in non_baseline[:15]:
                                cids = ",".join(str(c) for c in entry.supporting_chunk_ids[:3])
                                print(f"    [{entry.finding_type}] {entry.text[:140]} (chunks: {cids})")
                    
                    # ── Merge new chunks into workspace for chaining ──────
                    existing_ids = {c.chunk_id for c in last_v9_workspace.fulltext_chunks}
                    merged_count = 0
                    for cc in td_result.selected_chunks:
                        if cc.chunk_id not in existing_ids:
                            last_v9_workspace.fulltext_chunks.append(
                                WorkspaceChunk(
                                    chunk_id=cc.chunk_id,
                                    text=cc.text,
                                    doc_id=cc.doc_id,
                                    page=cc.page,
                                    source_label=cc.collection_slug,
                                    collection_slug=cc.collection_slug,
                                    score=cc.score,
                                )
                            )
                            existing_ids.add(cc.chunk_id)
                            merged_count += 1
                    
                    new_total = len(last_v9_workspace.fulltext_chunks)
                    print(f"\n  Evidence: {merged_count} new chunks merged into workspace")
                    print(f"  Workspace now: {new_total} chunks (was {baseline_chunks})")
                    
                    # ── Re-synthesize answer using enriched workspace ─────
                    # Run V9 or V11 on the expanded workspace with a small tool budget
                    # so it focuses on synthesis rather than more retrieval.
                    # Build Findings Brief from Think Deeper FindingStore for grounded claims.
                    print(f"\n  Re-synthesizing answer with enriched evidence...")
                    try:
                        from retrieval.agent.v9_runner import format_v9_result
                        from retrieval.agent.v9_deep_findings import build_findings_brief
                        chunk_id_to_label = {c.chunk_id: f"{c.collection_slug or ''} {c.page or ''}".strip()
                                            for c in td_result.selected_chunks}
                        findings_brief = build_findings_brief(
                            td_result.finding_store_entries, top_n=10,
                            chunk_id_to_label=chunk_id_to_label,
                        ) if td_result.finding_store_entries else None
                        if last_use_v9_fallback:
                            from retrieval.agent.v9_runner import run_v9_query
                            resynthesized = run_v9_query(
                                conn=conn,
                                question=last_v9_question,
                                verbose=False,
                                _resume_workspace=last_v9_workspace,
                                max_tool_calls=3,  # small budget: focus on synthesis, not more retrieval
                                _findings_brief=findings_brief,
                            )
                        else:
                            from retrieval.agent.v11_runner import run_v11_query
                            resynthesized = run_v11_query(
                                conn=conn,
                                question=last_v9_question,
                                verbose=False,
                                _resume_workspace=last_v9_workspace,
                                max_tool_calls=3,
                                use_lightweight_pem=os.getenv("V11_USE_LIGHTWEIGHT_PEM", "0").strip().lower() in ("1", "true", "yes"),
                                _findings_brief=findings_brief,
                            )
                        print(f"\n{'='*60}")
                        print(f"  UPDATED ANSWER (with Think Deeper evidence)")
                        print(f"{'='*60}")
                        print(format_v9_result(resynthesized, include_verification=True))
                        
                        # Update workspace with the synthesis result
                        if resynthesized.workspace:
                            last_v9_workspace = resynthesized.workspace
                    except Exception as synth_err:
                        print(f"  Warning: Re-synthesis failed: {synth_err}", file=sys.stderr)
                        # Fallback: show Think Deeper narrative so user always gets something
                        if td_result.narrative:
                            print(f"\n{'='*60}")
                            print(f"  THINK DEEPER SUMMARY (re-synthesis skipped)")
                            print(f"{'='*60}")
                            print(td_result.narrative)
                        print(f"  (New evidence is still merged into workspace for next /deeper)")
                    
                    print(f"{'='*60}")
                    print(f"\n  Type /deeper again to continue deepening, or enter a new query.\n")
                    
                except Exception as e:
                    print(f"\nThink Deeper error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/v7":
                if not arg:
                    print("Usage: /v7 <query>")
                    print("V7 extends V6 with citation enforcement")
                    print("Examples:")
                    print("  /v7 who were members of the Silvermaster network?")
                    print("  /v7 list Soviet agents in the Treasury Department")
                    print("\nV7 features (all V6 features PLUS):")
                    print("  - Claim enumeration: extracts atomic claims from answer")
                    print("  - Stop gate: validates all claims have citations")
                    print("  - Expanded summary: output shows claims with references")
                    print("  - Citation enforcement: uncited claims are dropped")
                    continue
                
                query_text = arg.strip()
                
                if not query_text:
                    print("Error: Please provide a query text after /v7")
                    continue
                
                # Use V7 workflow
                print(f"\n[V7 Citation Enforcement Mode] Processing: \"{query_text}\"...")
                try:
                    from retrieval.agent.v7_runner import run_v7_query, format_v7_result
                    
                    result = run_v7_query(
                        conn=conn,
                        question=query_text,
                        max_bottleneck_spans=40,
                        max_rounds=5,
                        drop_uncited_claims=True,
                        verbose=True,
                        pre_bundling_mode=pre_bundling_mode,
                        bottleneck_grading_mode=bottleneck_mode,
                    )
                    
                    # Print formatted result
                    print(format_v7_result(result, include_evidence=False, include_v6_trace=False))
                    
                    # Create result_set for /summarize compatibility
                    if result.trace and result.trace.v6_trace and result.trace.v6_trace.rounds:
                        try:
                            from retrieval.ops import log_retrieval_run
                            from scripts.execute_plan import create_result_set
                            
                            # Get chunk IDs from bottlenecked spans
                            chunk_ids = []
                            spans = getattr(result.trace.v6_trace, 'bottleneck_spans', None) or []
                            for span in spans:
                                cid = getattr(span, 'chunk_id', None)
                                if cid and cid not in chunk_ids:
                                    chunk_ids.append(cid)
                            
                            if chunk_ids:
                                chunk_pv = os.getenv("DEFAULT_CHUNK_PV", "chunk_v1_full")
                                
                                run_id = log_retrieval_run(
                                    conn,
                                    query_text=f"[AGENTIC_V7] {query_text}",
                                    search_type="hybrid",
                                    chunk_pv=chunk_pv,
                                    embedding_model=None,
                                    top_k=len(chunk_ids),
                                    returned_chunk_ids=chunk_ids,
                                    retrieval_config_json={
                                        "mode": "agentic_v7",
                                        "citation_validation_passed": result.citation_validation_passed,
                                        "claims_valid": result.trace.claims_valid if result.trace else 0,
                                        "claims_dropped": result.trace.claims_unsupported if result.trace else 0,
                                    },
                                    auto_commit=False,
                                    session_id=session_id,
                                )
                                conn.commit()
                                
                                result_set_id_v7 = create_result_set(
                                    conn,
                                    run_id=run_id,
                                    chunk_ids=chunk_ids,
                                    name=f"V7: {query_text[:40]}... (run {run_id})",
                                    session_id=session_id,
                                )
                                current_result_set_id = result_set_id_v7
                                print(f"\n  Created result set #{result_set_id_v7}", file=sys.stderr)
                        except Exception as e:
                            print(f"  Warning: Could not create result set: {e}", file=sys.stderr)
                    
                    print()
                        
                except Exception as e:
                    print(f"\nV7 workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/discover":
                if not arg:
                    print("Usage: /discover <query> [--thorough]")
                    print("Run discovery loop only (debug mode)")
                    print("Shows coverage metrics and span previews without interpretation")
                    continue
                
                thorough = "--thorough" in arg
                query_text = arg.replace("--thorough", "").strip()
                
                if not query_text:
                    print("Error: Please provide a query text after /discover")
                    continue
                
                print(f"\n[Discovery Debug Mode] Processing: \"{query_text}\"...")
                try:
                    from retrieval.agent.v4_runner import V4Runner
                    from retrieval.agent.v4_discover import run_discovery
                    from retrieval.agent.v4_discovery_metrics import (
                        THOROUGH_BUDGETS, DEFAULT_BUDGETS as DISCOVERY_BUDGETS
                    )
                    from retrieval.agent.v3_plan import generate_plan
                    from retrieval.agent.executor import ToolExecutor
                    
                    budgets = THOROUGH_BUDGETS if thorough else DISCOVERY_BUDGETS
                    
                    # Run initial V3 plan to get starting chunks
                    print("  [Initial] Running V3 plan for seed chunks...")
                    plan = generate_plan(query_text, conn)
                    executor = ToolExecutor(verbose=True)
                    exec_result = executor.execute_plan(plan, conn)
                    
                    print(f"  [Initial] Got {len(exec_result.chunk_ids)} initial chunks")
                    
                    # Run discovery loop
                    expanded_chunks, discovery_trace = run_discovery(
                        query=query_text,
                        initial_chunk_ids=exec_result.chunk_ids,
                        conn=conn,
                        budgets=budgets,
                        verbose=True,
                    )
                    
                    # Print detailed results
                    print("\n" + "=" * 60)
                    print("Discovery Results")
                    print("=" * 60)
                    print(f"  Initial chunks: {len(exec_result.chunk_ids)}")
                    print(f"  Final chunks: {len(expanded_chunks)}")
                    print(f"  Discovery rounds: {len(discovery_trace.rounds)}")
                    print(f"  Total time: {discovery_trace.total_elapsed_ms:.0f}ms")
                    
                    if discovery_trace.final_coverage:
                        cov = discovery_trace.final_coverage
                        print(f"\n  Coverage:")
                        print(f"    - Total spans: {cov.total_spans}")
                        print(f"    - Unique docs: {cov.unique_docs}")
                        print(f"    - List-like spans: {cov.list_like_span_count}")
                        print(f"    - Definitional spans: {cov.definitional_span_count}")
                        
                        if cov.entity_attest_counts:
                            print(f"    - Top entity attestations:")
                            for surface, count in sorted(
                                cov.entity_attest_counts.items(), 
                                key=lambda x: -x[1]
                            )[:5]:
                                print(f"        '{surface}': {count}")
                    
                    print(f"\n  Stop reason: {discovery_trace.stop_decision.reason}")
                    print()
                    
                except Exception as e:
                    print(f"\nDiscovery error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            elif cmd == "/traditional":
                if not arg:
                    print("Usage: /traditional <query>")
                    continue
                # Force traditional (non-agentic) mode for this query
                print(f"\n[Traditional Mode] Planning query: \"{arg}\"...")
                try:
                    # Generate plan using the planner
                    plan = plan_query(conn, session_id, arg)
                    if not plan:
                        print("Failed to create plan. Try again.")
                        continue
                    plan_id = plan["id"]
                    plan_json = plan["plan_json"]
                    
                    print(f"\n--- Research Plan #{plan_id} ---")
                    print(format_plan(plan_json))
                    print()
                    
                    # Ask for approval
                    response = input("Execute this plan? [Y/n/q]: ").strip().lower()
                    if response in ("q", "quit"):
                        continue
                    if response not in ("", "y", "yes"):
                        print("Plan not executed.\n")
                        continue
                    
                    print("\nApproving and executing plan...")
                    approve_plan(conn, plan_id)
                    
                    result = execute_plan_direct(conn, plan_id)
                    
                    if result and result.get("result_set_id"):
                        current_result_set_id = result["result_set_id"]
                        hit_count = result.get("hit_count", 0)
                        
                        if hit_count > 0:
                            print(f"\n[OK] Found {hit_count} results (Result Set #{current_result_set_id})")
                            preview = get_result_preview(conn, current_result_set_id, limit=5)
                            if preview:
                                print("\nTop results preview:")
                                print(format_results_preview(preview))
                            print("\nCommands:")
                            print("  /evidence    - Show detailed evidence with full text")
                            print("  /summarize   - Generate AI summary of results\n")
                        else:
                            print(f"\nQuery executed but found 0 matching chunks (Result Set #{current_result_set_id})")
                            print("Try broadening your search terms or removing filters.\n")
                    else:
                        print("\n[FAIL] Execution did not return results.\n")
                except Exception as e:
                    print(f"\nTraditional workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                continue
            
            else:
                print(f"Unknown command: {cmd}. Type /help for available commands.")
            
            continue
        
        # Process as a query - V11 is the DEFAULT (canonical embeddings, stripped V9)
        # USE_V9_AGENT=1 → v9; /v9 → v9, /v10 → v10, /v7 → v7, etc.
        
        query_text = user_input
        use_v9_as_default = os.getenv("USE_V9_AGENT", "0").strip().lower() in ("1", "true", "yes")
        if use_v9_as_default:
            print(f"\n[V9 Mode] Processing: \"{query_text}\"... (USE_V9_AGENT=1)")
        else:
            print(f"\n[V11 Mode] Processing: \"{query_text}\"...")
        
        try:
            from retrieval.agent.v9_runner import format_v9_result
            if use_v9_as_default:
                from retrieval.agent.v9_runner import run_v9_query
                result = run_v9_query(conn=conn, question=query_text, verbose=True, max_tool_calls=query_max_turns)
            else:
                from retrieval.agent.v11_runner import run_v11_query
                result = run_v11_query(
                    conn=conn,
                    question=query_text,
                    verbose=True,
                    max_tool_calls=query_max_turns,
                    use_lightweight_pem=os.getenv("V11_USE_LIGHTWEIGHT_PEM", "0").strip().lower() in ("1", "true", "yes"),
                )
            
            # Save workspace for /deeper
            if result.workspace:
                last_v9_workspace = result.workspace
                last_v9_question = query_text
                last_use_v9_fallback = use_v9_as_default
            
            # Create v9_run early (needed for Tier 1 and /deeper)
            last_v9_run_id = None
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                from retrieval.agent.v9_session import create_run as create_v9_run, update_run_status
                v9_run = create_v9_run(
                    conn, session_id=session_id, query_text=query_text,
                    mode="new_retrieval",
                )
                last_v9_run_id = v9_run.run_id
                update_run_status(conn, v9_run.run_id, "paused")
                mode_tag = "V9" if use_v9_as_default else "V11"
                print(f"  {mode_tag} run #{v9_run.run_id} created (paused, ready for /deeper)", file=sys.stderr)
            except Exception as e:
                print(f"  Warning: Could not create v9_run for Think Deeper: {e}", file=sys.stderr)
            
            # Tier 1 auto-deepen: run BEFORE we output. Defer output until Tier 1 is done.
            if tier1_auto and last_v9_workspace and last_v9_question:
                from retrieval.agent.v9_deep_tiering import (
                    compute_tier1_triggers,
                    think_deeper_tier1,
                    _count_grounded_claims,
                )
                triggers = compute_tier1_triggers(result, query_text)
                run_tier1, use_micro = triggers.should_run_tier1()
                if run_tier1:
                    if triggers.reasons:
                        print(f"  [Tier 1] Auto-deepen triggered: {', '.join(triggers.reasons)}", file=sys.stderr)
                    original_grounded = _count_grounded_claims(result)
                    try:
                        td_result = think_deeper_tier1(
                            conn=conn,
                            seed_question=last_v9_question,
                            workspace=last_v9_workspace,
                            micro=use_micro,
                            verbose=True,
                            v9_run_id=last_v9_run_id,
                        )
                        # Merge new chunks into workspace
                        from retrieval.agent.v9_types import WorkspaceChunk
                        existing_ids = {c.chunk_id for c in last_v9_workspace.fulltext_chunks}
                        merged_count = 0
                        for cc in td_result.selected_chunks:
                            if cc.chunk_id not in existing_ids:
                                last_v9_workspace.fulltext_chunks.append(
                                    WorkspaceChunk(
                                        chunk_id=cc.chunk_id,
                                        text=cc.text,
                                        doc_id=cc.doc_id,
                                        page=cc.page,
                                        source_label=cc.collection_slug,
                                        collection_slug=cc.collection_slug,
                                        score=cc.score,
                                    )
                                )
                                existing_ids.add(cc.chunk_id)
                                merged_count += 1
                        if merged_count > 0:
                            # Re-synthesize with enriched evidence
                            from retrieval.agent.v9_runner import format_v9_result
                            from retrieval.agent.v9_deep_findings import build_findings_brief
                            chunk_id_to_label = {c.chunk_id: f"{c.collection_slug or ''} {c.page or ''}".strip()
                                                for c in td_result.selected_chunks}
                            findings_brief = build_findings_brief(
                                td_result.finding_store_entries, top_n=10,
                                chunk_id_to_label=chunk_id_to_label,
                            ) if td_result.finding_store_entries else None
                            print(f"  Re-synthesizing with {merged_count} new chunks...", file=sys.stderr)
                            if use_v9_as_default:
                                from retrieval.agent.v9_runner import run_v9_query
                                resynth = run_v9_query(
                                    conn=conn,
                                    question=last_v9_question,
                                    verbose=False,
                                    _resume_workspace=last_v9_workspace,
                                    max_tool_calls=3,
                                    _findings_brief=findings_brief,
                                )
                            else:
                                from retrieval.agent.v11_runner import run_v11_query
                                resynth = run_v11_query(
                                    conn=conn,
                                    question=last_v9_question,
                                    verbose=False,
                                    _resume_workspace=last_v9_workspace,
                                    max_tool_calls=3,
                                    use_lightweight_pem=os.getenv("V11_USE_LIGHTWEIGHT_PEM", "0").strip().lower() in ("1", "true", "yes"),
                                    _findings_brief=findings_brief,
                                )
                            # Only adopt re-synth if it's better (has grounded claims)
                            resynth_grounded = _count_grounded_claims(resynth)
                            if resynth_grounded >= original_grounded:
                                result = resynth
                                if result.workspace:
                                    last_v9_workspace = result.workspace
                                extra = " (micro)" if use_micro else ""
                                print(f"  [Tier 1] Quick deeper pass{extra} added {merged_count} chunks.", file=sys.stderr)
                            else:
                                print(f"  [Tier 1] Re-synthesis produced fewer grounded claims; keeping original answer.", file=sys.stderr)
                        elif merged_count == 0:
                            print(f"  [Tier 1] No new chunks; keeping original answer.", file=sys.stderr)
                    except Exception as tier1_err:
                        print(f"  Warning: Tier 1 auto-deepen failed: {tier1_err}", file=sys.stderr)
            
            # Output once, after Tier 1 (if any) is complete
            print(format_v9_result(result, include_verification=True))
            
            # Create result set for summarize compatibility
            if result.workspace and result.workspace.chunks:
                # Clear any aborted transaction state
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    chunk_ids = result.workspace.chunk_ids()
                    if chunk_ids:
                        chunk_pv = None
                        with conn.cursor() as cur:
                            cur.execute("SELECT current_chunk_pv FROM retrieval_config_current LIMIT 1")
                            row = cur.fetchone()
                            if row:
                                chunk_pv = row[0]
                        
                        if chunk_pv:
                            from retrieval.ops import log_retrieval_run
                            from scripts.execute_plan import create_result_set
                            
                            mode_tag = "V9" if use_v9_as_default else "V11"
                            run_id = log_retrieval_run(
                                conn=conn,
                                query_text=f"[AGENTIC_{mode_tag}] {query_text}",
                                search_type="hybrid",
                                chunk_pv=chunk_pv,
                                embedding_model=None,
                                top_k=len(chunk_ids),
                                returned_chunk_ids=chunk_ids,
                                retrieval_config_json={
                                    "mode": f"agentic_{mode_tag.lower()}",
                                },
                                auto_commit=False,
                                session_id=session_id,
                            )
                            conn.commit()
                            
                            result_set_id = create_result_set(
                                conn,
                                run_id=run_id,
                                chunk_ids=chunk_ids,
                                name=f"{mode_tag}: {query_text[:40]}... (run {run_id})",
                                session_id=session_id,
                            )
                            current_result_set_id = result_set_id
                            print(f"  Created result set #{result_set_id}", file=sys.stderr)
                except Exception as e:
                    print(f"  Warning: Could not create result set: {e}", file=sys.stderr)
            
            print()
                
        except Exception as e:
            print(f"\nV11 workflow error: {e}")
            import traceback
            traceback.print_exc()
            print()
        continue
        
        # Ask for approval unless auto-execute
        try:
            if auto_execute:
                execute = True
            else:
                response = input("Execute this plan? [Y/n/q]: ").strip().lower()
                execute = response in ("", "y", "yes")
                if response in ("q", "quit"):
                    continue
            
            if execute:
                print("\nApproving and executing plan...")
                
                # Approve
                approve_plan(conn, plan_id)
                
                # Execute
                result = execute_plan_direct(conn, plan_id)
                
                if result and result.get("error"):
                    # Execution failed with error
                    print(f"\n[FAIL] Execution failed:")
                    error_msg = result.get("error", "Unknown error")
                    # Show first few lines of error
                    error_lines = error_msg.split('\n')[:5]
                    for line in error_lines:
                        if line.strip():
                            print(f"  {line}")
                    if len(error_msg.split('\n')) > 5:
                        print("  ...")
                    print()
                elif result and result.get("result_set_id"):
                    current_result_set_id = result["result_set_id"]
                    hit_count = result.get("hit_count", 0)
                    
                    if hit_count > 0:
                        print(f"\n[OK] Found {hit_count} results (Result Set #{current_result_set_id})")
                        
                        # Show preview (top 5 results)
                        preview = get_result_preview(conn, current_result_set_id, limit=5)
                        if preview:
                            print("\nTop results preview:")
                            print(format_results_preview(preview))
                        
                        print("\nCommands:")
                        print("  /evidence    - Show detailed evidence with full text")
                        print("  /summarize   - Generate AI summary of results")
                        print("  <new query>  - Run a new search\n")
                    else:
                        print(f"\n⚠ Query executed but found 0 matching chunks (Result Set #{current_result_set_id})")
                        print("Try broadening your search terms or removing filters.\n")
                else:
                    print("\n[FAIL] Execution did not return results.")
                    # Show debug info
                    if result:
                        output = result.get("output", "")
                        if output:
                            print("\nExecution output:")
                            # Show last 20 lines
                            lines = [l for l in output.split('\n') if l.strip()]
                            for line in lines[-20:]:
                                print(f"  {line}")
                        else:
                            print("  (no output captured)")
                    print()
            else:
                print("Plan not executed. Enter a new query or /quit to exit.\n")
        
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    conn.close()

# =============================================================================
# One-Shot Mode
# =============================================================================

def one_shot_query(session_label: str, query: str, auto_execute: bool = True, force_agentic: bool = False, force_v2: bool = False, pre_bundling_mode: str = "micro", bottleneck_mode: str = "score"):
    """Execute a single query and print results."""
    conn = get_conn()
    
    # Get or create session
    session_id = create_session(conn, session_label)
    
    print(f"Session: {session_label} (ID: {session_id})")
    print(f"Query: {query}")
    print()
    
    # Check if V2 mode requested
    if force_v2:
        print("[V2 Agentic Mode]")
        print()
        try:
            result = execute_agentic_v2_query(conn, session_id, query)
            print(format_agentic_v2_result(result))
            conn.close()
            return
        except Exception as e:
            print(f"V2 Agentic workflow error: {e}")
            import traceback
            traceback.print_exc()
            print("\nFalling back to V1 agentic mode...\n")
    
    # Check if we should use agentic mode (enabled by default for all queries)
    # Default agentic path is V10 (Scope-aware Alias Identity Layer)
    use_agentic = force_agentic or AGENTIC_MODE_ENABLED
    
    if use_agentic:
        print("[V10 Mode - Scope-aware Alias Identity Layer]")
        print()
        try:
            from retrieval.agent.v10_runner import run_v10_query, format_v10_result
            v10_result = run_v10_query(conn=conn, question=query, verbose=True)
            print(format_v10_result(v10_result, include_identity=True))
            conn.close()
            return
        except Exception as e:
            print(f"V10 workflow error: {e}")
            import traceback
            traceback.print_exc()
            print("\nFalling back to traditional mode...\n")
    
    # Traditional mode
    try:
        # Generate plan
        plan = plan_query(conn, session_id, query)
        if not plan:
            print("Failed to create plan.")
            conn.close()
            return
        plan_id = plan["id"]
        plan_json = plan["plan_json"]
        
        print("Research Plan:")
        print(format_plan(plan_json))
        print()
        
        if auto_execute:
            # Approve and execute
            approve_plan(conn, plan_id)
            
            result = execute_plan_direct(conn, plan_id)
            
            if result and result.get("result_set_id"):
                result_set_id = result["result_set_id"]
                hit_count = result.get("hit_count", 0)
                
                print(f"Found {hit_count} results")
                print()
                
                # Show preview
                preview = get_result_preview(conn, result_set_id, limit=10)
                if preview:
                    print("Results:")
                    print(format_results_preview(preview))
                
                # Summarize
                if os.getenv("OPENAI_API_KEY"):
                    print("\n\nSummary:")
                    summary = summarize_results(conn, result_set_id)
                    print(summary)
    
    except Exception as e:
        print(f"Error: {e}")
    
    conn.close()

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Friday Research Console - Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/friday_cli.py
  
  # With a specific session
  python scripts/friday_cli.py --session "Rosenberg research"
  
  # One-shot query
  python scripts/friday_cli.py --query "Who was involved in atomic espionage?"
  
  # One-shot with auto-execute
  python scripts/friday_cli.py -q "Oppenheimer security clearance" --auto-execute
  
  # Force agentic mode
  python scripts/friday_cli.py -q "Who were Rosenberg's handlers?" --agentic
  
  # Disable agentic mode
  python scripts/friday_cli.py -q "silvermaster" --no-agentic
"""
    )
    
    parser.add_argument(
        "--session", "-s",
        type=str,
        default="cli-session",
        help="Session ID or label (default: cli-session)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query (non-interactive mode)"
    )
    parser.add_argument(
        "--auto-execute", "-y",
        action="store_true",
        help="Automatically approve and execute plans"
    )
    parser.add_argument(
        "--agentic", "-a",
        action="store_true",
        help="Force V1 agentic workflow (Plan→Execute→Verify→Render)"
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use V2 agentic workflow (FocusBundle + constraints)"
    )
    parser.add_argument(
        "--no-agentic",
        action="store_true",
        help="Disable agentic mode, use traditional workflow"
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List recent sessions and exit"
    )
    parser.add_argument(
        "--pre-bundling",
        choices=["off", "passthrough", "micro", "semantic"],
        default="micro",
        help="Pre-bundling mode (default: micro)"
    )
    parser.add_argument(
        "--bottleneck-mode",
        choices=["score", "absolute", "tournament"],
        default="score",
        help="Bottleneck grading: score (fastest, no LLM), absolute, tournament (default: score)"
    )
    parser.add_argument(
        "--deeper-max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Think Deeper: max turns (default 8). Use 2-3 for quick tests. Env: THINK_DEEPER_MAX_STEPS"
    )
    parser.add_argument(
        "--query-max-turns",
        type=int,
        default=None,
        metavar="N",
        help="Initial query: max tool calls (V9/V11) or rounds (V10). Default 5. Use 2 for fast /deeper setup. Env: V11_QUERY_MAX_TURNS"
    )
    parser.add_argument(
        "--no-tier1-auto",
        action="store_true",
        help="Disable auto Tier 1 deepen (run Tier 1 only when /deeper is invoked)"
    )
    
    args = parser.parse_args()
    
    # Update agentic mode based on flags
    global AGENTIC_MODE_ENABLED
    if args.no_agentic:
        AGENTIC_MODE_ENABLED = False
    
    # List sessions mode
    if args.list_sessions:
        conn = get_conn()
        sessions = list_sessions(conn, limit=20)
        print("\nRecent sessions:")
        for sid, label, created in sessions:
            print(f"  [{sid}] {label} ({created.strftime('%Y-%m-%d %H:%M')})")
        conn.close()
        return
    
    # Get or create session
    conn = get_conn()
    try:
        session_id = resolve_session(conn, args.session)
    except ValueError:
        session_id = create_session(conn, args.session)
        print(f"Created new session: {args.session}")
    conn.close()
    
    # Deeper max steps: CLI flag > env var > default
    deeper_max_steps = args.deeper_max_steps
    if deeper_max_steps is None:
        deeper_max_steps = int(os.getenv("THINK_DEEPER_MAX_STEPS", "8"))
    
    # Query max turns: CLI flag > env var > default (speeds up initial retrieval for /deeper)
    query_max_turns = args.query_max_turns
    if query_max_turns is None:
        query_max_turns = int(os.getenv("V11_QUERY_MAX_TURNS", "5"))
    
    # Deeper Tier 2: max tool calls (default 10, can go to 15)
    deeper_max_tool_calls = int(os.getenv("THINK_DEEPER_MAX_TOOL_CALLS", "10"))
    
    # Tier 1 auto-deepen: enable unless --no-tier1-auto
    tier1_auto = not args.no_tier1_auto and os.getenv("THINK_DEEPER_TIER1_AUTO", "1").strip().lower() in ("1", "true", "yes")
    
    # One-shot or interactive
    if args.query:
        one_shot_query(args.session, args.query, args.auto_execute, force_agentic=args.agentic, force_v2=args.v2, pre_bundling_mode=args.pre_bundling, bottleneck_mode=args.bottleneck_mode)
    else:
        interactive_mode(session_id, args.auto_execute, pre_bundling_mode=args.pre_bundling, bottleneck_mode=args.bottleneck_mode, deeper_max_steps=deeper_max_steps, deeper_max_tool_calls=deeper_max_tool_calls, query_max_turns=query_max_turns, tier1_auto=tier1_auto)


if __name__ == "__main__":
    main()
