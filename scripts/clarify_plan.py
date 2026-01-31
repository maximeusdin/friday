#!/usr/bin/env python3
"""
clarify_plan.py --plan <id> --choice <index|text>

Handles interactive clarification for ambiguous plans.
Stores user's choice and regenerates the plan with clarified context.

Usage:
    # Choose by index (1-based)
    python scripts/clarify_plan.py --plan 123 --choice 1
    
    # Choose by text (partial match)
    python scripts/clarify_plan.py --plan 123 --choice "Ethel Rosenberg"
    
    # Interactive mode (prompts for choice)
    python scripts/clarify_plan.py --plan 123

The clarified plan supersedes the original plan (status='superseded').
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import Json


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL environment variable")
    return psycopg2.connect(dsn)


def load_plan(conn, plan_id: int) -> Dict[str, Any]:
    """Load a plan by ID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                id, session_id, status, user_utterance, plan_json, 
                plan_hash, query_lang_version, created_at
            FROM research_plans
            WHERE id = %s
            """,
            (plan_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Plan not found: {plan_id}")
        
        return {
            "id": row[0],
            "session_id": row[1],
            "status": row[2],
            "user_utterance": row[3],
            "plan_json": row[4],
            "plan_hash": row[5],
            "query_lang_version": row[6],
            "created_at": row[7],
        }


def get_clarification_choices(plan: Dict[str, Any]) -> List[str]:
    """Extract clarification choices from a plan."""
    plan_json = plan.get("plan_json", {})
    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)
    
    if not plan_json.get("needs_clarification"):
        raise ValueError(f"Plan {plan['id']} does not need clarification (needs_clarification=false)")
    
    choices = plan_json.get("choices", [])
    if not choices:
        raise ValueError(f"Plan {plan['id']} has needs_clarification=true but no choices provided")
    
    return choices


def resolve_choice(choices: List[str], choice_input: str) -> tuple[int, str]:
    """
    Resolve user's choice input to a (index, choice_text) tuple.
    
    Accepts:
    - Numeric index (1-based)
    - Partial text match (case-insensitive)
    """
    # Try numeric index first
    try:
        idx = int(choice_input)
        if 1 <= idx <= len(choices):
            return idx - 1, choices[idx - 1]
        raise ValueError(f"Index {idx} out of range. Choose 1-{len(choices)}.")
    except ValueError:
        pass
    
    # Try text match
    choice_lower = choice_input.lower()
    matches = []
    for i, c in enumerate(choices):
        if choice_lower in c.lower():
            matches.append((i, c))
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(
            f"Ambiguous choice '{choice_input}' matches multiple options:\n" +
            "\n".join(f"  {i+1}. {c}" for i, c in matches)
        )
    else:
        raise ValueError(f"No match found for '{choice_input}'")


def supersede_plan(conn, plan_id: int) -> None:
    """Mark a plan as superseded."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE research_plans SET status = 'superseded' WHERE id = %s",
            (plan_id,),
        )
        conn.commit()


def regenerate_plan_with_choice(
    conn,
    original_plan: Dict[str, Any],
    choice_index: int,
    choice_text: str,
) -> int:
    """
    Regenerate the plan with the user's choice.
    
    This modifies the original utterance to include the clarified entity name,
    then calls plan_query.py to generate a new plan.
    
    Returns the new plan_id.
    """
    import subprocess
    
    plan_json = original_plan.get("plan_json", {})
    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)
    
    # Extract clarification context
    clarification_ctx = plan_json.get("_clarification_context", {})
    ambiguous_name = clarification_ctx.get("ambiguous_name", "")
    candidates = clarification_ctx.get("candidates", [])
    
    # Build clarified utterance
    original_utterance = original_plan["user_utterance"]
    
    # If we have entity resolution context, inject the chosen entity_id
    # Otherwise, append the choice to make it explicit
    if candidates and choice_index < len(candidates):
        chosen_entity = candidates[choice_index]
        entity_id = chosen_entity.get("entity_id")
        canonical_name = chosen_entity.get("canonical_name", choice_text)
        
        # Replace the ambiguous name with the canonical name
        if ambiguous_name and ambiguous_name.lower() in original_utterance.lower():
            # Case-insensitive replacement
            import re
            pattern = re.escape(ambiguous_name)
            clarified_utterance = re.sub(
                pattern, 
                canonical_name, 
                original_utterance, 
                flags=re.IGNORECASE
            )
        else:
            clarified_utterance = f"{original_utterance} (specifically: {canonical_name})"
    else:
        # Fallback: append the chosen text to clarify
        clarified_utterance = f"{original_utterance} (specifically: {choice_text})"
    
    # Store clarification metadata
    clarification_metadata = {
        "original_plan_id": original_plan["id"],
        "choice_index": choice_index,
        "choice_text": choice_text,
        "original_utterance": original_utterance,
        "clarified_utterance": clarified_utterance,
    }
    
    # Mark original plan as superseded
    supersede_plan(conn, original_plan["id"])
    
    # Call plan_query.py with the clarified utterance
    from scripts.plan_query import (
        resolve_session_id, get_session_summary, get_recent_result_sets,
        get_most_recent_retrieval_run, get_collections, get_conversation_history,
        parse_collection_scope, detect_deictics, detect_all_results, 
        resolve_deictics_to_result_set, resolve_entities_in_utterance, 
        build_llm_prompt, call_llm_structured, normalize_plan_dict, 
        validate_and_normalize, inject_within_result_set_if_needed,
        inject_all_result_sets, build_execution_envelope, compute_plan_hash,
        ResearchPlan,
    )
    from retrieval.date_parser import resolve_dates_in_utterance
    
    session_id = original_plan["session_id"]
    query_lang_version = original_plan.get("query_lang_version", "qir_v1")
    
    # Gather context
    session_summary = get_session_summary(conn, session_id)
    recent_result_sets = get_recent_result_sets(conn, session_id)
    recent_run = get_most_recent_retrieval_run(conn, session_id)
    collections = get_collections(conn)
    conversation_history = get_conversation_history(conn, session_id, limit=10)
    
    # Parse collection scope from clarified utterance
    cleaned_utterance, explicit_collection_scope = parse_collection_scope(clarified_utterance)
    
    # Detect deictics
    detected_deictics = detect_deictics(cleaned_utterance)
    wants_all_results = detect_all_results(cleaned_utterance)
    resolved_rs_id = resolve_deictics_to_result_set(detected_deictics, recent_result_sets)
    
    # Resolve dates
    try:
        date_context, date_info = resolve_dates_in_utterance(clarified_utterance)
    except Exception:
        date_context = []
    
    # Resolve entities (this time should resolve unambiguously)
    entity_context, resolved_entities, clarification_needed = resolve_entities_in_utterance(
        conn, clarified_utterance
    )
    
    if clarification_needed:
        # Still ambiguous - save as clarification plan
        clarification_needed["query"]["raw"] = clarified_utterance
        clarification_needed["_metadata"] = {
            "clarification_history": [clarification_metadata]
        }
        plan_dict = normalize_plan_dict(clarification_needed)
        plan_hash = compute_plan_hash(plan_dict)
        
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_plans (
                    session_id, plan_json, plan_hash, query_lang_version,
                    retrieval_impl_version, status, user_utterance, parent_plan_id
                )
                VALUES (%s, %s, %s, %s, %s, 'proposed', %s, %s)
                RETURNING id
                """,
                (session_id, Json(plan_dict), plan_hash, query_lang_version,
                 "retrieval_v1", clarified_utterance, original_plan["id"]),
            )
            plan_id = cur.fetchone()[0]
            conn.commit()
        
        print(f"Warning: Clarification still needed. New plan ID: {plan_id}", file=sys.stderr)
        return plan_id
    
    # Build prompt and call LLM
    prompt = build_llm_prompt(
        clarified_utterance,
        session_summary,
        recent_result_sets,
        collections,
        detected_deictics,
        entity_context,
        conversation_history,
        date_context,
        query_lang_version,
    )
    
    print("Calling LLM to generate clarified plan...", file=sys.stderr)
    plan_dict_raw = call_llm_structured(prompt, model=None)
    
    # Process as usual
    plan_dict_raw["query"]["raw"] = clarified_utterance
    plan_dict = normalize_plan_dict(plan_dict_raw)
    
    # Add clarification metadata
    plan_dict.setdefault("_metadata", {})
    plan_dict["_metadata"]["clarification_history"] = [clarification_metadata]
    
    # Inject deictics if needed
    prims = plan_dict.get("query", {}).get("primitives", [])
    has_within_without_id = any(
        isinstance(p, dict) and p.get("type") == "WITHIN_RESULT_SET" 
        and (p.get("result_set_id") is None or (isinstance(p.get("result_set_id"), int) and p.get("result_set_id") <= 0))
        for p in prims
    )
    
    if wants_all_results and recent_result_sets:
        plan_dict = inject_all_result_sets(plan_dict, recent_result_sets)
    elif has_within_without_id and recent_result_sets:
        injection_rs_id = resolved_rs_id if resolved_rs_id is not None else recent_result_sets[0]["id"]
        plan_dict = inject_within_result_set_if_needed(plan_dict, injection_rs_id)
    elif resolved_rs_id is not None:
        plan_dict = inject_within_result_set_if_needed(plan_dict, resolved_rs_id)
    
    # Inject collection filter if needed
    if explicit_collection_scope:
        prims = plan_dict.get("query", {}).get("primitives", [])
        has_collection_filter = any(
            isinstance(p, dict) and 
            p.get("type") == "FILTER_COLLECTION" and 
            p.get("slug") == explicit_collection_scope
            for p in prims
        )
        if not has_collection_filter:
            prims.append({"type": "FILTER_COLLECTION", "slug": explicit_collection_scope})
            plan_dict.setdefault("query", {})["primitives"] = prims
    
    # Validate
    plan_dict, errors = validate_and_normalize(plan_dict)
    if errors:
        raise RuntimeError(f"Plan validation failed: {errors}")
    
    # Parse and build envelope
    plan = ResearchPlan.from_dict(plan_dict)
    plan.query.raw = clarified_utterance
    
    execution_envelope = build_execution_envelope(
        plan,
        recent_run=recent_run,
        default_chunk_pv="chunk_v1_full",
        default_k=20,
        explicit_collection_scope=explicit_collection_scope,
    )
    plan.execution_envelope = execution_envelope
    
    # Compile
    plan.compile()
    
    # Save
    plan_dict = plan.to_dict()
    plan_hash = compute_plan_hash(plan_dict)
    
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_plans (
                session_id, plan_json, plan_hash, query_lang_version,
                retrieval_impl_version, status, user_utterance, parent_plan_id
            )
            VALUES (%s, %s, %s, %s, %s, 'proposed', %s, %s)
            RETURNING id
            """,
            (session_id, Json(plan_dict), plan_hash, query_lang_version,
             "retrieval_v1", clarified_utterance, original_plan["id"]),
        )
        plan_id = cur.fetchone()[0]
        conn.commit()
    
    return plan_id


def interactive_choice(choices: List[str]) -> tuple[int, str]:
    """Prompt user to choose interactively."""
    print("\nPlease choose one of the following options:")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    print()
    
    while True:
        try:
            user_input = input("Enter your choice (number or text): ").strip()
            if not user_input:
                continue
            return resolve_choice(choices, user_input)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Handle clarification for ambiguous plans"
    )
    ap.add_argument(
        "--plan", type=int, required=True,
        help="Plan ID that needs clarification"
    )
    ap.add_argument(
        "--choice", type=str, default=None,
        help="Choice index (1-based) or text to match"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making changes"
    )
    args = ap.parse_args()
    
    conn = get_conn()
    try:
        # Load the plan
        plan = load_plan(conn, args.plan)
        print(f"Loaded plan {plan['id']} (status: {plan['status']})", file=sys.stderr)
        
        # Get choices
        choices = get_clarification_choices(plan)
        print(f"Plan requires clarification with {len(choices)} choices", file=sys.stderr)
        
        # Resolve choice
        if args.choice:
            choice_index, choice_text = resolve_choice(choices, args.choice)
        else:
            choice_index, choice_text = interactive_choice(choices)
        
        print(f"Selected: {choice_index + 1}. {choice_text}", file=sys.stderr)
        
        if args.dry_run:
            print("\n[DRY RUN] Would regenerate plan with choice:", file=sys.stderr)
            print(f"  Original utterance: {plan['user_utterance']}", file=sys.stderr)
            print(f"  Choice: {choice_text}", file=sys.stderr)
            return
        
        # Regenerate the plan
        new_plan_id = regenerate_plan_with_choice(conn, plan, choice_index, choice_text)
        print(f"\n✅ New plan created with ID: {new_plan_id}", file=sys.stderr)
        print(f"   Original plan {plan['id']} marked as 'superseded'", file=sys.stderr)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
