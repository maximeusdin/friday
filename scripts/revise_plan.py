#!/usr/bin/env python3
"""
revise_plan.py --plan-id <id> --text "..."

NL → Patch (primitive-level revisions)

- Uses OpenAI Structured Outputs (json_schema, strict=true) to generate patch ops
- Patches operate ONLY on query.primitives (no full plan rewrites)
- Creates a new research_plans row with parent_plan_id=<old plan>
- Status = proposed
- Every revision is auditable

STRICT schema requirements (OpenAI Structured Outputs):
- No oneOf/anyOf
- Every object: additionalProperties=false
- Every object: required must include ALL keys in properties
- "Optional" fields must exist but can be null/empty
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import Json
from openai import OpenAI

from retrieval.primitives import (
    ResearchPlan,
    validate_plan_json,
    compute_plan_hash,
)
from retrieval.plan_validation import validate_primitives, validate_plan

# =============================================================================
# Database
# =============================================================================

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL environment variable")
    return psycopg2.connect(dsn)


def get_plan(conn, plan_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve plan by ID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, session_id, user_utterance, plan_json, plan_hash,
                   query_lang_version, retrieval_impl_version, status, parent_plan_id
            FROM research_plans
            WHERE id = %s
            """,
            (plan_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "session_id": row[1],
            "user_utterance": row[2],
            "plan_json": row[3],
            "plan_hash": row[4],
            "query_lang_version": row[5],
            "retrieval_impl_version": row[6],
            "status": row[7],
            "parent_plan_id": row[8],
        }

# =============================================================================
# Patch Operations
# =============================================================================

def _primitive_matches(primitive: Dict[str, Any], match: Dict[str, Any]) -> bool:
    """Check if primitive matches all criteria in match dict."""
    if not match:
        return False
    for key, value in match.items():
        if primitive.get(key) != value:
            return False
    return True


def apply_patch(plan_dict: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply patch operations to plan_dict.
    Returns new plan_dict (does not modify original).
    
    Invariant B: Detects accidental filter removal and warns.
    """
    new_plan = json.loads(json.dumps(plan_dict))  # deep copy

    new_plan.setdefault("query", {})
    new_plan["query"].setdefault("primitives", [])
    prims: List[Dict[str, Any]] = new_plan["query"]["primitives"]
    
    # Track original filter primitives for safeguard
    original_filters = [
        p for p in prims 
        if isinstance(p, dict) and p.get("type") in ("FILTER_COLLECTION", "FILTER_DOCUMENT", "FILTER_DATE_RANGE")
    ]

    for op in patch.get("ops", []):
        op_type = op.get("op")

        if op_type == "ADD_PRIMITIVE":
            value = op.get("value")
            if isinstance(value, dict) and value.get("type"):
                prims.append(value)

        elif op_type == "REMOVE_PRIMITIVE":
            match = op.get("match") or {}
            # Invariant B: Check if removing a filter primitive
            removed_filters = [
                p for p in prims 
                if _primitive_matches(p, match) and 
                   isinstance(p, dict) and 
                   p.get("type") in ("FILTER_COLLECTION", "FILTER_DOCUMENT", "FILTER_DATE_RANGE")
            ]
            if removed_filters:
                filter_types = [p.get("type") for p in removed_filters]
                print(f"WARNING: Patch removes filter primitives: {filter_types}", file=sys.stderr)
                print(f"  This may change the plan's scope. Ensure this is intentional.", file=sys.stderr)
            prims[:] = [p for p in prims if not _primitive_matches(p, match)]

        elif op_type == "MODIFY_PRIMITIVE":
            match = op.get("match") or {}
            updates = op.get("updates") or {}
            for i, p in enumerate(prims):
                if _primitive_matches(p, match):
                    original_type = p.get("type")
                    prims[i] = {**p, **updates}
                    if original_type:
                        prims[i]["type"] = original_type
                    break

        elif op_type == "REPLACE_PRIMITIVE":
            match = op.get("match") or {}
            value = op.get("value")
            if isinstance(value, dict) and value.get("type"):
                for i, p in enumerate(prims):
                    if _primitive_matches(p, match):
                        prims[i] = value
                        break

    new_plan["query"]["primitives"] = prims
    return new_plan

# =============================================================================
# LLM Prompt
# =============================================================================

def build_patch_prompt(original_utterance: str, revision_instruction: str, current_plan: Dict[str, Any]) -> str:
    prims_str = json.dumps(current_plan.get("query", {}).get("primitives", []), indent=2)

    return f"""Generate patch operations to revise a query plan.

ORIGINAL QUERY:
"{original_utterance}"

REVISION INSTRUCTION:
"{revision_instruction}"

CURRENT PLAN PRIMITIVES (ONLY THESE MAY BE CHANGED):
{prims_str}

RULES:
- Return ONLY JSON matching the schema (no prose).
- Ops may only target query.primitives. Do NOT rewrite the whole plan.
- Prefer minimal changes.
- Match primitives using exact fields in "match" (typically type + value, or type + result_set_id).

OP TYPES:
- ADD_PRIMITIVE: add a new primitive (e.g., TERM, PHRASE, FILTER_COLLECTION)
- REMOVE_PRIMITIVE: remove any primitive matching "match"
- MODIFY_PRIMITIVE: update fields on a matched primitive
- REPLACE_PRIMITIVE: replace a matched primitive entirely

EXAMPLES:
- "broaden to include X" -> ADD_PRIMITIVE with TERM/PHRASE for X
- "remove Y" -> REMOVE_PRIMITIVE matching TERM/PHRASE Y
- "change treasury to state department" -> REPLACE_PRIMITIVE
"""

# =============================================================================
# STRICT JSON Schema for patch output (FIXED)
# =============================================================================

def get_patch_json_schema() -> Dict[str, Any]:
    """
    Strict mode requires:
      - additionalProperties=false everywhere
      - required includes all keys in properties everywhere

    Strategy:
      - op objects ALWAYS include: op, value, match, updates
      - value/match/updates are "fat primitive" objects with all keys required,
        nullable values allowed
      - We'll compact away irrelevant fields after parsing.
    """

    primitive_type_enum = [
        "TERM","PHRASE","WITHIN_RESULT_SET","EXCLUDE_RESULT_SET",
        "FILTER_COLLECTION","FILTER_DOCUMENT","FILTER_DATE_RANGE",
        "SET_TOP_K","SET_SEARCH_TYPE","SET_TERM_MODE","OR_GROUP",
        "TOGGLE_CONCORDANCE_EXPANSION",
        "ENTITY","CO_OCCURS_WITH","INTERSECT_DATE_WINDOWS","FILTER_COUNTRY",
        "CO_LOCATED","RELATION_EVIDENCE","REQUIRE_EVIDENCE","GROUP_BY",
    ]

    nullable_str = {"type": ["string", "null"]}
    nullable_int = {"type": ["integer", "null"]}
    nullable_bool = {"type": ["boolean", "null"]}
    nullable_value = {"type": ["string", "integer", "null"]}

    empty_obj = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    primitive_properties = {
        "type": {"type": ["string", "null"], "enum": primitive_type_enum + [None]},
        "value": nullable_value,
        "slug": nullable_str,
        "document_id": nullable_int,
        "result_set_id": nullable_int,
        "entity_id": nullable_int,
        "entity_a": nullable_int,
        "entity_b": nullable_int,
        "start": nullable_str,
        "end": nullable_str,
        "window": nullable_str,
        "scope": nullable_str,
        "field": nullable_str,
        "evidence_type": nullable_str,
        "enabled": nullable_bool,
        "source_slug": nullable_str,
        "primitives": {"type": "array", "items": empty_obj},
    }
    primitive_required = list(primitive_properties.keys())

    fat_primitive = {
        "type": "object",
        "properties": primitive_properties,
        "required": primitive_required,
        "additionalProperties": False,
    }

    op_properties = {
        "op": {
            "type": "string",
            "enum": ["ADD_PRIMITIVE", "REMOVE_PRIMITIVE", "MODIFY_PRIMITIVE", "REPLACE_PRIMITIVE"],
        },
        "value": fat_primitive,
        "match": fat_primitive,
        "updates": fat_primitive,
    }
    op_required = list(op_properties.keys())

    op_schema = {
        "type": "object",
        "properties": op_properties,
        "required": op_required,
        "additionalProperties": False,
    }

    top_properties = {"ops": {"type": "array", "items": op_schema}}
    top_required = list(top_properties.keys())

    return {
        "type": "object",
        "properties": top_properties,
        "required": top_required,
        "additionalProperties": False,
    }

# =============================================================================
# Compacting helpers
# =============================================================================

def _strip_nulls(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[k] = _strip_nulls(v)
        return out
    if isinstance(obj, list):
        return [_strip_nulls(x) for x in obj if x is not None]
    return obj


def _clean_primitive_value(value: Any) -> Any:
    """
    Clean primitive value by removing JSON artifacts.
    Returns cleaned value, or None if value is completely consumed by artifacts.
    """
    if not isinstance(value, str):
        return value
    
    # Remove trailing JSON artifacts (more aggressive cleaning)
    # Pattern: "Treasury},{" -> "Treasury"
    cleaned = value
    # Remove trailing patterns like "},{" or "}]," or "}," or "}"
    while cleaned.endswith(("},", "}]", "}", "],", "]", ",")):
        cleaned = cleaned[:-1].rstrip()
    # Remove leading patterns like "{", "[", ","
    while cleaned.startswith(("{", "[", ",")):
        cleaned = cleaned[1:].lstrip()
    # Final strip of whitespace
    cleaned = cleaned.strip()
    
    # If value was completely consumed by artifacts, return None
    if not cleaned.strip():
        return None
    
    return cleaned

def _compact_primitive(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    d2 = _strip_nulls(d)
    if "type" not in d2:
        return {}
    if d2.get("type") is None:
        # In strict schema, type can be null; treat as absent.
        d2.pop("type", None)

    # Clean TERM and PHRASE values to remove JSON artifacts
    ptype = d2.get("type")
    if ptype in ("TERM", "PHRASE") and "value" in d2:
        cleaned_value = _clean_primitive_value(d2["value"])
        if cleaned_value is None:
            return {}  # Skip this primitive if value is empty after cleaning
        d2["value"] = cleaned_value

    # Drop primitives recursion unless OR_GROUP (and we still don't allow nested in patches)
    if d2.get("type") != "OR_GROUP":
        d2.pop("primitives", None)
    else:
        d2["primitives"] = []

    # Keep only meaningful keys
    keep = {}
    for k, v in d2.items():
        if v is None:
            continue
        keep[k] = v
    return keep


def _compact_patch(patch: Dict[str, Any]) -> Dict[str, Any]:
    patch = _strip_nulls(patch)
    out_ops = []
    for op in patch.get("ops", []):
        if not isinstance(op, dict):
            continue
        op_type = op.get("op")
        if not op_type:
            continue

        op_out: Dict[str, Any] = {"op": op_type}

        if op_type == "ADD_PRIMITIVE":
            v = _compact_primitive(op.get("value"))
            if v:
                op_out["value"] = v

        elif op_type == "REMOVE_PRIMITIVE":
            m = _compact_primitive(op.get("match"))
            if m:
                op_out["match"] = m

        elif op_type == "MODIFY_PRIMITIVE":
            m = _compact_primitive(op.get("match"))
            u = _compact_primitive(op.get("updates"))
            if m:
                op_out["match"] = m
            if u:
                op_out["updates"] = u

        elif op_type == "REPLACE_PRIMITIVE":
            m = _compact_primitive(op.get("match"))
            v = _compact_primitive(op.get("value"))
            if m:
                op_out["match"] = m
            if v:
                op_out["value"] = v

        out_ops.append(op_out)

    return {"ops": out_ops}

# =============================================================================
# LLM Call (Structured Outputs)
# =============================================================================

def call_llm_structured(prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    if model is None:
        model = os.getenv("OPENAI_MODEL_PLAN", "gpt-5-mini")

    client = OpenAI(api_key=api_key)
    schema = get_patch_json_schema()

    request_params: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY JSON matching the schema. No prose."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "patch_operations",
                "schema": schema,
                "strict": True,
            },
        },
    }

    # Do NOT pass temperature=None
    fixed_temp_models = ["gpt-5-mini", "o1", "o1-mini", "o1-preview"]
    use_temp = not any(model == m or model.startswith(f"{m}-") for m in fixed_temp_models)
    if use_temp:
        request_params["temperature"] = 0.1

    resp = client.beta.chat.completions.parse(**request_params)
    if not resp.choices:
        raise RuntimeError("OpenAI returned no choices")

    msg = resp.choices[0].message
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            patch = parsed.model_dump()
        elif hasattr(parsed, "dict"):
            patch = parsed.dict()
        elif isinstance(parsed, dict):
            patch = parsed
        else:
            patch = json.loads(json.dumps(parsed, default=str))
        # _compact_patch will clean values via _compact_primitive -> _clean_primitive_value
        return _compact_patch(patch)

    # fallback: parsed may be None on gpt-5-mini even though content is JSON
    content = getattr(msg, "content", None)
    if not content:
        raise RuntimeError("Structured Outputs failed: message.parsed is None and message.content empty")

    patch = json.loads(content)
    return _compact_patch(patch)

# =============================================================================
# Persistence
# =============================================================================

def save_revised_plan(
    conn,
    session_id: int,
    parent_plan_id: int,
    revised_plan_dict: Dict[str, Any],
    revision_instruction: str,
    query_lang_version: str,
) -> int:
    plan_hash = compute_plan_hash(revised_plan_dict)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT retrieval_impl_version, user_utterance FROM research_plans WHERE id = %s",
            (parent_plan_id,),
        )
        row = cur.fetchone()
        retrieval_impl_version = row[0] if row else "unknown"
        original_utterance = row[1] if row else ""

        combined_utterance = f"{original_utterance} [revised: {revision_instruction}]"

        cur.execute(
            """
            INSERT INTO research_plans
            (session_id, user_utterance, plan_json, plan_hash,
             query_lang_version, retrieval_impl_version, status, parent_plan_id)
            VALUES (%s, %s, %s, %s, %s, %s, 'proposed', %s)
            RETURNING id
            """,
            (
                session_id,
                combined_utterance,
                Json(revised_plan_dict),
                plan_hash,
                query_lang_version,
                retrieval_impl_version,
                parent_plan_id,
            ),
        )
        plan_id = cur.fetchone()[0]
        conn.commit()
        return plan_id

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Revise a query plan using natural language patches")
    ap.add_argument("--plan-id", type=int, required=True, help="ID of plan to revise")
    ap.add_argument("--text", type=str, required=True, help="Revision instruction")
    ap.add_argument("--model", type=str, default=None, help="OpenAI model (default: OPENAI_MODEL_PLAN or gpt-5-mini)")
    ap.add_argument("--query-lang-version", type=str, default="qir_v1", help="Query language version")
    ap.add_argument("--dry-run", action="store_true", help="Don't save to database, just show revised plan")
    args = ap.parse_args()

    conn = get_conn()
    try:
        original_plan = get_plan(conn, args.plan_id)
        if not original_plan:
            print(f"ERROR: Plan {args.plan_id} not found", file=sys.stderr)
            sys.exit(1)

        print(f"Original Plan ID: {args.plan_id}", file=sys.stderr)
        print(f"Original Utterance: {original_plan['user_utterance']}", file=sys.stderr)
        print(f"Revision Instruction: {args.text}", file=sys.stderr)
        
        # Show original plan primitives for debugging
        original_prims = original_plan["plan_json"].get("query", {}).get("primitives", [])
        print(f"\nOriginal Plan Primitives ({len(original_prims)}):", file=sys.stderr)
        from retrieval.primitives import QueryPlan
        for i, p_dict in enumerate(original_prims, 1):
            if isinstance(p_dict, dict):
                p_type = p_dict.get("type", "UNKNOWN")
                if p_type == "TERM":
                    print(f"  {i}. TERM: \"{p_dict.get('value', '')}\"", file=sys.stderr)
                elif p_type == "PHRASE":
                    print(f"  {i}. PHRASE: \"{p_dict.get('value', '')}\"", file=sys.stderr)
                elif p_type == "WITHIN_RESULT_SET":
                    print(f"  {i}. WITHIN_RESULT_SET: result_set_id={p_dict.get('result_set_id', 'N/A')}", file=sys.stderr)
                else:
                    print(f"  {i}. {p_type}: {json.dumps(p_dict, default=str)}", file=sys.stderr)

        prompt = build_patch_prompt(
            original_plan["user_utterance"],
            args.text,
            original_plan["plan_json"],
        )

        print("Calling LLM to generate patch (Structured Outputs)...", file=sys.stderr)
        patch = call_llm_structured(prompt, model=args.model)

        ops = patch.get("ops", [])
        print(f"Generated patch with {len(ops)} operations", file=sys.stderr)

        revised_plan_dict = apply_patch(original_plan["plan_json"], patch)

        # Preserve original query.raw
        revised_plan_dict.setdefault("query", {})
        revised_plan_dict["query"]["raw"] = original_plan["plan_json"]["query"]["raw"]
        
        # Preserve execution_envelope from original plan (if present)
        original_envelope = original_plan["plan_json"].get("execution_envelope")
        if original_envelope:
            revised_plan_dict["execution_envelope"] = original_envelope

        # Validate revised plan structure
        errors = validate_plan_json(revised_plan_dict)
        if errors:
            print("ERROR: Revised plan validation failed:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            print(json.dumps(revised_plan_dict, indent=2)[:1200], file=sys.stderr)
            sys.exit(1)
        
        # Strict validation for JSON leakage
        prims = revised_plan_dict.get("query", {}).get("primitives", [])
        leakage_errors = validate_primitives(prims)
        if leakage_errors:
            print("ERROR: Primitive value validation failed (JSON leakage detected):", file=sys.stderr)
            for err in leakage_errors:
                print(f"  - {err}", file=sys.stderr)
            print(json.dumps(revised_plan_dict, indent=2)[:1200], file=sys.stderr)
            sys.exit(1)
        
        # Validate plan invariants (envelope-primitive consistency)
        invariant_errors = validate_plan(revised_plan_dict)
        if invariant_errors:
            print("ERROR: Revised plan invariant validation failed:", file=sys.stderr)
            for err in invariant_errors:
                print(f"  - {err}", file=sys.stderr)
            print(json.dumps(revised_plan_dict, indent=2)[:1200], file=sys.stderr)
            sys.exit(1)

        # Parse & compile to verify
        try:
            plan = ResearchPlan.from_dict(revised_plan_dict)
            plan.query.raw = revised_plan_dict["query"]["raw"]
            
            # Rebuild execution envelope if needed (inherit from parent or use defaults)
            if not plan.execution_envelope:
                # Import here to avoid circular dependency (sys already imported at top)
                from pathlib import Path
                REPO_ROOT = Path(__file__).resolve().parents[1]
                if str(REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT))
                from scripts.plan_query import get_most_recent_retrieval_run, build_execution_envelope
                recent_run = get_most_recent_retrieval_run(conn, original_plan["session_id"])
                plan.execution_envelope = build_execution_envelope(
                    plan,
                    recent_run=recent_run,
                    default_chunk_pv="chunk_v1_full",
                    default_k=20,
                    explicit_collection_scope=None,  # Don't re-parse scope on revision
                )
                # Update revised_plan_dict with new envelope
                revised_plan_dict["execution_envelope"] = plan.execution_envelope
            
            plan.compile()
        except Exception as e:
            print(f"ERROR: Plan compilation failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Display patch ops + primitives
        print("\n" + "=" * 80, file=sys.stderr)
        print("PATCH OPERATIONS", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        for i, op in enumerate(ops, 1):
            print(f"{i}. {op.get('op')}: {json.dumps(op, indent=2)}", file=sys.stderr)

        print("\n" + "=" * 80, file=sys.stderr)
        print("REVISED PLAN SUMMARY", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"\nPrimitives ({len(plan.query.primitives)}):", file=sys.stderr)

        from retrieval.primitives import QueryPlan
        for i, p in enumerate(plan.query.primitives, 1):
            p_dict = QueryPlan._primitive_to_dict(p)
            p_type = p_dict.get("type", "UNKNOWN")
            if p_type == "TERM":
                print(f"  {i}. TERM: \"{p_dict.get('value', '')}\"", file=sys.stderr)
            elif p_type == "PHRASE":
                print(f"  {i}. PHRASE: \"{p_dict.get('value', '')}\"", file=sys.stderr)
            elif p_type == "WITHIN_RESULT_SET":
                print(f"  {i}. WITHIN_RESULT_SET: result_set_id={p_dict.get('result_set_id', 'N/A')}", file=sys.stderr)
            elif p_type == "FILTER_COLLECTION":
                print(f"  {i}. FILTER_COLLECTION: slug=\"{p_dict.get('slug', '')}\"", file=sys.stderr)
            else:
                print(f"  {i}. {p_type}: {json.dumps(p_dict, default=str)}", file=sys.stderr)

        if getattr(plan, "compiled", None):
            compiled = plan.compiled
            if isinstance(compiled, dict):
                tsq = compiled.get("tsquery", {})
                if isinstance(tsq, dict):
                    print(f"\nCompiled tsquery: {tsq.get('text', 'N/A')}", file=sys.stderr)

        # Save
        if not args.dry_run:
            revised_plan_id = save_revised_plan(
                conn,
                original_plan["session_id"],
                args.plan_id,
                revised_plan_dict,
                args.text,
                args.query_lang_version,
            )
            print(
                f"\n✅ Revised plan saved with ID: {revised_plan_id} (session={original_plan['session_id']}, status: proposed, parent: {args.plan_id})",
                file=sys.stderr,
            )
        else:
            print("\n[DRY RUN] Revised plan not saved to database", file=sys.stderr)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
