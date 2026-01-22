#!/usr/bin/env python3
"""
plan_query.py --session <id|label> --text "..."

NL -> Plan (primitive-only), with deterministic deictic resolution.
- Uses OpenAI Structured Outputs (json_schema, strict=true)
- LLM generates plan JSON ONLY (no prose)
- Code resolves deictics deterministically
- Plan is validated, compiled, rendered, and stored as status='proposed'
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
    TermPrimitive,
    PhrasePrimitive,
    WithinResultSetPrimitive,
    FilterCollectionPrimitive,
    FilterDocumentPrimitive,
    SetTopKPrimitive,
    SetSearchTypePrimitive,
)
from retrieval.plan_validation import validate_primitives, validate_plan

# =============================================================================
# DB
# =============================================================================

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL environment variable")
    return psycopg2.connect(dsn)

# =============================================================================
# Session resolution / context
# =============================================================================

def resolve_session_id(conn, session_arg: str) -> int:
    with conn.cursor() as cur:
        if session_arg.isdigit():
            cur.execute("SELECT id FROM research_sessions WHERE id = %s", (int(session_arg),))
            row = cur.fetchone()
            if row:
                return row[0]
        cur.execute("SELECT id FROM research_sessions WHERE label = %s", (session_arg,))
        row = cur.fetchone()
        if row:
            return row[0]
    raise SystemExit(f"Session not found: '{session_arg}' (not an ID or label)")

def get_session_summary(conn, session_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT label, created_at FROM research_sessions WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return "Session not found"
        label, created_at = row

        cur.execute(
            """
            SELECT
                COUNT(DISTINCT rr.id) AS runs,
                COUNT(DISTINCT rs.id) AS result_sets
            FROM research_sessions s
            LEFT JOIN retrieval_runs rr ON rr.session_id = s.id
            LEFT JOIN result_sets rs ON rs.session_id = s.id
            WHERE s.id = %s
            """,
            (session_id,),
        )
        runs, result_sets = cur.fetchone() or (0, 0)
        return f"Session '{label}' (created {created_at.strftime('%Y-%m-%d')}, {runs} runs, {result_sets} result sets)"

def get_recent_result_sets(conn, session_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, created_at
            FROM result_sets
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        return [{"id": r[0], "label": r[1], "created_at": r[2].isoformat()} for r in cur.fetchall()]

def get_most_recent_retrieval_run(conn, session_id: int) -> Optional[Dict[str, Any]]:
    """Get the most recent retrieval_run from a session for inheritance."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                id,
                chunk_pv,
                top_k,
                retrieval_config_json,
                search_type,
                created_at
            FROM retrieval_runs
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "chunk_pv": row[1],
            "top_k": row[2],
            "retrieval_config_json": row[3],
            "search_type": row[4],
            "created_at": row[5],
        }

def get_collections(conn) -> List[Dict[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT slug, title FROM collections ORDER BY slug")
        return [{"slug": r[0], "title": r[1]} for r in cur.fetchall()]

def parse_collection_scope(utterance: str) -> tuple[str, Optional[str]]:
    """
    Deterministically parse collection scope from utterance.
    Returns (cleaned_utterance, collection_slug).
    
    Removes phrases like "in venona", "from vassiliev", etc. and returns the collection slug.
    """
    import re
    # Known collection slugs
    collections = ["venona", "vassiliev", "silvermaster"]
    
    cleaned = utterance
    found_slug = None
    
    for slug in collections:
        # Match "in venona", "from venona", "venona collection", etc.
        patterns = [
            rf'\bin\s+{slug}\b',
            rf'\bfrom\s+{slug}\b',
            rf'\b{slug}\s+collection\b',
            rf'\b{slug}\s+documents\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                found_slug = slug
                # Remove the matched phrase
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
                break
        if found_slug:
            break
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned, found_slug

# =============================================================================
# Deictics
# =============================================================================

DEICTIC_PATTERNS = [
    r"\bthose\s+(results?|ones?|chunks?|documents?)\b",
    r"\bthat\s+(result|one|chunk|document)\b",
    r"\bearlier\s+(results?|ones?)\b",
    r"\bprevious\s+(results?|ones?)\b",
    r"\babove\s+(results?|ones?)\b",
    r"\bprior\s+(results?|ones?)\b",
    r"\blast\s+(results?|ones?)\b",
    r"\bmost\s+recent\s+(results?|ones?)\b",
]

ALL_RESULTS_PATTERNS = [
    r"\ball\s+(of\s+)?(the\s+)?results?\s+(so\s+far|thus\s+far|up\s+to\s+now)\b",
    r"\ball\s+(of\s+)?(the\s+)?result\s+sets?\s+(so\s+far|thus\s+far|up\s+to\s+now)\b",
    r"\bevery\s+(result|result\s+set)\s+(so\s+far|thus\s+far|up\s+to\s+now)\b",
    r"\ball\s+(of\s+)?(the\s+)?results?\b",  # fallback for "all results"
]

def detect_deictics(utterance: str) -> Dict[str, bool]:
    utterance_lower = utterance.lower()
    detected: Dict[str, bool] = {}
    for pattern in DEICTIC_PATTERNS:
        if re.search(pattern, utterance_lower, re.IGNORECASE):
            key = (
                pattern.replace(r"\b", "")
                .replace("\\s+", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("?", "")
                .replace("|", "_or_")
                .strip("_")
            )
            detected[key] = True
    return detected

def detect_all_results(utterance: str) -> bool:
    """Detect if utterance refers to 'all results' / 'all of the results so far'."""
    utterance_lower = utterance.lower()
    for pattern in ALL_RESULTS_PATTERNS:
        if re.search(pattern, utterance_lower, re.IGNORECASE):
            return True
    return False

def resolve_deictics_to_result_set(detected: Dict[str, bool], recent_result_sets: List[Dict[str, Any]]) -> Optional[int]:
    if not detected or not recent_result_sets:
        return None
    return recent_result_sets[0]["id"]

# =============================================================================
# Prompt
# =============================================================================

def build_llm_prompt(
    utterance: str,
    session_summary: str,
    recent_result_sets: List[Dict[str, Any]],
    collections: List[Dict[str, str]],
    detected_deictics: Dict[str, bool],
    query_lang_version: str = "qir_v1",
) -> str:
    allowed_primitives = [
        "TERM","PHRASE","WITHIN_RESULT_SET","EXCLUDE_RESULT_SET",
        "FILTER_COLLECTION","FILTER_DOCUMENT","FILTER_DATE_RANGE",
        "SET_TOP_K","SET_SEARCH_TYPE","SET_TERM_MODE","OR_GROUP",
        "TOGGLE_CONCORDANCE_EXPANSION",
        "ENTITY","CO_OCCURS_WITH","INTERSECT_DATE_WINDOWS","FILTER_COUNTRY",
        "CO_LOCATED","RELATION_EVIDENCE","REQUIRE_EVIDENCE","GROUP_BY",
    ]

    rs_ctx = "\nRecent result_sets in this session:\n"
    if recent_result_sets:
        for rs in recent_result_sets:
            rs_ctx += f"  - ID {rs['id']}: {rs['label']} (created {rs['created_at']})\n"
    else:
        rs_ctx = "\nNo result_sets in this session yet.\n"

    col_ctx = "\nAvailable collections:\n"
    if collections:
        for col in collections:
            col_ctx += f"  - slug: {col['slug']}, title: {col['title']}\n"
    else:
        col_ctx = "\nNo collections available.\n"

    deictic_warning = ""
    if detected_deictics:
        most_recent = recent_result_sets[0]["id"] if recent_result_sets else "N/A"
        deictic_warning = f"""
⚠️  DEICTIC DETECTED: references like "those results", "prev results", "earlier results", etc.
   MANDATORY: Include WITHIN_RESULT_SET primitive in your plan (result_set_id can be null/omitted).
   FORBIDDEN: Do NOT set needs_clarification=true - deictics are handled automatically by the system.
   The system will inject result_set_id {most_recent} automatically - just include WITHIN_RESULT_SET.
"""

    return f"""Convert the user's natural language query into a structured query plan using primitives.

USER UTTERANCE:
"{utterance}"

SESSION CONTEXT:
{session_summary}
{rs_ctx}
{col_ctx}
{deictic_warning}

QUERY LANGUAGE VERSION: {query_lang_version}

ALLOWED PRIMITIVE TYPES:
{', '.join(allowed_primitives)}

INSTRUCTIONS:
- Output JSON only matching the schema.
- Always include query.raw exactly as provided.
- Convert utterance into query.primitives.
- Use TERM for single words; PHRASE for exact phrases.
- Use FILTER_COLLECTION only if clearly mentioned.
- Add retrieval controls only if clearly implied.
- When deictic references are detected (those/prev/earlier results), you MUST include WITHIN_RESULT_SET primitive (result_set_id can be null/omitted).

HARD CONSTRAINTS:
- If deictic warning is shown above, you MUST include WITHIN_RESULT_SET and MUST NOT set needs_clarification=true.
- Only set needs_clarification=true for genuine ambiguities unrelated to result set references (e.g., ambiguous person names).
- Only use allowed primitive types.
- IDs/slugs must match the provided context.
"""

# =============================================================================
# Structured Outputs schema (STRICT) — FIXED
# =============================================================================

def get_plan_json_schema() -> Dict[str, Any]:
    primitive_type_enum = [
        "TERM","PHRASE","WITHIN_RESULT_SET","EXCLUDE_RESULT_SET",
        "FILTER_COLLECTION","FILTER_DOCUMENT","FILTER_DATE_RANGE",
        "SET_TOP_K","SET_SEARCH_TYPE","SET_TERM_MODE","OR_GROUP",
        "TOGGLE_CONCORDANCE_EXPANSION",
        "ENTITY","CO_OCCURS_WITH","INTERSECT_DATE_WINDOWS","FILTER_COUNTRY",
        "CO_LOCATED","RELATION_EVIDENCE","REQUIRE_EVIDENCE","GROUP_BY"
    ]

    nullable_str = {"type": ["string", "null"]}
    nullable_int = {"type": ["integer", "null"]}
    nullable_bool = {"type": ["boolean", "null"]}
    nullable_value = {"type": ["string", "integer", "null"]}

    # This is the key fix:
    # Any schema with {"type":"object"} MUST include additionalProperties:false (strict requirement)
    empty_object_strict = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    # Leaf primitive schema (no meaningful recursion)
    # Still includes "primitives" field (required), but it must be an array of STRICT empty objects.
    leaf_properties = {
        "type": {"type": "string", "enum": primitive_type_enum},
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
        "primitives": {"type": "array", "items": empty_object_strict},
    }
    leaf_required = list(leaf_properties.keys())

    leaf_primitive_schema = {
        "type": "object",
        "properties": leaf_properties,
        "required": leaf_required,
        "additionalProperties": False,
    }

    # Top-level primitive schema: OR_GROUP.primitives uses leaf primitives.
    prim_properties = dict(leaf_properties)
    prim_properties["primitives"] = {"type": "array", "items": leaf_primitive_schema}
    prim_required = list(prim_properties.keys())

    primitive_schema = {
        "type": "object",
        "properties": prim_properties,
        "required": prim_required,
        "additionalProperties": False,
    }

    query_properties = {
        "raw": {"type": "string"},
        "primitives": {"type": "array", "items": primitive_schema},
    }
    query_required = list(query_properties.keys())

    top_properties = {
        "query": {
            "type": "object",
            "properties": query_properties,
            "required": query_required,
            "additionalProperties": False,
        },
        "needs_clarification": {"type": "boolean"},
        "choices": {"type": "array", "items": {"type": "string"}},
    }
    top_required = list(top_properties.keys())

    return {
        "type": "object",
        "properties": top_properties,
        "required": top_required,
        "additionalProperties": False,
    }

# =============================================================================
# LLM call (Structured Outputs, GPT-5-mini)
# =============================================================================

def call_llm_structured(prompt: str, model: Optional[str]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    if model is None:
        model = os.getenv("OPENAI_MODEL_PLAN", "gpt-5-mini")

    client = OpenAI(api_key=api_key)
    schema_dict = get_plan_json_schema()

    request_params: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only matching the schema. No prose."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "research_plan", "schema": schema_dict, "strict": True},
        },
    }

    resp = client.beta.chat.completions.parse(**request_params)

    if not resp.choices:
        raise RuntimeError("OpenAI returned no choices")

    msg = resp.choices[0].message

    # Prefer SDK-parsed object when available
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            result = parsed.model_dump()
        elif hasattr(parsed, "dict"):
            result = parsed.dict()
        elif isinstance(parsed, dict):
            result = parsed
        else:
            # Fallback: serialize and parse
            # WARNING: json.dumps(parsed, default=str) can corrupt values if parsed contains
            # complex objects. Try to extract dict representation first.
            try:
                # Try to get dict representation without double-serialization
                if hasattr(parsed, "__dict__"):
                    result = dict(parsed.__dict__)
                else:
                    # Last resort: serialize and parse (may corrupt values)
                    result = json.loads(json.dumps(parsed, default=str))
            except Exception:
                # If all else fails, serialize and parse
                result = json.loads(json.dumps(parsed, default=str))
        
        # Clean primitive values immediately after parsing to catch JSON leakage at source
        result = _clean_llm_output(result)
        return result

    # ---- IMPORTANT: SDK sometimes leaves parsed=None even though content is valid JSON ----
    content = msg.content
    if not content:
        raise RuntimeError(f"Structured output missing both parsed and content. Model={model}")

    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model returned non-JSON content despite json_schema response_format. Error: {e}. "
                           f"Content preview: {content[:300]!r}")

    # Minimal sanity checks (OpenAI should already enforce schema server-side)
    if not isinstance(obj, dict) or "query" not in obj:
        raise RuntimeError(f"JSON parsed but missing required fields. Got: {str(obj)[:300]}")

    # Clean primitive values immediately after parsing to catch JSON leakage at source
    obj = _clean_llm_output(obj)
    return obj

def _clean_llm_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean LLM output by removing JSON artifacts from primitive values.
    This is the root cause fix - clean values immediately after parsing.
    """
    if not isinstance(obj, dict):
        return obj
    
    # Clean query.primitives values
    if "query" in obj and isinstance(obj["query"], dict):
        prims = obj["query"].get("primitives", [])
        if isinstance(prims, list):
            cleaned_prims = []
            for p in prims:
                if not isinstance(p, dict):
                    cleaned_prims.append(p)
                    continue
                
                ptype = p.get("type")
                cleaned_p = dict(p)
                
                # Clean TERM and PHRASE values
                if ptype in ("TERM", "PHRASE") and "value" in cleaned_p:
                    value = cleaned_p["value"]
                    if isinstance(value, str):
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
                        if cleaned:
                            cleaned_p["value"] = cleaned
                        else:
                            continue  # Skip this primitive if value is empty
                
                # Recursively clean OR_GROUP primitives
                if ptype == "OR_GROUP" and "primitives" in cleaned_p:
                    nested = cleaned_p["primitives"]
                    if isinstance(nested, list):
                        cleaned_nested = []
                        for np in nested:
                            if isinstance(np, dict) and np.get("type") in ("TERM", "PHRASE") and "value" in np:
                                np_value = np["value"]
                                if isinstance(np_value, str):
                                    # Remove trailing JSON artifacts (more aggressive cleaning)
                                    np_cleaned = np_value
                                    while np_cleaned.endswith(("},", "}]", "}", "],", "]", ",")):
                                        np_cleaned = np_cleaned[:-1].rstrip()
                                    while np_cleaned.startswith(("{", "[", ",")):
                                        np_cleaned = np_cleaned[1:].lstrip()
                                    np_cleaned = np_cleaned.strip()
                                    if np_cleaned:
                                        np["value"] = np_cleaned
                                        cleaned_nested.append(np)
                            else:
                                cleaned_nested.append(np)
                        cleaned_p["primitives"] = cleaned_nested
                
                cleaned_prims.append(cleaned_p)
            
            obj["query"]["primitives"] = cleaned_prims
    
    return obj


# =============================================================================
# Normalize: strip nulls / empty fields into your minimal internal IR
# =============================================================================

PRIMITIVE_FIELDS = [
    "type","value","slug","document_id","result_set_id",
    "entity_id","entity_a","entity_b","start","end","window",
    "scope","field","evidence_type","enabled","source_slug","primitives"
]

def _strip_nulls(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                vv = _strip_nulls(v)
                out[k] = vv
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_strip_nulls(x) for x in obj if x is not None]
    return obj

def normalize_plan_dict(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    plan_dict = _strip_nulls(plan_dict)
    plan_dict.setdefault("needs_clarification", False)
    plan_dict.setdefault("choices", [])
    plan_dict.setdefault("query", {})
    plan_dict["query"].setdefault("raw", "")
    plan_dict["query"].setdefault("primitives", [])

    new_prims: List[Dict[str, Any]] = []
    for p in plan_dict["query"]["primitives"]:
        if not isinstance(p, dict) or "type" not in p:
            continue
        ptype = p.get("type")

        compact = {k: p[k] for k in PRIMITIVE_FIELDS if k in p and p[k] is not None}
        
        # Fix common LLM mistakes: FILTER_COLLECTION uses 'slug', not 'source_slug'
        if ptype == "FILTER_COLLECTION":
            if "source_slug" in compact and "slug" not in compact:
                compact["slug"] = compact.pop("source_slug")
            # Also handle case where LLM uses 'value' instead of 'slug'
            if "value" in compact and "slug" not in compact:
                compact["slug"] = compact.pop("value")
        
        # Clean primitive values: strip JSON artifacts from TERM and PHRASE values
        if ptype in ("TERM", "PHRASE") and "value" in compact:
            value = compact["value"]
            if isinstance(value, str):
                # Remove trailing JSON artifacts (}, {, ], [, commas)
                cleaned = value.rstrip("},]{[, ")
                # Remove leading JSON artifacts
                cleaned = cleaned.lstrip("{[, ")
                # If value was completely consumed by artifacts, skip this primitive
                if not cleaned.strip():
                    continue
                compact["value"] = cleaned
        
        # Strip meaningless primitives field unless OR_GROUP
        if ptype != "OR_GROUP":
            compact.pop("primitives", None)
        else:
            nested = compact.get("primitives", [])
            nested2 = []
            if isinstance(nested, list):
                for np in nested:
                    if isinstance(np, dict) and "type" in np:
                        np2 = {k: np[k] for k in PRIMITIVE_FIELDS if k in np and np[k] is not None}
                        # Fix FILTER_COLLECTION in nested primitives too
                        if np2.get("type") == "FILTER_COLLECTION":
                            if "source_slug" in np2 and "slug" not in np2:
                                np2["slug"] = np2.pop("source_slug")
                            if "value" in np2 and "slug" not in np2:
                                np2["slug"] = np2.pop("value")
                        # Clean nested TERM/PHRASE values too
                        if np2.get("type") in ("TERM", "PHRASE") and "value" in np2:
                            value = np2["value"]
                            if isinstance(value, str):
                                # Remove trailing JSON artifacts (more aggressive cleaning)
                                cleaned = value
                                while cleaned.endswith(("},", "}]", "}", "],", "]", ",")):
                                    cleaned = cleaned[:-1].rstrip()
                                while cleaned.startswith(("{", "[", ",")):
                                    cleaned = cleaned[1:].lstrip()
                                cleaned = cleaned.strip()
                                if cleaned:
                                    np2["value"] = cleaned
                                else:
                                    continue  # Skip this nested primitive if value is empty
                        np2.pop("primitives", None)  # no recursion in internal IR
                        nested2.append(np2)
            compact["primitives"] = nested2

        new_prims.append(compact)

    plan_dict["query"]["primitives"] = new_prims
    return plan_dict

# =============================================================================
# Deterministic deictic injection
# =============================================================================

def inject_within_result_set_if_needed(plan_dict: Dict[str, Any], rs_id: Optional[int]) -> Dict[str, Any]:
    """Inject result_set_id into WITHIN_RESULT_SET primitives when missing."""
    if rs_id is None:
        return plan_dict
    
    query = plan_dict.get("query", {})
    prims = query.get("primitives", [])
    
    # Check if there's an existing WITHIN_RESULT_SET that needs result_set_id filled in
    found_within = False
    for i, p in enumerate(prims):
        if isinstance(p, dict) and p.get("type") == "WITHIN_RESULT_SET":
            found_within = True
            # If result_set_id is missing or invalid, inject it
            result_set_id = p.get("result_set_id")
            if result_set_id is None or (isinstance(result_set_id, int) and result_set_id <= 0):
                prims[i]["result_set_id"] = rs_id
            break
    
    # If no WITHIN_RESULT_SET exists, add one
    if not found_within:
        prims.append({"type": "WITHIN_RESULT_SET", "result_set_id": rs_id})
    
    # Ensure the updated list is set back
    query["primitives"] = prims
    plan_dict["query"] = query
    return plan_dict

def inject_all_result_sets(plan_dict: Dict[str, Any], recent_result_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inject multiple WITHIN_RESULT_SET primitives for 'all results' queries."""
    if not recent_result_sets:
        return plan_dict
    
    query = plan_dict.get("query", {})
    prims = query.get("primitives", [])
    
    # Remove any existing WITHIN_RESULT_SET primitives (we'll replace with all)
    prims = [p for p in prims if not (isinstance(p, dict) and p.get("type") == "WITHIN_RESULT_SET")]
    
    # Add one WITHIN_RESULT_SET per result set
    for rs in recent_result_sets:
        prims.append({"type": "WITHIN_RESULT_SET", "result_set_id": rs["id"]})
    
    query["primitives"] = prims
    plan_dict["query"] = query
    return plan_dict

# =============================================================================
# Validation
# =============================================================================

def validate_and_normalize(plan_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    if "query" not in plan_dict or not isinstance(plan_dict["query"], dict):
        return plan_dict, ["Missing 'query' field in plan"]
    if "primitives" not in plan_dict["query"]:
        return plan_dict, ["Missing 'query.primitives' field in plan"]
    if plan_dict.get("needs_clarification"):
        return plan_dict, ["LLM requested clarification - cannot proceed"]
    try:
        # First validate structure (existing validation)
        errors.extend(validate_plan_json(plan_dict))
        # Then validate primitives (JSON leakage) - envelope not required at this stage
        if not errors:  # Only check primitives if structure is valid
            # Don't require envelope yet - it will be built after this validation
            errors.extend(validate_plan(plan_dict, require_envelope=False))
    except Exception as e:
        errors.append(f"Validation error: {e}")
    return plan_dict, errors

# =============================================================================
# Rendering
# =============================================================================

def render_plan_summary(plan: ResearchPlan, resolved_deictics: Dict[str, Any], session_summary: str) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("QUERY PLAN SUMMARY")
    lines.append("=" * 80)
    lines.append(f"\nSession: {session_summary}")
    lines.append(f"\nRaw Utterance: \"{plan.query.raw}\"")

    if resolved_deictics:
        lines.append("\n🔍 Resolved References:")
        detected = resolved_deictics.get("detected", {})
        resolved_to = resolved_deictics.get("resolved_to") or resolved_deictics.get("resolved_result_set_id")
        if detected:
            lines.append(f"  - detected: {', '.join(sorted(detected.keys()))}")
        if resolved_to is not None:
            lines.append(f"  - resolved_to: result_set_id {resolved_to}")

    lines.append("\n📋 Primitives:")
    for i, p in enumerate(plan.query.primitives, 1):
        if isinstance(p, TermPrimitive):
            lines.append(f"  {i}. TERM: \"{p.value}\"")
        elif isinstance(p, PhrasePrimitive):
            lines.append(f"  {i}. PHRASE: \"{p.value}\"")
        elif isinstance(p, WithinResultSetPrimitive):
            lines.append(f"  {i}. WITHIN_RESULT_SET: result_set_id={p.result_set_id}")
        elif isinstance(p, FilterCollectionPrimitive):
            lines.append(f"  {i}. FILTER_COLLECTION: slug=\"{p.slug}\"")
        elif isinstance(p, FilterDocumentPrimitive):
            lines.append(f"  {i}. FILTER_DOCUMENT: document_id={p.document_id}")
        elif isinstance(p, SetTopKPrimitive):
            lines.append(f"  {i}. SET_TOP_K: {p.value}")
        elif isinstance(p, SetSearchTypePrimitive):
            lines.append(f"  {i}. SET_SEARCH_TYPE: {p.value}")
        else:
            lines.append(f"  {i}. {getattr(p, 'type', 'PRIMITIVE')}: {p}")

    if plan.compiled:
        lines.append("\n⚙️  Compiled Components:")
        tsq = plan.compiled.get("tsquery")
        if isinstance(tsq, dict):
            lines.append(f"  - tsquery_text: {tsq.get('text', 'N/A')}")
            lines.append(f"  - tsquery_sql:  {tsq.get('sql', 'N/A')}")
        exp = plan.compiled.get("expanded")
        if isinstance(exp, dict):
            lines.append(f"  - expanded_text: \"{exp.get('expanded_text', 'N/A')}\"")
        scope = plan.compiled.get("scope")
        if isinstance(scope, dict):
            where_sql = scope.get("where_sql", "")
            lines.append(f"  - scope SQL: {where_sql[:80]}{'...' if len(where_sql) > 80 else ''}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)

# =============================================================================
# Persistence
# =============================================================================

def build_execution_envelope(
    plan: ResearchPlan,
    recent_run: Optional[Dict[str, Any]] = None,
    default_chunk_pv: str = "chunk_v1_full",
    default_k: int = 20,
    explicit_collection_scope: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build execution envelope from plan primitives and inherited defaults.
    Records inheritance provenance if defaults came from a prior run.
    
    Args:
        explicit_collection_scope: Collection slug parsed deterministically from utterance (e.g., "venona")
    """
    envelope: Dict[str, Any] = {}
    inherited_fields: List[str] = []
    inherited_from_retrieval_run_id: Optional[int] = None
    
    # Extract collection scope from FILTER_COLLECTION primitives OR explicit scope
    collection_slugs = []
    for p in plan.query.primitives:
        if isinstance(p, FilterCollectionPrimitive):
            collection_slugs.append(p.slug)
    
    # Explicit scope from deterministic parsing takes precedence
    if explicit_collection_scope:
        envelope["collection_scope"] = [explicit_collection_scope]
    elif collection_slugs:
        envelope["collection_scope"] = collection_slugs
    else:
        # No explicit collection filter - use "ALL" or inherit
        if recent_run:
            # Try to infer from recent run (if it had collection filters)
            # For now, default to "ALL" if no explicit filter
            envelope["collection_scope"] = "ALL"
        else:
            envelope["collection_scope"] = "ALL"
    
    # Extract k from SET_TOP_K primitive
    k = default_k
    for p in plan.query.primitives:
        if isinstance(p, SetTopKPrimitive):
            k = p.value
            break
    
    # If k wasn't in primitives, inherit from recent run
    if k == default_k and recent_run and recent_run.get("top_k"):
        k = recent_run["top_k"]
        inherited_fields.append("k")
        if inherited_from_retrieval_run_id is None:
            inherited_from_retrieval_run_id = recent_run["id"]
    
    envelope["k"] = k
    
    # Extract chunk_pipeline_version (default or inherit)
    chunk_pv = default_chunk_pv
    if recent_run and recent_run.get("chunk_pv"):
        chunk_pv = recent_run["chunk_pv"]
        inherited_fields.append("chunk_pipeline_version")
        if inherited_from_retrieval_run_id is None:
            inherited_from_retrieval_run_id = recent_run["id"]
    
    envelope["chunk_pipeline_version"] = chunk_pv
    
    # Extract retrieval_config from SET_SEARCH_TYPE and other primitives
    search_type = "hybrid"  # default
    for p in plan.query.primitives:
        if isinstance(p, SetSearchTypePrimitive):
            search_type = p.value
            break
    
    # Build retrieval_config
    retrieval_config: Dict[str, Any] = {
        "search_type": search_type,
    }
    
    # Inherit retrieval_config_json from recent run if available
    if recent_run and recent_run.get("retrieval_config_json"):
        inherited_config = recent_run["retrieval_config_json"]
        if isinstance(inherited_config, dict):
            # Merge inherited config, but let primitives override
            retrieval_config.update(inherited_config)
            retrieval_config["search_type"] = search_type  # Ensure search_type from primitives takes precedence
            inherited_fields.append("retrieval_config")
            if inherited_from_retrieval_run_id is None:
                inherited_from_retrieval_run_id = recent_run["id"]
    
    envelope["retrieval_config"] = retrieval_config
    
    # Record inheritance provenance
    if inherited_from_retrieval_run_id:
        envelope["inherited_from_retrieval_run_id"] = inherited_from_retrieval_run_id
        envelope["inherited_fields"] = inherited_fields
    
    return envelope

def save_plan(
    conn,
    session_id: int,
    plan: ResearchPlan,
    user_utterance: str,
    query_lang_version: str = "qir_v1",
) -> int:
    plan_dict = plan.to_dict()
    plan_hash = compute_plan_hash(plan_dict)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_plans (
                session_id,
                plan_json,
                plan_hash,
                query_lang_version,
                retrieval_impl_version,
                status,
                user_utterance
            )
            VALUES (%s, %s, %s, %s, %s, 'proposed', %s)
            RETURNING id
            """,
            (session_id, Json(plan_dict), plan_hash, query_lang_version, "retrieval_v1", user_utterance),
        )
        plan_id = cur.fetchone()[0]
        conn.commit()
        return plan_id

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Convert NL query to structured plan using LLM")
    ap.add_argument("--session", type=str, required=True, help="Session ID or label")
    ap.add_argument("--text", type=str, required=True, help="Natural language query")
    ap.add_argument("--model", type=str, default=None, help="OpenAI model (default: OPENAI_MODEL_PLAN or gpt-5-mini)")
    ap.add_argument("--query-lang-version", type=str, default="qir_v1", help="Query language version")
    ap.add_argument("--dry-run", action="store_true", help="Don't save to database, just show plan")
    args = ap.parse_args()

    conn = get_conn()
    try:
        session_id = resolve_session_id(conn, args.session)
        print(f"Session: {args.session} (ID: {session_id})", file=sys.stderr)

        session_summary = get_session_summary(conn, session_id)
        recent_result_sets = get_recent_result_sets(conn, session_id)
        recent_run = get_most_recent_retrieval_run(conn, session_id)
        collections = get_collections(conn)

        # Deterministically parse collection scope BEFORE LLM call
        cleaned_utterance, explicit_collection_scope = parse_collection_scope(args.text)
        
        detected_deictics = detect_deictics(cleaned_utterance)
        wants_all_results = detect_all_results(cleaned_utterance)
        resolved_rs_id = resolve_deictics_to_result_set(detected_deictics, recent_result_sets)

        resolved_deictics_meta: Dict[str, Any] = {}
        if detected_deictics:
            resolved_deictics_meta["detected"] = detected_deictics
            if resolved_rs_id is not None:
                resolved_deictics_meta["resolved_to"] = resolved_rs_id
                resolved_deictics_meta["resolved_result_set_id"] = resolved_rs_id

        prompt = build_llm_prompt(
            args.text,
            session_summary,
            recent_result_sets,
            collections,
            detected_deictics,
            args.query_lang_version,
        )

        print("Calling LLM to generate plan (Structured Outputs)...", file=sys.stderr)
        plan_dict_raw = call_llm_structured(prompt, model=args.model)
        
        # Debug: log raw LLM output to catch JSON leakage at source
        if os.getenv("DEBUG_LLM_OUTPUT"):
            print(f"DEBUG: Raw LLM output: {json.dumps(plan_dict_raw, indent=2)[:1000]}", file=sys.stderr)
        
        plan_dict = plan_dict_raw

        # enforce raw utterance (use original, not cleaned)
        if "query" not in plan_dict or not isinstance(plan_dict["query"], dict):
            raise RuntimeError("Structured Outputs returned invalid plan: missing query object")
        plan_dict["query"]["raw"] = args.text  # Store original utterance

        # normalize away schema-inflated nulls
        plan_dict = normalize_plan_dict(plan_dict)

        # deterministic injection: fill in missing result_set_id for WITHIN_RESULT_SET primitives
        prims = plan_dict.get("query", {}).get("primitives", [])
        has_within_without_id = any(
            isinstance(p, dict) and p.get("type") == "WITHIN_RESULT_SET" 
            and (p.get("result_set_id") is None or (isinstance(p.get("result_set_id"), int) and p.get("result_set_id") <= 0))
            for p in prims
        )
        
        if wants_all_results and recent_result_sets:
            # "All results" - inject multiple WITHIN_RESULT_SET primitives (one per result set)
            plan_dict = inject_all_result_sets(plan_dict, recent_result_sets)
        elif has_within_without_id and recent_result_sets:
            # Single WITHIN_RESULT_SET without ID - inject most recent
            injection_rs_id = resolved_rs_id if resolved_rs_id is not None else recent_result_sets[0]["id"]
            plan_dict = inject_within_result_set_if_needed(plan_dict, injection_rs_id)
        elif resolved_rs_id is not None:
            # Deictics detected but no WITHIN_RESULT_SET yet - add it
            plan_dict = inject_within_result_set_if_needed(plan_dict, resolved_rs_id)
        
        # Inject FILTER_COLLECTION primitive if we have explicit_collection_scope but LLM didn't generate it
        # This ensures primitives remain the source of truth (primitives → envelope, not the reverse)
        if explicit_collection_scope:
            prims = plan_dict.get("query", {}).get("primitives", [])
            has_collection_filter = any(
                isinstance(p, dict) and 
                p.get("type") == "FILTER_COLLECTION" and 
                p.get("slug") == explicit_collection_scope
                for p in prims
            )
            if not has_collection_filter:
                # Inject FILTER_COLLECTION primitive to match explicit scope
                prims.append({"type": "FILTER_COLLECTION", "slug": explicit_collection_scope})
                plan_dict.setdefault("query", {})["primitives"] = prims

        # Initial validation (envelope not required yet - it will be built next)
        plan_dict, errors = validate_and_normalize(plan_dict)
        if errors:
            print("ERROR: Plan validation failed:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            print(json.dumps(plan_dict, indent=2)[:1200], file=sys.stderr)
            sys.exit(1)

        # parse
        plan = ResearchPlan.from_dict(plan_dict)
        plan.query.raw = args.text

        # Build execution envelope (inherits from recent run if available)
        execution_envelope = build_execution_envelope(
            plan,
            recent_run=recent_run,
            default_chunk_pv="chunk_v1_full",
            default_k=20,
            explicit_collection_scope=explicit_collection_scope,  # Pass deterministically parsed scope
        )
        plan.execution_envelope = execution_envelope
        
        # Final validation with envelope (check invariants)
        plan_dict_with_envelope = plan.to_dict()
        final_errors = validate_plan(plan_dict_with_envelope, require_envelope=True)
        if final_errors:
            print("ERROR: Final plan validation failed (after envelope build):", file=sys.stderr)
            for err in final_errors:
                print(f"  - {err}", file=sys.stderr)
            print(json.dumps(plan_dict_with_envelope, indent=2)[:1200], file=sys.stderr)
            sys.exit(1)

        # compile
        plan.compile()

        # render
        print(render_plan_summary(plan, resolved_deictics_meta, session_summary))

        # persist
        if not args.dry_run:
            plan_id = save_plan(conn, session_id, plan, args.text, args.query_lang_version)
            print(f"\n✅ Plan saved with ID: {plan_id} (session={session_id}, status: proposed)", file=sys.stderr)
        else:
            print("\n[DRY RUN] Plan not saved to database", file=sys.stderr)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
