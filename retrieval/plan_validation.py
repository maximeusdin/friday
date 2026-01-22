#!/usr/bin/env python3
"""
Strict validation for research plans to catch JSON leakage and invalid values.
Also validates plan invariants (envelope-primitive consistency, etc.).
"""

import re
from typing import List, Dict, Any, Optional

# Patterns that indicate JSON leakage
BAD_CHARS = re.compile(r'[\{\}\[\]]')
BAD_JSONISH = re.compile(r'("op"\s*:|ADD_PRIMITIVE|REMOVE_PRIMITIVE|MODIFY_PRIMITIVE|REPLACE_PRIMITIVE|PATCH OPERATIONS|execution_envelope|plan_json)', re.I)
CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f-\x9f]')

def validate_primitive_value(p_type: str, value: Any) -> List[str]:
    """
    Validate a primitive value for JSON leakage and invalid content.
    Returns list of error messages (empty if valid).
    """
    errors: List[str] = []
    
    if not isinstance(value, str):
        return errors  # Non-string values (ints, etc.) are fine
    
    v = value.strip()
    
    # Length bounds
    if len(v) == 0:
        errors.append(f"{p_type} value is empty")
    elif len(v) > 120:
        errors.append(f"{p_type} value too long ({len(v)} chars, max 120): {v[:50]}...")
    
    # Check for JSON leakage patterns
    if BAD_CHARS.search(v):
        errors.append(f"{p_type} value contains JSON characters ({{}}[]): {v!r}")
    
    if BAD_JSONISH.search(v):
        errors.append(f"{p_type} value contains JSON-like patterns: {v!r}")
    
    if CONTROL_CHARS.search(v):
        errors.append(f"{p_type} value contains control characters: {v!r}")
    
    # Check for common JSON artifacts
    if v.endswith('}') or v.endswith(']') or v.endswith('},') or v.endswith('],'):
        errors.append(f"{p_type} value ends with JSON artifact: {v!r}")
    
    if v.startswith('{') or v.startswith('['):
        errors.append(f"{p_type} value starts with JSON bracket: {v!r}")
    
    return errors

def validate_primitives(primitives: List[Dict[str, Any]]) -> List[str]:
    """
    Validate all primitives for JSON leakage and invalid values.
    Returns list of error messages (empty if all valid).
    """
    errors: List[str] = []
    
    for i, p in enumerate(primitives):
        if not isinstance(p, dict):
            errors.append(f"Primitive {i} is not a dict: {type(p)}")
            continue
        
        p_type = p.get("type")
        if not p_type:
            errors.append(f"Primitive {i} missing 'type' field")
            continue
        
        # Validate TERM and PHRASE values
        if p_type in ("TERM", "PHRASE"):
            value = p.get("value")
            if value is not None:
                value_errors = validate_primitive_value(p_type, value)
                for err in value_errors:
                    errors.append(f"Primitive {i} ({p_type}): {err}")
        
        # Recursively validate OR_GROUP primitives
        if p_type == "OR_GROUP":
            nested = p.get("primitives", [])
            if isinstance(nested, list):
                nested_errors = validate_primitives(nested)
                for err in nested_errors:
                    errors.append(f"Primitive {i} (OR_GROUP) nested: {err}")
    
    return errors

def derive_collection_scope_from_primitives(primitives: List[Dict[str, Any]]) -> Optional[List[str]]:
    """
    Derive collection_scope from FILTER_COLLECTION primitives.
    Returns list of collection slugs, or None if no filters present.
    """
    collection_slugs = []
    for p in primitives:
        if isinstance(p, dict) and p.get("type") == "FILTER_COLLECTION":
            slug = p.get("slug")
            if slug:
                collection_slugs.append(slug)
    return collection_slugs if collection_slugs else None

def validate_plan_invariants(plan_json: Dict[str, Any], require_envelope: bool = True) -> List[str]:
    """
    Validate plan invariants:
    - Envelope must be derivable from primitives (Invariant A)
    - No contradictory filters
    - All required envelope fields present
    
    Args:
        require_envelope: If True, envelope must exist. If False, only validate if envelope exists.
    
    Returns list of error messages (empty if valid).
    """
    errors: List[str] = []
    
    query = plan_json.get("query", {})
    primitives = query.get("primitives", [])
    envelope = plan_json.get("execution_envelope", {})
    
    # Check envelope presence
    if not envelope:
        if require_envelope:
            errors.append("Plan missing execution_envelope")
            return errors
        else:
            # Envelope not required yet - skip invariant checks
            return errors
    
    # Invariant A: Envelope collection_scope must match primitives
    envelope_scope = envelope.get("collection_scope")
    derived_scope = derive_collection_scope_from_primitives(primitives)
    
    # Normalize "ALL" vs None vs []
    if envelope_scope == "ALL" or envelope_scope is None:
        envelope_scope_normalized = None
    elif isinstance(envelope_scope, list):
        envelope_scope_normalized = sorted(envelope_scope) if envelope_scope else None
    else:
        envelope_scope_normalized = [envelope_scope] if envelope_scope else None
    
    if derived_scope is not None:
        derived_scope_normalized = sorted(derived_scope)
        if envelope_scope_normalized != derived_scope_normalized:
            errors.append(
                f"Envelope-primitive mismatch: envelope.collection_scope={envelope_scope_normalized}, "
                f"but FILTER_COLLECTION primitives yield {derived_scope_normalized}"
            )
    elif envelope_scope_normalized is not None and envelope_scope != "ALL":
        # Primitives don't specify collection, but envelope does (except "ALL" is OK)
        errors.append(
            f"Envelope has collection_scope={envelope_scope} but no FILTER_COLLECTION primitives found"
        )
    
    # Check for contradictory filters (e.g., include venona + exclude venona)
    collection_filters = [p for p in primitives if isinstance(p, dict) and p.get("type") == "FILTER_COLLECTION"]
    collection_slugs = [p.get("slug") for p in collection_filters if p.get("slug")]
    if len(collection_slugs) != len(set(collection_slugs)):
        errors.append(f"Duplicate FILTER_COLLECTION primitives: {collection_slugs}")
    
    # Check required envelope fields
    required_fields = ["collection_scope", "chunk_pipeline_version", "retrieval_config", "k"]
    for field in required_fields:
        if field not in envelope:
            errors.append(f"execution_envelope missing required field: {field}")
    
    return errors

def validate_plan(plan_json: Dict[str, Any], require_envelope: bool = True) -> List[str]:
    """
    Comprehensive plan validation:
    - Primitive value validation (JSON leakage, length, etc.)
    - Plan structure validation
    - Plan invariants (envelope-primitive consistency)
    
    Args:
        require_envelope: If True, envelope must exist. If False, only validate if envelope exists.
                         Use False during initial validation (before envelope is built),
                         True during final validation (before save/execute).
    
    Returns list of error messages (empty if valid).
    """
    errors: List[str] = []
    
    # Validate primitives for JSON leakage
    query = plan_json.get("query", {})
    primitives = query.get("primitives", [])
    if isinstance(primitives, list):
        errors.extend(validate_primitives(primitives))
    
    # Validate plan invariants (only if envelope exists, or if required)
    errors.extend(validate_plan_invariants(plan_json, require_envelope=require_envelope))
    
    return errors
