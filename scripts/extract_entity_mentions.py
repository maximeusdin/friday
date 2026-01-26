#!/usr/bin/env python3
"""
extract_entity_mentions.py [options]

Extract entity mentions from chunks using exact alias matching.

Scans chunk text for exact matches against entity_aliases.alias_norm and
persists high-precision matches to entity_mentions.

Usage:
    # Dry run to see counts
    python scripts/extract_entity_mentions.py --collection venona --dry-run
    
    # Extract for a collection
    python scripts/extract_entity_mentions.py --collection venona
    
    # Extract for specific document
    python scripts/extract_entity_mentions.py --document-id 123
    
    # Extract for chunk range (testing)
    python scripts/extract_entity_mentions.py --chunk-id-range 1:1000 --limit 100
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, DefaultDict
from collections import defaultdict
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import execute_values

from retrieval.entity_resolver import normalize_alias
from retrieval.ops import get_conn

# ============================================================================
# Configuration constants (policy-driven thresholds)
# ============================================================================

# Collision tiering thresholds
COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES = 5  # Single token with >5 candidates = harmless

# Per-alias-class candidate count thresholds for enqueuing
COLLISION_ADJUDICABLE_MAX_CANDIDATES = {
    'covername': 5,  # Covernames might warrant review at ≤5
    'person_full': 2,  # Person full names more restrictive
    'org': 3,  # Default
    'place': 3,  # Default
    'person_given': 1,  # Very restrictive (usually harmless anyway)
    'role_title': 3,  # Default
    'generic_word': 0,  # Never enqueue
    None: 3,  # Default for unclassified
}

# Context gate behavior
CONTEXT_GATE_FAILURE_ENQUEUE = False  # If True, enqueue context gate failures; if False, just log

# Stopwords/function words that should not be matched unless explicitly covername
STOPWORDS = {
    'to', 'is', 'as', 'of', 'in', 'and', 'or', 'but', 'the', 'a', 'an', 'for', 'on', 'at', 'by', 'from', 'with',
    'about', 'into', 'through', 'during', 'including', 'against', 'among', 'throughout', 'despite', 'towards',
    'upon', 'concerning', 'up', 'over', 'under', 'above', 'below', 'between', 'within', 'without', 'across',
    'after', 'before', 'behind', 'beyond', 'near', 'around', 'along', 'beside', 'besides', 'except', 'plus',
    'minus', 'per', 'via', 'versus', 'vs', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'can', 'shall', 'ought', 'need', 'dare', 'used'
}

# Single letters that should not be matched
SINGLE_LETTERS = set('abcdefghijklmnopqrstuvwxyz')


@dataclass
class AliasInfo:
    """Information about an alias for matching"""
    entity_id: int
    original_alias: str
    alias_norm: str
    entity_type: str
    is_auto_match: bool
    min_chars: int
    match_case: str  # 'any', 'case_sensitive', 'upper_only', 'titlecase_only'
    match_mode: str  # 'token', 'substring', 'phrase'
    is_numeric_entity: bool
    alias_class: Optional[str]  # 'covername', 'person_given', 'person_full', etc.
    allow_ambiguous_person_token: bool
    requires_context: Optional[str]


def parse_chunk_id_range(range_str: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse --chunk-id-range format: 'start:end' or 'start:' or ':end'"""
    if ':' not in range_str:
        raise ValueError(f"Invalid chunk-id-range format: {range_str}. Use 'start:end'")
    
    parts = range_str.split(':', 1)
    start_str = parts[0].strip() if parts[0].strip() else None
    end_str = parts[1].strip() if parts[1].strip() else None
    
    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None
    
    if start is not None and end is not None and start > end:
        raise ValueError(f"Invalid chunk-id-range: start ({start}) > end ({end})")
    
    return (start, end)


def get_chunks_query(
    conn,
    *,
    collection_slug: Optional[str] = None,
    document_id: Optional[int] = None,
    chunk_id_start: Optional[int] = None,
    chunk_id_end: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Tuple[int, str, Optional[int]]]:
    """
    Query chunks with filters. Returns list of (chunk_id, text, document_id).
    
    document_id comes from chunk_metadata (denormalized).
    """
    conditions = []
    params = []
    
    if collection_slug:
        conditions.append("cm.collection_slug = %s")
        params.append(collection_slug)
    
    if document_id:
        conditions.append("cm.document_id = %s")
        params.append(document_id)
    
    if chunk_id_start is not None:
        conditions.append("c.id >= %s")
        params.append(chunk_id_start)
    
    if chunk_id_end is not None:
        conditions.append("c.id <= %s")
        params.append(chunk_id_end)
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
        SELECT 
            c.id AS chunk_id,
            c.text,
            cm.document_id
        FROM chunks c
        LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE {where_clause}
        ORDER BY c.id
        {limit_clause}
    """
    
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def tokenize_text(text: str) -> List[Tuple[int, int, str]]:
    """
    Tokenize text preserving positions, handling dotted acronyms, hyphens, and slashes.
    Returns list of (start_pos, end_pos, token) tuples.
    
    Tokenization must align with how normalize_alias works:
    - normalize_alias removes ALL punctuation: dots, hyphens, slashes, etc.
    - So "U.S." → "us", "pre-war" → "prewar", "CIA/MI6" → "ciami6"
    
    Strategy: Tokenize to capture these as single tokens, then normalize_alias will
    remove the punctuation consistently.
    
    Patterns:
    - Dotted acronyms: "U.S.", "F.B.I." → keep as one token
    - Hyphenated words: "pre-war", "Anglo-American" → keep as one token
    - Slash forms: "CIA/MI6" → keep as one token (though dates like "3/4/45" will also match)
    - Apostrophes: "People's" → keep as one token
    """
    tokens = []
    
    # Pattern to match tokens that may contain internal punctuation:
    # - Word characters (letters, digits, underscore)
    # - Dots (for acronyms like U.S.)
    # - Hyphens (for hyphenated words like pre-war)
    # - Slashes (for forms like CIA/MI6)
    # - Apostrophes (for possessives like People's)
    #
    # Match: word chars, optionally followed by punctuation+word chars
    # This captures: "U.S.", "pre-war", "CIA/MI6", "People's", "F.B.I.", "Anglo-American"
    # Then normalize_alias removes punctuation: "us", "prewar", "ciami6", "peoples", "fbi", "angloamerican"
    #
    # Pattern breakdown:
    # \b - word boundary
    # [\w']+ - one or more word chars or apostrophes (handles "People's")
    # (?:[./-][\w']+)* - zero or more of: punctuation (dot/hyphen/slash) followed by word chars
    # \b - word boundary
    pattern = r'\b[\w\']+(?:[./-][\w\']+)*\b'
    
    for match in re.finditer(pattern, text):
        token = match.group(0)
        # Skip if token is only punctuation (edge case)
        if not re.search(r'[\w]', token):
            continue
        tokens.append((match.start(), match.end(), token))
    
    return tokens


def generate_ngrams(tokens: List[Tuple[int, int, str]], max_n: int = 5) -> List[Tuple[int, int, str, int]]:
    """
    Generate n-grams from tokens (1-gram through max_n-gram).
    Returns list of (start_pos, end_pos, ngram_text, n) tuples.
    ngram_text is normalized (space-separated tokens, normalized via normalize_alias).
    
    Each token is normalized individually using normalize_alias, which:
    - Removes punctuation (dots, hyphens, slashes, etc.)
    - Lowercases
    - Collapses whitespace
    
    This ensures alignment: "U.S." → normalize_alias → "us", "pre-war" → "prewar"
    So if an alias was stored as "U.S." (normalized to "us"), we'll match it correctly.
    """
    ngrams = []
    for n in range(1, min(max_n + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            token_slice = tokens[i:i + n]
            start_pos = token_slice[0][0]
            end_pos = token_slice[-1][1]
            # Normalize each token individually (matches alias normalization)
            # normalize_alias removes all punctuation, so "U.S." → "us", "pre-war" → "prewar"
            ngram_tokens = [normalize_alias(t[2]) for t in token_slice]
            ngram_text = ' '.join(ngram_tokens)
            ngrams.append((start_pos, end_pos, ngram_text, n))
    return ngrams


def check_case_match(surface: str, alias_info: AliasInfo) -> bool:
    """Check if surface text matches case requirements."""
    if alias_info.match_case == 'any':
        return True
    elif alias_info.match_case == 'case_sensitive':
        return surface == alias_info.original_alias
    elif alias_info.match_case == 'upper_only':
        # All letters must be uppercase (allow punctuation)
        return surface.isupper() or all(c.isupper() or not c.isalpha() for c in surface)
    elif alias_info.match_case == 'titlecase_only':
        # First letter uppercase, rest lowercase (allow punctuation)
        if not surface:
            return False
        first_letter = surface[0]
        rest = surface[1:]
        return first_letter.isupper() and (not rest or rest.islower() or all(not c.isalpha() or c.islower() for c in rest))
    else:
        return True  # Unknown case policy, allow


def is_roman_numeral(token: str) -> bool:
    """Check if token looks like a roman numeral (II, III, IV, etc.)."""
    if not token:
        return False
    token_upper = token.upper()
    # Simple check: all characters are valid roman numeral letters
    valid_roman_chars = {'I', 'V', 'X', 'L', 'C', 'D', 'M'}
    return all(c in valid_roman_chars for c in token_upper) and len(token_upper) >= 2


def is_candidate_eligible_for_matching(alias_info: AliasInfo, alias_norm: str, surface: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check if a single candidate is eligible for matching (per-candidate eligibility).
    
    Returns (is_eligible, reason_if_not).
    
    This is used to filter candidates before dominance resolution.
    """
    tokens = alias_norm.split()
    is_single_token = len(tokens) == 1
    
    if not is_single_token:
        # Multi-token phrases are eligible
        return True, None
    
    token = tokens[0].lower()
    
    # Single letter - not eligible unless covername
    if len(token) == 1 and token in SINGLE_LETTERS:
        if alias_info.alias_class != 'covername':
            return False, "single_letter"
    
    # Stopword/function word - fast-path for uppercase surface + covername
    if token in STOPWORDS:
        if alias_info.alias_class == 'covername' and alias_info.match_case == 'upper_only':
            # Fast-path: if surface is uppercase, eligible
            if surface is not None:
                is_uppercase = surface.isupper() or all(c.isupper() or not c.isalpha() for c in surface)
                if is_uppercase:
                    return True, None
            return False, "stopword_covername_not_uppercase"
        else:
            # Not a covername, or not upper_only
            return False, "stopword"
    
    # Purely numeric tokens - not eligible unless numeric_entity or covername
    if token.isdigit() and len(token) <= 2:
        if not alias_info.is_numeric_entity and alias_info.alias_class != 'covername':
            return False, "small_integer"
    
    # Roman numerals - eligible only if covername or numeric_entity
    if is_roman_numeral(token):
        if alias_info.alias_class != 'covername' and not alias_info.is_numeric_entity:
            return False, "roman_numeral_alone"
    
    return True, None


def is_alias_eligible_for_matching(alias_norm: str, alias_infos: List[AliasInfo], surface: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check if alias_norm is eligible for matching (at least one candidate is eligible).
    
    Returns (is_eligible, reason_if_not).
    
    Changed to per-candidate eligibility: eligible if ANY candidate is eligible for this surface.
    This allows dominance to pick the covername when surface is uppercase, even if other candidates are stopwords.
    """
    # Check if any candidate is eligible
    for ai in alias_infos:
        is_eligible, reason = is_candidate_eligible_for_matching(ai, alias_norm, surface)
        if is_eligible:
            return True, None
    
    # No candidate is eligible
    # Return reason from first candidate (for logging)
    if alias_infos:
        _, reason = is_candidate_eligible_for_matching(alias_infos[0], alias_norm, surface)
        return False, reason
    return False, "no_candidates"


def find_dominant_candidate(
    alias_norm: str, 
    alias_infos: List[AliasInfo], 
    surface: str,
    preferred_entity_id_map: Optional[Dict[Tuple[Optional[str], str], int]] = None,
    scope: Optional[str] = None,  # Collection slug or other scope identifier
) -> Tuple[Optional[AliasInfo], Optional[str], Optional[str]]:
    """
    Find dominant candidate for auto-resolution of collisions.
    
    Returns (AliasInfo if exactly one candidate should be auto-selected, None otherwise, rule_used, rule_detail).
    rule_used: 'rule0' (preferred_entity_id), 'rule1', 'rule2', 'rule3', or None
    rule_detail: Additional context (e.g., which dominance pair fired, candidate counts)
    
    Rules (in order):
    0. If preferred_entity_id_map specifies a preferred entity for this alias_norm (with scope) → select it
    1. If exactly one candidate is eligible + auto-matchable given surface → select it
    2. If exactly one candidate matches stronger policy (e.g., upper_only satisfied, others not) → select it
    3. If safe dominance pairs resolve uniquely (e.g., place vs generic_word) → select it
    4. Otherwise → None (requires adjudication)
    """
    if len(alias_infos) <= 1:
        return (alias_infos[0] if alias_infos else None, None, None)
    
    # Rule 0: Preferred entity_id (if configured, with scope support)
    if preferred_entity_id_map:
        # Try scoped first (scope, alias_norm), then global (None, alias_norm)
        scope_key = (scope, alias_norm)
        global_key = (None, alias_norm)
        
        preferred_id = None
        if scope_key in preferred_entity_id_map:
            preferred_id = preferred_entity_id_map[scope_key]
        elif global_key in preferred_entity_id_map:
            preferred_id = preferred_entity_id_map[global_key]
        
        if preferred_id is not None:
            preferred_candidates = [ai for ai in alias_infos if ai.entity_id == preferred_id]
            if len(preferred_candidates) == 1:
                return (preferred_candidates[0], 'rule0', f"preferred_entity_id={preferred_id}")
    
    # Rule 1: Exactly one candidate is eligible + auto-matchable given surface
    # Rule 1b: For non-covernames (place/org/person_full), ignore case unless explicitly strict
    # Rule 1c: Only count candidates where is_auto_match is explicitly true (not defaulted)
    
    # Filter candidates by is_auto_match and case (with lenient case for non-covernames)
    eligible_auto_match_candidates = []
    for ai in alias_infos:
        if not ai.is_auto_match:
            continue
        
        # Case matching: strict for covername/person_given, lenient for place/org/person_full
        if ai.alias_class in ('covername', 'person_given'):
            # Require case match for covernames and person given names
            if not check_case_match(surface, ai):
                continue
        else:
            # For place/org/person_full: ignore case unless explicitly strict
            if ai.match_case in ('case_sensitive', 'upper_only', 'titlecase_only'):
                # Explicitly strict case requirement - enforce it
                if not check_case_match(surface, ai):
                    continue
            # Otherwise: match_case='any' or unset, allow through
        
        eligible_auto_match_candidates.append(ai)
    
    if len(eligible_auto_match_candidates) == 1:
        # Verify all others are either not auto_match or fail case/policy
        others_auto_match = [ai for ai in alias_infos if ai.is_auto_match and ai not in eligible_auto_match_candidates]
        if len(others_auto_match) == 0:
            return (eligible_auto_match_candidates[0], 'rule1', f"unique_auto_match (total_candidates={len(alias_infos)})")
    
    # Rule 1c: Unique is_auto_match=true (even if case doesn't match for non-covernames)
    # This catches cases where only one candidate has is_auto_match=true
    auto_match_only = [ai for ai in alias_infos if ai.is_auto_match]
    if len(auto_match_only) == 1:
        # For non-covernames, allow even if case doesn't match
        ai = auto_match_only[0]
        if ai.alias_class not in ('covername', 'person_given'):
            return (ai, 'rule1c', f"unique_auto_match_ignore_case (total_candidates={len(alias_infos)})")
    
    # Rule 2: Exactly one matches stronger policy (case matching)
    # RESTRICTED: Only for covernames, or when match_case is explicitly constraining (not 'any')
    case_match_candidates = []
    for ai in alias_infos:
        # Only consider if match_case is actually constraining
        if ai.match_case == 'any':
            continue  # Skip 'any' - not constraining
        if check_case_match(surface, ai):
            case_match_candidates.append(ai)
    
    if len(case_match_candidates) == 1:
        # Only use Rule 2 for covernames or when match_case is explicitly set
        ai = case_match_candidates[0]
        if ai.alias_class == 'covername' or ai.match_case != 'any':
            return (ai, 'rule2', f"unique_case_match_constrained (total_candidates={len(alias_infos)})")
    
    # Rule 3: Safe dominance pairs (not blanket ordering)
    # Only resolve when one candidate clearly dominates another in a safe way
    # Avoid ambiguous cases like "place vs org" (e.g., "Washington", "Moscow")
    safe_dominance_pairs = [
        ('place', 'generic_word'),
        ('org', 'generic_word'),
        ('person_full', 'generic_word'),
        ('covername', 'generic_word'),
        ('place', 'person_given'),  # Place names vs given names (usually safe)
        ('org', 'person_given'),
    ]
    
    for dominant_class, subordinate_class in safe_dominance_pairs:
        dominant_candidates = [ai for ai in alias_infos if ai.alias_class == dominant_class]
        subordinate_candidates = [ai for ai in alias_infos if ai.alias_class == subordinate_class]
        
        # If exactly one dominant and at least one subordinate, and no other classes
        if len(dominant_candidates) == 1 and len(subordinate_candidates) >= 1:
            other_classes = [ai for ai in alias_infos 
                            if ai.alias_class not in (dominant_class, subordinate_class) 
                            and ai.alias_class is not None]
            if len(other_classes) == 0:
                pair_label = f"{dominant_class}>{subordinate_class}"
                detail = f"{pair_label} (dominant=1, subordinate={len(subordinate_candidates)}, total={len(alias_infos)})"
                return (dominant_candidates[0], 'rule3', detail)
    
    # No dominant candidate found
    return (None, None, None)


def is_collision_high_value(alias_norm: str, alias_infos: List[AliasInfo], surface: str) -> Tuple[bool, bool, Optional[str]]:
    """
    Determine if a collision is "high-value" and worth enqueuing for review.
    
    Returns (is_high_value, should_enqueue, threshold_info).
    threshold_info: Description of which threshold was applied (for logging)
    
    High-value collisions must satisfy:
    - alias_class in {covername, person_full, org, place}
    - not in stopword/generic eligibility blacklist
    
    Should enqueue if ALSO:
    - candidates ≤ per-class threshold (from COLLISION_ADJUDICABLE_MAX_CANDIDATES)
    - surface passes case rules (strict for covername/person_given, lenient for place/org/person_full)
    """
    # Must have at least one high-value alias_class
    high_value_classes = {'covername', 'person_full', 'org', 'place'}
    has_high_value = any(ai.alias_class in high_value_classes for ai in alias_infos)
    if not has_high_value:
        return False, False, None
    
    # Check eligibility (not stopword/generic)
    is_eligible, reason = is_alias_eligible_for_matching(alias_norm, alias_infos)
    if not is_eligible:
        return False, False, None
    
    # Determine per-class threshold
    # Use the most restrictive (lowest) threshold among high-value classes present
    candidate_count = len(alias_infos)
    threshold_classes = []
    
    # Collect thresholds for all high-value classes present
    class_thresholds = []
    for ai in alias_infos:
        if ai.alias_class in high_value_classes:
            class_threshold = COLLISION_ADJUDICABLE_MAX_CANDIDATES.get(ai.alias_class, COLLISION_ADJUDICABLE_MAX_CANDIDATES[None])
            class_thresholds.append((ai.alias_class, class_threshold))
            if ai.alias_class not in threshold_classes:
                threshold_classes.append(ai.alias_class)
    
    # Use the most restrictive (lowest) threshold
    if not class_thresholds:
        max_threshold = COLLISION_ADJUDICABLE_MAX_CANDIDATES[None]
        threshold_classes = ['default']
    else:
        max_threshold = min(threshold for _, threshold in class_thresholds)
    
    threshold_info = f"threshold={max_threshold} (classes={','.join(threshold_classes)}, candidates={candidate_count})"
    
    # Hard rule: never enqueue if too many candidates (per-class threshold)
    if candidate_count > max_threshold:
        return True, False, threshold_info  # High-value but too many candidates (log separately)
    
    # Case matching: strict for covername/person_given, lenient for place/org/person_full
    has_strict_case_required = any(ai.alias_class in ('covername', 'person_given') for ai in alias_infos)
    
    if has_strict_case_required:
        # Require case match for covername/person_given
        has_valid_case = any(check_case_match(surface, ai) 
                            for ai in alias_infos 
                            if ai.alias_class in ('covername', 'person_given'))
        if not has_valid_case:
            return True, False, f"{threshold_info}, case_mismatch"  # High-value but case doesn't match
    else:
        # For place/org/person_full: case match is nice but not required unless explicitly requested
        # Check if any candidate explicitly requires case matching
        has_explicit_case_requirement = any(
            ai.match_case in ('case_sensitive', 'upper_only', 'titlecase_only') 
            for ai in alias_infos
        )
        if has_explicit_case_requirement:
            # If case is explicitly required, enforce it
            has_valid_case = any(check_case_match(surface, ai) for ai in alias_infos)
            if not has_valid_case:
                return True, False, f"{threshold_info}, case_mismatch"  # High-value but case doesn't match
        # Otherwise: case is lenient, allow through
    
    return True, True, threshold_info  # High-value and should enqueue


def is_collision_harmless(alias_norm: str, alias_infos: List[AliasInfo]) -> bool:
    """
    Determine if a collision is "harmless" (too common/ambiguous to be worth reviewing).
    
    Uses configurable thresholds (COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES).
    
    Collision is harmless if:
    - Single token AND many candidates (>threshold)
    - Person given name (single token) with multiple entities (unless explicitly allowed)
    - Generic word class
    - Not eligible for matching (stopwords, etc.)
    """
    if len(alias_infos) <= 1:
        return False
    
    # Check eligibility first
    is_eligible, reason = is_alias_eligible_for_matching(alias_norm, alias_infos)
    if not is_eligible:
        return True  # Not eligible = harmless (don't treat as collision)
    
    tokens = alias_norm.split()
    is_single_token = len(tokens) == 1
    
    # Many candidates (>threshold) for single token = harmless
    if is_single_token and len(alias_infos) > COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES:
        return True
    
    # Person given name with multiple entities (unless explicitly allowed)
    if is_single_token:
        person_given_count = sum(1 for ai in alias_infos 
                                 if ai.entity_type == 'person' 
                                 and ai.alias_class == 'person_given'
                                 and not ai.allow_ambiguous_person_token)
        if person_given_count > 0 and len(alias_infos) > 1:
            return True
    
    # Generic word class = harmless
    if any(ai.alias_class == 'generic_word' for ai in alias_infos):
        return True
    
    return False


def is_entity_like_ngram(ngram_text: str) -> bool:
    """
    Check if n-gram looks entity-like (for debugging unmatched n-grams).
    
    Filters out common function words and only tracks potentially interesting n-grams:
    - TitleCase (e.g., "Moscow", "New York")
    - ALLCAPS (e.g., "CIA", "MI6")
    - Contains digits+letters (e.g., "MI6", "3M")
    - Length ≥ 3 (filters out "to", "is", etc.)
    """
    tokens = ngram_text.split()
    
    # Must be at least 3 chars total
    if len(ngram_text.replace(' ', '')) < 3:
        return False
    
    # Check if any token is TitleCase or ALLCAPS
    for token in tokens:
        if len(token) >= 2:
            # TitleCase: first letter uppercase, rest lowercase
            if token[0].isupper() and token[1:].islower():
                return True
            # ALLCAPS: all letters uppercase
            if token.isupper() and token.isalpha():
                return True
            # Contains digits + letters (e.g., "MI6", "3M")
            if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
                return True
    
    return False


def extract_surface_from_tokens(
    chunk_text: str,
    token_start: int,
    token_end: int,
    original_tokens: List[Tuple[int, int, str]]
) -> Tuple[str, str, str]:
    """
    Extract surface text from chunk using token boundaries.
    
    Returns (surface, surface_norm, surface_quality).
    surface_quality: 'exact' if we can extract from original, 'approx' if normalized fallback.
    """
    # Extract from original text at token positions
    if token_start < len(chunk_text) and token_end <= len(chunk_text):
        surface = chunk_text[token_start:token_end].strip()
        surface_norm = normalize_alias(surface)
        surface_quality = 'exact'
    else:
        surface = ''
        surface_norm = ''
        surface_quality = 'approx'
    
    return surface, surface_norm, surface_quality


def load_all_aliases(conn, *, collection_slug: Optional[str] = None) -> Tuple[Dict[str, List[AliasInfo]], Set[str]]:
    """
    Load all entity aliases with their normalized forms and policy settings.
    Returns tuple: (aliases_by_norm dict, alias_norm_set)
    
    aliases_by_norm: dict mapping alias_norm -> [AliasInfo, ...]
    alias_norm_set: set of all alias_norms for fast lookup
    
    Note: Multiple entities can have the same alias_norm (e.g., "Smith", "Washington").
    If collection_slug is provided, can filter by collection (future: add provenance to aliases).
    """
    with conn.cursor() as cur:
        # Check which policy columns exist
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'entity_aliases' 
            AND column_name IN ('is_auto_match', 'min_chars', 'match_case', 'match_mode', 
                                'is_numeric_entity', 'alias_class', 'allow_ambiguous_person_token', 'requires_context')
        """)
        policy_cols = {row[0] for row in cur.fetchall()}
        
        # Build query with available columns
        select_cols = [
            "ea.entity_id",
            "ea.alias",
            "ea.alias_norm",
            "e.entity_type",
        ]
        
        if 'is_auto_match' in policy_cols:
            select_cols.append("COALESCE(ea.is_auto_match, true) AS is_auto_match")
        else:
            select_cols.append("true AS is_auto_match")
        
        if 'min_chars' in policy_cols:
            select_cols.append("COALESCE(ea.min_chars, 1) AS min_chars")
        elif 'min_token_len' in policy_cols:
            select_cols.append("COALESCE(ea.min_token_len, 1) AS min_chars")
        else:
            select_cols.append("1 AS min_chars")
        
        select_cols.append("COALESCE(ea.match_case, 'any') AS match_case")
        select_cols.append("COALESCE(ea.match_mode, 'token') AS match_mode")
        
        if 'is_numeric_entity' in policy_cols:
            select_cols.append("COALESCE(ea.is_numeric_entity, false) AS is_numeric_entity")
        else:
            select_cols.append("false AS is_numeric_entity")
        
        if 'alias_class' in policy_cols:
            select_cols.append("ea.alias_class")
        else:
            select_cols.append("NULL AS alias_class")
        
        if 'allow_ambiguous_person_token' in policy_cols:
            select_cols.append("COALESCE(ea.allow_ambiguous_person_token, false) AS allow_ambiguous_person_token")
        else:
            select_cols.append("false AS allow_ambiguous_person_token")
        
        if 'requires_context' in policy_cols:
            select_cols.append("ea.requires_context")
        else:
            select_cols.append("NULL AS requires_context")
        
        query = f"""
            SELECT {', '.join(select_cols)}
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.entity_id
            WHERE ea.is_matchable = true
            ORDER BY ea.entity_id, ea.id
        """
        
        cur.execute(query)
        
        aliases_by_norm: DefaultDict[str, List[AliasInfo]] = defaultdict(list)
        alias_norm_set: Set[str] = set()
        
        for row in cur.fetchall():
            entity_id = row[0]
            alias = row[1]
            alias_norm = row[2]
            entity_type = row[3]
            is_auto_match = row[4]
            
            min_chars = row[5]
            match_case = row[6]
            match_mode = row[7]
            is_numeric_entity = row[8]
            alias_class = row[9]
            allow_ambiguous_person_token = row[10]
            requires_context = row[11]
            
            # Fallback: set alias_class based on entity_type if NULL
            # Handle all entity_type values, not just person/org/place
            if alias_class is None:
                tokens = alias_norm.split()
                # Map entity_type to alias_class
                if entity_type == 'cover_name' or entity_type == 'covername':
                    alias_class = 'covername'
                elif entity_type == 'person':
                    alias_class = 'person_given' if len(tokens) == 1 else 'person_full'
                elif entity_type == 'org':
                    alias_class = 'org'
                elif entity_type == 'place':
                    alias_class = 'place'
                elif entity_type in ('other', 'topic', 'role'):
                    # Generic/other types default to generic_word
                    alias_class = 'generic_word'
                # Otherwise leave as None (unknown types)
            
            # Default is_auto_match=false for generic_word unless explicitly whitelisted
            # This prevents "FRIENDS" (other) from matching when it shouldn't
            if alias_class == 'generic_word':
                # Only allow auto-match if explicitly set to true in DB
                # If is_auto_match was defaulted (COALESCE), it's likely false
                # We can't easily detect "explicit vs defaulted" here, so use a heuristic:
                # Allow only if alias is ALLCAPS and short (likely a codeword) or explicitly whitelisted
                if is_auto_match:
                    # Check if it looks like a codeword (ALLCAPS, short)
                    alias_upper = alias.upper()
                    is_codeword_like = (alias == alias_upper and 
                                       len(alias) <= 6 and 
                                       alias.isalpha())
                    if not is_codeword_like:
                        # Not a codeword - disable auto-match for generic_word
                        is_auto_match = False
            
            # Detect placeholder/template aliases and disable auto-match
            # Patterns like "Unidentified Soviet intelligence source/agent"
            placeholder_patterns = [
                'unidentified',
                'unknown',
                'unnamed',
                'template',
                'placeholder',
                'source/agent',
                'intelligence source',
            ]
            alias_lower = alias.lower()
            is_placeholder = any(pattern in alias_lower for pattern in placeholder_patterns)
            if is_placeholder and is_auto_match:
                # Disable auto-match for placeholder aliases
                is_auto_match = False
                # Optionally set alias_class to generic_word
                if alias_class is None:
                    alias_class = 'generic_word'
            
            # Constrain single-token person names aggressively
            # viktor, aleksej, john, etc. will always collide
            if (entity_type == 'person' 
                and alias_class == 'person_given' 
                and len(tokens) == 1
                and not allow_ambiguous_person_token):
                # Default to is_auto_match=false for single-token person given names
                # unless explicitly allowed
                is_auto_match = False
            
            # Set case policies for covernames vs places
            # Covernames should match only when uppercase (MOSCOW)
            # Places can match with any case (Moscow, moscow, MOSCOW)
            if alias_class == 'covername' and match_case == 'any':
                # Default covernames to upper_only
                match_case = 'upper_only'
            elif alias_class == 'place' and match_case == 'any':
                # Places default to 'any' (case-insensitive) - already set
                pass  # Keep 'any'
            
            alias_info = AliasInfo(
                entity_id=entity_id,
                original_alias=alias,
                alias_norm=alias_norm,
                entity_type=entity_type,
                is_auto_match=is_auto_match,
                min_chars=min_chars,
                match_case=match_case,
                match_mode=match_mode,
                is_numeric_entity=is_numeric_entity,
                alias_class=alias_class,
                allow_ambiguous_person_token=allow_ambiguous_person_token,
                requires_context=requires_context,
            )
            
            aliases_by_norm[alias_norm].append(alias_info)
            alias_norm_set.add(alias_norm)
        
        return dict(aliases_by_norm), alias_norm_set


def is_purely_numeric(alias: str) -> bool:
    """Check if alias is purely numeric (digits with optional trivial punctuation)."""
    # Remove common punctuation (periods, commas, dashes)
    cleaned = re.sub(r'[.,\-]', '', alias.strip())
    # Check if remaining is all digits
    return cleaned.isdigit()


def has_contentful_token(tokens: List[str], min_chars: int = 3) -> bool:
    """
    Check if at least one token is contentful (not all stopwords, has sufficient length).
    
    A token is contentful if:
    - Length ≥ min_chars AND
    - Not a stopword AND
    - (Starts with a letter OR is an acronym like "U.S.", "NKVD", etc.)
    """
    STOP_WORDS = {'to', 'of', 'in', 'on', 'at', 'for', 'the', 'a', 'an', 'and', 'or', 'but'}
    for token in tokens:
        token_clean = token.strip().lower()
        if len(token_clean) >= min_chars and token_clean not in STOP_WORDS:
            # Check if starts with letter OR is acronym-like (all caps, dots, etc.)
            if token_clean and (token_clean[0].isalpha() or is_acronym_like(token_clean)):
                return True
    return False


def is_acronym_like(token: str) -> bool:
    """
    Check if token looks like an acronym (e.g., "u.s.", "nkvd", "f.b.i.").
    Patterns: all letters, or letters with dots/periods.
    """
    # Remove dots and check if remaining is all letters
    cleaned = token.replace('.', '').replace(',', '')
    if not cleaned:
        return False
    # Acronym-like if: all letters, length 2-10, and original had dots or is short
    return cleaned.isalpha() and 2 <= len(cleaned) <= 10 and ('.' in token or len(cleaned) <= 5)


def find_exact_matches(
    chunk_text: str,
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    chunk_id: int,  # Required for collision queue items
    document_id: int,  # Required for collision queue items
    rejection_stats: Optional[Dict[str, Dict[str, int]]] = None,
    collision_queue: Optional[List[Dict]] = None,
    matched_alias_norms: Optional[Set[str]] = None,  # Track matched alias_norms for debugging
    unmatched_ngrams: Optional[Dict[str, int]] = None,  # Track frequent unmatched n-grams for debugging
    preferred_entity_id_map: Optional[Dict[Tuple[Optional[str], str], int]] = None,  # Optional: (scope, alias_norm) -> preferred entity_id
    scope: Optional[str] = None,  # Collection slug or other scope identifier for preferred_entity_id lookups
) -> List[Tuple[int, str, str, str, int, int]]:
    """
    Find exact alias matches using tokenization and n-gram lookup (V2).
    
    Strategy:
    1. Tokenize ORIGINAL chunk text (not normalized) - ensures alignment with alias normalization
    2. Generate n-grams (1-5 tokens), normalizing each token individually via normalize_alias
    3. Lookup n-grams in alias_norm_set (hash lookup, not regex iteration)
    4. Longest-match per-position (left→right): at each position, try longest phrase first
    5. Apply policy checks (is_auto_match, min_chars, case matching, context gates)
    6. Extract surface via token boundaries
    
    Normalization alignment: We tokenize original text, then normalize each token. This matches
    how aliases are stored (normalize_alias removes punctuation, lowercases, etc.).
    
    Returns list of (entity_id, surface, surface_norm, surface_quality, start_char, end_char) tuples.
    """
    matches = []
    
    # Tokenize ORIGINAL chunk text (not normalized) - we normalize tokens individually
    # This ensures alignment: tokenize original, normalize each token = same as alias normalization
    original_tokens = tokenize_text(chunk_text)
    if not original_tokens:
        return matches
    
    # Longest-match per-position (left→right): at each position, try longest phrase first
    # Track matched positions to prevent overlaps
    matched_positions: Set[Tuple[int, int]] = set()
    
    # Process left→right: for each starting position, try longest match first
    # OPTIMIZATION: Build n-grams on-the-fly instead of scanning the whole list
    i = 0
    while i < len(original_tokens):
        # Try longest n-gram starting at position i (max_n down to 1)
        matched = False
        for n in range(min(5, len(original_tokens) - i), 0, -1):
            # Build n-gram on-the-fly from original_tokens[i:i+n]
            if i + n > len(original_tokens):
                continue
            
            # Get token span
            token_span = original_tokens[i:i+n]
            token_start = token_span[0][0]
            token_end = token_span[-1][1]
            
            # Normalize each token individually and join
            normalized_tokens = []
            for tok_start, tok_end, tok_text in token_span:
                normalized_tokens.append(normalize_alias(tok_text))
            ngram_text = ' '.join(normalized_tokens)
            
            ngram_start, ngram_end = token_start, token_end
            
            # Skip if this position is already matched
            if (ngram_start, ngram_end) in matched_positions:
                continue
            
            # Fast lookup in alias set (hash lookup, not regex)
            if ngram_text not in alias_norm_set:
                # Track unmatched n-grams for debugging (only entity-like ones)
                # Use surface (original text) for entity-like check, not normalized
                if unmatched_ngrams is not None:
                    # Extract surface for entity-like check
                    surface_for_check, _, _ = extract_surface_from_tokens(chunk_text, ngram_start, ngram_end, original_tokens)
                    if surface_for_check and is_entity_like_ngram(surface_for_check):
                        unmatched_ngrams[ngram_text] = unmatched_ngrams.get(ngram_text, 0) + 1
                continue
            
            # Get alias infos for this ngram
            alias_infos = aliases_by_norm.get(ngram_text, [])
            if not alias_infos:
                continue
            
            # Extract surface early (needed for collision resolution)
            surface, surface_norm, _ = extract_surface_from_tokens(chunk_text, ngram_start, ngram_end, original_tokens)
            if not surface:
                surface = alias_infos[0].original_alias if alias_infos else ''  # Fallback
                surface_norm = ngram_text
            if not surface_norm:
                surface_norm = normalize_alias(surface) if surface else ngram_text
            
            # Check eligibility first (before collision handling)
            # Pass surface so eligibility can check case for covernames (MAY/REF/WILL)
            # Only track not_eligible if n-gram would have matched (is in alias_norm_set)
            # This answers: "how often did text contain 'TO' but we skipped it?"
            is_eligible, eligibility_reason = is_alias_eligible_for_matching(ngram_text, alias_infos, surface=surface)
            if not is_eligible:
                # Not eligible - log and skip (don't treat as collision)
                # Only count if this n-gram is in alias_norm_set (would have matched)
                if rejection_stats is not None and ngram_text in alias_norm_set:
                    rejection_stats.setdefault('not_eligible', {})
                    rejection_stats['not_eligible'][f"{ngram_text} ({eligibility_reason})"] = \
                        rejection_stats['not_eligible'].get(f"{ngram_text} ({eligibility_reason})", 0) + 1
                continue  # Try shorter n-gram
            
            # Track if this is a collision (for policy block tracking)
            is_collision = len(alias_infos) > 1
            
            # Resolve collision if present, otherwise use single candidate
            resolved_alias_info: Optional[AliasInfo] = None
            
            if is_collision:
                # 1) Filtered-single-candidate resolution: filter by policies first
                # This is more robust than Rule 0/1/2/3 and will catch cases where
                # only one candidate passes all policy checks
                filtered_candidates = []
                for ai in alias_infos:
                    # Check eligibility
                    is_eligible, _ = is_candidate_eligible_for_matching(ai, ngram_text, surface)
                    if not is_eligible:
                        continue
                    
                    # Check is_auto_match
                    if not ai.is_auto_match:
                        continue
                    
                    # Check case (lenient for non-covernames)
                    if ai.alias_class in ('covername', 'person_given'):
                        if not check_case_match(surface, ai):
                            continue
                    else:
                        # For place/org/person_full: enforce only if explicitly strict
                        if ai.match_case in ('case_sensitive', 'upper_only', 'titlecase_only'):
                            if not check_case_match(surface, ai):
                                continue
                    
                    # Check min_chars (for first token)
                    tokens = ngram_text.split()
                    if len(tokens) == 1:
                        if len(tokens[0]) < ai.min_chars:
                            continue
                    else:
                        max_token_len = max(len(t) for t in tokens)
                        if max_token_len < ai.min_chars:
                            continue
                    
                    # Check numeric policy
                    if is_purely_numeric(ai.original_alias) and not ai.is_numeric_entity:
                        continue
                    
                    filtered_candidates.append(ai)
                
                # If exactly one candidate passes all filters, use it
                if len(filtered_candidates) == 1:
                    resolved_alias_info = filtered_candidates[0]
                    # Track this as "filtered single candidate"
                    if rejection_stats is not None:
                        rejection_stats.setdefault('collision_auto_resolved', {})
                        rejection_stats['collision_auto_resolved'][f"{ngram_text} (filtered_single_candidate)"] = \
                            rejection_stats['collision_auto_resolved'].get(f"{ngram_text} (filtered_single_candidate)", 0) + 1
                    if rejection_stats is not None:
                        rejection_stats.setdefault('collision_logged', {})
                        rejection_stats['collision_logged'][ngram_text] = \
                            rejection_stats['collision_logged'].get(ngram_text, 0) + 1
                else:
                    # 2) Dominance resolution
                    dominant, rule_used, rule_detail = find_dominant_candidate(
                        ngram_text, alias_infos, surface, preferred_entity_id_map, scope=scope
                    )
                    if dominant is not None:
                        resolved_alias_info = dominant
                        # Track auto-resolution for auditing
                        if rejection_stats is not None:
                            rejection_stats.setdefault('collision_auto_resolved', {})
                            rule_label = {
                                'rule0': 'preferred_entity_id',
                                'rule1': 'is_auto_match',
                                'rule1c': 'unique_auto_match_ignore_case',
                                'rule2': 'case_match_constrained',
                                'rule3': 'safe_dominance'
                            }.get(rule_used, rule_used)
                            detail_str = f" ({rule_detail})" if rule_detail else ""
                            rejection_stats['collision_auto_resolved'][f"{ngram_text} ({rule_label}{detail_str})"] = \
                                rejection_stats['collision_auto_resolved'].get(f"{ngram_text} ({rule_label}{detail_str})", 0) + 1
                        # Also count in collision_logged for visibility
                        if rejection_stats is not None:
                            rejection_stats.setdefault('collision_logged', {})
                            rejection_stats['collision_logged'][ngram_text] = \
                                rejection_stats['collision_logged'].get(ngram_text, 0) + 1
                    else:
                        # No dominant candidate - log "dominance attempted but no rule fired"
                        if rejection_stats is not None:
                            rejection_stats.setdefault('collision_dominance_none', {})
                            rejection_stats['collision_dominance_none'][ngram_text] = \
                                rejection_stats['collision_dominance_none'].get(ngram_text, 0) + 1
                
                # If collision is UNRESOLVED, handle it (enqueue or skip)
                if resolved_alias_info is None:
                    # UNRESOLVED collision → maybe enqueue, then skip mention emission
                    if is_collision_harmless(ngram_text, alias_infos):
                        # Harmless collision - just log, don't enqueue
                        if rejection_stats is not None:
                            rejection_stats.setdefault('collision_harmless', {})
                            rejection_stats['collision_harmless'][ngram_text] = \
                                rejection_stats['collision_harmless'].get(ngram_text, 0) + 1
                        continue  # Try shorter n-gram
                    
                    # Check if high-value (worth enqueuing)
                    is_high_value, should_enqueue, threshold_info = is_collision_high_value(ngram_text, alias_infos, surface)
                    
                    if is_high_value and should_enqueue:
                        # High-value collision - enqueue for review
                        if collision_queue is not None:
                            candidate_entity_ids = [ai.entity_id for ai in alias_infos]
                            # Extract context
                            context_start = max(0, ngram_start - 100)
                            context_end = min(len(chunk_text), ngram_end + 100)
                            context_excerpt = chunk_text[context_start:context_end]
                            
                            collision_queue.append({
                                'chunk_id': chunk_id,  # Set immediately
                                'document_id': document_id,  # Set immediately
                                'surface': surface,  # Actual surface from chunk
                                'alias_norm': ngram_text,  # For idempotency and joining
                                'surface_norm': surface_norm,  # Normalized form for debugging
                                'context_excerpt': context_excerpt,
                                'candidate_entity_ids': candidate_entity_ids,
                                'method': 'alias_exact_collision',
                                'method_version': 'v1',
                            })
                        
                        if rejection_stats is not None:
                            rejection_stats.setdefault('collision_high_value_enqueued', {})
                            detail_str = f" ({threshold_info})" if threshold_info else ""
                            rejection_stats['collision_high_value_enqueued'][f"{ngram_text}{detail_str}"] = \
                                rejection_stats['collision_high_value_enqueued'].get(f"{ngram_text}{detail_str}", 0) + 1
                    elif is_high_value:
                        # High-value but too many candidates or case mismatch
                        if rejection_stats is not None:
                            rejection_stats.setdefault('collision_high_value_too_many', {})
                            detail_str = f" ({threshold_info})" if threshold_info else ""
                            rejection_stats['collision_high_value_too_many'][f"{ngram_text}{detail_str}"] = \
                                rejection_stats['collision_high_value_too_many'].get(f"{ngram_text}{detail_str}", 0) + 1
                    
                    # Log collision (even if not enqueued)
                    if rejection_stats is not None:
                        rejection_stats.setdefault('collision_logged', {})
                        rejection_stats['collision_logged'][ngram_text] = \
                            rejection_stats['collision_logged'].get(ngram_text, 0) + 1
                    
                    continue  # Skip - requires adjudication or not high-value
            else:
                # Single entity match - use it directly
                resolved_alias_info = alias_infos[0]
            
            # RESOLVED collision or single candidate → proceed with policy checks
            alias_info = resolved_alias_info
            
            # Context gate check (if requires_context is set)
            if alias_info.requires_context:
                # TODO: Implement context gate checking
                # For now, if requires_context is set, we skip (don't auto-match)
                if rejection_stats is not None:
                    rejection_stats.setdefault('context_gate', {})
                    rejection_stats['context_gate'][f"{alias_info.original_alias} (requires_context={alias_info.requires_context})"] = \
                        rejection_stats['context_gate'].get(f"{alias_info.original_alias} (requires_context={alias_info.requires_context})", 0) + 1
                
                if CONTEXT_GATE_FAILURE_ENQUEUE and collision_queue is not None:
                    # Surface already extracted above
                    if not surface:
                        surface = alias_info.original_alias
                        surface_norm = ngram_text
                    if not surface_norm:
                        surface_norm = normalize_alias(surface) if surface else ngram_text
                    
                    # Optionally enqueue for review
                    collision_queue.append({
                        'chunk_id': chunk_id,
                        'document_id': document_id,
                        'surface': surface,
                        'alias_norm': ngram_text,
                        'surface_norm': surface_norm,
                        'context_excerpt': chunk_text[max(0, ngram_start - 100):min(len(chunk_text), ngram_end + 100)],
                        'candidate_entity_ids': [alias_info.entity_id],
                        'method': 'context_gate_failed',
                        'method_version': 'v1',
                    })
                
                continue  # Skip - context gate not implemented yet
            
            # Policy check 1: is_auto_match
            if not alias_info.is_auto_match:
                if rejection_stats is not None:
                    rejection_stats.setdefault('policy', {})
                    rejection_stats['policy'][f"{alias_info.original_alias} (auto_match disabled)"] = \
                        rejection_stats['policy'].get(f"{alias_info.original_alias} (auto_match disabled)", 0) + 1
                    # Track resolved collision that got blocked
                    if is_collision:
                        rejection_stats.setdefault('collision_resolved_policy_blocked', {})
                        rejection_stats['collision_resolved_policy_blocked'][f"{ngram_text} (is_auto_match=false)"] = \
                            rejection_stats['collision_resolved_policy_blocked'].get(f"{ngram_text} (is_auto_match=false)", 0) + 1
                continue  # Try shorter n-gram
            
            # Policy check 2: Reject purely numeric (unless explicitly allowed)
            if is_purely_numeric(alias_info.original_alias) and not alias_info.is_numeric_entity:
                if rejection_stats is not None:
                    rejection_stats.setdefault('policy', {})
                    rejection_stats['policy'][f"{alias_info.original_alias} (purely numeric)"] = \
                        rejection_stats['policy'].get(f"{alias_info.original_alias} (purely numeric)", 0) + 1
                continue  # Try shorter n-gram
            
            # Policy check 3: min_chars
            tokens = ngram_text.split()
            if len(tokens) == 1:
                # Single token: apply min_chars to token
                if len(tokens[0]) < alias_info.min_chars:
                    if rejection_stats is not None:
                        rejection_stats.setdefault('policy', {})
                        rejection_stats['policy'][f"{alias_info.original_alias} (chars < {alias_info.min_chars})"] = \
                            rejection_stats['policy'].get(f"{alias_info.original_alias} (chars < {alias_info.min_chars})", 0) + 1
                    continue  # Try shorter n-gram
            else:
                # Multi-token: require at least one token ≥ min_chars
                max_token_len = max(len(t) for t in tokens)
                if max_token_len < alias_info.min_chars:
                    if rejection_stats is not None:
                        rejection_stats.setdefault('policy', {})
                        rejection_stats['policy'][f"{alias_info.original_alias} (phrase: no token ≥ {alias_info.min_chars})"] = \
                            rejection_stats['policy'].get(f"{alias_info.original_alias} (phrase: no token ≥ {alias_info.min_chars})", 0) + 1
                    continue  # Try shorter n-gram
            
            # Policy check 4: Person given name (single token) - be conservative
            if (alias_info.entity_type == 'person' 
                and alias_info.alias_class == 'person_given' 
                and len(tokens) == 1
                and not alias_info.allow_ambiguous_person_token):
                # Check if this alias_norm maps to multiple entities in full set
                # (We already checked collision above, but this is a conservative gate)
                # For now, allow if no collision was detected
                pass  # Allow through - collision check above handles it
            
            # Policy check 5: Contentful token for phrases
            if len(tokens) > 1:
                if not has_contentful_token(tokens, min_chars=3):
                    if rejection_stats is not None:
                        rejection_stats.setdefault('policy', {})
                        rejection_stats['policy'][f"{alias_info.original_alias} (phrase: no contentful token)"] = \
                            rejection_stats['policy'].get(f"{alias_info.original_alias} (phrase: no contentful token)", 0) + 1
                    continue  # Try shorter n-gram
            
            # Surface already extracted above (for collision handling)
            # Just get quality (re-extract to get quality flag)
            _, _, surface_quality = extract_surface_from_tokens(
                chunk_text, ngram_start, ngram_end, original_tokens
            )
            
            # Ensure surface and surface_norm are set (should already be from above, but double-check)
            if not surface:
                # Fallback to original alias
                surface = alias_info.original_alias
                surface_norm = ngram_text
                surface_quality = 'approx'
            
            # Ensure surface_norm is set
            if not surface_norm:
                surface_norm = normalize_alias(surface) if surface else ngram_text
            
            # Policy check 6: Case matching
            if not check_case_match(surface, alias_info):
                if rejection_stats is not None:
                    rejection_stats.setdefault('case_mismatch', {})
                    rejection_stats['case_mismatch'][alias_info.original_alias] = \
                        rejection_stats['case_mismatch'].get(alias_info.original_alias, 0) + 1
                    # Track resolved collision that got blocked
                    if is_collision:
                        rejection_stats.setdefault('collision_resolved_policy_blocked', {})
                        rejection_stats['collision_resolved_policy_blocked'][f"{ngram_text} (case_mismatch)"] = \
                            rejection_stats['collision_resolved_policy_blocked'].get(f"{ngram_text} (case_mismatch)", 0) + 1
                continue  # Try shorter n-gram
            
            # All checks passed - create match
            matches.append((
                alias_info.entity_id,
                surface,
                surface_norm,  # Always store (not just when approx)
                surface_quality,
                None,  # start_char (v1)
                None,  # end_char (v1)
            ))
            
            # Track matched alias_norm for debugging
            if matched_alias_norms is not None:
                matched_alias_norms.add(ngram_text)
            
            # Mark position as matched (prevent shorter overlaps)
            matched_positions.add((ngram_start, ngram_end))
            
            # Advance past this match (consume n tokens)
            i += n
            matched = True
            break  # Found match, move to next position
        
        if not matched:
            # No match at this position, advance 1 token
            i += 1
    
    return matches


def extract_mentions_batch(
    conn,
    chunks: List[Tuple[int, str, Optional[int]]],
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    *,
    dry_run: bool = False,
    show_samples: bool = False,
    max_samples: int = 10,
    rejection_stats: Optional[Dict[str, Dict[str, int]]] = None,
    collision_queue: Optional[List[Dict]] = None,
    matched_alias_norms: Optional[Set[str]] = None,
    unmatched_ngrams: Optional[Dict[str, int]] = None,
    preferred_entity_id_map: Optional[Dict[Tuple[Optional[str], str], int]] = None,
    scope: Optional[str] = None,
) -> Tuple[int, int, List[Dict]]:
    """
    Extract mentions for a batch of chunks.
    Returns (chunks_processed, mentions_found, sample_mentions).
    """
    mentions_to_insert = []
    sample_mentions = []
    chunks_processed = 0
    chunks_without_metadata = 0
    
    for chunk_id, chunk_text, document_id in chunks:
        chunks_processed += 1
        
        if document_id is None:
            chunks_without_metadata += 1
            if chunks_without_metadata <= 5:  # Log first few warnings
                print(f"  Warning: chunk_id={chunk_id} has no chunk_metadata (skipping)", file=sys.stderr)
            continue
        
        # Find exact matches (with policy checks)
        # Collision queue items will have chunk_id set immediately in find_exact_matches
        # Note: We pass original chunk_text (not normalized) - tokenization normalizes each token
        matches = find_exact_matches(
            chunk_text, 
            aliases_by_norm,
            alias_norm_set,
            chunk_id,
            document_id,
            rejection_stats=rejection_stats,
            collision_queue=collision_queue,
            matched_alias_norms=matched_alias_norms,
            unmatched_ngrams=unmatched_ngrams,
            preferred_entity_id_map=preferred_entity_id_map,
        )
        
        # Prepare mentions for insertion
        for entity_id, surface, surface_norm, surface_quality, start_char, end_char in matches:
            mention = {
                'entity_id': entity_id,
                'chunk_id': chunk_id,
                'document_id': document_id,
                'surface': surface,
                'surface_norm': surface_norm,  # Always store (for debugging and downstream normalization)
                'surface_quality': surface_quality,  # 'exact' or 'approx' - historians should filter/triage approx
                'start_char': start_char,
                'end_char': end_char,
                'confidence': 1.0,
                'method': 'alias_exact',
            }
            mentions_to_insert.append(mention)
            
            # Collect samples if requested
            if show_samples and len(sample_mentions) < max_samples:
                sample_mentions.append(mention)
    
    if not dry_run and mentions_to_insert:
        # Insert mentions with idempotency
        # Check for existing mentions to avoid duplicates (idempotency)
        with conn.cursor() as cur:
            # Build tuples for checking
            check_tuples = [
                (m['chunk_id'], m['entity_id'], m['surface'], m['method'])
                for m in mentions_to_insert
            ]
            
            # Query existing mentions using a temp table approach
            # Create temp table with values to check
            cur.execute("""
                CREATE TEMP TABLE IF NOT EXISTS mention_check (
                    chunk_id BIGINT,
                    entity_id BIGINT,
                    surface TEXT,
                    method TEXT
                ) ON COMMIT DROP
            """)
            
            # Insert check values
            execute_values(
                cur,
                "INSERT INTO mention_check (chunk_id, entity_id, surface, method) VALUES %s",
                check_tuples,
                template=None,
                page_size=1000,
            )
            
            # Find existing mentions
            cur.execute("""
                SELECT mc.chunk_id, mc.entity_id, mc.surface, mc.method
                FROM mention_check mc
                INNER JOIN entity_mentions em ON
                    em.chunk_id = mc.chunk_id
                    AND em.entity_id = mc.entity_id
                    AND em.surface = mc.surface
                    AND em.method = mc.method
            """)
            
            existing = set(cur.fetchall())
            
            # Filter to only new mentions
            new_mentions = [
                m for m in mentions_to_insert
                if (m['chunk_id'], m['entity_id'], m['surface'], m['method']) not in existing
            ]
            
            if new_mentions:
                # Check if surface_quality columns exist
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'entity_mentions' 
                    AND column_name IN ('surface_norm', 'surface_quality')
                """)
                has_surface_quality = len(cur.fetchall()) == 2
                
                if has_surface_quality:
                    # Use execute_values for batch insert with surface quality
                    execute_values(
                        cur,
                        """
                        INSERT INTO entity_mentions 
                            (entity_id, chunk_id, document_id, surface, surface_norm, surface_quality, 
                             start_char, end_char, confidence, method)
                        VALUES %s
                        """,
                        [
                            (
                                m['entity_id'],
                                m['chunk_id'],
                                m['document_id'],
                                m['surface'],
                                m.get('surface_norm'),
                                m.get('surface_quality', 'exact'),
                                m['start_char'],
                                m['end_char'],
                                m['confidence'],
                                m['method'],
                            )
                            for m in new_mentions
                        ],
                        template=None,
                        page_size=1000,
                    )
                else:
                    # Fallback for old schema
                    execute_values(
                        cur,
                        """
                        INSERT INTO entity_mentions 
                            (entity_id, chunk_id, document_id, surface, start_char, end_char, confidence, method)
                        VALUES %s
                        """,
                        [
                            (
                                m['entity_id'],
                                m['chunk_id'],
                                m['document_id'],
                                m['surface'],
                                m['start_char'],
                                m['end_char'],
                                m['confidence'],
                                m['method'],
                            )
                            for m in new_mentions
                        ],
                        template=None,
                        page_size=1000,
                    )
                conn.commit()
                return chunks_processed, len(new_mentions), sample_mentions
            else:
                conn.commit()  # Commit even if no new mentions (for temp table cleanup)
                return chunks_processed, 0, sample_mentions
    
    # Dry run or no mentions to insert
    return chunks_processed, len(mentions_to_insert), sample_mentions

def bulk_person_candidates(cur, patterns: List[Dict]) -> Dict[int, List[Tuple[int, str]]]:
    """
    Return mapping: alias_id -> [(person_entity_id, person_name), ...]
    computed set-wise (fast) rather than 1 query per pattern.
    """
    if not patterns:
        return {}

    cur.execute("""
        CREATE TEMP TABLE tmp_covername_person_aliases (
          alias_id BIGINT PRIMARY KEY,
          person_alias TEXT NOT NULL,
          person_alias_norm TEXT NOT NULL
        ) ON COMMIT DROP;
    """)

    rows = [(p["alias_id"], p["person_alias"], p["person_alias_norm"]) for p in patterns]
    execute_values(
        cur,
        "INSERT INTO tmp_covername_person_aliases (alias_id, person_alias, person_alias_norm) VALUES %s",
        rows,
        page_size=2000
    )

    cur.execute("""
        WITH person_aliases AS (
          SELECT ea.entity_id AS person_id, ea.alias_norm
          FROM entity_aliases ea
          JOIN entities e ON e.id = ea.entity_id
          WHERE e.entity_type = 'person'
        )
        SELECT
          t.alias_id,
          p.id AS person_id,
          p.canonical_name AS person_name
        FROM tmp_covername_person_aliases t
        JOIN entities p
          ON p.entity_type = 'person'
        LEFT JOIN person_aliases pa
          ON pa.person_id = p.id AND pa.alias_norm = t.person_alias_norm
        WHERE
          LOWER(TRIM(p.canonical_name)) = LOWER(TRIM(t.person_alias))
          OR pa.person_id IS NOT NULL
        ORDER BY t.alias_id, p.id;
    """)

    mapping: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for alias_id, pid, pname in cur.fetchall():
        mapping[alias_id].append((pid, pname))
    return dict(mapping)


def main():
    ap = argparse.ArgumentParser(
        description="Extract entity mentions from chunks using exact alias matching"
    )
    ap.add_argument("--collection", type=str, help="Filter by collection slug")
    ap.add_argument("--document-id", type=int, help="Filter by document ID")
    ap.add_argument("--chunk-id-range", type=str, help="Filter by chunk ID range (format: start:end)")
    ap.add_argument("--dry-run", action="store_true", help="Print counts only, no inserts")
    ap.add_argument("--show-samples", action="store_true", help="Show sample mentions (works with --dry-run)")
    ap.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples to show (default: 10)")
    ap.add_argument("--limit", type=int, help="Limit number of chunks processed")
    ap.add_argument("--batch-size", type=int, default=100, help="Process chunks in batches (default: 100)")
    
    args = ap.parse_args()
    
    # Validate arguments
    if not any([args.collection, args.document_id, args.chunk_id_range]):
        ap.error("Must specify at least one of: --collection, --document-id, --chunk-id-range")
    
    chunk_id_start = None
    chunk_id_end = None
    if args.chunk_id_range:
        chunk_id_start, chunk_id_end = parse_chunk_id_range(args.chunk_id_range)
    
    conn = get_conn()
    try:
        # Load all aliases once (memory-efficient for typical entity counts)
        print("Loading entity aliases...", file=sys.stderr)
        aliases_by_norm, alias_norm_set = load_all_aliases(conn, collection_slug=args.collection)
        print(f"  Loaded {len(aliases_by_norm)} unique normalized aliases ({len(alias_norm_set)} in lookup set)", file=sys.stderr)
        
        # Diagnostic: Print top 50 alias_norms by candidate count
        print("\nTop 50 alias_norms by candidate count (for alias_class diagnostic):", file=sys.stderr)
        sorted_aliases = sorted(aliases_by_norm.items(), key=lambda x: len(x[1]), reverse=True)[:50]
        mis_typed_entities = []  # Track potential mis-typed entities
        for alias_norm, infos in sorted_aliases:
            if len(infos) > 1:
                entity_types = set(ai.entity_type for ai in infos)
                alias_classes = set(ai.alias_class for ai in infos if ai.alias_class)
                sample_names = [ai.original_alias for ai in infos[:3]]
                sample_entity_types = [ai.entity_type for ai in infos[:3]]
                print(f"  {alias_norm}: {len(infos)} candidates, types={entity_types}, classes={alias_classes}, samples={sample_names}", file=sys.stderr)
                
                # Detect mis-typed entities: orgs stored as people, places as people, etc.
                # Example: "communist party" should be org, not person_full
                if 'person' in entity_types and len(entity_types) > 1:
                    # Mixed types - likely mis-typed
                    org_like = any('party' in ai.original_alias.lower() or 
                                  'committee' in ai.original_alias.lower() or
                                  'bureau' in ai.original_alias.lower() or
                                  'ministry' in ai.original_alias.lower()
                                  for ai in infos)
                    place_like = any(ai.alias_class == 'place' for ai in infos)
                    if org_like or place_like:
                        mis_typed_entities.append((alias_norm, entity_types, sample_names))
        
        if mis_typed_entities:
            print("\n⚠️  Potential mis-typed entities (orgs/places stored as people):", file=sys.stderr)
            for alias_norm, types, samples in mis_typed_entities[:20]:
                print(f"  {alias_norm}: types={types}, samples={samples}", file=sys.stderr)
            print("", file=sys.stderr)
        
        # Diagnostic: Show entity_type distribution
        entity_type_counts = {}
        for infos in aliases_by_norm.values():
            for ai in infos:
                entity_type_counts[ai.entity_type] = entity_type_counts.get(ai.entity_type, 0) + 1
        print(f"Entity type distribution: {entity_type_counts}", file=sys.stderr)
        print("", file=sys.stderr)
        
        # Query chunks
        print("Querying chunks...", file=sys.stderr)
        chunks = get_chunks_query(
            conn,
            collection_slug=args.collection,
            document_id=args.document_id,
            chunk_id_start=chunk_id_start,
            chunk_id_end=chunk_id_end,
            limit=args.limit,
        )
        total_chunks = len(chunks)
        print(f"  Found {total_chunks} chunks to process", file=sys.stderr)
        
        if total_chunks == 0:
            print("No chunks to process", file=sys.stderr)
            return
        
        # Process in batches
        batch_size = args.batch_size
        total_mentions = 0
        total_processed = 0
        all_samples = []
        rejection_stats = {}  # Track rejection statistics by category
        collision_queue = []  # Track alias collisions for review (not auto-inserted)
        matched_alias_norms = set()  # Track matched alias_norms for debugging
        unmatched_ngrams = {}  # Track frequent unmatched n-grams for debugging
        # Optional: (scope, alias_norm) -> preferred entity_id (can be loaded from DB or config)
        # Scope can be collection_slug, document_id, or None for global defaults
        # Example: preferred_entity_id_map = {
        #     (None, "moscow"): 123,  # Global default: Moscow (place)
        #     ("venona", "moscow"): 456,  # Venona-specific: MOSCOW (covername)
        # }
        preferred_entity_id_map: Dict[Tuple[Optional[str], str], int] = {}
        
        # TODO: Load preferred_entity_id_map from:
        # - Config file (JSON/YAML)
        # - Database table (preferred_entity_resolutions)
        # - Command-line argument
        # For now, empty map (can be populated manually for testing)
        
        print(f"\nProcessing chunks in batches of {batch_size}...", file=sys.stderr)
        if args.dry_run:
            print("  [DRY RUN] No changes will be made\n", file=sys.stderr)
        print("  Note: Alias collisions will be logged but not auto-resolved (requires review)\n", file=sys.stderr)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            processed, mentions, samples = extract_mentions_batch(
                conn,
                batch,
                aliases_by_norm,
                alias_norm_set,
                dry_run=args.dry_run,
                show_samples=args.show_samples,
                max_samples=args.max_samples,
                rejection_stats=rejection_stats,
                collision_queue=collision_queue,
                matched_alias_norms=matched_alias_norms,
                unmatched_ngrams=unmatched_ngrams,
                preferred_entity_id_map=preferred_entity_id_map,
                scope=args.collection,  # Pass collection slug as scope
            )
            
            total_processed += processed
            total_mentions += mentions
            if args.show_samples:
                all_samples.extend(samples)
                # Stop collecting if we have enough samples
                if len(all_samples) >= args.max_samples:
                    all_samples = all_samples[:args.max_samples]
                    break
            
            # Progress output
            rate = total_processed / (i + len(batch)) if (i + len(batch)) > 0 else 0
            print(
                f"  Batch {batch_num}/{total_batches}: "
                f"processed {total_processed}/{total_chunks} chunks, "
                f"found {total_mentions} mentions "
                f"({mentions} in this batch)",
                file=sys.stderr
            )
        
        # Summary
        print(f"\n{'Would extract' if args.dry_run else 'Extracted'}:", file=sys.stderr)
        print(f"  Chunks processed: {total_processed}", file=sys.stderr)
        print(f"  Mentions found: {total_mentions}", file=sys.stderr)
        
        # Show samples if requested
        if args.show_samples and all_samples:
            print(f"\nSample mentions (showing {len(all_samples)} of {total_mentions}):", file=sys.stderr)
            # Get entity names for display
            with conn.cursor() as cur:
                entity_ids = [m['entity_id'] for m in all_samples]
                if entity_ids:
                    cur.execute("""
                        SELECT id, canonical_name, entity_type
                        FROM entities
                        WHERE id = ANY(%s)
                    """, (entity_ids,))
                    entity_info = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
                    
                    for i, m in enumerate(all_samples, 1):
                        entity_name, entity_type = entity_info.get(m['entity_id'], ('?', '?'))
                        print(
                            f"  {i}. chunk_id={m['chunk_id']}, entity_id={m['entity_id']} "
                            f"({entity_name}, {entity_type}), surface='{m['surface']}'",
                            file=sys.stderr
                        )
        
        # Show collision statistics
        if collision_queue:
            print(f"\nAlias Collisions (high-value, require review/adjudication):", file=sys.stderr)
            print(f"  Found {len(collision_queue)} high-value collisions (enqueued)", file=sys.stderr)
            print(f"  These are logged but NOT auto-resolved (precision-first approach)", file=sys.stderr)
            print(f"  Use NER→review_queue workflow to adjudicate collisions", file=sys.stderr)
            print(f"  Note: Many collisions are auto-resolved via dominant candidate rules\n", file=sys.stderr)
        
        # Show debugging stats: matched alias_norms
        if matched_alias_norms:
            print(f"\nMatched Alias Norms (for debugging):", file=sys.stderr)
            print(f"  Total unique alias_norms matched: {len(matched_alias_norms)}", file=sys.stderr)
            if args.show_samples:
                # Show sample matched alias_norms
                sample_norms = sorted(list(matched_alias_norms))[:20]
                print(f"  Sample matched alias_norms:", file=sys.stderr)
                for norm in sample_norms:
                    print(f"    {norm}", file=sys.stderr)
            print("", file=sys.stderr)
        
        # Show debugging stats: frequent unmatched n-grams
        if unmatched_ngrams:
            print(f"\nFrequent Unmatched N-grams (for debugging normalization):", file=sys.stderr)
            print(f"  Total unique unmatched n-grams: {len(unmatched_ngrams)}", file=sys.stderr)
            # Show top 20 most frequent unmatched n-grams
            sorted_unmatched = sorted(unmatched_ngrams.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"  Top unmatched n-grams (may indicate normalization mismatch):", file=sys.stderr)
            for ngram, count in sorted_unmatched:
                print(f"    {ngram}: {count}", file=sys.stderr)
            print("", file=sys.stderr)
        
        # Show collision tiering thresholds (for auditability)
        print(f"\nCollision Tiering Thresholds (configurable):", file=sys.stderr)
        print(f"  Harmless (single token): >{COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES} candidates", file=sys.stderr)
        print(f"  Adjudicable (multi-token/acronym): ≤{COLLISION_ADJUDICABLE_MAX_CANDIDATES} candidates", file=sys.stderr)
        print("", file=sys.stderr)
        
        # Show rejection statistics by category
        if rejection_stats:
            print(f"\nRejection Statistics (alias-exact baseline):", file=sys.stderr)
            print(f"  Note: Alias-exact is intentionally incomplete. Rejections indicate:", file=sys.stderr)
            print(f"    - Policy blocks: Correctly blocked (need manual review/adjudication)", file=sys.stderr)
            print(f"    - Surface alignment: OCR/punctuation differences (surface_quality='approx')", file=sys.stderr)
            print(f"    - Case mismatch: Case policy enforcement (upper_only, titlecase_only, etc.)", file=sys.stderr)
            print(f"    - Context gate: requires_context not implemented yet", file=sys.stderr)
            print(f"    - Not eligible: Stopwords/single letters/small integers (not treated as collisions)", file=sys.stderr)
            print(f"    - Collision auto-resolved: Dominant candidate rules fired (Rule 0/1/2/3) - mentions were emitted", file=sys.stderr)
            print(f"      Rule 0: preferred_entity_id (from preferred_entity_id_map)", file=sys.stderr)
            print(f"      Rule 1: unique is_auto_match=true (case-lenient for place/org/person_full)", file=sys.stderr)
            print(f"      Rule 1c: unique is_auto_match=true (ignore case for non-covernames)", file=sys.stderr)
            print(f"      Rule 2: unique case_match (covernames only, or explicitly constraining)", file=sys.stderr)
            print(f"      Rule 3: safe dominance pairs (place>generic_word, etc.)", file=sys.stderr)
            print(f"      Filtered single candidate: exactly one candidate passes all policy filters", file=sys.stderr)
            print(f"    - Collision harmless: Too common/ambiguous (not enqueued, visible in stats)", file=sys.stderr)
            print(f"    - Collision logged: All collisions (including auto-resolved and low-value)", file=sys.stderr)
            print(f"    - Collision high-value too many: High-value but exceeds per-class threshold (log only)", file=sys.stderr)
            print(f"      Thresholds: covername={COLLISION_ADJUDICABLE_MAX_CANDIDATES.get('covername', 5)}, person_full={COLLISION_ADJUDICABLE_MAX_CANDIDATES.get('person_full', 2)}, org/place={COLLISION_ADJUDICABLE_MAX_CANDIDATES.get('org', 3)}", file=sys.stderr)
            print(f"    - Collision high-value enqueued: High-value collisions requiring human review (enqueued)", file=sys.stderr)
            print(f"  Missing entities should be handled via NER→review_queue workflow\n", file=sys.stderr)
            
            # Always print collision_auto_resolved summary (even if empty)
            auto_resolved_total = sum(rejection_stats.get('collision_auto_resolved', {}).values())
            print(f"  COLLISION AUTO RESOLVED TOTAL: {auto_resolved_total}", file=sys.stderr)
            if auto_resolved_total == 0:
                print(f"    (No collisions were auto-resolved - check dominance rules)", file=sys.stderr)
            print("", file=sys.stderr)
            
            for category, stats in rejection_stats.items():
                if stats:
                    print(f"  {category.upper().replace('_', ' ')}:", file=sys.stderr)
                    # Sort by count descending
                    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
                    # Show top 10 for collision_auto_resolved and collision_resolved_policy_blocked (important for auditing)
                    top_n = 10 if category in ('collision_auto_resolved', 'collision_resolved_policy_blocked', 'collision_dominance_none') else 15
                    for reason, count in sorted_stats[:top_n]:
                        print(f"    {reason}: {count}", file=sys.stderr)
                    if len(sorted_stats) > top_n:
                        print(f"    ... and {len(sorted_stats) - top_n} more", file=sys.stderr)
                    print("", file=sys.stderr)
        
        if args.dry_run:
            print("\nRun without --dry-run to actually insert mentions", file=sys.stderr)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
