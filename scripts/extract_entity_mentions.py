#!/usr/bin/env python3
"""
extract_entity_mentions.py [options]

DROP-IN REPLACEMENT

Fixes your immediate crash:
  psycopg2.errors.InFailedSqlTransaction: current transaction is aborted ...

Root cause:
  The old script probes for optional columns via SELECT <col> ... inside a transaction.
  When a column does not exist, Postgres raises UndefinedColumn and the transaction
  becomes "aborted" until ROLLBACK. The old code catches the exception but does not
  rollback, so the *next* query fails with InFailedSqlTransaction.

This replacement:
  - Detects optional columns via information_schema.columns (no failing queries).
  - Keeps your CLI flags and overall behavior: exact + partial + fuzzy, citation-based
    collision resolution, collision queue + summary CSV, idempotent inserts.

Notes:
  - It intentionally stays “precision-first”: unresolved high-value collisions are
    enqueued (logged + CSV) rather than guessed.
  - It preserves your method names: alias_exact / alias_partial / alias_fuzzy.

Usage:
  python scripts/extract_entity_mentions.py --collection venona --enable-partial --enable-fuzzy --summary-csv match_summary.csv --limit 10
"""

import os
import sys
import argparse
import re
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, DefaultDict, Any
from collections import defaultdict
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import execute_values, Json

# Aho-Corasick for fast multi-pattern matching
try:
    import ahocorasick
    _USE_AHOCORASICK = True
except ImportError:
    _USE_AHOCORASICK = False
    print("WARNING: pyahocorasick not installed. Using slow matching algorithm.", file=sys.stderr)
    print("  Install with: pip install pyahocorasick", file=sys.stderr)

from retrieval.entity_resolver import normalize_alias
from retrieval.ops import get_conn
from retrieval.surface_norm import normalize_surface
from retrieval.proposal_gating import compute_group_key, compute_candidate_set_hash

# Citation parsing for disambiguation
from concordance.validate_entity_mentions_from_citations import parse_citation_text, CitationLocation

# =============================================================================
# Policy constants
# =============================================================================

COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES = 5

COLLISION_ADJUDICABLE_MAX_CANDIDATES = {
    "covername": 5,
    "person_full": 2,
    "org": 3,
    "place": 3,
    "person_given": 1,
    "role_title": 3,
    "generic_word": 0,
    None: 3,
}

CONTEXT_GATE_FAILURE_ENQUEUE = False  # unchanged

STOPWORDS = {
    "to", "is", "as", "of", "in", "and", "or", "but", "the", "a", "an", "for", "on", "at", "by", "from", "with",
    "about", "into", "through", "during", "including", "against", "among", "throughout", "despite", "towards",
    "upon", "concerning", "up", "over", "under", "above", "below", "between", "within", "without", "across",
    "after", "before", "behind", "beyond", "near", "around", "along", "beside", "besides", "except", "plus",
    "minus", "per", "via", "versus", "vs", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "will", "would", "could", "should", "may", "might", "must",
    "can", "shall", "ought", "need", "dare", "used",
    # demonstratives / common non-entity determiners
    "this", "that", "these", "those",
}
SINGLE_LETTERS = set("abcdefghijklmnopqrstuvwxyz")

# Improvement A: Never-match list for months and document-structure words
# These should never be auto-matched at the mention extractor layer
NEVER_MATCH = {
    # Months
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # Document structure / metadata
    "ref", "reference", "page", "pages", "pm", "am", "find", "second", "minute", "hour",
    "first", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "section", "chapter", "part", "volume", "vol", "no", "number", "num",
}

# Improvement B: Known geographic/org tokens that should prefer place/org over covername
# unless codename context is present
KNOWN_GEOGRAPHIC_ORG_TOKENS = {
    # Major cities
    "moscow", "leningrad", "kiev", "washington", "london", "paris", "berlin", "tokyo",
    "new york", "los angeles", "chicago", "san francisco", "boston", "philadelphia",
    "cairo", "mexico", "mexico city", "rome", "vienna", "prague", "warsaw", "budapest",
    "stockholm", "helsinki", "oslo", "copenhagen", "amsterdam", "brussels", "zurich",
    "madrid", "lisbon", "athens", "istanbul", "shanghai", "beijing", "hong kong",
    # Countries / regions
    "ussr", "soviet", "soviet union", "russia", "united states", "usa", "uk", "britain", "england",
    "france", "germany", "japan", "china", "canada", "australia", "italy", "spain",
    "poland", "hungary", "czechoslovakia", "romania", "bulgaria", "yugoslavia",
    "egypt", "india", "korea", "vietnam", "cuba", "brazil", "argentina",
    # Organizations (common acronyms)
    "kgb", "gru", "mgb", "nkvd", "fbi", "cia", "nsa", "sis", "mi6",
    "cpusa", "comintern", "nsc", "oss", "state department", "treasury",
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class AliasInfo:
    entity_id: int
    original_alias: str
    alias_norm: str
    entity_type: str
    is_auto_match: bool
    min_chars: int
    match_case: str          # 'any' | 'case_sensitive' | 'upper_only' | 'titlecase_only'
    match_mode: str          # (kept for compatibility; not deeply used)
    is_numeric_entity: bool
    alias_class: Optional[str]
    allow_ambiguous_person_token: bool
    requires_context: Optional[str]
    alias_type: Optional[str] = None  # Phase 1-5: Track alias_type for derived aliases and filtering


@dataclass
class MatchCandidate:
    alias_info: AliasInfo
    surface: str
    surface_norm: str
    start_pos: int
    end_pos: int
    match_type: str          # 'exact' | 'partial' | 'fuzzy'
    match_confidence: float
    token_norm: str


# =============================================================================
# Utilities
# =============================================================================

def parse_chunk_id_range(range_str: str) -> Tuple[Optional[int], Optional[int]]:
    if ":" not in range_str:
        raise ValueError(f"Invalid chunk-id-range format: {range_str}. Use 'start:end'")
    a, b = range_str.split(":", 1)
    a = a.strip() or None
    b = b.strip() or None
    start = int(a) if a else None
    end = int(b) if b else None
    if start is not None and end is not None and start > end:
        raise ValueError(f"Invalid chunk-id-range: start ({start}) > end ({end})")
    return start, end


def get_chunks_query(
    conn,
    *,
    collection_slug: Optional[str] = None,
    document_id: Optional[int] = None,
    chunk_id_start: Optional[int] = None,
    chunk_id_end: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Tuple[int, str, Optional[int]]]:
    conditions, params = [], []
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
    q = f"""
        SELECT c.id AS chunk_id, c.text, cm.document_id
        FROM chunks c
        LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE {where_clause}
        ORDER BY c.id
        {limit_clause}
    """
    with conn.cursor() as cur:
        cur.execute(q, params)
        return cur.fetchall()


def tokenize_text(text: str) -> List[Tuple[int, int, str]]:
    # Capture dotted acronyms, hyphens, slashes, apostrophes similarly to your original.
    pattern = r"\b[\w']+(?:[./-][\w']+)*\b"
    out: List[Tuple[int, int, str]] = []
    for m in re.finditer(pattern, text):
        tok = m.group(0)
        if not re.search(r"[\w]", tok):
            continue
        out.append((m.start(), m.end(), tok))
    return out


def extract_surface_from_tokens(
    chunk_text: str,
    token_start: int,
    token_end: int,
    original_tokens: List[Tuple[int, int, str]],
) -> Tuple[str, str, str]:
    if token_start < len(chunk_text) and token_end <= len(chunk_text):
        surface = chunk_text[token_start:token_end].strip()

        # Include adjacent [...] if it appears to be part of the mention
        if token_start > 0 and chunk_text[token_start - 1] == "[":
            if token_end < len(chunk_text) and chunk_text[token_end] == "]":
                surface = chunk_text[token_start - 1: token_end + 1].strip()
            else:
                search_end = min(token_end + 20, len(chunk_text))
                close = chunk_text.find("]", token_end, search_end)
                if close != -1:
                    surface = chunk_text[token_start - 1: close + 1].strip()

        surface_norm = normalize_alias(surface)
        return surface, surface_norm, "exact"

    return "", "", "approx"


def check_case_match(surface: str, ai: AliasInfo, chunk_text: Optional[str] = None, start_pos: Optional[int] = None) -> bool:
    """
    Check if surface matches alias case requirements.
    
    Improvement 1A: Places and person tokens are case-insensitive by default.
    Improvement 1B: OCR lowercase mode detection - if chunk is overwhelmingly lowercase,
    treat titlecase_only as 'any' for place and person_* classes.
    """
    # Improvement 1A: Places and person tokens default to case-insensitive
    if ai.alias_class in ("place", "person_given") or ai.alias_type == "derived_last_name":
        # Override match_case to 'any' unless explicitly forced (covername keeps strict)
        if ai.match_case != "upper_only":  # Don't override upper_only (for acronyms)
            return surface.lower() == ai.original_alias.lower()
    
    # Improvement 1B: OCR lowercase mode detection
    if chunk_text is not None and start_pos is not None:
        if ai.alias_class in ("place", "person_given") or ai.alias_type == "derived_last_name":
            if ai.match_case == "titlecase_only":
                # Check if chunk is overwhelmingly lowercase (±200 char window)
                window_start = max(0, start_pos - 200)
                window_end = min(len(chunk_text), start_pos + 200)
                window_text = chunk_text[window_start:window_end]
                
                # Count uppercase vs lowercase letters
                upper_count = sum(1 for c in window_text if c.isupper())
                lower_count = sum(1 for c in window_text if c.islower())
                total_letters = upper_count + lower_count
                
                if total_letters > 0:
                    upper_ratio = upper_count / total_letters
                    # If < 2% uppercase, treat as OCR lowercase mode
                    if upper_ratio < 0.02:
                        return surface.lower() == ai.original_alias.lower()
    
    if ai.match_case == "any":
        return True
    if ai.match_case == "case_sensitive":
        # For person_given aliases with case_sensitive, be lenient: compare case-insensitively
        # This handles cases where alias is stored as "yakubovich" but text has "Yakubovich"
        if ai.alias_class == "person_given":
            return surface.lower() == ai.original_alias.lower()
        return surface == ai.original_alias
    if ai.match_case == "upper_only":
        return surface.isupper() or all(c.isupper() or not c.isalpha() for c in surface)
    if ai.match_case == "titlecase_only":
        if not surface:
            return False
        first = surface[0]
        rest = surface[1:]
        # For person_given, also allow case-insensitive comparison
        if ai.alias_class == "person_given":
            return surface.lower() == ai.original_alias.lower()
        return first.isupper() and (not rest or rest.islower() or all((not c.isalpha()) or c.islower() for c in rest))
    return True


def is_roman_numeral(token: str) -> bool:
    if not token:
        return False
    t = token.upper()
    valid = {"I", "V", "X", "L", "C", "D", "M"}
    return len(t) >= 2 and all(c in valid for c in t)


def check_codename_context(chunk_text: str, start_pos: int, end_pos: int, surface: str) -> bool:
    """
    Phase 3: Check if ambiguous codename has context signals:
    - Quoted/bracketed: "LINK", ['LINK'], [LINK], (LINK)
    - Covername markers nearby: cover name, codenamed, cryptonym, alias, aka
    - ALLCAPS
    """
    # Check if surface is quoted/bracketed (check ±5 chars around)
    context_window_start = max(0, start_pos - 5)
    context_window_end = min(len(chunk_text), end_pos + 5)
    context_window = chunk_text[context_window_start:context_window_end]
    
    # Check quotes/brackets
    if any(marker in context_window for marker in ['"', "'", '[', ']', '(', ')']):
        return True
    
    # Check ALLCAPS
    if surface.isupper() and len(surface) >= 3:
        return True
    
    # Check for codename markers (±60 chars)
    extended_start = max(0, start_pos - 60)
    extended_end = min(len(chunk_text), end_pos + 60)
    extended_context = chunk_text[extended_start:extended_end].lower()
    
    markers = [
        "cover name", "codenamed", "cryptonym", "alias", 
        "also known as", "aka", "code name", "codename"
    ]
    if any(marker in extended_context for marker in markers):
        return True
    
    return False


def check_place_context(chunk_text: str, start_pos: int, end_pos: int, surface: str) -> bool:
    """
    Improvement 3: Check if token appears in place-like context.
    Returns True if context suggests this is a location (not a codename).
    
    Signals:
    - Location prepositions nearby: in/at/from/to/near/around/into/throughout
    - Not ALLCAPS (unless it's clearly a place name)
    - Not quoted/bracketed (places are rarely quoted)
    """
    # Check if surface is ALLCAPS - likely codename, not place
    if surface.isupper() and len(surface) >= 3:
        return False
    
    # Check if quoted/bracketed - likely codename
    context_window_start = max(0, start_pos - 5)
    context_window_end = min(len(chunk_text), end_pos + 5)
    context_window = chunk_text[context_window_start:context_window_end]
    if any(marker in context_window for marker in ['"', "'", '[', ']', '(', ')']):
        return False
    
    # Check for location prepositions (±60 chars)
    extended_start = max(0, start_pos - 60)
    extended_end = min(len(chunk_text), end_pos + 60)
    extended_context = chunk_text[extended_start:extended_end].lower()
    
    location_prepositions = [
        " in ", " at ", " from ", " to ", " near ", " around ", 
        " into ", " throughout ", " within ", " outside ", " inside ",
        " of ", " on ", " upon "
    ]
    if any(prep in extended_context for prep in location_prepositions):
        return True
    
    return False


def check_surname_context(chunk_text: str, start_pos: int, end_pos: int, surface: str) -> bool:
    """
    Improvement 5: Check if surname appears in name-like context.
    Returns True if context suggests this is a person name (not random word).
    
    Signals:
    - Adjacent honorific/title: mr, mrs, ms, comrade, dr, doctor, colonel, general, etc.
    - Followed/preceded by an initial (J. Smith, Smith J.)
    - Followed by another capitalized token (when case exists)
    - Appears in a "name list" context (comma-separated roster lines)
    """
    extended_start = max(0, start_pos - 40)
    extended_end = min(len(chunk_text), end_pos + 40)
    extended_context = chunk_text[extended_start:extended_end]
    extended_context_lower = extended_context.lower()
    
    # Check for honorifics/titles before the surname
    honorifics = [
        "mr ", "mrs ", "ms ", "miss ", "comrade ", "dr ", "doctor ", 
        "colonel ", "general ", "captain ", "major ", "professor ", "prof ",
        "sir ", "lord ", "lady ", "president ", "minister ", "ambassador "
    ]
    for honorific in honorifics:
        # Check if honorific appears before the surname (within ±30 chars)
        honorific_pos = extended_context_lower.find(honorific)
        if honorific_pos >= 0 and honorific_pos < (start_pos - extended_start):
            return True
    
    # Check for initials (J. Smith or Smith J.)
    # Pattern: single letter followed by period, then surname, OR surname then single letter period
    before_context = chunk_text[max(0, start_pos - 10):start_pos]
    after_context = chunk_text[end_pos:min(len(chunk_text), end_pos + 10)]
    
    # Pattern: "J. Smith" or "Smith J."
    import re
    if re.search(r'\b[A-Z]\.\s*' + re.escape(surface), before_context, re.IGNORECASE):
        return True
    if re.search(re.escape(surface) + r'\s+[A-Z]\.', after_context, re.IGNORECASE):
        return True
    
    # Check if followed by another capitalized token (when case exists)
    # This suggests it's part of a name sequence
    words_after = after_context.split()
    if words_after:
        next_word = words_after[0].strip('.,;:!?')
        if next_word and next_word[0].isupper() and len(next_word) > 1:
            return True
    
    # Check for comma-separated name lists (roster context)
    # Pattern: "Smith, Jones, Brown" or "Smith, Jones and Brown"
    list_pattern = r'\b' + re.escape(surface) + r'\s*[,;]\s*[A-Z][a-z]+'
    if re.search(list_pattern, extended_context):
        return True
    
    return False


def is_candidate_eligible_for_matching(ai: AliasInfo, alias_norm: str, surface: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    toks = alias_norm.split()
    
    # ALL CAPS detection: if surface is all caps, assume codename/organization
    is_all_caps = False
    if surface:
        # Check if surface is all caps (all alphabetic chars are uppercase)
        is_all_caps = surface.isupper() or (len(surface) >= 3 and all(c.isupper() or not c.isalpha() for c in surface if c.isalpha()))
    
    if len(toks) != 1:
        return True, None
    token = toks[0].lower()

    # Improvement 2: Stop single-letter aliases from generating high-value collisions
    # Remove covername exception - single letters should not be matchable regardless of class
    if len(token) == 1:
        return False, "single_letter"

    # Improvement A: Hard stop for months and document-structure words
    if token in NEVER_MATCH:
        return False, "never_match"

    if token in STOPWORDS:
        if ai.alias_class == "covername" and ai.match_case == "upper_only":
            if surface is not None:
                is_upper = surface.isupper() or all(c.isupper() or not c.isalpha() for c in surface)
                if is_upper:
                    return True, None
            return False, "stopword_covername_not_uppercase"
        # Allow the rare case where a single-letter stopword-like token is used as an ALL-CAPS codename.
        # Important: do NOT let longer stopwords like FROM/WITH/TO/THE slip through just because
        # the surrounding text is in ALL CAPS headers.
        if is_all_caps and len(token) == 1 and ai.alias_class == "covername":
            return True, None
        return False, "stopword"

    # After stopword gating: ALL CAPS is a strong codename/org signal.
    # This handles cases like "NEIGHBOUR", "GRU", "KGB" etc.
    if is_all_caps:
        return True, None

    if token.isdigit() and len(token) <= 2 and (not ai.is_numeric_entity) and ai.alias_class != "covername":
        return False, "small_integer"

    if is_roman_numeral(token) and ai.alias_class != "covername" and (not ai.is_numeric_entity):
        return False, "roman_numeral_alone"

    return True, None


def is_alias_eligible_for_matching(alias_norm: str, alias_infos: List[AliasInfo], surface: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    for ai in alias_infos:
        ok, _ = is_candidate_eligible_for_matching(ai, alias_norm, surface)
        if ok:
            return True, None
    if alias_infos:
        _, reason = is_candidate_eligible_for_matching(alias_infos[0], alias_norm, surface)
        return False, reason
    return False, "no_candidates"


def is_purely_numeric(alias: str) -> bool:
    cleaned = re.sub(r"[.,\-]", "", alias.strip())
    return cleaned.isdigit()


def has_contentful_token(tokens: List[str], min_chars: int = 3) -> bool:
    stop = {"to", "of", "in", "on", "at", "for", "the", "a", "an", "and", "or", "but"}
    for t in tokens:
        tt = t.strip().lower()
        if len(tt) >= min_chars and tt not in stop:
            if tt and tt[0].isalpha():
                return True
    return False


# =============================================================================
# Levenshtein (fast if installed)
# =============================================================================

try:
    from Levenshtein import distance as _fast_levenshtein_distance
    _USE_FAST_LEVENSHTEIN = True
except ImportError:
    _USE_FAST_LEVENSHTEIN = False


def levenshtein_distance(s1: str, s2: str) -> int:
    if _USE_FAST_LEVENSHTEIN:
        return _fast_levenshtein_distance(s1, s2)

    # Pure Python fallback
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = previous_row[j + 1] + 1
            dele = current_row[j] + 1
            sub = previous_row[j] + (c1 != c2)
            current_row.append(min(ins, dele, sub))
        previous_row = current_row
    return previous_row[-1]


# =============================================================================
# DB batch helpers
# =============================================================================

def batch_load_chunk_metadata(cur, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    cur.execute("""
        SELECT
            cp.chunk_id,
            array_agg(p.id ORDER BY cp.span_order) as page_ids,
            array_agg(p.pdf_page_number ORDER BY cp.span_order)
                FILTER (WHERE p.pdf_page_number IS NOT NULL) as pdf_pages,
            MAX(cm.document_id) as document_id
        FROM chunk_pages cp
        JOIN pages p ON p.id = cp.page_id
        LEFT JOIN chunk_metadata cm ON cm.chunk_id = cp.chunk_id
        WHERE cp.chunk_id = ANY(%s)
        GROUP BY cp.chunk_id
    """, (chunk_ids,))
    out: Dict[int, Dict[str, Any]] = {}
    for chunk_id, page_ids, pdf_pages, document_id in cur.fetchall():
        out[chunk_id] = {
            "page_ids": page_ids or [],
            "pdf_pages": pdf_pages or [],
            "document_id": document_id,
        }
    for cid in chunk_ids:
        out.setdefault(cid, {"page_ids": [], "pdf_pages": [], "document_id": None})
    return out


def batch_load_document_cache(cur) -> Dict[str, int]:
    cur.execute("SELECT id, LOWER(source_name) FROM documents")
    return {name: did for (did, name) in cur.fetchall()}


def batch_load_entity_citations(cur, entity_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    if not entity_ids:
        return {}
    cur.execute("""
        SELECT entity_id, citation_text, page_list
        FROM entity_citations
        WHERE entity_id = ANY(%s)
    """, (entity_ids,))
    out: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for eid, citation_text, page_list in cur.fetchall():
        out[eid].append({"citation_text": citation_text, "page_list": page_list})
    return dict(out)


def get_canonical_names(cur, entity_ids: List[int]) -> Dict[int, str]:
    if not entity_ids:
        return {}
    cur.execute("SELECT id, canonical_name FROM entities WHERE id = ANY(%s)", (entity_ids,))
    return {i: n for (i, n) in cur.fetchall()}


def batch_load_entity_equivalence(cur, entity_ids: List[int]) -> Dict[int, Set[int]]:
    """
    Load bidirectional equivalence relationships from entity_links.
    
    Returns dict mapping entity_id -> set of equivalent entity_ids.
    Handles cases where ACHIEVEMENT and DOSTIZHENIE are alternate names for the same concept.
    
    NOTE: The shared aliases heuristic (self-join on entity_aliases) has been REMOVED
    due to poor performance. It was causing massive slowdowns on large alias sets.
    If needed, pre-compute equivalence relationships in a separate migration.
    """
    if not entity_ids:
        return {}
    
    equiv: DefaultDict[int, Set[int]] = defaultdict(set)
    
    # Check entity_links table for equivalence relationships
    # (This assumes entity_links has link_type='equivalent' or similar)
    try:
        cur.execute("""
            SELECT DISTINCT from_entity_id, to_entity_id
            FROM entity_links
            WHERE from_entity_id = ANY(%s) AND to_entity_id = ANY(%s)
               AND link_type IN ('equivalent', 'alternate_name', 'same_as')
        """, (entity_ids, entity_ids))
        for from_id, to_id in cur.fetchall():
            if from_id in entity_ids and to_id in entity_ids:
                equiv[from_id].add(to_id)
                equiv[to_id].add(from_id)
    except psycopg2.errors.UndefinedTable:
        pass  # entity_links table doesn't exist
    except Exception:
        pass  # Graceful degradation
    
    # NOTE: Removed expensive self-join on entity_aliases for shared aliases detection.
    # If you need this feature, pre-compute it in a migration table instead.
    
    return dict(equiv)


def are_candidates_duplicates(
    alias_infos: List[AliasInfo],
    canonical_names: Optional[Dict[int, str]] = None,
    cur=None,
    entity_equivalence: Optional[Dict[int, Set[int]]] = None,
) -> bool:
    if len(alias_infos) <= 1:
        return False
    if canonical_names is None:
        if cur is None:
            return False
        canonical_names = get_canonical_names(cur, [ai.entity_id for ai in alias_infos])
    # Check 1: Same canonical name (normalized)
    norms = set()
    for ai in alias_infos:
        name = canonical_names.get(ai.entity_id)
        if name:
            norms.add(normalize_alias(name))
    if len(norms) == 1:
        return True
    
    # Check 2: Entity equivalence relationships (alternate names for same concept)
    # Example: ACHIEVEMENT and DOSTIZHENIE are alternate names for the same entity
    if entity_equivalence and len(alias_infos) == 2:
        eid1, eid2 = alias_infos[0].entity_id, alias_infos[1].entity_id
        if eid2 in entity_equivalence.get(eid1, set()) or eid1 in entity_equivalence.get(eid2, set()):
            return True
    
    # Check 3: If all candidates are transitively equivalent (form a connected equivalence set)
    if entity_equivalence and len(alias_infos) > 1:
        entity_ids = {ai.entity_id for ai in alias_infos}
        # Build equivalence graph and check if all entities are in same connected component
        visited = set()
        def dfs(eid):
            if eid in visited:
                return
            visited.add(eid)
            for equiv_eid in entity_equivalence.get(eid, set()):
                if equiv_eid in entity_ids:
                    dfs(equiv_eid)
        
        # Start DFS from first entity
        if entity_ids:
            dfs(next(iter(entity_ids)))
            # If we visited all entities, they're all equivalent
            if visited == entity_ids:
                return True
    
    return False


def expand_page_ranges(pages: List[Tuple[int, Optional[int]]]) -> List[int]:
    expanded: List[int] = []
    for start, end in pages:
        if end is None:
            expanded.append(start)
        else:
            expanded.extend(list(range(start, end + 1)))
    return sorted(set(expanded))


def find_documents_for_citation(cur, citation_location: CitationLocation) -> List[int]:
    src = citation_location.source.lower().strip()
    cur.execute("""
        SELECT id
        FROM documents
        WHERE LOWER(source_name) LIKE %s
           OR LOWER(source_name) LIKE %s
    """, (f"%{src}%", f"{src}%"))
    return [row[0] for row in cur.fetchall()]


def resolve_collision_with_citations(
    alias_infos: List[AliasInfo],
    document_id: int,
    pdf_page_numbers: List[int],
    *,
    entity_citations_cache: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    document_cache: Optional[Dict[str, int]] = None,
    cur=None,
) -> Tuple[Optional[AliasInfo], float, Optional[str]]:
    if not pdf_page_numbers:
        return None, 0.0, None

    best: Tuple[Optional[AliasInfo], float, Optional[str]] = (None, 0.0, None)

    for ai in alias_infos:
        citations = (entity_citations_cache or {}).get(ai.entity_id, [])
        best_score = 0.0
        best_method = None

        for c in citations:
            ctext = c.get("citation_text")
            if not ctext:
                continue
            try:
                locs = parse_citation_text(ctext)
            except Exception:
                continue

            for loc in locs:
                # resolve doc ids
                citation_doc_ids: List[int] = []
                src_norm = loc.source.lower().strip()
                if document_cache is not None:
                    if src_norm in document_cache:
                        citation_doc_ids = [document_cache[src_norm]]
                    elif cur is not None:
                        citation_doc_ids = find_documents_for_citation(cur, loc)
                elif cur is not None:
                    citation_doc_ids = find_documents_for_citation(cur, loc)

                if document_id not in citation_doc_ids:
                    continue

                # Context-based resolution: use document/page as ground truth
                # If document matches, prefer this entity even without exact page match
                citation_pages = expand_page_ranges(loc.pages)
                
                if citation_pages:
                    # Page-level matching (preferred)
                    overlap = set(pdf_page_numbers) & set(citation_pages)
                    if overlap:
                        overlap_ratio = len(overlap) / max(1, len(pdf_page_numbers))
                        if len(overlap) == len(pdf_page_numbers) and len(overlap) == len(citation_pages):
                            score, method = 1.0, "citation_exact"
                        elif overlap_ratio >= 0.8:
                            score, method = 0.9, "citation_fuzzy"
                        else:
                            score, method = 0.7, "citation_fuzzy"
                    else:
                        # Document matches but pages don't - still prefer this entity
                        # (e.g., "NEIGHBOUR" is KGB in this document, GRU in another)
                        score, method = 0.6, "citation_document_match"
                else:
                    # No page info in citation, but document matches - still prefer
                    score, method = 0.6, "citation_document_match"

                if score > best_score:
                    best_score, best_method = score, method

        if best_score > best[1]:
            best = (ai, best_score, best_method)

    # Context-based resolution: prefer any document match as ground truth
    winner, score, method = best
    if winner is None:
        return None, 0.0, None

    # Find runner-up score
    scores = []
    for ai in alias_infos:
        if ai.entity_id == winner.entity_id:
            continue
        # pessimistic: treat others as 0.0 unless they also match
        # (we’d need to recompute; not worth it here).
        scores.append(0.0)
    # Lower threshold: accept document-level matches (0.6) as valid resolution
    # This handles cases like "NEIGHBOUR" = KGB in one doc, GRU in another
    # The document/page context is ground truth - trust it
    if score >= 0.6:
        return winner, score, method
    
    return None, 0.0, None


# =============================================================================
# Matching tiers: exact / partial / fuzzy
# =============================================================================

def build_partial_match_index(aliases_by_norm: Dict[str, List[AliasInfo]], min_token_len: int = 4) -> Dict[str, List[str]]:
    """
    Phase 5: Build partial match index ONLY for:
    - Derived surname aliases (alias_type='derived_last_name')
    - Derived acronym aliases (alias_type='derived_acronym')
    - Last token of person_full aliases (if >= 4 chars)
    """
    # Import GENERIC_WORDS_TO_EXCLUDE from concordance ingest (same list we use for filtering)
    # For now, use a local copy to avoid circular imports
    GENERIC_WORDS = {
        "i", "which", "their", "them", "what", "when", "there", "whom", "this", "that", "these", "those",
        "work", "city", "group", "pages", "serial", "known", "sent", "ref", "pm", "case", "time", "terms",
        "refer", "secret", "affairs", "minister", "given", "funds", "reply", "link", "telegraph", "working",
        "note", "general", "real", "financial", "doctor", "agent", "cases", "currency", "cutting", "ministry",
        "reference", "distant", "president", "chief", "cipher", "internal", "line", "also",
    }
    
    idx: DefaultDict[str, List[str]] = defaultdict(list)
    
    for alias_norm, alias_infos in aliases_by_norm.items():
        # Only index aliases that are auto-matchable
        if not any(ai.is_auto_match for ai in alias_infos):
            continue
        
        tokens = alias_norm.split()
        
        for ai in alias_infos:
            # Phase 5: Only index derived aliases or last tokens of person_full
            if ai.alias_type == "derived_last_name":
                # This is already a surname alias
                if len(tokens) == 1 and len(tokens[0]) >= min_token_len:
                    idx[tokens[0]].append(alias_norm)
            
            elif ai.alias_type == "derived_acronym":
                # Index acronyms (they're already single tokens)
                if len(alias_norm) >= 2 and len(alias_norm) <= 6:
                    idx[alias_norm].append(alias_norm)
            
            elif ai.alias_class == "person_full" and ai.is_auto_match:
                # Index last token only (surname)
                if len(tokens) > 1:
                    surname = tokens[-1]
                    if len(surname) >= min_token_len and surname.lower() not in GENERIC_WORDS:
                        idx[surname].append(alias_norm)
    
    return dict(idx)


def is_meaningful_partial_match(token_norm: str, alias_norm: str) -> bool:
    if len(token_norm) < 4:
        return False
    toks = alias_norm.split()
    for a in toks:
        if token_norm == a:
            return True
        if a.endswith(token_norm) and len(a) <= len(token_norm) + 2:
            return True
    return False


def find_partial_candidates(
    chunk_text: str,
    original_tokens: List[Tuple[int, int, str]],
    aliases_by_norm: Dict[str, List[AliasInfo]],
    partial_index: Dict[str, List[str]],
    min_partial_len: int = 4,
) -> List[MatchCandidate]:
    out: List[MatchCandidate] = []
    for s, e, tok in original_tokens:
        tnorm = normalize_alias(tok)
        if len(tnorm) < min_partial_len:
            continue
        for alias_norm in partial_index.get(tnorm, []):
            if not is_meaningful_partial_match(tnorm, alias_norm):
                continue
            tokens_in_alias = alias_norm.split()
            is_last_name = len(tokens_in_alias) > 1 and tnorm == tokens_in_alias[-1]
            conf = 0.7 if is_last_name else 0.5
            for ai in aliases_by_norm.get(alias_norm, []):
                out.append(MatchCandidate(ai, tok, tnorm, s, e, "partial", conf, tnorm))
    return out


def find_partial_candidates_optimized(
    chunk_text: str,
    normalized_tokens: List[Tuple[int, int, str, str]],  # (start, end, tok, norm)
    aliases_by_norm: Dict[str, List[AliasInfo]],
    partial_index: Dict[str, List[str]],
    min_partial_len: int = 4,
) -> List[MatchCandidate]:
    """Optimized version that uses pre-normalized tokens."""
    out: List[MatchCandidate] = []
    for s, e, tok, tnorm in normalized_tokens:
        if len(tnorm) < min_partial_len:
            continue
        for alias_norm in partial_index.get(tnorm, []):
            if not is_meaningful_partial_match(tnorm, alias_norm):
                continue
            tokens_in_alias = alias_norm.split()
            is_last_name = len(tokens_in_alias) > 1 and tnorm == tokens_in_alias[-1]
            conf = 0.7 if is_last_name else 0.5
            for ai in aliases_by_norm.get(alias_norm, []):
                out.append(MatchCandidate(ai, tok, tnorm, s, e, "partial", conf, tnorm))
    return out


def find_fuzzy_candidates(
    original_tokens: List[Tuple[int, int, str]],
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    *,
    max_edit_distance: int = 2,
    min_len: int = 4,
) -> List[MatchCandidate]:
    aliases_by_len: DefaultDict[int, List[str]] = defaultdict(list)
    for a in alias_norm_set:
        aliases_by_len[len(a)].append(a)

    out: List[MatchCandidate] = []
    for s, e, tok in original_tokens:
        tnorm = normalize_alias(tok)
        if len(tnorm) < min_len:
            continue
        tlen = len(tnorm)
        for L in range(max(1, tlen - max_edit_distance), tlen + max_edit_distance + 1):
            for alias_norm in aliases_by_len.get(L, []):
                if abs(len(alias_norm) - tlen) > max_edit_distance:
                    continue
                d = levenshtein_distance(tnorm, alias_norm)
                if 0 < d <= max_edit_distance:
                    conf = 1.0 - (d * 0.2)  # 0.8 for 1, 0.6 for 2
                    for ai in aliases_by_norm.get(alias_norm, []):
                        out.append(MatchCandidate(ai, tok, tnorm, s, e, "fuzzy", conf, tnorm))
    return out


def find_fuzzy_candidates_optimized(
    normalized_tokens: List[Tuple[int, int, str, str]],  # (start, end, tok, norm)
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    *,
    max_edit_distance: int = 2,
    min_len: int = 4,
) -> List[MatchCandidate]:
    """Optimized version that uses pre-normalized tokens."""
    aliases_by_len: DefaultDict[int, List[str]] = defaultdict(list)
    for a in alias_norm_set:
        aliases_by_len[len(a)].append(a)

    out: List[MatchCandidate] = []
    for s, e, tok, tnorm in normalized_tokens:
        if len(tnorm) < min_len:
            continue
        tlen = len(tnorm)
        for L in range(max(1, tlen - max_edit_distance), tlen + max_edit_distance + 1):
            for alias_norm in aliases_by_len.get(L, []):
                if abs(len(alias_norm) - tlen) > max_edit_distance:
                    continue
                d = levenshtein_distance(tnorm, alias_norm)
                if 0 < d <= max_edit_distance:
                    conf = 1.0 - (d * 0.2)  # 0.8 for 1, 0.6 for 2
                    for ai in aliases_by_norm.get(alias_norm, []):
                        out.append(MatchCandidate(ai, tok, tnorm, s, e, "fuzzy", conf, tnorm))
    return out


# =============================================================================
# Collision policy
# =============================================================================

def find_dominant_candidate(
    alias_norm: str,
    alias_infos: List[AliasInfo],
    surface: str,
    preferred_entity_id_map: Optional[Dict[Tuple[Optional[str], str], int]] = None,
    ban_entity_map: Optional[Dict[Tuple[Optional[str], str], Set[int]]] = None,
    ban_surface_map: Optional[Dict[Tuple[Optional[str], str], bool]] = None,
    scope: Optional[str] = None,
    chunk_text: Optional[str] = None,
    start_pos: Optional[int] = None,
    end_pos: Optional[int] = None,
    canonical_names_cache: Optional[Dict[int, str]] = None,  # NEW: for better geo/org resolution
) -> Tuple[Optional[AliasInfo], Optional[str], Optional[str]]:
    if len(alias_infos) <= 1:
        return (alias_infos[0] if alias_infos else None, None, None)

    # Rule 0a: Apply bans FIRST (before prefer)
    # This removes banned entities from consideration entirely
    alias_infos_filtered, ban_reason = apply_bans_to_candidates(
        alias_norm, alias_infos, ban_entity_map, ban_surface_map, scope
    )
    
    if ban_reason:
        # If surface is entirely banned, return None
        if "surface_banned" in ban_reason:
            return None, "rule0a_ban", ban_reason
        # If some entities were banned, continue with filtered list
        alias_infos = alias_infos_filtered
        if len(alias_infos) == 0:
            return None, "rule0a_ban", f"all_candidates_banned({ban_reason})"
        if len(alias_infos) == 1:
            return alias_infos[0], "rule0a_ban", f"single_after_ban({ban_reason})"

    # Rule 0b: preferred mapping (scoped then global)
    if preferred_entity_id_map:
        pref = None
        if (scope, alias_norm) in preferred_entity_id_map:
            pref = preferred_entity_id_map[(scope, alias_norm)]
        elif (None, alias_norm) in preferred_entity_id_map:
            pref = preferred_entity_id_map[(None, alias_norm)]
        if pref is not None:
            cands = [ai for ai in alias_infos if ai.entity_id == pref]
            if len(cands) == 1:
                return cands[0], "rule0b_prefer", f"preferred_entity_id={pref}"

    # Rule 1: only one candidate is eligible + auto-match + case (strict for covername/person_given)
    eligible: List[AliasInfo] = []
    for ai in alias_infos:
        if not ai.is_auto_match:
            continue
        ok, _ = is_candidate_eligible_for_matching(ai, alias_norm, surface)
        if not ok:
            continue
        if ai.alias_class == "covername":
            # Covernames: always enforce case matching (even if match_case='any', default is upper_only)
            if not check_case_match(surface, ai, chunk_text, start_pos):
                continue
        elif ai.alias_class == "person_given":
            # Person given names: respect match_case setting
            # If match_case='any', check_case_match returns True (case-insensitive)
            # If match_case='case_sensitive' or 'titlecase_only', enforce it
            if not check_case_match(surface, ai, chunk_text, start_pos):
                continue
        else:
            # Other alias classes: only enforce if match_case is not 'any'
            if ai.match_case in ("case_sensitive", "upper_only", "titlecase_only") and not check_case_match(surface, ai, chunk_text, start_pos):
                continue
        eligible.append(ai)
    if len(eligible) == 1:
        return eligible[0], "rule1", f"unique_filtered_candidate(total={len(alias_infos)})"

    # Rule 2: safe dominance pairs
    safe_pairs = [
        ("place", "generic_word"),
        ("org", "generic_word"),
        ("person_full", "generic_word"),
        ("covername", "generic_word"),
        ("place", "person_given"),
        ("org", "person_given"),
    ]
    for dom, sub in safe_pairs:
        doms = [ai for ai in alias_infos if ai.alias_class == dom]
        subs = [ai for ai in alias_infos if ai.alias_class == sub]
        if len(doms) == 1 and len(subs) >= 1:
            others = [ai for ai in alias_infos if ai.alias_class not in (dom, sub) and ai.alias_class is not None]
            if not others:
                return doms[0], "rule2", f"{dom}>{sub}(dom=1,sub={len(subs)},total={len(alias_infos)})"

    # Improvement B: Deterministic dominance rule for known geographic/org tokens
    # If alias_norm is a known geographic/org token, prefer the entity whose
    # canonical name best matches the alias (simplest/most direct match)
    alias_norm_lower = alias_norm.lower()
    if alias_norm_lower in KNOWN_GEOGRAPHIC_ORG_TOKENS and canonical_names_cache:
        # Score by how well the canonical name matches the alias
        # Prefer entities where canonical_name == alias (the actual place/org)
        scored_cands: List[Tuple[float, int, AliasInfo]] = []  # (score, -entity_id for stable sort, ai)
        for ai in alias_infos:
            canonical = canonical_names_cache.get(ai.entity_id, "")
            canonical_lower = canonical.lower().strip()
            canonical_norm = normalize_alias(canonical_lower)
            
            # Score based on how well canonical name matches alias
            score = 0.0
            
            # Exact match: canonical name IS the alias (e.g., "Moscow" for moscow)
            if canonical_norm == alias_norm_lower:
                score = 1.0
            # Canonical starts with alias (e.g., "Moscow, Russia" for moscow)
            elif canonical_norm.startswith(alias_norm_lower):
                score = 0.9 - (len(canonical) / 1000.0)  # Shorter is better
            # Alias is in canonical (e.g., "1943 Moscow conference" contains moscow)
            elif alias_norm_lower in canonical_norm:
                score = 0.5 - (len(canonical) / 1000.0)  # Shorter is better
            # Derived last name aliases get very low score
            elif ai.alias_type == "derived_last_name":
                score = 0.1
            else:
                score = 0.3
            
            scored_cands.append((score, -ai.entity_id, ai))  # Negative entity_id for stable tiebreaker (lower ID wins)
        
        if scored_cands:
            scored_cands.sort(key=lambda x: (-x[0], x[1]))  # Highest score first, then lowest entity_id
            best_score, _, best_cand = scored_cands[0]
            second_score = scored_cands[1][0] if len(scored_cands) > 1 else 0.0
            
            # Auto-resolve if best score is >= 0.9 (clear canonical match)
            if best_score >= 0.9:
                canonical = canonical_names_cache.get(best_cand.entity_id, "?")
                return best_cand, "rule_known_geo", f"canonical='{canonical[:30]}...'(score={best_score:.2f})"
            # Or if there's a significant gap between best and second
            elif best_score >= 0.5 and best_score - second_score >= 0.3:
                canonical = canonical_names_cache.get(best_cand.entity_id, "?")
                return best_cand, "rule_known_geo", f"canonical='{canonical[:30]}...'(gap={best_score - second_score:.2f})"
    
    # Fallback to class-based rules
    if chunk_text is not None and start_pos is not None and end_pos is not None:
        if alias_norm_lower in KNOWN_GEOGRAPHIC_ORG_TOKENS:
            place_cands = [ai for ai in alias_infos if ai.alias_class == "place"]
            org_cands = [ai for ai in alias_infos if ai.alias_class == "org"]
            covername_cands = [ai for ai in alias_infos if ai.alias_class == "covername"]
            
            # Check if codename context is present
            has_codename_context = check_codename_context(chunk_text, start_pos, end_pos, surface)
            
            # If codename context is present, allow covername to win (if it's the only covername)
            if has_codename_context and len(covername_cands) == 1:
                # Only prefer covername if there's no place/org candidate, or if covername is clearly intended
                if not place_cands and not org_cands:
                    return covername_cands[0], "rule2_known_token", f"covername>known_geo_org(context=codename)"
            
            # Otherwise, prefer place/org over covername for known geographic/org tokens
            if len(place_cands) == 1 and len(covername_cands) >= 1:
                if not has_codename_context:
                    return place_cands[0], "rule2_known_token", f"place>covername(known_geo_token={alias_norm_lower})"
            
            if len(org_cands) == 1 and len(covername_cands) >= 1:
                if not has_codename_context:
                    return org_cands[0], "rule2_known_token", f"org>covername(known_geo_token={alias_norm_lower})"

    # Improvement 3: Context-based dominance pairs (place vs covername, org vs covername, person_given vs covername)
    if chunk_text is not None and start_pos is not None and end_pos is not None:
        # Place vs covername
        place_cands = [ai for ai in alias_infos if ai.alias_class == "place"]
        covername_cands = [ai for ai in alias_infos if ai.alias_class == "covername"]
        if len(place_cands) == 1 and len(covername_cands) >= 1:
            # Choose place if context suggests location
            if check_place_context(chunk_text, start_pos, end_pos, surface):
                return place_cands[0], "rule2_context", f"place>covername(context=place_like)"
            # Choose covername if context suggests codename
            if check_codename_context(chunk_text, start_pos, end_pos, surface):
                if len(covername_cands) == 1:
                    return covername_cands[0], "rule2_context", f"covername>place(context=codename_like)"
        
        # Org vs covername
        org_cands = [ai for ai in alias_infos if ai.alias_class == "org"]
        if len(org_cands) == 1 and len(covername_cands) >= 1:
            # Choose org if it's acronym-like and appears in org context (not quoted/ALLCAPS)
            if not surface.isupper() and not any(marker in chunk_text[max(0, start_pos-5):min(len(chunk_text), end_pos+5)] for marker in ['"', "'", '[', ']']):
                # Check for org-like context (the USSR, in USSR, etc.)
                extended_start = max(0, start_pos - 60)
                extended_end = min(len(chunk_text), end_pos + 60)
                extended_context = chunk_text[extended_start:extended_end].lower()
                if any(marker in extended_context for marker in [" the ", " of ", " in ", " at "]):
                    return org_cands[0], "rule2_context", f"org>covername(context=org_like)"
            # Choose covername if quoted/ALLCAPS + codename markers
            if check_codename_context(chunk_text, start_pos, end_pos, surface):
                if len(covername_cands) == 1:
                    return covername_cands[0], "rule2_context", f"covername>org(context=codename_like)"
        
        # Person_given vs covername (e.g., "viktor")
        person_given_cands = [ai for ai in alias_infos if ai.alias_class == "person_given"]
        if len(person_given_cands) == 1 and len(covername_cands) >= 1:
            # Choose person if not ALLCAPS and not quoted
            if not surface.isupper() and not any(marker in chunk_text[max(0, start_pos-5):min(len(chunk_text), end_pos+5)] for marker in ['"', "'", '[', ']']):
                return person_given_cands[0], "rule2_context", f"person_given>covername(context=name_like)"
            # Choose covername if quoted/ALLCAPS + codename markers
            if check_codename_context(chunk_text, start_pos, end_pos, surface):
                if len(covername_cands) == 1:
                    return covername_cands[0], "rule2_context", f"covername>person_given(context=codename_like)"

    return None, None, None


def is_collision_high_value(alias_norm: str, alias_infos: List[AliasInfo], surface: str) -> Tuple[bool, bool, Optional[str]]:
    hv = {"covername", "person_full", "org", "place"}
    has_hv = any(ai.alias_class in hv for ai in alias_infos)
    if not has_hv:
        return False, False, None

    ok, _ = is_alias_eligible_for_matching(alias_norm, alias_infos, surface=surface)
    if not ok:
        return False, False, None

    cand_n = len(alias_infos)
    thresholds = []
    classes = []
    for ai in alias_infos:
        if ai.alias_class in hv:
            thresholds.append(COLLISION_ADJUDICABLE_MAX_CANDIDATES.get(ai.alias_class, COLLISION_ADJUDICABLE_MAX_CANDIDATES[None]))
            if ai.alias_class not in classes:
                classes.append(ai.alias_class)
    max_thr = min(thresholds) if thresholds else COLLISION_ADJUDICABLE_MAX_CANDIDATES[None]
    info = f"threshold={max_thr} (classes={','.join(classes) if classes else 'default'}, candidates={cand_n})"
    if cand_n > max_thr:
        return True, False, info

    has_strict = any(ai.alias_class in ("covername", "person_given") for ai in alias_infos)
    if has_strict:
        if not any(check_case_match(surface, ai) for ai in alias_infos if ai.alias_class in ("covername", "person_given")):
            return True, False, f"{info}, case_mismatch"
    else:
        if any(ai.match_case in ("case_sensitive", "upper_only", "titlecase_only") for ai in alias_infos):
            if not any(check_case_match(surface, ai) for ai in alias_infos):
                return True, False, f"{info}, case_mismatch"

    return True, True, info


def is_collision_harmless(
    alias_norm: str,
    alias_infos: List[AliasInfo],
    canonical_names: Optional[Dict[int, str]] = None,
    cur=None,
    entity_equivalence: Optional[Dict[int, Set[int]]] = None,
) -> bool:
    if len(alias_infos) <= 1:
        return False

    ok, _ = is_alias_eligible_for_matching(alias_norm, alias_infos)
    if not ok:
        return True

    # Duplicates/equivalents are NOT harmless (data quality problem)
    if are_candidates_duplicates(alias_infos, canonical_names, cur, entity_equivalence):
        return False

    # Covernames and full names: never harmless
    if any(ai.alias_class == "covername" for ai in alias_infos):
        return False
    if any(ai.alias_class == "person_full" for ai in alias_infos):
        return False

    # Generic words: harmless
    if any(ai.alias_class == "generic_word" for ai in alias_infos):
        return True

    # Given names with too many candidates: harmless
    toks = alias_norm.split()
    if len(toks) == 1:
        if any(ai.alias_class == "person_given" and not ai.allow_ambiguous_person_token for ai in alias_infos):
            if len(alias_infos) > 10:
                return True

    return False


# =============================================================================
# Alias loader (FIXED: no aborted transactions)
# =============================================================================

def load_all_aliases(
    conn,
    *,
    collection_slug: Optional[str] = None,
    concordance_source_slug: Optional[str] = None,
) -> Tuple[Dict[str, List[AliasInfo]], Set[str]]:
    start_time = time.time()

    with conn.cursor() as cur:
        # Detect columns safely (NO failing SELECTs).
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'entity_aliases'
        """)
        ea_cols = {r[0] for r in cur.fetchall()}

        # Also detect older min_token_len if present
        has_min_chars = "min_chars" in ea_cols
        has_min_token_len = "min_token_len" in ea_cols

        select_cols = [
            "ea.entity_id",
            "ea.alias",
            "ea.alias_norm",
            "e.entity_type",
        ]

        if "is_auto_match" in ea_cols:
            select_cols.append("COALESCE(ea.is_auto_match, true) AS is_auto_match")
        else:
            select_cols.append("true AS is_auto_match")

        if has_min_chars:
            select_cols.append("COALESCE(ea.min_chars, 1) AS min_chars")
        elif has_min_token_len:
            select_cols.append("COALESCE(ea.min_token_len, 1) AS min_chars")
        else:
            select_cols.append("1 AS min_chars")

        select_cols.append("COALESCE(ea.match_case, 'any') AS match_case")
        select_cols.append("COALESCE(ea.match_mode, 'token') AS match_mode")

        if "is_numeric_entity" in ea_cols:
            select_cols.append("COALESCE(ea.is_numeric_entity, false) AS is_numeric_entity")
        else:
            select_cols.append("false AS is_numeric_entity")

        if "alias_class" in ea_cols:
            select_cols.append("ea.alias_class")
        else:
            select_cols.append("NULL AS alias_class")

        if "allow_ambiguous_person_token" in ea_cols:
            select_cols.append("COALESCE(ea.allow_ambiguous_person_token, false) AS allow_ambiguous_person_token")
        else:
            select_cols.append("false AS allow_ambiguous_person_token")

        if "requires_context" in ea_cols:
            select_cols.append("ea.requires_context")
        else:
            select_cols.append("NULL AS requires_context")

        if "alias_type" in ea_cols:
            select_cols.append("ea.alias_type")
        else:
            select_cols.append("NULL AS alias_type")

        # Optional: limit to a single concordance source slug (prevents cross-ingest replicates).
        #
        # We filter by ENTITY (not just alias row) using an EXISTS clause, so we keep the
        # entity’s matchable aliases together as a coherent universe, but only for entities
        # that are connected to the requested concordance source.
        source_filter_sql = ""
        source_filter_params: List[Any] = []
        if concordance_source_slug:
            source_filter_sql = """
                AND EXISTS (
                    SELECT 1
                    FROM entity_aliases ea_filter
                    JOIN concordance_entries ce_filter ON ce_filter.id = ea_filter.entry_id
                    JOIN concordance_sources cs_filter ON cs_filter.id = ce_filter.source_id
                    WHERE ea_filter.entity_id = ea.entity_id
                      AND cs_filter.slug = %s
                )
            """
            source_filter_params = [concordance_source_slug]

        # Identify entities with "unidentified/unknown/..." aliases via alias_norm
        search_terms = [
            "unidentified", "unknown", "unnamed", "intelligence source",
            "intelligence officer", "source/agent", "officer/agent",
            "soviet intelligence", "intelligence agent"
        ]
        search_norms = [normalize_alias(t) for t in search_terms]
        # NOTE: if source_filter_sql is active, it adds an extra %s placeholder (slug),
        # so we must pass that parameter too.
        params: List[Any] = [search_norms]
        params.extend(source_filter_params)
        cur.execute(
            """
            SELECT DISTINCT ea.entity_id
            FROM entity_aliases ea
            WHERE ea.alias_norm = ANY(%s)
            """
            + source_filter_sql
            + """
            """,
            params,
        )
        entities_with_unidentified = {row[0] for row in cur.fetchall()}

        q = f"""
            SELECT {', '.join(select_cols)}
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.entity_id
            WHERE ea.is_matchable = true
            {source_filter_sql}
            ORDER BY ea.entity_id, ea.id
        """
        cur.execute(q, source_filter_params)
        rows = cur.fetchall()

    aliases_by_norm: DefaultDict[str, List[AliasInfo]] = defaultdict(list)
    alias_norm_set: Set[str] = set()

    for row in rows:
        entity_id = row[0]
        alias = row[1]
        alias_norm = row[2]
        entity_type = row[3]
        is_auto_match = bool(row[4])
        min_chars = int(row[5])
        match_case = row[6] or "any"
        match_mode = row[7] or "token"
        is_numeric_entity = bool(row[8])
        alias_class = row[9]
        allow_ambiguous_person_token = bool(row[10])
        requires_context = row[11]
        alias_type = row[12] if len(row) > 12 else None

        # Improvement 1A: Coerce match_case to 'any' for places and person tokens
        # (unless explicitly upper_only for acronyms)
        if alias_class in ("place", "person_given") or alias_type == "derived_last_name":
            if match_case != "upper_only":  # Don't override acronyms
                match_case = "any"

        tokens = alias_norm.split()

        # Fallback alias_class
        if alias_class is None:
            if entity_type in ("cover_name", "covername"):
                alias_class = "covername"
            elif entity_type == "person":
                alias_class = "person_given" if len(tokens) == 1 else "person_full"
            elif entity_type == "org":
                alias_class = "org"
            elif entity_type == "place":
                alias_class = "place"
            elif entity_type in ("other", "topic", "role"):
                alias_class = "generic_word"
        else:
            # Normalize common alias_class variants coming from different ingests/exports.
            # The rest of the pipeline expects canonical values like "covername", not "cover_name".
            if isinstance(alias_class, str):
                cls = alias_class.strip().lower().replace("-", "_").replace(" ", "_")
                if cls in ("cover_name", "covername"):
                    alias_class = "covername"
                elif cls in ("person_full", "personfull"):
                    alias_class = "person_full"
                elif cls in ("person_given", "persongiven", "given_name"):
                    alias_class = "person_given"

        # Entity type gate: only allow auto-match for high-value entity types
        # (person, covername, org, place). Other types (other, topic, role) are excluded
        # unless they're ALLCAPS codenames (handled separately below).
        HIGH_VALUE_ENTITY_TYPES = {"person", "cover_name", "covername", "org", "place"}
        if entity_type not in HIGH_VALUE_ENTITY_TYPES:
            # For non-high-value types, only allow if it's an ALLCAPS codename/acronym
            # (might be legitimate codename that wasn't classified correctly)
            is_allcaps_codename = (
                alias.isupper() and 
                len(alias) >= 2 and 
                len(alias) <= 10 and 
                alias.replace("-", "").replace("/", "").isalnum()
            )
            if not is_allcaps_codename:
                is_auto_match = False
        
        # Phase 3: Mark ambiguous covernames with requires_context
        AMBIGUOUS_COVERNAMES = {
            "link", "achievement", "master", "group", "general", "president",
            "information", "foreign", "neighbour", "neighbors", "neighbours",
            "chief", "doctor", "agent", "officer", "director", "secretary"
        }
        requires_context = None
        if alias_class == "covername" and alias_norm.lower() in AMBIGUOUS_COVERNAMES:
            requires_context = "codename_like"

        # generic_word: disable auto-match unless it looks like a short ALLCAPS codeword
        if alias_class == "generic_word" and is_auto_match:
            is_codeword_like = (alias == alias.upper() and len(alias) <= 6 and alias.isalpha())
            if not is_codeword_like:
                is_auto_match = False

        # person_given single-token: allow by default (important for last-name-only references,
        # including partial matching), but keep a minimal safety guard for very short tokens.
        #
        # Rationale:
        # - We already block stopwords / single letters / small integers elsewhere.
        # - Single-token person mentions are often surnames in-text; disabling them breaks recall.
        # - Collisions with many candidates are handled downstream (harmless / enqueue / citations).
        if entity_type == "person" and alias_class == "person_given" and len(tokens) == 1 and not allow_ambiguous_person_token:
            if len(tokens[0]) < 4:
                is_auto_match = False

        # covername case policy (LOOSENED)
        #
        # Previously we forced covernames to upper_only by default. That made OCR / sentence-case
        # variants (and bracket variants) fail with "case_mismatch" even when the alias clearly
        # refers to the same entity (e.g. ACHIEVEMENT/DOSTIZHENIE).
        #
        # New behavior: respect the DB value. If DB leaves match_case as 'any', we keep it 'any'
        # (case-insensitive). This reduces "case_mismatch" unresolved collisions for covernames.

        # Improvement C: Re-scope the "unidentified override"
        # Only apply to aliases that are meaningful (multi-token, or ALLCAPS, or high-value alias_class),
        # never to generic single tokens
        if (not is_auto_match) and (entity_id in entities_with_unidentified):
            tokens = alias_norm.split()
            is_multi_token = len(tokens) > 1
            is_allcaps = alias.isupper() and len(alias) >= 2
            is_high_value_class = alias_class in ("covername", "person_full", "org", "place")
            
            # Only re-enable if the alias itself is meaningful
            if is_multi_token or is_allcaps or is_high_value_class:
                is_auto_match = True

        # Ensure requires_context is set (may have been set above for ambiguous covernames)
        if requires_context is None:
            # Check if DB has requires_context set
            if "requires_context" in ea_cols and len(row) > len(select_cols):
                requires_context = row[select_cols.index("requires_context") if "requires_context" in select_cols else -1]
            else:
                requires_context = None
        
        ai = AliasInfo(
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
            alias_type=alias_type,
        )
        aliases_by_norm[alias_norm].append(ai)
        alias_norm_set.add(alias_norm)

    # Phase 4.2: Apply DF-based rules (per document, but we'll use max DF across all docs for now)
    # In practice, we'll check DF per document during matching, but for initial filtering
    # we use the maximum DF across all documents
    with conn.cursor() as cur:
        # Check if alias_stats table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'alias_stats'
            )
        """)
        has_alias_stats = cur.fetchone()[0]
        
        if has_alias_stats and alias_norm_set:
            cur.execute("""
                SELECT alias_norm, MAX(df_percent) as max_df_percent
                FROM alias_stats
                WHERE alias_norm = ANY(%s)
                GROUP BY alias_norm
            """, (list(alias_norm_set),))
            
            df_map = {row[0]: row[1] for row in cur.fetchall()}
            
            # Improvement 4: Tighten DF rules - apply DF suppression primarily to generic tokens,
            # not real names/places/persons/orgs
            AMBIGUOUS_COVERNAMES = {
                "link", "achievement", "master", "group", "general", "president",
                "information", "foreign", "neighbour", "neighbors", "neighbours",
                "chief", "doctor", "agent", "officer", "director", "secretary"
            }
            
            for alias_norm, alias_infos in aliases_by_norm.items():
                max_df_percent = df_map.get(alias_norm, 0)
                
                for ai in alias_infos:
                    tokens = alias_norm.split()
                    
                    # Only apply DF suppression to:
                    # - generic_word
                    # - role_title (if we have that class)
                    # - ambiguous covernames
                    should_apply_df = (
                        ai.alias_class == "generic_word" or
                        ai.alias_class == "role_title" or
                        (ai.alias_class == "covername" and alias_norm.lower() in AMBIGUOUS_COVERNAMES)
                    )
                    
                    # Do NOT DF-suppress:
                    # - place
                    # - person_given / derived surnames
                    # - person_full
                    # - org acronyms we care about
                    if not should_apply_df:
                        continue
                    
                    # Rule: if DF > 0.5% and single-token and not ALLCAPS-acronym
                    if max_df_percent > 0.5 and len(tokens) == 1:
                        # Whitelist: known valuable acronyms
                        whitelist = {"kgb", "gru", "mgb", "nkvd", "ussr", "fbi", "cia", "nsa", "sis", "mi6"}
                        if alias_norm.lower() not in whitelist:
                            # Check if it's an ALLCAPS codename/acronym
                            if not (ai.alias_class == "covername" and ai.original_alias.isupper() and len(ai.original_alias) <= 6):
                                ai.is_auto_match = False
                    
                    # Rule: if DF > 2%, disable matching entirely (unless whitelisted)
                    if max_df_percent > 2.0:
                        whitelist = {"kgb", "gru", "mgb", "nkvd", "ussr", "fbi", "cia", "nsa", "sis", "mi6"}
                        if alias_norm.lower() not in whitelist:
                            ai.is_auto_match = False
                    
                    # For derived surnames: only suppress if DF is extreme AND surname is generic
                    if ai.alias_type == "derived_last_name":
                        generic_surnames = {"brown", "king", "smith", "jones", "white", "black", "green", "miller", "davis", "wilson"}
                        if alias_norm.lower() in generic_surnames and max_df_percent > 5.0:
                            ai.is_auto_match = False

    load_time = time.time() - start_time
    if load_time > 1.0:
        print(f"  [PERF] load_all_aliases took {load_time:.2f}s", file=sys.stderr)

    return dict(aliases_by_norm), alias_norm_set


def load_entities_with_definitions(conn) -> Set[int]:
    """
    Pre-load all entity IDs that have definition aliases.
    This avoids querying the database for each unresolved collision.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT entity_id
            FROM entity_aliases
            WHERE alias_type = 'definition'
        """)
        return {row[0] for row in cur.fetchall()}


# =============================================================================
# Aho-Corasick Fast Matching
# =============================================================================

def build_ahocorasick_automaton(alias_norm_set: Set[str]) -> Any:
    """
    Build Aho-Corasick automaton from normalized aliases.
    This is O(total_alias_characters) and should be done ONCE at startup.
    """
    if not _USE_AHOCORASICK:
        return None
    
    A = ahocorasick.Automaton()
    for alias_norm in alias_norm_set:
        # Store the alias_norm as the value (we'll look up alias_infos later)
        A.add_word(alias_norm, alias_norm)
    A.make_automaton()
    return A


def find_matches_for_chunk_fast(
    chunk_text: str,
    *,
    chunk_id: int,
    document_id: int,
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    ac_automaton: Any,  # Aho-Corasick automaton
    enable_partial: bool,
    enable_fuzzy: bool,
    partial_index: Optional[Dict[str, List[str]]],
    chunk_metadata_cache: Dict[int, Dict[str, Any]],
    entity_citations_cache: Dict[int, List[Dict[str, Any]]],
    entity_equivalence_cache: Dict[int, Set[int]],
    document_cache: Dict[str, int],
    preferred_entity_id_map: Optional[Dict[Tuple[Optional[str], str], int]],
    ban_entity_map: Optional[Dict[Tuple[Optional[str], str], Set[int]]],
    ban_surface_map: Optional[Dict[Tuple[Optional[str], str], bool]],
    scope: Optional[str],
    rejection_stats: Dict[str, Dict[str, int]],
    collision_queue: List[Dict[str, Any]],
    entities_with_definitions: Set[int],
    canonical_names_cache: Dict[int, str],
    batch_cursor,
    conn,
) -> Tuple[List[Tuple[int, str, str, str, Optional[int], Optional[int], str, float]], Dict[str, int]]:
    """
    Fast matching using Aho-Corasick algorithm.
    Complexity: O(text_length + number_of_matches) instead of O(tokens * 5).
    """
    if not chunk_text or not chunk_text.strip():
        return [], {"exact": 0, "partial": 0, "fuzzy": 0}
    
    # Step 1: Normalize the entire chunk text (same normalization as aliases)
    # We need to track position mapping from normalized -> original
    chunk_text_norm = normalize_alias(chunk_text)
    
    # Step 2: Find all matches using Aho-Corasick (O(text_length + matches))
    # CRITICAL: Filter to only word-boundary matches to avoid substring matches like "al" inside "special"
    raw_matches: List[Tuple[int, int, str]] = []  # (start_pos, end_pos, alias_norm)
    for end_pos, alias_norm in ac_automaton.iter(chunk_text_norm):
        start_pos = end_pos - len(alias_norm) + 1
        end_pos_excl = end_pos + 1
        
        # Check word boundaries in normalized text
        # Start boundary: either at position 0 or preceded by non-alphanumeric
        start_ok = (start_pos == 0) or (not chunk_text_norm[start_pos - 1].isalnum())
        # End boundary: either at end or followed by non-alphanumeric
        end_ok = (end_pos_excl >= len(chunk_text_norm)) or (not chunk_text_norm[end_pos_excl].isalnum() if end_pos_excl < len(chunk_text_norm) else True)
        
        if start_ok and end_ok:
            raw_matches.append((start_pos, end_pos_excl, alias_norm))
    
    if not raw_matches:
        return [], {"exact": 0, "partial": 0, "fuzzy": 0}
    
    # Step 3: Handle overlaps (prefer longest match at each position)
    # Sort by start position, then by length descending (longest first)
    match_spans: List[Tuple[int, int, str]] = raw_matches
    
    # Sort by start position, then by length descending (longest first)
    match_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    
    # Remove overlapping matches (keep longest at each position)
    filtered_matches: List[Tuple[int, int, str]] = []
    last_end = -1
    for start, end, alias_norm in match_spans:
        if start >= last_end:
            filtered_matches.append((start, end, alias_norm))
            last_end = end
        elif end > last_end:
            # This match starts inside previous but extends further
            # Check if it's a longer match that should replace
            if filtered_matches and start == filtered_matches[-1][0]:
                # Same start - keep longer one
                if end - start > filtered_matches[-1][1] - filtered_matches[-1][0]:
                    filtered_matches[-1] = (start, end, alias_norm)
                    last_end = end
    
    # Step 4: Map normalized positions back to original text
    # Build a simple mapping by finding the surface in original text
    matches_out: List[Tuple[int, str, str, str, Optional[int], Optional[int], str, float]] = []
    match_type_stats = {"exact": 0, "partial": 0, "fuzzy": 0}
    matched_positions: Set[Tuple[int, int]] = set()
    
    for norm_start, norm_end, alias_norm in filtered_matches:
        alias_infos = aliases_by_norm.get(alias_norm, [])
        if not alias_infos:
            continue
        
        # FAST surface extraction: use position mapping from normalized to original
        # The normalized text has same character count (lowercase + punctuation removed)
        # Use ratio-based estimation for start/end positions
        ratio_start = norm_start / max(1, len(chunk_text_norm))
        ratio_end = norm_end / max(1, len(chunk_text_norm))
        est_start = int(ratio_start * len(chunk_text))
        est_end = int(ratio_end * len(chunk_text))
        
        # Find word boundaries around estimated position
        # Expand to word boundaries
        while est_start > 0 and chunk_text[est_start - 1].isalnum():
            est_start -= 1
        while est_end < len(chunk_text) and chunk_text[est_end - 1].isalnum():
            est_end += 1
        
        surface = chunk_text[est_start:est_end].strip()
        surface_start = est_start
        surface_end = est_end
        
        # Validate: check if normalized surface matches alias_norm
        if normalize_alias(surface) != alias_norm:
            # Fallback: use original alias
            surface = alias_infos[0].original_alias
            surface_start = None
            surface_end = None
            surface_quality = "approx"
        else:
            surface_quality = "exact"
        
        # Check if already matched at this position
        pos_key = (surface_start, surface_end) if surface_start is not None else (norm_start, norm_end)
        if pos_key in matched_positions:
            continue
        
        surface_norm = alias_norm
        
        # Eligibility check
        ok, reason = is_alias_eligible_for_matching(alias_norm, alias_infos, surface=surface)
        if not ok:
            rejection_stats.setdefault("not_eligible", {})
            key = f"{alias_norm} ({reason})"
            rejection_stats["not_eligible"][key] = rejection_stats["not_eligible"].get(key, 0) + 1
            continue
        
        # Collision resolution (same as original)
        resolved: Optional[AliasInfo] = None
        is_collision = len(alias_infos) > 1
        
        if is_collision:
            entity_ids_for_collision = [ai.entity_id for ai in alias_infos]
            canonical = {eid: canonical_names_cache.get(eid, "") for eid in entity_ids_for_collision}
            
            # Try duplicate resolution
            try:
                if are_candidates_duplicates(alias_infos, canonical, batch_cursor, entity_equivalence_cache):
                    am = [ai for ai in alias_infos if ai.is_auto_match]
                    resolved = am[0] if am else alias_infos[0]
                    rejection_stats.setdefault("collision_auto_resolved", {})
                    k = f"{alias_norm} (duplicate_resolved)"
                    rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
            except Exception:
                pass
            
            # Citation resolution
            if resolved is None:
                meta = chunk_metadata_cache.get(chunk_id, {})
                pdf_pages = meta.get("pdf_pages", [])
                doc_id_for_chunk = meta.get("document_id") or document_id
                if pdf_pages and doc_id_for_chunk:
                    try:
                        cit_ai, cit_conf, cit_method = resolve_collision_with_citations(
                            alias_infos, doc_id_for_chunk, pdf_pages,
                            entity_citations_cache=entity_citations_cache,
                            document_cache=document_cache,
                            cur=batch_cursor
                        )
                        if cit_ai:
                            resolved = cit_ai
                            rejection_stats.setdefault("collision_auto_resolved", {})
                            k = f"{alias_norm} (citation_match_{cit_method})"
                            rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
                    except Exception:
                        pass
            
            # Dominance resolution (includes ban logic as Rule 0a)
            if resolved is None:
                dom, rule, detail = find_dominant_candidate(
                    alias_norm, alias_infos, surface,
                    preferred_entity_id_map=preferred_entity_id_map,
                    ban_entity_map=ban_entity_map,
                    ban_surface_map=ban_surface_map,
                    scope=scope,
                    chunk_text=chunk_text,
                    start_pos=surface_start,
                    end_pos=surface_end,
                    canonical_names_cache=canonical_names_cache,
                )
                if dom is not None:
                    resolved = dom
                    rejection_stats.setdefault("collision_auto_resolved", {})
                    k = f"{alias_norm} (dominance_{rule})"
                    rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
                elif rule and "ban" in rule:
                    # Surface or entities were banned, skip this match
                    rejection_stats.setdefault("banned", {})
                    rejection_stats["banned"][f"{alias_norm} ({rule})"] = rejection_stats["banned"].get(f"{alias_norm} ({rule})", 0) + 1
                    continue
            
            # Definition preference for single-token
            if resolved is None and len(alias_norm.split()) == 1:
                def_cands = [ai for ai in alias_infos if ai.entity_id in entities_with_definitions]
                if len(def_cands) == 1:
                    resolved = def_cands[0]
                    rejection_stats.setdefault("collision_auto_resolved", {})
                    k = f"{alias_norm} (definition_preference)"
                    rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
            
            # Still unresolved - enqueue for review
            if resolved is None:
                is_harmless = False
                try:
                    is_harmless = is_collision_harmless(alias_norm, alias_infos, canonical, batch_cursor, entity_equivalence_cache)
                except Exception:
                    pass
                
                hv, enqueue, info = is_collision_high_value(alias_norm, alias_infos, surface)
                collision_category = "harmless" if is_harmless else ("high_value_enqueued" if hv and enqueue else "dominance_none")
                
                ctx_s = max(0, (surface_start or 0) - 100)
                ctx_e = min(len(chunk_text), (surface_end or len(chunk_text)) + 100)
                collision_queue.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "surface": surface,
                    "alias_norm": alias_norm,
                    "surface_norm": surface_norm,
                    "start_char": surface_start,
                    "end_char": surface_end,
                    "context_excerpt": chunk_text[ctx_s:ctx_e],
                    "candidate_entity_ids": [ai.entity_id for ai in alias_infos],
                    "method": "alias_exact_collision",
                    "method_version": "v1",
                    "collision_metadata": {
                        "is_high_value": hv,
                        "is_harmless": is_harmless,
                        "collision_category": collision_category,
                    },
                })
                continue
        else:
            # Single candidate
            resolved = alias_infos[0]
        
        if resolved is None:
            continue
        
        # Policy and case checks
        ai = resolved
        if not ai.is_auto_match:
            rejection_stats.setdefault("policy", {})
            rejection_stats["policy"][f"{alias_norm} (auto_match=false)"] = rejection_stats["policy"].get(f"{alias_norm} (auto_match=false)", 0) + 1
            continue
        
        if not check_case_match(surface, ai, chunk_text, surface_start):
            rejection_stats.setdefault("case_mismatch", {})
            rejection_stats["case_mismatch"][ai.original_alias] = rejection_stats["case_mismatch"].get(ai.original_alias, 0) + 1
            continue
        
        # Context check for covernames
        if ai.requires_context == "codename_like":
            if not check_codename_context(chunk_text, surface_start or 0, surface_end or len(surface), surface):
                rejection_stats.setdefault("context_gate", {})
                rejection_stats["context_gate"][f"{ai.original_alias} (requires_context)"] = rejection_stats["context_gate"].get(f"{ai.original_alias} (requires_context)", 0) + 1
                continue
        
        # Success! Add match
        matched_positions.add(pos_key)
        matches_out.append((
            ai.entity_id,
            surface,
            surface_norm,
            surface_quality,
            surface_start,
            surface_end,
            "exact",
            1.0
        ))
        match_type_stats["exact"] += 1
    
    return matches_out, match_type_stats


# =============================================================================
# Core matching per chunk (original - fallback if Aho-Corasick not available)
# =============================================================================

def find_matches_for_chunk(
    chunk_text: str,
    *,
    chunk_id: int,
    document_id: int,
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    enable_partial: bool,
    enable_fuzzy: bool,
    partial_index: Optional[Dict[str, List[str]]],
    chunk_metadata_cache: Dict[int, Dict[str, Any]],
    entity_citations_cache: Dict[int, List[Dict[str, Any]]],
    entity_equivalence_cache: Dict[int, Set[int]],
    document_cache: Dict[str, int],
    preferred_entity_id_map: Optional[Dict[Tuple[Optional[str], str], int]],
    ban_entity_map: Optional[Dict[Tuple[Optional[str], str], Set[int]]],
    ban_surface_map: Optional[Dict[Tuple[Optional[str], str], bool]],
    scope: Optional[str],
    rejection_stats: Dict[str, Dict[str, int]],
    collision_queue: List[Dict[str, Any]],
    entities_with_definitions: Set[int],
    canonical_names_cache: Dict[int, str],
    batch_cursor,  # Reusable cursor for the batch
    conn,
) -> Tuple[List[Tuple[int, str, str, str, Optional[int], Optional[int], str, float]], Dict[str, int]]:
    """
    Returns:
      matches: tuples (entity_id, surface, surface_norm, surface_quality, start_char, end_char, match_type, confidence)
      match_type_stats: counts
    """
    original_tokens = tokenize_text(chunk_text)
    if not original_tokens:
        return [], {"exact": 0, "partial": 0, "fuzzy": 0}

    # OPTIMIZATION: Pre-normalize all tokens once (avoids repeated normalize_alias calls)
    normalized_tokens: List[Tuple[int, int, str, str]] = [
        (start, end, tok, normalize_alias(tok)) for (start, end, tok) in original_tokens
    ]
    
    # SIMPLE OPTIMIZATION: Pre-compute single-token norms for fast lookup
    # The set lookup `ngram_norm in alias_norm_set` is the hot path
    single_token_norms = [norm for (_, _, _, norm) in normalized_tokens]

    matches_out: List[Tuple[int, str, str, str, Optional[int], Optional[int], str, float]] = []
    matched_positions: Set[Tuple[int, int]] = set()        # any accepted match
    exact_positions: Set[Tuple[int, int]] = set()          # accepted exact matches
    match_type_stats = {"exact": 0, "partial": 0, "fuzzy": 0}

    # -------------------------
    # Tier 1: Exact (n-gram, longest-first)
    # -------------------------
    i = 0
    while i < len(original_tokens):
        matched_here = False
        # Improvement: Largest complete match priority
        # Check n-grams longest-first, but break early if we find a match at max length
        # This ensures we prefer "Soviet Union" over "Soviet" without checking all shorter variants
        best_match = None
        best_n = 0
        max_n = min(5, len(original_tokens) - i)
        
        for n in range(max_n, 0, -1):
            span_tokens = normalized_tokens[i:i + n]
            s = span_tokens[0][0]
            e = span_tokens[-1][1]
            if (s, e) in matched_positions:
                continue

            # Build n-gram directly (avoid function call overhead)
            if n == 1:
                ngram_norm = single_token_norms[i]
            else:
                ngram_norm = " ".join(single_token_norms[i:i + n])
            
            if ngram_norm not in alias_norm_set:
                # Single-word expansion: if single-word codename/person name, try expanding window
                if n == 1 and i + 1 < len(normalized_tokens):
                    # Try expanding to include next word(s) for better context matching
                    for expand_n in range(2, min(4, len(normalized_tokens) - i + 1)):
                        expanded_norm = " ".join(single_token_norms[i:i + expand_n])
                        if expanded_norm in alias_norm_set:
                            # Found expanded match - use it instead
                            span_tokens = normalized_tokens[i:i + expand_n]
                            ngram_norm = expanded_norm
                            s = normalized_tokens[i][0]
                            e = normalized_tokens[i + expand_n - 1][1]
                            n = expand_n
                            break
                    else:
                        continue  # No expanded match found
                else:
                    continue

            alias_infos = aliases_by_norm.get(ngram_norm, [])
            if not alias_infos:
                continue
            
            # Found a match - since we're checking longest-first, this is the longest possible match
            # Use it immediately (no need to check shorter variants)
            best_match = (span_tokens, s, e, ngram_norm, alias_infos)
            best_n = n
            break  # Early exit: we found the longest match
        
        # Use the longest match found
        if best_match is None:
            i += 1
            continue
        
        span_tokens, s, e, ngram_norm, alias_infos = best_match

        # OPTIMIZATION: Use pre-normalized ngram_norm instead of re-extracting
        surface, _, surface_quality = extract_surface_from_tokens(chunk_text, s, e, original_tokens)
        # OPTIMIZATION: Use pre-computed normalized form (faster than re-normalizing)
        surface_norm = ngram_norm  # Use the pre-computed normalized form
        if not surface:
            surface = alias_infos[0].original_alias
            surface_norm = ngram_norm
            surface_quality = "approx"

        ok, reason = is_alias_eligible_for_matching(ngram_norm, alias_infos, surface=surface)
        if not ok:
            rejection_stats.setdefault("not_eligible", {})
            key = f"{ngram_norm} ({reason})"
            rejection_stats["not_eligible"][key] = rejection_stats["not_eligible"].get(key, 0) + 1
            i += best_n  # Skip past this position
            continue

        # Improvement 5: Surname context requirements for derived surnames
        # Only match derived surnames if they appear in name-like context
        derived_surname_cands = [ai for ai in alias_infos if ai.alias_type == "derived_last_name"]
        if derived_surname_cands and len(alias_infos) == len(derived_surname_cands):
            # All candidates are derived surnames - require context
            if not check_surname_context(chunk_text, s, e, surface):
                rejection_stats.setdefault("surname_context_required", {})
                rejection_stats["surname_context_required"][ngram_norm] = rejection_stats["surname_context_required"].get(ngram_norm, 0) + 1
                i += best_n  # Skip past this position
                continue
        elif derived_surname_cands:
            # Mixed candidates - only accept derived surname if context is good
            non_derived = [ai for ai in alias_infos if ai.alias_type != "derived_last_name"]
            if non_derived:
                # Prefer non-derived if derived doesn't have context
                if not check_surname_context(chunk_text, s, e, surface):
                    # Remove derived surnames from consideration
                    alias_infos = non_derived
                    if not alias_infos:
                        rejection_stats.setdefault("surname_context_required", {})
                        rejection_stats["surname_context_required"][ngram_norm] = rejection_stats["surname_context_required"].get(ngram_norm, 0) + 1
                        i += best_n  # Skip past this position
                        continue

        resolved: Optional[AliasInfo] = None
        is_collision = len(alias_infos) > 1

        # Collision resolution
        if is_collision:
            # duplicates check - use pre-loaded cache instead of querying
            entity_ids_for_collision = [ai.entity_id for ai in alias_infos]
            canonical = {eid: canonical_names_cache.get(eid, "") for eid in entity_ids_for_collision}
            try:
                # Use batch cursor instead of creating new one
                if are_candidates_duplicates(alias_infos, canonical, batch_cursor, entity_equivalence_cache):
                        # Prefer citation match; else first auto_match; else first.
                        meta = chunk_metadata_cache.get(chunk_id, {})
                        pdf_pages = meta.get("pdf_pages", [])
                        doc_id_for_chunk = meta.get("document_id") or document_id
                        if pdf_pages and doc_id_for_chunk:
                            # Use batch cursor instead of creating new one
                            cit_ai, cit_conf, cit_method = resolve_collision_with_citations(
                                alias_infos, doc_id_for_chunk, pdf_pages,
                                entity_citations_cache=entity_citations_cache,
                                document_cache=document_cache,
                                cur=batch_cursor  # Reuse batch cursor
                            )
                            if cit_ai:
                                resolved = cit_ai
                                rejection_stats.setdefault("collision_auto_resolved", {})
                                k = f"{ngram_norm} (duplicate_resolved_{cit_method})"
                                rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
                        if resolved is None:
                            am = [ai for ai in alias_infos if ai.is_auto_match]
                            resolved = am[0] if am else alias_infos[0]
                            rejection_stats.setdefault("collision_auto_resolved", {})
                            k = f"{ngram_norm} (duplicate_resolved)"
                            rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
            except Exception:
                pass

            # citation resolution
            if resolved is None:
                meta = chunk_metadata_cache.get(chunk_id, {})
                pdf_pages = meta.get("pdf_pages", [])
                doc_id_for_chunk = meta.get("document_id") or document_id
                if pdf_pages and doc_id_for_chunk:
                    try:
                        # Use batch cursor instead of creating new one
                        cit_ai, cit_conf, cit_method = resolve_collision_with_citations(
                            alias_infos, doc_id_for_chunk, pdf_pages,
                            entity_citations_cache=entity_citations_cache,
                            document_cache=document_cache,
                            cur=batch_cursor  # Reuse batch cursor
                        )
                        if cit_ai:
                            resolved = cit_ai
                            rejection_stats.setdefault("collision_auto_resolved", {})
                            k = f"{ngram_norm} (citation_match_{cit_method})"
                            rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
                    except Exception:
                        pass

            # dominance (includes ban logic as Rule 0a)
            if resolved is None:
                dom, rule, detail = find_dominant_candidate(
                    ngram_norm, alias_infos, surface,
                    preferred_entity_id_map=preferred_entity_id_map,
                    ban_entity_map=ban_entity_map,
                    ban_surface_map=ban_surface_map,
                    scope=scope,
                    chunk_text=chunk_text,
                    start_pos=s,
                    end_pos=e,
                    canonical_names_cache=canonical_names_cache,
                )
                if dom is not None:
                    resolved = dom
                    rejection_stats.setdefault("collision_auto_resolved", {})
                    k = f"{ngram_norm} (dominance_{rule}{(' ' + detail) if detail else ''})"
                    rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
                elif rule and "ban" in rule:
                    # Surface or entities were banned, skip this match
                    rejection_stats.setdefault("banned", {})
                    rejection_stats["banned"][f"{ngram_norm} ({rule})"] = rejection_stats["banned"].get(f"{ngram_norm} ({rule})", 0) + 1
                    continue
                else:
                    rejection_stats.setdefault("collision_dominance_none", {})
                    rejection_stats["collision_dominance_none"][ngram_norm] = rejection_stats["collision_dominance_none"].get(ngram_norm, 0) + 1

            # unresolved: enqueue ALL for review (including harmless ones)
            if resolved is None:
                # Use pre-loaded canonical names cache instead of querying
                canonical_for_check = canonical if canonical else {eid: canonical_names_cache.get(eid, "") for eid in entity_ids_for_collision}
                is_harmless = False
                try:
                    # Use batch cursor (function doesn't need it if canonical_names provided, but safer)
                    is_harmless = is_collision_harmless(ngram_norm, alias_infos, canonical_for_check, batch_cursor, entity_equivalence_cache)
                    if is_harmless:
                        rejection_stats.setdefault("collision_harmless", {})
                        rejection_stats["collision_harmless"][ngram_norm] = rejection_stats["collision_harmless"].get(ngram_norm, 0) + 1
                except Exception:
                    pass

                # Improvement: For single-token unresolved matches, prefer entities with definitions
                # Use pre-loaded cache instead of querying database
                # NOTE: This happens AFTER logging collision_dominance_none, so stats may show
                # collisions that get resolved here. This is intentional - we want to track
                # that these collisions occurred even if they were auto-resolved.
                if len(ngram_norm.split()) == 1:
                    # Check if any candidate has a definition (using pre-loaded cache)
                    def_cands = [ai for ai in alias_infos if ai.entity_id in entities_with_definitions]
                    if len(def_cands) == 1:
                        resolved = def_cands[0]
                        rejection_stats.setdefault("collision_auto_resolved", {})
                        k = f"{ngram_norm} (definition_preference)"
                        rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
                        # Don't enqueue if resolved by definition preference
                        i += best_n
                        continue
                
                # Only enqueue if still unresolved after all resolution attempts
                if resolved is None:
                    hv, enqueue, info = is_collision_high_value(ngram_norm, alias_infos, surface)
                    rejection_stats.setdefault("collision_logged", {})
                    rejection_stats["collision_logged"][ngram_norm] = rejection_stats["collision_logged"].get(ngram_norm, 0) + 1

                    # Determine collision category for metadata
                    if is_harmless:
                        collision_category = "harmless"
                    elif hv and enqueue:
                        collision_category = "high_value_enqueued"
                    elif hv:
                        collision_category = "high_value_too_many"
                    else:
                        collision_category = "dominance_none"

                    # Enqueue ALL unresolved collisions for review (including harmless ones)
                    # Store metadata about collision type for filtering/prioritization
                    ctx_s = max(0, s - 100)
                    ctx_e = min(len(chunk_text), e + 100)
                    collision_queue.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "surface": surface,
                        "alias_norm": ngram_norm,
                        "surface_norm": surface_norm,
                        "start_char": s,
                        "end_char": e,
                        "context_excerpt": chunk_text[ctx_s:ctx_e],
                        "candidate_entity_ids": [ai.entity_id for ai in alias_infos],
                        "method": "alias_exact_collision",
                        "method_version": "v1",
                        "collision_metadata": {
                            "is_high_value": hv,
                            "is_harmless": is_harmless,
                            "collision_category": collision_category,
                            "enqueue_flag": enqueue,
                            "info": info,
                        },
                    })
                    
                    # Log statistics (for reporting)
                    if hv and enqueue:
                        rejection_stats.setdefault("collision_high_value_enqueued", {})
                        key = f"{ngram_norm}{(' (' + info + ')') if info else ''}"
                        rejection_stats["collision_high_value_enqueued"][key] = rejection_stats["collision_high_value_enqueued"].get(key, 0) + 1
                    elif hv:
                        rejection_stats.setdefault("collision_high_value_too_many", {})
                        key = f"{ngram_norm}{(' (' + info + ')') if info else ''}"
                        rejection_stats["collision_high_value_too_many"][key] = rejection_stats["collision_high_value_too_many"].get(key, 0) + 1
                    # Note: collision_dominance_none and collision_harmless are already logged above

                    i += best_n  # Skip past this position
                    continue  # unresolved collision -> no mention emitted

        else:
            resolved = alias_infos[0]

        # policy checks after resolution (for both collision and non-collision cases)
        ai = resolved
        
        # ALL CAPS override: if surface OR original_alias is ALL CAPS, override is_auto_match=False
        # ALL CAPS is a strong signal it's a codename/organization, regardless of alias_class or entity_type
        is_all_caps_surface = surface and (surface.isupper() or (len(surface) >= 3 and all(c.isupper() or not c.isalpha() for c in surface if c.isalpha())))
        is_all_caps_alias = ai.original_alias and (ai.original_alias.isupper() or (len(ai.original_alias) >= 3 and all(c.isupper() or not c.isalpha() for c in ai.original_alias if c.isalpha())))
        
        # Override if ANY of these conditions:
        # 1. Surface is ALL CAPS (strongest signal - text itself is ALL CAPS)
        # 2. Original alias is ALL CAPS (database alias is ALL CAPS = codename)
        # 3. Entity is covername/org AND surface is ALL CAPS
        should_override = False
        if is_all_caps_surface:
            # Surface text is ALL CAPS - always override (text itself signals codename)
            should_override = True
        elif is_all_caps_alias:
            # Original alias is ALL CAPS - treat as codename regardless of other factors
            should_override = True
        elif (ai.entity_type in ("cover_name", "covername", "org") or 
              ai.alias_class in ("covername", "org")):
            # Entity is covername/org - allow override even if not ALL CAPS
            # (but prefer ALL CAPS cases above)
            pass  # Don't override for non-ALL-CAPS covernames/orgs
        
        if should_override:
            # Override: ALL CAPS covernames/orgs should be auto-matched even if DB says no
            # This handles cases like "NEIGHBOUR", "GRU", "KGB" that might have is_auto_match=False
            # but are clearly codenames when written in ALL CAPS
            ai = AliasInfo(
                entity_id=ai.entity_id,
                original_alias=ai.original_alias,
                alias_norm=ai.alias_norm,
                entity_type=ai.entity_type,
                is_auto_match=True,  # Override to True
                min_chars=ai.min_chars,
                match_case=ai.match_case,
                match_mode=ai.match_mode,
                is_numeric_entity=ai.is_numeric_entity,
                alias_class=ai.alias_class,
                allow_ambiguous_person_token=ai.allow_ambiguous_person_token,
                requires_context=ai.requires_context,
                alias_type=ai.alias_type,
            )
        
        # Phase 3: Real requires_context implementation
        if ai.requires_context == "codename_like":
            if not check_codename_context(chunk_text, s, e, surface):
                rejection_stats.setdefault("context_gate", {})
                key = f"{ai.original_alias} (requires_context={ai.requires_context})"
                rejection_stats["context_gate"][key] = rejection_stats["context_gate"].get(key, 0) + 1
                if CONTEXT_GATE_FAILURE_ENQUEUE:
                    collision_queue.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "surface": surface,
                        "alias_norm": ngram_norm,
                        "surface_norm": surface_norm,
                        "start_char": s,
                        "end_char": e,
                        "context_excerpt": chunk_text[max(0, s - 100):min(len(chunk_text), e + 100)],
                        "candidate_entity_ids": [ai.entity_id],
                        "method": "context_gate_failed",
                        "method_version": "v1",
                    })
                i += best_n  # Skip past this position
                continue
        elif ai.requires_context:
            # Other requires_context types - reject for now (can be extended later)
            rejection_stats.setdefault("context_gate", {})
            key = f"{ai.original_alias} (requires_context={ai.requires_context})"
            rejection_stats["context_gate"][key] = rejection_stats["context_gate"].get(key, 0) + 1
            if CONTEXT_GATE_FAILURE_ENQUEUE:
                collision_queue.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "surface": surface,
                    "alias_norm": ngram_norm,
                    "surface_norm": surface_norm,
                    "start_char": s,
                    "end_char": e,
                    "context_excerpt": chunk_text[max(0, s - 100):min(len(chunk_text), e + 100)],
                    "candidate_entity_ids": [ai.entity_id],
                    "method": "context_gate_failed",
                    "method_version": "v1",
                })
            i += best_n  # Skip past this position
            continue

        if not ai.is_auto_match:
            rejection_stats.setdefault("policy", {})
            key = f"{ai.original_alias} (auto_match disabled)"
            rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
            i += best_n  # Skip past this position
            continue

        if is_purely_numeric(ai.original_alias) and not ai.is_numeric_entity:
            rejection_stats.setdefault("policy", {})
            key = f"{ai.original_alias} (purely numeric)"
            rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
            i += best_n  # Skip past this position
            continue

        toks = ngram_norm.split()
        if len(toks) == 1:
            if len(toks[0]) < ai.min_chars:
                rejection_stats.setdefault("policy", {})
                key = f"{ai.original_alias} (chars < {ai.min_chars})"
                rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
                i += best_n  # Skip past this position
                continue
        else:
            if max(len(t) for t in toks) < ai.min_chars:
                rejection_stats.setdefault("policy", {})
                key = f"{ai.original_alias} (phrase: no token ≥ {ai.min_chars})"
                rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
                i += best_n  # Skip past this position
                continue
            if not has_contentful_token(toks, min_chars=3):
                rejection_stats.setdefault("policy", {})
                key = f"{ai.original_alias} (phrase: no contentful token)"
                rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
                i += best_n  # Skip past this position
                continue

        if not check_case_match(surface, ai, chunk_text, s):
            rejection_stats.setdefault("case_mismatch", {})
            rejection_stats["case_mismatch"][ai.original_alias] = rejection_stats["case_mismatch"].get(ai.original_alias, 0) + 1
            i += best_n  # Skip past this position
            continue

        # Track successful auto-matches by term (use normalized surface for stable grouping)
        rejection_stats.setdefault("auto_matched_terms", {})
        rejection_stats["auto_matched_terms"][surface_norm] = rejection_stats["auto_matched_terms"].get(surface_norm, 0) + 1

        matches_out.append((ai.entity_id, surface, surface_norm, surface_quality, None, None, "exact", 1.0))
        match_type_stats["exact"] += 1
        matched_positions.add((s, e))
        exact_positions.add((s, e))
        i += best_n  # Skip past the matched tokens
        matched_here = True
        continue  # Continue to next iteration of while loop

    # -------------------------
    # Tier 2/3: Partial + Fuzzy candidates (skip spans already exact-matched)
    # -------------------------
    all_candidates: List[MatchCandidate] = []
    if enable_partial:
        if partial_index is None:
            partial_index = build_partial_match_index(aliases_by_norm)
        # OPTIMIZATION: Use normalized_tokens to avoid re-normalization in find_partial_candidates
        all_candidates.extend(find_partial_candidates_optimized(chunk_text, normalized_tokens, aliases_by_norm, partial_index))

    if enable_fuzzy:
        # OPTIMIZATION: Skip fuzzy matching for tokens already matched exactly
        # Only do fuzzy matching for unmatched tokens (avoids wasted computation)
        unmatched_tokens = [
            (s, e, tok, norm) for (s, e, tok, norm) in normalized_tokens
            if (s, e) not in exact_positions and len(norm) >= 4
        ]
        if unmatched_tokens:
            # OPTIMIZATION: Pass normalized tokens to avoid re-normalization
            all_candidates.extend(find_fuzzy_candidates_optimized(unmatched_tokens, aliases_by_norm, alias_norm_set))

    # Group by span
    by_span: DefaultDict[Tuple[int, int], List[MatchCandidate]] = defaultdict(list)
    for c in all_candidates:
        if (c.start_pos, c.end_pos) in exact_positions:
            continue
        by_span[(c.start_pos, c.end_pos)].append(c)

    for (s, e), cands in by_span.items():
        if (s, e) in matched_positions:
            continue

        surface, _, surface_quality = extract_surface_from_tokens(chunk_text, s, e, original_tokens)
        if not surface:
            continue
        # OPTIMIZATION: Pre-compute normalized surface from tokens if possible
        # (extract_surface_from_tokens normalizes, but we can skip if we have the ngram_norm)

        # group by entity
        by_entity: DefaultDict[int, List[MatchCandidate]] = defaultdict(list)
        for c in cands:
            by_entity[c.alias_info.entity_id].append(c)

        chosen: Optional[AliasInfo] = None
        chosen_match_type = None
        chosen_conf = 0.0

        if len(by_entity) > 1:
            alias_infos = [c.alias_info for c in cands]
            meta = chunk_metadata_cache.get(chunk_id, {})
            pdf_pages = meta.get("pdf_pages", [])
            doc_id_for_chunk = meta.get("document_id") or document_id

            # citation disambiguation first
            if pdf_pages and doc_id_for_chunk:
                try:
                    # Use batch cursor instead of creating new one
                    cit_ai, cit_conf, cit_method = resolve_collision_with_citations(
                        alias_infos, doc_id_for_chunk, pdf_pages,
                        entity_citations_cache=entity_citations_cache,
                        document_cache=document_cache,
                        cur=batch_cursor  # Reuse batch cursor
                    )
                    if cit_ai:
                        chosen = cit_ai
                        base = max((c for c in cands if c.alias_info.entity_id == chosen.entity_id), key=lambda x: x.match_confidence, default=None)
                        if base:
                            chosen_match_type = base.match_type
                            chosen_conf = min(1.0, base.match_confidence * 1.2)
                        else:
                            chosen_match_type = "fuzzy"
                            chosen_conf = cit_conf
                        rejection_stats.setdefault("collision_auto_resolved", {})
                        k = f"{surface_norm} (citation_match_{cit_method})"
                        rejection_stats["collision_auto_resolved"][k] = rejection_stats["collision_auto_resolved"].get(k, 0) + 1
                except Exception:
                    pass

            # fallback: best confidence
            if chosen is None:
                best = max(cands, key=lambda x: x.match_confidence)
                chosen = best.alias_info
                chosen_match_type = best.match_type
                chosen_conf = best.match_confidence

        else:
            best = max(cands, key=lambda x: x.match_confidence)
            chosen = best.alias_info
            chosen_match_type = best.match_type
            chosen_conf = best.match_confidence

        if chosen is None:
            continue

        # Eligibility + policy checks (mirror Tier 1 exact)
        ok, reason = is_candidate_eligible_for_matching(chosen, surface_norm, surface)
        if not ok:
            rejection_stats.setdefault("not_eligible", {})
            key = f"{surface_norm} ({reason})"
            rejection_stats["not_eligible"][key] = rejection_stats["not_eligible"].get(key, 0) + 1
            continue

        ai = chosen

        # ALL CAPS override for policy gate (same intent as Tier 1 exact)
        is_all_caps_surface = surface and (surface.isupper() or (len(surface) >= 3 and all(c.isupper() or not c.isalpha() for c in surface if c.isalpha())))
        is_all_caps_alias = ai.original_alias and (ai.original_alias.isupper() or (len(ai.original_alias) >= 3 and all(c.isupper() or not c.isalpha() for c in ai.original_alias if c.isalpha())))
        if is_all_caps_surface or is_all_caps_alias:
            ai = AliasInfo(
                entity_id=ai.entity_id,
                original_alias=ai.original_alias,
                alias_norm=ai.alias_norm,
                entity_type=ai.entity_type,
                is_auto_match=True,  # override
                min_chars=ai.min_chars,
                match_case=ai.match_case,
                match_mode=ai.match_mode,
                is_numeric_entity=ai.is_numeric_entity,
                alias_class=ai.alias_class,
                allow_ambiguous_person_token=ai.allow_ambiguous_person_token,
                requires_context=ai.requires_context,
                alias_type=ai.alias_type,
            )

        if not ai.is_auto_match:
            rejection_stats.setdefault("policy", {})
            key = f"{ai.original_alias} (auto_match disabled)"
            rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
            continue

        if is_purely_numeric(ai.original_alias) and not ai.is_numeric_entity:
            rejection_stats.setdefault("policy", {})
            key = f"{ai.original_alias} (purely numeric)"
            rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
            continue

        toks = surface_norm.split()
        if len(toks) == 1:
            if len(toks[0]) < ai.min_chars:
                rejection_stats.setdefault("policy", {})
                key = f"{ai.original_alias} (chars < {ai.min_chars})"
                rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
                continue
        else:
            if max(len(t) for t in toks) < ai.min_chars:
                rejection_stats.setdefault("policy", {})
                key = f"{ai.original_alias} (phrase: no token ≥ {ai.min_chars})"
                rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
                continue
            if not has_contentful_token(toks, min_chars=3):
                rejection_stats.setdefault("policy", {})
                key = f"{ai.original_alias} (phrase: no contentful token)"
                rejection_stats["policy"][key] = rejection_stats["policy"].get(key, 0) + 1
                continue

        if not check_case_match(surface, ai, chunk_text, s):
            rejection_stats.setdefault("case_mismatch", {})
            rejection_stats["case_mismatch"][ai.original_alias] = rejection_stats["case_mismatch"].get(ai.original_alias, 0) + 1
            continue

        # Phase 3: Context check for partial/fuzzy matches too
        if ai.requires_context == "codename_like":
            if not check_codename_context(chunk_text, s, e, surface):
                rejection_stats.setdefault("context_gate", {})
                key = f"{ai.original_alias} (requires_context={ai.requires_context})"
                rejection_stats["context_gate"][key] = rejection_stats["context_gate"].get(key, 0) + 1
                continue

        # Track successful auto-matches (partial/fuzzy too)
        rejection_stats.setdefault("auto_matched_terms", {})
        rejection_stats["auto_matched_terms"][surface_norm] = rejection_stats["auto_matched_terms"].get(surface_norm, 0) + 1

        matches_out.append((ai.entity_id, surface, surface_norm, surface_quality, None, None, chosen_match_type or "exact", float(chosen_conf)))
        match_type_stats[chosen_match_type or "exact"] = match_type_stats.get(chosen_match_type or "exact", 0) + 1
        matched_positions.add((s, e))

    return matches_out, match_type_stats


# =============================================================================
# Batch extraction + insert (idempotent)
# =============================================================================

def extract_mentions_batch(
    conn,
    chunks: List[Tuple[int, str, Optional[int]]],
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    *,
    dry_run: bool,
    show_samples: bool,
    max_samples: int,
    rejection_stats: Dict[str, Dict[str, int]],
    collision_queue: List[Dict[str, Any]],
    preferred_entity_id_map: Dict[Tuple[Optional[str], str], int],
    ban_entity_map: Optional[Dict[Tuple[Optional[str], str], Set[int]]],
    ban_surface_map: Optional[Dict[Tuple[Optional[str], str], bool]],
    scope: Optional[str],
    enable_partial: bool,
    enable_fuzzy: bool,
    # CRITICAL: Pre-loaded global caches (loaded ONCE before all batches)
    document_cache: Dict[str, int],
    entity_citations_cache: Dict[int, List[Dict[str, Any]]],
    entity_equivalence_cache: Dict[int, Set[int]],
    canonical_names_cache: Dict[int, str],
    entities_with_definitions: Set[int],
    partial_index: Optional[Dict[str, List[str]]],
    ac_automaton: Any = None,  # Aho-Corasick automaton for fast matching
) -> Tuple[int, int, List[Dict[str, Any]], Dict[str, int]]:
    mentions_to_insert: List[Dict[str, Any]] = []
    sample_mentions: List[Dict[str, Any]] = []
    chunks_processed = 0

    # Only load batch-specific data (chunk metadata changes per batch)
    with conn.cursor() as cur:
        chunk_ids = [cid for (cid, _, _) in chunks]
        chunk_metadata_cache = batch_load_chunk_metadata(cur, chunk_ids)

    total_in_batch = len(chunks)
    batch_match_stats = {"exact": 0, "partial": 0, "fuzzy": 0}

    # Create a single reusable cursor for the entire batch (CRITICAL PERFORMANCE FIX)
    # This avoids creating hundreds/thousands of cursors inside collision resolution loops
    batch_cursor = conn.cursor()

    # Timing breakdown for profiling
    time_find_matches = 0.0
    time_build_mentions = 0.0
    
    try:
        for idx, (chunk_id, chunk_text, document_id) in enumerate(chunks, 1):
            chunks_processed += 1
            if idx % 50 == 0 or idx == 1:  # Print less frequently for speed
                print(
                    f"    Processing chunk {idx}/{total_in_batch} (chunk_id={chunk_id}, found {len(mentions_to_insert)} mentions so far)...",
                    file=sys.stderr,
                    end="\r",
                )
                sys.stderr.flush()

            if document_id is None:
                continue

            t0 = time.time()
            
            # Use fast Aho-Corasick matching if available
            if ac_automaton is not None and _USE_AHOCORASICK:
                matches, match_stats = find_matches_for_chunk_fast(
                    chunk_text,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    aliases_by_norm=aliases_by_norm,
                    alias_norm_set=alias_norm_set,
                    ac_automaton=ac_automaton,
                    enable_partial=enable_partial,
                    enable_fuzzy=enable_fuzzy,
                    partial_index=partial_index,
                    chunk_metadata_cache=chunk_metadata_cache,
                    entity_citations_cache=entity_citations_cache,
                    entity_equivalence_cache=entity_equivalence_cache,
                    document_cache=document_cache,
                    preferred_entity_id_map=preferred_entity_id_map,
                    ban_entity_map=ban_entity_map,
                    ban_surface_map=ban_surface_map,
                    scope=scope,
                    rejection_stats=rejection_stats,
                    collision_queue=collision_queue,
                    entities_with_definitions=entities_with_definitions,
                    canonical_names_cache=canonical_names_cache,
                    batch_cursor=batch_cursor,
                    conn=conn,
                )
            else:
                matches, match_stats = find_matches_for_chunk(
                    chunk_text,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    aliases_by_norm=aliases_by_norm,
                    alias_norm_set=alias_norm_set,
                    enable_partial=enable_partial,
                    enable_fuzzy=enable_fuzzy,
                    partial_index=partial_index,
                    chunk_metadata_cache=chunk_metadata_cache,
                    entity_citations_cache=entity_citations_cache,
                    entity_equivalence_cache=entity_equivalence_cache,
                    document_cache=document_cache,
                    preferred_entity_id_map=preferred_entity_id_map,
                    ban_entity_map=ban_entity_map,
                    ban_surface_map=ban_surface_map,
                    scope=scope,
                    rejection_stats=rejection_stats,
                    collision_queue=collision_queue,
                    entities_with_definitions=entities_with_definitions,
                    canonical_names_cache=canonical_names_cache,
                    batch_cursor=batch_cursor,  # Reuse single cursor for entire batch
                    conn=conn,
                )
            time_find_matches += time.time() - t0

            t1 = time.time()
            for k, v in match_stats.items():
                batch_match_stats[k] = batch_match_stats.get(k, 0) + v

            for (entity_id, surface, surface_norm, surface_quality, start_char, end_char, match_type, confidence) in matches:
                method_map = {"exact": "alias_exact", "partial": "alias_partial", "fuzzy": "alias_fuzzy"}
                method = method_map.get(match_type, "alias_exact")

                mention = {
                    "entity_id": entity_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "surface": surface,
                    "surface_norm": surface_norm,
                    "surface_quality": surface_quality,
                    "start_char": start_char,
                    "end_char": end_char,
                    "confidence": float(confidence),
                    "method": method,
                    "match_type": match_type,
                }
                mentions_to_insert.append(mention)
                if show_samples and len(sample_mentions) < max_samples:
                    sample_mentions.append(mention)
            time_build_mentions += time.time() - t1
    finally:
        # Always close the batch cursor
        batch_cursor.close()
    
    # Print timing breakdown
    if chunks_processed > 0:
        print(f"\n  [TIMING] find_matches: {time_find_matches:.2f}s ({time_find_matches/chunks_processed*1000:.1f}ms/chunk), build_mentions: {time_build_mentions:.2f}s", file=sys.stderr)

    # clear progress line
    if total_in_batch > 0:
        print(" " * 100, file=sys.stderr, end="\r")
        sys.stderr.flush()

    if dry_run or not mentions_to_insert:
        return chunks_processed, len(mentions_to_insert), sample_mentions, batch_match_stats

    # OPTIMIZATION: Optimized duplicate check using temp table with index
    # Pre-detect columns once to avoid repeated queries
    with conn.cursor() as cur:
        # Detect surface_quality columns once (before building insert data)
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='entity_mentions'
              AND column_name IN ('surface_norm','surface_quality')
        """)
        cols = {r[0] for r in cur.fetchall()}
        has_surface_quality = ("surface_norm" in cols) and ("surface_quality" in cols)

        if not mentions_to_insert:
            conn.commit()
            return chunks_processed, 0, sample_mentions, batch_match_stats

        # OPTIMIZATION: Use temp table with UNLOGGED for faster inserts (PostgreSQL 9.1+)
        # UNLOGGED tables are faster but data is lost on crash (fine for temp tables)
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS mention_check (
                chunk_id BIGINT,
                entity_id BIGINT,
                surface TEXT,
                method TEXT
            ) ON COMMIT DROP
        """)
        # Create unique index for faster lookups and duplicate prevention
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mention_check_unique 
            ON mention_check(chunk_id, entity_id, surface, method)
        """)
        
        check_tuples = [(m["chunk_id"], m["entity_id"], m["surface"], m["method"]) for m in mentions_to_insert]
        execute_values(cur, "INSERT INTO mention_check (chunk_id, entity_id, surface, method) VALUES %s ON CONFLICT DO NOTHING", check_tuples, page_size=1000)

        # OPTIMIZATION: Use EXISTS instead of JOIN for faster duplicate check
        cur.execute("""
            SELECT mc.chunk_id, mc.entity_id, mc.surface, mc.method
            FROM mention_check mc
            WHERE EXISTS (
                SELECT 1 FROM entity_mentions em
                WHERE em.chunk_id = mc.chunk_id
                  AND em.entity_id = mc.entity_id
                  AND em.surface = mc.surface
                  AND em.method = mc.method
            )
        """)
        existing = set(cur.fetchall())

        new_mentions = [m for m in mentions_to_insert if (m["chunk_id"], m["entity_id"], m["surface"], m["method"]) not in existing]

        if not new_mentions:
            conn.commit()
            return chunks_processed, 0, sample_mentions, batch_match_stats

        # Insert only new mentions
        if has_surface_quality:
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
                        m["entity_id"], m["chunk_id"], m["document_id"], m["surface"],
                        m.get("surface_norm"), m.get("surface_quality", "exact"),
                        m["start_char"], m["end_char"], m["confidence"], m["method"]
                    )
                    for m in new_mentions
                ],
                page_size=1000
            )
        else:
            execute_values(
                cur,
                """
                INSERT INTO entity_mentions
                    (entity_id, chunk_id, document_id, surface, start_char, end_char, confidence, method)
                VALUES %s
                """,
                [
                    (m["entity_id"], m["chunk_id"], m["document_id"], m["surface"], m["start_char"], m["end_char"], m["confidence"], m["method"])
                    for m in new_mentions
                ],
                page_size=1000
            )

    conn.commit()
    return chunks_processed, len(new_mentions), sample_mentions, batch_match_stats


# =============================================================================
# Override Mappings (Unified prefer/ban system)
# =============================================================================

@dataclass
class OverrideConfig:
    """Configuration for entity alias overrides."""
    # prefer: (scope, surface_norm) -> forced_entity_id
    prefer_map: Dict[Tuple[Optional[str], str], int]
    # ban_entity: (scope, surface_norm) -> set of banned entity_ids
    ban_entity_map: Dict[Tuple[Optional[str], str], Set[int]]
    # ban_surface: (scope, surface_norm) -> True if surface is entirely banned
    ban_surface_map: Dict[Tuple[Optional[str], str], bool]


def load_override_config(conn) -> OverrideConfig:
    """
    Load unified override configuration from entity_alias_overrides table.
    
    This is the primary mechanism for human overrides:
    - forced_entity_id: prefer this entity for this surface
    - banned_entity_id: never match this entity for this surface
    - banned: never match any entity for this surface
    
    Returns: OverrideConfig with prefer_map, ban_entity_map, ban_surface_map
    """
    prefer_map: Dict[Tuple[Optional[str], str], int] = {}
    ban_entity_map: Dict[Tuple[Optional[str], str], Set[int]] = {}
    ban_surface_map: Dict[Tuple[Optional[str], str], bool] = {}
    
    # Load from unified entity_alias_overrides table (highest priority)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    surface_norm,
                    scope,
                    CASE WHEN scope = 'collection' THEN scope_collection_id::TEXT
                         WHEN scope = 'document' THEN scope_document_id::TEXT
                         ELSE NULL END as scope_id,
                    forced_entity_id,
                    banned_entity_id,
                    banned
                FROM entity_alias_overrides
            """)
            for row in cur.fetchall():
                surface_norm, scope_type, scope_id, forced_eid, banned_eid, banned = row
                
                # Determine scope key
                if scope_type == 'global' or scope_type is None:
                    scope_key = None
                elif scope_type == 'collection' and scope_id:
                    # Look up collection slug
                    cur.execute("SELECT slug FROM collections WHERE id = %s", (int(scope_id),))
                    slug_row = cur.fetchone()
                    scope_key = slug_row[0] if slug_row else scope_id
                else:
                    scope_key = scope_id
                
                key = (scope_key, surface_norm)
                
                # Process overrides
                if forced_eid:
                    prefer_map[key] = forced_eid
                if banned_eid:
                    if key not in ban_entity_map:
                        ban_entity_map[key] = set()
                    ban_entity_map[key].add(banned_eid)
                if banned:
                    ban_surface_map[key] = True
    except Exception as e:
        # Table might not exist yet
        print(f"Note: entity_alias_overrides table not loaded: {e}", file=sys.stderr)
    
    return OverrideConfig(
        prefer_map=prefer_map,
        ban_entity_map=ban_entity_map,
        ban_surface_map=ban_surface_map,
    )


def load_preferred_mappings(
    conn,
    csv_path: Optional[str] = None,
    table_name: Optional[str] = None,
) -> Dict[Tuple[Optional[str], str], int]:
    """
    Improvement 6: Load preferred entity mappings from CSV or DB table.
    
    CSV format: scope,alias_norm,preferred_entity_id
    DB table format: scope (nullable), alias_norm, preferred_entity_id
    
    Also loads from:
    - entity_alias_overrides table (unified system, highest priority)
    - entity_alias_preferred table (legacy, for backward compatibility)
    
    Returns: Dict mapping (scope, alias_norm) -> preferred_entity_id
    """
    mappings: Dict[Tuple[Optional[str], str], int] = {}
    
    # Load from legacy entity_alias_preferred table first
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT scope, alias_norm, preferred_entity_id
                FROM entity_alias_preferred
            """)
            for row in cur.fetchall():
                scope, alias_norm, preferred_entity_id = row
                if alias_norm and preferred_entity_id:
                    mappings[(scope, alias_norm)] = preferred_entity_id
    except Exception:
        # Table might not exist yet - that's okay
        pass
    
    # Load from unified entity_alias_overrides table (overrides legacy)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    surface_norm,
                    scope,
                    CASE WHEN scope = 'collection' THEN scope_collection_id::TEXT
                         ELSE NULL END as scope_id,
                    forced_entity_id
                FROM entity_alias_overrides
                WHERE forced_entity_id IS NOT NULL
            """)
            for row in cur.fetchall():
                surface_norm, scope_type, scope_id, forced_eid = row
                
                # Determine scope key
                if scope_type == 'global' or scope_type is None:
                    scope_key = None
                elif scope_type == 'collection' and scope_id:
                    cur.execute("SELECT slug FROM collections WHERE id = %s", (int(scope_id),))
                    slug_row = cur.fetchone()
                    scope_key = slug_row[0] if slug_row else None
                else:
                    scope_key = None
                
                if surface_norm and forced_eid:
                    mappings[(scope_key, surface_norm)] = forced_eid
    except Exception:
        pass
    
    # Load from CSV if provided
    if csv_path:
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scope = row.get('scope', '').strip() or None
                alias_norm = row.get('alias_norm', '').strip()
                preferred_entity_id = int(row.get('preferred_entity_id', 0))
                if alias_norm and preferred_entity_id:
                    mappings[(scope, alias_norm)] = preferred_entity_id
    
    # Load from custom table if provided
    if table_name:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT scope, alias_norm, preferred_entity_id
                FROM {table_name}
            """)
            for row in cur.fetchall():
                scope, alias_norm, preferred_entity_id = row
                if alias_norm and preferred_entity_id:
                    mappings[(scope, alias_norm)] = preferred_entity_id
    
    return mappings


def load_ban_mappings(conn) -> Tuple[Dict[Tuple[Optional[str], str], Set[int]], Dict[Tuple[Optional[str], str], bool]]:
    """
    Load ban mappings from entity_alias_overrides table.
    
    Returns:
        - ban_entity_map: (scope, surface_norm) -> set of banned entity_ids
        - ban_surface_map: (scope, surface_norm) -> True if surface entirely banned
    """
    ban_entity_map: Dict[Tuple[Optional[str], str], Set[int]] = {}
    ban_surface_map: Dict[Tuple[Optional[str], str], bool] = {}
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    surface_norm,
                    scope,
                    CASE WHEN scope = 'collection' THEN scope_collection_id::TEXT
                         ELSE NULL END as scope_id,
                    banned_entity_id,
                    banned
                FROM entity_alias_overrides
                WHERE banned_entity_id IS NOT NULL OR banned = TRUE
            """)
            for row in cur.fetchall():
                surface_norm, scope_type, scope_id, banned_eid, banned = row
                
                # Determine scope key
                if scope_type == 'global' or scope_type is None:
                    scope_key = None
                elif scope_type == 'collection' and scope_id:
                    cur.execute("SELECT slug FROM collections WHERE id = %s", (int(scope_id),))
                    slug_row = cur.fetchone()
                    scope_key = slug_row[0] if slug_row else None
                else:
                    scope_key = None
                
                key = (scope_key, surface_norm)
                
                if banned_eid:
                    if key not in ban_entity_map:
                        ban_entity_map[key] = set()
                    ban_entity_map[key].add(banned_eid)
                if banned:
                    ban_surface_map[key] = True
    except Exception:
        pass
    
    return ban_entity_map, ban_surface_map


def apply_bans_to_candidates(
    alias_norm: str,
    alias_infos: List[AliasInfo],
    ban_entity_map: Optional[Dict[Tuple[Optional[str], str], Set[int]]],
    ban_surface_map: Optional[Dict[Tuple[Optional[str], str], bool]],
    scope: Optional[str],
) -> Tuple[List[AliasInfo], Optional[str]]:
    """
    Apply ban overrides to filter candidates.
    
    Rule 0a: Check surface ban (scoped then global)
    Rule 0b: Filter out banned entities
    
    Returns: (filtered_candidates, ban_reason or None)
    """
    # Rule 0a: Check if surface is entirely banned
    if ban_surface_map:
        if (scope, alias_norm) in ban_surface_map:
            return [], f"surface_banned(scope={scope})"
        if (None, alias_norm) in ban_surface_map:
            return [], "surface_banned(global)"
    
    # Rule 0b: Filter out banned entities
    if ban_entity_map:
        banned_eids: Set[int] = set()
        # Scoped bans first
        if (scope, alias_norm) in ban_entity_map:
            banned_eids.update(ban_entity_map[(scope, alias_norm)])
        # Global bans
        if (None, alias_norm) in ban_entity_map:
            banned_eids.update(ban_entity_map[(None, alias_norm)])
        
        if banned_eids:
            filtered = [ai for ai in alias_infos if ai.entity_id not in banned_eids]
            if len(filtered) < len(alias_infos):
                removed = len(alias_infos) - len(filtered)
                return filtered, f"entities_banned(removed={removed},ids={sorted(banned_eids)})"
    
    return alias_infos, None


# =============================================================================
# Review Queue Population
# =============================================================================

def populate_review_queue_from_collision_queue(
    conn,
    collision_queue: List[Dict[str, Any]],
) -> int:
    """
    Populate mention_review_queue from collision_queue items.
    
    This function inserts collision_queue items into the mention_review_queue table
    for human review/adjudication.
    """
    if not collision_queue:
        return 0
    
    # Build review items with candidates
    review_items = []
    
    for item in collision_queue:
        chunk_id = item.get('chunk_id')
        document_id = item.get('document_id')
        surface = item.get('surface', '')
        context_excerpt = item.get('context_excerpt', '')
        candidate_entity_ids = item.get('candidate_entity_ids', [])
        start_char = item.get('start_char')
        end_char = item.get('end_char')
        collision_metadata = item.get('collision_metadata', {})
        
        if not chunk_id or not document_id or not surface:
            continue
        
        if not candidate_entity_ids:
            continue
        
        # Build candidates JSONB with entity information
        # Sort entity IDs to ensure consistent ordering for deduplication
        candidate_entity_ids_sorted = sorted(set(candidate_entity_ids))
        candidates = []
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, canonical_name, entity_type
                FROM entities
                WHERE id = ANY(%s)
                ORDER BY id
            """, (candidate_entity_ids_sorted,))
            for entity_id, canonical_name, entity_type in cur.fetchall():
                candidates.append({
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                    "score": 1.0,  # Default score for collision candidates
                })
        
        if not candidates:
            continue
        
        # Add collision metadata to candidates JSONB for review/adjudication
        # This allows filtering and prioritization in the review UI
        candidates_with_metadata = {
            "entities": candidates,
            "collision_metadata": {
                "is_high_value": collision_metadata.get("is_high_value", False),
                "is_harmless": collision_metadata.get("is_harmless", False),
                "collision_category": collision_metadata.get("collision_category", "unknown"),
                "enqueue_flag": collision_metadata.get("enqueue_flag", False),
                "info": collision_metadata.get("info"),
            }
        }
        
        # Compute grouping keys for batch processing
        surface_norm = normalize_surface(surface)
        candidate_set_hash = compute_candidate_set_hash(candidate_entity_ids_sorted)
        group_key = compute_group_key(surface_norm, candidate_entity_ids_sorted)
        
        review_items.append({
            'mention_type': 'entity',
            'chunk_id': chunk_id,
            'document_id': document_id,
            'surface': surface,
            'surface_norm': surface_norm,
            'start_char': start_char,
            'end_char': end_char,
            'context_excerpt': context_excerpt or surface,
            'candidates': Json(candidates_with_metadata),
            'candidate_entity_ids': candidate_entity_ids_sorted,  # Store sorted IDs for deduplication
            'candidate_set_hash': candidate_set_hash,
            'group_key': group_key,
        })
    
    if not review_items:
        return 0
    
    # Deduplicate review_items by (chunk_id, surface, candidate_entity_ids)
    # Same collision might appear multiple times in collision_queue (e.g., from different batches)
    # Use candidate_entity_ids for deduplication (more reliable than JSON comparison)
    seen_keys = set()
    unique_review_items = []
    skipped_in_queue = 0
    for item in review_items:
        # Create a unique key based on chunk_id, surface, and sorted entity IDs
        # Sort entity IDs to ensure consistent comparison even if order differs
        candidate_ids = tuple(sorted(item.get('candidate_entity_ids', [])))
        key = (item['chunk_id'], item['surface'], item['mention_type'], candidate_ids)
        key = (
            item['chunk_id'],
            item['mention_type'],
            item['surface'],
            tuple(sorted(item.get('candidate_entity_ids', []))),
            item.get('start_char'),
            item.get('end_char'),
            )
        if key not in seen_keys:
            seen_keys.add(key)
            unique_review_items.append(item)
        else:
            skipped_in_queue += 1  # Duplicate collision in collision_queue - skip it
    
    if not unique_review_items:
        return 0
    
    # Insert into review queue with duplicate handling
    # PostgreSQL doesn't support ON CONFLICT with expression-based unique indexes,
    # so we check for duplicates first and use savepoints to handle race conditions
    inserted = 0
    skipped_duplicates = 0
    errors = []
    with conn.cursor() as cur:
        for item in unique_review_items:
            # Use savepoint to handle errors without aborting the transaction
            savepoint_name = f"sp_review_{item['chunk_id']}_{hash(item['surface']) % 10000}"
            # Check if item already exists (before attempting insert)
            # Match the unique index: (chunk_id, surface, mention_type, md5(candidates::text)) WHERE status = 'pending'
            # We check for pending items specifically to match the unique constraint behavior
            cur.execute("""
                SELECT id, status
                FROM mention_review_queue
                WHERE chunk_id = %s AND surface = %s AND mention_type = %s
                  AND md5(candidates::text) = md5(%s::jsonb::text)
                  AND status = 'pending'
                LIMIT 1
            """, (item['chunk_id'], item['surface'], item['mention_type'], item['candidates']))
            existing_row = cur.fetchone()
            
            if existing_row:
                existing_id, existing_status = existing_row
                skipped_duplicates += 1
                # Always show debug for first 10, then summary
                if skipped_duplicates <= 10:
                    print(f"  [DEBUG] Already exists (id={existing_id}, status={existing_status}): chunk_id={item['chunk_id']}, surface='{item['surface']}'", file=sys.stderr)
                elif skipped_duplicates == 11:
                    print(f"  [DEBUG] ... and more duplicates (showing first 10)", file=sys.stderr)
                continue  # Skip - already in queue as pending
            
            # Item doesn't exist - insert it
            try:
                cur.execute(f"SAVEPOINT {savepoint_name}")
                
                # Insert new item - let unique constraint handle race conditions
                cur.execute("""
                    INSERT INTO mention_review_queue
                        (mention_type, chunk_id, document_id, surface, surface_norm, start_char, end_char,
                         context_excerpt, candidates, status, group_key, candidate_set_hash, candidate_entity_ids)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item['mention_type'],
                    item['chunk_id'],
                    item['document_id'],
                    item['surface'],
                    item.get('surface_norm'),
                    item.get('start_char'),
                    item.get('end_char'),
                    item['context_excerpt'],
                    item['candidates'],  # Json object will be properly serialized by psycopg2
                    'pending',
                    item.get('group_key'),
                    item.get('candidate_set_hash'),
                    item.get('candidate_entity_ids'),  # BIGINT[] array
                ))
                inserted += 1
                if inserted <= 3:  # Debug first few
                    print(f"  [DEBUG] Inserted: chunk_id={item['chunk_id']}, surface='{item['surface']}'", file=sys.stderr)
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            except psycopg2.IntegrityError as e:
                # Race condition - another process inserted it between our check and insert
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                except:
                    pass
                skipped_duplicates += 1
                if skipped_duplicates <= 5:
                    print(f"  [DEBUG] Race condition duplicate: chunk_id={item['chunk_id']}, surface='{item['surface']}'", file=sys.stderr)
                continue
            except Exception as e:
                # Other errors - log but continue
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                except:
                    pass
                errors.append(f"Error for chunk_id={item['chunk_id']}, surface='{item['surface']}': {e}")
                continue
    
    conn.commit()
    if skipped_duplicates > 0 or skipped_in_queue > 0:
        print(f"  (Skipped {skipped_duplicates} duplicates already in review queue, {skipped_in_queue} duplicates within collision_queue)", file=sys.stderr)
    if inserted < len(unique_review_items):
        print(f"  (Warning: Only {inserted} of {len(unique_review_items)} unique items were inserted)", file=sys.stderr)
        print(f"  (Note: Skipped items may have been reviewed/accepted/rejected in a previous run)", file=sys.stderr)
    if errors:
        print(f"  (Errors encountered: {len(errors)} items failed)", file=sys.stderr)
        for err in errors[:5]:  # Show first 5 errors
            print(f"    - {err}", file=sys.stderr)
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors", file=sys.stderr)
    return inserted


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Extract entity mentions from chunks using multi-tier matching (exact, partial, fuzzy) with citation-based disambiguation"
    )
    ap.add_argument("--collection", type=str, help="Filter by collection slug")
    ap.add_argument("--document-id", type=int, help="Filter by document ID")
    ap.add_argument("--chunk-id-range", type=str, help="Filter by chunk ID range (format: start:end)")
    ap.add_argument("--dry-run", action="store_true", help="Print counts only, no inserts")
    ap.add_argument("--show-samples", action="store_true", help="Show sample mentions (works with --dry-run)")
    ap.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples to show (default: 10)")
    ap.add_argument("--limit", type=int, help="Limit number of chunks processed")
    ap.add_argument("--batch-size", type=int, default=500, help="Process chunks in batches (default: 500, increased for performance)")
    ap.add_argument("--enable-partial", action="store_true", help="Enable partial matching (default: False)")
    ap.add_argument("--enable-fuzzy", action="store_true", help="Enable fuzzy matching (default: False)")
    ap.add_argument("--skip-diagnostics", action="store_true", help="Skip expensive diagnostic operations (faster startup)")
    ap.add_argument("--summary-csv", type=str, help="Output match summary statistics to CSV file")
    ap.add_argument(
        "--concordance-source-slug",
        type=str,
        help="Only match entities tied to this concordance source slug (e.g. vassiliev_venona_index_full)",
    )
    ap.add_argument(
        "--preferred-mappings-csv",
        type=str,
        help="Load preferred entity mappings from CSV file (format: scope,alias_norm,preferred_entity_id). "
             "Note: entity_alias_preferred table is always loaded automatically.",
    )
    ap.add_argument(
        "--preferred-mappings-table",
        type=str,
        help="Load preferred entity mappings from additional DB table (columns: scope, alias_norm, preferred_entity_id). "
             "Note: entity_alias_preferred table is always loaded automatically.",
    )

    args = ap.parse_args()

    if not any([args.collection, args.document_id, args.chunk_id_range]):
        ap.error("Must specify at least one of: --collection, --document-id, --chunk-id-range")

    chunk_id_start = chunk_id_end = None
    if args.chunk_id_range:
        chunk_id_start, chunk_id_end = parse_chunk_id_range(args.chunk_id_range)

    conn = get_conn()
    try:
        if _USE_FAST_LEVENSHTEIN:
            print("Using optimized Levenshtein distance (python-Levenshtein) for fuzzy matching", file=sys.stderr)
        else:
            print("WARNING: Using slow Python Levenshtein implementation. Install python-Levenshtein for 5-10x faster fuzzy matching:", file=sys.stderr)
            print("  pip install python-Levenshtein", file=sys.stderr)

        # If partial enabled, verify DB constraint supports alias_partial if present
        if args.enable_partial:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT check_clause
                    FROM information_schema.check_constraints
                    WHERE constraint_name = 'entity_mentions_method_check'
                """)
                row = cur.fetchone()
                if row and ("'alias_partial'" not in row[0]):
                    print("ERROR: Database schema does not support 'alias_partial' method.", file=sys.stderr)
                    print("  Run migration 0026 to add support:", file=sys.stderr)
                    print("  make ner-methods-and-text-quality", file=sys.stderr)
                    sys.exit(1)

        print("Loading entity aliases...", file=sys.stderr)
        aliases_by_norm, alias_norm_set = load_all_aliases(
            conn,
            collection_slug=args.collection,
            concordance_source_slug=args.concordance_source_slug,
        )
        print(f"  Loaded {len(aliases_by_norm)} unique normalized aliases ({len(alias_norm_set)} in lookup set)", file=sys.stderr)

        if not args.skip_diagnostics:
            print("\nTop 50 alias_norms by candidate count (for alias_class diagnostic):", file=sys.stderr)
            sorted_aliases = sorted(aliases_by_norm.items(), key=lambda x: len(x[1]), reverse=True)[:50]
            for alias_norm, infos in sorted_aliases:
                if len(infos) <= 1:
                    continue
                types = {ai.entity_type for ai in infos}
                classes = {ai.alias_class for ai in infos if ai.alias_class}
                samples = [ai.original_alias for ai in infos[:3]]
                print(f"  {alias_norm}: {len(infos)} candidates, types={types}, classes={classes}, samples={samples}", file=sys.stderr)
            # entity type distribution
            counts: Dict[str, int] = {}
            for infos in aliases_by_norm.values():
                for ai in infos:
                    counts[ai.entity_type] = counts.get(ai.entity_type, 0) + 1
            print(f"\nEntity type distribution: {counts}", file=sys.stderr)
            print("", file=sys.stderr)

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
            return

        batch_size = args.batch_size
        total_mentions = 0
        total_processed = 0
        all_samples: List[Dict[str, Any]] = []
        rejection_stats: Dict[str, Dict[str, int]] = {}
        collision_queue: List[Dict[str, Any]] = []

        # Improvement 6: Load preferred mappings
        preferred_entity_id_map = load_preferred_mappings(
            conn,
            csv_path=args.preferred_mappings_csv,
            table_name=args.preferred_mappings_table,
        )
        if preferred_entity_id_map:
            print(f"Loaded {len(preferred_entity_id_map)} preferred entity mappings", file=sys.stderr)

        # Load ban mappings from entity_alias_overrides
        ban_entity_map, ban_surface_map = load_ban_mappings(conn)
        if ban_entity_map or ban_surface_map:
            print(f"Loaded {len(ban_entity_map)} entity bans, {len(ban_surface_map)} surface bans", file=sys.stderr)

        print(f"\nProcessing chunks in batches of {batch_size}...", file=sys.stderr)
        if args.dry_run:
            print("  [DRY RUN] No changes will be made", file=sys.stderr)
        print("  Note: Unresolved high-value collisions will be enqueued (not guessed)", file=sys.stderr)
        modes = ["exact"] + (["partial"] if args.enable_partial else []) + (["fuzzy"] if args.enable_fuzzy else [])
        print(f"  Matching modes: {', '.join(modes)}\n", file=sys.stderr)

        # =============================================================================
        # CRITICAL OPTIMIZATION: Pre-load ALL global caches ONCE before batch loop
        # This was previously done inside each batch, causing massive slowdown!
        # =============================================================================
        print("Pre-loading global caches (one-time operation)...", file=sys.stderr)
        cache_start_time = time.time()
        
        with conn.cursor() as cur:
            print("  Loading document cache...", file=sys.stderr)
            document_cache = batch_load_document_cache(cur)
            
            print("  Loading entity citations...", file=sys.stderr)
            all_entity_ids = list({ai.entity_id for infos in aliases_by_norm.values() for ai in infos})
            entity_citations_cache = batch_load_entity_citations(cur, all_entity_ids)
            
            print("  Loading entity equivalence relationships...", file=sys.stderr)
            entity_equivalence_cache = batch_load_entity_equivalence(cur, all_entity_ids)
            
            print("  Loading canonical names...", file=sys.stderr)
            canonical_names_cache = get_canonical_names(cur, all_entity_ids)
        
        print("  Loading entities with definitions...", file=sys.stderr)
        entities_with_definitions = load_entities_with_definitions(conn)
        
        print("  Building partial match index...", file=sys.stderr)
        partial_index = build_partial_match_index(aliases_by_norm) if args.enable_partial else None
        
        # Build Aho-Corasick automaton for fast multi-pattern matching
        ac_automaton = None
        if _USE_AHOCORASICK:
            print("  Building Aho-Corasick automaton for fast matching...", file=sys.stderr)
            ac_start = time.time()
            ac_automaton = build_ahocorasick_automaton(alias_norm_set)
            ac_time = time.time() - ac_start
            print(f"    Automaton built in {ac_time:.2f}s ({len(alias_norm_set)} patterns)", file=sys.stderr)
        else:
            print("  WARNING: Aho-Corasick not available, using slow matching", file=sys.stderr)
        
        cache_elapsed = time.time() - cache_start_time
        print(f"  Global caches loaded in {cache_elapsed:.2f}s", file=sys.stderr)
        print(f"    - Documents: {len(document_cache)}", file=sys.stderr)
        print(f"    - Entity citations: {len(entity_citations_cache)}", file=sys.stderr)
        print(f"    - Entity equivalence: {len(entity_equivalence_cache)}", file=sys.stderr)
        print(f"    - Canonical names: {len(canonical_names_cache)}", file=sys.stderr)
        print(f"    - Entities with definitions: {len(entities_with_definitions)}", file=sys.stderr)
        if partial_index:
            print(f"    - Partial index entries: {len(partial_index)}", file=sys.stderr)
        if ac_automaton:
            print(f"    - Aho-Corasick automaton: {len(alias_norm_set)} patterns", file=sys.stderr)
        print("", file=sys.stderr)

        total_match_stats = {"exact": 0, "partial": 0, "fuzzy": 0}
        extraction_start_time = time.time()

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            # Time the batch processing
            batch_start_time = time.time()
            
            processed, mentions, samples, batch_match_stats = extract_mentions_batch(
                conn,
                batch,
                aliases_by_norm,
                alias_norm_set,
                dry_run=args.dry_run,
                show_samples=args.show_samples,
                max_samples=args.max_samples,
                rejection_stats=rejection_stats,
                collision_queue=collision_queue,
                preferred_entity_id_map=preferred_entity_id_map,
                ban_entity_map=ban_entity_map,
                ban_surface_map=ban_surface_map,
                scope=args.collection,
                enable_partial=args.enable_partial,
                enable_fuzzy=args.enable_fuzzy,
                # CRITICAL: Pass pre-loaded global caches
                document_cache=document_cache,
                entity_citations_cache=entity_citations_cache,
                entity_equivalence_cache=entity_equivalence_cache,
                canonical_names_cache=canonical_names_cache,
                entities_with_definitions=entities_with_definitions,
                partial_index=partial_index,
                ac_automaton=ac_automaton,  # Fast Aho-Corasick matching
            )

            batch_elapsed = time.time() - batch_start_time
            chunks_per_sec = processed / batch_elapsed if batch_elapsed > 0 else 0

            total_processed += processed
            total_mentions += mentions
            for k, v in batch_match_stats.items():
                total_match_stats[k] = total_match_stats.get(k, 0) + v

            if args.show_samples:
                all_samples.extend(samples)
                if len(all_samples) >= args.max_samples:
                    all_samples = all_samples[:args.max_samples]

            print(
                f"  Batch {batch_num}/{total_batches}: processed {min(i + len(batch), total_chunks)}/{total_chunks} chunks "
                f"({processed} chunks, {batch_elapsed:.2f}s, {chunks_per_sec:.1f} chunks/sec), "
                f"inserted {total_mentions} mentions (+{mentions} this batch)",
                file=sys.stderr
            )

        total_elapsed = time.time() - extraction_start_time
        avg_chunks_per_sec = total_processed / total_elapsed if total_elapsed > 0 else 0
        
        print(f"\n{'Would extract' if args.dry_run else 'Extracted'}:", file=sys.stderr)
        print(f"  Chunks processed: {total_processed}", file=sys.stderr)
        print(f"  Mentions found: {total_mentions}", file=sys.stderr)
        print(f"  Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)", file=sys.stderr)
        print(f"  Average speed: {avg_chunks_per_sec:.1f} chunks/sec", file=sys.stderr)

        if args.enable_partial or args.enable_fuzzy:
            print("\n  Match type breakdown:", file=sys.stderr)
            for mt in ["exact", "partial", "fuzzy"]:
                c = total_match_stats.get(mt, 0)
                pct = (c * 100.0 / total_mentions) if total_mentions else 0.0
                print(f"    {mt}: {c} ({pct:.1f}%)", file=sys.stderr)

        # Summary categories (kept compatible with your CSV)
        print(f"\n{'='*70}", file=sys.stderr)
        print("MATCH SUMMARY", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        auto_matched_count = total_mentions
        print(f"\n✅ AUTO-MATCHED (Successfully inserted): {auto_matched_count:,}", file=sys.stderr)

        rejection_categories = {
            "policy": "Policy blocks (auto_match disabled, min_chars, numeric, etc.)",
            "not_eligible": "Not eligible (stopwords, single letters, etc.)",
            "case_mismatch": "Case mismatch (case policy enforcement)",
            "context_gate": "Context gate (requires_context not implemented)",
        }

        total_rejected = 0
        print(f"\n❌ REJECTED (Blocked by policy/rules):", file=sys.stderr)
        any_rej = False
        for cat, desc in rejection_categories.items():
            if cat in rejection_stats:
                n = sum(rejection_stats[cat].values())
                if n:
                    any_rej = True
                    total_rejected += n
                    print(f"   {cat}: {n:,} - {desc}", file=sys.stderr)
        if not any_rej:
            print("   None", file=sys.stderr)

        unresolved_categories = {
            "collision_high_value_enqueued": "High-value collisions (enqueued for review)",
            "collision_high_value_too_many": "High-value collisions (too many candidates/case mismatch)",
            "collision_dominance_none": "Collisions (no dominant candidate found)",
            "collision_harmless": "Harmless collisions (too common/ambiguous)",
        }

        total_unresolved = 0
        print(f"\n⚠️  UNRESOLVED (Requires review/adjudication):", file=sys.stderr)
        any_unres = False
        for cat, desc in unresolved_categories.items():
            if cat in rejection_stats:
                n = sum(rejection_stats[cat].values())
                if n:
                    any_unres = True
                    total_unresolved += n
                    print(f"   {cat}: {n:,} - {desc}", file=sys.stderr)
        if not any_unres:
            print("   None", file=sys.stderr)

        auto_resolved = sum(rejection_stats.get("collision_auto_resolved", {}).values())
        if auto_resolved:
            print(f"\n🔧 AUTO-RESOLVED COLLISIONS: {auto_resolved:,}", file=sys.stderr)
            print("   (included in auto-matched count above)", file=sys.stderr)

        print(f"\n{'─'*70}", file=sys.stderr)
        print("SUMMARY:", file=sys.stderr)
        print(f"  ✅ Auto-matched:     {auto_matched_count:>10,}", file=sys.stderr)
        print(f"  ❌ Rejected:         {total_rejected:>10,}", file=sys.stderr)
        print(f"  ⚠️  Unresolved:       {total_unresolved:>10,}", file=sys.stderr)
        if auto_resolved:
            print(f"  🔧 Auto-resolved:    {auto_resolved:>10,}", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        # Populate mention_review_queue from collision_queue items
        if collision_queue and not args.dry_run:
            try:
                inserted_count = populate_review_queue_from_collision_queue(conn, collision_queue)
                unique_count = len(set(
                    (item.get('chunk_id'), item.get('surface'), tuple(sorted(item.get('candidate_entity_ids', []))))
                    for item in collision_queue
                    if item.get('chunk_id') and item.get('surface') and item.get('candidate_entity_ids')
                ))
                print(f"\n📋 Populated {inserted_count} collision items into mention_review_queue (from {len(collision_queue)} total, {unique_count} unique)", file=sys.stderr)
            except Exception as e:
                # Don't fail the entire extraction if review queue population fails
                print(f"\n⚠️  Warning: Failed to populate mention_review_queue: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        # Write CSV
        if args.summary_csv:
            import csv
            csv_path = Path(args.summary_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load canonical names for all entities mentioned in stats
            entity_ids_for_canonical = set()
            for cat_stats in rejection_stats.values():
                if isinstance(cat_stats, dict):
                    for term in cat_stats.keys():
                        # Extract entity IDs from collision queue items
                        pass
            
            # Also get entity IDs from collision_queue
            for item in collision_queue:
                entity_ids_for_canonical.update(item.get("candidate_entity_ids", []))
            
            canonical_map = {}
            if entity_ids_for_canonical:
                with conn.cursor() as cur_canon:
                    cur_canon.execute("SELECT id, canonical_name FROM entities WHERE id = ANY(%s)", (list(entity_ids_for_canonical),))
                    canonical_map = {row[0]: row[1] for row in cur_canon.fetchall()}
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["category", "subcategory", "term", "count", "canonical_names", "description"])
                w.writerow(["auto_matched", "total", "", auto_matched_count, "", "Successfully inserted mentions"])

                # Optional breakdown: all auto-matched terms (surface_norm)
                if rejection_stats.get("auto_matched_terms"):
                    terms = rejection_stats["auto_matched_terms"]
                    w.writerow(["auto_matched", "terms", "[TOTAL]", sum(terms.values()), "", "Auto-matched terms (surface_norm)"])
                    for term, cnt in sorted(terms.items(), key=lambda x: x[1], reverse=True):
                        w.writerow(["auto_matched", "terms", term, cnt, "", "Auto-matched terms (surface_norm)"])

                for cat, desc in rejection_categories.items():
                    if cat in rejection_stats and rejection_stats[cat]:
                        total = sum(rejection_stats[cat].values())
                        w.writerow(["rejected", cat, "[TOTAL]", total, "", desc])
                        for term, cnt in sorted(rejection_stats[cat].items(), key=lambda x: x[1], reverse=True):
                            w.writerow(["rejected", cat, term, cnt, "", desc])

                for cat, desc in unresolved_categories.items():
                    if cat in rejection_stats and rejection_stats[cat]:
                        total = sum(rejection_stats[cat].values())
                        w.writerow(["unresolved", cat, "[TOTAL]", total, "", desc])
                        for term, cnt in sorted(rejection_stats[cat].items(), key=lambda x: x[1], reverse=True):
                            # Try to extract entity IDs from term if it's in collision_queue format
                            canonical_names_str = ""
                            if collision_queue:
                                for item in collision_queue:
                                    if item.get("alias_norm") == term or item.get("surface") == term:
                                        entity_ids = item.get("candidate_entity_ids", [])
                                        canonicals = [canonical_map.get(eid, f"entity_{eid}") for eid in entity_ids]
                                        canonical_names_str = "; ".join(canonicals)
                                        break
                            w.writerow(["unresolved", cat, term, cnt, canonical_names_str, desc])

                if auto_resolved:
                    w.writerow(["auto_resolved", "total", "[TOTAL]", auto_resolved, "Collisions automatically resolved"])
                    for term, cnt in sorted(rejection_stats.get("collision_auto_resolved", {}).items(), key=lambda x: x[1], reverse=True):
                        w.writerow(["auto_resolved", "total", term, cnt, "Collisions automatically resolved"])

                for mt in ["exact", "partial", "fuzzy"]:
                    w.writerow(["match_type", mt, "", total_match_stats.get(mt, 0), f"{mt} match type"])

                w.writerow(["summary", "auto_matched", "", auto_matched_count, "Total auto-matched mentions"])
                w.writerow(["summary", "rejected", "", total_rejected, "Total rejected mentions"])
                w.writerow(["summary", "unresolved", "", total_unresolved, "Total unresolved collisions"])
                if auto_resolved:
                    w.writerow(["summary", "auto_resolved", "", auto_resolved, "Total auto-resolved collisions"])

                if collision_queue:
                    w.writerow(["collision_queue", "high_value", "[TOTAL]", len(collision_queue), "", "High-value collisions enqueued for review"])
                    for item in collision_queue:
                        alias_norm = item.get("alias_norm", "")
                        surface = item.get("surface", "")
                        cands = item.get("candidate_entity_ids", [])
                        canonicals = [canonical_map.get(eid, f"entity_{eid}") for eid in cands]
                        canonical_names_str = "; ".join(canonicals)
                        w.writerow(["collision_queue", "high_value", f"{alias_norm} (surface: {surface})", 1, canonical_names_str, f"Candidates: {', '.join(map(str, cands))}"])

            print(f"\n📊 Match summary written to: {csv_path}", file=sys.stderr)

        # Samples
        if args.show_samples and all_samples:
            print(f"\nSample mentions (showing {len(all_samples)} of {total_mentions}):", file=sys.stderr)
            with conn.cursor() as cur:
                eids = [m["entity_id"] for m in all_samples]
                cur.execute("SELECT id, canonical_name, entity_type FROM entities WHERE id = ANY(%s)", (eids,))
                info = {i: (n, t) for (i, n, t) in cur.fetchall()}
            for i, m in enumerate(all_samples, 1):
                name, et = info.get(m["entity_id"], ("?", "?"))
                mt = m.get("match_type", "exact")
                conf = m.get("confidence", 1.0)
                tag = f"[{mt}]" if mt != "exact" else ""
                print(
                    f"  {i}. chunk_id={m['chunk_id']}, entity_id={m['entity_id']} ({name}, {et}), "
                    f"surface='{m['surface']}' {tag} conf={conf:.2f}",
                    file=sys.stderr
                )

        if args.dry_run:
            print("\nRun without --dry-run to actually insert mentions", file=sys.stderr)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
