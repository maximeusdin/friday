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
from retrieval.concept_expansion import expand_query_concepts, expand_concept

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


def get_conversation_history(conn, session_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch recent conversation messages for context.
    
    Returns messages in chronological order (oldest first).
    Truncates content to avoid prompt bloat.
    """
    with conn.cursor() as cur:
        # Check if research_messages table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'research_messages'
            )
        """)
        if not cur.fetchone()[0]:
            return []  # Table doesn't exist yet
        
        cur.execute("""
            SELECT role, content, plan_id, result_set_id, created_at
            FROM research_messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (session_id, limit))
        
        rows = cur.fetchall()
        if not rows:
            return []
        
        # Reverse to chronological order (oldest first)
        messages = []
        for row in reversed(rows):
            role, content, plan_id, result_set_id, created_at = row
            # Truncate content to avoid prompt bloat
            truncated = content[:200] + "..." if len(content) > 200 else content
            messages.append({
                "role": role,
                "content": truncated,
                "plan_id": plan_id,
                "result_set_id": result_set_id,
                "created_at": created_at.isoformat() if created_at else None,
            })
        
        return messages


def summarize_conversation_history(history: List[Dict[str, Any]], max_chars: int = 1000) -> str:
    """
    Create a summary of conversation history for the LLM prompt.
    
    Prioritizes recent messages and ensures we don't exceed max_chars.
    """
    if not history:
        return ""
    
    lines = []
    total_chars = 0
    
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        
        # Format based on role
        if role == "user":
            line = f"User: {content}"
        elif role == "assistant":
            line = f"Assistant: {content}"
        elif role == "system":
            line = f"[System: {content}]"
        else:
            line = f"{role}: {content}"
        
        # Check if adding this line would exceed max_chars
        if total_chars + len(line) + 1 > max_chars:
            # Add ellipsis and stop
            if lines:
                lines.insert(0, "... (earlier messages truncated)")
            break
        
        lines.append(line)
        total_chars += len(line) + 1
    
    return "\n".join(lines)

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
# Entity Resolution (Pre-LLM)
# =============================================================================

# Patterns for detecting potential entity names in utterances
# These help identify capitalized names, quoted strings, and common entity patterns
ENTITY_PATTERNS = [
    # Quoted names: "Ethel Rosenberg", 'CPUSA'
    r'"([^"]+)"',
    r"'([^']+)'",
    # Capitalized multi-word names: Ethel Rosenberg, Julius Rosenberg
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
    # Single capitalized words that could be names (but filter common words)
    r'\b([A-Z][a-z]{2,})\b',
    # All-caps acronyms: CPUSA, KGB, FBI
    r'\b([A-Z]{2,})\b',
]

# Common words to exclude from entity detection
ENTITY_STOPWORDS = {
    # Common English words that start capitalized
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "out", "up", "down",
    "off", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "now", "new", "old", "first", "last", "long",
    "great", "little", "good", "bad", "high", "low", "small", "large",
    # Query-related words
    "find", "search", "show", "get", "list", "display", "documents",
    "mentions", "results", "chunks", "pages", "evidence", "references",
    "collection", "session", "query", "plan", "filter", "within",
    # Common sentence starters
    "what", "who", "which", "whose", "whom", "this", "that", "these",
    "those", "and", "but", "or", "if", "while", "because", "although",
    "however", "therefore", "thus", "hence", "indeed", "moreover",
    # Document types
    "document", "report", "memo", "letter", "cable", "transcript",
    # Time-related
    "today", "yesterday", "tomorrow", "week", "month", "year", "date",
}


def extract_potential_entity_names(utterance: str) -> List[str]:
    """
    Extract potential entity names from the utterance using patterns.
    Returns a list of candidate name strings to look up.
    """
    candidates = []
    seen_lower = set()
    
    for pattern in ENTITY_PATTERNS:
        for match in re.finditer(pattern, utterance):
            name = match.group(1) if match.lastindex else match.group(0)
            name = name.strip()
            
            # Skip empty, too short, or stopwords
            if not name or len(name) < 2:
                continue
            
            name_lower = name.lower()
            if name_lower in ENTITY_STOPWORDS:
                continue
            
            # Skip if we've already seen this (case-insensitive dedup)
            if name_lower in seen_lower:
                continue
            
            seen_lower.add(name_lower)
            candidates.append(name)

    # Post-pass: avoid redundant single-token candidates when a longer phrase exists.
    #
    # Example: after clarification we may have "Julius Rosenberg" in the utterance.
    # The single-token pattern also matches "Rosenberg", which can re-trigger
    # ambiguity even though the longer phrase already disambiguates.
    multiword = [c for c in candidates if " " in c.strip()]
    if multiword:
        multiword_lowers = [f" {mw.lower()} " for mw in multiword]
        filtered: List[str] = []
        for c in candidates:
            c_stripped = c.strip()
            if " " not in c_stripped:
                token = f" {c_stripped.lower()} "
                if any(token in mw for mw in multiword_lowers):
                    # Drop redundant single token (covered by a longer candidate)
                    continue
            filtered.append(c)
        candidates = filtered

    return candidates


def resolve_entities_in_utterance(
    conn,
    utterance: str,
    *,
    best_guess_mode: Optional[bool] = None,
    confidence_threshold: float = 0.85,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Optional[Dict[str, Any]]]:
    """
    Pre-LLM entity resolution. Finds entity names in utterance
    and resolves them to entity_ids.
    
    Args:
        conn: Database connection
        utterance: User's query text
        best_guess_mode: If True, pick top candidate without clarification for ambiguous matches.
                        If None, uses ENTITY_BEST_GUESS env var (default: False).
        confidence_threshold: Minimum confidence delta to consider a match "clear".
                             If top match is this much better than second, use it without clarification.
    
    Returns:
        (entity_context, resolved_map, clarification_needed)
        - entity_context: list of dicts for LLM prompt with resolved entities (includes confidence)
        - resolved_map: {surface -> entity_id} for unambiguous matches
        - clarification_needed: dict with needs_clarification=True and choices if ambiguous
    """
    from retrieval.entity_resolver import entity_lookup, EntityLookupResult
    
    # Determine best-guess mode
    if best_guess_mode is None:
        best_guess_mode = os.getenv("ENTITY_BEST_GUESS", "").lower() in ("1", "true", "yes")
    
    entity_context: List[Dict[str, Any]] = []
    resolved_map: Dict[str, int] = {}
    
    # Extract potential entity names from utterance
    candidate_names = extract_potential_entity_names(utterance)
    
    for name in candidate_names:
        try:
            result: EntityLookupResult = entity_lookup(conn, name)
        except Exception as e:
            # If entity lookup fails (e.g., table doesn't exist), skip gracefully
            print(f"Warning: Entity lookup failed for '{name}': {e}", file=sys.stderr)
            continue
        
        # Get all matches sorted by confidence
        all_matches = sorted(
            result.exact_matches + result.near_matches,
            key=lambda c: c.confidence,
            reverse=True
        )
        
        if not all_matches:
            # No matches - LLM will handle as TERM/PHRASE
            continue
        
        # Check if we have a clear winner
        is_clear_match = False
        if len(all_matches) == 1:
            is_clear_match = True
        elif len(all_matches) >= 2:
            # Clear match if top confidence is significantly better than second
            conf_delta = all_matches[0].confidence - all_matches[1].confidence
            is_clear_match = conf_delta >= confidence_threshold
        
        if result.ambiguous and not is_clear_match:
            # Ambiguous - check if we should use best-guess mode
            if best_guess_mode:
                # Best-guess mode: use top candidate
                print(f"  Entity '{name}' ambiguous, using best guess: {all_matches[0].canonical_name} "
                      f"(confidence: {all_matches[0].confidence:.2f})", file=sys.stderr)
                best_match = all_matches[0]
                resolved_map[name] = best_match.entity_id
                entity_context.append({
                    "surface": name,
                    "entity_id": best_match.entity_id,
                    "canonical_name": best_match.canonical_name,
                    "entity_type": best_match.entity_type,
                    "confidence": best_match.confidence,
                    "match_method": best_match.match_method,
                    "is_best_guess": True,
                    "alternatives": [
                        {
                            "entity_id": c.entity_id,
                            "canonical_name": c.canonical_name,
                            "confidence": float(c.confidence),
                        }
                        for c in all_matches[1:3]  # Include top 2 alternatives
                    ],
                })
                continue
            
            # Need clarification
            choices = [
                f"{c.canonical_name} ({c.entity_type}, ID: {c.entity_id}, conf: {c.confidence:.2f})"
                for c in all_matches[:5]  # Limit to top 5 choices
            ]
            return [], {}, {
                "needs_clarification": True,
                "choices": choices,
                "query": {
                    "raw": utterance,
                    "primitives": []
                },
                "_clarification_context": {
                    "ambiguous_name": name,
                    "candidates": [
                        {
                            "entity_id": c.entity_id,
                            "canonical_name": c.canonical_name,
                            "entity_type": c.entity_type,
                            # Ensure JSON-serializable (EntityCandidate.confidence can be Decimal)
                            "confidence": float(c.confidence),
                        }
                        for c in all_matches[:5]
                    ],
                    "best_guess": {
                        "entity_id": all_matches[0].entity_id,
                        "canonical_name": all_matches[0].canonical_name,
                        "confidence": float(all_matches[0].confidence),
                    }
                }
            }
        
        # Single match or clear best match
        best_match = all_matches[0]
        resolved_map[name] = best_match.entity_id
        entity_context.append({
            "surface": name,
            "entity_id": best_match.entity_id,
            "canonical_name": best_match.canonical_name,
            "entity_type": best_match.entity_type,
            "confidence": best_match.confidence,
            "match_method": best_match.match_method,
        })
    
    return entity_context, resolved_map, None


# =============================================================================
# Prompt
# =============================================================================

def build_llm_prompt(
    utterance: str,
    session_summary: str,
    recent_result_sets: List[Dict[str, Any]],
    collections: List[Dict[str, str]],
    detected_deictics: Dict[str, bool],
    entity_context: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    date_context: List[Dict[str, Any]],
    query_lang_version: str = "qir_v1",
) -> str:
    allowed_primitives = [
        "TERM","PHRASE","WITHIN_RESULT_SET","EXCLUDE_RESULT_SET",
        "FILTER_COLLECTION","FILTER_DOCUMENT","FILTER_DATE_RANGE",
        "SET_TOP_K","SET_SEARCH_TYPE","SET_TERM_MODE","OR_GROUP",
        "TOGGLE_CONCORDANCE_EXPANSION",
        "ENTITY","CO_OCCURS_WITH","INTERSECT_DATE_WINDOWS","FILTER_COUNTRY",
        "CO_LOCATED","RELATION_EVIDENCE","REQUIRE_EVIDENCE","GROUP_BY",
        "COUNT","SET_QUERY_MODE",
        # Two-Mode Retrieval primitives
        "SET_RETRIEVAL_MODE","SET_SIMILARITY_THRESHOLD",
        "RELATED_ENTITIES","ENTITY_ROLE","EXCEPT_ENTITIES",
        # Index Retrieval primitives (entity)
        "FIRST_MENTION","FIRST_CO_MENTION","MENTIONS",
        # Index Retrieval primitives (date)
        "DATE_RANGE_FILTER","DATE_MENTIONS","FIRST_DATE_MENTION",
        # Index Retrieval primitives (place)
        "PLACE_MENTIONS","RELATED_PLACES","WITHIN_COUNTRY",
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

    # Entity context section
    entity_ctx = ""
    if entity_context:
        entity_ctx = "\nRESOLVED ENTITIES (use these entity_ids in ENTITY primitives):\n"
        for ent in entity_context:
            entity_ctx += f"  - \"{ent['surface']}\" -> entity_id={ent['entity_id']} ({ent['canonical_name']}, {ent['entity_type']})\n"
        entity_ctx += """
IMPORTANT: When the query mentions a resolved entity above, use the ENTITY primitive with the provided entity_id.
Example: For "find mentions of Ethel Rosenberg" where Ethel Rosenberg resolves to entity_id=42:
  {"type": "ENTITY", "entity_id": 42}
"""

    # Conversation history section
    conv_ctx = ""
    if conversation_history:
        conv_summary = summarize_conversation_history(conversation_history, max_chars=800)
        if conv_summary:
            conv_ctx = f"""
CONVERSATION HISTORY:
{conv_summary}

Note: The user may reference previous queries or results. Use this context to understand follow-up questions like "what about X?" or "tell me more about Y".
"""

    # Date context section
    date_ctx = ""
    if date_context:
        date_ctx = "\nRESOLVED DATES (use these in FILTER_DATE_RANGE primitives):\n"
        for d in date_context:
            if d.get("start") and d.get("end"):
                date_ctx += f"  - \"{d['expression']}\" -> start: {d['start']}, end: {d['end']}\n"
            elif d.get("start"):
                date_ctx += f"  - \"{d['expression']}\" -> start: {d['start']} (open-ended)\n"
            elif d.get("end"):
                date_ctx += f"  - \"{d['expression']}\" -> end: {d['end']} (open-ended)\n"
        date_ctx += """
IMPORTANT: When the query mentions dates resolved above, use FILTER_DATE_RANGE primitive with the resolved dates.
Example: For "documents from the 1940s" resolved to start: 1940-01-01, end: 1949-12-31:
  {"type": "FILTER_DATE_RANGE", "start": "1940-01-01", "end": "1949-12-31"}
"""

    deictic_warning = ""
    if detected_deictics:
        most_recent = recent_result_sets[0]["id"] if recent_result_sets else "N/A"
        deictic_warning = f"""
⚠️  DEICTIC DETECTED: references like "those results", "prev results", "earlier results", etc.
   MANDATORY: Include WITHIN_RESULT_SET primitive in your plan (result_set_id can be null/omitted).
   FORBIDDEN: Do NOT set needs_clarification=true - deictics are handled automatically by the system.
   The system will inject result_set_id {most_recent} automatically - just include WITHIN_RESULT_SET.
"""

    # Two-Mode Retrieval guidance
    mode_guidance = """
RETRIEVAL MODES:
You may emit SET_RETRIEVAL_MODE("conversational" | "thorough") to suggest a mode.
However, if the user has explicitly selected a mode via UI, that takes precedence.

When to suggest "thorough" mode:
- User says "everything", "exhaustive", "all mentions", "don't miss anything"
- User explicitly asks for comprehensive or complete results
- Query is about gathering complete evidence across all documents

When to suggest "conversational" mode (or omit the primitive):
- Typical questions: "tell me about X", "what happened when Y"
- User wants quick answers with explanations
- Query is exploratory rather than exhaustive

NEW PRIMITIVES AVAILABLE:

RELATED_ENTITIES(entity_id, window="document", top_n=20)
  - Analysis primitive: finds entities co-occurring with target
  - Use for: "who else appears with X", "related people", "associated organizations"
  
ENTITY_ROLE(role="person|org|place", entity_id=None)
  - Scope primitive: filter to entities of a specific type
  - Use for: "organizations mentioned", "places in these documents", "all people"
  - REQUIRED: role must be one of: "person", "org", "place", "event", "document"
  - DO NOT USE for general searches - only when explicitly filtering by entity type
  
EXCEPT_ENTITIES([entity_ids])
  - Scope primitive: exclude chunks mentioning these entities
  - Use for: "show me X but not Y", "exclude mentions of Z"

SET_SIMILARITY_THRESHOLD(threshold)
  - Control primitive: set minimum vector similarity [-1, 1]
  - Default: 0.35 (conversational), 0.25 (thorough)
  - Use for: tuning precision/recall tradeoff

INDEX PRIMITIVES (for deterministic mention lookups):

FIRST_MENTION(entity_id)
  - Index primitive: find chronologically first mention of an entity
  - Use for: "when was X first mentioned", "earliest reference to Y"
  - Returns deterministic results from entity_mentions index

FIRST_CO_MENTION(entity_ids)
  - Index primitive: find first time multiple entities appear together
  - Use for: "when were X and Y first mentioned together"
  - Requires at least 2 entity_ids

MENTIONS(entity_id)
  - Index primitive: find all mentions of an entity, paginated
  - Use for: "every instance of X", "all references to Y", "locate mentions of Z"
  - In conversational mode: returns first answer_k
  - In thorough mode: returns all, paginated

DATE_RANGE_FILTER(date_start, date_end)
  - Index primitive: filter to chunks with date mentions in range
  - Use for: "documents mentioning dates between 1944-1945"
  - time_basis: "mentioned_date" (dates in text) or "document_date" (doc metadata)

DATE_MENTIONS(date_start, date_end)
  - Index primitive: find chunks mentioning dates in range
  - Similar to DATE_RANGE_FILTER but orders by mentioned date

FIRST_DATE_MENTION(entity_id)
  - Index primitive: earliest dated mention of an entity
  - Use for: "when is X first dated in the documents"

PLACE_MENTIONS(place_entity_id)
  - Index primitive: find all mentions of a place
  - Use for: "mentions of Moscow", "references to Washington"

RELATED_PLACES(entity_id)
  - Analysis primitive: find places co-mentioned with an entity
  - Use for: "where is X mentioned", "locations associated with Y"

WITHIN_COUNTRY(country)
  - Scope primitive: filter to chunks mentioning places in a country
  - Use for: "mentions in the USSR", "documents about France"

WHEN TO USE INDEX vs SEARCH PRIMITIVES:
- Use INDEX primitives when:
  * User asks "when", "first", "earliest", "all instances", "locate"
  * Query is about specific entity mentions, not semantic similarity
  * Deterministic, reproducible results are required
  
- Use SEARCH primitives (TERM, PHRASE, ENTITY with search) when:
  * User asks "tell me about", "what happened", "summarize"
  * Query is exploratory or requires semantic understanding
  * Fuzzy matching and ranking by relevance are needed
"""

    return f"""Convert the user's natural language query into a structured query plan using primitives.

USER UTTERANCE:
"{utterance}"

SESSION CONTEXT:
{session_summary}
{rs_ctx}
{col_ctx}
{entity_ctx}
{date_ctx}
{conv_ctx}
{deictic_warning}
{mode_guidance}

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
- When entity names are resolved above, use ENTITY primitives with the provided entity_id.
- When dates are resolved above, use FILTER_DATE_RANGE primitives with the resolved start/end dates.
- Use conversation history to understand follow-up questions and contextual references.

IMPORTANT - KEEP PLANS SIMPLE:
- DO NOT add SET_TOP_K unless the user explicitly asks to limit results (e.g., "show me 10 results")
- DO NOT add ENTITY_ROLE unless filtering by entity type is explicitly requested (e.g., "show me only organizations")
- DO NOT add SET_RETRIEVAL_MODE unless user explicitly says "thorough" or "exhaustive"
- Most queries only need: PHRASE/TERM + optional FILTER_COLLECTION + optional ENTITY (for resolved names)
- When in doubt, use fewer primitives - let the system defaults handle the rest

CRITICAL - EXTRACT THE RIGHT TERMS:
- For questions like "is there information about X", the search should focus on X, not other words
- Technical terms and proper nouns (2+ words) should be PHRASE, not TERM: "proximity fuse", "atomic bomb", "spy ring"
- Do NOT drop important nouns in favor of common words like "Soviets", "Americans", "documents"
- If the query asks about a specific topic (weapon, person, event), that topic MUST be in the primitives
- Example: "Soviets getting the proximity fuse" → PHRASE("proximity fuse") is essential, "Soviets" is secondary

HARD CONSTRAINTS:
- If deictic warning is shown above, you MUST include WITHIN_RESULT_SET and MUST NOT set needs_clarification=true.
- If entities are resolved above, use ENTITY primitives with the provided entity_ids.
- If dates are resolved above, use FILTER_DATE_RANGE primitives with the resolved dates.
- Only set needs_clarification=true for genuine ambiguities that couldn't be resolved.
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
        "CO_LOCATED","RELATION_EVIDENCE","REQUIRE_EVIDENCE","GROUP_BY",
        "COUNT","SET_QUERY_MODE",
        # Two-Mode Retrieval primitives
        "SET_RETRIEVAL_MODE","SET_SIMILARITY_THRESHOLD",
        "RELATED_ENTITIES","ENTITY_ROLE","EXCEPT_ENTITIES",
        # Index Retrieval primitives (entity)
        "FIRST_MENTION","FIRST_CO_MENTION","MENTIONS",
        # Index Retrieval primitives (date)
        "DATE_RANGE_FILTER","DATE_MENTIONS","FIRST_DATE_MENTION",
        # Index Retrieval primitives (place)
        "PLACE_MENTIONS","RELATED_PLACES","WITHIN_COUNTRY",
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
        # Two-Mode Retrieval fields
        "mode": nullable_str,  # for SET_RETRIEVAL_MODE
        "threshold": {"type": ["number", "null"]},  # for SET_SIMILARITY_THRESHOLD
        "top_n": nullable_int,  # for RELATED_ENTITIES
        "role": nullable_str,  # for ENTITY_ROLE
        "entity_ids": {"type": ["array", "null"], "items": {"type": "integer"}},  # for EXCEPT_ENTITIES, FIRST_CO_MENTION
        # Index Retrieval fields
        "order_by": nullable_str,  # "chronological" | "document_order"
        "time_basis": nullable_str,  # "mentioned_date" | "document_date"
        "date_start": nullable_str,  # ISO date string for DATE_RANGE_FILTER
        "date_end": nullable_str,  # ISO date string for DATE_RANGE_FILTER
        "place_entity_id": nullable_int,  # for PLACE_MENTIONS
        "country": nullable_str,  # for WITHIN_COUNTRY
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
    """
    Validate and normalize the plan dict.
    
    Note: needs_clarification=True is NOT an error - it's a valid plan state
    that indicates the user needs to make a choice before the plan can be executed.
    """
    errors: List[str] = []
    if "query" not in plan_dict or not isinstance(plan_dict["query"], dict):
        return plan_dict, ["Missing 'query' field in plan"]
    if "primitives" not in plan_dict["query"]:
        return plan_dict, ["Missing 'query.primitives' field in plan"]
    
    # Don't fail on needs_clarification - it's a valid plan state
    # The backend/UI will handle prompting the user for clarification
    if plan_dict.get("needs_clarification"):
        # For clarification plans, skip primitive validation since they may be empty
        return plan_dict, []
    
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
    *,
    resolution_metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Save a plan to the database.
    
    Args:
        conn: Database connection
        session_id: Session ID
        plan: ResearchPlan object
        user_utterance: Original user query
        query_lang_version: Query language version
        resolution_metadata: Optional metadata about pre-LLM resolution:
            - resolved_dates: List of date expressions resolved to absolute ranges
            - resolved_entities: List of entity resolutions with confidence
            - resolution_timestamp: When resolution was performed
    """
    plan_dict = plan.to_dict()
    
    # Add resolution metadata if provided (for reproducibility)
    if resolution_metadata:
        plan_dict["_metadata"] = plan_dict.get("_metadata", {})
        plan_dict["_metadata"]["resolution"] = {
            **resolution_metadata,
            "resolved_at": __import__("datetime").datetime.now().isoformat(),
        }
    
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
        conversation_history = get_conversation_history(conn, session_id, limit=10)

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

        # Pre-LLM date resolution
        print("Resolving dates in utterance...", file=sys.stderr)
        try:
            from retrieval.date_parser import resolve_dates_in_utterance
            date_context, date_info = resolve_dates_in_utterance(args.text)
            if date_context:
                print(f"  Resolved {len(date_context)} date expressions:", file=sys.stderr)
                for d in date_context:
                    print(f"    - \"{d['expression']}\" -> {d.get('start', '?')} to {d.get('end', '?')}", file=sys.stderr)
        except ImportError as e:
            print(f"  Warning: Date parsing unavailable: {e}", file=sys.stderr)
            date_context = []
        except Exception as e:
            print(f"  Warning: Date parsing failed: {e}", file=sys.stderr)
            date_context = []
        
        # Pre-LLM entity resolution
        print("Resolving entities in utterance...", file=sys.stderr)
        entity_context, resolved_entities, clarification_needed = resolve_entities_in_utterance(
            conn, args.text
        )
        
        if entity_context:
            print(f"  Resolved {len(entity_context)} entities:", file=sys.stderr)
            for ent in entity_context:
                print(f"    - \"{ent['surface']}\" -> {ent['canonical_name']} (ID: {ent['entity_id']})", file=sys.stderr)
        
        # Concept expansion: expand conceptual phrases to entities and terms
        print("Expanding concepts in utterance...", file=sys.stderr)
        try:
            concept_primitives, expansion_notes = expand_query_concepts(
                conn, args.text, entity_context
            )
            if concept_primitives:
                print(f"  Expanded {len(concept_primitives)} concepts:", file=sys.stderr)
                if expansion_notes:
                    print(f"    {expansion_notes}", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Concept expansion failed: {e}", file=sys.stderr)
            concept_primitives = []
            expansion_notes = ""
        
        # If entity resolution requires clarification, return early
        if clarification_needed:
            print("Entity resolution requires clarification", file=sys.stderr)
            # Save as a clarification-needed plan
            clarification_needed["query"]["raw"] = args.text
            plan_dict = normalize_plan_dict(clarification_needed)
            
            if not args.dry_run:
                # Save the clarification plan directly
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
                        (session_id, Json(plan_dict), plan_hash, args.query_lang_version, "retrieval_v1", args.text),
                    )
                    plan_id = cur.fetchone()[0]
                    conn.commit()
                print(f"\n⚠️  Clarification needed. Plan saved with ID: {plan_id}", file=sys.stderr)
                print(json.dumps(plan_dict, indent=2))
            else:
                print("\n[DRY RUN] Clarification plan not saved to database", file=sys.stderr)
                print(json.dumps(plan_dict, indent=2))
            sys.exit(0)

        prompt = build_llm_prompt(
            args.text,
            session_summary,
            recent_result_sets,
            collections,
            detected_deictics,
            entity_context,
            conversation_history,
            date_context,
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

        # Inject concept-expanded primitives (entity + term fallbacks)
        prims = plan_dict.get("query", {}).get("primitives", [])
        
        # First, add TERM fallbacks for any PHRASE primitives (phrases can be too restrictive)
        phrase_terms_to_add = []
        for p in prims:
            if isinstance(p, dict) and p.get("type") == "PHRASE" and p.get("value"):
                phrase_value = p.get("value", "")
                # Extract significant words from phrase for term fallback
                words = phrase_value.split()
                for word in words:
                    word_clean = word.strip().lower()
                    # Keep words that are likely significant (3+ chars, not common words)
                    common_words = {"the", "and", "for", "about", "from", "with", "that", "this", "what", "how", "who", "when", "where", "why", "are", "was", "were", "been", "being", "have", "has", "had", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "there", "their", "they", "them", "these", "those", "some", "any", "all", "each", "every", "both", "few", "more", "most", "other", "such", "only", "same", "into", "over", "after", "before", "between", "under", "again", "then", "once", "here", "just", "also", "very", "even", "back", "well", "much", "still", "own", "see", "now", "get", "got", "getting", "soviets", "americans", "british", "information", "documents", "files"}
                    if len(word_clean) >= 3 and word_clean not in common_words:
                        phrase_terms_to_add.append(word_clean)
        
        # CRITICAL: Check if the query mentions important multi-word terms that the LLM may have missed
        # These are technical/historical terms that should be searched as phrases
        important_phrases = [
            "proximity fuse", "vt fuse", "radio fuse",
            "atomic bomb", "atomic energy", "manhattan project",
            "spy ring", "spy network", "espionage network",
            "white notebook", "black notebook", "yellow notebook",
            "code name", "cover name",
        ]
        query_lower = (plan_dict.get("query", {}).get("raw", "") or "").lower()
        for phrase in important_phrases:
            if phrase in query_lower:
                # Check if this phrase is already in primitives
                existing_phrases = {
                    (p.get("value") or "").lower() for p in prims 
                    if isinstance(p, dict) and p.get("type") == "PHRASE"
                }
                existing_terms = {
                    (p.get("value") or "").lower() for p in prims 
                    if isinstance(p, dict) and p.get("type") == "TERM"
                }
                # Add as both PHRASE and TERM for maximum coverage
                if phrase not in existing_phrases:
                    prims.append({"type": "PHRASE", "value": phrase})
                    print(f"  Added missing PHRASE: '{phrase}'", file=sys.stderr)
                # Also add individual words as terms
                for word in phrase.split():
                    if word not in existing_terms and len(word) >= 3:
                        prims.append({"type": "TERM", "value": word})
                        print(f"  Added TERM fallback: '{word}'", file=sys.stderr)
        
        existing_terms = {
            (p.get("value") or "").lower() for p in prims 
            if isinstance(p, dict) and p.get("type") == "TERM" and p.get("value")
        }
        
        for term in phrase_terms_to_add:
            if term not in existing_terms:
                prims.append({"type": "TERM", "value": term})
                existing_terms.add(term)
                print(f"  Added TERM fallback: '{term}' (from PHRASE)", file=sys.stderr)
        
        # Now add concept-expanded primitives
        if concept_primitives:
            # Merge expanded primitives, avoiding duplicates
            existing_entity_ids = {
                p.get("entity_id") for p in prims 
                if isinstance(p, dict) and p.get("type") in ("ENTITY", "RELATED_ENTITIES") and p.get("entity_id")
            }
            
            for cp in concept_primitives:
                if cp.get("type") in ("ENTITY", "RELATED_ENTITIES"):
                    if cp.get("entity_id") not in existing_entity_ids:
                        prims.append(cp)
                        existing_entity_ids.add(cp.get("entity_id"))
                elif cp.get("type") == "TERM":
                    term_lower = (cp.get("value") or "").lower()
                    if term_lower and term_lower not in existing_terms:
                        prims.append(cp)
                        existing_terms.add(term_lower)
                else:
                    prims.append(cp)
            
            plan_dict["query"]["primitives"] = prims
            print(f"  Injected {len(concept_primitives)} concept-expanded primitives", file=sys.stderr)
        else:
            plan_dict["query"]["primitives"] = prims
        
        # Ensure hybrid search is used by default for semantic queries
        # Only skip hybrid if explicitly set to "lex" or "index" or if using pure index primitives
        has_search_type = any(
            isinstance(p, dict) and p.get("type") == "SET_SEARCH_TYPE"
            for p in prims
        )
        has_index_only = all(
            isinstance(p, dict) and p.get("type") in (
                "FIRST_MENTION", "MENTIONS", "FIRST_CO_MENTION", 
                "DATE_MENTIONS", "PLACE_MENTIONS", "FILTER_COLLECTION",
                "FILTER_DOCUMENT", "SET_TOP_K", "SET_RETRIEVAL_MODE"
            )
            for p in prims if isinstance(p, dict) and p.get("type")
        )
        
        if not has_search_type and not has_index_only:
            # Default to hybrid search for all semantic queries
            prims.append({"type": "SET_SEARCH_TYPE", "value": "hybrid"})
            plan_dict["query"]["primitives"] = prims
            print("  Using hybrid search (vector + lexical) by default", file=sys.stderr)

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
            # Build resolution metadata for reproducibility
            resolution_metadata = {}
            if date_context:
                resolution_metadata["resolved_dates"] = [
                    {
                        "expression": d["expression"],
                        "start": d.get("start"),
                        "end": d.get("end"),
                    }
                    for d in date_context
                ]
            if entity_context:
                resolution_metadata["resolved_entities"] = [
                    {
                        "surface": e["surface"],
                        "entity_id": e["entity_id"],
                        "canonical_name": e["canonical_name"],
                        "confidence": e.get("confidence"),
                        "match_method": e.get("match_method"),
                        "is_best_guess": e.get("is_best_guess", False),
                        # Include alternatives for best-guess resolutions (for later review)
                        "alternatives": e.get("alternatives", []) if e.get("is_best_guess") else [],
                    }
                    for e in entity_context
                ]
                # Record resolution settings for reproducibility
                resolution_metadata["entity_resolution_settings"] = {
                    "best_guess_mode": os.getenv("ENTITY_BEST_GUESS", "").lower() in ("1", "true", "yes"),
                    "confidence_threshold": 0.85,  # Default threshold
                }
            
            plan_id = save_plan(
                conn, session_id, plan, args.text, args.query_lang_version,
                resolution_metadata=resolution_metadata if resolution_metadata else None,
            )
            print(f"\n✅ Plan saved with ID: {plan_id} (session={session_id}, status: proposed)", file=sys.stderr)
        else:
            print("\n[DRY RUN] Plan not saved to database", file=sys.stderr)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
