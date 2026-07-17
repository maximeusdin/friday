"""
V9 Follow-up Execution Path.

Answers questions using ONLY an existing evidence set — no retrieval tools.
Uses FTS join against chunks.text for full context.
Concordance-aware: expands aliases bidirectionally so follow-ups using
codenames or canonical names match the right evidence.
"""
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from retrieval.agent.v9_session import (
    search_evidence_set,
    _expand_query_for_evidence_search,
    RunRecord,
)


# =============================================================================
# Constants
# =============================================================================

_FOLLOWUP_MODEL = os.getenv("V9_FOLLOWUP_MODEL", "gpt-4.1-mini-2025-04-14")
_FOLLOWUP_MAX_CHUNKS = 20    # max evidence items to include in context
_FOLLOWUP_MAX_TOKENS = 2000  # max answer tokens


# =============================================================================
# Follow-up answer schema (structured output)
# =============================================================================

_FOLLOWUP_SCHEMA = {
    "name": "followup_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer based solely on the provided evidence.",
            },
            "cited_chunk_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Chunk IDs from the evidence set that support the answer.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "insufficient"],
                "description": (
                    "high: well-supported answer. "
                    "medium: partial support. "
                    "low: weak support. "
                    "insufficient: evidence set does not contain enough to answer."
                ),
            },
            "suggestion": {
                "type": "string",
                "description": (
                    "If confidence is low/insufficient, suggest next action: "
                    "'think_deeper' (resume search) or 'new_retrieval' (new query). "
                    "Empty string if confidence is high/medium."
                ),
            },
        },
        "required": ["answer", "cited_chunk_ids", "confidence", "suggestion"],
        "additionalProperties": False,
    },
}


_FOLLOWUP_SYSTEM = """\
You are a historical research assistant answering a follow-up question.

You have access ONLY to the evidence chunks provided below. You MUST NOT \
fabricate information or cite chunks not in the evidence set.

Rules:
1. Answer using ONLY the provided evidence chunks.
2. Cite sources using the document label in brackets, e.g. [Vassiliev p42] or [Venona p20].
   Do NOT mention chunk IDs in your answer text — use only document names and page numbers.
3. In cited_chunk_ids, list the chunk IDs that correspond to the sources you cite.
4. If the evidence is insufficient, set confidence to "insufficient" and \
   suggest "think_deeper" (to extend the previous search) or "new_retrieval" \
   (to start a new search).
5. Be concise but thorough. Include relevant quotes when helpful.
6. If the question asks about a specific passage, quote it directly."""


# =============================================================================
# Follow-up execution
# =============================================================================

def _build_alias_context(conn, evidence_set_id: int) -> str:
    """Build alias context for follow-up prompt from entities linked to evidence set.

    Queries evidence items → chunks → entity mentions to build a compact
    identity map. This helps the follow-up LLM understand that different
    name forms refer to the same person.

    Example output:
      "Known identities in this evidence set:
       PAL / Robert = Nathan Gregory Silvermaster
       LIBERAL = Julius Rosenberg"
    """
    try:
        with conn.cursor() as cur:
            # Find entity_ids that appear in this evidence set's chunks
            cur.execute("""
                SELECT DISTINCT e.id, e.canonical_name, array_agg(DISTINCT ea.alias) AS aliases
                FROM evidence_items ei
                JOIN chunks c ON c.id = ei.chunk_id
                JOIN entity_aliases ea ON c.text ILIKE '%%' || ea.alias || '%%'
                JOIN entities e ON e.id = ea.entity_id
                WHERE ei.evidence_set_id = %s
                  AND LENGTH(ea.alias) >= 3
                GROUP BY e.id, e.canonical_name
                LIMIT 10
            """, (evidence_set_id,))
            rows = cur.fetchall()

        if not rows:
            return ""

        parts = []
        for eid, canonical, aliases in rows:
            # Filter aliases to those actually different from canonical
            diff_aliases = [a for a in (aliases or []) if a and a.lower() != canonical.lower()]
            if diff_aliases:
                alias_str = " / ".join(diff_aliases[:4])
                parts.append(f"{alias_str} = {canonical}")
        if not parts:
            return ""
        return "\nKnown identities in this evidence set:\n" + "\n".join(f"  {p}" for p in parts[:8]) + "\n"
    except Exception:
        return ""  # best-effort


def execute_followup(
    conn,
    user_message: str,
    evidence_set_id: int,
    *,
    original_query: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute a follow-up query against an existing evidence set.

    No retrieval tools are called. Uses FTS join to chunks.text
    with bidirectional concordance expansion.

    Args:
        conn: database connection
        user_message: the follow-up question
        evidence_set_id: target evidence set to search within
        original_query: the original query that produced this evidence set
        verbose: log to stderr

    Returns:
        {
            "answer": str,
            "cited_chunk_ids": List[int],
            "confidence": str,
            "suggestion": str,
            "evidence_items_searched": int,
            "elapsed_ms": float,
        }
    """
    t0 = time.time()

    # Step 1: Search evidence set using FTS (with concordance expansion)
    # search_evidence_set now handles bidirectional alias expansion internally
    results = search_evidence_set(
        conn, evidence_set_id, user_message, limit=_FOLLOWUP_MAX_CHUNKS,
    )

    # Log expanded variants for debugging
    if verbose:
        variants = _expand_query_for_evidence_search(conn, user_message)
        print(
            f"  [V9 Follow-up] evidence_set={evidence_set_id}, "
            f"FTS results={len(results)}, "
            f"query_variants={variants}",
            file=sys.stderr,
        )

    if not results:
        # No matching evidence — return insufficient
        elapsed = (time.time() - t0) * 1000
        return {
            "answer": "I have no immediate answer, what would you like to do?",
            "cited_chunk_ids": [],
            "confidence": "insufficient",
            "suggestion": "new_retrieval",
            "evidence_items_searched": 0,
            "elapsed_ms": elapsed,
        }

    # Step 2: Build evidence context — use document labels (not chunk IDs)
    evidence_lines = []
    chunk_id_to_label: Dict[int, str] = {}
    for i, item in enumerate(results):
        chunk_id = item.get("chunk_id", 0)
        chunk_text = item.get("chunk_text", item.get("quote_text", ""))
        locators = item.get("locators_json", {})
        if isinstance(locators, str):
            try:
                locators = json.loads(locators)
            except Exception:
                locators = {}
        source = locators.get("source_label", locators.get("collection_slug", "")) or "unknown"
        source = str(source).replace("_", " ").title()
        page = locators.get("page", "")
        # Build human-readable label: "Vassiliev p42" or "Vassiliev" if no page
        page_num = ""
        if page:
            m = re.search(r"(\d+)", str(page))
            if m:
                page_num = f" p{m.group(1)}"
        label = f"{source}{page_num}".strip() or f"Source {chunk_id}"
        chunk_id_to_label[chunk_id] = label
        adjacency = " [context/adjacent]" if item.get("is_adjacency") else ""

        evidence_lines.append(
            f"--- {label}{adjacency} ---\n"
            f"{chunk_text}\n"
        )

    evidence_text = "\n".join(evidence_lines)

    # Step 3: Build alias context for the prompt
    alias_context = _build_alias_context(conn, evidence_set_id)

    # Step 4: Build prompt
    context_note = ""
    if original_query:
        context_note = f"\nOriginal search query: {original_query}\n"

    user_prompt = (
        f"{context_note}"
        f"{alias_context}"
        f"\n--- Evidence Set ({len(results)} chunks) ---\n\n"
        f"{evidence_text}\n\n"
        f"--- Follow-up Question ---\n{user_message}"
    )

    # Step 4: LLM call
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model=_FOLLOWUP_MODEL,
            messages=[
                {"role": "system", "content": _FOLLOWUP_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_completion_tokens=_FOLLOWUP_MAX_TOKENS,
            response_format={"type": "json_schema", "json_schema": _FOLLOWUP_SCHEMA},
        )
        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
            elapsed = (time.time() - t0) * 1000

            # Validate cited_chunk_ids against evidence set
            valid_cids = {item.get("chunk_id") for item in results}
            result["cited_chunk_ids"] = [
                cid for cid in result.get("cited_chunk_ids", [])
                if cid in valid_cids
            ]

            result["chunk_id_to_label"] = chunk_id_to_label
            result["evidence_items_searched"] = len(results)
            result["elapsed_ms"] = elapsed

            if verbose:
                print(
                    f"  [V9 Follow-up] confidence={result['confidence']}, "
                    f"cited={len(result['cited_chunk_ids'])}, "
                    f"elapsed={elapsed:.0f}ms",
                    file=sys.stderr,
                )
            return result
    except Exception as e:
        if verbose:
            print(f"  [V9 Follow-up] LLM error: {e}", file=sys.stderr)

    # Fallback
    elapsed = (time.time() - t0) * 1000
    return {
        "answer": "Unable to generate a follow-up answer. Please try a new retrieval.",
        "cited_chunk_ids": [],
        "confidence": "insufficient",
        "suggestion": "new_retrieval",
        "evidence_items_searched": len(results),
        "elapsed_ms": elapsed,
    }


# =============================================================================
# Verifier: follow-up invariants
# =============================================================================

def verify_followup_result(
    result: Dict[str, Any],
    evidence_set_id: int,
    conn,
) -> List[str]:
    """Verify follow-up invariants. Returns list of violations (empty = OK)."""
    violations = []

    # 1. All cited_chunk_ids must be in the target evidence set
    cited = result.get("cited_chunk_ids", [])
    if cited:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id FROM evidence_items
                WHERE evidence_set_id = %s AND chunk_id = ANY(%s)
            """, (evidence_set_id, cited))
            valid = {row[0] for row in cur.fetchall()}
            invalid = [cid for cid in cited if cid not in valid]
            if invalid:
                violations.append(
                    f"FOLLOW_UP cited chunks not in evidence set: {invalid}"
                )

    return violations
