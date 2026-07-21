"""
V9 Runner (V9.4) - Investigation Loop with Structured Outputs.

Uses OpenAI Structured Outputs (json_schema, strict:true) to guarantee
schema-conformant JSON on every model turn.  A single discriminated-union
schema branched on ``final`` replaces all manual JSON parsing, markdown
stripping, reissue logic, and recovery turns.

Two channels per model call:
- structured content  (always present, conforms to V9_OUTPUT_SCHEMA)
- optional tool_calls (execute + merge into workspace)

Counters: tool_calls_executed, model_turns (no reissues -- schema eliminates them).
"""
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from retrieval.agent.v9_types import (
    ResearchWorkspace,
    V9Synthesis,
    V9Claim,
    V9Result,
    GroundedClaim,
    WorkspaceChunk,
    WorkspaceEntity,
    EntityCandidate,
    CatalogHit,
    SufficiencyCheck,
    ScopeFilter,
    InvestigationState,
    InvestigationStep,
    WorkspaceDelta,
    ResponsivenessResult,
    ProgressSignal,
    RosterEntry,
    TimelineEntry,
    EvidenceEntry,
    RelationshipEdge,
    IdentityResolution,
)
from retrieval.agent.v9_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    V9_MODEL,
    V9_MAX_WORKSPACE_CHUNKS,
    V9_MAX_TOOL_CALLS,
)
from retrieval.agent.v9_context import (
    build_context_pack,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_CHUNK_CHAR_CAP,
    DEFAULT_SNIPPET_LEN,
    DEFAULT_MAX_CATALOG_ROWS,
    DEFAULT_MAX_FULLTEXT,
    _estimate_tokens,
)
from retrieval.agent.v9_workspace import (
    merge_search_result,
    merge_fetched_chunks,
    merge_catalog_hits,
    merge_entities,
    merge_entity_candidates,
    compute_progress_signal,
    append_note,
    apply_pin_suggestions,
    merge_evidence_summary_update,
    build_chunk_doc_map,
    link_chunks_to_entities,
    build_alias_context_for_summarizer,
    resolve_surfaced_alias,
    validate_hypotheses_for_entity,
)
from retrieval.agent.v9_summarize import summarize_delta_chunks
from retrieval.agent.v9_tools import (
    search_chunks,
    fetch_chunks,
    expand_entities,
    _load_catalog,
    resolve_question_entities,
    _confidence_from_matched_via,
)
from retrieval.agent.v9_pem_lane import pem_lane_seed_chunks as _pem_lane_seed_chunks
from retrieval.agent.v9_v10_resolver import resolve_keywords_via_v10_spans as _resolve_keywords_via_v10_spans
from retrieval.agent.tools import entity_surfaces_tool, entity_mentions_tool, _lookup_entity_by_name
from retrieval.agent.v9_grounding import ground_claims, ground_roster_entries
from retrieval.agent.v9_verify import build_verification_report


# =============================================================================
# Concordance: prime workspace with entity CANDIDATES (not committed entities)
# =============================================================================

def _prime_workspace_from_question(
    conn,
    question: str,
    workspace: ResearchWorkspace,
    content_keywords: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Resolve entity-like terms in the question via concordance and auto-expand
    high-confidence matches into the workspace.

    Two modes depending on whether a lightweight router provided content_keywords:

    A) With content_keywords (preferred):
       Only the router's extracted keywords are resolved — no blind extraction
       of every word (which wastes queries on "who", "was", etc. and can trigger
       pg_trgm errors that abort the transaction).

    B) Without content_keywords (fallback):
       Uses resolve_question_entities + direct alias reverse lookup (legacy path).

    In both cases, high-confidence candidates are auto-expanded: the entity's
    canonical name, aliases, and mention chunk IDs are loaded into the workspace
    so the model starts with evidence pointers ready to fetch.
    """
    # Ensure clean transaction before any DB work.
    # The caller (CLI, API) may have left the connection with an open or
    # aborted transaction from loading session data, etc.
    try:
        conn.rollback()
    except Exception:
        pass

    if content_keywords:
        # ---- Router-guided resolution: only look up the actual entities ----
        _resolve_keywords(conn, content_keywords, workspace, question=question, verbose=verbose)
    else:
        # ---- Legacy path: extract every word and try to resolve ----
        raw = resolve_question_entities(conn, question, scope=workspace.scope)
        if raw.get("candidates"):
            merge_entity_candidates(workspace, raw["candidates"])

        # Defensive rollback before Phase 2 (legacy path only)
        try:
            conn.rollback()
        except Exception:
            pass

        import re as _re
        resolved_terms = {c.query_term.lower() for c in workspace.entity_candidates}
        words = _re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,24}", question)
        for word in words:
            if word.lower() in resolved_terms or len(word) < 2:
                continue
            entity_id, canonical, etype, matched = _lookup_entity_by_name(
                conn, word, scope=workspace.scope
            )
            if entity_id:
                merge_entity_candidates(workspace, [
                    EntityCandidate(
                        query_term=word,
                        entity_id=entity_id,
                        canonical_name=canonical or "",
                        entity_type=etype,
                        matched_via=matched or "",
                        accepted=False,
                        confidence=_confidence_from_matched_via(matched or ""),
                        ambiguous=False,
                    )
                ])

    # Auto-expand high-confidence, non-ambiguous candidates
    _auto_expand_candidates(conn, workspace)


def _get_entity_mention_count(conn, entity_id: int) -> int:
    """Fast indexed count of entity_mentions for ranking disambiguation."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM entity_mentions WHERE entity_id = %s",
                (entity_id,),
            )
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def _resolve_keywords(
    conn,
    keywords: List[str],
    workspace: ResearchWorkspace,
    *,
    question: str = "",
    verbose: bool = True,
) -> None:
    """Resolve content keywords from the router. PEM is sole source (see docs/entity_resolution_pem_only.md).

    Resolution order:
    1. V10 span-based resolution (when PEM enabled) — uses PEM with prior_count
    2. _lookup_entity_by_name — PEM-only via page_entity_mentions

    No concordance, entity_aliases, or fuzzy fallback.
    """
    _v9_pem_enabled = os.getenv("V9_PEM_LANE_ENABLED", "1").strip().lower() in ("1", "true", "yes")

    # Step 0: V10 span-based resolution (when PEM enabled)
    v10_resolved: Dict[str, Tuple[int, str]] = {}
    if _v9_pem_enabled and question:
        scope_colls = workspace.scope.collections if workspace.scope and workspace.scope.collections else None
        v10_resolved = _resolve_keywords_via_v10_spans(
            conn, question, keywords,
            scope_collections=scope_colls,
        )
        for kw, (entity_id, canonical_name) in v10_resolved.items():
            merge_entity_candidates(workspace, [
                EntityCandidate(
                    query_term=kw,
                    entity_id=entity_id,
                    canonical_name=canonical_name or "",
                    entity_type=None,
                    matched_via="v10_span_lattice",
                    accepted=False,
                    confidence="exact",
                    ambiguous=False,
                )
            ])
            if verbose:
                print(
                    f"  [V9 Resolve] V10 span hit for '{kw}' -> {canonical_name} (id={entity_id})",
                    file=sys.stderr,
                )

    for kw in keywords:
        kw = kw.strip()
        if not kw or len(kw) < 2:
            continue
        if kw in v10_resolved:
            continue

        # Step 1: PEM-only via _lookup_entity_by_name
        try:
            conn.rollback()
        except Exception:
            pass

        try:
            entity_id, canonical, etype, matched = _lookup_entity_by_name(
                conn, kw, scope=workspace.scope
            )
            if entity_id:
                print(
                    f"  [V9 Resolve] PEM hit for '{kw}': entity_id={entity_id}, "
                    f"canonical='{canonical}', via={matched}",
                    file=sys.stderr,
                )
                merge_entity_candidates(workspace, [
                    EntityCandidate(
                        query_term=kw,
                        entity_id=entity_id,
                        canonical_name=canonical or "",
                        entity_type=etype,
                        matched_via=matched or "",
                        accepted=False,
                        confidence=_confidence_from_matched_via(matched or ""),
                        ambiguous=False,
                    )
                ])
            else:
                print(
                    f"  [V9 Resolve] No PEM hit for '{kw}'",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"  [V9 Resolve] PEM lookup raised for '{kw}': {e}",
                file=sys.stderr,
            )


def _fuzzy_entity_search(
    conn,
    term: str,
) -> Optional[EntityCandidate]:
    """Mention-count-ranked entity search with word decomposition.

    Collects ALL candidate entities across multiple matching strategies,
    then ranks by:  match_type_weight + 0.3 * log10(mention_count + 1)

    This ensures high-mention entities (e.g. "Nathan Gregory Silvermaster")
    beat low-mention ones (e.g. "Helen Silvermaster") when both match.

    Tries the full term first, then individual words if the full term
    yields no candidates (e.g. "Silvermaster network" → "Silvermaster").

    Returns the highest-scored EntityCandidate, or None.
    """
    import math

    if not term or len(term) < 2:
        return None

    term = term.strip()

    # Try full term first, then individual words
    terms_to_try = [term]
    words = [w.strip() for w in term.split() if len(w.strip()) > 2]
    skip_words = {"the", "and", "for", "with", "network", "group", "members",
                  "was", "were", "who", "what", "use", "from", "about"}
    for w in words:
        if w.lower() not in skip_words and w != term:
            terms_to_try.append(w)

    from retrieval.ops import concordance_expand_terms

    for try_term in terms_to_try:
        # Collect: (entity_id, canonical_name, entity_type, matched_via, weight)
        candidates = []

        try:
            with conn.cursor() as cur:
                # 1) Exact canonical name (weight 1.00)
                cur.execute(
                    "SELECT id, canonical_name, entity_type "
                    "FROM entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 5",
                    (try_term,),
                )
                for row in cur.fetchall():
                    candidates.append((row[0], row[1] or "", row[2] or "", "exact", 1.00))

                # 2) Exact alias (weight 0.85)
                cur.execute("""
                    SELECT DISTINCT e.id, e.canonical_name, e.entity_type
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE LOWER(ea.alias) = LOWER(%s)
                    LIMIT 5
                """, (try_term,))
                for row in cur.fetchall():
                    if not any(c[0] == row[0] for c in candidates):
                        candidates.append((row[0], row[1] or "", row[2] or "", "alias", 0.85))

                # 3) Concordance expansion (weight 0.75)
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    expanded = concordance_expand_terms(conn=conn, text=try_term, max_aliases_out=10)
                except Exception:
                    expanded = []
                for alias in expanded:
                    if alias.lower() == try_term.lower():
                        continue
                    with conn.cursor() as cur2:
                        cur2.execute(
                            "SELECT id, canonical_name, entity_type "
                            "FROM entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 5",
                            (alias,),
                        )
                        for row in cur2.fetchall():
                            if not any(c[0] == row[0] for c in candidates):
                                candidates.append((row[0], row[1] or "", row[2] or "",
                                                   f"concordance:{alias}", 0.75))
                        cur2.execute("""
                            SELECT DISTINCT e.id, e.canonical_name, e.entity_type
                            FROM entities e
                            JOIN entity_aliases ea ON ea.entity_id = e.id
                            WHERE LOWER(ea.alias) = LOWER(%s)
                            LIMIT 5
                        """, (alias,))
                        for row in cur2.fetchall():
                            if not any(c[0] == row[0] for c in candidates):
                                candidates.append((row[0], row[1] or "", row[2] or "",
                                                   f"concordance:{alias}", 0.75))

                # 4+5) Partial LIKE (weight 0.50) — only if no candidates yet
                if not candidates:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    with conn.cursor() as cur3:
                        cur3.execute("""
                            SELECT id, canonical_name, entity_type
                            FROM entities
                            WHERE LOWER(canonical_name) LIKE LOWER(%s)
                            ORDER BY LENGTH(canonical_name)
                            LIMIT 10
                        """, (f"%{try_term}%",))
                        for row in cur3.fetchall():
                            candidates.append((row[0], row[1] or "", row[2] or "", "partial", 0.50))

                        cur3.execute("""
                            SELECT e.id, e.canonical_name, e.entity_type
                            FROM entities e
                            JOIN entity_aliases ea ON ea.entity_id = e.id
                            WHERE LOWER(ea.alias) LIKE LOWER(%s)
                            GROUP BY e.id, e.canonical_name, e.entity_type
                            ORDER BY MIN(LENGTH(ea.alias))
                            LIMIT 10
                        """, (f"%{try_term}%",))
                        for row in cur3.fetchall():
                            if not any(c[0] == row[0] for c in candidates):
                                candidates.append((row[0], row[1] or "", row[2] or "",
                                                   "alias_partial", 0.50))

            if not candidates:
                continue  # try next term decomposition

            # Score all candidates by match weight + mention popularity
            scored = []
            try:
                conn.rollback()
            except Exception:
                pass
            with conn.cursor() as cur_mc:
                for eid, cname, etype, via, weight in candidates:
                    cur_mc.execute(
                        "SELECT COUNT(*) FROM entity_mentions WHERE entity_id = %s",
                        (eid,),
                    )
                    mc = cur_mc.fetchone()[0]
                    final_score = weight + 0.3 * math.log10(mc + 1)
                    scored.append((final_score, mc, eid, cname, etype, via))

            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0]

            print(
                f"  [V9 Resolve] _fuzzy ranked '{try_term}': "
                f"{best[3]} (id={best[2]}, mentions={best[1]}, "
                f"score={best[0]:.3f}, via={best[5]}, total_candidates={len(scored)})",
                file=sys.stderr,
            )
            if len(scored) > 1:
                for s in scored[:5]:
                    print(
                        f"    candidate: {s[3]} (id={s[2]}, mentions={s[1]}, "
                        f"score={s[0]:.3f}, via={s[5]})",
                        file=sys.stderr,
                    )

            return _candidate_from_best(
                {"entity_id": best[2], "canonical_name": best[3],
                 "entity_type": best[4], "score": best[0],
                 "matched_via": best[5]}, term,
            )

        except Exception as e:
            print(
                f"  [V9 Resolve] _fuzzy_entity_search error for '{try_term}': {e}",
                file=sys.stderr,
            )
            try:
                conn.rollback()
            except Exception:
                pass

    return None


def _candidate_from_best(best: dict, query_term: str) -> EntityCandidate:
    """Convert a best-match dict to an EntityCandidate."""
    score = best["score"]
    if score >= 0.85:
        confidence = "exact"
    elif score >= 0.7:
        confidence = "concordance"
    elif score >= 0.5:
        confidence = "partial"
    else:
        confidence = "inferred"

    return EntityCandidate(
        query_term=query_term,
        entity_id=best["entity_id"],
        canonical_name=best["canonical_name"],
        entity_type=best["entity_type"],
        matched_via=best["matched_via"],
        accepted=False,
        confidence=confidence,
        ambiguous=False,
    )


def _try_alias_reverse_lookup(
    conn,
    term: str,
    workspace: ResearchWorkspace,
) -> None:
    """Try to resolve a term via direct entity_aliases table lookup.

    Checks exact alias match (case-insensitive) and alias_norm match.
    """
    import re as _re

    already_resolved = {c.query_term.lower() for c in workspace.entity_candidates}
    if term.lower() in already_resolved:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.canonical_name, e.entity_type, ea.alias, ea.kind
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE LOWER(ea.alias) = LOWER(%s)
                LIMIT 3
            """, (term,))
            rows = cur.fetchall()

            if not rows:
                term_norm = _re.sub(r"[^a-z0-9 ]", "", term.lower()).strip()
                if term_norm:
                    cur.execute("""
                        SELECT e.id, e.canonical_name, e.entity_type, ea.alias, ea.kind
                        FROM entities e
                        JOIN entity_aliases ea ON ea.entity_id = e.id
                        WHERE ea.alias_norm = %s
                        LIMIT 3
                    """, (term_norm,))
                    rows = cur.fetchall()

            for row in rows:
                eid, canonical, etype, alias, kind = row
                if kind in ("primary", "code_name"):
                    confidence = "exact"
                elif kind in ("alt", "initials"):
                    confidence = "concordance"
                else:
                    confidence = "partial"

                merge_entity_candidates(workspace, [
                    EntityCandidate(
                        query_term=term,
                        entity_id=eid,
                        canonical_name=canonical or "",
                        entity_type=etype,
                        matched_via=f"alias_reverse:{alias}",
                        accepted=False,
                        confidence=confidence,
                        ambiguous=len(rows) > 1,
                    )
                ])
    except Exception:
        pass


def _auto_expand_candidates(
    conn,
    workspace: ResearchWorkspace,
    max_auto_expand: int = 3,
) -> None:
    """Auto-expand high-confidence, non-ambiguous entity candidates.

    For each qualifying candidate:
    1. Accept the candidate (promote to entity)
    2. Load canonical name + aliases via entity_surfaces_tool
    3. Load mention chunk IDs into the catalog
    4. Run concordance expansion to get all name forms

    This means the model starts with entities confirmed and mention chunks
    in the catalog, ready for fetch_chunks. No wasted tool call on
    expand_entities for obvious identity resolutions.
    """
    # Defensive: clear any aborted transaction state from priming queries.
    # All prior operations were SELECTs so nothing is lost.
    try:
        conn.rollback()
    except Exception:
        pass

    candidates_to_expand = [
        c for c in workspace.entity_candidates
        if not c.accepted
        and not c.ambiguous
        and c.confidence in ("exact", "concordance", "partial")
        and c.entity_id
    ]

    if not candidates_to_expand:
        return

    # Cap to avoid spending too long on priming
    candidates_to_expand = candidates_to_expand[:max_auto_expand]

    for candidate in candidates_to_expand:
        try:
            # Step 1-3: Use expand_entities (which internally calls entity_surfaces_tool
            # and entity_mentions_tool) to load full entity data
            raw = expand_entities(
                conn,
                entity_ids=[candidate.entity_id],
                include_mentions=True,
                include_comentions=False,
                mentions_top_k=30,  # bounded mention count
                scope=workspace.scope,
            )

            resolved_canonical = None
            db_aliases: List[str] = []

            for e_data in raw.get("entities", []):
                eid = e_data["entity_id"]
                # Accept the candidate
                workspace.accept_candidate(eid)
                resolved_canonical = e_data.get("canonical_name", "")
                db_aliases = e_data.get("aliases", [])
                # Merge full entity with aliases
                merge_entities(workspace, [
                    WorkspaceEntity(
                        entity_id=eid,
                        canonical_name=resolved_canonical,
                        aliases=db_aliases,
                        entity_type=e_data.get("entity_type"),
                    )
                ])

            # Load mention chunk IDs into catalog for the model to see
            mention_cids = raw.get("chunk_ids", [])
            if mention_cids:
                cat = _load_catalog(conn, mention_cids[:50], {})
                merge_catalog_hits(workspace, cat)

            # B: Do NOT inject concordance into operational aliases.
            # entity.aliases = db_aliases only (from entity_surfaces).
            # Retrieval uses PEM-backed surfaces; display can show entity_aliases.
            total_aliases = len(db_aliases)
            alias_preview = ", ".join(db_aliases[:8])
            print(
                f"  [V9 EntityLink] Auto-expanded '{candidate.query_term}' -> "
                f"'{resolved_canonical or candidate.canonical_name}' "
                f"(id={candidate.entity_id}, {len(mention_cids)} mentions, "
                f"{total_aliases} aliases: [{alias_preview}])",
                file=sys.stderr,
            )

        except Exception as e:
            print(
                f"  [V9] Auto-expand failed for '{candidate.query_term}': {e}",
                file=sys.stderr,
            )
            # Leave as pending candidate — model can still try expand_entities


# =============================================================================
# Token budget constants  (structured-output schema adds ~1000 token overhead)
# =============================================================================

TOOL_TURN_MAX_TOKENS = 300      # tool turns: only scratchpad + nulls (~150 tokens)
# final turns: full narrative + claims + artifact. Raised 4096->8000 because broad/roster
# queries over many chunks produced a large artifact that overflowed 4096, truncating the
# JSON and forcing a thin "minimal context" retry (observed: needle answers that HAD the
# evidence chunk still came back as false negatives). gpt-4.1-mini supports up to 16k output.
SYNTHESIS_MAX_TOKENS = int(os.getenv("V9_SYNTHESIS_MAX_TOKENS", "8000"))

# Maximum messages to keep in history (system + last N).
# Prevents unbounded message growth which kills 30k TPM budgets.
MAX_HISTORY_MESSAGES = 12

# Retry shrink levels for 429 TPM errors
_SHRINK_LEVELS = [
    # Level 0 (default)
    {"token_budget": 6000, "chunk_char_cap": DEFAULT_CHUNK_CHAR_CAP,
     "snippet_len": DEFAULT_SNIPPET_LEN, "max_catalog_rows": 20,
     "max_fulltext": 10, "max_completion_tokens": TOOL_TURN_MAX_TOKENS},
    # Level 1: halve catalog, shorter snippets
    {"token_budget": 4000, "chunk_char_cap": 800, "snippet_len": 60,
     "max_catalog_rows": 10, "max_fulltext": 6, "max_completion_tokens": 250},
    # Level 2: minimal context
    {"token_budget": 2500, "chunk_char_cap": 500, "snippet_len": 40,
     "max_catalog_rows": 6, "max_fulltext": 4, "max_completion_tokens": 200},
]


# =============================================================================
# Message history trimming
# =============================================================================

def _trim_messages(messages: List[Dict[str, Any]], max_keep: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, Any]]:
    """
    Keep system message + last ``max_keep`` non-system messages.

    Respects tool_call_id integrity: if a tool-result message is kept, its
    parent assistant message (with matching tool_calls) is kept too, even if
    that pushes us slightly over max_keep.
    """
    if len(messages) <= max_keep + 1:  # +1 for system
        return messages

    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]

    # Take the tail
    tail = rest[-max_keep:]

    # If the first message in tail is a tool result, we need its parent assistant message.
    # Walk backwards from the cut point to find the assistant message with tool_calls.
    cut = len(rest) - max_keep
    while tail and tail[0]["role"] == "tool" and cut > 0:
        cut -= 1
        tail = rest[cut:]

    return system + tail


# =============================================================================
# Deterministic scope detection
# =============================================================================

_COLLECTION_PATTERNS = {
    "vassiliev": ["vassiliev", "vassiliev notebooks", "vassiliev's notebooks"],
    "venona": ["venona", "venona decrypts", "venona cables"],
    "fbi": ["fbi", "fbi files", "fbi records"],
}

_SCOPE_TRIGGERS = re.compile(
    r"\b(?:only\s+from|only\s+in|from\s+the|citing\s+only|cite\s+only|exclusively\s+from|restrict\s+to)\b",
    re.IGNORECASE,
)

_DATE_RANGE_RE = re.compile(
    r"\b(?:between|from|during)\s+(\d{4})\s*(?:to|and|\u2013|-)\s*(\d{4})\b",
    re.IGNORECASE,
)

# scope: inline syntax, e.g. "scope:vassiliev" or "scope:collection=venona"
_INLINE_SCOPE_RE = re.compile(
    r"scope:(?:collection=)?(\w+)",
    re.IGNORECASE,
)


@dataclass
class ScopeDetectionResult:
    """Result of detect_scope_override_and_filters.

    Key semantic change: dates are NOT scope overrides. They are filters
    that always merge with whatever scope is active.
    - scope_override: collection/doc directives only. Competes with user selection.
    - filter_overrides: date_from/to only. Always merged regardless of scope source.
    """
    scope_override: Optional[ScopeFilter]   # collection/doc directives only
    filter_overrides: ScopeFilter            # date_from/to only (always merged)
    stripped_query: str                      # query with directives removed
    reason: str                             # human-readable explanation
    has_override: bool                      # True if scope_override is non-empty


# Full archive force triggers
_FULL_ARCHIVE_RE = re.compile(
    r"\b(?:full\s+archive|all\s+collections|entire\s+archive|whole\s+archive|every\s+collection)\b",
    re.IGNORECASE,
)


def detect_scope_override_and_filters(question: str) -> ScopeDetectionResult:
    """Separate scope overrides (collections/docs) from filter overrides (dates).

    Supports:
    1. Inline syntax: "scope:vassiliev" or "scope:collection=venona"
    2. Natural language triggers: "only from vassiliev", "restrict to venona"
    3. Full archive force: "full archive", "all collections"
    4. Date ranges: "between 1944 and 1946" (always filter, not override)
    """
    q_lower = question.lower()
    collections: List[str] = []
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    reasons: List[str] = []
    force_full_archive = False

    # Check for full archive force
    if _FULL_ARCHIVE_RE.search(question):
        force_full_archive = True
        reasons.append("contains full archive directive")

    # Mode 1: Inline scope: syntax
    if not force_full_archive:
        for m in _INLINE_SCOPE_RE.finditer(question):
            slug = m.group(1).lower()
            if slug in _COLLECTION_PATTERNS and slug not in collections:
                collections.append(slug)
                reasons.append(f"inline scope:{slug}")

        # Mode 2: Natural language triggers
        if not collections:
            for slug, patterns in _COLLECTION_PATTERNS.items():
                for pattern in patterns:
                    if pattern in q_lower:
                        if (_SCOPE_TRIGGERS.search(question)
                                or f"only {pattern}" in q_lower
                                or f"from {pattern}" in q_lower):
                            if slug not in collections:
                                collections.append(slug)
                                reasons.append(f"contains phrase '{pattern}'")

    # Date ranges (always filter, not override)
    dm = _DATE_RANGE_RE.search(question)
    if dm:
        date_from = dm.group(1)
        date_to = dm.group(2)

    # Build stripped query
    stripped = strip_scope_syntax(question)

    # Determine scope override
    scope_override: Optional[ScopeFilter] = None
    has_override = False

    if force_full_archive:
        # Explicit full archive = ScopeFilter() (empty = no restriction)
        scope_override = ScopeFilter()
        has_override = True
    elif collections:
        scope_override = ScopeFilter(collections=collections)
        has_override = True

    # Date filters are always separate (merge with any scope)
    filter_overrides = ScopeFilter(date_from=date_from, date_to=date_to)

    reason_str = "; ".join(reasons) if reasons else "no scope directives detected"

    return ScopeDetectionResult(
        scope_override=scope_override,
        filter_overrides=filter_overrides,
        stripped_query=stripped,
        reason=reason_str,
        has_override=has_override,
    )


def detect_scope(question: str) -> ScopeFilter:
    """Light deterministic parse for collection and date constraints.

    Legacy wrapper around detect_scope_override_and_filters.
    Returns a single ScopeFilter combining both overrides and filters.
    """
    result = detect_scope_override_and_filters(question)
    # Combine scope override and filter overrides into a single ScopeFilter
    collections = None
    if result.scope_override and result.scope_override.collections:
        collections = result.scope_override.collections
    return ScopeFilter(
        collections=collections,
        date_from=result.filter_overrides.date_from,
        date_to=result.filter_overrides.date_to,
    )


def strip_scope_syntax(question: str) -> str:
    """Remove scope: directives and full-archive phrases from the question text."""
    cleaned = _INLINE_SCOPE_RE.sub("", question)
    cleaned = _FULL_ARCHIVE_RE.sub("", cleaned)
    # Also strip scope trigger phrases that were matched
    cleaned = _SCOPE_TRIGGERS.sub("", cleaned)
    return cleaned.strip()


# =============================================================================
# Lightweight query parser (GPT-4o-mini)
# =============================================================================

V9_ROUTER_MODEL = os.getenv("V9_ROUTER_MODEL", "gpt-4.1-mini-2025-04-14")

_ROUTER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_parse",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "collections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document collection slug filters mentioned by user (venona, vassiliev, fbi, nsa). Empty array if none.",
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date filter in YYYY or YYYY-MM-DD. Empty string if none.",
                },
                "date_to": {
                    "type": "string",
                    "description": "End date filter in YYYY or YYYY-MM-DD. Empty string if none.",
                },
                "content_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Entity names, codenames, people, organizations, or specific "
                        "terms to look up in the concordance/entity database. "
                        "Proper nouns and codenames ONLY — no generic words."
                    ),
                },
                "intent": {
                    "type": "string",
                    "enum": [
                        "identity",
                        "timeline",
                        "roster",
                        "evidence",
                        "relationship",
                        "general",
                    ],
                },
                "reformulated_query": {
                    "type": "string",
                    "description": "Clean, precise restatement of the user's question.",
                },
            },
            "required": [
                "collections",
                "date_from",
                "date_to",
                "content_keywords",
                "intent",
                "reformulated_query",
            ],
            "additionalProperties": False,
        },
    },
}

_ROUTER_SYSTEM_PROMPT = """\
You are a query parser for a historical intelligence archive containing \
Venona decrypts, Vassiliev notebooks, FBI/NSA investigation files, and \
other declassified intelligence documents.

Extract structured metadata from the user's question.

## content_keywords (CRITICAL)

Extract ONLY the entity names, codenames, people names, organization names, \
or specific terms that should be looked up in the entity/concordance database. \
These are proper nouns, codenames (often ALL CAPS like PAL, LIBERAL, ALES), \
named groups (Silvermaster group), or organizations (GRU, NKVD, FBI).

Do NOT include generic words (who, what, when, tell, about, was, the, etc.). \
Do NOT include verbs, adjectives, or common English words.

Examples:
- "who was pal" -> ["PAL"]
- "tell me about the Silvermaster group" -> ["Silvermaster"]
- "what was the relationship between PAL and LIBERAL" -> ["PAL", "LIBERAL"]
- "when did the FBI investigate Alger Hiss" -> ["FBI", "Alger Hiss"]
- "list all Soviet agents in the Treasury Department" -> ["Treasury Department"]
- "what do the vassiliev notebooks say about Fuchs" -> ["Fuchs"]

If a codename is mentioned, always UPPERCASE it (PAL not pal, LIBERAL not liberal).

## collections

If the user specifies document collections to search, list their slugs:
- venona (Venona decrypts / cables)
- vassiliev (Vassiliev notebooks)
- fbi (FBI files)
- nsa (NSA files)
Empty array if the user does not restrict to specific collections.

## date_from, date_to

Extract date ranges if explicitly mentioned. YYYY or YYYY-MM-DD format. \
Empty string if not specified.

## intent

Classify the user's primary question intent:
- identity: who is X, what is codename X, identify a person
- timeline: when did X happen, chronological sequence of events
- roster: list of people/agents in a group, network, or organization
- evidence: find specific documents, quotes, primary source material
- relationship: how are X and Y connected, what is the link between X and Y
- general: broad or multi-faceted informational question

## reformulated_query

Rewrite the question to be precise and specific. Preserve the user's \
original intent but make it clear and unambiguous."""


@dataclass
class QueryParse:
    """Structured output from the lightweight query parser."""
    content_keywords: List[str]
    collections: List[str]
    date_from: str
    date_to: str
    intent: str
    reformulated_query: str


def _lightweight_parse_query(
    question: str,
    *,
    verbose: bool = True,
) -> QueryParse:
    """Use GPT-4o-mini to extract scope, content keywords, and intent.

    This runs BEFORE the investigation loop. The content_keywords are used
    for concordance expansion (instead of blindly extracting every word).
    The scope is merged with deterministic scope detection. The intent guides
    the investigation goal.

    Falls back to regex extraction if the API call fails.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_parse_query(question)

    t0 = time.time()
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=V9_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format=_ROUTER_SCHEMA,
            temperature=0,
            max_tokens=300,
        )

        parsed = json.loads(response.choices[0].message.content)
        result = QueryParse(
            content_keywords=parsed.get("content_keywords", []),
            collections=parsed.get("collections", []),
            date_from=parsed.get("date_from", ""),
            date_to=parsed.get("date_to", ""),
            intent=parsed.get("intent", "general"),
            reformulated_query=parsed.get("reformulated_query", question),
        )
        elapsed = (time.time() - t0) * 1000
        if verbose:
            print(
                f"  [V9 Router] {elapsed:.0f}ms | "
                f"keywords={result.content_keywords}, "
                f"intent={result.intent}, "
                f"collections={result.collections}",
                file=sys.stderr,
            )
        return result

    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        if verbose:
            print(
                f"  [V9 Router] Failed ({elapsed:.0f}ms): {e} — falling back to regex",
                file=sys.stderr,
            )
        return _fallback_parse_query(question)


def _fallback_parse_query(question: str) -> QueryParse:
    """Regex fallback when the router API call is unavailable.

    Extracts likely entity names by looking for:
    - ALL-CAPS words of 2+ chars (likely codenames: PAL, LIBERAL, ALES)
    - Title-case words of 3+ chars that aren't sentence-initial (likely names)
    - Quoted strings (explicit entity references)
    """
    keywords: List[str] = []

    # Quoted strings first
    for m in re.finditer(r'["\']([^"\']{1,50})["\']', question):
        token = m.group(1).strip()
        if token and token not in keywords:
            keywords.append(token)

    # ALL-CAPS words (likely codenames)
    for m in re.finditer(r'\b([A-Z][A-Z0-9]{1,24})\b', question):
        token = m.group(1)
        if token not in keywords:
            keywords.append(token)

    # Title-case words not at sentence start (likely proper names)
    words = question.split()
    for i, w in enumerate(words):
        clean = re.sub(r'[^A-Za-z]', '', w)
        if (
            clean
            and clean[0].isupper()
            and not clean.isupper()
            and len(clean) >= 3
            and i > 0
        ):
            if clean not in keywords:
                keywords.append(clean)

    return QueryParse(
        content_keywords=keywords,
        collections=[],
        date_from="",
        date_to="",
        intent="general",
        reformulated_query=question,
    )


# =============================================================================
# OpenAI tool definitions (unchanged)
# =============================================================================

TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "Find candidate evidence by searching the archive. Returns a *catalog* of hits "
                "(NOT full text): each hit includes chunk_id, score, doc_id, page, collection, "
                "and a ~300-char snippet.\n\n"
                "Snippets are truncated previews; you MUST call fetch_chunks to read or quote text.\n"
                "Results may include OCR noise; prefer multiple supporting chunks over a single hit.\n"
                "You can (and should) run multiple searches with different phrasings.\n\n"
                "Typical workflow: search_chunks -> pick 5-30 promising chunk_ids -> fetch_chunks.\n\n"
                "Query semantics: query is plain natural-language text. Words are tokenized, "
                "stemmed, and OR'd together with fuzzy spelling variants. Boolean operators "
                "(OR, AND, NOT) are NOT supported and will be treated as ordinary words. "
                "Codenames and aliases are auto-expanded via concordance (e.g. 'PAL' also matches "
                "'Silvermaster'). To find different facets, run separate searches with different "
                "phrasings rather than combining terms in one query.\n\n"
                "mode parameter:\n"
                "  'hybrid' (default): semantic + lexical hybrid search. Best for general queries.\n"
                "  'lexical_exact': exact substring match. IMPORTANT: pass a SINGLE name or codename "
                "(e.g. 'PAL', 'Silvermaster'). Do NOT use phrases or multiple words — they often return "
                "nothing. Run separate searches for each name. Best for alias mapping lines, "
                "index entries, or finding exact occurrences of a name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. For hybrid: natural-language text. "
                            "For lexical_exact: use a SINGLE name or codename only (phrases often return nothing)."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max catalog hits to return (default 50)",
                        "default": 50,
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional collection filter. If scope is set by the system, respect it.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "lexical_exact"],
                        "description": (
                            "Search mode. 'hybrid' (default): semantic+lexical. "
                            "'lexical_exact': exact substring match; query must be a SINGLE name or codename, "
                            "not a phrase."
                        ),
                        "default": "hybrid",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_chunks",
            "description": (
                "Load full text and metadata for evidence. Use after search_chunks or expand_entities.\n\n"
                "IMPORTANT: Use exactly ONE of these three calling modes:\n"
                "  Mode 1 - By chunk_ids: {\"chunk_ids\": [1,2,3]}  (neighbors +/-1 auto-included)\n"
                "  Mode 2 - Doc slice:    {\"doc_id\": 50, \"around_chunk_id\": 999, \"window\": 6}\n"
                "  Mode 3 - Doc pages:    {\"doc_id\": 50, \"page_start\": 5, \"page_end\": 7}\n\n"
                "Do NOT mix modes (e.g. do not pass chunk_ids together with doc_id).\n\n"
                "Returns full-text WorkspaceChunks with doc/page/collection metadata. "
                "Auto-fetched neighbor chunks are marked is_neighbor=true.\n\n"
                "Typical workflow: search_chunks -> review snippets -> fetch_chunks(chunk_ids=[...]) "
                "to load the most promising hits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Mode 1: Chunk IDs to load (neighbors +/-1 auto-included)",
                    },
                    "doc_id": {
                        "type": "integer",
                        "description": "Mode 2/3: Document ID",
                    },
                    "around_chunk_id": {
                        "type": "integer",
                        "description": "Mode 2: Center chunk ID within the document",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Mode 2: Total chunks to read around center (default 4)",
                        "default": 4,
                    },
                    "page_start": {
                        "type": "integer",
                        "description": "Mode 3: Start page number (inclusive)",
                    },
                    "page_end": {
                        "type": "integer",
                        "description": "Mode 3: End page number (inclusive)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_entities",
            "description": (
                "Resolve entity names or IDs to canonical entities, aliases, and evidence pointers.\n\n"
                "Input: provide either names=[...] or entity_ids=[...] (or both).\n\n"
                "IMPORTANT: This is the tool that CONFIRMS identity resolution. When the workspace "
                "shows 'Resolved identities (candidates)' with [PENDING] status, calling this tool "
                "with the entity_id or name will accept the candidate and load its evidence.\n\n"
                "Output includes (when available):\n"
                "- entities[]: each with entity_id, canonical_name, entity_type, aliases[]\n"
                "- mention_chunk_ids[]: chunks where the entity appears\n"
                "- co_entities[] (when include_comentions=true): entities that co-occur with the "
                "target in the same chunks, ranked by co-occurrence count\n"
                "- co_chunk_ids[]: chunks where co-occurring entities appear together\n"
                "- errors[]: names that could not be resolved\n\n"
                "Name resolution uses PEM (page_entity_mentions) only. E.g. 'PAL' resolves to "
                "'Silvermaster' if in PEM. See docs/entity_resolution_pem_only.md.\n\n"
                "Use cases:\n"
                "- Alias/codename resolution: resolve a surface form to a canonical entity.\n"
                "- Roster/network discovery: set include_comentions=true to find entities that "
                "co-occur with the target; then fetch_chunks on returned chunk_ids to extract "
                "supported members.\n\n"
                "Typical identity workflow: see candidate in workspace -> expand_entities(names=[...]) "
                "-> candidate becomes accepted -> fetch mention chunks -> find mapping text -> answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Entity IDs to expand (use IDs from candidates in workspace)",
                    },
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Names or surface forms to resolve (may match via aliases/codenames; "
                            "returns best match if ambiguous)"
                        ),
                    },
                    "include_mentions": {
                        "type": "boolean",
                        "description": "Return chunk IDs mentioning the resolved entities (default true)",
                        "default": True,
                    },
                    "include_comentions": {
                        "type": "boolean",
                        "description": (
                            "Return co-mentioned entities and supporting chunk IDs. "
                            "Essential for roster/network questions (default false)"
                        ),
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_search_result_set",
            "description": (
                "Summarize an exhaustive Search result set (from 'all instances' queries). "
                "Use when the user has run a concordance search and wants a summary of the results. "
                "Returns coverage, sample items, and a brief summary. "
                "The result_set_id comes from a previous 'all instances' response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result_set_id": {
                        "type": "string",
                        "description": "UUID of the search result set (from previous 'all instances' response)",
                    },
                },
                "required": ["result_set_id"],
            },
        },
    },
]


# =============================================================================
# Structured Outputs: discriminated union schema on ``final``
# =============================================================================

_IDENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "alias": {"type": "string"},
        "canonical": {"type": "string"},
        "entity_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "basis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "matched_via": {"type": "string"},
                    "surface": {"type": "string"},
                },
                "required": ["type", "matched_via", "surface"],
                "additionalProperties": False,
            },
        },
        "support_chunk_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["alias", "canonical", "entity_id", "basis", "support_chunk_ids"],
    "additionalProperties": False,
}

_ROSTER_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Real/canonical name when resolved. Use 'Name (Codename)' format if codename is relevant; never codename alone when real name is known.",
        },
        "role": {"type": "string"},
        "support_chunk_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["name", "role", "support_chunk_ids"],
    "additionalProperties": False,
}

_TIMELINE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "event": {"type": "string"},
        "support_chunk_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["date", "event", "support_chunk_ids"],
    "additionalProperties": False,
}

_EVIDENCE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "quote": {
            "type": "string",
            "description": "Evidence quote or summary. Use real names (Name (Codename)) when codenames were resolved; do not list codenames alone.",
        },
        "source": {"type": "string"},
        "page": {"type": "string"},
        "chunk_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    },
    "required": ["quote", "source", "page", "chunk_id"],
    "additionalProperties": False,
}

_RELATIONSHIP_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_a": {
            "type": "string",
            "description": "Real/canonical name when resolved; avoid codenames when mapping exists.",
        },
        "relation": {"type": "string"},
        "entity_b": {
            "type": "string",
            "description": "Real/canonical name when resolved; avoid codenames when mapping exists.",
        },
        "support_chunk_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["entity_a", "relation", "entity_b", "support_chunk_ids"],
    "additionalProperties": False,
}

_SCRATCHPAD_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "leads": {"type": "array", "items": {"type": "string"}},
        "hypotheses": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "next_actions": {"type": "array", "items": {"type": "string"}},
        "ready_to_synthesize": {"type": "boolean"},
        "pin_suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional: bullet_ids you want to keep pinned in memory",
        },
    },
    "required": ["goal", "leads", "hypotheses", "gaps", "next_actions", "ready_to_synthesize", "pin_suggestions"],
    "additionalProperties": False,
}

_EVIDENCE_SPAN_SCHEMA = {
    "type": "object",
    "properties": {
        "chunk_id": {"type": "integer"},
        "sentence_index": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "quote": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["chunk_id", "sentence_index", "quote"],
    "additionalProperties": False,
}

_CLAIMS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Claim text. Use real names (Name (Codename)) when codenames were resolved; do not list codenames alone.",
            },
            "confidence": {"type": "string"},
            "requires_citation": {"type": "boolean"},
            "citation_chunk_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Chunk IDs that directly support this claim. "
                    "Required for factual claims (requires_citation=true). "
                    "Use chunk IDs from your fetched fulltext or evidence bullets."
                ),
            },
            "evidence": {
                "type": "array",
                "items": _EVIDENCE_SPAN_SCHEMA,
                "description": (
                    "Optional span refs (chunk_id + sentence_index). "
                    "Pick 1-2 sentences max per claim. If authoritative, overrides citation_chunk_ids."
                ),
            },
        },
        "required": ["text", "confidence", "requires_citation", "citation_chunk_ids", "evidence"],
        "additionalProperties": False,
    },
}

_SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "argument": {"type": "string"},
        "remaining_gaps": {"type": "array", "items": {"type": "string"}},
        "next_best_actions_if_more_time": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sufficient", "argument", "remaining_gaps", "next_best_actions_if_more_time"],
    "additionalProperties": False,
}

_RESPONSIVENESS_SCHEMA = {
    "type": "object",
    "properties": {
        "addressed_question": {"type": "boolean"},
        "what_i_delivered": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "array", "items": {"type": "string"}},
        "why_missing": {"type": "string"},
    },
    "required": ["addressed_question", "what_i_delivered", "missing", "why_missing"],
    "additionalProperties": False,
}

_ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "identity":      {"anyOf": [_IDENTITY_SCHEMA, {"type": "null"}]},
        "roster":        {"anyOf": [{"type": "array", "items": _ROSTER_ITEM_SCHEMA}, {"type": "null"}]},
        "timeline":      {"anyOf": [{"type": "array", "items": _TIMELINE_ITEM_SCHEMA}, {"type": "null"}]},
        "evidence":      {"anyOf": [{"type": "array", "items": _EVIDENCE_ITEM_SCHEMA}, {"type": "null"}]},
        "relationships": {"anyOf": [{"type": "array", "items": _RELATIONSHIP_ITEM_SCHEMA}, {"type": "null"}]},
    },
    "required": ["identity", "roster", "timeline", "evidence", "relationships"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Discriminated union via nullable branches:
#   final=false (tool turn):  scratchpad populated, synthesis fields null
#   final=true  (final turn): scratchpad null, synthesis fields populated
# OpenAI strict mode requires root type=object; true anyOf union not supported.
# Instead, every branch field is anyOf [schema, null].
# ---------------------------------------------------------------------------

V9_OUTPUT_SCHEMA = {
    "name": "v9_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            # ---- Discriminator ----
            "final": {
                "type": "boolean",
                "description": (
                    "false while investigating (tool turns), true for final synthesis. "
                    "When false: populate scratchpad_update, set all other fields to null. "
                    "When true: populate narrative/claims/sufficiency/responsiveness/artifact, set scratchpad_update to null."
                ),
            },
            # ---- Planning branch (populate when final=false; null when final=true) ----
            "scratchpad_update": {
                "anyOf": [_SCRATCHPAD_SCHEMA, {"type": "null"}],
            },
            # ---- Synthesis branch (populate when final=true; null when final=false) ----
            "narrative":      {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "claims":         {"anyOf": [_CLAIMS_SCHEMA, {"type": "null"}]},
            "sufficiency":    {"anyOf": [_SUFFICIENCY_SCHEMA, {"type": "null"}]},
            "responsiveness": {"anyOf": [_RESPONSIVENESS_SCHEMA, {"type": "null"}]},
            "artifact":       {"anyOf": [_ARTIFACT_SCHEMA, {"type": "null"}]},
        },
        "required": [
            "final", "scratchpad_update", "narrative", "claims",
            "sufficiency", "responsiveness", "artifact",
        ],
        "additionalProperties": False,
    },
}

V9_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": V9_OUTPUT_SCHEMA,
}


# =============================================================================
# Tool execution (unchanged)
# =============================================================================

def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    conn,
    workspace: ResearchWorkspace,
    progress_callback: Optional[Any] = None,
    session_id: Optional[int] = None,
) -> Tuple[Dict[str, Any], str]:
    """Execute one tool, merge into workspace, return (result_dict, brief_summary)."""
    scope = workspace.scope

    # Defensive: clear any aborted transaction state before executing tool.
    # All prior operations in the investigation loop are SELECTs, so safe.
    try:
        conn.rollback()
    except Exception:
        pass

    try:
        if name == "search_chunks":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 50)
            collections = arguments.get("collections")
            mode = arguments.get("mode", "hybrid")

            # No alias expansion here — PEM lane and entity resolution already inject
            # alias-aware chunks. Use query as-is.
            result, catalog = search_chunks(
                conn, query=query, top_k=top_k,
                collections=collections, scope=scope,
                mode=mode,
                resolution_query=query,
            )
            merge_search_result(workspace, result, catalog, query)

            # Add concordance-resolved entities as CANDIDATES (with confidence)
            resolution = result.metadata.get("concordance_resolution") or []
            for r in resolution:
                eid = r.get("entity_id")
                if not eid:
                    continue
                matched_via = r.get("matched_via", "concordance/search")
                from retrieval.agent.v9_tools import _confidence_from_matched_via
                merge_entity_candidates(workspace, [
                    EntityCandidate(
                        query_term=r.get("query_term", ""),
                        entity_id=eid,
                        canonical_name=r.get("canonical_name", ""),
                        matched_via=matched_via,
                        accepted=False,
                        confidence=_confidence_from_matched_via(matched_via),
                    )
                ])

            out = {
                "tool": "search_chunks",
                "mode": mode,
                "count": len(catalog),
                "hits": [
                    {
                        "chunk_id": h.chunk_id, "score": round(h.score, 3),
                        "doc_id": h.doc_id, "page": h.page,
                        "collection": h.collection,
                        "snippet": h.snippet[:300],
                    }
                    for h in catalog[:60]
                ],
                "success": result.success,
                "error": result.error,
            }
            if resolution:
                out["concordance_resolution"] = resolution
            summary = f"query='{query[:60]}' mode={mode} -> {len(catalog)} hits"
            return out, summary

        if name == "fetch_chunks":
            chunk_ids = arguments.get("chunk_ids")
            doc_id = arguments.get("doc_id")
            around_chunk_id = arguments.get("around_chunk_id")
            window = arguments.get("window", 4)
            page_start = arguments.get("page_start")
            page_end = arguments.get("page_end")

            if doc_id and not around_chunk_id and page_start is None and page_end is None:
                return {"error": "doc_id requires either around_chunk_id (mode 2) or page_start/page_end (mode 3)", "tool": name}, "error: bad mode"
            if not chunk_ids and not doc_id:
                return {"error": "Provide chunk_ids (mode 1), or doc_id + around_chunk_id (mode 2), or doc_id + page_start + page_end (mode 3)", "tool": name}, "error: no args"

            chunks = fetch_chunks(
                conn,
                chunk_ids=chunk_ids,
                doc_id=doc_id,
                around_chunk_id=around_chunk_id,
                window=window,
                page_start=page_start,
                page_end=page_end,
            )

            # Scope enforcement: filter out chunks outside scope collections
            if scope and scope.collections:
                scope_set = {s.lower() for s in scope.collections}
                pre_count = len(chunks)
                chunks = [c for c in chunks if not c.source_label or c.source_label.lower() in scope_set]
                if len(chunks) < pre_count:
                    print(
                        f"  [V9] Scope filter: dropped {pre_count - len(chunks)} out-of-scope chunks from fetch",
                        file=sys.stderr,
                    )

            merge_fetched_chunks(workspace, chunks)

            # --- Entity linking: tag chunks with linked_entity_ids ---
            link_chunks_to_entities(workspace, chunks)

            # --- Summarize delta chunks into evidence memory ---
            delta_cids = [c.chunk_id for c in chunks if c.chunk_id not in workspace._summarized_chunk_ids]
            summarizer_note = ""
            if delta_cids:
                delta_chunks_for_summary = [c for c in chunks if c.chunk_id in set(delta_cids)]
                # Build alias context so summarizer uses canonical names
                alias_ctx = build_alias_context_for_summarizer(workspace)
                try:
                    ev_update = summarize_delta_chunks(
                        delta_chunks_for_summary, workspace.question,
                        alias_context=alias_ctx,
                    )
                    cdm = build_chunk_doc_map(workspace)
                    merge_evidence_summary_update(workspace, ev_update, cdm)
                    summarizer_note = f", summarized {len(delta_chunks_for_summary)} -> {len(ev_update.bullets)} bullets"
                    print(
                        f"  [V9] Summarizer: delta_chunks={len(delta_chunks_for_summary)}, "
                        f"out_bullets={len(ev_update.bullets)}",
                        file=sys.stderr,
                    )
                    # Emit evidence_update with actual bullet content
                    if progress_callback and ev_update.bullets:
                        chunk_to_page = {c.chunk_id: _parse_page_no(c.page) for c in workspace.fulltext_chunks if c.page}
                        all_doc_ids = list({did for b in ev_update.bullets for did in (b.doc_ids or [])})
                        doc_names = _get_doc_source_names(conn, all_doc_ids)
                        _emit_progress(progress_callback, "evidence_update", "completed",
                            f"Discovered {len(ev_update.bullets)} new evidence bullets",
                            {
                                "bullets": [
                                    {
                                        "text": b.text,
                                        "tags": b.tags,
                                        "chunk_ids": b.supporting_chunk_ids,
                                        "doc_ids": b.doc_ids,
                                        "pages": [chunk_to_page.get(cid) for cid in b.supporting_chunk_ids],
                                        "source_names": [doc_names.get(did, "") for did in (b.doc_ids or [])],
                                    }
                                    for b in ev_update.bullets
                                ],
                                "open_questions": ev_update.open_questions,
                                "leads": ev_update.leads,
                                "total_bullet_count": len(workspace._bullet_index),
                            })
                except Exception as e:
                    print(f"  [V9] Summarizer error: {e}", file=sys.stderr)

            out = {
                "tool": "fetch_chunks",
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text[:1500],
                        "source": c.source_label, "page": c.page,
                        "is_neighbor": c.is_neighbor,
                    }
                    for c in chunks
                ],
                "count": len(chunks),
            }
            summary = f"fetched {len(chunks)} chunks{summarizer_note}"
            return out, summary

        if name == "expand_entities":
            entity_ids = arguments.get("entity_ids")
            names = arguments.get("names")
            include_mentions = arguments.get("include_mentions", True)
            include_comentions = arguments.get("include_comentions", False)

            raw = expand_entities(
                conn,
                entity_ids=entity_ids,
                names=names,
                include_mentions=include_mentions,
                include_comentions=include_comentions,
                scope=scope,
            )

            for e in raw.get("entities", []):
                eid = e["entity_id"]
                canonical = e.get("canonical_name", "")
                db_aliases = e.get("aliases", [])
                etype = e.get("entity_type", "")
                workspace.accept_candidate(eid)

                # Use entity_aliases only. No concordance. PEM lane + entity resolution
                # drive retrieval; db_aliases for display.
                merge_entities(workspace, [
                    WorkspaceEntity(
                        entity_id=eid,
                        canonical_name=canonical,
                        aliases=list(db_aliases),
                        entity_type=etype,
                    )
                ])

                # Validate any alias hypotheses that point to this entity
                validated_count = validate_hypotheses_for_entity(
                    workspace, eid, reason="expand_entities"
                )
                if validated_count > 0:
                    print(
                        f"  [V9 AliasHyp] expand_entities validated {validated_count} "
                        f"hypotheses for entity {eid} ({canonical})",
                        file=sys.stderr,
                    )

                # Log entity linking result
                alias_str = ", ".join(db_aliases[:8]) if db_aliases else "(none)"
                print(
                    f"  [V9 EntityLink] {canonical} (id={eid}, type={etype}) "
                    f"aliases=[{alias_str}]",
                    file=sys.stderr,
                )

            mention_cids = raw.get("chunk_ids", [])
            if mention_cids:
                cat = _load_catalog(conn, mention_cids[:100], {})
                merge_catalog_hits(workspace, cat)

            co_cids = raw.get("co_chunk_ids", [])
            if co_cids:
                cat = _load_catalog(conn, co_cids[:100], {})
                merge_catalog_hits(workspace, cat)

            response: Dict[str, Any] = {
                "tool": "expand_entities",
                "entities": raw["entities"],
                "mention_chunk_ids_count": len(mention_cids),
                "errors": raw.get("errors", []),
            }
            if include_comentions:
                response["co_entities"] = raw.get("co_entities", [])[:30]
                response["co_chunk_ids_count"] = len(co_cids)

            ent_names = [e.get("canonical_name", "?") for e in raw.get("entities", [])]
            summary = f"resolved: {ent_names}, {len(mention_cids)} mentions"
            return response, summary

        if name == "summarize_search_result_set":
            result_set_id = arguments.get("result_set_id")
            if not result_set_id:
                return {"error": "result_set_id is required", "tool": name}, "error: missing result_set_id"

            # Ownership: verify result set belongs to current session (API layer already asserted session ownership)
            if session_id is None:
                return {
                    "error": "Ownership verification requires session context (use Chat API)",
                    "tool": name,
                }, "error: no session context"

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT srs.status, srs.total_hits, srs.coverage_json, srs.query_raw,
                           srs.mode, srs.is_exhaustive
                    FROM search_result_sets srs
                    WHERE srs.id = %s AND srs.session_id = %s
                    """,
                    (result_set_id, session_id),
                )
                row = cur.fetchone()
            if not row:
                return {
                    "error": "Search result set not found or not owned by this session",
                    "tool": name,
                }, "error: result set not found"

            status, total_hits, coverage_json, query_raw, mode, is_exhaustive = row
            if status != "complete":
                return {
                    "error": f"Search result set not ready (status={status})",
                    "tool": name,
                }, f"error: status {status}"

            # Fetch first 15 items for sample
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT h.snippet, col.slug, d.source_name, h.pdf_page_number
                    FROM search_result_page_hits h
                    JOIN collections col ON col.id = h.collection_id
                    JOIN documents d ON d.id = h.document_id
                    WHERE h.result_set_id = %s
                    ORDER BY h.collection_id, h.document_id, h.page_seq, h.page_id
                    LIMIT 15
                    """,
                    (result_set_id,),
                )
                rows = cur.fetchall()

            sample_items = []
            for r in rows:
                snippet, col_slug, doc_title, pdf_page = r
                sample_items.append({
                    "collection": col_slug or "",
                    "document": (doc_title or "")[:80],
                    "page": pdf_page,
                    "snippet": (snippet or "")[:200],
                })

            coverage = coverage_json or {}
            collections_breakdown = coverage.get("collections", [])
            summary_text = (
                f"Query: {query_raw or 'unknown'}. "
                f"Total: {total_hits or 0} page hits. "
                f"Mode: {mode or 'exact'}. "
                f"Exhaustive: {bool(is_exhaustive)}. "
            )
            if collections_breakdown:
                parts = [f"{c.get('slug', '?')}: {c.get('hits', 0)}" for c in collections_breakdown[:5]]
                summary_text += f"Collections: {', '.join(parts)}."

            response = {
                "tool": "summarize_search_result_set",
                "result_set_id": result_set_id,
                "query": query_raw,
                "total_hits": total_hits or 0,
                "is_exhaustive": bool(is_exhaustive),
                "mode": mode,
                "coverage": coverage,
                "sample_items": sample_items,
            }
            return response, summary_text[:200]

        return {"error": f"Unknown tool: {name}"}, f"error: unknown tool {name}"
    except Exception as e:
        return {"error": str(e), "tool": name}, f"error: {str(e)[:80]}"


# =============================================================================
# Investigation state update (from structured scratchpad_update)
# =============================================================================

def _update_investigation(investigation: InvestigationState, sp: dict) -> None:
    """Update investigation state from a scratchpad_update dict."""
    if "goal" in sp:
        investigation.goal = sp["goal"]
    if "leads" in sp:
        investigation.leads = sp["leads"] if isinstance(sp["leads"], list) else [sp["leads"]]
    if "hypotheses" in sp:
        investigation.hypotheses = sp["hypotheses"] if isinstance(sp["hypotheses"], list) else [sp["hypotheses"]]
    if "gaps" in sp:
        investigation.gaps = sp["gaps"] if isinstance(sp["gaps"], list) else [sp["gaps"]]
    if "next_actions" in sp:
        investigation.next_actions = sp["next_actions"] if isinstance(sp["next_actions"], list) else [sp["next_actions"]]
    if "ready_to_synthesize" in sp:
        investigation.ready_to_synthesize = bool(sp["ready_to_synthesize"])


# =============================================================================
# Content parsing  (trivial with Structured Outputs)
# =============================================================================

def _parse_content(content: Optional[str]) -> Optional[dict]:
    """Parse structured-output content.  Returns dict or None.
    Handles: empty, markdown-wrapped JSON, truncation."""
    if not content or not str(content).strip():
        return None
    s = str(content).strip()

    # Strip markdown code blocks (model may wrap even with response_format)
    for pattern in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
        m = re.search(pattern, s)
        if m:
            s = m.group(1).strip()
            break

    # Extract first { ... } if there's extra text (truncation, preamble)
    if "{" in s and "}" in s:
        start = s.index("{")
        depth = 0
        end = -1
        for i, c in enumerate(s[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > start:
            s = s[start : end + 1]

    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    # Repair truncated JSON (model hit max_completion_tokens mid-response)
    if s.startswith("{") and not s.rstrip().endswith("}"):
        try:
            repair = s.rstrip()
            # Close an open string (narrative is most likely to be truncated)
            if '"narrative":"' in repair and '","claims"' not in repair and '","sufficiency"' not in repair:
                if repair.endswith("\\"):
                    repair += '"'
                repair += '"'
            repair += ',"claims":[],"sufficiency":null,"responsiveness":null,"artifact":null}'
            data = json.loads(repair)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _build_artifact_dict(artifact: dict) -> Dict[str, Any]:
    """Strip null-valued keys from the artifact so downstream code sees a clean dict."""
    return {k: v for k, v in artifact.items() if v is not None} if artifact else {}


def _build_minimal_synthesis_context(
    workspace: ResearchWorkspace,
    question: str,
    scope_note: str,
) -> str:
    """Build a minimal context for synthesis retry when structured output fails.

    Strips catalog hits and fulltext chunk bodies to reduce token pressure.
    Keeps: question, scope, investigation state, evidence memory view only.
    """
    from retrieval.agent.v9_context import select_evidence_memory_view, _render_bullet

    parts = [f"Question: {question}\n"]
    if scope_note:
        parts.append(f"{scope_note}\n")

    inv = workspace.investigation
    parts.append("Investigation state:\n")
    if inv.goal:
        parts.append(f"  Goal: {inv.goal}\n")
    if inv.gaps:
        parts.append(f"  Gaps: {inv.gaps}\n")
    parts.append(f"  Ready to synthesize: {inv.ready_to_synthesize}\n")

    # Evidence memory view (the key context)
    if workspace.evidence_memory:
        view = select_evidence_memory_view(workspace, question, inv.gaps)
        if view.pinned_bullets:
            parts.append(f"\n== Pinned Evidence ({len(view.pinned_bullets)}) ==\n")
            for b in view.pinned_bullets:
                parts.append(_render_bullet(b))
        if view.recent_bullets:
            parts.append(f"\n== Recent Evidence ({len(view.recent_bullets)}) ==\n")
            for b in view.recent_bullets:
                parts.append(_render_bullet(b))
        if view.top_relevant_bullets:
            parts.append(f"\n== Relevant Evidence ({len(view.top_relevant_bullets)}) ==\n")
            for b in view.top_relevant_bullets:
                parts.append(_render_bullet(b))

    # Entities (compact)
    if workspace.entities:
        parts.append("\nEntities:\n")
        for e in workspace.entities[-10:]:
            aliases = ", ".join(e.aliases[:3]) if e.aliases else ""
            parts.append(f"  {e.canonical_name} (id={e.entity_id}) {aliases}\n")

    # Counts only (no bodies)
    parts.append(f"\nWorkspace: {len(workspace.fulltext_chunks)} chunks loaded, "
                 f"{len(workspace.catalog_hits)} catalog hits.\n")

    parts.append(
        "\nSynthesize your final answer now. Set final=true. "
        "Every factual claim must include chunk_ids for grounding. "
        "If you cannot ground a claim, note it in sufficiency.remaining_gaps.\n"
    )

    return "".join(parts)


def _build_needs_more_evidence_synthesis(
    workspace: ResearchWorkspace,
    question: str,
) -> V9Synthesis:
    """Build a safe fallback synthesis when structured output fails entirely.

    Instead of freeform narrative (hallucination-prone), returns a structured
    response based on evidence memory + recommended next actions.
    """
    # Gather top bullets with their chunk_ids
    bullet_summaries = []
    all_support_chunks: List[int] = []
    for bid in workspace.pinned_bullet_ids[:5]:
        b = workspace._bullet_index.get(bid)
        if b:
            bullet_summaries.append(f"- {b.text} (chunks: {b.supporting_chunk_ids})")
            all_support_chunks.extend(b.supporting_chunk_ids[:3])

    # Fill from recent if not enough pinned
    if len(bullet_summaries) < 5 and workspace.evidence_memory:
        for update in reversed(workspace.evidence_memory[-3:]):
            for b in update.bullets:
                if len(bullet_summaries) >= 5:
                    break
                if b.bullet_id not in set(workspace.pinned_bullet_ids[:5]):
                    bullet_summaries.append(f"- {b.text} (chunks: {b.supporting_chunk_ids})")
                    all_support_chunks.extend(b.supporting_chunk_ids[:3])

    narrative = (
        f"Investigation incomplete — structured output failed during synthesis.\n\n"
        f"Question: {question}\n\n"
    )
    if bullet_summaries:
        narrative += "Key findings from evidence memory:\n" + "\n".join(bullet_summaries) + "\n\n"
    narrative += (
        "These findings are summary-derived and need chunk-level verification. "
        "Re-run the query or fetch the supporting chunks listed above for grounded answers."
    )

    return V9Synthesis(
        final=True,
        narrative=narrative,
        claims=[],
        sufficiency=SufficiencyCheck(
            sufficient=False,
            argument="Structured output failed; findings are ungrounded summaries.",
            remaining_gaps=["All claims need chunk-level verification"],
        ),
    )


# =============================================================================
# Finalization validation  (Layer A mostly handled by schema; Layer B remains)
# =============================================================================

def _normalize_for_match(s: str) -> str:
    """Normalize text for Jaccard / substring matching."""
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two normalized strings."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _check_summary_derived_claims(
    synthesis: V9Synthesis,
    workspace: ResearchWorkspace,
    grounded_claim_chunk_ids: dict,  # claim_text -> set of citation_chunk_ids
) -> List[str]:
    """Check if any factual claims appear to be derived from evidence memory
    bullets without proper chunk-level grounding."""
    issues: List[str] = []
    if not workspace._bullet_index:
        return issues

    for claim in synthesis.claims:
        if not claim.requires_citation:
            continue
        claim_norm = _normalize_for_match(claim.text)
        if not claim_norm:
            continue

        cited = grounded_claim_chunk_ids.get(claim.text, set())
        if cited:
            continue  # already grounded, no issue

        for bid, bullet in workspace._bullet_index.items():
            bullet_norm = _normalize_for_match(bullet.text)
            if not bullet_norm:
                continue
            # Check substring or high Jaccard
            is_match = (
                bullet_norm in claim_norm
                or _jaccard_similarity(claim_norm, bullet_norm) >= 0.7
            )
            if is_match:
                issues.append(
                    f"Claim appears summary-derived (matches bullet B:{bid}); "
                    f"fetch a supporting chunk before finalizing."
                )
                break  # one issue per claim is enough

    return issues


def _validate_finalization(
    synthesis: V9Synthesis,
    question: str,
    tool_calls_executed: int,
    max_tool_calls: int,
    workspace: Optional[ResearchWorkspace] = None,
) -> Tuple[bool, List[str]]:
    """
    Two-layer finalization validation.

    Layer A (hard, schema): With Structured Outputs most structural checks
    are guaranteed.  We still check semantic requirements the schema can't
    enforce (non-empty narrative, non-null sufficiency on final turns, etc.).

    Layer B (soft, model-owned responsiveness): can trigger continuation.

    Returns (valid, issues).
    """
    issues: List[str] = []
    budget_left = tool_calls_executed < max_tool_calls - 2

    # --- Layer A: semantic checks the schema can't enforce ---
    if not synthesis.narrative:
        issues.append("narrative is empty")

    if not synthesis.sufficiency:
        issues.append("sufficiency is null (required on final turns)")
    else:
        if not synthesis.sufficiency.argument:
            issues.append("sufficiency.argument is empty")

    resp = synthesis.responsiveness
    if not resp or not isinstance(resp, dict):
        issues.append("responsiveness is null (required on final turns)")
    else:
        # Delivery consistency check
        if resp.get("addressed_question") is True:
            has_artifact = bool(synthesis.artifact)
            has_delivery_explain = (
                bool(resp.get("what_i_delivered"))
                and isinstance(resp.get("why_missing"), str)
                and resp.get("why_missing") != ""
            )
            if not has_artifact and not has_delivery_explain:
                issues.append(
                    "addressed_question=true requires either non-empty artifact "
                    "or both what_i_delivered and why_missing"
                )

    if issues:
        return False, issues

    # --- Layer B: soft responsiveness ---
    if resp and resp.get("addressed_question") is False and budget_left:
        return False, ["model reports not responsive; continue searching"]

    if (synthesis.sufficiency
            and synthesis.sufficiency.sufficient
            and resp
            and resp.get("addressed_question") is True
            and not synthesis.artifact
            and not resp.get("what_i_delivered")):
        return False, ["sufficient=true, addressed=true, but nothing delivered"]

    if os.getenv("V9_USE_PATTERN_RESPONSIVENESS") == "1":
        pass

    # --- Layer C: ungrounded claim check ---
    # Any factual claim (requires_citation=true) MUST have citation_chunk_ids OR evidence.
    ungrounded = []
    for claim in synthesis.claims:
        has_provenance = (claim.citation_chunk_ids or
                         (claim.evidence and any(e.chunk_id for e in claim.evidence)))
        if claim.requires_citation and not has_provenance:
            ungrounded.append(claim.text[:80])

    if ungrounded and budget_left:
        ungrounded_str = "; ".join(ungrounded[:3])
        return False, [
            f"{len(ungrounded)} factual claim(s) missing provenance: {ungrounded_str}. "
            "Every requires_citation=true claim must include citation_chunk_ids or evidence (chunk_id)."
        ]

    # --- Layer C2: grounded content requirement (narrative + roster must be grounded) ---
    # When addressed_question=true, we need at least one grounded claim OR one grounded roster entry.
    # Narrative is only shown when we have grounded claims; roster entries need valid support_chunk_ids.
    if resp and resp.get("addressed_question") is True and budget_left:
        has_grounded_claim = any(
            c.citation_chunk_ids or (c.evidence and any(e.chunk_id for e in c.evidence))
            for c in synthesis.claims if c.requires_citation
        )
        roster = synthesis.get_roster()
        has_grounded_roster = any(
            r.support_chunk_ids for r in roster
        )
        if not has_grounded_claim and not has_grounded_roster:
            return False, [
                "addressed_question=true requires grounded content: add citation_chunk_ids or evidence to claims "
                "and/or support_chunk_ids to roster entries. Narrative and roster must be backed by evidence."
            ]
        # If roster has entries but none have support_chunk_ids, require them
        if roster and not has_grounded_roster:
            return False, [
                "Roster entries must include support_chunk_ids (chunk IDs from fetched evidence). "
                "Each member must cite at least one chunk that supports their identification."
            ]

    # --- Layer C3: anti-give-up — an "insufficient, couldn't find it" final with most of
    # the tool budget unused is a premature surrender, not an answer. Push the model back
    # into the search loop with concrete reformulation guidance (the record-language
    # rewrite that recovers vocabulary-mismatch failures like the Morris Childs case).
    if (synthesis.sufficiency
            and not synthesis.sufficiency.sufficient
            and tool_calls_executed < max_tool_calls - 4):
        _has_grounded = any(
            c.citation_chunk_ids or (c.evidence and any(e.chunk_id for e in c.evidence))
            for c in synthesis.claims
        )
        if not _has_grounded:
            return False, [
                "You reported insufficient evidence with NO grounded claims, but most of the "
                "tool budget remains — do not give up yet. Reformulate in the archive's own "
                "record language (add words like memorandum, report, teletype, statement, "
                "'initial contact', informant, plus the key names), run search + search_lexical "
                "on the rarest name, THEN finalize with whatever you find."
            ]

    # --- Layer D: summary-derived claim check (evidence memory grounding nudge) ---
    if workspace and workspace._bullet_index and budget_left:
        # Build map of which claims are already grounded by citation_chunk_ids or evidence
        grounded_map: Dict[str, set] = {}
        for claim in synthesis.claims:
            cids = set(claim.citation_chunk_ids or [])
            if claim.evidence:
                cids.update(e.chunk_id for e in claim.evidence if e.chunk_id)
            if cids:
                grounded_map[claim.text] = cids
        nudge_issues = _check_summary_derived_claims(synthesis, workspace, grounded_map)
        if nudge_issues:
            return False, nudge_issues[:2]  # cap to avoid overwhelming the model

    return True, []


# =============================================================================
# Delta computation (unchanged)
# =============================================================================

def _snapshot_counts(workspace: ResearchWorkspace) -> Dict[str, int]:
    return {
        "catalog": len(workspace.catalog_hits),
        "fulltext": len(workspace.fulltext_chunks),
        "entities": len(workspace.entities),
        "candidates": len(workspace.entity_candidates),
    }


def _compute_delta(
    workspace: ResearchWorkspace,
    prev_counts: Dict[str, int],
    tools_called: List[str],
) -> WorkspaceDelta:
    return WorkspaceDelta(
        new_catalog=max(0, len(workspace.catalog_hits) - prev_counts.get("catalog", 0)),
        new_fulltext=max(0, len(workspace.fulltext_chunks) - prev_counts.get("fulltext", 0)),
        new_entities=max(0, len(workspace.entities) - prev_counts.get("entities", 0)),
        new_candidates=max(0, len(workspace.entity_candidates) - prev_counts.get("candidates", 0)),
        tools_called=tools_called,
    )


# =============================================================================
# API call with 429 retry + context shrinking  (now includes response_format)
# =============================================================================

def _call_with_retry(
    client,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    max_completion_tokens: int,
    workspace: ResearchWorkspace,
    delta: WorkspaceDelta,
    question: str,
    scope_note: str,
    verbose: bool = True,
    max_retries: int = 2,
    findings_brief: Optional[str] = None,
):
    """
    Call OpenAI with Structured Outputs + automatic 429 TPM retry.
    """
    import openai

    for attempt in range(max_retries + 1):
        shrink = _SHRINK_LEVELS[min(attempt, len(_SHRINK_LEVELS) - 1)]
        completion_tokens = min(max_completion_tokens, shrink["max_completion_tokens"]) if attempt > 0 else max_completion_tokens

        if attempt > 0:
            ctx = build_context_pack(
                workspace, delta,
                token_budget=shrink["token_budget"],
                chunk_char_cap=shrink["chunk_char_cap"],
                snippet_len=shrink["snippet_len"],
                max_catalog_rows=shrink["max_catalog_rows"],
                max_fulltext=shrink["max_fulltext"],
                findings_brief=findings_brief,
            )
            user_content = USER_PROMPT_TEMPLATE.format(
                question=question,
                scope_note=scope_note,
                context=ctx,
            )
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["content"] = user_content
                    break

            if verbose:
                est = _estimate_tokens(user_content)
                print(
                    f"  [V9] Retry {attempt}: shrunk context to ~{est} tokens, "
                    f"max_completion={completion_tokens}",
                    file=sys.stderr,
                )

        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,  # dropped by accident in 613ded1 — without it every
                                       # retry-path call raised "Missing required arguments"
                # Temperature 0 was tried to cut run-to-run variance but the 18-probe re-grade
                # showed it INCREASED tunneling (greedy locks the agent into narrow single-
                # collection retrieval) and net-worsened the grades — so default stays 0.2.
                # Tunable via V9_AGENT_TEMPERATURE for controlled experiments.
                "temperature": float(os.getenv("V9_AGENT_TEMPERATURE", "0.2")),
                "max_completion_tokens": completion_tokens,
                "response_format": V9_RESPONSE_FORMAT,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            response = client.chat.completions.create(**kwargs)
            return response
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            error_msg = str(e)
            err_type = type(e).__name__
            if attempt < max_retries:
                wait_time = 2 * (attempt + 1)
                if verbose:
                    print(
                        f"  [V9] API error ({err_type}) (attempt {attempt + 1}/{max_retries + 1}): {error_msg[:120]}. "
                        f"Shrinking context and retrying in {wait_time}s...",
                        file=sys.stderr,
                    )
                time.sleep(wait_time)
            else:
                raise


# =============================================================================
# Optional auditor LLM (unchanged)
# =============================================================================

def _run_auditor(
    client,
    synthesis: V9Synthesis,
    question: str,
    model: str = "gpt-4.1-mini-2025-04-14",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Call gpt-4.1-mini-2025-04-14 to check responsiveness. Action-oriented.
    Returns {responsive: bool, why_not: str, recommended_tool_calls: [{tool, args, why}]}.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Answer narrative: {synthesis.narrative[:1500]}\n\n"
        f"Artifact keys: {list(synthesis.artifact.keys()) if synthesis.artifact else []}\n\n"
        f"Sufficiency: {synthesis.sufficiency.to_dict() if synthesis.sufficiency else 'none'}\n\n"
        "Is this answer responsive to the question? "
        "If not, what's missing, and what 1-3 tool calls would help?\n\n"
        'Respond in JSON: {"responsive": true/false, "why_not": "...", '
        '"recommended_tool_calls": [{"tool": "...", "args": {...}, "why": "..."}]}'
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(os.getenv("V9_AGENT_TEMPERATURE", "0.2")),
            max_completion_tokens=300,
        )
        content = resp.choices[0].message.content or ""
        parsed = _parse_content(content)
        if parsed:
            return parsed
        return {"responsive": True, "why_not": "", "recommended_tool_calls": []}
    except Exception as e:
        if verbose:
            print(f"  [V9] Auditor error: {e}", file=sys.stderr)
        return {"responsive": True, "why_not": "", "recommended_tool_calls": []}


# =============================================================================
# Main loop
# =============================================================================

MAX_MODEL_TURNS = 40


def _parse_page_no(page: Optional[str]) -> Optional[int]:
    """Parse page string like 'p5' or '5' to int."""
    if not page:
        return None
    s = str(page).strip().lstrip("pP")
    try:
        return int(s) if s else None
    except ValueError:
        return None


def _get_doc_source_names(conn, doc_ids: List[int]) -> Dict[int, str]:
    """Look up source_name for each doc_id. Returns {doc_id: source_name}."""
    if not doc_ids:
        return {}
    ids = [d for d in doc_ids if d]
    if not ids:
        return {}
    out: Dict[int, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source_name FROM documents WHERE id = ANY(%s)",
                (ids,),
            )
            for row in cur.fetchall():
                out[row[0]] = (row[1] or "").strip()
    except Exception:
        pass
    return out


def _emit_progress(callback, step: str, status: str, message: str, details: Optional[Dict] = None):
    """Safely emit a progress event if callback is provided."""
    if callback:
        try:
            callback(step, status, message, details or {})
        except Exception:
            pass  # Never let progress reporting crash the runner


def _citation_lock_fetch_missing(
    conn,
    workspace: ResearchWorkspace,
    synthesis: V9Synthesis,
    cap: int = 20,
    verbose: bool = True,
) -> None:
    """
    Citation lock: ensure cited chunks are in workspace before grounding.
    Collects cited IDs from claims, roster, timeline, evidence, relationships;
    fetches missing (up to cap), merges into workspace.
    Auto-fetch always; repair loop is caller's responsibility (only when zero grounded).
    """
    cited_ids: set = set()
    for c in synthesis.claims:
        cited_ids.update(c.citation_chunk_ids or [])
        for e in (c.evidence or []):
            if e.chunk_id:
                cited_ids.add(e.chunk_id)
    for r in synthesis.get_roster():
        cited_ids.update(r.support_chunk_ids or [])
    for t in synthesis.get_timeline():
        cited_ids.update(t.support_chunk_ids or [])
    for e in synthesis.get_evidence():
        if e.chunk_id:
            cited_ids.add(e.chunk_id)
    for edge in synthesis.get_relationships():
        cited_ids.update(edge.support_chunk_ids or [])
    ident = synthesis.get_identity()
    if ident and ident.basis:
        for b in ident.basis:
            cited_ids.update(b.get("support_chunk_ids") or [])

    loaded_ids = set(workspace.fulltext_chunk_ids())
    missing_ids = [cid for cid in cited_ids if cid not in loaded_ids]
    if not missing_ids:
        return

    to_fetch = missing_ids[:cap]
    if verbose:
        print(
            f"  [V9 CitationLock] Missing {len(missing_ids)} cited chunks; "
            f"fetching {len(to_fetch)} (cap={cap})",
            file=sys.stderr,
        )
    chunks = fetch_chunks(
        conn,
        chunk_ids=to_fetch,
        include_neighbors=False,
    )
    merge_fetched_chunks(workspace, chunks)


def run_v9_query(
    conn,
    question: str,
    model: str = V9_MODEL,
    max_workspace_chunks: int = V9_MAX_WORKSPACE_CHUNKS,
    max_tool_calls: int = V9_MAX_TOOL_CALLS,
    scope: Optional[ScopeFilter] = None,
    verbose: bool = True,
    _resume_workspace: Optional[ResearchWorkspace] = None,
    progress_callback: Optional[Any] = None,
    _findings_brief: Optional[str] = None,
    session_id: Optional[int] = None,
    search_result_set_id: Optional[str] = None,
) -> V9Result:
    """
    Run the V9.4 Investigation Loop with Structured Outputs.

    Every model turn returns two channels:
    - structured JSON content (guaranteed by schema: final + scratchpad_update + synthesis fields)
    - optional tool_calls (search / fetch / expand_entities)

    Branch on ``output["final"]``:
    - false  -> update investigation state from scratchpad, execute any tool calls
    - true   -> validate finalization, optionally run auditor, return result

    Args:
        _resume_workspace: if provided (think_deeper), use this pre-populated workspace
            instead of creating a fresh one. The workspace already contains evidence
            from the previous run and investigation state.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for V9")

    # Step 1a: Deterministic scope detection (inline syntax + natural language)
    detected_scope = detect_scope(question)
    if scope:
        if scope.collections:
            detected_scope.collections = scope.collections
        if scope.date_from:
            detected_scope.date_from = scope.date_from
        if scope.date_to:
            detected_scope.date_to = scope.date_to

    # Strip scope: directives from question text so model sees clean question
    clean_question = strip_scope_syntax(question)

    # Step 1b: Lightweight router — extract content keywords, scope, intent
    # This is a fast GPT-4o-mini call that identifies the actual entity names
    # to resolve (instead of blindly extracting every word from the question).
    query_parse: Optional[QueryParse] = None
    if _resume_workspace is None:
        query_parse = _lightweight_parse_query(clean_question, verbose=verbose)

        # Merge router scope with deterministic scope (router fills gaps)
        if query_parse.collections and not detected_scope.collections:
            detected_scope.collections = query_parse.collections
        if query_parse.date_from and not detected_scope.date_from:
            detected_scope.date_from = query_parse.date_from
        if query_parse.date_to and not detected_scope.date_to:
            detected_scope.date_to = query_parse.date_to

    # Use resumed workspace or create fresh
    if _resume_workspace is not None:
        workspace = _resume_workspace
        workspace.scope = detected_scope
        if verbose:
            print(
                f"  [V9] Resuming with pre-populated workspace: "
                f"{len(workspace.fulltext_chunks)} chunks, "
                f"{len(workspace.entities)} entities",
                file=sys.stderr,
            )
    else:
        # Use reformulated query from router if available (cleaner, more specific)
        ws_question = (
            query_parse.reformulated_query
            if query_parse and query_parse.reformulated_query
            else clean_question
        )
        workspace = ResearchWorkspace(question=ws_question, scope=detected_scope)

        # Set investigation goal from intent
        if query_parse:
            _INTENT_GOALS = {
                "identity": "Identify the person/entity and establish their real name, aliases, and role.",
                "timeline": "Establish a chronological sequence of events with dates and sources.",
                "roster": "Compile a list of people/agents with their roles and affiliations.",
                "evidence": "Locate specific documentary evidence and primary source material.",
                "relationship": "Map the connections and relationships between the named entities.",
                "general": "Research the topic thoroughly and provide a comprehensive answer.",
            }
            workspace.investigation.goal = _INTENT_GOALS.get(
                query_parse.intent, _INTENT_GOALS["general"]
            )

    client = OpenAI(api_key=api_key)

    # Step 2: Prime workspace with entity resolution + auto-expand (skip if resuming)
    if _resume_workspace is None:
        content_keywords = query_parse.content_keywords if query_parse else None
        _prime_workspace_from_question(conn, clean_question, workspace, content_keywords=content_keywords, verbose=verbose)

    if verbose and workspace.entity_candidates:
        candidates_display = "; ".join(
            f"{c.query_term} -> {c.canonical_name} [{'ACCEPTED' if c.accepted else 'PENDING'}]"
            for c in workspace.entity_candidates
        )
        print(f"  [V9] Entity candidates from question: {candidates_display}", file=sys.stderr)
    if verbose and workspace.entities:
        for e in workspace.entities:
            alias_str = ", ".join(e.aliases[:8]) if e.aliases else "(none)"
            print(
                f"  [V9] Confirmed entity: {e.canonical_name} "
                f"(id={e.entity_id}, aliases: {alias_str})",
                file=sys.stderr,
            )
    if verbose and workspace.catalog_hits:
        print(f"  [V9] Catalog pre-loaded with {len(workspace.catalog_hits)} mention chunks", file=sys.stderr)

    # Step 2.5: PEM lane (alias-scoped seeding) — run when not resuming
    _v9_pem_enabled = os.getenv("V9_PEM_LANE_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    if _resume_workspace is None and _v9_pem_enabled:
        try:
            pem_result = _pem_lane_seed_chunks(
                conn, workspace, detected_scope, clean_question, verbose=verbose
            )
            if pem_result.chunk_ids:
                pem_cat = _load_catalog(conn, pem_result.chunk_ids, {cid: 0.95 for cid in pem_result.chunk_ids})
                merge_catalog_hits(workspace, pem_cat)
                workspace.pem_seed_chunk_ids = pem_result.chunk_ids
                workspace._pem_cache = pem_result.pem_cache
                workspace._pem_canonical_map = pem_result.canonical_map
                if verbose:
                    print(
                        f"  [V9] PEM lane: seeded {len(pem_result.chunk_ids)} chunks "
                        f"(surfaces={pem_result.seeded_surfaces[:5]}, entities={list(dict.fromkeys(pem_result.seeded_entities))[:5]})",
                        file=sys.stderr,
                    )
        except Exception as e:
            if verbose:
                print(f"  [V9] PEM lane error (continuing without): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    if verbose and not detected_scope.is_empty():
        print(f"  [V9] Scope detected: {detected_scope.to_dict()}", file=sys.stderr)

    # Progress: entity resolution complete
    entity_names = [e.canonical_name for e in workspace.entities]
    _emit_progress(progress_callback, "entity_resolution", "completed",
        f"Resolved {len(entity_names)} entities" + (f": {', '.join(entity_names[:5])}" if entity_names else ""),
        {"entities": entity_names, "catalog_hits": len(workspace.catalog_hits)})

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Scope note for user prompt
    scope_note = ""
    if not detected_scope.is_empty():
        parts = []
        if detected_scope.collections:
            parts.append(f"collections={detected_scope.collections}")
        if detected_scope.date_from:
            parts.append(f"date_from={detected_scope.date_from}")
        if detected_scope.date_to:
            parts.append(f"date_to={detected_scope.date_to}")
        scope_note = f"[System scope filter applied: {', '.join(parts)}. All searches are filtered accordingly.]"
    if search_result_set_id:
        scope_note += (
            f" [Context: The previous message returned a search result set (id={search_result_set_id}). "
            f"If the user asks to summarize or describe those results, use summarize_search_result_set(result_set_id=\"{search_result_set_id}\").]"
        )

    # Step 3: Investigation loop
    tool_calls_executed = 0
    model_turns = 0
    done = False
    synthesis: Optional[V9Synthesis] = None
    prev_counts = _snapshot_counts(workspace)
    tools_called_this_turn: List[str] = []
    step_idx = 0

    while not done and tool_calls_executed < max_tool_calls and model_turns < MAX_MODEL_TURNS:
        # Compute delta from last turn
        delta = _compute_delta(workspace, prev_counts, tools_called_this_turn)
        tools_called_this_turn = []

        _emit_progress(progress_callback, "turn_prepare", "running",
            f"Preparing turn {model_turns + 1}...",
            {"turn": model_turns + 1, "tool_calls_used": tool_calls_executed})

        # Build context pack (budget-constrained)
        ctx = build_context_pack(workspace, delta, conn=conn, findings_brief=_findings_brief)
        user_content = USER_PROMPT_TEMPLATE.format(
            question=clean_question,
            scope_note=scope_note,
            context=ctx,
        )
        messages.append({"role": "user", "content": user_content})

        # Trim history to prevent unbounded message growth
        messages = _trim_messages(messages)

        # Dynamic max_completion_tokens:
        #  - Synthesis level when model says it's ready
        #  - Synthesis level on last possible turn (budget exhausted)
        #  - Tool-turn level otherwise (keep small to stay within TPM)
        inv = workspace.investigation
        gaps_empty = not inv.gaps or inv.gaps == [""] or inv.gaps == []
        is_last_turn = tool_calls_executed >= max_tool_calls
        if (inv.ready_to_synthesize and gaps_empty) or is_last_turn:
            max_ct = SYNTHESIS_MAX_TOKENS
        else:
            max_ct = TOOL_TURN_MAX_TOKENS

        # Progress: turn start
        _emit_progress(progress_callback, "turn_start", "running",
            f"Turn {model_turns + 1}: {tool_calls_executed}/{max_tool_calls} tools used",
            {"turn": model_turns + 1, "tool_calls_used": tool_calls_executed,
             "tool_calls_budget": max_tool_calls,
             "catalog_hits": len(workspace.catalog_hits),
             "fulltext_chunks": len(workspace.fulltext_chunks)})

        _emit_progress(progress_callback, "model_call", "running",
            "Analyzing evidence and planning next steps...",
            {"turn": model_turns + 1})

        if verbose:
            est = _estimate_tokens(user_content)
            total_hist = sum(_estimate_tokens(m.get("content", "") or "") for m in messages)
            print(
                f"  [V9] Turn {model_turns}: catalog={len(workspace.catalog_hits)}, "
                f"fulltext={len(workspace.fulltext_chunks)}, "
                f"entities={len(workspace.entities)}, "
                f"candidates={len(workspace.entity_candidates)}, "
                f"tool_calls={tool_calls_executed}/{max_tool_calls}, "
                f"msgs={len(messages)}, hist~{total_hist}tok, "
                f"user_msg ~{est} tokens, max_ct={max_ct}",
                file=sys.stderr,
            )
            if workspace.evidence_memory:
                total_bullets = len(workspace._bullet_index)
                pinned = len(workspace.pinned_bullet_ids)
                print(
                    f"  [V9] EvidenceMemory: updates={len(workspace.evidence_memory)}, "
                    f"bullets={total_bullets}, pinned={pinned}",
                    file=sys.stderr,
                )

        try:
            response = _call_with_retry(
                client, model, messages, TOOLS_DEF, max_ct,
                workspace, delta, clean_question, scope_note, verbose,
                findings_brief=_findings_brief,
            )
            msg = response.choices[0].message
        except Exception as e:
            if verbose:
                print(f"  [V9] Model call error (timeout/rate-limit/connection): {e}", file=sys.stderr)
            synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)
            done = True
            continue
        model_turns += 1
        if not msg:
            break

        # -- Parse structured content (schema-guaranteed when present) --
        output = _parse_content(msg.content)

        # -- Update investigation from scratchpad (always available in output) --
        if output and isinstance(output.get("scratchpad_update"), dict):
            _update_investigation(workspace.investigation, output["scratchpad_update"])
            # Handle pin suggestions from the model
            pin_sugs = output["scratchpad_update"].get("pin_suggestions", [])
            if pin_sugs and isinstance(pin_sugs, list):
                apply_pin_suggestions(workspace, pin_sugs)

        # ================================================================
        # Branch A: Tool calls present  (final should be false)
        # ================================================================
        if msg.tool_calls:
            # Append assistant message
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute tool calls
            for tc in msg.tool_calls:
                if tool_calls_executed >= max_tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "Tool budget exhausted"}),
                    })
                    continue

                tool_calls_executed += 1
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                if verbose:
                    print(f"  [V9] Tool #{tool_calls_executed}: {name}({args})", file=sys.stderr)

                # Progress: tool call — emit context-rich message for UI feedback
                _brief_args = {k: (str(v)[:80] if isinstance(v, str) else v) for k, v in list(args.items())[:3]}
                _query = args.get("query", "")
                _chunk_ids = args.get("chunk_ids") or []
                if name == "search_chunks" and _query:
                    _msg = f"Searching for: {_query[:60]}{'...' if len(_query) > 60 else ''}"
                elif name == "fetch_chunks":
                    _n = len(_chunk_ids) if _chunk_ids else args.get("page_end") or "?"
                    _msg = f"Loading {_n} passages..."
                else:
                    _msg = f"{name}({', '.join(f'{k}={v}' for k, v in _brief_args.items())})"[:120]
                _emit_progress(progress_callback, "tool_call", "running",
                    _msg,
                    {"tool": name, "args": _brief_args, "tool_call_number": tool_calls_executed})

                result, summary = _execute_tool(
                    name, args, conn, workspace,
                    progress_callback=progress_callback,
                    session_id=session_id,
                )
                result_str = json.dumps(result, default=str)[:8000]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

                # Record investigation step
                rationale = workspace.investigation.goal or ""
                step = InvestigationStep(
                    step_idx=step_idx,
                    action=name,
                    rationale=rationale,
                    inputs=args,
                    outputs_summary=summary[:200],
                    added_catalog=max(0, len(workspace.catalog_hits) - prev_counts.get("catalog", 0)),
                    added_fulltext=max(0, len(workspace.fulltext_chunks) - prev_counts.get("fulltext", 0)),
                    added_entities=max(0, len(workspace.entities) - prev_counts.get("entities", 0)),
                )
                workspace.investigation.trace.append(step)
                append_note(workspace, f"[step {step_idx}] {name}: {summary[:120]}")
                tools_called_this_turn.append(name)
                step_idx += 1

                # Workspace cap check
                total_ws = len(workspace.fulltext_chunks) + len(workspace.catalog_hits)
                if total_ws >= max_workspace_chunks:
                    if verbose:
                        print(f"  [V9] Workspace cap reached ({total_ws})", file=sys.stderr)
                    break

            prev_counts = _snapshot_counts(workspace)

            # Budget exhausted -> request synthesis (no tools, schema still enforced)
            if tool_calls_executed >= max_tool_calls:
                _emit_progress(progress_callback, "synthesis", "running",
                    "Synthesizing answer from evidence...", {"reason": "budget_exhausted"})
                if verbose:
                    print(f"  [V9] Tool budget exhausted ({tool_calls_executed}/{max_tool_calls}). Requesting synthesis.", file=sys.stderr)
                messages.append({
                    "role": "user",
                    "content": (
                        "Tool budget exhausted. Synthesize now with whatever evidence you have. "
                        "Set final=true, fill sufficiency.remaining_gaps, fill responsiveness."
                    ),
                })
                messages = _trim_messages(messages)
                try:
                    synth_response = _call_with_retry(
                        client, model, messages, None, SYNTHESIS_MAX_TOKENS,
                        workspace, delta, clean_question, scope_note, verbose,
                        findings_brief=_findings_brief,
                    )
                    synth_msg = synth_response.choices[0].message
                except Exception as e:
                    if verbose:
                        print(f"  [V9] Synthesis API error (timeout/rate-limit/connection): {e}", file=sys.stderr)
                    synth_msg = None
                model_turns += 1
                if synth_msg and synth_msg.content:
                    synth_data = _parse_content(synth_msg.content)
                    if synth_data:
                        synth_data["artifact"] = _build_artifact_dict(synth_data.get("artifact") or {})
                        synthesis = V9Synthesis.from_dict(synth_data)
                    else:
                        # Budget exhausted AND structured output failed:
                        # use safe evidence-based fallback, never freeform narrative.
                        if verbose:
                            print("  [V9] Budget-exhausted synthesis failed structured output; "
                                  "using needs_more_evidence fallback", file=sys.stderr)
                        synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)
                    messages.append({"role": "assistant", "content": synth_msg.content})
                done = True
            continue

        # ================================================================
        # Branch B: No tool calls  (should be final=true)
        # With strict Structured Outputs we should always get valid JSON here;
        # null/empty or unparseable content indicates API quirk or truncation.
        #
        # Retry strategy:
        #  1. Retry WITH schema, but reduced context (drop catalog/fulltext
        #     bodies) and modestly higher completion tokens (1200).
        #  2. If still fails: build a safe "needs_more_evidence" synthesis
        #     from evidence memory bullets, never freeform narrative.
        # ================================================================
        if not output:
            if verbose:
                _reason = "null" if msg.content is None else "empty" if not (msg.content or "").strip() else "invalid JSON (likely truncated)"
                print(f"  [V9] Structured output missing ({_reason}); retrying with schema + minimal context", file=sys.stderr)

            # Build a lean user message (no catalog, no fulltext bodies)
            minimal_ctx = _build_minimal_synthesis_context(workspace, clean_question, scope_note)
            # Keep system + swap to minimal context-only message list
            system_msgs = [m for m in messages if m["role"] == "system"]
            retry_messages = system_msgs + [{"role": "user", "content": minimal_ctx}]

            _RETRY_SYNTHESIS_TOKENS = 1200
            try:
                response = _call_with_retry(
                    client, model, retry_messages, None, _RETRY_SYNTHESIS_TOKENS,
                    workspace, delta, clean_question, scope_note, verbose,
                    findings_brief=_findings_brief,
                )
                msg = response.choices[0].message
            except Exception as e:
                if verbose:
                    print(f"  [V9] Synthesis retry API error (timeout/rate-limit/connection): {e}", file=sys.stderr)
                msg = None
            model_turns += 1
            output = _parse_content(msg.content) if msg else None
            if output and isinstance(output.get("scratchpad_update"), dict):
                _update_investigation(workspace.investigation, output["scratchpad_update"])

            if not output:
                # Still no valid structured output. Produce a safe needs_more_evidence
                # synthesis from evidence memory, never freeform narrative.
                if verbose:
                    print("  [V9] Structured output still missing after minimal-context retry; "
                          "building needs_more_evidence fallback", file=sys.stderr)
                messages.append({"role": "assistant", "content": msg.content if msg else ""})
                synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)
                done = True
                continue

        messages.append({"role": "assistant", "content": msg.content or ""})

        if not output.get("final"):
            # Model says final=false but didn't call tools
            if verbose:
                print("  [V9] final=false without tool calls -- nudging", file=sys.stderr)
            messages.append({
                "role": "user",
                "content": (
                    "You set final=false but did not call any tools. "
                    "Either call tools to continue investigating, or set final=true to synthesize."
                ),
            })
            continue

        # --- Finalization attempt ---
        _emit_progress(progress_callback, "synthesis", "running",
            "Synthesizing final answer...", {})
        output["artifact"] = _build_artifact_dict(output.get("artifact") or {})
        # Apply defaults for null sufficiency/responsiveness (schema allows null but validation requires them)
        if output.get("final"):
            if not output.get("sufficiency"):
                output["sufficiency"] = {
                    "sufficient": False,
                    "argument": "Model did not populate; assuming insufficient.",
                    "remaining_gaps": ["Unknown"],
                    "next_best_actions_if_more_time": [],
                }
            resp = output.get("responsiveness")
            if not resp or not isinstance(resp, dict):
                output["responsiveness"] = {
                    "addressed_question": False,
                    "what_i_delivered": [],
                    "missing": ["Structured output incomplete"],
                    "why_missing": "Model did not populate responsiveness.",
                }
        synthesis = V9Synthesis.from_dict(output)

        valid, issues = _validate_finalization(synthesis, clean_question, tool_calls_executed, max_tool_calls, workspace=workspace)
        if not valid and tool_calls_executed < max_tool_calls - 2:
            if verbose:
                print(f"  [V9] Finalization issues: {issues}", file=sys.stderr)
            messages.append({
                "role": "user",
                "content": f"Finalization issues: {'; '.join(issues)}. Continue searching or fix the output.",
            })
            continue

        # Optional auditor
        if (valid
                and os.getenv("V9_USE_AUDITOR") == "1"
                and tool_calls_executed < max_tool_calls - 2):
            _emit_progress(progress_callback, "auditor", "running",
                "Running responsiveness auditor...", {})
            auditor_result = _run_auditor(client, synthesis, clean_question, verbose=verbose)
            if not auditor_result.get("responsive") and tool_calls_executed < max_tool_calls - 2:
                feedback = f"Auditor: {auditor_result.get('why_not', '')}"
                recs = auditor_result.get("recommended_tool_calls", [])
                if recs:
                    feedback += f"\nSuggested actions (you decide): {json.dumps(recs[:3])}"
                messages.append({"role": "user", "content": feedback})
                if verbose:
                    print(f"  [V9] Auditor says not responsive: {auditor_result.get('why_not', '')}", file=sys.stderr)
                continue

        # Citation lock: auto-fetch cited chunks before grounding
        _emit_progress(progress_callback, "citation_lock", "running",
            "Ensuring cited chunks are loaded...", {})
        _citation_lock_fetch_missing(conn, workspace, synthesis, cap=20, verbose=verbose)

        # Trial grounding: repair only if zero grounded claims after fetch
        trial_grounded = ground_claims(synthesis.claims, workspace)
        grounded_count = sum(1 for g in trial_grounded if g.status == "grounded")
        if grounded_count == 0 and tool_calls_executed < max_tool_calls - 2:
            messages.append({
                "role": "user",
                "content": (
                    "No grounded claims: your citation_chunk_ids reference chunks not in the workspace. "
                    "Call fetch_chunks on the cited chunk IDs before finalizing."
                ),
            })
            if verbose:
                print("  [V9] Citation lock: zero grounded claims after fetch; repair loop", file=sys.stderr)
            continue

        done = True

    # Fallback synthesis
    if not synthesis:
        synthesis = V9Synthesis(
            final=True,
            narrative="I could not produce a structured answer. You may need to run the query again or check the evidence.",
            claims=[],
        )

    # =================================================================
    # Ingest model-surfaced identity into alias hypotheses
    # =================================================================
    if synthesis.artifact:
        ident = synthesis.get_identity()
        if ident and ident.alias and ident.canonical:
            # Try to resolve through workspace + DB
            h = resolve_surfaced_alias(
                workspace,
                alias_text=ident.alias,
                conn=conn,
                turn_idx=model_turns,
            )
            if h:
                print(
                    f"  [V9 AliasHyp] identity ingestion: "
                    f"{ident.alias} -> entity {h.entity_id} "
                    f"(status={h.status}, reason={h.validated_reason})",
                    file=sys.stderr,
                )

    # =================================================================
    # Post-hoc grounding + advisory verification
    # =================================================================
    _emit_progress(progress_callback, "grounding", "running",
        "Grounding claims against evidence...", {})
    grounded = ground_claims(synthesis.claims, workspace)
    grounded_roster = ground_roster_entries(synthesis.get_roster(), workspace)

    _emit_progress(progress_callback, "verification", "running",
        "Building verification report...", {})

    # A: Build PEM operational alias map for AliasMap + retrieval
    try:
        from retrieval.agent.v9_pem_lane import build_pem_operational_alias_map
        pem_map = build_pem_operational_alias_map(
            conn, workspace, detected_scope, verbose=verbose,
        )
        if pem_map is not None:
            workspace._pem_operational_alias_map = pem_map
    except Exception as e:
        if verbose:
            print(f"  [V9] PEM operational map failed: {e}", file=sys.stderr)

    report = build_verification_report(grounded, synthesis, grounded_roster=grounded_roster)

    return V9Result(
        narrative=synthesis.narrative,
        claims=grounded,
        grounded_roster=grounded_roster,
        verification=report,
        sufficiency=synthesis.sufficiency,
        synthesis=synthesis,
        workspace=workspace,
        investigation_trace=list(workspace.investigation.trace),
    )


class V9Runner:
    """Runner for V9 Investigation Loop."""

    def run(self, question: str, conn, progress_callback=None) -> V9Result:
        return run_v9_query(conn, question, verbose=True, progress_callback=progress_callback)


def format_v9_result(result: V9Result, include_verification: bool = True) -> str:
    """Format V9 result for display."""
    return result.format_answer()
