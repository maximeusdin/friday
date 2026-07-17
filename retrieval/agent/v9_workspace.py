"""
V9 Workspace (V9.4) - Merge helpers and progress signal.

V9.4 changes:
- append_hypothesis removed (hypotheses now in InvestigationState)
- format_workspace_for_context deprecated (replaced by v9_context.build_context_pack)
- All merge helpers and compute_progress_signal preserved
"""
import re
import sys
from typing import List, Dict, Any, Optional, Set

from retrieval.agent.v9_types import (
    ResearchWorkspace,
    CatalogHit,
    WorkspaceChunk,
    WorkspaceEntity,
    EntityCandidate,
    AliasHypothesis,
    ProgressSignal,
    ScopeFilter,
    EvidenceBullet,
    EvidenceSummaryUpdate,
    compute_bullet_id,
    _is_critical_pin,
)
from retrieval.agent.tools import ToolResult


# =============================================================================
# Token estimation (no tiktoken dependency -- chars / 4 is standard for English)
# =============================================================================

def _estimate_tokens(text: str) -> int:
    """Estimate token count for text (1 token ~ 4 chars for English/mixed content)."""
    return max(1, len(text) // 4)


# =============================================================================
# Merge helpers
# =============================================================================

def merge_catalog_hits(workspace: ResearchWorkspace, hits: List[CatalogHit]) -> None:
    """Merge catalog hits into workspace (dedupe by chunk_id)."""
    seen = set(workspace.catalog_chunk_ids())
    for h in hits:
        if h.chunk_id not in seen:
            seen.add(h.chunk_id)
            workspace.catalog_hits.append(h)


def merge_search_result(workspace: ResearchWorkspace, result: ToolResult, catalog: List[CatalogHit], query: str) -> None:
    """
    Merge search results: add catalog hits (no full text fetch here).
    Full text is only loaded when the model calls fetch_chunks.
    """
    merge_catalog_hits(workspace, catalog)
    if query and query.strip():
        workspace._search_queries.append(query.strip())


def merge_fetched_chunks(workspace: ResearchWorkspace, chunks: List[WorkspaceChunk]) -> None:
    """Merge fetched chunks into fulltext workspace (dedupe by chunk_id)."""
    seen = set(workspace.fulltext_chunk_ids())
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            workspace.fulltext_chunks.append(c)


def _sanitize_alias(alias: str, canonical_name: str = "") -> Optional[str]:
    """Sanitize a single alias string.  Returns cleaned alias or None to discard.

    Filters out:
    - Empty / whitespace-only strings
    - Strings > 50 chars (no real name or codename is that long)
    - Strings with > 3 space-separated tokens (real names are <=3 words,
      e.g. "Nathan Gregory Silvermaster"; anything longer is concordance noise)
    - Exact duplicates of the canonical name (case-insensitive)
    - Strings containing possessives (e.g. "vassiliev's") — context fragments
    - Strings where all tokens are lowercase and contain common descriptor words
      (e.g. "pel russian vassiliev's") — these are concordance noise, not aliases
    """
    a = (alias or "").strip()
    if not a:
        return None
    # Too long — not a real alias
    if len(a) > 50:
        return None
    # Possessive or apostrophe-s — likely context fragment, not an alias
    if "'s" in a or "'s" in a:
        return None
    # Too many tokens — real person names are at most 3 words
    tokens = a.split()
    if len(tokens) > 3:
        return None
    # Multi-word aliases that are ALL lowercase are likely noise fragments
    # (real names have capitalized words: "Nathan Gregory Silvermaster")
    if len(tokens) >= 2 and a == a.lower():
        return None
    # Skip if identical to canonical name
    if canonical_name and a.lower() == canonical_name.lower():
        return None
    return a


def _sanitize_aliases(aliases: List[str], canonical_name: str = "") -> List[str]:
    """Sanitize a list of aliases, deduplicating and filtering garbage."""
    seen: Set[str] = set()
    out: List[str] = []
    for alias in aliases:
        cleaned = _sanitize_alias(alias, canonical_name)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


def merge_entities(workspace: ResearchWorkspace, entities: List[WorkspaceEntity]) -> None:
    """Merge entities (dedupe by entity_id). These are accepted/confirmed entities.

    If an entity with the same entity_id already exists, update it with any
    new aliases rather than skipping.  This handles the case where
    accept_candidate creates a barebones entity and a later expand_entities
    call provides the full alias list.

    All aliases are sanitized on entry to remove concordance noise.
    """
    existing_map = {e.entity_id: e for e in workspace.entities}
    for e in entities:
        # Sanitize aliases before merging
        clean_aliases = _sanitize_aliases(e.aliases, e.canonical_name)

        if e.entity_id in existing_map:
            # Update existing entity with new aliases
            existing = existing_map[e.entity_id]
            existing_aliases = set(a.lower() for a in existing.aliases)
            for alias in clean_aliases:
                if alias.lower() not in existing_aliases:
                    existing.aliases.append(alias)
                    existing_aliases.add(alias.lower())
            # Update entity_type if missing
            if not existing.entity_type and e.entity_type:
                existing.entity_type = e.entity_type
        else:
            e.aliases = clean_aliases
            existing_map[e.entity_id] = e
            workspace.entities.append(e)


def merge_entity_candidates(workspace: ResearchWorkspace, candidates: List[EntityCandidate]) -> None:
    """Merge entity candidates (dedupe by entity_id). Candidates are NOT accepted entities."""
    seen = {c.entity_id for c in workspace.entity_candidates}
    for c in candidates:
        if c.entity_id not in seen:
            seen.add(c.entity_id)
            workspace.entity_candidates.append(c)


def append_note(workspace: ResearchWorkspace, note: str) -> None:
    if note and note.strip():
        workspace.notes.append(note.strip())


# =============================================================================
# Investigation-time concordance: entity linking + alias expansion
# =============================================================================

def build_alias_expansion_set(workspace: ResearchWorkspace) -> Dict[str, List[str]]:
    """Build a bidirectional name -> [alternate forms] map from workspace entities.

    Used to expand search queries with known aliases before hitting the DB.
    Supports BOTH directions:
      - canonical_name -> [alias1, alias2, ...]
      - alias -> [canonical_name, other_aliases...]

    Only includes accepted/confirmed entities.
    """
    expansion: Dict[str, List[str]] = {}
    for e in workspace.entities:
        if e.aliases:
            # Direction 1: canonical -> aliases
            expansion[e.canonical_name] = list(e.aliases)
            # Direction 2: alias -> [canonical + other aliases]
            for alias in e.aliases:
                if alias.lower() != e.canonical_name.lower():
                    other_forms = [e.canonical_name] + [
                        a for a in e.aliases if a.lower() != alias.lower()
                    ]
                    expansion[alias] = other_forms
    return expansion


def _query_contains_term_word_boundary(query_lower: str, term: str) -> bool:
    """Check if term appears as a whole word in query (avoids 'john' matching 'john moseley')."""
    if not term or len(term) < 2:
        return False
    t = term.lower().strip()
    if not t:
        return False
    import re as _re
    pattern = _re.compile(r"\b" + _re.escape(t) + r"\b")
    return pattern.search(query_lower) is not None


def expand_query_with_aliases(
    query: str,
    workspace: ResearchWorkspace,
    conn=None,
) -> str:
    """Expand a search query by appending known alias forms.

    B: PEM-truthy operational aliases only for retrieval. When conn is provided
    and scope includes alias-scoped corpora, uses PEM-backed surfaces only.
    Falls back to entity.aliases when PEM map is None (no alias-scoped).

    Supports BOTH directions:
      - name in query -> add aliases (canonical "Silvermaster" -> add "PAL")
      - alias in query -> add canonical name (codename "PAL" -> add "Silvermaster")
    """
    query_lower = query.lower()
    expansions: List[str] = []
    matched_any = False

    # B: PEM operational aliases first (when conn + scope available)
    if conn is not None and workspace.scope:
        try:
            from retrieval.agent.v9_pem_lane import build_pem_operational_alias_map
            pem_map = build_pem_operational_alias_map(
                conn, workspace, workspace.scope, verbose=False,
            )
            if pem_map is not None:
                # Build canonical -> [surfaces] for bidirectional expansion
                canon_to_surfaces: Dict[str, List[str]] = {}
                for surface_norm, entry in pem_map.items():
                    c = entry.get("canonical", "")
                    if c:
                        canon_to_surfaces.setdefault(c, []).append(surface_norm)
                for canonical, surfaces in canon_to_surfaces.items():
                    all_forms = [canonical] + surfaces
                    for form in all_forms:
                        if _query_contains_term_word_boundary(query_lower, form):
                            matched_any = True
                            for other in all_forms:
                                if not _query_contains_term_word_boundary(query_lower, other) and len(other) >= 2:
                                    expansions.append(other)
                            break
                if matched_any and expansions:
                    seen: Set[str] = set()
                    unique: List[str] = []
                    for x in expansions:
                        if x.lower() not in seen:
                            seen.add(x.lower())
                            unique.append(x)
                    return f"{query} ({' '.join(unique[:6])})"
        except Exception:
            pass  # PEM failed, fall through to entity_aliases

    # Fallback: workspace.entities (canonical -> aliases, alias -> canonical)
    for e in workspace.entities:
        all_names = [e.canonical_name] + list(e.aliases)
        matched = False
        for name in all_names:
            if _query_contains_term_word_boundary(query_lower, name):
                matched = True
                break
        if matched:
            matched_any = True
            for name in all_names:
                if not _query_contains_term_word_boundary(query_lower, name) and len(name) >= 2:
                    expansions.append(name)

    # Also check candidates (with high confidence) for reverse expansion
    for c in workspace.entity_candidates:
        if c.confidence in ("exact", "concordance") and not c.ambiguous:
            # If the query term matches the candidate's query_term or canonical_name (word-boundary)
            if _query_contains_term_word_boundary(query_lower, c.query_term) or _query_contains_term_word_boundary(query_lower, c.canonical_name):
                # Add the other direction
                if not _query_contains_term_word_boundary(query_lower, c.canonical_name) and len(c.canonical_name) >= 2:
                    expansions.append(c.canonical_name)
                    matched_any = True
                if not _query_contains_term_word_boundary(query_lower, c.query_term) and len(c.query_term) >= 2:
                    expansions.append(c.query_term)
                    matched_any = True

    # Check alias hypotheses (validated -> expand freely, proposed -> expand cautiously)
    for h in workspace.alias_hypotheses.values():
        if h.status == "rejected":
            continue
        alias_lower = h.alias_text.lower()
        # Find the entity this hypothesis maps to
        target_entity = None
        for e in workspace.entities:
            if e.entity_id == h.entity_id:
                target_entity = e
                break

        if target_entity:
            # If the alias appears in query (word-boundary), add canonical name
            if _query_contains_term_word_boundary(query_lower, h.alias_text):
                if not _query_contains_term_word_boundary(query_lower, target_entity.canonical_name) and len(target_entity.canonical_name) >= 2:
                    # Validated: always expand. Proposed: only if not already matched
                    if h.status == "validated" or not matched_any:
                        expansions.append(target_entity.canonical_name)
                        matched_any = True
            # If canonical appears in query (word-boundary), add alias
            elif _query_contains_term_word_boundary(query_lower, target_entity.canonical_name):
                if not _query_contains_term_word_boundary(query_lower, h.alias_text) and len(h.alias_text) >= 2:
                    if h.status == "validated":
                        expansions.append(h.alias_text)
                        matched_any = True

    # Direction 2: if no match in workspace, try direct DB reverse lookup
    # This handles the critical case: user queries with a codename that hasn't
    # been resolved yet (e.g., "PAL" before any entity resolution happened)
    if not matched_any and conn is not None:
        # Extract candidate terms from query (2+ char words)
        import re as _re
        words = _re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,24}", query)
        for word in words:
            if len(word) < 2:
                continue
            try:
                with conn.cursor() as cur:
                    # Reverse lookup: is this word an alias? -> get canonical name
                    cur.execute("""
                        SELECT DISTINCT e.canonical_name
                        FROM entities e
                        JOIN entity_aliases ea ON ea.entity_id = e.id
                        WHERE LOWER(ea.alias) = LOWER(%s)
                        LIMIT 3
                    """, (word,))
                    for row in cur.fetchall():
                        canonical = row[0]
                        if canonical and canonical.lower() not in query_lower:
                            expansions.append(canonical)
                            matched_any = True

                    # Also check alias_norm for normalized match
                    if not matched_any:
                        word_norm = _re.sub(r"[^a-z0-9 ]", "", word.lower()).strip()
                        if word_norm:
                            cur.execute("""
                                SELECT DISTINCT e.canonical_name
                                FROM entities e
                                JOIN entity_aliases ea ON ea.entity_id = e.id
                                WHERE ea.alias_norm = %s
                                LIMIT 3
                            """, (word_norm,))
                            for row in cur.fetchall():
                                canonical = row[0]
                                if canonical and canonical.lower() not in query_lower:
                                    expansions.append(canonical)
            except Exception:
                pass  # DB lookup failed, continue without

    if expansions:
        # Dedupe and cap at 6 expansion terms
        seen: Set[str] = set()
        unique: List[str] = []
        for x in expansions:
            if x.lower() not in seen:
                seen.add(x.lower())
                unique.append(x)
        return f"{query} ({' '.join(unique[:6])})"
    return query


def link_chunks_to_entities(workspace: ResearchWorkspace, chunks: List[WorkspaceChunk]) -> None:
    """Tag chunks with linked_entity_ids by checking if entity names/aliases appear in text.

    System-derived, not model-provided. Runs after fetch_chunks, before summarizer.

    Checks BOTH directions (bidirectional concordance):
    1. Accepted entities: matches on canonical name AND all aliases
    2. High-confidence candidates: matches on canonical_name AND query_term
       (query_term is often the alias form, canonical is the resolved name)

    This ensures that a chunk mentioning "PAL" gets linked to the entity
    whose canonical_name is "Nathan Gregory Silvermaster" and vice versa.
    """
    # Build a flat list of (entity_id, name_lower) pairs from both sources
    entity_names: List[tuple] = []

    # From accepted entities (canonical + all aliases — full bidirectional)
    for e in workspace.entities:
        for name in [e.canonical_name] + list(e.aliases):
            if len(name) >= 2:
                entity_names.append((e.entity_id, name.lower()))

    # From candidates (only high-confidence, non-ambiguous)
    # Include BOTH query_term (alias direction) and canonical_name (name direction)
    seen_eids = {e.entity_id for e in workspace.entities}
    for c in workspace.entity_candidates:
        if c.entity_id in seen_eids:
            continue
        if c.confidence in ("exact", "concordance", "partial") and not c.ambiguous:
            # Add canonical name (name→chunk matching)
            if c.canonical_name and len(c.canonical_name) >= 2:
                entity_names.append((c.entity_id, c.canonical_name.lower()))
            # Add query term which is often the alias/codename form (alias→chunk matching)
            if c.query_term and len(c.query_term) >= 2:
                entity_names.append((c.entity_id, c.query_term.lower()))
            # Add matched_via surface if it reveals another name form
            if c.matched_via:
                # Extract the surface form from matched_via like "alias_reverse:PAL"
                if ":" in c.matched_via:
                    surface = c.matched_via.split(":", 1)[1].strip()
                    if surface and len(surface) >= 2 and surface.lower() not in {
                        (c.canonical_name or "").lower(), (c.query_term or "").lower()
                    }:
                        entity_names.append((c.entity_id, surface.lower()))

    if not entity_names:
        return

    for chunk in chunks:
        text_lower = chunk.text.lower()
        linked: Set[int] = set(chunk.linked_entity_ids)
        for eid, name_lower in entity_names:
            if name_lower in text_lower:
                linked.add(eid)
        chunk.linked_entity_ids = sorted(linked)


def derive_bullet_entity_links(
    bullet: EvidenceBullet,
    workspace: ResearchWorkspace,
) -> List[int]:
    """Derive linked_entity_ids for a bullet from its supporting chunks.

    Union of linked_entity_ids across all supporting chunks.
    Also checks bullet text for entity name mentions (text-level linking).
    """
    linked: Set[int] = set()

    # From supporting chunks
    chunk_map = {c.chunk_id: c for c in workspace.fulltext_chunks}
    for cid in bullet.supporting_chunk_ids:
        c = chunk_map.get(cid)
        if c and c.linked_entity_ids:
            linked.update(c.linked_entity_ids)

    # From bullet text (lightweight text match)
    text_lower = bullet.text.lower()
    for e in workspace.entities:
        for name in [e.canonical_name] + list(e.aliases):
            if len(name) >= 2 and name.lower() in text_lower:
                linked.add(e.entity_id)

    return sorted(linked)


def build_alias_context_for_summarizer(workspace: ResearchWorkspace) -> str:
    """Build a compact alias context string for the summarizer prompt.

    Tells the summarizer which names refer to the same individual,
    so bullets use canonical names and don't create redundant entries.

    Includes BOTH accepted entities AND high-confidence candidates, so the
    summarizer benefits from concordance knowledge even before the model
    explicitly accepts a candidate (e.g., "Pal" -> Silvermaster).

    Example output:
      Known identities: PAL / Robert / Silvermaster = Nathan Gregory Silvermaster;
      LIBERAL = Julius Rosenberg
    """
    parts: List[str] = []
    seen_eids: Set[int] = set()

    # Topic/operation cover-words that keep getting mis-linked to common nouns and produce garbage
    # glosses ("atomic (Balloon)") — these are never real people, so never use them to unify ids.
    _NON_PERSON_TOKENS = {
        "balloon", "ballon", "atomic", "enormous", "enormoz", "uranium", "bomb", "plutonium",
        "tube", "corporation", "bank", "project", "operation",
    }
    _NON_PERSON_ENTITY_TYPES_WS = {
        "cover_name", "covername", "codename", "organization", "organisation", "org", "place",
        "location", "gpe", "operation", "event", "topic", "project", "facility", "vessel",
    }

    def _cover_junk(canonical: str, aliases: List[str], etype: Optional[str] = None) -> bool:
        c = (canonical or "").strip()
        if not c or not any(ch.isalpha() for ch in c):
            return True
        # KNOWN non-person entity types (cover names, orgs, places, operations) don't unify people.
        # Unknown/None types are kept so a mislabeled real person is never wrongly dropped.
        if (etype or "").lower() in _NON_PERSON_ENTITY_TYPES_WS:
            return True
        al = {a.strip().lower() for a in (aliases or [])}
        # Self-referential concordance junk (canonical_name IS one of its own cover names).
        if c.lower() in al:
            return True
        # A single-token canonical or any alias that is a topic/operation cover-word.
        if c.lower() in _NON_PERSON_TOKENS or (al & _NON_PERSON_TOKENS):
            return True
        return False

    # From accepted entities
    for e in workspace.entities:
        seen_eids.add(e.entity_id)
        if _cover_junk(e.canonical_name, e.aliases, getattr(e, "entity_type", None)):
            continue
        if e.aliases:
            # drop any alias identical to the canonical (adds no info, risks a self-gloss)
            al = [a for a in e.aliases if a.strip().lower() != (e.canonical_name or "").strip().lower()][:5]
            if al:
                alias_str = " / ".join(al)
                parts.append(f"{alias_str} = {e.canonical_name}")
            else:
                parts.append(e.canonical_name)
        else:
            parts.append(e.canonical_name)

    # From high-confidence candidates (alias -> canonical mapping)
    for c in workspace.entity_candidates:
        if c.entity_id in seen_eids:
            continue
        if c.confidence in ("exact", "concordance") and not c.ambiguous:
            if c.query_term and c.canonical_name and c.query_term.lower() != c.canonical_name.lower():
                parts.append(f"{c.query_term} = {c.canonical_name}")
                seen_eids.add(c.entity_id)

    # From alias hypotheses (validated = confident, proposed = tentative)
    seen_alias_keys: Set[str] = set()
    for h in workspace.alias_hypotheses.values():
        if h.status == "rejected":
            continue
        akey = f"{h.alias_text.lower()}:{h.entity_id}"
        if akey in seen_alias_keys:
            continue
        seen_alias_keys.add(akey)
        # Find canonical name for this entity
        canonical = ""
        for e in workspace.entities:
            if e.entity_id == h.entity_id:
                canonical = e.canonical_name
                break
        if not canonical:
            for c in workspace.entity_candidates:
                if c.entity_id == h.entity_id:
                    canonical = c.canonical_name
                    break
        if canonical and h.alias_text.lower() != canonical.lower():
            if h.status == "validated":
                parts.append(f"{h.alias_text} = {canonical}")
            else:
                parts.append(f"{h.alias_text} =? {canonical} (tentative)")

    if not parts:
        return ""
    return "Known identities: " + "; ".join(parts[:10])


# =============================================================================
# Alias Hypothesis Management
# =============================================================================

# Common first names / short words that should NOT be proposed as aliases
# unless they also appear as a known alias in the DB.
_COMMON_NAME_STOPLIST = {
    "robert", "bob", "bill", "william", "john", "james", "george", "charles",
    "david", "michael", "richard", "thomas", "edward", "henry", "joseph",
    "frank", "paul", "jack", "peter", "sam", "samuel", "harry", "carl",
    "albert", "walter", "arthur", "fred", "frederick", "alexander", "alex",
    "mary", "elizabeth", "helen", "anna", "margaret", "ruth", "alice",
    "dorothy", "louise", "marie", "catherine", "virginia", "gloria",
    "the", "this", "that", "with", "from", "agent", "source", "subject",
    "mr", "mrs", "ms", "dr", "sir", "comrade",
}


def _alias_key(alias_text: str, entity_id: int) -> tuple:
    """Canonical dict key for alias_hypotheses."""
    return (alias_text.strip().lower(), entity_id)


def propose_alias(
    workspace: ResearchWorkspace,
    alias_text: str,
    entity_id: int,
    supporting_chunk_ids: List[int],
    turn_idx: int = 0,
) -> Optional[AliasHypothesis]:
    """Propose a new alias hypothesis.  Returns the hypothesis if created,
    None if it was filtered by the stoplist or already exists.

    Stoplist gate: a single common first name (e.g., "Robert") is NOT
    proposed unless it already exists in the DB entity_aliases table
    (checked via workspace.entities aliases).
    """
    alias_clean = alias_text.strip()
    if not alias_clean or entity_id <= 0:
        return None

    key = _alias_key(alias_clean, entity_id)
    if key in workspace.alias_hypotheses:
        # Already exists — merge chunk evidence
        h = workspace.alias_hypotheses[key]
        existing_cids = set(h.supporting_chunk_ids)
        for cid in supporting_chunk_ids:
            if cid not in existing_cids:
                h.supporting_chunk_ids.append(cid)
                existing_cids.add(cid)
        return h

    # Stoplist gate: block common first names (case-insensitive, single-token)
    tokens = alias_clean.split()
    if len(tokens) == 1 and alias_clean.lower() in _COMMON_NAME_STOPLIST:
        # Allow if this name is already a known DB alias for this entity
        is_known = False
        for e in workspace.entities:
            if e.entity_id == entity_id:
                for a in e.aliases:
                    if a.lower() == alias_clean.lower():
                        is_known = True
                        break
                break
        if not is_known:
            return None

    h = AliasHypothesis(
        alias_text=alias_clean,
        entity_id=entity_id,
        supporting_chunk_ids=list(supporting_chunk_ids),
        status="proposed",
        created_turn_idx=turn_idx,
    )
    workspace.alias_hypotheses[key] = h
    return h


def validate_alias(
    workspace: ResearchWorkspace,
    alias_text: str,
    entity_id: int,
    reason: str = "",
) -> bool:
    """Promote a proposed hypothesis to validated. Returns True if found and updated."""
    key = _alias_key(alias_text, entity_id)
    h = workspace.alias_hypotheses.get(key)
    if h and h.status == "proposed":
        h.status = "validated"
        h.validated_reason = reason
        return True
    return False


def reject_alias(
    workspace: ResearchWorkspace,
    alias_text: str,
    entity_id: int,
) -> bool:
    """Reject a hypothesis. Returns True if found and updated."""
    key = _alias_key(alias_text, entity_id)
    h = workspace.alias_hypotheses.get(key)
    if h and h.status == "proposed":
        h.status = "rejected"
        return True
    return False


def get_hypothesis(
    workspace: ResearchWorkspace,
    alias_text: str,
    entity_id: int,
) -> Optional[AliasHypothesis]:
    """Retrieve a hypothesis by key."""
    return workspace.alias_hypotheses.get(_alias_key(alias_text, entity_id))


def get_active_hypotheses(
    workspace: ResearchWorkspace,
    entity_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[AliasHypothesis]:
    """Get hypotheses, optionally filtered by entity_id and/or status."""
    out = []
    for h in workspace.alias_hypotheses.values():
        if entity_id is not None and h.entity_id != entity_id:
            continue
        if status is not None and h.status != status:
            continue
        out.append(h)
    return out


# =============================================================================
# Resolve surfaced alias (Step A: deterministic, Step B: co-occurrence)
# =============================================================================

def resolve_surfaced_alias(
    workspace: ResearchWorkspace,
    alias_text: str,
    conn=None,
    turn_idx: int = 0,
) -> Optional[AliasHypothesis]:
    """Resolve a model-surfaced alias to a workspace entity.

    Step A (deterministic): exact DB lookups + concordance expansion.
    Step B (fallback): local evidence co-occurrence in workspace chunks.

    Returns the AliasHypothesis if successfully proposed/validated, else None.
    """
    alias_clean = alias_text.strip()
    if not alias_clean:
        return None

    # ---- Step A: Deterministic resolution ----
    # A1: Check if alias_text matches any workspace entity's known aliases
    for e in workspace.entities:
        all_names = [e.canonical_name.lower()] + [a.lower() for a in e.aliases]
        if alias_clean.lower() in all_names:
            # Already a known alias for this entity — validate immediately
            h = propose_alias(workspace, alias_clean, e.entity_id, [], turn_idx)
            if h:
                h.status = "validated"
                h.validated_reason = "exact_workspace_match"
            return h

    # A2: DB lookup — is this alias in entity_aliases?
    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.id, e.canonical_name
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE LOWER(ea.alias) = LOWER(%s)
                    LIMIT 5
                """, (alias_clean,))
                db_candidates = cur.fetchall()

                if db_candidates:
                    # Pick the candidate that overlaps with workspace entities
                    workspace_eids = {e.entity_id for e in workspace.entities}
                    for eid, canonical in db_candidates:
                        if eid in workspace_eids:
                            h = propose_alias(workspace, alias_clean, eid, [], turn_idx)
                            if h:
                                h.status = "validated"
                                h.validated_reason = "db_alias_match"
                            return h
                    # No workspace overlap — propose with first DB match
                    eid, canonical = db_candidates[0]
                    h = propose_alias(workspace, alias_clean, eid, [], turn_idx)
                    if h:
                        h.status = "validated"
                        h.validated_reason = "db_alias_match"
                    return h
        except Exception:
            pass

    # A3: DB lookup — is this a canonical name?
    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, canonical_name
                    FROM entities
                    WHERE LOWER(canonical_name) = LOWER(%s)
                    LIMIT 3
                """, (alias_clean,))
                rows = cur.fetchall()
                if rows:
                    workspace_eids = {e.entity_id for e in workspace.entities}
                    for eid, canonical in rows:
                        if eid in workspace_eids:
                            h = propose_alias(workspace, alias_clean, eid, [], turn_idx)
                            if h:
                                h.status = "validated"
                                h.validated_reason = "db_canonical_match"
                            return h
        except Exception:
            pass

    # ---- Step B: Fallback — local evidence co-occurrence ----
    # Find chunks containing alias_text, then see which entity IDs co-occur
    alias_lower = alias_clean.lower()
    entity_chunk_counts: Dict[int, int] = {}  # entity_id -> count of chunks with co-occurrence
    alias_chunk_ids: List[int] = []

    for chunk in workspace.fulltext_chunks:
        if alias_lower in chunk.text.lower():
            alias_chunk_ids.append(chunk.chunk_id)
            for eid in chunk.linked_entity_ids:
                entity_chunk_counts[eid] = entity_chunk_counts.get(eid, 0) + 1

    if not entity_chunk_counts or not alias_chunk_ids:
        return None

    # Self-reference exclusion: if entity's canonical name IS the alias_text,
    # that's identity, not a real alias mapping.
    # Also require a margin: top candidate must have >= 2x the runner-up
    ranked = sorted(entity_chunk_counts.items(), key=lambda x: x[1], reverse=True)

    # Filter out self-references
    filtered = [
        (eid, cnt) for eid, cnt in ranked
        if not _is_self_reference(workspace, alias_clean, eid)
    ]
    if not filtered:
        return None

    top_eid, top_count = filtered[0]
    runner_up_count = filtered[1][1] if len(filtered) > 1 else 0

    # Margin check: top must have at least 2x the runner-up (or runner_up == 0)
    has_margin = runner_up_count == 0 or top_count >= 2 * runner_up_count

    if top_count >= 2 and has_margin:
        # Strong co-occurrence with margin — propose as validated
        h = propose_alias(workspace, alias_clean, top_eid, alias_chunk_ids, turn_idx)
        if h:
            h.status = "validated"
            h.validated_reason = f"chunk_majority({top_count}/{top_count + runner_up_count})"
        return h
    elif top_count >= 1:
        # Weak co-occurrence — propose only
        h = propose_alias(workspace, alias_clean, top_eid, alias_chunk_ids, turn_idx)
        return h

    return None


def _is_self_reference(workspace: ResearchWorkspace, alias_text: str, entity_id: int) -> bool:
    """Check if alias_text is essentially the entity's own canonical name."""
    for e in workspace.entities:
        if e.entity_id == entity_id:
            if alias_text.lower() == e.canonical_name.lower():
                return True
            # Also check if the alias is just a substring of canonical (e.g. "Silvermaster" from "Nathan Gregory Silvermaster")
            # This is intentionally NOT blocked — short forms are valid aliases
            return False
    return False


# =============================================================================
# Validate hypotheses on expand_entities confirmation
# =============================================================================

def validate_hypotheses_for_entity(
    workspace: ResearchWorkspace,
    entity_id: int,
    reason: str = "expand_entities",
) -> int:
    """When expand_entities confirms an entity, validate any proposed
    hypotheses that point to that entity. Returns count validated."""
    count = 0
    for h in workspace.alias_hypotheses.values():
        if h.entity_id == entity_id and h.status == "proposed":
            h.status = "validated"
            h.validated_reason = reason
            count += 1
    return count


# =============================================================================
# Evidence memory: pinning
# =============================================================================

# Tags that trigger unconditional auto-pin (these are rare and high-value)
_UNCONDITIONAL_PIN_TAGS = {"warning", "contradiction"}
# Tags that only pin when combined with evidence strength
_CONDITIONAL_PIN_TAGS = {"identity", "alias", "codename", "roster"}
_AUTO_PIN_CONTENT_CUES = ["aka", "a.k.a.", "alias", "codename", "identified as"]

# Max auto-pins per single summarizer update (prevents over-pinning on dense updates)
MAX_AUTOPINS_PER_UPDATE = 2


def apply_auto_pin(bullet: EvidenceBullet) -> bool:
    """Evaluate and apply auto-pin rules to a bullet.  Returns True if pinned.

    Pin rules (tightened to prevent over-pinning):
    - Unconditional: warning/contradiction tags always pin.
    - Conditional: identity/alias/codename/roster tags OR content cues
      ONLY pin if ALSO have evidence strength (>=2 chunks OR >=2 docs).
    - Pure evidence strength alone does NOT auto-pin.
    """
    reasons: List[str] = []

    tag_set = {t.lower() for t in bullet.tags}
    has_evidence_strength = (
        len(bullet.supporting_chunk_ids) >= 2
        or len(set(bullet.doc_ids)) >= 2
    )

    # Unconditional: warning/contradiction always pin
    unconditional_matches = tag_set & _UNCONDITIONAL_PIN_TAGS
    if unconditional_matches:
        reasons.append("tag:" + ",".join(unconditional_matches))

    # Conditional: identity/alias/codename/roster + evidence strength
    conditional_matches = tag_set & _CONDITIONAL_PIN_TAGS
    if conditional_matches and has_evidence_strength:
        reasons.append("tag+evidence:" + ",".join(conditional_matches))

    # Content cues + evidence strength
    if not conditional_matches and has_evidence_strength:
        text_lower = bullet.text.lower()
        for cue in _AUTO_PIN_CONTENT_CUES:
            if cue in text_lower:
                reasons.append(f"content+evidence:{cue}")
                break

    if reasons:
        bullet.pinned = True
        bullet.pin_reason = ";".join(reasons)
        return True
    return False


def enforce_pin_cap(workspace: ResearchWorkspace, max_pinned: int = 10) -> None:
    """Evict non-critical pins if we exceed the cap.

    Eviction priority:
      1. Keep critical pins (warning/identity/contradiction/roster).
      2. Among non-critical, drop oldest first (by created_at).
    """
    if len(workspace.pinned_bullet_ids) <= max_pinned:
        return

    # Partition
    critical_ids: List[str] = []
    non_critical: List[EvidenceBullet] = []
    for bid in workspace.pinned_bullet_ids:
        b = workspace._bullet_index.get(bid)
        if not b:
            continue
        if _is_critical_pin(b):
            critical_ids.append(bid)
        else:
            non_critical.append(b)

    # Sort non-critical oldest-first so we drop from the front
    non_critical.sort(key=lambda b: b.created_at or "")

    # How many non-critical can we keep?
    keep_count = max(0, max_pinned - len(critical_ids))
    to_unpin = non_critical[:len(non_critical) - keep_count] if len(non_critical) > keep_count else []

    for b in to_unpin:
        b.pinned = False
        b.pin_reason = ""

    # Rebuild pinned list: critical (newest-first) + surviving non-critical (newest-first)
    surviving = non_critical[len(non_critical) - keep_count:] if keep_count else []
    surviving.sort(key=lambda b: b.created_at or "", reverse=True)
    critical_bullets = sorted(
        [workspace._bullet_index[bid] for bid in critical_ids if bid in workspace._bullet_index],
        key=lambda b: b.created_at or "", reverse=True,
    )
    workspace.pinned_bullet_ids = (
        [b.bullet_id for b in critical_bullets]
        + [b.bullet_id for b in surviving]
    )


def apply_pin_suggestions(workspace: ResearchWorkspace, suggestions: List[str]) -> None:
    """Accept model-suggested pins (subject to cap enforcement)."""
    for bid in suggestions:
        if not isinstance(bid, str):
            continue
        b = workspace._bullet_index.get(bid)
        if b and not b.pinned:
            b.pinned = True
            b.pin_reason = "model_suggested"
            workspace.pinned_bullet_ids.append(bid)
    enforce_pin_cap(workspace)


# =============================================================================
# Evidence memory: merge + trim
# =============================================================================

MAX_EVIDENCE_UPDATES = 100
MAX_TOTAL_BULLETS = 600


def build_chunk_doc_map(workspace: ResearchWorkspace) -> Dict[int, int]:
    """Map chunk_id -> doc_id from fulltext workspace chunks."""
    return {c.chunk_id: c.doc_id for c in workspace.fulltext_chunks if c.doc_id is not None}


def merge_evidence_summary_update(
    workspace: ResearchWorkspace,
    update: EvidenceSummaryUpdate,
    chunk_doc_map: Dict[int, int],
) -> None:
    """Merge one summarizer update into the workspace evidence memory.

    For each bullet:
    - Recompute bullet_id (canonical).
    - Always derive doc_ids from chunk_doc_map (never trust summarizer).
    - Derive linked_entity_ids from supporting chunks + text matching.
    - Set created_at from the parent update.
    - If duplicate bullet_id: merge supporting_chunk_ids + tags, re-derive, re-pin.
    - If new: apply auto-pin, add to index.
    """
    autopins_this_update = 0
    for bullet in update.bullets:
        # Recompute bullet_id
        bid = compute_bullet_id(bullet.text, bullet.supporting_chunk_ids)
        if not bid:
            continue
        bullet.bullet_id = bid

        # Always derive doc_ids
        bullet.doc_ids = sorted(set(
            chunk_doc_map[cid]
            for cid in bullet.supporting_chunk_ids
            if cid in chunk_doc_map
        ))

        # Derive linked_entity_ids from supporting chunks + text
        bullet.linked_entity_ids = derive_bullet_entity_links(bullet, workspace)

        # Set created_at
        bullet.created_at = update.created_at

        existing = workspace._bullet_index.get(bid)
        if existing:
            # Merge metadata into existing bullet
            merged_chunks = sorted(set(existing.supporting_chunk_ids) | set(bullet.supporting_chunk_ids))
            existing.supporting_chunk_ids = merged_chunks
            existing.tags = list(set(existing.tags) | set(bullet.tags))[:3]
            # Keep the first validated support quote; adopt the new one if we had none
            if not existing.support_quote and bullet.support_quote:
                existing.support_quote = bullet.support_quote
                existing.quote_chunk_id = bullet.quote_chunk_id
            existing.doc_ids = sorted(set(
                chunk_doc_map[cid]
                for cid in existing.supporting_chunk_ids
                if cid in chunk_doc_map
            ))
            # Re-derive entity links (may have gained new chunks)
            existing.linked_entity_ids = derive_bullet_entity_links(existing, workspace)
            # Re-evaluate pin (may now qualify due to stronger evidence)
            was_pinned = existing.pinned
            if autopins_this_update < MAX_AUTOPINS_PER_UPDATE:
                apply_auto_pin(existing)
                if existing.pinned and not was_pinned:
                    workspace.pinned_bullet_ids.append(bid)
                    autopins_this_update += 1
        else:
            # New bullet — apply auto-pin if budget allows
            if autopins_this_update < MAX_AUTOPINS_PER_UPDATE:
                if apply_auto_pin(bullet):
                    autopins_this_update += 1
            workspace._bullet_index[bid] = bullet
            if bullet.pinned:
                workspace.pinned_bullet_ids.append(bid)

    # Append update to memory
    workspace.evidence_memory.append(update)
    workspace._summarized_chunk_ids.update(update.generated_from_chunk_ids)

    # Enforce caps
    enforce_pin_cap(workspace)
    trim_evidence_memory(workspace)


def trim_evidence_memory(workspace: ResearchWorkspace) -> None:
    """Trim oldest updates if memory exceeds caps.

    Safety valve: if all old updates contain pinned bullets, allow trimming
    updates that contain only non-critical pinned bullets (unpin first).
    """
    total_bullets = sum(len(u.bullets) for u in workspace.evidence_memory)
    if (len(workspace.evidence_memory) <= MAX_EVIDENCE_UPDATES
            and total_bullets <= MAX_TOTAL_BULLETS):
        return

    # enforce_pin_cap first to free space
    enforce_pin_cap(workspace)

    pinned_set = set(workspace.pinned_bullet_ids)
    trimmed = False

    while (len(workspace.evidence_memory) > MAX_EVIDENCE_UPDATES
           or sum(len(u.bullets) for u in workspace.evidence_memory) > MAX_TOTAL_BULLETS):

        # Pass 1: try to trim updates with no pinned bullets (oldest first)
        removed = False
        for i, update in enumerate(workspace.evidence_memory):
            has_pinned = any(b.bullet_id in pinned_set for b in update.bullets)
            if not has_pinned:
                workspace.evidence_memory.pop(i)
                removed = True
                trimmed = True
                break

        if removed:
            continue

        # Pass 2 (stall safety valve): unpin non-critical bullets and trim
        for i, update in enumerate(workspace.evidence_memory):
            all_non_critical = all(
                not _is_critical_pin(b) for b in update.bullets if b.bullet_id in pinned_set
            )
            if all_non_critical:
                # Unpin all bullets in this update
                for b in update.bullets:
                    if b.pinned:
                        b.pinned = False
                        b.pin_reason = ""
                workspace.evidence_memory.pop(i)
                trimmed = True
                removed = True
                break

        if not removed:
            # Cannot trim further (all remaining updates have critical pins)
            break

    if trimmed:
        workspace.rehydrate_evidence_index()


# =============================================================================
# Progress signal
# =============================================================================

# Simple regex for person-name-like tokens (capitalized multi-word)
_NAME_RE = re.compile(r"\b[A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15}){1,3}\b")


def compute_progress_signal(workspace: ResearchWorkspace) -> ProgressSignal:
    """Compute lightweight progress signal for display to model."""
    # Current doc IDs
    cur_doc_ids: Set[int] = set()
    for c in workspace.fulltext_chunks:
        if c.doc_id:
            cur_doc_ids.add(c.doc_id)
    new_docs = len(cur_doc_ids - workspace._prev_doc_ids)

    # Person names from fulltext (cheap NER: capitalized multi-word)
    cur_names: Set[str] = set()
    for c in workspace.fulltext_chunks:
        for m in _NAME_RE.finditer(c.text[:2000]):
            cur_names.add(m.group())
    new_names = len(cur_names - workspace._prev_person_names)

    # Duplicate rate: how many catalog hits were already in fulltext
    ft_ids = set(workspace.fulltext_chunk_ids())
    cat_ids = set(workspace.catalog_chunk_ids())
    overlap = len(ft_ids & cat_ids)
    dup_rate = overlap / max(len(cat_ids), 1)

    # Update prev state for next call
    workspace._prev_doc_ids = cur_doc_ids.copy()
    workspace._prev_person_names = cur_names.copy()

    return ProgressSignal(
        new_docs_added=new_docs,
        new_person_names_found=new_names,
        duplicate_rate=dup_rate,
        search_queries_used=list(workspace._search_queries),
        total_catalog_hits=len(workspace.catalog_hits),
        total_fulltext_loaded=len(workspace.fulltext_chunks),
    )


# =============================================================================
# Context budget parameters (kept for backward compatibility)
# =============================================================================

DEFAULT_USER_MSG_BUDGET = 8000
DEFAULT_CHUNK_CHAR_CAP = 1200
DEFAULT_SNIPPET_LEN = 120
DEFAULT_MAX_CATALOG_ROWS = 25
DEFAULT_MAX_FULLTEXT = 12


# =============================================================================
# DEPRECATED: format_workspace_for_context
#
# Replaced by v9_context.build_context_pack in V9.4.
# Kept for backward compatibility but NOT called by the new runner.
# =============================================================================

def format_workspace_for_context(
    workspace: ResearchWorkspace,
    progress: Optional[ProgressSignal] = None,
    *,
    token_budget: int = DEFAULT_USER_MSG_BUDGET,
    chunk_char_cap: int = DEFAULT_CHUNK_CHAR_CAP,
    snippet_len: int = DEFAULT_SNIPPET_LEN,
    max_catalog_rows: int = DEFAULT_MAX_CATALOG_ROWS,
    max_fulltext: int = DEFAULT_MAX_FULLTEXT,
    max_fulltext_chars: int = 0,
) -> str:
    """
    DEPRECATED: Use v9_context.build_context_pack instead.

    Token-budgeted workspace formatter (kept for backward compatibility).
    """
    import warnings
    warnings.warn(
        "format_workspace_for_context is deprecated. Use v9_context.build_context_pack instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if max_fulltext_chars and max_fulltext_chars < chunk_char_cap:
        chunk_char_cap = max_fulltext_chars

    parts: List[str] = []
    used_tokens = 0

    def _add(text: str) -> bool:
        nonlocal used_tokens
        t = _estimate_tokens(text)
        if used_tokens + t > token_budget:
            return False
        parts.append(text)
        used_tokens += t
        return True

    # Header
    header_lines = [
        "=== Research Workspace ===",
        f"Question: {workspace.question}",
    ]
    if not workspace.scope.is_empty():
        scope_parts = []
        if workspace.scope.collections:
            scope_parts.append(f"collections={workspace.scope.collections}")
        if workspace.scope.date_from:
            scope_parts.append(f"date_from={workspace.scope.date_from}")
        if workspace.scope.date_to:
            scope_parts.append(f"date_to={workspace.scope.date_to}")
        header_lines.append(f"Scope filter (enforced): {', '.join(scope_parts)}")
    header_lines.append("")
    if progress:
        header_lines.append(progress.format())
        header_lines.append("")
    _add("\n".join(header_lines))

    # Candidates
    pending_candidates = [c for c in workspace.entity_candidates if not c.accepted]
    accepted_candidates = [c for c in workspace.entity_candidates if c.accepted]
    if pending_candidates:
        cand_block = "Resolved identities (candidates -- call expand_entities to accept/confirm):\n"
        for c in pending_candidates:
            via = f", matched_via={c.matched_via}" if c.matched_via else ""
            etype = f", type={c.entity_type}" if c.entity_type else ""
            cand_block += (
                f"  {c.query_term} -> {c.canonical_name} "
                f"(entity_id={c.entity_id}{via}{etype}) [PENDING]\n"
            )
        _add(cand_block)

    if accepted_candidates:
        acc_block = "Resolved identities (accepted):\n"
        for c in accepted_candidates:
            acc_block += f"  {c.query_term} -> {c.canonical_name} (entity_id={c.entity_id}) [ACCEPTED]\n"
        _add(acc_block)

    # Entities
    if workspace.entities:
        ent_lines = "\nEntities (confirmed):\n"
        for e in workspace.entities[-15:]:
            aliases = ", ".join(e.aliases[:3]) if e.aliases else ""
            ent_lines += f"  {e.canonical_name} (id={e.entity_id}) {aliases}\n"
        _add(ent_lines)

    # Notes
    if workspace.notes:
        note_block = "Your notes:\n"
        for n in workspace.notes[-5:]:
            note_block += f"  - {n[:200]}\n"
        _add(note_block)

    # Fulltext
    ft_header = f"\nFull text loaded ({len(workspace.fulltext_chunks)} chunks):\n"
    _add(ft_header)

    sorted_ft = sorted(
        workspace.fulltext_chunks,
        key=lambda c: c.score if c.score else 0.0,
        reverse=True,
    )
    ft_shown = 0
    for c in sorted_ft:
        if ft_shown >= max_fulltext:
            break
        text = c.text[:chunk_char_cap]
        if len(c.text) > chunk_char_cap:
            text += "..."
        nbr_tag = " [neighbor]" if c.is_neighbor else ""
        source = c.source_label or ""
        page = c.page or ""
        line = f'  [chunk_id={c.chunk_id}]{nbr_tag} ({source} {page}): "{text}"\n'
        if not _add(line):
            break
        ft_shown += 1
    if len(workspace.fulltext_chunks) > ft_shown:
        _add(f"  ... {len(workspace.fulltext_chunks) - ft_shown} more fulltext chunks not shown\n")

    # Catalog
    ft_ids = set(workspace.fulltext_chunk_ids())
    not_yet_fetched = [h for h in workspace.catalog_hits if h.chunk_id not in ft_ids]
    cat_header = f"\nCatalog hits ({len(workspace.catalog_hits)} total, {len(not_yet_fetched)} not yet fetched):\n"
    _add(cat_header)

    cat_shown = 0
    for h in not_yet_fetched:
        if cat_shown >= max_catalog_rows:
            break
        snippet = h.snippet[:snippet_len].replace("\n", " ")
        line = (
            f'  [{h.chunk_id}] s={h.score:.2f} ({h.collection or ""} {h.page or ""}) '
            f'"{snippet}..."\n'
        )
        if not _add(line):
            break
        cat_shown += 1
    if len(not_yet_fetched) > cat_shown:
        _add(f"  ... and {len(not_yet_fetched) - cat_shown} more catalog hits not shown\n")

    # Uncertainty
    if workspace.uncertainty_flags:
        uf = "Uncertainty:\n"
        for u in workspace.uncertainty_flags[-3:]:
            uf += f"  ? {u[:150]}\n"
        _add(uf)

    return "".join(parts)
