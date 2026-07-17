"""
V9 Context (V9.5) - Token-budgeted context pack builder with Evidence Memory.

Builds a user-message context pack that fits within a hard token budget.
Priority order is fixed; each section is added only if budget permits.

V9.5 additions:
- select_evidence_memory_view(): pick pinned / recent / top-K bullets
- build_context_pack now renders the evidence memory view instead of raw notes

V9+PEM: two-section catalog (Primary vs Alias-Scoped), PEM mapping injection on fulltext.
"""
import logging
import re
import sys
from typing import List, Optional, Set, Any

from retrieval.agent.v9_types import (
    ResearchWorkspace,
    WorkspaceDelta,
    InvestigationState,
    EvidenceBullet,
    EvidenceMemoryView,
    EvidenceSummaryUpdate,
)


# =============================================================================
# Token estimation (chars / 4, no tiktoken dependency)
# =============================================================================

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_TOKEN_BUDGET = 6000    # lowered: structured-output schema adds ~1000 token overhead
DEFAULT_CHUNK_CHAR_CAP = 1200
DEFAULT_SNIPPET_LEN = 120
DEFAULT_MAX_CATALOG_ROWS = 15
DEFAULT_MAX_FULLTEXT = 4       # only newest fetched chunks (delta excerpts)

MAX_BULLETS_PER_UPDATE = 6     # matches summarizer cap
N_PEM_CATALOG_ROWS = 10        # alias-scoped evidence section
K_PRIMARY_CATALOG_ROWS = 15    # primary evidence section

logger = logging.getLogger(__name__)


# =============================================================================
# Keyword extraction (for relevance scoring)
# =============================================================================

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
    "and", "or", "for", "on", "at", "by", "with", "from", "that",
    "this", "it", "as", "be", "has", "had", "not", "but", "who",
    "what", "which", "where", "when", "how", "did", "does", "do",
}


def extract_keywords(text: str) -> Set[str]:
    """Split on non-alphanum, lowercase, drop stopwords, short tokens, and numeric-only."""
    tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS and not t.isdigit()}


# =============================================================================
# Evidence memory view selector
# =============================================================================

def select_evidence_memory_view(
    workspace: ResearchWorkspace,
    question: str,
    gaps: List[str],
    *,
    max_pinned: int = 10,
    recent_updates: int = 3,
    max_relevant: int = 10,
    total_cap: int = 20,
) -> EvidenceMemoryView:
    """Select the subset of evidence memory to show the model.

    Returns three disjoint bullet lists:
      - pinned_bullets: always-visible key facts
      - recent_bullets: from the last N updates
      - top_relevant_bullets: scored by keyword overlap with question/gaps
    Plus aggregated open_questions, leads, warnings from recent updates.
    """
    if not workspace.evidence_memory:
        return EvidenceMemoryView()

    seen_ids: Set[str] = set()

    # --- 1. Pinned ---
    pinned_bullets: List[EvidenceBullet] = []
    for bid in workspace.pinned_bullet_ids[:max_pinned]:
        b = workspace._bullet_index.get(bid)
        if b and bid not in seen_ids:
            pinned_bullets.append(b)
            seen_ids.add(bid)

    # --- 2. Recent ---
    recent_bullets: List[EvidenceBullet] = []
    recent_cap = recent_updates * MAX_BULLETS_PER_UPDATE
    recent_slices = workspace.evidence_memory[-recent_updates:]
    for update in recent_slices:
        for b in update.bullets:
            if len(recent_bullets) >= recent_cap:
                break
            if b.bullet_id not in seen_ids:
                recent_bullets.append(b)
                seen_ids.add(b.bullet_id)

    # --- 3. Top relevant (scored by keyword overlap + entity linkage) ---
    question_kw = extract_keywords(question)
    gap_kw: Set[str] = set()
    for g in gaps:
        gap_kw |= extract_keywords(g)

    # Collect doc_ids already represented in pinned + recent for diversity scoring
    represented_docs: Set[int] = set()
    for b in pinned_bullets + recent_bullets:
        represented_docs.update(b.doc_ids)

    # Collect entity IDs relevant to the question (from workspace entities + candidates)
    question_entity_ids: Set[int] = set()
    q_lower = question.lower()
    for e in workspace.entities:
        all_names = [e.canonical_name] + list(e.aliases)
        for name in all_names:
            if name.lower() in q_lower:
                question_entity_ids.add(e.entity_id)
                break
    for c in workspace.entity_candidates:
        if c.query_term.lower() in q_lower:
            question_entity_ids.add(c.entity_id)

    scored: List[tuple] = []  # (score, bullet)
    for bid, b in workspace._bullet_index.items():
        if bid in seen_ids:
            continue
        bullet_kw = extract_keywords(b.text) | {t.lower() for t in b.tags}
        score = 0
        if gap_kw & bullet_kw:
            score += 2
        if question_kw & bullet_kw:
            score += 2
        # Entity linkage bonus: bullet is about an entity the question asks about
        if question_entity_ids and b.linked_entity_ids:
            if set(b.linked_entity_ids) & question_entity_ids:
                score += 3
        if b.doc_ids and not (set(b.doc_ids) & represented_docs):
            score += 1
        if len(b.supporting_chunk_ids) >= 2:
            score += 1
        if score > 0:
            scored.append((score, b))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_relevant: List[EvidenceBullet] = []
    for _score, b in scored[:max_relevant]:
        if len(pinned_bullets) + len(recent_bullets) + len(top_relevant) >= total_cap:
            break
        top_relevant.append(b)
        seen_ids.add(b.bullet_id)

    # --- 4. Aggregate open_questions / leads / warnings from recent updates ---
    oq: List[str] = []
    leads: List[str] = []
    warnings: List[str] = []
    for update in recent_slices:
        oq.extend(update.open_questions)
        leads.extend(update.leads)
        warnings.extend(update.warnings)

    return EvidenceMemoryView(
        pinned_bullets=pinned_bullets,
        recent_bullets=recent_bullets,
        top_relevant_bullets=top_relevant,
        open_questions=oq[:4],
        leads=leads[:6],
        warnings=warnings[:3],
    )


# =============================================================================
# Bullet rendering helper
# =============================================================================

def _render_bullet(b: EvidenceBullet) -> str:
    """Render one evidence bullet as a compact context line."""
    chunks_str = ",".join(str(c) for c in b.supporting_chunk_ids[:6])
    line = f"  - [B:{b.bullet_id}] {b.text} (chunks: {chunks_str})"
    if b.tags:
        line += f" [tags: {','.join(b.tags)}]"
    return line + "\n"


# =============================================================================
# Context pack builder
# =============================================================================

def build_context_pack(
    workspace: ResearchWorkspace,
    delta: WorkspaceDelta,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    chunk_char_cap: int = DEFAULT_CHUNK_CHAR_CAP,
    snippet_len: int = DEFAULT_SNIPPET_LEN,
    max_catalog_rows: int = DEFAULT_MAX_CATALOG_ROWS,
    max_fulltext: int = DEFAULT_MAX_FULLTEXT,
    conn: Any = None,
    findings_brief: Optional[str] = None,
) -> str:
    """
    Build a token-budgeted context pack for the user message.

    Content priority (this order, budget-constrained):
      1. Question + scope
      2. Tool availability reminder
      3. Investigation state (goal, gaps, next_actions, ready_to_synthesize)
      4. Delta since last turn
      5. Evidence Memory View (pinned / recent / relevant / open_questions)
      6. Entity candidates (pending, prominent)
      7. Accepted entities (compact)
      8. Fulltext chunks (newest delta excerpts only, capped)
      9. Catalog hits (compact lines)
     10. Uncertainty flags
    """
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

    # -- 0. Findings Brief (Think Deeper re-synthesis scaffold) --
    if findings_brief:
        _add(f"\n{findings_brief}\n")

    # -- 1. Question (always) --
    header = f"=== Research Workspace ===\nQuestion: {workspace.question}\n"
    if not workspace.scope.is_empty():
        scope_parts = []
        if workspace.scope.collections:
            scope_parts.append(f"collections={workspace.scope.collections}")
        if workspace.scope.date_from:
            scope_parts.append(f"date_from={workspace.scope.date_from}")
        if workspace.scope.date_to:
            scope_parts.append(f"date_to={workspace.scope.date_to}")
        header += f"Scope filter (enforced): {', '.join(scope_parts)}\n"
    _add(header)

    # -- 2. Tool availability reminder (always, tiny) --
    _add(
        "Tools: search_chunks(mode=hybrid|lexical_exact), "
        "fetch_chunks(chunk_ids|doc_slice|pages), "
        "expand_entities(names|ids, include_comentions)\n"
    )

    # -- 3. Investigation state (always) --
    inv = workspace.investigation
    inv_lines = "Investigation state:\n"
    if inv.goal:
        inv_lines += f"  Goal: {inv.goal}\n"
    if inv.gaps:
        inv_lines += f"  Gaps: {inv.gaps}\n"
    if inv.next_actions:
        inv_lines += f"  Next actions: {inv.next_actions}\n"
    inv_lines += f"  Ready to synthesize: {inv.ready_to_synthesize}\n"
    if inv.hypotheses:
        inv_lines += f"  Hypotheses: {inv.hypotheses[:5]}\n"
    if inv.leads:
        inv_lines += f"  Leads: {inv.leads[:5]}\n"
    _add(inv_lines)

    # -- 4. Delta since last turn --
    if delta.tools_called or delta.new_catalog or delta.new_fulltext:
        _add(f"{delta.format()}\n")

    # -- 5. Evidence Memory View --
    if workspace.evidence_memory:
        view = select_evidence_memory_view(
            workspace,
            workspace.question,
            workspace.investigation.gaps,
        )
        total_shown = (
            len(view.pinned_bullets)
            + len(view.recent_bullets)
            + len(view.top_relevant_bullets)
        )

        if view.pinned_bullets:
            block = f"== Pinned Evidence ({len(view.pinned_bullets)}) ==\n"
            for b in view.pinned_bullets:
                block += _render_bullet(b)
            _add(block)

        if view.recent_bullets:
            block = f"== Recent Evidence ({len(view.recent_bullets)}) ==\n"
            for b in view.recent_bullets:
                block += _render_bullet(b)
            _add(block)

        if view.top_relevant_bullets:
            block = f"== Relevant Evidence ({len(view.top_relevant_bullets)}) ==\n"
            for b in view.top_relevant_bullets:
                block += _render_bullet(b)
            _add(block)

        # Open questions / leads / warnings
        meta_lines = ""
        if view.open_questions:
            meta_lines += "Open questions: " + "; ".join(view.open_questions[:4]) + "\n"
        if view.leads:
            meta_lines += "Leads: " + "; ".join(view.leads[:6]) + "\n"
        if view.warnings:
            meta_lines += "Warnings: " + "; ".join(view.warnings[:3]) + "\n"
        if meta_lines:
            _add(meta_lines)

        # Log view composition
        print(
            f"  [V9] MemoryView: pinned={len(view.pinned_bullets)}, "
            f"recent={len(view.recent_bullets)}, "
            f"top={len(view.top_relevant_bullets)}, "
            f"total={total_shown}",
            file=sys.stderr,
        )
    else:
        # Fallback: show notes if no evidence memory yet
        if workspace.notes:
            note_block = "Notes (recent):\n"
            for n in workspace.notes[-5:]:
                note_block += f"  - {n[:200]}\n"
            _add(note_block)

    # -- 6. Confirmed entities (prominent, with aliases and mention chunk guidance) --
    if workspace.entities:
        # Show confirmed entities FIRST — these are the most important context
        ent_lines = "\nConfirmed entities (auto-resolved from concordance/entity index):\n"
        for e in workspace.entities[-15:]:
            aliases = ", ".join(e.aliases[:5]) if e.aliases else ""
            ent_lines += f"  {e.canonical_name} (id={e.entity_id})"
            if aliases:
                ent_lines += f" a.k.a. {aliases}"
            ent_lines += "\n"
        # If catalog has chunks (from auto-expand), tell the model to use them
        ft_ids = set(workspace.fulltext_chunk_ids())
        entity_mention_chunks = [h for h in workspace.catalog_hits if h.chunk_id not in ft_ids]
        if entity_mention_chunks and not workspace.fulltext_chunks:
            ent_lines += (
                f"  >> {len(entity_mention_chunks)} mention chunks are in the catalog below. "
                "Call fetch_chunks on the top chunk_ids to load full text for these entities.\n"
            )
        _add(ent_lines)

    # -- 7. Entity candidates still pending --
    pending = [c for c in workspace.entity_candidates if not c.accepted]
    accepted_cands = [c for c in workspace.entity_candidates if c.accepted]
    if pending:
        cand_block = "Resolved identities (candidates -- call expand_entities to accept):\n"
        for c in pending:
            via = f", matched_via={c.matched_via}" if c.matched_via else ""
            etype = f", type={c.entity_type}" if c.entity_type else ""
            cand_block += (
                f"  {c.query_term} -> {c.canonical_name} "
                f"(entity_id={c.entity_id}{via}{etype}) [PENDING]\n"
            )
        _add(cand_block)
    if accepted_cands:
        acc_block = "Resolved identities (accepted):\n"
        for c in accepted_cands:
            acc_block += f"  {c.query_term} -> {c.canonical_name} (entity_id={c.entity_id}) [ACCEPTED]\n"
        _add(acc_block)

    # -- 8. Available chunks for citation (fulltext only) --
    # Model must cite from this list; catalog hits cannot be cited (no full text).
    if workspace.fulltext_chunks:
        cite_header = (
            "\nAvailable chunks for citation (use these chunk_ids in citation_chunk_ids):\n"
            "For each factual claim, set citation_chunk_ids to IDs from the list below. "
            "Invalid or missing IDs cause the claim to be dropped.\n"
        )
        _add(cite_header)
        _CITE_SNIPPET_LEN = 250
        for c in workspace.fulltext_chunks[-20:]:  # last 20 to fit budget
            snippet = c.text[:_CITE_SNIPPET_LEN].replace("\n", " ")
            if len(c.text) > _CITE_SNIPPET_LEN:
                snippet += "..."
            label = c.source_label or "unknown"
            page = c.page or ""
            line = f"  chunk_id={c.chunk_id} ({label} {page}): \"{snippet}\"\n"
            if not _add(line):
                break
        if len(workspace.fulltext_chunks) > 20:
            _add(f"  ... and {len(workspace.fulltext_chunks) - 20} more fulltext chunks (summarized in Evidence Memory)\n")

    # -- 8b. Fulltext chunks (newest delta excerpts only, budget-capped) + PEM mapping injection --
    ft_header = f"\nFull text loaded ({len(workspace.fulltext_chunks)} chunks):\n"
    _add(ft_header)

    entities_in_play: Set[int] = set()
    for e in workspace.entities:
        entities_in_play.add(e.entity_id)
    for c in workspace.entity_candidates:
        if c.entity_id:
            entities_in_play.add(c.entity_id)

    recent_ft = workspace.fulltext_chunks[-max_fulltext:] if workspace.fulltext_chunks else []
    ft_shown = 0
    for c in reversed(recent_ft):
        text = c.text[:chunk_char_cap]
        if len(c.text) > chunk_char_cap:
            text += "..."
        nbr_tag = " [neighbor]" if c.is_neighbor else ""
        source = c.source_label or ""
        page = c.page or ""
        line = f'  [chunk_id={c.chunk_id}]{nbr_tag} ({source} {page}): "{text}"\n'

        # PEM mapping injection (Hook B)
        pem_cache = getattr(workspace, "_pem_cache", None)
        pem_canonical = getattr(workspace, "_pem_canonical_map", None)
        if conn and pem_cache and pem_canonical:
            try:
                from retrieval.agent.v9_pem_lane import (
                    build_pem_mapping_block_for_chunk,
                    get_chunk_page_ids,
                )
                page_ids = get_chunk_page_ids(conn, c.chunk_id)
                if page_ids:
                    mapping_block = build_pem_mapping_block_for_chunk(
                        c.chunk_id,
                        c.text,
                        page_ids,
                        pem_cache,
                        pem_canonical,
                        entities_in_play,
                    )
                    if mapping_block:
                        line = line.rstrip() + mapping_block + "\n"
            except Exception as ex:
                logger.debug("[V9 context] PEM mapping for chunk %d failed: %s", c.chunk_id, ex)

        if not _add(line):
            break
        ft_shown += 1
    if len(workspace.fulltext_chunks) > ft_shown:
        _add(f"  ... {len(workspace.fulltext_chunks) - ft_shown} more fulltext chunks not shown (summarized in Evidence Memory above)\n")

    # -- 9. Catalog hits: two-section (Primary Evidence | Alias-Scoped Evidence) --
    ft_ids = set(workspace.fulltext_chunk_ids())
    not_yet_fetched = [h for h in workspace.catalog_hits if h.chunk_id not in ft_ids]
    pem_ids = set(getattr(workspace, "pem_seed_chunk_ids", None) or [])
    primary_hits = [h for h in not_yet_fetched if h.chunk_id not in pem_ids]
    pem_hits = [h for h in not_yet_fetched if h.chunk_id in pem_ids]

    # Primary Evidence
    cat_header = f"\nPrimary Evidence ({len(workspace.catalog_hits)} total, {len(not_yet_fetched)} not yet fetched):\n"
    _add(cat_header)
    cat_shown = 0
    for h in primary_hits:
        if cat_shown >= K_PRIMARY_CATALOG_ROWS:
            break
        snippet = h.snippet[:snippet_len].replace("\n", " ")
        line = (
            f'  [{h.chunk_id}] s={h.score:.2f} ({h.collection or ""} {h.page or ""}) '
            f'"{snippet}..."\n'
        )
        if not _add(line):
            break
        cat_shown += 1
    if len(primary_hits) > cat_shown:
        _add(f"  ... and {len(primary_hits) - cat_shown} more primary hits not shown\n")

    # Alias-Scoped Evidence (PEM lane seed) with prompt hint
    if pem_hits:
        pem_hint = "Alias-Scoped Evidence is seeded from the mention index; use it when interpreting codenames and aliases.\n"
        _add(pem_hint)
        pem_header = f"\nAlias-Scoped Evidence (Mention Index Seed, {len(pem_hits)} chunks):\n"
        _add(pem_header)
        pem_shown = 0
        for h in pem_hits[:N_PEM_CATALOG_ROWS]:
            snippet = h.snippet[:snippet_len].replace("\n", " ")
            line = (
                f'  [{h.chunk_id}] s={h.score:.2f} ({h.collection or ""} {h.page or ""}) '
                f'"{snippet}..."\n'
            )
            if not _add(line):
                break
            pem_shown += 1
        if len(pem_hits) > pem_shown:
            _add(f"  ... and {len(pem_hits) - pem_shown} more alias-scoped hits not shown\n")

    # -- 10. Uncertainty flags --
    if workspace.uncertainty_flags:
        uf = "Uncertainty:\n"
        for u in workspace.uncertainty_flags[-3:]:
            uf += f"  ? {u[:150]}\n"
        _add(uf)

    return "".join(parts)
