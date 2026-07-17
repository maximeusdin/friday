"""
V11 Context — Token-budgeted context pack builder (no PEM injection, optional lightweight PEM).

Same as V9 context except:
- No per-chunk PEM mapping injection
- No two-section catalog (Primary vs Alias-Scoped) — single catalog
- Optional: append lightweight PEM mention block after fulltext when use_lightweight_pem=True
  (only for venona/vassiliev chunks, top N bundles, capped)
"""
import logging
import os
import re
import sys
from collections import Counter
from typing import List, Optional, Set, Any

from retrieval.agent.v11_types import (
    V11ResearchWorkspace,
    WorkspaceDelta,
    InvestigationState,
    EvidenceBullet,
    EvidenceMemoryView,
    EvidenceSummaryUpdate,
)
from retrieval.agent.v9_context import (
    select_evidence_memory_view,
    _render_bullet,
    _estimate_tokens,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_CHUNK_CHAR_CAP,
    DEFAULT_SNIPPET_LEN,
    DEFAULT_MAX_CATALOG_ROWS,
    DEFAULT_MAX_FULLTEXT,
)

logger = logging.getLogger(__name__)

# Lightweight PEM — cap at source to avoid crowding non-V/V chunks from context
LIGHTWEIGHT_PEM_COLLECTIONS = frozenset({"venona", "vassiliev"})

# Transcript collections: prepend a speaker/witness attribution line to each
# chunk in the LLM context (sourced from the canonical attribution manifest).
TRANSCRIPT_COLLECTIONS = frozenset({
    "rosenberg_grand_jury", "brothman_moskowitz_grand_jury", "rosenberg_trial_transcripts",
})


def _fetch_transcript_attribution(conn, chunk_ids):
    """chunk_id -> one-line speaker attribution, from chunk_embeddings_canonical.rewrite_manifest."""
    out = {}
    if not conn or not chunk_ids:
        return out
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, rewrite_manifest FROM chunk_embeddings_canonical "
                "WHERE chunk_id = ANY(%s)",
                (list(chunk_ids),),
            )
            for cid, man in cur.fetchall():
                m = man[0] if isinstance(man, list) and man else {}
                line = None
                if m.get("witness"):
                    parts = [f"Speaker: {m['witness']} (grand jury witness)"]
                    if m.get("examiner"):
                        parts.append(f"examined by {m['examiner']}")
                    if m.get("date"):
                        parts.append(str(m["date"]))
                    line = ", ".join(parts)
                elif m.get("speaker"):
                    role = f" ({m['role']})" if m.get("role") else ""
                    line = f"Speaker: {m['speaker']}{role} — U.S. v. Rosenberg trial (1951)"
                if line:
                    out[cid] = line
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    return out
LIGHTWEIGHT_PEM_MAX_BUNDLES = 6
LIGHTWEIGHT_PEM_MAX_LINES = 40
LIGHTWEIGHT_PEM_MAX_CHARS = 1600
LIGHTWEIGHT_PEM_MAX_ENTITIES = 25
LIGHTWEIGHT_PEM_MAX_ALIASES_PER_ENTITY = 8
LIGHTWEIGHT_PEM_MAX_TOKENS = 600
LIGHTWEIGHT_PEM_SHRINK_LINES = 20  # when bundle near token limit


def _extract_keywords(text: str) -> Set[str]:
    """Split on non-alphanum, lowercase, drop stopwords."""
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
        "and", "or", "for", "on", "at", "by", "with", "from", "that",
    }
    tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS and not t.isdigit()}


def build_context_pack(
    workspace: V11ResearchWorkspace,
    delta: WorkspaceDelta,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    chunk_char_cap: int = DEFAULT_CHUNK_CHAR_CAP,
    snippet_len: int = DEFAULT_SNIPPET_LEN,
    max_catalog_rows: int = DEFAULT_MAX_CATALOG_ROWS,
    max_fulltext: int = DEFAULT_MAX_FULLTEXT,
    conn: Any = None,
    use_lightweight_pem: bool = False,
    dump_pem_light: bool = False,
    findings_brief: Optional[str] = None,
) -> str:
    """
    Build a token-budgeted context pack for the user message.

    When use_lightweight_pem=True and conn is provided:
    - After fulltext chunks, append a [MENTION_INDEX] block for venona/vassiliev chunks
    - Only for top N bundles (by recency), capped max_lines/max_chars
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

    # -- 2. Tool availability reminder --
    _add(
        "Tools: search, search_broad + fetch_diverse, resolve_codenames(terms=[...]) for codename tokens, "
        "search_canonical/search_lexical (use token only), expand_query, expand_from_evidence, fetch_chunks\n"
    )

    # -- 3. Investigation state --
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

        meta_lines = ""
        if view.open_questions:
            meta_lines += "Open questions: " + "; ".join(view.open_questions[:4]) + "\n"
        if view.leads:
            meta_lines += "Leads: " + "; ".join(view.leads[:6]) + "\n"
        if view.warnings:
            meta_lines += "Warnings: " + "; ".join(view.warnings[:3]) + "\n"
        if meta_lines:
            _add(meta_lines)

        print(
            f"  [V11] MemoryView: pinned={len(view.pinned_bullets)}, "
            f"recent={len(view.recent_bullets)}, "
            f"top={len(view.top_relevant_bullets)}, "
            f"total={total_shown}",
            file=sys.stderr,
        )
    else:
        if workspace.notes:
            note_block = "Notes (recent):\n"
            for n in workspace.notes[-5:]:
                note_block += f"  - {n[:200]}\n"
            _add(note_block)

    # -- 6. Confirmed entities --
    if workspace.entities:
        ent_lines = "\nConfirmed entities:\n"
        for e in workspace.entities[-15:]:
            aliases = ", ".join(e.aliases[:5]) if e.aliases else ""
            ent_lines += f"  {e.canonical_name} (id={e.entity_id})"
            if aliases:
                ent_lines += f" a.k.a. {aliases}"
            ent_lines += "\n"
        _add(ent_lines)

    # -- 7. Entity candidates --
    pending = [c for c in workspace.entity_candidates if not c.accepted]
    accepted_cands = [c for c in workspace.entity_candidates if c.accepted]
    if pending:
        cand_block = "Resolved identities (candidates):\n"
        for c in pending:
            via = f", matched_via={c.matched_via}" if c.matched_via else ""
            cand_block += (
                f"  {c.query_term} -> {c.canonical_name} "
                f"(entity_id={c.entity_id}{via}) [PENDING]\n"
            )
        _add(cand_block)
    if accepted_cands:
        acc_block = "Resolved identities (accepted):\n"
        for c in accepted_cands:
            acc_block += f"  {c.query_term} -> {c.canonical_name} (entity_id={c.entity_id}) [ACCEPTED]\n"
        _add(acc_block)

    # -- 8. Available chunks for citation (fulltext only) --
    if workspace.fulltext_chunks:
        cite_header = (
            "\nAvailable chunks for citation (use these chunk_ids in citation_chunk_ids):\n"
            "For each factual claim, set citation_chunk_ids to IDs from the list below. "
            "Invalid or missing IDs cause the claim to be dropped.\n"
        )
        _add(cite_header)
        _CITE_SNIPPET_LEN = 250
        for c in workspace.fulltext_chunks[-20:]:
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

    # -- 8b. Fulltext chunks (no per-chunk PEM injection) --
    ft_header = f"\nFull text loaded ({len(workspace.fulltext_chunks)} chunks):\n"
    _add(ft_header)

    recent_ft = workspace.fulltext_chunks[-max_fulltext:] if workspace.fulltext_chunks else []
    ft_shown = 0
    pem_bundle_chunk_ids: List[List[int]] = []  # chunks per "bundle" for lightweight PEM

    # Speaker attribution for transcript chunks (so the model knows who is speaking)
    transcript_attr = _fetch_transcript_attribution(
        conn, [c.chunk_id for c in recent_ft if c.collection_slug in TRANSCRIPT_COLLECTIONS]
    )

    for c in reversed(recent_ft):
        text = c.text[:chunk_char_cap]
        if len(c.text) > chunk_char_cap:
            text += "..."
        attr = transcript_attr.get(c.chunk_id)
        if attr:
            text = f"[{attr}]\n{text}"
        nbr_tag = " [neighbor]" if c.is_neighbor else ""
        source = c.source_label or ""
        page = c.page or ""
        line = f'  [chunk_id={c.chunk_id}]{nbr_tag} ({source} {page}): "{text}"\n'

        if not _add(line):
            break
        ft_shown += 1
        # Collect chunk_ids for PEM (each chunk = 1-chunk bundle for simplicity)
        if c.collection_slug in LIGHTWEIGHT_PEM_COLLECTIONS:
            pem_bundle_chunk_ids.append([c.chunk_id])

    if len(workspace.fulltext_chunks) > ft_shown:
        _add(f"  ... {len(workspace.fulltext_chunks) - ft_shown} more fulltext chunks not shown\n")

    if os.getenv("DIAG_SCOPE_BIAS", "").strip() in ("1", "true", "yes") and workspace.fulltext_chunks:
        coll_counts = Counter(
            (c.source_label or c.collection_slug or "unknown") for c in workspace.fulltext_chunks
        )
        tok_by_coll: dict = {}
        for c in workspace.fulltext_chunks:
            lbl = c.source_label or c.collection_slug or "unknown"
            tok_by_coll[lbl] = tok_by_coll.get(lbl, 0) + _estimate_tokens(c.text[:chunk_char_cap])
        ft_tok = sum(tok_by_coll.values())
        print(
            f"  [V11 Context] fulltext by collection: {dict(coll_counts)}, "
            f"fulltext tokens ~{ft_tok}, catalog rows {len(workspace.catalog_hits)}",
            file=sys.stderr,
        )
        if tok_by_coll:
            print(f"  [V11 Context] tokens by collection: {tok_by_coll}", file=sys.stderr)

    # -- 8b. Lightweight PEM block (evidence-time mention index) --
    if use_lightweight_pem and conn and pem_bundle_chunk_ids:
        try:
            from retrieval.pem_light import build_mention_index_for_pages, get_page_ids_for_chunks

            # Cap to top N bundles
            bundles_to_annotate = pem_bundle_chunk_ids[:LIGHTWEIGHT_PEM_MAX_BUNDLES]
            all_page_ids: List[int] = []
            for chunk_ids in bundles_to_annotate:
                pids = get_page_ids_for_chunks(conn, chunk_ids)
                all_page_ids.extend(pids)
            all_page_ids = list(dict.fromkeys(all_page_ids))  # dedupe

            if all_page_ids:
                # Shrink if already near token limit
                remaining_tokens = token_budget - used_tokens
                max_lines = LIGHTWEIGHT_PEM_MAX_LINES
                if remaining_tokens < 1500:  # ~60 lines * 25 tokens/line
                    max_lines = LIGHTWEIGHT_PEM_SHRINK_LINES

                block, manifest = build_mention_index_for_pages(
                    conn,
                    all_page_ids,
                    max_lines=max_lines,
                    max_chars=LIGHTWEIGHT_PEM_MAX_CHARS,
                    max_entities=LIGHTWEIGHT_PEM_MAX_ENTITIES,
                    max_aliases_per_entity=LIGHTWEIGHT_PEM_MAX_ALIASES_PER_ENTITY,
                    max_tokens=LIGHTWEIGHT_PEM_MAX_TOKENS,
                    collection_slugs=("venona", "vassiliev"),
                )

                if block and _add(block):
                    if dump_pem_light:
                        inc = len(manifest.get("included", []))
                        sk = len(manifest.get("skipped", []))
                        preview = block[:200] + "..." if len(block) > 200 else block
                        print(
                            f"  [V11 PEM-light] pages={len(all_page_ids)}, "
                            f"included={inc}, skipped={sk}\n  preview: {preview!r}",
                            file=sys.stderr,
                        )
        except Exception as ex:
            logger.debug("[V11 context] Lightweight PEM failed: %s", ex)

    # -- 9. Catalog hits (single section, no PEM split) --
    ft_ids = set(workspace.fulltext_chunk_ids())
    not_yet_fetched = [h for h in workspace.catalog_hits if h.chunk_id not in ft_ids]

    # Soft diminishing-returns penalty (display-only, not stored): break ties for under-rep collections
    overrep_lambda = float(os.getenv("CATALOG_OVERREP_LAMBDA", "0"))
    if overrep_lambda > 0 and not_yet_fetched:
        visible_n = min(30, len(not_yet_fetched))
        ft_coll_counts = Counter(
            c.collection_slug or c.source_label or ""
            for c in workspace.fulltext_chunks
        )
        cat_coll_counts = Counter(
            (h.collection or h.collection_slug or "") for h in not_yet_fetched[:visible_n]
        )
        total_visible = sum(ft_coll_counts.values()) + sum(cat_coll_counts.values())
        overrep = set()
        if total_visible > 0:
            for c, cnt in (ft_coll_counts + cat_coll_counts).items():
                if c and (cnt / total_visible) > 0.6:
                    overrep.add(c)
        if overrep:
            def _adj_score(h):
                s = h.score or 0.0
                coll = h.collection or h.collection_slug or ""
                return s - overrep_lambda if coll in overrep else s
            not_yet_fetched = sorted(not_yet_fetched, key=_adj_score, reverse=True)

    # Collection-aware sampler: interleave non-dominant corpora so agent sees diverse entry points
    use_sampler = os.getenv("CATALOG_DIVERSITY_SAMPLER", "1") != "0"
    top_k_display = int(os.getenv("CATALOG_TOP_K", "8"))
    per_coll = int(os.getenv("CATALOG_PER_COLLECTION", "2"))
    if use_sampler and not_yet_fetched:
        top_k_dom = min(30, len(not_yet_fetched))
        coll_counts = Counter(
            (h.collection or h.collection_slug or "") for h in not_yet_fetched[:top_k_dom]
        )
        dominant = coll_counts.most_common(1)[0][0] if coll_counts else ""
        queues: dict = {}
        for h in not_yet_fetched:
            c = h.collection or h.collection_slug or ""
            if c and c != dominant:
                queues.setdefault(c, []).append(h)
        out: list = []
        added_per_coll: dict = {}
        colls = list(queues.keys())
        while len(out) < max_catalog_rows:
            added = False
            for c in colls:
                if added_per_coll.get(c, 0) >= per_coll:
                    continue
                if queues[c]:
                    out.append(queues[c].pop(0))
                    added_per_coll[c] = added_per_coll.get(c, 0) + 1
                    added = True
                    if len(out) >= max_catalog_rows:
                        break
            if not added:
                break
        seen = {h.chunk_id for h in out}
        for h in not_yet_fetched:
            if len(out) >= max_catalog_rows:
                break
            if h.chunk_id not in seen:
                out.append(h)
                seen.add(h.chunk_id)
        not_yet_fetched = out

    cat_header = f"\nEvidence catalog ({len(workspace.catalog_hits)} total, {len(not_yet_fetched)} not yet fetched):\n"
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
        _add(f"  ... and {len(not_yet_fetched) - cat_shown} more hits not shown\n")

    # -- 10. Uncertainty flags --
    if workspace.uncertainty_flags:
        uf = "Uncertainty:\n"
        for u in workspace.uncertainty_flags[-3:]:
            uf += f"  ? {u[:150]}\n"
        _add(uf)

    return "".join(parts)
