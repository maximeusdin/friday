"""
V10 Lexicon lifecycle — build, update, serialize, rehydrate.

Manages:
- LexiconV10 construction from SpanLattice + initial entity candidates
- Loading AliasReferentRules from DB
- update_from_mentions() with referent-rule-first logic
- Contextual + general hypothesis promotion
- build_entity_forms() for EntityBoost form generation
- Serialisation for ThinkDeeper persistence
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    CODENAME_ALIAS_KINDS,
    AliasContext,
    AliasReferentRule,
    AliasMappingHypothesis,
    ChunkMention,
    ChunkMentionsV10,
    ChunkSignal,
    LexiconV10,
    ResolvedAlias,
    SpanCandidate,
    SpanEntry,
    SpanLatticeV10,
    SpanSelection,
)
from retrieval.agent.v10_resolve import (
    load_referent_rules_for_evidence,
    resolve_alias_candidates,
)

logger = logging.getLogger(__name__)


# =============================================================================
# build_entity_forms()  —  V10.0e
# =============================================================================

def build_entity_forms(
    conn,
    entity_id: int,
    scope: Optional[List[str]] = None,
) -> List[str]:
    """Generate surface forms for EntityBoost.forms[].

    Returns the set of surface forms that should be used during global
    retrieval to match the entity across all corpus surface variants.

    Sources:
    - canonical_name (always)
    - Non-codename aliases (transliterations, alt spellings, etc.) — globally valid
    - Codename aliases (code_name, covername) — ONLY if scope includes venona/vassiliev
    - First/last name forms for person entities (for partial matching)

    This solves the "Semenov" vs "Semenoff" problem.
    """
    forms: List[str] = []
    seen: Set[str] = set()

    def _add(form: str) -> None:
        fl = form.lower().strip()
        if fl and fl not in seen:
            seen.add(fl)
            forms.append(form.strip())

    # Determine if codename aliases should be included
    scope_includes_alias_collections = False
    if scope:
        scope_includes_alias_collections = bool(
            set(scope) & ALIAS_SCOPED_COLLECTIONS
        )

    try:
        with conn.cursor() as cur:
            # 1. Canonical name
            cur.execute(
                "SELECT canonical_name, entity_type FROM entities WHERE id = %s",
                (entity_id,),
            )
            row = cur.fetchone()
            if not row:
                return forms
            canonical_name, entity_type = row
            _add(canonical_name)

            # 2. Known aliases
            cur.execute("""
                SELECT alias, kind FROM entity_aliases
                WHERE entity_id = %s
                ORDER BY kind, alias
            """, (entity_id,))
            for alias, kind in cur.fetchall():
                is_codename = kind in CODENAME_ALIAS_KINDS
                if is_codename and not scope_includes_alias_collections:
                    continue  # exclude codenames unless scope allows
                _add(alias)

            # 3. First/last name forms for person entities
            # Do NOT add first_name alone — common first names (e.g. Jacob from Golos, Jacob)
            # pollute expansion. Keep last_name and "Last, First" for structured aliases.
            if entity_type == "person" and canonical_name:
                parts = canonical_name.strip().split()
                if len(parts) >= 2:
                    first_name = parts[0]
                    last_name = parts[-1]
                    _add(last_name)
                    _add(f"{last_name}, {first_name}")

    except Exception as e:
        logger.warning("build_entity_forms failed for entity %d: %s", entity_id, e)
        try:
            conn.rollback()
        except Exception:
            pass

    return forms


# =============================================================================
# Lexicon construction
# =============================================================================

def build_lexicon_from_lattice(
    conn,
    lattice: SpanLatticeV10,
    selection: Optional[SpanSelection] = None,
) -> LexiconV10:
    """Build initial LexiconV10 from a SpanLattice (and optional span selection).

    Registers all entities referenced by selected spans (or all spans if
    no selection) and seeds the alias namespace indexes.
    """
    lex = LexiconV10()

    # Determine which spans to use
    if selection and selection.chosen_span_ids:
        chosen_ids = set(selection.chosen_span_ids)
        spans_to_use = [s for s in lattice.spans if s.span_id in chosen_ids]
    else:
        spans_to_use = lattice.spans

    # Register entities and aliases from span candidates
    for span in spans_to_use:
        for cand in span.candidates:
            # Register entity
            lex.register_entity(
                entity_id=cand.entity_id,
                canonical_name=cand.canonical_name,
                global_variants=[],
                entity_type=None,
            )

            # Seed scoped alias namespaces
            if cand.match_type in ("alias", "codename") and cand.alias_type:
                for coll in cand.valid_collections:
                    if coll == "*":
                        continue
                    # aliases_by_entity_scoped
                    lex.aliases_by_entity_scoped.setdefault(
                        cand.entity_id, {}
                    ).setdefault(coll, [])
                    if span.text.lower() not in [
                        a.lower() for a in lex.aliases_by_entity_scoped[cand.entity_id][coll]
                    ]:
                        lex.aliases_by_entity_scoped[cand.entity_id][coll].append(span.text)

                    # entities_by_alias_scoped
                    lex.entities_by_alias_scoped.setdefault(coll, {}).setdefault(
                        span.text.lower(), []
                    )
                    if cand.entity_id not in lex.entities_by_alias_scoped[coll][span.text.lower()]:
                        lex.entities_by_alias_scoped[coll][span.text.lower()].append(cand.entity_id)

            # Seed general hypotheses for alias spans
            if cand.match_type in ("alias", "codename"):
                for coll in cand.valid_collections:
                    if coll == "*":
                        continue
                    gen_key = (coll, span.text.lower(), None, None, None)
                    if gen_key not in lex.alias_mapping_hypotheses:
                        lex.set_hypothesis(AliasMappingHypothesis(
                            collection_slug=coll,
                            alias_text=span.text.lower(),
                            candidates=[cand],
                            status="unresolved",
                        ))
                    else:
                        # Add candidate to existing hypothesis if not already present
                        existing = lex.alias_mapping_hypotheses[gen_key]
                        existing_ids = {c.entity_id for c in existing.candidates}
                        if cand.entity_id not in existing_ids:
                            existing.candidates.append(cand)

    return lex


def load_referent_rules_into_lexicon(
    conn,
    lexicon: LexiconV10,
    alias_doc_keys: Optional[List[Tuple[str, str, int]]] = None,
) -> None:
    """Load AliasReferentRules from DB into the lexicon cache.

    If alias_doc_keys is None, loads rules for all (collection, alias, document)
    tuples present in the lexicon's existing evidence.
    """
    if alias_doc_keys is None:
        # Derive keys from current evidence
        alias_doc_keys = []
        for key, hyp in lexicon.alias_mapping_hypotheses.items():
            if hyp.document_id is not None:
                alias_doc_keys.append((hyp.collection_slug, hyp.alias_text, hyp.document_id))

    rules = load_referent_rules_for_evidence(conn, alias_doc_keys)
    for rule in rules:
        lexicon.add_referent_rule(rule)


# =============================================================================
# update_from_mentions()
# =============================================================================

# Signal types that can trigger hypothesis promotion
_STRONG_SIGNALS = {"alias_equation", "identified_as"}
_MODERATE_SIGNALS = {"aka", "cryptonym_marker", "parenthetical"}


def update_from_mentions(
    conn,
    lexicon: LexiconV10,
    mentions: ChunkMentionsV10,
    alias_table_cache: Optional[Dict[str, List[SpanCandidate]]] = None,
) -> None:
    """Update lexicon from extracted chunk mentions.

    Called AFTER extraction completes — never from within resolve_alias_candidates().
    This keeps the dependency direction acyclic.

    If alias_table_cache is provided (e.g. from _lookup_alias_table_batch), Step 4
    of the resolver uses it instead of per-alias DB lookups.

    For each alias_surface mention:
    1. Consult referent rules for (collection, alias, document_id, page_no)
    2. If rule matched (confirmed) -> attach contextual mapping support, optionally confirm
    3. If no rule hit -> fall back to collection-level alias candidates + signals
    4. Signals can trigger contextual hypothesis creation or promotion
    """
    collection = mentions.collection_slug
    doc_id = mentions.document_id
    page_no = mentions.page_no

    # Only process alias mentions in alias-scoped collections
    if collection not in ALIAS_SCOPED_COLLECTIONS:
        return

    for mention in mentions.mentions:
        if mention.kind != "alias_surface":
            continue

        alias_lower = mention.surface.lower().strip()
        if not alias_lower:
            continue

        # Use the resolver to get the current resolution
        context = AliasContext(
            collection_slug=collection,
            document_id=doc_id,
            page_no=page_no,
        )
        resolved = resolve_alias_candidates(
            conn, alias_lower, context, lexicon,
            alias_table_cache=alias_table_cache,
        )

        support_entry = {
            "chunk_id": mentions.chunk_id,
            "signal_type": "mention",
            "document_id": doc_id,
            "page_no": page_no,
        }

        # Handle contextual hypothesis
        if resolved.source == "referent_rule" and resolved.locked_entity_id:
            # Referent rule matched — ensure contextual hypothesis reflects this
            _ensure_contextual_hypothesis(
                lexicon, collection, alias_lower, doc_id, page_no,
                entity_id=resolved.locked_entity_id,
                status="confirmed" if resolved.status == "confirmed" else "provisional",
                support_entry=support_entry,
                candidates=resolved.candidates,
            )
        elif resolved.source in ("contextual_hypothesis", "general_hypothesis") and resolved.locked_entity_id:
            # Existing hypothesis — add evidence support
            _add_support_to_hypothesis(
                lexicon, collection, alias_lower, doc_id,
                support_entry=support_entry,
            )
        elif resolved.source == "alias_table" and resolved.locked_entity_id:
            # Unambiguous alias table lookup — create/update contextual hypothesis
            _ensure_contextual_hypothesis(
                lexicon, collection, alias_lower, doc_id, page_no,
                entity_id=resolved.locked_entity_id,
                status="provisional",
                support_entry=support_entry,
                candidates=resolved.candidates,
            )
        else:
            # Ambiguous or unknown — create/update general hypothesis with candidates
            _ensure_general_hypothesis(
                lexicon, collection, alias_lower,
                candidates=resolved.candidates,
                support_entry=support_entry,
            )

        # Register any resolved entity
        if resolved.locked_entity_id is not None:
            for cand in resolved.candidates:
                if cand.entity_id == resolved.locked_entity_id:
                    lexicon.register_entity(
                        entity_id=cand.entity_id,
                        canonical_name=cand.canonical_name,
                    )
                    lexicon.add_entity_evidence(cand.entity_id, mentions.chunk_id)
                    break

    # Process signals for hypothesis promotion
    _process_signals_for_promotion(lexicon, mentions)


def _ensure_contextual_hypothesis(
    lexicon: LexiconV10,
    collection: str,
    alias_lower: str,
    doc_id: int,
    page_no: Optional[int],
    entity_id: int,
    status: str,
    support_entry: Dict[str, Any],
    candidates: Optional[List[SpanCandidate]] = None,
) -> None:
    """Create or update a contextual hypothesis for (collection, alias, document)."""
    existing = lexicon.get_hypothesis(collection, alias_lower, doc_id)
    if existing:
        # Update existing
        if support_entry not in existing.support:
            existing.support.append(support_entry)
        # Promote if new status is stronger
        status_order = {"unresolved": 0, "provisional": 1, "confirmed": 2}
        if status_order.get(status, 0) > status_order.get(existing.status, 0):
            existing.status = status
            existing.confidence = min(1.0, existing.confidence + 0.2)
    else:
        # Create new contextual hypothesis
        lexicon.set_hypothesis(AliasMappingHypothesis(
            collection_slug=collection,
            alias_text=alias_lower,
            candidates=candidates or [],
            status=status,
            confidence=0.6 if status == "confirmed" else 0.3,
            support=[support_entry],
            document_id=doc_id,
            page_from=page_no,
            page_to=page_no,
        ))


def _add_support_to_hypothesis(
    lexicon: LexiconV10,
    collection: str,
    alias_lower: str,
    doc_id: int,
    support_entry: Dict[str, Any],
) -> None:
    """Add support evidence to an existing contextual or general hypothesis."""
    # Try contextual first
    ctx_hyps = lexicon.get_contextual_hypotheses(collection, alias_lower, doc_id)
    for h in ctx_hyps:
        if support_entry not in h.support:
            h.support.append(support_entry)
            h.confidence = min(1.0, h.confidence + 0.1)
        return

    # Fall back to general
    gen = lexicon.get_general_hypothesis(collection, alias_lower)
    if gen:
        if support_entry not in gen.support:
            gen.support.append(support_entry)
            gen.confidence = min(1.0, gen.confidence + 0.05)


def _ensure_general_hypothesis(
    lexicon: LexiconV10,
    collection: str,
    alias_lower: str,
    candidates: List[SpanCandidate],
    support_entry: Dict[str, Any],
) -> None:
    """Create or update a general (collection-wide) hypothesis."""
    gen = lexicon.get_general_hypothesis(collection, alias_lower)
    if gen:
        if support_entry not in gen.support:
            gen.support.append(support_entry)
        # Add any new candidates
        existing_ids = {c.entity_id for c in gen.candidates}
        for c in candidates:
            if c.entity_id not in existing_ids:
                gen.candidates.append(c)
                existing_ids.add(c.entity_id)
    else:
        lexicon.set_hypothesis(AliasMappingHypothesis(
            collection_slug=collection,
            alias_text=alias_lower,
            candidates=list(candidates),
            status="unresolved",
            confidence=0.1,
            support=[support_entry],
            document_id=None,  # general
        ))


def _process_signals_for_promotion(
    lexicon: LexiconV10,
    mentions: ChunkMentionsV10,
) -> None:
    """Use detected signals to promote alias hypotheses.

    Strong signals (alias_equation, identified_as) can promote to confirmed.
    Moderate signals (aka, cryptonym_marker, parenthetical) can promote to provisional.
    """
    collection = mentions.collection_slug
    doc_id = mentions.document_id
    page_no = mentions.page_no

    if collection not in ALIAS_SCOPED_COLLECTIONS:
        return

    for signal in mentions.signals:
        if not signal.entity_a:
            continue

        alias_lower = signal.entity_a.lower().strip()
        support_entry = {
            "chunk_id": mentions.chunk_id,
            "signal_type": signal.signal_type,
            "document_id": doc_id,
            "page_no": page_no,
        }

        if signal.signal_type in _STRONG_SIGNALS:
            # Strong signal -> can confirm contextual hypothesis
            if signal.entity_b:
                # Try to find the entity_id for entity_b
                entity_id = _find_entity_id_by_name(lexicon, signal.entity_b)
                if entity_id:
                    _ensure_contextual_hypothesis(
                        lexicon, collection, alias_lower, doc_id, page_no,
                        entity_id=entity_id,
                        status="confirmed",
                        support_entry=support_entry,
                    )
        elif signal.signal_type in _MODERATE_SIGNALS:
            # Moderate signal -> add support, potentially promote to provisional
            ctx_hyps = lexicon.get_contextual_hypotheses(collection, alias_lower, doc_id)
            for h in ctx_hyps:
                if h.status == "unresolved":
                    h.status = "provisional"
                    h.confidence = max(h.confidence, 0.4)
                if support_entry not in h.support:
                    h.support.append(support_entry)


def _find_entity_id_by_name(lexicon: LexiconV10, name: str) -> Optional[int]:
    """Try to find an entity_id in the lexicon by canonical name."""
    name_lower = name.lower().strip()
    for eid, info in lexicon.entities_in_play.items():
        if info.get("canonical_name", "").lower().strip() == name_lower:
            return eid
    return None


# =============================================================================
# Serialisation / Rehydration
# =============================================================================

# =============================================================================
# Alias backfill — cross-collection alias namespace population
# =============================================================================

def backfill_alias_namespace(
    conn,
    lexicon: LexiconV10,
    entity_ids: Optional[Set[int]] = None,
    max_aliases_per_entity: int = 10,
) -> Dict[int, List[str]]:
    """For each entity in entities_in_play (or specified subset), query entity_aliases
    for Venona/Vassiliev codenames and populate the scoped alias namespaces.

    Returns {entity_id: [newly_added_aliases]} for downstream use.

    This is deterministic and cheap (single DB query per batch).

    Note: entity_aliases has no per-collection provenance column. Aliases with
    kind='code_name' are populated into BOTH venona and vassiliev scoped namespaces.
    If per-collection provenance is added to entity_aliases in the future, filter
    by valid_collections here instead.
    """
    target_ids = list(entity_ids or lexicon.entities_in_play.keys())
    # Skip entities already backfilled
    target_ids = [eid for eid in target_ids if eid not in lexicon._backfilled_entity_ids]
    if not target_ids:
        return {}

    new_aliases: Dict[int, List[str]] = {}

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, alias, kind
                FROM entity_aliases
                WHERE entity_id = ANY(%s)
                  AND kind IN ('code_name')
                ORDER BY entity_id, kind, alias
            """, (target_ids,))
            rows = cur.fetchall()

        for entity_id, alias_text, kind in rows:
            alias_lower = alias_text.lower()
            # Populate aliases_by_entity_scoped for both venona and vassiliev
            for coll in ALIAS_SCOPED_COLLECTIONS:
                lex_aliases = lexicon.aliases_by_entity_scoped.setdefault(
                    entity_id, {}
                ).setdefault(coll, [])
                if alias_text not in lex_aliases and len(lex_aliases) < max_aliases_per_entity:
                    lex_aliases.append(alias_text)
                    new_aliases.setdefault(entity_id, []).append(alias_text)

                # Reverse index
                lex_entities = lexicon.entities_by_alias_scoped.setdefault(
                    coll, {}
                ).setdefault(alias_lower, [])
                if entity_id not in lex_entities:
                    lex_entities.append(entity_id)

        lexicon._backfilled_entity_ids.update(target_ids)

    except Exception as e:
        logger.warning("backfill_alias_namespace failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    return new_aliases


def serialize_lexicon(lexicon: LexiconV10) -> Dict[str, Any]:
    """Serialise lexicon for persistence (ThinkDeeper resume state)."""
    return lexicon.to_dict()


def rehydrate_lexicon(
    conn,
    data: Dict[str, Any],
    evidence_mentions: Optional[List[ChunkMentionsV10]] = None,
) -> LexiconV10:
    """Rehydrate LexiconV10 from persisted data + DB.

    1. Load persisted state (entities, hypotheses)
    2. Re-load referent rules from DB
    3. Optionally re-process mentions for fresh resolution
    """
    lexicon = LexiconV10.from_dict(data)

    # Collect (collection, alias, doc_id) keys from hypotheses
    alias_doc_keys: List[Tuple[str, str, int]] = []
    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        if hyp.document_id is not None:
            alias_doc_keys.append((hyp.collection_slug, hyp.alias_text, hyp.document_id))

    # Also collect from evidence mentions
    if evidence_mentions:
        for cm in evidence_mentions:
            if cm.collection_slug in ALIAS_SCOPED_COLLECTIONS:
                for mention in cm.mentions:
                    if mention.kind == "alias_surface" and cm.document_id:
                        alias_doc_keys.append(
                            (cm.collection_slug, mention.surface.lower(), cm.document_id)
                        )

    # De-duplicate
    alias_doc_keys = list(set(alias_doc_keys))

    # Re-load referent rules from DB (they may have been updated between runs)
    if alias_doc_keys and conn is not None:
        load_referent_rules_into_lexicon(conn, lexicon, alias_doc_keys)

    return lexicon
