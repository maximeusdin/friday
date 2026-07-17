"""
V10 Central Alias Resolver — resolve_alias_candidates().

Single resolution path used by extraction, verifier, and renderer.
Prevents divergence between how different pipeline stages interpret aliases.

Dependency direction (no circular imports):
    extract -> resolve -> lexicon (read-only lookup)
                       -> DB (alias_referent_rules, entity_aliases)

    lexicon.update_from_mentions <- extract output (ChunkMentionsV10)

resolve NEVER calls extract.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    AliasContext,
    AliasReferentRule,
    AliasMappingHypothesis,
    LexiconV10,
    ResolvedAlias,
    SpanCandidate,
)

logger = logging.getLogger(__name__)

# Referent rule status priority for tie-breaking
_RULE_STATUS_PRIORITY = {"confirmed": 3, "possible": 2, "rejected": 1}


# =============================================================================
# Referent rule precedence (deterministic tie-breaking)
# =============================================================================

def _select_best_referent_rules(
    rules: List[AliasReferentRule],
    page_no: Optional[int],
) -> List[AliasReferentRule]:
    """Apply deterministic precedence to select the best referent rule(s).

    Precedence order:
    1. Page-scoped beats doc-wide (page_from is not None)
    2. Narrower interval wins (smallest page_to - page_from)
    3. Status priority: confirmed > possible > rejected
    4. If still tied: return all remaining (agent handles)
    """
    if not rules:
        return []

    # Filter to rules that cover the given page
    matching = [r for r in rules if r.covers_page(page_no) and r.status != "rejected"]
    if not matching:
        return []

    # Partition: page-scoped vs doc-wide
    page_scoped = [r for r in matching if r.page_from is not None]
    doc_wide = [r for r in matching if r.page_from is None]

    # Prefer page-scoped over doc-wide
    candidates = page_scoped if page_scoped else doc_wide

    if len(candidates) <= 1:
        return candidates

    # Among page-scoped: prefer narrowest interval
    if page_scoped:
        min_width = min(r.interval_width or 0 for r in candidates)
        candidates = [r for r in candidates if (r.interval_width or 0) == min_width]
        if len(candidates) <= 1:
            return candidates

    # Status priority tie-break
    max_prio = max(_RULE_STATUS_PRIORITY.get(r.status, 0) for r in candidates)
    candidates = [
        r for r in candidates
        if _RULE_STATUS_PRIORITY.get(r.status, 0) == max_prio
    ]

    return candidates


# =============================================================================
# Main resolver
# =============================================================================

def resolve_alias_candidates(
    conn,
    alias_text: str,
    context: AliasContext,
    lexicon: LexiconV10,
    alias_table_cache: Optional[Dict[str, List[SpanCandidate]]] = None,
) -> ResolvedAlias:
    """Central alias resolution — single path for all consumers.

    Resolution chain (most specific to most general):
    1. Referent rules (doc/page-scoped, from DB / lexicon cache)
    2. Contextual hypothesis (doc-scoped in lexicon)
    3. General hypothesis (collection-wide in lexicon)
    4. Alias table fallback (entity_aliases in DB)
    5. No match → status=unknown

    Returns a ResolvedAlias which is NEVER persisted (recomputed on rehydrate).
    """
    result = ResolvedAlias(
        alias_text=alias_text,
        context_document_id=context.document_id,
        context_page_no=context.page_no,
    )

    # Skip resolution entirely if not in an alias-scoped collection
    if context.collection_slug not in ALIAS_SCOPED_COLLECTIONS:
        result.status = "unknown"
        result.source = "out_of_scope"
        return result

    alias_lower = alias_text.lower().strip()
    if not alias_lower:
        return result

    # --- Step 1: Referent rules ---
    if context.document_id is not None:
        rules = lexicon.get_referent_rules(
            context.collection_slug, alias_lower, context.document_id
        )
        best = _select_best_referent_rules(rules, context.page_no)

        if len(best) == 1 and best[0].status == "confirmed":
            rule = best[0]
            result.locked_entity_id = rule.entity_id
            result.status = "confirmed"
            result.source = "referent_rule"
            result.candidates = [SpanCandidate(
                entity_id=rule.entity_id,
                canonical_name="",  # caller can enrich
                match_type="codename",
                valid_collections=[context.collection_slug],
                source="referent_rule",
            )]
            return result

        if best:
            # Multiple or 'possible' rules — add as candidates
            for rule in best:
                result.candidates.append(SpanCandidate(
                    entity_id=rule.entity_id,
                    canonical_name="",
                    match_type="codename",
                    collision="med" if len(best) > 1 else "low",
                    valid_collections=[context.collection_slug],
                    source="referent_rule",
                ))
            if len(best) == 1 and best[0].status == "possible":
                result.status = "provisional"
                result.source = "referent_rule"
                result.locked_entity_id = best[0].entity_id
            else:
                result.status = "ambiguous"
                result.source = "referent_rule"
            return result

    # --- Step 2: Contextual hypothesis ---
    if context.document_id is not None:
        ctx_hyps = lexicon.get_contextual_hypotheses(
            context.collection_slug, alias_lower, context.document_id
        )
        for h in ctx_hyps:
            if h.status in ("confirmed", "provisional"):
                if h.candidates:
                    result.candidates = list(h.candidates)
                    result.locked_entity_id = h.candidates[0].entity_id
                result.status = h.status
                result.source = "contextual_hypothesis"
                return result

    # --- Step 3: General hypothesis ---
    gen_h = lexicon.get_general_hypothesis(context.collection_slug, alias_lower)
    if gen_h and gen_h.status in ("confirmed", "provisional"):
        if gen_h.candidates:
            result.candidates = list(gen_h.candidates)
            result.locked_entity_id = gen_h.candidates[0].entity_id
        result.status = gen_h.status
        result.source = "general_hypothesis"
        return result

    # --- Step 4: Alias table fallback (DB or batch cache) ---
    if alias_table_cache is not None and alias_lower in alias_table_cache:
        candidates = alias_table_cache[alias_lower]
    elif conn is not None:
        try:
            candidates = _lookup_alias_table(conn, alias_lower)
        except Exception as e:
            candidates = []
            logger.warning("Alias table lookup failed for '%s': %s", alias_text, e)
    else:
        candidates = []

    if len(candidates) == 1:
        result.candidates = candidates
        result.locked_entity_id = candidates[0].entity_id
        result.status = "confirmed"
        result.source = "alias_table"
        return result
    elif candidates:
        result.candidates = candidates
        result.status = "ambiguous"
        result.source = "alias_table"
        return result

    # --- Step 5: No match ---
    result.status = "unknown"
    result.source = "none"
    return result


# =============================================================================
# DB helpers
# =============================================================================


def _lookup_alias_table_batch(
    conn, alias_lowers: List[str]
) -> Dict[str, List[SpanCandidate]]:
    """Batch lookup entity_aliases for many alias texts in one query.

    Returns dict: alias_lower -> list of SpanCandidate (same shape as _lookup_alias_table).
    Only includes keys that had at least one row; missing aliases are omitted.
    """
    out: Dict[str, List[SpanCandidate]] = {}
    if not alias_lowers or conn is None:
        return out
    # Deduplicate and filter empty
    keys = list({a.strip().lower() for a in alias_lowers if a and a.strip()})
    if not keys:
        return out
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT LOWER(ea.alias) AS alias_norm, e.id, e.canonical_name, e.entity_type, ea.kind
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE LOWER(ea.alias) = ANY(%s)
                ORDER BY alias_norm, e.id
                """,
                (keys,),
            )
            for row in cur.fetchall():
                alias_norm, entity_id, canonical, etype, kind = row
                if alias_norm not in out:
                    out[alias_norm] = []
                lst = out[alias_norm]
                # Per-alias limit 10 (match single lookup behavior)
                if len(lst) >= 10:
                    continue
                lst.append(
                    SpanCandidate(
                        entity_id=entity_id,
                        canonical_name=canonical or "",
                        match_type="alias",
                        alias_type=kind,
                        collision="low" if len(lst) == 0 else "med",
                        valid_collections=(
                            ["venona", "vassiliev"]
                            if kind in ("code_name", "covername")
                            else ["*"]
                        ),
                        source="alias_table",
                    )
                )
        for alias_norm in out:
            lst = out[alias_norm]
            if len(lst) > 2:
                for c in lst:
                    c.collision = "high"
            elif len(lst) > 1:
                for c in lst:
                    c.collision = "med"
    except Exception as e:
        logger.warning("Batch alias table lookup failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return out


def _lookup_alias_table(conn, alias_lower: str) -> List[SpanCandidate]:
    """Look up entity_aliases table for the given alias text."""
    candidates: List[SpanCandidate] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT e.id, e.canonical_name, e.entity_type, ea.kind
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE LOWER(ea.alias) = %s
                LIMIT 10
            """, (alias_lower,))
            for row in cur.fetchall():
                entity_id, canonical, etype, kind = row
                candidates.append(SpanCandidate(
                    entity_id=entity_id,
                    canonical_name=canonical or "",
                    match_type="alias",
                    alias_type=kind,
                    collision="low" if len(candidates) == 0 else "med",
                    valid_collections=["venona", "vassiliev"] if kind in ("code_name", "covername") else ["*"],
                    source="alias_table",
                ))
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    # Update collision based on final count
    if len(candidates) > 2:
        for c in candidates:
            c.collision = "high"
    elif len(candidates) > 1:
        for c in candidates:
            c.collision = "med"
    return candidates


def load_referent_rules_for_evidence(
    conn,
    alias_doc_keys: List[tuple],
) -> List[AliasReferentRule]:
    """Load referent rules from DB for a set of (collection_slug, alias_text, document_id) tuples.

    Used during rehydration to repopulate the lexicon cache.
    """
    if not alias_doc_keys or conn is None:
        return []

    rules: List[AliasReferentRule] = []
    try:
        with conn.cursor() as cur:
            for collection_slug, alias_text, document_id in alias_doc_keys:
                cur.execute("""
                    SELECT id, collection_slug, alias_text, document_id,
                           page_from, page_to, entity_id, status, note
                    FROM alias_referent_rules
                    WHERE collection_slug = %s
                      AND LOWER(alias_text) = LOWER(%s)
                      AND document_id = %s
                      AND status != 'rejected'
                    ORDER BY page_from NULLS LAST, page_to NULLS LAST
                """, (collection_slug, alias_text, document_id))
                for row in cur.fetchall():
                    rules.append(AliasReferentRule(
                        rule_id=row[0],
                        collection_slug=row[1],
                        alias_text=row[2],
                        document_id=row[3],
                        page_from=row[4],
                        page_to=row[5],
                        entity_id=row[6],
                        status=row[7],
                        note=row[8] or "",
                    ))
    except Exception as e:
        logger.warning("Failed to load referent rules: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    return rules
