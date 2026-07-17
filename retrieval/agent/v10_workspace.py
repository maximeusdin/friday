"""
V10 Workspace — Merge helpers adapted for V10 types.

Handles persistence and rehydration of V10 identity artifacts
for ThinkDeeper integration:
- SpanLattice serialization
- ChunkMentions persistence (on evidence_items)
- LexiconV10 state (on v9_runs resume_state_json)
- Mapping hypotheses (on v9_runs mapping_hypotheses_json)
- Referent rule reloading from DB

PERSIST contract:
  SpanLatticeV10         -> v9_runs.span_lattice_json
  ChunkMentionsV10       -> evidence_items.chunk_mentions_json (per chunk)
  AliasMappingHypothesis -> v9_runs.mapping_hypotheses_json
  ResolutionPlanV10      -> v9_runs.resolution_plan_json (optional)

RECOMPUTE on rehydrate:
  alias_referent_rules   -> loaded from DB for seen (alias, doc) keys
  ResolvedAlias per mention -> recomputed from chain (avoids stale locks)
  LexiconV10 global layer -> rebuilt from SpanLattice + ChunkMentions + hypotheses
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    AliasContext,
    AliasMappingHypothesis,
    ChunkMentionsV10,
    LexiconV10,
    ResolutionPlanV10,
    SpanLatticeV10,
)
from retrieval.agent.v10_lexicon import (
    load_referent_rules_into_lexicon,
    rehydrate_lexicon,
)
from retrieval.agent.v10_resolve import resolve_alias_candidates

logger = logging.getLogger(__name__)


# =============================================================================
# Persist V10 artifacts
# =============================================================================

def persist_v10_run_state(
    conn,
    v9_run_id: int,
    lattice: Optional[SpanLatticeV10] = None,
    lexicon: Optional[LexiconV10] = None,
    plan: Optional[ResolutionPlanV10] = None,
) -> None:
    """Persist V10 artifacts to v9_runs table.

    Updates the existing v9_runs row with V10-specific JSON columns.
    """
    updates: List[str] = []
    params: List[Any] = []

    if lattice is not None:
        updates.append("span_lattice_json = %s")
        params.append(json.dumps(lattice.to_dict()))

    if lexicon is not None:
        # Persist mapping hypotheses separately
        hypotheses = {}
        for key, hyp in lexicon.alias_mapping_hypotheses.items():
            hypotheses[str(key)] = hyp.to_dict()
        updates.append("mapping_hypotheses_json = %s")
        params.append(json.dumps(hypotheses))

    if plan is not None:
        updates.append("resolution_plan_json = %s")
        params.append(json.dumps(plan.to_dict()))

    if not updates:
        return

    params.append(v9_run_id)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE v9_runs SET {', '.join(updates)} WHERE id = %s",
                params,
            )
        conn.commit()
    except Exception as e:
        logger.warning("Failed to persist V10 run state: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass


def persist_chunk_mentions(
    conn,
    evidence_set_id: int,
    chunk_mentions: Dict[int, ChunkMentionsV10],
) -> None:
    """Persist ChunkMentionsV10 onto evidence_items.chunk_mentions_json.

    Each evidence item gets the mentions for its chunk, with embedded
    document_id and page_no for rehydration independence.
    """
    if not chunk_mentions:
        return

    try:
        with conn.cursor() as cur:
            for chunk_id, cm in chunk_mentions.items():
                cm_json = json.dumps(cm.to_dict())
                cur.execute("""
                    UPDATE evidence_items
                    SET chunk_mentions_json = %s
                    WHERE evidence_set_id = %s AND chunk_id = %s
                """, (cm_json, evidence_set_id, chunk_id))
        conn.commit()
    except Exception as e:
        logger.warning("Failed to persist chunk mentions: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass


# =============================================================================
# Rehydrate V10 artifacts
# =============================================================================

def load_v10_run_state(
    conn,
    v9_run_id: int,
) -> Tuple[Optional[SpanLatticeV10], Optional[Dict[str, Any]], Optional[ResolutionPlanV10]]:
    """Load V10 artifacts from v9_runs table.

    Returns (lattice, hypotheses_raw, plan) — hypotheses are raw dict
    because LexiconV10 rehydration needs to merge them with other state.
    """
    lattice = None
    hypotheses_raw = None
    plan = None

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT span_lattice_json, mapping_hypotheses_json, resolution_plan_json
                FROM v9_runs
                WHERE id = %s
            """, (v9_run_id,))
            row = cur.fetchone()
            if row:
                if row[0]:
                    data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    lattice = SpanLatticeV10.from_dict(data)
                if row[1]:
                    hypotheses_raw = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                if row[2]:
                    data = row[2] if isinstance(row[2], dict) else json.loads(row[2])
                    plan = ResolutionPlanV10.from_dict(data)
    except Exception as e:
        logger.warning("Failed to load V10 run state: %s", e)

    return lattice, hypotheses_raw, plan


def load_chunk_mentions_from_evidence(
    conn,
    evidence_set_id: int,
) -> Dict[int, ChunkMentionsV10]:
    """Load persisted ChunkMentionsV10 from evidence_items."""
    mentions: Dict[int, ChunkMentionsV10] = {}

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, chunk_mentions_json
                FROM evidence_items
                WHERE evidence_set_id = %s
                  AND chunk_mentions_json IS NOT NULL
            """, (evidence_set_id,))
            for row in cur.fetchall():
                chunk_id, cm_json = row
                if cm_json:
                    data = cm_json if isinstance(cm_json, dict) else json.loads(cm_json)
                    mentions[chunk_id] = ChunkMentionsV10.from_dict(data)
    except Exception as e:
        logger.warning("Failed to load chunk mentions: %s", e)

    return mentions


def rehydrate_v10_state(
    conn,
    v9_run_id: int,
    evidence_set_id: int,
) -> Tuple[Optional[SpanLatticeV10], Optional[LexiconV10], Dict[int, ChunkMentionsV10], Optional[ResolutionPlanV10]]:
    """Full V10 rehydration for ThinkDeeper resume.

    Steps:
    1. Load SpanLattice, hypotheses, plan from v9_runs
    2. Load ChunkMentionsV10 from evidence_items
    3. Rebuild LexiconV10:
       a. Start from persisted hypotheses
       b. Re-load referent rules from DB (may have been updated)
       c. Fold in entity state from lattice + mentions
    4. Re-run resolve_alias_candidates() for each alias mention
       (ensures contextual mappings are current, avoids stale locks)

    Returns (lattice, lexicon, chunk_mentions, plan)
    """
    # Step 1: Load run state
    lattice, hypotheses_raw, plan = load_v10_run_state(conn, v9_run_id)

    # Step 2: Load chunk mentions
    chunk_mentions = load_chunk_mentions_from_evidence(conn, evidence_set_id)

    # Step 3: Rebuild lexicon
    lexicon_data: Dict[str, Any] = {}
    if hypotheses_raw:
        lexicon_data["alias_mapping_hypotheses"] = hypotheses_raw

    lexicon = rehydrate_lexicon(
        conn,
        lexicon_data,
        evidence_mentions=list(chunk_mentions.values()),
    )

    # If we have a lattice, register entities from it
    if lattice:
        for span in lattice.spans:
            for cand in span.candidates:
                lexicon.register_entity(
                    entity_id=cand.entity_id,
                    canonical_name=cand.canonical_name,
                )

    # Register entities from chunk mentions
    for cm in chunk_mentions.values():
        for mention in cm.mentions:
            if mention.kind == "entity_surface":
                for cand in mention.candidates:
                    lexicon.register_entity(
                        entity_id=cand.entity_id,
                        canonical_name=cand.canonical_name,
                    )
                    lexicon.add_entity_evidence(cand.entity_id, cm.chunk_id)

    # Step 4: Re-resolve alias mentions (recompute ResolvedAlias)
    for cm in chunk_mentions.values():
        if cm.collection_slug not in ALIAS_SCOPED_COLLECTIONS:
            continue
        context = AliasContext(
            collection_slug=cm.collection_slug,
            document_id=cm.document_id,
            page_no=cm.page_no,
        )
        for mention in cm.mentions:
            if mention.kind == "alias_surface":
                try:
                    mention.resolved = resolve_alias_candidates(
                        conn, mention.surface, context, lexicon
                    )
                except Exception as e:
                    logger.debug("Re-resolution failed for '%s': %s", mention.surface, e)

    return lattice, lexicon, chunk_mentions, plan


# =============================================================================
# V10 novelty metrics
# =============================================================================

def compute_v10_novelty(
    lexicon: LexiconV10,
    prev_lexicon: Optional[LexiconV10] = None,
) -> Dict[str, Any]:
    """Compute V10-specific novelty metrics for ThinkDeeper scoring.

    Returns a dict of novelty items that can be added to the judge's
    delta scoring.
    """
    metrics: Dict[str, Any] = {
        "new_mapping_confirmed": 0,
        "ambiguity_reduced": 0,
        "new_collection_covered_for_entity": 0,
        "new_evidence_support_for_entity": 0,
        "context_mapping_confirmed": 0,
        "new_doc_resolved_for_alias": 0,
    }

    if prev_lexicon is None:
        return metrics

    # Count new confirmed mappings
    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        prev_hyp = prev_lexicon.alias_mapping_hypotheses.get(key)
        if hyp.status == "confirmed" and (not prev_hyp or prev_hyp.status != "confirmed"):
            metrics["new_mapping_confirmed"] += 1
            if hyp.is_contextual:
                metrics["context_mapping_confirmed"] += 1

        # Ambiguity reduced
        if prev_hyp and hyp.status != "unresolved" and prev_hyp.status == "unresolved":
            metrics["ambiguity_reduced"] += 1

    # New documents resolved for aliases
    prev_doc_alias_keys: Set[Tuple[str, str, int]] = set()
    for key, hyp in prev_lexicon.alias_mapping_hypotheses.items():
        if hyp.is_contextual and hyp.status in ("confirmed", "provisional"):
            prev_doc_alias_keys.add((hyp.collection_slug, hyp.alias_text, hyp.document_id))

    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        if hyp.is_contextual and hyp.status in ("confirmed", "provisional"):
            doc_key = (hyp.collection_slug, hyp.alias_text, hyp.document_id)
            if doc_key not in prev_doc_alias_keys:
                metrics["new_doc_resolved_for_alias"] += 1

    # New entity evidence
    for eid, support in lexicon.entity_support.items():
        prev_support = prev_lexicon.entity_support.get(eid, {})
        new_chunks = set(support.get("support_chunks", [])) - set(
            prev_support.get("support_chunks", [])
        )
        if new_chunks:
            metrics["new_evidence_support_for_entity"] += 1

    return metrics
