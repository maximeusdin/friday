"""
V10.3 Exploration/Exploitation Controller — deterministic policy layer.

This module provides a scoreboard, coverage map, gap detection, and mode
selection that guides the LLM's investigation without overriding its tool
choices. It injects "Strategy Hint JSON" into the conversation context.

Invariants:
- Alias surfaces are only extracted + used in Venona/Vassiliev.
- Backfill adds alias namespaces, never mappings.
- Mappings are evidence-backed; contextual rules override general hypotheses.
- Locked alias->entity only when context-safe.
- Verification is entity-id based and context-aware.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    CatalogHitV10,
    LexiconV10,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Policy dataclasses
# =============================================================================

@dataclass
class GapItem:
    """A detected gap in investigation coverage."""
    gap_type: str  # "unseen_collection", "unresolved_alias", "low_evidence_entity", "unexplored_backlink"
    description: str
    priority: float = 1.0  # higher = more important
    suggested_action: str = ""
    target_collection: Optional[str] = None
    target_entity_id: Optional[int] = None
    target_alias: Optional[str] = None


@dataclass
class CoverageMapV10:
    """Tracks which collections and documents have been searched."""
    collections_searched: Dict[str, int] = field(default_factory=dict)  # slug -> search_count
    documents_seen: Set[int] = field(default_factory=set)
    chunks_by_collection: Dict[str, int] = field(default_factory=dict)  # slug -> chunk_count
    total_searches: int = 0
    total_chunks: int = 0

    def collection_coverage_ratio(self, known_collections: List[str]) -> float:
        """What fraction of known collections have been searched at least once."""
        if not known_collections:
            return 1.0
        searched = sum(1 for c in known_collections if c in self.collections_searched)
        return searched / len(known_collections)


@dataclass
class ScoreboardV10:
    """Tracks investigation progress metrics."""
    entities_discovered: int = 0
    contextual_mappings_confirmed: int = 0
    contextual_mappings_provisional: int = 0
    ambiguous_aliases: int = 0
    total_support_chunks: int = 0
    backlinks_used: int = 0
    round_number: int = 0

    # History for trend detection
    entity_history: List[int] = field(default_factory=list)
    mapping_history: List[int] = field(default_factory=list)


@dataclass
class ExploreExploitHint:
    """The strategy hint delivered to the LLM."""
    mode: str  # "explore" or "exploit"
    reason: str
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    suggested_searches: List[Dict[str, Any]] = field(default_factory=list)
    stall_warning: Optional[str] = None


# =============================================================================
# Coverage + Scoreboard update functions
# =============================================================================

def update_coverage_from_hits(
    coverage: CoverageMapV10,
    hits: List[CatalogHitV10],
) -> None:
    """Update coverage map from a list of search hits."""
    for h in hits:
        if h.collection_slug:
            coverage.collections_searched[h.collection_slug] = (
                coverage.collections_searched.get(h.collection_slug, 0) + 1
            )
            coverage.chunks_by_collection[h.collection_slug] = (
                coverage.chunks_by_collection.get(h.collection_slug, 0) + 1
            )
        if h.document_id:
            coverage.documents_seen.add(h.document_id)
        coverage.total_chunks += 1
    coverage.total_searches += 1


def update_scoreboard_from_enrichment(
    scoreboard: ScoreboardV10,
    lexicon: LexiconV10,
) -> None:
    """Update scoreboard from current lexicon state."""
    scoreboard.entities_discovered = len(lexicon.entities_in_play)
    scoreboard.total_support_chunks = sum(
        len(info.get("evidence_chunk_ids", []))
        for info in lexicon.entities_in_play.values()
    )

    confirmed = 0
    provisional = 0
    ambiguous = 0
    for hyp in lexicon.alias_mapping_hypotheses.values():
        if hyp.status == "confirmed":
            confirmed += 1
        elif hyp.status == "provisional":
            provisional += 1
        elif hyp.status in ("unresolved", "ambiguous"):
            ambiguous += 1

    scoreboard.contextual_mappings_confirmed = confirmed
    scoreboard.contextual_mappings_provisional = provisional
    scoreboard.ambiguous_aliases = ambiguous
    scoreboard.round_number += 1

    # Record history
    scoreboard.entity_history.append(scoreboard.entities_discovered)
    scoreboard.mapping_history.append(confirmed + provisional)

    # Count backlinks used
    scoreboard.backlinks_used = sum(
        1 for eid in lexicon.entities_in_play
        if lexicon.aliases_by_entity_scoped.get(eid)
    )


# =============================================================================
# Gap detection
# =============================================================================

# Known collections that should ideally be covered
KNOWN_COLLECTIONS = [
    "venona", "vassiliev", "fbi_files", "nsa_files",
    "congressional_records", "academic_sources",
]


def compute_gaps(
    coverage: CoverageMapV10,
    scoreboard: ScoreboardV10,
    lexicon: LexiconV10,
    lattice: Optional[Any] = None,
    last_warnings: Optional[List[str]] = None,
) -> List[GapItem]:
    """Detect gaps in investigation coverage.

    Gap types:
    1. Unseen collections: collections not yet searched
    2. Unresolved aliases: aliases with ambiguous/unresolved status
    3. Low-evidence entities: entities with <2 support chunks
    4. Unexplored backlinks: entities with alias backlinks not yet searched
    5. Unresolved referents: selected spans with no candidates
    6. Ambiguous referents: selected spans with multiple candidates
    7. Alias surface needs index: alias span present but no index lookup yet
    8. Alias power blocked: agent attempted alias boosts without permission
    """
    gaps: List[GapItem] = []

    # 5. Unresolved referents (from lattice)
    if lattice is not None and hasattr(lattice, 'spans'):
        for span in lattice.spans:
            if span.resolution_status == "unresolved" and not span.candidates:
                gaps.append(GapItem(
                    gap_type="unresolved_referent",
                    description=(
                        f"Span '{span.text}' (norm_key='{span.norm_key}') has no candidates"
                    ),
                    priority=2.5,
                    suggested_action=(
                        f"Call resolve_referent_v10(surface_text='{span.text}', mode='broad') "
                        f"to expand candidate set"
                    ),
                    target_alias=span.norm_key,
                ))
            elif span.resolution_status == "ambiguous":
                gaps.append(GapItem(
                    gap_type="ambiguous_referent",
                    description=(
                        f"Span '{span.text}' has {len(span.candidates)} candidates"
                    ),
                    priority=2.0,
                    suggested_action=(
                        f"Call resolve_referent_v10 or alias_index_sample_v10 "
                        f"to disambiguate '{span.text}'"
                    ),
                    target_alias=span.norm_key,
                ))
            # 7. Alias surface needs index lookup
            if span.span_kind == "alias_surface":
                gaps.append(GapItem(
                    gap_type="alias_surface_needs_index",
                    description=(
                        f"Alias span '{span.text}' may benefit from alias_index_summary/lookup"
                    ),
                    priority=1.5,
                    suggested_action=(
                        f"Call alias_index_summary_v10(alias_surface='{span.text}') "
                        f"to check occurrence distribution"
                    ),
                    target_alias=span.norm_key,
                ))

    # 8. Alias power blocked (detect from last tool warnings)
    if last_warnings:
        for w in last_warnings:
            if "alias_boost_dropped_no_permission" in w:
                gaps.append(GapItem(
                    gap_type="alias_power_blocked",
                    description=w,
                    priority=3.0,
                    suggested_action=(
                        "Call grant_alias_power_v10 for the blocked alias before retrying"
                    ),
                ))

    # 1. Unseen collections
    for coll in KNOWN_COLLECTIONS:
        if coll not in coverage.collections_searched:
            gaps.append(GapItem(
                gap_type="unseen_collection",
                description=f"Collection '{coll}' has not been searched yet",
                priority=2.0 if coll in ALIAS_SCOPED_COLLECTIONS else 1.0,
                suggested_action=f"Search '{coll}' with relevant entity boosts",
                target_collection=coll,
            ))

    # 2. Unresolved/ambiguous aliases (only alias-scoped collections)
    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        if hyp.status in ("unresolved", "ambiguous"):
            if hyp.collection_slug in ALIAS_SCOPED_COLLECTIONS:
                n_cands = len(hyp.candidates)
                gaps.append(GapItem(
                    gap_type="unresolved_alias",
                    description=(
                        f"Alias '{hyp.alias_text}' in {hyp.collection_slug} "
                        f"doc {hyp.document_id} has {n_cands} candidates"
                    ),
                    priority=2.5,
                    suggested_action=(
                        f"Search {hyp.collection_slug} for alias '{hyp.alias_text}' "
                        f"with alias_boosts_scoped to resolve"
                    ),
                    target_collection=hyp.collection_slug,
                    target_alias=hyp.alias_text,
                ))

    # 3. Low-evidence entities
    for eid, info in lexicon.entities_in_play.items():
        evidence = info.get("evidence_chunk_ids", [])
        if len(evidence) < 2:
            canonical = info.get("canonical_name", f"entity_{eid}")
            gaps.append(GapItem(
                gap_type="low_evidence_entity",
                description=f"Entity '{canonical}' (id={eid}) has only {len(evidence)} support chunk(s)",
                priority=1.5,
                suggested_action=f"Search for '{canonical}' across collections",
                target_entity_id=eid,
            ))

    # 4. Unexplored backlinks (entities with aliases but alias collections unsearched for them)
    for eid, info in lexicon.entities_in_play.items():
        alias_data = lexicon.aliases_by_entity_scoped.get(eid, {})
        for coll, aliases in alias_data.items():
            if aliases and coverage.chunks_by_collection.get(coll, 0) < 5:
                canonical = info.get("canonical_name", f"entity_{eid}")
                gaps.append(GapItem(
                    gap_type="unexplored_backlink",
                    description=(
                        f"Entity '{canonical}' has aliases {aliases[:3]} in {coll}, "
                        f"but only {coverage.chunks_by_collection.get(coll, 0)} chunks from {coll}"
                    ),
                    priority=2.0,
                    suggested_action=(
                        f"Search '{coll}' with alias_boosts_scoped for "
                        f"{', '.join(aliases[:2])}"
                    ),
                    target_collection=coll,
                    target_entity_id=eid,
                ))

    # Sort by priority descending
    gaps.sort(key=lambda g: -g.priority)
    return gaps


# =============================================================================
# Mode selection
# =============================================================================

def choose_mode(
    scoreboard: ScoreboardV10,
    gaps: List[GapItem],
) -> str:
    """Choose explore or exploit mode based on scoreboard and gaps.

    Explore when:
    - Many high-priority gaps remain
    - Few entities discovered
    - Early in investigation (< 3 rounds)

    Exploit when:
    - Few gaps remain
    - Multiple entities with evidence
    - Ambiguous aliases that need targeted resolution
    """
    high_priority_gaps = [g for g in gaps if g.priority >= 2.0]
    unresolved_alias_gaps = [g for g in gaps if g.gap_type == "unresolved_alias"]

    # Early rounds: always explore
    if scoreboard.round_number < 3:
        return "explore"

    # Many high-priority gaps: explore
    if len(high_priority_gaps) > 3:
        return "explore"

    # Few entities discovered: explore
    if scoreboard.entities_discovered < 2:
        return "explore"

    # Unresolved aliases need targeted exploitation
    if unresolved_alias_gaps and scoreboard.entities_discovered >= 2:
        return "exploit"

    # Good coverage + entities: exploit
    if scoreboard.entities_discovered >= 3 and scoreboard.total_support_chunks >= 6:
        return "exploit"

    return "explore"


# =============================================================================
# Strategy Hint builder
# =============================================================================

def build_explore_exploit_hint(
    mode: str,
    gaps: List[GapItem],
    scoreboard: ScoreboardV10,
    coverage: CoverageMapV10,
    progress_delta: Any = None,  # ProgressDelta from runner
    stall_rounds: int = 0,
) -> str:
    """Build the Strategy Hint message content for the LLM.

    Returns a formatted string suitable for injection into messages.
    """
    hint = ExploreExploitHint(
        mode=mode,
        reason=_mode_reason(mode, scoreboard, gaps),
    )

    # Top gaps
    for g in gaps[:5]:
        hint.gaps.append({
            "type": g.gap_type,
            "description": g.description,
            "priority": g.priority,
            "action": g.suggested_action,
        })

    # Suggested searches based on mode
    if mode == "explore":
        # Suggest searching unseen collections
        for g in gaps:
            if g.gap_type == "unseen_collection" and g.target_collection:
                hint.suggested_searches.append({
                    "type": "explore_collection",
                    "collection": g.target_collection,
                    "action": g.suggested_action,
                })
            if len(hint.suggested_searches) >= 3:
                break
    else:
        # Suggest resolving aliases, referents, and deepening evidence
        for g in gaps:
            if g.gap_type == "unresolved_alias":
                hint.suggested_searches.append({
                    "type": "resolve_alias",
                    "alias": g.target_alias,
                    "collection": g.target_collection,
                    "action": g.suggested_action,
                })
            elif g.gap_type == "low_evidence_entity":
                hint.suggested_searches.append({
                    "type": "deepen_evidence",
                    "entity_id": g.target_entity_id,
                    "action": g.suggested_action,
                })
            elif g.gap_type == "unresolved_referent":
                hint.suggested_searches.append({
                    "type": "resolve_referent",
                    "norm_key": g.target_alias,
                    "action": g.suggested_action,
                })
            elif g.gap_type == "ambiguous_referent":
                hint.suggested_searches.append({
                    "type": "disambiguate_referent",
                    "norm_key": g.target_alias,
                    "action": g.suggested_action,
                })
            elif g.gap_type == "alias_surface_needs_index":
                hint.suggested_searches.append({
                    "type": "alias_index_check",
                    "alias": g.target_alias,
                    "action": g.suggested_action,
                })
            elif g.gap_type == "alias_power_blocked":
                hint.suggested_searches.append({
                    "type": "grant_alias_power",
                    "action": g.suggested_action,
                })
            if len(hint.suggested_searches) >= 5:
                break

    # Stall warning
    if stall_rounds >= 2:
        hint.stall_warning = (
            f"Investigation has stalled for {stall_rounds} rounds. "
            "Consider broadening search, trying different entity forms, "
            "or synthesizing with current evidence (final=true)."
        )

    # Build the message
    hint_json = {
        "mode": hint.mode,
        "reason": hint.reason,
        "scoreboard": {
            "entities_discovered": scoreboard.entities_discovered,
            "confirmed_mappings": scoreboard.contextual_mappings_confirmed,
            "provisional_mappings": scoreboard.contextual_mappings_provisional,
            "ambiguous_aliases": scoreboard.ambiguous_aliases,
            "total_support_chunks": scoreboard.total_support_chunks,
            "round": scoreboard.round_number,
            "collections_covered": len(coverage.collections_searched),
        },
        "top_gaps": hint.gaps[:5],
        "suggested_actions": hint.suggested_searches[:3],
    }
    if hint.stall_warning:
        hint_json["stall_warning"] = hint.stall_warning

    # Progress delta info
    if progress_delta is not None:
        try:
            hint_json["progress_delta"] = {
                "new_entities": progress_delta.new_entities,
                "new_mappings": progress_delta.new_mappings,
                "ambiguity_reduced": progress_delta.ambiguity_reduced,
                "new_support_chunks": progress_delta.new_support_chunks,
                "score": progress_delta.score,
                "is_stalled": progress_delta.is_stalled,
            }
        except AttributeError:
            pass

    content = (
        f"## Strategy Hint\n"
        f"Mode: **{hint.mode.upper()}** — {hint.reason}\n\n"
        f"```json\n{json.dumps(hint_json, indent=2)}\n```"
    )

    return content


def _mode_reason(mode: str, scoreboard: ScoreboardV10, gaps: List[GapItem]) -> str:
    """Generate a human-readable reason for the chosen mode."""
    if mode == "explore":
        unseen = [g for g in gaps if g.gap_type == "unseen_collection"]
        low_ev = [g for g in gaps if g.gap_type == "low_evidence_entity"]
        parts = []
        if unseen:
            parts.append(f"{len(unseen)} unseen collection(s)")
        if low_ev:
            parts.append(f"{len(low_ev)} low-evidence entity(ies)")
        if scoreboard.round_number < 3:
            parts.append("early in investigation")
        return "Broaden search. " + (", ".join(parts) if parts else "coverage gaps remain")
    else:
        unresolved = [g for g in gaps if g.gap_type == "unresolved_alias"]
        parts = []
        if unresolved:
            parts.append(f"{len(unresolved)} unresolved alias(es) to resolve")
        if scoreboard.entities_discovered >= 3:
            parts.append(f"{scoreboard.entities_discovered} entities with evidence")
        return "Deepen evidence. " + (", ".join(parts) if parts else "good coverage, focus on quality")
