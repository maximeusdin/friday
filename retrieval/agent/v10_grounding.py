"""
V10 Grounding — Entity-aware, scope-aware claim grounding.

Extends V9 grounding with:
- Entity-ID-based satisfaction (not just text overlap)
- Contextual alias resolution via resolve_alias_candidates()
- Collection-scoped alias validation
- Key invariant: a chunk from doc A where PAL=X does NOT serve as
  evidence for entity Y even if PAL maps to Y in doc B

Uses the central resolver for consistent alias interpretation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    AliasContext,
    ChunkMentionsV10,
    LexiconV10,
    ResolvedAlias,
)
from retrieval.agent.v10_resolve import resolve_alias_candidates
from retrieval.agent.v9_types import (
    GroundedClaim,
    V9Claim,
    WorkspaceChunk,
)

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


# =============================================================================
# Entity-aware chunk satisfaction
# =============================================================================

def _chunk_satisfies_entity(
    conn,
    entity_id: int,
    chunk: WorkspaceChunk,
    chunk_mentions: Optional[ChunkMentionsV10],
    lexicon: LexiconV10,
) -> bool:
    """Does this chunk contain evidence for the given entity?

    Satisfaction via:
    1. Canonical name or global variant appears in chunk (any collection)
    2. Alias/codename appears in chunk AND:
       - chunk is in Venona/Vassiliev, AND
       - resolve_alias_candidates() for this chunk's doc/page context
         returns the entity with status confirmed or provisional
    """
    entity_info = lexicon.entities_in_play.get(entity_id, {})
    canonical = entity_info.get("canonical_name", "")
    chunk_lower = _normalize(chunk.text)

    # Check 1: canonical name or global variant
    if canonical and _normalize(canonical) in chunk_lower:
        return True

    for variant in entity_info.get("global_variants", []):
        if variant and _normalize(variant) in chunk_lower:
            return True

    # Check 2: alias/codename in alias-scoped collection
    collection = chunk.collection_slug or ""
    if collection not in ALIAS_SCOPED_COLLECTIONS:
        return False

    # Use chunk mentions if available
    if chunk_mentions:
        for mention in chunk_mentions.mentions:
            if mention.kind != "alias_surface":
                continue

            # Use the resolved alias (if available) or resolve now
            resolved = mention.resolved
            if not resolved:
                context = AliasContext(
                    collection_slug=collection,
                    document_id=chunk_mentions.document_id,
                    page_no=chunk_mentions.page_no,
                )
                resolved = resolve_alias_candidates(
                    conn, mention.surface, context, lexicon
                )

            # Check if resolution points to our entity
            if (
                resolved
                and resolved.locked_entity_id == entity_id
                and resolved.status in ("confirmed", "provisional")
            ):
                return True
    else:
        # No mentions available — check aliases from lexicon
        aliases_for_entity = lexicon.aliases_by_entity_scoped.get(entity_id, {})
        collection_aliases = aliases_for_entity.get(collection, [])

        for alias in collection_aliases:
            if _normalize(alias) in chunk_lower:
                # Need to verify this alias resolves to our entity in this doc
                context = AliasContext(
                    collection_slug=collection,
                    document_id=chunk.doc_id,
                    page_no=_parse_page_no(chunk.page),
                )
                resolved = resolve_alias_candidates(
                    conn, alias, context, lexicon
                )
                if (
                    resolved
                    and resolved.locked_entity_id == entity_id
                    and resolved.status in ("confirmed", "provisional")
                ):
                    return True

    return False


def _parse_page_no(page_str: Optional[str]) -> Optional[int]:
    """Parse 'p5' -> 5."""
    if not page_str:
        return None
    if page_str.startswith("p"):
        try:
            return int(page_str[1:])
        except ValueError:
            return None
    return None


# =============================================================================
# Claim-level entity extraction
# =============================================================================

def _extract_claim_entity_ids(
    claim_text: str,
    lexicon: LexiconV10,
) -> Set[int]:
    """Extract entity_ids referenced in a claim's text."""
    entity_ids: Set[int] = set()
    text_lower = _normalize(claim_text)

    for eid, info in lexicon.entities_in_play.items():
        canonical = info.get("canonical_name", "")
        if canonical and _normalize(canonical) in text_lower:
            entity_ids.add(eid)

        for variant in info.get("global_variants", []):
            if variant and _normalize(variant) in text_lower:
                entity_ids.add(eid)

    return entity_ids


# =============================================================================
# Main grounding function
# =============================================================================

def ground_claims_v10(
    conn,
    claims: List[V9Claim],
    chunks: List[WorkspaceChunk],
    chunk_mentions: Dict[int, ChunkMentionsV10],
    lexicon: LexiconV10,
    min_overlap: float = 0.15,
    strong_overlap: float = 0.35,
) -> List[GroundedClaim]:
    """Entity-aware, scope-aware claim grounding.

    Grounding sources (in priority order):
    1. Model-provided citation_chunk_ids (validated + entity-checked)
    2. Entity-ID satisfaction via resolve_alias_candidates()
    3. Standard chunk text overlap (fallback)

    Key invariant: an alias in doc A confirmed as entity X does NOT
    serve as evidence for entity Y, even if the same alias maps to Y
    in doc B.
    """
    loaded_chunk_ids = {c.chunk_id for c in chunks}
    chunk_map = {c.chunk_id: c for c in chunks}

    out: List[GroundedClaim] = []

    for claim in claims:
        if not claim.requires_citation:
            out.append(GroundedClaim(
                claim=claim, status="grounded", citation_chunk_ids=[],
                note="No citation required",
            ))
            continue

        # Extract entity_ids from claim text
        claim_entity_ids = _extract_claim_entity_ids(claim.text, lexicon)

        # --- Priority 1: Model-provided citation_chunk_ids ---
        if claim.citation_chunk_ids:
            valid_ids = [cid for cid in claim.citation_chunk_ids if cid in loaded_chunk_ids]
            if valid_ids:
                # Verify entity satisfaction for each cited chunk
                verified_ids = []
                for cid in valid_ids:
                    chunk = chunk_map.get(cid)
                    if not chunk:
                        continue
                    cm = chunk_mentions.get(cid)
                    if not claim_entity_ids:
                        # No entity context — accept based on text overlap
                        verified_ids.append(cid)
                    else:
                        # Check if chunk satisfies at least one claim entity
                        for eid in claim_entity_ids:
                            if _chunk_satisfies_entity(conn, eid, chunk, cm, lexicon):
                                verified_ids.append(cid)
                                break
                        else:
                            # Fallback: accept if text overlap is strong enough
                            overlap = _claim_chunk_overlap(claim, chunk)
                            if overlap >= min_overlap:
                                verified_ids.append(cid)

                if verified_ids:
                    claim.linked_entity_ids = sorted(claim_entity_ids)
                    out.append(GroundedClaim(
                        claim=claim, status="grounded",
                        citation_chunk_ids=verified_ids[:5],
                        note="Grounded via model citations + entity verification",
                    ))
                    continue

        # --- Priority 2: Entity-ID satisfaction ---
        if claim_entity_ids:
            supporting_ids = []
            for chunk in chunks:
                cm = chunk_mentions.get(chunk.chunk_id)
                for eid in claim_entity_ids:
                    if _chunk_satisfies_entity(conn, eid, chunk, cm, lexicon):
                        supporting_ids.append(chunk.chunk_id)
                        break
                if len(supporting_ids) >= 5:
                    break

            if supporting_ids:
                claim.linked_entity_ids = sorted(claim_entity_ids)
                out.append(GroundedClaim(
                    claim=claim, status="grounded",
                    citation_chunk_ids=supporting_ids[:5],
                    note="Grounded via entity-ID satisfaction",
                ))
                continue

        # --- Priority 3: Standard chunk text overlap ---
        best_score = 0.0
        best_ids: List[int] = []
        for chunk in chunks:
            score = _claim_chunk_overlap(claim, chunk)
            if score >= min_overlap:
                if score > best_score:
                    best_score = score
                    best_ids = [chunk.chunk_id]
                elif score == best_score and chunk.chunk_id not in best_ids:
                    best_ids.append(chunk.chunk_id)

        if best_score >= strong_overlap and best_ids:
            status = "grounded"
            note = None
        elif best_score >= min_overlap and best_ids:
            status = "weak"
            note = "Partial or indirect evidence"
        else:
            status = "unsupported"
            best_ids = []
            note = "No citation found"

        claim.linked_entity_ids = sorted(claim_entity_ids)
        out.append(GroundedClaim(
            claim=claim, status=status,
            citation_chunk_ids=best_ids[:5], note=note,
        ))

    return out


def _claim_chunk_overlap(claim: V9Claim, chunk: WorkspaceChunk) -> float:
    """Fraction of claim words that appear in chunk text."""
    claim_words = set(_normalize(claim.text).split())
    if not claim_words:
        return 0.0
    chunk_words = set(_normalize(chunk.text).split())
    if not chunk_words:
        return 0.0
    return len(claim_words & chunk_words) / len(claim_words)
