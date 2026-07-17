"""
V9 Grounding (V9.3) - Post-hoc citation binding for claims.

V9.3 changes:
- Identity claims can be grounded against the entity index (not just chunk text)
- If a claim references an alias->canonical mapping that exists in workspace.entities
  or entity_candidates (accepted), it counts as grounded via entity_index.

V9.4+ changes:
- ground_roster_entries(): Validate roster support_chunk_ids against workspace chunks.
  Only roster entries with valid citations are considered grounded.

V9.5 (provenance-first):
- Overlap can NEVER produce status="grounded". Only provenance (model-provided
  citation_chunk_ids or valid evidence spans) can.
- Overlap scoring suggests candidate chunk IDs only; stored in note as suggested_chunk_ids,
  never attached as citation_chunk_ids (avoids "echo = grounded" circularity).
- Entity index without model citations: status="weak", not grounded.
"""
from typing import List

from retrieval.agent.v9_types import (
    ResearchWorkspace,
    V9Claim,
    GroundedClaim,
    GroundedRosterEntry,
    RosterEntry,
    WorkspaceChunk,
    EvidenceSpanRef,
)
from retrieval.agent.v9_spans import get_sentence_count


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _claim_chunk_overlap(claim: V9Claim, chunk: WorkspaceChunk) -> float:
    """Fraction of claim words that appear in chunk text."""
    claim_words = set(_normalize(claim.text).split())
    if not claim_words:
        return 0.0
    chunk_words = set(_normalize(chunk.text).split())
    if not chunk_words:
        return 0.0
    return len(claim_words & chunk_words) / len(claim_words)


def _claim_matches_entity_index(claim: V9Claim, workspace: ResearchWorkspace) -> bool:
    """
    Check if a claim is an identity assertion that can be grounded against
    the entity index (workspace.entities + accepted entity_candidates).

    An identity claim looks like "X was Y" or "X = Y" where X is an alias
    and Y is a canonical name (or vice versa).

    We check: does the claim text contain both an alias/query_term AND the
    canonical_name of any entity in the workspace?
    """
    claim_lower = _normalize(claim.text)
    if not claim_lower:
        return False

    # Check accepted candidates (most relevant for identity grounding)
    for c in workspace.entity_candidates:
        if c.accepted:
            qt = _normalize(c.query_term)
            cn = _normalize(c.canonical_name)
            if qt and cn and qt in claim_lower and cn in claim_lower:
                return True

    # Check committed entities with their aliases
    for e in workspace.entities:
        cn = _normalize(e.canonical_name)
        if cn and cn in claim_lower:
            for alias in e.aliases:
                al = _normalize(alias)
                if al and al in claim_lower:
                    return True

    return False


def _validate_evidence_spans(
    evidence: List[EvidenceSpanRef],
    loaded_chunk_ids: set,
    chunk_map: dict,
) -> List[int]:
    """
    Validate evidence spans. Returns list of valid chunk_ids for grounding.
    If evidence exists and is valid, use it (authoritative). Otherwise return [].
    """
    if not evidence:
        return []
    valid_ids = []
    for e in evidence:
        if e.chunk_id not in loaded_chunk_ids:
            continue
        chunk = chunk_map.get(e.chunk_id)
        if not chunk:
            continue
        if e.sentence_index is not None:
            n = get_sentence_count(chunk.text)
            if e.sentence_index < 0 or e.sentence_index >= n:
                continue
        valid_ids.append(e.chunk_id)
    return valid_ids


def _derive_claim_entity_links(claim: V9Claim, citation_cids: List[int], workspace: ResearchWorkspace) -> None:
    """Derive linked_entity_ids on a claim from its citation chunks + text matching.

    System-derived, not model-provided. Mutates claim.linked_entity_ids in place.
    Entity links are optional enrichment, never a gate. Claims are promoted based on
    citation_chunk_ids, not linked_entity_ids.
    """
    linked: set = set()
    chunk_map = {c.chunk_id: c for c in workspace.fulltext_chunks}
    for cid in citation_cids:
        c = chunk_map.get(cid)
        if c and c.linked_entity_ids:
            linked.update(c.linked_entity_ids)
    # Also check claim text for entity mentions
    text_lower = claim.text.lower()
    for e in workspace.entities:
        for name in [e.canonical_name] + list(e.aliases):
            if len(name) >= 2 and name.lower() in text_lower:
                linked.add(e.entity_id)
    claim.linked_entity_ids = sorted(linked)


def ground_claims(
    claims: List[V9Claim],
    workspace: ResearchWorkspace,
    min_overlap: float = 0.15,
    strong_overlap: float = 0.35,
) -> List[GroundedClaim]:
    """
    Bind claims to supporting evidence and derive entity linkage.

    Grounding sources (in priority order):
    1. Model-provided citation_chunk_ids (validated against loaded chunks) -> grounded.
    2. Entity index basis: identity mapping exists but no model citations -> weak
       (overlap suggests candidates only, never as citation_chunk_ids).
    3. Chunk text overlap: suggests candidates only -> heuristic (never grounded).

    Overlap can never produce status="grounded". Only provenance (valid citation_chunk_ids
    or valid evidence spans) can. Overlap-derived IDs go in note as suggested_chunk_ids.
    """
    chunks = workspace.fulltext_chunks  # Ground against loaded full text
    loaded_chunk_ids = {c.chunk_id for c in chunks}
    chunk_map = {c.chunk_id: c for c in chunks}
    out = []
    for claim in claims:
        if not claim.requires_citation:
            out.append(GroundedClaim(
                claim=claim, status="grounded", citation_chunk_ids=[],
                note="No citation required",
            ))
            continue

        # --- Priority 0: Evidence spans (authoritative when present) ---
        # If evidence exists and valid, use it. Ignore citation_chunk_ids except as fallback.
        if claim.evidence:
            valid_ids = _validate_evidence_spans(
                claim.evidence, loaded_chunk_ids, chunk_map
            )
            if valid_ids:
                _derive_claim_entity_links(claim, valid_ids, workspace)
                out.append(GroundedClaim(
                    claim=claim, status="grounded",
                    citation_chunk_ids=valid_ids[:5],
                    note="Grounded via evidence spans",
                ))
                continue

        # --- Priority 1: Model-provided citation_chunk_ids ---
        # The model is now required to provide chunk_ids for factual claims.
        # Validate they were actually fetched.
        if claim.citation_chunk_ids:
            valid_ids = [cid for cid in claim.citation_chunk_ids if cid in loaded_chunk_ids]
            if valid_ids:
                _derive_claim_entity_links(claim, valid_ids, workspace)
                out.append(GroundedClaim(
                    claim=claim, status="grounded",
                    citation_chunk_ids=valid_ids[:5],
                    note="Grounded via model-provided citation_chunk_ids",
                ))
                continue
            # Model cited chunks we don't have — fall through to heuristic

        # --- Priority 2: Entity index basis (identity claims) ---
        # Has provenance signal (entity mapping) but no model citation_chunk_ids.
        # Overlap suggests candidates only; never attach as citation_chunk_ids.
        if _claim_matches_entity_index(claim, workspace):
            supporting_ids = []
            for c in chunks:
                if _claim_chunk_overlap(claim, c) >= min_overlap:
                    supporting_ids.append(c.chunk_id)
                    if len(supporting_ids) >= 3:
                        break
            _derive_claim_entity_links(claim, supporting_ids, workspace)
            out.append(GroundedClaim(
                claim=claim, status="weak",
                citation_chunk_ids=[],
                note=f"Entity index (no provenance): suggested {supporting_ids[:5]} for debugging",
            ))
            continue

        # --- Priority 3: Overlap-only (no provenance) ---
        # Overlap suggests candidate chunk IDs only; never attach as citation_chunk_ids.
        # All overlap-derived results get status="heuristic".
        best_score = 0.0
        best_ids: List[int] = []
        for c in chunks:
            score = _claim_chunk_overlap(claim, c)
            if score >= min_overlap:
                if score > best_score:
                    best_score = score
                    best_ids = [c.chunk_id]
                elif score == best_score and c.chunk_id not in best_ids:
                    best_ids.append(c.chunk_id)
        if best_score >= min_overlap and best_ids:
            status = "heuristic"
            note = f"Overlap-only (no provenance): suggested {best_ids[:5]} for debugging"
        else:
            status = "unsupported"
            best_ids = []
            note = "No citation found"
        _derive_claim_entity_links(claim, best_ids, workspace)
        out.append(GroundedClaim(
            claim=claim, status=status,
            citation_chunk_ids=[], note=note,
        ))
    return out


def ground_roster_entries(
    roster: List[RosterEntry],
    workspace: ResearchWorkspace,
) -> List[GroundedRosterEntry]:
    """
    Validate roster entries against workspace chunks.

    Each entry's support_chunk_ids must reference chunks in workspace.fulltext_chunks.
    - grounded: all cited chunks exist and are non-empty
    - weak: some cited chunks exist
    - unsupported: no valid citations
    """
    loaded_chunk_ids = {c.chunk_id for c in workspace.fulltext_chunks}
    out: List[GroundedRosterEntry] = []
    for entry in roster:
        cids = entry.support_chunk_ids or []
        valid_ids = [cid for cid in cids if cid in loaded_chunk_ids]
        if valid_ids and len(valid_ids) == len(cids):
            status = "grounded"
        elif valid_ids:
            status = "weak"
        else:
            status = "unsupported"
        out.append(GroundedRosterEntry(
            entry=entry,
            status=status,
            valid_chunk_ids=valid_ids[:5],
        ))
    return out
