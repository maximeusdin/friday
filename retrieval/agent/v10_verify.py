"""
V10 Verifier + Renderer — Alias-annotated rendering and scope-aware verification.

V10.3b: Two responsibilities:
1. **Verification**: Check entity_id satisfaction via contextual-first alias resolution.
   A chunk from doc A where PAL=X does NOT serve as evidence for entity Y.
2. **Rendering**: Per-chunk alias annotation via resolve_alias_candidates().
   The same alias renders differently in different document quotes:
   - PAL (Person X) in doc A
   - PAL (Person Y) in doc B

Both use the central resolver for consistent interpretation.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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


# =============================================================================
# Scope-aware verification
# =============================================================================

def verify_claim_v10(
    conn,
    claim: GroundedClaim,
    chunks: Dict[int, WorkspaceChunk],
    chunk_mentions: Dict[int, ChunkMentionsV10],
    lexicon: LexiconV10,
) -> GroundedClaim:
    """Verify a grounded claim using entity-ID + contextual alias resolution.

    For each cited chunk:
    - If chunk contains canonical name/global variant -> valid
    - If chunk is in Venona/Vassiliev and contains alias:
      - Resolve via resolve_alias_candidates() with chunk's doc/page context
      - Valid only if resolution points to the claim's entity with confirmed/provisional status

    Returns the claim with updated status and notes.
    """
    if claim.status == "unsupported":
        return claim

    if not claim.citation_chunk_ids:
        return claim

    entity_ids = set(getattr(claim.claim, 'linked_entity_ids', []) or [])
    if not entity_ids:
        # No entity context — can't verify, keep existing status
        return claim

    verified_chunks: List[int] = []
    violation_notes: List[str] = []

    for cid in claim.citation_chunk_ids:
        chunk = chunks.get(cid)
        if not chunk:
            continue

        cm = chunk_mentions.get(cid)
        chunk_valid = False
        chunk_lower = chunk.text.lower()

        for eid in entity_ids:
            entity_info = lexicon.entities_in_play.get(eid, {})
            canonical = entity_info.get("canonical_name", "")

            # Check canonical name
            if canonical and canonical.lower() in chunk_lower:
                chunk_valid = True
                break

            # Check global variants
            for variant in entity_info.get("global_variants", []):
                if variant and variant.lower() in chunk_lower:
                    chunk_valid = True
                    break
            if chunk_valid:
                break

            # Check alias resolution (Venona/Vassiliev only)
            collection = chunk.collection_slug or ""
            if collection in ALIAS_SCOPED_COLLECTIONS and cm:
                for mention in cm.mentions:
                    if mention.kind != "alias_surface":
                        continue

                    resolved = mention.resolved
                    if not resolved:
                        context = AliasContext(
                            collection_slug=collection,
                            document_id=cm.document_id,
                            page_no=cm.page_no,
                        )
                        resolved = resolve_alias_candidates(
                            conn, mention.surface, context, lexicon
                        )

                    if (
                        resolved
                        and resolved.locked_entity_id == eid
                        and resolved.status in ("confirmed", "provisional")
                    ):
                        chunk_valid = True
                        break

                if chunk_valid:
                    break

        if chunk_valid:
            verified_chunks.append(cid)
        else:
            violation_notes.append(
                f"chunk {cid} ({chunk.collection_slug}): "
                f"alias resolution does not confirm entity {list(entity_ids)}"
            )

    # Update claim status based on verification
    if verified_chunks:
        claim.citation_chunk_ids = verified_chunks[:5]
        if violation_notes:
            claim.note = (claim.note or "") + " | Partial verification: " + "; ".join(violation_notes[:3])
        else:
            claim.note = (claim.note or "") + " | V10 entity-verified"
    else:
        claim.status = "unsupported"
        claim.note = "V10 verification failed: " + "; ".join(violation_notes[:3])

    return claim


def verify_claims_v10(
    conn,
    claims: List[GroundedClaim],
    chunks: Dict[int, WorkspaceChunk],
    chunk_mentions: Dict[int, ChunkMentionsV10],
    lexicon: LexiconV10,
) -> List[GroundedClaim]:
    """Verify all grounded claims using V10 entity-aware verification."""
    return [
        verify_claim_v10(conn, claim, chunks, chunk_mentions, lexicon)
        for claim in claims
    ]


# =============================================================================
# Alias-annotated rendering
# =============================================================================

def render_chunk_with_aliases(
    conn,
    chunk: WorkspaceChunk,
    chunk_mentions: Optional[ChunkMentionsV10],
    lexicon: LexiconV10,
) -> str:
    """Render a chunk with alias annotations.

    For Venona/Vassiliev chunks containing aliases:
    - If resolved (confirmed/provisional): annotate "PAL (Nathan Gregory Silvermaster)"
    - If ambiguous: annotate "PAL (unresolved — possible: X, Y)"
    - If unknown: leave unannotated

    The same alias renders differently in different document quotes
    because resolution is contextual (doc/page-specific).

    Returns annotated text.
    """
    collection = chunk.collection_slug or ""
    text = chunk.text

    if collection not in ALIAS_SCOPED_COLLECTIONS:
        return text

    if not chunk_mentions or not chunk_mentions.mentions:
        return text

    # Collect alias annotations (sorted by position, reversed for replacement)
    annotations: List[Tuple[int, int, str, str]] = []  # (start, end, original, annotated)

    for mention in chunk_mentions.mentions:
        if mention.kind != "alias_surface":
            continue

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

        if not resolved:
            continue

        if resolved.status in ("confirmed", "provisional"):
            # Find canonical name
            canonical = ""
            if resolved.locked_entity_id:
                entity_info = lexicon.entities_in_play.get(resolved.locked_entity_id, {})
                canonical = entity_info.get("canonical_name", "")
                if not canonical and resolved.candidates:
                    canonical = resolved.candidates[0].canonical_name

            if canonical:
                status_tag = "" if resolved.status == "confirmed" else " [provisional]"
                annotated = f"{mention.surface} ({canonical}{status_tag})"
                annotations.append((mention.start, mention.end, mention.surface, annotated))

        elif resolved.status == "ambiguous":
            possible = ", ".join(
                c.canonical_name for c in resolved.candidates[:3] if c.canonical_name
            )
            if possible:
                annotated = f"{mention.surface} (unresolved — possible: {possible})"
                annotations.append((mention.start, mention.end, mention.surface, annotated))

    if not annotations:
        return text

    # Apply annotations in reverse order (to preserve positions)
    annotations.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate overlapping annotations (keep the first one at each position)
    seen_positions: Set[int] = set()
    unique_annotations = []
    for start, end, orig, annotated in annotations:
        if start not in seen_positions:
            seen_positions.add(start)
            unique_annotations.append((start, end, orig, annotated))

    result = text
    for start, end, orig, annotated in unique_annotations:
        if start < len(result) and result[start:end] == orig:
            result = result[:start] + annotated + result[end:]

    return result


def render_narrative_with_aliases(
    conn,
    narrative: str,
    chunks: Dict[int, WorkspaceChunk],
    chunk_mentions: Dict[int, ChunkMentionsV10],
    lexicon: LexiconV10,
) -> str:
    """Post-process the narrative to annotate any alias references.

    Looks for chunk quotes within the narrative and annotates aliases.
    Also annotates alias references in the narrative text itself.
    """
    # For each alias that has a confirmed/provisional mapping, replace
    # bare alias mentions in the narrative with annotated form
    result = narrative

    for key, hyp in lexicon.alias_mapping_hypotheses.items():
        if hyp.status not in ("confirmed", "provisional"):
            continue
        if not hyp.candidates:
            continue

        alias = hyp.alias_text
        canonical = hyp.candidates[0].canonical_name
        if not alias or not canonical:
            continue

        # Only annotate if alias is in ALL CAPS (likely a codename)
        if not alias.isupper():
            continue

        # Replace bare codename with annotated form (case-sensitive)
        # But only outside of existing annotations
        pattern = re.compile(
            r'\b' + re.escape(alias) + r'\b(?!\s*\()',
            re.IGNORECASE
        )

        scope_note = ""
        if hyp.is_contextual:
            scope_note = f" in doc {hyp.document_id}"

        replacement = f"{alias} ({canonical}{scope_note})"
        result = pattern.sub(replacement, result, count=5)  # limit replacements

    return result


# =============================================================================
# Mapping provenance for rendering
# =============================================================================

def get_mapping_provenance(
    lexicon: LexiconV10,
    alias_text: str,
    collection_slug: str,
    document_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Get the provenance trail for a specific alias mapping.

    Returns information about which chunk/signal established the mapping.
    Used for rendering the evidence trail.
    """
    result: Dict[str, Any] = {
        "alias": alias_text,
        "collection": collection_slug,
        "document_id": document_id,
        "status": "unknown",
        "source": "none",
        "support": [],
    }

    alias_lower = alias_text.lower()

    # Check contextual hypothesis first
    if document_id is not None:
        ctx_hyps = lexicon.get_contextual_hypotheses(collection_slug, alias_lower, document_id)
        for h in ctx_hyps:
            if h.status in ("confirmed", "provisional"):
                result["status"] = h.status
                result["source"] = "contextual_hypothesis"
                result["entity"] = h.candidates[0].canonical_name if h.candidates else "?"
                result["support"] = h.support[:5]
                return result

    # Check general hypothesis
    gen = lexicon.get_general_hypothesis(collection_slug, alias_lower)
    if gen and gen.status in ("confirmed", "provisional"):
        result["status"] = gen.status
        result["source"] = "general_hypothesis"
        result["entity"] = gen.candidates[0].canonical_name if gen.candidates else "?"
        result["support"] = gen.support[:5]
        return result

    return result
