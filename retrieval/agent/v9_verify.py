"""
V9 Verify (V9.4) - Advisory verification with generic artifact checks.

V9.4 changes:
- Removed task-type-specific audits (_audit_roster, _audit_timeline, etc.)
- Replaced with generic _audit_artifact that checks artifact shape and coverage
- build_verification_report no longer requires ResponsivenessResult or TaskType
- Optional auditor LLM is in v9_runner.py (not here)
"""
from typing import List, Optional, Dict, Any

from retrieval.agent.v9_types import (
    V9VerificationReport,
    GroundedClaim,
    V9Synthesis,
    ResponsivenessResult,
)


def _audit_artifact(synthesis: V9Synthesis) -> List[str]:
    """
    Generic artifact audit -- interface-level checks, not task-type gates.

    Checks each known artifact key for emptiness and support_chunk_ids coverage.
    Reports artifact keys present for trace/logging.
    """
    notes: List[str] = []
    artifact = synthesis.artifact

    if not artifact:
        notes.append("No artifact sections provided.")
        return notes

    # Check each known artifact key
    if "roster" in artifact:
        entries = artifact["roster"]
        if not entries:
            notes.append("Artifact has roster key but it is empty.")
        elif isinstance(entries, list):
            with_support = sum(1 for e in entries if isinstance(e, dict) and e.get("support_chunk_ids"))
            notes.append(f"Roster: {len(entries)} entries, {with_support} with support_chunk_ids.")

    if "identity" in artifact:
        ident = artifact["identity"]
        if isinstance(ident, dict):
            if not ident.get("canonical"):
                notes.append("Identity section missing canonical name.")
            if not ident.get("support_chunk_ids") and not ident.get("basis"):
                notes.append("Identity section has no support_chunk_ids or basis.")
        else:
            notes.append("Identity section is not a dict.")

    if "timeline" in artifact:
        entries = artifact["timeline"]
        if not entries:
            notes.append("Artifact has timeline key but it is empty.")
        elif isinstance(entries, list):
            with_support = sum(1 for e in entries if isinstance(e, dict) and e.get("support_chunk_ids"))
            notes.append(f"Timeline: {len(entries)} entries, {with_support} with support_chunk_ids.")

    if "evidence" in artifact:
        entries = artifact["evidence"]
        if not entries:
            notes.append("Artifact has evidence key but it is empty.")
        elif isinstance(entries, list):
            with_chunks = sum(1 for e in entries if isinstance(e, dict) and e.get("chunk_id"))
            notes.append(f"Evidence: {len(entries)} entries, {with_chunks} with chunk_id.")

    if "relationships" in artifact:
        entries = artifact["relationships"]
        if not entries:
            notes.append("Artifact has relationships key but it is empty.")
        elif isinstance(entries, list):
            with_support = sum(1 for e in entries if isinstance(e, dict) and e.get("support_chunk_ids"))
            notes.append(f"Relationships: {len(entries)} entries, {with_support} with support_chunk_ids.")

    # Summary of keys present
    notes.append(f"Artifact keys present: {list(artifact.keys())}")

    return notes


def build_verification_report(
    grounded_claims: List[GroundedClaim],
    synthesis: Optional[V9Synthesis] = None,
    responsiveness: Optional[ResponsivenessResult] = None,
    grounded_roster: Optional[List] = None,
) -> V9VerificationReport:
    """Build advisory report: grounding stats + artifact audit."""
    grounded = sum(1 for g in grounded_claims if g.status == "grounded")
    weak = sum(1 for g in grounded_claims if g.status == "weak")
    heuristic = sum(1 for g in grounded_claims if g.status == "heuristic")
    unsupported = sum(1 for g in grounded_claims if g.status == "unsupported")
    # weak_claims in report = weak + heuristic (unverified)
    weak_total = weak + heuristic
    notes: List[str] = []
    artifact_notes: List[str] = []

    if weak:
        notes.append(f"{weak} claim(s) have partial evidence (entity index, no provenance).")
    if heuristic:
        notes.append(f"{heuristic} claim(s) are overlap-only (no provenance).")
    if unsupported:
        notes.append(f"{unsupported} claim(s) have no citation found (kept but marked).")

    for g in grounded_claims:
        if g.note and g.status != "grounded":
            notes.append(f"Claim: \"{g.claim.text[:60]}...\" -- {g.note}")

    # Artifact audit (generic, not task-type-specific)
    if synthesis:
        artifact_notes = _audit_artifact(synthesis)

    # Roster grounding summary
    if grounded_roster is not None:
        g_count = sum(1 for gr in grounded_roster if gr.status == "grounded")
        w_count = sum(1 for gr in grounded_roster if gr.status == "weak")
        u_count = sum(1 for gr in grounded_roster if gr.status == "unsupported")
        if grounded_roster:
            artifact_notes.append(
                f"Roster grounding: {g_count} grounded, {w_count} weak, {u_count} unsupported (filtered from display)."
            )

    return V9VerificationReport(
        grounded_claims=grounded,
        weak_claims=weak_total,
        unsupported_claims=unsupported,
        responsiveness=responsiveness,
        artifact_notes=artifact_notes[:15],
        notes=notes[:15],
    )
