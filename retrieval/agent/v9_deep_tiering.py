"""
V9 Think Deeper — Tiering logic.

Tier 0: Normal agent loop (V9/V11) — fast, 3–5 tool calls
Tier 1: Auto-deepen (small ThinkDeeper) — triggered by quality signals
Tier 2: User-invoked /deeper — full ThinkDeeper, larger budget

Triggers for Tier 1:
- Hard: 0 grounded claims, finalization invalid, sufficiency=false
- Soft: roster intent, narrow coverage, high overlap + low frontier
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from retrieval.agent.v9_deep_rails import _infer_roster_intent


@dataclass
class Tier1TriggerResult:
    """Result of tier1 trigger evaluation."""
    hard_triggered: bool = False
    soft_triggered: bool = False
    reasons: list = field(default_factory=list)

    def should_run_tier1(self) -> tuple[bool, bool]:
        """
        Returns (run_tier1, use_micro).
        - Hard triggered: run full Tier 1 (2 steps, 4 tool calls)
        - Soft only: run micro Tier 1 (1 step, 2 tool calls)
        - Neither: don't run
        """
        if self.hard_triggered:
            return True, False
        if self.soft_triggered:
            return True, True
        return False, False


def _count_grounded_claims(result) -> int:
    """Count claims with status=grounded and valid citation_chunk_ids, plus grounded roster entries."""
    claims = getattr(result, "claims", []) or []
    grounded = sum(
        1 for c in claims
        if getattr(c, "status", "") == "grounded"
        and getattr(c, "citation_chunk_ids", [])
    )
    roster = getattr(result, "grounded_roster", []) or []
    grounded_roster = sum(
        1 for gr in roster
        if getattr(gr, "status", "") == "grounded"
        and (getattr(gr, "valid_chunk_ids", None) or getattr(getattr(gr, "entry", None), "support_chunk_ids", []))
    )
    return grounded + grounded_roster


def _is_finalization_invalid(result) -> bool:
    """Check if synthesis/finalization is missing required fields."""
    syn = getattr(result, "synthesis", None)
    if not syn:
        return True
    suf = getattr(syn, "sufficiency", None)
    if not suf:
        return True
    if not getattr(suf, "argument", ""):
        return True
    return False


def _is_sufficiency_false(result) -> bool:
    """Check if sufficiency.sufficient is False."""
    syn = getattr(result, "synthesis", None)
    if not syn:
        return True
    suf = getattr(syn, "sufficiency", None)
    if not suf:
        return True
    return not getattr(suf, "sufficient", False)


def _is_coverage_narrow(result) -> bool:
    """Evidence dominated by one doc/collection."""
    workspace = getattr(result, "workspace", None)
    if not workspace:
        return False
    chunks = getattr(workspace, "fulltext_chunks", None) or getattr(workspace, "chunks", [])
    if not chunks or len(chunks) < 3:
        return False
    coll_counts: dict = {}
    doc_counts: dict = {}
    for c in chunks:
        coll = getattr(c, "collection_slug", None) or getattr(c, "source_label", "") or ""
        doc = getattr(c, "doc_id", None) or 0
        if coll:
            coll_counts[coll] = coll_counts.get(coll, 0) + 1
        if doc:
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
    total = len(chunks)
    if coll_counts:
        max_coll = max(coll_counts.values())
        if max_coll / total > 0.75:
            return True
    if doc_counts:
        max_doc = max(doc_counts.values())
        if max_doc / total > 0.6:
            return True
    return False


def compute_tier1_triggers(
    result,
    question: str,
    *,
    tool_calls_used: int = 0,
    max_tool_calls: int = 5,
) -> Tier1TriggerResult:
    """
    Evaluate tier1 triggers after a Tier 0 (V9/V11) run.

    Hard triggers (always auto-deepen):
    - grounded_claims_count == 0
    - finalization invalid / missing required fields
    - sufficiency.sufficient == False

    Soft triggers (auto-deepen if cheap):
    - Roster intent detected
    - Coverage narrow (one doc/collection dominates)
    - High overlap + low frontier (we don't have this from Tier 0 run; skip for now)
    """
    out = Tier1TriggerResult()

    grounded = _count_grounded_claims(result)
    if grounded == 0:
        out.hard_triggered = True
        out.reasons.append("0 grounded claims")

    if _is_finalization_invalid(result):
        out.hard_triggered = True
        out.reasons.append("finalization invalid or missing required fields")

    if _is_sufficiency_false(result):
        out.hard_triggered = True
        out.reasons.append("sufficiency.sufficient == False")

    if _infer_roster_intent(question or "", None):
        out.soft_triggered = True
        out.reasons.append("roster intent detected")

    if _is_coverage_narrow(result):
        out.soft_triggered = True
        out.reasons.append("coverage narrow (one doc/collection dominates)")

    return out


# Tier 1 config: small budget, patch-the-answer mode
TIER1_MAX_STEPS = 2
TIER1_MAX_TOOL_CALLS = 4
TIER1_MICRO_MAX_STEPS = 1
TIER1_MICRO_MAX_TOOL_CALLS = 2  # Soft triggers only: very cheap


def think_deeper_tier1(
    conn,
    seed_question: str,
    workspace,
    *,
    micro: bool = False,
    verbose: bool = True,
    v9_run_id: Optional[int] = None,
):
    """
    Run Tier 1 (small) ThinkDeeper: patch-the-answer mode.

    micro=True: 1 step, 2 tool calls (for soft triggers only)
    micro=False: 2 steps, 4 tool calls (for hard triggers)
    """
    from retrieval.agent.v9_deep_runner import think_deeper
    max_steps = TIER1_MICRO_MAX_STEPS if micro else TIER1_MAX_STEPS
    max_tool_calls = TIER1_MICRO_MAX_TOOL_CALLS if micro else TIER1_MAX_TOOL_CALLS
    return think_deeper(
        conn=conn,
        seed_question=seed_question,
        workspace=workspace,
        user_followup=None,
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        v9_run_id=v9_run_id,
    )
