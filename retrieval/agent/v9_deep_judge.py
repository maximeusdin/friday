"""
V9 Think Deeper — LLM Judge (Policy Brain).

Two responsibilities per step:
  1. Select best action from Actor's 2-3 proposals (pre-execution).
  2. Score the delta after execution (post-execution).

Information barrier: Judge sees ProposalForJudge (no Actor rationale),
FindingStore summary, coverage stats, and a small evidence sample.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set

from retrieval.agent.v9_deep_types import (
    CandidateChunk,
    DeepState,
    Finding,
    JudgeVerdict,
    ProposalForJudge,
    ResearchDirective,
    _parse_gaps,
)
from retrieval.agent.v9_deep_prompts import (
    JUDGE_SELECT_SYSTEM_PROMPT,
    JUDGE_SCORE_SYSTEM_PROMPT,
    build_judge_select_user_prompt,
    build_judge_score_user_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_MODEL = os.getenv("V9_DEEP_JUDGE_MODEL", "gpt-4.1-mini-2025-04-14")

MAX_EVIDENCE_SAMPLE = 8


# ── Evidence sample builder (deterministic) ──────────────────────────────────

def build_evidence_sample(
    new_chunks: List[CandidateChunk],
    state: DeepState,
) -> List[str]:
    """Deterministic evidence sample builder for Judge consumption.

    Selection policy:
      1. Top-6 by retrieval score (after rails)
      2. Plus 1-2 that maximize entity/date novelty vs baseline
      3. Always sorted by (doc_id, page) for deterministic ordering
      4. Hard cap at MAX_EVIDENCE_SAMPLE
      5. Each entry is: '[collection p.X] chunk_text[:300]'
    """
    if not new_chunks:
        return []

    # Top by score
    by_score = sorted(new_chunks, key=lambda c: c.score, reverse=True)
    top_6 = by_score[:6]
    top_6_ids = {c.chunk_id for c in top_6}

    # Novelty candidates: entities not in baseline
    baseline_eids = state.baseline_entity_ids
    novelty_candidates = []
    for c in by_score[6:]:
        if c.chunk_id in top_6_ids:
            continue
        new_eids = set(c.entity_ids) - baseline_eids
        new_dates = bool(c.date_spans)
        if new_eids or new_dates:
            novelty_candidates.append((len(new_eids) + int(new_dates), c))
    novelty_candidates.sort(key=lambda t: t[0], reverse=True)
    novelty_picks = [t[1] for t in novelty_candidates[:2]]

    # Combine and sort deterministically
    combined = list(top_6) + novelty_picks
    combined.sort(key=lambda c: (c.doc_id, c.page or ""))
    combined = combined[:MAX_EVIDENCE_SAMPLE]

    # Format: include chunk_id so Judge can cite valid IDs
    result = []
    for c in combined:
        label = c.collection_slug or "unknown"
        page = c.page or ""
        text = c.text[:300]
        result.append(f"[chunk_id={c.chunk_id} doc={label} {page}] {text}")
    return result


# ── Coverage stats builder ───────────────────────────────────────────────────

def build_coverage_stats(state: DeepState) -> Dict[str, Any]:
    """Build coverage statistics for Judge consumption."""
    doc_ids = state.selected_doc_ids
    collection_slugs = {c.collection_slug for c in state.selected_chunks if c.collection_slug}
    entity_ids = set()
    for c in state.selected_chunks:
        entity_ids.update(c.entity_ids)

    return {
        "total_chunks": len(state.selected_chunks),
        "total_docs": len(doc_ids),
        "collections": sorted(collection_slugs),
        "total_entities": len(entity_ids),
        "baseline_chunks": len(state.baseline_chunk_ids),
        "new_chunks": len(state.selected_chunk_ids - state.baseline_chunk_ids),
        "total_findings": state.finding_store.total_count() if state.finding_store else 0,
        "step": state.step,
        "tool_calls_used": state.tool_calls_used,
    }


def _format_coverage_stats(stats: Dict[str, Any]) -> str:
    """Format coverage stats as text for prompt."""
    lines = [
        f"Total chunks: {stats['total_chunks']} ({stats['new_chunks']} new vs baseline)",
        f"Documents: {stats['total_docs']}",
        f"Collections: {', '.join(stats['collections']) if stats['collections'] else 'none'}",
        f"Entities: {stats['total_entities']}",
        f"Findings: {stats['total_findings']}",
        f"Step: {stats['step']}, Tool calls used: {stats['tool_calls_used']}",
    ]
    return "\n".join(lines)


# ── Judge: select action ─────────────────────────────────────────────────────

def judge_select_action(
    seed_question: str,
    directive: ResearchDirective,
    prev_verdict: Optional[JudgeVerdict],
    findings_summary: str,
    coverage_stats: Dict[str, Any],
    actor_proposals: List[ProposalForJudge],
    step_number: int,
    *,
    pressure_summary: str = "",
    gap_types_summary: str = "",
    must_target_unseen: bool = False,
    unseen_satisfying_indices: Optional[List[int]] = None,
    recent_failures: Optional[List[str]] = None,
    model: str = "",
    verbose: bool = True,
) -> int:
    """Pre-execution: Judge picks best action from Actor proposals.

    Returns index of selected action (0-based).
    Records full prompt+inputs for debugging.
    """
    from openai import OpenAI

    model = model or _DEFAULT_JUDGE_MODEL
    client = OpenAI()

    proposals_json = json.dumps(
        [p.to_dict() for p in actor_proposals], indent=2
    )
    user_prompt = build_judge_select_user_prompt(
        seed_question=seed_question,
        findings_summary=findings_summary,
        coverage_stats=_format_coverage_stats(coverage_stats),
        proposals_json=proposals_json,
        step_number=step_number,
        pressure_summary=pressure_summary,
        gap_types_summary=gap_types_summary,
        must_target_unseen=must_target_unseen,
        unseen_satisfying_indices=unseen_satisfying_indices,
        recent_failures=recent_failures,
    )

    if verbose:
        print(f"  [ThinkDeeper] Judge selecting from {len(actor_proposals)} proposals...",
              file=sys.stderr)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SELECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        idx = int(parsed.get("selected_index", 0))
        # Clamp to valid range
        idx = max(0, min(idx, len(actor_proposals) - 1))

        if verbose:
            reasoning = parsed.get("reasoning", "")[:120]
            print(f"  [ThinkDeeper] Judge selected proposal {idx}: {reasoning}",
                  file=sys.stderr)
        return idx

    except Exception as e:
        logger.warning("Judge select_action failed: %s; defaulting to proposal 0", e)
        return 0


# ── Judge: score delta ───────────────────────────────────────────────────────

def judge_score_delta(
    seed_question: str,
    directive: ResearchDirective,
    prev_verdict: Optional[JudgeVerdict],
    findings_summary: str,
    coverage_stats: Dict[str, Any],
    new_evidence_sample: List[str],
    step_number: int,
    *,
    model: str = "",
    verbose: bool = True,
) -> JudgeVerdict:
    """Post-execution: Judge scores delta and recommends stop/continue."""
    from openai import OpenAI

    model = model or _DEFAULT_JUDGE_MODEL
    client = OpenAI()

    # Build prev scores summary
    prev_summary = ""
    if prev_verdict:
        prev_summary = (
            f"answeredness={prev_verdict.answeredness:.2f}, "
            f"material_novelty={prev_verdict.material_novelty:.2f}, "
            f"confidence={prev_verdict.confidence:.2f}"
        )

    directive_summary = (
        f"Primary: {directive.primary_question}\n"
        f"Follow-up: {directive.user_directive or '(none)'}\n"
        f"Weights: coverage={directive.weights.coverage}, "
        f"novelty={directive.weights.novelty}, "
        f"support={directive.weights.support}"
    )

    evidence_text = "\n".join(new_evidence_sample) if new_evidence_sample else "(no new evidence)"

    user_prompt = build_judge_score_user_prompt(
        seed_question=seed_question,
        directive_summary=directive_summary,
        findings_summary=findings_summary,
        coverage_stats=_format_coverage_stats(coverage_stats),
        evidence_sample=evidence_text,
        step_number=step_number,
        prev_scores_summary=prev_summary,
    )

    if verbose:
        print(f"  [ThinkDeeper] Judge scoring delta (step {step_number})...",
              file=sys.stderr)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SCORE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        verdict = _parse_verdict(parsed)

        if verbose:
            print(
                f"  [ThinkDeeper] Judge scores: ans={verdict.answeredness:.2f} "
                f"nov={verdict.material_novelty:.2f} conf={verdict.confidence:.2f} "
                f"stop={verdict.stop_recommendation} div={verdict.self_consistency_divergence:.2f}",
                file=sys.stderr,
            )
        return verdict

    except Exception as e:
        logger.warning("Judge score_delta failed: %s; returning default verdict", e)
        return JudgeVerdict(
            answeredness=0.0,
            material_novelty=0.0,
            confidence=0.0,
            stop_recommendation=False,
            ev_next_step_retrieve=0.5,
            ev_next_step_expand=0.5,
        )


def _parse_verdict(d: Dict[str, Any]) -> JudgeVerdict:
    """Parse Judge LLM output into JudgeVerdict."""
    # Reconciled scores
    reconciled = d.get("reconciled", {})
    rating_a = d.get("rating_a", {})
    rating_b = d.get("rating_b", {})

    answeredness = float(reconciled.get("answeredness", d.get("answeredness", 0.0)))
    material_novelty = float(reconciled.get("material_novelty", d.get("material_novelty", 0.0)))
    confidence = float(reconciled.get("confidence", d.get("confidence", 0.0)))
    exploration_quality = float(reconciled.get("exploration_quality", d.get("exploration_quality", 0.0)))

    # Self-consistency divergence
    div = float(d.get("self_consistency_divergence", 0.0))
    if not div and rating_a and rating_b:
        # Compute it
        diffs = []
        for key in ("answeredness", "material_novelty", "confidence"):
            a_val = float(rating_a.get(key, 0.0))
            b_val = float(rating_b.get(key, 0.0))
            diffs.append(abs(a_val - b_val))
        div = max(diffs) if diffs else 0.0

    # New findings
    findings = []
    for f_dict in d.get("new_findings", []):
        findings.append(Finding(
            text=f_dict.get("text", ""),
            cited_chunk_ids=[int(c) for c in f_dict.get("cited_chunk_ids", [])],
            finding_type=f_dict.get("finding_type", "context"),
        ))

    raw_gaps = d.get("top_gaps", [])[:3]
    gaps = _parse_gaps(raw_gaps)

    return JudgeVerdict(
        answeredness=answeredness,
        material_novelty=material_novelty,
        confidence=confidence,
        exploration_quality=exploration_quality,
        top_gaps=gaps,
        top_gap_target_phrase=d.get("top_gap_target_phrase"),
        new_findings=findings[:3],
        stop_recommendation=bool(d.get("stop_recommendation", False)),
        stop_reason=d.get("stop_reason"),
        ev_next_step_retrieve=float(d.get("ev_next_step_retrieve", 0.5)),
        ev_next_step_expand=float(d.get("ev_next_step_expand", 0.5)),
        doc_overflow_request=d.get("doc_overflow_request"),
        rating_a=rating_a,
        rating_b=rating_b,
        reconciled=reconciled,
        self_consistency_divergence=div,
    )


# ── Verdict validation (deterministic, post-LLM) ────────────────────────────

def pivot_top_gap_to_lead(verdict_history: list, lead_pool) -> Optional[str]:
    """If exploration_quality < 0.3 for last 2 steps and lead_pool has leads, return lead-based phrase."""
    if not lead_pool or not getattr(lead_pool, "leads", []):
        return None
    if len(verdict_history) < 2:
        return None
    last_two = verdict_history[-2:]
    if not all(getattr(v, "exploration_quality", 0.5) < 0.3 for v in last_two):
        return None
    top = lead_pool.leads[0]
    lead_type = getattr(top, "type", "")
    value = getattr(top, "value", "")
    if not value:
        return None
    return f"pursue lead: {value} ({lead_type})"


def validate_verdict(
    verdict: JudgeVerdict,
    valid_chunk_ids: Set[int],
) -> JudgeVerdict:
    """Clamp invalid chunk_id citations, adjust novelty score.

    If a finding cites chunk_ids not in valid_chunk_ids, the finding is dropped
    entirely.  If findings are dropped, material_novelty is reduced
    proportionally.
    """
    if not verdict.new_findings:
        return verdict

    original_count = len(verdict.new_findings)
    validated = []
    for f in verdict.new_findings:
        cited = [int(c) for c in f.cited_chunk_ids]
        if not cited:
            continue
        if all(c in valid_chunk_ids for c in cited):
            validated.append(f)
        else:
            logger.debug("Dropping finding with invalid chunk_ids: %s", f.cited_chunk_ids)

    # Adjust novelty proportionally
    if original_count > 0 and len(validated) < original_count:
        ratio = len(validated) / original_count
        adjusted_novelty = verdict.material_novelty * ratio
    else:
        adjusted_novelty = verdict.material_novelty

    # Return updated verdict (JudgeVerdict is mutable dataclass)
    verdict.new_findings = validated
    verdict.material_novelty = adjusted_novelty
    return verdict
