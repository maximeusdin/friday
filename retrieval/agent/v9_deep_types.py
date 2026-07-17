"""
V9 Think Deeper — Type definitions.

Actor/Judge architecture:
  - Actor proposes 2-3 candidate actions
  - Judge selects best + scores delta
  - Controller executes, applies rails, updates state
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def compute_finding_id(text: str, chunk_ids: List[int]) -> str:
    """Deterministic finding identity (same algo as compute_bullet_id)."""
    normalized = _normalize_text(text)
    sorted_ids = ",".join(str(c) for c in sorted(chunk_ids))
    raw = f"{normalized}:{sorted_ids}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ── ResearchDirective ────────────────────────────────────────────────────────

@dataclass
class MustInclude:
    entity_ids: List[int] = field(default_factory=list)
    date_ranges: List[Dict[str, str]] = field(default_factory=list)  # [{"from":"…","to":"…"}]
    collections: List[str] = field(default_factory=list)


@dataclass
class AvoidSpec:
    doc_ids: List[int] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)


@dataclass
class DirectiveWeights:
    coverage: float = 1.0
    novelty: float = 1.0
    support: float = 1.0
    verification: float = 1.0


@dataclass
class ResearchDirective:
    """Immutable description of what this Think Deeper run should achieve."""
    primary_question: str
    user_directive: Optional[str] = None
    must_answer: List[str] = field(default_factory=list)
    must_include: MustInclude = field(default_factory=MustInclude)
    avoid: AvoidSpec = field(default_factory=AvoidSpec)
    weights: DirectiveWeights = field(default_factory=DirectiveWeights)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_question": self.primary_question,
            "user_directive": self.user_directive,
            "must_answer": self.must_answer,
            "must_include": {
                "entity_ids": self.must_include.entity_ids,
                "date_ranges": self.must_include.date_ranges,
                "collections": self.must_include.collections,
            },
            "avoid": {
                "doc_ids": self.avoid.doc_ids,
                "claims": self.avoid.claims,
            },
            "weights": {
                "coverage": self.weights.coverage,
                "novelty": self.weights.novelty,
                "support": self.weights.support,
                "verification": self.weights.verification,
            },
        }


# ── Typed Gap (Judge output) ──────────────────────────────────────────────────

GAP_TYPE_COVERAGE = "coverage"      # need independent corroboration, new sources
GAP_TYPE_PRECISION = "precision"     # need mechanism/date/relationship detail
GAP_TYPE_ENTITY = "entity"          # need to identify who/what "X" refers to
GAP_TYPE_CONTRADICTION = "contradiction"  # need to resolve source disagreement

VALID_GAP_TYPES = {GAP_TYPE_COVERAGE, GAP_TYPE_PRECISION, GAP_TYPE_ENTITY, GAP_TYPE_CONTRADICTION}


@dataclass
class Gap:
    """Typed gap for Judge to reason about exploit vs explore."""
    type: str = "precision"   # coverage | precision | entity | contradiction
    target: str = ""          # retrievable target phrase
    priority: float = 0.5      # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "target": self.target, "priority": self.priority}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Gap":
        t = d.get("type", "precision")
        if t not in VALID_GAP_TYPES:
            t = "precision"
        return cls(
            type=t,
            target=d.get("target", "") or d.get("text", ""),
            priority=float(d.get("priority", 0.5)),
        )

    def to_target_string(self) -> str:
        """Legacy: string form for backward compat."""
        return self.target or f"[{self.type}]"


# ── NextAction (Actor output) ───────────────────────────────────────────────

# Proposal intent for explore/exploit frontier
PROPOSAL_INTENT_EXPLOIT = "exploit"    # tighten, triangulate, local expansion
PROPOSAL_INTENT_EXPLORE = "explore"    # new docs, collections, entities
PROPOSAL_INTENT_BALANCED = "balanced"   # mixed or unclear

# Action space — no SELECT (selection is controller policy).
ACTION_RETRIEVE = "RETRIEVE"
ACTION_EXPAND_SEEDS = "EXPAND_SEEDS"
ACTION_SYNTHESIZE = "SYNTHESIZE"
ACTION_VERIFY = "VERIFY"
ACTION_STOP = "STOP"

VALID_ACTIONS = {ACTION_RETRIEVE, ACTION_EXPAND_SEEDS, ACTION_SYNTHESIZE,
                 ACTION_VERIFY, ACTION_STOP}

# RETRIEVE modes (Actor must specify explicitly, never implicit)
RETRIEVE_MODE_HYBRID = "hybrid"
RETRIEVE_MODE_LEXICAL = "lexical"
RETRIEVE_MODE_MENTIONS = "mentions"

VALID_RETRIEVE_MODES = {RETRIEVE_MODE_HYBRID, RETRIEVE_MODE_LEXICAL,
                        RETRIEVE_MODE_MENTIONS}
RETRIEVE_MODE_EVIDENCE_LEADS = "evidence_leads"
RETRIEVE_MODE_ADJACENT = "adjacent"

# Query origin (for evidence-led exploration)
QUERY_ORIGIN_LEAD_CHASE = "LEAD_CHASE"
QUERY_ORIGIN_SEED_PARAPHRASE = "SEED_PARAPHRASE"
QUERY_ORIGIN_GAP_TARGET = "GAP_TARGET"
QUERY_ORIGIN_COUNTEREVIDENCE = "COUNTEREVIDENCE"

VALID_QUERY_ORIGINS = {
    QUERY_ORIGIN_LEAD_CHASE,
    QUERY_ORIGIN_SEED_PARAPHRASE,
    QUERY_ORIGIN_GAP_TARGET,
    QUERY_ORIGIN_COUNTEREVIDENCE,
}

# Tool-call cost per action type (deterministic accounting).
TOOL_CALL_UNITS: Dict[str, int] = {
    ACTION_RETRIEVE: 2,       # search + fetch
    ACTION_EXPAND_SEEDS: 1,   # expand_entities
    ACTION_SYNTHESIZE: 0,     # LLM only
    ACTION_VERIFY: 0,         # LLM only
    ACTION_STOP: 0,
}


@dataclass
class NextAction:
    """A single Actor proposal.  `why` and `expected_improvements` are stripped
    before the Judge sees proposals (information barrier)."""
    action: str                           # one of VALID_ACTIONS
    params: Dict[str, Any] = field(default_factory=dict)
    why: str = ""                         # Actor rationale (NOT shown to Judge)
    expected_improvements: List[str] = field(default_factory=list)  # NOT shown
    proposal_intent: str = PROPOSAL_INTENT_BALANCED  # exploit | explore | balanced
    query_origin: str = QUERY_ORIGIN_SEED_PARAPHRASE
    leads_used: List[str] = field(default_factory=list)  # stable lead_ids e.g. ["3f9a12ab"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "params": self.params,
            "why": self.why,
            "expected_improvements": self.expected_improvements,
            "proposal_intent": self.proposal_intent,
            "query_origin": self.query_origin,
            "leads_used": self.leads_used,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NextAction":
        action = d.get("action", ACTION_STOP)
        if action not in VALID_ACTIONS:
            action = ACTION_STOP
        intent = d.get("proposal_intent", PROPOSAL_INTENT_BALANCED)
        if intent not in (PROPOSAL_INTENT_EXPLOIT, PROPOSAL_INTENT_EXPLORE, PROPOSAL_INTENT_BALANCED):
            intent = PROPOSAL_INTENT_BALANCED
        qo = d.get("query_origin", QUERY_ORIGIN_SEED_PARAPHRASE)
        if qo not in VALID_QUERY_ORIGINS:
            qo = QUERY_ORIGIN_SEED_PARAPHRASE
        leads = d.get("leads_used", [])
        if isinstance(leads, list):
            leads = [str(x) for x in leads if x]
        return cls(
            action=action,
            params=d.get("params", {}),
            why=d.get("why", ""),
            expected_improvements=d.get("expected_improvements", []),
            proposal_intent=intent,
            query_origin=qo,
            leads_used=leads,
        )


# ── ProposalForJudge (stripped, frozen) ──────────────────────────────────────

@dataclass(frozen=True)
class ProposalForJudge:
    """What the Judge sees for each Actor proposal.
    This is the ONLY type passed to judge functions — enforced by type sigs."""
    action: str
    params: Dict[str, Any]
    tool_cost_estimate: int
    expected_new_docs_targeted: int
    budget_remaining_after: int
    proposal_intent: str = PROPOSAL_INTENT_BALANCED  # exploit | explore | balanced
    satisfies_unseen_constraint: bool = False  # targets new collections/docs not yet in evidence
    query_origin: str = QUERY_ORIGIN_SEED_PARAPHRASE
    leads_used: Tuple[str, ...] = ()  # stable lead_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "params": dict(self.params),
            "tool_cost_estimate": self.tool_cost_estimate,
            "expected_new_docs_targeted": self.expected_new_docs_targeted,
            "budget_remaining_after": self.budget_remaining_after,
            "proposal_intent": self.proposal_intent,
            "satisfies_unseen_constraint": self.satisfies_unseen_constraint,
            "query_origin": self.query_origin,
            "leads_used": list(self.leads_used),
        }


def strip_for_judge(
    proposals: List[NextAction],
    tool_calls_used: int,
    max_tool_calls: int,
    selected_collections: Optional[Set[str]] = None,
) -> List[ProposalForJudge]:
    """Hard-strip Actor rationale.  Only way to create ProposalForJudge objects."""
    selected_collections = selected_collections or set()
    result: List[ProposalForJudge] = []
    for p in proposals:
        cost = TOOL_CALL_UNITS.get(p.action, 0)
        # Micro-retrieve: lexical + top_k<=10 is genuinely cheaper (1 unit)
        if p.action == ACTION_RETRIEVE and cost == 2:
            mode = (p.params or {}).get("mode", "")
            top_k = (p.params or {}).get("top_k", 10)
            if mode == "lexical" and top_k <= 10:
                cost = 1
        # Estimate new docs targeted from scope params
        scope = p.params.get("scope", {})
        target_docs = len(scope.get("doc_ids", []))
        target_colls = scope.get("collections") or []
        target_coll_set = {c.lower() for c in target_colls} if isinstance(target_colls, list) else set()
        estimated_new = max(target_docs, len(target_colls), 1) if p.action == ACTION_RETRIEVE else 0
        intent = p.proposal_intent if p.proposal_intent else PROPOSAL_INTENT_BALANCED
        # satisfies_unseen_constraint: RETRIEVE targets collection not yet covered
        satisfies_unseen = False
        if p.action == ACTION_RETRIEVE and target_coll_set:
            sel_lower = {c.lower() for c in selected_collections}
            satisfies_unseen = bool(target_coll_set - sel_lower)
        leads_tuple = tuple(p.leads_used) if p.leads_used else ()
        result.append(ProposalForJudge(
            action=p.action,
            params=dict(p.params),  # shallow copy for safety
            tool_cost_estimate=cost,
            expected_new_docs_targeted=estimated_new,
            budget_remaining_after=max_tool_calls - tool_calls_used - cost,
            proposal_intent=intent,
            satisfies_unseen_constraint=satisfies_unseen,
            query_origin=p.query_origin,
            leads_used=leads_tuple,
        ))
    return result


# ── Lead / LeadPool (evidence-led exploration) ───────────────────────────────

LEAD_TYPE_ENTITY = "entity"
LEAD_TYPE_CODENAME = "codename"
LEAD_TYPE_ORG = "org"
LEAD_TYPE_DOC = "doc"
LEAD_TYPE_DATE = "date"

VALID_LEAD_TYPES = {LEAD_TYPE_ENTITY, LEAD_TYPE_CODENAME, LEAD_TYPE_ORG, LEAD_TYPE_DOC, LEAD_TYPE_DATE}


def _lead_id_hash(lead_type: str, value: str, entity_id: Optional[int], doc_id: Optional[int]) -> str:
    """Deterministic stable lead ID for cross-step reference."""
    norm_val = (value or "").strip().lower()
    ref = str(entity_id if entity_id is not None else doc_id if doc_id is not None else "")
    raw = f"{lead_type}:{norm_val}:{ref}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


@dataclass
class Lead:
    """Evidence-derived pivot for exploration. Stable lead_id for Actor reference."""
    lead_id: str
    type: str  # entity | codename | org | doc | date
    value: str
    entity_id: Optional[int] = None
    doc_id: Optional[int] = None
    support_chunk_ids: List[int] = field(default_factory=list)
    first_seen_step: int = 0
    last_seen_step: int = 0

    def to_prompt_line(self) -> str:
        extra = ""
        if self.entity_id is not None:
            extra = f" (entity_id={self.entity_id})"
        elif self.doc_id is not None:
            extra = f" (doc_id={self.doc_id})"
        return f"{self.lead_id}: [{self.type}] {self.value}{extra}"


class LeadPool:
    """Ranked, deduped leads from admitted chunks. Merged across steps by lead_id."""

    def __init__(self, leads: Optional[List[Lead]] = None):
        self.leads: List[Lead] = list(leads) if leads else []

    def to_prompt_section(self, max_leads: int = 15) -> str:
        if not self.leads:
            return ""
        lines = ["## LeadPool (cite these in RETRIEVE)", ""]
        for lead in self.leads[:max_leads]:
            lines.append(lead.to_prompt_line())
        lines.append("")
        lines.append(
            "CONSTRAINT: At least one proposed RETRIEVE action must be lead-chasing: "
            "cite 1–2 lead_ids. For entity leads (entity_id shown), use mode=\"mentions\" with entity_ids=[<int>] "
            "— use the integer entity_id from the lead (e.g. entity_id=123 → entity_ids=[123]), NOT the lead_id."
        )
        return "\n".join(lines)


# ── CandidateChunk ───────────────────────────────────────────────────────────

@dataclass
class CandidateChunk:
    chunk_id: int
    doc_id: int
    collection_slug: str = ""
    page: Optional[str] = None
    score: float = 0.0
    embedding: Optional[List[float]] = None   # loaded lazily
    entity_ids: List[int] = field(default_factory=list)
    date_spans: List[str] = field(default_factory=list)
    text: str = ""
    is_new_vs_baseline: bool = False
    source_step: int = 0


# ── Finding / FindingEntry ───────────────────────────────────────────────────

@dataclass
class Finding:
    """Judge-produced finding (pre-validation)."""
    text: str
    cited_chunk_ids: List[int] = field(default_factory=list)
    finding_type: str = "context"  # relationship|mechanism|time_linkage|contradiction|context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "cited_chunk_ids": self.cited_chunk_ids,
            "finding_type": self.finding_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Finding":
        return cls(
            text=d.get("text", ""),
            cited_chunk_ids=d.get("cited_chunk_ids", []),
            finding_type=d.get("finding_type", "context"),
        )


@dataclass
class FindingEntry:
    """Validated, deduped finding stored in FindingStore."""
    finding_id: str
    text: str
    finding_type: str
    supporting_chunk_ids: List[int]
    source_doc_ids: List[int]
    source_step: int
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "text": self.text,
            "finding_type": self.finding_type,
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "source_doc_ids": self.source_doc_ids,
            "source_step": self.source_step,
            "created_at": self.created_at,
        }


# ── JudgeVerdict ─────────────────────────────────────────────────────────────

def _parse_gaps(raw: Any) -> List["Gap"]:
    """Parse top_gaps from Judge output (typed dicts or legacy strings)."""
    if not raw:
        return []
    out: List[Gap] = []
    for item in raw[:5]:
        if isinstance(item, dict) and ("type" in item or "target" in item):
            out.append(Gap.from_dict(item))
        elif isinstance(item, str) and item.strip():
            out.append(Gap(type="precision", target=item.strip(), priority=0.5))
    return out


@dataclass
class JudgeVerdict:
    # Action selection (pre-execution)
    selected_action_index: int = 0
    selection_reasoning: str = ""

    # Delta scoring (post-execution)
    answeredness: float = 0.0
    material_novelty: float = 0.0
    confidence: float = 0.0
    exploration_quality: float = 0.0
    top_gaps: List[Gap] = field(default_factory=list)  # typed: coverage|precision|entity|contradiction
    top_gap_target_phrase: Optional[str] = None
    new_findings: List[Finding] = field(default_factory=list)
    stop_recommendation: bool = False
    stop_reason: Optional[str] = None
    ev_next_step_retrieve: float = 0.5
    ev_next_step_expand: float = 0.5
    doc_overflow_request: Optional[List[int]] = None

    # Self-consistency
    rating_a: Dict[str, float] = field(default_factory=dict)
    rating_b: Dict[str, float] = field(default_factory=dict)
    reconciled: Dict[str, float] = field(default_factory=dict)
    self_consistency_divergence: float = 0.0

    @property
    def ev_next_step(self) -> float:
        """Best EV across action types — used for stop decisions."""
        return max(self.ev_next_step_retrieve, self.ev_next_step_expand)

    @property
    def top_gaps_as_strings(self) -> List[str]:
        """Legacy: string list for consumers that expect List[str]."""
        return [g.to_target_string() for g in self.top_gaps]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_action_index": self.selected_action_index,
            "selection_reasoning": self.selection_reasoning,
            "answeredness": self.answeredness,
            "material_novelty": self.material_novelty,
            "confidence": self.confidence,
            "top_gaps": [g.to_dict() for g in self.top_gaps],
            "top_gap_target_phrase": self.top_gap_target_phrase,
            "new_findings": [f.to_dict() for f in self.new_findings],
            "stop_recommendation": self.stop_recommendation,
            "stop_reason": self.stop_reason,
            "ev_next_step_retrieve": self.ev_next_step_retrieve,
            "ev_next_step_expand": self.ev_next_step_expand,
            "doc_overflow_request": self.doc_overflow_request,
            "rating_a": self.rating_a,
            "rating_b": self.rating_b,
            "reconciled": self.reconciled,
            "self_consistency_divergence": self.self_consistency_divergence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgeVerdict":
        raw_gaps = d.get("top_gaps", [])
        gaps = _parse_gaps(raw_gaps)
        top_phrase = d.get("top_gap_target_phrase")
        if not top_phrase and gaps:
            top_phrase = gaps[0].target
        return cls(
            selected_action_index=d.get("selected_action_index", 0),
            selection_reasoning=d.get("selection_reasoning", ""),
            answeredness=d.get("answeredness", 0.0),
            material_novelty=d.get("material_novelty", 0.0),
            confidence=d.get("confidence", 0.0),
            top_gaps=gaps,
            top_gap_target_phrase=top_phrase,
            new_findings=[Finding.from_dict(f) for f in d.get("new_findings", [])],
            stop_recommendation=d.get("stop_recommendation", False),
            stop_reason=d.get("stop_reason"),
            ev_next_step_retrieve=d.get("ev_next_step_retrieve", 0.5),
            ev_next_step_expand=d.get("ev_next_step_expand", 0.5),
            doc_overflow_request=d.get("doc_overflow_request"),
            rating_a=d.get("rating_a", {}),
            rating_b=d.get("rating_b", {}),
            reconciled=d.get("reconciled", {}),
            self_consistency_divergence=d.get("self_consistency_divergence", 0.0),
        )


# ── DeepState ────────────────────────────────────────────────────────────────
# Forward-reference FindingStore — actual class lives in v9_deep_findings.py.

@dataclass
class DeepState:
    """Mutable state for a Think Deeper run.
    Built from ResearchWorkspace + evidence_set at init."""
    seed_question: str
    seed_embedding: List[float]
    directive: ResearchDirective

    # Evidence tracking
    baseline_chunk_ids: Set[int] = field(default_factory=set)
    baseline_doc_ids: Set[int] = field(default_factory=set)
    baseline_entity_ids: Set[int] = field(default_factory=set)
    selected_chunks: List[CandidateChunk] = field(default_factory=list)
    candidate_chunks: List[CandidateChunk] = field(default_factory=list)

    # Deterministic state — FindingStore set after construction
    finding_store: Any = None  # FindingStore (avoids circular import)

    # Budget tracking
    step: int = 0
    tool_calls_used: int = 0
    new_doc_count_by_step: List[int] = field(default_factory=list)
    new_findings_count_by_step: List[int] = field(default_factory=list)
    filtered_admitted_by_step: List[int] = field(default_factory=list)
    zero_admissible_streak: int = 0
    # Action failure memory: action -> consecutive 0-admitted count
    action_failure_counts: Dict[str, int] = field(default_factory=dict)
    # Zero-hits recovery: consecutive RETRIEVE actions that returned 0 raw candidates
    consecutive_zero_hits: int = 0
    # Frontier metrics per step: {new_entity_count, new_doc_count, new_collection_count, overlap_with_prev_queries}
    frontier_metrics_by_step: List[Dict[str, Any]] = field(default_factory=list)

    # LeadPool (evidence-led exploration)
    lead_pool: Optional[Any] = None  # LeadPool

    # History
    action_history: List[NextAction] = field(default_factory=list)
    verdict_history: List[JudgeVerdict] = field(default_factory=list)

    # Convenience
    @property
    def selected_chunk_ids(self) -> Set[int]:
        return {c.chunk_id for c in self.selected_chunks}

    @property
    def selected_doc_ids(self) -> Set[int]:
        return {c.doc_id for c in self.selected_chunks}


# ── ThinkDeeperResult ────────────────────────────────────────────────────────

@dataclass
class NoveltyReport:
    new_docs: List[Dict[str, Any]] = field(default_factory=list)     # [{doc_id, label}]
    new_entities: List[Dict[str, Any]] = field(default_factory=list)  # [{entity_id, name}]
    what_changed: str = ""
    remaining_gaps: List[str] = field(default_factory=list)
    suggested_queries: List[str] = field(default_factory=list)  # LLM-generated queries to explore
    # V10 identity-layer novelty metrics
    new_mapping_confirmed: int = 0          # alias mapping promoted to confirmed
    ambiguity_reduced: int = 0              # hypothesis candidate set narrowed
    new_collection_covered_for_entity: int = 0  # entity searched in new collection
    new_evidence_support_for_entity: int = 0    # new supporting chunk for entity
    context_mapping_confirmed: int = 0      # confirmed mapping for (alias, doc, page-range)
    new_doc_resolved_for_alias: int = 0     # alias resolved in a new document

    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_docs": self.new_docs,
            "new_entities": self.new_entities,
            "what_changed": self.what_changed,
            "remaining_gaps": self.remaining_gaps,
            "suggested_queries": self.suggested_queries,
            "new_mapping_confirmed": self.new_mapping_confirmed,
            "ambiguity_reduced": self.ambiguity_reduced,
            "new_collection_covered_for_entity": self.new_collection_covered_for_entity,
            "new_evidence_support_for_entity": self.new_evidence_support_for_entity,
            "context_mapping_confirmed": self.context_mapping_confirmed,
            "new_doc_resolved_for_alias": self.new_doc_resolved_for_alias,
        }


@dataclass
class ThinkDeeperResult:
    """Final output of the Think Deeper controller."""
    narrative: str = ""
    claims: List[Any] = field(default_factory=list)  # List[GroundedClaim]
    verification: Any = None   # V9VerificationReport
    novelty_report: NoveltyReport = field(default_factory=NoveltyReport)
    stop_reason: str = ""
    verdict_history: List[JudgeVerdict] = field(default_factory=list)
    finding_store_entries: List[FindingEntry] = field(default_factory=list)
    steps_executed: int = 0
    tool_calls_used: int = 0
    elapsed_ms: float = 0.0
    # Carry forward for evidence persistence
    selected_chunks: List[CandidateChunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative,
            "novelty_report": self.novelty_report.to_dict(),
            "stop_reason": self.stop_reason,
            "steps_executed": self.steps_executed,
            "tool_calls_used": self.tool_calls_used,
            "elapsed_ms": self.elapsed_ms,
        }
