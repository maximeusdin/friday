"""
V9 Types - Investigation Loop (V9.4)

V9.4 changes (from V9.3):
- TaskType enum removed entirely
- InvestigationState / InvestigationStep / WorkspaceDelta added
- V9Synthesis: final flag, artifact dict, model-owned responsiveness
- SufficiencyCheck: argument + remaining_gaps (always required)
- ResearchWorkspace: investigation state, notes kept, hypotheses moved
"""
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


# Topic/operation cover-words that are never real people. They get mis-linked to common nouns in
# the concordance (e.g. an entity 'Balloon' with alias 'Atomic') and, if used to gloss output,
# produce garbage like "atomic (Balloon)". Never expand these as identities.
_NON_PERSON_GLOSS_TOKENS = {
    "balloon", "ballon", "atomic", "enormous", "enormoz", "uranium", "bomb", "plutonium",
    "tube", "corporation", "bank", "project", "operation",
}


# =============================================================================
# Bullet ID utilities (canonical, single source of truth)
# =============================================================================

def _normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def compute_bullet_id(text: str, chunk_ids: List[int]) -> str:
    """Deterministic bullet identity.  Returns '' if chunk_ids is empty."""
    if not chunk_ids:
        return ""
    normalized = _normalize_text(text)
    sorted_ids = ",".join(str(cid) for cid in sorted(chunk_ids))
    raw = f"{normalized}:{sorted_ids}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# =============================================================================
# Scope filter (deterministic, injected into tool calls)
# =============================================================================

@dataclass
class ScopeFilter:
    """Hard retrieval filter, applied outside the model.

    Precedence rule: when both document_ids and collections are present,
    document_ids wins and collections is ignored in SQL filtering.
    """
    collections: Optional[List[str]] = None
    document_ids: Optional[List[int]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    def is_empty(self) -> bool:
        return (not self.collections and not self.document_ids
                and not self.date_from and not self.date_to)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collections": self.collections,
            "document_ids": self.document_ids,
            "date_from": self.date_from,
            "date_to": self.date_to,
        }


# =============================================================================
# Entity candidate (from concordance -- uncommitted until agent accepts)
# =============================================================================

@dataclass
class AliasHypothesis:
    """A model-surfaced alias hypothesis, scoped to the current run/workspace.

    Never becomes global truth automatically. Status transitions:
      proposed -> validated  (when deterministic resolution confirms it)
      proposed -> rejected   (when evidence contradicts it)
    """
    alias_text: str
    entity_id: int
    supporting_chunk_ids: List[int] = field(default_factory=list)
    status: str = "proposed"      # "proposed" | "validated" | "rejected"
    created_turn_idx: int = 0
    validated_reason: str = ""    # e.g. "expand_entities", "chunk_majority", "concordance"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alias_text": self.alias_text,
            "entity_id": self.entity_id,
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "status": self.status,
            "created_turn_idx": self.created_turn_idx,
            "validated_reason": self.validated_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AliasHypothesis":
        return cls(
            alias_text=d.get("alias_text", ""),
            entity_id=d.get("entity_id", 0),
            supporting_chunk_ids=d.get("supporting_chunk_ids", []),
            status=d.get("status", "proposed"),
            created_turn_idx=d.get("created_turn_idx", 0),
            validated_reason=d.get("validated_reason", ""),
        )


@dataclass
class EntityCandidate:
    """A concordance-resolved identity candidate. NOT accepted evidence until
    the agent explicitly references it or calls expand_entities.

    confidence: how strong the resolution is.
      - "exact"    : exact canonical or alias match
      - "partial"  : partial name match
      - "concordance" : concordance expansion match
      - "inferred" : model-inferred (lower trust)
    ambiguous: True if multiple candidates share the same query_term.
    """
    query_term: str
    entity_id: int
    canonical_name: str
    entity_type: Optional[str] = None
    matched_via: str = ""
    accepted: bool = False
    confidence: str = "exact"       # "exact" | "partial" | "concordance" | "inferred"
    ambiguous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_term": self.query_term,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "matched_via": self.matched_via,
            "accepted": self.accepted,
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
        }


# =============================================================================
# Catalog hit (from search -- snippet only, not full text)
# =============================================================================

@dataclass
class CatalogHit:
    """A search result preview: enough to triage, not full text."""
    chunk_id: int
    score: float
    doc_id: Optional[int] = None
    page: Optional[str] = None
    collection: Optional[str] = None
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "doc_id": self.doc_id,
            "page": self.page,
            "collection": self.collection,
            "snippet": self.snippet[:400],
        }


# =============================================================================
# Workspace chunk (full text, from fetch)
# =============================================================================

@dataclass
class WorkspaceChunk:
    """A chunk with full text loaded via fetch_chunks."""
    chunk_id: int
    text: str
    doc_id: Optional[int] = None
    page: Optional[str] = None
    source_label: Optional[str] = None
    collection_slug: Optional[str] = None   # canonical collection slug (for scope filtering)
    score: Optional[float] = None
    is_neighbor: bool = False
    linked_entity_ids: List[int] = field(default_factory=list)  # system-derived at fetch time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text[:2000] + "..." if len(self.text) > 2000 else self.text,
            "doc_id": self.doc_id,
            "page": self.page,
            "source_label": self.source_label,
            "collection_slug": self.collection_slug,
            "score": self.score,
            "is_neighbor": self.is_neighbor,
            "linked_entity_ids": self.linked_entity_ids,
        }


@dataclass
class WorkspaceEntity:
    """An entity confirmed in the workspace (accepted by agent or via expand_entities)."""
    entity_id: int
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    entity_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "entity_type": self.entity_type,
        }


# =============================================================================
# Investigation state (V9.4: per-turn scratchpad + trace)
# =============================================================================

@dataclass
class InvestigationStep:
    """One step in the investigation trace."""
    step_idx: int
    action: str             # tool name or "synthesize"
    rationale: str          # from scratchpad_update
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs_summary: str = ""
    added_catalog: int = 0
    added_fulltext: int = 0
    added_entities: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "action": self.action,
            "rationale": self.rationale,
            "inputs": self.inputs,
            "outputs_summary": self.outputs_summary,
            "added_catalog": self.added_catalog,
            "added_fulltext": self.added_fulltext,
            "added_entities": self.added_entities,
        }


@dataclass
class InvestigationState:
    """Model-owned investigation state, updated per-turn via scratchpad."""
    goal: str = ""
    leads: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    ready_to_synthesize: bool = False
    trace: List[InvestigationStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "leads": self.leads,
            "hypotheses": self.hypotheses,
            "gaps": self.gaps,
            "next_actions": self.next_actions,
            "ready_to_synthesize": self.ready_to_synthesize,
            "trace_steps": len(self.trace),
        }


@dataclass
class WorkspaceDelta:
    """What changed since the last model turn."""
    new_catalog: int = 0
    new_fulltext: int = 0
    new_entities: int = 0
    new_candidates: int = 0
    tools_called: List[str] = field(default_factory=list)

    def format(self) -> str:
        parts = []
        if self.new_catalog:
            parts.append(f"+{self.new_catalog} catalog")
        if self.new_fulltext:
            parts.append(f"+{self.new_fulltext} fulltext")
        if self.new_entities:
            parts.append(f"+{self.new_entities} entities")
        if self.new_candidates:
            parts.append(f"+{self.new_candidates} candidates")
        if self.tools_called:
            parts.append(f"tools: {self.tools_called}")
        return "Delta: " + (", ".join(parts) if parts else "no changes")


# =============================================================================
# Progress signal
# =============================================================================

@dataclass
class ProgressSignal:
    """Lightweight computed signal shown to model each turn."""
    new_docs_added: int = 0
    new_person_names_found: int = 0
    duplicate_rate: float = 0.0
    search_queries_used: List[str] = field(default_factory=list)
    total_catalog_hits: int = 0
    total_fulltext_loaded: int = 0

    def format(self) -> str:
        queries_str = ", ".join(f'"{q}"' for q in self.search_queries_used[-6:])
        return (
            f"Progress: +{self.new_docs_added} new docs, "
            f"+{self.new_person_names_found} new person names, "
            f"{self.duplicate_rate:.0%} duplicate rate. "
            f"Catalog: {self.total_catalog_hits} hits. "
            f"Full text loaded: {self.total_fulltext_loaded}. "
            f"Queries used so far: [{queries_str}]"
        )


# =============================================================================
# Evidence memory (append-only bullet summaries with provenance)
# =============================================================================

@dataclass
class EvidenceBullet:
    """A compact evidence summary bullet with provenance pointers.

    linked_entity_ids: system-derived from supporting chunks at merge time.
    Tracks which workspace entities this bullet is about — enables entity-aware
    view selection and cross-bullet unification.
    """
    bullet_id: str                              # from compute_bullet_id()
    text: str                                   # max ~220 chars (enforced post-parse)
    supporting_chunk_ids: List[int]             # required >= 1
    created_at: str = ""                        # ISO timestamp, set at merge time
    tags: List[str] = field(default_factory=list)     # <= 3, freeform
    doc_ids: List[int] = field(default_factory=list)  # ALWAYS derived at merge, never trusted from summarizer
    linked_entity_ids: List[int] = field(default_factory=list)  # system-derived at merge
    pinned: bool = False
    pin_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bullet_id": self.bullet_id,
            "text": self.text,
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "created_at": self.created_at,
            "tags": self.tags,
            "doc_ids": self.doc_ids,
            "linked_entity_ids": self.linked_entity_ids,
            "pinned": self.pinned,
            "pin_reason": self.pin_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceBullet":
        return cls(
            bullet_id=d.get("bullet_id", ""),
            text=d.get("text", ""),
            supporting_chunk_ids=d.get("supporting_chunk_ids", []),
            created_at=d.get("created_at", ""),
            tags=d.get("tags", []),
            doc_ids=d.get("doc_ids", []),
            linked_entity_ids=d.get("linked_entity_ids", []),
            pinned=d.get("pinned", False),
            pin_reason=d.get("pin_reason", ""),
        )


@dataclass
class EvidenceSummaryUpdate:
    """One summariser pass: compact bullets + open questions + leads + warnings."""
    update_id: str                              # uuid4
    generated_from_chunk_ids: List[int]
    summarizer_model: str
    created_at: str                             # ISO timestamp
    bullets: List[EvidenceBullet] = field(default_factory=list)   # <= 6
    open_questions: List[str] = field(default_factory=list)       # <= 4
    leads: List[str] = field(default_factory=list)                # <= 6
    warnings: List[str] = field(default_factory=list)             # <= 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_id": self.update_id,
            "generated_from_chunk_ids": self.generated_from_chunk_ids,
            "summarizer_model": self.summarizer_model,
            "created_at": self.created_at,
            "bullets": [b.to_dict() for b in self.bullets],
            "open_questions": self.open_questions,
            "leads": self.leads,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceSummaryUpdate":
        return cls(
            update_id=d.get("update_id", ""),
            generated_from_chunk_ids=d.get("generated_from_chunk_ids", []),
            summarizer_model=d.get("summarizer_model", ""),
            created_at=d.get("created_at", ""),
            bullets=[EvidenceBullet.from_dict(b) for b in d.get("bullets", [])],
            open_questions=d.get("open_questions", []),
            leads=d.get("leads", []),
            warnings=d.get("warnings", []),
        )


@dataclass
class EvidenceMemoryView:
    """Lightweight container for the subset of evidence memory shown to the model."""
    pinned_bullets: List[EvidenceBullet] = field(default_factory=list)
    recent_bullets: List[EvidenceBullet] = field(default_factory=list)
    top_relevant_bullets: List[EvidenceBullet] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    leads: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Pin criticality helper (shared by workspace + rehydrate)
# =============================================================================

_CRITICAL_PIN_KEYWORDS = {"warning", "identity", "contradiction", "roster"}


def _is_critical_pin(bullet: EvidenceBullet) -> bool:
    return any(kw in (bullet.pin_reason or "") for kw in _CRITICAL_PIN_KEYWORDS)


# =============================================================================
# Research Workspace
# =============================================================================

@dataclass
class ResearchWorkspace:
    """Accumulated state the model can see and reason over."""
    question: str
    scope: ScopeFilter = field(default_factory=ScopeFilter)
    catalog_hits: List[CatalogHit] = field(default_factory=list)
    fulltext_chunks: List[WorkspaceChunk] = field(default_factory=list)
    entities: List[WorkspaceEntity] = field(default_factory=list)
    entity_candidates: List[EntityCandidate] = field(default_factory=list)
    investigation: InvestigationState = field(default_factory=InvestigationState)
    notes: List[str] = field(default_factory=list)          # append-only rolling digest
    uncertainty_flags: List[str] = field(default_factory=list)
    # Evidence memory layer
    evidence_memory: List[EvidenceSummaryUpdate] = field(default_factory=list)
    pinned_bullet_ids: List[str] = field(default_factory=list)
    _bullet_index: Dict[str, EvidenceBullet] = field(default_factory=dict)   # cache-only
    _summarized_chunk_ids: set = field(default_factory=set)                  # cache-only
    # Alias hypothesis layer  (keyed by (alias_text_lower, entity_id))
    alias_hypotheses: Dict[tuple, "AliasHypothesis"] = field(default_factory=dict)
    # Internal tracking
    _search_queries: List[str] = field(default_factory=list)
    _prev_doc_ids: set = field(default_factory=set)
    _prev_person_names: set = field(default_factory=set)
    # PEM lane (V9 + PEM integration)
    pem_seed_chunk_ids: List[int] = field(default_factory=list)
    _pem_cache: Optional[Dict] = field(default=None, repr=False)  # page_id -> pem_rows
    _pem_canonical_map: Optional[Dict[int, str]] = field(default=None, repr=False)
    # PEM operational alias map (A: PEM-truthy — set by runner, used for AliasMap + retrieval)
    _pem_operational_alias_map: Optional[Dict[str, Dict[str, Any]]] = field(default=None, repr=False)

    def catalog_chunk_ids(self) -> List[int]:
        return [h.chunk_id for h in self.catalog_hits]

    def fulltext_chunk_ids(self) -> List[int]:
        return [c.chunk_id for c in self.fulltext_chunks]

    def all_chunk_ids(self) -> List[int]:
        seen = set()
        out = []
        for cid in self.fulltext_chunk_ids() + self.catalog_chunk_ids():
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out

    @property
    def chunks(self) -> List[WorkspaceChunk]:
        return self.fulltext_chunks

    def chunk_ids(self) -> List[int]:
        return self.all_chunk_ids()

    def rehydrate_evidence_index(self) -> None:
        """Rebuild _bullet_index, _summarized_chunk_ids, pinned_bullet_ids from
        evidence_memory.  Pinned list is sorted deterministically: critical pins
        first, then newest-first by created_at within each group."""
        self._bullet_index.clear()
        self._summarized_chunk_ids.clear()
        pinned_bullets: List[EvidenceBullet] = []
        for update in self.evidence_memory:
            self._summarized_chunk_ids.update(update.generated_from_chunk_ids)
            for b in update.bullets:
                self._bullet_index[b.bullet_id] = b
                if b.pinned:
                    pinned_bullets.append(b)
        # Partition into critical / non-critical, each sorted newest-first
        critical = sorted(
            [b for b in pinned_bullets if _is_critical_pin(b)],
            key=lambda b: b.created_at or "", reverse=True,
        )
        non_critical = sorted(
            [b for b in pinned_bullets if not _is_critical_pin(b)],
            key=lambda b: b.created_at or "", reverse=True,
        )
        self.pinned_bullet_ids = [b.bullet_id for b in critical + non_critical]

    def accept_candidate(self, entity_id: int) -> None:
        """Mark a candidate as accepted and promote to entities[]."""
        for c in self.entity_candidates:
            if c.entity_id == entity_id and not c.accepted:
                c.accepted = True
                if not any(e.entity_id == entity_id for e in self.entities):
                    self.entities.append(WorkspaceEntity(
                        entity_id=c.entity_id,
                        canonical_name=c.canonical_name,
                        entity_type=c.entity_type,
                    ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "scope": self.scope.to_dict(),
            "catalog_hits": len(self.catalog_hits),
            "fulltext_chunks": len(self.fulltext_chunks),
            "entities": [e.to_dict() for e in self.entities],
            "entity_candidates": [c.to_dict() for c in self.entity_candidates],
            "investigation": self.investigation.to_dict(),
            "alias_hypotheses": [h.to_dict() for h in self.alias_hypotheses.values()],
        }


# =============================================================================
# Sufficiency check (model self-report)
# =============================================================================

@dataclass
class SufficiencyCheck:
    """Model's self-reported sufficiency assessment."""
    sufficient: bool = False
    argument: str = ""
    remaining_gaps: List[str] = field(default_factory=list)  # always required, even if empty
    next_best_actions_if_more_time: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "argument": self.argument,
            "remaining_gaps": self.remaining_gaps,
            "next_best_actions_if_more_time": self.next_best_actions_if_more_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SufficiencyCheck":
        return cls(
            sufficient=data.get("sufficient", False),
            argument=data.get("argument", data.get("justification", "")),
            remaining_gaps=data.get("remaining_gaps", []),
            next_best_actions_if_more_time=data.get("next_best_actions_if_more_time", []),
        )


# =============================================================================
# Structured output section types (used for parsing artifact entries)
# =============================================================================

@dataclass
class RosterEntry:
    """A person identified in a roster artifact."""
    name: str
    role: str = ""
    support_chunk_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "role": self.role, "support_chunk_ids": self.support_chunk_ids}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RosterEntry":
        return cls(
            name=d.get("name", ""),
            role=d.get("role", ""),
            support_chunk_ids=d.get("support_chunk_ids", d.get("evidence_chunk_ids", [])),
        )


@dataclass
class TimelineEntry:
    """A dated event in a timeline artifact."""
    date: str
    event: str
    support_chunk_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"date": self.date, "event": self.event, "support_chunk_ids": self.support_chunk_ids}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TimelineEntry":
        return cls(
            date=d.get("date", ""),
            event=d.get("event", ""),
            support_chunk_ids=d.get("support_chunk_ids", d.get("evidence_chunk_ids", [])),
        )


@dataclass
class EvidenceEntry:
    """A piece of evidence in an evidence artifact."""
    quote: str
    source: str = ""
    page: str = ""
    chunk_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"quote": self.quote, "source": self.source, "page": self.page, "chunk_id": self.chunk_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceEntry":
        return cls(quote=d.get("quote", ""), source=d.get("source", ""), page=d.get("page", ""), chunk_id=d.get("chunk_id"))


@dataclass
class RelationshipEdge:
    """A relationship edge in a relationships artifact."""
    entity_a: str
    relation: str
    entity_b: str
    support_chunk_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"entity_a": self.entity_a, "relation": self.relation, "entity_b": self.entity_b, "support_chunk_ids": self.support_chunk_ids}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationshipEdge":
        return cls(
            entity_a=d.get("entity_a", ""),
            relation=d.get("relation", ""),
            entity_b=d.get("entity_b", ""),
            support_chunk_ids=d.get("support_chunk_ids", d.get("evidence_chunk_ids", [])),
        )


# =============================================================================
# Identity resolution (parsed from artifact["identity"])
# =============================================================================

@dataclass
class IdentityResolution:
    """Resolved identity: maps an alias/codename to a canonical person."""
    alias: str
    canonical: str
    entity_id: Optional[int] = None
    basis: List[Dict[str, Any]] = field(default_factory=list)
    support_chunk_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alias": self.alias,
            "canonical": self.canonical,
            "entity_id": self.entity_id,
            "basis": self.basis,
            "support_chunk_ids": self.support_chunk_ids,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IdentityResolution":
        return cls(
            alias=d.get("alias", ""),
            canonical=d.get("canonical", ""),
            entity_id=d.get("entity_id"),
            basis=d.get("basis", []),
            support_chunk_ids=d.get("support_chunk_ids", d.get("supporting_chunk_ids", [])),
        )


# =============================================================================
# Synthesis (model output -- V9.4: final flag + artifact dict)
# =============================================================================


@dataclass
class EvidenceSpanRef:
    """Reference to a specific span within a chunk. Provenance-backed evidence."""
    chunk_id: int
    sentence_index: Optional[int] = None  # 0-based (preferred)
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    quote: Optional[str] = None  # 1-2 sentences max, ~200 chars

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceSpanRef":
        return cls(
            chunk_id=int(data.get("chunk_id", 0)),
            sentence_index=data.get("sentence_index") if data.get("sentence_index") is not None else None,
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            quote=data.get("quote"),
        )


@dataclass
class V9Claim:
    """A single claim from the model (before grounding).

    citation_chunk_ids: chunk IDs that directly support this claim.
    Required for factual claims (requires_citation=True).

    evidence: span refs (chunk_id + sentence_index). If present, authoritative
    over citation_chunk_ids. Never fill missing evidence by overlap.

    linked_entity_ids: system-derived post-hoc from citation chunks.
    NOT model-provided — derived during grounding.
    """
    text: str
    confidence: str  # "high" | "medium" | "low"
    requires_citation: bool = True
    citation_chunk_ids: List[int] = field(default_factory=list)
    evidence: List["EvidenceSpanRef"] = field(default_factory=list)
    linked_entity_ids: List[int] = field(default_factory=list)  # system-derived post-hoc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "requires_citation": self.requires_citation,
            "citation_chunk_ids": self.citation_chunk_ids,
            "evidence": [
                {"chunk_id": e.chunk_id, "sentence_index": e.sentence_index, "quote": e.quote}
                for e in self.evidence
            ],
            "linked_entity_ids": self.linked_entity_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "V9Claim":
        raw_ids = data.get("citation_chunk_ids") or []
        chunk_ids = []
        for cid in raw_ids:
            try:
                chunk_ids.append(int(cid))
            except (ValueError, TypeError):
                pass
        raw_evidence = data.get("evidence") or []
        evidence = [
            EvidenceSpanRef.from_dict(e) for e in raw_evidence
            if isinstance(e, dict) and e.get("chunk_id") is not None
        ]
        return cls(
            text=data.get("text", ""),
            confidence=data.get("confidence", "medium"),
            requires_citation=data.get("requires_citation", True),
            citation_chunk_ids=chunk_ids,
            evidence=evidence,
        )


@dataclass
class V9Synthesis:
    """Model output: final flag + narrative + claims + sufficiency + responsiveness + artifact."""
    final: bool = False
    narrative: str = ""
    claims: List[V9Claim] = field(default_factory=list)
    sufficiency: Optional[SufficiencyCheck] = None
    responsiveness: Optional[Dict[str, Any]] = None  # model self-report
    artifact: Dict[str, Any] = field(default_factory=dict)
    # artifact keys: identity, roster, timeline, evidence, relationships
    # Each entry should include support_chunk_ids where possible

    def get_identity(self) -> Optional[IdentityResolution]:
        """Parse artifact['identity'] into typed IdentityResolution if present."""
        ident = self.artifact.get("identity")
        if ident and isinstance(ident, dict):
            return IdentityResolution.from_dict(ident)
        return None

    def get_roster(self) -> List[RosterEntry]:
        """Parse artifact['roster'] into typed RosterEntry list if present."""
        roster = self.artifact.get("roster", [])
        if isinstance(roster, list):
            return [RosterEntry.from_dict(r) for r in roster if isinstance(r, dict)]
        return []

    def get_timeline(self) -> List[TimelineEntry]:
        """Parse artifact['timeline'] into typed TimelineEntry list if present."""
        timeline = self.artifact.get("timeline", [])
        if isinstance(timeline, list):
            return [TimelineEntry.from_dict(t) for t in timeline if isinstance(t, dict)]
        return []

    def get_evidence(self) -> List[EvidenceEntry]:
        """Parse artifact['evidence'] into typed EvidenceEntry list if present."""
        evidence = self.artifact.get("evidence", [])
        if isinstance(evidence, list):
            return [EvidenceEntry.from_dict(e) for e in evidence if isinstance(e, dict)]
        return []

    def get_relationships(self) -> List[RelationshipEdge]:
        """Parse artifact['relationships'] into typed RelationshipEdge list if present."""
        edges = self.artifact.get("relationships", [])
        if isinstance(edges, list):
            return [RelationshipEdge.from_dict(e) for e in edges if isinstance(e, dict)]
        return []

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "final": self.final,
            "narrative": self.narrative,
            "claims": [c.to_dict() for c in self.claims],
            "sufficiency": self.sufficiency.to_dict() if self.sufficiency else None,
            "responsiveness": self.responsiveness,
        }
        if self.artifact:
            d["artifact"] = self.artifact
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "V9Synthesis":
        raw_claims = data.get("claims") or []  # handles None from nullable schema
        claims = [V9Claim.from_dict(c) for c in raw_claims]
        suf_data = data.get("sufficiency")
        sufficiency = SufficiencyCheck.from_dict(suf_data) if suf_data else None
        return cls(
            final=data.get("final", False),
            narrative=data.get("narrative") or "",  # handles None from nullable schema
            claims=claims,
            sufficiency=sufficiency,
            responsiveness=data.get("responsiveness"),
            artifact=data.get("artifact") or {},  # handles None from nullable schema
        )


# =============================================================================
# Grounding (post-hoc)
# =============================================================================

@dataclass
class GroundedClaim:
    """A claim after citation binding."""
    claim: V9Claim
    status: str  # "grounded" | "weak" | "unsupported"
    citation_chunk_ids: List[int] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim.to_dict(), "status": self.status, "citation_chunk_ids": self.citation_chunk_ids, "note": self.note}


@dataclass
class GroundedRosterEntry:
    """A roster entry after citation validation against workspace chunks."""
    entry: RosterEntry
    status: str  # "grounded" | "weak" | "unsupported"
    valid_chunk_ids: List[int] = field(default_factory=list)


# =============================================================================
# Responsiveness (advisory -- can trigger extension, never blocks)
# =============================================================================

@dataclass
class ResponsivenessResult:
    """Advisory responsiveness check result."""
    responsive: bool = True
    issues: List[str] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"responsive": self.responsive, "issues": self.issues, "suggestion": self.suggestion}


# =============================================================================
# Verification (advisory)
# =============================================================================

@dataclass
class V9VerificationReport:
    """Advisory only -- never blocks or drops the answer."""
    grounded_claims: int = 0
    weak_claims: int = 0
    unsupported_claims: int = 0
    responsiveness: Optional[ResponsivenessResult] = None
    artifact_notes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounded_claims": self.grounded_claims,
            "weak_claims": self.weak_claims,
            "unsupported_claims": self.unsupported_claims,
            "responsiveness": self.responsiveness.to_dict() if self.responsiveness else None,
            "artifact_notes": self.artifact_notes,
            "notes": self.notes,
        }


# =============================================================================
# Result
# =============================================================================

@dataclass
class V9Result:
    """Complete V9 query result."""
    narrative: str
    claims: List[GroundedClaim] = field(default_factory=list)
    grounded_roster: List[GroundedRosterEntry] = field(default_factory=list)
    verification: Optional[V9VerificationReport] = None
    sufficiency: Optional[SufficiencyCheck] = None
    synthesis: Optional[V9Synthesis] = None
    workspace: Optional[ResearchWorkspace] = None
    investigation_trace: List[InvestigationStep] = field(default_factory=list)
    trace: Any = None

    # -----------------------------------------------------------------
    # Concordance helpers (used at render time)
    # -----------------------------------------------------------------

    def _build_chunk_citation_map(self) -> Dict[int, str]:
        """Map chunk_id -> human-readable citation label from workspace chunks.

        Example: {12345: "Vassiliev p3072", 67890: "Venona p42"}
        """
        if not self.workspace:
            return {}
        cmap: Dict[int, str] = {}
        for c in self.workspace.fulltext_chunks:
            parts = []
            if c.source_label:
                parts.append(c.source_label.replace("_", " ").title())
            if c.page:
                parts.append(c.page)
            label = " ".join(parts) if parts else f"chunk {c.chunk_id}"
            cmap[c.chunk_id] = label
        return cmap

    def build_citation_detail_map(self) -> Dict[str, Dict[str, Any]]:
        """Map citation label -> {chunk_id, document_id, page} for frontend linking.

        Produces the inverse of _build_chunk_citation_map(), enriched with doc_id
        and integer page number so the frontend can open the PDF viewer directly.

        Example: {"Vassiliev P4": {"chunk_id": 123, "document_id": 67, "page": 4}}
        """
        import re as _re
        if not self.workspace:
            return {}
        detail_map: Dict[str, Dict[str, Any]] = {}
        for c in self.workspace.fulltext_chunks:
            parts = []
            if c.source_label:
                parts.append(c.source_label.replace("_", " ").title())
            if c.page:
                parts.append(c.page)
            label = " ".join(parts) if parts else f"chunk {c.chunk_id}"
            if label not in detail_map:
                page_num = None
                if c.page:
                    m = _re.search(r'(\d+)', c.page)
                    if m:
                        page_num = int(m.group(1))
                detail_map[label] = {
                    "chunk_id": c.chunk_id,
                    "document_id": c.doc_id,
                    "page": page_num,
                }
        return detail_map

    def _format_chunk_citations(
        self,
        chunk_ids: List[int],
        cmap: Dict[int, str],
        max_show: int = 3,
        shown_pages: Optional[Set[str]] = None,
    ) -> str:
        """Render chunk IDs as human-readable citations.

        Example: "[Vassiliev p3072, Venona p42]"

        When `shown_pages` is provided, any document-page label already mentioned earlier
        in the answer is dropped (and newly shown labels are recorded), so the same page is
        never cited twice across the response. If every label was already shown, returns "".
        """
        if not chunk_ids:
            return ""
        # Deduplicate labels within this bracket, and against pages already shown earlier.
        seen: Set[str] = set()
        labels: List[str] = []
        for cid in chunk_ids:
            label = cmap.get(cid, f"chunk:{cid}")
            if label in seen:
                continue
            seen.add(label)
            if shown_pages is not None and label in shown_pages:
                continue
            labels.append(label)
        if shown_pages is not None:
            shown_pages.update(labels)
        if not labels:
            return ""
        display = labels[:max_show]
        if len(labels) > max_show:
            display.append(f"+{len(labels) - max_show} more")
        return f"[{', '.join(display)}]"

    @staticmethod
    def _is_valid_entity_for_linking(canonical_name: str) -> bool:
        """Check if an entity's canonical name is valid for entity linking.

        Rejects garbage entity names that are clearly not people/orgs:
        - Book citations: "Perjury , 183–84; West Weinstein"
        - Index fragments with numbers or page references
        - Strings with semicolons, em-dashes, or excess punctuation
        """
        import re as _re
        if not canonical_name or len(canonical_name) < 2:
            return False
        # Reject strings with numbers (page refs), semicolons, em-dashes
        if _re.search(r'[0-9;–—]', canonical_name):
            return False
        # Reject strings with multiple commas
        if canonical_name.count(',') > 1:
            return False
        return True

    def _evidence_mentions(self, terms: List[str]) -> int:
        """Count how many evidence chunks mention any of the given terms.

        This is a lightweight text-scan over the workspace chunks already in
        memory — no DB call.  Used to score entity relevance when multiple
        entities claim the same alias.
        """
        if not self.workspace or not self.workspace.chunks:
            return 0
        terms_lower = [t.lower() for t in terms if t]
        if not terms_lower:
            return 0
        count = 0
        for ch in self.workspace.chunks:
            txt_lower = (ch.text or "").lower()
            if any(t in txt_lower for t in terms_lower):
                count += 1
        return count

    def _build_alias_map(self) -> Dict[str, Dict[str, Any]]:
        """Build alias -> canonical_name map for entity linking in output.

        PEM-truthy (A): When workspace._pem_operational_alias_map is set,
        use it exclusively. Only PEM-backed surfaces in scope drive expansion.

        Fallback (no PEM or no alias-scoped):
        1. For each workspace entity, register every alias as a *claim*.
        2. Resolve collisions via evidence scoring.
        3. Accepted candidates added last.
        """
        import sys as _sys

        if not self.workspace:
            return {}

        # PEM-truthy: use operational map when provided (even if empty)
        if getattr(self.workspace, "_pem_operational_alias_map", None) is not None:
            amap = self.workspace._pem_operational_alias_map
            if amap:
                print(
                    f"  [V9 AliasMap] {len(amap)} PEM-backed entries (operational only)",
                    file=_sys.stderr,
                )
            return dict(amap) if amap else {}

        # ------------------------------------------------------------------
        # Fallback: Per-entity claims  (no merging between entities)
        # ------------------------------------------------------------------
        # claim_key (alias lower) -> list of (canonical_name, original_form, entity)
        claims: Dict[str, List[tuple]] = {}

        valid_entities = []
        # Also build a set of all canonical names for validation in step 3
        canonical_names_lower: Set[str] = set()

        for e in self.workspace.entities:
            if not self._is_valid_entity_for_linking(e.canonical_name):
                continue
            # Skip non-person entities and topic/operation cover-words: these are never real
            # people and glossing them produces garbage like "atomic (Balloon)".
            etype = (getattr(e, "entity_type", None) or "").lower()
            if etype and etype not in ("person", "people"):
                continue
            if e.canonical_name.lower() in _NON_PERSON_GLOSS_TOKENS:
                continue
            valid_entities.append(e)
            canonical_names_lower.add(e.canonical_name.lower())

        for e in valid_entities:
            # Target is ALWAYS canonical_name — the authoritative identifier
            target = e.canonical_name

            # Register each alias as a claim pointing to this canonical
            for alias in e.aliases:
                alias = (alias or "").strip()
                if len(alias) < 2:
                    continue
                key = alias.lower()
                if key == target.lower():
                    continue  # skip self-mapping
                if key in _NON_PERSON_GLOSS_TOKENS:
                    continue  # never gloss a common topic/cover word ("Atomic" -> "Balloon")
                claims.setdefault(key, []).append((target, alias, e))

        # ------------------------------------------------------------------
        # Step 2: Resolve collisions via evidence scoring
        # ------------------------------------------------------------------
        amap: Dict[str, Dict[str, Any]] = {}

        # Cache evidence scores per entity_id to avoid rescanning
        _ev_cache: Dict[int, int] = {}

        def _entity_evidence_score(ent: "WorkspaceEntity") -> int:
            eid = ent.entity_id
            if eid not in _ev_cache:
                search_terms = [ent.canonical_name] + list(ent.aliases)
                _ev_cache[eid] = self._evidence_mentions(search_terms)
            return _ev_cache[eid]

        for alias_key, claimants in claims.items():
            if len(claimants) == 1:
                # Unambiguous — only one entity claims this alias
                target, form, ent = claimants[0]
                amap[form] = {
                    "canonical": target,
                    "confidence": "exact",
                    "ambiguous": False,
                }
            else:
                # Multiple entities claim this alias — score by evidence
                scored = []
                for target, form, ent in claimants:
                    score = _entity_evidence_score(ent)
                    scored.append((score, target, form, ent))
                scored.sort(key=lambda x: x[0], reverse=True)

                best_score = scored[0][0]
                second_score = scored[1][0] if len(scored) > 1 else 0

                # Use winning claimant's form for the map key
                form_to_use = scored[0][2]

                if best_score > 0 and best_score > second_score:
                    # Clear winner — enough evidence gap
                    amap[form_to_use] = {
                        "canonical": scored[0][1],
                        "confidence": "exact",
                        "ambiguous": False,
                    }
                else:
                    # Tied or no evidence — mark ambiguous, do not expand
                    amap[form_to_use] = {
                        "canonical": scored[0][1],
                        "confidence": "exact",
                        "ambiguous": True,
                    }

        # ------------------------------------------------------------------
        # Step 3: Accepted candidates (weaker, never overwrite entity entries)
        #         Only add if the candidate's canonical_name is itself a known
        #         entity canonical — ensures we map alias→canonical, not
        #         alias→alias.
        # ------------------------------------------------------------------
        for c in self.workspace.entity_candidates:
            if c.accepted and c.query_term and c.canonical_name:
                qt = c.query_term
                cn = c.canonical_name
                if qt.lower() == cn.lower() or len(qt) < 2:
                    continue
                # Only emit if canonical_name is a real entity canonical
                if cn.lower() not in canonical_names_lower:
                    continue
                if qt not in amap:
                    amap[qt] = {
                        "canonical": cn,
                        "confidence": c.confidence,
                        "ambiguous": c.ambiguous,
                    }

        # ------------------------------------------------------------------
        # Step 4: Log alias map for diagnostics
        # ------------------------------------------------------------------
        if amap:
            print(
                f"  [V9 AliasMap] {len(amap)} entries from "
                f"{len(valid_entities)} valid entities + "
                f"{sum(1 for c in self.workspace.entity_candidates if c.accepted)} accepted candidates",
                file=_sys.stderr,
            )
            for alias, entry in sorted(amap.items()):
                amb = " [AMBIGUOUS]" if entry.get("ambiguous") else ""
                print(
                    f"    {alias} -> {entry['canonical']}{amb}",
                    file=_sys.stderr,
                )
        else:
            print(
                f"  [V9 AliasMap] EMPTY — entities={len(self.workspace.entities) if self.workspace else 0}, "
                f"candidates={len(self.workspace.entity_candidates) if self.workspace else 0}",
                file=_sys.stderr,
            )

        return amap

    def _expand_entity_refs(self, text: str, alias_map: Dict[str, Dict[str, Any]]) -> str:
        """Expand first occurrence of known aliases in text.

        Expansion rules based on confidence/ambiguity:
        - exact/partial + not ambiguous: "ALIAS (canonical_name)"
        - concordance + not ambiguous:   "ALIAS (canonical_name)"
        - ambiguous:                     "ALIAS (unresolved codename)"
        - inferred:                      no expansion (too risky)
        """
        if not alias_map:
            return text
        import re as _re
        expanded: Set[str] = set()
        result = text
        # Sort by length descending so longer aliases match first
        for alias in sorted(alias_map, key=len, reverse=True):
            entry = alias_map[alias]
            canonical = entry["canonical"]
            confidence = entry["confidence"]
            ambiguous = entry["ambiguous"]

            # Skip inferred-only resolutions (too risky)
            if confidence == "inferred":
                continue

            if canonical.lower() in expanded:
                continue  # Already expanded this canonical

            # Check if alias appears as whole word (case-insensitive)
            pattern = _re.compile(r'\b' + _re.escape(alias) + r'\b', _re.IGNORECASE)
            match = pattern.search(result)
            if match:
                # Guard: a mixed-case alias that is a person's FIRST name (immediately followed by
                # a capitalized surname) is part of a real name, not a standalone codename — do not
                # expand it, or "Jacob Golos" wrongly becomes "Jacob (William Perl) Golos". ALL-CAPS
                # occurrences are unmistakable codenames and are always expanded.
                matched_text = match.group()
                if not matched_text.isupper():
                    after = result[match.end():match.end() + 30]
                    if _re.match(r'\s+[A-Z][a-z]', after):
                        continue
                # Don't expand if canonical is already nearby (within 50 chars)
                start = max(0, match.start() - 50)
                end = min(len(result), match.end() + 50)
                neighborhood = result[start:end].lower()
                if canonical.lower() in neighborhood:
                    continue

                # Choose expansion label based on confidence/ambiguity
                if ambiguous:
                    replacement = f"{match.group()} (unresolved codename)"
                else:
                    replacement = f"{match.group()} ({canonical})"

                result = result[:match.start()] + replacement + result[match.end():]
                expanded.add(canonical.lower())
        return result

    def _log_entity_linking_summary(self) -> None:
        """Log entity linking summary for diagnostics (called once at render time)."""
        import sys as _sys
        if not self.workspace:
            return
        print(
            f"  [V9 EntityLinking] workspace has {len(self.workspace.entities)} entities, "
            f"{len(self.workspace.entity_candidates)} candidates "
            f"({sum(1 for c in self.workspace.entity_candidates if c.accepted)} accepted)",
            file=_sys.stderr,
        )
        for e in self.workspace.entities:
            alias_str = ", ".join(e.aliases[:6]) if e.aliases else "(none)"
            print(
                f"    Entity: {e.canonical_name} (id={e.entity_id}) "
                f"aliases=[{alias_str}]",
                file=_sys.stderr,
            )

    # -----------------------------------------------------------------
    # Main rendering
    # -----------------------------------------------------------------

    def format_answer(self) -> str:
        """User-facing answer constructed from grounded claims + concordance expansion.

        Grounding-first rendering:
        - Primary content: grounded claims with chunk-derived citations.
        - Model narrative is secondary and labeled as "draft/unverified" if
          any factual claims lack grounding.
        - Artifact sections show chunk-derived page/source citations.
        - Entity aliases expanded to 'ALIAS (canonical)' with ambiguity gate.
        """
        lines = []
        syn = self.synthesis
        cmap = self._build_chunk_citation_map()
        self._log_entity_linking_summary()
        alias_map = self._build_alias_map()

        # Classify claims by grounding status
        # Findings: grounded only (validated provenance).
        # Unverified: weak + heuristic (partial or overlap-only, no provenance).
        grounded_claims = [c for c in self.claims if c.status == "grounded" and c.citation_chunk_ids]
        unverified_claims = [c for c in self.claims if c.status in ("weak", "heuristic")]
        unsupported_claims = [c for c in self.claims if c.status == "unsupported"]
        factual_claims = [c for c in self.claims if c.claim.requires_citation]
        all_grounded = len(unsupported_claims) == 0 and len(unverified_claims) == 0

        # Evidence bullets are grounded by design: they have supporting_chunk_ids (provenance).
        # When synthesis yields no grounded claims, promote bullets to grounded so all evidence is surfaced as findings.
        if not grounded_claims and self.workspace and hasattr(self.workspace, "_bullet_index"):
            loaded_cids = {c.chunk_id for c in self.workspace.fulltext_chunks}
            for b in getattr(self.workspace, "_bullet_index", {}).values():
                valid_cids = [cid for cid in (b.supporting_chunk_ids or []) if cid in loaded_cids]
                if valid_cids:
                    grounded_claims.append(GroundedClaim(
                        claim=V9Claim(text=b.text, confidence="medium", requires_citation=True),
                        status="grounded",
                        citation_chunk_ids=valid_cids[:5],
                        note="Grounded via evidence summarizer",
                    ))

        # Dedup findings by normalized text so the same statement isn't repeated.
        _seen_finding_keys: Set[str] = set()
        _deduped_gc: List[GroundedClaim] = []
        for gc in grounded_claims:
            key = re.sub(r"\W+", " ", (gc.claim.text or "").lower()).strip()[:100]
            if key and key in _seen_finding_keys:
                continue
            _seen_finding_keys.add(key)
            _deduped_gc.append(gc)
        grounded_claims = _deduped_gc

        # Global citation tracker: a given document-page label is shown at most once across
        # the whole answer, so the same page is never mentioned twice. The first line to cite
        # a page shows it; later references to the same page render without a redundant bracket.
        shown_pages: Set[str] = set()

        # Identity resolution at the top (from artifact)
        # Cross-reference with workspace entities to ensure we display the
        # correct canonical_name (the model may emit a partial/wrong name).
        if syn:
            ident = syn.get_identity()
            if ident:
                display_canonical = ident.canonical
                display_alias = ident.alias

                # Try to resolve via entity_id first
                if ident.entity_id and self.workspace:
                    for e in self.workspace.entities:
                        if e.entity_id == ident.entity_id:
                            display_canonical = e.canonical_name
                            break

                # If no entity_id, try matching by alias text against workspace
                if not ident.entity_id and self.workspace:
                    for e in self.workspace.entities:
                        all_names = [e.canonical_name.lower()] + [a.lower() for a in e.aliases]
                        if ident.alias.lower() in all_names or ident.canonical.lower() in all_names:
                            display_canonical = e.canonical_name
                            break

                # Also check alias_hypotheses for validated mappings
                if self.workspace and hasattr(self.workspace, 'alias_hypotheses'):
                    for h in self.workspace.alias_hypotheses.values():
                        if h.status == "validated" and h.alias_text.lower() == ident.alias.lower():
                            for e in self.workspace.entities:
                                if e.entity_id == h.entity_id:
                                    display_canonical = e.canonical_name
                                    break
                            break

                lines.append(f"{display_alias} = {display_canonical}")
                if ident.basis:
                    for b in ident.basis[:3]:
                        btype = b.get('type', 'entity_index')
                        matched = b.get('matched_via', '')
                        cids = b.get('support_chunk_ids', [])
                        cite = self._format_chunk_citations(cids, cmap, shown_pages=shown_pages) if cids else ""
                        line = f"  (source: {btype}, matched via: {matched})"
                        if cite:
                            line += f" {cite}"
                        lines.append(line)
                lines.append("")

        # --- Primary content: grounded claims with citations ---
        if grounded_claims:
            lines.append("Findings:")
            for gc in grounded_claims:
                cite = self._format_chunk_citations(gc.citation_chunk_ids, cmap, shown_pages=shown_pages)
                claim_text = self._expand_entity_refs(gc.claim.text[:200], alias_map)
                lines.append(f"  - {claim_text} {cite}".rstrip())
            lines.append("")

        # Unverified claims (weak + heuristic — partial or overlap-only evidence, no provenance)
        if unverified_claims:
            lines.append("Unverified (partial or overlap-only evidence):")
            for gc in unverified_claims:
                cite = self._format_chunk_citations(gc.citation_chunk_ids, cmap, shown_pages=shown_pages)
                claim_text = self._expand_entity_refs(gc.claim.text[:200], alias_map)
                line = f"  - {claim_text}"
                if cite:
                    line += f" {cite}"
                if gc.note:
                    line += f" ({gc.note})"
                lines.append(line)
            lines.append("")

        # --- Narrative: show when we have grounded content (claims or bullet-derived) ---
        if self.narrative and grounded_claims:
            narrative = self._expand_entity_refs(self.narrative, alias_map)
            if not all_grounded:
                lines.append("--- Narrative (draft/unverified — claims above are the grounded findings) ---")
            else:
                lines.append("--- Summary ---")
            lines.append(narrative)
        elif not grounded_claims and self.narrative:
            lines.append("--- No grounded summary available — no claims with valid citations ---")

        # --- Roster: grounded/weak only — entries must have valid support_chunk_ids ---
        grounded_roster = [gr for gr in self.grounded_roster if gr.status in ("grounded", "weak")]
        if grounded_roster:
            lines.append("")
            lines.append("Members identified:")
            for gr in grounded_roster:
                r = gr.entry
                cite_ids = gr.valid_chunk_ids or r.support_chunk_ids
                name = self._expand_entity_refs(r.name, alias_map)
                role = f" ({self._expand_entity_refs(r.role, alias_map)})" if r.role else ""
                cite = self._format_chunk_citations(cite_ids, cmap, shown_pages=shown_pages) if cite_ids else ""
                line = f"  - {name}{role}"
                if cite:
                    line += f" {cite}"
                if gr.status == "weak":
                    line += " (partial evidence)"
                lines.append(line)

        # --- Other artifact sections with citations ---
        if syn:

            timeline = syn.get_timeline()
            if timeline:
                lines.append("")
                lines.append("Timeline:")
                for t in timeline:
                    cite = self._format_chunk_citations(t.support_chunk_ids, cmap, shown_pages=shown_pages) if t.support_chunk_ids else ""
                    event = self._expand_entity_refs(t.event, alias_map)
                    line = f"  {t.date}: {event}"
                    if cite:
                        line += f" {cite}"
                    lines.append(line)

            evidence = syn.get_evidence()
            if evidence:
                ev_lines: List[str] = []
                for e in evidence:
                    # Prefer chunk-derived citation over model-provided source/page
                    if e.chunk_id and e.chunk_id in cmap:
                        cite_label = cmap[e.chunk_id]
                    elif e.source or e.page:
                        cite_label = f"{e.source} {e.page}".strip()
                    else:
                        cite_label = "unattributed"
                    # Skip any document-page already mentioned earlier in the answer (findings,
                    # members, timeline) or already listed in this Evidence section — one quote
                    # per page, no duplicate page mentions.
                    if cite_label != "unattributed" and cite_label in shown_pages:
                        continue
                    if cite_label != "unattributed":
                        shown_pages.add(cite_label)
                    quote = self._expand_entity_refs(e.quote[:200], alias_map)
                    ev_lines.append(f'  [{cite_label}]: "{quote}"')
                if ev_lines:
                    lines.append("")
                    lines.append("Evidence:")
                    lines.extend(ev_lines)

            edges = syn.get_relationships()
            if edges:
                lines.append("")
                lines.append("Relationships:")
                for edge in edges:
                    cite = self._format_chunk_citations(edge.support_chunk_ids, cmap, shown_pages=shown_pages) if edge.support_chunk_ids else ""
                    entity_a = self._expand_entity_refs(edge.entity_a, alias_map)
                    entity_b = self._expand_entity_refs(edge.entity_b, alias_map)
                    relation = self._expand_entity_refs(edge.relation, alias_map)
                    line = f"  {entity_a} --{relation}-- {entity_b}"
                    if cite:
                        line += f" {cite}"
                    lines.append(line)

        # Unsupported claims (explicit warning)
        if unsupported_claims:
            lines.append("")
            lines.append("Unsupported claims (no citation found):")
            for gc in unsupported_claims[:5]:
                lines.append(f"  - {gc.claim.text[:120]}")

        # Verification summary
        if self.verification and (self.verification.weak_claims or self.verification.unsupported_claims):
            lines.append("")
            lines.append("---")
            lines.append(
                f"Verification: {self.verification.grounded_claims} grounded, "
                f"{self.verification.weak_claims} unverified (weak/heuristic), "
                f"{self.verification.unsupported_claims} unsupported."
            )
            all_notes = self.verification.artifact_notes + self.verification.notes
            for n in all_notes[:5]:
                lines.append(f"  * {n}")
        if self.verification and self.verification.responsiveness and not self.verification.responsiveness.responsive:
            lines.append(f"  Note: {self.verification.responsiveness.suggestion}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative,
            "claims": [c.to_dict() for c in self.claims],
            "verification": self.verification.to_dict() if self.verification else None,
            "sufficiency": self.sufficiency.to_dict() if self.sufficiency else None,
            "synthesis": self.synthesis.to_dict() if self.synthesis else None,
            "investigation_trace": [s.to_dict() for s in self.investigation_trace],
        }
