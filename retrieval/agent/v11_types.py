"""
V11 Types — stripped-down V9 (no PEM, no query entity resolution).

Re-exports V9 types except ResearchWorkspace, which omits PEM fields:
- pem_seed_chunk_ids
- _pem_cache
- _pem_canonical_map
- _pem_operational_alias_map

Alias map and other state (alias_hypotheses, entities, evidence_memory, etc.) retained.
"""
from retrieval.agent.v9_types import (
    _normalize_text,
    compute_bullet_id,
    ScopeFilter,
    AliasHypothesis,
    EntityCandidate,
    CatalogHit,
    WorkspaceChunk,
    WorkspaceEntity,
    InvestigationStep,
    InvestigationState,
    WorkspaceDelta,
    ProgressSignal,
    EvidenceBullet,
    EvidenceSummaryUpdate,
    EvidenceMemoryView,
    SufficiencyCheck,
    RosterEntry,
    TimelineEntry,
    EvidenceEntry,
    RelationshipEdge,
    IdentityResolution,
    V9Claim,
    V9Synthesis,
    GroundedClaim,
    GroundedRosterEntry,
    ResponsivenessResult,
    V9VerificationReport,
    V9Result,
)
from retrieval.agent.v9_types import _is_critical_pin as _v9_critical_pin


def _rehydrate_evidence_index(workspace: "V11ResearchWorkspace") -> None:
    """Rebuild _bullet_index, _summarized_chunk_ids, pinned_bullet_ids from evidence_memory."""
    workspace._bullet_index.clear()
    workspace._summarized_chunk_ids.clear()
    pinned_bullets = []
    for update in workspace.evidence_memory:
        workspace._summarized_chunk_ids.update(update.generated_from_chunk_ids)
        for b in update.bullets:
            workspace._bullet_index[b.bullet_id] = b
            if b.pinned:
                pinned_bullets.append(b)
    critical = sorted(
        [b for b in pinned_bullets if _v9_critical_pin(b)],
        key=lambda b: b.created_at or "", reverse=True,
    )
    non_critical = sorted(
        [b for b in pinned_bullets if not _v9_critical_pin(b)],
        key=lambda b: b.created_at or "", reverse=True,
    )
    workspace.pinned_bullet_ids = [b.bullet_id for b in critical + non_critical]


from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


@dataclass
class V11ResearchWorkspace:
    """V11 workspace: same as V9 but no PEM lane fields."""
    question: str
    scope: ScopeFilter = field(default_factory=ScopeFilter)
    catalog_hits: List[CatalogHit] = field(default_factory=list)
    fulltext_chunks: List[WorkspaceChunk] = field(default_factory=list)
    entities: List[WorkspaceEntity] = field(default_factory=list)
    entity_candidates: List[EntityCandidate] = field(default_factory=list)
    investigation: InvestigationState = field(default_factory=InvestigationState)
    notes: List[str] = field(default_factory=list)
    uncertainty_flags: List[str] = field(default_factory=list)
    evidence_memory: List[EvidenceSummaryUpdate] = field(default_factory=list)
    pinned_bullet_ids: List[str] = field(default_factory=list)
    _bullet_index: Dict[str, EvidenceBullet] = field(default_factory=dict)
    _summarized_chunk_ids: set = field(default_factory=set)
    alias_hypotheses: Dict[tuple, AliasHypothesis] = field(default_factory=dict)
    _search_queries: List[str] = field(default_factory=list)
    _prev_doc_ids: set = field(default_factory=set)
    _prev_person_names: set = field(default_factory=set)

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
        _rehydrate_evidence_index(self)

    def accept_candidate(self, entity_id: int) -> None:
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


# Alias for compatibility
ResearchWorkspace = V11ResearchWorkspace
