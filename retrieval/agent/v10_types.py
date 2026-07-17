"""
V10 Types — Scope-Aware Alias Identity Layer.

Core data structures for the V10 identity substrate:
- SpanLatticeV10: query candidate parse with overlapping spans
- LexiconV10: two-layer identity memory (global entities + scoped aliases)
- ChunkMentionsV10: per-chunk extraction artifacts
- ResolutionPlanV10: agent decisions per round
- AliasReferentRule: doc/page-scoped referent mappings
- AliasMappingHypothesis: contextual + general alias hypotheses
- ResolvedAlias: output of the central resolver

Invariants:
  I1 — Alias semantics are collection-scoped (venona/vassiliev only)
  I2 — Entity identity is global (entity_id works everywhere)
  I3 — Enumeration is deterministic; selection is agentic
  I4 — No free-token alias drift
  I5 — Closed loop: retrieval -> evidence -> identity -> retrieval
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Span lattice (query candidate parse)
# =============================================================================

@dataclass
class SpanCandidate:
    """One candidate entity mapping for a query span."""
    entity_id: int
    canonical_name: str
    match_type: str  # canonical|alias|codename|fuzzy_fullname|fuzzy_alias|synthesized_fullname
    alias_type: Optional[str] = None  # code_name|alt|primary|misspelling|initials|ru_translit|...
    surface_kind: str = "name"  # codename_alias|general_alias|acronym|name|phrase
    collision: str = "low"  # low|med|high — how many entities share this surface
    valid_collections: List[str] = field(default_factory=lambda: ["*"])
    edit_distance: Optional[int] = None
    source: str = ""  # alias_table|canonical_table|concordance|entity_metadata
    score: float = 0.0  # composite score for ranking
    prior_count_global: int = 0  # aggregate doc_freq or mention_count
    prior_count_by_collection: Dict[str, int] = field(default_factory=dict)
    # collection_slug -> count (populated when occurrence-level index exists)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "match_type": self.match_type,
            "alias_type": self.alias_type,
            "surface_kind": self.surface_kind,
            "collision": self.collision,
            "valid_collections": self.valid_collections,
            "edit_distance": self.edit_distance,
            "source": self.source,
            "score": self.score,
        }
        if self.prior_count_global:
            d["prior_count_global"] = self.prior_count_global
        if self.prior_count_by_collection:
            d["prior_count_by_collection"] = self.prior_count_by_collection
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpanCandidate":
        return cls(
            entity_id=d.get("entity_id", 0),
            canonical_name=d.get("canonical_name", ""),
            match_type=d.get("match_type", ""),
            alias_type=d.get("alias_type"),
            surface_kind=d.get("surface_kind", "name"),
            collision=d.get("collision", "low"),
            valid_collections=d.get("valid_collections", ["*"]),
            edit_distance=d.get("edit_distance"),
            source=d.get("source", ""),
            score=d.get("score", 0.0),
            prior_count_global=d.get("prior_count_global", 0),
            prior_count_by_collection=d.get("prior_count_by_collection", {}),
        )


@dataclass
class SpanEntry:
    """One span in the query with its candidate entity mappings."""
    span_id: str
    text: str
    start: int
    end: int
    norm_key: str = ""  # normalized form used for lookups
    span_kind: str = "phrase"  # name|alias_surface|phrase
    resolution_status: str = "unresolved"  # resolved|ambiguous|unresolved
    candidates: List[SpanCandidate] = field(default_factory=list)
    overlaps: List[str] = field(default_factory=list)    # span_ids that partially overlap
    dominates: List[str] = field(default_factory=list)   # span_ids this span fully covers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "norm_key": self.norm_key,
            "span_kind": self.span_kind,
            "resolution_status": self.resolution_status,
            "candidates": [c.to_dict() for c in self.candidates],
            "overlaps": self.overlaps,
            "dominates": self.dominates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpanEntry":
        return cls(
            span_id=d.get("span_id", ""),
            text=d.get("text", ""),
            start=d.get("start", 0),
            end=d.get("end", 0),
            norm_key=d.get("norm_key", ""),
            span_kind=d.get("span_kind", "phrase"),
            resolution_status=d.get("resolution_status", "unresolved"),
            candidates=[SpanCandidate.from_dict(c) for c in d.get("candidates", [])],
            overlaps=d.get("overlaps", []),
            dominates=d.get("dominates", []),
        )


@dataclass
class SpanLatticeV10:
    """Lattice of overlapping spans with candidate entity/alias mappings.

    Produced by spot_query_spans_v10(). No decisions are made here —
    the LLM selects a non-overlapping set in Stage A.
    """
    query: str
    spans: List[SpanEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "spans": [s.to_dict() for s in self.spans],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpanLatticeV10":
        return cls(
            query=d.get("query", ""),
            spans=[SpanEntry.from_dict(s) for s in d.get("spans", [])],
        )


# =============================================================================
# Alias referent rules (doc/page-scoped — the primary anti-ambiguity mechanism)
# =============================================================================

@dataclass
class AliasReferentRule:
    """A known alias->entity mapping scoped to a specific document
    (and optionally page range).

    These come from curated data, prior confirmed hypotheses, or
    deterministic extraction (e.g. "KING = Julius Rosenberg" found
    in a specific Venona decrypt).

    Consulted BEFORE any general collection-level hypothesis.
    """
    collection_slug: str
    alias_text: str
    document_id: int
    page_from: Optional[int] = None   # None = entire document
    page_to: Optional[int] = None
    entity_id: int = 0
    status: str = "confirmed"   # confirmed|possible|rejected
    note: str = ""
    rule_id: Optional[int] = None  # DB primary key, if loaded from DB

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_slug": self.collection_slug,
            "alias_text": self.alias_text,
            "document_id": self.document_id,
            "page_from": self.page_from,
            "page_to": self.page_to,
            "entity_id": self.entity_id,
            "status": self.status,
            "note": self.note,
            "rule_id": self.rule_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AliasReferentRule":
        return cls(
            collection_slug=d.get("collection_slug", ""),
            alias_text=d.get("alias_text", ""),
            document_id=d.get("document_id", 0),
            page_from=d.get("page_from"),
            page_to=d.get("page_to"),
            entity_id=d.get("entity_id", 0),
            status=d.get("status", "confirmed"),
            note=d.get("note", ""),
            rule_id=d.get("rule_id"),
        )

    def covers_page(self, page_no: Optional[int]) -> bool:
        """Does this rule apply to the given page?"""
        if self.page_from is None:
            return True  # doc-wide rule
        if page_no is None:
            return True  # no page info — doc-wide rules match
        return self.page_from <= page_no <= (self.page_to or self.page_from)

    @property
    def interval_width(self) -> Optional[int]:
        """Width of the page interval.  None for doc-wide rules."""
        if self.page_from is None:
            return None
        return (self.page_to or self.page_from) - self.page_from


# =============================================================================
# Alias mapping hypotheses (contextual + general, unified dataclass)
# =============================================================================

# Status priority for tie-breaking (higher = stronger)
_STATUS_PRIORITY = {"confirmed": 3, "possible": 2, "rejected": 1}


@dataclass
class AliasMappingHypothesis:
    """An alias->entity mapping hypothesis.

    When document_id is set, this is a *contextual* hypothesis scoped to
    a specific document (and optionally page range).
    When document_id is None, this is a *general* collection-level hypothesis.

    Critical invariant: confirming a contextual hypothesis does NOT
    auto-promote the general hypothesis.
    """
    collection_slug: str
    alias_text: str
    candidates: List[SpanCandidate] = field(default_factory=list)
    status: str = "unresolved"  # unresolved|provisional|confirmed
    confidence: float = 0.0
    support: List[Dict[str, Any]] = field(default_factory=list)
    # Contextual scope (None = general/collection-wide)
    document_id: Optional[int] = None
    page_from: Optional[int] = None
    page_to: Optional[int] = None

    @property
    def is_contextual(self) -> bool:
        return self.document_id is not None

    @property
    def hypothesis_key(self) -> Tuple:
        """Key for dict lookup. Contextual includes doc/page; general does not."""
        if self.is_contextual:
            return (self.collection_slug, self.alias_text, self.document_id,
                    self.page_from, self.page_to)
        return (self.collection_slug, self.alias_text, None, None, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_slug": self.collection_slug,
            "alias_text": self.alias_text,
            "candidates": [c.to_dict() for c in self.candidates],
            "status": self.status,
            "confidence": self.confidence,
            "support": self.support,
            "document_id": self.document_id,
            "page_from": self.page_from,
            "page_to": self.page_to,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AliasMappingHypothesis":
        return cls(
            collection_slug=d.get("collection_slug", ""),
            alias_text=d.get("alias_text", ""),
            candidates=[SpanCandidate.from_dict(c) for c in d.get("candidates", [])],
            status=d.get("status", "unresolved"),
            confidence=d.get("confidence", 0.0),
            support=d.get("support", []),
            document_id=d.get("document_id"),
            page_from=d.get("page_from"),
            page_to=d.get("page_to"),
        )


# =============================================================================
# Resolved alias (output of the central resolver)
# =============================================================================

@dataclass
class ResolvedAlias:
    """Output of resolve_alias_candidates() — the single resolution path.

    This is produced at extraction time and recomputed on rehydrate
    (never persisted, to avoid stale locks).
    """
    alias_text: str
    candidates: List[SpanCandidate] = field(default_factory=list)
    locked_entity_id: Optional[int] = None  # set only when safe
    status: str = "unknown"   # confirmed|provisional|ambiguous|unknown
    source: str = ""          # referent_rule|contextual_hypothesis|general_hypothesis|alias_table
    context_document_id: Optional[int] = None
    context_page_no: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alias_text": self.alias_text,
            "candidates": [c.to_dict() for c in self.candidates],
            "locked_entity_id": self.locked_entity_id,
            "status": self.status,
            "source": self.source,
            "context_document_id": self.context_document_id,
            "context_page_no": self.context_page_no,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResolvedAlias":
        return cls(
            alias_text=d.get("alias_text", ""),
            candidates=[SpanCandidate.from_dict(c) for c in d.get("candidates", [])],
            locked_entity_id=d.get("locked_entity_id"),
            status=d.get("status", "unknown"),
            source=d.get("source", ""),
            context_document_id=d.get("context_document_id"),
            context_page_no=d.get("context_page_no"),
        )


# =============================================================================
# Alias context (input to the central resolver)
# =============================================================================

@dataclass
class AliasContext:
    """Context for alias resolution — passed to resolve_alias_candidates()."""
    collection_slug: str
    document_id: Optional[int] = None
    page_no: Optional[int] = None


# =============================================================================
# Chunk mentions (per-chunk extraction artifacts)
# =============================================================================

@dataclass
class ChunkMention:
    """A single mention (entity surface or alias surface) within a chunk."""
    surface: str
    start: int
    end: int
    kind: str  # entity_surface|alias_surface
    candidates: List[SpanCandidate] = field(default_factory=list)
    # Populated at extraction time, NOT persisted (recomputed on rehydrate)
    resolved: Optional[ResolvedAlias] = None

    def to_dict(self, include_resolved: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "surface": self.surface,
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "candidates": [c.to_dict() for c in self.candidates],
        }
        if include_resolved and self.resolved:
            d["resolved"] = self.resolved.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkMention":
        return cls(
            surface=d.get("surface", ""),
            start=d.get("start", 0),
            end=d.get("end", 0),
            kind=d.get("kind", ""),
            candidates=[SpanCandidate.from_dict(c) for c in d.get("candidates", [])],
            # resolved is NOT restored from persisted data (recomputed)
        )


@dataclass
class ChunkSignal:
    """A high-signal pattern detected within a chunk (alias equation, aka, etc.)."""
    signal_type: str  # alias_equation|aka|cryptonym_marker|parenthetical|identified_as|co_mention
    text: str = ""
    confidence: float = 0.0
    entity_a: Optional[str] = None
    entity_b: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "text": self.text,
            "confidence": self.confidence,
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkSignal":
        return cls(
            signal_type=d.get("signal_type", ""),
            text=d.get("text", ""),
            confidence=d.get("confidence", 0.0),
            entity_a=d.get("entity_a"),
            entity_b=d.get("entity_b"),
        )


@dataclass
class ChunkMentionsV10:
    """Per-chunk extraction artifact: mentions + signals.

    document_id and page_no are embedded directly so that ThinkDeeper
    rehydration is independent from DB joins.
    """
    chunk_id: int
    collection_slug: str
    document_id: int
    page_no: Optional[int] = None
    mentions: List[ChunkMention] = field(default_factory=list)
    signals: List[ChunkSignal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "collection_slug": self.collection_slug,
            "document_id": self.document_id,
            "page_no": self.page_no,
            # Mentions are serialised WITHOUT resolved field (recomputed)
            "mentions": [m.to_dict(include_resolved=False) for m in self.mentions],
            "signals": [s.to_dict() for s in self.signals],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkMentionsV10":
        return cls(
            chunk_id=d.get("chunk_id", 0),
            collection_slug=d.get("collection_slug", ""),
            document_id=d.get("document_id", 0),
            page_no=d.get("page_no"),
            mentions=[ChunkMention.from_dict(m) for m in d.get("mentions", [])],
            signals=[ChunkSignal.from_dict(s) for s in d.get("signals", [])],
        )


# =============================================================================
# CatalogHitV10 (extends V9 CatalogHit with doc/page provenance)
# =============================================================================

@dataclass
class CatalogHitV10:
    """Search result preview with full doc/page provenance for contextual
    alias resolution.  Extends the V9 CatalogHit concept."""
    chunk_id: int
    score: float
    doc_id: Optional[int] = None
    page: Optional[str] = None           # display form, e.g. "p5"
    collection: Optional[str] = None
    snippet: str = ""
    # V10 additions
    document_id: Optional[int] = None    # from chunk_metadata.document_id
    page_id: Optional[int] = None        # raw page row ID
    page_no: Optional[int] = None        # resolved PDF page number
    collection_slug: Optional[str] = None  # from chunk_metadata.collection_slug
    # Enrichment: "pem_seed" = PEM lane seed (prioritized for LLM extraction); "search" = search hit
    origin: Optional[str] = None          # pem_seed | search | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "doc_id": self.doc_id,
            "page": self.page,
            "collection": self.collection,
            "snippet": self.snippet[:400],
            "document_id": self.document_id,
            "page_id": self.page_id,
            "page_no": self.page_no,
            "collection_slug": self.collection_slug,
        }


# =============================================================================
# Structured search boosts
# =============================================================================

@dataclass
class EntityBoost:
    """Boost for global entity retrieval — uses canonical + global variant forms."""
    entity_id: int
    forms: List[str] = field(default_factory=list)  # from build_entity_forms()
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "forms": self.forms,
            "weight": self.weight,
        }


@dataclass
class AliasScopedBoost:
    """Scoped alias boost — only fires within the specified collection.

    locked_entity_id semantics: "this alias is being used to retrieve
    evidence for that entity hypothesis" — NOT "the alias uniquely
    identifies that entity globally."

    Set locked_entity_id ONLY when:
    - A contextual AliasReferentRule matched (confirmed for doc/page), OR
    - A contextual AliasMappingHypothesis is confirmed, OR
    - The alias is truly unambiguous in that collection.
    """
    collection_slug: str
    alias_text: str
    locked_entity_id: Optional[int] = None
    weight: float = 1.0
    case_mode: str = "insensitive"  # insensitive|exact

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_slug": self.collection_slug,
            "alias_text": self.alias_text,
            "locked_entity_id": self.locked_entity_id,
            "weight": self.weight,
            "case_mode": self.case_mode,
        }


@dataclass
class MatchProvenance:
    """Per-chunk provenance from search: which boost fired, which form matched."""
    chunk_id: int
    boost_type: str = ""  # entity|alias|none
    matched_form: str = ""
    collection_scope: Optional[str] = None
    entity_id: Optional[int] = None
    locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "boost_type": self.boost_type,
            "matched_form": self.matched_form,
            "collection_scope": self.collection_scope,
            "entity_id": self.entity_id,
            "locked": self.locked,
        }


# =============================================================================
# LexiconV10 (two-layer identity memory)
# =============================================================================

@dataclass
class LexiconV10:
    """Two-layer identity memory: global entities + scoped aliases.

    The lexicon accumulates identity state across retrieval rounds.
    It is serialisable for ThinkDeeper persistence and rebuildable
    from persisted evidence on rehydrate.
    """
    # Global layer
    entities_in_play: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # entity_id -> {canonical_name, global_variants, types, evidence_chunk_ids}
    entity_support: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # entity_id -> {support_score, support_chunks}

    # Scoped alias namespaces
    aliases_by_entity_scoped: Dict[int, Dict[str, List[str]]] = field(default_factory=dict)
    # entity_id -> {collection_slug -> [alias_text]}
    entities_by_alias_scoped: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    # collection_slug -> {alias_text -> [entity_ids]}

    # Contextual referent rules (loaded from DB, cached in memory)
    alias_referent_rules: Dict[str, List[AliasReferentRule]] = field(default_factory=dict)
    # keyed by f"{collection_slug}:{alias_text}:{document_id}"

    # Two-tier hypothesis store
    alias_mapping_hypotheses: Dict[tuple, AliasMappingHypothesis] = field(default_factory=dict)
    # Contextual: key = (collection_slug, alias_text, document_id, page_from, page_to)
    # General:    key = (collection_slug, alias_text, None, None, None)

    # Alias permissions — index-backed grant state
    # Key: (collection_slug, alias_surface_norm, entity_id_or_none)
    # Value: {"status": "confirmed"|"provisional", "index_truth_level": ..., "granted_at_round": int}
    alias_permissions: Dict[Tuple[str, str, Optional[int]], Dict[str, Any]] = field(
        default_factory=dict
    )
    # Revision id: track index version to invalidate stale permissions on rehydrate
    alias_index_revision: Optional[str] = None

    # Resolved referents — sticky entity commitment per span
    # Key: stable referent key (norm_key, start, end) as string
    # Value: {"entity_id": int, "status": "confirmed"|"provisional", "support_chunk_ids": []}
    resolved_referents: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Transient: tracks which entities have been backfilled (NOT serialized)
    _backfilled_entity_ids: Set[int] = field(default_factory=set)

    # ---- Referent rule helpers ----

    @staticmethod
    def _referent_key(collection_slug: str, alias_text: str, document_id: int) -> str:
        return f"{collection_slug}:{alias_text.lower()}:{document_id}"

    def get_referent_rules(
        self, collection_slug: str, alias_text: str, document_id: int
    ) -> List[AliasReferentRule]:
        """Get all referent rules for (collection, alias, document)."""
        key = self._referent_key(collection_slug, alias_text, document_id)
        return self.alias_referent_rules.get(key, [])

    def add_referent_rule(self, rule: AliasReferentRule) -> None:
        key = self._referent_key(rule.collection_slug, rule.alias_text, rule.document_id)
        self.alias_referent_rules.setdefault(key, []).append(rule)

    # ---- Hypothesis helpers ----

    def get_hypothesis(
        self,
        collection_slug: str,
        alias_text: str,
        document_id: Optional[int] = None,
        page_from: Optional[int] = None,
        page_to: Optional[int] = None,
    ) -> Optional[AliasMappingHypothesis]:
        key = (collection_slug, alias_text.lower(), document_id, page_from, page_to)
        return self.alias_mapping_hypotheses.get(key)

    def set_hypothesis(self, hypothesis: AliasMappingHypothesis) -> None:
        key = (
            hypothesis.collection_slug,
            hypothesis.alias_text.lower(),
            hypothesis.document_id,
            hypothesis.page_from,
            hypothesis.page_to,
        )
        self.alias_mapping_hypotheses[key] = hypothesis

    def get_contextual_hypotheses(
        self, collection_slug: str, alias_text: str, document_id: int
    ) -> List[AliasMappingHypothesis]:
        """Get all contextual hypotheses for (collection, alias, document)."""
        prefix = (collection_slug, alias_text.lower(), document_id)
        return [
            h for key, h in self.alias_mapping_hypotheses.items()
            if key[:3] == prefix and h.is_contextual
        ]

    def get_general_hypothesis(
        self, collection_slug: str, alias_text: str
    ) -> Optional[AliasMappingHypothesis]:
        """Get the general (collection-wide) hypothesis."""
        key = (collection_slug, alias_text.lower(), None, None, None)
        return self.alias_mapping_hypotheses.get(key)

    # ---- Entity helpers ----

    def register_entity(
        self,
        entity_id: int,
        canonical_name: str,
        global_variants: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
    ) -> None:
        if entity_id not in self.entities_in_play:
            self.entities_in_play[entity_id] = {
                "canonical_name": canonical_name,
                "global_variants": global_variants or [],
                "types": entity_type,
                "evidence_chunk_ids": [],
            }
        if entity_id not in self.entity_support:
            self.entity_support[entity_id] = {
                "support_score": 0.0,
                "support_chunks": [],
            }

    def add_entity_evidence(self, entity_id: int, chunk_id: int) -> None:
        if entity_id in self.entities_in_play:
            cids = self.entities_in_play[entity_id]["evidence_chunk_ids"]
            if chunk_id not in cids:
                cids.append(chunk_id)
        if entity_id in self.entity_support:
            sup = self.entity_support[entity_id]
            if chunk_id not in sup["support_chunks"]:
                sup["support_chunks"].append(chunk_id)
                sup["support_score"] = float(len(sup["support_chunks"]))

    # ---- Serialisation ----

    # ---- Permission helpers ----

    def has_alias_permission(
        self,
        collection_slug: str,
        alias_surface_norm: str,
        entity_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Check if a permission exists.  Returns the permission dict or None.

        If entity_id is None, checks for any permission for (coll, alias, *).
        """
        # Exact match first
        exact_key = (collection_slug, alias_surface_norm, entity_id)
        perm = self.alias_permissions.get(exact_key)
        if perm:
            return perm
        # Wildcard (no lock) permission
        wild_key = (collection_slug, alias_surface_norm, None)
        perm = self.alias_permissions.get(wild_key)
        if perm:
            return perm
        # If entity_id was None, also check any entity-specific permission
        if entity_id is None:
            for key, val in self.alias_permissions.items():
                if key[0] == collection_slug and key[1] == alias_surface_norm:
                    return val
        return None

    def grant_permission(
        self,
        collection_slug: str,
        alias_surface_norm: str,
        entity_id: Optional[int],
        status: str = "confirmed",
        index_truth_level: str = "derived",
        granted_at_round: int = 0,
    ) -> None:
        """Grant alias permission."""
        key = (collection_slug, alias_surface_norm, entity_id)
        self.alias_permissions[key] = {
            "status": status,
            "index_truth_level": index_truth_level,
            "granted_at_round": granted_at_round,
        }

    # ---- Resolved referent helpers ----

    @staticmethod
    def referent_key(norm_key: str, start: int, end: int) -> str:
        """Stable key for resolved_referents dict (survives round changes)."""
        return f"{norm_key}:{start}:{end}"

    def get_resolved_referent(
        self, norm_key: str, start: int, end: int
    ) -> Optional[Dict[str, Any]]:
        key = self.referent_key(norm_key, start, end)
        return self.resolved_referents.get(key)

    def set_resolved_referent(
        self,
        norm_key: str, start: int, end: int,
        entity_id: int, status: str = "confirmed",
        support_chunk_ids: Optional[List[int]] = None,
    ) -> None:
        key = self.referent_key(norm_key, start, end)
        self.resolved_referents[key] = {
            "entity_id": entity_id,
            "status": status,
            "support_chunk_ids": support_chunk_ids or [],
        }

    # ---- Serialisation ----

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for persistence.  Referent rules are NOT serialised
        (reloaded from DB on rehydrate)."""
        # Serialize alias_permissions: tuple keys -> string keys
        perms_ser: Dict[str, Any] = {}
        for (coll, alias, eid), val in self.alias_permissions.items():
            key_str = f"{coll}|{alias}|{eid if eid is not None else ''}"
            perms_ser[key_str] = val
        return {
            "entities_in_play": self.entities_in_play,
            "entity_support": self.entity_support,
            "aliases_by_entity_scoped": {
                str(k): v for k, v in self.aliases_by_entity_scoped.items()
            },
            "entities_by_alias_scoped": self.entities_by_alias_scoped,
            "alias_mapping_hypotheses": {
                str(k): v.to_dict()
                for k, v in self.alias_mapping_hypotheses.items()
            },
            "alias_permissions": perms_ser,
            "alias_index_revision": self.alias_index_revision,
            "resolved_referents": self.resolved_referents,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LexiconV10":
        lex = cls()
        lex.entities_in_play = d.get("entities_in_play", {})
        # Ensure entity_id keys are ints
        lex.entities_in_play = {
            int(k): v for k, v in lex.entities_in_play.items()
        }
        lex.entity_support = {
            int(k): v for k, v in d.get("entity_support", {}).items()
        }
        # aliases_by_entity_scoped
        raw_abes = d.get("aliases_by_entity_scoped", {})
        lex.aliases_by_entity_scoped = {int(k): v for k, v in raw_abes.items()}
        lex.entities_by_alias_scoped = d.get("entities_by_alias_scoped", {})
        # hypotheses
        for key_str, h_dict in d.get("alias_mapping_hypotheses", {}).items():
            h = AliasMappingHypothesis.from_dict(h_dict)
            lex.set_hypothesis(h)
        # alias_permissions
        for key_str, val in d.get("alias_permissions", {}).items():
            parts = key_str.split("|")
            if len(parts) == 3:
                coll, alias, eid_str = parts
                eid = int(eid_str) if eid_str else None
                lex.alias_permissions[(coll, alias, eid)] = val
        lex.alias_index_revision = d.get("alias_index_revision")
        lex.resolved_referents = d.get("resolved_referents", {})
        return lex


# =============================================================================
# ResolutionPlanV10 (agent decisions per round)
# =============================================================================

@dataclass
class SpanSelection:
    """LLM's structured output for span selection in Stage A."""
    chosen_span_ids: List[str] = field(default_factory=list)
    suppressed_span_ids: List[str] = field(default_factory=list)
    entity_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {entity_id, confidence, reason}
    alias_spans: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {span_id, activate_alias_resolution}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_span_ids": self.chosen_span_ids,
            "suppressed_span_ids": self.suppressed_span_ids,
            "entity_hypotheses": self.entity_hypotheses,
            "alias_spans": self.alias_spans,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpanSelection":
        return cls(
            chosen_span_ids=d.get("chosen_span_ids", []),
            suppressed_span_ids=d.get("suppressed_span_ids", []),
            entity_hypotheses=d.get("entity_hypotheses", []),
            alias_spans=d.get("alias_spans", []),
        )


@dataclass
class ResolutionPlanV10:
    """Agent decisions produced each round; ThinkDeeper can revise."""
    selected_spans: List[SpanEntry] = field(default_factory=list)
    span_selection: Optional[SpanSelection] = None
    entity_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    alias_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    branching: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_steps: List[Dict[str, Any]] = field(default_factory=list)
    promotion_actions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_spans": [s.to_dict() for s in self.selected_spans],
            "span_selection": self.span_selection.to_dict() if self.span_selection else None,
            "entity_hypotheses": self.entity_hypotheses,
            "alias_hypotheses": self.alias_hypotheses,
            "branching": self.branching,
            "retrieval_steps": self.retrieval_steps,
            "promotion_actions": self.promotion_actions,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResolutionPlanV10":
        return cls(
            selected_spans=[SpanEntry.from_dict(s) for s in d.get("selected_spans", [])],
            span_selection=(
                SpanSelection.from_dict(d["span_selection"])
                if d.get("span_selection") else None
            ),
            entity_hypotheses=d.get("entity_hypotheses", []),
            alias_hypotheses=d.get("alias_hypotheses", []),
            branching=d.get("branching", []),
            retrieval_steps=d.get("retrieval_steps", []),
            promotion_actions=d.get("promotion_actions", []),
        )


# =============================================================================
# Collections that support alias semantics
# =============================================================================

ALIAS_SCOPED_COLLECTIONS = frozenset({"venona", "vassiliev"})
"""Collections where alias/codename tokens have semantic meaning.
Alias extraction and alias boosts are restricted to these."""

CODENAME_ALIAS_KINDS = frozenset({"code_name", "covername"})
"""entity_aliases.kind values that represent Venona/Vassiliev codenames."""


# =============================================================================
# LLM extraction types (Live Lexicon Enrichment)
# =============================================================================

SURFACE_RATIONALE_VALUES = (
    "all_caps_codename",
    "known_entity_name",
    "new_proper_name",
    "contextual_alias_usage",
    "abbreviation_as_name",
    "signal_component",
)


@dataclass
class ExtractedSurface:
    """A raw surface span identified by the LLM extractor (or deterministic fallback).

    Not yet resolved — resolution happens in map_surfaces_v10().
    """
    text: str
    start: int
    end: int
    kind: str  # entity_surface | alias_surface
    confidence: float = 0.0
    rationale: str = ""  # one of SURFACE_RATIONALE_VALUES


@dataclass
class ExtractedSignal:
    """A signal pattern identified by the LLM extractor."""
    signal_type: str  # alias_equation | aka | identified_as | parenthetical | cryptonym_marker
    alias: str = ""
    entity_name: str = ""
    text: str = ""
    confidence: float = 0.0


@dataclass
class ExtractionContext:
    """Context passed to the LLM extractor for a chunk."""
    collection_slug: str
    document_id: int
    page_no: Optional[int] = None
    known_entities: List[str] = field(default_factory=list)
    known_aliases: List[str] = field(default_factory=list)  # includes backfilled aliases
    is_alias_scoped: bool = False
    blocked_alias_like: List[str] = field(default_factory=list)


@dataclass
class EnrichmentSummary:
    """Summary of enrichment performed after a search tool call."""
    chunks_extracted: int = 0
    chunks_llm_extracted: int = 0
    chunks_deterministic: int = 0
    new_contextual_mappings: List[Dict[str, Any]] = field(default_factory=list)
    new_general_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    ambiguities_opened: List[Dict[str, Any]] = field(default_factory=list)
    ambiguities_closed: List[Dict[str, Any]] = field(default_factory=list)
    new_signals: List[Dict[str, Any]] = field(default_factory=list)
    aliases_backfilled: Dict[int, List[str]] = field(default_factory=dict)
    recommended_boosts: Optional[Dict[str, Any]] = None
    # Timing (for verbose / CLI)
    batch_load_ms: Optional[float] = None
    extract_total_ms: Optional[float] = None
    llm_call_count: int = 0
    llm_concurrent: bool = False
    llm_latency_p50_ms: Optional[float] = None
    llm_latency_p95_ms: Optional[float] = None
