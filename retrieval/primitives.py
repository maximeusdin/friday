"""
Primitive Query IR (Intermediate Representation)

Primitive queries act as an intermediate representation between natural language
intent and executable retrieval. They are the only part of the plan that expresses
query semantics.

All primitives are explicit, deterministic, and reference concrete artifacts (IDs).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


# ============================================================================
# Primitive Types
# ============================================================================

@dataclass
class TermPrimitive:
    """TERM(value) - Single term for lexical/vector search"""
    type: Literal["TERM"] = "TERM"
    value: str = ""

    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("TERM primitive requires non-empty value")


@dataclass
class PhrasePrimitive:
    """PHRASE(value) - Multi-word phrase for exact matching"""
    type: Literal["PHRASE"] = "PHRASE"
    value: str = ""

    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("PHRASE primitive requires non-empty value")


@dataclass
class WithinResultSetPrimitive:
    """WITHIN_RESULT_SET(result_set_id) - Scope search to chunks in a result set"""
    type: Literal["WITHIN_RESULT_SET"] = "WITHIN_RESULT_SET"
    result_set_id: int = 0

    def __post_init__(self):
        if self.result_set_id <= 0:
            raise ValueError("WITHIN_RESULT_SET primitive requires positive result_set_id")


@dataclass
class ExcludeResultSetPrimitive:
    """EXCLUDE_RESULT_SET(result_set_id) - Exclude chunks from a result set"""
    type: Literal["EXCLUDE_RESULT_SET"] = "EXCLUDE_RESULT_SET"
    result_set_id: int = 0

    def __post_init__(self):
        if self.result_set_id <= 0:
            raise ValueError("EXCLUDE_RESULT_SET primitive requires positive result_set_id")


@dataclass
class FilterCollectionPrimitive:
    """FILTER_COLLECTION(slug) - Filter to specific collection"""
    type: Literal["FILTER_COLLECTION"] = "FILTER_COLLECTION"
    slug: str = ""

    def __post_init__(self):
        if not self.slug or not self.slug.strip():
            raise ValueError("FILTER_COLLECTION primitive requires non-empty slug")


@dataclass
class FilterDocumentPrimitive:
    """FILTER_DOCUMENT(document_id) - Filter to specific document"""
    type: Literal["FILTER_DOCUMENT"] = "FILTER_DOCUMENT"
    document_id: int = 0

    def __post_init__(self):
        if self.document_id <= 0:
            raise ValueError("FILTER_DOCUMENT primitive requires positive document_id")


@dataclass
class SetTopKPrimitive:
    """SET_TOP_K(int) - Set number of results to return"""
    type: Literal["SET_TOP_K"] = "SET_TOP_K"
    value: int = 20

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("SET_TOP_K primitive requires positive value")


@dataclass
class SetSearchTypePrimitive:
    """SET_SEARCH_TYPE(lex|vector|hybrid) - Set retrieval method"""
    type: Literal["SET_SEARCH_TYPE"] = "SET_SEARCH_TYPE"
    value: Literal["lex", "vector", "hybrid"] = "hybrid"


@dataclass
class ToggleConcordanceExpansionPrimitive:
    """TOGGLE_CONCORDANCE_EXPANSION(bool, source_slug?) - Enable/disable expansion"""
    type: Literal["TOGGLE_CONCORDANCE_EXPANSION"] = "TOGGLE_CONCORDANCE_EXPANSION"
    enabled: bool = True
    source_slug: Optional[str] = None


@dataclass
class SetTermModePrimitive:
    """SET_TERM_MODE(AND|OR) - Control how TERM primitives are combined"""
    type: Literal["SET_TERM_MODE"] = "SET_TERM_MODE"
    value: Literal["AND", "OR"] = "AND"


@dataclass
class OrGroupPrimitive:
    """OR_GROUP([...]) - Explicit OR grouping of primitives"""
    type: Literal["OR_GROUP"] = "OR_GROUP"
    primitives: List["Primitive"] = None  # Forward reference
    
    def __post_init__(self):
        if self.primitives is None:
            self.primitives = []
        if not self.primitives:
            raise ValueError("OR_GROUP primitive requires at least one primitive")


# ============================================================================
# Entity & Actor Primitives
# ============================================================================

@dataclass
class EntityPrimitive:
    """ENTITY(entity_id) - Reference to a specific entity (person, org, or place)"""
    type: Literal["ENTITY"] = "ENTITY"
    entity_id: int = 0

    def __post_init__(self):
        if self.entity_id <= 0:
            raise ValueError("ENTITY primitive requires positive entity_id")


# ============================================================================
# Co-Occurrence Primitives
# ============================================================================

@dataclass
class CoOccursWithPrimitive:
    """
    CO_OCCURS_WITH(entity_a, entity_b, window) - Find chunks where two entities co-occur
    
    Returns chunks containing evidence of both entities appearing together within
    the specified window. Unlike CO_LOCATED which is symmetric, CO_OCCURS_WITH 
    conceptually treats entity_a as the "primary" and entity_b as "must co-occur with".
    (In practice, the SQL is symmetric.)
    
    Window semantics:
    - "chunk" (default): BOTH entities must have mentions in the SAME chunk.
      This is the strictest match - direct co-mention within the same text block.
      SQL: EXISTS(em for entity_a in chunk) AND EXISTS(em for entity_b in chunk)
      
    - "document": Both entities must appear somewhere in the SAME document,
      but not necessarily in the same chunk. entity_a could be on page 1, entity_b on page 5.
      SQL: Complex join checking both entities have mentions in chunks of same document.
    
    Expected JSON format:
      {"type": "CO_OCCURS_WITH", "entity_a": 123, "entity_b": 456, "window": "document"}
    
    Never use single entity_id - always specify both entity_a and entity_b.
    
    Deduplication: Each qualifying chunk is returned ONCE, regardless of how many
    times each entity is mentioned within that chunk.
    """
    type: Literal["CO_OCCURS_WITH"] = "CO_OCCURS_WITH"
    entity_a: int = 0
    entity_b: int = 0
    window: Literal["chunk", "document"] = "chunk"

    def __post_init__(self):
        if self.entity_a <= 0:
            raise ValueError(
                "CO_OCCURS_WITH primitive requires positive entity_a. "
                "Expected format: {\"type\": \"CO_OCCURS_WITH\", \"entity_a\": 123, \"entity_b\": 456, \"window\": \"chunk\"}"
            )
        if self.entity_b <= 0:
            raise ValueError(
                "CO_OCCURS_WITH primitive requires positive entity_b. "
                "Expected format: {\"type\": \"CO_OCCURS_WITH\", \"entity_a\": 123, \"entity_b\": 456, \"window\": \"chunk\"}"
            )
        if self.entity_a == self.entity_b:
            raise ValueError("CO_OCCURS_WITH primitive requires different entities (entity_a != entity_b)")


# ============================================================================
# Temporal Primitives
# ============================================================================

@dataclass
class FilterDateRangePrimitive:
    """FILTER_DATE_RANGE(start?, end?) - Filter chunks by date range"""
    type: Literal["FILTER_DATE_RANGE"] = "FILTER_DATE_RANGE"
    start: Optional[str] = None  # ISO date string (YYYY-MM-DD)
    end: Optional[str] = None  # ISO date string (YYYY-MM-DD)

    def __post_init__(self):
        if self.start is None and self.end is None:
            raise ValueError("FILTER_DATE_RANGE primitive requires at least start or end")
        # Basic date format validation (YYYY-MM-DD)
        if self.start and not self._is_valid_date(self.start):
            raise ValueError(f"FILTER_DATE_RANGE start must be YYYY-MM-DD format, got: {self.start}")
        if self.end and not self._is_valid_date(self.end):
            raise ValueError(f"FILTER_DATE_RANGE end must be YYYY-MM-DD format, got: {self.end}")

    @staticmethod
    def _is_valid_date(date_str: str) -> bool:
        """Basic validation for YYYY-MM-DD format"""
        import re
        return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))


@dataclass
class IntersectDateWindowsPrimitive:
    """INTERSECT_DATE_WINDOWS(entity_a, entity_b) - Find overlapping date windows"""
    type: Literal["INTERSECT_DATE_WINDOWS"] = "INTERSECT_DATE_WINDOWS"
    entity_a: int = 0
    entity_b: int = 0

    def __post_init__(self):
        if self.entity_a <= 0:
            raise ValueError("INTERSECT_DATE_WINDOWS primitive requires positive entity_a")
        if self.entity_b <= 0:
            raise ValueError("INTERSECT_DATE_WINDOWS primitive requires positive entity_b")
        if self.entity_a == self.entity_b:
            raise ValueError("INTERSECT_DATE_WINDOWS primitive requires different entities")


# ============================================================================
# Geographic Primitives
# ============================================================================

@dataclass
class FilterCountryPrimitive:
    """FILTER_COUNTRY(country_code) - Filter by country code"""
    type: Literal["FILTER_COUNTRY"] = "FILTER_COUNTRY"
    value: str = ""  # ISO country code (e.g., "US", "GB")

    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("FILTER_COUNTRY primitive requires non-empty country code")
        # Basic validation: 2-3 character uppercase code
        if not self.value.isupper() or len(self.value) < 2 or len(self.value) > 3:
            raise ValueError(f"FILTER_COUNTRY value must be 2-3 character uppercase code, got: {self.value}")


@dataclass
class CoLocatedPrimitive:
    """
    CO_LOCATED(entity_a, entity_b, scope) - Find chunks where two entities co-occur
    
    Returns chunks that contain evidence of both entities appearing together,
    useful for relationship discovery and co-mention analysis.
    
    Scope semantics:
    - "chunk" (default): BOTH entity_a AND entity_b must have mentions in the SAME chunk.
      This is the strictest co-location - both entities named within the same text block.
      Typical use: Find direct evidence of two people being mentioned together.
      SQL: EXISTS(entity_mentions for entity_a in chunk) AND EXISTS(entity_mentions for entity_b in chunk)
      
    - "document": Both entities must appear somewhere in the SAME document,
      but not necessarily in the same chunk. entity_a could be on page 1, entity_b on page 5.
      Typical use: Find documents that discuss both entities (broader association).
      SQL: Complex join checking both entities have mentions in chunks of same document.
    
    NOT supported yet:
    - "page": Same physical page (would need page_num metadata)
    - "paragraph": Same paragraph (would need paragraph segmentation)
    - "window:N": Within N characters/tokens of each other
    
    Deduplication: Each qualifying chunk is returned ONCE, regardless of how many
    times each entity is mentioned within that chunk.
    
    Note: Order of entity_a/entity_b does not matter - CO_LOCATED(A,B) == CO_LOCATED(B,A)
    """
    type: Literal["CO_LOCATED"] = "CO_LOCATED"
    entity_a: int = 0
    entity_b: int = 0
    scope: Literal["chunk", "document"] = "chunk"

    def __post_init__(self):
        if self.entity_a <= 0:
            raise ValueError("CO_LOCATED primitive requires positive entity_a")
        if self.entity_b <= 0:
            raise ValueError("CO_LOCATED primitive requires positive entity_b")
        if self.entity_a == self.entity_b:
            raise ValueError("CO_LOCATED primitive requires different entities")


# ============================================================================
# Relationship Discovery Primitives
# ============================================================================

@dataclass
class RelationEvidencePrimitive:
    """RELATION_EVIDENCE(entity_a, entity_b) - Retrieve relationship evidence"""
    type: Literal["RELATION_EVIDENCE"] = "RELATION_EVIDENCE"
    entity_a: int = 0
    entity_b: int = 0

    def __post_init__(self):
        if self.entity_a <= 0:
            raise ValueError("RELATION_EVIDENCE primitive requires positive entity_a")
        if self.entity_b <= 0:
            raise ValueError("RELATION_EVIDENCE primitive requires positive entity_b")
        if self.entity_a == self.entity_b:
            raise ValueError("RELATION_EVIDENCE primitive requires different entities")


# ============================================================================
# Evidence Shaping Primitives
# ============================================================================

@dataclass
class RequireEvidencePrimitive:
    """REQUIRE_EVIDENCE(type) - Require specific evidence type in results"""
    type: Literal["REQUIRE_EVIDENCE"] = "REQUIRE_EVIDENCE"
    evidence_type: Literal["citation", "quote", "chunk"] = "chunk"


@dataclass
class GroupByPrimitive:
    """GROUP_BY(field) - Group results by metadata field"""
    type: Literal["GROUP_BY"] = "GROUP_BY"
    field: str = ""  # e.g., "country", "document_id", "collection_slug"

    def __post_init__(self):
        if not self.field or not self.field.strip():
            raise ValueError("GROUP_BY primitive requires non-empty field")


# ============================================================================
# Aggregation Primitives
# ============================================================================

@dataclass
class CountPrimitive:
    """
    COUNT(group_by?) - Count results without returning full result set.
    
    Used for queries like "how many mentions of X?" or "count documents by collection".
    When group_by is specified, returns counts grouped by that field.
    
    Examples:
        COUNT() - Total count of matching results
        COUNT(group_by="entity") - Count grouped by entity
        COUNT(group_by="document") - Count grouped by document
        COUNT(group_by="collection") - Count grouped by collection
    """
    type: Literal["COUNT"] = "COUNT"
    group_by: Optional[str] = None  # 'entity', 'document', 'collection', or None for total
    
    def __post_init__(self):
        valid_group_by = {None, "entity", "document", "collection", "date", "country"}
        if self.group_by is not None and self.group_by not in valid_group_by:
            raise ValueError(f"COUNT group_by must be one of {valid_group_by}, got: {self.group_by}")


@dataclass
class SetQueryModePrimitive:
    """
    SET_QUERY_MODE(mode) - Set the query execution mode.
    
    Modes:
        'retrieve' (default) - Return full result set with chunks
        'count' - Return only counts (faster, no chunk data)
        'aggregate' - Return aggregated statistics
    """
    type: Literal["SET_QUERY_MODE"] = "SET_QUERY_MODE"
    value: Literal["retrieve", "count", "aggregate"] = "retrieve"


# ============================================================================
# Speaker/Transcript Primitives
# ============================================================================

@dataclass
class SpeakerFilterPrimitive:
    """
    SPEAKER_FILTER(speakers, mode) - Filter chunks by speaker attribution.
    
    For transcript collections (e.g., McCarthy hearings), constrains retrieval
    to chunks where specific speakers appear.
    
    Examples:
        {"type": "SPEAKER_FILTER", "speakers": ["MR WELCH"], "mode": "include"}
        {"type": "SPEAKER_FILTER", "speakers": ["__STAGE__"], "mode": "exclude"}
    
    Implementation:
        SQL: primary_speaker_norm = ANY($1) OR speaker_norms && $1::text[]
    """
    type: Literal["SPEAKER_FILTER"] = "SPEAKER_FILTER"
    speakers: List[str] = None  # List of speaker_norm values
    mode: Literal["include", "exclude"] = "include"
    
    def __post_init__(self):
        if self.speakers is None:
            self.speakers = []
        if not self.speakers:
            raise ValueError("SPEAKER_FILTER primitive requires at least one speaker")
        # Normalize speaker names to uppercase
        self.speakers = [s.upper().strip() for s in self.speakers]


@dataclass
class SpeakerBoostPrimitive:
    """
    SPEAKER_BOOST(speakers, weight) - Boost score for chunks with target speakers.
    
    Unlike SPEAKER_FILTER, this doesn't exclude chunks—it reranks results
    to favor chunks where the target speakers appear.
    
    Examples:
        {"type": "SPEAKER_BOOST", "speakers": ["SENATOR MCCARTHY"], "weight": 1.5}
    
    Implementation:
        If speaker_norms overlaps with target speakers, multiply score by weight.
    """
    type: Literal["SPEAKER_BOOST"] = "SPEAKER_BOOST"
    speakers: List[str] = None  # List of speaker_norm values
    weight: float = 1.5  # Score multiplier
    
    def __post_init__(self):
        if self.speakers is None:
            self.speakers = []
        if not self.speakers:
            raise ValueError("SPEAKER_BOOST primitive requires at least one speaker")
        if self.weight <= 0:
            raise ValueError("SPEAKER_BOOST primitive requires positive weight")
        # Normalize speaker names to uppercase
        self.speakers = [s.upper().strip() for s in self.speakers]


@dataclass
class QuoteAttributionModePrimitive:
    """
    QUOTE_ATTRIBUTION_MODE(enabled, speaker?) - Control quote attribution in synthesis.
    
    When enabled, instructs the synthesis step to only quote spans where the
    target speaker is active. Requires chunk_turns or speaker_turn_spans data.
    
    Examples:
        {"type": "QUOTE_ATTRIBUTION_MODE", "enabled": true, "speaker": "MR WELCH"}
        {"type": "QUOTE_ATTRIBUTION_MODE", "enabled": true}  # Quote any speaker deterministically
    
    This is a synthesis directive, not a filter—it doesn't change retrieval
    but changes how quotes are extracted from retrieved chunks.
    """
    type: Literal["QUOTE_ATTRIBUTION_MODE"] = "QUOTE_ATTRIBUTION_MODE"
    enabled: bool = True
    speaker: Optional[str] = None  # Optional: restrict quotes to specific speaker
    
    def __post_init__(self):
        if self.speaker:
            self.speaker = self.speaker.upper().strip()


# Forward reference for recursive type (OR_GROUP contains primitives)
Primitive = Union[
    TermPrimitive,
    PhrasePrimitive,
    WithinResultSetPrimitive,
    ExcludeResultSetPrimitive,
    FilterCollectionPrimitive,
    FilterDocumentPrimitive,
    SetTopKPrimitive,
    SetSearchTypePrimitive,
    ToggleConcordanceExpansionPrimitive,
    SetTermModePrimitive,
    OrGroupPrimitive,
    EntityPrimitive,
    CoOccursWithPrimitive,
    FilterDateRangePrimitive,
    IntersectDateWindowsPrimitive,
    FilterCountryPrimitive,
    CoLocatedPrimitive,
    RelationEvidencePrimitive,
    RequireEvidencePrimitive,
    GroupByPrimitive,
    # Aggregation primitives
    CountPrimitive,
    SetQueryModePrimitive,
    # Speaker/Transcript primitives
    SpeakerFilterPrimitive,
    SpeakerBoostPrimitive,
    QuoteAttributionModePrimitive,
]

# Note: OrGroupPrimitive uses forward reference (string literal) for recursive type


# ============================================================================
# Plan JSON Structure
# ============================================================================

@dataclass
class QueryPlan:
    """Query section of a research plan"""
    raw: str  # Original user utterance
    primitives: List[Primitive]  # Explicit primitive representation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "raw": self.raw,
            "primitives": [self._primitive_to_dict(p) for p in self.primitives]
        }

    @staticmethod
    def _primitive_to_dict(p: Primitive) -> Dict[str, Any]:
        """Convert primitive to dictionary"""
        if isinstance(p, TermPrimitive):
            return {"type": "TERM", "value": p.value}
        elif isinstance(p, PhrasePrimitive):
            return {"type": "PHRASE", "value": p.value}
        elif isinstance(p, WithinResultSetPrimitive):
            return {"type": "WITHIN_RESULT_SET", "result_set_id": p.result_set_id}
        elif isinstance(p, ExcludeResultSetPrimitive):
            return {"type": "EXCLUDE_RESULT_SET", "result_set_id": p.result_set_id}
        elif isinstance(p, FilterCollectionPrimitive):
            return {"type": "FILTER_COLLECTION", "slug": p.slug}
        elif isinstance(p, FilterDocumentPrimitive):
            return {"type": "FILTER_DOCUMENT", "document_id": p.document_id}
        elif isinstance(p, SetTopKPrimitive):
            return {"type": "SET_TOP_K", "value": p.value}
        elif isinstance(p, SetSearchTypePrimitive):
            return {"type": "SET_SEARCH_TYPE", "value": p.value}
        elif isinstance(p, ToggleConcordanceExpansionPrimitive):
            result = {"type": "TOGGLE_CONCORDANCE_EXPANSION", "enabled": p.enabled}
            if p.source_slug:
                result["source_slug"] = p.source_slug
            return result
        elif isinstance(p, SetTermModePrimitive):
            return {"type": "SET_TERM_MODE", "value": p.value}
        elif isinstance(p, OrGroupPrimitive):
            return {
                "type": "OR_GROUP",
                "primitives": [QueryPlan._primitive_to_dict(sub_p) for sub_p in p.primitives]
            }
        elif isinstance(p, EntityPrimitive):
            return {"type": "ENTITY", "entity_id": p.entity_id}
        elif isinstance(p, CoOccursWithPrimitive):
            return {"type": "CO_OCCURS_WITH", "entity_a": p.entity_a, "entity_b": p.entity_b, "window": p.window}
        elif isinstance(p, FilterDateRangePrimitive):
            result = {"type": "FILTER_DATE_RANGE"}
            if p.start:
                result["start"] = p.start
            if p.end:
                result["end"] = p.end
            return result
        elif isinstance(p, IntersectDateWindowsPrimitive):
            return {"type": "INTERSECT_DATE_WINDOWS", "entity_a": p.entity_a, "entity_b": p.entity_b}
        elif isinstance(p, FilterCountryPrimitive):
            return {"type": "FILTER_COUNTRY", "value": p.value}
        elif isinstance(p, CoLocatedPrimitive):
            return {"type": "CO_LOCATED", "entity_a": p.entity_a, "entity_b": p.entity_b, "scope": p.scope}
        elif isinstance(p, RelationEvidencePrimitive):
            return {"type": "RELATION_EVIDENCE", "entity_a": p.entity_a, "entity_b": p.entity_b}
        elif isinstance(p, RequireEvidencePrimitive):
            return {"type": "REQUIRE_EVIDENCE", "evidence_type": p.evidence_type}
        elif isinstance(p, GroupByPrimitive):
            return {"type": "GROUP_BY", "field": p.field}
        # Speaker/Transcript primitives
        elif isinstance(p, SpeakerFilterPrimitive):
            return {"type": "SPEAKER_FILTER", "speakers": p.speakers, "mode": p.mode}
        elif isinstance(p, SpeakerBoostPrimitive):
            return {"type": "SPEAKER_BOOST", "speakers": p.speakers, "weight": p.weight}
        elif isinstance(p, QuoteAttributionModePrimitive):
            result = {"type": "QUOTE_ATTRIBUTION_MODE", "enabled": p.enabled}
            if p.speaker:
                result["speaker"] = p.speaker
            return result
        # Aggregation primitives
        elif isinstance(p, CountPrimitive):
            result = {"type": "COUNT"}
            if p.group_by:
                result["group_by"] = p.group_by
            return result
        elif isinstance(p, SetQueryModePrimitive):
            return {"type": "SET_QUERY_MODE", "value": p.value}
        else:
            raise ValueError(f"Unknown primitive type: {type(p)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryPlan:
        """Parse from dictionary"""
        raw = data.get("raw", "")
        primitives_data = data.get("primitives", [])
        primitives = [cls._dict_to_primitive(p) for p in primitives_data]
        return cls(raw=raw, primitives=primitives)

    @staticmethod
    def _dict_to_primitive(data: Dict[str, Any]) -> Primitive:
        """Convert dictionary to primitive"""
        ptype = data.get("type")
        if ptype == "TERM":
            return TermPrimitive(value=data["value"])
        elif ptype == "PHRASE":
            return PhrasePrimitive(value=data["value"])
        elif ptype == "WITHIN_RESULT_SET":
            return WithinResultSetPrimitive(result_set_id=data["result_set_id"])
        elif ptype == "EXCLUDE_RESULT_SET":
            return ExcludeResultSetPrimitive(result_set_id=data["result_set_id"])
        elif ptype == "FILTER_COLLECTION":
            return FilterCollectionPrimitive(slug=data["slug"])
        elif ptype == "FILTER_DOCUMENT":
            return FilterDocumentPrimitive(document_id=data["document_id"])
        elif ptype == "SET_TOP_K":
            return SetTopKPrimitive(value=data["value"])
        elif ptype == "SET_SEARCH_TYPE":
            return SetSearchTypePrimitive(value=data["value"])
        elif ptype == "TOGGLE_CONCORDANCE_EXPANSION":
            return ToggleConcordanceExpansionPrimitive(
                enabled=data.get("enabled", True),
                source_slug=data.get("source_slug")
            )
        elif ptype == "SET_TERM_MODE":
            return SetTermModePrimitive(value=data["value"])
        elif ptype == "OR_GROUP":
            sub_primitives = [QueryPlan._dict_to_primitive(sub_p) for sub_p in data["primitives"]]
            return OrGroupPrimitive(primitives=sub_primitives)
        elif ptype == "ENTITY":
            return EntityPrimitive(entity_id=data["entity_id"])
        elif ptype == "CO_OCCURS_WITH":
            # New format: entity_a + entity_b
            if "entity_a" in data and "entity_b" in data:
                return CoOccursWithPrimitive(
                    entity_a=data["entity_a"],
                    entity_b=data["entity_b"],
                    window=data.get("window", "chunk")
                )
            # Backward compatibility: entity_id + other_entity_id (legacy format)
            elif "entity_id" in data and "other_entity_id" in data:
                return CoOccursWithPrimitive(
                    entity_a=data["entity_id"],
                    entity_b=data["other_entity_id"],
                    window=data.get("window", "chunk")
                )
            # Error: single entity_id is no longer valid
            elif "entity_id" in data:
                raise ValueError(
                    f"CO_OCCURS_WITH requires two entities (entity_a and entity_b). "
                    f"Got only entity_id={data['entity_id']}. "
                    f"Expected format: {{\"type\": \"CO_OCCURS_WITH\", \"entity_a\": 123, \"entity_b\": 456, \"window\": \"chunk\"}}"
                )
            else:
                raise ValueError(
                    f"CO_OCCURS_WITH missing required fields. "
                    f"Expected format: {{\"type\": \"CO_OCCURS_WITH\", \"entity_a\": 123, \"entity_b\": 456, \"window\": \"chunk\"}}"
                )
        elif ptype == "FILTER_DATE_RANGE":
            return FilterDateRangePrimitive(
                start=data.get("start"),
                end=data.get("end")
            )
        elif ptype == "INTERSECT_DATE_WINDOWS":
            return IntersectDateWindowsPrimitive(
                entity_a=data["entity_a"],
                entity_b=data["entity_b"]
            )
        elif ptype == "FILTER_COUNTRY":
            return FilterCountryPrimitive(value=data["value"])
        elif ptype == "CO_LOCATED":
            return CoLocatedPrimitive(
                entity_a=data["entity_a"],
                entity_b=data["entity_b"],
                scope=data.get("scope", "chunk")
            )
        elif ptype == "RELATION_EVIDENCE":
            return RelationEvidencePrimitive(
                entity_a=data["entity_a"],
                entity_b=data["entity_b"]
            )
        elif ptype == "REQUIRE_EVIDENCE":
            return RequireEvidencePrimitive(evidence_type=data.get("evidence_type", "chunk"))
        elif ptype == "GROUP_BY":
            return GroupByPrimitive(field=data["field"])
        # Aggregation primitives
        elif ptype == "COUNT":
            return CountPrimitive(group_by=data.get("group_by"))
        elif ptype == "SET_QUERY_MODE":
            return SetQueryModePrimitive(value=data.get("value", "retrieve"))
        # Speaker/Transcript primitives
        elif ptype == "SPEAKER_FILTER":
            return SpeakerFilterPrimitive(
                speakers=data["speakers"],
                mode=data.get("mode", "include")
            )
        elif ptype == "SPEAKER_BOOST":
            return SpeakerBoostPrimitive(
                speakers=data["speakers"],
                weight=data.get("weight", 1.5)
            )
        elif ptype == "QUOTE_ATTRIBUTION_MODE":
            return QuoteAttributionModePrimitive(
                enabled=data.get("enabled", True),
                speaker=data.get("speaker")
            )
        else:
            raise ValueError(f"Unknown primitive type: {ptype}")


@dataclass
class ResearchPlan:
    """Complete research plan structure"""
    query: QueryPlan
    compiled: Optional[Dict[str, Any]] = None  # Compiled query components
    # Execution envelope (self-contained parameters)
    execution_envelope: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plan_json format"""
        result = {
            "query": self.query.to_dict()
        }
        if self.compiled:
            result["compiled"] = self.compiled
        if self.execution_envelope:
            result["execution_envelope"] = self.execution_envelope
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResearchPlan:
        """Parse from plan_json"""
        query_data = data.get("query", {})
        query = QueryPlan.from_dict(query_data)
        compiled = data.get("compiled")
        execution_envelope = data.get("execution_envelope")
        return cls(query=query, compiled=compiled, execution_envelope=execution_envelope)
    
    def compile(self) -> None:
        """
        Compile primitives to executable query components.
        Updates self.compiled with deterministic compilation results.
        """
        self.compiled = compile_primitives(self.query.primitives)


# ============================================================================
# Plan Hash (Deterministic)
# ============================================================================

def compute_plan_hash(plan_json: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of plan_json.
    
    Uses sorted JSON serialization to ensure consistent hashing regardless of
    key order or whitespace.
    """
    # Sort keys and use compact JSON for deterministic hashing
    json_str = json.dumps(plan_json, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]  # 16-char hash


# ============================================================================
# Validation
# ============================================================================

def validate_plan_json(plan_json: Dict[str, Any]) -> List[str]:
    """
    Validate plan_json structure and primitives.
    Returns list of error messages (empty if valid).
    """
    errors: List[str] = []
    
    # Check top-level structure
    if "query" not in plan_json:
        errors.append("plan_json missing 'query' section")
        return errors
    
    query_data = plan_json["query"]
    
    # Check query.raw
    if "raw" not in query_data:
        errors.append("query section missing 'raw' field")
    elif not query_data["raw"] or not str(query_data["raw"]).strip():
        errors.append("query.raw must be non-empty")
    
    # Check query.primitives
    if "primitives" not in query_data:
        errors.append("query section missing 'primitives' field")
    elif not isinstance(query_data["primitives"], list):
        errors.append("query.primitives must be a list")
    else:
        # Validate each primitive
        for i, p in enumerate(query_data["primitives"]):
            if not isinstance(p, dict):
                errors.append(f"primitive[{i}] must be a dictionary")
                continue
            
            ptype = p.get("type")
            if not ptype:
                errors.append(f"primitive[{i}] missing 'type' field")
                continue
            
            # Validate primitive-specific fields
            try:
                QueryPlan._dict_to_primitive(p)
            except (ValueError, KeyError) as e:
                errors.append(f"primitive[{i}] ({ptype}): {str(e)}")
    
    return errors


# ============================================================================
# Normalization (Canonical)
# ============================================================================

def normalize_term(value: str) -> str:
    """
    Canonical normalization for terms: stable casing, punctuation stripping.
    
    Rules:
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse internal whitespace
    - Remove punctuation (keep alphanumeric and spaces)
    
    Ensures "Treasury", "treasury,,", "TREASURY" all normalize to "treasury"
    """
    if not value:
        return ""
    
    # Lowercase
    normalized = value.lower()
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    # Remove punctuation (keep alphanumeric and spaces)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def normalize_phrase(value: str) -> List[str]:
    """
    Canonical normalization for phrases: returns list of normalized words.
    
    Same normalization as normalize_term, but returns word list for phrase handling.
    """
    normalized = normalize_term(value)
    if not normalized:
        return []
    
    # Split into words and filter empty
    words = [w for w in normalized.split() if w]
    return words


# ============================================================================
# Deterministic Compilation
# ============================================================================

def _canonical_order_primitives(primitives: List[Primitive]) -> List[Primitive]:
    """
    Canonically order primitives for deterministic compilation.
    
    Order by: (type, value/id, ...)
    This ensures same primitives in different order produce same compiled output.
    """
    def primitive_key(p: Primitive) -> Tuple[str, Any]:
        """
        Generate sort key for primitive.
        
        Ordering rules:
        - Type first (alphabetical)
        - For ID-bearing primitives: (type, id)
        - For value-bearing primitives: (type, normalized_value)
        - For booleans/controls: (type, stringified_value)
        - For tuples: (type, tuple_of_values)
        """
        if isinstance(p, TermPrimitive):
            return ("TERM", normalize_term(p.value))
        elif isinstance(p, PhrasePrimitive):
            return ("PHRASE", normalize_term(p.value))  # Use normalized for stable ordering
        elif isinstance(p, WithinResultSetPrimitive):
            return ("WITHIN_RESULT_SET", p.result_set_id)
        elif isinstance(p, ExcludeResultSetPrimitive):
            return ("EXCLUDE_RESULT_SET", p.result_set_id)
        elif isinstance(p, FilterCollectionPrimitive):
            return ("FILTER_COLLECTION", p.slug.lower())  # Normalize for stable ordering
        elif isinstance(p, FilterDocumentPrimitive):
            return ("FILTER_DOCUMENT", p.document_id)
        elif isinstance(p, SetTopKPrimitive):
            return ("SET_TOP_K", p.value)
        elif isinstance(p, SetSearchTypePrimitive):
            return ("SET_SEARCH_TYPE", p.value)
        elif isinstance(p, ToggleConcordanceExpansionPrimitive):
            # Stable ordering: (enabled, source_slug)
            return ("TOGGLE_CONCORDANCE_EXPANSION", (p.enabled, (p.source_slug or "").lower()))
        elif isinstance(p, SetTermModePrimitive):
            return ("SET_TERM_MODE", p.value)
        elif isinstance(p, OrGroupPrimitive):
            # Sort sub-primitives recursively, then use as key
            sorted_sub = _canonical_order_primitives(p.primitives)
            return ("OR_GROUP", tuple(primitive_key(sp) for sp in sorted_sub))
        elif isinstance(p, EntityPrimitive):
            return ("ENTITY", p.entity_id)
        elif isinstance(p, CoOccursWithPrimitive):
            # Sort entity IDs for stable ordering (order shouldn't matter)
            return ("CO_OCCURS_WITH", (min(p.entity_a, p.entity_b), max(p.entity_a, p.entity_b), p.window))
        elif isinstance(p, FilterDateRangePrimitive):
            # Use normalized dates for stable ordering
            return ("FILTER_DATE_RANGE", (p.start or "", p.end or ""))
        elif isinstance(p, IntersectDateWindowsPrimitive):
            # Order entity IDs consistently (smaller first)
            entity_a, entity_b = (p.entity_a, p.entity_b) if p.entity_a < p.entity_b else (p.entity_b, p.entity_a)
            return ("INTERSECT_DATE_WINDOWS", (entity_a, entity_b))
        elif isinstance(p, FilterCountryPrimitive):
            return ("FILTER_COUNTRY", p.value.upper())  # Normalize to uppercase
        elif isinstance(p, CoLocatedPrimitive):
            # Order entity IDs consistently (smaller first)
            entity_a, entity_b = (p.entity_a, p.entity_b) if p.entity_a < p.entity_b else (p.entity_b, p.entity_a)
            return ("CO_LOCATED", (entity_a, entity_b, p.scope))
        elif isinstance(p, RelationEvidencePrimitive):
            # Order entity IDs consistently (smaller first)
            entity_a, entity_b = (p.entity_a, p.entity_b) if p.entity_a < p.entity_b else (p.entity_b, p.entity_a)
            return ("RELATION_EVIDENCE", (entity_a, entity_b))
        elif isinstance(p, RequireEvidencePrimitive):
            return ("REQUIRE_EVIDENCE", p.evidence_type)
        elif isinstance(p, GroupByPrimitive):
            return ("GROUP_BY", p.field.lower())  # Normalize for stable ordering
        else:
            return (type(p).__name__, str(p))
    
    return sorted(primitives, key=primitive_key)


def compile_primitives_to_tsquery_text(primitives: List[Primitive]) -> str:
    """
    Compile primitives to websearch-style query text for websearch_to_tsquery.
    
    Rules for websearch_to_tsquery:
    - AND: whitespace (e.g., "hiss treasury")
    - OR: use "OR" keyword (e.g., "hiss OR chambers")
    - Phrases: use quotes (e.g., '"soviet intelligence"')
    - NOT: use - prefix (e.g., "-excluded")
    
    Returns plain text string suitable for websearch_to_tsquery('simple', %s).
    """
    # Canonically order primitives for determinism
    ordered_primitives = _canonical_order_primitives(primitives)
    
    # Track term mode (default: AND)
    term_mode = "AND"
    
    # Collect query parts as websearch-style text
    query_parts: List[str] = []
    current_terms: List[str] = []
    
    for p in ordered_primitives:
        if isinstance(p, SetTermModePrimitive):
            # Apply term mode to accumulated terms, then switch mode
            if current_terms:
                if term_mode == "OR":
                    query_parts.append(" OR ".join(current_terms))
                else:
                    query_parts.extend(current_terms)
                current_terms = []
            term_mode = p.value
        elif isinstance(p, TermPrimitive):
            # Normalize term
            normalized = normalize_term(p.value)
            if normalized:
                current_terms.append(normalized)
        elif isinstance(p, PhrasePrimitive):
            # Flush accumulated terms first
            if current_terms:
                if term_mode == "OR":
                    query_parts.append(" OR ".join(current_terms))
                else:
                    query_parts.extend(current_terms)
                current_terms = []
            
            # Normalize phrase and quote it
            words = normalize_phrase(p.value)
            if words:
                # Join words and quote for phrase matching
                phrase = " ".join(words)
                # Escape embedded double quotes in phrase (for websearch syntax)
                # Note: apostrophes are fine as-is (parameterization handles SQL escaping)
                phrase_escaped = phrase.replace('"', '\\"')
                query_parts.append(f'"{phrase_escaped}"')
        elif isinstance(p, OrGroupPrimitive):
            # Flush accumulated terms first
            if current_terms:
                if term_mode == "OR":
                    query_parts.append(" OR ".join(current_terms))
                else:
                    query_parts.extend(current_terms)
                current_terms = []
            
            # Compile OR_GROUP sub-primitives
            or_parts: List[str] = []
            for sub_p in p.primitives:
                if isinstance(sub_p, TermPrimitive):
                    normalized = normalize_term(sub_p.value)
                    if normalized:
                        or_parts.append(normalized)
                elif isinstance(sub_p, PhrasePrimitive):
                    words = normalize_phrase(sub_p.value)
                    if words:
                        phrase = " ".join(words)
                        # Escape embedded double quotes (apostrophes are fine)
                        phrase_escaped = phrase.replace('"', '\\"')
                        or_parts.append(f'"{phrase_escaped}"')
            
            if or_parts:
                if len(or_parts) == 1:
                    query_parts.append(or_parts[0])
                else:
                    query_parts.append(" OR ".join(or_parts))
    
    # Flush any remaining terms
    if current_terms:
        if term_mode == "OR":
            query_parts.append(" OR ".join(current_terms))
        else:
            query_parts.extend(current_terms)
    
    if not query_parts:
        return ""  # Empty string for websearch_to_tsquery
    
    # Join with spaces (AND semantics in websearch_to_tsquery)
    return " ".join(query_parts)


def compile_primitives_to_tsquery(primitives: List[Primitive]) -> Tuple[str, Tuple[str, List[str]]]:
    """
    Compile primitives to PostgreSQL tsquery using websearch_to_tsquery for safe quoting.
    
    This function generates websearch-style query text and wraps it in websearch_to_tsquery().
    The query text is generated by compile_primitives_to_tsquery_text().
    
    Returns tuple: (tsquery_text_for_debug, (sql_template, params))
    - tsquery_text_for_debug: human-readable query text (for export/debugging)
    - sql_template: SQL template like "websearch_to_tsquery('simple', %s)"
    - params: List with single parameter (the query text)
    """
    query_text = compile_primitives_to_tsquery_text(primitives)
    
    if not query_text:
        # No-match sentinel
        tsquery_text = "___nomatch___"
        sql_template = "to_tsquery('simple', %s)"
        params = ["___nomatch___"]
        return (tsquery_text, (sql_template, params))
    
    # Return parameterized form using websearch_to_tsquery
    sql_template = "websearch_to_tsquery('simple', %s)"
    params = [query_text]
    
    return (query_text, (sql_template, params))


def _escape_tsquery_term(term: str) -> str:
    """
    Escape a single normalized term for tsquery. Handles special characters.
    
    Assumes term is already normalized (lowercase, punctuation stripped).
    """
    if not term:
        return ""
    
    # Escape special tsquery characters by replacing with space
    # Characters: & | ! ( ) : * '
    term = term.replace("&", " ").replace("|", " ").replace("!", " ")
    term = term.replace("(", " ").replace(")", " ").replace(":", " ")
    term = term.replace("*", " ").replace("'", " ")
    
    # Collapse whitespace
    term = re.sub(r'\s+', ' ', term).strip()
    
    if not term:
        return ""
    
    # If multiple words, wrap in parentheses and AND them
    words = term.split()
    if len(words) == 1:
        return words[0]
    else:
        return f"({' & '.join(words)})"


def _escape_tsquery_phrase_adjacency(words: List[str]) -> str:
    """
    Convert normalized phrase words to tsquery adjacency query using <-> operator.
    
    Example: ["soviet", "intelligence"] → "soviet <-> intelligence"
    
    This provides phrase-like behavior (words must be adjacent).
    """
    if not words:
        return ""
    
    # Escape each word for tsquery
    escaped_words = []
    for word in words:
        # Escape special characters
        word_escaped = word.replace("&", "").replace("|", "").replace("!", "")
        word_escaped = word_escaped.replace("(", "").replace(")", "").replace(":", "")
        word_escaped = word_escaped.replace("*", "").replace("'", "")
        if word_escaped:
            escaped_words.append(word_escaped)
    
    if not escaped_words:
        return ""
    
    if len(escaped_words) == 1:
        return escaped_words[0]
    
    # Join with <-> operator for adjacency
    return " <-> ".join(escaped_words)


def compile_primitives_to_expanded(primitives: List[Primitive]) -> Dict[str, Any]:
    """
    Compile primitives to structured expanded query format.
    
    Returns dictionary with:
    - base_text: normalized base query text (before expansion, websearch syntax)
    - expanded_text: text after expansion (same as base_text if no expansion, websearch syntax)
    - expansions: list of expansion records (empty if no expansion)
    
    For Day 9: expansions are empty, but structure is ready for future expansion.
    
    Note: 
    - Uses websearch syntax: whitespace for AND, "OR" keyword for OR, quotes for phrases
    - This format is used by both vector search and tsquery (single source of truth)
    - SET_TERM_MODE: affects how TERMs are combined (AND vs OR)
    - OR_GROUP: compiled with "OR" keyword
    - PHRASE: quoted for proper phrase matching
    """
    # Use the same compilation logic as tsquery_text (websearch syntax)
    # This ensures expanded_text and tsquery params are aligned
    base_text = compile_primitives_to_tsquery_text(primitives)
    
    # For now, no expansion - expanded_text = base_text
    # Future: apply concordance expansion, synonyms, etc.
    # When expansion is added, expanded_text will be base_text + expansions
    expanded_text = base_text  # No expansion yet
    
    return {
        "base_text": base_text,
        "expanded_text": expanded_text,
        "expansions": [],  # Empty for Day 9
    }


def compile_primitives_to_scope(primitives: List[Primitive], chunk_id_expr: str = "c.id") -> Tuple[str, List[Any], Dict[str, bool]]:
    """
    Compile primitives to SQL WHERE constraints with parameterized queries.
    
    Returns tuple: (sql_fragment, params, required_joins)
    - sql_fragment: SQL fragment with %s placeholders
    - params: List of parameter values in order
    - required_joins: Dict with join information needed for execution
    
    Semantics:
    - WITHIN_RESULT_SET(id): chunks must be in result_set.chunk_ids array
      SQL: {chunk_id_expr} = ANY((SELECT chunk_ids FROM result_sets WHERE id = %s))
    - EXCLUDE_RESULT_SET(id): chunks must NOT be in result_set.chunk_ids array
      SQL: {chunk_id_expr} != ALL((SELECT chunk_ids FROM result_sets WHERE id = %s))
    - FILTER_COLLECTION(slug): filter by collection_slug via chunk_metadata
      SQL: cm.collection_slug = %s
    - FILTER_DOCUMENT(id): filter by document_id via chunk_metadata
      SQL: cm.document_id = %s
    - FILTER_DATE_RANGE(start, end): filter by date range via date_mentions or chunk_metadata
      SQL: EXISTS (SELECT 1 FROM date_mentions WHERE chunk_id = c.id AND ...) 
           OR fallback to cm.date_max >= %s AND cm.date_min <= %s
    - ENTITY(entity_id): filter chunks containing entity mentions
      SQL: EXISTS (SELECT 1 FROM entity_mentions WHERE chunk_id = c.id AND entity_id = %s)
    
    Returns ("", [], {}) if no scope constraints.
    """
    conditions: List[str] = []
    params: List[Any] = []
    required_joins: Dict[str, bool] = {
        "chunk_metadata": False,
        "entity_mentions": False,
        "date_mentions": False,
    }
    
    # Collect all WITHIN_RESULT_SET primitives to OR them together
    within_result_sets: List[int] = []
    
    # Collect ENTITY primitives to handle together
    entity_ids: List[int] = []
    
    for p in primitives:
        if isinstance(p, WithinResultSetPrimitive):
            within_result_sets.append(p.result_set_id)
        elif isinstance(p, EntityPrimitive):
            entity_ids.append(p.entity_id)
    
    # Handle multiple WITHIN_RESULT_SET primitives by OR-ing them together
    if within_result_sets:
        if len(within_result_sets) == 1:
            # Single result set - simple condition
            conditions.append(
                f"{chunk_id_expr} = ANY((SELECT chunk_ids FROM result_sets WHERE id = %s))"
            )
            params.append(within_result_sets[0])
        else:
            # Multiple result sets - OR them together
            or_conditions = []
            for rs_id in within_result_sets:
                or_conditions.append(
                    f"{chunk_id_expr} = ANY((SELECT chunk_ids FROM result_sets WHERE id = %s))"
                )
                params.append(rs_id)
            conditions.append(f"({' OR '.join(or_conditions)})")
    
    # Handle ENTITY primitives - chunks must contain mentions of these entities
    if entity_ids:
        if len(entity_ids) == 1:
            # Single entity - simple EXISTS check
            conditions.append(
                f"EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = {chunk_id_expr} AND em.entity_id = %s)"
            )
            params.append(entity_ids[0])
            required_joins["entity_mentions"] = True
        else:
            # Multiple entities - chunks must contain mentions of ANY of them (OR)
            or_conditions = []
            for entity_id in entity_ids:
                or_conditions.append(
                    f"EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = {chunk_id_expr} AND em.entity_id = %s)"
                )
                params.append(entity_id)
            conditions.append(f"({' OR '.join(or_conditions)})")
            required_joins["entity_mentions"] = True
    
    # Process other scope primitives
    for p in primitives:
        if isinstance(p, WithinResultSetPrimitive) or isinstance(p, EntityPrimitive):
            # Already handled above
            continue
        elif isinstance(p, ExcludeResultSetPrimitive):
            # Exclude chunks in result set
            conditions.append(
                f"{chunk_id_expr} != ALL((SELECT chunk_ids FROM result_sets WHERE id = %s))"
            )
            params.append(p.result_set_id)
        elif isinstance(p, FilterCollectionPrimitive):
            # Filter by collection slug (via chunk_metadata)
            conditions.append("cm.collection_slug = %s")
            params.append(p.slug)
            required_joins["chunk_metadata"] = True
        elif isinstance(p, FilterDocumentPrimitive):
            # Filter by document_id (via chunk_metadata)
            conditions.append("cm.document_id = %s")
            params.append(p.document_id)
            required_joins["chunk_metadata"] = True
        elif isinstance(p, FilterDateRangePrimitive):
            # Filter by date range - prefer date_mentions table, fallback to chunk_metadata
            date_conditions: List[str] = []
            date_params: List[Any] = []
            
            # Build date_mentions EXISTS condition (more precise)
            dm_conditions = []
            if p.start:
                dm_conditions.append("dm.date_end >= %s")
                date_params.append(p.start)
            if p.end:
                dm_conditions.append("dm.date_start <= %s")
                date_params.append(p.end)
            
            if dm_conditions:
                # Use date_mentions for precise date filtering
                dm_sql = f"EXISTS (SELECT 1 FROM date_mentions dm WHERE dm.chunk_id = {chunk_id_expr} AND {' AND '.join(dm_conditions)})"
                date_conditions.append(dm_sql)
                params.extend(date_params)
                required_joins["date_mentions"] = True
                
                # Also add fallback to chunk_metadata for chunks without explicit date mentions
                # but with date metadata (broader recall)
                cm_conditions = []
                cm_params = []
                if p.start:
                    cm_conditions.append("cm.date_max >= %s")
                    cm_params.append(p.start)
                if p.end:
                    cm_conditions.append("cm.date_min <= %s")
                    cm_params.append(p.end)
                if cm_conditions:
                    cm_sql = f"({' AND '.join(cm_conditions)})"
                    date_conditions.append(cm_sql)
                    params.extend(cm_params)
                    required_joins["chunk_metadata"] = True
                
                # Combine: chunk has date mention OR chunk metadata matches
                if len(date_conditions) > 1:
                    conditions.append(f"({' OR '.join(date_conditions)})")
                else:
                    conditions.append(date_conditions[0])
        elif isinstance(p, FilterCountryPrimitive):
            # Filter by country (would need country metadata in chunk_metadata)
            # For now, placeholder - would need to add country field to chunk_metadata
            # conditions.append("cm.country_code = %s")
            # params.append(p.value)
            pass  # Not implemented yet - requires schema addition
        elif isinstance(p, CoOccursWithPrimitive):
            # CO_OCCURS_WITH: Find chunks where two entities co-occur
            # Both entity_a and entity_b must have mentions within the specified window
            # The window parameter controls scope:
            # - "chunk": both entities mentioned in same chunk (strictest)
            # - "document": both entities mentioned somewhere in same document (broader)
            if p.window == "chunk":
                # Both entities mentioned in same chunk
                conditions.append(
                    f"""(
                        EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = {chunk_id_expr} AND em.entity_id = %s)
                        AND EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = {chunk_id_expr} AND em.entity_id = %s)
                    )"""
                )
                params.append(p.entity_a)
                params.append(p.entity_b)
                required_joins["entity_mentions"] = True
            elif p.window == "document":
                # Both entities mentioned somewhere in same document
                # Uses entity_mentions.document_id (denormalized) for performance
                conditions.append(
                    f"""EXISTS (
                        SELECT 1 FROM entity_mentions em_a
                        JOIN entity_mentions em_b ON em_b.document_id = em_a.document_id AND em_b.entity_id = %s
                        WHERE em_a.entity_id = %s
                        AND em_a.document_id = (SELECT cm.document_id FROM chunk_metadata cm WHERE cm.chunk_id = {chunk_id_expr} LIMIT 1)
                    )"""
                )
                params.append(p.entity_b)
                params.append(p.entity_a)
                required_joins["entity_mentions"] = True
                required_joins["chunk_metadata"] = True
        elif isinstance(p, CoLocatedPrimitive):
            # CO_LOCATED: Find chunks where two entities appear together
            # Both entity_a and entity_b must have mentions in the specified scope
            if p.scope == "chunk":
                # Both entities mentioned in same chunk
                conditions.append(
                    f"""(
                        EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = {chunk_id_expr} AND em.entity_id = %s)
                        AND EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = {chunk_id_expr} AND em.entity_id = %s)
                    )"""
                )
                params.append(p.entity_a)
                params.append(p.entity_b)
                required_joins["entity_mentions"] = True
            elif p.scope == "document":
                # Both entities mentioned somewhere in same document
                # Uses entity_mentions.document_id (denormalized) for performance
                # Checks that both entities have mentions in the same document as the target chunk
                conditions.append(
                    f"""EXISTS (
                        SELECT 1 FROM entity_mentions em_a
                        JOIN entity_mentions em_b ON em_b.document_id = em_a.document_id AND em_b.entity_id = %s
                        WHERE em_a.entity_id = %s
                        AND em_a.document_id = (SELECT cm.document_id FROM chunk_metadata cm WHERE cm.chunk_id = {chunk_id_expr} LIMIT 1)
                    )"""
                )
                params.append(p.entity_b)
                params.append(p.entity_a)
                required_joins["entity_mentions"] = True
                required_joins["chunk_metadata"] = True
        elif isinstance(p, RelationEvidencePrimitive):
            # RELATION_EVIDENCE: Find chunks that might contain evidence of relationship between two entities
            # Similar to CO_LOCATED at chunk level, but could be extended with relationship tables
            conditions.append(
                f"""(
                    EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = {chunk_id_expr} AND em.entity_id = %s)
                    AND EXISTS (SELECT 1 FROM entity_mentions em WHERE em.chunk_id = {chunk_id_expr} AND em.entity_id = %s)
                )"""
            )
            params.append(p.entity_a)
            params.append(p.entity_b)
            required_joins["entity_mentions"] = True
    
    if not conditions:
        return ("", [], required_joins)
    
    sql_fragment = " AND ".join(conditions)
    return (sql_fragment, params, required_joins)


def compile_primitives(primitives: List[Primitive], *, chunk_id_expr: str = "c.id") -> Dict[str, Any]:
    """
    Compile all primitives to executable query components.
    
    Returns dictionary with:
    - tsquery: structured tsquery (sql_template, params) + debug text
    - expanded: structured expanded query (base_text, expanded_text, expansions)
    - scope: structured scope constraints (where_sql, params)
    
    This is deterministic: same primitives → same compiled outputs (via canonical ordering).
    
    Args:
        primitives: List of primitives to compile
        chunk_id_expr: SQL expression for chunk ID (default: "c.id")
                      Use "chunk_id", "rc.chunk_id", etc. based on query context
    """
    # Canonically order primitives for determinism
    ordered_primitives = _canonical_order_primitives(primitives)
    
    # Compile expanded text first (single source of truth)
    expanded = compile_primitives_to_expanded(ordered_primitives)
    
    # Derive tsquery text from expanded.expanded_text (aligned with vector search)
    # Use expanded_text as the base, which will include expansions when added
    tsquery_query_text = expanded["expanded_text"]
    
    # If empty, use no-match sentinel
    if not tsquery_query_text:
        tsquery_text = "___nomatch___"
        tsquery_sql = "to_tsquery('simple', %s)"
        tsquery_params = ["___nomatch___"]
    else:
        # Use websearch_to_tsquery with the expanded text
        tsquery_text = tsquery_query_text  # Debug string
        tsquery_sql = "websearch_to_tsquery('simple', %s)"
        tsquery_params = [tsquery_query_text]
    
    # Compile scope constraints
    scope_sql, scope_params, required_joins = compile_primitives_to_scope(ordered_primitives, chunk_id_expr=chunk_id_expr)
    
    result = {
        "tsquery": {
            "sql": tsquery_sql,
            "params": tsquery_params,
            "text": tsquery_text,  # Debug/export string
        },
        "expanded": expanded,
    }
    
    if scope_sql:
        result["scope"] = {
            "where_sql": scope_sql,
            "params": scope_params,
            "required_joins": required_joins,  # Indicates which tables need to be joined
        }
    
    return result
