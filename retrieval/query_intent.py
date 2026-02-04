"""
Query Intent Decomposition for Agentic V2.

This module defines the QueryContract which decomposes queries into:
- Scope (collection/doc filters - never used as anchors)
- Targets (anchor entities/terms the FocusBundle centers on)
- Constraints (soft-scored, hard-evidence requirements)

The agent produces this contract; execution is deterministic and auditable.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class FocusBundleMode(Enum):
    """Mode for FocusBundle construction."""
    KEYWORD_INTENT = "keyword_intent"       # existence, roster queries
    TARGET_ANCHORED = "target_anchored"     # relationship, affiliation queries


@dataclass
class TargetSpec:
    """
    An anchor target the FocusBundle should center on.
    
    For relationship queries: the person being asked about (e.g., Julius Rosenberg)
    For affiliation queries: the organization (e.g., OSS as a term)
    """
    kind: str                    # "entity" | "term"
    entity_id: Optional[int]     # resolved entity ID if kind="entity"
    surface_forms: List[str]     # aliases / codenames / surface strings
    required: bool = True        # if True, must appear in anchor spans


@dataclass
class ConstraintSpec:
    """
    A constraint the answer must satisfy.
    
    Constraints are soft-scored but enforced via evidence at render/verify time.
    This gives robustness + flexibility without letting the model "make stuff up."
    """
    name: str                    # "affiliation", "role", "relationship"
    subject: str = "candidate"   # usually "candidate" (the thing we're ranking)
    object: Optional[str] = None # e.g. "OSS", "Soviet intelligence", "Julius Rosenberg"
    strength: str = "soft"       # "soft" or "required"
    min_score: float = 0.0       # threshold for rendering
    
    @property
    def constraint_key(self) -> str:
        """
        Standard constraint key format (Contract C4).
        
        Examples:
        - "affiliation:OSS"
        - "role:Soviet intelligence"
        - "relationship:Julius Rosenberg"
        """
        return f"{self.name}:{self.object}" if self.object else self.name


@dataclass
class QueryContract:
    """
    Structured decomposition of a query into scope, targets, and constraints.
    
    This is the bridge between query understanding and execution.
    The agent produces this; execution is deterministic and auditable.
    
    Example for "officers closely associated with Julius Rosenberg":
        QueryContract(
            query_text="officers closely associated with Julius Rosenberg",
            mode=FocusBundleMode.TARGET_ANCHORED,
            targets=[TargetSpec(kind="entity", entity_id=123, surface_forms=["Julius Rosenberg"])],
            constraints=[
                ConstraintSpec(name="relationship", object="Julius Rosenberg", min_score=0.3),
                ConstraintSpec(name="role", object="officer", min_score=0.2),
            ],
        )
    
    Example for "Soviet agents in the OSS":
        QueryContract(
            query_text="Soviet agents in the OSS",
            mode=FocusBundleMode.TARGET_ANCHORED,
            targets=[TargetSpec(kind="term", entity_id=None, surface_forms=["OSS", "Office of Strategic Services"])],
            constraints=[
                ConstraintSpec(name="affiliation", object="OSS", min_score=0.3),
                ConstraintSpec(name="role", object="Soviet intelligence", min_score=0.3),
            ],
        )
    """
    query_text: str
    
    # Mode determines FocusBundle construction strategy
    mode: FocusBundleMode = FocusBundleMode.KEYWORD_INTENT
    
    # Scope-only filters (never used as anchors, avoid anchor pollution)
    scope_collections: List[str] = field(default_factory=list)
    scope_doc_ids: List[int] = field(default_factory=list)
    
    # Anchor targets (things the bundle should center on)
    targets: List[TargetSpec] = field(default_factory=list)
    
    # Constraints the answer must satisfy
    constraints: List[ConstraintSpec] = field(default_factory=list)
    
    def get_target_entity_ids(self) -> List[int]:
        """Get all entity IDs from targets."""
        return [t.entity_id for t in self.targets if t.entity_id is not None]
    
    def get_target_surface_forms(self) -> List[str]:
        """Get all surface forms from targets."""
        forms = []
        for t in self.targets:
            forms.extend(t.surface_forms)
        return forms
    
    def get_constraint_keys(self) -> List[str]:
        """Get all constraint keys."""
        return [c.constraint_key for c in self.constraints]
    
    def to_dict(self) -> dict:
        """Serialize for storage in focus_bundle_params."""
        return {
            "query_text": self.query_text,
            "mode": self.mode.value,
            "scope_collections": self.scope_collections,
            "scope_doc_ids": self.scope_doc_ids,
            "targets": [
                {
                    "kind": t.kind,
                    "entity_id": t.entity_id,
                    "surface_forms": t.surface_forms,
                    "required": t.required,
                }
                for t in self.targets
            ],
            "constraints": [
                {
                    "name": c.name,
                    "subject": c.subject,
                    "object": c.object,
                    "strength": c.strength,
                    "min_score": c.min_score,
                }
                for c in self.constraints
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QueryContract":
        """Deserialize from storage."""
        return cls(
            query_text=data["query_text"],
            mode=FocusBundleMode(data["mode"]),
            scope_collections=data.get("scope_collections", []),
            scope_doc_ids=data.get("scope_doc_ids", []),
            targets=[
                TargetSpec(
                    kind=t["kind"],
                    entity_id=t.get("entity_id"),
                    surface_forms=t["surface_forms"],
                    required=t.get("required", True),
                )
                for t in data.get("targets", [])
            ],
            constraints=[
                ConstraintSpec(
                    name=c["name"],
                    subject=c.get("subject", "candidate"),
                    object=c.get("object"),
                    strength=c.get("strength", "soft"),
                    min_score=c.get("min_score", 0.0),
                )
                for c in data.get("constraints", [])
            ],
        )


def build_keyword_intent_contract(query_text: str, collections: List[str] = None) -> QueryContract:
    """
    Build a simple KEYWORD_INTENT contract for existence/roster queries.
    
    No explicit targets or constraints - relies on query text for retrieval.
    """
    return QueryContract(
        query_text=query_text,
        mode=FocusBundleMode.KEYWORD_INTENT,
        scope_collections=collections or [],
    )


def build_relationship_contract(
    query_text: str,
    target_entity_id: int,
    target_name: str,
    target_aliases: List[str] = None,
    role_constraint: str = None,
    collections: List[str] = None,
) -> QueryContract:
    """
    Build a TARGET_ANCHORED contract for relationship queries.
    
    Example: "officers closely associated with Julius Rosenberg"
    """
    constraints = [
        ConstraintSpec(name="relationship", object=target_name, min_score=0.3),
    ]
    if role_constraint:
        constraints.append(ConstraintSpec(name="role", object=role_constraint, min_score=0.2))
    
    return QueryContract(
        query_text=query_text,
        mode=FocusBundleMode.TARGET_ANCHORED,
        scope_collections=collections or [],
        targets=[
            TargetSpec(
                kind="entity",
                entity_id=target_entity_id,
                surface_forms=[target_name] + (target_aliases or []),
            )
        ],
        constraints=constraints,
    )


def build_affiliation_contract(
    query_text: str,
    org_name: str,
    org_aliases: List[str] = None,
    role_constraint: str = None,
    collections: List[str] = None,
) -> QueryContract:
    """
    Build a TARGET_ANCHORED contract for affiliation queries.
    
    Example: "Soviet agents in the OSS"
    """
    constraints = [
        ConstraintSpec(name="affiliation", object=org_name, min_score=0.3),
    ]
    if role_constraint:
        constraints.append(ConstraintSpec(name="role", object=role_constraint, min_score=0.3))
    
    return QueryContract(
        query_text=query_text,
        mode=FocusBundleMode.TARGET_ANCHORED,
        scope_collections=collections or [],
        targets=[
            TargetSpec(
                kind="term",
                entity_id=None,
                surface_forms=[org_name] + (org_aliases or []),
            )
        ],
        constraints=constraints,
    )
