"""
Agent Plan Schema V2 for Agentic V2 Architecture.

Extends the original plan schema with:
- QueryContract integration (scope/targets/constraints)
- FocusBundle mode selection
- Constraint-aware configuration
- Expansion controls

The agent produces this plan; execution is deterministic and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from retrieval.query_intent import (
    QueryContract,
    FocusBundleMode,
    ConstraintSpec,
    TargetSpec,
)


@dataclass
class RetrievalLaneConfig:
    """Configuration for a retrieval lane."""
    lane_id: str                              # "hybrid", "lexical_must_hit", "entity_mentions"
    budget: int = 200                         # max results for this lane
    enabled: bool = True
    priority: int = 1                         # execution priority
    
    # Lane-specific options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FocusBundleConfig:
    """Configuration for FocusBundle construction."""
    top_n_spans: int = 80
    lambda_mmr: float = 0.7
    min_span_score: float = 0.3
    max_spans_per_doc: int = 10
    max_spans_per_chunk: int = 2
    context_fill_quota: int = 20


@dataclass
class ExpansionConfig:
    """Configuration for expansion loops."""
    enabled: bool = False
    rounds: int = 2
    max_entities: int = 10
    max_chunks: int = 500
    stability_threshold: float = 0.85
    
    # Expansion mode
    mode: str = "entity"  # "entity" or "term"


@dataclass
class OutputConfig:
    """Configuration for output rendering."""
    max_items: int = 25
    max_citations_per_item: int = 2
    conservative_language: bool = True  # Use hedging for weak support
    include_negative_findings: bool = True


@dataclass
class AgentPlanV2:
    """
    Agent's plan-of-record for V2 architecture.
    
    The agent outputs this plan, then it's executed deterministically.
    All decisions are recorded for reproducibility and audit.
    """
    
    # Query understanding
    query_text: str
    mode: FocusBundleMode = FocusBundleMode.KEYWORD_INTENT
    query_contract: Optional[QueryContract] = None
    
    # Retrieval configuration
    lanes: List[RetrievalLaneConfig] = field(default_factory=lambda: [
        RetrievalLaneConfig(lane_id="hybrid", budget=200),
        RetrievalLaneConfig(lane_id="lexical_must_hit", budget=100),
    ])
    
    # FocusBundle configuration
    focus_bundle: FocusBundleConfig = field(default_factory=FocusBundleConfig)
    
    # Constraints (from QueryContract or explicit)
    constraints: List[ConstraintSpec] = field(default_factory=list)
    constraint_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Expansion configuration
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    
    # Output configuration
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Primitives to execute (ordered)
    primitives: List[str] = field(default_factory=lambda: [
        "propose_candidates",
        "score_constraints",
        "apply_hubness",
        "render_answer",
        "verify_invariants",
    ])
    primitive_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata for reproducibility
    model_version: str = "gpt-4o-mini"
    temperature: float = 0.0  # Deterministic
    created_at: str = ""
    plan_version: str = "v2.0"
    
    def to_dict(self) -> dict:
        """Serialize for storage and audit."""
        return {
            "query_text": self.query_text,
            "mode": self.mode.value,
            "query_contract": self.query_contract.to_dict() if self.query_contract else None,
            "lanes": [
                {
                    "lane_id": l.lane_id,
                    "budget": l.budget,
                    "enabled": l.enabled,
                    "priority": l.priority,
                    "options": l.options,
                }
                for l in self.lanes
            ],
            "focus_bundle": {
                "top_n_spans": self.focus_bundle.top_n_spans,
                "lambda_mmr": self.focus_bundle.lambda_mmr,
                "min_span_score": self.focus_bundle.min_span_score,
                "max_spans_per_doc": self.focus_bundle.max_spans_per_doc,
                "max_spans_per_chunk": self.focus_bundle.max_spans_per_chunk,
                "context_fill_quota": self.focus_bundle.context_fill_quota,
            },
            "constraints": [
                {
                    "name": c.name,
                    "object": c.object,
                    "strength": c.strength,
                    "min_score": c.min_score,
                }
                for c in self.constraints
            ],
            "constraint_thresholds": self.constraint_thresholds,
            "expansion": {
                "enabled": self.expansion.enabled,
                "rounds": self.expansion.rounds,
                "max_entities": self.expansion.max_entities,
                "max_chunks": self.expansion.max_chunks,
                "stability_threshold": self.expansion.stability_threshold,
                "mode": self.expansion.mode,
            },
            "output": {
                "max_items": self.output.max_items,
                "max_citations_per_item": self.output.max_citations_per_item,
                "conservative_language": self.output.conservative_language,
                "include_negative_findings": self.output.include_negative_findings,
            },
            "primitives": self.primitives,
            "primitive_params": self.primitive_params,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "created_at": self.created_at,
            "plan_version": self.plan_version,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentPlanV2":
        """Deserialize from storage."""
        # Parse query_contract if present
        query_contract = None
        if data.get("query_contract"):
            query_contract = QueryContract.from_dict(data["query_contract"])
        
        # Parse lanes
        lanes = []
        for l in data.get("lanes", []):
            lanes.append(RetrievalLaneConfig(
                lane_id=l["lane_id"],
                budget=l.get("budget", 200),
                enabled=l.get("enabled", True),
                priority=l.get("priority", 1),
                options=l.get("options", {}),
            ))
        
        # Parse focus_bundle config
        fb_data = data.get("focus_bundle", {})
        focus_bundle = FocusBundleConfig(
            top_n_spans=fb_data.get("top_n_spans", 80),
            lambda_mmr=fb_data.get("lambda_mmr", 0.7),
            min_span_score=fb_data.get("min_span_score", 0.3),
            max_spans_per_doc=fb_data.get("max_spans_per_doc", 10),
            max_spans_per_chunk=fb_data.get("max_spans_per_chunk", 2),
            context_fill_quota=fb_data.get("context_fill_quota", 20),
        )
        
        # Parse constraints
        constraints = []
        for c in data.get("constraints", []):
            constraints.append(ConstraintSpec(
                name=c["name"],
                object=c.get("object"),
                strength=c.get("strength", "soft"),
                min_score=c.get("min_score", 0.0),
            ))
        
        # Parse expansion config
        exp_data = data.get("expansion", {})
        expansion = ExpansionConfig(
            enabled=exp_data.get("enabled", False),
            rounds=exp_data.get("rounds", 2),
            max_entities=exp_data.get("max_entities", 10),
            max_chunks=exp_data.get("max_chunks", 500),
            stability_threshold=exp_data.get("stability_threshold", 0.85),
            mode=exp_data.get("mode", "entity"),
        )
        
        # Parse output config
        out_data = data.get("output", {})
        output = OutputConfig(
            max_items=out_data.get("max_items", 25),
            max_citations_per_item=out_data.get("max_citations_per_item", 2),
            conservative_language=out_data.get("conservative_language", True),
            include_negative_findings=out_data.get("include_negative_findings", True),
        )
        
        return cls(
            query_text=data["query_text"],
            mode=FocusBundleMode(data.get("mode", "keyword_intent")),
            query_contract=query_contract,
            lanes=lanes,
            focus_bundle=focus_bundle,
            constraints=constraints,
            constraint_thresholds=data.get("constraint_thresholds", {}),
            expansion=expansion,
            output=output,
            primitives=data.get("primitives", []),
            primitive_params=data.get("primitive_params", {}),
            model_version=data.get("model_version", "gpt-4o-mini"),
            temperature=data.get("temperature", 0.0),
            created_at=data.get("created_at", ""),
            plan_version=data.get("plan_version", "v2.0"),
        )


def build_keyword_intent_plan(query_text: str, collections: List[str] = None) -> AgentPlanV2:
    """
    Build a plan for simple keyword intent queries.
    
    Uses standard retrieval lanes without targets or constraints.
    """
    from retrieval.query_intent import build_keyword_intent_contract
    
    contract = build_keyword_intent_contract(query_text, collections)
    
    return AgentPlanV2(
        query_text=query_text,
        mode=FocusBundleMode.KEYWORD_INTENT,
        query_contract=contract,
        lanes=[
            RetrievalLaneConfig(lane_id="hybrid", budget=200),
            RetrievalLaneConfig(lane_id="lexical_must_hit", budget=100),
        ],
    )


def build_relationship_plan(
    query_text: str,
    target_entity_id: int,
    target_name: str,
    target_aliases: List[str] = None,
    role_constraint: str = None,
    collections: List[str] = None,
) -> AgentPlanV2:
    """
    Build a plan for relationship queries.
    
    Example: "officers closely associated with Julius Rosenberg"
    """
    from retrieval.query_intent import build_relationship_contract
    
    contract = build_relationship_contract(
        query_text=query_text,
        target_entity_id=target_entity_id,
        target_name=target_name,
        target_aliases=target_aliases,
        role_constraint=role_constraint,
        collections=collections,
    )
    
    constraints = [
        ConstraintSpec(name="relationship", object=target_name, min_score=0.3),
    ]
    if role_constraint:
        constraints.append(ConstraintSpec(name="role", object=role_constraint, min_score=0.2))
    
    return AgentPlanV2(
        query_text=query_text,
        mode=FocusBundleMode.TARGET_ANCHORED,
        query_contract=contract,
        constraints=constraints,
        expansion=ExpansionConfig(enabled=True, mode="entity"),
        lanes=[
            RetrievalLaneConfig(lane_id="hybrid", budget=300),
            RetrievalLaneConfig(lane_id="entity_mentions", budget=200),
            RetrievalLaneConfig(lane_id="lexical_must_hit", budget=100),
        ],
    )


def build_affiliation_plan(
    query_text: str,
    org_name: str,
    org_aliases: List[str] = None,
    role_constraint: str = None,
    collections: List[str] = None,
) -> AgentPlanV2:
    """
    Build a plan for affiliation queries.
    
    Example: "Soviet agents in the OSS"
    """
    from retrieval.query_intent import build_affiliation_contract
    
    contract = build_affiliation_contract(
        query_text=query_text,
        org_name=org_name,
        org_aliases=org_aliases,
        role_constraint=role_constraint,
        collections=collections,
    )
    
    constraints = [
        ConstraintSpec(name="affiliation", object=org_name, min_score=0.3),
    ]
    if role_constraint:
        constraints.append(ConstraintSpec(name="role", object=role_constraint, min_score=0.3))
    
    return AgentPlanV2(
        query_text=query_text,
        mode=FocusBundleMode.TARGET_ANCHORED,
        query_contract=contract,
        constraints=constraints,
        lanes=[
            RetrievalLaneConfig(lane_id="hybrid", budget=300),
            RetrievalLaneConfig(lane_id="lexical_must_hit", budget=150),
        ],
    )
