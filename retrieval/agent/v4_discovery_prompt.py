"""
V4 Discovery Prompt - Schemas and prompts for discovery action planning.

This module defines:
- DiscoveryAction: A single tool action 4o can request
- DiscoveryPlan: A plan with multiple actions for one discovery round
- DiscoveryObservation: Compact state returned after action execution
- Prompt templates for action proposal

The discovery loop lets 4o iteratively drive tool calls like a human researcher,
learning from results and refining searches until coverage is sufficient.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum
import json


# =============================================================================
# Discovery Action Types
# =============================================================================

class DiscoveryTool(str, Enum):
    """Available tools for discovery actions."""
    # Search tools (return chunk_ids)
    HYBRID_SEARCH = "hybrid_search"
    VECTOR_SEARCH = "vector_search"
    LEXICAL_SEARCH = "lexical_search"
    LEXICAL_EXACT = "lexical_exact"
    
    # Entity tools (return entity info or chunk_ids)
    ENTITY_LOOKUP = "entity_lookup"
    ENTITY_SURFACES = "entity_surfaces"
    ENTITY_MENTIONS = "entity_mentions"
    CO_MENTION_EXPAND = "co_mention_expand"
    
    # Concordance tools
    CONCORDANCE_EXPAND = "concordance_expand"


# Tool parameter schemas for validation
TOOL_PARAM_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "hybrid_search": {
        "query": {"type": "string", "required": True},
        "top_k": {"type": "integer", "default": 200, "max": 200},
        "collections": {"type": "array", "items": "string"},
    },
    "vector_search": {
        "query": {"type": "string", "required": True},
        "top_k": {"type": "integer", "default": 200, "max": 200},
        "collections": {"type": "array", "items": "string"},
    },
    "lexical_search": {
        "terms": {"type": "array", "items": "string", "required": True},
        "top_k": {"type": "integer", "default": 200, "max": 200},
        "collections": {"type": "array", "items": "string"},
    },
    "lexical_exact": {
        "term": {"type": "string", "required": True},
        "top_k": {"type": "integer", "default": 200, "max": 200},
        "collections": {"type": "array", "items": "string"},
    },
    "entity_lookup": {
        "name": {"type": "string", "required": True},
    },
    "entity_surfaces": {
        "entity_id": {"type": "integer", "required": True},
    },
    "entity_mentions": {
        "entity_id": {"type": "integer", "required": True},
        "top_k": {"type": "integer", "default": 100, "max": 200},
    },
    "co_mention_expand": {
        "entity_id": {"type": "integer", "required": True},
        "top_k": {"type": "integer", "default": 20, "max": 50},
    },
    "concordance_expand": {
        "term": {"type": "string", "required": True},
        "max_aliases": {"type": "integer", "default": 25, "max": 50},
    },
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class DiscoveryAction:
    """A single tool action for discovery."""
    tool: str
    params: Dict[str, Any]
    rationale: str = ""  # Why this action (for audit)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "params": self.params,
            "rationale": self.rationale,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryAction":
        return cls(
            tool=data.get("tool", ""),
            params=data.get("params", {}),
            rationale=data.get("rationale", ""),
        )
    
    def action_hash(self) -> str:
        """Hash for deduplication."""
        import hashlib
        content = f"{self.tool}:{json.dumps(self.params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def validate(self) -> List[str]:
        """Validate action against schema. Returns list of errors."""
        errors = []
        
        if self.tool not in TOOL_PARAM_SCHEMAS:
            errors.append(f"Unknown tool: {self.tool}")
            return errors
        
        schema = TOOL_PARAM_SCHEMAS[self.tool]
        
        # Check required params
        for param_name, param_spec in schema.items():
            if param_spec.get("required") and param_name not in self.params:
                errors.append(f"Missing required param '{param_name}' for {self.tool}")
        
        # Check param bounds
        for param_name, value in self.params.items():
            if param_name in schema:
                spec = schema[param_name]
                if spec.get("type") == "integer" and isinstance(value, int):
                    max_val = spec.get("max")
                    if max_val and value > max_val:
                        errors.append(f"Param '{param_name}' exceeds max ({value} > {max_val})")
        
        return errors


@dataclass
class DiscoveryPlan:
    """A plan for one discovery round."""
    round_num: int
    actions: List[DiscoveryAction]
    stop: bool = False
    stop_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "actions": [a.to_dict() for a in self.actions],
            "stop": self.stop,
            "stop_reason": self.stop_reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryPlan":
        return cls(
            round_num=data.get("round_num", 0),
            actions=[DiscoveryAction.from_dict(a) for a in data.get("actions", [])],
            stop=data.get("stop", False),
            stop_reason=data.get("stop_reason", ""),
        )


@dataclass
class ToolSummary:
    """Compact summary of a tool execution result."""
    tool: str
    params_summary: str  # Short summary of params
    chunk_count: int
    new_chunk_count: int  # Chunks not seen before
    entity_ids: List[int]
    top_docs: List[int]  # Top document IDs
    elapsed_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "params_summary": self.params_summary,
            "chunk_count": self.chunk_count,
            "new_chunk_count": self.new_chunk_count,
            "entity_ids": self.entity_ids[:10],  # Cap for prompt
            "top_docs": self.top_docs[:5],
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


@dataclass
class SpanPreview:
    """Preview of a top span for observation."""
    span_idx: int
    doc_id: int
    page_ref: str
    quote_preview: str  # First 150 chars
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.span_idx,
            "doc": self.doc_id,
            "page": self.page_ref,
            "quote": self.quote_preview,
            "score": round(self.score, 3),
        }


@dataclass
class CoverageSignals:
    """Deterministic coverage metrics."""
    entity_attest_counts: Dict[str, int]  # surface -> count
    list_like_span_count: int
    definitional_span_count: int
    doc_concentration: Dict[int, int]  # doc_id -> span count
    unique_docs: int
    unique_pages: int
    total_spans: int
    marginal_gain_pct: float  # % new spans/docs this round
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_attests": dict(list(self.entity_attest_counts.items())[:10]),
            "list_spans": self.list_like_span_count,
            "definitional_spans": self.definitional_span_count,
            "unique_docs": self.unique_docs,
            "unique_pages": self.unique_pages,
            "total_spans": self.total_spans,
            "marginal_gain_pct": round(self.marginal_gain_pct, 1),
        }


@dataclass
class DiscoveryObservation:
    """
    Compact observation returned to 4o after action execution.
    
    This is what the model sees to plan next actions.
    """
    round_num: int
    tool_summaries: List[ToolSummary]
    coverage: CoverageSignals
    span_previews: List[SpanPreview]  # Top 5-10 spans
    candidate_entities: List[Dict[str, Any]]  # {id, name, mention_count}
    known_aliases: Dict[str, List[str]]  # term -> aliases found
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_num,
            "tools": [t.to_dict() for t in self.tool_summaries],
            "coverage": self.coverage.to_dict(),
            "top_spans": [s.to_dict() for s in self.span_previews[:10]],
            "entities": self.candidate_entities[:15],
            "aliases": {k: v[:5] for k, v in list(self.known_aliases.items())[:10]},
        }
    
    def to_prompt_text(self) -> str:
        """Format observation for LLM prompt."""
        lines = [f"=== Discovery Round {self.round_num} Results ===\n"]
        
        # Tool summaries
        lines.append("Tool Executions:")
        for ts in self.tool_summaries:
            if ts.error:
                lines.append(f"  - {ts.tool}({ts.params_summary}): ERROR - {ts.error}")
            else:
                lines.append(f"  - {ts.tool}({ts.params_summary}): {ts.chunk_count} chunks ({ts.new_chunk_count} new)")
        
        # Coverage
        lines.append(f"\nCoverage Metrics:")
        lines.append(f"  - Total spans: {self.coverage.total_spans}")
        lines.append(f"  - Unique docs: {self.coverage.unique_docs}")
        lines.append(f"  - List-like spans: {self.coverage.list_like_span_count}")
        lines.append(f"  - Definitional spans: {self.coverage.definitional_span_count}")
        lines.append(f"  - Marginal gain: {self.coverage.marginal_gain_pct:.1f}%")
        
        # Top entity attests
        if self.coverage.entity_attest_counts:
            top_attests = sorted(self.coverage.entity_attest_counts.items(), 
                                key=lambda x: -x[1])[:5]
            lines.append(f"\n  Top entity attestations:")
            for surface, count in top_attests:
                lines.append(f"    - '{surface}': {count} spans")
        
        # Top spans
        lines.append(f"\nTop {len(self.span_previews)} Span Previews:")
        for sp in self.span_previews[:5]:
            lines.append(f"  [{sp.span_idx}] (doc:{sp.doc_id}, {sp.page_ref}, score:{sp.score:.2f})")
            lines.append(f"      \"{sp.quote_preview}...\"")
        
        # Candidate entities
        if self.candidate_entities:
            lines.append(f"\nCandidate Entities ({len(self.candidate_entities)}):")
            for e in self.candidate_entities[:10]:
                lines.append(f"  - {e.get('id')}: {e.get('name')} ({e.get('mention_count', 0)} mentions)")
        
        # Known aliases
        if self.known_aliases:
            lines.append(f"\nKnown Aliases:")
            for term, aliases in list(self.known_aliases.items())[:5]:
                lines.append(f"  - {term}: {aliases[:3]}")
        
        return "\n".join(lines)


# =============================================================================
# Prompt Templates
# =============================================================================

DISCOVERY_SYSTEM_PROMPT = """You are a research discovery agent. Your job is to iteratively search archival materials to find comprehensive evidence for a research query.

You have access to these tools:
- hybrid_search(query, top_k): Combined vector + lexical search (best for broad queries)
- vector_search(query, top_k): Semantic similarity search
- lexical_search(terms[], top_k): All terms must appear (precise matching)
- lexical_exact(term, top_k): Exact substring match (names, phrases)
- entity_lookup(name): Find entity ID by name
- entity_surfaces(entity_id): Get all surface forms (canonical + aliases) for an entity
- entity_mentions(entity_id, top_k): Find all chunks mentioning an entity
- co_mention_expand(entity_id, top_k): Find entities that co-occur with a given entity
- concordance_expand(term): Get aliases/variants from concordance

Strategy tips:
1. Start with hybrid_search for broad coverage
2. Use entity_lookup to find entity IDs for key people/organizations
3. Use entity_mentions to find all references to key entities
4. Use lexical_exact for specific names, codenames, or phrases
5. Use co_mention_expand to discover related entities
6. Check aliases via entity_surfaces or concordance_expand for variant spellings

Stop when:
- Coverage is sufficient (many list-like or definitional spans found)
- Marginal gain is low (<10% new content per round)
- Key entities are well-attested in the evidence"""


DISCOVERY_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": [t.value for t in DiscoveryTool]
                    },
                    "params": {"type": "object"},
                    "rationale": {"type": "string"}
                },
                "required": ["tool", "params"],
                "additionalProperties": False
            },
            "maxItems": 6
        },
        "stop": {"type": "boolean"},
        "stop_reason": {"type": "string"}
    },
    "required": ["actions", "stop"],
    "additionalProperties": False
}


def build_discovery_plan_prompt(
    query: str,
    round_num: int,
    previous_observation: Optional[DiscoveryObservation] = None,
    constraints: Optional[Dict[str, Any]] = None,
    seen_action_hashes: Optional[set] = None,
) -> str:
    """
    Build prompt for 4o to propose discovery actions.
    
    Args:
        query: User's research query
        round_num: Current discovery round (1-indexed)
        previous_observation: Results from previous round (None for round 1)
        constraints: Query constraints (collections, dates, etc.)
        seen_action_hashes: Set of action hashes already executed
    
    Returns:
        Prompt string for action planning
    """
    lines = [f"RESEARCH QUERY: {query}\n"]
    
    if constraints:
        lines.append("CONSTRAINTS:")
        if constraints.get("collections"):
            lines.append(f"  - Collections: {constraints['collections']}")
        if constraints.get("date_from"):
            lines.append(f"  - Date range: {constraints.get('date_from')} to {constraints.get('date_to', 'present')}")
        lines.append("")
    
    lines.append(f"DISCOVERY ROUND: {round_num}\n")
    
    if previous_observation:
        lines.append(previous_observation.to_prompt_text())
        lines.append("")
    else:
        lines.append("This is the first discovery round. Start with broad searches.\n")
    
    if seen_action_hashes:
        lines.append(f"NOTE: {len(seen_action_hashes)} actions already executed. Avoid duplicates.\n")
    
    lines.append("""
TASK: Propose up to 6 tool actions to expand evidence coverage.

Consider:
1. What key entities/concepts need better attestation?
2. Are there aliases or variant spellings to search?
3. Would entity-based pivots (mentions, co-mentions) find more evidence?
4. Is coverage sufficient to stop discovery?

OUTPUT JSON with:
{
  "actions": [
    {"tool": "...", "params": {...}, "rationale": "..."},
    ...
  ],
  "stop": false,  // true if coverage is sufficient
  "stop_reason": ""  // explain if stopping
}
""")
    
    return "\n".join(lines)


def build_initial_plan_prompt(
    query: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """Build prompt for initial discovery plan (before any observations)."""
    lines = [f"RESEARCH QUERY: {query}\n"]
    
    if constraints:
        lines.append("CONSTRAINTS:")
        if constraints.get("collections"):
            lines.append(f"  - Collections: {constraints['collections']}")
        lines.append("")
    
    lines.append("""
TASK: Generate an initial discovery plan to find evidence for this query.

For a roster/network query, good initial actions include:
1. hybrid_search with the main topic (e.g., "Silvermaster network members")
2. lexical_exact for specific names mentioned in the query
3. entity_lookup for key people/organizations mentioned

For a narrative/evidence query:
1. hybrid_search with the main claim
2. vector_search for semantic variants
3. lexical_search with key terms that must appear

Propose up to 6 actions. These will be executed in parallel.

OUTPUT JSON:
{
  "actions": [
    {"tool": "hybrid_search", "params": {"query": "...", "top_k": 200}, "rationale": "..."},
    ...
  ],
  "stop": false,
  "stop_reason": ""
}
""")
    
    return "\n".join(lines)
