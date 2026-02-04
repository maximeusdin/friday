"""
V3 Plan Schema - Structured plan that LLM produces and we persist.

The plan defines:
- Query text and constraints
- Ordered tool steps to execute
- Budgets (max rounds, chunks, spans)
- Stop conditions

Plans are deterministic: same query + context = same plan hash.
"""

import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from retrieval.agent import DEFAULT_BUDGETS, V3_MODEL_DEFAULT
from retrieval.agent.executor import ToolStep
from retrieval.agent.tools import get_tools_for_prompt, TOOL_REGISTRY


@dataclass
class PlanConstraints:
    """Constraints for the search plan."""
    collections: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    doc_ids: Optional[List[int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "collections": self.collections,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "doc_ids": self.doc_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanConstraints":
        return cls(
            collections=data.get("collections"),
            date_from=data.get("date_from"),
            date_to=data.get("date_to"),
            doc_ids=data.get("doc_ids"),
        )


@dataclass
class PlanBudgets:
    """Budget limits for plan execution."""
    max_rounds: int = 2
    max_steps: int = 8
    max_chunks: int = 200
    max_cite_spans: int = 120
    max_harvest_spans: int = 240
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "max_steps": self.max_steps,
            "max_chunks": self.max_chunks,
            "max_cite_spans": self.max_cite_spans,
            "max_harvest_spans": self.max_harvest_spans,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanBudgets":
        return cls(
            max_rounds=data.get("max_rounds", DEFAULT_BUDGETS["max_rounds"]),
            max_steps=data.get("max_steps", DEFAULT_BUDGETS["max_steps"]),
            max_chunks=data.get("max_chunks", DEFAULT_BUDGETS["max_chunks"]),
            max_cite_spans=data.get("max_cite_spans", DEFAULT_BUDGETS["max_cite_spans"]),
            max_harvest_spans=data.get("max_harvest_spans", DEFAULT_BUDGETS["max_harvest_spans"]),
        )


@dataclass
class AgentPlanV3:
    """
    V3 Agent Plan - structured plan of tool calls.
    
    The LLM produces this JSON, and the executor runs it deterministically.
    """
    query_text: str
    constraints: PlanConstraints
    steps: List[ToolStep]
    budgets: PlanBudgets
    stop_conditions: Dict[str, Any] = field(default_factory=dict)
    model_version: str = ""
    plan_hash: str = ""
    reasoning: str = ""
    
    def __post_init__(self):
        if not self.model_version:
            self.model_version = V3_MODEL_DEFAULT
        if not self.plan_hash:
            self.plan_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute deterministic hash of plan content."""
        content = {
            "query_text": self.query_text,
            "constraints": self.constraints.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "budgets": self.budgets.to_dict(),
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "constraints": self.constraints.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "budgets": self.budgets.to_dict(),
            "stop_conditions": self.stop_conditions,
            "model_version": self.model_version,
            "plan_hash": self.plan_hash,
            "reasoning": self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPlanV3":
        return cls(
            query_text=data.get("query_text", ""),
            constraints=PlanConstraints.from_dict(data.get("constraints", {})),
            steps=[ToolStep.from_dict(s) for s in data.get("steps", [])],
            budgets=PlanBudgets.from_dict(data.get("budgets", {})),
            stop_conditions=data.get("stop_conditions", {}),
            model_version=data.get("model_version", V3_MODEL_DEFAULT),
            plan_hash=data.get("plan_hash", ""),
            reasoning=data.get("reasoning", ""),
        )


# =============================================================================
# Plan Generation Prompt
# =============================================================================

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "constraints": {
            "type": "object",
            "properties": {
                "collections": {"type": ["array", "null"], "items": {"type": "string"}},
                "date_from": {"type": ["string", "null"]},
                "date_to": {"type": ["string", "null"]},
            },
            "required": ["collections", "date_from", "date_to"],
            "additionalProperties": False,
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "enum": list(TOOL_REGISTRY.keys())},
                    "params": {"type": "object", "additionalProperties": True},
                    "description": {"type": "string"},
                },
                "required": ["tool_name", "params", "description"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["constraints", "steps", "reasoning"],
    "additionalProperties": False,
}


def build_plan_prompt(query: str, context: Optional[Dict] = None) -> str:
    """Build comprehensive prompt for plan generation."""
    context = context or {}
    
    tools_desc = get_tools_for_prompt()
    
    collections_hint = ""
    if context.get("available_collections"):
        collections_hint = f"\nAvailable collections: {', '.join(context['available_collections'])}"
    
    # Detect query type for strategy hints
    query_lower = query.lower()
    query_type_hint = ""
    
    if any(w in query_lower for w in ["who", "members", "network", "group", "agents"]):
        query_type_hint = """
DETECTED: ROSTER/NETWORK QUERY
Recommended strategy:
1. hybrid_search for broad coverage
2. entity_lookup for key entity
3. co_mention_entities to discover network members
4. entity_mentions for each discovered member
"""
    elif any(w in query_lower for w in ["when", "timeline", "first", "earliest"]):
        query_type_hint = """
DETECTED: TIMELINE QUERY  
Recommended strategy:
1. entity_lookup for key entities
2. first_mention for chronological anchors
3. entity_mentions for full context
"""
    elif any(w in query_lower for w in ["evidence", "proof", "show that", "demonstrate"]):
        query_type_hint = """
DETECTED: EVIDENCE QUERY
Recommended strategy:
1. hybrid_search with claim text
2. lexical_search with key technical terms
3. vector_search for semantic variants
"""
    
    return f"""Generate a comprehensive search plan to answer this query.

QUERY: {query}
{collections_hint}

{tools_desc}
{query_type_hint}

PLANNING GUIDELINES:

1. **Use multiple tool types**: Combine search tools (hybrid, lexical) with entity tools for best coverage
2. **Chain entity tools**: Use entity_lookup FIRST to get IDs, then entity_mentions or co_mention_entities
3. **Expand names**: Use entity_surfaces or expand_aliases to get name variants, then search for them
4. **Be thorough**: Use up to {DEFAULT_BUDGETS['max_steps']} steps - more is better for complex queries
5. **Prioritize precision**: entity_mentions is more precise than lexical search for known entities

PARAMETER GUIDELINES:
- hybrid_search: query (string), top_k (default 200)
- vector_search: query (string), top_k (default 200)
- lexical_search: terms (list of strings), top_k (default 200)
- lexical_exact: term (string), top_k (default 200)
- entity_lookup: name (string)
- entity_surfaces: entity_id (integer)
- entity_mentions: entity_id (integer), top_k (default 200)
- co_mention_entities: entity_id (integer), top_k (default 30)
- expand_aliases: term (string)

CONSTRAINTS:
- Only set collections if query explicitly mentions them (e.g., "in the Vassiliev notebooks")
- Set date_from/date_to only if query mentions dates
- Leave as null if not applicable

OUTPUT: JSON with:
- constraints: scope filters
- steps: ordered list of tool calls (aim for 4-8 steps for complex queries)
- reasoning: explanation of your strategy

EXAMPLE STEP:
{{"tool_name": "entity_lookup", "params": {{"name": "Silvermaster"}}, "description": "Find entity ID for Silvermaster"}}
{{"tool_name": "entity_mentions", "params": {{"entity_id": 123, "top_k": 200}}, "description": "Get all chunks mentioning Silvermaster"}}
"""


def generate_plan(query: str, conn, context: Optional[Dict] = None) -> AgentPlanV3:
    """
    Generate a plan using LLM.
    
    Args:
        query: The user's query
        conn: Database connection (for context like available collections)
        context: Optional additional context
    
    Returns:
        AgentPlanV3 ready for execution
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to default plan
        return _default_plan(query)
    
    # Get available collections for context
    context = context or {}
    if not context.get("available_collections"):
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT slug FROM collections ORDER BY slug")
                context["available_collections"] = [row[0] for row in cur.fetchall()]
        except Exception:
            context["available_collections"] = []
    
    # Build prompt
    prompt = build_plan_prompt(query, context)
    
    # Call LLM
    model = os.getenv("OPENAI_MODEL_V3", V3_MODEL_DEFAULT)
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a search planner. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        
        content = response.choices[0].message.content
        if not content:
            return _default_plan(query)
        
        data = json.loads(content)
        
        # Build plan from response
        return AgentPlanV3(
            query_text=query,
            constraints=PlanConstraints.from_dict(data.get("constraints", {})),
            steps=[ToolStep.from_dict(s) for s in data.get("steps", [])],
            budgets=PlanBudgets(),
            reasoning=data.get("reasoning", ""),
            model_version=model,
        )
        
    except Exception as e:
        print(f"    Plan generation error: {e}", file=sys.stderr)
        return _default_plan(query)


def _default_plan(query: str) -> AgentPlanV3:
    """Create a default plan when LLM fails."""
    return AgentPlanV3(
        query_text=query,
        constraints=PlanConstraints(),
        steps=[
            ToolStep(
                tool_name="hybrid_search",
                params={"query": query, "top_k": 200},
                description="Initial broad search",
            ),
        ],
        budgets=PlanBudgets(),
        reasoning="Default plan: hybrid search on query",
    )


def revise_plan(
    original_plan: AgentPlanV3,
    errors: List[str],
    stats: Dict[str, Any],
    conn,
) -> AgentPlanV3:
    """
    Revise a plan based on verification errors.
    
    Args:
        original_plan: The plan that was executed
        errors: List of error messages from verification
        stats: Statistics from the evidence set
        conn: Database connection
    
    Returns:
        Revised AgentPlanV3
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _expand_plan(original_plan)
    
    prompt = f"""The search plan failed verification. Revise it.

ORIGINAL QUERY: {original_plan.query_text}

ORIGINAL STEPS:
{json.dumps([s.to_dict() for s in original_plan.steps], indent=2)}

ERRORS:
{chr(10).join(f"- {e}" for e in errors[:5])}

STATISTICS:
- Total spans: {stats.get('total_spans_mined', 0)}
- Cite spans: {stats.get('cite_span_count', 0)}
- Unique docs: {stats.get('unique_docs', 0)}

REVISION GUIDANCE:
- If 0 spans: try different search terms or expand aliases
- If errors mention "missing citation": add more specific searches
- If errors mention "quote mismatch": search is finding wrong content

{get_tools_for_prompt()}

OUTPUT: JSON with revised steps and reasoning."""

    model = os.getenv("OPENAI_MODEL_V3", V3_MODEL_DEFAULT)
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are revising a failed search plan. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        
        content = response.choices[0].message.content
        if not content:
            return _expand_plan(original_plan)
        
        data = json.loads(content)
        
        # Parse steps from LLM response, fall back to original if not present
        llm_steps = data.get("steps")
        if llm_steps and isinstance(llm_steps, list):
            new_steps = [ToolStep.from_dict(s) for s in llm_steps]
        else:
            # Keep original steps if LLM didn't provide new ones
            new_steps = list(original_plan.steps)
        
        return AgentPlanV3(
            query_text=original_plan.query_text,
            constraints=original_plan.constraints,
            steps=new_steps,
            budgets=original_plan.budgets,
            reasoning=data.get("reasoning", "Revised plan"),
            model_version=model,
        )
        
    except Exception as e:
        print(f"    Plan revision error: {e}", file=sys.stderr)
        return _expand_plan(original_plan)


def _expand_plan(plan: AgentPlanV3) -> AgentPlanV3:
    """Expand a plan with additional searches when revision fails."""
    # Add a lexical search if not already present
    has_lexical = any(s.tool_name == "lexical_search" for s in plan.steps)
    
    new_steps = list(plan.steps)
    if not has_lexical:
        # Extract key terms from query
        terms = [t for t in plan.query_text.split() if len(t) > 3][:3]
        if terms:
            new_steps.append(ToolStep(
                tool_name="lexical_search",
                params={"terms": terms, "top_k": 200},
                description="Fallback lexical search",
            ))
    
    return AgentPlanV3(
        query_text=plan.query_text,
        constraints=plan.constraints,
        steps=new_steps,
        budgets=plan.budgets,
        reasoning="Expanded plan with additional searches",
    )


# Import for type hints
import sys
