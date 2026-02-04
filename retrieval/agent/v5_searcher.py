"""
V5 Searcher - The agent that decides what tools to call

The Searcher has complete freedom to call any tool. The only controls are:
- Budget limits (enforced by controller)
- Stop contract (must have citations for all claims)
- Grader feedback (what evidence is good)

No lane logic. No predetermined tool sequences.
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from retrieval.agent.v5_types import (
    EvidenceStore,
    V5Budgets,
    ToolCallAction,
    StopAnswerAction,
    StopInsufficientAction,
    ActionType,
)
from retrieval.agent.tools import get_tools_for_prompt, TOOL_REGISTRY


# =============================================================================
# Configuration
# =============================================================================

SEARCHER_MODEL = "gpt-4o"  # Main reasoning model
SEARCHER_TEMPERATURE = 0.3


# =============================================================================
# Searcher Prompt
# =============================================================================

SEARCHER_SYSTEM_PROMPT = """You are a research agent searching historical archives to answer questions.

You have complete freedom to call any available tool. Your goal: find the best evidence to answer the question.

KEY RULES:
1. You can call ANY tool - there's no required sequence
2. You may STOP_ANSWER only if EVERY major claim has citations to evidence IDs with strength ≥2
3. If you can't find sufficient evidence, use STOP_INSUFFICIENT
4. Think strategically about what evidence you still need

The Grader (not you) decides what counts as good evidence. Focus on FINDING candidates, not judging them."""


def build_searcher_prompt(
    question: str,
    budgets: V5Budgets,
    steps_used: int,
    grader_calls_used: int,
    evidence_store: EvidenceStore,
    last_observation: str,
    conversation_context: Optional[str] = None,
) -> str:
    """Build the prompt for the searcher to decide next action."""
    
    steps_remaining = budgets.max_steps - steps_used
    
    # Get tool descriptions
    tools_section = _get_tools_description()
    
    # Evidence store view
    evidence_view = evidence_store.get_compact_view()
    
    # Claims coverage
    coverage = evidence_store.get_claims_coverage()
    if coverage:
        coverage_lines = ["Claims with evidence:"]
        for claim, eids in list(coverage.items())[:10]:
            coverage_lines.append(f"  - {claim[:60]}... (cites: {', '.join(eids[:3])})")
        coverage_section = "\n".join(coverage_lines)
    else:
        coverage_section = "No claims have evidence yet."
    
    # Citation-ready count
    citation_ready = len(evidence_store.get_citation_ready_items(budgets.min_citation_strength))
    
    context_section = ""
    if conversation_context:
        context_section = f"\nCONVERSATION CONTEXT:\n{conversation_context}\n"
    
    return f"""QUESTION: {question}
{context_section}
BUDGET STATUS:
- Steps remaining: {steps_remaining} / {budgets.max_steps}
- Evidence store: {len(evidence_store)} / {budgets.evidence_budget} items
- Citation-ready items (strength ≥{budgets.min_citation_strength}): {citation_ready}
- Grader calls used: {grader_calls_used} / {budgets.max_grader_calls}

{evidence_view}

{coverage_section}

LAST OBSERVATION:
{last_observation}

{tools_section}

YOUR OPTIONS:
1. CALL_TOOL - Call a tool to find more evidence
2. STOP_ANSWER - Stop with a complete answer (only if ALL claims have citations)
3. STOP_INSUFFICIENT - Stop because evidence is insufficient

OUTPUT FORMAT (JSON):
For CALL_TOOL:
{{"action": "CALL_TOOL", "tool_name": "...", "params": {{...}}, "rationale": "why this tool now"}}

For STOP_ANSWER:
{{"action": "STOP_ANSWER", "answer": "your complete answer", "major_claims": [
  {{"claim": "claim text", "evidence_ids": ["ev_xxx", "ev_yyy"]}},
  ...
]}}

For STOP_INSUFFICIENT:
{{"action": "STOP_INSUFFICIENT", "partial_answer": "what you can say", "what_missing": "what evidence is needed", "suggested_tool": {{"tool_name": "...", "params": {{...}}}}}}

REMEMBER: You may only STOP_ANSWER if every claim cites evidence with strength ≥{budgets.min_citation_strength}.
If you're unsure, keep searching or use STOP_INSUFFICIENT."""


def _get_tools_description() -> str:
    """Get compact tool descriptions for the searcher."""
    
    lines = ["AVAILABLE TOOLS:"]
    
    for name, spec in TOOL_REGISTRY.items():
        # Get parameter info
        params = []
        for pname, pinfo in spec.params_schema.items():
            required = "(required)" if pinfo.get("required", False) else "(optional)"
            params.append(f"{pname} {required}")
        
        param_str = ", ".join(params) if params else "no params"
        lines.append(f"  {name}: {spec.description[:80]}...")
        lines.append(f"    params: {param_str}")
    
    return "\n".join(lines)


# =============================================================================
# Action Parsing
# =============================================================================

def parse_searcher_action(
    response_text: str,
) -> Union[ToolCallAction, StopAnswerAction, StopInsufficientAction, None]:
    """Parse the searcher's response into an action."""
    
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None
    
    action_type = data.get("action", "").upper()
    
    if action_type == "CALL_TOOL":
        return ToolCallAction(
            tool_name=data.get("tool_name", ""),
            params=data.get("params", {}),
            rationale=data.get("rationale", ""),
        )
    
    elif action_type == "STOP_ANSWER":
        return StopAnswerAction(
            answer=data.get("answer", ""),
            major_claims=data.get("major_claims", []),
        )
    
    elif action_type == "STOP_INSUFFICIENT":
        suggested = data.get("suggested_tool")
        suggested_action = None
        if suggested:
            suggested_action = ToolCallAction(
                tool_name=suggested.get("tool_name", ""),
                params=suggested.get("params", {}),
            )
        return StopInsufficientAction(
            partial_answer=data.get("partial_answer", ""),
            what_missing=data.get("what_missing", ""),
            suggested_next_tool=suggested_action,
        )
    
    return None


# =============================================================================
# Searcher Class
# =============================================================================

class Searcher:
    """
    The agent that decides what tools to call.
    
    Has complete freedom - no lane logic, no predetermined sequences.
    """
    
    def __init__(
        self,
        model: str = SEARCHER_MODEL,
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self.total_calls = 0
    
    def decide_action(
        self,
        question: str,
        budgets: V5Budgets,
        steps_used: int,
        grader_calls_used: int,
        evidence_store: EvidenceStore,
        last_observation: str,
        conversation_context: Optional[str] = None,
    ) -> Union[ToolCallAction, StopAnswerAction, StopInsufficientAction]:
        """
        Decide the next action to take.
        
        Returns one of:
        - ToolCallAction: Call a tool
        - StopAnswerAction: Stop with complete answer
        - StopInsufficientAction: Stop with partial answer
        """
        start = time.time()
        
        # Check budget
        if steps_used >= budgets.max_steps:
            if self.verbose:
                print("    [Searcher] Budget exhausted, forcing stop", file=sys.stderr)
            return self._force_stop(evidence_store, budgets)
        
        # Build prompt
        prompt = build_searcher_prompt(
            question=question,
            budgets=budgets,
            steps_used=steps_used,
            grader_calls_used=grader_calls_used,
            evidence_store=evidence_store,
            last_observation=last_observation,
            conversation_context=conversation_context,
        )
        
        # Call LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if self.verbose:
                print("    [Searcher] No API key, using fallback", file=sys.stderr)
            return self._fallback_action(evidence_store)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            if self.verbose:
                print(f"    [Searcher] Deciding next action...", file=sys.stderr)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SEARCHER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=SEARCHER_TEMPERATURE,
                max_tokens=1500,
            )
            
            self.total_calls += 1
            elapsed = (time.time() - start) * 1000
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_action(evidence_store)
            
            action = parse_searcher_action(content)
            
            if action is None:
                if self.verbose:
                    print(f"    [Searcher] Failed to parse action, using fallback", file=sys.stderr)
                return self._fallback_action(evidence_store)
            
            if self.verbose:
                if isinstance(action, ToolCallAction):
                    print(f"    [Searcher] -> CALL_TOOL: {action.tool_name} ({elapsed:.0f}ms)", file=sys.stderr)
                elif isinstance(action, StopAnswerAction):
                    print(f"    [Searcher] -> STOP_ANSWER ({len(action.major_claims)} claims) ({elapsed:.0f}ms)", file=sys.stderr)
                elif isinstance(action, StopInsufficientAction):
                    print(f"    [Searcher] -> STOP_INSUFFICIENT: {action.what_missing[:50]}... ({elapsed:.0f}ms)", file=sys.stderr)
            
            return action
            
        except Exception as e:
            if self.verbose:
                print(f"    [Searcher] Error: {e}", file=sys.stderr)
            return self._fallback_action(evidence_store)
    
    def _force_stop(
        self,
        evidence_store: EvidenceStore,
        budgets: V5Budgets,
    ) -> Union[StopAnswerAction, StopInsufficientAction]:
        """Force a stop when budget is exhausted."""
        
        citation_ready = evidence_store.get_citation_ready_items(budgets.min_citation_strength)
        
        if citation_ready:
            # We have some evidence - create a partial answer
            claims = []
            for item in citation_ready[:5]:
                claims.append({
                    "claim": item.claim_supported,
                    "evidence_ids": [item.evidence_id],
                })
            
            return StopAnswerAction(
                answer="Based on the available evidence: " + "; ".join(
                    item.claim_supported for item in citation_ready[:5]
                ),
                major_claims=claims,
            )
        else:
            return StopInsufficientAction(
                partial_answer="Insufficient evidence found within budget.",
                what_missing="Strong supporting evidence for the question",
            )
    
    def _fallback_action(self, evidence_store: EvidenceStore) -> ToolCallAction:
        """Fallback action when LLM fails."""
        # Default to hybrid search
        return ToolCallAction(
            tool_name="hybrid_search",
            params={"query": "relevant evidence", "top_k": 50},
            rationale="Fallback: LLM decision failed",
        )


# =============================================================================
# Observation Builder
# =============================================================================

def build_observation(
    tool_name: str,
    candidates_count: int,
    items_added: List[str],
    items_evicted: List[str],
    grader_summary: Dict[str, int],
    evidence_store: EvidenceStore,
    concordance_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Build observation feedback for the searcher."""
    
    lines = []
    
    # Tool result
    lines.append(f"Tool '{tool_name}' returned {candidates_count} candidates.")
    
    # Concordance expansion info
    if concordance_info:
        aliases = concordance_info.get("aliases", [])
        entity_via = concordance_info.get("entity_matched_via", "")
        
        if aliases:
            lines.append(f"Concordance expanded search to include aliases: {aliases[:5]}{'...' if len(aliases) > 5 else ''}")
        
        if entity_via and "concordance" in entity_via:
            lines.append(f"Entity resolved via concordance: {entity_via}")
    
    # Grading summary
    supporting = grader_summary.get("supporting", 0)
    strong = grader_summary.get("strong", 0)
    lines.append(f"Graded: {supporting} supporting the question, {strong} with strength ≥2.")
    
    # Store mutations
    if items_added:
        lines.append(f"Added to evidence store: {', '.join(items_added[:5])}" + 
                    (f" and {len(items_added) - 5} more" if len(items_added) > 5 else ""))
    else:
        lines.append("No new items added to evidence store.")
    
    if items_evicted:
        lines.append(f"Evicted (replaced by better evidence): {', '.join(items_evicted)}")
    
    # Coverage hint
    coverage = evidence_store.get_claims_coverage()
    if coverage:
        lines.append(f"Current coverage: {len(coverage)} distinct claims supported.")
    
    return "\n".join(lines)
