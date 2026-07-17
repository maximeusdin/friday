"""
V9 Router — Intent classification and query reference resolution.

Determines whether a user message should be handled as:
- NEW_RETRIEVAL: new search + evidence set
- FOLLOW_UP: answer from existing evidence set (no tools)
- THINK_DEEPER: resume a paused run with extended budget

Uses a small LLM call for intent classification.
"""
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from retrieval.agent.v9_session import (
    RecentQueryContext,
    RunRecord,
)


# =============================================================================
# Router output
# =============================================================================

@dataclass
class RouterDecision:
    """Output of the router."""
    intent: str                     # "new_retrieval" | "follow_up" | "think_deeper"
    target_run_id: Optional[int] = None
    target_evidence_set_id: Optional[int] = None
    confidence: float = 0.0
    reasoning: str = ""
    query_text: str = ""            # cleaned query for execution
    ref_run_id: Optional[int] = None       # which run the user is referencing
    ref_evidence_set_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "target_run_id": self.target_run_id,
            "target_evidence_set_id": self.target_evidence_set_id,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "query_text": self.query_text,
            "ref_run_id": self.ref_run_id,
            "ref_evidence_set_id": self.ref_evidence_set_id,
        }


# =============================================================================
# Query reference resolution (heuristic layer)
# =============================================================================

_BACK_REFERENCE_CUES = [
    "that one", "the previous", "earlier", "last search", "last query",
    "the one about", "that search", "that result", "those results",
    "the same", "that answer", "go back to",
]

_FOLLOW_UP_CUES = [
    "what does this mean", "who is he", "who is she", "who are they",
    "in that quote", "explain this", "what about", "more detail",
    "tell me more", "can you clarify", "what did they say",
    "in the same document", "from that source", "that passage",
    "what else", "any other", "related to this",
]

_THINK_DEEPER_CUES = [
    "think deeper", "keep searching", "continue searching",
    "dig deeper", "search more", "find more", "investigate further",
    "extend the search", "look harder", "go deeper",
]


def _build_entity_name_set(entity: Dict[str, Any]) -> set:
    """Build a set of all name forms for an entity (canonical + aliases), lowercased.

    Supports bidirectional matching: whether the user mentions the canonical
    name or any alias, we detect the reference.
    """
    names = set()
    canonical = entity.get("canonical_name", "")
    if canonical and len(canonical) >= 3:
        names.add(canonical.lower())
    for alias in entity.get("aliases", []):
        if alias and len(alias) >= 2:
            names.add(alias.lower())
    return names


def resolve_query_reference(
    user_message: str,
    context: RecentQueryContext,
) -> Dict[str, Any]:
    """Resolve 'that one / previous / the Silvermaster one' to a run/evidence set.

    Supports bidirectional concordance matching: matches if the user mentions
    ANY form of an entity name (canonical OR alias) that appears in a run's
    top_entities_json. This means:
      - "tell me more about PAL" matches a run about Silvermaster (alias→canonical)
      - "the Silvermaster search" matches a run tagged with PAL (canonical→alias)

    Returns: {ref_run_id, ref_evidence_set_id, confidence, method}
    """
    msg_lower = user_message.lower()

    # Check for explicit back-references
    for cue in _BACK_REFERENCE_CUES:
        if cue in msg_lower:
            # Default to most recent run
            if context.runs:
                run = context.runs[0]  # most recent (ordered DESC)
                return {
                    "ref_run_id": run.run_id,
                    "ref_evidence_set_id": run.evidence_set_id,
                    "confidence": 0.7,
                    "method": f"back_reference_cue:{cue}",
                }

    # Check for entity overlap with recent runs (bidirectional: name AND alias)
    for run in context.runs:
        if not run.top_entities_json:
            continue
        for entity in run.top_entities_json:
            # Build the full set of name forms (canonical + all aliases)
            all_names = _build_entity_name_set(entity)
            for name_lower in all_names:
                if name_lower in msg_lower:
                    # Found a match — determine which name form matched
                    matched_name = name_lower
                    canonical = entity.get("canonical_name", matched_name)
                    return {
                        "ref_run_id": run.run_id,
                        "ref_evidence_set_id": run.evidence_set_id,
                        "confidence": 0.8,
                        "method": f"entity_overlap:{matched_name} (canonical={canonical})",
                    }

    # Check for label match
    for run in context.runs:
        if run.label and run.label.lower() in msg_lower:
            return {
                "ref_run_id": run.run_id,
                "ref_evidence_set_id": run.evidence_set_id,
                "confidence": 0.9,
                "method": f"label_match:{run.label}",
            }

    # Check for query_text overlap (partial match on previous queries)
    for run in context.runs:
        if run.query_text:
            # Extract significant words from the previous query
            import re as _re
            prev_words = {w.lower() for w in _re.findall(r"[A-Za-z]{3,}", run.query_text)}
            msg_words = {w.lower() for w in _re.findall(r"[A-Za-z]{3,}", user_message)}
            # Require at least 2 significant word overlap
            overlap = prev_words & msg_words - {"the", "who", "what", "was", "were", "about", "from", "with"}
            if len(overlap) >= 2:
                return {
                    "ref_run_id": run.run_id,
                    "ref_evidence_set_id": run.evidence_set_id,
                    "confidence": 0.6,
                    "method": f"query_text_overlap:{overlap}",
                }

    # Default to active run if exists
    if context.active_run_id and context.active_evidence_set_id:
        return {
            "ref_run_id": context.active_run_id,
            "ref_evidence_set_id": context.active_evidence_set_id,
            "confidence": 0.5,
            "method": "default_active",
        }

    return {"ref_run_id": None, "ref_evidence_set_id": None, "confidence": 0.0, "method": "none"}


# =============================================================================
# Intent classification (LLM-based)
# =============================================================================

_ROUTER_SCHEMA = {
    "name": "intent_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["new_retrieval", "follow_up", "think_deeper"],
                "description": (
                    "new_retrieval: user asks a new question that requires searching the archive. "
                    "follow_up: user asks about something already retrieved (clarification, detail, "
                    "explanation of existing evidence). "
                    "think_deeper: user explicitly asks to continue/extend a previous search."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the classification.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence 0-1.",
            },
        },
        "required": ["intent", "reasoning", "confidence"],
        "additionalProperties": False,
    },
}

_ROUTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": _ROUTER_SCHEMA,
}

_ROUTER_SYSTEM = """\
You classify user messages in a historical research assistant.

Context: The user has a session with previous search runs. Each run searched \
an archive of declassified intelligence documents and produced an evidence set.

Classify the user's intent:

- **new_retrieval**: The user asks a NEW question that requires searching the archive. \
  Examples: "Who was PAL?", "What evidence links Silvermaster to GRU?", \
  "Find mentions of atomic espionage in Venona"
- **follow_up**: The user asks about something ALREADY retrieved — a clarification, \
  detail, or explanation of existing evidence. No new search needed. \
  Examples: "What does that quote mean?", "Who is 'he' in that passage?", \
  "Tell me more about the third result", "Can you explain that?"
- **think_deeper**: The user EXPLICITLY asks to continue, extend, or deepen a previous \
  search. Examples: "Think deeper", "Keep searching", "Dig deeper", "Find more on this"

Rules:
- Be conservative: if unsure between follow_up and new_retrieval, choose new_retrieval.
- think_deeper requires explicit intent to continue searching, not just "more detail".
- follow_up requires that an evidence set exists to answer from."""


def classify_intent(
    user_message: str,
    context: RecentQueryContext,
    *,
    model: str = "gpt-4.1-mini-2025-04-14",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Classify user intent using a small LLM call.

    Returns: {"intent": str, "reasoning": str, "confidence": float}
    """
    # Fast path: check for explicit think_deeper cues first
    msg_lower = user_message.lower().strip()
    for cue in _THINK_DEEPER_CUES:
        if cue in msg_lower:
            return {
                "intent": "think_deeper",
                "reasoning": f"Explicit cue: '{cue}'",
                "confidence": 0.95,
            }

    # Build context summary for the router
    context_parts = []
    if context.runs:
        context_parts.append("Recent searches in this session:")
        for r in context.runs[:5]:
            status_note = f" [{r.status}]" if r.status != "completed" else ""
            label_note = f' "{r.label}"' if r.label else ""
            context_parts.append(
                f"  Q{r.query_index}: {r.query_text[:80]}{label_note}{status_note}"
            )
            if r.evidence_summary:
                context_parts.append(f"    Summary: {r.evidence_summary[:100]}")
    else:
        context_parts.append("No previous searches in this session.")

    if context.active_run_status == "paused":
        context_parts.append(f"\nActive run is PAUSED (can be resumed with think_deeper).")
    elif context.active_evidence_set_id:
        context_parts.append(f"\nActive evidence set exists (follow_up is possible).")

    context_text = "\n".join(context_parts)

    # LLM call
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: heuristic-only
        return _heuristic_classify(user_message, context)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": (
                    f"Session context:\n{context_text}\n\n"
                    f"User message: {user_message}\n\n"
                    f"Classify intent."
                )},
            ],
            temperature=0.0,
            max_completion_tokens=150,
            response_format=_ROUTER_RESPONSE_FORMAT,
        )
        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
            if verbose:
                print(f"  [V9 Router] LLM: {result}", file=sys.stderr)
            return result
    except Exception as e:
        if verbose:
            print(f"  [V9 Router] LLM error: {e}, falling back to heuristic", file=sys.stderr)

    return _heuristic_classify(user_message, context)


def _heuristic_classify(
    user_message: str,
    context: RecentQueryContext,
) -> Dict[str, Any]:
    """Fallback heuristic classification when LLM is unavailable."""
    msg_lower = user_message.lower()

    # Follow-up cues
    for cue in _FOLLOW_UP_CUES:
        if cue in msg_lower:
            if context.active_evidence_set_id:
                return {
                    "intent": "follow_up",
                    "reasoning": f"Follow-up cue: '{cue}'",
                    "confidence": 0.7,
                }

    # Default: new retrieval
    return {
        "intent": "new_retrieval",
        "reasoning": "No follow-up or think-deeper cues detected; defaulting to new retrieval.",
        "confidence": 0.6,
    }


# =============================================================================
# Main router entry point
# =============================================================================

def route_message(
    user_message: str,
    context: RecentQueryContext,
    *,
    explicit_action: Optional[str] = None,  # "think_deeper" from API
    intent_hint: Optional[str] = None,  # "new_retrieval" from escalation "Start new search"
    carry_context: Optional[Dict[str, Any]] = None,  # run_id, evidence_set_id from UI
    verbose: bool = False,
) -> RouterDecision:
    """Route a user message to the appropriate execution path.

    Args:
        user_message: the user's text
        context: recent query context from the session
        explicit_action: override from API (e.g. "think_deeper" button)
        intent_hint: from carry_context (e.g. "new_retrieval" from escalation)
        verbose: log to stderr

    Returns:
        RouterDecision with intent, targets, and reasoning
    """
    # Step 1: Resolve query references
    ref = resolve_query_reference(user_message, context)

    # Step 2: Intent hint override (escalation "Start new search" forces fresh retrieval)
    if intent_hint == "new_retrieval":
        return RouterDecision(
            intent="new_retrieval",
            target_run_id=None,
            target_evidence_set_id=None,
            confidence=1.0,
            reasoning="Explicit new_retrieval from escalation button.",
            query_text=user_message,
            ref_run_id=ref.get("ref_run_id"),
            ref_evidence_set_id=ref.get("ref_evidence_set_id"),
        )

    # Step 3: Explicit action override
    if explicit_action == "think_deeper":
        carry = carry_context or {}
        target_run = carry.get("run_id") or ref.get("ref_run_id") or context.active_run_id
        target_es = carry.get("evidence_set_id") or ref.get("ref_evidence_set_id") or context.active_evidence_set_id
        return RouterDecision(
            intent="think_deeper",
            target_run_id=target_run,
            target_evidence_set_id=target_es,
            confidence=1.0,
            reasoning="Explicit think_deeper action from user/UI.",
            query_text=user_message,
            ref_run_id=ref.get("ref_run_id"),
            ref_evidence_set_id=ref.get("ref_evidence_set_id"),
        )

    # Step 4: LLM intent classification
    classification = classify_intent(user_message, context, verbose=verbose)
    intent = classification.get("intent", "new_retrieval")
    confidence = classification.get("confidence", 0.5)
    reasoning = classification.get("reasoning", "")

    # Step 5: Validate and build decision
    if intent == "think_deeper":
        target_run = ref.get("ref_run_id") or context.active_run_id
        target_es = ref.get("ref_evidence_set_id") or context.active_evidence_set_id
        if not target_run:
            # Can't think deeper without a run to resume
            intent = "new_retrieval"
            reasoning += " (No run to resume; falling back to new_retrieval)"

    elif intent == "follow_up":
        target_es = ref.get("ref_evidence_set_id") or context.active_evidence_set_id
        target_run = ref.get("ref_run_id") or context.active_run_id
        if not target_es:
            # Hard rule: follow_up requires evidence set
            intent = "new_retrieval"
            reasoning += " (No evidence set; falling back to new_retrieval)"

    else:
        # new_retrieval
        target_run = None
        target_es = None

    if verbose:
        print(
            f"  [V9 Router] intent={intent}, confidence={confidence:.2f}, "
            f"ref_run={ref.get('ref_run_id')}, ref_es={ref.get('ref_evidence_set_id')}, "
            f"method={ref.get('method')}",
            file=sys.stderr,
        )

    return RouterDecision(
        intent=intent,
        target_run_id=target_run if intent != "new_retrieval" else None,
        target_evidence_set_id=target_es if intent != "new_retrieval" else None,
        confidence=confidence,
        reasoning=reasoning,
        query_text=user_message,
        ref_run_id=ref.get("ref_run_id"),
        ref_evidence_set_id=ref.get("ref_evidence_set_id"),
    )
