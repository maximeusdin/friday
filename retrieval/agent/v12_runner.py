"""
V12 runner — clarification round-trip in front of the V11 investigation loop.

Flow (initial query submission only, per current scope):
  1. First pass (no answers yet): generate 0–3 follow-up questions. If any, return
     a V12ClarificationPending (the caller surfaces the questions to the user and
     stores the plan in the message metadata). If none, fall straight through to
     the normal V11 investigation.
  2. Resume pass (answers + the stored plan): incorporate answers -> augmented
     question + accepted-identity seeds + clarification notes -> run V11.

This keeps all V11 behavior intact; v12 only adds an optional pre-step.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from retrieval.agent.v12_clarifier import (
    ClarificationPlan, ClarificationAnswer, generate_clarifications, incorporate_answers,
)
from retrieval.agent.v11_runner import run_v11_query


@dataclass
class V12ClarificationPending:
    """Returned when the agent wants the user to answer follow-up question(s)
    before the investigation starts."""
    plan: ClarificationPlan
    question: str

    needs_clarification: bool = True

    def to_dict(self):
        return {"needs_clarification": True, "question": self.question, **self.plan.to_dict()}


def run_v12_query(
    conn,
    question: str,
    *,
    clarification_answers: Optional[List[ClarificationAnswer]] = None,
    clarification_plan: Optional[ClarificationPlan] = None,
    max_questions: int = 3,
    use_llm: bool = True,
    **v11_kwargs: Any,
) -> Union[V12ClarificationPending, Any]:
    """Returns V12ClarificationPending (ask the user) OR a V9Result (investigation ran)."""
    answered = clarification_answers is not None

    if not answered:
        plan = generate_clarifications(conn, question, max_questions=max_questions, use_llm=use_llm)
        if plan.needed:
            return V12ClarificationPending(plan=plan, question=question)
        # nothing worth asking -> straight to investigation
        return run_v11_query(conn, question, **v11_kwargs)

    # Resume: fold the answers into the investigation seed.
    plan = clarification_plan or ClarificationPlan(questions=[])
    outcome = incorporate_answers(conn, question, plan, clarification_answers)
    return run_v11_query(
        conn,
        outcome.augmented_question,
        seed_entity_candidates=outcome.seed_entities,
        clarification_notes=outcome.clarification_notes,
        **v11_kwargs,
    )
