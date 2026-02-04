"""
V6 Responsiveness Verifier - Does the answer satisfy the question?

Current problem: "verification passed" when answer admits it didn't find members.

Responsiveness checks (all via LLM, no heuristics):
- Does the answer actually deliver what was asked?
- For roster: is it a list of people with citations?
- If can't answer: does it say "insufficient evidence" rather than hedging?

This is separate from grounding verification - you can have a grounded but non-responsive answer.
"""
import os
import json
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from retrieval.agent.v6_query_parser import ParsedQuery, TaskType


# =============================================================================
# Responsiveness Status
# =============================================================================

class ResponsivenessStatus(Enum):
    RESPONSIVE = "responsive"  # Answer satisfies the question
    PARTIALLY_RESPONSIVE = "partially_responsive"  # Partial answer
    NOT_RESPONSIVE = "not_responsive"  # Answer doesn't satisfy
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # Correctly says "can't answer"


# =============================================================================
# Responsiveness Result
# =============================================================================

@dataclass
class ResponsivenessResult:
    """Result of responsiveness verification."""
    
    status: ResponsivenessStatus = ResponsivenessStatus.NOT_RESPONSIVE
    
    # Detailed checks
    delivers_requested_format: bool = False  # List for roster, timeline for timeline
    items_have_citations: bool = False  # Each item cites evidence
    items_are_correct_type: bool = False  # People for roster, dates for timeline
    acknowledges_limitations: bool = False  # Says what it couldn't find
    
    # For roster queries
    members_listed: List[str] = field(default_factory=list)
    members_with_citations: List[str] = field(default_factory=list)
    generic_items: List[str] = field(default_factory=list)  # Orgs listed as "members"
    
    # Explanation
    explanation: str = ""
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "delivers_requested_format": self.delivers_requested_format,
            "items_have_citations": self.items_have_citations,
            "items_are_correct_type": self.items_are_correct_type,
            "acknowledges_limitations": self.acknowledges_limitations,
            "members_listed": self.members_listed,
            "members_with_citations": self.members_with_citations,
            "generic_items": self.generic_items,
            "explanation": self.explanation,
            "issues": self.issues,
        }


# =============================================================================
# Responsiveness Prompt
# =============================================================================

RESPONSIVENESS_SYSTEM_PROMPT = """You are verifying that an answer is RESPONSIVE to the question asked.

This is NOT about whether the answer is grounded in evidence (that's checked separately).
This is about whether the answer DELIVERS what was requested.

For roster queries ("who were members"):
- RESPONSIVE: Lists specific people with citations
- NOT RESPONSIVE: Admits "I couldn't find members" or lists only organizations
- PARTIALLY RESPONSIVE: Lists some people but with caveats

For timeline queries ("when did X happen"):
- RESPONSIVE: Provides dates/sequence
- NOT RESPONSIVE: Discusses X without dates

Key principle: An answer that says "I couldn't find sufficient evidence" is CORRECTLY 
non-responsive - that's better than pretending to answer with weak evidence."""


def build_responsiveness_prompt(
    question: str,
    task_type: TaskType,
    answer: str,
    claims: List[Dict[str, Any]],
) -> str:
    """Build prompt for responsiveness check."""
    
    task_requirements = {
        TaskType.ROSTER_ENUMERATION: """
REQUIRED FOR ROSTER:
- Output must be a list of PEOPLE (not orgs)
- Each person must have at least one citation
- If no members found, must explicitly say "insufficient evidence"
- "Associated with" is NOT the same as "member of"
""",
        TaskType.TIMELINE: """
REQUIRED FOR TIMELINE:
- Output must include dates or temporal sequence
- Events must be ordered chronologically
- Each dated event should have a citation
""",
        TaskType.EVIDENCE_SEARCH: """
REQUIRED FOR EVIDENCE SEARCH:
- Must cite specific documents/sources
- Each piece of evidence should be quoted
""",
    }.get(task_type, "Output must directly answer the question asked.")
    
    claims_text = ""
    if claims:
        claims_lines = []
        for c in claims[:10]:
            claim = c.get("claim", c.get("text", ""))
            citations = c.get("evidence_ids", c.get("citations", []))
            claims_lines.append(f"- {claim[:100]}... (cites: {citations})")
        claims_text = f"\nCLAIMS IN ANSWER:\n" + "\n".join(claims_lines)
    
    return f"""QUESTION: {question}
TASK TYPE: {task_type.value}

{task_requirements}

ANSWER PROVIDED:
{answer}
{claims_text}

Evaluate responsiveness:

1. delivers_requested_format: Does the answer provide what was asked? (list for roster, etc.)
2. items_have_citations: Do the items cite evidence?
3. items_are_correct_type: Are the items the right type? (people for roster, not orgs)
4. acknowledges_limitations: Does it honestly say what it couldn't find?

For roster queries, also extract:
- members_listed: Names listed as members
- members_with_citations: Names that have citations
- generic_items: Items that are orgs/places, not people

Output JSON:
{{
  "status": "responsive" | "partially_responsive" | "not_responsive" | "insufficient_evidence",
  "delivers_requested_format": true/false,
  "items_have_citations": true/false,
  "items_are_correct_type": true/false,
  "acknowledges_limitations": true/false,
  "members_listed": ["Name1", "Name2"],
  "members_with_citations": ["Name1"],
  "generic_items": ["Treasury Department"],
  "explanation": "Brief explanation of verdict",
  "issues": ["Issue 1", "Issue 2"]
}}"""


# =============================================================================
# Responsiveness Verifier
# =============================================================================

class ResponsivenessVerifier:
    """
    Verifies that an answer is responsive to the question.
    
    This catches:
    - Roster answers that list orgs instead of people
    - Answers that admit failure but "pass" grounding checks
    - Hedging answers ("associated with" vs "member of")
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
    
    def verify(
        self,
        answer: str,
        claims: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
    ) -> ResponsivenessResult:
        """
        Verify that the answer is responsive.
        
        Args:
            answer: The generated answer text
            claims: The claims/items in the answer
            parsed_query: The parsed query with task type
        
        Returns:
            ResponsivenessResult with detailed checks
        """
        result = ResponsivenessResult()
        
        # CRITICAL: Pre-check for insufficient evidence answers
        # This prevents the LLM from hallucinating "responsive" for empty results
        insufficient_indicators = [
            "insufficient evidence",
            "no documents matched",
            "no evidence found",
            "could not find",
            "no results found",
            "unable to find evidence",
        ]
        answer_lower = answer.lower()
        if any(indicator in answer_lower for indicator in insufficient_indicators):
            result.status = ResponsivenessStatus.INSUFFICIENT_EVIDENCE
            result.acknowledges_limitations = True
            result.explanation = "Answer correctly acknowledges insufficient evidence"
            if self.verbose:
                print(f"  [Responsiveness] Status: {result.status.value}", file=sys.stderr)
                print(f"    Pre-check: Answer indicates insufficient evidence", file=sys.stderr)
            return result
        
        # Also check if claims list is empty
        if not claims:
            result.status = ResponsivenessStatus.INSUFFICIENT_EVIDENCE
            result.delivers_requested_format = False
            result.items_have_citations = False
            result.explanation = "No claims/evidence provided"
            if self.verbose:
                print(f"  [Responsiveness] Status: {result.status.value}", file=sys.stderr)
                print(f"    Pre-check: No claims in answer", file=sys.stderr)
            return result
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_verify(answer, claims, parsed_query)
        
        prompt = build_responsiveness_prompt(
            question=parsed_query.original_query,
            task_type=parsed_query.task_type,
            answer=answer,
            claims=claims,
        )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RESPONSIVENESS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1000,
            )
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_verify(answer, claims, parsed_query)
            
            data = json.loads(content)
            
            # Map status
            status_str = data.get("status", "not_responsive")
            try:
                result.status = ResponsivenessStatus(status_str)
            except ValueError:
                result.status = ResponsivenessStatus.NOT_RESPONSIVE
            
            result.delivers_requested_format = data.get("delivers_requested_format", False)
            result.items_have_citations = data.get("items_have_citations", False)
            result.items_are_correct_type = data.get("items_are_correct_type", False)
            result.acknowledges_limitations = data.get("acknowledges_limitations", False)
            result.members_listed = data.get("members_listed", [])
            result.members_with_citations = data.get("members_with_citations", [])
            result.generic_items = data.get("generic_items", [])
            result.explanation = data.get("explanation", "")
            result.issues = data.get("issues", [])
            
            if self.verbose:
                print(f"  [Responsiveness] Status: {result.status.value}", file=sys.stderr)
                if result.issues:
                    print(f"    Issues: {result.issues[:3]}", file=sys.stderr)
                if result.members_with_citations:
                    print(f"    Members with citations: {result.members_with_citations[:5]}", file=sys.stderr)
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"  [Responsiveness] Error: {e}", file=sys.stderr)
            return self._fallback_verify(answer, claims, parsed_query)
    
    def _fallback_verify(
        self,
        answer: str,
        claims: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
    ) -> ResponsivenessResult:
        """Simple fallback verification."""
        
        result = ResponsivenessResult()
        answer_lower = answer.lower()
        
        # Check for common non-responsive patterns
        non_responsive_phrases = [
            "i couldn't find",
            "no direct evidence",
            "insufficient evidence",
            "unable to identify",
            "associated with",  # Not the same as "member of"
        ]
        
        if any(phrase in answer_lower for phrase in non_responsive_phrases):
            result.status = ResponsivenessStatus.NOT_RESPONSIVE
            result.acknowledges_limitations = True
            result.explanation = "Contains non-responsive phrases"
            return result
        
        # Check if it's a list (for roster)
        if parsed_query.task_type == TaskType.ROSTER_ENUMERATION:
            # Simple check: has bullet points or numbered list?
            has_list = any(c in answer for c in ["•", "-", "1.", "2."])
            result.delivers_requested_format = has_list
            
            # Check if claims have citations
            if claims:
                cited = [c for c in claims if c.get("evidence_ids") or c.get("citations")]
                result.items_have_citations = len(cited) > 0
        
        result.status = (
            ResponsivenessStatus.RESPONSIVE if result.delivers_requested_format 
            else ResponsivenessStatus.PARTIALLY_RESPONSIVE
        )
        
        return result


# =============================================================================
# Convenience function
# =============================================================================

def verify_responsiveness(
    answer: str,
    claims: List[Dict[str, Any]],
    parsed_query: ParsedQuery,
    verbose: bool = True,
) -> ResponsivenessResult:
    """
    Verify that an answer is responsive to the question.
    
    This should be called AFTER grounding verification.
    An answer can be grounded but not responsive (or vice versa).
    """
    verifier = ResponsivenessVerifier(verbose=verbose)
    return verifier.verify(answer, claims, parsed_query)
