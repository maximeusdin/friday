"""
V5 Grader - LLM-based evidence evaluation

The Grader is the sole authority on whether a span is good evidence.
Uses gpt-4o-mini for speed (as specified in the design).

No heuristics, no regex, no keyword overlap - pure semantic judgment.
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from retrieval.agent.v5_types import (
    CandidateSpan,
    GraderResult,
    EvidenceItem,
    EvidenceStatus,
)


# =============================================================================
# Configuration
# =============================================================================

GRADER_MODEL = "gpt-4o-mini"  # Fast model to avoid bottlenecks
GRADER_BATCH_SIZE = 10  # Grade this many spans per LLM call
GRADER_TEMPERATURE = 0.1  # Low temp for consistency


# =============================================================================
# Grader Prompt
# =============================================================================

GRADER_SYSTEM_PROMPT = """You are an evidence grader for historical research questions.

Your job: evaluate whether a text span is useful evidence for answering a specific question.

You must be:
- CONSERVATIVE: Only mark as supporting if the span DIRECTLY addresses the question
- PRECISE: Identify exactly what claim the span supports (if any)
- HONEST: If the span is tangential or irrelevant, say so

For each span, provide structured JSON output."""


def build_grader_prompt(question: str, candidates: List[CandidateSpan]) -> str:
    """Build prompt for grading a batch of candidate spans."""
    
    spans_section = []
    for i, cand in enumerate(candidates):
        text = cand.span_text[:800]  # Truncate long spans
        context = cand.surrounding_context[:200] if cand.surrounding_context else ""
        
        span_entry = f"""
SPAN {i} (id: {cand.candidate_id}):
Source: {cand.source_label or 'unknown'}
Text: "{text}"
{f'Context: "{context}"' if context else ''}
"""
        spans_section.append(span_entry)
    
    spans_text = "\n---\n".join(spans_section)
    
    return f"""QUESTION: {question}

CANDIDATE SPANS TO GRADE:
{spans_text}

For EACH span, output a JSON object with these fields:
- "candidate_id": the span's ID (string)
- "supports_question": does this span help answer the question? (boolean)
- "support_strength": 0-3 scale (0=none, 1=weak/tangential, 2=moderate/relevant, 3=strong/direct answer)
- "quote_grade": is this span directly quotable as evidence? (boolean)
- "claim_supported": if supports_question is true, what specific claim does it support? (short string, 1 sentence max)
- "notes": brief explanation of your judgment (1 sentence)

Output a JSON object with a "grades" array containing one object per span.

Example output format:
{{"grades": [
  {{"candidate_id": "abc123", "supports_question": true, "support_strength": 2, "quote_grade": true, "claim_supported": "X was a member of Y organization", "notes": "Directly states membership."}},
  {{"candidate_id": "def456", "supports_question": false, "support_strength": 0, "quote_grade": false, "claim_supported": "", "notes": "Discusses unrelated topic."}}
]}}

BE CONSERVATIVE. If uncertain, rate lower. Better to miss some evidence than to include bad evidence."""


# =============================================================================
# Grader Class
# =============================================================================

class Grader:
    """LLM-based evidence grader using gpt-4o-mini."""
    
    def __init__(
        self,
        model: str = GRADER_MODEL,
        batch_size: int = GRADER_BATCH_SIZE,
        verbose: bool = True,
    ):
        self.model = model
        self.batch_size = batch_size
        self.verbose = verbose
        self.total_calls = 0
        self.total_candidates_graded = 0
    
    def grade_candidates(
        self,
        question: str,
        candidates: List[CandidateSpan],
    ) -> List[GraderResult]:
        """
        Grade a list of candidate spans.
        
        Batches candidates for efficiency.
        Returns GraderResult for each candidate.
        """
        if not candidates:
            return []
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            batch_results = self._grade_batch(question, batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def _grade_batch(
        self,
        question: str,
        batch: List[CandidateSpan],
    ) -> List[GraderResult]:
        """Grade a single batch of candidates."""
        
        start = time.time()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if self.verbose:
                print("    [Grader] No API key, using fallback grades", file=sys.stderr)
            return self._fallback_grades(batch)
        
        prompt = build_grader_prompt(question, batch)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=GRADER_TEMPERATURE,
                max_tokens=2000,
            )
            
            self.total_calls += 1
            elapsed = (time.time() - start) * 1000
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_grades(batch)
            
            data = json.loads(content)
            grades_data = data.get("grades", [])
            
            # Map results back to candidates
            results = []
            grades_by_id = {g.get("candidate_id"): g for g in grades_data}
            
            for cand in batch:
                grade_data = grades_by_id.get(cand.candidate_id, {})
                result = GraderResult(
                    candidate_id=cand.candidate_id,
                    supports_question=grade_data.get("supports_question", False),
                    support_strength=grade_data.get("support_strength", 0),
                    quote_grade=grade_data.get("quote_grade", False),
                    claim_supported=grade_data.get("claim_supported", ""),
                    notes=grade_data.get("notes", "No grade returned"),
                )
                results.append(result)
                self.total_candidates_graded += 1
            
            if self.verbose:
                supporting = sum(1 for r in results if r.supports_question)
                strong = sum(1 for r in results if r.support_strength >= 2)
                print(f"    [Grader] Batch: {len(batch)} spans -> {supporting} supporting, {strong} strong ({elapsed:.0f}ms)", 
                      file=sys.stderr)
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"    [Grader] Error: {e}", file=sys.stderr)
            return self._fallback_grades(batch)
    
    def _fallback_grades(self, batch: List[CandidateSpan]) -> List[GraderResult]:
        """Return conservative fallback grades when LLM fails."""
        results = []
        for cand in batch:
            results.append(GraderResult(
                candidate_id=cand.candidate_id,
                supports_question=False,
                support_strength=0,
                quote_grade=False,
                claim_supported="",
                notes="Fallback: grading failed",
            ))
            self.total_candidates_graded += 1
        return results
    
    def grade_to_evidence_item(
        self,
        candidate: CandidateSpan,
        grade: GraderResult,
        current_step: int,
    ) -> Optional[EvidenceItem]:
        """
        Convert a graded candidate to an EvidenceItem if it's worth keeping.
        
        Returns None if the candidate shouldn't be added to the store.
        """
        # Only keep candidates that support the question
        if not grade.supports_question:
            return None
        
        # Create the evidence item
        return EvidenceItem(
            evidence_id=f"ev_{candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            span_text=candidate.span_text,
            chunk_id=candidate.chunk_id,
            doc_id=candidate.doc_id,
            page=candidate.page,
            source_label=candidate.source_label,
            support_strength=grade.support_strength,
            quote_grade=grade.quote_grade,
            claim_supported=grade.claim_supported,
            grader_notes=grade.notes,
            status=EvidenceStatus.ACTIVE,
            added_at_step=current_step,
        )


# =============================================================================
# Pairwise Grader (Alternative - more stable)
# =============================================================================

PAIRWISE_SYSTEM_PROMPT = """You are comparing two pieces of evidence for a research question.
Your job: decide which piece of evidence is MORE useful for answering the question.
Be decisive. If they're roughly equal, pick the one that's more directly relevant."""


def build_pairwise_prompt(question: str, span_a: str, span_b: str) -> str:
    """Build prompt for pairwise comparison."""
    return f"""QUESTION: {question}

EVIDENCE A:
"{span_a[:600]}"

EVIDENCE B:
"{span_b[:600]}"

Which is better evidence for answering the question?
Output JSON: {{"winner": "A" or "B", "reason": "brief explanation"}}"""


class TournamentGrader:
    """
    Tournament-style grader using pairwise comparisons.
    
    This is the SCALABLE version of rating - instead of scoring everything,
    compare each new candidate to the current worst span in the store.
    If better, swap. This mimics ChatGPT's "constantly replacing weaker context."
    
    More stable than absolute scoring - LLMs are better at comparisons.
    """
    
    def __init__(self, model: str = GRADER_MODEL, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.total_comparisons = 0
        self.wins = 0
        self.losses = 0
    
    def compare(
        self,
        question: str,
        new_span_text: str,
        existing_span_text: str,
        new_claim: str = "",
        existing_claim: str = "",
    ) -> Dict[str, Any]:
        """
        Compare a new candidate against existing evidence.
        
        Returns: {"winner": "new" or "existing", "reason": str, "confidence": float}
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback: assume new is not better (conservative)
            return {"winner": "existing", "reason": "no API key", "confidence": 0.5}
        
        prompt = self._build_tournament_prompt(
            question, new_span_text, existing_span_text, new_claim, existing_claim
        )
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300,
            )
            
            self.total_comparisons += 1
            
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                winner_raw = data.get("winner", "B")
                
                # Map A/B to new/existing
                winner = "new" if winner_raw == "A" else "existing"
                
                if winner == "new":
                    self.wins += 1
                else:
                    self.losses += 1
                
                return {
                    "winner": winner,
                    "reason": data.get("reason", ""),
                    "confidence": data.get("confidence", 0.7),
                }
            
        except Exception as e:
            if self.verbose:
                print(f"    [Tournament] Error: {e}", file=sys.stderr)
        
        # Fallback: conservative
        return {"winner": "existing", "reason": "comparison failed", "confidence": 0.5}
    
    def _build_tournament_prompt(
        self,
        question: str,
        new_text: str,
        existing_text: str,
        new_claim: str,
        existing_claim: str,
    ) -> str:
        """Build prompt for tournament comparison."""
        
        claim_a = f"\nClaim A supports: {new_claim}" if new_claim else ""
        claim_b = f"\nClaim B supports: {existing_claim}" if existing_claim else ""
        
        return f"""QUESTION: {question}

EVIDENCE A (new candidate):
"{new_text[:600]}"
{claim_a}

EVIDENCE B (currently in store):
"{existing_text[:600]}"
{claim_b}

Which is BETTER evidence for answering the question?
Consider:
- Specificity (names, dates, facts vs general statements)
- Directness (does it answer the question or is it tangential?)
- Quotability (can you cite this directly?)

Output JSON:
{{
  "winner": "A" or "B",
  "reason": "brief explanation",
  "confidence": 0.0-1.0
}}

Be decisive. If roughly equal, prefer the more specific one."""
    
    def should_replace(
        self,
        question: str,
        new_span_text: str,
        worst_span_text: str,
        new_claim: str = "",
        worst_claim: str = "",
        confidence_threshold: float = 0.6,
    ) -> Tuple[bool, str]:
        """
        Decide if new span should replace the worst in store.
        
        Returns: (should_replace: bool, reason: str)
        """
        result = self.compare(
            question, new_span_text, worst_span_text, new_claim, worst_claim
        )
        
        if result["winner"] == "new" and result["confidence"] >= confidence_threshold:
            return True, result["reason"]
        else:
            return False, result["reason"]


# Alias for backwards compatibility
PairwiseGrader = TournamentGrader
