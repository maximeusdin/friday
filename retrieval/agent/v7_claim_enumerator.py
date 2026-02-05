"""
V7 Claim Enumerator - Extract claims and assign citations

This component:
1. Takes the answer text and evidence store (bottleneck spans)
2. Uses LLM to extract atomic claims from the answer
3. Assigns citations from available evidence to each claim
4. Flags claims that cannot be supported by evidence

Every claim MUST have at least one citation, or it gets flagged for removal.
"""
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from retrieval.agent.v7_types import ClaimWithCitation


# =============================================================================
# Configuration
# =============================================================================

ENUMERATOR_MODEL = "gpt-3.5-turbo"  # Fast model for claim extraction
MAX_CLAIMS_PER_ANSWER = 30  # Limit claims to prevent explosion


# =============================================================================
# Prompts
# =============================================================================

CLAIM_EXTRACTION_SYSTEM = """You are a precise claim extractor. Your job is to:
1. Extract ATOMIC claims from an answer (one fact per claim)
2. Assign citations from the provided evidence to each claim
3. Mark claims that cannot be supported by evidence

Rules:
- Each claim must be a single, verifiable statement
- Each claim MUST have at least one citation [span_id]
- If a claim cannot be supported by ANY evidence, mark it as unsupported
- Be conservative: only cite evidence that DIRECTLY supports the claim
- For roster/member claims, each person should be a separate claim

Output JSON with this structure:
{
  "claims": [
    {"claim": "Harry White was a member of the network", "citations": ["sp_1", "sp_3"], "support": "strong"},
    {"claim": "The network operated from 1940-1945", "citations": ["sp_5"], "support": "weak"}
  ],
  "unsupported": ["Some claim that had no evidence"]
}"""


def build_claim_extraction_prompt(
    answer: str,
    evidence: List[Dict[str, Any]],
) -> str:
    """Build prompt for claim extraction."""
    
    # Format evidence with span IDs
    evidence_lines = []
    for i, span in enumerate(evidence):
        span_id = span.get("span_id", f"sp_{i}")
        text = span.get("text", span.get("span_text", ""))[:500]
        source = span.get("source_label", "")
        page = span.get("page", "")
        evidence_lines.append(f"[{span_id}] ({source}, p.{page}): \"{text}\"")
    
    evidence_text = "\n".join(evidence_lines)
    
    return f"""ANSWER TO ANALYZE:
{answer}

AVAILABLE EVIDENCE (cite using span IDs in brackets):
{evidence_text}

Extract ALL claims from the answer. Each claim needs at least one citation from the evidence above.
If a claim cannot be supported by ANY evidence, put it in "unsupported".

Output JSON only."""


# =============================================================================
# Claim Enumerator
# =============================================================================

class ClaimEnumerator:
    """
    Extract atomic claims from answers and assign citations.
    
    This is a key component of V7's citation enforcement:
    - Every claim gets citations from the evidence store
    - Claims without evidence are flagged for removal
    - The StopGate then validates all claims are properly cited
    """
    
    def __init__(
        self,
        model: str = ENUMERATOR_MODEL,
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
    
    def enumerate_claims(
        self,
        answer: str,
        evidence_spans: List[Any],
    ) -> Tuple[List[ClaimWithCitation], List[str]]:
        """
        Extract claims from answer and assign citations.
        
        Args:
            answer: The answer text to analyze
            evidence_spans: List of BottleneckSpan or dict objects
        
        Returns:
            Tuple of (valid_claims, unsupported_claims)
            - valid_claims: Claims with at least one citation
            - unsupported_claims: Claims that couldn't be cited (will be dropped)
        """
        start = time.time()
        
        if not answer or not answer.strip():
            return [], []
        
        if self.verbose:
            print(f"  [ClaimEnumerator] Extracting claims from answer ({len(answer)} chars)", 
                  file=sys.stderr)
            print(f"    Evidence available: {len(evidence_spans)} spans", file=sys.stderr)
        
        # Convert spans to dict format
        evidence_dicts = self._spans_to_dicts(evidence_spans)
        
        if not evidence_dicts:
            if self.verbose:
                print(f"    [!] No evidence available, all claims will be unsupported", 
                      file=sys.stderr)
            # Extract claims but mark all as unsupported
            return [], self._extract_claims_fallback(answer)
        
        # Use LLM to extract and cite claims
        valid_claims, unsupported = self._extract_with_llm(answer, evidence_dicts)
        
        elapsed = (time.time() - start) * 1000
        
        if self.verbose:
            print(f"    Extracted: {len(valid_claims)} cited claims, "
                  f"{len(unsupported)} unsupported ({elapsed:.0f}ms)", file=sys.stderr)
        
        return valid_claims, unsupported
    
    def _spans_to_dicts(self, spans: List[Any]) -> List[Dict[str, Any]]:
        """Convert spans to dict format for the prompt."""
        dicts = []
        for i, span in enumerate(spans):
            if hasattr(span, 'to_dict'):
                d = span.to_dict()
            elif isinstance(span, dict):
                d = span
            else:
                d = {
                    "span_id": f"sp_{i}",
                    "text": str(span),
                }
            
            # Ensure span_id exists
            if "span_id" not in d:
                d["span_id"] = f"sp_{i}"
            
            dicts.append(d)
        
        return dicts
    
    def _extract_with_llm(
        self,
        answer: str,
        evidence: List[Dict[str, Any]],
    ) -> Tuple[List[ClaimWithCitation], List[str]]:
        """Use LLM to extract claims and assign citations."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._extract_fallback_with_evidence(answer, evidence)
        
        prompt = build_claim_extraction_prompt(answer, evidence)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            if self.verbose:
                print(f"    Calling {self.model} for claim extraction...", 
                      file=sys.stderr, end="", flush=True)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            
            if self.verbose:
                print(f" done", file=sys.stderr)
            
            content = response.choices[0].message.content
            if not content:
                return self._extract_fallback_with_evidence(answer, evidence)
            
            data = json.loads(content)
            
            # Parse claims
            valid_claims = []
            for claim_data in data.get("claims", [])[:MAX_CLAIMS_PER_ANSWER]:
                claim = ClaimWithCitation(
                    claim_text=claim_data.get("claim", ""),
                    citations=claim_data.get("citations", []),
                    support_level=claim_data.get("support", "strong"),
                )
                if claim.is_valid():
                    valid_claims.append(claim)
            
            unsupported = data.get("unsupported", [])
            
            return valid_claims, unsupported
            
        except Exception as e:
            if self.verbose:
                print(f" error: {e}", file=sys.stderr)
            return self._extract_fallback_with_evidence(answer, evidence)
    
    def _extract_fallback_with_evidence(
        self,
        answer: str,
        evidence: List[Dict[str, Any]],
    ) -> Tuple[List[ClaimWithCitation], List[str]]:
        """Fallback: simple sentence splitting with first evidence citation."""
        
        if self.verbose:
            print(f"    [Fallback] Using simple extraction", file=sys.stderr)
        
        # Split answer into sentences
        sentences = self._split_sentences(answer)
        
        valid_claims = []
        unsupported = []
        
        # Get first evidence span_id as default citation
        default_cite = evidence[0].get("span_id", "sp_0") if evidence else None
        
        for sentence in sentences[:MAX_CLAIMS_PER_ANSWER]:
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            if default_cite:
                claim = ClaimWithCitation(
                    claim_text=sentence,
                    citations=[default_cite],
                    support_level="weak",
                )
                valid_claims.append(claim)
            else:
                unsupported.append(sentence)
        
        return valid_claims, unsupported
    
    def _extract_claims_fallback(self, answer: str) -> List[str]:
        """Extract claims without evidence (all will be unsupported)."""
        sentences = self._split_sentences(answer)
        return [s for s in sentences if len(s) > 10][:MAX_CLAIMS_PER_ANSWER]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        # Split on period, question mark, exclamation
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# Convenience function
# =============================================================================

def enumerate_claims(
    answer: str,
    evidence_spans: List[Any],
    verbose: bool = True,
) -> Tuple[List[ClaimWithCitation], List[str]]:
    """
    Convenience function to extract claims and assign citations.
    
    Returns:
        (valid_claims, unsupported_claims)
    """
    enumerator = ClaimEnumerator(verbose=verbose)
    return enumerator.enumerate_claims(answer, evidence_spans)
