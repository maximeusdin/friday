"""
Prompt Template v1 for LLM Summarization

FROZEN - DO NOT EDIT
Created: 2024-02
Version: summ_v1

To make changes, create v2.py instead.
"""

from typing import List, Optional

PROMPT_VERSION = "summ_v1"

SYSTEM_PROMPT = """You are a research assistant helping historians analyze archival documents from Cold War era collections.

Your task is to synthesize evidence from search results into clear, factual claims with citations.

CRITICAL RULES:
1. Every claim MUST cite at least one evidence bundle using [B1], [B2], etc.
2. Only make claims that are directly supported by the provided evidence.
3. Do not invent names, dates, places, or facts not found in the evidence.
4. If evidence is ambiguous or contradictory, set confidence to "low" and explain.
5. Output valid JSON matching the required schema exactly.

CONFIDENCE LEVELS:
- "high": Multiple pieces of evidence from different documents support this claim
- "medium": Evidence supports the claim but from limited sources
- "low": Evidence is ambiguous, contradictory, or single-source speculation"""


def build_user_prompt(
    question: Optional[str],
    bundles_text: str,
    bundle_count: int,
    max_claims: int = 10,
    include_themes: bool = True,
) -> str:
    """
    Build the user prompt for LLM synthesis.
    
    Args:
        question: Optional user question to answer
        bundles_text: Formatted evidence bundles
        bundle_count: Number of bundles provided
        max_claims: Maximum number of claims to generate
        include_themes: Whether to identify themes
    
    Returns:
        Formatted user prompt string
    """
    
    question_section = ""
    if question:
        question_section = f"""
USER QUESTION:
{question}

Focus your claims on answering this question based on the evidence provided.
"""
    else:
        question_section = """
No specific question provided. Summarize the key findings from the evidence.
"""

    themes_instruction = ""
    if include_themes:
        themes_instruction = """
"themes": [
    {
      "theme": "Brief theme title",
      "description": "One sentence description",
      "evidence": ["B1", "B5"]  // Bundle IDs supporting this theme
    }
  ],"""

    return f"""{question_section}
EVIDENCE ({bundle_count} bundles):
{bundles_text}

OUTPUT REQUIREMENTS:
Generate a JSON object with the following structure:

{{
  "claims": [
    {{
      "claim_id": "C1",
      "claim": "A clear factual claim supported by evidence",
      "citations": ["B1", "B3"],  // REQUIRED: At least one bundle ID
      "confidence": "high|medium|low",
      "limitations": "Optional note about evidence gaps or caveats"
    }}
  ],
  {themes_instruction}
  "entities_mentioned": ["Name1", "Name2"],  // Key people/orgs/places mentioned
  "coverage_notes": "Optional note about what the evidence covers",
  "followups": ["Suggested follow-up question 1", "Suggested follow-up question 2"]
}}

IMPORTANT:
- Generate up to {max_claims} claims maximum
- Every claim MUST have at least one citation in the "citations" array
- Citations must be valid bundle IDs from the evidence (B1, B2, etc. up to B{bundle_count})
- Do not cite bundle IDs that don't exist
- If you cannot make any supported claims, return an empty claims array

Respond ONLY with the JSON object, no additional text."""
