"""
Query Analysis - LLM-driven query decomposition.

Separates scope terms (collections, dates) from evidence terms (anchor concepts).
This replaces heuristic anchor extraction with structured LLM output.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class QueryAnalysis:
    """
    Structured decomposition of a user query.
    
    Produced by LLM in Phase 0 before any retrieval.
    Separates "what to search in" (scope) from "what to search for" (anchors).
    """
    
    # Original query
    query_text: str
    
    # Scope constraints - WHERE to search
    scope_filters: Dict[str, Any] = field(default_factory=dict)
    # Example: {"collections": ["vassiliev"], "date_from": "1944", "date_to": "1945"}
    
    # Core concepts - WHAT the query is asking about (for retrieval query)
    core_concepts: List[str] = field(default_factory=list)
    # Example: ["proximity fuse acquisition", "Soviet espionage atomic"]
    
    # Anchor terms - distinctive terms that MUST appear in evidence
    anchor_terms: List[str] = field(default_factory=list)
    # Example: ["proximity fuse", "vt fuse", "radio proximity"]
    # Guardrails: 1-4 tokens each, no stopwords
    
    # Terms to exclude from anchoring (scope pollution)
    do_not_anchor: List[str] = field(default_factory=list)
    # Example: ["vassiliev notebooks", "soviets", "espionage"]
    
    # Initial synonym suggestions from LLM
    suggested_synonyms: List[str] = field(default_factory=list)
    # Example: ["vt fuse", "proximity fuze", "variable time fuse"]
    
    # Confidence in the analysis (for debugging)
    confidence: float = 1.0
    
    # Raw LLM reasoning (for audit)
    reasoning: str = ""
    
    def get_retrieval_query(self) -> str:
        """Build the query string for retrieval."""
        if self.core_concepts:
            return " ".join(self.core_concepts)
        return self.query_text
    
    def get_all_anchor_terms(self) -> List[str]:
        """Get anchor terms including synonyms."""
        all_terms = list(self.anchor_terms)
        for syn in self.suggested_synonyms:
            if syn not in all_terms:
                all_terms.append(syn)
        return all_terms
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "query_text": self.query_text,
            "scope_filters": self.scope_filters,
            "core_concepts": self.core_concepts,
            "anchor_terms": self.anchor_terms,
            "do_not_anchor": self.do_not_anchor,
            "suggested_synonyms": self.suggested_synonyms,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryAnalysis":
        """Deserialize from dict."""
        return cls(
            query_text=data.get("query_text", ""),
            scope_filters=data.get("scope_filters", {}),
            core_concepts=data.get("core_concepts", []),
            anchor_terms=data.get("anchor_terms", []),
            do_not_anchor=data.get("do_not_anchor", []),
            suggested_synonyms=data.get("suggested_synonyms", []),
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning", ""),
        )


# JSON Schema for OpenAI Structured Outputs
# Note: strict mode requires ALL properties to be in "required"
QUERY_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "scope_filters": {
            "type": "object",
            "properties": {
                "collections": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Collection slugs to search in"
                },
                "date_from": {
                    "type": ["string", "null"],
                    "description": "Start date (YYYY or YYYY-MM-DD)"
                },
                "date_to": {
                    "type": ["string", "null"],
                    "description": "End date (YYYY or YYYY-MM-DD)"
                }
            },
            "additionalProperties": False,
            "required": ["collections", "date_from", "date_to"]
        },
        "core_concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main concepts the query is asking about (1-3 phrases)"
        },
        "anchor_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Distinctive terms that should appear in evidence (1-4 tokens each)"
        },
        "do_not_anchor": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Scope/context terms to exclude from anchoring"
        },
        "suggested_synonyms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alternative forms of anchor terms (OCR variants, abbreviations)"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in analysis (0.0-1.0)"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the decomposition"
        }
    },
    "required": ["core_concepts", "anchor_terms", "scope_filters", "do_not_anchor", "suggested_synonyms", "confidence", "reasoning"],
    "additionalProperties": False
}


# =============================================================================
# LLM Call for Query Analysis
# =============================================================================

def get_collection_names(conn) -> List[str]:
    """Get list of collection slugs from database."""
    with conn.cursor() as cur:
        cur.execute("SELECT slug FROM collections ORDER BY slug")
        return [row[0] for row in cur.fetchall()]


def build_query_analysis_prompt(query_text: str, collection_names: List[str]) -> str:
    """Build prompt for query analysis LLM call."""
    collections_str = ", ".join(collection_names) if collection_names else "(no collections found)"
    
    return f"""Analyze this research query and decompose it into structured components.

QUERY: {query_text}

KNOWN COLLECTION NAMES (for reference - only use if EXPLICITLY mentioned):
{collections_str}

YOUR TASK:
Separate "where to search" (scope) from "what to search for" (anchors/concepts).

GUIDELINES:

1. scope_filters.collections:
   - ONLY set if the user EXPLICITLY mentions a collection name (e.g., "in the Vassiliev notebooks")
   - Set to null if no collection is explicitly mentioned
   - Do NOT infer collections from subject matter (e.g., "Soviet" does NOT imply any specific collection)
   - Most queries should have collections: null (search all)

2. core_concepts:
   - What is the query fundamentally asking about? (1-3 phrases)
   - This is used for semantic retrieval
   - Example: "proximity fuse espionage" for a query about proximity fuse technology

3. anchor_terms:
   - Distinctive terms that MUST appear in evidence for it to be relevant
   - These are used for lexical matching
   - Keep to 1-4 tokens each
   - Focus on technical terms, proper nouns, specific phrases
   - Example: ["proximity fuse"] - NOT ["soviets", "acquisition", "espionage"]
   - The MORE SPECIFIC the better

4. do_not_anchor:
   - Terms that are context/scope but should NOT be required in evidence
   - Generic terms like "soviet", "acquisition", "espionage", "documents", "evidence", "information"
   - These would create false negatives if required

5. suggested_synonyms:
   - Alternative forms of anchor terms that might appear due to:
     * OCR errors ("fuze" vs "fuse")
     * Abbreviations ("VT" for "variable time")
     * Historical terminology ("radio proximity" vs "proximity fuse")
   - Keep to 1-4 tokens each

CRITICAL RULES:
- scope_filters.collections should be NULL unless user explicitly names a collection
- Anchor terms should be SPECIFIC and DISTINCTIVE (technical terms, proper nouns)
- Do NOT anchor on generic words like "soviet", "acquisition", "evidence", "espionage"
- When in doubt, put terms in do_not_anchor (false positives are better than false negatives)

OUTPUT: JSON matching the schema."""


def analyze_query(query_text: str, conn) -> QueryAnalysis:
    """
    Use LLM to decompose query into scope vs evidence terms.
    
    This replaces heuristic anchor extraction with structured LLM output.
    """
    import os
    import json
    from openai import OpenAI
    
    # Get collection names for context
    try:
        collection_names = get_collection_names(conn)
    except Exception:
        collection_names = []
    
    # Build prompt
    prompt = build_query_analysis_prompt(query_text, collection_names)
    
    # Call LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")
    
    model = os.getenv("OPENAI_MODEL_PLAN", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a query analyzer for a historical research system. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "query_analysis", "schema": QUERY_ANALYSIS_SCHEMA, "strict": True},
        },
    }
    
    resp = client.beta.chat.completions.parse(**request_params)
    
    if not resp.choices:
        raise RuntimeError("OpenAI returned no choices")
    
    msg = resp.choices[0].message
    
    # Parse response
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            result = parsed.model_dump()
        elif hasattr(parsed, "dict"):
            result = parsed.dict()
        elif isinstance(parsed, dict):
            result = parsed
        else:
            result = json.loads(json.dumps(parsed, default=str))
    else:
        content = msg.content
        if not content:
            raise RuntimeError("LLM returned no content")
        result = json.loads(content)
    
    # Build QueryAnalysis
    return QueryAnalysis(
        query_text=query_text,
        scope_filters=result.get("scope_filters", {}),
        core_concepts=result.get("core_concepts", []),
        anchor_terms=result.get("anchor_terms", []),
        do_not_anchor=result.get("do_not_anchor", []),
        suggested_synonyms=result.get("suggested_synonyms", []),
        confidence=result.get("confidence", 1.0),
        reasoning=result.get("reasoning", ""),
    )
