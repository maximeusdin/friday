"""
V6 Query Parser - Split query into CONTROL vs CONTENT

The key insight: never entity-link tokens from the instruction/constraint part.
"Provide" is CONTROL, "Silvermaster" is CONTENT.
"Vassiliev" in "cite Vassiliev notebooks" is CONTROL (collection selector), not a person.

Output:
- scope_constraints: collection filters, date ranges
- task_type: roster_enumeration, timeline, evidence_search, etc.
- topic_terms: entities/phrases to search for (CONTENT)
- output_requirements: what the answer must include
- control_tokens: tokens NOT to entity-link
- content_tokens: tokens TO entity-link
"""
import os
import json
import sys
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Task Types
# =============================================================================

class TaskType(Enum):
    ROSTER_ENUMERATION = "roster_enumeration"  # "Who were members of X?"
    TIMELINE = "timeline"  # "When did X happen?"
    EVIDENCE_SEARCH = "evidence_search"  # "Find evidence of X"
    RELATIONSHIP = "relationship"  # "How was X connected to Y?"
    FACTUAL = "factual"  # "What was X's role?"
    COMPARISON = "comparison"  # "Compare X and Y"
    UNKNOWN = "unknown"


# =============================================================================
# Parsed Query
# =============================================================================

@dataclass
class ParsedQuery:
    """Result of parsing a user query."""
    
    original_query: str
    
    # Task classification
    task_type: TaskType = TaskType.UNKNOWN
    
    # Scope constraints (not for entity linking)
    scope_constraints: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"collections": ["vassiliev"], "date_from": "1940", "date_to": "1950"}
    
    # Topic terms (FOR entity linking)
    topic_terms: List[str] = field(default_factory=list)
    # e.g., ["Silvermaster network", "Silvermaster"]
    
    # Output requirements
    output_requirements: List[str] = field(default_factory=list)
    # e.g., ["must cite Vassiliev notebooks", "list format"]
    
    # Token classification
    control_tokens: Set[str] = field(default_factory=set)  # Don't entity-link
    content_tokens: Set[str] = field(default_factory=set)  # Do entity-link
    
    # Confidence
    parse_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "task_type": self.task_type.value,
            "scope_constraints": self.scope_constraints,
            "topic_terms": self.topic_terms,
            "output_requirements": self.output_requirements,
            "control_tokens": list(self.control_tokens),
            "content_tokens": list(self.content_tokens),
            "parse_confidence": self.parse_confidence,
        }
    
    def is_token_content(self, token: str) -> bool:
        """Check if a token should be entity-linked."""
        token_lower = token.lower()
        # Explicit content
        if token_lower in {t.lower() for t in self.content_tokens}:
            return True
        # In topic terms
        for topic in self.topic_terms:
            if token_lower in topic.lower():
                return True
        # Explicit control
        if token_lower in {t.lower() for t in self.control_tokens}:
            return False
        # Default: uncertain
        return False


# =============================================================================
# Parser Prompt
# =============================================================================

PARSER_SYSTEM_PROMPT = """You are parsing a research query to separate CONTROL tokens from CONTENT tokens.

CONTROL tokens: instructions, constraints, corpus selectors, output format requests
  - "Provide", "List", "Find", "cite", "only from", "in the"
  - Collection names when used as filters: "Vassiliev notebooks", "VENONA"
  - Date ranges: "between 1940 and 1950"

CONTENT tokens: the actual subject matter to search for
  - Person names: "Silvermaster", "Harry White"
  - Organization names: "the network", "Treasury Department"
  - Events: "espionage", "recruitment"

The CRITICAL rule: if "Vassiliev" appears in "cite only from Vassiliev notebooks", 
it's a CONTROL token (collection filter), NOT a person to search for.

Output structured JSON with clear separation."""


def build_parser_prompt(query: str) -> str:
    """Build prompt for query parsing."""
    
    return f"""Parse this research query:

QUERY: "{query}"

Extract:
1. task_type: What kind of answer is expected?
   - roster_enumeration: "Who were members of X?" → expects a list of people
   - timeline: "When did X happen?" → expects dates/sequence
   - evidence_search: "Find evidence of X" → expects supporting documents
   - relationship: "How was X connected to Y?" → expects relationship description
   - factual: "What was X's role?" → expects factual answer
   - comparison: "Compare X and Y" → expects comparison

2. scope_constraints: Filters on the search (NOT for entity linking)
   - collections: ["vassiliev", "venona"] if specified
   - date_from, date_to: if date ranges mentioned
   
3. topic_terms: The actual subjects to search for (WILL be entity-linked)
   - Person/org names that are the TOPIC
   - Key concepts to find

4. output_requirements: What the answer must include
   - "must cite X", "list format", "with dates"

5. control_tokens: Words/phrases that are INSTRUCTIONS (do NOT entity-link these)
   - Verbs: "provide", "list", "find", "cite"
   - Collection names when used as filters
   - Format instructions

6. content_tokens: Words/phrases that are CONTENT (DO entity-link these)
   - Names of people/orgs being asked about
   - Key topics

Output JSON:
{{
  "task_type": "roster_enumeration",
  "scope_constraints": {{"collections": ["vassiliev"]}},
  "topic_terms": ["Silvermaster network", "Silvermaster"],
  "output_requirements": ["must cite Vassiliev notebooks", "list members"],
  "control_tokens": ["provide", "list", "cite", "vassiliev notebooks", "only from"],
  "content_tokens": ["Silvermaster network", "Silvermaster", "members"],
  "confidence": 0.9
}}

CRITICAL: If a collection name like "Vassiliev" appears as a filter/source constraint,
it goes in control_tokens, NOT content_tokens."""


# =============================================================================
# Query Parser
# =============================================================================

class QueryParser:
    """
    Parses user queries to separate CONTROL from CONTENT.
    
    This prevents entity-linking "Provide" or treating "Vassiliev" 
    as a person when it's a collection filter.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
    
    def parse(self, query: str) -> ParsedQuery:
        """Parse a query into structured components."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_parse(query)
        
        prompt = build_parser_prompt(query)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1000,
            )
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_parse(query)
            
            data = json.loads(content)
            
            # Map task type
            task_type_str = data.get("task_type", "unknown")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.UNKNOWN
            
            parsed = ParsedQuery(
                original_query=query,
                task_type=task_type,
                scope_constraints=data.get("scope_constraints", {}),
                topic_terms=data.get("topic_terms", []),
                output_requirements=data.get("output_requirements", []),
                control_tokens=set(data.get("control_tokens", [])),
                content_tokens=set(data.get("content_tokens", [])),
                parse_confidence=data.get("confidence", 0.5),
            )
            
            if self.verbose:
                print(f"  [QueryParser] LLM response received", file=sys.stderr)
                print(f"  [QueryParser] Raw classification:", file=sys.stderr)
                print(f"    task_type: {task_type.value}", file=sys.stderr)
                print(f"    topic_terms: {parsed.topic_terms}", file=sys.stderr)
                print(f"    content_tokens: {list(parsed.content_tokens)}", file=sys.stderr)
                print(f"    control_tokens: {list(parsed.control_tokens)}", file=sys.stderr)
                print(f"    scope_constraints: {parsed.scope_constraints}", file=sys.stderr)
                print(f"    output_requirements: {parsed.output_requirements}", file=sys.stderr)
                print(f"    confidence: {parsed.parse_confidence}", file=sys.stderr)
            
            return parsed
            
        except Exception as e:
            if self.verbose:
                print(f"  [QueryParser] Error: {e}", file=sys.stderr)
            return self._fallback_parse(query)
    
    def _fallback_parse(self, query: str) -> ParsedQuery:
        """Simple rule-based fallback parsing."""
        
        query_lower = query.lower()
        
        # Detect task type
        task_type = TaskType.UNKNOWN
        if any(w in query_lower for w in ["who were", "list members", "members of"]):
            task_type = TaskType.ROSTER_ENUMERATION
        elif any(w in query_lower for w in ["when", "timeline", "chronology"]):
            task_type = TaskType.TIMELINE
        elif any(w in query_lower for w in ["evidence", "proof", "documentation"]):
            task_type = TaskType.EVIDENCE_SEARCH
        elif any(w in query_lower for w in ["connected", "relationship", "between"]):
            task_type = TaskType.RELATIONSHIP
        
        # Common control tokens
        control_tokens = {
            "provide", "list", "find", "show", "give", "tell", "explain",
            "cite", "citing", "from", "only", "must", "should",
            "using", "based on", "according to",
        }
        
        # Extract potential topic terms (capitalized phrases)
        topic_terms = []
        # Simple extraction: find capitalized words that aren't at sentence start
        words = query.split()
        for i, word in enumerate(words):
            clean = word.strip(",.?!\"'()[]")
            if clean and clean[0].isupper() and i > 0:
                if clean.lower() not in control_tokens:
                    topic_terms.append(clean)
        
        # Detect collection constraints
        scope = {}
        if "vassiliev" in query_lower:
            scope["collections"] = ["vassiliev"]
            control_tokens.add("vassiliev")
        if "venona" in query_lower:
            if "collections" not in scope:
                scope["collections"] = []
            scope["collections"].append("venona")
            control_tokens.add("venona")
        
        return ParsedQuery(
            original_query=query,
            task_type=task_type,
            scope_constraints=scope,
            topic_terms=topic_terms,
            control_tokens=control_tokens,
            content_tokens=set(topic_terms),
            parse_confidence=0.3,
        )


# =============================================================================
# Token Classifier (lightweight alternative)
# =============================================================================

def classify_tokens(query: str, parsed: ParsedQuery) -> Dict[str, str]:
    """
    Classify each token as CONTROL or CONTENT.
    
    Returns: {token: "CONTROL" | "CONTENT" | "UNKNOWN"}
    """
    result = {}
    
    # Tokenize simply
    tokens = re.findall(r'\b\w+\b', query)
    
    for token in tokens:
        token_lower = token.lower()
        
        # Check explicit classifications
        if token_lower in {t.lower() for t in parsed.control_tokens}:
            result[token] = "CONTROL"
        elif token_lower in {t.lower() for t in parsed.content_tokens}:
            result[token] = "CONTENT"
        elif token in parsed.topic_terms or any(token in t for t in parsed.topic_terms):
            result[token] = "CONTENT"
        else:
            result[token] = "UNKNOWN"
    
    return result
