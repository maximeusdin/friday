"""
V3 Claims Schema - Strict claim format with required citations.

Every claim:
- Has text stating a finding
- Has 1+ EvidenceRef citations
- Has confidence level (supported | suggestive)
- Optionally flags if asserting a relationship

Claims are synthesized from the EvidenceSet cite_spans only.
"""

import os
import json
import hashlib
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from retrieval.agent import V3_MODEL_DEFAULT


@dataclass
class EvidenceRef:
    """Reference to evidence supporting a claim."""
    doc_id: int
    chunk_id: int
    start_char: int
    end_char: int
    quote: str                        # Must match span text
    page_ref: Optional[str] = None
    span_id: Optional[str] = None     # For linking back to EvidenceSet
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote": self.quote[:500],  # Cap for storage
            "page_ref": self.page_ref,
            "span_id": self.span_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRef":
        return cls(
            doc_id=data.get("doc_id", 0),
            chunk_id=data.get("chunk_id", 0),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            quote=data.get("quote", ""),
            page_ref=data.get("page_ref"),
            span_id=data.get("span_id"),
        )


@dataclass
class ClaimV3:
    """
    A claim with strict citation requirements.
    
    V3 Rule: Every claim must have at least 1 EvidenceRef with a quote.
    
    Entity Attestation Contract:
    - about_entities: entity_ids the claim asserts facts about
    - For each entity_id, at least one cited quote must contain a recognized
      surface form (canonical name or alias) of that entity
    - If the claim is about a relationship, both parties should be in about_entities
    """
    claim_id: str
    text: str                         # The claim statement
    evidence: List[EvidenceRef]       # 1+ required
    confidence: str                   # "supported" | "suggestive"
    about_entities: List[int] = field(default_factory=list)  # REQUIRED: entity_ids being asserted about
    about_literals: List[str] = field(default_factory=list)  # Optional: dates, numbers, places asserted
    notes: Optional[str] = None
    asserts_relationship: bool = False  # For relationship cue check
    
    def __post_init__(self):
        if not self.claim_id:
            content = f"{self.text}:{len(self.evidence)}:{self.about_entities}"
            self.claim_id = hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "about_entities": self.about_entities,
            "about_literals": self.about_literals,
            "notes": self.notes,
            "asserts_relationship": self.asserts_relationship,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimV3":
        return cls(
            claim_id=data.get("claim_id", ""),
            text=data.get("text", ""),
            evidence=[EvidenceRef.from_dict(e) for e in data.get("evidence", [])],
            confidence=data.get("confidence", "suggestive"),
            about_entities=data.get("about_entities", []),
            about_literals=data.get("about_literals", []),
            notes=data.get("notes"),
            asserts_relationship=data.get("asserts_relationship", False),
        )


@dataclass
class ClaimBundleV3:
    """Bundle of claims with metadata."""
    claims: List[ClaimV3]
    query_text: str
    evidence_set_id: str
    model_version: str = ""
    bundle_id: str = ""
    
    def __post_init__(self):
        if not self.model_version:
            self.model_version = V3_MODEL_DEFAULT
        if not self.bundle_id:
            content = f"{self.query_text}:{len(self.claims)}:{self.evidence_set_id}"
            self.bundle_id = hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "query_text": self.query_text,
            "evidence_set_id": self.evidence_set_id,
            "model_version": self.model_version,
            "claims": [c.to_dict() for c in self.claims],
            "claim_count": len(self.claims),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimBundleV3":
        return cls(
            claims=[ClaimV3.from_dict(c) for c in data.get("claims", [])],
            query_text=data.get("query_text", ""),
            evidence_set_id=data.get("evidence_set_id", ""),
            model_version=data.get("model_version", V3_MODEL_DEFAULT),
            bundle_id=data.get("bundle_id", ""),
        )


# =============================================================================
# Claim Synthesis Prompt
# =============================================================================

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices into the provided spans list"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["supported", "suggestive"]
                    },
                    "about_entities": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Entity IDs the claim asserts facts about"
                    },
                    "about_literals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Literal values (dates, numbers) the claim asserts"
                    },
                    "asserts_relationship": {"type": "boolean"},
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["text", "evidence_indices", "confidence", "about_entities", "asserts_relationship"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["claims", "reasoning"],
    "additionalProperties": False,
}


def build_synthesis_prompt(
    query: str,
    cite_spans: List["EvidenceSpan"],
    entity_context: Optional[List[Dict[str, Any]]] = None,
    max_claims: int = 10,
) -> str:
    """Build prompt for claim synthesis."""
    
    # Format spans for prompt
    spans_text = []
    for i, span in enumerate(cite_spans[:50]):  # Limit for prompt size
        spans_text.append(f"[{i}] (chunk:{span.chunk_id}, p:{span.page_ref})\n{span.quote[:300]}...")
    
    spans_formatted = "\n\n".join(spans_text)
    
    # Format entity context if available
    entity_section = ""
    if entity_context:
        entity_lines = []
        for ent in entity_context[:100]:  # Limit entities shown
            eid = ent.get("entity_id", ent.get("id", "?"))
            name = ent.get("canonical_name", ent.get("name", "Unknown"))
            aliases = ent.get("aliases", [])
            alias_str = f" (aliases: {', '.join(aliases[:3])})" if aliases else ""
            entity_lines.append(f"  {eid}: {name}{alias_str}")
        entity_section = f"""
KNOWN ENTITIES (use these entity_ids in about_entities):
{chr(10).join(entity_lines)}
"""
    
    return f"""Synthesize claims from evidence to answer this query.

QUERY: {query}

EVIDENCE SPANS (cite by index):
{spans_formatted}
{entity_section}
SYNTHESIS RULES:

1. Each claim must cite at least 1 evidence span by index
2. Only make claims that are DIRECTLY supported by the quotes
3. Do NOT introduce facts, names, or dates not in the quoted evidence
4. Set confidence:
   - "supported": clear, explicit evidence in quote
   - "suggestive": evidence implies but doesn't directly state
5. Set asserts_relationship=true if claim states a connection between people/orgs
6. Maximum {max_claims} claims

ENTITY ATTESTATION (CRITICAL):
- For each claim, include about_entities: the entity_ids for the specific entities the claim asserts facts about
- If the claim is about a relationship between two entities, BOTH entity_ids must be in about_entities
- If you cannot identify entity_ids confidently, do NOT emit the claim
- The cited evidence MUST contain the name/alias of each entity in about_entities
- about_literals: include any specific dates, numbers, or places asserted in the claim

CRITICAL:
- Do NOT make claims about entities not mentioned in the quotes
- Do NOT infer relationships unless explicitly stated
- If evidence is sparse, produce fewer claims rather than overreach
- Empty about_entities is acceptable ONLY for general factual claims (e.g., "The report discusses procurement activities")

OUTPUT: JSON with claims array, each claim has:
- text: the claim statement
- evidence_indices: list of span indices (e.g., [0, 3])
- confidence: "supported" or "suggestive"
- about_entities: list of entity_ids the claim asserts about (can be empty for general claims)
- about_literals: list of literal values (dates, numbers) asserted
- asserts_relationship: true/false
- notes: optional context
"""


def synthesize_claims(
    query: str,
    evidence_set: "EvidenceSet",
    conn,
    max_claims: int = 10,
    entity_context: Optional[List[Dict[str, Any]]] = None,
) -> ClaimBundleV3:
    """
    Synthesize claims from evidence using LLM.
    
    Args:
        query: The user's query
        evidence_set: EvidenceSet with cite_spans
        conn: Database connection (for loading entity context if not provided)
        max_claims: Maximum claims to generate
        entity_context: Optional list of entities with {entity_id, canonical_name, aliases}
    
    Returns:
        ClaimBundleV3 with synthesized claims
    """
    from openai import OpenAI
    from retrieval.agent.v3_evidence import EvidenceSpan
    
    if not evidence_set.cite_spans:
        return ClaimBundleV3(
            claims=[],
            query_text=query,
            evidence_set_id=evidence_set.evidence_set_id,
        )
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _deterministic_claims(query, evidence_set)
    
    # Load entity context if not provided
    if entity_context is None:
        entity_context = _load_entity_context_from_spans(conn, evidence_set.cite_spans)
    
    # Build prompt
    prompt = build_synthesis_prompt(query, evidence_set.cite_spans, entity_context, max_claims)
    
    # Call LLM
    model = os.getenv("OPENAI_MODEL_V3", V3_MODEL_DEFAULT)
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a claim synthesizer. Only make claims supported by evidence. Include about_entities for entity attestation. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        
        content = response.choices[0].message.content
        if not content:
            return _deterministic_claims(query, evidence_set)
        
        data = json.loads(content)
        
        # Convert LLM output to ClaimV3 objects
        claims = []
        for claim_data in data.get("claims", []):
            # Resolve evidence indices to EvidenceRefs
            evidence_refs = []
            for idx in claim_data.get("evidence_indices", []):
                if 0 <= idx < len(evidence_set.cite_spans):
                    span = evidence_set.cite_spans[idx]
                    evidence_refs.append(EvidenceRef(
                        doc_id=span.doc_id,
                        chunk_id=span.chunk_id,
                        start_char=span.start_char,
                        end_char=span.end_char,
                        quote=span.quote,
                        page_ref=span.page_ref,
                        span_id=span.span_id,
                    ))
            
            if evidence_refs:  # Only add claims with valid evidence
                # Parse about_entities (ensure they're integers)
                about_entities = []
                for eid in claim_data.get("about_entities", []):
                    if isinstance(eid, int):
                        about_entities.append(eid)
                    elif isinstance(eid, str) and eid.isdigit():
                        about_entities.append(int(eid))
                
                claims.append(ClaimV3(
                    claim_id="",  # Will be computed
                    text=claim_data.get("text", ""),
                    evidence=evidence_refs,
                    confidence=claim_data.get("confidence", "suggestive"),
                    about_entities=about_entities,
                    about_literals=claim_data.get("about_literals", []),
                    asserts_relationship=claim_data.get("asserts_relationship", False),
                    notes=claim_data.get("notes"),
                ))
        
        return ClaimBundleV3(
            claims=claims,
            query_text=query,
            evidence_set_id=evidence_set.evidence_set_id,
            model_version=model,
        )
        
    except Exception as e:
        print(f"    Claim synthesis error: {e}", file=sys.stderr)
        return _deterministic_claims(query, evidence_set)


def _load_entity_context_from_spans(conn, cite_spans: List["EvidenceSpan"]) -> List[Dict[str, Any]]:
    """
    Load entity context from the database based on entities mentioned in cite_spans.
    
    This provides the LLM with entity_ids it can use in about_entities.
    """
    if not cite_spans or not conn:
        return []
    
    # Get unique chunk_ids from spans
    chunk_ids = list({s.chunk_id for s in cite_spans})
    
    if not chunk_ids:
        return []
    
    try:
        with conn.cursor() as cur:
            # Find entities mentioned in these chunks via entity_mentions
            cur.execute("""
                SELECT DISTINCT e.id, e.canonical_name, e.entity_type
                FROM entities e
                JOIN entity_mentions em ON em.entity_id = e.id
                WHERE em.chunk_id = ANY(%s)
                ORDER BY e.canonical_name
                LIMIT 200
            """, (chunk_ids,))
            
            entities = []
            entity_ids = []
            for row in cur.fetchall():
                entity_id, canonical_name, entity_type = row
                entities.append({
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                    "aliases": [],
                })
                entity_ids.append(entity_id)
            
            # Load aliases for these entities
            if entity_ids:
                cur.execute("""
                    SELECT entity_id, alias
                    FROM entity_aliases
                    WHERE entity_id = ANY(%s)
                """, (entity_ids,))
                
                alias_map = {}
                for row in cur.fetchall():
                    eid, alias = row
                    if eid not in alias_map:
                        alias_map[eid] = []
                    alias_map[eid].append(alias)
                
                # Attach aliases to entities
                for ent in entities:
                    ent["aliases"] = alias_map.get(ent["entity_id"], [])
            
            return entities
            
    except Exception as e:
        print(f"    Warning: Could not load entity context: {e}", file=sys.stderr)
        return []


def _deterministic_claims(query: str, evidence_set: "EvidenceSet") -> ClaimBundleV3:
    """
    Fallback deterministic claim generation.
    
    Creates simple "evidence found" claims for top spans.
    """
    claims = []
    
    for span in evidence_set.cite_spans[:5]:
        # Create a simple evidence-found claim
        preview = span.quote[:100].strip()
        preview = ' '.join(preview.split())
        
        claims.append(ClaimV3(
            claim_id="",
            text=f"Evidence found in {span.page_ref}: \"{preview}...\"",
            evidence=[EvidenceRef(
                doc_id=span.doc_id,
                chunk_id=span.chunk_id,
                start_char=span.start_char,
                end_char=span.end_char,
                quote=span.quote,
                page_ref=span.page_ref,
                span_id=span.span_id,
            )],
            confidence="suggestive",
            asserts_relationship=False,
        ))
    
    return ClaimBundleV3(
        claims=claims,
        query_text=query,
        evidence_set_id=evidence_set.evidence_set_id,
    )


# Type hints for imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from retrieval.agent.v3_evidence import EvidenceSet, EvidenceSpan
