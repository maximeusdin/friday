"""
V6 Entity Linker - Only link CONTENT tokens, with retrieval filtering

Key principles:
1. NEVER entity-link CONTROL tokens (verbs, constraints, collection names)
2. Each linked entity has use_for_retrieval: true/false
3. Only entities central to the question get use_for_retrieval=true

This prevents:
- "Provide" → entity_id=8625
- "Vassiliev" (collection) → person_id
- Random low-confidence matches being used for retrieval
"""
import os
import json
import sys
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from retrieval.agent.v6_query_parser import ParsedQuery, TaskType


# =============================================================================
# Linked Entity
# =============================================================================

@dataclass
class LinkedEntity:
    """An entity linked from the query."""
    
    surface_form: str  # The text that was linked
    entity_id: int  # Database entity ID
    canonical_name: str  # Canonical name from DB
    entity_type: str  # person, org, location, etc.
    
    # Linking metadata
    link_confidence: float  # 0-1, how confident in the match
    match_type: str  # "exact", "alias", "partial", "concordance"
    
    # CRITICAL: Should this be used for retrieval?
    use_for_retrieval: bool = False
    retrieval_reason: str = ""  # Why yes/no
    
    # Token classification
    is_topic_entity: bool = False  # Was this from topic_terms?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_form": self.surface_form,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "link_confidence": self.link_confidence,
            "match_type": self.match_type,
            "use_for_retrieval": self.use_for_retrieval,
            "retrieval_reason": self.retrieval_reason,
            "is_topic_entity": self.is_topic_entity,
        }


# =============================================================================
# Entity Linking Result
# =============================================================================

@dataclass
class EntityLinkingResult:
    """Result of entity linking on a parsed query."""
    
    linked_entities: List[LinkedEntity] = field(default_factory=list)
    
    # Entities to use for retrieval (filtered subset)
    retrieval_entities: List[LinkedEntity] = field(default_factory=list)
    
    # Stats
    total_linked: int = 0
    used_for_retrieval: int = 0
    rejected_control_tokens: List[str] = field(default_factory=list)
    
    def get_retrieval_entity_ids(self) -> List[int]:
        """Get entity IDs that should be used for retrieval."""
        return [e.entity_id for e in self.retrieval_entities]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "linked_entities": [e.to_dict() for e in self.linked_entities],
            "retrieval_entity_ids": self.get_retrieval_entity_ids(),
            "total_linked": self.total_linked,
            "used_for_retrieval": self.used_for_retrieval,
            "rejected_control_tokens": self.rejected_control_tokens,
        }


# =============================================================================
# Entity Linker
# =============================================================================

class EntityLinker:
    """
    Links entities from CONTENT tokens only.
    
    Each linked entity gets a use_for_retrieval decision based on:
    1. Is it from topic_terms? (high priority)
    2. Is it central to answering the question? (LLM decision)
    3. Is the link confidence high enough?
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        verbose: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
    
    def link(
        self,
        parsed_query: ParsedQuery,
        conn,
    ) -> EntityLinkingResult:
        """
        Link entities from topic terms only.
        
        Args:
            parsed_query: The parsed query with CONTROL/CONTENT separation
            conn: Database connection
        
        Returns:
            EntityLinkingResult with filtered retrieval entities
        """
        result = EntityLinkingResult()
        
        # Only link topic terms (CONTENT), not control tokens
        terms_to_link = parsed_query.topic_terms
        control_tokens = parsed_query.control_tokens
        
        if self.verbose:
            print(f"  [EntityLinker] Starting entity linking", file=sys.stderr)
            print(f"    Input topic_terms to link: {terms_to_link}", file=sys.stderr)
            print(f"    Control tokens to skip: {list(control_tokens)[:10]}", file=sys.stderr)
        
        for term in terms_to_link:
            # Skip if this term is actually a control token
            if term.lower() in {t.lower() for t in control_tokens}:
                result.rejected_control_tokens.append(term)
                if self.verbose:
                    print(f"    [SKIP] \"{term}\" - in control_tokens, NOT linking", file=sys.stderr)
                continue
            
            if self.verbose:
                print(f"    [TRY] Linking \"{term}\"...", file=sys.stderr)
            
            # Try to link this term
            linked = self._link_term(term, conn, parsed_query)
            
            if linked:
                linked.is_topic_entity = True
                result.linked_entities.append(linked)
                result.total_linked += 1
                
                if self.verbose:
                    print(f"         FOUND: [{linked.entity_id}] {linked.canonical_name} ({linked.match_type}, conf={linked.link_confidence:.2f})", file=sys.stderr)
                
                # Decide if this should be used for retrieval
                self._decide_retrieval_use(linked, parsed_query)
                
                if linked.use_for_retrieval:
                    result.retrieval_entities.append(linked)
                    result.used_for_retrieval += 1
                    if self.verbose:
                        print(f"         → USE FOR RETRIEVAL: {linked.retrieval_reason}", file=sys.stderr)
                else:
                    if self.verbose:
                        print(f"         → NOT for retrieval: {linked.retrieval_reason}", file=sys.stderr)
            else:
                # Try individual words if the phrase wasn't found
                # e.g., "Silvermaster network" -> try "Silvermaster"
                words = [w.strip() for w in term.split() if len(w.strip()) > 2]
                found_via_word = False
                
                for word in words:
                    if word.lower() in {"the", "and", "for", "with", "network", "group", "members"}:
                        continue
                    
                    if self.verbose:
                        print(f"         [TRY WORD] \"{word}\"...", file=sys.stderr)
                    
                    linked = self._link_term(word, conn, parsed_query)
                    if linked:
                        linked.is_topic_entity = True
                        linked.surface_form = term  # Keep original surface form
                        result.linked_entities.append(linked)
                        result.total_linked += 1
                        
                        if self.verbose:
                            print(f"         FOUND via word: [{linked.entity_id}] {linked.canonical_name} ({linked.match_type}, conf={linked.link_confidence:.2f})", file=sys.stderr)
                        
                        self._decide_retrieval_use(linked, parsed_query)
                        
                        if linked.use_for_retrieval:
                            result.retrieval_entities.append(linked)
                            result.used_for_retrieval += 1
                            if self.verbose:
                                print(f"         → USE FOR RETRIEVAL: {linked.retrieval_reason}", file=sys.stderr)
                        else:
                            if self.verbose:
                                print(f"         → NOT for retrieval: {linked.retrieval_reason}", file=sys.stderr)
                        
                        found_via_word = True
                        break  # Found one, that's enough for this term
                
                if not found_via_word and self.verbose:
                    print(f"         NOT FOUND in entity database", file=sys.stderr)
        
        if self.verbose:
            print(f"  [EntityLinker] Complete: {result.total_linked} linked, {result.used_for_retrieval} for retrieval", 
                  file=sys.stderr)
        
        return result
    
    def _link_term(
        self,
        term: str,
        conn,
        parsed_query: ParsedQuery,
    ) -> Optional[LinkedEntity]:
        """Try to link a single term to an entity."""
        
        term = term.strip()
        if not term:
            return None
        
        try:
            with conn.cursor() as cur:
                # Try exact canonical name match
                cur.execute(
                    "SELECT id, canonical_name, entity_type FROM entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 1",
                    (term,)
                )
                row = cur.fetchone()
                if row:
                    return LinkedEntity(
                        surface_form=term,
                        entity_id=row[0],
                        canonical_name=row[1],
                        entity_type=row[2] or "unknown",
                        link_confidence=0.95,
                        match_type="exact",
                    )
                
                # Try alias match
                cur.execute("""
                    SELECT e.id, e.canonical_name, e.entity_type
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE LOWER(ea.alias) = LOWER(%s)
                    LIMIT 1
                """, (term,))
                row = cur.fetchone()
                if row:
                    return LinkedEntity(
                        surface_form=term,
                        entity_id=row[0],
                        canonical_name=row[1],
                        entity_type=row[2] or "unknown",
                        link_confidence=0.85,
                        match_type="alias",
                    )
                
                # Try concordance expansion
                from retrieval.ops import concordance_expand_terms
                expanded = concordance_expand_terms(conn, term, max_aliases_out=5)
                
                for alias in expanded:
                    if alias.lower() != term.lower():
                        cur.execute(
                            "SELECT id, canonical_name, entity_type FROM entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 1",
                            (alias,)
                        )
                        row = cur.fetchone()
                        if row:
                            return LinkedEntity(
                                surface_form=term,
                                entity_id=row[0],
                                canonical_name=row[1],
                                entity_type=row[2] or "unknown",
                                link_confidence=0.75,
                                match_type=f"concordance:{alias}",
                            )
                
                # Try partial match (lower confidence)
                cur.execute("""
                    SELECT id, canonical_name, entity_type 
                    FROM entities 
                    WHERE LOWER(canonical_name) LIKE LOWER(%s)
                    ORDER BY LENGTH(canonical_name)
                    LIMIT 1
                """, (f"%{term}%",))
                row = cur.fetchone()
                if row:
                    return LinkedEntity(
                        surface_form=term,
                        entity_id=row[0],
                        canonical_name=row[1],
                        entity_type=row[2] or "unknown",
                        link_confidence=0.5,
                        match_type="partial",
                    )
                
        except Exception as e:
            if self.verbose:
                print(f"    [!] Error linking '{term}': {e}", file=sys.stderr)
        
        return None
    
    def _decide_retrieval_use(
        self,
        entity: LinkedEntity,
        parsed_query: ParsedQuery,
    ):
        """Decide if an entity should be used for retrieval."""
        
        # Rule 1: Must meet confidence threshold
        if entity.link_confidence < self.confidence_threshold:
            entity.use_for_retrieval = False
            entity.retrieval_reason = f"Low confidence ({entity.link_confidence:.2f})"
            return
        
        # Rule 2: Must be from topic terms (we already filtered this, but double-check)
        if not entity.is_topic_entity:
            entity.use_for_retrieval = False
            entity.retrieval_reason = "Not a topic entity"
            return
        
        # Rule 3: For roster queries, prefer person entities
        if parsed_query.task_type == TaskType.ROSTER_ENUMERATION:
            if entity.entity_type == "person":
                entity.use_for_retrieval = True
                entity.retrieval_reason = "Person entity for roster query"
            elif entity.entity_type == "org":
                # Orgs can be used too (e.g., "Silvermaster network")
                entity.use_for_retrieval = True
                entity.retrieval_reason = "Org entity for roster query"
            else:
                entity.use_for_retrieval = False
                entity.retrieval_reason = f"Non-person/org type: {entity.entity_type}"
            return
        
        # Default: use if high confidence
        if entity.link_confidence >= 0.7:
            entity.use_for_retrieval = True
            entity.retrieval_reason = f"High confidence ({entity.link_confidence:.2f})"
        else:
            entity.use_for_retrieval = False
            entity.retrieval_reason = f"Moderate confidence ({entity.link_confidence:.2f})"


# =============================================================================
# LLM-based retrieval decision (optional, more accurate)
# =============================================================================

RETRIEVAL_DECISION_PROMPT = """Given a research question and a linked entity, decide:
Should this entity be used as a retrieval seed?

QUESTION: {question}
TASK TYPE: {task_type}

ENTITY:
- Surface form: "{surface_form}"
- Canonical name: "{canonical_name}"
- Type: {entity_type}
- Link confidence: {confidence}

Answer:
- use_for_retrieval: true if this entity is CENTRAL to answering the question
- reason: brief explanation

For roster queries ("who were members"), the subject of the query (e.g., "Silvermaster network")
should have use_for_retrieval=true, but random entities that happen to match should not.

Output JSON: {{"use_for_retrieval": true/false, "reason": "..."}}"""


def llm_decide_retrieval(
    entity: LinkedEntity,
    parsed_query: ParsedQuery,
    model: str = "gpt-4o-mini",
) -> bool:
    """Use LLM to decide if entity should be used for retrieval."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return entity.link_confidence >= 0.7
    
    prompt = RETRIEVAL_DECISION_PROMPT.format(
        question=parsed_query.original_query,
        task_type=parsed_query.task_type.value,
        surface_form=entity.surface_form,
        canonical_name=entity.canonical_name,
        entity_type=entity.entity_type,
        confidence=entity.link_confidence,
    )
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
        )
        
        content = response.choices[0].message.content
        if content:
            data = json.loads(content)
            entity.use_for_retrieval = data.get("use_for_retrieval", False)
            entity.retrieval_reason = data.get("reason", "LLM decision")
            return entity.use_for_retrieval
            
    except Exception:
        pass
    
    return entity.link_confidence >= 0.7
