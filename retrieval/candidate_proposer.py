"""
Candidate Proposer for Agentic V2.

Extracts candidates from FocusBundle for scoring.

Contract C7: Concrete candidate extraction via:
- propose_person_candidates: from entity_mentions overlapping spans
- propose_codename_candidates: from ALL CAPS tokens (marked unresolved)
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from retrieval.focus_bundle import FocusBundle


@dataclass
class ProposedCandidate:
    """
    A candidate proposed for constraint scoring.
    
    Candidates can be:
    - Resolved entities (entity_id set, is_resolved=True)
    - Unresolved tokens/codenames (entity_id=None, is_resolved=False)
    """
    entity_id: Optional[int]
    key: str                    # stable ID: f"entity:{entity_id}" or f"token:{surface}"
    display_name: str
    entity_type: Optional[str]  # "person", "organization", etc.
    is_resolved: bool           # True if entity_id, False if unresolved token
    source_span_ids: List[str]  # spans where this candidate appears
    mention_count: int = 1      # number of mentions (for initial ranking)
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other):
        if not isinstance(other, ProposedCandidate):
            return False
        return self.key == other.key


# Stopwords for codename extraction
CODENAME_STOPWORDS = {
    # Common words that appear in ALL CAPS
    'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'BEEN', 'WERE',
    'WILL', 'CAN', 'NOT', 'BUT', 'ALL', 'ANY', 'WHO', 'WHAT', 'WHEN', 'WHERE',
    
    # Organizations that shouldn't be person candidates
    'USA', 'USSR', 'FBI', 'CIA', 'KGB', 'NKVD', 'GRU', 'OSS', 'DOS', 'DOJ',
    'NSA', 'DIA', 'ONI', 'MID', 'SIS', 'MI5', 'MI6', 'GRU',
    
    # Countries/Places
    'NEW', 'YORK', 'WASHINGTON', 'MOSCOW', 'LONDON', 'PARIS', 'BERLIN',
    
    # Time-related
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
    
    # Document markers
    'TOP', 'SECRET', 'CONFIDENTIAL', 'CLASSIFIED', 'MEMO', 'REPORT',
}


def propose_person_candidates(
    focus_bundle: "FocusBundle",
    conn,
    min_mentions: int = 1,
) -> List[ProposedCandidate]:
    """
    Propose person candidates from entity_mentions overlapping FocusBundle spans.
    
    Uses the precomputed mention_span_index for O(1) lookup (Contract C9).
    
    Args:
        focus_bundle: The FocusBundle to extract candidates from
        conn: Database connection
        min_mentions: Minimum number of mentions required
    
    Returns:
        List of ProposedCandidate for persons
    """
    if not focus_bundle.mention_span_index:
        return []
    
    # Get all entity IDs from the index
    entity_ids = list(focus_bundle.mention_span_index.keys())
    if not entity_ids:
        return []
    
    # Batch lookup entity info
    cur = conn.cursor()
    cur.execute("""
        SELECT id, canonical_name, entity_type
        FROM entities
        WHERE id = ANY(%s) AND entity_type = 'person'
    """, (entity_ids,))
    
    candidates = []
    for entity_id, canonical_name, entity_type in cur.fetchall():
        span_ids = focus_bundle.mention_span_index.get(entity_id, [])
        
        if len(span_ids) >= min_mentions:
            candidates.append(ProposedCandidate(
                entity_id=entity_id,
                key=f"entity:{entity_id}",
                display_name=canonical_name,
                entity_type=entity_type,
                is_resolved=True,
                source_span_ids=span_ids,
                mention_count=len(span_ids),
            ))
    
    # Sort by mention count descending
    candidates.sort(key=lambda c: -c.mention_count)
    
    return candidates


def propose_organization_candidates(
    focus_bundle: "FocusBundle",
    conn,
    min_mentions: int = 1,
) -> List[ProposedCandidate]:
    """
    Propose organization candidates from entity_mentions.
    
    Useful for affiliation queries where orgs are candidates.
    """
    if not focus_bundle.mention_span_index:
        return []
    
    entity_ids = list(focus_bundle.mention_span_index.keys())
    if not entity_ids:
        return []
    
    cur = conn.cursor()
    cur.execute("""
        SELECT id, canonical_name, entity_type
        FROM entities
        WHERE id = ANY(%s) AND entity_type = 'organization'
    """, (entity_ids,))
    
    candidates = []
    for entity_id, canonical_name, entity_type in cur.fetchall():
        span_ids = focus_bundle.mention_span_index.get(entity_id, [])
        
        if len(span_ids) >= min_mentions:
            candidates.append(ProposedCandidate(
                entity_id=entity_id,
                key=f"entity:{entity_id}",
                display_name=canonical_name,
                entity_type=entity_type,
                is_resolved=True,
                source_span_ids=span_ids,
                mention_count=len(span_ids),
            ))
    
    candidates.sort(key=lambda c: -c.mention_count)
    return candidates


def propose_codename_candidates(
    focus_bundle: "FocusBundle",
    min_mentions: int = 2,
    max_candidates: int = 50,
) -> List[ProposedCandidate]:
    """
    Propose unresolved codename candidates from ALL CAPS tokens in FocusBundle.
    
    Marked as unresolved; may be promoted if entity linking succeeds later.
    
    Args:
        focus_bundle: The FocusBundle to extract candidates from
        min_mentions: Minimum number of mentions required (default 2)
        max_candidates: Maximum candidates to return
    
    Returns:
        List of ProposedCandidate for unresolved codenames
    """
    codename_spans: Dict[str, List[str]] = defaultdict(list)
    
    for span in focus_bundle.spans:
        # Find ALL CAPS tokens (3-15 chars)
        caps = re.findall(r'\b[A-Z]{3,15}\b', span.text)
        
        for cap in caps:
            if cap not in CODENAME_STOPWORDS:
                codename_spans[cap].append(span.span_id)
    
    candidates = []
    for codename, span_ids in codename_spans.items():
        if len(span_ids) >= min_mentions:
            # Dedupe span_ids
            unique_spans = list(dict.fromkeys(span_ids))
            
            candidates.append(ProposedCandidate(
                entity_id=None,
                key=f"token:{codename}",
                display_name=codename,
                entity_type=None,
                is_resolved=False,
                source_span_ids=unique_spans,
                mention_count=len(unique_spans),
            ))
    
    # Sort by mention count descending
    candidates.sort(key=lambda c: -c.mention_count)
    
    return candidates[:max_candidates]


def propose_all_candidates(
    focus_bundle: "FocusBundle",
    conn,
    include_persons: bool = True,
    include_codenames: bool = True,
    include_orgs: bool = False,
    min_person_mentions: int = 1,
    min_codename_mentions: int = 2,
    min_org_mentions: int = 1,
) -> List[ProposedCandidate]:
    """
    Propose all candidates from FocusBundle.
    
    Combines person entities and unresolved codenames, deduplicated.
    
    Args:
        focus_bundle: The FocusBundle
        conn: Database connection
        include_persons: Include resolved person entities
        include_codenames: Include unresolved codenames
        include_orgs: Include organization entities
        min_person_mentions: Minimum mentions for person candidates
        min_codename_mentions: Minimum mentions for codename candidates
        min_org_mentions: Minimum mentions for org candidates
    
    Returns:
        Combined list of candidates, sorted by mention count
    """
    candidates = []
    seen_keys: Set[str] = set()
    
    # Person candidates (resolved entities)
    if include_persons:
        persons = propose_person_candidates(focus_bundle, conn, min_person_mentions)
        for c in persons:
            if c.key not in seen_keys:
                candidates.append(c)
                seen_keys.add(c.key)
    
    # Organization candidates
    if include_orgs:
        orgs = propose_organization_candidates(focus_bundle, conn, min_org_mentions)
        for c in orgs:
            if c.key not in seen_keys:
                candidates.append(c)
                seen_keys.add(c.key)
    
    # Codename candidates (unresolved)
    if include_codenames:
        codenames = propose_codename_candidates(focus_bundle, min_codename_mentions)
        for c in codenames:
            if c.key not in seen_keys:
                candidates.append(c)
                seen_keys.add(c.key)
    
    # Sort by mention count, then by key for determinism
    candidates.sort(key=lambda c: (-c.mention_count, c.key))
    
    return candidates


def filter_candidates_by_target(
    candidates: List[ProposedCandidate],
    target_entity_ids: List[int],
) -> List[ProposedCandidate]:
    """
    Remove target entities from candidates.
    
    For relationship queries, we don't want the target (e.g., Julius Rosenberg)
    to appear in the results.
    """
    target_set = set(target_entity_ids)
    return [c for c in candidates if c.entity_id not in target_set]


def try_resolve_codename(
    codename: str,
    conn,
) -> Optional[ProposedCandidate]:
    """
    Try to resolve a codename to an entity via entity_aliases.
    
    Returns resolved ProposedCandidate if found, None otherwise.
    """
    cur = conn.cursor()
    
    # Check entity_aliases
    cur.execute("""
        SELECT ea.entity_id, e.canonical_name, e.entity_type
        FROM entity_aliases ea
        JOIN entities e ON e.id = ea.entity_id
        WHERE UPPER(ea.alias) = %s OR UPPER(ea.alias_norm) = %s
        LIMIT 1
    """, (codename, codename.lower()))
    
    row = cur.fetchone()
    if row:
        return ProposedCandidate(
            entity_id=row[0],
            key=f"entity:{row[0]}",
            display_name=row[1],
            entity_type=row[2],
            is_resolved=True,
            source_span_ids=[],  # Will be populated separately
            mention_count=1,
        )
    
    return None


def batch_resolve_codenames(
    codename_candidates: List[ProposedCandidate],
    conn,
) -> Dict[str, ProposedCandidate]:
    """
    Batch resolve codenames to entities.
    
    Returns dict mapping codename -> resolved candidate (if found).
    """
    if not codename_candidates:
        return {}
    
    codenames = [c.display_name for c in codename_candidates if not c.is_resolved]
    if not codenames:
        return {}
    
    cur = conn.cursor()
    
    # Batch lookup
    cur.execute("""
        SELECT UPPER(ea.alias) as codename, ea.entity_id, e.canonical_name, e.entity_type
        FROM entity_aliases ea
        JOIN entities e ON e.id = ea.entity_id
        WHERE UPPER(ea.alias) = ANY(%s)
    """, (codenames,))
    
    resolved = {}
    for codename, entity_id, canonical_name, entity_type in cur.fetchall():
        # Find original candidate to get span_ids
        original = next((c for c in codename_candidates if c.display_name == codename), None)
        span_ids = original.source_span_ids if original else []
        
        resolved[codename] = ProposedCandidate(
            entity_id=entity_id,
            key=f"entity:{entity_id}",
            display_name=canonical_name,
            entity_type=entity_type,
            is_resolved=True,
            source_span_ids=span_ids,
            mention_count=len(span_ids),
        )
    
    return resolved
