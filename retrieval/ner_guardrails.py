"""
NER Guardrails - Critical safety checks for corpus-wide NER extraction.

These guardrails prevent bad lexicon growth and queue flooding.
They must be applied BEFORE any auto-linking or queue insertion.

Usage:
    from retrieval.ner_guardrails import NERGuardrails
    
    guardrails = NERGuardrails(conn)
    
    # Check if a surface can be auto-linked
    can_link, reason = guardrails.can_auto_link(surface_norm, match_info)
    
    # Check if a surface should be queued for review
    should_queue, reason = guardrails.should_queue_for_review(surface_norm, stats)
    
    # Check if CREATE_NEW is allowed
    can_create, reason = guardrails.can_auto_create_entity(surface_norm, ner_label)
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum requirements for queueing (prevent queue flooding)
MIN_DOCS_FOR_QUEUE = 2           # Must appear in at least 2 documents
MIN_MENTIONS_FOR_QUEUE = 3       # OR at least 3 total mentions
MIN_SPAN_LENGTH_FOR_QUEUE = 4    # AND span must be at least 4 chars
MIN_NER_SCORE_FOR_QUEUE = 0.5    # AND NER score must be >= 0.5

# Fuzzy match restrictions
MIN_LENGTH_FOR_FUZZY_AUTO = 6    # Fuzzy auto-link only for strings >= 6 chars
MIN_FUZZY_SCORE_FOR_AUTO = 0.85  # Fuzzy score must be >= 0.85 for auto-link

# Danger classes that require extra scrutiny
DANGER_ALIAS_CLASSES = {
    'covername', 'code_name', 'initials', 
    'person_surname', 'surname_only'
}
# Note: 'acronym' removed - well-known acronyms like FBI, CIA, KGB are fine

# Well-known acronyms that can be auto-accepted even as single tokens
KNOWN_SAFE_ACRONYMS = {
    'fbi', 'cia', 'kgb', 'nkvd', 'gpu', 'ogpu', 'mgb', 'gru', 'nsa',
    'oss', 'mi5', 'mi6', 'smersh', 'cpusa', 'comintern',
    'un', 'nato', 'ussr', 'usa', 'uk',
}

# Common stopwords, titles, time words to suppress
SUPPRESSED_TOKENS = {
    # Stopwords
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'you', 'your',
    'i', 'me', 'my', 'who', 'whom', 'which', 'what', 'where', 'when', 'why', 'how',
    
    # Titles (when alone)
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'gen', 'col', 'lt', 'sgt',
    'sir', 'lord', 'lady', 'rev', 'hon', 'judge', 'justice',
    
    # Month names
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    
    # Weekdays
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    
    # Common generic words that NER often picks up
    'government', 'state', 'department', 'office', 'bureau', 'agency',
    'committee', 'commission', 'council', 'board', 'center', 'institute',
    'american', 'soviet', 'russian', 'british', 'german', 'french',
    'communist', 'democratic', 'republican', 'liberal', 'conservative',
}

# Regex patterns for junk
JUNK_PATTERNS = [
    re.compile(r'^[0-9\s\-\.\/\(\)]+$'),      # Pure numbers/punctuation
    re.compile(r'^[A-Z]{1,2}[0-9]+$'),         # Codes like A1, B12
    re.compile(r'^[0-9]+[A-Za-z]{1,2}$'),      # Like 12th, 3rd
    re.compile(r'^[^a-zA-Z]*$'),               # No letters at all
    re.compile(r'^\W+$'),                      # Only non-word chars
    re.compile(r'^.{1,2}$'),                   # Too short (1-2 chars)
    re.compile(r'^[a-z]{1,3}$'),               # Very short lowercase
]


class NERGuardrails:
    """
    Safety guardrails for NER-based entity extraction.
    
    Prevents:
    - Auto-creating entities from ambiguous single-token surfaces
    - Auto-linking fuzzy matches for short/dangerous strings
    - Queue flooding from low-evidence surfaces
    - Generic word pollution
    """
    
    def __init__(self, conn=None):
        """
        Initialize guardrails.
        
        Args:
            conn: Database connection (optional, for loading blocklists)
        """
        self.conn = conn
        self._blocked_surfaces: Optional[Set[str]] = None
        self._matchable_aliases: Optional[Dict[str, dict]] = None
    
    def _load_blocked_surfaces(self) -> Set[str]:
        """Load blocked surfaces from database."""
        if self._blocked_surfaces is not None:
            return self._blocked_surfaces
        
        if not self.conn:
            return set()
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT DISTINCT variant_key 
            FROM ocr_variant_blocklist 
            WHERE variant_key IS NOT NULL
        """)
        self._blocked_surfaces = {row[0] for row in cur.fetchall()}
        return self._blocked_surfaces
    
    def _load_matchable_aliases(self) -> Dict[str, dict]:
        """Load matchable alias info from database."""
        if self._matchable_aliases is not None:
            return self._matchable_aliases
        
        if not self.conn:
            return {}
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT 
                alias_norm,
                entity_id,
                alias_class,
                is_auto_match,
                is_matchable
            FROM entity_aliases
            WHERE is_matchable = true
        """)
        
        self._matchable_aliases = {}
        for row in cur.fetchall():
            alias_norm = row[0]
            if alias_norm not in self._matchable_aliases:
                self._matchable_aliases[alias_norm] = {
                    'entity_ids': [],
                    'alias_class': row[2],
                    'is_auto_match': row[3],
                }
            self._matchable_aliases[alias_norm]['entity_ids'].append(row[1])
        
        return self._matchable_aliases
    
    # =========================================================================
    # JUNK DETECTION
    # =========================================================================
    
    def is_junk_surface(self, surface_norm: str) -> Tuple[bool, str]:
        """
        Check if a surface is obviously junk.
        
        Returns (is_junk, reason).
        """
        # Empty or whitespace
        if not surface_norm or not surface_norm.strip():
            return True, 'empty'
        
        # Too short
        if len(surface_norm) < 3:
            return True, 'too_short'
        
        # Matches junk patterns
        for pattern in JUNK_PATTERNS:
            if pattern.match(surface_norm):
                return True, f'junk_pattern:{pattern.pattern}'
        
        # Suppressed tokens
        if surface_norm.lower() in SUPPRESSED_TOKENS:
            return True, 'suppressed_token'
        
        # In blocklist
        blocked = self._load_blocked_surfaces()
        if surface_norm in blocked:
            return True, 'blocklisted'
        
        # Mostly digits
        alpha_count = sum(c.isalpha() for c in surface_norm)
        if alpha_count / len(surface_norm) < 0.5:
            return True, 'mostly_non_alpha'
        
        return False, ''
    
    def is_single_token(self, surface_norm: str) -> bool:
        """Check if surface is a single token."""
        return len(surface_norm.split()) == 1
    
    # =========================================================================
    # AUTO-LINK GUARDRAILS
    # =========================================================================
    
    def can_auto_link(
        self,
        surface_norm: str,
        match_info: Dict,
    ) -> Tuple[bool, str]:
        """
        Check if a surface can be auto-linked to an entity.
        
        Args:
            surface_norm: Normalized surface text
            match_info: Dict with keys:
                - match_type: 'exact' or 'fuzzy'
                - match_score: similarity score (0-1)
                - entity_id: matched entity ID
                - alias_class: class of the matched alias
                - is_auto_match: whether alias allows auto-matching
        
        Returns (can_link, reason).
        """
        match_type = match_info.get('match_type')
        match_score = match_info.get('match_score', 0)
        alias_class = match_info.get('alias_class')
        is_auto_match = match_info.get('is_auto_match', False)
        
        # Basic junk check
        is_junk, junk_reason = self.is_junk_surface(surface_norm)
        if is_junk:
            return False, f'junk:{junk_reason}'
        
        # EXACT MATCH RULES
        if match_type == 'exact':
            # Exact match must be to a matchable, auto-match alias
            if not is_auto_match:
                return False, 'exact_but_not_auto_match'
            
            # Danger classes need extra evidence even for exact
            if alias_class in DANGER_ALIAS_CLASSES:
                # Single-token danger class = queue, don't auto-link
                if self.is_single_token(surface_norm):
                    return False, f'single_token_danger_class:{alias_class}'
            
            return True, 'exact_match_auto'
        
        # FUZZY MATCH RULES
        if match_type == 'fuzzy':
            # Too short for fuzzy auto-link
            if len(surface_norm) < MIN_LENGTH_FOR_FUZZY_AUTO:
                return False, f'fuzzy_too_short:{len(surface_norm)}<{MIN_LENGTH_FOR_FUZZY_AUTO}'
            
            # Score too low
            if match_score < MIN_FUZZY_SCORE_FOR_AUTO:
                return False, f'fuzzy_score_too_low:{match_score:.2f}<{MIN_FUZZY_SCORE_FOR_AUTO}'
            
            # Danger classes NEVER auto-link on fuzzy
            if alias_class in DANGER_ALIAS_CLASSES:
                return False, f'fuzzy_danger_class:{alias_class}'
            
            # Single-token fuzzy = never auto-link
            if self.is_single_token(surface_norm):
                return False, 'fuzzy_single_token'
            
            # OK: long string, high score, safe class
            return True, f'fuzzy_high_confidence:{match_score:.2f}'
        
        # Unknown match type
        return False, f'unknown_match_type:{match_type}'
    
    # =========================================================================
    # QUEUE GUARDRAILS
    # =========================================================================
    
    def should_queue_for_review(
        self,
        surface_norm: str,
        stats: Dict,
    ) -> Tuple[bool, str]:
        """
        Check if a surface should be queued for human review.
        
        Prevents queue flooding by requiring minimum evidence.
        
        Args:
            surface_norm: Normalized surface text
            stats: Dict with keys:
                - doc_count: documents containing this surface
                - mention_count: total mentions
                - ner_score: NER acceptance score
                - ner_label: NER label (PERSON, ORG, etc.)
        
        Returns (should_queue, reason).
        """
        # Basic junk check
        is_junk, junk_reason = self.is_junk_surface(surface_norm)
        if is_junk:
            return False, f'junk:{junk_reason}'
        
        doc_count = stats.get('doc_count', 0)
        mention_count = stats.get('mention_count', 0)
        ner_score = stats.get('ner_score', 0)
        
        # Must meet minimum evidence thresholds
        has_doc_evidence = doc_count >= MIN_DOCS_FOR_QUEUE
        has_mention_evidence = mention_count >= MIN_MENTIONS_FOR_QUEUE
        has_length = len(surface_norm) >= MIN_SPAN_LENGTH_FOR_QUEUE
        has_score = ner_score >= MIN_NER_SCORE_FOR_QUEUE
        
        # Need EITHER doc evidence OR mention evidence
        # AND length AND score
        if not (has_doc_evidence or has_mention_evidence):
            return False, f'insufficient_evidence:docs={doc_count},mentions={mention_count}'
        
        if not has_length:
            return False, f'too_short:{len(surface_norm)}<{MIN_SPAN_LENGTH_FOR_QUEUE}'
        
        if not has_score:
            return False, f'low_ner_score:{ner_score:.2f}<{MIN_NER_SCORE_FOR_QUEUE}'
        
        return True, f'queued:docs={doc_count},mentions={mention_count}'
    
    # =========================================================================
    # CREATE_NEW GUARDRAILS
    # =========================================================================
    
    def can_auto_create_entity(
        self,
        surface_norm: str,
        ner_label: Optional[str],
        doc_count: int = 0,
        consistency: float = 0,
    ) -> Tuple[bool, str]:
        """
        Check if a surface can trigger auto-creation of a new entity.
        
        CRITICAL: This is the most dangerous operation. Be conservative but practical.
        
        Args:
            surface_norm: Normalized surface text
            ner_label: NER label (PERSON, ORG, GPE, etc.)
            doc_count: Number of documents containing this surface
            consistency: NER label consistency (0-1)
        
        Returns (can_create, reason).
        """
        # Basic junk check
        is_junk, junk_reason = self.is_junk_surface(surface_norm)
        if is_junk:
            return False, f'junk:{junk_reason}'
        
        # RULE 1: Single-token handling
        if self.is_single_token(surface_norm):
            # Exception: Known safe acronyms (FBI, CIA, KGB, etc.)
            if surface_norm.lower() in KNOWN_SAFE_ACRONYMS:
                if doc_count >= 5 and consistency >= 0.9:
                    return True, f'known_acronym:docs={doc_count}'
            
            # Single-token PERSON = surname trap, never auto-create
            if ner_label == 'PERSON':
                return False, 'single_token_person'
            
            # Other single tokens need VERY high evidence
            if doc_count >= 20 and consistency >= 0.95:
                return True, f'single_token_high_evidence:docs={doc_count}'
            
            return False, f'single_token_insufficient:docs={doc_count}'
        
        # RULE 2: Must have strong corpus evidence
        if doc_count < 10:
            return False, f'insufficient_doc_count:{doc_count}<10'
        
        if consistency < 0.8:
            return False, f'low_consistency:{consistency:.2f}<0.8'
        
        # RULE 3: Multi-token but still might be generic
        # Check for patterns like "THE COMMITTEE", "STATE DEPARTMENT"
        tokens = surface_norm.lower().split()
        generic_tokens = sum(1 for t in tokens if t in SUPPRESSED_TOKENS)
        if generic_tokens / len(tokens) > 0.5:
            return False, 'too_many_generic_tokens'
        
        # RULE 4: Label-based rules
        # GPE/LOC are fine for multi-token (New York, Soviet Union, etc.)
        safe_labels = {'PERSON', 'ORG', 'GPE', 'LOC', 'FAC'}
        if ner_label not in safe_labels:
            # NORP (nationalities like "American") should not create entities
            return False, f'unsafe_label_for_auto_create:{ner_label}'
        
        # Passed all checks
        return True, f'auto_create_allowed:docs={doc_count},consistency={consistency:.2f}'
    
    # =========================================================================
    # SINGLE-TOKEN PERSON HANDLING (Surname Trap)
    # =========================================================================
    
    def handle_single_token_person(
        self,
        surface_norm: str,
        match_info: Optional[Dict] = None,
    ) -> Tuple[str, str]:
        """
        Special handling for single-token PERSON labels.
        
        This is the most common source of errors: "SMITH" could be:
        - John Smith (person)
        - Smith & Co (org)
        - Smith County (place)
        - Generic surname (many people)
        
        Returns (action, reason) where action is:
        - 'auto_link': can auto-link (rare: exact match to unique entity)
        - 'queue_review': must go to human review
        - 'reject': don't even queue (too ambiguous)
        """
        if not self.is_single_token(surface_norm):
            return 'not_applicable', 'not_single_token'
        
        # Check if it's an exact match to a known alias
        aliases = self._load_matchable_aliases()
        alias_info = aliases.get(surface_norm.lower())
        
        if alias_info:
            entity_ids = alias_info['entity_ids']
            alias_class = alias_info['alias_class']
            is_auto = alias_info['is_auto_match']
            
            # Only auto-link if:
            # 1. It's an exact match
            # 2. It maps to exactly ONE entity (not ambiguous)
            # 3. It's NOT a surname-only alias
            # 4. The alias allows auto-matching
            if len(entity_ids) == 1 and is_auto:
                if alias_class not in {'person_surname', 'surname_only'}:
                    return 'auto_link', f'unique_safe_alias:{entity_ids[0]}'
            
            # Multiple entities or surname = queue
            return 'queue_review', f'ambiguous_or_surname:entities={len(entity_ids)},class={alias_class}'
        
        # No match = queue only if high evidence (handled elsewhere)
        return 'queue_review', 'no_match_single_token'


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_guardrails_to_candidates(
    conn,
    candidates: list,
) -> Tuple[list, list, list]:
    """
    Apply guardrails to a list of candidates.
    
    Returns:
        auto_link: candidates that can be auto-linked
        queue: candidates that should be queued for review
        reject: candidates that should be rejected
    """
    guardrails = NERGuardrails(conn)
    
    auto_link = []
    queue = []
    reject = []
    
    for cand in candidates:
        surface_norm = cand.get('surface_norm', '')
        
        # Check if junk
        is_junk, reason = guardrails.is_junk_surface(surface_norm)
        if is_junk:
            cand['reject_reason'] = reason
            reject.append(cand)
            continue
        
        # Check if can auto-link (if there's a match)
        if cand.get('match_type'):
            can_link, reason = guardrails.can_auto_link(surface_norm, cand)
            if can_link:
                cand['link_reason'] = reason
                auto_link.append(cand)
                continue
        
        # Check if should queue
        should_q, reason = guardrails.should_queue_for_review(
            surface_norm,
            {
                'doc_count': cand.get('doc_count', 1),
                'mention_count': cand.get('mention_count', 1),
                'ner_score': cand.get('ner_accept_score', 0.5),
            }
        )
        
        if should_q:
            cand['queue_reason'] = reason
            queue.append(cand)
        else:
            cand['reject_reason'] = reason
            reject.append(cand)
    
    return auto_link, queue, reject
