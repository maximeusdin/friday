#!/usr/bin/env python3
"""
NER Integration for OCR Entity Extraction

Integrates spaCy NER as a signal source without over-cleaning OCR text.
Follows conservative post-filtering to avoid junk floods.

Key principles:
1. NER as signal, not truth - enhances existing lexicon-based extraction
2. Aggressive post-filters for OCR tolerance
3. Deduplication against existing candidates
4. Offset preservation for citation accuracy
5. Uses canonical normalize_surface() for consistency with rest of pipeline
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)

# Import canonical normalizer - MUST use same normalization as rest of pipeline
try:
    from retrieval.normalization import normalize_for_fts as normalize_surface
except ImportError:
    # Fallback if normalization module not available
    def normalize_surface(s: str) -> str:
        return s.lower().strip()

# Date parsing imports
from datetime import date, datetime
import calendar

# Month name mappings for date parsing
MONTH_NAMES = {
    'january': 1, 'jan': 1, 'jan.': 1,
    'february': 2, 'feb': 2, 'feb.': 2,
    'march': 3, 'mar': 3, 'mar.': 3,
    'april': 4, 'apr': 4, 'apr.': 4,
    'may': 5,
    'june': 6, 'jun': 6, 'jun.': 6,
    'july': 7, 'jul': 7, 'jul.': 7,
    'august': 8, 'aug': 8, 'aug.': 8,
    'september': 9, 'sep': 9, 'sep.': 9, 'sept': 9, 'sept.': 9,
    'october': 10, 'oct': 10, 'oct.': 10,
    'november': 11, 'nov': 11, 'nov.': 11,
    'december': 12, 'dec': 12, 'dec.': 12,
}

# Try to import spacy
try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. NER features disabled.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# SpaCy label mapping to our coarse types
SPACY_LABEL_MAP = {
    'PERSON': 'person',
    'ORG': 'org',
    'GPE': 'place',
    'LOC': 'place',
    'FAC': 'place',
    'NORP': 'norp',  # Nationalities, religious/political groups - treat as weak
}

# Labels we accept for entity extraction (NORP is optional/weak)
ACCEPTED_LABELS = {'PERSON', 'ORG', 'GPE', 'LOC', 'FAC'}
WEAK_LABELS = {'NORP'}

# Date-related labels
DATE_LABELS = {'DATE', 'TIME'}

# Token constraints
MIN_TOKENS = 1
MAX_TOKENS = 6
MIN_CHARS = 3

# Context hint keywords (reuse from ocr_utils)
PERSON_HINTS = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'doctor', 'prof', 'professor',
    'gen', 'general', 'col', 'colonel', 'maj', 'major', 'capt', 'captain',
    'lt', 'lieutenant', 'sgt', 'sergeant', 'comrade', 'agent', 'officer',
    'senator', 'congressman', 'ambassador', 'secretary', 'director',
}

ORG_HINTS = {
    'department', 'ministry', 'bureau', 'agency', 'committee', 'commission',
    'office', 'division', 'section', 'institute', 'university', 'college',
    'corporation', 'company', 'inc', 'corp', 'ltd', 'foundation', 'association',
    'party', 'union', 'council', 'board', 'service', 'administration',
}

LOC_HINTS = {
    'in', 'from', 'to', 'at', 'near', 'of', 'city', 'state', 'country',
    'region', 'district', 'province', 'county', 'street', 'avenue', 'road',
}

# Covername/alias markers - indicates span might be intentional codename
CODENAME_MARKERS = {
    'codename', 'code name', 'codenamed', 'code-name', 'covername', 
    'cover name', 'pseudonym', 'alias', 'known as', 'aka', 'a.k.a',
    'called', 'nicknamed', 'designated', 'referred to as',
}

# Quote patterns for detecting quoted occurrences
QUOTE_CHARS = {'"', "'", '"', '"', ''', '''}

# Words to always reject as entities
REJECT_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'as',
    'this', 'that', 'these', 'those', 'it', 'he', 'she', 'they', 'we',
    'said', 'says', 'told', 'asked', 'replied', 'stated', 'reported',
    'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'fourth', 'fifth',
    'ussr', 'usa', 'uk', 'us',  # Too common in these documents
}


# =============================================================================
# TEXT PREPROCESSING WITH OFFSET MAPPING
# =============================================================================

@dataclass
class OffsetMapping:
    """Maps cleaned text positions back to original text positions."""
    original_text: str
    cleaned_text: str
    # Map from cleaned char index to original char index
    clean_to_original: List[int] = field(default_factory=list)
    
    def get_original_span(self, clean_start: int, clean_end: int) -> Tuple[int, int]:
        """Convert cleaned text span to original text span."""
        if not self.clean_to_original:
            return clean_start, clean_end
        
        orig_start = self.clean_to_original[clean_start] if clean_start < len(self.clean_to_original) else len(self.original_text)
        # For end, we want the position after the last character
        orig_end = self.clean_to_original[clean_end - 1] + 1 if clean_end > 0 and clean_end <= len(self.clean_to_original) else len(self.original_text)
        
        return orig_start, orig_end


def preprocess_ocr_for_ner(text: str) -> OffsetMapping:
    """
    Preprocess OCR text for NER without over-cleaning.
    
    Does:
    - Fix hyphenation across line breaks ("Rosen-\nberg" → "Rosenberg")
    - Collapse repeated whitespace
    - Strip obvious boilerplate lines
    - Keep original offsets via mapping
    
    Does NOT:
    - Correct spelling
    - Remove all punctuation
    - Lowercase (preserve case signals)
    """
    if not text:
        return OffsetMapping(original_text='', cleaned_text='', clean_to_original=[])
    
    # Build character mapping as we clean
    clean_chars = []
    clean_to_orig = []
    
    i = 0
    while i < len(text):
        char = text[i]
        
        # Fix hyphenation across line breaks: "word-\n" followed by lowercase
        if char == '-' and i + 1 < len(text):
            # Look ahead for newline followed by lowercase letter
            j = i + 1
            while j < len(text) and text[j] in ' \t':
                j += 1
            if j < len(text) and text[j] == '\n':
                j += 1
                while j < len(text) and text[j] in ' \t':
                    j += 1
                if j < len(text) and text[j].islower():
                    # Skip the hyphen and whitespace/newline
                    i = j
                    continue
        
        # Collapse repeated whitespace to single space
        if char in ' \t\r':
            if clean_chars and clean_chars[-1] != ' ':
                clean_chars.append(' ')
                clean_to_orig.append(i)
            i += 1
            continue
        
        # Keep newlines but normalize
        if char == '\n':
            if clean_chars and clean_chars[-1] != '\n' and clean_chars[-1] != ' ':
                clean_chars.append('\n')
                clean_to_orig.append(i)
            i += 1
            continue
        
        # Keep all other characters
        clean_chars.append(char)
        clean_to_orig.append(i)
        i += 1
    
    cleaned_text = ''.join(clean_chars)
    
    return OffsetMapping(
        original_text=text,
        cleaned_text=cleaned_text,
        clean_to_original=clean_to_orig
    )


# =============================================================================
# NER EXTRACTION
# =============================================================================

@dataclass
class NERSpan:
    """A span identified by NER."""
    surface: str
    start_char: int  # In ORIGINAL text
    end_char: int    # In ORIGINAL text
    label: str       # spaCy label (PERSON, ORG, etc.)
    entity_type: str # Our type (person, org, place)
    accept_score: float
    case_score: float
    context_score: float
    token_count: int
    context_hints: Dict[str, any] = field(default_factory=dict)
    is_weak_label: bool = False


@dataclass
class DateSpan:
    """A date span identified by NER."""
    surface: str
    start_char: int  # In ORIGINAL text
    end_char: int    # In ORIGINAL text
    label: str       # DATE or TIME
    date_start: Optional[str] = None  # ISO format YYYY-MM-DD or None if unparseable
    date_end: Optional[str] = None    # ISO format YYYY-MM-DD or None
    precision: str = 'unknown'  # day, month, year, range, unknown
    confidence: float = 0.5


def parse_date_surface(surface: str) -> Tuple[Optional[str], Optional[str], str, float]:
    """
    Parse a date surface text into structured date information.
    
    Args:
        surface: The date text (e.g., "June 1945", "23 June 1945", "1943-1945")
    
    Returns:
        (date_start, date_end, precision, confidence)
        - date_start: ISO format YYYY-MM-DD or None
        - date_end: ISO format YYYY-MM-DD or None  
        - precision: 'day', 'month', 'year', 'range', or 'unknown'
        - confidence: 0.0-1.0
    """
    surface_clean = surface.strip().lower()
    
    # Try to extract year(s)
    year_pattern = r'\b(1[89]\d{2}|20[0-2]\d)\b'
    years = re.findall(year_pattern, surface_clean)
    
    if not years:
        return None, None, 'unknown', 0.3
    
    # Extract month if present
    month = None
    for month_name, month_num in MONTH_NAMES.items():
        if month_name in surface_clean:
            month = month_num
            break
    
    # Extract day if present
    day = None
    day_pattern = r'\b([1-9]|[12]\d|3[01])\s*(?:st|nd|rd|th)?\b'
    day_matches = re.findall(day_pattern, surface_clean)
    if day_matches:
        # Take first reasonable day number
        for d in day_matches:
            d_int = int(d)
            if 1 <= d_int <= 31:
                day = d_int
                break
    
    # Also try patterns like "23 June" or "June 23"
    if not day:
        day_month_pattern = r'\b(\d{1,2})\s+(?:' + '|'.join(MONTH_NAMES.keys()) + r')'
        match = re.search(day_month_pattern, surface_clean)
        if match:
            d_int = int(match.group(1))
            if 1 <= d_int <= 31:
                day = d_int
        else:
            # Try "June 23" pattern
            month_day_pattern = r'(?:' + '|'.join(MONTH_NAMES.keys()) + r')\s+(\d{1,2})'
            match = re.search(month_day_pattern, surface_clean)
            if match:
                d_int = int(match.group(1))
                if 1 <= d_int <= 31:
                    day = d_int
    
    # Check for range patterns
    if len(years) >= 2:
        year_start = int(years[0])
        year_end = int(years[-1])
        if year_start < year_end:
            # It's a range
            try:
                date_start = date(year_start, 1, 1).isoformat()
                date_end = date(year_end, 12, 31).isoformat()
                return date_start, date_end, 'range', 0.9
            except ValueError:
                pass
    
    # Single year
    year = int(years[0])
    
    try:
        if day and month:
            # Full date
            d = date(year, month, day)
            return d.isoformat(), d.isoformat(), 'day', 1.0
        elif month:
            # Month precision
            last_day = calendar.monthrange(year, month)[1]
            date_start = date(year, month, 1).isoformat()
            date_end = date(year, month, last_day).isoformat()
            return date_start, date_end, 'month', 0.8
        else:
            # Year precision
            date_start = date(year, 1, 1).isoformat()
            date_end = date(year, 12, 31).isoformat()
            return date_start, date_end, 'year', 0.6
    except ValueError:
        # Invalid date
        return None, None, 'unknown', 0.3


class NERExtractor:
    """SpaCy-based NER with aggressive post-filtering for OCR."""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize NER extractor.
        
        Args:
            model_name: SpaCy model (en_core_web_lg recommended for speed,
                       en_core_web_trf for accuracy on clean text)
        """
        if not SPACY_AVAILABLE:
            raise RuntimeError("spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_lg")
        
        try:
            # Load with only NER enabled for speed
            self.nlp = spacy.load(model_name, disable=['parser', 'lemmatizer'])
            # Increase max length for large documents
            self.nlp.max_length = 2_000_000
        except OSError:
            raise RuntimeError(f"SpaCy model '{model_name}' not found. Run: python -m spacy download {model_name}")
        
        self.model_name = model_name
        
        # Stats tracking
        self.stats = {
            'raw_spans': 0,
            'after_label_filter': 0,
            'after_length_filter': 0,
            'after_case_filter': 0,
            'after_context_filter': 0,
            'after_score_filter': 0,
            'final_accepted': 0,
        }
    
    def reset_stats(self):
        """Reset extraction statistics."""
        for key in self.stats:
            self.stats[key] = 0
    
    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 50,
        n_process: int = 1,
        acceptance_threshold: float = 0.5,
    ) -> List[List[NERSpan]]:
        """
        Extract entities from multiple texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for spaCy pipe
            n_process: Number of processes (1 for safety with DB connections)
            acceptance_threshold: Minimum accept_score to keep span
        
        Returns:
            List of lists of NERSpan objects (one list per input text)
        """
        results = []
        
        # Preprocess all texts
        mappings = [preprocess_ocr_for_ner(text) for text in texts]
        cleaned_texts = [m.cleaned_text for m in mappings]
        
        # Process in batches
        docs = self.nlp.pipe(cleaned_texts, batch_size=batch_size, n_process=n_process)
        
        for doc, mapping in zip(docs, mappings):
            spans = self._extract_from_doc(doc, mapping, acceptance_threshold)
            results.append(spans)
        
        return results
    
    def extract(
        self,
        text: str,
        acceptance_threshold: float = 0.5,
    ) -> List[NERSpan]:
        """
        Extract entities from a single text.
        
        Args:
            text: Text to process
            acceptance_threshold: Minimum accept_score to keep span
        
        Returns:
            List of NERSpan objects
        """
        mapping = preprocess_ocr_for_ner(text)
        doc = self.nlp(mapping.cleaned_text)
        return self._extract_from_doc(doc, mapping, acceptance_threshold)
    
    def _extract_from_doc(
        self,
        doc: 'Doc',
        mapping: OffsetMapping,
        acceptance_threshold: float,
    ) -> List[NERSpan]:
        """Extract and filter entities from a spaCy Doc."""
        spans = []
        
        for ent in doc.ents:
            self.stats['raw_spans'] += 1
            
            # 1. Label filter
            if ent.label_ not in ACCEPTED_LABELS and ent.label_ not in WEAK_LABELS:
                continue
            self.stats['after_label_filter'] += 1
            
            entity_type = SPACY_LABEL_MAP.get(ent.label_)
            is_weak = ent.label_ in WEAK_LABELS
            
            # 2. Length/token constraints
            token_count = len(list(ent))
            if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
                continue
            if len(ent.text.strip()) < MIN_CHARS:
                continue
            
            # Reject if mostly digits/punct
            alpha_count = sum(1 for c in ent.text if c.isalpha())
            if alpha_count < len(ent.text) * 0.5:
                continue
            self.stats['after_length_filter'] += 1
            
            # 3. Case-shape constraints
            surface = ent.text.strip()
            case_score = self._compute_case_score(surface, token_count, entity_type)
            if case_score < 0.1:
                continue
            self.stats['after_case_filter'] += 1
            
            # 4. Context gating
            context_hints = self._extract_context_hints(doc, ent.start_char, ent.end_char)
            context_score = self._compute_context_score(context_hints, entity_type)
            
            # For weak labels (NORP), require strong context
            if is_weak and context_score < 0.3:
                continue
            self.stats['after_context_filter'] += 1
            
            # 5. Compute acceptance score
            # ner_accept_score = 0.5*case_score + 0.3*context_score + 0.2*quality_score
            quality_score = 0.5  # Default, can be adjusted based on doc quality
            accept_score = 0.5 * case_score + 0.3 * context_score + 0.2 * quality_score
            
            if accept_score < acceptance_threshold:
                continue
            self.stats['after_score_filter'] += 1
            
            # 6. Map back to original offsets
            orig_start, orig_end = mapping.get_original_span(ent.start_char, ent.end_char)
            
            # Get surface from original text for accuracy
            orig_surface = mapping.original_text[orig_start:orig_end].strip()
            if not orig_surface:
                orig_surface = surface
            
            spans.append(NERSpan(
                surface=orig_surface,
                start_char=orig_start,
                end_char=orig_end,
                label=ent.label_,
                entity_type=entity_type,
                accept_score=accept_score,
                case_score=case_score,
                context_score=context_score,
                token_count=token_count,
                context_hints=context_hints,
                is_weak_label=is_weak,
            ))
            self.stats['final_accepted'] += 1
        
        return spans
    
    def _compute_case_score(self, surface: str, token_count: int, entity_type: str) -> float:
        """
        Compute case-based acceptance score.
        
        Rewards:
        - TitleCase for person/org/place
        - ALLCAPS for cover names
        - Mixed case for multi-token
        
        Penalizes:
        - All lowercase single tokens (likely common words)
        """
        surface_lower = surface.lower()
        
        # Reject known bad words
        if surface_lower in REJECT_WORDS:
            return 0.0
        
        # Single token all lowercase = very low score
        if token_count == 1 and surface.islower():
            return 0.1
        
        # Multi-token all lowercase = low score
        if token_count > 1 and surface.islower():
            return 0.2
        
        # ALLCAPS = good for cover names
        if surface.isupper() and len(surface) >= 3:
            return 0.9
        
        # TitleCase = good for names
        words = surface.split()
        title_count = sum(1 for w in words if w and w[0].isupper())
        if title_count == len(words):
            return 0.85
        
        # Mixed case = moderate
        if title_count > 0:
            return 0.6
        
        return 0.3
    
    def _extract_context_hints(
        self,
        doc: 'Doc',
        start_char: int,
        end_char: int,
        window_tokens: int = 8,
    ) -> Dict[str, any]:
        """
        Extract context hints around a span.
        
        Returns dict with:
        - person: int (count of person hints)
        - org: int (count of org hints)
        - loc: int (count of location hints)
        - codename: int (count of codename/alias markers)
        - quoted: bool (whether span appears to be in quotes)
        - alias_marker: int (count of alias-specific markers)
        """
        hints = {
            'person': 0, 
            'org': 0, 
            'loc': 0, 
            'codename': 0, 
            'alias_marker': 0,
            'quoted': False,
        }
        
        # Get text around the span
        text = doc.text
        
        # Find window boundaries
        window_start = max(0, start_char - 100)
        window_end = min(len(text), end_char + 100)
        
        context = text[window_start:window_end].lower()
        context_raw = text[window_start:window_end]  # Keep case for quote detection
        
        # Count type hints
        for hint in PERSON_HINTS:
            if hint in context:
                hints['person'] += 1
        
        for hint in ORG_HINTS:
            if hint in context:
                hints['org'] += 1
        
        for hint in LOC_HINTS:
            if hint in context:
                hints['loc'] += 1
        
        # Count codename/alias markers
        for marker in CODENAME_MARKERS:
            if marker in context:
                hints['codename'] += 1
                if marker in ('alias', 'known as', 'aka', 'a.k.a', 'called'):
                    hints['alias_marker'] += 1
        
        # Check if span is quoted
        # Look for quote chars immediately before/after the span in original text
        span_in_context_start = start_char - window_start
        span_in_context_end = end_char - window_start
        
        # Check for quotes around span
        if span_in_context_start > 0 and span_in_context_end < len(context_raw):
            char_before = context_raw[span_in_context_start - 1] if span_in_context_start > 0 else ''
            char_after = context_raw[span_in_context_end] if span_in_context_end < len(context_raw) else ''
            
            if char_before in QUOTE_CHARS or char_after in QUOTE_CHARS:
                hints['quoted'] = True
        
        return hints
    
    def _compute_context_score(self, hints: Dict[str, int], entity_type: str) -> float:
        """Compute context-based score."""
        if entity_type == 'person':
            return min(1.0, hints['person'] * 0.3)
        elif entity_type == 'org':
            return min(1.0, hints['org'] * 0.2)
        elif entity_type == 'place':
            return min(1.0, hints['loc'] * 0.25)
        else:
            return 0.3  # Default for NORP etc.
    
    # =========================================================================
    # DATE EXTRACTION
    # =========================================================================
    
    def extract_dates_batch(
        self,
        texts: List[str],
        batch_size: int = 50,
        n_process: int = 1,
    ) -> List[List[DateSpan]]:
        """
        Extract dates from multiple texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for spaCy pipe
            n_process: Number of processes (1 for safety with DB connections)
        
        Returns:
            List of lists of DateSpan objects (one list per input text)
        """
        results = []
        
        # Preprocess all texts
        mappings = [preprocess_ocr_for_ner(text) for text in texts]
        cleaned_texts = [m.cleaned_text for m in mappings]
        
        # Process in batches
        docs = self.nlp.pipe(cleaned_texts, batch_size=batch_size, n_process=n_process)
        
        for doc, mapping in zip(docs, mappings):
            dates = self._extract_dates_from_doc(doc, mapping)
            results.append(dates)
        
        return results
    
    def extract_dates(self, text: str) -> List[DateSpan]:
        """
        Extract dates from a single text.
        
        Args:
            text: Text to process
        
        Returns:
            List of DateSpan objects
        """
        mapping = preprocess_ocr_for_ner(text)
        doc = self.nlp(mapping.cleaned_text)
        return self._extract_dates_from_doc(doc, mapping)
    
    def _extract_dates_from_doc(
        self,
        doc: 'Doc',
        mapping: OffsetMapping,
    ) -> List[DateSpan]:
        """Extract date spans from a spaCy Doc."""
        dates = []
        
        for ent in doc.ents:
            # Only process DATE and TIME entities
            if ent.label_ not in DATE_LABELS:
                continue
            
            surface = ent.text.strip()
            
            # Skip very short or obviously bad surfaces
            if len(surface) < 4:
                continue
            
            # Skip if it looks like just a number without context
            if surface.isdigit() and len(surface) < 4:
                continue
            
            # Parse the date surface
            date_start, date_end, precision, confidence = parse_date_surface(surface)
            
            # Map back to original offsets
            orig_start, orig_end = mapping.get_original_span(ent.start_char, ent.end_char)
            
            # Get surface from original text for accuracy
            orig_surface = mapping.original_text[orig_start:orig_end].strip()
            if not orig_surface:
                orig_surface = surface
            
            dates.append(DateSpan(
                surface=orig_surface,
                start_char=orig_start,
                end_char=orig_end,
                label=ent.label_,
                date_start=date_start,
                date_end=date_end,
                precision=precision,
                confidence=confidence,
            ))
        
        return dates


# =============================================================================
# DEDUPLICATION
# =============================================================================

def dedupe_ner_against_candidates(
    ner_spans: List[NERSpan],
    existing_candidates: List[Dict],
    overlap_threshold: float = 0.6,
) -> Tuple[List[NERSpan], List[Dict]]:
    """
    Deduplicate NER spans against existing candidates.
    
    If spaCy proposes a span that overlaps an existing candidate by >overlap_threshold:
    - Keep the existing candidate
    - Attach NER type hints as extra signals
    
    Also checks for surface_norm match (using canonical normalizer) to catch
    cases where offsets might differ slightly due to preprocessing.
    
    Args:
        ner_spans: Spans from NER
        existing_candidates: Existing candidates with 'start_char', 'end_char', 'surface_norm'
        overlap_threshold: Minimum overlap to consider duplicate
    
    Returns:
        (new_spans, enhanced_candidates) - spans that don't overlap, and
        candidates with NER signals attached
    """
    new_spans = []
    enhanced = []
    enhanced_ids = set()  # Track which candidates we've already enhanced
    
    # Build set of existing surface_norms for quick lookup
    existing_norms = {c.get('surface_norm', ''): c for c in existing_candidates if c.get('surface_norm')}
    
    for span in ner_spans:
        overlaps = False
        best_overlap = 0.0
        best_candidate = None
        
        # Check position overlap
        for cand in existing_candidates:
            overlap = compute_span_overlap(
                span.start_char, span.end_char,
                cand.get('start_char', 0), cand.get('end_char', 0)
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_candidate = cand
        
        # Also check surface_norm match (catches offset drift from preprocessing)
        span_norm = normalize_surface(span.surface)
        if span_norm in existing_norms and best_overlap < overlap_threshold:
            # Surface matches even if positions don't overlap well
            norm_candidate = existing_norms[span_norm]
            if norm_candidate.get('id') not in enhanced_ids:
                best_candidate = norm_candidate
                best_overlap = 1.0  # Treat as full overlap
        
        if best_overlap >= overlap_threshold:
            # Attach NER signals to existing candidate (including context features)
            if best_candidate and best_candidate.get('id') not in enhanced_ids:
                best_candidate['ner_label'] = span.label
                best_candidate['ner_type_hint'] = span.entity_type
                best_candidate['ner_accept_score'] = span.accept_score
                best_candidate['ner_context_features'] = span.context_hints  # Pass context hints
                enhanced.append(best_candidate)
                enhanced_ids.add(best_candidate.get('id'))
            overlaps = True
        
        if not overlaps:
            new_spans.append(span)
    
    return new_spans, enhanced


def compute_span_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
    """Compute overlap ratio between two spans."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_len = overlap_end - overlap_start
    span1_len = end1 - start1
    span2_len = end2 - start2
    
    if span1_len == 0 or span2_len == 0:
        return 0.0
    
    # Return max overlap ratio
    return max(overlap_len / span1_len, overlap_len / span2_len)


# =============================================================================
# TYPE-GATED RESOLUTION
# =============================================================================

def apply_ner_type_gating(
    candidates: List[Dict],
    alias_infos: List[Dict],
) -> List[Dict]:
    """
    Use NER type hints to gate alias retrieval.
    
    When ner_type_hint=person, boost or restrict search to:
    - person aliases/classes
    - disallow org-only classes unless exact match
    
    Args:
        candidates: Candidates with optional ner_type_hint
        alias_infos: Available aliases with entity_type
    
    Returns:
        Filtered/boosted alias list
    """
    result = []
    
    for cand in candidates:
        ner_hint = cand.get('ner_type_hint')
        
        if not ner_hint:
            # No NER hint, keep all aliases
            result.extend(alias_infos)
            continue
        
        for alias in alias_infos:
            alias_type = alias.get('entity_type', '').lower()
            
            # Type matching
            if ner_hint == 'person' and alias_type in ('person', 'cover_name', ''):
                alias['ner_boost'] = 0.05
                result.append(alias)
            elif ner_hint == 'org' and alias_type in ('org', 'organization', ''):
                alias['ner_boost'] = 0.05
                result.append(alias)
            elif ner_hint == 'place' and alias_type in ('place', 'location', ''):
                alias['ner_boost'] = 0.05
                result.append(alias)
            else:
                # Type mismatch - keep but no boost
                result.append(alias)
    
    return result


# =============================================================================
# CLUSTER-LEVEL NER AGGREGATION
# =============================================================================

def aggregate_ner_for_cluster(mentions: List[Dict]) -> Dict:
    """
    Aggregate NER information at cluster level for review exports.
    
    Returns:
        {
            'ner_label_counts': {'PERSON': 5, 'ORG': 2},
            'dominant_label': 'PERSON',
            'label_consistency': 0.71,  # 5/7
            'docs_with_ner': 4,
            'ner_confidence_avg': 0.65,
        }
    """
    label_counts = {}
    scores = []
    docs_with_ner = set()
    
    for m in mentions:
        label = m.get('ner_label')
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
            scores.append(m.get('ner_accept_score', 0.5))
            if m.get('document_id'):
                docs_with_ner.add(m['document_id'])
    
    total = sum(label_counts.values())
    dominant_label = max(label_counts, key=label_counts.get) if label_counts else None
    
    return {
        'ner_label_counts': label_counts,
        'dominant_label': dominant_label,
        'label_consistency': label_counts.get(dominant_label, 0) / total if total > 0 else 0,
        'docs_with_ner': len(docs_with_ner),
        'ner_confidence_avg': sum(scores) / len(scores) if scores else 0,
    }
