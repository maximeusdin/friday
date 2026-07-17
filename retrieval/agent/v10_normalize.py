"""
V10 Shared Normalizer — single source of truth for surface normalization.

All Stage A lookups, resolver tools, alias-index tools, and permission keys
use these functions.  If they drift, "permission exists but search dropped it"
bugs follow.

Three functions:
  normalize_surface_for_lookup(text) -> norm_key
      Used for all DB/tool lookups (Stage A, resolver, alias index, permission key).
  normalize_alias_surface(text) -> alias_norm
      Same as normalize_surface_for_lookup; alias-specific alias if needed later.
  normalize_for_display(text) -> display
      Minimal cleanup for user-facing display.  NOT used for keys.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Apostrophe normalization (smart quotes → ASCII)
# ---------------------------------------------------------------------------
_APOSTROPHE_MAP = str.maketrans({
    "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
    "\u201A": "'",   # SINGLE LOW-9 QUOTATION MARK
    "\u02BC": "'",   # MODIFIER LETTER APOSTROPHE
    "\u02BB": "'",   # MODIFIER LETTER TURNED COMMA
    "\u0060": "'",   # GRAVE ACCENT (sometimes used as apostrophe)
})

# ---------------------------------------------------------------------------
# Possessive stripping patterns
# ---------------------------------------------------------------------------
_POSSESSIVE_RE = re.compile(r"'s$|s'$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Edge punctuation (strip leading/trailing non-alphanumeric)
# ---------------------------------------------------------------------------
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def normalize_surface_for_lookup(text: str) -> str:
    """Canonical normalization for all DB/tool lookups and permission keys.

    Steps:
      1. Normalize Unicode (NFC)
      2. Normalize apostrophes (smart quotes → ASCII ')
      3. casefold() (locale-safe lowercase)
      4. Strip possessive ('s / s')
      5. Strip edge punctuation
      6. Collapse whitespace
      7. Strip outer whitespace
    """
    if not text:
        return ""
    # 1. Unicode NFC
    s = unicodedata.normalize("NFC", text)
    # 2. Apostrophes
    s = s.translate(_APOSTROPHE_MAP)
    # 3. casefold (NOT lower — handles locale-specific case e.g. ß → ss)
    s = s.casefold()
    # 4. Strip possessive
    s = _POSSESSIVE_RE.sub("", s)
    # 5. Edge punctuation
    s = _EDGE_PUNCT_RE.sub("", s)
    # 6. Collapse internal whitespace
    s = " ".join(s.split())
    return s


def normalize_alias_surface(text: str) -> str:
    """Alias-specific normalization.

    Currently identical to normalize_surface_for_lookup.  Kept as a separate
    entry point so alias-specific divergence (e.g. extra transliteration)
    can be added later without touching lookup code.
    """
    return normalize_surface_for_lookup(text)


def normalize_for_display(text: str) -> str:
    """Minimal cleanup for user-facing display.  NOT used for lookup keys."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)
    s = s.translate(_APOSTROPHE_MAP)
    s = " ".join(s.split())
    return s.strip()


# ---------------------------------------------------------------------------
# Stopword check (reusable across modules)
# ---------------------------------------------------------------------------

STOP_WORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "who", "whom", "whose", "what", "which", "that", "this", "these", "those",
    "where", "when", "how", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "out", "up", "down", "off",
    "and", "or", "but", "nor", "not", "so", "yet", "both", "either", "neither",
    "if", "then", "than", "as", "once", "while", "until", "since", "because",
    "no", "yes", "all", "any", "each", "every", "some", "such",
    "only", "also", "just", "very", "too", "more", "most", "much", "many",
    "new", "old", "first", "last", "same", "other", "own",
    "there", "here", "now", "well", "still", "already", "even",
    "say", "said", "says", "tell", "told", "ask", "asked",
    "get", "got", "give", "gave", "go", "went", "come", "came",
    "make", "made", "take", "took", "see", "saw", "know", "knew",
    "think", "thought", "find", "found", "want", "need",
    "use", "used", "try", "tried", "keep", "kept",
    "let", "put", "set", "run", "show", "help", "turn",
    "list", "provide", "describe", "explain", "evidence", "role",
    "according", "based", "regarding", "concerning",
})


def is_stopword_only(text: str) -> bool:
    """Return True if every token in *text* is a stopword."""
    tokens = text.split()
    return bool(tokens) and all(t in STOP_WORDS for t in tokens)


def is_mostly_stopwords(text: str, threshold: float = 0.8) -> bool:
    """Return True if >= threshold of tokens are stopwords."""
    tokens = text.split()
    if not tokens:
        return True
    stop_count = sum(1 for t in tokens if t in STOP_WORDS)
    return stop_count / len(tokens) >= threshold
