"""
Detect "all instances" / exhaustive search intent from user messages.

When the user asks for "all instances of X", "every mention", "without exception", etc.,
we should auto-run Search instead of Chat retrieval (which returns top-k, not exhaustive).
"""
import re
from typing import Optional, Tuple


# Phrases that indicate exhaustive enumeration intent
_EXHAUSTIVE_CUES = [
    r"\ball\s+instances?\s+of\b",
    r"\bevery\s+mention\b",
    r"\bevery\s+reference\b",
    r"\bwithout\s+exception\b",
    r"\bexhaustive\b",
    r"\bcomplete\s+list\b",
    r"\ball\s+mentions?\b",
    r"\ball\s+references?\b",
    r"\beach\s+and\s+every\b",
    r"\bwithout\s+missing\b",
]

_EXHAUSTIVE_PATTERN = re.compile(
    "|".join(f"({p})" for p in _EXHAUSTIVE_CUES),
    re.IGNORECASE,
)


def _extract_query_from_all_instances(text: str) -> Optional[str]:
    """
    Extract the entity/query from "all instances of X" style messages.
    Returns the X part, or None if we can't extract it meaningfully.
    """
    text = (text or "").strip()
    if not text or len(text) < 5:
        return None

    # "all instances of Harry Dexter White" -> "Harry Dexter White"
    m = re.search(r"\ball\s+instances?\s+of\s+(.+?)(?:\.|$|\?)", text, re.IGNORECASE | re.DOTALL)
    if m:
        q = m.group(1).strip()
        if len(q) >= 2:
            return q

    # "every mention of X"
    m = re.search(r"\bevery\s+mention\s+(?:of\s+)?(.+?)(?:\.|$|\?)", text, re.IGNORECASE | re.DOTALL)
    if m:
        q = m.group(1).strip()
        if len(q) >= 2:
            return q

    # "exhaustive list of X" / "complete list of X"
    m = re.search(r"(?:exhaustive|complete)\s+list\s+(?:of\s+)?(.+?)(?:\.|$|\?)", text, re.IGNORECASE | re.DOTALL)
    if m:
        q = m.group(1).strip()
        if len(q) >= 2:
            return q

    # "all mentions of X" / "all references to X"
    m = re.search(r"\ball\s+(?:mentions?|references?)\s+(?:of|to)\s+(.+?)(?:\.|$|\?)", text, re.IGNORECASE | re.DOTALL)
    if m:
        q = m.group(1).strip()
        if len(q) >= 2:
            return q

    # Fallback: if message contains exhaustive cue, use the whole message as query
    # (minus the cue phrase) - e.g. "without exception: Harry Dexter White"
    if _EXHAUSTIVE_PATTERN.search(text):
        # Remove the cue and use rest
        cleaned = _EXHAUSTIVE_PATTERN.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) >= 2:
            return cleaned

    return None


def detect_all_instances_intent(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if the user is asking for exhaustive enumeration ("all instances").

    Returns:
        (is_all_instances, extracted_query)
        - is_all_instances: True if the message indicates exhaustive intent
        - extracted_query: The entity/query to search for, or None
    """
    text = (text or "").strip()
    if not text:
        return False, None

    if not _EXHAUSTIVE_PATTERN.search(text):
        return False, None

    query = _extract_query_from_all_instances(text)
    return True, query


def is_query_too_broad_for_auto_run(query: str, scope_mode: str) -> bool:
    """
    Throttle: don't auto-run Search for ultra-broad full-archive queries.

    Returns True if we should NOT auto-run (show "Run exhaustive search" button instead).
    """
    if scope_mode != "full_archive":
        return False  # Custom scope is fine to auto-run

    q = (query or "").strip()
    if not q:
        return True

    # Single short token (e.g. "white") -> throttle
    tokens = re.findall(r"[a-zA-Z0-9]+", q)
    if len(tokens) == 1 and len(tokens[0]) < 5:
        return True

    # Phrase or multiple terms -> safe to auto-run
    return False
