from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


# -----------------------
# AST node types
# -----------------------

@dataclass(frozen=True)
class QueryNode:
    """Base type for parsed query language nodes."""


@dataclass(frozen=True)
class Exact(QueryNode):
    term: str


@dataclass(frozen=True)
class And(QueryNode):
    terms: Tuple[str, ...]


@dataclass(frozen=True)
class Near(QueryNode):
    a: str
    b: str
    window_words: int


# -----------------------
# Parsing helpers
# -----------------------

_FIND_RE = re.compile(
    r"""^\s*find\s*\(\s*(?P<body>.+?)\s*\)\s*$""",
    re.IGNORECASE | re.DOTALL,
)

# Matches:
#   "X"
#   "X AND Y"
#   "X NEAR Y"
_QUOTED_RE = re.compile(r'^\s*"(?P<q>.*)"\s*$', re.DOTALL)

# Matches optional named args after the quoted query inside find(...)
# Example:
#   find("X NEAR Y", window=20)
_ARGS_RE = re.compile(
    r"""^(?P<quoted>"(?:[^"\\]|\\.)*")\s*(?:,\s*(?P<args>.+))?$""",
    re.DOTALL,
)

# Very small arg parser: window=123 (optionally with spaces)
_WINDOW_RE = re.compile(r"""(?:^|,)\s*window\s*=\s*(?P<n>\d+)\s*(?:,|$)""", re.IGNORECASE)


def _unescape_quoted(s: str) -> str:
    """
    Input is the inside of quotes, already captured (may include \" or \\).
    We only support a minimal escape set for convenience.
    """
    s = s.replace(r"\\", "\\")
    s = s.replace(r"\"", '"')
    return s


def _split_top_level_and(q: str) -> List[str]:
    """
    Split on AND tokens at top level (no parentheses support in v1).
    We treat AND case-insensitively and require it to be surrounded by whitespace.
    """
    parts = re.split(r"\s+AND\s+", q, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _split_top_level_near(q: str) -> Optional[Tuple[str, str]]:
    """
    Split on a single NEAR token at top level.
    Returns (left, right) if exactly one NEAR is present, else None.
    """
    parts = re.split(r"\s+NEAR\s+", q, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    left = parts[0].strip()
    right = parts[1].strip()
    if not left or not right:
        return None
    return left, right


# -----------------------
# Public API
# -----------------------

class QueryParseError(ValueError):
    pass


def parse_user_query(s: str, *, default_near_window: int = 20) -> QueryNode:
    """
    Parse a tiny, explicit query language used for deterministic retrieval ops.

    Supported forms:
      - find("X")
      - find("X AND Y")
      - find("X AND Y AND Z")
      - find("X NEAR Y", window=N)

    If input does NOT match find(...), we treat it as Exact(term=s).

    Notes:
      - No OR / NOT / parentheses in v1 (can add later).
      - Intentionally kept this small to preserve auditability.
    """
    s = (s or "").strip()
    if not s:
        raise QueryParseError("Empty query")

    m = _FIND_RE.match(s)
    if not m:
        # Not using find(...); interpret as exact term
        return Exact(term=s)

    body = m.group("body").strip()
    m2 = _ARGS_RE.match(body)
    if not m2:
        raise QueryParseError('Expected: find("...") or find("...", window=N)')

    quoted = m2.group("quoted")
    args = m2.group("args") or ""

    qm = _QUOTED_RE.match(quoted)
    if not qm:
        raise QueryParseError('Expected quoted query string like "X" inside find(...)')

    q_raw = _unescape_quoted(qm.group("q")).strip()
    if not q_raw:
        raise QueryParseError("Empty quoted query in find(...)")

    # Parse window=... if provided (only meaningful for NEAR)
    window_words = default_near_window
    if args:
        wm = _WINDOW_RE.search(args)
        if wm:
            window_words = int(wm.group("n"))

    # Decide which operator the quoted query implies
    near_split = _split_top_level_near(q_raw)
    if near_split is not None:
        a, b = near_split
        return Near(a=a, b=b, window_words=window_words)

    and_parts = _split_top_level_and(q_raw)
    if len(and_parts) >= 2:
        return And(terms=tuple(and_parts))

    return Exact(term=q_raw)


def to_debug_string(node: QueryNode) -> str:
    """
    A stable, explicit representation for logging/debugging.
    (Separate from the original user string, which you should also log verbatim.)
    """
    if isinstance(node, Exact):
        return f'EXACT("{node.term}")'
    if isinstance(node, And):
        return "AND(" + ", ".join(f'"{t}"' for t in node.terms) + ")"
    if isinstance(node, Near):
        return f'NEAR("{node.a}", "{node.b}", window_words={node.window_words})'
    return f"UNKNOWN({type(node).__name__})"
