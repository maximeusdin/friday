"""
Search tab query parser: user query string -> primitives.

Produces existing primitives (TermPrimitive, PhrasePrimitive, OrGroupPrimitive,
ExcludeTermPrimitive, ExcludePhrasePrimitive, FilterCollectionPrimitive,
FilterDocumentPrimitive) for compatibility with scope compilation and tsquery.

Parser rules:
- Implicit AND between adjacent terms
- Unary NOT binds tighter
- Phrase + prefix mixing (e.g. "harry dex* white") rejected in MVP
- Field filters: collection:slug, doc:123 compile to Filter primitives
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from retrieval.primitives import (
    ExcludePhrasePrimitive,
    ExcludeTermPrimitive,
    FilterCollectionPrimitive,
    FilterDocumentPrimitive,
    OrGroupPrimitive,
    PhrasePrimitive,
    TermPrimitive,
    normalize_phrase,
    normalize_term,
)


class ExcludeOrGroupPrimitive:
    """NOT (A OR B OR ...). Used when expanding ExcludeTerm/ExcludePhrase with aliases."""

    def __init__(self, primitives: List[Any]) -> None:
        self.primitives = primitives


class SearchQueryParseError(ValueError):
    """Raised when search query parsing fails."""

    pass


def _tokenize(s: str) -> List[Tuple[str, str]]:
    """
    Tokenize query string into (type, value) pairs.
    Types: AND, OR, NOT, LPAREN, RPAREN, PHRASE, TERM, COLLECTION_FILTER, DOC_FILTER
    """
    s = s.strip()
    if not s:
        return []

    tokens: List[Tuple[str, str]] = []
    i = 0
    n = len(s)

    while i < n:
        # Skip whitespace
        if s[i].isspace():
            i += 1
            continue

        # Operators (case-insensitive)
        if i + 2 <= n and s[i : i + 3].upper() == "AND" and (
            i + 3 >= n or s[i + 3].isspace() or s[i + 3] in "()"
        ):
            tokens.append(("AND", "AND"))
            i += 3
            continue
        if i + 1 <= n and s[i : i + 2].upper() == "OR" and (
            i + 2 >= n or s[i + 2].isspace() or s[i + 2] in "()"
        ):
            tokens.append(("OR", "OR"))
            i += 2
            continue
        if i + 2 <= n and s[i : i + 3].upper() == "NOT" and (
            i + 3 >= n or s[i + 3].isspace() or s[i + 3] in "()"
        ):
            tokens.append(("NOT", "NOT"))
            i += 3
            continue

        # Parentheses
        if s[i] == "(":
            tokens.append(("LPAREN", "("))
            i += 1
            continue
        if s[i] == ")":
            tokens.append(("RPAREN", ")"))
            i += 1
            continue

        # Quoted phrase
        if s[i] == '"':
            j = i + 1
            while j < n and s[j] != '"':
                if s[j] == "\\":
                    j += 1
                j += 1
            if j >= n:
                raise SearchQueryParseError("Unclosed quote")
            phrase = s[i + 1 : j].replace('\\"', '"').strip()
            tokens.append(("PHRASE", phrase))
            i = j + 1
            continue

        # Field filters: collection:slug or doc:123
        if s[i].isalnum() or s[i] in "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] in "_-"):
                j += 1
            ident = s[i:j]
            if j < n and s[j] == ":":
                # field:value
                k = j + 1
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k] == '"':
                    # quoted value
                    end = k + 1
                    while end < n and s[end] != '"':
                        if s[end] == "\\":
                            end += 1
                        end += 1
                    val = s[k + 1 : end].replace('\\"', '"').strip()
                    k = end + 1
                else:
                    # unquoted value
                    k = j + 1
                    while k < n and s[k].isspace():
                        k += 1
                    start_val = k
                    while k < n and not s[k].isspace() and s[k] not in "()":
                        k += 1
                    val = s[start_val:k].strip()
                ident_lower = ident.lower()
                if ident_lower == "collection":
                    tokens.append(("COLLECTION_FILTER", val))
                elif ident_lower in ("doc", "document"):
                    try:
                        tokens.append(("DOC_FILTER", int(val)))
                    except ValueError:
                        raise SearchQueryParseError(f"doc filter requires integer, got: {val}")
                else:
                    # Unknown field, treat as term
                    tokens.append(("TERM", ident))
                i = k
            else:
                # Check for prefix (term*)
                if j < n and s[j] == "*":
                    # MVP: reject phrase + prefix mixing
                    tokens.append(("TERM", ident + "*"))
                    i = j + 1
                else:
                    tokens.append(("TERM", ident))
                    i = j
            continue

        # Single char or unknown
        i += 1

    return tokens


def _parse_or(tokens: List[Tuple[str, str]], pos: int) -> Tuple[Any, int]:
    """Parse OR expression (lowest precedence)."""
    left, pos = _parse_and(tokens, pos)
    while pos < len(tokens) and tokens[pos][0] == "OR":
        pos += 1
        right, pos = _parse_and(tokens, pos)
        if isinstance(left, OrGroupPrimitive):
            left = OrGroupPrimitive(primitives=left.primitives + [right])
        elif isinstance(right, OrGroupPrimitive):
            left = OrGroupPrimitive(primitives=[left] + right.primitives)
        else:
            left = OrGroupPrimitive(primitives=[left, right])
    return left, pos


def _parse_and(tokens: List[Tuple[str, str]], pos: int) -> Tuple[Any, int]:
    """Parse AND expression (implicit between adjacent)."""
    left, pos = _parse_not(tokens, pos)
    while pos < len(tokens):
        tok_type, tok_val = tokens[pos]
        if tok_type in ("OR", "RPAREN"):
            break
        if tok_type == "AND":
            pos += 1
        right, pos = _parse_not(tokens, pos)
        # Flatten into list for AND - we'll emit multiple primitives with AND semantics
        if isinstance(left, list):
            left.append(right)
        else:
            left = [left, right]
    if isinstance(left, list):
        if len(left) == 1:
            return left[0], pos
        # Multiple AND operands: return as list for later flattening
        return left, pos
    return left, pos


def _parse_not(tokens: List[Tuple[str, str]], pos: int) -> Tuple[Any, int]:
    """Parse NOT expression (unary, binds tight)."""
    if pos < len(tokens) and tokens[pos][0] == "NOT":
        pos += 1
        inner, pos = _parse_not(tokens, pos)
        if isinstance(inner, PhrasePrimitive):
            return ExcludePhrasePrimitive(value=inner.value), pos
        if isinstance(inner, TermPrimitive):
            return ExcludeTermPrimitive(value=inner.value), pos
        if isinstance(inner, OrGroupPrimitive):
            # NOT (A OR B) -> (NOT A) AND (NOT B) - each excluded
            excluded = [ExcludeTermPrimitive(value=p.value) if isinstance(p, TermPrimitive)
                        else ExcludePhrasePrimitive(value=p.value) if isinstance(p, PhrasePrimitive)
                        else p for p in inner.primitives]
            return excluded, pos
        raise SearchQueryParseError("NOT requires term, phrase, or OR group")
    return _parse_primary(tokens, pos)


def _parse_primary(tokens: List[Tuple[str, str]], pos: int) -> Tuple[Any, int]:
    """Parse primary: term, phrase, filter, or parenthesized expression."""
    if pos >= len(tokens):
        raise SearchQueryParseError("Unexpected end of query")

    tok_type, tok_val = tokens[pos]
    pos += 1

    if tok_type == "LPAREN":
        inner, pos = _parse_or(tokens, pos)
        if pos >= len(tokens) or tokens[pos][0] != "RPAREN":
            raise SearchQueryParseError("Missing closing parenthesis")
        pos += 1
        return inner, pos

    if tok_type == "PHRASE":
        if not tok_val.strip():
            raise SearchQueryParseError("Empty phrase")
        return PhrasePrimitive(value=tok_val), pos

    if tok_type == "TERM":
        if not tok_val.strip():
            raise SearchQueryParseError("Empty term")
        # MVP: reject prefix in phrase context; for now allow term*
        if tok_val.endswith("*"):
            # Prefix: strip * and pass as term (executor will use prefix tsquery)
            return TermPrimitive(value=tok_val[:-1]), pos
        return TermPrimitive(value=tok_val), pos

    if tok_type == "COLLECTION_FILTER":
        return FilterCollectionPrimitive(slug=tok_val), pos

    if tok_type == "DOC_FILTER":
        return FilterDocumentPrimitive(document_id=tok_val), pos

    raise SearchQueryParseError(f"Unexpected token: {tok_type} {tok_val!r}")


def _flatten_and(primitives: List[Any]) -> List[Any]:
    """Flatten nested AND lists into a single list."""
    result: List[Any] = []
    for p in primitives:
        if isinstance(p, list):
            result.extend(_flatten_and(p))
        else:
            result.append(p)
    return result


def parse_search_query(query: str) -> List[Any]:
    """
    Parse search query string into list of primitives.

    Supports: AND, OR, NOT, parentheses, quoted phrases, collection:slug, doc:123.
    Implicit AND between adjacent terms.
    """
    if not query or not query.strip():
        raise SearchQueryParseError("Empty query")

    tokens = _tokenize(query)
    if not tokens:
        raise SearchQueryParseError("No tokens")

    # Split into scope filters (to be first) and text primitives
    scope_primitives: List[Any] = []
    text_primitives: List[Any] = []

    pos = 0
    try:
        parsed, pos = _parse_or(tokens, pos)
        if pos < len(tokens):
            raise SearchQueryParseError(f"Unexpected token at position {pos}: {tokens[pos]}")

        # Flatten AND groups and separate scope vs text
        flat = _flatten_and([parsed] if not isinstance(parsed, list) else parsed)
        for p in flat:
            if isinstance(p, (FilterCollectionPrimitive, FilterDocumentPrimitive)):
                scope_primitives.append(p)
            else:
                text_primitives.append(p)

        return scope_primitives + text_primitives
    except (ValueError, IndexError) as e:
        if isinstance(e, SearchQueryParseError):
            raise
        raise SearchQueryParseError(str(e)) from e


def _escape_tsquery_param(val: str) -> str:
    """Escape special tsquery chars in parameter value."""
    for c in "&|!():*'":
        val = val.replace(c, " ")
    return re.sub(r"\s+", " ", val).strip()


def compile_search_primitives_to_tsquery(
    primitives: List[Any],
) -> Tuple[str, str, List[Any]]:
    """
    Compile search primitives to PostgreSQL tsquery (to_tsquery + phraseto_tsquery).

    Returns (tsquery_sql, debug_text, params).
    Uses 'simple' config for Search (no stemming).
    """
    from retrieval.primitives import (
        ExcludePhrasePrimitive,
        ExcludeTermPrimitive,
        OrGroupPrimitive,
        PhrasePrimitive,
        TermPrimitive,
    )

    # Filter to text-match primitives only
    text_prims = [
        p for p in primitives
        if isinstance(
            p,
            (TermPrimitive, PhrasePrimitive, OrGroupPrimitive, ExcludeTermPrimitive, ExcludePhrasePrimitive, ExcludeOrGroupPrimitive),
        )
    ]
    if not text_prims:
        return "to_tsquery('simple', '___nomatch___')", "___nomatch___", ["___nomatch___"]

    parts: List[str] = []
    params: List[Any] = []
    debug_parts: List[str] = []

    for p in text_prims:
        if isinstance(p, TermPrimitive):
            norm = normalize_term(p.value)
            if norm:
                escaped = _escape_tsquery_param(norm)
                if escaped:
                    # Multi-word terms (e.g. from alias expansion) need phraseto_tsquery
                    if " " in norm:
                        parts.append("phraseto_tsquery('simple', %s)")
                    else:
                        parts.append("to_tsquery('simple', %s)")
                    params.append(escaped)
                    debug_parts.append(norm)
        elif isinstance(p, PhrasePrimitive):
            words = normalize_phrase(p.value)
            if words:
                phrase = " ".join(words)
                escaped = _escape_tsquery_param(phrase)
                if escaped:
                    parts.append("phraseto_tsquery('simple', %s)")
                    params.append(escaped)
                    debug_parts.append(f'"{phrase}"')
        elif isinstance(p, ExcludeTermPrimitive):
            norm = normalize_term(p.value)
            if norm:
                escaped = _escape_tsquery_param(norm)
                if escaped:
                    if " " in norm:
                        parts.append("!! phraseto_tsquery('simple', %s)")
                    else:
                        parts.append("!! to_tsquery('simple', %s)")
                    params.append(escaped)
                    debug_parts.append(f"!{norm}")
        elif isinstance(p, ExcludePhrasePrimitive):
            words = normalize_phrase(p.value)
            if words:
                phrase = " ".join(words)
                escaped = _escape_tsquery_param(phrase)
                if escaped:
                    parts.append("!! phraseto_tsquery('simple', %s)")
                    params.append(escaped)
                    debug_parts.append(f'!"{phrase}"')
        elif isinstance(p, OrGroupPrimitive):
            or_parts: List[str] = []
            or_params: List[Any] = []
            or_debug: List[str] = []
            for sub in p.primitives:
                if isinstance(sub, TermPrimitive):
                    norm = normalize_term(sub.value)
                    if norm:
                        escaped = _escape_tsquery_param(norm)
                        if escaped:
                            if " " in norm:
                                or_parts.append("phraseto_tsquery('simple', %s)")
                            else:
                                or_parts.append("to_tsquery('simple', %s)")
                            or_params.append(escaped)
                            or_debug.append(norm)
                elif isinstance(sub, PhrasePrimitive):
                    words = normalize_phrase(sub.value)
                    if words:
                        phrase = " ".join(words)
                        escaped = _escape_tsquery_param(phrase)
                        if escaped:
                            or_parts.append("phraseto_tsquery('simple', %s)")
                            or_params.append(escaped)
                            or_debug.append(f'"{phrase}"')
            if or_parts:
                or_sql = " || ".join(f"({x})" for x in or_parts)
                parts.append(f"({or_sql})")
                params.extend(or_params)
                debug_parts.append("(" + " OR ".join(or_debug) + ")")
        elif isinstance(p, ExcludeOrGroupPrimitive):
            or_parts = []
            or_params = []
            or_debug = []
            for sub in p.primitives:
                if isinstance(sub, TermPrimitive):
                    norm = normalize_term(sub.value)
                    if norm:
                        escaped = _escape_tsquery_param(norm)
                        if escaped:
                            if " " in norm:
                                or_parts.append("phraseto_tsquery('simple', %s)")
                            else:
                                or_parts.append("to_tsquery('simple', %s)")
                            or_params.append(escaped)
                            or_debug.append(norm)
                elif isinstance(sub, PhrasePrimitive):
                    words = normalize_phrase(sub.value)
                    if words:
                        phrase = " ".join(words)
                        escaped = _escape_tsquery_param(phrase)
                        if escaped:
                            or_parts.append("phraseto_tsquery('simple', %s)")
                            or_params.append(escaped)
                            or_debug.append(f'"{phrase}"')
            if or_parts:
                or_sql = " || ".join(f"({x})" for x in or_parts)
                parts.append(f"!! ({or_sql})")
                params.extend(or_params)
                debug_parts.append("!(" + " OR ".join(or_debug) + ")")

    if not parts:
        return "to_tsquery('simple', '___nomatch___')", "___nomatch___", ["___nomatch___"]

    # Combine with AND
    sql = " && ".join(f"({p})" for p in parts)
    debug_text = " AND ".join(debug_parts)
    return sql, debug_text, params
