"""
Relative Date Expression Parser

Parses natural language date expressions into absolute date ranges.
Used for pre-LLM resolution of temporal references in queries.

Examples:
    "last month" -> (2026-12-01, 2026-12-31)
    "5 years before 1945" -> (1940-01-01, 1940-12-31)
    "two years ago" -> (2024-01-01, 2024-12-31)
    "during the 1940s" -> (1940-01-01, 1949-12-31)
    "after 1950" -> (1951-01-01, None)
    "before December 1945" -> (None, 1945-11-30)
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

try:
    from dateutil.relativedelta import relativedelta
    from dateutil.parser import parse as dateutil_parse
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False


@dataclass
class DateRange:
    """A resolved date range."""
    start: Optional[str]  # ISO format: YYYY-MM-DD
    end: Optional[str]    # ISO format: YYYY-MM-DD
    original_expression: str
    confidence: float = 1.0


# Number words to integers
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}

# Month names
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_number(s: str) -> Optional[int]:
    """Parse a number from string (digit or word)."""
    s = s.lower().strip()
    if s.isdigit():
        return int(s)
    return NUMBER_WORDS.get(s)


def _get_reference_date() -> date:
    """Get the reference date (today)."""
    return date.today()


def _format_date(d: date) -> str:
    """Format date as ISO string."""
    return d.strftime("%Y-%m-%d")


def _last_day_of_month(year: int, month: int) -> date:
    """Get the last day of a month."""
    if DATEUTIL_AVAILABLE:
        next_month = date(year, month, 1) + relativedelta(months=1)
        return next_month - relativedelta(days=1)
    else:
        # Manual calculation
        if month == 12:
            return date(year, 12, 31)
        return date(year, month + 1, 1) - timedelta(days=1)


def parse_relative_date(expr: str, reference_date: Optional[date] = None) -> Optional[DateRange]:
    """
    Parse a relative date expression into an absolute date range.
    
    Args:
        expr: The date expression to parse (e.g., "last month", "5 years before 1945")
        reference_date: Reference date for relative expressions (defaults to today)
    
    Returns:
        DateRange with start and end dates, or None if parsing fails.
    """
    if reference_date is None:
        reference_date = _get_reference_date()
    
    expr_lower = expr.lower().strip()
    
    # Try each pattern in order
    result = None
    result = result or _parse_relative_period(expr_lower, reference_date)
    result = result or _parse_years_before_after(expr_lower, reference_date)
    result = result or _parse_decade(expr_lower)
    result = result or _parse_absolute_year(expr_lower)
    result = result or _parse_before_after_year(expr_lower)
    result = result or _parse_month_year(expr_lower)
    
    if result:
        result.original_expression = expr
    
    return result


def _parse_relative_period(expr: str, ref: date) -> Optional[DateRange]:
    """Parse expressions like 'last month', 'this year', 'two years ago'."""
    if not DATEUTIL_AVAILABLE:
        return None
    
    # "last month", "last year", "last week"
    match = re.match(r"last\s+(month|year|week)", expr)
    if match:
        unit = match.group(1)
        if unit == "month":
            start = (date(ref.year, ref.month, 1) - relativedelta(months=1))
            end = date(ref.year, ref.month, 1) - relativedelta(days=1)
        elif unit == "year":
            start = date(ref.year - 1, 1, 1)
            end = date(ref.year - 1, 12, 31)
        elif unit == "week":
            start = ref - relativedelta(weeks=1, weekday=0)  # Monday of last week
            end = start + relativedelta(days=6)
        return DateRange(
            start=_format_date(start),
            end=_format_date(end),
            original_expression=expr,
        )
    
    # "this month", "this year"
    match = re.match(r"this\s+(month|year)", expr)
    if match:
        unit = match.group(1)
        if unit == "month":
            start = date(ref.year, ref.month, 1)
            end = _last_day_of_month(ref.year, ref.month)
        elif unit == "year":
            start = date(ref.year, 1, 1)
            end = date(ref.year, 12, 31)
        return DateRange(
            start=_format_date(start),
            end=_format_date(end),
            original_expression=expr,
        )
    
    # "N years/months ago"
    match = re.match(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(years?|months?)\s+ago", expr)
    if match:
        n = _parse_number(match.group(1))
        if n is None:
            return None
        unit = match.group(2).rstrip("s")
        if unit == "year":
            start = date(ref.year - n, 1, 1)
            end = date(ref.year - n, 12, 31)
        elif unit == "month":
            target = ref - relativedelta(months=n)
            start = date(target.year, target.month, 1)
            end = _last_day_of_month(target.year, target.month)
        return DateRange(
            start=_format_date(start),
            end=_format_date(end),
            original_expression=expr,
        )
    
    return None


def _parse_years_before_after(expr: str, ref: date) -> Optional[DateRange]:
    """Parse expressions like '5 years before 1945', 'two years after 1940'."""
    # "N years before YYYY"
    match = re.match(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+before\s+(\d{4})",
        expr
    )
    if match:
        n = _parse_number(match.group(1))
        year = int(match.group(2))
        if n is not None:
            target_year = year - n
            return DateRange(
                start=f"{target_year}-01-01",
                end=f"{target_year}-12-31",
                original_expression=expr,
            )
    
    # "N years after YYYY"
    match = re.match(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+after\s+(\d{4})",
        expr
    )
    if match:
        n = _parse_number(match.group(1))
        year = int(match.group(2))
        if n is not None:
            target_year = year + n
            return DateRange(
                start=f"{target_year}-01-01",
                end=f"{target_year}-12-31",
                original_expression=expr,
            )
    
    return None


def _parse_decade(expr: str) -> Optional[DateRange]:
    """Parse expressions like '1940s', 'the 1940s', 'during the 1940s'."""
    match = re.search(r"(?:the\s+)?(\d{3})0s", expr)
    if match:
        decade_start = int(match.group(1)) * 10
        return DateRange(
            start=f"{decade_start}-01-01",
            end=f"{decade_start + 9}-12-31",
            original_expression=expr,
        )
    
    # "early/mid/late 1940s"
    match = re.search(r"(early|mid|late)\s+(\d{3})0s", expr)
    if match:
        modifier = match.group(1)
        decade_start = int(match.group(2)) * 10
        if modifier == "early":
            return DateRange(
                start=f"{decade_start}-01-01",
                end=f"{decade_start + 3}-12-31",
                original_expression=expr,
            )
        elif modifier == "mid":
            return DateRange(
                start=f"{decade_start + 3}-01-01",
                end=f"{decade_start + 6}-12-31",
                original_expression=expr,
            )
        elif modifier == "late":
            return DateRange(
                start=f"{decade_start + 7}-01-01",
                end=f"{decade_start + 9}-12-31",
                original_expression=expr,
            )
    
    return None


def _parse_absolute_year(expr: str) -> Optional[DateRange]:
    """Parse expressions like 'in 1945', '1945'."""
    # Standalone year
    match = re.match(r"^(?:in\s+)?(\d{4})$", expr)
    if match:
        year = int(match.group(1))
        # Sanity check for reasonable historical years
        if 1800 <= year <= 2100:
            return DateRange(
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                original_expression=expr,
            )
    
    # "from YYYY to YYYY"
    match = re.search(r"from\s+(\d{4})\s+to\s+(\d{4})", expr)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        if start_year <= end_year and 1800 <= start_year <= 2100:
            return DateRange(
                start=f"{start_year}-01-01",
                end=f"{end_year}-12-31",
                original_expression=expr,
            )
    
    # "between YYYY and YYYY"
    match = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", expr)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        if start_year <= end_year and 1800 <= start_year <= 2100:
            return DateRange(
                start=f"{start_year}-01-01",
                end=f"{end_year}-12-31",
                original_expression=expr,
            )
    
    return None


def _parse_before_after_year(expr: str) -> Optional[DateRange]:
    """Parse expressions like 'before 1945', 'after 1950'."""
    # "before YYYY"
    match = re.search(r"before\s+(\d{4})", expr)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 2100:
            return DateRange(
                start=None,
                end=f"{year - 1}-12-31",
                original_expression=expr,
            )
    
    # "after YYYY"
    match = re.search(r"after\s+(\d{4})", expr)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 2100:
            return DateRange(
                start=f"{year + 1}-01-01",
                end=None,
                original_expression=expr,
            )
    
    return None


def _parse_month_year(expr: str) -> Optional[DateRange]:
    """Parse expressions like 'December 1945', 'in March 1942'."""
    # Month + Year pattern
    month_pattern = "|".join(MONTHS.keys())
    match = re.search(rf"(?:in\s+)?({month_pattern})\s+(\d{{4}})", expr, re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        year = int(match.group(2))
        month = MONTHS.get(month_name)
        if month and 1800 <= year <= 2100:
            start = date(year, month, 1)
            end = _last_day_of_month(year, month)
            return DateRange(
                start=_format_date(start),
                end=_format_date(end),
                original_expression=expr,
            )
    
    # "before Month YYYY"
    match = re.search(rf"before\s+({month_pattern})\s+(\d{{4}})", expr, re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        year = int(match.group(2))
        month = MONTHS.get(month_name)
        if month and 1800 <= year <= 2100:
            # End at the last day of the previous month
            target = date(year, month, 1)
            if DATEUTIL_AVAILABLE:
                end = target - relativedelta(days=1)
            else:
                from datetime import timedelta
                end = target - timedelta(days=1)
            return DateRange(
                start=None,
                end=_format_date(end),
                original_expression=expr,
            )
    
    return None


def extract_date_expressions(utterance: str) -> List[Tuple[str, DateRange]]:
    """
    Extract and parse all date expressions from an utterance.
    
    Returns a list of (matched_text, DateRange) tuples.
    """
    results = []
    
    # Patterns to search for (in order of specificity)
    patterns = [
        # "N years before/after YYYY"
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+(?:before|after)\s+\d{4}\b",
        # "from YYYY to YYYY"
        r"\bfrom\s+\d{4}\s+to\s+\d{4}\b",
        # "between YYYY and YYYY"
        r"\bbetween\s+\d{4}\s+and\s+\d{4}\b",
        # "early/mid/late 1940s"
        r"\b(?:early|mid|late)\s+\d{3}0s\b",
        # "the 1940s" or "1940s"
        r"\b(?:the\s+)?\d{3}0s\b",
        # "before/after YYYY"
        r"\b(?:before|after)\s+\d{4}\b",
        # "Month YYYY" or "in Month YYYY"
        r"\b(?:in\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{4}\b",
        # "last/this month/year"
        r"\b(?:last|this)\s+(?:month|year|week)\b",
        # "N years/months ago"
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:years?|months?)\s+ago\b",
        # "in YYYY"
        r"\bin\s+\d{4}\b",
    ]
    
    seen_spans = set()  # Track matched spans to avoid duplicates
    
    for pattern in patterns:
        for match in re.finditer(pattern, utterance, re.IGNORECASE):
            span = (match.start(), match.end())
            
            # Skip if this span overlaps with a previous match
            if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                continue
            
            matched_text = match.group(0)
            date_range = parse_relative_date(matched_text)
            
            if date_range:
                results.append((matched_text, date_range))
                seen_spans.add(span)
    
    return results


def resolve_dates_in_utterance(utterance: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Resolve all date expressions in an utterance.
    
    Returns:
        (date_context, resolved_info_string)
        - date_context: list of dicts with original expression and resolved range
        - resolved_info_string: human-readable summary of resolutions
    """
    date_context: List[Dict[str, Any]] = []
    
    extracted = extract_date_expressions(utterance)
    
    for matched_text, date_range in extracted:
        date_context.append({
            "expression": matched_text,
            "original": date_range.original_expression,
            "start": date_range.start,
            "end": date_range.end,
            "confidence": date_range.confidence,
        })
    
    # Build info string
    if date_context:
        lines = ["Resolved dates:"]
        for d in date_context:
            if d["start"] and d["end"]:
                lines.append(f"  - \"{d['expression']}\" -> {d['start']} to {d['end']}")
            elif d["start"]:
                lines.append(f"  - \"{d['expression']}\" -> from {d['start']}")
            elif d["end"]:
                lines.append(f"  - \"{d['expression']}\" -> until {d['end']}")
        resolved_info = "\n".join(lines)
    else:
        resolved_info = ""
    
    return date_context, resolved_info
