#!/usr/bin/env python3
"""
Ingest an expert-curated concordance index PDF into Postgres tables:
  concordance_sources, concordance_entries, entities, entity_aliases, entity_links, entity_citations

This version is layout-aware (indent / x-position based via pdfplumber) with a regex fallback,
while preserving all functionality from your prior script:

- marker-based slicing ("The Index" on page 7, 1-based)
- lossless concordance_entries.raw_text
- parsing: aliases, links, citations, scoped "As X:" / "Translated as X:" blocks
- dry-run mode (prints parsed objects)
- DB upsert/ensure helpers
- LIMIT, verbose
- --segment {auto,layout,regex} (default: auto)
- --compare prints a side-by-side comparison of regex vs layout segmentation

FIXES INCLUDED (based on your latest issues):
1) Footnote mode now properly triggers on dash-separator lines (ordering bug fixed).
2) De-duplicated regex definitions (no shadowed constants).
3) Layout headword detection stricter (guards against As/Translated, footnote numbering, lowercase continuations, bare ints).
4) Footnote/biblio blocks:
   - Layout segmentation: skips footnote blocks entirely (never become entries).
   - Parsing: ignores footnote tail sections for alias/citation parsing, but keeps raw_text lossless.
5) Citation chunking: keeps citations that mention Venona/Vassiliev and look citation-like, while avoiding narrative sentences.
6) Entity type classification:
   - DOES NOT blindly treat all-caps acronyms as cover names.
   - Treat as cover_name if headword says cover name / entry looks like cover-name class / body indicates cover name.
7) Prevents self-links (from_entity_id == to_entity_id) from being inserted.

Dependencies:
- psycopg2
- pypdf or PyPDF2 (for marker finding / fallback)
- pdfplumber (optional; enables layout segmentation)
  pip install pdfplumber
"""

import warnings
# Suppress SyntaxWarnings about escape sequences in raw regex strings (must be before other imports)
# These are false positives - the patterns are correct for regex
warnings.filterwarnings('ignore', category=SyntaxWarning)

import os
import re
import argparse
import time
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import psycopg2


# -----------------------
# Footnote removal from raw text
# -----------------------

# Pattern to detect if a line looks like a new entry (not indented, starts with capital or quote)
# Defined here so it can be used by footnote removal functions
_RE_ENTRY_START_FOOTNOTE = re.compile(r'^[^\s]|^\s{0,10}[A-Z"""]')

def remove_footnotes_from_layout_lines(lines: List["PdfLine"]) -> List["PdfLine"]:
    """
    Remove footnotes from layout-extracted lines.
    
    Similar to remove_footnotes_from_text but works with PdfLine objects.
    Filters out lines that start footnotes and their continuations.
    Preserves entry continuations that come after footnotes.
    """
    output_lines = []
    i = 0
    
    # Patterns that indicate entry continuation (not footnote continuation)
    # Entry continuations often have:
    # - Section headers with colons: "Pseudonyms and work names:", "Cover name in Venona:"
    # - Citation patterns: "As X:", "As "Y":"
    # - Lists of names: "Michael Green, Alexander Hansen"
    # - Quoted names: "Bill", "Will", "William"
    entry_continuation_patterns = [
        r'^[^:]*:\s*[A-Z]',  # "Pseudonyms and work names: ..." or "Cover name in Venona: ..."
        r'^As\s+["""]?[A-Z]',  # "As Akhmerov:" or "As "Jung":"
        r'^[A-Z][a-z]+\s+and\s+[A-Z]',  # "Michael Green and Alexander Hansen"
        r'^[A-Z][a-z]+,\s+[A-Z]',  # "Michael Green, Alexander Hansen"
        r'^["""][A-Z]',  # "Bill", "Will", "William"
        r'^[A-Z][A-Z\s]+\[',  # "MAYOR [MER" - all caps with brackets
    ]
    entry_continuation_re = re.compile('|'.join(entry_continuation_patterns))
    
    while i < len(lines):
        line = lines[i]
        stripped = line.text.strip() if line.text else ""
        
        # Check if this line starts a footnote (number + period)
        if _RE_FOOTNOTE_START.match(stripped):
            # Skip this footnote line
            i += 1
            
            # Track if we've seen a period that might end the footnote
            footnote_lines_skipped = 0
            max_footnote_lines = 5  # Footnotes are typically short (1-5 lines)
            
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.text.strip() if next_line.text else ""
                
                # Empty lines end the footnote
                if not next_stripped:
                    break
                
                # If next line starts a new footnote, continue skipping
                if _RE_FOOTNOTE_START.match(next_stripped):
                    i += 1
                    continue
                
                # Check if this looks like a new entry
                # Entry starts are typically at left margin (low x0) or start with capital/quote
                is_indented = next_line.x0 > 50  # More than 50 points from left = indented
                looks_like_new_entry = (
                    next_line.x0 <= 50 and 
                    next_line.text and 
                    _RE_ENTRY_START_FOOTNOTE.match(next_line.text)
                )
                
                if looks_like_new_entry:
                    # This is a new entry, stop skipping
                    break
                
                # Check if this looks like entry continuation (even if indented)
                # Entry continuations often have patterns like "Pseudonyms:", "Cover name:", "As X:"
                if next_line.text and entry_continuation_re.match(next_stripped):
                    # This is entry continuation, stop skipping and keep it
                    break
                
                # Check if we've skipped too many lines (footnotes are short)
                footnote_lines_skipped += 1
                if footnote_lines_skipped > max_footnote_lines:
                    # Probably hit entry continuation, stop skipping
                    break
                
                # Check if line ends with period and looks like it completes a citation
                # If it's indented and ends with period, it might be the end of the footnote
                if is_indented and next_stripped.endswith('.'):
                    # Might be end of footnote, but could also be entry continuation
                    # If next non-empty line also looks like entry continuation, stop here
                    # Otherwise, this is probably footnote continuation
                    j = i + 1
                    while j < len(lines) and (not lines[j].text or not lines[j].text.strip()):
                        j += 1
                    if j < len(lines) and lines[j].text:
                        next_non_empty = lines[j].text.strip()
                        if entry_continuation_re.match(next_non_empty):
                            # Next line is entry continuation, so this line is too - stop skipping
                            break
                    # This looks like end of footnote, stop skipping
                    break
                
                # This is a continuation of the footnote, skip it
                i += 1
        else:
            # Not a footnote, keep the line
            output_lines.append(line)
            i += 1
    
    return output_lines


def remove_footnotes_from_text(text: str) -> str:
    """
    Remove footnotes from raw extracted text.
    
    Footnotes:
    - Always start with a number and period (e.g., "8. Lota, GRU i Atomnaya Bomba.")
    - May span multiple lines
    - May appear multiple in a row
    - If a footnote interrupts an entry, continuation on next page will be indented
    - New entries are not indented (entry starts are at left margin)
    
    Strategy:
    - Detect lines starting with number + period
    - Remove those lines and any continuation lines that look like citation text
    - Stop when we hit text that looks like entry continuation (e.g., "Pseudonyms", "Cover name", "As X:")
    - Or when we hit a new entry (not indented, starts with capital/quote)
    """
    lines = text.split("\n")
    output_lines = []
    i = 0
    
    # Patterns that indicate entry continuation (not footnote continuation)
    # Entry continuations often have:
    # - Section headers with colons: "Pseudonyms and work names:", "Cover name in Venona:"
    # - Citation patterns: "As X:", "As "Y":"
    # - Lists of names: "Michael Green, Alexander Hansen"
    # - Quoted names: "Bill", "Will", "William"
    entry_continuation_patterns = [
        r'^[^:]*:\s*[A-Z]',  # "Pseudonyms and work names: ..." or "Cover name in Venona: ..."
        r'^As\s+["""]?[A-Z]',  # "As Akhmerov:" or "As "Jung":"
        r'^[A-Z][a-z]+\s+and\s+[A-Z]',  # "Michael Green and Alexander Hansen"
        r'^[A-Z][a-z]+,\s+[A-Z]',  # "Michael Green, Alexander Hansen"
        r'^["""][A-Z]',  # "Bill", "Will", "William"
        r'^[A-Z][A-Z\s]+\[',  # "MAYOR [MER" - all caps with brackets
    ]
    entry_continuation_re = re.compile('|'.join(entry_continuation_patterns))
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check if this line starts a footnote (number + period)
        if _RE_FOOTNOTE_START.match(stripped):
            # Skip this footnote line
            i += 1
            
            # Track if we've seen a period that might end the footnote
            footnote_lines_skipped = 0
            max_footnote_lines = 5  # Footnotes are typically short (1-5 lines)
            
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                
                # Empty lines end the footnote
                if not next_stripped:
                    break
                
                # If next line starts a new footnote, continue skipping
                if _RE_FOOTNOTE_START.match(next_stripped):
                    i += 1
                    continue
                
                # Check if this looks like a new entry (not indented, starts with capital/quote)
                is_indented = len(next_line) - len(next_line.lstrip()) > 10  # More than 10 spaces = indented
                looks_like_new_entry = _RE_ENTRY_START_FOOTNOTE.match(next_line) and not is_indented
                
                if looks_like_new_entry:
                    # This is a new entry, stop skipping
                    break
                
                # Check if this looks like entry continuation (even if indented)
                # Entry continuations often have patterns like "Pseudonyms:", "Cover name:", "As X:"
                if entry_continuation_re.match(next_stripped):
                    # This is entry continuation, stop skipping and keep it
                    break
                
                # Check if we've skipped too many lines (footnotes are short)
                footnote_lines_skipped += 1
                if footnote_lines_skipped > max_footnote_lines:
                    # Probably hit entry continuation, stop skipping
                    break
                
                # Check if line ends with period and looks like it completes a citation
                # If it's indented and ends with period, it might be the end of the footnote
                if is_indented and next_stripped.endswith('.'):
                    # Might be end of footnote, but could also be entry continuation
                    # If next non-empty line also looks like entry continuation, stop here
                    # Otherwise, this is probably footnote continuation
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        next_non_empty = lines[j].strip()
                        if entry_continuation_re.match(next_non_empty):
                            # Next line is entry continuation, so this line is too - stop skipping
                            break
                    # This looks like end of footnote, stop skipping
                    break
                
                # This is a continuation of the footnote, skip it
                i += 1
        else:
            # Not a footnote, keep the line
            output_lines.append(line)
            i += 1
    
    return "\n".join(output_lines)


# -----------------------
# PDF extraction helpers (text-only fallback)
# -----------------------

def _extract_pdf_text_pages(pdf_path: str) -> List[str]:
    """Returns a list of page texts (best-effort). Prefers pypdf; falls back to PyPDF2."""
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(pdf_path)
        return [(p.extract_text() or "") for p in reader.pages]
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(pdf_path)
            return [(p.extract_text() or "") for p in reader.pages]
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract PDF text from {pdf_path}. Install pypdf (preferred) or PyPDF2. Error: {e}"
            )


def slice_index_text_from_pdf(
    pdf_path: str,
    marker: str = "The Index",
    marker_page_1based: int = 7,
) -> str:
    """
    Extracts text from the PDF starting AFTER the first occurrence of `marker`
    on `marker_page_1based`, plus all subsequent pages.

    Used for:
      - marker discovery
      - regex segmentation fallback
    """
    pages = _extract_pdf_text_pages(pdf_path)
    if marker_page_1based < 1 or marker_page_1based > len(pages):
        raise RuntimeError(f"marker_page_1based={marker_page_1based} out of range (PDF has {len(pages)} pages)")

    idx0 = marker_page_1based - 1
    page_text = pages[idx0] or ""

    norm = re.sub(r"\s+", " ", page_text)
    m = re.search(re.escape(marker), norm) or re.search(re.escape(marker), norm, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(f'Marker "{marker}" not found on PDF page {marker_page_1based}')

    raw_pos = page_text.find(marker)
    if raw_pos == -1:
        raw_pos = page_text.lower().find(marker.lower())

    sliced_first = page_text if raw_pos == -1 else page_text[raw_pos + len(marker):]
    tail_pages = pages[idx0 + 1:]
    return "\n".join([sliced_first] + tail_pages)


# -----------------------
# Layout-aware extraction (pdfplumber)
# -----------------------

@dataclass
class PdfLine:
    text: str
    x0: float
    page_no_1based: int


def _try_import_pdfplumber():
    try:
        import pdfplumber  # type: ignore
        return pdfplumber
    except Exception:
        return None


def _marker_line_reached(lines: List[PdfLine], marker: str) -> Optional[int]:
    marker_lc = marker.lower()
    for i, ln in enumerate(lines):
        if marker_lc in (ln.text or "").lower():
            return i
    return None


def extract_index_lines_layout(
    pdf_path: str,
    marker: str = "The Index",
    marker_page_1based: int = 7,
    y_tolerance: float = 2.5,
    verbose: bool = False,
) -> List[PdfLine]:
    """
    Layout extractor: returns lines after the marker, preserving x0 positions.

    Uses pdfplumber words with x0/top, clusters into lines by 'top' within y_tolerance,
    then joins words in x order to produce line.text and line.x0=min(x0).

    Raises RuntimeError if pdfplumber isn't installed or marker isn't found.
    """
    pdfplumber = _try_import_pdfplumber()
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed (pip install pdfplumber)")

    out: List[PdfLine] = []
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        if verbose:
            print(f"[layout extract] PDF has {n_pages} pages", file=sys.stderr)
        if marker_page_1based < 1 or marker_page_1based > n_pages:
            raise RuntimeError(f"marker_page_1based={marker_page_1based} out of range (PDF has {n_pages} pages)")

        p0 = marker_page_1based - 1
        marker_page = pdf.pages[p0]

        def page_to_lines(page, page_no_1based: int) -> List[PdfLine]:
            if verbose and page_no_1based % 10 == 0:
                print(f"[layout extract] Processing page {page_no_1based}...", file=sys.stderr)
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=True,
            ) or []

            if not words:
                # Fallback: no coordinates available; treat lines as x0=0
                txt = page.extract_text() or ""
                return [
                    PdfLine(text=ln, x0=0.0, page_no_1based=page_no_1based)
                    for ln in txt.split("\n")
                    if ln.strip()
                ]

            # cluster by top within tolerance
            words_sorted = sorted(words, key=lambda w: (float(w.get("top", 0.0)), float(w.get("x0", 0.0))))
            clusters: List[List[dict]] = []
            for w in words_sorted:
                top = float(w.get("top", 0.0))
                if not clusters:
                    clusters.append([w])
                    continue
                prev_top = float(clusters[-1][0].get("top", 0.0))
                if abs(top - prev_top) <= y_tolerance:
                    clusters[-1].append(w)
                else:
                    clusters.append([w])

            lines_out: List[PdfLine] = []
            for cl in clusters:
                cl_sorted = sorted(cl, key=lambda w: float(w.get("x0", 0.0)))
                txt = " ".join([w.get("text", "") for w in cl_sorted]).strip()
                if not txt:
                    continue
                x0 = min(float(w.get("x0", 0.0)) for w in cl_sorted)
                lines_out.append(PdfLine(text=txt, x0=x0, page_no_1based=page_no_1based))
            return lines_out

        if verbose:
            print(f"[layout extract] Extracting marker page {marker_page_1based}...", file=sys.stderr)
        marker_lines = page_to_lines(marker_page, marker_page_1based)
        idx = _marker_line_reached(marker_lines, marker)
        if idx is None:
            raise RuntimeError(f'Marker "{marker}" not found in layout text on page {marker_page_1based}')

        # Lines strictly after the marker line on the marker page
        out.extend(marker_lines[idx + 1:])
        if verbose:
            print(f"[layout extract] Found {len(marker_lines[idx + 1:])} lines after marker on page {marker_page_1based}", file=sys.stderr)

        # Subsequent pages
        pages_to_process = n_pages - (p0 + 1)
        if verbose:
            print(f"[layout extract] Processing {pages_to_process} subsequent pages...", file=sys.stderr)
        for pi in range(p0 + 1, n_pages):
            out.extend(page_to_lines(pdf.pages[pi], pi + 1))
            if verbose and (pi - p0) % 10 == 0:
                print(f"[layout extract] Processed {pi - p0}/{pages_to_process} pages, {len(out)} lines so far...", file=sys.stderr)

    if verbose:
        print(f"[layout extract] Total lines extracted: {len(out)}", file=sys.stderr)
    return out


# -----------------------
# Text normalization (parsing only)
# -----------------------

def normalize_for_parsing(text: str) -> str:
    """
    Best-effort cleanup for parsing:
    - join hyphenated line breaks: 'Anglo-\\nAmerican' -> 'AngloAmerican' (word char joins only)
    - normalize CRLF
    - trim trailing spaces
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join([ln.rstrip() for ln in text.split("\n")])
    return text


# -----------------------
# Shared regex primitives
# -----------------------

_RE_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$")
_RE_SEPARATOR = re.compile(r"^\s*[-—]{3,}\s*$")
_RE_LONG_DASH_LINE = re.compile(r"^\s*[—-]{5,}\s*$")

_RE_AS_LINE = re.compile(r"^(As|Translated as)\b", flags=re.IGNORECASE)

_RE_FOOTNOTE_START = re.compile(r"^\s*\d+\.\s+")     # "4. ..."
_RE_FOOTNOTE_LIKE = re.compile(r"^\s*\d+\.\s+\S")    # "4. X"
# Pattern to detect if a line looks like a new entry (not indented, starts with capital or quote)
_RE_ENTRY_START = re.compile(r'^[^\s]|^\s{0,10}[A-Z"""]')
_RE_BARE_SMALL_INT_LINE = re.compile(r"^\s*\d{1,3}\s*$")

# Regex segmentation entry-start candidates
_RE_COLON_START = re.compile(r'^[\"“\']?[A-Z0-9]\S.{0,220}:\s')
_RE_UNDECIPHERED_START = re.compile(r"^Undeciphered Name No\.\s*\d+\b")
_RE_SOURCE_NO_START = re.compile(r"^\d+\s*,\s*Source No\.\b")

_RE_PERIOD_START = re.compile(
    r'^[\"“\']?[^.\n]{1,220}\.\s+(?:Unidentified|Venona|Vassiliev|See|Cover|Translated)\b'
)

_RE_PERSON_DOT_START = re.compile(
    r'^[A-Z][A-Za-z\'\-]+,\s+[A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){0,4}\.\s+\S'
)


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _RE_PAGE_NUMBER.match(s):
        return True
    if _RE_SEPARATOR.match(s):
        return True
    if _RE_LONG_DASH_LINE.match(s):
        return True
    return False


# -----------------------
# Entry segmentation (regex fallback)
# -----------------------

def is_entry_start_regex(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if is_noise_line(s):
        return False
    if _RE_FOOTNOTE_LIKE.match(s) or _RE_FOOTNOTE_START.match(s):
        return False
    if _RE_AS_LINE.match(s):
        return False
    if s[0].islower():
        return False

    if _RE_UNDECIPHERED_START.match(s):
        return True
    if _RE_SOURCE_NO_START.match(s):
        return True
    if _RE_COLON_START.match(s):
        return True
    if _RE_PERSON_DOT_START.match(s):
        return True
    if _RE_PERIOD_START.match(s):
        return True
    return False


def segment_entries_regex(text: str) -> List[str]:
    lines = text.split("\n")
    blocks: List[List[str]] = []
    cur: List[str] = []

    for ln in lines:
        if is_noise_line(ln):
            continue

        if is_entry_start_regex(ln):
            if cur:
                blocks.append(cur)
            cur = [ln]
        else:
            if not cur:
                continue
            cur.append(ln)

    if cur:
        blocks.append(cur)

    out: List[str] = []
    for b in blocks:
        while b and not b[0].strip():
            b = b[1:]
        while b and not b[-1].strip():
            b = b[:-1]
        if not b:
            continue
        out.append("\n".join(b).strip())
    return out


# -----------------------
# Entry segmentation (layout-based)
# -----------------------

def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    v = sorted(vals)
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return v[f]
    return v[f] + (v[c] - v[f]) * (k - f)


def _looks_like_headword_text_layout(s: str) -> bool:
    """
    Strict guards for layout segmentation:
    - never start on As/Translated blocks
    - never start on footnote numbering
    - never start on noise / separators
    - never start on bare small ints
    - never start on lowercase continuation lines
    """
    t = s.strip()
    if not t:
        return False
    if _RE_AS_LINE.match(t):
        return False
    if _RE_FOOTNOTE_START.match(t) or _RE_FOOTNOTE_LIKE.match(t):
        return False
    if is_noise_line(t):
        return False
    if _RE_BARE_SMALL_INT_LINE.match(t):
        return False
    if t[0].islower():
        return False
    return True


def segment_entries_layout(
    lines: List[PdfLine],
    indent_delta: float = 6.0,
    verbose: bool = False,
) -> List[str]:
    """
    Segments using x0 indentation:
    - Compute left_margin as 5th percentile of x0 across non-empty lines.
    - Define headword if x0 <= left_margin + indent_delta AND passes guards.
    - Continuations are indented or are guard-block lines.

    Footnotes:
    - Enter footnote mode on dash separator lines OR numbered-footnote starts.
    - While in footnote mode: skip all lines until a valid new headword line appears.
    - Dash lines must trigger footnote mode BEFORE being filtered as noise (ordering matters).
    """
    usable = [ln for ln in lines if (ln.text and ln.text.strip())]
    if not usable:
        return []

    x0s = [ln.x0 for ln in usable]
    left_margin = _percentile(x0s, 0.05)
    threshold = left_margin + indent_delta

    if verbose:
        print(f"[layout] left_margin≈{left_margin:.2f}  threshold≈{threshold:.2f}  lines={len(usable)}")

    # Patterns that indicate entry continuation (not a new entry)
    # These can be indented (after footnotes) or at left margin
    # Must be specific to avoid matching actual new entries
    entry_continuation_patterns = [
        r'^Pseudonyms\s+and\s+work\s+names:',  # "Pseudonyms and work names:"
        r'^Cover\s+name\s+in\s+(Venona|Vassiliev)',  # "Cover name in Venona:" or "Cover name in Vassiliev's notebooks:"
        r'^Cover\s+names\s+in\s+(Venona|Vassiliev)',  # "Cover names in Venona:"
        r'^As\s+["""]?[A-Z]',  # "As Akhmerov:" or "As "Jung":"
        r'^[A-Z][a-z]+\s+and\s+[A-Z][a-z]+',  # "Michael Green and Alexander Hansen" (both capitalized words)
        r'^[A-Z][a-z]+,\s+[A-Z][a-z]+',  # "Michael Green, Alexander Hansen" (both capitalized words)
        r'^["""][A-Z][a-z]+["""]',  # "Bill", "Will", "William" (quoted capitalized word)
        r'^[A-Z][A-Z\s]+\[',  # "MAYOR [MER" - all caps with brackets
    ]
    entry_continuation_re = re.compile('|'.join(entry_continuation_patterns), re.IGNORECASE)
    
    def is_layout_headword_line(ln: PdfLine) -> bool:
        return (ln.x0 <= threshold) and _looks_like_headword_text_layout(ln.text)
    
    def is_entry_continuation(ln: PdfLine) -> bool:
        """Check if line looks like entry continuation (can be indented after footnotes)"""
        return entry_continuation_re.match(ln.text.strip()) is not None

    blocks: List[List[str]] = []
    cur: List[str] = []
    in_footnotes = False

    for ln in usable:
        line = ln.text
        txt = (line or "").strip()
        if not txt:
            continue

        # IMPORTANT: enter footnote mode BEFORE noise-line skipping
        if _RE_LONG_DASH_LINE.match(txt) or _RE_SEPARATOR.match(txt) or _RE_FOOTNOTE_START.match(txt):
            in_footnotes = True
            continue

        # Skip noise-ish lines (page numbers, separators, etc.)
        if is_noise_line(txt):
            continue

        # While in footnotes, skip everything until a new headword begins
        # BUT: if we see entry continuation patterns, exit footnote mode and treat as continuation
        if in_footnotes:
            if is_entry_continuation(ln):
                # This is entry continuation after a footnote, exit footnote mode and continue with current entry
                in_footnotes = False
                # Don't treat as new headword, just add to current entry
                if cur:
                    cur.append(line)
                else:
                    # No current entry, start a new one (shouldn't happen, but handle gracefully)
                    cur = [line]
                continue
            elif is_layout_headword_line(ln):
                in_footnotes = False
            else:
                continue

        # Normal entry handling
        if is_layout_headword_line(ln):
            # This is a real new entry headword
            if cur:
                blocks.append(cur)
            cur = [line]
        else:
            if not cur:
                continue
            cur.append(line)

    if cur:
        blocks.append(cur)

    out: List[str] = []
    for b in blocks:
        while b and not b[0].strip():
            b = b[1:]
        while b and not b[-1].strip():
            b = b[:-1]
        if not b:
            continue
        out.append("\n".join(b).strip())
    return out


# -----------------------
# Parsing primitives
# -----------------------

def _normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
              .replace("‘", "'").replace("’", "'")
              .replace("„", '"').replace("‟", '"'))


def strip_outer_quotes(s: str) -> str:
    s2 = s.strip()
    if len(s2) >= 2:
        if (s2[0] == '"' and s2[-1] == '"') or (s2[0] == "'" and s2[-1] == "'"):
            return s2[1:-1].strip()
    return s2


def remove_question_marks(text: str) -> str:
    """Remove question marks (uncertainty markers) from text."""
    return text.replace("?", "").strip()


def looks_like_person_head(head: str) -> bool:
    """Check if text looks like a person name (comma-delimited or title case with space)."""
    head = head.strip()
    # Comma-delimited: "Last, First"
    if "," in head:
        return True
    # Title case with space: "First Last" (2-4 words, all title case)
    words = head.split()
    if 2 <= len(words) <= 4:
        if all(w and w[0].isupper() for w in words):
            return True
    return False


def invert_comma_delimited_name(name: str) -> str:
    """
    Invert comma-delimited person name: "Rogov, Alexander" -> "Alexander Rogov"
    Also handles question marks: "Kalinin, ?" -> "Kalinin"
    """
    name = name.strip()
    if "," not in name:
        return name
    
    parts = [p.strip() for p in name.split(",", 1)]
    if len(parts) != 2:
        return name
    
    last_name, first_part = parts[0], parts[1]
    
    # Remove question marks (uncertainty markers)
    first_part = first_part.replace("?", "").strip()
    
    if not first_part:
        # Just last name: "Kalinin, ?" -> "Kalinin"
        return last_name
    
    # Invert: "Rogov, Alexander" -> "Alexander Rogov"
    return f"{first_part} {last_name}".strip()


def split_head_and_body(first_line: str) -> Tuple[str, str, str]:
    """
    Returns (head_raw, body_firstline_remainder, delimiter_used)
    delimiter_used is ':' or '.'
    """
    if ":" in first_line:
        left, right = first_line.split(":", 1)
        return left.strip(), right.strip(), ":"
    m = re.search(r"\.", first_line)
    if not m:
        return first_line.strip(), "", "."
    i = m.start()
    left = first_line[:i].strip()
    right = first_line[i + 1:].strip()
    return left, right, "."


def remove_trailing_descriptor_paren(head: str) -> str:
    m = re.search(r"\(([^)]{1,120})\)\s*$", head.strip())
    if not m:
        return head
    inner = m.group(1).lower()
    if any(k in inner for k in ["cover name", "russian original", "vassiliev", "venona", "original"]):
        return head[:m.start()].strip()
    return head


def extract_quoted_strings(s: str) -> List[str]:
    s = _normalize_quotes(s)
    out = re.findall(r'"([^"]+)"', s)
    return [x.strip() for x in out if x.strip()]


def extract_bracket_tokens(s: str) -> List[str]:
    s = _normalize_quotes(s)
    items: List[str] = []
    for inner in re.findall(r"\[([^\]]+)\]", s):
        inner = inner.strip()
        inner = strip_outer_quotes(inner.replace('"', '').replace("'", ""))
        parts = re.split(r"\s+and\s+|,\s*", inner)
        for p in parts:
            p = p.strip()
            if p:
                items.append(p)
    return items


def extract_head_synonyms(head_clean: str) -> List[str]:
    base = re.sub(r"\[[^\]]*\]", "", head_clean).strip()
    parts = re.split(r"\s+and\s+", base)
    return [strip_outer_quotes(p.strip()) for p in parts if p.strip()]


@dataclass
class ParsedDefinition:
    definition_text: str
    definition_type: Optional[str] = None  # e.g., "expansion", "gloss", "description", "original", "translation"
    notes: Optional[str] = None


def extract_definitions_from_body(body_text: str, entity_type: str) -> Tuple[List[ParsedDefinition], str]:
    """
    Extract definition(s) from body text, returning definitions and the remainder for citation parsing.
    
    Rules:
    - For colon entries, take the first sentence/clause as definition (unless it's "See ...", "Cover name ...", etc.)
    - Handle bilingual definitions (split on em-dash/en-dash)
    - Handle multiple definitions separated by semicolons/commas
    - Return remainder text (after definitions) for citation parsing
    
    Returns:
        (definitions_list, remainder_text)
    """
    definitions: List[ParsedDefinition] = []
    remainder = body_text
    
    # Skip if body starts with cross-reference or other non-definition patterns
    body_lower = body_text.strip().lower()
    if body_lower.startswith(("see ", "cover name", "as ", "translated as")):
        return definitions, remainder
    
    # Extract definition(s) - can span multiple sentences until we hit citation patterns
    # Look for definition text that ends when we see:
    # 1. Citation patterns like "Venona ..." or "Vassiliev ..." followed by document names and page numbers
    # 2. "Cover name in Venona/Vassiliev:" patterns (these are structured covername sections, not definitions)
    # 3. "As X:" patterns (these are scoped citation sections, not definitions)
    
    # Pattern: citation starts with "Venona" or "Vassiliev" followed by document name (like "USA Naval GRU", "Black Notebook")
    # and then page numbers. This distinguishes citations from definitions that mention "Venona analysts" etc.
    # Use regular string to avoid SyntaxWarning (Python sees \s as invalid escape in raw strings)
    citation_start_pattern = '\\b(Venona|Vassiliev(?:[’'']s)?)\\s+[A-Z][^,]*,\\s*\\d'
    
    # Pattern: "Cover name in Venona/Vassiliev:" - these are structured sections, not definitions
    covername_section_pattern = '\\bCover\\s+names?\\s+in\\s+(?:Venona|Vassiliev[’'']s?\\s+notebooks?)'
    
    # Pattern: "As X:" - these are scoped citation sections, not definitions
    # Require "As" to be capitalized (not lowercase "as" which is a common word)
    scoped_citation_pattern = '\\bAs\\s+["""""]?[^""""":]+["""""]?\\s*:'
    
    # Find where definition should end (earliest of: citation, covername section, or scoped citation)
    citation_match = re.search(citation_start_pattern, body_text, re.IGNORECASE)
    covername_match = re.search(covername_section_pattern, body_text, re.IGNORECASE)
    scoped_match = re.search(scoped_citation_pattern, body_text, re.IGNORECASE)
    
    # Find the earliest match
    end_positions = []
    if citation_match:
        end_positions.append(('citation', citation_match.start()))
    if covername_match:
        end_positions.append(('covername', covername_match.start()))
    if scoped_match:
        # Verify that "As" is capitalized OR lowercase "as" followed by uppercase word
        scoped_pos = scoped_match.start()
        scoped_text_at_pos = body_text[scoped_pos:scoped_pos+10]
        # Accept if: "As " (capitalized) OR "as " followed by uppercase letter (e.g., "as UCN/25:")
        if scoped_text_at_pos.startswith('As ') or (scoped_text_at_pos.startswith('as ') and len(scoped_text_at_pos) > 3 and scoped_text_at_pos[3].isupper()):
            end_positions.append(('scoped', scoped_pos))
    
    if end_positions:
        # Use the earliest position
        end_positions.sort(key=lambda x: x[1])
        end_type, end_pos = end_positions[0]
        # Extract everything before the end marker as potential definition
        definition_text = body_text[:end_pos].strip()
        remainder = body_text[end_pos:].strip()
    else:
        # No clear citation pattern - try to extract first sentence(s) that look like definition
        sentences = re.split(r'(\.\s+)', body_text)
        definition_parts = []
        for i in range(0, len(sentences), 2):  # sentences are at even indices
            sentence = sentences[i] if i < len(sentences) else ""
            if not sentence.strip():
                continue
            # Stop if sentence looks like a citation (Venona/Vassiliev + document name + page numbers)
            if re.match('^(Venona|Vassiliev(?:[’'']s)?)\\s+[A-Z]', sentence.strip(), re.IGNORECASE):
                # Check if it has page numbers (citation-like)
                if re.search(',\\s*\\d', sentence):
                    break
            definition_parts.append(sentence)
            if i + 1 < len(sentences):
                definition_parts.append(sentences[i + 1])  # Add the period/space separator
        
        definition_text = ''.join(definition_parts).strip()
        # Remainder is everything after the definition
        if definition_text:
            remainder = body_text[len(definition_text):].strip()
        else:
            remainder = body_text
    
    # Process definition_text if we found one
    if definition_text:
        # Check if this looks like a definition (not a citation)
        if len(definition_text.split()) >= 2:
            # Fix incomplete definitions that end mid-phrase
            # If definition ends with "referred to" or "also referred to", try to extend it
            definition_lower = definition_text.lower().strip()
            if definition_lower.endswith(('referred to', 'also referred to', 'also referred')):
                # Look for the complete phrase pattern: "referred to as X, Y, and Z"
                extended_match = re.search(
                    r'(?:also\s+)?referred\s+to\s+as\s+([^.]+)',
                    body_text,
                    re.IGNORECASE
                )
                if extended_match:
                    # Extend definition to include the full "referred to as" phrase
                    extended_end = extended_match.end()
                    # Find the end of the phrase (period, semicolon, or start of next major clause)
                    phrase_end_match = re.search(r'[.;]', body_text[extended_end:])
                    if phrase_end_match:
                        definition_text = body_text[:extended_end + phrase_end_match.start()].strip()
                    else:
                        # If no clear end, take up to the next sentence or citation
                        next_sentence = re.search(r'\.\s+[A-Z]', body_text[extended_end:])
                        if next_sentence:
                            definition_text = body_text[:extended_end + next_sentence.start()].strip()
                        else:
                            definition_text = body_text[:extended_end].strip()
            
            # Check for bilingual definition (em-dash/en-dash separator)
            dash_chars = r'[–—-]'
            dash_pattern = rf'\s+{dash_chars}\s+'
            
            if re.search(dash_pattern, definition_text):
                # Split into original and translation
                parts = re.split(dash_pattern, definition_text, 1)
                if len(parts) >= 2:
                    original = parts[0].strip().rstrip('.').strip()
                    translation = parts[1].strip().rstrip('.').strip()
                    # Remove trailing parentheticals
                    original = re.sub(r'\s*\([^)]*\)\s*$', '', original).strip()
                    translation = re.sub(r'\s*\([^)]*\)\s*$', '', translation).strip()
                    
                    if original and len(original.split()) >= 2:
                        definitions.append(ParsedDefinition(
                            definition_text=original,
                            definition_type="original",
                            notes="Original language definition"
                        ))
                    if translation and len(translation.split()) >= 2:
                        definitions.append(ParsedDefinition(
                            definition_text=translation,
                            definition_type="translation",
                            notes="English translation"
                        ))
            else:
                # Single definition (can be multi-sentence), remove trailing parentheticals
                defn_clean = re.sub(r'\s*\([^)]*\)\s*$', '', definition_text).strip().rstrip('.').strip()
                if defn_clean:
                    definitions.append(ParsedDefinition(
                        definition_text=defn_clean,
                        definition_type="expansion",
                        notes="Definitional expansion"
                    ))
    
    return definitions, remainder


def infer_referent_from_body_start(body: str, entry_key: Optional[str] = None) -> Optional[str]:
    b = body.strip()
    if not b:
        return None
    b_norm = _normalize_quotes(b)
    
    # If entry_key is provided and body starts with it, extract name from entry_key instead
    # This handles cases like entry_key="BARCH (cover name in Venona) Semen Kremer"
    # and body="BARCH (cover name in Venona) Semen Kremer. Venona London GRU..."
    # BUT: If the body contains "was identified" or "judged that" patterns, prefer those
    # (e.g., "BELKA [SQUIRREL] was identified... possibly Ann Sidorovich")
    has_was_pattern = bool(re.search(r'\b(was\s+(?:identified|possibly|probably|likely)|judged\s+that)', b_norm, flags=re.IGNORECASE))
    
    if entry_key:
        entry_key_norm = _normalize_quotes(entry_key.strip())
        
        # Special case: malformed entry keys like:
        #   "Beigel, Rose. Also know as Rose Arenal, wife of ..."
        # We must prefer the "Also know as ..." referent over the initial "Beigel, Rose" fragment.
        also_know_match = re.search(
            r'\balso\s+know\s+as\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            entry_key_norm,
            flags=re.IGNORECASE
        )
        if also_know_match:
            also_name = also_know_match.group(1).strip()
            if ',' in also_name:
                also_name = also_name.split(',', 1)[0].strip()
            if also_name and looks_like_person_head(also_name):
                return infer_referent_from_body_start(also_name, entry_key=None)
        
        # Always strip repeated headword prefixes from the body if present.
        # This is critical because test inputs (and some OCR exports) repeat:
        #   HEAD: HEAD: <definition...>
        # and we want to extract the referent from the definition, not from the cover-name headword.
        if b_norm.lower().startswith(entry_key_norm.lower()):
            remaining = b_norm[len(entry_key_norm):].strip()
            # Allow common delimiters after the repeated headword (":", ".", whitespace)
            remaining = re.sub(r'^[\s\.:;-]+', '', remaining)
            if remaining:
                b_norm = remaining
            else:
                return None
        
        # If entry_key encodes a referent after "(cover name ...)" like:
        #   "BARCH (cover name in Venona) Semen Kremer"
        # prefer that.
        if "(cover name" in entry_key_norm.lower() and not has_was_pattern:
            name_match = re.search(r'\(cover name[^)]+\)\s+(.+)', entry_key_norm, flags=re.IGNORECASE)
            if name_match:
                name_part = name_match.group(1).strip()
                if '.' in name_part:
                    name_part = name_part.split('.', 1)[0].strip()
                if name_part:
                    return infer_referent_from_body_start(name_part, entry_key=None)
    
    # If the cleaned body starts with a plain person name followed by a temporal/locative qualifier,
    # short-circuit early. This handles common patterns like:
    #   "Vasily Zarubin in mid- and late 1930s. Vassiliev ..."
    #   "Harold Smeltzer starting in October 1944. ..."
    lead_name = re.match(
        r'^((?:[A-Z]\.\s+)?[A-Z][a-z]+(?:\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+){1,2})\s+'
        r'(circa|prior to|after|before|from|until|since|during|in|starting in|beginning in)\b',
        b_norm
    )
    if lead_name:
        return strip_outer_quotes(lead_name.group(1).strip())
    
    # Check if body contains "was identified" or "judged that" patterns BEFORE period detection
    # If so, we'll handle those patterns separately and they take precedence
    has_was_pattern_in_body = bool(re.search(r'\b(was\s+(?:identified|possibly|probably|likely)|judged\s+that)', b_norm, flags=re.IGNORECASE))
    
    # Find the first period that ends the name (not an abbreviation).
    #
    # IMPORTANT: Even if the body contains "was identified..." later, we still want to capture the
    # first sentence when it contains a clean referent (e.g., "A. Slavyagin, KGB officer. ...").
    # The "was identified/possibly" fallback is then applied *conditionally* (only when the first
    # sentence is cover-name-like), rather than forcing us to keep scanning past the first period.
    first_sentence = ""
    i = 0
    while i < len(b_norm):
        if b_norm[i] == '.':
            # Check if this is an abbreviation (single letter before period)
            is_abbreviation = False
            if i > 0:
                # Find the start of the word before this period
                word_start = i - 1
                while word_start > 0 and b_norm[word_start-1].isalpha():
                    word_start -= 1
                word_before_period = b_norm[word_start:i].strip()
                # If it's a single letter, check if there's more text after
                if len(word_before_period) == 1 and word_before_period.isalpha():
                    # Check if there's a space and capital letter after (e.g., "J. Robert")
                    if i + 2 < len(b_norm) and b_norm[i+1] == ' ' and b_norm[i+2].isupper():
                        # It's an abbreviation that's part of a name - continue past it
                        is_abbreviation = True
                    elif i + 1 < len(b_norm):
                        # Check what comes after the period
                        next_char = b_norm[i+1]
                        if next_char == ' ':
                            # Space after period - check next word
                            if i + 2 < len(b_norm) and b_norm[i+2].isupper():
                                # Capital letter after space - likely part of name
                                is_abbreviation = True
                            elif i + 2 < len(b_norm) and b_norm[i+2].islower():
                                # Lowercase after space - likely sentence end
                                is_abbreviation = False
                            else:
                                # End of string or other - likely sentence end
                                is_abbreviation = False
                        else:
                            # No space after period - likely sentence end
                            is_abbreviation = False
                    else:
                        # End of string - likely sentence end
                        is_abbreviation = False
            
            if not is_abbreviation:
                # This is a real period - stop here
                first_sentence = b_norm[:i].strip()
                break
        i += 1
    
    # Fallback: if we didn't find a period, take up to first period (even if abbreviation).
    # If we have a "was" pattern anywhere in the body and couldn't find a clean sentence boundary,
    # take a wider window so the "was identified/possibly" extractor has enough context.
    if not first_sentence:
        if has_was_pattern_in_body:
            # For "was" patterns, take more text (up to 200 chars or next sentence-ending period)
            # This allows the pattern matching to find the full name
            first_sentence = b_norm[:min(200, len(b_norm))].strip()
        else:
            first_sentence = b_norm.split(".", 1)[0].strip()
    
    if not first_sentence:
        return None

    # Remove qualifiers from the beginning (likely, probably, etc.)
    # Also handle quoted qualifiers like "Probably"
    # Also handle "then" qualifier (e.g., "then "Meter"")
    qualifiers = (
        "likely", "probably", "possibly", "perhaps", "maybe", "unidentified", 
        "?", "described as", "initial", "abbreviation", "refers to", 
        "translated as", "see "
    )
    
    # Remove quoted qualifiers first (e.g., "Probably" or "Likely")
    first_sentence = re.sub(r'^["""](probably|likely|possibly|perhaps|maybe)["""]\s+', '', first_sentence, flags=re.IGNORECASE)
    
    # Remove "then" qualifier (can be before quotes or unquoted)
    # Handle "then "Name"" -> extract just "Name" (stop at closing quote)
    # Also handle "then "Name" [bracket]" -> extract just "Name"
    # Also handle "then METER and METRE" -> this should be handled at a higher level to create two aliases
    # Use a more specific pattern that handles the full quoted string
    then_quoted_match = re.match(r'^then\s+["""]([^"""]+)["""]', first_sentence, flags=re.IGNORECASE)
    if then_quoted_match:
        # Extract the quoted name
        quoted_name = then_quoted_match.group(1).strip()
        # Remove any brackets that might follow (e.g., "Meter" ["Metr"])
        remaining_after_match = first_sentence[then_quoted_match.end():].strip()
        # If there are brackets, we've already extracted the name, so use it
        first_sentence = quoted_name
    else:
        # Handle "then NAME" (unquoted) or "then NAME [BRACKET]"
        # Also handle "then METER and METRE" - for now, extract everything after "then"
        # The "and" splitting will be handled at a higher level when processing aliases
        if re.match(r'^then\s+', first_sentence, flags=re.IGNORECASE):
            first_sentence = re.sub(r'^then\s+', '', first_sentence, flags=re.IGNORECASE)
            # Remove any leading/trailing quotes that might be left
            first_sentence = strip_outer_quotes(first_sentence)
            # If there are brackets, extract just the part before brackets
            bracket_pos = first_sentence.find('[')
            if bracket_pos > 0:
                first_sentence = first_sentence[:bracket_pos].strip()
            # Remove quotes again after bracket removal
            first_sentence = strip_outer_quotes(first_sentence)
    
    # Update lowercase version after "then" handling
    first_sentence_lower = first_sentence.lower()
    
    # Then remove unquoted qualifiers
    for qualifier in qualifiers:
        if first_sentence_lower.startswith(qualifier):
            # Remove the qualifier and any following comma/colon
            remaining = first_sentence[len(qualifier):].strip()
            remaining = re.sub(r'^[,:\s]+', '', remaining)
            if remaining:
                first_sentence = remaining
            else:
                return None
            break
    
    # Remove any remaining quotes around the name
    first_sentence = strip_outer_quotes(first_sentence)
    
    # Fix incomplete quotes (e.g., """Glan" -> "Glan", """Sonya" -> "Sonya")
    # Handle cases where there are multiple opening quotes but no closing quotes
    # Try multiple patterns to catch different quote types
    first_sentence = re.sub(r'^["""""]+([^"""""]+)["""""]*$', r'\1', first_sentence)
    # Also handle cases with just opening quotes and no closing quotes
    first_sentence = re.sub(r'^["""]+([^"""]+)$', r'\1', first_sentence)
    # Handle mixed quote types (curly quotes and straight quotes)
    first_sentence = re.sub(r'^["""]+([^"""]+)$', r'\1', first_sentence)
    
    if not first_sentence:
        return None
    
    # Handle patterns like "X was identified as Y" or "X was possibly Y" or "judged that X was Y"
    # Extract the name after "was" or "was possibly" or similar
    # IMPORTANT: These patterns need to handle multi-part names with periods (e.g., "J. Robert Oppenheimer")
    # But prioritize the first sentence - only use "was identified as" if we don't have a good name from first_sentence
    # Search in the FIRST SENTENCE only to avoid matching later sentences
    was_patterns = [
        # Pattern 1: "judged that X was possibly Y" - extract Y (handle quoted qualifier)
        # Capture everything after "was possibly" until sentence-ending period (not abbreviation period)
        r'judged\s+that\s+\w+\s+was\s+["""]?(possibly|probably|likely)?["""]?\s+((?:[A-Z]\.\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)',
        # Pattern 2: "was identified as Y" or "was possibly Y"
        r'was\s+(?:identified\s+as|possibly|probably|likely)\s+["""]?((?:[A-Z]\.\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)',
        # Pattern 3: "was possibly Y" (quoted qualifier)
        r'was\s+["""]?(possibly|probably|likely)["""]?\s+((?:[A-Z]\.\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)',
    ]
    # Search in the first sentence only to avoid matching later sentences
    # Only use "was identified as" pattern if first_sentence doesn't already contain a good name.
    # In OCR/index text, the first sentence is often just the cover name itself (e.g., "BELKA [SQUIRREL] was identified..."),
    # in which case we DO want to fall back to the "was possibly/identified as" extractor.
    use_was_pattern = False
    if first_sentence:
        fs = first_sentence.strip()
        fs_lc = fs.lower()
        words = fs.split()
        looks_like_coverish_surface = (
            ("cover name" in fs_lc) or
            (fs.isupper() and len(words) <= 3) or
            bool(re.match(r"^[A-Z0-9][A-Z0-9'\/\-\s]*\s*(\[[^\]]+\])?$", fs)) or
            # e.g. "BELKA [SQUIRREL] was identified ...", where the first sentence begins with the cover name.
            bool(re.match(r"^[A-Z0-9]{2,}(?:\s*\[[^\]]+\])?\s+was\s+", fs))
        )
        if looks_like_coverish_surface:
            use_was_pattern = True
        elif len(words) < 3 or not any(w and w[0].isupper() for w in words):
            use_was_pattern = True
    else:
        use_was_pattern = True
    
    if use_was_pattern:
        for pattern in was_patterns:
            match = re.search(pattern, first_sentence if first_sentence else b_norm, flags=re.IGNORECASE)
            if match:
                # Get the last non-empty group (the name)
                name = None
                for i in range(match.lastindex, 0, -1):
                    candidate = match.group(i)
                    if candidate and candidate.strip() and not candidate.lower() in ('possibly', 'probably', 'likely'):
                        name = candidate.strip()
                        name = strip_outer_quotes(name)
                        if name:
                            break
                if name:
                    # The pattern should have captured the full name, but check if there's more after the match
                    # Look for sentence-ending period after the name
                    search_text = first_sentence if first_sentence else b_norm
                    match_end = match.end()
                    if match_end < len(search_text):
                        remaining = search_text[match_end:].strip()
                        # If remaining starts with a period followed by space and capital, it's sentence end
                        # Otherwise, if it starts with space and capital, it might be part of the name
                        if remaining and not remaining.startswith('.'):
                            # Check if there's more of the name (e.g., if pattern didn't capture everything)
                            # Look for pattern like "J. Robert Oppenheimer" where we might have only captured "J"
                            if name.endswith('.') and len(name) == 2 and name[0].isalpha():
                                # Single letter - look for continuation
                                next_word_match = re.match(r'\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', remaining)
                                if next_word_match:
                                    name = name + next_word_match.group(0)
                                    # Check for even more
                                    remaining_after = remaining[len(next_word_match.group(0)):].strip()
                                    if remaining_after and not remaining_after.startswith('.'):
                                        next_word_match2 = re.match(r'\s+([A-Z][a-z]+)', remaining_after)
                                        if next_word_match2:
                                            name = name + ' ' + next_word_match2.group(1)
                    # Stop at sentence-ending period if present
                    if '.' in name:
                        # Check if the period is at the end and followed by nothing (sentence end)
                        # or if it's an abbreviation (single letter before period)
                        period_pos = name.rfind('.')
                        if period_pos == len(name) - 1:
                            # Period at end - check if it's an abbreviation
                            if period_pos > 0 and len(name) == 2 and name[0].isalpha():
                                # Single letter abbreviation - keep it
                                pass
                            else:
                                # Remove trailing period (sentence end)
                                name = name[:-1].strip()
                    first_sentence = name
                    break
    
    # If the result starts with "unidentified" or other descriptive text, return None
    # (we don't want to create entities for "Unidentified Soviet intelligence source/agent")
    descriptive_starts = (
        "unidentified", "described as", "error for", "soviet intelligence source/agent",
        "soviet intelligence source", "intelligence source/agent", "intelligence source",
        "cover name", "likely", "probably", "possibly", "analysts", "judged", "was identified",
        "venona", "vassiliev", "notebook"
    )
    first_sentence_lower = first_sentence.lower()
    if any(first_sentence_lower.startswith(desc) for desc in descriptive_starts):
        return None
    
    # Handle citation text at the start (e.g., "133–34; Vassiliev Yellow Notebook... Vasily Zarubin")
    # If it starts with numbers or citation patterns, try to extract name after the citation
    # Pattern: "18, 21, 72, 83; Vassiliev Yellow Notebook #3, 7, 9 Vasily Zarubin"
    # But also handle: "Vasily Zarubin in mid- and late 1930s. Vassiliev White Notebook #1, 133–34..."
    # In the second case, the name comes BEFORE the citation, so we should extract it before the period
    
    # First, check if it starts with citation text (numbers followed by Vassiliev/Venona)
    if re.match(r'^[\d\s,;–-]+(?:Vassiliev|Venona)', first_sentence):
        # Pattern: "18, 21, 72, 83; Vassiliev Yellow Notebook #3, 7, 9 Vasily Zarubin"
        citation_match = re.match(r'^([\d\s,;–-]+(?:Vassiliev|Venona)[^A-Z]*?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', first_sentence)
        if citation_match:
            name_after_citation = citation_match.group(2).strip()
            if name_after_citation:
                # Check if there's more to the name (e.g., "Vasily Zarubin")
                remaining = first_sentence[citation_match.end():].strip()
                # Try to capture full name if it continues
                full_name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', remaining)
                if full_name_match:
                    name_after_citation = full_name_match.group(1).strip()
                # Recursively process the name part
                return infer_referent_from_body_start(name_after_citation, entry_key=None)
        
        # Fallback: citation-leading strings often contain a real person name near the end.
        # Extract the LAST plausible "First Last" pair, excluding obvious notebook/document words.
        stop_words = {
            'venona', 'vassiliev', 'notebook', 'notebooks', 'yellow', 'white', 'black',
            'special', 'studies', 'decryptions', 'message', 'messages', 'kgb', 'gru'
        }
        pairs = re.findall(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', first_sentence)
        for a, b2 in reversed(pairs):
            if a.lower() in stop_words or b2.lower() in stop_words:
                continue
            return infer_referent_from_body_start(f"{a} {b2}", entry_key=None)
        # If it starts with just numbers/citation and no name follows, return None
        if re.match(r'^\d+[–-]\d+', first_sentence) or re.match(r'^\d+[–-]\d+;\s+Vassiliev', first_sentence):
            return None
    
    # Clean up: remove any trailing parentheticals or explanatory text
    # Stop at opening parenthesis if it appears early (likely explanatory)
    paren_pos = first_sentence.find('(')
    if paren_pos > 0 and paren_pos < len(first_sentence) * 0.5:  # Parenthesis in first half
        first_sentence = first_sentence[:paren_pos].strip()
    
    # Handle "in [Name]" patterns that are part of citations/references
    # Pattern: "Raina in Alexander Vassiliev" -> "Raina"
    # But be careful: "in the United States" should not be removed
    # Only remove if it's "in [Capitalized Name]" pattern and the part before is a single name
    # Also handle "Andrey Raina" -> should keep full name, not just "Raina"
    in_name_match = re.search(r'\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$', first_sentence)
    if in_name_match:
        # Check if the part before "in" looks like a complete name
        before_in = first_sentence[:in_name_match.start()].strip()
        # Only remove "in [Name]" if before_in is a single word (like "Raina")
        # If it's multiple words (like "Andrey Raina"), keep the full name
        if before_in and looks_like_person_head(before_in):
            # Check if it's a single word or multiple words
            words_before = before_in.split()
            if len(words_before) == 1:
                # Single word - remove "in [Name]" pattern
                first_sentence = before_in
            # If multiple words, keep the full name (e.g., "Andrey Raina" stays as is)
    
    if not first_sentence:
        return None
    
    # If the result is a comma-delimited name that looks like a person, invert it
    # "Beigel, Rose" -> "Rose Beigel"
    if looks_like_person_head(first_sentence):
        first_sentence = invert_comma_delimited_name(first_sentence)

    # Handle comma-separated descriptions
    # If there's a comma, check if the part after is descriptive text (not a name)
    # Patterns like: "Unidentified Soviet intelligence source/agent, cover name..." -> stop at comma
    # Also handle: "Beigel, Rose. Also know as Rose Arenal, wife of Luis Arenal."
    # BUT: If the sentence already has a period before the comma handling, we should have stopped there
    # So this comma handling is mainly for cases where there's no period yet
    if "," in first_sentence:
        parts = first_sentence.split(",", 1)
        if len(parts) == 2:
            first_part = parts[0].strip()
            second_part = parts[1].strip()
            
            # Check if second part is a temporal qualifier (starts with circa, prior to, after, etc.)
            temporal_pattern = r'^(circa|prior to|after|before|from|until|since|in|during)\s+\d'
            if re.match(temporal_pattern, second_part, flags=re.IGNORECASE):
                first_sentence = first_part
            # Check if first part is descriptive text (unidentified, described as, etc.)
            # If so, return None - we don't want descriptive text as a referent
            descriptive_starts = (
                "unidentified", "described as", "likely", "probably", "possibly",
                "soviet intelligence source/agent", "soviet intelligence source",
                "intelligence source/agent", "intelligence source", "error for"
            )
            if any(first_part.lower().startswith(desc) for desc in descriptive_starts):
                return None
            # Check if second part is descriptive text (cover name, source/agent, etc.)
            elif any(desc in second_part.lower() for desc in ("cover name", "source/agent", "intelligence", "described as")):
                # Stop at the comma - the name is in the first part
                first_sentence = first_part
            # Check if second part starts with job title (KGB officer, Soviet intelligence officer, etc.)
            # Pattern: "A. Slavyagin, KGB officer" -> use "A. Slavyagin"
            elif re.match(r'^(kgb\s+officer|soviet\s+intelligence\s+officer|intelligence\s+officer)', second_part, flags=re.IGNORECASE):
                # The name is in the first part
                first_sentence = first_part
            # Handle patterns like "Beigel, Rose. Also know as Rose Arenal, wife of..."
            # Extract just the name part before "Also know as" or similar
            elif re.search(r'\b(also\s+know\s+as|also\s+called|wife\s+of|husband\s+of)', second_part, flags=re.IGNORECASE):
                # The name is in the first part, but might need to be inverted
                first_sentence = first_part
                # Special case: "Beigel, Rose. Also know as Rose Arenal" 
                # Extract "Rose Arenal" from "Also know as" part if it's a person name
                also_know_match = re.search(r'\balso\s+know\s+as\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', second_part, flags=re.IGNORECASE)
                if also_know_match:
                    also_name = also_know_match.group(1).strip()
                    # Stop at comma if present (e.g., "Rose Arenal, wife of...")
                    if ',' in also_name:
                        also_name = also_name.split(',')[0].strip()
                    if also_name and looks_like_person_head(also_name):
                        # Use the "Also know as" name instead (e.g., "Rose Arenal")
                        first_sentence = also_name
            # Check if second part starts with colon (e.g., "Cover name in Venona: ROSE")
            elif second_part.startswith(':'):
                # Stop at the comma - the name is in the first part
                first_sentence = first_part
            else:
                # Not clearly descriptive, keep the comma-separated form
                first_sentence = first_sentence

    # Strip prefixes like "pseudonym", "KGB officer", "Soviet intelligence officer" before extracting name
    # Pattern: "pseudonym X" or "KGB officer X" or "Soviet intelligence officer X" -> extract "X"
    # Special case: "pseudonym X Y" -> extract Y as main name (X is the pseudonym, Y is the real name)
    prefix_patterns = [
        (r'^pseudonym\s+', True),  # True means handle "pseudonym X Y" pattern
        (r'^kgb\s+officer\s+', False),
        (r'^soviet\s+intelligence\s+officer\s+', False),
        (r'^soviet\s+intelligence\s+agent\s+', False),
        (r'^intelligence\s+officer\s+', False),
        (r'^intelligence\s+agent\s+', False),
    ]
    for pattern, handle_pseudonym_pattern in prefix_patterns:
        if re.match(pattern, first_sentence, flags=re.IGNORECASE):
            # Extract what comes after the prefix
            remaining = re.sub(pattern, '', first_sentence, flags=re.IGNORECASE).strip()
            if remaining:
                # Handle "pseudonym X Y" pattern - extract Y as main name, X is the pseudonym
                # Pattern: "pseudonym Anatoly Gromov Anatoly Gorsky" -> extract "Anatoly Gorsky"
                if handle_pseudonym_pattern:
                    # Check if there are two capitalized names (likely "First Last First Last")
                    # Split into words and look for pattern where we have two names
                    words = remaining.split()
                    if len(words) >= 4:
                        # Check if we have two capitalized words followed by two more capitalized words
                        # This is a heuristic - "Anatoly Gromov Anatoly Gorsky"
                        # Take the last two words as the real name
                        first_sentence = ' '.join(words[-2:]).strip()
                    else:
                        first_sentence = remaining
                else:
                    first_sentence = remaining
                break
    
    # Trim temporal qualifiers: "Jack Soble prior to ..." -> "Jack Soble"
    # Handle "in mid- and late 1930s" pattern
    # Also handle "circa 1943–1944" pattern
    # Also handle "starting in October 1944" pattern
    first_sentence = re.sub(r'\s+in\s+mid-\s+and\s+late\s+\d{4}s', '', first_sentence, flags=re.IGNORECASE).strip()
    
    first_sentence = re.split(
        r"\b(circa|prior to|after|before|from|until|since|in mid- and late|starting in|beginning in)\b",
        first_sentence,
        flags=re.IGNORECASE
    )[0].strip()

    # Strip trailing parenthetical
    first_sentence = re.sub(r"\s*\([^)]*\)\s*$", "", first_sentence).strip()

    return strip_outer_quotes(first_sentence)


def _looks_like_citationish(p: str) -> bool:
    """
    Heuristic to accept citation chunks while rejecting narrative:
    - must mention Venona or Vassiliev
    - should also contain a comma+page-ish pattern or a notebook/decrypt-ish phrase
    """
    p0 = p.strip()
    if not p0:
        return False

    if ("Venona" not in p0) and ("Vassiliev" not in p0) and ("Vassiliev’s" not in p0) and ("Vassiliev's" not in p0):
        return False

    # strong acceptors
    if re.search(r"\bNotebook\b", p0, flags=re.IGNORECASE):
        return True
    if re.search(r"\bKGB\b|\bGRU\b", p0):
        return True
    if re.search(r"\bSpecial Studies\b", p0):
        return True

    # page-ish: comma then digit or digit range
    if re.search(r",\s*\d", p0):
        return True
    if re.search(r"\b\d+\s*[–-]\s*\d+\b", p0):
        return True

    # fallback: if it starts with Venona/Vassiliev, accept
    if re.match(r"^(Venona|Vassiliev(?:[’']s)?)(\b|,)", p0):
        return True

    return False


def parse_citation_chunks(text: str) -> List[str]:
    """
    Split a citation run into chunks, primarily on ';'.

    Accept chunks that look citation-like, avoiding narrative sentences that merely mention Venona/Vassiliev.
    """
    chunks: List[str] = []
    for part in text.split(";"):
        p = part.strip()
        if not p:
            continue
        p_norm = p.lstrip()
        # Skip narrative text that doesn't look like citations
        # Reject chunks that are clearly narrative (e.g., "as the anglicized version...")
        if p_norm.lower().startswith(('as the ', 'as a ', 'as an ')):
            # Likely narrative text, not a citation
            continue
        if _looks_like_citationish(p_norm):
            chunks.append(p_norm)
    return chunks


def best_effort_parse_citation_fields(
    citation_text: str
) -> Tuple[Optional[str], Optional[str], Optional[List[int]]]:
    ct = citation_text.strip()
    collection_slug = None
    if ct.startswith("Venona"):
        collection_slug = "venona"
    elif ct.startswith("Vassiliev") or ct.startswith("Vassiliev’s") or ct.startswith("Vassiliev's"):
        collection_slug = "vassiliev"

    if "," in ct:
        document_label = ct.split(",", 1)[0].strip()
        tail = ct.split(",", 1)[1].strip()
    else:
        document_label = ct
        tail = ""

    ints: List[int] = []
    if tail:
        # Parse page numbers, handling ranges like "230–31" or "63–74"
        # Split on commas first
        page_parts = tail.split(',')
        
        for part in page_parts:
            part = part.strip()
            if not part:
                continue
            
            # Match page ranges like "74–75" or "112–13" (with en-dash or hyphen)
            range_match = re.match(r'^(\d+)[–-](\d+)$', part)
            single_match = re.match(r'^(\d+)$', part)
            
            if range_match:
                start = int(range_match.group(1))
                end_str = range_match.group(2)
                end = int(end_str)
                
                # Handle abbreviated ranges like "112–13" (means 112-113)
                # or "230–31" (means 230-231)
                if end < start:
                    start_str = str(start)
                    if len(end_str) < len(start_str):
                        # Abbreviated: reconstruct full end number
                        prefix_len = len(start_str) - len(end_str)
                        prefix = start_str[:prefix_len]
                        end = int(prefix + end_str)
                
                # Expand range: add all pages from start to end (inclusive)
                for page_num in range(start, end + 1):
                    ints.append(page_num)
            
            elif single_match:
                page_num = int(single_match.group(1))
                ints.append(page_num)
    
    return collection_slug, document_label, (ints if ints else None)


# -----------------------
# Parsed structures
# -----------------------

@dataclass
class ParsedLink:
    link_type: str
    from_name: str
    to_name: str
    confidence: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ParsedAlias:
    alias: str
    alias_type: str
    confidence: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ParsedCitation:
    citation_text: str
    alias_label: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ParsedEntry:
    entry_key: str
    entry_seq: int
    raw_text: str
    entity_canonical: str
    entity_type: str
    confidence: Optional[str] = None
    notes: Optional[str] = None
    aliases: List[ParsedAlias] = field(default_factory=list)
    links: List[ParsedLink] = field(default_factory=list)
    citations: List[ParsedCitation] = field(default_factory=list)
    definitions: List[ParsedDefinition] = field(default_factory=list)  # Structured definitions, not aliases
    is_crossref_only: bool = False
    crossref_target: Optional[str] = None


# -----------------------
# Entity typing
# -----------------------

def classify_entity_type(entry_key: str, body: str) -> str:
    """
    Conservative typing:
    - 'cover name' if the headword explicitly says so, or body indicates cover name, or known cover-name classes.
    - 'person' if looks like "Lastname, Firstname".
    - otherwise 'topic' or 'other'.
    """
    key = entry_key.strip()
    key_core = remove_trailing_descriptor_paren(key).strip()
    key_lc = _normalize_quotes(key).lower()
    body_lc = (body or "").lower()

    # Explicit descriptor in headword
    if "cover name" in key_lc:
        return "cover_name"

    # Person
    if looks_like_person_head(key_core):
        return "person"

    # Cover-name classes
    if key_core.lower().startswith("undeciphered name no."):
        return "cover_name"
    if "source no." in key_core.lower():
        return "cover_name"

    # Body indicates a cover-name entry
    if "cover name in venona" in body_lc or "cover name in vassiliev" in body_lc:
        return "cover_name"

    # Do NOT auto-classify acronyms as cover_name anymore.
    # (AAF, FBI, USA etc. are often topics/acronyms; cover names are handled via explicit cues above.)

    t = (key_core + " " + (body or "")).lower()
    if any(k in t for k in ["related subjects", "references in", "all of vassiliev", "devoted to"]):
        return "topic"

    return "other"


# -----------------------
# Footnote stripping for parsing (raw_text remains lossless)
# -----------------------

def strip_footnotes_from_block_for_parsing(block: str) -> str:
    """
    Keep raw_text lossless, but ignore appendix/footnote sections when parsing aliases/citations.
    Stops at a long dash separator, separator line, or a numbered footnote start.
    """
    lines = block.split("\n")
    out = []
    for ln in lines:
        s = ln.strip()
        if _RE_LONG_DASH_LINE.match(s) or _RE_SEPARATOR.match(s):
            break
        if _RE_FOOTNOTE_START.match(s) or _RE_FOOTNOTE_LIKE.match(s):
            break
        out.append(ln)
    return "\n".join(out).strip()


# -----------------------
# Entry parsing
# -----------------------

def parse_entry_block(block: str, entry_seq: int) -> ParsedEntry:
    raw_text = block
    parse_block = strip_footnotes_from_block_for_parsing(block)

    lines = parse_block.split("\n")
    if not lines or not lines[0].strip():
        lines = raw_text.split("\n")

    first_line = lines[0].strip()
    # Drop bare small-int continuation lines (line wrap artifacts like " 7")
    rest_lines = [ln for ln in lines[1:] if not _RE_BARE_SMALL_INT_LINE.match(ln.strip())]

    head_raw, body_first, delim = split_head_and_body(first_line)
    head_raw_norm = _normalize_quotes(head_raw).strip()
    body_joined = "\n".join([body_first] + rest_lines).strip()

    body_first_lc = _normalize_quotes(body_first).strip().lower()
    is_crossref = body_first_lc.startswith("see ")
    crossref_target = None

    entry_key = strip_outer_quotes(head_raw_norm).strip()
    entity_type = classify_entity_type(entry_key, body_joined)

    head_no_desc = remove_trailing_descriptor_paren(head_raw_norm).strip()
    
    # Remove question marks from headword (uncertainty markers)
    head_no_desc = remove_question_marks(head_no_desc)

    quoted = extract_quoted_strings(head_no_desc)
    bracket_tokens = extract_bracket_tokens(head_no_desc)
    head_syns = extract_head_synonyms(head_no_desc)

    if quoted:
        canonical = strip_outer_quotes(quoted[0])
    elif head_syns:
        canonical = strip_outer_quotes(head_syns[0])
    else:
        canonical = re.sub(r"\[[^\]]*\]", "", head_no_desc).strip()
        canonical = strip_outer_quotes(canonical)

    canonical = canonical.strip()
    if delim == ".":
        canonical = canonical.rstrip(".").strip()
    
    # Remove question marks from canonical name
    canonical = remove_question_marks(canonical)
    
    # Remove parenthetical qualifiers from canonical name (e.g., "Amtorg (AMTORG)" -> "Amtorg")
    # But only if it's a simple parenthetical at the end, not part of the name structure
    canonical = re.sub(r'\s*\([^)]*\)\s*$', '', canonical).strip()
    
    # Strip ellipses from cover-name headwords (e.g., "BAL..." -> "BAL")
    if entity_type == "cover_name":
        canonical = re.sub(r'\.\.\.+$', '', canonical).strip()
    
    # Invert comma-delimited person names: "Rogov, Alexander" -> "Alexander Rogov"
    if entity_type == "person" and looks_like_person_head(canonical):
        canonical = invert_comma_delimited_name(canonical)
    
    # Normalize simple "or" variants in person headwords:
    # "Leopol or Leopolo Arenal" -> "Leopol Arenal"
    if entity_type == "person" and " or " in canonical.lower():
        m_or = re.match(r'^([A-Z][a-z]+)\s+or\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)$', canonical)
        if m_or:
            canonical = f"{m_or.group(1)} {m_or.group(3)}".strip()

    # Crossref-only entries: "X: See Y."
    if is_crossref:
        m = re.match(r'^see\s+["“]?([^".;]+)["”]?', _normalize_quotes(body_first).strip(), flags=re.IGNORECASE)
        if m:
            crossref_target = m.group(1).strip()
            # Clean the crossref_target (remove trailing period, normalize quotes)
            crossref_target = crossref_target.rstrip(".").strip()
            crossref_target = strip_outer_quotes(crossref_target)
        
        # Clean entry_key for crossref entries (remove brackets, normalize quotes)
        # entry_key should be the canonical form, not including brackets
        cleaned_entry_key = canonical if canonical else entry_key
        cleaned_entry_key = strip_outer_quotes(cleaned_entry_key)
        
        pe = ParsedEntry(
            entry_key=cleaned_entry_key,
            entry_seq=entry_seq,
            raw_text=raw_text,
            entity_canonical=canonical if canonical else cleaned_entry_key,
            entity_type=entity_type,
            is_crossref_only=True,
            crossref_target=crossref_target,
        )
        # Add canonical as alias
        if canonical:
            pe.aliases.append(ParsedAlias(alias=canonical, alias_type="canonical"))
        # Add bracket variants as aliases
        for bt in bracket_tokens:
            bt2 = bt.strip()
            bt2 = remove_question_marks(bt2)
            if bt2 and bt2.lower() != pe.entity_canonical.lower():
                pe.aliases.append(ParsedAlias(alias=bt2, alias_type="bracket_variant"))
        # Add the cleaned entry_key as a "see" alias pointing to the crossref target
        pe.aliases.append(ParsedAlias(alias=cleaned_entry_key, alias_type="see"))
        return pe

    pe = ParsedEntry(
        entry_key=entry_key,
        entry_seq=entry_seq,
        raw_text=raw_text,
        entity_canonical=canonical if canonical else entry_key,
        entity_type=entity_type,
    )

    # Extract definitions BEFORE parsing citations
    # This prevents definitions from being mixed into citations
    body_norm = _normalize_quotes(body_joined)
    pe.definitions, body_after_definitions = extract_definitions_from_body(body_norm, entity_type)
    
    # Use body_after_definitions for citation parsing (not the original body)
    # This ensures citations are clean and don't include definition text

    # Canonical alias
    if pe.entity_canonical:
        pe.aliases.append(ParsedAlias(alias=pe.entity_canonical, alias_type="canonical"))
    
    # For person entities: also add the original comma-delimited form as alias
    if entity_type == "person" and looks_like_person_head(head_no_desc):
        original_form = re.sub(r"\[[^\]]*\]", "", head_no_desc).strip()
        original_form = strip_outer_quotes(original_form)
        original_form = remove_question_marks(original_form)
        if delim == ".":
            original_form = original_form.rstrip(".").strip()
        # Clean up trailing commas
        original_form = original_form.rstrip(",").strip()
        if original_form and original_form.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=original_form, alias_type="original_form"))

    # Head synonyms
    for syn in head_syns:
        syn2 = syn.strip()
        # Remove question marks
        syn2 = remove_question_marks(syn2)
        if delim == ".":
            syn2 = syn2.rstrip(".").strip()
        if syn2 and syn2.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=syn2, alias_type="head_syn"))

    # Bracket variants (you said you trust these)
    for bt in bracket_tokens:
        bt2 = bt.strip()
        # Remove question marks
        bt2 = remove_question_marks(bt2)
        if delim == ".":
            bt2 = bt2.rstrip(".").strip()
        if bt2 and bt2.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=bt2, alias_type="bracket_variant"))

    # Additional quoted variants
    for q in quoted[1:]:
        q2 = strip_outer_quotes(q.strip())
        # Remove question marks
        q2 = remove_question_marks(q2)
        if delim == ".":
            q2 = q2.rstrip(".").strip()
        if q2 and q2.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=q2, alias_type="head_quote_variant"))

    # Definitional expansion for acronym-ish entries: "AAF: Army Air Force, U.S."
    # Keep this as alias expansion; do NOT force entity_type=cover_name.
    if pe.entity_canonical.isupper() and 2 <= len(pe.entity_canonical) <= 10:
        defn = _normalize_quotes(body_first).strip()
        defn = defn.split(".", 1)[0].strip()
        if defn and not defn.lower().startswith(("cover name", "see ", "as ")):
            defn2 = defn.strip().strip(";").strip()
            if len(defn2.split()) >= 2:
                pe.aliases.append(ParsedAlias(
                    alias=defn2,
                    alias_type="definition",
                    confidence="certain",
                    notes="Index definitional expansion"
                ))

    # Body: work name
    for m in re.finditer(r'work name\s+["“]([^"”]+)["”]', _normalize_quotes(body_joined), flags=re.IGNORECASE):
        wn = m.group(1).strip()
        if wn:
            pe.aliases.append(ParsedAlias(alias=wn, alias_type="work_name"))

    # Body: Name spelled X
    m = re.search(r"Name spelled\s+([A-Za-z][A-Za-z'\-]+)\b", body_joined)
    if m:
        sp = m.group(1).strip()
        if sp:
            pe.aliases.append(ParsedAlias(alias=sp, alias_type="spelling_variant", notes="Name spelled in texts"))
    
    # Body: "Also referred to as X, Y, and Z" or "referred to as X, Y, and Z"
    # Extract individual aliases from comma/and-separated lists
    referred_to_pattern = r'(?:also\s+)?referred\s+to\s+as\s+([^.;]+)'
    for m in re.finditer(referred_to_pattern, body_joined, re.IGNORECASE):
        aliases_text = m.group(1).strip()
        # Split on commas and "and"
        # Handle patterns like "StateD, SD, and DOS" or "X, Y, and Z"
        parts = re.split(r',\s*(?:and\s+)?', aliases_text)
        for part in parts:
            part = part.strip()
            # Remove trailing "and" if present
            part = re.sub(r'\s+and\s*$', '', part, flags=re.IGNORECASE).strip()
            if part:
                # Clean up the alias
                part = strip_outer_quotes(part)
                part = remove_question_marks(part)
                if part and part.lower() != pe.entity_canonical.lower():
                    # Check if this alias is already in the list
                    if not any(a.alias.lower() == part.lower() for a in pe.aliases):
                        pe.aliases.append(ParsedAlias(
                            alias=part,
                            alias_type="definition",
                            confidence="certain",
                            notes="Also referred to as"
                        ))

    # Body: cover name(s) in Venona/Vassiliev mentioned inside person/topic entry
    # Handle both "Cover name" (singular) and "Cover names" (plural)
    # Also handle multiple names like "ARNOLD [ARNOL'D] and FAKIR" or comma-separated lists
    for m in re.finditer(
        r"Cover\s+names?\s+in\s+(Venona|Vassiliev[’']s notebooks)\s*:\s*([^.;]+?)(?:\.|;|$)",
        _normalize_quotes(body_joined),
        flags=re.IGNORECASE
    ):
        names_text = m.group(2).strip()
        source_type = m.group(1)
        
        # Normalize whitespace (replace newlines with spaces) before splitting
        # This ensures comma splitting works correctly even if text has line breaks
        names_text = re.sub(r'\s+', ' ', names_text).strip()
        
        # Extract individual names from the text
        # Handle patterns like:
        # - "ARNOLD [ARNOL'D] and FAKIR" (split on "and")
        # - "Jung" ["Yung"] (1930s), "Mer" (1942–1944), "Albert" (starting August 1944), and "Gold" (comma-separated)
        # - "MAYOR [MER and MĒR]" (comma inside brackets should NOT split)
        # Split on commas, but NOT inside brackets or quotes
        comma_parts = []
        current = []
        bracket_depth = 0
        quote_char = None
        
        i = 0
        while i < len(names_text):
            char = names_text[i]
            
            # Track quotes (handle both straight and curly quotes)
            if char in '"""' and (i == 0 or names_text[i-1] != '\\'):
                if quote_char is None:
                    quote_char = char
                elif char == quote_char:
                    quote_char = None
            # Track brackets (only when not in quotes)
            elif quote_char is None:
                if char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                elif char == ',' and bracket_depth == 0:
                    # This comma is outside brackets/quotes, safe to split
                    part = ''.join(current).strip()
                    if part:
                        comma_parts.append(part)
                    current = []
                    i += 1
                    continue
            
            current.append(char)
            i += 1
        
        # Add the last part
        if current:
            part = ''.join(current).strip()
            if part:
                comma_parts.append(part)
        
        # If no commas, try splitting on "and" instead, but NOT inside brackets
        if len(comma_parts) == 1:
            # Split on "and" but not inside brackets
            name_parts = []
            current = []
            bracket_depth = 0
            quote_char = None
            
            i = 0
            text = names_text
            while i < len(text):
                char = text[i]
                
                # Track quotes
                if char in '"""' and (i == 0 or text[i-1] != '\\'):
                    if quote_char is None:
                        quote_char = char
                    elif char == quote_char:
                        quote_char = None
                # Track brackets (only when not in quotes)
                elif quote_char is None:
                    if char == '[':
                        bracket_depth += 1
                    elif char == ']':
                        bracket_depth -= 1
                    elif bracket_depth == 0:
                        # Check if this is "and" (case insensitive, with word boundaries)
                        if i < len(text) - 3:
                            remaining = text[i:]
                            match = re.match(r'\s+and\s+', remaining, flags=re.IGNORECASE)
                            if match:
                                part = ''.join(current).strip()
                                if part:
                                    name_parts.append(part)
                                current = []
                                # Skip the "and"
                                i += len(match.group(0))
                                continue
                
                current.append(char)
                i += 1
            
            if current:
                part = ''.join(current).strip()
                if part:
                    name_parts.append(part)
        else:
            name_parts = []
            for part in comma_parts:
                part = part.strip()
                # Remove "and" from beginning or end if present (handle with or without surrounding whitespace)
                part = re.sub(r'^(and\s+)|(\s+and\s*)$', '', part, flags=re.IGNORECASE).strip()
                if part:
                    name_parts.append(part)
        
        for name_part in name_parts:
            name_part = name_part.strip()
            if not name_part:
                continue
            
            # Strip ellipsis from cover names (e.g., "BAL..." -> "BAL")
            name_part = re.sub(r'\.\.\.+$', '', name_part).strip()
            
            # Strip "then" qualifier if present (e.g., "then METER and METRE")
            if re.match(r'^then\s+', name_part, flags=re.IGNORECASE):
                name_part = re.sub(r'^then\s+', '', name_part, flags=re.IGNORECASE).strip()
                # Remove quotes if present
                name_part = strip_outer_quotes(name_part)
            
            # Strip parenthetical qualifiers first
            # Handle patterns like: "(prior to 1941)", "(1941–1945)", "(1930s)", "(AMTORG)"
            name_part = re.sub(r'\s*\([^)]*\)\s*$', '', name_part).strip()
            # Strip temporal qualifiers like "in 1954", "in May 1944", "starting August 1944"
            # These can appear at the end of the name part
            name_part = re.sub(r'\s+(?:in|starting|from|until|during)\s+\d{4}(?:\s+\d{1,2})?\s*$', '', name_part, flags=re.IGNORECASE).strip()
            name_part = re.sub(r'\s+(?:in|starting|from|until|during)\s+[A-Z][a-z]+\s+\d{4}\s*$', '', name_part, flags=re.IGNORECASE).strip()
            # Also handle "in 1954" pattern (simpler version)
            name_part = re.sub(r'\s+in\s+\d{4}\s*$', '', name_part, flags=re.IGNORECASE).strip()
            
            # Handle "and" patterns in the name part itself (e.g., "then METER and METRE")
            # Split on "and" if present (but not inside brackets)
            # Process each part as a separate cover name
            # Handle "METER and METRE [METR]" - split on "and" before processing brackets
            if ' and ' in name_part:
                # Check if "and" is outside brackets
                and_pos = name_part.find(' and ')
                bracket_start = name_part.find('[')
                # If brackets exist, check if "and" comes before them
                if bracket_start == -1 or and_pos < bracket_start:
                    # "and" is outside brackets or no brackets - safe to split
                    and_parts = re.split(r'\s+and\s+', name_part, flags=re.IGNORECASE)
                    for and_part in and_parts:
                        and_part = and_part.strip()
                        if and_part:
                            # Process this part as if it were a separate name_part
                            # Normalize and process it
                            and_part_normalized = re.sub(r'\s+', ' ', and_part).strip()
                            and_part_normalized = strip_outer_quotes(and_part_normalized)
                            if and_part_normalized:
                                if not any(a.alias.lower() == and_part_normalized.lower() for a in pe.aliases):
                                    pe.aliases.append(ParsedAlias(
                                        alias=and_part_normalized,
                                        alias_type="covername_from_body",
                                        confidence="certain",
                                        notes=f"Cover name in {source_type}"
                                    ))
                                pe.links.append(ParsedLink(
                                    link_type="cover_name_of",
                                    from_name=and_part_normalized,
                                    to_name=pe.entity_canonical,
                                    confidence="certain",
                                    notes=f"Cover name in {source_type}"
                                ))
                    continue
            
            # Extract the main name (before brackets if present)
            # Pattern: "ARNOLD [ARNOL'D]" -> extract "ARNOLD" and "ARNOL'D"
            # Pattern: "MAYOR [MER and MĒR]" -> extract "MAYOR" and handle "MER and MĒR"
            # Normalize whitespace (replace newlines with spaces) to handle line breaks
            name_part_normalized = re.sub(r'\s+', ' ', name_part).strip()
            
            # First, try to extract bracket content using a more robust method
            bracket_start = name_part_normalized.find('[')
            bracket_end = name_part_normalized.find(']', bracket_start) if bracket_start >= 0 else -1
            
            # Handle incomplete brackets (e.g., "DEPARTMENT [OTDEL" without closing bracket)
            if bracket_start >= 0 and bracket_end == -1:
                # Incomplete bracket - just extract the part before the bracket
                main_name = name_part_normalized[:bracket_start].strip()
                main_name = strip_outer_quotes(main_name)
                main_name = re.sub(r'\s*\([^)]*\)\s*$', '', main_name).strip()
                if main_name:
                    if not any(a.alias.lower() == main_name.lower() for a in pe.aliases):
                        pe.aliases.append(ParsedAlias(
                            alias=main_name,
                            alias_type="covername_from_body",
                            confidence="certain",
                            notes=f"Cover name in {source_type}"
                        ))
                    pe.links.append(ParsedLink(
                        link_type="cover_name_of",
                        from_name=main_name,
                        to_name=pe.entity_canonical,
                        confidence="certain",
                        notes=f"Cover name in {source_type}"
                    ))
            elif bracket_start >= 0 and bracket_end > bracket_start:
                # We have brackets - extract main name and bracket content
                main_name = name_part_normalized[:bracket_start].strip()
                bracket_content = name_part_normalized[bracket_start+1:bracket_end].strip()
                
                # Clean main name (remove quotes, parentheticals)
                main_name = strip_outer_quotes(main_name)
                main_name = re.sub(r'\s*\([^)]*\)\s*$', '', main_name).strip()
                # Fix incomplete quotes (e.g., """Glan" -> "Glan")
                main_name = re.sub(r'^["""""]+([^"""""]+)["""""]*$', r'\1', main_name)
                main_name = re.sub(r'^["""]+([^"""]+)$', r'\1', main_name)
                
                # Additional safety: if main_name still contains brackets or parentheticals, extract just the quoted part
                if '[' in main_name or '(' in main_name:
                    quoted_match = re.search(r'["""]([^"""]+)["""]', main_name)
                    if quoted_match:
                        main_name = quoted_match.group(1).strip()
                
                if main_name:
                    # Add as alias
                    if not any(a.alias.lower() == main_name.lower() for a in pe.aliases):
                        pe.aliases.append(ParsedAlias(
                            alias=main_name,
                            alias_type="covername_from_body",
                            confidence="certain",
                            notes=f"Cover name in {source_type}"
                        ))
                    # Add link
                    pe.links.append(ParsedLink(
                        link_type="cover_name_of",
                        from_name=main_name,
                        to_name=pe.entity_canonical,
                        confidence="certain",
                        notes=f"Cover name in {source_type}"
                    ))
                
                # Handle bracket content - it might contain "and" (e.g., "MER and MĒR")
                if bracket_content:
                    bracket_content = strip_outer_quotes(bracket_content)
                    # Split on "and" if present, but be careful not to split quoted parts
                    if ' and ' in bracket_content.lower():
                        bracket_parts = re.split(r'\s+and\s+', bracket_content, flags=re.IGNORECASE)
                        for bp in bracket_parts:
                            bp = strip_outer_quotes(bp.strip())
                            if bp and not any(a.alias.lower() == bp.lower() for a in pe.aliases):
                                pe.aliases.append(ParsedAlias(
                                    alias=bp,
                                    alias_type="covername_from_body",
                                    confidence="certain",
                                    notes=f"Cover name variant in {source_type}"
                                ))
                    else:
                        # Single bracket variant
                        if bracket_content and not any(a.alias.lower() == bracket_content.lower() for a in pe.aliases):
                            pe.aliases.append(ParsedAlias(
                                alias=bracket_content,
                                alias_type="covername_from_body",
                                confidence="certain",
                                notes=f"Cover name variant in {source_type}"
                            ))
            else:
                # No brackets, just a simple name
                # Use regex match as fallback
                bracket_match = re.match(r'^(?:["""]([^"""]+)["""]|([^\[\]]+?))\s*$', name_part_normalized)
                if bracket_match:
                    name = (bracket_match.group(1) or bracket_match.group(2) or "").strip()
                    name = strip_outer_quotes(name)
                    # Fix incomplete quotes (e.g., """Sonya" -> "Sonya", """Glan" -> "Glan")
                    name = re.sub(r'^["""""]+([^"""""]+)["""""]*$', r'\1', name)
                    name = re.sub(r'^["""]+([^"""]+)$', r'\1', name)
                    # Additional safety: if name still contains brackets or parentheticals, extract just the quoted part
                    if '[' in name or '(' in name:
                        quoted_match = re.search(r'["""]([^"""]+)["""]', name)
                        if quoted_match:
                            name = quoted_match.group(1).strip()
                    
                    if name:
                        # Add as alias
                        if not any(a.alias.lower() == name.lower() for a in pe.aliases):
                            pe.aliases.append(ParsedAlias(
                                alias=name,
                                alias_type="covername_from_body",
                                confidence="certain",
                                notes=f"Cover name in {source_type}"
                            ))
                        # Add link
                        pe.links.append(ParsedLink(
                            link_type="cover_name_of",
                            from_name=name,
                            to_name=pe.entity_canonical,
                            confidence="certain",
                            notes=f"Cover name in {source_type}"
                        ))

    # Defensive extraction: ensure bracket-only variants in "Cover name(s) in Venona: ..."
    # are always emitted as separate aliases (e.g., "ROSE [ROZA]" -> include "ROZA"),
    # even if earlier parsing paths missed them due to malformed punctuation.
    for m in re.finditer(
        r"Cover\s+names?\s+in\s+Venona\s*:\s*([^.;]+?)(?:\.|;|$)",
        _normalize_quotes(body_joined),
        flags=re.IGNORECASE
    ):
        names_text = m.group(1) or ""
        for bt in re.findall(r'\[([^\]]+)\]', names_text):
            bt = strip_outer_quotes(bt.strip())
            if not bt:
                continue
            # Split bracket content on "and" and commas (conservative)
            parts = re.split(r'\s+and\s+|,\s*', bt, flags=re.IGNORECASE)
            for p in parts:
                p = strip_outer_quotes(p.strip())
                if p and len(p) >= 2 and not any(a.alias.lower() == p.lower() for a in pe.aliases):
                    pe.aliases.append(ParsedAlias(
                        alias=p,
                        alias_type="covername_from_body",
                        confidence="certain",
                        notes="Bracket variant from Cover name(s) in Venona"
                    ))

    # Repair pass: ensure any cover-name link sources appear as aliases, and recover bracket variants
    # tied to that cover name (e.g. "ROSE [ROZA]" should yield alias "ROZA" even if upstream parsing missed it).
    alias_lc = {a.alias.lower() for a in pe.aliases if a.alias}
    # Use raw_text here (lossless) because some malformed inputs repeat header fragments,
    # and we still want to recover bracket variants like "ROSE [ROZA]" from later in the block.
    body_norm2 = _normalize_quotes(raw_text)
    for lk in pe.links:
        if lk.link_type != "cover_name_of":
            continue
        cover = strip_outer_quotes((lk.from_name or "").strip())
        if not cover:
            continue
        if cover.lower() not in alias_lc:
            pe.aliases.append(ParsedAlias(
                alias=cover,
                alias_type="covername_from_body",
                confidence="certain",
                notes="Recovered from cover_name_of link"
            ))
            alias_lc.add(cover.lower())
        
        # Recover bracket variants adjacent to this cover name in the body.
        pat = re.compile(rf'\b{re.escape(cover)}\s*\[([^\]]+)\]', flags=re.IGNORECASE)
        for m2 in pat.finditer(body_norm2):
            inner = strip_outer_quotes((m2.group(1) or "").strip())
            if not inner:
                continue
            parts = re.split(r'\s+and\s+|,\s*', inner, flags=re.IGNORECASE)
            for p in parts:
                p = strip_outer_quotes(p.strip())
                if p and len(p) >= 2 and p.lower() not in alias_lc:
                    pe.aliases.append(ParsedAlias(
                        alias=p,
                        alias_type="covername_from_body",
                        confidence="certain",
                        notes="Recovered bracket variant for cover name"
                    ))
                    alias_lc.add(p.lower())

    # Extra safety: recover bracket variants from explicit "Cover name(s) in Venona" segments,
    # allowing for malformed punctuation (missing ":" / odd spacing).
    for m3 in re.finditer(
        r'Cover\s+names?\s+in\s+Venona\s*:?\s*([^.\n]+)',
        body_norm2,
        flags=re.IGNORECASE
    ):
        seg = m3.group(1) or ""
        for bt in re.findall(r'\[([^\]]+)\]', seg):
            bt = strip_outer_quotes(bt.strip())
            if not bt:
                continue
            parts = re.split(r'\s+and\s+|,\s*', bt, flags=re.IGNORECASE)
            for p in parts:
                p = strip_outer_quotes(p.strip())
                if p and len(p) >= 2 and p.lower() not in alias_lc:
                    pe.aliases.append(ParsedAlias(
                        alias=p,
                        alias_type="covername_from_body",
                        confidence="certain",
                        notes="Recovered from Cover name(s) in Venona segment"
                    ))
                    alias_lc.add(p.lower())

    # Body: changed to
    mchg = re.search(r"\b([A-Z0-9][A-Z0-9'\-]+)\s+was changed to\s+([A-Z0-9][A-Z0-9'\-]+)\b", body_joined)
    if mchg:
        old, new = mchg.group(1).strip(), mchg.group(2).strip()
        pe.links.append(ParsedLink(
            link_type="changed_to",
            from_name=old,
            to_name=new,
            confidence="certain",
            notes="Cover name changed"
        ))

    # If entry is a cover_name, infer referent from body start (conservative)
    if pe.entity_type == "cover_name":
        referent = infer_referent_from_body_start(body_first + " " + "\n".join(rest_lines), entry_key=pe.entry_key)
        if referent:
            # Remove question marks from referent
            referent = remove_question_marks(referent)
            
            # Handle "or" patterns (e.g., "Leopol or Leopolo Arenal")
            # Split into separate entities if "or" is present
            if " or " in referent.lower():
                # Split on "or" but preserve the surname if it comes after
                # Pattern: "Leopol or Leopolo Arenal" -> ["Leopol Arenal", "Leopolo Arenal"]
                or_parts = re.split(r'\s+or\s+', referent, flags=re.IGNORECASE)
                if len(or_parts) == 2:
                    first_part = or_parts[0].strip()
                    second_part = or_parts[1].strip()
                    # Check if second part is just a first name (no surname)
                    # If so, the surname might be in the first part
                    # For now, create separate entities for each
                    # This is a complex case that might need manual review
                    # For now, use the first part as the referent
                    referent = first_part
            
            # Only invert cover-name entries into persons when the HEADWORD is explicitly a cover-name entry
            # (e.g., includes "(cover name in ...)" or is an all-caps covername surface).
            # This preserves cases like '"Glan"' where reviewers/tests expect the entity to remain cover_name.
            should_invert_to_person = (
                ("cover name" in (pe.entry_key or "").lower()) or
                (pe.entity_canonical.isupper() and len(pe.entity_canonical) >= 2)
            )
            
            # Avoid creating self-links from broken referent extraction
            if referent.strip().lower() == pe.entity_canonical.strip().lower():
                referent = None
            
            # Check if referent looks like a person name and inversion is allowed
            if referent and should_invert_to_person and looks_like_person_head(referent):
                # Invert if comma-delimited
                referent = invert_comma_delimited_name(referent)
                
                # INVERT: The referent (person) should be the entity, not the cover name
                # Make the cover name(s) aliases of the person entity
                old_canonical = pe.entity_canonical
                old_aliases = pe.aliases.copy()
                
                # Handle "and" in cover names (e.g., "METER and METRE", "BELKA [SQUIRREL]")
                # Check if old_canonical contains "and" or brackets with multiple names
                cover_names_to_add = []
                if '[' in old_canonical and ']' in old_canonical:
                    # Extract bracket content
                    bracket_match = re.search(r'\[([^\]]+)\]', old_canonical)
                    if bracket_match:
                        bracket_content = bracket_match.group(1)
                        main_name = re.sub(r'\s*\[[^\]]+\]\s*', '', old_canonical).strip()
                        cover_names_to_add.append(main_name)
                        # Check if bracket content has "and"
                        if ' and ' in bracket_content.lower():
                            bracket_parts = re.split(r'\s+and\s+', bracket_content, flags=re.IGNORECASE)
                            for bp in bracket_parts:
                                bp = bp.strip()
                                if bp:
                                    cover_names_to_add.append(bp)
                        else:
                            cover_names_to_add.append(bracket_content.strip())
                    else:
                        cover_names_to_add.append(old_canonical)
                elif ' and ' in old_canonical.lower():
                    # Split on "and"
                    and_parts = re.split(r'\s+and\s+', old_canonical, flags=re.IGNORECASE)
                    for ap in and_parts:
                        ap = ap.strip()
                        if ap:
                            cover_names_to_add.append(ap)
                else:
                    cover_names_to_add.append(old_canonical)
                
                # Change entity to the person (referent)
                pe.entity_canonical = referent
                pe.entity_type = "person"
                
                # Clear existing aliases and rebuild
                pe.aliases = []
                pe.aliases.append(ParsedAlias(alias=referent, alias_type="canonical"))
                
                # Add the cover name(s) as aliases
                for cover_name in cover_names_to_add:
                    cover_name = strip_outer_quotes(cover_name.strip())
                    if cover_name and not any(a.alias.lower() == cover_name.lower() for a in pe.aliases):
                        pe.aliases.append(ParsedAlias(alias=cover_name, alias_type="cover_name", confidence="certain", notes="Cover name from entry headword"))
                
                # Add bracket variants as cover name aliases
                for old_alias in old_aliases:
                    if old_alias.alias_type == "bracket_variant":
                        pe.aliases.append(ParsedAlias(
                            alias=old_alias.alias,
                            alias_type="cover_name",
                            confidence=old_alias.confidence,
                            notes="Cover name variant from entry headword"
                        ))
                
                # Add link: cover name -> person (but now person is the entity)
                # Add links for all cover names
                for cover_name in cover_names_to_add:
                    cover_name = strip_outer_quotes(cover_name.strip())
                    if cover_name:
                        pe.links.append(ParsedLink(
                            link_type="cover_name_of",
                            from_name=cover_name,
                            to_name=referent,
                            confidence="certain",
                            notes="Cover name referent from entry header"
                        ))
            else:
                # Referent doesn't look like a person, keep original structure
                if referent:
                    pe.links.append(ParsedLink(
                        link_type="cover_name_of",
                        from_name=pe.entity_canonical,
                        to_name=referent,
                        confidence="certain",
                        notes="Cover name referent from entry header"
                    ))

    # Scoped citation sections: As "X": ... / Translated as X: ...
    scoped_spans: List[Tuple[int, int]] = []
    scoped_pat = re.compile(r'\b(As|Translated as)\s+["“]?([^"”:]+)["”]?\s*:', flags=re.IGNORECASE)

    matches = list(scoped_pat.finditer(body_after_definitions))
    for i, m in enumerate(matches):
        # Check that "As" is capitalized OR lowercase "as" followed by uppercase word
        # The pattern matches "As" or "Translated as", but we need to filter out lowercase "as"
        # unless it's followed by an uppercase word (e.g., "as UCN/25:")
        match_start = body_after_definitions[m.start():m.start()+10]
        if match_start.lower().startswith('as ') and not match_start.startswith('As '):
            # Check if lowercase "as" is followed by uppercase letter (e.g., "as UCN/25:")
            if len(match_start) > 3 and not match_start[3].isupper():
                # Skip lowercase "as" matches that aren't followed by uppercase (common word, not a citation marker)
                continue
        label = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body_after_definitions)
        scoped_text = body_after_definitions[start:end].strip()
        
        # Don't add to spans yet - we'll only add if it passes validation
        # scoped_spans.append((m.start(), end))

        if label:
            # Validate that this is actually a scoped citation, not narrative text
            # Two checks:
            # 1. The label should look like a name/alias (not a descriptive phrase like "the anglicized")
            # 2. The text after "As X:" should be citation-like (Venona/Vassiliev + document + page numbers)
            
            # Check 1: Label should be a name-like string, not a descriptive phrase
            # Reject common descriptive phrases that appear in narrative
            # Check BEFORE truncation to catch phrases like "the anglicized version of their"
            label_lower_original = label.lower().strip()
            descriptive_phrases = {
                'the anglicized', 'the anglicized version', 'the version', 'the form',
                'the name', 'the surname', 'the cover name', 'the real name'
            }
            # Check if it starts with descriptive article or is in our list
            if (label_lower_original in descriptive_phrases or 
                label_lower_original.startswith(('the ', 'a ', 'an '))):
                # Likely narrative text, not a citation marker - skip this match
                continue
            
            # Check 2: Scoped citations should be followed by citation-like text
            looks_like_citation = False
            if scoped_text:
                # Check if scoped_text contains citation patterns (Venona/Vassiliev + document + page numbers)
                # This is more flexible than requiring it at the start
                # Pattern: Venona/Vassiliev + capital letter + comma + digit (page number)
                citation_check = re.search('\\b(Venona|Vassiliev(?:[’'']s)?)\\s+[A-Z][^,]*,\\s*\\d', scoped_text, re.IGNORECASE)
                if citation_check:
                    looks_like_citation = True
                # Also check for simpler patterns (just Venona/Vassiliev + comma + digit)
                elif re.search('\\b(Venona|Vassiliev(?:[’'']s)?)[^,]*,\\s*\\d', scoped_text, re.IGNORECASE):
                    looks_like_citation = True
                # Pattern 3: Very lenient - just check if it mentions Venona/Vassiliev and has page numbers
                elif (re.search('\\b(Venona|Vassiliev(?:[’'']s)?)', scoped_text, re.IGNORECASE) and 
                      re.search(',\\s*\\d', scoped_text)):
                    looks_like_citation = True
            
            # Only process if it looks like a real scoped citation
            if not looks_like_citation:
                continue  # Skip this match - it's narrative text, not a citation marker
            
            # Add to spans only after validation passes
            scoped_spans.append((m.start(), end))
            
            # Clean up label - it should be just the name, not a whole sentence
            # If label contains periods or commas (indicating it captured too much), truncate at first punctuation
            if '.' in label:
                # Likely captured too much - take first part before period
                label = label.split('.')[0].strip()
            # Remove trailing commas
            label = label.rstrip(',').strip()
            # Remove question marks
            label = remove_question_marks(label)
            
            # Limit to reasonable length for names (but allow codes like "UCN/25")
            # Scoped labels are typically short names (1-3 words) or codes
            words = label.split()
            if len(words) > 3:
                # Take first 2 words as the label (most names are 1-2 words)
                label = ' '.join(words[:2]).strip()
            
            # Handle "and" patterns in labels (e.g., "METER and METRE")
            # Split on "and" if present
            if ' and ' in label.lower():
                label_parts = re.split(r'\s+and\s+', label, flags=re.IGNORECASE)
                for lp in label_parts:
                    lp = lp.strip()
                    if not lp:
                        continue
                    # Handle brackets in each part
                    bracket_match = re.search(r'^(.+?)\s*\[([^\]]+)\]$', lp)
                    if bracket_match:
                        main_name = bracket_match.group(1).strip()
                        bracket_name = bracket_match.group(2).strip()
                        # Add main name as scoped label
                        if main_name:
                            pe.aliases.append(ParsedAlias(alias=main_name, alias_type="scoped_label"))
                        # Add bracket content as separate alias (covername variant)
                        if bracket_name and len(bracket_name) >= 2:
                            pe.aliases.append(ParsedAlias(
                                alias=bracket_name,
                                alias_type="covername_from_body",
                                confidence="certain",
                                notes="Bracket variant from scoped citation"
                            ))
                    else:
                        # Remove any trailing brackets that might be left
                        lp = re.sub(r'\s*\[[^\]]*\]\s*$', '', lp).strip()
                        if lp:
                            pe.aliases.append(ParsedAlias(alias=lp, alias_type="scoped_label"))
            # If label contains brackets like "LIGHT [SVET]", extract both parts separately
            elif '[' in label and ']' in label:
                bracket_match = re.search(r'^(.+?)\s*\[([^\]]+)\]$', label)
                if bracket_match:
                    main_name = bracket_match.group(1).strip()
                    bracket_name = bracket_match.group(2).strip()
                    # Add main name as scoped label
                    if main_name:
                        pe.aliases.append(ParsedAlias(alias=main_name, alias_type="scoped_label"))
                    # Add bracket content as separate alias (covername variant)
                    if bracket_name and len(bracket_name) >= 2:
                        pe.aliases.append(ParsedAlias(
                            alias=bracket_name,
                            alias_type="covername_from_body",
                            confidence="certain",
                            notes="Bracket variant from scoped citation"
                        ))
                else:
                    pe.aliases.append(ParsedAlias(alias=label, alias_type="scoped_label"))
            else:
                pe.aliases.append(ParsedAlias(alias=label, alias_type="scoped_label"))

            # Only add citations if we have a valid label (validation passed)
            for chunk in parse_citation_chunks(scoped_text):
                pe.citations.append(ParsedCitation(
                    citation_text=chunk,
                    alias_label=label,
                    notes=f"{m.group(1).title()} {label}"
                ))

    # Unscoped citations: remove scoped regions and parse remainder
    # Use body_after_definitions (not body_norm) to avoid including definitions in citations
    if matches:
        rem = body_after_definitions
        for (a, b) in reversed(scoped_spans):
            rem = rem[:a] + " " + rem[b:]
        for chunk in parse_citation_chunks(rem):
            pe.citations.append(ParsedCitation(citation_text=chunk, alias_label=None, notes=None))
    else:
        for chunk in parse_citation_chunks(body_after_definitions):
            pe.citations.append(ParsedCitation(citation_text=chunk, alias_label=None, notes=None))

    # Final safety: recover ALL-CAPS bracket variants anywhere in the raw block.
    # This is a conservative salvage step for cases where header repetition or malformed punctuation
    # causes the structured "Cover name in Venona: X [Y]" parser to miss Y (e.g., missing ROZA).
    # We only accept simple all-caps/number tokens to avoid pulling in narrative bracketed phrases.
    raw_norm = _normalize_quotes(raw_text)
    for inner in re.findall(r'\[([A-Z0-9][A-Z0-9\'\-]{1,})\]', raw_norm):
        inner = strip_outer_quotes(inner.strip())
        if not inner:
            continue
        key = inner.lower()
        if not any(a.alias.strip().lower() == key for a in pe.aliases):
            pe.aliases.append(ParsedAlias(
                alias=inner,
                alias_type="covername_from_body",
                confidence="certain",
                notes="Recovered all-caps bracket variant (salvage)"
            ))

    # Deduplicate aliases
    seen = set()
    deduped: List[ParsedAlias] = []
    for al in pe.aliases:
        key = al.alias.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(al)
    pe.aliases = deduped

    return pe


# -----------------------
# DB helpers (idempotent-ish)
# -----------------------

def get_conn(db_url: Optional[str] = None):
    dsn = db_url or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL (or pass --db-url)")
    return psycopg2.connect(dsn)


def ensure_source(cur, slug: str, title: str, notes: Optional[str]) -> int:
    cur.execute(
        """
        INSERT INTO concordance_sources(slug, title, notes)
        VALUES (%s, %s, %s)
        ON CONFLICT (slug) DO UPDATE
          SET title = EXCLUDED.title,
              notes = COALESCE(EXCLUDED.notes, concordance_sources.notes)
        RETURNING id;
        """,
        (slug, title, notes),
    )
    return int(cur.fetchone()[0])


def upsert_entry(cur, source_id: int, entry_key: str, entry_seq: int, raw_text: str) -> int:
    cur.execute(
        """
        INSERT INTO concordance_entries(source_id, entry_key, entry_seq, raw_text)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_id, entry_key, entry_seq) DO UPDATE
          SET raw_text = EXCLUDED.raw_text
        RETURNING id;
        """,
        (source_id, entry_key, entry_seq, raw_text),
    )
    return int(cur.fetchone()[0])


def find_entity_any_type(cur, source_id: int, canonical_name: str) -> Optional[Tuple[int, str]]:
    cur.execute(
        """
        SELECT id, entity_type
        FROM entities
        WHERE source_id = %s AND canonical_name = %s
        ORDER BY id
        LIMIT 1;
        """,
        (source_id, canonical_name),
    )
    row = cur.fetchone()
    if not row:
        return None
    return int(row[0]), str(row[1])


def ensure_entity(
    cur,
    source_id: int,
    entry_id: Optional[int],
    canonical_name: str,
    entity_type: str,
    confidence: Optional[str],
    notes: Optional[str],
) -> int:
    existing = find_entity_any_type(cur, source_id, canonical_name)
    if existing:
        eid, etype = existing
        upgrade = (etype == "other" and entity_type != "other")
        cur.execute("SELECT entry_id FROM entities WHERE id=%s;", (eid,))
        cur_entry_id = cur.fetchone()[0]
        set_entry = (cur_entry_id is None and entry_id is not None)

        if upgrade or set_entry or (notes is not None and notes.strip()):
            cur.execute(
                """
                UPDATE entities
                SET entity_type = CASE WHEN %s THEN %s ELSE entity_type END,
                    entry_id = CASE WHEN %s THEN %s ELSE entry_id END,
                    confidence = COALESCE(%s, confidence),
                    notes = COALESCE(%s, notes)
                WHERE id = %s;
                """,
                (upgrade, entity_type, set_entry, entry_id, confidence, notes, eid),
            )
        return eid

    cur.execute(
        """
        INSERT INTO entities(source_id, entry_id, canonical_name, entity_type, confidence, notes)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (source_id, entry_id, canonical_name, entity_type, confidence, notes),
    )
    return int(cur.fetchone()[0])


def normalize_alias_for_db(alias: str) -> str:
    """
    Normalize alias for alias_norm column: lowercase, strip punctuation, collapse whitespace.
    Matches the normalization used in migrations/0019_entities.sql.
    """
    # Lowercase
    normalized = alias.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    # Trim
    normalized = normalized.strip()
    return normalized


def _first_alpha_is_upper(s: str) -> bool:
    for ch in s:
        if ch.isalpha():
            return ch.isupper()
    return False


# Common English words that should NOT be aliases even if capitalized
# These are generic/common words that appear in definitions but aren't proper nouns
GENERIC_WORDS_TO_EXCLUDE = {
    # Pronouns/determiners (even when capitalized)
    "i", "which", "their", "them", "what", "when", "there", "whom", "this", "that", "these", "those",
    # Common nouns (generic, not proper nouns)
    "work", "city", "group", "pages", "serial", "known", "sent", "ref", "pm", "case", "time", "terms",
    "refer", "secret", "affairs", "minister", "given", "funds", "reply", "link", "telegraph", "working",
    "note", "general", "real", "financial", "doctor", "agent", "cases", "currency", "cutting", "ministry",
    "reference", "distant", "president", "chief", "cipher", "note", "terms",
    # Single letters (lowercase/titlecase only - ALLCAPS single letters like "A", "M" are kept as potential codenames)
    "m", "a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
}


def filter_alias_to_capitalized_tokens(alias: str) -> Optional[str]:
    """
    Keep only capitalized tokens (Titlecase/ALLCAPS) from an alias, excluding generic/common words.

    Rules:
    - If *all* tokens are lowercase (no token whose first alpha char is uppercase), drop the alias (return None).
    - If mixed-case, keep only the tokens whose first alpha char is uppercase.
    - Preserve ALLCAPS acronyms/codenames.
    - Exclude generic/common English words even if capitalized (e.g., "I", "Which", "Work", "City").
    - Single letters: keep if ALLCAPS (e.g., "A", "M" - might be codenames), exclude if lowercase/titlecase.

    Examples:
    - "NEIGHBOUR or the NEIGHBOURS in GRU Venona traffic" -> "NEIGHBOUR NEIGHBOURS GRU Venona"
    - "from" -> None
    - "Army Air Force" -> "Army Air Force"
    - "work name \"The New York Times\"" -> "The New York Times"
    - "I" -> None (generic word, titlecase)
    - "A" -> "A" (ALLCAPS single letter, might be codename)
    - "Which" -> None (generic word)
    - "Work" -> None (generic word)
    - "PM" -> "PM" (ALLCAPS multi-char, might be codename)
    """
    if not alias:
        return None

    raw_tokens = alias.strip().split()
    if not raw_tokens:
        return None

    kept: List[str] = []
    for tok in raw_tokens:
        # Strip common surrounding punctuation, but keep internal punctuation like O'Neil, U.S., B-29.
        core = tok.strip("[](){}<>\"'“”‘’.,;:!?")
        if not core:
            continue
        if _first_alpha_is_upper(core):
            # Check if this is a generic word to exclude (case-insensitive check)
            core_lower = core.lower()
            if core_lower in GENERIC_WORDS_TO_EXCLUDE:
                # For single letters: keep if ALLCAPS (might be codename like "A", "M"), exclude if lowercase/titlecase
                if len(core) == 1:
                    if core.isupper():
                        # ALLCAPS single letter - keep it (might be codename)
                        kept.append(core)
                    # else: lowercase/titlecase single letter - skip it
                    continue
                # For multi-char words: skip generic words unless ALLCAPS (might be codename like "PM")
                if not core.isupper():
                    continue
            kept.append(core)

    if not kept:
        return None

    out = " ".join(kept).strip()
    return out or None


def ensure_alias(cur, source_id: int, entry_id: Optional[int], entity_id: int, al: ParsedAlias) -> Optional[int]:
    filtered = filter_alias_to_capitalized_tokens(al.alias)
    if filtered is None:
        return None

    alias_norm = normalize_alias_for_db(filtered)
    
    # Phase 1.1: Tier A - Set is_auto_match based on alias_type
    AUTO_MATCH_ALIAS_TYPES = {
        "canonical", "original_form", "bracket_variant", 
        "cover_name", "covername_from_body"
    }
    
    # Conditionally allow head_syn (only if not generic word)
    is_auto_match = al.alias_type in AUTO_MATCH_ALIAS_TYPES
    if al.alias_type == "head_syn":
        # Check if it's a generic word
        if filtered.lower() not in GENERIC_WORDS_TO_EXCLUDE:
            is_auto_match = True
    
    # Never auto-match these types
    NEVER_AUTO_MATCH_TYPES = {
        "definition", "scoped_label", "see", "work_name", "spelling_variant"
    }
    if al.alias_type in NEVER_AUTO_MATCH_TYPES:
        is_auto_match = False
    
    # Phase 1.2: Ban single-letter and 2-letter covernames unless quoted/bracketed/acronym
    tokens = alias_norm.split()
    if len(tokens) == 1:
        token = tokens[0]
        # Check if this is a covername (we'll check entity_type later, but also check alias_type)
        is_covername_alias = al.alias_type in ("cover_name", "covername_from_body", "bracket_variant")
        
        # Single letter: only allow if quoted/bracketed in original alias
        if len(token) == 1:
            original_has_quotes = '"' in al.alias or "'" in al.alias or '[' in al.alias or '(' in al.alias
            if is_covername_alias and not original_has_quotes:
                is_auto_match = False
        
        # 2-letter: only allow if ALLCAPS acronym (KGB, GRU, MGB, etc.)
        elif len(token) == 2:
            known_acronyms = {"kgb", "gru", "mgb", "nkvd", "sis", "mi6", "fbi", "cia", "nsa", "ussr"}
            if is_covername_alias:
                if token.lower() not in known_acronyms or not filtered.isupper():
                    is_auto_match = False
    
    # Phase 1.3: Stop auto-matching generic label entities
    GENERIC_LABEL_ALIASES = {
        "president", "general", "group", "ref", "minister", "chief", 
        "doctor", "agent", "officer", "director", "secretary"
    }
    if alias_norm.lower() in GENERIC_LABEL_ALIASES and len(tokens) == 1:
        is_auto_match = False
    
    # Check for existing alias using the unique constraint key: (entity_id, alias_norm)
    # This matches the database constraint entity_aliases_entity_id_alias_norm_key
    cur.execute(
        """
        SELECT id, is_auto_match FROM entity_aliases
        WHERE entity_id=%s AND alias_norm=%s
        ORDER BY id
        LIMIT 1;
        """,
        (entity_id, alias_norm),
    )
    row = cur.fetchone()
    if row:
        existing_id = int(row[0])
        existing_is_auto_match = row[1] if len(row) > 1 else None
        
        # Update is_auto_match if it exists and our computed value differs
        # (allows manual overrides to persist, but applies our rules as defaults)
        if existing_is_auto_match is None or existing_is_auto_match != is_auto_match:
            cur.execute(
                """
                UPDATE entity_aliases 
                SET is_auto_match = %s
                WHERE id = %s
                """,
                (is_auto_match, existing_id),
            )
        return existing_id

    # Check if is_auto_match column exists
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'entity_aliases' AND column_name = 'is_auto_match'
    """)
    has_is_auto_match = cur.fetchone() is not None

    if has_is_auto_match:
        cur.execute(
            """
            INSERT INTO entity_aliases(source_id, entry_id, entity_id, alias, alias_norm, alias_type, confidence, notes, is_auto_match)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (source_id, entry_id, entity_id, filtered, alias_norm, al.alias_type, al.confidence, al.notes, is_auto_match),
        )
    else:
        cur.execute(
            """
            INSERT INTO entity_aliases(source_id, entry_id, entity_id, alias, alias_norm, alias_type, confidence, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (source_id, entry_id, entity_id, filtered, alias_norm, al.alias_type, al.confidence, al.notes),
        )
    return int(cur.fetchone()[0])


def ensure_link(
    cur,
    source_id: int,
    entry_id: Optional[int],
    from_entity_id: int,
    to_entity_id: int,
    link: ParsedLink,
) -> int:
    cur.execute(
        """
        SELECT id FROM entity_links
        WHERE source_id=%s
          AND from_entity_id=%s
          AND to_entity_id=%s
          AND link_type=%s
          AND COALESCE(notes,'') = COALESCE(%s,'')
        ORDER BY id
        LIMIT 1;
        """,
        (source_id, from_entity_id, to_entity_id, link.link_type, link.notes),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])

    cur.execute(
        """
        INSERT INTO entity_links(source_id, entry_id, from_entity_id, to_entity_id, link_type, confidence, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (source_id, entry_id, from_entity_id, to_entity_id, link.link_type, link.confidence, link.notes),
    )
    return int(cur.fetchone()[0])


def ensure_citation(
    cur,
    source_id: int,
    entry_id: Optional[int],
    entity_id: Optional[int],
    alias_id: Optional[int],
    link_id: Optional[int],
    cit: ParsedCitation,
) -> int:
    cur.execute(
        """
        SELECT id FROM entity_citations
        WHERE source_id=%s
          AND COALESCE(entry_id,0)=COALESCE(%s,0)
          AND COALESCE(entity_id,0)=COALESCE(%s,0)
          AND COALESCE(alias_id,0)=COALESCE(%s,0)
          AND COALESCE(link_id,0)=COALESCE(%s,0)
          AND citation_text=%s
          AND COALESCE(notes,'') = COALESCE(%s,'')
        ORDER BY id
        LIMIT 1;
        """,
        (source_id, entry_id, entity_id, alias_id, link_id, cit.citation_text, cit.notes),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])

    collection_slug, document_label, page_list = best_effort_parse_citation_fields(cit.citation_text)

    cur.execute(
        """
        INSERT INTO entity_citations(
          source_id, entry_id, entity_id, alias_id, link_id,
          citation_text, collection_slug, document_label, page_list, notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (source_id, entry_id, entity_id, alias_id, link_id,
         cit.citation_text, collection_slug, document_label, page_list, cit.notes),
    )
    return int(cur.fetchone()[0])


# -----------------------
# Orchestration / utilities
# -----------------------

def print_parsed(pe: ParsedEntry):
    print("=" * 100)
    print(f"entry_key={pe.entry_key!r}  seq={pe.entry_seq}  entity={pe.entity_canonical!r}  type={pe.entity_type}")
    print(f"first_line: {pe.raw_text.splitlines()[0][:200]}")
    if pe.is_crossref_only:
        print(f"  CROSSREF: {pe.entry_key!r} -> {pe.crossref_target!r}")
    if pe.definitions:
        print("  definitions:")
        for d in pe.definitions:
            def_type_str = f" [{d.definition_type}]" if d.definition_type else ""
            notes_str = f" ({d.notes})" if d.notes else ""
            print(f"    - {d.definition_text!r}{def_type_str}{notes_str}")
    if pe.aliases:
        print("  aliases:")
        for a in pe.aliases:
            extra = f" conf={a.confidence}" if a.confidence else ""
            print(f"    - {a.alias!r} [{a.alias_type}]{extra}")
    if pe.links:
        print("  links:")
        for l in pe.links:
            print(f"    - {l.link_type}: {l.from_name!r} -> {l.to_name!r}  conf={l.confidence}  notes={l.notes}")
    if pe.citations:
        print(f"  citations: {len(pe.citations)}")
        for c in pe.citations[:8]:
            if c.alias_label:
                print(f"    - ({c.notes}) attach_to_alias={c.alias_label!r}: {c.citation_text[:160]}")
            else:
                print(f"    - {c.citation_text[:160]}")
        if len(pe.citations) > 8:
            print(f"    ... ({len(pe.citations)-8} more)")
    print()


def _entry_key_for_block(block: str) -> str:
    first = block.splitlines()[0].strip()
    head_raw, _, _ = split_head_and_body(first)
    return strip_outer_quotes(_normalize_quotes(head_raw).strip())


def compare_segmentations(blocks_a: List[str], blocks_b: List[str], max_show: int = 30):
    """Compares two segmentations by ordered first-line strings."""
    a_heads = [b.splitlines()[0].strip() for b in blocks_a]
    b_heads = [b.splitlines()[0].strip() for b in blocks_b]

    print(f"[compare] A blocks: {len(blocks_a)}  B blocks: {len(blocks_b)}")

    mism = []
    i = j = 0
    while i < len(a_heads) and j < len(b_heads) and len(mism) < max_show:
        if a_heads[i] == b_heads[j]:
            i += 1
            j += 1
            continue
        mism.append((i, a_heads[i], j, b_heads[j]))
        i += 1
        j += 1

    if mism:
        print(f"[compare] showing up to {max_show} mismatched head lines:")
        for (i, ah, j, bh) in mism:
            print(f"  - A[{i}] {ah[:180]}")
            print(f"    B[{j}] {bh[:180]}")
    else:
        print("[compare] no mismatches found in first-line sequence (within inspected window)")


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to concordance PDF file")
    ap.add_argument("--source-slug", required=True, help="concordance_sources.slug")
    ap.add_argument("--source-title", required=True, help="concordance_sources.title")
    ap.add_argument("--source-notes", default=None, help="Optional notes for concordance_sources")

    ap.add_argument("--marker", default="The Index", help='Marker string on page 7, default "The Index"')
    ap.add_argument("--marker-page", type=int, default=7, help="1-based page number that contains marker (default 7)")

    ap.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    ap.add_argument("--dry-run", action="store_true", help="Parse and print; do not write to DB")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N entries (useful with --dry-run)")
    ap.add_argument("--verbose", action="store_true", help="Print more debug info")

    ap.add_argument(
        "--segment",
        choices=["auto", "layout", "regex"],
        default="auto",
        help="Segmentation method: layout uses pdfplumber x0 indentation; regex uses heuristics; auto tries layout then falls back.",
    )
    ap.add_argument(
        "--indent-delta",
        type=float,
        default=6.0,
        help="Layout segmentation: how far from left margin a line can be and still count as headword (default 6.0).",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Compare layout vs regex segmentation counts + mismatched first lines (dry-run friendly).",
    )

    args = ap.parse_args()

    print(f"Starting PDF ingestion: {args.pdf}", file=sys.stderr)
    print(f"Source: {args.source_slug} - {args.source_title}", file=sys.stderr)
    print(f"Segmentation method: {args.segment}", file=sys.stderr)
    
    # Regex path needs marker-sliced plain text
    print(f"Extracting text from PDF (marker='{args.marker}' on page {args.marker_page})...", file=sys.stderr)
    extract_start = time.time()
    raw_index_text = slice_index_text_from_pdf(args.pdf, marker=args.marker, marker_page_1based=args.marker_page)
    extract_time = time.time() - extract_start
    print(f"✅ PDF text extraction complete: {len(raw_index_text)} characters in {extract_time:.1f}s", file=sys.stderr)
    
    # Remove footnotes from raw text before processing
    print("Removing footnotes from extracted text...", file=sys.stderr)
    raw_index_text = remove_footnotes_from_text(raw_index_text)
    print(f"✅ Footnote removal complete: {len(raw_index_text)} characters remaining", file=sys.stderr)
    
    print("Normalizing text and segmenting entries (regex method)...", file=sys.stderr)
    norm_start = time.time()
    norm_text = normalize_for_parsing(raw_index_text)
    blocks_regex = segment_entries_regex(norm_text)
    norm_time = time.time() - norm_start
    print(f"✅ Regex segmentation complete: {len(blocks_regex)} entries found in {norm_time:.1f}s", file=sys.stderr)

    blocks_layout: Optional[List[str]] = None
    layout_error: Optional[str] = None

    # Try layout-based segmentation if needed
    if args.segment in ("auto", "layout"):
        print("Extracting layout information from PDF (this may take several minutes for large PDFs)...", file=sys.stderr)
        layout_extract_start = time.time()
        try:
            layout_lines = extract_index_lines_layout(args.pdf, marker=args.marker, marker_page_1based=args.marker_page, verbose=True)
            layout_extract_time = time.time() - layout_extract_start
            print(f"✅ Layout extraction complete: {len(layout_lines)} lines extracted in {layout_extract_time:.1f}s", file=sys.stderr)
            
            # Remove footnotes from layout lines
            print("Removing footnotes from layout lines...", file=sys.stderr)
            layout_lines = remove_footnotes_from_layout_lines(layout_lines)
            print(f"✅ Footnote removal complete: {len(layout_lines)} lines remaining", file=sys.stderr)
            
            print(f"Segmenting entries using layout method (indent_delta={args.indent_delta})...", file=sys.stderr)
            layout_seg_start = time.time()
            blocks_layout = segment_entries_layout(layout_lines, indent_delta=args.indent_delta, verbose=args.verbose)
            layout_seg_time = time.time() - layout_seg_start
            print(f"✅ Layout segmentation complete: {len(blocks_layout)} entries found in {layout_seg_time:.1f}s", file=sys.stderr)
        except Exception as e:
            layout_error = str(e)
            blocks_layout = None
            print(f"❌ Layout extraction/segmentation failed: {layout_error}", file=sys.stderr)
            if args.segment == "layout":
                raise RuntimeError(f"Layout segmentation requested but failed: {layout_error}")

    if args.compare:
        if blocks_layout is None:
            print(f"[compare] layout segmentation unavailable: {layout_error}")
        else:
            compare_segmentations(blocks_regex, blocks_layout, max_show=40)
        if args.dry_run:
            return

    # Choose segmentation
    print("\n" + "="*60, file=sys.stderr)
    print("SEGMENTATION SUMMARY", file=sys.stderr)
    print("="*60, file=sys.stderr)
    if args.segment == "regex":
        blocks = blocks_regex
        print(f"✅ Using REGEX method: {len(blocks)} entries", file=sys.stderr)
    elif args.segment == "layout":
        if blocks_layout is None:
            raise RuntimeError(f"Layout segmentation requested but unavailable: {layout_error}")
        blocks = blocks_layout
        print(f"✅ Using LAYOUT (position-aware) method: {len(blocks)} entries", file=sys.stderr)
        print(f"   (Regex method found {len(blocks_regex)} entries for comparison)", file=sys.stderr)
    else:  # auto
        if blocks_layout is not None and len(blocks_layout) >= max(1, int(0.7 * len(blocks_regex))):
            blocks = blocks_layout
            print(f"✅ Auto-selected LAYOUT (position-aware) method: {len(blocks)} entries", file=sys.stderr)
            print(f"   (Regex method found {len(blocks_regex)} entries)", file=sys.stderr)
        else:
            blocks = blocks_regex
            print(f"✅ Auto-selected REGEX method: {len(blocks)} entries", file=sys.stderr)
            print(f"   (Layout method unavailable or suspicious: {layout_error})", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    if args.limit is not None:
        blocks = blocks[:args.limit]

    if args.verbose or args.dry_run:
        print(f"Segmented {len(blocks)} entry blocks")

    # Deterministic entry_seq per entry_key (within this run)
    key_counts: Dict[str, int] = {}
    parsed: List[ParsedEntry] = []

    total_blocks = len(blocks)
    parse_start_time = time.time()
    last_reported_pct = -1
    
    print(f"Parsing {total_blocks} entries...", file=sys.stderr)

    for idx, block in enumerate(blocks, 1):
        entry_key = _entry_key_for_block(block)
        key_counts.setdefault(entry_key, 0)
        key_counts[entry_key] += 1
        entry_seq = key_counts[entry_key]

        pe = parse_entry_block(block, entry_seq=entry_seq)
        pe.entry_key = entry_key
        pe.entry_seq = entry_seq
        parsed.append(pe)

        if args.dry_run:
            print_parsed(pe)
        
        # Progress reporting: every 10%
        progress_pct = (idx / total_blocks) * 100
        current_10pct = int(progress_pct // 10)
        last_10pct = int(last_reported_pct // 10)
        
        if current_10pct > last_10pct or idx == total_blocks:
            elapsed = time.time() - parse_start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total_blocks - idx) / rate if rate > 0 else 0
            print(
                f"[Progress] {idx}/{total_blocks} ({progress_pct:.1f}%) | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Rate: {rate:.1f} entries/s | "
                f"ETA: {remaining:.1f}s",
                file=sys.stderr
            )
            last_reported_pct = progress_pct
    
    parse_time = time.time() - parse_start_time
    print(f"✅ Parsing complete: {total_blocks} entries in {parse_time:.1f}s", file=sys.stderr)

    if args.dry_run:
        print("\n(dry-run) ✅ done")
        return

    # Write to DB
    conn = get_conn(args.db_url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            source_id = ensure_source(cur, args.source_slug, args.source_title, args.source_notes)
        conn.commit()

        total_parsed = len(parsed)
        db_start_time = time.time()
        last_reported_pct = -1
        
        print(f"Writing {total_parsed} entries to database...", file=sys.stderr)

        for idx, pe in enumerate(parsed, 1):
            with conn.cursor() as cur:
                entry_id = upsert_entry(cur, source_id, pe.entry_key, pe.entry_seq, pe.raw_text)

                # Crossref-only entries: map entry_key as alias to target
                if pe.is_crossref_only:
                    if not pe.crossref_target:
                        continue
                    target_name = strip_outer_quotes(_normalize_quotes(pe.crossref_target).strip())
                    target_eid = ensure_entity(cur, source_id, None, target_name, "other", None, None)
                    ensure_alias(cur, source_id, entry_id, target_eid, ParsedAlias(alias=pe.entry_key, alias_type="see"))
                    continue

                # Ensure main entity
                eid = ensure_entity(cur, source_id, entry_id, pe.entity_canonical, pe.entity_type, pe.confidence, pe.notes)

                # Ensure aliases
                alias_ids_by_text: Dict[str, int] = {}
                for al in pe.aliases:
                    aid = ensure_alias(cur, source_id, entry_id, eid, al)
                    if aid is not None:
                        alias_ids_by_text[al.alias.lower()] = aid

                # Ensure links
                for lk in pe.links:
                    from_name = strip_outer_quotes(_normalize_quotes(lk.from_name).strip())
                    to_name = strip_outer_quotes(_normalize_quotes(lk.to_name).strip())

                    # Heuristic: cover-name-ish links create cover_name entities for from_name
                    if lk.link_type in ("cover_name_of", "changed_to"):
                        from_eid = ensure_entity(cur, source_id, None, from_name, "cover_name", lk.confidence, None)
                    else:
                        from_eid = ensure_entity(cur, source_id, None, from_name, "other", lk.confidence, None)

                    to_type = "person" if looks_like_person_head(to_name) else "other"
                    to_eid = ensure_entity(cur, source_id, None, to_name, to_type, lk.confidence, None)

                    # Skip self-links (DB constraint expects from != to)
                    if from_eid == to_eid:
                        if args.verbose:
                            print(f"[warn] skipping self-link {lk.link_type}: {from_name!r} -> {to_name!r} (entity_id={from_eid})")
                        continue

                    ensure_link(cur, source_id, entry_id, from_eid, to_eid, lk)

                # Ensure citations
                for cit in pe.citations:
                    alias_id = None
                    if cit.alias_label:
                        alias_id = alias_ids_by_text.get(cit.alias_label.lower())
                        if alias_id is None:
                            alias_id = ensure_alias(
                                cur,
                                source_id,
                                entry_id,
                                eid,
                                ParsedAlias(alias=cit.alias_label, alias_type="scoped_label"),
                            )
                            if alias_id is not None:
                                alias_ids_by_text[cit.alias_label.lower()] = alias_id

                    ensure_citation(
                        cur,
                        source_id=source_id,
                        entry_id=entry_id,
                        entity_id=(None if alias_id else eid),
                        alias_id=alias_id,
                        link_id=None,
                        cit=cit,
                    )

            conn.commit()
            
            # Progress reporting: every 10%
            progress_pct = (idx / total_parsed) * 100
            current_10pct = int(progress_pct // 10)
            last_10pct = int(last_reported_pct // 10)
            
            if current_10pct > last_10pct or idx == total_parsed:
                elapsed = time.time() - db_start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = (total_parsed - idx) / rate if rate > 0 else 0
                print(
                    f"[DB Progress] {idx}/{total_parsed} ({progress_pct:.1f}%) | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"Rate: {rate:.1f} entries/s | "
                    f"ETA: {remaining:.1f}s",
                    file=sys.stderr
                )
                last_reported_pct = progress_pct
        
        db_time = time.time() - db_start_time
        total_time = time.time() - parse_start_time
        print(f"✅ Ingest complete: source_slug={args.source_slug}  entries={len(parsed)}", file=sys.stderr)
        print(f"   Total time: {total_time:.1f}s (parse: {parse_time:.1f}s, DB: {db_time:.1f}s)", file=sys.stderr)

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
