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

import os
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import psycopg2


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
        if marker_page_1based < 1 or marker_page_1based > n_pages:
            raise RuntimeError(f"marker_page_1based={marker_page_1based} out of range (PDF has {n_pages} pages)")

        p0 = marker_page_1based - 1
        marker_page = pdf.pages[p0]

        def page_to_lines(page, page_no_1based: int) -> List[PdfLine]:
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

        marker_lines = page_to_lines(marker_page, marker_page_1based)
        idx = _marker_line_reached(marker_lines, marker)
        if idx is None:
            raise RuntimeError(f'Marker "{marker}" not found in layout text on page {marker_page_1based}')

        # Lines strictly after the marker line on the marker page
        out.extend(marker_lines[idx + 1:])

        # Subsequent pages
        for pi in range(p0 + 1, n_pages):
            out.extend(page_to_lines(pdf.pages[pi], pi + 1))

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

    def is_layout_headword_line(ln: PdfLine) -> bool:
        return (ln.x0 <= threshold) and _looks_like_headword_text_layout(ln.text)

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
        if in_footnotes:
            if is_layout_headword_line(ln):
                in_footnotes = False
            else:
                continue

        # Normal entry handling
        if is_layout_headword_line(ln):
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


def looks_like_person_head(head: str) -> bool:
    return "," in head.strip()


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


def infer_referent_from_body_start(body: str) -> Optional[str]:
    b = body.strip()
    if not b:
        return None
    b_norm = _normalize_quotes(b)
    first_sentence = b_norm.split(".", 1)[0].strip()
    if not first_sentence:
        return None

    bad_starts = (
        "unidentified", "likely", "?", "described as", "initial", "abbreviation",
        "refers to", "translated as", "see "
    )
    if first_sentence.lower().startswith(bad_starts):
        return None

    # Trim temporal qualifiers: "Jack Soble prior to ..." -> "Jack Soble"
    first_sentence = re.split(
        r"\b(prior to|after|before|from|until|since)\b",
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
    if tail and re.fullmatch(r"[0-9,\s]+", tail):
        for tok in tail.split(","):
            tok = tok.strip()
            if tok.isdigit():
                ints.append(int(tok))
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

    # Crossref-only entries: "X: See Y."
    if is_crossref:
        m = re.match(r'^see\s+["“]?([^".;]+)["”]?', _normalize_quotes(body_first).strip(), flags=re.IGNORECASE)
        if m:
            crossref_target = m.group(1).strip()
        pe = ParsedEntry(
            entry_key=entry_key,
            entry_seq=entry_seq,
            raw_text=raw_text,
            entity_canonical=canonical if canonical else entry_key,
            entity_type=entity_type,
            is_crossref_only=True,
            crossref_target=crossref_target,
        )
        pe.aliases.append(ParsedAlias(alias=entry_key, alias_type="see"))
        return pe

    pe = ParsedEntry(
        entry_key=entry_key,
        entry_seq=entry_seq,
        raw_text=raw_text,
        entity_canonical=canonical if canonical else entry_key,
        entity_type=entity_type,
    )

    # Canonical alias
    if pe.entity_canonical:
        pe.aliases.append(ParsedAlias(alias=pe.entity_canonical, alias_type="canonical"))

    # Head synonyms
    for syn in head_syns:
        syn2 = syn.strip()
        if delim == ".":
            syn2 = syn2.rstrip(".").strip()
        if syn2 and syn2.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=syn2, alias_type="head_syn"))

    # Bracket variants (you said you trust these)
    for bt in bracket_tokens:
        bt2 = bt.strip()
        if delim == ".":
            bt2 = bt2.rstrip(".").strip()
        if bt2 and bt2.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=bt2, alias_type="bracket_variant"))

    # Additional quoted variants
    for q in quoted[1:]:
        q2 = strip_outer_quotes(q.strip())
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

    # Body: cover name in Venona/Vassiliev mentioned inside person/topic entry
    for m in re.finditer(
        r"Cover\s+name\s+in\s+(Venona|Vassiliev[’']s notebooks)\s*:\s*([\"“]?)([^\"”.;,]+)\2",
        _normalize_quotes(body_joined),
        flags=re.IGNORECASE
    ):
        cn = m.group(3).strip()
        if cn:
            pe.links.append(ParsedLink(
                link_type="cover_name_of",
                from_name=cn,
                to_name=pe.entity_canonical,
                confidence="certain",
                notes=f"Cover name in {m.group(1)}"
            ))

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
        referent = infer_referent_from_body_start(body_first + " " + "\n".join(rest_lines))
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

    body_norm = _normalize_quotes(body_joined)
    matches = list(scoped_pat.finditer(body_norm))
    for i, m in enumerate(matches):
        label = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body_norm)
        scoped_text = body_norm[start:end].strip()
        scoped_spans.append((m.start(), end))

        if label:
            pe.aliases.append(ParsedAlias(alias=label, alias_type="scoped_label"))

        for chunk in parse_citation_chunks(scoped_text):
            pe.citations.append(ParsedCitation(
                citation_text=chunk,
                alias_label=label,
                notes=f"{m.group(1).title()} {label}"
            ))

    # Unscoped citations: remove scoped regions and parse remainder
    if matches:
        rem = body_norm
        for (a, b) in reversed(scoped_spans):
            rem = rem[:a] + " " + rem[b:]
        for chunk in parse_citation_chunks(rem):
            pe.citations.append(ParsedCitation(citation_text=chunk, alias_label=None, notes=None))
    else:
        for chunk in parse_citation_chunks(body_norm):
            pe.citations.append(ParsedCitation(citation_text=chunk, alias_label=None, notes=None))

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


def ensure_alias(cur, source_id: int, entry_id: Optional[int], entity_id: int, al: ParsedAlias) -> int:
    cur.execute(
        """
        SELECT id FROM entity_aliases
        WHERE source_id=%s AND entity_id=%s AND alias=%s
        ORDER BY id
        LIMIT 1;
        """,
        (source_id, entity_id, al.alias),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])

    cur.execute(
        """
        INSERT INTO entity_aliases(source_id, entry_id, entity_id, alias, alias_type, confidence, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (source_id, entry_id, entity_id, al.alias, al.alias_type, al.confidence, al.notes),
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

    # Regex path needs marker-sliced plain text
    raw_index_text = slice_index_text_from_pdf(args.pdf, marker=args.marker, marker_page_1based=args.marker_page)
    norm_text = normalize_for_parsing(raw_index_text)
    blocks_regex = segment_entries_regex(norm_text)

    blocks_layout: Optional[List[str]] = None
    layout_error: Optional[str] = None

    try:
        layout_lines = extract_index_lines_layout(args.pdf, marker=args.marker, marker_page_1based=args.marker_page)
        blocks_layout = segment_entries_layout(layout_lines, indent_delta=args.indent_delta, verbose=args.verbose)
    except Exception as e:
        layout_error = str(e)
        blocks_layout = None

    if args.compare:
        if blocks_layout is None:
            print(f"[compare] layout segmentation unavailable: {layout_error}")
        else:
            compare_segmentations(blocks_regex, blocks_layout, max_show=40)
        if args.dry_run:
            return

    # Choose segmentation
    if args.segment == "regex":
        blocks = blocks_regex
        if args.verbose:
            print(f"[segment] using regex blocks={len(blocks)}")
    elif args.segment == "layout":
        if blocks_layout is None:
            raise RuntimeError(f"layout segmentation requested but unavailable: {layout_error}")
        blocks = blocks_layout
        if args.verbose:
            print(f"[segment] using layout blocks={len(blocks)}")
    else:  # auto
        if blocks_layout is not None and len(blocks_layout) >= max(1, int(0.7 * len(blocks_regex))):
            blocks = blocks_layout
            if args.verbose:
                print(f"[segment] auto chose layout blocks={len(blocks)} (regex would be {len(blocks_regex)})")
        else:
            blocks = blocks_regex
            if args.verbose:
                print(f"[segment] auto chose regex blocks={len(blocks)} (layout unavailable or suspicious: {layout_error})")

    if args.limit is not None:
        blocks = blocks[:args.limit]

    if args.verbose or args.dry_run:
        print(f"Segmented {len(blocks)} entry blocks")

    # Deterministic entry_seq per entry_key (within this run)
    key_counts: Dict[str, int] = {}
    parsed: List[ParsedEntry] = []

    for block in blocks:
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

        for pe in parsed:
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

        print(f"✅ ingest complete: source_slug={args.source_slug}  entries={len(parsed)}")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
