#!/usr/bin/env python3
"""

Ingest an expert-curated concordance index PDF into Postgres tables:
  concordance_sources, concordance_entries, entities, entity_aliases, entity_links, entity_citations

Key behaviors (per your spec + edge cases discussed):
- Only ingest the concordance index body: everything AFTER the marker "The Index" on PDF page 7 (1-based).
- Lossless storage of each entry as concordance_entries.raw_text.
- Robust entry segmentation across:
  - "Headword: ..." entries
  - "Headword. Unidentified. ..." entries (no colon)
  - Numeric / numbered / undeciphered formats (e.g., "10", 12, Source No., Undeciphered Name No. 20)
  - Hyphenated line-wrap artifacts from PDF extraction
  - Footnote blocks and separator lines (ignored or kept within the current entry, never start their own entry)
- Query expansion oriented parsing:
  - headword aliases: quotes, brackets, and "and" synonyms (including translation variants by default)
  - body aliases: "work name", "Name spelled", "Russian original ... See ...", "Cover name in ...: ..."
  - links: cover-name ↔ referent, "changed to", "See ..." crossrefs
  - citations: scoped by "As X:" and "Translated as X:" blocks, otherwise best-effort citations split on ';'
- Dry run mode prints what would be inserted without writing anything.

Usage examples:
  python concordance/ingest_concordance.py \
    --pdf data/concordance.pdf \
    --source-slug venona_vassiliev_concordance_v1 \
    --source-title "Venona/Vassiliev Concordance Index" \
    --dry-run --limit 50

  python concordance/ingest_concordance.py \
    --pdf data/concordance.pdf \
    --source-slug venona_vassiliev_concordance_v1 \
    --source-title "Venona/Vassiliev Concordance Index"

Env:
  DATABASE_URL (required unless --db-url is provided)

Dependencies:
  - psycopg2
  - pypdf (preferred) or PyPDF2
"""

import os
import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Iterable

import psycopg2


# -----------------------
# PDF extraction helpers
# -----------------------

def _extract_pdf_text_pages(pdf_path: str) -> List[str]:
    """
    Returns a list of page texts (best-effort). Prefers pypdf; falls back to PyPDF2.
    """
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(pdf_path)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return pages
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(pdf_path)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return pages
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract PDF text from {pdf_path}. Install pypdf (preferred) or PyPDF2. Error: {e}"
            )


def slice_index_text_from_pdf(pdf_path: str, marker: str = "The Index", marker_page_1based: int = 7) -> str:
    """
    Extracts text from the PDF starting AFTER the first occurrence of `marker`
    on `marker_page_1based`, plus all subsequent pages.
    """
    pages = _extract_pdf_text_pages(pdf_path)
    if marker_page_1based < 1 or marker_page_1based > len(pages):
        raise RuntimeError(f"marker_page_1based={marker_page_1based} out of range (PDF has {len(pages)} pages)")

    idx0 = marker_page_1based - 1
    page_text = pages[idx0] or ""

    # Normalize whitespace for finding the marker robustly
    norm = re.sub(r"\s+", " ", page_text)
    m = re.search(re.escape(marker), norm)
    if not m:
        # Try a more forgiving match (case-insensitive)
        m = re.search(re.escape(marker), norm, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(f'Marker "{marker}" not found on PDF page {marker_page_1based}')

    # We slice the *original* page text by finding the marker in a lightly-normalized version
    # and falling back to cutting the original at the first raw occurrence if possible.
    raw_pos = page_text.find(marker)
    if raw_pos == -1:
        # Case-insensitive raw find
        raw_pos = page_text.lower().find(marker.lower())
    if raw_pos == -1:
        # Worst case: just keep the page text and rely on later segmentation
        sliced_first = page_text
    else:
        sliced_first = page_text[raw_pos + len(marker):]

    tail_pages = pages[idx0 + 1:]
    all_text = "\n".join([sliced_first] + tail_pages)
    return all_text


# -----------------------
# Text normalization (for parsing only)
# -----------------------

def normalize_for_parsing(text: str) -> str:
    """
    Best-effort cleanup for parsing:
    - join hyphenated line breaks: 'Anglo-\nAmerican' -> 'AngloAmerican' (or 'Anglo-American' if hyphen is real)
      We only join when both sides are word chars.
    - normalize weird whitespace
    """
    # Join hyphen + newline in the middle of a word
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize CRLF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces on lines
    text = "\n".join([ln.rstrip() for ln in text.split("\n")])
    return text

def remove_footnote_blocks_for_parsing(text: str) -> str:
    lines = text.split("\n")
    out = []
    in_fn = False

    for ln in lines:
        s = ln.strip()

        # Start footnote block when we see a footnote marker after a dash separator
        if _RE_LONG_DASH_LINE.match(s) or _RE_SEPARATOR.match(s):
            in_fn = True
            continue

        # Or start when a numbered footnote line appears
        if _RE_FOOTNOTE_START.match(s) or _RE_FOOTNOTE_LIKE.match(s):
            in_fn = True
            continue

        # End footnote block when we hit a new entry start candidate.
        # IMPORTANT: use a lightweight heuristic here to avoid recursion:
        if in_fn and (_RE_COLON_START.match(s) or _RE_UNDECIPHERED_START.match(s) or _RE_SOURCE_NO_START.match(s)):
            in_fn = False

        if in_fn:
            continue

        # Drop stray single integers that are almost always wrap artifacts in citations
        if _RE_BARE_PAGEISH_INT.match(s):
            continue

        out.append(ln)

    return "\n".join(out)

def strip_footnotes_from_block_for_parsing(block: str) -> str:
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
# Entry segmentation
# -----------------------

_RE_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$")
_RE_SEPARATOR = re.compile(r"^\s*[-—]{3,}\s*$")

_RE_COLON_START = re.compile(r"^\S.{0,220}:\s")
_RE_UNDECIPHERED_START = re.compile(r"^Undeciphered Name No\.\s*\d+\b")
_RE_SOURCE_NO_START = re.compile(r"^\d+\s*,\s*Source No\.\b")
_RE_PERIOD_START = re.compile(
    r'^[\"“\']?[^.\n]{1,220}\.\s+(?:Unidentified|Venona|Vassiliev|As|See|Cover|Translated)\b'
)
_RE_LONG_DASH_LINE = re.compile(r"^\s*[—-]{5,}\s*$")  # emdash or hyphen runs
_RE_FOOTNOTE_START = re.compile(r'^\s*\d+\.\s+["“]')  # 4. “Comintern...
_RE_BARE_PAGEISH_INT = re.compile(r"^\s*\d{1,3}\s*$")  # handles stray "7" lines

# Footnote/bibliography lines that should not start a new entry
_RE_FOOTNOTE_LIKE = re.compile(r"^\s*\d+\.\s+\S")

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



def is_entry_start(line: str) -> bool:
    s = line.strip()
    if not s or is_noise_line(s) or _RE_FOOTNOTE_LIKE.match(s):
        return False

    # Continuations often start lowercase (in, via, and, etc.)
    # Do not block digit-start (numeric entries)
    if s[0].islower():
        return False
    if _RE_FOOTNOTE_LIKE.match(s):
        return False
    if _RE_FOOTNOTE_START.match(s):
        return False

    if _RE_UNDECIPHERED_START.match(s):
        return True
    if _RE_SOURCE_NO_START.match(s):
        return True
    if _RE_COLON_START.match(s):
        return True
    if _RE_PERIOD_START.match(s):
        return True
    return False


def segment_entries(text: str) -> List[str]:
    """
    Segments the normalized text into entry blocks (raw blocks as strings).
    """
    lines = text.split("\n")
    blocks: List[List[str]] = []
    cur: List[str] = []

    for ln in lines:
        if is_noise_line(ln):
            continue

        if is_entry_start(ln):
            if cur:
                blocks.append(cur)
            cur = [ln]
        else:
            if not cur:
                # We are before the first entry; ignore
                continue
            cur.append(ln)

    if cur:
        blocks.append(cur)

    # Trim each block, keep original newlines
    out = []
    for b in blocks:
        # Remove leading/trailing blank lines within block
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
    # Normalize curly quotes to straight quotes
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
    # Heuristic: "Lastname, Firstname" or "Lastname, ?" etc.
    h = head.strip()
    if "," in h:
        return True
    # Some person entries could be "Bruno Pontecorvo:" (no comma), but this is rare in your samples.
    return False


def classify_entity_type(entry_key: str, body: str) -> str:
    t = (entry_key + " " + body).lower()
    if "cover name" in t:
        return "cover_name"
    if "undeciphered name no." in t:
        return "cover_name"
    if "source no." in t and ("venona" in t or "gru" in t):
        return "cover_name"
    if looks_like_person_head(entry_key):
        return "person"
    # Topic-ish
    if any(k in t for k in ["related subjects", "references in", "all of vassiliev", "devoted to"]):
        return "topic"
    return "other"


def split_head_and_body(first_line: str) -> Tuple[str, str, str]:
    """
    Returns (head_raw, body_firstline_remainder, delimiter_used)
    delimiter_used is ':' or '.'
    """
    if ":" in first_line:
        left, right = first_line.split(":", 1)
        return left.strip(), right.strip(), ":"
    # fallback: split on first period
    m = re.search(r"\.", first_line)
    if not m:
        return first_line.strip(), "", "."
    i = m.start()
    left = first_line[:i].strip()
    right = first_line[i+1:].strip()
    return left, right, "."


def remove_trailing_descriptor_paren(head: str) -> str:
    """
    Removes a trailing parenthetical descriptor if it looks like index metadata.
    """
    m = re.search(r"\(([^)]{1,120})\)\s*$", head.strip())
    if not m:
        return head
    inner = m.group(1).lower()
    if any(k in inner for k in ["cover name", "russian original", "vassiliev", "venona", "original"]):
        return head[:m.start()].strip()
    return head


def extract_quoted_strings(s: str) -> List[str]:
    s = _normalize_quotes(s)
    out = []
    # Double quotes
    out.extend(re.findall(r'"([^"]+)"', s))
    return [x.strip() for x in out if x.strip()]


def extract_bracket_tokens(s: str) -> List[str]:
    """
    Extract tokens inside [...] possibly separated by 'and' or commas.
    Keeps apostrophes like ARNOL'D and hyphen forms like 101-y.
    """
    s = _normalize_quotes(s)
    items: List[str] = []
    for inner in re.findall(r"\[([^\]]+)\]", s):
        inner = inner.strip()
        # Strip any quotes inside brackets
        inner = strip_outer_quotes(inner.replace('"', '').replace("'", ""))
        # Split on ' and ' and commas
        parts = re.split(r"\s+and\s+|,\s*", inner)
        for p in parts:
            p = p.strip()
            if p:
                items.append(p)
    return items


def extract_head_synonyms(head_clean: str) -> List[str]:
    """
    Extract synonyms from head outside brackets using 'and' (e.g., "BEAR CUBS and BEARCUBS").
    """
    base = re.sub(r"\[[^\]]*\]", "", head_clean).strip()
    # If base contains ' and ' split; otherwise single
    parts = re.split(r"\s+and\s+", base)
    parts = [strip_outer_quotes(p.strip()) for p in parts if p.strip()]
    return parts


def infer_referent_from_body_start(body: str) -> Optional[str]:
    """
    If a cover-name entry looks like: ': <referent>. ...', return <referent>.
    Conservative: skip if starts with Unidentified / Likely / ? etc.
    """
    b = body.strip()
    if not b:
        return None
    # Remove leading parentheses/quotes junk
    b_norm = _normalize_quotes(b)
    # Take up to first period
    first_sentence = b_norm.split(".", 1)[0].strip()
    if not first_sentence:
        return None
    bad_starts = ("unidentified", "likely", "?", "described as", "initial", "abbreviation", "refers to", "translated as", "see ")
    if first_sentence.lower().startswith(bad_starts):
        return None
    # Avoid grabbing "Soviet intelligence officer Andrey Raina, pseudonym ..."
    # We'll still take the full name phrase and let aliasing handle.
    return strip_outer_quotes(first_sentence)


def parse_citation_chunks(text: str) -> List[str]:
    """
    Split a citation run into chunks, primarily on ';'.
    Keeps chunks that contain Venona/Vassiliev.
    """
    chunks = []
    for part in text.split(";"):
        p = part.strip()
        if not p:
            continue
        if ("Venona" in p) or ("Vassiliev" in p) or ("Vassiliev’s" in p) or ("Vassiliev's" in p):
            chunks.append(p)
    return chunks


def best_effort_parse_citation_fields(citation_text: str) -> Tuple[Optional[str], Optional[str], Optional[List[int]]]:
    """
    Returns (collection_slug, document_label, page_list)
    Only parses page_list when it's a clean comma-separated list of integers.
    """
    ct = citation_text.strip()
    collection_slug = None
    if ct.startswith("Venona"):
        collection_slug = "venona"
    elif ct.startswith("Vassiliev") or ct.startswith("Vassiliev’s") or ct.startswith("Vassiliev's"):
        collection_slug = "vassiliev"

    # document_label: up to first comma
    if "," in ct:
        document_label = ct.split(",", 1)[0].strip()
        tail = ct.split(",", 1)[1].strip()
    else:
        document_label = ct
        tail = ""

    # parse ints if possible
    ints = []
    if tail and re.fullmatch(r"[0-9,\s]+", tail):
        for tok in tail.split(","):
            tok = tok.strip()
            if tok.isdigit():
                ints.append(int(tok))
        if not ints:
            ints = []
    else:
        ints = []

    return collection_slug, document_label, (ints if ints else None)


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
    alias_label: Optional[str] = None  # attach to alias if present
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


def parse_entry_block(block: str, entry_seq: int) -> ParsedEntry:
    """
    Parses a single entry block into entity/aliases/links/citations.
    Always conservative: anything not confidently parsed remains only in raw_text.
    """
    raw_text = block
    parse_block = strip_footnotes_from_block_for_parsing(block)
    lines = parse_block.split("\n")
    first_line = lines[0].strip()
    rest_lines = lines[1:]
    lines = block.split("\n")
    first_line = lines[0].strip()
    rest_lines = lines[1:]
    rest_lines = [ln for ln in rest_lines if not _RE_BARE_PAGEISH_INT.match(ln.strip())]

    head_raw, body_first, delim = split_head_and_body(first_line)
    head_raw_norm = _normalize_quotes(head_raw).strip()
    body_joined = "\n".join([body_first] + rest_lines).strip()

    # Crossref-only: "X: See Y."
    body_first_lc = _normalize_quotes(body_first).strip().lower()
    is_crossref = body_first_lc.startswith("see ")
    crossref_target = None

    entry_key = strip_outer_quotes(head_raw_norm).strip()

    # Determine entity type (may get updated later)
    entity_type = classify_entity_type(entry_key, body_joined)

    # Prepare head parsing
    head_no_desc = remove_trailing_descriptor_paren(head_raw_norm)
    head_no_desc = head_no_desc.strip()

    # Extract head aliases
    quoted = extract_quoted_strings(head_no_desc)
    bracket_tokens = extract_bracket_tokens(head_no_desc)
    head_syns = extract_head_synonyms(head_no_desc)

    # Canonical choice
    if quoted:
        canonical = strip_outer_quotes(quoted[0])
    elif head_syns:
        canonical = strip_outer_quotes(head_syns[0])
    else:
        # fall back: strip bracket portion and punctuation
        canonical = re.sub(r"\[[^\]]*\]", "", head_no_desc).strip()
        canonical = strip_outer_quotes(canonical)

    canonical = canonical.strip()

    # Only strip trailing '.' when the head/body delimiter was '.' (i.e., "Headword. Unidentified...")
    # For colon-delimited entries like "Aarons, L.A.:", keep the dot(s) in initials.
    if delim == ".":
        canonical = canonical.rstrip(".").strip()

    # Crossref handling: don't create a new entity; map alias -> target later
    if is_crossref:
        # Try to extract the target after "See"
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
        # The alias is the entry_key (or canonical) pointing to crossref_target
        # We still emit it as an alias parse artifact for dry-run visibility.
        pe.aliases.append(ParsedAlias(alias=entry_key, alias_type="see"))
        return pe

    pe = ParsedEntry(
        entry_key=entry_key,
        entry_seq=entry_seq,
        raw_text=raw_text,
        entity_canonical=canonical if canonical else entry_key,
        entity_type=entity_type,
    )

    # Always include canonical alias
    if pe.entity_canonical:
        pe.aliases.append(ParsedAlias(alias=pe.entity_canonical, alias_type="canonical"))

    # Add head synonyms (outside bracket)
    for syn in head_syns:
        syn2 = syn.strip().rstrip(".")
        if syn2 and syn2.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=syn2, alias_type="head_syn"))

    # Add bracket variants (translation/original/etc.) — included by default in expansion per your policy
    for bt in bracket_tokens:
        bt2 = bt.strip().rstrip(".")
        if bt2 and bt2.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=bt2, alias_type="bracket_variant"))

    # If head had multiple quoted strings but we only used first as canonical, include the others
    for q in quoted[1:]:
        q2 = strip_outer_quotes(q.strip().rstrip("."))
        if q2 and q2.lower() != pe.entity_canonical.lower():
            pe.aliases.append(ParsedAlias(alias=q2, alias_type="head_quote_variant"))

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

    # Body: cover name in Venona/Vassiliev mentioned inside a person/topic entry
    # Example: "Cover name in Venona: SERGEJ."
    for m in re.finditer(r"Cover name in\s+(Venona|Vassiliev[’']s notebooks)\s*:\s*([\"“]?)([^\"”.;,]+)\2",
                         _normalize_quotes(body_joined), flags=re.IGNORECASE):
        cn = m.group(3).strip()
        if cn:
            # Create a link between cover name entity and this entity (direction: cover_name_of -> current)
            pe.links.append(ParsedLink(
                link_type="cover_name_of",
                from_name=cn,
                to_name=pe.entity_canonical,
                confidence="certain",
                notes=f"Cover name in {m.group(1)}"
            ))

    # Body: changed to (cover name evolution)
    # Example: "FAKIR was changed to ARNOLD"
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

    # If this entry is a cover_name, infer referent from the beginning of body
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

    # Scoped citation sections: As "X": ...  / Translated as X: ...
    # We'll collect these first; anything left over becomes unscoped citations.
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

        # Ensure label alias exists (for attaching citations)
        if label:
            pe.aliases.append(ParsedAlias(alias=label, alias_type="scoped_label"))

        for chunk in parse_citation_chunks(scoped_text):
            pe.citations.append(ParsedCitation(
                citation_text=chunk,
                alias_label=label,
                notes=f"{m.group(1).title()} {label}"
            ))

    # Unscoped citations: remove scoped portions and parse remaining
    if matches:
        # Remove scoped regions
        rem = body_norm
        # Replace from last to first to keep indices valid
        for (a, b) in reversed(scoped_spans):
            rem = rem[:a] + " " + rem[b:]
        for chunk in parse_citation_chunks(rem):
            pe.citations.append(ParsedCitation(citation_text=chunk, alias_label=None, notes=None))
    else:
        for chunk in parse_citation_chunks(body_norm):
            pe.citations.append(ParsedCitation(citation_text=chunk, alias_label=None, notes=None))

    # Deduplicate aliases (by lowercase text)
    seen = set()
    deduped = []
    for al in pe.aliases:
        key = al.alias.strip().lower()
        if not key:
            continue
        if key in seen:
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


def ensure_entity(cur, source_id: int, entry_id: Optional[int], canonical_name: str,
                  entity_type: str, confidence: Optional[str], notes: Optional[str]) -> int:
    """
    Ensures a single entity row per (source_id, canonical_name) in practice, by searching any-type first.
    Updates type from 'other' to a more specific type when seen later.
    """
    existing = find_entity_any_type(cur, source_id, canonical_name)
    if existing:
        eid, etype = existing
        # Upgrade type if current is 'other' and new is more specific
        upgrade = (etype == "other" and entity_type != "other")
        # Populate entry_id if missing
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

    # Insert new entity
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
    # Insert-if-not-exists (no uniqueness constraint)
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


def ensure_link(cur, source_id: int, entry_id: Optional[int], from_entity_id: int, to_entity_id: int, link: ParsedLink) -> int:
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


def ensure_citation(cur, source_id: int, entry_id: Optional[int],
                    entity_id: Optional[int], alias_id: Optional[int], link_id: Optional[int],
                    cit: ParsedCitation) -> int:
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
# Orchestration
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
            print(f"    - {a.alias!r} [{a.alias_type}]")
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

    args = ap.parse_args()

    # 1) Slice index body from PDF
    raw_index_text = slice_index_text_from_pdf(args.pdf, marker=args.marker, marker_page_1based=args.marker_page)

    # 2) Normalize for parsing
    norm_text = normalize_for_parsing(raw_index_text)
    parse_text = remove_footnote_blocks_for_parsing(norm_text)


    # 3) Segment into entries
    blocks = segment_entries(parse_text)
    if args.limit is not None:
        blocks = blocks[:args.limit]

    if args.verbose or args.dry_run:
        print(f"Segmented {len(blocks)} entry blocks")

    # 4) Assign deterministic entry_seq per entry_key (within this run)
    key_counts: Dict[str, int] = {}

    parsed: List[ParsedEntry] = []
    for block in blocks:
        first = block.splitlines()[0].strip()
        head_raw, _, _ = split_head_and_body(first)
        entry_key = strip_outer_quotes(_normalize_quotes(head_raw).strip())
        key_counts.setdefault(entry_key, 0)
        key_counts[entry_key] += 1
        entry_seq = key_counts[entry_key]

        pe = parse_entry_block(block, entry_seq=entry_seq)
        # Use the segmentation-derived entry_key for seq grouping consistency
        pe.entry_key = entry_key
        pe.entry_seq = entry_seq
        parsed.append(pe)

        if args.dry_run:
            print_parsed(pe)

    if args.dry_run:
        print("\n(dry-run) ✅ done")
        return

    # 5) Write to DB
    conn = get_conn(args.db_url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            source_id = ensure_source(cur, args.source_slug, args.source_title, args.source_notes)

        for pe in parsed:
            with conn.cursor() as cur:
                entry_id = upsert_entry(cur, source_id, pe.entry_key, pe.entry_seq, pe.raw_text)

                # Crossref-only entries: map entry_key as alias to target entity
                if pe.is_crossref_only:
                    if not pe.crossref_target:
                        # Still keep the entry; nothing else to do
                        continue
                    target_name = strip_outer_quotes(_normalize_quotes(pe.crossref_target).strip())
                    target_eid = ensure_entity(cur, source_id, None, target_name, "other", None, None)
                    ensure_alias(cur, source_id, entry_id, target_eid, ParsedAlias(alias=pe.entry_key, alias_type="see"))
                    continue

                # Ensure main entity for this entry
                eid = ensure_entity(cur, source_id, entry_id, pe.entity_canonical, pe.entity_type, pe.confidence, pe.notes)

                # Ensure aliases
                alias_ids_by_text: Dict[str, int] = {}
                for al in pe.aliases:
                    aid = ensure_alias(cur, source_id, entry_id, eid, al)
                    alias_ids_by_text[al.alias.lower()] = aid

                # Ensure links (create/from/to entities)
                link_ids: List[int] = []
                for lk in pe.links:
                    from_name = strip_outer_quotes(_normalize_quotes(lk.from_name).strip())
                    to_name = strip_outer_quotes(_normalize_quotes(lk.to_name).strip())

                    # Heuristic: cover-name-ish links create cover_name entities for from_name
                    if lk.link_type in ("cover_name_of", "changed_to"):
                        from_eid = ensure_entity(cur, source_id, None, from_name, "cover_name", lk.confidence, None)
                    else:
                        from_eid = ensure_entity(cur, source_id, None, from_name, "other", lk.confidence, None)

                    # to side: if it looks like a person head (comma) treat as person, else other
                    to_type = "person" if looks_like_person_head(to_name) else "other"
                    to_eid = ensure_entity(cur, source_id, None, to_name, to_type, lk.confidence, None)

                    lid = ensure_link(cur, source_id, entry_id, from_eid, to_eid, lk)
                    link_ids.append(lid)

                # Ensure citations:
                # - attach to alias if alias_label is present and we can resolve it,
                # - otherwise attach to the main entity.
                for cit in pe.citations:
                    alias_id = None
                    if cit.alias_label:
                        alias_id = alias_ids_by_text.get(cit.alias_label.lower())
                        if alias_id is None:
                            # Create the alias on the main entity if missing
                            alias_id = ensure_alias(cur, source_id, entry_id, eid, ParsedAlias(alias=cit.alias_label, alias_type="scoped_label"))
                            alias_ids_by_text[cit.alias_label.lower()] = alias_id

                    ensure_citation(
                        cur,
                        source_id=source_id,
                        entry_id=entry_id,
                        entity_id=(None if alias_id else eid),
                        alias_id=alias_id,
                        link_id=None,
                        cit=cit
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
