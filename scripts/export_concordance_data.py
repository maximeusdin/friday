#!/usr/bin/env python3
"""
Export concordance data to CSV files for examination.

Exports:
- concordance_entries.csv: All entries with their parsed data
- entry_document_pages.csv: One row per (alias_norm, canonical_name, document): each alias at that
  document/location on its own line; pages merged when the same alias appears multiple times
- entities.csv: All entities from concordance
- entity_aliases.csv: All aliases with their types
- entity_links.csv: All relationships (cover_name_of, changed_to, etc.)
- entity_citations.csv: All citations with scoped labels
- entity_mentions.csv: Extracted entity mentions with document and span information
"""

import os
import re
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn
from retrieval.surface_norm import normalize_surface

# Import citation parser to extract pages grouped by document
from concordance.validate_entity_mentions_from_citations import (
    parse_citation_text,
    parse_page_numbers,
)


def expand_page_ranges(pages: List[Tuple[int, Optional[int]]]) -> List[int]:
    """
    Expand page ranges into individual page numbers.
    
    Example: [(230, 231), (236, None), (314, None)] -> [230, 231, 236, 314]
    """
    expanded = []
    for start, end in pages:
        if end is None:
            expanded.append(start)
        else:
            # Expand range: start to end (inclusive)
            for page_num in range(start, end + 1):
                expanded.append(page_num)
    return sorted(set(expanded))  # Remove duplicates and sort


# Page list must be only digits, spaces, commas, en-dash, hyphen (reject prose)
_PAGE_LIST_RE = re.compile(r'^\d[\d\s,–\-]*$', re.UNICODE)

# Strip "See X." and "See Also X." cross-ref phrases so citation blocks are not hidden
_SEE_ALSO_RE = re.compile(r'\bSee\s+Also\s+[^.]+\.', re.IGNORECASE)
_SEE_RE = re.compile(r'\bSee\s+[^.]+\.', re.IGNORECASE)


def _strip_see_see_also(text: str) -> str:
    """Remove 'See X.' and 'See Also X.' phrases so we don't skip citation blocks that follow."""
    t = _SEE_ALSO_RE.sub(' ', text)
    t = _SEE_RE.sub(' ', t)
    return t


# Start of a document citation (Vassiliev or Venona)
_VASSILIEV_VENONA_RE = re.compile(r'(Vassiliev|Venona)', re.IGNORECASE)

# Garbage alias_norms to exclude from entry_document_pages (matches ingest_concordance_tab_aware)
_GARBAGE_ALIAS_NORMS = frozenset({
    'american', 'bureau', 'soviet', 'russian', 'department', 'states', 'united',
    'unidentified', 'unknown', 'information', 'intelligence',
    'partial', 'german', 'a', 'kgb',
})
_GARBAGE_SUBSTRINGS = ('unidentified', 'unknown', 'ussr', 'venona', 'vassiliev', 'undeciphered')


def _is_garbage_alias(alias_norm: str) -> bool:
    """Skip aliases that poison PEM (matches ingest exclusions)."""
    if not alias_norm:
        return True
    t = alias_norm.lower()
    if t in _GARBAGE_ALIAS_NORMS:
        return True
    return any(s in t for s in _GARBAGE_SUBSTRINGS)


_PHRASE_DEDUP_STOPWORDS = frozenset({"of", "the", "a", "an", "and", "or", "for", "de", "la", "le"})


def _phrase_key(alias_norm: str) -> str:
    """Normalize phrase for deduping (office of X = office X = X office)."""
    if not alias_norm:
        return ""
    words = [w for w in alias_norm.lower().split() if w not in _PHRASE_DEDUP_STOPWORDS]
    return " ".join(sorted(words)) if words else ""


def _is_boring_alias(alias_norm: str, canonical_norm: str) -> bool:
    """Skip case-only (Jacob=JACOB) or phrase-equivalent (office X = office of X) variants."""
    if not alias_norm or not canonical_norm:
        return False
    if alias_norm == canonical_norm:
        return True  # case-only variant
    if len(alias_norm.split()) > 1 and _phrase_key(alias_norm) == _phrase_key(canonical_norm):
        return True  # phrase-equivalent
    return False


def _extract_one_document_block(block: str) -> Optional[Tuple[str, List[int]]]:
    """Parse a single block that starts with Vassiliev or Venona: document title and page list."""
    comma_pos = block.find(',')
    if comma_pos < 0:
        return None
    document_title = block[:comma_pos].strip()
    after_comma = block[comma_pos + 1:].strip()
    # Truncate at semicolon so we only take this document's page list
    if ';' in after_comma:
        after_comma = after_comma.split(';')[0].strip()
    if after_comma.endswith('.'):
        after_comma = after_comma[:-1].strip()
    if not after_comma or not _PAGE_LIST_RE.match(after_comma):
        return None
    page_tuples = parse_page_numbers(after_comma)
    expanded = expand_page_ranges(page_tuples)
    if not expanded:
        return None
    return (document_title, expanded)


def _split_on_vassiliev_venona(block: str) -> List[str]:
    """
    Split block on each occurrence of Vassiliev or Venona so each segment starts with
    a document citation. Returns list of sub-blocks (each starting with Vassiliev or Venona).
    """
    sub_blocks = []
    for m in _VASSILIEV_VENONA_RE.finditer(block):
        start = m.start()
        sub = block[start:].strip()
        if sub:
            sub_blocks.append(sub)
    return sub_blocks


def parse_documents_and_pages_from_raw_text(raw_text: str) -> List[Tuple[str, List[int]]]:
    """
    Extract every source document and page list from raw_text.
    
    - Strips "See X." and "See Also X." so citations after cross-refs are not missed.
    - Splits on semicolons; for each block, splits again on every Vassiliev/Venona so
      each occurrence yields a clean document title and page list (one row per document).
    - Validates page list (digits/ranges only). Ranges (e.g. 66–67, 103–4) are expanded.
    """
    if not raw_text:
        return []
    
    # Normalize: single line, collapse spaces; strip See / See Also cross-ref phrases
    text = re.sub(r'\s+', ' ', raw_text.strip())
    text = _strip_see_see_also(text)
    
    result = []
    for block in text.split(';'):
        block = block.strip()
        if not block:
            continue
        # Split on every Vassiliev/Venona so each occurrence can be a clean document row
        for sub_block in _split_on_vassiliev_venona(block):
            one = _extract_one_document_block(sub_block)
            if one:
                doc_title, pages = one
                # Keep only if title is "clean" (no second Vassiliev/Venona in the title)
                if len(_VASSILIEV_VENONA_RE.findall(doc_title)) <= 1:
                    result.append(one)
    
    return result


def parse_citation_pages_by_document(citation_text: str) -> List[Tuple[str, List[int]]]:
    """
    Parse citation text and return list of (document_name, page_list) tuples
    in the order they appear in the citation text.
    
    Example:
        "Vassiliev Black Notebook, 51–54, 58; Venona New York KGB 1943, 26, 28"
        -> [("Vassiliev Black Notebook", [51, 52, 53, 54, 58]), 
            ("Venona New York KGB 1943", [26, 28])]
    """
    if not citation_text:
        return []
    
    # Parse citation text to get locations
    citation_locations = parse_citation_text(citation_text)
    
    # Group pages by document (source)
    result = []
    for loc in citation_locations:
        # Expand ranges to individual pages
        expanded_pages = expand_page_ranges(loc.pages)
        if expanded_pages:
            result.append((loc.source, expanded_pages))
    
    return result


def export_concordance_entries(output_dir: str, source_slug: Optional[str] = None, conn=None):
    """Export concordance entries to CSV."""
    if conn is None:
        conn = get_conn()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with conn.cursor() as cur:
        # Export concordance_entries
        query = """
            SELECT 
                ce.id,
                cs.slug AS source_slug,
                cs.title AS source_title,
                ce.entry_key,
                ce.entry_seq,
                ce.raw_text
            FROM concordance_entries ce
            JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, ce.entry_seq"
        
        cur.execute(query, params)
        entries_rows = cur.fetchall()
        
        with open(output_dir / "concordance_entries.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'source_slug', 'source_title', 'entry_key', 'entry_seq', 'raw_text'])
            for row in entries_rows:
                writer.writerow(row)
        
        print(f"Exported concordance_entries to {output_dir / 'concordance_entries.csv'}", file=sys.stderr)
        
        # Entry -> list of (alias_norm, canonical_name, alias_type) for this source
        # Crossrefs ("see"): ingest attaches entry_key as alias to the *target* entity (alias_type="see").
        # So crossref entries have no entity of their own; their only alias points to the target.
        entry_aliases_query = """
            SELECT ce.id, ea.alias_norm, e.canonical_name, ea.alias_type
            FROM concordance_entries ce
            JOIN concordance_sources cs ON ce.source_id = cs.id
            JOIN entity_aliases ea ON ea.entry_id = ce.id
            JOIN entities e ON e.id = ea.entity_id
        """
        if source_slug:
            entry_aliases_query += " WHERE cs.slug = %s"
        entry_aliases_query += " ORDER BY ce.id"
        cur.execute(entry_aliases_query, [source_slug] if source_slug else [])
        entry_to_aliases = defaultdict(list)
        see_aliases = set()  # (alias_norm, canonical_norm) that are cross-refs
        for entry_id, alias_norm, canonical_name, alias_type in cur.fetchall():
            if alias_norm and canonical_name is not None:
                canonical_norm = normalize_surface(canonical_name) if canonical_name else ""
                entry_to_aliases[entry_id].append((alias_norm, canonical_name, alias_type))
                if alias_type == "see":
                    see_aliases.add((alias_norm, canonical_norm))
        
        # Export entry_document_pages: one row per (alias_norm, canonical_name, document_norm); clean names only
        grouped = defaultdict(lambda: {"document_title": None, "canonical_name": None, "pages": []})
        for row in entries_rows:
            entry_id, source_slug, source_title, entry_key, entry_seq, raw_text = row
            aliases = entry_to_aliases.get(entry_id)
            if not aliases:
                continue
            doc_pages_list = parse_documents_and_pages_from_raw_text(raw_text or '')
            for document_title, pages in doc_pages_list:
                document_norm = normalize_surface(document_title) if document_title else ""
                if not document_norm:
                    continue
                for alias_norm, canonical_name, _alias_type in aliases:
                    if _is_garbage_alias(alias_norm):
                        continue
                    canonical_norm = normalize_surface(canonical_name) if canonical_name else ""
                    # Collapse phrase-equivalent aliases (office X = office of X = X office) into canonical form
                    effective_alias = canonical_norm if _is_boring_alias(alias_norm, canonical_norm) else alias_norm
                    key = (effective_alias, canonical_norm, document_norm)
                    if grouped[key]["document_title"] is None:
                        grouped[key]["document_title"] = document_title
                        grouped[key]["canonical_name"] = canonical_name
                    grouped[key]["pages"].extend(pages)
        # Propagate target entity's document/pages to see-aliases (cross-refs have no blocks in raw_text)
        canonical_to_docs = defaultdict(list)
        for (alias_norm, canonical_norm, document_norm), data in grouped.items():
            pages_list = sorted(set(data["pages"]))
            canonical_to_docs[canonical_norm].append((
                document_norm,
                data["document_title"],
                data["canonical_name"],
                pages_list,
            ))
        for (alias_norm, canonical_norm) in see_aliases:
            if _is_garbage_alias(alias_norm):
                continue
            effective_alias = canonical_norm if _is_boring_alias(alias_norm, canonical_norm) else alias_norm
            for document_norm, document_title, canonical_name, pages_list in canonical_to_docs.get(canonical_norm, []):
                key = (effective_alias, canonical_norm, document_norm)
                if grouped[key]["document_title"] is None:
                    grouped[key]["document_title"] = document_title
                    grouped[key]["canonical_name"] = canonical_name
                grouped[key]["pages"].extend(pages_list)
        # One row per (alias_norm, canonical_name, document): each alias at that location gets its own line
        with open(output_dir / "entry_document_pages.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['alias_norm', 'canonical_name', 'document_title', 'document_norm', 'pages'])
            for (alias_norm, canonical_norm, document_norm), data in sorted(grouped.items()):
                if _is_garbage_alias(alias_norm):
                    continue
                pages_sorted = sorted(set(data["pages"]))
                pages_str = ','.join(str(p) for p in pages_sorted)
                writer.writerow([
                    alias_norm,
                    data["canonical_name"],
                    data["document_title"],
                    document_norm,
                    pages_str,
                ])
        print(f"Exported entry_document_pages to {output_dir / 'entry_document_pages.csv'}", file=sys.stderr)
        
        # Export entities (from concordance sources)
        query = """
            SELECT DISTINCT
                e.id,
                e.entity_type,
                e.canonical_name,
                e.confidence,
                e.notes,
                cs.slug AS source_slug
            FROM entities e
            JOIN entity_aliases ea ON ea.entity_id = e.id
            JOIN concordance_entries ce ON ce.id = ea.entry_id
            JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, e.id"
        
        cur.execute(query, params)
        
        with open(output_dir / "entities.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'entity_type', 'canonical_name', 'confidence', 'notes', 'source_slug'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported entities to {output_dir / 'entities.csv'}", file=sys.stderr)
        
        # Export entity_aliases
        query = """
            SELECT 
                ea.id,
                ea.entity_id,
                e.canonical_name,
                ea.alias,
                ea.alias_norm,
                ea.alias_type,
                ea.confidence,
                ea.notes,
                cs.slug AS source_slug,
                ce.entry_key
            FROM entity_aliases ea
            JOIN entities e ON ea.entity_id = e.id
            LEFT JOIN concordance_entries ce ON ce.id = ea.entry_id
            LEFT JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, ea.entity_id, ea.id"
        
        cur.execute(query, params)
        
        with open(output_dir / "entity_aliases.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'entity_id', 'canonical_name', 'alias', 'alias_norm', 'alias_type', 'confidence', 'notes', 'source_slug', 'entry_key'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported entity_aliases to {output_dir / 'entity_aliases.csv'}", file=sys.stderr)
        
        # Export entity_links
        query = """
            SELECT 
                el.id,
                el.from_entity_id,
                e1.canonical_name AS from_name,
                el.to_entity_id,
                e2.canonical_name AS to_name,
                el.link_type,
                el.confidence,
                el.notes,
                cs.slug AS source_slug,
                ce.entry_key
            FROM entity_links el
            JOIN entities e1 ON el.from_entity_id = e1.id
            JOIN entities e2 ON el.to_entity_id = e2.id
            LEFT JOIN concordance_entries ce ON ce.id = el.entry_id
            LEFT JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, el.id"
        
        cur.execute(query, params)
        
        with open(output_dir / "entity_links.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'from_entity_id', 'from_name', 'to_entity_id', 'to_name', 'link_type', 'confidence', 'notes', 'source_slug', 'entry_key'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported entity_links to {output_dir / 'entity_links.csv'}", file=sys.stderr)
        
        # Export entity_citations
        query = """
            SELECT 
                ec.id,
                ec.entity_id,
                e.canonical_name,
                ec.citation_text,
                ea.alias AS alias_label,
                ec.collection_slug,
                ec.document_label,
                ec.page_list,
                ec.notes,
                cs.slug AS source_slug,
                ce.entry_key
            FROM entity_citations ec
            JOIN entities e ON ec.entity_id = e.id
            LEFT JOIN entity_aliases ea ON ec.alias_id = ea.id
            LEFT JOIN concordance_entries ce ON ce.id = ec.entry_id
            LEFT JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, ec.entity_id, ec.id"
        
        cur.execute(query, params)
        
        with open(output_dir / "entity_citations.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'entity_id', 'canonical_name', 'citation_text', 'alias_label', 'collection_slug', 'document_label', 'page_list', 'notes', 'source_slug', 'entry_key'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported entity_citations to {output_dir / 'entity_citations.csv'}", file=sys.stderr)
        
        # Export entity_mentions (extracted mentions with document and span info)
        # Include concordance source information and citation information
        query = """
            SELECT 
                -- Use MIN(em.id) to pick one mention ID when deduplicating by entity_id + chunk_id
                MIN(em.id) AS id,
                em.entity_id,
                e.canonical_name,
                e.entity_type,
                em.chunk_id,
                em.document_id,
                d.source_name AS document_name,
                c.slug AS collection_slug,
                c.title AS collection_title,
                -- Aggregate multiple mentions in same chunk: take first surface, max confidence, earliest created_at
                MIN(em.surface) AS surface,
                MIN(em.surface_norm) AS surface_norm,
                MIN(em.surface_quality) AS surface_quality,
                MIN(em.start_char) AS start_char,
                MIN(em.end_char) AS end_char,
                MAX(em.confidence) AS confidence,
                -- Take the first method (alphabetically) if multiple
                MIN(em.method) AS method,
                MIN(em.created_at) AS created_at,
                -- Get page information from chunk_metadata (same for all mentions in chunk)
                cm.first_page_id,
                p.pdf_page_number,
                p.logical_page_label,
                -- Get concordance source information (same for all mentions of same entity)
                cs_concordance.slug AS concordance_source_slug,
                cs_concordance.title AS concordance_source_title,
                ce_concordance.entry_key AS concordance_entry_key,
                ce_concordance.entry_seq AS concordance_entry_seq,
                -- Get citation information (aggregate all citations for this entity)
                -- Show all citation texts and document labels
                -- Note: citation_page_lists will be computed in Python from citation_texts to match order
                STRING_AGG(DISTINCT ec_citation.citation_text, ' | ' ORDER BY ec_citation.citation_text) AS citation_texts,
                STRING_AGG(DISTINCT ec_citation.document_label, ' | ' ORDER BY ec_citation.document_label) AS citation_document_labels
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.id
            JOIN documents d ON em.document_id = d.id
            JOIN collections c ON d.collection_id = c.id
            LEFT JOIN chunk_metadata cm ON em.chunk_id = cm.chunk_id
            LEFT JOIN pages p ON cm.first_page_id = p.id
            -- Join to get concordance source info through entity_aliases
            LEFT JOIN (
                SELECT DISTINCT ON (ea.entity_id)
                    ea.entity_id,
                    ce.id AS entry_id,
                    ce.entry_key,
                    ce.entry_seq,
                    cs.id AS source_id,
                    cs.slug AS source_slug,
                    cs.title AS source_title
                FROM entity_aliases ea
                JOIN concordance_entries ce ON ce.id = ea.entry_id
                JOIN concordance_sources cs ON ce.source_id = cs.id
                WHERE ea.entry_id IS NOT NULL
                ORDER BY ea.entity_id, ea.id
            ) ea_concordance ON ea_concordance.entity_id = e.id
            LEFT JOIN concordance_entries ce_concordance ON ce_concordance.id = ea_concordance.entry_id
            LEFT JOIN concordance_sources cs_concordance ON cs_concordance.id = ea_concordance.source_id
            -- Join to get citation information for this entity
            LEFT JOIN entity_citations ec_citation ON ec_citation.entity_id = e.id
            GROUP BY 
                em.entity_id, e.canonical_name, e.entity_type,
                em.chunk_id, em.document_id, d.source_name,
                c.slug, c.title,
                cm.first_page_id, p.pdf_page_number, p.logical_page_label,
                cs_concordance.slug, cs_concordance.title,
                ce_concordance.entry_key, ce_concordance.entry_seq
            ORDER BY em.entity_id, em.document_id, em.chunk_id
        """
        
        # Add WHERE clause if source_slug is specified
        # Filter to only include entity_mentions for entities that have aliases from the specified source
        if source_slug:
            # Insert WHERE clause before GROUP BY
            query = query.replace(
                "LEFT JOIN entity_citations ec_citation ON ec_citation.entity_id = e.id",
                """LEFT JOIN entity_citations ec_citation ON ec_citation.entity_id = e.id
            WHERE EXISTS (
                SELECT 1 FROM entity_aliases ea_filter
                JOIN concordance_entries ce_filter ON ce_filter.id = ea_filter.entry_id
                JOIN concordance_sources cs_filter ON cs_filter.id = ce_filter.source_id
                WHERE ea_filter.entity_id = e.id AND cs_filter.slug = %s
            )"""
            )
            params = [source_slug]
        else:
            params = []
        
        cur.execute(query, params)
        
        with open(output_dir / "entity_mentions.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'entity_id', 'canonical_name', 'entity_type',
                'chunk_id', 'document_id', 'document_name',
                'collection_slug', 'collection_title',
                'surface', 'surface_norm', 'surface_quality',
                'start_char', 'end_char',
                'confidence', 'method', 'created_at',
                'first_page_id', 'pdf_page_number', 'logical_page_label',
                'concordance_source_slug', 'concordance_source_title',
                'concordance_entry_key', 'concordance_entry_seq',
                'citation_texts', 'citation_document_labels', 'citation_page_lists'
            ])
            
            for row in cur.fetchall():
                # Parse citation_texts to extract page lists grouped by document
                # citation_texts is second-to-last column, citation_document_labels is last
                row_list = list(row)
                citation_texts = row_list[-2] if len(row_list) >= 2 else None
                citation_page_lists = None
                
                if citation_texts:
                    # Parse each citation text and extract pages grouped by document
                    citation_text_parts = citation_texts.split(' | ')
                    page_list_parts = []
                    
                    for citation_text_part in citation_text_parts:
                        # Parse this citation text to get pages by document
                        doc_pages = parse_citation_pages_by_document(citation_text_part)
                        
                        # Format as "{page1,page2,page3}" for each document
                        formatted_pages = []
                        for doc_name, pages in doc_pages:
                            if pages:
                                pages_str = ','.join(str(p) for p in sorted(pages))
                                formatted_pages.append(f"{{{pages_str}}}")
                        
                        if formatted_pages:
                            page_list_parts.append(' | '.join(formatted_pages))
                    
                    if page_list_parts:
                        citation_page_lists = ' | '.join(page_list_parts)
                
                # Insert citation_page_lists between citation_texts and citation_document_labels
                # row_list structure: [...citation_texts, citation_document_labels]
                # We want: [...citation_texts, citation_page_lists, citation_document_labels]
                row_list.insert(-1, citation_page_lists)
                writer.writerow(row_list)
        
        print(f"Exported entity_mentions to {output_dir / 'entity_mentions.csv'}", file=sys.stderr)
        
        # Print summary
        if source_slug:
            cur.execute("""
                SELECT COUNT(*) FROM concordance_entries ce
                JOIN concordance_sources cs ON ce.source_id = cs.id
                WHERE cs.slug = %s
            """, (source_slug,))
            entry_count = cur.fetchone()[0]
            print(f"\nSummary for source '{source_slug}':", file=sys.stderr)
            print(f"  Entries: {entry_count}", file=sys.stderr)


def get_most_recent_source_slug(conn) -> Optional[str]:
    """Get the most recent concordance source slug that has at least one entry."""
    with conn.cursor() as cur:
        # Only consider sources that have entries (ignore empty placeholder sources)
        cur.execute("""
            SELECT cs.slug
            FROM concordance_sources cs
            WHERE EXISTS (SELECT 1 FROM concordance_entries ce WHERE ce.source_id = cs.id)
            ORDER BY cs.created_at DESC NULLS LAST, cs.id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        return row[0] if row else None


def main():
    parser = argparse.ArgumentParser(description="Export concordance data to CSV files")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="concordance_export",
        help="Output directory for CSV files (default: concordance_export)",
    )
    parser.add_argument(
        "--source-slug",
        "-s",
        default=None,
        help="Filter by source slug (e.g., 'vassiliev_venona_index_full_capitalized'). If not provided, uses the most recent source slug.",
    )
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        # Auto-detect most recent source slug if not provided
        source_slug = args.source_slug
        if not source_slug:
            source_slug = get_most_recent_source_slug(conn)
            if not source_slug:
                print("ERROR: No source slug provided and no sources found in database.", file=sys.stderr)
                sys.exit(1)
            print(f"Using most recent source slug: {source_slug}", file=sys.stderr)
        
        export_concordance_entries(args.output_dir, source_slug, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
