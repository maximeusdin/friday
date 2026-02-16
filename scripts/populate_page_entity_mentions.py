#!/usr/bin/env python3
"""
Populate page_entity_mentions — V10.2 page-level mention facts.

This script builds the single truth substrate for "surface S on page P refers
to entity E in collection C."  It populates only from authoritative sources:

1. **alias_referent_rules**: Manual rules mapping (collection, alias, document, pages) → entity_id.
2. **Concordance**: entry_document_pages.csv (alias_norm, canonical_name, document, pages)
   from entity_aliases. Uses canonical_name to disambiguate (e.g. cabin → OSS vs cabin → Cabin).

No derived data from entity_mentions — PEM is concordance-only to avoid pollution
from OCR fuzzy matches (e.g. "bureau", "american" linked via substring matches).

After population it:
  - Updates the stored pipeline version (app_kv).
  - Runs an index health acceptance check (coverage %, alignment %).
  - Logs a summary.

Usage:
    python scripts/populate_page_entity_mentions.py
    python scripts/populate_page_entity_mentions.py --truncate   # wipe and rebuild
    python scripts/populate_page_entity_mentions.py --dry-run    # preview counts only
    python scripts/populate_page_entity_mentions.py --run-cleanup-apply cleanup_session_vassiliev_venona_index_20260210.json  # apply merges/deletes from JSON only (no coded garbage)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn
from retrieval.agent.v10_normalize import normalize_surface_for_lookup
from retrieval.agent.v10_page_bridge import set_index_revision

# Reuse concordance document matching helpers
from concordance.validate_entity_mentions_from_citations import (
    normalize_document_name,
    build_citation_to_document_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("populate_pem")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Surface norms that must never appear in PEM (collection slugs / doc-type tokens, not entities)
BLOCKED_SURFACE_NORMS = frozenset({"huac", "fbicomrap"})

# Venona: concordance document titles sometimes differ from PDF filenames.
# Maps norm(concordance_title) -> norm(pdf_filename) for fallback lookup.
VENONA_DOCUMENT_ALIASES: Dict[str, str] = {
    "kgblondon": "londonkgb",
    "newyork1945": "newyorkkgb1945",
    "secretwritingsnewyorkbuenosaires": "newyorkbuenosairessecretwritings",
    # Volume/edition suffixes: "Venona New York KGB 1944 53" -> base doc
    "newyorkkgb531944": "newyorkkgb1944",
    "newyorkkgb81945": "newyorkkgb1945",
    "newyorkkgb891945": "newyorkkgb1945",
    # Malformed year range "1941– 42" (space after en-dash) normalizes wrong
    "newyorkkgb421941": "newyorkkgb1941-1942",
    # Volume number: "Venona USA GRU 40" -> "Venona USA GRU"
    "usagru40": "usagru",
    # Sub-citation "Venona: KOROBOV..." -> New York KGB 1944 (page 269-70)
    "venonakorobovasostrovskij26970": "newyorkkgb1944",
}

# Index health thresholds
COVERAGE_WARN_THRESHOLD = 0.95
ALIGNMENT_WARN_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Step 1: Authoritative rows from alias_referent_rules
# ---------------------------------------------------------------------------

def populate_from_alias_referent_rules(
    conn,
    pipeline_version: str,
    dry_run: bool = False,
) -> int:
    """Insert authoritative rows from alias_referent_rules.

    Each rule maps (collection_slug, alias_text, document_id, page_from..page_to)
    to entity_id.  We expand to individual page_ids.
    """
    count = 0
    try:
        with conn.cursor() as cur:
            logger.info("Loading alias_referent_rules...")
            cur.execute("""
                SELECT id, collection_slug, alias_text, document_id,
                       page_from, page_to, entity_id, status
                FROM alias_referent_rules
                WHERE status = 'confirmed'
            """)
            rules = cur.fetchall()
            logger.info("  Processing %d rules...", len(rules))
            
            batch = []
            batch_size = 1000

            for idx, (rule_id, coll, alias_text, doc_id, pg_from, pg_to, entity_id, status) in enumerate(rules):
                if idx % 100 == 0 and idx > 0:
                    logger.info("  Processed %d/%d rules (%d rows queued)...", 
                              idx, len(rules), len(batch))
                surface_norm = normalize_surface_for_lookup(alias_text)
                if not surface_norm or surface_norm in BLOCKED_SURFACE_NORMS:
                    continue

                # Get pages for this document within the rule's range
                if pg_from is not None:
                    cur.execute("""
                        SELECT id FROM pages
                        WHERE document_id = %s
                          AND pdf_page_number >= %s
                          AND pdf_page_number <= %s
                    """, (doc_id, pg_from, pg_to or pg_from))
                else:
                    # doc-wide rule: all pages
                    cur.execute("""
                        SELECT id FROM pages WHERE document_id = %s
                    """, (doc_id,))

                page_ids = [r[0] for r in cur.fetchall()]
                if not page_ids:
                    continue

                for page_id in page_ids:
                    if dry_run:
                        count += 1
                    else:
                        batch.append((
                            coll, doc_id, page_id, surface_norm, alias_text,
                            entity_id, pipeline_version
                        ))
                        
                        # Flush batch using COPY
                        if len(batch) >= batch_size:
                            from io import StringIO
                            copy_buffer = StringIO()
                            for row in batch:
                                copy_buffer.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\tauthoritative\t{row[6]}\talias_referent_rules\n")
                            copy_buffer.seek(0)
                            cur.copy_from(copy_buffer, 'page_entity_mentions',
                                        columns=['collection_slug', 'document_id', 'page_id', 'surface_norm', 'surface_raw', 'entity_id', 'truth_level', 'pipeline_version', 'source'])
                            count += len(batch)
                            conn.commit()
                            logger.info("  Flushed %d rows (total: %d)", len(batch), count)
                            batch = []

            # Flush remaining batch
            if batch and not dry_run:
                from io import StringIO
                copy_buffer = StringIO()
                for row in batch:
                    copy_buffer.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\tauthoritative\t{row[6]}\talias_referent_rules\n")
                copy_buffer.seek(0)
                cur.copy_from(copy_buffer, 'page_entity_mentions',
                            columns=['collection_slug', 'document_id', 'page_id', 'surface_norm', 'surface_raw', 'entity_id', 'truth_level', 'pipeline_version', 'source'])
                count += len(batch)
                conn.commit()
                logger.info("  Final flush: %d rows (total: %d)", len(batch), count)
    except Exception as e:
        logger.error("populate_from_alias_referent_rules failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    logger.info("Authoritative (alias_referent_rules): %d rows %s",
                count, "(dry run)" if dry_run else "inserted/updated")
    return count


# ---------------------------------------------------------------------------
# Step 2: Authoritative rows from concordance entries
# ---------------------------------------------------------------------------

def populate_from_concordance(
    conn,
    pipeline_version: str,
    dry_run: bool = False,
) -> int:
    """Insert authoritative rows from concordance via entry_document_pages.csv.

    Uses the pre-parsed concordance export CSV which already has document+page
    mappings extracted. Much faster than re-parsing raw_text.
    """
    import csv
    count = 0
    unresolved_docs: Counter = Counter()
    unresolved_pages: Counter = Counter()
    
    # Find the concordance export CSV
    csv_path = REPO_ROOT / "concordance_export" / "entry_document_pages.csv"
    if not csv_path.exists():
        logger.warning("concordance_export/entry_document_pages.csv not found, skipping concordance")
        return 0

    try:
        with conn.cursor() as cur:
            logger.info("Building document maps...")
            doc_map_venona = build_citation_to_document_map(cur, "venona")
            doc_map_vassiliev = build_citation_to_document_map(cur, "vassiliev")
            logger.info("  Venona: %d normalized keys", len(doc_map_venona))
            logger.info("  Vassiliev: %d normalized keys", len(doc_map_vassiliev))
            if len(doc_map_venona) == 0:
                logger.warning("Venona doc_map is empty. Ensure Venona PDFs are ingested (scripts/ingest_venona_pdf.py).")

            # Build page map (only for venona and vassiliev)
            # - page_map: (doc_id, pdf_page_number) -> page_id (exact match: message start page)
            # - page_map_logical: (doc_id, int(logical_label)) -> page_id (for Vassiliev p.xx)
            # - page_map_venona_span: (doc_id, pdf_page_num) -> page_id for ANY pdf page in a message's span.
            #   Venona pages are message-level: pdf_page_number = start of message. Messages span
            #   multiple PDF pages. Concordance cites physical PDF pages. We must map continuation
            #   pages (e.g. page 758) to the message that contains them.
            logger.info("Building page lookup map...")
            cur.execute("""
                SELECT p.document_id, p.pdf_page_number, p.logical_page_label, p.id, c.slug
                FROM pages p
                JOIN documents d ON d.id = p.document_id
                JOIN collections c ON c.id = d.collection_id
                WHERE c.slug IN ('venona', 'vassiliev')
            """)
            page_map = {}
            page_map_logical = {}
            # Venona: per doc, pages ordered by pdf_page_number; each message spans to next start - 1
            venona_pages_by_doc: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for doc_id, pdf_page_num, logical_label, page_id, coll_slug in cur.fetchall():
                if pdf_page_num is not None:
                    page_map[(doc_id, int(pdf_page_num))] = page_id
                if logical_label:
                    try:
                        page_map_logical[(doc_id, int(logical_label))] = page_id
                    except (ValueError, TypeError):
                        pass
                if coll_slug == "venona" and pdf_page_num is not None:
                    venona_pages_by_doc[doc_id].append((int(pdf_page_num), page_id))

            # Build page_map_venona_span: for each PDF page in [start, next_start), map to page_id
            page_map_venona_span: Dict[Tuple[int, int], int] = {}
            for doc_id, entries in venona_pages_by_doc.items():
                entries.sort(key=lambda x: x[0])
                for i, (start_pdf, page_id) in enumerate(entries):
                    end_pdf = entries[i + 1][0] - 1 if i + 1 < len(entries) else start_pdf + 10000
                    for pdf_page in range(start_pdf, end_pdf + 1):
                        page_map_venona_span[(doc_id, pdf_page)] = page_id

            logger.info("  Loaded %d pages by pdf_page_number, %d by logical_page_label, %d Venona span mappings",
                       len(page_map), len(page_map_logical), len(page_map_venona_span))

            # Build canonical_name -> entity_id map (primary: CSV has canonical_name)
            logger.info("Building canonical_name -> entity_id map...")
            cur.execute("""
                SELECT id, canonical_name FROM entities
            """)
            canonical_to_entity: Dict[str, int] = {}
            for entity_id, name in cur.fetchall():
                if name:
                    key = name.strip().lower()
                    if key not in canonical_to_entity:
                        canonical_to_entity[key] = entity_id
            logger.info("  Loaded %d entities by canonical_name", len(canonical_to_entity))

            # Build alias -> entity_ids (fallback when canonical_name not found)
            logger.info("Building alias -> entity_id fallback map...")
            cur.execute("""
                SELECT DISTINCT ea.alias_norm, ea.entity_id
                FROM entity_aliases ea
                WHERE ea.entity_id IS NOT NULL
            """)
            alias_to_entity: Dict[str, List[int]] = {}
            for alias_norm, entity_id in cur.fetchall():
                if alias_norm not in alias_to_entity:
                    alias_to_entity[alias_norm] = []
                alias_to_entity[alias_norm].append(entity_id)
            logger.info("  Loaded %d unique aliases", len(alias_to_entity))

            # Process CSV
            logger.info("Processing entry_document_pages.csv...")
            total_rows = 0
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Count total for progress
                total_rows = sum(1 for _ in f) - 1  # subtract header
                f.seek(0)
                
                reader = csv.DictReader(f)
                logger.info("  Total CSV rows: %d", total_rows)

                # Use dict to deduplicate by (page_id, surface_norm, entity_id)
                # This ensures no duplicates in COPY batch
                batch_dict = {}
                all_keys_seen = set()  # Track across ALL flushes
                batch_size = 10000  # Large batches for speed
                skipped_no_entity = 0
                skipped_no_doc = 0
                skipped_no_pages = 0
                
                for idx, row in enumerate(reader):
                    if idx == 0:
                        logger.info("  Starting CSV processing...")
                    if idx % 1000 == 0 and idx > 0:
                        logger.info("  Processed %d/%d CSV rows (%d PEM rows queued, %d skipped: no_entity=%d no_doc=%d no_pages=%d)...", 
                                  idx, total_rows, len(batch_dict), 
                                  skipped_no_entity + skipped_no_doc + skipped_no_pages,
                                  skipped_no_entity, skipped_no_doc, skipped_no_pages)
                    
                    alias_norm = row['alias_norm']
                    canonical_name = row['canonical_name']
                    document_title = row['document_title']
                    pages_str = row['pages']

                    if alias_norm in BLOCKED_SURFACE_NORMS:
                        continue
                    # Skip boring case-only variants (gnome->GNOME)
                    canonical_key = canonical_name.strip().lower() if canonical_name else ""
                    if canonical_key and alias_norm == canonical_key:
                        continue

                    # Parse page numbers (comma-separated)
                    try:
                        page_nums = [int(p.strip()) for p in pages_str.split(',') if p.strip()]
                    except ValueError:
                        continue
                    
                    if not page_nums:
                        continue
                    
                    # Resolve entity_id: prefer canonical_name (CSV disambiguates cabin→OSS vs cabin→Cabin)
                    entity_id = canonical_to_entity.get(canonical_key)
                    if entity_id is None:
                        # Fallback: alias_norm only (ambiguous if multiple entities share alias)
                        entity_ids = alias_to_entity.get(alias_norm, [])
                        if not entity_ids:
                            skipped_no_entity += 1
                            continue
                        entity_id = entity_ids[0]
                    
                    # Determine collection and doc_map
                    doc_title_lower = document_title.lower()
                    if doc_title_lower.startswith("venona"):
                        collection_slug = "venona"
                        doc_map = doc_map_venona
                    elif doc_title_lower.startswith("vassiliev"):
                        collection_slug = "vassiliev"
                        doc_map = doc_map_vassiliev
                    else:
                        continue
                    
                    # Resolve document
                    norm_doc = normalize_document_name(document_title)
                    matches = doc_map.get(norm_doc, [])
                    if not matches and collection_slug == "venona":
                        alias = VENONA_DOCUMENT_ALIASES.get(norm_doc)
                        if alias:
                            matches = doc_map.get(alias, [])
                    # Fallback: doc_map may have keys where norm_doc is substring (e.g. specialstudies)
                    # or concordance uses different naming than PDF
                    if not matches and collection_slug == "venona" and norm_doc:
                        for k, v in doc_map.items():
                            if k == norm_doc:
                                matches = v
                                break
                            # norm_doc as prefix of key (e.g. specialstudies in specialstudiespdf...)
                            if k.startswith(norm_doc) and len(k) < len(norm_doc) + 100:
                                matches = v
                                break
                            # key as prefix of norm_doc (e.g. specialstudies when key is specialstudies)
                            if norm_doc.startswith(k) and len(norm_doc) < len(k) + 50:
                                matches = v
                                break
                    if not matches:
                        unresolved_docs[document_title] += 1
                        skipped_no_doc += 1
                        continue
                    doc_id, doc_name = matches[0]
                    
                    # Resolve pages and queue
                    for page_num in page_nums:
                        # Try pdf_page_number first, then logical_page_label
                        page_id = page_map.get((doc_id, page_num))
                        if not page_id:
                            page_id = page_map_logical.get((doc_id, page_num))
                        # Venona: messages span multiple PDF pages; concordance cites any page in span
                        if not page_id and collection_slug == "venona":
                            page_id = page_map_venona_span.get((doc_id, page_num))
                        
                        if not page_id:
                            unresolved_pages[(document_title, page_num)] += 1
                            skipped_no_pages += 1
                            continue
                        
                        if dry_run:
                            count += 1
                        else:
                            # Deduplicate using key (page_id, surface_norm, entity_id)
                            row_key = (page_id, alias_norm, entity_id)
                            if row_key in all_keys_seen:
                                continue
                            all_keys_seen.add(row_key)
                            batch_dict[row_key] = (
                                collection_slug, doc_id, page_id, alias_norm,
                                alias_norm, entity_id, pipeline_version
                            )
                            
                            # Flush batch using COPY (much faster than INSERT)
                            if len(batch_dict) >= batch_size:
                                logger.info("  Flushing batch of %d rows via COPY (CSV row %d/%d)...", 
                                          len(batch_dict), idx, total_rows)
                                from io import StringIO
                                copy_buffer = StringIO()
                                for row in batch_dict.values():
                                    # Format: collection_slug, document_id, page_id, surface_norm, surface_raw, entity_id, truth_level, pipeline_version, source
                                    copy_buffer.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\tauthoritative\t{row[6]}\tconcordance\n")
                                copy_buffer.seek(0)
                                
                                cur.copy_from(
                                    copy_buffer,
                                    'page_entity_mentions',
                                    columns=['collection_slug', 'document_id', 'page_id', 'surface_norm', 'surface_raw', 'entity_id', 'truth_level', 'pipeline_version', 'source']
                                )
                                count += len(batch_dict)
                                logger.info("  Committing...")
                                conn.commit()
                                logger.info("  Flushed %d rows (total: %d)", len(batch_dict), count)
                                batch_dict = {}

            # Flush remaining batch
            logger.info("  CSV loop complete, flushing final batch...")
            if batch_dict and not dry_run:
                logger.info("  Final batch: %d rows", len(batch_dict))
                from io import StringIO
                copy_buffer = StringIO()
                for row in batch_dict.values():
                    copy_buffer.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\tauthoritative\t{row[6]}\tconcordance\n")
                copy_buffer.seek(0)
                
                cur.copy_from(
                    copy_buffer,
                    'page_entity_mentions',
                    columns=['collection_slug', 'document_id', 'page_id', 'surface_norm', 'surface_raw', 'entity_id', 'truth_level', 'pipeline_version', 'source']
                )
                count += len(batch_dict)
                logger.info("  Committing final batch...")
                conn.commit()
                logger.info("  Final flush complete: %d rows (total: %d)", len(batch_dict), count)
            
            logger.info("  CSV processing complete. Total skipped: %d (no_entity=%d, no_doc=%d, no_pages=%d)",
                       skipped_no_entity + skipped_no_doc + skipped_no_pages,
                       skipped_no_entity, skipped_no_doc, skipped_no_pages)
    except Exception as e:
        logger.error("populate_from_concordance failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    if unresolved_docs:
        top_unresolved = sorted(unresolved_docs.items(), key=lambda x: -x[1])[:15]
        logger.warning("Top unresolved documents (%d unique): %s", len(unresolved_docs), top_unresolved)
    if unresolved_pages:
        top_pages = sorted(unresolved_pages.items(), key=lambda x: -x[1])[:15]
        logger.warning("Top unresolved (doc, page) pairs (%d unique): %s", len(unresolved_pages), top_pages)

    logger.info("Authoritative (concordance): %d rows %s",
                count, "(dry run)" if dry_run else "inserted/updated")
    return count


# ---------------------------------------------------------------------------
# Step 3: Multi-word alias expansion (optional)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    # Common English words that should never become aliases
    "the", "and", "for", "was", "are", "but", "not", "you", "all",
    "her", "his", "one", "our", "out", "has", "had", "its", "who",
    "how", "may", "new", "now", "old", "see", "way", "did", "get",
    "let", "say", "she", "too", "use", "also", "from", "been", "have",
    "this", "that", "with", "they", "will", "each", "make", "like",
    "long", "look", "many", "some", "than", "them", "then", "what",
    "when", "your", "into", "over", "such", "take", "come", "could",
    "made", "about", "after", "other", "which", "their", "there",
    "would", "first", "these", "where", "being", "between",
    # Common words likely to appear in corpus context
    "soviet", "american", "russian", "united", "states", "department",
    "intelligence", "agent", "cover", "name", "code", "group", "case",
    "report", "information", "secret", "source", "office", "military",
    "foreign", "national", "security", "service", "chief", "section",
    "member", "officer", "general", "special", "director", "affairs",
    "political", "work", "worker", "embassy", "consul", "station",
    # Short common words that are noise
    "man", "men", "big", "two", "set", "run", "put", "end", "far",
    "own", "off", "top", "red", "war", "part", "left", "right",
    "back", "only", "just", "most", "very", "well", "still", "every",
])


def expand_multiword_aliases(
    conn,
    pipeline_version: str,
    dry_run: bool = False,
) -> int:
    """Expand multi-word aliases to single-word aliases when unique.
    
    For example, if "aileron eleron" only appears for one entity, add
    "aileron" and "eleron" as separate aliases for that entity.
    
    Safeguards:
    - Only words >= 3 chars
    - Only words unique to one entity in the entire index
    - Excludes common English stopwords
    - Excludes words that already exist as single-word surfaces in PEM
    
    Uses a single bulk SQL query for maximum speed.
    """
    count = 0
    try:
        with conn.cursor() as cur:
            logger.info("Analyzing multi-word aliases...")
            
            if dry_run:
                # Dry run: just count with same exclusions
                stopwords_list = list(_STOPWORDS)
                cur.execute("""
                    SELECT DISTINCT surface_norm 
                    FROM page_entity_mentions 
                    WHERE surface_norm NOT LIKE '%% %%'
                """)
                existing_singles = {r[0] for r in cur.fetchall()}
                all_exclusions = list(_STOPWORDS | existing_singles | BLOCKED_SURFACE_NORMS)
                
                cur.execute("""
                    WITH multiword_surfaces AS (
                        SELECT entity_id, surface_norm
                        FROM page_entity_mentions
                        WHERE surface_norm LIKE '%% %%'
                        GROUP BY entity_id, surface_norm
                    ),
                    words_extracted AS (
                        SELECT 
                            entity_id,
                            unnest(string_to_array(surface_norm, ' ')) AS word
                        FROM multiword_surfaces
                    ),
                    word_entity_counts AS (
                        SELECT 
                            word,
                            array_agg(DISTINCT entity_id) AS entity_ids
                        FROM words_extracted
                        WHERE length(word) >= 4
                          AND word != ALL(%s)
                        GROUP BY word
                    ),
                    unique_words AS (
                        SELECT 
                            word,
                            entity_ids[1] AS entity_id
                        FROM word_entity_counts
                        WHERE array_length(entity_ids, 1) = 1
                    ),
                    expansion_rows AS (
                        SELECT DISTINCT
                            pem.collection_slug,
                            pem.document_id,
                            pem.page_id,
                            uw.word,
                            uw.entity_id
                        FROM unique_words uw
                        JOIN page_entity_mentions pem ON pem.entity_id = uw.entity_id
                            AND pem.surface_norm LIKE '%% %%'
                    )
                    SELECT COUNT(*) FROM expansion_rows
                """, (all_exclusions,))
                row = cur.fetchone()
                count = row[0] if row else 0
            else:
                # Real run: use INSERT SELECT for maximum speed
                logger.info("  Expanding multi-word aliases via bulk INSERT SELECT...")
                
                # Build stopword exclusion list
                stopwords_list = list(_STOPWORDS)
                
                # Also get existing single-word surfaces to avoid duplicating them
                cur.execute("""
                    SELECT DISTINCT surface_norm 
                    FROM page_entity_mentions 
                    WHERE surface_norm NOT LIKE '%% %%'
                """)
                existing_singles = {r[0] for r in cur.fetchall()}
                logger.info("  Excluding %d stopwords + %d existing single-word surfaces", 
                          len(stopwords_list), len(existing_singles))
                
                # Combine exclusions (stopwords, existing single-word surfaces, blocked norms)
                all_exclusions = list(_STOPWORDS | existing_singles | BLOCKED_SURFACE_NORMS)
                
                cur.execute("""
                    WITH multiword_surfaces AS (
                        SELECT entity_id, surface_norm
                        FROM page_entity_mentions
                        WHERE surface_norm LIKE '%% %%'
                        GROUP BY entity_id, surface_norm
                    ),
                    words_extracted AS (
                        SELECT 
                            entity_id,
                            unnest(string_to_array(surface_norm, ' ')) AS word
                        FROM multiword_surfaces
                    ),
                    word_entity_counts AS (
                        SELECT 
                            word,
                            array_agg(DISTINCT entity_id) AS entity_ids
                        FROM words_extracted
                        WHERE length(word) >= 4
                          AND word != ALL(%s)
                        GROUP BY word
                    ),
                    unique_words AS (
                        SELECT 
                            word,
                            entity_ids[1] AS entity_id
                        FROM word_entity_counts
                        WHERE array_length(entity_ids, 1) = 1
                    )
                    INSERT INTO page_entity_mentions
                        (collection_slug, document_id, page_id, surface_norm, surface_raw,
                         entity_id, truth_level, pipeline_version, source)
                    SELECT DISTINCT
                        pem.collection_slug,
                        pem.document_id,
                        pem.page_id,
                        uw.word,
                        uw.word,
                        uw.entity_id,
                        'derived',
                        %s,
                        'multiword_expansion'
                    FROM unique_words uw
                    JOIN page_entity_mentions pem ON pem.entity_id = uw.entity_id
                        AND pem.surface_norm LIKE '%% %%'
                    ON CONFLICT (page_id, surface_norm, entity_id) DO NOTHING
                """, (all_exclusions, pipeline_version))
                count = cur.rowcount
                conn.commit()
                logger.info("  Inserted %d single-word expansion rows", count)
    
    except Exception as e:
        import traceback
        logger.error("expand_multiword_aliases failed: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        try:
            conn.rollback()
        except Exception:
            pass
    
    logger.info("Multi-word alias expansion: %d rows %s", 
               count, "(dry run)" if dry_run else "added")
    return count


# ---------------------------------------------------------------------------
# Step 5: Index health acceptance check
# ---------------------------------------------------------------------------

def run_health_check(conn) -> Dict:
    """Compute index health metrics after rebuild.

    Returns dict with:
      - total_rows: total in page_entity_mentions
      - coverage: % of rows with >= 1 chunk via chunk_pages
      - alignment: % of covered rows where at least one chunk has entity_mentions
      - high_entropy_surfaces: top 50 surfaces by page-count with high entropy
    """
    metrics: Dict = {}
    try:
        with conn.cursor() as cur:
            # Total rows
            cur.execute("SELECT COUNT(*) FROM page_entity_mentions")
            total = cur.fetchone()[0]
            metrics["total_rows"] = total

            if total == 0:
                metrics["coverage"] = 0.0
                metrics["alignment"] = 0.0
                metrics["high_entropy_surfaces"] = []
                return metrics

            # Coverage: % of rows with >=1 chunk via chunk_pages
            cur.execute("""
                SELECT COUNT(DISTINCT pem.id)
                FROM page_entity_mentions pem
                JOIN chunk_pages cp ON cp.page_id = pem.page_id
            """)
            covered = cur.fetchone()[0]
            metrics["coverage"] = round(covered / total, 4)

            # Alignment: of covered rows, % where chunk has entity_mentions for same entity
            if covered > 0:
                cur.execute("""
                    SELECT COUNT(DISTINCT pem.id)
                    FROM page_entity_mentions pem
                    JOIN chunk_pages cp ON cp.page_id = pem.page_id
                    JOIN entity_mentions em ON em.chunk_id = cp.chunk_id
                                           AND em.entity_id = pem.entity_id
                """)
                aligned = cur.fetchone()[0]
                metrics["alignment"] = round(aligned / covered, 4)
            else:
                metrics["alignment"] = 0.0

            # High entropy surfaces: top 50 surfaces with most distinct entities
            cur.execute("""
                SELECT surface_norm,
                       COUNT(DISTINCT entity_id) AS n_entities,
                       COUNT(DISTINCT page_id) AS n_pages
                FROM page_entity_mentions
                GROUP BY surface_norm
                HAVING COUNT(DISTINCT entity_id) > 1
                ORDER BY COUNT(DISTINCT entity_id) DESC, COUNT(DISTINCT page_id) DESC
                LIMIT 50
            """)
            metrics["high_entropy_surfaces"] = [
                {"surface_norm": row[0], "n_entities": row[1], "n_pages": row[2]}
                for row in cur.fetchall()
            ]

    except Exception as e:
        logger.error("Health check failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    return metrics


def log_health_check(metrics: Dict) -> bool:
    """Log health check results and return True if healthy."""
    total = metrics.get("total_rows", 0)
    coverage = metrics.get("coverage", 0.0)
    alignment = metrics.get("alignment", 0.0)
    high_entropy = metrics.get("high_entropy_surfaces", [])

    logger.info("=== Index Health Check ===")
    logger.info("Total rows: %d", total)
    logger.info("Coverage (rows with chunks): %.1f%%", coverage * 100)
    logger.info("Alignment (covered + entity_mentions): %.1f%%", alignment * 100)

    if high_entropy:
        logger.info("Top high-entropy surfaces (ambiguous aliases):")
        for s in high_entropy[:10]:
            logger.info("  %s: %d entities, %d pages",
                        s["surface_norm"], s["n_entities"], s["n_pages"])

    healthy = True
    if coverage < COVERAGE_WARN_THRESHOLD:
        logger.warning("Coverage %.1f%% is below threshold %.1f%%",
                        coverage * 100, COVERAGE_WARN_THRESHOLD * 100)
        healthy = False
    if alignment < ALIGNMENT_WARN_THRESHOLD:
        logger.warning("Alignment %.1f%% is below threshold %.1f%%",
                        alignment * 100, ALIGNMENT_WARN_THRESHOLD * 100)
        healthy = False

    if healthy:
        logger.info("Index health: OK")
    else:
        logger.warning("Index health: WARN (see above)")

    return healthy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Populate page_entity_mentions")
    parser.add_argument("--truncate", action="store_true",
                        help="Truncate table before populating")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count rows only, do not insert")
    parser.add_argument("--skip-concordance", action="store_true",
                        help="Skip concordance-based authoritative rows")
    parser.add_argument("--skip-referent-rules", action="store_true",
                        help="Skip alias_referent_rules-based rows")
    parser.add_argument("--expand-multiword", action="store_true",
                        help="Expand multi-word aliases to single words when unique")
    parser.add_argument("--skip-health-check", action="store_true",
                        help="Skip index health check")
    parser.add_argument("--run-cleanup-apply", metavar="PATH",
                        help="Apply cleanup from session JSON only (no coded garbage rules). Runs cleanup_concordance.py --apply-file PATH "
                        "before populate; merges and deletes come solely from the JSON file.")
    args = parser.parse_args()

    # Step 0a: Optionally apply cleanup session before populate
    if args.run_cleanup_apply:
        path = args.run_cleanup_apply
        # Extract slug from filename: cleanup_session_SLUG.json -> SLUG
        stem = Path(path).stem
        if stem.startswith("cleanup_session_"):
            slug = stem[len("cleanup_session_"):]
        else:
            logger.error("--run-cleanup-apply path must match cleanup_session_SLUG.json")
            sys.exit(1)
        logger.info("Running cleanup_concordance.py --apply-file %s --slug %s", path, slug)
        rc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "cleanup_concordance.py"),
             "--apply-file", path, "--slug", slug],
            cwd=str(REPO_ROOT),
        )
        if rc.returncode != 0:
            logger.error("cleanup_concordance.py exited with code %d", rc.returncode)
            sys.exit(rc.returncode)
        logger.info("Cleanup apply done.")

    pipeline_version = str(int(time.time()))
    logger.info("Pipeline version: %s", pipeline_version)

    conn = get_conn()

    # Step 0: Optionally truncate
    if args.truncate and not args.dry_run:
        logger.info("Truncating page_entity_mentions...")
        with conn.cursor() as cur:
            cur.execute("TRUNCATE page_entity_mentions")
            conn.commit()

    total = 0

    # Step 1: Authoritative from alias_referent_rules
    if not args.skip_referent_rules:
        total += populate_from_alias_referent_rules(conn, pipeline_version, args.dry_run)

    # Step 2: Authoritative from concordance
    if not args.skip_concordance:
        total += populate_from_concordance(conn, pipeline_version, args.dry_run)

    # Step 3: Multi-word alias expansion (optional)
    if args.expand_multiword:
        total += expand_multiword_aliases(conn, pipeline_version, args.dry_run)

    # Step 4: Update revision
    if not args.dry_run:
        set_index_revision(conn, pipeline_version)
        logger.info("Updated page_entity_mentions_revision to %s", pipeline_version)

    logger.info("Total rows: %d %s", total, "(dry run)" if args.dry_run else "")

    # Step 5: Health check
    if not args.skip_health_check and not args.dry_run:
        metrics = run_health_check(conn)
        log_health_check(metrics)

    conn.close()


if __name__ == "__main__":
    main()
