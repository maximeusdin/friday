"""
Canonical embedding text augmentation — PEM bridge (page-scoped).

Builds text_canonical = chunk.text + MENTION_INDEX block for alias-scoped
collections (venona, vassiliev). Uses page_entity_mentions (PEM) via chunk_pages.
Only includes surfaces that actually appear in the chunk text (word-boundary match).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Config
DOMINANT_THRESHOLD = 0.80  # Keep mapping if entity_id >= 80% of occurrences
MAX_MAPPINGS_PER_CHUNK = 50
MAX_BLOCK_CHARS = 6000  # 4–8KB range; use 6KB
MIN_SURFACE_LEN = 2  # Exclude single-char surfaces
GENERIC_SURFACE_DENYLIST = frozenset({
    "moscow", "washington", "soviet", "russia", "usa", "london", "new york",
    "berlin", "paris", "state", "department", "committee", "bureau",
})

SOURCE_PAGE_PEM_BRIDGE = "page_pem_bridge"
RULE_UNIQUE = "unique"
RULE_DOMINANT = "dominant"
RULE_MANUAL_OVERRIDE = "manual_override"
RULE_SUPPRESSED_AMBIGUOUS = "suppressed_ambiguous"


@dataclass
class MappingEntry:
    """One surface→entity mapping with evidence."""
    surface_norm: str
    entity_id: int
    canonical_name: str
    rule: str
    pages: List[int]
    count: int
    total: int
    ambiguity: bool


def _surface_in_text(surface_norm: str, text: str) -> bool:
    """True if surface appears in text (word-boundary match, case-insensitive)."""
    if not surface_norm or not text:
        return False
    parts = surface_norm.split()
    pattern = r"\b" + r"\s+".join(re.escape(w) for w in parts) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _is_generic_surface(surface_norm: str, entity_type: Optional[str]) -> bool:
    """Exclude generic surfaces unless PERSON/ORG and confident."""
    sn = (surface_norm or "").strip().lower()
    if not sn or len(sn) < MIN_SURFACE_LEN:
        return True
    if sn in GENERIC_SURFACE_DENYLIST:
        # Allow for PERSON/ORG (codenames, org acronyms)
        if entity_type in ("person", "org"):
            return False  # Don't exclude
        return True
    return False


def canonicalize_chunk(
    conn,
    chunk_id: int,
    text: str,
    *,
    alias_scoped_collections: Sequence[str] = ("venona", "vassiliev"),
    dominant_threshold: float = DOMINANT_THRESHOLD,
    max_mappings: int = MAX_MAPPINGS_PER_CHUNK,
    max_block_chars: int = MAX_BLOCK_CHARS,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build canonical text for a chunk using PEM page-scoped bridge.

    Returns (text_canonical, rewrite_manifest).
    - text_canonical: text + MENTION_INDEX block (or just text if no mappings)
    - rewrite_manifest: list of {surface_norm, entity_id, canonical_name, source, rule, pages, evidence}
    """
    if not text:
        text = ""

    # Step 1: Get page_ids for this chunk
    page_ids: List[int] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT page_id FROM chunk_pages
                WHERE chunk_id = %s
                ORDER BY span_order ASC
            """, (chunk_id,))
            page_ids = [r[0] for r in cur.fetchall()]
    except Exception as e:
        logger.debug("canonicalize_chunk: chunk_pages failed %s: %s", chunk_id, e)
        try:
            conn.rollback()
        except Exception:
            pass
        return text, []

    if not page_ids:
        return text, []

    # Step 2: Get PEM rows for these pages
    pem_rows: List[Tuple[str, int, str, str]] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pem.surface_norm,
                       pem.entity_id,
                       pem.source,
                       pem.collection_slug
                FROM page_entity_mentions pem
                WHERE pem.page_id = ANY(%s)
                  AND pem.collection_slug = ANY(%s)
            """, (page_ids, list(alias_scoped_collections)))
            pem_rows = cur.fetchall()
    except Exception as e:
        logger.debug("canonicalize_chunk: PEM query failed %s: %s", chunk_id, e)
        try:
            conn.rollback()
        except Exception:
            pass
        return text, []

    if not pem_rows:
        return text, []

    # Step 3: Group by surface_norm; count entity_id occurrences
    surface_entity_counts: Dict[str, Dict[int, int]] = {}  # surface -> {entity_id: count}
    surface_sources: Dict[str, str] = {}
    for surface_norm, entity_id, source, _ in pem_rows:
        if surface_norm not in surface_entity_counts:
            surface_entity_counts[surface_norm] = {}
            surface_sources[surface_norm] = source or ""
        surface_entity_counts[surface_norm][entity_id] = (
            surface_entity_counts[surface_norm].get(entity_id, 0) + 1
        )
        surface_sources[surface_norm] = source or surface_sources.get(surface_norm, "")

    # Step 4: Dominant rule — keep if entity_id >= dominant_threshold
    candidate_mappings: List[MappingEntry] = []
    for surface_norm, entity_counts in surface_entity_counts.items():
        total = sum(entity_counts.values())
        if total == 0:
            continue
        # Best entity by count
        best_entity_id = max(entity_counts.keys(), key=lambda e: entity_counts[e])
        best_count = entity_counts[best_entity_id]
        share = best_count / total
        if share < dominant_threshold:
            continue  # suppressed_ambiguous
        ambiguity = share < 1.0
        rule = RULE_UNIQUE if share >= 1.0 else RULE_DOMINANT
        candidate_mappings.append(MappingEntry(
            surface_norm=surface_norm,
            entity_id=best_entity_id,
            canonical_name="",  # Resolve below
            rule=rule,
            pages=page_ids,
            count=best_count,
            total=total,
            ambiguity=ambiguity,
        ))

    if not candidate_mappings:
        return text, []

    # Step 5: Resolve entity_id → canonical_name, entity_type
    entity_ids = list({m.entity_id for m in candidate_mappings})
    entity_info: Dict[int, Tuple[str, str]] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, canonical_name, entity_type
                FROM entities WHERE id = ANY(%s)
            """, (entity_ids,))
            for row in cur.fetchall():
                entity_info[row[0]] = (row[1] or "", row[2] or "")
    except Exception as e:
        logger.debug("canonicalize_chunk: entities query failed %s: %s", chunk_id, e)
        try:
            conn.rollback()
        except Exception:
            pass
        return text, []

    # Step 6: Apply filters (generic surface denylist, min length, must appear in chunk)
    filtered: List[MappingEntry] = []
    for m in candidate_mappings:
        canonical_name, entity_type = entity_info.get(m.entity_id, ("", ""))
        m.canonical_name = canonical_name
        if _is_generic_surface(m.surface_norm, entity_type):
            continue
        if len(m.surface_norm) < MIN_SURFACE_LEN:
            continue
        if not _surface_in_text(m.surface_norm, text):
            continue
        filtered.append(m)

    # Step 7: Sort (count desc, surface_norm), cap
    filtered.sort(key=lambda x: (-x.count, x.surface_norm))
    filtered = filtered[:max_mappings]

    # Step 8: Build manifest
    manifest: List[Dict[str, Any]] = []
    for m in filtered:
        manifest.append({
            "surface_norm": m.surface_norm,
            "entity_id": m.entity_id,
            "canonical_name": m.canonical_name,
            "source": SOURCE_PAGE_PEM_BRIDGE,
            "rule": m.rule,
            "pages": m.pages,
            "evidence": {
                "count": m.count,
                "total": m.total,
                "ambiguity": m.ambiguity,
            },
        })

    # Step 9: Build annotation block
    lines: List[str] = []
    lines.append("\n\n[MENTION_INDEX page_scoped collection=alias_scoped]")
    block_chars = len(lines[0]) + 30  # approx header + footer
    for m in filtered:
        line = f"{m.surface_norm} => {m.canonical_name}"
        if block_chars + len(line) + 2 > max_block_chars:
            break
        lines.append(line)
        block_chars += len(line) + 1
    lines.append("[/MENTION_INDEX]")

    annotation_block = "\n".join(lines)
    text_canonical = text.rstrip() + annotation_block

    return text_canonical, manifest


def canonicalize_batch(
    conn,
    batch: List[Tuple[int, str]],
    *,
    alias_scoped_collections: Sequence[str] = ("venona", "vassiliev"),
    dominant_threshold: float = DOMINANT_THRESHOLD,
    max_mappings: int = MAX_MAPPINGS_PER_CHUNK,
    max_block_chars: int = MAX_BLOCK_CHARS,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Batch version: canonicalize many chunks with 3 DB round-trips instead of 3*N.

    Returns list of (text_canonical, rewrite_manifest) in same order as batch.
    """
    if not batch:
        return []

    chunk_ids = [cid for cid, _ in batch]
    texts = {cid: (t or "") for cid, t in batch}

    # Step 1: All chunk_pages for batch (one query)
    chunk_to_pages: Dict[int, List[int]] = {cid: [] for cid in chunk_ids}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, page_id FROM chunk_pages
                WHERE chunk_id = ANY(%s)
                ORDER BY chunk_id, span_order ASC
            """, (chunk_ids,))
            for chunk_id, page_id in cur.fetchall():
                chunk_to_pages.setdefault(chunk_id, []).append(page_id)
    except Exception as e:
        logger.debug("canonicalize_batch: chunk_pages failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return [(texts[cid], []) for cid, _ in batch]

    all_page_ids = list({pid for pages in chunk_to_pages.values() for pid in pages})
    if not all_page_ids:
        return [(texts[cid], []) for cid, _ in batch]

    # Step 2: All PEM for those pages (one query)
    page_to_pem: Dict[int, List[Tuple[str, int, str, str]]] = {pid: [] for pid in all_page_ids}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pem.page_id, pem.surface_norm, pem.entity_id, pem.source, pem.collection_slug
                FROM page_entity_mentions pem
                WHERE pem.page_id = ANY(%s) AND pem.collection_slug = ANY(%s)
            """, (all_page_ids, list(alias_scoped_collections)))
            for page_id, surface_norm, entity_id, source, _ in cur.fetchall():
                page_to_pem.setdefault(page_id, []).append((surface_norm, entity_id, source, ""))
    except Exception as e:
        logger.debug("canonicalize_batch: PEM failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return [(texts[cid], []) for cid, _ in batch]

    # Step 3: Build candidate_mappings per chunk (in memory)
    all_entity_ids: set = set()
    chunk_candidates: Dict[int, List[MappingEntry]] = {}

    for chunk_id, text in batch:
        page_ids = chunk_to_pages.get(chunk_id, [])
        if not page_ids:
            chunk_candidates[chunk_id] = []
            continue

        pem_rows = []
        for pid in page_ids:
            pem_rows.extend(page_to_pem.get(pid, []))

        if not pem_rows:
            chunk_candidates[chunk_id] = []
            continue

        surface_entity_counts: Dict[str, Dict[int, int]] = {}
        for surface_norm, entity_id, source, _ in pem_rows:
            if surface_norm not in surface_entity_counts:
                surface_entity_counts[surface_norm] = {}
            surface_entity_counts[surface_norm][entity_id] = (
                surface_entity_counts[surface_norm].get(entity_id, 0) + 1
            )

        candidate_mappings: List[MappingEntry] = []
        for surface_norm, entity_counts in surface_entity_counts.items():
            total = sum(entity_counts.values())
            if total == 0:
                continue
            best_entity_id = max(entity_counts.keys(), key=lambda e: entity_counts[e])
            best_count = entity_counts[best_entity_id]
            share = best_count / total
            if share < dominant_threshold:
                continue
            ambiguity = share < 1.0
            rule = RULE_UNIQUE if share >= 1.0 else RULE_DOMINANT
            candidate_mappings.append(MappingEntry(
                surface_norm=surface_norm,
                entity_id=best_entity_id,
                canonical_name="",
                rule=rule,
                pages=page_ids,
                count=best_count,
                total=total,
                ambiguity=ambiguity,
            ))
            all_entity_ids.add(best_entity_id)

        chunk_candidates[chunk_id] = candidate_mappings

    # Step 4: Resolve entities (one query)
    entity_info: Dict[int, Tuple[str, str]] = {}
    if all_entity_ids:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, canonical_name, entity_type
                    FROM entities WHERE id = ANY(%s)
                """, (list(all_entity_ids),))
                for row in cur.fetchall():
                    entity_info[row[0]] = (row[1] or "", row[2] or "")
        except Exception as e:
            logger.debug("canonicalize_batch: entities failed: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    # Step 5: For each chunk, filter, sort, build manifest + text
    results: List[Tuple[str, List[Dict[str, Any]]]] = []
    for chunk_id, _ in batch:
        text = texts[chunk_id]
        candidate_mappings = chunk_candidates.get(chunk_id, [])

        filtered: List[MappingEntry] = []
        for m in candidate_mappings:
            canonical_name, entity_type = entity_info.get(m.entity_id, ("", ""))
            m.canonical_name = canonical_name
            if _is_generic_surface(m.surface_norm, entity_type):
                continue
            if len(m.surface_norm) < MIN_SURFACE_LEN:
                continue
            if not _surface_in_text(m.surface_norm, text):
                continue
            filtered.append(m)

        filtered.sort(key=lambda x: (-x.count, x.surface_norm))
        filtered = filtered[:max_mappings]

        manifest: List[Dict[str, Any]] = []
        for m in filtered:
            manifest.append({
                "surface_norm": m.surface_norm,
                "entity_id": m.entity_id,
                "canonical_name": m.canonical_name,
                "source": SOURCE_PAGE_PEM_BRIDGE,
                "rule": m.rule,
                "pages": m.pages,
                "evidence": {"count": m.count, "total": m.total, "ambiguity": m.ambiguity},
            })

        lines = ["\n\n[MENTION_INDEX page_scoped collection=alias_scoped]"]
        block_chars = len(lines[0]) + 30
        for m in filtered:
            line = f"{m.surface_norm} => {m.canonical_name}"
            if block_chars + len(line) + 2 > max_block_chars:
                break
            lines.append(line)
            block_chars += len(line) + 1
        lines.append("[/MENTION_INDEX]")
        text_canonical = text.rstrip() + "\n".join(lines)
        results.append((text_canonical, manifest))

    return results
