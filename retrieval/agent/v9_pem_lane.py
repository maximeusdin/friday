"""
V9 PEM Lane — deterministic alias-surface seeding + chunk annotation for V9.

Integrates with ResearchWorkspace (not LexiconV10). Guarantees codename/alias
recall when aliases exist (e.g. OSS → CABIN in Venona/Vassiliev) without
relying on vector search or LLM surface extraction.

Flow:
  1. Query → candidate surfaces (quoted, ALL CAPS, short tokens)
  2. Surfaces → entity candidates via PEM distribution
  3. Seed entities from Step 2 + workspace.entities + workspace.entity_candidates
  4. Entity → alias surfaces (from PEM)
  5. Multi-entity prioritization (when ≥2 entities, query-derived filter)
  6. Coverage pass: pages per alias surface
  7. Pages → chunks via chunk_pages

Entry points:
  pem_lane_seed_chunks(conn, workspace, scope, query_text) -> PemLaneResult
  build_pem_mapping_block_for_chunk(...) -> str

Invariants: Deterministic, capped, no LLM.
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from retrieval.agent.v10_normalize import normalize_surface_for_lookup
from retrieval.agent.v10_page_bridge import chunks_for_pages, get_index_revision, has_page_entity_mentions

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ALIAS_SCOPED = frozenset({"venona", "vassiliev"})
MAX_SEED_CHUNKS_TOTAL = 25
MAX_ALIAS_SURFACES_PER_ENTITY = 8
PAGES_PER_ALIAS = 1  # pages to add per surface (min one chunk per alias)
PAGES_PER_ALIAS_CANDIDATES = 5  # candidate pages to try when first is shared
MAX_PAGES_PER_ENTITY = 20
MAX_CHUNKS_PER_PAGE = 2
MULTI_ENTITY_CHUNK_QUOTA = 8
MULTI_ENTITY_PAGE_LIMIT = 10
SURFACE_POOL_LIMIT = 50
MAX_MAPPING_LINES = 10

# Source ranking: lower = more authoritative
_SOURCE_RANK: Dict[str, int] = {
    "alias_referent_rules": 0,
    "concordance": 1,
    "entity_mentions": 2,
    "entity_aliases": 3,
}
_SOURCE_RANK_DEFAULT = 9

_CODENAME_TOKEN_RE = re.compile(r"\b([A-Z]{4,12})\b")
_QUOTED_STRING_RE = re.compile(r"""["']([^"']{2,30})["']""")
# Word boundary for Rule B: avoid matching CABIN inside CABINET
_WORD_BOUNDARY_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9\-]{1,24})\b")

# Generic tokens that are NOT stable operational aliases (D: stoplist)
OPERATIONAL_ALIAS_STOPLIST = frozenset({
    "american", "bureau", "research", "information", "service", "partial",
    "unidentified", "office", "agency", "department", "committee",
    "federal", "intelligence", "security", "division", "section",
})


# =============================================================================
# PemLaneResult
# =============================================================================

@dataclass
class PemLaneResult:
    """Output of the PEM lane seeding pipeline."""
    chunk_ids: List[int] = field(default_factory=list)
    page_ids: List[int] = field(default_factory=list)
    seeded_surfaces: List[str] = field(default_factory=list)
    seeded_entities: List[int] = field(default_factory=list)
    query_derived_entity_ids: Set[int] = field(default_factory=set)
    reason_codes: List[str] = field(default_factory=list)
    pem_revision: str = "0"
    stats: Dict[str, Any] = field(default_factory=dict)
    chunk_surface_map: Dict[int, str] = field(default_factory=dict)
    # Cache for mapping builder: {page_id: [(surface_norm, surface_raw, entity_id, source, truth_level, collection_slug)]}
    pem_cache: Dict[int, List[Tuple[Any, ...]]] = field(default_factory=dict)
    # entity_id -> canonical_name (bulk-fetched)
    canonical_map: Dict[int, str] = field(default_factory=dict)


# =============================================================================
# Helpers
# =============================================================================

def _source_rank(source: Optional[str]) -> int:
    if source is None:
        return _SOURCE_RANK_DEFAULT
    return _SOURCE_RANK.get(source, _SOURCE_RANK_DEFAULT)


def _truth_rank(truth_level: Optional[str]) -> int:
    return 0 if truth_level == "authoritative" else 1


def _is_canonical_ish(surface_norm: str, canonical_name: str) -> bool:
    """Exclude surfaces that are essentially the canonical name."""
    canonical_norm = normalize_surface_for_lookup(canonical_name)
    if not surface_norm or not canonical_norm:
        return False
    if len(surface_norm) <= 5 and " " not in surface_norm:
        return False
    if surface_norm == canonical_norm:
        return True
    surface_tokens = set(surface_norm.split())
    if len(surface_tokens) >= 2:
        canonical_tokens = set(canonical_norm.split())
        if surface_tokens <= canonical_tokens:
            return True
    return False


def _detect_candidate_surfaces(query_text: str) -> List[str]:
    """Extract tokens from query: quoted, ALL CAPS."""
    results: List[str] = []
    for m in _CODENAME_TOKEN_RE.finditer(query_text):
        results.append(query_text[m.start():m.end()])
    for m in _QUOTED_STRING_RE.finditer(query_text):
        results.append(m.group(1))
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,24}", query_text)
    for w in words:
        if len(w) >= 2 and len(w) <= 6 and w not in results:
            results.append(w)
    seen: Set[str] = set()
    deduped: List[str] = []
    for s in results:
        norm = normalize_surface_for_lookup(s)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def _surface_exists_in_pem(conn, surface_norm: str, collections: List[str]) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM page_entity_mentions
                WHERE surface_norm = %s AND collection_slug = ANY(%s)
                LIMIT 1
            """, (surface_norm, collections))
            return cur.fetchone() is not None
    except Exception as e:
        logger.debug("[PEM] surface_exists check failed for %s: %s", surface_norm, e)
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def _is_generic_operational_surface(surface_norm: str, count_pages: int = 0) -> bool:
    """Filter out ultra-generic surfaces from operational alias sets (D)."""
    if not surface_norm or len(surface_norm) < 2:
        return True
    norm_lower = surface_norm.strip().lower()
    if norm_lower in OPERATIONAL_ALIAS_STOPLIST:
        return True
    # Cheap heuristic: lowercase common words with huge page count
    if (
        surface_norm.islower()
        and len(surface_norm) <= 8
        and count_pages > 100
    ):
        return True
    return False


def _surface_to_entity_distribution(
    conn, surface_norm: str, collections: List[str]
) -> List[Tuple[int, int]]:
    """Returns [(entity_id, count_pages), ...] ordered by count DESC."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, COUNT(DISTINCT page_id) AS cnt
                FROM page_entity_mentions
                WHERE surface_norm = %s AND collection_slug = ANY(%s)
                GROUP BY entity_id
                ORDER BY cnt DESC, entity_id ASC
                LIMIT 5
            """, (surface_norm, collections))
            return [(r[0], r[1]) for r in cur.fetchall()]
    except Exception as e:
        logger.debug("[PEM] surface_to_entity failed for %s: %s", surface_norm, e)
        return []


def resolve_keyword_via_pem(
    conn,
    keyword: str,
    scope,
    *,
    min_share: float = 0.7,
    verbose: bool = False,
) -> Optional[Tuple[int, str]]:
    """
    Resolve a keyword to (entity_id, canonical_name) via PEM when scope includes
    alias-scoped collections (venona, vassiliev). Uses page_entity_mentions as
    ground truth for codenames/aliases in those collections.

    Returns None if: PEM disabled, no alias-scoped in scope, ambiguous, or no hit.
    When scope.collections is empty, do NOT assume V/V — PEM runs only when agent
    explicitly invokes search_canonical (or resolve_codename). Keeps full archive
    powerful without bias.
    """
    if not keyword or len(keyword.strip()) < 2:
        return None
    if not has_page_entity_mentions(conn):
        return None

    scope_collections = scope.collections if scope and scope.collections else None
    if scope_collections:
        alias_scoped = [c for c in scope_collections if c in ALIAS_SCOPED]
    else:
        alias_scoped = []  # No V/V default when scope empty
    if not alias_scoped:
        return None

    surface_norm = normalize_surface_for_lookup(keyword.strip())
    if not surface_norm:
        return None

    dist = _surface_to_entity_distribution(conn, surface_norm, alias_scoped)
    if not dist:
        return None
    top_entity_id, top_count = dist[0]
    total = sum(c for _, c in dist)
    share = top_count / total if total else 1.0

    if share < min_share or len(dist) > 3:
        if verbose:
            logger.info("[PEM] resolve_keyword '%s': ambiguous (share=%.2f, entities=%d)", keyword, share, len(dist))
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, canonical_name FROM entities WHERE id = %s", (top_entity_id,))
            row = cur.fetchone()
            if row:
                canonical = row[1] or ""
                if verbose:
                    logger.info("[PEM] resolve_keyword '%s' -> %s (id=%d, share=%.2f)", keyword, canonical, top_entity_id, share)
                return (top_entity_id, canonical)
    except Exception:
        pass
    return None


# =============================================================================
# PEM operational alias map (A: PEM-truthy AliasMap)
# =============================================================================

def build_pem_operational_alias_map(
    conn,
    workspace,  # ResearchWorkspace
    scope,
    *,
    min_top_share: float = 0.8,
    verbose: bool = False,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Build alias map from PEM only, scoped to alias-scoped corpora.

    Rule: Only include (surface_norm -> entity_id) if:
      1. It exists in PEM in scope (venona/vassiliev)
      2. It is locally strong: surface maps to only one entity OR top_share >= 0.8

    Returns {surface_norm: {canonical, confidence, ambiguous}}.
    Returns None if PEM disabled or no alias-scoped in scope.
    When scope.collections is empty, do NOT assume V/V.
    """
    if not has_page_entity_mentions(conn):
        return None

    scope_collections = scope.collections if scope and scope.collections else None
    if scope_collections:
        alias_scoped = [c for c in scope_collections if c in ALIAS_SCOPED]
    else:
        alias_scoped = []  # No V/V default when scope empty
    if not alias_scoped:
        return None

    entity_ids: Set[int] = set()
    for e in workspace.entities:
        entity_ids.add(e.entity_id)
    for c in workspace.entity_candidates:
        if c.accepted and c.entity_id:
            entity_ids.add(c.entity_id)
    if not entity_ids:
        return None

    entity_canonical: Dict[int, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, canonical_name FROM entities WHERE id = ANY(%s)",
                (list(entity_ids),),
            )
            for row in cur.fetchall():
                entity_canonical[row[0]] = row[1] or ""
    except Exception as e:
        logger.debug("[PEM operational] entity fetch failed: %s", e)
        return None

    amap: Dict[str, Dict[str, Any]] = {}

    for eid in entity_ids:
        canonical = entity_canonical.get(eid, "")
        if not canonical:
            continue
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT surface_norm, COUNT(DISTINCT page_id) AS cnt
                    FROM page_entity_mentions
                    WHERE entity_id = %s AND collection_slug = ANY(%s)
                    GROUP BY surface_norm
                    ORDER BY cnt DESC
                    LIMIT 80
                """, (eid, alias_scoped))
                rows = cur.fetchall()
        except Exception as e:
            logger.debug("[PEM operational] surfaces for %d failed: %s", eid, e)
            continue

        canonical_norm = normalize_surface_for_lookup(canonical)
        for surface_norm, count_pages in rows:
            if _is_canonical_ish(surface_norm, canonical):
                continue
            if surface_norm == canonical_norm:
                continue
            if _is_generic_operational_surface(surface_norm, count_pages):
                continue

            dist = _surface_to_entity_distribution(conn, surface_norm, alias_scoped)
            if not dist:
                continue
            top_eid, top_count = dist[0]
            total = sum(c for _, c in dist)
            share = top_count / total if total else 1.0

            if top_eid != eid:
                continue
            if len(dist) > 1 and share < min_top_share:
                continue

            amap[surface_norm] = {
                "canonical": canonical,
                "confidence": "exact",
                "ambiguous": False,
            }

    if verbose and amap:
        logger.info("[PEM operational] built %d entries for %d entities", len(amap), len(entity_ids))
    return amap


# =============================================================================
# Main pipeline
# =============================================================================

def pem_lane_seed_chunks(
    conn,
    workspace,  # ResearchWorkspace
    scope,
    query_text: str,
    verbose: bool = True,
) -> PemLaneResult:
    """
    Seed chunks from PEM for V9. Merges into workspace and returns PEM cache.
    """
    result = PemLaneResult()
    try:
        result.pem_revision = get_index_revision(conn)
    except Exception:
        result.pem_revision = "0"

    # --- Scope check ---
    scope_collections = scope.collections if scope and scope.collections else None
    if scope_collections:
        alias_scoped = [c for c in scope_collections if c in ALIAS_SCOPED]
    else:
        alias_scoped = list(ALIAS_SCOPED)

    if not alias_scoped:
        logger.info("[PEM] Skipped: no alias-scoped collections in scope")
        if verbose:
            print("  [V9 PEM] Skipped: no venona/vassiliev in scope", file=sys.stderr)
        result.reason_codes.append("pem_lane_skipped_no_alias_scoped")
        return result

    if not has_page_entity_mentions(conn):
        logger.info("[PEM] Skipped: page_entity_mentions table empty or missing")
        if verbose:
            print("  [V9 PEM] Skipped: page_entity_mentions table empty or missing", file=sys.stderr)
        result.reason_codes.append("pem_lane_skipped_no_pem_table")
        return result

    if verbose:
        logger.info(
            "[PEM] Starting lane: scope=%s, query=%r, pem_revision=%s",
            alias_scoped, query_text[:80], result.pem_revision,
        )
        print(f"  [V9 PEM] Lane starting: scope={alias_scoped}", file=sys.stderr)

    # --- Step 2.1: Query → candidate surfaces ---
    raw_surfaces = _detect_candidate_surfaces(query_text)
    candidate_surfaces = [
        s for s in raw_surfaces
        if _surface_exists_in_pem(conn, s, alias_scoped)
    ]
    if verbose:
        logger.info("[PEM] Step 2.1: raw_surfaces=%s, candidate_surfaces=%s", raw_surfaces[:10], candidate_surfaces)

    # --- Step 2.2: Surfaces → entity candidates ---
    query_derived_entity_ids: Set[int] = set()
    for s in candidate_surfaces:
        dist = _surface_to_entity_distribution(conn, s, alias_scoped)
        if not dist:
            continue
        top_entity_id, top_count = dist[0]
        total = sum(c for _, c in dist)
        share = top_count / total if total else 1.0
        if share >= 0.8 or len(dist) == 1:
            query_derived_entity_ids.add(top_entity_id)
            if verbose:
                logger.info("[PEM] Step 2.2: surface=%s -> entity_id=%d (share=%.2f)", s, top_entity_id, share)
        elif len(dist) >= 2 and dist[1][1] >= top_count * 0.5:
            query_derived_entity_ids.add(top_entity_id)
            query_derived_entity_ids.add(dist[1][0])

    # --- Step 2.3: Seed entities-in-play ---
    entity_ids: Set[int] = set(query_derived_entity_ids)
    for e in workspace.entities:
        entity_ids.add(e.entity_id)
    for c in workspace.entity_candidates:
        if c.entity_id:
            entity_ids.add(c.entity_id)

    # Codename-rescue: no entities but query has codename token
    if not entity_ids and candidate_surfaces:
        for s in candidate_surfaces:
            dist = _surface_to_entity_distribution(conn, s, alias_scoped)
            if dist and len(dist) <= 3:
                entity_ids.add(dist[0][0])
                query_derived_entity_ids.add(dist[0][0])
                result.reason_codes.append(f"pem_lane_codename_rescue:{s}")
                break

    if not entity_ids:
        logger.info("[PEM] Skipped: no entities in play")
        result.reason_codes.append("pem_lane_skipped_no_entities")
        return result

    result.query_derived_entity_ids = query_derived_entity_ids
    if verbose:
        logger.info("[PEM] Step 2.3: entity_ids=%s, query_derived=%s", entity_ids, query_derived_entity_ids)

    # Order: workspace entities first (confirmed by user), then candidates, then query-derived
    ordered_entity_ids: List[int] = []
    for e in workspace.entities:
        if e.entity_id in entity_ids and e.entity_id not in ordered_entity_ids:
            ordered_entity_ids.append(e.entity_id)
    for c in workspace.entity_candidates:
        if c.entity_id and c.entity_id in entity_ids and c.entity_id not in ordered_entity_ids:
            ordered_entity_ids.append(c.entity_id)
    for eid in entity_ids:
        if eid not in ordered_entity_ids:
            ordered_entity_ids.append(eid)

    # --- Step 2.4: Entity → alias surfaces ---
    entity_canonical: Dict[int, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, canonical_name FROM entities WHERE id = ANY(%s)",
                (list(entity_ids),),
            )
            for row in cur.fetchall():
                entity_canonical[row[0]] = row[1] or ""
                result.canonical_map[row[0]] = row[1] or ""
    except Exception as e:
        logger.warning("[PEM] entity canonical fetch failed: %s", e)

    entity_surfaces: Dict[int, List[str]] = {}
    for eid in ordered_entity_ids:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT surface_norm, COUNT(DISTINCT page_id) AS cnt
                    FROM page_entity_mentions
                    WHERE entity_id = %s AND collection_slug = ANY(%s)
                    GROUP BY surface_norm
                    ORDER BY cnt DESC, surface_norm ASC
                    LIMIT %s
                """, (eid, alias_scoped, SURFACE_POOL_LIMIT))
                rows = cur.fetchall()
        except Exception as e:
            logger.debug("[PEM] entity surfaces failed for %d: %s", eid, e)
            rows = []

        canonical = entity_canonical.get(eid, "")
        kept: List[str] = []
        for surface_norm, _ in rows:
            if _is_canonical_ish(surface_norm, canonical):
                continue
            if len(kept) >= MAX_ALIAS_SURFACES_PER_ENTITY:
                break
            kept.append(surface_norm)

        # Fallback: if PEM has no surfaces for this entity, use entity_aliases from DB
        # (only add aliases that exist in PEM and map to this entity)
        if not kept:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT ea.alias FROM entity_aliases ea
                        WHERE ea.entity_id = %s
                        ORDER BY LENGTH(ea.alias) ASC
                        LIMIT %s
                    """, (eid, MAX_ALIAS_SURFACES_PER_ENTITY * 2))
                    for row in cur.fetchall():
                        alias = (row[0] or "").strip()
                        if not alias or len(alias) < 2:
                            continue
                        alias_norm = normalize_surface_for_lookup(alias)
                        if not alias_norm or _is_canonical_ish(alias_norm, canonical):
                            continue
                        if alias_norm in kept:
                            continue
                        dist = _surface_to_entity_distribution(conn, alias_norm, alias_scoped)
                        if dist and dist[0][0] == eid:
                            kept.append(alias_norm)
                            if len(kept) >= MAX_ALIAS_SURFACES_PER_ENTITY:
                                break
                if kept and verbose:
                    logger.info("[PEM] Step 2.4 fallback: entity_id=%d surfaces from entity_aliases=%s", eid, kept[:5])
            except Exception as e:
                logger.debug("[PEM] entity_aliases fallback failed for %d: %s", eid, e)

        entity_surfaces[eid] = kept
        if verbose and kept:
            logger.info("[PEM] Step 2.4: entity_id=%d surfaces=%s", eid, kept[:5])

    # --- Step 2.5: Forced retrieval ---
    all_page_ids: List[int] = []
    all_page_id_set: Set[int] = set()
    page_to_surface: Dict[int, str] = {}
    multi_entity_page_ids: List[int] = []

    # 2.5a Multi-entity prioritization (when ≥2 entities and query-derived)
    if len(entity_ids) >= 2 and query_derived_entity_ids:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT page_id, COUNT(DISTINCT entity_id) AS entity_count
                    FROM page_entity_mentions
                    WHERE entity_id = ANY(%s) AND collection_slug = ANY(%s)
                    GROUP BY page_id
                    HAVING COUNT(DISTINCT entity_id) >= 2
                    ORDER BY entity_count DESC, page_id ASC
                    LIMIT %s
                """, (list(entity_ids), alias_scoped, MULTI_ENTITY_PAGE_LIMIT))
                multi_entity_page_ids = [r[0] for r in cur.fetchall()]
        except Exception as e:
            logger.debug("[PEM] multi-entity query failed: %s", e)

        if multi_entity_page_ids:
            # Cap per doc (max 2 pages per document for diversity)
            doc_counts: Dict[int, int] = {}
            limited: List[int] = []
            page_docs: Dict[int, int] = {}
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT page_id, MIN(document_id) AS doc_id
                        FROM page_entity_mentions
                        WHERE page_id = ANY(%s)
                        GROUP BY page_id
                    """, (multi_entity_page_ids,))
                    page_docs = {r[0]: r[1] for r in cur.fetchall()}
            except Exception:
                pass

            for pid in multi_entity_page_ids:
                doc_id = page_docs.get(pid, pid)
                if doc_counts.get(doc_id, 0) >= 2:
                    continue
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
                limited.append(pid)
                if len(limited) >= MULTI_ENTITY_PAGE_LIMIT:
                    break

            multi_entity_page_ids = limited[:MULTI_ENTITY_PAGE_LIMIT]
            for pid in multi_entity_page_ids:
                if pid not in all_page_id_set:
                    all_page_id_set.add(pid)
                    all_page_ids.append(pid)
                    page_to_surface[pid] = "(multi_entity)"

            if verbose:
                logger.info("[PEM] Step 2.5a: multi_entity_page_ids=%s", multi_entity_page_ids[:5])

    # 2.5b Coverage pass (round-robin: ensure each entity gets surfaces before any gets more)
    excluded_pages = set(all_page_ids)
    page_limit = MAX_PAGES_PER_ENTITY * len(entity_ids)
    # Build (eid, surface) queue with round-robin order: interleave by entity
    eid_surface_pairs: List[Tuple[int, str]] = []
    surfaces_by_eid = {eid: list(surfaces) for eid, surfaces in entity_surfaces.items() if surfaces}
    round_idx = 0
    while any(surfaces_by_eid.values()):
        did_add = False
        for eid in ordered_entity_ids:
            if eid not in surfaces_by_eid or not surfaces_by_eid[eid]:
                continue
            surface_norm = surfaces_by_eid[eid].pop(0)
            eid_surface_pairs.append((eid, surface_norm))
            did_add = True
        if not did_add:
            break

    for eid, surface_norm in eid_surface_pairs:
        if len(all_page_ids) - len(multi_entity_page_ids) >= page_limit:
            break
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT document_id, page_id
                    FROM page_entity_mentions
                    WHERE entity_id = %s AND surface_norm = %s AND collection_slug = ANY(%s)
                    ORDER BY document_id ASC, page_id ASC
                    LIMIT %s
                """, (eid, surface_norm, alias_scoped, PAGES_PER_ALIAS_CANDIDATES))
                rows = cur.fetchall()
        except Exception:
            continue

        for doc_id, page_id in rows:
            if page_id in excluded_pages:
                continue
            excluded_pages.add(page_id)
            all_page_id_set.add(page_id)
            all_page_ids.append(page_id)
            page_to_surface[page_id] = surface_norm
            result.seeded_surfaces.append(surface_norm)
            result.seeded_entities.append(eid)
            break

    if not all_page_ids:
        logger.info("[PEM] No pages selected")
        result.reason_codes.append("pem_lane_no_pem_coverage")
        return result

    # --- Pages → chunks ---
    chunk_ids = chunks_for_pages(
        conn, all_page_ids,
        max_chunks_per_page=MAX_CHUNKS_PER_PAGE,
    )
    chunk_ids = chunk_ids[:MAX_SEED_CHUNKS_TOTAL]

    # Multi-entity chunks first (up to quota)
    chunk_entity_count: Dict[int, int] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cp.chunk_id, COUNT(DISTINCT pem.entity_id) AS ec
                FROM chunk_pages cp
                JOIN page_entity_mentions pem ON pem.page_id = cp.page_id
                    AND pem.collection_slug = ANY(%s)
                WHERE cp.page_id = ANY(%s)
                GROUP BY cp.chunk_id
            """, (alias_scoped, all_page_ids))
            for chunk_id, ec in cur.fetchall():
                chunk_entity_count[chunk_id] = ec
    except Exception:
        pass

    multi_chunks = [c for c in chunk_ids if chunk_entity_count.get(c, 0) >= 2]
    multi_chunks = multi_chunks[:MULTI_ENTITY_CHUNK_QUOTA]
    rest_chunks = [c for c in chunk_ids if c not in set(multi_chunks)]
    result.chunk_ids = multi_chunks + rest_chunks

    result.page_ids = all_page_ids
    result.seeded_entities = list(dict.fromkeys(result.seeded_entities))

    # --- Build PEM cache for mapping builder ---
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT page_id, surface_norm, surface_raw, entity_id, source, truth_level, collection_slug
                FROM page_entity_mentions
                WHERE page_id = ANY(%s) AND collection_slug = ANY(%s)
            """, (all_page_ids, alias_scoped))
            for row in cur.fetchall():
                pid, sn, sr, eid, src, truth, coll = row
                if pid not in result.pem_cache:
                    result.pem_cache[pid] = []
                result.pem_cache[pid].append((sn, sr or sn, eid, src or "", truth or "", coll or ""))
    except Exception as e:
        logger.warning("[PEM] pem_cache build failed: %s", e)

    result.reason_codes.append("pem_lane_ok")
    result.stats = {
        "surfaces_selected": len(set(result.seeded_surfaces)),
        "pages_selected": len(all_page_ids),
        "chunks_selected": len(result.chunk_ids),
        "multi_entity_pages": len(multi_entity_page_ids),
    }

    logger.info(
        "[PEM] Lane complete: %d chunks, %d pages, surfaces=%s, entities=%s",
        len(result.chunk_ids), len(result.page_ids),
        result.seeded_surfaces[:5], list(entity_ids)[:5],
    )
    if verbose:
        print(
            f"  [V9 PEM] Lane complete: {len(result.chunk_ids)} chunks, "
            f"{len(result.page_ids)} pages, stats={result.stats}",
            file=sys.stderr,
        )
    return result


# =============================================================================
# Chunk PEM annotation (page-scoped, model-only)
# =============================================================================

def build_pem_mapping_block_for_chunk(
    chunk_id: int,
    chunk_text: str,
    page_ids: List[int],
    pem_cache: Dict[int, List[Tuple[Any, ...]]],
    canonical_map: Dict[int, str],
    entities_in_play: Set[int],
    alias_scoped: Sequence[str] = tuple(ALIAS_SCOPED),
    max_mappings: int = MAX_MAPPING_LINES,
) -> str:
    """
    Build page-scoped mention-index annotation for a chunk.
    Rules A/B/C: unambiguous, ALL CAPS/quoted (boundary match), question-relevant.
    Never inject ambiguous mappings. Caps/quotes only make eligible; unambiguity required.
    """
    if not page_ids or not pem_cache:
        return ""

    # Collect PEM rows for this chunk's pages
    pem_rows: List[Tuple[Any, ...]] = []
    for pid in page_ids:
        pem_rows.extend(pem_cache.get(pid, []))

    if not pem_rows:
        return ""

    # Group by surface_norm; collect distinct entity_ids; keep best source
    surface_entities: Dict[str, Set[int]] = {}
    surface_source: Dict[str, str] = {}
    surface_raw_map: Dict[str, str] = {}
    for surface_norm, surface_raw_val, entity_id, source, truth, coll in pem_rows:
        if surface_norm not in surface_entities:
            surface_entities[surface_norm] = set()
            surface_source[surface_norm] = source or ""
            surface_raw_map[surface_norm] = surface_raw_val or surface_norm
        surface_entities[surface_norm].add(entity_id)
        if source and _source_rank(source) < _source_rank(surface_source.get(surface_norm, "")):
            surface_source[surface_norm] = source

    # Tokens in chunk: ALL CAPS and quoted (for Rule B)
    all_caps_tokens = {normalize_surface_for_lookup(m.group(1)) for m in _CODENAME_TOKEN_RE.finditer(chunk_text)}
    quoted_tokens = {normalize_surface_for_lookup(m.group(1)) for m in _QUOTED_STRING_RE.finditer(chunk_text)}

    # Eligibility: only unambiguous (single entity_id). Never inject ambiguous.
    eligible: List[Tuple[str, int, str]] = []
    for surface_norm, eids in surface_entities.items():
        if len(eids) != 1:
            continue
        eid = next(iter(eids))
        canonical = canonical_map.get(eid, f"entity_{eid}")

        include = False
        # Rule A: unambiguous — always eligible as baseline
        include = True

        # Rule B: surface appears in chunk as ALL CAPS or quoted (boundary match)
        sn_lower = surface_norm.lower() if surface_norm else ""
        raw_lower = normalize_surface_for_lookup(surface_raw_map.get(surface_norm, surface_norm))
        if sn_lower and (sn_lower in all_caps_tokens or sn_lower in quoted_tokens):
            include = True
        if raw_lower and (raw_lower in all_caps_tokens or raw_lower in quoted_tokens):
            include = True

        # Rule C: entity in play (question-relevant)
        if eid in entities_in_play:
            include = True

        if include:
            eligible.append((surface_norm, eid, canonical))

    if not eligible:
        return ""

    # Sort
    eligible.sort(key=lambda x: (_source_rank(surface_source.get(x[0], "")), x[0]))
    eligible = eligible[:max_mappings]

    lines = ["\n[MENTION_INDEX page_scoped collection=venona]"]
    for surface_norm, eid, canonical in eligible:
        lines.append(f"{surface_norm} => {canonical}")
    lines.append("[/MENTION_INDEX]")
    return "\n".join(lines)


def get_chunk_page_ids(conn, chunk_id: int) -> List[int]:
    """Get page_ids for a chunk from chunk_pages."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT page_id FROM chunk_pages WHERE chunk_id = %s ORDER BY span_order ASC",
                (chunk_id,),
            )
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []
