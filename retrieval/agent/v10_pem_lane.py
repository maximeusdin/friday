"""
V10.2 PEM Lane — deterministic alias-surface seeding + chunk annotation.

Guarantees codename recall when aliases exist (e.g. OSS → CABIN in Venona/Vassiliev)
without relying on LLM surface extraction.

Flow:
  1. Candidate surfaces: PEM (observed) + entity_aliases (augmentation). Rank by
     pem_observed, truth_level/source, page coverage, surface_norm.
  2. Ground surfaces to pages: batched page_entity_mentions query. Only surfaces
     with PEM rows seed pages (alias-only surfaces with zero rows do not).
  3. Round-robin page selection: take 1 page per surface per round until
     PAGES_PER_SURFACE per surface or caps. Ensures diversity (e.g. izba hut
     gets pages even if oss dominates).
  4. General entity pages fallback when surface-grounding yields few pages.
  5. Pages → chunks via chunks_for_pages, cap MAX_SEED_CHUNKS_TOTAL.

Implementation notes (speed):
  - Batch queries: one entity_aliases per entity, one PEM grounding per entity
    with surface_norm ANY(:surfaces) (fast with indexes).
  - Cache: entity_id -> normalized alias surfaces in-memory per run (alias_cache).
  - Two surface lists: surfaces_grounded (have pages), surfaces_hint_only (no pages; telemetry).

Entry points:
  pem_lane_seed_chunks(conn, lexicon, scope, query_text) -> PemLaneResult
  build_chunk_pem_annotation(conn, chunk_id, alias_scoped_collections, ...) -> str

Invariants: Deterministic, capped, no LLM.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from retrieval.agent.v10_normalize import normalize_surface_for_lookup
from retrieval.agent.v10_page_bridge import chunks_for_pages, get_index_revision
from retrieval.agent.v10_types import ALIAS_SCOPED_COLLECTIONS

logger = logging.getLogger(__name__)

# =============================================================================
# Caps (start safe — can be raised later)
# =============================================================================

MAX_SEED_CHUNKS_TOTAL = 30
MAX_SURFACES_PER_ENTITY = 6
PAGES_PER_SURFACE = 2
MAX_PAGES_PER_ENTITY = 20      # cap pages per entity from surface grounding
PAGES_GENERAL_PER_ENTITY = 4   # fallback when surface-grounding yields few pages
MAX_CHUNKS_PER_PAGE = 3
SURFACE_POOL_LIMIT = 50        # fetch up to this many surfaces before filtering
MAX_ANNOTATION_MAPPINGS = 8    # cap for per-chunk annotation lines

# Source ranking: lower = more authoritative
_SOURCE_RANK: Dict[str, int] = {
    "alias_referent_rules": 0,
    "concordance": 1,
    "entity_mentions": 2,
    "entity_aliases": 3,
}
_SOURCE_RANK_DEFAULT = 9

# Codename-rescue regex: all-caps token 4–12 chars (letters only)
_CODENAME_TOKEN_RE = re.compile(r"\b([A-Z]{4,12})\b")

# Quoted string pattern (single or double quotes)
_QUOTED_STRING_RE = re.compile(r"""["']([^"']{2,30})["']""")


# =============================================================================
# PemLaneResult
# =============================================================================

@dataclass
class PemLaneResult:
    """Output of the PEM lane seeding pipeline."""
    chunk_ids: List[int] = field(default_factory=list)
    page_ids: List[int] = field(default_factory=list)
    seeded_surfaces: List[str] = field(default_factory=list)  # surfaces that seeded pages
    seeded_entities: List[int] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    pem_revision: str = "0"
    stats: Dict[str, Any] = field(default_factory=dict)

    # Per-chunk provenance: chunk_id -> primary seeding surface
    chunk_surface_map: Dict[int, str] = field(default_factory=dict)

    # Telemetry: surfaces_grounded (have pages) vs surfaces_hint_only (no pages, not used)
    surfaces_grounded: List[str] = field(default_factory=list)
    surfaces_hint_only: List[str] = field(default_factory=list)


# =============================================================================
# Internal helpers
# =============================================================================

def _source_rank(source: Optional[str]) -> int:
    """Numeric rank for a PEM source (lower = more authoritative)."""
    if source is None:
        return _SOURCE_RANK_DEFAULT
    return _SOURCE_RANK.get(source, _SOURCE_RANK_DEFAULT)


def _truth_rank(truth_level: Optional[str]) -> int:
    """Numeric rank for truth_level (lower = more authoritative)."""
    if truth_level == "authoritative":
        return 0
    return 1  # 'derived' or anything else


def _is_canonical_ish(surface_norm: str, canonical_name: str) -> bool:
    """Conservative filter: is surface_norm essentially the canonical name?

    Rules (conservative — avoid dropping useful abbreviations):
      - Exact match after normalization.
      - Surface is a token subset of canonical (e.g. "office of strategic services"
        split into tokens contains surface as one of them — but only if surface
        is multi-word AND overlaps significantly).
      - Short all-caps tokens (e.g. "oss", "nkvd") are NEVER filtered.
    """
    canonical_norm = normalize_surface_for_lookup(canonical_name)
    if not surface_norm or not canonical_norm:
        return False

    # Never filter short all-caps-like surfaces (after casefolding they're lowercase)
    # Detect by checking if the original would have been all-caps: len <= 5 and no spaces
    if len(surface_norm) <= 5 and " " not in surface_norm:
        return False

    # Exact match
    if surface_norm == canonical_norm:
        return True

    # Token subset: canonical tokens fully contain surface tokens
    # Only apply when surface has 2+ tokens (to avoid filtering single words
    # that happen to appear in canonical)
    surface_tokens = set(surface_norm.split())
    if len(surface_tokens) >= 2:
        canonical_tokens = set(canonical_norm.split())
        if surface_tokens <= canonical_tokens:
            return True

    return False


def _detect_codename_tokens(query_text: str) -> List[str]:
    """Extract codename-like tokens from query for the rescue path.

    Returns normalized surface_norms for all-caps 4-12 char tokens
    and quoted strings found in the query.
    """
    results: List[str] = []
    # All-caps tokens
    for m in _CODENAME_TOKEN_RE.finditer(query_text):
        results.append(normalize_surface_for_lookup(m.group(1)))
    # Quoted strings
    for m in _QUOTED_STRING_RE.finditer(query_text):
        results.append(normalize_surface_for_lookup(m.group(1)))
    # Dedupe preserving order
    seen: set = set()
    deduped: List[str] = []
    for s in results:
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


# =============================================================================
# Step 1: Candidate surfaces (PEM + entity_aliases)
# =============================================================================

def _collect_candidate_surfaces(
    conn,
    entity_id: int,
    canonical_name: str,
    collections: List[str],
    debug: bool = False,
    alias_cache: Optional[Dict[int, List[str]]] = None,
) -> List[Tuple[str, int, str, str, bool]]:
    """Collect and rank candidate surfaces for an entity.

    Returns list of (surface_norm, count_pages, source, truth_level, pem_observed)
    sorted by: pem_observed first, then truth/source, then page coverage, then surface_norm.

    Surfaces from entity_aliases with no PEM rows will have count_pages=0 and
    pem_observed=False — they are candidates for grounding; if no pages in Step 2,
    they will not seed pages.

    alias_cache: optional dict entity_id -> normalized surfaces; populated on first fetch.
    """
    # A. Pull PEM surfaces (one query per entity)
    pem_surfaces: Dict[str, Tuple[int, str, str]] = {}  # surface -> (count, source, truth)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT surface_norm,
                       COUNT(DISTINCT page_id) AS entity_page_count,
                       MIN(source) AS min_source,
                       MIN(truth_level) AS min_truth
                FROM page_entity_mentions
                WHERE entity_id = %(eid)s
                  AND collection_slug = ANY(%(colls)s)
                GROUP BY surface_norm
                ORDER BY COUNT(DISTINCT page_id) DESC, surface_norm ASC
                LIMIT %(pool_limit)s
            """, {"eid": entity_id, "colls": collections, "pool_limit": SURFACE_POOL_LIMIT})
            for row in cur.fetchall():
                surface_norm, count, source, truth = row[0], row[1], row[2] or "", row[3] or ""
                if _is_canonical_ish(surface_norm, canonical_name):
                    continue
                pem_surfaces[surface_norm] = (count, source, truth)
    except Exception as e:
        logger.warning("PEM surfaces query failed for entity %d: %s", entity_id, e)
        try:
            conn.rollback()
        except Exception:
            pass

    if debug:
        print(
            f"[PEM debug] entity {entity_id} ({canonical_name}): "
            f"PEM surfaces={list(pem_surfaces.keys())[:10]}",
            flush=True,
        )

    # B. Pull entity_aliases (augmentation) — one query per entity, cache by entity_id
    alias_surfaces: Set[str] = set()
    cache = alias_cache if alias_cache is not None else {}
    if entity_id in cache:
        for surface_norm in cache[entity_id]:
            if surface_norm not in pem_surfaces and not _is_canonical_ish(surface_norm, canonical_name):
                alias_surfaces.add(surface_norm)
    else:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT alias FROM entity_aliases WHERE entity_id = %(eid)s ORDER BY id ASC",
                    {"eid": entity_id},
                )
                for (alias,) in cur.fetchall():
                    if not alias or not alias.strip():
                        continue
                    surface_norm = normalize_surface_for_lookup(alias)
                    if not surface_norm or surface_norm in pem_surfaces:
                        continue
                    if _is_canonical_ish(surface_norm, canonical_name):
                        continue
                    alias_surfaces.add(surface_norm)
                cache[entity_id] = list(alias_surfaces)
        except Exception as e:
            logger.debug("entity_aliases for entity %d failed: %s", entity_id, e)
            try:
                conn.rollback()
            except Exception:
                pass

    # Merge and rank: pem_observed first, then truth/source, then coverage, then surface
    ranked: List[Tuple[str, int, str, str, bool]] = []
    for surface_norm in pem_surfaces:
        count, source, truth = pem_surfaces[surface_norm]
        ranked.append((surface_norm, count, source, truth, True))

    for surface_norm in alias_surfaces:
        ranked.append((surface_norm, 0, "entity_aliases", "", False))

    def _rank_key(r):
        s, count, source, truth, pem = r
        return (
            not pem,  # pem_observed first (True=1, False=0 -> pem first)
            _truth_rank(truth) if truth else 1,
            _source_rank(source),
            -count,  # higher coverage first
            s,
        )

    ranked.sort(key=_rank_key)
    kept = ranked[:MAX_SURFACES_PER_ENTITY]

    if debug:
        print(
            f"[PEM debug] entity {entity_id}: ranked surfaces "
            f"{[(s, cnt, pem) for s, cnt, _, _, pem in kept]}",
            flush=True,
        )
    return kept


# =============================================================================
# Step 2: Ground surfaces to pages (batched PEM query)
# =============================================================================

def _ground_surfaces_to_pages(
    conn,
    entity_id: int,
    surface_norms: List[str],
    collections: List[str],
) -> Dict[str, List[Tuple[int, int]]]:
    """Batched grounding: which pages does PEM say (entity, surface) appear on?

    Returns Dict[surface_norm, List[(document_id, page_id)]] ordered by
    surface_norm ASC, document_id ASC, page_id ASC. Authoritative first when available.
    """
    if not surface_norms:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT surface_norm, document_id, page_id, truth_level, source
                FROM page_entity_mentions
                WHERE entity_id = %(eid)s
                  AND collection_slug = ANY(%(colls)s)
                  AND surface_norm = ANY(%(surfaces)s)
                ORDER BY surface_norm ASC, document_id ASC, page_id ASC
            """, {
                "eid": entity_id,
                "colls": collections,
                "surfaces": surface_norms,
            })
            rows = cur.fetchall()
    except Exception as e:
        logger.debug("_ground_surfaces_to_pages failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return {}

    # Group by surface_norm; optionally sort authoritative first
    by_surface: Dict[str, List[Tuple[int, int, int, str]]] = {}
    for surface_norm, doc_id, page_id, truth, source in rows:
        if surface_norm not in by_surface:
            by_surface[surface_norm] = []
        by_surface[surface_norm].append((doc_id, page_id, _truth_rank(truth), _source_rank(source)))

    out: Dict[str, List[Tuple[int, int]]] = {}
    for s, entries in by_surface.items():
        entries.sort(key=lambda e: (e[2], e[3], e[0], e[1]))  # truth, source, doc, page
        out[s] = [(e[0], e[1]) for e in entries]
    return out


# =============================================================================
# Step 3: Round-robin page selection (fair across surfaces)
# =============================================================================

def _select_pages_round_robin(
    surface_pages: Dict[str, List[Tuple[int, int]]],
    ranked_surfaces: List[str],
    pages_per_surface: int = PAGES_PER_SURFACE,
    max_pages_per_entity: int = MAX_PAGES_PER_ENTITY,
) -> List[Tuple[int, str]]:
    """Select pages fairly across surfaces using round-robin.

    Take 1 page from each surface per round until PAGES_PER_SURFACE per surface
    or max_pages_per_entity. Avoids high-frequency surfaces crowding out rarer ones.
    Returns [(page_id, surface_norm), ...].
    """
    queues: Dict[str, List[Tuple[int, int]]] = {}
    for s in ranked_surfaces:
        pages = surface_pages.get(s, [])
        if pages:
            queues[s] = list(pages)

    if not queues:
        return []

    selected: List[Tuple[int, str]] = []
    seen_page_ids: Set[int] = set()
    per_surface_count: Dict[str, int] = {s: 0 for s in queues}

    while len(selected) < max_pages_per_entity:
        added = False
        for s in ranked_surfaces:
            if s not in queues or per_surface_count[s] >= pages_per_surface:
                continue
            if len(selected) >= max_pages_per_entity:
                break
            while queues[s]:
                doc_id, page_id = queues[s].pop(0)
                if page_id not in seen_page_ids:
                    seen_page_ids.add(page_id)
                    selected.append((page_id, s))
                    per_surface_count[s] += 1
                    added = True
                    break
            if not queues[s]:
                del queues[s]
        if not added:
            break

    return selected


# =============================================================================
# Step 4: General entity pages (fallback)
# =============================================================================

def _get_general_entity_pages(
    conn,
    entity_id: int,
    collections: List[str],
    exclude_page_ids: Set[int],
    limit: int = PAGES_GENERAL_PER_ENTITY,
) -> List[int]:
    """Return general pages for an entity, excluding already-selected pages."""
    try:
        with conn.cursor() as cur:
            if exclude_page_ids:
                cur.execute("""
                    SELECT page_id FROM (
                        SELECT page_id, MIN(document_id) AS min_doc
                        FROM page_entity_mentions
                        WHERE entity_id = %(eid)s
                          AND collection_slug = ANY(%(colls)s)
                          AND page_id != ALL(%(exclude)s)
                        GROUP BY page_id
                    ) sub
                    ORDER BY min_doc ASC, page_id ASC
                    LIMIT %(lim)s
                """, {
                    "eid": entity_id,
                    "colls": collections,
                    "exclude": list(exclude_page_ids),
                    "lim": limit,
                })
            else:
                cur.execute("""
                    SELECT page_id FROM (
                        SELECT page_id, MIN(document_id) AS min_doc
                        FROM page_entity_mentions
                        WHERE entity_id = %(eid)s
                          AND collection_slug = ANY(%(colls)s)
                        GROUP BY page_id
                    ) sub
                    ORDER BY min_doc ASC, page_id ASC
                    LIMIT %(lim)s
                """, {
                    "eid": entity_id,
                    "colls": collections,
                    "lim": limit,
                })
            return [r[0] for r in cur.fetchall()]
    except Exception as e:
        logger.debug("_get_general_entity_pages failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return []


# =============================================================================
# Codename-rescue: surface_norm → entity_id via PEM
# =============================================================================

def _resolve_codename_from_pem(
    conn,
    surface_norm: str,
    collections: List[str],
) -> Optional[Tuple[int, str]]:
    """Look up a codename-like surface in PEM, return (entity_id, canonical_name) or None.

    Returns the entity with the most pages for this surface. If tied or
    ambiguous (>3 entities), returns None.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pem.entity_id,
                       e.canonical_name,
                       COUNT(DISTINCT pem.page_id) AS cnt
                FROM page_entity_mentions pem
                JOIN entities e ON e.id = pem.entity_id
                WHERE pem.surface_norm = %(surface)s
                  AND pem.collection_slug = ANY(%(colls)s)
                GROUP BY pem.entity_id, e.canonical_name
                ORDER BY cnt DESC, pem.entity_id ASC
                LIMIT 4
            """, {"surface": surface_norm, "colls": collections})
            rows = cur.fetchall()
    except Exception as e:
        logger.debug("_resolve_codename_from_pem failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return None

    if not rows:
        return None
    if len(rows) > 3:
        # Too ambiguous
        return None
    # Return top entity (deterministic: highest count, then lowest entity_id)
    return (rows[0][0], rows[0][1])


# =============================================================================
# Main entry: pem_lane_seed_chunks
# =============================================================================

def pem_lane_seed_chunks(
    conn,
    lexicon,  # LexiconV10 (avoid circular import)
    scope,    # Optional[ScopeFilter]
    query_text: str,
    debug: bool = False,
) -> PemLaneResult:
    """Seed chunks from PEM. Set PEM_DEBUG=1 for per-entity surface selection debug."""
    import os
    if not debug:
        debug = os.environ.get("PEM_DEBUG", "").strip().lower() in ("1", "true", "yes")
    """Deterministic PEM lane: entity→surfaces→pages→chunks.

    Trigger (two paths):
      Entity path: alias-scoped collections in scope AND entities_in_play >= 1.
      Codename-rescue: alias-scoped in scope, no entities, query has codename token.

    Returns PemLaneResult with seeded chunk_ids, surfaces, reason_codes, etc.
    """
    result = PemLaneResult()

    # Record PEM revision for determinism
    try:
        result.pem_revision = get_index_revision(conn)
    except Exception:
        result.pem_revision = "0"

    # --- Trigger check ---
    scope_collections = scope.collections if scope and scope.collections else None
    if scope_collections:
        alias_scoped = [c for c in scope_collections if c in ALIAS_SCOPED_COLLECTIONS]
    else:
        # If no scope restriction, use all alias-scoped collections
        alias_scoped = list(ALIAS_SCOPED_COLLECTIONS)

    if not alias_scoped:
        result.reason_codes.append("pem_lane_skipped_no_alias_scoped")
        return result

    # Collect entity targets: from entities_in_play or codename-rescue
    entity_targets: List[Tuple[int, str]] = []  # (entity_id, canonical_name)

    entities_in_play = getattr(lexicon, "entities_in_play", {})
    if entities_in_play:
        for eid, info in entities_in_play.items():
            canonical = info.get("canonical_name", "")
            entity_targets.append((eid, canonical))
    else:
        # Codename-rescue path
        codename_tokens = _detect_codename_tokens(query_text)
        if not codename_tokens:
            result.reason_codes.append("pem_lane_skipped_no_entities")
            return result

        for token_norm in codename_tokens:
            resolved = _resolve_codename_from_pem(conn, token_norm, alias_scoped)
            if resolved:
                entity_targets.append(resolved)
                result.reason_codes.append(f"pem_lane_codename_rescue:{token_norm}")
                break  # One rescue entity is enough
        if not entity_targets:
            result.reason_codes.append("pem_lane_skipped_no_entities")
            return result

    # --- Seeding algorithm ---
    all_page_ids: List[int] = []
    all_page_id_set: Set[int] = set()
    seeded_surfaces: List[str] = []
    surfaces_grounded: List[str] = []
    surfaces_hint_only: List[str] = []
    seeded_entities: List[int] = []
    page_to_surface: Dict[int, str] = {}

    # Per-run cache: entity_id -> normalized alias surfaces (avoids repeat queries)
    alias_cache: Dict[int, List[str]] = {}

    for entity_id, canonical_name in entity_targets:
        seeded_entities.append(entity_id)

        # Step 1: Collect candidate surfaces (PEM + entity_aliases)
        candidates = _collect_candidate_surfaces(
            conn, entity_id, canonical_name, alias_scoped,
            debug=debug, alias_cache=alias_cache,
        )
        surface_list = [c[0] for c in candidates]

        if debug:
            print(
                f"[PEM debug] entity {entity_id} ({canonical_name}) → candidates: {surface_list}",
                flush=True,
            )

        # Step 2: Ground surfaces to pages (batched PEM query, fast with indexes)
        surface_pages = _ground_surfaces_to_pages(
            conn, entity_id, surface_list, alias_scoped
        )

        grounded_set = set(surface_pages.keys())
        for s in surface_list:
            if s in grounded_set and s not in surfaces_grounded:
                surfaces_grounded.append(s)
            elif s not in grounded_set and s not in surfaces_hint_only:
                surfaces_hint_only.append(s)

        # Step 3: Round-robin page selection (fair across surfaces)
        selected = _select_pages_round_robin(
            surface_pages, surface_list,
            pages_per_surface=PAGES_PER_SURFACE,
            max_pages_per_entity=MAX_PAGES_PER_ENTITY,
        )

        entity_has_grounded_surfaces = False
        for pid, surface_norm in selected:
            if pid not in all_page_id_set:
                all_page_id_set.add(pid)
                all_page_ids.append(pid)
                page_to_surface[pid] = surface_norm
                if surface_norm not in seeded_surfaces:
                    seeded_surfaces.append(surface_norm)
                entity_has_grounded_surfaces = True

        if not entity_has_grounded_surfaces:
            result.reason_codes.append(f"pem_lane_no_grounded_surfaces:{entity_id}")

        # Step 4: General entity pages fallback (when surface-grounding yields few pages)
        if len(selected) < PAGES_GENERAL_PER_ENTITY:
            general_pages = _get_general_entity_pages(
                conn, entity_id, alias_scoped, all_page_id_set,
                limit=PAGES_GENERAL_PER_ENTITY - len(selected),
            )
            for pid in general_pages:
                if pid not in all_page_id_set:
                    all_page_id_set.add(pid)
                    all_page_ids.append(pid)

    if not all_page_ids:
        result.reason_codes.append("pem_lane_no_pem_coverage")
        result.seeded_entities = seeded_entities
        return result

    # Step D: pages → chunks
    chunk_ids = chunks_for_pages(
        conn, all_page_ids,
        max_chunks_per_page=MAX_CHUNKS_PER_PAGE,
    )

    # Global cap
    cap_hit = len(chunk_ids) > MAX_SEED_CHUNKS_TOTAL
    chunk_ids = chunk_ids[:MAX_SEED_CHUNKS_TOTAL]

    # Build chunk→surface provenance map
    # We need chunk→page mapping to trace back
    chunk_surface_map: Dict[int, str] = {}
    try:
        from retrieval.agent.v10_page_bridge import pages_to_chunks_map
        p2c = pages_to_chunks_map(conn, all_page_ids, max_chunks_per_page=MAX_CHUNKS_PER_PAGE)
        for pid, cids in p2c.items():
            surface = page_to_surface.get(pid, "")
            if surface:
                for cid in cids:
                    if cid in set(chunk_ids) and cid not in chunk_surface_map:
                        chunk_surface_map[cid] = surface
    except Exception as e:
        logger.debug("chunk_surface_map build failed: %s", e)

    # Populate result
    result.chunk_ids = chunk_ids
    result.page_ids = all_page_ids
    result.seeded_surfaces = list(dict.fromkeys(seeded_surfaces))  # dedupe, preserve order
    result.seeded_entities = seeded_entities
    result.surfaces_grounded = list(dict.fromkeys(surfaces_grounded))
    result.surfaces_hint_only = list(dict.fromkeys(surfaces_hint_only))

    if debug:
        print(
            f"[PEM debug] Final seeded_surfaces (all entities): {result.seeded_surfaces}\n"
            f"[PEM debug] Per-entity contribution: entities={seeded_entities}, "
            f"cap_hit={cap_hit}, chunks={len(chunk_ids)}",
            flush=True,
        )
    result.chunk_surface_map = chunk_surface_map
    if not result.reason_codes:
        result.reason_codes.append("pem_lane_ok")
    result.stats = {
        "surfaces_selected": len(seeded_surfaces),
        "pages_selected": len(all_page_ids),
        "chunks_selected": len(chunk_ids),
        "cap_hit": cap_hit,
    }

    logger.info(
        "PEM lane: %d chunks seeded (revision=%s, surfaces=%s, entities=%s)",
        len(chunk_ids),
        result.pem_revision,
        result.seeded_surfaces[:5],
        result.seeded_entities[:5],
    )

    return result


# =============================================================================
# Chunk PEM annotation (page-scoped, model-only)
# =============================================================================

def build_chunk_pem_annotation(
    conn,
    chunk_id: int,
    alias_scoped_collections: Sequence[str] = tuple(ALIAS_SCOPED_COLLECTIONS),
    max_mappings: int = MAX_ANNOTATION_MAPPINGS,
) -> str:
    """Build a page-scoped mention-index annotation for a chunk.

    Returns a model-only annotation block like:
        [MENTION_INDEX page_scoped collection=venona]
        cabin => Office of Strategic Services
        [/MENTION_INDEX]

    Only includes surfaces that map to exactly one entity_id across the
    chunk's page set (unambiguous). Returns empty string if no mappings.
    """
    if not chunk_id:
        return ""

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
        logger.debug("build_chunk_pem_annotation: chunk_pages query failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return ""

    if not page_ids:
        return ""

    # Step 2: Get PEM rows for these pages in alias-scoped collections
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
        logger.debug("build_chunk_pem_annotation: PEM query failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return ""

    if not pem_rows:
        return ""

    # Step 3: Group by surface_norm; keep only unambiguous (exactly one entity_id)
    surface_entities: Dict[str, Set[int]] = {}
    surface_source: Dict[str, str] = {}
    surface_collection: Dict[str, str] = {}
    for surface_norm, entity_id, source, coll_slug in pem_rows:
        if surface_norm not in surface_entities:
            surface_entities[surface_norm] = set()
            surface_source[surface_norm] = source or ""
            surface_collection[surface_norm] = coll_slug
        surface_entities[surface_norm].add(entity_id)
        # Keep most authoritative source
        if _source_rank(source) < _source_rank(surface_source.get(surface_norm)):
            surface_source[surface_norm] = source or ""

    unambiguous: List[Tuple[str, int, str, str]] = []
    for surface_norm, eids in surface_entities.items():
        if len(eids) == 1:
            eid = next(iter(eids))
            unambiguous.append((
                surface_norm,
                eid,
                surface_source.get(surface_norm, ""),
                surface_collection.get(surface_norm, ""),
            ))

    if not unambiguous:
        return ""

    # Step 4: Sort by source_rank then surface_norm; cap
    unambiguous.sort(key=lambda r: (_source_rank(r[2]), r[0]))
    unambiguous = unambiguous[:max_mappings]

    # Step 5: Resolve entity_id → canonical_name
    entity_ids = list({eid for _, eid, _, _ in unambiguous})
    canonical_map: Dict[int, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, canonical_name FROM entities WHERE id = ANY(%s)
            """, (entity_ids,))
            for row in cur.fetchall():
                canonical_map[row[0]] = row[1]
    except Exception as e:
        logger.debug("build_chunk_pem_annotation: entities query failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    # Step 6: Build annotation block
    # Determine collection for header (use most common, or first)
    collections_used = {coll for _, _, _, coll in unambiguous}
    coll_label = sorted(collections_used)[0] if collections_used else "alias_scoped"

    lines: List[str] = []
    lines.append(f"\n[MENTION_INDEX page_scoped collection={coll_label}]")
    for surface_norm, eid, source, coll in unambiguous:
        canonical = canonical_map.get(eid, f"entity_{eid}")
        lines.append(f"{surface_norm} => {canonical}")
    lines.append("[/MENTION_INDEX]")

    return "\n".join(lines)
