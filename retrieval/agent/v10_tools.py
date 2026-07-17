"""
V10 Tools — Resolver, Alias-Index, and Permission tools (mention index).

Phase 2 tools:
  resolve_referent_v10     — read-only candidate expansion from entity/alias tables
  alias_index_summary_v10  — stats from mention index (aggregate or occurrence)
  alias_index_lookup_v10   — occurrence-level lookup with chunk_ids
  aliases_for_entity_v10   — reverse: entity_id → alias surfaces
  alias_index_sample_v10   — balanced sample across entity_ids

Phase 3 tools:
  grant_alias_power_v10    — index-backed permission granting
  surface_top_referent_v10 — one-call top-referent helper

All tools use page_entity_mentions as the primary truth substrate when
the table is populated, with fallback to entity_mentions / entity_aliases.

Tool contract (V10.2):
  Every mention-index tool returns:
    surface_raw            — the input surface exactly as provided
    surface_norm_used      — the normalized form actually used in lookups
    effective_collections  — collections the query was scoped to
    index_revision_used    — PEM revision for cache-busting / audit

  Truth-level semantics:
    Summary calculations aggregate ACROSS truth levels per entity.
    Reports best_truth_level per entity (authoritative > concordance > derived).

  Fallback telemetry:
    Each fallback path increments a counter (surface_summary_fallback_used,
    surface_lookup_fallback_used, aliases_for_entity_fallback_used,
    grant_alias_power_fallback_used). Use get_telemetry() to inspect.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from retrieval.agent.v10_normalize import (
    normalize_surface_for_lookup,
    normalize_alias_surface,
    is_stopword_only,
)
from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    CODENAME_ALIAS_KINDS,
    LexiconV10,
)
from retrieval.agent.v10_page_bridge import (
    chunks_for_pages,
    get_index_revision,
    has_page_entity_mentions,
)

logger = logging.getLogger(__name__)

# Telemetry counters (module-level; reset per process or query as needed)
_telemetry: Dict[str, int] = {}


def _incr(key: str) -> None:
    """Increment a telemetry counter."""
    _telemetry[key] = _telemetry.get(key, 0) + 1
    logger.debug("telemetry: %s = %d", key, _telemetry[key])


def get_telemetry() -> Dict[str, int]:
    return dict(_telemetry)


def reset_telemetry() -> None:
    _telemetry.clear()


# =============================================================================
# Helper: surface_kind classification
# =============================================================================

def _classify_kind(kind: Optional[str]) -> str:
    """Classify an entity_aliases.kind value into surface_kind."""
    if kind and kind in CODENAME_ALIAS_KINDS:
        return "codename_alias"
    if kind in ("primary", "alt", "misspelling", "ru_translit"):
        return "general_alias"
    if kind == "initials":
        return "acronym"
    return "general_alias"


# =============================================================================
# Helper: scope echo
# =============================================================================

def _scope_echo(conn, collections: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build the scope echo dict included in every tool response."""
    revision = get_index_revision(conn)
    return {
        "effective_collections": collections or [],
        "index_revision_used": revision,
    }


# =============================================================================
# Helper: determine if we can use page_entity_mentions
# =============================================================================

def _use_pem(conn) -> bool:
    """Check if page_entity_mentions exists and has rows."""
    return has_page_entity_mentions(conn)


# =============================================================================
# Tool A: resolve_referent_v10
# =============================================================================

def tool_resolve_referent(
    conn,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Read-only candidate expansion from canonical/alias/acronym namespace.

    Args (from tool call):
        surface_text: str — the surface to resolve
        mode: "strict" | "broad" — strict = exact only; broad = adds fuzzy
        scope_hint: list[str] | None — optional collection list

    Returns JSON with candidates, each labeled with surface_kind and source.
    Includes ambiguity flags per plan C1.
    """
    surface_text = args.get("surface_text", "")
    mode = args.get("mode", "strict")
    scope_hint = args.get("scope_hint")

    if not surface_text:
        return {"error": "surface_text is required"}

    norm_key = normalize_surface_for_lookup(surface_text)
    if not norm_key:
        return {"error": "surface_text normalizes to empty", "norm_key": ""}

    candidates: List[Dict[str, Any]] = []
    entity_ids_seen: Set[int] = set()

    # --- Exact canonical ---
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, canonical_name, entity_type "
                "FROM entities WHERE LOWER(canonical_name) = %s LIMIT 10",
                (norm_key,),
            )
            for entity_id, canonical, etype in cur.fetchall():
                if entity_id in entity_ids_seen:
                    continue
                entity_ids_seen.add(entity_id)
                candidates.append({
                    "entity_id": entity_id,
                    "canonical_name": canonical,
                    "entity_type": etype,
                    "surface_kind": "name",
                    "source": "canonical_table",
                    "match_type": "canonical",
                })
    except Exception as e:
        logger.debug("resolve_referent canonical error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    # --- Exact alias ---
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT e.id, e.canonical_name, e.entity_type, ea.kind
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE LOWER(ea.alias) = %s
                LIMIT 10
            """, (norm_key,))
            for entity_id, canonical, etype, kind in cur.fetchall():
                if entity_id in entity_ids_seen:
                    continue
                entity_ids_seen.add(entity_id)
                candidates.append({
                    "entity_id": entity_id,
                    "canonical_name": canonical,
                    "entity_type": etype,
                    "surface_kind": _classify_kind(kind),
                    "source": "alias_table",
                    "match_type": "alias",
                    "alias_kind": kind,
                })
    except Exception as e:
        logger.debug("resolve_referent alias error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    # --- Fuzzy (broad mode only, key length >= 5) ---
    if mode == "broad" and len(norm_key) >= 5 and not is_stopword_only(norm_key):
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, canonical_name, entity_type,
                           similarity(LOWER(canonical_name), %s) AS sim
                    FROM entities
                    WHERE similarity(LOWER(canonical_name), %s) > 0.5
                    ORDER BY sim DESC LIMIT 5
                """, (norm_key, norm_key))
                for entity_id, canonical, etype, sim in cur.fetchall():
                    if entity_id in entity_ids_seen:
                        continue
                    entity_ids_seen.add(entity_id)
                    candidates.append({
                        "entity_id": entity_id,
                        "canonical_name": canonical,
                        "entity_type": etype,
                        "surface_kind": "name",
                        "source": "canonical_table",
                        "match_type": "fuzzy",
                        "similarity": round(sim, 3),
                    })
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT e.id, e.canonical_name, e.entity_type, ea.kind,
                           similarity(LOWER(ea.alias), %s) AS sim
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE similarity(LOWER(ea.alias), %s) > 0.5
                    ORDER BY sim DESC LIMIT 5
                """, (norm_key, norm_key))
                for entity_id, canonical, etype, kind, sim in cur.fetchall():
                    if entity_id in entity_ids_seen:
                        continue
                    entity_ids_seen.add(entity_id)
                    candidates.append({
                        "entity_id": entity_id,
                        "canonical_name": canonical,
                        "entity_type": etype,
                        "surface_kind": _classify_kind(kind),
                        "source": "alias_table",
                        "match_type": "fuzzy_alias",
                        "alias_kind": kind,
                        "similarity": round(sim, 3),
                    })
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

    # Ambiguity flags (Phase C1)
    ambiguity = "none"
    ambiguity_score = 0.0
    if len(candidates) == 0:
        ambiguity = "no_match"
    elif len(candidates) > 1:
        ambiguity = "ambiguous"
        ambiguity_score = 1.0 - (1.0 / len(candidates))

    result = {
        "surface_raw": surface_text,
        "surface_norm_used": norm_key,
        "norm_key": norm_key,
        "candidates": candidates,
        "total_candidates": len(candidates),
        "ambiguity": ambiguity,
        "ambiguity_score": round(ambiguity_score, 3),
    }
    result.update(_scope_echo(conn, scope_hint))
    return result


# =============================================================================
# Tool B: alias_index_summary_v10 (surface_summary_v10)
# =============================================================================

def tool_alias_index_summary(
    conn,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Corpus-grounded ambiguity check for a surface.

    Uses page_entity_mentions as the primary source with page-count semantics
    (count_pages = count(distinct page_id)).  Falls back to entity_mentions
    if PEM is empty.

    V10.2 truth-level semantics: aggregates ACROSS truth levels per entity
    to avoid splitting the same entity across rows. Reports best_truth_level
    per entity (authoritative > concordance > derived).

    Args:
        alias_surface: str
        collections: list[str] | None

    Returns JSON with entities, count_pages, ambiguity, scope echo,
    surface_raw, surface_norm_used.
    """
    alias_surface = args.get("alias_surface", "")
    collections = args.get("collections")

    norm = normalize_alias_surface(alias_surface)
    if not norm:
        return {"error": "alias_surface normalizes to empty"}

    scope = _scope_echo(conn, collections)
    use_pem = _use_pem(conn)

    entities: List[Dict[str, Any]] = []
    index_truth_level = "derived"

    if use_pem:
        # --- Primary: page_entity_mentions ---
        # Aggregate ACROSS truth levels per entity (V10.2 fix).
        # Compute best_truth_level per entity separately.
        try:
            with conn.cursor() as cur:
                coll_filter = ""
                params: list = [norm]
                if collections:
                    coll_filter = " AND pem.collection_slug = ANY(%s)"
                    params.append(collections)
                cur.execute(f"""
                    SELECT pem.entity_id, e.canonical_name,
                           COUNT(DISTINCT pem.page_id) AS count_pages,
                           COUNT(*) AS count_rows,
                           ARRAY_AGG(DISTINCT pem.collection_slug) AS top_collections,
                           CASE
                             WHEN BOOL_OR(pem.truth_level = 'authoritative') THEN 'authoritative'
                             WHEN BOOL_OR(pem.truth_level = 'concordance') THEN 'concordance'
                             ELSE 'derived'
                           END AS best_truth_level
                    FROM page_entity_mentions pem
                    JOIN entities e ON e.id = pem.entity_id
                    WHERE pem.surface_norm = %s{coll_filter}
                    GROUP BY pem.entity_id, e.canonical_name
                    ORDER BY count_pages DESC
                    LIMIT 20
                """, params)
                for eid, canonical, count_pages, count_rows, top_colls, best_truth in cur.fetchall():
                    entities.append({
                        "entity_id": eid,
                        "canonical_name": canonical,
                        "count_pages": count_pages,
                        "count_rows": count_rows,
                        "top_collections": top_colls or [],
                        "best_truth_level": best_truth,
                    })
                    if best_truth == "authoritative":
                        index_truth_level = "authoritative"
        except Exception as e:
            logger.debug("alias_index_summary PEM query failed: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    if not entities:
        # --- Fallback: entity_mentions ---
        _incr("surface_summary_fallback_used")
        try:
            with conn.cursor() as cur:
                coll_filter = ""
                params = [norm]
                if collections:
                    coll_filter = " AND cm.collection_slug = ANY(%s)"
                    params.append(collections)
                cur.execute(f"""
                    SELECT em.entity_id, e.canonical_name,
                           cm.collection_slug,
                           COUNT(*) as cnt
                    FROM entity_mentions em
                    JOIN entities e ON e.id = em.entity_id
                    JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
                    WHERE LOWER(em.surface) = %s{coll_filter}
                    GROUP BY em.entity_id, e.canonical_name, cm.collection_slug
                    ORDER BY cnt DESC
                    LIMIT 20
                """, params)
                rows = cur.fetchall()
                if rows:
                    index_truth_level = "derived"
                    # Aggregate by entity
                    entity_agg: Dict[int, Dict] = {}
                    for eid, canonical, coll, cnt in rows:
                        if eid not in entity_agg:
                            entity_agg[eid] = {
                                "entity_id": eid,
                                "canonical_name": canonical,
                                "count_pages": 0,  # approx from chunks
                                "count_rows": 0,
                                "top_collections": [],
                                "best_truth_level": "derived",
                            }
                        entity_agg[eid]["count_rows"] += cnt
                        entity_agg[eid]["count_pages"] += cnt  # rough approx
                        if coll and coll not in entity_agg[eid]["top_collections"]:
                            entity_agg[eid]["top_collections"].append(coll)
                    entities = sorted(entity_agg.values(),
                                      key=lambda x: -x["count_pages"])
        except Exception as e:
            logger.debug("alias_index_summary fallback error: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    # --- Fallback: alias_lexicon_index ---
    if not entities:
        _incr("surface_summary_fallback_used")
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ali.entity_id, e.canonical_name,
                           ali.doc_freq, ali.mention_count
                    FROM alias_lexicon_index ali
                    JOIN entities e ON e.id = ali.entity_id
                    WHERE ali.alias_norm = %s
                    ORDER BY ali.doc_freq DESC
                    LIMIT 10
                """, (norm,))
                for eid, canonical, doc_freq, mention_count in cur.fetchall():
                    entities.append({
                        "entity_id": eid,
                        "canonical_name": canonical,
                        "count_pages": doc_freq or 0,
                        "count_rows": mention_count or 0,
                        "top_collections": [],
                        "best_truth_level": "partial",
                    })
                if entities:
                    index_truth_level = "partial"
        except Exception as e:
            logger.debug("alias_index_summary fallback error: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    # Compute ambiguity
    total_pages = sum(e["count_pages"] for e in entities)
    unique_entities = len(entities)
    ambiguity_score = 0.0
    top_share = 0.0
    if unique_entities > 1 and total_pages > 0:
        top_share = entities[0]["count_pages"] / total_pages
        ambiguity_score = 1.0 - top_share
    elif unique_entities == 1:
        top_share = 1.0

    result = {
        "surface_raw": alias_surface,
        "surface_norm_used": norm,
        "surface_norm": norm,
        "entities": entities[:10],
        "total_pages": total_pages,
        "unique_entities": unique_entities,
        "ambiguity_score": round(ambiguity_score, 3),
        "top_share": round(top_share, 3),
        "index_truth_level": index_truth_level,
    }
    result.update(scope)
    return result


# =============================================================================
# Tool C: alias_index_lookup_v10 (surface_lookup_v10)
# =============================================================================

def tool_alias_index_lookup(
    conn,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Return citeable evidence locations from page_entity_mentions.

    Primary: query pages from page_entity_mentions, deterministic sampling,
    convert pages → chunk_ids via chunks_for_pages bridge.

    Sampling rules (fully deterministic):
    - Order by (document_id, page_id) deterministically
    - Per-doc cap first, then global cap (top_n)
    - No random sampling

    Args:
        alias_surface: str
        collections: list[str] | None
        entity_id: int | None — filter to specific entity
        top_n: int (default 10)
        per_doc_limit: int (default 3)

    Returns occurrences with document_id, page_id, collection_slug,
    entity_id, chunk_ids[], and scope echo.
    """
    alias_surface = args.get("alias_surface", "")
    collections = args.get("collections")
    entity_id_filter = args.get("entity_id")
    top_n = min(args.get("top_n", 10), 30)
    per_doc_limit = min(args.get("per_doc_limit", 3), 10)

    norm = normalize_alias_surface(alias_surface)
    if not norm:
        return {"error": "alias_surface normalizes to empty"}

    scope = _scope_echo(conn, collections)
    use_pem = _use_pem(conn)

    occurrences: List[Dict[str, Any]] = []

    if use_pem:
        # --- Primary: page_entity_mentions ---
        try:
            with conn.cursor() as cur:
                conditions = ["pem.surface_norm = %s"]
                params: list = [norm]
                if collections:
                    conditions.append("pem.collection_slug = ANY(%s)")
                    params.append(collections)
                if entity_id_filter:
                    conditions.append("pem.entity_id = %s")
                    params.append(entity_id_filter)
                where = " AND ".join(conditions)

                # Deterministic ordering: document_id ASC, page_id ASC
                cur.execute(f"""
                    SELECT pem.entity_id, e.canonical_name,
                           pem.collection_slug, pem.document_id,
                           pem.page_id,
                           p.pdf_page_number, p.logical_page_label,
                           pem.truth_level
                    FROM page_entity_mentions pem
                    JOIN entities e ON e.id = pem.entity_id
                    JOIN pages p ON p.id = pem.page_id
                    WHERE {where}
                    ORDER BY pem.document_id ASC, pem.page_id ASC
                """, params)

                # Apply per-doc cap then global cap
                doc_counts: Dict[int, int] = {}
                page_ids_for_bridge: List[int] = []
                raw_rows = []

                for row in cur.fetchall():
                    eid, canonical, coll, doc_id, page_id, pdf_page, label, truth = row
                    count = doc_counts.get(doc_id, 0)
                    if count >= per_doc_limit:
                        continue
                    doc_counts[doc_id] = count + 1
                    raw_rows.append(row)
                    page_ids_for_bridge.append(page_id)
                    if len(raw_rows) >= top_n:
                        break

                # Bridge: page_ids → chunk_ids (V10.2: prefer supporting chunks)
                from retrieval.agent.v10_page_bridge import pages_to_chunks_map
                page_chunk_map = pages_to_chunks_map(
                    conn, page_ids_for_bridge,
                    prefer_entity_id=entity_id_filter,
                )

                for row in raw_rows:
                    eid, canonical, coll, doc_id, page_id, pdf_page, label, truth = row
                    chunk_ids = page_chunk_map.get(page_id, [])
                    occurrences.append({
                        "entity_id": eid,
                        "canonical_name": canonical,
                        "collection_slug": coll,
                        "document_id": doc_id,
                        "page_id": page_id,
                        "page_display": label or (f"p{pdf_page}" if pdf_page else "?"),
                        "chunk_ids": chunk_ids,
                        "truth_level": truth,
                    })
        except Exception as e:
            logger.debug("alias_index_lookup PEM query failed: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    if not occurrences:
        # --- Fallback: entity_mentions ---
        _incr("surface_lookup_fallback_used")
        try:
            with conn.cursor() as cur:
                conditions = ["LOWER(em.surface) = %s"]
                params = [norm]
                if collections:
                    conditions.append("cm.collection_slug = ANY(%s)")
                    params.append(collections)
                if entity_id_filter:
                    conditions.append("em.entity_id = %s")
                    params.append(entity_id_filter)
                where = " AND ".join(conditions)

                cur.execute(f"""
                    SELECT em.entity_id, e.canonical_name,
                           cm.collection_slug, cm.document_id,
                           em.chunk_id,
                           cm.first_page_id
                    FROM entity_mentions em
                    JOIN entities e ON e.id = em.entity_id
                    JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
                    WHERE {where}
                    ORDER BY cm.document_id, em.chunk_id
                    LIMIT %s
                """, params + [top_n * 3])

                doc_counts = {}
                for eid, canonical, coll, doc_id, chunk_id, page_id in cur.fetchall():
                    count = doc_counts.get(doc_id, 0)
                    if count >= per_doc_limit:
                        continue
                    doc_counts[doc_id] = count + 1
                    occurrences.append({
                        "entity_id": eid,
                        "canonical_name": canonical,
                        "collection_slug": coll,
                        "document_id": doc_id,
                        "page_id": page_id,
                        "chunk_ids": [chunk_id],
                        "truth_level": "derived",
                    })
                    if len(occurrences) >= top_n:
                        break
        except Exception as e:
            logger.debug("alias_index_lookup fallback error: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    if not occurrences:
        _incr("surface_lookup_returned_zero")

    result = {
        "surface_raw": alias_surface,
        "surface_norm_used": norm,
        "surface_norm": norm,
        "occurrences": occurrences,
        "total_returned": len(occurrences),
    }
    result.update(scope)
    return result


# =============================================================================
# Tool D: aliases_for_entity_v10
# =============================================================================

def tool_aliases_for_entity(
    conn,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Entity → surfaces with page-count semantics.

    Primary: page_entity_mentions grouped by surface_norm.
    Fallback: entity_aliases + entity_mentions scoped_aliases.

    V10.2: filters "boring" surfaces by default — drops surfaces that are
    just the entity's canonical name/acronym norm, very common stopwords,
    and surfaces with count_pages < min_pages (default 1).

    Args:
        entity_id: int
        collections: list[str] | None — filter to alias-scoped only
        min_pages: int (default 1) — minimum page count for scoped aliases
    """
    entity_id = args.get("entity_id")
    collections = args.get("collections")
    min_pages = args.get("min_pages", 1)

    if not entity_id:
        return {"error": "entity_id is required"}

    scope = _scope_echo(conn, collections)
    use_pem = _use_pem(conn)

    general_aliases: List[Dict[str, Any]] = []
    codename_aliases: List[Dict[str, Any]] = []
    scoped_aliases: List[Dict[str, Any]] = []

    # Get canonical name first (needed for boring-surface filter)
    canonical_name = ""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
            row = cur.fetchone()
            if row:
                canonical_name = row[0]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # Build boring-surface set: canonical name norm, common stopwords
    _boring_norms: Set[str] = set()
    if canonical_name:
        _boring_norms.add(canonical_name.lower().strip())
        # Add individual words if short (acronyms)
        parts = canonical_name.split()
        for p in parts:
            if len(p) <= 3:
                _boring_norms.add(p.lower().strip())
    # Common stopword surfaces that aren't helpful as aliases
    _boring_norms.update({
        "the", "and", "for", "not", "but", "are", "was", "has", "had",
        "his", "her", "him", "she", "who", "that", "this", "from",
    })

    def _is_boring(surface_norm: str) -> bool:
        """Check if a surface is boring (canonical name, stopword, etc.)."""
        return surface_norm.strip().lower() in _boring_norms or is_stopword_only(surface_norm)

    if use_pem:
        # --- Primary: page_entity_mentions ---
        try:
            target_colls = list(collections or ALIAS_SCOPED_COLLECTIONS)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT surface_norm,
                           COUNT(DISTINCT page_id) AS count_pages,
                           COUNT(*) AS count_rows,
                           ARRAY_AGG(DISTINCT collection_slug) AS colls
                    FROM page_entity_mentions
                    WHERE entity_id = %s
                      AND collection_slug = ANY(%s)
                    GROUP BY surface_norm
                    ORDER BY count_pages DESC
                    LIMIT 30
                """, (entity_id, target_colls))
                for surface_norm, count_pages, count_rows, colls in cur.fetchall():
                    # V10.2: filter boring surfaces
                    if _is_boring(surface_norm):
                        continue
                    if count_pages < min_pages:
                        continue
                    scoped_aliases.append({
                        "surface_norm": surface_norm,
                        "count_pages": count_pages,
                        "count_rows": count_rows,
                        "collections": colls or [],
                        "surface_kind": "codename_alias",
                    })
        except Exception as e:
            logger.debug("aliases_for_entity PEM error: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    # --- General aliases from entity_aliases (always useful) ---
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ea.alias, ea.kind, ea.alias_norm
                FROM entity_aliases ea
                WHERE ea.entity_id = %s
                ORDER BY ea.kind, ea.alias
                LIMIT 30
            """, (entity_id,))
            for alias, kind, alias_norm in cur.fetchall():
                # V10.2: filter boring surfaces from general/codename lists
                if _is_boring(alias.lower().strip()):
                    continue
                entry = {
                    "alias": alias,
                    "kind": kind,
                    "surface_kind": _classify_kind(kind),
                }
                if kind in CODENAME_ALIAS_KINDS:
                    codename_aliases.append(entry)
                else:
                    general_aliases.append(entry)
    except Exception as e:
        logger.debug("aliases_for_entity general error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    # --- Fallback scoped_aliases from entity_mentions (if PEM had nothing) ---
    if not scoped_aliases:
        _incr("aliases_for_entity_fallback_used")
        try:
            target_colls = list(collections or ALIAS_SCOPED_COLLECTIONS)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT LOWER(em.surface) AS alias_norm,
                           cm.collection_slug,
                           COUNT(*) AS cnt
                    FROM entity_mentions em
                    JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
                    WHERE em.entity_id = %s
                      AND cm.collection_slug = ANY(%s)
                    GROUP BY alias_norm, cm.collection_slug
                    ORDER BY cnt DESC
                    LIMIT 20
                """, (entity_id, target_colls))
                for alias_norm, coll, cnt in cur.fetchall():
                    if _is_boring(alias_norm):
                        continue
                    scoped_aliases.append({
                        "surface_norm": alias_norm,
                        "count_pages": cnt,  # approx
                        "count_rows": cnt,
                        "collections": [coll],
                        "surface_kind": "codename_alias",
                    })
        except Exception as e:
            logger.debug("aliases_for_entity scoped fallback error: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    result = {
        "entity_id": entity_id,
        "canonical_name": canonical_name,
        "general_aliases": general_aliases[:10],
        "codename_aliases": codename_aliases[:10],
        "scoped_aliases": scoped_aliases[:10],
    }
    result.update(scope)
    return result


# =============================================================================
# Tool: alias_index_sample_v10
# =============================================================================

def tool_alias_index_sample(
    conn,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Balanced sample across entity_ids for an alias surface.

    Wraps lookup with automatic per-entity balancing and doc diversification.

    Args:
        alias_surface: str
        scope: str | None — collection_slug
        per_entity_limit: int (default 3)
        diversify_by_doc: bool (default True)
    """
    alias_surface = args.get("alias_surface", "")
    scope_coll = args.get("scope")
    per_entity_limit = min(args.get("per_entity_limit", 3), 5)

    collections = [scope_coll] if scope_coll else None

    # First get summary to know entity distribution
    summary = tool_alias_index_summary(conn, {
        "alias_surface": alias_surface,
        "collections": collections,
    })

    if summary.get("error"):
        return summary

    # Then get balanced lookup
    lookup = tool_alias_index_lookup(conn, {
        "alias_surface": alias_surface,
        "collections": collections,
        "top_n": per_entity_limit * max(summary.get("unique_entities", 1), 1),
        "per_doc_limit": per_entity_limit,
    })

    return {
        "surface_norm": summary.get("surface_norm", ""),
        "summary": {
            "total_pages": summary.get("total_pages", 0),
            "unique_entities": summary.get("unique_entities", 0),
            "ambiguity_score": summary.get("ambiguity_score", 0),
            "top_share": summary.get("top_share", 0),
        },
        "sample": lookup.get("occurrences", []),
        "index_truth_level": summary.get("index_truth_level", "partial"),
        **_scope_echo(conn, collections),
    }


# =============================================================================
# Tool E: grant_alias_power_v10 (Phase 4.2)
# =============================================================================

# Trapdoor policy for boost-only (coll, alias, None) grants
GRANT_TOP_SHARE_THRESHOLD = 0.90
GRANT_MIN_PAGES_THRESHOLD = 5


def tool_grant_alias_power(
    conn,
    args: Dict[str, Any],
    lexicon: LexiconV10,
    current_round: int = 0,
) -> Dict[str, Any]:
    """Deterministic permission grant.  Only gates privileged alias operations.

    Grant conditions:
    - collection_scope must be alias-scoped
    - page_entity_mentions must have rows for (coll, surface_norm, entity_id)
    - For boost-only (entity_id=None): trapdoor requires top_share >= 0.90
      AND total count_pages >= min_pages

    Args:
        alias_surface: str
        entity_id: int | None — None for boost-only (no lock)
        collection_scope: str
    """
    alias_surface = args.get("alias_surface", "")
    entity_id = args.get("entity_id")
    collection_scope = args.get("collection_scope", "")

    alias_norm = normalize_alias_surface(alias_surface)
    if not alias_norm:
        _incr("grant_alias_power_denied_empty_surface")
        return {"error": "alias_surface normalizes to empty", "granted": False}

    if collection_scope not in ALIAS_SCOPED_COLLECTIONS:
        _incr("grant_alias_power_denied_wrong_scope")
        return {
            "error": f"collection_scope '{collection_scope}' is not alias-scoped",
            "granted": False,
            "reason": "grant_alias_power_denied_wrong_scope",
        }

    use_pem = _use_pem(conn)
    index_truth_level = "derived"
    has_hit = False

    if use_pem:
        # --- Primary: page_entity_mentions ---
        try:
            with conn.cursor() as cur:
                if entity_id is not None:
                    # Entity-specific grant: just need at least one row.
                    # V10.2: if ANY authoritative row exists for this triple,
                    # upgrade to authoritative (even if derived rows also exist).
                    cur.execute("""
                        SELECT COUNT(DISTINCT page_id),
                               BOOL_OR(truth_level = 'authoritative') AS has_authoritative
                        FROM page_entity_mentions
                        WHERE collection_slug = %s
                          AND surface_norm = %s
                          AND entity_id = %s
                    """, (collection_scope, alias_norm, entity_id))
                    row = cur.fetchone()
                    has_hit = row and row[0] > 0
                    if has_hit and row[1]:
                        index_truth_level = "authoritative"
                else:
                    # Boost-only (entity_id=None): trapdoor policy
                    cur.execute("""
                        SELECT entity_id,
                               COUNT(DISTINCT page_id) AS count_pages
                        FROM page_entity_mentions
                        WHERE collection_slug = %s
                          AND surface_norm = %s
                        GROUP BY entity_id
                        ORDER BY count_pages DESC
                    """, (collection_scope, alias_norm))
                    rows = cur.fetchall()
                    if rows:
                        total_pages = sum(r[1] for r in rows)
                        top_entity_pages = rows[0][1]
                        top_share = top_entity_pages / total_pages if total_pages > 0 else 0

                        if (top_share >= GRANT_TOP_SHARE_THRESHOLD
                                and total_pages >= GRANT_MIN_PAGES_THRESHOLD):
                            has_hit = True
                        else:
                            _incr("grant_alias_power_denied_ambiguous_boost_only")
                            return {
                                "granted": False,
                                "alias_norm": alias_norm,
                                "collection_scope": collection_scope,
                                "entity_id": None,
                                "reason": "grant_alias_power_denied_ambiguous_boost_only",
                                "top_share": round(top_share, 3),
                                "total_pages": total_pages,
                                "hint": "Provide entity_id for explicit grant, or alias is too ambiguous for boost-only.",
                            }
        except Exception as e:
            logger.debug("grant_alias_power PEM check error: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    if not has_hit:
        # --- Fallback: entity_mentions ---
        _incr("grant_alias_power_fallback_used")
        try:
            with conn.cursor() as cur:
                conditions = [
                    "LOWER(em.surface) = %s",
                    "cm.collection_slug = %s",
                ]
                params: list = [alias_norm, collection_scope]
                if entity_id is not None:
                    conditions.append("em.entity_id = %s")
                    params.append(entity_id)

                where = " AND ".join(conditions)
                cur.execute(f"""
                    SELECT COUNT(*) FROM entity_mentions em
                    JOIN chunk_metadata cm ON cm.chunk_id = em.chunk_id
                    WHERE {where}
                    LIMIT 1
                """, params)
                row = cur.fetchone()
                has_hit = row and row[0] > 0
                if has_hit:
                    index_truth_level = "derived"
        except Exception as e:
            logger.debug("grant_alias_power fallback error: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    # Fallback: check entity_aliases if entity_mentions doesn't have it
    if not has_hit and entity_id is not None:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1 FROM entity_aliases ea
                    WHERE ea.entity_id = %s
                      AND LOWER(ea.alias) = %s
                      AND ea.kind = ANY(%s)
                    LIMIT 1
                """, (entity_id, alias_norm, list(CODENAME_ALIAS_KINDS)))
                has_hit = cur.fetchone() is not None
                if has_hit:
                    index_truth_level = "partial"
        except Exception as e:
            logger.debug("grant_alias_power alias fallback error: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    if not has_hit:
        _incr("grant_alias_power_denied_no_index_hit")
        return {
            "granted": False,
            "alias_norm": alias_norm,
            "collection_scope": collection_scope,
            "entity_id": entity_id,
            "reason": "grant_alias_power_denied_no_index_hit",
        }

    # Grant
    status = "confirmed" if index_truth_level == "authoritative" else "provisional"
    lexicon.grant_permission(
        collection_slug=collection_scope,
        alias_surface_norm=alias_norm,
        entity_id=entity_id,
        status=status,
        index_truth_level=index_truth_level,
        granted_at_round=current_round,
    )
    _incr("grant_alias_power_success")

    # Determine lock eligibility
    from retrieval.agent.v10_runner import _is_lock_safe
    lock_eligible = False
    if entity_id is not None and status == "confirmed":
        lock_eligible = _is_lock_safe(lexicon, alias_surface, collection_scope, entity_id)

    result = {
        "granted": True,
        "surface_raw": alias_surface,
        "surface_norm_used": alias_norm,
        "alias_norm": alias_norm,
        "collection_scope": collection_scope,
        "entity_id": entity_id,
        "permission_status": status,
        "index_truth_level": index_truth_level,
        "lock_eligible": lock_eligible,
    }
    result.update(_scope_echo(conn, [collection_scope]))
    return result


# =============================================================================
# Tool F: surface_top_referent_v10 (Phase C8)
# =============================================================================

def tool_surface_top_referent(
    conn,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """One-call helper: top referent for a surface in scope.

    Wrapper over surface_summary: compute distribution by count_pages,
    pick top entity and its share.

    Args:
        surface: str
        collections: list[str] | None
        min_share: float (default 0.7)
    """
    surface = args.get("surface", "")
    collections = args.get("collections")
    min_share = args.get("min_share", 0.7)

    summary = tool_alias_index_summary(conn, {
        "alias_surface": surface,
        "collections": collections,
    })

    if summary.get("error"):
        return summary

    entities = summary.get("entities", [])
    if not entities:
        return {
            "surface_raw": surface,
            "surface_norm_used": summary.get("surface_norm", ""),
            "surface_norm": summary.get("surface_norm", ""),
            "entity_id": None,
            "share": 0.0,
            "is_ambiguous": True,
            "reason": "no_entities_found",
            **_scope_echo(conn, collections),
        }

    top = entities[0]
    top_share = summary.get("top_share", 0.0)
    is_ambiguous = top_share < min_share or len(entities) > 1

    return {
        "surface_raw": surface,
        "surface_norm_used": summary.get("surface_norm", ""),
        "surface_norm": summary.get("surface_norm", ""),
        "entity_id": top["entity_id"],
        "canonical_name": top["canonical_name"],
        "share": top_share,
        "is_ambiguous": is_ambiguous,
        "total_entities": len(entities),
        "total_pages": summary.get("total_pages", 0),
        **_scope_echo(conn, collections),
    }
