"""
V10 Span Enumeration — spot_query_spans_v10().

Deterministic span enumeration with:
- N-gram generation (n=1..5) from query tokens
- Shared normalizer for match keys (v10_normalize)
- Entity/alias lookup with collection scoping + surface_kind
- Synthesized fullname candidates from entity metadata
- Overlap + dominates tracking between spans
- Collision counting (low/med/high)
- Resolution status (resolved/ambiguous/unresolved)
- Lattice size cap to prevent explosion
- No uppercase gate — eligibility by key length + stopword check
- Unresolved referent-likely spans (conservatively emitted)

The SpanLattice is presented to the LLM which selects a non-overlapping
set in Stage A.  This module NEVER makes selection decisions.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from retrieval.agent.v10_normalize import (
    normalize_surface_for_lookup,
    is_stopword_only,
    is_mostly_stopwords,
)
from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    CODENAME_ALIAS_KINDS,
    SpanCandidate,
    SpanEntry,
    SpanLatticeV10,
)
from retrieval.agent.v10_page_bridge import has_page_entity_mentions

logger = logging.getLogger(__name__)

# Max n-gram length (in tokens)
MAX_NGRAM = 5

# Lattice size cap — prevents huge JSONs that degrade agentic quality
MAX_SPANS = 40

# Minimum key length for fuzzy matching
MIN_FUZZY_KEY_LEN = 5

# Cache: does the DB have pg_trgm (for similarity() calls)?
_pg_trgm_available: Optional[bool] = None


def _has_pg_trgm(conn) -> bool:
    """Check once (per process) whether pg_trgm extension is installed."""
    global _pg_trgm_available
    if _pg_trgm_available is not None:
        return _pg_trgm_available
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
            _pg_trgm_available = cur.fetchone() is not None
    except Exception:
        _pg_trgm_available = False
        try:
            conn.rollback()
        except Exception:
            pass
    logger.debug("pg_trgm available: %s", _pg_trgm_available)
    return _pg_trgm_available


def _looks_like_name(text: str) -> bool:
    """Heuristic: does this n-gram look like it could be an entity/alias name?

    Used only as a weak score bonus for candidates, NEVER as a gate.
    """
    words = text.split()
    if not words:
        return False
    return any(w[0].isupper() for w in words if w)


def _has_referent_signal(ngram_text: str, norm_key: str) -> bool:
    """Does this n-gram look referent-likely (worth emitting even without candidates)?

    Criteria (conservative):
    - Multi-token phrase (not mostly stopwords), or
    - Contains apostrophe/hyphen and key length >= 4
    """
    tokens = norm_key.split()
    if len(tokens) >= 2 and not is_mostly_stopwords(norm_key, threshold=0.8):
        return True
    if len(norm_key) >= 4 and ("'" in ngram_text or "-" in ngram_text):
        return True
    return False


def _classify_surface_kind(
    match_type: str, alias_kind: Optional[str]
) -> str:
    """Classify a candidate into surface_kind.

    codename_alias — code_name/covername aliases in alias-scoped corpora
    general_alias  — alt/primary/misspelling/initials aliases (global)
    acronym        — (future: when acronym lexicon exists; for now fold into general_alias)
    name           — canonical name match or synthesized fullname
    phrase         — default/fallback
    """
    if match_type in ("canonical", "fuzzy_fullname", "synthesized_fullname"):
        return "name"
    if alias_kind and alias_kind in CODENAME_ALIAS_KINDS:
        return "codename_alias"
    if match_type in ("alias", "fuzzy_alias", "codename"):
        # codename match_type but alias_kind not in CODENAME_ALIAS_KINDS → general
        if alias_kind and alias_kind in CODENAME_ALIAS_KINDS:
            return "codename_alias"
        return "general_alias"
    return "phrase"


# =============================================================================
# Query tokenisation
# =============================================================================

_TOKEN_RE = re.compile(r"[A-Za-z0-9'\u2019\u0400-\u04FF\-]+")  # word tokens (ASCII + Cyrillic + apostrophe + hyphen)


def _tokenise(query: str) -> List[Tuple[str, int, int]]:
    """Split query into (token_text, char_start, char_end) tuples."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(query)]


# =============================================================================
# N-gram span generation
# =============================================================================

def _generate_ngrams(
    tokens: List[Tuple[str, int, int]],
    query: str,
    max_n: int = MAX_NGRAM,
) -> List[Tuple[str, int, int]]:
    """Generate all n-grams (n=1..max_n) as (text, char_start, char_end).

    The text is taken from the original query (preserving spacing/case).
    """
    ngrams: List[Tuple[str, int, int]] = []
    for n in range(1, min(max_n + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            start = tokens[i][1]
            end = tokens[i + n - 1][2]
            text = query[start:end]
            ngrams.append((text, start, end))
    return ngrams


# =============================================================================
# DB lookups
# =============================================================================

def _lookup_canonical(
    conn, norm_key: str
) -> List[Tuple[int, str, str]]:
    """Exact canonical name lookup using casefolded key.
    Returns [(entity_id, canonical_name, entity_type)]."""
    results = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, canonical_name, entity_type "
                "FROM entities WHERE LOWER(canonical_name) = %s LIMIT 10",
                (norm_key,),
            )
            results = cur.fetchall()
    except Exception as e:
        logger.debug("Canonical lookup error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return results


def _lookup_aliases(
    conn, norm_key: str
) -> List[Tuple[int, str, str, str]]:
    """Exact alias lookup using casefolded key.
    Returns [(entity_id, canonical_name, entity_type, kind)]."""
    results = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT e.id, e.canonical_name, e.entity_type, ea.kind
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE LOWER(ea.alias) = %s
                LIMIT 10
            """, (norm_key,))
            results = cur.fetchall()
    except Exception as e:
        logger.debug("Alias lookup error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return results


def _lookup_pem_candidates(
    conn,
    norm_key: str,
    scope_collections: Optional[List[str]] = None,
) -> List[Tuple[int, str, str, int]]:
    """Look up candidates from page_entity_mentions for short tokens.

    Returns [(entity_id, canonical_name, collection_slug, count_pages)].
    Only used for short tokens (<=4 chars) that wouldn't otherwise get
    candidates from exact alias/canonical lookup.

    V10.2: scope-aware — uses scope_collections when available, falling
    back to ALIAS_SCOPED_COLLECTIONS only when scope implies alias corpora
    are active. Prevents biasing spans toward alias-scoped corpora when
    the user's query is outside that scope.
    """
    target_colls = list(scope_collections) if scope_collections else list(ALIAS_SCOPED_COLLECTIONS)
    results: List[Tuple[int, str, str, int]] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pem.entity_id, e.canonical_name,
                       pem.collection_slug,
                       COUNT(DISTINCT pem.page_id) AS count_pages
                FROM page_entity_mentions pem
                JOIN entities e ON e.id = pem.entity_id
                WHERE pem.surface_norm = %s
                  AND pem.collection_slug = ANY(%s)
                GROUP BY pem.entity_id, e.canonical_name, pem.collection_slug
                ORDER BY count_pages DESC
                LIMIT 10
            """, (norm_key, target_colls))
            results = cur.fetchall()
    except Exception as e:
        logger.debug("PEM candidate lookup error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return results


def resolve_surface_to_entity_ids(
    conn,
    norm_key: str,
    scope_collections: Optional[List[str]] = None,
    max_entities: int = 10,
) -> List[int]:
    """Resolve a normalized surface to entity_ids via canonical, alias, and PEM lookups.

    Public helper for search expansion and other callers that need entity resolution
    without building a full SpanLattice. Returns deduplicated entity_ids in order of
    discovery (canonical first, then aliases, then PEM).

    Args:
        conn: Database connection.
        norm_key: Normalized surface (from normalize_surface_for_lookup).
        scope_collections: Optional list of collection slugs for PEM lookup.
            When None, uses ALIAS_SCOPED_COLLECTIONS.
        max_entities: Maximum number of entity_ids to return (default 10).

    Returns:
        List of entity_ids, deduplicated, ordered by lookup priority.
    """
    seen: Set[int] = set()
    result: List[int] = []

    for entity_id, _, _ in _lookup_canonical(conn, norm_key):
        if entity_id not in seen and len(result) < max_entities:
            seen.add(entity_id)
            result.append(entity_id)

    for entity_id, _, _, _ in _lookup_aliases(conn, norm_key):
        if entity_id not in seen and len(result) < max_entities:
            seen.add(entity_id)
            result.append(entity_id)

    for entity_id, _, _, _ in _lookup_pem_candidates(conn, norm_key, scope_collections):
        if entity_id not in seen and len(result) < max_entities:
            seen.add(entity_id)
            result.append(entity_id)

    return result


def _lookup_pem_prior_counts(
    conn, norm_key: str
) -> Dict[int, Dict[str, int]]:
    """Look up mention-index priors from page_entity_mentions.

    Returns {entity_id: {collection_slug: count_pages}}.
    Attaches as prior to SpanCandidate when available.
    """
    result: Dict[int, Dict[str, int]] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, collection_slug,
                       COUNT(DISTINCT page_id) AS count_pages
                FROM page_entity_mentions
                WHERE surface_norm = %s
                GROUP BY entity_id, collection_slug
            """, (norm_key,))
            for eid, coll, count_pages in cur.fetchall():
                result.setdefault(eid, {})[coll] = count_pages
    except Exception as e:
        logger.debug("PEM prior count lookup error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return result


def _lookup_alias_prior_counts(
    conn, norm_key: str
) -> Dict[int, int]:
    """Look up aggregate prior counts for an alias surface.
    Returns {entity_id: doc_freq} from alias_lexicon_index if available."""
    result: Dict[int, int] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, COALESCE(doc_freq, 0)
                FROM alias_lexicon_index
                WHERE alias_norm = %s
            """, (norm_key,))
            for eid, freq in cur.fetchall():
                result[eid] = freq
    except Exception as e:
        logger.debug("Prior count lookup error (alias_lexicon_index may not exist): %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return result


def _lookup_fuzzy_canonical(
    conn, norm_key: str, threshold: float = 0.6
) -> List[Tuple[int, str, str, float]]:
    """Fuzzy canonical name lookup using trigram similarity.
    Returns [(entity_id, canonical_name, entity_type, similarity)]."""
    results = []
    if len(norm_key) < 3 or not _has_pg_trgm(conn):
        return results
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, canonical_name, entity_type,
                       similarity(LOWER(canonical_name), %s) AS sim
                FROM entities
                WHERE similarity(LOWER(canonical_name), %s) > %s
                ORDER BY sim DESC
                LIMIT 5
            """, (norm_key, norm_key, threshold))
            results = cur.fetchall()
    except Exception as e:
        logger.debug("Fuzzy canonical lookup error (may lack pg_trgm): %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return results


def _lookup_fuzzy_aliases(
    conn, norm_key: str, threshold: float = 0.6
) -> List[Tuple[int, str, str, str, float]]:
    """Fuzzy alias lookup.
    Returns [(entity_id, canonical_name, entity_type, kind, similarity)]."""
    results = []
    if len(norm_key) < 3 or not _has_pg_trgm(conn):
        return results
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT e.id, e.canonical_name, e.entity_type, ea.kind,
                       similarity(LOWER(ea.alias), %s) AS sim
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                WHERE similarity(LOWER(ea.alias), %s) > %s
                ORDER BY sim DESC
                LIMIT 5
            """, (norm_key, norm_key, threshold))
            results = cur.fetchall()
    except Exception as e:
        logger.debug("Fuzzy alias lookup error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    return results


# =============================================================================
# Synthesized fullname candidates
# =============================================================================

def _parse_name_parts(canonical_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a canonical name into (first_name, last_name).

    Handles patterns like "Julius Rosenberg", "Harry Dexter White", etc.
    Returns (None, None) for organisations or unparseable names.
    """
    parts = canonical_name.strip().split()
    if len(parts) < 2:
        return (None, None)
    return (parts[0], parts[-1])


def _generate_synthesized_candidates(
    conn,
    norm_key: str,
    entity_ids_seen: Set[int],
) -> List[Tuple[int, str, str]]:
    """For multi-word spans, check if they match "First Last" for any
    person entity (even if that exact string isn't in alias table).

    Returns [(entity_id, canonical_name, entity_type)]."""
    results = []
    parts = norm_key.split()
    if len(parts) < 2:
        return results

    first = parts[0]
    last = parts[-1]

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, canonical_name, entity_type
                FROM entities
                WHERE entity_type = 'person'
                  AND LOWER(canonical_name) LIKE %s
                  AND LOWER(canonical_name) LIKE %s
                LIMIT 5
            """, (first + '%', '%' + last))
            for row in cur.fetchall():
                if row[0] not in entity_ids_seen:
                    results.append(row)
    except Exception as e:
        logger.debug("Synthesized fullname lookup error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    return results


# =============================================================================
# Collision counting
# =============================================================================

def _collision_level(count: int) -> str:
    if count <= 1:
        return "low"
    elif count <= 3:
        return "med"
    return "high"


# =============================================================================
# Resolution status
# =============================================================================

def _compute_resolution_status(candidates: List[SpanCandidate]) -> str:
    """Compute resolution status from candidates list."""
    if not candidates:
        return "unresolved"
    if len(candidates) == 1:
        return "resolved"
    return "ambiguous"


# =============================================================================
# Span kind classification
# =============================================================================

def _compute_span_kind(candidates: List[SpanCandidate]) -> str:
    """Derive overall span_kind from candidate surface_kinds."""
    if not candidates:
        return "phrase"
    kinds = {c.surface_kind for c in candidates}
    if "codename_alias" in kinds:
        return "alias_surface"
    if "general_alias" in kinds or "acronym" in kinds:
        return "alias_surface"
    if "name" in kinds:
        return "name"
    return "phrase"


# =============================================================================
# Lattice pruning (cap at MAX_SPANS)
# =============================================================================

def _span_sort_key(span: SpanEntry) -> Tuple:
    """Sort key for lattice pruning: best spans first.

    Priority order:
    1. resolved > ambiguous > unresolved  (lower rank = better)
    2. Longer spans (more tokens)
    3. More candidates
    """
    status_rank = {"resolved": 0, "ambiguous": 1, "unresolved": 2}
    return (
        status_rank.get(span.resolution_status, 3),
        -(span.end - span.start),  # longer is better (negative for ascending sort)
        -len(span.candidates),      # more candidates is better
    )


def _prune_and_recompute(
    spans: List[SpanEntry], max_spans: int = MAX_SPANS
) -> List[SpanEntry]:
    """Prune lattice to max_spans and recompute overlap/dominance."""
    if len(spans) <= max_spans:
        return spans

    # Sort by quality and keep top max_spans
    spans_sorted = sorted(spans, key=_span_sort_key)
    kept = spans_sorted[:max_spans]
    kept_ids = {s.span_id for s in kept}

    # Recompute overlap/dominance on kept set
    for s in kept:
        s.overlaps = [sid for sid in s.overlaps if sid in kept_ids]
        s.dominates = [sid for sid in s.dominates if sid in kept_ids]

    # Re-sort by position for deterministic ordering
    kept.sort(key=lambda s: (s.start, -s.end))
    return kept


# =============================================================================
# Compute overlaps and dominance
# =============================================================================

def _compute_overlaps_dominates(span_list: List[SpanEntry]) -> None:
    """Compute overlap and dominance relationships in place."""
    for i, s in enumerate(span_list):
        s.overlaps = []
        s.dominates = []
    for i, s in enumerate(span_list):
        for j, t in enumerate(span_list):
            if i == j:
                continue
            # s dominates t (s fully contains t)
            if s.start <= t.start and s.end >= t.end and (s.start, s.end) != (t.start, t.end):
                if t.span_id not in s.dominates:
                    s.dominates.append(t.span_id)
            # Partial overlap
            elif s.start < t.end and s.end > t.start and (s.start, s.end) != (t.start, t.end):
                if t.span_id not in s.overlaps:
                    s.overlaps.append(t.span_id)


# =============================================================================
# Main entry point
# =============================================================================

def spot_query_spans_v10(
    conn,
    query: str,
    scope_hint: Optional[List[str]] = None,
) -> SpanLatticeV10:
    """Deterministically enumerate spans and candidates from a query.

    Algorithm:
    1. Tokenise query, generate n-grams (n=1..5)
    2. Normalize each n-gram via shared normalizer (casefold, possessive strip, etc.)
    3. For each n-gram:
       a. Exact canonical + alias lookup (eligible unless stopword-only)
       b. Fuzzy matching (eligible only when len(key) >= MIN_FUZZY_KEY_LEN, not stopword)
       c. Synthesized fullname for multi-word spans
    4. Set surface_kind, valid_collections, prior_count per candidate
    5. Emit unresolved spans for referent-likely n-grams even without candidates
    6. Compute collision levels, resolution_status, span_kind
    7. Compute overlaps + dominates
    8. Prune to MAX_SPANS and recompute links

    Args:
        conn: Database connection
        query: User's query text
        scope_hint: Optional list of collection_slugs in the user's active scope.
            Used to set resolution_status for codename_alias when scope has no
            alias-scoped collections.

    Returns: SpanLatticeV10 (no decisions — selection is agentic)
    """
    tokens = _tokenise(query)
    if not tokens:
        return SpanLatticeV10(query=query)

    ngrams = _generate_ngrams(tokens, query)

    # Determine if active scope includes alias-scoped collections
    scope_has_alias = True  # default: assume alias scope is reachable
    if scope_hint is not None:
        scope_has_alias = bool(set(scope_hint) & ALIAS_SCOPED_COLLECTIONS)

    # Check if page_entity_mentions is available (for short-token lookup + priors)
    pem_available = False
    try:
        pem_available = has_page_entity_mentions(conn)
    except Exception:
        pass

    # Build spans: (start, end) -> SpanEntry
    span_map: Dict[Tuple[int, int], SpanEntry] = {}

    for ngram_text, start, end in ngrams:
        # Compute normalized match key using shared normalizer
        norm_key = normalize_surface_for_lookup(ngram_text)
        if not norm_key:
            continue

        # Skip stopword-only single-token n-grams
        word_count = len(norm_key.split())
        if word_count == 1 and is_stopword_only(norm_key):
            continue

        candidates: List[SpanCandidate] = []
        entity_ids_seen: Set[int] = set()
        is_short = len(norm_key) <= 4

        # --- Exact canonical name matches ---
        # Eligible: always (unless stopword-only, already skipped above)
        for entity_id, canonical, etype in _lookup_canonical(conn, norm_key):
            if entity_id in entity_ids_seen:
                continue
            entity_ids_seen.add(entity_id)
            candidates.append(SpanCandidate(
                entity_id=entity_id,
                canonical_name=canonical,
                match_type="canonical",
                surface_kind="name",
                valid_collections=["*"],
                source="canonical_table",
                score=1.0,
            ))

        # --- Exact alias matches ---
        # Eligible: always (unless stopword-only, already skipped)
        for entity_id, canonical, etype, kind in _lookup_aliases(conn, norm_key):
            if entity_id in entity_ids_seen:
                continue
            entity_ids_seen.add(entity_id)
            is_codename = kind in CODENAME_ALIAS_KINDS
            surface_kind = _classify_surface_kind(
                "codename" if is_codename else "alias", kind
            )
            valid_colls = list(ALIAS_SCOPED_COLLECTIONS) if is_codename else ["*"]
            candidates.append(SpanCandidate(
                entity_id=entity_id,
                canonical_name=canonical,
                match_type="codename" if is_codename else "alias",
                alias_type=kind,
                surface_kind=surface_kind,
                valid_collections=valid_colls,
                source="alias_table",
                score=0.9 if is_codename else 0.85,
            ))

        # --- Fuzzy matching ---
        # Eligibility: len(key) >= MIN_FUZZY_KEY_LEN, not stopword-only, no short tokens
        # No uppercase gate (Refinements 1.2): uppercase is only a weak score bonus
        if not candidates and not is_short and len(norm_key) >= MIN_FUZZY_KEY_LEN and not is_stopword_only(norm_key):
            uppercase_bonus = 0.05 if _looks_like_name(ngram_text) else 0.0

            for entity_id, canonical, etype, sim in _lookup_fuzzy_canonical(conn, norm_key):
                if entity_id in entity_ids_seen:
                    continue
                entity_ids_seen.add(entity_id)
                candidates.append(SpanCandidate(
                    entity_id=entity_id,
                    canonical_name=canonical,
                    match_type="fuzzy_fullname",
                    surface_kind="name",
                    valid_collections=["*"],
                    edit_distance=int((1.0 - sim) * len(norm_key)),
                    source="canonical_table",
                    score=0.6 + (sim * 0.2) + uppercase_bonus,
                ))

            for entity_id, canonical, etype, kind, sim in _lookup_fuzzy_aliases(conn, norm_key):
                if entity_id in entity_ids_seen:
                    continue
                entity_ids_seen.add(entity_id)
                is_codename = kind in CODENAME_ALIAS_KINDS
                surface_kind = _classify_surface_kind("fuzzy_alias", kind)
                valid_colls = list(ALIAS_SCOPED_COLLECTIONS) if is_codename else ["*"]
                candidates.append(SpanCandidate(
                    entity_id=entity_id,
                    canonical_name=canonical,
                    match_type="fuzzy_alias",
                    alias_type=kind,
                    surface_kind=surface_kind,
                    valid_collections=valid_colls,
                    edit_distance=int((1.0 - sim) * len(norm_key)),
                    source="alias_table",
                    score=0.5 + (sim * 0.2) + uppercase_bonus,
                ))

        # --- Short token PEM candidates (Phase F5) ---
        # For short tokens (<=4 chars) without candidates, check page_entity_mentions.
        # V10.2: scope-aware — only query collections in the active scope.
        # If scope has no alias-scoped collections, skip entirely to avoid bias.
        if not candidates and is_short and pem_available and scope_has_alias:
            # Use scope_hint when available to restrict PEM lookup;
            # intersect with ALIAS_SCOPED_COLLECTIONS for safety.
            pem_scope: Optional[List[str]] = None
            if scope_hint is not None:
                pem_scope = list(set(scope_hint) & ALIAS_SCOPED_COLLECTIONS)
                if not pem_scope:
                    pem_scope = None  # scope has no alias colls; skip
            for eid, canonical, coll, count_pages in _lookup_pem_candidates(
                conn, norm_key, scope_collections=pem_scope,
            ):
                if eid in entity_ids_seen:
                    continue
                entity_ids_seen.add(eid)
                candidates.append(SpanCandidate(
                    entity_id=eid,
                    canonical_name=canonical,
                    match_type="codename",
                    surface_kind="codename_alias",
                    valid_collections=[coll],
                    source="page_entity_mentions",
                    score=0.85,
                    prior_count_global=count_pages,
                    prior_count_by_collection={coll: count_pages},
                ))

        # --- Synthesized fullname candidates (multi-word only) ---
        if len(norm_key.split()) >= 2:
            for entity_id, canonical, etype in _generate_synthesized_candidates(
                conn, norm_key, entity_ids_seen
            ):
                entity_ids_seen.add(entity_id)
                # Score: synthesized fullname gets a boost if both first+last matched
                candidates.append(SpanCandidate(
                    entity_id=entity_id,
                    canonical_name=canonical,
                    match_type="synthesized_fullname",
                    surface_kind="name",
                    valid_collections=["*"],
                    source="entity_metadata",
                    score=0.95,  # Strong signal: both given+surname matched
                ))

        # --- Prior counts (best-effort from alias_lexicon_index + PEM) ---
        if candidates:
            prior_counts = _lookup_alias_prior_counts(conn, norm_key)
            for c in candidates:
                if not c.prior_count_global:
                    c.prior_count_global = prior_counts.get(c.entity_id, 0)

            # Augment with PEM priors when available (Phase F5)
            if pem_available:
                pem_priors = _lookup_pem_prior_counts(conn, norm_key)
                for c in candidates:
                    eid_priors = pem_priors.get(c.entity_id, {})
                    if eid_priors:
                        c.prior_count_by_collection.update(eid_priors)
                        total = sum(eid_priors.values())
                        if total > c.prior_count_global:
                            c.prior_count_global = total
                        # Set valid_collections from PEM when codename_alias
                        if c.surface_kind == "codename_alias" and c.valid_collections == ["*"]:
                            c.valid_collections = list(eid_priors.keys())

        # --- Scope-aware status for codename aliases (Refinements 14, 20) ---
        # If active scope has no alias-scoped collections, mark codename candidates
        # as inactive so the model doesn't accidentally boost them
        if not scope_has_alias:
            for c in candidates:
                if c.surface_kind == "codename_alias":
                    # Keep the candidate but signal it's out-of-scope
                    # (resolution_status will be set below; valid_collections stays scoped)
                    pass  # resolution_status handled at span level

        # --- Collision level ---
        if candidates:
            collision = _collision_level(len(candidates))
            for c in candidates:
                c.collision = collision

        # --- Resolution status and span kind ---
        resolution_status = _compute_resolution_status(candidates)
        span_kind = _compute_span_kind(candidates)

        # If all candidates are codename_alias and scope has no alias collections,
        # downgrade to unresolved (Refinements 20)
        if not scope_has_alias and candidates:
            all_codename = all(c.surface_kind == "codename_alias" for c in candidates)
            if all_codename:
                resolution_status = "unresolved"

        # --- Create span (with candidates OR if referent-likely) ---
        should_emit = bool(candidates) or _has_referent_signal(ngram_text, norm_key)
        if should_emit:
            span_id = f"sp_{start}_{end}"
            span_map[(start, end)] = SpanEntry(
                span_id=span_id,
                text=ngram_text,
                start=start,
                end=end,
                norm_key=norm_key,
                span_kind=span_kind,
                resolution_status=resolution_status,
                candidates=candidates,
            )

    # --- Compute overlaps + dominates ---
    span_list = list(span_map.values())
    _compute_overlaps_dominates(span_list)

    # --- Sort by position (longer spans first at same start) ---
    span_list.sort(key=lambda s: (s.start, -s.end))

    # --- Prune to MAX_SPANS ---
    span_list = _prune_and_recompute(span_list, MAX_SPANS)

    return SpanLatticeV10(query=query, spans=span_list)
