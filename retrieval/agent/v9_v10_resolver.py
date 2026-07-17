"""
V9 <- V10 Entity Resolver Bridge.

Uses V10's span lattice (spot_query_spans_v10) for entity resolution when
V9 needs to map query keywords to entities. V10's resolver combines:
- Exact canonical + alias lookup
- Fuzzy matching
- PEM candidates with prior_count_by_collection
- Synthesized fullname for multi-word spans

When scope includes venona/vassiliev, we prefer candidates with highest
PEM prior_count in those collections (breaks OSS -> OSS veterans org vs
Office of Strategic Services).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from retrieval.agent.v10_normalize import normalize_surface_for_lookup
from retrieval.agent.v10_spans import spot_query_spans_v10
from retrieval.agent.v10_types import ALIAS_SCOPED_COLLECTIONS

logger = logging.getLogger(__name__)


def resolve_keywords_via_v10_spans(
    conn,
    query: str,
    keywords: List[str],
    scope_collections: Optional[List[str]] = None,
) -> Dict[str, Tuple[int, str]]:
    """
    Resolve content keywords to (entity_id, canonical_name) using V10's span lattice.

    For each keyword, finds the matching span and picks the best candidate.
    When scope includes venona/vassiliev, prefers the entity with highest
    prior_count in those collections (PEM-derived, breaks alias collisions).

    Returns: {keyword: (entity_id, canonical_name)} for keywords that resolved.
    """
    if not query or not keywords:
        return {}

    # Use alias-scoped when scope empty (default for PEM)
    if scope_collections:
        scope_hint = list(set(scope_collections) | ALIAS_SCOPED_COLLECTIONS)
    else:
        scope_hint = list(ALIAS_SCOPED_COLLECTIONS)

    try:
        lattice = spot_query_spans_v10(conn, query, scope_hint=scope_hint)
    except Exception as e:
        logger.warning("[V9-V10] spot_query_spans_v10 failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return {}

    alias_colls = set(ALIAS_SCOPED_COLLECTIONS)
    result: Dict[str, Tuple[int, str]] = {}

    for kw in keywords:
        kw = kw.strip()
        if not kw or len(kw) < 2:
            continue
        norm_kw = normalize_surface_for_lookup(kw)
        if not norm_kw:
            continue

        # Find best matching span: prefer exact norm_key, then longest containing match
        best_span = None
        for span in lattice.spans:
            if span.norm_key == norm_kw:
                best_span = span
                break
        if best_span is None:
            for span in lattice.spans:
                if norm_kw in span.norm_key or span.norm_key in norm_kw:
                    if best_span is None or len(span.norm_key) > len(best_span.norm_key):
                        best_span = span

        if not best_span or not best_span.candidates:
            continue

        # Pick best candidate: when alias-scoped in scope, prefer PEM prior in venona/vassiliev
        def _alias_prior(c) -> int:
            return sum(
                c.prior_count_by_collection.get(coll, 0)
                for coll in alias_colls
            )

        def _score_key(c):
            return (
                _alias_prior(c),  # higher PEM prior in alias colls first
                c.score,
                c.prior_count_global,
            )

        best = max(best_span.candidates, key=_score_key)
        result[kw] = (best.entity_id, best.canonical_name or "")

    return result
