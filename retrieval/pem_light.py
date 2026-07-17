"""
Lightweight PEM — Evidence-Time Mention Index.

Annotates retrieved evidence with surface→canonical mappings.
No PEM-driven query planning, no span replacement, no "entities in play".
Just: annotate the evidence you already retrieved before sending to the model.

Usage:
  mention_block, manifest = build_mention_index_for_pages(conn, page_ids, ...)
  bundle_text_for_model = bundle_text_original + mention_block
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default denylist: geography + generic org words (conservative)
DEFAULT_DENYLIST = frozenset({
    "moscow", "washington", "soviet", "russia", "usa", "london", "new york",
    "berlin", "paris", "state", "department", "committee", "bureau",
    "american", "government", "intelligence", "agency", "service",
})


def build_mention_index_for_pages(
    conn,
    page_ids: List[int],
    *,
    max_lines: int = 60,
    max_chars: int = 2400,
    max_entities: int = 25,
    max_aliases_per_entity: int = 8,
    max_tokens: Optional[int] = 600,
    dominant_frac: float = 0.8,
    min_surface_len: int = 4,
    allow_entity_types: Optional[Set[str]] = None,
    deny_surfaces: Optional[Set[str]] = None,
    collection_slugs: Tuple[str, ...] = ("venona", "vassiliev"),
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a MENTION_INDEX block for the given pages.

    Returns (mention_block, manifest).
    mention_block is a string like:
      \\n\\n[MENTION_INDEX]\\nSURF => Canon (9/11)\\n...\\n[/MENTION_INDEX]
    manifest includes counts + rules for debugging.
    """
    if allow_entity_types is None:
        allow_entity_types = {"PERSON", "ORG", "CODENAME"}
    deny_surfaces = deny_surfaces or DEFAULT_DENYLIST

    if not page_ids:
        return "", {"page_ids": [], "included": [], "skipped": []}

    # Fetch PEM rows
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pem.surface_norm, pem.entity_id
            FROM page_entity_mentions pem
            WHERE pem.page_id = ANY(%s)
              AND pem.collection_slug = ANY(%s)
        """, (page_ids, list(collection_slugs)))
        rows = cur.fetchall()

    # surface_norm -> entity_id -> count, and total per surface
    surface_entity_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: Counter())
    for surface_norm, entity_id in rows:
        if surface_norm and entity_id:
            surface_entity_counts[surface_norm][entity_id] += 1

    # Fetch canonical names for entity_ids we'll need
    all_entity_ids = set()
    for counts in surface_entity_counts.values():
        all_entity_ids.update(counts.keys())
    entity_id_to_canonical: Dict[int, str] = {}
    entity_id_to_type: Dict[int, Optional[str]] = {}
    if all_entity_ids:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, canonical_name, entity_type
                FROM entities
                WHERE id = ANY(%s)
            """, (list(all_entity_ids),))
            for eid, canonical, etype in cur.fetchall():
                entity_id_to_canonical[eid] = canonical or ""
                entity_id_to_type[eid] = etype

    # Decide mapping per surface. Cap at source: max_entities, max_aliases_per_entity, max_chars, max_tokens.
    included: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    lines: List[str] = []
    char_count = 0
    entity_alias_count: Dict[int, int] = {}
    entities_emitted = 0

    def _estimate_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    token_count = 0

    # Build list of (surface, entity_id, canonical, rule, top, total) for included
    for surface_norm, counts in sorted(
        surface_entity_counts.items(),
        key=lambda x: (-sum(x[1].values()), x[0]),
    ):
        total = sum(counts.values())
        if total == 0:
            continue

        # Filter: min length
        if len(surface_norm) < min_surface_len:
            skipped.append({"surface_norm": surface_norm, "reason": "too_short", "len": len(surface_norm)})
            continue

        # Filter: denylist
        if surface_norm.lower() in deny_surfaces:
            skipped.append({"surface_norm": surface_norm, "reason": "denylist"})
            continue

        # Get top entity
        sorted_ents = sorted(counts.items(), key=lambda x: -x[1])
        top_entity_id, top_count = sorted_ents[0]

        # Entity type filter (if we have entity_type and it's not in allow list, skip)
        etype = entity_id_to_type.get(top_entity_id)
        if etype and allow_entity_types and etype.upper() not in {t.upper() for t in allow_entity_types}:
            skipped.append({"surface_norm": surface_norm, "reason": "entity_type", "entity_type": etype})
            continue

        # Rule: unique or dominant
        if len(counts) == 1:
            rule = "unique"
        elif top_count / total >= dominant_frac:
            rule = "dominant"
        else:
            skipped.append({
                "surface_norm": surface_norm,
                "reason": "ambiguous",
                "counts": {str(k): v for k, v in sorted(counts.items(), key=lambda x: -x[1])},
            })
            continue

        canonical = entity_id_to_canonical.get(top_entity_id, "?")
        # Cap: max aliases per entity
        n_aliases = entity_alias_count.get(top_entity_id, 0)
        if n_aliases >= max_aliases_per_entity:
            skipped.append({"surface_norm": surface_norm, "reason": "max_aliases_per_entity", "entity_id": top_entity_id})
            continue
        # Cap: max entities (count when we'd add a new entity)
        if n_aliases == 0 and entities_emitted >= max_entities:
            skipped.append({"surface_norm": surface_norm, "reason": "max_entities"})
            continue

        included.append({
            "surface_norm": surface_norm,
            "entity_id": top_entity_id,
            "canonical": canonical,
            "rule": rule,
            "top": top_count,
            "total": total,
        })

        # Emit line
        line = f"{surface_norm} => {canonical} ({top_count}/{total})"
        line_tokens = _estimate_tokens(line)
        if char_count + len(line) + 1 > max_chars or len(lines) >= max_lines:
            break
        if max_tokens and token_count + line_tokens > max_tokens:
            break
        lines.append(line)
        char_count += len(line) + 1
        token_count += line_tokens
        entity_alias_count[top_entity_id] = n_aliases + 1
        if n_aliases == 0:
            entities_emitted += 1

    block = ""
    if lines:
        block = "\n\n[MENTION_INDEX]\n" + "\n".join(lines) + "\n[/MENTION_INDEX]"

    manifest = {
        "page_ids": list(page_ids),
        "included": included,
        "skipped": skipped,
    }
    return block, manifest


def get_page_ids_for_chunks(conn, chunk_ids: List[int]) -> List[int]:
    """Get distinct page_ids for chunks via chunk_pages."""
    if not chunk_ids:
        return []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT page_id FROM chunk_pages
            WHERE chunk_id = ANY(%s)
            ORDER BY page_id
        """, (chunk_ids,))
        return [r[0] for r in cur.fetchall()]
