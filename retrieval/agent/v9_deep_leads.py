"""
V9 Think Deeper — Lead extraction from admitted chunks.

Deterministic extraction of leads (entity, org, doc, date) for evidence-led
exploration. Entity leads come from entity_mentions (no regex codenames).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from retrieval.agent.v9_deep_types import (
    CandidateChunk,
    Lead,
    LeadPool,
    LEAD_TYPE_DATE,
    LEAD_TYPE_DOC,
    LEAD_TYPE_ENTITY,
    LEAD_TYPE_ORG,
    _lead_id_hash,
)

logger = logging.getLogger(__name__)

# Org regex fallback: Capitalized phrases (1–4 words)
_ORG_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b")

# Year range regex
_YEAR_RE = re.compile(r"\b(19[0-4][0-9]|19[5-9][0-9]|20[0-2][0-9])\b")

# Entity mention threshold for +2 score
_ENTITY_MENTION_MIN = 2

# Max leads per pool
_MAX_LEADS = 20


def _extract_entity_leads(
    conn,
    chunk_ids: List[int],
    baseline_entity_ids: Set[int],
    current_step: int,
    baseline_doc_ids: Set[int],
    baseline_collection_slugs: Set[str],
) -> List[Tuple[Lead, int]]:
    """Extract entity leads from entity_mentions. Returns (Lead, score)."""
    if not chunk_ids:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT em.entity_id, e.canonical_name, e.entity_type,
                       COUNT(DISTINCT em.chunk_id) AS chunk_count,
                       COUNT(*) AS mention_count
                FROM entity_mentions em
                JOIN entities e ON e.id = em.entity_id
                WHERE em.chunk_id = ANY(%s)
                GROUP BY em.entity_id, e.canonical_name, e.entity_type
            """, (list(chunk_ids),))
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("entity_mentions query failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return []

    # Get chunk->doc and chunk->collection for "new doc" scoring
    chunk_to_doc: Dict[int, int] = {}
    chunk_to_coll: Dict[int, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, document_id, collection_slug
                FROM chunk_metadata WHERE chunk_id = ANY(%s)
            """, (list(chunk_ids),))
            for row in cur.fetchall():
                chunk_to_doc[row[0]] = row[1] or 0
                chunk_to_coll[row[0]] = row[2] or ""
    except Exception as e:
        logger.warning("chunk_metadata query failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    result: List[Tuple[Lead, int]] = []
    for entity_id, canonical_name, entity_type, chunk_count, mention_count in rows:
        if entity_id in baseline_entity_ids:
            continue
        if not canonical_name or not str(canonical_name).strip():
            continue

        support_cids = []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id FROM entity_mentions WHERE entity_id = %s AND chunk_id = ANY(%s)",
                    (entity_id, list(chunk_ids)),
                )
                support_cids = [r[0] for r in cur.fetchall()]
        except Exception as e:
            logger.warning("entity_mentions support_chunk_ids query failed: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

        score = 0
        # +3 if from new docs/collections
        for cid in support_cids:
            doc_id = chunk_to_doc.get(cid)
            coll = chunk_to_coll.get(cid, "")
            if doc_id and doc_id not in baseline_doc_ids:
                score += 3
                break
            if coll and coll.lower() not in {c.lower() for c in baseline_collection_slugs}:
                score += 3
                break
        # +2 if entity with mentions > X
        if mention_count >= _ENTITY_MENTION_MIN:
            score += 2
        # +2 if appears in multiple chunks
        if chunk_count >= 2:
            score += 2

        lead_type = LEAD_TYPE_ORG if (entity_type and str(entity_type).lower() in ("org", "unit")) else LEAD_TYPE_ENTITY
        lead_id = _lead_id_hash(lead_type, canonical_name, entity_id, None)
        lead = Lead(
            lead_id=lead_id,
            type=lead_type,
            value=canonical_name,
            entity_id=entity_id,
            support_chunk_ids=support_cids[:20],
            first_seen_step=current_step,
            last_seen_step=current_step,
        )
        result.append((lead, score))
    return result


def _extract_org_leads_fallback(
    chunks: List[CandidateChunk],
    current_step: int,
) -> List[Tuple[Lead, int]]:
    """Regex fallback for org leads when entity types unavailable."""
    phrase_to_chunks: Dict[str, List[int]] = {}
    for c in chunks:
        if not c.text:
            continue
        for m in _ORG_PHRASE_RE.finditer(c.text):
            phrase = m.group(1)
            if len(phrase) < 6:
                continue
            if phrase not in phrase_to_chunks:
                phrase_to_chunks[phrase] = []
            if c.chunk_id not in phrase_to_chunks[phrase]:
                phrase_to_chunks[phrase].append(c.chunk_id)

    result: List[Tuple[Lead, int]] = []
    for phrase, cids in phrase_to_chunks.items():
        score = 2 if len(cids) >= 2 else 0
        lead_id = _lead_id_hash(LEAD_TYPE_ORG, phrase, None, None)
        lead = Lead(
            lead_id=lead_id,
            type=LEAD_TYPE_ORG,
            value=phrase,
            support_chunk_ids=cids[:20],
            first_seen_step=current_step,
            last_seen_step=current_step,
        )
        result.append((lead, score))
    return result


def _extract_doc_leads(
    conn,
    chunks: List[CandidateChunk],
    baseline_doc_ids: Set[int],
    current_step: int,
    baseline_collection_slugs: Set[str],
) -> List[Tuple[Lead, int]]:
    """Extract doc leads from chunks not in baseline."""
    doc_to_chunks: Dict[int, List[int]] = {}
    doc_to_coll: Dict[int, str] = {}
    for c in chunks:
        if c.doc_id and c.doc_id not in baseline_doc_ids:
            if c.doc_id not in doc_to_chunks:
                doc_to_chunks[c.doc_id] = []
                doc_to_coll[c.doc_id] = c.collection_slug or ""
            doc_to_chunks[c.doc_id].append(c.chunk_id)

    if not doc_to_chunks:
        return []

    doc_labels: Dict[int, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.id, d.title, c.slug
                FROM documents d
                LEFT JOIN collections c ON c.id = d.collection_id
                WHERE d.id = ANY(%s)
            """, (list(doc_to_chunks.keys()),))
            for row in cur.fetchall():
                title = (row[1] or "").strip() or "Untitled"
                slug = row[2] or ""
                doc_labels[row[0]] = f"{slug} {title}"[:60] if slug else title[:60]
    except Exception:
        for did in doc_to_chunks:
            doc_labels[did] = f"Doc {did}"

    result: List[Tuple[Lead, int]] = []
    baseline_coll_lower = {c.lower() for c in baseline_collection_slugs}
    for doc_id, cids in doc_to_chunks.items():
        score = 3  # +3 new doc
        coll = doc_to_coll.get(doc_id, "")
        if coll and coll.lower() not in baseline_coll_lower:
            score += 2
        if len(cids) >= 2:
            score += 2

        label = doc_labels.get(doc_id, f"Doc {doc_id}")
        lead_id = _lead_id_hash(LEAD_TYPE_DOC, label, None, doc_id)
        lead = Lead(
            lead_id=lead_id,
            type=LEAD_TYPE_DOC,
            value=label,
            doc_id=doc_id,
            support_chunk_ids=cids[:20],
            first_seen_step=current_step,
            last_seen_step=current_step,
        )
        result.append((lead, score))
    return result


def _extract_date_leads(chunks: List[CandidateChunk], current_step: int) -> List[Tuple[Lead, int]]:
    """Extract date leads from date_spans or year regex."""
    seen: Dict[str, List[int]] = {}
    for c in chunks:
        for span in (c.date_spans or []):
            if span and len(span) >= 4:
                if span not in seen:
                    seen[span] = []
                seen[span].append(c.chunk_id)
        if not c.date_spans and c.text:
            for m in _YEAR_RE.finditer(c.text):
                yr = m.group(1)
                if yr not in seen:
                    seen[yr] = []
                if c.chunk_id not in seen[yr]:
                    seen[yr].append(c.chunk_id)

    result: List[Tuple[Lead, int]] = []
    for val, cids in seen.items():
        score = 2 if len(cids) >= 2 else 0
        lead_id = _lead_id_hash(LEAD_TYPE_DATE, val, None, None)
        lead = Lead(
            lead_id=lead_id,
            type=LEAD_TYPE_DATE,
            value=val,
            support_chunk_ids=cids[:20],
            first_seen_step=current_step,
            last_seen_step=current_step,
        )
        result.append((lead, score))
    return result


def extract_leads(
    conn,
    newly_admitted_chunks: List[CandidateChunk],
    baseline_entity_ids: Set[int],
    baseline_doc_ids: Set[int],
    current_step: int,
    prev_lead_pool: Optional[LeadPool] = None,
    baseline_collection_slugs: Optional[Set[str]] = None,
) -> LeadPool:
    """
    Extract and rank leads from newly admitted chunks.
    Merges with prev_lead_pool by lead_id; updates last_seen_step and support_chunk_ids.
    """
    if not newly_admitted_chunks:
        return prev_lead_pool or LeadPool()

    chunk_ids = [c.chunk_id for c in newly_admitted_chunks]
    baseline_collection_slugs = baseline_collection_slugs or set()
    # Wider baseline from prev pool if we had chunks before
    if prev_lead_pool:
        for lead in prev_lead_pool.leads:
            if lead.doc_id:
                baseline_doc_ids = baseline_doc_ids | {lead.doc_id}

    scored: List[Tuple[Lead, int]] = []

    # Entity leads (prefer entity types; org/unit go to org)
    scored.extend(_extract_entity_leads(
        conn, chunk_ids, baseline_entity_ids, current_step,
        baseline_doc_ids, baseline_collection_slugs,
    ))

    # Org leads: we get org from entity_leads when entity_type in (org, unit).
    # Regex fallback only if no org entities found
    org_from_entities = any(l.type == LEAD_TYPE_ORG for l, _ in scored)
    if not org_from_entities:
        scored.extend(_extract_org_leads_fallback(newly_admitted_chunks, current_step))

    # No regex codename extraction: rely on entity leads (entity_mentions) instead.
    # Codenames that are properly linked appear as entity leads with entity_id.

    # Doc leads
    scored.extend(_extract_doc_leads(
        conn, newly_admitted_chunks, baseline_doc_ids,
        current_step, baseline_collection_slugs,
    ))

    # Date leads
    scored.extend(_extract_date_leads(newly_admitted_chunks, current_step))

    # Build merged pool by lead_id
    by_id: Dict[str, Tuple[Lead, int]] = {}
    for lead, score in scored:
        if lead.lead_id in by_id:
            existing_lead, existing_score = by_id[lead.lead_id]
            merged_cids = list(dict.fromkeys(existing_lead.support_chunk_ids + lead.support_chunk_ids))[:20]
            merged_lead = Lead(
                lead_id=lead.lead_id,
                type=lead.type,
                value=lead.value,
                entity_id=lead.entity_id or existing_lead.entity_id,
                doc_id=lead.doc_id or existing_lead.doc_id,
                support_chunk_ids=merged_cids,
                first_seen_step=min(existing_lead.first_seen_step, lead.first_seen_step),
                last_seen_step=current_step,
            )
            by_id[lead.lead_id] = (merged_lead, max(existing_score, score) + 2)  # +2 multi-chunk
        else:
            by_id[lead.lead_id] = (lead, score)

    # Merge with prev pool
    if prev_lead_pool:
        for lead in prev_lead_pool.leads:
            if lead.lead_id not in by_id:
                updated = Lead(
                    lead_id=lead.lead_id,
                    type=lead.type,
                    value=lead.value,
                    entity_id=lead.entity_id,
                    doc_id=lead.doc_id,
                    support_chunk_ids=lead.support_chunk_ids,
                    first_seen_step=lead.first_seen_step,
                    last_seen_step=current_step,
                )
                by_id[lead.lead_id] = (updated, 0)

    # Sort by score desc, cap
    sorted_leads = sorted(by_id.values(), key=lambda x: -x[1])
    final = [lead for lead, _ in sorted_leads[:_MAX_LEADS]]

    return LeadPool(leads=final)
