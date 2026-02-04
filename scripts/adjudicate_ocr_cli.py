#!/usr/bin/env python3
"""
Simple CLI for adjudicating OCR resolution results.

Actions:
  a / accept [N]  - Accept link to entity (default: top candidate, or specify N for Nth candidate)
  r / reject      - Reject as junk/wrong (adds to blocklist)
  s / skip        - Skip without deciding
  c / context     - Show more context around the span
  n / new         - Propose new entity for this surface
  q / quit        - Save progress and exit

For multi-word surface_norms (2+ tokens), accept/reject decisions are auto-applied
to ALL pending instances of that surface_norm.

Usage:
    python scripts/adjudicate_ocr_cli.py
    python scripts/adjudicate_ocr_cli.py --collection venona --limit 100
    python scripts/adjudicate_ocr_cli.py --status queue  # Only queued items
    python scripts/adjudicate_ocr_cli.py --status resolved --score-max 0.80  # Review borderline resolved
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from retrieval.normalization import normalize_for_fts as normalize_surface


# Common stopwords - surfaces containing these should NOT auto-apply globally
STOPWORDS = {
    'was', 'were', 'is', 'are', 'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at',
    'for', 'by', 'with', 'and', 'or', 'but', 'that', 'this', 'from', 'as',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
    'he', 'she', 'it', 'they', 'we', 'you', 'i', 'his', 'her', 'its', 'their',
    'our', 'your', 'my', 'who', 'whom', 'which', 'what', 'where', 'when', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
}


import re


def clean_ocr_text(text: str) -> str:
    """Clean up common OCR artifacts in text."""
    if not text:
        return ""
    
    # Fix common OCR character substitutions
    fixes = [
        (r'jB', 'S'),           # jB -> S (common OCR error)
        (r'(?<=[a-z])(?=[A-Z])', ' '),  # Split merged words (camelCase)
        (r'\s+', ' '),          # Normalize whitespace
    ]
    result = text
    for pattern, replacement in fixes:
        result = re.sub(pattern, replacement, result)
    
    return result.strip()


def extract_context_phrases(context: str, surface: str, min_words: int = 2, max_words: int = 7) -> List[str]:
    """
    Extract potential entity phrases from context that include or are near the surface.
    Returns normalized phrases for searching.
    """
    if not context:
        return []
    
    # Clean OCR artifacts
    clean = clean_ocr_text(context)
    
    # Remove non-alpha characters but keep spaces
    clean = re.sub(r'[^a-zA-Z\s]', ' ', clean)
    clean = ' '.join(clean.split())
    
    words = clean.split()
    if len(words) < min_words:
        return []
    
    # Find position of surface in context (approximate)
    surface_lower = surface.lower()
    surface_idx = -1
    for i, w in enumerate(words):
        if surface_lower in w.lower():
            surface_idx = i
            break
    
    phrases = set()
    
    # Generate n-grams around the surface position
    for n in range(min_words, min(max_words + 1, len(words) + 1)):
        for i in range(max(0, len(words) - n + 1)):
            # Prioritize phrases near or containing the surface
            if surface_idx >= 0:
                # Only include phrases within 3 words of surface
                if i > surface_idx + 3 or i + n < surface_idx - 3:
                    continue
            
            phrase = ' '.join(words[i:i+n])
            # Skip if all stopwords
            phrase_words = phrase.lower().split()
            if not all(w in STOPWORDS for w in phrase_words):
                phrases.add(phrase.lower())
    
    return list(phrases)


def search_context_entities(conn, context: str, surface: str, limit: int = 5) -> List[Dict]:
    """
    Search for entities by fuzzy matching phrases extracted from context.
    Returns deduplicated entity matches with their matched phrase.
    """
    phrases = extract_context_phrases(context, surface)
    if not phrases:
        return []
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Search for entities matching these phrases using trigram similarity
    all_matches = []
    
    for phrase in phrases[:20]:  # Limit to avoid too many queries
        # Use trigram similarity search
        cur.execute("""
            SELECT DISTINCT ON (e.id)
                e.id, 
                e.canonical_name, 
                e.entity_type,
                ea.alias_norm AS matched_alias,
                similarity(ea.alias_norm, %s) AS sim_score,
                %s AS matched_phrase
            FROM entities e
            JOIN entity_aliases ea ON ea.entity_id = e.id
            WHERE similarity(ea.alias_norm, %s) > 0.3
            ORDER BY e.id, similarity(ea.alias_norm, %s) DESC
            LIMIT 3
        """, (phrase, phrase, phrase, phrase))
        
        for row in cur.fetchall():
            all_matches.append(dict(row))
    
    # Deduplicate by entity_id, keeping highest similarity
    best_by_entity: Dict[int, Dict] = {}
    for match in all_matches:
        eid = match['id']
        if eid not in best_by_entity or match['sim_score'] > best_by_entity[eid]['sim_score']:
            best_by_entity[eid] = match
    
    # Sort by similarity descending
    results = sorted(best_by_entity.values(), key=lambda x: x['sim_score'], reverse=True)
    
    return results[:limit]


def search_entities_fuzzy(conn, query: str, limit: int = 5, min_similarity: float = 0.35) -> List[Dict]:
    """
    Search for entities using trigram fuzzy matching on aliases.
    Returns matches with similarity score.
    """
    if not query or len(query) < 3:
        return []
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    query_norm = query.lower().strip()
    
    cur.execute("""
        SELECT DISTINCT ON (e.id)
            e.id, 
            e.canonical_name, 
            e.entity_type,
            ea.alias_norm AS matched_alias,
            similarity(ea.alias_norm, %s) AS sim_score,
            %s AS matched_phrase
        FROM entities e
        JOIN entity_aliases ea ON ea.entity_id = e.id
        WHERE similarity(ea.alias_norm, %s) > %s
        ORDER BY e.id, similarity(ea.alias_norm, %s) DESC
    """, (query_norm, query_norm, query_norm, min_similarity, query_norm))
    
    results = [dict(row) for row in cur.fetchall()]
    
    # Sort by similarity descending and limit
    results.sort(key=lambda x: x['sim_score'], reverse=True)
    return results[:limit]


def should_apply_globally(surface_norm: str) -> bool:
    """
    Determine if a decision on this surface should apply globally (all documents).
    
    Returns True if:
    - Surface has 2+ words AND
    - None of the words are stopwords
    """
    words = surface_norm.lower().split()
    if len(words) < 2:
        return False
    # Only apply globally if NO word is a stopword
    return not any(w in STOPWORDS for w in words)


def get_conn():
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'localhost'),
        port=int(os.environ.get('POSTGRES_PORT', '5432')),
        dbname=os.environ.get('POSTGRES_DB', 'neh'),
        user=os.environ.get('POSTGRES_USER', 'neh'),
        password=os.environ.get('POSTGRES_PASSWORD', 'neh')
    )


@dataclass
class ReviewItem:
    candidate_id: int
    chunk_id: int
    document_id: int
    raw_span: str
    surface_norm: str
    char_start: int
    char_end: int
    resolution_status: str
    resolution_score: Optional[float]
    top_candidates: List[Dict]
    document_name: str
    collection_slug: str
    chunk_text: str


def get_review_items(
    conn,
    *,
    collection: Optional[str] = None,
    status: str = 'queue',
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    limit: int = 100,
) -> List[ReviewItem]:
    """Fetch items for review."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Increase statement timeout for this query
    cur.execute("SET statement_timeout = '120s'")
    
    conditions = ["mc.resolution_status = %s"]
    params: List = [status]
    
    if collection:
        conditions.append("col.slug = %s")
        params.append(collection)
    
    if score_min is not None:
        conditions.append("mc.resolution_score >= %s")
        params.append(score_min)
    
    if score_max is not None:
        conditions.append("mc.resolution_score <= %s")
        params.append(score_max)
    
    where_clause = " AND ".join(conditions)
    
    # Use a subquery to get candidate IDs first (faster with index on resolution_status)
    # then join to get full data - avoids sorting full rows
    cur.execute(f"""
        WITH candidate_ids AS (
            SELECT mc.id
            FROM mention_candidates mc
            JOIN documents d ON d.id = mc.document_id
            JOIN collections col ON col.id = d.collection_id
            WHERE {where_clause}
            ORDER BY mc.id
            LIMIT %s
        )
        SELECT
            mc.id AS candidate_id,
            mc.chunk_id,
            mc.document_id,
            mc.raw_span,
            mc.surface_norm,
            mc.char_start,
            mc.char_end,
            mc.resolution_status,
            mc.resolution_score,
            mc.top_candidates,
            d.source_name AS document_name,
            col.slug AS collection_slug,
            c.text AS chunk_text
        FROM mention_candidates mc
        JOIN candidate_ids ci ON ci.id = mc.id
        JOIN documents d ON d.id = mc.document_id
        JOIN collections col ON col.id = d.collection_id
        JOIN chunks c ON c.id = mc.chunk_id
        ORDER BY mc.id
    """, params + [limit])
    
    items = []
    for row in cur.fetchall():
        top_cands = row['top_candidates'] or []
        if isinstance(top_cands, str):
            try:
                top_cands = json.loads(top_cands)
            except Exception:
                top_cands = []
        
        items.append(ReviewItem(
            candidate_id=row['candidate_id'],
            chunk_id=row['chunk_id'],
            document_id=row['document_id'],
            raw_span=row['raw_span'],
            surface_norm=row['surface_norm'],
            char_start=row['char_start'],
            char_end=row['char_end'],
            resolution_status=row['resolution_status'],
            resolution_score=float(row['resolution_score']) if row['resolution_score'] else None,
            top_candidates=top_cands if isinstance(top_cands, list) else [],
            document_name=row['document_name'] or '',
            collection_slug=row['collection_slug'] or '',
            chunk_text=row['chunk_text'] or '',
        ))
    
    return items


def get_entity_name(conn, entity_id: int) -> str:
    """Get entity canonical name."""
    cur = conn.cursor()
    cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
    row = cur.fetchone()
    return row[0] if row else f"(entity {entity_id})"


def count_same_surface(conn, surface_norm: str, status: str = 'queue') -> int:
    """Count how many items have the same surface_norm."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM mention_candidates WHERE surface_norm = %s AND resolution_status = %s",
        (surface_norm, status)
    )
    return cur.fetchone()[0]


def count_same_surface_in_doc(conn, surface_norm: str, document_id: int, status: str = 'queue') -> int:
    """Count how many items have the same surface_norm within a specific document."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM mention_candidates WHERE surface_norm = %s AND document_id = %s AND resolution_status = %s",
        (surface_norm, document_id, status)
    )
    return cur.fetchone()[0]


def find_adjacent_surfaces(conn, item: 'ReviewItem', max_gap: int = 5) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find surfaces adjacent to the current item in the same chunk.
    
    Returns (before, after) where each is a dict with surface info or None.
    max_gap: maximum characters between surfaces to consider them adjacent.
    Only returns surfaces that look like names (capitalized, not common words).
    """
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Find surface immediately BEFORE this one
    cur.execute("""
        SELECT id, raw_span, surface_norm, char_start, char_end, resolution_status
        FROM mention_candidates
        WHERE chunk_id = %s
          AND char_end <= %s
          AND char_end >= %s - %s
        ORDER BY char_end DESC
        LIMIT 1
    """, (item.chunk_id, item.char_start, item.char_start, max_gap))
    before = cur.fetchone()
    
    # Filter out surfaces where ALL words are stopwords
    if before:
        words = before['surface_norm'].lower().split()
        if all(w in STOPWORDS for w in words):
            before = None
    
    # Find surface immediately AFTER this one
    cur.execute("""
        SELECT id, raw_span, surface_norm, char_start, char_end, resolution_status
        FROM mention_candidates
        WHERE chunk_id = %s
          AND char_start >= %s
          AND char_start <= %s + %s
        ORDER BY char_start ASC
        LIMIT 1
    """, (item.chunk_id, item.char_end, item.char_end, max_gap))
    after = cur.fetchone()
    
    # Filter out surfaces where ALL words are stopwords
    if after:
        words = after['surface_norm'].lower().split()
        if all(w in STOPWORDS for w in words):
            after = None
    
    return (dict(before) if before else None, dict(after) if after else None)


def apply_accept(
    conn,
    item: ReviewItem,
    entity_id: int,
    apply_all: bool = False,
) -> int:
    """
    Accept: link candidate(s) to entity.
    
    - Always applies to all instances within the same document.
    - If apply_all=True (multi-word surfaces), also applies across ALL documents.
    
    Returns count of items updated.
    """
    cur = conn.cursor()
    
    if apply_all:
        # Update all with same surface_norm across ALL documents
        cur.execute("""
            UPDATE mention_candidates
            SET resolution_status = 'resolved',
                resolved_entity_id = %s,
                resolution_method = 'manual_accept',
                resolved_at = NOW()
            WHERE surface_norm = %s AND resolution_status = 'queue'
            RETURNING id
        """, (entity_id, item.surface_norm))
        updated_ids = [r[0] for r in cur.fetchall()]
        
        # Also insert entity_mentions for all
        cur.execute("""
            INSERT INTO entity_mentions (entity_id, chunk_id, document_id, surface, surface_norm, start_char, end_char, confidence, method)
            SELECT %s, chunk_id, document_id, raw_span, surface_norm, char_start, char_end, 1.0, 'human'
            FROM mention_candidates
            WHERE id = ANY(%s)
            ON CONFLICT (chunk_id, entity_id, start_char, end_char) DO NOTHING
        """, (entity_id, updated_ids))
        
        # Add to allowlist
        cur.execute("""
            INSERT INTO ocr_variant_allowlist (variant_key, entity_id, source)
            VALUES (%s, %s, 'cli_adjudicate')
            ON CONFLICT (variant_key, entity_id) DO NOTHING
        """, (item.surface_norm, entity_id))
        
        conn.commit()
        return len(updated_ids)
    else:
        # Update all instances within the SAME DOCUMENT
        cur.execute("""
            UPDATE mention_candidates
            SET resolution_status = 'resolved',
                resolved_entity_id = %s,
                resolution_method = 'manual_accept',
                resolved_at = NOW()
            WHERE surface_norm = %s 
              AND document_id = %s
              AND resolution_status = 'queue'
            RETURNING id
        """, (entity_id, item.surface_norm, item.document_id))
        updated_ids = [r[0] for r in cur.fetchall()]
        
        # Insert entity_mentions for all updated
        if updated_ids:
            cur.execute("""
                INSERT INTO entity_mentions (entity_id, chunk_id, document_id, surface, surface_norm, start_char, end_char, confidence, method)
                SELECT %s, chunk_id, document_id, raw_span, surface_norm, char_start, char_end, 1.0, 'human'
                FROM mention_candidates
                WHERE id = ANY(%s)
                ON CONFLICT (chunk_id, entity_id, start_char, end_char) DO NOTHING
            """, (entity_id, updated_ids))
        
        conn.commit()
        return len(updated_ids) if updated_ids else 1


def apply_reject(
    conn,
    item: ReviewItem,
    apply_all: bool = False,
) -> int:
    """
    Reject: mark as junk and add to blocklist.
    
    - Always applies to all instances within the same document.
    - If apply_all=True (multi-word surfaces), also applies across ALL documents and adds to blocklist.
    
    Returns count of items updated.
    """
    cur = conn.cursor()
    
    if apply_all:
        # Reject across ALL documents
        cur.execute("""
            UPDATE mention_candidates
            SET resolution_status = 'ignore',
                resolution_method = 'manual_reject',
                resolved_at = NOW()
            WHERE surface_norm = %s AND resolution_status = 'queue'
            RETURNING id
        """, (item.surface_norm,))
        updated_ids = [r[0] for r in cur.fetchall()]
        
        # Add to blocklist (global)
        cur.execute("""
            INSERT INTO ocr_variant_blocklist (variant_key, block_type, reason, source)
            VALUES (%s, 'exact', 'manual_reject', 'cli_adjudicate')
            ON CONFLICT (variant_key, pattern_signature) DO NOTHING
        """, (item.surface_norm,))
        
        conn.commit()
        return len(updated_ids)
    else:
        # Reject all instances within the SAME DOCUMENT
        cur.execute("""
            UPDATE mention_candidates
            SET resolution_status = 'ignore',
                resolution_method = 'manual_reject',
                resolved_at = NOW()
            WHERE surface_norm = %s
              AND document_id = %s
              AND resolution_status = 'queue'
            RETURNING id
        """, (item.surface_norm, item.document_id))
        updated_ids = [r[0] for r in cur.fetchall()]
        
        conn.commit()
        return len(updated_ids) if updated_ids else 1


def dedupe_candidates_by_entity(candidates: List[Dict]) -> List[Dict]:
    """
    Deduplicate candidates by entity_id, keeping only the highest-scoring match per entity.
    """
    if not candidates:
        return []
    
    best_by_entity: Dict[int, Dict] = {}
    for cand in candidates:
        eid = cand.get('entity_id')
        if not eid:
            continue
        score = cand.get('score', cand.get('combined_score', 0)) or 0
        
        if eid not in best_by_entity:
            best_by_entity[eid] = cand
        else:
            existing_score = best_by_entity[eid].get('score', best_by_entity[eid].get('combined_score', 0)) or 0
            if score > existing_score:
                best_by_entity[eid] = cand
    
    # Sort by score descending
    deduped = sorted(best_by_entity.values(), key=lambda c: c.get('score', c.get('combined_score', 0)) or 0, reverse=True)
    return deduped


def get_context(item: ReviewItem, window: int = 300) -> str:
    """Get context around the span."""
    text = item.chunk_text
    if not text:
        return "(no text available)"
    
    start = max(0, item.char_start - window)
    end = min(len(text), item.char_end + window)
    
    before = text[start:item.char_start]
    span = text[item.char_start:item.char_end]
    after = text[item.char_end:end]
    
    # Clean up whitespace
    before = ' '.join(before.split())
    span = ' '.join(span.split())
    after = ' '.join(after.split())
    
    return f"...{before} >>>{span}<<< {after}..."


def display_item(item: ReviewItem, conn, idx: int, total: int) -> Tuple[Optional[Dict], Optional[Dict], List[Dict]]:
    """Display a review item. Returns (adjacent_before, adjacent_after, all_candidates)."""
    print("\n" + "=" * 70)
    print(f"[{idx + 1}/{total}]  Surface: {item.raw_span!r}")
    print(f"         Normalized: {item.surface_norm}")
    print(f"         Status: {item.resolution_status}  Score: {item.resolution_score or 'n/a'}")
    print(f"         Doc: {item.document_name} ({item.collection_slug})")
    print("-" * 70)
    
    # Show brief context
    brief_context = get_context(item, window=80)
    print(f"Context: {brief_context}")
    print("-" * 70)
    
    # Check for adjacent surfaces
    adj_before, adj_after = find_adjacent_surfaces(conn, item)
    combined_surface = None
    
    if adj_before or adj_after:
        parts = []
        if adj_before:
            parts.append(adj_before['raw_span'])
        parts.append(item.raw_span)
        if adj_after:
            parts.append(adj_after['raw_span'])
        combined_surface = ' '.join(parts)
        
        print(f"*** ADJACENT SURFACES DETECTED ***")
        if adj_before:
            status_mark = "✓" if adj_before['resolution_status'] != 'queue' else "○"
            print(f"  Before: '{adj_before['raw_span']}' ({adj_before['resolution_status']}) {status_mark}")
        print(f"  Current: '{item.raw_span}' ← YOU ARE HERE")
        if adj_after:
            status_mark = "✓" if adj_after['resolution_status'] != 'queue' else "○"
            print(f"  After:  '{adj_after['raw_span']}' ({adj_after['resolution_status']}) {status_mark}")
        print(f"  Combined: '{combined_surface}'")
        print("-" * 70)
    
    # Show candidates (deduplicated by entity)
    deduped_candidates = dedupe_candidates_by_entity(item.top_candidates)
    if deduped_candidates:
        print("Top candidates:")
        for i, cand in enumerate(deduped_candidates[:5], 1):
            eid = cand.get('entity_id')
            ename = cand.get('entity_name') or get_entity_name(conn, eid) if eid else '?'
            score = cand.get('score', cand.get('combined_score', '?'))
            alias = cand.get('alias_norm', '')
            tier = cand.get('proposal_tier', '')
            tier_str = f" [tier {tier}]" if tier else ""
            print(f"  {i}. {ename} (id={eid}, score={score}, alias={alias!r}){tier_str}")
    else:
        print("No candidates available.")
    
    # Context-based fuzzy matching is now opt-in via [f]ind command
    # (too slow to run automatically)
    
    # Show count of same surface_norm (in doc and globally)
    same_in_doc = count_same_surface_in_doc(conn, item.surface_norm, item.document_id, item.resolution_status)
    same_global = count_same_surface(conn, item.surface_norm, item.resolution_status)
    will_apply_globally = should_apply_globally(item.surface_norm)
    
    if same_in_doc > 1:
        print(f"\n*** {same_in_doc} instances in this document. Decision applies to all in doc. ***")
    if will_apply_globally and same_global > same_in_doc:
        print(f"*** {same_global} total across all docs. Multi-word (no stopwords): will apply globally. ***")
    elif len(item.surface_norm.split()) >= 2 and same_global > same_in_doc:
        print(f"*** {same_global} total across all docs. Contains stopword: applies to THIS DOC ONLY. ***")
    
    print("-" * 70)
    cmds = "[a]ccept [N], [r]eject, [s]kip, [c]ontext, [n]ew, [l]ink, [f]ind fuzzy"
    if adj_before or adj_after:
        cmds += ", [j]oin"
    cmds += ", [q]uit"
    print(f"Commands: {cmds}")
    
    return adj_before, adj_after, deduped_candidates or []


def search_entities(conn, query: str, limit: int = 10) -> List[Dict]:
    """Search for entities by name or alias."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Search by canonical name or alias
    # Use a subquery to handle DISTINCT properly with ORDER BY
    cur.execute("""
        SELECT id, canonical_name, entity_type, matched_alias
        FROM (
            SELECT DISTINCT ON (e.id) 
                   e.id, e.canonical_name, e.entity_type,
                   ea.alias_norm AS matched_alias,
                   CASE WHEN e.canonical_name ILIKE %s THEN 0 ELSE 1 END AS sort_priority
            FROM entities e
            LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
            WHERE e.canonical_name ILIKE %s
               OR ea.alias_norm ILIKE %s
            ORDER BY e.id, sort_priority
        ) sub
        ORDER BY sort_priority, canonical_name
        LIMIT %s
    """, (f'{query}%', f'%{query}%', f'%{query}%', limit))
    
    return [dict(row) for row in cur.fetchall()]


def get_entity_by_id(conn, entity_id: int) -> Optional[Dict]:
    """Get entity by ID."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, canonical_name, entity_type
        FROM entities
        WHERE id = %s
    """, (entity_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def prompt_new_entity(item: ReviewItem) -> Optional[Dict]:
    """Prompt user to define a new entity."""
    print("\n--- Create New Entity ---")
    print(f"Surface: {item.raw_span!r}")
    
    name = input("Entity name (blank to cancel): ").strip()
    if not name:
        return None
    
    etype = input("Entity type (person/org/place) [person]: ").strip().lower() or 'person'
    if etype not in ('person', 'org', 'place'):
        print(f"Invalid type '{etype}', using 'person'")
        etype = 'person'
    
    return {
        'name': name,
        'type': etype,
        'surface_norm': item.surface_norm,
    }


def get_or_create_cli_source(conn) -> int:
    """Get or create the concordance source for CLI-created entities."""
    cur = conn.cursor()
    
    # Try to get existing CLI source
    cur.execute("SELECT id FROM concordance_sources WHERE slug = 'cli_adjudicate'")
    row = cur.fetchone()
    if row:
        return row[0]
    
    # Create it
    cur.execute("""
        INSERT INTO concordance_sources (slug, title, notes)
        VALUES ('cli_adjudicate', 'CLI Adjudication', 'Entities created via adjudicate_ocr_cli.py')
        RETURNING id
    """)
    source_id = cur.fetchone()[0]
    conn.commit()
    return source_id


def create_entity_and_link(conn, item: ReviewItem, entity_info: Dict, apply_all: bool) -> Tuple[int, int]:
    """
    Create a new entity and link the surface to it.
    
    Returns (entity_id, count_linked).
    """
    cur = conn.cursor()
    
    # Get or create CLI source
    source_id = get_or_create_cli_source(conn)
    
    # Create entity
    cur.execute("""
        INSERT INTO entities (canonical_name, entity_type, source_id)
        VALUES (%s, %s, %s)
        RETURNING id
    """, (entity_info['name'], entity_info['type'], source_id))
    entity_id = cur.fetchone()[0]
    
    # Add alias (canonical name as primary alias)
    cur.execute("""
        INSERT INTO entity_aliases (entity_id, alias, alias_norm, kind, source_id)
        VALUES (%s, %s, %s, 'primary', %s)
        ON CONFLICT (entity_id, alias_norm) DO NOTHING
    """, (entity_id, entity_info['name'], normalize_surface(entity_info['name']), source_id))
    
    # Also add the surface as an alias if different
    if normalize_surface(entity_info['name']) != item.surface_norm:
        cur.execute("""
            INSERT INTO entity_aliases (entity_id, alias, alias_norm, kind, source_id)
            VALUES (%s, %s, %s, 'alt', %s)
            ON CONFLICT (entity_id, alias_norm) DO NOTHING
        """, (entity_id, item.raw_span, item.surface_norm, source_id))
    
    conn.commit()
    
    # Now link
    count = apply_accept(conn, item, entity_id, apply_all=apply_all)
    
    return entity_id, count


def main():
    parser = argparse.ArgumentParser(description='CLI for adjudicating OCR results')
    parser.add_argument('--collection', help='Filter by collection slug')
    parser.add_argument('--status', default='queue', help='Resolution status to review (default: queue)')
    parser.add_argument('--score-min', type=float, help='Minimum resolution score')
    parser.add_argument('--score-max', type=float, help='Maximum resolution score')
    parser.add_argument('--limit', type=int, default=100, help='Max items to load')
    args = parser.parse_args()
    
    conn = get_conn()
    
    if not args.collection:
        print("TIP: Use --collection <slug> to filter by collection (faster query)")
        print("     e.g., --collection silvermaster")
    
    print("Loading items for review...")
    items = get_review_items(
        conn,
        collection=args.collection,
        status=args.status,
        score_min=args.score_min,
        score_max=args.score_max,
        limit=args.limit,
    )
    
    if not items:
        print("No items to review.")
        return
    
    print(f"Loaded {len(items)} items for review.")
    
    # Stats
    stats = {
        'accepted': 0,
        'rejected': 0,
        'skipped': 0,
        'new_entities': 0,
    }
    
    # Track skipped surfaces to avoid showing them again this session
    skipped_surfaces: set = set()
    
    idx = 0
    while idx < len(items):
        item = items[idx]
        
        # Skip if this surface was already skipped this session
        if item.surface_norm in skipped_surfaces:
            idx += 1
            continue
        
        # Check if already processed (might have been batch-applied)
        cur = conn.cursor()
        cur.execute("SELECT resolution_status FROM mention_candidates WHERE id = %s", (item.candidate_id,))
        row = cur.fetchone()
        if row and row[0] != args.status:
            # Already processed, skip
            idx += 1
            continue
        
        adj_before, adj_after, all_candidates = display_item(item, conn, idx, len(items))
        
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            cmd = 'q'
        
        if not cmd or cmd == 's' or cmd == 'skip':
            skipped_surfaces.add(item.surface_norm)
            stats['skipped'] += 1
            idx += 1
            print(f"Skipped: '{item.surface_norm}' won't appear again this session.")
            continue
        
        if cmd == 'q' or cmd == 'quit':
            break
        
        if cmd == 'c' or cmd == 'context':
            # Show more context
            print("\n" + "-" * 70)
            print("Extended context:")
            print(get_context(item, window=500))
            # Don't advance; let user see context then decide
            continue
        
        if cmd == 'f' or cmd == 'find':
            # Run fuzzy context search
            print("\nSearching context with fuzzy matching...")
            
            context_for_search = get_context(item, window=150)
            context_matches = search_context_entities(conn, context_for_search, item.raw_span, limit=5)
            
            # Also search combined surface with adjacent text
            if adj_before or adj_after:
                combined_parts = []
                if adj_before:
                    combined_parts.append(adj_before['raw_span'])
                combined_parts.append(item.raw_span)
                if adj_after:
                    combined_parts.append(adj_after['raw_span'])
                
                combined_phrase = ' '.join(combined_parts)
                combined_clean = clean_ocr_text(combined_phrase)
                combined_clean = re.sub(r'[^a-zA-Z\s]', ' ', combined_clean)
                combined_clean = ' '.join(combined_clean.split())
                
                combined_results = search_entities_fuzzy(conn, combined_clean, limit=3, min_similarity=0.25)
                context_matches.extend(combined_results)
                
                if adj_after:
                    short_phrase = f"{item.raw_span} {adj_after['raw_span']}"
                    short_clean = re.sub(r'[^a-zA-Z\s]', ' ', short_phrase)
                    short_clean = ' '.join(short_clean.split())
                    short_results = search_entities_fuzzy(conn, short_clean, limit=3, min_similarity=0.25)
                    context_matches.extend(short_results)
                
                if adj_before and adj_before['raw_span'].lower() not in STOPWORDS:
                    reverse_phrase = f"{adj_before['raw_span']} {item.raw_span}"
                    reverse_clean = re.sub(r'[^a-zA-Z\s]', ' ', reverse_phrase)
                    reverse_clean = ' '.join(reverse_clean.split())
                    reverse_results = search_entities_fuzzy(conn, reverse_clean, limit=3, min_similarity=0.25)
                    context_matches.extend(reverse_results)
            
            # Filter and deduplicate
            existing_ids = {c.get('entity_id') for c in all_candidates}
            seen_ids = set()
            unique_matches = []
            for m in context_matches:
                if m['id'] not in existing_ids and m['id'] not in seen_ids:
                    seen_ids.add(m['id'])
                    unique_matches.append(m)
            
            if unique_matches:
                print("\nContext-based matches (fuzzy):")
                start_num = len(all_candidates) + 1
                for i, match in enumerate(unique_matches[:5], start_num):
                    sim_pct = int(match['sim_score'] * 100)
                    print(f"  {i}. {match['canonical_name']} (id={match['id']}, {sim_pct}% match)")
                    print(f"      matched: '{match['matched_phrase']}' ~ '{match['matched_alias']}'")
                
                # Add to candidates for selection
                for match in unique_matches[:5]:
                    all_candidates.append({
                        'entity_id': match['id'],
                        'entity_name': match['canonical_name'],
                        'score': match['sim_score'],
                        'alias_norm': match['matched_alias'],
                    })
                print(f"\nYou can now use 'a {start_num}' etc. to accept these matches.")
            else:
                print("No additional matches found in context.")
            
            continue
        
        if cmd.startswith('a') or cmd.startswith('accept'):
            # Accept
            parts = cmd.split()
            cand_idx = 0  # default to first candidate
            if len(parts) > 1:
                try:
                    cand_idx = int(parts[1]) - 1
                except ValueError:
                    print("Invalid candidate number.")
                    continue
            
            # Use combined candidates (includes context matches)
            if not all_candidates:
                print("No candidates available. Use 'n' to create new entity.")
                continue
            
            if cand_idx < 0 or cand_idx >= len(all_candidates):
                print(f"Invalid candidate number. Choose 1-{len(all_candidates)}.")
                continue
            
            selected = all_candidates[cand_idx]
            entity_id = selected.get('entity_id')
            if not entity_id:
                print("Candidate has no entity_id.")
                continue
            
            entity_name = selected.get('entity_name') or get_entity_name(conn, entity_id)
            
            # Check if should apply globally (multi-word without stopwords)
            apply_all = should_apply_globally(item.surface_norm)
            
            count = apply_accept(conn, item, entity_id, apply_all=apply_all)
            stats['accepted'] += count
            
            if count > 1:
                scope = "globally" if apply_all else "in this document"
                print(f"Accepted: linked {count} instances {scope} -> {entity_name}")
            else:
                print(f"Accepted: linked -> {entity_name}")
            
            idx += 1
            continue
        
        if cmd.startswith('r') or cmd.startswith('reject'):
            # Reject
            apply_all = should_apply_globally(item.surface_norm)
            
            count = apply_reject(conn, item, apply_all=apply_all)
            stats['rejected'] += count
            
            if count > 1:
                scope = "globally" if apply_all else "in this document"
                print(f"Rejected: blocked {count} instances {scope}")
            else:
                print("Rejected: marked as junk")
            
            idx += 1
            continue
        
        if cmd == 'n' or cmd == 'new':
            # New entity
            entity_info = prompt_new_entity(item)
            if not entity_info:
                print("Cancelled.")
                continue
            
            apply_all = should_apply_globally(item.surface_norm)
            
            entity_id, count = create_entity_and_link(conn, item, entity_info, apply_all=apply_all)
            stats['new_entities'] += 1
            stats['accepted'] += count
            
            print(f"Created entity '{entity_info['name']}' (id={entity_id}) and linked {count} instance(s).")
            idx += 1
            continue
        
        if cmd.startswith('l') or cmd.startswith('link'):
            # Link to existing entity by ID or search
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: l <entity_id> or l <search term>")
                print("Example: l 69380  or  l maurice halperin")
                continue
            
            arg = parts[1].strip()
            entity = None
            
            # Try as entity ID first
            try:
                eid = int(arg)
                entity = get_entity_by_id(conn, eid)
                if not entity:
                    print(f"No entity found with ID {eid}")
                    continue
            except ValueError:
                # Search by name
                results = search_entities(conn, arg)
                if not results:
                    print(f"No entities found matching '{arg}'")
                    continue
                
                print(f"\nSearch results for '{arg}':")
                for i, e in enumerate(results[:10], 1):
                    alias_info = f" (matched: {e['matched_alias']})" if e.get('matched_alias') else ""
                    print(f"  {i}. {e['canonical_name']} (id={e['id']}, type={e['entity_type']}){alias_info}")
                
                try:
                    choice = input("\nSelect number (or blank to cancel): ").strip()
                    if not choice:
                        print("Cancelled.")
                        continue
                    choice_idx = int(choice) - 1
                    if choice_idx < 0 or choice_idx >= len(results):
                        print("Invalid selection.")
                        continue
                    entity = results[choice_idx]
                except (ValueError, EOFError, KeyboardInterrupt):
                    print("Cancelled.")
                    continue
            
            entity_id = entity['id']
            entity_name = entity['canonical_name']
            
            apply_all = should_apply_globally(item.surface_norm)

            count = apply_accept(conn, item, entity_id, apply_all=apply_all)
            stats['accepted'] += count
            
            if count > 1:
                scope = "globally" if apply_all else "in this document"
                print(f"Linked: {count} instances {scope} -> {entity_name} (id={entity_id})")
            else:
                print(f"Linked: -> {entity_name} (id={entity_id})")
            
            idx += 1
            continue
        
        if cmd == 'j' or cmd == 'join':
            # Join adjacent surfaces and search for combined entity
            if not adj_before and not adj_after:
                print("No adjacent surfaces detected.")
                continue
            
            # Build combined surface
            parts = []
            if adj_before:
                parts.append(adj_before['raw_span'])
            parts.append(item.raw_span)
            if adj_after:
                parts.append(adj_after['raw_span'])
            combined_raw = ' '.join(parts)
            combined_norm = normalize_surface(combined_raw)
            
            print(f"\nSearching for: '{combined_raw}' (normalized: '{combined_norm}')")
            
            # Search for the combined surface (exact first, then fuzzy)
            results = search_entities(conn, combined_norm)
            if not results:
                results = search_entities(conn, combined_raw)
            
            # Try fuzzy matching if exact search fails
            fuzzy_results = []
            if not results or len(results) < 3:
                print("Trying fuzzy matching...")
                # Clean up OCR artifacts
                combined_clean = clean_ocr_text(combined_raw)
                combined_clean = re.sub(r'[^a-zA-Z\s]', ' ', combined_clean)
                combined_clean = ' '.join(combined_clean.split())
                
                fuzzy_results = search_entities_fuzzy(conn, combined_clean, limit=5, min_similarity=0.25)
                
                # Also try without stopwords
                words = combined_clean.split()
                content_words = [w for w in words if w.lower() not in STOPWORDS]
                if len(content_words) >= 2:
                    content_phrase = ' '.join(content_words)
                    fuzzy_results.extend(search_entities_fuzzy(conn, content_phrase, limit=5, min_similarity=0.25))
                
                # Deduplicate fuzzy results
                seen = {r['id'] for r in results} if results else set()
                unique_fuzzy = []
                for r in fuzzy_results:
                    if r['id'] not in seen:
                        seen.add(r['id'])
                        unique_fuzzy.append(r)
                fuzzy_results = unique_fuzzy
            
            if not results and not fuzzy_results:
                print(f"No entities found matching '{combined_raw}'")
                print("Use 'l <name>' to search manually, or 'n' to create new entity.")
                continue
            
            # Combine results for display
            all_join_results = []
            
            if results:
                print(f"\nExact matches:")
                for i, e in enumerate(results[:5], 1):
                    alias_info = f" (matched: {e['matched_alias']})" if e.get('matched_alias') else ""
                    print(f"  {i}. {e['canonical_name']} (id={e['id']}){alias_info}")
                    all_join_results.append(e)
            
            if fuzzy_results:
                print(f"\nFuzzy matches:")
                start_num = len(all_join_results) + 1
                for i, e in enumerate(fuzzy_results[:5], start_num):
                    sim_pct = int(e.get('sim_score', 0) * 100)
                    print(f"  {i}. {e['canonical_name']} (id={e['id']}, {sim_pct}% match)")
                    print(f"      matched: '{e.get('matched_phrase', '')}' ~ '{e.get('matched_alias', '')}'")
                    all_join_results.append(e)
            
            try:
                choice = input("\nSelect number (or blank to cancel): ").strip()
                if not choice:
                    print("Cancelled.")
                    continue
                choice_idx = int(choice) - 1
                if choice_idx < 0 or choice_idx >= len(all_join_results):
                    print(f"Invalid selection. Choose 1-{len(all_join_results)}.")
                    continue
                entity = all_join_results[choice_idx]
            except (ValueError, EOFError, KeyboardInterrupt):
                print("Cancelled.")
                continue
            
            entity_id = entity['id']
            entity_name = entity['canonical_name']
            
            # Mark all adjacent items as resolved (ignore them, they're part of this combined mention)
            cur = conn.cursor()
            resolved_count = 0
            
            # Resolve the current item
            count = apply_accept(conn, item, entity_id, apply_all=False)
            resolved_count += count
            
            # Mark adjacent items as 'ignore' since they're part of the combined surface
            if adj_before and adj_before['resolution_status'] == 'queue':
                cur.execute("""
                    UPDATE mention_candidates
                    SET resolution_status = 'ignore',
                        resolution_method = 'joined_to_adjacent',
                        resolved_at = NOW()
                    WHERE id = %s
                """, (adj_before['id'],))
                resolved_count += 1
            
            if adj_after and adj_after['resolution_status'] == 'queue':
                cur.execute("""
                    UPDATE mention_candidates
                    SET resolution_status = 'ignore',
                        resolution_method = 'joined_to_adjacent',
                        resolved_at = NOW()
                    WHERE id = %s
                """, (adj_after['id'],))
                resolved_count += 1
            
            conn.commit()
            stats['accepted'] += resolved_count
            
            print(f"Joined: '{combined_raw}' -> {entity_name} (id={entity_id})")
            print(f"  Resolved {resolved_count} surface(s)")
            
            idx += 1
            continue
        
        print(f"Unknown command: {cmd}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Accepted:     {stats['accepted']} mentions")
    print(f"Rejected:     {stats['rejected']} mentions")
    print(f"Skipped:      {len(skipped_surfaces)} unique surfaces")
    print(f"New entities: {stats['new_entities']}")
    print("=" * 70)
    
    conn.close()


if __name__ == '__main__':
    main()
