"""
V10 Chunk Mention Extraction — deterministic + LLM-guided extraction.

Two extraction paths:
1. Deterministic (regex + lexicon lookup): extract_chunk_mentions_v10_deterministic()
2. LLM-guided (gpt-4.1-mini-2025-04-14): llm_extract_surfaces_v10() + map_surfaces_v10()

Both produce ChunkMentionsV10. The dispatcher extract_mentions_dispatched()
routes between them with caching.

Dependency direction:
    extract -> resolve (for alias mentions)
    lexicon.update_from_mentions <- extract output (called separately)
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from retrieval.agent.v10_types import (
    ALIAS_SCOPED_COLLECTIONS,
    CODENAME_ALIAS_KINDS,
    AliasContext,
    ChunkMention,
    ChunkMentionsV10,
    ChunkSignal,
    ExtractedSignal,
    ExtractedSurface,
    ExtractionContext,
    LexiconV10,
    ResolvedAlias,
    SpanCandidate,
)
from retrieval.agent.v10_resolve import resolve_alias_candidates

logger = logging.getLogger(__name__)

# =============================================================================
# Signal detection patterns (from codename_resolution.py)
# =============================================================================

# Reuse the same patterns but output structured ChunkSignal objects
SIGNAL_PATTERNS = [
    # "TWAIN" was [identified as] Name
    (
        r'(?P<code>[A-Z]{3,})["\']?\s+(?:was\s+)?(?:identified\s+as|is)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        "identified_as",
        0.95,
    ),
    # Name (TWAIN)
    (
        r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((?P<code>[A-Z]{3,})\)',
        "parenthetical",
        0.85,
    ),
    # Name, codenamed TWAIN
    (
        r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:code)?named\s+(?P<code>[A-Z]{3,})',
        "cryptonym_marker",
        0.90,
    ),
    # TWAIN = Name
    (
        r'(?P<code>[A-Z]{3,})\s*[=:]\s*(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        "alias_equation",
        0.80,
    ),
    # "TWAIN" (Name)
    (
        r'["\']?(?P<code>[A-Z]{3,})["\']?\s*\((?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\)',
        "parenthetical",
        0.85,
    ),
    # cover name "TWAIN" was Name
    (
        r'cover\s+name\s+["\']?(?P<code>[A-Z]{3,})["\']?\s+(?:was|is)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        "identified_as",
        0.95,
    ),
    # Name's cover name was "TWAIN"
    (
        r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[\'\s]+s?\s*cover\s+name\s+(?:was|is)\s+["\']?(?P<code>[A-Z]{3,})',
        "identified_as",
        0.90,
    ),
    # also known as / aka
    (
        r'(?P<code>[A-Z]{3,})\s+(?:also\s+known\s+as|a\.?k\.?a\.?)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        "aka",
        0.85,
    ),
    # Name, also known as CODENAME
    (
        r'(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,?\s+(?:also\s+known\s+as|a\.?k\.?a\.?)\s+(?P<code>[A-Z]{3,})',
        "aka",
        0.85,
    ),
]


# =============================================================================
# Entity surface detection
# =============================================================================

def _find_entity_surfaces(
    text: str,
    lexicon: LexiconV10,
) -> List[ChunkMention]:
    """Find entity canonical name surfaces in text.

    Matches against all entities currently in the lexicon.
    Always active regardless of collection.
    """
    mentions: List[ChunkMention] = []
    text_lower = text.lower()

    for entity_id, info in lexicon.entities_in_play.items():
        canonical = info.get("canonical_name", "")
        if not canonical:
            continue

        # Search for canonical name (case-insensitive)
        canon_lower = canonical.lower()
        start = 0
        while True:
            idx = text_lower.find(canon_lower, start)
            if idx == -1:
                break
            end = idx + len(canon_lower)

            # Check word boundaries (avoid matching substrings)
            if _is_word_boundary(text, idx, end):
                mentions.append(ChunkMention(
                    surface=text[idx:end],
                    start=idx,
                    end=end,
                    kind="entity_surface",
                    candidates=[SpanCandidate(
                        entity_id=entity_id,
                        canonical_name=canonical,
                        match_type="canonical",
                        valid_collections=["*"],
                        source="lexicon",
                    )],
                ))

            start = end

        # Also search for global variants
        for variant in info.get("global_variants", []):
            if not variant:
                continue
            var_lower = variant.lower()
            start = 0
            while True:
                idx = text_lower.find(var_lower, start)
                if idx == -1:
                    break
                end = idx + len(var_lower)
                if _is_word_boundary(text, idx, end):
                    mentions.append(ChunkMention(
                        surface=text[idx:end],
                        start=idx,
                        end=end,
                        kind="entity_surface",
                        candidates=[SpanCandidate(
                            entity_id=entity_id,
                            canonical_name=canonical,
                            match_type="alias",
                            valid_collections=["*"],
                            source="lexicon",
                        )],
                    ))
                start = end

    return mentions


# =============================================================================
# Alias surface detection
# =============================================================================

def _find_alias_surfaces(
    text: str,
    collection_slug: str,
    lexicon: LexiconV10,
) -> List[ChunkMention]:
    """Find alias surfaces in text.

    ONLY active when collection_slug is in ALIAS_SCOPED_COLLECTIONS.
    Invariant I1 enforcement.
    """
    if collection_slug not in ALIAS_SCOPED_COLLECTIONS:
        return []

    mentions: List[ChunkMention] = []
    text_lower = text.lower()

    # Get aliases for this collection from the lexicon
    alias_map = lexicon.entities_by_alias_scoped.get(collection_slug, {})

    for alias_text, entity_ids in alias_map.items():
        if not alias_text:
            continue

        alias_lower = alias_text.lower()
        start = 0
        while True:
            idx = text_lower.find(alias_lower, start)
            if idx == -1:
                break
            end = idx + len(alias_lower)

            if _is_word_boundary(text, idx, end):
                candidates = []
                for eid in entity_ids:
                    einfo = lexicon.entities_in_play.get(eid, {})
                    candidates.append(SpanCandidate(
                        entity_id=eid,
                        canonical_name=einfo.get("canonical_name", ""),
                        match_type="codename",
                        alias_type="code_name",
                        collision=_collision_level(len(entity_ids)),
                        valid_collections=[collection_slug],
                        source="lexicon",
                    ))

                mentions.append(ChunkMention(
                    surface=text[idx:end],
                    start=idx,
                    end=end,
                    kind="alias_surface",
                    candidates=candidates,
                ))

            start = end

    # Also scan for all-caps words that might be codenames not yet in lexicon
    # (minimum 3 chars, all uppercase)
    for m in re.finditer(r'\b([A-Z]{3,})\b', text):
        word = m.group(1)
        word_lower = word.lower()

        # Skip if already found via lexicon
        already_found = any(
            men.surface.lower() == word_lower and men.start == m.start()
            for men in mentions
        )
        if already_found:
            continue

        # Check if this looks like a codename (not a common abbreviation)
        if _is_likely_codename(word):
            mentions.append(ChunkMention(
                surface=word,
                start=m.start(),
                end=m.end(),
                kind="alias_surface",
                candidates=[],  # no candidates yet — will be resolved
            ))

    return mentions


def _collect_alias_surfaces_from_text(
    text: str,
    collection_slug: str,
    lexicon: LexiconV10,
) -> Set[str]:
    """Collect alias surface strings that would need DB resolution.

    Used to pre-scan chunk texts for batch alias lookup. Returns unique
    surface.lower() values that extract_chunk_mentions_v10_deterministic
    would pass to resolve_alias_candidates.
    """
    surfaces: Set[str] = set()
    if not text or collection_slug not in ALIAS_SCOPED_COLLECTIONS:
        return surfaces
    text_lower = text.lower()
    alias_map = lexicon.entities_by_alias_scoped.get(collection_slug, {})
    for alias_text, entity_ids in alias_map.items():
        if alias_text and alias_text.lower() in text_lower:
            surfaces.add(alias_text.lower().strip())
    for m in re.finditer(r"\b([A-Z]{3,})\b", text):
        word = m.group(1)
        if _is_likely_codename(word):
            surfaces.add(word.lower())
    return surfaces


def _is_likely_codename(word: str) -> bool:
    """Heuristic: is this all-caps word likely a codename?"""
    # Common non-codename abbreviations to exclude
    common_abbrevs = {
        "FBI", "CIA", "NSA", "KGB", "GRU", "NKVD", "MGB", "SVR",
        "USA", "USSR", "THE", "AND", "FOR", "NOT", "BUT", "ARE",
        "WAS", "HAS", "HAD", "HIS", "HER", "HIM", "SHE", "HIS",
        "TOP", "SECRET", "CLASSIFIED", "DECLASSIFIED",
        "FROM", "SUBJECT", "DATE", "MEMO", "NOTE",
        "VOL", "REF", "SEE", "ALSO", "PAGE", "COPY",
    }
    return word not in common_abbrevs and len(word) >= 3


# =============================================================================
# Signal detection
# =============================================================================

def _detect_signals(text: str, collection_slug: str) -> List[ChunkSignal]:
    """Detect high-signal patterns in chunk text.

    Only produces alias-related signals for alias-scoped collections.
    """
    signals: List[ChunkSignal] = []

    if collection_slug not in ALIAS_SCOPED_COLLECTIONS:
        return signals

    for pattern, signal_type, confidence in SIGNAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                code = match.group("code")
                name = match.group("name")
            except (IndexError, AttributeError):
                continue

            if code and name:
                signals.append(ChunkSignal(
                    signal_type=signal_type,
                    text=match.group(0)[:200],
                    confidence=confidence,
                    entity_a=code.upper(),
                    entity_b=name,
                ))

    # Co-mention detection: alias near canonical name (weak signal)
    # Look for codenames within 100 chars of known entity names
    if collection_slug in ALIAS_SCOPED_COLLECTIONS:
        codename_pattern = re.compile(r'\b[A-Z]{3,}\b')
        name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')

        codenames = [(m.group(), m.start(), m.end()) for m in codename_pattern.finditer(text)]
        names = [(m.group(), m.start(), m.end()) for m in name_pattern.finditer(text)]

        for code, cs, ce in codenames:
            if not _is_likely_codename(code):
                continue
            for name, ns, ne in names:
                distance = min(abs(cs - ne), abs(ns - ce))
                if distance < 100:
                    # Check it's not already captured by a stronger signal
                    already_captured = any(
                        s.entity_a and s.entity_a.upper() == code.upper() and
                        s.entity_b and s.entity_b.lower() == name.lower()
                        for s in signals
                    )
                    if not already_captured:
                        signals.append(ChunkSignal(
                            signal_type="co_mention",
                            text=f"{code} near {name}",
                            confidence=0.3,
                            entity_a=code.upper(),
                            entity_b=name,
                        ))

    return signals


# =============================================================================
# Helpers
# =============================================================================

def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """Check that the match is at word boundaries."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def _collision_level(count: int) -> str:
    if count <= 1:
        return "low"
    elif count <= 3:
        return "med"
    return "high"


# =============================================================================
# Main entry point
# =============================================================================

def extract_chunk_mentions_v10_deterministic(
    conn,
    chunk_id: int,
    collection_slug: str,
    document_id: int,
    page_no: Optional[int],
    text: str,
    lexicon: LexiconV10,
    alias_table_cache: Optional[Dict[str, List[SpanCandidate]]] = None,
) -> ChunkMentionsV10:
    """Extract structured mentions + signals from a single chunk.

    Algorithm:
    1. Always extract entity surfaces (canonical names from lexicon)
    2. Extract alias surfaces ONLY if collection is Venona/Vassiliev (I1)
    3. For each alias mention, call resolve_alias_candidates() for contextual resolution
    4. Detect signals (alias_equation, aka, etc.)

    Returns ChunkMentionsV10 with embedded document_id and page_no
    for ThinkDeeper rehydration independence.
    """
    result = ChunkMentionsV10(
        chunk_id=chunk_id,
        collection_slug=collection_slug,
        document_id=document_id,
        page_no=page_no,
    )

    if not text or not text.strip():
        return result

    # Step 1: Entity surface extraction (always)
    entity_mentions = _find_entity_surfaces(text, lexicon)
    result.mentions.extend(entity_mentions)

    # Step 2: Alias surface extraction (Venona/Vassiliev only)
    alias_mentions = _find_alias_surfaces(text, collection_slug, lexicon)
    result.mentions.extend(alias_mentions)

    # Step 3: Resolve alias mentions via central resolver
    context = AliasContext(
        collection_slug=collection_slug,
        document_id=document_id,
        page_no=page_no,
    )
    for mention in result.mentions:
        if mention.kind == "alias_surface":
            try:
                resolved = resolve_alias_candidates(
                    conn, mention.surface, context, lexicon,
                    alias_table_cache=alias_table_cache,
                )
                mention.resolved = resolved
            except Exception as e:
                logger.debug("Alias resolution failed for '%s': %s", mention.surface, e)

    # Step 4: Signal detection
    signals = _detect_signals(text, collection_slug)
    result.signals.extend(signals)

    # Deduplicate mentions by (surface, start, end)
    seen: Set[Tuple[str, int, int]] = set()
    unique_mentions: List[ChunkMention] = []
    for m in result.mentions:
        key = (m.surface.lower(), m.start, m.end)
        if key not in seen:
            seen.add(key)
            unique_mentions.append(m)
    result.mentions = unique_mentions

    return result


# Backward-compat alias
extract_chunk_mentions_v10 = extract_chunk_mentions_v10_deterministic


# =============================================================================
# LLM-guided surface extraction
# =============================================================================

def llm_extract_surfaces_v10(
    client,
    chunk_text: str,
    context: ExtractionContext,
    model: str = "gpt-4.1-mini-2025-04-14",
    temperature: float = 0.0,
) -> Tuple[List[ExtractedSurface], List[ExtractedSignal]]:
    """LLM-guided surface extraction from a single chunk (best-effort).

    Returns (surfaces, signals). Validates verbatim spans.
    Invalid surfaces are discarded individually. Whatever valid surfaces
    remain are returned — NEVER triggers full fallback. With V10.2,
    extraction is advisory; surface discovery is driven by the mention index.
    """
    from retrieval.agent.v10_prompts import (
        V10_EXTRACTION_PROMPT,
        V10_EXTRACTION_SCHEMA,
        BLOCKED_ALIAS_LIKE,
    )

    collection_desc = "alias-scoped (Venona/Vassiliev)" if context.is_alias_scoped else "standard"
    prompt = V10_EXTRACTION_PROMPT.format(
        collection_slug=context.collection_slug,
        collection_description=collection_desc,
        document_id=context.document_id,
        page_no=context.page_no or "unknown",
        known_entities=", ".join(context.known_entities[:30]) or "none",
        known_aliases=", ".join(context.known_aliases[:50]) or "none",
        blocked_alias_like=", ".join(context.blocked_alias_like or BLOCKED_ALIAS_LIKE),
    )

    import random
    import time as _time

    max_retries = 3
    base_delay = 1.0
    raw = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk_text[:3000]},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": V10_EXTRACTION_SCHEMA,
                },
            )
            raw = json.loads(response.choices[0].message.content)
            break
        except Exception as e:
            is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
            if is_rate_limit and attempt < max_retries:
                retry_after = getattr(e, "retry_after", None)
                delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
                jitter = random.uniform(0, delay * 0.3)
                wait_s = delay + jitter
                logger.info(
                    "LLM extraction 429 (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, wait_s, e,
                )
                print(
                    f"[V10] Extraction rate limit (429) — retry {attempt + 1}/{max_retries + 1} in {wait_s:.1f}s...",
                    file=sys.stderr,
                    flush=True,
                )
                _time.sleep(wait_s)
                continue
            logger.warning("LLM extraction failed: %s", e)
            return [], []
    if raw is None:
        return [], []

    # Use the exact text we sent to the LLM for validation (avoids OOB and consistency)
    text_for_validation = chunk_text[:3000]

    # Parse and validate surfaces
    surfaces: List[ExtractedSurface] = []
    valid_count = 0
    total_count = 0
    for s in raw.get("surfaces", []):
        total_count += 1
        es = ExtractedSurface(
            text=s.get("text", ""),
            start=s.get("start", 0),
            end=s.get("end", 0),
            kind=s.get("kind", "entity_surface"),
            confidence=s.get("confidence", 0.0),
            rationale=s.get("rationale", ""),
        )
        # Verbatim span validation: slice must exactly equal returned text
        def _slice_matches(s_start: int, s_end: int) -> bool:
            if s_start < 0 or s_end > len(text_for_validation) or s_start >= s_end:
                return False
            return text_for_validation[s_start:s_end] == es.text

        if _slice_matches(es.start, es.end):
            valid_count += 1
            surfaces.append(es)
        else:
            # Many models use 1-based or end-inclusive indexing; try fallbacks
            fixed = False
            for (try_start, try_end) in [
                (es.start - 1, es.end),      # 1-based start
                (es.start, es.end + 1),      # end-inclusive
                (es.start - 1, es.end - 1),   # 1-based start, end-inclusive
            ]:
                if _slice_matches(try_start, try_end):
                    es = ExtractedSurface(
                        text=es.text,
                        start=try_start,
                        end=try_end,
                        kind=es.kind,
                        confidence=es.confidence,
                        rationale=es.rationale,
                    )
                    valid_count += 1
                    surfaces.append(es)
                    fixed = True
                    break
            if not fixed:
                logger.debug(
                    "Discarding invalid surface '%s' at [%d:%d] (expected '%s')",
                    es.text, es.start, es.end,
                    text_for_validation[es.start:es.end] if es.start >= 0 and es.end <= len(text_for_validation) else "<OOB>",
                )

    # Best-effort: log invalid surfaces but NEVER trigger fallback.
    # With V10.2 mention-index architecture, extraction is advisory only;
    # surface discovery is driven by PEM candidates + agentic tools.
    if total_count > 0 and valid_count < total_count:
        invalid_count = total_count - valid_count
        pct = round(100.0 * invalid_count / total_count, 1)
        logger.info(
            "LLM extraction best-effort: %d/%d surfaces invalid (%.1f%%) — "
            "returning %d valid (no fallback)",
            invalid_count, total_count, pct, valid_count,
        )

    # HARDENING: Hard-discard alias_surface results when NOT alias-scoped
    if not context.is_alias_scoped:
        pre_count = len(surfaces)
        surfaces = [s for s in surfaces if s.kind != "alias_surface"]
        if pre_count != len(surfaces):
            logger.debug(
                "Discarded %d alias_surface results from non-alias-scoped collection '%s'",
                pre_count - len(surfaces), context.collection_slug,
            )

    # Parse signals
    signals: List[ExtractedSignal] = []
    for sig in raw.get("signals", []):
        signals.append(ExtractedSignal(
            signal_type=sig.get("type", ""),
            alias=sig.get("alias", ""),
            entity_name=sig.get("entity_name", ""),
            text=sig.get("text", ""),
            confidence=sig.get("confidence", 0.0),
        ))

    return surfaces, signals


# =============================================================================
# Deterministic surface-to-mention mapping
# =============================================================================

def map_surfaces_v10(
    conn,
    surfaces: List[ExtractedSurface],
    signals: List[ExtractedSignal],
    chunk_id: int,
    collection_slug: str,
    document_id: int,
    page_no: Optional[int],
    lexicon: LexiconV10,
    alias_table_cache: Optional[Dict[str, List[SpanCandidate]]] = None,
) -> ChunkMentionsV10:
    """Map raw LLM-extracted surfaces to ChunkMentionsV10.

    Uses resolve_alias_candidates() for alias surfaces.
    Surfaces are stable; mappings change as lexicon improves.
    """
    result = ChunkMentionsV10(
        chunk_id=chunk_id,
        collection_slug=collection_slug,
        document_id=document_id,
        page_no=page_no,
    )

    context = AliasContext(
        collection_slug=collection_slug,
        document_id=document_id,
        page_no=page_no,
    )

    for surface in surfaces:
        if surface.kind == "alias_surface" and collection_slug in ALIAS_SCOPED_COLLECTIONS:
            # Resolve alias via central resolver
            candidates: List[SpanCandidate] = []
            resolved: Optional[ResolvedAlias] = None
            try:
                resolved = resolve_alias_candidates(
                    conn, surface.text, context, lexicon,
                    alias_table_cache=alias_table_cache,
                )
                candidates = resolved.candidates if resolved else []
            except Exception as e:
                logger.debug("Alias resolution failed for '%s': %s", surface.text, e)

            mention = ChunkMention(
                surface=surface.text,
                start=surface.start,
                end=surface.end,
                kind="alias_surface",
                candidates=candidates,
                resolved=resolved,
            )
            result.mentions.append(mention)

        elif surface.kind == "entity_surface":
            # Entity surface — try to match to lexicon entities
            candidates = []
            for eid, info in lexicon.entities_in_play.items():
                canonical = info.get("canonical_name", "")
                if canonical and surface.text.lower() in canonical.lower() or canonical.lower() in surface.text.lower():
                    candidates.append(SpanCandidate(
                        entity_id=eid,
                        canonical_name=canonical,
                        match_type="canonical",
                        valid_collections=["*"],
                        source="lexicon",
                    ))
                    break

            mention = ChunkMention(
                surface=surface.text,
                start=surface.start,
                end=surface.end,
                kind="entity_surface",
                candidates=candidates,
            )
            result.mentions.append(mention)

    # Map signals to ChunkSignal
    for sig in signals:
        result.signals.append(ChunkSignal(
            signal_type=sig.signal_type,
            text=sig.text[:200],
            confidence=sig.confidence,
            entity_a=sig.alias.upper() if sig.alias else None,
            entity_b=sig.entity_name if sig.entity_name else None,
        ))

    return result


# =============================================================================
# Extraction dispatcher (routes LLM vs deterministic, with cache)
# =============================================================================

def extract_mentions_dispatched(
    conn,
    client,
    chunk_id: int,
    collection_slug: str,
    document_id: int,
    page_no: Optional[int],
    text: str,
    lexicon: LexiconV10,
    mode: str = "llm",
    surfaces_cache: Optional[Dict[Tuple, Tuple[List[ExtractedSurface], List[ExtractedSignal]]]] = None,
    alias_table_cache: Optional[Dict[str, List[SpanCandidate]]] = None,
) -> ChunkMentionsV10:
    """Dispatch extraction: LLM (alias-scoped) or deterministic, with cache.

    Cache key: (chunk_id, EXTRACTOR_VERSION, text_hash).
    Cache stores (surfaces, signals) together.
    text_hash ensures cache invalidation on text changes or snippet vs full text.

    Routes:
    1. Cached -> re-map with current lexicon (no LLM call)
    2. LLM path (alias-scoped + client available) -> extract, validate, cache, map
       V10.2: LLM results are best-effort. If extraction returns nothing,
       we return empty mentions — NO silent fallback to deterministic.
    3. Deterministic path — only when mode="deterministic" explicitly
    """
    import hashlib
    from retrieval.agent.v10_prompts import EXTRACTOR_VERSION, BLOCKED_ALIAS_LIKE

    text_hash = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    cache_key = (chunk_id, EXTRACTOR_VERSION, text_hash)

    # 1. Check cache
    if surfaces_cache is not None and cache_key in surfaces_cache:
        cached_surfaces, cached_signals = surfaces_cache[cache_key]
        return map_surfaces_v10(
            conn, cached_surfaces, cached_signals,
            chunk_id, collection_slug, document_id, page_no, lexicon,
            alias_table_cache=alias_table_cache,
        )

    # 2. LLM path (alias-scoped collections only)
    is_alias_scoped = collection_slug in ALIAS_SCOPED_COLLECTIONS
    if mode == "llm" and is_alias_scoped and client is not None:
        # Build extraction context
        known_entities = [
            info.get("canonical_name", "")
            for info in lexicon.entities_in_play.values()
        ][:30]

        known_aliases: List[str] = []
        for eid in lexicon.entities_in_play:
            for coll_aliases in lexicon.aliases_by_entity_scoped.get(eid, {}).values():
                known_aliases.extend(coll_aliases)
        known_aliases = list(set(known_aliases))[:50]

        ctx = ExtractionContext(
            collection_slug=collection_slug,
            document_id=document_id,
            page_no=page_no,
            known_entities=known_entities,
            known_aliases=known_aliases,
            is_alias_scoped=True,
            blocked_alias_like=list(BLOCKED_ALIAS_LIKE),
        )

        surfaces, signals = llm_extract_surfaces_v10(client, text, ctx)

        # Always cache and use LLM results (even partial/empty).
        # V10.2: extraction is best-effort advisory. NEVER fall through to
        # deterministic as a hidden control-flow fallback — that fights the
        # "surfaces are hypotheses grounded by mention index" philosophy.
        if surfaces_cache is not None:
            surfaces_cache[cache_key] = (surfaces, signals)
        if surfaces or signals:
            return map_surfaces_v10(
                conn, surfaces, signals,
                chunk_id, collection_slug, document_id, page_no, lexicon,
                alias_table_cache=alias_table_cache,
            )
        # LLM returned nothing — return empty mentions (no silent fallback)
        logger.debug(
            "LLM extraction returned empty for chunk %d — no fallback to deterministic",
            chunk_id,
        )
        return ChunkMentionsV10(
            chunk_id=chunk_id,
            collection_slug=collection_slug,
            document_id=document_id,
            page_no=page_no,
        )

    # 3. Deterministic path — only when explicitly requested (mode != "llm")
    return extract_chunk_mentions_v10_deterministic(
        conn, chunk_id, collection_slug, document_id, page_no, text, lexicon,
        alias_table_cache=alias_table_cache,
    )
