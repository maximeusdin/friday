"""
V9 Tools (V9.3) - Concordance integrated into every tool.

V9.3 changes:
- search_chunks: mode parameter ("hybrid" | "lexical_exact")
- Concordance resolution returns EntityCandidate, not WorkspaceEntity
- resolve_question_entities returns candidates (not committed entities)

Existing:
- search_chunks  -> returns snippet catalog + auto-resolves query terms via concordance
- fetch_chunks   -> auto-neighbors + doc_id/around_chunk_id modes
- expand_entities -> accepts names or entity_ids; include_comentions for roster power
"""
import re
import time
from typing import List, Dict, Any, Optional

from retrieval.agent.tools import (
    hybrid_search_tool,
    lexical_exact_tool,
    entity_surfaces_tool,
    entity_mentions_tool,
    co_mention_entities_tool,
    ToolResult,
    _lookup_entity_by_name,
)
from retrieval.agent.v9_types import CatalogHit, WorkspaceChunk, WorkspaceEntity, EntityCandidate, ScopeFilter


# =============================================================================
# Concordance: resolve query/question terms to entity candidates
# =============================================================================

def _resolve_query_entities(
    conn,
    text: str,
    *,
    max_candidates: int = 20,
    scope: Optional[ScopeFilter] = None,
) -> List[Dict[str, Any]]:
    """
    Extract candidate terms from text and resolve each via PEM (page_entity_mentions).
    Returns list of {query_term, entity_id, canonical_name, entity_type, matched_via},
    deduped by entity_id. See docs/entity_resolution_pem_only.md.
    """
    if not text or not text.strip():
        return []

    # Stopwords: skip entity-linking for common words that produce noise
    _STOPWORDS_FOR_ENTITY_LINK = frozenset({
        "a", "an", "the", "with", "of", "for", "to", "in", "on", "at", "by", "from",
        "and", "or", "associated", "activities", "ends", "replaced", "agency",
        "that", "this", "it", "as", "be", "has", "had", "not", "but", "who", "what",
        "which", "where", "when", "how", "did", "does", "do", "is", "are", "was", "were",
    })

    text = text.strip()
    candidates: List[str] = []

    # Quoted strings first (e.g. "Pal", 'LIBERAL')
    for m in re.finditer(r'["\']([^"\']{1,50})["\']', text):
        candidates.append(m.group(1).strip())

    # Single words: 2+ chars, allow digits (e.g. Pal, LIBERAL, NKVD)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,24}", text)
    for w in words:
        if w not in candidates and len(w) >= 2:
            candidates.append(w)

    # Two-word phrases (Title Case or ALL CAPS)
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if phrase not in candidates:
            candidates.append(phrase)

    seen_entity_ids: set = set()
    out: List[Dict[str, Any]] = []

    for term in candidates[:max_candidates]:
        if not term or len(term) < 2:
            continue
        if term.lower() in _STOPWORDS_FOR_ENTITY_LINK:
            continue
        try:
            entity_id, canonical_name, entity_type, matched_via = _lookup_entity_by_name(conn, term, scope=scope)
            if entity_id and entity_id not in seen_entity_ids:
                seen_entity_ids.add(entity_id)
                out.append({
                    "query_term": term,
                    "entity_id": entity_id,
                    "canonical_name": canonical_name or "",
                    "entity_type": entity_type,
                    "matched_via": matched_via or "",
                })
        except Exception:
            continue

    return out


def _confidence_from_matched_via(matched_via: str) -> str:
    """Derive confidence level from the matched_via string.

    Returns: "exact" | "partial" | "concordance" | "inferred"
    """
    mv = (matched_via or "").lower()
    if "exact" in mv or "canonical" in mv or "alias" in mv or "pem" in mv:
        return "exact"
    if "partial" in mv:
        return "partial"
    if "concordance" in mv:
        return "concordance"
    return "inferred"


def resolve_question_entities(
    conn,
    question: str,
    scope: Optional[ScopeFilter] = None,
) -> Dict[str, Any]:
    """
    Resolve all entity-like terms in the question via PEM.
    Returns entity candidates (NOT committed entities) for workspace priming.
    Does NOT call expand_entities or load mention chunks -- that's the agent's job.

    Each candidate includes a confidence level derived from matched_via.
    """
    resolved = _resolve_query_entities(conn, question, scope=scope)
    if not resolved:
        return {"candidates": [], "resolution": []}

    # Detect ambiguity: multiple candidates for the same query_term
    term_counts: Dict[str, int] = {}
    for r in resolved:
        qt = r["query_term"].lower()
        term_counts[qt] = term_counts.get(qt, 0) + 1

    candidates = [
        EntityCandidate(
            query_term=r["query_term"],
            entity_id=r["entity_id"],
            canonical_name=r["canonical_name"],
            entity_type=r.get("entity_type"),
            matched_via=r.get("matched_via", ""),
            accepted=False,
            confidence=_confidence_from_matched_via(r.get("matched_via", "")),
            ambiguous=term_counts.get(r["query_term"].lower(), 1) > 1,
        )
        for r in resolved
    ]

    resolution = [
        {"query_term": r["query_term"], "canonical_name": r["canonical_name"], "entity_id": r["entity_id"]}
        for r in resolved
    ]
    return {
        "candidates": candidates,
        "resolution": resolution,
    }


# =============================================================================
# search_chunks  (mode: hybrid | lexical_exact; concordance resolution on query)
# =============================================================================

def search_chunks(
    conn,
    query: str,
    top_k: int = 50,
    collections: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    scope: Optional[ScopeFilter] = None,
    mode: str = "hybrid",
    resolution_query: Optional[str] = None,
) -> tuple[ToolResult, List[CatalogHit]]:
    """
    Search the archive.  Returns (ToolResult, list of CatalogHit).

    resolution_query: when provided (e.g. original query before alias expansion),
    use it for concordance resolution instead of query. Avoids redundant lookups
    for expansion terms we already know.
    Each CatalogHit has chunk_id, score, doc_id, page, collection, snippet (~300 chars).
    If scope is provided, its collections/date filters are merged in (enforced).

    mode:
      "hybrid" (default) - semantic + lexical hybrid search with concordance expansion
      "lexical_exact" - exact substring match with alias expansion (good for finding
                        exact phrases, mapping lines like '"Pal" -- Silvermaster')
    """
    # Enforce scope: when provided, it truly limits — caller cannot escape
    # Precedence: document_ids > collections (applied in _build_where)
    document_ids = None
    if scope and not scope.is_empty():
        if scope.document_ids:
            document_ids = scope.document_ids
        elif scope.collections:
            if collections:
                in_scope = set(scope.collections)
                intersected = [c for c in collections if c in in_scope]
                collections = intersected if intersected else scope.collections
            else:
                collections = scope.collections
        if scope.date_from and not date_from:
            date_from = scope.date_from
        if scope.date_to and not date_to:
            date_to = scope.date_to

    if not query or not str(query).strip():
        return (
            ToolResult(
                tool_name="search_chunks", params={"query": query, "top_k": top_k},
                chunk_ids=[], scores={}, metadata={"message": "Empty query"},
                elapsed_ms=0, success=True,
            ),
            [],
        )
    top_k = min(max(int(top_k) if top_k else 50, 1), 500)

    if mode == "lexical_exact":
        result = lexical_exact_tool(
            conn=conn, term=query.strip(), top_k=top_k,
            collections=collections, document_ids=document_ids,
            expand_aliases=True,
        )
    else:
        result = hybrid_search_tool(
            conn=conn, query=query.strip(), top_k=top_k,
            collections=collections, document_ids=document_ids,
            date_from=date_from, date_to=date_to,
            expand_concordance=True, fuzzy_enabled=True,
        )

    # Resolution: use original query only (skip expansion terms we already know)
    text_for_resolution = (resolution_query or query).strip()

    if not result.chunk_ids or not result.success:
        # Still run concordance resolution so the model sees e.g. "Pal" -> Silvermaster
        resolution = _resolve_query_entities(conn, text_for_resolution)
        result.metadata["concordance_resolution"] = [
            {"query_term": r["query_term"], "canonical_name": r["canonical_name"], "entity_id": r["entity_id"]}
            for r in resolution
        ]
        return result, []

    # Load snippets + metadata for returned chunk_ids
    catalog = _load_catalog(conn, result.chunk_ids, result.scores)

    # Concordance: resolve entity-like terms from original query only (not expansion terms)
    resolution = _resolve_query_entities(conn, text_for_resolution)
    result.metadata["concordance_resolution"] = [
        {"query_term": r["query_term"], "canonical_name": r["canonical_name"], "entity_id": r["entity_id"]}
        for r in resolution
    ]
    return result, catalog


def _resolve_page_ids(conn, page_ids: List[int]) -> Dict[int, int]:
    """Batch-resolve page row IDs to actual PDF page numbers.

    Returns: {page_row_id: pdf_page_number}.  Missing/null entries are omitted.
    """
    if not page_ids:
        return {}
    unique_ids = list(set(pid for pid in page_ids if pid))
    if not unique_ids:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, COALESCE(pdf_page_number, page_seq) FROM pages WHERE id = ANY(%s)",
                (unique_ids,),
            )
            return {r[0]: r[1] for r in cur.fetchall() if r[1] is not None}
    except Exception:
        return {}


def _page_ref(page_id, page_map: Dict[int, int] = None) -> Optional[str]:
    """Format page reference using resolved PDF page number if available."""
    if not page_id:
        return None
    if page_map and page_id in page_map:
        return f"p{page_map[page_id]}"
    # NEVER emit the raw page_id (a global DB key) as a page number — it breaks the viewer
    # ("Page 3846 / 139"). Omit the page instead.
    return None


def _load_catalog(conn, chunk_ids: List[int], scores: Dict[int, float]) -> List[CatalogHit]:
    """Load snippet + metadata for chunk IDs (lightweight, ~300 chars each)."""
    if not chunk_ids:
        return []
    chunk_ids = list(chunk_ids)[:300]
    try:
        conn.rollback()
    except Exception:
        pass
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id,
                   LEFT(COALESCE(c.clean_text, c.text), 300),
                   cm.document_id,
                   COALESCE(p.pdf_page_number, p.page_seq) AS page_num,
                   cm.collection_slug
            FROM chunks c
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
            LEFT JOIN pages p ON p.id = cm.first_page_id
            WHERE c.id = ANY(%s)
        """, (chunk_ids,))
        rows = cur.fetchall()
    row_map = {r[0]: r for r in rows}
    out = []
    for cid in chunk_ids:
        r = row_map.get(cid)
        if not r:
            continue
        out.append(CatalogHit(
            chunk_id=cid,
            score=scores.get(cid, 0.0),
            doc_id=r[2],
            page=f"p{r[3]}" if r[3] else None,
            collection=r[4],
            snippet=(r[1] or "").strip(),
        ))
    return out


# =============================================================================
# fetch_chunks  (neighbor expansion + doc-slice modes)
# =============================================================================

def fetch_chunks(
    conn,
    chunk_ids: Optional[List[int]] = None,
    *,
    doc_id: Optional[int] = None,
    around_chunk_id: Optional[int] = None,
    window: int = 4,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    include_neighbors: bool = True,
    neighbor_before: int = 1,
    neighbor_after: int = 1,
) -> List[WorkspaceChunk]:
    """
    Load full text.  Three modes:

    1. By chunk_ids (default) -- auto-fetches +/-neighbor_before/after neighbors per chunk.
    2. By doc_id + around_chunk_id + window -- reads a contiguous slice of a document.
    3. By doc_id + page_start/page_end -- reads pages of a document.

    Returns list of WorkspaceChunk with is_neighbor flag on auto-fetched neighbors.
    """
    try:
        conn.rollback()
    except Exception:
        pass

    # -- Mode 2: doc slice around a chunk --
    if doc_id and around_chunk_id:
        from retrieval.ops import get_chunk_neighbors
        nbrs = get_chunk_neighbors(conn, around_chunk_id, before=window // 2, after=window // 2, include_seed=True)
        # Resolve page_ids to real PDF page numbers
        pmap = _resolve_page_ids(conn, [n.page_id for n in nbrs if n.page_id])
        return [
            WorkspaceChunk(
                chunk_id=n.chunk_id, text=n.text or "",
                doc_id=n.document_id, page=_page_ref(n.page_id, pmap),
                source_label=n.collection_slug, collection_slug=n.collection_slug,
                is_neighbor=(n.chunk_id != around_chunk_id),
            )
            for n in nbrs
        ]

    # -- Mode 3: doc pages --
    if doc_id and (page_start is not None or page_end is not None):
        from retrieval.ops import get_document_chunks
        all_chunks = []
        ps = page_start or 1
        pe = page_end or ps
        for pid in range(ps, pe + 1):
            doc_chunks = get_document_chunks(conn, doc_id, page_id=pid, limit=30)
            for dc in doc_chunks:
                all_chunks.append(dc)
        # Resolve page_ids to real PDF page numbers
        pmap = _resolve_page_ids(conn, [dc.page_id for dc in all_chunks if dc.page_id])
        return [
            WorkspaceChunk(
                chunk_id=dc.chunk_id, text=dc.text or "",
                doc_id=dc.document_id, page=_page_ref(dc.page_id, pmap),
                source_label=dc.collection_slug, collection_slug=dc.collection_slug,
            )
            for dc in all_chunks
        ]

    # -- Mode 1: by chunk_ids with auto-neighbors --
    if not chunk_ids:
        return []
    chunk_ids = list(chunk_ids)[:200]

    # Fetch requested chunks (join pages table for real PDF page number)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id, COALESCE(c.clean_text, c.text), cm.document_id,
                   COALESCE(p.pdf_page_number, p.page_seq) AS page_num,
                   cm.collection_slug
            FROM chunks c
            LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
            LEFT JOIN pages p ON p.id = cm.first_page_id
            WHERE c.id = ANY(%s)
        """, (chunk_ids,))
        rows = cur.fetchall()

    primary = []
    for row in rows:
        primary.append(WorkspaceChunk(
            chunk_id=row[0], text=row[1] or "",
            doc_id=row[2], page=f"p{row[3]}" if row[3] else None,
            source_label=row[4], collection_slug=row[4],
        ))

    if not include_neighbors or (neighbor_before == 0 and neighbor_after == 0):
        return primary

    # Auto-fetch neighbors for each primary chunk
    from retrieval.ops import get_chunk_neighbors
    seen_ids = {c.chunk_id for c in primary}
    raw_neighbors = []
    for pc in primary:
        try:
            nbrs = get_chunk_neighbors(conn, pc.chunk_id, before=neighbor_before, after=neighbor_after, include_seed=False)
            for n in nbrs:
                if n.chunk_id not in seen_ids:
                    seen_ids.add(n.chunk_id)
                    raw_neighbors.append(n)
        except Exception:
            pass  # neighbor fetch is best-effort

    # Resolve page_ids to real PDF page numbers for neighbors
    pmap = _resolve_page_ids(conn, [n.page_id for n in raw_neighbors if n.page_id])
    neighbor_chunks = [
        WorkspaceChunk(
            chunk_id=n.chunk_id, text=n.text or "",
            doc_id=n.document_id, page=_page_ref(n.page_id, pmap),
            source_label=n.collection_slug, collection_slug=n.collection_slug,
            is_neighbor=True,
        )
        for n in raw_neighbors
    ]

    return primary + neighbor_chunks


# =============================================================================
# expand_entities (V9.2: accepts names[], include_comentions)
# =============================================================================

def expand_entities(
    conn,
    entity_ids: Optional[List[int]] = None,
    names: Optional[List[str]] = None,
    include_mentions: bool = True,
    include_comentions: bool = False,
    mentions_top_k: int = 50,
    comentions_top_k: int = 35,
    collections: Optional[List[str]] = None,
    scope: Optional[ScopeFilter] = None,
) -> Dict[str, Any]:
    """
    Resolve entity IDs or names to canonical names, aliases, mention chunk IDs,
    and optionally co-mentioned entity IDs (the "roster power move").

    Accepts EITHER entity_ids or names (or both). Names are resolved via
    entity_surfaces_tool (PEM-only; see docs/entity_resolution_pem_only.md).

    If include_comentions=True, also calls co_mention_entities_tool for each
    resolved entity and returns co-mentioned entities + their chunk IDs.
    """
    # Enforce scope: when provided, intersect with caller (caller cannot escape)
    if scope and not scope.is_empty():
        if scope.collections:
            if collections:
                in_scope = set(scope.collections)
                intersected = [c for c in collections if c in in_scope]
                collections = intersected if intersected else scope.collections
            else:
                collections = scope.collections

    entity_ids = list(entity_ids or [])[:20]
    names = list(names or [])[:20]

    if not entity_ids and not names:
        return {"entities": [], "errors": [], "co_entities": [], "chunk_ids": []}

    entities_out: List[Dict[str, Any]] = []
    errors: List[str] = []
    all_chunk_ids: List[int] = []
    co_entities_out: List[Dict[str, Any]] = []
    co_chunk_ids: List[int] = []
    resolved_ids: List[int] = []

    # -- Resolve names to entity_ids first --
    for name in names:
        if not name or not name.strip():
            continue
        # Use entity_surfaces_tool with name param to resolve via PEM
        res = entity_surfaces_tool(conn, name=name.strip())
        if res.success and res.metadata.get("canonical_name"):
            eid = res.metadata.get("entity_id")
            if eid and eid not in entity_ids:
                entity_ids.append(eid)
        else:
            # Fall back: try the mention tool directly by name
            from retrieval.agent.tools import entity_lookup_tool
            lookup = entity_lookup_tool(conn, name=name.strip())
            if lookup.success and lookup.metadata.get("entity_id"):
                eid = lookup.metadata["entity_id"]
                if eid not in entity_ids:
                    entity_ids.append(eid)
            else:
                errors.append(f"Could not resolve name '{name}' to entity")

    # -- Process each entity_id --
    for eid in entity_ids:
        try:
            eid = int(eid)
        except (ValueError, TypeError):
            errors.append(f"Invalid entity_id: {eid}")
            continue

        res = entity_surfaces_tool(conn, entity_id=eid)
        if not res.success or not res.metadata.get("canonical_name"):
            errors.append(f"Entity {eid} not found")
            continue

        canonical = res.metadata.get("canonical_name", "")
        aliases = list(res.metadata.get("surfaces", []))
        entity_type = res.metadata.get("entity_type")
        ent: Dict[str, Any] = {
            "entity_id": eid,
            "canonical_name": canonical,
            "aliases": aliases,
            "entity_type": entity_type,
        }

        # -- Mentions --
        if include_mentions:
            mentions_res = entity_mentions_tool(
                conn, entity_id=eid, top_k=mentions_top_k,
                collections=collections,
            )
            if mentions_res.success and mentions_res.chunk_ids:
                ent["mention_chunk_ids"] = mentions_res.chunk_ids
                all_chunk_ids.extend(mentions_res.chunk_ids)

        # -- Co-mentions (the roster power move) --
        if include_comentions:
            co_res = co_mention_entities_tool(
                conn, entity_id=eid, top_k=comentions_top_k,
                collections=collections,
            )
            if co_res.success:
                co_ents = co_res.metadata.get("co_entities", [])
                ent["co_entities"] = co_ents
                for ce in co_ents:
                    if ce not in co_entities_out:
                        co_entities_out.append(ce)
                if co_res.chunk_ids:
                    co_chunk_ids.extend(co_res.chunk_ids)

        resolved_ids.append(eid)
        entities_out.append(ent)

    # Dedupe chunk_ids
    all_chunk_ids = list(dict.fromkeys(all_chunk_ids))
    co_chunk_ids = list(dict.fromkeys(co_chunk_ids))

    # When no mention chunks but entities resolved, provide suggested retrieval queries
    # so the controller can RETRIEVE using aliases/codenames
    suggested_retrieval_queries: List[str] = []
    if not all_chunk_ids and entities_out:
        for ent in entities_out:
            canonical = ent.get("canonical_name", "")
            aliases = ent.get("aliases", [])
            if canonical:
                suggested_retrieval_queries.append(canonical)
            for a in aliases[:5]:  # cap aliases per entity
                if a and a not in suggested_retrieval_queries:
                    suggested_retrieval_queries.append(a)
        suggested_retrieval_queries = suggested_retrieval_queries[:10]  # cap total

    return {
        "entities": entities_out,
        "errors": errors,
        "chunk_ids": all_chunk_ids if include_mentions else [],
        "co_entities": co_entities_out,
        "co_chunk_ids": co_chunk_ids,
        "suggested_retrieval_queries": suggested_retrieval_queries,
    }
