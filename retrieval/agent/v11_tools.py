"""
V11 Tools — stripped-down V9 (no concordance expansion, no query entity resolution).

- search_chunks: hybrid/lexical_exact WITHOUT expand_concordance, no resolution
- fetch_chunks: same as V9
- search_broad: per-collection shortlist (top 2 each) for broad scan
- No expand_entities (canonical embeddings handle alias retrieval)
"""
import os
from typing import List, Dict, Any, Optional

from retrieval.agent.tools import (
    hybrid_search_tool,
    lexical_exact_tool,
    ToolResult,
)
from retrieval.agent.v9_types import CatalogHit, WorkspaceChunk, ScopeFilter


def _resolve_page_ids(conn, page_ids: List[int]) -> Dict[int, int]:
    """Batch-resolve page row IDs to actual PDF page numbers."""
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
    if not page_id:
        return None
    if page_map and page_id in page_map:
        return f"p{page_map[page_id]}"
    # NEVER fall back to the raw page_id: it is a global DB key, not a page number, and using it
    # produces broken viewer links (e.g. "Page 3846 / 139"). Omit the page instead.
    return None


def _load_catalog(conn, chunk_ids: List[int], scores: Dict[int, float]) -> List[CatalogHit]:
    """Load snippet + metadata for chunk IDs."""
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


def search_chunks(
    conn,
    query: str,
    top_k: int = 50,
    collections: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    scope: Optional[ScopeFilter] = None,
    mode: str = "hybrid",
    use_canonical: bool = False,
    expand_concordance: bool = False,
) -> tuple[ToolResult, List[CatalogHit]]:
    """
    Search the archive.

    V11 default: no concordance expansion, no query entity resolution.
    V13 passes expand_concordance=True to bridge canonical<->codename on the FTS side.
    Returns (ToolResult, list of CatalogHit).
    """
    document_ids = None
    if scope and not scope.is_empty():
        if scope.document_ids:
            document_ids = scope.document_ids
        elif scope.collections and not collections:
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
            expand_aliases=False,  # V11: no alias expansion
        )
    else:
        result = hybrid_search_tool(
            conn=conn, query=query.strip(), top_k=top_k,
            collections=collections, document_ids=document_ids,
            date_from=date_from, date_to=date_to,
            expand_concordance=expand_concordance,  # V11: False; V13: True
            fuzzy_enabled=True,
            use_canonical_embeddings=use_canonical,
        )

    if not result.chunk_ids or not result.success:
        return result, []

    catalog = _load_catalog(conn, result.chunk_ids, result.scores)
    # Diagnostics: collections seen in returned top-k
    from collections import Counter
    coll_counts = Counter(h.collection or h.collection_slug or "unknown" for h in catalog)
    result.metadata["collections_seen_in_top_k"] = dict(coll_counts)
    return result, catalog


CODENAME_COLLECTIONS = ("venona", "vassiliev")


def resolve_codenames(
    conn,
    terms: List[str],
    top_k_per_term: int = 20,
    scope: Optional[ScopeFilter] = None,
) -> tuple[Dict[str, Any], List[CatalogHit]]:
    """
    Resolve codename tokens in V/V using lexical_exact + canonical search.
    Cheap and targeted: pass individual terms (PAL, LIBERAL, Silvermaster), not full questions.
    Returns (result_dict, flat_catalog) for merge into workspace.
    """
    vv = list(CODENAME_COLLECTIONS)
    if scope and scope.collections:
        vv = [c for c in vv if c in {x.lower() for x in scope.collections}]
    if not vv:
        return {"terms": terms, "mappings": {}, "catalog": []}, []

    mappings: Dict[str, Dict[str, Any]] = {}
    all_chunk_ids: set = set()
    all_scores: Dict[int, float] = {}
    catalog_hits: List[CatalogHit] = []

    for t in [x.strip() for x in terms if x and str(x).strip()][:10]:
        if not t:
            continue
        lex_result = lexical_exact_tool(
            conn=conn, term=t, top_k=top_k_per_term,
            collections=vv, expand_aliases=False,
        )
        can_result, can_catalog = search_chunks(
            conn, query=t, top_k=top_k_per_term,
            collections=vv, scope=scope, mode="hybrid",
            use_canonical=True,
        )
        lex_ids = set(lex_result.chunk_ids or [])
        can_ids = set(can_result.chunk_ids or [])
        combined = lex_ids | can_ids
        for cid in combined:
            all_chunk_ids.add(cid)
            all_scores[cid] = max(
                all_scores.get(cid, 0),
                lex_result.scores.get(cid, 0),
                can_result.scores.get(cid, 0),
            )
        mappings[t] = {
            "chunk_ids": sorted(combined),
            "lexical_hits": len(lex_ids),
            "canonical_hits": len(can_ids),
        }
        catalog_hits.extend(can_catalog)
        if lex_result.chunk_ids:
            lex_cat = _load_catalog(conn, lex_result.chunk_ids, lex_result.scores)
            catalog_hits.extend(lex_cat)

    # Dedupe catalog by chunk_id, preserve order
    seen = set()
    deduped: List[CatalogHit] = []
    for h in catalog_hits:
        if h.chunk_id not in seen:
            seen.add(h.chunk_id)
            deduped.append(h)

    return {
        "terms": terms,
        "mappings": mappings,
        "total_chunk_ids": len(all_chunk_ids),
        "catalog_preview": [
            {"term": t, "chunk_ids": m["chunk_ids"][:10], "lexical": m["lexical_hits"], "canonical": m["canonical_hits"]}
            for t, m in mappings.items()
        ],
    }, deduped


def search_broad(
    conn,
    query: str,
    top_per_collection: int = 2,
    top_k: int = 400,
    scope: Optional[ScopeFilter] = None,
) -> Dict[str, Any]:
    """
    Broad scan: per-collection shortlist (top N each) so agent sees entry points
    from other corpora. One query. Use when you want to see if other corpora
    have relevant hits before diving deep.
    """
    default_top_k = int(os.getenv("SEARCH_BROAD_TOP_K", "400"))
    top_k = top_k if top_k else default_top_k
    _, catalog = search_chunks(
        conn, query, top_k=top_k, scope=scope, mode="hybrid",
    )
    by_coll: Dict[str, List[CatalogHit]] = {}
    for h in catalog:
        c = h.collection or h.collection_slug or ""
        if c not in by_coll:
            by_coll[c] = []
        if len(by_coll[c]) < top_per_collection:
            by_coll[c].append(h)
    collections_out = [
        {"slug": c, "hits": [{"chunk_id": h.chunk_id, "score": h.score, "page": h.page, "snippet": h.snippet[:100]} for h in hits]}
        for c, hits in sorted(by_coll.items())
    ]
    return {
        "collections": collections_out,
        "total_hits": len(catalog),
        "flat_catalog": catalog,
    }


def expand_query(
    conn,
    query: str,
    scope: Optional[ScopeFilter] = None,
) -> tuple[List[str], Dict[str, Any]]:
    """
    Expand query terms via concordance within scope.
    Returns (expanded_terms, metadata). Agent can then call search/search_canonical with expanded query.
    """
    from retrieval.ops import concordance_expand_terms

    terms: List[str] = []
    meta: Dict[str, Any] = {"original_query": query, "expanded_terms": []}
    try:
        terms = concordance_expand_terms(conn=conn, text=query.strip(), max_aliases_out=25)
        meta["expanded_terms"] = terms[:15]
    except Exception as e:
        meta["error"] = str(e)
    return terms, meta


def expand_from_evidence(
    conn,
    chunk_ids: List[int],
    scope: Optional[ScopeFilter] = None,
) -> tuple[List[str], Dict[str, Any]]:
    """
    Extract names/codenames from retrieved chunks, then concordance-expand.
    Preserves lead-chasing without automatic bias. Returns (expanded_terms, metadata).
    """
    from retrieval.agent.v9_dispatch import _extract_expansion_entities
    from retrieval.config import CONCORDANCE_EXPANSION_TARGET_COLLECTIONS

    meta: Dict[str, Any] = {"chunk_ids": chunk_ids[:20], "entities": [], "expanded_terms": []}
    scope_collections = scope.collections if scope and scope.collections else None
    if scope_collections:
        target = [c for c in scope_collections if c in CONCORDANCE_EXPANSION_TARGET_COLLECTIONS]
    else:
        target = list(CONCORDANCE_EXPANSION_TARGET_COLLECTIONS)

    if not target:
        return [], meta

    entities = _extract_expansion_entities(conn, chunk_ids)
    meta["entities"] = [
        {"id": e["id"], "canonical_name": e["canonical_name"], "aliases": e.get("aliases", [])}
        for e in entities[:10]
    ]
    canonical_names = [e["canonical_name"] for e in entities[:8] if e.get("canonical_name")]
    return canonical_names, meta


def fetch_diverse_from_catalog(
    conn,
    workspace: Any,
    total: int = 20,
    per_collection: int = 3,
) -> List[WorkspaceChunk]:
    """
    Fetch chunks from catalog with collection diversity.
    Prefers unfetched. Takes up to per_collection from each collection, round-robin.
    Use after search_broad to avoid fetching 30/30 from one collection.
    """
    meth = getattr(workspace, "fulltext_chunk_ids", None)
    ft_ids = set(meth() if callable(meth) else (meth or [])) if meth else set()
    if not ft_ids and hasattr(workspace, "fulltext_chunks"):
        ft_ids = {c.chunk_id for c in workspace.fulltext_chunks}
    unfetched = [h for h in getattr(workspace, "catalog_hits", []) if h.chunk_id not in ft_ids]
    if not unfetched:
        return []
    by_coll: Dict[str, List[CatalogHit]] = {}
    for h in unfetched:
        c = h.collection or h.collection_slug or "unknown"
        by_coll.setdefault(c, []).append(h)
    colls = list(by_coll.keys())
    selected: List[int] = []
    for _ in range(per_collection):
        for c in colls:
            if by_coll[c] and len(selected) < total:
                selected.append(by_coll[c].pop(0).chunk_id)
        if len(selected) >= total:
            break
    while len(selected) < total and any(by_coll[c] for c in colls):
        for c in colls:
            if len(selected) >= total:
                break
            if by_coll[c]:
                selected.append(by_coll[c].pop(0).chunk_id)
    if not selected:
        return []
    return fetch_chunks(conn, chunk_ids=selected[:total])


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
    """Load full text. Same as V9."""
    from retrieval.agent.v9_tools import fetch_chunks as _v9_fetch
    return _v9_fetch(
        conn,
        chunk_ids=chunk_ids,
        doc_id=doc_id,
        around_chunk_id=around_chunk_id,
        window=window,
        page_start=page_start,
        page_end=page_end,
        include_neighbors=include_neighbors,
        neighbor_before=neighbor_before,
        neighbor_after=neighbor_after,
    )
