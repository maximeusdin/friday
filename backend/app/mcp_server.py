"""
Friday MCP connector — remote MCP server exposing Friday's archive to Claude.

Mounted by app.main at /mcp (Streamable HTTP transport, stateless). Read-only and
unauthenticated by design: it exposes only public-archive retrieval — search,
document/page text, the concordance, and the collection list. Nothing session-
or user-scoped goes through here. Claude (claude.ai / Desktop / Claude Code) is
the reasoning and synthesis layer; these tools are its retrieval primitives.

Searches run through the same executor as the Search tab (alias expansion,
PEM enumeration, NL keyword-relax fallback) and persist as ordinary result sets
with origin='mcp' under a synthetic user, so they never appear in any real
user's UI but remain queryable for usage telemetry.

Claude.ai's Research feature expects a `search` + `fetch` tool pair, which is
why those two tools carry those exact names.
"""
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.services.db import get_conn

# Synthetic owner for result sets created by connector searches.
MCP_USER_SUB = "mcp-connector"

# Public site base for viewer links Claude can cite (opens the scanned page).
PUBLIC_SITE = os.getenv("FRIDAY_PUBLIC_URL", "https://fridayarchive.org").rstrip("/")

# The SDK's DNS-rebinding protection rejects any Host header not in this list
# with "421 Invalid Host header". Defaults cover only localhost, which blocks
# the public endpoint — allow prod + local dev (override via env if the domain
# ever moves).
_ALLOWED_HOSTS = [
    h.strip()
    for h in os.getenv(
        "FRIDAY_MCP_ALLOWED_HOSTS",
        "api.fridayarchive.org,localhost,127.0.0.1,localhost:8000,127.0.0.1:8000",
    ).split(",")
    if h.strip()
]

mcp = FastMCP(
    "Friday Archive",
    instructions=(
        "Friday is a research console over declassified Cold War intelligence "
        "archives: the Venona decrypts, Alexander Vassiliev's KGB notebooks, "
        "FBI files (Silvermaster, Comintern), congressional testimony, and "
        "related collections. Use `search` to find pages, `fetch` to read a "
        "document or page, and `lookup_entity` to resolve people and "
        "codenames (cover names) to their aliases — codenames are pervasive "
        "in this corpus, so resolve them before concluding a person is absent. "
        "Cite documents with their viewer_url so readers can open the scanned page."
    ),
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_ALLOWED_HOSTS,
        # Server-side MCP clients (claude.ai, Claude Code) send no Origin
        # header, which passes validation. These cover browser-based clients
        # like the MCP inspector; ":*" is the SDK's port-wildcard syntax.
        allowed_origins=[
            "https://claude.ai",
            "https://fridayarchive.org",
            "https://www.fridayarchive.org",
            "http://localhost:*",
            "http://127.0.0.1:*",
        ],
    ),
)


# =============================================================================
# search
# =============================================================================

def _resolve_slugs_to_ids(conn, slugs: List[str]) -> Dict[str, int]:
    """Map collection slugs to ids; silently drops unknown slugs."""
    if not slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT slug, id FROM collections WHERE slug = ANY(%s)", (list(slugs),))
        return {slug: cid for (slug, cid) in cur.fetchall()}


def _search_impl(
    query: str,
    scope_text: Optional[str],
    collection: Optional[str],
    mode: str,
    limit: int,
) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"error": "Query is required."}
    original_query = query
    if mode not in ("exact", "fuzzy"):
        mode = "exact"
    limit = max(1, min(limit, 50))

    from retrieval.agent.scope_nl import detect_nl_scope, resolve_scope_text, strip_nl_scope

    conn = get_conn()
    try:
        scope: Dict[str, Any] = {"mode": "full_archive"}
        scoped_slugs: List[str] = []
        notices: List[str] = []
        explicit_full_archive = False

        # 1. Free-text scope parameter ("the Rosenberg files and Venona").
        if scope_text and scope_text.strip():
            res = resolve_scope_text(conn, scope_text)
            if res.exclusion:
                return {
                    "error": (
                        f"Scope '{scope_text}' excludes collections, which isn't supported — "
                        "scopes are include-lists only."
                    ),
                    "hint": "Call list_collections and pass a scope naming the collections to INCLUDE.",
                }
            if res.full_archive:
                explicit_full_archive = True
            elif res.collections:
                scoped_slugs.extend(res.collections)
            else:
                colls = _list_collections_impl().get("collections", [])
                return {
                    "error": f"Could not match scope '{scope_text}' to any collection.",
                    "collections": [
                        {"slug": c["slug"], "title": c["title"]} for c in colls
                    ],
                    "hint": "Re-run with scope naming one or more of these collections (or 'full archive').",
                }

        # 2. Explicit slug parameter (kept for backward compatibility; unioned in).
        if collection:
            if collection not in _resolve_slugs_to_ids(conn, [collection]):
                return {
                    "error": f"Unknown collection slug '{collection}'.",
                    "hint": "Call list_collections to see valid slugs, or omit the parameter to search the full archive.",
                }
            if collection not in scoped_slugs:
                scoped_slugs.append(collection)

        # 3. Scope phrases inside the query itself ("Perlo group in the Venona
        #    decrypts"): always strip them from the retrieval query so they
        #    don't pollute the keyword match; apply them as scope only when no
        #    explicit 'full archive' was requested.
        det = detect_nl_scope(conn, query)
        if det.collections:
            stripped = strip_nl_scope(query, det.matched_phrases)
            # If nothing meaningful survives ("the Venona decrypts" -> "the"),
            # the query was pure scope: ask for search terms rather than
            # searching for stopwords.
            leftover = [
                w for w in re.findall(r"\w+", stripped.lower())
                if w not in ("the", "a", "an", "of", "in", "on", "for", "about", "from", "all")
            ]
            if not leftover:
                return {
                    "error": (
                        f"'{original_query}' names where to search "
                        f"({', '.join(det.collections)}) but not what to search for."
                    ),
                    "hint": "Re-run with search terms in query and the collection(s) in scope.",
                }
            query = stripped
            if explicit_full_archive:
                notices.append(
                    "Searched the full archive as requested; removed the scope wording "
                    f"({', '.join(det.matched_phrases)}) from the query terms."
                )
            else:
                new_slugs = [s for s in det.collections if s not in scoped_slugs]
                scoped_slugs.extend(new_slugs)
                if not scope_text and not collection:
                    notices.append(
                        f"Scoped to {', '.join(det.collections)} based on your query wording. "
                        "Pass scope='full archive' to search everything instead."
                    )
                elif new_slugs:
                    notices.append(
                        f"Also included {', '.join(new_slugs)}, named in the query wording."
                    )

        if scoped_slugs:
            slug_to_id = _resolve_slugs_to_ids(conn, scoped_slugs)
            missing = [s for s in scoped_slugs if s not in slug_to_id]
            if missing and not slug_to_id:
                return {
                    "error": f"Collections not found: {', '.join(missing)}.",
                    "hint": "Call list_collections to see what exists.",
                }
            if missing:
                notices.append(f"Ignored unknown collections: {', '.join(missing)}.")
            scoped_slugs = [s for s in scoped_slugs if s in slug_to_id]
            scope = {
                "mode": "custom",
                "included_collection_ids": [slug_to_id[s] for s in scoped_slugs],
            }

        result_set_id = str(uuid.uuid4())
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_result_sets
                (id, user_sub, session_id, scope_json, query_raw, mode, unit, sort_order,
                 alias_expand, is_exhaustive, status, origin, origin_query)
                VALUES (%s, %s, NULL, %s, %s, %s, 'page', 'canonical', true, %s, 'running', 'mcp', %s)
                """,
                (result_set_id, MCP_USER_SUB, json.dumps(scope), query, mode, mode == "exact", original_query),
            )
        conn.commit()

        from retrieval.search_executor import run_search

        out = run_search(conn, result_set_id, query, scope, alias_expand=True, mode=mode)
        if out.get("status") == "error":
            return {
                "error": out.get("error", "Search failed"),
                "hint": (
                    "Search matches exact terms (boolean AND/OR, quoted phrases). "
                    "Try fewer, more distinctive keywords, or mode='fuzzy' for OCR-garbled spellings."
                ),
            }

        # Same NL fallback the Search tab uses: a full sentence ANDs every word
        # (including verbs that never co-occur with the answer) and returns zero.
        if out.get("total_hits", 0) == 0:
            from app.routes.search import _keyword_relax, _looks_like_nl_query

            if _looks_like_nl_query(query):
                relaxed = _keyword_relax(query)
                if relaxed and relaxed.lower() != query.lower():
                    out2 = run_search(conn, result_set_id, relaxed, scope, alias_expand=True, mode=mode)
                    if out2.get("status") != "error" and out2.get("total_hits", 0) > 0:
                        out = out2
                        notices.append(
                            f'No exact matches for the full phrase; searched the keywords instead: "{relaxed}".'
                        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT h.document_id, d.source_name, col.slug, col.title,
                       COALESCE(h.pdf_page_number, h.page_seq) AS pdf_page,
                       h.snippet
                FROM search_result_page_hits h
                JOIN collections col ON col.id = h.collection_id
                JOIN documents d ON d.id = h.document_id
                WHERE h.result_set_id = %s
                ORDER BY h.collection_id, h.document_id, h.page_seq, h.page_id
                LIMIT %s
                """,
                (result_set_id, limit),
            )
            rows = cur.fetchall()

        results = [
            {
                "document_id": doc_id,
                "document": doc_title,
                "collection": col_title or col_slug,
                "collection_slug": col_slug,
                "pdf_page": pdf_page,
                "snippet": snippet,
                "viewer_url": f"{PUBLIC_SITE}/?document_id={doc_id}&pdf_page={pdf_page}",
            }
            for (doc_id, doc_title, col_slug, col_title, pdf_page, snippet) in rows
        ]

        total = out.get("total_hits", len(results))
        resp: Dict[str, Any] = {
            "query": query,
            "scope": {"collections": scoped_slugs} if scoped_slugs else "full_archive",
            "total_hits": total,
            "showing": len(results),
            "results": results,
        }
        if notices:
            resp["notice"] = " ".join(notices)
        if total > len(results):
            resp["note"] = (
                f"{total - len(results)} more hits not shown. Narrow the query, scope to a "
                "collection, or raise limit (max 50)."
            )
        return resp
    finally:
        conn.close()


@mcp.tool()
async def search(
    query: str,
    scope: Optional[str] = None,
    collection: Optional[str] = None,
    mode: str = "exact",
    limit: int = 15,
) -> Dict[str, Any]:
    """Search declassified Cold War intelligence archives (Venona decrypts,
    Vassiliev's KGB notebooks, FBI files, congressional testimony and more).

    Call this whenever a question touches Soviet espionage, the KGB/GRU, Venona,
    American communism, atomic spying, or any person or codename from that world.
    The engine matches exact keywords with boolean syntax: terms are ANDed,
    `OR` between terms, "quoted phrases" for exact phrases. Known aliases and
    codenames are expanded automatically. Prefer 1-3 distinctive terms over full
    sentences. Use mode='fuzzy' to catch OCR-garbled spellings of a name.

    To restrict where to search, pass `scope` in plain language — e.g.
    "the Venona decrypts and Vassiliev notebooks", "FBI Silvermaster file",
    "Rosenberg grand jury" or "full archive". Multiple collections are fine;
    the response echoes which collections were actually searched. If the user's
    request names its own scope ("...in the Venona cables"), that is detected
    automatically. `collection` (a single slug from list_collections) still
    works and is unioned in.

    Returns page-level hits with snippets; use `fetch` to read a full page or
    document, and cite viewer_url so readers can open the scanned page.
    """
    return await anyio.to_thread.run_sync(lambda: _search_impl(query, scope, collection, mode, limit))


# =============================================================================
# fetch
# =============================================================================

def _fetch_impl(document_id: int, pdf_page: Optional[int], max_chars: int) -> Dict[str, Any]:
    max_chars = max(1000, min(max_chars, 100_000))
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id, d.source_name, d.source_ref, d.volume,
                       c.slug, c.title,
                       (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id)
                FROM documents d
                JOIN collections c ON c.id = d.collection_id
                WHERE d.id = %s
                """,
                (document_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"error": f"Document {document_id} not found."}

        doc_id, source_name, source_ref, volume, col_slug, col_title, page_count = row
        doc: Dict[str, Any] = {
            "document_id": doc_id,
            "document": source_name,
            "collection": col_title or col_slug,
            "collection_slug": col_slug,
            "volume": volume,
            "page_count": page_count,
            "viewer_url": f"{PUBLIC_SITE}/?document_id={doc_id}",
        }

        if pdf_page is not None:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT raw_text FROM pages WHERE document_id = %s AND pdf_page_number = %s",
                    (document_id, pdf_page),
                )
                prow = cur.fetchone()
            if not prow:
                return {**doc, "error": f"Page {pdf_page} not found (document has {page_count} pages)."}
            doc["pdf_page"] = pdf_page
            doc["viewer_url"] = f"{PUBLIC_SITE}/?document_id={doc_id}&pdf_page={pdf_page}"
            doc["text"] = (prow[0] or "")[:max_chars]
            return doc

        # Whole document: concatenate pages with markers up to max_chars.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pdf_page_number, raw_text FROM pages
                WHERE document_id = %s
                ORDER BY pdf_page_number
                """,
                (document_id,),
            )
            pages = cur.fetchall()

        parts = []
        used = 0
        truncated_at = None
        for pnum, text in pages:
            block = f"--- page {pnum} ---\n{text or ''}\n"
            if used + len(block) > max_chars:
                truncated_at = pnum
                break
            parts.append(block)
            used += len(block)

        doc["text"] = "".join(parts)
        if truncated_at is not None:
            doc["truncated"] = True
            doc["note"] = (
                f"Text truncated at page {truncated_at} of {page_count}. "
                "Call fetch again with pdf_page set to read a specific page."
            )
        return doc
    finally:
        conn.close()


@mcp.tool()
async def fetch(document_id: int, pdf_page: Optional[int] = None, max_chars: int = 20000) -> Dict[str, Any]:
    """Fetch the OCR text of an archival document (or a single page of it) by
    document_id from `search` results.

    Pass pdf_page to read one page (best after a search hit); omit it to read the
    document from the start, truncated to max_chars. Returns metadata plus text
    and a viewer_url that opens the scanned original — include that URL when
    citing. OCR of 1940s typescript and handwriting is imperfect; garbled words
    are OCR artifacts, not the source.
    """
    return await anyio.to_thread.run_sync(lambda: _fetch_impl(document_id, pdf_page, max_chars))


# =============================================================================
# lookup_entity
# =============================================================================

def _lookup_entity_impl(name: str, limit: int) -> Dict[str, Any]:
    name = (name or "").strip()
    if not name:
        return {"error": "Name is required."}
    limit = max(1, min(limit, 25))

    conn = get_conn()
    try:
        like = f"%{name}%"
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id, e.canonical_name, e.entity_type, e.description,
                       COALESCE(array_agg(DISTINCT a.alias)
                                FILTER (WHERE a.alias IS NOT NULL), '{}')
                FROM entities e
                LEFT JOIN entity_aliases a ON a.entity_id = e.id
                WHERE e.id IN (
                    SELECT DISTINCT e2.id
                    FROM entities e2
                    LEFT JOIN entity_aliases a2 ON a2.entity_id = e2.id
                    WHERE e2.canonical_name ILIKE %s OR a2.alias ILIKE %s
                )
                GROUP BY e.id
                ORDER BY (e.canonical_name ILIKE %s) DESC, e.canonical_name ASC
                LIMIT %s
                """,
                (like, like, name, limit),
            )
            rows = cur.fetchall()

        entries = [
            {
                "canonical_name": canonical or "",
                "entity_type": etype,
                "description": desc,
                "aliases": sorted({x for x in (aliases or []) if x and x != canonical}),
            }
            for (_eid, canonical, etype, desc, aliases) in rows
        ]
        if not entries:
            return {
                "query": name,
                "matches": [],
                "note": (
                    "No concordance entry. The person may still appear in documents — "
                    "try `search`, including mode='fuzzy' for OCR variants."
                ),
            }
        return {"query": name, "matches": entries}
    finally:
        conn.close()


@mcp.tool()
async def lookup_entity(name: str, limit: int = 10) -> Dict[str, Any]:
    """Resolve a person, organization, or codename against Friday's concordance
    of entities and aliases (cover names).

    Call this before concluding someone is absent from the archives, and whenever
    a codename appears in a document: Soviet intelligence referred to people
    almost exclusively by cover names (e.g. ALES, GOOD GIRL, ANTENNA), and the
    same person often has several. Returns canonical names with all known
    aliases — then `search` those aliases to find every mention.
    """
    return await anyio.to_thread.run_sync(lambda: _lookup_entity_impl(name, limit))


# =============================================================================
# list_collections
# =============================================================================

def _list_collections_impl() -> Dict[str, Any]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.slug, c.title, c.description, COUNT(d.id) AS document_count
                FROM collections c
                LEFT JOIN documents d ON d.collection_id = c.id
                GROUP BY c.id
                ORDER BY c.title
                """
            )
            rows = cur.fetchall()
        return {
            "collections": [
                {"slug": slug, "title": title, "description": desc, "document_count": n}
                for (slug, title, desc, n) in rows
            ]
        }
    finally:
        conn.close()


@mcp.tool()
async def list_collections() -> Dict[str, Any]:
    """List the archive's collections (slug, title, description, document count).

    Use the slugs to scope `search` to one collection — e.g. only the Venona
    decrypts or only the Vassiliev notebooks.
    """
    return await anyio.to_thread.run_sync(_list_collections_impl)
