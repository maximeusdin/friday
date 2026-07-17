"""
Search tab executor: run FTS query, enumerate page hits, materialize.

Uses tsv_simple (non-stemming) for Search. Joins chunk_pages without span_order=1
to capture all pages for matched chunks. Picks deterministic representative chunk
per page. Computes snippets in Python.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values

from retrieval.agent.v9_session import session_scope_to_filter
from retrieval.agent.v9_types import ScopeFilter
from retrieval.ops import SearchFilters, _build_where, concordance_expand_terms, _extract_query_terms
from retrieval.search_expansion import pem_expand_surfaces_for_query, pem_enumerate_pages_for_entity
from retrieval.agent.v10_page_bridge import chunks_for_pages, has_page_entity_mentions, pages_to_chunks_map
from retrieval.agent.v10_normalize import normalize_surface_for_lookup
from retrieval.agent.v10_spans import resolve_surface_to_entity_ids
from retrieval.agent.v10_types import ALIAS_SCOPED_COLLECTIONS
from retrieval.search_query import (
    ExcludeOrGroupPrimitive,
    compile_search_primitives_to_tsquery,
    parse_search_query,
    SearchQueryParseError,
)
from retrieval.primitives import (
    ExcludePhrasePrimitive,
    ExcludeTermPrimitive,
    FilterCollectionPrimitive,
    FilterDocumentPrimitive,
    OrGroupPrimitive,
    PhrasePrimitive,
    TermPrimitive,
)

logger = logging.getLogger(__name__)

# Fuzzy mode: cap results, min query length for full-archive
FUZZY_MAX_RESULTS = 2000

# Snippet prefetch: compute snippets for first N hits only; rest fetched on demand
PREFETCH_SNIPPETS = 100


def _scope_is_pem_only(filters: SearchFilters) -> bool:
    """
    True iff scope is explicitly limited to venona and/or vassiliev.
    PEM is only available and relevant for those collections.
    When scope is full_archive or includes other collections, use lexical/fuzzy only.
    """
    colls = filters.collection_slugs
    if not colls:
        return False
    return all(c in ALIAS_SCOPED_COLLECTIONS for c in colls)
FUZZY_MIN_QUERY_LEN_FULL_ARCHIVE = 4
FUZZY_SIMILARITY_THRESHOLD = 0.3


def _run_fuzzy_page_hits_query(
    conn,
    query_raw: str,
    filters: SearchFilters,
    max_results: int = FUZZY_MAX_RESULTS,
) -> List[Tuple[int, int, int, int, int, int, str]]:
    """
    Run trigram fuzzy search, return (collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text).
    Uses word_similarity on COALESCE(clean_text, text). Always non-exhaustive.
    """
    terms = _extract_query_terms(query_raw)
    if not terms:
        return []

    params: Dict[str, Any] = {"max_results": max_results, "threshold": FUZZY_SIMILARITY_THRESHOLD}
    scope_where = _build_where(filters, params)

    similarity_conditions = []
    for i, term in enumerate(terms):
        param_key = f"term_{i}"
        params[param_key] = term
        similarity_conditions.append(
            f"word_similarity(%({param_key})s, COALESCE(c.clean_text, c.text))"
        )
    max_similarity_expr = "GREATEST(" + ", ".join(similarity_conditions) + ")"

    sql = f"""
    WITH matched_chunks AS (
      SELECT c.id, c.text, {max_similarity_expr} AS rank
      FROM chunks c
      JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = c.pipeline_version
      WHERE {scope_where}
        AND {max_similarity_expr} >= %(threshold)s
    ),
    chunk_pages_flat AS (
      SELECT mc.id AS chunk_id, mc.text, mc.rank,
             p.id AS page_id, p.document_id, p.page_seq,
             COALESCE(p.pdf_page_number, p.page_seq) AS pdf_page_number,
             d.collection_id
      FROM matched_chunks mc
      JOIN chunk_pages cp ON cp.chunk_id = mc.id
      JOIN pages p ON p.id = cp.page_id
      JOIN documents d ON d.id = p.document_id
    ),
    ranked_per_page AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY collection_id, document_id, page_seq
        ORDER BY rank DESC NULLS LAST, page_id ASC, chunk_id ASC
      ) AS rn
      FROM chunk_pages_flat
    ),
    dedup_per_pdf_page AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY collection_id, document_id, COALESCE(pdf_page_number, page_seq)
        ORDER BY rank DESC NULLS LAST, page_seq ASC, chunk_id ASC
      ) AS rn2
      FROM ranked_per_page WHERE rn = 1
    ),
    -- Collapse a multi-page chunk to one hit at its first page (see _run_page_hits_query).
    dedup_per_chunk AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY chunk_id ORDER BY page_seq ASC, page_id ASC
      ) AS rn3
      FROM dedup_per_pdf_page WHERE rn2 = 1
    ),
    limited AS (
      SELECT collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text
      FROM dedup_per_chunk WHERE rn3 = 1
      ORDER BY collection_id, document_id, page_seq
      LIMIT %(max_results)s
    )
    SELECT collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text
    FROM limited
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _expand_term(conn, value: str, filters: SearchFilters) -> List[str]:
    """Expand a single term/phrase. PEM only when scope is venona/vassiliev; else literal."""
    if not _scope_is_pem_only(filters):
        return [value]
    if has_page_entity_mentions(conn):
        colls = filters.collection_slugs or list(ALIAS_SCOPED_COLLECTIONS)
        return pem_expand_surfaces_for_query(
            conn, value,
            collection_slugs=colls,
            document_ids=filters.document_ids,
            max_surfaces=15,
        )
    return concordance_expand_terms(conn, value, max_aliases_out=15)


def _expand_primitives_with_aliases(
    conn,
    primitives: List[Any],
    filters: SearchFilters,
) -> Tuple[List[Any], Dict[str, List[str]], str]:
    """
    Expand Term/Phrase primitives via PEM-attested or concordance aliases.
    Returns (expanded_primitives, expanded_terms_json, query_display).
    """
    expanded_terms: Dict[str, List[str]] = {}
    expanded: List[Any] = []
    display_parts: List[str] = []

    for p in primitives:
        if isinstance(p, (FilterCollectionPrimitive, FilterDocumentPrimitive)):
            expanded.append(p)
            continue

        if isinstance(p, TermPrimitive):
            terms = _expand_term(conn, p.value, filters)
            if terms:
                expanded_terms[p.value] = terms
                if len(terms) > 1:
                    expanded.append(OrGroupPrimitive(primitives=[TermPrimitive(value=t) for t in terms]))
                    display_parts.append(f"({terms[0]} OR ...)")
                else:
                    expanded.append(p)
                    display_parts.append(terms[0])
            else:
                expanded.append(p)
                display_parts.append(p.value)

        elif isinstance(p, PhrasePrimitive):
            terms = _expand_term(conn, p.value, filters)
            if terms:
                expanded_terms[p.value] = terms
                if len(terms) > 1:
                    expanded.append(OrGroupPrimitive(primitives=[PhrasePrimitive(value=t) for t in terms]))
                    display_parts.append(f'("{terms[0]}" OR ...)')
                else:
                    expanded.append(p)
                    display_parts.append(f'"{terms[0]}"')
            else:
                expanded.append(p)
                display_parts.append(f'"{p.value}"')

        elif isinstance(p, ExcludeTermPrimitive):
            terms = _expand_term(conn, p.value, filters)
            if terms:
                expanded_terms[p.value] = terms
                expanded.append(ExcludeOrGroupPrimitive(primitives=[TermPrimitive(value=t) for t in terms]))
                display_parts.append(f"!({terms[0]} OR ...)")
            else:
                expanded.append(p)
                display_parts.append(f"!{p.value}")

        elif isinstance(p, ExcludePhrasePrimitive):
            terms = _expand_term(conn, p.value, filters)
            if terms:
                expanded_terms[p.value] = terms
                expanded.append(ExcludeOrGroupPrimitive(primitives=[PhrasePrimitive(value=t) for t in terms]))
                display_parts.append(f'!("{terms[0]}" OR ...)')
            else:
                expanded.append(p)
                display_parts.append(f'!"{p.value}"')

        elif isinstance(p, OrGroupPrimitive):
            or_expanded: List[Any] = []
            for sub in p.primitives:
                if isinstance(sub, TermPrimitive):
                    terms = _expand_term(conn, sub.value, filters)
                    if terms:
                        expanded_terms[sub.value] = terms
                        for t in terms:
                            or_expanded.append(TermPrimitive(value=t))
                    else:
                        or_expanded.append(sub)
                elif isinstance(sub, PhrasePrimitive):
                    terms = _expand_term(conn, sub.value, filters)
                    if terms:
                        expanded_terms[sub.value] = terms
                        for t in terms:
                            or_expanded.append(PhrasePrimitive(value=t))
                    else:
                        or_expanded.append(sub)
                else:
                    or_expanded.append(sub)
            expanded.append(OrGroupPrimitive(primitives=or_expanded))
            display_parts.append("(OR ...)")
        else:
            expanded.append(p)
            display_parts.append("...")

    query_display = " AND ".join(display_parts) if display_parts else ""
    return expanded, expanded_terms, query_display


def _get_collections_to_search(conn, filters: SearchFilters) -> List[Tuple[int, str, str]]:
    """Return (id, slug, title) for each collection in scope. For streaming progress."""
    with conn.cursor() as cur:
        if filters.document_ids:
            cur.execute(
                """
                SELECT DISTINCT d.collection_id, c.slug, c.title
                FROM documents d
                JOIN collections c ON c.id = d.collection_id
                WHERE d.id = ANY(%s)
                ORDER BY d.collection_id
                """,
                (filters.document_ids,),
            )
            return cur.fetchall()
        if filters.collection_slugs:
            cur.execute(
                "SELECT id, slug, title FROM collections WHERE slug = ANY(%s) ORDER BY id",
                (filters.collection_slugs,),
            )
        else:
            cur.execute("SELECT id, slug, title FROM collections ORDER BY id")
        return cur.fetchall()


def _scope_and_filters_to_search_filters(
    conn,
    scope_json: Dict[str, Any],
    filter_primitives: List[Any],
) -> SearchFilters:
    """Build SearchFilters from scope_json + Filter primitives from query."""
    scope_filter = session_scope_to_filter(conn, scope_json)

    collections = list(scope_filter.collections) if scope_filter.collections else None
    document_ids = list(scope_filter.document_ids) if scope_filter.document_ids else None

    for p in filter_primitives:
        if isinstance(p, FilterCollectionPrimitive):
            if collections is None:
                collections = []
            collections.append(p.slug)
        elif isinstance(p, FilterDocumentPrimitive):
            if document_ids is None:
                document_ids = []
            document_ids.append(p.document_id)

    return SearchFilters(
        collection_slugs=collections,
        document_ids=document_ids,
        date_from=scope_filter.date_from,
        date_to=scope_filter.date_to,
    )


def _run_page_hits_query(
    conn,
    tsquery_sql: str,
    tsquery_params: List[Any],
    filters: SearchFilters,
    tsv_col: str = "tsv_simple",
) -> List[Tuple[int, int, int, int, int, int, str]]:
    """
    Run FTS query, return (collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text).
    """
    import os
    params: Dict[str, Any] = {"cem": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")}
    scope_where = _build_where(filters, params)

    # Convert tsquery %s placeholders to named params (used twice: @@ and ts_rank_cd)
    import re
    named_params: Dict[str, Any] = dict(params)
    tsq_idx = [0]

    def replacer(_m):
        key = f"tsq{tsq_idx[0]}"
        tsq_idx[0] += 1
        return f"%({key})s"

    tsquery_sql_named = re.sub(r"%s", replacer, tsquery_sql)
    n_tsq = tsq_idx[0]
    for i, v in enumerate(tsquery_params):
        named_params[f"tsq{i}"] = v
    # Second occurrence (for ts_rank_cd) - use tsq0b, tsq1b, ... with same values
    tsquery_sql_named2 = re.sub(r"%\(tsq(\d+)\)s", lambda m: f"%(tsq{m.group(1)}b)s", tsquery_sql_named)
    for i, v in enumerate(tsquery_params):
        named_params[f"tsq{i}b"] = v

    sql = f"""
    WITH matched_chunks AS (
      SELECT c.id, c.text,
             GREATEST(
               ts_rank_cd(c.{tsv_col}, ({tsquery_sql_named2})),
               COALESCE(ts_rank_cd(cec.text_canonical_tsv, ({tsquery_sql_named2})), 0)
             ) AS rank
      FROM chunks c
      JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = c.pipeline_version
      LEFT JOIN chunk_embeddings_canonical cec
        ON cec.chunk_id = c.id
        AND cec.pipeline_version = c.pipeline_version
        AND cec.embedding_model = %(cem)s
      WHERE (c.{tsv_col} @@ ({tsquery_sql_named})
             OR cec.text_canonical_tsv @@ ({tsquery_sql_named}))
        AND {scope_where}
    ),
    chunk_pages_flat AS (
      SELECT mc.id AS chunk_id, mc.text, mc.rank,
             p.id AS page_id, p.document_id, p.page_seq,
             COALESCE(p.pdf_page_number, p.page_seq) AS pdf_page_number,
             d.collection_id
      FROM matched_chunks mc
      JOIN chunk_pages cp ON cp.chunk_id = mc.id
      JOIN pages p ON p.id = cp.page_id
      JOIN documents d ON d.id = p.document_id
    ),
    ranked_per_page AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY collection_id, document_id, page_seq
        ORDER BY rank DESC NULLS LAST, page_id ASC, chunk_id ASC
      ) AS rn
      FROM chunk_pages_flat
    ),
    dedup_per_pdf_page AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY collection_id, document_id, COALESCE(pdf_page_number, page_seq)
        ORDER BY rank DESC NULLS LAST, page_seq ASC, chunk_id ASC
      ) AS rn2
      FROM ranked_per_page WHERE rn = 1
    ),
    -- A single chunk can span several PDF pages (e.g. Silvermaster chunks average ~6 pages),
    -- which otherwise yields one identical-snippet hit per page. Collapse to one hit per
    -- matching chunk, shown at its first page.
    dedup_per_chunk AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY chunk_id
        ORDER BY page_seq ASC, page_id ASC
      ) AS rn3
      FROM dedup_per_pdf_page WHERE rn2 = 1
    )
    SELECT collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text
    FROM dedup_per_chunk WHERE rn3 = 1
    ORDER BY collection_id, document_id, page_seq
    """
    with conn.cursor() as cur:
        cur.execute(sql, named_params)
        return cur.fetchall()


def _try_pem_enumeration_path(
    conn,
    query_raw: str,
    filters: SearchFilters,
) -> Optional[List[Tuple[int, int, int, int, int, int, str]]]:
    """
    Pattern 3A: when query is entity-intentful, enumerate pages from PEM.
    PEM is only available for venona/vassiliev; skip when scope includes other collections.
    Returns (collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text)
    or None to fall through to FTS.
    """
    if not _scope_is_pem_only(filters):
        return None
    if not os.environ.get("SEARCH_PEM_ENUMERATE_ENTITIES", "1").strip().lower() in ("1", "true", "yes"):
        return None
    if not has_page_entity_mentions(conn):
        return None

    q = (query_raw or "").strip()
    tokens = q.split()
    if len(tokens) < 2 or len(tokens) > 5:
        return None

    norm_key = normalize_surface_for_lookup(q)
    if not norm_key:
        return None

    colls = filters.collection_slugs or list(ALIAS_SCOPED_COLLECTIONS)
    if not colls:
        return None

    entity_ids = resolve_surface_to_entity_ids(conn, norm_key, scope_collections=colls, max_entities=1)
    if not entity_ids:
        return None

    entity_id = entity_ids[0]
    page_rows = pem_enumerate_pages_for_entity(conn, entity_id, filters, max_pages=500)
    if not page_rows:
        return None

    # Build (collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text)
    result: List[Tuple[int, int, int, int, int, int, str]] = []
    page_ids = [r[2] for r in page_rows]
    p2c = pages_to_chunks_map(conn, page_ids, max_chunks_per_page=1, prefer_entity_id=entity_id)
    all_chunk_ids = [cids[0] for cids in p2c.values() if cids]
    if not all_chunk_ids:
        return None

    with conn.cursor() as cur:
        cur.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (all_chunk_ids,))
        chunk_texts = {r[0]: (r[1] or "") for r in cur.fetchall()}

    for coll_id, doc_id, page_id, page_seq, pdf_page in page_rows:
        cids = p2c.get(page_id, [])
        if not cids:
            continue
        chunk_id = cids[0]
        text = chunk_texts.get(chunk_id, "")
        result.append((coll_id, doc_id, page_id, page_seq, pdf_page, chunk_id, text))

    return result if result else None


def _lexical_primitives_from_query(
    conn,
    query_raw: str,
    scope_json: Dict[str, Any],
    alias_expand: bool,
) -> Tuple[List[Any], List[Any], Optional[Dict[str, Any]], Optional[str], List[Any], SearchFilters]:
    """
    Get text primitives for lexical search. Tries parse; on failure uses term extraction.
    Returns (text_primitives, scope_primitives, expanded_terms_json, query_display, primitives_for_ast, filters).
    """
    try:
        primitives = parse_search_query(query_raw)
    except SearchQueryParseError:
        terms = _extract_query_terms(query_raw)
        if not terms:
            filters = _scope_and_filters_to_search_filters(conn, scope_json, [])
            return [], [], None, None, [], filters
        fallback = [OrGroupPrimitive(primitives=[TermPrimitive(value=t) for t in terms])]
        scope_primitives: List[Any] = []
        filters = _scope_and_filters_to_search_filters(conn, scope_json, [])
        return fallback, scope_primitives, None, None, fallback, filters

    scope_primitives = [p for p in primitives if isinstance(p, (FilterCollectionPrimitive, FilterDocumentPrimitive))]
    text_primitives = [p for p in primitives if p not in scope_primitives]
    filters = _scope_and_filters_to_search_filters(conn, scope_json, scope_primitives)
    expanded_terms_json: Optional[Dict[str, Any]] = None
    query_display: Optional[str] = None
    if alias_expand:
        expanded_all, expanded_terms_json, query_display = _expand_primitives_with_aliases(conn, primitives, filters)
        scope_primitives = [p for p in expanded_all if isinstance(p, (FilterCollectionPrimitive, FilterDocumentPrimitive))]
        text_primitives = [p for p in expanded_all if p not in scope_primitives]
    return text_primitives, scope_primitives, expanded_terms_json, query_display, primitives, filters


def _compute_snippet(conn, chunk_id: int, chunk_text: str, phrases: List[str], max_chars: int = 200) -> str:
    """Extract snippet around first match. Uses simple substring if match_trace unavailable."""
    try:
        from retrieval.match_trace import get_phrase_positions_on_demand
        matches = get_phrase_positions_on_demand(
            chunk_id, phrases, conn, case_sensitive=False, context_chars=max_chars // 2, text=chunk_text
        )
        if matches:
            return matches[0].get("context", chunk_text[:max_chars]) or chunk_text[:max_chars]
    except Exception:
        pass
    return (chunk_text[:max_chars] + "…") if len(chunk_text) > max_chars else chunk_text


def run_search(
    conn,
    result_set_id: str,
    query_raw: str,
    scope_json: Dict[str, Any],
    *,
    alias_expand: bool = True,
    mode: str = "exact",
    exact_only: bool = False,
    on_progress: Optional[Callable[[str, int, int, Optional[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """
    Execute search, materialize page hits, update search_result_sets.

    When mode=fuzzy and exact_only=True: run exact/lexical phase only, return status=exact_complete.
    Call run_search_expand_fuzzy to add fuzzy matches.

    on_progress(phase, current, total, extra=None): optional callback.
    phase in ("collection", "snippets"); for "collection" current/total = collections done,
    extra={"slug","title","hits"}; for "snippets" current/total = hits processed.

    Returns dict with total_hits, coverage_json, status.
    """
    if mode == "fuzzy":
        # Guardrails: full_archive + short query → require min length
        scope_mode = scope_json.get("mode", "full_archive")
        if scope_mode == "full_archive":
            q_clean = (query_raw or "").strip()
            if len(q_clean) < FUZZY_MIN_QUERY_LEN_FULL_ARCHIVE:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE search_result_sets
                        SET status = 'error', error_message = %s
                        WHERE id = %s
                        """,
                        (f"Fuzzy search on full archive requires at least {FUZZY_MIN_QUERY_LEN_FULL_ARCHIVE} characters. Narrow scope or use exact search.", result_set_id),
                    )
                conn.commit()
                return {"status": "error", "error": "Query too short for fuzzy full-archive search"}

        # Lexical phase: parse or fallback to term extraction
        text_primitives, scope_primitives, expanded_terms_json, query_display, primitives, filters = _lexical_primitives_from_query(
            conn, query_raw, scope_json, alias_expand
        )
    else:
        try:
            primitives = parse_search_query(query_raw)
        except SearchQueryParseError as e:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE search_result_sets
                    SET status = 'error', error_message = %s
                    WHERE id = %s
                    """,
                    (str(e), result_set_id),
                )
            conn.commit()
            return {"status": "error", "error": str(e)}

        scope_primitives = [p for p in primitives if isinstance(p, (FilterCollectionPrimitive, FilterDocumentPrimitive))]
        text_primitives = [p for p in primitives if p not in scope_primitives]
        filters = _scope_and_filters_to_search_filters(conn, scope_json, scope_primitives)

        expanded_terms_json: Optional[Dict[str, Any]] = None
        query_display: Optional[str] = None
        if alias_expand:
            expanded_all, expanded_terms_json, query_display = _expand_primitives_with_aliases(conn, primitives, filters)
            scope_primitives = [p for p in expanded_all if isinstance(p, (FilterCollectionPrimitive, FilterDocumentPrimitive))]
            text_primitives = [p for p in expanded_all if p not in scope_primitives]
    is_exhaustive = mode == "exact"

    # Extract phrases for snippet
    from retrieval.primitives import PhrasePrimitive, TermPrimitive
    phrases: List[str] = []
    if mode == "fuzzy":
        phrases = _extract_query_terms(query_raw)
    else:
        for p in text_primitives:
            if isinstance(p, PhrasePrimitive):
                phrases.append(p.value)
            elif isinstance(p, TermPrimitive):
                phrases.append(p.value)

    is_exhaustive = mode == "exact"

    # Get collections to search (for streaming progress)
    collections_to_search = _get_collections_to_search(conn, filters)
    if not collections_to_search:
        rows = []
    else:
        tsquery_sql, _, tsquery_params = compile_search_primitives_to_tsquery(text_primitives)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'tsv_simple'
            """)
            tsv_col = "tsv_simple" if cur.fetchone() else "tsv"

        all_rows: List[Tuple[int, int, int, int, int, int, str]] = []
        hit_rank = 0
        total_cols = len(collections_to_search)

        for col_idx, (coll_id, slug, title) in enumerate(collections_to_search):
            col_filters = SearchFilters(
                collection_slugs=[slug],
                document_ids=filters.document_ids,
                date_from=filters.date_from,
                date_to=filters.date_to,
            )
            # Exact phase: lexical or PEM
            pem_rows = _try_pem_enumeration_path(conn, query_raw, col_filters)
            if pem_rows is not None:
                col_rows = pem_rows
            else:
                col_rows = _run_page_hits_query(conn, tsquery_sql, tsquery_params, col_filters, tsv_col=tsv_col)

            if on_progress:
                on_progress("collection", col_idx + 1, total_cols, {"slug": slug, "title": title or slug, "hits": len(col_rows)})

            insert_rows = []
            for r in col_rows:
                hit_rank += 1
                collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text = r
                if hit_rank <= PREFETCH_SNIPPETS:
                    snippet = _compute_snippet(conn, chunk_id, text or "", phrases)
                else:
                    snippet = ""  # Fetched on demand via fetch_more_snippets
                insert_rows.append((result_set_id, collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, snippet, hit_rank))

            all_rows.extend(col_rows)
            if insert_rows:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO search_result_page_hits
                        (result_set_id, collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, snippet, hit_rank)
                        VALUES %s
                        ON CONFLICT (result_set_id, collection_id, document_id, page_id) DO NOTHING
                        """,
                        insert_rows,
                    )

        rows = all_rows

        # Fuzzy phase: when mode=fuzzy and not exact_only, add trigram matches (dedup via ON CONFLICT)
        if mode == "fuzzy" and exact_only:
            collection_hits_from_fuzzy = None
            total_from_fuzzy = None
        elif mode == "fuzzy" and not exact_only:
            for col_idx, (coll_id, slug, title) in enumerate(collections_to_search):
                col_filters = SearchFilters(
                    collection_slugs=[slug],
                    document_ids=filters.document_ids,
                    date_from=filters.date_from,
                    date_to=filters.date_to,
                )
                fuzzy_rows = _run_fuzzy_page_hits_query(conn, query_raw, col_filters)
                if not fuzzy_rows:
                    continue
                insert_rows = []
                for r in fuzzy_rows:
                    hit_rank += 1
                    collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text = r
                    if hit_rank <= PREFETCH_SNIPPETS:
                        snippet = _compute_snippet(conn, chunk_id, text or "", phrases)
                    else:
                        snippet = ""
                    insert_rows.append((result_set_id, collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, snippet, hit_rank))
                if insert_rows:
                    with conn.cursor() as cur:
                        execute_values(
                            cur,
                            """
                            INSERT INTO search_result_page_hits
                            (result_set_id, collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, snippet, hit_rank)
                            VALUES %s
                            ON CONFLICT (result_set_id, collection_id, document_id, page_id) DO NOTHING
                            """,
                            insert_rows,
                        )
                if on_progress:
                    on_progress("collection", total_cols + col_idx + 1, total_cols * 2, {"slug": slug, "title": title or slug, "hits": len(fuzzy_rows), "phase": "fuzzy"})
            # Recompute total and collection_hits from table after fuzzy merge
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT collection_id, COUNT(*) FROM search_result_page_hits WHERE result_set_id = %s GROUP BY collection_id",
                    (result_set_id,),
                )
                collection_hits_from_fuzzy = {r[0]: r[1] for r in cur.fetchall()}
                cur.execute("SELECT COUNT(*) FROM search_result_page_hits WHERE result_set_id = %s", (result_set_id,))
                total_from_fuzzy = cur.fetchone()[0]
        else:
            collection_hits_from_fuzzy = None
            total_from_fuzzy = None

    with conn.cursor() as cur:
        cur.execute("SELECT id, slug, title FROM collections")
        collections = {r[0]: {"id": r[0], "slug": r[1], "title": r[2]} for r in cur.fetchall()}

    if collection_hits_from_fuzzy is not None:
        collection_hits = collection_hits_from_fuzzy
        total_hits_val = total_from_fuzzy
    else:
        collection_hits = {}
        for r in rows:
            cid = r[0]
            collection_hits[cid] = collection_hits.get(cid, 0) + 1
        total_hits_val = len(rows)

    coverage = {
        "collections": [
            {"id": cid, "slug": collections.get(cid, {}).get("slug", ""), "title": collections.get(cid, {}).get("title", ""), "hits": cnt}
            for cid, cnt in sorted(collection_hits.items())
        ],
        "total_hits": total_hits_val,
        "collections_searched": len(collection_hits),
        "collections_total": len(collections),
        "missing_collections": [],
        "phrases": phrases,  # for fetch_more_snippets
    }

    status = "exact_complete" if (mode == "fuzzy" and exact_only) else "complete"

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE search_result_sets
            SET total_hits = %s, coverage_json = %s, status = %s, query_ast_json = %s,
                expanded_terms_json = %s, query_display = %s, is_exhaustive = %s
            WHERE id = %s
            """,
            (
                total_hits_val,
                json.dumps(coverage),
                status,
                json.dumps([{"type": type(p).__name__} for p in primitives]),
                json.dumps(expanded_terms_json) if expanded_terms_json else None,
                query_display,
                is_exhaustive,
                result_set_id,
            ),
        )

    conn.commit()
    return {"total_hits": total_hits_val, "coverage_json": coverage, "status": status}


def run_search_expand_fuzzy(
    conn,
    result_set_id: str,
    *,
    on_progress: Optional[Callable[[str, int, int, Optional[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """
    Expand a result set (status=exact_complete) with fuzzy/trigram matches.
    Reads query, scope from search_result_sets. Deduplicates via ON CONFLICT.
    Returns dict with total_hits, status='complete'.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT query_raw, scope_json, alias_expand
            FROM search_result_sets
            WHERE id = %s AND status = 'exact_complete'
            """,
            (result_set_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"status": "error", "error": "Result set not found or not in exact_complete state"}
        query_raw, scope_json_raw, alias_expand = row[0], row[1], row[2]
    scope_json = scope_json_raw if isinstance(scope_json_raw, dict) else (json.loads(scope_json_raw) if scope_json_raw else {"mode": "full_archive"})

    # Guardrails
    scope_mode = scope_json.get("mode", "full_archive")
    if scope_mode == "full_archive":
        q_clean = (query_raw or "").strip()
        if len(q_clean) < FUZZY_MIN_QUERY_LEN_FULL_ARCHIVE:
            return {"status": "error", "error": "Query too short for fuzzy expansion"}

    text_primitives, _, _, _, _, filters = _lexical_primitives_from_query(conn, query_raw, scope_json, alias_expand or True)
    phrases = _extract_query_terms(query_raw)
    collections_to_search = _get_collections_to_search(conn, filters)
    if not collections_to_search:
        return {"total_hits": 0, "status": "complete"}

    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(hit_rank), 0) FROM search_result_page_hits WHERE result_set_id = %s", (result_set_id,))
        hit_rank = cur.fetchone()[0] or 0

    total_cols = len(collections_to_search)
    for col_idx, (coll_id, slug, title) in enumerate(collections_to_search):
        col_filters = SearchFilters(
            collection_slugs=[slug],
            document_ids=filters.document_ids,
            date_from=filters.date_from,
            date_to=filters.date_to,
        )
        fuzzy_rows = _run_fuzzy_page_hits_query(conn, query_raw, col_filters)
        if not fuzzy_rows:
            continue
        insert_rows = []
        for r in fuzzy_rows:
            hit_rank += 1
            collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, text = r
            if hit_rank <= PREFETCH_SNIPPETS:
                snippet = _compute_snippet(conn, chunk_id, text or "", phrases)
            else:
                snippet = ""
            insert_rows.append((result_set_id, collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, snippet, hit_rank))
        if insert_rows:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO search_result_page_hits
                    (result_set_id, collection_id, document_id, page_id, page_seq, pdf_page_number, chunk_id, snippet, hit_rank)
                    VALUES %s
                    ON CONFLICT (result_set_id, collection_id, document_id, page_id) DO NOTHING
                    """,
                    insert_rows,
                )
        if on_progress:
            on_progress("collection", col_idx + 1, total_cols, {"slug": slug, "title": title or slug, "hits": len(fuzzy_rows), "phase": "fuzzy"})

    with conn.cursor() as cur:
        cur.execute(
            "SELECT collection_id, COUNT(*) FROM search_result_page_hits WHERE result_set_id = %s GROUP BY collection_id",
            (result_set_id,),
        )
        collection_hits = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute("SELECT COUNT(*) FROM search_result_page_hits WHERE result_set_id = %s", (result_set_id,))
        total_hits_val = cur.fetchone()[0]
        cur.execute("SELECT id, slug, title FROM collections")
        collections = {r[0]: {"id": r[0], "slug": r[1], "title": r[2]} for r in cur.fetchall()}

    coverage = {
        "collections": [
            {"id": cid, "slug": collections.get(cid, {}).get("slug", ""), "title": collections.get(cid, {}).get("title", ""), "hits": cnt}
            for cid, cnt in sorted(collection_hits.items())
        ],
        "total_hits": total_hits_val,
        "collections_searched": len(collection_hits),
        "collections_total": len(collections),
        "missing_collections": [],
        "phrases": phrases,
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE search_result_sets
            SET total_hits = %s, coverage_json = %s, status = 'complete'
            WHERE id = %s
            """,
            (total_hits_val, json.dumps(coverage), result_set_id),
        )
    conn.commit()
    return {"total_hits": total_hits_val, "coverage_json": coverage, "status": "complete"}


def fetch_more_snippets(
    conn,
    result_set_id: str,
    *,
    batch_size: int = 100,
) -> int:
    """
    Compute snippets for the next batch of hits that have empty snippets.
    Returns the number of snippets computed.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT srs.coverage_json
            FROM search_result_sets srs
            WHERE srs.id = %s AND srs.status IN ('complete', 'exact_complete')
            """,
            (result_set_id,),
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return 0
        raw = row[0]
        if isinstance(raw, dict):
            coverage = raw
        elif isinstance(raw, str):
            coverage = json.loads(raw) if raw else {}
        else:
            coverage = {}
    phrases = coverage.get("phrases") or []
    if not phrases:
        # Fallback: extract from query_raw
        with conn.cursor() as cur:
            cur.execute(
                "SELECT query_raw, mode FROM search_result_sets WHERE id = %s",
                (result_set_id,),
            )
            r = cur.fetchone()
            if r:
                query_raw, mode = r[0], r[1] or "exact"
                phrases = _extract_query_terms(query_raw)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT h.collection_id, h.document_id, h.page_id, h.chunk_id,
                   COALESCE(c.clean_text, c.text) AS text
            FROM search_result_page_hits h
            JOIN chunks c ON c.id = h.chunk_id
            WHERE h.result_set_id = %s
              AND (h.snippet IS NULL OR h.snippet = '')
            ORDER BY h.hit_rank
            LIMIT %s
            """,
            (result_set_id, batch_size),
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    # Compute all snippets in memory (no DB; we have text from the SELECT)
    update_tuples = []
    for collection_id, document_id, page_id, chunk_id, text in rows:
        snippet = _compute_snippet(conn, chunk_id, text or "", phrases)
        update_tuples.append((result_set_id, collection_id, document_id, page_id, snippet))

    # Single batched UPDATE
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            UPDATE search_result_page_hits h
            SET snippet = v.snippet
            FROM (VALUES %s) AS v(result_set_id, collection_id, document_id, page_id, snippet)
            WHERE h.result_set_id = v.result_set_id
              AND h.collection_id = v.collection_id
              AND h.document_id = v.document_id
              AND h.page_id = v.page_id
            """,
            update_tuples,
            template="(%s::uuid, %s, %s, %s, %s)",
        )
        updated = cur.rowcount
    conn.commit()
    return updated
