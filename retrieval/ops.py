from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import Json
from retrieval.fuzzy_lex import (
    FuzzyLexConfig,
    compile_hybrid_or_tsquery,
)

# -----------------------
# Concordance query expansion
# -----------------------

def _normalize_q(s: str) -> str:
    s = (s or "").strip()
    # light normalization only; keep deterministic
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_key(s: str) -> str:
    """
    Deterministic normalization for concordance matching.
    Goal: make "Army Air Force, U.S." match stored variants like "Army Air Force, U".
    """
    s = (s or "").lower().strip()
    s = s.replace("’", "'")

    # remove periods (U.S. -> US) and normalize punctuation to spaces
    s = s.replace(".", "")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # conservative fix: PDF sometimes truncates trailing "U.S." to "U"
    # Treat trailing token "u" as "us".
    if s.endswith(" u"):
        s = s[:-2] + " us"

    return s


def concordance_expand_terms(
    conn,
    text: str,
    *,
    source_slug: str = "venona_vassiliev_concordance_v3",
    max_entities_scan: int = 5000,
    max_aliases_out: int = 25,
) -> List[str]:
    """
    Expand a user query into concordance aliases/canonical terms.

    Option B: Python-side normalization:
      1) Pull candidate entities/aliases for a source_slug.
      2) Normalize everything with _norm_key.
      3) Find entity_ids where canonical/alias matches the query key.
      4) Return canonical + all aliases for those entities.

    This is deterministic and easier to debug than SQL regexp normalization.
    """
    q = (text or "").strip()
    if not q:
        return []

    q_key = _norm_key(q)
    if not q_key:
        return []

    # 1) Find the concordance source id
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM concordance_sources WHERE slug = %s LIMIT 1;",
            (source_slug,),
        )
        row = cur.fetchone()
        if not row:
            return []
        source_id = int(row[0])

    # 2) Pull entities + aliases (bounded scan; should be fine at your current scale)
    #    We fetch canonical + alias text with entity_id for matching.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id AS entity_id, e.canonical_name AS term
            FROM entities e
            WHERE e.source_id = %s
            ORDER BY e.id
            LIMIT %s;
            """,
            (source_id, max_entities_scan),
        )
        entity_rows = cur.fetchall()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ea.entity_id, ea.alias AS term
            FROM entity_aliases ea
            WHERE ea.source_id = %s
            ORDER BY ea.entity_id, ea.id;
            """,
            (source_id,),
        )
        alias_rows = cur.fetchall()

    # 3) Determine which entity_ids match the query key
    matched_entity_ids = set()

    def _consider(entity_id: int, term: str):
        if not term:
            return
        if _norm_key(term) == q_key:
            matched_entity_ids.add(entity_id)

    for entity_id, term in entity_rows:
        _consider(int(entity_id), term)

    for entity_id, term in alias_rows:
        _consider(int(entity_id), term)

    if not matched_entity_ids:
        return []

    # 4) Collect expansions (canonical + aliases for matched entities)
    out: List[str] = []
    seen = set()

    def _emit(term: str):
        t = (term or "").strip()
        if not t:
            return
        tlc = t.lower()
        if tlc in seen:
            return
        seen.add(tlc)
        out.append(t)

    for entity_id, term in entity_rows:
        if int(entity_id) in matched_entity_ids:
            _emit(term)

    for entity_id, term in alias_rows:
        if int(entity_id) in matched_entity_ids:
            _emit(term)

    # Prefer shorter terms first (nicer for query-variant generation)
    out.sort(key=lambda s: (len(s), s.lower()))

    return out[:max_aliases_out]


def concordance_expand_terms_fuzzy(
    conn,
    text: str,
    *,
    source_slug: str = "venona_vassiliev_concordance_v3",
    similarity_threshold: float = 0.6,
    max_candidates: int = 5,
    max_aliases_out: int = 25,
    timeout_seconds: float = 30.0,
    verbose: bool = False,
) -> List[str]:
    """
    Fuzzy expansion: find similar concordance entries using pg_trgm similarity.
    
    Used as fallback when exact expansion fails (e.g., user typo or OCR error in query).
    Uses word_similarity to find similar entity/alias names, then returns all aliases
    for those matched entities.
    
    Args:
        conn: Database connection
        text: Query text (may contain typos)
        source_slug: Concordance source slug
        similarity_threshold: Minimum similarity score (0.0-1.0), default 0.6
        max_candidates: Maximum number of similar entities to consider
        max_aliases_out: Maximum number of expansion terms to return
    
    Returns:
        List of expansion terms (canonical names + aliases) from similar entities
    """
    q = (text or "").strip()
    if not q:
        return []
    
    # 1) Find the concordance source id
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM concordance_sources WHERE slug = %s LIMIT 1;",
                (source_slug,),
            )
            row = cur.fetchone()
            if not row or len(row) < 1:
                if verbose:
                    print(f" [No concordance source found for '{source_slug}']")
                return []
            source_id = int(row[0])
    except (IndexError, TypeError, ValueError) as e:
        if verbose:
            print(f" [ERROR finding source: {e}]")
        return []
    except Exception as e:
        if verbose:
            print(f" [ERROR finding source: {e}]")
        try:
            conn.rollback()
        except Exception:
            pass
        return []
    
    # 2) Find similar entities using trigram similarity
    # Optimized: use GIST index with % operator, compute similarity only in ORDER BY
    matched_entity_ids = set()
    
    import time
    start_time = time.time()
    
    try:
        if verbose:
            print(f"[Fuzzy expansion] Searching entities for '{q}'...", end="", flush=True)
        
        with conn.cursor() as cur:
            # Set statement timeout (PostgreSQL-level timeout, in milliseconds)
            # Use SET LOCAL so it only affects this transaction, not the session
            try:
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_seconds * 1000)}")
            except Exception:
                # Fallback if LOCAL not supported (shouldn't happen in PostgreSQL)
                try:
                    cur.execute(f"SET statement_timeout = {int(timeout_seconds * 1000)}")
                except Exception as timeout_error:
                    if verbose:
                        print(f" [ERROR setting timeout: {timeout_error}]")
                    conn.rollback()
                    return []
            
            # Optimized query: compute similarity once, use GIST index with % operator
            # The % operator uses the GIST index more efficiently than word_similarity() in WHERE
            query_start = time.time()
            try:
                # Validate parameters before executing
                if not isinstance(source_id, int) or source_id <= 0:
                    if verbose:
                        print(f" [ERROR: Invalid source_id: {source_id}]")
                    return []
                if not isinstance(max_candidates, int) or max_candidates <= 0:
                    if verbose:
                        print(f" [ERROR: Invalid max_candidates: {max_candidates}]")
                    return []
                if not q or not isinstance(q, str):
                    if verbose:
                        print(f" [ERROR: Invalid query string: {q}]")
                    return []
                
                # Debug: log parameters if verbose
                if verbose:
                    print(f" [DEBUG: source_id={source_id}, q='{q}', max_candidates={max_candidates}]", end="", flush=True)
                
                # Use psycopg2's parameterized query with explicit parameter list
                # The % operator for trigram similarity needs to be escaped as %% in the SQL string
                # but psycopg2 handles %s placeholders, so we need to be careful
                query_sql = """
                    SELECT DISTINCT 
                        e.id AS entity_id,
                        e.canonical_name AS term,
                        GREATEST(
                            word_similarity(%s, e.canonical_name),
                            word_similarity(%s, LOWER(e.canonical_name))
                        ) AS sim_score
                    FROM entities e
                    WHERE e.source_id = %s
                      AND (
                          e.canonical_name %% %s
                          OR LOWER(e.canonical_name) %% LOWER(%s)
                      )
                    ORDER BY GREATEST(
                        word_similarity(%s, e.canonical_name),
                        word_similarity(%s, LOWER(e.canonical_name))
                    ) DESC
                    LIMIT %s
                """
                params = (q, q, source_id, q, q, q, q, max_candidates)
                
                cur.execute(query_sql, params)
            except (IndexError, TypeError, ValueError) as query_error:
                if verbose:
                    import traceback
                    print(f" [QUERY ERROR (type/value): {query_error}]")
                    print(f" [TRACEBACK: {traceback.format_exc()}]")
                try:
                    conn.rollback()
                except Exception:
                    pass
                return []
            except Exception as query_error:
                error_msg = str(query_error)
                if verbose:
                    import traceback
                    print(f" [QUERY ERROR: {error_msg}]")
                    print(f" [TRACEBACK: {traceback.format_exc()}]")
                try:
                    conn.rollback()
                except Exception:
                    pass
                # Check if it's a tuple index error in the error message itself
                if "tuple index out of range" in error_msg.lower():
                    return []
                # For other errors, still return empty for graceful degradation
                return []
            query_time = time.time() - query_start
            
            # Filter by threshold after getting candidates (faster than WHERE clause)
            entity_matches = 0
            try:
                rows = cur.fetchall()
                for row in rows:
                    try:
                        if not row or len(row) < 3:
                            if verbose:
                                print(f" [WARNING: Unexpected row format: {row}]")
                            continue
                        entity_id, term, sim_score = row[0], row[1], row[2]
                        if sim_score and sim_score >= similarity_threshold:
                            matched_entity_ids.add(int(entity_id))
                            entity_matches += 1
                            if len(matched_entity_ids) >= max_candidates:
                                break
                    except (IndexError, TypeError) as row_error:
                        if verbose:
                            print(f" [WARNING: Error processing row {row}: {row_error}]")
                        continue
            except Exception as fetch_error:
                if verbose:
                    print(f" [ERROR fetching results: {fetch_error}]")
                try:
                    conn.rollback()  # Rollback on error
                except Exception:
                    pass
                # Don't raise - return empty list for graceful degradation
                return []
            
            # Reset timeout (SET LOCAL automatically resets at end of transaction, but reset explicitly)
            try:
                cur.execute("RESET statement_timeout")
            except Exception:
                pass
            
            if verbose:
                elapsed = time.time() - start_time
                print(f" found {entity_matches} matches ({query_time:.2f}s, {elapsed:.2f}s total)")
        
        if verbose:
            print(f"[Fuzzy expansion] Searching aliases for '{q}'...", end="", flush=True)
        
        with conn.cursor() as cur:
            try:
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_seconds * 1000)}")
            except Exception:
                try:
                    cur.execute(f"SET statement_timeout = {int(timeout_seconds * 1000)}")
                except Exception as timeout_error:
                    if verbose:
                        print(f" [ERROR setting timeout: {timeout_error}]")
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    return []
            
            # Same optimization for aliases
            query_start = time.time()
            try:
                # Validate parameters before executing
                if not isinstance(source_id, int) or source_id <= 0:
                    if verbose:
                        print(f" [ERROR: Invalid source_id: {source_id}]")
                    return []
                if not isinstance(max_candidates, int) or max_candidates <= 0:
                    if verbose:
                        print(f" [ERROR: Invalid max_candidates: {max_candidates}]")
                    return []
                
                # Escape % operator as %% for psycopg2 (trigram similarity operator)
                # psycopg2 uses %s for parameters, so literal % must be %%
                cur.execute(
                    """
                    SELECT DISTINCT 
                        ea.entity_id,
                        ea.alias AS term,
                        GREATEST(
                            word_similarity(%s, ea.alias),
                            word_similarity(%s, LOWER(ea.alias))
                        ) AS sim_score
                    FROM entity_aliases ea
                    WHERE ea.source_id = %s
                      AND (
                          ea.alias %% %s
                          OR LOWER(ea.alias) %% LOWER(%s)
                      )
                    ORDER BY GREATEST(
                        word_similarity(%s, ea.alias),
                        word_similarity(%s, LOWER(ea.alias))
                    ) DESC
                    LIMIT %s
                    """,
                    (q, q, source_id, q, q, q, q, max_candidates),
                )
            except (IndexError, TypeError, ValueError) as query_error:
                if verbose:
                    print(f" [QUERY ERROR (type/value): {query_error}]")
                try:
                    conn.rollback()
                except Exception:
                    pass
                return []
            except Exception as query_error:
                error_msg = str(query_error)
                if verbose:
                    print(f" [QUERY ERROR: {error_msg}]")
                try:
                    conn.rollback()
                except Exception:
                    pass
                # Check if it's a tuple index error in the error message itself
                if "tuple index out of range" in error_msg.lower():
                    return []
                # For other errors, still return empty for graceful degradation
                return []
            query_time = time.time() - query_start
            
            alias_matches = 0
            try:
                rows = cur.fetchall()
                for row in rows:
                    try:
                        if not row or len(row) < 3:
                            if verbose:
                                print(f" [WARNING: Unexpected row format: {row}]")
                            continue
                        entity_id, term, sim_score = row[0], row[1], row[2]
                        if sim_score and sim_score >= similarity_threshold:
                            matched_entity_ids.add(int(entity_id))
                            alias_matches += 1
                            if len(matched_entity_ids) >= max_candidates:
                                break
                    except (IndexError, TypeError) as row_error:
                        if verbose:
                            print(f" [WARNING: Error processing row {row}: {row_error}]")
                        continue
            except Exception as fetch_error:
                if verbose:
                    print(f" [ERROR fetching results: {fetch_error}]")
                try:
                    conn.rollback()  # Rollback on error
                except Exception:
                    pass
                # Don't raise - return empty list for graceful degradation
                return []
            
            # Reset timeout (SET LOCAL automatically resets at end of transaction, but reset explicitly)
            try:
                cur.execute("RESET statement_timeout")
            except Exception:
                pass
            
            if verbose:
                elapsed = time.time() - start_time
                print(f" found {alias_matches} matches ({query_time:.2f}s, {elapsed:.2f}s total)")
                        
    except Exception as e:
        # Rollback transaction on any error
        try:
            conn.rollback()
        except Exception:
            pass
        
        # If timeout or any error, return empty (graceful degradation)
        if verbose:
            elapsed = time.time() - start_time
            if "timeout" in str(e).lower() or "statement_timeout" in str(e).lower():
                print(f" [TIMEOUT after {elapsed:.2f}s]")
            else:
                print(f" [ERROR: {str(e)}]")
        if "timeout" in str(e).lower() or "statement_timeout" in str(e).lower() or "tuple index out of range" in str(e).lower():
            return []
        # For other errors, still return empty to allow graceful degradation
        return []
    
    if not matched_entity_ids:
        if verbose:
            elapsed = time.time() - start_time
            print(f"[Fuzzy expansion] No matches found ({elapsed:.2f}s total)")
        return []
    
    if verbose:
        print(f"[Fuzzy expansion] Found {len(matched_entity_ids)} matching entities, collecting aliases...", end="", flush=True)
    
    # 3) Collect all aliases for matched entities (same logic as exact expansion)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id AS entity_id, e.canonical_name AS term
            FROM entities e
            WHERE e.source_id = %s
            ORDER BY e.id
            LIMIT %s;
            """,
            (source_id, 5000),  # Same limit as exact expansion
        )
        entity_rows = cur.fetchall()
    
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ea.entity_id, ea.alias AS term
            FROM entity_aliases ea
            WHERE ea.source_id = %s
            ORDER BY ea.entity_id, ea.id;
            """,
            (source_id,),
        )
        alias_rows = cur.fetchall()
    
    # 4) Collect expansions (canonical + aliases for matched entities)
    out: List[str] = []
    seen = set()
    
    def _emit(term: str):
        t = (term or "").strip()
        if not t:
            return
        tlc = t.lower()
        if tlc in seen:
            return
        seen.add(tlc)
        out.append(t)
    
    for entity_id, term in entity_rows:
        if int(entity_id) in matched_entity_ids:
            _emit(term)
    
    for entity_id, term in alias_rows:
        if int(entity_id) in matched_entity_ids:
            _emit(term)
    
    # Prefer shorter terms first (nicer for query-variant generation)
    out.sort(key=lambda s: (len(s), s.lower()))
    
    result = out[:max_aliases_out]
    
    if verbose:
        elapsed = time.time() - start_time
        print(f" collected {len(result)} expansion terms ({elapsed:.2f}s total)")
    
    return result


def build_expanded_query_string(
    user_query: str,
    expansions: List[str],
    *,
    max_append: int = 12,
) -> str:
    """
    For vector embeddings: append expansions in a natural-language way.
    Keep it short to avoid drowning the embedding in noise.
    """
    uq = _normalize_q(user_query)
    if not expansions:
        return uq

    # Don’t re-add the original query
    exp = [e for e in expansions if e.lower() != uq.lower()]
    if not exp:
        return uq

    exp = exp[:max_append]
    return uq + " (also known as: " + "; ".join(exp) + ")"


# -----------------------
# Types
# -----------------------

@dataclass(frozen=True)
class SearchFilters:
    chunk_pv: str = "chunk_v1_full"
    meta_pv: Optional[str] = None  # defaults to chunk_pv if None

    # optional filters
    collection_slugs: Optional[List[str]] = None
    document_id: Optional[int] = None  
    date_from: Optional[str] = None  # YYYY-MM-DD
    date_to: Optional[str] = None    # YYYY-MM-DD
    
    # exclusion filters (for pagination/novelty - V7 Phase 2)
    exclude_chunk_ids: Optional[List[int]] = None      # granular exclusion
    exclude_page_ids: Optional[List[int]] = None       # reduces near-duplicate churn
    exclude_document_ids: Optional[List[int]] = None   # broader exclusion


@dataclass
class ChunkHit:
    chunk_id: int
    collection_slug: str
    document_id: int
    first_page_id: int
    last_page_id: int
    date_min: Optional[str]
    date_max: Optional[str]
    preview: str

    # scoring (one or more will be populated depending on mode)
    score: Optional[float] = None
    distance: Optional[float] = None
    r_vec: Optional[int] = None
    r_lex: Optional[int] = None
    r_soft_lex: Optional[int] = None  # Soft lex rank (for approximate matches)
    soft_lex_score: Optional[float] = None  # Similarity score from soft lex

    # optional extras
    match_debug: Optional[str] = None
    
    # expansion metadata (query-level, same for all hits in a result set)
    expand_concordance: Optional[bool] = None
    expanded_query_text: Optional[str] = None
    expansion_terms: Optional[List[str]] = None
    concordance_source_slug: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------
# DB + embedding helpers
# -----------------------

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def log_retrieval_run(
    conn,
    query_text: str,
    search_type: str,
    chunk_pv: str,
    embedding_model: Optional[str],
    top_k: int,
    returned_chunk_ids: List[int],
    *,
    expanded_query_text: Optional[str] = None,
    expansion_terms: Optional[List[str]] = None,
    expand_concordance: bool = False,
    concordance_source_slug: Optional[str] = None,
    # forward-compatible "immutable receipt" fields
    query_lang_version: str = "qv1",
    retrieval_impl_version: str = "retrieval_v1",
    normalization_version: Optional[str] = None,
    retrieval_config_json: Optional[Dict[str, Any]] = None,
    # vector stability fields
    vector_metric: Optional[str] = "cosine",
    embedding_dim: Optional[int] = 1536,
    embed_text_version: Optional[str] = "embed_text_v1",
    # lexical explainability
    tsquery_text: Optional[str] = None,
    # Transaction control: if auto_commit=True, commit immediately; if False, caller commits
    auto_commit: bool = False,
    # Session tracking
    session_id: Optional[int] = None,
) -> int:
    """
    Log a retrieval run to retrieval_runs table.
    Includes expansion tracking if provided.
    
    By default (auto_commit=False), does NOT commit. Caller should commit after
    inserting evidence to ensure both operations are in the same transaction.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO retrieval_runs(
                query_text,
                expanded_query_text,
                expansion_terms,
                expand_concordance,
                concordance_source_slug,
                search_type,
                chunk_pv,
                embedding_model,
                top_k,
                returned_chunk_ids,
                query_lang_version,
                retrieval_impl_version,
                normalization_version,
                retrieval_config_json,
                vector_metric,
                embedding_dim,
                embed_text_version,
                tsquery_text,
                session_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                query_text,
                expanded_query_text,
                expansion_terms,
                expand_concordance,
                concordance_source_slug,
                search_type,
                chunk_pv,
                embedding_model,
                top_k,
                returned_chunk_ids,
                query_lang_version,
                retrieval_impl_version,
                normalization_version,
                Json(retrieval_config_json or {}),
                vector_metric,
                embedding_dim,
                embed_text_version,
                tsquery_text,
                session_id,
            ),
        )
        run_id = int(cur.fetchone()[0])
        # Only commit if explicitly requested (for backward compatibility)
        # Otherwise, caller commits after inserting evidence
        if auto_commit:
            conn.commit()
        return run_id


def _parse_lexemes_from_tsquery(tsquery_text: Optional[str]) -> Optional[List[str]]:
    """
    Parse lexemes from tsquery string (e.g., "venona | berlin | 1944" -> ["venona", "berlin", "1944"]).
    Simple deterministic parsing: split on | and strip whitespace/quotes.
    """
    if not tsquery_text or tsquery_text == "___nomatch___":
        return None
    
    # Split on | and clean up
    lexemes = []
    for part in tsquery_text.split("|"):
        part = part.strip()
        # Remove quotes if present
        part = part.strip("'\"")
        if part and part != "___nomatch___":
            lexemes.append(part)
    
    return lexemes if lexemes else None


def _fetch_highlights(
    conn,
    chunk_ids: List[int],
    tsquery_text: Optional[str],
    chunk_pv: str,
) -> Dict[int, str]:
    """
    Fetch ts_headline highlights for chunks using the tsquery.
    Returns dict mapping chunk_id -> highlight text.
    """
    if not tsquery_text or tsquery_text == "___nomatch___" or not chunk_ids:
        return {}
    
    highlights: Dict[int, str] = {}
    
    with conn.cursor() as cur:
        # Use ts_headline to generate highlights
        # tsquery_text is in websearch format, so use websearch_to_tsquery
        cur.execute(
            """
            SELECT
                c.id AS chunk_id,
                ts_headline(
                    'simple',
                    COALESCE(c.clean_text, c.text),
                    websearch_to_tsquery('simple', %s),
                    'StartSel=<mark>, StopSel=</mark>, MaxWords=35, MinWords=15'
                ) AS highlight
            FROM chunks c
            WHERE c.id = ANY(%s::bigint[])
              AND c.pipeline_version = %s
              AND to_tsvector('simple', COALESCE(c.clean_text, c.text)) @@ websearch_to_tsquery('simple', %s)
            """,
            (tsquery_text, chunk_ids, chunk_pv, tsquery_text),
        )
        
        for chunk_id, highlight in cur.fetchall():
            if highlight:
                # Clean up the highlight (remove mark tags for now, or keep them)
                highlight_clean = highlight.replace("<mark>", "").replace("</mark>", "")
                highlights[chunk_id] = highlight_clean.strip()
    
    return highlights


def _insert_run_chunk_evidence(
    conn,
    *,
    retrieval_run_id: int,
    hits: List["ChunkHit"],
    search_type: str,
    tsquery_text: Optional[str] = None,
    matched_lexemes: Optional[List[str]] = None,
    rrf_k: Optional[int] = None,
    embedding_model: Optional[str] = None,
    vector_metric: Optional[str] = "cosine",
    embedding_dim: Optional[int] = 1536,
    embed_text_version: Optional[str] = "embed_text_v1",
    chunk_pv: Optional[str] = None,
    original_query: Optional[str] = None,  # Original user query (for approx_lex.query_terms)
    # Transaction control: if auto_commit=True, commit immediately; if False, caller commits
    auto_commit: bool = False,
) -> None:
    """
    Persist per-chunk evidence for a run.
    Designed to satisfy acceptance criteria: one evidence row per returned chunk.
    
    Rank is set based on the order of hits (enumerate(hits, start=1)), which should
    match the order of returned_chunk_ids from the retrieval run.
    
    By default (auto_commit=False), does NOT commit. Caller should commit after
    log_retrieval_run to ensure both operations are in the same transaction.
    """
    # Fetch highlights for lex/hybrid runs if we have tsquery_text
    highlights: Dict[int, str] = {}
    if search_type in ("lex", "hybrid") and tsquery_text and chunk_pv:
        chunk_ids_list = [h.chunk_id for h in hits]
        highlights = _fetch_highlights(conn, chunk_ids_list, tsquery_text, chunk_pv)
    
    rows: List[Tuple[Any, ...]] = []

    for rank, h in enumerate(hits, start=1):
        score_lex = None
        score_vec = None
        score_hybrid = None

        explain: Dict[str, Any] = {
            "search_type": search_type,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "vector_metric": vector_metric,
            "embed_text_version": embed_text_version,
        }

        if search_type == "vector":
            # Vector-only search: populate semantic section with semantic_terms
            score_vec = float(h.distance) if h.distance is not None else None
            explain["distance"] = float(h.distance) if h.distance is not None else None
            
            # Extract semantic terms: query terms + chunk keywords
            query_terms = []
            chunk_keywords = []
            semantic_terms = []
            
            if original_query:
                query_terms = _extract_semantic_terms_from_query(original_query)
            
            # Fetch chunk text to extract keywords
            if chunk_pv:
                try:
                    with conn.cursor() as kw_cur:
                        kw_cur.execute(
                            """
                            SELECT COALESCE(c.clean_text, c.text)
                            FROM chunks c
                            WHERE c.id = %s AND c.pipeline_version = %s
                            LIMIT 1
                            """,
                            (h.chunk_id, chunk_pv),
                        )
                        chunk_row = kw_cur.fetchone()
                        if chunk_row and chunk_row[0]:
                            chunk_text = chunk_row[0]
                            chunk_keywords = _extract_chunk_keywords(chunk_text, max_keywords=5)
                except Exception:
                    # If keyword extraction fails, continue without chunk keywords
                    pass
            
            semantic_terms = _combine_semantic_terms(query_terms, chunk_keywords, max_total=8)
            
            explain["semantic"] = {
                "method": "query_terms+chunk_keywords_v1",
                "distance": float(h.distance) if h.distance is not None else None,
                "r_vec": h.r_vec,
                "semantic_terms": semantic_terms,
            }
            # For vector-only, we don't have lexical terms, but we can note it's semantic-only
            if embedding_model:
                explain["semantic"]["model"] = embedding_model

        elif search_type == "hybrid":
            score_hybrid = float(h.score) if h.score is not None else None
            explain.update({"rrf_score": score_hybrid, "r_vec": h.r_vec, "r_lex": h.r_lex, "rrf_k": rrf_k})
            # Optional: persist component contributions if we have ranks.
            if rrf_k is not None:
                if h.r_vec is not None:
                    score_vec = float(1.0 / (rrf_k + h.r_vec))
                if h.r_lex is not None:
                    score_lex = float(1.0 / (rrf_k + h.r_lex))
            
            # Populate semantic section if vector component exists
            if h.r_vec is not None:
                # Extract semantic terms: query terms + chunk keywords
                query_terms = []
                chunk_keywords = []
                semantic_terms = []
                
                if original_query:
                    query_terms = _extract_semantic_terms_from_query(original_query)
                
                # Fetch chunk text to extract keywords
                if chunk_pv:
                    try:
                        with conn.cursor() as kw_cur:
                            kw_cur.execute(
                                """
                                SELECT COALESCE(c.clean_text, c.text)
                                FROM chunks c
                                WHERE c.id = %s AND c.pipeline_version = %s
                                LIMIT 1
                                """,
                                (h.chunk_id, chunk_pv),
                            )
                            chunk_row = kw_cur.fetchone()
                            if chunk_row and chunk_row[0]:
                                chunk_text = chunk_row[0]
                                chunk_keywords = _extract_chunk_keywords(chunk_text, max_keywords=5)
                    except Exception:
                        # If keyword extraction fails, continue without chunk keywords
                        pass
                
                semantic_terms = _combine_semantic_terms(query_terms, chunk_keywords, max_total=8)
                
                explain["semantic"] = {
                    "method": "query_terms+chunk_keywords_v1",
                    "r_vec": h.r_vec,
                    "contribution": float(1.0 / (rrf_k + h.r_vec)) if rrf_k else None,
                    "semantic_terms": semantic_terms,
                }
                if embedding_model:
                    explain["semantic"]["model"] = embedding_model
            
            # Check if this is an approximate match (soft lex only, no exact match)
            is_approximate_only = h.r_soft_lex is not None and h.r_lex is None
            
            # Always populate lex section if there's an exact match
            if h.r_lex is not None:
                # Exact match: populate lex section
                if matched_lexemes is None and tsquery_text:
                    matched_lexemes = _parse_lexemes_from_tsquery(tsquery_text)
                
                explain["lex"] = {
                    "matched_lexemes": matched_lexemes or [],
                    "method": "exact_fts",
                    "r_lex": h.r_lex,
                }
                if rrf_k is not None:
                    explain["lex"]["contribution"] = float(1.0 / (rrf_k + h.r_lex))
            
            # Populate approx_lex section if there's an approximate match
            if h.r_soft_lex is not None:
                # Approximate match: populate approx_lex in explain_json
                # Extract query terms from original query (what user searched)
                query_terms = []
                if original_query:
                    query_terms = _extract_query_terms(original_query)
                
                # Parse matched terms from tsquery_text (what actually matched in document)
                matched_terms = []
                if tsquery_text:
                    matched_terms = _parse_lexemes_from_tsquery(tsquery_text)
                
                explain["approx_lex"] = {
                    "method": "trigram_v1",
                    "score": float(h.soft_lex_score) if h.soft_lex_score is not None else None,
                    "r_soft_lex": h.r_soft_lex,
                    "query_terms": query_terms,  # What user searched (e.g., ["silvermastre"])
                    "matched_terms": matched_terms,  # What actually matched (e.g., ["silvermaster"])
                }
                
                # For approximate-only matches, use matched_terms (what actually matched) in matched_lexemes
                if is_approximate_only and matched_lexemes is None:
                    matched_lexemes = matched_terms

        else:
            # lexical primitives: matched_lexemes + highlight are the key evidence
            # Parse matched_lexemes from tsquery_text (the actual query used at runtime)
            if matched_lexemes is None and tsquery_text:
                matched_lexemes = _parse_lexemes_from_tsquery(tsquery_text)
            
            # Populate lex section for lexical-only searches
            explain["lex"] = {
                "matched_lexemes": matched_lexemes or [],
                "method": "exact_fts",
            }

        # Use ts_headline highlight if available, otherwise preview
        highlight = highlights.get(h.chunk_id) or h.preview or None
        
        # For approximate matches, try to generate highlight using matched terms
        # This helps historians see what actually matched, not just what they searched
        if search_type == "hybrid" and h.r_soft_lex is not None and h.r_lex is None:
            # Approximate match: try to highlight matched terms instead of query terms
            if matched_lexemes and chunk_pv:
                # Generate highlight using matched terms (what actually matched)
                matched_terms_tsquery = " | ".join(matched_lexemes)
                try:
                    with conn.cursor() as highlight_cur:
                        highlight_cur.execute(
                            """
                            SELECT ts_headline(
                                'simple',
                                COALESCE(c.clean_text, c.text),
                                to_tsquery('simple', %s),
                                'StartSel=<mark>, StopSel=</mark>, MaxWords=35, MinWords=15'
                            ) AS highlight
                            FROM chunks c
                            WHERE c.id = %s AND c.pipeline_version = %s
                            LIMIT 1
                            """,
                            (matched_terms_tsquery, h.chunk_id, chunk_pv),
                        )
                        highlight_row = highlight_cur.fetchone()
                        if highlight_row and highlight_row[0]:
                            highlight = highlight_row[0]
                            # Store highlight in explain_json
                            if "approx_lex" in explain:
                                explain["approx_lex"]["highlight"] = highlight
                except Exception:
                    # If highlight generation fails, use existing highlight or preview
                    pass
        
        # Store highlight in lex section if available
        if highlight and search_type in ("lex", "hybrid"):
            if "lex" in explain:
                explain["lex"]["highlight"] = highlight
            elif search_type == "hybrid" and h.r_lex is not None:
                # Hybrid with exact lex match but no lex section yet
                if "lex" not in explain:
                    explain["lex"] = {"method": "exact_fts", "r_lex": h.r_lex}
                explain["lex"]["highlight"] = highlight
        
        # Acceptance: For lex/hybrid, ensure we have either matched_lexemes or highlight
        if search_type in ("lex", "hybrid"):
            if not matched_lexemes and not highlight:
                # Fallback: use preview as highlight if we have nothing
                highlight = h.preview or None

        rows.append(
            (
                retrieval_run_id,
                h.chunk_id,
                rank,
                score_lex,
                score_vec,
                score_hybrid,
                matched_lexemes,
                highlight,
                Json(explain),
            )
        )

    if not rows:
        # No evidence to insert (shouldn't happen, but be defensive)
        return
    
    with conn.cursor() as cur:
        try:
            cur.executemany(
                """
                INSERT INTO retrieval_run_chunk_evidence(
                    retrieval_run_id,
                    chunk_id,
                    rank,
                    score_lex,
                    score_vec,
                    score_hybrid,
                    matched_lexemes,
                    highlight,
                    explain_json
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (retrieval_run_id, chunk_id) DO NOTHING;
                """,
                rows,
            )
            # Only commit if explicitly requested (for backward compatibility)
            # Otherwise, caller commits after log_retrieval_run to ensure both operations are in the same transaction
            if auto_commit:
                conn.commit()
        except Exception as e:
            # Rollback on error and re-raise
            conn.rollback()
            raise RuntimeError(f"Failed to insert evidence for run {retrieval_run_id}: {e}") from e


def embed_query(text: str) -> List[float]:
    """Embed query using OpenAI embeddings. Assumes chunks.embedding is vector(1536)."""
    from openai import OpenAI  # pip install openai
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    # Defensive normalization + validation (prevents OpenAI 400 "$.input is invalid")
    text = _normalize_q(text)
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        raise ValueError("embed_query got empty input text")
    client = OpenAI()
    # Some environments are sensitive to input shape; pass as a single string.
    resp = client.embeddings.create(model=model, input=text)
    vec = resp.data[0].embedding
    if len(vec) != 1536:
        raise RuntimeError(f"Query embedding dim {len(vec)} != 1536 (expected vector(1536))")
    return vec


def vector_literal(vec: Sequence[float]) -> str:
    # pgvector accepts: '[1,2,3]'::vector
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _meta_pv(filters: SearchFilters) -> str:
    return filters.meta_pv or filters.chunk_pv


def _build_where(filters: SearchFilters, params: Dict[str, Any]) -> str:
    """
    Build a generic WHERE clause for chunks + chunk_metadata.
    Assumes caller uses:
      FROM chunks c
      JOIN chunk_metadata cm ON cm.chunk_id = c.id
    """
    where = [
        "c.pipeline_version = %(chunk_pv)s",
        "cm.pipeline_version = %(meta_pv)s",
    ]
    params["chunk_pv"] = filters.chunk_pv
    params["meta_pv"] = _meta_pv(filters)

    if filters.collection_slugs:
        where.append("cm.collection_slug = ANY(%(collection_slugs)s)")
        params["collection_slugs"] = filters.collection_slugs

    if filters.document_id is not None:
        where.append("cm.document_id = %(document_id)s")
        params["document_id"] = filters.document_id

    if filters.date_from:
        # include chunks that might overlap the range
        where.append("(cm.date_max IS NULL OR cm.date_max >= %(date_from)s::date)")
        params["date_from"] = filters.date_from

    if filters.date_to:
        where.append("(cm.date_min IS NULL OR cm.date_min <= %(date_to)s::date)")
        params["date_to"] = filters.date_to

    # Exclusion filters (V7 Phase 2 - pagination/novelty controls)
    if filters.exclude_chunk_ids:
        where.append("c.id != ALL(%(exclude_chunk_ids)s)")
        params["exclude_chunk_ids"] = filters.exclude_chunk_ids

    if filters.exclude_page_ids:
        where.append("cm.first_page_id != ALL(%(exclude_page_ids)s)")
        params["exclude_page_ids"] = filters.exclude_page_ids

    if filters.exclude_document_ids:
        where.append("cm.document_id != ALL(%(exclude_document_ids)s)")
        params["exclude_document_ids"] = filters.exclude_document_ids

    return " AND ".join(where)


def _clean_preview(s: str) -> str:
    return (s or "").replace("\n", " ").strip()


def _fetch_hits(
    cur, 
    rows: List[Tuple], 
    mode: str,
    *,
    expand_concordance: Optional[bool] = None,
    expanded_query_text: Optional[str] = None,
    expansion_terms: Optional[List[str]] = None,
    concordance_source_slug: Optional[str] = None,
) -> List[ChunkHit]:
    hits: List[ChunkHit] = []
    for r in rows:
        # common columns we always select
        # Handle variable number of columns (with or without soft_lex fields)
        if len(r) >= 13:
            # Has soft_lex fields (r_soft_lex, soft_lex_score)
            (
                chunk_id,
                collection_slug,
                document_id,
                first_page_id,
                last_page_id,
                date_min,
                date_max,
                preview,
                score,
                distance,
                r_vec,
                r_lex,
                r_soft_lex,
                soft_lex_score,
            ) = r[:14]  # Take first 14 elements
        else:
            # No soft_lex fields (legacy format)
            (
                chunk_id,
                collection_slug,
                document_id,
                first_page_id,
                last_page_id,
                date_min,
                date_max,
                preview,
                score,
                distance,
                r_vec,
                r_lex,
            ) = r
            r_soft_lex = None
            soft_lex_score = None

        hit = ChunkHit(
            chunk_id=chunk_id,
            collection_slug=collection_slug,
            document_id=document_id,
            first_page_id=first_page_id,
            last_page_id=last_page_id,
            date_min=str(date_min) if date_min is not None else None,
            date_max=str(date_max) if date_max is not None else None,
            preview=_clean_preview(preview),
            score=float(score) if score is not None else None,
            distance=float(distance) if distance is not None else None,
            r_vec=r_vec,
            r_lex=r_lex,
            r_soft_lex=r_soft_lex,
            soft_lex_score=float(soft_lex_score) if soft_lex_score is not None else None,
            expand_concordance=expand_concordance,
            expanded_query_text=expanded_query_text,
            expansion_terms=expansion_terms,
            concordance_source_slug=concordance_source_slug,
        )
        hits.append(hit)
    return hits


# -----------------------
# Lexical primitives
# -----------------------

def lex_exact(
    conn,
    term: str,
    *,
    filters: SearchFilters,
    k: int = 20,
    preview_chars: int = 2000,
    case_sensitive: bool = False,
    log_run: bool = True,
) -> List[ChunkHit]:
    """
    Lexical "exact" (transparent): substring match on COALESCE(clean_text, text).
    Note: this is not token-exact; it is literal substring match (optionally case-sensitive).
    """
    params: Dict[str, Any] = {"k": k, "preview_chars": preview_chars}
    where_sql = _build_where(filters, params)

    if case_sensitive:
        # LIKE is case-sensitive depending on collation; safest: use POSITION on raw string
        # We do a simple POSITION on the display text.
        params["term"] = term
        match_sql = "POSITION(%(term)s IN COALESCE(c.clean_text, c.text)) > 0"
    else:
        params["pat"] = f"%{term}%"
        match_sql = "COALESCE(c.clean_text, c.text) ILIKE %(pat)s"

    sql = f"""
    SELECT
      c.id AS chunk_id,
      cm.collection_slug,
      cm.document_id,
      cm.first_page_id,
      cm.last_page_id,
      cm.date_min,
      cm.date_max,
      LEFT(COALESCE(c.clean_text, c.text), %(preview_chars)s) AS preview,
      NULL::double precision AS score,
      NULL::double precision AS distance,
      NULL::int AS r_vec,
      NULL::int AS r_lex
    FROM chunks c
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE {where_sql}
      AND ({match_sql})
    ORDER BY cm.document_id, cm.first_page_id, c.id
    LIMIT %(k)s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    hits = _fetch_hits(cur=None, rows=rows, mode="lex_exact")
    
    # Log retrieval run
    if log_run:
        chunk_ids = [h.chunk_id for h in hits]
        # Build tsquery_text for explainability (simple term -> tsquery)
        tsquery_text = _fts_or_query(term)
        run_id = log_retrieval_run(
            conn,
            query_text=term,
            search_type="lex",
            chunk_pv=filters.chunk_pv or "all",  # Use "all" when no filter specified
            embedding_model=None,  # Lexical search doesn't use embeddings
            top_k=k,
            returned_chunk_ids=chunk_ids,
            expanded_query_text=None,  # Lexical functions don't support expansion
            expansion_terms=None,
            expand_concordance=False,
            concordance_source_slug=None,
            tsquery_text=tsquery_text,
            auto_commit=False,  # Commit after evidence insert
        )
        _insert_run_chunk_evidence(
            conn,
            retrieval_run_id=run_id,
            hits=hits,
            search_type="lex",
            tsquery_text=tsquery_text,
            matched_lexemes=[term],
            embedding_model=None,
            chunk_pv=filters.chunk_pv or "all",
            auto_commit=False,  # Commit after both operations
        )
        # Commit both operations in the same transaction
        conn.commit()
    
    return hits


def lex_and(
    conn,
    terms: Sequence[str],
    *,
    filters: SearchFilters,
    k: int = 20,
    preview_chars: int = 2000,
    case_sensitive: bool = False,
    log_run: bool = True,
) -> List[ChunkHit]:
    """
    Lexical AND: all terms must occur within the same chunk (substring semantics).
    """
    if len(terms) < 2:
        raise ValueError("lex_and requires at least 2 terms")

    params: Dict[str, Any] = {"k": k, "preview_chars": preview_chars}
    where_sql = _build_where(filters, params)

    clauses: List[str] = []
    if case_sensitive:
        for i, t in enumerate(terms):
            params[f"t{i}"] = t
            clauses.append(f"POSITION(%(t{i})s IN COALESCE(c.clean_text, c.text)) > 0")
    else:
        for i, t in enumerate(terms):
            params[f"p{i}"] = f"%{t}%"
            clauses.append(f"COALESCE(c.clean_text, c.text) ILIKE %(p{i})s")

    match_sql = " AND ".join(clauses)

    sql = f"""
    SELECT
      c.id AS chunk_id,
      cm.collection_slug,
      cm.document_id,
      cm.first_page_id,
      cm.last_page_id,
      cm.date_min,
      cm.date_max,
      LEFT(COALESCE(c.clean_text, c.text), %(preview_chars)s) AS preview,
      NULL::double precision AS score,
      NULL::double precision AS distance,
      NULL::int AS r_vec,
      NULL::int AS r_lex
    FROM chunks c
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE {where_sql}
      AND ({match_sql})
    ORDER BY cm.document_id, cm.first_page_id, c.id
    LIMIT %(k)s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    hits = _fetch_hits(cur=None, rows=rows, mode="lex_and")
    
    # Log retrieval run (query_text is the AND-separated terms)
    if log_run:
        query_text = " AND ".join(terms)
        chunk_ids = [h.chunk_id for h in hits]
        # Build tsquery_text: OR all terms together (for highlight generation)
        tsquery_text = _fts_or_query(" ".join(terms))
        run_id = log_retrieval_run(
            conn,
            query_text=query_text,
            search_type="lex",
            chunk_pv=filters.chunk_pv or "all",  # Use "all" when no filter specified
            embedding_model=None,  # Lexical search doesn't use embeddings
            top_k=k,
            returned_chunk_ids=chunk_ids,
            expanded_query_text=None,  # Lexical functions don't support expansion
            expansion_terms=None,
            expand_concordance=False,
            concordance_source_slug=None,
            tsquery_text=tsquery_text,
            auto_commit=False,  # Commit after evidence insert
        )
        _insert_run_chunk_evidence(
            conn,
            retrieval_run_id=run_id,
            hits=hits,
            search_type="lex",
            tsquery_text=tsquery_text,
            matched_lexemes=list(terms),
            embedding_model=None,
            chunk_pv=filters.chunk_pv or "all",
            auto_commit=False,  # Commit after both operations
        )
        # Commit both operations in the same transaction
        conn.commit()
    
    return hits


def _tokenize_for_near(text: str) -> List[str]:
    # lightweight word tokenization; deterministic
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _min_token_distance(toks: List[str], a: str, b: str) -> Optional[int]:
    a = a.lower()
    b = b.lower()
    apos = [i for i, t in enumerate(toks) if t == a]
    bpos = [i for i, t in enumerate(toks) if t == b]
    if not apos or not bpos:
        return None
    best = None
    for i in apos:
        for j in bpos:
            d = abs(i - j)
            if best is None or d < best:
                best = d
    return best


def lex_near(
    conn,
    a: str,
    b: str,
    *,
    window_words: int,
    filters: SearchFilters,
    k: int = 20,
    preview_chars: int = 2000,
    candidate_pool: int = 2000,
    log_run: bool = True,
) -> List[ChunkHit]:
    """
    Proximity search (stretch-friendly):
      1) SQL prefilter: chunks containing both a and b (ILIKE)
      2) Python verify: min token distance <= window_words
    This avoids working on tsquery proximity edge-cases, while staying deterministic and auditable.
    """
    params: Dict[str, Any] = {
        "candidate_pool": candidate_pool,
        "preview_chars": preview_chars,
        "pa": f"%{a}%",
        "pb": f"%{b}%",
    }
    where_sql = _build_where(filters, params)

    sql = f"""
    SELECT
      c.id AS chunk_id,
      cm.collection_slug,
      cm.document_id,
      cm.first_page_id,
      cm.last_page_id,
      cm.date_min,
      cm.date_max,
      COALESCE(c.clean_text, c.text) AS full_text,
      LEFT(COALESCE(c.clean_text, c.text), %(preview_chars)s) AS preview
    FROM chunks c
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE {where_sql}
      AND COALESCE(c.clean_text, c.text) ILIKE %(pa)s
      AND COALESCE(c.clean_text, c.text) ILIKE %(pb)s
    ORDER BY cm.document_id, cm.first_page_id, c.id
    LIMIT %(candidate_pool)s;
    """

    hits: List[ChunkHit] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Verify proximity deterministically
    verified: List[Tuple[Optional[int], Tuple]] = []
    for r in rows:
        (
            chunk_id,
            collection_slug,
            document_id,
            first_page_id,
            last_page_id,
            date_min,
            date_max,
            full_text,
            preview,
        ) = r
        toks = _tokenize_for_near(full_text)
        dist = _min_token_distance(toks, a, b)
        if dist is not None and dist <= window_words:
            verified.append((dist, r))

    # Sort by closest match, then earliest provenance
    verified.sort(key=lambda x: (x[0] if x[0] is not None else 10**9, x[1][2], x[1][3], x[1][0]))

    for dist, r in verified[:k]:
        (
            chunk_id,
            collection_slug,
            document_id,
            first_page_id,
            last_page_id,
            date_min,
            date_max,
            _full_text,
            preview,
        ) = r
        hits.append(
            ChunkHit(
                chunk_id=chunk_id,
                collection_slug=collection_slug,
                document_id=document_id,
                first_page_id=first_page_id,
                last_page_id=last_page_id,
                date_min=str(date_min) if date_min is not None else None,
                date_max=str(date_max) if date_max is not None else None,
                preview=_clean_preview(preview),
                score=None,
                distance=None,
                r_vec=None,
                r_lex=None,
                match_debug=f"token_dist={dist} window={window_words}",
            )
        )

    # Log retrieval run (query_text represents the NEAR query)
    if log_run:
        query_text = f"{a} NEAR {b} (window={window_words})"
        chunk_ids = [h.chunk_id for h in hits]
        # Build tsquery_text: OR both terms together (for highlight generation)
        tsquery_text = _fts_or_query(f"{a} {b}")
        run_id = log_retrieval_run(
            conn,
            query_text=query_text,
            search_type="lex",
            chunk_pv=filters.chunk_pv or "all",  # Use "all" when no filter specified
            embedding_model=None,  # Lexical search doesn't use embeddings
            top_k=k,
            returned_chunk_ids=chunk_ids,
            expanded_query_text=None,  # Lexical functions don't support expansion
            expansion_terms=None,
            expand_concordance=False,
            concordance_source_slug=None,
            tsquery_text=tsquery_text,
            auto_commit=False,  # Commit after evidence insert
        )
        _insert_run_chunk_evidence(
            conn,
            retrieval_run_id=run_id,
            hits=hits,
            search_type="lex",
            tsquery_text=tsquery_text,
            matched_lexemes=[a, b],
            embedding_model=None,
            chunk_pv=filters.chunk_pv or "all",
            auto_commit=False,  # Commit after both operations
        )
        # Commit both operations in the same transaction
        conn.commit()

    return hits


# -----------------------
# Vector + Hybrid (RRF)
# -----------------------

def vector_search(
    conn,
    query: str,
    *,
    filters: SearchFilters,
    k: int = 20,
    preview_chars: int = 2000,
    probes: int = 20,
    expand_concordance: bool = True,
    concordance_source_slug: str = "venona_vassiliev_concordance_v3",
    log_run: bool = True,
    session_id: Optional[int] = None,
) -> List[ChunkHit]:
    params: Dict[str, Any] = {"k": k, "preview_chars": preview_chars}
    where_sql = _build_where(filters, params)

    query_for_embedding = query
    exp_terms: List[str] = []
    if expand_concordance:
        exp_terms = concordance_expand_terms(conn, query, source_slug=concordance_source_slug)
        if exp_terms:
            query_for_embedding = build_expanded_query_string(query, exp_terms)

    qvec = embed_query(query_for_embedding)
    params["qvec"] = vector_literal(qvec)

    sql = f"""
    SELECT
      c.id AS chunk_id,
      cm.collection_slug,
      cm.document_id,
      cm.first_page_id,
      cm.last_page_id,
      cm.date_min,
      cm.date_max,
      LEFT(COALESCE(c.clean_text, c.text), %(preview_chars)s) AS preview,
      NULL::double precision AS score,
      (c.embedding <=> %(qvec)s::vector) AS distance,
      NULL::int AS r_vec,
      NULL::int AS r_lex
    FROM chunks c
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE {where_sql}
      AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(qvec)s::vector
    LIMIT %(k)s;
    """

    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = %s;", (probes,))
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    hits = _fetch_hits(
        cur=None,
        rows=rows,
        mode="vector",
        expand_concordance=expand_concordance if expand_concordance else None,
        expanded_query_text=query_for_embedding if expand_concordance else None,
        expansion_terms=exp_terms if expand_concordance and exp_terms else None,
        concordance_source_slug=concordance_source_slug if expand_concordance else None,
    )
    
    # Log retrieval run
    if log_run:
        embedding_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        chunk_ids = [h.chunk_id for h in hits]
        run_id = log_retrieval_run(
            conn,
            query_text=query,
            search_type="vector",
            chunk_pv=filters.chunk_pv or "all",  # Use "all" when no filter specified
            embedding_model=embedding_model,
            top_k=k,
            returned_chunk_ids=chunk_ids,
            expanded_query_text=query_for_embedding if expand_concordance else None,
            expansion_terms=exp_terms if expand_concordance else None,
            expand_concordance=expand_concordance,
            concordance_source_slug=concordance_source_slug if expand_concordance else None,
            vector_metric="cosine",
            embedding_dim=1536,
            embed_text_version="embed_text_v1",
            retrieval_config_json={
                "probes": probes,
            },
            auto_commit=False,  # Commit after evidence insert
            session_id=session_id,
        )
        _insert_run_chunk_evidence(
            conn,
            retrieval_run_id=run_id,
            hits=hits,
            search_type="vector",
            matched_lexemes=None,
            embedding_model=embedding_model,
            vector_metric="cosine",
            embedding_dim=1536,
            embed_text_version="embed_text_v1",
            chunk_pv=filters.chunk_pv or "all",
            original_query=query,
            auto_commit=False,  # Commit after both operations
        )
        # Commit both operations in the same transaction
        conn.commit()
    
    return hits


def _fts_or_query(user_query: str) -> str:
    """
    OR tokens together (so lexical doesn't go empty on natural language queries).
    This is for hybrid/lex ranking; NOT for 'exact match' primitives.
    """
    q = user_query.lower().strip()
    q = q.replace("u.s.", "usa").replace("u.s", "usa").replace("us ", "usa ")
    toks = re.findall(r"[a-z0-9_']+", q)
    toks = [t for t in toks if (len(t) >= 3 or t.isdigit())]
    toks = sorted(set(toks))
    if not toks:
        return "___nomatch___"
    return " | ".join(toks)


def _soft_lex_match(
    conn,
    query_terms: List[str],
    filters: SearchFilters,
    k: int,
    threshold: float = 0.3,
    normalization_version: Optional[str] = None,
) -> List[Tuple[int, float]]:
    """
    Soft lexical matching using pg_trgm similarity.
    
    Returns list of (chunk_id, similarity_score) tuples, ranked by score descending.
    Only returns chunks with similarity >= threshold.
    
    Args:
        conn: Database connection
        query_terms: List of query terms to match
        filters: Search filters
        k: Maximum number of results
        threshold: Minimum similarity score (0.0-1.0)
        normalization_version: Normalization version to apply (e.g., 'norm_v2')
    
    Returns:
        List of (chunk_id, similarity_score) tuples
    """
    if not query_terms:
        return []
    
    # Normalize query terms if normalization version specified
    from retrieval.normalization import normalize_query_term
    normalized_terms = []
    for term in query_terms:
        if normalization_version:
            normalized_term = normalize_query_term(term, normalization_version)
        else:
            normalized_term = term.lower().strip()
        if normalized_term:
            normalized_terms.append(normalized_term)
    
    if not normalized_terms:
        return []
    
    params: Dict[str, Any] = {
        "k": k,
        "threshold": threshold,
        "normalized_terms": normalized_terms,
    }
    where_sql = _build_where(filters, params)
    
    # Build similarity query: for each term, compute max similarity across all terms
    # Use word_similarity for better word-level matching
    similarity_conditions = []
    for i, term in enumerate(normalized_terms):
        param_key = f"term_{i}"
        params[param_key] = term
        # Use word_similarity for better matching (matches word within text)
        similarity_conditions.append(
            f"word_similarity(%({param_key})s, COALESCE(c.clean_text, c.text))"
        )
    
    # Take max similarity across all terms
    max_similarity_expr = "GREATEST(" + ", ".join(similarity_conditions) + ")"
    
    sql = f"""
    SELECT
        c.id AS chunk_id,
        {max_similarity_expr} AS similarity_score
    FROM chunks c
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    WHERE {where_sql}
      AND {max_similarity_expr} >= %(threshold)s
    ORDER BY similarity_score DESC
    LIMIT %(k)s
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [(row[0], float(row[1])) for row in cur.fetchall()]


def _extract_query_terms(query: str) -> List[str]:
    """
    Extract individual terms from a query string for soft lex matching.
    Similar to _fts_or_query but returns list instead of tsquery string.
    """
    q = query.lower().strip()
    q = q.replace("u.s.", "usa").replace("u.s", "usa").replace("us ", "usa ")
    toks = re.findall(r"[a-z0-9_']+", q)
    toks = [t for t in toks if (len(t) >= 3 or t.isdigit())]
    return sorted(set(toks))


def _extract_semantic_terms_from_query(query: str) -> List[str]:
    """
    Extract semantic terms (keyphrases/tokens) from query for vector explainability.
    Returns deduplicated, stopword-filtered tokens (2-8 items target).
    """
    # Use same tokenization as _extract_query_terms but keep more terms
    q = query.lower().strip()
    q = q.replace("u.s.", "usa").replace("u.s", "usa").replace("us ", "usa ")
    toks = re.findall(r"[a-z0-9_']+", q)
    # Keep tokens >= 2 chars (more permissive than lexical)
    toks = [t for t in toks if (len(t) >= 2 or t.isdigit())]
    # Filter stopwords
    stopwords = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "between", "explain", "relationship"}
    toks = [t for t in toks if t not in stopwords]
    # Dedup and sort
    return sorted(set(toks))[:8]  # Cap at 8


def _extract_chunk_keywords(chunk_text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract keywords from chunk text using simple heuristic:
    - Capitalized tokens (likely proper nouns/entities)
    - Simple TF heuristic: frequent tokens (excluding common words)
    
    Returns up to max_keywords items.
    """
    if not chunk_text:
        return []
    
    # Extract capitalized words (likely entities)
    capitalized = re.findall(r'\b[A-Z][a-z]+\b', chunk_text)
    
    # Extract all tokens and count frequency
    all_toks = re.findall(r"[a-zA-Z0-9']+", chunk_text.lower())
    
    # Filter common words
    common_words = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from", "by", "at", "as", "is", "was",
        "are", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "this", "that", "these", "those", "it", "its", "they", "them", "their", "there", "then", "than",
    }
    all_toks = [t for t in all_toks if len(t) >= 3 and t not in common_words]
    
    # Count frequency
    tok_counts = Counter(all_toks)
    
    # Get top frequent tokens
    top_toks = [tok for tok, count in tok_counts.most_common(max_keywords)]
    
    # Combine capitalized + top frequent, dedup, limit
    keywords = []
    seen = set()
    for kw in capitalized[:max_keywords] + top_toks:
        kw_lower = kw.lower()
        if kw_lower not in seen and len(kw_lower) >= 2:
            keywords.append(kw_lower)
            seen.add(kw_lower)
            if len(keywords) >= max_keywords:
                break
    
    return keywords


def _combine_semantic_terms(query_terms: List[str], chunk_keywords: List[str], max_total: int = 8) -> List[str]:
    """
    Combine query terms and chunk keywords into semantic_terms list (2-8 items).
    Prioritizes query terms, then adds chunk keywords if space available.
    """
    combined = []
    seen = set()
    
    # Add query terms first (up to half of max_total)
    query_limit = min(len(query_terms), max_total // 2)
    for term in query_terms[:query_limit]:
        if term not in seen:
            combined.append(term)
            seen.add(term)
    
    # Add chunk keywords if space available
    remaining = max_total - len(combined)
    for kw in chunk_keywords[:remaining]:
        if kw not in seen:
            combined.append(kw)
            seen.add(kw)
            if len(combined) >= max_total:
                break
    
    # Ensure at least 2 items if possible
    if len(combined) < 2 and query_terms:
        for term in query_terms:
            if term not in seen:
                combined.append(term)
                seen.add(term)
                if len(combined) >= 2:
                    break
    
    return combined[:max_total]


def hybrid_rrf(
    conn,
    query: str,
    *,
    filters: SearchFilters,
    k: int = 20,
    preview_chars: int = 2000,
    probes: int = 20,
    top_n_vec: int = 200,
    top_n_lex: int = 200,
    rrf_k: int = 50,
    expand_concordance: bool = True,
    concordance_source_slug: str = "venona_vassiliev_concordance_v3",
    log_run: bool = True,
    use_soft_lex: bool = False,
    soft_lex_threshold: float = 0.3,
    soft_lex_max_results: int = 50,
    soft_lex_weight: float = 0.5,
    soft_lex_trigram_threshold: float = 0.1,
    normalization_version: Optional[str] = None,
    fuzzy_lex_enabled: bool = True,
    fuzzy_lex_min_similarity: float = 0.4,
    fuzzy_lex_top_k_per_token: int = 5,
    fuzzy_lex_max_total_variants: int = 50,
    session_id: Optional[int] = None,
) -> List[ChunkHit]:
    params: Dict[str, Any] = {
        "k": k,
        "preview_chars": preview_chars,
        "top_n_vec": top_n_vec,
        "top_n_lex": top_n_lex,
        "rrf_k": rrf_k,
        "soft_lex_trigram_threshold": soft_lex_trigram_threshold,
    }
    where_sql = _build_where(filters, params)

    # Optionally expand query for embeddings + lexical ranking.
    query_for_embedding = query
    exp_terms: List[str] = []
    if expand_concordance:
        # Try exact expansion first
        exp_terms = concordance_expand_terms(conn, query, source_slug=concordance_source_slug)
        
        # If exact expansion failed and soft lex is enabled, try fuzzy expansion
        # This handles user typos and OCR errors in the query
        # Note: Fuzzy expansion can be slow; it has a 30-second timeout and will gracefully fail
        # We use a higher max_candidates to ensure we find all entities that exact expansion would find
        # This ensures typos produce the same expansion terms as correct spelling
        if not exp_terms and use_soft_lex:
            try:
                exp_terms = concordance_expand_terms_fuzzy(
                    conn, 
                    query, 
                    source_slug=concordance_source_slug,
                    similarity_threshold=0.6,  # Higher threshold for expansion (more conservative)
                    max_candidates=20,  # Increased from 5 to find more entities (ensures same coverage as exact expansion)
                    max_aliases_out=25,  # Same as exact expansion
                    timeout_seconds=30.0,  # Timeout after 30 seconds
                    verbose=True,  # Show progress
                )
            except Exception:
                # If fuzzy expansion fails or times out, continue without expansion
                # Soft lex matching will still work on the original query
                exp_terms = []
        
        if exp_terms:
            query_for_embedding = build_expanded_query_string(query, exp_terms)

    qvec = embed_query(query_for_embedding)
    params["qvec"] = vector_literal(qvec)

    # Build lexical tsquery with per-token fuzzy expansion (fallback to OR tokens if dictionary missing)
    lex_aug = query if not exp_terms else (query + " " + " ".join(exp_terms))
    fuzzy_config = FuzzyLexConfig(
        enabled=fuzzy_lex_enabled,
        min_similarity=fuzzy_lex_min_similarity,
        top_k_per_token=fuzzy_lex_top_k_per_token,
        max_total_variants=fuzzy_lex_max_total_variants,
        min_token_len=3,
        stopwords=None,
    )
    build_id = None
    fuzzy_expansions: Dict[str, List[Dict[str, Any]]] = {}
    fuzzy_tokens: List[str] = []
    tsq = _fts_or_query(lex_aug)  # default fallback
    try:
        # choose a single collection slug for scoping if provided
        collection_slug = None
        if filters.collection_slugs and len(filters.collection_slugs) == 1:
            collection_slug = filters.collection_slugs[0]
        tsq_new, build_id, fuzzy_expansions, fuzzy_tokens = compile_hybrid_or_tsquery(
            lex_aug,
            conn=conn,
            chunk_pv=filters.chunk_pv or "all",
            collection_slug=collection_slug,
            norm_version=normalization_version or "norm_v1",
            config=fuzzy_config,
        )
        if tsq_new:
            tsq = tsq_new
    except Exception:
        # On any error, gracefully fall back to old behavior
        pass
    params["tsq"] = tsq
    
    # Determine query language version and retrieval implementation version
    query_lang_version = "qv2_softlex" if use_soft_lex else "qv1"
    retrieval_impl_version = "retrieval_v2_softlex" if use_soft_lex else "retrieval_v1"
    
    # Extract query terms for soft lex matching
    query_terms = _extract_query_terms(query)
    if exp_terms:
        # Add expansion terms for soft lex matching too
        # Limit to top 10 expansion terms to avoid performance issues with too many similarity computations
        # See docs/performance_enhancements.md for details
        limited_exp_terms = exp_terms[:10]
        for exp_term in limited_exp_terms:
            exp_terms_list = _extract_query_terms(exp_term)
            query_terms.extend(exp_terms_list)
        query_terms = sorted(set(query_terms))  # Deduplicate
    
    # Build SQL with optional soft lex CTE
    soft_lex_cte = ""
    soft_lex_join = ""
    soft_lex_score = ""
    soft_lex_select_fields = "NULL::int AS r_soft_lex, NULL::real AS soft_lex_score"
    soft_lex_chunk_id_coalesce = ""
    
    if use_soft_lex and query_terms:
        # Normalize query terms if normalization version specified
        normalized_terms = []
        if normalization_version:
            from retrieval.normalization import normalize_query_term
            for term in query_terms:
                normalized_term = normalize_query_term(term, normalization_version)
                if normalized_term:
                    normalized_terms.append(normalized_term)
        else:
            normalized_terms = query_terms
        
        if normalized_terms:
            # Log soft lex query building
            print(f"[Soft lex] Building query with {len(normalized_terms)} normalized terms...", end="", flush=True)
            
            # Build similarity conditions for soft lex
            similarity_conditions = []
            for i, term in enumerate(normalized_terms):
                param_key = f"soft_term_{i}"
                params[param_key] = term
                similarity_conditions.append(
                    f"word_similarity(%({param_key})s, COALESCE(c.clean_text, c.text))"
                )
            
            max_similarity_expr = "GREATEST(" + ", ".join(similarity_conditions) + ")"
            params["soft_lex_threshold"] = soft_lex_threshold
            params["soft_lex_max_results"] = soft_lex_max_results
            params["soft_lex_weight"] = soft_lex_weight
            params["soft_lex_trigram_threshold"] = soft_lex_trigram_threshold
            
            soft_lex_cte = f""",
    soft_lex AS (
      SELECT
        c.id AS chunk_id,
        {max_similarity_expr} AS similarity_score,
        row_number() OVER (ORDER BY {max_similarity_expr} DESC) AS r_soft_lex
      FROM chunks c
      JOIN chunk_metadata cm ON cm.chunk_id = c.id
      WHERE {where_sql}
        AND {max_similarity_expr} >= %(soft_lex_trigram_threshold)s
        AND {max_similarity_expr} >= %(soft_lex_threshold)s
      LIMIT %(soft_lex_max_results)s
    )"""
            
            soft_lex_join = """
      FULL OUTER JOIN soft_lex USING (chunk_id)"""
            
            # Add soft lex contribution with reduced weight
            soft_lex_score = f" + COALESCE({soft_lex_weight} * (1.0 / (%(rrf_k)s + soft_lex.r_soft_lex)), 0.0)"
            
            # Update select fields for soft lex
            soft_lex_select_fields = "soft_lex.r_soft_lex, soft_lex.similarity_score AS soft_lex_score"
            soft_lex_chunk_id_coalesce = ", soft_lex.chunk_id"
            
            print(" query built", flush=True)

    sql = f"""
    WITH
    vec AS (
      SELECT
        c.id AS chunk_id,
        row_number() OVER (ORDER BY c.embedding <=> %(qvec)s::vector) AS r_vec
      FROM chunks c
      JOIN chunk_metadata cm ON cm.chunk_id = c.id
      WHERE {where_sql}
        AND c.embedding IS NOT NULL
      ORDER BY c.embedding <=> %(qvec)s::vector
      LIMIT %(top_n_vec)s
    ),
    lex AS (
      SELECT
        c.id AS chunk_id,
        row_number() OVER (
          ORDER BY ts_rank_cd(
            to_tsvector('simple', COALESCE(c.clean_text, c.text)),
            to_tsquery('simple', %(tsq)s)
          ) DESC
        ) AS r_lex
      FROM chunks c
      JOIN chunk_metadata cm ON cm.chunk_id = c.id
      WHERE {where_sql}
        AND to_tsvector('simple', COALESCE(c.clean_text, c.text)) @@ to_tsquery('simple', %(tsq)s)
      LIMIT %(top_n_lex)s
    ){soft_lex_cte},
    fused AS (
      SELECT
        COALESCE(vec.chunk_id, lex.chunk_id{soft_lex_chunk_id_coalesce}) AS chunk_id,
        COALESCE(1.0 / (%(rrf_k)s + vec.r_vec), 0.0) +
        COALESCE(1.0 / (%(rrf_k)s + lex.r_lex), 0.0){soft_lex_score} AS score,
        vec.r_vec,
        lex.r_lex,
        {soft_lex_select_fields}
      FROM vec
      FULL OUTER JOIN lex USING (chunk_id){soft_lex_join}
    )
    SELECT
      c.id AS chunk_id,
      cm.collection_slug,
      cm.document_id,
      cm.first_page_id,
      cm.last_page_id,
      cm.date_min,
      cm.date_max,
      LEFT(COALESCE(c.clean_text, c.text), %(preview_chars)s) AS preview,
      fused.score AS score,
      NULL::double precision AS distance,
      fused.r_vec,
      fused.r_lex,
      fused.r_soft_lex,
      fused.soft_lex_score
    FROM fused
    JOIN chunks c ON c.id = fused.chunk_id
    JOIN chunk_metadata cm ON cm.chunk_id = c.id
    ORDER BY fused.score DESC
    LIMIT %(k)s;
    """
    
    # Note: r_soft_lex and soft_lex_score are always selected (NULL when soft_lex not used)
    # This ensures consistent column count for _fetch_hits

    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = %s;", (probes,))
        
        # Log query execution start if soft lex is enabled
        if use_soft_lex and query_terms:
            import time
            print(f"[Soft lex] Executing hybrid query...", end="", flush=True)
            query_start_time = time.time()
        
        # Execute the query
        cur.execute(sql, params)
        
        # Log query execution completion if soft lex is enabled
        if use_soft_lex and query_terms:
            query_time = time.time() - query_start_time
            print(f" completed in {query_time:.2f}s", end="", flush=True)
        
        rows = cur.fetchall()
        
        # Log final results if soft lex is enabled
        if use_soft_lex and query_terms:
            print(f" ({len(rows)} results)", flush=True)
    
    hits = _fetch_hits(
        cur=None,
        rows=rows,
        mode="hybrid",
        expand_concordance=expand_concordance if expand_concordance else None,
        expanded_query_text=query_for_embedding if expand_concordance else None,
        expansion_terms=exp_terms if expand_concordance and exp_terms else None,
        concordance_source_slug=concordance_source_slug if expand_concordance else None,
    )
    
    # Log retrieval run
    if log_run:
        embedding_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        chunk_ids = [h.chunk_id for h in hits]
        # tsq is the tsquery_text already built above
        # Build retrieval config with soft lex + fuzzy settings if used
        retrieval_config = {
            "probes": probes,
            "top_n_vec": top_n_vec,
            "top_n_lex": top_n_lex,
            "rrf_k": rrf_k,
            "fuzzy_lex": {
                "enabled": fuzzy_lex_enabled,
                "min_similarity": fuzzy_lex_min_similarity,
                "top_k_per_token": fuzzy_lex_top_k_per_token,
                "max_total_variants": fuzzy_lex_max_total_variants,
                "dictionary_build_id": build_id,
                "tokens": fuzzy_tokens,
                "expansions": fuzzy_expansions,
            },
        }
        if use_soft_lex:
            retrieval_config.update({
                "soft_lex_threshold": soft_lex_threshold,
                "soft_lex_max_results": soft_lex_max_results,
                "soft_lex_weight": soft_lex_weight,
            })
        
        run_id = log_retrieval_run(
            conn,
            query_text=query,
            search_type="hybrid",
            chunk_pv=filters.chunk_pv or "all",  # Use "all" when no filter specified
            embedding_model=embedding_model,
            top_k=k,
            returned_chunk_ids=chunk_ids,
            expanded_query_text=query_for_embedding if expand_concordance else None,
            expansion_terms=exp_terms if expand_concordance else None,
            expand_concordance=expand_concordance,
            concordance_source_slug=concordance_source_slug if expand_concordance else None,
            query_lang_version=query_lang_version,  # Set to 'qv2_softlex' if soft lex used
            retrieval_impl_version=retrieval_impl_version,  # Set to 'retrieval_v2_softlex' if soft lex used
            normalization_version=normalization_version,  # Pass normalization version
            vector_metric="cosine",
            embedding_dim=1536,
            embed_text_version="embed_text_v1",
            retrieval_config_json=retrieval_config,
            tsquery_text=tsq,  # Pass the tsquery_text for lexical explainability
            auto_commit=False,  # Commit after evidence insert
            session_id=session_id,
        )
        _insert_run_chunk_evidence(
            conn,
            retrieval_run_id=run_id,
            hits=hits,
            search_type="hybrid",
            tsquery_text=tsq,  # Pass tsquery_text for highlight generation and matched_lexemes parsing
            matched_lexemes=None,  # Will be parsed from tsquery_text
            rrf_k=rrf_k,
            embedding_model=embedding_model,
            vector_metric="cosine",
            embedding_dim=1536,
            embed_text_version="embed_text_v1",
            chunk_pv=filters.chunk_pv or "all",
            original_query=query,  # Pass original query for approx_lex.query_terms
            auto_commit=False,  # Commit after both operations
        )
        # Commit both operations in the same transaction
        conn.commit()
    
    return hits


# =============================================================================
# Phase 4: Threshold-Based Search and SQL CTE RRF
# =============================================================================
# These functions implement Two-Mode Retrieval with:
# - Threshold-based vector search (similarity >= threshold, not top-k)
# - SQL CTE for RRF (deterministic, efficient for large result sets)
# - Mode-specific ranking policies

from retrieval.config import (
    VectorMetricConfig, DEFAULT_VECTOR_CONFIG,
    ConversationalModeConfig, ThoroughModeConfig,
)


@dataclass
class ThresholdSearchResult:
    """Result from threshold-based search."""
    chunk_id: int
    score_lexical: Optional[float] = None
    score_vector: Optional[float] = None
    score_hybrid: Optional[float] = None
    vector_distance: Optional[float] = None
    vector_similarity: Optional[float] = None
    rank_lexical: Optional[int] = None
    rank_vector: Optional[int] = None
    rank: Optional[int] = None
    in_lexical: bool = False
    in_vector: bool = False
    document_id: Optional[int] = None


@dataclass
class ThresholdSearchMetadata:
    """Metadata from threshold-based search."""
    total_lexical: int = 0
    total_vector: int = 0
    total_combined: int = 0
    total_before_cap: int = 0
    cap_applied: bool = False
    cap_value: Optional[int] = None
    threshold_used: float = 0.0
    vector_config: Optional[Dict[str, Any]] = None
    mode: str = "conversational"


def vector_search_threshold(
    conn,
    embedding: List[float],
    threshold: float,
    *,
    max_hits: Optional[int] = None,
    scope_sql: Optional[str] = None,
    scope_params: Optional[List[Any]] = None,
    vector_config: VectorMetricConfig = DEFAULT_VECTOR_CONFIG,
    filters: Optional[SearchFilters] = None,
) -> Tuple[List[ThresholdSearchResult], ThresholdSearchMetadata]:
    """
    Threshold-based vector search with explicit metric semantics.
    
    Unlike top-k search, this returns ALL chunks above the similarity threshold,
    then optionally caps for UX protection.
    
    Args:
        conn: Database connection
        embedding: Query embedding vector
        threshold: Minimum similarity (in transformed space, e.g., [-1, 1] for cosine)
        max_hits: Optional cap on results (for conversational mode)
        scope_sql: WHERE clause from scope primitives
        scope_params: Parameters for scope SQL
        vector_config: Metric configuration for similarity calculation
        filters: Additional SearchFilters
        
    Returns:
        (results, metadata) tuple
        
    Note: For cosine distance (<=>) with similarity = 1 - distance:
        - similarity range is [-1, 1] (can be negative)
        - threshold of 0.3 means distance <= 0.7
    """
    # Convert threshold to distance for SQL
    max_distance = vector_config.transform_to_distance(threshold)
    
    # Convert embedding to vector literal for PostgreSQL
    embedding_literal = vector_literal(embedding)
    
    # Build base WHERE clause
    params: Dict[str, Any] = {
        "embedding": embedding_literal,
        "max_distance": max_distance,
        "threshold": threshold,
    }
    
    where_clauses = ["c.embedding IS NOT NULL"]
    
    # Add filter clauses if provided
    if filters:
        filter_params: Dict[str, Any] = {}
        filter_sql = _build_where(filters, filter_params)
        if filter_sql and filter_sql != "TRUE":
            where_clauses.append(filter_sql)
            params.update(filter_params)
    
    # Add scope SQL if provided
    if scope_sql:
        processed_scope_sql = scope_sql
        if scope_params:
            # Convert list params to named params
            for i, p in enumerate(scope_params):
                params[f"scope_p{i}"] = p
                processed_scope_sql = processed_scope_sql.replace("%s", f"%(scope_p{i})s", 1)
        where_clauses.append(f"({processed_scope_sql})")
    
    where_sql = " AND ".join(where_clauses)
    
    # Query with threshold filter using cosine distance operator
    # Note: (1 - distance) gives similarity
    sql = f"""
        WITH threshold_results AS (
            SELECT 
                c.id AS chunk_id,
                c.embedding {vector_config.operator} %(embedding)s::vector AS distance,
                1.0 - (c.embedding {vector_config.operator} %(embedding)s::vector) AS similarity,
                cm.document_id,
                ROW_NUMBER() OVER (
                    ORDER BY c.embedding {vector_config.operator} %(embedding)s::vector ASC, c.id ASC
                ) AS rank
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE {where_sql}
              AND (c.embedding {vector_config.operator} %(embedding)s::vector) <= %(max_distance)s
        )
        SELECT chunk_id, distance, similarity, document_id, rank
        FROM threshold_results
        ORDER BY rank
    """
    
    # Add limit if max_hits specified
    if max_hits:
        sql += f" LIMIT {max_hits}"
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    # Build results
    results = []
    for row in rows:
        chunk_id, distance, similarity, document_id, rank = row
        results.append(ThresholdSearchResult(
            chunk_id=chunk_id,
            vector_distance=float(distance) if distance else None,
            vector_similarity=float(similarity) if similarity else None,
            score_vector=float(similarity) if similarity else None,
            rank_vector=int(rank),
            rank=int(rank),
            in_vector=True,
            document_id=document_id,
        ))
    
    # Get total count above threshold (before any cap)
    total_count = len(results)
    cap_applied = max_hits is not None and total_count >= max_hits
    
    # If capped, get actual total for metadata
    if cap_applied:
        count_sql = f"""
            SELECT COUNT(*) FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE {where_sql}
              AND (c.embedding {vector_config.operator} %(embedding)s::vector) <= %(max_distance)s
        """
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total_count = cur.fetchone()[0]
    
    metadata = ThresholdSearchMetadata(
        total_vector=total_count,
        total_combined=len(results),
        total_before_cap=total_count,
        cap_applied=cap_applied,
        cap_value=max_hits if cap_applied else None,
        threshold_used=threshold,
        vector_config=vector_config.to_dict(),
        mode="conversational" if max_hits else "thorough",
    )
    
    return results, metadata


def hybrid_rrf_sql(
    conn,
    query: str,
    embedding: List[float],
    *,
    similarity_threshold: float = 0.3,
    combine_mode: str = "union",  # "union" or "intersection"
    max_hits: Optional[int] = None,
    scope_sql: Optional[str] = None,
    scope_params: Optional[List[Any]] = None,
    vector_config: VectorMetricConfig = DEFAULT_VECTOR_CONFIG,
    filters: Optional[SearchFilters] = None,
    rrf_k: int = 60,
    lex_limit: int = 1000,  # Max lexical results to consider
    vec_limit: int = 1000,  # Max vector results to consider
    retrieval_mode: str = "conversational",
) -> Tuple[List[ThresholdSearchResult], ThresholdSearchMetadata]:
    """
    Hybrid search with RRF ranking using SQL CTE (default implementation).
    
    This is the PRIMARY RRF implementation per the plan. It's:
    - Deterministic (same query = same results)
    - Faster (no large set pulls into Python)
    - Memory-efficient for large result sets
    
    Args:
        conn: Database connection
        query: Search query text (for lexical search)
        embedding: Query embedding (for vector search)
        similarity_threshold: Minimum vector similarity
        combine_mode: "union" (any match) or "intersection" (both required)
        max_hits: Optional cap on results
        scope_sql: WHERE clause from scope primitives
        scope_params: Parameters for scope SQL
        vector_config: Metric configuration
        filters: Additional SearchFilters
        rrf_k: RRF constant (60 is standard)
        lex_limit: Max lexical candidates
        vec_limit: Max vector candidates
        retrieval_mode: "conversational" or "thorough"
        
    Returns:
        (results, metadata) tuple
    """
    # Convert threshold to distance
    max_distance = vector_config.transform_to_distance(similarity_threshold)
    
    # Convert embedding to vector literal for PostgreSQL
    embedding_literal = vector_literal(embedding)
    
    # Build tsquery from query
    # Use simple tokenization for now - could use compile_primitives_to_tsquery for more complex queries
    tsq = " & ".join(query.lower().split())
    if not tsq:
        tsq = query.lower().replace(" ", " & ") if query else ""
    
    # Build base WHERE clause
    params: Dict[str, Any] = {
        "embedding": embedding_literal,
        "max_distance": max_distance,
        "threshold": similarity_threshold,
        "tsq": tsq,
        "rrf_k": rrf_k,
        "lex_limit": lex_limit,
        "vec_limit": vec_limit,
        "not_found_rank": 100000,
    }
    
    where_clauses = ["TRUE"]
    
    # Add filter clauses if provided
    if filters:
        filter_params: Dict[str, Any] = {}
        filter_sql = _build_where(filters, filter_params)
        if filter_sql and filter_sql != "TRUE":
            where_clauses.append(filter_sql)
            params.update(filter_params)
    
    # Add scope SQL if provided
    if scope_sql:
        processed_scope_sql = scope_sql
        if scope_params:
            # Convert list params to named params
            for i, p in enumerate(scope_params):
                params[f"scope_p{i}"] = p
                processed_scope_sql = processed_scope_sql.replace("%s", f"%(scope_p{i})s", 1)
        where_clauses.append(f"({processed_scope_sql})")
    
    where_sql = " AND ".join(where_clauses)
    
    # Determine join type based on combine_mode
    join_type = "INNER JOIN" if combine_mode == "intersection" else "FULL OUTER JOIN"
    
    # Build the SQL CTE query
    sql = f"""
    WITH lex_ranked AS (
        SELECT 
            c.id AS chunk_id,
            ts_rank_cd(
                to_tsvector('simple', COALESCE(c.clean_text, c.text)),
                to_tsquery('simple', %(tsq)s)
            ) AS score,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(
                    to_tsvector('simple', COALESCE(c.clean_text, c.text)),
                    to_tsquery('simple', %(tsq)s)
                ) DESC, c.id ASC
            ) AS rank
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE {where_sql}
          AND to_tsvector('simple', COALESCE(c.clean_text, c.text)) @@ to_tsquery('simple', %(tsq)s)
        LIMIT %(lex_limit)s
    ),
    vec_ranked AS (
        SELECT 
            c.id AS chunk_id,
            1.0 - (c.embedding {vector_config.operator} %(embedding)s::vector) AS similarity,
            c.embedding {vector_config.operator} %(embedding)s::vector AS distance,
            ROW_NUMBER() OVER (
                ORDER BY c.embedding {vector_config.operator} %(embedding)s::vector ASC, c.id ASC
            ) AS rank
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE {where_sql}
          AND c.embedding IS NOT NULL
          AND (c.embedding {vector_config.operator} %(embedding)s::vector) <= %(max_distance)s
        LIMIT %(vec_limit)s
    ),
    combined AS (
        SELECT 
            COALESCE(l.chunk_id, v.chunk_id) AS chunk_id,
            l.rank AS rank_lex,
            l.score AS score_lex,
            v.rank AS rank_vec,
            v.similarity AS score_vec,
            v.distance AS vector_distance,
            1.0 / (%(rrf_k)s + COALESCE(l.rank, %(not_found_rank)s)) + 
            1.0 / (%(rrf_k)s + COALESCE(v.rank, %(not_found_rank)s)) AS rrf_score,
            l.chunk_id IS NOT NULL AS in_lexical,
            v.chunk_id IS NOT NULL AS in_vector
        FROM lex_ranked l
        {join_type} vec_ranked v ON l.chunk_id = v.chunk_id
    ),
    ranked AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (ORDER BY rrf_score DESC, chunk_id ASC) AS final_rank
        FROM combined
    )
    SELECT 
        r.chunk_id,
        r.rank_lex,
        r.score_lex,
        r.rank_vec,
        r.score_vec,
        r.vector_distance,
        r.rrf_score,
        r.in_lexical,
        r.in_vector,
        r.final_rank,
        cm.document_id
    FROM ranked r
    JOIN chunk_metadata cm ON cm.chunk_id = r.chunk_id
    ORDER BY r.final_rank
    """
    
    # Add limit if max_hits specified
    if max_hits:
        sql += f" LIMIT {max_hits}"
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    # Build results
    results = []
    for row in rows:
        (chunk_id, rank_lex, score_lex, rank_vec, score_vec, 
         vector_distance, rrf_score, in_lexical, in_vector, 
         final_rank, document_id) = row
        
        results.append(ThresholdSearchResult(
            chunk_id=chunk_id,
            score_lexical=float(score_lex) if score_lex else None,
            score_vector=float(score_vec) if score_vec else None,
            score_hybrid=float(rrf_score) if rrf_score else None,
            vector_distance=float(vector_distance) if vector_distance else None,
            vector_similarity=float(score_vec) if score_vec else None,
            rank_lexical=int(rank_lex) if rank_lex else None,
            rank_vector=int(rank_vec) if rank_vec else None,
            rank=int(final_rank),
            in_lexical=bool(in_lexical),
            in_vector=bool(in_vector),
            document_id=document_id,
        ))
    
    # Get totals for metadata
    total_combined = len(results)
    cap_applied = max_hits is not None and total_combined >= max_hits
    
    # If capped, we need to get actual total
    total_before_cap = total_combined
    if cap_applied:
        count_sql = f"""
        WITH lex_ranked AS (
            SELECT c.id AS chunk_id
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE {where_sql}
              AND to_tsvector('simple', COALESCE(c.clean_text, c.text)) @@ to_tsquery('simple', %(tsq)s)
            LIMIT %(lex_limit)s
        ),
        vec_ranked AS (
            SELECT c.id AS chunk_id
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
            WHERE {where_sql}
              AND c.embedding IS NOT NULL
              AND (c.embedding {vector_config.operator} %(embedding)s::vector) <= %(max_distance)s
            LIMIT %(vec_limit)s
        )
        SELECT COUNT(*) FROM (
            SELECT chunk_id FROM lex_ranked
            {"INTERSECT" if combine_mode == "intersection" else "UNION"}
            SELECT chunk_id FROM vec_ranked
        ) combined
        """
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total_before_cap = cur.fetchone()[0]
    
    metadata = ThresholdSearchMetadata(
        total_lexical=sum(1 for r in results if r.in_lexical),
        total_vector=sum(1 for r in results if r.in_vector),
        total_combined=total_combined,
        total_before_cap=total_before_cap,
        cap_applied=cap_applied,
        cap_value=max_hits if cap_applied else None,
        threshold_used=similarity_threshold,
        vector_config=vector_config.to_dict(),
        mode=retrieval_mode,
    )
    
    return results, metadata


def compute_thorough_rank(results: List[ThresholdSearchResult]) -> List[ThresholdSearchResult]:
    """
    Recompute ranks for thorough mode using deterministic (document_id, chunk_id) ordering.
    
    Thorough mode ignores scores and uses stable ordering based on document structure.
    Chunks without document_id sort LAST using sentinel value.
    
    Args:
        results: List of search results
        
    Returns:
        Results with updated rank field
    """
    SENTINEL = 2147483647  # Max int for NULL document_id
    
    # Sort by (document_id, chunk_id) - NULL document_id sorts last
    sorted_results = sorted(
        results,
        key=lambda r: (r.document_id if r.document_id else SENTINEL, r.chunk_id)
    )
    
    # Assign new ranks
    for i, result in enumerate(sorted_results, 1):
        result.rank = i
    
    return sorted_results


# =============================================================================
# V7 Phase 2: Chunk Neighbors / Continuation
# =============================================================================

@dataclass
class ChunkNeighbor:
    """A neighboring chunk with position relative to the seed chunk."""
    chunk_id: int
    text: str
    position: int  # Negative = before seed, positive = after seed, 0 = seed itself
    document_id: Optional[int] = None
    page_id: Optional[int] = None
    collection_slug: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "position": self.position,
            "document_id": self.document_id,
            "page_id": self.page_id,
            "collection_slug": self.collection_slug,
        }


def get_chunk_neighbors(
    conn,
    chunk_id: int,
    before: int = 2,
    after: int = 2,
    include_seed: bool = True,
) -> List[ChunkNeighbor]:
    """
    V7 Phase 2: Get neighboring chunks from the same document.
    
    This enables the agent to expand context around a relevant chunk,
    supporting "continuation" and "context expansion" patterns.
    
    Args:
        conn: Database connection
        chunk_id: The seed chunk ID
        before: Number of chunks to retrieve before the seed
        after: Number of chunks to retrieve after the seed
        include_seed: Whether to include the seed chunk in results
        
    Returns:
        List of ChunkNeighbor objects in document order (before → seed → after)
        
    Algorithm:
        1. Find the document_id of the seed chunk
        2. Find chunks in the same document, ordered by page_id and chunk_id
        3. Return the window around the seed chunk
    """
    
    if before < 0 or after < 0:
        raise ValueError("before and after must be non-negative")
    
    neighbors: List[ChunkNeighbor] = []
    
    try:
        with conn.cursor() as cur:
            # First, get the seed chunk's document_id and ordering info
            cur.execute("""
                SELECT cm.document_id, cm.first_page_id, c.id
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE c.id = %s
            """, (chunk_id,))
            
            seed_row = cur.fetchone()
            if not seed_row:
                return []  # Seed chunk not found
            
            document_id, seed_page_id, _ = seed_row
            
            if not document_id:
                # No document_id - can only return the seed chunk itself
                if include_seed:
                    cur.execute("""
                        SELECT c.id, COALESCE(c.clean_text, c.text), cm.document_id, 
                               cm.first_page_id, cm.collection_slug
                        FROM chunks c
                        LEFT JOIN chunk_metadata cm ON cm.chunk_id = c.id
                        WHERE c.id = %s
                    """, (chunk_id,))
                    row = cur.fetchone()
                    if row:
                        neighbors.append(ChunkNeighbor(
                            chunk_id=row[0],
                            text=row[1] or "",
                            position=0,
                            document_id=row[2],
                            page_id=row[3],
                            collection_slug=row[4],
                        ))
                return neighbors
            
            # Get all chunks from the same document, ordered by page_id and chunk_id
            cur.execute("""
                SELECT c.id, COALESCE(c.clean_text, c.text), cm.document_id,
                       cm.first_page_id, cm.collection_slug
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE cm.document_id = %s
                ORDER BY cm.first_page_id NULLS LAST, c.id ASC
            """, (document_id,))
            
            doc_chunks = cur.fetchall()
            
            # Find the index of the seed chunk
            seed_index = None
            for i, row in enumerate(doc_chunks):
                if row[0] == chunk_id:
                    seed_index = i
                    break
            
            if seed_index is None:
                return []  # Seed chunk not found in document (shouldn't happen)
            
            # Calculate window bounds
            start_index = max(0, seed_index - before)
            end_index = min(len(doc_chunks), seed_index + after + 1)
            
            # Build neighbor list with positions relative to seed
            for i in range(start_index, end_index):
                row = doc_chunks[i]
                position = i - seed_index
                
                # Skip seed if not including it
                if position == 0 and not include_seed:
                    continue
                
                neighbors.append(ChunkNeighbor(
                    chunk_id=row[0],
                    text=row[1] or "",
                    position=position,
                    document_id=row[2],
                    page_id=row[3],
                    collection_slug=row[4],
                ))
            
            return neighbors
            
    except Exception as e:
        # Rollback on error
        try:
            conn.rollback()
        except:
            pass
        raise


def get_document_chunks(
    conn,
    document_id: int,
    page_id: Optional[int] = None,
    limit: int = 50,
) -> List[ChunkNeighbor]:
    """
    V7 Phase 2: Get all chunks from a document, optionally filtered by page.
    
    Useful when the agent wants to read a full document or page.
    
    Args:
        conn: Database connection
        document_id: The document to retrieve chunks from
        page_id: Optional page_id to filter to a specific page
        limit: Maximum number of chunks to return
        
    Returns:
        List of ChunkNeighbor objects in document order
    """
    
    chunks: List[ChunkNeighbor] = []
    
    try:
        with conn.cursor() as cur:
            if page_id is not None:
                cur.execute("""
                    SELECT c.id, COALESCE(c.clean_text, c.text), cm.document_id,
                           cm.first_page_id, cm.collection_slug
                    FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE cm.document_id = %s AND cm.first_page_id = %s
                    ORDER BY c.id ASC
                    LIMIT %s
                """, (document_id, page_id, limit))
            else:
                cur.execute("""
                    SELECT c.id, COALESCE(c.clean_text, c.text), cm.document_id,
                           cm.first_page_id, cm.collection_slug
                    FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE cm.document_id = %s
                    ORDER BY cm.first_page_id NULLS LAST, c.id ASC
                    LIMIT %s
                """, (document_id, limit))
            
            rows = cur.fetchall()
            
            for i, row in enumerate(rows):
                chunks.append(ChunkNeighbor(
                    chunk_id=row[0],
                    text=row[1] or "",
                    position=i,  # Position in document order
                    document_id=row[2],
                    page_id=row[3],
                    collection_slug=row[4],
                ))
            
            return chunks
            
    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        raise
