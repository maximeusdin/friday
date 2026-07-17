"""
V9 Session & Evidence Set Management.

Handles:
- Session lifecycle (create, update active pointers)
- Run lifecycle (create, pause, resume, complete)
- Evidence set management (create, populate, adjacency expand, cap/prune)
- Step persistence wrapper
- Resume state serialization
- Concordance-aware evidence search (bidirectional alias expansion)
"""
import hashlib
import json
import logging
import re as _re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from retrieval.agent.v9_types import (
    ResearchWorkspace,
    V9Result,
    WorkspaceChunk,
    ScopeFilter,
    AliasHypothesis,
)


# =============================================================================
# Data models (Python-side mirrors of DB tables)
# =============================================================================

@dataclass
class SessionState:
    """Mirror of research_sessions row (v9 fields)."""
    session_id: int
    label: str = ""
    active_run_id: Optional[int] = None
    active_evidence_set_id: Optional[int] = None
    active_run_status: str = "idle"  # idle|running|paused|completed|failed
    scope_json: Dict[str, Any] = field(default_factory=lambda: {"mode": "full_archive"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "label": self.label,
            "active_run_id": self.active_run_id,
            "active_evidence_set_id": self.active_evidence_set_id,
            "active_run_status": self.active_run_status,
            "scope_json": self.scope_json,
        }


@dataclass
class RunRecord:
    """Mirror of v9_runs row."""
    run_id: int
    session_id: int
    query_text: str
    query_index: int = 0
    label: Optional[str] = None
    mode: str = "new_retrieval"     # new_retrieval | think_deeper
    status: str = "running"         # running | paused | completed | failed
    last_step_idx: int = 0
    budgets_json: Dict[str, Any] = field(default_factory=dict)
    resume_state_json: Optional[Dict[str, Any]] = None
    evidence_set_id: Optional[int] = None
    evidence_summary: Optional[str] = None
    top_entities_json: Optional[List[Dict[str, Any]]] = None
    run_scope_json: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "query_text": self.query_text,
            "query_index": self.query_index,
            "label": self.label,
            "mode": self.mode,
            "status": self.status,
            "last_step_idx": self.last_step_idx,
            "evidence_set_id": self.evidence_set_id,
            "evidence_summary": self.evidence_summary,
            "run_scope_json": self.run_scope_json,
        }


@dataclass
class EvidenceItem:
    """Mirror of evidence_items row."""
    item_id: int = 0
    evidence_set_id: int = 0
    chunk_id: int = 0
    quote_text: Optional[str] = None
    locators_json: Dict[str, Any] = field(default_factory=dict)
    retrieval_score: Optional[float] = None
    rank: Optional[int] = None
    source_step_idx: Optional[int] = None
    is_adjacency: bool = False
    dedup_hash: str = ""


@dataclass
class RunStep:
    """Mirror of v9_run_steps row."""
    step_id: int = 0
    run_id: int = 0
    step_idx: int = 0
    lane: Optional[str] = None
    tool_name: str = ""
    tool_args_json: Dict[str, Any] = field(default_factory=dict)
    tool_result_refs_json: Optional[Dict[str, Any]] = None
    elapsed_ms: Optional[float] = None


@dataclass
class RecentQueryContext:
    """Context for the router: recent runs in this session."""
    runs: List[RunRecord] = field(default_factory=list)
    active_run_id: Optional[int] = None
    active_evidence_set_id: Optional[int] = None
    active_run_status: str = "idle"


# =============================================================================
# Constants
# =============================================================================

EVIDENCE_SET_CAP = 200          # max items per evidence set
ADJACENCY_BEFORE = 1            # chunks before each evidence chunk
ADJACENCY_AFTER = 1             # chunks after each evidence chunk


# =============================================================================
# Dedup hash
# =============================================================================

def _compute_dedup_hash(evidence_set_id: int, chunk_id: int) -> str:
    """Deterministic hash for dedup: sha1(evidence_set_id:chunk_id)[:16]."""
    raw = f"{evidence_set_id}:{chunk_id}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# =============================================================================
# Session management (DB operations)
# =============================================================================

def load_session(conn, session_id: int) -> Optional[SessionState]:
    """Load session state from DB."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'research_sessions' AND column_name = 'scope_json'
        """)
        has_scope_json = cur.fetchone() is not None

        if has_scope_json:
            cur.execute("""
                SELECT id, label, active_run_id, active_evidence_set_id, active_run_status,
                       scope_json
                FROM research_sessions WHERE id = %s
            """, (session_id,))
        else:
            cur.execute("""
                SELECT id, label, active_run_id, active_evidence_set_id, active_run_status
                FROM research_sessions WHERE id = %s
            """, (session_id,))
        row = cur.fetchone()
        if not row:
            return None
        scope = row[5] if has_scope_json else {"mode": "full_archive"}
        if isinstance(scope, str):
            try:
                scope = json.loads(scope)
            except Exception:
                scope = {"mode": "full_archive"}
        elif scope is None:
            scope = {"mode": "full_archive"}
        return SessionState(
            session_id=row[0], label=row[1] or "",
            active_run_id=row[2], active_evidence_set_id=row[3],
            active_run_status=row[4] or "idle",
            scope_json=scope,
        )


def create_session(conn, label: str) -> SessionState:
    """Create a new session."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO research_sessions (label, active_run_status, updated_at)
            VALUES (%s, 'idle', now())
            RETURNING id
        """, (label,))
        sid = cur.fetchone()[0]
        conn.commit()
    return SessionState(session_id=sid, label=label)


def update_session_active(
    conn,
    session_id: int,
    *,
    active_run_id: Optional[int] = None,
    active_evidence_set_id: Optional[int] = None,
    active_run_status: Optional[str] = None,
) -> None:
    """Update the active pointers on a session."""
    sets = ["updated_at = now()"]
    params: List[Any] = []
    if active_run_id is not None:
        sets.append("active_run_id = %s")
        params.append(active_run_id)
    if active_evidence_set_id is not None:
        sets.append("active_evidence_set_id = %s")
        params.append(active_evidence_set_id)
    if active_run_status is not None:
        sets.append("active_run_status = %s")
        params.append(active_run_status)
    params.append(session_id)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE research_sessions SET {', '.join(sets)} WHERE id = %s",
            params,
        )
        conn.commit()


def update_session_scope(conn, session_id: int, scope_json: Dict[str, Any]) -> None:
    """Update the user-selected scope on a session."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'research_sessions' AND column_name = 'scope_json'
        """)
        if cur.fetchone() is None:
            return  # Column does not exist; skip update
        cur.execute("""
            UPDATE research_sessions
            SET scope_json = %s, updated_at = now()
            WHERE id = %s
        """, (json.dumps(scope_json), session_id))
        conn.commit()


def session_scope_to_filter(conn, scope_json: Dict[str, Any]) -> ScopeFilter:
    """Convert stored session scope to a ScopeFilter for retrieval.

    If mode is full_archive, returns an empty filter (no restriction).
    If mode is custom, resolves collection IDs to slugs and includes document IDs.
    """
    return _scope_json_to_filter(conn, scope_json)


def run_scope_to_filter(conn, run_scope_json: Optional[Dict[str, Any]]) -> ScopeFilter:
    """Convert run_scope_json (from v9_runs) to ScopeFilter for Think Deeper.

    Uses same structure as session scope: mode, included_collection_ids,
    included_document_ids. Returns empty filter if None or full_archive.
    """
    if not run_scope_json:
        return ScopeFilter()
    return _scope_json_to_filter(conn, run_scope_json)


def _scope_json_to_filter(conn, scope_json: Dict[str, Any]) -> ScopeFilter:
    """Convert scope JSON (session or run) to ScopeFilter."""
    if scope_json.get("mode") == "full_archive":
        return ScopeFilter()

    collection_ids = scope_json.get("included_collection_ids") or []
    slugs = _resolve_collection_slugs(conn, collection_ids) if collection_ids else None
    doc_ids = scope_json.get("included_document_ids") or []
    date_from = scope_json.get("filters", {}).get("date_from") if isinstance(scope_json.get("filters"), dict) else None
    date_to = scope_json.get("filters", {}).get("date_to") if isinstance(scope_json.get("filters"), dict) else None
    return ScopeFilter(
        collections=slugs or None,
        document_ids=doc_ids or None,
        date_from=date_from,
        date_to=date_to,
    )


def _resolve_collection_slugs(conn, collection_ids: List[int]) -> Optional[List[str]]:
    """Resolve collection IDs to slugs."""
    if not collection_ids:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug FROM collections WHERE id = ANY(%s)",
            (collection_ids,),
        )
        slugs = [r[0] for r in cur.fetchall()]
    return slugs if slugs else None


def normalize_scope(scope_json: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize scope for override comparison.

    IMPORTANT: only extracts scope-identity fields (mode, doc_ids, collection_ids).
    Ignores 'filters', 'expansion', 'source', 'reason' -- those are run metadata,
    not scope identity. This is critical because run_scope_json contains all of
    those sub-objects but session scope_json does not.
    """
    mode = scope_json.get("mode", "full_archive")
    if mode == "full_archive":
        return {"mode": "full_archive"}
    doc_ids = sorted(scope_json.get("included_document_ids") or [])
    if doc_ids:
        # document_ids wins; ignore collections
        return {"mode": "custom", "included_document_ids": doc_ids}
    col_ids = sorted(scope_json.get("included_collection_ids") or [])
    return {"mode": "custom", "included_collection_ids": col_ids}


# =============================================================================
# Run management
# =============================================================================

def create_run(
    conn,
    session_id: int,
    query_text: str,
    *,
    mode: str = "new_retrieval",
    budgets: Optional[Dict[str, Any]] = None,
    run_scope_json: Optional[Dict[str, Any]] = None,
) -> RunRecord:
    """Create a new run and its evidence set. Returns RunRecord."""
    with conn.cursor() as cur:
        # Get next query_index
        cur.execute(
            "SELECT COALESCE(MAX(query_index), -1) + 1 FROM v9_runs WHERE session_id = %s",
            (session_id,),
        )
        query_index = cur.fetchone()[0]

        # Create evidence set
        cur.execute("""
            INSERT INTO evidence_sets (session_id, is_active)
            VALUES (%s, TRUE) RETURNING id
        """, (session_id,))
        evidence_set_id = cur.fetchone()[0]

        # Create run
        budgets_json = json.dumps(budgets or {"max_tool_calls": 10, "max_model_turns": 10})
        run_scope_str = json.dumps(run_scope_json) if run_scope_json else None
        cur.execute("""
            INSERT INTO v9_runs
                (session_id, query_text, query_index, mode, status,
                 budgets_json, evidence_set_id, run_scope_json)
            VALUES (%s, %s, %s, %s, 'running', %s, %s, %s)
            RETURNING id
        """, (session_id, query_text, query_index, mode, budgets_json,
              evidence_set_id, run_scope_str))
        run_id = cur.fetchone()[0]

        # Link evidence set to run
        cur.execute(
            "UPDATE evidence_sets SET run_id = %s WHERE id = %s",
            (run_id, evidence_set_id),
        )

        # Update session active pointers
        cur.execute("""
            UPDATE research_sessions
            SET active_run_id = %s, active_evidence_set_id = %s,
                active_run_status = 'running', updated_at = now()
            WHERE id = %s
        """, (run_id, evidence_set_id, session_id))

        conn.commit()

    return RunRecord(
        run_id=run_id, session_id=session_id,
        query_text=query_text, query_index=query_index,
        mode=mode, status="running",
        budgets_json=budgets or {"max_tool_calls": 10, "max_model_turns": 10},
        evidence_set_id=evidence_set_id,
        run_scope_json=run_scope_json,
    )


def load_run(conn, run_id: int) -> Optional[RunRecord]:
    """Load a run from DB."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, session_id, query_text, query_index, label, mode, status,
                   last_step_idx, budgets_json, resume_state_json,
                   evidence_set_id, evidence_summary, top_entities_json,
                   run_scope_json
            FROM v9_runs WHERE id = %s
        """, (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        # Parse run_scope_json
        rscope = row[13]
        if isinstance(rscope, str):
            try:
                rscope = json.loads(rscope)
            except Exception:
                rscope = None
        return RunRecord(
            run_id=row[0], session_id=row[1], query_text=row[2],
            query_index=row[3], label=row[4], mode=row[5], status=row[6],
            last_step_idx=row[7],
            budgets_json=row[8] if isinstance(row[8], dict) else json.loads(row[8] or "{}"),
            resume_state_json=row[9] if isinstance(row[9], dict) else (json.loads(row[9]) if row[9] else None),
            evidence_set_id=row[10],
            evidence_summary=row[11],
            top_entities_json=row[12] if isinstance(row[12], list) else (json.loads(row[12]) if row[12] else None),
            run_scope_json=rscope if isinstance(rscope, dict) else None,
        )


def update_run_status(
    conn,
    run_id: int,
    status: str,
    *,
    last_step_idx: Optional[int] = None,
    resume_state_json: Optional[Dict[str, Any]] = None,
    evidence_summary: Optional[str] = None,
    top_entities_json: Optional[List[Dict[str, Any]]] = None,
    label: Optional[str] = None,
) -> None:
    """Update run status and optional fields."""
    sets = ["status = %s", "updated_at = now()"]
    params: List[Any] = [status]
    if last_step_idx is not None:
        sets.append("last_step_idx = %s")
        params.append(last_step_idx)
    if resume_state_json is not None:
        sets.append("resume_state_json = %s")
        params.append(json.dumps(resume_state_json))
    if evidence_summary is not None:
        sets.append("evidence_summary = %s")
        params.append(evidence_summary)
    if top_entities_json is not None:
        sets.append("top_entities_json = %s")
        params.append(json.dumps(top_entities_json))
    if label is not None:
        sets.append("label = %s")
        params.append(label)
    params.append(run_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE v9_runs SET {', '.join(sets)} WHERE id = %s", params)
        conn.commit()


def update_run_scope_json(conn, run_id: int, run_scope_json: Dict[str, Any]) -> None:
    """Update run_scope_json independently of run status.

    Dedicated function to avoid coupling scope/expansion metadata updates
    with status transitions. Called after Stage 1.5 expansion.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE v9_runs SET run_scope_json = %s, updated_at = now() WHERE id = %s",
            (json.dumps(run_scope_json), run_id),
        )
        conn.commit()


def load_recent_runs(conn, session_id: int, limit: int = 5) -> RecentQueryContext:
    """Load recent runs for the router."""
    session = load_session(conn, session_id)
    if not session:
        return RecentQueryContext()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, session_id, query_text, query_index, label, mode, status,
                   last_step_idx, budgets_json, resume_state_json,
                   evidence_set_id, evidence_summary, top_entities_json
            FROM v9_runs
            WHERE session_id = %s
            ORDER BY query_index DESC
            LIMIT %s
        """, (session_id, limit))
        rows = cur.fetchall()

    runs = []
    for row in rows:
        runs.append(RunRecord(
            run_id=row[0], session_id=row[1], query_text=row[2],
            query_index=row[3], label=row[4], mode=row[5], status=row[6],
            last_step_idx=row[7],
            budgets_json=row[8] if isinstance(row[8], dict) else json.loads(row[8] or "{}"),
            resume_state_json=row[9] if isinstance(row[9], dict) else (json.loads(row[9]) if row[9] else None),
            evidence_set_id=row[10],
            evidence_summary=row[11],
            top_entities_json=row[12] if isinstance(row[12], list) else (json.loads(row[12]) if row[12] else None),
        ))

    return RecentQueryContext(
        runs=runs,
        active_run_id=session.active_run_id,
        active_evidence_set_id=session.active_evidence_set_id,
        active_run_status=session.active_run_status,
    )


# =============================================================================
# Evidence set management
# =============================================================================

def add_evidence_items(
    conn,
    evidence_set_id: int,
    chunks: List[WorkspaceChunk],
    *,
    step_idx: int = 0,
    scores: Optional[Dict[int, float]] = None,
) -> int:
    """Add chunks to an evidence set (with dedup). Returns count added."""
    if not chunks:
        return 0
    scores = scores or {}
    added = 0
    with conn.cursor() as cur:
        for rank_offset, c in enumerate(chunks):
            dedup = _compute_dedup_hash(evidence_set_id, c.chunk_id)
            locators = {
                "doc_id": c.doc_id,
                "page": c.page,
                "source_label": c.source_label,
                "collection_slug": c.collection_slug or c.source_label,
            }
            try:
                cur.execute("""
                    INSERT INTO evidence_items
                        (evidence_set_id, chunk_id, quote_text, locators_json,
                         retrieval_score, rank, source_step_idx, is_adjacency, dedup_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (evidence_set_id, dedup_hash) DO NOTHING
                """, (
                    evidence_set_id, c.chunk_id,
                    c.text[:500] if c.text else None,
                    json.dumps(locators),
                    scores.get(c.chunk_id),
                    rank_offset,
                    step_idx,
                    c.is_neighbor,
                    dedup,
                ))
                if cur.rowcount > 0:
                    added += 1
            except Exception:
                pass  # dedup collision, skip
        conn.commit()
    return added


def add_adjacency_chunks(
    conn,
    evidence_set_id: int,
    chunk_ids: List[int],
    step_idx: int = 0,
) -> int:
    """Expand evidence set with adjacent chunks (before/after each chunk).

    Uses get_chunk_neighbors from retrieval/ops.py which does proper
    document-aware ordering (page_id + chunk_id) instead of fragile
    ID arithmetic. This ensures the correct previous/next chunk is
    selected even when chunk IDs are non-contiguous across documents.

    Returns count of new items added.
    """
    if not chunk_ids:
        return 0

    from retrieval.ops import get_chunk_neighbors

    added = 0
    for cid in chunk_ids:
        try:
            neighbors = get_chunk_neighbors(
                conn, cid,
                before=ADJACENCY_BEFORE,
                after=ADJACENCY_AFTER,
                include_seed=False,
            )
        except Exception:
            continue  # neighbor fetch is best-effort

        with conn.cursor() as cur:
            for nbr in neighbors:
                neighbor_id = nbr.chunk_id
                dedup = _compute_dedup_hash(evidence_set_id, neighbor_id)
                locators = {
                    "doc_id": nbr.document_id,
                    "page": f"p{nbr.page_id}" if nbr.page_id else None,
                    "source_label": nbr.collection_slug,
                    "collection_slug": nbr.collection_slug,
                }
                try:
                    cur.execute("""
                        INSERT INTO evidence_items
                            (evidence_set_id, chunk_id, quote_text, locators_json,
                             retrieval_score, rank, source_step_idx, is_adjacency, dedup_hash)
                        VALUES (%s, %s, %s, %s, NULL, NULL, %s, TRUE, %s)
                        ON CONFLICT (evidence_set_id, dedup_hash) DO NOTHING
                    """, (
                        evidence_set_id, neighbor_id,
                        (nbr.text or "")[:500],
                        json.dumps(locators),
                        step_idx,
                        dedup,
                    ))
                    if cur.rowcount > 0:
                        added += 1
                except Exception:
                    pass  # dedup collision or DB error, skip
            conn.commit()
    return added


def prune_evidence_set(conn, evidence_set_id: int, cap: int = EVIDENCE_SET_CAP) -> int:
    """Prune evidence set to cap, keeping best rank + newest. Returns count removed."""
    with conn.cursor() as cur:
        # Count current items
        cur.execute("SELECT COUNT(*) FROM evidence_items WHERE evidence_set_id = %s", (evidence_set_id,))
        total = cur.fetchone()[0]
        if total <= cap:
            return 0

        # Delete oldest/lowest-rank items beyond cap
        # Keep: non-adjacency first, then by rank ASC, then by created_at DESC
        cur.execute("""
            DELETE FROM evidence_items
            WHERE id IN (
                SELECT id FROM evidence_items
                WHERE evidence_set_id = %s
                ORDER BY
                    is_adjacency ASC,        -- keep non-adjacency first
                    COALESCE(rank, 99999),    -- keep best rank
                    created_at DESC           -- keep newest
                OFFSET %s
            )
        """, (evidence_set_id, cap))
        removed = cur.rowcount
        conn.commit()
        return removed


def get_evidence_set_size(conn, evidence_set_id: int) -> int:
    """Get the number of items in an evidence set."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM evidence_items WHERE evidence_set_id = %s", (evidence_set_id,))
        return cur.fetchone()[0]


def get_evidence_set_document_count(conn, evidence_set_id: int) -> int:
    """Get the number of distinct documents in an evidence set."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(DISTINCT (locators_json->>'doc_id'))
            FROM evidence_items
            WHERE evidence_set_id = %s
              AND locators_json->>'doc_id' IS NOT NULL
        """, (evidence_set_id,))
        return cur.fetchone()[0] or 0


# =============================================================================
# Follow-up evidence-only search (with bidirectional concordance expansion)
# =============================================================================

def _expand_query_for_evidence_search(conn, query: str) -> List[str]:
    """Expand a follow-up query with concordance aliases in BOTH directions.

    Returns a list of query variants for FTS:
      - Original query
      - Canonical names for any alias terms in the query (alias→name)
      - Alias forms for any canonical names in the query (name→alias)

    This ensures follow-up searches within an evidence set match chunks
    regardless of which name form they use.

    Example: query "PAL" returns ["PAL", "Nathan Gregory Silvermaster"]
    Example: query "Silvermaster" returns ["Silvermaster", "PAL", "Robert"]
    """
    variants = [query.strip()]
    if not query or not query.strip():
        return variants

    words = _re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,24}", query)
    if not words:
        return variants

    seen_lower = {query.strip().lower()}

    for word in words:
        if len(word) < 2:
            continue
        try:
            with conn.cursor() as cur:
                # Direction 1: alias→canonical (word is an alias, find canonical)
                cur.execute("""
                    SELECT DISTINCT e.canonical_name
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE LOWER(ea.alias) = LOWER(%s)
                    LIMIT 3
                """, (word,))
                for row in cur.fetchall():
                    canonical = (row[0] or "").strip()
                    if canonical and canonical.lower() not in seen_lower:
                        variants.append(canonical)
                        seen_lower.add(canonical.lower())

                # Also try alias_norm for normalized match
                word_norm = _re.sub(r"[^a-z0-9 ]", "", word.lower()).strip()
                if word_norm:
                    cur.execute("""
                        SELECT DISTINCT e.canonical_name
                        FROM entities e
                        JOIN entity_aliases ea ON ea.entity_id = e.id
                        WHERE ea.alias_norm = %s
                        LIMIT 3
                    """, (word_norm,))
                    for row in cur.fetchall():
                        canonical = (row[0] or "").strip()
                        if canonical and canonical.lower() not in seen_lower:
                            variants.append(canonical)
                            seen_lower.add(canonical.lower())

                # Direction 2: canonical→aliases (word is a canonical name, find aliases)
                cur.execute("""
                    SELECT ea.alias
                    FROM entities e
                    JOIN entity_aliases ea ON ea.entity_id = e.id
                    WHERE LOWER(e.canonical_name) = LOWER(%s)
                    LIMIT 8
                """, (word,))
                for row in cur.fetchall():
                    alias = (row[0] or "").strip()
                    if alias and len(alias) >= 2 and alias.lower() not in seen_lower:
                        variants.append(alias)
                        seen_lower.add(alias.lower())
        except Exception as e:
            logging.getLogger(__name__).warning("_expand_query_for_evidence_search DB lookup failed: %s", e)
            try:
                conn.rollback()
            except Exception:
                pass

    return variants[:10]  # cap at 10 variants


def search_evidence_set(
    conn,
    evidence_set_id: int,
    query: str,
    *,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search within an evidence set using FTS on chunks.tsv.

    Applies bidirectional concordance expansion so that:
      - alias queries match chunks containing canonical names
      - canonical name queries match chunks containing aliases

    Returns evidence items with full chunk text. No retrieval tools called.
    """
    if not query or not query.strip():
        # Return top-ranked items if no query
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ei.id, ei.chunk_id, ei.quote_text, ei.locators_json,
                       ei.retrieval_score, ei.rank, ei.is_adjacency,
                       c.text AS chunk_text
                FROM evidence_items ei
                JOIN chunks c ON c.id = ei.chunk_id
                WHERE ei.evidence_set_id = %s
                ORDER BY COALESCE(ei.rank, 99999), ei.created_at
                LIMIT %s
            """, (evidence_set_id, limit))
            return _rows_to_dicts(cur)

    # Expand query with concordance aliases (both directions)
    query_variants = _expand_query_for_evidence_search(conn, query)

    # Build a combined tsquery using OR across all variants
    # This ensures "PAL" also matches chunks mentioning "Silvermaster" and vice versa
    if len(query_variants) == 1:
        # Simple case: single query
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ei.id, ei.chunk_id, ei.quote_text, ei.locators_json,
                       ei.retrieval_score, ei.rank, ei.is_adjacency,
                       c.text AS chunk_text,
                       ts_rank_cd(c.tsv, plainto_tsquery('english', %s)) AS fts_score
                FROM evidence_items ei
                JOIN chunks c ON c.id = ei.chunk_id
                WHERE ei.evidence_set_id = %s
                  AND c.tsv @@ plainto_tsquery('english', %s)
                ORDER BY fts_score DESC
                LIMIT %s
            """, (query, evidence_set_id, query, limit))
            return _rows_to_dicts(cur)
    else:
        # Expanded case: combine variants with OR for broad matching
        # Build: plainto_tsquery(v1) || plainto_tsquery(v2) || ...
        tsquery_parts = []
        params: List[Any] = []
        for i, variant in enumerate(query_variants):
            tsquery_parts.append(f"plainto_tsquery('english', %s)")
            params.append(variant)

        combined_tsquery = " || ".join(tsquery_parts)

        # The ranking uses the original query for relevance
        sql = f"""
            SELECT ei.id, ei.chunk_id, ei.quote_text, ei.locators_json,
                   ei.retrieval_score, ei.rank, ei.is_adjacency,
                   c.text AS chunk_text,
                   ts_rank_cd(c.tsv, {combined_tsquery}) AS fts_score
            FROM evidence_items ei
            JOIN chunks c ON c.id = ei.chunk_id
            WHERE ei.evidence_set_id = %s
              AND c.tsv @@ ({combined_tsquery})
            ORDER BY fts_score DESC
            LIMIT %s
        """
        # params: [variant1, variant2, ..., variantN]  (for ranking tsquery)
        #       + [variant1, variant2, ..., variantN]  (for WHERE filter tsquery)
        #       + [evidence_set_id, limit]
        all_params = params + [evidence_set_id] + params + [limit]

        with conn.cursor() as cur:
            cur.execute(sql, all_params)
            results = _rows_to_dicts(cur)

        # If expanded query found nothing, fall back to ILIKE on original query
        # as a safety net (handles trigram/partial matches FTS misses)
        if not results:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ei.id, ei.chunk_id, ei.quote_text, ei.locators_json,
                           ei.retrieval_score, ei.rank, ei.is_adjacency,
                           c.text AS chunk_text,
                           0.0 AS fts_score
                    FROM evidence_items ei
                    JOIN chunks c ON c.id = ei.chunk_id
                    WHERE ei.evidence_set_id = %s
                      AND c.text ILIKE %s
                    ORDER BY COALESCE(ei.rank, 99999)
                    LIMIT %s
                """, (evidence_set_id, f"%{query.strip()}%", limit))
                results = _rows_to_dicts(cur)

        return results


def _rows_to_dicts(cur) -> List[Dict[str, Any]]:
    """Convert cursor results to list of dicts."""
    cols = [desc[0] for desc in cur.description] if cur.description else []
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# =============================================================================
# Step persistence wrapper
# =============================================================================

def persist_step(
    conn,
    run_id: int,
    step_idx: int,
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    lane: Optional[str] = None,
    result_refs: Optional[Dict[str, Any]] = None,
    elapsed_ms: Optional[float] = None,
) -> None:
    """Persist a single tool step to the DB."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO v9_run_steps
                (run_id, step_idx, lane, tool_name, tool_args_json,
                 tool_result_refs_json, elapsed_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, step_idx) DO UPDATE SET
                tool_result_refs_json = EXCLUDED.tool_result_refs_json,
                elapsed_ms = EXCLUDED.elapsed_ms
        """, (
            run_id, step_idx, lane, tool_name,
            json.dumps(tool_args),
            json.dumps(result_refs) if result_refs else None,
            elapsed_ms,
        ))
        # Update run last_step_idx
        cur.execute(
            "UPDATE v9_runs SET last_step_idx = %s, updated_at = now() WHERE id = %s",
            (step_idx, run_id),
        )
        conn.commit()


# =============================================================================
# Resume state serialization
# =============================================================================

def save_resume_state(
    conn,
    run_id: int,
    state: Dict[str, Any],
) -> None:
    """Save controller resume state to the run."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE v9_runs SET resume_state_json = %s, updated_at = now()
            WHERE id = %s
        """, (json.dumps(state), run_id))
        conn.commit()


def build_resume_state(
    workspace: ResearchWorkspace,
    tool_calls_executed: int,
    model_turns: int,
    step_idx: int,
    max_tool_calls: int,
) -> Dict[str, Any]:
    """Build a minimal resume state dict from current controller state.

    This state is sufficient to resume from pause:
    - Budgets used so far (tool_calls, model_turns)
    - Position (step_idx)
    - Investigation state (goal, gaps, hypotheses, leads, next_actions)
    - Entity state (resolved entities + candidates)
    - Seen chunk IDs (for dedup in resumed search)
    - Evidence memory counts
    """
    # Serialize entities for rehydration
    entities = [
        {
            "entity_id": e.entity_id,
            "canonical_name": e.canonical_name,
            "aliases": e.aliases[:8],
            "entity_type": e.entity_type,
        }
        for e in workspace.entities[:20]
    ]
    candidates = [
        c.to_dict() for c in workspace.entity_candidates[:20]
    ]

    return {
        "tool_calls_executed": tool_calls_executed,
        "model_turns": model_turns,
        "step_idx": step_idx,
        "max_tool_calls": max_tool_calls,
        "seen_chunk_ids": sorted(workspace._summarized_chunk_ids),
        "catalog_count": len(workspace.catalog_hits),
        "fulltext_count": len(workspace.fulltext_chunks),
        "entity_count": len(workspace.entities),
        "evidence_memory_updates": len(workspace.evidence_memory),
        # Investigation state for resume
        "investigation_goal": workspace.investigation.goal,
        "investigation_gaps": workspace.investigation.gaps,
        "investigation_hypotheses": workspace.investigation.hypotheses,
        "investigation_leads": workspace.investigation.leads,
        "investigation_next_actions": workspace.investigation.next_actions,
        # Entity state for rehydration
        "entities": entities,
        "entity_candidates": candidates,
        # Search history
        "search_queries": list(workspace._search_queries[-10:]),
        # Alias hypotheses for resume
        "alias_hypotheses": [h.to_dict() for h in workspace.alias_hypotheses.values()],
    }


def rehydrate_workspace_from_evidence(
    conn,
    evidence_set_id: int,
    question: str,
    resume_state: Dict[str, Any],
    scope: Optional[ScopeFilter] = None,
) -> ResearchWorkspace:
    """Rehydrate a workspace from a saved evidence set + resume state.

    Loads evidence items, rebuilds workspace chunks, restores entity state,
    and re-populates investigation state so the runner can continue from
    where it left off.
    """
    workspace = ResearchWorkspace(
        question=question,
        scope=scope or ScopeFilter(),
    )

    # Restore investigation state
    inv = workspace.investigation
    inv.goal = resume_state.get("investigation_goal", "")
    inv.gaps = resume_state.get("investigation_gaps", [])
    inv.hypotheses = resume_state.get("investigation_hypotheses", [])
    inv.leads = resume_state.get("investigation_leads", [])
    inv.next_actions = resume_state.get("investigation_next_actions", [])

    # Restore seen chunk IDs
    workspace._summarized_chunk_ids = set(resume_state.get("seen_chunk_ids", []))
    workspace._search_queries = list(resume_state.get("search_queries", []))

    # Restore alias hypotheses
    for h_data in resume_state.get("alias_hypotheses", []):
        h = AliasHypothesis.from_dict(h_data)
        key = (h.alias_text.strip().lower(), h.entity_id)
        workspace.alias_hypotheses[key] = h

    # Restore entities
    from retrieval.agent.v9_types import WorkspaceEntity, EntityCandidate
    for e_data in resume_state.get("entities", []):
        workspace.entities.append(WorkspaceEntity(
            entity_id=e_data["entity_id"],
            canonical_name=e_data.get("canonical_name", ""),
            aliases=e_data.get("aliases", []),
            entity_type=e_data.get("entity_type"),
        ))
    for c_data in resume_state.get("entity_candidates", []):
        workspace.entity_candidates.append(EntityCandidate(
            query_term=c_data.get("query_term", ""),
            entity_id=c_data.get("entity_id", 0),
            canonical_name=c_data.get("canonical_name", ""),
            entity_type=c_data.get("entity_type"),
            matched_via=c_data.get("matched_via", ""),
            accepted=c_data.get("accepted", False),
            confidence=c_data.get("confidence", "exact"),
            ambiguous=c_data.get("ambiguous", False),
        ))

    # Load evidence items as workspace chunks
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ei.chunk_id, c.text, ei.locators_json, ei.is_adjacency,
                   ei.retrieval_score
            FROM evidence_items ei
            JOIN chunks c ON c.id = ei.chunk_id
            WHERE ei.evidence_set_id = %s
            ORDER BY COALESCE(ei.rank, 99999)
        """, (evidence_set_id,))
        rows = cur.fetchall()

    import uuid
    from retrieval.agent.v9_types import (
        WorkspaceChunk,
        CatalogHit,
        EvidenceBullet,
        EvidenceSummaryUpdate,
        compute_bullet_id,
    )

    # Build synthetic evidence_memory from chunks so FindingStore can seed.
    # Think Deeper requires evidence_memory to promote claims; without it,
    # FindingStore seeds empty and we get "No grounded summary available".
    bullets_for_seed: list = []
    for row in rows:
        cid, text, locators, is_adj, score = row
        if isinstance(locators, str):
            try:
                locators = json.loads(locators)
            except Exception:
                locators = {}
        elif locators is None:
            locators = {}

        chunk = WorkspaceChunk(
            chunk_id=cid,
            text=text or "",
            doc_id=locators.get("doc_id"),
            page=locators.get("page"),
            source_label=locators.get("source_label"),
            collection_slug=locators.get("collection_slug") or locators.get("source_label"),
            score=score,
            is_neighbor=bool(is_adj),
        )
        workspace.fulltext_chunks.append(chunk)

        # Also add to catalog for context
        workspace.catalog_hits.append(CatalogHit(
            chunk_id=cid,
            score=score or 0.0,
            doc_id=locators.get("doc_id"),
            page=locators.get("page"),
            collection=locators.get("source_label"),
            snippet=(text or "")[:300],
        ))

        # Synthetic bullet for FindingStore seeding (Think Deeper requires evidence_memory)
        snippet = (text or "").strip()[:220]
        if snippet and cid:
            bid = compute_bullet_id(snippet, [cid])
            if bid:
                bullets_for_seed.append(EvidenceBullet(
                    bullet_id=bid,
                    text=snippet,
                    supporting_chunk_ids=[cid],
                    doc_ids=[locators.get("doc_id")] if locators.get("doc_id") else [],
                ))

    if bullets_for_seed:
        # Cap bullets to avoid overwhelming FindingStore (seed_from_evidence_summary uses all)
        bullets_for_seed = bullets_for_seed[:30]
        workspace.evidence_memory.append(EvidenceSummaryUpdate(
            update_id=str(uuid.uuid4()),
            generated_from_chunk_ids=[b.supporting_chunk_ids[0] for b in bullets_for_seed if b.supporting_chunk_ids],
            summarizer_model="rehydrate",
            created_at=datetime.now(timezone.utc).isoformat(),
            bullets=bullets_for_seed,
        ))

    return workspace


# =============================================================================
# Evidence summary generation (post-completion)
# =============================================================================

def generate_evidence_summary(workspace: ResearchWorkspace) -> str:
    """Generate a short evidence summary from workspace state."""
    parts = []
    if workspace.investigation.goal:
        parts.append(f"Goal: {workspace.investigation.goal}")

    # Top findings from evidence memory
    if workspace._bullet_index:
        top_bullets = []
        for bid in workspace.pinned_bullet_ids[:3]:
            b = workspace._bullet_index.get(bid)
            if b:
                top_bullets.append(b.text[:100])
        if top_bullets:
            parts.append("Key findings: " + "; ".join(top_bullets))

    parts.append(f"Evidence: {len(workspace.fulltext_chunks)} chunks, "
                 f"{len(workspace.entities)} entities")
    return " | ".join(parts)[:500]


def extract_top_entities(workspace: ResearchWorkspace) -> List[Dict[str, Any]]:
    """Extract top entities for query reference resolution."""
    entities = []
    for e in workspace.entities[:10]:
        entities.append({
            "entity_id": e.entity_id,
            "canonical_name": e.canonical_name,
            "aliases": e.aliases[:5],
        })
    return entities
