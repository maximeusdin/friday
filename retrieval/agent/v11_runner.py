"""
V11 Runner — stripped-down V9 (no PEM lane, no query entity resolution).

Same engine as V9: agent selects actions, investigates with tools, maintains
alias map and evidence memory. No PEM-driven query planning, no alias expansion
in search, no priming from question.

Lightweight PEM: optional evidence-time mention index (use_lightweight_pem=True)
to annotate venona/vassiliev chunks with surface→canonical mappings for A/B testing.
"""
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from retrieval.agent.v11_types import (
    V11ResearchWorkspace,
    V9Synthesis,
    V9Claim,
    V9Result,
    GroundedClaim,
    WorkspaceChunk,
    WorkspaceEntity,
    EntityCandidate,
    CatalogHit,
    SufficiencyCheck,
    ScopeFilter,
    InvestigationState,
    InvestigationStep,
    WorkspaceDelta,
)
from retrieval.agent.v9_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    V9_MODEL,
    V9_MAX_WORKSPACE_CHUNKS,
    V9_MAX_TOOL_CALLS,
)
from retrieval.agent.v11_context import build_context_pack
from retrieval.agent.v9_context import _estimate_tokens
from retrieval.agent.v9_workspace import (
    merge_search_result,
    merge_fetched_chunks,
    merge_catalog_hits,
    merge_entities,
    merge_entity_candidates,
    compute_progress_signal,
    append_note,
    apply_pin_suggestions,
    merge_evidence_summary_update,
    build_chunk_doc_map,
    link_chunks_to_entities,
    build_alias_context_for_summarizer,
)
from retrieval.agent.v11_tools import (
    search_chunks,
    search_broad,
    resolve_codenames,
    fetch_chunks,
    fetch_diverse_from_catalog,
    expand_query,
    expand_from_evidence,
)
from retrieval.agent.tools import ToolResult
from retrieval.agent.v9_summarize import summarize_delta_chunks
from retrieval.agent.v9_grounding import ground_claims, ground_roster_entries
from retrieval.agent.v9_verify import build_verification_report

# Reuse V9 runner utilities
from retrieval.agent.v9_runner import (
    detect_scope,
    strip_scope_syntax,
    detect_scope_override_and_filters,
    _snapshot_counts,
    _compute_delta,
    _trim_messages,
    _call_with_retry,
    _parse_content,
    _build_artifact_dict,
    _build_minimal_synthesis_context,
    _build_needs_more_evidence_synthesis,
    _update_investigation,
    _validate_finalization,
    V9_OUTPUT_SCHEMA,
    V9_RESPONSE_FORMAT,
    MAX_HISTORY_MESSAGES,
    MAX_MODEL_TURNS,
    TOOL_TURN_MAX_TOKENS,
    SYNTHESIS_MAX_TOKENS,
)

# Tool execution uses _load_catalog from v9
from retrieval.agent.v9_tools import _load_catalog
from retrieval.agent.v9_dispatch import _extract_expansion_entities

# V11: search (default), search_canonical, search_lexical, expand_query, expand_from_evidence, fetch_chunks
V11_TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Default search: hybrid (semantic + lexical) with neutral embeddings. "
                "Use first for broad coverage. Returns a catalog of hits. MUST call fetch_chunks to read full text.\n\n"
                "For topics with aliases/codenames (Silvermaster, Hiss, Rosenberg), follow with search_canonical — "
                "V/V depth tool; use after coverage scan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (natural-language text)"},
                    "top_k": {"type": "integer", "description": "Max catalog hits (default 50)", "default": 50},
                    "collections": {"type": "array", "items": {"type": "string"}, "description": "Optional collection filter. Respect scope if set by system."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_canonical",
            "description": (
                "V/V-only canonical (augmented) search. Uses alias-resolved embeddings. "
                "Pass a single codename token (PAL, Silvermaster), NOT the full question. "
                "Prefer resolve_codenames(terms=[...]) for multiple tokens — it batches lexical + canonical."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. Silvermaster network)"},
                    "top_k": {"type": "integer", "description": "Max catalog hits (default 50)", "default": 50},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_lexical",
            "description": (
                "Lexical exact substring match. Pass SINGLE name or codename (e.g. PAL, Silvermaster). "
                "For codename resolution in V/V, prefer resolve_codenames(terms=[...])."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Single name or codename (e.g. PAL, Silvermaster)"},
                    "top_k": {"type": "integer", "description": "Max catalog hits (default 50)", "default": 50},
                    "collections": {"type": "array", "items": {"type": "string"}, "description": "Optional collection filter."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_codenames",
            "description": (
                "Resolve codename tokens in Venona/Vassiliev. Pass individual terms (PAL, LIBERAL, Silvermaster), "
                "NOT the full question. Uses lexical_exact + canonical (augmented) search per term. "
                "Returns mapping candidates with supporting chunk_ids. Best UX for codename resolution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {"type": "array", "items": {"type": "string"}, "description": "Codename tokens to resolve (e.g. [\"PAL\", \"LIBERAL\", \"Silvermaster\"])"},
                    "top_k_per_term": {"type": "integer", "description": "Max hits per term (default 20)", "default": 20},
                },
                "required": ["terms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_broad",
            "description": (
                "Broad scan: returns top N per collection (default 2). "
                "Use for discovery. After search_broad, use fetch_diverse — do NOT fetch 30 chunks from one collection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_per_collection": {"type": "integer", "description": "Max hits per collection (default 2)", "default": 2},
                    "top_k": {"type": "integer", "description": "Total hits to fetch before grouping (default 400)", "default": 400},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_query",
            "description": "Expand query terms via concordance. Returns expanded terms. Then call search or search_canonical with expanded query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query to expand via concordance"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_from_evidence",
            "description": "Extract names/codenames from retrieved chunks, concordance-expand. Pass chunk_ids from your catalog. Returns expanded terms for a follow-up search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "integer"}, "description": "Chunk IDs from catalog/fulltext to extract entities from"},
                },
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_chunks",
            "description": (
                "Load full text and metadata for evidence. Use after search.\n\n"
                "IMPORTANT: Use exactly ONE of these three calling modes:\n"
                "  Mode 1 - By chunk_ids: {\"chunk_ids\": [1,2,3]}  (neighbors +/-1 auto-included)\n"
                "  Mode 2 - Doc slice:    {\"doc_id\": 50, \"around_chunk_id\": 999, \"window\": 6}\n"
                "  Mode 3 - Doc pages:    {\"doc_id\": 50, \"page_start\": 5, \"page_end\": 7}\n\n"
                "Do NOT mix modes (e.g. do not pass chunk_ids together with doc_id).\n\n"
                "Returns full-text WorkspaceChunks with doc/page/collection metadata. "
                "Auto-fetched neighbor chunks are marked is_neighbor=true.\n\n"
                "Typical workflow: search -> review snippets -> fetch_chunks(chunk_ids=[...]) "
                "to load the most promising hits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "integer"}, "description": "Mode 1: Chunk IDs to load (neighbors +/-1 auto-included)"},
                    "doc_id": {"type": "integer", "description": "Mode 2/3: Document ID"},
                    "around_chunk_id": {"type": "integer", "description": "Mode 2: Center chunk ID within the document"},
                    "window": {"type": "integer", "description": "Mode 2: Total chunks to read around center (default 4)", "default": 4},
                    "page_start": {"type": "integer", "description": "Mode 3: Start page number (inclusive)"},
                    "page_end": {"type": "integer", "description": "Mode 3: End page number (inclusive)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_diverse",
            "description": (
                "Fetch chunks from catalog with collection diversity. "
                "Use after search_broad — do NOT fetch 30 chunks from one collection. "
                "Selects up to per_collection from each collection (default 3), round-robin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "total": {"type": "integer", "description": "Max chunks to fetch (default 20)", "default": 20},
                    "per_collection": {"type": "integer", "description": "Max per collection (default 3)", "default": 3},
                },
            },
        },
    },
]


def _extract_and_merge_entities_from_chunks(
    conn, workspace: V11ResearchWorkspace, chunk_ids: List[int]
) -> int:
    """Extract entities from chunk_ids (via entity_mentions) and merge into workspace.
    Returns count of new entities merged."""
    if not chunk_ids:
        return 0
    try:
        entities_data = _extract_expansion_entities(conn, chunk_ids[:100])
        if not entities_data:
            return 0
        prev = len(workspace.entities)
        to_merge = [
            WorkspaceEntity(
                entity_id=e["id"],
                canonical_name=e.get("canonical_name", ""),
                aliases=e.get("aliases", []),
            )
            for e in entities_data
            if e.get("id") and e.get("canonical_name")
        ]
        merge_entities(workspace, to_merge)
        return max(0, len(workspace.entities) - prev)
    except Exception as e:
        print(f"  [V11] Entity extraction failed: {e}", file=sys.stderr)
        return 0


def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    conn,
    workspace: V11ResearchWorkspace,
    progress_callback: Optional[Any] = None,
) -> Tuple[Dict[str, Any], str]:
    """Execute one tool using V11 tools (no concordance expansion, no resolution)."""
    scope = workspace.scope

    try:
        conn.rollback()
    except Exception:
        pass

    try:
        if name in ("search", "search_canonical", "search_lexical", "search_chunks"):
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 50)
            collections = arguments.get("collections")

            if name == "search_lexical":
                mode = "lexical_exact"
                use_canonical = False
            elif name == "search_canonical":
                mode = "hybrid"
                use_canonical = True
                collections = ["venona", "vassiliev"]
            elif name == "search_chunks":
                mode = arguments.get("mode", "hybrid")
                use_canonical = False
            else:
                mode = "hybrid"
                use_canonical = False

            result, catalog = search_chunks(
                conn, query=query, top_k=top_k,
                collections=collections, scope=scope, mode=mode,
                use_canonical=use_canonical,
                expand_concordance=getattr(workspace, "_v13_enhanced", False),
            )
            merge_catalog_hits(workspace, catalog)
            if query and query.strip():
                workspace._search_queries.append(query.strip())

            if os.getenv("DIAG_SCOPE_BIAS", "").strip() in ("1", "true", "yes") or name == "search_canonical":
                coll_counts = result.metadata.get("collections_seen_in_top_k") or Counter(h.collection or "unknown" for h in catalog)
                idx = result.metadata.get("index_used", "?")
                print(
                    f"  [V11] Search hits by collection: {dict(coll_counts)}, index_used={idx}",
                    file=sys.stderr,
                )

            out = {
                "tool": name,
                "mode": mode,
                "count": len(catalog),
                "hits": [
                    {"chunk_id": h.chunk_id, "score": round(h.score, 3),
                     "doc_id": h.doc_id, "page": h.page,
                     "collection": h.collection, "snippet": h.snippet[:300]}
                    for h in catalog[:60]
                ],
                "success": result.success,
                "error": result.error,
            }
            if result.metadata.get("collections_seen_in_top_k"):
                out["collections_seen_in_top_k"] = result.metadata["collections_seen_in_top_k"]
            return out, f"query='{query[:60]}' {name} -> {len(catalog)} hits"

        if name == "search_broad":
            query = arguments.get("query", "")
            top_per_collection = arguments.get("top_per_collection", 2)
            top_k = arguments.get("top_k", 400)
            result = search_broad(
                conn, query=query,
                top_per_collection=top_per_collection, top_k=top_k,
                scope=scope,
            )
            flat_catalog = result.get("flat_catalog", [])
            merge_catalog_hits(workspace, flat_catalog)
            if query and query.strip():
                workspace._search_queries.append(query.strip())
            collections_out = result.get("collections", [])
            out = {
                "tool": "search_broad",
                "collections": collections_out,
                "total_hits": result.get("total_hits", 0),
                "success": True,
            }
            return out, f"search_broad '{query[:40]}' -> {len(flat_catalog)} hits across {len(collections_out)} collections"

        if name == "expand_query":
            query = arguments.get("query", "")
            terms, meta = expand_query(conn, query=query, scope=scope)
            out = {"tool": "expand_query", "expanded_terms": terms[:15], "metadata": meta}
            return out, f"expand_query '{query[:40]}' -> {len(terms)} terms"

        if name == "expand_from_evidence":
            chunk_ids = arguments.get("chunk_ids", [])
            terms, meta = expand_from_evidence(conn, chunk_ids=chunk_ids, scope=scope)
            for e_data in meta.get("entities", []):
                if e_data.get("id") and e_data.get("canonical_name"):
                    merge_entities(workspace, [
                        WorkspaceEntity(
                            entity_id=e_data["id"],
                            canonical_name=e_data["canonical_name"],
                            aliases=e_data.get("aliases", []),
                        )
                    ])
            out = {"tool": "expand_from_evidence", "expanded_terms": terms, "metadata": meta}
            return out, f"expand_from_evidence {len(chunk_ids)} chunks -> {len(terms)} terms"

        if name == "resolve_codenames":
            terms = arguments.get("terms", [])
            top_k_per_term = arguments.get("top_k_per_term", 20)
            if isinstance(terms, str):
                terms = [t.strip() for t in terms.split(",") if t.strip()]
            result_dict, catalog = resolve_codenames(
                conn, terms=terms, top_k_per_term=top_k_per_term, scope=scope,
            )
            merge_catalog_hits(workspace, catalog)
            for t in (x.strip() for x in terms if x):
                if t and t not in workspace._search_queries:
                    workspace._search_queries.append(t)
            out = {
                "tool": "resolve_codenames",
                "mappings": result_dict["mappings"],
                "catalog_preview": result_dict.get("catalog_preview", []),
                "total_chunk_ids": result_dict.get("total_chunk_ids", 0),
                "success": True,
            }
            return out, f"resolve_codenames {terms[:5]} -> {result_dict.get('total_chunk_ids', 0)} chunks"

        if name == "fetch_diverse":
            total = arguments.get("total", 20)
            per_collection = arguments.get("per_collection", 3)
            chunks = fetch_diverse_from_catalog(conn, workspace, total=total, per_collection=per_collection)
            if scope and scope.collections:
                scope_set = {s.lower() for s in scope.collections}
                chunks = [c for c in chunks if not c.source_label or c.source_label.lower() in scope_set]
            merge_fetched_chunks(workspace, chunks)
            chunk_ids = [c.chunk_id for c in chunks]
            _extract_and_merge_entities_from_chunks(conn, workspace, chunk_ids)
            link_chunks_to_entities(workspace, chunks)
            delta_cids = [c.chunk_id for c in chunks if c.chunk_id not in workspace._summarized_chunk_ids]
            summarizer_note = ""
            if delta_cids:
                delta_chunks_for_summary = [c for c in chunks if c.chunk_id in set(delta_cids)]
                alias_ctx = build_alias_context_for_summarizer(workspace)
                try:
                    ev_update = summarize_delta_chunks(
                        delta_chunks_for_summary, workspace.question,
                        alias_context=alias_ctx,
                    )
                    cdm = build_chunk_doc_map(workspace)
                    merge_evidence_summary_update(workspace, ev_update, cdm)
                    summarizer_note = f", summarized {len(delta_chunks_for_summary)} -> {len(ev_update.bullets)} bullets"
                    print(
                        f"  [V11] Summarizer: delta_chunks={len(delta_chunks_for_summary)}, "
                        f"out_bullets={len(ev_update.bullets)}",
                        file=sys.stderr,
                    )
                    if progress_callback and ev_update.bullets:
                        chunk_to_page = {c.chunk_id: _parse_page_no(c.page) for c in workspace.fulltext_chunks if c.page}
                        all_doc_ids = list({did for b in ev_update.bullets for did in (b.doc_ids or [])})
                        doc_names = _get_doc_source_names(conn, all_doc_ids)
                        _emit_progress(progress_callback, "evidence_update", "completed",
                            f"Discovered {len(ev_update.bullets)} new evidence bullets",
                            {
                                "bullets": [
                                    {
                                        "text": b.text,
                                        "tags": b.tags,
                                        "chunk_ids": b.supporting_chunk_ids,
                                        "doc_ids": b.doc_ids,
                                        "pages": [chunk_to_page.get(cid) for cid in b.supporting_chunk_ids],
                                        "source_names": [doc_names.get(did, "") for did in (b.doc_ids or [])],
                                    }
                                    for b in ev_update.bullets
                                ],
                                "open_questions": ev_update.open_questions,
                                "leads": ev_update.leads,
                                "total_bullet_count": len(workspace._bullet_index),
                            })
                except Exception as e:
                    print(f"  [V11] Summarizer error: {e}", file=sys.stderr)
            coll_counts = Counter(c.source_label or "unknown" for c in chunks)
            return (
                {"tool": "fetch_diverse", "fetched": len(chunks), "collections": dict(coll_counts), "success": True},
                f"fetch_diverse -> {len(chunks)} chunks from {len(coll_counts)} collections{summarizer_note}",
            )

        if name == "fetch_chunks":
            chunk_ids = arguments.get("chunk_ids")
            doc_id = arguments.get("doc_id")
            around_chunk_id = arguments.get("around_chunk_id")
            window = arguments.get("window", 4)
            page_start = arguments.get("page_start")
            page_end = arguments.get("page_end")

            if doc_id and not around_chunk_id and page_start is None and page_end is None:
                return {"error": "doc_id requires either around_chunk_id (mode 2) or page_start/page_end (mode 3)", "tool": name}, "error: bad mode"
            if not chunk_ids and not doc_id:
                return {"error": "Provide chunk_ids (mode 1), or doc_id + around_chunk_id (mode 2), or doc_id + page_start + page_end (mode 3)", "tool": name}, "error: no args"

            chunks = fetch_chunks(
                conn,
                chunk_ids=chunk_ids,
                doc_id=doc_id,
                around_chunk_id=around_chunk_id,
                window=window,
                page_start=page_start,
                page_end=page_end,
            )

            if scope and scope.collections:
                scope_set = {s.lower() for s in scope.collections}
                pre_count = len(chunks)
                chunks = [c for c in chunks if not c.source_label or c.source_label.lower() in scope_set]
                if len(chunks) < pre_count:
                    print(
                        f"  [V11] Scope filter: dropped {pre_count - len(chunks)} out-of-scope chunks",
                        file=sys.stderr,
                    )

            merge_fetched_chunks(workspace, chunks)
            fetch_chunk_ids = [c.chunk_id for c in chunks]
            _extract_and_merge_entities_from_chunks(conn, workspace, fetch_chunk_ids)
            link_chunks_to_entities(workspace, chunks)

            delta_cids = [c.chunk_id for c in chunks if c.chunk_id not in workspace._summarized_chunk_ids]
            summarizer_note = ""
            if delta_cids:
                delta_chunks_for_summary = [c for c in chunks if c.chunk_id in set(delta_cids)]
                alias_ctx = build_alias_context_for_summarizer(workspace)
                try:
                    ev_update = summarize_delta_chunks(
                        delta_chunks_for_summary, workspace.question,
                        alias_context=alias_ctx,
                    )
                    cdm = build_chunk_doc_map(workspace)
                    merge_evidence_summary_update(workspace, ev_update, cdm)
                    summarizer_note = f", summarized {len(delta_chunks_for_summary)} -> {len(ev_update.bullets)} bullets"
                    print(
                        f"  [V11] Summarizer: delta_chunks={len(delta_chunks_for_summary)}, "
                        f"out_bullets={len(ev_update.bullets)}",
                        file=sys.stderr,
                    )
                    # Emit evidence_update so frontend Investigation panel shows bullets (V9 behavior)
                    if progress_callback and ev_update.bullets:
                        chunk_to_page = {c.chunk_id: _parse_page_no(c.page) for c in workspace.fulltext_chunks if c.page}
                        all_doc_ids = list({did for b in ev_update.bullets for did in (b.doc_ids or [])})
                        doc_names = _get_doc_source_names(conn, all_doc_ids)
                        _emit_progress(progress_callback, "evidence_update", "completed",
                            f"Discovered {len(ev_update.bullets)} new evidence bullets",
                            {
                                "bullets": [
                                    {
                                        "text": b.text,
                                        "tags": b.tags,
                                        "chunk_ids": b.supporting_chunk_ids,
                                        "doc_ids": b.doc_ids,
                                        "pages": [chunk_to_page.get(cid) for cid in b.supporting_chunk_ids],
                                        "source_names": [doc_names.get(did, "") for did in (b.doc_ids or [])],
                                    }
                                    for b in ev_update.bullets
                                ],
                                "open_questions": ev_update.open_questions,
                                "leads": ev_update.leads,
                                "total_bullet_count": len(workspace._bullet_index),
                            })
                except Exception as e:
                    print(f"  [V11] Summarizer error: {e}", file=sys.stderr)

            out = {
                "tool": "fetch_chunks",
                "chunks": [
                    {"chunk_id": c.chunk_id, "text": c.text[:1500],
                     "source": c.source_label, "page": c.page, "is_neighbor": c.is_neighbor}
                    for c in chunks
                ],
                "count": len(chunks),
            }
            return out, f"fetched {len(chunks)} chunks{summarizer_note}"

        return {"error": f"Unknown tool: {name}"}, f"error: unknown tool {name}"
    except Exception as e:
        return {"error": str(e), "tool": name}, f"error: {str(e)[:80]}"


def _get_doc_source_names(conn, doc_ids: List[int]) -> Dict[int, str]:
    """Look up source_name for each doc_id. Returns {doc_id: source_name}."""
    if not doc_ids:
        return {}
    ids = [d for d in doc_ids if d]
    if not ids:
        return {}
    out: Dict[int, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source_name FROM documents WHERE id = ANY(%s)",
                (ids,),
            )
            for row in cur.fetchall():
                out[row[0]] = (row[1] or "").strip()
    except Exception:
        pass
    return out


def _parse_page_no(page: Optional[str]) -> Optional[int]:
    """Parse page string like 'p5' or '5' to int."""
    if not page:
        return None
    s = str(page).strip().lstrip("pP")
    try:
        return int(s) if s else None
    except ValueError:
        return None


def _emit_progress(callback, step: str, status: str, message: str, details: Optional[Dict] = None) -> None:
    if callback:
        try:
            callback(step, status, message, details or {})
        except Exception:
            pass


_SUGGEST_QUERY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "suggest_query_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "suggested_query": {
                    "type": "string",
                    "description": "Exactly one research question.",
                },
            },
            "required": ["suggested_query"],
            "additionalProperties": False,
        },
    },
}


def _suggest_better_query_heuristic(question: str) -> str:
    """Fallback when LLM suggestion fails."""
    q = (question or "").strip()
    if not q or len(q) < 3:
        return "documents mentioning [person or topic]"
    q_lower = q.lower()
    skip = {"what", "did", "do", "who", "is", "was", "were", "the", "a", "an", "in", "on", "at"}
    words = [w for w in q.replace("?", "").split() if w.lower() not in skip]
    if not words:
        return f'documents about "{q}"'
    topic = " ".join(words[:4])
    if "what" in q_lower or "did" in q_lower or "do" in q_lower:
        return f'documents mentioning {topic}'
    if "who" in q_lower:
        return f'{topic} in the archive'
    return f'documents about "{topic}"'


def _suggest_better_query(
    question: str,
    workspace: "V11ResearchWorkspace",
    conn,
    verbose: bool = True,
) -> str:
    """Suggest a clearer query using an LLM grounded to the archive.

    Returns exactly one question that is:
    - Relevant to the user's original query
    - Grounded to what exists in the archive (collections, search snippets)
    - A single, specific research question
    """
    q = (question or "").strip()
    if not q or len(q) < 2:
        return _suggest_better_query_heuristic(question)

    # Build archive context for grounding
    collection_names: List[str] = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT slug FROM collections ORDER BY slug")
            collection_names = [row[0] for row in cur.fetchall()]
    except Exception:
        pass

    collections_str = ", ".join(collection_names[:20]) if collection_names else "venona, vassiliev, and other historical collections"

    search_queries = getattr(workspace, "_search_queries", None) or []
    catalog_hits = getattr(workspace, "catalog_hits", None) or []
    snippets: List[str] = []
    collections_seen: set = set()
    for h in catalog_hits[:15]:
        coll = getattr(h, "collection", None) or getattr(h, "collection_slug", None)
        if coll:
            collections_seen.add(coll)
        snip = (getattr(h, "snippet", "") or "")[:200]
        if snip.strip():
            snippets.append(snip.strip())

    context_parts = [
        f"Archive: Historical intelligence documents (diplomatic cables, Soviet espionage files). "
        f"Collections: {collections_str}.",
        f"User's original query: {q}",
    ]
    if search_queries:
        context_parts.append(f"Searches run: {search_queries[:5]}")
    if snippets:
        context_parts.append("Sample snippets from archive (for grounding):")
        for i, s in enumerate(snippets[:10], 1):
            context_parts.append(f"  {i}. {s}...")
    elif collections_seen:
        context_parts.append(f"Searched collections: {', '.join(sorted(collections_seen)[:10])}")

    prompt = f"""You suggest a better research query for a historical archive search.

{chr(10).join(context_parts)}

Rules:
- Output exactly ONE question in suggested_query.
- The question must be directly relevant to the user's original query.
- The question must be answerable from the archive (names, documents, events in intelligence history).
- Do not ask multiple questions. No bullet points. No "or" alternatives.
- Keep it concise (under 80 chars). Example form: "What role did [name] play in [context]?" or "Documents mentioning [name] and [topic]."
"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _suggest_better_query_heuristic(question)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL_PLAN", "gpt-4.1-mini-2025-04-14")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Output JSON only. suggested_query must be a single research question."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=150,
            response_format=_SUGGEST_QUERY_RESPONSE_FORMAT,
        )
        content = resp.choices[0].message.content if resp.choices else None
        if not content or not content.strip():
            return _suggest_better_query_heuristic(question)
        raw = json.loads(content)
        suggested = (raw.get("suggested_query") or "").strip()
        # Enforce single question: take first sentence, remove multiple questions
        if "?" in suggested:
            suggested = suggested.split("?")[0].strip() + "?"
        else:
            suggested = suggested.rstrip(".") + "?" if suggested else ""
        if len(suggested) < 10 or len(suggested) > 200:
            return _suggest_better_query_heuristic(question)
        # Reject if obviously not relevant (no meaningful overlap with original)
        stop = {"what", "did", "do", "who", "is", "was", "were", "the", "a", "an", "in", "on", "at", "and"}
        orig_words = set(re.findall(r"[a-zA-Z]{2,}", q.lower())) - stop
        sugg_words = set(re.findall(r"[a-zA-Z]{2,}", suggested.lower()))
        if orig_words and not (orig_words & sugg_words):
            return _suggest_better_query_heuristic(question)
        return suggested
    except Exception as e:
        if verbose:
            print(f"  [V11] Query suggestion LLM failed: {e}", file=sys.stderr)
        return _suggest_better_query_heuristic(question)


def run_v11_query(
    conn,
    question: str,
    model: str = V9_MODEL,
    max_workspace_chunks: int = V9_MAX_WORKSPACE_CHUNKS,
    max_tool_calls: int = V9_MAX_TOOL_CALLS,
    scope: Optional[ScopeFilter] = None,
    verbose: bool = True,
    use_lightweight_pem: bool = False,
    dump_pem_light: bool = False,
    progress_callback: Optional[Any] = None,
    _resume_workspace: Optional[Any] = None,
    _findings_brief: Optional[str] = None,
    seed_entity_candidates: Optional[List[Any]] = None,
    clarification_notes: Optional[List[str]] = None,
    engine_profile: str = "v11",
) -> V9Result:
    """
    Run the V11 Investigation Loop (stripped V9).

    No query entity resolution, no PEM lane, no alias expansion in search.
    Optional: use_lightweight_pem appends [MENTION_INDEX] block for venona/vassiliev chunks.

    engine_profile: "v11" (default, unchanged) or "v13" — the latter adds query planning,
      agreement-priming of the workspace, concordance-expanded search, and an
      anti-false-negative guard. See retrieval/agent/v13_planner.py.

    _resume_workspace: when provided (e.g. from /deeper), reuse this workspace instead
    of creating a fresh one. Workspace must have fulltext_chunks, catalog_hits, entities, etc.
    """
    _v13 = (engine_profile == "v13")
    if _v13:
        # V13 always uses the lightweight [MENTION_INDEX] so synthesis can bridge
        # codenames->canonical (Sound->Golos, Harry->Rabinovich) in venona/vassiliev.
        use_lightweight_pem = True
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for V11")

    if progress_callback:
        progress_callback("context_build", "running", "Building context...", {})
    detected_scope = detect_scope(question)
    if scope:
        if scope.collections:
            detected_scope.collections = scope.collections
        if scope.date_from:
            detected_scope.date_from = scope.date_from
        if scope.date_to:
            detected_scope.date_to = scope.date_to

    clean_question = strip_scope_syntax(question)

    if _resume_workspace is not None:
        workspace = _resume_workspace
        workspace.scope = detected_scope
        if verbose:
            print(
                f"  [V11] Resuming with pre-populated workspace: "
                f"{len(workspace.fulltext_chunks)} chunks, {len(workspace.entities)} entities",
                file=sys.stderr,
            )
    else:
        workspace = V11ResearchWorkspace(question=clean_question, scope=detected_scope)
        workspace.investigation.goal = "Research the topic thoroughly and provide a comprehensive answer."

    # V13: plan the query + prime the workspace with agreement-ranked evidence before the loop.
    _v13_plan = None
    if _v13 and _resume_workspace is None:
        try:
            from retrieval.agent.v13_planner import plan_query, prime_workspace
            setattr(workspace, "_v13_enhanced", True)
            _v13_plan = plan_query(clean_question, verbose=verbose)
            if progress_callback:
                progress_callback("retrieval_prepare", "running", "Planning + priming search...", {})
            prime_workspace(
                conn, workspace, _v13_plan, detected_scope,
                verbose=verbose, progress_callback=progress_callback,
            )
        except Exception as _pe:
            if verbose:
                print(f"  [V13] priming failed (continuing as V11): {_pe}", file=sys.stderr)
            try:
                conn.rollback()
            except Exception:
                pass

    # V12 clarification seeds: pre-accept user-confirmed identities and surface the
    # user's clarifications so the investigation starts from resolved intent.
    if seed_entity_candidates:
        have = {(c.query_term, c.entity_id) for c in workspace.entity_candidates}
        for cand in seed_entity_candidates:
            if (cand.query_term, cand.entity_id) not in have:
                workspace.entity_candidates.append(cand)
    if clarification_notes:
        workspace.notes.append("User clarifications: " + " ".join(clarification_notes))
        workspace.investigation.goal += (
            " The user clarified intent: " + " ".join(clarification_notes)
        )

    client = OpenAI(api_key=api_key)

    if verbose and not detected_scope.is_empty():
        print(f"  [V11] Scope: {detected_scope.to_dict()}", file=sys.stderr)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    scope_note = ""
    if not detected_scope.is_empty():
        parts = []
        if detected_scope.collections:
            parts.append(f"collections={detected_scope.collections}")
        if detected_scope.date_from:
            parts.append(f"date_from={detected_scope.date_from}")
        if detected_scope.date_to:
            parts.append(f"date_to={detected_scope.date_to}")
        scope_note = f"[System scope filter applied: {', '.join(parts)}. All searches are filtered accordingly.]"

    tool_calls_executed = 0
    model_turns = 0
    done = False
    synthesis: Optional[V9Synthesis] = None
    prev_counts = _snapshot_counts(workspace)
    tools_called_this_turn: List[str] = []

    while not done and tool_calls_executed < max_tool_calls and model_turns < MAX_MODEL_TURNS:
        delta = _compute_delta(workspace, prev_counts, tools_called_this_turn)
        tools_called_this_turn = []

        _emit_progress(progress_callback, "turn_prepare", "running",
            f"Preparing turn {model_turns + 1}...",
            {"turn": model_turns + 1, "tool_calls_used": tool_calls_executed})
        ctx = build_context_pack(
            workspace, delta,
            conn=conn,
            use_lightweight_pem=use_lightweight_pem,
            dump_pem_light=dump_pem_light,
            findings_brief=_findings_brief,
        )
        user_content = USER_PROMPT_TEMPLATE.format(
            question=clean_question,
            scope_note=scope_note,
            context=ctx,
        )
        messages.append({"role": "user", "content": user_content})
        messages = _trim_messages(messages)

        inv = workspace.investigation
        gaps_empty = not inv.gaps or inv.gaps == [""] or inv.gaps == []
        is_last_turn = tool_calls_executed >= max_tool_calls
        if (inv.ready_to_synthesize and gaps_empty) or is_last_turn:
            max_ct = SYNTHESIS_MAX_TOKENS
        else:
            max_ct = TOOL_TURN_MAX_TOKENS

        _emit_progress(progress_callback, "turn_start", "running",
            f"Turn {model_turns + 1}: {tool_calls_executed}/{max_tool_calls} tools used",
            {"turn": model_turns + 1, "tool_calls_used": tool_calls_executed,
             "tool_calls_budget": max_tool_calls, "catalog_hits": len(workspace.catalog_hits),
             "fulltext_chunks": len(workspace.fulltext_chunks)})

        _emit_progress(progress_callback, "model_call", "running",
            "Analyzing evidence and planning next steps...",
            {"turn": model_turns + 1})

        if verbose:
            est = _estimate_tokens(user_content)
            total_hist = sum(_estimate_tokens(m.get("content", "") or "") for m in messages)
            print(
                f"  [V11] Turn {model_turns}: catalog={len(workspace.catalog_hits)}, "
                f"fulltext={len(workspace.fulltext_chunks)}, entities={len(workspace.entities)}, "
                f"tool_calls={tool_calls_executed}/{max_tool_calls}, hist~{total_hist}tok",
                file=sys.stderr,
            )

        try:
            response = _call_with_retry(
                client, model, messages, V11_TOOLS_DEF, max_ct,
                workspace, delta, clean_question, scope_note, verbose,
            )
            msg = response.choices[0].message
        except Exception as e:
            if verbose:
                print(f"  [V11] Model call error (timeout/rate-limit/connection): {e}", file=sys.stderr)
            synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)
            if not workspace._bullet_index:
                suggested = _suggest_better_query(clean_question, workspace, conn, verbose)
                synthesis = V9Synthesis(
                    final=True,
                    narrative=(
                        f"I couldn't complete the search due to a temporary error. "
                        f"Try again, or try: {suggested}"
                    ),
                    claims=[],
                    sufficiency=SufficiencyCheck(
                        sufficient=False,
                        argument="API error during synthesis.",
                        remaining_gaps=["Try again shortly"],
                    ),
                )
            synthesis.final = True
            done = True
            prev_counts = _snapshot_counts(workspace)
            continue

        model_turns += 1

        # V9-style flow: branch on tool_calls first (like V9)
        content = _parse_content(msg.content)
        if content and isinstance(content.get("scratchpad_update"), dict):
            _update_investigation(workspace.investigation, content["scratchpad_update"])
            apply_pin_suggestions(workspace, content["scratchpad_update"].get("pin_suggestions") or [])

        tool_calls = msg.tool_calls or []

        # Branch A: Tool calls present — execute and continue (content optional, like V9)
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })
            step_idx = len(workspace.investigation.trace)

            for tc in tool_calls:
                if tool_calls_executed >= max_tool_calls:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"error": "Tool budget exhausted"})})
                    continue

                name = tc.function.name if hasattr(tc.function, "name") else tc.function.get("name", "")
                args_str = tc.function.arguments if hasattr(tc.function, "arguments") else tc.function.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}

                tool_calls_executed += 1
                # Progress: emit before executing so UI shows context for each step
                _brief_args = {k: (str(v)[:80] if isinstance(v, str) else v) for k, v in list(args.items())[:3]}
                _query = args.get("query", "")
                _chunk_ids = args.get("chunk_ids") or []
                if name in ("search", "search_canonical", "search_lexical", "search_chunks") and _query:
                    _msg = f"Searching for: {_query[:60]}{'...' if len(_query) > 60 else ''}"
                elif name == "fetch_chunks":
                    _n = len(_chunk_ids) if _chunk_ids else args.get("page_end") or "?"
                    _msg = f"Loading {_n} passages..."
                elif name == "fetch_diverse":
                    _msg = "Loading diverse passages across collections..."
                elif name == "resolve_codenames":
                    _terms = args.get("terms", [])
                    _msg = f"Resolving codenames: {_terms[:5]}"
                else:
                    _msg = f"{name}({', '.join(f'{k}={v}' for k, v in _brief_args.items())})"[:120]
                _emit_progress(progress_callback, "tool_call", "running",
                    _msg,
                    {"tool": name, "args": _brief_args, "tool_call_number": tool_calls_executed})
                out, summary = _execute_tool(name, args, conn, workspace, progress_callback)
                tools_called_this_turn.append(name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(out, default=str)[:16000],
                })

                step = InvestigationStep(
                    step_idx=step_idx,
                    action=name,
                    rationale=workspace.investigation.goal or "",
                    inputs=args,
                    outputs_summary=summary[:200],
                    added_catalog=max(0, len(workspace.catalog_hits) - prev_counts.get("catalog", 0)),
                    added_fulltext=max(0, len(workspace.fulltext_chunks) - prev_counts.get("fulltext", 0)),
                    added_entities=max(0, len(workspace.entities) - prev_counts.get("entities", 0)),
                )
                workspace.investigation.trace.append(step)
                append_note(workspace, f"[step {step_idx}] {name}: {summary[:120]}")
                step_idx += 1

                if verbose:
                    print(f"  [V11] Tool #{tool_calls_executed}: {name}: {summary}", file=sys.stderr)

            prev_counts = _snapshot_counts(workspace)

            # Budget exhausted -> request synthesis (like V9)
            if tool_calls_executed >= max_tool_calls:
                _emit_progress(progress_callback, "synthesis", "running",
                    "Synthesizing answer from evidence...", {"reason": "budget_exhausted"})
                if verbose:
                    print(f"  [V11] Tool budget exhausted ({tool_calls_executed}/{max_tool_calls}). Requesting synthesis.", file=sys.stderr)
                # Use minimal context to avoid truncation (full history can exceed 20k tokens)
                minimal_ctx = _build_minimal_synthesis_context(workspace, clean_question, scope_note)
                synth_prompt = "Tool budget exhausted. " + minimal_ctx
                system_msgs = [m for m in messages if m["role"] == "system"]
                synth_messages = system_msgs + [{"role": "user", "content": synth_prompt}]
                try:
                    synth_response = _call_with_retry(
                        client, model, synth_messages, None, SYNTHESIS_MAX_TOKENS,
                        workspace, delta, clean_question, scope_note, verbose,
                    )
                    synth_msg = synth_response.choices[0].message
                except Exception as e:
                    if verbose:
                        print(f"  [V11] Synthesis API error (timeout/rate-limit/connection): {e}", file=sys.stderr)
                    synth_msg = None
                model_turns += 1
                if synth_msg and synth_msg.content:
                    synth_data = _parse_content(synth_msg.content)
                    if synth_data:
                        synth_data["artifact"] = _build_artifact_dict(synth_data.get("artifact") or {})
                        synthesis = V9Synthesis.from_dict(synth_data)
                    else:
                        synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)
                    messages.append({"role": "assistant", "content": synth_msg.content})
                else:
                    synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)
                synthesis.final = True
                done = True
            continue

        # Branch B: No tool calls — need content for final synthesis (like V9)
        if not content:
            if verbose:
                _reason = "null" if msg.content is None else "empty" if not (msg.content or "").strip() else "invalid JSON (likely truncated)"
                print(f"  [V11] Structured output missing ({_reason}); retrying with minimal context", file=sys.stderr)
            minimal_ctx = _build_minimal_synthesis_context(workspace, clean_question, scope_note)
            system_msgs = [m for m in messages if m["role"] == "system"]
            retry_messages = system_msgs + [{"role": "user", "content": minimal_ctx}]
            try:
                response = _call_with_retry(
                    client, model, retry_messages, None, SYNTHESIS_MAX_TOKENS,
                    workspace, delta, clean_question, scope_note, verbose,
                )
                msg = response.choices[0].message
            except Exception as e:
                if verbose:
                    print(f"  [V11] Synthesis retry API error (timeout/rate-limit/connection): {e}", file=sys.stderr)
                msg = None
            model_turns += 1
            content = _parse_content(msg.content) if msg else None
            if content and isinstance(content.get("scratchpad_update"), dict):
                _update_investigation(workspace.investigation, content["scratchpad_update"])

        if not content:
            if verbose:
                print("  [V11] Structured output still missing after retry; building needs_more_evidence fallback", file=sys.stderr)
            messages.append({"role": "assistant", "content": msg.content if msg else ""})
            synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)
            if not workspace._bullet_index:
                suggested = _suggest_better_query(clean_question, workspace, conn, verbose)
                synthesis = V9Synthesis(
                    final=True,
                    narrative=(
                        f"I couldn't find enough evidence to answer that. "
                        f"Try a more specific search, for example: {suggested}"
                    ),
                    claims=[],
                    sufficiency=SufficiencyCheck(
                        sufficient=False,
                        argument="Structured output failed; no evidence found.",
                        remaining_gaps=["Try rephrasing with a clearer topic or scope"],
                    ),
                )
            synthesis.final = True
            done = True
            prev_counts = _snapshot_counts(workspace)
            continue

        messages.append({"role": "assistant", "content": msg.content or ""})

        if not content.get("final"):
            if verbose:
                print("  [V11] final=false without tool calls — nudging", file=sys.stderr)
            messages.append({
                "role": "user",
                "content": (
                    "You set final=false but did not call any tools. "
                    "Either call tools to continue investigating, or set final=true to synthesize."
                ),
            })
            prev_counts = _snapshot_counts(workspace)
            continue

        # Finalization
        content["artifact"] = _build_artifact_dict(content.get("artifact") or {})
        # Apply defaults for null sufficiency/responsiveness
        if content.get("final"):
            if not content.get("sufficiency"):
                content["sufficiency"] = {
                    "sufficient": False,
                    "argument": "Model did not populate; assuming insufficient.",
                    "remaining_gaps": ["Unknown"],
                    "next_best_actions_if_more_time": [],
                }
            resp = content.get("responsiveness")
            if not resp or not isinstance(resp, dict):
                content["responsiveness"] = {
                    "addressed_question": False,
                    "what_i_delivered": [],
                    "missing": ["Structured output incomplete"],
                    "why_missing": "Model did not populate responsiveness.",
                }
        synthesis = V9Synthesis.from_dict(content)
        valid, issues = _validate_finalization(
            synthesis, clean_question, tool_calls_executed, max_tool_calls, workspace=workspace
        )
        if not valid and issues:
            if verbose:
                print(f"  [V11] Finalization validation: {issues}", file=sys.stderr)
        done = True
        prev_counts = _snapshot_counts(workspace)

    if not synthesis and content and not content.get("final"):
        synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)
        if not workspace._bullet_index:
            suggested = _suggest_better_query(clean_question, workspace, conn, verbose)
            synthesis = V9Synthesis(
                final=True,
                narrative=(
                    f"I couldn't find enough evidence to answer that. "
                    f"Try a more specific search, for example: {suggested}"
                ),
                claims=[],
                sufficiency=SufficiencyCheck(
                    sufficient=False,
                    argument="No synthesis produced.",
                    remaining_gaps=["Try rephrasing with a clearer topic or scope"],
                ),
            )
        synthesis.final = True

    if not synthesis:
        suggested = _suggest_better_query(clean_question, workspace, conn, verbose)
        synthesis = V9Synthesis(
            final=True,
            narrative=(
                f"I couldn't find enough evidence to answer that. "
                f"Try a more specific search, for example: {suggested}"
            ),
            claims=[],
            sufficiency=SufficiencyCheck(
                sufficient=False,
                argument="No synthesis produced.",
                remaining_gaps=["Try rephrasing with a clearer topic or scope"],
            ),
        )

    _emit_progress(progress_callback, "synthesis", "running",
        "Synthesizing answer from evidence...", {"claims": len(synthesis.claims or [])})
    grounded_claims = ground_claims(synthesis.claims, workspace)
    grounded_roster = ground_roster_entries(synthesis.get_roster(), workspace)
    verification = build_verification_report(
        grounded_claims, synthesis=synthesis, grounded_roster=grounded_roster
    )

    result = V9Result(
        narrative=synthesis.narrative,
        claims=grounded_claims,
        grounded_roster=grounded_roster,
        verification=verification,
        sufficiency=synthesis.sufficiency,
        synthesis=synthesis,
        workspace=workspace,
        investigation_trace=workspace.investigation.trace,
    )

    # V13: never let "not retrieved" become a confident "no evidence exists".
    if _v13:
        try:
            from retrieval.agent.v13_planner import apply_anti_false_negative
            apply_anti_false_negative(result, workspace, verbose=verbose)
        except Exception as _ge:
            if verbose:
                print(f"  [V13] guard failed (non-fatal): {_ge}", file=sys.stderr)

    return result


def _parse_synthesis(content: dict) -> Optional[V9Synthesis]:
    """Parse synthesis from structured output."""
    try:
        return V9Synthesis.from_dict({
            "final": content.get("final", False),
            "narrative": content.get("narrative") or "",
            "claims": content.get("claims") or [],
            "sufficiency": content.get("sufficiency"),
            "responsiveness": content.get("responsiveness"),
            "artifact": content.get("artifact") or {},
        })
    except Exception:
        return None
