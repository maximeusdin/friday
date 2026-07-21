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
    boolean_search,
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
            "name": "boolean_search",
            "description": (
                "Deterministic whole-word boolean search over EVERY page (the same engine the "
                "researcher's Search tab uses). Syntax: AND, OR, NOT, parentheses, \"quoted phrases\". "
                "Returns total_hits, per-collection counts, and the hit pages in reading order — "
                "exhaustive, not a sample. Aliases/codenames auto-expand.\n\n"
                "Work like a researcher: probe broad, READ THE COUNTS, then narrow. "
                "Example: OSS -> 693 hits; (OSS OR \"Office of Strategic Services\") AND (Soviet OR NKVD) "
                "-> 222; AND agent -> readable. When a count is <=120 the hit list you see IS complete — "
                "ideal for rosters/enumerations (\"names of X in Y\"): walk ALL hits so no member is "
                "missed, then fetch_chunks the relevant ones. Prefer this over semantic search whenever "
                "exact terms are known; prefer semantic search for conceptual/paraphrased questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Boolean query, e.g. (OSS OR \"Office of Strategic Services\") AND (agent OR informant) NOT rumor"},
                },
                "required": ["query"],
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

        if name == "boolean_search":
            query = arguments.get("query", "")
            sess = getattr(workspace, "_search_session", None) or {}
            res = boolean_search(
                conn, query, scope=scope,
                session_id=sess.get("session_id"),
                user_sub=sess.get("user_sub") or "chat-engine",
                origin_query=workspace.question or "",
            )
            if res.get("error"):
                return ({"tool": name, "error": res["error"], "success": False},
                        f"boolean_search error: {res['error'][:80]}")
            # Merge hit pages into the catalog so fetch_chunks can read them.
            # Deterministic whole-word hits outrank sampled semantic hits.
            hits = res.get("hits") or []
            catalog = [
                CatalogHit(
                    chunk_id=h["chunk_id"], score=0.9, doc_id=h.get("document_id"),
                    page=str(h.get("pdf_page") or ""), collection=h.get("collection"),
                    snippet=h.get("snippet") or "",
                )
                for h in hits if h.get("chunk_id")
            ]
            merge_catalog_hits(workspace, catalog)
            if query.strip():
                workspace._search_queries.append(query.strip())
            # Track chat-run result sets so the response can surface them as Search tabs
            try:
                workspace._boolean_result_sets.append({
                    "result_set_id": res.get("result_set_id"),
                    "query": query, "total_hits": res.get("total_hits", 0),
                })
            except AttributeError:
                workspace._boolean_result_sets = [{
                    "result_set_id": res.get("result_set_id"),
                    "query": query, "total_hits": res.get("total_hits", 0),
                }]
            out = {
                "tool": name,
                "query": query,
                "total_hits": res.get("total_hits", 0),
                "per_collection": res.get("per_collection", {}),
                "complete_list_shown": not res.get("truncated"),
                "hits": [
                    {"chunk_id": h["chunk_id"], "collection": h.get("collection"),
                     "page": h.get("pdf_page"), "snippet": (h.get("snippet") or "")[:220]}
                    for h in hits
                ],
                "success": True,
            }
            note = (f"boolean '{query[:50]}' -> {res.get('total_hits', 0)} page hits "
                    f"({'complete' if not res.get('truncated') else f'{len(hits)} shown'})")
            return out, note

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
                                    _bullet_payload(conn, b, chunk_to_page, doc_names)
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
                                    _bullet_payload(conn, b, chunk_to_page, doc_names)
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


# Common words carry no signal for deciding which page a quote sits on; a page
# can share dozens of these with any quote. Only distinctive tokens vote.
_QUOTE_PAGE_STOPWORDS = frozenset(
    "the a an and or of to in on at by for with from as is was were are be been "
    "that this these those it its he she his her they them their we you i not no "
    "but if than then so all any who whom which what when where would could "
    "should has have had will may can do did does".split()
)


def _run_insufficiency_mine(conn, workspace, plan, question: str, *, verbose: bool = False) -> str:
    """Pool-walk escalation for a rejected give-up: boolean pool from the question's
    entities AND record vocabulary, mined exhaustively; extracted facts (verbatim,
    validated quotes) are fetched into the workspace and summarized into the pushback
    message so the resynthesis can cite them."""
    import re as _re
    from retrieval.agent.v11_tools import boolean_search, mine_pool

    plan = plan or {}
    ents = [str(e).strip() for e in (plan.get("entities") or []) if str(e).strip()]
    if not ents:
        return ""
    # Primary term: most discriminating token of the first entity
    ptoks = [t for t in _re.split(r"[,\s]+", ents[0]) if len(t) >= 3]
    primary = max(ptoks, key=len, default=ents[0])
    # OR-group: distinctive non-entity tokens from the records rewrite + anchors,
    # falling back to generic record vocabulary
    vocab = []
    for q in (plan.get("records_queries") or []):
        for t in _re.split(r"\s+", str(q)):
            tl = t.strip().lower()
            if len(tl) >= 5 and tl not in {x.lower() for x in ptoks} and tl not in vocab:
                vocab.append(tl)
    for a in (plan.get("anchors") or []):
        al = str(a).strip().lower()
        if len(al) >= 5 and al != primary.lower() and al not in vocab:
            vocab.append(al)
    if not vocab:
        vocab = ["statement", "report", "memorandum", "letter", "interview"]
    or_group = vocab[:6]
    bq = f"{primary} AND ({' OR '.join(or_group)})"

    sess = getattr(workspace, "_search_session", None) or {}
    bres = boolean_search(conn, bq, scope=workspace.scope,
                          session_id=sess.get("session_id"),
                          user_sub=sess.get("user_sub") or "chat-engine",
                          origin_query=question, max_hits_returned=150)
    pool = [h["chunk_id"] for h in (bres.get("hits") or [])]
    if not pool:
        return ""
    if verbose:
        print(f"  [V14] insufficiency mine: pool '{bq}' -> {bres.get('total_hits')} hits, "
              f"mining {len(pool)}", file=sys.stderr)
    spec = (f"Every explicit statement relevant to answering: {question}. Extract the "
            f"verbatim sentence, any date it contains, and note in role whether it is a "
            f"first-hand record, a report, or a recollection.")
    mres = mine_pool(conn, pool, spec, question, max_chunks=120, verbose=verbose)
    entries = mres.get("entries") or []
    if not entries:
        return ""

    # Fetch the top-cited chunks into the workspace so citations ground-check downstream
    top_cids = list(dict.fromkeys(
        c["chunk_id"] for e in entries[:8] for c in e["citations"]))[:10]
    try:
        fetched = fetch_chunks(conn, chunk_ids=top_cids)
        merge_fetched_chunks(workspace, fetched)
    except Exception:
        pass

    lines = [f"POOL MINE RESULTS — exhaustive read of {mres.get('mined')} pages matching "
             f"[{bq}] (explicit records outrank recollections; cite the chunk_ids):"]
    for e in entries[:8]:
        for c in e["citations"][:2]:
            d = f" (date: {c['date']})" if c.get("date") else ""
            q_txt = (c.get("quote") or "")[:260]
            lines.append(f'- [chunk {c["chunk_id"]}]{d} "{q_txt}"')
    lines.append("Use these VERBATIM findings to answer with citations now.")
    return "\n".join(lines)


def _run_answer_term_chase(conn, workspace, terms, question, *, verbose=False):
    """One-hop chase: a passage NAMED the answer's subject (e.g. "pumpkin papers")
    even though the question's wording never could. Pool the term, read every page,
    return verbatim findings for the resynthesis. Chase output cannot spawn chases."""
    from retrieval.agent.v11_tools import boolean_search, mine_pool
    sess = getattr(workspace, "_search_session", None) or {}
    lines = []
    for t in terms:
        t["chased"] = True
        term = t.get("term") or ""
        if len(term) < 3:
            continue
        bq = f'"{term}"' if " " in term else term
        bres = boolean_search(conn, bq, scope=workspace.scope,
                              session_id=sess.get("session_id"),
                              user_sub=sess.get("user_sub") or "chat-engine",
                              origin_query=question, max_hits_returned=80)
        pool = [h["chunk_id"] for h in (bres.get("hits") or [])]
        if not pool:
            continue
        if verbose:
            print(f"  [V14] answer-term chase: '{term}' -> {bres.get('total_hits')} hits",
                  file=sys.stderr)
        spec = (f"Every statement explaining what {term} is/was and how it relates to: "
                f"{question}. Verbatim sentences with any dates.")
        mres = mine_pool(conn, pool, spec, question, max_chunks=60, verbose=verbose)
        for e in (mres.get("entries") or [])[:5]:
            for c in e["citations"][:2]:
                q_txt = (c.get("quote") or "")[:240]
                lines.append(f'- [chunk {c["chunk_id"]}] "{q_txt}"')
        top_cids = list(dict.fromkeys(
            c["chunk_id"] for e in (mres.get("entries") or [])[:5]
            for c in e["citations"]))[:8]
        try:
            fetched = fetch_chunks(conn, chunk_ids=top_cids)
            merge_fetched_chunks(workspace, fetched)
        except Exception:
            pass
    if not lines:
        return ""
    head = ("ANSWER-TERM CHASE — a passage named the specific subject of the question; "
            "every page mentioning it has now been read. Verbatim findings (cite chunk_ids):")
    return head + chr(10) + chr(10).join(lines) + chr(10) + "Revise your answer to use these findings."


def _lookup_quote_page(conn, chunk_id: int, quote: str) -> Optional[int]:
    """Find which of a chunk's PDF pages contains the quote, by matching the quote
    against each page's raw_text (chunk_pages char offsets are unpopulated, so text
    matching is the reliable mapping). Chunks span 1-6 pages, so this is cheap.
    Matching is space-insensitive (line wrapping can differ between chunk text and
    page raw_text). Returns None when the quote can't be placed (viewer falls back
    to first page)."""
    from retrieval.agent.v9_summarize import _norm_with_map
    q_norm, _ = _norm_with_map(quote or "")
    if len(q_norm) < 15:
        return None
    q_sf = q_norm.replace(" ", "")  # space-free form
    q_tokens = {
        t for t in q_norm.split(" ")
        if len(t) >= 3 and t not in _QUOTE_PAGE_STOPWORDS
    }
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.pdf_page_number, p.raw_text
                FROM chunk_pages cp JOIN pages p ON p.id = cp.page_id
                WHERE cp.chunk_id = %s
                ORDER BY cp.span_order
                """,
                (chunk_id,),
            )
            rows = cur.fetchall()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    if len(rows) == 1:
        return int(rows[0][0]) if rows[0][0] is not None else None
    best_page, best_score = None, 0.0
    for pdf_page, raw_text in rows:
        if pdf_page is None or not raw_text:
            continue
        p_norm, _ = _norm_with_map(raw_text)
        if q_sf in p_norm.replace(" ", ""):
            return int(pdf_page)
        p_tokens = set(p_norm.split(" "))
        score = len(q_tokens & p_tokens) / max(len(q_tokens), 1)
        if score > best_score:
            best_page, best_score = int(pdf_page), score
    return best_page if best_score >= 0.6 else None


def _bullet_payload(conn, b, chunk_to_page: Dict[int, Optional[int]],
                    doc_names: Dict[int, str]) -> Dict[str, Any]:
    """Serialize an EvidenceBullet for the evidence_update SSE payload, including the
    verbatim support quote + its exact PDF page (for on-page highlighting)."""
    payload: Dict[str, Any] = {
        "text": b.text,
        "tags": b.tags,
        "chunk_ids": b.supporting_chunk_ids,
        "doc_ids": b.doc_ids,
        "pages": [chunk_to_page.get(cid) for cid in b.supporting_chunk_ids],
        "source_names": [doc_names.get(did, "") for did in (b.doc_ids or [])],
    }
    if getattr(b, "support_quote", "") and getattr(b, "quote_chunk_id", None):
        payload["quote"] = b.support_quote
        payload["quote_chunk_id"] = b.quote_chunk_id
        payload["quote_page"] = _lookup_quote_page(conn, b.quote_chunk_id, b.support_quote)
    return payload


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
            temperature=float(os.getenv("V9_AGENT_TEMPERATURE", "0.2")),
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
    search_session: Optional[Dict[str, Any]] = None,
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
    # v14 inherits every v13 behaviour and adds coverage-first retrieval for
    # roster/count/aggregation intents (see v13_planner.prime_workspace).
    _v13 = engine_profile in ("v13", "v14")
    _v14 = (engine_profile == "v14")
    if _v13:
        # V13+ always uses the lightweight [MENTION_INDEX] so synthesis can bridge
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

    # Session identity for boolean_search persistence (chat-run searches become
    # session result sets the researcher can open from the Search tab).
    workspace._search_session = search_session or {}

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
                coverage=_v14,
            )
            # Enumeration questions: steer the agent to the deterministic boolean
            # engine (the researcher's instrument) instead of leaving tool choice
            # to habit — semantic sampling misses roster members that whole-word
            # boolean intersection cannot.
            if (_v13_plan.get("intent") or "").lower() in ("roster", "count"):
                append_note(workspace,
                    "Enumeration question: use boolean_search (AND/OR/NOT, whole-word, "
                    "exhaustive page hits with per-collection counts) to expand and verify "
                    "the roster — e.g. probe the group/agency name, read total_hits, narrow "
                    "with AND until <=120, then walk ALL hits. Do not rely on semantic "
                    "top-k sampling alone for membership lists.")
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
    finalize_retries = 0  # bounded pushback on rejected finalizations (anti-give-up)
    answer_chase_done = False  # one-hop answer-term chase (FRIDAY_ANSWER_TERMS)
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
                # The FINAL synthesis call happens once — no per-turn resend multiplier —
                # so it gets a rich evidence pack (all bullets + key chunks) instead of the
                # loop-sized squeeze. Falls back to the minimal context on failure.
                _synth_budget = int(os.getenv("V9_SYNTHESIS_CONTEXT_TOKENS", "25000"))
                try:
                    rich_ctx = build_context_pack(
                        workspace, delta,
                        token_budget=_synth_budget,
                        max_fulltext=24,
                        findings_brief=_findings_brief,
                    )
                    synth_prompt = (
                        "Tool budget exhausted. Synthesize the FINAL answer now from the "
                        "evidence below. Set final=true, fill sufficiency and responsiveness.\n\n"
                        + rich_ctx
                    )
                except Exception:
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

        # Answer-term chase (one hop, before accepting a final): a fetched passage
        # NAMED the thing the question asks about — read every page naming it first.
        if (content.get("final")
                and os.getenv("FRIDAY_ANSWER_TERMS", "0") == "1"
                and not answer_chase_done):
            _pending = [t for t in (getattr(workspace, "_answer_terms", None) or [])
                        if not t.get("chased")]
            if _pending:
                answer_chase_done = True
                try:
                    _chase_msg = _run_answer_term_chase(
                        conn, workspace, _pending[:2], clean_question, verbose=verbose)
                except Exception as _ce:
                    _chase_msg = ""
                    if verbose:
                        print(f"  [V14] answer-term chase failed: {_ce}", file=sys.stderr)
                if _chase_msg:
                    messages.append({"role": "user", "content": _chase_msg})
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
            # Act on the verdict (previously log-only): push the model back into the
            # loop with the concrete issues, bounded to 2 retries so a stubborn model
            # can't ping-pong. This is the anti-give-up mechanism — a "couldn't find
            # it" final with unused budget gets sent back to search, not shipped.
            if finalize_retries < 2:
                finalize_retries += 1
                pushback = (
                    "Your finalization was rejected: " + " | ".join(issues[:3]) +
                    " Address these now — continue investigating with tools if needed, "
                    "then finalize."
                )
                # Insufficiency mine: a rejected "couldn't find it" with budget left
                # gets a POOL WALK, not just exhortation — a boolean pool from the
                # question's entities x record vocabulary, every page read.
                # FRIDAY_INSUFFICIENCY_MINE=0 disables.
                if (finalize_retries == 1
                        and os.getenv("FRIDAY_INSUFFICIENCY_MINE", "1") == "1"
                        and any("do not give up" in i for i in issues)):
                    try:
                        mine_msg = _run_insufficiency_mine(
                            conn, workspace, _v13_plan, clean_question, verbose=verbose)
                        if mine_msg:
                            pushback += chr(10) + chr(10) + mine_msg
                    except Exception as _im_err:
                        if verbose:
                            print(f"  [V14] insufficiency mine failed: {_im_err}",
                                  file=sys.stderr)
                messages.append({"role": "user", "content": pushback})
                synthesis = None
                prev_counts = _snapshot_counts(workspace)
                continue
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

    # V14: answer-faithfulness pass for specific-answer questions — read the actual
    # answer-bearing passages and answer directly (fixes needles the synthesis omitted and
    # self-contradictions where the answer negates its own evidence). Skips roster (own path).
    if _v14 and _v13_plan and _v13_plan.get("intent") not in ("roster", "list", "enumerate"):
        try:
            from retrieval.agent.v13_planner import grounded_finalize
            grounded_finalize(conn, result, workspace, _v13_plan, clean_question, verbose=verbose)
        except Exception as _fe:
            if verbose:
                print(f"  [V14] grounded_finalize failed (non-fatal): {_fe}", file=sys.stderr)

    # V14: for roster/list intents, assemble a corpus-wide roster from the evidence bullets
    # instead of relying on the truncation-prone full synthesis (which cherry-picks 2-3 names).
    # EXCEPT when the target is a GROUP (network/ring/apparatus): the person-enumerator is the
    # wrong tool — the question wants named groups, which the synthesis narrative supplies — so
    # leave the synthesis in place rather than replacing it with a list of individuals.
    _GROUP_TARGETS = {"network", "networks", "ring", "rings", "group", "groups", "apparatus",
                      "cell", "cells", "organization", "organizations", "organisation",
                      "organisations", "operation", "operations"}
    if _v14 and _v13_plan and (_v13_plan.get("intent") in ("roster", "list", "enumerate")):
        tgt = _v13_plan.get("enumeration_target") or "person"
        # Token-based match so multi-word targets ("espionage networks", "spy rings") also route
        # to the group enumerator, not just the bare noun.
        _tgt_is_group = bool({w for w in re.split(r"[^a-z]+", tgt.lower()) if w} & _GROUP_TARGETS)
        if _tgt_is_group:
            # A group question ("what networks operated") wants NAMED GROUPS, not individuals —
            # enumerate the named networks/rings, falling back to the synthesis narrative only if
            # we can't ground at least two.
            try:
                from retrieval.agent.v13_planner import assemble_group_roster
                group_claims = assemble_group_roster(conn, workspace, question=clean_question, verbose=verbose)
                if len(group_claims) >= 2:
                    result.claims = group_claims
                    summary = f"Identified {len(group_claims)} named Soviet espionage network(s) across the corpus."
                    result.narrative = summary
                    try:
                        result.grounded_roster = []
                        result._authoritative_narrative = True
                    except Exception:
                        pass
                    if result.synthesis:
                        result.synthesis.narrative = summary
                        if hasattr(result.synthesis, "artifact"):
                            try:
                                result.synthesis.artifact = {}
                            except Exception:
                                pass
                elif verbose:
                    print(f"  [V14] group-target roster ('{tgt}') -> <2 groups, kept synthesis", file=sys.stderr)
            except Exception as _ge:
                if verbose:
                    print(f"  [V14] group-roster failed (non-fatal): {_ge}", file=sys.stderr)
        else:
            try:
                from retrieval.agent.v13_planner import assemble_roster
                roster_claims = assemble_roster(conn, workspace, tgt, question=clean_question,
                                                plan=_v13_plan, verbose=verbose)
                if len(roster_claims) >= 3:  # only override when we actually enumerated a roster
                    result.claims = roster_claims
                    summary = f"Identified {len(roster_claims)} {tgt} linked to Soviet intelligence across the corpus."
                    result.narrative = summary
                    # Drop the stale synthesis artifact/roster so format_answer doesn't render a
                    # leftover "Members identified"/"Timeline" block (with the officers/codenames the
                    # grounded roster just filtered out) alongside the clean roster. Mark the
                    # summary authoritative so it isn't stamped "draft/unverified".
                    try:
                        result.grounded_roster = []
                        result._authoritative_narrative = True
                    except Exception:
                        pass
                    if result.synthesis:
                        result.synthesis.narrative = summary
                        if hasattr(result.synthesis, "artifact"):
                            try:
                                result.synthesis.artifact = {}
                            except Exception:
                                pass
            except Exception as _re:
                if verbose:
                    print(f"  [V14] roster assembly failed (non-fatal): {_re}", file=sys.stderr)

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
