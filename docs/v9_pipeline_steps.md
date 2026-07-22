# V9 Pipeline — Full Description

This document describes **every step** of the V9 (V9.4/V9.5) pipeline from user message to final answer. V9 is session-aware and can be entered via **dispatch** (session + routing) or by calling **run_v9_query** directly.

---

## Entry points

1. **`dispatch_message(conn, session_id, user_message, ...)`** in `retrieval/agent/v9_dispatch.py`  
   - Used by the CLI/API when a session exists. Routes the message then runs the chosen path.
2. **`run_v9_query(conn, question, ...)`** in `retrieval/agent/v9_runner.py`  
   - Core investigation loop. Can be called directly or by dispatch for **new_retrieval** and **think_deeper**.

---

## Dispatch layer (session-aware)

### 1. Load session and recent context

- **`load_session(conn, session_id)`** — ensure session exists.
- **`load_recent_runs(conn, session_id)`** — recent runs and evidence sets for routing.

### 2. Route the message

- **`route_message(user_message, context, explicit_action=..., verbose=...)`** in `v9_router.py`  
  - Classifies intent: **new_retrieval** | **follow_up** | **think_deeper**.
  - Uses heuristics (e.g. “think deeper”, “that one”, “explain this”) and optionally an LLM.
  - Resolves references (e.g. “the Silvermaster one”) to `target_run_id` / `target_evidence_set_id`.
  - Returns **RouterDecision** (intent, target_run_id, target_evidence_set_id, confidence, reasoning, query_text).

### 3. Execute by intent

- **follow_up** → **`_run_follow_up(...)`**  
  - **`execute_followup(conn, user_message, evidence_set_id=..., ...)`** in `v9_followup.py`.  
  - Answers from the **existing evidence set** only (no search/fetch/expand tools).  
  - Returns answer, suggestion, scope_meta, escalations (e.g. “think deeper”, “new search”).

- **think_deeper** → **`_run_think_deeper(...)`**  
  - Load run and resume state; **`rehydrate_workspace_from_evidence(conn, workspace, evidence_set_id)`**.  
  - Call **`run_v9_query(conn, question, _resume_workspace=workspace, ...)`** with the rehydrated workspace and extended tool budget.  
  - Persist new evidence and steps as in new_retrieval.

- **new_retrieval** (default) → **`_run_new_retrieval(...)`**  
  - Resolve scope: **`_resolve_scope_for_run(conn, session, user_message, ...)`** → scope filter + run_scope_json.  
  - Call **`run_v9_query(conn, user_message, scope=..., ...)`** (see “Core: run_v9_query” below).  
  - After **run_v9_query** returns: run **Stage 1.5** (if applicable), then persist evidence and run state.

---

## Core: run_v9_query (Stage 1 = investigation loop)

**run_v9_query** does: scope + router parse → prime workspace → investigation loop (tools + structured output) → grounding + verification. There is no separate “Stage 1” name in code; the first retrieval is the first tool use inside the loop. “Stage 1” here means “the retrieval done during the investigation loop”. “Stage 1.5” runs **after** run_v9_query returns, in dispatch.

### Step 1a — Scope (deterministic)

- **`detect_scope(question)`** — inline syntax (e.g. `scope=collections=venona`) and natural-language scope detection.
- If caller passed **scope**, merge (collections, date_from, date_to).
- **`strip_scope_syntax(question)`** → **clean_question** (model never sees scope directives).

### Step 1b — Lightweight router (optional LLM)

- **`_lightweight_parse_query(clean_question)`** (only when not resuming):
  - **Model:** GPT-4o-mini with **V9_ROUTER_MODEL** and a small schema.
  - **Output:** **QueryParse** — `content_keywords`, `collections`, `date_from`, `date_to`, `intent`, `reformulated_query`.
  - Used to restrict entity resolution to **content_keywords** (no blind extraction of every word) and to set **workspace.investigation.goal** from intent (identity, timeline, roster, evidence, relationship, general).
  - On API failure: **`_fallback_parse_query(question)`** — regex extraction of ALL-CAPS, title-case, quoted strings.
- Merge router scope into **detected_scope** where not already set.

### Step 1c — Workspace creation or resume

- If **`_resume_workspace`** is provided: use it, set **workspace.scope = detected_scope**.
- Else: **`ResearchWorkspace(question=reformulated_query or clean_question, scope=detected_scope)`**.

### Step 2 — Prime workspace (entity resolution + auto-expand)

- **`_prime_workspace_from_question(conn, clean_question, workspace, content_keywords=query_parse.content_keywords)`**

  **A) With content_keywords (router path):**

  - **`_resolve_keywords(conn, keywords, workspace)`** for each keyword:
    - **concordance_expand_terms** → expanded terms; for each term: entities (canonical + alias) + **entity_mentions** count; score = 0.75 + 0.3*log10(mention_count+1).
    - **Partial LIKE** on canonical_name and alias (e.g. “Silvermaster” → Nathan Gregory Silvermaster, Helen Silvermaster); score 0.50 + log term.
    - **Direct alias** lookup (entity_aliases); **fuzzy** (trigram) on canonical + aliases if needed.
    - Rank by score; add **EntityCandidate** via **merge_entity_candidates(workspace, ...)**.

  **B) Without content_keywords (legacy):**

  - **`resolve_question_entities(conn, question, scope=workspace.scope)`** → raw candidates → **merge_entity_candidates**.
  - For each word in the question not already resolved: **`_try_alias_reverse_lookup(conn, word, workspace)`**.

  **C) Auto-expand (both paths):**

  - **`_auto_expand_candidates(conn, workspace, max_auto_expand=3)`**
    - Take candidates with confidence in (`exact`, `concordance`, `partial`), not accepted, not ambiguous.
    - For each (up to 3): **`expand_entities(conn, entity_ids=[...], include_mentions=True, mentions_top_k=30, scope=workspace.scope)`** → load canonical, aliases, mention chunk IDs.
    - **workspace.accept_candidate(eid)**; **merge_entities**; **merge_catalog_hits** with **`_load_catalog(conn, mention_cids)`**.
    - **concordance_expand_terms** on canonical/query_term; merge new aliases into workspace entity.

  After priming, the workspace has: **entity_candidates**, **entities**, and optionally **catalog_hits** (mention chunks) and **fulltext_chunks** (if any pre-loaded).

### Step 3 — Investigation loop

- **messages** = `[{"role": "system", "content": SYSTEM_PROMPT}]`.
- **scope_note** = human-readable scope filter for the user prompt.
- Loop: **`while not done and tool_calls_executed < max_tool_calls and model_turns < MAX_MODEL_TURNS`**.

#### 3.1 — Build user message and call LLM

- **`delta = _compute_delta(workspace, prev_counts, tools_called_this_turn)`**.
- **`ctx = build_context_pack(workspace, delta)`** in `v9_context.py`:
  - Token-budgeted pack: investigation state (goal, gaps, notes, trace), catalog (snippets), fulltext (newest fetched), **evidence memory view** (pinned + recent + top-relevant bullets), sufficiency.
- **user_content = USER_PROMPT_TEMPLATE.format(question=clean_question, scope_note=scope_note, context=ctx)**.
- **messages.append({"role": "user", "content": user_content})**.
- **messages = _trim_messages(messages)** (cap history length).
- **max_completion_tokens**: SYNTHESIS_MAX_TOKENS if ready_to_synthesize and gaps empty or last turn; else TOOL_TURN_MAX_TOKENS.
- **`response = _call_with_retry(client, model, messages, TOOLS_DEF, max_ct, workspace, delta, clean_question, scope_note, verbose)`** — single LLM call with **V9_OUTPUT_SCHEMA** (structured output) and tools.
- **msg = response.choices[0].message**; **model_turns += 1**.

#### 3.2 — Parse structured content

- **`output = _parse_content(msg.content)`** — JSON with **final**, **scratchpad_update**, **synthesis**.
- **scratchpad_update** → **`_update_investigation(workspace.investigation, output["scratchpad_update"])`** (goal, gaps, notes, trace, etc.).
- **pin_suggestions** (if present) → **`apply_pin_suggestions(workspace, pin_sugs)`**.

#### 3.3 — Branch A: Tool calls present

- Append assistant message (content + tool_calls) to **messages**.
- For each tool call:
  - If **tool_calls_executed >= max_tool_calls**: append error tool message; continue.
  - **tool_calls_executed += 1**; parse **arguments**.
  - **`result, summary = _execute_tool(name, args, conn, workspace, progress_callback)`**:
    - **search_chunks**: **expand_query_with_aliases(query, workspace, conn)** → **search_chunks(conn, query=expanded_query, top_k, collections, scope, mode)**; **merge_search_result(workspace, result, catalog, query)**; merge any **concordance_resolution** into **entity_candidates**.
    - **fetch_chunks**: **fetch_chunks(conn, chunk_ids=..., doc_id=..., around_chunk_id=..., window=..., page_start/page_end=...)**; scope-filter chunks; **merge_fetched_chunks(workspace, chunks)**; **link_chunks_to_entities(workspace, chunks)**; **summarize_delta_chunks(...)** → evidence memory update; **merge_evidence_summary_update(workspace, ev_update)**.
    - **expand_entities**: **expand_entities(conn, entity_ids=..., names=..., include_mentions=..., scope=...)**; for each entity: accept_candidate, **concordance_expand_terms** on canonical/names, merge entities + catalog hits.
  - Append tool result to **messages**; append **InvestigationStep** to **workspace.investigation.trace**; **append_note(workspace, ...)**.
  - If **len(fulltext_chunks) + len(catalog_hits) >= max_workspace_chunks**: break.
- **prev_counts = _snapshot_counts(workspace)**.

#### 3.4 — Budget exhausted

- If **tool_calls_executed >= max_tool_calls**: append user message “Tool budget exhausted. Synthesize now… Set final=true …”; call LLM again (no tools); parse synthesis → **synthesis = V9Synthesis.from_dict(synth_data)**; **done = True**. If structured output fails: **synthesis = _build_needs_more_evidence_synthesis(workspace, clean_question)**.
- **continue** (next iteration).

#### 3.5 — Branch B: No tool calls (synthesis expected)

- If **output** is missing: build **minimal_ctx** (no catalog/fulltext bodies); retry LLM with minimal context. If still no output: **synthesis = _build_needs_more_evidence_synthesis(...)**; **done = True**; continue.
- Append assistant message.
- If **not output.get("final")**: append nudge (“You set final=false but did not call tools…”); continue.
- **synthesis = V9Synthesis.from_dict(output)**.
- **valid, issues = _validate_finalization(synthesis, clean_question, tool_calls_executed, max_tool_calls, workspace)**. If not valid and budget allows: append user message with issues; continue.
- **Optional auditor** (if **V9_USE_AUDITOR=1**): **`_run_auditor(client, synthesis, clean_question)`**; if not responsive, append feedback and continue.
- **done = True**.

### Step 4 — Post-loop: identity ingestion

- If **synthesis.artifact** has identity (alias + canonical): **`resolve_surfaced_alias(workspace, alias_text=..., conn=conn, turn_idx=model_turns)`** and merge into workspace/DB as needed.

### Step 5 — Grounding and verification

- **`grounded = ground_claims(synthesis.claims, workspace)`** in `v9_grounding.py` — match claims to workspace chunks and set citation status.
- **`report = build_verification_report(grounded, synthesis)`** in `v9_verify.py`.

### Step 6 — Return

- **return V9Result(narrative=synthesis.narrative, claims=grounded, verification=report, sufficiency=synthesis.sufficiency, synthesis=synthesis, workspace=workspace, investigation_trace=...)**.

---

## Stage 1.5 — Targeted concordance expansion (dispatch only)

- Runs **after** **run_v9_query** returns, inside **`_run_new_retrieval`**, only when **result.workspace** and **result.workspace.fulltext_chunks** exist and **run_scope_json** is set.
- **`maybe_expand_from_target_collections(conn, result.workspace.fulltext_chunks, scope, run_scope_json, retrieval_ctx, run_id=run.run_id, verbose=...)`** in `v9_dispatch.py`:
  1. **Target collections** = CONCORDANCE_EXPANSION_TARGET_COLLECTIONS (e.g. venona, vassiliev).
  2. **Filter** fulltext_chunks by **collection_slug** → **target_chunk_ids**.
  3. If none: set run_scope_json expansion.triggered = false; return.
  4. **`_compute_expansion_scope(conn, scope_filter, target_collections)`** — intersect user scope with target collections; if empty, skip.
  5. **`_extract_expansion_entities(conn, target_chunk_ids)`** — entities from concordance tables for those chunks.
  6. **Expansion query** = space-joined canonical names (top 8).
  7. **search_chunks(conn, expansion_query, top_k=..., scope=expansion_scope, mode=...)**.
  8. **fetch_chunks_with_neighbors** for new chunk_ids (cap CONCORDANCE_EXPANSION_MAX_EXTRA_CHUNKS); apply **CONCORDANCE_EXPANSION_SCORE_PENALTY** to scores.
  9. Merge **extra_chunks** into **result.workspace.fulltext_chunks** (de-dup by chunk_id).
- **update_run_scope_json(conn, run.run_id, run_scope_json)**.

---

## Persistence (new_retrieval / think_deeper)

- **add_evidence_items(conn, run.evidence_set_id, ws.fulltext_chunks, step_idx=..., scores=...)**.
- **add_adjacency_chunks(conn, run.evidence_set_id, primary_cids, step_idx=...)**.
- **prune_evidence_set(conn, run.evidence_set_id)**.
- **persist_step(conn, run.run_id, step.step_idx, tool_name=..., tool_args=..., lane=..., result_refs=...)** for each step in **workspace.investigation.trace**.
- **generate_evidence_summary(ws)**; **extract_top_entities(ws)**.
- **resume_state = build_resume_state(ws, tool_calls_executed=..., model_turns=0, step_idx=..., max_tool_calls=...)**.
- **update_run_status(conn, run.run_id, run_status, last_step_idx=..., resume_state_json=resume_state, evidence_summary=..., top_entities_json=..., label=...)**.
- **update_session_active(conn, session_id, active_run_id=..., active_evidence_set_id=..., active_run_status=...)**.

---

## Tools (summary)

| Tool | Effect |
|------|--------|
| **search_chunks** | expand_query_with_aliases → search_chunks(conn, ...) → merge_search_result; merge concordance_resolution into entity_candidates. |
| **fetch_chunks** | fetch_chunks(conn, chunk_ids / doc_id+around_chunk_id / doc_id+page_start+page_end); scope filter; merge_fetched_chunks; link_chunks_to_entities; summarize_delta_chunks → evidence memory. |
| **expand_entities** | expand_entities(conn, entity_ids, names, include_mentions, scope); accept_candidate; concordance_expand_terms on canonical/names; merge_entities + merge_catalog_hits. |

---

## Constants (representative)

- **V9_MAX_TOOL_CALLS** (default 5 in dispatch; overridable).
- **V9_MAX_WORKSPACE_CHUNKS** — cap on catalog + fulltext.
- **MAX_MODEL_TURNS** — max LLM turns.
- **TOOL_TURN_MAX_TOKENS** / **SYNTHESIS_MAX_TOKENS** — completion token caps.
- **CONCORDANCE_EXPANSION_TARGET_COLLECTIONS**, **CONCORDANCE_EXPANSION_MAX_EXTRA_CHUNKS**, **CONCORDANCE_EXPANSION_SCORE_PENALTY** — Stage 1.5.

---

## End-to-end flow (new_retrieval)

1. **dispatch_message** → route_message → intent **new_retrieval**.
2. **_resolve_scope_for_run** → scope + run_scope_json.
3. **run_v9_query**: scope + strip; _lightweight_parse_query → QueryParse; workspace; _prime_workspace_from_question (resolve keywords or legacy resolve + _auto_expand_candidates).
4. Loop: build_context_pack → user message → _call_with_retry (LLM + tools) → parse scratchpad + pin_suggestions; if tool_calls → _execute_tool (search/fetch/expand), merge, trace, notes; if budget exhausted → force synthesis; if no tool_calls and final → validate (+ optional auditor) → done.
5. resolve_surfaced_alias from synthesis.artifact; ground_claims; build_verification_report; return V9Result.
6. **_run_new_retrieval** continues: **maybe_expand_from_target_collections** (Stage 1.5) → merge extra chunks into workspace; add_evidence_items; add_adjacency_chunks; prune_evidence_set; persist_step; generate_evidence_summary; extract_top_entities; build_resume_state; update_run_status; update_session_active.
7. Return DispatchResult (answer, cited_chunk_ids, run_id, evidence_set_id, run_status, can_think_deeper, citation_map, etc.).
