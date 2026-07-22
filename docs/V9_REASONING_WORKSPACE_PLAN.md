# V9 Agent Plan: Reasoning Workspace Loop (GPT-4o)

## Executive Summary

V9 replaces the **agentic pipeline** (tools driving the model) with a **Reasoning Workspace Loop**: the model drives the tools, and the system owns the audit. A single strong reasoning model (GPT-4o) has free reign over retrieval; synthesis is produced first (narrative + structured claims); then post-hoc grounding and verification run as advisory, non-blocking steps.

**Core principle:** *The model owns the plan. The system owns the audit.*

---

## Table of Contents

1. [Core Design Shift](#1-core-design-shift)
2. [Step 1: Reasoning Model with Free Reign](#2-step-1-reasoning-model-with-free-reign)
3. [Step 2: Research Workspace Object](#3-step-2-research-workspace-object)
4. [Step 3: Model-Driven Iterative Retrieval](#4-step-3-model-driven-iterative-retrieval)
5. [Step 4: Synthesis First, Grounding Second](#5-step-4-synthesis-first-grounding-second)
6. [Step 5: Post-Hoc Grounding](#6-step-5-post-hoc-grounding)
7. [Step 6: Advisory Verification](#7-step-6-advisory-verification)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Mapping to Existing Codebase](#9-mapping-to-existing-codebase)
10. [Out of Scope for V9 (Explicitly Removed)](#10-out-of-scope-for-v9)

---

## 1. Core Design Shift

| Current (V6/V7) | V9 |
|-----------------|-----|
| Intent detection → entity linking → retrieval rounds → bottleneck → synthesis → verification | Question → LLM (unconstrained) → tools → workspace → synthesis → post-hoc grounding → advisory verification |
| System imposes rounds, bottleneck, lanes, modes | No system-imposed rounds; model decides when to search/fetch/expand and when to answer |
| Verification can block or drop claims | Verification is advisory; answer is never killed |
| Many tools (hybrid, lexical, entity_mentions, co_mention, expand_aliases, neighbors, …) | Three tools: `search_chunks`, `fetch_chunks`, `expand_entities` |
| Plan compiler, query parser, progress gate, tournament grading | None of these |

Flow:

```
Question
  ↓
LLM (Reasoning Model, unconstrained)
  ↓
Tool calls (search, fetch, expand, recall)
  ↓
Large Evidence Workspace
  ↓
LLM synthesizes answer
  ↓
Post-hoc grounding & verification (advisory)
```

---

## 2. Step 1: Reasoning Model with Free Reign

### 2.1 Single Model

- **Model:** GPT-4o (or GPT-4.1 / o4-style when available).
- **No intent detection** — no `QueryParser`, no CONTROL/CONTENT split.
- **No plan compiler** — no `generate_plan` / `AgentPlanV3`.
- **No bottleneck** — no fixed cap on evidence size during reasoning; workspace can grow (with a high safety cap, e.g. 500 chunks).
- **No tournament** — no pairwise grading or Elo; retrieval is purely tool-driven.

### 2.2 Minimal Tool Set

Only three tools (or four if “recall” is a separate read from workspace):

| Tool | Purpose | Maps to existing |
|------|----------|-------------------|
| `search_chunks(query, top_k, filters?)` | Semantic/lexical search over the archive | `hybrid_search` (or thin wrapper over `hybrid_rrf` + filters); optional `lexical_search`/`lexical_exact` as single “search” with mode, or keep one unified search. |
| `fetch_chunks(chunk_ids)` | Get full text (and metadata) for given chunk IDs | New; use `_load_chunk_texts`-style query (e.g. `retrieval/observations._load_chunk_texts` or equivalent in `retrieval/ops` / new v9 module). |
| `expand_entities(entity_ids)` | Resolve entity IDs to names/aliases and optionally get mention counts or related chunks | `entity_surfaces` + optional `entity_mentions`; expose as one “expand” call. |

No lanes, no modes, no retries at this stage. The model calls what it needs.

### 2.3 Configuration

- `V9_MODEL` default: `gpt-4.1-mini-2025-04-14`.
- `V9_MAX_WORKSPACE_CHUNKS`: e.g. 500 (hard cap to avoid runaway context).
- `V9_MAX_TOOL_CALLS_PER_TURN`: e.g. 10 (optional, to avoid infinite loops).
- No `max_rounds` in the V6/V7 sense; the loop is “until model returns final answer or hits a safety limit”.

---

## 3. Step 2: Research Workspace Object

### 3.1 Definition

Introduce a **ResearchWorkspace** that accumulates state the model can see and reason over. It is the single accumulation point for evidence and model-written scratchpad.

```python
@dataclass
class ResearchWorkspace:
    question: str
    notes: List[str]           # Model-written scratchpad (not user-visible)
    hypotheses: List[str]     # Model-written (not user-visible)
    chunks: List[WorkspaceChunk]  # Raw text + chunk_id, doc_id, optional score/source
    entities: List[WorkspaceEntity]  # Discovered entities (id, name, aliases, optional mention count)
    uncertainty_flags: List[str]     # Model or system flags (e.g. "gap: no direct evidence for X")
```

- **notes / hypotheses:** Filled by the model via structured message or tool (e.g. “append_note”, “append_hypothesis”) or by parsing assistant messages. Not shown in the final UI; used for logs and optional historian view.
- **chunks:** Append-only. Each time `search_chunks` or `fetch_chunks` returns, merge into `workspace.chunks` (dedupe by `chunk_id`), and inject into the next turn’s context.
- **entities:** Populated when the model calls `expand_entities` or when we infer from tool results; optional.

### 3.2 Persistence and Context

- Workspace is in-memory for the duration of one query.
- For each assistant turn, the system formats the current workspace (e.g. “Current evidence: …”, “Your notes: …”) and appends it to the conversation so the model can revise its understanding and decide next tool calls or final answer.

---

## 4. Step 3: Model-Driven Iterative Retrieval

### 4.1 Prompt Style

System prompt (concise):

- You are analyzing a historical archive.
- Use tools as needed. You may search multiple times and revise your understanding.
- Accuracy is more important than speed.
- When you have enough evidence to answer, provide your answer in the required format (see below).

No prescribed “rounds”; the model can:

- Call `search_chunks("Silvermaster proximity fuse")` → then `fetch_chunks([...])` → write a note (“suggests indirect transfer”) → then `search_chunks("proximity fuse Soviet intelligence")` → etc., until it decides to synthesize.

### 4.2 Loop Logic

1. **Input:** `question` + optional initial context (e.g. collection filters).
2. **Loop:**
   - Append current **workspace summary** (chunks, notes, hypotheses) to messages.
   - Call LLM with tools: `search_chunks`, `fetch_chunks`, `expand_entities`, and optionally `append_note` / `append_hypothesis` (or derive notes from assistant text).
   - If LLM returns **tool_calls:** execute them, merge results into `ResearchWorkspace`, then repeat.
   - If LLM returns **final answer** (narrative + structured claims): exit loop.
3. **Safety:** Stop after `V9_MAX_TOOL_CALLS` total or when total `workspace.chunks` exceeds `V9_MAX_WORKSPACE_CHUNKS`.

This is similar in spirit to V6’s “LLM decides tools,” but without query parser, entity linker pre-step, bottleneck, or round summaries. The only “round” is the turn; the model can use many turns of tool use before answering.

---

## 5. Step 4: Synthesis First, Grounding Second

### 5.1 Two Outputs from the Model

The model is prompted to produce:

1. **Narrative answer (free-form, human-like)**  
   Example:  
   *“There is suggestive but not conclusive evidence in the Vassiliev notebooks that Soviet intelligence became aware of proximity fuse technology through U.S. sources in the early 1940s. The material points more toward intelligence awareness than direct acquisition…”*  
   - No citations yet. Hedges and uncertainty allowed. This is what users see as the main answer.

2. **Claim decomposition (structured, internal)**  
   Example:
   ```json
   {
     "claims": [
       {
         "text": "Soviet intelligence was aware of U.S. proximity fuse development",
         "confidence": "medium",
         "requires_citation": true
       }
     ]
   }
   ```

So: **synthesis first** (narrative + claims), **grounding second** (done by the system in Step 5).

### 5.2 Output Schema (for LLM)

Define a clear schema for the “final answer” message, e.g.:

- `narrative`: string (the user-facing answer).
- `claims`: array of `{ text, confidence, requires_citation }`.

The model is not required to attach citations to claims; the post-hoc grounding step will do that.

---

## 6. Step 5: Post-Hoc Grounding

After the model produces the narrative and claims:

- For each claim with `requires_citation: true`, run **citation binding** against the workspace chunks (and optionally spans):
  - Prefer **direct quote** match;
  - Allow **partial support** or **indirect evidence**;
  - Allow outcome **“no citation found”** without failing the answer.
- Outcomes:
  - **Grounded** — at least one supporting chunk/span attached.
  - **Weakly supported** — flagged (e.g. inference across docs, or weak match).
  - **Unsupported** — no citation found; claim is kept but marked (e.g. in verification report and optional UI).

Do **not** retry reasoning or re-prompt the model when grounding fails. “ChatGPT wouldn’t.”

### 6.1 Reuse / New Code

- Citation binding can reuse ideas from `v4_verify` (supporting phrase in cited text), `v7_claim_enumerator` (claim–evidence association), and existing span/chunk loading.
- New module: **v9_grounding** — input: narrative, claims, workspace chunks; output: per-claim grounding status and optional span/chunk refs.

---

## 7. Step 6: Verification Becomes Advisory, Not Blocking

### 7.1 VerificationReport (Advisory)

Verifier output shape:

```python
@dataclass
class V9VerificationReport:
    grounded_claims: int
    weak_claims: int
    unsupported_claims: int
    notes: List[str]  # e.g. "Claim 2 relies on inference across documents"
```

- **Shown to:** historians, power users, logs.
- **Not used to:** block the answer, drop claims, or force a retry. The narrative and claims are always returned; the report is for transparency and trust.

### 7.2 Implementation

- New module: **v9_verify** — consumes grounded claims (from v9_grounding) and workspace, produces `V9VerificationReport`.
- Reuse V4-style checks where useful (e.g. supporting phrase present, span valid), but output only advisory fields; no `passed: bool` that gates the response.

---

## 8. Implementation Roadmap

### Phase A: Foundation (no UI/API switch yet)

1. **v9_types.py**  
   - `ResearchWorkspace`, `WorkspaceChunk`, `WorkspaceEntity`, `V9Claim`, `V9Synthesis`, `V9VerificationReport`, `V9Result`.

2. **v9_tools.py**  
   - Implement `search_chunks`, `fetch_chunks`, `expand_entities` (thin wrappers over existing `retrieval/agent/tools.py` and `retrieval/observations._load_chunk_texts` or new fetch in `retrieval/ops`).  
   - Optional: `append_note` / `append_hypothesis` as tools that update workspace.

3. **v9_workspace.py**  
   - Logic to merge tool results into `ResearchWorkspace`, dedupe chunks, format workspace for LLM context (e.g. “Current evidence: …”, “Notes: …”).

### Phase B: Reasoning Loop

4. **v9_runner.py**  
   - Main loop: system prompt + user question → LLM with tools → execute tool calls → update workspace → repeat until final answer or safety limit.  
   - Parse final assistant message for narrative + claims (structured output or JSON block).

5. **v9_prompts.py**  
   - System prompt (archive analyst, accuracy over speed, optional note/hypothesis instructions).  
   - Output format description (narrative + claims schema).

### Phase C: Post-Hoc Pipeline

6. **v9_grounding.py**  
   - Input: narrative, claims, workspace chunks (and optionally spans).  
   - Output: per-claim grounding (grounded / weak / unsupported) and citation refs.

7. **v9_verify.py**  
   - Input: grounded claims, workspace.  
   - Output: `V9VerificationReport` (advisory only).

8. **v9_runner.py (wire-up)**  
   - After synthesis: call v9_grounding → v9_verify → build `V9Result` (narrative, claims, report).

### Phase D: Integration

9. **Chat API**  
   - Feature flag or query param to call `run_v9_query` instead of `run_v7_query`; return narrative + optional claims + verification report in response.

10. **CLI**  
    - In `scripts/friday_cli.py`, add v9 path (e.g. `--agent v9`) that calls `run_v9_query` and prints narrative + report.

11. **Exports**  
    - `retrieval/agent/__init__.py`: export `V9Runner`, `run_v9_query`, `V9Result`, `ResearchWorkspace`, `V9VerificationReport`.

---

## 9. Mapping to Existing Codebase

| V9 Component | Reuse / New | Location / Notes |
|--------------|-------------|-------------------|
| `search_chunks` | Reuse | `tools.hybrid_search_tool`; optional filters from `SearchFilters`. |
| `fetch_chunks` | New (logic exists elsewhere) | Implement in v9_tools using same SQL as `observations._load_chunk_texts` or add `get_chunks_by_ids(conn, chunk_ids)` in `retrieval/ops` or `retrieval/lanes`. |
| `expand_entities` | Reuse | `entity_surfaces_tool`, `entity_mentions_tool` or combine in one wrapper. |
| Workspace chunk list | New | v9_workspace: merge and dedupe from tool results. |
| Notes/hypotheses | New | v9_workspace: append from tool or from parsed assistant message. |
| Synthesis parsing | New | v9_runner: parse JSON or structured output for narrative + claims. |
| Grounding | New + reuse ideas | v9_grounding; reuse phrase-in-chunk and span logic from v4_verify / v7_claim_enumerator. |
| Verification report | New | v9_verify; reuse checks from v4_verify but output only advisory. |
| Chat entrypoint | Reuse pattern | `backend/app/routes/chat.py`: add branch for v9 similar to `run_v7_chat_query`. |
| CLI | Reuse pattern | `scripts/friday_cli.py`: add v9 option alongside v6/v7. |

---

## 10. Out of Scope for V9 (Explicitly Removed)

- **Intent detection / query parsing** — no V6-style QueryParser or CONTROL/CONTENT.
- **Plan compiler** — no V3-style generate_plan or multi-step plan execution.
- **Entity linker as mandatory pre-step** — model may use `expand_entities` when it wants; no forced linking before retrieval.
- **Evidence bottleneck** — no fixed 40-span cap before synthesis; workspace has a high cap (e.g. 500 chunks).
- **Tournament / pairwise grading** — no Elo or bottleneck_grading_mode.
- **Lanes / modes** — no lexical vs entity “lanes”; one search tool (or one semantic + one lexical if we keep two).
- **Retries on verification failure** — verification is advisory only; no automatic retry of reasoning.
- **Stop gate that blocks answer** — no V7-style “drop uncited claims” or “citation_validation_passed” that hides content. Unsupported claims stay, marked.

---

## Summary

- **V9 = Reasoning Workspace Loop:** one strong model (GPT-4o), three tools, workspace accumulation, synthesis first (narrative + claims), then post-hoc grounding and advisory verification.
- **New modules:** `v9_types`, `v9_tools`, `v9_workspace`, `v9_runner`, `v9_grounding`, `v9_verify`, `v9_prompts`.
- **Reuse:** existing search/entity tools and DB layer, citation/verification ideas (without blocking behavior).
- **Integration:** chat route + CLI behind flag or `--agent v9`.

This plan is ready to be broken into tickets and implemented phase by phase.
