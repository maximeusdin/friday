# V9 Session-Aware Retrieval System

## Overview

The V9 session system adds stateful, multi-turn retrieval to the investigation loop. It supports three execution paths:

| Intent | Description | Tool Calls | Evidence Source |
|--------|-------------|------------|-----------------|
| **NEW_RETRIEVAL** | New search + new evidence set | Yes (default 5) | Fresh retrieval |
| **FOLLOW_UP** | Answer from existing evidence | None | Existing evidence set |
| **THINK_DEEPER** | Resume paused run | Yes (extended budget) | Existing + new evidence |

## Architecture

```
User Message
     │
     ▼
┌─────────────┐
│   Router     │  LLM-based intent classification
│  (gpt-4.1-mini-2025-04-14)│  + heuristic query reference resolution
└──────┬──────┘
       │
   ┌───┴───┬──────────────┐
   ▼       ▼              ▼
NEW_RETR  FOLLOW_UP    THINK_DEEPER
   │       │              │
   │       │              │
   ▼       ▼              ▼
v9_runner  FTS join     v9_runner
(tools)   (no tools)   (resume + extend)
   │       │              │
   └───┬───┴──────────────┘
       ▼
  Evidence Set
  (persisted)
```

## Data Model

### 1. Sessions (`research_sessions`)

Extended with active pointers:

| Column | Type | Description |
|--------|------|-------------|
| `active_run_id` | BIGINT nullable | Currently active run |
| `active_evidence_set_id` | BIGINT nullable | Currently active evidence set |
| `active_run_status` | TEXT | `idle\|running\|paused\|completed\|failed` |
| `updated_at` | TIMESTAMPTZ | Last activity |

### 2. Runs (`v9_runs`)

Each retrieval query creates a run:

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | |
| `session_id` | BIGINT FK | Parent session |
| `query_text` | TEXT | Original question |
| `query_index` | INT | Monotonic per session (0, 1, 2...) |
| `label` | TEXT nullable | Short title (auto-generated) |
| `mode` | TEXT | `new_retrieval\|think_deeper` |
| `status` | TEXT | `running\|paused\|completed\|failed` |
| `last_step_idx` | INT | Last completed step index |
| `budgets_json` | JSONB | `{max_tool_calls: 5}` |
| `resume_state_json` | JSONB nullable | Controller state for think_deeper |
| `evidence_set_id` | BIGINT FK | Associated evidence set |
| `evidence_summary` | TEXT nullable | Auto-generated summary |
| `top_entities_json` | JSONB nullable | Top entities for query reference |

### 3. Evidence Sets (`evidence_sets`)

Bounded local corpus per run:

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | |
| `session_id` | BIGINT FK | Parent session |
| `run_id` | BIGINT FK nullable | Creating run |
| `is_active` | BOOLEAN | Currently active |

### 4. Evidence Items (`evidence_items`)

Chunk references within an evidence set. **No chunk text duplication** — joins to `chunks.text` at read time.

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | |
| `evidence_set_id` | BIGINT FK | Parent evidence set |
| `chunk_id` | BIGINT | FK to chunks(id) |
| `quote_text` | TEXT nullable | Short snippet (<=500 chars) |
| `locators_json` | JSONB | `{doc_id, page, source_label}` |
| `retrieval_score` | FLOAT nullable | Original score |
| `rank` | INT nullable | Position in results |
| `source_step_idx` | INT nullable | Which step produced this |
| `is_adjacency` | BOOLEAN | Adjacency-expanded chunk |
| `dedup_hash` | TEXT | `sha1(evidence_set_id:chunk_id)[:16]` |

**Constraints:**
- `UNIQUE(evidence_set_id, dedup_hash)` — one chunk per evidence set
- `INDEX(evidence_set_id, chunk_id)` — fast lookup
- `INDEX(evidence_set_id, rank)` — fast ranking

### 5. Run Steps (`v9_run_steps`)

Tool trace per run:

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | |
| `run_id` | BIGINT FK | Parent run |
| `step_idx` | INT | Monotonic per run |
| `lane` | TEXT nullable | e.g. 'search', 'fetch' |
| `tool_name` | TEXT | Tool that was called |
| `tool_args_json` | JSONB | Arguments |
| `tool_result_refs_json` | JSONB nullable | Summary of results |
| `elapsed_ms` | FLOAT nullable | Execution time |

**Constraints:**
- `UNIQUE(run_id, step_idx)` — monotonic steps

## Behaviors

### NEW_RETRIEVAL

1. Create a new `v9_runs` record with `query_index = last + 1`
2. Create a new `evidence_sets` record, mark as active
3. Update session active pointers
4. Execute `run_v9_query()` with default budget (5 tool calls)
5. Persist evidence items with adjacency expansion
6. Save resume state for potential think_deeper
7. Generate evidence summary + top entities for query reference
8. Set run status to `completed` (if sufficient) or `paused`

### FOLLOW_UP

1. Determine target `evidence_set_id` (via router)
2. Execute FTS search scoped to evidence set:
   ```sql
   SELECT ei.*, c.text AS chunk_text
   FROM evidence_items ei
   JOIN chunks c ON c.id = ei.chunk_id
   WHERE ei.evidence_set_id = $target
     AND c.tsv @@ plainto_tsquery('english', $query)
   ORDER BY ts_rank_cd(c.tsv, ...) DESC
   LIMIT 20
   ```
3. Answer using **only** returned chunks (full text via join)
4. Cite evidence item chunk IDs
5. If insufficient: suggest `think_deeper` or `new_retrieval`

**Invariants (hard rules):**
- FOLLOW_UP **never** calls retrieval tools
- Citations come **only** from the target evidence set
- Requires `target_evidence_set_id` to be non-null

### THINK_DEEPER

1. Load target run + resume state
2. Increase tool budget by `THINK_DEEPER_EXTRA_BUDGET` (10)
3. Continue from saved state
4. Append new steps with `step_idx > previous_last_step_idx`
5. Add new evidence items + adjacency expansion to same evidence set
6. Update resume state
7. Set status to `completed` or `paused`

**Invariants:**
- Same `run_id` as target
- `step_idx` is monotonic
- New evidence has `source_step_idx > previous_last_step_idx`

## Router

### Query Reference Resolution (Heuristic)

Resolves "that one / previous / the Silvermaster one" to a run/evidence set:

1. **Back-reference cues**: "that one", "previous", "last search" → most recent run
2. **Entity overlap**: If user mentions entity from `top_entities_json` → that run
3. **Label match**: If user mentions run label → that run
4. **Default**: Active run if exists

### Intent Classification (LLM)

Uses `gpt-4.1-mini-2025-04-14` with structured output:

- **new_retrieval**: New question requiring archive search
- **follow_up**: Clarification/detail about existing evidence
- **think_deeper**: Explicit request to extend search

**Conservative rule**: If unsure, prefer `NEW_RETRIEVAL` over incorrect `FOLLOW_UP`.

**Fast path**: Explicit think_deeper cues ("think deeper", "keep searching", "dig deeper") bypass LLM.

## Evidence Set as Bounded Local Corpus

### No Chunk Text Duplication

Evidence items store only `chunk_id`. Follow-ups join to `chunks.text` at read time.

Benefits:
- No duplicated text storage
- Always consistent with source
- Full chunk context available for follow-up answers

### Adjacency Expansion

When adding evidence chunks, also add:
- **Previous chunk** (same document)
- **Next chunk** (same document)

Neighbor chunks get:
- `is_adjacency = TRUE`
- `retrieval_score = NULL`
- Same `source_step_idx`

This improves pronoun resolution, context, and nearby definitions.

### Evidence Set Cap

Default cap: **200 items** per evidence set.

Pruning priority:
1. Keep non-adjacency chunks first
2. Keep best-ranked items
3. Keep newest items
4. Remove oldest adjacency chunks first

## Resume State

Minimal, stable JSON saved to `v9_runs.resume_state_json`:

```json
{
  "tool_calls_executed": 5,
  "model_turns": 3,
  "step_idx": 5,
  "max_tool_calls": 5,
  "seen_chunk_ids": [1, 2, 3, ...],
  "catalog_count": 50,
  "fulltext_count": 12,
  "entity_count": 3,
  "evidence_memory_updates": 2,
  "investigation_goal": "...",
  "investigation_gaps": ["..."]
}
```

## API

### POST `/sessions/{id}/v9/message`

**Request:**
```json
{
  "text": "Who was PAL?",
  "action": "default"
}
```

`action` can be `"default"` or `"think_deeper"`.

**Response:**
```json
{
  "intent": "new_retrieval",
  "answer": "...",
  "cited_chunk_ids": [123, 456],
  "confidence": "high",
  "active_run_id": 42,
  "active_run_status": "completed",
  "active_evidence_set_id": 17,
  "referenced_run_id": null,
  "referenced_evidence_set_id": null,
  "can_think_deeper": false,
  "routing_reasoning": "New question requiring archive search",
  "routing_confidence": 0.95,
  "suggestion": "",
  "elapsed_ms": 12340.5
}
```

## UI Integration

| Feature | Description |
|---------|-------------|
| Run history | Show `query_index + label` for each run in session |
| Think deeper button | Show when `can_think_deeper = true` |
| Evidence badge | "Answering from Q7: Silvermaster roster" when follow-up targets older run |
| Active run indicator | Show `active_run_status` (running/paused/completed) |

## Default Configuration

| Setting | Value | Environment Variable |
|---------|-------|---------------------|
| Max tool calls (default) | **5** | `V9_MAX_TOOL_CALLS` |
| Think deeper extra budget | 10 | — |
| Evidence set cap | 200 | — |
| Adjacency before/after | 1/1 | — |
| Router model | `gpt-4.1-mini-2025-04-14` | — |
| Follow-up model | `gpt-4.1-mini-2025-04-14` | `V9_FOLLOWUP_MODEL` |
| Follow-up max chunks | 20 | — |

## Files

| File | Purpose |
|------|---------|
| `migrations/0054_v9_sessions_evidence.sql` | Database schema |
| `retrieval/agent/v9_session.py` | Session, run, evidence set management + step persistence |
| `retrieval/agent/v9_router.py` | Intent classification + query reference resolution |
| `retrieval/agent/v9_followup.py` | Follow-up execution path |
| `retrieval/agent/v9_dispatch.py` | Main entry point — routes to execution paths |
| `retrieval/agent/v9_runner.py` | Core retrieval runner (unchanged, default 5 tool calls) |
| `backend/app/routes/chat.py` | API endpoint (`POST /{session_id}/v9/message`) |

## Implementation Order

1. Schema + data models (`0054_v9_sessions_evidence.sql`, `v9_session.py`)
2. Evidence set management (create, populate, adjacency, cap/prune, dedup)
3. Follow-up evidence-only search (`v9_followup.py`)
4. Router with LLM intent classification (`v9_router.py`)
5. Step persistence wrapper + resume state
6. Think deeper resume (`v9_dispatch.py`)
7. Main dispatch entry point + API endpoint
8. Query reference resolution (pronouns/previous queries)
9. Verifiers + invariant enforcement

## Acceptance Criteria

- [ ] Follow-up answers use full chunk text from `chunks.text`, scoped to evidence set, no tool calls
- [ ] Follow-up can reference prior runs ("that one") and correctly targets that run's evidence set
- [ ] Think deeper resumes same `run_id` from saved state and appends steps/evidence
- [ ] Adjacency expansion increases correctness for pronoun/definition follow-ups
- [ ] Default max tool calls is 5 for speed
- [ ] Router uses LLM for intent classification, falls back to heuristics
- [ ] Evidence sets are capped at 200 items with intelligent pruning
- [ ] All steps are persisted to `v9_run_steps` for observability
- [ ] Resume state is saved/loaded correctly for think_deeper
