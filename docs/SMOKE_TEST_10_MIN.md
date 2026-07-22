## Comprehensive Manual Smoke Test

Run from repo root (`C:\Users\maxim\friday`) in PowerShell.

### Prereqs

```powershell
# Set environment (adjust as needed)
$env:DATABASE_URL = "postgresql://neh:neh@localhost:5432/neh"
$env:OPENAI_API_KEY = "sk-..."  # Only needed for plan generation + summarization
$env:PYTHONIOENCODING = "utf-8"  # Avoid Unicode errors on Windows
```

- Backend running (if you want to hit the summarize API): `http://127.0.0.1:8000`

---

## Core Concepts

### What does "scope-only" mean?

**Scope-only queries** are entity-reference queries that do NOT perform semantic/vector search. They return chunks based purely on entity mention matches.

- **Example**: "Find mentions of Julius Rosenberg" → returns chunks containing `entity_mentions` for that entity
- **How it works**: The `expanded_text` is empty, so we skip `embed_query()` and directly fetch chunks from `scope_chunk_ids`
- **Key distinction**: This is NOT "aboutness" retrieval (semantic similarity). It's exact entity mention lookup.
- **Use case**: When you want every chunk that mentions an entity, not chunks semantically related to them.

If you want semantic retrieval for an entity, include descriptive terms: "Find discussions about Julius Rosenberg's role in espionage" (this triggers hybrid search).

---

### Plan Lifecycle States

| Status | Description | Transitions |
|--------|-------------|-------------|
| `proposed` | Initial state after `plan_query.py` | → `approved`, `rejected`, `needs_clarification`, `superseded` |
| `needs_clarification` | Ambiguous entities detected | → `superseded` (after clarification creates new plan) |
| `approved` | Ready for execution | → `executed` |
| `rejected` | User rejected the plan | Terminal state |
| `superseded` | Replaced by a clarified plan | Terminal state (see `parent_plan_id`) |
| `executed` | Successfully executed | Can be re-executed with `--force` |

**Failure metadata**: If execution fails, the plan stays in `approved` or `executed` state but `plan_json._metadata.last_error` contains error details and a partial `retrieval_run` is logged for audit.

---

## Core Tests (A-D)

### A) Ambiguous entity → clarify → approve → execute

1) Create a plan that should be ambiguous (example: "Rosenberg"):

```powershell
python scripts/plan_query.py --session 1 --text "Find mentions of Rosenberg"
```

2) Note the printed `Plan saved with ID: <PLAN_ID>`. Try to approve it (should be blocked):

```powershell
python scripts/approve_plan.py --plan-id <PLAN_ID>
```

3) Clarify it (pick a choice index). This creates a **new** plan ID:

```powershell
python scripts/clarify_plan.py --plan <PLAN_ID> --choice 1
# Output: "New plan created with ID: <NEW_PLAN_ID>"
```

4) Approve the **new** plan (not the original):

```powershell
python scripts/approve_plan.py --plan-id <NEW_PLAN_ID>
```

5) Execute it (should create `retrieval_run_id` and `result_set_id`):

```powershell
python scripts/execute_plan.py --plan-id <NEW_PLAN_ID>
```

6) Verify status + IDs:

```powershell
python scripts/admin_resolution_audit.py --plan-id <NEW_PLAN_ID>
```

---

### B) COUNT query → summarize blocked → materialize → summarize works

1) Create a COUNT-style query:

```powershell
python scripts/plan_query.py --session 1 --text "How many documents mention Silvermaster?"
```

2) Approve + execute:

```powershell
python scripts/approve_plan.py --plan-id <COUNT_PLAN_ID>
python scripts/execute_plan.py --plan-id <COUNT_PLAN_ID>
```

Expected:
- `result_set_id` is NULL
- `plan_json._metadata.execution_mode = "count"`
- output includes JSON with `mode=count`, `total_count`, `retrieval_run_id`

3) Confirm there is no result set yet:

```powershell
python scripts/admin_resolution_audit.py --plan-id <COUNT_PLAN_ID> --json
```

4) Summarize should fail (no result set):

```powershell
# This should return error or empty - no result_set_id to summarize
```

5) **Materialize** (drill-down to retrieval):

> **Note**: COUNT-mode plans are already `executed`. Materialization creates a new retrieval run, so you must use `--force --materialize` together.

```powershell
python scripts/execute_plan.py --plan-id <COUNT_PLAN_ID> --force --materialize
```

6) Re-check the plan; now it should have a `result_set_id` and `execution_mode = "retrieve"`:

```powershell
python scripts/admin_resolution_audit.py --plan-id <COUNT_PLAN_ID> --json
```

7) Call summarize endpoint (replace `<RESULT_SET_ID>`):

```powershell
$body = @{ summary_type = "brief" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/result-sets/<RESULT_SET_ID>/summarize" -ContentType "application/json" -Body $body
```

Expected: summary response with `summary`, `chunk_count`, `summarized_count`.

---

### C) --force re-exec → history appends + truncation flag

1) Pick any executed retrieval plan with a result set (from section A) and run force re-exec 25 times:

```powershell
1..25 | ForEach-Object { python scripts/execute_plan.py --plan-id <PLAN_ID> --force 2>&1 | Out-Null }
```

2) Inspect execution history:

```powershell
python scripts/admin_resolution_audit.py --plan-id <PLAN_ID> --json
```

Expected in `plan_json._metadata`:
- `executions` length capped at 20
- `executions_truncated: true` eventually

---

### D) Induced failure → failed retrieval_run + last_error recorded

This uses a built-in hook that fails after plan load (so audit logging can run).

1) Ensure you have an approved plan (or use `--force` if it's already executed). Then run:

```powershell
$env:NEH_INDUCE_FAILURE = "1"
python scripts/execute_plan.py --plan-id <PLAN_ID> --force
$env:NEH_INDUCE_FAILURE = ""
```

2) Verify failure recorded:

```powershell
python scripts/admin_resolution_audit.py --plan-id <PLAN_ID>
```

Expected:
- "Last Error" printed with type/message/time
- A `retrieval_runs` row created for the failed attempt (the script prints the failed run id)

---

## Extended Test Scenarios

### E) Entity-only query (scope-only execution)

Tests scope-only execution path where we find chunks by entity mentions without vector search.

```powershell
# 1) Create plan with just an entity reference
python scripts/plan_query.py --session 1 --text "Find all mentions of Julius Rosenberg"
# Note: Plan saved with ID: <PLAN_ID>

# 2) If ambiguous, clarify (choose "Julius Rosenberg" not "Ethel"):
python scripts/clarify_plan.py --plan <PLAN_ID> --choice <INDEX>
# Note: New plan created with ID: <NEW_PLAN_ID>

# 3) Approve + execute the NEW plan (after clarification)
python scripts/approve_plan.py --plan-id <NEW_PLAN_ID>
python scripts/execute_plan.py --plan-id <NEW_PLAN_ID>
```

Expected: 
- Returns chunks containing entity mentions (not empty!)
- Check the guardrail doesn't fire ("INVARIANT VIOLATION" warning)
- This is scope-only retrieval, NOT semantic search

---

### F) Phrase search

```powershell
# Exact phrase match
python scripts/plan_query.py --session 1 --text "Find documents containing the phrase 'atomic secrets'"

python scripts/approve_plan.py --plan-id <PLAN_ID>
python scripts/execute_plan.py --plan-id <PLAN_ID>
```

Expected: Plan contains `PHRASE` primitive, results contain the exact phrase.

---

### G) Date range filtering

```powershell
# 1) Absolute dates
python scripts/plan_query.py --session 1 --text "Find FBI reports about espionage from 1945 to 1950"

# 2) Relative dates (tests date_parser)
python scripts/plan_query.py --session 1 --text "Find documents from the last 5 years of the 1940s"

python scripts/approve_plan.py --plan-id <PLAN_ID>
python scripts/execute_plan.py --plan-id <PLAN_ID>
```

Expected: Plan contains `FILTER_DATE_RANGE` primitive with resolved dates in metadata.

---

### H) Collection filtering

```powershell
# Filter to specific collection
python scripts/plan_query.py --session 1 --text "Search for Silvermaster in the Venona collection"

python scripts/approve_plan.py --plan-id <PLAN_ID>
python scripts/execute_plan.py --plan-id <PLAN_ID>
```

Expected: Plan contains `FILTER_COLLECTION` primitive, results only from that collection.

---

### I) Co-occurrence queries

```powershell
# Enable best-guess to avoid ambiguity prompts for known entities
$env:ENTITY_BEST_GUESS = "1"

# CO_OCCURS_WITH: Find chunks where both entities appear together
python scripts/plan_query.py --session 1 --text "Find documents where both Harry Dexter White and Silvermaster are mentioned together"

$env:ENTITY_BEST_GUESS = ""

python scripts/approve_plan.py --plan-id <PLAN_ID>
python scripts/execute_plan.py --plan-id <PLAN_ID>
```

Expected: 
- Plan contains `CO_OCCURS_WITH` primitive with `entity_a` and `entity_b` (two entities, NOT single `entity_id`)
- Format: `{"type": "CO_OCCURS_WITH", "entity_a": 67279, "entity_b": 72144, "window": "chunk"}`
- Scope narrows to chunks containing both entities

---

### J) Search type modes

```powershell
# 1) Vector search (semantic)
python scripts/plan_query.py --session 1 --text "Find discussions about Soviet intelligence operations" 

# 2) Lexical search (keyword)
python scripts/plan_query.py --session 1 --text "Search for the exact term 'NKVD' using lexical search"

# 3) Hybrid (default)
python scripts/plan_query.py --session 1 --text "Find references to atomic bomb project"
```

Expected: Different `SET_SEARCH_TYPE` values (vector/lex/hybrid) based on query intent.

---

### K) Result set chaining (WITHIN_RESULT_SET)

> **Important**: `WITHIN_RESULT_SET` requires an existing result_set. If the LLM outputs this primitive, ensure the referenced result_set exists (from a prior executed plan).

```powershell
# 1) First, execute a broad query to create a result_set
python scripts/plan_query.py --session 1 --text "Find all documents about espionage"
python scripts/approve_plan.py --plan-id <PLAN_ID_1>
python scripts/execute_plan.py --plan-id <PLAN_ID_1>
# Note the result_set_id from output (e.g., "Result set created: ID=30")

# 2) Narrow down within that result set
python scripts/plan_query.py --session 1 --text "Within result set <RESULT_SET_ID>, find mentions of atomic"
python scripts/approve_plan.py --plan-id <PLAN_ID_2>
python scripts/execute_plan.py --plan-id <PLAN_ID_2>
```

Expected: 
- Second query has `WITHIN_RESULT_SET` primitive referencing the first result_set
- Results are a subset of the first query's results
- If `result_set_id` doesn't exist, execution will fail with clear error

---

### L) Best-guess mode for entity resolution

Skips ambiguity prompts by picking highest-confidence match automatically.

```powershell
# Enable best-guess mode
$env:ENTITY_BEST_GUESS = "1"

python scripts/plan_query.py --session 1 --text "Find mentions of Rosenberg"

$env:ENTITY_BEST_GUESS = ""
```

Expected:
- No `needs_clarification` flag
- `_metadata.resolution.entities` shows `is_best_guess: true`
- Plan is directly approvable

---

### M) Plan rejection

```powershell
# Create a plan
python scripts/plan_query.py --session 1 --text "Some test query"

# Reject it with a reason
python scripts/approve_plan.py --plan-id <PLAN_ID> --reject --reason "Not what I wanted"

# Show plan to verify rejection
python scripts/approve_plan.py --plan-id <PLAN_ID> --show
```

Expected: Status is `rejected`, reason is in `_metadata.rejection_reason`.

---

### N) Aggregation queries (GROUP_BY)

```powershell
# Count by collection
python scripts/plan_query.py --session 1 --text "How many mentions of Silvermaster are there, grouped by collection?"

python scripts/approve_plan.py --plan-id <PLAN_ID>
python scripts/execute_plan.py --plan-id <PLAN_ID>
```

Expected: COUNT mode with `group_by: "collection"`, returns buckets.

---

### O) Multi-entity query

```powershell
$env:ENTITY_BEST_GUESS = "1"
python scripts/plan_query.py --session 1 --text "Find documents mentioning both Harry Dexter White and Lauchlin Currie"
$env:ENTITY_BEST_GUESS = ""

python scripts/approve_plan.py --plan-id <PLAN_ID>
python scripts/execute_plan.py --plan-id <PLAN_ID>
```

Expected: Multiple `ENTITY` primitives or `CO_LOCATED`, scope intersection.

---

### P) Admin tools

```powershell
# 1) List recent plans with resolution info
python scripts/admin_resolution_audit.py --recent 10

# 2) Find plans using best-guess
python scripts/admin_resolution_audit.py --best-guess

# 3) Find plans with low-confidence resolutions
python scripts/admin_resolution_audit.py --low-confidence --threshold 0.7

# 4) Validate data integrity
python scripts/admin_validate_integrity.py
```

---

### Q) Conversation context (session continuity)

```powershell
# 1) Start a session with initial query
python scripts/plan_query.py --session 1 --text "Tell me about the Silvermaster spy ring"
python scripts/approve_plan.py --plan-id <PLAN_ID_1>
python scripts/execute_plan.py --plan-id <PLAN_ID_1>

# 2) Follow-up in same session (should have context)
python scripts/plan_query.py --session 1 --text "What about their connections to the Treasury Department?"
python scripts/approve_plan.py --plan-id <PLAN_ID_2>
python scripts/execute_plan.py --plan-id <PLAN_ID_2>

# 3) Drill down further
python scripts/plan_query.py --session 1 --text "Focus on Harry Dexter White specifically"
```

Expected: Later queries may reference context from earlier in the session.

---

## Production-Grade Checks

### R) Cold start session (no inherited state)

Tests that entity-only queries work on a fresh session with no prior history.

```powershell
# Use a brand new session ID that has never been used
$env:ENTITY_BEST_GUESS = "1"
python scripts/plan_query.py --session 99999 --text "Find mentions of Silvermaster"
$env:ENTITY_BEST_GUESS = ""

python scripts/approve_plan.py --plan-id <PLAN_ID>
python scripts/execute_plan.py --plan-id <PLAN_ID>
```

Expected:
- Non-zero chunks returned
- No "pipeline version" errors (scope-only path doesn't depend on inherited pipeline)
- Guardrail doesn't fire

---

### S) Large result set / API pagination

Tests that broad queries don't return megabytes of data in one response.

```powershell
# Run a broad query that matches many chunks
$env:ENTITY_BEST_GUESS = "1"
python scripts/plan_query.py --session 1 --text "Find all FBI documents"
$env:ENTITY_BEST_GUESS = ""

python scripts/approve_plan.py --plan-id <PLAN_ID>
python scripts/execute_plan.py --plan-id <PLAN_ID>
```

Check:
- Response returns quickly (< 30s)
- Only top-k chunks returned (default 20), not all matching chunks
- Preview text is truncated (not full chunk content)

If using the API:
```powershell
# Check API response size is reasonable
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/result-sets/<RESULT_SET_ID>" | ConvertTo-Json | Measure-Object -Character
```

Expected: Response < 100KB for typical result sets.

---

### T) Index verification (EXPLAIN ANALYZE)

Verifies that performance indexes were created and are being used.

```sql
-- Check indexes exist
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'entity_mentions' 
  AND indexname LIKE 'idx_entity_mentions%';

-- Expected indexes:
-- idx_entity_mentions_chunk_entity(chunk_id, entity_id)
-- idx_entity_mentions_document_entity(document_id, entity_id)

-- Verify CO_OCCURS_WITH document-window uses the index
EXPLAIN (ANALYZE, BUFFERS) 
SELECT c.id
FROM chunks c
WHERE EXISTS (
    SELECT 1 FROM entity_mentions em_a
    JOIN entity_mentions em_b ON em_b.document_id = em_a.document_id AND em_b.entity_id = 72144
    WHERE em_a.entity_id = 67279
    AND em_a.document_id = (SELECT cm.document_id FROM chunk_metadata cm WHERE cm.chunk_id = c.id LIMIT 1)
)
LIMIT 20;

-- Look for: "Index Scan using idx_entity_mentions_document_entity"
-- NOT: "Seq Scan on entity_mentions"
```

If indexes don't exist, run migration:
```powershell
python scripts/run_migration.py migrations/0040_cooccurrence_performance_indexes.sql
```

---

### U) Quick validation commands

```powershell
# Check database connectivity
python -c "import psycopg2, os; c=psycopg2.connect(os.environ['DATABASE_URL']); print('DB OK'); c.close()"

# Check entity resolver
python -c "from retrieval.entity_resolver import resolve_entity_name; print(resolve_entity_name('Silvermaster'))"

# Check primitives compile
python -c "from retrieval.primitives import *; p=TermPrimitive(value='test'); print(p)"

# Run primitive compilation tests
python tests/test_primitives_compilation.py --skip-db
```

---

## Troubleshooting

### Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Missing DATABASE_URL` | Env not set | `$env:DATABASE_URL = "..."` |
| `Decimal is not JSON serializable` | DB returns Decimal | Cast to `float()` in code |
| `column X does not exist` | Schema mismatch | Check migrations, use `cm.document_id` not `c.document_id` |
| `embed_query got empty input` | Entity-only query | Should use scope-only path (this is a bug if it fires) |
| `result_sets_chunk_ids_nonempty` | 0 hits returned | Normal for narrow scopes, no result_set created |
| `needs_clarification` blocking | Ambiguous entity | Use `clarify_plan.py` or `ENTITY_BEST_GUESS=1` |
| `result_sets are immutable` | Tried to update result_set | Use `--force` to create new run/set |
| `CO_OCCURS_WITH requires two entities` | Old plan format | Update to use `entity_a`/`entity_b` |

### Useful SQL queries

```sql
-- Recent plans
SELECT id, status, created_at, plan_json->>'utterance' 
FROM research_plans ORDER BY id DESC LIMIT 10;

-- Check result sets
SELECT rs.id, rs.retrieval_run_id, array_length(rs.chunk_ids, 1) as chunks
FROM result_sets rs ORDER BY rs.id DESC LIMIT 10;

-- Entity mention counts
SELECT e.canonical_name, COUNT(*) 
FROM entity_mentions em JOIN entities e ON e.id = em.entity_id 
GROUP BY e.canonical_name ORDER BY COUNT(*) DESC LIMIT 20;

-- Check plan execution details
SELECT id, status, 
       plan_json->'_metadata'->>'execution_mode' as mode,
       plan_json->'_metadata'->>'retrieval_run_id' as run_id,
       plan_json->'_metadata'->>'result_set_id' as result_set_id
FROM research_plans WHERE id = <PLAN_ID>;

-- Check for plans with errors
SELECT id, status,
       plan_json->'_metadata'->'last_error'->>'error_type' as error_type,
       plan_json->'_metadata'->'last_error'->>'message' as error_msg
FROM research_plans 
WHERE plan_json->'_metadata'->'last_error' IS NOT NULL
ORDER BY id DESC LIMIT 10;

-- Verify indexes exist
SELECT indexname FROM pg_indexes 
WHERE tablename = 'entity_mentions' 
ORDER BY indexname;
```
