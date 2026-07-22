# Mention Index Implementation Plan

## Overview

This plan outlines the implementation of a deterministic, auditable mention index system that extracts entity and date mentions from chunk text and stores them as structured evidence links. This enables historian-grade workflows (counting, co-occurrence, timeline filters) without relying on embeddings for membership queries.

## Design Principles

1. **Assistive, not authoritative**: Extraction proposes candidates; only deterministic gates or human approval persist "mentions"
2. **No hidden state**: Pipelines are replayable; outputs are idempotent; thresholds are fixed and logged
3. **Evidence, not interpretation**: Mentions are "this string appears here," never "this implies a relationship"

## Implementation Phases

### Phase 1: Database Schema (date_mentions table)

**Objective**: Create the `date_mentions` table mirroring `entity_mentions` structure.

**Tasks**:
1. Create migration file `migrations/0020_date_mentions.sql`
2. Define table schema with all required fields:
   - `id` (bigserial primary key)
   - `chunk_id` (references chunks, cascade delete)
   - `document_id` (denormalized, references documents)
   - `surface` (exact substring as seen)
   - `start_char`, `end_char` (optional, nullable for v1)
   - `date_start`, `date_end` (date fields, equal for single-day)
   - `precision` (day, month, year, range, unknown)
   - `confidence` (real, 0.0-1.0)
   - `method` (regex_day, regex_month, regex_year, regex_range, human)
   - `created_at` (timestamptz)
3. Create indexes:
   - `(chunk_id)`
   - `(document_id)`
   - `(date_start, date_end)`
   - `(surface)` (optional, for review workflows)
4. Add uniqueness constraint for idempotency:
   - `(chunk_id, surface, date_start, date_end, method)` - prevents duplicates
5. Add table comments for documentation

**Deliverable**: Migration file that can be applied cleanly

**Dependencies**: None

**Estimated Time**: 1-2 hours

---

### Phase 2: Entity Mention Extraction Pipeline

**Objective**: Create `extract_entity_mentions.py` that scans chunks for exact alias matches and persists to `entity_mentions`.

**Tasks**:
1. Create `scripts/extract_entity_mentions.py` with CLI interface:
   - `--collection <slug>` - filter by collection
   - `--document-id <id>` - filter by document
   - `--chunk-id-range <start>:<end>` - filter by chunk ID range
   - `--dry-run` - print counts only, no inserts
   - `--limit <n>` - limit number of chunks processed
   - `--since-run-id <id>` - process chunks since last run (optional)
   - Progress output (chunks processed, mentions found, rate)

2. Core extraction logic:
   - Query chunks based on scope filters
   - For each chunk:
     - Normalize chunk text using `normalize_alias()` from `entity_resolver`
     - Query all `entity_aliases` with their `alias_norm` values
     - Find exact substring matches in normalized chunk text
     - For each match:
       - Extract `surface` (original substring from chunk)
       - Compute `start_char`, `end_char` (optional, can be NULL for v1)
       - Get `document_id` from `chunk_metadata` (denormalized)
       - Set `method='alias_exact'`
       - Set `confidence=1.0` (fixed for exact matches)
       - Set `matched_rule_id=NULL` (not used for alias_exact)

3. Idempotency handling:
   - Use `ON CONFLICT DO NOTHING` on unique constraint
   - Or check existence before insert (less efficient but clearer)
   - Unique constraint: `(chunk_id, entity_id, surface, method)` or similar

4. Batch processing:
   - Process chunks in batches (e.g., 100 at a time)
   - Use transactions for each batch
   - Log progress periodically

5. Error handling:
   - Handle missing `document_id` in chunk_metadata gracefully
   - Log warnings for chunks without metadata
   - Continue processing on individual chunk errors

**Deliverable**: Working CLI script that can extract entity mentions deterministically

**Dependencies**: Phase 1 (schema), existing `entity_mentions` table, `entity_aliases` table

**Estimated Time**: 4-6 hours

---

### Phase 3: Date Mention Extraction Pipeline

**Objective**: Create `extract_date_mentions.py` that extracts explicit date expressions using deterministic regex patterns.

**Tasks**:
1. Create `scripts/extract_date_mentions.py` with same CLI interface as entity extractor:
   - `--collection <slug>`
   - `--document-id <id>`
   - `--chunk-id-range <start>:<end>`
   - `--dry-run`
   - `--limit <n>`
   - `--since-run-id <id>`
   - Progress output

2. Date parsing module (`retrieval/date_parser.py`):
   - Create deterministic regex patterns for:
     - Full dates: "23 June 1945", "June 23, 1945", "1945-06-23"
     - Month-year: "June 1945", "Jun 1945", "06/1945"
     - Year only: "1945"
     - Ranges: "June–July 1945", "1943–1945", "1943-1945"
   - Parse to `(date_start, date_end, precision)`:
     - Single day: `date_start = date_end`, `precision='day'`
     - Month: `date_start = first day of month`, `date_end = last day`, `precision='month'`
     - Year: `date_start = Jan 1`, `date_end = Dec 31`, `precision='year'`
     - Range: `date_start = start`, `date_end = end`, `precision='range'`
   - Fixed confidence values:
     - `day=1.0` (high confidence for explicit dates)
     - `month=0.8` (slightly lower for month-only)
     - `year=0.6` (lower for year-only, more ambiguous)
     - `range=0.9` (high for explicit ranges)
   - Method names: `regex_day`, `regex_month`, `regex_year`, `regex_range`

3. Core extraction logic:
   - Query chunks based on scope filters
   - For each chunk:
     - Apply date regex patterns to chunk text
     - For each match:
       - Extract `surface` (original substring)
       - Parse to `(date_start, date_end, precision)`
       - Get `document_id` from `chunk_metadata`
       - Set `method` based on pattern matched
       - Set `confidence` based on precision
       - `start_char`, `end_char` can be NULL for v1

4. Idempotency:
   - Unique constraint: `(chunk_id, surface, date_start, date_end, method)`
   - Use `ON CONFLICT DO NOTHING`

5. Edge cases:
   - Handle ambiguous dates (e.g., "06/07/1945" - US vs UK format)
     - Default to US format (MM/DD/YYYY) for MVP
     - Log ambiguous cases for review
   - Handle OCR noise (e.g., "I945" instead of "1945")
     - Be conservative: only match clear patterns
     - Lower confidence for noisy matches if detected

**Deliverable**: Working CLI script + date parsing module

**Dependencies**: Phase 1 (date_mentions schema)

**Estimated Time**: 6-8 hours

---

### Phase 4: Review/Adjudication CLI

**Objective**: Create minimal review workflow for ambiguous/non-deterministic cases.

**Tasks**:
1. Create migration `migrations/0021_mention_review_queue.sql`:
   - Table: `mention_review_queue`
   - Fields:
     - `id` (bigserial primary key)
     - `mention_type` (entity/date)
     - `chunk_id` (references chunks)
     - `surface` (text found)
     - `context_excerpt` (short slice around mention, ~200 chars)
     - `candidates` (jsonb - for entity: `[{entity_id, canonical_name, score}]`, for date: `[{date_start, date_end, precision}]`)
     - `method` (text - e.g., 'alias_exact_collision', 'ner_v1', 'fuzzy_proposal')
     - `method_version` (text - e.g., 'v1', 'ner_v1', for idempotency and versioning)
     - `status` (pending, accepted, rejected)
     - `decision` (jsonb - accepted entity_id or date range)
     - `created_at`, `reviewed_at`
   - Indexes: `(status)`, `(mention_type, status)`, `(chunk_id)`
   - Uniqueness: `(chunk_id, surface, method, method_version)` for idempotency

2. Create `scripts/review_mentions.py` CLI with CSV export/import workflow:
   - Commands:
     - `export [--output <file.csv>] [--status pending]` - export pending reviews to CSV
       - CSV columns:
         - `review_id` - unique ID for this review
         - `mention_type` - entity/date
         - `chunk_id` - chunk containing the mention
         - `surface` - text found in chunk
         - `context_excerpt` - surrounding text (~200 chars)
         - `candidates` - JSON string of candidates (for entity: `[{entity_id, canonical_name, score}]`, for date: `[{date_start, date_end, precision}]`)
         - `decision` - (empty, to be filled by human)
         - `notes` - (empty, optional notes field)
       - For entity reviews: include candidate entities with canonical_name, entity_id, and match score
       - For date reviews: include parsed date options with date_start, date_end, precision
     - `import --input <file.csv>` - import reviewed CSV and apply decisions
       - Reads CSV with `review_id` and `decision` columns
       - Decision format:
         - Entity: `accept:<entity_id>` or `reject` or `new_entity:<canonical_name>:<entity_type>`
         - Date: `accept:<date_start>:<date_end>:<precision>` or `reject`
       - Updates `mention_review_queue` with decisions
       - For accepted entities: creates `entity_mentions` records
       - For accepted dates: creates `date_mentions` records
       - Updates `status` to 'accepted' or 'rejected' and sets `reviewed_at`
     - `stats` - show counts of pending/accepted/rejected reviews
   
   - CSV workflow:
     1. Human runs `review_mentions.py export --output reviews.csv`
     2. Opens CSV in spreadsheet (Excel, Google Sheets, etc.)
     3. Reviews each row, fills in `decision` column
     4. Optionally adds `notes` for context
     5. Saves CSV
     6. Runs `review_mentions.py import --input reviews.csv`
     7. System applies decisions and updates database

3. Integration with extraction:
   - Extraction scripts can optionally write ambiguous cases to review queue
   - For MVP, focus on entity extraction writing fuzzy matches to queue
   - Date extraction can write ambiguous parses to queue

**Deliverable**: Review queue table + CSV export/import CLI tool

**Dependencies**: Phase 2, Phase 3 (extraction scripts)

**Estimated Time**: 4-6 hours

**CSV Format Details**:
- Export includes all information needed for human review in spreadsheet-friendly format
- Decision column uses simple text format for easy editing
- Supports batch processing of many reviews at once
- Import validates decisions and provides error reporting for invalid entries
- Supports adding notes/comments for audit trail

---

### Phase 5: Plan/Compiler Integration

**Objective**: Ensure plan execution can use mention-driven primitives (ENTITY and DATE_RANGE).

**Tasks**:
1. Update `compile_primitives_to_scope()` in `retrieval/primitives.py`:
   - For `EntityPrimitive`:
     - Generate SQL that joins `entity_mentions`:
       ```sql
       c.id IN (
         SELECT chunk_id FROM entity_mentions 
         WHERE entity_id = %s
       )
       ```
     - Apply this as a scope constraint
   - For `FilterDateRangePrimitive`:
     - Option 1: Keep existing chunk_metadata filtering (date_min/date_max)
     - Option 2: Add mention-based filtering:
       ```sql
       c.id IN (
         SELECT chunk_id FROM date_mentions
         WHERE date_start <= %s AND date_end >= %s
       )
       ```
     - For MVP: Support both (mention-based is more precise but may have lower recall)

2. Update `execute_plan.py`:
   - Ensure compiled scope constraints include mention-based filters
   - Test with a plan containing `ENTITY(entity_id=...)` primitive

3. Testing:
   - Create test plan with `ENTITY` primitive
   - Extract mentions for test entity
   - Execute plan and verify chunks returned match mentions

**Deliverable**: Plan execution supports ENTITY and DATE_RANGE via mentions

**Dependencies**: Phase 2, Phase 3 (extraction), existing plan execution

**Estimated Time**: 3-4 hours

---

### Phase 6: Tests (Regression-Grade)

**Objective**: Add comprehensive tests for idempotency, scope correctness, and end-to-end queries.

**Tasks**:
1. Create `tests/test_mention_extraction.py`:
   - Test idempotency:
     - Run `extract_entity_mentions.py` twice on same scope → row count unchanged
     - Run `extract_date_mentions.py` twice → row count unchanged
   - Test scope correctness:
     - Extract for `--collection venona` → no mentions outside venona
     - Extract for `--document-id X` → no mentions outside document X
   - Test end-to-end query:
     - Seed test entity + alias
     - Extract mentions in bounded collection
     - Query via `ENTITY(entity_id=...)` → returns chunks with mentions
   - Test audit fields:
     - Verify `method`, `confidence`, `created_at` are populated
     - Verify `document_id` is denormalized correctly

2. Create `tests/test_date_parsing.py`:
   - Test date patterns:
     - "23 June 1945" → (1945-06-23, 1945-06-23, day)
     - "June 1945" → (1945-06-01, 1945-06-30, month)
     - "1945" → (1945-01-01, 1945-12-31, year)
     - "1943–1945" → (1943-01-01, 1945-12-31, range)
   - Test edge cases:
     - Ambiguous formats
     - OCR noise
     - Invalid dates

3. Integration test:
   - Create test fixture with known entities and dates
   - Run extraction
   - Verify mentions match expected results
   - Execute plan with ENTITY primitive
   - Verify results

**Deliverable**: Test suite with >90% coverage of extraction logic

**Dependencies**: All previous phases

**Estimated Time**: 4-6 hours

---

## File Structure

```
migrations/
  0020_date_mentions.sql          # Phase 1
  0021_mention_review_queue.sql    # Phase 4

scripts/
  extract_entity_mentions.py       # Phase 2
  extract_date_mentions.py         # Phase 3
  review_mentions.py               # Phase 4

retrieval/
  date_parser.py                   # Phase 3 (new module)
  primitives.py                    # Phase 5 (update compile_primitives_to_scope)

tests/
  test_mention_extraction.py       # Phase 6
  test_date_parsing.py             # Phase 6
```

## Implementation Order

1. **Phase 1** (Schema) - Foundation, no dependencies
2. **Phase 2** (Entity extraction) - Can be done in parallel with Phase 3
3. **Phase 3** (Date extraction) - Can be done in parallel with Phase 2
4. **Phase 4** (Review queue) - Depends on Phase 2/3
5. **Phase 5** (Compiler integration) - Depends on Phase 2/3
6. **Phase 6** (Tests) - Depends on all phases

## Acceptance Criteria Checklist

- [ ] `date_mentions` table exists with indexes + uniqueness rule; migration applied cleanly
- [ ] `extract_entity_mentions.py` can populate `entity_mentions` from alias-exact matches
- [ ] `extract_date_mentions.py` can populate `date_mentions` using deterministic parsing rules
- [ ] Both extractors are idempotent and support `--dry-run` plus scoped runs
- [ ] Review queue + `review_mentions.py` exists with CSV export/import workflow for human adjudication
- [ ] Compiler/executor can run at least one mention-driven primitive end-to-end (ENTITY and/or DATE_RANGE)
- [ ] Test suite includes idempotency + scope correctness + end-to-end mention query

## Stretch Goals (if time remains)

- Add "proposal-only" lane for fuzzy alias matches (write to review queue, not to mentions)
- Store optional `start_char`/`end_char` spans for dates (if cheap to compute reliably)
- Add simple "top entities by mentions" and "mentions per year" debug commands (SQL/report)

## Risk Mitigation

1. **OCR noise**: Prioritize precision over recall; use conservative patterns
2. **Performance**: Use batch processing and indexes; test on large collections
3. **Ambiguity**: Log ambiguous cases to review queue; don't auto-resolve
4. **Idempotency**: Use database constraints; test thoroughly with duplicate runs

## Notes

- All extraction is offline/on-ingest; query execution reads only persisted mentions
- No runtime NER; mention extraction is a separate pipeline
- All thresholds and methods are fixed and logged for auditability
- Focus on precision over recall for MVP; improvements come via review queue
- CSV-based adjudication enables batch review in spreadsheets, better for processing many cases at once
