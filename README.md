# Friday

Historical document retrieval and research system.

## Requirements

### Database Extensions

- **pgvector** - Required for vector similarity search (enabled in schema)
- **pg_trgm** - Required for soft lexical retrieval (qv2_softlex). Enable with:
  ```bash
  make enable-pg-trgm
  ```

## Core Primitives

### Result Sets

A **result_set** is a named, immutable snapshot of retrieval output. It stores an ordered list of chunk IDs from a specific retrieval run, allowing you to:

- **Cite specific results** - Save curated outputs for later reference
- **Compare over time** - Track how results change as data evolves
- **Export for analysis** - Generate CSV exports with full metadata

**Key properties:**
- Immutable: Once created, cannot be modified or deleted
- Provenance: Links to exact `retrieval_run` with full query parameters
- Replayable: Can regenerate from `retrieval_runs.returned_chunk_ids` and parameters

**Usage:**
```bash
# Save a retrieval run as a result_set
python scripts/save_result_set.py --run-id 42 --name "Oppenheimer search"

# Export to CSV
python scripts/export_result_set.py --id 1 --out results.csv
```

### Research Sessions

A **research_session** is a loose container for grouping related retrieval runs and result sets. It provides:

- **Organization** - Group queries and results by research topic
- **Trail tracking** - See all queries and outputs for a session
- **No constraints** - Sessions are optional; runs/sets can exist independently

**Key properties:**
- Optional: `retrieval_runs` and `result_sets` can exist without a session
- Loose coupling: Deleting a session doesn't delete its runs/sets (sets `session_id` to NULL)
- Named: Each session has a unique label for easy identification

**Usage:**
```sql
-- Create a session
INSERT INTO research_sessions (label) VALUES ('Atomic spies investigation');

-- Link runs/sets to session (via session_id FK)
UPDATE retrieval_runs SET session_id = 1 WHERE id IN (42, 43, 44);
UPDATE result_sets SET session_id = 1 WHERE id IN (1, 2, 3);
```

### Exporting Results

Export any result_set to CSV with full traceability:

```bash
python scripts/export_result_set.py --id <result_set_id> --out results.csv
```

**CSV columns:**
- `chunk_id` - Unique chunk identifier
- `document_id` - Document reference
- `collection_slug` - Collection name (e.g., "venona")
- `first_page_id`, `last_page_id` - Page range
- `snippet` - Text preview (default 350 chars)
- `matched_terms` - Query terms found in chunk (pipe-separated)
  - Exact matches: `term1|term2`
  - Approximate matches: `~term1|~term2` (prefix `~` indicates OCR/typo tolerance)
  - Mixed: `term1|~term2|term3`
- `match_type` (optional, with `--include-match-type`): `exact`, `approximate`, `semantic`, `mixed`

**Full workflow:**
```bash
# 1. Run query (creates retrieval_run)
python scripts/query_chunks.py "Oppenheimer" --mode hybrid

# 2. Save as result_set
python scripts/save_result_set.py --run-id <id> --name "Oppenheimer results"

# 3. Export to CSV
python scripts/export_result_set.py --id <result_set_id> --out oppenheimer.csv

# Export with match type column (shows exact vs approximate matches)
python scripts/export_result_set.py --id <result_set_id> --out oppenheimer.csv --include-match-type
```

## Guarantees

### Immutability

- **Result sets** cannot be updated or deleted (enforced by database triggers)
- **Retrieval runs** are append-only; logged once and never modified
- Ensures reproducibility: saved results remain stable over time

### Provenance

Every result_set links to its source `retrieval_run`, which contains:
- Original query text
- Expanded query (if expansion was used)
- Search type (lex/vector/hybrid)
- Pipeline version
- Exact chunk IDs returned
- Timestamp

**Traceability chain:**
```
result_set → retrieval_run → query parameters → chunk_ids → chunks → documents → pages
```

### Replayability

Any result_set can be regenerated because:
1. `retrieval_runs` stores exact `returned_chunk_ids`
2. All query parameters are logged (filters, expansion, search type)
3. Pipeline versions are tracked
4. Chunks are immutable (pipeline_version determines content)

**Replay query:**
```sql
-- Get original chunk IDs from retrieval_run
SELECT returned_chunk_ids FROM retrieval_runs WHERE id = <retrieval_run_id>;

-- Or get from result_set
SELECT chunk_ids FROM result_sets WHERE id = <result_set_id>;
```

## See Also

- `docs/workflow_example.md` - Complete workflow guide
- `docs/retrieval_logging.md` - Retrieval logging details
- `docs/goals_status.md` - System goals and status
