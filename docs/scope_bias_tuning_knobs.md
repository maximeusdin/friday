# Scope Bias Tuning Knobs

Reference for tuning retrieval diversity and citation mix after implementing the Scope Bias Diagnostics plan.

## Diagnostics

Set `DIAG_SCOPE_BIAS=1` to enable:

- **Search hits by collection**: Per-collection distribution of catalog hits from each search call
- **fulltext by collection**: Per-collection distribution of chunks in context
- **tokens by collection**: Token breakdown per collection in fulltext

Run DoD queries with diagnostics:

```bash
DIAG_SCOPE_BIAS=1 pytest tests/test_dod_scope_queries.py -v -s
```

## Tuning Knobs

### 1. Context packing

**File:** `retrieval/agent/v11_context.py`

| Knob | Default | Purpose |
|------|---------|---------|
| `chunk_char_cap` | from v9_context | Max chars per chunk in fulltext |
| `max_fulltext` | from v9_context | Max fulltext chunks shown |
| `max_catalog_rows` | from v9_context | Max catalog rows in context |
| `LIGHTWEIGHT_PEM_MAX_BUNDLES` | 6 | PEM blocks for V/V chunks |
| `LIGHTWEIGHT_PEM_MAX_LINES` | 40 | Lines in mention index |
| `LIGHTWEIGHT_PEM_MAX_CHARS` | 1600 | Chars in mention index |
| `LIGHTWEIGHT_PEM_MAX_ENTITIES` | 25 | Entities in PEM |
| `LIGHTWEIGHT_PEM_MAX_ALIASES_PER_ENTITY` | 8 | Aliases per entity |
| `LIGHTWEIGHT_PEM_MAX_TOKENS` | 600 | Token cap for PEM block |

**File:** `retrieval/pem_light.py` — caps at source for PEM generation.

### 2. Hybrid weighting

**File:** `retrieval/ops.py`

| Knob | Location | Purpose |
|------|----------|---------|
| `rrf_k` | hybrid_rrf() | RRF constant (default 50) |
| `top_n_vec` | hybrid_rrf() | Vector lane size |
| `top_n_lex` | hybrid_rrf() | Lexical lane size |

For codenames/acronyms: lexical may need more weight. Consider:
- RRF weights / k values
- Lexical normalization (stemming, punctuation, acronym behavior)
- All-caps token handling (lexical should carry more weight)

### 3. Grounded-claims thresholds

**Files:** `retrieval/agent/v9_grounding.py`, `retrieval/agent/v9_verify.py`

Audit for implicit thresholds:

- Minimum quote overlap
- Minimum citation count
- "Must have page" vs "chunk_id is fine"
- Dedup logic that collapses non-V/V claims as "similar"

### 4. Global entity lookup

**File:** `retrieval/agent/tools.py` — `_lookup_entity_global()`

- Deduping entities across multiple alias rows
- Returning enough candidates (and good metadata) for disambiguation
- When to skip (avoid wasting cycles on common words)

## Confidence check

Run before tuning to confirm non-V/V retrieval works:

```bash
python scripts/confidence_check_scope.py --query "Silvermaster network" --exclude venona,vassiliev
```

Expects non-empty hits from collections other than Venona/Vassiliev.
