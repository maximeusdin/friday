# 2-Tier Alias System Implementation Summary

## ✅ Completed Implementation

All phases of the 2-tier alias system have been implemented.

---

## Phase 1: Immediate Fixes ✅

### 1.1 Alias Type-Based `is_auto_match` ✅
**File**: `concordance/ingest_concordance_tab_aware.py` - `ensure_alias()`

- **Auto-match allowed**: `canonical`, `original_form`, `bracket_variant`, `cover_name`, `covername_from_body`
- **Conditionally allowed**: `head_syn` (only if not generic word)
- **Never auto-match**: `definition`, `scoped_label`, `see`, `work_name`, `spelling_variant`

### 1.2 Single/2-Letter Covername Restrictions ✅
**File**: `concordance/ingest_concordance_tab_aware.py` - `ensure_alias()`

- Single letters: Only allowed if quoted/bracketed in original alias
- 2-letter tokens: Only allowed if ALLCAPS and in known acronym list (KGB, GRU, MGB, etc.)

### 1.3 Generic Label Entity Restrictions ✅
**File**: `concordance/ingest_concordance_tab_aware.py` - `ensure_alias()`

- Standalone generic labels (`president`, `general`, `group`, `ref`, etc.) are not auto-matched

---

## Phase 2: Derived Alias Types ✅

### 2.1 Derived Surname Aliases ✅
**File**: `scripts/derive_surname_aliases.py`

- Creates `derived_last_name` aliases for person entities
- Extracts last token from multi-word `canonical`/`original_form` aliases
- Only if surname >= 4 chars and not generic word
- Sets: `alias_class='person_last'`, `is_auto_match=true`, `min_chars=4`, `match_case='titlecase_only'`

**Usage**:
```bash
python scripts/derive_surname_aliases.py --source-slug "vassiliev_venona_index_full_capitalized"
```

### 2.2 Derived Acronym Aliases ✅
**File**: `scripts/derive_acronym_aliases.py`

- Creates `derived_acronym` aliases for org entities
- Matches canonical names against known acronym expansion patterns
- Sets: `alias_class='org'`, `is_auto_match=true`, `min_chars=2`, `match_case='upper_only'`

**Usage**:
```bash
python scripts/derive_acronym_aliases.py --source-slug "vassiliev_venona_index_full_capitalized"
```

---

## Phase 3: Real `requires_context` Implementation ✅

### 3.1 Mark Ambiguous Covernames ✅
**File**: `scripts/extract_entity_mentions.py` - `load_all_aliases()`

- Ambiguous covernames (`link`, `achievement`, `master`, `group`, `general`, etc.) are marked with `requires_context='codename_like'`

### 3.2 Context Check Function ✅
**File**: `scripts/extract_entity_mentions.py` - `check_codename_context()`

- Checks for:
  - Quoted/bracketed: `"LINK"`, `['LINK']`, `[LINK]`, `(LINK)`
  - ALLCAPS surface text
  - Codename markers nearby (±60 chars): `cover name`, `codenamed`, `cryptonym`, `alias`, `aka`, `code name`

### 3.3 Context Gate Enforcement ✅
**File**: `scripts/extract_entity_mentions.py` - `find_matches_for_chunk()`

- Applied to both exact matches (Tier 1) and partial/fuzzy matches (Tier 2/3)
- Rejects matches that don't satisfy context requirements

---

## Phase 4: Frequency-Based Auto-Downgrade ✅

### 4.1 DF Computation Script ✅
**File**: `scripts/compute_alias_frequency.py`

- Computes document frequency (DF) **per document_id** for each alias_norm
- Stores in `alias_stats` table: `(alias_norm, document_id, df_chunks, total_chunks, df_percent)`
- Uses `ILIKE` for approximate matching (fast, good enough for frequency estimation)

**Usage**:
```bash
python scripts/compute_alias_frequency.py --collection venona --source-slug "vassiliev_venona_index_full_capitalized"
```

### 4.2 DF-Based Rules Application ✅
**File**: `scripts/extract_entity_mentions.py` - `load_all_aliases()`

- **DF > 0.5%**: Single-token aliases disabled unless ALLCAPS acronym or whitelisted
- **DF > 2%**: All aliases disabled unless whitelisted (KGB, GRU, MGB, NKVD, USSR, FBI, CIA, NSA, SIS, MI6)
- Uses maximum DF across all documents for initial filtering

---

## Phase 5: Refine Partial Matching ✅

### 5.1 Updated Partial Match Index ✅
**File**: `scripts/extract_entity_mentions.py` - `build_partial_match_index()`

- **Only indexes**:
  - Derived surname aliases (`alias_type='derived_last_name'`)
  - Derived acronym aliases (`alias_type='derived_acronym'`)
  - Last token of `person_full` aliases (if >= 4 chars, not generic)

- **No longer indexes**: Generic words from multi-word aliases (prevents "affairs" matching "Internal Affairs")

---

## Database Schema Changes

### New Table: `alias_stats`
```sql
CREATE TABLE alias_stats (
    alias_norm TEXT NOT NULL,
    document_id INTEGER NOT NULL,
    df_chunks INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    df_percent NUMERIC NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (alias_norm, document_id)
);
CREATE INDEX idx_alias_stats_alias_norm ON alias_stats(alias_norm);
```

### Updated: `entity_aliases`
- `is_auto_match` column now set based on alias_type rules during ingest
- `requires_context` column used for ambiguous covernames
- `alias_type` column used to identify derived aliases

---

## Workflow

### 1. Ingest Concordance
```bash
python concordance/ingest_concordance_tab_aware.py \
  --pdf data/raw/index/Vassiliev_Notebooks_and_Venona_Index-Concordance.pdf \
  --source-slug "vassiliev_venona_index_full_capitalized_v2" \
  --source-title "Vassiliev Venona Index (2-Tier System)" \
  --segment layout
```

### 2. Derive Surname and Acronym Aliases
```bash
python scripts/derive_surname_aliases.py --source-slug "vassiliev_venona_index_full_capitalized_v2"
python scripts/derive_acronym_aliases.py --source-slug "vassiliev_venona_index_full_capitalized_v2"
```

### 3. Compute DF Statistics
```bash
python scripts/compute_alias_frequency.py \
  --collection venona \
  --source-slug "vassiliev_venona_index_full_capitalized_v2"
```

### 4. Extract Entity Mentions
```bash
python scripts/extract_entity_mentions.py \
  --collection venona \
  --concordance-source-slug "vassiliev_venona_index_full_capitalized_v2" \
  --enable-partial \
  --summary-csv match_summary.csv
```

### 5. Export CSV Files
```bash
python scripts/export_concordance_data.py \
  --source-slug "vassiliev_venona_index_full_capitalized_v2" \
  --output-dir concordance_export
```

---

## Expected Impact

### Precision Improvements
- ✅ Eliminates `definition` type aliases (e.g., "soviet", "american", "light bomber")
- ✅ Eliminates single-letter junk (I, A, M) unless quoted/acronym
- ✅ Eliminates generic labels (president, general, group) as standalone matches
- ✅ Eliminates high-DF generic words (>0.5% or >2% depending on token count)
- ✅ Eliminates ambiguous covernames without context signals

### Recall Preservation
- ✅ Derived surname aliases enable last-name-only matching
- ✅ Derived acronym aliases enable acronym matching
- ✅ Partial matching still works for surnames (bounded risk)
- ✅ Context gating preserves legitimate codename mentions (they usually have signals)

---

## Testing Checklist

- [ ] Run ingest with new source slug
- [ ] Run surname derivation script
- [ ] Run acronym derivation script
- [ ] Run DF computation script
- [ ] Run extraction and verify reduced junk matches
- [ ] Check `match_summary.csv` for precision improvements
- [ ] Verify legitimate entities still match (recall check)

---

## Notes

- DF computation uses `ILIKE` for speed - this is approximate but sufficient for frequency estimation
- DF thresholds (0.5%, 2%) are configurable - adjust based on your corpus size
- Whitelist for high-DF acronyms can be expanded as needed
- Context gating can be extended to support other `requires_context` types in the future
