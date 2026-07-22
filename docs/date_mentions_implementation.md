# Date Mentions Extraction - Implementation Summary

## Overview

Implemented deterministic date extraction from chunk text, mirroring the `entity_mentions` pattern but for dates.

## Components

### 1. Database Schema

**Migration**: `migrations/0020_date_mentions.sql` (already exists)

**Table**: `date_mentions`
- `id` (bigserial primary key)
- `chunk_id` (references chunks)
- `document_id` (denormalized for convenience)
- `surface` (exact substring as seen)
- `start_char`, `end_char` (optional character positions)
- `date_start`, `date_end` (DATE fields, equal for single-day dates)
- `precision` ('day', 'month', 'year', 'range', 'unknown')
- `confidence` (0.0-1.0, fixed per method)
- `method` ('regex_day', 'regex_month', 'regex_year', 'regex_range', 'human')
- `created_at` (timestamp)

**Indexes**:
- `(chunk_id)`
- `(document_id)`
- `(date_start, date_end)`
- `(surface)` (for review workflows)
- `(method, confidence)`

**Uniqueness**: `(chunk_id, surface, date_start, date_end, method)` - prevents duplicates

### 2. Extraction Script

**File**: `scripts/extract_date_mentions.py`

**Features**:
- Deterministic regex patterns (no ML dependencies)
- Fixed confidence scores per pattern type
- Idempotent inserts (ON CONFLICT DO NOTHING)
- Same CLI ergonomics as `extract_entity_mentions.py`

**Supported Patterns**:

1. **Day precision** (confidence: 1.0):
   - "23 June 1945"
   - "June 23, 1945"
   - "23rd June 1945"
   - "23/06/1945" (tries both DD/MM/YYYY and MM/DD/YYYY)

2. **Month precision** (confidence: 0.8):
   - "June 1945"
   - "Jun 1945"

3. **Year precision** (confidence: 0.6):
   - "1945"
   - "1943"

4. **Ranges** (confidence: 0.9):
   - "June–July 1945"
   - "1943–1945"

**Date Range Handling**:
- Single-day dates: `date_start = date_end`
- Month dates: `date_start = first day of month`, `date_end = last day of month`
- Year dates: `date_start = Jan 1`, `date_end = Dec 31`
- Ranges: `date_start` and `date_end` span the range

## Usage

### Basic Usage

```bash
# Dry run to see what would be extracted
python scripts/extract_date_mentions.py \
  --collection venona \
  --dry-run \
  --limit 10

# Extract dates
python scripts/extract_date_mentions.py \
  --collection venona

# Filter by document
python scripts/extract_date_mentions.py \
  --document-id 123

# Filter by chunk range
python scripts/extract_date_mentions.py \
  --chunk-id-range 1000:2000

# Test on sample text
python scripts/extract_date_mentions.py \
  --test-text "The meeting was held on 23 June 1945 in Moscow."
```

### CLI Options

- `--collection <slug>`: Filter by collection slug
- `--document-id <id>`: Filter by document ID
- `--chunk-id-range <start:end>`: Filter by chunk ID range
- `--dry-run`: Print counts only, no inserts
- `--show-samples`: Show sample date mentions
- `--max-samples <n>`: Maximum samples to show (default: 10)
- `--limit <n>`: Limit number of chunks processed
- `--batch-size <n>`: Process in batches (default: 100)
- `--test-text <text>`: Test extraction on provided text

## Implementation Details

### Pattern Matching Order

1. **Ranges first** (more specific, checked before single dates)
2. **Day precision** (most specific single dates)
3. **Month precision**
4. **Year precision** (least specific)

This ensures "June–July 1945" is matched as a range, not as two separate month dates.

### Idempotency

Uses `ON CONFLICT (chunk_id, surface, date_start, date_end, method) DO NOTHING` to prevent duplicates when re-running extraction.

### Confidence Scores

Fixed per method (not dynamic):
- `regex_day`: 1.0
- `regex_month`: 0.8
- `regex_year`: 0.6
- `regex_range`: 0.9

These reflect the reliability of each pattern type.

## Example Output

```
Found 1000 chunks to process
Processing chunks in batches of 100...
  Batch 1/10: processed 100/1000 chunks, inserted 45 date mentions (+45 this batch)
  Batch 2/10: processed 200/1000 chunks, inserted 89 date mentions (+44 this batch)
  ...

======================================================================
SUMMARY:
  ✅ Processed:           1,000 chunks
  📅 Date mentions:         450
======================================================================
```

## Integration with Entity Extraction

Date extraction is **separate** from entity extraction:
- `extract_entity_mentions.py`: Extracts entities (persons, orgs, places, covernames)
- `extract_date_mentions.py`: Extracts dates

Both can run on the same chunks - they complement each other.

## Database Migration

Run migration:
```bash
make date-mentions
```

Or directly:
```bash
docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0020_date_mentions.sql
```

## Next Steps

1. **Run migration** (if not already done)
2. **Test extraction** on sample collection:
   ```bash
   python scripts/extract_date_mentions.py --collection venona --dry-run --limit 100
   ```
3. **Review samples** to verify pattern matching
4. **Run full extraction**:
   ```bash
   python scripts/extract_date_mentions.py --collection venona
   ```

## Future Enhancements

- More date formats (e.g., "circa 1945", "early 1945")
- Relative dates ("yesterday", "next week")
- Date normalization (handling ambiguous formats)
- Integration with entity mentions (linking dates to events/entities)
