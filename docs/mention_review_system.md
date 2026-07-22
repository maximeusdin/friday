# Mention Review System - Implementation Summary

## Overview

Implemented a review/adjudication workflow for ambiguous or non-deterministic mentions (both entity and date mentions) without requiring a UI. This enables human reviewers to quickly resolve collisions and ambiguous cases.

## Components

### 1. Database Schema

**Migration**: `migrations/0030_mention_review_queue.sql`

**Table**: `mention_review_queue`
- `id` (bigserial primary key)
- `mention_type` ('entity' or 'date')
- `chunk_id` (references chunks)
- `document_id` (denormalized for convenience)
- `surface` (exact substring as seen)
- `start_char`, `end_char` (optional character positions)
- `context_excerpt` (short slice of surrounding text, ±100 chars)
- `candidates` (JSONB array of candidate matches)
  - For entities: `[{"entity_id": 123, "canonical_name": "...", "score": 0.8}, ...]`
  - For dates: `[{"date_start": "1945-06-23", "date_end": "1945-06-23", "precision": "day", "confidence": 1.0}, ...]`
- `status` ('pending', 'accepted', 'rejected')
- `decision` (JSONB decision data)
  - For entities: `{"entity_id": 123, "alias_norm": "..."}`
  - For dates: `{"date_start": "1945-06-23", "date_end": "1945-06-23", "precision": "day"}`
- `note` (optional reviewer note)
- `created_at`, `reviewed_at` (timestamps)

**Indexes**:
- `(status, mention_type)` - for filtering pending reviews
- `(chunk_id)` - for chunk-based queries
- `(document_id)` - for document-based queries
- `(created_at DESC)` - for chronological ordering
- `(mention_type, status, created_at DESC)` - composite for common queries
- GIN index on `candidates` - for JSONB queries (finding by entity_id)

### 2. Review CLI Tool

**File**: `scripts/review_mentions.py`

**Commands**:

1. **`list`** - List pending reviews
   ```bash
   python scripts/review_mentions.py list
   python scripts/review_mentions.py list --type entity
   python scripts/review_mentions.py list --limit 10
   ```

2. **`show <id>`** - Show details and context for a specific review item
   ```bash
   python scripts/review_mentions.py show 123
   ```

3. **`accept <id>`** - Accept a review item
   ```bash
   # For entity mentions
   python scripts/review_mentions.py accept 123 --entity-id 456
   python scripts/review_mentions.py accept 123 --entity-id 456 --add-alias "yakubovich"
   
   # For date mentions
   python scripts/review_mentions.py accept 123 --date-start 1945-06-23 --date-end 1945-06-23 --precision day
   ```

4. **`reject <id>`** - Reject a review item
   ```bash
   python scripts/review_mentions.py reject 123 --note "Not a valid mention"
   ```

**Features**:
- Lists pending reviews with candidate information
- Shows full context and candidate details
- Accepts entity mentions with optional alias addition
- Accepts date mentions with date range and precision
- Rejects invalid mentions with optional notes
- Validates entity IDs and date formats
- Updates review status and timestamps

### 3. Populate Review Queue Helper

**File**: `scripts/populate_review_queue.py`

Helper script to populate the review queue from collision_queue items (from `extract_entity_mentions.py`).

```bash
python scripts/populate_review_queue.py --from-json collision_queue.json
python scripts/populate_review_queue.py --from-csv match_summary.csv --dry-run
```

## Usage Workflow

### Step 1: Run Extraction

Extract entity mentions, which will populate collision_queue for ambiguous cases:

```bash
python scripts/extract_entity_mentions.py --collection venona --summary-csv match_summary.csv
```

### Step 2: Populate Review Queue

(Optional) If you have collision_queue items, populate the review queue:

```bash
# Export collision_queue from extract_entity_mentions.py as JSON
# Then populate review queue
python scripts/populate_review_queue.py --from-json collision_queue.json
```

### Step 3: Review Items

List pending reviews:

```bash
python scripts/review_mentions.py list
```

Show details for a specific item:

```bash
python scripts/review_mentions.py show 123
```

Accept or reject:

```bash
# Accept entity mention
python scripts/review_mentions.py accept 123 --entity-id 456

# Accept and add alias
python scripts/review_mentions.py accept 123 --entity-id 456 --add-alias "yakubovich"

# Reject invalid mention
python scripts/review_mentions.py reject 123 --note "Not a valid entity"
```

## Integration with Extraction Scripts

The review queue can be populated from:

1. **Collision queue items** from `extract_entity_mentions.py`
   - High-value collisions that couldn't be automatically resolved
   - Stored in `collision_queue` list during extraction

2. **Manual insertion** via SQL:
   ```sql
   INSERT INTO mention_review_queue
     (mention_type, chunk_id, document_id, surface, context_excerpt, candidates, status)
   VALUES
     ('entity', 123, 456, 'yakubovich', '...context...', '[{"entity_id": 789, "canonical_name": "Yakubovich", "score": 0.8}]', 'pending');
   ```

3. **Future**: Date mention collisions (when date extraction has ambiguous cases)

## Database Migration

Run migration:

```bash
make mention-review-queue
```

Or directly:

```bash
docker exec -i $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME) -v ON_ERROR_STOP=1 < migrations/0030_mention_review_queue.sql
```

## Example Output

### List Command

```
Found 5 pending review(s):

  [123] entity | 'yakubovich' | Yakubovich, I. (ID: 456), Yakubovich, V. (ID: 789) ... (+2 more)
  [124] entity | 'NEIGHBOUR' | KGB (ID: 123), GRU (ID: 456)
  [125] date   | 'June 1945' | 1945-06-01 to 1945-06-30, 1945-06-15 to 1945-06-15
```

### Show Command

```
======================================================================
Review ID: 123
Type: entity
Status: pending
Surface: 'yakubovich'
Chunk ID: 456
Document ID: 789
Created: 2025-01-27 10:30:00+00:00

Candidates:
  1. Entity ID 456: Yakubovich, Ivan (score: 0.85)
  2. Entity ID 789: Yakubovich, Viktor (score: 0.80)
  3. Entity ID 123: Yakubovich, A. (score: 0.75)

Context excerpt:
  ...the meeting with yakubovich was held in Moscow. The agent reported...
======================================================================
```

## Future Enhancements

- **Bulk operations**: Accept/reject multiple items at once
- **Search/filter**: Filter by entity type, document, date range
- **Export**: Export review decisions to CSV/JSON
- **Statistics**: Show review statistics (pending/accepted/rejected counts)
- **Integration**: Auto-populate review queue during extraction
- **Date mentions**: Support for date mention collisions
- **Web UI**: Optional web interface for review (future)

## Notes

- Review items are **idempotent** - re-running population won't create duplicates
- Decisions are stored as JSONB for flexibility
- The `add-alias` feature automatically creates entity aliases when accepting
- Review queue supports both entity and date mentions in a single table
- Context excerpts are limited to ±100 chars for readability
