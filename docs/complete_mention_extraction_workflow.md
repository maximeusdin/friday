# Complete Mention Extraction Workflow

This document provides the complete sequence of commands to:
1. Compute Document Frequency (DF) statistics for collision assistance
2. Extract all entity mentions from your corpus
3. Extract all date mentions from your corpus
4. Export CSV files for analysis
5. Populate review queue (happens automatically during entity extraction)

## Prerequisites

Ensure your database environment variables are set:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neh
export DB_USER=neh
export DB_PASS=neh
```

Or if using Docker:
```bash
# Connect to your database container
docker exec -it neh_postgres psql -U neh -d neh
```

## Step-by-Step Commands

### Step 1: Compute Document Frequency (DF) Statistics

This computes how frequently each alias appears per document, which helps with collision resolution.

```bash
python scripts/compute_alias_frequency.py \
  --source-slug "vassiliev_venona_index_full_capitalized"
```

**What this does:**
- Computes DF (document frequency) for each `alias_norm` per `document_id`
- Stores results in `alias_stats` table
- Used by `extract_entity_mentions.py` to suppress common/generic terms

**Expected output:**
- Progress messages showing chunks processed
- Summary: "Completed DF computation for X aliases across Y documents"

---

### Step 2: Extract Entity Mentions

Extract all entity mentions from your corpus. This will automatically populate the review queue with unresolved collisions.

```bash
python scripts/extract_entity_mentions.py \
  --collection venona \
  --concordance-source-slug "vassiliev_venona_index_full_capitalized" \
  --enable-partial \
  --summary-csv match_summary.csv
```

**For Vassiliev collection:**
```bash
python scripts/extract_entity_mentions.py \
  --collection vassiliev \
  --concordance-source-slug "vassiliev_venona_index_full_capitalized" \
  --enable-partial \
  --summary-csv match_summary.csv
```

**For both collections (run separately):**
```bash
# Venona
python scripts/extract_entity_mentions.py \
  --collection venona \
  --concordance-source-slug "vassiliev_venona_index_full_capitalized" \
  --enable-partial \
  --summary-csv match_summary_venona.csv

# Vassiliev
python scripts/extract_entity_mentions.py \
  --collection vassiliev \
  --concordance-source-slug "vassiliev_venona_index_full_capitalized" \
  --enable-partial \
  --summary-csv match_summary_vassiliev.csv
```

**What this does:**
- Scans all chunks in the specified collection
- Matches entity aliases (exact + partial matching)
- Inserts high-confidence matches into `entity_mentions` table
- Automatically populates `mention_review_queue` with unresolved collisions
- Outputs summary statistics to CSV

**Key flags:**
- `--enable-partial`: Enables partial matching (surname matching, etc.)
- `--enable-fuzzy`: Optional - enables fuzzy matching (slower, more matches)
- `--summary-csv`: Outputs match statistics for analysis
- `--concordance-source-slug`: Only matches entities from this source

**Note:** The review queue is automatically populated during extraction. All unresolved collisions (harmless, high-value, dominance-none) are enqueued with metadata.

---

### Step 3: Extract Date Mentions

Extract all explicit date expressions from your corpus.

```bash
python scripts/extract_date_mentions.py \
  --collection venona
```

**For Vassiliev collection:**
```bash
python scripts/extract_date_mentions.py \
  --collection vassiliev
```

**For both collections:**
```bash
# Venona
python scripts/extract_date_mentions.py --collection venona

# Vassiliev
python scripts/extract_date_mentions.py --collection vassiliev
```

**What this does:**
- Scans all chunks for explicit date expressions
- Uses deterministic regex patterns to extract dates
- Parses dates into structured format (date_start, date_end, precision)
- Inserts into `date_mentions` table with method and confidence

**Supported formats:**
- "23 June 1945" (day precision)
- "June 1945" (month precision)
- "1945" (year precision)
- "1943-1945" (range precision)

---

### Step 4: Export CSV Files

Export all extracted data to CSV files for analysis.

```bash
python scripts/export_concordance_data.py \
  --source-slug "vassiliev_venona_index_full_capitalized" \
  --output-dir concordance_export
```

**What this exports:**
- `entity_mentions.csv`: All extracted entity mentions
- `entities.csv`: All entities from concordance
- `entity_aliases.csv`: All aliases with their types
- `entity_links.csv`: All relationships
- `entity_citations.csv`: All citations

**Note:** If you don't specify `--source-slug`, it will use the most recent source slug automatically.

---

### Step 5: Review Queue Status (Optional)

Check what's in the review queue:

```bash
# List all pending reviews
python scripts/review_mentions.py list

# List only high-value collisions
python scripts/review_mentions.py list --high-value

# Show statistics
python scripts/review_mentions.py stats

# Show details for a specific item
python scripts/review_mentions.py show <id>
```

---

## Complete Workflow (All-in-One)

If you want to run everything in sequence:

```bash
# 1. Compute DF statistics
python scripts/compute_alias_frequency.py \
  --source-slug "vassiliev_venona_index_full_capitalized"

# 2. Extract entity mentions (Venona)
python scripts/extract_entity_mentions.py \
  --collection venona \
  --concordance-source-slug "vassiliev_venona_index_full_capitalized" \
  --enable-partial \
  --summary-csv match_summary_venona.csv

# 3. Extract entity mentions (Vassiliev)
python scripts/extract_entity_mentions.py \
  --collection vassiliev \
  --concordance-source-slug "vassiliev_venona_index_full_capitalized" \
  --enable-partial \
  --summary-csv match_summary_vassiliev.csv

# 4. Extract date mentions (Venona)
python scripts/extract_date_mentions.py --collection venona

# 5. Extract date mentions (Vassiliev)
python scripts/extract_date_mentions.py --collection vassiliev

# 6. Export CSV files
python scripts/export_concordance_data.py \
  --source-slug "vassiliev_venona_index_full_capitalized" \
  --output-dir concordance_export

# 7. Check review queue
python scripts/review_mentions.py stats
```

---

## Verification

After running all commands, verify your data:

```sql
-- Check entity mentions count
SELECT COUNT(*) FROM entity_mentions;

-- Check date mentions count
SELECT COUNT(*) FROM date_mentions;

-- Check review queue status
SELECT status, COUNT(*) 
FROM mention_review_queue 
GROUP BY status;

-- Check DF statistics
SELECT COUNT(*) FROM alias_stats;
```

---

## Troubleshooting

### If extraction is slow:
- Use `--limit` flag to test on a subset first
- Use `--skip-diagnostics` to skip expensive operations
- Process collections separately

### If you want to re-run extraction:
- All scripts are idempotent (safe to re-run)
- Uses `ON CONFLICT DO NOTHING` or unique constraints
- Won't create duplicates

### If review queue seems empty:
- Check that collisions actually occurred (check match_summary.csv)
- Review queue only contains unresolved collisions
- Use `--all-statuses` flag: `python scripts/review_mentions.py list --all-statuses`

---

## Next Steps

After completing this workflow, you can:

1. **Review ambiguous mentions**: Use `review_mentions.py` to adjudicate collisions
2. **Query mentions**: Use ENTITY and DATE_RANGE primitives in plans
3. **Analyze data**: Use exported CSV files for analysis
4. **Build queries**: Use mention-driven primitives in plan execution

---

## Notes

- **DF computation** should be run before entity extraction for best collision assistance
- **Entity extraction** automatically populates review queue - no separate step needed
- **Date extraction** is independent and can be run anytime
- **CSV export** can be run anytime to get current state
- All operations are **idempotent** - safe to re-run
