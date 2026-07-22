# Verifying Entity Mentions in Chunks

This guide explains how to verify that extracted entity mentions actually appear in chunks at the expected page numbers.

## Overview

The `verify_mentions_in_chunks.py` script checks:

1. **Chunk existence**: The chunk referenced by the mention actually exists
2. **Text presence**: The surface text appears in the chunk
3. **Character positions**: The start_char/end_char positions are valid and match the surface text
4. **Document matching**: The chunk's document_id matches the mention's document_id
5. **Page overlap**: The chunk spans pages that overlap with pages from the entity's citations

## Usage

### From CSV file

```bash
# Verify all mentions in the CSV
python concordance/verify_mentions_in_chunks.py --csv-file concordance_export/entity_mentions.csv

# Verify mentions for a specific entity
python concordance/verify_mentions_in_chunks.py --csv-file concordance_export/entity_mentions.csv --entity-name "Vladimir Pravdin"

# Limit to first 100 mentions for quick check
python concordance/verify_mentions_in_chunks.py --csv-file concordance_export/entity_mentions.csv --limit 100

# Verbose output showing details
python concordance/verify_mentions_in_chunks.py --csv-file concordance_export/entity_mentions.csv --entity-name "Vladimir Pravdin" --verbose
```

### From database

```bash
# Verify mentions for a specific entity by name
python concordance/verify_mentions_in_chunks.py --entity-name "Vladimir Pravdin"

# Verify mentions for a specific entity by ID
python concordance/verify_mentions_in_chunks.py --entity-id 36270

# Limit to first 50 mentions
python concordance/verify_mentions_in_chunks.py --limit 50
```

## What Gets Checked

### 1. Chunk Text Contains Surface
- Verifies that the `surface` field (e.g., "VLADIMIR PRAVDIN") appears in the chunk text
- Uses case-insensitive matching if exact match fails

### 2. Character Positions
- Validates that `start_char` and `end_char` are within the chunk text bounds
- Checks that `start_char < end_char`
- Verifies that the text at those positions matches the `surface` field

### 3. Document Matching
- Ensures the chunk's `document_id` (from `chunk_metadata`) matches the mention's `document_id`
- This prevents false positives from chunks in different documents

### 4. Page Overlap
- Gets all pages associated with the chunk (via `chunk_pages`)
- Gets all pages from citations for the entity
- Checks if there's any overlap between chunk pages and citation pages
- Note: This is a soft check - mentions are still considered valid even without page overlap (the overlap check helps identify potential issues)

## Output

The script provides:

- **Summary statistics**: Count of valid vs invalid mentions
- **Error details**: For invalid mentions, shows:
  - Entity name
  - Chunk ID
  - Surface text
  - Specific error message

### Example Output

```
Reading concordance_export/entity_mentions.csv...
Filtered to 15 rows for entity 'Vladimir Pravdin'

Verifying 15 mentions...

================================================================================
Row 3 - Entity: Vladimir Pravdin
  Chunk ID: 5969
  Surface: VLADIMIR PRAVDIN
  Document mismatch: chunk has doc_id 42, mention has 43
================================================================================

================================================================================
Summary:
  Valid:   14
  Invalid: 1
  Total:   15
================================================================================

✗ Found 1 invalid mentions
```

## Common Issues

### Document Mismatch
- **Cause**: Chunk metadata points to a different document than the mention
- **Fix**: Check the extraction logic to ensure chunks are correctly linked to documents

### Surface Text Not Found
- **Cause**: The extracted surface text doesn't appear in the chunk
- **Possible reasons**:
  - Normalization issues (e.g., "VLADIMIR" vs "Vladimir")
  - Text was extracted from a different chunk
  - Chunk text was modified after extraction

### Character Position Out of Range
- **Cause**: `start_char` or `end_char` exceed the chunk text length
- **Fix**: Check the extraction logic to ensure positions are calculated correctly

### No Page Overlap
- **Cause**: Chunk pages don't overlap with citation pages
- **Note**: This is a warning, not necessarily an error - chunks can span multiple pages, and the mention might be on a page that's not in the citation list

## Integration with Export Verification

This script complements `verify_citation_export.py`:

- **`verify_citation_export.py`**: Verifies that `citation_page_lists` correctly match `citation_texts`
- **`verify_mentions_in_chunks.py`**: Verifies that mentions actually appear in chunks at the expected locations

Together, these scripts ensure:
1. Citation data is correctly parsed and exported
2. Extracted mentions are correctly linked to chunks and pages
