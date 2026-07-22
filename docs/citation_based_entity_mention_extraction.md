# Citation-Based Entity Mention Extraction

## Overview

This document describes the improved entity mention extraction pipeline that uses citation information from the concordance index to target specific document spans where entities are known to appear. This approach helps resolve ambiguities by focusing extraction on chunks that correspond to expert-curated citations.

## Problem Statement

When extracting entity mentions from chunks, ambiguities can arise:

1. **Ambiguous Names**: Common names like "Albert" can refer to multiple entities
2. **Context-Dependent**: The same surface text might refer to different entities in different documents
3. **Cover Names**: Cover names like "AKIM" or "ALBERT" might be ambiguous without document context

The original extraction pipeline (`scripts/extract_entity_mentions.py`) scans all chunks and matches aliases, but doesn't use citation information to target specific chunks where entities are known to appear.

## Solution: Citation-Based Targeting

The improved pipeline (`concordance/extract_entity_mentions_from_citations.py`) uses citation information to:

1. **Target Specific Chunks**: Only extract mentions from chunks that correspond to cited document+page combinations
2. **Resolve Ambiguities**: By knowing which document a mention appears in, we can better resolve ambiguous names
3. **Improve Precision**: Focus extraction on expert-curated locations rather than scanning all chunks

## How It Works

### 1. Citation Parsing

The pipeline reads entity citations from the `entity_citations` table and parses citation text to extract:

- **Document names**: e.g., "Vassiliev Black Notebook", "Venona New York KGB 1943"
- **Page numbers**: e.g., "79", "55, 63–74, 153"
- **Year ranges**: e.g., "1941–42" (for Venona documents)

Example citation:
```
"Albert" ["Al'bert"] (cover name in Vassiliev's notebooks): Iskhak Akhmerov. 
Vassiliev Black Notebook, 79; 
Vassiliev White Notebook #1, 55, 63–74, 153; 
Vassiliev White Notebook #2, 3, 8, 17–18, 24, 26–27, 31, 39; 
Vassiliev White Notebook #3, 17–22, 24, 28, 33, 36–37, 39, 42, 46, 53, 55–56, 
60, 71, 76–77, 86, 106, 131; 
Vassiliev Yellow Notebook #2, 34, 62, 66, 74, 77, 79, 83–85.
```

### 2. Document Mapping

The pipeline maps citation sources to database documents using normalization:

- **Normalization**: Converts citation sources and document names to a comparable format
  - "Vassiliev Black Notebook" → "blacknotebook"
  - "Vassiliev_Black_Notebook.pdf" → "blacknotebook"
- **Year Matching**: Handles year ranges (e.g., "1941–42" matches "1941-1942")
- **Fuzzy Matching**: Falls back to word-based matching if exact match fails

### 3. Page Matching

For each matched document, the pipeline finds pages that correspond to citation page numbers:

- **PDF Page Numbers**: Matches `pdf_page_number` in `pages` table
- **Logical Page Labels**: Matches `logical_page_label` if PDF page number unavailable
- **Page Ranges**: Handles ranges like "63–74" (finds all pages in range)

### 4. Chunk Finding

The pipeline finds chunks that span the matched pages:

- **Via `chunk_pages`**: Uses the `chunk_pages` bridge table to find chunks that include any of the matched pages
- **Document Verification**: Ensures chunks belong to the correct document via `chunk_metadata.document_id`

### 5. Mention Extraction

The pipeline extracts entity mentions from the targeted chunks using the same extraction logic as `scripts/extract_entity_mentions.py`:

- **Alias Matching**: Matches normalized aliases against chunk text
- **Policy Checks**: Applies policy checks (case matching, min_chars, etc.)
- **Collision Resolution**: Handles alias collisions using dominance rules

### 6. Storage

Mentions are stored in the `entity_mentions` table with proper provenance:

- `entity_id`: The entity being mentioned
- `chunk_id`: The chunk containing the mention
- `document_id`: The document containing the chunk (denormalized for speed)
- `surface`: The exact surface text from the chunk
- `surface_norm`: Normalized surface text
- `surface_quality`: 'exact' or 'approx' (for OCR/punctuation differences)
- `method`: 'alias_exact' (citation-based extraction)

## Usage

### Extract Mentions for a Specific Entity

```bash
# Extract mentions for "Albert"
python concordance/extract_entity_mentions_from_citations.py --entity-name "Albert"

# Extract mentions for entity ID 123
python concordance/extract_entity_mentions_from_citations.py --entity-id 123
```

### Extract Mentions for All Entities

```bash
# Extract mentions for all entities with citations
python concordance/extract_entity_mentions_from_citations.py --all-entities

# Filter by collection
python concordance/extract_entity_mentions_from_citations.py --collection venona --all-entities
```

### Dry Run

```bash
# See what would be extracted without inserting
python concordance/extract_entity_mentions_from_citations.py --entity-name "Albert" --dry-run
```

### Verbose Output

```bash
# Print detailed progress information
python concordance/extract_entity_mentions_from_citations.py --entity-name "Albert" --verbose
```

## Example Output

```
Loading entity aliases...
  Loaded 17977 unique normalized aliases

Processing 1 entity/entities...

[1/1] Processing: Albert (ID: 456, Type: cover_name)
  Found 1 citation record(s) for Albert
  Found 15 unique chunk(s) to process
  Extracted 23 mention(s) from 15 chunk(s)
  Citations: 1 processed, 1 with documents, 1 with pages, 1 with chunks
  Chunks: 15 processed, Mentions: 23 found

================================================================================
SUMMARY
================================================================================
Entities processed: 1
Total citations processed: 1
  - With documents: 1
  - With pages: 1
  - With chunks: 1
  - No document match: 0
  - No pages match: 0
  - No chunks match: 0
Total chunks processed: 15
Total mentions extracted: 23
```

## Benefits

1. **Precision**: Only extracts mentions from chunks where entities are known to appear
2. **Ambiguity Resolution**: Document context helps resolve ambiguous names
3. **Efficiency**: Targets specific chunks rather than scanning all chunks
4. **Provenance**: Mentions are linked to specific document+page combinations
5. **Expert-Curated**: Uses expert-curated citations as ground truth

## Limitations

1. **Citation Coverage**: Only extracts mentions for entities that have citations
2. **Page Matching**: Relies on accurate page number matching (PDF page numbers or logical labels)
3. **Chunk Spanning**: If a chunk spans multiple pages, it will be included if any page matches
4. **Document Matching**: Requires accurate document name normalization

## Future Improvements

1. **Span Extraction**: Extract exact character spans within chunks (currently stores surface text)
2. **Context Windows**: Include surrounding context for better ambiguity resolution
3. **Confidence Scoring**: Score mentions based on citation match quality
4. **Validation**: Compare citation-based mentions with full-scan mentions to find discrepancies
5. **Incremental Updates**: Only re-extract mentions when citations change

## Related Files

- `concordance/extract_entity_mentions_from_citations.py`: Main extraction script
- `concordance/validate_entity_mentions_from_citations.py`: Validation script (checks if citations match chunks)
- `scripts/extract_entity_mentions.py`: Original full-scan extraction script
- `concordance/ingest_concordance_tab_aware.py`: Ingests citations from concordance index

## Database Schema

Key tables used:

- `entity_citations`: Stores citation text, document_label, page_list
- `documents`: Document metadata (source_name, collection_id)
- `pages`: Page metadata (pdf_page_number, logical_page_label, document_id)
- `chunks`: Chunk text and embeddings
- `chunk_pages`: Bridge table linking chunks to pages
- `chunk_metadata`: Chunk metadata (document_id, first_page_id)
- `entity_mentions`: Extracted mentions (entity_id, chunk_id, document_id, surface)
