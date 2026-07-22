# Citation to Document Mapping - Summary

## Problem Statement

Concordance entries contain citations like:
```
Venona New York KGB 1941–42, 16, 74–75; Venona New York KGB 1943, 112–13, 161–62, 221
```

We need to:
1. Parse these citations to extract source, year, and page numbers
2. Map citations to actual documents in our database
3. Verify that chunks from those document pages contain entity mentions

## Solution Architecture

### 1. Citation Parsing (`parse_citation_text`)

**Input:** Citation text string (may contain newlines)
**Output:** List of `CitationLocation` objects

**Features:**
- Normalizes whitespace (handles multi-line citations)
- Splits on semicolons to get separate citation groups
- Extracts:
  - Source: "Venona New York KGB", "Venona San Francisco KGB", etc.
  - Year/Volume: "1941–42", "1943" (optional)
  - Pages: Individual (16) or ranges (74–75, 112–13 → 112-113)

**Example:**
```python
citation = "Venona New York KGB 1943, 112–13, 161–62, 221"
locations = parse_citation_text(citation)
# Returns: [CitationLocation(
#     source="Venona New York KGB",
#     year_range="1943",
#     pages=[(112, 113), (161, 162), (221, None)]
# )]
```

### 2. Document Name Normalization (`normalize_document_name`)

**Purpose:** Convert both citation formats and document source_names to a comparable form

**Normalization Steps:**
1. Remove file extension (.pdf, .txt)
2. Remove "Venona" or "Vassiliev" prefix (handles spaces and underscores)
3. Normalize year ranges:
   - `"1941–42"` (en-dash) → `"1941-1942"` (hyphen, full year)
   - `"1941-42"` → `"1941-1942"`
4. Extract year part
5. Remove year from name temporarily
6. Lowercase, remove punctuation/underscores, collapse whitespace
7. Add year back

**Examples:**
- `"Venona New York KGB 1943"` → `"newyorkkgb1943"`
- `"Venona_New_York_KGB_1943.pdf"` → `"newyorkkgb1943"` ✓ Match!
- `"Venona New York KGB 1941–42"` → `"newyorkkgb1941-1942"`
- `"Venona_New_York_KGB_1941-42.pdf"` → `"newyorkkgb1941-1942"` ✓ Match!

### 3. Document Mapping (`build_citation_to_document_map`)

**Purpose:** Build a dictionary mapping normalized names to document IDs

**Process:**
1. Query all documents in collection (venona or vassiliev)
2. For each document:
   - Normalize `source_name` (e.g., "Venona_New_York_KGB_1943.pdf")
   - Add to map: `normalized_name -> [(doc_id, doc_name), ...]`
   - Also add with volume if present

**Result:** Fast lookup dictionary for exact matching

### 4. Document Finding (`find_documents_for_citation`)

**Process:**
1. Normalize citation source (with year if available)
2. Build citation-to-document map for collection
3. Try exact match first
4. If no match and year present:
   - Try match without year
   - Filter results by year
5. Fallback: Fuzzy matching with word boundaries

**Returns:** List of `(document_id, document_name)` tuples

### 5. Page Matching (`find_pages_for_citation`)

**Process:**
1. Get all pages for the document
2. Match pages by:
   - `pdf_page_number` (primary) - exact match or within range
   - `logical_page_label` (fallback) - substring match
3. Return list of matching page IDs

### 6. Chunk Validation (`find_chunks_with_entity`)

**Process:**
1. Get `document_id`s from matched pages
2. Find chunks via `chunk_pages` that are associated with those pages
3. **Verify document_id**: Use `chunk_metadata.document_id` to ensure chunks belong to correct document
4. Check `entity_mentions` table for entity mentions in those chunks

**Critical:** Uses `chunk_metadata.document_id` to ensure chunks are correctly linked to their source documents. This prevents false matches where chunks from different documents might be associated with the same page number.

## Data Flow

```
Concordance Citation
  ↓
parse_citation_text()
  ↓
CitationLocation (source, year, pages)
  ↓
normalize_document_name()
  ↓
build_citation_to_document_map()
  ↓
find_documents_for_citation()
  ↓
find_pages_for_citation()
  ↓
find_chunks_with_entity()
  ↓
Validation Result
```

## Key Design Decisions

1. **Normalization-based matching**: More reliable than fuzzy string matching
2. **Document ID verification**: Uses `chunk_metadata.document_id` to ensure correct document provenance
3. **Year range normalization**: Handles abbreviated years (1941–42 → 1941-1942)
4. **Multi-line citation handling**: Normalizes whitespace before parsing

## Testing

Run the test script to verify:

```bash
# Test normalization
python concordance/test_normalization_simple.py

# Test full mapping
python concordance/test_citation_document_mapping.py --test-mapping
```

## Current Status

✅ **Normalization**: Working correctly - all test cases pass
✅ **Citation parsing**: Handles multi-line citations
✅ **Document matching**: Uses normalization dictionary for precise matching
✅ **Chunk validation**: Verifies document_id via chunk_metadata

## Next Steps

1. Run entity mention extraction if not done:
   ```bash
   python scripts/extract_entity_mentions.py --collection venona
   ```

2. Test validation on real entity:
   ```bash
   python concordance/validate_entity_mentions_from_citations.py --entity-name "AKIM"
   ```

3. Review results and refine mapping for edge cases if needed
