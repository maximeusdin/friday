# Citation-Based Entity Mention Validation

## Overview

The concordance index contains citation information that specifies where each entity is mentioned in the source documents. This system extracts those citations and validates that entity mentions exist in the corresponding chunks in our database.

## How It Works

### 1. Citation Extraction

Citations in concordance entries follow patterns like:

```
Venona New York KGB 1941–42, 16, 74–75; Venona New York KGB 1943, 112–13, 161–62, 221
```

This means:
- **Source**: "Venona New York KGB"
- **Year/Volume**: "1941–42" or "1943"
- **Pages**: Individual pages (16) or ranges (74–75, 112–13, 161–62, 221)

### 2. Citation Parsing

The `parse_citation_text()` function extracts:
- Source name (e.g., "Venona New York KGB", "Venona Special Studies", "Vassiliev Yellow Notebook")
- Year/volume ranges (e.g., "1941–42", "1943")
- Page numbers (individual: "16", ranges: "74–75", abbreviated ranges: "112–13" → 112-113)

### 3. Document Mapping

The system maps citations to documents using a normalization-based dictionary:

1. **Normalization**: Both citation sources and document source_names are normalized:
   - `"Venona New York KGB 1943"` → `"newyorkkgb1943"`
   - `"Venona_New_York_KGB_1943.pdf"` → `"newyorkkgb1943"`
   - Handles underscores, spaces, year ranges, and file extensions

2. **Mapping Dictionary**: Builds a map of `normalized_name -> [(doc_id, doc_name), ...]` for each collection

3. **Matching**: 
   - Exact match on normalized names (preferred)
   - Fallback to matching without year, then filtering by year
   - Last resort: fuzzy matching with word boundaries

This ensures precise matching and avoids false positives (e.g., "Venona San Francisco KGB" won't match "Venona Bogota KGB").

### 4. Page Matching

For each document, the system finds pages that match the citation page numbers:
- Matches by `pdf_page_number` (if available)
- Matches by `logical_page_label` (fallback)
- Handles page ranges (e.g., pages 74-75)

### 5. Entity Mention Validation

For each matched page, the system:
1. Finds chunks associated with those pages (via `chunk_pages` table)
2. Checks if those chunks have entity mentions (via `entity_mentions` table)
3. Reports validation status:
   - **validated**: Entity mentions found in chunks
   - **no_mentions**: Chunks exist but no entity mentions
   - **no_pages**: Document found but pages don't match
   - **no_document**: No matching document found

## Usage

### Basic Usage

```bash
# Validate by entity ID
python concordance/validate_entity_mentions_from_citations.py --entity-id 123

# Validate by entity name
python concordance/validate_entity_mentions_from_citations.py --entity-name "AKIM"

# Validate by concordance entry key
python concordance/validate_entity_mentions_from_citations.py --entry-key "AKIM (cover name in Venona)"
```

### Example: AKIM Validation

```bash
# Test citation parsing
python concordance/example_akim_validation.py --test-parsing

# Validate entity mentions
python concordance/example_akim_validation.py --validate
```

## Citation Format Examples

### Venona Citations

```
Venona New York KGB 1941–42, 16, 74–75
Venona New York KGB 1943, 112–13, 161–62, 221
Venona San Francisco KGB, 144
Venona Special Studies, 3–4, 93
```

### Vassiliev Citations

```
Vassiliev Yellow Notebook #2, 18, 21, 72, 83
Vassiliev White Notebook #1, 133–34
Vassiliev's notebooks, 52
```

## Page Number Formats

The parser handles:
- **Single pages**: `16`, `221`
- **Full ranges**: `74–75` (pages 74 to 75)
- **Abbreviated ranges**: `112–13` (pages 112 to 113), `161–62` (pages 161 to 162)

## Database Schema Requirements

The validation system requires:

1. **Documents table**: `documents` with `collection_id`, `source_name`, `volume`
   - `source_name` is typically the PDF filename (e.g., "Venona_New_York_KGB_1943.pdf")
   
2. **Pages table**: `pages` with `document_id`, `pdf_page_number`, `logical_page_label`
   - `pdf_page_number` is used to match citation page numbers
   
3. **Chunks table**: `chunks` with chunk text
   
4. **Chunk metadata**: `chunk_metadata` with `document_id` denormalized
   - **Critical**: `chunk_metadata.document_id` must match `pages.document_id` for validation
   - This ensures chunks are correctly linked to their source documents
   
5. **Chunk pages**: `chunk_pages` linking chunks to pages
   - Used to find chunks associated with specific pages
   
6. **Entity mentions**: `entity_mentions` with `entity_id`, `chunk_id`
   - Populated by `extract_entity_mentions.py`

## Integration with Concordance Ingestion

Citations are already extracted during concordance ingestion and stored in:
- `entity_citations` table with `citation_text`, `collection_slug`, `document_label`, `page_list`

The validation script reads from `entity_citations` to get citation information for each entity.

## Use Cases

1. **Quality Assurance**: Verify that entity mentions exist where citations say they should
2. **Ambiguity Resolution**: When multiple entities match the same string, use citation locations to disambiguate
3. **Coverage Analysis**: Identify gaps where citations exist but mentions are missing
4. **Data Validation**: Ensure concordance citations match actual document content

## Limitations

1. **Document Matching**: 
   - Uses normalization-based matching (handles most cases)
   - May need manual mapping table for edge cases where source names don't match
   - Multi-line citations in CSV need whitespace normalization (fixed)

2. **Page Number Matching**: 
   - Relies on `pdf_page_number` or `logical_page_label` matching citation page numbers
   - Page ranges (e.g., "74–75") match any page in the range

3. **Entity Mentions**: 
   - Requires `entity_mentions` table to be populated (via `extract_entity_mentions.py`)
   - If mentions are missing, validation will show "no_mentions" status

4. **Year/Volume Matching**: 
   - Year ranges in citations (e.g., "1941–42") are normalized to "1941-1942"
   - Matches document names containing the year (e.g., "Venona_New_York_KGB_1941-42.pdf")
   - Volume field matching is optional (year in source_name is primary)

## Future Enhancements

1. **Fuzzy Document Matching**: Improve matching when source names don't exactly match
2. **Page Number Normalization**: Handle different page numbering schemes
3. **Batch Validation**: Validate all entities at once
4. **Report Generation**: Generate detailed reports of validation results
5. **Automatic Mention Creation**: Create entity mentions based on citation locations if missing
