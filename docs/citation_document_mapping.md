# Citation to Document Mapping System

## Problem

Concordance citations use formats like:
- `"Venona New York KGB 1943, 112–13, 161–62, 221"`
- `"Venona San Francisco KGB, 144"`

But documents in the database are stored with source_names like:
- `"Venona_New_York_KGB_1943.pdf"`
- `"Venona_San_Francisco_KGB.pdf"`

We need to map between these formats to validate entity mentions.

## Solution

### 1. Normalization Function

The `normalize_document_name()` function converts both formats to a comparable normalized form:

```python
def normalize_document_name(name: str) -> str:
    """
    Normalizes:
    - "Venona New York KGB 1943" -> "newyorkkgb1943"
    - "Venona_New_York_KGB_1943.pdf" -> "newyorkkgb1943"
    - "Venona New York KGB 1941–42" -> "newyorkkgb1941-1942"
    """
```

**Normalization steps:**
1. Remove "Venona" or "Vassiliev" prefix
2. Remove file extension (.pdf, .txt)
3. Normalize year ranges:
   - `"1941–42"` (en-dash) → `"1941-1942"` (hyphen, full year)
   - `"1941-42"` → `"1941-1942"`
4. Extract year part
5. Remove year from name temporarily
6. Lowercase, remove punctuation/underscores, collapse whitespace
7. Add year back

### 2. Citation Parsing

The `parse_citation_text()` function:
- Normalizes whitespace (handles multi-line citations)
- Splits on semicolons to get separate citation groups
- Extracts source, year/volume, and page numbers
- Handles abbreviated page ranges (e.g., "112–13" → 112-113)

### 3. Document Mapping

The `build_citation_to_document_map()` function:
- Queries all documents in a collection
- Normalizes each document's source_name
- Creates a dictionary: `normalized_name -> [(doc_id, doc_name), ...]`
- Handles multiple documents with same normalized name (different volumes)

### 4. Document Matching

The `find_documents_for_citation()` function:
1. Normalizes the citation source
2. Builds the citation-to-document map for the collection
3. Tries exact match first
4. Falls back to matching without year, then filtering by year
5. Last resort: fuzzy matching with word boundaries

## Example

**Citation:** `"Venona New York KGB 1943, 112–13, 161–62, 221"`

**Parsing:**
- Source: `"Venona New York KGB"`
- Year: `"1943"`
- Pages: `[(112, 113), (161, 162), (221, None)]`

**Normalization:**
- Citation normalized: `"newyorkkgb1943"`
- Document `"Venona_New_York_KGB_1943.pdf"` normalized: `"newyorkkgb1943"`
- **Match!**

## Database Schema

Chunks link to documents via:
1. **chunk_pages** table: `chunk_id` → `page_id`
2. **pages** table: `page_id` → `document_id`
3. **chunk_metadata** table: `chunk_id` → `document_id` (denormalized)

The validation uses `chunk_metadata.document_id` to ensure chunks belong to the correct document.

## Usage

```python
from concordance.validate_entity_mentions_from_citations import (
    parse_citation_text,
    normalize_document_name,
    find_documents_for_citation
)

# Parse citation
citations = parse_citation_text("Venona New York KGB 1943, 112–13, 161–62")

# Find matching documents
for citation in citations:
    documents = find_documents_for_citation(cur, citation)
    # Returns: [(doc_id, doc_name), ...]
```

## Testing

Run the test script to verify normalization:

```bash
python concordance/test_citation_document_mapping.py --test-normalization
python concordance/test_citation_document_mapping.py --test-mapping
```

## Known Issues

1. **Multi-line citations**: Citations split across lines in CSV need whitespace normalization (fixed)
2. **Year range formats**: Handles `1941–42`, `1941-42`, `1941-1942` (fixed)
3. **Fuzzy matching**: May match wrong documents if source names are similar (improved with word-boundary matching)
4. **Volume matching**: Year ranges in citations may not exactly match volume fields

## Future Improvements

1. **Caching**: Cache the citation-to-document map instead of rebuilding each time
2. **Manual mapping table**: For edge cases where automatic matching fails
3. **Year normalization**: Better handling of different year range formats
4. **Document metadata**: Store normalized names in document metadata for faster lookup
