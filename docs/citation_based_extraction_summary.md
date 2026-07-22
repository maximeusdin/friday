# Citation-Based Entity Mention Extraction - Summary

## What Was Built

An improved entity mention extraction pipeline (`concordance/extract_entity_mentions_from_citations.py`) that uses citation information from the concordance index to target specific document spans where entities are known to appear.

## Key Features

1. **Citation-Based Targeting**: Only extracts mentions from chunks that correspond to expert-curated citations
2. **Document Mapping**: Uses normalization to map citation sources (e.g., "Vassiliev Black Notebook") to database documents
3. **Page Matching**: Finds pages matching citation page numbers (handles ranges like "63–74")
4. **Chunk Finding**: Identifies chunks that span the matched pages via `chunk_pages` bridge table
5. **Provenance Tracking**: Stores mentions with proper document+page provenance

## Example: "Albert"

For the entity "Albert" (cover name for Iskhak Akhmerov), the pipeline:

1. **Reads Citations**: 
   - "Vassiliev Black Notebook, 79"
   - "Vassiliev White Notebook #1, 55, 63–74, 153"
   - "Vassiliev White Notebook #2, 3, 8, 17–18, 24, 26–27, 31, 39"
   - etc.

2. **Maps to Documents**: 
   - "Vassiliev Black Notebook" → `Vassiliev_Black_Notebook.pdf` (document_id: 10)
   - "Vassiliev White Notebook #1" → `Vassiliev_White_Notebook_1.pdf` (document_id: 12)
   - etc.

3. **Finds Pages**: 
   - For document 10, finds page with `pdf_page_number = 79`
   - For document 12, finds pages with `pdf_page_number` in [55, 63-74, 153]
   - etc.

4. **Finds Chunks**: 
   - Finds chunks that span these pages via `chunk_pages` table
   - Verifies chunks belong to correct document via `chunk_metadata.document_id`

5. **Extracts Mentions**: 
   - Extracts entity mentions from these targeted chunks
   - Stores with proper provenance (document_id, chunk_id, surface text)

## Results

For "Albert", the pipeline found:
- 6 citation records
- 5 citations with matching documents (1 didn't match: "Venona Special Studies")
- 14 citations with matching pages
- 14 citations with matching chunks
- 1,267 unique chunks to process
- 3,166 mentions would be extracted

## Benefits

1. **Resolves Ambiguities**: By targeting specific documents, we can better resolve ambiguous names
   - "Albert" in Vassiliev notebooks → Iskhak Akhmerov
   - "Albert" elsewhere → might be a different person

2. **Improves Precision**: Only extracts from chunks where entities are known to appear
   - Reduces false positives
   - Focuses on expert-curated locations

3. **Efficiency**: Targets specific chunks rather than scanning all chunks
   - Faster processing
   - Lower memory usage

4. **Provenance**: Mentions are linked to specific document+page combinations
   - Enables citation-based validation
   - Supports ambiguity resolution workflows

## Usage

```bash
# Extract mentions for a specific entity
python concordance/extract_entity_mentions_from_citations.py --entity-name "Albert"

# Extract for all entities with citations
python concordance/extract_entity_mentions_from_citations.py --all-entities

# Dry run to see what would be extracted
python concordance/extract_entity_mentions_from_citations.py --entity-name "Albert" --dry-run
```

## Next Steps

1. **Run Full Extraction**: Extract mentions for all entities with citations
2. **Validate Results**: Compare citation-based mentions with full-scan mentions
3. **Resolve Ambiguities**: Use document context to resolve ambiguous entity mentions
4. **Improve Matching**: Enhance document/page matching for edge cases (e.g., "Venona Special Studies")

## Related Documentation

- `docs/citation_based_entity_mention_extraction.md`: Full technical documentation
- `concordance/extract_entity_mentions_from_citations.py`: Main extraction script
- `concordance/validate_entity_mentions_from_citations.py`: Validation script
