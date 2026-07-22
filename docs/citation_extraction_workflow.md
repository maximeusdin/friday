# Citation-Based Entity Mention Extraction Workflow

## Overview

The citation-based entity mention extraction is a two-step process:

1. **Ingest Concordance Index**: Parse the concordance PDF and store entity citations
2. **Extract Mentions**: Use citations to target specific chunks and extract entity mentions

## Step 1: Ingest Concordance Index

The ingest script (`concordance/ingest_concordance_tab_aware.py`) parses the concordance index PDF and stores:

- **Entities**: Canonical entity names and types
- **Entity Aliases**: Alternative names, cover names, etc.
- **Entity Links**: Relationships between entities
- **Entity Citations**: Citation text with document and page information

### What Gets Stored

For each citation, the ingest script stores in `entity_citations`:

- `citation_text`: Full citation text (e.g., "Vassiliev Black Notebook, 79")
- `collection_slug`: Parsed collection (e.g., "vassiliev", "venona")
- `document_label`: Parsed document name (e.g., "Vassiliev Black Notebook")
- `page_list`: Parsed page numbers as a list (e.g., [79])

### Running the Ingest

```bash
# Basic ingest
python concordance/ingest_concordance_tab_aware.py data/concordance_index.pdf \
    --source-slug concordance \
    --source-title "Concordance Index"

# With options
python concordance/ingest_concordance_tab_aware.py data/concordance_index.pdf \
    --source-slug concordance \
    --source-title "Concordance Index" \
    --segment auto \
    --verbose

# Dry run to preview
python concordance/ingest_concordance_tab_aware.py data/concordance_index.pdf \
    --source-slug concordance \
    --source-title "Concordance Index" \
    --dry-run
```

## Step 2: Extract Entity Mentions from Citations

After ingestion, run the extraction script (`concordance/extract_entity_mentions_from_citations.py`) to:

1. Read entity citations from `entity_citations` table
2. Parse citations to extract document names and page numbers
3. Map citations to database documents
4. Find chunks that span the cited pages
5. Extract entity mentions from those targeted chunks
6. Store mentions in `entity_mentions` table

### Running the Extraction

```bash
# Extract for all entities with citations
python concordance/extract_entity_mentions_from_citations.py --all-entities

# Extract for a specific entity
python concordance/extract_entity_mentions_from_citations.py --entity-name "Albert"

# Dry run to preview
python concordance/extract_entity_mentions_from_citations.py --all-entities --dry-run

# Filter by collection
python concordance/extract_entity_mentions_from_citations.py --collection venona --all-entities

# Verbose output
python concordance/extract_entity_mentions_from_citations.py --all-entities --verbose
```

## Complete Workflow

### Option 1: Run Both Steps Separately

```bash
# Step 1: Ingest
python concordance/ingest_concordance_tab_aware.py data/concordance_index.pdf \
    --source-slug concordance \
    --source-title "Concordance Index"

# Step 2: Extract (after reviewing ingest results)
python concordance/extract_entity_mentions_from_citations.py --all-entities --dry-run  # Preview
python concordance/extract_entity_mentions_from_citations.py --all-entities  # Actual extraction
```

### Option 2: Use Workflow Script (Linux/Mac)

```bash
chmod +x concordance/run_citation_based_extraction.sh
./concordance/run_citation_based_extraction.sh data/concordance_index.pdf \
    --source-slug concordance \
    --source-title "Concordance Index"
```

### Option 3: Use Workflow Script (Windows PowerShell)

```powershell
.\concordance\run_citation_based_extraction.ps1 data\concordance_index.pdf `
    --source-slug concordance `
    --source-title "Concordance Index"
```

## Verification

After running both steps, verify the results:

```bash
# Check citation counts
python -c "
import psycopg2
conn = psycopg2.connect(host='localhost', dbname='neh', user='neh', password='neh')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM entity_citations')
print(f'Entity citations: {cur.fetchone()[0]}')
cur.execute('SELECT COUNT(*) FROM entity_mentions')
print(f'Entity mentions: {cur.fetchone()[0]}')
cur.close()
conn.close()
"

# Check mentions for a specific entity
python concordance/show_entity_document_links.py --entity-name "Albert" --show-chunks
```

## Important Notes

1. **Ingest First**: Always run the ingest script before extraction
2. **Dry Run First**: Use `--dry-run` to preview what will be extracted
3. **Citation Coverage**: Only entities with citations will have mentions extracted
4. **Document Matching**: Extraction relies on accurate document name normalization
5. **Page Matching**: Extraction relies on accurate page number matching (PDF page numbers or logical labels)

## Troubleshooting

### No Documents Found for Citations

If citations don't match documents:
- Check document names in database: `python concordance/list_documents_in_chunks.py`
- Verify normalization: `python concordance/test_citation_document_mapping.py`
- Check citation parsing: `python concordance/show_entity_document_links.py --entity-name "ENTITY" --verbose`

### No Pages Found for Citations

If pages don't match:
- Check page numbers in database: `SELECT pdf_page_number, logical_page_label FROM pages WHERE document_id = X`
- Verify page matching logic in `find_pages_for_citation`

### No Chunks Found for Pages

If chunks don't match pages:
- Check chunk_pages mapping: `SELECT * FROM chunk_pages WHERE page_id = X`
- Verify document_id in chunk_metadata matches expected document

## Related Documentation

- `docs/citation_based_entity_mention_extraction.md`: Full technical documentation
- `docs/citation_based_extraction_summary.md`: Quick summary
- `concordance/ingest_concordance_tab_aware.py`: Ingest script
- `concordance/extract_entity_mentions_from_citations.py`: Extraction script
