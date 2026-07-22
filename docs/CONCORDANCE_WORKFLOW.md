# Complete Workflow: Ingest Concordance Index and Extract Entity Mentions

This guide walks you through the complete process of:
1. Ingesting the concordance index PDF
2. Exporting concordance data
3. Extracting entity mentions from chunks

For **Venona** and **Vassiliev** collections.

## Prerequisites

- Database is running and accessible
- Concordance index PDF file(s) available
- Documents and chunks already ingested for venona and vassiliev collections

## Step 1: Ingest Concordance Index

The concordance index contains expert-curated entity information with aliases, citations, and relationships.

### For Venona + Vassiliev Combined Index

If you have a single PDF that covers both Venona and Vassiliev:

```bash
python concordance/ingest_concordance_tab_aware.py \
  --pdf "path/to/concordance_index.pdf" \
  --source-slug "vassiliev_venona_index" \
  --source-title "Vassiliev-Venona Combined Index" \
  --source-notes "Expert-curated index covering both Vassiliev notebooks and Venona decrypts"
```

### For Separate Indexes

If you have separate PDFs:

**Venona Index:**
```bash
python concordance/ingest_concordance_tab_aware.py \
  --pdf "path/to/venona_index.pdf" \
  --source-slug "venona_index" \
  --source-title "Venona Index" \
  --source-notes "Expert-curated index for Venona decrypts"
```

**Vassiliev Index:**
```bash
python concordance/ingest_concordance_tab_aware.py \
  --pdf "path/to/vassiliev_index.pdf" \
  --source-slug "vassiliev_index" \
  --source-title "Vassiliev Index" \
  --source-notes "Expert-curated index for Vassiliev notebooks"
```

### Options

- `--dry-run`: Parse and print without writing to database (useful for testing)
- `--limit N`: Only process first N entries (useful with --dry-run)
- `--verbose`: Print detailed debug information
- `--segment {auto,layout,regex}`: Segmentation method (default: auto)
- `--marker "The Index"`: Marker string to find start of index (default: "The Index")
- `--marker-page 7`: Page number containing marker (default: 7, 1-based)

### What This Creates

- `concordance_sources`: Metadata about the index source
- `concordance_entries`: Raw entry text from the PDF
- `entities`: Entity records extracted from entries
- `entity_aliases`: Aliases for each entity (e.g., "Yakubovich", "Unidentified Soviet intelligence officer/agent")
- `entity_links`: Relationships between entities (e.g., "As X:" links)
- `entity_citations`: Citations with document names and page numbers

## Step 2: Export Concordance Data (Optional but Recommended)

Export the ingested data to CSV files for inspection:

```bash
# Export all concordance sources
python scripts/export_concordance_data.py --output-dir concordance_export

# Or export specific source
python scripts/export_concordance_data.py \
  --output-dir concordance_export \
  --source-slug "vassiliev_venona_index"
```

### Output Files

- `concordance_entries.csv`: All entries from the index
- `entities.csv`: All entities extracted
- `entity_aliases.csv`: All aliases for entities
- `entity_links.csv`: Relationships between entities
- `entity_citations.csv`: Citations with document/page references
- `entity_mentions.csv`: Existing mentions (if any)

## Step 3: Extract Entity Mentions from Chunks

Now extract entity mentions from your document chunks using the concordance index for disambiguation.

### For Venona Collection

```bash
python scripts/extract_entity_mentions.py \
  --collection venona \
  --enable-partial \
  --enable-fuzzy
```

### For Vassiliev Collection

```bash
python scripts/extract_entity_mentions.py \
  --collection vassiliev \
  --enable-partial \
  --enable-fuzzy
```

### Options

- `--dry-run`: Show what would be extracted without inserting (recommended first)
- `--limit N`: Process only first N chunks (for testing)
- `--show-samples`: Show sample mentions found
- `--max-samples N`: Number of samples to show (default: 10)
- `--enable-partial`: Enable partial matching (last names, etc.) - **enabled by default**
- `--enable-fuzzy`: Enable fuzzy matching (misspellings) - **enabled by default**
- `--disable-partial`: Disable partial matching
- `--disable-fuzzy`: Disable fuzzy matching
- `--batch-size N`: Process chunks in batches (default: 100)

### What This Creates

- `entity_mentions`: Mentions found in chunks with:
  - `entity_id`: The entity matched
  - `chunk_id`: The chunk containing the mention
  - `document_id`: The document containing the chunk
  - `surface`: The actual text found (e.g., "Yakubovich", "Smith")
  - `surface_norm`: Normalized version
  - `surface_quality`: 'exact', 'approx'
  - `confidence`: Match confidence (1.0 for exact, 0.7-0.8 for partial/fuzzy)
  - `method`: 'alias_exact', 'alias_partial', 'alias_fuzzy', etc.

## Step 4: Verify Results

### Check Match Statistics

The extraction script outputs:
- Total mentions found
- Match type breakdown (exact/partial/fuzzy)
- Collision statistics
- Policy blocks

### Verify Specific Entities

Check if specific entities were extracted:

```sql
SELECT 
  em.id,
  e.canonical_name,
  em.surface,
  em.method,
  em.confidence,
  cm.document_id,
  d.source_name
FROM entity_mentions em
JOIN entities e ON em.entity_id = e.id
JOIN chunk_metadata cm ON em.chunk_id = cm.chunk_id
JOIN documents d ON cm.document_id = d.id
WHERE e.canonical_name LIKE '%Yakubovich%'
ORDER BY em.id DESC
LIMIT 20;
```

### Export Results

Export entity mentions to CSV:

```bash
python scripts/export_concordance_data.py --output-dir concordance_export
```

This will update `entity_mentions.csv` with all mentions including newly extracted ones.

## Complete Example Workflow

```bash
# 1. Ingest concordance index (dry-run first to verify)
python concordance/ingest_concordance_tab_aware.py \
  --pdf "data/vassiliev_venona_index.pdf" \
  --source-slug "vassiliev_venona_index" \
  --source-title "Vassiliev-Venona Combined Index" \
  --dry-run \
  --limit 10

# 2. If dry-run looks good, ingest for real
python concordance/ingest_concordance_tab_aware.py \
  --pdf "data/vassiliev_venona_index.pdf" \
  --source-slug "vassiliev_venona_index" \
  --source-title "Vassiliev-Venona Combined Index"

# 3. Export to verify ingestion
python scripts/export_concordance_data.py \
  --output-dir concordance_export \
  --source-slug "vassiliev_venona_index"

# 4. Extract mentions from Venona (dry-run first)
python scripts/extract_entity_mentions.py \
  --collection venona \
  --enable-partial \
  --enable-fuzzy \
  --dry-run \
  --limit 50

# 5. If dry-run looks good, extract for real
python scripts/extract_entity_mentions.py \
  --collection venona \
  --enable-partial \
  --enable-fuzzy

# 6. Extract mentions from Vassiliev
python scripts/extract_entity_mentions.py \
  --collection vassiliev \
  --enable-partial \
  --enable-fuzzy

# 7. Export final results
python scripts/export_concordance_data.py --output-dir concordance_export
```

## Troubleshooting

### "Yakubovich (auto_match disabled)"

This should now be fixed - entities with "unidentified" aliases have all their aliases enabled. If you still see this, the entity may not have an "unidentified" alias, or there's another issue.

### No matches found

- Check that documents/chunks are ingested for the collection
- Verify entity aliases exist: `SELECT COUNT(*) FROM entity_aliases;`
- Check if aliases are matchable: `SELECT COUNT(*) FROM entity_aliases WHERE is_matchable = true;`

### Too many collisions

- Review collision statistics in the output
- Check if citation disambiguation is working (requires `entity_citations` to be populated)
- Consider adjusting collision thresholds in the code

## Next Steps

After extraction:
1. Review match type breakdown to see how many partial/fuzzy matches were found
2. Check collision statistics to see which entities need review
3. Use citation-based disambiguation to resolve ambiguous matches
4. Export and review `entity_mentions.csv` for quality assurance
