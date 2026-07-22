# Concordance Ingestion Guide

## Overview

The concordance ingestion system (`concordance/ingest_concordance_tab_aware.py`) parses expert-curated concordance index PDFs and extracts structured data into PostgreSQL tables. The system handles complex patterns in historical intelligence documents, including cover names, aliases, citations, and entity relationships.

## Architecture

### Input
- PDF files containing concordance entries
- Each entry has a headword (entity name) and body text (description, citations, aliases)

### Output Tables
1. **concordance_sources** - Source documents (e.g., "Vassiliev and Venona Index")
2. **concordance_entries** - Raw entry text with metadata
3. **entities** - Canonical entity names (persons, cover names, topics, etc.)
4. **entity_aliases** - Alternative names for entities
5. **entity_links** - Relationships between entities (e.g., cover_name_of)
6. **entity_citations** - Citations referencing entities

## Processing Pipeline

### 1. PDF Segmentation

The system extracts entry blocks from PDFs using two methods:

#### Layout-Based Segmentation (Preferred)
- Uses `pdfplumber` to detect indentation and x-position
- Identifies headwords by left-aligned position
- Handles multi-line entries with proper continuation detection
- Skips footnote sections automatically

#### Regex-Based Segmentation (Fallback)
- Pattern-based detection of entry boundaries
- Works when layout information is unavailable
- Less accurate but more robust

### 2. Entry Parsing

Each entry block is parsed into a `ParsedEntry` object:

```python
@dataclass
class ParsedEntry:
    entity_canonical: str      # Main entity name
    entity_type: str           # "person", "cover_name", "topic", "other"
    aliases: List[ParsedAlias] # Alternative names
    links: List[ParsedLink]    # Relationships to other entities
    citations: List[ParsedCitation] # Source citations
    entry_key: str             # Original headword
    raw_text: str              # Full entry text
```

### 3. Entity Type Classification

The system classifies entries into types:

- **person**: Named individuals (e.g., "Kalinin, Mikhail Ivanovich")
- **cover_name**: Intelligence cover names (e.g., "KALIBR [CALIBER]")
- **topic**: Subject entries (e.g., "Argentina and Argentinians")
- **other**: Miscellaneous entries

Classification rules:
- Person: Headword looks like a name (comma-delimited, capitalized words)
- Cover name: Headword contains "(cover name in...)" or body mentions cover names
- Topic: Contains phrases like "related subjects", "references in"

### 4. Name Extraction and Normalization

#### Headword Processing
- **Comma inversion**: "Kalinin, Mikhail Ivanovich" → "Mikhail Ivanovich Kalinin"
- **Question mark removal**: "Kalinin, ?" → "Kalinin"
- **Quote cleanup**: `"""Glan"` → `"Glan"` → `Glan`
- **Ellipsis removal**: "BAL..." → "BAL"

#### Alias Extraction
Aliases are extracted from multiple sources:

1. **Bracket variants**: `KALIBR [CALIBER and CALIBRE]` → extracts "CALIBER" and "CALIBRE"
2. **Quoted names**: `"Kalibr"` → extracts "Kalibr"
3. **Scoped citations**: `As "Meter": ...` → extracts "Meter" as alias
4. **Cover name sections**: `Cover name in Venona: ROSE [ROZA]` → extracts "ROSE" and "ROZA"

#### "And" Pattern Splitting
- `METER and METRE` → separate aliases "METER" and "METRE"
- `BELKA [SQUIRREL]` → separate aliases "BELKA" and "SQUIRREL"
- Handles brackets: `METER and METRE [METR]` → splits before bracket processing

### 5. Referent Extraction (Cover Names → Persons)

For cover name entries, the system extracts the person they refer to. This is the most complex part of the parsing process.

#### Extraction Process

1. **Entry Key Parsing**
   - `BARCH (cover name in Venona) Semen Kremer` → extracts "Semen Kremer"
   - Handles malformed keys: `Beigel, Rose. Also know as...` → extracts from body instead

2. **Body Text Analysis**
   - Extracts first sentence before citations
   - Handles abbreviations: "J. Robert Oppenheimer" (period is part of name)
   - Stops at sentence-ending periods (not abbreviation periods)

3. **Prefix Stripping**
   - `pseudonym Andrey Shevchenko` → extracts "Andrey Shevchenko"
   - `KGB officer A. Slavyagin` → extracts "A. Slavyagin"
   - `Soviet intelligence officer Andrey Raina` → extracts "Andrey Raina"
   - Special case: `pseudonym Anatoly Gromov Anatoly Gorsky` → extracts "Anatoly Gorsky" (last two words)

4. **Temporal Qualifier Removal**
   - `Frank Oppenheimer circa 1943–1944` → "Frank Oppenheimer"
   - `Harold Smeltzer starting in October 1944` → "Harold Smeltzer"
   - `Vasily Zarubin in mid- and late 1930s` → "Vasily Zarubin"
   - Patterns: "circa", "starting in", "beginning in", "prior to", "after", "before", "from", "until", "since"

5. **Citation Text Handling**
   - `133–34; Vassiliev Yellow Notebook... Vasily Zarubin` → extracts "Vasily Zarubin"
   - `18, 21, 72, 83; Vassiliev Yellow Notebook #3, 7, 9 Vasily Zarubin` → extracts "Vasily Zarubin"
   - Extracts names that appear after citation patterns

6. **"In [Name]" Pattern Removal**
   - `Raina in Alexander Vassiliev` → "Raina" (only if single word before "in")
   - `Andrey Raina in Alexander Vassiliev` → "Andrey Raina" (preserves full name)
   - Removes citation references like "in Alexander Vassiliev's unpublished summary"

7. **"Also Know As" Extraction**
   - `Beigel, Rose. Also know as Rose Arenal...` → extracts "Rose Arenal" from "Also know as" part
   - Handles both entry key and body text patterns

8. **"Was Identified As" Pattern**
   - `was identified as Raina` → extracts "Raina"
   - `was possibly Ann Sidorovich` → extracts "Ann Sidorovich"
   - Only used if first sentence doesn't contain a clear name
   - Prevents later sentences from overriding first sentence extraction

9. **"Then" Qualifier Handling**
   - `then "Myrna"` → extracts "Myrna"
   - `then METER and METRE` → splits into "METER" and "METRE"
   - Strips "then" before processing aliases

10. **Quote Cleanup**
    - `"""Sonya"` → "Sonya"
    - `"""Glan"` → "Glan"
    - Handles mixed quote types (curly and straight quotes)
    - Multiple cleanup patterns handle various quote combinations

11. **Comma Handling**
    - `A. Slavyagin, KGB officer` → extracts "A. Slavyagin" (stops at comma if second part is job title)
    - `Beigel, Rose. Also know as...` → extracts from "Also know as" part

### 6. Link Creation

The system creates relationships between entities:

- **cover_name_of**: Cover name → Person (e.g., "KALIBR" → "David Greenglass")
- **alias_of**: Alternative name → Canonical name
- **changed_to**: Old cover name → New cover name

### 7. Citation Parsing

Citations are extracted from body text:

- **Scoped citations**: `As "Meter": Venona New York KGB 1944, 255` → links citation to "Meter" alias
- **Unscoped citations**: General citations for the entity
- **Citation validation**: Only includes citations that mention Venona/Vassiliev and contain page numbers

## Key Features

### Entity Inversion

When a cover name entry has a person referent, the system inverts the relationship:

- **Before**: Entity = "KALIBR" (cover_name), Link: "KALIBR" → "David Greenglass"
- **After**: Entity = "David Greenglass" (person), Aliases: ["KALIBR", "CALIBER", "CALIBRE"]

This ensures persons are the primary entities, with cover names as aliases.

### Complex Pattern Handling

The system handles many edge cases:

- **Multiple aliases**: `KALIBR [CALIBER and CALIBRE]` → 3 aliases
- **Temporal qualifiers**: Strips "circa", "starting in", "prior to", etc.
- **Citation interference**: Extracts names even when surrounded by citations
- **Malformed entry keys**: Falls back to body text extraction
- **Quote variations**: Handles straight quotes, curly quotes, and mixed types
- **"Or" patterns**: "Leopol or Leopolo Arenal" → uses first part (full splitting requires manual review)

### Error Handling

- **Self-links prevention**: Skips links where from_entity_id == to_entity_id
- **Question mark handling**: Removes "?" from names but preserves uncertainty in metadata
- **Incomplete brackets**: Handles `DEPARTMENT [OTDEL` (missing closing bracket)

## Usage

### Basic Ingestion

```bash
python concordance/ingest_concordance_tab_aware.py \
    --pdf path/to/concordance.pdf \
    --source-slug vassiliev_venona_index \
    --source-title "Vassiliev and Venona Index"
```

### Re-running the ingest (same pipeline)

You can re-run the ingest with the **same pipeline** (same script and options) to repopulate from the same PDF. There is no separate "pipeline version" flag; the code in `ingest_concordance_tab_aware.py` is the pipeline.

1. **Use the same source slug and title** if you want to overwrite/update the same logical source (the script upserts by `source_id`, `entry_key`, `entry_seq`). Or use a **new** slug/title to create a separate source (e.g. `concordance_index_export_20260210`).

2. **Run the ingest** (replace paths and slug/title with your values):

   ```bash
   python concordance/ingest_concordance_tab_aware.py \
       --pdf path/to/your/concordance_index.pdf \
       --source-slug vassiliev_venona_index_20260130 \
       --source-title "concordance_index_export_20260130_updated"
   ```

3. **Optional:** `--segment auto` (default), `--segment layout`, or `--segment regex`; `--marker "The Index"` and `--marker-page 7` if your PDF differs.

4. **After ingest**, run cleanup with adjudication instead of bulk delete so you can keep or fix long names (e.g. "Office of Strategic Services"):
   - **One-time confirmation:** `python scripts/cleanup_concordance.py --slug YOUR_SLUG --confirm`
   - **Interactive (d)elete / (s)kip / (e)dit per item:** `python scripts/cleanup_concordance.py --slug YOUR_SLUG --adjudicate`

5. **Optional downstream:** citation-based extraction, export, etc., using the same source slug.

### Dry Run (Testing)

```bash
python concordance/ingest_concordance_tab_aware.py \
    --pdf path/to/concordance.pdf \
    --dry-run \
    --limit 10
```

### Quick Test

```bash
# Test parsing on sample entries
python concordance/quick_test.py

# Test a specific entry
python concordance/quick_test.py --entry "Entry text here..."
```

## Testing

### Smoke Test

Run the smoke test to validate all fixed patterns:

```bash
python concordance/test_ingest_smoke.py
```

This validates fixes for real-world malformed entries:
- **Referent extraction**: Extracts person names from cover name entries
  - Strips prefixes ("pseudonym", "KGB officer", etc.)
  - Removes temporal qualifiers ("circa", "starting in", etc.)
  - Handles citation text interference
  - Extracts from "Also know as" patterns
- **Alias splitting**: Splits "and" patterns into separate aliases
  - "METER and METRE" → two aliases
  - "BELKA [SQUIRREL]" → two aliases
- **Quote cleanup**: Removes incomplete quotes
  - `"""Sonya"` → `Sonya`
  - `"""Glan"` → `Glan`
- **Ellipsis removal**: Strips trailing ellipsis
  - "BAL..." → "BAL"
- **Entity inversion**: Cover names become aliases of person entities

### Test Cases

See `concordance/test_cases_validation.py` for specific test cases covering:
- Multiple covernames with brackets and "and"
- Cross-reference covername aliases
- Unidentified covernames
- Person name comma inversion
- Question mark handling

## Common Issues and Fixes

### Issue: Malformed Referents

**Symptom**: Referent contains extra text (e.g., "Raina in Alexander Vassiliev")

**Fix**: "In [Name]" pattern removal - only removes if single word before "in"

### Issue: Combined Aliases

**Symptom**: "METER and METRE" appears as single alias

**Fix**: "And" pattern splitting - splits before bracket processing

### Issue: Incomplete Quotes

**Symptom**: `"""Sonya"` instead of `Sonya`

**Fix**: Multiple quote cleanup patterns handle mixed quote types

### Issue: Temporal Qualifiers in Referents

**Symptom**: "Frank Oppenheimer circa 1943–1944" instead of "Frank Oppenheimer"

**Fix**: Temporal qualifier removal includes "circa", "starting in", "beginning in"

### Issue: Citation Text in Referents

**Symptom**: "18, 21, 72, 83; Vassiliev... Vasily Zarubin" instead of "Vasily Zarubin"

**Fix**: Citation pattern matching extracts name after citation text

### Issue: Job Titles in Referents

**Symptom**: "A. Slavyagin, KGB officer" instead of "A. Slavyagin"

**Fix**: Comma handling stops at job titles (KGB officer, Soviet intelligence officer, etc.)

## Data Flow

```
PDF File
  ↓
Entry Segmentation (Layout/Regex)
  ↓
Entry Blocks
  ↓
parse_entry_block()
  ↓
ParsedEntry
  ↓
Entity Type Classification
  ↓
Name Extraction & Normalization
  ↓
Referent Extraction (if cover_name)
  ↓
Alias Extraction
  ↓
Link Creation
  ↓
Citation Parsing
  ↓
Database Insertion
  ↓
PostgreSQL Tables
```

## Key Functions

- `parse_entry_block()`: Main parsing function
- `infer_referent_from_body_start()`: Extracts person referent from cover name entries
- `looks_like_person_head()`: Determines if text looks like a person name
- `invert_comma_delimited_name()`: Converts "Last, First" to "First Last"
- `strip_outer_quotes()`: Removes surrounding quotes
- `parse_citation_chunks()`: Splits citation text into chunks

## Dependencies

- `psycopg2`: PostgreSQL database connection
- `pdfplumber`: PDF layout analysis (optional but recommended)
- `pypdf` or `PyPDF2`: PDF text extraction (fallback)
