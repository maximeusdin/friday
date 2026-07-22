# Document Ingest Reference

This document tracks all corpus ingestion work, including strategies, scripts, and results for each collection. Use this as a reference when adding new sources.

---

## Table of Contents

1. [Overview](#overview)
2. [McCarthy Transcripts](#mccarthy-transcripts)
3. [Rosenberg FBI Files](#rosenberg-fbi-files)
4. [FBI SOLO Operation Files](#fbi-solo-operation-files)
5. [HUAC Hearings](#huac-hearings)
6. [HUAC Reports](#huac-reports)
7. [FBI COMRAP Files](#fbi-comrap-files)
8. [Common Patterns](#common-patterns)
9. [Adding New Collections](#adding-new-collections)
10. [Two-Stage vs One-Stage Pipelines](#two-stage-vs-one-stage-pipelines)
11. [Venona Chunking](#venona-chunking)
12. [Vassiliev Chunking](#vassiliev-chunking)
13. [Silvermaster Chunking](#silvermaster-chunking)
14. [Strategy Comparison](#strategy-comparison-venona--vassiliev-vs-huac)
15. [Complete Coverage Summary](#complete-coverage-summary)

---

## Overview

### Pipeline Architecture

All ingests follow this general flow:

```
PDF Files → Text Extraction (PyMuPDF) → Normalization → Chunking → Database
                                              ↓
                                    Collection-specific strategy
```

### Database Schema

Key tables used by all ingests:
- `collections` - Collection metadata (slug, title, description)
- `documents` - Individual source files
- `pages` - Page-level content with raw text
- `chunks` - Chunked text for retrieval
- `chunk_pages` - Maps chunks to source pages
- `chunk_metadata` - Additional chunk context (content_type, document_id)

### Running Ingests

All ingest scripts are in `scripts/` and can be run directly:

```bash
python scripts/ingest_<collection>.py [options]
```

Common options:
- `--dry-run` - Analyze without writing to database
- `--limit N` - Process only N files (for testing)
- `--input-dir` - Override default input directory

Assessment scripts verify ingest quality:

```bash
python scripts/assess_<collection>_ingest.py
```

---

## McCarthy Transcripts

### Source
- **Location**: `data/raw/mccarthy/`
- **Format**: PDF transcripts of McCarthy hearings
- **Content**: Congressional hearing transcripts with speaker turns

### Strategy: Turn-Aware Chunking

McCarthy transcripts are structured dialogues with clear speaker patterns. The ingest preserves speaker turns as the fundamental unit.

**Key insight**: Congressional hearings have a Q&A structure where context depends on who is speaking. Chunking by character count would break mid-sentence or mid-exchange.

### Scripts
- **Ingest**: `scripts/ingest_mccarthy_v2.py`
- **Assessment**: `scripts/assess_mccarthy_ingest.py`

### Implementation Details

1. **Speaker Detection**: Regex patterns identify speaker labels:
   - `Mr. COHN:`, `Senator McCARTHY:`, `The CHAIRMAN:`, etc.
   - Pattern: `^(Mr\.|Mrs\.|Miss|Dr\.|Senator|The)\s+[A-Z]+[A-Za-z]*\.?\s*:`

2. **Turn Extraction**: Text is split into turns, each with:
   - Speaker name
   - Turn text
   - Page references

3. **Chunk Assembly**: Turns are grouped into chunks:
   - Target: ~4000 chars per chunk
   - Respects turn boundaries (never splits mid-turn)
   - Overlap: includes last turn of previous chunk

4. **Turn Tracking**: `transcript_turns` and `chunk_turns` tables track:
   - Speaker identity per turn
   - Turn sequence within document
   - Chunk-turn relationships

### Results

| Metric | Value |
|--------|-------|
| Collection slug | `mccarthy` |
| Documents | ~50 |
| Content type | `transcript` |
| Avg chunk size | ~4,000 chars |

### Quality Checks
- Verify speaker names are normalized
- Check turn boundaries are preserved
- Confirm Q&A exchanges stay together

---

## Rosenberg FBI Files

### Source
- **Location**: `data/raw/rosenberg/`
- **Format**: 181 PDF files of FBI case documents
- **Content**: FBI investigative files on Julius Rosenberg espionage case

### Strategy: Document-Aware Chunking

FBI case files contain multiple logical documents within each PDF (memos, reports, teletypes). The ingest detects document boundaries and chunks accordingly.

**Key insight**: Each PDF contains multiple FBI documents with distinct headers. Treating each PDF as a single document would lose the memo-level structure.

### Scripts
- **Ingest**: `scripts/ingest_rosenberg.py`
- **Assessment**: `scripts/assess_rosenberg_ingest.py`

### Implementation Details

1. **Page Classification**: Each page is classified:
   - `cover_page` - Cover sheets, routing slips (skipped)
   - `file_index` - Index pages (skipped)
   - `primary` - Main content (processed)

2. **Document Boundary Detection**: New documents detected by:
   - Memo headers (TO/FROM/SUBJECT)
   - Classification stamps
   - Date headers
   - FBI form numbers

3. **Metadata Extraction**: From document headers:
   - Date
   - TO/FROM fields
   - Subject line
   - File number
   - Document type (memo, teletype, report, letter)

4. **Chunking**: Within detected documents:
   - Target: ~6000 chars
   - Respects document boundaries
   - Page-based splitting for large documents

5. **Index Size Handling**: PostgreSQL trigram index has 8KB limit:
   - `safe_truncate_bytes()` function
   - Retry logic with progressively smaller sizes
   - Savepoints for graceful failure handling

### Results

| Metric | Value |
|--------|-------|
| Collection slug | `rosenberg` |
| Documents | 181 |
| Total pages | ~18,000 |
| Content types | `fbi_memo`, `fbi_teletype`, `fbi_report`, `fbi_letter` |
| Avg chunk size | ~5,400 chars |
| Page coverage | ~96% |

### Quality Checks
- Documents without chunks: 0
- Page coverage > 95%
- Content type distribution reasonable

---

## FBI SOLO Operation Files

### Source
- **Location**: `data/raw/solo/`
- **Format**: 122 PDF files from FBI case 100-HQ-428091
- **Content**: FBI surveillance of CPUSA through informants Morris and Jack Childs (1958-1977)

### Strategy: Memo-Aware Chunking

SOLO files contain thousands of individual FBI memos. The ingest detects memo boundaries and keeps each memo as a single chunk when possible.

**Key insight**: Each memo is a self-contained intelligence unit. Preserving memo integrity improves retrieval quality and enables proper attribution.

### Scripts
- **Ingest**: `scripts/ingest_solo.py`
- **Assessment**: `scripts/assess_solo_ingest.py`

### Implementation Details

1. **File Types**: Three categories identified:
   - Serial Files (115 files, 23,607 pages) - Main case serials
   - EBF Files (3 files, 431 pages) - Electronic surveillance
   - SOLO Files (4 files, 79 pages) - Summary documents

2. **Memo Detection**: Score-based boundary detection:
   ```
   +2 points: TO: DIRECTOR/SAC
   +2 points: FROM: SAC/DIRECTOR
   +2 points: AIRTEL/TELETYPE header
   +1.5 points: MEMORANDUM header
   +1.5 points: "Priority or Method of Mailing"
   +1 point: OFFICE MEMORANDUM
   +1 point: RE:/SUBJECT: line
   +0.5 points: File number (100-428091)
   +0.5 points: Date
   
   Threshold: score >= 2.0 indicates memo start
   ```

3. **Memo Type Classification**:
   - `fbi_memo` - Standard memoranda (52%)
   - `fbi_airtel` - Urgent air communications (33%)
   - `fbi_teletype` - Teletype messages (6%)
   - `fbi_urgent` - Urgent priority messages (1%)
   - `fbi_document` - Other/unclassified (8%)

4. **Chunking Strategy**:
   - Small memos (<6000 chars): Single chunk
   - Large memos (>6000 chars): Split by page with 500-char overlap
   - Track memo_part and memo_total_parts for split memos

5. **Metadata Extraction**:
   - TO/FROM fields
   - Subject line
   - Date
   - Serial range (from filename)

### Results

| Metric | Value |
|--------|-------|
| Collection slug | `solo` |
| Documents | 122 |
| Total pages | 24,117 |
| Memos detected | 6,295 |
| Total chunks | 9,529 |
| Avg pages/memo | 3.7 |
| Avg chunks/memo | 1.5 |
| Page coverage | 94.3% |

### Chunk Size Distribution

| Size | Count | % |
|------|-------|---|
| <1k | 954 | 10.0% |
| 1k-2k | 1,843 | 19.3% |
| 2k-3k | 1,406 | 14.8% |
| 3k-4k | 810 | 8.5% |
| 4k-5k | 984 | 10.3% |
| 5k-6k | 1,769 | 18.6% |
| 6k-7k | 1,692 | 17.8% |
| >7k | 71 | 0.7% |

### Search Quality Test

| Term | Chunks | Description |
|------|--------|-------------|
| SOLO | 4,232 | Operation name |
| Communist Party | 3,880 | Target organization |
| Soviet | 5,575 | Foreign connection |
| Morris Childs | 291 | Key informant |
| Jack Childs | 79 | Key informant |
| Director | 4,962 | FBI communication |
| Chicago | 4,863 | Primary field office |

### Quality Checks
- Documents without chunks: 0 ✓
- Empty chunks: 2 (negligible)
- Small chunks (<200): 17 (0.2%) ✓
- Page coverage: 94.3% ✓

---

## HUAC Hearings

### Source
- **Location**: `data/raw/umamerican_hearings/` (transcript files only)
- **Format**: PDF transcripts of House Un-American Activities Committee hearings
- **Content**: Congressional hearing transcripts with speaker turns

### Strategy: Turn-Aware Chunking

HUAC hearing transcripts are structured dialogues similar to McCarthy hearings. The ingest detects speaker patterns (accounting for OCR errors) and preserves speaker turns as the fundamental unit.

**Key hearings included**:
- 1948 Communist Espionage hearings (Alger Hiss case)
- 1947 Hollywood hearings (motion picture industry)
- Soviet activity investigations

### Scripts
- **Ingest**: `scripts/ingest_huac_hearings.py`
- **Assessment**: `scripts/assess_huac_hearings_ingest.py`

### Implementation Details

1. **Document Classification**: Automatically detects transcripts vs reports
   - Speaker density > 50 in pages 5-55 → transcript
   - Otherwise → report (skipped, handled by reports script)

2. **Speaker Detection**: OCR-tolerant pattern matching
   ```python
   # Handles mixed case from OCR: MR NIXON, Mr. Nixon, Mr. Nixox
   r"^((?:Mr|Mrs|Ms|Miss|Dr|Senator|Representative|The)\s*[.\s]*\s*[A-Za-z][A-Za-z]+)\s*[.\s]+\s*(?=[A-Z])"
   ```

3. **HUAC-Specific Roles**:
   - `chair` - THE CHAIRMAN, MR THOMAS, MR WOOD
   - `committee` - MR NIXON, MR MUNDT, MR RANKIN, etc.
   - `counsel` - MR STRIPLING, MR TAVENNER, MR RUSSELL
   - `witness` - Default for other speakers

4. **Metadata Extraction**:
   - Year from filename or content
   - Topic keywords (Hollywood, Hiss, Chambers, Espionage, Soviet)

### Results

| Metric | Value |
|--------|-------|
| Collection slug | `huac_hearings` |
| Documents | 8 |
| Total pages | 2,354 |
| Total turns | 33,144 |
| Total chunks | 2,139 |
| Avg chunk length | 4,177 chars |
| Avg turns/chunk | 17.5 |

### Notable Speakers Found

| Speaker | Chunks | Role |
|---------|--------|------|
| MR STRIPLING | 787 | Chief Investigator |
| THE CHAIRMAN | 636 | Committee Chair |
| MR NIXON | 388 | Committee Member |
| MR HISS | 180 | Famous Witness |
| MR CHAMBERS | 121 | Famous Witness |
| MISS BENTLEY | 106 | Famous Witness |

### Search Quality Test

| Term | Chunks | Description |
|------|--------|-------------|
| Communist Party | 806 | Target organization |
| espionage | 921 | Investigation focus |
| Hollywood | 377 | 1947 hearings |
| Alger Hiss | 141 | Famous case |

### Quality Checks
- Documents without chunks: 0 ✓
- Chunks < 200 chars: 0 ✓
- Empty chunks: 0 ✓

---

## HUAC Reports

### Source
- **Location**: `data/raw/unamerican_reports/` + report files from `data/raw/umamerican_hearings/`
- **Format**: PDF annual reports and special publications
- **Content**: Committee findings, analyses, and documentation (1948-1964)

### Strategy: Page-Based Chunking

HUAC reports are narrative documents without speaker turns. Uses simple page-based chunking with character targets.

### Scripts
- **Ingest**: `scripts/ingest_huac_reports.py`
- **Assessment**: `scripts/assess_huac_reports_ingest.py`

### Implementation Details

1. **Document Types**:
   - 17 Annual HUAC reports (1948-1964)
   - 4 Special publications (Communist Conspiracy, Spotlight on Spies, etc.)

2. **Chunking Parameters**:
   - Target: 5000 chars per chunk
   - Max: 7000 chars
   - Overlap: 500 chars between chunks

3. **Content Types Generated**:
   - `huac_annual_report_{year}` - For dated reports
   - `huac_report` - For undated or special reports

4. **Page Classification**:
   - Skip nearly blank pages (<100 chars)
   - Mark but keep TOC and index pages

### Results

| Metric | Value |
|--------|-------|
| Collection slug | `huac_reports` |
| Documents | 21 |
| Total pages | 2,335 |
| Pages used | 2,166 (92.8%) |
| Total chunks | 1,956 |
| Avg chunk length | 3,782 chars |
| Median chunk length | 3,737 chars |

### Chunk Size Distribution

| Size | Count | % |
|------|-------|---|
| <2k | 5 | 0.3% |
| 2k-4k | 1,480 | 75.7% |
| 4k-6k | 470 | 24.0% |
| >6k | 1 | 0.1% |

### Coverage by Year

Full annual report coverage from 1948-1964, plus special publications.

### Search Quality Test

| Term | Chunks | Description |
|------|--------|-------------|
| Communist Party | 1,148 | Main subject |
| Soviet Union | 386 | Foreign power |
| subversive | 394 | Key term |
| investigation | 632 | Activity type |

### Quality Checks
- Documents without chunks: 0 ✓
- Chunks < 200 chars: 0 ✓
- Page coverage: 100% ✓

---

## Common Patterns

### OCR Text Normalization

All FBI document ingests share common OCR cleanup:

```python
def normalize_text(raw_text: str) -> str:
    text = raw_text
    text = text.replace("\u00a0", " ")  # NBSP -> space
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # Fix hyphenation
    text = re.sub(r'[ \t]+', ' ', text)  # Collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse newlines
    return text.strip()
```

### Handling Trigram Index Limits

PostgreSQL GIST trigram indexes have an 8KB row limit. Use this pattern:

```python
def safe_truncate_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore')

def insert_chunk(cur, text: str, ...):
    max_sizes = [None, 6000, 4000, 2500]
    for i, max_size in enumerate(max_sizes):
        text_to_insert = text if max_size is None else safe_truncate_bytes(text, max_size)
        savepoint = f"insert_retry_{i}"
        try:
            cur.execute(f"SAVEPOINT {savepoint}")
            cur.execute("INSERT INTO chunks ...", (text_to_insert, ...))
            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            return result
        except Exception as e:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            if "index row requires" in str(e) and "maximum size is 8191" in str(e):
                continue
            raise
```

### Collection Registration

All collections use this pattern:

```python
COLLECTION_SLUG = "collection_name"
COLLECTION_TITLE = "Human-Readable Title"
COLLECTION_DESCRIPTION = """Multi-line description..."""

def get_or_create_collection(cur) -> int:
    cur.execute("SELECT id FROM collections WHERE slug = %s", (COLLECTION_SLUG,))
    r = cur.fetchone()
    if r:
        return int(r[0])
    cur.execute(
        "INSERT INTO collections (slug, title, description) VALUES (%s, %s, %s) RETURNING id",
        (COLLECTION_SLUG, COLLECTION_TITLE, COLLECTION_DESCRIPTION),
    )
    return int(cur.fetchone()[0])
```

---

## Adding New Collections

### Checklist

1. **Analyze Source Material**
   - [ ] Examine file format and structure
   - [ ] Identify document boundaries
   - [ ] Assess OCR quality
   - [ ] Determine optimal chunking strategy

2. **Create Ingest Script**
   - [ ] Copy closest existing script as template
   - [ ] Implement source-specific detection patterns
   - [ ] Add appropriate content type classification
   - [ ] Handle edge cases (large docs, poor OCR, etc.)

3. **Test Thoroughly**
   - [ ] Run with `--dry-run --limit 5`
   - [ ] Check chunk size distribution
   - [ ] Verify content type classification
   - [ ] Test on full dataset

4. **Create Assessment Script**
   - [ ] Adapt from existing assessment script
   - [ ] Add source-specific quality checks
   - [ ] Include relevant search terms

5. **Document**
   - [ ] Add section to this document
   - [ ] Record key decisions and results
   - [ ] Note any issues or limitations

### Strategy Selection Guide

| Source Type | Recommended Strategy | Example |
|-------------|---------------------|---------|
| Transcripts with speakers | Turn-aware | McCarthy |
| FBI case files with memos | Memo-aware | SOLO, Rosenberg |
| Single long documents | Page-based with overlap | EBF reports |
| Structured forms | Form-field extraction | (future) |
| Mixed content | Hybrid with classification | (future) |

---

## FBI COMRAP Files

### Source
- **Location**: `data/raw/FBICOMRAP/`
- **Format**: 3 PDF files (FBI Comintern Apparatus investigation)
- **Content**: FBI investigation reports on Communist International activities

### Strategy: Page-Based Chunking with Boilerplate Detection

COMRAP files are FBI investigative reports with standardized boilerplate headers.

### Scripts
- **Ingest**: `scripts/ingest_fbicomrap.py`

### Implementation Details

1. **Boilerplate Detection**: Identifies repeated patterns across pages (threshold 35%)
2. **OCR Normalization**: Fixes hyphenation, collapses whitespace
3. **Chunking**: Target 5000 chars, max 8000, 1000 char overlap

### Results

| Metric | Value |
|--------|-------|
| Collection slug | `fbicomrap` |
| Documents | 3 |
| Total pages | 613 |
| Total chunks | 314 |
| Pipeline version | `fbicomrap_v1` |

---

## Two-Stage vs One-Stage Pipelines

The codebase uses two distinct pipeline architectures for ingesting documents:

### One-Stage Pipeline (Ingest + Chunk Together)

Used by: **McCarthy, Rosenberg, SOLO, HUAC Hearings, HUAC Reports**

```
PDF → ingest_*.py → pages + chunks tables (one script does everything)
```

**Characteristics**:
- Single script handles PDF extraction, page storage, and chunking
- Chunks are created directly during ingest
- Tightly coupled: re-chunking requires re-ingesting
- Simpler to run (one command)

**When to use**:
- When chunking logic is specific to the source format
- When you want a simple, single-command workflow
- When re-chunking is unlikely to be needed

### Two-Stage Pipeline (Ingest, then Chunk Separately)

Used by: **Venona, Vassiliev, Silvermaster**

```
Stage 1: PDF → ingest_*.py → pages table (stores raw content)
Stage 2: pages → chunk_corpus.py → chunks table (rerunnable)
```

**Characteristics**:
- Ingest script only extracts pages (no chunking)
- Separate chunking script reads from pages table
- Loosely coupled: can re-chunk without re-ingesting
- More flexible for experimentation

**When to use**:
- When you want to experiment with different chunking strategies
- When the source has complex structure requiring iteration
- When page-level data has independent value (citations, markers)

### Unified Chunking Script: `chunk_corpus.py`

The two-stage collections share a single chunking script with collection-specific strategies:

```bash
# Venona: 1 message/page = 1 chunk (trivial mapping)
python scripts/chunk_corpus.py --collection venona --pipeline-version chunk_v2_venona_msg_fallback24k

# Vassiliev: marker-aware chunking using p.xx boundaries
python scripts/chunk_corpus.py --collection vassiliev --pipeline-version chunk_v2_vass_marker_stream_4k --max-chars 4000

# Silvermaster: FBI structure-aware chunking
python scripts/chunk_corpus.py --collection silvermaster --pipeline-version chunk_v2_silver_4k --max-chars 4000
```

**Strategy routing** (from `pick_strategy()`):
```python
if collection_slug == "venona":
    return "venona_message"
if collection_slug == "vassiliev":
    return "vassiliev_marker_stream"
if collection_slug == "silvermaster":
    return "silvermaster_structured"
return "fallback_paragraph"
```

### Pipeline Comparison Table

| Collection | Architecture | Ingest Script | Chunking |
|------------|--------------|---------------|----------|
| McCarthy | One-stage | `ingest_mccarthy_v2.py` | Built-in (turn-aware) |
| Rosenberg | One-stage | `ingest_rosenberg.py` | Built-in (document-aware) |
| SOLO | One-stage | `ingest_solo.py` | Built-in (memo-aware) |
| HUAC Hearings | One-stage | `ingest_huac_hearings.py` | Built-in (turn-aware) |
| HUAC Reports | One-stage | `ingest_huac_reports.py` | Built-in (page-based) |
| Venona | Two-stage | `ingest_venona_pdf.py` | `chunk_corpus.py` (message) |
| Vassiliev | Two-stage | `ingest_vassiliev_pdf.py` | `chunk_corpus.py` (marker) |
| Silvermaster | Two-stage | (separate ingest) | `chunk_corpus.py` (structured) |

---

## Venona Chunking

### Overview

Venona uses a two-stage pipeline where:
1. **Ingest** (`ingest_venona_pdf.py`): Stores decoded cables as "message pages" (one row per cable, not per PDF page)
2. **Chunk** (`chunk_corpus.py`): Maps messages 1:1 to chunks (with fallback for oversized messages)

### Chunking Strategy: `venona_message`

Since pages are already message-level units from the ingest, chunking is essentially a pass-through:

```python
def chunk_venona_one_message_per_page(pages, oversize_max_chars=24000):
    for p in pages:
        txt = (p.raw_text or "").strip()
        if not txt:
            continue
        
        # Normal case: 1 message = 1 chunk
        if len(txt) <= oversize_max_chars:
            out.append(ChunkSpec(text=txt, clean_text=txt, page_ids_in_order=[p.id]))
            continue
        
        # Fallback: split oversized messages by paragraphs
        paras = split_paragraphs(txt)
        # ... pack paragraphs into <= oversize_max_chars chunks
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--venona-max-chars` | 24000 | Maximum chars before splitting |
| `--pipeline-version` | Required | Version string for chunk tracking |

### Why This Works

- Venona cables are typically 500-5000 characters
- The 24k fallback handles rare oversized messages
- Message integrity is preserved (no mid-cable splits)
- Citation accuracy maintained (chunk → single message page)

---

## Vassiliev Chunking

### Overview

Vassiliev uses a two-stage pipeline where:
1. **Ingest** (`ingest_vassiliev_pdf.py`): Stores raw PDF pages with `p.xx` markers in metadata
2. **Chunk** (`chunk_corpus.py`): Creates marker-aware chunks that respect notebook pagination

### Chunking Strategy: `vassiliev_marker_stream`

The Vassiliev notebooks contain `p.xx` markers indicating original notebook page numbers (distinct from PDF page numbers). The chunking strategy:

1. **Treats markers as boundaries** (excluded from chunk text)
2. **Groups content between markers** into segments
3. **Splits segments** into ≤4000 char chunks by paragraph
4. **Never merges across marker boundaries** (preserves citation integrity)

```python
PXX_LINE_RE = re.compile(r"(?m)^\s*(p\.\s*\d+)\s*$")

def iter_vassiliev_marker_segments(pages):
    """
    Rules:
    - p.xx lines are BOUNDARIES, excluded from segment text
    - Text between markers forms a "segment"
    - Segments can span multiple PDF pages
    - Consecutive markers with no content are ignored (no empty chunks)
    """
    for p in pages:
        for ln in page_lines:
            mk = is_marker_line(ln)
            if mk is not None:
                flush_if_content()  # End current segment
                current_marker = mk  # Start new segment
                continue
            # Non-marker content added to current segment
            buf_lines.append(ln)
```

### Marker Examples

```
p.71        ← Marker line (boundary, excluded from chunks)
Some text from the notebook that discusses...
More content here...
p.72        ← Next marker (starts new segment)
Different content...
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--max-chars` | 4000 | Target chunk size |
| `--pipeline-version` | Required | Version string |

### Why This Works

- `p.xx` markers indicate original research notes pagination
- Keeping marker boundaries ensures citation accuracy
- 4000 char limit produces reasonable retrieval units
- Multi-PDF-page segments stay together until marker change

---

## Silvermaster Chunking

### Overview

Silvermaster uses a two-stage pipeline for FBI case files with complex structure:
1. **Ingest**: Stores raw PDF pages
2. **Chunk** (`chunk_corpus.py`): Structure-aware chunking with OCR quality gating

### Chunking Strategy: `silvermaster_structured`

FBI Silvermaster files contain multiple document types with varying OCR quality. The strategy:

1. **Detects document unit boundaries** (memo headers, FOIPA sheets)
2. **Isolates low-quality OCR pages** (prevents contamination)
3. **Cleans OCR artifacts** (garbage lines, boilerplate)
4. **Chunks within units** (never merges across memo boundaries)

### Document Boundary Detection

```python
# Strong unit anchors (at top of page only)
UNIT_ANCHOR_RE = re.compile(
    r"(?im)^\s*(FEDERAL\s+BUREAU\s+OF\s+INVESTIGATION|"
    r"Office\s+Memorandum|UNITED\s+STATES\s+GOVERNMENT|"
    r"AIR(TEL|MAIL)|MEMORANDUM)\b"
)

# FOIPA deleted page sheets → always standalone unit
FOIPA_SHEET_RE = re.compile(
    r"(?is)FOIPA\s+DELETED\s+PAGE\s+INFORMATION\s+SHEET|"
    r"Page\(s\)\s+withheld|Deleted\s+Page"
)

# Subject/date lines near top of page
SUBJECT_RE = re.compile(r"(?im)^\s*(Re:|SUBJECT:|SUBJ:)\s+.+$")
DATE_LINE_RE = re.compile(r"(?im)^\s*(January|February|...)\\s+\\d{1,2},\\s+\\d{4}\\s*$")
```

### OCR Quality Gating

```python
def page_quality_score(text):
    """
    Returns 0..1 where 1 is clean English text.
    Based on: alphabetic ratio + word density
    """
    letters = sum(ch.isalpha() for ch in s)
    letter_ratio = letters / max(len(s), 1)
    # ... combine with word density
    return score

# Pages with score < 0.25 are isolated (kept for citation, but not merged with neighbors)
```

### OCR Garbage Detection

```python
def is_ocr_garbage_line(line):
    # 6+ repeated characters
    if re.search(r"(.)\1\1\1\1\1", s):
        return True
    # Low alphanumeric ratio in long lines
    if len(s) >= 30 and alnum / len(s) < 0.35:
        return True
    # Scattered single characters
    if re.fullmatch(r"(?:[A-Za-z0-9]\s+){10,}[A-Za-z0-9]?", s):
        return True
    return False
```

### Text Cleaning

```python
def clean_silvermaster_text(raw):
    # Remove boilerplate-only lines
    # Remove OCR garbage lines
    # Dehyphenate: "inter-\nnational" → "international"
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
    # Collapse excessive whitespace
    return text.strip()
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--max-chars` | 4000 | Target chunk size |
| `--pipeline-version` | Required | Version string |

### Why This Works

- FBI files have highly variable OCR quality
- Isolating bad pages prevents retrieval pollution
- Memo boundaries are semantically meaningful
- Cleaning improves search and embedding quality

---

The Venona and Vassiliev collections use fundamentally different ingest strategies than the HUAC and other hearing/FBI document collections. Understanding these differences is important for choosing the right approach for new sources.

### Venona: Message-Level Page Storage

**Script**: `scripts/ingest_venona_pdf.py`

**Strategy**: Store each decoded cable/message as a single "page" row (not a physical PDF page).

**Key Characteristics**:
- **Logical unit**: Individual decoded Soviet cable message
- **Page definition**: One row per message (may span multiple PDF pages)
- **No chunks table**: Messages stored directly in `pages` table
- **Heavy metadata**: USSR ref no, From/To routing, cable numbers, dates, reissue flags

**Message Detection**:
```python
# Strong signal: USSR Ref. No anywhere on page
USSR_REF_ANY_RE = re.compile(r"(?i)\bUSSR\s+Ref\.?\s*No\.?\s*:\s*([A-Za-z0-9/()\-]+)")

# Alternative: From + To + No cluster in header zone
# Requires all three within 15 lines of each other
```

**Why This Approach**:
- Venona cables are self-contained intelligence units
- Each cable has structured routing metadata (From/To/No)
- Multi-page cables should stay together as single retrievable units
- Downstream chunking not needed - messages are retrieval-ready

### Vassiliev: PDF Page-Level Storage

**Script**: `scripts/ingest_vassiliev_pdf.py`

**Strategy**: Store each physical PDF page as-is, with notation markers in metadata.

**Key Characteristics**:
- **Logical unit**: Original PDF page
- **Page definition**: `logical_page_label = f"pdf.{pdf_page_number}"`
- **No chunking**: Raw pages stored for downstream processing
- **Separate metadata**: `page_metadata` table stores `p.xx` markers

**Marker Extraction**:
```python
# Extract p.xx markers (internal notebook pagination)
PXX_ANYWHERE_RE = re.compile(r"(?m)^\s*(p\.\s*\d+)\s*$")
```

**Why This Approach**:
- Vassiliev notebooks are handwritten notes with complex structure
- `p.xx` markers indicate original notebook pagination (not PDF page)
- Preserves original page boundaries for citation accuracy
- Chunking deferred to downstream processing where more context is available

### HUAC Hearings: Turn-Aware Chunking

**Script**: `scripts/ingest_huac_hearings.py`

**Strategy**: Parse speaker turns and chunk by dialogue structure.

**Key Characteristics**:
- **Logical unit**: Speaker turn (one person's statement)
- **Chunks**: Groups of turns (~4000 chars, never split mid-turn)
- **Speaker tracking**: `speaker_norms`, `primary_speaker_norm`, turn spans
- **Full pipeline**: Pages → Turns → Chunks (all in one ingest)

**Why This Approach**:
- Hearing transcripts are structured dialogues
- Retrieval should preserve Q&A context
- Speaker attribution is critical (who said what)
- Similar to McCarthy hearings

### HUAC Reports: Page-Based Chunking

**Script**: `scripts/ingest_huac_reports.py`

**Strategy**: Simple character-based chunks from narrative text.

**Key Characteristics**:
- **Logical unit**: Text segment (~5000 chars)
- **No special structure**: Reports are narrative prose
- **Page tracking**: Chunks linked to source pages
- **Year-based content types**: `huac_annual_report_{year}`

**Why This Approach**:
- Reports don't have speaker turns or message boundaries
- Simple chunking sufficient for retrieval
- Year metadata enables temporal filtering

### Comparison Table

| Aspect | Venona | Vassiliev | Silvermaster | HUAC Hearings | HUAC Reports |
|--------|--------|-----------|--------------|---------------|--------------|
| **Logical unit** | Cable message | PDF page | Document unit | Speaker turn | Text segment |
| **Uses chunks table** | Via chunk_corpus | Via chunk_corpus | Via chunk_corpus | Yes (built-in) | Yes (built-in) |
| **Boundary detection** | USSR Ref/From/To/No | p.xx markers | FBI headers/FOIPA | Speaker labels | Character count |
| **Metadata storage** | `page_metadata` | `page_metadata` | `chunk_metadata` | `chunk_metadata` | `chunk_metadata` |
| **Multi-page handling** | Concatenates | N/A | Grouped by unit | Via turn spans | Via page overlap |
| **OCR quality handling** | N/A | N/A | Quality gating | N/A | N/A |
| **Pipeline** | Two-stage | Two-stage | Two-stage | One-stage | One-stage |

### When to Use Each Strategy

| Source Type | Recommended Strategy | Example |
|-------------|---------------------|---------|
| Decoded cables/messages with headers | Venona-style (message pages) | Intelligence cables |
| Handwritten notebooks | Vassiliev-style (PDF pages + markers) | Research notes, journals |
| Hearing transcripts | HUAC Hearings-style (turn-aware) | Congressional hearings |
| Narrative reports/publications | HUAC Reports-style (page-based) | Annual reports, analyses |
| FBI memos/communications | SOLO-style (memo-aware) | FBI case files |
| FBI files with variable OCR quality | Silvermaster-style (structure + quality gating) | Large FBI case files |

---

## Version History

| Date | Collection | Notes |
|------|------------|-------|
| 2026-01 | McCarthy | Turn-aware chunking v2 (5 docs, 4,316 pages, 5,637 chunks) |
| 2026-01 | Rosenberg | Document-aware chunking (181 docs, 14,999 pages, 6,293 chunks) |
| 2026-01 | SOLO | Memo-aware chunking (122 docs, 24,117 pages, 9,527 chunks) |
| 2026-01 | HUAC Hearings | Turn-aware chunking (5 transcript docs, 1,836 pages, 1,813 chunks) |
| 2026-01 | HUAC Reports | Page-based chunking (25 docs: 17 annual + 8 special, 3,089 pages, 2,546 chunks) |
| 2026-01 | FBI COMRAP | Page-based chunking (3 docs, 613 pages, 314 chunks) |
| 2026-01 | Venona | Two-stage: message-level pages + 1:1 chunking (45 docs, 3,048 pages, 3,048 chunks) |
| 2026-01 | Vassiliev | Two-stage: PDF pages + marker-aware chunking (9 docs, 1,164 pages, 4,694 chunks) |
| 2026-01 | Silvermaster | Two-stage: PDF pages + structure-aware chunking (155 docs, 24,970 pages, 11,645 chunks) |

---

## Complete Coverage Summary

All files in `data/raw/` are either ingested or explicitly excluded:

| Directory | Files | Collection | Status |
|-----------|-------|------------|--------|
| `FBICOMRAP/` | 3 | fbicomrap | ✓ Ingested |
| `index/` | 3 | — | Reference index (not primary source) |
| `mccarthy/` | 5 | mccarthy | ✓ Ingested |
| `rosenberg/` | 181 | rosenberg | ✓ Ingested |
| `silvermaster/` | 154 | silvermaster | ✓ Ingested |
| `solo/` | 122 | solo | ✓ Ingested |
| `umamerican_hearings/` | 13 | huac_hearings (5) + huac_reports (8) | ✓ Split by type |
| `unamerican_reports/` | 17 | huac_reports | ✓ Ingested |
| `vassiliev/` | 9 | vassiliev | ✓ Ingested |
| `venona/` | 45 | venona | ✓ Ingested |

**Total corpus**: 9 collections, 550 documents, 78,152 pages, 45,517 chunks
