# Strategic Overview: From Concordance Index to Entity Mentions

## Executive Summary

The concordance index ingest creates a **curated knowledge base** that drives precise, context-aware entity mention extraction. Instead of scanning all chunks blindly, we use expert-curated citations to target specific document spans where entities are known to appear, dramatically improving precision and enabling disambiguation.

## The Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CONCORDANCE INDEX INGEST                                     │
│    (ingest_concordance_tab_aware.py)                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Parses PDF concordance index           │
        │ Extracts:                              │
        │  • Entity names & aliases              │
        │  • Citations (documents + pages)      │
        │  • Relationships (cover_name_of, etc)│
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Stores in database:                   │
        │  • entities                           │
        │  • entity_aliases                     │
        │  • entity_citations                   │
        │  • entity_links                       │
        └───────────────────────────────────────┘
                            │
                            │ (parallel process)
                            │
        ┌───────────────────────────────────────┐
        │ 2. DOCUMENT INGEST                    │
        │    (ingest_venona_pdf.py, etc.)       │
        │                                       │
        │ Ingests actual source documents:      │
        │  • Venona cables                      │
        │  • Vassiliev notebooks                 │
        │                                       │
        │ Creates:                              │
        │  • documents                          │
        │  • pages                              │
        │  • chunks                             │
        │  • chunk_metadata                     │
        │  • chunk_pages                        │
        └───────────────────────────────────────┘
                            │
                            │ (both processes complete)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ENTITY MENTION EXTRACTION                                    │
│    (extract_entity_mentions_from_citations.py)                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ For each entity with citations:        │
        │                                       │
        │ 1. Read citations from                │
        │    entity_citations table             │
        │                                       │
        │ 2. Parse citation_text to extract:    │
        │    • Document names                   │
        │    • Page numbers (with ranges)       │
        │                                       │
        │ 3. Map citations → documents:          │
        │    • Normalize document names         │
        │    • Match to documents table         │
        │                                       │
        │ 4. Map citations → pages:              │
        │    • Find pages matching page numbers │
        │    • Handle page ranges               │
        │                                       │
        │ 5. Map pages → chunks:                 │
        │    • Find chunks spanning those pages │
        │    • Verify document_id matches       │
        │                                       │
        │ 6. Extract mentions from chunks:      │
        │    • Use entity aliases               │
        │    • Apply policy checks              │
        │    • Resolve collisions               │
        │                                       │
        │ 7. Store in entity_mentions:           │
        │    • With full provenance              │
        │    • Linked to citations              │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ entity_mentions table populated        │
        │ with high-precision, context-aware     │
        │ entity mentions                        │
        └───────────────────────────────────────┘
```

## Strategic Advantages

### 1. **Precision Over Recall**

**Traditional approach**: Scan all chunks, match all aliases → High recall, but many false positives and ambiguities.

**Citation-based approach**: Only extract from chunks where entities are **known** to appear → Lower recall, but much higher precision.

**Why this matters**: 
- Reduces noise in downstream analysis
- Enables confident entity resolution
- Focuses computational resources on verified locations

### 2. **Context-Aware Disambiguation**

The concordance index provides **document + page context** for each entity mention:

```
Example: "Albert" appears in:
- Vassiliev Black Notebook, page 79
- Venona New York KGB 1943, pages 26, 28

When extracting:
- "Albert" in Black Notebook page 79 → Iskhak Akhmerov
- "Albert" in Venona 1943 page 26 → Iskhak Akhmerov
- "Albert" in other documents → May be different entity (or not extracted)
```

**Strategic value**:
- Resolves ambiguous aliases using document context
- Prevents false matches across documents
- Maintains entity identity across different source collections

### 3. **Expert-Curated Knowledge Base**

The concordance index represents **expert knowledge** about where entities appear:

- **Cover names** are linked to real identities
- **Page numbers** are precisely cited
- **Document names** are standardized
- **Relationships** are explicitly recorded

**Strategic value**:
- Leverages domain expertise encoded in the index
- Provides ground truth for validation
- Enables quality control (we can verify mentions match citations)

### 4. **Provenance and Traceability**

Each entity mention is linked back to:
- **Source citation**: Which citation led to this extraction
- **Document**: Which document contains the mention
- **Pages**: Which pages the chunk spans
- **Chunk**: The exact text span

**Strategic value**:
- Full audit trail for each mention
- Can verify extractions against source citations
- Enables debugging and quality improvement
- Supports citation-based retrieval

### 5. **Efficiency**

Instead of:
```
Scan ALL chunks (potentially millions)
  → Match ALL aliases
    → Resolve ALL collisions
      → Filter by policy
```

We do:
```
For each entity:
  → Get citations (expert-curated locations)
    → Find matching chunks (targeted search)
      → Extract mentions (only from verified locations)
```

**Strategic value**:
- Faster processing (only process relevant chunks)
- Lower computational cost
- Scales better as document corpus grows

## Key Design Decisions

### 1. **Two-Phase Process**

**Phase 1: Ingest Concordance Index**
- Parse PDF → Extract entities, aliases, citations
- Store in normalized database tables
- Creates the "knowledge base" of where entities appear

**Phase 2: Extract Mentions**
- Use citations to target specific chunks
- Extract mentions using alias matching
- Store with full provenance

**Why separate**: Allows re-extraction without re-ingesting the concordance index. The concordance index is relatively stable; mentions can be re-extracted as extraction logic improves.

### 2. **Citation Parsing and Normalization**

Citations come in various formats:
- "Vassiliev Black Notebook, 79"
- "Venona New York KGB 1943, 26, 28"
- "Venona New York KGB 1941–42, 16, 74–75"

We normalize:
- Document names (remove prefixes, handle underscores, etc.)
- Page numbers (expand ranges like "74–75" → [74, 75])
- Year ranges (handle "1941–42" vs "1941-1942")

**Why this matters**: Robust matching between citation sources and database documents enables reliable targeting.

### 3. **Document Verification**

When finding chunks for a citation, we verify:
- Chunk's `document_id` matches the citation's document
- Chunk spans the cited pages
- Chunk belongs to the correct collection

**Why this matters**: Prevents false positives from chunks in different documents that happen to have matching page numbers.

### 4. **Alias Policy System**

The extraction uses policy-driven matching:
- `is_auto_match`: Whether alias can be automatically matched
- `min_chars`: Minimum character length
- `match_case`: Case sensitivity requirements
- `alias_class`: Type of alias (covername, person_full, etc.)
- `requires_context`: Whether context is needed

**Why this matters**: Allows fine-grained control over which aliases are matched and how, preventing false positives while maintaining recall.

## Data Relationships

```
concordance_entries
    │
    ├─→ entity_aliases (aliases extracted from entries)
    │       │
    │       └─→ entities (canonical entity)
    │
    └─→ entity_citations (citations extracted from entries)
            │
            └─→ entities (entity being cited)
                    │
                    └─→ entity_mentions (mentions extracted using citations)
                            │
                            ├─→ chunks (chunk containing mention)
                            │       │
                            │       └─→ chunk_metadata (document_id, pages)
                            │
                            └─→ documents (document containing mention)
```

## Example: "Albert" Extraction

### Step 1: Concordance Ingest
```
Entry: "Albert" ["Al'bert"] (cover name in Vassiliev's notebooks): Iskhak Akhmerov.
Citations: 
  - Vassiliev Black Notebook, 79
  - Vassiliev White Notebook #1, 55, 63–74, 153
  ...

Stored in:
  - entities: {id: 123, canonical_name: "Iskhak Akhmerov", ...}
  - entity_aliases: {entity_id: 123, alias: "Albert", alias_norm: "albert", ...}
  - entity_citations: {entity_id: 123, citation_text: "Vassiliev Black Notebook, 79", ...}
```

### Step 2: Document Ingest (parallel)
```
Ingested documents:
  - documents: {id: 42, source_name: "Vassiliev_Black_Notebook.pdf", ...}
  - pages: {document_id: 42, pdf_page_number: 79, ...}
  - chunks: {id: 6083, text: "...", ...}
  - chunk_metadata: {chunk_id: 6083, document_id: 42, first_page_id: 9754, ...}
  - chunk_pages: {chunk_id: 6083, page_id: 9754, ...}
```

### Step 3: Mention Extraction
```
For entity 123 ("Iskhak Akhmerov"):
  1. Get citations → "Vassiliev Black Notebook, 79"
  2. Parse citation → document: "Vassiliev Black Notebook", pages: [79]
  3. Map to document → document_id: 42
  4. Find pages → page_ids: [9754] (where pdf_page_number = 79)
  5. Find chunks → chunk_ids: [6083] (chunks spanning page 9754)
  6. Extract mentions from chunk 6083:
     - Search for "albert" (normalized alias)
     - Find match in chunk text
     - Extract surface: "Albert"
     - Store in entity_mentions:
       {
         entity_id: 123,
         chunk_id: 6083,
         document_id: 42,
         surface: "Albert",
         surface_norm: "albert",
         ...
       }
```

## Benefits Summary

1. **High Precision**: Only extract from verified locations
2. **Context-Aware**: Document + page context enables disambiguation
3. **Expert-Curated**: Leverages domain knowledge from concordance index
4. **Traceable**: Full provenance from citation to mention
5. **Efficient**: Targeted extraction vs. full scan
6. **Validatable**: Can verify mentions match citations
7. **Flexible**: Can re-extract as logic improves without re-ingesting index

## Future Enhancements

1. **Incremental Updates**: Re-extract only for entities with new citations
2. **Confidence Scoring**: Higher confidence for mentions matching citations exactly
3. **Cross-Validation**: Compare citation-based mentions with full-scan mentions
4. **Citation Expansion**: Use citation context to find additional mentions nearby
5. **Relationship Extraction**: Use entity_links to find related mentions
