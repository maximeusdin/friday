# Chunk Processing Performance Analysis

## Problem Statement

Chunk processing in `extract_entity_mentions.py` is extremely slow, taking up to **1 minute per chunk**. This is the primary bottleneck after fixing alias loading.

## Root Cause Analysis

### 1. **Database Queries Inside Matching Loop** (CRITICAL - Lines 1434-1450, 1805-1827)

**Issue**: For every collision (multiple entities matching the same text), the code makes multiple database queries INSIDE the matching loop:

```python
if conn and document_id is not None:
    with conn.cursor() as cur:
        page_ids = get_chunk_page_ids(cur, chunk_id)  # Query 1
        citation_resolved, citation_confidence, citation_method = \
            resolve_collision_with_citations(
                cur, alias_infos, document_id, page_ids, chunk_id
            )
```

**Inside `resolve_collision_with_citations`**:
- `get_chunk_page_ids()`: Query to `chunk_pages` table
- `get_pdf_page_numbers()`: Query to `pages` table  
- For each candidate entity: `get_entity_citations()`: Query to `entity_citations` table
- For each citation: `find_documents_for_citation()`: Query to `documents` table with LIKE patterns

**Impact**:
- If a chunk has 10 collisions, that's **40+ database queries per chunk**
- Each query has network latency (even localhost: ~1-5ms)
- Each query has query planning overhead
- **Total overhead: 40-200ms per collision × 10 collisions = 400ms-2s per chunk**

**Location**: 
- Line 1434-1450: Exact match collision resolution
- Line 1805-1827: Partial/fuzzy match collision resolution

### 2. **Fuzzy Matching Levenshtein Distance** (CRITICAL - Lines 1228-1288)

**Issue**: For each token in the chunk, calculates Levenshtein distance against many aliases:

```python
for token_start, token_end, token_text in original_tokens:
    token_norm = normalize_alias(token_text)
    # For each alias with similar length...
    for alias_norm in aliases_by_length[alias_len]:
        distance = levenshtein_distance(token_norm, alias_norm)  # O(n*m) operation
```

**Complexity**:
- Levenshtein distance is O(n×m) where n=token length, m=alias length
- For a chunk with 100 tokens and 17,021 aliases:
  - Even with length filtering (checking ~100 aliases per token)
  - That's 100 tokens × 100 aliases × O(10×10) = **1,000,000 distance calculations**
- Each calculation involves dynamic programming with nested loops

**Impact**:
- **10-30 seconds per chunk** for fuzzy matching alone
- CPU-bound operation (no I/O, but very expensive)

**Location**: Lines 1228-1288 (`find_fuzzy_matches`)

### 3. **Partial Matching Index Rebuilt Per Chunk** (MODERATE - Line 1755)

**Issue**: Builds partial match index for every chunk:

```python
if enable_partial:
    partial_index = build_partial_match_index(aliases_by_norm)  # Rebuilt every chunk!
    partial_candidates = find_partial_matches(...)
```

**Impact**:
- Index building is O(n) where n=number of aliases (17,021)
- Rebuilding for each chunk: **50-200ms per chunk × thousands of chunks**
- Should be built once and reused

**Location**: Line 1755

### 4. **N-gram Matching Loop** (MODERATE - Lines 1359-1749)

**Issue**: For each token position, tries n-grams from length 5 down to 1:

```python
while i < len(original_tokens):
    for n in range(min(5, len(original_tokens) - i), 0, -1):
        # Build n-gram, normalize, check alias set
        # If collision, resolve with citations (DB queries!)
```

**Complexity**:
- For a chunk with 100 tokens: tries ~500 n-gram combinations
- Each n-gram: normalization, hash lookup, policy checks
- If collision: triggers citation resolution (DB queries)

**Impact**:
- **100-500ms per chunk** for n-gram matching
- Amplified by collision resolution overhead

**Location**: Lines 1359-1749

### 5. **Multiple Normalization Operations** (MINOR)

**Issue**: Normalizes tokens multiple times:
- Once for exact matching
- Again for partial matching  
- Again for fuzzy matching

**Impact**: 
- **10-50ms per chunk** (minor but adds up)

## Performance Breakdown (Estimated)

For a typical chunk (~500 words, ~100 tokens):

| Operation | Time | Percentage |
|-----------|------|------------|
| **Fuzzy matching** (Levenshtein) | 10-30s | 50-70% |
| **Citation resolution queries** (10 collisions) | 2-5s | 20-30% |
| **N-gram exact matching** | 0.5-1s | 5-10% |
| **Partial matching** | 0.2-0.5s | 2-5% |
| **Index building** | 0.1-0.2s | 1-2% |
| **Other overhead** | 0.1-0.3s | 1-2% |
| **TOTAL** | **13-37s** | **100%** |

## Optimization Strategy

### Priority 1: Eliminate Database Queries from Loop

#### 1.1 Pre-fetch Chunk Metadata
**Solution**: Load all chunk metadata (page_ids, pdf_page_numbers) in batch before processing:

```python
# Before processing chunks
chunk_metadata = {}
with conn.cursor() as cur:
    cur.execute("""
        SELECT cp.chunk_id, array_agg(p.id ORDER BY cp.span_order) as page_ids,
               array_agg(p.pdf_page_number ORDER BY cp.span_order) FILTER (WHERE p.pdf_page_number IS NOT NULL) as pdf_pages
        FROM chunk_pages cp
        JOIN pages p ON p.id = cp.page_id
        WHERE cp.chunk_id = ANY(%s)
        GROUP BY cp.chunk_id
    """, (chunk_ids,))
    for row in cur.fetchall():
        chunk_metadata[row[0]] = {'page_ids': row[1], 'pdf_pages': row[2]}
```

**Expected Impact**: Eliminate 2 queries per collision → **80-90% reduction in query overhead**

#### 1.2 Pre-fetch Entity Citations
**Solution**: Load all entity citations in batch:

```python
# Before processing chunks
entity_citations_cache = {}
with conn.cursor() as cur:
    cur.execute("""
        SELECT entity_id, citation_text, page_list
        FROM entity_citations
        WHERE entity_id = ANY(%s)
    """, (all_entity_ids,))
    for row in cur.fetchall():
        entity_citations_cache.setdefault(row[0], []).append({
            'citation_text': row[1], 'page_list': row[2]
        })
```

**Expected Impact**: Eliminate N queries per collision (N = number of candidates) → **90-95% reduction in citation query overhead**

#### 1.3 Cache Document Lookups
**Solution**: Cache document name → document_id mappings:

```python
# Before processing chunks
document_cache = {}
with conn.cursor() as cur:
    cur.execute("SELECT id, LOWER(source_name) as name FROM documents")
    for row in cur.fetchall():
        document_cache[row[1]] = row[0]
```

**Expected Impact**: Eliminate LIKE queries → **100% reduction in document lookup overhead**

**Total Impact**: **Reduce citation resolution from 2-5s to 50-200ms per chunk**

### Priority 2: Optimize Fuzzy Matching

#### 2.1 Use Faster Edit Distance Library
**Solution**: Use `python-Levenshtein` (C implementation):

```python
from Levenshtein import distance as levenshtein_distance
```

**Expected Impact**: **5-10x faster** → Reduce fuzzy matching from 10-30s to 1-3s per chunk

#### 2.2 Limit Fuzzy Matching Scope
**Solution**: Only fuzzy match tokens that:
- Are >= 4 characters
- Don't match exact or partial
- Are likely entity names (capitalized, not stopwords)

**Expected Impact**: **Reduce candidates by 50-70%** → Reduce fuzzy matching time by 50-70%

#### 2.3 Use Trigram Similarity Instead
**Solution**: Use PostgreSQL trigram similarity (already indexed):

```python
# Pre-compute trigram similarity scores in batch query
cur.execute("""
    SELECT alias_norm, similarity(alias_norm, %s) as sim
    FROM entity_aliases
    WHERE alias_norm % %s  -- Trigram operator (uses index)
      AND similarity(alias_norm, %s) >= 0.6
    ORDER BY sim DESC
    LIMIT 20
""", (token_norm, token_norm, token_norm))
```

**Expected Impact**: **10-50x faster** → Reduce fuzzy matching from 10-30s to 0.2-1s per chunk

**Total Impact**: **Reduce fuzzy matching from 10-30s to 0.2-3s per chunk**

### Priority 3: Cache Partial Match Index

#### 3.1 Build Index Once
**Solution**: Build partial index once before processing chunks:

```python
# Before processing chunks
partial_index = build_partial_match_index(aliases_by_norm) if enable_partial else None

# In find_exact_matches, reuse the index
if enable_partial:
    partial_candidates = find_partial_matches(..., partial_index=partial_index)
```

**Expected Impact**: **Eliminate 50-200ms per chunk** → Save 50-200ms × thousands of chunks

### Priority 4: Optimize N-gram Matching

#### 4.1 Early Exit on Match
**Solution**: Already implemented (breaks after match), but can optimize further:
- Skip n-grams that start with stopwords
- Skip n-grams shorter than min_chars

**Expected Impact**: **10-20% faster n-gram matching**

## Expected Performance Improvements

### Current State
- **Per chunk**: 13-37 seconds (average ~20 seconds)
- **For 1000 chunks**: ~5.5 hours

### After Priority 1 (Eliminate DB Queries)
- **Per chunk**: 11-32 seconds (saves 2-5s)
- **For 1000 chunks**: ~4.5 hours
- **Improvement: 15-20% faster**

### After Priority 2 (Optimize Fuzzy Matching)
- **Per chunk**: 2-5 seconds (saves 8-25s)
- **For 1000 chunks**: ~1 hour
- **Improvement: 80-85% faster**

### After Priority 3 (Cache Index)
- **Per chunk**: 2-5 seconds (saves 50-200ms)
- **For 1000 chunks**: ~1 hour
- **Improvement: Additional 1-2%**

### After Priority 4 (Optimize N-grams)
- **Per chunk**: 1.8-4.5 seconds (saves 0.2-0.5s)
- **For 1000 chunks**: ~50 minutes
- **Improvement: Additional 5-10%**

### Combined Impact
- **Per chunk**: **1.8-4.5 seconds** (down from 13-37 seconds)
- **For 1000 chunks**: **~50 minutes** (down from ~5.5 hours)
- **Total improvement: 85-90% faster**

## Implementation Priority

1. **Phase 1** (Immediate): Pre-fetch chunk metadata and entity citations
   - **Impact**: 15-20% faster
   - **Effort**: Medium (requires refactoring)

2. **Phase 2** (High Impact): Optimize fuzzy matching (use trigram similarity)
   - **Impact**: 80-85% faster
   - **Effort**: Medium (requires testing)

3. **Phase 3** (Quick Win): Cache partial match index
   - **Impact**: 1-2% faster
   - **Effort**: Low (simple refactor)

4. **Phase 4** (Polish): Optimize n-gram matching
   - **Impact**: 5-10% faster
   - **Effort**: Low (simple optimizations)

## Code Locations

### Critical Performance Issues
- **Lines 1434-1450**: Citation resolution in exact match collisions
- **Lines 1805-1827**: Citation resolution in partial/fuzzy collisions
- **Lines 1228-1288**: Fuzzy matching with Levenshtein distance
- **Line 1755**: Partial index rebuilding per chunk

### Helper Functions to Optimize
- `resolve_collision_with_citations()` (lines 1058-1144)
- `get_chunk_page_ids()` (lines 973-984)
- `get_pdf_page_numbers()` (lines 987-1001)
- `get_entity_citations()` (lines 1004-1017)
- `find_documents_for_citation()` (lines 1035-1055)
- `find_fuzzy_matches()` (lines 1228-1288)
- `build_partial_match_index()` (lines 1160-1173)

## Testing Strategy

1. **Profile current performance**: Time each operation
2. **Implement Phase 1**: Measure improvement
3. **Implement Phase 2**: Measure improvement
4. **Verify correctness**: Ensure matches are still correct
5. **Load test**: Process 100+ chunks and measure total time

## Notes

- Fuzzy matching is the biggest bottleneck (50-70% of time)
- Database queries are the second biggest bottleneck (20-30% of time)
- Both can be optimized significantly with batch operations and caching
- Consider making fuzzy matching optional or configurable per collection
