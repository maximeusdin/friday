# Performance Analysis: extract_entity_mentions.py

## Problem Statement

The `load_all_aliases()` function in `scripts/extract_entity_mentions.py` is unacceptably slow when loading 17,021 entity aliases. The script hangs at "Loading entity aliases..." for an extended period.

## Root Cause Analysis

### 1. **Information Schema Query (Lines 693-700)**
**Issue**: Queries `information_schema.columns` on every run to check which policy columns exist.

**Impact**: 
- System catalog queries are slower than regular table queries
- This check happens every single time the script runs
- No caching mechanism

**Code Location**:
```python
cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'entity_aliases' 
    AND column_name IN ('is_auto_match', 'min_chars', 'match_case', 'match_mode', 
                        'is_numeric_entity', 'alias_class', 'allow_ambiguous_person_token', 'requires_context')
""")
```

### 2. **"Unidentified" Query with Multiple LIKE Conditions (Lines 748-761)**
**Issue**: Query uses multiple `LOWER(ea.alias) LIKE '%pattern%'` conditions with leading wildcards.

**Impact**:
- `LIKE '%pattern%'` (leading wildcard) cannot use indexes
- `LOWER()` function prevents index usage on `alias` column
- Multiple OR conditions force sequential scan
- No index exists on `alias` column (only `alias_norm` is indexed)
- This query scans ALL aliases, not just matchable ones

**Code Location**:
```python
cur.execute("""
    SELECT DISTINCT ea.entity_id
    FROM entity_aliases ea
    WHERE LOWER(ea.alias) LIKE '%unidentified%'
       OR LOWER(ea.alias) LIKE '%unknown%'
       OR LOWER(ea.alias) LIKE '%unnamed%'
       ...
""")
```

### 3. **Missing Index on `is_matchable` Column**
**Issue**: The main query filters by `WHERE ea.is_matchable = true` but there's no index on this column.

**Impact**:
- PostgreSQL must scan all rows to filter by `is_matchable`
- With 17,021+ aliases, this is a full table scan

**Code Location**:
```python
query = f"""
    SELECT {', '.join(select_cols)}
    FROM entity_aliases ea
    JOIN entities e ON e.id = ea.entity_id
    WHERE ea.is_matchable = true
    ORDER BY ea.entity_id, ea.id
"""
```

### 4. **Expensive Post-Load Diagnostics (Lines 2192-2229)**
**Issue**: After loading aliases, the script performs expensive diagnostic operations:
- Sorts all aliases by candidate count
- Iterates through top 50
- Performs multiple string operations and checks
- Builds entity type distribution statistics

**Impact**:
- These operations happen synchronously before processing chunks
- Adds significant overhead even if not needed for production runs

### 5. **Inefficient Python Processing Loop (Lines 778-877)**
**Issue**: For each of 17,021+ rows:
- Creates AliasInfo dataclass objects
- Performs string splits (`tokens = alias_norm.split()`)
- Multiple conditional checks and string operations
- Dictionary lookups and updates

**Impact**:
- Python overhead for large datasets
- Could benefit from batch processing or more efficient data structures

## Performance Improvement Strategy

### Priority 1: Database Indexes (Immediate Impact)

#### 1.1 Add Index on `is_matchable`
```sql
CREATE INDEX IF NOT EXISTS idx_entity_aliases_is_matchable 
ON entity_aliases(is_matchable) 
WHERE is_matchable = true;  -- Partial index (smaller, faster)
```

**Expected Impact**: 10-50x faster filtering on matchable aliases

#### 1.2 Add Composite Index for Main Query
```sql
CREATE INDEX IF NOT EXISTS idx_entity_aliases_matchable_entity_id 
ON entity_aliases(is_matchable, entity_id, id) 
WHERE is_matchable = true;
```

**Expected Impact**: Faster ORDER BY and JOIN operations

#### 1.3 Add Index on `alias` Column (Lowercase)
```sql
-- For the "unidentified" query, create a functional index
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_lower_trgm 
ON entity_aliases USING GIN (LOWER(alias) gin_trgm_ops);
```

**Expected Impact**: Faster pattern matching for "unidentified" detection

**Alternative**: Store a pre-computed `alias_lower` column and index it:
```sql
ALTER TABLE entity_aliases ADD COLUMN IF NOT EXISTS alias_lower TEXT;
UPDATE entity_aliases SET alias_lower = LOWER(alias) WHERE alias_lower IS NULL;
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_lower_trgm 
ON entity_aliases USING GIN (alias_lower gin_trgm_ops);
```

### Priority 2: Query Optimization

#### 2.1 Cache Schema Information
**Solution**: Check schema once and cache results, or use a simpler approach:
- Option A: Use try/except to detect missing columns
- Option B: Assume all columns exist (safer for production)
- Option C: Store schema version in a metadata table

**Code Change**:
```python
# Instead of querying information_schema, try selecting columns directly
# PostgreSQL will raise an error if column doesn't exist
try:
    cur.execute("SELECT is_auto_match FROM entity_aliases LIMIT 1")
    has_is_auto_match = True
except psycopg2.errors.UndefinedColumn:
    has_is_auto_match = False
```

**Expected Impact**: Eliminate 50-200ms schema query overhead

#### 2.2 Optimize "Unidentified" Query
**Solution A**: Use `alias_norm` instead of `alias` (already indexed):
```python
# Pre-normalize the search terms
search_terms = ['unidentified', 'unknown', 'unnamed', 'intelligence source', 
                'intelligence officer', 'source/agent', 'officer/agent',
                'soviet intelligence', 'intelligence agent']
search_norms = [normalize_alias(term) for term in search_terms]

cur.execute("""
    SELECT DISTINCT ea.entity_id
    FROM entity_aliases ea
    WHERE ea.alias_norm = ANY(%s)
""", (search_norms,))
```

**Solution B**: Use full-text search or trigram matching:
```python
cur.execute("""
    SELECT DISTINCT ea.entity_id
    FROM entity_aliases ea
    WHERE ea.alias_norm % ANY(%s)  -- Trigram operator
       OR similarity(ea.alias_norm, %s) > 0.3
""", (search_norms, search_norms[0]))
```

**Expected Impact**: 5-20x faster "unidentified" detection

#### 2.3 Combine Queries
**Solution**: Combine the "unidentified" check with the main query using a LEFT JOIN or subquery:
```python
query = f"""
    SELECT {', '.join(select_cols)},
           CASE WHEN unidentified_entities.entity_id IS NOT NULL THEN true ELSE false END AS has_unidentified_alias
    FROM entity_aliases ea
    JOIN entities e ON e.id = ea.entity_id
    LEFT JOIN (
        SELECT DISTINCT entity_id 
        FROM entity_aliases 
        WHERE alias_norm IN ({','.join(['%s'] * len(search_norms))})
    ) unidentified_entities ON unidentified_entities.entity_id = ea.entity_id
    WHERE ea.is_matchable = true
    ORDER BY ea.entity_id, ea.id
"""
```

**Expected Impact**: Single query instead of two, reducing round-trips

### Priority 3: Code Optimization

#### 3.1 Make Diagnostics Optional
**Solution**: Add `--skip-diagnostics` flag to skip expensive diagnostic operations:
```python
if not args.skip_diagnostics:
    # Diagnostic code here
```

**Expected Impact**: Save 1-5 seconds on production runs

#### 3.2 Optimize Python Processing
**Solution**: 
- Use list comprehensions where possible
- Pre-compute token splits
- Use more efficient data structures

**Code Change**:
```python
# Pre-compute tokens for all aliases in batch
alias_tokens_map = {alias_norm: alias_norm.split() for alias_norm in alias_norm_set}

# Use in loop
tokens = alias_tokens_map.get(alias_norm, [])
```

**Expected Impact**: 10-30% faster Python processing

#### 3.3 Use Bulk Operations
**Solution**: If possible, use PostgreSQL array operations or bulk inserts for processing:
```python
# Instead of processing row by row, fetch all at once
rows = cur.fetchall()
# Process in memory (already doing this, but optimize the loop)
```

### Priority 4: Caching Strategy

#### 4.1 Cache Alias Data
**Solution**: Cache loaded aliases to a pickle file or database table:
```python
cache_file = Path(f".cache/aliases_{collection_slug}_{schema_version}.pkl")
if cache_file.exists() and not args.no_cache:
    aliases_by_norm, alias_norm_set = pickle.load(cache_file.open('rb'))
else:
    aliases_by_norm, alias_norm_set = load_all_aliases(conn, collection_slug=args.collection)
    pickle.dump((aliases_by_norm, alias_norm_set), cache_file.open('wb'))
```

**Expected Impact**: Near-instant loading on subsequent runs (if aliases haven't changed)

#### 4.2 Invalidate Cache on Schema Changes
**Solution**: Track schema version or last migration timestamp:
```python
cur.execute("""
    SELECT MAX(version) FROM schema_migrations 
    WHERE table_name = 'entity_aliases'
""")
schema_version = cur.fetchone()[0]
```

## Implementation Plan

### Phase 1: Quick Wins (Immediate)
1. ✅ Add index on `is_matchable` (partial index)
2. ✅ Add composite index for main query
3. ✅ Optimize "unidentified" query to use `alias_norm`
4. ✅ Make diagnostics optional

**Expected Time Savings**: 50-80% reduction in load time

### Phase 2: Query Optimization (Short-term)
1. Cache schema information or use try/except
2. Combine queries where possible
3. Optimize Python processing loop

**Expected Time Savings**: Additional 10-20% improvement

### Phase 3: Advanced Optimization (Long-term)
1. Implement caching mechanism
2. Add `alias_lower` column for better pattern matching
3. Consider materialized views for frequently accessed data

**Expected Time Savings**: Near-instant loading on cached runs

## Expected Performance Improvements

### Current State
- Loading 17,021 aliases: **~30-60 seconds** (estimated based on user report)

### After Phase 1
- Loading 17,021 aliases: **~5-15 seconds**
- **Improvement: 70-80% faster**

### After Phase 2
- Loading 17,021 aliases: **~3-10 seconds**
- **Improvement: 80-90% faster**

### After Phase 3 (with cache)
- First run: **~3-10 seconds**
- Subsequent runs: **<1 second**
- **Improvement: 95-98% faster on cached runs**

## Monitoring

Add timing information to track improvements:
```python
import time
start_time = time.time()
aliases_by_norm, alias_norm_set = load_all_aliases(conn, collection_slug=args.collection)
load_time = time.time() - start_time
print(f"  Loaded {len(aliases_by_norm)} unique normalized aliases in {load_time:.2f}s", file=sys.stderr)
```

## Testing

1. Run `EXPLAIN ANALYZE` on queries before and after index creation
2. Measure actual load times with timing code
3. Verify query plans use indexes
4. Test with different collection sizes

## Notes

- The `is_matchable` column may not exist yet - check migrations
- Consider adding a `last_updated` timestamp to entity_aliases for cache invalidation
- The "unidentified" query might be unnecessary if we can mark these entities differently in the database
