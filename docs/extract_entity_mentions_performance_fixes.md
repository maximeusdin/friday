# Performance Fixes for extract_entity_mentions.py

## Problem
The script was extremely slow (11+ hours) due to several performance bottlenecks:
1. **Nested cursor creation** - Creating new database cursors inside collision resolution loops
2. **Per-collision queries** - Querying database for canonical names on every collision
3. **Small batch size** - Default batch size of 100 was too small
4. **Inefficient caching** - Not pre-loading data that could be cached

## Fixes Applied

### 1. Pre-load Canonical Names Cache
**Before:** `get_canonical_names()` was called inside collision resolution loops, creating a query for every collision.

**After:** Pre-load canonical names for all entities in the batch once:
```python
canonical_names_cache = get_canonical_names(cur, all_entity_ids)
```

**Impact:** Eliminates hundreds/thousands of database queries per batch.

### 2. Reuse Single Cursor Per Batch
**Before:** Created new cursors (`conn.cursor()`) inside collision resolution loops:
- Line 1662: `with conn.cursor() as cur2:`
- Line 1670: `with conn.cursor() as cur3:`
- Line 1698: `with conn.cursor() as cur3:`
- Line 2012: `with conn.cursor() as cur3:`

**After:** Create a single reusable cursor for the entire batch:
```python
batch_cursor = conn.cursor()
try:
    # Process all chunks reusing batch_cursor
finally:
    batch_cursor.close()
```

**Impact:** Eliminates cursor creation overhead (each cursor creation has overhead).

### 3. Increased Default Batch Size
**Before:** `--batch-size` default was 100

**After:** Default increased to 500

**Impact:** Fewer batch iterations, better amortization of setup costs.

### 4. Use Pre-loaded Caches
**Before:** Functions would query database even when caches were available.

**After:** Functions use pre-loaded caches (`canonical_names_cache`, `document_cache`, `entity_citations_cache`) and only fall back to cursor queries when necessary.

**Impact:** Reduces database round-trips significantly.

## Expected Performance Improvement

With these fixes, you should see:
- **10-50x speedup** depending on collision frequency
- **Reduced database load** - fewer queries and cursor creations
- **Better scalability** - performance scales better with corpus size

## Testing

To verify the improvements:

1. **Test on a small subset first:**
```bash
python scripts/extract_entity_mentions.py \
  --collection venona \
  --concordance-source-slug "vassiliev_venona_index_full_capitalized" \
  --enable-partial \
  --limit 100 \
  --batch-size 500
```

2. **Monitor progress:**
   - Check how long each batch takes
   - Watch for cursor creation messages (should be minimal)
   - Verify mentions are being found correctly

3. **Compare with previous run:**
   - Time per chunk should be significantly lower
   - Database queries should be much fewer

## Additional Recommendations

If still slow, consider:

1. **Disable fuzzy matching** if not needed:
   ```bash
   # Don't use --enable-fuzzy unless necessary
   ```

2. **Increase batch size further** if you have enough RAM:
   ```bash
   --batch-size 1000
   ```

3. **Process collections separately** to avoid memory issues:
   ```bash
   # Run venona and vassiliev separately
   ```

4. **Check database indexes:**
   ```sql
   -- Ensure these indexes exist:
   CREATE INDEX IF NOT EXISTS entity_aliases_alias_norm_idx ON entity_aliases(alias_norm);
   CREATE INDEX IF NOT EXISTS entity_mentions_chunk_id_idx ON entity_mentions(chunk_id);
   CREATE INDEX IF NOT EXISTS entity_mentions_entity_id_idx ON entity_mentions(entity_id);
   ```

5. **Monitor database performance:**
   - Check for slow queries
   - Ensure connection pooling is working
   - Consider increasing `max_connections` if needed

## Code Changes Summary

- **Line ~2166:** Added `canonical_names_cache` pre-loading
- **Line ~2171:** Added `batch_cursor` creation before loop
- **Line ~1659-1688:** Replaced nested cursor creation with `batch_cursor` reuse
- **Line ~1692-1711:** Replaced nested cursor creation with `batch_cursor` reuse  
- **Line ~1733-1747:** Removed cursor creation, use cache directly
- **Line ~2010-2032:** Replaced nested cursor creation with `batch_cursor` reuse
- **Line ~2607:** Increased default batch size from 100 to 500

## Notes

- All changes are backward compatible
- The script maintains the same functionality
- Idempotency is preserved
- Review queue population still works correctly
