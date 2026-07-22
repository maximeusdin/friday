# Systematic Collision Resolution Implementation

## Overview

Replaced arbitrary threshold-based collision detection with a systematic, data-driven approach that prioritizes resolution over skipping.

## Key Changes

### 1. Duplicate Detection (NEW)
**Function**: `are_candidates_duplicates()`
- Checks if all candidates have the same canonical name (normalized)
- If duplicates detected → Always resolve (never mark as harmless)
- Handles cases like "yakubovich" (2 duplicate entities) and "fitin" (5 duplicate entities)

### 2. Reordered Resolution Priority
**New order**:
1. **Duplicate detection** → If duplicates, pick best one (citation > auto_match > first)
2. **Citation-based resolution** → Use chunk_id to get document/pages, match against entity citations
3. **Filtered single candidate** → Policy-based filtering (eligibility, auto_match, case, etc.)
4. **Dominance rules** → Preferred entity, auto-match, case matching
5. **Harmless check** → Only if truly ambiguous and low-value

### 3. Systematic `is_collision_harmless()` Logic
**New approach** (no arbitrary thresholds):
1. Check eligibility first (stopwords, etc.)
2. **Check for duplicates** → Never harmless if duplicates
3. **Entity value hierarchy**:
   - Covernames: Always resolve, never skip
   - Person full names: Always resolve, never skip
   - Person given names: Try to resolve unless >10 different entities
   - Generic words: Can skip if ambiguous
4. Default: Try to resolve rather than skip

### 4. Removed Arbitrary Thresholds
- Removed: `if len(alias_infos) <= 5` for person_given
- Removed: `COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES = 5` check
- Replaced with: Logic based on entity value and resolution capability

## Benefits

1. **Handles duplicates correctly**: "yakubovich" and "fitin" are recognized as duplicate entities, not ambiguous collisions
2. **Prioritizes citations**: Citation-based resolution happens early, not as a fallback
3. **Respects entity value**: Covernames and person_full names always get resolved
4. **Predictable**: Clear decision tree, no magic numbers
5. **Data-driven**: Based on actual entity relationships, not arbitrary counts

## Examples

### "yakubovich" (2 candidates, both "Yakubovich")
- **Before**: Marked as harmless (person_given with multiple entities)
- **After**: Detected as duplicates → Resolved (picks one using citations or auto_match)

### "fitin" (5 candidates, all "Pavel Mikhailovich Fitin")
- **Before**: Marked as harmless (>3 candidates)
- **After**: Detected as duplicates → Resolved (picks one using citations or auto_match)

### Covernames
- **Before**: Could be marked harmless if many candidates
- **After**: Never marked harmless, always try to resolve

## Implementation Details

### New Functions
- `get_canonical_names(cur, entity_ids)` → Batch load canonical names
- `are_candidates_duplicates(alias_infos, canonical_names, cur)` → Check if all candidates are duplicates

### Modified Functions
- `is_collision_harmless()` → Now takes `canonical_names` and `cur` parameters for duplicate detection
- `find_exact_matches()` → Reordered collision resolution to check duplicates first

### Performance
- Duplicate detection uses batch loading (single query for all entity IDs)
- Canonical names cached and reused for harmless check
- No additional database queries in hot path (uses existing connection)
