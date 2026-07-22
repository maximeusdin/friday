# Collision Detection System Analysis

## Current Problems

### 1. Arbitrary Thresholds
- `COLLISION_HARMLESS_SINGLE_TOKEN_MAX_CANDIDATES = 5` - Why 5? Why not 3 or 10?
- `if len(alias_infos) <= 5` for person_given - Why 5? This was 3 before, changed to 5 for "fitin"
- `COLLISION_ADJUDICABLE_MAX_CANDIDATES` has different thresholds per class with no clear rationale

### 2. Inconsistent Logic
- Covernames: Special boolean check (never harmless)
- Person given names: Threshold-based (≤5 try to resolve, >5 harmless)
- Generic words: Always harmless
- No systematic approach across all alias classes

### 3. No Duplicate Detection
- Doesn't check if candidates are duplicates (same canonical name)
- "fitin" has 5 candidates, all for "Pavel Mikhailovich Fitin" - these are duplicates, not a real collision
- "yakubovich" has 2 candidates, both "Yakubovich" - duplicates, not a real collision

### 4. Citation Resolution as Fallback
- Citation-based resolution is tried, but only after marking as harmless
- Should be the PRIMARY method, not a fallback
- If citations can resolve it, why mark as harmless first?

### 5. No Consideration of Entity Value
- Doesn't distinguish between:
  - High-value entities (covernames, person_full names)
  - Low-value entities (person_given names, generic words)
  - Duplicate entities (same person, different records)

## Proposed Systematic Approach

### Principle 1: Duplicate Detection First
**Rule**: If all candidates have the same canonical name (normalized), they're duplicates.
- **Action**: Always try to resolve (use citation-based resolution, pick one)
- **Never mark as harmless**: Duplicates are a data quality issue, not ambiguity

### Principle 2: Citation-Based Resolution is Primary
**Rule**: If we have chunk_id, document_id, and page numbers, use citations FIRST.
- **Action**: Try citation-based resolution before any threshold checks
- **Only if citations fail**: Then apply other rules

### Principle 3: Entity Value Hierarchy
**Rule**: Different entity types/classes have different importance:
1. **Covernames**: Always resolve, never skip
2. **Person full names**: High value, always try to resolve
3. **Person given names**: Lower value, but still try if few candidates
4. **Generic words**: Low value, can skip if ambiguous

### Principle 4: Clear Decision Tree
```
1. Are all candidates duplicates? → Resolve (pick one, merge later)
2. Can citations resolve it? → Use citation resolution
3. Is it a covername? → Always resolve, enqueue if can't resolve
4. Is it a person_full name? → Always resolve, enqueue if can't resolve
5. Is it a person_given name?
   - ≤3 candidates: Try to resolve
   - >3 candidates: Mark as harmless (too ambiguous)
6. Is it a generic word? → Mark as harmless
7. Default: Try to resolve, enqueue if can't
```

### Principle 5: No Arbitrary Numbers
**Rule**: Thresholds should be based on:
- **Data quality**: Can we actually resolve this?
- **Entity importance**: Is this entity worth the effort?
- **Resolution methods available**: Do we have citations? Context?

## Implementation Strategy

### Step 1: Add Duplicate Detection
```python
def are_candidates_duplicates(alias_infos: List[AliasInfo], conn) -> bool:
    """Check if all candidates have the same canonical name."""
    if len(alias_infos) <= 1:
        return False
    
    # Get canonical names for all candidates
    entity_ids = [ai.entity_id for ai in alias_infos]
    canonical_names = get_canonical_names(conn, entity_ids)
    
    # Normalize and compare
    normalized_names = {normalize_name(name) for name in canonical_names.values()}
    return len(normalized_names) == 1
```

### Step 2: Reorder Resolution Logic
```python
def resolve_collision(alias_infos, chunk_id, ...):
    # 1. Check for duplicates FIRST
    if are_candidates_duplicates(alias_infos, conn):
        return pick_best_duplicate(alias_infos)  # Use citations, or pick first
    
    # 2. Try citation-based resolution
    if chunk_id and document_id and pdf_pages:
        resolved = resolve_with_citations(...)
        if resolved:
            return resolved
    
    # 3. Apply entity value rules (no arbitrary thresholds)
    if has_covername(alias_infos):
        return try_resolve_or_enqueue(...)
    
    if has_person_full(alias_infos):
        return try_resolve_or_enqueue(...)
    
    # ... etc
```

### Step 3: Remove Arbitrary Thresholds
- Replace `if len(alias_infos) <= 5` with logic based on:
  - Can we resolve it? (citations, context)
  - Is it worth resolving? (entity importance)
  - Are they duplicates? (data quality)

## Benefits

1. **Predictable**: Clear rules, no magic numbers
2. **Data-driven**: Based on actual entity relationships, not arbitrary counts
3. **Handles duplicates**: Recognizes when "collision" is actually duplicate entities
4. **Prioritizes citations**: Uses the best available resolution method first
5. **Maintainable**: Clear logic, easy to understand and modify
