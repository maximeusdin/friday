# P3 Enhancement: Advanced Query Features

## Overview

This issue tracks the remaining **P3 (Low Priority)** features from the original query system specification. These are advanced features that enhance usability but are not required for core research workflows.

All P0-P2 features are now complete and in production. These P3 items are candidates for future development cycles.

---

## Feature 1: Geographic Proximity Queries

### Current State
- `FILTER_COUNTRY` primitive exists and works for country-level filtering
- No support for city/region or coordinate-based queries

### Requirements
- [ ] Add `FILTER_CITY` primitive for city-level filtering
- [ ] Add `FILTER_REGION` primitive for state/province filtering
- [ ] Add `FILTER_COORDINATES` primitive with radius parameter
- [ ] Support queries like "documents near Washington DC" or "within 50km of coordinates"
- [ ] Integrate with geographic metadata in `chunk_metadata` or `page_metadata`

### Implementation Notes
- May require PostGIS extension for spatial queries
- Needs geographic metadata population (lat/long for locations mentioned)
- Consider using geocoding service for location name → coordinates
- Performance: spatial indexes needed for coordinate queries

### Example Queries
```
"Find documents mentioning events near Los Angeles"
"Documents from the New York field office"
"Reports within 100 miles of Oak Ridge, Tennessee"
```

### Effort Estimate
Medium-High (requires schema changes + PostGIS integration)

---

## Feature 2: Query Expansion & Suggestions

### Current State
- Concordance expansion exists for known entity aliases
- No automatic query suggestions or relevance feedback

### Requirements
- [ ] Suggest related terms based on corpus co-occurrence
- [ ] Suggest query expansions for low-result queries
- [ ] Track query performance metrics (result counts, user interactions)
- [ ] Implement basic relevance feedback loop
- [ ] Provide "did you mean?" suggestions for potential typos

### Implementation Notes
- Use concordance index and entity co-occurrence for term suggestions
- Track `(query, result_count, user_clicked_results)` for learning
- Consider TF-IDF or embedding similarity for related term discovery
- Rate limit suggestions to avoid overwhelming users

### Example Interactions
```
User: "Find mentions of atomic bomb"
System suggests: "Related terms: Manhattan Project, nuclear weapons, Los Alamos"

User: "Find Rosneberg documents"  
System suggests: "Did you mean: Rosenberg?"
```

### Effort Estimate
Medium (requires usage tracking infrastructure + ML/statistics)

---

## Feature 3: NOT/NEGATION Primitives

### Current State
- `OR_GROUP` primitive supports OR combinations
- `SET_TERM_MODE` supports AND/OR for terms
- No explicit negation/exclusion for terms or phrases

### Requirements
- [ ] Add `EXCLUDE_TERM` primitive: exclude chunks containing a term
- [ ] Add `EXCLUDE_PHRASE` primitive: exclude chunks containing a phrase
- [ ] Add `NOT_ENTITY` primitive: exclude chunks mentioning an entity
- [ ] Support nested negation in boolean expressions
- [ ] Validate that negation doesn't create empty result sets

### Implementation Notes
- Compilation: `NOT EXISTS (SELECT 1 FROM ... WHERE text @@ to_tsquery(...))`
- Consider performance implications of NOT queries (can be expensive)
- May need to limit negation depth to prevent pathological queries

### Example Queries
```
"Find FBI documents about espionage NOT mentioning Rosenberg"
"Documents about Soviet intelligence excluding VENONA collection"
"Mentions of Treasury Department but not Harry Dexter White"
```

### Primitive Design
```python
@dataclass
class ExcludeTermPrimitive:
    type: Literal["EXCLUDE_TERM"] = "EXCLUDE_TERM"
    value: str = ""

@dataclass  
class ExcludePhrasePrimitive:
    type: Literal["EXCLUDE_PHRASE"] = "EXCLUDE_PHRASE"
    value: str = ""

@dataclass
class NotEntityPrimitive:
    type: Literal["NOT_ENTITY"] = "NOT_ENTITY"
    entity_id: int = 0
```

### Effort Estimate
Small-Medium (straightforward primitive addition + compilation)

---

## Feature 4: Result Set Comparison Tool

### Current State
- Result sets can be created and queried independently
- No direct comparison capability between result sets
- `WITHIN_RESULT_SET` and `EXCLUDE_RESULT_SET` exist but require manual orchestration

### Requirements
- [ ] Create `compare_result_sets.py` CLI tool
- [ ] Generate comparison reports: overlap, unique to A, unique to B
- [ ] Calculate Jaccard similarity and other set metrics
- [ ] Support "what's new in result set B compared to A?"
- [ ] Optionally store comparisons in database for later retrieval

### Implementation Notes
- Core logic: set intersection, difference, symmetric difference
- Output formats: JSON, markdown table, CSV
- Consider storing comparisons in `result_set_comparisons` table
- Visualizations could be added later (Venn diagrams, etc.)

### Example Usage
```bash
# Compare two result sets
python scripts/compare_result_sets.py --set-a 25 --set-b 30

# Output:
# Result Set Comparison
# =====================
# Set A (ID=25): 45 chunks
# Set B (ID=30): 62 chunks
# 
# Overlap: 23 chunks (Jaccard: 0.27)
# Unique to A: 22 chunks
# Unique to B: 39 chunks
#
# Sample overlapping chunks: [12345, 12346, 12350, ...]
# Sample unique to A: [11000, 11001, ...]
# Sample unique to B: [13000, 13001, ...]
```

### Effort Estimate
Small (straightforward set operations + CLI)

---

## Priority Ranking

| Feature | Impact | Effort | Recommended Priority |
|---------|--------|--------|---------------------|
| NOT/NEGATION primitives | High | Small | **Do first** |
| Result set comparison | Medium | Small | **Do second** |
| Query expansion | Medium | Medium | Do third |
| Geographic proximity | Low | High | Do last (or defer) |

---

## Acceptance Criteria (All Features)

For each feature:
- [ ] Unit tests for core logic
- [ ] Integration tests with database
- [ ] End-to-end workflow tests
- [ ] Smoke test added to `docs/SMOKE_TEST_10_MIN.md`
- [ ] Documentation updated

---

## Dependencies

- All P0-P2 features: ✅ Complete
- Entity system: ✅ Complete
- Primitive compilation system: ✅ Complete
- Result set system: ✅ Complete

---

## Related Work

- Original specification: Query & Conversation Features (#XX)
- Entity resolution implementation: ✅ Complete
- CO_OCCURS_WITH binary entities: ✅ Complete
- Performance indexes: ✅ Complete

---

## Labels

`enhancement` `P3` `future` `query-system`
