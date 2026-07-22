# Yakubovich Collision Analysis

## Summary

The alias "yakubovich" matches **2 different entities** that appear to be **duplicates** of the same person.

## Entity Details

### Entity 45240
- **Canonical Name**: Yakubovich
- **Entity Type**: person
- **Created**: 2026-01-26 17:02:54
- **Aliases**: 1 (Yakubovich)
- **Entity Mentions**: 0 (not yet matched)
- **Citations**: "Venona New York KGB 1945, 162"

### Entity 56335
- **Canonical Name**: Yakubovich
- **Entity Type**: person
- **Created**: 2026-01-27 07:09:20 (created 1 day later)
- **Aliases**: 1 (Yakubovich)
- **Entity Mentions**: 0 (not yet matched)
- **Citations**: "Venona New York KGB 1945, 162"

## Why It's Classified as "Harmless"

The collision is classified as **"harmless"** because:

1. **Single token**: "yakubovich" is a single-token person name (last name)
2. **Multiple entities**: There are 2 entities with this alias
3. **Person given name rule**: Single-token person names with multiple entities are considered too ambiguous to auto-match
4. **allow_ambiguous_person_token=False**: Both aliases have this flag set to False

This prevents false positives from common last names like "Yakubovich", "Smith", "Johnson", etc.

## Evidence They Are Duplicates

1. **Same canonical name**: Both entities have identical canonical names
2. **Same citation**: Both have the exact same citation: "Venona New York KGB 1945, 162"
3. **Same alias**: Both have only one alias: "Yakubovich"
4. **No mentions yet**: Neither entity has been matched to any chunks yet

## Recommendation: Merge Entities

These two entities should be **merged into one**. The later-created entity (56335) should likely be merged into the earlier one (45240), or vice versa depending on which has more complete data.

## Using Citation-Based Disambiguation

Yes, we **can** use `citation_document_labels` and `citation_page_lists` from the concordance export to resolve collisions!

### How It Works

The `resolve_collision_with_citations()` function (lines 1157-1240 in `extract_entity_mentions.py`) already implements this:

1. **Load entity citations**: For each candidate entity, load all citations from `entity_citations` table
2. **Parse citations**: Extract document names and page numbers from citation text
3. **Match document**: Check if the mention's document matches any citation document
4. **Match pages**: Check if the mention's PDF page numbers overlap with citation page numbers
5. **Score match**: Calculate confidence based on overlap quality

### Current Implementation

The function uses:
- `entity_citations` table (loaded via `batch_load_entity_citations()`)
- Citation text parsing (via `parse_citation_text()`)
- Document name matching (via `find_documents_for_citation()`)
- Page range expansion (via `expand_page_ranges()`)

### CSV Data Structure

The `concordance_export/entity_mentions.csv` includes:
- `citation_texts`: Full citation text (e.g., "Venona New York KGB 1945, 162")
- `citation_document_labels`: Extracted document labels (e.g., "Venona New York KGB 1945")
- `citation_page_lists`: Extracted page lists (e.g., "162")

These are computed from the `entity_citations` table during export.

### How to Use for Disambiguation

When a collision occurs:

1. **Get mention context**: Document ID and PDF page numbers where "yakubovich" appears
2. **Check citations**: For each candidate entity (45240, 56335), check if their citations match:
   - Document name matches (or is similar)
   - Page numbers overlap
3. **Resolve**: If only one entity has matching citations, use that one
4. **If both match**: Use the one with better page overlap, or mark as unresolved

### Example

If "yakubovich" appears in:
- Document: "Venona New York KGB 1945"
- Page: 162

Then:
- Entity 45240: Has citation "Venona New York KGB 1945, 162" → **MATCH** ✓
- Entity 56335: Has citation "Venona New York KGB 1945, 162" → **MATCH** ✓

Since both match, this suggests they're duplicates. But if one had a different citation (e.g., "Venona New York KGB 1943, 50"), we could distinguish them.

## Next Steps

1. **Merge the duplicate entities** (45240 and 56335)
2. **Re-run extraction** to see if "yakubovich" can now be matched
3. **Consider setting `allow_ambiguous_person_token=true`** if this is a known person who should be auto-matched
4. **Use citation-based disambiguation** for future collisions where citations differ
