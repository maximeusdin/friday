# Strategic Overview: Using Concordance Index for Entity Mention Disambiguation

## Executive Summary

The concordance index serves as a **disambiguation tool** for entity mention extraction. We extract mentions from chunks broadly, but when multiple entities could match the same alias (ambiguity), we use citation context (document + page) from the concordance index to resolve which entity is actually being referenced. This allows us to extract mentions that aren't in the concordance index while leveraging expert knowledge to resolve ambiguous cases.

## The Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CONCORDANCE INDEX INGEST                                     │
│    (ingest_concordance_tab_aware.py)                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Creates disambiguation knowledge base: │
        │  • entities                             │
        │  • entity_aliases                       │
        │  • entity_citations (document + pages)  │
        │  • entity_links                        │
        └───────────────────────────────────────┘
                            │
                            │ (parallel process)
                            │
        ┌───────────────────────────────────────┐
        │ 2. DOCUMENT INGEST                    │
        │    (ingest_venona_pdf.py, etc.)       │
        │                                       │
        │ Creates chunks with provenance:       │
        │  • documents                          │
        │  • pages                              │
        │  • chunks                             │
        │  • chunk_metadata (document_id)      │
        │  • chunk_pages                        │
        └───────────────────────────────────────┘
                            │
                            │ (both processes complete)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ENTITY MENTION EXTRACTION                                    │
│    (extract_entity_mentions.py - full scan)                     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Extract mentions from ALL chunks:     │
        │                                       │
        │ 1. Scan chunks                        │
        │ 2. Match aliases (normalized)         │
        │ 3. Find potential entity matches      │
        │ 4. Identify ambiguous cases:          │
        │    • Same alias → multiple entities   │
        │    • Same surface → multiple aliases  │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 4. DISAMBIGUATION                      │
        │    (using concordance citations)       │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ For each ambiguous mention:            │
        │                                       │
        │ 1. Get mention context:                │
        │    • document_id                       │
        │    • page_ids (from chunk_pages)      │
        │                                       │
        │ 2. For each candidate entity:          │
        │    • Get citations from               │
        │      entity_citations                  │
        │    • Check if citation matches:        │
        │      - Same document?                  │
        │      - Overlapping pages?               │
        │                                       │
        │ 3. Resolve ambiguity:                  │
        │    • If ONE entity has matching       │
        │      citation → resolve to that       │
        │    • If MULTIPLE entities match →     │
        │      use dominance rules or enqueue   │
        │    • If NO entities match →           │
        │      may be new entity or enqueue     │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Store resolved mentions:               │
        │  • entity_id (resolved)                │
        │  • chunk_id                            │
        │  • document_id                         │
        │  • surface                             │
        │  • confidence (higher if citation      │
        │    matched)                             │
        │  • disambiguation_method               │
        └───────────────────────────────────────┘
```

## Strategic Advantages

### 1. **Broad Coverage with Smart Disambiguation**

**Approach**: Extract from all chunks, use concordance index to resolve ambiguities.

**Benefits**:
- **High Recall**: Extract mentions even if not in concordance index
- **High Precision**: Use expert knowledge to resolve ambiguous cases
- **Discover New Entities**: Can identify entities not yet in concordance index

### 2. **Context-Based Resolution**

When multiple entities share the same alias, use document + page context:

```
Example: "Albert" found in chunk from "Vassiliev Black Notebook, page 79"

Candidate entities:
- Entity A: Iskhak Akhmerov (has citation: "Vassiliev Black Notebook, 79")
- Entity B: Albert Einstein (has citation: "Other Document, 123")
- Entity C: Albert Smith (no citations)

Resolution:
- Entity A matches citation (same document, same page) → RESOLVE to Entity A
- Entity B doesn't match → REJECT
- Entity C has no citations → MAYBE (could be new entity or missing citation)
```

### 3. **Handles Missing Citations**

The concordance index may not have all mentions:
- Some entities may not be in the index
- Some mentions may be in documents not yet indexed
- Some citations may be incomplete

**Our approach**: Still extract these mentions, but flag them for review or use other disambiguation methods.

### 4. **Confidence Scoring**

Mentions can be scored based on citation match:

```
High Confidence (1.0):
  - Alias matches exactly
  - Citation matches (document + page)
  - Single candidate entity

Medium Confidence (0.8):
  - Alias matches
  - Citation matches, but multiple candidates
  - Resolved using dominance rules

Low Confidence (0.5):
  - Alias matches
  - No citation match
  - May be new entity or missing citation

Very Low Confidence (0.3):
  - Ambiguous alias
  - No citation match
  - Multiple candidates, no resolution
```

## Implementation Strategy

### Integration with Existing Extraction Pipeline

The citation-based disambiguation integrates into the existing `extract_entity_mentions.py` collision resolution logic. The current flow is:

```
1. Find alias match
2. If collision (multiple entities):
   a. Filter by policies → if single candidate remains, use it
   b. Apply dominance rules → if one dominates, use it
   c. If still unresolved → enqueue for review
```

**New flow with citation disambiguation**:

```
1. Find alias match
2. If collision (multiple entities):
   a. **NEW: Check citations** → if one entity has matching citation, use it
   b. Filter by policies → if single candidate remains, use it
   c. Apply dominance rules → if one dominates, use it
   d. If still unresolved → enqueue for review
```

### Citation-Based Disambiguation Function

Add a new function to check citations during collision resolution:

```python
def resolve_collision_with_citations(
    cur,
    alias_infos: List[AliasInfo],
    document_id: int,
    page_ids: List[int],
    chunk_id: int
) -> Tuple[Optional[AliasInfo], float, str]:
    """
    Resolve collision by checking if any candidate entity has citations
    that match the mention's document and pages.
    
    Returns:
        (resolved_alias_info, confidence_score, method)
        - resolved_alias_info: The entity to use, or None if no match
        - confidence_score: 0.0-1.0 based on citation match quality
        - method: 'citation_exact', 'citation_fuzzy', or None
    """
    # Get PDF page numbers for the chunk's pages
    pdf_page_numbers = get_pdf_page_numbers(cur, page_ids)
    
    # Check citations for each candidate entity
    candidate_scores = []
    for ai in alias_infos:
        # Get all citations for this entity
        citations = get_entity_citations(cur, ai.entity_id)
        
        best_score = 0.0
        for citation in citations:
            # Parse citation to get document and pages
            citation_locations = parse_citation_text(citation['citation_text'])
            
            for loc in citation_locations:
                # Check if citation document matches mention document
                citation_docs = find_documents_for_citation(cur, loc)
                citation_doc_ids = [doc_id for doc_id, _ in citation_docs]
                
                if document_id not in citation_doc_ids:
                    continue  # Different document
                
                # Check page overlap
                citation_pages = expand_page_ranges(loc.pages)
                overlap = set(pdf_page_numbers) & set(citation_pages)
                
                if overlap:
                    # Calculate match score
                    if len(overlap) == len(pdf_page_numbers) and len(overlap) == len(citation_pages):
                        score = 1.0  # Exact match
                    elif len(overlap) >= len(pdf_page_numbers) * 0.8:
                        score = 0.9  # Most pages match
                    else:
                        score = 0.7  # Some pages match
                    
                    best_score = max(best_score, score)
        
        candidate_scores.append((ai, best_score))
    
    # Find best match
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    if candidate_scores[0][1] >= 0.7:  # Threshold for citation match
        if len(candidate_scores) == 1 or candidate_scores[1][1] < candidate_scores[0][1] * 0.8:
            # Clear winner
            return candidate_scores[0][0], candidate_scores[0][1], 'citation_match'
    
    return None, 0.0, None
```

### Integration Point in find_exact_matches

Modify the collision resolution section in `find_exact_matches`:

```python
if is_collision:
    # 0) NEW: Citation-based resolution (highest priority)
    if document_id is not None:
        page_ids = get_chunk_page_ids(cur, chunk_id)
        citation_resolved, citation_confidence, citation_method = \
            resolve_collision_with_citations(
                cur, alias_infos, document_id, page_ids, chunk_id
            )
        if citation_resolved:
            resolved_alias_info = citation_resolved
            # Track citation-based resolution
            if rejection_stats is not None:
                rejection_stats.setdefault('collision_auto_resolved', {})
                rejection_stats['collision_auto_resolved'][
                    f"{ngram_text} (citation_match_{citation_method})"
                ] = rejection_stats['collision_auto_resolved'].get(
                    f"{ngram_text} (citation_match_{citation_method})", 0
                ) + 1
            # Continue to store mention with citation confidence
    
    # 1) Filtered-single-candidate resolution (if citation didn't resolve)
    if resolved_alias_info is None:
        # ... existing filtered_candidates logic ...
    
    # 2) Dominance resolution (if still unresolved)
    if resolved_alias_info is None:
        # ... existing dominance rules logic ...
```

### Phase 3: Store with Disambiguation Metadata

Store mentions with additional fields:

```sql
entity_mentions:
  - entity_id (resolved)
  - chunk_id
  - document_id
  - surface
  - confidence
  - disambiguation_method: 
    - 'direct_match' (single entity, no ambiguity)
    - 'citation_match' (resolved using citation)
    - 'dominance_rule' (resolved using policy)
    - 'unresolved' (needs review)
    - 'unknown_entity' (not in concordance index)
  - citation_match_score (0.0-1.0)
  - candidate_entities (array of entity_ids considered)
```

## Example: Disambiguation Flow

### Scenario: "Albert" found in chunk

**Chunk context**:
- document_id: 42 ("Vassiliev Black Notebook")
- pages: [79]
- surface: "Albert"
- alias_norm: "albert"

**Step 1: Find candidate entities**
```
Entities with alias "albert":
- Entity 123: Iskhak Akhmerov
- Entity 456: Albert Einstein  
- Entity 789: Albert Smith
```

**Step 2: Check citations for each entity**

Entity 123 (Iskhak Akhmerov):
```
Citations:
- "Vassiliev Black Notebook, 79" ✓ MATCHES
- "Vassiliev White Notebook #1, 55"
```

Entity 456 (Albert Einstein):
```
Citations:
- "Other Document, 123" ✗ NO MATCH
```

Entity 789 (Albert Smith):
```
Citations: None
```

**Step 3: Resolve**
```
Entity 123: citation_match_score = 1.0 (exact match)
Entity 456: citation_match_score = 0.0 (no match)
Entity 789: citation_match_score = 0.0 (no citations)

Resolution: Entity 123 (Iskhak Akhmerov)
Confidence: 1.0
Disambiguation_method: 'citation_match'
```

### Scenario: No citation match

**Chunk context**:
- document_id: 42
- pages: [200]
- surface: "Albert"
- alias_norm: "albert"

**Candidates**: Same as above

**Citation check**:
- Entity 123: Citations don't include page 200
- Entity 456: No match
- Entity 789: No citations

**Resolution options**:
1. **Use dominance rules**: If Entity 123 is a covername and others are not, prefer Entity 123
2. **Enqueue for review**: Mark as unresolved, needs human review
3. **Mark as unknown**: Could be a new entity or missing citation

## Key Design Decisions

### 1. **Two-Pass Extraction**

**Pass 1: Extract all mentions**
- Scan all chunks
- Match aliases
- Identify ambiguous cases

**Pass 2: Disambiguate**
- Use citation context
- Apply resolution rules
- Store with confidence scores

**Why two passes**: Allows us to see all potential mentions first, then apply disambiguation logic systematically.

### 2. **Citation Matching Strategy**

**Exact match**: Document + page match exactly
**Fuzzy match**: Document matches, pages nearby (within N pages)
**Document-only match**: Document matches, different pages

**Why fuzzy matching**: Chunks may span multiple pages, citations may reference nearby pages.

### 3. **Handling Unknown Entities**

When a mention doesn't match any entity in the concordance index:

**Options**:
1. **Store as unknown**: Create placeholder entity or flag for review
2. **Use other signals**: Document context, surrounding text, etc.
3. **Defer**: Mark for human review

**Why this matters**: Allows discovery of entities not yet in the concordance index.

### 4. **Confidence Propagation**

Confidence scores flow through the pipeline:

```
Alias match confidence (from extraction)
  × Citation match confidence (from disambiguation)
  = Final confidence
```

**Why this matters**: Downstream systems can filter by confidence, prioritize high-confidence mentions.

## Comparison: Targeting vs. Disambiguation

### Citation-Based Targeting (Previous Approach)
```
Pros:
  - Very high precision
  - Only extracts from verified locations
  - Fast (only process cited chunks)

Cons:
  - Lower recall (misses uncited mentions)
  - Can't discover new entities
  - Requires complete citation coverage
```

### Citation-Based Disambiguation (New Approach)
```
Pros:
  - High recall (extracts from all chunks)
  - High precision (resolves ambiguities)
  - Can discover new entities
  - Works with incomplete citations

Cons:
  - More processing (full scan + disambiguation)
  - Some mentions may remain unresolved
  - Requires good citation coverage for disambiguation
```

## Implementation Plan

### Step 1: Modify Extraction to Track Ambiguity

```python
# In extract_entity_mentions.py
matches = find_exact_matches(...)

# For each match:
if len(candidate_entities) > 1:
    # Mark as ambiguous
    mention['is_ambiguous'] = True
    mention['candidate_entity_ids'] = [e.id for e in candidate_entities]
else:
    mention['is_ambiguous'] = False
    mention['entity_id'] = candidate_entities[0].id
```

### Step 2: Add Disambiguation Pass

```python
# New function: disambiguate_mentions()
def disambiguate_mentions(conn, ambiguous_mentions):
    for mention in ambiguous_mentions:
        # Get mention context
        document_id = mention['document_id']
        page_ids = get_chunk_pages(mention['chunk_id'])
        
        # Check citations for each candidate
        for candidate_entity_id in mention['candidate_entity_ids']:
            citations = get_entity_citations(candidate_entity_id)
            match_score = check_citation_match(
                document_id, page_ids, citations
            )
            # ... resolve based on scores
```

### Step 3: Store with Metadata

```python
# Store mention with disambiguation info
insert_mention({
    'entity_id': resolved_entity_id,
    'confidence': final_confidence,
    'disambiguation_method': method,
    'citation_match_score': match_score,
    'candidate_entity_ids': [...]
})
```

## Benefits Summary

1. **High Recall**: Extract mentions from all chunks, not just cited ones
2. **High Precision**: Use citations to resolve ambiguous cases
3. **Discoverability**: Can find entities not in concordance index
4. **Robustness**: Works even with incomplete citation coverage
5. **Confidence Scoring**: Provides quality signals for downstream use
6. **Flexibility**: Can combine citation matching with other disambiguation methods

## How It Works: Step-by-Step

### Example: Extracting "Albert" from a Chunk

**Input**: Chunk from "Vassiliev Black Notebook", page 79, containing text "Albert mentioned..."

**Step 1: Alias Matching**
```
1. Tokenize chunk → find "Albert"
2. Normalize → "albert"
3. Lookup in alias_norm_set → FOUND
4. Get all entities with alias "albert":
   - Entity 123: Iskhak Akhmerov (covername)
   - Entity 456: Albert Einstein (person_full)
   - Entity 789: Albert Smith (person_full)
```

**Step 2: Collision Detection**
```
Multiple entities match → COLLISION detected
```

**Step 3: Citation-Based Disambiguation** (NEW)
```
For each candidate entity, check citations:

Entity 123 (Iskhak Akhmerov):
  Citations:
    - "Vassiliev Black Notebook, 79" ✓
    - "Vassiliev White Notebook #1, 55"
  Match: document_id=42, page=79 → EXACT MATCH
  Score: 1.0

Entity 456 (Albert Einstein):
  Citations:
    - "Other Document, 123"
  Match: No match
  Score: 0.0

Entity 789 (Albert Smith):
  Citations: None
  Score: 0.0

Resolution: Entity 123 (highest score, clear winner)
Method: 'citation_match'
Confidence: 1.0
```

**Step 4: Store Mention**
```
entity_mentions:
  entity_id: 123
  chunk_id: 6083
  document_id: 42
  surface: "Albert"
  confidence: 1.0
  method: 'alias_exact' (or could encode 'citation_match' in method)
```

### Example: No Citation Match

**Input**: Chunk from "Vassiliev Black Notebook", page 200, containing "Albert..."

**Step 1-2**: Same as above (collision detected)

**Step 3: Citation Check**
```
Entity 123: Citations don't include page 200 → Score: 0.0
Entity 456: No match → Score: 0.0
Entity 789: No citations → Score: 0.0

No citation match → Continue to policy/dominance rules
```

**Step 4: Fallback to Existing Rules**
```
Apply policy filters:
- Entity 123: covername, is_auto_match=true → PASSES
- Entity 456: person_full, is_auto_match=true → PASSES
- Entity 789: person_full, is_auto_match=false → FILTERED OUT

Apply dominance rules:
- Entity 123 (covername) vs Entity 456 (person_full)
- Covername dominates → Entity 123 selected

Resolution: Entity 123
Method: 'dominance_rule'
Confidence: 0.8 (lower than citation match)
```

### Example: Unknown Entity

**Input**: Chunk containing "NewPerson" (not in concordance index)

**Step 1: Alias Matching**
```
Normalize "NewPerson" → "newperson"
Lookup in alias_norm_set → NOT FOUND
```

**Step 2: Store or Skip**
```
Options:
1. Skip (current behavior) - only extract known aliases
2. Store as unknown_entity - flag for review
3. Create placeholder entity - for discovery
```

**With citation disambiguation**: This mention would still be extracted if we choose option 2 or 3, but wouldn't benefit from citation matching since it's not in the index.

## Handling Partial and Fuzzy Matches

### The Challenge

The current extraction uses **exact normalized matching**: only matches if the normalized alias exactly matches the normalized text. This misses:

1. **Partial matches**: "Smith" when alias is "John Smith"
2. **Fuzzy matches**: "Akhmeroff" when alias is "Akhmerov" (misspelling)
3. **OCR errors**: "Akhmerov" when alias is "Akhmerov" but text has OCR corruption

### Solution: Multi-Tier Matching with Citation Disambiguation

Extend the matching pipeline to try multiple matching strategies, then use citations to resolve ambiguities:

```
Tier 1: Exact Normalized Match (current)
  "albert" → matches "albert" exactly
  Confidence: 1.0

Tier 2: Partial Match (substring)
  "smith" → matches "john smith" (last name only)
  Confidence: 0.7 (lower because partial)

Tier 3: Fuzzy Match (edit distance)
  "akhmeroff" → matches "akhmerov" (1 character difference)
  Confidence: 0.6 (lower because fuzzy)
```

### Integration with Citation Disambiguation

When partial or fuzzy matches create ambiguities, citations provide crucial disambiguation:

**Example: Partial Match "Smith"**

```
Chunk text: "...Smith was mentioned..."
Location: "Vassiliev Black Notebook, page 79"

Tier 2 Partial Match finds:
- Entity A: John Smith (alias: "John Smith", alias_norm: "john smith")
- Entity B: Mary Smith (alias: "Mary Smith", alias_norm: "mary smith")
- Entity C: Smith Corporation (alias: "Smith", alias_norm: "smith")

All match "smith" as substring → COLLISION

Citation Check:
- Entity A: Citations include "Vassiliev Black Notebook, 79" ✓
- Entity B: Citations don't include this document
- Entity C: Citations don't include this document

Resolution: Entity A (John Smith)
Confidence: 0.7 (partial match) × 1.0 (citation match) = 0.7
```

**Example: Fuzzy Match "Akhmeroff"**

```
Chunk text: "...Akhmeroff reported..."
Location: "Vassiliev Black Notebook, page 79"

Tier 3 Fuzzy Match finds:
- Entity A: Iskhak Akhmerov (alias: "Akhmerov", edit_distance=1)
- Entity B: Other Akhmerov (alias: "Akhmerov", edit_distance=1)

Both have edit_distance=1 → COLLISION

Citation Check:
- Entity A: Citations include "Vassiliev Black Notebook, 79" ✓
- Entity B: Citations don't include this document

Resolution: Entity A (Iskhak Akhmerov)
Confidence: 0.6 (fuzzy match) × 1.0 (citation match) = 0.6
```

### Implementation Strategy

#### Step 1: Extend Matching Logic

```python
def find_matches_with_fuzzy(
    chunk_text: str,
    aliases_by_norm: Dict[str, List[AliasInfo]],
    alias_norm_set: Set[str],
    chunk_id: int,
    document_id: int,
    max_edit_distance: int = 2,
    min_partial_length: int = 4
) -> List[Tuple[AliasInfo, str, float, str]]:
    """
    Find matches using multiple strategies:
    - Exact normalized match
    - Partial match (substring)
    - Fuzzy match (edit distance)
    
    Returns: List of (alias_info, surface, confidence, match_type)
    """
    matches = []
    
    # Tokenize and normalize chunk text
    tokens = tokenize_text(chunk_text)
    normalized_tokens = [normalize_alias(t[2]) for t in tokens]
    
    # Tier 1: Exact matches (existing logic)
    exact_matches = find_exact_matches(...)
    for match in exact_matches:
        matches.append((match, 1.0, 'exact'))
    
    # Tier 2: Partial matches (substring)
    # For each token, check if it's a substring of any alias
    for token_norm in normalized_tokens:
        if len(token_norm) < min_partial_length:
            continue
        
        # Find aliases where token is a substring
        for alias_norm, alias_infos in aliases_by_norm.items():
            if token_norm in alias_norm or alias_norm in token_norm:
                # Check if this is a meaningful partial match
                # (e.g., "smith" in "john smith" is meaningful)
                if is_meaningful_partial_match(token_norm, alias_norm):
                    for ai in alias_infos:
                        matches.append((ai, token_norm, 0.7, 'partial'))
    
    # Tier 3: Fuzzy matches (edit distance)
    # For each token, find aliases within edit distance
    for token_norm in normalized_tokens:
        if len(token_norm) < min_partial_length:
            continue
        
        for alias_norm in alias_norm_set:
            distance = levenshtein_distance(token_norm, alias_norm)
            if distance <= max_edit_distance:
                # Calculate confidence based on distance
                confidence = 1.0 - (distance * 0.2)  # 0.8 for distance=1, 0.6 for distance=2
                alias_infos = aliases_by_norm.get(alias_norm, [])
                for ai in alias_infos:
                    matches.append((ai, token_norm, confidence, 'fuzzy'))
    
    return matches
```

#### Step 2: Group Matches and Detect Collisions

```python
# Group matches by position in chunk
matches_by_position = group_matches_by_position(matches)

for position, position_matches in matches_by_position.items():
    # If multiple entities match at same position → collision
    if len(set(m.ai.entity_id for m in position_matches)) > 1:
        # Use citation disambiguation
        resolved = resolve_with_citations(
            position_matches, document_id, page_ids
        )
```

#### Step 3: Citation Disambiguation for Partial/Fuzzy Matches

```python
def resolve_with_citations(
    candidate_matches: List[Tuple[AliasInfo, float, str]],
    document_id: int,
    page_ids: List[int],
    cur
) -> Optional[Tuple[AliasInfo, float]]:
    """
    Resolve ambiguous matches (exact, partial, or fuzzy) using citations.
    
    Returns: (resolved_alias_info, final_confidence) or None
    """
    # Score each candidate based on:
    # 1. Match quality (exact > partial > fuzzy)
    # 2. Citation match
    
    candidate_scores = []
    for ai, match_confidence, match_type in candidate_matches:
        # Get citation match score
        citation_score = check_citation_match(
            cur, ai.entity_id, document_id, page_ids
        )
        
        # Combined score: match confidence × citation boost
        if citation_score > 0.8:
            # Strong citation match boosts confidence
            final_confidence = min(1.0, match_confidence * 1.2)
        elif citation_score > 0.5:
            # Moderate citation match
            final_confidence = match_confidence * 1.1
        else:
            # No citation match - use match confidence as-is
            final_confidence = match_confidence
        
        candidate_scores.append((ai, final_confidence, match_type))
    
    # Find best candidate
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    if candidate_scores[0][1] > 0.6:  # Threshold
        if len(candidate_scores) == 1 or \
           candidate_scores[1][1] < candidate_scores[0][1] * 0.8:
            return candidate_scores[0][0], candidate_scores[0][1]
    
    return None
```

### Match Type Confidence Levels

```
Exact Match:
  - Normalized alias exactly matches normalized text
  - Confidence: 1.0
  - Example: "albert" → "albert"

Partial Match (Last Name):
  - Token is a meaningful substring of alias
  - Confidence: 0.7
  - Example: "smith" → "john smith"
  - Requires: Token length >= 4, appears as word boundary

Partial Match (First Name):
  - Less reliable than last name
  - Confidence: 0.5
  - Example: "john" → "john smith"
  - Only if no last name matches found

Fuzzy Match (Edit Distance 1):
  - One character difference
  - Confidence: 0.8
  - Example: "akhmeroff" → "akhmerov"

Fuzzy Match (Edit Distance 2):
  - Two character differences
  - Confidence: 0.6
  - Example: "akhmeroff" → "akhmerov" (if distance=2)
```

### Citation Boost

Citations can boost confidence for partial/fuzzy matches:

```
Partial Match + Citation Match:
  0.7 (partial) × 1.2 (citation boost) = 0.84

Fuzzy Match + Citation Match:
  0.8 (fuzzy) × 1.2 (citation boost) = 0.96

Partial Match + No Citation:
  0.7 (partial) × 1.0 (no boost) = 0.7
```

### Handling Ambiguous Partial/Fuzzy Matches

When multiple entities match via partial/fuzzy:

**Strategy 1: Citation Disambiguation (Preferred)**
```
If one entity has citation match → use it
If multiple entities have citation match → use highest citation score
If no entities have citation match → fall back to other rules
```

**Strategy 2: Match Quality**
```
Prefer exact matches over partial
Prefer partial matches over fuzzy
Prefer shorter edit distance
```

**Strategy 3: Entity Type**
```
Prefer covernames over person_full (for partial matches)
Prefer person_full over generic_word
```

### Example: Complex Disambiguation

**Scenario**: Chunk contains "Akhmeroff" (misspelling) in "Vassiliev Black Notebook, page 79"

```
Step 1: Matching
- Exact match: None
- Partial match: None (not a substring)
- Fuzzy match (distance=1):
  - Entity A: Iskhak Akhmerov (alias: "Akhmerov")
  - Entity B: Other Person (alias: "Akhmerov") 
  Both match with confidence 0.8

Step 2: Collision Detection
Multiple entities match → COLLISION

Step 3: Citation Disambiguation
- Entity A: Has citation "Vassiliev Black Notebook, 79" → Score: 1.0
- Entity B: No matching citation → Score: 0.0

Step 4: Resolution
- Entity A: 0.8 (fuzzy) × 1.2 (citation boost) = 0.96
- Entity B: 0.8 (fuzzy) × 1.0 (no boost) = 0.8
- Resolution: Entity A (Iskhak Akhmerov)
- Final confidence: 0.96
- Method: 'fuzzy_citation_match'
```

## Future Enhancements

1. **Multi-Signal Disambiguation**: Combine citations with:
   - Document metadata (dates, senders, recipients)
   - Surrounding text context
   - Entity relationships
   - Temporal patterns

2. **Learning from Resolutions**: Track which disambiguation methods work best

3. **Citation Expansion**: If citation matches, look for nearby mentions of same entity

4. **Incremental Updates**: Re-disambiguate only when new citations added

5. **Adaptive Thresholds**: Adjust confidence thresholds based on match type and citation quality

6. **OCR-Aware Fuzzy Matching**: Use OCR error models to improve fuzzy matching accuracy
