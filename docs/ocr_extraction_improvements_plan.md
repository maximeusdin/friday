# OCR Entity Extraction Improvements - Implementation Plan

## Overview
Comprehensive improvements to handle OCR errors in entity extraction, focusing on cheap wins and sophisticated ranking.

## 1. Page/Block Hygiene (Cheap Wins)

### 1.1 Drop Non-Letter Lines
**Implementation**: `scripts/text_hygiene.py`
- Calculate per-line letter ratio
- Drop lines with < 30% letters (configurable)
- Preserve line numbers for context

### 1.2 Collapse Hyphenation
**Implementation**: `scripts/text_hygiene.py`
- Detect hyphenated words split across lines
- Pattern: `word-\nword` → `wordword` or `word word`
- Use dictionary check to validate

### 1.3 Normalize Characters
**Implementation**: `scripts/text_hygiene.py`
- Quotation marks: `"` `"` `'` `'` → `"`
- Ligatures: `ﬁ` → `fi`, `ﬂ` → `fl`
- Dashes: `—` `–` `-` → `-`
- Unicode normalization

### 1.4 Boilerplate Detection
**Implementation**: `scripts/text_hygiene.py`
- Keywords: "Deleted", "FOIPA", "Page", "Confidential"
- Header/footer patterns (repeated, page numbers)
- Downweight or exclude from extraction

## 2. Ranking Model (Not Single Threshold)

### 2.1 Fast Approximate Retrieval
**Implementation**: `scripts/extract_entities_ranked.py`
- **Top-K retrieval** (K=10-20) using:
  - Trigram similarity on `alias_norm`
  - Token overlap (Jaccard)
  - Length proximity (penalize large length differences)
  - Optional phonetic match (Soundex/Metaphone)

### 2.2 Final Score Combination
**Implementation**: `scripts/extract_entities_ranked.py`
- **Normalized edit distance**: Levenshtein / max(len(s1), len(s2))
- **Trigram Jaccard**: intersection/union of trigrams
- **Token-level edit**: for multi-token names
- **OCR confusion penalties/bonuses**: weighted substitutions

### 2.3 Ranking Formula
```
final_score = (
    0.3 * (1 - normalized_edit_distance) +
    0.3 * trigram_jaccard +
    0.2 * token_overlap +
    0.1 * length_proximity +
    0.1 * ocr_confusion_bonus
)
```

## 3. OCR-Aware Edit Distance

### 3.1 Confusion Table
**Implementation**: `scripts/ocr_confusion.py`
```python
OCR_CONFUSION_WEIGHTS = {
    ('rn', 'm'): 0.3,  # Low cost (common error)
    ('m', 'rn'): 0.3,
    ('cl', 'd'): 0.3,
    ('d', 'cl'): 0.3,
    ('l', '1'): 0.2,
    ('1', 'l'): 0.2,
    ('O', '0'): 0.2,
    ('0', 'O'): 0.2,
    ('I', 'l'): 0.2,
    ('l', 'I'): 0.2,
    ('S', '5'): 0.2,
    ('5', 'S'): 0.2,
    ('vv', 'w'): 0.3,
    ('w', 'vv'): 0.3,
    ('H', 'N'): 0.4,
    ('N', 'H'): 0.4,
    # Character drops (apostrophe, hyphen)
    ("'", ''): 0.1,
    ('-', ''): 0.1,
}
```

### 3.2 Weighted Damerau-Levenshtein
**Implementation**: `scripts/ocr_confusion.py`
- Use confusion weights instead of uniform cost
- Prefer substitutions that match OCR patterns
- Still deterministic, no combinatorial explosion

## 4. Local Context Features

### 4.1 Context Window Extraction
**Implementation**: `scripts/context_features.py`
- Extract ±8 tokens around entity span
- Extract ±1 line around entity span
- Extract document zone (header/body/footer)

### 4.2 Person Hints
**Implementation**: `scripts/context_features.py`
- Titles: Mr, Mrs, Dr, Ms, Prof, Senator, Representative
- First name dictionary (common names)
- Initials pattern: `C. M. J.` or `C.M.J.`
- Honorifics: Jr, Sr, III

### 4.3 Org Hints
**Implementation**: `scripts/context_features.py`
- Keywords: Bureau, Department, Committee, Office, Company, Corp, Inc, University, Division, Agency, Administration
- Patterns: "The X", "X of Y"

### 4.4 Place Hints
**Implementation**: `scripts/context_features.py`
- Prepositions: in, at, from, to, near
- State abbreviations: NY, CA, TX, etc.
- City patterns: "City of X", "X, State"

### 4.5 Boilerplate Zones
**Implementation**: `scripts/context_features.py`
- Detect header/footer regions
- Page number patterns
- Downweight entities in these zones

### 4.6 Context-Based Adjustments
**Implementation**: `scripts/extract_entities_ranked.py`
- **Adjust SpaCy confidence**:
  - Person with title → +0.1 confidence
  - Org with keyword → +0.1 confidence
  - Place with preposition → +0.1 confidence
  - In boilerplate zone → -0.2 confidence

- **Veto mislabels**:
  - Single token "NY" as ORG → change to GPE
  - "Red Amy" as person with "Army" context → change to org

## 5. Document-Level Consistency

### 5.1 Mention Clustering
**Implementation**: `scripts/document_consistency.py`
- Within same document, cluster mentions by:
  - OCR-weighted edit distance
  - Trigram similarity
  - Token overlap

### 5.2 Anchor-Based Correction
**Implementation**: `scripts/document_consistency.py`
- Identify high-confidence linked mentions (e.g., "John Service" matched well)
- Use as anchors
- Pull weaker variants toward anchor as proposals
- Example: "Iiv Tork Olty" → if "New York" exists as anchor, propose correction

### 5.3 Consistency Scoring
**Implementation**: `scripts/document_consistency.py`
- If entity appears multiple times:
  - Most common variant gets boost
  - Inconsistent variants get downweighted
  - Flag for review if variants are very different

## Implementation Order

### Phase 1: Cheap Wins (1-2 days)
1. Text hygiene (drop non-letter lines, collapse hyphenation, normalize)
2. Boilerplate detection
3. Update extraction pipeline to use cleaned text

### Phase 2: Ranking Model (2-3 days)
1. Implement top-K retrieval with multiple features
2. Implement final score combination
3. Replace single threshold with ranking

### Phase 3: OCR-Aware Distance (1-2 days)
1. Build confusion table
2. Implement weighted Damerau-Levenshtein
3. Integrate into ranking model

### Phase 4: Context Features (2-3 days)
1. Extract context windows
2. Implement hint detection (person/org/place)
3. Adjust confidence based on context
4. Implement veto rules

### Phase 5: Document Consistency (2-3 days)
1. Implement mention clustering
2. Anchor-based correction
3. Consistency scoring

## Files to Create/Modify

### New Files
1. `scripts/text_hygiene.py` - Text cleaning and normalization
2. `scripts/ocr_confusion.py` - OCR confusion weights and weighted edit distance
3. `scripts/context_features.py` - Context hint extraction
4. `scripts/document_consistency.py` - Document-level consistency
5. `scripts/extract_entities_ranked.py` - New ranked extraction pipeline

### Modified Files
1. `scripts/extract_entities_hybrid.py` - Integrate new components
2. `scripts/extract_entities_fuzzy_known.py` - Use ranking model
3. `config/ner_config.yaml` - Add new configuration options
4. `scripts/test_ner_extraction.py` - Test new features

## Configuration

### `config/ner_config.yaml` additions:
```yaml
text_hygiene:
  min_letter_ratio: 0.3  # Drop lines with < 30% letters
  collapse_hyphenation: true
  normalize_quotes: true
  normalize_ligatures: true
  boilerplate_keywords: ["Deleted", "FOIPA", "Page", "Confidential"]
  
ranking:
  top_k: 20  # Retrieve top 20 candidates
  weights:
    edit_distance: 0.3
    trigram_jaccard: 0.3
    token_overlap: 0.2
    length_proximity: 0.1
    ocr_confusion: 0.1
  
ocr_confusion:
  enabled: true
  weights_file: "config/ocr_confusion_weights.yaml"
  
context:
  window_tokens: 8  # ±8 tokens
  window_lines: 1   # ±1 line
  person_titles: ["Mr", "Mrs", "Dr", "Ms", "Prof", "Senator"]
  org_keywords: ["Bureau", "Department", "Committee", "Office"]
  place_prepositions: ["in", "at", "from", "to", "near"]
  
consistency:
  enabled: true
  cluster_threshold: 0.7  # Similarity threshold for clustering
  anchor_confidence_threshold: 0.8  # Minimum confidence for anchor
```

## Expected Improvements

1. **Precision**: Fewer false positives from garbage lines
2. **Recall**: Better matching of OCR-garbled entities
3. **Accuracy**: Context-aware corrections (e.g., "Red Amy" → "Red Army")
4. **Consistency**: Document-level entity normalization
5. **Confidence**: More accurate confidence scores

## Testing Strategy

1. Test on Silvermaster OCR document
2. Compare before/after metrics:
   - Precision/Recall
   - Correction rate
   - Confidence distribution
3. Manual review of top corrections
4. Validate against known entities from concordance
