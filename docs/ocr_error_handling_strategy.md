# OCR Error Handling Strategy for Entity Extraction

## Current Situation

The PDF text extraction is now working correctly (using PyMuPDF), but OCR errors are causing garbled entity names:
- "Ihilip Jacob Jaffewere" → likely "Philip Jacob Jaffe"
- "Red Amy" → likely "Red Army" (org, not person)
- "Mdse Lukenbill" → likely "Mildred Lukenbill"
- "Iiv Tork Olty" → likely "New York City"
- "Mltad Stataa" → likely "United States"
- "Tte Swptoabssr" → likely "The Subcommittee"
- "Coatmnlat Pyrtj" → likely "Committee Party"
- "Xrleh Republicm" → likely "Czech Republic"
- "Sfeily Wwker" → likely "Shirley Walker"

## Multi-Tier Strategy

### Tier 1: Fuzzy Matching Against Known Entities (Already Implemented)
**Status**: ✅ Implemented in `extract_entities_fuzzy_known.py`

**How it works**:
- Uses PostgreSQL `pg_trgm` extension for trigram similarity
- Matches extracted surface forms against `entity_aliases` table
- Lower similarity threshold for OCR documents (0.6 vs 0.7 for clean)

**Limitations**:
- Only works if entity already exists in database
- May miss new entities or entities not yet in concordance

**Improvements needed**:
1. **Increase candidate surface extraction** - Currently limited to 200 candidates for OCR
2. **Lower similarity threshold further** - Consider 0.55-0.6 for OCR (vs 0.6-0.7)
3. **Multi-token matching** - Handle cases where OCR splits/merges tokens
4. **Character-level fuzzy matching** - Use Levenshtein distance for common OCR errors

### Tier 2: OCR Error Pattern Correction
**Status**: ⚠️ Partially implemented (needs enhancement)

**Common OCR Error Patterns**:
1. **Character substitutions**:
   - `rn` → `m` ("Mltad" → "United")
   - `rn` → `rr` ("Stataa" → "States")
   - `cl` → `d` ("Coatmnlat" → "Committee")
   - `e` → `c` ("Jaffewere" → "Jaffe")
   - `v` → `w` ("Wwker" → "Walker")
   - `i` → `l` ("Ihilip" → "Philip")
   - `h` → `li` ("Xrleh" → "Czech")

2. **Missing characters**:
   - "Red Amy" → "Red Army" (missing 'r')
   - "Saw Tork" → "New York" (missing 'N', 'e')

3. **Extra characters**:
   - "Jaffewere" → "Jaffe" (extra "were")

4. **Character transpositions**:
   - "Swptoabssr" → "Subcommittee" (multiple transpositions)

**Implementation Approach**:
```python
# Character substitution dictionary for common OCR errors
OCR_SUBSTITUTIONS = {
    'rn': ['m', 'rr', 'nn'],
    'cl': ['d'],
    'vv': ['w'],
    'ii': ['ll', 'h'],
    'ee': ['c'],
    # ... more patterns
}

# Generate candidate corrections
def generate_ocr_corrections(surface: str) -> List[str]:
    candidates = [surface]
    # Apply character-level corrections
    # Use edit distance to find closest matches
    return candidates
```

### Tier 3: Context-Aware Entity Validation
**Status**: ❌ Not implemented

**Strategy**:
1. **Validate against document context**:
   - Check if entity appears multiple times (consistency)
   - Look for surrounding context clues (titles, organizations)
   - Cross-reference with other mentions

2. **Entity type validation**:
   - "Red Amy" classified as person → should be org ("Red Army")
   - Use surrounding words to correct entity type

3. **Pattern-based validation**:
   - Person names: Usually 2-3 words, capitalized
   - Organizations: Often contain "Committee", "Bureau", "Office"
   - Places: Often preceded by "in", "at", "from"

### Tier 4: Post-Extraction Review Queue
**Status**: ✅ Implemented in `review_entity_extractions.py`

**Enhancements needed**:
1. **Prioritize low-confidence OCR entities** for review
2. **Group similar entities** for batch review
3. **Suggest corrections** based on fuzzy matches
4. **Track correction patterns** to improve future extraction

### Tier 5: Machine Learning-Based Correction
**Status**: ❌ Not implemented (future enhancement)

**Approach**:
1. **Train OCR error correction model**:
   - Use existing entity_aliases as ground truth
   - Learn common OCR error patterns
   - Character-level sequence-to-sequence model

2. **Use transformer models**:
   - Fine-tune BERT/RoBERTa on OCR text
   - Better context understanding
   - Can correct multiple errors simultaneously

## Immediate Action Plan

### Phase 1: Enhance Fuzzy Matching (1-2 days)
1. **Lower similarity thresholds for OCR**:
   - Current: 0.6-0.7
   - Proposed: 0.55-0.65 for OCR documents
   - Configurable per document quality

2. **Increase candidate extraction**:
   - Current: 200 candidates for OCR
   - Proposed: 500-1000 candidates
   - Use sliding window for multi-token entities

3. **Character-level fuzzy matching**:
   - Add Levenshtein distance as secondary metric
   - Handle common OCR substitutions
   - Weight trigram similarity + edit distance

### Phase 2: OCR Pattern Correction (2-3 days)
1. **Build OCR error pattern dictionary**:
   - Analyze existing entity_aliases vs extracted entities
   - Identify common substitution patterns
   - Create correction rules

2. **Implement pattern-based correction**:
   - Generate candidate corrections for garbled entities
   - Match against known entities
   - Use confidence scoring

3. **Multi-token entity handling**:
   - Detect when OCR splits entities ("Iiv Tork" → "New York")
   - Detect when OCR merges entities
   - Reconstruct full entity names

### Phase 3: Context Validation (2-3 days)
1. **Entity consistency checking**:
   - Track entity mentions across document
   - Use most common variant
   - Flag inconsistencies for review

2. **Context-aware type correction**:
   - Use surrounding words to validate entity type
   - Fix misclassifications (e.g., "Red Amy" as person → org)

3. **Pattern-based validation**:
   - Validate person names against name patterns
   - Validate organizations against org patterns
   - Validate places against place patterns

### Phase 4: Review Queue Enhancement (1-2 days)
1. **Prioritize OCR entities**:
   - Lower confidence threshold for OCR review queue
   - Group similar entities together
   - Suggest corrections from fuzzy matches

2. **Batch review interface**:
   - Show multiple variants side-by-side
   - Quick accept/reject/correct actions
   - Learn from corrections

## Configuration Updates

### `config/ner_config.yaml` Changes:
```yaml
ocr:
  pattern_weight: 0.2  # Lower weight for pattern (more noise)
  ner_weight: 0.2      # Lower weight for NER (less reliable)
  fuzzy_weight: 0.6    # Higher weight for fuzzy (most reliable)
  confidence_threshold: 0.55  # Lower threshold for OCR
  fuzzy_similarity_threshold: 0.55  # Lower similarity threshold
  levenshtein_threshold: 0.8  # New: edit distance threshold
  max_candidates: 1000  # Increased from 200
  ocr_correction_enabled: true  # New: enable OCR correction
```

## Metrics to Track

1. **Precision/Recall**:
   - Measure before/after OCR correction
   - Track false positives/negatives

2. **Correction Rate**:
   - % of entities that get corrected
   - Most common correction patterns

3. **Review Queue Size**:
   - Track entities needing review
   - Time to resolution

4. **Fuzzy Match Success Rate**:
   - % of entities matched via fuzzy matching
   - Average similarity scores

## Example: "Ihilip Jacob Jaffewere" Correction

**Current Flow**:
1. Pattern-based extractor finds "Ihilip Jacob Jaffewere" as person
2. NER extractor finds it as person
3. Fuzzy matching checks against known entities
4. If "Philip Jacob Jaffe" exists in DB with similarity > 0.55 → match

**Enhanced Flow**:
1. Pattern-based extractor finds "Ihilip Jacob Jaffewere"
2. Generate OCR corrections:
   - "Philip Jacob Jaffewere" (i→P)
   - "Ihilip Jacob Jaffe" (remove "were")
   - "Philip Jacob Jaffe" (both corrections)
3. Fuzzy match all candidates
4. If "Philip Jacob Jaffe" matches with similarity > 0.55 → use corrected version
5. Context validation: Check if "Jaffe" appears elsewhere in document
6. If consistent → high confidence; if inconsistent → flag for review

## Long-Term Vision

1. **Self-Improving System**:
   - Learn from review corrections
   - Update OCR error patterns automatically
   - Improve similarity thresholds based on feedback

2. **Domain-Specific Models**:
   - Train OCR correction on historical documents
   - Learn domain-specific entity patterns
   - Adapt to different document types

3. **Hybrid Human-AI Workflow**:
   - AI suggests corrections
   - Human validates high-value entities
   - System learns from corrections

## Next Steps

1. **Immediate**: Implement Phase 1 (enhance fuzzy matching)
2. **Short-term**: Implement Phase 2 (OCR pattern correction)
3. **Medium-term**: Implement Phase 3 (context validation)
4. **Long-term**: Consider ML-based correction (Phase 5)

## Files to Create/Modify

1. **New**: `scripts/extract_entities_ocr_correction.py`
   - OCR error pattern correction
   - Character substitution handling
   - Multi-token entity reconstruction

2. **Modify**: `scripts/extract_entities_fuzzy_known.py`
   - Lower similarity thresholds for OCR
   - Increase candidate extraction
   - Add Levenshtein distance

3. **Modify**: `scripts/extract_entities_hybrid.py`
   - Integrate OCR correction into pipeline
   - Context-aware validation
   - Improved confidence scoring

4. **Modify**: `config/ner_config.yaml`
   - Add OCR-specific thresholds
   - Configure correction parameters

5. **New**: `scripts/analyze_ocr_errors.py`
   - Analyze extraction results
   - Identify common error patterns
   - Generate correction rules
