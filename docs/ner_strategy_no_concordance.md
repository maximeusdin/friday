# Entity Extraction Strategy for Documents Without Concordance Index

## Overview

This document outlines a strategy for extracting entity mentions from documents that don't have a curated concordance index. We need to handle two document classes:

1. **OCR Documents**: Noisy text with OCR errors, character substitutions, spacing issues
2. **Clean Text Documents**: High-quality text without OCR artifacts

## Architecture Principles

1. **Leverage Existing Infrastructure**: Reuse `entity_mentions` table, normalization pipeline, and collision handling
2. **Confidence-Based Filtering**: All extractions get confidence scores; low-confidence go to review queue
3. **Multi-Tier Approach**: Combine rule-based patterns, statistical NER, and fuzzy matching
4. **Document-Class Aware**: Different strategies for OCR vs clean text
5. **Incremental Refinement**: Start conservative, expand based on validation

## Document Classification

### Detection Strategy

Add `text_quality` field to `documents.metadata` or `chunks`:

```sql
-- Option 1: Add to documents.metadata JSONB
-- Option 2: Add column to chunks table
ALTER TABLE chunks ADD COLUMN text_quality TEXT 
  CHECK (text_quality IN ('ocr', 'clean', 'unknown')) 
  DEFAULT 'unknown';
```

**Heuristics for classification:**
- **OCR indicators**: 
  - High frequency of common OCR errors (rn→m, cl→d, etc.)
  - Inconsistent spacing
  - Mixed case patterns (e.g., "Tlie" instead of "The")
  - Character confusion patterns (0/O, 1/l/I)
- **Clean text indicators**:
  - Consistent punctuation
  - Proper capitalization
  - Low character substitution rate
  - Well-formed sentences

## Extraction Pipeline

### Phase 1: Rule-Based Pattern Matching

**For both OCR and Clean Text:**

1. **Named Entity Patterns** (regex-based):
   - Person names: `[A-Z][a-z]+ [A-Z][a-z]+` (capitalized words)
   - Organizations: Patterns with "Inc.", "Corp.", "Ministry", "Department", etc.
   - Places: Capitalized words in geographic contexts
   - Dates: Already handled by `date_mentions` table

2. **Context Clues**:
   - Titles: "Mr.", "Dr.", "General", "Colonel"
   - Organization markers: "of", "for", "in" (e.g., "Ministry of Foreign Affairs")
   - Geographic markers: "in", "from", "to" + capitalized place name

**Implementation:**
- Create `scripts/extract_entities_pattern_based.py`
- Use existing `retrieval.normalization` for text normalization
- Output candidate entities with confidence scores

### Phase 2: Statistical NER (SpaCy/Transformers)

**For Clean Text (primary):**

Use pre-trained NER models:
- **SpaCy** `en_core_web_sm` or `en_core_web_lg` (fast, good for clean text)
- **spaCy transformers** `en_core_web_trf` (more accurate, slower)
- **Custom fine-tuned model** (if domain-specific training data available)

**For OCR Text (secondary, with lower confidence):**

- Use same models but with:
  - Pre-processing: OCR error correction heuristics
  - Post-processing: Fuzzy matching against known entities
  - Lower confidence thresholds

**Implementation:**
- Create `scripts/extract_entities_ner.py`
- Use `spacy` or `transformers` library
- Map NER labels to our entity types:
  - `PERSON` → `person`
  - `ORG` → `org`
  - `GPE`, `LOC` → `place`

### Phase 3: Fuzzy Matching Against Known Entities

**For both document types:**

1. Extract candidate surface forms from text
2. Normalize using `retrieval.normalization.normalize_alias()`
3. Match against existing `entity_aliases` table:
   - Exact match: confidence = 1.0
   - Fuzzy match (trigram): confidence = similarity_score
   - Partial match: confidence = 0.7-0.8

**Advantages:**
- Leverages existing entity knowledge from concordance-indexed documents
- Handles OCR errors via fuzzy matching
- Reuses existing collision resolution logic

### Phase 4: Hybrid Approach (Recommended)

Combine all three phases with weighted confidence:

```
final_confidence = (
    pattern_confidence * 0.2 +
    ner_confidence * 0.4 +
    fuzzy_match_confidence * 0.4
)
```

## OCR-Specific Handling

### Pre-Processing

1. **OCR Error Correction Dictionary**:
   ```python
   OCR_CORRECTIONS = {
       'rn': 'm',  # common OCR error
       'cl': 'd',
       'vv': 'w',
       # ... expand based on corpus analysis
   }
   ```

2. **Character Normalization**:
   - Handle common substitutions (0/O, 1/l/I)
   - Normalize spacing
   - Fix common word splits ("tlie" → "the")

3. **Confidence Adjustment**:
   - Reduce confidence for OCR text by 0.1-0.2
   - Mark `surface_quality = 'approx'` more liberally

### Fuzzy Matching Priority

For OCR documents, prioritize fuzzy matching:
- Use trigram similarity (pg_trgm)
- Lower similarity threshold (0.6 vs 0.7 for clean text)
- Allow more character substitutions

## Clean Text Handling

### Higher Precision Expected

1. **Stricter Pattern Matching**:
   - Require proper capitalization
   - Enforce punctuation rules
   - Validate against known patterns

2. **NER Model Priority**:
   - Use best available model (transformer-based)
   - Higher confidence thresholds
   - More aggressive extraction

3. **Surface Quality**:
   - Mark `surface_quality = 'exact'` when possible
   - Preserve original capitalization

## Integration with Existing System

### Reuse Existing Components

1. **Normalization**: `retrieval.normalization.normalize_alias()`
2. **Entity Resolution**: `retrieval.entity_resolver`
3. **Collision Handling**: Existing logic from `extract_entity_mentions.py`
4. **Database Schema**: `entity_mentions` table (already supports `method`, `confidence`, `surface_quality`)

### New Fields Needed

```sql
-- Add method types for NER extraction
ALTER TABLE entity_mentions 
  DROP CONSTRAINT IF EXISTS entity_mentions_method_check;

ALTER TABLE entity_mentions
  ADD CONSTRAINT entity_mentions_method_check
  CHECK (method IN (
    'alias_exact',      -- From concordance matching
    'alias_partial',    -- From concordance matching
    'alias_fuzzy',      -- From concordance matching
    'pattern_based',    -- Rule-based pattern matching
    'ner_spacy',        -- SpaCy NER model
    'ner_transformer',  -- Transformer-based NER
    'fuzzy_known',      -- Fuzzy match against known entities
    'hybrid',           -- Combined approach
    'human'             -- Manual review
  ));
```

### Review Queue Integration

Low-confidence extractions go to `entity_resolution_reviews`:
- Pattern matches with confidence < 0.7
- NER extractions with confidence < 0.8 (clean) or < 0.6 (OCR)
- Fuzzy matches with similarity < 0.7
- Any collision that can't be auto-resolved

## Implementation Plan

### Step 1: Pattern-Based Extractor (Week 1)

**File**: `scripts/extract_entities_pattern_based.py`

**Features**:
- Regex patterns for person/org/place names
- Context-aware extraction
- Confidence scoring based on pattern strength
- Integration with existing normalization

**Output**: Candidate entities with confidence scores

### Step 2: NER-Based Extractor (Week 2)

**File**: `scripts/extract_entities_ner.py`

**Features**:
- SpaCy integration
- Transformer model option (optional)
- Document-class aware processing
- Confidence mapping from model scores

**Dependencies**:
```bash
pip install spacy
python -m spacy download en_core_web_sm
# Optional: python -m spacy download en_core_web_trf
```

### Step 3: Fuzzy Matching Against Known Entities (Week 2)

**File**: `scripts/extract_entities_fuzzy_known.py`

**Features**:
- Extract candidate surface forms
- Match against `entity_aliases` using trigram similarity
- Reuse existing collision resolution
- OCR-aware similarity thresholds

### Step 4: Hybrid Extractor (Week 3)

**File**: `scripts/extract_entities_hybrid.py`

**Features**:
- Combines all three approaches
- Weighted confidence scoring
- Collision resolution across methods
- Review queue integration

### Step 5: Document Classification (Week 1, parallel)

**File**: `scripts/classify_text_quality.py`

**Features**:
- OCR vs clean text detection
- Updates `chunks.text_quality` or `documents.metadata`
- Batch processing support

## Quality Assurance

### Validation Metrics

1. **Precision**: % of extracted entities that are correct
2. **Recall**: % of actual entities that were extracted
3. **F1 Score**: Harmonic mean of precision and recall
4. **Confidence Calibration**: Do confidence scores correlate with accuracy?

### Review Workflow

1. **Automatic High-Confidence**: confidence ≥ 0.9 → auto-insert
2. **Medium-Confidence Review**: 0.7 ≤ confidence < 0.9 → review queue
3. **Low-Confidence Log**: confidence < 0.7 → log only, manual review

### Sampling Strategy

- Extract from sample documents first
- Validate against known entities (if available)
- Adjust thresholds based on validation results
- Expand to full corpus

## Configuration

### Per-Document-Class Settings

```python
# config/ner_config.yaml
ocr:
  ner_model: "en_core_web_sm"  # Faster model
  confidence_threshold: 0.6
  fuzzy_similarity_threshold: 0.6
  surface_quality_default: "approx"
  pattern_weight: 0.3
  ner_weight: 0.3
  fuzzy_weight: 0.4

clean:
  ner_model: "en_core_web_trf"  # Better model
  confidence_threshold: 0.8
  fuzzy_similarity_threshold: 0.7
  surface_quality_default: "exact"
  pattern_weight: 0.2
  ner_weight: 0.5
  fuzzy_weight: 0.3
```

## Example Usage

```bash
# Classify document quality first
python scripts/classify_text_quality.py --collection my_collection

# Extract entities (auto-detects OCR vs clean)
python scripts/extract_entities_hybrid.py \
  --collection my_collection \
  --enable-pattern \
  --enable-ner \
  --enable-fuzzy-known \
  --confidence-threshold 0.7 \
  --dry-run

# Review queue for manual adjudication
python scripts/review_entity_extractions.py \
  --collection my_collection \
  --min-confidence 0.7 \
  --max-confidence 0.9
```

## Future Enhancements

1. **Domain-Specific Training**: Fine-tune NER models on historical documents
2. **Active Learning**: Use review feedback to improve models
3. **Cross-Document Linking**: Link entities across documents using embeddings
4. **Temporal Disambiguation**: Use date context to resolve ambiguous entities
5. **Multilingual Support**: Extend to Russian, German, etc.

## Comparison with Concordance-Based Approach

| Aspect | Concordance-Based | NER-Based (No Concordance) |
|--------|------------------|---------------------------|
| **Precision** | Very High (curated) | Medium-High (model-dependent) |
| **Recall** | Limited to index | Higher (extracts all entities) |
| **Speed** | Fast (exact matching) | Slower (model inference) |
| **Coverage** | Only indexed entities | All entities in text |
| **Review Needed** | Low (high confidence) | Medium (confidence-based) |
| **OCR Handling** | Fuzzy matching helps | Requires pre-processing |

## Conclusion

The hybrid approach combining pattern matching, statistical NER, and fuzzy matching against known entities provides the best balance for documents without concordance indexes. OCR documents require more aggressive fuzzy matching and lower confidence thresholds, while clean text can leverage higher-precision NER models.

The key is to:
1. Start conservative with high confidence thresholds
2. Use review queue for medium-confidence extractions
3. Iteratively refine based on validation results
4. Leverage existing entity knowledge via fuzzy matching
