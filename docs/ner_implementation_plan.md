# NER Implementation Plan: Documents Without Concordance Index

## Overview

This plan outlines the step-by-step implementation of entity extraction for documents without concordance indexes, handling both OCR and clean text documents.

**Timeline**: 3-4 weeks  
**Components**: 5 main scripts + database migrations + testing infrastructure

---

## Phase 0: Prerequisites & Setup (Days 1-2)

### Task 0.1: Database Schema Updates
**File**: `migrations/0026_ner_methods_and_text_quality.sql`

**Deliverables**:
- Add `text_quality` column to `chunks` table
- Update `entity_mentions.method` constraint to include new methods
- Add indexes for performance

**SQL**:
```sql
BEGIN;

-- Add text_quality to chunks
ALTER TABLE chunks 
  ADD COLUMN IF NOT EXISTS text_quality TEXT 
  CHECK (text_quality IN ('ocr', 'clean', 'unknown')) 
  DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS idx_chunks_text_quality 
  ON chunks(text_quality);

-- Update entity_mentions method constraint
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

COMMIT;
```

**Testing**:
- Verify constraint accepts new method values
- Test text_quality column updates

---

### Task 0.2: Dependencies Installation
**File**: `requirements.txt` (update)

**Deliverables**:
- Add spacy and related dependencies
- Document installation steps

**Dependencies to add**:
```txt
spacy>=3.7.0
# Optional: transformers>=4.30.0  # For transformer-based NER
```

**Installation script**: `scripts/setup_ner.sh`
```bash
#!/bin/bash
pip install spacy>=3.7.0
python -m spacy download en_core_web_sm
# Optional: python -m spacy download en_core_web_trf
```

**Testing**:
- Verify spacy imports work
- Test model loading

---

### Task 0.3: Configuration File
**File**: `config/ner_config.yaml`

**Deliverables**:
- YAML configuration for OCR vs clean text settings
- Per-method configuration

**Content**:
```yaml
ocr:
  ner_model: "en_core_web_sm"
  confidence_threshold: 0.6
  fuzzy_similarity_threshold: 0.6
  surface_quality_default: "approx"
  pattern_weight: 0.3
  ner_weight: 0.3
  fuzzy_weight: 0.4
  ocr_error_correction: true

clean:
  ner_model: "en_core_web_trf"  # Better model
  confidence_threshold: 0.8
  fuzzy_similarity_threshold: 0.7
  surface_quality_default: "exact"
  pattern_weight: 0.2
  ner_weight: 0.5
  fuzzy_weight: 0.3
  ocr_error_correction: false

patterns:
  person:
    min_tokens: 2
    max_tokens: 5
    require_title: false
    titles: ["Mr.", "Mrs.", "Ms.", "Dr.", "General", "Colonel", "Captain"]
  
  organization:
    markers: ["Inc.", "Corp.", "Ltd.", "Ministry", "Department", "Bureau", "Agency"]
    require_marker: false
  
  place:
    geographic_markers: ["in", "from", "to", "near", "at"]
    require_marker: false

review_queue:
  auto_insert_threshold: 0.9
  review_threshold_min: 0.7
  review_threshold_max: 0.9
  log_only_threshold: 0.7
```

**Testing**:
- Verify YAML parsing works
- Test configuration access

---

## Phase 1: Document Classification (Days 3-5)

### Task 1.1: Text Quality Classifier
**File**: `scripts/classify_text_quality.py`

**Features**:
- Detect OCR vs clean text
- Update `chunks.text_quality` or `documents.metadata`
- Batch processing support
- Dry-run mode

**Heuristics to implement**:
1. **OCR Error Detection**:
   - Common OCR errors: rn→m, cl→d, vv→w
   - Character confusion: 0/O, 1/l/I
   - Inconsistent spacing patterns
   - Mixed case anomalies ("Tlie" instead of "The")

2. **Clean Text Indicators**:
   - Consistent punctuation
   - Proper capitalization
   - Well-formed sentences
   - Low character substitution rate

**Algorithm**:
```python
def classify_text_quality(text: str) -> str:
    ocr_score = 0
    clean_score = 0
    
    # Check for OCR errors
    ocr_errors = count_ocr_errors(text)
    ocr_score += ocr_errors * 0.1
    
    # Check for inconsistent spacing
    spacing_issues = detect_spacing_issues(text)
    ocr_score += spacing_issues * 0.05
    
    # Check for proper capitalization
    capitalization_score = check_capitalization(text)
    clean_score += capitalization_score * 0.3
    
    # Check punctuation consistency
    punctuation_score = check_punctuation(text)
    clean_score += punctuation_score * 0.2
    
    # Determine classification
    if ocr_score > 0.3:
        return 'ocr'
    elif clean_score > 0.6:
        return 'clean'
    else:
        return 'unknown'
```

**Command-line interface**:
```bash
python scripts/classify_text_quality.py \
  --collection my_collection \
  --update-chunks \
  --dry-run
```

**Testing**:
- Test on known OCR documents
- Test on known clean text documents
- Verify classification accuracy

**Dependencies**: None (pure Python)

---

## Phase 2: Pattern-Based Extraction (Days 6-8)

### Task 2.1: Pattern-Based Entity Extractor
**File**: `scripts/extract_entities_pattern_based.py`

**Features**:
- Regex patterns for person/org/place names
- Context-aware extraction
- Confidence scoring
- Integration with normalization

**Patterns to implement**:

1. **Person Names**:
   ```python
   PERSON_PATTERNS = [
       r'\b([A-Z][a-z]+ [A-Z][a-z]+)',  # "John Smith"
       r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|General|Colonel|Captain)\s+([A-Z][a-z]+ [A-Z][a-z]+)',  # With title
       r'\b([A-Z][a-z]+, [A-Z][a-z]+)',  # "Smith, John"
   ]
   ```

2. **Organizations**:
   ```python
   ORG_PATTERNS = [
       r'\b([A-Z][a-z]+ (?:Ministry|Department|Bureau|Agency|Office))',
       r'\b([A-Z][a-z]+ (?:Inc\.|Corp\.|Ltd\.|LLC))',
       r'\b((?:The )?[A-Z][a-z]+ (?:of|for) [A-Z][a-z]+)',
   ]
   ```

3. **Places**:
   ```python
   PLACE_PATTERNS = [
       r'\b(in|from|to|near|at)\s+([A-Z][a-z]+(?:, [A-Z][a-z]+)?)',  # Geographic context
       r'\b([A-Z][a-z]+(?:, [A-Z]{2})?)',  # City, State format
   ]
   ```

**Confidence Scoring**:
- Pattern strength (stronger patterns = higher confidence)
- Context presence (with context = higher confidence)
- Length validation (reasonable length = higher confidence)

**Output format**:
```python
{
    'entity_type': 'person',
    'surface': 'John Smith',
    'start_char': 100,
    'end_char': 110,
    'confidence': 0.75,
    'pattern_used': 'person_with_title',
    'context': 'Dr. John Smith'
}
```

**Command-line interface**:
```bash
python scripts/extract_entities_pattern_based.py \
  --collection my_collection \
  --entity-type person \
  --confidence-threshold 0.7 \
  --dry-run
```

**Testing**:
- Test patterns on sample text
- Verify confidence scores
- Test edge cases (false positives)

**Dependencies**: 
- `retrieval.normalization`

---

## Phase 3: NER-Based Extraction (Days 9-12)

### Task 3.1: SpaCy NER Extractor
**File**: `scripts/extract_entities_ner.py`

**Features**:
- SpaCy integration
- Document-class aware processing
- Confidence mapping
- Batch processing

**Implementation**:

```python
import spacy
from typing import List, Dict

class NERExtractor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)
    
    def extract(self, text: str, text_quality: str = "unknown") -> List[Dict]:
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_type = self.map_ner_label(ent.label_)
            if entity_type:
                confidence = self.calculate_confidence(ent, text_quality)
                entities.append({
                    'entity_type': entity_type,
                    'surface': ent.text,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'confidence': confidence,
                    'ner_label': ent.label_,
                    'ner_confidence': ent.score if hasattr(ent, 'score') else 1.0
                })
        
        return entities
    
    def map_ner_label(self, label: str) -> Optional[str]:
        mapping = {
            'PERSON': 'person',
            'ORG': 'org',
            'GPE': 'place',  # Geopolitical entity
            'LOC': 'place',  # Location
        }
        return mapping.get(label)
    
    def calculate_confidence(self, ent, text_quality: str) -> float:
        base_confidence = 0.8
        
        # Adjust for text quality
        if text_quality == 'ocr':
            base_confidence -= 0.15
        elif text_quality == 'clean':
            base_confidence += 0.1
        
        # Adjust for entity length
        if len(ent.text.split()) > 3:
            base_confidence += 0.05
        
        return min(1.0, max(0.0, base_confidence))
```

**Command-line interface**:
```bash
python scripts/extract_entities_ner.py \
  --collection my_collection \
  --model en_core_web_sm \
  --confidence-threshold 0.7 \
  --dry-run
```

**Testing**:
- Test on sample documents
- Verify entity type mapping
- Test confidence calibration
- Performance testing (speed)

**Dependencies**:
- `spacy`
- `retrieval.normalization`

---

### Task 3.2: OCR Error Correction (Optional Enhancement)
**File**: `scripts/ocr_correction.py`

**Features**:
- Common OCR error corrections
- Pre-processing for OCR text
- Configurable correction dictionary

**Implementation**:
```python
OCR_CORRECTIONS = {
    'rn': 'm',  # common OCR error
    'cl': 'd',
    'vv': 'w',
    'li': 'h',
    'rn': 'rn',  # Keep if intentional
}

def correct_ocr_errors(text: str) -> str:
    # Apply corrections based on context
    # Be conservative - only fix obvious errors
    pass
```

**Testing**:
- Test on known OCR errors
- Verify doesn't break valid text

---

## Phase 4: Fuzzy Matching Against Known Entities (Days 13-15)

### Task 4.1: Fuzzy Known Entity Matcher
**File**: `scripts/extract_entities_fuzzy_known.py`

**Features**:
- Extract candidate surface forms
- Match against `entity_aliases` using trigram similarity
- Reuse existing collision resolution
- OCR-aware similarity thresholds

**Implementation**:

```python
from retrieval.entity_resolver import normalize_alias
from retrieval.ops import get_conn

def extract_candidate_surfaces(text: str) -> List[str]:
    """Extract potential entity surface forms from text"""
    # Use pattern matching or NER to get candidates
    # Return list of surface strings
    pass

def fuzzy_match_against_known(
    surface: str,
    conn,
    text_quality: str = "unknown",
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """Match surface against known entities using trigram similarity"""
    
    surface_norm = normalize_alias(surface)
    
    # Query using pg_trgm similarity
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            ea.entity_id,
            ea.alias,
            ea.alias_norm,
            e.entity_type,
            similarity(ea.alias_norm, %s) as sim_score
        FROM entity_aliases ea
        JOIN entities e ON ea.entity_id = e.id
        WHERE ea.alias_norm % %s  -- Trigram similarity operator
        AND similarity(ea.alias_norm, %s) >= %s
        ORDER BY sim_score DESC
    """, (surface_norm, surface_norm, surface_norm, similarity_threshold))
    
    matches = []
    for row in cur.fetchall():
        matches.append({
            'entity_id': row[0],
            'alias': row[1],
            'alias_norm': row[2],
            'entity_type': row[3],
            'similarity': row[4],
            'confidence': row[4]  # Use similarity as confidence
        })
    
    return matches
```

**Command-line interface**:
```bash
python scripts/extract_entities_fuzzy_known.py \
  --collection my_collection \
  --similarity-threshold 0.7 \
  --dry-run
```

**Testing**:
- Test fuzzy matching accuracy
- Test similarity thresholds
- Test collision resolution

**Dependencies**:
- `retrieval.entity_resolver`
- `retrieval.ops`
- PostgreSQL `pg_trgm` extension

---

## Phase 5: Hybrid Extractor (Days 16-20)

### Task 5.1: Hybrid Entity Extractor
**File**: `scripts/extract_entities_hybrid.py`

**Features**:
- Combines pattern, NER, and fuzzy matching
- Weighted confidence scoring
- Collision resolution across methods
- Review queue integration

**Implementation**:

```python
from scripts.extract_entities_pattern_based import extract_pattern_based
from scripts.extract_entities_ner import NERExtractor
from scripts.extract_entities_fuzzy_known import fuzzy_match_against_known
import yaml

class HybridExtractor:
    def __init__(self, config_path: str = "config/ner_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.ner_extractor = NERExtractor()
    
    def extract(self, text: str, text_quality: str, conn) -> List[Dict]:
        all_candidates = []
        
        # Phase 1: Pattern-based
        if self.config.get('enable_pattern', True):
            pattern_results = extract_pattern_based(text)
            for result in pattern_results:
                result['method'] = 'pattern_based'
                result['weight'] = self.config[text_quality]['pattern_weight']
            all_candidates.extend(pattern_results)
        
        # Phase 2: NER-based
        if self.config.get('enable_ner', True):
            ner_results = self.ner_extractor.extract(text, text_quality)
            for result in ner_results:
                result['method'] = 'ner_spacy'
                result['weight'] = self.config[text_quality]['ner_weight']
            all_candidates.extend(ner_results)
        
        # Phase 3: Fuzzy matching against known
        if self.config.get('enable_fuzzy_known', True):
            fuzzy_results = fuzzy_match_against_known(text, conn, text_quality)
            for result in fuzzy_results:
                result['method'] = 'fuzzy_known'
                result['weight'] = self.config[text_quality]['fuzzy_weight']
            all_candidates.extend(fuzzy_results)
        
        # Combine and resolve collisions
        resolved = self.resolve_collisions(all_candidates, text_quality)
        
        return resolved
    
    def resolve_collisions(self, candidates: List[Dict], text_quality: str) -> List[Dict]:
        """Resolve collisions using weighted confidence"""
        # Group by position
        # Apply collision resolution rules
        # Return final list
        pass
    
    def calculate_final_confidence(self, candidate: Dict, text_quality: str) -> float:
        """Calculate weighted confidence score"""
        base_confidence = candidate['confidence']
        weight = candidate['weight']
        
        # Weighted combination if multiple methods agree
        # ...
        
        return final_confidence
```

**Command-line interface**:
```bash
python scripts/extract_entities_hybrid.py \
  --collection my_collection \
  --enable-pattern \
  --enable-ner \
  --enable-fuzzy-known \
  --confidence-threshold 0.7 \
  --dry-run
```

**Testing**:
- Test hybrid combination
- Verify weighted confidence
- Test collision resolution
- End-to-end testing

**Dependencies**:
- All previous extractors
- `yaml` for config

---

## Phase 6: Review Queue Integration (Days 21-22)

### Task 6.1: Review Queue Manager
**File**: `scripts/review_entity_extractions.py`

**Features**:
- Query review queue
- Display candidates for review
- Accept/reject/reassign entities
- Update `entity_mentions` based on decisions

**Implementation**:

```python
def get_review_queue(conn, min_confidence: float = 0.7, max_confidence: float = 0.9):
    """Get entities in review queue"""
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            em.id,
            em.surface,
            em.confidence,
            em.method,
            e.canonical_name,
            e.entity_type,
            c.text,
            cm.document_id,
            d.source_name
        FROM entity_mentions em
        JOIN entities e ON em.entity_id = e.id
        JOIN chunks c ON em.chunk_id = c.id
        JOIN chunk_metadata cm ON c.id = cm.chunk_id
        JOIN documents d ON cm.document_id = d.id
        WHERE em.confidence >= %s AND em.confidence < %s
        ORDER BY em.confidence DESC
    """, (min_confidence, max_confidence))
    
    return cur.fetchall()

def review_entity(mention_id: int, decision: str, reviewer: str, notes: str):
    """Record review decision"""
    # Update entity_resolution_reviews table
    # Optionally update entity_mentions
    pass
```

**Command-line interface**:
```bash
python scripts/review_entity_extractions.py \
  --collection my_collection \
  --min-confidence 0.7 \
  --max-confidence 0.9 \
  --interactive
```

**Testing**:
- Test review queue queries
- Test decision recording
- Test batch review

**Dependencies**:
- Database access

---

## Phase 7: Testing & Validation (Days 23-25)

### Task 7.1: Unit Tests
**Files**: `tests/test_pattern_extraction.py`, `tests/test_ner_extraction.py`, etc.

**Coverage**:
- Pattern matching accuracy
- NER extraction accuracy
- Fuzzy matching accuracy
- Confidence scoring
- Collision resolution

### Task 7.2: Integration Tests
**File**: `tests/test_hybrid_extraction.py`

**Coverage**:
- End-to-end extraction
- Review queue workflow
- Database integration

### Task 7.3: Validation Script
**File**: `scripts/validate_ner_extraction.py`

**Features**:
- Compare extracted entities against gold standard (if available)
- Calculate precision/recall/F1
- Confidence calibration analysis
- Generate validation report

**Metrics**:
- Precision: % of extracted entities that are correct
- Recall: % of actual entities that were extracted
- F1 Score: Harmonic mean
- Confidence calibration: Do scores correlate with accuracy?

---

## Phase 8: Documentation & Deployment (Days 26-28)

### Task 8.1: User Documentation
**File**: `docs/ner_extraction_guide.md`

**Content**:
- Usage examples
- Configuration guide
- Troubleshooting
- Best practices

### Task 8.2: API Documentation
**File**: Docstrings in all scripts

### Task 8.3: Deployment Checklist
- [ ] All migrations applied
- [ ] Dependencies installed
- [ ] Configuration files in place
- [ ] Tests passing
- [ ] Sample run successful

---

## Implementation Timeline

```
Week 1:
├── Days 1-2: Phase 0 (Setup)
├── Days 3-5: Phase 1 (Classification)
└── Days 6-8: Phase 2 (Pattern-based)

Week 2:
├── Days 9-12: Phase 3 (NER-based)
└── Days 13-15: Phase 4 (Fuzzy matching)

Week 3:
├── Days 16-20: Phase 5 (Hybrid)
└── Days 21-22: Phase 6 (Review queue)

Week 4:
├── Days 23-25: Phase 7 (Testing)
└── Days 26-28: Phase 8 (Documentation)
```

---

## Risk Mitigation

### Risk 1: NER Model Performance
**Mitigation**: 
- Start with SpaCy small model (fast)
- Upgrade to transformer model if needed
- Allow fallback to pattern-based

### Risk 2: OCR Classification Accuracy
**Mitigation**:
- Conservative classification (default to 'unknown')
- Allow manual override
- Iterative refinement based on feedback

### Risk 3: Performance Issues
**Mitigation**:
- Batch processing
- Database indexing
- Optional parallel processing

### Risk 4: Low Precision/Recall
**Mitigation**:
- Conservative confidence thresholds
- Review queue for validation
- Iterative improvement based on metrics

---

## Success Criteria

1. **Functionality**:
   - [ ] All extractors working
   - [ ] Hybrid combination working
   - [ ] Review queue functional
   - [ ] Classification working

2. **Quality**:
   - [ ] Precision > 0.7 (clean text)
   - [ ] Precision > 0.6 (OCR text)
   - [ ] Recall > 0.5 (both)
   - [ ] Confidence scores calibrated

3. **Performance**:
   - [ ] Process 1000 chunks/minute (pattern-based)
   - [ ] Process 100 chunks/minute (NER-based)
   - [ ] Acceptable memory usage

4. **Integration**:
   - [ ] Works with existing entity system
   - [ ] Review queue integrated
   - [ ] Database migrations applied

---

## Next Steps

1. **Review this plan** with team
2. **Set up development environment** (Phase 0)
3. **Start with classification** (Phase 1) - lowest risk
4. **Iterate based on feedback** from each phase
5. **Validate early and often** - don't wait until end

---

## Questions to Resolve

1. **Gold standard data**: Do we have labeled data for validation?
2. **Performance requirements**: What's acceptable processing speed?
3. **Review capacity**: How many entities can be manually reviewed?
4. **Model selection**: Start with SpaCy small or transformer model?
5. **Deployment**: When should this be production-ready?
