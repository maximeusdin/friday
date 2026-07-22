# NER Implementation Checklist

Quick reference checklist for tracking implementation progress.

## Phase 0: Prerequisites & Setup
- [ ] Create migration `0026_ner_methods_and_text_quality.sql`
- [ ] Add `text_quality` column to `chunks` table
- [ ] Update `entity_mentions.method` constraint
- [ ] Add indexes for performance
- [ ] Update `requirements.txt` with spacy
- [ ] Create `scripts/setup_ner.sh` installation script
- [ ] Create `config/ner_config.yaml` configuration file
- [ ] Test database migrations
- [ ] Test dependency installation

## Phase 1: Document Classification
- [ ] Create `scripts/classify_text_quality.py`
- [ ] Implement OCR error detection heuristics
- [ ] Implement clean text indicators
- [ ] Add batch processing support
- [ ] Add dry-run mode
- [ ] Test on known OCR documents
- [ ] Test on known clean text documents
- [ ] Verify classification accuracy

## Phase 2: Pattern-Based Extraction
- [ ] Create `scripts/extract_entities_pattern_based.py`
- [ ] Implement person name patterns
- [ ] Implement organization patterns
- [ ] Implement place name patterns
- [ ] Add context-aware extraction
- [ ] Implement confidence scoring
- [ ] Integrate with normalization
- [ ] Add command-line interface
- [ ] Test patterns on sample text
- [ ] Test edge cases

## Phase 3: NER-Based Extraction
- [ ] Create `scripts/extract_entities_ner.py`
- [ ] Implement SpaCy integration
- [ ] Implement entity type mapping (PERSON→person, etc.)
- [ ] Implement confidence calculation
- [ ] Add document-class aware processing
- [ ] Add batch processing
- [ ] Test on sample documents
- [ ] Test confidence calibration
- [ ] Performance testing
- [ ] (Optional) Create `scripts/ocr_correction.py`

## Phase 4: Fuzzy Matching Against Known Entities
- [ ] Create `scripts/extract_entities_fuzzy_known.py`
- [ ] Implement candidate surface extraction
- [ ] Implement trigram similarity matching
- [ ] Add OCR-aware similarity thresholds
- [ ] Integrate with collision resolution
- [ ] Add command-line interface
- [ ] Test fuzzy matching accuracy
- [ ] Test similarity thresholds

## Phase 5: Hybrid Extractor
- [ ] Create `scripts/extract_entities_hybrid.py`
- [ ] Integrate pattern-based extractor
- [ ] Integrate NER extractor
- [ ] Integrate fuzzy matcher
- [ ] Implement weighted confidence scoring
- [ ] Implement collision resolution across methods
- [ ] Add review queue integration
- [ ] Add command-line interface
- [ ] Test hybrid combination
- [ ] End-to-end testing

## Phase 6: Review Queue Integration
- [ ] Create `scripts/review_entity_extractions.py`
- [ ] Implement review queue queries
- [ ] Implement decision recording
- [ ] Add interactive review mode
- [ ] Add batch review support
- [ ] Test review queue workflow

## Phase 7: Testing & Validation
- [ ] Create `tests/test_pattern_extraction.py`
- [ ] Create `tests/test_ner_extraction.py`
- [ ] Create `tests/test_fuzzy_matching.py`
- [ ] Create `tests/test_hybrid_extraction.py`
- [ ] Create `scripts/validate_ner_extraction.py`
- [ ] Implement precision/recall calculation
- [ ] Implement confidence calibration analysis
- [ ] Generate validation reports
- [ ] Run full test suite

## Phase 8: Documentation & Deployment
- [ ] Create `docs/ner_extraction_guide.md`
- [ ] Add docstrings to all scripts
- [ ] Create usage examples
- [ ] Create troubleshooting guide
- [ ] Create deployment checklist
- [ ] Review all documentation

## Success Criteria
- [ ] Precision > 0.7 (clean text)
- [ ] Precision > 0.6 (OCR text)
- [ ] Recall > 0.5 (both)
- [ ] Confidence scores calibrated
- [ ] Performance targets met
- [ ] Integration with existing system working
- [ ] Review queue functional

## Notes
- Start with Phase 0 and Phase 1 (lowest risk)
- Test each phase before moving to next
- Iterate based on validation results
- Keep confidence thresholds conservative initially
