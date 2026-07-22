# Clean Text Entity Extraction - Current State

## Overview

You have a complete **3-tier entity extraction system** for documents without concordance indexes, designed to handle both OCR and clean text documents.

## Implemented Components

### 1. **Pattern-Based Extraction** (`scripts/extract_entities_pattern_based.py`)
- **Method**: Rule-based regex patterns
- **Extracts**: Person names, organizations, places
- **Features**:
  - Person name patterns (full names, titles, last-first format)
  - Organization patterns (acronyms, company markers, "The X" patterns)
  - Place patterns (geographic context, city-state format)
  - Filters out PDF metadata and common document words
- **Confidence**: Base confidence scores (0.7-0.9) adjusted by pattern type
- **Usage**:
  ```bash
  python scripts/extract_entities_pattern_based.py \
    --collection my_collection \
    --entity-type person --entity-type org \
    --confidence-threshold 0.7 \
    --dry-run
  ```

### 2. **NER-Based Extraction** (`scripts/extract_entities_ner.py`)
- **Method**: Statistical NER models (SpaCy)
- **Models**: 
  - `en_core_web_sm` (faster, for OCR)
  - `en_core_web_trf` (better, for clean text)
- **Extracts**: PERSON, ORG, GPE/LOC/FAC (mapped to person/org/place)
- **Features**:
  - Confidence scoring adjusted for text quality
  - Length-based confidence adjustments
  - Capitalization checks
- **Usage**:
  ```bash
  python scripts/extract_entities_ner.py \
    --collection my_collection \
    --model en_core_web_trf \
    --confidence-threshold 0.8 \
    --dry-run
  ```

### 3. **Fuzzy Matching** (`scripts/extract_entities_fuzzy_known.py`)
- **Method**: Trigram similarity (pg_trgm) against known entities
- **Extracts**: Matches text against existing `entity_aliases` table
- **Features**:
  - Adjusts similarity threshold based on text quality
  - OCR: lower threshold (0.6), clean: higher threshold (0.7)
  - Handles OCR errors and variations
- **Usage**:
  ```bash
  python scripts/extract_entities_fuzzy_known.py \
    --collection my_collection \
    --similarity-threshold 0.7 \
    --confidence-threshold 0.7 \
    --dry-run
  ```

### 4. **Hybrid Extractor** (`scripts/extract_entities_hybrid.py`) ⭐ **Recommended**
- **Method**: Combines all three methods with weighted confidence
- **Features**:
  - Weighted combination based on text quality
  - OCR: pattern (30%), NER (30%), fuzzy (40%)
  - Clean: pattern (20%), NER (50%), fuzzy (30%)
  - Resolves collisions by position
  - Quality-specific confidence thresholds
- **Configuration**: `config/ner_config.yaml`
- **Usage**:
  ```bash
  python scripts/extract_entities_hybrid.py \
    --collection my_collection \
    --enable-pattern --enable-ner --enable-fuzzy-known \
    --config config/ner_config.yaml \
    --dry-run
  ```

### 5. **Text Quality Classification** (`scripts/classify_text_quality.py`)
- **Purpose**: Classify documents as OCR vs clean text
- **Sets**: `chunks.text_quality` column ('ocr', 'clean', 'unknown')
- **Usage**:
  ```bash
  python scripts/classify_text_quality.py \
    --collection my_collection \
    --dry-run
  ```

### 6. **Review Queue** (`scripts/review_entity_extractions.py`)
- **Purpose**: Review medium-confidence extractions
- **Features**: Interactive review, confidence-based filtering
- **Usage**:
  ```bash
  python scripts/review_entity_extractions.py \
    --collection my_collection \
    --min-confidence 0.7 \
    --max-confidence 0.9 \
    --interactive
  ```

## Database Schema

### Migration: `migrations/0026_ner_methods_and_text_quality.sql`
- Adds `text_quality` column to `chunks` table
- Updates `entity_mentions.method` constraint to include:
  - `pattern_based`
  - `ner_spacy`
  - `ner_transformer`
  - `fuzzy_known`
  - `hybrid`
  - `human`

## Configuration

### `config/ner_config.yaml`
- **OCR settings**: Lower thresholds, faster models, more fuzzy matching
- **Clean text settings**: Higher thresholds, better models, less fuzzy matching
- **Pattern configuration**: Person/org/place pattern settings
- **Review queue**: Auto-insert vs review thresholds

## Workflow

### Typical Workflow for Clean Text Documents

1. **Classify text quality**:
   ```bash
   python scripts/classify_text_quality.py --collection my_collection
   ```

2. **Run hybrid extraction** (recommended):
   ```bash
   python scripts/extract_entities_hybrid.py \
     --collection my_collection \
     --enable-pattern --enable-ner --enable-fuzzy-known \
     --dry-run  # Test first
   ```

3. **Review results**:
   ```bash
   python scripts/review_entity_extractions.py \
     --collection my_collection \
     --min-confidence 0.7 \
     --max-confidence 0.9
   ```

4. **Run without dry-run**:
   ```bash
   python scripts/extract_entities_hybrid.py \
     --collection my_collection \
     --enable-pattern --enable-ner --enable-fuzzy-known
   ```

## Integration with Concordance-Based Extraction

These scripts are **separate** from `extract_entity_mentions.py`:
- **Concordance-based** (`extract_entity_mentions.py`): Matches against known aliases from concordance indexes
- **NER-based** (these scripts): Discovers new entities from clean text

**Both can run on the same documents** - they complement each other:
- Concordance extraction: High precision, finds known entities
- NER extraction: Discovers new entities not in concordance

## Current Status

✅ **Fully Implemented**:
- Pattern-based extraction
- NER-based extraction (SpaCy)
- Fuzzy matching against known entities
- Hybrid combination
- Text quality classification
- Review queue system
- Configuration system

## Next Steps / Potential Improvements

1. **Integration**: Combine NER-discovered entities with concordance-based extraction
2. **Deduplication**: Merge entities discovered by both methods
3. **Performance**: Optimize fuzzy matching queries
4. **Transformer models**: Add support for transformer-based NER (e.g., spaCy transformers)
5. **Domain-specific models**: Train/fine-tune models on historical documents
6. **Entity linking**: Link discovered entities to existing entities in database

## Key Differences from Concordance Extraction

| Feature | Concordance-Based | NER-Based |
|---------|------------------|-----------|
| **Input** | Known aliases from concordance | Raw text |
| **Method** | Exact/partial/fuzzy matching | Statistical models + patterns |
| **Precision** | Very high (known entities) | Medium-high (model-dependent) |
| **Recall** | Limited to known entities | Discovers new entities |
| **Use Case** | Documents with concordance indexes | Documents without concordance |
| **Collision Resolution** | Complex pipeline (citations, dominance) | Simple confidence-based |

## Example: Running on Clean Text Collection

```bash
# 1. Classify text quality
python scripts/classify_text_quality.py --collection clean_text_collection

# 2. Extract entities (hybrid approach)
python scripts/extract_entities_hybrid.py \
  --collection clean_text_collection \
  --enable-pattern \
  --enable-ner \
  --enable-fuzzy-known \
  --limit 100 \
  --dry-run

# 3. Review medium-confidence extractions
python scripts/review_entity_extractions.py \
  --collection clean_text_collection \
  --min-confidence 0.7 \
  --max-confidence 0.9

# 4. Run for real
python scripts/extract_entities_hybrid.py \
  --collection clean_text_collection \
  --enable-pattern \
  --enable-ner \
  --enable-fuzzy-known
```
