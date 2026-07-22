# Rule-Based vs ML-Based NER: Comparison

## Current Implementation

We have **three extraction methods**:

1. **Pattern-Based (Rule-Based)**: Regex patterns matching capitalized words, acronyms, etc.
2. **SpaCy NER (ML-Based)**: Statistical models trained on large corpora
3. **Fuzzy Matching**: Match against known entities using trigram similarity

## The Problem with Rule-Based Only

### OCR Documents
- **Issue**: Random acronyms from PDF encoding (`DSN`, `FNNS`, `QTL`, `TVZM`, `ITF`, `QDU`, `RCN`, `YII`)
- **Why**: Pattern matcher sees any 3-6 uppercase letters and assumes it's an organization
- **Solution**: ML-based NER models understand context and are trained to distinguish real entities from encoding artifacts

### Clean Text Documents  
- **Issue**: Common words extracted as entities (`SECOND`, `OFFICE`, `THOMAS`, `KARL`, `MUNDT`, `JOHN`)
- **Why**: Pattern matcher sees capitalized words and assumes entities
- **Solution**: NER models understand that "SECOND" in "SECOND SESSION" is not an entity, and "THOMAS" alone might be a person name, not an organization

## Why SpaCy NER is Better

### 1. Context Understanding
- **Rule-based**: Sees "THOMAS" → matches pattern → assumes organization
- **NER**: Sees "THOMAS" in context → understands it's likely a person name (PERSON label)

### 2. Trained on Real Data
- SpaCy models are trained on millions of sentences
- They learn patterns like:
  - "John Smith" → PERSON (not two separate entities)
  - "House of Representatives" → ORG (not "House" as separate entity)
  - "New York" → GPE (place, not person)

### 3. Handles Variations
- OCR errors: "Akhmerov" vs "Akhmeroff" → NER can still recognize as person
- Different formats: "Smith, John" vs "John Smith" → both recognized

### 4. Filters Noise
- PDF encoding strings: NER won't recognize `DSN`, `FNNS` as entities
- Common words: NER understands "SECOND" in "SECOND SESSION" is not an entity

## When to Use Each Method

### Pattern-Based (Rule-Based)
**Use when:**
- You need very specific patterns (e.g., "Ministry of X")
- You want deterministic, explainable matches
- You're extracting domain-specific entities not in NER training data

**Limitations:**
- Many false positives (matches patterns that aren't entities)
- Can't distinguish context (e.g., "House" as building vs organization)
- Struggles with OCR noise

### SpaCy NER (ML-Based)
**Use when:**
- You want fewer false positives
- You're dealing with OCR documents (better noise filtering)
- You need context-aware extraction
- Standard entity types (person, org, place) are sufficient

**Limitations:**
- Requires spacy installation and model download
- Slower than pattern matching
- May miss domain-specific entities not in training data
- Less explainable (black box model)

### Hybrid Approach (Recommended)
**Best of both worlds:**
- Use NER for standard entities (person, org, place)
- Use pattern matching for domain-specific patterns
- Use fuzzy matching to leverage existing entity knowledge
- Combine with weighted confidence scores

## Test Results Comparison

### OCR Document (Silvermaster)
**Pattern-Based**: 8 entities found
- `DSN`, `FNNS`, `QTL`, `TVZM`, `ITF`, `QDU`, `RCN`, `YII`
- **Problem**: These are PDF encoding artifacts, not real entities

**Expected with NER**: 
- Should filter out most PDF encoding strings
- Will only extract entities that look like real names/places/organizations
- Much fewer false positives

### Clean Text Document
**Pattern-Based**: 78 entities found
- `SECOND`, `OFFICE`, `THOMAS`, `KARL`, `MUNDT`, `JOHN` (as organizations)
- `Public Law`, `New Jersey`, `South Dakota` (as persons - wrong type)
- **Problem**: Common words and wrong entity type assignments

**Expected with NER**:
- `THOMAS`, `KARL`, `MUNDT`, `JOHN` → Should be recognized as PERSON, not ORG
- `New Jersey`, `South Dakota` → Should be recognized as GPE (place), not PERSON
- `SECOND`, `OFFICE` → Should be filtered out (not entities)
- Much more accurate entity type assignment

## Recommendation

**For production use, always use the hybrid approach** (`extract_entities_hybrid.py`):

1. **NER handles standard entities** with better accuracy
2. **Pattern matching catches domain-specific entities** NER might miss
3. **Fuzzy matching leverages existing knowledge** from concordance-indexed documents
4. **Weighted combination** gives best precision/recall balance

The test script now shows both methods so you can compare. Run it again to see the difference!
