# GitHub Issue: Fix entity_mentions Table Pollution (Including OCR Mentions)

## Title
Fix polluted `entity_mentions` table, including OCR-derived garbage

## Problem

The `entity_mentions` table is polluted with low-quality and incorrect extractions. This affects:

- **Entity-based retrieval** (`entity_mentions_tool`, `expand_entities`) — returns chunks with spurious or garbled mentions
- **Downstream systems** — migration 0064 removed `page_entity_mentions` rows derived from `entity_mentions` because they polluted PEM with generic words (e.g. "american", "bureau", "soviet") via OCR fuzzy matching
- **Citation-based extraction** — relies on clean `entity_mentions` for targeting and validation

### Observed pollution sources

1. **OCR-derived garbage**
   - OCR fuzzy matching links generic/common words to entities (e.g. "american", "bureau", "soviet")
   - OCR-garbled text classified as entities: "Ihilip Jacob Jaffewere" (Philip Jacob Jaffe), "Red Amy" (Red Army), "Iiv Tork Olty" (New York City), "Mltad Stataa" (United States)
   - Document metadata/headers: "FILE", "VOLUME", "HEW", "TOBX", "HSO", "JACK", "HBSPI", "BUREAU", "FOIPA"
   - NER false positives on OCR noise: ".j", "••", "AS5SBr-AM^", "G^De\nDeleted"

2. **OCR error patterns**
   - Character substitutions: rn→m, cl→d, i→l, v→w
   - Missing/extra characters
   - Token splitting/merging

3. **Method mix**
   - `ocr_lexicon`, `ocr_fuzzy`, `alias_exact`, `alias_partial`, etc. — OCR methods tend to produce more false positives than concordance-based methods

## Proposed approach

### 1. Identify and quantify pollution

- [ ] Query `entity_mentions` by `method` (e.g. `ocr_lexicon`, `ocr_fuzzy`, `alias_fuzzy`) and surface text patterns
- [ ] Flag low-confidence mentions (`confidence` below threshold)
- [ ] Cross-reference `surface` / `surface_norm` against known garbage patterns (metadata, OCR artifacts)
- [ ] Identify entities with unusually high mention counts from OCR-only sources

### 2. Fix OCR mentions

- [ ] Add or improve OCR error correction before matching (see `docs/ocr_extraction_results_analysis.md`, `docs/ner_strategy_no_concordance.md`)
- [ ] Strengthen garbage filter for OCR surfaces (document headers, common OCR artifacts)
- [ ] Lower similarity or confidence thresholds for OCR-derived mentions, or exclude them from entity-based retrieval until validated
- [ ] Add `surface_quality` or source flags to distinguish OCR-derived vs concordance-derived mentions

### 3. Cleanup migration

- [ ] Add migration to delete or mark polluted rows:
  - By `method` (e.g. `ocr_lexicon`, `ocr_fuzzy`) where surface matches garbage patterns
  - By entity_id where entity is known garbage (e.g. generic words incorrectly linked)
  - By confidence threshold (e.g. below 0.6 for OCR methods)
- [ ] Consider soft-delete or `is_valid` flag for audit trail instead of hard delete

### 4. Prevention

- [ ] Toughen extraction pipeline: stricter garbage filters, OCR-aware validation
- [ ] Add validation step before insert (e.g. surface not in blocklist, confidence above method-specific threshold)
- [ ] Document acceptable methods and confidence floors per source (concordance vs OCR)

## References

- `migrations/0064_pem_alias_guard.sql` — PEM pollution from entity_mentions
- `docs/ocr_extraction_results_analysis.md` — OCR garbling patterns and extraction issues
- `docs/ner_strategy_no_concordance.md` — OCR vs clean text handling
- `docs/citation_based_entity_mention_extraction.md` — citation-based extraction
- `scripts/extract_entity_mentions.py` — main extraction script
- `scripts/adjudicate_ocr_cli.py` — OCR adjudication flow

## Acceptance criteria

- [ ] No generic words (american, bureau, soviet, etc.) linked as entity mentions via OCR fuzzy matching
- [ ] OCR-garbled surfaces (e.g. "Iiv Tork Olty") not present as valid mentions
- [ ] Document metadata/headers excluded from entity_mentions
- [ ] Clear path to re-run extraction on cleaned corpus without reintroducing pollution
- [ ] `entity_mentions` row count reduction documented (before/after)
