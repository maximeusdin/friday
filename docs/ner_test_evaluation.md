# NER Extraction Test Evaluation

## Test Results Summary

### OCR Document (Silvermaster PDF)
- **Classification**: `unknown` (should be `ocr` or needs PDF text extraction)
- **Issue**: Reading PDF file as raw text extracts PDF metadata strings instead of actual content
- **False Positives**: PDF format strings (`PDF`, `Linearized`, `Root`, `ID`, `Info`, `Size`, etc.)

### Clean Text Document (Committee Un-American Activities)
- **Classification**: `clean` ✓ (correct!)
- **Issue**: Too many false positives from common capitalized words
- **False Positives**: Common words (`OF`, `ON`, `TO`, `UN`, `PRINT`, `REPORT`) matched as organizations

## Issues Identified

### 1. PDF File Handling
**Problem**: Test script reads PDF files as plain text, extracting PDF format metadata instead of actual document content.

**Solution**: 
- Use PDF extraction library (PyMuPDF, pdfplumber) to extract text
- Or test on actual `.txt` files, not `.pdf` files
- Added PDF detection and filtering in test script

### 2. Pattern Matching False Positives
**Problem**: Pattern-based extractor matches:
- Common stopwords (`OF`, `ON`, `TO`, `UN`)
- PDF metadata strings (`PDF`, `Linearized`, `Root`)
- Single-word capitalized terms that aren't entities

**Fixes Applied**:
- Expanded stopword list for acronym matching
- Added PDF metadata keyword filtering
- Added common word filtering for place extraction

### 3. OCR Classification
**Problem**: OCR document classified as `unknown` instead of `ocr`.

**Reason**: PDF metadata doesn't contain typical OCR error patterns (it's format data, not OCR'd text).

**Fix Applied**: Added PDF metadata detection to skip classification when reading raw PDF format data.

## Recommendations

### Immediate Fixes
1. ✅ **Applied**: Expanded stopword filtering
2. ✅ **Applied**: Added PDF metadata filtering  
3. ✅ **Applied**: Improved PDF detection in classification

### Next Steps
1. **Use proper PDF extraction**: For PDF files, use `PyMuPDF` or `pdfplumber` to extract actual text
2. **Test on actual text files**: Use `.txt` files extracted from PDFs, not raw PDF files
3. **Improve pattern validation**: Add more context-aware filtering (e.g., don't match "OF" unless it's part of "House of Representatives")
4. **Add NER model testing**: Test SpaCy NER extraction (should have fewer false positives than pattern matching)

## Expected Improvements

After fixes:
- **Fewer false positives**: Stopwords and PDF metadata filtered out
- **Better classification**: PDF files detected and handled appropriately
- **More accurate extraction**: Only legitimate entities extracted

## Testing Recommendations

1. **For PDF files**: Extract text first using proper PDF library, then test on extracted text
2. **For clean text**: Test on actual `.txt` files, not PDFs
3. **Validate results**: Manually review sample extractions to tune thresholds
4. **Compare methods**: Test pattern-based vs NER-based vs hybrid to see which performs best
