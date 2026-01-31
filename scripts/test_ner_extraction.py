#!/usr/bin/env python3
"""
Test script for NER extraction on specific documents.

Tests OCR and clean text classification and extraction.
"""

import os
import sys
import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.classify_text_quality import classify_text_quality
from scripts.extract_entities_pattern_based import extract_pattern_based
from scripts.text_hygiene import clean_text

# Try to import NER extractor
try:
    from scripts.extract_entities_ner import NERExtractor
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False
    print("Warning: NER extractor not available. Install spacy: pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)

from scripts.extract_entities_hybrid import combine_entities, load_config
from scripts.extract_entities_fuzzy_known import extract_candidate_surfaces, fuzzy_match_against_known
from retrieval.ops import get_conn


def test_document(file_path: str, collection_slug: str, document_name: str):
    """Test NER extraction on a document file."""
    print(f"\n{'='*60}")
    print(f"Testing: {document_name}")
    print(f"File: {file_path}")
    print(f"{'='*60}\n")
    
    # Read file
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    # Check if PDF file - use PyMuPDF to extract text properly
    if file_path_obj.suffix.lower() == '.pdf':
        print(f"   PDF file detected. Extracting text using PyMuPDF...")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path_obj))
            text_pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    text_pages.append(page_text)
            doc.close()
            
            if text_pages:
                text = '\n'.join(text_pages)
                print(f"   Extracted text from {len(text_pages)} pages ({len(text):,} characters)")
            else:
                print(f"   Warning: No text found in PDF. File may be image-based or corrupted.")
                print(f"   Falling back to raw text reading...")
                with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
        except ImportError:
            print(f"   Warning: PyMuPDF not installed. Install with: pip install PyMuPDF")
            print(f"   Falling back to raw text reading (may include PDF metadata)...")
            with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            print(f"   Error extracting PDF text: {e}")
            print(f"   Falling back to raw text reading...")
            with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    else:
        # Not a PDF, read normally
        with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    
    # Apply text hygiene (especially important for OCR)
    if file_path_obj.suffix.lower() == '.pdf' or 'ocr' in document_name.lower():
        print("   Applying text hygiene (cleaning OCR artifacts)...")
        hygiene_result = clean_text(
            text,
            min_letter_ratio=0.3,
            collapse_hyphens=True,
            normalize_chars=True,
            detect_boilerplate=True
        )
        text = hygiene_result['cleaned_text']
        if hygiene_result['dropped_lines']:
            print(f"   Dropped {len(hygiene_result['dropped_lines'])} non-letter lines")
        if hygiene_result['boilerplate_zones']:
            total_boilerplate_lines = sum(
                end - start for zones in hygiene_result['boilerplate_zones'].values()
                for start, end in zones
            )
            print(f"   Detected {total_boilerplate_lines} boilerplate lines")
        print()
    
    # Test classification first (using sample)
    print("1. Text Quality Classification:")
    quality = classify_text_quality(text[:10000])  # Sample first 10k chars for classification
    print(f"   Classification: {quality}")
    
    # Determine extraction window size (use more for OCR, less for clean)
    # For OCR/unknown, use up to 100k chars; for clean, use up to 50k chars
    extraction_window = 100000 if quality == 'ocr' or quality == 'unknown' else 50000
    extraction_text = text[:extraction_window] if len(text) > extraction_window else text
    print(f"   Using {len(extraction_text):,} characters for extraction")
    
    # Show sample of extracted text for debugging
    print(f"\n   Sample of extracted text (first 2000 characters):")
    print(f"   {'-'*60}")
    lines = extraction_text[:2000].split('\n')
    for i, line in enumerate(lines[:20]):  # Show first 20 lines
        if line.strip():
            print(f"   [{i+1:3d}] {line[:80]}")
    if len(lines) > 20:
        print(f"   ... ({len(lines) - 20} more lines)")
    print(f"   {'-'*60}")
    
    # Text statistics
    word_count = len(re.findall(r'\b\w+\b', extraction_text))
    line_count = len([l for l in extraction_text.split('\n') if l.strip()])
    alpha_chars = len(re.findall(r'[a-zA-Z]', extraction_text))
    digit_chars = len(re.findall(r'[0-9]', extraction_text))
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', extraction_text))
    print(f"\n   Text statistics:")
    print(f"   - Total characters: {len(extraction_text):,}")
    print(f"   - Words: {word_count:,}")
    print(f"   - Lines: {line_count:,}")
    print(f"   - Letters: {alpha_chars:,} ({alpha_chars/len(extraction_text)*100:.1f}%)")
    print(f"   - Digits: {digit_chars:,} ({digit_chars/len(extraction_text)*100:.1f}%)")
    print(f"   - Special chars: {special_chars:,} ({special_chars/len(extraction_text)*100:.1f}%)")
    print()
    
    # Test pattern extraction
    print("2. Pattern-Based Extraction (rule-based, sample):")
    pattern_entities = extract_pattern_based(extraction_text)
    print(f"   Found {len(pattern_entities)} entities")
    for e in pattern_entities[:10]:  # Show first 10
        print(f"     {e['entity_type']}: '{e['surface']}' (confidence: {e['confidence']:.2f})")
    print()
    
    # Test NER extraction (if available) - this is the intelligent ML-based approach
    ner_entities = []
    if NER_AVAILABLE:
        print("3. SpaCy NER Extraction (ML-based, sample):")
        try:
            extractor = NERExtractor("en_core_web_sm")
            ner_entities = extractor.extract(extraction_text, quality)
            print(f"   Found {len(ner_entities)} entities")
            if ner_entities:
                for e in ner_entities[:10]:  # Show first 10
                    print(f"     {e['entity_type']}: '{e['surface']}' (confidence: {e['confidence']:.2f}, label: {e['ner_label']})")
            else:
                print("   No entities found")
        except Exception as e:
            print(f"   Error: {e}")
            print(f"   Install spacy model: python -m spacy download en_core_web_sm")
        print()
    else:
        print("3. SpaCy NER Extraction: Not available")
        print("   Install: pip install spacy && python -m spacy download en_core_web_sm")
        print()
    
    # Test hybrid extraction (combines all methods)
    print("4. Hybrid Extraction (combines pattern + NER + fuzzy, sample):")
    try:
        config = load_config()
        quality_config = config.get(quality, config.get('clean', {}))
        confidence_threshold = quality_config.get('confidence_threshold', 0.7)
        
        # Get fuzzy matches (if DB available)
        fuzzy_entities = []
        try:
            conn = get_conn()
            candidates = extract_candidate_surfaces(extraction_text)
            fuzzy_similarity_threshold = quality_config.get('fuzzy_similarity_threshold', 0.7)
            
            # For OCR, check more candidates since there's more noise
            candidate_limit = 200 if quality == 'ocr' or quality == 'unknown' else 100
            for start_pos, end_pos, surface in candidates[:candidate_limit]:
                matches = fuzzy_match_against_known(
                    conn,
                    surface,
                    quality,
                    fuzzy_similarity_threshold,
                    max_results=1
                )
                if matches:
                    best_match = matches[0]
                    fuzzy_entities.append({
                        'entity_type': best_match['entity_type'],
                        'surface': surface,
                        'start_char': start_pos,
                        'end_char': end_pos,
                        'confidence': best_match['similarity'],
                        'entity_id': best_match['entity_id'],
                        'matched_alias': best_match['alias']
                    })
            conn.close()
        except Exception as e:
            # DB not available or error - skip fuzzy matching
            pass
        
        # Combine all methods
        combined = combine_entities(
            pattern_entities,
            ner_entities,
            fuzzy_entities,
            config,
            quality
        )
        
        # Filter by confidence threshold
        filtered = [
            e for e in combined 
            if e.get('final_confidence', e.get('confidence', 0)) >= confidence_threshold
        ]
        
        # Filter out obvious garbage (PDF metadata, encoding strings)
        def is_garbage(surface: str) -> bool:
            """Check if surface looks like PDF metadata or encoding garbage."""
            surface_lower = surface.lower()
            
            # PDF metadata patterns (expanded)
            pdf_patterns = [
                '/info', '/metadata', '/colorspace', '/devicegray', '/stream', 
                'obj', 'endobj', '/font', '/type', '/subtype', '/length',
                '/filter', '/width', '/height', '/bitspercomponent', '/xobject',
                '/image', '/page', '/pages', '/catalog', '/root', '/xref',
                '/trailer', '/startxref', '/linearized', '/version'
            ]
            if any(x in surface_lower for x in pdf_patterns):
                return True
            
            # Encoding strings (mostly non-printable or special chars)
            if len(surface) > 20 and not re.search(r'[a-zA-Z]{3,}', surface):
                return True
            
            # Hash-like strings (hex sequences)
            if re.match(r'^[a-f0-9]{16,}$', surface_lower):
                return True
            
            # Too many special chars (more aggressive for OCR)
            special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', surface)) / max(len(surface), 1)
            threshold = 0.25 if quality == 'ocr' or quality == 'unknown' else 0.3
            if special_char_ratio > threshold:
                return True
            
            # Very short strings with special chars (likely encoding artifacts)
            if len(surface) <= 5 and re.search(r'[^a-zA-Z0-9]', surface):
                return True
            
            # Strings that are mostly numbers/special chars
            if len(surface) > 5:
                alpha_ratio = len(re.findall(r'[a-zA-Z]', surface)) / len(surface)
                if alpha_ratio < 0.3:  # Less than 30% letters
                    return True
            
            # Common PDF encoding artifacts
            if re.search(r'[<>\[\](){}]', surface) and len(surface) < 10:
                return True
            
            return False
        
        final_entities = [e for e in filtered if not is_garbage(e['surface'])]
        
        print(f"   Combined: {len(combined)} entities")
        print(f"   After confidence filter (>{confidence_threshold:.2f}): {len(filtered)} entities")
        print(f"   After garbage filter: {len(final_entities)} entities")
        print()
        print("   Top results:")
        # Sort by final confidence
        final_entities.sort(key=lambda e: e.get('final_confidence', e.get('confidence', 0)), reverse=True)
        for e in final_entities[:15]:  # Show top 15
            methods = e.get('methods', [e.get('method', 'unknown')])
            conf = e.get('final_confidence', e.get('confidence', 0))
            print(f"     {e['entity_type']}: '{e['surface'][:60]}' (confidence: {conf:.2f}, methods: {', '.join(methods)})")
        
    except Exception as e:
        print(f"   Error running hybrid extraction: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Note: Full extraction would require ingesting document first
    print("5. Full extraction requires document to be ingested into database.")
    print("   Use: python scripts/extract_entities_hybrid.py --collection <slug> --document-id <id>")
    print("   The hybrid approach combines pattern + NER + fuzzy matching for best results.")


def main():
    parser = argparse.ArgumentParser(
        description="Test NER extraction on specific documents"
    )
    parser.add_argument(
        "--ocr-file",
        default="data/raw/silvermaster/pdf/FBI File Silvermaster Part 1 November 1945_text.pdf",
        help="OCR document file to test"
    )
    parser.add_argument(
        "--clean-file",
        default="data/raw/committee_unamerican/Report of the Committee of Un-American Activities 1948_djvu.txt",
        help="Clean text document file to test"
    )
    parser.add_argument(
        "--test-ocr-only",
        action="store_true",
        help="Test only OCR file"
    )
    parser.add_argument(
        "--test-clean-only",
        action="store_true",
        help="Test only clean text file"
    )
    
    args = parser.parse_args()
    
    if not args.test_clean_only:
        test_document(
            args.ocr_file,
            "silvermaster",
            "Silvermaster OCR Document"
        )
    
    if not args.test_ocr_only:
        test_document(
            args.clean_file,
            "committee_unamerican",
            "Committee Un-American Activities Clean Text"
        )
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
