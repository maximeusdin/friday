#!/usr/bin/env python3
"""
NER-based entity extraction using SpaCy models.

Extracts person names, organizations, and places using statistical NER models.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spacy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)

import psycopg2
from retrieval.entity_resolver import normalize_alias
from retrieval.ops import get_conn

# NER label to entity type mapping
NER_LABEL_MAP = {
    'PERSON': 'person',
    'ORG': 'org',
    'GPE': 'place',  # Geopolitical entity
    'LOC': 'place',  # Location
    'FAC': 'place',  # Facility
}


class NERExtractor:
    """SpaCy-based NER extractor."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        if not SPACY_AVAILABLE:
            raise RuntimeError("spacy not available. Install with: pip install spacy")
        
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(
                f"SpaCy model '{model_name}' not found. "
                f"Download with: python -m spacy download {model_name}"
            )
    
    def extract(self, text: str, text_quality: str = "unknown") -> List[Dict]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract from
            text_quality: 'ocr', 'clean', or 'unknown'
        
        Returns:
            List of entity dictionaries
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_type = NER_LABEL_MAP.get(ent.label_)
            if entity_type:
                confidence = self.calculate_confidence(ent, text_quality)
                
                entities.append({
                    'entity_type': entity_type,
                    'surface': ent.text,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'confidence': confidence,
                    'ner_label': ent.label_,
                    'ner_confidence': getattr(ent, 'score', 1.0) if hasattr(ent, 'score') else 1.0
                })
        
        return entities
    
    def calculate_confidence(self, ent, text_quality: str) -> float:
        """
        Calculate confidence score for entity.
        
        Base confidence depends on text quality and entity characteristics.
        """
        # Base confidence from model (if available)
        base_confidence = getattr(ent, 'score', 0.8) if hasattr(ent, 'score') else 0.8
        
        # Adjust for text quality
        if text_quality == 'ocr':
            base_confidence -= 0.15
        elif text_quality == 'clean':
            base_confidence += 0.1
        
        # Adjust for entity length (longer entities often more reliable)
        token_count = len(ent.text.split())
        if token_count > 3:
            base_confidence += 0.05
        elif token_count == 1:
            base_confidence -= 0.1  # Single tokens less reliable
        
        # Adjust for capitalization (proper names should be capitalized)
        if ent.text and ent.text[0].isupper():
            base_confidence += 0.05
        
        return min(1.0, max(0.0, base_confidence))


def process_chunks(
    conn,
    extractor: NERExtractor,
    collection_slug: Optional[str] = None,
    document_id: Optional[int] = None,
    confidence_threshold: float = 0.7,
    limit: Optional[int] = None,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Process chunks and extract entities using NER."""
    cur = conn.cursor()
    
    # Build query
    conditions = []
    params = []
    
    if collection_slug:
        conditions.append("""
            EXISTS (
                SELECT 1 FROM chunk_metadata cm
                JOIN documents d ON cm.document_id = d.id
                JOIN collections c ON d.collection_id = c.id
                WHERE cm.chunk_id = chunks.id AND c.slug = %s
            )
        """)
        params.append(collection_slug)
    
    if document_id:
        conditions.append("""
            EXISTS (
                SELECT 1 FROM chunk_metadata cm
                WHERE cm.chunk_id = chunks.id AND cm.document_id = %s
            )
        """)
        params.append(document_id)
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
        SELECT c.id, c.text, c.text_quality, cm.document_id
        FROM chunks c
        JOIN chunk_metadata cm ON c.id = cm.chunk_id
        WHERE {where_clause}
        ORDER BY c.id
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query, params)
    chunks = cur.fetchall()
    
    total_entities = 0
    entities_to_insert = []
    
    for chunk_id, text, text_quality, doc_id in chunks:
        if not text:
            continue
        
        text_quality = text_quality or 'unknown'
        entities = extractor.extract(text, text_quality)
        
        for entity in entities:
            if entity['confidence'] >= confidence_threshold:
                total_entities += 1
                
                if not dry_run:
                    # Check if entity exists
                    surface_norm = normalize_alias(entity['surface'])
                    cur.execute("""
                        SELECT id FROM entities
                        WHERE entity_type = %s
                        AND LOWER(canonical_name) = LOWER(%s)
                        LIMIT 1
                    """, (entity['entity_type'], entity['surface']))
                    
                    entity_row = cur.fetchone()
                    if entity_row:
                        entity_id = entity_row[0]
                    else:
                        # Create new entity
                        cur.execute("""
                            INSERT INTO entities (entity_type, canonical_name)
                            VALUES (%s, %s)
                            RETURNING id
                        """, (entity['entity_type'], entity['surface']))
                        entity_id = cur.fetchone()[0]
                    
                    # Determine method name
                    method = 'ner_spacy'
                    if 'transformer' in str(extractor.nlp.meta.get('name', '')):
                        method = 'ner_transformer'
                    
                    # Insert mention
                    entities_to_insert.append((
                        entity_id,
                        chunk_id,
                        doc_id,
                        entity['surface'],
                        entity.get('start_char'),
                        entity.get('end_char'),
                        entity['confidence'],
                        method
                    ))
    
    if not dry_run and entities_to_insert:
        cur.executemany("""
            INSERT INTO entity_mentions
            (entity_id, chunk_id, document_id, surface, start_char, end_char, confidence, method)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, entities_to_insert)
        conn.commit()
        print(f"Inserted {len(entities_to_insert)} entity mentions", file=sys.stderr)
    
    return len(chunks), total_entities


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities using SpaCy NER models"
    )
    parser.add_argument(
        "--collection",
        help="Collection slug"
    )
    parser.add_argument(
        "--document-id",
        type=int,
        help="Document ID"
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="SpaCy model to use (default: en_core_web_sm)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of chunks to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't insert, just show what would be extracted"
    )
    parser.add_argument(
        "--test-text",
        help="Test extraction on provided text string"
    )
    
    args = parser.parse_args()
    
    if not SPACY_AVAILABLE:
        print("Error: spacy not installed. Install with: pip install spacy", file=sys.stderr)
        sys.exit(1)
    
    try:
        extractor = NERExtractor(args.model)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.test_text:
        entities = extractor.extract(args.test_text)
        print(f"Found {len(entities)} entities:")
        for e in entities:
            print(f"  {e['entity_type']}: '{e['surface']}' (confidence: {e['confidence']:.2f}, label: {e['ner_label']})")
        return
    
    conn = get_conn()
    try:
        chunks_processed, entities_found = process_chunks(
            conn,
            extractor,
            collection_slug=args.collection,
            document_id=args.document_id,
            confidence_threshold=args.confidence_threshold,
            limit=args.limit,
            dry_run=args.dry_run
        )
        
        print(f"\nProcessed {chunks_processed} chunks", file=sys.stderr)
        print(f"Found {entities_found} entities above threshold", file=sys.stderr)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
