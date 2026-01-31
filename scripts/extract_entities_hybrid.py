#!/usr/bin/env python3
"""
Hybrid entity extraction combining pattern-based, NER-based, and fuzzy matching.

Combines multiple extraction methods with weighted confidence scoring.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from retrieval.ops import get_conn

# Import extractors
from scripts.extract_entities_pattern_based import extract_pattern_based
from scripts.extract_entities_fuzzy_known import extract_candidate_surfaces, fuzzy_match_against_known

try:
    from scripts.extract_entities_ner import NERExtractor
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False


def load_config(config_path: str = "config/ner_config.yaml") -> dict:
    """Load NER configuration."""
    config_file = Path(REPO_ROOT) / config_path
    if not config_file.exists():
        # Use defaults
        return {
            'ocr': {
                'pattern_weight': 0.3,
                'ner_weight': 0.3,
                'fuzzy_weight': 0.4,
                'confidence_threshold': 0.6,
                'fuzzy_similarity_threshold': 0.6,
            },
            'clean': {
                'pattern_weight': 0.2,
                'ner_weight': 0.5,
                'fuzzy_weight': 0.3,
                'confidence_threshold': 0.8,
                'fuzzy_similarity_threshold': 0.7,
            }
        }
    
    with open(config_file) as f:
        return yaml.safe_load(f)


def combine_entities(
    pattern_entities: List[Dict],
    ner_entities: List[Dict],
    fuzzy_entities: List[Dict],
    config: dict,
    text_quality: str = "unknown"
) -> List[Dict]:
    """
    Combine entities from multiple methods with weighted confidence.
    
    Resolves collisions by position and combines confidence scores.
    """
    quality_config = config.get(text_quality, config.get('clean', {}))
    
    # Group by position
    position_map = defaultdict(list)
    
    # Add pattern entities
    for entity in pattern_entities:
        key = (entity['start_char'], entity['end_char'])
        entity['method'] = 'pattern_based'
        entity['weight'] = quality_config.get('pattern_weight', 0.2)
        position_map[key].append(entity)
    
    # Add NER entities
    for entity in ner_entities:
        key = (entity['start_char'], entity['end_char'])
        entity['method'] = 'ner_spacy'
        entity['weight'] = quality_config.get('ner_weight', 0.4)
        position_map[key].append(entity)
    
    # Add fuzzy entities
    for entity in fuzzy_entities:
        key = (entity['start_char'], entity['end_char'])
        entity['method'] = 'fuzzy_known'
        entity['weight'] = quality_config.get('fuzzy_weight', 0.4)
        position_map[key].append(entity)
    
    # Resolve collisions and combine
    resolved = []
    for (start, end), entities in position_map.items():
        if len(entities) == 1:
            # Single match - use as-is
            entity = entities[0]
            entity['final_confidence'] = entity['confidence'] * entity['weight']
            resolved.append(entity)
        else:
            # Multiple matches at same position - resolve collision
            # Group by entity_type
            by_type = defaultdict(list)
            for e in entities:
                by_type[e['entity_type']].append(e)
            
            # For each type, take highest weighted confidence
            best_by_type = {}
            for entity_type, type_entities in by_type.items():
                best = max(type_entities, key=lambda e: e['confidence'] * e['weight'])
                best_by_type[entity_type] = best
            
            # If all agree on type, combine confidences
            if len(best_by_type) == 1:
                entity_type = list(best_by_type.keys())[0]
                best_entity = best_by_type[entity_type]
                
                # Weighted average of confidences
                total_weight = sum(e['weight'] for e in entities)
                weighted_conf = sum(
                    e['confidence'] * e['weight'] for e in entities
                ) / total_weight if total_weight > 0 else best_entity['confidence']
                
                best_entity['final_confidence'] = weighted_conf
                best_entity['methods'] = [e['method'] for e in entities]
                resolved.append(best_entity)
            else:
                # Multiple types - take highest confidence
                best = max(best_by_type.values(), key=lambda e: e['confidence'] * e['weight'])
                best['final_confidence'] = best['confidence'] * best['weight']
                best['methods'] = [e['method'] for e in entities]
                resolved.append(best)
    
    return resolved


def process_chunks(
    conn,
    config: dict,
    collection_slug: Optional[str] = None,
    document_id: Optional[int] = None,
    enable_pattern: bool = True,
    enable_ner: bool = True,
    enable_fuzzy: bool = True,
    confidence_threshold: Optional[float] = None,
    limit: Optional[int] = None,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Process chunks using hybrid extraction."""
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
    
    # Initialize NER extractor if enabled
    ner_extractor = None
    if enable_ner and NER_AVAILABLE:
        try:
            model_name = config.get('clean', {}).get('ner_model', 'en_core_web_sm')
            ner_extractor = NERExtractor(model_name)
        except Exception as e:
            print(f"Warning: Could not load NER extractor: {e}", file=sys.stderr)
            enable_ner = False
    
    total_entities = 0
    entities_to_insert = []
    
    for chunk_id, text, text_quality, doc_id in chunks:
        if not text:
            continue
        
        text_quality = text_quality or 'unknown'
        quality_config = config.get(text_quality, config.get('clean', {}))
        
        # Use quality-specific threshold if not provided
        threshold = confidence_threshold or quality_config.get('confidence_threshold', 0.7)
        
        # Extract using each method
        pattern_entities = []
        ner_entities = []
        fuzzy_entities = []
        
        if enable_pattern:
            pattern_entities = extract_pattern_based(text)
        
        if enable_ner and ner_extractor:
            ner_entities = ner_extractor.extract(text, text_quality)
        
        if enable_fuzzy:
            candidates = extract_candidate_surfaces(text)
            for start_pos, end_pos, surface in candidates:
                if len(surface) < 3 or len(surface) > 50:
                    continue
                
                matches = fuzzy_match_against_known(
                    conn,
                    surface,
                    text_quality,
                    similarity_threshold=quality_config.get('fuzzy_similarity_threshold', 0.7)
                )
                if matches:
                    best_match = matches[0]
                    fuzzy_entities.append({
                        'entity_type': best_match['entity_type'],
                        'surface': surface,
                        'start_char': start_pos,
                        'end_char': end_pos,
                        'confidence': best_match['confidence'],
                        'entity_id': best_match['entity_id']
                    })
        
        # Combine entities
        combined = combine_entities(
            pattern_entities,
            ner_entities,
            fuzzy_entities,
            config,
            text_quality
        )
        
        # Filter by threshold and insert
        for entity in combined:
            final_conf = entity.get('final_confidence', entity['confidence'])
            if final_conf >= threshold:
                total_entities += 1
                
                if not dry_run:
                    # Get or create entity
                    entity_id = entity.get('entity_id')
                    if not entity_id:
                        # Check if exists
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
                    
                    # Determine method
                    methods = entity.get('methods', [entity.get('method', 'hybrid')])
                    method = 'hybrid' if len(methods) > 1 else methods[0]
                    
                    entities_to_insert.append((
                        entity_id,
                        chunk_id,
                        doc_id,
                        entity['surface'],
                        entity.get('start_char'),
                        entity.get('end_char'),
                        final_conf,
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
        description="Extract entities using hybrid approach (pattern + NER + fuzzy)"
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
        "--config",
        default="config/ner_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--enable-pattern",
        action="store_true",
        default=True,
        help="Enable pattern-based extraction (default: True)"
    )
    parser.add_argument(
        "--enable-ner",
        action="store_true",
        default=True,
        help="Enable NER-based extraction (default: True)"
    )
    parser.add_argument(
        "--enable-fuzzy-known",
        action="store_true",
        default=True,
        help="Enable fuzzy matching against known entities (default: True)"
    )
    parser.add_argument(
        "--disable-pattern",
        action="store_true",
        help="Disable pattern-based extraction"
    )
    parser.add_argument(
        "--disable-ner",
        action="store_true",
        help="Disable NER-based extraction"
    )
    parser.add_argument(
        "--disable-fuzzy-known",
        action="store_true",
        help="Disable fuzzy matching"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Override confidence threshold from config"
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
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine enabled methods
    enable_pattern = args.enable_pattern and not args.disable_pattern
    enable_ner = args.enable_ner and not args.disable_ner
    enable_fuzzy = args.enable_fuzzy_known and not args.disable_fuzzy_known
    
    if not (enable_pattern or enable_ner or enable_fuzzy):
        print("Error: At least one extraction method must be enabled", file=sys.stderr)
        sys.exit(1)
    
    conn = get_conn()
    try:
        chunks_processed, entities_found = process_chunks(
            conn,
            config,
            collection_slug=args.collection,
            document_id=args.document_id,
            enable_pattern=enable_pattern,
            enable_ner=enable_ner,
            enable_fuzzy=enable_fuzzy,
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
