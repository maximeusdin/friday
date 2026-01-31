#!/usr/bin/env python3
"""
Simplified OCR Review Export

Features:
1. Auto-approve high-confidence matches (reduces review volume 60-80%)
2. Tiered batch exports by priority/type
3. Pre-filled review decisions (exception-based review)
4. Simplified columns for quick human decisions
5. Clear action indicators

Output batches:
- auto_approved.csv: High-confidence, no review needed (for audit)
- batch_1_confirm_match.csv: Existing entity matches, just confirm/reject
- batch_2_choose_entity.csv: Multiple candidates, pick one
- batch_3_new_entity.csv: No match, needs new entity or block
- batch_4_review_junk.csv: Likely junk, confirm block
- summary.csv: Overview statistics

Usage:
    python scripts/export_simplified_review.py --output-dir review_batches/
    python scripts/export_simplified_review.py --output-dir review_batches/ --auto-approve-threshold 0.85
"""

import argparse
import csv
import hashlib
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

sys.path.insert(0, '.')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds for auto-approval
AUTO_APPROVE_SCORE_THRESHOLD = 85.0  # Priority score threshold
AUTO_APPROVE_MIN_MENTIONS = 10       # Minimum mentions for auto-approve
JUNK_INDICATORS = {'center', 'office', 'department', 'section', 'division', 
                   'committee', 'unit', 'group', 'bureau', 'branch', 'area',
                   'the', 'and', 'for', 'with'}

# Common English words that are likely junk when appearing as all-lowercase surfaces
# Excludes proper nouns that could be cover names (e.g., HOPE, COUNTRY)
COMMON_ENGLISH_WORDS = {
    # Articles, prepositions, conjunctions
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'so', 'as', 'at', 'by',
    'for', 'in', 'of', 'on', 'to', 'up', 'with', 'from', 'into', 'over', 'after',
    'before', 'between', 'under', 'above', 'below', 'through', 'during', 'about',
    # Additional common words that appear as OCR variants
    'trust', 'hope', 'country', 'count', 'american', 'america', 'duke', 'land',
    'lane', 'pole', 'polo', 'frost', 'ernst', 'luke', 'africa', 'jean', 'jane',
    'juan', 'jose', 'kennedy', 'coventry', 'agent', 'contact', 'source', 'code',
    # Common verbs
    'be', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'shall', 'can', 'get', 'got', 'make', 'made', 'go', 'went', 'come',
    'came', 'take', 'took', 'give', 'gave', 'see', 'saw', 'know', 'knew', 'think',
    'thought', 'say', 'said', 'tell', 'told', 'ask', 'asked', 'use', 'used',
    'find', 'found', 'put', 'set', 'try', 'tried', 'leave', 'left', 'call',
    'called', 'keep', 'kept', 'let', 'begin', 'began', 'seem', 'seemed', 'help',
    'show', 'showed', 'hear', 'heard', 'play', 'run', 'ran', 'move', 'moved',
    'live', 'lived', 'believe', 'hold', 'held', 'bring', 'brought', 'happen',
    'write', 'wrote', 'provide', 'sit', 'sat', 'stand', 'stood', 'lose', 'lost',
    'pay', 'paid', 'meet', 'met', 'include', 'continue', 'learn', 'change',
    'lead', 'led', 'understand', 'watch', 'follow', 'stop', 'create', 'speak',
    'read', 'allow', 'add', 'spend', 'spent', 'grow', 'grew', 'open', 'walk',
    'win', 'won', 'offer', 'remember', 'consider', 'appear', 'buy', 'bought',
    'wait', 'serve', 'die', 'died', 'send', 'sent', 'expect', 'build', 'built',
    'stay', 'fall', 'fell', 'cut', 'reach', 'kill', 'remain', 'suggest', 'raise',
    'pass', 'sell', 'sold', 'require', 'report', 'decide', 'pull',
    # Common nouns (generic, not names)
    'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child', 'world',
    'life', 'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program',
    'question', 'work', 'government', 'number', 'night', 'point', 'home', 'water',
    'room', 'mother', 'father', 'area', 'money', 'story', 'fact', 'month', 'lot',
    'right', 'study', 'book', 'eye', 'job', 'word', 'business', 'issue', 'side',
    'kind', 'head', 'house', 'service', 'friend', 'family', 'power', 'hour',
    'game', 'line', 'end', 'member', 'law', 'car', 'city', 'community', 'name',
    'president', 'team', 'minute', 'idea', 'body', 'information', 'back', 'face',
    'others', 'level', 'office', 'door', 'health', 'person', 'art', 'war',
    'history', 'party', 'result', 'change', 'morning', 'reason', 'research',
    'moment', 'air', 'teacher', 'force', 'education', 'foot', 'boy', 'age',
    'policy', 'process', 'music', 'market', 'sense', 'nation', 'plan', 'college',
    'interest', 'death', 'experience', 'effect', 'use', 'class', 'control',
    'care', 'field', 'development', 'role', 'effort', 'rate', 'heart', 'drug',
    'show', 'leader', 'light', 'voice', 'wife', 'police', 'mind', 'difference',
    'period', 'value', 'behavior', 'security', 'building', 'action', 'activity',
    'drive', 'arm', 'table', 'risk', 'attention', 'director', 'center', 'section',
    'society', 'season', 'project', 'summer', 'tax', 'evidence', 'data', 'model',
    'source', 'position', 'ground', 'film', 'region', 'patient', 'term', 'test',
    'theory', 'town', 'need', 'nature', 'view', 'response', 'article', 'series',
    'event', 'stage', 'star', 'figure', 'growth', 'loss', 'order', 'range',
    'street', 'trial', 'page', 'south', 'north', 'east', 'west', 'letter', 'news',
    'paper', 'food', 'land', 'support', 'board', 'court', 'production', 'agency',
    # Common adjectives
    'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other',
    'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early',
    'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'local', 'sure',
    'free', 'better', 'true', 'whole', 'special', 'hard', 'best', 'clear', 'recent',
    'certain', 'personal', 'open', 'red', 'difficult', 'available', 'likely',
    'short', 'single', 'medical', 'current', 'wrong', 'private', 'past', 'foreign',
    'fine', 'common', 'poor', 'natural', 'significant', 'similar', 'hot', 'dead',
    'central', 'happy', 'serious', 'ready', 'simple', 'left', 'physical', 'general',
    'environmental', 'financial', 'blue', 'democratic', 'dark', 'various', 'entire',
    'close', 'legal', 'religious', 'cold', 'final', 'main', 'green', 'nice', 'huge',
    'popular', 'traditional', 'cultural', 'top', 'low', 'real', 'full', 'late',
    # Common adverbs
    'also', 'just', 'only', 'now', 'very', 'even', 'back', 'there', 'still',
    'well', 'here', 'then', 'again', 'never', 'always', 'really', 'most', 'often',
    'however', 'already', 'ever', 'far', 'perhaps', 'later', 'almost', 'yet',
    'probably', 'certainly', 'today', 'together', 'usually', 'rather', 'soon',
    'indeed', 'away', 'actually', 'sometimes', 'finally', 'ago', 'thus', 'especially',
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
    'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'some', 'any', 'no',
    'not', 'only', 'own', 'same', 'than', 'too', 'such', 'many', 'much',
    # Common document/OCR noise
    'page', 'copy', 'file', 'note', 'item', 'list', 'form', 'type', 'date',
    'ref', 'see', 'per', 'via', 'etc', 'vol', 'pp', 'no', 'nos', 'fig', 'tab',
}

# Batch sizes
MAX_BATCH_SIZE = 100

CONTEXT_CHARS = 120


def get_conn():
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'localhost'),
        port=int(os.environ.get('POSTGRES_PORT', '5432')),
        dbname=os.environ.get('POSTGRES_DB', 'neh'),
        user=os.environ.get('POSTGRES_USER', 'neh'),
        password=os.environ.get('POSTGRES_PASSWORD', 'neh')
    )


def clean_text(text: str) -> str:
    """Clean text for CSV."""
    if not text:
        return ''
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:500]  # Truncate long text


def is_likely_junk(proposed_name: str, variants: List[str]) -> bool:
    """
    Check if cluster is likely junk based on name patterns.
    
    CONSERVATIVE: Only flag obvious junk, not ambiguous cases.
    Real cover names like "JAN", "HOPE" should NOT be flagged (they're uppercase).
    But lowercase dictionary words like "trust", "land" ARE junk.
    """
    if not proposed_name:
        return False
    
    name_lower = proposed_name.lower().strip()
    
    # Single common generic word (but not proper nouns that could be cover names)
    if name_lower in JUNK_INDICATORS:
        return True
    
    # All lowercase + common English dictionary word = likely junk
    # Cover names are typically UPPERCASE, so "HOPE" is kept but "hope" is junk
    if proposed_name.islower() and name_lower in COMMON_ENGLISH_WORDS:
        return True
    
    # Very short (1-2 chars) AND lowercase
    if len(name_lower) <= 2 and not proposed_name.isupper():
        return True
    
    # Pure punctuation or starts with multiple punctuation marks
    if len(name_lower) > 0 and all(c in '.,;:!?()-[]{}"\' ' for c in proposed_name[:3]):
        return True
    
    # Numbers only
    if name_lower.isdigit():
        return True
    
    return False


def get_best_example_context(conn, cluster_id: str, variant_examples: List[str]) -> str:
    """Get best example context for a cluster."""
    if not variant_examples:
        return ''
    
    cur = conn.cursor()
    
    # Try to get context from mention_review_queue
    cur.execute("""
        SELECT context_excerpt, surface
        FROM mention_review_queue
        WHERE surface_norm = ANY(%s)
        AND context_excerpt IS NOT NULL
        AND context_excerpt != ''
        LIMIT 1
    """, (variant_examples[:5],))
    
    row = cur.fetchone()
    if row and row[0]:
        return clean_text(row[0])
    
    return f"Variants: {', '.join(variant_examples[:5])}"


def count_dictionary_word_variants(variants: List[str]) -> int:
    """Count how many variants are lowercase dictionary words."""
    count = 0
    for variant in variants:
        v_clean = variant.lower().strip()
        # If variant is all lowercase and is a common word, count it
        if variant.islower() and v_clean in COMMON_ENGLISH_WORDS:
            count += 1
    return count


def has_significant_dictionary_variants(variants: List[str], threshold: int = 2) -> bool:
    """
    Check if significant portion of variants are lowercase dictionary words.
    
    More lenient than checking ANY - only flags if multiple top variants are dict words.
    This avoids flagging clusters where only 1 noisy OCR variant is a common word.
    """
    # Only check first 5 variants (highest frequency)
    top_variants = variants[:5]
    dict_count = count_dictionary_word_variants(top_variants)
    
    # Flag if threshold or more of top variants are dictionary words
    return dict_count >= threshold


def classify_cluster(cluster: Dict) -> Tuple[str, str, str]:
    """
    Classify cluster into review category.
    
    Returns: (category, pre_filled_decision, action_hint)
    """
    has_entity = cluster.get('canonical_entity_id') is not None
    recommendation = cluster.get('recommendation', '')
    priority = float(cluster.get('priority_score', 0))
    mentions = cluster.get('total_mentions', 0)
    has_danger = cluster.get('maps_to_multiple_entities') or cluster.get('has_type_conflict')
    proposed = cluster.get('proposed_canonical', '')
    
    # Get variant examples for junk check
    variants = []
    if cluster.get('variant_examples'):
        variants = [v.split(' (')[0] for v in cluster['variant_examples'].split(' | ')]
    
    # Category 1: Likely junk - needs block confirmation
    if is_likely_junk(proposed, variants):
        return 'junk', 'BLOCK', '[BLOCK?] Generic/junk term'
    
    # Check if SIGNIFICANT portion of variants are dictionary words
    # Single dictionary word variant is OK (noise), but multiple is concerning
    has_significant_dict = has_significant_dictionary_variants(variants, threshold=2)
    dict_count = count_dictionary_word_variants(variants[:5])
    
    # Category 2: High-confidence auto-approve
    # ONLY if variants don't have significant dictionary word contamination
    if (has_entity and 
        recommendation == 'SAFE_ADD' and 
        priority >= AUTO_APPROVE_SCORE_THRESHOLD and
        mentions >= AUTO_APPROVE_MIN_MENTIONS and
        not has_danger and
        not has_significant_dict):
        return 'auto_approve', 'AUTO_APPROVED', '[AUTO] High-confidence match'
    
    # Category 3: Has entity match, just confirm
    # If has dictionary variants, add warning but still pre-fill APPROVE_MERGE
    if has_entity and recommendation == 'SAFE_ADD' and not has_danger:
        if has_significant_dict:
            return 'confirm_match', 'APPROVE_MERGE', f'[CONFIRM] Match found, but {dict_count} variants are common words - review'
        return 'confirm_match', 'APPROVE_MERGE', '[CONFIRM] Match looks good'
    
    # Category 4: Has entity but needs review (danger flags or low confidence)
    if has_entity and has_danger:
        return 'choose_entity', '', '[CHOOSE] Multiple candidates or type conflict'
    
    # Category 5: No entity match - needs new entity or block
    if not has_entity:
        return 'new_entity', '', '[DECIDE] Create new entity or block'
    
    # Default: needs review
    return 'choose_entity', '', '[REVIEW] Needs human decision'


def load_all_clusters(conn) -> List[Dict]:
    """Load all pending clusters with details including NER info."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT 
            c.cluster_id,
            c.proposed_canonical,
            c.canonical_entity_id,
            c.canonical_source,
            c.variant_count,
            c.total_mentions,
            c.doc_count,
            c.maps_to_multiple_entities,
            c.has_type_conflict,
            c.recommendation,
            c.priority_score,
            e.canonical_name as entity_name,
            e.entity_type,
            (
                SELECT string_agg(v.variant_key || ' (' || v.mention_count || ')', ' | ' ORDER BY v.mention_count DESC)
                FROM (SELECT variant_key, mention_count FROM ocr_cluster_variants WHERE cluster_id = c.cluster_id LIMIT 5) v
            ) as variant_examples,
            (
                SELECT array_agg(DISTINCT v.variant_key)
                FROM ocr_cluster_variants v
                WHERE v.cluster_id = c.cluster_id
            ) as all_variant_keys,
            -- NER aggregation: get dominant NER label for this cluster's variants
            (
                SELECT mode() WITHIN GROUP (ORDER BY mrq.ner_label) 
                FROM mention_review_queue mrq 
                WHERE mrq.surface_norm = ANY(
                    SELECT v2.variant_key FROM ocr_cluster_variants v2 WHERE v2.cluster_id = c.cluster_id
                ) AND mrq.ner_label IS NOT NULL
            ) as ner_dominant_label,
            (
                SELECT COUNT(DISTINCT mrq.ner_label)
                FROM mention_review_queue mrq 
                WHERE mrq.surface_norm = ANY(
                    SELECT v2.variant_key FROM ocr_cluster_variants v2 WHERE v2.cluster_id = c.cluster_id
                ) AND mrq.ner_label IS NOT NULL
            ) as ner_label_count,
            (
                SELECT COUNT(*)
                FROM mention_review_queue mrq 
                WHERE mrq.surface_norm = ANY(
                    SELECT v2.variant_key FROM ocr_cluster_variants v2 WHERE v2.cluster_id = c.cluster_id
                ) AND mrq.ner_label IS NOT NULL
            ) as ner_tagged_count
        FROM ocr_variant_clusters c
        LEFT JOIN entities e ON e.id = c.canonical_entity_id
        WHERE c.status = 'pending'
        ORDER BY c.priority_score DESC
    """)
    
    return [dict(row) for row in cur.fetchall()]


def build_simplified_row(conn, cluster: Dict, category: str, pre_filled: str, action_hint: str) -> Dict:
    """Build a simplified review row."""
    
    # Get example context
    variant_keys = cluster.get('all_variant_keys', []) or []
    context = get_best_example_context(conn, cluster['cluster_id'], variant_keys)
    
    # Build document sources summary
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT col.title
        FROM mention_review_queue mrq
        JOIN documents d ON d.id = mrq.document_id
        JOIN collections col ON col.id = d.collection_id
        WHERE mrq.surface_norm = ANY(%s)
        LIMIT 3
    """, (variant_keys[:10],))
    sources = [row[0] for row in cur.fetchall()]
    
    # Build NER info string
    ner_info = ''
    ner_label = cluster.get('ner_dominant_label')
    ner_count = cluster.get('ner_tagged_count', 0)
    ner_labels = cluster.get('ner_label_count', 0)
    if ner_label and ner_count > 0:
        if ner_labels == 1:
            ner_info = f"NER:{ner_label} ({ner_count}x)"
        else:
            ner_info = f"NER:{ner_label} ({ner_count}x, {ner_labels} labels)"
    
    return {
        # Identifiers
        'cluster_id': cluster['cluster_id'],
        
        # What reviewer sees first
        'action': action_hint,
        'proposed_name': cluster.get('proposed_canonical', ''),
        'variant_examples': cluster.get('variant_examples', ''),
        
        # Pre-filled decision (reviewer can change)
        'review_decision': pre_filled,
        
        # Context for decision
        'example_context': context,
        'sources': ' | '.join(sources) if sources else '',
        
        # Stats
        'mentions': cluster.get('total_mentions', 0),
        'docs': cluster.get('doc_count', 0),
        'variants': cluster.get('variant_count', 0),
        
        # NER info (if available)
        'ner_info': ner_info,
        
        # Existing entity info (if any)
        'existing_entity_id': cluster.get('canonical_entity_id', ''),
        'existing_entity_name': cluster.get('entity_name', ''),
        'existing_entity_type': cluster.get('entity_type', ''),
        
        # For new entities
        'entity_description': '',
        
        # Reviewer notes
        'reviewer_notes': '',
        
        # System info (for import)
        'priority_score': float(cluster.get('priority_score', 0)),
        'recommendation': cluster.get('recommendation', ''),
        'danger_flags': 'MULTI_ENTITY' if cluster.get('maps_to_multiple_entities') else (
            'TYPE_CONFLICT' if cluster.get('has_type_conflict') else ''
        ),
    }


def write_batch(rows: List[Dict], filepath: str, include_instructions: bool = True) -> int:
    """Write a batch to CSV with optional instructions header."""
    if not rows:
        return 0
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        if include_instructions:
            # Write instructions as comments
            f.write("# REVIEW INSTRUCTIONS\n")
            f.write("# - review_decision: APPROVE_MERGE (link to existing entity), APPROVE_NEW_ENTITY (create new), BLOCK (junk), or leave blank to skip\n")
            f.write("# - entity_description: For new entities, describe: role, affiliation, time period\n")
            f.write("# - Pre-filled decisions are suggestions - change if you disagree\n")
            f.write("#\n")
        
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)


def apply_auto_approvals(conn, auto_approved: List[Dict], reviewer: str) -> int:
    """Apply auto-approved decisions to database."""
    if not auto_approved:
        return 0
    
    cur = conn.cursor()
    count = 0
    
    for cluster in auto_approved:
        cluster_id = cluster['cluster_id']
        entity_id = cluster['existing_entity_id']
        
        if not entity_id:
            continue
        
        # Get variant keys for this cluster
        cur.execute("""
            SELECT variant_key FROM ocr_cluster_variants WHERE cluster_id = %s
        """, (cluster_id,))
        variant_keys = [row[0] for row in cur.fetchall()]
        
        # Add to allowlist
        for vk in variant_keys:
            cur.execute("""
                INSERT INTO ocr_variant_allowlist (variant_key, entity_id, reason, source)
                VALUES (%s, %s, %s, 'auto_approve')
                ON CONFLICT (variant_key, entity_id) DO NOTHING
            """, (vk, entity_id, f"Auto-approved from cluster {cluster_id}"))
        
        # Update cluster status
        cur.execute("""
            UPDATE ocr_variant_clusters
            SET status = 'approved', review_decision = 'AUTO_APPROVED',
                reviewed_by = %s, reviewed_at = NOW()
            WHERE cluster_id = %s
        """, (reviewer, cluster_id))
        
        count += 1
    
    conn.commit()
    return count


def main():
    global AUTO_APPROVE_SCORE_THRESHOLD
    
    parser = argparse.ArgumentParser(description='Export simplified review batches')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--auto-approve-threshold', type=float, default=AUTO_APPROVE_SCORE_THRESHOLD,
                       help='Priority score threshold for auto-approval')
    parser.add_argument('--apply-auto-approve', action='store_true',
                       help='Apply auto-approvals to database')
    parser.add_argument('--max-per-batch', type=int, default=MAX_BATCH_SIZE,
                       help='Maximum clusters per batch file')
    parser.add_argument('--reviewer', default='auto_export', help='Reviewer name for auto-approvals')
    args = parser.parse_args()
    
    AUTO_APPROVE_SCORE_THRESHOLD = args.auto_approve_threshold
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    conn = get_conn()
    
    print("=" * 60)
    print("SIMPLIFIED OCR REVIEW EXPORT")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    print(f"Auto-approve threshold: {args.auto_approve_threshold}")
    print()
    
    # Load all clusters
    print("Loading clusters...")
    clusters = load_all_clusters(conn)
    print(f"  Found {len(clusters)} pending clusters")
    
    if not clusters:
        print("No clusters to process.")
        return
    
    # Classify clusters
    print("\nClassifying clusters...")
    batches = defaultdict(list)
    
    for cluster in clusters:
        category, pre_filled, action_hint = classify_cluster(cluster)
        row = build_simplified_row(conn, cluster, category, pre_filled, action_hint)
        batches[category].append(row)
    
    # Print classification stats
    print("\nClassification results:")
    for cat, rows in sorted(batches.items()):
        print(f"  {cat}: {len(rows)} clusters")
    
    # Calculate what needs human review
    human_review_count = sum(len(rows) for cat, rows in batches.items() if cat != 'auto_approve')
    auto_approve_count = len(batches.get('auto_approve', []))
    
    print(f"\n  AUTO-APPROVED: {auto_approve_count} (no review needed)")
    print(f"  NEEDS REVIEW: {human_review_count}")
    if clusters:
        reduction = (auto_approve_count / len(clusters)) * 100
        print(f"  Review reduction: {reduction:.0f}%")
    
    # Export batches
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exported_files = {}
    
    print("\nExporting batches...")
    
    # 1. Auto-approved (for audit, not for review)
    if batches['auto_approve']:
        path = os.path.join(args.output_dir, f'auto_approved_{timestamp}.csv')
        count = write_batch(batches['auto_approve'], path, include_instructions=False)
        exported_files['auto_approved'] = (path, count)
        print(f"  auto_approved: {count} clusters (audit only)")
    
    # 2. Confirm match - just need yes/no
    if batches['confirm_match']:
        rows = batches['confirm_match'][:args.max_per_batch]
        path = os.path.join(args.output_dir, f'batch_1_confirm_match_{timestamp}.csv')
        count = write_batch(rows, path)
        exported_files['confirm_match'] = (path, count)
        print(f"  batch_1_confirm_match: {count} clusters")
        
        # Additional batches if needed
        remaining = batches['confirm_match'][args.max_per_batch:]
        batch_num = 2
        while remaining:
            batch = remaining[:args.max_per_batch]
            remaining = remaining[args.max_per_batch:]
            path = os.path.join(args.output_dir, f'batch_1_{batch_num}_confirm_match_{timestamp}.csv')
            write_batch(batch, path)
            batch_num += 1
    
    # 3. Choose entity - multiple candidates
    if batches['choose_entity']:
        rows = batches['choose_entity'][:args.max_per_batch]
        path = os.path.join(args.output_dir, f'batch_2_choose_entity_{timestamp}.csv')
        count = write_batch(rows, path)
        exported_files['choose_entity'] = (path, count)
        print(f"  batch_2_choose_entity: {count} clusters")
    
    # 4. New entity - needs creation or block
    if batches['new_entity']:
        rows = batches['new_entity'][:args.max_per_batch]
        path = os.path.join(args.output_dir, f'batch_3_new_entity_{timestamp}.csv')
        count = write_batch(rows, path)
        exported_files['new_entity'] = (path, count)
        print(f"  batch_3_new_entity: {count} clusters")
    
    # 5. Junk - likely blocks
    if batches['junk']:
        rows = batches['junk'][:args.max_per_batch]
        path = os.path.join(args.output_dir, f'batch_4_review_junk_{timestamp}.csv')
        count = write_batch(rows, path)
        exported_files['junk'] = (path, count)
        print(f"  batch_4_review_junk: {count} clusters")
    
    # Write summary
    summary_path = os.path.join(args.output_dir, f'summary_{timestamp}.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Count', 'File', 'Action Required'])
        writer.writerow(['auto_approved', auto_approve_count, 'auto_approved.csv', 'None (audit only)'])
        writer.writerow(['confirm_match', len(batches.get('confirm_match', [])), 'batch_1_confirm_match.csv', 'Confirm or reject pre-filled APPROVE_MERGE'])
        writer.writerow(['choose_entity', len(batches.get('choose_entity', [])), 'batch_2_choose_entity.csv', 'Choose entity or block'])
        writer.writerow(['new_entity', len(batches.get('new_entity', [])), 'batch_3_new_entity.csv', 'Create new entity or block'])
        writer.writerow(['junk', len(batches.get('junk', [])), 'batch_4_review_junk.csv', 'Confirm BLOCK or keep'])
        writer.writerow([])
        writer.writerow(['TOTAL_PENDING', len(clusters), '', ''])
        writer.writerow(['AUTO_APPROVED', auto_approve_count, '', ''])
        writer.writerow(['NEEDS_REVIEW', human_review_count, '', ''])
        writer.writerow(['REDUCTION_%', f'{reduction:.0f}%' if clusters else '0%', '', ''])
    
    print(f"  summary: {summary_path}")
    
    # Apply auto-approvals if requested
    if args.apply_auto_approve and batches['auto_approve']:
        print(f"\nApplying {auto_approve_count} auto-approvals to database...")
        applied = apply_auto_approvals(conn, batches['auto_approve'], args.reviewer)
        print(f"  Applied: {applied}")
    
    # Write manifest
    manifest_path = os.path.join(args.output_dir, f'manifest_{timestamp}.txt')
    with open(manifest_path, 'w') as f:
        f.write(f"Export timestamp: {timestamp}\n")
        f.write(f"Auto-approve threshold: {args.auto_approve_threshold}\n")
        f.write(f"Total clusters: {len(clusters)}\n")
        f.write(f"Auto-approved: {auto_approve_count}\n")
        f.write(f"Needs review: {human_review_count}\n")
        f.write(f"\nFiles:\n")
        for name, (path, count) in exported_files.items():
            file_hash = hashlib.md5(open(path, 'rb').read()).hexdigest()[:12]
            f.write(f"  {os.path.basename(path)}: {count} clusters (MD5: {file_hash})\n")
    
    print(f"\nManifest: {manifest_path}")
    
    # Print review instructions
    print("\n" + "=" * 60)
    print("REVIEWER INSTRUCTIONS")
    print("=" * 60)
    print("""
REVIEW ORDER (easiest first):

1. batch_4_review_junk.csv
   - Pre-filled: BLOCK
   - Action: Confirm blocks or change to keep
   - Time: ~5 min

2. batch_1_confirm_match.csv  
   - Pre-filled: APPROVE_MERGE
   - Action: Scan for errors, change if wrong
   - Time: ~10 min per 100

3. batch_3_new_entity.csv
   - Pre-filled: blank
   - Action: APPROVE_NEW_ENTITY + add description, or BLOCK
   - Time: ~20 min per 50

4. batch_2_choose_entity.csv
   - Pre-filled: blank  
   - Action: Pick entity, or BLOCK if ambiguous
   - Time: ~15 min per 50

COLUMNS:
- action: What the system suggests
- review_decision: Your decision (edit this!)
  - APPROVE_MERGE: Link to existing_entity_id
  - APPROVE_NEW_ENTITY: Create new entity
  - BLOCK: This is junk
  - (blank): Skip for now
- entity_description: For new entities, add role/affiliation/dates

IMPORT:
  python scripts/apply_ocr_adjudication.py <reviewed_file.csv>
""")
    
    conn.close()


if __name__ == '__main__':
    main()
