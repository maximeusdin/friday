#!/usr/bin/env python3
"""
NER Extraction Runner with Alias Lexicon Integration

Runs spaCy NER on documents and integrates with the existing entity extraction pipeline.
Key features:
- Extracts entities using spaCy NER
- Matches extracted spans against the alias lexicon (entity_aliases)
- Uses both exact and fuzzy (trigram) matching
- Links matched spans to known entities
- Stores unmatched spans as proposals for review
- Extracts date signals (optional)

IMPORTANT:
- Run migration 0036_spacy_ner_signals.sql before using this script
- Uses canonical normalize_surface() for consistency with rest of pipeline
- quality_score and ner_accept_score are SEPARATE concepts
- Skip-processed uses chunk_ner_runs table, not row existence

Usage:
    # Test on specific documents first (dry run)
    python scripts/run_ner_extraction.py --doc-ids 123,456,789 --dry-run
    
    # Run on a collection with alias matching
    python scripts/run_ner_extraction.py --collection venona --limit 100
    
    # Skip alias matching (just extract raw NER spans)
    python scripts/run_ner_extraction.py --collection venona --no-match-aliases
    
    # Adjust fuzzy matching threshold
    python scripts/run_ner_extraction.py --collection venona --fuzzy-threshold 0.5
    
    # Full corpus run
    python scripts/run_ner_extraction.py --all --batch-size 50
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values, Json

from retrieval.ner_integration import (
    NERExtractor, NERSpan, DateSpan, SPACY_AVAILABLE,
    dedupe_ner_against_candidates, aggregate_ner_for_cluster
)
from retrieval.normalization import normalize_for_fts as normalize_surface
from retrieval.ner_guardrails import NERGuardrails

# Pipeline version for tracking
PIPELINE_VERSION = '1.0'

# Stricter threshold for NER-only candidates (secondary proposer policy)
NER_STRICT_THRESHOLD = 0.6

# Default quality score for NER-only candidates (neutral value)
# This ensures downstream resolver can handle them without NULL issues
NER_DEFAULT_QUALITY_SCORE = 0.5

# Commit every N chunks for performance (batch commit)
COMMIT_BATCH_SIZE = 50


def get_conn():
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'localhost'),
        port=int(os.environ.get('POSTGRES_PORT', '5432')),
        dbname=os.environ.get('POSTGRES_DB', 'neh'),
        user=os.environ.get('POSTGRES_USER', 'neh'),
        password=os.environ.get('POSTGRES_PASSWORD', 'neh')
    )

_STOP_REQUESTED = False


def _install_interrupt_handlers() -> None:
    """
    Ensure Ctrl+C (SIGINT) and SIGTERM lead to a *graceful* stop:
    finish current batch, commit progress, exit.
    """
    def _handler(signum, frame):
        global _STOP_REQUESTED
        if not _STOP_REQUESTED:
            _STOP_REQUESTED = True
            # stderr so it shows up even if stdout is redirected/buffered
            print(
                "\n[INTERRUPT] Stop requested. Finishing current batch, committing progress, then exiting...",
                file=sys.stderr,
                flush=True,
            )

    # SIGTERM isn't available on some Windows runtimes, so be defensive.
    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def check_ner_schema(conn, warn_only: bool = False, check_dates: bool = True) -> bool:
    """
    Check if NER schema is ready. Does NOT modify schema.
    
    Args:
        conn: Database connection
        warn_only: If True, print warning instead of raising error
        check_dates: If True, also check ner_date_signals table
    
    Returns True if ready, raises error (or warns) if not.
    """
    cur = conn.cursor()
    missing = []
    
    # Check for NER columns on mention_candidates
    required_mc_columns = ['ner_label', 'ner_source', 'ner_type_hint', 'ner_accept_score', 'ner_context_features']
    for col in required_mc_columns:
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'mention_candidates' AND column_name = %s
        """, (col,))
        if not cur.fetchone():
            missing.append(f"mention_candidates.{col}")
    
    # Check for chunk_ner_runs table and required columns
    cur.execute("""
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'chunk_ner_runs'
    """)
    if not cur.fetchone():
        missing.append("table chunk_ner_runs")
    else:
        # Check for required columns in chunk_ner_runs
        required_cnr_columns = [
            'spans_extracted', 'spans_upserted', 'spans_enhanced_existing',
            'spans_dropped_overlap', 'spans_dropped_filters', 'updated_at'
        ]
        for col in required_cnr_columns:
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'chunk_ner_runs' AND column_name = %s
            """, (col,))
            if not cur.fetchone():
                missing.append(f"chunk_ner_runs.{col}")
    
    # Check for unique constraint (required for ON CONFLICT)
    cur.execute("""
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'mention_candidates_span_unique'
    """)
    if not cur.fetchone():
        missing.append("constraint mention_candidates_span_unique")
    
    # Check for ner_date_signals table if date extraction is enabled
    if check_dates:
        cur.execute("""
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'ner_date_signals'
        """)
        if not cur.fetchone():
            missing.append("table ner_date_signals")
        else:
            # Check for required columns
            required_date_columns = [
                'surface_norm', 'ner_label', 'ner_model', 
                'parsed_date_start', 'parsed_date_end', 'parsed_precision',
                'raw_parser', 'parser_payload'
            ]
            for col in required_date_columns:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'ner_date_signals' AND column_name = %s
                """, (col,))
                if not cur.fetchone():
                    missing.append(f"ner_date_signals.{col}")
            
            # Check for unique constraint (required for ON CONFLICT)
            cur.execute("""
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'ner_date_signals_span_unique'
            """)
            if not cur.fetchone():
                missing.append("constraint ner_date_signals_span_unique")
        
        # Check for date tracking columns in chunk_ner_runs
        for col in ['date_spans_extracted', 'date_spans_saved', 'dates_enabled']:
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'chunk_ner_runs' AND column_name = %s
            """, (col,))
            if not cur.fetchone():
                missing.append(f"chunk_ner_runs.{col}")
    
    if missing:
        msg = (
            f"NER schema incomplete. Missing: {', '.join(missing)}\n"
            "Run migrations first:\n"
            "  make ner-migrate\n"
            "Or manually:\n"
            "  psql -U neh -d neh -f migrations/0036_spacy_ner_signals.sql\n"
            "  psql -U neh -d neh -f migrations/0037_ner_date_extraction.sql"
        )
        if warn_only:
            print(f"WARNING: {msg}")
            return False
        else:
            raise RuntimeError(msg)
    
    return True


def get_chunks_for_processing(
    conn,
    collection: Optional[str] = None,
    doc_ids: Optional[List[int]] = None,
    limit: Optional[int] = None,
    model: str = 'en_core_web_lg',
    threshold: float = 0.5,
    skip_processed: bool = True,
    require_dates: bool = False,
) -> List[Dict]:
    """
    Get chunks that need NER processing.
    
    Uses chunk_ner_runs table for proper skip-processed logic,
    NOT existence of NER rows (which could be partial).
    
    Args:
        require_dates: If True, only skip chunks where dates were also extracted.
                      This prevents "I ran without dates, now I want dates" from
                      incorrectly skipping chunks.
    """
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    conditions = ["c.text IS NOT NULL", "LENGTH(c.text) > 50"]
    params: List = []
    
    if collection:
        # Accept either slug (e.g., "venona") or title (e.g., "Venona Decrypts")
        conditions.append("(col.slug = %s OR col.title = %s)")
        params.extend([collection, collection])
    
    if doc_ids:
        conditions.append("d.id = ANY(%s)")
        params.append(doc_ids)
    
    if skip_processed:
        # Skip chunks that have a completed run with same model/threshold/version
        # If require_dates, also require dates_enabled=true in the previous run
        dates_condition = "AND cnr.dates_enabled = TRUE" if require_dates else ""
        conditions.append(f"""
            NOT EXISTS (
                SELECT 1 FROM chunk_ner_runs cnr 
                WHERE cnr.chunk_id = c.id 
                AND cnr.model = %s 
                AND cnr.threshold = %s 
                AND cnr.pipeline_version = %s
                AND cnr.status = 'completed'
                {dates_condition}
            )
        """)
        params.extend([model, threshold, PIPELINE_VERSION])
    
    where_clause = "WHERE " + " AND ".join(conditions)
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    # Note: chunks don't have direct document_id; use chunk_metadata table
    query = f"""
        SELECT c.id as chunk_id, c.text, cm.document_id, d.source_name as doc_name,
               col.title as collection_name
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id
        JOIN documents d ON d.id = cm.document_id
        JOIN collections col ON col.id = d.collection_id
        {where_clause}
        ORDER BY c.id
        {limit_clause}
    """
    
    cur.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def get_existing_candidates(conn, chunk_id: int) -> List[Dict]:
    """Get existing candidates for a chunk (for deduplication)."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, raw_span as surface, surface_norm, char_start as start_char, char_end as end_char,
               quality_score, resolved_entity_id as entity_id, ner_label, ner_type_hint
        FROM mention_candidates
        WHERE chunk_id = %s
    """, (chunk_id,))
    return [dict(row) for row in cur.fetchall()]


# =============================================================================
# ALIAS LEXICON INTEGRATION
# Match NER spans against alias_lexicon_index (corpus-derived seed lexicon)
# =============================================================================

def batch_lookup_exact_aliases(
    conn,
    surface_norms: List[str],
) -> Dict[str, List[Dict]]:
    """
    Batch lookup exact matches in alias_lexicon_index.
    
    Uses the corpus-derived lexicon which has:
    - doc_freq, mention_count: corpus evidence
    - corpus_confidence: average confidence from corpus
    - proposal_tier: 1=auto-accept, 2=queue-eligible
    - alias_class: person_given, person_last, covername, etc.
    
    Returns dict mapping surface_norm -> list of matching aliases with entity info.
    Falls back to entity_aliases if lexicon is empty.
    """
    if not surface_norms:
        return {}
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # First try alias_lexicon_index (has corpus stats)
    cur.execute("""
        SELECT 
            ali.alias_norm,
            ali.entity_id,
            ali.alias_norm as alias,
            ali.alias_class as kind,
            ali.proposal_tier,
            ali.doc_freq,
            ali.mention_count,
            ali.corpus_confidence,
            CASE WHEN ali.proposal_tier = 1 THEN true ELSE false END as is_auto_match,
            e.canonical_name as entity_name,
            e.entity_type
        FROM alias_lexicon_index ali
        JOIN entities e ON e.id = ali.entity_id
        WHERE ali.alias_norm = ANY(%s)
        ORDER BY 
            ali.alias_norm,
            ali.proposal_tier NULLS LAST,  -- Tier 1 first
            ali.doc_freq DESC NULLS LAST   -- Higher frequency first
    """, (list(set(surface_norms)),))
    
    results: Dict[str, List[Dict]] = {}
    for row in cur.fetchall():
        norm = row['alias_norm']
        if norm not in results:
            results[norm] = []
        results[norm].append(dict(row))
    
    # If lexicon is empty, fall back to entity_aliases
    if not results:
        cur.execute("""
            SELECT 
                ea.alias_norm,
                ea.entity_id,
                ea.alias,
                ea.kind,
                ea.is_auto_match,
                NULL as proposal_tier,
                NULL as doc_freq,
                NULL as mention_count,
                NULL as corpus_confidence,
                e.canonical_name as entity_name,
                e.entity_type
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.entity_id
            WHERE ea.alias_norm = ANY(%s)
              AND ea.is_matchable = true
            ORDER BY ea.alias_norm, ea.entity_id
        """, (list(set(surface_norms)),))
        
        for row in cur.fetchall():
            norm = row['alias_norm']
            if norm not in results:
                results[norm] = []
            results[norm].append(dict(row))
    
    return results


def batch_lookup_fuzzy_aliases(
    conn,
    surface_norms: List[str],
    similarity_threshold: float = 0.4,
    max_results_per_norm: int = 5,
) -> Dict[str, List[Dict]]:
    """
    Batch fuzzy lookup using trigram similarity against alias_lexicon_index.
    
    Uses the GIN trigram index on alias_lexicon_index for fast fuzzy matching.
    Incorporates corpus stats into scoring.
    
    Returns dict mapping surface_norm -> list of similar aliases with scores.
    """
    if not surface_norms:
        return {}
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    results: Dict[str, List[Dict]] = {}
    
    for norm in surface_norms:
        if len(norm) < 3:
            continue  # Trigram needs at least 3 chars
        
        # Use alias_lexicon_index with GIN trigram index
        cur.execute("""
            SELECT 
                ali.alias_norm,
                ali.entity_id,
                ali.alias_norm as alias,
                ali.alias_class as kind,
                ali.proposal_tier,
                ali.doc_freq,
                ali.mention_count,
                ali.corpus_confidence,
                CASE WHEN ali.proposal_tier = 1 THEN true ELSE false END as is_auto_match,
                e.canonical_name as entity_name,
                e.entity_type,
                similarity(ali.alias_norm, %s) as sim_score,
                -- Combined score: trigram + corpus evidence
                (
                    similarity(ali.alias_norm, %s) * 0.6 +
                    COALESCE(ali.corpus_confidence, 0.5) * 0.2 +
                    CASE 
                        WHEN ali.doc_freq > 10 THEN 0.2
                        WHEN ali.doc_freq > 3 THEN 0.1
                        ELSE 0.0
                    END
                ) as combined_score
            FROM alias_lexicon_index ali
            JOIN entities e ON e.id = ali.entity_id
            WHERE similarity(ali.alias_norm, %s) > %s
            ORDER BY combined_score DESC, ali.doc_freq DESC NULLS LAST
            LIMIT %s
        """, (norm, norm, norm, similarity_threshold, max_results_per_norm))
        
        matches = [dict(row) for row in cur.fetchall()]
        
        # If lexicon empty, fall back to entity_aliases
        if not matches:
            cur.execute("""
                SELECT 
                    ea.alias_norm,
                    ea.entity_id,
                    ea.alias,
                    ea.kind,
                    ea.is_auto_match,
                    NULL as proposal_tier,
                    NULL as doc_freq,
                    NULL as mention_count,
                    NULL as corpus_confidence,
                    e.canonical_name as entity_name,
                    e.entity_type,
                    similarity(ea.alias_norm, %s) as sim_score,
                    similarity(ea.alias_norm, %s) as combined_score
                FROM entity_aliases ea
                JOIN entities e ON e.id = ea.entity_id
                WHERE ea.is_matchable = true
                  AND similarity(ea.alias_norm, %s) > %s
                ORDER BY sim_score DESC
                LIMIT %s
            """, (norm, norm, norm, similarity_threshold, max_results_per_norm))
            matches = [dict(row) for row in cur.fetchall()]
        
        if matches:
            results[norm] = matches
    
    return results


def match_spans_against_lexicon(
    conn,
    spans: List[NERSpan],
    use_fuzzy: bool = True,
    fuzzy_threshold: float = 0.4,
) -> Tuple[List[Dict], List[NERSpan]]:
    """
    Match NER spans against the alias lexicon.
    
    Returns:
        matched_spans: List of dicts with span + match info (entity_id, score, etc.)
        unmatched_spans: List of NERSpan that had no match
    """
    if not spans:
        return [], []
    
    # Build normalized surfaces
    span_norms = [(s, normalize_surface(s.surface)) for s in spans]
    all_norms = [norm for _, norm in span_norms]
    
    # Phase 1: Exact matches
    exact_matches = batch_lookup_exact_aliases(conn, all_norms)
    
    matched_spans = []
    unmatched_spans = []
    norms_needing_fuzzy = []
    
    for span, norm in span_norms:
        if norm in exact_matches:
            # Found exact match(es)
            aliases = exact_matches[norm]
            best_match = aliases[0]  # Ranked by tier, then doc_freq
            
            matched_spans.append({
                'span': span,
                'surface_norm': norm,
                'match_type': 'exact',
                'entity_id': best_match['entity_id'],
                'entity_name': best_match['entity_name'],
                'entity_type': best_match['entity_type'],
                'matched_alias': best_match['alias'],
                'alias_kind': best_match['kind'],
                'is_auto_match': best_match['is_auto_match'],
                'match_score': 1.0,
                # Corpus stats from lexicon
                'proposal_tier': best_match.get('proposal_tier'),
                'doc_freq': best_match.get('doc_freq'),
                'corpus_confidence': best_match.get('corpus_confidence'),
                'candidate_entities': aliases,  # All matching entities
            })
        else:
            unmatched_spans.append(span)
            norms_needing_fuzzy.append((span, norm))
    
    # Phase 2: Fuzzy matches for unmatched spans
    if use_fuzzy and norms_needing_fuzzy:
        fuzzy_norms = [norm for _, norm in norms_needing_fuzzy]
        fuzzy_matches = batch_lookup_fuzzy_aliases(
            conn, fuzzy_norms, similarity_threshold=fuzzy_threshold
        )
        
        still_unmatched = []
        for span, norm in norms_needing_fuzzy:
            if norm in fuzzy_matches:
                aliases = fuzzy_matches[norm]
                best_match = aliases[0]  # Ranked by combined_score (trigram + corpus)
                
                matched_spans.append({
                    'span': span,
                    'surface_norm': norm,
                    'match_type': 'fuzzy',
                    'entity_id': best_match['entity_id'],
                    'entity_name': best_match['entity_name'],
                    'entity_type': best_match['entity_type'],
                    'matched_alias': best_match['alias'],
                    'alias_kind': best_match['kind'],
                    'is_auto_match': best_match['is_auto_match'],
                    'match_score': best_match.get('combined_score', best_match['sim_score']),
                    # Corpus stats from lexicon
                    'proposal_tier': best_match.get('proposal_tier'),
                    'doc_freq': best_match.get('doc_freq'),
                    'corpus_confidence': best_match.get('corpus_confidence'),
                    'candidate_entities': aliases,
                })
            else:
                still_unmatched.append(span)
        
        unmatched_spans = still_unmatched
    
    return matched_spans, unmatched_spans


def record_ner_run(
    conn,
    chunk_id: int,
    model: str,
    threshold: float,
    spans_extracted: int,
    spans_upserted: int,  # Honest name: attempted upserts, not "new inserts"
    spans_enhanced_existing: int,
    spans_dropped_overlap: int,
    spans_dropped_filters: int,
    date_spans_extracted: int = 0,
    date_spans_saved: int = 0,
    dates_enabled: bool = False,
    status: str = 'completed',
) -> None:
    """Record NER processing run for a chunk with detailed stats (entity + date)."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chunk_ner_runs 
            (chunk_id, model, threshold, pipeline_version, 
             spans_extracted, spans_upserted, spans_enhanced_existing,
             spans_dropped_overlap, spans_dropped_filters,
             date_spans_extracted, date_spans_saved, dates_enabled,
             status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id, model, threshold, pipeline_version) DO UPDATE SET
            spans_extracted = EXCLUDED.spans_extracted,
            spans_upserted = EXCLUDED.spans_upserted,
            spans_enhanced_existing = EXCLUDED.spans_enhanced_existing,
            spans_dropped_overlap = EXCLUDED.spans_dropped_overlap,
            spans_dropped_filters = EXCLUDED.spans_dropped_filters,
            date_spans_extracted = EXCLUDED.date_spans_extracted,
            date_spans_saved = EXCLUDED.date_spans_saved,
            dates_enabled = EXCLUDED.dates_enabled,
            status = EXCLUDED.status,
            updated_at = NOW()
    """, (chunk_id, model, threshold, PIPELINE_VERSION, 
          spans_extracted, spans_upserted, spans_enhanced_existing,
          spans_dropped_overlap, spans_dropped_filters,
          date_spans_extracted, date_spans_saved, dates_enabled,
          status))


def is_single_token_lowercase_junk(span: NERSpan) -> bool:
    """
    Check if span is single-token lowercase that should be rejected.
    
    Even with strict threshold, spaCy produces lots of single-token lowercase
    junk in OCR. Require additional signals to accept these.
    
    Escape hatches for legitimate covernames/entities:
    - Quoted occurrence
    - Codename/alias markers nearby
    - Strong type-specific hints
    - ORG label with org hints (organizations often lowercase in OCR)
    - GPE/LOC label with location hints (places often lowercase in OCR)
    """
    if span.token_count != 1:
        return False
    
    if not span.surface.islower():
        return False
    
    # Single-token lowercase - check for saving graces
    hints = span.context_hints
    
    # Covername escape hatches
    if hints.get('quoted', False):
        return False  # In quotes - likely intentional
    
    if hints.get('codename', 0) >= 1:
        return False  # Near codename markers
    
    if hints.get('alias_marker', 0) >= 1:
        return False  # Near alias markers (aka, alias, known as)
    
    # Strong person hints (Mr., Dr., etc.) can save it
    if hints.get('person', 0) >= 2:
        return False
    
    # Strong org hints can save it
    if hints.get('org', 0) >= 2:
        return False
    
    # ORG label with moderate org hints - organizations often lowercase in OCR
    if span.label == 'ORG' and hints.get('org', 0) >= 1:
        return False
    
    # GPE/LOC label with strong location hints - places often lowercase in OCR
    if span.label in ('GPE', 'LOC') and hints.get('loc', 0) >= 2:
        return False
    
    # Otherwise, reject single-token lowercase
    return True


def save_ner_candidates(
    conn,
    chunk_id: int,
    document_id: int,
    spans: List[NERSpan],
    dry_run: bool = False,
    alias_matches: Optional[Dict[str, Dict]] = None,
) -> int:
    """
    Save NER spans as new candidates using bulk insert.
    
    IMPORTANT:
    - Uses canonical normalize_surface() for surface_norm
    - Sets quality_score to neutral value (not NULL) for resolver compatibility
    - Sets ner_accept_score separately
    - Uses bulk execute_values with ON CONFLICT for performance + idempotency
    - Requires real unique constraint: mention_candidates_span_unique
    - If alias_matches provided, sets resolved_entity_id for matched spans
    
    Args:
        alias_matches: Dict mapping surface_norm -> match info with entity_id
    """
    if not spans:
        return 0
    
    alias_matches = alias_matches or {}
    
    if dry_run:
        for span in spans:
            surface_norm = normalize_surface(span.surface)
            match = alias_matches.get(surface_norm)
            if match:
                entity_info = f" -> entity={match['entity_id']} ({match['match_type']})"
            else:
                entity_info = ""
            print(f"    [DRY RUN] Would add: {span.surface!r} ({span.label}) score={span.accept_score:.2f}{entity_info}")
        return len(spans)
    
    cur = conn.cursor()
    
    # Prepare records using canonical normalizer
    records = []
    for span in spans:
        surface_norm = normalize_surface(span.surface)
        
        # Check for alias match
        match = alias_matches.get(surface_norm)
        resolved_entity_id = match['entity_id'] if match else None
        resolution_method = f"ner_{match['match_type']}" if match else None
        resolution_score = match['match_score'] if match else None
        
        # Build top_candidates from match info
        top_candidates = None
        if match and match.get('candidate_entities'):
            top_candidates = Json([{
                'entity_id': c['entity_id'],
                'entity_name': c['entity_name'],
                'alias_norm': c['alias_norm'],
                'score': c.get('combined_score', c.get('sim_score', 1.0)),
                'match_type': match['match_type'],
                # Corpus stats from lexicon
                'doc_freq': c.get('doc_freq'),
                'proposal_tier': c.get('proposal_tier'),
                'corpus_confidence': float(c['corpus_confidence']) if c.get('corpus_confidence') else None,
            } for c in match['candidate_entities'][:5]])
        
        records.append((
            chunk_id,
            document_id,
            span.surface,
            surface_norm,
            span.start_char,
            span.end_char,
            NER_DEFAULT_QUALITY_SCORE,
            'spacy',
            span.label,
            span.entity_type,
            span.accept_score,
            Json(span.context_hints),
            resolved_entity_id,
            resolution_method,
            resolution_score,
            top_candidates,
        ))
    
    # Bulk insert with ON CONFLICT
    # Uses the real unique constraint: mention_candidates_span_unique
    # ON CONFLICT: preserve existing values, only fill NER blanks
    # Note: DO UPDATE always "affects" the row, so rowcount includes updates
    # We track new inserts vs updates separately in chunk_ner_runs
    execute_values(
        cur,
        """
        INSERT INTO mention_candidates 
            (chunk_id, document_id, raw_span, surface_norm, char_start, char_end, 
             quality_score, ner_source, ner_label, ner_type_hint, ner_accept_score, 
             ner_context_features, resolved_entity_id, resolution_method, 
             resolution_score, top_candidates)
        VALUES %s
        ON CONFLICT ON CONSTRAINT mention_candidates_span_unique DO UPDATE SET
            ner_source = COALESCE(mention_candidates.ner_source, EXCLUDED.ner_source),
            ner_label = COALESCE(mention_candidates.ner_label, EXCLUDED.ner_label),
            ner_type_hint = COALESCE(mention_candidates.ner_type_hint, EXCLUDED.ner_type_hint),
            ner_accept_score = COALESCE(mention_candidates.ner_accept_score, EXCLUDED.ner_accept_score),
            ner_context_features = COALESCE(mention_candidates.ner_context_features, EXCLUDED.ner_context_features),
            resolved_entity_id = COALESCE(mention_candidates.resolved_entity_id, EXCLUDED.resolved_entity_id),
            resolution_method = COALESCE(mention_candidates.resolution_method, EXCLUDED.resolution_method),
            resolution_score = COALESCE(mention_candidates.resolution_score, EXCLUDED.resolution_score),
            top_candidates = COALESCE(mention_candidates.top_candidates, EXCLUDED.top_candidates)
        """,
        records,
        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    
    # Return count of records attempted
    # Note: This includes both inserts and updates due to DO UPDATE
    # For precise counts, chunk_ner_runs tracks spans_new_inserted separately
    return len(records)


def update_existing_with_ner_signals(
    conn,
    enhanced_candidates: List[Dict],
    dry_run: bool = False,
) -> int:
    """
    Update existing candidates with NER signals.
    
    This ADDS NER info to existing candidates without overwriting
    their quality_score or other fields. Only fills blanks.
    
    Updates: ner_source, ner_label, ner_type_hint, ner_accept_score, ner_context_features
    """
    if not enhanced_candidates:
        return 0
    
    if dry_run:
        for cand in enhanced_candidates:
            print(f"    [DRY RUN] Would enhance: {cand.get('surface')!r} with NER={cand.get('ner_label')}")
        return len(enhanced_candidates)
    
    cur = conn.cursor()
    
    for cand in enhanced_candidates:
        if 'id' in cand:
            # Only fill blanks - preserve existing NER signals if any
            # Include ner_context_features for consistency with save_ner_candidates
            context_features = cand.get('ner_context_features')
            cur.execute("""
                UPDATE mention_candidates SET
                    ner_source = COALESCE(ner_source, 'spacy'),
                    ner_label = COALESCE(ner_label, %s),
                    ner_type_hint = COALESCE(ner_type_hint, %s),
                    ner_accept_score = COALESCE(ner_accept_score, %s),
                    ner_context_features = COALESCE(ner_context_features, %s)
                WHERE id = %s
            """, (
                cand.get('ner_label'),
                cand.get('ner_type_hint'),
                cand.get('ner_accept_score'),
                Json(context_features) if context_features else None,
                cand['id']
            ))
    
    return len(enhanced_candidates)


def save_ner_date_signals(
    conn,
    chunk_id: int,
    document_id: int,
    dates: List[DateSpan],
    model_name: str,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Save extracted dates to ner_date_signals table (NOT date_mentions).
    
    NER dates are signals that require verification before promotion
    to the authoritative date_mentions table.
    
    Uses bulk insert with ON CONFLICT for idempotency.
    
    Returns:
        Tuple of (attempted_count, actual_inserted_count)
    """
    if not dates:
        return 0, 0
    
    if dry_run:
        for d in dates:
            print(f"    [DRY RUN] Would add date signal: {d.surface!r} -> {d.date_start} to {d.date_end} ({d.precision})")
        return len(dates), len(dates)
    
    cur = conn.cursor()
    
    # Prepare records with parser payload for debugging
    records = []
    for d in dates:
        surface_norm = normalize_surface(d.surface)
        # Build parser payload for debugging / downstream promotion
        parser_payload = {
            'original_surface': d.surface,
            'ner_label': d.label,
            'char_start': d.start_char,
            'char_end': d.end_char,
            'precision': d.precision,
            'confidence': d.confidence,
        }
        if d.date_start:
            parser_payload['parsed_start'] = d.date_start
        if d.date_end:
            parser_payload['parsed_end'] = d.date_end
        
        records.append((
            chunk_id,
            document_id,
            d.surface,
            surface_norm,
            d.start_char,
            d.end_char,
            d.label,  # ner_label (DATE or TIME)
            model_name,
            'spacy_ent',  # raw_parser
            Json(parser_payload),  # parser_payload
            d.date_start,
            d.date_end,
            d.precision,
            d.confidence,
        ))
    
    # Bulk insert with ON CONFLICT DO NOTHING using named constraint
    # Use execute_values with fetch=True to get accurate insert count
    # (cur.rowcount and fetchall() are unreliable with execute_values)
    try:
        rows = execute_values(
            cur,
            """
            INSERT INTO ner_date_signals 
                (chunk_id, document_id, surface, surface_norm, start_char, end_char,
                 ner_label, ner_model, raw_parser, parser_payload,
                 parsed_date_start, parsed_date_end, parsed_precision, parse_confidence)
            VALUES %s
            ON CONFLICT ON CONSTRAINT ner_date_signals_span_unique DO NOTHING
            RETURNING 1
            """,
            records,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            fetch=True
        )
        inserted = len(rows) if rows else 0
    except TypeError:
        # Older psycopg2 doesn't support fetch=True
        # Fall back to attempted count (not ideal but better than 0)
        execute_values(
            cur,
            """
            INSERT INTO ner_date_signals 
                (chunk_id, document_id, surface, surface_norm, start_char, end_char,
                 ner_label, ner_model, raw_parser, parser_payload,
                 parsed_date_start, parsed_date_end, parsed_precision, parse_confidence)
            VALUES %s
            ON CONFLICT ON CONSTRAINT ner_date_signals_span_unique DO NOTHING
            """,
            records,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        # Can't get accurate count without fetch=True; use -1 to signal "unknown"
        inserted = -1
    
    return len(records), inserted


def main():
    global _STOP_REQUESTED
    parser = argparse.ArgumentParser(description='Run NER extraction on documents')
    parser.add_argument('--collection', help='Collection to process')
    parser.add_argument('--doc-ids', help='Comma-separated document IDs')
    parser.add_argument('--all', action='store_true', 
                       help='Confirm processing all documents (required if no --collection or --doc-ids)')
    parser.add_argument('--limit', type=int, help='Limit number of chunks')
    parser.add_argument('--batch-size', type=int, default=50, help='SpaCy batch size')
    parser.add_argument('--model', default='en_core_web_lg', help='SpaCy model name')
    parser.add_argument('--threshold', type=float, default=0.5, help='NER acceptance threshold')
    parser.add_argument('--strict-threshold', type=float, default=NER_STRICT_THRESHOLD,
                       help='Stricter threshold for NER-only candidates (secondary proposer)')
    parser.add_argument('--no-dates', action='store_true', 
                       help='Skip date extraction (dates extracted by default)')
    parser.add_argument('--match-aliases', action='store_true', default=True,
                       help='Match NER spans against alias lexicon (default: True)')
    parser.add_argument('--no-match-aliases', action='store_true',
                       help='Skip alias matching (just extract raw NER spans)')
    parser.add_argument('--fuzzy-threshold', type=float, default=0.4,
                       help='Similarity threshold for fuzzy alias matching (default: 0.4)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--reprocess', action='store_true', 
                       help='Reprocess all chunks (disable skip-processed)')
    args = parser.parse_args()

    _install_interrupt_handlers()
    
    # Skip-processed is default ON; --reprocess turns it OFF
    args.skip_processed = not args.reprocess
    
    # Derive extract_dates from --no-dates flag
    args.extract_dates = not args.no_dates
    
    # Derive match_aliases from --no-match-aliases flag
    args.match_aliases = not args.no_match_aliases
    
    if not SPACY_AVAILABLE:
        print("ERROR: spaCy not installed. Run:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_lg")
        sys.exit(1)
    
    if not args.collection and not args.doc_ids and not args.all:
        print("ERROR: Specify --collection, --doc-ids, or --all")
        sys.exit(1)
    
    print("=" * 60)
    print("NER EXTRACTION + ALIAS MATCHING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Strict threshold (new candidates): {args.strict_threshold}")
    print(f"Extract dates: {args.extract_dates}")
    print(f"Match aliases: {args.match_aliases}" + (f" (fuzzy threshold: {args.fuzzy_threshold})" if args.match_aliases else ""))
    print(f"Batch size: {args.batch_size}")
    print(f"Pipeline version: {PIPELINE_VERSION}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Initialize
    conn = get_conn()
    
    # Check schema - run even in dry-run mode (warn only in dry-run)
    schema_ok = check_ner_schema(conn, warn_only=args.dry_run, check_dates=args.extract_dates)
    if not schema_ok and args.dry_run:
        print("Continuing dry-run despite missing schema...\n")
    
    # Initialize guardrails (critical safety checks)
    guardrails = NERGuardrails(conn)
    print("Guardrails initialized")
    
    print("Loading spaCy model...")
    extractor = NERExtractor(model_name=args.model)
    print(f"Model loaded: {args.model}")
    
    # Get chunks to process
    doc_ids = None
    if args.doc_ids:
        doc_ids = [int(x.strip()) for x in args.doc_ids.split(',')]
    
    print("\nFetching chunks...")
    chunks = get_chunks_for_processing(
        conn,
        collection=args.collection,
        doc_ids=doc_ids,
        limit=args.limit,
        model=args.model,
        threshold=args.threshold,
        skip_processed=args.skip_processed,
        require_dates=args.extract_dates,  # If dates wanted, don't skip chunks missing dates
    )
    print(f"Found {len(chunks)} chunks to process")
    if chunks:
        chunk_ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id") is not None]
        if chunk_ids:
            print(f"Chunk id range in this run: {min(chunk_ids)} .. {max(chunk_ids)}")
            if args.limit:
                print(
                    "Note: --limit caps the number of *unprocessed* chunks selected. "
                    "If you re-run without --reprocess, it will keep advancing to the next chunks."
                )
    
    if not chunks:
        print("No chunks to process.")
        return
    
    # Process chunks
    stats = {
        'chunks_processed': 0,
        'raw_ner_spans': 0,
        'candidate_upserts': 0,  # Includes both new inserts and updates
        'enhanced_existing': 0,
        'filtered_overlap': 0,
        'filtered_threshold': 0,
        'filtered_lowercase': 0,
        'alias_exact_matches': 0,
        'alias_fuzzy_matches': 0,
        'alias_no_match': 0,
        'date_spans_extracted': 0,
        'date_signals_saved': 0,
    }
    
    print("\nProcessing...")
    
    # Track chunks for batch commit
    chunks_since_commit = 0
    
    # Process in batches
    batch_texts = []
    batch_chunks = []
    
    for i, chunk in enumerate(chunks):
        if _STOP_REQUESTED:
            break

        batch_texts.append(chunk['text'] or '')
        batch_chunks.append(chunk)
        
        # Process batch when full or at end
        if _STOP_REQUESTED or len(batch_texts) >= args.batch_size or i == len(chunks) - 1:
            try:
                # Extract NER spans (entities)
                batch_results = extractor.extract_batch(
                    batch_texts,
                    batch_size=args.batch_size,
                    acceptance_threshold=args.threshold,
                )
                
                # Extract dates if enabled
                batch_dates = None
                if args.extract_dates:
                    batch_dates = extractor.extract_dates_batch(
                        batch_texts,
                        batch_size=args.batch_size,
                    )
                
                # Process each chunk's results
                for idx, (chunk_info, spans) in enumerate(zip(batch_chunks, batch_results)):
                    chunk_id = chunk_info['chunk_id']
                    doc_id = chunk_info['document_id']
                    
                    raw_count = len(spans)
                    stats['raw_ner_spans'] += raw_count
                    
                    # Get existing candidates for deduplication
                    existing = get_existing_candidates(conn, chunk_id)
                    
                    # Dedupe against existing candidates
                    non_overlapping_spans, enhanced_existing = dedupe_ner_against_candidates(spans, existing)
                    dropped_overlap = len(spans) - len(non_overlapping_spans)
                    stats['filtered_overlap'] += dropped_overlap
                    
                    # Apply STRICTER threshold for new NER-only candidates
                    above_threshold_spans = [s for s in non_overlapping_spans if s.accept_score >= args.strict_threshold]
                    dropped_threshold = len(non_overlapping_spans) - len(above_threshold_spans)
                    stats['filtered_threshold'] += dropped_threshold
                    
                    # Filter single-token lowercase junk (dictionary word flood prevention)
                    accepted_new_spans = [s for s in above_threshold_spans if not is_single_token_lowercase_junk(s)]
                    dropped_lowercase = len(above_threshold_spans) - len(accepted_new_spans)
                    stats['filtered_lowercase'] += dropped_lowercase
                    
                    # Match against alias lexicon if enabled
                    matched_spans = []
                    unmatched_spans = accepted_new_spans
                    
                    if args.match_aliases and accepted_new_spans:
                        matched_spans, unmatched_spans = match_spans_against_lexicon(
                            conn, 
                            accepted_new_spans,
                            use_fuzzy=True,
                            fuzzy_threshold=args.fuzzy_threshold,
                        )
                        
                        # Track match stats
                        for m in matched_spans:
                            if m['match_type'] == 'exact':
                                stats['alias_exact_matches'] += 1
                            else:
                                stats['alias_fuzzy_matches'] += 1
                        stats['alias_no_match'] += len(unmatched_spans)
                    
                    # ==========================================================
                    # APPLY GUARDRAILS - Critical safety checks
                    # ==========================================================
                    auto_link_spans = []
                    queue_spans = []
                    rejected_by_guardrail = 0
                    
                    for m in matched_spans:
                        surface_norm = m['surface_norm']
                        
                        # Check if can auto-link
                        can_link, reason = guardrails.can_auto_link(surface_norm, {
                            'match_type': m['match_type'],
                            'match_score': m['match_score'],
                            'entity_id': m['entity_id'],
                            'alias_class': m.get('alias_kind'),
                            'is_auto_match': m.get('is_auto_match', False),
                        })
                        
                        if can_link:
                            m['auto_link'] = True
                            m['link_reason'] = reason
                            auto_link_spans.append(m)
                        else:
                            # Can't auto-link, check if should queue
                            m['auto_link'] = False
                            m['no_link_reason'] = reason
                            should_queue, q_reason = guardrails.should_queue_for_review(
                                surface_norm,
                                {'doc_count': m.get('doc_freq', 1), 'mention_count': 1, 
                                 'ner_score': m['span'].accept_score}
                            )
                            if should_queue:
                                queue_spans.append(m)
                            else:
                                rejected_by_guardrail += 1
                    
                    # Check unmatched spans against queue guardrails
                    for span in unmatched_spans:
                        surface_norm = normalize_surface(span.surface)
                        should_queue, reason = guardrails.should_queue_for_review(
                            surface_norm,
                            {'doc_count': 1, 'mention_count': 1, 'ner_score': span.accept_score}
                        )
                        if not should_queue:
                            rejected_by_guardrail += 1
                    
                    stats['rejected_by_guardrail'] = stats.get('rejected_by_guardrail', 0) + rejected_by_guardrail
                    stats['auto_linked'] = stats.get('auto_linked', 0) + len(auto_link_spans)
                    stats['queued_for_review'] = stats.get('queued_for_review', 0) + len(queue_spans)
                    
                    # Log in dry-run
                    if args.dry_run and (matched_spans or unmatched_spans):
                        for m in auto_link_spans:
                            entity_info = f"entity={m['entity_id']}:{m['entity_name']}"
                            match_info = f"{m['match_type']}@{m['match_score']:.2f}"
                            corpus_info = ""
                            if m.get('doc_freq'):
                                corpus_info = f" [docs={m['doc_freq']}, tier={m.get('proposal_tier', '?')}]"
                            print(f"    [AUTO-LINK] {m['span'].surface!r} -> {entity_info} ({match_info}){corpus_info}")
                        
                        for m in queue_spans:
                            entity_info = f"entity={m['entity_id']}:{m['entity_name']}"
                            print(f"    [QUEUE] {m['span'].surface!r} -> {entity_info} (reason: {m.get('no_link_reason', '?')})")
                        
                        if rejected_by_guardrail > 0:
                            print(f"    [GUARDRAIL] Rejected {rejected_by_guardrail} spans")
                    
                    # Save new candidates (both matched and unmatched)
                    entity_upserts = 0
                    if accepted_new_spans:
                        # Pass match info to save function
                        entity_upserts = save_ner_candidates(
                            conn, chunk_id, doc_id, accepted_new_spans, args.dry_run,
                            alias_matches={normalize_surface(m['span'].surface): m for m in matched_spans}
                        )
                        stats['candidate_upserts'] += entity_upserts
                    
                    # Update existing with NER signals
                    enhanced_count = 0
                    if enhanced_existing:
                        enhanced_count = update_existing_with_ner_signals(conn, enhanced_existing, args.dry_run)
                        stats['enhanced_existing'] += enhanced_count
                    
                    # Extract and save date signals (NOT date_mentions - NER is not authoritative)
                    date_spans_extracted = 0
                    date_spans_saved = 0
                    if batch_dates:
                        dates = batch_dates[idx]
                        if dates:
                            date_spans_extracted = len(dates)
                            stats['date_spans_extracted'] += date_spans_extracted
                            date_attempted, date_inserted = save_ner_date_signals(
                                conn, chunk_id, doc_id, dates, args.model, args.dry_run
                            )
                            # date_inserted is -1 if fetch=True not supported (older psycopg2)
                            if date_inserted < 0:
                                # For display: use attempted as estimate
                                stats['date_signals_saved'] += date_attempted
                                stats['date_insert_count_estimated'] = True
                                # For DB: store 0 (unknown) - don't persist misleading data
                                # The important info (date_spans_extracted, dates_enabled) is accurate
                                date_spans_saved = 0
                            else:
                                date_spans_saved = date_inserted
                                stats['date_signals_saved'] += date_inserted
                    
                    # Record the run with detailed stats (entity + date)
                    if not args.dry_run:
                        record_ner_run(
                            conn, chunk_id, args.model, args.threshold,
                            spans_extracted=raw_count,
                            spans_upserted=entity_upserts,  # "upserted" not "new_inserted" - honest naming
                            spans_enhanced_existing=enhanced_count,
                            spans_dropped_overlap=dropped_overlap,
                            spans_dropped_filters=dropped_threshold + dropped_lowercase,
                            date_spans_extracted=date_spans_extracted,
                            date_spans_saved=date_spans_saved,
                            dates_enabled=args.extract_dates,
                        )
                    
                    stats['chunks_processed'] += 1
                    chunks_since_commit += 1
                    
                    # Progress
                    if stats['chunks_processed'] % 100 == 0:
                        print(f"  Processed {stats['chunks_processed']}/{len(chunks)} chunks...")
                
                # Batch commit for performance
                if not args.dry_run and chunks_since_commit >= COMMIT_BATCH_SIZE:
                    conn.commit()
                    chunks_since_commit = 0
                    
                # If user requested stop, commit and exit cleanly after finishing this batch
                if _STOP_REQUESTED:
                    if not args.dry_run:
                        conn.commit()
                    print("\n[INTERRUPT] Exiting cleanly. Re-run the same command to resume.")
                    conn.close()
                    return

            except KeyboardInterrupt:
                # Treat as graceful stop; commit what we've completed so far.
                _STOP_REQUESTED = True
                if not args.dry_run:
                    conn.commit()
                print("\n[INTERRUPT] Exiting cleanly. Re-run the same command to resume.")
                conn.close()
                return
            except Exception as e:
                # Rollback on error, log, and re-raise
                if not args.dry_run:
                    conn.rollback()
                print(f"\nERROR processing batch at chunk {i}: {e}")
                print(f"  Rolled back transaction. {stats['chunks_processed']} chunks completed before error.")
                raise
            
            # Clear batch
            batch_texts = []
            batch_chunks = []
    
    # Final commit
    if not args.dry_run:
        conn.commit()
    
    # Print stats
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Chunks processed: {stats['chunks_processed']}")
    print(f"Raw NER spans: {stats['raw_ner_spans']}")
    print(f"Filtered (overlap with existing): {stats['filtered_overlap']}")
    print(f"Filtered (below strict threshold): {stats['filtered_threshold']}")
    print(f"Filtered (single-token lowercase): {stats['filtered_lowercase']}")
    print(f"Candidate spans upserted (new+updated): {stats['candidate_upserts']}")
    print(f"Existing candidates enhanced (via update): {stats['enhanced_existing']}")
    
    if args.match_aliases:
        total_matched = stats['alias_exact_matches'] + stats['alias_fuzzy_matches']
        total_candidates = total_matched + stats['alias_no_match']
        if total_candidates > 0:
            match_rate = (total_matched / total_candidates) * 100
            print(f"\nAlias matching:")
            print(f"  Exact matches: {stats['alias_exact_matches']}")
            print(f"  Fuzzy matches: {stats['alias_fuzzy_matches']}")
            print(f"  No match (new proposals): {stats['alias_no_match']}")
            print(f"  Match rate: {match_rate:.1f}%")
        
        # Guardrail stats
        print(f"\nGuardrail decisions:")
        print(f"  Auto-linked (safe): {stats.get('auto_linked', 0)}")
        print(f"  Queued for review: {stats.get('queued_for_review', 0)}")
        print(f"  Rejected by guardrails: {stats.get('rejected_by_guardrail', 0)}")
    
    if args.extract_dates:
        print(f"\nDate extraction:")
        print(f"  Date spans extracted (raw): {stats['date_spans_extracted']}")
        estimated_note = " [estimated - older psycopg2]" if stats.get('date_insert_count_estimated') else ""
        print(f"  Date signals saved: {stats['date_signals_saved']}{estimated_note} (to ner_date_signals, pending review)")
    
    print("\nExtractor filter stats:")
    for key, val in extractor.stats.items():
        print(f"  {key}: {val}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made to database.")
    
    conn.close()


if __name__ == '__main__':
    main()
