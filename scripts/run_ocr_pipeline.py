#!/usr/bin/env python3
"""
OCR Pipeline Orchestrator

Single command to run the complete OCR extraction pipeline:
1. Check/refresh lexicon
2. Extract candidates
3. Resolve in batches
4. Report metrics

Supports resumption and incremental processing.

Usage:
    python scripts/run_ocr_pipeline.py --collection silvermaster --max-chunks 1000
    python scripts/run_ocr_pipeline.py --collection rosenberg --resume
    python scripts/run_ocr_pipeline.py --all-ocr --max-chunks 5000
"""

import argparse
import sys
import time
import uuid
from datetime import datetime
from typing import Optional, Dict

import psycopg2
from psycopg2.extras import Json

sys.path.insert(0, '.')


def _progress_bar(current: int, total: Optional[int], width: int = 32) -> str:
    """Render a simple ASCII progress bar."""
    if total is None or total <= 0:
        return f"[{'?' * min(width, 10)}]"
    current = max(0, min(current, total))
    filled = int(round(width * (current / total))) if total else 0
    filled = max(0, min(filled, width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _run_subprocess_streaming(cmd: list[str], on_line=None) -> tuple[int, str]:
    """
    Run a subprocess, stream stdout/stderr live, and optionally parse lines.

    Returns: (returncode, combined_output)
    """
    import subprocess

    # Merge stderr into stdout so we don't deadlock on separate pipes.
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    out_lines: list[str] = []
    assert p.stdout is not None
    for raw_line in p.stdout:
        line = raw_line.rstrip("\n")
        out_lines.append(line)
        if on_line is not None:
            try:
                on_line(line)
            except Exception:
                # Never let progress parsing crash the pipeline.
                pass
        print(line)

    rc = p.wait()
    return rc, "\n".join(out_lines)


# OCR-heavy collections (vs clean text)
OCR_COLLECTIONS = ['silvermaster', 'rosenberg', 'solo', 'fbicomrap']


def get_conn():
    return psycopg2.connect(
        host='localhost', port=5432, dbname='neh', user='neh', password='neh'
    )


def check_lexicon_freshness(conn, lexicon_source_slug: Optional[str] = None) -> Dict:
    """Check if lexicon needs refresh.

    If lexicon_source_slug is provided, compares lexicon size against the number
    of matchable aliases for that concordance source only (so we don't
    constantly "refresh" due to historical aliases from older ingests).
    """
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*), MAX(updated_at) FROM alias_lexicon_index")
    row = cur.fetchone()
    count, last_update = row
    
    alias_count = None
    if lexicon_source_slug:
        # Best-effort: only if entity_aliases has source_id and concordance_sources exists.
        try:
            cur.execute("""
                SELECT COUNT(*)
                FROM entity_aliases ea
                JOIN concordance_sources cs ON cs.id = ea.source_id
                WHERE ea.is_matchable = TRUE
                  AND cs.slug = %s
            """, (lexicon_source_slug,))
            alias_count = cur.fetchone()[0]
        except Exception:
            alias_count = None

    if alias_count is None:
        cur.execute("SELECT COUNT(*) FROM entity_aliases WHERE is_matchable = TRUE")
        alias_count = cur.fetchone()[0]
    
    return {
        'lexicon_count': count or 0,
        'alias_count': alias_count,
        'last_update': last_update,
        'needs_refresh': count == 0 or count < alias_count * 0.9
    }


def refresh_lexicon(lexicon_source_slug: Optional[str] = None):
    """Refresh the alias lexicon."""
    print("\n=== Refreshing Alias Lexicon ===")
    import subprocess
    cmd = ['python', 'scripts/build_alias_lexicon.py', '--rebuild']
    if lexicon_source_slug:
        cmd.extend(['--source-slug', lexicon_source_slug])
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def get_pipeline_status(conn, collection: Optional[str] = None) -> Dict:
    """Get current pipeline status."""
    cur = conn.cursor()
    
    # Count chunks to process
    if collection:
        cur.execute("""
            SELECT COUNT(DISTINCT c.id)
            FROM chunks c
            JOIN chunk_pages cp ON cp.chunk_id = c.id
            JOIN pages p ON p.id = cp.page_id
            JOIN documents d ON d.id = p.document_id
            JOIN collections col ON col.id = d.collection_id
            WHERE col.slug = %s
        """, (collection,))
    else:
        cur.execute("""
            SELECT COUNT(DISTINCT c.id)
            FROM chunks c
            JOIN chunk_pages cp ON cp.chunk_id = c.id
            JOIN pages p ON p.id = cp.page_id
            JOIN documents d ON d.id = p.document_id
            JOIN collections col ON col.id = d.collection_id
            WHERE col.slug = ANY(%s)
        """, (OCR_COLLECTIONS,))
    
    total_chunks = cur.fetchone()[0]
    
    # Candidates by status
    cond = "WHERE col.slug = %s" if collection else "WHERE col.slug = ANY(%s)"
    param = collection if collection else OCR_COLLECTIONS
    
    cur.execute(f"""
        SELECT mc.resolution_status, COUNT(*)
        FROM mention_candidates mc
        JOIN documents d ON d.id = mc.document_id
        JOIN collections col ON col.id = d.collection_id
        {cond}
        GROUP BY mc.resolution_status
    """, (param,))
    
    status_counts = dict(cur.fetchall())
    
    return {
        'total_chunks': total_chunks,
        'candidates_pending': status_counts.get('pending', 0),
        'candidates_resolved': status_counts.get('resolved', 0),
        'candidates_queued': status_counts.get('queue', 0),
        'candidates_ignored': status_counts.get('ignore', 0),
        'total_candidates': sum(status_counts.values())
    }


def create_run_record(conn, config: dict) -> str:
    """Create an extraction run record."""
    cur = conn.cursor()
    batch_id = f"ocr_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    cur.execute("""
        INSERT INTO ocr_extraction_runs (batch_id, collection_slug, config, status)
        VALUES (%s, %s, %s, 'running')
        RETURNING id
    """, (batch_id, config.get('collection'), Json(config)))
    
    conn.commit()
    return batch_id


def update_run_record(conn, batch_id: str, stats: dict, status: str = 'completed'):
    """Update run record with final stats."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE ocr_extraction_runs SET
            status = %s,
            completed_at = NOW(),
            chunks_processed = %s,
            candidates_generated = %s,
            candidates_resolved = %s,
            candidates_queued = %s,
            candidates_ignored = %s,
            link_rate = %s,
            queue_rate = %s,
            junk_rate = %s
        WHERE batch_id = %s
    """, (
        status,
        stats.get('chunks_processed', 0),
        stats.get('candidates_generated', 0),
        stats.get('resolved', 0),
        stats.get('queued', 0),
        stats.get('ignored', 0),
        stats.get('link_rate'),
        stats.get('queue_rate'),
        stats.get('junk_rate'),
        batch_id
    ))
    conn.commit()


def run_candidate_extraction(
    collection: Optional[str],
    max_chunks: Optional[int],
    batch_id: str
) -> Dict:
    """Run candidate extraction phase."""
    print("\n=== Phase 1: Candidate Extraction ===")
    
    # Use -u for unbuffered output so progress lines appear immediately
    cmd = ['python', '-u', 'scripts/extract_ocr_candidates.py']
    if collection:
        cmd.extend(['--collection', collection])
    if max_chunks:
        cmd.extend(['--limit', str(max_chunks)])
    
    # Stream output live + parse progress for a progress bar.
    start_time = time.time()
    progress = {
        "chunks_total": None,
        "chunks_done": 0,
        "candidates": 0,
    }

    def on_line(line: str):
        # Examples:
        #   Found 1234 chunks to process
        #   Processed 100/1234 chunks, 5678 candidates...
        nonlocal progress
        if line.startswith("Found ") and " chunks to process" in line:
            try:
                n = int(line.split("Found ", 1)[1].split(" chunks", 1)[0].strip())
                progress["chunks_total"] = n
            except Exception:
                return

        if "Processed " in line and " chunks, " in line and " candidates" in line:
            try:
                # "  Processed X/Y chunks, Z candidates..."
                after = line.split("Processed ", 1)[1].strip()
                xy = after.split(" chunks", 1)[0].strip()  # "X/Y"
                x_str, y_str = xy.split("/", 1)
                x = int(x_str)
                y = int(y_str)
                progress["chunks_done"] = x
                progress["chunks_total"] = y

                after_chunks = after.split("chunks, ", 1)[1]
                z_str = after_chunks.split(" candidates", 1)[0].strip()
                progress["candidates"] = int(z_str)

                elapsed = max(0.1, time.time() - start_time)
                rate = x / elapsed
                bar = _progress_bar(x, y)
                pct = (100.0 * x / y) if y else 0.0
                print(f"    {bar} {pct:5.1f}%  chunks={x:,}/{y:,}  cand={progress['candidates']:,}  ({rate:.1f} chunks/s)")
            except Exception:
                return

    rc, combined = _run_subprocess_streaming(cmd, on_line=on_line)
    if rc != 0:
        raise RuntimeError(f"Candidate extraction failed (exit code {rc}).")

    # Parse output for stats (basic)
    stats = {'chunks_processed': 0, 'candidates_generated': 0}
    for line in combined.split('\n'):
        if 'Chunks processed:' in line:
            try:
                stats['chunks_processed'] = int(line.split(':')[1].strip())
            except:
                pass
        if 'Candidates generated:' in line:
            try:
                stats['candidates_generated'] = int(line.split(':')[1].strip())
            except:
                pass
    
    return stats


def count_pending_candidates(conn, collection: Optional[str]) -> int:
    """Count pending candidates for progress reporting."""
    with conn.cursor() as cur:
        if collection:
            cur.execute("""
                SELECT COUNT(*)
                FROM mention_candidates mc
                JOIN documents d ON d.id = mc.document_id
                JOIN collections col ON col.id = d.collection_id
                WHERE col.slug = %s AND mc.resolution_status = 'pending'
            """, (collection,))
        else:
            cur.execute("""
                SELECT COUNT(*)
                FROM mention_candidates mc
                JOIN documents d ON d.id = mc.document_id
                JOIN collections col ON col.id = d.collection_id
                WHERE col.slug = ANY(%s) AND mc.resolution_status = 'pending'
            """, (OCR_COLLECTIONS,))
        return int(cur.fetchone()[0] or 0)


def run_resolution(
    conn,
    collection: Optional[str],
    batch_size: int = 500
) -> Dict:
    """Run resolution phase."""
    print("\n=== Phase 2: Resolution ===")
    
    # Use -u for unbuffered output so progress lines appear immediately
    cmd = ['python', '-u', 'scripts/resolve_ocr_candidates_v2.py', '--batch-size', str(batch_size)]
    if collection:
        cmd.extend(['--collection', collection])
    
    total_pending = count_pending_candidates(conn, collection)
    if total_pending:
        print(f"Pending candidates to resolve: {total_pending:,}")

    start_time = time.time()
    processed = {"done": 0, "total": total_pending}

    def on_line(line: str):
        # Example:
        #   "  Batch 12: 500 candidates in 3.2s (155/sec)"
        if line.strip().startswith("Batch ") and " candidates in " in line:
            try:
                # Extract K from "Batch N: K candidates ..."
                after_colon = line.split(":", 1)[1].strip()
                k_str = after_colon.split(" candidates", 1)[0].strip()
                k = int(k_str)
                processed["done"] += k

                total = processed["total"] if processed["total"] > 0 else None
                bar = _progress_bar(processed["done"], total) if total else _progress_bar(processed["done"], None)
                pct = (100.0 * processed["done"] / processed["total"]) if processed["total"] else 0.0
                elapsed = max(0.1, time.time() - start_time)
                rate = processed["done"] / elapsed
                if processed["total"]:
                    print(f"    {bar} {pct:5.1f}%  resolved_batches_done={processed['done']:,}/{processed['total']:,}  ({rate:.0f}/s)")
                else:
                    print(f"    {bar} done={processed['done']:,}  ({rate:.0f}/s)")
            except Exception:
                return

    rc, combined = _run_subprocess_streaming(cmd, on_line=on_line)
    if rc != 0:
        raise RuntimeError(f"Resolution failed (exit code {rc}).")
    
    # Parse output
    stats = {'resolved': 0, 'queued': 0, 'ignored': 0, 'total': 0}
    for line in combined.split('\n'):
        if 'Resolved:' in line and '%' in line:
            try:
                stats['resolved'] = int(line.split(':')[1].split('(')[0].strip())
            except:
                pass
        if 'Queued:' in line and '%' in line:
            try:
                stats['queued'] = int(line.split(':')[1].split('(')[0].strip())
            except:
                pass
        if 'Ignored:' in line and '%' in line:
            try:
                stats['ignored'] = int(line.split(':')[1].split('(')[0].strip())
            except:
                pass
        if 'Total:' in line:
            try:
                stats['total'] = int(line.split(':')[1].strip())
            except:
                pass
    
    # Compute rates
    total_non_junk = stats['resolved'] + stats['queued']
    if total_non_junk > 0:
        stats['link_rate'] = stats['resolved'] / total_non_junk
    if stats['total'] > 0:
        stats['queue_rate'] = stats['queued'] / stats['total']
        stats['junk_rate'] = stats['ignored'] / stats['total']
    
    return stats


def print_final_metrics(conn, collection: Optional[str], start_time: float):
    """Print final pipeline metrics."""
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    status = get_pipeline_status(conn, collection)
    
    print(f"\nCollection: {collection or 'all OCR'}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()
    print("Candidate Status:")
    print(f"  Pending:  {status['candidates_pending']}")
    print(f"  Resolved: {status['candidates_resolved']}")
    print(f"  Queued:   {status['candidates_queued']}")
    print(f"  Ignored:  {status['candidates_ignored']}")
    print(f"  Total:    {status['total_candidates']}")
    
    # Link rate
    resolved = status['candidates_resolved']
    queued = status['candidates_queued']
    if resolved + queued > 0:
        print(f"\nLink Rate: {100 * resolved / (resolved + queued):.1f}%")
    
    # Entity mentions from OCR
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM entity_mentions WHERE method = 'ocr_lexicon'
    """)
    ocr_mentions = cur.fetchone()[0]
    print(f"\nOCR Entity Mentions: {ocr_mentions}")
    
    # Review queue items
    cur.execute("""
        SELECT COUNT(*) FROM mention_review_queue WHERE status = 'pending'
    """)
    pending_review = cur.fetchone()[0]
    print(f"Pending Review Items: {pending_review}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='OCR Pipeline Orchestrator')
    parser.add_argument('--collection', help='Collection slug to process')
    parser.add_argument('--all-ocr', action='store_true', help='Process all OCR collections')
    parser.add_argument('--max-chunks', type=int, help='Maximum chunks to process')
    parser.add_argument('--batch-size', type=int, default=500, help='Resolution batch size')
    parser.add_argument('--lexicon-source-slug', type=str, default=None,
                        help='If set, refresh/check alias lexicon using only aliases from this concordance source slug')
    parser.add_argument('--skip-lexicon', action='store_true', help='Skip lexicon refresh check')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip extraction (resolve existing)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing candidates')
    parser.add_argument('--status', action='store_true', help='Show status only')
    args = parser.parse_args()
    
    conn = get_conn()
    start_time = time.time()
    
    # Validate args
    if not args.collection and not args.all_ocr and not args.status:
        parser.error("Specify --collection or --all-ocr")
    
    collection = args.collection if args.collection else None
    
    print("=" * 70)
    print("OCR EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Collection: {collection or 'all OCR'}")
    print(f"Max chunks: {args.max_chunks or 'unlimited'}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Status check
    if args.status:
        status = get_pipeline_status(conn, collection)
        print("Current Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")
        conn.close()
        return
    
    # Check lexicon
    if not args.skip_lexicon:
        lex_status = check_lexicon_freshness(conn, lexicon_source_slug=args.lexicon_source_slug)
        print(f"Lexicon: {lex_status['lexicon_count']} entries (aliases: {lex_status['alias_count']})")
        if lex_status['needs_refresh']:
            print("  Lexicon needs refresh...")
            if not refresh_lexicon(args.lexicon_source_slug):
                print("ERROR: Lexicon refresh failed")
                sys.exit(1)
        else:
            print("  Lexicon is current.")
    
    # Create run record
    config = {
        'collection': collection,
        'all_ocr': args.all_ocr,
        'max_chunks': args.max_chunks,
        'batch_size': args.batch_size,
        'resume': args.resume,
        'lexicon_source_slug': args.lexicon_source_slug,
    }
    batch_id = create_run_record(conn, config)
    print(f"\nRun ID: {batch_id}")
    
    try:
        # Phase 1: Extraction
        extraction_stats = {}
        if not args.skip_extraction and not args.resume:
            extraction_stats = run_candidate_extraction(
                collection, args.max_chunks, batch_id
            )
        else:
            print("\n=== Skipping extraction (using existing candidates) ===")
        
        # Phase 2: Resolution
        resolution_stats = run_resolution(conn, collection, args.batch_size)
        
        # Combine stats
        final_stats = {
            **extraction_stats,
            **resolution_stats
        }
        
        # Update run record
        update_run_record(conn, batch_id, final_stats, 'completed')
        
        # Final metrics
        print_final_metrics(conn, collection, start_time)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        update_run_record(conn, batch_id, {}, 'failed')
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
