#!/usr/bin/env python3
"""
Compute document frequency (DF) for each alias_norm per document_id.

DF = how many chunks (per document) contain this alias_norm.
Stores results in alias_stats table for use in extraction filtering.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn
from retrieval.entity_resolver import normalize_alias


def compute_alias_frequency(conn, collection_slug: str = None, source_slug: str = None):
    """
    Compute document frequency (DF) for each alias_norm:
    - Per document_id: how many chunks contain this token/span
    - Store in alias_stats table
    """
    with conn.cursor() as cur:
        # Create alias_stats table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alias_stats (
                alias_norm TEXT NOT NULL,
                document_id INTEGER NOT NULL,
                df_chunks INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                df_percent NUMERIC NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (alias_norm, document_id)
            )
        """)
        
        # Create index for faster lookups
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_alias_stats_alias_norm 
            ON alias_stats(alias_norm)
        """)
        
        # Get all aliases to check (optionally filtered by source)
        # Note: derived aliases (surname/acronym) have entry_id=NULL, so we need to handle that
        if source_slug:
            # First check if source exists
            cur.execute("SELECT id FROM concordance_sources WHERE slug = %s", (source_slug,))
            source_row = cur.fetchone()
            if not source_row:
                print(f"ERROR: Source slug '{source_slug}' not found in database", file=sys.stderr)
                print("Available sources:", file=sys.stderr)
                cur.execute("SELECT slug FROM concordance_sources ORDER BY slug")
                for (slug,) in cur.fetchall():
                    print(f"  - {slug}", file=sys.stderr)
                return
            source_id = source_row[0]
            
            # For derived aliases (entry_id IS NULL), filter by source_id
            # For regular aliases, filter by entry_id -> source_id
            alias_query = """
                SELECT DISTINCT ea.alias_norm
                FROM entity_aliases ea
                WHERE (
                    (ea.entry_id IS NULL AND ea.source_id = %s)
                    OR
                    (ea.entry_id IS NOT NULL AND EXISTS (
                        SELECT 1 FROM concordance_entries ce
                        JOIN concordance_sources cs ON cs.id = ce.source_id
                        WHERE ce.id = ea.entry_id AND cs.slug = %s
                    ))
                )
            """
            alias_params = [source_id, source_slug]
        else:
            alias_query = """
                SELECT DISTINCT ea.alias_norm
                FROM entity_aliases ea
            """
            alias_params = []
        
        cur.execute(alias_query, alias_params)
        alias_norms = [row[0] for row in cur.fetchall()]
        
        if len(alias_norms) == 0:
            print(f"WARNING: No aliases found for source slug '{source_slug}'", file=sys.stderr)
            if source_slug:
                # Debug: check what aliases exist
                cur.execute("SELECT COUNT(*) FROM entity_aliases")
                total_aliases = cur.fetchone()[0]
                print(f"Total aliases in database: {total_aliases}", file=sys.stderr)
                
                cur.execute("""
                    SELECT COUNT(*) FROM entity_aliases ea
                    WHERE ea.entry_id IS NULL AND ea.source_id = %s
                """, (source_id,))
                derived_count = cur.fetchone()[0]
                print(f"Derived aliases (entry_id=NULL) for source_id {source_id}: {derived_count}", file=sys.stderr)
                
                cur.execute("""
                    SELECT COUNT(*) FROM entity_aliases ea
                    WHERE ea.entry_id IS NOT NULL AND EXISTS (
                        SELECT 1 FROM concordance_entries ce
                        JOIN concordance_sources cs ON cs.id = ce.source_id
                        WHERE ce.id = ea.entry_id AND cs.slug = %s
                    )
                """, (source_slug,))
                regular_count = cur.fetchone()[0]
                print(f"Regular aliases for source slug '{source_slug}': {regular_count}", file=sys.stderr)
            return
        
        print(f"Computing DF for {len(alias_norms)} aliases...", file=sys.stderr)
        
        # Get document IDs to process
        # chunks -> chunk_metadata -> documents (chunks don't have direct document_id)
        doc_query = """
            SELECT DISTINCT cm.document_id, COUNT(DISTINCT c.id) as total_chunks
            FROM chunks c
            JOIN chunk_metadata cm ON cm.chunk_id = c.id
        """
        doc_params = []
        
        if collection_slug:
            doc_query += """
                WHERE cm.collection_slug = %s
            """
            doc_params.append(collection_slug)
        
        doc_query += """
            GROUP BY cm.document_id
            HAVING cm.document_id IS NOT NULL
        """
        
        cur.execute(doc_query, doc_params)
        documents = cur.fetchall()
        
        print(f"Processing {len(documents)} documents × {len(alias_norms)} aliases = {len(documents) * len(alias_norms)} operations...", file=sys.stderr)
        print("This may take a while. Progress will be shown every 100 operations.", file=sys.stderr)
        
        total_ops = len(documents) * len(alias_norms)
        processed_ops = 0
        processed_docs = 0
        
        # Batch inserts for better performance
        batch_size = 100
        batch_data = []
        
        for doc_id, total_chunks in documents:
            if doc_id is None:
                continue
            processed_docs += 1
            
            # Get all chunks for this document once
            cur.execute("""
                SELECT c.id, c.text
                FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                WHERE cm.document_id = %s
            """, (doc_id,))
            doc_chunks = {chunk_id: chunk_text.lower() for chunk_id, chunk_text in cur.fetchall()}
            
            # For each alias, count chunks in this document that contain it
            for alias_norm in alias_norms:
                processed_ops += 1
                if processed_ops % 100 == 0:
                    pct = (processed_ops / total_ops) * 100
                    print(f"  Progress: {processed_ops}/{total_ops} ({pct:.1f}%) - doc {processed_docs}/{len(documents)}", file=sys.stderr, end="\r")
                
                # Count chunks containing this alias (case-insensitive)
                chunk_count = sum(1 for chunk_text in doc_chunks.values() if alias_norm.lower() in chunk_text)
                
                if chunk_count > 0:
                    df_percent = (chunk_count / total_chunks) * 100 if total_chunks > 0 else 0
                    batch_data.append((alias_norm, doc_id, chunk_count, total_chunks, df_percent))
                    
                    # Insert in batches
                    if len(batch_data) >= batch_size:
                        from psycopg2.extras import execute_values
                        execute_values(
                            cur,
                            """
                            INSERT INTO alias_stats (alias_norm, document_id, df_chunks, total_chunks, df_percent)
                            VALUES %s
                            ON CONFLICT (alias_norm, document_id) 
                            DO UPDATE SET 
                                df_chunks = EXCLUDED.df_chunks,
                                total_chunks = EXCLUDED.total_chunks,
                                df_percent = EXCLUDED.df_percent,
                                updated_at = NOW()
                            """,
                            batch_data,
                            page_size=batch_size
                        )
                        batch_data = []
        
        # Insert remaining batch
        if batch_data:
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """
                INSERT INTO alias_stats (alias_norm, document_id, df_chunks, total_chunks, df_percent)
                VALUES %s
                ON CONFLICT (alias_norm, document_id) 
                DO UPDATE SET 
                    df_chunks = EXCLUDED.df_chunks,
                    total_chunks = EXCLUDED.total_chunks,
                    df_percent = EXCLUDED.df_percent,
                    updated_at = NOW()
                """,
                batch_data,
                page_size=batch_size
            )
        
        print(f"\nCompleted DF computation for {len(alias_norms)} aliases across {len(documents)} documents", file=sys.stderr)
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Compute document frequency (DF) for aliases per document")
    parser.add_argument("--collection", help="Filter by collection slug")
    parser.add_argument("--source-slug", help="Filter aliases by concordance source slug")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be computed without inserting")
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        if args.dry_run:
            print("DRY RUN: Would compute DF statistics", file=sys.stderr)
            return
        compute_alias_frequency(conn, collection_slug=args.collection, source_slug=args.source_slug)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
