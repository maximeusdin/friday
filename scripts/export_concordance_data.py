#!/usr/bin/env python3
"""
Export concordance data to CSV files for examination.

Exports:
- concordance_entries.csv: All entries with their parsed data
- entities.csv: All entities from concordance
- entity_aliases.csv: All aliases with their types
- entity_links.csv: All relationships (cover_name_of, changed_to, etc.)
- entity_citations.csv: All citations with scoped labels
"""

import os
import sys
import csv
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn


def export_concordance_entries(output_dir: str, source_slug: str = None, conn=None):
    """Export concordance entries to CSV."""
    if conn is None:
        conn = get_conn()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with conn.cursor() as cur:
        # Export concordance_entries
        query = """
            SELECT 
                ce.id,
                cs.slug AS source_slug,
                cs.title AS source_title,
                ce.entry_key,
                ce.entry_seq,
                ce.raw_text
            FROM concordance_entries ce
            JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, ce.entry_seq"
        
        cur.execute(query, params)
        
        with open(output_dir / "concordance_entries.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'source_slug', 'source_title', 'entry_key', 'entry_seq', 'raw_text'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported concordance_entries to {output_dir / 'concordance_entries.csv'}", file=sys.stderr)
        
        # Export entities (from concordance sources)
        query = """
            SELECT DISTINCT
                e.id,
                e.entity_type,
                e.canonical_name,
                e.confidence,
                e.notes,
                cs.slug AS source_slug
            FROM entities e
            JOIN entity_aliases ea ON ea.entity_id = e.id
            JOIN concordance_entries ce ON ce.id = ea.entry_id
            JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, e.id"
        
        cur.execute(query, params)
        
        with open(output_dir / "entities.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'entity_type', 'canonical_name', 'confidence', 'notes', 'source_slug'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported entities to {output_dir / 'entities.csv'}", file=sys.stderr)
        
        # Export entity_aliases
        query = """
            SELECT 
                ea.id,
                ea.entity_id,
                e.canonical_name,
                ea.alias,
                ea.alias_norm,
                ea.alias_type,
                ea.confidence,
                ea.notes,
                cs.slug AS source_slug,
                ce.entry_key
            FROM entity_aliases ea
            JOIN entities e ON ea.entity_id = e.id
            LEFT JOIN concordance_entries ce ON ce.id = ea.entry_id
            LEFT JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, ea.entity_id, ea.id"
        
        cur.execute(query, params)
        
        with open(output_dir / "entity_aliases.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'entity_id', 'canonical_name', 'alias', 'alias_norm', 'alias_type', 'confidence', 'notes', 'source_slug', 'entry_key'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported entity_aliases to {output_dir / 'entity_aliases.csv'}", file=sys.stderr)
        
        # Export entity_links
        query = """
            SELECT 
                el.id,
                el.from_entity_id,
                e1.canonical_name AS from_name,
                el.to_entity_id,
                e2.canonical_name AS to_name,
                el.link_type,
                el.confidence,
                el.notes,
                cs.slug AS source_slug,
                ce.entry_key
            FROM entity_links el
            JOIN entities e1 ON el.from_entity_id = e1.id
            JOIN entities e2 ON el.to_entity_id = e2.id
            LEFT JOIN concordance_entries ce ON ce.id = el.entry_id
            LEFT JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, el.id"
        
        cur.execute(query, params)
        
        with open(output_dir / "entity_links.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'from_entity_id', 'from_name', 'to_entity_id', 'to_name', 'link_type', 'confidence', 'notes', 'source_slug', 'entry_key'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported entity_links to {output_dir / 'entity_links.csv'}", file=sys.stderr)
        
        # Export entity_citations
        query = """
            SELECT 
                ec.id,
                ec.entity_id,
                e.canonical_name,
                ec.citation_text,
                ea.alias AS alias_label,
                ec.collection_slug,
                ec.document_label,
                ec.page_list,
                ec.notes,
                cs.slug AS source_slug,
                ce.entry_key
            FROM entity_citations ec
            JOIN entities e ON ec.entity_id = e.id
            LEFT JOIN entity_aliases ea ON ec.alias_id = ea.id
            LEFT JOIN concordance_entries ce ON ce.id = ec.entry_id
            LEFT JOIN concordance_sources cs ON ce.source_id = cs.id
        """
        params = []
        if source_slug:
            query += " WHERE cs.slug = %s"
            params.append(source_slug)
        query += " ORDER BY cs.slug, ec.entity_id, ec.id"
        
        cur.execute(query, params)
        
        with open(output_dir / "entity_citations.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'entity_id', 'canonical_name', 'citation_text', 'alias_label', 'collection_slug', 'document_label', 'page_list', 'notes', 'source_slug', 'entry_key'])
            for row in cur.fetchall():
                writer.writerow(row)
        
        print(f"Exported entity_citations to {output_dir / 'entity_citations.csv'}", file=sys.stderr)
        
        # Print summary
        if source_slug:
            cur.execute("""
                SELECT COUNT(*) FROM concordance_entries ce
                JOIN concordance_sources cs ON ce.source_id = cs.id
                WHERE cs.slug = %s
            """, (source_slug,))
            entry_count = cur.fetchone()[0]
            print(f"\nSummary for source '{source_slug}':", file=sys.stderr)
            print(f"  Entries: {entry_count}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Export concordance data to CSV files")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="concordance_export",
        help="Output directory for CSV files (default: concordance_export)",
    )
    parser.add_argument(
        "--source-slug",
        "-s",
        default=None,
        help="Filter by source slug (e.g., 'vassiliev_venona_index_small'). If not specified, exports all sources.",
    )
    args = parser.parse_args()
    
    conn = get_conn()
    try:
        export_concordance_entries(args.output_dir, args.source_slug, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
