#!/usr/bin/env python3
"""
Show detailed information about specific entities.

Usage:
    python scripts/show_entity_details.py 45240 56335
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2

def get_conn():
    """Get database connection using environment variables."""
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "neh")
    DB_USER = os.getenv("DB_USER", "neh")
    DB_PASS = os.getenv("DB_PASS", "neh")
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

def show_entity_details(entity_ids):
    """Show detailed information about entities."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for entity_id in entity_ids:
                # Get entity basic info
                cur.execute("""
                    SELECT 
                        id,
                        entity_type,
                        canonical_name,
                        created_at
                    FROM entities
                    WHERE id = %s
                """, (entity_id,))
                
                row = cur.fetchone()
                if not row:
                    print(f"\nEntity {entity_id}: NOT FOUND", file=sys.stderr)
                    continue
                
                entity_id_db, entity_type, canonical_name, created_at = row
                
                print(f"\n{'='*80}", file=sys.stderr)
                print(f"ENTITY ID: {entity_id_db}", file=sys.stderr)
                print(f"{'='*80}", file=sys.stderr)
                print(f"Canonical Name: {canonical_name}", file=sys.stderr)
                print(f"Entity Type: {entity_type}", file=sys.stderr)
                print(f"Created: {created_at}", file=sys.stderr)
                
                # Get all aliases
                cur.execute("""
                    SELECT 
                        id,
                        alias,
                        alias_norm,
                        alias_class,
                        is_matchable,
                        is_auto_match,
                        match_case,
                        min_chars,
                        allow_ambiguous_person_token,
                        created_at
                    FROM entity_aliases
                    WHERE entity_id = %s
                    ORDER BY alias
                """, (entity_id,))
                
                aliases = cur.fetchall()
                print(f"\nAliases ({len(aliases)} total):", file=sys.stderr)
                for alias_row in aliases:
                    alias_id, alias, alias_norm, alias_class, is_matchable, is_auto_match, match_case, min_chars, allow_ambiguous, created_at = alias_row
                    print(f"  - '{alias}' (norm: '{alias_norm}')", file=sys.stderr)
                    print(f"    class: {alias_class}, matchable: {is_matchable}, auto: {is_auto_match}", file=sys.stderr)
                    print(f"    match_case: {match_case}, min_chars: {min_chars}, allow_ambiguous: {allow_ambiguous}", file=sys.stderr)
                
                # Get entity mentions count
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM entity_mentions
                    WHERE entity_id = %s
                """, (entity_id,))
                mention_count = cur.fetchone()[0]
                print(f"\nEntity Mentions: {mention_count}", file=sys.stderr)
                
                # Get entity mentions with chunk/document info
                cur.execute("""
                    SELECT 
                        em.id,
                        em.chunk_id,
                        em.document_id,
                        em.surface,
                        d.source_name,
                        d.collection_id
                    FROM entity_mentions em
                    JOIN documents d ON d.id = em.document_id
                    WHERE em.entity_id = %s
                    LIMIT 5
                """, (entity_id,))
                
                mentions = cur.fetchall()
                if mentions:
                    print(f"\nSample Mentions (showing first 5 of {mention_count}):", file=sys.stderr)
                    for mention_id, chunk_id, doc_id, surface, source_name, coll_id in mentions:
                        print(f"  - Mention ID: {mention_id}, Surface: '{surface}'", file=sys.stderr)
                        print(f"    Document: {source_name} (doc_id: {doc_id})", file=sys.stderr)
                
                # Check for citations from entity_citations table
                try:
                    cur.execute("""
                        SELECT 
                            citation_text,
                            document_label,
                            page_list
                        FROM entity_citations
                        WHERE entity_id = %s
                        LIMIT 5
                    """, (entity_id,))
                    citations = cur.fetchall()
                    if citations:
                        print(f"\nCitations from entity_citations table ({len(citations)} shown):", file=sys.stderr)
                        for citation_text, doc_label, page_list in citations:
                            print(f"  - Citation: {citation_text[:100]}...", file=sys.stderr)
                            print(f"    Document: {doc_label}", file=sys.stderr)
                            print(f"    Pages: {page_list}", file=sys.stderr)
                    else:
                        print(f"\nNo citations in entity_citations table", file=sys.stderr)
                except psycopg2.errors.UndefinedTable:
                    print(f"\nentity_citations table does not exist", file=sys.stderr)
            
            print(f"\n{'='*80}", file=sys.stderr)
            
    finally:
        conn.close()

def main():
    ap = argparse.ArgumentParser(description="Show detailed information about entities")
    ap.add_argument("entity_ids", nargs="+", type=int, help="Entity IDs to show")
    
    args = ap.parse_args()
    show_entity_details(args.entity_ids)

if __name__ == "__main__":
    main()
