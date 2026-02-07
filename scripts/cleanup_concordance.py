#!/usr/bin/env python3
"""Clean up garbage entities and aliases from the concordance source.

Garbage = canonical_name or alias with > 3 words, or containing
semicolons, em-dashes, or digits (page references / sentence fragments).

Usage:
    python scripts/cleanup_concordance.py [--dry-run] [--db DATABASE_URL]
"""
import argparse
import os
import sys
from typing import List

import psycopg2

# Batch size for slow operations (UPDATE mention_candidates, etc.)
BATCH_SIZE = 5000
# Default statement timeout: 2 hours (Postgres uses milliseconds)
DEFAULT_STATEMENT_TIMEOUT_MS = 2 * 60 * 60 * 1000

GARBAGE_CONDITION_ALIAS = """
    array_length(string_to_array(trim(alias), ' '), 1) > 3
    OR alias ~ '[0-9;–—]'
"""

GARBAGE_CONDITION_ENTITY = """
    array_length(string_to_array(trim(canonical_name), ' '), 1) > 3
    OR canonical_name ~ '[0-9;–—]'
"""


def _batched(ids: List[int], size: int):
    """Yield chunks of ids of at most `size`."""
    for i in range(0, len(ids), size):
        yield ids[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="Clean concordance garbage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count, don't delete")
    parser.add_argument("--db", default=None,
                        help="Database URL (default: DATABASE_URL env var or local)")
    parser.add_argument("--slug", default="vassiliev_venona_index_20260130",
                        help="Concordance source slug to clean")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Statement timeout in seconds (default: 7200). Use 0 for no limit.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for slow updates (default: {BATCH_SIZE})")
    args = parser.parse_args()

    db_url = args.db or os.environ.get(
        "DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"
    )
    print(f"Connecting to: {db_url.split('@')[-1]}")

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    # Allow long-running statements (default 2 hours; 0 = no limit)
    timeout_ms = (args.timeout if args.timeout is not None else 7200) * 1000
    if timeout_ms > 0:
        cur.execute("SET statement_timeout = %s", (str(timeout_ms),))
        print(f"Statement timeout: {timeout_ms // 1000}s")
    else:
        cur.execute("SET statement_timeout = 0")
        print("Statement timeout: off")

    # Use explicit slug (or override via --slug)
    slug = args.slug
    print(f"Concordance source slug: {slug}")

    cur.execute("SELECT id FROM concordance_sources WHERE slug = %s", (slug,))
    row = cur.fetchone()
    if not row:
        print(f"ERROR: No concordance_source with slug={slug}")
        sys.exit(1)
    source_id = row[0]
    print(f"Source ID: {source_id}")

    # Count totals
    cur.execute("SELECT COUNT(*) FROM entities e WHERE e.source_id = %s", (source_id,))
    total_entities = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM entity_aliases ea WHERE ea.source_id = %s", (source_id,))
    total_aliases = cur.fetchone()[0]
    print(f"\nTotal entities: {total_entities}")
    print(f"Total aliases:  {total_aliases}")

    # Count garbage aliases
    cur.execute(f"""
        SELECT COUNT(*)
        FROM entity_aliases ea
        WHERE ea.source_id = %s AND ({GARBAGE_CONDITION_ALIAS})
    """, (source_id,))
    garbage_aliases = cur.fetchone()[0]
    print(f"\nGarbage aliases to delete: {garbage_aliases}")

    # Show sample garbage aliases
    cur.execute(f"""
        SELECT ea.alias, ea.entity_id
        FROM entity_aliases ea
        WHERE ea.source_id = %s AND ({GARBAGE_CONDITION_ALIAS})
        ORDER BY length(ea.alias) DESC
        LIMIT 20
    """, (source_id,))
    samples = cur.fetchall()
    if samples:
        print("  Sample garbage aliases:")
        for alias, eid in samples:
            print(f"    [{eid}] {alias!r}")

    # Count garbage entities
    cur.execute(f"""
        SELECT COUNT(*)
        FROM entities e
        WHERE e.source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
    """, (source_id,))
    garbage_entities = cur.fetchone()[0]
    print(f"\nGarbage entities to delete: {garbage_entities}")

    # Show sample garbage entities
    cur.execute(f"""
        SELECT e.canonical_name, e.id
        FROM entities e
        WHERE e.source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
        ORDER BY length(e.canonical_name) DESC
        LIMIT 20
    """, (source_id,))
    samples = cur.fetchall()
    if samples:
        print("  Sample garbage entities:")
        for name, eid in samples:
            print(f"    [{eid}] {name!r}")

    if args.dry_run:
        print("\n--dry-run: no changes made.")
        conn.rollback()
        conn.close()
        return

    if garbage_aliases == 0 and garbage_entities == 0:
        print("\nNo garbage found. Nothing to do.")
        conn.close()
        return

    # Fetch garbage entity IDs once for batched steps
    cur.execute(f"""
        SELECT id FROM entities
        WHERE source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
        ORDER BY id
    """, (source_id,))
    garbage_entity_ids: List[int] = [r[0] for r in cur.fetchall()]
    assert len(garbage_entity_ids) == garbage_entities, "entity count mismatch"

    batch_size = args.batch_size
    total_steps = 8  # aliases, mention_candidates, remaining aliases, citations, links, ocr_variant_clusters, entities
    step = 0

    # Step 1: Delete garbage aliases
    step += 1
    print(f"\n[Step {step}/{total_steps}] Deleting {garbage_aliases} garbage aliases...")
    cur.execute(f"""
        DELETE FROM entity_aliases
        WHERE source_id = %s AND ({GARBAGE_CONDITION_ALIAS})
    """, (source_id,))
    print(f"  -> Deleted {cur.rowcount} alias rows.")
    conn.commit()

    # Step 2: Null out mention_candidates.resolved_entity_id in batches (avoids statement timeout)
    step += 
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'mention_candidates'
              AND column_name = 'resolved_entity_id'
        )
    """)
    if cur.fetchone()[0] and garbage_entity_ids:
        n_batches = (len(garbage_entity_ids) + batch_size - 1) // batch_size
        print(f"[Step {step}/{total_steps}] Clearing mention_candidates.resolved_entity_id for {len(garbage_entity_ids)} entities (batches of {batch_size})...")
        total_updated = 0
        for i, batch in enumerate(_batched(garbage_entity_ids, batch_size), 1):
            placeholders = ",".join(["%s"] * len(batch))
            cur.execute(
                f"UPDATE mention_candidates SET resolved_entity_id = NULL WHERE resolved_entity_id IN ({placeholders})",
                batch,
            )
            total_updated += cur.rowcount
            print(f"  -> Batch {i}/{n_batches}: {cur.rowcount} rows updated (total: {total_updated})", flush=True)
            conn.commit()
        print(f"  -> Done: {total_updated} mention_candidates rows cleared.")
    else:
        print(f"[Step {step}/{total_steps}] Skipping mention_candidates (table/column missing or no entities).")

    # Step 3: Delete remaining aliases of garbage entities
    step += 1
    print(f"[Step {step}/{total_steps}] Deleting remaining aliases of garbage entities...")
    placeholders = ",".join(["%s"] * len(garbage_entity_ids))
    cur.execute(f"DELETE FROM entity_aliases WHERE entity_id IN ({placeholders})", garbage_entity_ids)
    print(f"  -> Deleted {cur.rowcount} alias rows.")
    conn.commit()

    # Step 4: Delete entity_citations
    step += 1
    print(f"[Step {step}/{total_steps}] Deleting entity_citations...")
    cur.execute(f"DELETE FROM entity_citations WHERE entity_id IN ({placeholders})", garbage_entity_ids)
    print(f"  -> Deleted {cur.rowcount} rows.")
    conn.commit()

    # Step 5: Delete entity_links
    step += 1
    print(f"[Step {step}/{total_steps}] Deleting entity_links...")
    cur.execute(
        f"DELETE FROM entity_links WHERE from_entity_id IN ({placeholders}) OR to_entity_id IN ({placeholders})",
        garbage_entity_ids + garbage_entity_ids,
    )
    print(f"  -> Deleted {cur.rowcount} rows.")
    conn.commit()

    # Step 6: Clear ocr_variant_clusters.canonical_entity_id (FK references entities)
    step += 1
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'ocr_variant_clusters'
              AND column_name = 'canonical_entity_id'
        )
    """)
    if cur.fetchone()[0] and garbage_entity_ids:
        n_batches = (len(garbage_entity_ids) + batch_size - 1) // batch_size
        print(f"[Step {step}/{total_steps}] Clearing ocr_variant_clusters.canonical_entity_id for {len(garbage_entity_ids)} entities (batches of {batch_size})...")
        total_updated = 0
        for i, batch in enumerate(_batched(garbage_entity_ids, batch_size), 1):
            ph = ",".join(["%s"] * len(batch))
            cur.execute(
                f"UPDATE ocr_variant_clusters SET canonical_entity_id = NULL WHERE canonical_entity_id IN ({ph})",
                batch,
            )
            total_updated += cur.rowcount
            print(f"  -> Batch {i}/{n_batches}: {cur.rowcount} rows updated (total: {total_updated})", flush=True)
            conn.commit()
        print(f"  -> Done: {total_updated} ocr_variant_clusters rows cleared.")
    else:
        print(f"[Step {step}/{total_steps}] Skipping ocr_variant_clusters (table/column missing or no entities).")

    # Step 7: Delete the garbage entities themselves
    step += 1
    print(f"[Step {step}/{total_steps}] Deleting {garbage_entities} garbage entity rows...")
    cur.execute(f"""
        DELETE FROM entities
        WHERE source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
    """, (source_id,))
    print(f"  -> Deleted {cur.rowcount} entity rows.")
    conn.commit()

    print("\nDone! All changes committed.")

    # Final counts
    cur.execute("SELECT COUNT(*) FROM entities e WHERE e.source_id = %s", (source_id,))
    print(f"Remaining entities: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM entity_aliases ea WHERE ea.source_id = %s", (source_id,))
    print(f"Remaining aliases:  {cur.fetchone()[0]}")

    conn.close()


if __name__ == "__main__":
    main()
