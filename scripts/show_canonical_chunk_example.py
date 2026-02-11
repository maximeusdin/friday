#!/usr/bin/env python3
"""
Show an example chunk with MENTION_INDEX (what the model sees for canonical embeddings).

CHUNK ARCHITECTURE (no second copy of chunks):
  - Same chunk_ids everywhere. embed_canonical_chunks does NOT create new chunks.
  - venona/vassiliev: canonicalize_batch augments chunk.text with PEM-derived
    [MENTION_INDEX] block; that augmented text is embedded and stored in
    chunk_embeddings_canonical. Original chunks table unchanged.
  - Non-PEM collections (rosenberg, etc.): bulk_copy_canonical copies chunks.embedding
    and chunks.text into chunk_embeddings_canonical. No augmentation. Same chunk_ids.

  entity_mentions references chunks(id). Same chunk_ids => entity_mentions should
  still be valid IF it was populated for those chunks. The issue is entity_mentions
  was never populated for venona/vassiliev (PEM is the source there).

Fetches a chunk from chunk_embeddings_canonical that has PEM augmentation
(venona/vassiliev). Displays the full text_canonical including the [MENTION_INDEX]
block appended at the end.

Usage:
    python scripts/show_canonical_chunk_example.py
    python scripts/show_canonical_chunk_example.py --chunk-id 12345
    python scripts/show_canonical_chunk_example.py --entity-name "Jacob Golos"
    python scripts/show_canonical_chunk_example.py --recent 1   # most recently rebuilt (after limit run)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _print_synthetic_example():
    """Print a synthetic example of what the model sees (no DB required)."""
    example = r'''
GOLOS (Jacob Golos) was born on April 30, 1890, in Russia. He joined the Communist
Party USA in 1919 and served as a district organizer in Detroit. Through the 1920s
he worked in the Soviet Union as business manager of Kuzbas. He was also secretary
of Technical Aid for Soviet Russia. In November 1943 the NKGB recommended him for
the Order of the Red Star.

[MENTION_INDEX page_scoped collection=alias_scoped]
golos => Jacob Golos
sound => Jacob Golos
zvuk => Jacob Golos
john => Jacob Golos
raisin => Jacob Golos
rasin => Jacob Golos
tasin => Jacob Golos
[/MENTION_INDEX]
'''
    print("=" * 80)
    print("SYNTHETIC EXAMPLE (--example flag; no DB)")
    print("=" * 80)
    print()
    print("--- text_canonical (what the model sees when using canonical embeddings) ---")
    print()
    print(example.strip())
    print()
    print("--- explanation ---")
    print("The chunk text is followed by a [MENTION_INDEX] block built from PEM")
    print("(page_entity_mentions). Each line maps surface_norm => canonical_name.")
    print("The model sees both the raw text and these alias->entity mappings,")
    print("improving retrieval for codenames (sound, zvuk, john, etc.) -> Jacob Golos.")
    print()


def get_conn():
    import psycopg2

    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "neh"),
        user=os.getenv("DB_USER", "neh"),
        password=os.getenv("DB_PASS", "neh"),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Show example chunk with MENTION_INDEX (what the model sees)"
    )
    ap.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="Specific chunk to show; default finds one with MENTION_INDEX",
    )
    ap.add_argument(
        "--entity-name",
        type=str,
        default=None,
        help="Find a chunk mentioning this entity (e.g. 'Jacob Golos')",
    )
    ap.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Limit to collection (venona, vassiliev); default either",
    )
    ap.add_argument(
        "--example",
        action="store_true",
        help="Show synthetic example (no DB); use when DB unavailable",
    )
    ap.add_argument(
        "--recent",
        type=int,
        default=None,
        metavar="N",
        help="Show Nth most recently created canonical chunk (default 1); use after rebuild",
    )
    args = ap.parse_args()

    if args.example:
        _print_synthetic_example()
        return

    conn = get_conn()
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    collections = ["venona", "vassiliev"]
    if args.collection:
        collections = [args.collection]

    with conn.cursor() as cur:
        if args.recent is not None:
            n = max(1, args.recent)
            cur.execute(
                """
                SELECT cec.chunk_id, cec.text_canonical, cec.rewrite_manifest,
                       cm.collection_slug, cm.document_id,
                       d.source_name
                FROM chunk_embeddings_canonical cec
                JOIN chunk_metadata cm ON cm.chunk_id = cec.chunk_id
                  AND cm.pipeline_version = cec.pipeline_version
                LEFT JOIN documents d ON d.id = cm.document_id
                WHERE cec.embedding_model = %s
                  AND cm.collection_slug = ANY(%s)
                  AND cec.text_canonical LIKE '%%[MENTION_INDEX%%'
                ORDER BY cec.created_at DESC, cec.id DESC
                OFFSET %s LIMIT 1
                """,
                (embed_model, collections, n - 1),
            )
        elif args.chunk_id:
            cur.execute(
                """
                SELECT cec.chunk_id, cec.text_canonical, cec.rewrite_manifest,
                       cm.collection_slug, cm.document_id,
                       d.source_name
                FROM chunk_embeddings_canonical cec
                JOIN chunk_metadata cm ON cm.chunk_id = cec.chunk_id
                  AND cm.pipeline_version = cec.pipeline_version
                LEFT JOIN documents d ON d.id = cm.document_id
                WHERE cec.chunk_id = %s
                  AND cec.embedding_model = %s
                  AND cm.collection_slug = ANY(%s)
                LIMIT 1
                """,
                (args.chunk_id, embed_model, collections),
            )
        elif args.entity_name:
            cur.execute(
                """
                SELECT e.id FROM entities e
                WHERE LOWER(e.canonical_name) = LOWER(%s)
                   OR EXISTS (
                     SELECT 1 FROM entity_aliases ea
                     WHERE ea.entity_id = e.id AND LOWER(ea.alias) = LOWER(%s)
                   )
                LIMIT 1
                """,
                (args.entity_name, args.entity_name),
            )
            row = cur.fetchone()
            if not row:
                print(f"Entity '{args.entity_name}' not found.", file=sys.stderr)
                sys.exit(1)
            entity_id = row[0]
            cur.execute(
                """
                SELECT cec.chunk_id
                FROM chunk_embeddings_canonical cec
                JOIN chunk_metadata cm ON cm.chunk_id = cec.chunk_id
                  AND cm.pipeline_version = cec.pipeline_version
                JOIN chunk_pages cp ON cp.chunk_id = cec.chunk_id
                JOIN page_entity_mentions pem ON pem.page_id = cp.page_id
                  AND pem.collection_slug = cm.collection_slug
                WHERE cec.embedding_model = %s
                  AND cm.collection_slug = ANY(%s)
                  AND pem.entity_id = %s
                  AND cec.text_canonical LIKE '%%[MENTION_INDEX%%'
                ORDER BY cec.chunk_id
                LIMIT 1
                """,
                (embed_model, collections, entity_id),
            )
            row = cur.fetchone()
            if not row:
                print(
                    f"No canonical chunk with MENTION_INDEX found for entity '{args.entity_name}'.",
                    file=sys.stderr,
                )
                sys.exit(1)
            args.chunk_id = row[0]
            cur.execute(
                """
                SELECT cec.chunk_id, cec.text_canonical, cec.rewrite_manifest,
                       cm.collection_slug, cm.document_id,
                       d.source_name
                FROM chunk_embeddings_canonical cec
                JOIN chunk_metadata cm ON cm.chunk_id = cec.chunk_id
                  AND cm.pipeline_version = cec.pipeline_version
                LEFT JOIN documents d ON d.id = cm.document_id
                WHERE cec.chunk_id = %s
                  AND cec.embedding_model = %s
                LIMIT 1
                """,
                (args.chunk_id, embed_model),
            )
        else:
            cur.execute(
                """
                SELECT cec.chunk_id, cec.text_canonical, cec.rewrite_manifest,
                       cm.collection_slug, cm.document_id,
                       d.source_name
                FROM chunk_embeddings_canonical cec
                JOIN chunk_metadata cm ON cm.chunk_id = cec.chunk_id
                  AND cm.pipeline_version = cec.pipeline_version
                WHERE cec.embedding_model = %s
                  AND cm.collection_slug = ANY(%s)
                  AND cec.text_canonical LIKE '%%[MENTION_INDEX%%'
                ORDER BY cec.chunk_id
                LIMIT 1
                """,
                (embed_model, collections),
            )

        row = cur.fetchone()
        if not row:
            print(
                "No canonical chunk with MENTION_INDEX found. "
                "Run embed_canonical_chunks for venona/vassiliev first.",
                file=sys.stderr,
            )
            sys.exit(1)

        chunk_id, text_canonical, manifest, collection_slug, doc_id, doc_name = row

    # Display
    print("=" * 80)
    print(f"CHUNK {chunk_id} | collection={collection_slug} | doc={doc_name or doc_id}")
    print("=" * 80)
    print()
    print("--- text_canonical (what the model sees when using canonical embeddings) ---")
    print()
    print(text_canonical)
    print()
    print("--- rewrite_manifest (PEM mappings used to build MENTION_INDEX) ---")
    if manifest:
        import json

        print(json.dumps(manifest[:10], indent=2))
        if len(manifest) > 10:
            print(f"  ... and {len(manifest) - 10} more")
    else:
        print("  (empty)")
    print()


if __name__ == "__main__":
    main()
