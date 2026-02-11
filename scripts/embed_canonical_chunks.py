"""
Dual-Index Canonical Embeddings — build chunk_embeddings_canonical (Index B).

- Alias-scoped (venona, vassiliev): PEM augmentation + embed
- Others: copy from chunks (no re-embedding)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.embed_venona_chunks import (
    embed_texts,
    get_conn,
    vector_literal,
    truncate_text,
    MAX_EMBED_CHARS,
)

from retrieval.agent.v10_types import ALIAS_SCOPED_COLLECTIONS
from retrieval.canonicalize import canonicalize_batch

ALIAS_SCOPED = tuple(ALIAS_SCOPED_COLLECTIONS)
EXPECTED_DIM = 1536
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def fetch_target_chunks(
    cur,
    chunk_pv: str,
    collection_slug: str,
    fill_missing_only: bool,
    limit: Optional[int],
    embedding_model: str,
) -> List[Tuple[int, str]]:
    """Fetch chunks to process. For fill_missing_only, exclude chunks that already have canonical row.
    For alias-scoped (venona, vassiliev): order by PEM coverage first so chunks with mappings
    are processed first (helps dry-run sampling; many Venona docs have no concordance citations).
    """
    where = """
        WHERE c.pipeline_version = %s
          AND cm.pipeline_version = %s
          AND cm.collection_slug = %s
    """
    params: List[Any] = [chunk_pv, chunk_pv, collection_slug]
    if fill_missing_only:
        where += """
          AND NOT EXISTS (
            SELECT 1 FROM chunk_embeddings_canonical cec
            WHERE cec.chunk_id = c.id
              AND cec.pipeline_version = %s
              AND cec.embedding_model = %s
          )
        """
        params.extend([chunk_pv, embedding_model])
    lim = " LIMIT %s" if limit else ""
    if limit:
        params.append(limit)

    # Alias-scoped: order by PEM coverage (chunks with mappings first)
    order_clause = "c.id"
    if collection_slug in ALIAS_SCOPED:
        order_clause = """(
            SELECT COUNT(*) FROM chunk_pages cp2
            JOIN page_entity_mentions pem ON pem.page_id = cp2.page_id
              AND pem.collection_slug = cm.collection_slug
            WHERE cp2.chunk_id = c.id
        ) DESC NULLS LAST, c.id"""

    sql = f"""
        SELECT c.id, c.text
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id
        {where}
        ORDER BY {order_clause}
        {lim}
    """
    cur.execute(sql, params)
    return [(r[0], r[1] or "") for r in cur.fetchall()]


def fetch_chunks_for_copy(
    cur,
    chunk_pv: str,
    collections: List[str],
    limit: Optional[int],
    embedding_model: str,
) -> List[Tuple[int, str, str]]:
    """Fetch chunks from non-alias-scoped collections for copy. Returns (chunk_id, text, embedding_literal)."""
    placeholders = ",".join(["%s"] * len(collections))
    params: List[Any] = [chunk_pv, chunk_pv] + list(collections) + [chunk_pv, embedding_model]
    lim = " LIMIT %s" if limit else ""
    if limit:
        params.append(limit)
    sql = f"""
        SELECT c.id, c.text, c.embedding::text
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id
        WHERE c.pipeline_version = %s
          AND cm.pipeline_version = %s
          AND cm.collection_slug IN ({placeholders})
          AND c.embedding IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM chunk_embeddings_canonical cec
            WHERE cec.chunk_id = c.id
              AND cec.pipeline_version = %s
              AND cec.embedding_model = %s
          )
        ORDER BY c.id
        {lim}
    """
    cur.execute(sql, params)
    return [(r[0], r[1] or "", r[2] or "") for r in cur.fetchall()]


def upsert_canonical(
    cur,
    chunk_id: int,
    pipeline_version: str,
    embedding_model: str,
    text_canonical: str,
    embedding_literal: Optional[str],
    rewrite_manifest: List[Dict[str, Any]],
):
    """Upsert into chunk_embeddings_canonical."""
    manifest_json = json.dumps(rewrite_manifest)
    if embedding_literal:
        cur.execute("""
            INSERT INTO chunk_embeddings_canonical
                (chunk_id, pipeline_version, embedding_model, text_canonical, embedding, rewrite_manifest)
            VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
            ON CONFLICT (chunk_id, pipeline_version, embedding_model)
            DO UPDATE SET
                text_canonical = EXCLUDED.text_canonical,
                embedding = EXCLUDED.embedding,
                rewrite_manifest = EXCLUDED.rewrite_manifest
        """, (chunk_id, pipeline_version, embedding_model, text_canonical, embedding_literal, manifest_json))
    else:
        cur.execute("""
            INSERT INTO chunk_embeddings_canonical
                (chunk_id, pipeline_version, embedding_model, text_canonical, rewrite_manifest)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (chunk_id, pipeline_version, embedding_model)
            DO UPDATE SET
                text_canonical = EXCLUDED.text_canonical,
                rewrite_manifest = EXCLUDED.rewrite_manifest
        """, (chunk_id, pipeline_version, embedding_model, text_canonical, manifest_json))


def run_alias_scoped_path(
    conn,
    chunk_pv: str,
    collection_slug: str,
    batch_size: int,
    rebuild: bool,
    fill_missing_only: bool,
    limit: Optional[int],
    dry_run: bool,
    embedding_model: str,
    sleep: float,
):
    """Process alias-scoped collection: canonicalize + embed."""
    with conn.cursor() as cur:
        targets = fetch_target_chunks(
            cur, chunk_pv, collection_slug, fill_missing_only, limit, embedding_model
        )
    total = len(targets)
    print(f"[alias-scoped] {collection_slug}: {total} chunks to process")
    if total == 0:
        return

    for i in range(0, total, batch_size):
        batch = targets[i : i + batch_size]
        chunk_ids = [cid for cid, _ in batch]
        batch_tuples = [(cid, t) for cid, t in batch]

        # Batch canonicalize: 3 DB round-trips instead of 3*N
        canonical_results = canonicalize_batch(conn, batch_tuples)
        text_canonicals = [r[0] for r in canonical_results]
        manifests = [r[1] for r in canonical_results]

        if dry_run:
            for j, (cid, _) in enumerate(batch):
                print(f"  [dry-run] chunk {cid}: manifest={len(manifests[j])} mappings")
                if manifests[j]:
                    print(f"    sample: {manifests[j][0]}")
            continue

        vectors = embed_texts([truncate_text(t)[0] for t in text_canonicals], verbose=True)
        for v in vectors:
            if len(v) != EXPECTED_DIM:
                raise RuntimeError(f"Embedding dim {len(v)} != {EXPECTED_DIM}")

        with conn.cursor() as cur:
            for j, cid in enumerate(chunk_ids):
                lit = vector_literal(vectors[j])
                upsert_canonical(
                    cur, cid, chunk_pv, embedding_model,
                    text_canonicals[j], lit, manifests[j],
                )
        conn.commit()
        print(f"  Processed {min(i + batch_size, total)}/{total}")
        if limit and (i + batch_size) >= total:
            print(f"  Chunk IDs: {chunk_ids}")

        if sleep > 0:
            time.sleep(sleep)


def fetch_collection_pipeline_versions(
    cur, collections: List[str], chunk_pv_override: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Get (collection_slug, pipeline_version) pairs. Uses chunk_pv_override if set, else discovers from chunk_metadata."""
    if chunk_pv_override:
        return [(coll, chunk_pv_override) for coll in collections]
    placeholders = ",".join(["%s"] * len(collections))
    cur.execute(f"""
        SELECT DISTINCT cm.collection_slug, cm.pipeline_version
        FROM chunk_metadata cm
        WHERE cm.collection_slug IN ({placeholders})
        ORDER BY cm.collection_slug, cm.pipeline_version
    """, list(collections))
    return [(r[0], r[1]) for r in cur.fetchall()]


def bulk_copy_canonical(
    cur,
    chunk_pv: str,
    collection_slug: str,
    embedding_model: str,
    limit: Optional[int],
) -> int:
    """Bulk copy: INSERT...SELECT in one round-trip. Returns row count."""
    lim = " LIMIT %s" if limit else ""
    params: List[Any] = [embedding_model, chunk_pv, collection_slug, embedding_model]
    if limit:
        params.append(limit)
    cur.execute(f"""
        INSERT INTO chunk_embeddings_canonical
            (chunk_id, pipeline_version, embedding_model, text_canonical, embedding, rewrite_manifest)
        SELECT c.id, c.pipeline_version, %s, COALESCE(c.text, ''), c.embedding, '[]'::jsonb
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.id AND cm.pipeline_version = c.pipeline_version
        WHERE c.pipeline_version = %s
          AND cm.collection_slug = %s
          AND c.embedding IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM chunk_embeddings_canonical cec
            WHERE cec.chunk_id = c.id
              AND cec.pipeline_version = c.pipeline_version
              AND cec.embedding_model = %s
          )
        ORDER BY c.id
        {lim}
        ON CONFLICT (chunk_id, pipeline_version, embedding_model)
        DO UPDATE SET
            text_canonical = EXCLUDED.text_canonical,
            embedding = EXCLUDED.embedding,
            rewrite_manifest = EXCLUDED.rewrite_manifest
    """, params)
    return cur.rowcount


def run_copy_path(
    conn,
    chunk_pv: Optional[str],
    collections: List[str],
    limit: Optional[int],
    dry_run: bool,
    embedding_model: str,
    batch_size: int,
):
    """Copy embeddings from chunks for non-alias-scoped collections.
    Uses bulk INSERT...SELECT (one round-trip per collection)."""
    with conn.cursor() as cur:
        pairs = fetch_collection_pipeline_versions(cur, collections, chunk_pv)
    if not pairs:
        print(f"[copy] non-alias collections {collections}: no (collection, pipeline_version) pairs found")
        return

    for coll, pv in pairs:
        with conn.cursor() as cur:
            rows = fetch_chunks_for_copy(cur, pv, [coll], limit, embedding_model)
        total = len(rows)
        print(f"[copy] {coll} (chunk_pv={pv}): {total} chunks to copy")
        if total == 0:
            continue

        if dry_run:
            print(f"  [dry-run] would copy {total} chunks")
            continue

        with conn.cursor() as cur:
            n = bulk_copy_canonical(cur, pv, coll, embedding_model, limit)
        conn.commit()
        print(f"  Copied {n} chunks")


def main():
    ap = argparse.ArgumentParser(description="Build chunk_embeddings_canonical (Index B)")
    ap.add_argument("--chunk-pv", default=None, help="Override pipeline version; default discovers per collection from chunk_metadata")
    ap.add_argument("--collection-slug", default=None, help="Single collection; ignored if --all-collections")
    ap.add_argument("--all-collections", action="store_true", help="Process both alias-scoped and copy paths")
    ap.add_argument("--batch-size", type=int, default=64, help="Chunks per batch")
    ap.add_argument("--concurrency", type=int, default=None, help="Parallel API requests per batch (EMBED_CONCURRENCY)")
    ap.add_argument("--rebuild", action="store_true", help="Delete existing canonical rows for target set first")
    ap.add_argument("--fill-missing-only", action="store_true", help="Only process chunks without canonical row")
    ap.add_argument("--limit", type=int, default=None, help="Max chunks per collection (for testing)")
    ap.add_argument("--dry-run", action="store_true", help="Canonicalize and log; skip embed and DB write")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    args = ap.parse_args()

    if args.rebuild and args.fill_missing_only:
        raise RuntimeError("Choose only one: --rebuild OR --fill-missing-only")

    if args.concurrency is not None:
        os.environ["EMBED_CONCURRENCY"] = str(args.concurrency)
    if os.getenv("EMBED_CONCURRENCY", "1") != "1":
        print(f"  [concurrency] EMBED_CONCURRENCY={os.getenv('EMBED_CONCURRENCY')} parallel requests per batch", file=sys.stderr)

    conn = get_conn()
    conn.autocommit = False

    try:
        if args.all_collections:
            # Alias-scoped: discover pipeline_version per collection (or use --chunk-pv override)
            with conn.cursor() as cur:
                alias_pairs = fetch_collection_pipeline_versions(cur, list(ALIAS_SCOPED), args.chunk_pv)
            for coll, pv in alias_pairs:
                if args.rebuild:
                    with conn.cursor() as cur:
                        cur.execute("""
                            DELETE FROM chunk_embeddings_canonical cec
                            WHERE cec.chunk_id IN (
                                SELECT cm.chunk_id FROM chunk_metadata cm
                                WHERE cm.collection_slug = %s AND cm.pipeline_version = %s
                            )
                            AND cec.pipeline_version = %s
                            AND cec.embedding_model = %s
                        """, (coll, pv, pv, args.embedding_model))
                    conn.commit()
                run_alias_scoped_path(
                    conn, pv, coll,
                    args.batch_size, args.rebuild, args.fill_missing_only,
                    args.limit, args.dry_run, args.embedding_model, args.sleep,
                )
            # Copy path: all collections except alias-scoped
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT collection_slug FROM chunk_metadata
                    WHERE collection_slug IS NOT NULL
                    AND collection_slug NOT IN %s
                """, (tuple(ALIAS_SCOPED),))
                other_colls = [r[0] for r in cur.fetchall()]
            if other_colls:
                if args.rebuild:
                    with conn.cursor() as cur:
                        pairs = fetch_collection_pipeline_versions(cur, other_colls, args.chunk_pv)
                        for coll, pv in pairs:
                            cur.execute("""
                                DELETE FROM chunk_embeddings_canonical cec
                                WHERE cec.chunk_id IN (
                                    SELECT cm.chunk_id FROM chunk_metadata cm
                                    WHERE cm.collection_slug = %s AND cm.pipeline_version = %s
                                )
                                AND cec.pipeline_version = %s
                                AND cec.embedding_model = %s
                            """, (coll, pv, pv, args.embedding_model))
                    conn.commit()
                run_copy_path(
                    conn, None, other_colls,  # discover pipeline_version per collection
                    args.limit, args.dry_run, args.embedding_model, args.batch_size,
                )
        else:
            coll = args.collection_slug or "venona"
            if coll in ALIAS_SCOPED:
                with conn.cursor() as cur:
                    alias_pairs = fetch_collection_pipeline_versions(cur, [coll], args.chunk_pv)
                for c, pv in alias_pairs:
                    if args.rebuild:
                        with conn.cursor() as cur:
                            cur.execute("""
                                DELETE FROM chunk_embeddings_canonical cec
                                WHERE cec.chunk_id IN (
                                    SELECT cm.chunk_id FROM chunk_metadata cm
                                    WHERE cm.collection_slug = %s AND cm.pipeline_version = %s
                                )
                                AND cec.pipeline_version = %s
                                AND cec.embedding_model = %s
                            """, (c, pv, pv, args.embedding_model))
                        conn.commit()
                    run_alias_scoped_path(
                        conn, pv, c,
                        args.batch_size, args.rebuild, args.fill_missing_only,
                        args.limit, args.dry_run, args.embedding_model, args.sleep,
                    )
            else:
                if args.rebuild:
                    with conn.cursor() as cur:
                        pairs = fetch_collection_pipeline_versions(cur, [coll], args.chunk_pv)
                        for c, pv in pairs:
                            cur.execute("""
                                DELETE FROM chunk_embeddings_canonical cec
                                WHERE cec.chunk_id IN (
                                    SELECT cm.chunk_id FROM chunk_metadata cm
                                    WHERE cm.collection_slug = %s AND cm.pipeline_version = %s
                                )
                                AND cec.pipeline_version = %s
                                AND cec.embedding_model = %s
                            """, (c, pv, pv, args.embedding_model))
                    conn.commit()
                run_copy_path(
                    conn, None, [coll],  # discover pipeline_version from chunk_metadata
                    args.limit, args.dry_run, args.embedding_model, args.batch_size,
                )
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
