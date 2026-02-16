#!/usr/bin/env python3
"""
Embed Vassiliev chunks into chunks.embedding (vector(1536)).

Standalone, modeled after embed_venona.py, but adds:
  - default collection_slug=vassiliev
  - skip-short policy (so reruns don't keep selecting trivial chunks)
  - optional embedding metadata columns if you added them (embedding_model, embedded_at, embedding_dim, embedding_status)

Usage examples:
  # Fill missing only (recommended)
  EMBED_PROVIDER=openai OPENAI_API_KEY=... \
    python scripts/embed_vassiliev_chunks.py --fill-missing-only

  # Rebuild (dangerous): wipe then re-embed
  python scripts/embed_vassiliev_chunks.py --rebuild

  # Embed only the first 2000 chunks
  python scripts/embed_vassiliev.py --fill-missing-only --limit 2000

Notes:
  - This assumes chunks.embedding is vector(1536) and matches text-embedding-3-small.
  - If you haven't added embedding metadata columns, this script will still work.
"""

import os
import time
import argparse
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras


# Use shared embed_texts (supports EMBED_CONCURRENCY, truncation)
from scripts.embed_venona_chunks import embed_texts


# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL env var (e.g. postgres://user:pass@host:5432/db)")
    return psycopg2.connect(dsn)


def vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def detect_optional_columns(cur) -> dict:
    """
    Detect whether optional embedding metadata columns exist on chunks.
    Returns dict with booleans:
      embedding_model, embedding_dim, embedded_at, embedding_status
    """
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='chunks'
          AND column_name IN ('embedding_model','embedding_dim','embedded_at','embedding_status');
        """
    )
    cols = {r[0] for r in cur.fetchall()}
    return {
        "embedding_model": "embedding_model" in cols,
        "embedding_dim": "embedding_dim" in cols,
        "embedded_at": "embedded_at" in cols,
        "embedding_status": "embedding_status" in cols,
    }


def fetch_target_chunks(
    cur,
    chunk_pv: str,
    collection_slug: str,
    fill_missing_only: bool,
    min_chars: int,
    limit: Optional[int],
) -> List[Tuple[int, str]]:
    """
    Deterministic selection for embedding.
    Key point: exclude short chunks in SQL so reruns are stable.
    """
    where = """
        WHERE c.pipeline_version = %s
          AND cm.pipeline_version = %s
          AND cm.collection_slug = %s
          AND length(btrim(c.text)) >= %s
    """
    if fill_missing_only:
        where += " AND c.embedding IS NULL"

    lim = " LIMIT %s" if limit else ""
    sql = f"""
        SELECT c.id, c.text
        FROM chunks c
        JOIN chunk_metadata cm
          ON cm.chunk_id = c.id
        {where}
        ORDER BY c.id
        {lim}
    """
    params = [chunk_pv, chunk_pv, collection_slug, min_chars]
    if limit:
        params.append(limit)

    cur.execute(sql, params)
    return [(r[0], r[1] or "") for r in cur.fetchall()]


def mark_skipped_short(
    cur,
    chunk_pv: str,
    collection_slug: str,
    min_chars: int,
    opt_cols: dict,
    model_name: str,
):
    """
    Mark short chunks so they don't keep showing up in "missing" checks.
    Only runs if embedding_status exists.
    """
    if not opt_cols.get("embedding_status", False):
        return

    # Mark only rows in target set that are short and still unembedded.
    now = datetime.now(timezone.utc)
    set_parts = ["embedding_status = 'SKIPPED_SHORT'"]

    if opt_cols.get("embedded_at", False):
        set_parts.append("embedded_at = %s")
    if opt_cols.get("embedding_model", False):
        # Optional: keep model for auditability; you can change to 'SKIPPED_SHORT' if preferred.
        set_parts.append("embedding_model = %s")
    if opt_cols.get("embedding_dim", False):
        set_parts.append("embedding_dim = %s")

    set_sql = ", ".join(set_parts)

    params = []
    if opt_cols.get("embedded_at", False):
        params.append(now)
    if opt_cols.get("embedding_model", False):
        params.append(model_name)
    if opt_cols.get("embedding_dim", False):
        params.append(1536)

    params.extend([chunk_pv, chunk_pv, collection_slug, min_chars])

    sql = f"""
        UPDATE chunks c
        SET {set_sql}
        FROM chunk_metadata cm
        WHERE cm.chunk_id = c.id
          AND c.pipeline_version = %s
          AND cm.pipeline_version = %s
          AND cm.collection_slug = %s
          AND length(btrim(c.text)) < %s
          AND c.embedding IS NULL
          AND (c.embedding_status IS NULL OR c.embedding_status <> 'SKIPPED_SHORT');
    """
    cur.execute(sql, params)


def null_out_embeddings(cur, chunk_pv: str, collection_slug: str):
    sql = """
        UPDATE chunks c
        SET embedding = NULL
        FROM chunk_metadata cm
        WHERE cm.chunk_id = c.id
          AND c.pipeline_version = %s
          AND cm.pipeline_version = %s
          AND cm.collection_slug = %s
    """
    cur.execute(sql, [chunk_pv, chunk_pv, collection_slug])


def update_embeddings(cur, ids: List[int], vectors: List[List[float]], opt_cols: dict, model_name: str):
    assert len(ids) == len(vectors)
    now = datetime.now(timezone.utc)

    expected_dim = 1536
    for v in vectors:
        if len(v) != expected_dim:
            raise RuntimeError(f"Embedding dim {len(v)} != {expected_dim}. Check model/provider.")

    # Build VALUES rows dynamically based on optional columns
    # Always: id, vec
    # Optional: embedding_model, embedding_dim, embedded_at, embedding_status
    rows = []
    for cid, vec in zip(ids, vectors):
        row = [cid, vector_literal(vec)]
        if opt_cols.get("embedding_model", False):
            row.append(model_name)
        if opt_cols.get("embedding_dim", False):
            row.append(expected_dim)
        if opt_cols.get("embedded_at", False):
            row.append(now)
        if opt_cols.get("embedding_status", False):
            row.append("OK")
        rows.append(tuple(row))

    # Construct template and UPDATE SET
    cols = ["id", "vec"]
    if opt_cols.get("embedding_model", False):
        cols.append("embedding_model")
    if opt_cols.get("embedding_dim", False):
        cols.append("embedding_dim")
    if opt_cols.get("embedded_at", False):
        cols.append("embedded_at")
    if opt_cols.get("embedding_status", False):
        cols.append("embedding_status")

    # SET clause
    set_parts = ["embedding = v.vec::vector"]
    if opt_cols.get("embedding_model", False):
        set_parts.append("embedding_model = v.embedding_model")
    if opt_cols.get("embedding_dim", False):
        set_parts.append("embedding_dim = v.embedding_dim")
    if opt_cols.get("embedded_at", False):
        set_parts.append("embedded_at = v.embedded_at")
    if opt_cols.get("embedding_status", False):
        set_parts.append("embedding_status = v.embedding_status")

    set_sql = ", ".join(set_parts)
    col_sql = ", ".join(cols)

    # psycopg2 execute_values template: number of %s should match row length
    template = "(" + ",".join(["%s"] * len(cols)) + ")"

    sql = f"""
        UPDATE chunks AS c
        SET {set_sql}
        FROM (VALUES %s) AS v({col_sql})
        WHERE c.id = v.id
    """
    psycopg2.extras.execute_values(cur, sql, rows, template=template)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-pv", default="chunk_v1_full")
    ap.add_argument("--collection-slug", default="vassiliev")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=None, help="Parallel API requests per batch (EMBED_CONCURRENCY)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true", help="NULL out then recompute embeddings for target set")
    ap.add_argument("--fill-missing-only", action="store_true", help="Only embed rows where embedding IS NULL")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between batches (rate limit)")
    ap.add_argument("--min-chars", type=int, default=40, help="Skip chunks shorter than this (trimmed) length")
    args = ap.parse_args()

    if args.concurrency is not None:
        os.environ["EMBED_CONCURRENCY"] = str(args.concurrency)

    if args.rebuild and args.fill_missing_only:
        raise RuntimeError("Choose only one: --rebuild OR --fill-missing-only")

    # Keep model name explicit (audit + consistency)
    model_name = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    conn = get_conn()
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            opt_cols = detect_optional_columns(cur)
            print(f"[cols] optional columns present: {opt_cols}")

            if args.rebuild:
                print(f"[rebuild] NULLing embeddings for chunk_pv={args.chunk_pv}, collection={args.collection_slug}")
                null_out_embeddings(cur, args.chunk_pv, args.collection_slug)
                conn.commit()

            # Mark short chunks as skipped (only if embedding_status exists)
            mark_skipped_short(cur, args.chunk_pv, args.collection_slug, args.min_chars, opt_cols, model_name)
            conn.commit()

            fill_missing = True if args.fill_missing_only else False
            targets = fetch_target_chunks(
                cur,
                chunk_pv=args.chunk_pv,
                collection_slug=args.collection_slug,
                fill_missing_only=fill_missing,
                min_chars=args.min_chars,
                limit=args.limit,
            )

        total = len(targets)
        print(f"Found {total} chunks to embed (chunk_pv={args.chunk_pv}, collection={args.collection_slug})")

        # Batch loop
        for i in range(0, total, args.batch_size):
            batch = targets[i : i + args.batch_size]
            ids = [cid for cid, _ in batch]
            texts = [t for _, t in batch]

            vectors = embed_texts(texts)

            with conn.cursor() as cur:
                update_embeddings(cur, ids, vectors, opt_cols, model_name)
            conn.commit()

            print(f"Embedded {min(i + args.batch_size, total)}/{total}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        print("✅ Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
