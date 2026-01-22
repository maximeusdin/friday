#!/usr/bin/env python3
import os
import time
import argparse
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras


# -----------------------------
# Embedding provider (pluggable)
# -----------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    provider = os.getenv("EMBED_PROVIDER", "").lower().strip()

    if provider in ("openai", "oai"):
        from openai import OpenAI  # type: ignore

        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        client = OpenAI()
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    raise RuntimeError(
        "No embedding provider configured.\n"
        "Set EMBED_PROVIDER=openai and OPENAI_API_KEY, or replace embed_texts()."
    )


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


def table_has_column(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s AND column_name=%s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def fetch_target_chunks(
    cur,
    chunk_pv: str,
    collection_slug: str,
    fill_missing_only: bool,
    min_chars: int,
    prefer_clean_text: bool,
    limit: Optional[int],
) -> List[Tuple[int, str]]:
    """
    Returns list of (chunk_id, text_to_embed).
    Uses chunk_metadata to filter by collection deterministically.
    """
    text_expr = "c.text"
    if prefer_clean_text and table_has_column(cur, "chunks", "clean_text"):
        # prefer clean_text but fall back to text
        text_expr = "COALESCE(NULLIF(c.clean_text, ''), c.text)"

    where = f"""
        WHERE c.pipeline_version = %s
          AND cm.pipeline_version = %s
          AND cm.collection_slug = %s
          AND length(trim({text_expr})) >= %s
    """
    if fill_missing_only:
        where += " AND c.embedding IS NULL"

    lim = " LIMIT %s" if limit else ""
    sql = f"""
        SELECT c.id, {text_expr} AS embed_text
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
    return [(int(r[0]), r[1] or "") for r in cur.fetchall()]


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


def update_embeddings_only(cur, ids: List[int], vectors: List[List[float]]):
    rows = [(cid, vector_literal(vec)) for cid, vec in zip(ids, vectors)]
    sql = """
        UPDATE chunks AS c
        SET embedding = v.vec::vector
        FROM (VALUES %s) AS v(id, vec)
        WHERE c.id = v.id
    """
    psycopg2.extras.execute_values(cur, sql, rows, template="(%s, %s)")


def update_embeddings_with_audit(
    cur,
    ids: List[int],
    vectors: List[List[float]],
    model_name: str,
    dim: int,
):
    rows = [(cid, vector_literal(vec), model_name, dim) for cid, vec in zip(ids, vectors)]
    sql = """
        UPDATE chunks AS c
        SET
          embedding = v.vec::vector,
          embedding_model = v.model,
          embedding_dim = v.dim,
          embedded_at = NOW(),
          embedding_status = 'OK'
        FROM (VALUES %s) AS v(id, vec, model, dim)
        WHERE c.id = v.id
    """
    psycopg2.extras.execute_values(cur, sql, rows, template="(%s, %s, %s, %s)")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-pv", default="chunk_v1_full")
    ap.add_argument("--collection-slug", default="silvermaster")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--prefer-clean-text", action="store_true", help="Embed clean_text if available (recommended for Silvermaster)")
    ap.add_argument("--rebuild", action="store_true", help="NULL out then recompute embeddings for target set")
    ap.add_argument("--fill-missing-only", action="store_true", help="Only embed rows where embedding IS NULL")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between batches (rate limit)")
    args = ap.parse_args()

    if args.rebuild and args.fill_missing_only:
        raise RuntimeError("Choose only one: --rebuild OR --fill-missing-only")

    model_name = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    expected_dim = 1536  # matches vector(1536)

    conn = get_conn()
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            if args.rebuild:
                print(f"[rebuild] NULLing embeddings for chunk_pv={args.chunk_pv}, collection={args.collection_slug}")
                null_out_embeddings(cur, args.chunk_pv, args.collection_slug)
                conn.commit()

            targets = fetch_target_chunks(
                cur,
                chunk_pv=args.chunk_pv,
                collection_slug=args.collection_slug,
                fill_missing_only=bool(args.fill_missing_only),
                min_chars=int(args.min_chars),
                prefer_clean_text=bool(args.prefer_clean_text),
                limit=args.limit,
            )

            # detect audit columns
            has_audit = all(
                table_has_column(cur, "chunks", col)
                for col in ("embedding_model", "embedding_dim", "embedded_at", "embedding_status")
            )

        total = len(targets)
        print(
            f"Found {total} chunks to embed "
            f"(chunk_pv={args.chunk_pv}, collection={args.collection_slug}, min_chars={args.min_chars}, "
            f"prefer_clean_text={args.prefer_clean_text})"
        )

        for i in range(0, total, args.batch_size):
            batch = targets[i : i + args.batch_size]
            ids = [cid for cid, _ in batch]
            texts = [t for _, t in batch]

            vectors = embed_texts(texts)

            for v in vectors:
                if len(v) != expected_dim:
                    raise RuntimeError(f"Embedding dim {len(v)} != {expected_dim}. Check model/provider.")

            with conn.cursor() as cur:
                if has_audit:
                    update_embeddings_with_audit(cur, ids, vectors, model_name=model_name, dim=expected_dim)
                else:
                    update_embeddings_only(cur, ids, vectors)

            conn.commit()
            print(f"Embedded {min(i + args.batch_size, total)}/{total}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        print("✅ Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
