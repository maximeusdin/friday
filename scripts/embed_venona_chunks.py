import os
import time
import argparse
import json
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras


# -----------------------------
# Embedding provider (pluggable)
# -----------------------------

# OpenAI embedding models have token limits:
# - text-embedding-3-small: 8191 tokens
# - text-embedding-3-large: 8191 tokens
# Very conservative: ~2.2 chars per token worst case, so ~18000 chars to be safe
MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "18000"))


def truncate_text(text: str, max_chars: int = MAX_EMBED_CHARS) -> Tuple[str, bool]:
    """
    Truncate text to max_chars if needed. Returns (text, was_truncated).
    Tries to truncate at a sentence or word boundary.
    """
    if len(text) <= max_chars:
        return text, False
    
    # Try to find a good break point
    truncated = text[:max_chars]
    
    # Try to break at last sentence
    for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars * 0.8:  # Only if we keep at least 80%
            return truncated[:last_sep + len(sep)].rstrip(), True
    
    # Fallback: break at last space
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        return truncated[:last_space], True
    
    return truncated, True


def embed_texts(texts: List[str], verbose: bool = True) -> List[List[float]]:
    """
    Replace this function with your actual embedding provider.

    Recommended pattern:
      - Put provider logic behind env vars
      - Keep outputs deterministic for identical input strings
      - Keep model + dimension stable while embedding into a given column

    For now, this raises with instructions.
    """
    provider = os.getenv("EMBED_PROVIDER", "").lower().strip()

    if provider in ("openai", "oai"):
        # Minimal OpenAI example (requires: pip install openai)
        # and env var OPENAI_API_KEY.
        from openai import OpenAI  # type: ignore

        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        client = OpenAI()

        # Truncate oversized texts
        processed_texts = []
        truncated_count = 0
        for t in texts:
            truncated, was_truncated = truncate_text(t)
            processed_texts.append(truncated)
            if was_truncated:
                truncated_count += 1
        
        if truncated_count > 0 and verbose:
            print(f"  [truncated {truncated_count}/{len(texts)} oversized chunks]", end="", flush=True)

        # OpenAI supports batching; keep batch sizes reasonable.
        resp = client.embeddings.create(model=model, input=processed_texts)
        return [d.embedding for d in resp.data]

    raise RuntimeError(
        "No embedding provider configured.\n"
        "Set EMBED_PROVIDER=openai and OPENAI_API_KEY, or replace embed_texts() "
        "with your local embedding model (sentence-transformers, etc.)."
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
    # pgvector accepts: '[1,2,3]'::vector
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def fetch_target_chunks(
    cur,
    chunk_pv: str,
    collection_slug: str,
    fill_missing_only: bool,
    limit: Optional[int],
) -> List[Tuple[int, str]]:
    where = """
        WHERE c.pipeline_version = %s
          AND cm.pipeline_version = %s
          AND cm.collection_slug = %s
    """
    # We join chunk_metadata to filter by collection_slug deterministically.
    # (You could also join documents/collections; chunk_metadata is already computed and stable.)
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
    params = [chunk_pv, chunk_pv, collection_slug]
    if limit:
        params.append(limit)

    cur.execute(sql, params)
    return [(r[0], r[1] or "") for r in cur.fetchall()]


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


def update_embeddings(cur, ids: List[int], vectors: List[List[float]]):
    assert len(ids) == len(vectors)

    # Use execute_values for speed.
    rows = [(cid, vector_literal(vec)) for cid, vec in zip(ids, vectors)]

    # embedding = v.vec::vector
    sql = """
        UPDATE chunks AS c
        SET embedding = v.vec::vector
        FROM (VALUES %s) AS v(id, vec)
        WHERE c.id = v.id
    """
    psycopg2.extras.execute_values(cur, sql, rows, template="(%s, %s)")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-pv", default="chunk_v1_full")
    ap.add_argument("--collection-slug", default="venona")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true", help="NULL out then recompute embeddings for target set")
    ap.add_argument("--fill-missing-only", action="store_true", help="Only embed rows where embedding IS NULL")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between batches (rate limit)")
    args = ap.parse_args()

    if args.rebuild and args.fill_missing_only:
        raise RuntimeError("Choose only one: --rebuild OR --fill-missing-only")

    conn = get_conn()
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            if args.rebuild:
                print(f"[rebuild] NULLing embeddings for chunk_pv={args.chunk_pv}, collection={args.collection_slug}")
                null_out_embeddings(cur, args.chunk_pv, args.collection_slug)
                conn.commit()

            fill_missing = True if args.fill_missing_only else False
            targets = fetch_target_chunks(
                cur,
                chunk_pv=args.chunk_pv,
                collection_slug=args.collection_slug,
                fill_missing_only=fill_missing,
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

            # Dimension check (must match vector(1536))
            expected_dim = 1536
            for v in vectors:
                if len(v) != expected_dim:
                    raise RuntimeError(f"Embedding dim {len(v)} != {expected_dim}. Check model / provider.")


            with conn.cursor() as cur:
                update_embeddings(cur, ids, vectors)
            conn.commit()

            print(f"Embedded {min(i + args.batch_size, total)}/{total}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        print("✅ Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
