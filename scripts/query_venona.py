import os
import argparse
import psycopg2


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def embed_query(text: str):
    from openai import OpenAI  # pip install openai
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    vec = resp.data[0].embedding
    if len(vec) != 1536:
        raise RuntimeError(f"Query embedding dim {len(vec)} != 1536 (expected vector(1536))")
    return vec


def vector_literal(vec):
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Query text (English or Russian)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk-pv", default="chunk_v1_full")
    ap.add_argument("--probes", type=int, default=10, help="ivfflat probes (higher = better recall)")
    ap.add_argument(
        "--mode",
        choices=["vector", "lex", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    ap.add_argument("--top-n-vec", type=int, default=200, help="Candidate pool from vector search (hybrid)")
    ap.add_argument("--top-n-lex", type=int, default=200, help="Candidate pool from lexical search (hybrid)")
    ap.add_argument("--rrf-k", type=int, default=50, help="RRF constant (larger downweights rank differences)")
    ap.add_argument("--preview-chars", type=int, default=2000, help="How many chars of chunk text to print")
    args = ap.parse_args()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Ensure ivfflat recall is decent when index is present
            cur.execute("SET ivfflat.probes = %s;", (args.probes,))

            # --- Lexical-only (no embeddings needed) ---
            if args.mode == "lex":
                # websearch_to_tsquery handles natural language queries better than plainto_tsquery.
                # If your Postgres is old and this errors, switch to plainto_tsquery.
                cur.execute(
                    f"""
                    SELECT
                      c.id,
                      cm.ussr_ref_no_set,
                      cm.sender_set,
                      cm.recipient_set,
                      cm.date_min,
                      cm.date_max,
                      LEFT(c.text, %s) AS preview,
                      ts_rank_cd(to_tsvector('simple', c.text),
                                 websearch_to_tsquery('simple', %s)) AS score,
                      NULL::int AS r_vec,
                      NULL::int AS r_lex
                    FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE c.pipeline_version = %s
                      AND cm.pipeline_version = %s
                      AND cm.collection_slug = 'venona'
                      AND to_tsvector('simple', c.text) @@ websearch_to_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s;
                    """,
                    (
                        args.preview_chars,
                        args.query,
                        args.chunk_pv,
                        args.chunk_pv,
                        args.query,
                        args.k,
                    ),
                )
                rows = cur.fetchall()

            else:
                # Vector needed for vector-only or hybrid
                qvec = embed_query(args.query)
                vec_lit = vector_literal(qvec)

                if args.mode == "vector":
                    cur.execute(
                        """
                        SELECT
                          c.id,
                          cm.ussr_ref_no_set,
                          cm.sender_set,
                          cm.recipient_set,
                          cm.date_min,
                          cm.date_max,
                          LEFT(c.text, %s) AS preview,
                          (c.embedding <=> %s::vector) AS score,
                          NULL::int AS r_vec,
                          NULL::int AS r_lex
                        FROM chunks c
                        JOIN chunk_metadata cm ON cm.chunk_id = c.id
                        WHERE c.pipeline_version = %s
                          AND cm.pipeline_version = %s
                          AND cm.collection_slug = 'venona'
                          AND c.embedding IS NOT NULL
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (
                            args.preview_chars,
                            vec_lit,
                            args.chunk_pv,
                            args.chunk_pv,
                            vec_lit,
                            args.k,
                        ),
                    )
                    rows = cur.fetchall()

                else:
                    # --- Hybrid via Reciprocal Rank Fusion (RRF) ---
                    # - vec: top-N by vector similarity
                    # - lex: top-N by full-text search from the same user query
                    # - fused score: 1/(k + rank_vec) + 1/(k + rank_lex)
                    cur.execute(
                        f"""
                        WITH
                        vec AS (
                          SELECT
                            c.id,
                            row_number() OVER (ORDER BY c.embedding <=> %(qvec)s::vector) AS r_vec
                          FROM chunks c
                          JOIN chunk_metadata cm ON cm.chunk_id = c.id
                          WHERE c.pipeline_version = %(pv)s
                            AND cm.pipeline_version = %(pv)s
                            AND cm.collection_slug = 'venona'
                            AND c.embedding IS NOT NULL
                          ORDER BY c.embedding <=> %(qvec)s::vector
                          LIMIT %(top_n_vec)s
                        ),
                        lex AS (
                          SELECT
                            c.id,
                            row_number() OVER (
                              ORDER BY ts_rank_cd(
                                to_tsvector('simple', c.text),
                                websearch_to_tsquery('simple', %(qtxt)s)
                              ) DESC
                            ) AS r_lex
                          FROM chunks c
                          JOIN chunk_metadata cm ON cm.chunk_id = c.id
                          WHERE c.pipeline_version = %(pv)s
                            AND cm.pipeline_version = %(pv)s
                            AND cm.collection_slug = 'venona'
                            AND to_tsvector('simple', c.text) @@ websearch_to_tsquery('simple', %(qtxt)s)
                          LIMIT %(top_n_lex)s
                        ),
                        fused AS (
                          SELECT
                            COALESCE(vec.id, lex.id) AS id,
                            COALESCE(1.0 / (%(rrf_k)s + vec.r_vec), 0.0) +
                            COALESCE(1.0 / (%(rrf_k)s + lex.r_lex), 0.0) AS score,
                            vec.r_vec,
                            lex.r_lex
                          FROM vec
                          FULL OUTER JOIN lex ON lex.id = vec.id
                        )
                        SELECT
                          c.id,
                          cm.ussr_ref_no_set,
                          cm.sender_set,
                          cm.recipient_set,
                          cm.date_min,
                          cm.date_max,
                          LEFT(c.text, %(preview_chars)s) AS preview,
                          fused.score,
                          fused.r_vec,
                          fused.r_lex
                        FROM fused
                        JOIN chunks c ON c.id = fused.id
                        JOIN chunk_metadata cm ON cm.chunk_id = c.id
                        ORDER BY fused.score DESC
                        LIMIT %(k)s;
                        """,
                        {
                            "qtxt": args.query,
                            "qvec": vec_lit,
                            "pv": args.chunk_pv,
                            "k": args.k,
                            "top_n_vec": args.top_n_vec,
                            "top_n_lex": args.top_n_lex,
                            "rrf_k": args.rrf_k,
                            "preview_chars": args.preview_chars,
                        },
                    )
                    rows = cur.fetchall()

        print(f"\n=== Top matches ({args.mode}) ===\n")
        for i, r in enumerate(rows, 1):
            (cid, refset, senderset, recipset, dmin, dmax, preview, score, r_vec, r_lex) = r
            if args.mode == "vector":
                print(f"[{i}] chunk_id={cid}  distance={float(score):.4f}")
            else:
                # lex + hybrid use "score" where higher is better
                print(f"[{i}] chunk_id={cid}  score={float(score):.6f}  r_vec={r_vec}  r_lex={r_lex}")

            print(f"    ussr_ref_no={refset}")
            print(f"    sender={senderset}  recipient={recipset}")
            print(f"    date_min={dmin}  date_max={dmax}")
            print(f"    preview: {preview.replace('\\n',' ')}")
            print()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
