import os
import re
import argparse
import sys
from pathlib import Path
# Ensure repo root is importable when running as a script: `python scripts/query_chunks.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2

from retrieval.ops import concordance_expand_terms, build_expanded_query_string


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


def fts_or_query(user_query: str) -> str:
    """
    Build a tsquery that ORs tokens together, so lexical doesn't go empty
    on natural language queries. Generic across sources.
    """
    q = user_query.lower()
    # normalize common forms
    q = q.replace("u.s.", "usa").replace("u.s", "usa").replace("us ", "usa ")
    toks = re.findall(r"[a-z0-9_']+", q)
    # keep tokens that are informative; allow 2-char if it's digits
    toks = [t for t in toks if (len(t) >= 3 or t.isdigit())]
    toks = sorted(set(toks))
    if not toks:
        return "___nomatch___"
    return " | ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Query text (any language)")
    ap.add_argument("--k", type=int, default=10)

    # Pipeline versioning
    ap.add_argument("--chunk-pv", default="chunk_v1_full")
    ap.add_argument("--meta-pv", default=None, help="chunk_metadata pipeline_version (defaults to --chunk-pv)")

    # Retrieval mode
    ap.add_argument("--mode", choices=["vector", "lex", "hybrid"], default="hybrid")

    # Query-time concordance expansion (enabled by default)
    ap.add_argument("--no-expand-concordance", action="store_true", 
                    help="Disable concordance expansion (default: enabled)")
    ap.add_argument("--concordance-source-slug", default="venona_vassiliev_concordance_v3",
                    help="concordance_sources.slug to use for expansion")
    ap.add_argument("--print-expansions", action="store_true", help="Print expansions and exit (no retrieval)")

    # ANN + fusion knobs
    ap.add_argument("--probes", type=int, default=20, help="ivfflat probes (higher = better recall)")
    ap.add_argument("--top-n-vec", type=int, default=200, help="candidate pool from vector search (hybrid)")
    ap.add_argument("--top-n-lex", type=int, default=200, help="candidate pool from lexical search (hybrid)")
    ap.add_argument("--rrf-k", type=int, default=50, help="RRF constant")

    # Output
    ap.add_argument("--preview-chars", type=int, default=2000)

    # Optional filters (generalizable)
    ap.add_argument("--collection-slug", action="append", default=[],
                    help="Optional filter. Repeatable: --collection-slug venona --collection-slug vassiliev")
    ap.add_argument("--document-id", type=int, default=None, help="Optional filter to a specific document_id")
    ap.add_argument("--date-from", default=None, help="Optional filter (YYYY-MM-DD), uses chunk_metadata.date_max >= date-from")
    ap.add_argument("--date-to", default=None, help="Optional filter (YYYY-MM-DD), uses chunk_metadata.date_min <= date-to")

    args = ap.parse_args()
    meta_pv = args.meta_pv or args.chunk_pv

    # Build generic WHERE filters
    where = [
        "c.pipeline_version = %(chunk_pv)s",
        "cm.pipeline_version = %(meta_pv)s",
    ]
    params = {
        "chunk_pv": args.chunk_pv,
        "meta_pv": meta_pv,
        "k": args.k,
        "preview_chars": args.preview_chars,
        "top_n_vec": args.top_n_vec,
        "top_n_lex": args.top_n_lex,
        "rrf_k": args.rrf_k,
    }

    if args.collection_slug:
        where.append("cm.collection_slug = ANY(%(collection_slugs)s)")
        params["collection_slugs"] = args.collection_slug

    if args.document_id is not None:
        where.append("cm.document_id = %(document_id)s")
        params["document_id"] = args.document_id

    if args.date_from:
        # include chunks that might overlap the range
        where.append("(cm.date_max IS NULL OR cm.date_max >= %(date_from)s::date)")
        params["date_from"] = args.date_from

    if args.date_to:
        where.append("(cm.date_min IS NULL OR cm.date_min <= %(date_to)s::date)")
        params["date_to"] = args.date_to

    where_sql = " AND ".join(where)

    conn = get_conn()
    try:
        # Expansion is ON by default (can disable with --no-expand-concordance)
        expand_concordance = not args.no_expand_concordance
        exp_terms = []
        if expand_concordance:
            exp_terms = concordance_expand_terms(conn, args.query, source_slug=args.concordance_source_slug)

        if args.print_expansions:
            print(f"query={args.query!r}")
            print(f"expand_concordance={expand_concordance} source_slug={args.concordance_source_slug!r}")
            print("expansions:")
            for t in exp_terms:
                print(f"  - {t}")
            return

        with conn.cursor() as cur:
            cur.execute("SET ivfflat.probes = %s;", (args.probes,))

            # Lexical augmentation: include expansions in the token stream (hybrid/lex modes)
            lex_aug = args.query if not exp_terms else (args.query + " " + " ".join(exp_terms))
            tsq = fts_or_query(lex_aug)
            params["tsq"] = tsq
            params["qtxt"] = args.query

            rows = []

            if args.mode == "lex":
                cur.execute(
                    f"""
                    SELECT
                      c.id,
                      cm.collection_slug,
                      cm.document_id,
                      cm.first_page_id,
                      cm.last_page_id,
                      cm.date_min,
                      cm.date_max,
                      LEFT(c.text, %(preview_chars)s) AS preview,
                      ts_rank_cd(to_tsvector('simple', c.text),
                                 to_tsquery('simple', %(tsq)s)) AS score,
                      NULL::int AS r_vec,
                      NULL::int AS r_lex
                    FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE {where_sql}
                      AND to_tsvector('simple', c.text) @@ to_tsquery('simple', %(tsq)s)
                    ORDER BY score DESC
                    LIMIT %(k)s;
                    """,
                    params,
                )
                rows = cur.fetchall()

            elif args.mode == "vector":
                query_for_embedding = args.query
                if exp_terms:
                    query_for_embedding = build_expanded_query_string(args.query, exp_terms)

                qvec = embed_query(query_for_embedding)
                params["qvec"] = vector_literal(qvec)
                cur.execute(
                    f"""
                    SELECT
                      c.id,
                      cm.collection_slug,
                      cm.document_id,
                      cm.first_page_id,
                      cm.last_page_id,
                      cm.date_min,
                      cm.date_max,
                      LEFT(c.text, %(preview_chars)s) AS preview,
                      (c.embedding <=> %(qvec)s::vector) AS score,
                      NULL::int AS r_vec,
                      NULL::int AS r_lex
                    FROM chunks c
                    JOIN chunk_metadata cm ON cm.chunk_id = c.id
                    WHERE {where_sql}
                      AND c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> %(qvec)s::vector
                    LIMIT %(k)s;
                    """,
                    params,
                )
                rows = cur.fetchall()

            else:
                # HYBRID
                query_for_embedding = args.query
                if exp_terms:
                    query_for_embedding = build_expanded_query_string(args.query, exp_terms)

                qvec = embed_query(query_for_embedding)
                params["qvec"] = vector_literal(qvec)

                cur.execute(
                    f"""
                    WITH
                    vec AS (
                      SELECT
                        c.id,
                        row_number() OVER (ORDER BY c.embedding <=> %(qvec)s::vector) AS r_vec
                      FROM chunks c
                      JOIN chunk_metadata cm ON cm.chunk_id = c.id
                      WHERE {where_sql}
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
                            to_tsquery('simple', %(tsq)s)
                          ) DESC
                        ) AS r_lex
                      FROM chunks c
                      JOIN chunk_metadata cm ON cm.chunk_id = c.id
                      WHERE {where_sql}
                        AND to_tsvector('simple', c.text) @@ to_tsquery('simple', %(tsq)s)
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
                      cm.collection_slug,
                      cm.document_id,
                      cm.first_page_id,
                      cm.last_page_id,
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
                    params,
                )
                rows = cur.fetchall()

        # Display expansion info if used
        if expand_concordance and exp_terms:
            expanded_str = build_expanded_query_string(args.query, exp_terms)
            print(f"\n=== Query Expansion ===")
            print(f"Original query: {args.query!r}")
            print(f"Expanded query: {expanded_str[:200]}")
            print(f"Expansion terms ({len(exp_terms)}): {', '.join(exp_terms[:5])}")
            if len(exp_terms) > 5:
                print(f"  ... ({len(exp_terms) - 5} more)")
            print()
        
        print(f"\n=== Top matches ({args.mode}) ===\n")
        for i, r in enumerate(rows, 1):
            (cid, collection_slug, document_id, first_page_id, last_page_id, dmin, dmax, preview, score, r_vec, r_lex) = r
            if args.mode == "vector":
                print(f"[{i}] chunk_id={cid}  distance={float(score):.4f}")
            else:
                print(f"[{i}] chunk_id={cid}  score={float(score):.6f}  r_vec={r_vec}  r_lex={r_lex}")
            print(f"    collection_slug={collection_slug}  document_id={document_id}  pages={first_page_id}->{last_page_id}")
            print(f"    date_min={dmin}  date_max={dmax}")
            print(f"    preview: {preview.replace('\\n',' ')}")
            print()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
