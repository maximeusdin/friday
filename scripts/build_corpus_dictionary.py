#!/usr/bin/env python3
"""
Build a materialized corpus dictionary of unique lexemes for per-token fuzzy lexical expansion.

This creates a new build in:
  - corpus_dictionary_builds
  - corpus_dictionary_lexemes

Usage (PowerShell):
  $env:DATABASE_URL="postgresql://neh:neh@localhost:5432/neh"
  python scripts/build_corpus_dictionary.py --chunk-pv chunk_v1_full --norm-version norm_v1
  python scripts/build_corpus_dictionary.py --chunk-pv chunk_v1_silvermaster_structured_4k --collection-slug silvermaster
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root importable when running as script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL")
    return psycopg2.connect(dsn)


def create_build(
    conn,
    *,
    chunk_pv: str,
    collection_slug: Optional[str],
    norm_version: str,
    notes: Optional[str],
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO corpus_dictionary_builds (chunk_pv, collection_slug, norm_version, notes)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (chunk_pv, collection_slug, norm_version, notes),
        )
        return int(cur.fetchone()[0])


def populate_lexemes(
    conn,
    *,
    build_id: int,
    chunk_pv: str,
    collection_slug: Optional[str],
) -> int:
    """
    Populate lexemes for the build by extracting lexemes from tsvectors.
    Stores chunk frequency (#chunks where lexeme appears at least once).
    """
    where = ["c.pipeline_version = %s", "cm.pipeline_version = %s"]
    params = [chunk_pv, chunk_pv]
    if collection_slug:
        where.append("cm.collection_slug = %s")
        params.append(collection_slug)

    where_sql = " AND ".join(where)

    sql = f"""
    WITH lex AS (
      SELECT
        c.id AS chunk_id,
        unnest(
          tsvector_to_array(
            to_tsvector('simple', COALESCE(c.clean_text, c.text))
          )
        ) AS lexeme
      FROM chunks c
      JOIN chunk_metadata cm ON cm.chunk_id = c.id
      WHERE {where_sql}
    ),
    freq AS (
      SELECT lexeme, COUNT(DISTINCT chunk_id)::int AS chunk_freq
      FROM lex
      GROUP BY lexeme
    )
    INSERT INTO corpus_dictionary_lexemes (build_id, lexeme, chunk_freq)
    SELECT %s AS build_id, f.lexeme, f.chunk_freq
    FROM freq f
    ON CONFLICT (build_id, lexeme) DO UPDATE
      SET chunk_freq = EXCLUDED.chunk_freq
    RETURNING 1;
    """

    with conn.cursor() as cur:
        cur.execute(sql, params + [build_id])
        # rowcount is unreliable for INSERT..SELECT with RETURNING; fetchall for count
        rows = cur.fetchall()
        return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Build materialized corpus dictionary (lexeme list)")
    ap.add_argument("--chunk-pv", required=True, help="chunks.pipeline_version / chunk_metadata.pipeline_version")
    ap.add_argument("--collection-slug", default=None, help="Optional scope to a single collection_slug")
    ap.add_argument("--norm-version", default="norm_v1", help="Normalization version label for the build")
    ap.add_argument("--notes", default=None, help="Optional notes stored with the build")
    args = ap.parse_args()

    conn = get_conn()
    try:
        # Ensure tables exist (migration should have run, but fail with clear error if not)
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'corpus_dictionary_builds';")
            if not cur.fetchone():
                raise RuntimeError("Missing corpus_dictionary tables. Run migration 0017 first (make corpus-dictionary).")

        build_id = create_build(
            conn,
            chunk_pv=args.chunk_pv,
            collection_slug=args.collection_slug,
            norm_version=args.norm_version,
            notes=args.notes,
        )
        print(f"Created build_id={build_id} for chunk_pv={args.chunk_pv!r} collection_slug={args.collection_slug!r}")

        n = populate_lexemes(
            conn,
            build_id=build_id,
            chunk_pv=args.chunk_pv,
            collection_slug=args.collection_slug,
        )
        conn.commit()
        print(f"Inserted/updated {n} lexemes into corpus_dictionary_lexemes")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

