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

OCR-variant channel (Phase 2):
  - After populating lexemes, skeletons (OCR-confusion-class collapses, see
    retrieval/ocr_variants.py) are computed python-side and stored in
    corpus_dictionary_lexemes.skeleton (migration 0075).
  - --include-raw-transcript additionally aggregates lexemes from the RAW OCR
    text (c.text) of chunks whose document carries metadata ? 'transcript',
    merging into the same build (chunk_freq summed on conflict). This keeps
    raw-OCR garble tokens (e.g. "fuer" for FUHR) findable as variant targets
    after a corrected transcript replaces clean_text.
  - --backfill-skeletons --build-id N only (re)computes skeletons for an
    existing build (no new build, no lexeme changes).
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

SKELETON_BATCH_SIZE = 1000

SKELETON_MIGRATION = "migrations/0075_ocr_variant_skeleton.sql"

# Mirrors _TRANSCRIPT_DOC_EXISTS in retrieval/ops.py (chunk_pages -> pages -> documents).
TRANSCRIPT_DOC_EXISTS = """EXISTS (
        SELECT 1
        FROM chunk_pages cp_tr
        JOIN pages p_tr ON p_tr.id = cp_tr.page_id
        JOIN documents d_tr ON d_tr.id = p_tr.document_id
        WHERE cp_tr.chunk_id = c.id
          AND d_tr.metadata ? 'transcript'
      )"""


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


def populate_raw_transcript_lexemes(
    conn,
    *,
    build_id: int,
    chunk_pv: str,
    collection_slug: Optional[str],
) -> int:
    """
    Merge lexemes from the RAW OCR text (c.text, NOT clean_text) of chunks whose
    document carries metadata ? 'transcript' into an existing build.

    Transcript-bearing documents keep raw OCR in chunks.text while clean_text
    holds the corrected transcript; the main populate pass reads
    COALESCE(clean_text, text) and therefore loses the raw garble tokens
    (e.g. "fuer" for FUHR). Those tokens must stay in the dictionary as
    OCR-variant targets, so they are aggregated here and merged with
    chunk_freq summed on conflict.
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
            to_tsvector('simple', c.text)
          )
        ) AS lexeme
      FROM chunks c
      JOIN chunk_metadata cm ON cm.chunk_id = c.id
      WHERE {where_sql}
        AND {TRANSCRIPT_DOC_EXISTS}
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
      SET chunk_freq = corpus_dictionary_lexemes.chunk_freq + EXCLUDED.chunk_freq
    RETURNING 1;
    """

    with conn.cursor() as cur:
        cur.execute(sql, params + [build_id])
        rows = cur.fetchall()
        return len(rows)


def _skeleton_column_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'corpus_dictionary_lexemes' AND column_name = 'skeleton';
            """
        )
        return cur.fetchone() is not None


def compute_skeletons(
    conn,
    *,
    build_id: int,
    batch_size: int = SKELETON_BATCH_SIZE,
) -> int:
    """
    (Re)compute corpus_dictionary_lexemes.skeleton python-side for one build,
    in batches. Returns the number of lexemes updated.

    Degrades gracefully: if the skeleton column (migration 0075) or the
    retrieval.ocr_variants module is missing, prints a warning and returns 0
    instead of failing the build.
    """
    if not _skeleton_column_exists(conn):
        print(
            "WARNING: corpus_dictionary_lexemes.skeleton column is missing; "
            f"run {SKELETON_MIGRATION} first. Skipping skeleton computation."
        )
        return 0

    # Lazy import: keep the base dictionary build working even if the
    # OCR-variant module is absent in this checkout.
    try:
        from retrieval.ocr_variants import load_confusion, skeleton
    except ImportError as exc:
        print(
            "WARNING: retrieval.ocr_variants not importable "
            f"({exc}); skipping skeleton computation."
        )
        return 0

    conf = load_confusion()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT lexeme FROM corpus_dictionary_lexemes
            WHERE build_id = %s
            ORDER BY lexeme;
            """,
            (build_id,),
        )
        lexemes = [row[0] for row in cur.fetchall()]

    updated = 0
    for start in range(0, len(lexemes), batch_size):
        batch = lexemes[start:start + batch_size]
        pairs = [(lx, skeleton(lx, conf)) for lx in batch]
        values_sql = ", ".join(["(%s::text, %s::text)"] * len(pairs))
        sql = f"""
        UPDATE corpus_dictionary_lexemes AS l
        SET skeleton = v.skeleton
        FROM (VALUES {values_sql}) AS v(lexeme, skeleton)
        WHERE l.build_id = %s AND l.lexeme = v.lexeme;
        """
        params = [x for pair in pairs for x in pair] + [build_id]
        with conn.cursor() as cur:
            cur.execute(sql, params)
        updated += len(pairs)

    return updated


def main():
    ap = argparse.ArgumentParser(description="Build materialized corpus dictionary (lexeme list)")
    ap.add_argument("--chunk-pv", default=None, help="chunks.pipeline_version / chunk_metadata.pipeline_version")
    ap.add_argument("--collection-slug", default=None, help="Optional scope to a single collection_slug")
    ap.add_argument("--norm-version", default="norm_v1", help="Normalization version label for the build")
    ap.add_argument("--notes", default=None, help="Optional notes stored with the build")
    ap.add_argument(
        "--include-raw-transcript",
        action="store_true",
        help="Also aggregate lexemes from raw OCR text (c.text) of transcript-bearing documents "
             "(documents.metadata ? 'transcript'), merged into the same build with chunk_freq summed",
    )
    ap.add_argument(
        "--backfill-skeletons",
        action="store_true",
        help="Maintenance mode: only (re)compute skeletons for an existing build (requires --build-id)",
    )
    ap.add_argument("--build-id", type=int, default=None, help="Existing build id for --backfill-skeletons")
    args = ap.parse_args()

    if args.backfill_skeletons:
        if args.build_id is None:
            ap.error("--backfill-skeletons requires --build-id")
        conn = get_conn()
        try:
            n = compute_skeletons(conn, build_id=args.build_id)
            conn.commit()
            print(f"Backfilled skeletons for {n} lexemes in build_id={args.build_id}")
        finally:
            conn.close()
        return

    if not args.chunk_pv:
        ap.error("--chunk-pv is required (unless using --backfill-skeletons)")

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

        if args.include_raw_transcript:
            n_raw = populate_raw_transcript_lexemes(
                conn,
                build_id=build_id,
                chunk_pv=args.chunk_pv,
                collection_slug=args.collection_slug,
            )
            conn.commit()
            print(f"Merged {n_raw} raw-transcript lexemes (chunk_freq summed on conflict)")

        n_skel = compute_skeletons(conn, build_id=build_id)
        conn.commit()
        print(f"Computed skeletons for {n_skel} lexemes")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

