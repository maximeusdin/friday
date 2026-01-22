#!/usr/bin/env python3
"""
Rebuild chunk_metadata for a given chunk pipeline, using a two-phase approach:

Phase A (common base rebuild):
- Delete + insert one row per chunk with *source-agnostic* fields only:
  - chunk_id, pipeline_version, derived_at
  - document_id, collection_slug
  - first_page_id, last_page_id
  - meta_raw carries provenance pointers (page_meta_pipeline_versions)
  - meta_raw.embedding is ONLY present when chunks.embedding IS NOT NULL:
      { model, dim, embedded_at }

Phase B (per-source augment):
- For each known collection slug, update chunk_metadata with source-specific fields:
  - venona: date_min/date_max, sender_set/recipient_set, ussr_ref_no_set, plus extras in meta_raw
  - vassiliev: marker sets (and optional label sets) into meta_raw
  - future sources: add another augment function + SQL block

This keeps strategies isolated and makes adding new sources low-risk.

Schema notes:
- Uses your current chunk_metadata columns (date_min/date_max, sender_set, recipient_set, ussr_ref_no_set, ...)
- If chunk_metadata.content_type exists, we populate it. If not, we skip it.
- Works even if page_metadata typed columns are empty (uses meta_raw for venona/vassiliev).

Usage
-----
export DATABASE_URL="postgresql://user:pass@host:5432/neh"

# rebuild everything for chunk pipeline, and augment venona + vassiliev
python scripts/build_chunk_metadata.py \
  --chunk-pipeline chunk_v1_full \
  --venona-page-meta-pipeline meta_v3_venona_headers \
  --vassiliev-page-meta-pipeline meta_v1_pxx_markers

# rebuild only venona
python scripts/build_chunk_metadata.py \
  --chunk-pipeline chunk_v1_full \
  --collection-slug venona \
  --venona-page-meta-pipeline meta_v3_venona_headers
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any

import psycopg2


# ---------------------------
# Safety validations
# ---------------------------

SQL_VALIDATE_NO_CROSS_DOC = """
SELECT cp.chunk_id, COUNT(DISTINCT p.document_id) AS distinct_docs
FROM chunk_pages cp
JOIN pages p  ON p.id = cp.page_id
JOIN chunks c ON c.id = cp.chunk_id
WHERE c.pipeline_version = %(chunk_pv)s
GROUP BY cp.chunk_id
HAVING COUNT(DISTINCT p.document_id) > 1;
"""

SQL_CHECK_CONTENT_TYPE_COL = """
SELECT EXISTS (
  SELECT 1
  FROM information_schema.columns
  WHERE table_schema = 'public'
    AND table_name = 'chunk_metadata'
    AND column_name = 'content_type'
);
"""

SQL_ACCEPT_MISSING = """
SELECT COUNT(*) AS missing
FROM chunks c
LEFT JOIN chunk_metadata cm
  ON cm.chunk_id = c.id
 AND cm.pipeline_version = c.pipeline_version
WHERE c.pipeline_version = %(chunk_pv)s
  AND (%(collection_slug)s IS NULL OR cm.collection_slug = %(collection_slug)s)
  AND cm.chunk_id IS NULL;
"""

SQL_ACCEPT_DUPES = """
SELECT COUNT(*) AS dupes
FROM (
  SELECT chunk_id, pipeline_version, COUNT(*) AS n
  FROM chunk_metadata
  WHERE pipeline_version = %(chunk_pv)s
    AND (%(collection_slug)s IS NULL OR collection_slug = %(collection_slug)s)
  GROUP BY chunk_id, pipeline_version
  HAVING COUNT(*) > 1
) t;
"""


# ---------------------------
# Phase A: base rebuild
# ---------------------------

def sql_base_rebuild(has_content_type: bool) -> str:
    content_type_cols = ""
    content_type_select = ""
    if has_content_type:
        content_type_cols = ",\n  content_type"
        # Simple default mapping; extend as add sources
        content_type_select = (
            ",\n  CASE\n"
            "    WHEN di.collection_slug = 'venona' THEN 'venona_message'\n"
            "    WHEN di.collection_slug = 'vassiliev' THEN 'vassiliev_excerpt'\n"
            "    ELSE NULL\n"
            "  END AS content_type"
        )

    return f"""
BEGIN;

DELETE FROM chunk_metadata
WHERE pipeline_version = %(chunk_pv)s
  AND (%(collection_slug)s IS NULL OR collection_slug = %(collection_slug)s);

WITH chunk_span AS (
  SELECT
    c.id AS chunk_id,
    c.pipeline_version,
    array_agg(cp.page_id ORDER BY cp.span_order) AS page_ids
  FROM chunks c
  JOIN chunk_pages cp ON cp.chunk_id = c.id
  WHERE c.pipeline_version = %(chunk_pv)s
  GROUP BY c.id, c.pipeline_version
),
chunk_doc AS (
  SELECT
    cs.chunk_id,
    cs.pipeline_version,
    cs.page_ids,
    p.document_id
  FROM chunk_span cs
  JOIN pages p ON p.id = cs.page_ids[1]
),
doc_info AS (
  SELECT
    d.id AS document_id,
    col.slug AS collection_slug
  FROM documents d
  JOIN collections col ON col.id = d.collection_id
)
INSERT INTO chunk_metadata (
  chunk_id,
  pipeline_version,
  derived_at,
  document_id,
  collection_slug,
  first_page_id,
  last_page_id,
  meta_raw
  {content_type_cols}
)
SELECT
  cd.chunk_id,
  cd.pipeline_version,
  now() AS derived_at,
  cd.document_id,
  di.collection_slug,
  cd.page_ids[1] AS first_page_id,
  cd.page_ids[array_length(cd.page_ids, 1)] AS last_page_id,

  -- meta_raw:
  --   1) provenance about which page_metadata pipelines were intended
  --   2) embedding provenance copied from chunks ONLY if embedding is present
  jsonb_strip_nulls(
    jsonb_build_object(
      'page_meta_pipeline_versions', jsonb_strip_nulls(jsonb_build_object(
        'venona', %(venona_pm_pv)s,
        'vassiliev', %(vassiliev_pm_pv)s
      )),
      'embedding', CASE
        WHEN c.embedding IS NOT NULL THEN
          jsonb_strip_nulls(jsonb_build_object(
            'model', c.embedding_model,
            'dim', c.embedding_dim,
            'embedded_at', c.embedded_at
          ))
        ELSE NULL
      END
    )
  ) AS meta_raw
  {content_type_select}
FROM chunk_doc cd
JOIN doc_info di ON di.document_id = cd.document_id
JOIN chunks c
  ON c.id = cd.chunk_id
 AND c.pipeline_version = cd.pipeline_version
WHERE (%(collection_slug)s IS NULL OR di.collection_slug = %(collection_slug)s);

COMMIT;
"""


# ---------------------------
# Phase B: source-specific augments
# ---------------------------

SQL_AUGMENT_VENONA = """
-- Populate venona-specific fields by aggregating page_metadata.meta_raw->'venona' over chunk_pages span.
WITH chunk_span AS (
  SELECT
    c.id AS chunk_id,
    array_agg(cp.page_id ORDER BY cp.span_order) AS page_ids
  FROM chunks c
  JOIN chunk_pages cp ON cp.chunk_id = c.id
  WHERE c.pipeline_version = %(chunk_pv)s
  GROUP BY c.id
),
venona_pm AS (
  SELECT
    page_id,
    (meta_raw->'venona'->>'ussr_ref_no') AS ussr_ref_no,
    (meta_raw->'venona'->>'from') AS sender,
    NULLIF(meta_raw->'venona'->>'message_date_iso', '')::date AS message_date,
    COALESCE(
      ARRAY(SELECT jsonb_array_elements_text(meta_raw->'venona'->'to')),
      ARRAY[]::text[]
    ) AS recipients,
    (meta_raw->'venona'->>'cable_no') AS cable_no,
    (meta_raw->'venona'->'cable_nos') AS cable_nos_json,
    COALESCE((meta_raw->'venona'->>'extract_flag')::boolean, false) AS is_extract
  FROM page_metadata
  WHERE pipeline_version = %(venona_pm_pv)s
),
agg AS (
  SELECT
    cs.chunk_id,

    MIN(vpm.message_date) AS date_min,
    MAX(vpm.message_date) AS date_max,

    ARRAY_REMOVE(ARRAY_AGG(DISTINCT vpm.sender), NULL) AS sender_set,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT r.recipient), NULL) AS recipient_set,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT vpm.ussr_ref_no), NULL) AS ussr_ref_no_set,

    COALESCE(bool_or(vpm.is_extract), false) AS extract_any,
    COALESCE(ARRAY_REMOVE(ARRAY_AGG(DISTINCT vpm.cable_no), NULL), ARRAY[]::text[]) AS cable_no_set,
    COALESCE(
      jsonb_agg(DISTINCT vpm.cable_nos_json) FILTER (WHERE vpm.cable_nos_json IS NOT NULL),
      '[]'::jsonb
    ) AS cable_nos_set_json
  FROM chunk_span cs
  LEFT JOIN venona_pm vpm ON vpm.page_id = ANY(cs.page_ids)
  LEFT JOIN LATERAL unnest(vpm.recipients) AS r(recipient) ON TRUE
  GROUP BY cs.chunk_id
)
UPDATE chunk_metadata cm
SET
  date_min = agg.date_min,
  date_max = agg.date_max,
  sender_set = agg.sender_set,
  recipient_set = agg.recipient_set,
  ussr_ref_no_set = agg.ussr_ref_no_set,
  meta_raw = cm.meta_raw || jsonb_build_object(
    'venona_extract_any', agg.extract_any,
    'venona_cable_no_set', to_jsonb(agg.cable_no_set),
    'venona_cable_nos_set', agg.cable_nos_set_json
  )
FROM agg
WHERE cm.pipeline_version = %(chunk_pv)s
  AND cm.chunk_id = agg.chunk_id
  AND cm.collection_slug = 'venona'
  AND (%(collection_slug)s IS NULL OR cm.collection_slug = %(collection_slug)s);
"""

SQL_AUGMENT_VASSILIEV = """
-- Populate vassiliev-specific helpers into meta_raw (markers + base_page_label set)
WITH chunk_span AS (
  SELECT
    c.id AS chunk_id,
    array_agg(cp.page_id ORDER BY cp.span_order) AS page_ids
  FROM chunks c
  JOIN chunk_pages cp ON cp.chunk_id = c.id
  WHERE c.pipeline_version = %(chunk_pv)s
  GROUP BY c.id
),
vpm AS (
  SELECT
    page_id,
    base_page_label,
    meta_raw
  FROM page_metadata
  WHERE pipeline_version = %(vassiliev_pm_pv)s
),
markers AS (
  SELECT
    page_id,
    jsonb_array_elements_text(meta_raw->'p_markers') AS p_marker
  FROM vpm
  WHERE meta_raw ? 'p_markers'
),
agg AS (
  SELECT
    cs.chunk_id,
    COALESCE(to_jsonb(ARRAY_REMOVE(ARRAY_AGG(DISTINCT vpm.base_page_label), NULL)), '[]'::jsonb) AS base_page_label_set,
    COALESCE(to_jsonb(ARRAY_REMOVE(ARRAY_AGG(DISTINCT m.p_marker), NULL)), '[]'::jsonb) AS p_marker_set
  FROM chunk_span cs
  LEFT JOIN vpm ON vpm.page_id = ANY(cs.page_ids)
  LEFT JOIN markers m ON m.page_id = ANY(cs.page_ids)
  GROUP BY cs.chunk_id
)
UPDATE chunk_metadata cm
SET meta_raw = cm.meta_raw || jsonb_build_object(
  'vassiliev_base_page_label_set', agg.base_page_label_set,
  'vassiliev_p_marker_set', agg.p_marker_set
)
FROM agg
WHERE cm.pipeline_version = %(chunk_pv)s
  AND cm.chunk_id = agg.chunk_id
  AND cm.collection_slug = 'vassiliev'
  AND (%(collection_slug)s IS NULL OR cm.collection_slug = %(collection_slug)s);
"""


# ---------------------------
# Utilities
# ---------------------------

@dataclass(frozen=True)
class Args:
    chunk_pv: str
    collection_slug: Optional[str]
    venona_pm_pv: Optional[str]
    vassiliev_pm_pv: Optional[str]
    skip_validate: bool


def require_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return dsn

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "neh")
    user = os.getenv("DB_USER", "neh")
    pw = os.getenv("DB_PASS", "neh")

    return f"host={host} port={port} dbname={name} user={user} password={pw}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild chunk_metadata for a chunk pipeline (base + per-source augments).")
    ap.add_argument("--chunk-pipeline", required=True, help="chunks.pipeline_version to rebuild (e.g. chunk_v1_full)")
    ap.add_argument("--collection-slug", default=None, help="Optional: only rebuild for this collection slug (e.g. venona)")
    ap.add_argument("--venona-page-meta-pipeline", default=None, help="Venona page_metadata pipeline (e.g. meta_v3_venona_headers)")
    ap.add_argument("--vassiliev-page-meta-pipeline", default=None, help="Vassiliev page_metadata pipeline (e.g. meta_v1_pxx_markers)")
    ap.add_argument("--skip-validate", action="store_true", help="Skip cross-document chunk validation (not recommended).")
    ns = ap.parse_args()

    args = Args(
        chunk_pv=ns.chunk_pipeline,
        collection_slug=ns.collection_slug,
        venona_pm_pv=ns.venona_page_meta_pipeline,
        vassiliev_pm_pv=ns.vassiliev_page_meta_pipeline,
        skip_validate=bool(ns.skip_validate),
    )

    # If a pm pipeline isn't provided, use "__NONE__" so joins yield empty sets safely.
    params: Dict[str, Any] = {
        "chunk_pv": args.chunk_pv,
        "collection_slug": args.collection_slug,
        "venona_pm_pv": args.venona_pm_pv or "__NONE__",
        "vassiliev_pm_pv": args.vassiliev_pm_pv or "__NONE__",
    }

    dsn = require_dsn()

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            if not args.skip_validate:
                cur.execute(SQL_VALIDATE_NO_CROSS_DOC, {"chunk_pv": args.chunk_pv})
                bad = cur.fetchall()
                if bad:
                    print("ERROR: Found chunks spanning multiple documents. Refusing to rebuild.", file=sys.stderr)
                    for chunk_id, distinct_docs in bad[:50]:
                        print(f"  chunk_id={chunk_id} distinct_docs={distinct_docs}", file=sys.stderr)
                    return 2

            cur.execute(SQL_CHECK_CONTENT_TYPE_COL)
            has_content_type = bool(cur.fetchone()[0])

            # Phase A: base rebuild
            cur.execute(sql_base_rebuild(has_content_type), params)

            # Phase B: per-source augments (only run if the corresponding page-meta pipeline was provided)
            if args.venona_pm_pv:
                cur.execute(SQL_AUGMENT_VENONA, params)

            if args.vassiliev_pm_pv:
                cur.execute(SQL_AUGMENT_VASSILIEV, params)

            # Acceptance: missing rows
            cur.execute(SQL_ACCEPT_MISSING, {"chunk_pv": args.chunk_pv, "collection_slug": args.collection_slug})
            missing = int(cur.fetchone()[0])

            # Acceptance: dupes
            cur.execute(SQL_ACCEPT_DUPES, {"chunk_pv": args.chunk_pv, "collection_slug": args.collection_slug})
            dupes = int(cur.fetchone()[0])

    scope = f"collection_slug={args.collection_slug}" if args.collection_slug else "collection_slug=ALL"
    if missing != 0 or dupes != 0:
        print(
            f"WARNING: rebuild completed but missing={missing} dupes={dupes} for chunk_pv={args.chunk_pv} ({scope})",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: chunk_metadata rebuilt for chunk_pv={args.chunk_pv} ({scope}) "
        f"venona_pm_pv={params['venona_pm_pv']} vassiliev_pm_pv={params['vassiliev_pm_pv']}"
    )
    if not has_content_type:
        print("NOTE: chunk_metadata.content_type column not found; skipping content_type population.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
