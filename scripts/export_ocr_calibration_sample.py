#!/usr/bin/env python3
"""
Export a stratified sample of OCR resolution outcomes for threshold calibration.

Goal: make it easy to spot-check a small, high-signal slice and decide whether to
increase/decrease thresholds/weights.

Outputs a CSV that you can edit in Excel/Sheets:
- Fill `review_judgement` (CORRECT / INCORRECT / JUNK / MISS / UNSURE)
- Optionally fill `review_correct_entity_id`
- Add notes

Then run scripts/analyze_ocr_calibration_sample.py on the reviewed CSV.

Sampling strategy:
- resolved_near_threshold: score in [resolved_min, resolved_max]
- queued_mid:             score in [queue_min, queue_max]
- ignored_borderline:     score in [ignore_min, ignore_max]

This script avoids expensive ORDER BY random() scans by using TABLESAMPLE on
mention_candidates and then fetching full details for a small candidate_id set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


OCR_COLLECTIONS = ["silvermaster", "rosenberg", "solo", "fbicomrap"]


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL environment variable")
    return psycopg2.connect(dsn)


def _parse_reviewed_candidate_ids_from_csv(path: Path) -> Set[int]:
    """
    Accepts either the raw exported sample CSV or a reviewed copy.
    Looks for candidate_id column.
    """
    if not path.exists():
        return set()
    ids: Set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        # Skip comment/instruction lines
        peek = f.read(4096)
        f.seek(0)
        if peek.startswith("#"):
            # Consume leading comment lines
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    return set()
                if not line.startswith("#"):
                    f.seek(pos)
                    break
        r = csv.DictReader(f)
        for row in r:
            v = (row.get("candidate_id") or "").strip()
            if not v:
                continue
            try:
                ids.add(int(v))
            except ValueError:
                continue
    return ids


def load_excluded_candidate_ids(paths: Sequence[str]) -> Set[int]:
    excluded: Set[int] = set()
    for p in paths:
        excluded |= _parse_reviewed_candidate_ids_from_csv(Path(p))
    return excluded


def _compute_excerpt(text: str, start: int, end: int, window: int = 160) -> str:
    """
    Char offsets are treated as 0-based.
    """
    if text is None:
        return ""
    n = len(text)
    s = max(0, start - window)
    e = min(n, end + window)
    excerpt = text[s:e]
    # Make it single-line for CSV ergonomics
    excerpt = excerpt.replace("\r\n", "\n").replace("\r", "\n")
    excerpt = " ".join(excerpt.split())
    return excerpt


def sample_candidate_ids(
    conn,
    *,
    where_sql: str,
    params: Sequence,
    n: int,
    excluded_ids: Set[int],
    tablesample_pcts: Sequence[float],
) -> List[int]:
    """
    Returns up to n candidate ids satisfying where_sql, excluding excluded_ids.
    """
    if n <= 0:
        return []

    collected: List[int] = []
    seen: Set[int] = set()

    with conn.cursor() as cur:
        for pct in tablesample_pcts:
            if len(collected) >= n:
                break

            # Pull an oversample (cheap) from a random page sample of the table.
            # Note: TABLESAMPLE pct is approximate and may return 0 rows for very small pct.
            cur.execute(
                f"""
                SELECT mc.id
                FROM (
                    SELECT
                      id,
                      document_id,
                      resolution_score,
                      resolution_status
                    FROM mention_candidates TABLESAMPLE SYSTEM ({pct})
                ) mc
                JOIN documents d ON d.id = mc.document_id
                JOIN collections col ON col.id = d.collection_id
                WHERE {where_sql}
                LIMIT %s
                """,
                [*params, max(n * 20, 500)],
            )
            for (cid,) in cur.fetchall():
                if cid in excluded_ids or cid in seen:
                    continue
                seen.add(cid)
                collected.append(int(cid))
                if len(collected) >= n:
                    break

    return collected[:n]


def fetch_candidate_details(conn, candidate_ids: Sequence[int]) -> List[dict]:
    if not candidate_ids:
        return []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
              mc.id AS candidate_id,
              mc.document_id,
              mc.chunk_id,
              mc.char_start,
              mc.char_end,
              mc.raw_span,
              mc.surface_norm,
              mc.doc_quality,
              mc.quality_score,
              mc.resolution_status,
              mc.resolution_method,
              mc.resolution_score,
              mc.resolution_margin,
              mc.top_candidates,
              d.source_name AS document_name,
              col.slug AS collection_slug,
              col.title AS collection_title,
              c.text AS chunk_text
            FROM mention_candidates mc
            JOIN documents d ON d.id = mc.document_id
            JOIN collections col ON col.id = d.collection_id
            JOIN chunks c ON c.id = mc.chunk_id
            WHERE mc.id = ANY(%s)
            """,
            (list(candidate_ids),),
        )
        return list(cur.fetchall())


def fetch_entity_names(conn, entity_ids: Iterable[int]) -> Dict[int, str]:
    ids = sorted({int(x) for x in entity_ids if x is not None})
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT id, canonical_name FROM entities WHERE id = ANY(%s)", (ids,))
        return {int(r[0]): r[1] for r in cur.fetchall()}


def _extract_top_candidates(top_candidates_json) -> List[dict]:
    if top_candidates_json is None:
        return []
    if isinstance(top_candidates_json, list):
        return top_candidates_json
    # psycopg2 may return JSONB as str in some configs
    if isinstance(top_candidates_json, str):
        try:
            v = json.loads(top_candidates_json)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def build_rows(
    *,
    group_name: str,
    details: List[dict],
    entity_name_by_id: Dict[int, str],
    top_k: int,
) -> List[dict]:
    rows: List[dict] = []
    for d in details:
        top = _extract_top_candidates(d.get("top_candidates"))[:top_k]
        top_entity_ids = [int(x.get("entity_id")) for x in top if x.get("entity_id") is not None]
        # Ensure names exist (should, but be defensive)
        _ = [entity_name_by_id.get(eid, "") for eid in top_entity_ids]

        excerpt = _compute_excerpt(
            d.get("chunk_text") or "",
            int(d.get("char_start") or 0),
            int(d.get("char_end") or 0),
        )

        row: Dict[str, object] = {
            "sample_group": group_name,
            "candidate_id": d.get("candidate_id"),
            "collection_slug": d.get("collection_slug"),
            "document_id": d.get("document_id"),
            "document_name": d.get("document_name"),
            "chunk_id": d.get("chunk_id"),
            "char_start": d.get("char_start"),
            "char_end": d.get("char_end"),
            "raw_span": d.get("raw_span"),
            "surface_norm": d.get("surface_norm"),
            "doc_quality": d.get("doc_quality"),
            "quality_score": float(d["quality_score"]) if d.get("quality_score") is not None else None,
            "resolution_status": d.get("resolution_status"),
            "resolution_method": d.get("resolution_method"),
            "resolution_score": float(d["resolution_score"]) if d.get("resolution_score") is not None else None,
            "resolution_margin": float(d["resolution_margin"]) if d.get("resolution_margin") is not None else None,
            "context_excerpt": excerpt,
        }

        # Expand top candidates into separate columns
        for i in range(1, top_k + 1):
            cand = top[i - 1] if i - 1 < len(top) else {}
            eid = cand.get("entity_id")
            row[f"cand_{i}_entity_id"] = eid
            row[f"cand_{i}_entity_name"] = entity_name_by_id.get(int(eid), "") if eid is not None else ""
            row[f"cand_{i}_alias_norm"] = cand.get("alias_norm", "")
            row[f"cand_{i}_score"] = cand.get("score")
            row[f"cand_{i}_trgm"] = cand.get("trgm")
            row[f"cand_{i}_tok"] = cand.get("tok")
            row[f"cand_{i}_edit"] = cand.get("edit")
            row[f"cand_{i}_proposal_tier"] = cand.get("proposal_tier")

        # Reviewer inputs (blank)
        row["review_judgement"] = ""  # CORRECT / INCORRECT / JUNK / MISS / UNSURE
        row["review_correct_entity_id"] = ""
        row["review_notes"] = ""

        rows.append(row)

    return rows


def write_csv(out_path: Path, rows: List[dict], meta_lines: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        for line in meta_lines:
            f.write(f"# {line}\n")
        if not rows:
            # Still write a header for ergonomics
            f.write("# No rows matched your filters.\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export stratified OCR calibration sample CSV")
    ap.add_argument("--out", default="ocr_calibration_sample.csv", help="Output CSV path")
    ap.add_argument("--exclude-reviewed-csv", action="append", default=[],
                    help="CSV file(s) from previous calibration runs; candidate_ids will be excluded")

    # Scope
    ap.add_argument("--collection", type=str, default=None, help="Limit to a single collection slug")
    ap.add_argument("--all-ocr", action="store_true", help="Limit to OCR collections (default if no --collection)")
    ap.add_argument("--all-corpus", action="store_true", help="Allow sampling across all collections")

    # Sizes
    ap.add_argument("--n-resolved", type=int, default=100, help="Rows in resolved_near_threshold band")
    ap.add_argument("--n-queued", type=int, default=100, help="Rows in queued_mid band")
    ap.add_argument("--n-ignored", type=int, default=50, help="Rows in ignored_borderline band")
    ap.add_argument("--top-k", type=int, default=5, help="Top candidate entities to expand into columns")

    # Bands (defaults aligned with resolver v2 constants)
    ap.add_argument("--resolved-min", type=float, default=0.72)
    ap.add_argument("--resolved-max", type=float, default=0.80)
    ap.add_argument("--queue-min", type=float, default=0.45)
    ap.add_argument("--queue-max", type=float, default=0.60)
    ap.add_argument("--ignore-min", type=float, default=0.35)
    ap.add_argument("--ignore-max", type=float, default=0.50)

    args = ap.parse_args()

    excluded_ids = load_excluded_candidate_ids([str(p) for p in args.exclude_reviewed_csv])

    conn = get_conn()
    try:
        # Scope filter (shared)
        if args.collection:
            scope_sql = "mc.resolution_score IS NOT NULL AND col.slug = %s"
            scope_params: List[object] = [args.collection]
            scope_note = f"collection={args.collection}"
        else:
            # Default to OCR collections unless user explicitly wants full corpus.
            if args.all_corpus:
                scope_sql = "mc.resolution_score IS NOT NULL"
                scope_params = []
                scope_note = "collections=ALL"
            else:
                scope_sql = "mc.resolution_score IS NOT NULL AND col.slug = ANY(%s)"
                scope_params = [OCR_COLLECTIONS]
                scope_note = f"collections={','.join(OCR_COLLECTIONS)}"

        tablesample_pcts = [0.1, 0.5, 1, 2, 5, 10, 20]

        # Sample ids per stratum
        resolved_ids = sample_candidate_ids(
            conn,
            where_sql=f"""
              {scope_sql}
              AND mc.resolution_status = 'resolved'
              AND mc.resolution_score >= %s AND mc.resolution_score <= %s
            """,
            params=[*scope_params, args.resolved_min, args.resolved_max],
            n=args.n_resolved,
            excluded_ids=excluded_ids,
            tablesample_pcts=tablesample_pcts,
        )
        excluded_ids |= set(resolved_ids)

        queued_ids = sample_candidate_ids(
            conn,
            where_sql=f"""
              {scope_sql}
              AND mc.resolution_status = 'queue'
              AND mc.resolution_score >= %s AND mc.resolution_score <= %s
            """,
            params=[*scope_params, args.queue_min, args.queue_max],
            n=args.n_queued,
            excluded_ids=excluded_ids,
            tablesample_pcts=tablesample_pcts,
        )
        excluded_ids |= set(queued_ids)

        ignored_ids = sample_candidate_ids(
            conn,
            where_sql=f"""
              {scope_sql}
              AND mc.resolution_status = 'ignore'
              AND mc.resolution_score >= %s AND mc.resolution_score <= %s
            """,
            params=[*scope_params, args.ignore_min, args.ignore_max],
            n=args.n_ignored,
            excluded_ids=excluded_ids,
            tablesample_pcts=tablesample_pcts,
        )

        all_ids = resolved_ids + queued_ids + ignored_ids
        details = fetch_candidate_details(conn, all_ids)
        by_id = {int(d["candidate_id"]): d for d in details}

        # Collect entity ids used in any top_candidates across selected rows
        entity_ids: Set[int] = set()
        for cid in all_ids:
            d = by_id.get(int(cid))
            if not d:
                continue
            top = _extract_top_candidates(d.get("top_candidates"))[: args.top_k]
            for t in top:
                eid = t.get("entity_id")
                if eid is not None:
                    try:
                        entity_ids.add(int(eid))
                    except Exception:
                        pass
        entity_name_by_id = fetch_entity_names(conn, entity_ids)

        # Build rows preserving requested ordering by strata
        rows: List[dict] = []
        rows += build_rows(
            group_name="resolved_near_threshold",
            details=[by_id[i] for i in resolved_ids if int(i) in by_id],
            entity_name_by_id=entity_name_by_id,
            top_k=args.top_k,
        )
        rows += build_rows(
            group_name="queued_mid",
            details=[by_id[i] for i in queued_ids if int(i) in by_id],
            entity_name_by_id=entity_name_by_id,
            top_k=args.top_k,
        )
        rows += build_rows(
            group_name="ignored_borderline",
            details=[by_id[i] for i in ignored_ids if int(i) in by_id],
            entity_name_by_id=entity_name_by_id,
            top_k=args.top_k,
        )

        meta = [
            "OCR calibration sample export",
            f"exported_at={datetime.utcnow().isoformat()}Z",
            f"scope={scope_note}",
            f"excluded_candidate_ids={len(load_excluded_candidate_ids([str(p) for p in args.exclude_reviewed_csv]))}",
            "Fill review_judgement with one of: CORRECT, INCORRECT, JUNK, MISS, UNSURE",
            "Optional: review_correct_entity_id (if INCORRECT but you know the right entity)",
        ]

        out_path = Path(args.out)
        write_csv(out_path, rows, meta_lines=meta)

        print(f"Wrote {len(rows)} rows to {out_path}")
        print(f"  resolved_near_threshold: {len(resolved_ids)}")
        print(f"  queued_mid:             {len(queued_ids)}")
        print(f"  ignored_borderline:     {len(ignored_ids)}")
        if args.exclude_reviewed_csv:
            print(f"Excluded from prior CSVs: {len(load_excluded_candidate_ids([str(p) for p in args.exclude_reviewed_csv]))}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

