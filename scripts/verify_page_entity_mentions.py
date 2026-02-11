#!/usr/bin/env python3
"""
Verify page_entity_mentions — post-table check that surfaces are found
in chunks where we expect them.

Catches mapping bugs (wrong document/page) or missing chunk coverage.

Contract (A5b):
  For a sample of rows (and at least for high-value cases like CABIN),
  resolve page_id → chunk_ids via chunk_pages.  Then verify that at least
  one of those chunks contains evidence of the (surface, entity) pair:
  an entity_mentions row for that chunk_id and entity_id, or the surface
  string appearing in the chunk text.

Usage:
    python scripts/verify_page_entity_mentions.py
    python scripts/verify_page_entity_mentions.py --surface cabin
    python scripts/verify_page_entity_mentions.py --sample-size 500
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn
from retrieval.agent.v10_page_bridge import pages_to_chunks_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_pem")

# Minimum fraction of rows with chunk evidence to pass
MIN_EVIDENCE_FRACTION = 0.70

# High-value surfaces to always check
HIGH_VALUE_SURFACES = ["cabin", "pal", "liberal", "antenna", "king"]


def verify_surface(
    conn,
    surface_norm: str,
    entity_id: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Verify a specific surface in page_entity_mentions.

    Returns dict with: total_rows, rows_with_chunks, rows_with_evidence,
    evidence_fraction, failures (list of dicts with details).
    """
    result = {
        "surface_norm": surface_norm,
        "entity_id": entity_id,
        "total_rows": 0,
        "rows_with_chunks": 0,
        "rows_with_evidence": 0,
        "evidence_fraction": 0.0,
        "failures": [],
    }

    try:
        with conn.cursor() as cur:
            # Get PEM rows for this surface
            conditions = ["pem.surface_norm = %s"]
            params: list = [surface_norm]
            if entity_id is not None:
                conditions.append("pem.entity_id = %s")
                params.append(entity_id)
            where = " AND ".join(conditions)

            cur.execute(f"""
                SELECT pem.id, pem.page_id, pem.entity_id, pem.document_id,
                       pem.collection_slug, pem.surface_raw
                FROM page_entity_mentions pem
                WHERE {where}
                ORDER BY pem.document_id, pem.page_id
                LIMIT 1000
            """, params)
            rows = cur.fetchall()
            result["total_rows"] = len(rows)

            if not rows:
                return result

            # Get all page_ids and resolve to chunks
            page_ids = [r[1] for r in rows]
            page_chunk_map = pages_to_chunks_map(conn, page_ids)

            for pem_id, page_id, eid, doc_id, coll, surface_raw in rows:
                chunk_ids = page_chunk_map.get(page_id, [])
                if not chunk_ids:
                    result["failures"].append({
                        "pem_id": pem_id,
                        "page_id": page_id,
                        "document_id": doc_id,
                        "reason": "no_chunks_for_page",
                    })
                    continue

                result["rows_with_chunks"] += 1

                # Check if any chunk has entity_mentions for this entity
                cur.execute("""
                    SELECT 1 FROM entity_mentions em
                    WHERE em.chunk_id = ANY(%s)
                      AND em.entity_id = %s
                    LIMIT 1
                """, (chunk_ids, eid))
                has_em = cur.fetchone() is not None

                if has_em:
                    result["rows_with_evidence"] += 1
                    continue

                # Fallback: check if surface appears in chunk text
                surface_check = (surface_raw or surface_norm).lower()
                cur.execute("""
                    SELECT 1 FROM chunks c
                    WHERE c.id = ANY(%s)
                      AND LOWER(COALESCE(c.clean_text, c.text)) LIKE %s
                    LIMIT 1
                """, (chunk_ids, f"%{surface_check}%"))
                has_text = cur.fetchone() is not None

                if has_text:
                    result["rows_with_evidence"] += 1
                else:
                    result["failures"].append({
                        "pem_id": pem_id,
                        "page_id": page_id,
                        "document_id": doc_id,
                        "chunk_ids": chunk_ids[:3],
                        "reason": "no_evidence_in_chunks",
                    })

    except Exception as e:
        logger.error("verify_surface failed for %s: %s", surface_norm, e)
        try:
            conn.rollback()
        except Exception:
            pass

    if result["total_rows"] > 0:
        result["evidence_fraction"] = round(
            result["rows_with_evidence"] / result["total_rows"], 4
        )

    return result


def verify_random_sample(
    conn,
    sample_size: int = 200,
    verbose: bool = False,
) -> Dict:
    """Verify a random sample of page_entity_mentions rows.

    Returns aggregate metrics.
    """
    result = {
        "sample_size": 0,
        "rows_with_chunks": 0,
        "rows_with_evidence": 0,
        "evidence_fraction": 0.0,
        "failures_by_surface": Counter(),
    }

    try:
        with conn.cursor() as cur:
            # Get random sample using stable hash ordering
            cur.execute("""
                SELECT pem.id, pem.page_id, pem.entity_id, pem.document_id,
                       pem.collection_slug, pem.surface_norm, pem.surface_raw
                FROM page_entity_mentions pem
                ORDER BY hashtext(pem.id::text)
                LIMIT %s
            """, (sample_size,))
            rows = cur.fetchall()
            result["sample_size"] = len(rows)

            if not rows:
                return result

            page_ids = [r[1] for r in rows]
            page_chunk_map = pages_to_chunks_map(conn, page_ids)

            for pem_id, page_id, eid, doc_id, coll, surface_norm, surface_raw in rows:
                chunk_ids = page_chunk_map.get(page_id, [])
                if not chunk_ids:
                    result["failures_by_surface"][surface_norm] += 1
                    continue

                result["rows_with_chunks"] += 1

                cur.execute("""
                    SELECT 1 FROM entity_mentions em
                    WHERE em.chunk_id = ANY(%s)
                      AND em.entity_id = %s
                    LIMIT 1
                """, (chunk_ids, eid))
                has_em = cur.fetchone() is not None

                if has_em:
                    result["rows_with_evidence"] += 1
                    continue

                surface_check = (surface_raw or surface_norm).lower()
                cur.execute("""
                    SELECT 1 FROM chunks c
                    WHERE c.id = ANY(%s)
                      AND LOWER(COALESCE(c.clean_text, c.text)) LIKE %s
                    LIMIT 1
                """, (chunk_ids, f"%{surface_check}%"))
                has_text = cur.fetchone() is not None

                if has_text:
                    result["rows_with_evidence"] += 1
                else:
                    result["failures_by_surface"][surface_norm] += 1

    except Exception as e:
        logger.error("verify_random_sample failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    if result["sample_size"] > 0:
        result["evidence_fraction"] = round(
            result["rows_with_evidence"] / result["sample_size"], 4
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify page_entity_mentions")
    parser.add_argument("--surface", type=str, default=None,
                        help="Verify a specific surface_norm (e.g. 'cabin')")
    parser.add_argument("--entity-id", type=int, default=None,
                        help="Filter to specific entity_id")
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Random sample size for general verification")
    parser.add_argument("--skip-sample", action="store_true",
                        help="Skip random sample, only check high-value surfaces")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    conn = get_conn()
    all_ok = True

    # Check specific surface if provided
    if args.surface:
        logger.info("Verifying surface: %s", args.surface)
        result = verify_surface(conn, args.surface, args.entity_id, args.verbose)
        _log_surface_result(result)
        if result["evidence_fraction"] < MIN_EVIDENCE_FRACTION and result["total_rows"] > 0:
            all_ok = False

    # Always check high-value surfaces
    logger.info("=== High-value surface checks ===")
    for surface in HIGH_VALUE_SURFACES:
        if args.surface and surface == args.surface:
            continue  # already checked
        result = verify_surface(conn, surface, verbose=args.verbose)
        _log_surface_result(result)
        if result["evidence_fraction"] < MIN_EVIDENCE_FRACTION and result["total_rows"] > 0:
            all_ok = False

    # Random sample
    if not args.skip_sample:
        logger.info("=== Random sample verification ===")
        sample_result = verify_random_sample(conn, args.sample_size, args.verbose)
        logger.info("Sample size: %d", sample_result["sample_size"])
        logger.info("Rows with chunks: %d", sample_result["rows_with_chunks"])
        logger.info("Rows with evidence: %d", sample_result["rows_with_evidence"])
        logger.info("Evidence fraction: %.1f%%", sample_result["evidence_fraction"] * 100)

        if sample_result["failures_by_surface"]:
            top_failures = sample_result["failures_by_surface"].most_common(10)
            logger.info("Top failing surfaces: %s", top_failures)

        if sample_result["evidence_fraction"] < MIN_EVIDENCE_FRACTION and sample_result["sample_size"] > 0:
            all_ok = False

    if all_ok:
        logger.info("=== Verification: PASS ===")
    else:
        logger.warning("=== Verification: WARN (evidence fraction below threshold) ===")

    conn.close()
    sys.exit(0 if all_ok else 1)


def _log_surface_result(result: Dict) -> None:
    """Log a surface verification result."""
    surface = result["surface_norm"]
    total = result["total_rows"]
    evidence = result["rows_with_evidence"]
    fraction = result["evidence_fraction"]

    if total == 0:
        logger.info("  %s: no rows in page_entity_mentions", surface)
    else:
        status = "OK" if fraction >= MIN_EVIDENCE_FRACTION else "WARN"
        logger.info("  %s: %d/%d with evidence (%.1f%%) [%s]",
                    surface, evidence, total, fraction * 100, status)
        if result["failures"] and len(result["failures"]) <= 5:
            for f in result["failures"]:
                logger.info("    FAIL: page_id=%s, doc_id=%s, reason=%s",
                           f.get("page_id"), f.get("document_id"), f.get("reason"))


if __name__ == "__main__":
    main()
