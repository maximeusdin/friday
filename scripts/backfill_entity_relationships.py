#!/usr/bin/env python3
"""
backfill_entity_relationships.py [options]

Backfill covername→person relationships from alias patterns, without N+1 queries.

Converts this bad pattern:
  covername entity (e.g., DOUGLAS) has a person-name alias "Joseph Katz"
into:
  DOUGLAS --covername_of--> Joseph Katz   (stored in entity_relationships)
and (optionally) removes the person-name alias from the covername entity
so alias matching stops colliding.

Key improvement:
- Uses ONE set-based join to find person candidates for ALL patterns at once.

Usage:
  python scripts/backfill_entity_relationships.py --dry-run
  python scripts/backfill_entity_relationships.py
  python scripts/backfill_entity_relationships.py --no-delete-aliases
  python scripts/backfill_entity_relationships.py --limit 2000 --dry-run
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from psycopg2.extras import execute_values

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.ops import get_conn


def require_table(cur, table: str) -> None:
    cur.execute(
        """
        SELECT EXISTS (
          SELECT 1 FROM information_schema.tables
          WHERE table_schema = 'public' AND table_name = %s
        )
        """,
        (table,),
    )
    ok = cur.fetchone()[0]
    if not ok:
        raise RuntimeError(f"Required table missing: {table}")


def discover_covername_types(cur) -> List[str]:
    """
    Find entity_type values that look like covernames.
    """
    cur.execute(
        """
        SELECT DISTINCT entity_type
        FROM entities
        WHERE entity_type ILIKE '%cover%'
           OR entity_type ILIKE '%code%'
           OR entity_type ILIKE '%crypt%'
        ORDER BY entity_type
        """
    )
    return [r[0] for r in cur.fetchall()]


def identify_patterns(
    cur,
    cover_types: List[str],
    *,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Find aliases attached to covername entities that look person-like.

    Returns dicts:
      alias_id, covername_entity_id, covername_name, covername_type,
      person_alias, person_alias_norm
    """
    if not cover_types:
        return []

    limit_sql = "LIMIT %s" if limit else ""
    params: List[object] = [cover_types]  # IMPORTANT: list -> adapted to text[]
    if limit:
        params.append(limit)

    # Heuristic: person-like aliases = NOT (ALLCAPS short) and has space OR length > 6
    cur.execute(
        f"""
        SELECT
          ea.id AS alias_id,
          e_cov.id AS covername_entity_id,
          e_cov.canonical_name AS covername_name,
          e_cov.entity_type AS covername_type,
          ea.alias AS person_alias,
          ea.alias_norm AS person_alias_norm
        FROM entities e_cov
        JOIN entity_aliases ea ON ea.entity_id = e_cov.id
        WHERE e_cov.entity_type = ANY(%s::text[])
          AND COALESCE(ea.is_matchable, true) = true
          -- exclude covername-like aliases: ALLCAPS and short
          AND NOT (
            ea.alias = UPPER(ea.alias)
            AND LENGTH(TRIM(ea.alias)) <= 6
          )
          -- person-like:
          AND (
            POSITION(' ' IN TRIM(ea.alias)) > 0
            OR LENGTH(TRIM(ea.alias)) > 6
          )
        ORDER BY e_cov.id, ea.alias_norm, ea.id
        {limit_sql}
        """,
        params,
    )

    out: List[Dict] = []
    for row in cur.fetchall():
        out.append(
            {
                "alias_id": row[0],
                "covername_entity_id": row[1],
                "covername_name": row[2],
                "covername_type": row[3],
                "person_alias": row[4],
                "person_alias_norm": row[5],
            }
        )
    return out


def bulk_person_candidates(
    cur,
    patterns: List[Dict],
) -> Dict[int, List[Tuple[int, str]]]:
    """
    Mapping alias_id -> list of (person_entity_id, person_name), computed set-wise.

    Matching logic:
    - Match persons where canonical_name equals alias (case/trim-insensitive)
      OR where person's own alias_norm matches the alias_norm.
    """
    if not patterns:
        return {}

    cur.execute(
        """
        CREATE TEMP TABLE tmp_cov_person_aliases (
          alias_id BIGINT PRIMARY KEY,
          person_alias TEXT NOT NULL,
          person_alias_norm TEXT NOT NULL
        ) ON COMMIT DROP;
        """
    )

    rows = [(p["alias_id"], p["person_alias"], p["person_alias_norm"]) for p in patterns]
    execute_values(
        cur,
        "INSERT INTO tmp_cov_person_aliases (alias_id, person_alias, person_alias_norm) VALUES %s",
        rows,
        page_size=5000,
    )

    cur.execute(
        """
        WITH person_aliases AS (
          SELECT ea.entity_id AS person_id, ea.alias_norm
          FROM entity_aliases ea
          JOIN entities e ON e.id = ea.entity_id
          WHERE e.entity_type = 'person'
        )
        SELECT
          t.alias_id,
          p.id AS person_id,
          p.canonical_name AS person_name
        FROM tmp_cov_person_aliases t
        JOIN entities p ON p.entity_type = 'person'
        LEFT JOIN person_aliases pa
          ON pa.person_id = p.id
         AND pa.alias_norm = t.person_alias_norm
        WHERE
          LOWER(TRIM(p.canonical_name)) = LOWER(TRIM(t.person_alias))
          OR pa.person_id IS NOT NULL
        ORDER BY t.alias_id, p.id;
        """
    )

    mapping: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for alias_id, pid, pname in cur.fetchall():
        mapping[alias_id].append((pid, pname))
    return dict(mapping)


def choose_single_candidate(
    candidates: List[Tuple[int, str]]
) -> Optional[Tuple[int, str]]:
    """
    Deterministic candidate choice rule:
    - If exactly 1 candidate, take it.
    - If multiple, return None (ambiguous).
    """
    if len(candidates) == 1:
        return candidates[0]
    return None


def insert_relationships(
    cur,
    resolved: List[Dict],
    *,
    source: str,
    confidence: float,
    dry_run: bool,
) -> int:
    """
    Insert covername_of relationships idempotently.
    """
    if not resolved:
        return 0
    if dry_run:
        return 0

    rows = []
    for r in resolved:
        rows.append(
            (
                r["covername_entity_id"],
                r["person_entity_id"],
                "covername_of",
                confidence,
                source,
                r.get("notes", None),
            )
        )

    execute_values(
        cur,
        """
        INSERT INTO entity_relationships
          (source_entity_id, target_entity_id, relationship_type, confidence, source, notes)
        VALUES %s
        ON CONFLICT (source_entity_id, target_entity_id, relationship_type) DO NOTHING
        """,
        rows,
        page_size=5000,
    )

    return len(rows)


def delete_aliases(
    cur,
    alias_ids: List[int],
    *,
    dry_run: bool,
) -> int:
    if not alias_ids:
        return 0
    if dry_run:
        return 0

    cur.execute(
        "DELETE FROM entity_aliases WHERE id = ANY(%s::bigint[])",
        (alias_ids,),
    )
    return cur.rowcount


def print_samples(resolved: List[Dict], ambiguous: List[Dict], max_n: int = 20) -> None:
    print("\n=== SAMPLE RESOLVED (will create relationship) ===", file=sys.stderr)
    for r in resolved[:max_n]:
        print(
            f"  {r['covername_name']} ({r['covername_entity_id']}) --covername_of--> "
            f"{r['person_name']} ({r['person_entity_id']})  [from alias '{r['person_alias']}']",
            file=sys.stderr,
        )
    if len(resolved) > max_n:
        print(f"  ... and {len(resolved) - max_n} more", file=sys.stderr)

    print("\n=== SAMPLE AMBIGUOUS (skipped) ===", file=sys.stderr)
    for a in ambiguous[:max_n]:
        cands = a.get("candidates", [])
        cand_str = ", ".join([f"{pid}:{pname}" for pid, pname in cands[:5]])
        if len(cands) > 5:
            cand_str += f", ... (+{len(cands)-5})"
        print(
            f"  alias '{a['person_alias']}' on {a['covername_name']} had {len(cands)} candidates: {cand_str}",
            file=sys.stderr,
        )
    if len(ambiguous) > max_n:
        print(f"  ... and {len(ambiguous) - max_n} more", file=sys.stderr)


def validate_post(cur) -> None:
    print("\n=== POST VALIDATION ===", file=sys.stderr)

    cur.execute(
        """
        SELECT COUNT(*)
        FROM entity_relationships
        WHERE relationship_type='covername_of'
        """
    )
    print(f"covername_of relationships: {cur.fetchone()[0]}", file=sys.stderr)

    cur.execute(
        """
        SELECT
          ea.alias_norm,
          COUNT(DISTINCT ea.entity_id) AS entity_count,
          array_agg(DISTINCT e.entity_type) AS entity_types
        FROM entity_aliases ea
        JOIN entities e ON e.id = ea.entity_id
        GROUP BY ea.alias_norm
        HAVING COUNT(DISTINCT ea.entity_id) > 1
        ORDER BY entity_count DESC
        LIMIT 20
        """
    )
    rows = cur.fetchall()
    print("\nTop remaining alias_norm collisions:", file=sys.stderr)
    for alias_norm, cnt, types in rows:
        print(f"  {alias_norm}: {cnt} entities, types={types}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Backfill covername_of relationships from alias patterns (fast set-based).")
    ap.add_argument("--dry-run", action="store_true", help="No DB writes; print what would happen.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of patterns scanned (for testing).")
    ap.add_argument("--no-delete-aliases", action="store_true", help="Do not delete person-like aliases from covernames.")
    ap.add_argument("--source", type=str, default="concordance_backfill", help="Provenance label for inserted edges.")
    ap.add_argument("--confidence", type=float, default=1.0, help="Confidence score for inserted edges (0..1).")
    ap.add_argument("--max-samples", type=int, default=20, help="How many samples to print for each category.")
    args = ap.parse_args()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            require_table(cur, "entities")
            require_table(cur, "entity_aliases")
            require_table(cur, "entity_relationships")

            cover_types = discover_covername_types(cur)
            if not cover_types:
                print("No covername-ish entity_type values found. Nothing to do.", file=sys.stderr)
                return

            print(f"Covername-like entity types: {cover_types}", file=sys.stderr)
            print("Scanning for covername aliases that look person-like...", file=sys.stderr)

            patterns = identify_patterns(cur, cover_types, limit=args.limit)
            print(f"Found {len(patterns)} covername aliases that look person-like.", file=sys.stderr)
            if not patterns:
                return

            print("Resolving person candidates (set-based join)...", file=sys.stderr)
            cand_map = bulk_person_candidates(cur, patterns)

            resolved: List[Dict] = []
            ambiguous: List[Dict] = []
            unmatched = 0

            for p in patterns:
                cands = cand_map.get(p["alias_id"], [])
                chosen = choose_single_candidate(cands)
                if chosen is None:
                    if len(cands) == 0:
                        unmatched += 1
                    ambiguous.append({**p, "candidates": cands})
                    continue

                person_id, person_name = chosen
                resolved.append(
                    {
                        **p,
                        "person_entity_id": person_id,
                        "person_name": person_name,
                        "notes": f"Migrated from alias: {p['person_alias']}",
                    }
                )

            multi = sum(1 for a in ambiguous if len(a.get("candidates", [])) > 1)
            print(
                f"Resolved uniquely: {len(resolved)} | ambiguous (multi-candidate): {multi} | no-match: {unmatched}",
                file=sys.stderr,
            )

            print_samples(resolved, ambiguous, max_n=args.max_samples)

            if args.dry_run:
                print("\n=== DRY RUN ===", file=sys.stderr)
                print(f"Would insert {len(resolved)} covername_of edges (skipping ambiguous/no-match).", file=sys.stderr)
                if not args.no_delete_aliases:
                    print(f"Would delete {len(resolved)} person-like aliases from covernames.", file=sys.stderr)
                else:
                    print("Would NOT delete aliases (--no-delete-aliases).", file=sys.stderr)
                return

            print("\nWriting relationships...", file=sys.stderr)
            inserted = insert_relationships(
                cur,
                resolved,
                source=args.source,
                confidence=float(args.confidence),
                dry_run=False,
            )

            removed = 0
            if not args.no_delete_aliases:
                print("Deleting migrated person-like aliases from covernames...", file=sys.stderr)
                removed = delete_aliases(cur, [r["alias_id"] for r in resolved], dry_run=False)

            conn.commit()
            print(f"Done. Attempted insert: {inserted} resolved edges.", file=sys.stderr)
            if not args.no_delete_aliases:
                print(f"Deleted {removed} aliases.", file=sys.stderr)

            validate_post(cur)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
