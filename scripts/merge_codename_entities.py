#!/usr/bin/env python3
"""Merge circular codename entity pairs and redundant codename entities.

The concordance index created separate entity rows for:
  - English codenames (ABRAHAM)
  - Russian transliterations (AVRAAM)
  - The actual person (Jack Soble)
...then cross-linked them as aliases.  This causes resolution to land on
the codename entity instead of the person entity ("PAL = Robert" bug).

This script:
  Phase 0: Introspects FK constraints referencing entities(id)
  Phase 1a: Discovers circular codename pairs (A<->B mutual aliases)
  Phase 1b: (optional) Discovers non-circular codename entities whose
            canonical_name appears as an alias of another entity
  Phase 2: Builds a merge plan using a scoring function (person-like
           canonical wins; ambiguous cases go to needs-review)
  Phase 3: Executes merges with SAVEPOINT-per-merge safety
  Phase 4: Reports stats, failures, and needs-review items

Usage:
    python scripts/merge_codename_entities.py --dry-run
    python scripts/merge_codename_entities.py
    python scripts/merge_codename_entities.py --include-noncircular
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 100
DEFAULT_TIMEOUT_S = 7200  # 2 hours

# Garbage alias filter (same as cleanup_concordance.py)
_GARBAGE_RE = re.compile(r"[0-9;–—]")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FKRef:
    """A foreign key referencing entities(id)."""
    table: str
    column: str
    on_delete: str  # 'CASCADE', 'SET NULL', 'RESTRICT', 'NO ACTION'


@dataclass
class MergePlan:
    """One merge operation: absorb donor into survivor."""
    donor_id: int
    donor_canonical: str
    survivor_id: int
    survivor_canonical: str
    reason: str


@dataclass
class NeedsReview:
    """Ambiguous group that needs manual review."""
    entity_ids: List[int]
    canonical_names: List[str]
    reason: str


@dataclass
class MergeResult:
    """Result of one merge attempt."""
    donor_id: int
    survivor_id: int
    success: bool
    error: str = ""
    elapsed_ms: float = 0.0
    step_timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _is_allcaps(name: str) -> bool:
    """True if name is all uppercase (codename-like)."""
    return name == name.upper() and any(c.isalpha() for c in name)


def _is_person_like(name: str) -> bool:
    """True if name looks like a person name (2-4 Title Case tokens)."""
    tokens = name.split()
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    return all(t[0].isupper() and not t.isupper() for t in tokens if len(t) > 1)


def _is_garbage_alias(alias: str) -> bool:
    """True if alias is garbage (>3 words, contains digits/punctuation)."""
    tokens = alias.split()
    if len(tokens) > 3:
        return True
    if _GARBAGE_RE.search(alias):
        return True
    return False


def _score_entity(cur, entity_id: int, canonical: str, entity_type: str) -> float:
    """Score an entity for survivor selection.

    Higher = more likely to be the real/canonical entity.

    +3  person-like canonical (2-4 Title Case tokens, not ALLCAPS)
    +2  broader mention coverage (distinct chunk count via log)
    +1  more non-garbage aliases
    -2  bucket entity type (cover_name, unidentified*)
    -1  ALLCAPS single token (codename)
    -1  high garbage-alias ratio (>50%)
    """
    score = 0.0

    # Person-like canonical
    if _is_person_like(canonical):
        score += 3.0
    elif not _is_allcaps(canonical) and len(canonical.split()) >= 2:
        score += 1.5  # multi-word non-ALLCAPS (partial credit)

    # ALLCAPS single token penalty
    if _is_allcaps(canonical) and len(canonical.split()) == 1:
        score -= 1.0

    # Bucket entity type penalty
    if entity_type and entity_type.lower() in ("cover_name", "unidentified soviet", "soviet"):
        score -= 2.0

    # Mention count boost
    cur.execute("SELECT COUNT(*) FROM entity_mentions WHERE entity_id = %s", (entity_id,))
    mention_count = cur.fetchone()[0]
    if mention_count > 0:
        score += min(2.0, 0.5 * math.log10(mention_count + 1))

    # Alias quality
    cur.execute("SELECT alias FROM entity_aliases WHERE entity_id = %s", (entity_id,))
    aliases = [r[0] for r in cur.fetchall()]
    total_aliases = len(aliases)
    garbage_aliases = sum(1 for a in aliases if _is_garbage_alias(a))
    non_garbage = total_aliases - garbage_aliases
    score += min(1.0, non_garbage * 0.1)

    # Garbage ratio penalty
    if total_aliases > 0 and garbage_aliases / total_aliases > 0.5:
        score -= 1.0

    return score


# ---------------------------------------------------------------------------
# Phase 0: FK introspection
# ---------------------------------------------------------------------------

def discover_fk_refs(cur) -> List[FKRef]:
    """Query pg_constraint for all FKs referencing entities(id)."""
    cur.execute("""
        SELECT
            conrelid::regclass::text AS table_name,
            a.attname AS column_name,
            CASE confdeltype
                WHEN 'c' THEN 'CASCADE'
                WHEN 'n' THEN 'SET NULL'
                WHEN 'r' THEN 'RESTRICT'
                WHEN 'a' THEN 'NO ACTION'
                ELSE 'UNKNOWN'
            END AS on_delete
        FROM pg_constraint c
        JOIN pg_attribute a
            ON a.attrelid = c.conrelid
            AND a.attnum = ANY(c.conkey)
        WHERE confrelid = 'entities'::regclass
          AND contype = 'f'
        ORDER BY conrelid::regclass::text, a.attname
    """)
    refs = []
    for row in cur.fetchall():
        refs.append(FKRef(table=row[0], column=row[1], on_delete=row[2]))
    return refs


# ---------------------------------------------------------------------------
# Phase 1a: Circular pair discovery
# ---------------------------------------------------------------------------

def discover_circular_pairs(cur) -> List[Tuple[int, str, int, str]]:
    """Find all circular alias pairs: A has alias B.canonical, B has alias A.canonical.

    Returns deduplicated list of (entity_a_id, entity_a_canonical, entity_b_id, entity_b_canonical)
    where entity_a_id < entity_b_id (to avoid duplicates).
    """
    cur.execute("""
        SELECT DISTINCT
            LEAST(ea1.entity_id, ea2.entity_id) AS id_lo,
            GREATEST(ea1.entity_id, ea2.entity_id) AS id_hi
        FROM entity_aliases ea1
        JOIN entities e1 ON e1.id = ea1.entity_id
        JOIN entities e2 ON LOWER(e2.canonical_name) = LOWER(ea1.alias)
        JOIN entity_aliases ea2
            ON ea2.entity_id = e2.id
            AND LOWER(ea2.alias) = LOWER(e1.canonical_name)
        WHERE ea1.entity_id != ea2.entity_id
    """)
    pair_ids = cur.fetchall()

    # Load canonical names
    pairs = []
    seen = set()
    for id_lo, id_hi in pair_ids:
        key = (id_lo, id_hi)
        if key in seen:
            continue
        seen.add(key)
        cur.execute(
            "SELECT id, canonical_name FROM entities WHERE id IN (%s, %s)",
            (id_lo, id_hi),
        )
        rows = {r[0]: r[1] for r in cur.fetchall()}
        if id_lo in rows and id_hi in rows:
            pairs.append((id_lo, rows[id_lo], id_hi, rows[id_hi]))

    return pairs


# ---------------------------------------------------------------------------
# Phase 1b: Non-circular codename entities
# ---------------------------------------------------------------------------

def discover_noncircular_codename_entities(cur) -> List[Tuple[int, str, int, str]]:
    """Find entities whose ALLCAPS canonical_name is an alias of a different entity,
    but NOT part of a circular pair.

    Returns list of (codename_entity_id, codename_canonical, host_entity_id, host_canonical).
    """
    cur.execute("""
        SELECT e_code.id, e_code.canonical_name,
               e_host.id, e_host.canonical_name
        FROM entities e_code
        JOIN entity_aliases ea ON LOWER(ea.alias) = LOWER(e_code.canonical_name)
        JOIN entities e_host ON e_host.id = ea.entity_id
        WHERE e_host.id != e_code.id
          AND e_code.canonical_name = UPPER(e_code.canonical_name)
          AND e_code.canonical_name ~ '[A-Z]'
        ORDER BY e_code.canonical_name
    """)
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Phase 2: Build merge plan
# ---------------------------------------------------------------------------

def find_person_host_for_entity(cur, entity_id: int, canonical: str) -> List[Tuple[int, str, str]]:
    """Find other entities that have this entity's canonical_name as an alias.

    Returns list of (host_entity_id, host_canonical, host_entity_type).
    These are the "person entity" candidates that this codename entity should merge into.
    """
    cur.execute("""
        SELECT DISTINCT e.id, e.canonical_name, e.entity_type
        FROM entities e
        JOIN entity_aliases ea ON ea.entity_id = e.id
        WHERE LOWER(ea.alias) = LOWER(%s)
          AND e.id != %s
    """, (canonical, entity_id))
    return cur.fetchall()


def build_merge_plan(
    cur,
    circular_pairs: List[Tuple[int, str, int, str]],
    noncircular: List[Tuple[int, str, int, str]],
) -> Tuple[List[MergePlan], List[NeedsReview]]:
    """Build the merge plan from discovered pairs.

    For each pair/group, determine survivor using scoring function.
    Ambiguous cases go to needs_review.
    """
    merges: List[MergePlan] = []
    needs_review: List[NeedsReview] = []
    # Track entities already planned for merge (as donors) to avoid conflicts
    planned_donors: Set[int] = set()

    # --- Process circular pairs ---
    for id_a, canon_a, id_b, canon_b in circular_pairs:
        if id_a in planned_donors or id_b in planned_donors:
            continue

        # Find third-party "person" entities that claim either codename as alias
        hosts_a = find_person_host_for_entity(cur, id_a, canon_a)
        hosts_b = find_person_host_for_entity(cur, id_b, canon_b)

        # Merge host lists and exclude the pair members themselves
        pair_ids = {id_a, id_b}
        all_hosts = {}
        for eid, ecanon, etype in hosts_a + hosts_b:
            if eid not in pair_ids and eid not in planned_donors:
                all_hosts[eid] = (ecanon, etype)

        if len(all_hosts) == 1:
            # Single clear host -> merge both codename entities into it
            host_id, (host_canon, host_type) = next(iter(all_hosts.items()))
            merges.append(MergePlan(
                donor_id=id_a, donor_canonical=canon_a,
                survivor_id=host_id, survivor_canonical=host_canon,
                reason=f"circular_pair_to_person({canon_a}/{canon_b}->{host_canon})",
            ))
            merges.append(MergePlan(
                donor_id=id_b, donor_canonical=canon_b,
                survivor_id=host_id, survivor_canonical=host_canon,
                reason=f"circular_pair_to_person({canon_a}/{canon_b}->{host_canon})",
            ))
            planned_donors.add(id_a)
            planned_donors.add(id_b)

        elif len(all_hosts) > 1:
            # Multiple hosts -> ambiguous, needs review
            all_ids = [id_a, id_b] + list(all_hosts.keys())
            all_names = [canon_a, canon_b] + [h[0] for h in all_hosts.values()]
            needs_review.append(NeedsReview(
                entity_ids=all_ids,
                canonical_names=all_names,
                reason=f"circular_pair_multiple_hosts({len(all_hosts)} candidates)",
            ))

        else:
            # No person host -> merge the pair into one entity
            # Use scoring to pick survivor
            cur.execute("SELECT entity_type FROM entities WHERE id = %s", (id_a,))
            type_a = (cur.fetchone() or ("",))[0] or ""
            cur.execute("SELECT entity_type FROM entities WHERE id = %s", (id_b,))
            type_b = (cur.fetchone() or ("",))[0] or ""

            score_a = _score_entity(cur, id_a, canon_a, type_a)
            score_b = _score_entity(cur, id_b, canon_b, type_b)

            if score_a >= score_b:
                survivor_id, survivor_canon = id_a, canon_a
                donor_id, donor_canon = id_b, canon_b
            else:
                survivor_id, survivor_canon = id_b, canon_b
                donor_id, donor_canon = id_a, canon_a

            merges.append(MergePlan(
                donor_id=donor_id, donor_canonical=donor_canon,
                survivor_id=survivor_id, survivor_canonical=survivor_canon,
                reason=f"circular_pair_no_person(scores:{score_a:.2f}/{score_b:.2f})",
            ))
            planned_donors.add(donor_id)

    # --- Process non-circular codename entities ---
    for code_id, code_canon, host_id, host_canon in noncircular:
        if code_id in planned_donors:
            continue
        if host_id in planned_donors:
            continue
        # If host_id is itself a codename entity (ALLCAPS), skip for now
        # (it'll be handled as a circular pair or a different noncircular)
        if code_id == host_id:
            continue

        # Check if multiple hosts exist for this codename
        hosts = find_person_host_for_entity(cur, code_id, code_canon)
        hosts = [(eid, ec, et) for eid, ec, et in hosts if eid not in planned_donors]

        if len(hosts) == 1:
            eid, ecanon, _ = hosts[0]
            merges.append(MergePlan(
                donor_id=code_id, donor_canonical=code_canon,
                survivor_id=eid, survivor_canonical=ecanon,
                reason=f"noncircular_codename_to_host({code_canon}->{ecanon})",
            ))
            planned_donors.add(code_id)
        elif len(hosts) > 1:
            all_ids = [code_id] + [h[0] for h in hosts]
            all_names = [code_canon] + [h[1] for h in hosts]
            needs_review.append(NeedsReview(
                entity_ids=all_ids,
                canonical_names=all_names,
                reason=f"noncircular_multiple_hosts({len(hosts)} candidates)",
            ))

    return merges, needs_review


# ---------------------------------------------------------------------------
# Phase 3: Execute merges
# ---------------------------------------------------------------------------

def execute_merge(
    conn,
    cur,
    plan: MergePlan,
    restrict_fks: List[FKRef],
) -> MergeResult:
    """Execute a single donor->survivor merge inside a SAVEPOINT.

    Steps:
      1. Re-point entity_relationships (bulk)
      2. Add donor canonical as alias of survivor
      3. Re-point aliases (bulk, dedup by unique constraint)
      4. Re-point mentions (bulk, dedup)
      5. Handle RESTRICT FK tables
      6. Delete donor entity
    """
    donor = plan.donor_id
    survivor = plan.survivor_id
    timings: Dict[str, float] = {}
    merge_t0 = time.time()

    try:
        cur.execute("SAVEPOINT merge_one")

        # Step 1: Re-point entity_relationships (4 queries, all bulk)
        t = time.time()
        cur.execute("""
            UPDATE entity_relationships
            SET source_entity_id = %s
            WHERE source_entity_id = %s
              AND NOT EXISTS (
                  SELECT 1 FROM entity_relationships er2
                  WHERE er2.source_entity_id = %s
                    AND er2.target_entity_id = entity_relationships.target_entity_id
                    AND er2.relationship_type = entity_relationships.relationship_type
              )
        """, (survivor, donor, survivor))
        cur.execute(
            "DELETE FROM entity_relationships WHERE source_entity_id = %s",
            (donor,),
        )
        cur.execute("""
            UPDATE entity_relationships
            SET target_entity_id = %s
            WHERE target_entity_id = %s
              AND NOT EXISTS (
                  SELECT 1 FROM entity_relationships er2
                  WHERE er2.target_entity_id = %s
                    AND er2.source_entity_id = entity_relationships.source_entity_id
                    AND er2.relationship_type = entity_relationships.relationship_type
              )
        """, (survivor, donor, survivor))
        cur.execute(
            "DELETE FROM entity_relationships WHERE target_entity_id = %s",
            (donor,),
        )
        timings["rels"] = (time.time() - t) * 1000

        # Step 2: Add donor's canonical_name as alias of survivor (1 query)
        t = time.time()
        donor_canon_norm = re.sub(r"[^\w\s]", "", plan.donor_canonical.lower()).strip()
        if donor_canon_norm:
            cur.execute("""
                INSERT INTO entity_aliases (entity_id, alias, alias_norm, kind)
                VALUES (%s, %s, %s, 'alt')
                ON CONFLICT (entity_id, alias_norm) DO NOTHING
            """, (survivor, plan.donor_canonical, donor_canon_norm))
        timings["add_alias"] = (time.time() - t) * 1000

        # Step 3: Re-point aliases from donor to survivor (2 queries, bulk)
        t = time.time()
        cur.execute("""
            DELETE FROM entity_aliases
            WHERE entity_id = %s
              AND alias_norm IN (
                  SELECT alias_norm FROM entity_aliases WHERE entity_id = %s
              )
        """, (donor, survivor))
        cur.execute(
            "UPDATE entity_aliases SET entity_id = %s WHERE entity_id = %s",
            (survivor, donor),
        )
        timings["aliases"] = (time.time() - t) * 1000

        # Step 4: Re-point entity_mentions (2 queries, bulk)
        t = time.time()
        cur.execute("""
            DELETE FROM entity_mentions
            WHERE entity_id = %s
              AND chunk_id IN (
                  SELECT chunk_id FROM entity_mentions WHERE entity_id = %s
              )
        """, (donor, survivor))
        cur.execute(
            "UPDATE entity_mentions SET entity_id = %s WHERE entity_id = %s",
            (survivor, donor),
        )
        timings["mentions"] = (time.time() - t) * 1000

        # Step 5: Handle RESTRICT FK tables
        t = time.time()
        for fk in restrict_fks:
            cur.execute("SAVEPOINT fk_repoint")
            try:
                cur.execute(
                    f'UPDATE "{fk.table}" SET "{fk.column}" = %s WHERE "{fk.column}" = %s',
                    (survivor, donor),
                )
                cur.execute("RELEASE SAVEPOINT fk_repoint")
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT fk_repoint")
                cur.execute("SAVEPOINT fk_repoint")
                try:
                    cur.execute(
                        f'UPDATE "{fk.table}" SET "{fk.column}" = NULL WHERE "{fk.column}" = %s',
                        (donor,),
                    )
                    cur.execute("RELEASE SAVEPOINT fk_repoint")
                except Exception:
                    cur.execute("ROLLBACK TO SAVEPOINT fk_repoint")
                    cur.execute(
                        f'DELETE FROM "{fk.table}" WHERE "{fk.column}" = %s',
                        (donor,),
                    )
        timings["restrict_fks"] = (time.time() - t) * 1000

        # Step 6: Delete donor entity (CASCADE handles remaining FKs)
        t = time.time()
        cur.execute("DELETE FROM entities WHERE id = %s", (donor,))
        timings["delete"] = (time.time() - t) * 1000

        cur.execute("RELEASE SAVEPOINT merge_one")
        elapsed_ms = (time.time() - merge_t0) * 1000
        return MergeResult(
            donor_id=donor, survivor_id=survivor, success=True,
            elapsed_ms=elapsed_ms, step_timings=timings,
        )

    except Exception as e:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT merge_one")
        except Exception:
            pass
        elapsed_ms = (time.time() - merge_t0) * 1000
        return MergeResult(
            donor_id=donor, survivor_id=survivor,
            success=False, error=str(e)[:200],
            elapsed_ms=elapsed_ms, step_timings=timings,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge circular codename entities into person entities"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview merge plan without making changes")
    parser.add_argument("--db", default=None,
                        help="Database URL (default: DATABASE_URL env var)")
    parser.add_argument("--slug", default="vassiliev_venona_index_20260130",
                        help="Concordance source slug")
    parser.add_argument("--include-noncircular", action="store_true",
                        help="Also process non-circular codename entities (Phase 1b)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                        help=f"Statement timeout in seconds (default: {DEFAULT_TIMEOUT_S})")
    parser.add_argument("--export-csv", default=None,
                        help="Path to write merge log CSV")
    args = parser.parse_args()

    db_url = args.db or os.environ.get(
        "DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"
    )
    print(f"Connecting to: {db_url.split('@')[-1]}")

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    # Statement timeout
    timeout_ms = args.timeout * 1000
    if timeout_ms > 0:
        cur.execute("SET statement_timeout = %s", (str(timeout_ms),))
        print(f"Statement timeout: {args.timeout}s")
    else:
        cur.execute("SET statement_timeout = 0")
        print("Statement timeout: off")

    # =====================================================================
    # Phase 0: FK introspection
    # =====================================================================
    print("\n" + "=" * 60)
    print("Phase 0: FK Introspection")
    print("=" * 60)

    all_fks = discover_fk_refs(cur)
    cascade_fks = [f for f in all_fks if f.on_delete == "CASCADE"]
    setnull_fks = [f for f in all_fks if f.on_delete == "SET NULL"]
    restrict_fks = [f for f in all_fks if f.on_delete in ("RESTRICT", "NO ACTION")]

    print(f"  Total FKs referencing entities(id): {len(all_fks)}")
    print(f"    CASCADE:  {len(cascade_fks)} (auto-handled)")
    for f in cascade_fks:
        print(f"      {f.table}.{f.column}")
    print(f"    SET NULL: {len(setnull_fks)} (auto-handled)")
    for f in setnull_fks:
        print(f"      {f.table}.{f.column}")
    print(f"    RESTRICT: {len(restrict_fks)} (must re-point before delete)")
    for f in restrict_fks:
        print(f"      {f.table}.{f.column}")

    # =====================================================================
    # Phase 1a: Discover circular pairs
    # =====================================================================
    print("\n" + "=" * 60)
    print("Phase 1a: Circular Codename Pairs")
    print("=" * 60)

    # Get entity count before
    cur.execute("SELECT COUNT(*) FROM entities")
    entities_before = cur.fetchone()[0]

    circular_pairs = discover_circular_pairs(cur)
    print(f"  Found {len(circular_pairs)} unique circular pairs")
    for id_a, canon_a, id_b, canon_b in circular_pairs[:20]:
        print(f"    {canon_a} ({id_a}) <-> {canon_b} ({id_b})")
    if len(circular_pairs) > 20:
        print(f"    ... and {len(circular_pairs) - 20} more")

    # =====================================================================
    # Phase 1b: Non-circular codename entities (optional)
    # =====================================================================
    noncircular = []
    if args.include_noncircular:
        print("\n" + "=" * 60)
        print("Phase 1b: Non-Circular Codename Entities")
        print("=" * 60)

        # Get all circular entity IDs to exclude them
        circular_ids = set()
        for id_a, _, id_b, _ in circular_pairs:
            circular_ids.add(id_a)
            circular_ids.add(id_b)

        raw_noncircular = discover_noncircular_codename_entities(cur)
        # Filter out entities already in circular pairs
        noncircular = [
            (code_id, code_canon, host_id, host_canon)
            for code_id, code_canon, host_id, host_canon in raw_noncircular
            if code_id not in circular_ids
        ]
        print(f"  Found {len(noncircular)} non-circular codename entities")
        for code_id, code_canon, host_id, host_canon in noncircular[:20]:
            print(f"    {code_canon} ({code_id}) -> {host_canon} ({host_id})")
        if len(noncircular) > 20:
            print(f"    ... and {len(noncircular) - 20} more")
    else:
        print("\n  (Phase 1b skipped -- use --include-noncircular to enable)")

    # =====================================================================
    # Phase 2: Build merge plan
    # =====================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Building Merge Plan")
    print("=" * 60)

    merges, needs_review = build_merge_plan(cur, circular_pairs, noncircular)

    print(f"\n  Merge operations planned: {len(merges)}")
    print(f"  Needs review (ambiguous): {len(needs_review)}")

    # Show merge plan
    if merges:
        print("\n  Planned merges:")
        for m in merges[:30]:
            print(f"    {m.donor_canonical} ({m.donor_id}) -> {m.survivor_canonical} ({m.survivor_id})")
            print(f"      reason: {m.reason}")
        if len(merges) > 30:
            print(f"    ... and {len(merges) - 30} more")

    if needs_review:
        print("\n  Needs review:")
        for nr in needs_review[:20]:
            ids_str = ", ".join(str(i) for i in nr.entity_ids)
            names_str = ", ".join(nr.canonical_names)
            print(f"    [{ids_str}] {names_str}")
            print(f"      reason: {nr.reason}")

    # =====================================================================
    # Dry-run exit
    # =====================================================================
    if args.dry_run:
        print("\n--dry-run: no changes made.")

        # Export needs-review to file
        if needs_review:
            review_path = args.export_csv or "needs_review.jsonl"
            if review_path.endswith(".csv"):
                review_path = review_path.replace(".csv", "_review.jsonl")
            else:
                review_path = "needs_review.jsonl"
            with open(review_path, "w") as f:
                for nr in needs_review:
                    json.dump({
                        "entity_ids": nr.entity_ids,
                        "canonical_names": nr.canonical_names,
                        "reason": nr.reason,
                    }, f)
                    f.write("\n")
            print(f"  Needs-review items written to: {review_path}")

        conn.rollback()
        conn.close()
        return

    if not merges:
        print("\nNo merges to perform.")
        conn.close()
        return

    # =====================================================================
    # Phase 3: Execute merges
    # =====================================================================
    COMMIT_BATCH = 10

    print("\n" + "=" * 60)
    print("Phase 3: Executing Merges")
    print("=" * 60)
    print(f"  Total merges: {len(merges)}, batch-commit every {COMMIT_BATCH}", flush=True)

    results: List[MergeResult] = []
    successes = 0
    failures = 0
    t0 = time.time()
    batch_ok = 0

    for i, plan in enumerate(merges, 1):
        result = execute_merge(conn, cur, plan, restrict_fks)
        results.append(result)

        if result.success:
            successes += 1
            batch_ok += 1
            # Show per-merge timing with slowest step
            slow_step = ""
            if result.step_timings:
                worst = max(result.step_timings.items(), key=lambda x: x[1])
                slow_step = f" [slowest: {worst[0]}={worst[1]:.0f}ms]"
            print(
                f"  [{i}/{len(merges)}] OK {plan.donor_canonical} -> "
                f"{plan.survivor_canonical} ({result.elapsed_ms:.0f}ms){slow_step}",
                flush=True,
            )
        else:
            failures += 1
            # Rollback just the failed savepoint (already done inside execute_merge)
            # But we need to commit what we have so far in this batch
            if batch_ok > 0:
                conn.commit()
                batch_ok = 0
            print(
                f"  [{i}/{len(merges)}] FAIL {plan.donor_canonical} ({plan.donor_id}) "
                f"-> {plan.survivor_canonical} ({plan.survivor_id}): {result.error}",
                flush=True,
            )

        # Batch commit
        if batch_ok >= COMMIT_BATCH:
            conn.commit()
            batch_ok = 0

        # Summary every 25 merges
        if i % 25 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(merges) - i) / rate if rate > 0 else 0
            print(
                f"  --- Progress: {i}/{len(merges)} ({successes} ok, {failures} fail, "
                f"{rate:.1f}/s, ETA {remaining:.0f}s) ---",
                flush=True,
            )

    # Final commit for any remaining batch
    if batch_ok > 0:
        conn.commit()

    elapsed_total = time.time() - t0
    print(
        f"\n  Phase 3 complete: {successes} ok, {failures} fail in {elapsed_total:.1f}s "
        f"({successes / elapsed_total:.1f}/s)" if elapsed_total > 0 else
        f"\n  Phase 3 complete: {successes} ok, {failures} fail",
        flush=True,
    )

    # =====================================================================
    # Phase 4: Report
    # =====================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Report")
    print("=" * 60)

    cur.execute("SELECT COUNT(*) FROM entities")
    entities_after = cur.fetchone()[0]
    print(f"  Entities before: {entities_before}")
    print(f"  Entities after:  {entities_after}")
    print(f"  Removed:         {entities_before - entities_after}")
    print(f"  Merges: {successes} successful, {failures} failed")
    print(f"  Needs review: {len(needs_review)} ambiguous groups")

    # Verify remaining circular pairs
    remaining_circular = discover_circular_pairs(cur)
    print(f"  Remaining circular pairs: {len(remaining_circular)}")
    if remaining_circular:
        for id_a, canon_a, id_b, canon_b in remaining_circular[:10]:
            print(f"    {canon_a} ({id_a}) <-> {canon_b} ({id_b})")

    # Export CSV
    if args.export_csv:
        with open(args.export_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "donor_id", "donor_canonical", "survivor_id", "survivor_canonical",
                "reason", "success", "error",
            ])
            for plan, result in zip(merges, results):
                writer.writerow([
                    plan.donor_id, plan.donor_canonical,
                    plan.survivor_id, plan.survivor_canonical,
                    plan.reason, result.success, result.error,
                ])
        print(f"  Merge log written to: {args.export_csv}")

    # Export needs-review
    if needs_review:
        review_path = "needs_review.jsonl"
        with open(review_path, "w") as f:
            for nr in needs_review:
                json.dump({
                    "entity_ids": nr.entity_ids,
                    "canonical_names": nr.canonical_names,
                    "reason": nr.reason,
                }, f)
                f.write("\n")
        print(f"  Needs-review items written to: {review_path}")

    # List failures
    failed_results = [r for r in results if not r.success]
    if failed_results:
        print(f"\n  Failed merges ({len(failed_results)}):")
        for r in failed_results[:20]:
            print(f"    donor={r.donor_id} -> survivor={r.survivor_id}: {r.error}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
