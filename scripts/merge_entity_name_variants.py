#!/usr/bin/env python3
"""Merge entities that differ only by stopwords or name variants.

Groups entities whose canonical_name matches after:
- Removing stopwords (the, a, an, of, and, or, etc.)
- Stripping title prefixes (Dr., Mr., Mrs., Ms., Prof., etc.)
- Normalizing to a comparable key (e.g., "Office of Strategic Services"
  vs "Office Strategic Services" → same key)

Uses O(n) grouping and batched per-group merges (minimal DB round-trips).

For each group, keeps the entity with the most complete name (or most aliases/mentions).
Uses the same merge execution as collapse_duplicate_entities / merge_codename_entities.

Usage:
    python scripts/merge_entity_name_variants.py
    python scripts/merge_entity_name_variants.py --apply
    python scripts/merge_entity_name_variants.py --apply --limit 50
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values


# =============================================================================
# Stopwords and titles
# =============================================================================

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "in", "on", "at", "by", "with",
    "to", "from", "as", "into", "through", "during", "before", "after",
})

# Titles to strip from the start of names (case-insensitive)
TITLE_PREFIXES = (
    "dr.", "mr.", "mrs.", "ms.", "miss", "prof.", "sir", "dame",
    "rev.", "gen.", "col.", "maj.", "lt.", "capt.", "sgt.",
)


def _normalize_for_comparison(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not name:
        return ""
    s = re.sub(r"[^\w\s]", " ", name.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _alias_norm(name: str) -> str:
    """Match DB alias_norm: remove non-word chars, lowercase, collapse spaces."""
    if not name:
        return ""
    s = re.sub(r"[^\w\s]", "", name.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_titles(name: str) -> str:
    """Remove common title prefixes from start of name."""
    s = name.strip()
    for prefix in TITLE_PREFIXES:
        if s.lower().startswith(prefix) and (len(s) == len(prefix) or s[len(prefix)].isspace()):
            s = s[len(prefix):].strip()
            break
    return s


def _stopword_normalized_key(name: str) -> str:
    """Key for exact stopword-normalized matching."""
    norm = _normalize_for_comparison(_strip_titles(name))
    tokens = [t for t in norm.split() if t not in STOPWORDS]
    return " ".join(sorted(tokens))


# =============================================================================
# Database
# =============================================================================

def get_conn():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", os.environ.get("POSTGRES_HOST", "localhost")),
        port=int(os.environ.get("DB_PORT", os.environ.get("POSTGRES_PORT", "5432"))),
        dbname=os.environ.get("DB_NAME", os.environ.get("POSTGRES_DB", "neh")),
        user=os.environ.get("DB_USER", os.environ.get("POSTGRES_USER", "neh")),
        password=os.environ.get("DB_PASS", os.environ.get("POSTGRES_PASSWORD", "neh")),
    )


@dataclass
class EntityInfo:
    id: int
    canonical_name: str
    entity_type: str
    alias_count: int
    mention_count: int


def _score_entity(info: EntityInfo) -> Tuple[int, int, int]:
    """Higher = better. Preference: longer name, more aliases, more mentions, lower id."""
    name_len = len(info.canonical_name)
    return (name_len, info.alias_count, info.mention_count)


def find_mergeable_groups(cur) -> Dict[str, List[EntityInfo]]:
    """
    Find groups of entities that match by stopword-normalized key. O(n).
    Returns dict: group_key -> list of EntityInfo (sorted by preference, winner first).
    """
    cur.execute("""
        SELECT e.id, e.canonical_name, e.entity_type,
               COALESCE(a.cnt, 0) AS alias_count,
               COALESCE(m.cnt, 0) AS mention_count
        FROM entities e
        LEFT JOIN (SELECT entity_id, COUNT(*) AS cnt FROM entity_aliases GROUP BY entity_id) a ON a.entity_id = e.id
        LEFT JOIN (SELECT entity_id, COUNT(*) AS cnt FROM entity_mentions GROUP BY entity_id) m ON m.entity_id = e.id
    """)
    rows = cur.fetchall()

    entities = [
        EntityInfo(
            id=r["id"],
            canonical_name=r["canonical_name"],
            entity_type=r["entity_type"],
            alias_count=r["alias_count"],
            mention_count=r["mention_count"],
        )
        for r in rows
    ]

    # O(n): group by stopword-normalized key
    by_key: Dict[str, List[EntityInfo]] = defaultdict(list)
    for e in entities:
        key = _stopword_normalized_key(e.canonical_name)
        if key:
            by_key[key].append(e)

    groups = {}
    for group in by_key.values():
        if len(group) > 1:
            group.sort(key=lambda x: _score_entity(x), reverse=True)
            group_key = f"{group[0].id}:{group[0].canonical_name[:40]}"
            groups[group_key] = group

    return groups


def _tables_that_exist(cur, candidates: List[str]) -> set:
    """Return set of table names that actually exist in the DB."""
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public' AND tablename = ANY(%s)
    """, (candidates,))
    return {r[0] for r in cur.fetchall()}


def _columns_that_exist(cur, table: str, candidates: List[str]) -> set:
    """Return set of column names that exist in the given table."""
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = ANY(%s)
    """, (table, candidates))
    return {r[0] for r in cur.fetchall()}


def _step(label: str, t0: float) -> float:
    """Print step header, return current time."""
    elapsed = time.monotonic() - t0
    print(f"  [{elapsed:6.1f}s] {label}...", end="", flush=True)
    return time.monotonic()


def _substep(label: str, t0: float) -> float:
    """Print substep (indented), return current time."""
    elapsed = time.monotonic() - t0
    print(f"    [{elapsed:6.1f}s] {label}...", end="", flush=True)
    return time.monotonic()


def _done(t_step: float, detail: str = "") -> None:
    """Print step completion."""
    dt = time.monotonic() - t_step
    suffix = f" ({detail})" if detail else ""
    print(f" done {dt:.1f}s{suffix}", flush=True)


def apply_merges_bulk(
    conn,
    groups_list: List[Tuple[str, List[EntityInfo]]],
) -> Dict[str, int]:
    """
    Apply ALL merges in bulk — one pass per table, not per group.
    Uses a temp table of (winner_id, donor_id) so every table operation
    is a single SQL statement that processes all groups at once.
    """
    stats: Dict[str, int] = {
        "groups": len(groups_list), "deleted": 0,
        "mentions": 0, "aliases": 0, "pem": 0,
    }
    t0 = time.monotonic()

    # ── Build merge map in Python ──────────────────────────────────
    merge_pairs = []        # (winner_id, donor_id)
    alias_inserts = []      # (winner_id, alias, alias_norm)  -- deduped per winner
    seen_per_winner: Dict[int, set] = defaultdict(set)

    for _, entities in groups_list:
        winner = entities[0]
        for donor in entities[1:]:
            merge_pairs.append((winner.id, donor.id))
            anorm = _alias_norm(donor.canonical_name)
            if anorm and anorm not in seen_per_winner[winner.id]:
                seen_per_winner[winner.id].add(anorm)
                alias_inserts.append((winner.id, donor.canonical_name, anorm))

    all_donor_ids = [p[1] for p in merge_pairs]
    print(f"  Merge map: {len(merge_pairs)} pairs, {len(alias_inserts)} alias inserts", flush=True)

    with conn.cursor() as cur:
        # ── Create temp merge map ──────────────────────────────────
        ts = _step("Creating merge map temp table", t0)
        t1 = _substep("INSERT rows", t0)
        cur.execute("""
            CREATE TEMP TABLE _merge_map (
                winner_id BIGINT NOT NULL,
                donor_id  BIGINT NOT NULL
            ) ON COMMIT DROP
        """)
        execute_values(cur, "INSERT INTO _merge_map (winner_id, donor_id) VALUES %s", merge_pairs)
        _done(t1, f"{len(merge_pairs)} rows")
        t1 = _substep("CREATE indexes", t0)
        cur.execute("CREATE INDEX _mm_donor ON _merge_map (donor_id)")
        cur.execute("CREATE INDEX _mm_winner ON _merge_map (winner_id)")
        cur.execute("ANALYZE _merge_map")
        _done(t1)
        _done(ts, f"{len(merge_pairs)} rows")

        # ── Check which optional tables exist ──────────────────────
        ts = _step("Checking table existence", t0)
        all_table_names = [
            "entity_mentions", "entity_aliases", "page_entity_mentions",
            "entity_links", "entity_relationships",
            "entity_alias_preferred", "entity_alias_overrides",
            "document_anchors", "ocr_variant_allowlist",
            "ocr_variant_clusters", "speaker_map", "alias_referent_rules",
            "entity_citations", "entity_resolution_reviews",
            "mention_candidates", "entities",
        ]
        existing = _tables_that_exist(cur, all_table_names)
        _done(ts, f"{len(existing)} tables found")

        def _has(t: str) -> bool:
            return t in existing

        # ── 1. entity_mentions ─────────────────────────────────────
        if _has("entity_mentions"):
            ts = _step("entity_mentions", t0)
            # 1a. Build winner's (winner_id, chunk_id, surface) for hash join
            t1 = _substep("build winner mention set", t0)
            cur.execute("""
                CREATE TEMP TABLE _winner_mentions (
                    winner_id BIGINT, chunk_id BIGINT, surface TEXT
                ) ON COMMIT DROP
            """)
            cur.execute("""
                INSERT INTO _winner_mentions (winner_id, chunk_id, surface)
                SELECT mm.winner_id, em.chunk_id, em.surface
                FROM entity_mentions em
                JOIN _merge_map mm ON em.entity_id = mm.winner_id
            """)
            n_winner = cur.rowcount if cur.rowcount else 0
            cur.execute("CREATE INDEX _wm_lookup ON _winner_mentions (winner_id, chunk_id, surface)")
            _done(t1, f"{n_winner} rows")
            # 1b. Delete conflicts via temp table (hash join, not correlated subquery)
            t1 = _substep("find+delete conflicts", t0)
            cur.execute("""
                CREATE TEMP TABLE _em_conflict_ids (id BIGINT PRIMARY KEY) ON COMMIT DROP
            """)
            cur.execute("""
                INSERT INTO _em_conflict_ids (id)
                SELECT DISTINCT em.id
                FROM entity_mentions em
                JOIN _merge_map mm ON em.entity_id = mm.donor_id
                JOIN _winner_mentions wm ON wm.winner_id = mm.winner_id
                    AND wm.chunk_id = em.chunk_id AND wm.surface = em.surface
            """)
            n_conflict = cur.rowcount
            cur.execute("DELETE FROM entity_mentions WHERE id IN (SELECT id FROM _em_conflict_ids)")
            _done(t1, f"{n_conflict} deleted")
            # 1c. Update remaining donor rows to winner
            t1 = _substep("update donor→winner", t0)
            cur.execute("""
                UPDATE entity_mentions em
                SET entity_id = mm.winner_id
                FROM _merge_map mm
                WHERE em.entity_id = mm.donor_id
            """)
            stats["mentions"] = cur.rowcount
            _done(t1, f"{stats['mentions']} updated")
            _done(ts, f"{n_conflict} conflicts, {stats['mentions']} transferred")

        # ── 2. entity_aliases ──────────────────────────────────────
        if _has("entity_aliases"):
            ts = _step("entity_aliases", t0)
            # 2a. Insert donor canonical names as aliases for winners
            if alias_inserts:
                t1 = _substep("insert donor names", t0)
                execute_values(
                    cur,
                    """
                    INSERT INTO entity_aliases (entity_id, alias, alias_norm, kind)
                    VALUES %s
                    ON CONFLICT (entity_id, alias_norm) DO NOTHING
                    """,
                    alias_inserts,
                    template="(%s, %s, %s, 'alt')",
                )
                _done(t1, f"{len(alias_inserts)} rows")
            # 2b. Delete donor aliases that conflict with winner's aliases
            t1 = _substep("delete conflicts", t0)
            cur.execute("""
                DELETE FROM entity_aliases ea
                USING _merge_map mm
                WHERE ea.entity_id = mm.donor_id
                AND EXISTS (
                    SELECT 1 FROM entity_aliases w
                    WHERE w.entity_id = mm.winner_id
                    AND w.alias_norm = ea.alias_norm
                )
            """)
            conflict_del = cur.rowcount
            _done(t1, f"{conflict_del} deleted")
            # 2c. Among donors → same winner, keep one per alias_norm
            t1 = _substep("dedup donor aliases", t0)
            cur.execute("""
                DELETE FROM entity_aliases ea
                USING _merge_map mm
                WHERE ea.entity_id = mm.donor_id
                AND ea.id NOT IN (
                    SELECT DISTINCT ON (mm2.winner_id, ea2.alias_norm) ea2.id
                    FROM entity_aliases ea2
                    JOIN _merge_map mm2 ON mm2.donor_id = ea2.entity_id
                    ORDER BY mm2.winner_id, ea2.alias_norm, ea2.id
                )
            """)
            dedup_del = cur.rowcount
            _done(t1, f"{dedup_del} deleted")
            # 2d. Update remaining
            t1 = _substep("update donor→winner", t0)
            cur.execute("""
                UPDATE entity_aliases ea
                SET entity_id = mm.winner_id
                FROM _merge_map mm
                WHERE ea.entity_id = mm.donor_id
            """)
            stats["aliases"] = cur.rowcount
            _done(t1, f"{stats['aliases']} updated")
            _done(ts, f"{conflict_del} conflicts, {dedup_del} deduped, {stats['aliases']} transferred")

        # ── 3. page_entity_mentions ────────────────────────────────
        if _has("page_entity_mentions"):
            ts = _step("page_entity_mentions", t0)
            # 3a. Pre-build winner's (page_id, surface_norm) set for fast conflict check
            t1 = _substep("build winner PEM set", t0)
            cur.execute("""
                CREATE TEMP TABLE _winner_pem (winner_id BIGINT, page_id BIGINT, surface_norm TEXT) ON COMMIT DROP
            """)
            cur.execute("""
                INSERT INTO _winner_pem
                SELECT mm.winner_id, pem.page_id, pem.surface_norm
                FROM page_entity_mentions pem
                JOIN _merge_map mm ON mm.winner_id = pem.entity_id
            """)
            cur.execute("CREATE INDEX _wp_lookup ON _winner_pem (winner_id, page_id, surface_norm)")
            _done(t1)
            # 3b. Delete donor rows that conflict with winner (hash join via temp table)
            t1 = _substep("find+delete conflicts", t0)
            cur.execute("""
                CREATE TEMP TABLE _pem_conflict_ids (id BIGINT PRIMARY KEY) ON COMMIT DROP
            """)
            cur.execute("""
                INSERT INTO _pem_conflict_ids (id)
                SELECT DISTINCT pem.id
                FROM page_entity_mentions pem
                JOIN _merge_map mm ON pem.entity_id = mm.donor_id
                JOIN _winner_pem w ON w.winner_id = mm.winner_id
                    AND w.page_id = pem.page_id AND w.surface_norm = pem.surface_norm
            """)
            conflict_del = cur.rowcount
            cur.execute("DELETE FROM page_entity_mentions WHERE id IN (SELECT id FROM _pem_conflict_ids)")
            _done(t1, f"{conflict_del} deleted")
            # 3c. Among donors→same winner, keep one per (winner_id, page_id, surface_norm)
            t1 = _substep("dedup donor PEM", t0)
            cur.execute("""
                CREATE TEMP TABLE _pem_keep (id BIGINT PRIMARY KEY) ON COMMIT DROP
            """)
            cur.execute("""
                INSERT INTO _pem_keep (id)
                SELECT DISTINCT ON (mm.winner_id, pem.page_id, pem.surface_norm) pem.id
                FROM page_entity_mentions pem
                JOIN _merge_map mm ON mm.donor_id = pem.entity_id
                ORDER BY mm.winner_id, pem.page_id, pem.surface_norm, pem.id
            """)
            cur.execute("""
                DELETE FROM page_entity_mentions pem
                WHERE pem.entity_id IN (SELECT donor_id FROM _merge_map)
                AND pem.id NOT IN (SELECT id FROM _pem_keep)
            """)
            dedup_del = cur.rowcount
            _done(t1, f"{dedup_del} deleted")
            # 3d. Update remaining
            t1 = _substep("update donor→winner", t0)
            cur.execute("""
                UPDATE page_entity_mentions pem
                SET entity_id = mm.winner_id
                FROM _merge_map mm
                WHERE pem.entity_id = mm.donor_id
            """)
            stats["pem"] = cur.rowcount
            _done(t1, f"{stats['pem']} updated")
            _done(ts, f"{conflict_del} conflicts, {dedup_del} deduped, {stats['pem']} transferred")

        # ── 4. entity_links ────────────────────────────────────────
        if _has("entity_links"):
            ts = _step("entity_links: remove self-loops + update", t0)
            cur.execute("""
                DELETE FROM entity_links el
                USING _merge_map mm
                WHERE (el.from_entity_id = mm.winner_id AND el.to_entity_id = mm.donor_id)
                   OR (el.from_entity_id = mm.donor_id AND el.to_entity_id = mm.winner_id)
            """)
            cur.execute("""
                UPDATE entity_links SET from_entity_id = mm.winner_id
                FROM _merge_map mm WHERE entity_links.from_entity_id = mm.donor_id
            """)
            cur.execute("""
                UPDATE entity_links SET to_entity_id = mm.winner_id
                FROM _merge_map mm WHERE entity_links.to_entity_id = mm.donor_id
            """)
            _done(ts)

        # ── 5. entity_relationships ────────────────────────────────
        if _has("entity_relationships"):
            ts = _step("entity_relationships: remove self-loops + update", t0)
            cur.execute("""
                DELETE FROM entity_relationships er
                USING _merge_map mm
                WHERE (er.source_entity_id = mm.winner_id AND er.target_entity_id = mm.donor_id)
                   OR (er.source_entity_id = mm.donor_id AND er.target_entity_id = mm.winner_id)
            """)
            cur.execute("""
                UPDATE entity_relationships SET source_entity_id = mm.winner_id
                FROM _merge_map mm WHERE entity_relationships.source_entity_id = mm.donor_id
            """)
            cur.execute("""
                UPDATE entity_relationships SET target_entity_id = mm.winner_id
                FROM _merge_map mm WHERE entity_relationships.target_entity_id = mm.donor_id
            """)
            _done(ts)

        # ── 6. Simple FK tables ────────────────────────────────────
        simple_fk = [
            ("entity_alias_preferred", "preferred_entity_id"),
            ("entity_alias_overrides", "forced_entity_id"),
            ("entity_alias_overrides", "banned_entity_id"),
            ("document_anchors", "entity_id"),
            ("ocr_variant_allowlist", "entity_id"),
            ("ocr_variant_clusters", "canonical_entity_id"),
            ("speaker_map", "entity_id"),
            ("alias_referent_rules", "entity_id"),
            ("entity_citations", "entity_id"),
            ("entity_resolution_reviews", "chosen_entity_id"),
        ]
        for table, column in simple_fk:
            if not _has(table):
                continue
            ts = _step(f"{table}.{column}: update", t0)
            cur.execute(f"""
                UPDATE "{table}" SET "{column}" = mm.winner_id
                FROM _merge_map mm
                WHERE "{table}"."{column}" = mm.donor_id
            """)
            _done(ts, f"{cur.rowcount} rows")

        # ── 7. mention_candidates ──────────────────────────────────
        if _has("mention_candidates"):
            mc_cols = ("resolved_entity_id", "anchored_to_entity_id",
                       "canonical_entity_id", "current_entity_id")
            for col in _columns_that_exist(cur, "mention_candidates", list(mc_cols)):
                ts = _step(f"mention_candidates.{col}: update", t0)
                cur.execute(f"""
                    UPDATE mention_candidates SET "{col}" = mm.winner_id
                    FROM _merge_map mm
                    WHERE mention_candidates."{col}" = mm.donor_id
                """)
                _done(ts, f"{cur.rowcount} rows")

        # ── 8. Delete donor entities ───────────────────────────────
        ts = _step("Deleting donor entities", t0)
        cur.execute("""
            DELETE FROM entities
            WHERE id IN (SELECT donor_id FROM _merge_map)
        """)
        stats["deleted"] = cur.rowcount
        _done(ts, f"{stats['deleted']} deleted")

    conn.commit()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge entities that differ only by stopwords or name variants",
    )
    ap.add_argument("--apply", action="store_true", help="Actually merge (default: dry-run)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of groups to process")
    ap.add_argument("--show", type=int, default=30, help="Show N groups in dry-run")
    ap.add_argument("--statement-timeout-ms", type=int, default=0, help="Postgres statement timeout")
    args = ap.parse_args()

    print("=" * 60, flush=True)
    print("MERGE ENTITY NAME VARIANTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Mode: {'APPLY (will modify DB)' if args.apply else 'DRY RUN (read-only)'}")
    print(f"  Matching: stopword removal, title stripping (O(n) grouping)")
    print("", flush=True)

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if args.statement_timeout_ms > 0:
                cur.execute(f"SET statement_timeout = {args.statement_timeout_ms}")

            print("Finding mergeable groups...", flush=True)
            t0 = time.monotonic()
            groups = find_mergeable_groups(cur)
            print(f"  Found {len(groups)} groups in {time.monotonic() - t0:.1f}s", flush=True)

        if not groups:
            print("No mergeable entities found.")
            return 0

        total_entities = sum(len(v) for v in groups.values())
        total_losers = total_entities - len(groups)


        print(f"\nFound {len(groups)} groups ({total_entities} entities)")
        print(f"Will merge {total_losers} entities into {len(groups)} winners\n")

        if not args.apply:
            print("=" * 60, flush=True)
            print("[DRY RUN] Sample groups:", flush=True)
            print("=" * 60, flush=True)
            for i, (_, entities) in enumerate(list(groups.items())[: args.show]):
                winner = entities[0]
                losers = entities[1:]
                names = ", ".join(f"'{e.canonical_name}' (id={e.id})" for e in entities)
                print(f"Group {i+1}: {names}")
                print(f"  WINNER: id={winner.id} '{winner.canonical_name}' aliases={winner.alias_count} mentions={winner.mention_count}")
                for loser in losers[:3]:
                    print(f"  MERGE:  id={loser.id} '{loser.canonical_name}' -> winner")
                if len(losers) > 3:
                    print(f"  ... and {len(losers) - 3} more")
                print()
            if len(groups) > args.show:
                print(f"... and {len(groups) - args.show} more groups\n")
            print("Run with --apply to merge.")
            return 0

        # Apply — use fresh connection so we start with clean transaction state
        print("=" * 60, flush=True)
        print("APPLYING MERGES", flush=True)
        print("=" * 60, flush=True)

        groups_list = list(groups.items())
        if args.limit:
            groups_list = groups_list[: args.limit]

        conn.close()
        conn = get_conn()

        t0 = time.monotonic()
        stats = apply_merges_bulk(conn, groups_list)
        elapsed = time.monotonic() - t0
        print("", flush=True)
        print("=" * 60, flush=True)
        print("COMPLETE", flush=True)
        print("=" * 60, flush=True)
        print(f"  Groups: {stats['groups']}")
        print(f"  Entities deleted: {stats['deleted']}")
        print(f"  Mentions transferred: {stats['mentions']}")
        print(f"  Aliases transferred: {stats['aliases']}")
        print(f"  Page entity mentions: {stats['pem']}")
        print(f"  Time: {elapsed:.1f}s")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
