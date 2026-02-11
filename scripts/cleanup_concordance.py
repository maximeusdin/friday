#!/usr/bin/env python3
"""Clean up garbage entities and aliases from the concordance source.

Garbage = canonical_name or alias with > 3 words, or containing
semicolons, em-dashes, or digits (page references / sentence fragments).

Integration: This script writes directly to the database (no separate import).
After running, re-export concordance CSVs so they match the cleaned DB.
See docs/cleanup_concordance_integration.md for the full workflow.

Usage:
    python scripts/cleanup_concordance.py [--dry-run] [--db DATABASE_URL]
    python scripts/cleanup_concordance.py --confirm          # ask once before bulk delete
    python scripts/cleanup_concordance.py --adjudicate       # interactive: (d)elete / (s)kip / (e)dit per item
    python scripts/cleanup_concordance.py --orphans-only    # list entities with no aliases (not alias-forms); then exit
    python scripts/cleanup_concordance.py --delete-orphans [--confirm]  # delete orphan entities (keep alias-forms), log full rows
    python scripts/cleanup_concordance.py --delete-garbage-entities [--confirm]  # bulk delete garbage entities + aliases; log full rows for reversibility
    python scripts/cleanup_concordance.py --restore-deleted [PATH] --confirm     # restore from orphan_runs and garbage_entity_runs in session JSON
    python scripts/cleanup_concordance.py --apply-file PATH --slug SLUG           # apply merges/deletes from JSON only (no coded garbage rules)
"""
import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

import psycopg2
from psycopg2 import extras as psycopg2_extras

# Batch size for slow operations (UPDATE mention_candidates, etc.)
BATCH_SIZE = 5000
# Batch size for restore inserts (execute_values page_size)
RESTORE_BATCH_SIZE = 1000
# Default statement timeout: 2 hours (Postgres uses milliseconds)
DEFAULT_STATEMENT_TIMEOUT_MS = 2 * 60 * 60 * 1000

GARBAGE_CONDITION_ALIAS = """
    array_length(string_to_array(trim(alias), ' '), 1) > 3
    OR alias ~ '[0-9;–—]'
"""

GARBAGE_CONDITION_ENTITY = """
    array_length(string_to_array(trim(canonical_name), ' '), 1) > 3
    OR canonical_name ~ '[0-9;–—]'
"""


def _batched(ids: List[int], size: int):
    """Yield chunks of ids of at most `size`."""
    for i in range(0, len(ids), size):
        yield ids[i : i + size]


def _normalize_alias_for_db(alias: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (match ingest alias_norm)."""
    import re
    s = (alias or "").lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _row_to_json(obj):
    """Make a DB row value JSON-serializable."""
    if obj is None:
        return None
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_row_to_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _row_to_json(v) for k, v in obj.items()}
    if hasattr(obj, "__float__") and not isinstance(obj, bool):  # Decimal
        return float(obj)
    return str(obj)


def _fetch_table_as_dicts(cur, sql: str, params: tuple) -> list:
    """Run query and return list of dicts (column -> JSON-serializable value)."""
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, [_row_to_json(v) for v in row])) for row in cur.fetchall()]


def _session_path(slug: str) -> str:
    """Return the path to the adjudication session file for this slug."""
    return f"cleanup_session_{slug}.json"


def _load_session(slug: str) -> dict:
    """Load an existing session or return an empty one."""
    path = _session_path(slug)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"alias_decisions": {}, "entity_decisions": {}}


def _save_session(slug: str, session: dict) -> None:
    """Write the session to disk (atomic-ish via write + flush)."""
    path = _session_path(slug)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


def _run_adjudicate(cur, conn, source_id: int, batch_size: int, slug: str = "") -> None:
    """Interactive: for each garbage alias/entity, (d)elete / (s)kip / (e)dit / (m)erge.

    Decisions are saved to a JSON session file after every choice so you
    can Ctrl-C at any time and resume later.  On resume the script skips
    items you already decided on and jumps to the first unseen one.
    """
    session = _load_session(slug)
    a_dec = session.setdefault("alias_decisions", {})
    e_dec = session.setdefault("entity_decisions", {})

    cur.execute(f"""
        SELECT ea.id, ea.alias, ea.entity_id
        FROM entity_aliases ea
        WHERE ea.source_id = %s AND ({GARBAGE_CONDITION_ALIAS})
        ORDER BY length(ea.alias) DESC
    """, (source_id,))
    garbage_aliases: List[Tuple[int, str, int]] = cur.fetchall()

    cur.execute(f"""
        SELECT e.id, e.canonical_name
        FROM entities e
        WHERE e.source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
        ORDER BY length(e.canonical_name) DESC
    """, (source_id,))
    garbage_entities: List[Tuple[int, str]] = cur.fetchall()

    # Pre-load all entity canonical names for this source (for "similar entity" lookups)
    cur.execute(
        "SELECT id, canonical_name, entity_type FROM entities WHERE source_id = %s ORDER BY id",
        (source_id,),
    )
    all_entities: List[Tuple[int, str, str]] = cur.fetchall()

    # Count how many we already decided
    n_prev_aliases = sum(1 for aid, _, _ in garbage_aliases if str(aid) in a_dec)
    n_prev_entities = sum(1 for eid, _ in garbage_entities if str(eid) in e_dec)
    if n_prev_aliases or n_prev_entities:
        print(f"\n  Resuming session from {_session_path(slug)}")
        print(f"  Previously decided: {n_prev_aliases}/{len(garbage_aliases)} aliases, "
              f"{n_prev_entities}/{len(garbage_entities)} entities")

    # ---- Aliases ----
    print("\n--- Garbage aliases ---")
    print("  Actions: (d)elete  (s)kip  (e)dit  (m)erge into entity")
    n_ga = len(garbage_aliases)
    for i, (aid, alias, eid) in enumerate(garbage_aliases, 1):
        aid_str = str(aid)
        if aid_str in a_dec:
            continue  # already decided in a previous session

        # Show the alias's parent entity for context
        cur.execute(
            "SELECT canonical_name, entity_type FROM entities WHERE id = %s",
            (eid,),
        )
        parent = cur.fetchone()
        parent_str = f"  parent entity: {parent[0]!r} ({parent[1]})" if parent else ""
        print(f"\n  [{i}/{n_ga}] alias_id={aid}  entity_id={eid}")
        print(f"    alias:  {alias!r}")
        if parent_str:
            print(f"   {parent_str}")
        while True:
            choice = input("    (d)elete / (s)kip / (e)dit / (m)erge [s]: ").strip().lower() or "s"
            if choice in ("d", "s", "e", "m"):
                break
            print("    Enter d, s, e, or m")

        if choice == "d":
            a_dec[aid_str] = {"action": "delete"}
        elif choice == "s":
            a_dec[aid_str] = {"action": "skip"}
        elif choice == "e":
            new_val = input("    New alias: ").strip()
            if new_val:
                a_dec[aid_str] = {"action": "edit", "new_value": new_val}
            else:
                a_dec[aid_str] = {"action": "skip"}
        elif choice == "m":
            print("    Merge: create/find an entity and attach a clean alias to it.")
            ent_name = input("    Entity canonical name (e.g. 'Ethniko Apeleftherotiko Metopo'): ").strip()
            if not ent_name:
                print("    Skipped (no entity name given).")
                a_dec[aid_str] = {"action": "skip"}
                _save_session(slug, session)
                continue
            new_alias = input("    Alias to attach (e.g. 'EAM', or Enter to skip alias): ").strip()
            ent_type = input("    Entity type [org/person/place/cover_name/topic/other] (default: org): ").strip().lower() or "org"
            if ent_type not in ("person", "org", "place", "cover_name", "topic", "other"):
                print(f"    Unknown type {ent_type!r}, defaulting to 'org'.")
                ent_type = "org"
            a_dec[aid_str] = {
                "action": "merge",
                "entity_name": ent_name,
                "alias_text": new_alias,
                "entity_type": ent_type,
            }

        _save_session(slug, session)

    # ---- Entities ----
    print("\n--- Garbage entities ---")
    n_ge = len(garbage_entities)
    for i, (eid, canonical_name) in enumerate(garbage_entities, 1):
        eid_str = str(eid)
        if eid_str in e_dec:
            continue  # already decided in a previous session

        # Show aliases belonging to this entity
        cur.execute(
            "SELECT alias, alias_type FROM entity_aliases WHERE entity_id = %s ORDER BY id",
            (eid,),
        )
        aliases_for_entity = cur.fetchall()

        # Show citations belonging to this entity
        cur.execute(
            "SELECT citation_text FROM entity_citations WHERE entity_id = %s ORDER BY id",
            (eid,),
        )
        citations_for_entity = cur.fetchall()

        # Find candidate "cleaner" entities
        words = canonical_name.split()
        candidates: List[Tuple[int, str, str]] = []
        cn_lower = canonical_name.lower()
        for other_id, other_name, other_type in all_entities:
            if other_id == eid:
                continue
            on_lower = other_name.lower()
            if len(other_name) < len(canonical_name):
                other_words = other_name.split()
                shared_prefix = 0
                for w1, w2 in zip(words, other_words):
                    if w1.lower() == w2.lower():
                        shared_prefix += 1
                    else:
                        break
                if on_lower in cn_lower or shared_prefix >= 2:
                    candidates.append((other_id, other_name, other_type))

        print(f"\n  [{i}/{n_ge}] entity_id={eid}")
        print(f"    name:  {canonical_name!r}")
        if aliases_for_entity:
            print(f"    aliases ({len(aliases_for_entity)}):")
            for a_text, a_type in aliases_for_entity[:8]:
                print(f"      - {a_text!r}  ({a_type})")
            if len(aliases_for_entity) > 8:
                print(f"      ... and {len(aliases_for_entity) - 8} more")
        else:
            print("    aliases: (none)")
        if citations_for_entity:
            print(f"    citations ({len(citations_for_entity)}):")
            for (ct,) in citations_for_entity[:5]:
                print(f"      - {ct!r}")
            if len(citations_for_entity) > 5:
                print(f"      ... and {len(citations_for_entity) - 5} more")
        if candidates:
            print(f"    ⚡ possible cleaner entities already in DB:")
            for c_id, c_name, c_type in candidates[:5]:
                print(f"      [{c_id}] {c_name!r} ({c_type})")
            if len(candidates) > 5:
                print(f"      ... and {len(candidates) - 5} more")
        else:
            print("    ⚠ no shorter matching entity found in DB")

        while True:
            choice = input("    (d)elete / (s)kip / (e)dit [s]: ").strip().lower() or "s"
            if choice in ("d", "s", "e"):
                break
            print("    Enter d, s, or e")

        if choice == "d":
            e_dec[eid_str] = {"action": "delete"}
        elif choice == "s":
            e_dec[eid_str] = {"action": "skip"}
        elif choice == "e":
            new_val = input("    New canonical_name: ").strip()
            if new_val:
                e_dec[eid_str] = {"action": "edit", "new_value": new_val}
            else:
                e_dec[eid_str] = {"action": "skip"}

        _save_session(slug, session)

    # ---- Rebuild action sets from the full session ----
    alias_delete_ids, alias_updates, alias_merges, entity_delete_ids, entity_updates = _session_to_actions(
        session
    )

    # --- Summary before applying ---
    n_del_a = len(alias_delete_ids)
    n_edit_a = len(alias_updates)
    n_merge = len(alias_merges)
    n_del_e = len(entity_delete_ids)
    n_edit_e = len(entity_updates)
    print(f"\n--- Pending actions (from session) ---")
    print(f"  Aliases:  {n_del_a} delete, {n_edit_a} edit, {n_merge} merge")
    print(f"  Entities: {n_del_e} delete, {n_edit_e} edit")
    if n_del_a + n_edit_a + n_merge + n_del_e + n_edit_e == 0:
        print("  Nothing to do.")
        return
    confirm = input("Apply? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted. No changes made. Session saved — re-run to resume or apply.")
        return

    _apply_actions(
        cur, conn, source_id,
        alias_delete_ids, alias_updates, alias_merges,
        entity_delete_ids, entity_updates,
    )

    # Clear adjudication decisions but keep orphan_runs for traceability
    sp = _session_path(slug)
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            cleared = json.load(f)
        cleared["alias_decisions"] = {}
        cleared["entity_decisions"] = {}
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(cleared, f, indent=2, ensure_ascii=False)
        print(f"  Cleared adjudication from session file: {sp}")

    print("\nAdjudication done.")


def _session_to_actions(session: dict, source_id: int = None) -> Tuple[Set[int], Dict[int, str], List[Tuple[int, str, str, str]], Set[int], Dict[int, str]]:
    """Build action sets from a session dict (alias_decisions, entity_decisions, deleted_entity_ids, or delete_entities_from_garbage_runs)."""
    a_dec = session.get("alias_decisions", {})
    e_dec = session.get("entity_decisions", {})

    alias_delete_ids: Set[int] = set()
    alias_updates: Dict[int, str] = {}
    alias_merges: List[Tuple[int, str, str, str]] = []

    for aid_str, dec in a_dec.items():
        aid = int(aid_str)
        if dec["action"] == "delete":
            alias_delete_ids.add(aid)
        elif dec["action"] == "edit":
            alias_updates[aid] = dec["new_value"]
        elif dec["action"] == "merge":
            alias_merges.append((
                aid,
                dec["entity_name"],
                dec.get("alias_text", ""),
                dec.get("entity_type", "org"),
            ))
            alias_delete_ids.add(aid)

    entity_delete_ids: Set[int] = set()
    entity_updates: Dict[int, str] = {}
    for eid_str, dec in e_dec.items():
        eid = int(eid_str)
        if dec["action"] == "delete":
            entity_delete_ids.add(eid)
        elif dec["action"] == "edit":
            entity_updates[eid] = dec["new_value"]

    # Explicit list of entity IDs to delete
    for eid in session.get("deleted_entity_ids", []):
        entity_delete_ids.add(int(eid))

    # Delete entities from garbage_entity_runs, except those in keep_entity_names
    if session.get("delete_entities_from_garbage_runs"):
        keep_names = frozenset(n.strip() for n in session.get("keep_entity_names", []))
        for run in session.get("garbage_entity_runs", []):
            if source_id is not None and run.get("source_id") != source_id:
                continue
            for ent in run.get("entities", []):
                if ent.get("canonical_name") in keep_names:
                    continue
                eid = ent.get("id")
                if eid is not None:
                    entity_delete_ids.add(int(eid))

    return alias_delete_ids, alias_updates, alias_merges, entity_delete_ids, entity_updates


def _apply_actions(
    cur,
    conn,
    source_id: int,
    alias_delete_ids: Set[int],
    alias_updates: Dict[int, str],
    alias_merges: List[Tuple[int, str, str, str]],
    entity_delete_ids: Set[int],
    entity_updates: Dict[int, str],
) -> None:
    """Apply session-derived actions to the database (edits, merges, then deletes)."""
    # Apply edits first (so updated rows no longer match garbage)
    if alias_updates:
        for aid, new_alias in alias_updates.items():
            norm = _normalize_alias_for_db(new_alias)
            cur.execute("UPDATE entity_aliases SET alias = %s, alias_norm = %s WHERE id = %s", (new_alias, norm, aid))
        print(f"  Updated {len(alias_updates)} alias(es).")
        conn.commit()
    if entity_updates:
        n_entity_updated = 0
        n_entity_merged = 0
        for eid, new_name in entity_updates.items():
            cur.execute("SELECT entity_type FROM entities WHERE id = %s", (eid,))
            row = cur.fetchone()
            entity_type = row[0] if row else "other"
            try:
                cur.execute("UPDATE entities SET canonical_name = %s WHERE id = %s", (new_name, eid))
                conn.commit()
                n_entity_updated += 1
            except psycopg2.IntegrityError:
                conn.rollback()
                # Duplicate (source_id, canonical_name, entity_type): merge eid into existing entity
                cur.execute(
                    "SELECT id FROM entities WHERE source_id = %s AND canonical_name = %s AND entity_type = %s AND id != %s",
                    (source_id, new_name, entity_type, eid),
                )
                target_row = cur.fetchone()
                if not target_row:
                    print(f"  [skip] entity [{eid}] -> {new_name!r}: existing row not found", file=sys.stderr)
                    continue
                target_eid = target_row[0]
                # Remove aliases that would duplicate (target already has same alias_norm); then reassign the rest
                cur.execute(
                    """DELETE FROM entity_aliases a1 WHERE a1.entity_id = %s
                       AND EXISTS (SELECT 1 FROM entity_aliases a2 WHERE a2.entity_id = %s AND a2.alias_norm = a1.alias_norm)""",
                    (eid, target_eid),
                )
                cur.execute("UPDATE entity_aliases SET entity_id = %s WHERE entity_id = %s", (target_eid, eid))
                cur.execute("UPDATE entity_citations SET entity_id = %s WHERE entity_id = %s", (target_eid, eid))
                cur.execute("DELETE FROM entity_links WHERE from_entity_id = %s AND to_entity_id = %s", (eid, eid))
                cur.execute("UPDATE entity_links SET from_entity_id = %s WHERE from_entity_id = %s", (target_eid, eid))
                cur.execute("UPDATE entity_links SET to_entity_id = %s WHERE to_entity_id = %s", (target_eid, eid))
                cur.execute("DELETE FROM entities WHERE id = %s", (eid,))
                conn.commit()
                n_entity_merged += 1
                print(f"  Merged entity [{eid}] into [{target_eid}] {new_name!r} (superset).")
        print(f"  Updated {n_entity_updated} entity(ies), merged {n_entity_merged} into existing.")

    # Apply merges: create/find entity, attach alias, garbage alias is already in delete set
    if alias_merges:
        for garbage_aid, ent_name, new_alias_text, ent_type in alias_merges:
            cur.execute(
                "SELECT id FROM entities WHERE source_id = %s AND canonical_name = %s ORDER BY id LIMIT 1",
                (source_id, ent_name),
            )
            row = cur.fetchone()
            if row:
                target_eid = row[0]
                print(f"  Merge: found existing entity [{target_eid}] {ent_name!r}")
            else:
                cur.execute(
                    "INSERT INTO entities (source_id, canonical_name, entity_type) VALUES (%s, %s, %s) RETURNING id",
                    (source_id, ent_name, ent_type),
                )
                target_eid = cur.fetchone()[0]
                print(f"  Merge: created entity [{target_eid}] {ent_name!r} ({ent_type})")

            if new_alias_text:
                norm = _normalize_alias_for_db(new_alias_text)
                cur.execute(
                    "SELECT id FROM entity_aliases WHERE entity_id = %s AND alias_norm = %s LIMIT 1",
                    (target_eid, norm),
                )
                if cur.fetchone():
                    print(f"    Alias {new_alias_text!r} already exists on entity [{target_eid}]")
                else:
                    cur.execute(
                        "INSERT INTO entity_aliases (source_id, entity_id, alias, alias_norm, alias_type) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (source_id, target_eid, new_alias_text, norm, "canonical"),
                    )
                    print(f"    Created alias {new_alias_text!r} on entity [{target_eid}]")
        conn.commit()
        print(f"  Applied {len(alias_merges)} merge(s).")

    if alias_delete_ids:
        placeholders = ",".join(["%s"] * len(alias_delete_ids))
        cur.execute(f"DELETE FROM entity_aliases WHERE id IN ({placeholders})", list(alias_delete_ids))
        print(f"  Deleted {cur.rowcount} alias(es).")
        conn.commit()

    if entity_delete_ids:
        ids_list = list(entity_delete_ids)
        placeholders = ",".join(["%s"] * len(ids_list))
        cur.execute(f"DELETE FROM entity_aliases WHERE entity_id IN ({placeholders})", ids_list)
        n_aliases = cur.rowcount
        cur.execute(f"DELETE FROM entity_citations WHERE entity_id IN ({placeholders})", ids_list)
        cur.execute(f"DELETE FROM entity_links WHERE from_entity_id IN ({placeholders}) OR to_entity_id IN ({placeholders})", ids_list + ids_list)
        cur.execute(f"DELETE FROM entities WHERE id IN ({placeholders})", ids_list)
        print(f"  Deleted {len(ids_list)} entity(ies) and {n_aliases} remaining alias(es).")
        conn.commit()


def _adapt_val_for_insert(val):
    """Convert values from JSON-loaded dicts for psycopg2 INSERT. Dict -> Json for JSONB; list passes through for array columns."""
    if isinstance(val, dict):
        return psycopg2_extras.Json(val)
    return val  # lists pass through - psycopg2 adapts to PostgreSQL array format


def _format_for_copy(val) -> str:
    """Format a value for PostgreSQL COPY text format (tab-delimited, \\N for NULL)."""
    if val is None:
        return "\\N"
    if isinstance(val, bool):
        return "t" if val else "f"
    if isinstance(val, (int, float)) or (hasattr(val, "__float__") and not isinstance(val, bool)):
        return str(val)
    if hasattr(val, "isoformat"):
        return val.isoformat().replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    if isinstance(val, dict):
        s = json.dumps(val, ensure_ascii=False).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
        return s
    if isinstance(val, list):
        return "{" + ",".join(str(x) for x in val) + "}"
    s = str(val).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    return s


def _insert_logged_rows_copy(cur, conn, table: str, rows: list, cols: list, on_conflict: bool = False, verbose: bool = True) -> int:
    """Bulk insert via COPY. If on_conflict: COPY to temp table, then INSERT ON CONFLICT DO NOTHING."""
    if not rows:
        return 0
    buf = io.BytesIO()
    cols_str = ",".join(f'"{c}"' for c in cols)
    for row in rows:
        line = "\t".join(_format_for_copy(row.get(c)) for c in cols) + "\n"
        buf.write(line.encode("utf-8"))
    buf.seek(0)
    try:
        if on_conflict and "id" in cols:
            staging = f"{table}_restore_staging"
            cur.execute(f"CREATE TEMP TABLE {staging} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
            cur.copy_expert(
                f"COPY {staging} ({cols_str}) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')",
                buf,
            )
            cur.execute(
                f"INSERT INTO {table} ({cols_str}) SELECT {cols_str} FROM {staging} ON CONFLICT (id) DO NOTHING"
            )
            n = cur.rowcount
        else:
            cur.copy_expert(
                f"COPY {table} ({cols_str}) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')",
                buf,
            )
            n = cur.rowcount
        if verbose:
            print(f"    COPY {len(rows)} rows", flush=True)
        return n
    except Exception:
        conn.rollback()
        raise


def _insert_logged_rows(
    cur, conn, table: str, rows: list, on_conflict_id_do_nothing: bool = False, batch_size: int = RESTORE_BATCH_SIZE,
    verbose: bool = True, commit: bool = True, use_copy: bool = True,
) -> int:
    """Re-insert rows from logged dicts. Uses COPY when possible (fast), else execute_values. Single transaction if commit=False."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    cols_str = ", ".join(f'"{c}"' for c in cols)
    template = "(" + ", ".join(["%s"] * len(cols)) + ")"
    sql = f"INSERT INTO {table} ({cols_str}) VALUES %s"
    if on_conflict_id_do_nothing and "id" in cols:
        sql += " ON CONFLICT (id) DO NOTHING"
    elif table in ("entity_links", "entity_citations", "entity_aliases") and "id" in cols:
        sql += " ON CONFLICT (id) DO NOTHING"

    # Prefer COPY for bulk speed (staging+ON CONFLICT for idempotent restore on entity_links/citations/aliases)
    use_copy_conflict = on_conflict_id_do_nothing or table in ("entity_links", "entity_citations", "entity_aliases")
    if use_copy:
        try:
            return _insert_logged_rows_copy(
                cur, conn, table, rows, cols,
                on_conflict=use_copy_conflict and "id" in cols,
                verbose=verbose,
            )
        except Exception as e:
            conn.rollback()
            print(f"    [fallback] COPY failed: {e}", flush=True)
            use_copy = False

    # Fallback: execute_values
    values_tuples = [tuple(_adapt_val_for_insert(row.get(c)) for c in cols) for row in rows]
    total_inserted = 0
    for i in range(0, len(values_tuples), batch_size):
        batch = values_tuples[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        n_batches = (len(values_tuples) + batch_size - 1) // batch_size
        if verbose and n_batches > 1:
            print(f"    batch {batch_num}/{n_batches} ({len(batch)} rows)...", flush=True)
        try:
            psycopg2_extras.execute_values(cur, sql, batch, template=template, page_size=len(batch))
            total_inserted += cur.rowcount
            if commit:
                conn.commit()
        except psycopg2.IntegrityError:
            conn.rollback()
            single_sql = f"INSERT INTO {table} ({cols_str}) VALUES ({template[1:-1]})"
            if on_conflict_id_do_nothing and "id" in cols:
                single_sql += " ON CONFLICT (id) DO NOTHING"
            for row in rows[i : i + batch_size]:
                vals = tuple(_adapt_val_for_insert(row.get(c)) for c in cols)
                try:
                    cur.execute(single_sql, vals)
                    total_inserted += cur.rowcount
                    if commit:
                        conn.commit()
                except psycopg2.IntegrityError:
                    conn.rollback()
    return total_inserted


def _restore_deleted_from_session(session: dict, cur, conn, source_id: int) -> None:
    """Restore entities/aliases/citations/links from orphan_runs and garbage_entity_runs. COPY + single transaction.
    Order: entities first (links reference them), then links (citations reference them), then aliases, then citations."""
    total_entities = 0
    total_aliases = 0
    total_citations = 0
    total_links = 0

    # Collect all rows from both run types (entity_links reference entities from both)
    all_entities = []
    all_links = []
    all_aliases = []
    all_citations = []

    for run in session.get("orphan_runs", []) + session.get("garbage_entity_runs", []):
        if run.get("source_id") != source_id:
            continue
        all_entities.extend(run.get("entities", []))
        all_links.extend(run.get("entity_links", []))
        all_citations.extend(run.get("entity_citations", []))
        all_aliases.extend(run.get("entity_aliases", []))
        all_aliases.extend(run.get("garbage_aliases", []))

    t_start = time.perf_counter()
    if all_entities:
        t0 = time.perf_counter()
        print(f"  Restoring {len(all_entities)} entities...", flush=True)
        total_entities += _insert_logged_rows(cur, conn, "entities", all_entities, on_conflict_id_do_nothing=True, commit=False, use_copy=False)
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Entity links reference entities; only restore links where both ends exist in DB
    cur.execute("SELECT id FROM entities WHERE source_id = %s", (source_id,))
    valid_entity_ids = frozenset(r[0] for r in cur.fetchall())
    n_before = len(all_links)
    all_links = [r for r in all_links if r.get("from_entity_id") in valid_entity_ids and r.get("to_entity_id") in valid_entity_ids]
    if n_before > len(all_links):
        print(f"  Skipping {n_before - len(all_links)} entity_links (from/to entity not in source).", flush=True)

    if all_links:
        t0 = time.perf_counter()
        print(f"  Restoring {len(all_links)} entity_links (COPY)...", flush=True)
        total_links += _insert_logged_rows(cur, conn, "entity_links", all_links, commit=False, use_copy=True)
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)
    if all_aliases:
        t0 = time.perf_counter()
        print(f"  Restoring {len(all_aliases)} entity_aliases (COPY)...", flush=True)
        total_aliases += _insert_logged_rows(cur, conn, "entity_aliases", all_aliases, commit=False, use_copy=True)
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)
    if all_citations:
        t0 = time.perf_counter()
        print(f"  Restoring {len(all_citations)} entity_citations (COPY)...", flush=True)
        total_citations += _insert_logged_rows(cur, conn, "entity_citations", all_citations, commit=False, use_copy=True)
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)

    conn.commit()
    print(f"  Restored: {total_entities} entities, {total_aliases} aliases, {total_citations} citations, {total_links} links in {time.perf_counter() - t_start:.1f}s total.")


def main():
    parser = argparse.ArgumentParser(description="Clean concordance garbage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count, don't delete")
    parser.add_argument("--db", default=None,
                        help="Database URL (default: DATABASE_URL env var or local)")
    parser.add_argument("--slug", default="vassiliev_venona_index_20260130",
                        help="Concordance source slug to clean")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Statement timeout in seconds (default: 7200). Use 0 for no limit.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for slow updates (default: {BATCH_SIZE})")
    parser.add_argument("--confirm", action="store_true",
                        help="Ask for confirmation before performing bulk delete")
    parser.add_argument("--adjudicate", action="store_true",
                        help="Interactive: for each item choose (d)elete / (s)kip / (e)dit")
    parser.add_argument("--apply-file", metavar="PATH", default=None,
                        help="Apply decisions from a saved session JSON file (e.g. cleanup_session_SLUG.json). Requires --slug. "
                        "Uses ONLY alias_decisions and entity_decisions from the JSON; never runs the coded garbage rules.")
    parser.add_argument("--keep-session", action="store_true",
                        help="When using --apply-file, do not delete the session file after applying.")
    parser.add_argument("--orphans-only", action="store_true",
                        help="Only list entities with no aliases that are not themselves aliases; then exit")
    parser.add_argument("--delete-orphans", action="store_true",
                        help="Delete orphan entities (no aliases); keep entities whose name is an alias elsewhere. Log full row data to session JSON (use with --confirm)")
    parser.add_argument("--delete-garbage-entities", action="store_true",
                        help="Bulk delete all garbage entities (and their aliases/citations/links); log full row data to session JSON for reversibility (use with --confirm)")
    parser.add_argument("--restore-deleted", metavar="PATH", nargs="?", const="",
                        help="Restore from orphan_runs and garbage_entity_runs in session JSON. Path defaults to cleanup_session_{slug}.json. Use with --confirm.")
    args = parser.parse_args()

    db_url = args.db or os.environ.get(
        "DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"
    )
    print(f"Connecting to: {db_url.split('@')[-1]}")

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    # Allow long-running statements (default 2 hours; 0 = no limit)
    timeout_ms = (args.timeout if args.timeout is not None else 7200) * 1000
    if timeout_ms > 0:
        cur.execute("SET statement_timeout = %s", (str(timeout_ms),))
        print(f"Statement timeout: {timeout_ms // 1000}s")
    else:
        cur.execute("SET statement_timeout = 0")
        print("Statement timeout: off")

    # Use explicit slug (or override via --slug)
    slug = args.slug
    print(f"Concordance source slug: {slug}")

    cur.execute("SELECT id FROM concordance_sources WHERE slug = %s", (slug,))
    row = cur.fetchone()
    if not row:
        print(f"ERROR: No concordance_source with slug={slug}")
        sys.exit(1)
    source_id = row[0]

    # Restore entities/aliases/citations/links from orphan_runs and garbage_entity_runs
    if args.restore_deleted is not None:
        path = os.path.abspath(args.restore_deleted) if args.restore_deleted else os.path.abspath(_session_path(slug))
        if not os.path.isfile(path):
            print(f"ERROR: Session file not found: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)
        n_orphan = len(session.get("orphan_runs", []))
        n_garbage = len(session.get("garbage_entity_runs", []))
        print(f"Loaded session from {path}")
        print(f"  orphan_runs: {n_orphan}, garbage_entity_runs: {n_garbage}")
        if n_orphan == 0 and n_garbage == 0:
            print("  Nothing to restore.")
            conn.close()
            return
        if args.confirm:
            reply = input("Restore deleted entities/aliases/citations/links to database? [y/N]: ").strip().lower()
            if reply != "y":
                print("Aborted.")
                conn.close()
                return
        _restore_deleted_from_session(session, cur, conn, source_id)
        for tbl in ("entities", "entity_aliases", "entity_citations", "entity_links"):
            cur.execute(
                f"SELECT setval(pg_get_serial_sequence('{tbl}', 'id'), GREATEST((SELECT COALESCE(MAX(id), 1) FROM {tbl}), 1))"
            )
        conn.commit()
        print("  Synced id sequences.")
        print("\nRestore done.")
        conn.close()
        return

    # Apply from a saved session JSON file (no interactive adjudication)
    # JSON-only: merges and deletes come solely from alias_decisions and entity_decisions.
    # The coded garbage rules (GARBAGE_CONDITION_*) are never run in this path.
    if args.apply_file:
        path = os.path.abspath(args.apply_file)
        if not os.path.isfile(path):
            print(f"ERROR: Session file not found: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)
        alias_delete_ids, alias_updates, alias_merges, entity_delete_ids, entity_updates = _session_to_actions(
            session, source_id=source_id
        )
        n_del_a = len(alias_delete_ids)
        n_edit_a = len(alias_updates)
        n_merge = len(alias_merges)
        n_del_e = len(entity_delete_ids)
        n_edit_e = len(entity_updates)
        print(f"Loaded session from {path}")
        print(f"  Aliases:  {n_del_a} delete, {n_edit_a} edit, {n_merge} merge")
        print(f"  Entities: {n_del_e} delete, {n_edit_e} edit")
        if n_del_a + n_edit_a + n_merge + n_del_e + n_edit_e == 0:
            print("  Nothing to do.")
            conn.close()
            return
        if args.confirm:
            confirm = input("Apply to database? [y/N]: ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                conn.close()
                return
        _apply_actions(
            cur, conn, source_id,
            alias_delete_ids, alias_updates, alias_merges,
            entity_delete_ids, entity_updates,
        )
        # Clear adjudication from session file but keep orphan_runs / garbage_entity_runs for traceability
        if os.path.abspath(path) == os.path.abspath(_session_path(slug)):
            session["alias_decisions"] = {}
            session["entity_decisions"] = {}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
            print(f"  Cleared adjudication from session file (kept orphan_runs / garbage_entity_runs).")
        print("\nApply done.")
        conn.close()
        return

    if args.orphans_only:
        cur.execute("""
            SELECT e.id, e.canonical_name, e.entity_type
            FROM entities e
            WHERE e.source_id = %s
              AND NOT EXISTS (
                SELECT 1 FROM entity_aliases ea
                WHERE ea.entity_id = e.id
              )
            ORDER BY e.id
        """, (source_id,))
        no_alias_raw = cur.fetchall()
        cur.execute(
            "SELECT alias, alias_norm FROM entity_aliases WHERE source_id = %s",
            (source_id,),
        )
        alias_rows = cur.fetchall()
        alias_raw_set = {a for a, _ in alias_rows if a}
        alias_norm_set = {n for _, n in alias_rows if n}
        out = []
        for eid, name, etype in no_alias_raw:
            if name and (name in alias_raw_set or _normalize_alias_for_db(name) in alias_norm_set):
                continue
            out.append((eid, name or "", etype))
        print(f"# {len(out)} entities (id\tcanonical_name\tentity_type)", file=sys.stderr)
        for eid, name, etype in out:
            print(f"{eid}\t{name}\t{etype}")
        conn.close()
        return

    if args.delete_orphans:
        cur.execute("""
            SELECT e.id, e.canonical_name, e.entity_type
            FROM entities e
            WHERE e.source_id = %s
              AND NOT EXISTS (
                SELECT 1 FROM entity_aliases ea
                WHERE ea.entity_id = e.id
              )
            ORDER BY e.id
        """, (source_id,))
        no_alias_raw = cur.fetchall()
        cur.execute(
            "SELECT alias, alias_norm FROM entity_aliases WHERE source_id = %s",
            (source_id,),
        )
        alias_rows = cur.fetchall()
        alias_raw_set = {a for a, _ in alias_rows if a}
        alias_norm_set = {n for _, n in alias_rows if n}
        to_delete = []
        for eid, name, etype in no_alias_raw:
            # Keep entities whose name is an alias elsewhere (they are "alias" entities, not true orphans)
            if name and (name in alias_raw_set or _normalize_alias_for_db(name) in alias_norm_set):
                continue
            to_delete.append(eid)

        if not to_delete:
            print("No orphan entities to delete (all no-alias entities are alias-forms and kept).", file=sys.stderr)
            conn.close()
            return

        ids_list = list(to_delete)
        placeholders = ",".join(["%s"] * len(ids_list))
        # Fetch full row data before any delete (for reversibility)
        log_entities = _fetch_table_as_dicts(
            cur, f"SELECT * FROM entities WHERE id IN ({placeholders})", tuple(ids_list)
        )
        log_entity_citations = _fetch_table_as_dicts(
            cur, f"SELECT * FROM entity_citations WHERE entity_id IN ({placeholders})", tuple(ids_list)
        )
        log_entity_links = _fetch_table_as_dicts(
            cur,
            f"SELECT * FROM entity_links WHERE from_entity_id IN ({placeholders}) OR to_entity_id IN ({placeholders})",
            tuple(ids_list) + tuple(ids_list),
        )
        # Orphans have no aliases by definition; log empty list for consistency
        log_entity_aliases = []

        log_path = _session_path(slug)
        if not args.confirm:
            reply = input(
                f"Delete {len(ids_list)} orphan entities and log full row data to {log_path}? [y/N]: "
            ).strip().lower()
            if reply != "y":
                print("Aborted.", file=sys.stderr)
                conn.close()
                return

        run_entry = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "source_id": source_id,
            "deleted_count": len(ids_list),
            "deleted_entity_ids": ids_list,
            "entities": log_entities,
            "entity_aliases": log_entity_aliases,
            "entity_citations": log_entity_citations,
            "entity_links": log_entity_links,
        }
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            log_data.setdefault("orphan_runs", []).append(run_entry)
        else:
            log_data = {"slug": slug, "orphan_runs": [run_entry]}
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        cur.execute(f"DELETE FROM entity_aliases WHERE entity_id IN ({placeholders})", ids_list)
        cur.execute(f"DELETE FROM entity_citations WHERE entity_id IN ({placeholders})", ids_list)
        cur.execute(
            f"DELETE FROM entity_links WHERE from_entity_id IN ({placeholders}) OR to_entity_id IN ({placeholders})",
            ids_list + ids_list,
        )
        cur.execute(f"DELETE FROM entities WHERE id IN ({placeholders})", ids_list)
        conn.commit()
        print(f"Deleted {len(ids_list)} orphan entities. Full row data logged to {log_path}", file=sys.stderr)
        conn.close()
        return

    print(f"Source ID: {source_id}")

    # Count totals
    cur.execute("SELECT COUNT(*) FROM entities e WHERE e.source_id = %s", (source_id,))
    total_entities = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM entity_aliases ea WHERE ea.source_id = %s", (source_id,))
    total_aliases = cur.fetchone()[0]
    print(f"\nTotal entities: {total_entities}")
    print(f"Total aliases:  {total_aliases}")

    # Count garbage aliases
    cur.execute(f"""
        SELECT COUNT(*)
        FROM entity_aliases ea
        WHERE ea.source_id = %s AND ({GARBAGE_CONDITION_ALIAS})
    """, (source_id,))
    garbage_aliases = cur.fetchone()[0]
    print(f"\nGarbage aliases to delete: {garbage_aliases}")

    # Show sample garbage aliases
    cur.execute(f"""
        SELECT ea.alias, ea.entity_id
        FROM entity_aliases ea
        WHERE ea.source_id = %s AND ({GARBAGE_CONDITION_ALIAS})
        ORDER BY length(ea.alias) DESC
        LIMIT 20
    """, (source_id,))
    samples = cur.fetchall()
    if samples:
        print("  Sample garbage aliases:")
        for alias, eid in samples:
            print(f"    [{eid}] {alias!r}")

    # Count garbage entities
    cur.execute(f"""
        SELECT COUNT(*)
        FROM entities e
        WHERE e.source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
    """, (source_id,))
    garbage_entities = cur.fetchone()[0]
    print(f"\nGarbage entities to delete: {garbage_entities}")

    # Show sample garbage entities
    cur.execute(f"""
        SELECT e.canonical_name, e.id
        FROM entities e
        WHERE e.source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
        ORDER BY length(e.canonical_name) DESC
        LIMIT 20
    """, (source_id,))
    samples = cur.fetchall()
    if samples:
        print("  Sample garbage entities:")
        for name, eid in samples:
            print(f"    [{eid}] {name!r}")

    # Entities with no aliases (orphans) — exclude those whose canonical_name
    # is an alias on another entity (they "are" aliases, not real standalone entities)
    cur.execute("""
        SELECT e.id, e.canonical_name, e.entity_type
        FROM entities e
        WHERE e.source_id = %s
          AND NOT EXISTS (
            SELECT 1 FROM entity_aliases ea
            WHERE ea.entity_id = e.id
          )
        ORDER BY e.id
    """, (source_id,))
    no_alias_entities_raw = cur.fetchall()

    cur.execute(
        "SELECT alias, alias_norm FROM entity_aliases WHERE source_id = %s",
        (source_id,),
    )
    alias_values = cur.fetchall()
    alias_raw_set = {a for a, _ in alias_values if a}
    alias_norm_set = {n for _, n in alias_values if n}

    no_alias_entities = []
    excluded_as_alias = 0
    for eid, name, etype in no_alias_entities_raw:
        if name and (name in alias_raw_set or _normalize_alias_for_db(name) in alias_norm_set):
            excluded_as_alias += 1
            continue
        no_alias_entities.append((eid, name, etype))

    no_alias_count = len(no_alias_entities)
    print(f"\nEntities with no aliases (dry-run: would delete {no_alias_count}):")
    if excluded_as_alias:
        print(f"  (Excluded {excluded_as_alias} whose name is an alias on another entity.)")
    if no_alias_entities:
        for eid, name, etype in no_alias_entities[:30]:
            print(f"    [{eid}] {name!r} ({etype})")
        if no_alias_count > 30:
            print(f"    ... and {no_alias_count - 30} more")
    else:
        print("  (none)")

    if args.dry_run:
        print("\n--dry-run: no changes made.")
        conn.rollback()
        conn.close()
        return

    if garbage_aliases == 0 and garbage_entities == 0:
        print("\nNo garbage found. Nothing to do.")
        conn.close()
        return

    # --confirm: single prompt before any destructive action
    if args.confirm:
        reply = input(f"\nDelete {garbage_aliases} garbage alias(es) and {garbage_entities} garbage entity(ies)? [y/N]: ").strip().lower()
        if reply != "y":
            print("Aborted.")
            conn.close()
            return
        print("Proceeding with bulk delete...")

    # --adjudicate: interactive (d)elete / (s)kip / (e)dit per item, then apply only chosen actions
    if args.adjudicate:
        _run_adjudicate(cur, conn, source_id, args.batch_size, slug=slug)
        conn.close()
        return

    # Fetch garbage entity IDs once for batched steps
    cur.execute(f"""
        SELECT id FROM entities
        WHERE source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
        ORDER BY id
    """, (source_id,))
    garbage_entity_ids: List[int] = [r[0] for r in cur.fetchall()]
    assert len(garbage_entity_ids) == garbage_entities, "entity count mismatch"

    # --delete-garbage-entities: bulk delete with full logging to session JSON (reversible)
    if args.delete_garbage_entities:
        placeholders = ",".join(["%s"] * len(garbage_entity_ids))
        # Fetch full rows before any delete (for restore)
        log_entities = _fetch_table_as_dicts(
            cur, f"SELECT * FROM entities WHERE id IN ({placeholders})", tuple(garbage_entity_ids)
        )
        log_entity_aliases = _fetch_table_as_dicts(
            cur, f"SELECT * FROM entity_aliases WHERE entity_id IN ({placeholders})", tuple(garbage_entity_ids)
        )
        log_entity_citations = _fetch_table_as_dicts(
            cur, f"SELECT * FROM entity_citations WHERE entity_id IN ({placeholders})", tuple(garbage_entity_ids)
        )
        log_entity_links = _fetch_table_as_dicts(
            cur,
            f"SELECT * FROM entity_links WHERE from_entity_id IN ({placeholders}) OR to_entity_id IN ({placeholders})",
            tuple(garbage_entity_ids) + tuple(garbage_entity_ids),
        )
        cur.execute(
            f"SELECT * FROM entity_aliases WHERE source_id = %s AND ({GARBAGE_CONDITION_ALIAS})",
            (source_id,),
        )
        cols = [d[0] for d in cur.description]
        log_garbage_aliases = [
            dict(zip(cols, [_row_to_json(v) for v in row])) for row in cur.fetchall()
        ]

        if not args.confirm:
            reply = input(
                f"Delete {garbage_aliases} garbage alias(es) and {garbage_entities} garbage entity(ies) "
                f"and log to {_session_path(slug)}? [y/N]: "
            ).strip().lower()
            if reply != "y":
                print("Aborted.")
                conn.close()
                return

        run_entry = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "source_id": source_id,
            "deleted_garbage_aliases_count": len(log_garbage_aliases),
            "deleted_entities_count": len(log_entities),
            "deleted_entity_ids": garbage_entity_ids,
            "garbage_aliases": log_garbage_aliases,
            "entities": log_entities,
            "entity_aliases": log_entity_aliases,
            "entity_citations": log_entity_citations,
            "entity_links": log_entity_links,
        }
        log_path = _session_path(slug)
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            log_data.setdefault("garbage_entity_runs", []).append(run_entry)
        else:
            log_data = {"slug": slug, "garbage_entity_runs": [run_entry]}
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"Logged to {log_path}", file=sys.stderr)

        batch_size = args.batch_size
        total_steps = 8
        step = 0
        # Step 1: Delete garbage aliases
        step += 1
        print(f"\n[Step {step}/{total_steps}] Deleting {garbage_aliases} garbage aliases...")
        cur.execute(f"""
            DELETE FROM entity_aliases
            WHERE source_id = %s AND ({GARBAGE_CONDITION_ALIAS})
        """, (source_id,))
        print(f"  -> Deleted {cur.rowcount} alias rows.")
        conn.commit()
        # Step 2: mention_candidates
        step += 1
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'mention_candidates'
                  AND column_name = 'resolved_entity_id'
            )
        """)
        if cur.fetchone()[0] and garbage_entity_ids:
            n_batches = (len(garbage_entity_ids) + batch_size - 1) // batch_size
            print(f"[Step {step}/{total_steps}] Clearing mention_candidates.resolved_entity_id...")
            for batch in _batched(garbage_entity_ids, batch_size):
                ph = ",".join(["%s"] * len(batch))
                cur.execute(f"UPDATE mention_candidates SET resolved_entity_id = NULL WHERE resolved_entity_id IN ({ph})", batch)
                conn.commit()
        else:
            print(f"[Step {step}/{total_steps}] Skipping mention_candidates.")
        # Step 3–7: remaining aliases, citations, links, ocr_variant_clusters, entities
        step += 1
        print(f"[Step {step}/{total_steps}] Deleting remaining aliases of garbage entities...")
        cur.execute(f"DELETE FROM entity_aliases WHERE entity_id IN ({placeholders})", garbage_entity_ids)
        print(f"  -> Deleted {cur.rowcount} alias rows.")
        conn.commit()
        step += 1
        print(f"[Step {step}/{total_steps}] Deleting entity_citations...")
        cur.execute(f"DELETE FROM entity_citations WHERE entity_id IN ({placeholders})", garbage_entity_ids)
        conn.commit()
        step += 1
        print(f"[Step {step}/{total_steps}] Deleting entity_links...")
        cur.execute(
            f"DELETE FROM entity_links WHERE from_entity_id IN ({placeholders}) OR to_entity_id IN ({placeholders})",
            garbage_entity_ids + garbage_entity_ids,
        )
        conn.commit()
        step += 1
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'ocr_variant_clusters'
                  AND column_name = 'canonical_entity_id'
            )
        """)
        if cur.fetchone()[0] and garbage_entity_ids:
            print(f"[Step {step}/{total_steps}] Clearing ocr_variant_clusters.canonical_entity_id...")
            for batch in _batched(garbage_entity_ids, batch_size):
                ph = ",".join(["%s"] * len(batch))
                cur.execute(f"UPDATE ocr_variant_clusters SET canonical_entity_id = NULL WHERE canonical_entity_id IN ({ph})", batch)
                conn.commit()
        else:
            print(f"[Step {step}/{total_steps}] Skipping ocr_variant_clusters.")
        step += 1
        print(f"[Step {step}/{total_steps}] Deleting {garbage_entities} garbage entity rows...")
        cur.execute(f"""
            DELETE FROM entities
            WHERE source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
        """, (source_id,))
        conn.commit()
        print("\nDone! All changes committed. Log written for reversibility.")
        conn.close()
        return

    batch_size = args.batch_size
    total_steps = 8  # aliases, mention_candidates, remaining aliases, citations, links, ocr_variant_clusters, entities
    step = 0

    # Step 1: Delete garbage aliases
    step += 1
    print(f"\n[Step {step}/{total_steps}] Deleting {garbage_aliases} garbage aliases...")
    cur.execute(f"""
        DELETE FROM entity_aliases
        WHERE source_id = %s AND ({GARBAGE_CONDITION_ALIAS})
    """, (source_id,))
    print(f"  -> Deleted {cur.rowcount} alias rows.")
    conn.commit()

    # Step 2: Null out mention_candidates.resolved_entity_id in batches (avoids statement timeout)
    step += 1
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'mention_candidates'
              AND column_name = 'resolved_entity_id'
        )
    """)
    if cur.fetchone()[0] and garbage_entity_ids:
        n_batches = (len(garbage_entity_ids) + batch_size - 1) // batch_size
        print(f"[Step {step}/{total_steps}] Clearing mention_candidates.resolved_entity_id for {len(garbage_entity_ids)} entities (batches of {batch_size})...")
        total_updated = 0
        for i, batch in enumerate(_batched(garbage_entity_ids, batch_size), 1):
            placeholders = ",".join(["%s"] * len(batch))
            cur.execute(
                f"UPDATE mention_candidates SET resolved_entity_id = NULL WHERE resolved_entity_id IN ({placeholders})",
                batch,
            )
            total_updated += cur.rowcount
            print(f"  -> Batch {i}/{n_batches}: {cur.rowcount} rows updated (total: {total_updated})", flush=True)
            conn.commit()
        print(f"  -> Done: {total_updated} mention_candidates rows cleared.")
    else:
        print(f"[Step {step}/{total_steps}] Skipping mention_candidates (table/column missing or no entities).")

    # Step 3: Delete remaining aliases of garbage entities
    step += 1
    print(f"[Step {step}/{total_steps}] Deleting remaining aliases of garbage entities...")
    placeholders = ",".join(["%s"] * len(garbage_entity_ids))
    cur.execute(f"DELETE FROM entity_aliases WHERE entity_id IN ({placeholders})", garbage_entity_ids)
    print(f"  -> Deleted {cur.rowcount} alias rows.")
    conn.commit()

    # Step 4: Delete entity_citations
    step += 1
    print(f"[Step {step}/{total_steps}] Deleting entity_citations...")
    cur.execute(f"DELETE FROM entity_citations WHERE entity_id IN ({placeholders})", garbage_entity_ids)
    print(f"  -> Deleted {cur.rowcount} rows.")
    conn.commit()

    # Step 5: Delete entity_links
    step += 1
    print(f"[Step {step}/{total_steps}] Deleting entity_links...")
    cur.execute(
        f"DELETE FROM entity_links WHERE from_entity_id IN ({placeholders}) OR to_entity_id IN ({placeholders})",
        garbage_entity_ids + garbage_entity_ids,
    )
    print(f"  -> Deleted {cur.rowcount} rows.")
    conn.commit()

    # Step 6: Clear ocr_variant_clusters.canonical_entity_id (FK references entities)
    step += 1
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'ocr_variant_clusters'
              AND column_name = 'canonical_entity_id'
        )
    """)
    if cur.fetchone()[0] and garbage_entity_ids:
        n_batches = (len(garbage_entity_ids) + batch_size - 1) // batch_size
        print(f"[Step {step}/{total_steps}] Clearing ocr_variant_clusters.canonical_entity_id for {len(garbage_entity_ids)} entities (batches of {batch_size})...")
        total_updated = 0
        for i, batch in enumerate(_batched(garbage_entity_ids, batch_size), 1):
            ph = ",".join(["%s"] * len(batch))
            cur.execute(
                f"UPDATE ocr_variant_clusters SET canonical_entity_id = NULL WHERE canonical_entity_id IN ({ph})",
                batch,
            )
            total_updated += cur.rowcount
            print(f"  -> Batch {i}/{n_batches}: {cur.rowcount} rows updated (total: {total_updated})", flush=True)
            conn.commit()
        print(f"  -> Done: {total_updated} ocr_variant_clusters rows cleared.")
    else:
        print(f"[Step {step}/{total_steps}] Skipping ocr_variant_clusters (table/column missing or no entities).")

    # Step 7: Delete the garbage entities themselves
    step += 1
    print(f"[Step {step}/{total_steps}] Deleting {garbage_entities} garbage entity rows...")
    cur.execute(f"""
        DELETE FROM entities
        WHERE source_id = %s AND ({GARBAGE_CONDITION_ENTITY})
    """, (source_id,))
    print(f"  -> Deleted {cur.rowcount} entity rows.")
    conn.commit()

    print("\nDone! All changes committed.")

    # Final counts
    cur.execute("SELECT COUNT(*) FROM entities e WHERE e.source_id = %s", (source_id,))
    print(f"Remaining entities: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM entity_aliases ea WHERE ea.source_id = %s", (source_id,))
    print(f"Remaining aliases:  {cur.fetchone()[0]}")

    conn.close()


if __name__ == "__main__":
    main()
