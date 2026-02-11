#!/usr/bin/env python3
"""
Reversible alias cleanup: remove uninformative alias links only.

Removes:
1. Links where alias_norm == canonical_name (case-insensitive) — redundant
2. Links where alias_norm contains "undeciphered name" — uninformative placeholder

Example: entity CABIN with alias "cabin" — redundant, delete.
Example: entity OSS with alias "cabin" — informative, KEEP.
Example: alias "undeciphered name no" — garbage, delete.

Logs all deleted rows to JSON for reversibility. Use --restore to re-insert.

Usage:
    python scripts/cleanup_aliases.py --slug vassiliev_venona_index_20260210 [--dry-run]
    python scripts/cleanup_aliases.py --slug vassiliev_venona_index_20260210 --confirm
    python scripts/cleanup_aliases.py --restore alias_cleanup_vassiliev_venona_index_20260210.json --confirm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2
from psycopg2 import extras as psycopg2_extras

def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (match alias_norm)."""
    if not s:
        return ""
    t = (s or "").lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _row_to_json(obj) -> object:
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
    if hasattr(obj, "__float__") and not isinstance(obj, bool):
        return float(obj)
    return str(obj)


def _session_path(slug: str) -> str:
    return f"alias_cleanup_{slug}.json"


def _find_aliases_to_delete(cur, source_id: int) -> list[dict]:
    """
    Find entity_aliases to delete: redundant canonical or "undeciphered name" placeholders.
    Per-entity: we only delete cabin->CABIN, never cabin->OSS.
    Returns list of full row dicts for logging.
    """
    cur.execute("""
        SELECT ea.*, e.canonical_name
        FROM entity_aliases ea
        JOIN entities e ON e.id = ea.entity_id
        WHERE ea.source_id = %s
        ORDER BY ea.entity_id, ea.id
    """, (source_id,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, [_row_to_json(v) for v in row])) for row in cur.fetchall()]

    to_delete: list[dict] = []
    canonical_norm_cache: dict[str, str] = {}

    for row in rows:
        canonical_name = row.get("canonical_name") or ""
        canonical_norm = canonical_norm_cache.get(canonical_name)
        if canonical_norm is None:
            canonical_norm = _normalize(canonical_name)
            canonical_norm_cache[canonical_name] = canonical_norm

        alias_norm = (row.get("alias_norm") or "").lower()
        if not alias_norm:
            continue

        # 1. Redundant: alias_norm == canonical_name (case-insensitive) for THIS entity
        if alias_norm == canonical_norm:
            to_delete.append(row)
            continue

        # 2. Uninformative placeholder: "undeciphered name" (no, n, n. 20, etc.)
        if "undeciphered name" in alias_norm:
            to_delete.append(row)

    return to_delete


def _adapt_val(val):
    """Convert JSON-loaded values for psycopg2 (dict -> Json for JSONB)."""
    if isinstance(val, dict):
        return psycopg2_extras.Json(val)
    return val


def _restore_from_session(cur, conn, path: str) -> None:
    """Re-insert aliases from session JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    aliases = data.get("deleted_aliases", [])
    if not aliases:
        print("  No deleted_aliases in session.")
        return

    cols = list(aliases[0].keys())
    cols_str = ", ".join(f'"{c}"' for c in cols)
    template = "(" + ", ".join(["%s"] * len(cols)) + ")"
    sql = f"INSERT INTO entity_aliases ({cols_str}) VALUES %s ON CONFLICT (id) DO NOTHING"

    values = [tuple(_adapt_val(row.get(c)) for c in cols) for row in aliases]
    psycopg2_extras.execute_values(cur, sql, values, template=template)
    conn.commit()
    print(f"  Restored {cur.rowcount} alias(es).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reversible alias cleanup (uninformative links only: alias==canonical)")
    parser.add_argument("--db", default=None, help="Database URL")
    parser.add_argument("--slug", default=None, help="Concordance source slug")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't delete")
    parser.add_argument("--confirm", action="store_true", help="Ask before delete")
    parser.add_argument("--restore", metavar="PATH", default=None,
                        help="Restore from session JSON (re-insert deleted aliases)")
    args = parser.parse_args()

    db_url = args.db or os.environ.get(
        "DATABASE_URL", "postgresql://neh:neh@localhost:5432/neh"
    )
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    if args.restore:
        path = os.path.abspath(args.restore)
        if not os.path.isfile(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        if args.confirm:
            reply = input("Restore deleted aliases to database? [y/N]: ").strip().lower()
            if reply != "y":
                print("Aborted.")
                conn.close()
                return
        _restore_from_session(cur, conn, path)
        conn.close()
        return

    if not args.slug:
        print("ERROR: --slug required (or use --restore)")
        sys.exit(1)

    cur.execute("SELECT id FROM concordance_sources WHERE slug = %s", (args.slug,))
    row = cur.fetchone()
    if not row:
        print(f"ERROR: No concordance_source with slug={args.slug}")
        sys.exit(1)
    source_id = row[0]

    to_delete = _find_aliases_to_delete(cur, source_id)
    if not to_delete:
        print("No uninformative alias links to delete.")
        conn.close()
        return

    # Strip canonical_name from row dicts (not a column in entity_aliases)
    for r in to_delete:
        r.pop("canonical_name", None)

    print(f"Found {len(to_delete)} alias link(s) to delete (redundant canonical or undeciphered name).")
    if args.dry_run:
        for r in to_delete[:10]:
            print(f"  [{r['id']}] entity_id={r['entity_id']} alias={r['alias']!r} alias_norm={r['alias_norm']!r}")
        if len(to_delete) > 10:
            print(f"  ... and {len(to_delete) - 10} more")
        conn.close()
        return

    if args.confirm:
        reply = input(f"Delete {len(to_delete)} alias(es) and log to {_session_path(args.slug)}? [y/N]: ").strip().lower()
        if reply != "y":
            print("Aborted.")
            conn.close()
            return

    session_path = REPO_ROOT / _session_path(args.slug)
    session = {
        "slug": args.slug,
        "source_id": source_id,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "deleted_count": len(to_delete),
        "deleted_aliases": to_delete,
    }
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)

    ids = [r["id"] for r in to_delete]
    placeholders = ",".join(["%s"] * len(ids))
    cur.execute(f"DELETE FROM entity_aliases WHERE id IN ({placeholders})", ids)
    conn.commit()
    print(f"Deleted {cur.rowcount} alias(es). Logged to {session_path}")
    conn.close()


if __name__ == "__main__":
    main()
