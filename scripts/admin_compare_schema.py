#!/usr/bin/env python3
"""
admin_compare_schema.py

Compare two PostgreSQL schemas to ensure a freshly migrated database matches a reference database.

This is intentionally deeper than admin_validate_integrity.py: it compares *schema objects*,
not row-level invariants.

What it compares (by default, schema = public):
- Extensions (name + version)
- Tables (including presence/absence)
- Columns (type, nullability, default)
- Constraints (via pg_get_constraintdef)
- Indexes (compares normalized index definitions; ignores index *names*)
- Views + materialized views (definition text)

Usage examples:
  # Compare current DB vs reference DB
  export DATABASE_URL='postgresql://neh:neh@localhost:5432/friday_clean'
  export DATABASE_URL_REF='postgresql://neh:neh@localhost:5432/neh'
  python scripts/admin_compare_schema.py

  # JSON output (useful for CI)
  python scripts/admin_compare_schema.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import psycopg2


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _normalize_indexdef(indexdef: str) -> str:
    """
    Normalize a CREATE INDEX statement so we can compare semantically without depending on index names.
    Example:
      CREATE INDEX idx_chunks_embedding_ivfflat ON public.chunks USING ivfflat (...)
    becomes:
      CREATE INDEX ON public.chunks USING ivfflat (...)
    """
    s = _norm_ws(indexdef)
    # Normalize "CREATE UNIQUE INDEX name ON" and "CREATE INDEX name ON"
    s = re.sub(r"^CREATE\s+UNIQUE\s+INDEX\s+\S+\s+ON\s+", "CREATE UNIQUE INDEX ON ", s, flags=re.IGNORECASE)
    s = re.sub(r"^CREATE\s+INDEX\s+\S+\s+ON\s+", "CREATE INDEX ON ", s, flags=re.IGNORECASE)
    return s


@dataclass(frozen=True)
class SchemaSnapshot:
    schema: str
    extensions: Dict[str, str]  # extname -> extversion
    tables: Set[str]
    columns: Dict[str, Dict[str, Dict[str, Any]]]  # table -> column -> properties
    constraints: Dict[str, Set[str]]  # table -> {normalized constraint def strings}
    indexes: Dict[str, Set[str]]  # table -> {normalized index definitions}
    views: Dict[str, str]  # viewname -> definition
    matviews: Dict[str, str]  # matviewname -> definition


def _fetchall_map(cur, sql: str, params: Tuple[Any, ...]) -> List[Tuple[Any, ...]]:
    cur.execute(sql, params)
    return cur.fetchall()


def snapshot_schema(conn, *, schema: str = "public") -> SchemaSnapshot:
    with conn.cursor() as cur:
        # Extensions (cluster-wide, but we still compare because migrations depend on them)
        extensions: Dict[str, str] = {}
        for extname, extversion in _fetchall_map(
            cur,
            "SELECT extname, extversion FROM pg_extension ORDER BY extname;",
            (),
        ):
            extensions[str(extname)] = str(extversion)

        # Tables
        tables = {
            r[0]
            for r in _fetchall_map(
                cur,
                """
                SELECT c.relname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = %s
                  AND c.relkind = 'r'
                ORDER BY c.relname;
                """,
                (schema,),
            )
        }

        # Columns (information_schema is good enough here; keeps output stable)
        columns: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for (
            table_name,
            column_name,
            data_type,
            udt_name,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
        ) in _fetchall_map(
            cur,
            """
            SELECT
              table_name,
              column_name,
              data_type,
              udt_name,
              is_nullable,
              column_default,
              character_maximum_length,
              numeric_precision,
              numeric_scale
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position;
            """,
            (schema,),
        ):
            t = str(table_name)
            c = str(column_name)
            columns.setdefault(t, {})[c] = {
                "data_type": str(data_type) if data_type is not None else None,
                "udt_name": str(udt_name) if udt_name is not None else None,
                "is_nullable": str(is_nullable),
                "default": _norm_ws(str(column_default)) if column_default is not None else None,
                "char_len": int(character_maximum_length) if character_maximum_length is not None else None,
                "num_precision": int(numeric_precision) if numeric_precision is not None else None,
                "num_scale": int(numeric_scale) if numeric_scale is not None else None,
            }

        # Constraints (includes PK/UK/FK/CHECK)
        constraints: Dict[str, Set[str]] = {}
        for table_name, condef in _fetchall_map(
            cur,
            """
            SELECT
              rel.relname AS table_name,
              pg_get_constraintdef(con.oid, true) AS condef
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = rel.relnamespace
            WHERE n.nspname = %s
              AND rel.relkind = 'r'
            ORDER BY rel.relname, con.conname;
            """,
            (schema,),
        ):
            t = str(table_name)
            constraints.setdefault(t, set()).add(_norm_ws(str(condef)))

        # Indexes (ignore index names)
        indexes: Dict[str, Set[str]] = {}
        for tablename, indexdef in _fetchall_map(
            cur,
            """
            SELECT tablename, indexdef
            FROM pg_indexes
            WHERE schemaname = %s
            ORDER BY tablename, indexname;
            """,
            (schema,),
        ):
            t = str(tablename)
            indexes.setdefault(t, set()).add(_normalize_indexdef(str(indexdef)))

        # Views + matviews
        views: Dict[str, str] = {}
        for viewname, definition in _fetchall_map(
            cur,
            """
            SELECT viewname, definition
            FROM pg_views
            WHERE schemaname = %s
            ORDER BY viewname;
            """,
            (schema,),
        ):
            views[str(viewname)] = _norm_ws(str(definition))

        matviews: Dict[str, str] = {}
        for matviewname, definition in _fetchall_map(
            cur,
            """
            SELECT matviewname, definition
            FROM pg_matviews
            WHERE schemaname = %s
            ORDER BY matviewname;
            """,
            (schema,),
        ):
            matviews[str(matviewname)] = _norm_ws(str(definition))

    return SchemaSnapshot(
        schema=schema,
        extensions=extensions,
        tables=tables,
        columns=columns,
        constraints=constraints,
        indexes=indexes,
        views=views,
        matviews=matviews,
    )


def _diff_sets(a: Set[str], b: Set[str]) -> Dict[str, List[str]]:
    return {
        "only_in_a": sorted(a - b),
        "only_in_b": sorted(b - a),
    }


def compare_snapshots(a: SchemaSnapshot, b: SchemaSnapshot) -> Dict[str, Any]:
    """
    Return a structured diff. "a" is the target DB, "b" is the reference DB.
    We expect "only_in_a" to usually be empty and "only_in_b" to indicate what's missing from a.
    """
    diff: Dict[str, Any] = {}

    # Extensions
    a_ext = set(f"{k}={v}" for k, v in a.extensions.items())
    b_ext = set(f"{k}={v}" for k, v in b.extensions.items())
    diff["extensions"] = _diff_sets(a_ext, b_ext)

    # Tables
    diff["tables"] = _diff_sets(a.tables, b.tables)

    # Columns
    col_diff: Dict[str, Any] = {"tables_with_differences": {}}
    all_tables = sorted(set(a.columns.keys()) | set(b.columns.keys()))
    for t in all_tables:
        a_cols = a.columns.get(t, {})
        b_cols = b.columns.get(t, {})
        if set(a_cols.keys()) != set(b_cols.keys()):
            col_diff["tables_with_differences"][t] = {
                "columns": _diff_sets(set(a_cols.keys()), set(b_cols.keys()))
            }
            continue
        # Same column names; compare properties
        prop_mismatches = {}
        for c in sorted(a_cols.keys()):
            if a_cols[c] != b_cols[c]:
                prop_mismatches[c] = {"a": a_cols[c], "b": b_cols[c]}
        if prop_mismatches:
            col_diff["tables_with_differences"][t] = {"property_mismatches": prop_mismatches}
    diff["columns"] = col_diff

    # Constraints
    con_diff: Dict[str, Any] = {"tables_with_differences": {}}
    for t in sorted(set(a.constraints.keys()) | set(b.constraints.keys())):
        a_cons = a.constraints.get(t, set())
        b_cons = b.constraints.get(t, set())
        if a_cons != b_cons:
            con_diff["tables_with_differences"][t] = _diff_sets(a_cons, b_cons)
    diff["constraints"] = con_diff

    # Indexes
    idx_diff: Dict[str, Any] = {"tables_with_differences": {}}
    for t in sorted(set(a.indexes.keys()) | set(b.indexes.keys())):
        a_idx = a.indexes.get(t, set())
        b_idx = b.indexes.get(t, set())
        if a_idx != b_idx:
            idx_diff["tables_with_differences"][t] = _diff_sets(a_idx, b_idx)
    diff["indexes"] = idx_diff

    # Views
    view_diff: Dict[str, Any] = {"only_in_a": [], "only_in_b": [], "definition_mismatches": {}}
    a_view_names = set(a.views.keys())
    b_view_names = set(b.views.keys())
    view_diff["only_in_a"] = sorted(a_view_names - b_view_names)
    view_diff["only_in_b"] = sorted(b_view_names - a_view_names)
    for v in sorted(a_view_names & b_view_names):
        if a.views[v] != b.views[v]:
            view_diff["definition_mismatches"][v] = {"a": a.views[v], "b": b.views[v]}
    diff["views"] = view_diff

    # Matviews
    mv_diff: Dict[str, Any] = {"only_in_a": [], "only_in_b": [], "definition_mismatches": {}}
    a_mv_names = set(a.matviews.keys())
    b_mv_names = set(b.matviews.keys())
    mv_diff["only_in_a"] = sorted(a_mv_names - b_mv_names)
    mv_diff["only_in_b"] = sorted(b_mv_names - a_mv_names)
    for v in sorted(a_mv_names & b_mv_names):
        if a.matviews[v] != b.matviews[v]:
            mv_diff["definition_mismatches"][v] = {"a": a.matviews[v], "b": b.matviews[v]}
    diff["matviews"] = mv_diff

    # Quick pass/fail
    def _has_differences(d: Any) -> bool:
        if isinstance(d, dict):
            return any(_has_differences(v) for v in d.values())
        if isinstance(d, list):
            return len(d) > 0
        return False

    diff["passed"] = not _has_differences(
        {
            "extensions": diff["extensions"],
            "tables": diff["tables"],
            "columns": diff["columns"]["tables_with_differences"],
            "constraints": diff["constraints"]["tables_with_differences"],
            "indexes": diff["indexes"]["tables_with_differences"],
            "views": {
                "only_in_a": diff["views"]["only_in_a"],
                "only_in_b": diff["views"]["only_in_b"],
                "definition_mismatches": diff["views"]["definition_mismatches"],
            },
            "matviews": {
                "only_in_a": diff["matviews"]["only_in_a"],
                "only_in_b": diff["matviews"]["only_in_b"],
                "definition_mismatches": diff["matviews"]["definition_mismatches"],
            },
        }
    )

    return diff


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two PostgreSQL schemas (target vs reference).")
    ap.add_argument("--schema", default="public", help="Schema name to compare (default: public)")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON diff")
    ap.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Target DATABASE_URL (defaults to env DATABASE_URL)",
    )
    ap.add_argument(
        "--database-url-ref",
        default=os.getenv("DATABASE_URL_REF"),
        help="Reference DATABASE_URL (defaults to env DATABASE_URL_REF)",
    )
    args = ap.parse_args()

    if not args.database_url or not args.database_url_ref:
        print("ERROR: Provide --database-url and --database-url-ref (or set DATABASE_URL and DATABASE_URL_REF).")
        sys.exit(2)

    conn_a = psycopg2.connect(args.database_url)
    conn_b = psycopg2.connect(args.database_url_ref)
    try:
        snap_a = snapshot_schema(conn_a, schema=args.schema)
        snap_b = snapshot_schema(conn_b, schema=args.schema)
        diff = compare_snapshots(snap_a, snap_b)

        if args.json:
            print(json.dumps(diff, indent=2, default=str))
        else:
            status = "✅ PASS" if diff["passed"] else "❌ FAIL"
            print(f"{status}: schema diff ({args.schema})")
            print("Note: 'a' = target, 'b' = reference. For your use-case, 'only_in_b' means missing from target.")

            def _print_section(name: str, payload: Any) -> None:
                if isinstance(payload, dict) and payload.get("only_in_a") == [] and payload.get("only_in_b") == []:
                    return
                print(f"\n=== {name} ===")
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        if isinstance(v, list) and not v:
                            continue
                        if isinstance(v, dict) and not v:
                            continue
                        print(f"- {k}: {v}")
                else:
                    print(payload)

            _print_section("extensions", diff["extensions"])
            _print_section("tables", diff["tables"])
            _print_section("views", diff["views"])
            _print_section("matviews", diff["matviews"])

            if diff["columns"]["tables_with_differences"]:
                print("\n=== columns: tables_with_differences ===")
                for t, d in diff["columns"]["tables_with_differences"].items():
                    print(f"- {t}: {list(d.keys())}")

            if diff["constraints"]["tables_with_differences"]:
                print("\n=== constraints: tables_with_differences ===")
                for t, d in diff["constraints"]["tables_with_differences"].items():
                    print(f"- {t}: only_in_a={len(d['only_in_a'])}, only_in_b={len(d['only_in_b'])}")

            if diff["indexes"]["tables_with_differences"]:
                print("\n=== indexes: tables_with_differences ===")
                for t, d in diff["indexes"]["tables_with_differences"].items():
                    print(f"- {t}: only_in_a={len(d['only_in_a'])}, only_in_b={len(d['only_in_b'])}")

        sys.exit(0 if diff["passed"] else 1)
    finally:
        conn_a.close()
        conn_b.close()


if __name__ == "__main__":
    main()

