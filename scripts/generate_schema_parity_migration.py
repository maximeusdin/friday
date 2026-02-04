#!/usr/bin/env python3
"""
generate_schema_parity_migration.py

Compare local (reference) and RDS (target) PostgreSQL schemas, then generate
SQL migration statements to bring RDS in line with local.

Compares:
- Tables (CREATE TABLE for missing tables)
- Columns (ALTER TABLE ADD COLUMN for missing columns)
- Indexes (CREATE INDEX IF NOT EXISTS for missing indexes)
- Constraints (ALTER TABLE ADD CONSTRAINT for missing constraints)

Usage:
  # Using environment variables
  export DATABASE_URL="postgresql://friday:PASS@rds-host:5432/friday?sslmode=..."
  export DATABASE_URL_REF="postgresql://neh:neh@localhost:5432/neh?sslmode=disable"
  
  python scripts/generate_schema_parity_migration.py
  
  # Or with explicit URLs
  python scripts/generate_schema_parity_migration.py \
    --target "$DATABASE_URL" \
    --reference "postgresql://neh:neh@localhost:5432/neh?sslmode=disable"
  
  # Output to migration file
  python scripts/generate_schema_parity_migration.py --output migrations/0053_rds_schema_parity.sql
  
  # Filter to specific tables
  python scripts/generate_schema_parity_migration.py --tables chunks,entities,entity_aliases
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    udt_name: str
    is_nullable: str
    column_default: Optional[str]
    char_len: Optional[int]
    num_precision: Optional[int]
    num_scale: Optional[int]


@dataclass
class IndexInfo:
    name: str
    definition: str
    table_name: str


@dataclass
class ConstraintInfo:
    name: str
    definition: str
    table_name: str


@dataclass
class SchemaSnapshot:
    tables: Set[str]
    columns: Dict[str, Dict[str, ColumnInfo]]  # table -> column_name -> info
    indexes: Dict[str, List[IndexInfo]]  # table -> list of indexes
    constraints: Dict[str, List[ConstraintInfo]]  # table -> list of constraints


def get_tables(conn, schema: str = "public") -> Set[str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relkind = 'r'
            ORDER BY c.relname
        """, (schema,))
        return {row[0] for row in cur.fetchall()}


def get_columns(conn, schema: str = "public") -> Dict[str, Dict[str, ColumnInfo]]:
    columns: Dict[str, Dict[str, ColumnInfo]] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                table_name, column_name, data_type, udt_name,
                is_nullable, column_default,
                character_maximum_length, numeric_precision, numeric_scale
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position
        """, (schema,))
        for row in cur.fetchall():
            table = row[0]
            col = ColumnInfo(
                name=row[1],
                data_type=row[2],
                udt_name=row[3],
                is_nullable=row[4],
                column_default=row[5],
                char_len=row[6],
                num_precision=row[7],
                num_scale=row[8],
            )
            columns.setdefault(table, {})[col.name] = col
    return columns


def get_indexes(conn, schema: str = "public") -> Dict[str, List[IndexInfo]]:
    indexes: Dict[str, List[IndexInfo]] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = %s
            ORDER BY tablename, indexname
        """, (schema,))
        for row in cur.fetchall():
            idx = IndexInfo(name=row[1], definition=row[2], table_name=row[0])
            indexes.setdefault(row[0], []).append(idx)
    return indexes


def get_constraints(conn, schema: str = "public") -> Dict[str, List[ConstraintInfo]]:
    constraints: Dict[str, List[ConstraintInfo]] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                rel.relname AS table_name,
                con.conname AS constraint_name,
                pg_get_constraintdef(con.oid, true) AS condef
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = rel.relnamespace
            WHERE n.nspname = %s AND rel.relkind = 'r'
            ORDER BY rel.relname, con.conname
        """, (schema,))
        for row in cur.fetchall():
            con = ConstraintInfo(name=row[1], definition=row[2], table_name=row[0])
            constraints.setdefault(row[0], []).append(con)
    return constraints


def snapshot_schema(conn, schema: str = "public") -> SchemaSnapshot:
    return SchemaSnapshot(
        tables=get_tables(conn, schema),
        columns=get_columns(conn, schema),
        indexes=get_indexes(conn, schema),
        constraints=get_constraints(conn, schema),
    )


def normalize_indexdef(indexdef: str) -> str:
    """Normalize index definition for comparison (remove index name)."""
    s = re.sub(r"\s+", " ", indexdef.strip())
    s = re.sub(r"^CREATE\s+UNIQUE\s+INDEX\s+\S+\s+ON\s+", "CREATE UNIQUE INDEX ON ", s, flags=re.IGNORECASE)
    s = re.sub(r"^CREATE\s+INDEX\s+\S+\s+ON\s+", "CREATE INDEX ON ", s, flags=re.IGNORECASE)
    return s


def normalize_constraint(condef: str) -> str:
    """Normalize constraint definition for comparison."""
    return re.sub(r"\s+", " ", condef.strip())


def generate_column_sql(table: str, col: ColumnInfo) -> str:
    """Generate ALTER TABLE ADD COLUMN statement."""
    type_str = col.udt_name
    if col.udt_name == "varchar" and col.char_len:
        type_str = f"VARCHAR({col.char_len})"
    elif col.udt_name == "numeric" and col.num_precision:
        if col.num_scale:
            type_str = f"NUMERIC({col.num_precision},{col.num_scale})"
        else:
            type_str = f"NUMERIC({col.num_precision})"
    elif col.udt_name in ("int4", "int8", "float8", "bool", "text", "jsonb", "timestamptz"):
        type_map = {
            "int4": "INTEGER",
            "int8": "BIGINT",
            "float8": "DOUBLE PRECISION",
            "bool": "BOOLEAN",
            "text": "TEXT",
            "jsonb": "JSONB",
            "timestamptz": "TIMESTAMPTZ",
        }
        type_str = type_map.get(col.udt_name, col.udt_name.upper())
    elif col.udt_name.startswith("_"):
        # Array type
        base = col.udt_name[1:]
        type_str = f"{base.upper()}[]"
    else:
        type_str = col.udt_name.upper()
    
    nullable = "" if col.is_nullable == "YES" else " NOT NULL"
    default = f" DEFAULT {col.column_default}" if col.column_default else ""
    
    return f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col.name} {type_str}{nullable}{default};"


def generate_migration(
    target: SchemaSnapshot,
    reference: SchemaSnapshot,
    tables_filter: Optional[Set[str]] = None,
) -> List[str]:
    """
    Generate SQL statements to bring target schema in line with reference.
    Returns list of SQL statements.
    """
    statements: List[str] = []
    
    # Filter tables if specified
    all_tables = reference.tables
    if tables_filter:
        all_tables = all_tables & tables_filter
    
    # 1. Missing tables
    missing_tables = all_tables - target.tables
    if missing_tables:
        statements.append("-- ============================================================================")
        statements.append("-- MISSING TABLES (manual review required - copying table structure)")
        statements.append("-- ============================================================================")
        for table in sorted(missing_tables):
            statements.append(f"-- TODO: CREATE TABLE {table} - copy structure from reference database")
        statements.append("")
    
    # 2. Missing columns
    columns_section = []
    for table in sorted(all_tables & target.tables):
        ref_cols = reference.columns.get(table, {})
        tgt_cols = target.columns.get(table, {})
        
        missing_cols = set(ref_cols.keys()) - set(tgt_cols.keys())
        if missing_cols:
            columns_section.append(f"-- {table}")
            for col_name in sorted(missing_cols):
                col = ref_cols[col_name]
                columns_section.append(generate_column_sql(table, col))
            columns_section.append("")
    
    if columns_section:
        statements.append("-- ============================================================================")
        statements.append("-- MISSING COLUMNS")
        statements.append("-- ============================================================================")
        statements.extend(columns_section)
    
    # 3. Missing indexes
    indexes_section = []
    for table in sorted(all_tables & target.tables):
        ref_idxs = reference.indexes.get(table, [])
        tgt_idxs = target.indexes.get(table, [])
        
        # Normalize for comparison
        tgt_normalized = {normalize_indexdef(idx.definition) for idx in tgt_idxs}
        
        for idx in ref_idxs:
            norm = normalize_indexdef(idx.definition)
            if norm not in tgt_normalized:
                # Rewrite to use IF NOT EXISTS
                defn = idx.definition
                if "IF NOT EXISTS" not in defn.upper():
                    defn = re.sub(
                        r"^CREATE\s+(UNIQUE\s+)?INDEX\s+",
                        r"CREATE \1INDEX IF NOT EXISTS ",
                        defn,
                        flags=re.IGNORECASE
                    )
                indexes_section.append(f"-- {table}.{idx.name}")
                indexes_section.append(f"{defn};")
                indexes_section.append("")
    
    if indexes_section:
        statements.append("-- ============================================================================")
        statements.append("-- MISSING INDEXES")
        statements.append("-- ============================================================================")
        statements.extend(indexes_section)
    
    # 4. Missing constraints (excluding primary keys which are usually created with table)
    constraints_section = []
    for table in sorted(all_tables & target.tables):
        ref_cons = reference.constraints.get(table, [])
        tgt_cons = target.constraints.get(table, [])
        
        # Normalize for comparison
        tgt_normalized = {normalize_constraint(con.definition) for con in tgt_cons}
        
        for con in ref_cons:
            norm = normalize_constraint(con.definition)
            if norm not in tgt_normalized:
                # Skip primary keys (usually created with table)
                if "PRIMARY KEY" in norm.upper():
                    continue
                constraints_section.append(f"-- {table}.{con.name}")
                # Use DO block with check for idempotent constraint creation
                # For UNIQUE constraints, we need to drop any existing index with same name first
                is_unique = "UNIQUE" in norm.upper()
                if is_unique:
                    constraints_section.append(f"""DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = '{con.name}'
    ) THEN
        DROP INDEX IF EXISTS {con.name};
        ALTER TABLE {table} ADD CONSTRAINT {con.name} {con.definition};
        RAISE NOTICE 'Created constraint {con.name}';
    ELSE
        RAISE NOTICE 'Constraint {con.name} already exists, skipping';
    END IF;
END $$;""")
                else:
                    constraints_section.append(f"""DO $$
BEGIN
    ALTER TABLE {table} ADD CONSTRAINT {con.name} {con.definition};
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Constraint {con.name} already exists, skipping';
END $$;""")
                constraints_section.append("")
    
    if constraints_section:
        statements.append("-- ============================================================================")
        statements.append("-- MISSING CONSTRAINTS (idempotent - safe to re-run)")
        statements.append("-- ============================================================================")
        statements.extend(constraints_section)
    
    return statements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SQL migration to bring target DB schema in line with reference"
    )
    parser.add_argument(
        "--target",
        default=os.getenv("DATABASE_URL"),
        help="Target database URL (RDS) - default: DATABASE_URL env var",
    )
    parser.add_argument(
        "--reference",
        default=os.getenv("DATABASE_URL_REF"),
        help="Reference database URL (local) - default: DATABASE_URL_REF env var",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--tables",
        default=None,
        help="Comma-separated list of tables to compare (default: all)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting",
    )
    
    args = parser.parse_args()
    
    if not args.target:
        print("ERROR: Target database URL required (--target or DATABASE_URL env var)")
        sys.exit(2)
    
    if not args.reference:
        print("ERROR: Reference database URL required (--reference or DATABASE_URL_REF env var)")
        sys.exit(2)
    
    tables_filter = None
    if args.tables:
        tables_filter = set(t.strip() for t in args.tables.split(","))
    
    # Mask passwords in display
    def mask_url(url: str) -> str:
        return re.sub(r":([^:@]+)@", ":***@", url)
    
    print(f"Target (RDS):    {mask_url(args.target)}", file=sys.stderr)
    print(f"Reference (local): {mask_url(args.reference)}", file=sys.stderr)
    if tables_filter:
        print(f"Tables filter: {', '.join(sorted(tables_filter))}", file=sys.stderr)
    print(file=sys.stderr)
    
    try:
        conn_target = psycopg2.connect(args.target)
        conn_ref = psycopg2.connect(args.reference)
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        print("Snapshotting target schema...", file=sys.stderr)
        target_snap = snapshot_schema(conn_target)
        
        print("Snapshotting reference schema...", file=sys.stderr)
        ref_snap = snapshot_schema(conn_ref)
        
        print("Generating migration SQL...", file=sys.stderr)
        statements = generate_migration(target_snap, ref_snap, tables_filter)
        
        if not statements:
            print("No schema differences found!", file=sys.stderr)
            sys.exit(0)
        
        # Build output
        header = [
            f"-- Migration: RDS Schema Parity",
            f"-- Generated: {datetime.now().isoformat()}",
            f"-- Brings target DB schema in line with reference",
            f"--",
            f"-- Target: {mask_url(args.target)}",
            f"-- Reference: {mask_url(args.reference)}",
            f"--",
            f"-- Run with: psql \"$DATABASE_URL\" -f <this_file>",
            f"",
        ]
        
        output = "\n".join(header + statements)
        
        if args.output:
            mode = "a" if args.append else "w"
            with open(args.output, mode, encoding="utf-8") as f:
                if args.append:
                    f.write("\n\n-- ============================================================================\n")
                    f.write(f"-- APPENDED: {datetime.now().isoformat()}\n")
                    f.write("-- ============================================================================\n\n")
                f.write(output)
            print(f"Written to: {args.output}", file=sys.stderr)
        else:
            print(output)
        
        # Summary
        print(file=sys.stderr)
        print(f"Generated {len([s for s in statements if s.strip() and not s.startswith('--')])} SQL statements", file=sys.stderr)
        
    finally:
        conn_target.close()
        conn_ref.close()


if __name__ == "__main__":
    main()
