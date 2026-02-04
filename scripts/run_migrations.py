#!/usr/bin/env python3
"""
scripts/run_migrations.py

Apply this repo's database schema + migrations to a target Postgres database.

Design goals:
- Use the *same ordering* as `make all-migrations` (important for dependencies).
- Run with `psql -v ON_ERROR_STOP=1 -X` for faithful execution of SQL files.
- Automatically append any `migrations/*.sql` not referenced by all-migrations
  (e.g. `0041_schema_parity_with_neh.sql`) so AWS runs don't miss new files.

Usage:
  export DATABASE_URL='postgresql://user:pass@host:5432/dbname?sslmode=require'
  python scripts/run_migrations.py

  # Print what would run, without executing:
  python scripts/run_migrations.py --plan
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Step:
    kind: str  # "file" | "sql"
    label: str  # human-friendly description
    target: str  # Makefile target name (or "extra-migrations")
    payload: str  # file path (relative) or SQL string


def _repo_root() -> Path:
    # scripts/ -> repo root
    return Path(__file__).resolve().parents[1]


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _parse_all_migrations_targets(makefile_text: str) -> List[str]:
    """
    Parse the `all-migrations:` dependency list, including backslash continuations.
    """
    lines = makefile_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("all-migrations:"):
            start_idx = i
            break
    if start_idx is None:
        raise RuntimeError("Could not find `all-migrations:` in Makefile.")

    buf = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        if i != start_idx and re.match(r"^[A-Za-z0-9_.-]+:\s*", line):
            break
        # strip comments
        line = line.split("#", 1)[0].rstrip()
        if line:
            buf.append(line)
        if not line.endswith("\\"):
            break
        i += 1

    joined = " ".join(s.rstrip("\\").strip() for s in buf).strip()
    # "all-migrations: dep1 dep2 dep3"
    _, deps = joined.split(":", 1)
    targets = [t for t in deps.split() if t.strip()]
    if not targets:
        raise RuntimeError("Parsed empty dependency list for `all-migrations:`.")
    return targets


def _parse_makefile_recipes(makefile_text: str) -> Dict[str, List[str]]:
    """
    Return mapping: target -> list of recipe lines (tab-indented).
    """
    recipes: Dict[str, List[str]] = {}
    current_target: Optional[str] = None

    for raw in makefile_text.splitlines():
        line = raw.rstrip("\n")

        # New target line (ignore special directives like .PHONY)
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:(.*)$", line)
        if m and not line.startswith("."):
            current_target = m.group(1)
            recipes.setdefault(current_target, [])
            continue

        if current_target and line.startswith("\t"):
            recipes[current_target].append(line.lstrip("\t"))

    return recipes


def _extract_steps_for_target(target: str, recipe_lines: Sequence[str]) -> List[Step]:
    steps: List[Step] = []

    # We intentionally only extract:
    # - SQL files piped into psql: "< path/to/file.sql"
    # - Inline psql commands: -c "SQL..."
    file_re = re.compile(r"<\s*([^\s]+\.sql)\b")
    c_re = re.compile(r'-c\s+"([^"]*)"')
    c_re2 = re.compile(r"-c\s+'([^']*)'")

    for line in recipe_lines:
        if line.strip().startswith("@echo"):
            continue

        for m in file_re.finditer(line):
            rel = m.group(1)
            steps.append(
                Step(
                    kind="file",
                    label=f"Apply {rel}",
                    target=target,
                    payload=rel,
                )
            )

        m = c_re.search(line) or c_re2.search(line)
        if m:
            sql = m.group(1).strip()
            if sql:
                steps.append(
                    Step(
                        kind="sql",
                        label=f"Run SQL: {sql[:60]}{'...' if len(sql) > 60 else ''}",
                        target=target,
                        payload=sql,
                    )
                )

    return steps


def build_plan(*, repo_root: Path, makefile_path: Path, include_extra_migrations: bool) -> List[Step]:
    mf = _read_text(makefile_path)
    all_targets = _parse_all_migrations_targets(mf)
    recipes = _parse_makefile_recipes(mf)

    plan: List[Step] = []
    seen_files: set[str] = set()

    for t in all_targets:
        if t not in recipes:
            raise RuntimeError(f"Target `{t}` referenced by all-migrations, but no recipe found.")
        steps = _extract_steps_for_target(t, recipes[t])
        for s in steps:
            if s.kind == "file":
                if s.payload not in seen_files:
                    seen_files.add(s.payload)
                    plan.append(s)
            else:
                plan.append(s)

    if include_extra_migrations:
        # Add any migrations not referenced by all-migrations (e.g. 0041_*.sql).
        mig_dir = repo_root / "migrations"
        extra = sorted(mig_dir.glob("*.sql"), key=lambda p: p.name)
        for p in extra:
            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            if rel not in seen_files:
                plan.append(
                    Step(
                        kind="file",
                        label=f"Apply {rel} (extra)",
                        target="extra-migrations",
                        payload=rel,
                    )
                )
                seen_files.add(rel)

    return plan


def _find_psql(psql_path: Optional[str]) -> str:
    if psql_path:
        return psql_path
    found = shutil.which("psql")
    if not found:
        raise RuntimeError(
            "Could not find `psql` in PATH. Install Postgres client tools or pass --psql /path/to/psql."
        )
    return found


def _run_psql_file(*, psql: str, database_url: str, file_path: Path, quiet: bool) -> None:
    cmd = [
        psql,
        database_url,
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(file_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if quiet else None)


def _run_psql_sql(*, psql: str, database_url: str, sql: str, quiet: bool) -> None:
    cmd = [
        psql,
        database_url,
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if quiet else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply db/ + migrations/ to DATABASE_URL in Makefile order.")
    ap.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Target database URL (defaults to env DATABASE_URL).",
    )
    ap.add_argument(
        "--makefile",
        default=None,
        help="Path to Makefile (defaults to repo root Makefile).",
    )
    ap.add_argument("--psql", default=None, help="Path to psql binary (defaults to PATH lookup).")
    ap.add_argument("--plan", action="store_true", help="Print planned steps and exit.")
    ap.add_argument(
        "--only-makefile-plan",
        action="store_true",
        help="Do NOT append extra migrations/*.sql that aren't referenced by all-migrations.",
    )
    ap.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip steps that look like ANALYZE commands.",
    )
    ap.add_argument(
        "--start-at",
        default=None,
        help="Start at the first step whose label contains this substring (useful for resume).",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress psql stdout (stderr still shows).")
    args = ap.parse_args()

    if not args.database_url:
        print("ERROR: Provide --database-url or set DATABASE_URL.", file=sys.stderr)
        sys.exit(2)

    repo_root = _repo_root()
    makefile_path = Path(args.makefile) if args.makefile else (repo_root / "Makefile")
    if not makefile_path.exists():
        print(f"ERROR: Makefile not found at {makefile_path}", file=sys.stderr)
        sys.exit(2)

    try:
        psql = _find_psql(args.psql)
        plan = build_plan(
            repo_root=repo_root,
            makefile_path=makefile_path,
            include_extra_migrations=not args.only_makefile_plan,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # Optional filters
    if args.skip_analyze:
        plan = [s for s in plan if not (s.kind == "sql" and "ANALYZE" in s.payload.upper())]

    if args.start_at:
        start_idx = None
        needle = args.start_at.lower()
        for i, s in enumerate(plan):
            if needle in s.label.lower():
                start_idx = i
                break
        if start_idx is None:
            print(f"ERROR: --start-at '{args.start_at}' did not match any step label.", file=sys.stderr)
            sys.exit(2)
        plan = plan[start_idx:]

    if args.plan:
        for i, s in enumerate(plan, start=1):
            print(f"{i:03d}. [{s.target}] {s.label}")
        return

    # Execute
    total = len(plan)
    for i, step in enumerate(plan, start=1):
        t0 = time.time()
        print(f"[{i}/{total}] {step.label}")
        try:
            if step.kind == "file":
                fp = repo_root / Path(step.payload)
                if not fp.exists():
                    raise RuntimeError(f"SQL file not found: {fp}")
                _run_psql_file(psql=psql, database_url=args.database_url, file_path=fp, quiet=args.quiet)
            elif step.kind == "sql":
                _run_psql_sql(psql=psql, database_url=args.database_url, sql=step.payload, quiet=args.quiet)
            else:
                raise RuntimeError(f"Unknown step kind: {step.kind}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Step failed ({step.label}) with exit code {e.returncode}", file=sys.stderr)
            sys.exit(e.returncode)
        except Exception as e:
            print(f"ERROR: Step failed ({step.label}): {e}", file=sys.stderr)
            sys.exit(1)
        dt = time.time() - t0
        print(f"  OK ({dt:.2f}s)")

    print("Done.")


if __name__ == "__main__":
    main()

