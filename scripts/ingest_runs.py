#!/usr/bin/env python3
"""
Shared ingest run bookkeeping.

Goal: make ingest scripts resumable/idempotent per-input file by tracking:
- source_key (stable identifier per input)
- source_fingerprint (sha256 if available; otherwise a cheap stat-based fingerprint)
- pipeline_version (so logic changes can force reruns)
- status (running/success/failed)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


def ensure_ingest_runs_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_runs (
          id BIGSERIAL PRIMARY KEY,
          source_key TEXT NOT NULL UNIQUE,
          source_sha256 TEXT NOT NULL,
          pipeline_version TEXT NOT NULL,
          status TEXT NOT NULL CHECK (status IN ('success','failed','running')),
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          finished_at TIMESTAMPTZ,
          error TEXT
        )
        """
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint_fast(path: Path) -> str:
    """
    A fast fingerprint when computing sha256 is too expensive.
    Not cryptographically strong; intended for "did the file probably change?" checks.
    """
    st = path.stat()
    return f"stat:size={st.st_size}:mtime_ns={st.st_mtime_ns}"


def should_run(cur, *, source_key: str, source_fingerprint: str, pipeline_version: str) -> bool:
    cur.execute(
        "SELECT source_sha256, pipeline_version, status FROM ingest_runs WHERE source_key=%s",
        (source_key,),
    )
    row = cur.fetchone()
    return (
        row is None
        or row[0] != source_fingerprint
        or row[1] != pipeline_version
        or row[2] != "success"
    )


def mark_running(cur, *, source_key: str, source_fingerprint: str, pipeline_version: str) -> None:
    cur.execute(
        """
        INSERT INTO ingest_runs (source_key, source_sha256, pipeline_version, status, started_at)
        VALUES (%s, %s, %s, 'running', now())
        ON CONFLICT (source_key) DO UPDATE
        SET source_sha256 = EXCLUDED.source_sha256,
            pipeline_version = EXCLUDED.pipeline_version,
            status='running',
            started_at=now(),
            finished_at=NULL,
            error=NULL
        """,
        (source_key, source_fingerprint, pipeline_version),
    )


def mark_success(cur, *, source_key: str) -> None:
    cur.execute(
        "UPDATE ingest_runs SET status='success', finished_at=now() WHERE source_key=%s",
        (source_key,),
    )


def mark_failed_best_effort(
    connect_fn: Callable[[], object],
    *,
    source_key: str,
    source_fingerprint: str,
    pipeline_version: str,
    error: str,
) -> None:
    try:
        conn = connect_fn()
        try:
            with conn:  # type: ignore[attr-defined]
                with conn.cursor() as cur:  # type: ignore[attr-defined]
                    ensure_ingest_runs_table(cur)
                    cur.execute(
                        """
                        INSERT INTO ingest_runs (source_key, source_sha256, pipeline_version, status, started_at, finished_at, error)
                        VALUES (%s, %s, %s, 'failed', now(), now(), %s)
                        ON CONFLICT (source_key) DO UPDATE
                        SET status='failed',
                            finished_at=now(),
                            error=EXCLUDED.error,
                            source_sha256=EXCLUDED.source_sha256,
                            pipeline_version=EXCLUDED.pipeline_version
                        """,
                        (source_key, source_fingerprint, pipeline_version, error),
                    )
        finally:
            conn.close()  # type: ignore[attr-defined]
    except Exception:
        # Never hide the original ingest error.
        pass

