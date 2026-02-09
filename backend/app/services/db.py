"""
Database connection helper.

Priority:
  1. DATABASE_URL env var (full connection string) — used if set.
  2. Individual DB_HOST / DB_USER / DB_PASS / DB_NAME / DB_PORT env vars —
     built into a DSN with proper URL-encoding of the password.
     This is the preferred production path because ECS can inject DB_PASS
     directly from the RDS-managed secret (which auto-rotates).

If neither is available, raise a clear error.
"""
import os
import logging
from urllib.parse import quote as urlquote

import psycopg2

log = logging.getLogger("friday.db")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing/misconfigured."""


def get_dsn() -> str:
    # 1. Explicit DATABASE_URL takes priority (local dev, overrides)
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn

    # 2. Build from individual components (production: DB_PASS from Secrets Manager)
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER", "friday")
    password = os.getenv("DB_PASS", "")
    dbname = os.getenv("DB_NAME", "friday")
    port = os.getenv("DB_PORT", "5432")
    sslmode = os.getenv("DB_SSLMODE", "require")

    if host and password:
        encoded_pass = urlquote(password, safe="")
        dsn = f"postgresql://{user}:{encoded_pass}@{host}:{port}/{dbname}?sslmode={sslmode}&connect_timeout=5"
        return dsn

    raise ConfigError(
        "Database not configured. Set DATABASE_URL, or set DB_HOST + DB_PASS."
    )


def get_conn(*, connect_timeout_seconds: int | None = None):
    """
    Get a database connection.

    Caller is responsible for closing the connection.
    """
    if connect_timeout_seconds is not None:
        return psycopg2.connect(get_dsn(), connect_timeout=connect_timeout_seconds)
    return psycopg2.connect(get_dsn())
