"""
Database connection helper.

Uses DATABASE_URL. If it's missing, raise a clear error so the API can return a
useful response instead of a generic 500 traceback.
"""
import os
import psycopg2


class ConfigError(RuntimeError):
    """Raised when required configuration is missing/misconfigured."""

def get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise ConfigError("Missing DATABASE_URL environment variable")
    return dsn


def get_conn():
    """
    Get a database connection.
    
    Uses DATABASE_URL environment variable.
    Caller is responsible for closing the connection.
    """
    return psycopg2.connect(get_dsn())
