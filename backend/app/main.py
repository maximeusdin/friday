"""
Friday Research Console - FastAPI Backend
"""
import os
import logging
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import sessions, plans, results, documents, meta
from app.services.db import ConfigError
from app.services.db import get_conn

# Optional: load .env files (dev convenience). Disabled by default so the app
# can run in a container with only environment variables.
if os.getenv("FRIDAY_LOAD_DOTENV") == "1":
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        load_dotenv(Path(__file__).parent.parent / ".env")
    except Exception:
        # If python-dotenv isn't installed or files are missing, ignore.
        pass

app = FastAPI(
    title="Friday Research Console",
    description="AI research assistant for Cold War archival materials",
    version="0.1.0",
)

log = logging.getLogger("friday")


def _mask_url(url: str) -> str:
    # Hide password if present in URL
    # e.g. postgresql://user:pass@host/db -> postgresql://user:***@host/db
    import re

    return re.sub(r"(postgres(?:ql)?://[^:]+:)([^@]+)@", r"\1***@", url)


@app.on_event("startup")
async def startup_event():
    db_url = os.getenv("DATABASE_URL", "")
    db_host = os.getenv("DB_HOST", "")
    log.info("ENV_CHECK: DATABASE_URL present=%s", bool(db_url))
    if db_url:
        log.info("ENV_CHECK: DATABASE_URL(masked)=%s", _mask_url(db_url))
    log.info("ENV_CHECK: DB_HOST=%s", db_host or "<missing>")


# Consistent error shape
@app.exception_handler(ConfigError)
def config_error_handler(_request: Request, exc: ConfigError):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "CONFIG_ERROR",
                "message": str(exc),
            }
        },
    )


@app.exception_handler(HTTPException)
def http_exception_handler(_request: Request, exc: HTTPException):
    """
    Ensure FastAPI HTTPExceptions return our stable ErrorResponse shape.
    """
    status = exc.status_code
    if status == 404:
        code = "NOT_FOUND"
    elif status == 400:
        code = "VALIDATION_ERROR"
    elif status == 409:
        code = "CONFLICT"
    else:
        code = f"HTTP_{status}"

    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": str(exc.detail),
            }
        },
    )

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev
        "http://127.0.0.1:3000",
        "http://fridayarchive.org",  # S3 production
        "http://www.fridayarchive.org",
        "https://fridayarchive.org",  # Future HTTPS
        "https://www.fridayarchive.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(meta.router, prefix="/api", tags=["meta"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(plans.router, prefix="/api/plans", tags=["plans"])
app.include_router(results.router, prefix="/api/result-sets", tags=["results"])
app.include_router(documents.router, prefix="/api", tags=["documents"])


@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
    - status: ok
    - build: short git SHA if available
    - db: connectivity status (SELECT 1)
    """
    build = meta.get_git_sha()
    db_ok = False
    db_error = None
    try:
        conn = get_conn(connect_timeout_seconds=2)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
            db_ok = True
        finally:
            conn.close()
    except Exception as e:
        db_error = str(e)

    out = {"status": "ok", "build": build, "db": {"ok": db_ok}}
    if db_error and not db_ok:
        out["db"]["error"] = db_error[:200]
    return out
