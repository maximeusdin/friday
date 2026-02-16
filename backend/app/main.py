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

from app.routes import meta  # used by /health
from app.services.db import ConfigError
from app.services.db import get_conn


def register_routers(app: FastAPI) -> None:
    """
    Register all routers in one place so the entrypoint can't forget one.
    Container runs: uvicorn app.main:app — so this module is the only place that matters.
    Imports are inside this function to avoid circular imports; any missing router will raise here.
    """
    from app.routes import auth_cognito, documents, meta, plans, results, sessions
    from app.routes import chat  # sessions + chat both under /api/sessions

    app.include_router(auth_cognito.router, prefix="/auth", tags=["auth"])
    app.include_router(meta.router, prefix="/api", tags=["meta"])
    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
    app.include_router(chat.router, prefix="/api/sessions", tags=["chat"])
    app.include_router(plans.router, prefix="/api/plans", tags=["plans"])
    app.include_router(results.router, prefix="/api/result-sets", tags=["results"])
    app.include_router(documents.router, prefix="/api", tags=["documents"])


# Load .env for local dev (when file exists or FRIDAY_LOAD_DOTENV=1).
# Does not override existing env vars; container env wins.
_envenv = REPO_ROOT / ".env"
if os.getenv("FRIDAY_LOAD_DOTENV") == "1" or _envenv.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        load_dotenv(Path(__file__).parent.parent / ".env")
    except Exception:
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
    db_pass = os.getenv("DB_PASS", "")
    log.info("ENV_CHECK: DATABASE_URL present=%s", bool(db_url))
    if db_url:
        log.info("ENV_CHECK: DATABASE_URL(masked)=%s", _mask_url(db_url))
    log.info("ENV_CHECK: DB_HOST=%s  DB_PASS present=%s", db_host or "<missing>", bool(db_pass))
    if db_host and db_pass:
        log.info("ENV_CHECK: DB config via components (rotation-safe)")
    elif db_url:
        log.info("ENV_CHECK: DB config via DATABASE_URL (static)")
    else:
        log.error("ENV_CHECK: NO DB CONFIG — both DATABASE_URL and DB_HOST/DB_PASS missing")


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


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch unhandled exceptions, log them, and return a consistent JSON 500.
    Without this, FastAPI returns plain "Internal Server Error" (21 chars) with no detail.
    """
    import traceback

    log.error(
        "Unhandled exception: %s\nPath: %s %s\nTraceback:\n%s",
        exc,
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    # In debug mode, include error detail in response (for local diagnosis)
    msg = str(exc)[:200] if os.getenv("FRIDAY_DEBUG") == "1" else "Internal server error"
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": msg,
            }
        },
    )

# CORS: frontend at https://fridayarchive.org, API at https://api.fridayarchive.org.
# Do not use allow_origins=["*"] with credentials — browser blocks it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fridayarchive.org",
        "https://www.fridayarchive.org",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://fridayarchive.org",
        "http://www.fridayarchive.org",
        "https://api.fridayarchive.org",
    ],
    allow_origin_regex=r"https?://([a-zA-Z0-9-]+\.)?fridayarchive\.org|https?://.*\.cloudfront\.net",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

# Register all routers (single place so ECS entrypoint can't miss auth)
register_routers(app)


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
