"""
Friday Research Console - FastAPI Backend
"""
import os
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.routes import sessions, plans, results, documents, meta
from app.services.db import ConfigError

# Load .env from repo root and backend/ (dev convenience)
load_dotenv(REPO_ROOT / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(
    title="Friday Research Console",
    description="AI research assistant for Cold War archival materials",
    version="0.1.0",
)

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

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev
        "http://127.0.0.1:3000",
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
    """Health check endpoint."""
    return {"status": "ok"}
