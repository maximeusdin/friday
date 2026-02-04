"""
Meta endpoints - health, version, contract info
"""
import os
import subprocess
from fastapi import APIRouter

router = APIRouter()

CONTRACT_VERSION = "v1"
API_VERSION = "0.1.0"


def get_git_sha() -> str:
    """Get current git SHA for build info."""
    sha = (os.getenv("GIT_SHA") or os.getenv("FRIDAY_GIT_SHA") or "").strip()
    if sha:
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


@router.get("/meta")
def get_meta():
    """
    Return API metadata including contract version.
    Useful for debugging and client compatibility checks.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "api_version": API_VERSION,
        "build": get_git_sha(),
    }
