"""
Auth: form login (POST /auth/login) + optional Cognito OAuth2.
- POST /auth/login → password / dev login (JSON body, sets session cookie).
- GET /auth/oauth/cognito/login → start Cognito OAuth redirect.
- GET /auth/oauth/cognito/callback → OAuth callback.
- GET /auth/me, POST|GET /auth/logout.
Cookies: Domain=.fridayarchive.org, Path=/, Secure, HttpOnly, SameSite=None for cross-site.
"""
import os
import time
import secrets
import logging
from urllib.parse import urlencode

import httpx
from jose import jwt
from fastapi import APIRouter, Request, Response, Depends, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse

router = APIRouter()
log = logging.getLogger("friday.auth")

# =========================
# Config (set as env vars)
# =========================
COGNITO_DOMAIN = os.environ.get("COGNITO_DOMAIN", "").rstrip("/")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "")
COGNITO_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET")
COGNITO_ISSUER = os.environ.get("COGNITO_ISSUER", "").rstrip("/")
UI_REDIRECT_AFTER_LOGIN = os.environ.get("UI_REDIRECT_AFTER_LOGIN", "https://fridayarchive.org/")
REDIRECT_URI = os.environ.get(
    "COGNITO_REDIRECT_URI", "https://api.fridayarchive.org/auth/oauth/cognito/callback"
)

SESSION_COOKIE_NAME = os.environ.get("SESSION_COOKIE_NAME", "friday_session")
# Domain with leading dot so cookie is sent from fridayarchive.org → api.fridayarchive.org
COOKIE_DOMAIN = os.environ.get("COOKIE_DOMAIN", ".fridayarchive.org")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
# SameSite=None typically required for cross-site (site → API); requires Secure=True
COOKIE_SAMESITE = "none" if COOKIE_SECURE else "lax"

APP_SESSION_SECRET = os.environ.get("APP_SESSION_SECRET", "")
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days
STATE_COOKIE_MAX_AGE = 600  # 5–10 min

_JWKS: dict | None = None
_JWKS_FETCHED_AT = 0.0
_JWKS_TTL_SECONDS = 3600


def _auth_configured() -> bool:
    return bool(COGNITO_DOMAIN and COGNITO_CLIENT_ID and COGNITO_ISSUER and APP_SESSION_SECRET)


async def _get_jwks() -> dict:
    global _JWKS, _JWKS_FETCHED_AT
    now = time.time()
    if _JWKS and (now - _JWKS_FETCHED_AT) < _JWKS_TTL_SECONDS:
        return _JWKS
    jwks_url = f"{COGNITO_ISSUER}/.well-known/jwks.json"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(jwks_url)
        r.raise_for_status()
        _JWKS = r.json()
        _JWKS_FETCHED_AT = now
        return _JWKS


def _set_state_cookie(resp: Response, state: str) -> None:
    resp.set_cookie(
        "oauth_state",
        state,
        max_age=STATE_COOKIE_MAX_AGE,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN,
        path="/",
    )


def _pop_state_cookie(resp: Response) -> None:
    resp.delete_cookie("oauth_state", domain=COOKIE_DOMAIN, path="/")


def _create_app_session(sub: str, email: str | None, exp_seconds: int = SESSION_TTL_SECONDS) -> str:
    now = int(time.time())
    payload = {"sub": sub, "email": email, "iat": now, "exp": now + exp_seconds}
    return jwt.encode(payload, APP_SESSION_SECRET, algorithm="HS256")


def _set_session_cookie(resp: Response, session_jwt: str) -> None:
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        session_jwt,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN,
        path="/",
    )


def _clear_session_cookie(resp: Response) -> None:
    resp.delete_cookie(SESSION_COOKIE_NAME, domain=COOKIE_DOMAIN, path="/")


def _read_app_session(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    try:
        return jwt.decode(token, APP_SESSION_SECRET, algorithms=["HS256"])
    except Exception:
        return None


def require_user(request: Request):
    """FastAPI dependency: require authenticated user; return session dict (sub, email, ...)."""
    sess = _read_app_session(request)
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sess


def _verify_cognito_id_token(id_token: str, jwks: dict) -> dict:
    headers = jwt.get_unverified_header(id_token)
    kid = headers.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="Missing kid")
    key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if not key:
        raise HTTPException(status_code=401, detail="Unknown kid")
    try:
        claims = jwt.decode(
            id_token,
            key,
            algorithms=[headers.get("alg", "RS256")],
            audience=COGNITO_CLIENT_ID,
            issuer=COGNITO_ISSUER,
            options={"verify_at_hash": False},
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid id_token: {e}")
    if claims.get("token_use") != "id":
        raise HTTPException(status_code=401, detail="token_use is not id")
    return claims


# =============================================================================
# Routes (mounted at prefix /auth in main)
# =============================================================================

# Simple (non-OAuth) login: POST with JSON body; sets session cookie.
# Frontend /signin page POSTs here with credentials: 'include'.
@router.post("/login")
async def auth_login_post(request: Request):
    """
    Non-OAuth login: accept JSON { email, password? }, create session, set cookie.
    Frontend /signin page should POST here with credentials: 'include'.
    """
    if not APP_SESSION_SECRET:
        raise HTTPException(status_code=503, detail="Auth not configured (APP_SESSION_SECRET)")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")
    email = (body.get("email") or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="email required")
    password = body.get("password") or ""
    simple_password = os.environ.get("SIMPLE_AUTH_PASSWORD")
    allow_insecure = os.environ.get("ALLOW_INSECURE_DEV_LOGIN", "").lower() == "true"
    # Production: require SIMPLE_AUTH_PASSWORD and validate. Dev: allow no check only if explicit.
    if allow_insecure:
        if simple_password and password != simple_password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    else:
        if not simple_password or simple_password == "":
            raise HTTPException(
                status_code=503,
                detail="SIMPLE_AUTH_PASSWORD must be set (or ALLOW_INSECURE_DEV_LOGIN=true for dev only)",
            )
        if password != simple_password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    sub = email  # use email as sub for simple auth
    session_jwt = _create_app_session(sub=sub, email=email)
    resp = JSONResponse({"ok": True})
    _set_session_cookie(resp, session_jwt)
    return resp


# OAuth (Cognito): separate paths so POST /auth/login is form/password only.
@router.get("/oauth/cognito/login")
async def auth_oauth_cognito_login():
    """Start Cognito OAuth2 flow: redirect to Cognito hosted UI."""
    if not _auth_configured():
        raise HTTPException(status_code=503, detail="Auth not configured")
    state = secrets.token_urlsafe(32)
    params = {
        "client_id": COGNITO_CLIENT_ID,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": REDIRECT_URI,
        "state": state,
    }
    url = f"{COGNITO_DOMAIN}/oauth2/authorize?{urlencode(params)}"
    resp = RedirectResponse(url=url, status_code=302)
    _set_state_cookie(resp, state)
    return resp


@router.get("/oauth/cognito/callback")
async def auth_oauth_cognito_callback(request: Request, code: str | None = None, state: str | None = None):
    if not _auth_configured():
        raise HTTPException(status_code=503, detail="Auth not configured")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")
    cookie_state = request.cookies.get("oauth_state")
    if not cookie_state or cookie_state != state:
        log.warning("Auth callback: state mismatch (no token/code logged)")
        raise HTTPException(status_code=400, detail="Invalid state")

    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": COGNITO_CLIENT_ID,
        "code": code,
        "redirect_uri": REDIRECT_URI,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    auth = (COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET) if COGNITO_CLIENT_SECRET else None

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(token_url, data=data, headers=headers, auth=auth)
        if r.status_code != 200:
            log.warning("Token exchange failed: status=%s body=%s", r.status_code, r.text[:200])
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {r.text[:200]}")
        token_payload = r.json()

    id_token = token_payload.get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="No id_token returned")

    jwks = await _get_jwks()
    claims = _verify_cognito_id_token(id_token, jwks)
    sub = claims["sub"]
    email = claims.get("email")

    session_jwt = _create_app_session(sub=sub, email=email)
    resp = RedirectResponse(url=UI_REDIRECT_AFTER_LOGIN, status_code=302)
    _pop_state_cookie(resp)
    _set_session_cookie(resp, session_jwt)
    return resp


@router.get("/me")
async def auth_me(request: Request):
    sess = _read_app_session(request)
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse({"sub": sess.get("sub"), "email": sess.get("email")})


@router.post("/logout")
async def auth_logout_post():
    resp = JSONResponse({"ok": True})
    _clear_session_cookie(resp)
    return resp


@router.get("/logout")
async def auth_logout_get():
    resp = RedirectResponse(url=UI_REDIRECT_AFTER_LOGIN.rstrip("/") or "/", status_code=302)
    _clear_session_cookie(resp)
    return resp
