"""
Auth: form login (POST /auth/login) + optional Cognito OAuth2.
- POST /auth/login → password / dev login (JSON body, sets session cookie).
- GET /auth/oauth/cognito/login → start Cognito OAuth redirect.
- GET /auth/oauth/cognito/callback → OAuth callback.
- GET /auth/me, POST|GET /auth/logout.
Cookies: Domain=.fridayarchive.org, Path=/, Secure, HttpOnly, SameSite=None for cross-site.
"""
import json
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
_COGNITO_CLIENT_SECRET_RAW = os.environ.get("COGNITO_CLIENT_SECRET")
COGNITO_ISSUER = os.environ.get("COGNITO_ISSUER", "").rstrip("/")


def _get_cognito_client_secret() -> str | None:
    """
    Return the raw Cognito client secret for /oauth2/token (Basic auth).
    Secrets Manager often stores JSON or adds newlines; we strip and optionally
    parse so the value sent to Cognito is exactly the raw secret.
    """
    raw = _COGNITO_CLIENT_SECRET_RAW
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    # If stored as JSON (e.g. {"secret":"..."} or {"COGNITO_CLIENT_SECRET":"..."})
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            for key in ("COGNITO_CLIENT_SECRET", "client_secret", "secret"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return raw
        except Exception:
            raise RuntimeError("COGNITO_CLIENT_SECRET is JSON but missing expected key")
    return raw


UI_REDIRECT_AFTER_LOGIN = os.environ.get("UI_REDIRECT_AFTER_LOGIN", "https://fridayarchive.org/")
# Backend callback: Cognito redirects here; we exchange code, set cookie, 302 to UI_REDIRECT_AFTER_LOGIN.
COGNITO_REDIRECT_URI_DEFAULT = "https://api.fridayarchive.org/auth/oauth/cognito/callback"
REDIRECT_URI = os.environ.get("COGNITO_REDIRECT_URI", COGNITO_REDIRECT_URI_DEFAULT)

SESSION_COOKIE_NAME = os.environ.get("SESSION_COOKIE_NAME", "friday_session")
# Domain with leading dot so cookie is sent from fridayarchive.org → api.fridayarchive.org
COOKIE_DOMAIN = os.environ.get("COOKIE_DOMAIN", ".fridayarchive.org")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
# SameSite=None typically required for cross-site (site → API); requires Secure=True
COOKIE_SAMESITE = "none" if COOKIE_SECURE else "lax"

def _parse_secret_env(raw: str, *keys: str) -> str:
    """Extract secret value from env var that may be a raw string or JSON wrapper."""
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            for k in keys:
                val = obj.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass
    return raw


APP_SESSION_SECRET = _parse_secret_env(
    os.environ.get("APP_SESSION_SECRET", ""),
    "APP_SESSION_SECRET", "secret",
)
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days
STATE_COOKIE_MAX_AGE = 600  # 5–10 min

# --- Startup diagnostics ---
log.info(
    "Auth config loaded: COOKIE_DOMAIN=%s  COOKIE_SECURE=%s  SAMESITE=%s  "
    "SESSION_COOKIE=%s  REDIRECT_URI=%s  UI_REDIRECT=%s  "
    "APP_SESSION_SECRET=%s  COGNITO_CLIENT_ID=%s",
    COOKIE_DOMAIN, COOKIE_SECURE, COOKIE_SAMESITE,
    SESSION_COOKIE_NAME, REDIRECT_URI, UI_REDIRECT_AFTER_LOGIN,
    f"{APP_SESSION_SECRET[:4]}…" if APP_SESSION_SECRET else "EMPTY",
    COGNITO_CLIENT_ID or "EMPTY",
)

_JWKS: dict | None = None
_JWKS_FETCHED_AT = 0.0
_JWKS_TTL_SECONDS = 3600


def _auth_configured() -> bool:
    return bool(COGNITO_DOMAIN and COGNITO_CLIENT_ID and COGNITO_ISSUER and APP_SESSION_SECRET)


def _auth_config_hint() -> str:
    """Return a hint for missing auth config (local dev)."""
    missing = []
    if not COGNITO_DOMAIN:
        missing.append("COGNITO_DOMAIN")
    if not COGNITO_CLIENT_ID:
        missing.append("COGNITO_CLIENT_ID")
    if not COGNITO_ISSUER:
        missing.append("COGNITO_ISSUER")
    if not APP_SESSION_SECRET:
        missing.append("APP_SESSION_SECRET")
    if missing:
        return f" Set in .env: {', '.join(missing)}. See .env.example."
    return ""


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


def _set_state_cookie(resp: Response, state: str, cookie_domain: str | None = None) -> None:
    domain = cookie_domain if cookie_domain else COOKIE_DOMAIN
    # For localhost, omit domain so cookie works for the host
    kwargs = {
        "max_age": STATE_COOKIE_MAX_AGE,
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": COOKIE_SAMESITE,
        "path": "/",
    }
    if domain and domain != "localhost":
        kwargs["domain"] = domain
    resp.set_cookie("oauth_state", state, **kwargs)


def _pop_state_cookie(resp: Response, cookie_domain: str | None = None) -> None:
    domain = cookie_domain if cookie_domain is not None else COOKIE_DOMAIN
    resp.delete_cookie("oauth_state", domain=domain or None, path="/")


def _create_app_session(sub: str, email: str | None, exp_seconds: int = SESSION_TTL_SECONDS) -> str:
    now = int(time.time())
    payload = {"sub": sub, "email": email, "iat": now, "exp": now + exp_seconds}
    return jwt.encode(payload, APP_SESSION_SECRET, algorithm="HS256")


def _set_session_cookie(
    resp: Response,
    session_jwt: str,
    cookie_domain: str | None = None,
    secure: bool | None = None,
) -> None:
    domain = cookie_domain if cookie_domain is not None else COOKIE_DOMAIN
    secure_val = secure if secure is not None else COOKIE_SECURE
    kwargs: dict = {
        "max_age": SESSION_TTL_SECONDS,
        "httponly": True,
        "secure": secure_val,
        "samesite": "lax" if not secure_val else COOKIE_SAMESITE,
        "path": "/",
    }
    if domain:
        kwargs["domain"] = domain
    resp.set_cookie(SESSION_COOKIE_NAME, session_jwt, **kwargs)


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
async def auth_oauth_cognito_login(
    request: Request,
    redirect_uri: str | None = None,
    return_url: str | None = None,
):
    """Start Cognito OAuth2 flow: redirect to Cognito hosted UI.

    Query params:
      redirect_uri: Where Cognito should redirect after login (default from env).
        Prod: https://api.fridayarchive.org/auth/oauth/cognito/callback
        Localhost: http://localhost:8000/auth/oauth/cognito/callback
      return_url: Where to send the user after exchange (optional).
    """
    if not _auth_configured():
        log.error("OAuth login: auth not configured (COGNITO_DOMAIN=%s, CLIENT_ID=%s, ISSUER=%s, SECRET=%s)",
                  bool(COGNITO_DOMAIN), bool(COGNITO_CLIENT_ID), bool(COGNITO_ISSUER), bool(APP_SESSION_SECRET))
        raise HTTPException(
            status_code=503,
            detail="Auth not configured." + _auth_config_hint(),
        )

    effective_redirect_uri = (redirect_uri or "").strip() or REDIRECT_URI
    state = secrets.token_urlsafe(32)

    params = {
        "client_id": COGNITO_CLIENT_ID,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": effective_redirect_uri,
        "state": state,
    }
    url = f"{COGNITO_DOMAIN}/oauth2/authorize?{urlencode(params)}"
    resp = RedirectResponse(url=url, status_code=302)

    # For localhost callback, use cookie domain that works (omit or localhost)
    cookie_domain = COOKIE_DOMAIN
    if "localhost" in effective_redirect_uri or "127.0.0.1" in effective_redirect_uri:
        cookie_domain = "localhost"  # Ensures cookie is sent when frontend calls backend
    _set_state_cookie(resp, state, cookie_domain)

    log.info("OAuth login: redirect_uri=%s state=%s…%s",
             effective_redirect_uri, state[:8], state[-4:])
    return resp


@router.post("/oauth/cognito/exchange")
async def auth_oauth_cognito_exchange(request: Request):
    """Exchange Cognito auth code for session. Called by frontend callback page.

    Body: { "code": "...", "state": "...", "redirect_uri": "..." }
    Validates state cookie, exchanges code, sets session cookie, returns { "ok": true }.
    """
    if not _auth_configured():
        raise HTTPException(status_code=503, detail="Auth not configured")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")
    code = (body.get("code") or "").strip()
    state = (body.get("state") or "").strip()
    redirect_uri = (body.get("redirect_uri") or "").strip()
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")
    if not redirect_uri:
        raise HTTPException(status_code=400, detail="Missing redirect_uri")

    cookie_state = request.cookies.get("oauth_state")
    if not cookie_state:
        log.error("OAuth exchange: oauth_state cookie MISSING")
        raise HTTPException(status_code=400, detail="Missing oauth_state cookie — start login again")
    if cookie_state != state:
        log.error("OAuth exchange: state MISMATCH")
        raise HTTPException(status_code=400, detail="State mismatch — start login again")

    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    client_secret = _get_cognito_client_secret()
    data = {
        "grant_type": "authorization_code",
        "client_id": COGNITO_CLIENT_ID,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    auth = (COGNITO_CLIENT_ID, client_secret) if client_secret else None

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(token_url, data=data, headers=headers, auth=auth)
        if r.status_code != 200:
            log.error("Token exchange FAILED: status=%s body=%s", r.status_code, r.text[:300])
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
    resp = JSONResponse({"ok": True})
    is_localhost = "localhost" in redirect_uri or "127.0.0.1" in redirect_uri
    state_cookie_domain = "localhost" if is_localhost else COOKIE_DOMAIN
    _pop_state_cookie(resp, cookie_domain=state_cookie_domain)
    # For localhost: domain=localhost, secure=False (localhost uses HTTP)
    cookie_domain = "localhost" if is_localhost else None
    _set_session_cookie(resp, session_jwt, cookie_domain=cookie_domain, secure=not is_localhost)
    log.info("OAuth exchange SUCCESS: sub=%s email=%s", sub, email)
    return resp


@router.get("/oauth/cognito/callback")
async def auth_oauth_cognito_callback(request: Request, code: str | None = None, state: str | None = None):
    # --- DIAGNOSTIC: log everything the browser sent ---
    all_cookie_names = list(request.cookies.keys())
    cookie_state = request.cookies.get("oauth_state")
    log.info(
        "OAuth callback: cookies_received=%s  oauth_state_cookie=%s  state_query=%s  "
        "host=%s  origin=%s  referer=%s",
        all_cookie_names,
        f"{cookie_state[:8]}…{cookie_state[-4:]}" if cookie_state else "MISSING",
        f"{state[:8]}…{state[-4:]}" if state else "MISSING",
        request.headers.get("host", "?"),
        request.headers.get("origin", "?"),
        request.headers.get("referer", "?")[:80] if request.headers.get("referer") else "?",
    )

    if not _auth_configured():
        raise HTTPException(status_code=503, detail="Auth not configured")
    if not code or not state:
        log.warning("OAuth callback: missing code=%s state=%s", bool(code), bool(state))
        raise HTTPException(status_code=400, detail="Missing code or state")
    if not cookie_state:
        log.error(
            "OAuth callback: oauth_state cookie MISSING. Browser did not send it. "
            "Cookie was set with domain=%s path=/ secure=%s samesite=%s. "
            "Possible causes: (1) cookie blocked by browser, (2) domain mismatch, "
            "(3) ALB/CDN stripping cookie header, (4) SameSite policy.",
            COOKIE_DOMAIN, COOKIE_SECURE, COOKIE_SAMESITE,
        )
        raise HTTPException(status_code=400, detail="Missing oauth_state cookie — see server logs")
    if cookie_state != state:
        log.error(
            "OAuth callback: state MISMATCH. cookie=%s…%s  query=%s…%s  "
            "This means a stale/different cookie was sent (e.g. second tab, cached redirect).",
            cookie_state[:8], cookie_state[-4:], state[:8], state[-4:],
        )
        raise HTTPException(status_code=400, detail="State mismatch — see server logs")

    # Token exchange: form-encoded body, Basic auth with raw client secret (no hash).
    # Secret must be exactly what Cognito has; strip whitespace / parse JSON if from Secrets Manager.
    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    client_secret = _get_cognito_client_secret()
    data = {
        "grant_type": "authorization_code",
        "client_id": COGNITO_CLIENT_ID,
        "code": code,
        "redirect_uri": REDIRECT_URI,
    }
    # Cognito accepts either Basic auth (client_id:client_secret) or client_id+client_secret in body.
    # We use Basic auth; do not send client_secret in body.
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    auth = (COGNITO_CLIENT_ID, client_secret) if client_secret else None

    log.info("OAuth callback: exchanging code at %s (client_secret present=%s)", token_url, bool(client_secret))
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(token_url, data=data, headers=headers, auth=auth)
        if r.status_code != 200:
            log.error("Token exchange FAILED: status=%s body=%s", r.status_code, r.text[:300])
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
    log.info(
        "OAuth callback SUCCESS: sub=%s email=%s → redirect to %s  "
        "session cookie domain=%s secure=%s samesite=%s",
        sub, email, UI_REDIRECT_AFTER_LOGIN,
        COOKIE_DOMAIN, COOKIE_SECURE, COOKIE_SAMESITE,
    )
    return resp


@router.get("/me")
async def auth_me(request: Request):
    cookie_names = list(request.cookies.keys())
    has_session = SESSION_COOKIE_NAME in request.cookies
    sess = _read_app_session(request)
    if not sess:
        log.info(
            "/auth/me → 401  cookies_present=%s  has_%s=%s  origin=%s",
            cookie_names, SESSION_COOKIE_NAME, has_session,
            request.headers.get("origin", "?"),
        )
        resp = JSONResponse({"detail": "Not authenticated"}, status_code=401)
        resp.headers["Cache-Control"] = "no-store"
        return resp
    log.info("/auth/me → 200  sub=%s email=%s", sess.get("sub"), sess.get("email"))
    resp = JSONResponse({"sub": sess.get("sub"), "email": sess.get("email")})
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
