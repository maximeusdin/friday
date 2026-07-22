# Friday – Cognito Auth Implementation Plan

This plan integrates AWS Cognito OAuth2 (code flow) with cookie-based session JWTs into the existing FastAPI backend and static/Next.js UI. Sessions are scoped by authenticated user (Cognito `sub`).

---

## 1. Backend: Dependencies

**File:** `backend/requirements.txt`

Add:

```
httpx>=0.27.0
python-jose[cryptography]>=3.3.0
```

Then:

```bash
pip install -r backend/requirements.txt
```

---

## 2. Backend: Auth Module

**New file:** `backend/app/routes/auth_cognito.py`

- Copy the full FastAPI router code you were given (login, callback, me, logout, JWKS, cookie helpers, `_verify_cognito_id_token`, `_create_app_session`, `_read_app_session`).
- Keep routes at **path prefix** `/auth` (so `/auth/login`, `/auth/callback`, `/auth/me`, `/auth/logout`). Request scopes: `openid email profile` only; do **not** request `phone` unless you need it.
- **Optional but recommended:** Make the router load config only when env vars are set, so the app can start without Cognito in local dev (e.g. guard with `if os.getenv("COGNITO_DOMAIN"):` and only `include_router` when True; or use defaults that skip real Cognito). If you prefer “auth required everywhere”, wire the router unconditionally and set all env vars in every environment.

**Export a dependency for protected routes:**

In the same file (or in a small `backend/app/deps.py`), add:

```python
from fastapi import Request, Depends, HTTPException

def require_user(request: Request):
    sess = _read_app_session(request)
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sess  # dict with sub, email, iat, exp
```

Use it in your **existing** API routers (which are already mounted under `/api/*`). Keep auth and API separate:

- **Auth routes:** `/auth/*` (login, callback, me, logout) — auth router only.
- **API routes:** `/api/*` (e.g. `/api/sessions`, `/api/plans`, …) — your existing routers; add `Depends(require_user)` there. Do **not** put protected routes under `/api/auth/...`.

Example: in `sessions.py`, the router is mounted at `prefix="/api/sessions"`, so the path is defined relative to that:

```python
from app.routes.auth_cognito import require_user
from fastapi import Depends

# In sessions.py: router is included with prefix="/api/sessions"
@router.get("", response_model=List[Session])
def list_sessions(user=Depends(require_user)):
    sub = user["sub"]
    # ...
```

---

## 3. Backend: Wire Router and CORS

**File:** `backend/app/main.py`

- Import and mount the auth router **without** a path prefix (routes already include `/auth`):

  ```python
  from app.routes import auth_cognito
  # ...
  app.include_router(auth_cognito.router)  # no prefix; routes are /auth/login, etc.
  ```

- **CORS (cookie-based auth):** Keep it tight. You don’t need bearer tokens for the browser UI, so avoid broad `allow_headers=["*"]` and `Authorization` unless you add bearer auth later (e.g. for CLI):
  - `allow_origins`: e.g. `["https://fridayarchive.org"]` (plus localhost for dev if needed).
  - `allow_credentials=True`.
  - `allow_headers`: `["Content-Type"]` is enough for JSON API calls. If you later add `Authorization` for bearer tokens, add it then.
  - `allow_methods`: include **OPTIONS** (for preflight) — e.g. `["GET", "POST", "PUT", "DELETE", "OPTIONS"]` or `["*"]`.
  - Do **not** add `Access-Control-Allow-Headers: Authorization` until you actually use bearer auth.

---

## 4. Backend: Protect API Endpoints

Apply `Depends(require_user)` to every endpoint that should be authenticated. At minimum, that’s all session-scoped and user-scoped resources:

| Router    | File           | Endpoints to protect |
|----------|----------------|-----------------------|
| sessions | `sessions.py`  | All (create, list, get, delete, messages, state) |
| chat     | `chat.py`      | All (chat, history, stream, v9 message, delete last-pending, etc.) |
| plans    | `plans.py`     | All (get, approve, execute, clarify) |
| results  | `results.py`   | All (get result set, aggregate, summarize, chunks, expand, match-traces, entities, facets, etc.) |
| documents| `documents.py` | Optional: keep `/documents/{id}` and `/evidence` public if you want unauthenticated read; otherwise protect. |
| meta     | `meta.py`      | Keep `/api/meta` and `/health` public (no auth). |

**Pattern per route:**

```python
from app.routes.auth_cognito import require_user
from fastapi import Depends

@router.get("", response_model=List[Session])
def list_sessions(user=Depends(require_user)):
    sub = user["sub"]
    # ...
```

**Session ownership (Phase 1.5 — do sooner than later):**

Treat “session ownership in DB” as part of the initial rollout, not a later phase. Otherwise any logged-in user could access any session by guessing IDs.

- **Migration:** Add `user_sub TEXT NOT NULL` to `research_sessions` (backfill existing rows if needed, e.g. with a sentinel or one-off script).
- **Enforce scoping:** In `list_sessions`, `get_session`, `delete_session`, and any other endpoint that touches a session, use `WHERE user_sub = %s` (with `user["sub"]`). For create: set `user_sub = user["sub"]`. For chat/plans/results: only allow access to sessions where `research_sessions.user_sub = user["sub"]` (e.g. join or verify ownership before proceeding).

- **Single ownership guard:** Add a reusable helper so you don’t forget on some route. Use **`assert_session_owned(session_id, sub)`** (or a small SQL helper that returns the session iff it belongs to `sub`, else raises 404). Call it at the start of every chat/plans/results endpoint that takes a `session_id` before doing anything else. That prevents accidental cross-user access even if a `WHERE user_sub=...` is missed somewhere.

- **Later:** A `users` table and internal `user_id` FK can wait; scoping by `user_sub` is enough for v1.

---

## 5. Environment Variables (API / ECS / Secrets Manager)

Set these where the FastAPI app runs (e.g. ECS task definition or Secrets Manager):

| Variable | Example / notes |
|----------|------------------|
| `COGNITO_DOMAIN` | `https://us-west-1b9vdzkuiu.auth.us-west-1.amazoncognito.com` (no trailing slash) |
| `COGNITO_CLIENT_ID` | Your Cognito app client ID |
| `COGNITO_CLIENT_SECRET` | If app client has a secret; otherwise omit or leave unset |
| `COGNITO_ISSUER` | `https://cognito-idp.us-west-1.amazonaws.com/us-west-1_b9VdZKUiu` (no trailing slash) |
| `COGNITO_REDIRECT_URI` | `https://api.fridayarchive.org/auth/callback` |
| `UI_REDIRECT_AFTER_LOGIN` | `https://fridayarchive.org/` |
| `APP_SESSION_SECRET` | Long random string (e.g. `secrets.token_urlsafe(32)`) for signing session JWTs |
| `COOKIE_DOMAIN` | `.fridayarchive.org` |
| `COOKIE_SECURE` | `true` in production |
| `SESSION_COOKIE_NAME` | Optional; default `friday_session` |

If you use a confidential client, the token exchange will use HTTP Basic with `COGNITO_CLIENT_ID` and `COGNITO_CLIENT_SECRET`; the code you have already supports that.

**Debugging (logging and errors):**  
Log token-exchange failures (HTTP status + response body or response text) and state mismatches at **INFO** or **WARN**. Do **not** log tokens, authorization codes, or session JWTs. That will save time when Cognito rejects something (e.g. wrong redirect_uri or client secret).

---

## 6. UI: Auth API Helpers and Credentials

**Sites to update:** `site/` (production static site at fridayarchive.org) and, if you use it, `frontend/` (Next.js dev).

- **Auth base URL:** For `site`, `API_BASE` is already `https://api.fridayarchive.org`. Auth routes live at that host: `/auth/login`, `/auth/me`, `/auth/logout`. So base URL for auth is the same as `API_BASE` (no `/api` suffix).
- **Credentials:** Every request to the API (including `/auth/me` and `/api/sessions`, etc.) must send cookies: use `credentials: "include"` in `fetch` for same-site cross-origin requests (e.g. from `https://fridayarchive.org` to `https://api.fridayarchive.org`). Use it **everywhere** — including any **SSE/stream** endpoints (e.g. chat stream, v9 message stream). If you have streaming endpoints, confirm they accept cookies and that CORS is configured so preflight and the stream request both work with credentials.

- **SSE / streaming:** If you use **streaming via `fetch`**, the existing `credentials: "include"` guidance is enough. If any stream endpoints are consumed with **`EventSource`**: EventSource always sends cookies automatically (same-origin or with CORS credentials), but the backend must send CORS headers that allow credentials (e.g. `Access-Control-Allow-Credentials: true` and an allowed origin). Note that EventSource cannot set custom headers, so auth must rely on cookies only for those requests.

**Changes:**

1. **Central `fetch` / `request`:**  
   In both `site/src/lib/api.ts` and `frontend/src/lib/api.ts`, add `credentials: "include"` to the default `fetch` options so **all** API calls (including streaming fetch) send and receive cookies. If you use `EventSource` for SSE, cookies are sent automatically; ensure CORS allows credentials for the stream origin.

2. **Auth helpers (add to `api.ts` or a small `auth.ts`):**
   - **Login (redirect):**  
     - `getLoginUrl()` → return `https://api.fridayarchive.org/auth/login` (or `${API_BASE}/auth/login`).  
     - “Sign in” button: `window.location.href = getLoginUrl()`.
   - **Me:**  
     - `getAuthMe(): Promise<{ sub: string; email?: string } | null>`  
       - `GET ${API_BASE}/auth/me` with `credentials: "include"`.  
       - On 200, return body; on 401, return `null` (or throw, depending on how you want to handle “logged out”).
   - **Logout:**  
     - `logout(): Promise<void>`  
       - `POST ${API_BASE}/auth/logout` with `credentials: "include"`.  
       - Then redirect to `/` or refresh state so the UI shows logged-out.  
       - Optional: also support **GET** `/auth/logout` for convenience (e.g. “log out” links); backend can clear the session cookie and redirect to `/` on GET as well.

3. **Types:**  
   Add a type for the current user, e.g. `User | null` with `sub` and optional `email`.

---

## 7. UI: Sign In / Logout and Logged-In State

- **Header or layout:**  
  - If `getAuthMe()` returns a user: show email (or “Signed in”) and a “Log out” button that calls `logout()` (and then redirect or update state).  
  - If not: show “Sign in” that navigates to `getLoginUrl()`.

- **State:**  
  - On app load, call `getAuthMe()` once (e.g. in a layout, provider, or top-level page).  
  - Store result in React state or a small auth context so the header and any “require auth” logic can use it.

- **Protected pages:**  
  - If you have pages that must be logged-in only, redirect to `getLoginUrl()` when `getAuthMe()` returns null; after login, Cognito and the callback redirect back to `UI_REDIRECT_AFTER_LOGIN` (e.g. home). A “return URL” (e.g. `?return_to=/app/session/123` encoded in state) can be added later; not needed for v1.

---

## 8. Callback, State, and Cookies

- **Callback:**  
  User hits `https://api.fridayarchive.org/auth/login` → Cognito → `https://api.fridayarchive.org/auth/callback?code=...&state=...`.  
  Backend exchanges code for tokens, verifies ID token, creates session JWT, sets cookie, redirects to `UI_REDIRECT_AFTER_LOGIN` (e.g. `https://fridayarchive.org/`).

- **Session JWT TTL and refresh:**  
  Session JWT TTL: **1–7 days** (e.g. 7 in your code). No refresh-token handling in v1 — users re-login when the session expires. That keeps v1 simple and avoids storing Cognito refresh tokens server-side.

- **Cookie flags (state explicitly):**  
  Set these on both cookies; don’t rely on defaults.  
  - **Session cookie:** `HttpOnly=True`, `Secure=True`, `SameSite=Lax`, `Domain=.fridayarchive.org`, `Path=/`, plus `max_age` matching your session TTL (e.g. 7 days).  
  - **State cookie:** same flags — `HttpOnly=True`, `Secure=True`, `SameSite=Lax`, `Domain=.fridayarchive.org`, `Path=/` — and a **short** `max_age` (e.g. 600 seconds).

- **OAuth state:**  
  The plan uses an `oauth_state` cookie (set on login, checked on callback). For v1 this is fine **if** the cookie is HttpOnly + Secure and you compare `state` from the query with the cookie with a simple equality check. Keep the state window short (5–10 minutes; you already use a short `max_age`). **Stronger option:** make state non-spoofable by either (a) storing `{state: true}` in a short-lived server-side store (e.g. in-memory cache or Redis), or (b) signing the state cookie with HMAC so attackers can’t forge it. For most setups, HttpOnly + Secure + short TTL is acceptable for v1.

- **Cookie domain:**  
  With `COOKIE_DOMAIN=.fridayarchive.org`, the cookie is sent to both `api.fridayarchive.org` and `fridayarchive.org`, so the next request from the UI to the API will include the session cookie when using `credentials: "include"`.

- **SameSite:**  
  Use **SameSite=Lax** for the session (and state) cookie. That’s what makes the OAuth redirect and cookie set work cleanly. Do **not** switch to `SameSite=None` unless you need cross-site embedding (e.g. iframes from another origin).

---

## 9. Optional: Users Table (later)

In `auth_cognito.py`, in the callback after `_verify_cognito_id_token`, you have `sub` and `email`. If you later want a `users` table:

- Add a migration: e.g. `users (id, cognito_sub UNIQUE, email, created_at, updated_at)`.
- In the callback, `INSERT ... ON CONFLICT (cognito_sub) DO UPDATE SET email = ..., updated_at = now()`.
- Then you can use integer `user_id` in `research_sessions` and join for display; the dependency still uses `sub` from the session JWT to look up or scope by user.

---

## 10. Checklist Summary

- [ ] Add `httpx` and `python-jose[cryptography]` to `backend/requirements.txt` and install.
- [ ] Create `backend/app/routes/auth_cognito.py` with router, cookie helpers, JWKS, `require_user`.
- [ ] In `backend/app/main.py`, `include_router(auth_cognito.router)`.
- [ ] Add `Depends(require_user)` to all session, chat, plan, and result endpoints (and optionally documents).
- [ ] Set all required env vars in ECS/Secrets Manager.
- [ ] In `site/src/lib/api.ts` (and `frontend` if used): add `credentials: "include"` to fetch; add `getLoginUrl`, `getAuthMe`, `logout` and user type.
- [ ] In UI: header/layout with Sign in / Log out and one-time `getAuthMe()` on load.
- [ ] (Phase 1.5) Migration: add `user_sub TEXT NOT NULL` to `research_sessions`; enforce `WHERE user_sub = :sub` on session get/list/delete and set `user_sub` on create; add `assert_session_owned(session_id, sub)` and use it in chat/plans/results before any work; confirm streaming/SSE endpoints use `credentials: "include"` and support CORS with credentials.

After this, auth is implemented: login via Cognito, session in cookie, `/auth/me` for current user, logout clearing cookie, API routes at `/api/*` protected by `require_user`, and session ownership enforced so users only see their own sessions. Optional: users table and GET `/auth/logout` later.
