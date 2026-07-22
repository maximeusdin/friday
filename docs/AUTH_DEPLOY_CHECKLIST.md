# Auth Deploy Checklist — Prevent Login Outages

Use this checklist when deploying or changing auth-related config. Auth failures usually stem from **redirect_uri mismatch** or **secret misconfig**.

## Before Deploy

- [ ] **COGNITO_REDIRECT_URI** in ECS task def = `https://api.fridayarchive.org/auth/oauth/cognito/callback` (no trailing slash)
- [ ] **Cognito app client** has that exact URL in Allowed callback URLs (User Pool → App integration → App client)
- [ ] **Secrets Manager** — `cognito-client` and `app-session-secret` have rotation **disabled** (`describe-secret` shows no `RotationEnabled`)

## After Deploy

- [ ] **Health check**: `curl https://api.fridayarchive.org/health` → `auth.redirect_uri_ok: true`, `auth.configured: true`
- [ ] **Test login** once manually

## If Auth Breaks Again

1. **Check CloudWatch logs** for `TOKEN_EXCHANGE using redirect_uri=...` — confirm it matches Cognito.
2. **Check `/health`** — if `redirect_uri_ok: false`, fix `COGNITO_REDIRECT_URI` in deploy script / task def.
3. **Secrets** — run `aws secretsmanager describe-secret --secret-id cognito-client`; if `LastRotatedDate` exists, rotation may have changed the value. Restore the value from Cognito app client.

## Do Not

- Enable rotation on `cognito-client` or `app-session-secret` (Cognito doesn't support auto-rotation)
- Change `COGNITO_REDIRECT_URI` without updating Cognito's allowed callback URLs
- Rely on frontend `redirect_uri` query param in production (backend ignores it and uses env)
