# Cognito OAuth Callback URLs

The OAuth flow uses the **backend callback** (not the frontend):

1. User clicks login → frontend navigates to `api.../auth/oauth/cognito/login?redirect_uri=api.../callback`
2. Backend redirects to Cognito Hosted UI with that `redirect_uri`
3. After login, Cognito redirects to `api.../auth/oauth/cognito/callback?code=...&state=...`
4. Backend exchanges code, sets session cookie, 302 to `https://fridayarchive.org/`

## AWS Console: User Pool → App integration → App client → Hosted UI

### Allowed callback URLs

- **Production:** `https://api.fridayarchive.org/auth/oauth/cognito/callback`
- **Local dev:** `http://localhost:8000/auth/oauth/cognito/callback`

### Allowed sign-out URLs

- **Production:** `https://fridayarchive.org/`
- **Local dev:** `http://localhost:3000/`
