'use client';

import { useEffect } from 'react';

/**
 * Cognito now redirects to the backend callback (api.../auth/oauth/cognito/callback).
 * This page is a fallback for old links; redirect to app home.
 */
export default function AuthCallbackPage() {
  useEffect(() => {
    window.location.href = '/';
  }, []);

  return (
    <div style={{ padding: '2rem', textAlign: 'center', color: '#6c757d' }}>
      Redirecting&hellip;
    </div>
  );
}
