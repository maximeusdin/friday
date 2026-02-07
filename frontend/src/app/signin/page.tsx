'use client';

import { useEffect } from 'react';
import { getLoginUrl } from '@/lib/api';

/**
 * Legacy /signin route – redirects to Cognito Hosted UI.
 * Kept so any bookmarks or old links still work.
 */
export default function SignInRedirect() {
  useEffect(() => {
    window.location.href = getLoginUrl();
  }, []);

  return (
    <div style={{ padding: '2rem', textAlign: 'center', color: '#6c757d' }}>
      Redirecting to sign in&hellip;
    </div>
  );
}
